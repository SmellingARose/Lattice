/*
 * Lattice — 3D Numerical Relativity
 * Hyperbolic relaxation solver for the Hamiltonian constraint.
 *
 * Solves nabla^2 u + (1/8) A^2 (psi_BL + u)^{-7} = 0 via damped wave:
 *   d_tau u = v
 *   d_tau v = -eta * v + c^2 * [nabla^2(u) + source(u)]
 *
 * where source(u) = (1/8) * A^2 * (psi_BL + u)^{-7}.
 *
 * Standalone mini-RK4 on 2 scalar fields (u, v). Does NOT reuse the
 * 25-field rk4_step() — wrapping 2 fields into a fake 25-field grid
 * would be fragile and wasteful.
 *
 * Memory: 10 arrays of npoints doubles + psi_BL + A2 = 12 total.
 *
 * Ref: Ruter et al. arXiv:1708.07358 (hyperbolic relaxation method)
 * Ref: NRPyElliptic arXiv:2111.02424 (practical implementation)
 */

#include "relaxation.h"
#include "bowen_york.h"
#include "kerr_quasi_isotropic.h"
#include "../numerics/finite_diff.h"
#include "../geometry/tensor_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Workspace for the relaxation solver */
typedef struct {
    double *u;          /* correction to psi_BL                     */
    double *v;          /* pseudo-time derivative of u              */
    double *rhs_u;      /* RHS for u equation                       */
    double *rhs_v;      /* RHS for v equation                       */
    double *scratch_u;  /* RK4 scratch for u                        */
    double *scratch_v;  /* RK4 scratch for v                        */
    double *accum_u;    /* RK4 accumulator for u                    */
    double *accum_v;    /* RK4 accumulator for v                    */
    double *psi_BL;     /* Brill-Lindquist conformal factor         */
    double *A2;         /* A_ij A^ij (precomputed, fixed)           */
    size_t npoints;
} relax_workspace_t;

static relax_workspace_t *relax_alloc(size_t npoints)
{
    relax_workspace_t *w = calloc(1, sizeof(relax_workspace_t));
    w->npoints = npoints;
    w->u         = calloc(npoints, sizeof(double));
    w->v         = calloc(npoints, sizeof(double));
    w->rhs_u     = calloc(npoints, sizeof(double));
    w->rhs_v     = calloc(npoints, sizeof(double));
    w->scratch_u = calloc(npoints, sizeof(double));
    w->scratch_v = calloc(npoints, sizeof(double));
    w->accum_u   = calloc(npoints, sizeof(double));
    w->accum_v   = calloc(npoints, sizeof(double));
    w->psi_BL    = calloc(npoints, sizeof(double));
    w->A2        = calloc(npoints, sizeof(double));
    return w;
}

static void relax_free(relax_workspace_t *w)
{
    free(w->u);
    free(w->v);
    free(w->rhs_u);
    free(w->rhs_v);
    free(w->scratch_u);
    free(w->scratch_v);
    free(w->accum_u);
    free(w->accum_v);
    free(w->psi_BL);
    free(w->A2);
    free(w);
}

/*
 * Robin boundary conditions: 1/r falloff extrapolation for u, v=0.
 * u -> 0 at infinity (correction vanishes far from punctures).
 * Applied in the ghost zones (width = GHOST_WIDTH).
 *
 * For a function f ~ f0/r at large r, the ghost value at distance r_g
 * from center is extrapolated from the last interior point at r_i via:
 *   f(r_g) = f(r_i) * r_i / r_g
 * We use a simpler approach: copy the last interior row and damp by the
 * 1/r ratio.  This is standard for elliptic solvers on finite grids.
 *
 * Actually, for the relaxation we use a simpler zero-Dirichlet approach:
 * set ghost zones to 0 for u (u=0 at boundary), v=0 at boundary.
 * This is sufficient since u ~ 1/r already and our domain is large enough.
 */
static void relax_apply_bc(double *u, double *v, const grid_t *g)
{
    int N = g->Ntotal;
    int gw = g->ghost;

    /* Zero-Dirichlet: set ghost zones to 0 */
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                if (i < gw || i >= N - gw ||
                    j < gw || j >= N - gw ||
                    k < gw || k >= N - gw) {
                    int idx = IDX(g, i, j, k);
                    u[idx] = 0.0;
                    v[idx] = 0.0;
                }
            }
        }
    }
}

/*
 * Compute RHS for the relaxation system at all interior points.
 *
 * d_tau u = v                                              (rhs_u)
 * d_tau v = -eta * v + c^2 * [Lap(u) + source]            (rhs_v)
 *
 * where source = (1/8) * A^2 * (psi_BL + u)^{-7}.
 *
 * Laplacian via existing 4th-order fd_d2() stencils.
 * Optional KO dissipation on u for stability (sigma_relax * fd_ko).
 */
static void relax_rhs(const relax_workspace_t *w, const grid_t *g,
                       double eta, double c2, double sigma_ko)
{
    int gw = g->ghost;
    int N = g->Ntotal;
    double dx = g->dx;
    int sx = STRIDE_X;
    int sy = STRIDE_Y(g);
    int sz = STRIDE_Z(g);

    for (int k = gw; k < N - gw; k++) {
        for (int j = gw; j < N - gw; j++) {
            for (int i = gw; i < N - gw; i++) {
                int idx = IDX(g, i, j, k);

                /* Laplacian of u via 4th-order FD */
                double lap_u = fd_d2(w->u, idx, sx, dx)
                             + fd_d2(w->u, idx, sy, dx)
                             + fd_d2(w->u, idx, sz, dx);

                /* Nonlinear source: (1/8) * A^2 * (psi_BL + u)^{-7} */
                double psi_total = w->psi_BL[idx] + w->u[idx];
                if (psi_total < 0.1) psi_total = 0.1;  /* floor for stability */
                double psi7_inv = 1.0 / (psi_total * psi_total * psi_total *
                                          psi_total * psi_total * psi_total *
                                          psi_total);
                double source = 0.125 * w->A2[idx] * psi7_inv;

                /* RHS for u: d_tau u = v */
                w->rhs_u[idx] = w->v[idx];

                /* RHS for v: d_tau v = -eta*v + c^2 * (Lap(u) + source) */
                w->rhs_v[idx] = -eta * w->v[idx] + c2 * (lap_u + source);

                /* Optional KO dissipation on u for numerical stability */
                if (sigma_ko > 0.0) {
                    double ko_u = fd_ko(w->u, idx, sx, dx)
                                + fd_ko(w->u, idx, sy, dx)
                                + fd_ko(w->u, idx, sz, dx);
                    w->rhs_u[idx] += sigma_ko * ko_u;
                }
            }
        }
    }
}

/*
 * L2 norm of v over the interior (convergence monitor).
 * ||v||_L2 = sqrt( sum(v^2) / N_interior )
 */
static double relax_v_l2(const double *v, const grid_t *g)
{
    int gw = g->ghost;
    int N  = g->Ntotal;
    double sum = 0.0;
    int count = 0;

    for (int k = gw; k < N - gw; k++) {
        for (int j = gw; j < N - gw; j++) {
            for (int i = gw; i < N - gw; i++) {
                int idx = IDX(g, i, j, k);
                sum += v[idx] * v[idx];
                count++;
            }
        }
    }

    return (count > 0) ? sqrt(sum / count) : 0.0;
}

/*
 * Classic 4-stage RK4 on the (u, v) system.
 * k1 = f(y_n)
 * k2 = f(y_n + dt/2 * k1)
 * k3 = f(y_n + dt/2 * k2)
 * k4 = f(y_n + dt * k3)
 * y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
 */
static void relax_rk4_step(relax_workspace_t *w, const grid_t *g,
                            double eta, double c2, double sigma_ko,
                            double dt)
{
    size_t np = w->npoints;
    int gw = g->ghost;
    int N  = g->Ntotal;

    /* Save initial state in scratch */
    memcpy(w->scratch_u, w->u, np * sizeof(double));
    memcpy(w->scratch_v, w->v, np * sizeof(double));

    /* Zero accumulators */
    memset(w->accum_u, 0, np * sizeof(double));
    memset(w->accum_v, 0, np * sizeof(double));

    double rk_weights[4] = { 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 };
    double rk_advance[4] = { 0.5, 0.5, 1.0, 0.0 };

    for (int stage = 0; stage < 4; stage++) {
        /* Compute RHS at current (u, v) */
        relax_rhs(w, g, eta, c2, sigma_ko);

        /* Accumulate: accum += weight * rhs */
        for (int k = gw; k < N - gw; k++) {
            for (int j = gw; j < N - gw; j++) {
                for (int i = gw; i < N - gw; i++) {
                    int idx = IDX(g, i, j, k);
                    w->accum_u[idx] += rk_weights[stage] * w->rhs_u[idx];
                    w->accum_v[idx] += rk_weights[stage] * w->rhs_v[idx];
                }
            }
        }

        /* Advance to next substage (except after stage 3) */
        if (stage < 3) {
            double a = rk_advance[stage] * dt;
            for (int k = gw; k < N - gw; k++) {
                for (int j = gw; j < N - gw; j++) {
                    for (int i = gw; i < N - gw; i++) {
                        int idx = IDX(g, i, j, k);
                        w->u[idx] = w->scratch_u[idx] + a * w->rhs_u[idx];
                        w->v[idx] = w->scratch_v[idx] + a * w->rhs_v[idx];
                    }
                }
            }
            /* Apply BCs after each substage for stencil safety */
            relax_apply_bc(w->u, w->v, g);
        }
    }

    /* Final update: y_{n+1} = y_n + dt * accum */
    for (int k = gw; k < N - gw; k++) {
        for (int j = gw; j < N - gw; j++) {
            for (int i = gw; i < N - gw; i++) {
                int idx = IDX(g, i, j, k);
                w->u[idx] = w->scratch_u[idx] + dt * w->accum_u[idx];
                w->v[idx] = w->scratch_v[idx] + dt * w->accum_v[idx];
            }
        }
    }
    relax_apply_bc(w->u, w->v, g);
}

double relaxation_solve(grid_t *g, int n_bh, const puncture_data_t *bhs,
                        double tol, int max_iter, int verbose)
{
    relax_workspace_t *w = relax_alloc(g->npoints);

    /* Precompute psi_BL and A^2 at every grid point */
    for (int k = 0; k < g->Ntotal; k++) {
        for (int j = 0; j < g->Ntotal; j++) {
            for (int i = 0; i < g->Ntotal; i++) {
                int idx = IDX(g, i, j, k);
                double x = COORD(g, i);
                double y = COORD(g, j);
                double z = COORD(g, k);

                w->psi_BL[idx] = brill_lindquist_psi(x, y, z, n_bh, bhs);

                double A_phys[3][3];
                bowen_york_Aij(A_phys, x, y, z, n_bh, bhs);
                w->A2[idx] = bowen_york_A2(A_phys);
            }
        }
    }

    /* Initialize u=0, v=0 (start from BL guess) */
    memset(w->u, 0, g->npoints * sizeof(double));
    memset(w->v, 0, g->npoints * sizeof(double));

    /* Solver parameters.
     * c = wave speed, eta = damping, CFL_relax determines dt_relax.
     * Ref: arXiv:2111.02424 Section 3.2 */
    double c = 1.0;
    double c2 = c * c;
    double CFL_relax = 0.5;
    double dt_relax = CFL_relax * g->dx;
    double eta = 6.0 / g->L;
    double sigma_ko = 0.1;  /* mild KO dissipation for stability */

    double v_l2 = relax_v_l2(w->v, g);
    int iter;

    if (verbose) {
        printf("    Relaxation: dx=%.4f, dt=%.4f, eta=%.4f, tol=%.2e\n",
               g->dx, dt_relax, eta, tol);
    }

    double prev_v_l2 = v_l2;
    for (iter = 0; iter < max_iter; iter++) {
        relax_rk4_step(w, g, eta, c2, sigma_ko, dt_relax);

        if ((iter + 1) % 500 == 0 || iter == 0) {
            v_l2 = relax_v_l2(w->v, g);
            if (verbose && ((iter + 1) % 2000 == 0 || iter == 0)) {
                printf("    iter %5d: ||v||_L2 = %.6e\n", iter + 1, v_l2);
            }
            if (v_l2 < tol) {
                if (verbose)
                    printf("    Converged at iter %d: ||v||_L2 = %.6e\n",
                           iter + 1, v_l2);
                break;
            }
            /* Early termination if stagnated */
            if ((iter + 1) % 1000 == 0) {
                if (v_l2 > 0.95 * prev_v_l2) {
                    if (verbose)
                        printf("    Stagnated at iter %d: ||v||_L2 = %.6e\n",
                               iter + 1, v_l2);
                    break;
                }
                prev_v_l2 = v_l2;
            }
        }
    }

    /* Check final residual */
    v_l2 = relax_v_l2(w->v, g);

    /* Build full psi = psi_BL + u and set CCZ4 fields */
    double *psi_full = calloc(g->npoints, sizeof(double));
    for (size_t idx = 0; idx < g->npoints; idx++)
        psi_full[idx] = w->psi_BL[idx] + w->u[idx];

    set_ccz4_from_psi(g, psi_full, n_bh, bhs);

    free(psi_full);
    relax_free(w);
    return v_l2;
}

/* ================================================================
 * Coupled 4-field relaxation solver for HiSpID data.
 *
 * Solves the coupled Hamiltonian + 3 momentum constraints:
 *   d_tau psi   = v_psi
 *   d_tau v_psi = -eta*v_psi + c^2*[Lap(psi) + S_H]
 *   d_tau V^i   = v_V^i
 *   d_tau v_V^i = -eta*v_V^i + c^2*[Lap(V^i) + (1/3)*d_i(div V) + S_M^i]
 *
 * where S_H = (R_tilde/8)*psi + (A^2/8)*psi^{-7}  (Hamiltonian source)
 * and S_M^i is the precomputed momentum constraint violation.
 *
 * Extends the 1-field solver (relax_workspace_t) to 4 field pairs.
 * Same mini-RK4 pattern, just more arrays.
 *
 * Ref: arXiv:1410.8607 (HiSpID), arXiv:1708.07358 (relaxation)
 * ================================================================ */

typedef struct {
    /* Evolved fields: 4 pairs of (u, v) */
    double *psi;        double *v_psi;
    double *V[3];       double *v_V[3];
    /* RHS */
    double *rhs_psi;    double *rhs_v_psi;
    double *rhs_V[3];   double *rhs_v_V[3];
    /* RK4 scratch (save initial state) */
    double *scratch_psi; double *scratch_v_psi;
    double *scratch_V[3]; double *scratch_v_V[3];
    /* RK4 accumulators */
    double *accum_psi;  double *accum_v_psi;
    double *accum_V[3]; double *accum_v_V[3];
    /* Background data (precomputed, fixed) */
    double *psi_BL;     /* Brill-Lindquist conformal factor */
    double *A2;         /* A_ij A^ij (from superposed Kerr + BY) */
    double *R_tilde;    /* conformal Ricci scalar from h_bg */
    double *S_M[3];     /* momentum constraint violation source */
    size_t npoints;
} coupled_workspace_t;

static coupled_workspace_t *coupled_alloc(size_t np)
{
    coupled_workspace_t *w = calloc(1, sizeof(coupled_workspace_t));
    w->npoints = np;

    w->psi         = calloc(np, sizeof(double));
    w->v_psi       = calloc(np, sizeof(double));
    w->rhs_psi     = calloc(np, sizeof(double));
    w->rhs_v_psi   = calloc(np, sizeof(double));
    w->scratch_psi = calloc(np, sizeof(double));
    w->scratch_v_psi = calloc(np, sizeof(double));
    w->accum_psi   = calloc(np, sizeof(double));
    w->accum_v_psi = calloc(np, sizeof(double));

    for (int d = 0; d < 3; d++) {
        w->V[d]         = calloc(np, sizeof(double));
        w->v_V[d]       = calloc(np, sizeof(double));
        w->rhs_V[d]     = calloc(np, sizeof(double));
        w->rhs_v_V[d]   = calloc(np, sizeof(double));
        w->scratch_V[d] = calloc(np, sizeof(double));
        w->scratch_v_V[d] = calloc(np, sizeof(double));
        w->accum_V[d]   = calloc(np, sizeof(double));
        w->accum_v_V[d] = calloc(np, sizeof(double));
    }

    w->psi_BL  = calloc(np, sizeof(double));
    w->A2      = calloc(np, sizeof(double));
    w->R_tilde = calloc(np, sizeof(double));
    for (int d = 0; d < 3; d++)
        w->S_M[d] = calloc(np, sizeof(double));

    return w;
}

static void coupled_free(coupled_workspace_t *w)
{
    free(w->psi);        free(w->v_psi);
    free(w->rhs_psi);    free(w->rhs_v_psi);
    free(w->scratch_psi); free(w->scratch_v_psi);
    free(w->accum_psi);  free(w->accum_v_psi);

    for (int d = 0; d < 3; d++) {
        free(w->V[d]);         free(w->v_V[d]);
        free(w->rhs_V[d]);    free(w->rhs_v_V[d]);
        free(w->scratch_V[d]); free(w->scratch_v_V[d]);
        free(w->accum_V[d]);  free(w->accum_v_V[d]);
    }

    free(w->psi_BL);
    free(w->A2);
    free(w->R_tilde);
    for (int d = 0; d < 3; d++)
        free(w->S_M[d]);

    free(w);
}

/* Zero-Dirichlet BCs for all 4 field pairs */
static void coupled_apply_bc(coupled_workspace_t *w, const grid_t *g)
{
    int N = g->Ntotal;
    int gw = g->ghost;

    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) {
                if (i < gw || i >= N - gw ||
                    j < gw || j >= N - gw ||
                    k < gw || k >= N - gw) {
                    int idx = IDX(g, i, j, k);
                    w->psi[idx] = 0.0;
                    w->v_psi[idx] = 0.0;
                    for (int d = 0; d < 3; d++) {
                        w->V[d][idx] = 0.0;
                        w->v_V[d][idx] = 0.0;
                    }
                }
            }
}

/*
 * Coupled RHS: Hamiltonian + 3 momentum constraints.
 *
 * Hamiltonian:
 *   rhs_psi   = v_psi
 *   rhs_v_psi = -eta*v_psi + c^2*[Lap(psi) + (R_tilde/8)*psi_total
 *                                  + (A^2/8)*psi_total^{-7}]
 *
 * Momentum (flat vector Laplacian):
 *   rhs_V^i   = v_V^i
 *   rhs_v_V^i = -eta*v_V^i + c^2*[Lap(V^i) + (1/3)*d_i(div V) + S_M^i]
 *
 * Ref: arXiv:1708.07358, arXiv:1410.8607
 */
static void coupled_rhs(coupled_workspace_t *w, const grid_t *g,
                        double eta, double c2, double sigma_ko)
{
    int gw = g->ghost;
    int N  = g->Ntotal;
    double dx = g->dx;
    int sx = STRIDE_X;
    int sy = STRIDE_Y(g);
    int sz = STRIDE_Z(g);
    int strides[3] = { sx, sy, sz };

    for (int k = gw; k < N - gw; k++) {
        for (int j = gw; j < N - gw; j++) {
            for (int i = gw; i < N - gw; i++) {
                int idx = IDX(g, i, j, k);

                /* --- Hamiltonian constraint --- */
                double lap_psi = fd_d2(w->psi, idx, sx, dx)
                               + fd_d2(w->psi, idx, sy, dx)
                               + fd_d2(w->psi, idx, sz, dx);

                double psi_total = w->psi_BL[idx] + w->psi[idx];
                if (psi_total < 0.1) psi_total = 0.1;

                double psi7_inv = 1.0;
                {
                    double p2 = psi_total * psi_total;
                    double p4 = p2 * p2;
                    psi7_inv = 1.0 / (p4 * p2 * psi_total);
                }

                /* S_H = (R_tilde/8)*psi_total + (A^2/8)*psi_total^{-7} */
                double source_H = w->R_tilde[idx] * 0.125 * psi_total
                                + w->A2[idx] * 0.125 * psi7_inv;

                w->rhs_psi[idx]   = w->v_psi[idx];
                w->rhs_v_psi[idx] = -eta * w->v_psi[idx]
                                   + c2 * (lap_psi + source_H);

                /* KO dissipation on psi */
                if (sigma_ko > 0.0) {
                    double ko = fd_ko(w->psi, idx, sx, dx)
                              + fd_ko(w->psi, idx, sy, dx)
                              + fd_ko(w->psi, idx, sz, dx);
                    w->rhs_psi[idx] += sigma_ko * ko;
                }

                /* --- Momentum constraints (vector Laplacian) --- */
                for (int d = 0; d < 3; d++) {
                    double lap_V = fd_d2(w->V[d], idx, sx, dx)
                                 + fd_d2(w->V[d], idx, sy, dx)
                                 + fd_d2(w->V[d], idx, sz, dx);

                    /* (1/3) * d_d(div V) using second derivatives.
                     * d_d(div V) = sum_e d^2 V^e / (dx_d dx_e) */
                    double d_divV = 0.0;
                    for (int e = 0; e < 3; e++) {
                        if (e == d)
                            d_divV += fd_d2(w->V[e], idx, strides[e], dx);
                        else
                            d_divV += fd_d2_mixed(w->V[e], idx,
                                                  strides[d], strides[e], dx);
                    }

                    w->rhs_V[d][idx]   = w->v_V[d][idx];
                    w->rhs_v_V[d][idx] = -eta * w->v_V[d][idx]
                                        + c2 * (lap_V + d_divV / 3.0
                                                + w->S_M[d][idx]);

                    /* KO dissipation on V^d */
                    if (sigma_ko > 0.0) {
                        double ko = fd_ko(w->V[d], idx, sx, dx)
                                  + fd_ko(w->V[d], idx, sy, dx)
                                  + fd_ko(w->V[d], idx, sz, dx);
                        w->rhs_V[d][idx] += sigma_ko * ko;
                    }
                }
            }
        }
    }
}

/* L2 residual: max of ||v_psi||_L2 and ||v_V^i||_L2 */
static double coupled_residual(const coupled_workspace_t *w, const grid_t *g)
{
    int gw = g->ghost;
    int N  = g->Ntotal;
    double sum_psi = 0.0;
    double sum_V = 0.0;
    int count = 0;

    for (int k = gw; k < N - gw; k++)
        for (int j = gw; j < N - gw; j++)
            for (int i = gw; i < N - gw; i++) {
                int idx = IDX(g, i, j, k);
                sum_psi += w->v_psi[idx] * w->v_psi[idx];
                for (int d = 0; d < 3; d++)
                    sum_V += w->v_V[d][idx] * w->v_V[d][idx];
                count++;
            }

    double l2_psi = (count > 0) ? sqrt(sum_psi / count) : 0.0;
    double l2_V   = (count > 0) ? sqrt(sum_V / (3 * count)) : 0.0;
    return (l2_psi > l2_V) ? l2_psi : l2_V;
}

/*
 * Classic 4-stage RK4 on 8 fields (4 pairs of u,v).
 * Same structure as relax_rk4_step, extended to all fields.
 */
static void coupled_rk4_step(coupled_workspace_t *w, const grid_t *g,
                              double eta, double c2, double sigma_ko,
                              double dt)
{
    size_t np = w->npoints;
    int gw = g->ghost;
    int N  = g->Ntotal;

    /* Save initial state */
    memcpy(w->scratch_psi,   w->psi,   np * sizeof(double));
    memcpy(w->scratch_v_psi, w->v_psi, np * sizeof(double));
    for (int d = 0; d < 3; d++) {
        memcpy(w->scratch_V[d],   w->V[d],   np * sizeof(double));
        memcpy(w->scratch_v_V[d], w->v_V[d], np * sizeof(double));
    }

    /* Zero accumulators */
    memset(w->accum_psi,   0, np * sizeof(double));
    memset(w->accum_v_psi, 0, np * sizeof(double));
    for (int d = 0; d < 3; d++) {
        memset(w->accum_V[d],   0, np * sizeof(double));
        memset(w->accum_v_V[d], 0, np * sizeof(double));
    }

    double rk_w[4] = { 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 };
    double rk_a[4] = { 0.5, 0.5, 1.0, 0.0 };

    for (int stage = 0; stage < 4; stage++) {
        coupled_rhs(w, g, eta, c2, sigma_ko);

        /* Accumulate */
        for (int k = gw; k < N - gw; k++)
            for (int j = gw; j < N - gw; j++)
                for (int i = gw; i < N - gw; i++) {
                    int idx = IDX(g, i, j, k);
                    w->accum_psi[idx]   += rk_w[stage] * w->rhs_psi[idx];
                    w->accum_v_psi[idx] += rk_w[stage] * w->rhs_v_psi[idx];
                    for (int d = 0; d < 3; d++) {
                        w->accum_V[d][idx]   += rk_w[stage] * w->rhs_V[d][idx];
                        w->accum_v_V[d][idx] += rk_w[stage] * w->rhs_v_V[d][idx];
                    }
                }

        /* Advance to next substage */
        if (stage < 3) {
            double a = rk_a[stage] * dt;
            for (int k = gw; k < N - gw; k++)
                for (int j = gw; j < N - gw; j++)
                    for (int i = gw; i < N - gw; i++) {
                        int idx = IDX(g, i, j, k);
                        w->psi[idx]   = w->scratch_psi[idx]   + a * w->rhs_psi[idx];
                        w->v_psi[idx] = w->scratch_v_psi[idx] + a * w->rhs_v_psi[idx];
                        for (int d = 0; d < 3; d++) {
                            w->V[d][idx]   = w->scratch_V[d][idx]   + a * w->rhs_V[d][idx];
                            w->v_V[d][idx] = w->scratch_v_V[d][idx] + a * w->rhs_v_V[d][idx];
                        }
                    }
            coupled_apply_bc(w, g);
        }
    }

    /* Final update */
    for (int k = gw; k < N - gw; k++)
        for (int j = gw; j < N - gw; j++)
            for (int i = gw; i < N - gw; i++) {
                int idx = IDX(g, i, j, k);
                w->psi[idx]   = w->scratch_psi[idx]   + dt * w->accum_psi[idx];
                w->v_psi[idx] = w->scratch_v_psi[idx] + dt * w->accum_v_psi[idx];
                for (int d = 0; d < 3; d++) {
                    w->V[d][idx]   = w->scratch_V[d][idx]   + dt * w->accum_V[d][idx];
                    w->v_V[d][idx] = w->scratch_v_V[d][idx] + dt * w->accum_v_V[d][idx];
                }
            }
    coupled_apply_bc(w, g);
}

/*
 * Precompute background data: psi_BL, A^2, R_tilde, S_M^i.
 *
 * R_tilde is computed numerically from the conformal metric h_bg via
 * finite differences of h_bg.  S_M^i is the divergence of the background
 * A_ij (momentum constraint violation of the superposed Kerr data).
 */
static void coupled_precompute(coupled_workspace_t *w, const grid_t *g,
                                int n_bh, const puncture_data_t *bhs)
{
    int N  = g->Ntotal;
    int gw = g->ghost;
    double dx = g->dx;
    size_t np = g->npoints;

    /* Temporary arrays for h_bg (6 components) and A_bg (6 components) */
    double *h_bg[6], *A_bg[6];
    for (int c = 0; c < 6; c++) {
        h_bg[c] = calloc(np, sizeof(double));
        A_bg[c] = calloc(np, sizeof(double));
    }

    /* Fill background arrays at every grid point */
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) {
                int idx = IDX(g, i, j, k);
                double x = COORD(g, i);
                double y = COORD(g, j);
                double z = COORD(g, k);

                /* Brill-Lindquist conformal factor */
                w->psi_BL[idx] = brill_lindquist_psi(x, y, z, n_bh, bhs);

                /* Superposed Kerr conformal metric */
                double h[3][3];
                hispid_conformal_metric(h, x, y, z, n_bh, bhs);
                h_bg[0][idx] = h[0][0]; h_bg[1][idx] = h[0][1];
                h_bg[2][idx] = h[0][2]; h_bg[3][idx] = h[1][1];
                h_bg[4][idx] = h[1][2]; h_bg[5][idx] = h[2][2];

                /* Superposed Kerr extrinsic curvature + BY momentum */
                double A_kerr[3][3];
                hispid_extrinsic(A_kerr, x, y, z, n_bh, bhs);

                /* Add Bowen-York A_ij for linear momentum contribution */
                double A_by[3][3];
                bowen_york_Aij(A_by, x, y, z, n_bh, bhs);

                /* Combined: A_total = A_kerr + A_by (BY handles far-field) */
                double A_total[3][3];
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        A_total[a][b] = A_kerr[a][b] + A_by[a][b];

                A_bg[0][idx] = A_total[0][0]; A_bg[1][idx] = A_total[0][1];
                A_bg[2][idx] = A_total[0][2]; A_bg[3][idx] = A_total[1][1];
                A_bg[4][idx] = A_total[1][2]; A_bg[5][idx] = A_total[2][2];

                /* A^2 = A_ij A^ij.  For now use flat contraction (h close to delta).
                 * A^2 = sum(A_ij^2) */
                w->A2[idx] = bowen_york_A2(A_total);
            }

    /* Compute R_tilde numerically from h_bg via FD.
     * R_tilde = h^{ij} R_ij where R_ij is computed from Christoffel symbols.
     * Only at interior points (ghost zones stay 0). */
    int sx = STRIDE_X;
    int sy = STRIDE_Y(g);
    int sz = STRIDE_Z(g);
    int strides[3] = { sx, sy, sz };

    /* Field index mapping for symmetric tensor: [i][j] -> component 0..5 */
    static const int sym_map[3][3] = {{0,1,2},{1,3,4},{2,4,5}};

    for (int k = gw; k < N - gw; k++)
        for (int j_idx = gw; j_idx < N - gw; j_idx++)
            for (int i = gw; i < N - gw; i++) {
                int idx = IDX(g, i, j_idx, k);

                /* Load h_ij */
                double h[3][3];
                h[0][0] = h_bg[0][idx]; h[0][1] = h_bg[1][idx]; h[0][2] = h_bg[2][idx];
                h[1][0] = h[0][1];      h[1][1] = h_bg[3][idx]; h[1][2] = h_bg[4][idx];
                h[2][0] = h[0][2];      h[2][1] = h[1][2];      h[2][2] = h_bg[5][idx];

                double h_UU[3][3];
                compute_inverse_sym(h, h_UU);

                /* First derivatives of h */
                double d1_h[3][3][3];
                for (int dir = 0; dir < 3; dir++)
                    for (int a = 0; a < 3; a++)
                        for (int b = a; b < 3; b++) {
                            double val = fd_d1(h_bg[sym_map[a][b]], idx,
                                               strides[dir], dx);
                            d1_h[a][b][dir] = val;
                            d1_h[b][a][dir] = val;
                        }

                /* Second derivatives of h (diagonal) */
                double d2_h[3][3][3][3];
                memset(d2_h, 0, sizeof(d2_h));
                for (int dir = 0; dir < 3; dir++)
                    for (int a = 0; a < 3; a++)
                        for (int b = a; b < 3; b++) {
                            double val = fd_d2(h_bg[sym_map[a][b]], idx,
                                               strides[dir], dx);
                            d2_h[a][b][dir][dir] = val;
                            d2_h[b][a][dir][dir] = val;
                        }
                /* Mixed second derivatives */
                for (int d1 = 0; d1 < 3; d1++)
                    for (int d2 = 0; d2 < d1; d2++)
                        for (int a = 0; a < 3; a++)
                            for (int b = a; b < 3; b++) {
                                double val = fd_d2_mixed(
                                    h_bg[sym_map[a][b]], idx,
                                    strides[d1], strides[d2], dx);
                                d2_h[a][b][d1][d2] = val;
                                d2_h[a][b][d2][d1] = val;
                                d2_h[b][a][d1][d2] = val;
                                d2_h[b][a][d2][d1] = val;
                            }

                /* Christoffel symbols */
                chris_t chris;
                compute_christoffel(d1_h, h_UU, &chris);

                /* Conformal Ricci scalar.
                 * R_ij = -0.5 h^{kl} d2_h_{ij,kl}
                 *       + 0.5 (h_{ki} d1_Gamma^k_j + h_{kj} d1_Gamma^k_i)
                 *       + 0.5 Gamma^k d1_h_{ij,k}
                 *       + Gamma-squared terms
                 * We use the simplified formula via Christoffel symbols. */
                double R_scalar = 0.0;
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++) {
                        double R_ab = 0.0;
                        for (int kk = 0; kk < 3; kk++) {
                            R_ab += 0.5 * chris.contracted[kk] * d1_h[a][b][kk];
                            for (int ll = 0; ll < 3; ll++) {
                                R_ab += -0.5 * h_UU[kk][ll] * d2_h[a][b][kk][ll];
                                /* Gamma-squared terms */
                                double chris_LLU_jkl = 0.0;
                                for (int mm = 0; mm < 3; mm++)
                                    chris_LLU_jkl += h_UU[kk][mm] * chris.LLL[b][ll][mm];
                                R_ab += chris.ULL[kk][ll][a] * chris_LLU_jkl;

                                double chris_LLU_ikl = 0.0;
                                for (int mm = 0; mm < 3; mm++)
                                    chris_LLU_ikl += h_UU[kk][mm] * chris.LLL[a][ll][mm];
                                R_ab += chris.ULL[kk][ll][b] * chris_LLU_ikl;

                                double chris_LLU_kkl = 0.0;
                                for (int mm = 0; mm < 3; mm++)
                                    chris_LLU_kkl += h_UU[kk][mm] * chris.LLL[kk][b][mm];
                                R_ab += chris.ULL[kk][a][ll] * chris_LLU_kkl;
                            }
                        }
                        R_scalar += h_UU[a][b] * R_ab;
                    }

                w->R_tilde[idx] = R_scalar;
            }

    /* Compute S_M^i: momentum constraint violation.
     * S_M^i = -d_j A_bg^{ij} (flat metric approximation: A^ij = A_ij)
     * This is a first-order quantity, computed from background data. */
    for (int k = gw; k < N - gw; k++)
        for (int j_idx = gw; j_idx < N - gw; j_idx++)
            for (int i = gw; i < N - gw; i++) {
                int idx = IDX(g, i, j_idx, k);

                for (int d = 0; d < 3; d++) {
                    double div_A = 0.0;
                    for (int e = 0; e < 3; e++) {
                        /* d_e A_{de} */
                        div_A += fd_d1(A_bg[sym_map[d][e]], idx,
                                       strides[e], dx);
                    }
                    w->S_M[d][idx] = -div_A;
                }
            }

    /* Free temporary arrays */
    for (int c = 0; c < 6; c++) {
        free(h_bg[c]);
        free(A_bg[c]);
    }
}

double relaxation_solve_coupled(grid_t *g, int n_bh, const puncture_data_t *bhs,
                                double tol, int max_iter, int verbose)
{
    coupled_workspace_t *w = coupled_alloc(g->npoints);

    if (verbose)
        printf("    HiSpID: precomputing background (QI Kerr metric)...\n");

    coupled_precompute(w, g, n_bh, bhs);

    /* Initialize: psi=0, v_psi=0, V^i=0, v_V^i=0 */
    memset(w->psi,   0, g->npoints * sizeof(double));
    memset(w->v_psi, 0, g->npoints * sizeof(double));
    for (int d = 0; d < 3; d++) {
        memset(w->V[d],   0, g->npoints * sizeof(double));
        memset(w->v_V[d], 0, g->npoints * sizeof(double));
    }

    /* Solver parameters (same as 1-field solver) */
    double c = 1.0;
    double c2 = c * c;
    double CFL_relax = 0.5;
    double dt_relax = CFL_relax * g->dx;
    double eta = 6.0 / g->L;
    double sigma_ko = 0.1;

    double residual = coupled_residual(w, g);

    if (verbose)
        printf("    HiSpID: dx=%.4f, dt=%.4f, eta=%.4f, tol=%.2e\n",
               g->dx, dt_relax, eta, tol);

    int iter;
    double prev_residual = residual;
    for (iter = 0; iter < max_iter; iter++) {
        coupled_rk4_step(w, g, eta, c2, sigma_ko, dt_relax);

        if ((iter + 1) % 200 == 0 || iter == 0) {
            residual = coupled_residual(w, g);
            if (verbose && ((iter + 1) % 1000 == 0 || iter == 0))
                printf("    iter %5d: residual = %.6e\n", iter + 1, residual);
            if (residual < tol) {
                if (verbose)
                    printf("    Converged at iter %d: residual = %.6e\n",
                           iter + 1, residual);
                break;
            }
            /* Early termination if stagnated (residual not improving) */
            if ((iter + 1) % 1000 == 0) {
                if (residual > 0.95 * prev_residual) {
                    if (verbose)
                        printf("    Stagnated at iter %d: residual = %.6e\n",
                               iter + 1, residual);
                    break;
                }
                prev_residual = residual;
            }
        }
    }

    residual = coupled_residual(w, g);

    /* Build full psi = psi_BL + correction */
    double *psi_full = calloc(g->npoints, sizeof(double));
    for (size_t idx = 0; idx < g->npoints; idx++)
        psi_full[idx] = w->psi_BL[idx] + w->psi[idx];

    /* Set CCZ4 fields with non-flat conformal metric */
    set_ccz4_from_hispid(g, psi_full, w->V, n_bh, bhs);

    free(psi_full);
    coupled_free(w);
    return residual;
}
