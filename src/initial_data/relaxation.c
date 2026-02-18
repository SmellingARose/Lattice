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
#include "../numerics/finite_diff.h"
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
