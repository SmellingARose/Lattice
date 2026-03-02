/*
 * Lattice — 3D Numerical Relativity
 * FAS Multigrid constraint solver (FMG + Newton-Gauss-Seidel).
 *
 * Solves nabla^2 u + S(u) = 0 using Full Multigrid (FMG) with Full
 * Approximation Scheme (FAS) V-cycles and Newton-Gauss-Seidel smoother.
 *
 * Two modes:
 *   1-field (BY):    Lap(psi) + (1/8)*A^2*psi_total^{-7} = 0
 *   4-field (HiSpID): Hamiltonian + 3 momentum constraints coupled
 *
 * Newton-GS smoother with 8-color ordering (GPU-compatible).
 * FMG achieves discretization accuracy in ~1.14 V-cycles of work.
 *
 * Ref: arXiv:0705.1486 (Natchu & Matzner, 4th-order MG for BH data)
 * Ref: arXiv:2510.11152 (GPU FAS multigrid, 8-color MCGS, 61x speedup)
 * Ref: arXiv:2501.13046 (GRTresna, open-source NR MG constraint solver)
 */

#include "relaxation.h"
#include "bowen_york.h"
#include "kerr_quasi_isotropic.h"
#include "../numerics/finite_diff.h"
#include "../geometry/tensor_utils.h"
#include "../amr/prolongation.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Center weight of fd_d2 for Jacobian diagonal computation.
 * 4th-order: -5/2,  6th-order: -49/18.
 * 3D Laplacian diagonal = 3 * FD_D2_CENTER_WEIGHT / dx^2. */
#if FD_ORDER == 6
#define FD_D2_CENTER_WEIGHT (-49.0 / 18.0)
#else
#define FD_D2_CENTER_WEIGHT (-5.0 / 2.0)
#endif

/* ================================================================
 * Multigrid parameters
 * Ref: arXiv:0705.1486 Section 3.2
 * ================================================================ */
#define MG_N_MIN     16   /* coarsest grid interior points per side */
#define MG_NU_PRE     4   /* pre-smoothing Newton-GS sweeps */
#define MG_NU_POST    4   /* post-smoothing Newton-GS sweeps */
#define MG_NU_COARSE 50   /* coarsest-level sweeps */
#define MG_MAX_LEVELS 8   /* max hierarchy depth */

/* ================================================================
 * Per-level multigrid data
 * ================================================================ */
typedef struct {
    double *psi;         /* conformal factor correction */
    double *V[3];        /* momentum correction (4-field only, else NULL) */

    double *f_psi;       /* FAS target RHS (0 on finest, tau-corrected on coarse) */
    double *f_V[3];

    double *save_psi;    /* saved restricted solution for correction extraction */
    double *save_V[3];

    double *L_psi;       /* operator evaluation scratch L(u) / residual */
    double *L_V[3];

    double *psi_BL;      /* Brill-Lindquist conformal factor (precomputed) */
    double *A2;          /* A_ij A^ij (precomputed) */
    double *R_tilde;     /* conformal Ricci scalar (4-field only) */
    double *S_M[3];      /* momentum source (4-field only) */

    int    N;            /* interior points per side */
    int    ghost;        /* = GHOST_WIDTH = 4 */
    int    Ntotal;       /* N + 2*ghost */
    double dx;
    double L;            /* domain size */
    size_t npoints;      /* Ntotal^3 */
} mg_level_t;

/* Indexing for multigrid level (same formula as IDX but using level Ntotal) */
#define MG_IDX(lev, i, j, k) \
    ((k) * (lev)->Ntotal * (lev)->Ntotal + (j) * (lev)->Ntotal + (i))

/* Coordinate of grid point i on a multigrid level (cell-centered) */
#define MG_COORD(lev, i) \
    (((i) - (lev)->ghost + 0.5) * (lev)->dx - (lev)->L * 0.5)

/* ================================================================
 * Level allocation / deallocation
 * ================================================================ */
static void mg_level_init(mg_level_t *lev, int N, double L, int four_field)
{
    lev->N      = N;
    lev->ghost  = GHOST_WIDTH;
    lev->Ntotal = N + 2 * GHOST_WIDTH;
    lev->dx     = L / N;
    lev->L      = L;
    lev->npoints = (size_t)lev->Ntotal * lev->Ntotal * lev->Ntotal;

    lev->psi      = calloc(lev->npoints, sizeof(double));
    lev->f_psi    = calloc(lev->npoints, sizeof(double));
    lev->save_psi = calloc(lev->npoints, sizeof(double));
    lev->L_psi    = calloc(lev->npoints, sizeof(double));
    lev->psi_BL   = calloc(lev->npoints, sizeof(double));
    lev->A2       = calloc(lev->npoints, sizeof(double));

    if (four_field) {
        lev->R_tilde = calloc(lev->npoints, sizeof(double));
        for (int d = 0; d < 3; d++) {
            lev->V[d]      = calloc(lev->npoints, sizeof(double));
            lev->f_V[d]    = calloc(lev->npoints, sizeof(double));
            lev->save_V[d] = calloc(lev->npoints, sizeof(double));
            lev->L_V[d]    = calloc(lev->npoints, sizeof(double));
            lev->S_M[d]    = calloc(lev->npoints, sizeof(double));
        }
    } else {
        lev->R_tilde = NULL;
        for (int d = 0; d < 3; d++) {
            lev->V[d] = lev->f_V[d] = lev->save_V[d] = NULL;
            lev->L_V[d] = lev->S_M[d] = NULL;
        }
    }
}

static void mg_level_free(mg_level_t *lev)
{
    free(lev->psi);      free(lev->f_psi);
    free(lev->save_psi); free(lev->L_psi);
    free(lev->psi_BL);   free(lev->A2);
    free(lev->R_tilde);
    for (int d = 0; d < 3; d++) {
        free(lev->V[d]);      free(lev->f_V[d]);
        free(lev->save_V[d]); free(lev->L_V[d]);
        free(lev->S_M[d]);
    }
}

/* ================================================================
 * Boundary conditions: zero-Dirichlet in ghost zones
 * ================================================================ */
static void mg_apply_bc_field(double *u, const mg_level_t *lev)
{
    int Nt = lev->Ntotal;
    int gw = lev->ghost;
    for (int k = 0; k < Nt; k++)
        for (int j = 0; j < Nt; j++)
            for (int i = 0; i < Nt; i++)
                if (i < gw || i >= Nt - gw ||
                    j < gw || j >= Nt - gw ||
                    k < gw || k >= Nt - gw)
                    u[MG_IDX(lev, i, j, k)] = 0.0;
}

static void mg_apply_bc(mg_level_t *lev, int four_field)
{
    mg_apply_bc_field(lev->psi, lev);
    if (four_field)
        for (int d = 0; d < 3; d++)
            mg_apply_bc_field(lev->V[d], lev);
}

/* ================================================================
 * Background precomputation — 1-field (BY Hamiltonian only)
 * ================================================================ */
static void mg_precompute_bg_1field(mg_level_t *lev, int n_bh,
                                    const puncture_data_t *bhs)
{
    int Nt = lev->Ntotal;
    for (int k = 0; k < Nt; k++)
        for (int j = 0; j < Nt; j++)
            for (int i = 0; i < Nt; i++) {
                int idx = MG_IDX(lev, i, j, k);
                double x = MG_COORD(lev, i);
                double y = MG_COORD(lev, j);
                double z = MG_COORD(lev, k);
                lev->psi_BL[idx] = brill_lindquist_psi(x, y, z, n_bh, bhs);
                double A_phys[3][3];
                bowen_york_Aij(A_phys, x, y, z, n_bh, bhs);
                lev->A2[idx] = bowen_york_A2(A_phys);
            }
}

/* ================================================================
 * Background precomputation — 4-field (HiSpID coupled system)
 *
 * Computes psi_BL, A^2, R_tilde (conformal Ricci scalar from h_bg),
 * and S_M^i (momentum source from div(A_bg)).
 *
 * R_tilde is computed numerically from the conformal metric h_bg via
 * finite differences of h_bg.  S_M^i is the divergence of the
 * background A_ij (momentum constraint violation of superposed Kerr).
 *
 * Ref: arXiv:1410.8607, coupled_precompute() logic
 * ================================================================ */
static void mg_precompute_bg_4field(mg_level_t *lev, int n_bh,
                                    const puncture_data_t *bhs)
{
    int Nt = lev->Ntotal;
    int gw = lev->ghost;
    double dx = lev->dx;
    double inv_dx = 1.0 / dx;
    size_t np = lev->npoints;

    double *h_bg[6], *A_bg[6];
    for (int c = 0; c < 6; c++) {
        h_bg[c] = calloc(np, sizeof(double));
        A_bg[c] = calloc(np, sizeof(double));
    }

    /* Fill background arrays at every grid point */
    for (int k = 0; k < Nt; k++)
        for (int j = 0; j < Nt; j++)
            for (int i = 0; i < Nt; i++) {
                int idx = MG_IDX(lev, i, j, k);
                double x = MG_COORD(lev, i);
                double y = MG_COORD(lev, j);
                double z = MG_COORD(lev, k);

                lev->psi_BL[idx] = brill_lindquist_psi(x, y, z, n_bh, bhs);

                double h[3][3];
                hispid_conformal_metric(h, x, y, z, n_bh, bhs);
                h_bg[0][idx] = h[0][0]; h_bg[1][idx] = h[0][1];
                h_bg[2][idx] = h[0][2]; h_bg[3][idx] = h[1][1];
                h_bg[4][idx] = h[1][2]; h_bg[5][idx] = h[2][2];

                double A_kerr[3][3];
                hispid_extrinsic(A_kerr, x, y, z, n_bh, bhs);
                double A_by[3][3];
                bowen_york_Aij(A_by, x, y, z, n_bh, bhs);

                double A_total[3][3];
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                        A_total[a][b] = A_kerr[a][b] + A_by[a][b];

                A_bg[0][idx] = A_total[0][0]; A_bg[1][idx] = A_total[0][1];
                A_bg[2][idx] = A_total[0][2]; A_bg[3][idx] = A_total[1][1];
                A_bg[4][idx] = A_total[1][2]; A_bg[5][idx] = A_total[2][2];

                lev->A2[idx] = bowen_york_A2(A_total);
            }

    /* Compute R_tilde from conformal metric via FD Christoffel/Ricci.
     * Ref: coupled_precompute() in old relaxation.c */
    int sx = 1;
    int sy = lev->Ntotal;
    int sz = lev->Ntotal * lev->Ntotal;
    int strides[3] = { sx, sy, sz };
    static const int sym_map[3][3] = {{0,1,2},{1,3,4},{2,4,5}};

    for (int k = gw; k < Nt - gw; k++)
        for (int j_idx = gw; j_idx < Nt - gw; j_idx++)
            for (int i = gw; i < Nt - gw; i++) {
                int idx = MG_IDX(lev, i, j_idx, k);

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
                                               strides[dir], inv_dx);
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
                                               strides[dir], inv_dx);
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
                                    strides[d1], strides[d2], inv_dx);
                                d2_h[a][b][d1][d2] = val;
                                d2_h[a][b][d2][d1] = val;
                                d2_h[b][a][d1][d2] = val;
                                d2_h[b][a][d2][d1] = val;
                            }

                /* Christoffel symbols and Ricci scalar */
                chris_t chris;
                compute_christoffel(d1_h, h_UU, &chris);

                double R_scalar = 0.0;
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++) {
                        double R_ab = 0.0;
                        for (int kk = 0; kk < 3; kk++) {
                            R_ab += 0.5 * chris.contracted[kk] * d1_h[a][b][kk];
                            for (int ll = 0; ll < 3; ll++) {
                                R_ab += -0.5 * h_UU[kk][ll] * d2_h[a][b][kk][ll];
                                double c1 = 0.0, c2 = 0.0, c3 = 0.0;
                                for (int mm = 0; mm < 3; mm++) {
                                    c1 += h_UU[kk][mm] * chris.LLL[b][ll][mm];
                                    c2 += h_UU[kk][mm] * chris.LLL[a][ll][mm];
                                    c3 += h_UU[kk][mm] * chris.LLL[kk][b][mm];
                                }
                                R_ab += chris.ULL[kk][ll][a] * c1;
                                R_ab += chris.ULL[kk][ll][b] * c2;
                                R_ab += chris.ULL[kk][a][ll] * c3;
                            }
                        }
                        R_scalar += h_UU[a][b] * R_ab;
                    }

                lev->R_tilde[idx] = R_scalar;
            }

    /* Compute S_M^i: momentum constraint source.
     * S_M^i = -d_j A_bg^{ij} (flat metric approx) */
    for (int k = gw; k < Nt - gw; k++)
        for (int j_idx = gw; j_idx < Nt - gw; j_idx++)
            for (int i = gw; i < Nt - gw; i++) {
                int idx = MG_IDX(lev, i, j_idx, k);
                for (int d = 0; d < 3; d++) {
                    double div_A = 0.0;
                    for (int e = 0; e < 3; e++)
                        div_A += fd_d1(A_bg[sym_map[d][e]], idx,
                                       strides[e], inv_dx);
                    lev->S_M[d][idx] = -div_A;
                }
            }

    for (int c = 0; c < 6; c++) {
        free(h_bg[c]);
        free(A_bg[c]);
    }
}

/* ================================================================
 * Restriction: cell-centered 8-child volume average (fine -> coarse)
 *
 * For cell-centered grids, the coarse cell center sits between 8 fine
 * children (2x2x2 block).  Simple volume averaging is the standard
 * cell-centered restriction operator.
 *
 * Ref: HPGMG finite-volume restriction (cell-centered)
 * Ref: Zingale, Computational Astrophysics (MG restriction for CC grids)
 * ================================================================ */
static void mg_restrict_field(const double *fine, int Nf_total,
                              double *coarse, int Nc_total,
                              int Nc, int ghost)
{
    int syf = Nf_total;
    int szf = Nf_total * Nf_total;
    int syc = Nc_total;
    int szc = Nc_total * Nc_total;

    for (int K = ghost; K < ghost + Nc; K++) {
        int k = ghost + 2 * (K - ghost);
        for (int J = ghost; J < ghost + Nc; J++) {
            int j = ghost + 2 * (J - ghost);
            for (int I = ghost; I < ghost + Nc; I++) {
                int i = ghost + 2 * (I - ghost);
                int f000 = k * szf + j * syf + i;
                coarse[K * szc + J * syc + I] = 0.125 * (
                    fine[f000]                 + fine[f000 + 1]
                  + fine[f000 + syf]           + fine[f000 + syf + 1]
                  + fine[f000 + szf]           + fine[f000 + szf + 1]
                  + fine[f000 + syf + szf]     + fine[f000 + syf + szf + 1]);
            }
        }
    }
}

/* ================================================================
 * V-cycle prolongation: trilinear interpolation, ADD to fine
 * Cell-centered: left child at -0.25*dx_c, right at +0.25*dx_c
 * ================================================================ */
static void mg_prolongate_add_field(const double *coarse, int Nc_total,
                                    double *fine, int Nf_total,
                                    int Nf, int ghost)
{
    int syc = Nc_total;
    int szc = Nc_total * Nc_total;
    int syf = Nf_total;
    int szf = Nf_total * Nf_total;

    for (int k = ghost; k < ghost + Nf; k++) {
        int Kc = ghost + (k - ghost) / 2;
        int ok = (k - ghost) % 2;
        /* left child: 0.75*Kc + 0.25*(Kc-1); right: 0.75*Kc + 0.25*(Kc+1) */
        int dk = ok ? 1 : -1;

        for (int j = ghost; j < ghost + Nf; j++) {
            int Jc = ghost + (j - ghost) / 2;
            int oj = (j - ghost) % 2;
            int dj = oj ? 1 : -1;

            for (int i = ghost; i < ghost + Nf; i++) {
                int Ic = ghost + (i - ghost) / 2;
                int oi = (i - ghost) % 2;
                int di = oi ? 1 : -1;

                /* Trilinear: product of 1D weights (0.75, 0.25) */
                double val = 0.0;
                for (int ck = 0; ck < 2; ck++) {
                    int CK = ck ? Kc + dk : Kc;
                    double wk = ck ? 0.25 : 0.75;
                    for (int cj = 0; cj < 2; cj++) {
                        int CJ = cj ? Jc + dj : Jc;
                        double wkj = wk * (cj ? 0.25 : 0.75);
                        for (int ci = 0; ci < 2; ci++) {
                            int CI = ci ? Ic + di : Ic;
                            val += wkj * (ci ? 0.25 : 0.75)
                                 * coarse[CK * szc + CJ * syc + CI];
                        }
                    }
                }
                fine[k * szf + j * syf + i] += val;
            }
        }
    }
}

/* ================================================================
 * FMG prolongation: 6th-order Lagrange, OVERWRITE fine
 * Same weights as AMR prolongation (prolong_w from prolongation.c)
 * Ref: Fornberg, SIAM Review 40 (1998) — Lagrange interpolation weights
 * ================================================================ */
#define FMG_STENCIL PROLONG_STENCIL

static void mg_prolongate_fmg_field(const double *coarse, int Nc_total,
                                    double *fine, int Nf_total,
                                    int Nf, int ghost)
{
    int half = FMG_STENCIL / 2;
    int syc = Nc_total;
    int szc = Nc_total * Nc_total;
    int syf = Nf_total;
    int szf = Nf_total * Nf_total;

    for (int k = ghost; k < ghost + Nf; k++) {
        int Kc = ghost + (k - ghost) / 2;
        int ok = (k - ghost) % 2;
        for (int j = ghost; j < ghost + Nf; j++) {
            int Jc = ghost + (j - ghost) / 2;
            int oj = (j - ghost) % 2;
            for (int i = ghost; i < ghost + Nf; i++) {
                int Ic = ghost + (i - ghost) / 2;
                int oi = (i - ghost) % 2;

                double val = 0.0;
                for (int sk = 0; sk < FMG_STENCIL; sk++) {
                    int wk = ok ? (FMG_STENCIL - 1 - sk) : sk;
                    for (int sj = 0; sj < FMG_STENCIL; sj++) {
                        int wj = oj ? (FMG_STENCIL - 1 - sj) : sj;
                        double wkj = prolong_w[wk] * prolong_w[wj];
                        for (int si = 0; si < FMG_STENCIL; si++) {
                            int wi = oi ? (FMG_STENCIL - 1 - si) : si;
                            int src = (Kc - half + sk) * szc
                                    + (Jc - half + sj) * syc
                                    + (Ic - half + si);
                            val += wkj * prolong_w[wi] * coarse[src];
                        }
                    }
                }
                fine[k * szf + j * syf + i] = val;
            }
        }
    }
}

/* ================================================================
 * Newton-GS smoother — 1-field (BY Hamiltonian)
 *
 * L(psi) = Lap(psi) + (1/8)*A^2*psi_total^{-7}
 * Jacobian diagonal: 3*FD_D2_CENTER_WEIGHT/dx^2 + dS/dpsi
 *   where dS/dpsi = -(7/8)*A^2*psi_total^{-8}
 *
 * 8-color ordering: color = (i%2) + 2*(j%2) + 4*(k%2)
 * Within each color all points are independent at ±1 distance.
 * Ref: arXiv:2510.11152 Section 3.1
 * ================================================================ */
static void newton_gs_sweep_1field(mg_level_t *lev)
{
    int gw = lev->ghost;
    int N  = lev->N;
    int Nt = lev->Ntotal;
    double dx = lev->dx;
    double inv_dx = 1.0 / dx;
    double dx2 = dx * dx;
    int sx = 1, sy = Nt, sz = Nt * Nt;
    double J_lap = 3.0 * FD_D2_CENTER_WEIGHT / dx2;

    for (int color = 0; color < 8; color++) {
        int c0 = color & 1;
        int c1 = (color >> 1) & 1;
        int c2 = (color >> 2) & 1;

        for (int k = gw + c2; k < gw + N; k += 2)
            for (int j = gw + c1; j < gw + N; j += 2)
                for (int i = gw + c0; i < gw + N; i += 2) {
                    int idx = k * sz + j * sy + i;

                    double psi_tot = lev->psi_BL[idx] + lev->psi[idx];
                    if (psi_tot < 0.1) psi_tot = 0.1;

                    double lap = fd_d2(lev->psi, idx, sx, inv_dx)
                               + fd_d2(lev->psi, idx, sy, inv_dx)
                               + fd_d2(lev->psi, idx, sz, inv_dx);

                    double p2 = psi_tot * psi_tot;
                    double p4 = p2 * p2;
                    double p7 = p4 * p2 * psi_tot;
                    double p8 = p7 * psi_tot;

                    double source = 0.125 * lev->A2[idx] / p7;
                    double residual = lap + source - lev->f_psi[idx];

                    double dS = -0.875 * lev->A2[idx] / p8;
                    double J_diag = J_lap + dS;

                    lev->psi[idx] -= residual / J_diag;
                }
    }
    mg_apply_bc_field(lev->psi, lev);
}

/* ================================================================
 * Newton-GS smoother — 4-field (HiSpID coupled)
 *
 * Hamiltonian: L(psi) = Lap(psi) + (R_tilde/8)*psi_tot + (A^2/8)*psi_tot^{-7}
 * Momentum:   L(V^d) = Lap(V^d) + (1/3)*d_d(div V) + S_M^d
 *
 * Psi Jacobian:  3*FD_D2_CENTER_WEIGHT/dx^2 + R_tilde/8 - (7/8)*A^2*psi_tot^{-8}
 * V^d Jacobian:  3*FD_D2_CENTER_WEIGHT/dx^2 + (1/3)*FD_D2_CENTER_WEIGHT/dx^2
 * ================================================================ */
static void newton_gs_sweep_4field(mg_level_t *lev)
{
    int gw = lev->ghost;
    int N  = lev->N;
    int Nt = lev->Ntotal;
    double dx = lev->dx;
    double inv_dx = 1.0 / dx;
    double dx2 = dx * dx;
    int sx = 1, sy = Nt, sz = Nt * Nt;
    int strides[3] = { sx, sy, sz };
    double J_lap = 3.0 * FD_D2_CENTER_WEIGHT / dx2;
    double J_V_diag = J_lap + (1.0 / 3.0) * FD_D2_CENTER_WEIGHT / dx2;

    for (int color = 0; color < 8; color++) {
        int c0 = color & 1;
        int c1 = (color >> 1) & 1;
        int c2 = (color >> 2) & 1;

        for (int k = gw + c2; k < gw + N; k += 2)
            for (int j = gw + c1; j < gw + N; j += 2)
                for (int i = gw + c0; i < gw + N; i += 2) {
                    int idx = k * sz + j * sy + i;

                    /* --- Hamiltonian (psi) --- */
                    double psi_tot = lev->psi_BL[idx] + lev->psi[idx];
                    if (psi_tot < 0.1) psi_tot = 0.1;

                    double lap_psi = fd_d2(lev->psi, idx, sx, inv_dx)
                                   + fd_d2(lev->psi, idx, sy, inv_dx)
                                   + fd_d2(lev->psi, idx, sz, inv_dx);

                    double p2 = psi_tot * psi_tot;
                    double p4 = p2 * p2;
                    double p7 = p4 * p2 * psi_tot;
                    double p8 = p7 * psi_tot;

                    double src_H = lev->R_tilde[idx] * 0.125 * psi_tot
                                 + lev->A2[idx] * 0.125 / p7;
                    double res_psi = lap_psi + src_H - lev->f_psi[idx];
                    double dS_psi = 0.125 * lev->R_tilde[idx]
                                  - 0.875 * lev->A2[idx] / p8;
                    lev->psi[idx] -= res_psi / (J_lap + dS_psi);

                    /* --- Momentum (V^d) --- */
                    for (int d = 0; d < 3; d++) {
                        double lap_V = fd_d2(lev->V[d], idx, sx, inv_dx)
                                     + fd_d2(lev->V[d], idx, sy, inv_dx)
                                     + fd_d2(lev->V[d], idx, sz, inv_dx);
                        double d_divV = 0.0;
                        for (int e = 0; e < 3; e++) {
                            if (e == d)
                                d_divV += fd_d2(lev->V[e], idx,
                                                strides[e], inv_dx);
                            else
                                d_divV += fd_d2_mixed(lev->V[e], idx,
                                                      strides[d],
                                                      strides[e], inv_dx);
                        }
                        double res_V = lap_V + d_divV / 3.0
                                     + lev->S_M[d][idx] - lev->f_V[d][idx];
                        lev->V[d][idx] -= res_V / J_V_diag;
                    }
                }
    }

    mg_apply_bc_field(lev->psi, lev);
    for (int d = 0; d < 3; d++)
        mg_apply_bc_field(lev->V[d], lev);
}

/* ================================================================
 * Operator evaluation: compute L(u) at all interior points
 * ================================================================ */
static void mg_compute_operator(mg_level_t *lev, int four_field)
{
    int gw = lev->ghost;
    int N  = lev->N;
    int Nt = lev->Ntotal;
    double dx = lev->dx;
    double inv_dx = 1.0 / dx;
    int sx = 1, sy = Nt, sz = Nt * Nt;
    int strides[3] = { sx, sy, sz };

    for (int k = gw; k < gw + N; k++)
        for (int j = gw; j < gw + N; j++)
            for (int i = gw; i < gw + N; i++) {
                int idx = k * sz + j * sy + i;

                /* Hamiltonian: L(psi) */
                double psi_tot = lev->psi_BL[idx] + lev->psi[idx];
                if (psi_tot < 0.1) psi_tot = 0.1;

                double lap = fd_d2(lev->psi, idx, sx, inv_dx)
                           + fd_d2(lev->psi, idx, sy, inv_dx)
                           + fd_d2(lev->psi, idx, sz, inv_dx);

                double p2 = psi_tot * psi_tot;
                double p4 = p2 * p2;
                double p7 = p4 * p2 * psi_tot;

                if (four_field) {
                    lev->L_psi[idx] = lap
                        + lev->R_tilde[idx] * 0.125 * psi_tot
                        + lev->A2[idx] * 0.125 / p7;
                } else {
                    lev->L_psi[idx] = lap + 0.125 * lev->A2[idx] / p7;
                }

                /* Momentum: L(V^d) */
                if (four_field) {
                    for (int d = 0; d < 3; d++) {
                        double lap_V = fd_d2(lev->V[d], idx, sx, inv_dx)
                                     + fd_d2(lev->V[d], idx, sy, inv_dx)
                                     + fd_d2(lev->V[d], idx, sz, inv_dx);
                        double d_divV = 0.0;
                        for (int e = 0; e < 3; e++) {
                            if (e == d)
                                d_divV += fd_d2(lev->V[e], idx,
                                                strides[e], inv_dx);
                            else
                                d_divV += fd_d2_mixed(lev->V[e], idx,
                                                      strides[d],
                                                      strides[e], inv_dx);
                        }
                        lev->L_V[d][idx] = lap_V + d_divV / 3.0
                                         + lev->S_M[d][idx];
                    }
                }
            }
}

/* ================================================================
 * Residual norm: ||f - L(u)||_L2
 * ================================================================ */
static double mg_residual_norm(mg_level_t *lev, int four_field)
{
    mg_compute_operator(lev, four_field);

    int gw = lev->ghost;
    int N  = lev->N;
    int Nt = lev->Ntotal;
    int sy = Nt, sz = Nt * Nt;
    double sum_psi = 0.0, sum_V = 0.0;
    int count = 0;

    for (int k = gw; k < gw + N; k++)
        for (int j = gw; j < gw + N; j++)
            for (int i = gw; i < gw + N; i++) {
                int idx = k * sz + j * sy + i;
                double r = lev->f_psi[idx] - lev->L_psi[idx];
                sum_psi += r * r;
                if (four_field)
                    for (int d = 0; d < 3; d++) {
                        double rv = lev->f_V[d][idx] - lev->L_V[d][idx];
                        sum_V += rv * rv;
                    }
                count++;
            }

    double l2_psi = (count > 0) ? sqrt(sum_psi / count) : 0.0;
    if (!four_field) return l2_psi;
    double l2_V = (count > 0) ? sqrt(sum_V / (3 * count)) : 0.0;
    return (l2_psi > l2_V) ? l2_psi : l2_V;
}

/* ================================================================
 * FAS V-cycle (recursive)
 *
 * FAS solves L(u) = f on the coarse grid where f includes the tau
 * correction: f_coarse = L_coarse(Restrict(u_fine)) + Restrict(r_fine).
 *
 * Ref: Trottenberg et al., Multigrid Methods, Algorithm 2.5.2
 * Ref: arXiv:0705.1486 Section 2.2
 * ================================================================ */
static void mg_vcycle(mg_level_t *levels, int n_levels, int level,
                      int four_field)
{
    mg_level_t *fine = &levels[level];

    /* Coarsest level: just smooth */
    if (level == n_levels - 1) {
        for (int s = 0; s < MG_NU_COARSE; s++) {
            if (four_field) newton_gs_sweep_4field(fine);
            else            newton_gs_sweep_1field(fine);
        }
        return;
    }

    mg_level_t *coarse = &levels[level + 1];
    int Nc = coarse->N;
    int gw = coarse->ghost;
    int sy_c = coarse->Ntotal;
    int sz_c = coarse->Ntotal * coarse->Ntotal;

    /* Pre-smooth */
    for (int s = 0; s < MG_NU_PRE; s++) {
        if (four_field) newton_gs_sweep_4field(fine);
        else            newton_gs_sweep_1field(fine);
    }

    /* Compute L(u) on fine, then residual r = f - L(u) stored in L arrays */
    mg_compute_operator(fine, four_field);
    {
        int Nf = fine->N;
        int gw_f = fine->ghost;
        int sy_f = fine->Ntotal;
        int sz_f = fine->Ntotal * fine->Ntotal;
        for (int k = gw_f; k < gw_f + Nf; k++)
            for (int j = gw_f; j < gw_f + Nf; j++)
                for (int i = gw_f; i < gw_f + Nf; i++) {
                    int idx = k * sz_f + j * sy_f + i;
                    fine->L_psi[idx] = fine->f_psi[idx] - fine->L_psi[idx];
                    if (four_field)
                        for (int d = 0; d < 3; d++)
                            fine->L_V[d][idx] =
                                fine->f_V[d][idx] - fine->L_V[d][idx];
                }
    }

    /* Restrict fine solution → coarse save */
    mg_restrict_field(fine->psi, fine->Ntotal,
                      coarse->save_psi, coarse->Ntotal, Nc, gw);
    mg_apply_bc_field(coarse->save_psi, coarse);

    /* Restrict fine residual → coarse f (temporary) */
    mg_restrict_field(fine->L_psi, fine->Ntotal,
                      coarse->f_psi, coarse->Ntotal, Nc, gw);

    if (four_field) {
        for (int d = 0; d < 3; d++) {
            mg_restrict_field(fine->V[d], fine->Ntotal,
                              coarse->save_V[d], coarse->Ntotal, Nc, gw);
            mg_apply_bc_field(coarse->save_V[d], coarse);
            mg_restrict_field(fine->L_V[d], fine->Ntotal,
                              coarse->f_V[d], coarse->Ntotal, Nc, gw);
        }
    }

    /* Set coarse initial guess = restricted solution */
    memcpy(coarse->psi, coarse->save_psi, coarse->npoints * sizeof(double));
    if (four_field)
        for (int d = 0; d < 3; d++)
            memcpy(coarse->V[d], coarse->save_V[d],
                   coarse->npoints * sizeof(double));

    /* Compute L(restricted_solution) on coarse */
    mg_compute_operator(coarse, four_field);

    /* Tau correction: f_coarse = L(save) + Restrict(r) */
    for (int k = gw; k < gw + Nc; k++)
        for (int j = gw; j < gw + Nc; j++)
            for (int i = gw; i < gw + Nc; i++) {
                int idx = k * sz_c + j * sy_c + i;
                coarse->f_psi[idx] += coarse->L_psi[idx];
                if (four_field)
                    for (int d = 0; d < 3; d++)
                        coarse->f_V[d][idx] += coarse->L_V[d][idx];
            }

    /* Recursive coarse solve */
    mg_vcycle(levels, n_levels, level + 1, four_field);

    /* Compute correction = coarse_solution - saved_restricted.
     * Store in save arrays (no longer needed). */
    for (int k = gw; k < gw + Nc; k++)
        for (int j = gw; j < gw + Nc; j++)
            for (int i = gw; i < gw + Nc; i++) {
                int idx = k * sz_c + j * sy_c + i;
                coarse->save_psi[idx] =
                    coarse->psi[idx] - coarse->save_psi[idx];
                if (four_field)
                    for (int d = 0; d < 3; d++)
                        coarse->save_V[d][idx] =
                            coarse->V[d][idx] - coarse->save_V[d][idx];
            }
    mg_apply_bc_field(coarse->save_psi, coarse);

    /* Prolongate correction and add to fine */
    mg_prolongate_add_field(coarse->save_psi, coarse->Ntotal,
                            fine->psi, fine->Ntotal, fine->N, fine->ghost);
    if (four_field)
        for (int d = 0; d < 3; d++) {
            mg_apply_bc_field(coarse->save_V[d], coarse);
            mg_prolongate_add_field(coarse->save_V[d], coarse->Ntotal,
                                   fine->V[d], fine->Ntotal,
                                   fine->N, fine->ghost);
        }

    mg_apply_bc(fine, four_field);

    /* Post-smooth */
    for (int s = 0; s < MG_NU_POST; s++) {
        if (four_field) newton_gs_sweep_4field(fine);
        else            newton_gs_sweep_1field(fine);
    }
}

/* ================================================================
 * FMG (Full Multigrid) outer loop
 *
 * Start on coarsest grid, solve there, then ascend — each solved
 * level provides the initial guess for the next finer level.
 * One FMG pass ~ 1.14 V-cycles of work, achieves discretization accuracy.
 *
 * Ref: Trottenberg et al., Algorithm 2.6.3
 * Ref: HPGMG finite-volume/source/mg.c FMG mode
 * ================================================================ */
static void mg_fmg(mg_level_t *levels, int n_levels, int four_field)
{
    int coarsest = n_levels - 1;

    /* Step 1: Solve on coarsest grid */
    for (int s = 0; s < MG_NU_COARSE; s++) {
        if (four_field) newton_gs_sweep_4field(&levels[coarsest]);
        else            newton_gs_sweep_1field(&levels[coarsest]);
    }

    /* Step 2: Ascend from coarsest to finest */
    for (int lev = coarsest - 1; lev >= 0; lev--) {
        /* 4th-order prolongation: high-quality initial guess */
        mg_prolongate_fmg_field(levels[lev + 1].psi, levels[lev + 1].Ntotal,
                                levels[lev].psi, levels[lev].Ntotal,
                                levels[lev].N, levels[lev].ghost);
        if (four_field)
            for (int d = 0; d < 3; d++)
                mg_prolongate_fmg_field(levels[lev + 1].V[d],
                                        levels[lev + 1].Ntotal,
                                        levels[lev].V[d],
                                        levels[lev].Ntotal,
                                        levels[lev].N, levels[lev].ghost);

        mg_apply_bc(&levels[lev], four_field);

        /* One V-cycle to polish */
        mg_vcycle(levels, n_levels, lev, four_field);
    }
}

/* ================================================================
 * Public API: 1-field Bowen-York Hamiltonian constraint
 * ================================================================ */
double relaxation_solve(grid_t *g, int n_bh, const puncture_data_t *bhs,
                        double tol, int max_iter, int verbose)
{
    /* Build multigrid hierarchy: N, N/2, ... down to MG_N_MIN */
    int n_levels = 1;
    {
        int N = g->N;
        while (N / (1 << n_levels) >= MG_N_MIN && n_levels < MG_MAX_LEVELS)
            n_levels++;
    }

    mg_level_t *levels = calloc(n_levels, sizeof(mg_level_t));
    for (int l = 0; l < n_levels; l++) {
        int N_l = g->N / (1 << l);
        mg_level_init(&levels[l], N_l, g->L, /*four_field=*/0);
        mg_precompute_bg_1field(&levels[l], n_bh, bhs);
    }

    if (verbose)
        printf("    FAS Multigrid: N=%d, %d levels, dx=%.4f, tol=%.2e\n",
               g->N, n_levels, levels[0].dx, tol);

    /* FMG pass */
    mg_fmg(levels, n_levels, /*four_field=*/0);

    double residual = mg_residual_norm(&levels[0], 0);
    if (verbose)
        printf("    FMG done: residual = %.6e\n", residual);

    /* Post-FMG V-cycles if needed */
    for (int cycle = 0; cycle < max_iter && residual > tol; cycle++) {
        mg_vcycle(levels, n_levels, 0, 0);
        residual = mg_residual_norm(&levels[0], 0);
        if (verbose)
            printf("    Post-FMG V-cycle %d: residual = %.6e\n",
                   cycle + 1, residual);
    }

    /* Build full psi = psi_BL + correction, set CCZ4 fields */
    double *psi_full = calloc(g->npoints, sizeof(double));
    for (size_t idx = 0; idx < g->npoints; idx++)
        psi_full[idx] = levels[0].psi_BL[idx] + levels[0].psi[idx];

    set_ccz4_from_psi(g, psi_full, n_bh, bhs);

    free(psi_full);
    for (int l = 0; l < n_levels; l++)
        mg_level_free(&levels[l]);
    free(levels);
    return residual;
}

/* ================================================================
 * Public API: 4-field HiSpID coupled system
 * ================================================================ */
double relaxation_solve_coupled(grid_t *g, int n_bh, const puncture_data_t *bhs,
                                double tol, int max_iter, int verbose)
{
    int n_levels = 1;
    {
        int N = g->N;
        while (N / (1 << n_levels) >= MG_N_MIN && n_levels < MG_MAX_LEVELS)
            n_levels++;
    }

    mg_level_t *levels = calloc(n_levels, sizeof(mg_level_t));
    for (int l = 0; l < n_levels; l++) {
        int N_l = g->N / (1 << l);
        mg_level_init(&levels[l], N_l, g->L, /*four_field=*/1);
    }

    if (verbose)
        printf("    HiSpID: precomputing background (QI Kerr metric)...\n");
    for (int l = 0; l < n_levels; l++)
        mg_precompute_bg_4field(&levels[l], n_bh, bhs);

    if (verbose)
        printf("    FAS Multigrid (4-field): N=%d, %d levels, dx=%.4f, tol=%.2e\n",
               g->N, n_levels, levels[0].dx, tol);

    mg_fmg(levels, n_levels, /*four_field=*/1);

    double residual = mg_residual_norm(&levels[0], 1);
    if (verbose)
        printf("    FMG done: residual = %.6e\n", residual);

    for (int cycle = 0; cycle < max_iter && residual > tol; cycle++) {
        mg_vcycle(levels, n_levels, 0, 1);
        residual = mg_residual_norm(&levels[0], 1);
        if (verbose)
            printf("    Post-FMG V-cycle %d: residual = %.6e\n",
                   cycle + 1, residual);
    }

    /* Build full psi and set CCZ4 fields */
    double *psi_full = calloc(g->npoints, sizeof(double));
    for (size_t idx = 0; idx < g->npoints; idx++)
        psi_full[idx] = levels[0].psi_BL[idx] + levels[0].psi[idx];

    double *V_arr[3] = { levels[0].V[0], levels[0].V[1], levels[0].V[2] };
    set_ccz4_from_hispid(g, psi_full, V_arr, n_bh, bhs);

    free(psi_full);
    for (int l = 0; l < n_levels; l++)
        mg_level_free(&levels[l]);
    free(levels);
    return residual;
}
