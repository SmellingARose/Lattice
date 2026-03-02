/*
 * Lattice — 3D Numerical Relativity
 * AMR FAS Multigrid constraint solver.
 *
 * Composite multigrid on block-structured AMR mesh above a uniform
 * FAS multigrid hierarchy.  Level 0 = full-domain uniform grid (same
 * resolution as the existing solver); levels 1+ = refinement near
 * punctures.  The uniform solver handles coarsening below level 0.
 *
 * Solver fields stored in grid_t array slots per block:
 *   fields[0]    = psi (solution)
 *   fields[1..3] = V^1..3 (4-field only)
 *   fields[4]    = psi_BL (background)
 *   fields[5]    = A^2 (background)
 *   fields[6]    = R_tilde (4-field background)
 *   fields[7..9] = S_M^1..3 (4-field background)
 *   rhs[0..3]    = f_psi, f_V (FAS target)
 *   scratch[0..3]= save_psi, save_V (for correction)
 *   accum[0..3]  = L_psi, L_V (operator evaluation)
 *
 * Ref: arXiv:0705.1486 (Natchu & Matzner, 4th-order MG for BH data)
 * Ref: arXiv:2510.11152 (GPU FAS multigrid, 8-color MCGS)
 * Ref: Trottenberg et al., Multigrid Methods (composite AMR MG)
 */

#include "relaxation_amr.h"
#include "relaxation.h"
#include "bowen_york.h"
#include "kerr_quasi_isotropic.h"
#include "../numerics/finite_diff.h"
#include "../geometry/tensor_utils.h"
#include "../amr/mesh.h"
#include "../amr/block.h"
#include "../amr/refine.h"
#include "../amr/ghost_exchange.h"
#include "../amr/prolongation.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Solver field slot indices within grid_t arrays
 * ================================================================ */
#define SOL_PSI   0   /* psi correction (fields[0]) */
#define SOL_V1    1   /* V^1 correction (fields[1]) */
#define SOL_V2    2   /* V^2 correction (fields[2]) */
#define SOL_V3    3   /* V^3 correction (fields[3]) */
#define BG_PSI_BL 4   /* Brill-Lindquist psi_BL (fields[4]) */
#define BG_A2     5   /* A_ij A^ij (fields[5]) */
#define BG_RTILDE 6   /* conformal Ricci scalar (fields[6]) */
#define BG_SM1    7   /* momentum source S_M^1 (fields[7]) */
#define BG_SM2    8   /* momentum source S_M^2 (fields[8]) */
#define BG_SM3    9   /* momentum source S_M^3 (fields[9]) */

#define MG_AMR_N_FIELDS 10  /* total solver fields per block */

/* Multigrid parameters (matching relaxation.c) */
#define MG_N_MIN      16
#define MG_NU_PRE      4
#define MG_NU_POST     4
#define MG_NU_COARSE  50
#define MG_MAX_LEVELS  8

/* FD_D2 center weight for Jacobian diagonal */
#if FD_ORDER == 6
#define FD_D2_CENTER_WEIGHT (-49.0 / 18.0)
#else
#define FD_D2_CENTER_WEIGHT (-5.0 / 2.0)
#endif

/* ================================================================
 * mg_amr_t: composite AMR + uniform multigrid hierarchy
 * ================================================================ */

/* Forward-declare mg_level_t (from relaxation.c, not exported).
 * We replicate the structure here since relaxation.c uses static functions. */
typedef struct {
    double *psi, *V[3];
    double *f_psi, *f_V[3];
    double *save_psi, *save_V[3];
    double *L_psi, *L_V[3];
    double *psi_BL, *A2, *R_tilde, *S_M[3];
    int N, ghost, Ntotal;
    double dx, L;
    size_t npoints;
} mg_level_amr_t;

typedef struct {
    mesh_t  *mesh;          /* AMR mesh owned by solver */
    int      n_amr_levels;  /* AMR levels above base (0 = level 0 only) */
    int      four_field;    /* 0 = 1-field BY, 1 = 4-field HiSpID */

    /* Uniform multigrid below AMR level 0 */
    mg_level_amr_t *mg_levels;
    int             n_mg_levels;

    /* Puncture data (referenced, not owned) */
    int              n_bh;
    const puncture_data_t *bhs;
} mg_amr_t;

/* ================================================================
 * Uniform MG level allocation (mirrors relaxation.c mg_level_init)
 * ================================================================ */
static void mg_level_amr_init(mg_level_amr_t *lev, int N, double L,
                               int four_field)
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

static void mg_level_amr_free(mg_level_amr_t *lev)
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

/* Indexing macros for uniform MG levels */
#define UMG_IDX(lev, i, j, k) \
    ((k) * (lev)->Ntotal * (lev)->Ntotal + (j) * (lev)->Ntotal + (i))
#define UMG_COORD(lev, i) \
    (((i) - (lev)->ghost + 0.5) * (lev)->dx - (lev)->L * 0.5)

/* ================================================================
 * Uniform MG helpers (reimplemented since relaxation.c's are static)
 * ================================================================ */

static void umg_apply_bc_field(double *u, const mg_level_amr_t *lev)
{
    int Nt = lev->Ntotal;
    int gw = lev->ghost;
    for (int k = 0; k < Nt; k++)
        for (int j = 0; j < Nt; j++)
            for (int i = 0; i < Nt; i++)
                if (i < gw || i >= Nt - gw ||
                    j < gw || j >= Nt - gw ||
                    k < gw || k >= Nt - gw)
                    u[UMG_IDX(lev, i, j, k)] = 0.0;
}

static void umg_apply_bc(mg_level_amr_t *lev, int four_field)
{
    umg_apply_bc_field(lev->psi, lev);
    if (four_field)
        for (int d = 0; d < 3; d++)
            umg_apply_bc_field(lev->V[d], lev);
}

/* 1-field Newton-GS smoother on uniform level */
static void umg_sweep_1field(mg_level_amr_t *lev)
{
    int gw = lev->ghost, N = lev->N, Nt = lev->Ntotal;
    double dx = lev->dx, dx2 = dx * dx;
    double inv_dx = 1.0 / dx;
    int sx = 1, sy = Nt, sz = Nt * Nt;
    double J_lap = 3.0 * FD_D2_CENTER_WEIGHT / dx2;

    for (int color = 0; color < 8; color++) {
        int c0 = color & 1, c1 = (color >> 1) & 1, c2 = (color >> 2) & 1;
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
                    lev->psi[idx] -= residual / (J_lap + dS);
                }
    }
    umg_apply_bc_field(lev->psi, lev);
}

/* 4-field Newton-GS smoother on uniform level */
static void umg_sweep_4field(mg_level_amr_t *lev)
{
    int gw = lev->ghost, N = lev->N, Nt = lev->Ntotal;
    double dx = lev->dx, dx2 = dx * dx;
    double inv_dx = 1.0 / dx;
    int sx = 1, sy = Nt, sz = Nt * Nt;
    int strides[3] = { sx, sy, sz };
    double J_lap = 3.0 * FD_D2_CENTER_WEIGHT / dx2;
    double J_V_diag = J_lap + (1.0 / 3.0) * FD_D2_CENTER_WEIGHT / dx2;

    for (int color = 0; color < 8; color++) {
        int c0 = color & 1, c1 = (color >> 1) & 1, c2 = (color >> 2) & 1;
        for (int k = gw + c2; k < gw + N; k += 2)
            for (int j = gw + c1; j < gw + N; j += 2)
                for (int i = gw + c0; i < gw + N; i += 2) {
                    int idx = k * sz + j * sy + i;
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
    umg_apply_bc_field(lev->psi, lev);
    for (int d = 0; d < 3; d++)
        umg_apply_bc_field(lev->V[d], lev);
}

/* Operator evaluation on uniform level */
static void umg_compute_operator(mg_level_amr_t *lev, int four_field)
{
    int gw = lev->ghost, N = lev->N, Nt = lev->Ntotal;
    double dx = lev->dx;
    double inv_dx = 1.0 / dx;
    int sx = 1, sy = Nt, sz = Nt * Nt;
    int strides[3] = { sx, sy, sz };

    for (int k = gw; k < gw + N; k++)
        for (int j = gw; j < gw + N; j++)
            for (int i = gw; i < gw + N; i++) {
                int idx = k * sz + j * sy + i;
                double psi_tot = lev->psi_BL[idx] + lev->psi[idx];
                if (psi_tot < 0.1) psi_tot = 0.1;
                double lap = fd_d2(lev->psi, idx, sx, inv_dx)
                           + fd_d2(lev->psi, idx, sy, inv_dx)
                           + fd_d2(lev->psi, idx, sz, inv_dx);
                double p2 = psi_tot * psi_tot;
                double p4 = p2 * p2;
                double p7 = p4 * p2 * psi_tot;
                if (four_field)
                    lev->L_psi[idx] = lap
                        + lev->R_tilde[idx] * 0.125 * psi_tot
                        + lev->A2[idx] * 0.125 / p7;
                else
                    lev->L_psi[idx] = lap + 0.125 * lev->A2[idx] / p7;

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

/* Cell-centered 8-child restriction (fine -> coarse) on uniform levels */
static void umg_restrict_field(const double *fine, int Nf_total,
                                double *coarse, int Nc_total,
                                int Nc, int ghost)
{
    int syf = Nf_total, szf = Nf_total * Nf_total;
    int syc = Nc_total, szc = Nc_total * Nc_total;
    for (int K = ghost; K < ghost + Nc; K++) {
        int k = ghost + 2 * (K - ghost);
        for (int J = ghost; J < ghost + Nc; J++) {
            int j = ghost + 2 * (J - ghost);
            for (int I = ghost; I < ghost + Nc; I++) {
                int i = ghost + 2 * (I - ghost);
                int f000 = k * szf + j * syf + i;
                coarse[K * szc + J * syc + I] = 0.125 * (
                    fine[f000]             + fine[f000 + 1]
                  + fine[f000 + syf]       + fine[f000 + syf + 1]
                  + fine[f000 + szf]       + fine[f000 + szf + 1]
                  + fine[f000 + syf + szf] + fine[f000 + syf + szf + 1]);
            }
        }
    }
}

/* Trilinear prolongation, ADD to fine (V-cycle correction) */
static void umg_prolongate_add(const double *coarse, int Nc_total,
                                double *fine, int Nf_total,
                                int Nf, int ghost)
{
    int syc = Nc_total, szc = Nc_total * Nc_total;
    int syf = Nf_total, szf = Nf_total * Nf_total;
    for (int k = ghost; k < ghost + Nf; k++) {
        int Kc = ghost + (k - ghost) / 2;
        int ok = (k - ghost) % 2;
        int dk = ok ? 1 : -1;
        for (int j = ghost; j < ghost + Nf; j++) {
            int Jc = ghost + (j - ghost) / 2;
            int oj = (j - ghost) % 2;
            int dj = oj ? 1 : -1;
            for (int i = ghost; i < ghost + Nf; i++) {
                int Ic = ghost + (i - ghost) / 2;
                int oi = (i - ghost) % 2;
                int di = oi ? 1 : -1;
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

/* FMG prolongation: 6th-order Lagrange, OVERWRITE fine */
#define FMG_STENCIL PROLONG_STENCIL
static void umg_prolongate_fmg(const double *coarse, int Nc_total,
                                double *fine, int Nf_total,
                                int Nf, int ghost)
{
    int half = FMG_STENCIL / 2;
    int syc = Nc_total, szc = Nc_total * Nc_total;
    int syf = Nf_total, szf = Nf_total * Nf_total;
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

/* Uniform V-cycle (reimplemented from relaxation.c) */
static void umg_vcycle(mg_level_amr_t *levels, int n_levels, int level,
                        int four_field)
{
    mg_level_amr_t *fine = &levels[level];
    if (level == n_levels - 1) {
        for (int s = 0; s < MG_NU_COARSE; s++) {
            if (four_field) umg_sweep_4field(fine);
            else            umg_sweep_1field(fine);
        }
        return;
    }
    mg_level_amr_t *coarse = &levels[level + 1];
    int Nc = coarse->N, gw = coarse->ghost;
    int sy_c = coarse->Ntotal, sz_c = coarse->Ntotal * coarse->Ntotal;

    /* Pre-smooth */
    for (int s = 0; s < MG_NU_PRE; s++) {
        if (four_field) umg_sweep_4field(fine);
        else            umg_sweep_1field(fine);
    }
    /* Compute L(u), then residual r = f - L(u) */
    umg_compute_operator(fine, four_field);
    {
        int Nf = fine->N, gw_f = fine->ghost;
        int sy_f = fine->Ntotal, sz_f = fine->Ntotal * fine->Ntotal;
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
    /* Restrict solution + residual */
    umg_restrict_field(fine->psi, fine->Ntotal,
                       coarse->save_psi, coarse->Ntotal, Nc, gw);
    umg_apply_bc_field(coarse->save_psi, coarse);
    umg_restrict_field(fine->L_psi, fine->Ntotal,
                       coarse->f_psi, coarse->Ntotal, Nc, gw);
    if (four_field) {
        for (int d = 0; d < 3; d++) {
            umg_restrict_field(fine->V[d], fine->Ntotal,
                              coarse->save_V[d], coarse->Ntotal, Nc, gw);
            umg_apply_bc_field(coarse->save_V[d], coarse);
            umg_restrict_field(fine->L_V[d], fine->Ntotal,
                              coarse->f_V[d], coarse->Ntotal, Nc, gw);
        }
    }
    /* Coarse initial guess = restricted */
    memcpy(coarse->psi, coarse->save_psi, coarse->npoints * sizeof(double));
    if (four_field)
        for (int d = 0; d < 3; d++)
            memcpy(coarse->V[d], coarse->save_V[d],
                   coarse->npoints * sizeof(double));
    /* Tau correction */
    umg_compute_operator(coarse, four_field);
    for (int k = gw; k < gw + Nc; k++)
        for (int j = gw; j < gw + Nc; j++)
            for (int i = gw; i < gw + Nc; i++) {
                int idx = k * sz_c + j * sy_c + i;
                coarse->f_psi[idx] += coarse->L_psi[idx];
                if (four_field)
                    for (int d = 0; d < 3; d++)
                        coarse->f_V[d][idx] += coarse->L_V[d][idx];
            }
    /* Recursive */
    umg_vcycle(levels, n_levels, level + 1, four_field);
    /* Correction = coarse - saved */
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
    umg_apply_bc_field(coarse->save_psi, coarse);
    /* Prolongate correction */
    umg_prolongate_add(coarse->save_psi, coarse->Ntotal,
                       fine->psi, fine->Ntotal, fine->N, fine->ghost);
    if (four_field)
        for (int d = 0; d < 3; d++) {
            umg_apply_bc_field(coarse->save_V[d], coarse);
            umg_prolongate_add(coarse->save_V[d], coarse->Ntotal,
                              fine->V[d], fine->Ntotal,
                              fine->N, fine->ghost);
        }
    umg_apply_bc(fine, four_field);
    /* Post-smooth */
    for (int s = 0; s < MG_NU_POST; s++) {
        if (four_field) umg_sweep_4field(fine);
        else            umg_sweep_1field(fine);
    }
}

/* Uniform FMG */
static void umg_fmg(mg_level_amr_t *levels, int n_levels, int four_field)
{
    int coarsest = n_levels - 1;
    for (int s = 0; s < MG_NU_COARSE; s++) {
        if (four_field) umg_sweep_4field(&levels[coarsest]);
        else            umg_sweep_1field(&levels[coarsest]);
    }
    for (int lev = coarsest - 1; lev >= 0; lev--) {
        umg_prolongate_fmg(levels[lev + 1].psi, levels[lev + 1].Ntotal,
                           levels[lev].psi, levels[lev].Ntotal,
                           levels[lev].N, levels[lev].ghost);
        if (four_field)
            for (int d = 0; d < 3; d++)
                umg_prolongate_fmg(levels[lev + 1].V[d],
                                   levels[lev + 1].Ntotal,
                                   levels[lev].V[d],
                                   levels[lev].Ntotal,
                                   levels[lev].N, levels[lev].ghost);
        umg_apply_bc(&levels[lev], four_field);
        umg_vcycle(levels, n_levels, lev, four_field);
    }
}

/* Background precomputation for uniform levels — 1-field */
static void umg_precompute_bg_1field(mg_level_amr_t *lev, int n_bh,
                                      const puncture_data_t *bhs)
{
    int Nt = lev->Ntotal;
    for (int k = 0; k < Nt; k++)
        for (int j = 0; j < Nt; j++)
            for (int i = 0; i < Nt; i++) {
                int idx = UMG_IDX(lev, i, j, k);
                double x = UMG_COORD(lev, i);
                double y = UMG_COORD(lev, j);
                double z = UMG_COORD(lev, k);
                lev->psi_BL[idx] = brill_lindquist_psi(x, y, z, n_bh, bhs);
                double A_phys[3][3];
                bowen_york_Aij(A_phys, x, y, z, n_bh, bhs);
                lev->A2[idx] = bowen_york_A2(A_phys);
            }
}

/* Background precomputation for uniform levels — 4-field (HiSpID) */
static void umg_precompute_bg_4field(mg_level_amr_t *lev, int n_bh,
                                      const puncture_data_t *bhs)
{
    int Nt = lev->Ntotal;
    int gw = lev->ghost;
    double dx = lev->dx;
    double inv_dx = 1.0 / dx;
    size_t np = lev->npoints;
    static const int sym_map[3][3] = {{0,1,2},{1,3,4},{2,4,5}};

    double *h_bg[6], *A_bg[6];
    for (int c = 0; c < 6; c++) {
        h_bg[c] = calloc(np, sizeof(double));
        A_bg[c] = calloc(np, sizeof(double));
    }
    for (int k = 0; k < Nt; k++)
        for (int j = 0; j < Nt; j++)
            for (int i = 0; i < Nt; i++) {
                int idx = UMG_IDX(lev, i, j, k);
                double x = UMG_COORD(lev, i);
                double y = UMG_COORD(lev, j);
                double z = UMG_COORD(lev, k);
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

    /* Compute R_tilde via FD Christoffel/Ricci */
    int sx = 1, sy = lev->Ntotal, sz = lev->Ntotal * lev->Ntotal;
    int strides[3] = { sx, sy, sz };
    for (int k = gw; k < Nt - gw; k++)
        for (int j_idx = gw; j_idx < Nt - gw; j_idx++)
            for (int i = gw; i < Nt - gw; i++) {
                int idx = UMG_IDX(lev, i, j_idx, k);
                double h[3][3];
                h[0][0] = h_bg[0][idx]; h[0][1] = h_bg[1][idx]; h[0][2] = h_bg[2][idx];
                h[1][0] = h[0][1]; h[1][1] = h_bg[3][idx]; h[1][2] = h_bg[4][idx];
                h[2][0] = h[0][2]; h[2][1] = h[1][2]; h[2][2] = h_bg[5][idx];
                double h_UU[3][3];
                compute_inverse_sym(h, h_UU);
                double d1_h[3][3][3];
                for (int dir = 0; dir < 3; dir++)
                    for (int a = 0; a < 3; a++)
                        for (int b = a; b < 3; b++) {
                            double val = fd_d1(h_bg[sym_map[a][b]], idx,
                                               strides[dir], inv_dx);
                            d1_h[a][b][dir] = val;
                            d1_h[b][a][dir] = val;
                        }
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

    /* Compute S_M^i: momentum constraint source */
    for (int k = gw; k < Nt - gw; k++)
        for (int j_idx = gw; j_idx < Nt - gw; j_idx++)
            for (int i = gw; i < Nt - gw; i++) {
                int idx = UMG_IDX(lev, i, j_idx, k);
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
 * AMR block-level operations
 * ================================================================ */

/* Apply zero-Dirichlet BCs on domain-boundary blocks */
static void amr_apply_bc_block(block_t *blk, int four_field)
{
    grid_t *g = blk->grid;
    int gw = g->ghost, N = g->N, Nt = g->Ntotal;
    int n_sol = four_field ? 4 : 1;

    for (int face = 0; face < 6; face++) {
        if (!blk->on_boundary[face]) continue;
        for (int s = 0; s < n_sol; s++) {
            double *u = g->fields[s];
            for (int k = 0; k < Nt; k++)
                for (int j = 0; j < Nt; j++)
                    for (int i = 0; i < Nt; i++) {
                        int is_ghost = 0;
                        switch (face) {
                        case 0: is_ghost = (i < gw); break;
                        case 1: is_ghost = (i >= gw + N); break;
                        case 2: is_ghost = (j < gw); break;
                        case 3: is_ghost = (j >= gw + N); break;
                        case 4: is_ghost = (k < gw); break;
                        case 5: is_ghost = (k >= gw + N); break;
                        }
                        if (is_ghost) u[IDX(g, i, j, k)] = 0.0;
                    }
        }
    }
}

/* Precompute backgrounds on a single block — 1-field */
static void amr_precompute_bg_1field_block(block_t *blk, int n_bh,
                                            const puncture_data_t *bhs)
{
    grid_t *g = blk->grid;
    int Nt = g->Ntotal;
    for (int k = 0; k < Nt; k++)
        for (int j = 0; j < Nt; j++)
            for (int i = 0; i < Nt; i++) {
                int idx = IDX(g, i, j, k);
                double x = BLOCK_COORD(blk, 0, i);
                double y = BLOCK_COORD(blk, 1, j);
                double z = BLOCK_COORD(blk, 2, k);
                g->fields[BG_PSI_BL][idx] =
                    brill_lindquist_psi(x, y, z, n_bh, bhs);
                double A_phys[3][3];
                bowen_york_Aij(A_phys, x, y, z, n_bh, bhs);
                g->fields[BG_A2][idx] = bowen_york_A2(A_phys);
            }
}

/* Precompute backgrounds on a single block — 4-field */
static void amr_precompute_bg_4field_block(block_t *blk, int n_bh,
                                            const puncture_data_t *bhs)
{
    grid_t *g = blk->grid;
    int Nt = g->Ntotal;
    int gw = g->ghost;
    double inv_dx = g->inv_dx;
    size_t np = g->npoints;
    static const int s_map[3][3] = {{0,1,2},{1,3,4},{2,4,5}};

    double *h_bg[6], *A_bg[6];
    for (int c = 0; c < 6; c++) {
        h_bg[c] = calloc(np, sizeof(double));
        A_bg[c] = calloc(np, sizeof(double));
    }
    for (int k = 0; k < Nt; k++)
        for (int j = 0; j < Nt; j++)
            for (int i = 0; i < Nt; i++) {
                int idx = IDX(g, i, j, k);
                double x = BLOCK_COORD(blk, 0, i);
                double y = BLOCK_COORD(blk, 1, j);
                double z = BLOCK_COORD(blk, 2, k);
                g->fields[BG_PSI_BL][idx] =
                    brill_lindquist_psi(x, y, z, n_bh, bhs);
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
                g->fields[BG_A2][idx] = bowen_york_A2(A_total);
            }

    /* Compute R_tilde */
    int sx = 1, sy = g->Ntotal, sz = g->Ntotal * g->Ntotal;
    int strides[3] = { sx, sy, sz };
    for (int k = gw; k < Nt - gw; k++)
        for (int jj = gw; jj < Nt - gw; jj++)
            for (int i = gw; i < Nt - gw; i++) {
                int idx = IDX(g, i, jj, k);
                double h[3][3];
                h[0][0] = h_bg[0][idx]; h[0][1] = h_bg[1][idx]; h[0][2] = h_bg[2][idx];
                h[1][0] = h[0][1]; h[1][1] = h_bg[3][idx]; h[1][2] = h_bg[4][idx];
                h[2][0] = h[0][2]; h[2][1] = h[1][2]; h[2][2] = h_bg[5][idx];
                double h_UU[3][3];
                compute_inverse_sym(h, h_UU);
                double d1_h[3][3][3];
                for (int dir = 0; dir < 3; dir++)
                    for (int a = 0; a < 3; a++)
                        for (int b = a; b < 3; b++) {
                            double val = fd_d1(h_bg[s_map[a][b]], idx,
                                               strides[dir], inv_dx);
                            d1_h[a][b][dir] = val;
                            d1_h[b][a][dir] = val;
                        }
                double d2_h[3][3][3][3];
                memset(d2_h, 0, sizeof(d2_h));
                for (int dir = 0; dir < 3; dir++)
                    for (int a = 0; a < 3; a++)
                        for (int b = a; b < 3; b++) {
                            double val = fd_d2(h_bg[s_map[a][b]], idx,
                                               strides[dir], inv_dx);
                            d2_h[a][b][dir][dir] = val;
                            d2_h[b][a][dir][dir] = val;
                        }
                for (int d1 = 0; d1 < 3; d1++)
                    for (int d2 = 0; d2 < d1; d2++)
                        for (int a = 0; a < 3; a++)
                            for (int b = a; b < 3; b++) {
                                double val = fd_d2_mixed(
                                    h_bg[s_map[a][b]], idx,
                                    strides[d1], strides[d2], inv_dx);
                                d2_h[a][b][d1][d2] = val;
                                d2_h[a][b][d2][d1] = val;
                                d2_h[b][a][d1][d2] = val;
                                d2_h[b][a][d2][d1] = val;
                            }
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
                g->fields[BG_RTILDE][idx] = R_scalar;
            }

    /* S_M^i */
    for (int k = gw; k < Nt - gw; k++)
        for (int jj = gw; jj < Nt - gw; jj++)
            for (int i = gw; i < Nt - gw; i++) {
                int idx = IDX(g, i, jj, k);
                for (int d = 0; d < 3; d++) {
                    double div_A = 0.0;
                    for (int e = 0; e < 3; e++)
                        div_A += fd_d1(A_bg[s_map[d][e]], idx,
                                       strides[e], inv_dx);
                    g->fields[BG_SM1 + d][idx] = -div_A;
                }
            }

    for (int c = 0; c < 6; c++) {
        free(h_bg[c]);
        free(A_bg[c]);
    }
}

/* Newton-GS smoother on a single block — 1-field */
static void smooth_block_1field(block_t *blk, int color)
{
    grid_t *g = blk->grid;
    int gw = g->ghost, N = g->N, Nt = g->Ntotal;
    double dx = g->dx, dx2 = dx * dx;
    double inv_dx = g->inv_dx;
    int sx = 1, sy = Nt, sz = Nt * Nt;
    double J_lap = 3.0 * FD_D2_CENTER_WEIGHT / dx2;
    int c0 = color & 1, c1 = (color >> 1) & 1, c2 = (color >> 2) & 1;

    double *psi = g->fields[SOL_PSI];
    double *psi_BL = g->fields[BG_PSI_BL];
    double *A2 = g->fields[BG_A2];
    double *f_psi = g->rhs[SOL_PSI];

    for (int k = gw + c2; k < gw + N; k += 2)
        for (int j = gw + c1; j < gw + N; j += 2)
            for (int i = gw + c0; i < gw + N; i += 2) {
                int idx = k * sz + j * sy + i;
                double psi_tot = psi_BL[idx] + psi[idx];
                if (psi_tot < 0.1) psi_tot = 0.1;
                double lap = fd_d2(psi, idx, sx, inv_dx)
                           + fd_d2(psi, idx, sy, inv_dx)
                           + fd_d2(psi, idx, sz, inv_dx);
                double p2 = psi_tot * psi_tot;
                double p4 = p2 * p2;
                double p7 = p4 * p2 * psi_tot;
                double p8 = p7 * psi_tot;
                double source = 0.125 * A2[idx] / p7;
                double residual = lap + source - f_psi[idx];
                double dS = -0.875 * A2[idx] / p8;
                psi[idx] -= residual / (J_lap + dS);
            }
}

/* Newton-GS smoother on a single block — 4-field */
static void smooth_block_4field(block_t *blk, int color)
{
    grid_t *g = blk->grid;
    int gw = g->ghost, N = g->N, Nt = g->Ntotal;
    double dx = g->dx, dx2 = dx * dx;
    double inv_dx = g->inv_dx;
    int sx = 1, sy = Nt, sz = Nt * Nt;
    int strides[3] = { sx, sy, sz };
    double J_lap = 3.0 * FD_D2_CENTER_WEIGHT / dx2;
    double J_V_diag = J_lap + (1.0 / 3.0) * FD_D2_CENTER_WEIGHT / dx2;
    int c0 = color & 1, c1 = (color >> 1) & 1, c2 = (color >> 2) & 1;

    double *psi = g->fields[SOL_PSI];
    double *psi_BL = g->fields[BG_PSI_BL];
    double *A2 = g->fields[BG_A2];
    double *R_tilde = g->fields[BG_RTILDE];
    double *f_psi = g->rhs[SOL_PSI];
    double *V[3] = { g->fields[SOL_V1], g->fields[SOL_V2], g->fields[SOL_V3] };
    double *f_V[3] = { g->rhs[SOL_V1], g->rhs[SOL_V2], g->rhs[SOL_V3] };
    double *S_M[3] = { g->fields[BG_SM1], g->fields[BG_SM2], g->fields[BG_SM3] };

    for (int k = gw + c2; k < gw + N; k += 2)
        for (int j = gw + c1; j < gw + N; j += 2)
            for (int i = gw + c0; i < gw + N; i += 2) {
                int idx = k * sz + j * sy + i;
                double psi_tot = psi_BL[idx] + psi[idx];
                if (psi_tot < 0.1) psi_tot = 0.1;
                double lap_psi = fd_d2(psi, idx, sx, inv_dx)
                               + fd_d2(psi, idx, sy, inv_dx)
                               + fd_d2(psi, idx, sz, inv_dx);
                double p2 = psi_tot * psi_tot;
                double p4 = p2 * p2;
                double p7 = p4 * p2 * psi_tot;
                double p8 = p7 * psi_tot;
                double src_H = R_tilde[idx] * 0.125 * psi_tot
                             + A2[idx] * 0.125 / p7;
                double res_psi = lap_psi + src_H - f_psi[idx];
                double dS_psi = 0.125 * R_tilde[idx]
                              - 0.875 * A2[idx] / p8;
                psi[idx] -= res_psi / (J_lap + dS_psi);

                for (int d = 0; d < 3; d++) {
                    double lap_V = fd_d2(V[d], idx, sx, inv_dx)
                                 + fd_d2(V[d], idx, sy, inv_dx)
                                 + fd_d2(V[d], idx, sz, inv_dx);
                    double d_divV = 0.0;
                    for (int e = 0; e < 3; e++) {
                        if (e == d)
                            d_divV += fd_d2(V[e], idx, strides[e], inv_dx);
                        else
                            d_divV += fd_d2_mixed(V[e], idx,
                                                  strides[d], strides[e], inv_dx);
                    }
                    double res_V = lap_V + d_divV / 3.0
                                 + S_M[d][idx] - f_V[d][idx];
                    V[d][idx] -= res_V / J_V_diag;
                }
            }
}

/* Ghost exchange for solver fields.
 *
 * Uses ghost_exchange_all_blocks() which includes non-leaf blocks.
 * The composite V-cycle operates on ALL blocks at each level (not just
 * leaves), so non-leaf blocks need their ghost zones filled from
 * same-level neighbors.  Without this, multi-root meshes diverge because
 * non-leaf blocks' ghost zones contain stale data, corrupting the FD
 * operator evaluation and tau correction.
 *
 * We do NOT use ghost_exchange_multilevel() because Phase 4 (coarse→fine
 * prolongation) overwrites the solver's zero Dirichlet BCs with non-zero
 * interpolated values, preventing convergence.
 *
 * After exchange, re-apply solver BCs on all blocks to ensure
 * domain-boundary ghost zones are zero. */
static void solver_ghost_exchange(mesh_t *m, int four_field)
{
    ghost_exchange_all_blocks(m);

    /* Re-apply zero-Dirichlet BCs on domain-boundary blocks at all levels */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk) continue;
        amr_apply_bc_block(blk, four_field);
    }
}

/* Smooth all leaf blocks at a given AMR level */
static void smooth_level(mesh_t *m, int level, int four_field)
{
    for (int color = 0; color < 8; color++) {
        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < m->num_blocks; b++) {
            block_t *blk = m->blocks[b];
            if (!blk || blk->loc.level != level) continue;
            if (four_field)
                smooth_block_4field(blk, color);
            else
                smooth_block_1field(blk, color);
        }
        /* Ghost exchange after each color to propagate boundary values */
        solver_ghost_exchange(m, four_field);
    }
}

/* Compute L(u) on all blocks at a given level */
static void compute_operator_level(mesh_t *m, int level, int four_field)
{
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || blk->loc.level != level) continue;
        grid_t *g = blk->grid;
        int gw = g->ghost, N = g->N, Nt = g->Ntotal;
        double inv_dx = g->inv_dx;
        int sx = 1, sy = Nt, sz = Nt * Nt;
        int strides[3] = { sx, sy, sz };

        double *psi = g->fields[SOL_PSI];
        double *psi_BL = g->fields[BG_PSI_BL];
        double *A2 = g->fields[BG_A2];
        double *L_psi = g->accum[SOL_PSI];

        for (int k = gw; k < gw + N; k++)
            for (int j = gw; j < gw + N; j++)
                for (int i = gw; i < gw + N; i++) {
                    int idx = k * sz + j * sy + i;
                    double psi_tot = psi_BL[idx] + psi[idx];
                    if (psi_tot < 0.1) psi_tot = 0.1;
                    double lap = fd_d2(psi, idx, sx, inv_dx)
                               + fd_d2(psi, idx, sy, inv_dx)
                               + fd_d2(psi, idx, sz, inv_dx);
                    double p2 = psi_tot * psi_tot;
                    double p4 = p2 * p2;
                    double p7 = p4 * p2 * psi_tot;
                    if (four_field)
                        L_psi[idx] = lap
                            + g->fields[BG_RTILDE][idx] * 0.125 * psi_tot
                            + A2[idx] * 0.125 / p7;
                    else
                        L_psi[idx] = lap + 0.125 * A2[idx] / p7;

                    if (four_field) {
                        double *V[3] = { g->fields[SOL_V1],
                                         g->fields[SOL_V2],
                                         g->fields[SOL_V3] };
                        for (int d = 0; d < 3; d++) {
                            double lap_V = fd_d2(V[d], idx, sx, inv_dx)
                                         + fd_d2(V[d], idx, sy, inv_dx)
                                         + fd_d2(V[d], idx, sz, inv_dx);
                            double d_divV = 0.0;
                            for (int e = 0; e < 3; e++) {
                                if (e == d)
                                    d_divV += fd_d2(V[e], idx,
                                                    strides[e], inv_dx);
                                else
                                    d_divV += fd_d2_mixed(V[e], idx,
                                                          strides[d],
                                                          strides[e], inv_dx);
                            }
                            g->accum[SOL_V1 + d][idx] = lap_V + d_divV / 3.0
                                + g->fields[BG_SM1 + d][idx];
                        }
                    }
                }
    }
}

/* Compute residual r = f - L(u) on a level (stores in accum slots) */
static void compute_residual_level(mesh_t *m, int level, int four_field)
{
    int n_sol = four_field ? 4 : 1;
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || blk->loc.level != level) continue;
        grid_t *g = blk->grid;
        int gw = g->ghost, N = g->N, Nt = g->Ntotal;
        int sy = Nt, sz = Nt * Nt;
        for (int k = gw; k < gw + N; k++)
            for (int j = gw; j < gw + N; j++)
                for (int i = gw; i < gw + N; i++) {
                    int idx = k * sz + j * sy + i;
                    for (int s = 0; s < n_sol; s++)
                        g->accum[s][idx] = g->rhs[s][idx] - g->accum[s][idx];
                }
    }
}

/* Save solution on a level into scratch slots */
static void save_level_solution(mesh_t *m, int level, int four_field)
{
    int n_sol = four_field ? 4 : 1;
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || blk->loc.level != level) continue;
        grid_t *g = blk->grid;
        for (int s = 0; s < n_sol; s++)
            memcpy(g->scratch[s], g->fields[s], g->npoints * sizeof(double));
    }
}

/* Restrict solver fields from fine level blocks to coarse level blocks.
 * For each coarse block overlapping fine blocks at fine_level:
 *   - Restrict fine solution → coarse fields[0..3] (overwrites)
 *   - Restrict fine residual → coarse rhs[0..3] (overwrites)
 *
 * Uses simple 8-child volume average (cell-centered restriction).
 * Fine block covers one octant of coarse block's domain. */
static void restrict_to_coarser_amr(mesh_t *m, int fine_level, int four_field)
{
    int n_sol = four_field ? 4 : 1;

    /* For each coarse block at fine_level-1 that has fine children */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *cblk = m->blocks[b];
        if (!cblk || cblk->loc.level != fine_level - 1) continue;
        /* Check if this coarse block has fine children */
        int has_children = 0;
        for (int c = 0; c < 8; c++)
            if (cblk->child_ids[c] >= 0) { has_children = 1; break; }
        if (!has_children) continue;

        grid_t *cg = cblk->grid;
        int ghost = cg->ghost;
        int N = cg->N;
        int half_N = N / 2;

        /* Process each child octant */
        for (int cz = 0; cz < 2; cz++)
            for (int cy = 0; cy < 2; cy++)
                for (int cx = 0; cx < 2; cx++) {
                    int octant = cx + (cy << 1) + (cz << 2);
                    int child_id = cblk->child_ids[octant];
                    if (child_id < 0) continue;
                    block_t *fblk = m->blocks[child_id];
                    if (!fblk) continue;
                    grid_t *fg = fblk->grid;
                    int f_ghost = fg->ghost;

                    int p_off_i = cx * half_N;
                    int p_off_j = cy * half_N;
                    int p_off_k = cz * half_N;

                    /* Restrict solution and residual */
                    for (int s = 0; s < n_sol; s++) {
                        /* Restrict solution: fields[s] */
                        const double *f_sol = fg->fields[s];
                        /* Restrict residual: accum[s] */
                        const double *f_res = fg->accum[s];

                        for (int pk = 0; pk < half_N; pk++)
                            for (int pj = 0; pj < half_N; pj++)
                                for (int pi = 0; pi < half_N; pi++) {
                                    int fi = f_ghost + 2 * pi;
                                    int fj = f_ghost + 2 * pj;
                                    int fk = f_ghost + 2 * pk;

                                    /* 8-child average for solution */
                                    int f000 = IDX(fg, fi, fj, fk);
                                    int sxf = 1;
                                    int syf = fg->Ntotal;
                                    int szf = fg->Ntotal * fg->Ntotal;
                                    double sol_val = 0.125 * (
                                        f_sol[f000]             + f_sol[f000 + sxf]
                                      + f_sol[f000 + syf]       + f_sol[f000 + syf + sxf]
                                      + f_sol[f000 + szf]       + f_sol[f000 + szf + sxf]
                                      + f_sol[f000 + syf + szf] + f_sol[f000 + syf + szf + sxf]);

                                    /* 8-child average for residual */
                                    double res_val = 0.125 * (
                                        f_res[f000]             + f_res[f000 + sxf]
                                      + f_res[f000 + syf]       + f_res[f000 + syf + sxf]
                                      + f_res[f000 + szf]       + f_res[f000 + szf + sxf]
                                      + f_res[f000 + syf + szf] + f_res[f000 + syf + szf + sxf]);

                                    int ci = ghost + p_off_i + pi;
                                    int cj = ghost + p_off_j + pj;
                                    int ck = ghost + p_off_k + pk;
                                    cg->fields[s][IDX(cg, ci, cj, ck)] = sol_val;
                                    /* Store restricted residual in rhs */
                                    cg->rhs[s][IDX(cg, ci, cj, ck)] = res_val;
                                }
                    }
                }
    }
}

/* Apply tau correction: f_coarse = L(save) + Restrict(r)
 * After restriction, rhs has Restrict(r). We need:
 *   rhs[s] += L(save) = accum[s] (which contains L evaluated on the
 *   restricted solution that was copied into fields[s]). */
static void apply_tau_correction(mesh_t *m, int level, int four_field)
{
    int n_sol = four_field ? 4 : 1;
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || blk->loc.level != level) continue;
        grid_t *g = blk->grid;
        int gw = g->ghost, N = g->N, Nt = g->Ntotal;
        int sy = Nt, sz = Nt * Nt;
        for (int k = gw; k < gw + N; k++)
            for (int j = gw; j < gw + N; j++)
                for (int i = gw; i < gw + N; i++) {
                    int idx = k * sz + j * sy + i;
                    for (int s = 0; s < n_sol; s++)
                        g->rhs[s][idx] += g->accum[s][idx];
                }
    }
}

/* Prolongate correction from coarse level to fine level.
 * correction = coarse_solution - saved_solution (stored in scratch).
 * Uses trilinear interpolation and adds to fine. */
static void prolongate_correction_amr(mesh_t *m, int coarse_level, int four_field)
{
    int n_sol = four_field ? 4 : 1;

    for (int b = 0; b < m->num_blocks; b++) {
        block_t *cblk = m->blocks[b];
        if (!cblk || cblk->loc.level != coarse_level) continue;
        int has_children = 0;
        for (int c = 0; c < 8; c++)
            if (cblk->child_ids[c] >= 0) { has_children = 1; break; }
        if (!has_children) continue;

        grid_t *cg = cblk->grid;
        int ghost = cg->ghost;
        int N = cg->N;
        int half_N = N / 2;

        /* First compute correction on coarse: fields[s] - scratch[s] */
        for (int s = 0; s < n_sol; s++) {
            for (size_t idx = 0; idx < cg->npoints; idx++)
                cg->scratch[s][idx] = cg->fields[s][idx] - cg->scratch[s][idx];
        }

        /* Prolongate to each child and add */
        for (int cz = 0; cz < 2; cz++)
            for (int cy = 0; cy < 2; cy++)
                for (int cx = 0; cx < 2; cx++) {
                    int octant = cx + (cy << 1) + (cz << 2);
                    int child_id = cblk->child_ids[octant];
                    if (child_id < 0) continue;
                    block_t *fblk = m->blocks[child_id];
                    if (!fblk) continue;
                    grid_t *fg = fblk->grid;
                    int f_ghost = fg->ghost;
                    int f_N = fg->N;

                    int c_off_i = cx * half_N;
                    int c_off_j = cy * half_N;
                    int c_off_k = cz * half_N;

                    for (int s = 0; s < n_sol; s++) {
                        double *correction = cg->scratch[s];
                        double *fine_sol = fg->fields[s];

                        /* Trilinear prolongation of correction → fine */
                        for (int fk = 0; fk < f_N; fk++) {
                            /* Map fine cell to coarse */
                            int Kc = ghost + c_off_k + fk / 2;
                            int ok = fk % 2;
                            int dk = ok ? 1 : -1;

                            for (int fj = 0; fj < f_N; fj++) {
                                int Jc = ghost + c_off_j + fj / 2;
                                int oj = fj % 2;
                                int dj = oj ? 1 : -1;

                                for (int fi = 0; fi < f_N; fi++) {
                                    int Ic = ghost + c_off_i + fi / 2;
                                    int oi = fi % 2;
                                    int di = oi ? 1 : -1;

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
                                                    * correction[IDX(cg, CI, CJ, CK)];
                                            }
                                        }
                                    }
                                    fine_sol[IDX(fg, f_ghost + fi,
                                                 f_ghost + fj, f_ghost + fk)] += val;
                                }
                            }
                        }
                    }
                }
    }
}

/* ================================================================
 * Level-0 <-> Uniform MG transfer
 * ================================================================ */

/* Copy AMR level-0 block data into mg_levels[0] uniform grid */
static void copy_amr_level0_to_mg(mg_amr_t *mg)
{
    mg_level_amr_t *lev = &mg->mg_levels[0];
    int n_sol = mg->four_field ? 4 : 1;

    /* Find all level-0 blocks and copy their fields */
    for (int b = 0; b < mg->mesh->num_blocks; b++) {
        block_t *blk = mg->mesh->blocks[b];
        if (!blk || blk->loc.level != 0) continue;
        grid_t *g = blk->grid;
        int gw = g->ghost;
        int N = g->N;

        /* Compute offset: block origin maps to uniform grid coords */
        double half_L = lev->L * 0.5;
        int off_i = (int)round((blk->origin[0] + half_L) / lev->dx);
        int off_j = (int)round((blk->origin[1] + half_L) / lev->dx);
        int off_k = (int)round((blk->origin[2] + half_L) / lev->dx);

        /* Copy solution: fields[0..3] -> lev->psi, V[0..2] */
        double *dst_arr[4] = { lev->psi, lev->V[0], lev->V[1], lev->V[2] };
        for (int s = 0; s < n_sol; s++) {
            double *src = g->fields[s];
            double *dst = dst_arr[s];
            for (int k = 0; k < N; k++)
                for (int j = 0; j < N; j++)
                    for (int i = 0; i < N; i++) {
                        int src_idx = IDX(g, gw + i, gw + j, gw + k);
                        int dst_idx = UMG_IDX(lev,
                            lev->ghost + off_i + i,
                            lev->ghost + off_j + j,
                            lev->ghost + off_k + k);
                        dst[dst_idx] = src[src_idx];
                    }
        }

        /* Copy RHS (FAS target): rhs[0..3] -> lev->f_psi, f_V[0..2] */
        double *f_dst[4] = { lev->f_psi, lev->f_V[0], lev->f_V[1], lev->f_V[2] };
        for (int s = 0; s < n_sol; s++) {
            double *src = g->rhs[s];
            double *dst = f_dst[s];
            for (int k = 0; k < N; k++)
                for (int j = 0; j < N; j++)
                    for (int i = 0; i < N; i++) {
                        int src_idx = IDX(g, gw + i, gw + j, gw + k);
                        int dst_idx = UMG_IDX(lev,
                            lev->ghost + off_i + i,
                            lev->ghost + off_j + j,
                            lev->ghost + off_k + k);
                        dst[dst_idx] = src[src_idx];
                    }
        }
    }
}

/* Copy mg_levels[0] solution back to AMR level-0 blocks */
static void copy_mg_to_amr_level0(mg_amr_t *mg)
{
    mg_level_amr_t *lev = &mg->mg_levels[0];
    int n_sol = mg->four_field ? 4 : 1;

    for (int b = 0; b < mg->mesh->num_blocks; b++) {
        block_t *blk = mg->mesh->blocks[b];
        if (!blk || blk->loc.level != 0) continue;
        grid_t *g = blk->grid;
        int gw = g->ghost;
        int N = g->N;

        double half_L = lev->L * 0.5;
        int off_i = (int)round((blk->origin[0] + half_L) / lev->dx);
        int off_j = (int)round((blk->origin[1] + half_L) / lev->dx);
        int off_k = (int)round((blk->origin[2] + half_L) / lev->dx);

        double *src_arr[4] = { lev->psi, lev->V[0], lev->V[1], lev->V[2] };
        for (int s = 0; s < n_sol; s++) {
            double *src = src_arr[s];
            double *dst = g->fields[s];
            for (int k = 0; k < N; k++)
                for (int j = 0; j < N; j++)
                    for (int i = 0; i < N; i++) {
                        int src_idx = UMG_IDX(lev,
                            lev->ghost + off_i + i,
                            lev->ghost + off_j + j,
                            lev->ghost + off_k + k);
                        int dst_idx = IDX(g, gw + i, gw + j, gw + k);
                        dst[dst_idx] = src[src_idx];
                    }
        }
    }
}

/* Apply zero-Dirichlet BCs on all blocks at a given level.
 * Critical for non-leaf blocks whose ghost zones aren't touched by
 * ghost_exchange_multilevel (which only processes leaf blocks).
 * Ref: Afivo octree-mg (Teunissen 2018) — BCs applied before every
 * operator evaluation and smoothing sweep on each level. */
static void apply_bcs_level(mesh_t *m, int level, int four_field)
{
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || blk->loc.level != level) continue;
        amr_apply_bc_block(blk, four_field);
    }
}

/* ================================================================
 * Composite V-cycle
 *
 * Ref: Afivo/octree-mg (Teunissen, CPC 2018) — FAS composite V-cycle
 * Ref: AMReX MLMG — composite multigrid on block-structured AMR
 * ================================================================ */
static void composite_vcycle(mg_amr_t *mg, int amr_level)
{
    mesh_t *m = mg->mesh;

    /* Base case: AMR level 0 -> hand off to uniform multigrid */
    if (amr_level == 0) {
        apply_bcs_level(m, 0, mg->four_field);
        copy_amr_level0_to_mg(mg);
        umg_vcycle(mg->mg_levels, mg->n_mg_levels, 0, mg->four_field);
        copy_mg_to_amr_level0(mg);
        apply_bcs_level(m, 0, mg->four_field);
        return;
    }

    /* Pre-smooth */
    for (int s = 0; s < MG_NU_PRE; s++)
        smooth_level(m, amr_level, mg->four_field);

    /* Compute residual */
    solver_ghost_exchange(m, mg->four_field);
    compute_operator_level(m, amr_level, mg->four_field);
    compute_residual_level(m, amr_level, mg->four_field);

    /* Restrict fine solution + residual to coarser level.
     * restrict_to_coarser_amr OVERWRITES fields[s] and rhs[s] on coarse
     * blocks that have fine children.  Leaf blocks (no children at
     * amr_level) are NOT touched — their rhs[s] would accumulate stale
     * L(u) terms across V-cycles, causing divergence.
     *
     * Fix: zero rhs on leaf blocks at the target level before tau
     * correction.  After tau, they get rhs = 0 + L(u) = L(u), which
     * is the correct FAS target for blocks with no fine-level residual
     * (the coarse solve returns delta = 0, i.e., no change). */
    restrict_to_coarser_amr(m, amr_level, mg->four_field);

    /* Zero rhs on leaf blocks at amr_level-1 (no fine children).
     * Non-leaf blocks had rhs overwritten by restriction above. */
    {
        int n_sol = mg->four_field ? 4 : 1;
        for (int b = 0; b < m->num_blocks; b++) {
            block_t *blk = m->blocks[b];
            if (!blk || blk->loc.level != amr_level - 1) continue;
            /* Check if this block has children at amr_level */
            int has_children = 0;
            for (int c = 0; c < 8; c++)
                if (blk->child_ids[c] >= 0) { has_children = 1; break; }
            if (has_children) continue;  /* non-leaf: rhs set by restriction */
            /* Leaf: zero rhs to prevent stale accumulation */
            grid_t *g = blk->grid;
            for (int s = 0; s < n_sol; s++)
                memset(g->rhs[s], 0, g->npoints * sizeof(double));
        }
    }

    /* Save restricted solution (FAS: correction = u_new - u_restricted) */
    save_level_solution(m, amr_level - 1, mg->four_field);

    /* Apply BCs on coarse level before operator evaluation.
     * ghost_exchange_multilevel skips non-leaf blocks, so the coarse
     * block's ghost zones must be set explicitly. */
    apply_bcs_level(m, amr_level - 1, mg->four_field);

    /* Tau correction: compute L on restricted solution, add to rhs */
    solver_ghost_exchange(m, mg->four_field);
    compute_operator_level(m, amr_level - 1, mg->four_field);
    apply_tau_correction(m, amr_level - 1, mg->four_field);

    /* Recursive coarse solve */
    composite_vcycle(mg, amr_level - 1);

    /* Apply BCs on coarse level before prolongation (UMG solve only
     * updates interior; ghost zones need BCs for correct correction) */
    apply_bcs_level(m, amr_level - 1, mg->four_field);

    /* Prolongate correction back to fine */
    prolongate_correction_amr(m, amr_level - 1, mg->four_field);

    /* Ghost exchange after prolongation */
    solver_ghost_exchange(m, mg->four_field);

    /* Post-smooth */
    for (int s = 0; s < MG_NU_POST; s++)
        smooth_level(m, amr_level, mg->four_field);
}

/* ================================================================
 * Composite FMG
 * ================================================================ */
static void composite_fmg(mg_amr_t *mg)
{
    /* Step 1: Solve on coarsest (uniform hierarchy) */
    umg_fmg(mg->mg_levels, mg->n_mg_levels, mg->four_field);
    copy_mg_to_amr_level0(mg);
    apply_bcs_level(mg->mesh, 0, mg->four_field);

    /* Step 2: Ascend through AMR levels */
    for (int L = 1; L <= mg->mesh->max_level; L++) {
        /* Prolongate solution from level L-1 to level L.
         * For each fine block, interpolate from its parent. */
        int n_sol = mg->four_field ? 4 : 1;
        for (int b = 0; b < mg->mesh->num_blocks; b++) {
            block_t *fblk = mg->mesh->blocks[b];
            if (!fblk || fblk->loc.level != L) continue;
            if (fblk->parent_id < 0) continue;
            block_t *pblk = mg->mesh->blocks[fblk->parent_id];
            if (!pblk || !pblk->grid) continue;

            grid_t *fg = fblk->grid;
            grid_t *pg = pblk->grid;
            int f_ghost = fg->ghost;
            int f_N = fg->N;
            int p_ghost = pg->ghost;
            int p_N = pg->N;
            int half_N = p_N / 2;

            /* Determine which octant this child is */
            int cx = fblk->loc.lx1 % 2;
            int cy = fblk->loc.lx2 % 2;
            int cz = fblk->loc.lx3 % 2;
            int c_off_i = cx * half_N;
            int c_off_j = cy * half_N;
            int c_off_k = cz * half_N;

            for (int s = 0; s < n_sol; s++) {
                double *csol = pg->fields[s];
                double *fsol = fg->fields[s];

                /* Trilinear interpolation from coarse to fine (overwrite) */
                for (int fk = 0; fk < f_N; fk++) {
                    int Kc = p_ghost + c_off_k + fk / 2;
                    int ok = fk % 2;
                    int dk = ok ? 1 : -1;
                    for (int fj = 0; fj < f_N; fj++) {
                        int Jc = p_ghost + c_off_j + fj / 2;
                        int oj = fj % 2;
                        int dj = oj ? 1 : -1;
                        for (int fi = 0; fi < f_N; fi++) {
                            int Ic = p_ghost + c_off_i + fi / 2;
                            int oi = fi % 2;
                            int di = oi ? 1 : -1;
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
                                             * csol[IDX(pg, CI, CJ, CK)];
                                    }
                                }
                            }
                            fsol[IDX(fg, f_ghost + fi, f_ghost + fj,
                                      f_ghost + fk)] = val;
                        }
                    }
                }
            }
        }

        /* Ghost exchange + BCs */
        solver_ghost_exchange(mg->mesh, mg->four_field);

        /* One composite V-cycle to polish */
        composite_vcycle(mg, L);
    }
}

/* ================================================================
 * Residual norm on finest AMR level (L2)
 * ================================================================ */
static double amr_residual_norm(mg_amr_t *mg)
{
    mesh_t *m = mg->mesh;
    int finest = m->max_level;

    solver_ghost_exchange(m, mg->four_field);
    compute_operator_level(m, finest, mg->four_field);

    double sum_psi = 0.0, sum_V = 0.0;
    int count = 0;

    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || !blk->is_leaf || blk->loc.level != finest) continue;
        grid_t *g = blk->grid;
        int gw = g->ghost, N = g->N, Nt = g->Ntotal;
        int sy = Nt, sz = Nt * Nt;
        for (int k = gw; k < gw + N; k++)
            for (int j = gw; j < gw + N; j++)
                for (int i = gw; i < gw + N; i++) {
                    int idx = k * sz + j * sy + i;
                    double r = g->rhs[SOL_PSI][idx] - g->accum[SOL_PSI][idx];
                    sum_psi += r * r;
                    if (mg->four_field)
                        for (int d = 0; d < 3; d++) {
                            double rv = g->rhs[SOL_V1 + d][idx]
                                      - g->accum[SOL_V1 + d][idx];
                            sum_V += rv * rv;
                        }
                    count++;
                }
    }

    double l2_psi = (count > 0) ? sqrt(sum_psi / count) : 0.0;
    if (!mg->four_field) return l2_psi;
    double l2_V = (count > 0) ? sqrt(sum_V / (3 * count)) : 0.0;
    return (l2_psi > l2_V) ? l2_psi : l2_V;
}

/* ================================================================
 * Mesh refinement near punctures (shared by solver and evolution paths)
 * ================================================================ */

/*
 * Refine a mesh near punctures for n_amr_levels additional levels.
 * Works on any mesh (solver-owned or evolution mesh).
 * At each level, blocks within r_refine = 8*dx_level of any puncture
 * are refined. After each level, the mesh is compacted and neighbors
 * rebuilt.
 *
 * Ref: Athena++ MeshRefinement pattern.
 */
void refine_mesh_near_punctures(mesh_t *m, int n_amr_levels,
                                int n_bh, const puncture_data_t *bhs)
{
    int base_level = m->max_level;

    for (int level = base_level; level < base_level + n_amr_levels; level++) {
        double dx_level = m->L / (double)(m->N_block * m->N_root * (1 << level));
        double r_refine = 8.0 * dx_level;

        /* Find leaf blocks at current max_level that are near punctures */
        int n_to_refine = 0;
        int *refine_ids = calloc(m->num_blocks, sizeof(int));

        for (int b = 0; b < m->num_blocks; b++) {
            block_t *blk = m->blocks[b];
            if (!blk || !blk->is_leaf || blk->loc.level != level) continue;

            /* Check if block bounding box is near any puncture.
             * Use AABB-point distance (min distance from puncture to block
             * extent) instead of block-center distance. This handles large
             * blocks correctly — a puncture inside a block has dist = 0. */
            double block_dx = blk->grid->dx;
            int N_blk = blk->grid->N;
            double bx_min = blk->origin[0];
            double by_min = blk->origin[1];
            double bz_min = blk->origin[2];
            double bx_max = bx_min + N_blk * block_dx;
            double by_max = by_min + N_blk * block_dx;
            double bz_max = bz_min + N_blk * block_dx;

            for (int p = 0; p < n_bh; p++) {
                double cx = bhs[p].center[0];
                double cy = bhs[p].center[1];
                double cz = bhs[p].center[2];
                /* Clamp puncture position to block bounds */
                double nx = (cx < bx_min) ? bx_min : (cx > bx_max) ? bx_max : cx;
                double ny = (cy < by_min) ? by_min : (cy > by_max) ? by_max : cy;
                double nz = (cz < bz_min) ? bz_min : (cz > bz_max) ? bz_max : cz;
                double dist = sqrt((nx-cx)*(nx-cx) + (ny-cy)*(ny-cy) + (nz-cz)*(nz-cz));
                if (dist < r_refine) {
                    refine_ids[n_to_refine++] = blk->id;
                    break;
                }
            }
        }

        /* Refine selected blocks */
        for (int r = 0; r < n_to_refine; r++) {
            block_t *blk = m->blocks[refine_ids[r]];
            if (blk && blk->is_leaf)
                mesh_refine_block(m, refine_ids[r]);
        }

        free(refine_ids);

        /* Compact and rebuild after each level */
        mesh_compact(m);
        mesh_rebuild_neighbors(m);
    }
}

/*
 * Create AMR solver mesh with refinement near punctures.
 * Level 0: full-domain N_root=1 root block at base resolution.
 * Level 1+: refine blocks within r_refine of any puncture.
 */
static mesh_t *create_solver_mesh(int N_base, double L, int n_amr_levels,
                                   int n_bh, const puncture_data_t *bhs)
{
    /* For the solver mesh, we use N_root=1 (single root block) with
     * N_block = N_base.  This gives a single level-0 block covering
     * the full domain at the same resolution as the uniform solver. */
    mesh_t *m = mesh_create_ex(1, N_base, L, RK_CLASSIC, MG_AMR_N_FIELDS);

    refine_mesh_near_punctures(m, n_amr_levels, n_bh, bhs);

    return m;
}

/* ================================================================
 * Public API: 1-field AMR solver
 * ================================================================ */
double relaxation_solve_amr(grid_t *g, int n_bh, const puncture_data_t *bhs,
                             double tol, int max_iter, int verbose,
                             int n_amr_levels)
{
    /* Fall back to uniform solver if no AMR levels */
    if (n_amr_levels <= 0)
        return relaxation_solve(g, n_bh, bhs, tol, max_iter, verbose);

    if (verbose)
        printf("[AMR-MG] Starting AMR solver: N_base=%d, %d AMR levels, tol=%.2e\n",
               g->N, n_amr_levels, tol);

    /* Create solver mesh */
    mesh_t *m = create_solver_mesh(g->N, g->L, n_amr_levels, n_bh, bhs);
    if (verbose)
        printf("[AMR-MG] Solver mesh: %d total blocks, %d leaves, max_level=%d\n",
               m->num_blocks, mesh_num_leaves(m), m->max_level);

    /* Build mg_amr_t */
    mg_amr_t mg;
    memset(&mg, 0, sizeof(mg));
    mg.mesh = m;
    mg.n_amr_levels = n_amr_levels;
    mg.four_field = 0;
    mg.n_bh = n_bh;
    mg.bhs = bhs;

    /* Build uniform MG hierarchy below level 0 */
    mg.n_mg_levels = 1;
    {
        int N = g->N;
        while (N / (1 << mg.n_mg_levels) >= MG_N_MIN
               && mg.n_mg_levels < MG_MAX_LEVELS)
            mg.n_mg_levels++;
    }
    mg.mg_levels = calloc(mg.n_mg_levels, sizeof(mg_level_amr_t));
    for (int l = 0; l < mg.n_mg_levels; l++) {
        int N_l = g->N / (1 << l);
        mg_level_amr_init(&mg.mg_levels[l], N_l, g->L, 0);
        umg_precompute_bg_1field(&mg.mg_levels[l], n_bh, bhs);
    }

    /* Precompute backgrounds on ALL blocks (including non-leaf coarse
     * blocks, since the composite V-cycle operates on all levels) */
    if (verbose) printf("[AMR-MG] Precomputing backgrounds...\n");
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk) continue;
        amr_precompute_bg_1field_block(blk, n_bh, bhs);
    }

    /* Zero solver solution and RHS on all blocks.
     * posix_memalign does not zero memory, so rhs contains garbage.
     * The finest-level equation is L(u) = 0, so rhs must be 0. */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk) continue;
        memset(blk->grid->fields[SOL_PSI], 0,
               blk->grid->npoints * sizeof(double));
        memset(blk->grid->rhs[SOL_PSI], 0,
               blk->grid->npoints * sizeof(double));
    }

    /* Ghost exchange so backgrounds are in ghost zones too */
    solver_ghost_exchange(m, 0);

    /* Run composite FMG */
    if (verbose) printf("[AMR-MG] Running composite FMG...\n");
    composite_fmg(&mg);

    double residual = amr_residual_norm(&mg);
    if (verbose) printf("[AMR-MG] FMG done: residual = %.6e\n", residual);

    /* Post-FMG V-cycles */
    for (int cycle = 0; cycle < max_iter && residual > tol; cycle++) {
        composite_vcycle(&mg, m->max_level);
        residual = amr_residual_norm(&mg);
        if (verbose)
            printf("[AMR-MG] V-cycle %d: residual = %.6e\n",
                   cycle + 1, residual);
    }

    /* Transfer AMR solution to the target grid.
     *
     * The output grid has level-0 resolution (uniform).  We copy only
     * level-0 data, which was updated by FAS V-cycle corrections from
     * fine levels and is consistent with the coarse FD stencils.
     *
     * NOTE: Restricting fine-level data to the coarse output grid creates
     * artifacts at refinement boundaries (discontinuities in FD stencils),
     * worsening apparent constraint quality.  For AMR evolution meshes,
     * each block should receive data directly from its corresponding
     * solver level — that path is in set_bowen_york_mesh() (TODO). */
    double *psi_full = calloc(g->npoints, sizeof(double));

    /* Copy level-0 data (fills everything including ghost zones) */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || blk->loc.level != 0 || !blk->grid) continue;
        grid_t *bg = blk->grid;

        double half_L = g->L * 0.5;
        int off_i = (int)round((blk->origin[0] + half_L) / g->dx);
        int off_j = (int)round((blk->origin[1] + half_L) / g->dx);
        int off_k = (int)round((blk->origin[2] + half_L) / g->dx);

        int Nt_blk = bg->Ntotal;
        for (int k = 0; k < Nt_blk; k++)
            for (int j = 0; j < Nt_blk; j++)
                for (int i = 0; i < Nt_blk; i++) {
                    int src_idx = IDX(bg, i, j, k);
                    int dst_i = off_i + i;
                    int dst_j = off_j + j;
                    int dst_k = off_k + k;
                    if (dst_i >= 0 && dst_i < g->Ntotal &&
                        dst_j >= 0 && dst_j < g->Ntotal &&
                        dst_k >= 0 && dst_k < g->Ntotal) {
                        int dst_idx = IDX(g, dst_i, dst_j, dst_k);
                        psi_full[dst_idx] = bg->fields[BG_PSI_BL][src_idx]
                                          + bg->fields[SOL_PSI][src_idx];
                    }
                }
    }

    /* Set CCZ4 fields */
    set_ccz4_from_psi(g, psi_full, n_bh, bhs);

    /* Cleanup */
    free(psi_full);
    for (int l = 0; l < mg.n_mg_levels; l++)
        mg_level_amr_free(&mg.mg_levels[l]);
    free(mg.mg_levels);
    mesh_free(m);

    return residual;
}

/* ================================================================
 * Public API: 4-field AMR solver (HiSpID coupled)
 * ================================================================ */
double relaxation_solve_coupled_amr(grid_t *g, int n_bh,
                                     const puncture_data_t *bhs,
                                     double tol, int max_iter, int verbose,
                                     int n_amr_levels)
{
    if (n_amr_levels <= 0)
        return relaxation_solve_coupled(g, n_bh, bhs, tol, max_iter, verbose);

    if (verbose)
        printf("[AMR-MG] Starting coupled AMR solver: N_base=%d, %d AMR levels\n",
               g->N, n_amr_levels);

    mesh_t *m = create_solver_mesh(g->N, g->L, n_amr_levels, n_bh, bhs);
    if (verbose)
        printf("[AMR-MG] Solver mesh: %d total blocks, %d leaves, max_level=%d\n",
               m->num_blocks, mesh_num_leaves(m), m->max_level);

    mg_amr_t mg;
    memset(&mg, 0, sizeof(mg));
    mg.mesh = m;
    mg.n_amr_levels = n_amr_levels;
    mg.four_field = 1;
    mg.n_bh = n_bh;
    mg.bhs = bhs;

    mg.n_mg_levels = 1;
    {
        int N = g->N;
        while (N / (1 << mg.n_mg_levels) >= MG_N_MIN
               && mg.n_mg_levels < MG_MAX_LEVELS)
            mg.n_mg_levels++;
    }
    mg.mg_levels = calloc(mg.n_mg_levels, sizeof(mg_level_amr_t));
    for (int l = 0; l < mg.n_mg_levels; l++) {
        int N_l = g->N / (1 << l);
        mg_level_amr_init(&mg.mg_levels[l], N_l, g->L, 1);
    }

    if (verbose)
        printf("[AMR-MG] Precomputing 4-field backgrounds...\n");
    for (int l = 0; l < mg.n_mg_levels; l++)
        umg_precompute_bg_4field(&mg.mg_levels[l], n_bh, bhs);

    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk) continue;
        amr_precompute_bg_4field_block(blk, n_bh, bhs);
    }

    /* Zero solver solution and RHS on all blocks.
     * posix_memalign does not zero memory, so rhs contains garbage.
     * The finest-level equation is L(u) = 0, so rhs must be 0. */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk) continue;
        grid_t *bg = blk->grid;
        memset(bg->fields[SOL_PSI], 0, bg->npoints * sizeof(double));
        memset(bg->fields[SOL_V1], 0, bg->npoints * sizeof(double));
        memset(bg->fields[SOL_V2], 0, bg->npoints * sizeof(double));
        memset(bg->fields[SOL_V3], 0, bg->npoints * sizeof(double));
        memset(bg->rhs[SOL_PSI], 0, bg->npoints * sizeof(double));
        memset(bg->rhs[SOL_V1], 0, bg->npoints * sizeof(double));
        memset(bg->rhs[SOL_V2], 0, bg->npoints * sizeof(double));
        memset(bg->rhs[SOL_V3], 0, bg->npoints * sizeof(double));
    }

    solver_ghost_exchange(m, 1);

    if (verbose) printf("[AMR-MG] Running composite FMG (4-field)...\n");
    composite_fmg(&mg);

    double residual = amr_residual_norm(&mg);
    if (verbose) printf("[AMR-MG] FMG done: residual = %.6e\n", residual);

    for (int cycle = 0; cycle < max_iter && residual > tol; cycle++) {
        composite_vcycle(&mg, m->max_level);
        residual = amr_residual_norm(&mg);
        if (verbose)
            printf("[AMR-MG] V-cycle %d: residual = %.6e\n",
                   cycle + 1, residual);
    }

    /* Transfer solution: pass 1 (level 0) + pass 2 (fine levels) */
    double *psi_full = calloc(g->npoints, sizeof(double));
    double *V_full[3];
    for (int d = 0; d < 3; d++)
        V_full[d] = calloc(g->npoints, sizeof(double));

    /* Pass 1: level-0 data fills everything */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || blk->loc.level != 0 || !blk->grid) continue;
        grid_t *bg = blk->grid;

        double half_L = g->L * 0.5;
        int off_i = (int)round((blk->origin[0] + half_L) / g->dx);
        int off_j = (int)round((blk->origin[1] + half_L) / g->dx);
        int off_k = (int)round((blk->origin[2] + half_L) / g->dx);

        int Nt_blk = bg->Ntotal;
        for (int k = 0; k < Nt_blk; k++)
            for (int j = 0; j < Nt_blk; j++)
                for (int i = 0; i < Nt_blk; i++) {
                    int src_idx = IDX(bg, i, j, k);
                    int dst_i = off_i + i;
                    int dst_j = off_j + j;
                    int dst_k = off_k + k;
                    if (dst_i >= 0 && dst_i < g->Ntotal &&
                        dst_j >= 0 && dst_j < g->Ntotal &&
                        dst_k >= 0 && dst_k < g->Ntotal) {
                        int dst_idx = IDX(g, dst_i, dst_j, dst_k);
                        psi_full[dst_idx] = bg->fields[BG_PSI_BL][src_idx]
                                          + bg->fields[SOL_PSI][src_idx];
                        for (int d = 0; d < 3; d++)
                            V_full[d][dst_idx] =
                                bg->fields[SOL_V1 + d][src_idx];
                    }
                }
    }

    /* Pass 2: overwrite with restricted data from finer leaf blocks */
    for (int level = 1; level <= m->max_level; level++) {
        int ratio = 1 << level;
        double inv_vol = 1.0 / (double)(ratio * ratio * ratio);

        for (int b = 0; b < m->num_blocks; b++) {
            block_t *blk = m->blocks[b];
            if (!blk || !blk->is_leaf || blk->loc.level != level) continue;
            if (!blk->grid) continue;
            grid_t *fg = blk->grid;
            int f_gw = fg->ghost;
            int f_N = fg->N;

            double half_L = g->L * 0.5;
            int base_i = (int)round((blk->origin[0] + half_L) / g->dx);
            int base_j = (int)round((blk->origin[1] + half_L) / g->dx);
            int base_k = (int)round((blk->origin[2] + half_L) / g->dx);

            int n_out = f_N / ratio;

            for (int ok = 0; ok < n_out; ok++)
                for (int oj = 0; oj < n_out; oj++)
                    for (int oi = 0; oi < n_out; oi++) {
                        double sum_psi = 0.0;
                        double sum_V[3] = {0.0, 0.0, 0.0};
                        for (int dk = 0; dk < ratio; dk++)
                            for (int dj = 0; dj < ratio; dj++)
                                for (int di = 0; di < ratio; di++) {
                                    int fi = f_gw + oi * ratio + di;
                                    int fj = f_gw + oj * ratio + dj;
                                    int fk = f_gw + ok * ratio + dk;
                                    int fidx = IDX(fg, fi, fj, fk);
                                    sum_psi += fg->fields[BG_PSI_BL][fidx]
                                             + fg->fields[SOL_PSI][fidx];
                                    for (int d = 0; d < 3; d++)
                                        sum_V[d] +=
                                            fg->fields[SOL_V1 + d][fidx];
                                }

                        int dst_i = g->ghost + base_i + oi;
                        int dst_j = g->ghost + base_j + oj;
                        int dst_k = g->ghost + base_k + ok;

                        if (dst_i >= 0 && dst_i < g->Ntotal &&
                            dst_j >= 0 && dst_j < g->Ntotal &&
                            dst_k >= 0 && dst_k < g->Ntotal) {
                            int dst_idx = IDX(g, dst_i, dst_j, dst_k);
                            psi_full[dst_idx] = sum_psi * inv_vol;
                            for (int d = 0; d < 3; d++)
                                V_full[d][dst_idx] = sum_V[d] * inv_vol;
                        }
                    }
        }
    }

    set_ccz4_from_hispid(g, psi_full, V_full, n_bh, bhs);

    free(psi_full);
    for (int d = 0; d < 3; d++)
        free(V_full[d]);
    for (int l = 0; l < mg.n_mg_levels; l++)
        mg_level_amr_free(&mg.mg_levels[l]);
    free(mg.mg_levels);
    mesh_free(m);

    return residual;
}

/* ================================================================
 * Public API: 1-field solver on external (evolution) mesh
 *
 * Solves the Hamiltonian constraint directly on the caller's mesh.
 * The mesh blocks' fields/rhs/scratch/accum arrays are used as
 * solver workspace — at t=0 these are idle.
 *
 * Ref: Tomida & Stone 2023 (Athena++ MG self-gravity on evolution mesh)
 * ================================================================ */
double relaxation_solve_amr_mesh(mesh_t *m, int n_bh, const puncture_data_t *bhs,
                                  double tol, int max_iter, int verbose,
                                  int n_amr_levels)
{
    if (verbose)
        printf("[AMR-MG-mesh] Starting 1-field solver on evolution mesh, "
               "%d AMR levels, tol=%.2e\n", n_amr_levels, tol);

    /* Verify accum arrays are present (requires RK_CLASSIC) */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || !blk->is_leaf) continue;
        if (!blk->grid->accum_block) {
            fprintf(stderr, "[AMR-MG-mesh] ERROR: accum arrays not allocated. "
                    "Use RK_CLASSIC for evolution mesh solver.\n");
            return -1.0;
        }
        break;
    }

    /* Refine mesh near punctures */
    if (n_amr_levels > 0)
        refine_mesh_near_punctures(m, n_amr_levels, n_bh, bhs);

    if (verbose)
        printf("[AMR-MG-mesh] Mesh: %d blocks, %d leaves, max_level=%d\n",
               m->num_blocks, mesh_num_leaves(m), m->max_level);

    /* Build mg_amr_t (borrows mesh, does not own it) */
    mg_amr_t mg;
    memset(&mg, 0, sizeof(mg));
    mg.mesh = m;
    mg.n_amr_levels = n_amr_levels;
    mg.four_field = 0;
    mg.n_bh = n_bh;
    mg.bhs = bhs;

    /* Build uniform MG hierarchy below level 0 */
    int N_eff = m->N_root * m->N_block;
    mg.n_mg_levels = 1;
    {
        int N = N_eff;
        while (N / (1 << mg.n_mg_levels) >= MG_N_MIN
               && mg.n_mg_levels < MG_MAX_LEVELS)
            mg.n_mg_levels++;
    }
    mg.mg_levels = calloc(mg.n_mg_levels, sizeof(mg_level_amr_t));
    for (int l = 0; l < mg.n_mg_levels; l++) {
        int N_l = N_eff / (1 << l);
        mg_level_amr_init(&mg.mg_levels[l], N_l, m->L, 0);
        umg_precompute_bg_1field(&mg.mg_levels[l], n_bh, bhs);
    }

    /* Precompute backgrounds on ALL blocks */
    if (verbose) printf("[AMR-MG-mesh] Precomputing backgrounds...\n");
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk) continue;
        amr_precompute_bg_1field_block(blk, n_bh, bhs);
    }

    /* Zero solver solution and RHS on all blocks.
     * posix_memalign does not zero memory, so rhs contains garbage.
     * The finest-level equation is L(u) = 0, so rhs must be 0.
     * Coarser-level rhs is set by tau correction during V-cycles. */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk) continue;
        memset(blk->grid->fields[SOL_PSI], 0,
               blk->grid->npoints * sizeof(double));
        memset(blk->grid->rhs[SOL_PSI], 0,
               blk->grid->npoints * sizeof(double));
    }

    solver_ghost_exchange(m, 0);

    /* Run composite FMG */
    if (verbose) printf("[AMR-MG-mesh] Running composite FMG...\n");
    composite_fmg(&mg);

    double residual = amr_residual_norm(&mg);
    if (verbose) printf("[AMR-MG-mesh] FMG done: residual = %.6e\n", residual);

    /* Post-FMG V-cycles */
    for (int cycle = 0; cycle < max_iter && residual > tol; cycle++) {
        composite_vcycle(&mg, m->max_level);
        residual = amr_residual_norm(&mg);
        if (verbose)
            printf("[AMR-MG-mesh] V-cycle %d: residual = %.6e\n",
                   cycle + 1, residual);
    }

    /* Free MG hierarchy only — mesh is caller-owned */
    for (int l = 0; l < mg.n_mg_levels; l++)
        mg_level_amr_free(&mg.mg_levels[l]);
    free(mg.mg_levels);

    return residual;
}

/* ================================================================
 * Public API: 4-field coupled solver on external (evolution) mesh
 * ================================================================ */
double relaxation_solve_coupled_amr_mesh(mesh_t *m, int n_bh,
                                          const puncture_data_t *bhs,
                                          double tol, int max_iter, int verbose,
                                          int n_amr_levels)
{
    if (verbose)
        printf("[AMR-MG-mesh] Starting 4-field solver on evolution mesh, "
               "%d AMR levels, tol=%.2e\n", n_amr_levels, tol);

    /* Verify accum arrays are present */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || !blk->is_leaf) continue;
        if (!blk->grid->accum_block) {
            fprintf(stderr, "[AMR-MG-mesh] ERROR: accum arrays not allocated. "
                    "Use RK_CLASSIC for evolution mesh solver.\n");
            return -1.0;
        }
        break;
    }

    /* Refine mesh near punctures */
    if (n_amr_levels > 0)
        refine_mesh_near_punctures(m, n_amr_levels, n_bh, bhs);

    if (verbose)
        printf("[AMR-MG-mesh] Mesh: %d blocks, %d leaves, max_level=%d\n",
               m->num_blocks, mesh_num_leaves(m), m->max_level);

    mg_amr_t mg;
    memset(&mg, 0, sizeof(mg));
    mg.mesh = m;
    mg.n_amr_levels = n_amr_levels;
    mg.four_field = 1;
    mg.n_bh = n_bh;
    mg.bhs = bhs;

    int N_eff = m->N_root * m->N_block;
    mg.n_mg_levels = 1;
    {
        int N = N_eff;
        while (N / (1 << mg.n_mg_levels) >= MG_N_MIN
               && mg.n_mg_levels < MG_MAX_LEVELS)
            mg.n_mg_levels++;
    }
    mg.mg_levels = calloc(mg.n_mg_levels, sizeof(mg_level_amr_t));
    for (int l = 0; l < mg.n_mg_levels; l++) {
        int N_l = N_eff / (1 << l);
        mg_level_amr_init(&mg.mg_levels[l], N_l, m->L, 1);
    }

    if (verbose)
        printf("[AMR-MG-mesh] Precomputing 4-field backgrounds...\n");
    for (int l = 0; l < mg.n_mg_levels; l++)
        umg_precompute_bg_4field(&mg.mg_levels[l], n_bh, bhs);

    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk) continue;
        amr_precompute_bg_4field_block(blk, n_bh, bhs);
    }

    /* Zero solver solution and RHS on all blocks.
     * posix_memalign does not zero memory, so rhs contains garbage.
     * The finest-level equation is L(u) = 0, so rhs must be 0. */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk) continue;
        grid_t *g = blk->grid;
        memset(g->fields[SOL_PSI], 0, g->npoints * sizeof(double));
        memset(g->fields[SOL_V1], 0, g->npoints * sizeof(double));
        memset(g->fields[SOL_V2], 0, g->npoints * sizeof(double));
        memset(g->fields[SOL_V3], 0, g->npoints * sizeof(double));
        memset(g->rhs[SOL_PSI], 0, g->npoints * sizeof(double));
        memset(g->rhs[SOL_V1], 0, g->npoints * sizeof(double));
        memset(g->rhs[SOL_V2], 0, g->npoints * sizeof(double));
        memset(g->rhs[SOL_V3], 0, g->npoints * sizeof(double));
    }

    solver_ghost_exchange(m, 1);

    if (verbose) printf("[AMR-MG-mesh] Running composite FMG (4-field)...\n");
    composite_fmg(&mg);

    double residual = amr_residual_norm(&mg);
    if (verbose) printf("[AMR-MG-mesh] FMG done: residual = %.6e\n", residual);

    for (int cycle = 0; cycle < max_iter && residual > tol; cycle++) {
        composite_vcycle(&mg, m->max_level);
        residual = amr_residual_norm(&mg);
        if (verbose)
            printf("[AMR-MG-mesh] V-cycle %d: residual = %.6e\n",
                   cycle + 1, residual);
    }

    /* Free MG hierarchy only — mesh is caller-owned */
    for (int l = 0; l < mg.n_mg_levels; l++)
        mg_level_amr_free(&mg.mg_levels[l]);
    free(mg.mg_levels);

    return residual;
}
