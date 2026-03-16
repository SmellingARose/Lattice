/*
 * Lattice — 3D Numerical Relativity
 * FAS Multigrid constraint solver (covering grid, level-by-level on AMR mesh).
 *
 * Level-by-level solve: each AMR level is solved independently from
 * coarse to fine.  Converged coarser levels provide fixed Dirichlet BCs,
 * eliminating all cross-level coupling.
 *
 * Architecture (covering grid FAS):
 *   For level L = 0 to max_level:
 *     1. Fill CF ghost zones from level L-1 (already converged)
 *     2. Gather all blocks at level L into a single temporary uniform
 *        grid (the "covering grid")
 *     3. Run proven single-grid FAS on the covering grid:
 *        a. FMG: coarsest→finest with V-cycle polish at each level
 *        b. Post-FMG V-cycles until convergence
 *        c. 8-color Newton-GS smoother
 *        d. No inter-block ghost exchange during MG (one contiguous grid)
 *     4. Scatter solution back to blocks
 *     5. Same-level ghost exchange on converged solution
 *
 * This replaces the old composite FAS multigrid and JFNK+BiCGSTAB approaches.
 * FMG converges in 1 pass per level. The file retains the jfnk_solver name
 * for API compatibility.
 *
 * Ref: arXiv:0705.1486 (Natchu & Matzner, 4th-order MG for BH data)
 * Ref: arXiv:2510.11152 (GPU FAS multigrid, 8-color MCGS, 61x speedup)
 */

#include "jfnk_solver.h"
#include "bowen_york.h"
#include "kerr_quasi_isotropic.h"
#include "../numerics/finite_diff.h"
#include "../geometry/tensor_utils.h"
#include "../amr/mesh.h"
#include "../amr/block.h"
#include "../amr/refine.h"
#include "../amr/ghost_exchange.h"
#include "../amr/prolongation.h"
#include "../amr/restriction.h"
#include "../amr/meshblock_pack.h"
#include "../backend/backend.h"
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

#define JFNK_N_FIELDS 10  /* total solver fields per block */

/* FD_D2 center weight for Jacobian diagonal */
#if FD_ORDER == 6
#define FD_D2_CENTER_WEIGHT (-49.0 / 18.0)
#else
#define FD_D2_CENTER_WEIGHT (-5.0 / 2.0)
#endif

/* ================================================================
 * Multigrid parameters
 * Ref: arXiv:0705.1486 Section 3.2
 * ================================================================ */
#define MG_N_MIN      4   /* coarsest MG grid interior points per side */
#define MG_NU_PRE     4   /* pre-smoothing Newton-GS sweeps */
#define MG_NU_POST    4   /* post-smoothing Newton-GS sweeps */
#define MG_NU_COARSE 50   /* coarsest-level sweeps */
#define MG_MAX_LEVELS 8   /* max within-block hierarchy depth */

/* ================================================================
 * Forward declarations
 * ================================================================ */
static void apply_bcs_level(mesh_t *m, int level, int four_field);

/* ================================================================
 * Ghost exchange wrappers for solver fields
 * ================================================================ */

/* Full ghost exchange at all levels */
static void solver_ghost_exchange_all(mesh_t *m, int four_field)
{
    for (int L = 0; L <= m->max_level; L++) {
        ghost_exchange_same_level_all(m, L);
        ghost_fill_cf_boundary(m, L);
        apply_bcs_level(m, L, four_field);
    }
}

/* Same-level ghost exchange + BCs for a single level */
static void solver_same_level_exchange(mesh_t *m, int level, int four_field)
{
    ghost_exchange_same_level_all(m, level);
    apply_bcs_level(m, level, four_field);
}

/* Full exchange at a single level (same-level + CF boundary) */
static void solver_full_exchange(mesh_t *m, int level, int four_field)
{
    ghost_exchange_same_level_all(m, level);
    ghost_fill_cf_boundary(m, level);
    apply_bcs_level(m, level, four_field);
}

/* ================================================================
 * Apply zero-Dirichlet BCs on domain-boundary blocks
 * ================================================================ */
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

static void apply_bcs_level(mesh_t *m, int level, int four_field)
{
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || blk->loc.level != level) continue;
        amr_apply_bc_block(blk, four_field);
    }
}

/* ================================================================
 * Background precomputation — 1-field (BY Hamiltonian)
 * ================================================================ */
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
                double A_tilde[3][3];
                bowen_york_Aij(A_tilde, x, y, z, n_bh, bhs);
                g->fields[BG_A2][idx] = bowen_york_A2(A_tilde);
            }
}

/* ================================================================
 * Background precomputation — 4-field (HiSpID coupled)
 * ================================================================ */
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
                g->fields[BG_A2][idx] = hispid_A2(A_total, h);
            }

    /* Compute R_tilde via FD Christoffel/Ricci */
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

    /* S_M^i: momentum constraint source */
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

/* ================================================================
 * Operator evaluation: compute L(u) on all blocks at a level.
 * Includes non-leaf blocks so they provide valid CF boundary data
 * for finer levels.  Result stored in accum[0..3].
 * ================================================================ */
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

/* ================================================================
 * Per-block within-block multigrid level data
 *
 * Each block (N=32 with ghost=4, Ntotal=40) coarsens internally:
 *   N=32 → N=16 → N=8 → N=4
 * The coarsened grids are temporary arrays. At the finest MG level
 * we work directly on the block's fields arrays.
 *
 * Ref: arXiv:0705.1486 Section 2.2 (FAS V-cycle)
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
    double inv_dx;       /* 1.0 / dx */
    size_t npoints;      /* Ntotal^3 */
} mg_level_t;

/* Indexing for multigrid level */
#define MG_IDX(lev, i, j, k) \
    ((k) * (lev)->Ntotal * (lev)->Ntotal + (j) * (lev)->Ntotal + (i))

/* ================================================================
 * MG level allocation / deallocation
 * ================================================================ */
static void mg_level_init(mg_level_t *lev, int N, double dx, int four_field)
{
    lev->N      = N;
    lev->ghost  = GHOST_WIDTH;
    lev->Ntotal = N + 2 * GHOST_WIDTH;
    lev->dx     = dx;
    lev->inv_dx = 1.0 / dx;
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
 * Boundary conditions: zero-Dirichlet in ghost zones (MG levels)
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
 * Restriction: cell-centered 8-child volume average (fine -> coarse)
 *
 * Ref: HPGMG finite-volume restriction (cell-centered)
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
 * Operator evaluation on mg_level_t: compute L(u) at all interior points
 * ================================================================ */
static void mg_compute_operator(mg_level_t *lev, int four_field)
{
    int gw = lev->ghost;
    int N  = lev->N;
    int Nt = lev->Ntotal;
    double inv_dx = lev->inv_dx;
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
 * Single-grid Newton-GS smoother — 1-field
 * 8-color ordering. Ref: arXiv:2510.11152 Section 3.1
 * ================================================================ */
static void mg_gs_sweep_1field(mg_level_t *lev)
{
    int gw = lev->ghost, N = lev->N, Nt = lev->Ntotal;
    double inv_dx = lev->inv_dx, dx2 = lev->dx * lev->dx;
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
                    double p2 = psi_tot * psi_tot, p4 = p2 * p2;
                    double p7 = p4 * p2 * psi_tot, p8 = p7 * psi_tot;
                    double res = lap + 0.125 * lev->A2[idx] / p7 - lev->f_psi[idx];
                    lev->psi[idx] -= res / (J_lap - 0.875 * lev->A2[idx] / p8);
                }
    }
    mg_apply_bc_field(lev->psi, lev);
}

/* Single-grid Newton-GS smoother — 4-field */
static void mg_gs_sweep_4field(mg_level_t *lev)
{
    int gw = lev->ghost, N = lev->N, Nt = lev->Ntotal;
    double inv_dx = lev->inv_dx, dx2 = lev->dx * lev->dx;
    int sx = 1, sy = Nt, sz = Nt * Nt;
    int strides[3] = { sx, sy, sz };
    double J_lap = 3.0 * FD_D2_CENTER_WEIGHT / dx2;
    double J_V = J_lap + (1.0 / 3.0) * FD_D2_CENTER_WEIGHT / dx2;

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
                    double p2 = psi_tot * psi_tot, p4 = p2 * p2;
                    double p7 = p4 * p2 * psi_tot, p8 = p7 * psi_tot;
                    double src = lev->R_tilde[idx] * 0.125 * psi_tot
                               + lev->A2[idx] * 0.125 / p7;
                    double dS = 0.125 * lev->R_tilde[idx]
                              - 0.875 * lev->A2[idx] / p8;
                    lev->psi[idx] -= (lap + src - lev->f_psi[idx]) / (J_lap + dS);

                    for (int d = 0; d < 3; d++) {
                        double lap_V = fd_d2(lev->V[d], idx, sx, inv_dx)
                                     + fd_d2(lev->V[d], idx, sy, inv_dx)
                                     + fd_d2(lev->V[d], idx, sz, inv_dx);
                        double d_divV = 0.0;
                        for (int e = 0; e < 3; e++) {
                            if (e == d) d_divV += fd_d2(lev->V[e], idx, strides[e], inv_dx);
                            else d_divV += fd_d2_mixed(lev->V[e], idx, strides[d], strides[e], inv_dx);
                        }
                        double res_V = lap_V + d_divV / 3.0 + lev->S_M[d][idx] - lev->f_V[d][idx];
                        lev->V[d][idx] -= res_V / J_V;
                    }
                }
    }
    mg_apply_bc(lev, 1);
}

static void mg_smooth(mg_level_t *lev, int four_field, int n_sweeps)
{
    for (int s = 0; s < n_sweeps; s++) {
        if (four_field) mg_gs_sweep_4field(lev);
        else            mg_gs_sweep_1field(lev);
    }
}

/* Residual norm: ||f - L(u)||_L2 */
static double mg_residual_norm(mg_level_t *lev, int four_field)
{
    mg_compute_operator(lev, four_field);
    int gw = lev->ghost, N = lev->N, Nt = lev->Ntotal;
    int sy = Nt, sz = Nt * Nt;
    double sum = 0.0;
    int count = 0;
    for (int k = gw; k < gw + N; k++)
        for (int j = gw; j < gw + N; j++)
            for (int i = gw; i < gw + N; i++) {
                int idx = k * sz + j * sy + i;
                double r = lev->f_psi[idx] - lev->L_psi[idx];
                sum += r * r;
                if (four_field)
                    for (int d = 0; d < 3; d++) {
                        double rv = lev->f_V[d][idx] - lev->L_V[d][idx];
                        sum += rv * rv;
                    }
                count++;
            }
    int nc = four_field ? 4 : 1;
    return (count > 0) ? sqrt(sum / (count * nc)) : 0.0;
}

/* ================================================================
 * Single-grid FAS V-cycle (recursive)
 * Ref: Trottenberg et al., Algorithm 2.5.2
 * ================================================================ */
static void mg_vcycle(mg_level_t *levels, int n_levels, int level,
                      int four_field)
{
    mg_level_t *fine = &levels[level];
    if (level == n_levels - 1) {
        mg_smooth(fine, four_field, MG_NU_COARSE);
        return;
    }
    mg_level_t *coarse = &levels[level + 1];
    int Nc = coarse->N, gw = coarse->ghost;
    int sy_c = coarse->Ntotal, sz_c = coarse->Ntotal * coarse->Ntotal;

    mg_smooth(fine, four_field, MG_NU_PRE);

    /* Compute residual r = f - L(u) */
    mg_compute_operator(fine, four_field);
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
                            fine->L_V[d][idx] = fine->f_V[d][idx] - fine->L_V[d][idx];
                }
    }

    /* Restrict solution + residual to coarse */
    mg_restrict_field(fine->psi, fine->Ntotal, coarse->save_psi, coarse->Ntotal, Nc, gw);
    mg_apply_bc_field(coarse->save_psi, coarse);
    mg_restrict_field(fine->L_psi, fine->Ntotal, coarse->f_psi, coarse->Ntotal, Nc, gw);
    if (four_field)
        for (int d = 0; d < 3; d++) {
            mg_restrict_field(fine->V[d], fine->Ntotal, coarse->save_V[d], coarse->Ntotal, Nc, gw);
            mg_apply_bc_field(coarse->save_V[d], coarse);
            mg_restrict_field(fine->L_V[d], fine->Ntotal, coarse->f_V[d], coarse->Ntotal, Nc, gw);
        }

    memcpy(coarse->psi, coarse->save_psi, coarse->npoints * sizeof(double));
    if (four_field)
        for (int d = 0; d < 3; d++)
            memcpy(coarse->V[d], coarse->save_V[d], coarse->npoints * sizeof(double));

    /* Tau correction */
    mg_compute_operator(coarse, four_field);
    for (int k = gw; k < gw + Nc; k++)
        for (int j = gw; j < gw + Nc; j++)
            for (int i = gw; i < gw + Nc; i++) {
                int idx = k * sz_c + j * sy_c + i;
                coarse->f_psi[idx] += coarse->L_psi[idx];
                if (four_field)
                    for (int d = 0; d < 3; d++)
                        coarse->f_V[d][idx] += coarse->L_V[d][idx];
            }

    mg_vcycle(levels, n_levels, level + 1, four_field);

    /* Correction = coarse - saved, prolongate to fine */
    for (int k = gw; k < gw + Nc; k++)
        for (int j = gw; j < gw + Nc; j++)
            for (int i = gw; i < gw + Nc; i++) {
                int idx = k * sz_c + j * sy_c + i;
                coarse->save_psi[idx] = coarse->psi[idx] - coarse->save_psi[idx];
                if (four_field)
                    for (int d = 0; d < 3; d++)
                        coarse->save_V[d][idx] = coarse->V[d][idx] - coarse->save_V[d][idx];
            }
    mg_apply_bc_field(coarse->save_psi, coarse);
    mg_prolongate_add_field(coarse->save_psi, coarse->Ntotal, fine->psi, fine->Ntotal, fine->N, fine->ghost);
    if (four_field)
        for (int d = 0; d < 3; d++) {
            mg_apply_bc_field(coarse->save_V[d], coarse);
            mg_prolongate_add_field(coarse->save_V[d], coarse->Ntotal, fine->V[d], fine->Ntotal, fine->N, fine->ghost);
        }
    mg_apply_bc(fine, four_field);
    mg_smooth(fine, four_field, MG_NU_POST);
}

/* Single-grid FMG */
static void mg_fmg(mg_level_t *levels, int n_levels, int four_field)
{
    mg_smooth(&levels[n_levels - 1], four_field, MG_NU_COARSE);
    for (int lev = n_levels - 2; lev >= 0; lev--) {
        mg_prolongate_fmg_field(levels[lev + 1].psi, levels[lev + 1].Ntotal,
                                levels[lev].psi, levels[lev].Ntotal,
                                levels[lev].N, levels[lev].ghost);
        if (four_field)
            for (int d = 0; d < 3; d++)
                mg_prolongate_fmg_field(levels[lev + 1].V[d], levels[lev + 1].Ntotal,
                                        levels[lev].V[d], levels[lev].Ntotal,
                                        levels[lev].N, levels[lev].ghost);
        mg_apply_bc(&levels[lev], four_field);
        mg_vcycle(levels, n_levels, lev, four_field);
    }
}

/* ================================================================
 * Covering grid FAS solver for a single AMR level.
 *
 * Gathers all blocks at one AMR level into a single temporary uniform
 * grid (the "covering grid"), runs the proven single-grid FAS solver
 * on it, then scatters the solution back to blocks.
 *
 * No inter-block ghost exchange during MG — it's one contiguous grid.
 * Uses 100% proven single-grid FAS code (FMG + V-cycles + N-GS smoother).
 *
 * Ref: AMReX MLMG uses this for levels with few boxes.
 * ================================================================ */
static double fas_solve_level(mesh_t *m, int amr_level, int four_field,
                              double tol, int max_iter, int verbose,
                              int n_bh, const puncture_data_t *bhs)
{
    /* Collect all blocks at this AMR level */
    int n_blocks = 0;
    for (int b = 0; b < m->num_blocks; b++)
        if (m->blocks[b] && m->blocks[b]->loc.level == amr_level)
            n_blocks++;
    if (n_blocks == 0) return 0.0;

    /* Compute bounding box of all blocks at this level */
    double lo[3] = { 1e30,  1e30,  1e30};
    double hi[3] = {-1e30, -1e30, -1e30};
    double dx = 0.0;
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || blk->loc.level != amr_level) continue;
        dx = blk->grid->dx;
        int N = blk->grid->N;
        for (int d = 0; d < 3; d++) {
            if (blk->origin[d] < lo[d]) lo[d] = blk->origin[d];
            double block_hi = blk->origin[d] + N * dx;
            if (block_hi > hi[d]) hi[d] = block_hi;
        }
    }

    /* Create covering grid */
    int Ncov = (int)round((hi[0] - lo[0]) / dx);
    if (Ncov < 8) Ncov = 8;
    double Lcov = Ncov * dx;

    if (verbose > 1)
        printf("      [FAS] Covering grid: N=%d, dx=%.4f, L=%.2f\n",
               Ncov, dx, Lcov);

    /* Build MG hierarchy on covering grid */
    int n_mg_levels = 1;
    {
        int N = Ncov;
        while (N / (1 << n_mg_levels) >= MG_N_MIN && n_mg_levels < MG_MAX_LEVELS)
            n_mg_levels++;
    }

    mg_level_t *levels = calloc(n_mg_levels, sizeof(mg_level_t));
    for (int l = 0; l < n_mg_levels; l++) {
        int N_l = Ncov / (1 << l);
        mg_level_init(&levels[l], N_l, Lcov, four_field);
    }

    /* Precompute backgrounds on covering grid.
     * The covering grid center = midpoint of bounding box. */
    double center[3];
    for (int d = 0; d < 3; d++)
        center[d] = 0.5 * (lo[d] + hi[d]);

    /* Precompute backgrounds on finest MG level */
    {
        mg_level_t *lev = &levels[0];
        int Nt = lev->Ntotal;
        int gw_l = lev->ghost;
        double inv_dx_l = lev->inv_dx;
        static const int s_map[3][3] = {{0,1,2},{1,3,4},{2,4,5}};

        /* Temporary arrays for 4-field h_bg/A_bg */
        double *h_bg[6] = {NULL}, *A_bg[6] = {NULL};
        if (four_field) {
            for (int c = 0; c < 6; c++) {
                h_bg[c] = calloc(lev->npoints, sizeof(double));
                A_bg[c] = calloc(lev->npoints, sizeof(double));
            }
        }

        for (int k = 0; k < Nt; k++)
            for (int j = 0; j < Nt; j++)
                for (int i = 0; i < Nt; i++) {
                    int idx = k * Nt * Nt + j * Nt + i;
                    double x = center[0] + ((i - gw_l + 0.5) * lev->dx - Lcov * 0.5);
                    double y = center[1] + ((j - gw_l + 0.5) * lev->dx - Lcov * 0.5);
                    double z = center[2] + ((k - gw_l + 0.5) * lev->dx - Lcov * 0.5);
                    lev->psi_BL[idx] = brill_lindquist_psi(x, y, z, n_bh, bhs);

                    if (!four_field) {
                        double A_tilde[3][3];
                        bowen_york_Aij(A_tilde, x, y, z, n_bh, bhs);
                        lev->A2[idx] = bowen_york_A2(A_tilde);
                    } else {
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
                        lev->A2[idx] = hispid_A2(A_total, h);
                    }
                }

        /* 4-field: compute R_tilde and S_M via FD on covering grid */
        if (four_field) {
            int sx = 1, sy = Nt, sz = Nt * Nt;
            int strides[3] = { sx, sy, sz };
            for (int k = gw_l; k < Nt - gw_l; k++)
                for (int jj = gw_l; jj < Nt - gw_l; jj++)
                    for (int i = gw_l; i < Nt - gw_l; i++) {
                        int idx = k * sz + jj * sy + i;
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
                                                       strides[dir], inv_dx_l);
                                    d1_h[a][b][dir] = val;
                                    d1_h[b][a][dir] = val;
                                }
                        double d2_h[3][3][3][3];
                        memset(d2_h, 0, sizeof(d2_h));
                        for (int dir = 0; dir < 3; dir++)
                            for (int a = 0; a < 3; a++)
                                for (int b = a; b < 3; b++) {
                                    double val = fd_d2(h_bg[s_map[a][b]], idx,
                                                       strides[dir], inv_dx_l);
                                    d2_h[a][b][dir][dir] = val;
                                    d2_h[b][a][dir][dir] = val;
                                }
                        for (int d1 = 0; d1 < 3; d1++)
                            for (int d2 = 0; d2 < d1; d2++)
                                for (int a = 0; a < 3; a++)
                                    for (int b = a; b < 3; b++) {
                                        double val = fd_d2_mixed(h_bg[s_map[a][b]], idx,
                                                                 strides[d1], strides[d2], inv_dx_l);
                                        d2_h[a][b][d1][d2] = val; d2_h[a][b][d2][d1] = val;
                                        d2_h[b][a][d1][d2] = val; d2_h[b][a][d2][d1] = val;
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
                                        R_ab += chris.ULL[kk][ll][a] * c1
                                              + chris.ULL[kk][ll][b] * c2
                                              + chris.ULL[kk][a][ll] * c3;
                                    }
                                }
                                R_scalar += h_UU[a][b] * R_ab;
                            }
                        lev->R_tilde[idx] = R_scalar;

                        for (int d = 0; d < 3; d++) {
                            double div_A = 0.0;
                            for (int e = 0; e < 3; e++)
                                div_A += fd_d1(A_bg[s_map[d][e]], idx,
                                               strides[e], inv_dx_l);
                            lev->S_M[d][idx] = -div_A;
                        }
                    }
            for (int c = 0; c < 6; c++) { free(h_bg[c]); free(A_bg[c]); }
        }
    }

    /* Restrict backgrounds to coarsened MG levels */
    for (int l = 1; l < n_mg_levels; l++) {
        mg_level_t *fine_l = &levels[l - 1];
        mg_level_t *coarse_l = &levels[l];
        mg_restrict_field(fine_l->psi_BL, fine_l->Ntotal,
                          coarse_l->psi_BL, coarse_l->Ntotal,
                          coarse_l->N, coarse_l->ghost);
        mg_restrict_field(fine_l->A2, fine_l->Ntotal,
                          coarse_l->A2, coarse_l->Ntotal,
                          coarse_l->N, coarse_l->ghost);
        if (four_field) {
            mg_restrict_field(fine_l->R_tilde, fine_l->Ntotal,
                              coarse_l->R_tilde, coarse_l->Ntotal,
                              coarse_l->N, coarse_l->ghost);
            for (int d = 0; d < 3; d++)
                mg_restrict_field(fine_l->S_M[d], fine_l->Ntotal,
                                  coarse_l->S_M[d], coarse_l->Ntotal,
                                  coarse_l->N, coarse_l->ghost);
        }
    }

    /* Gather block solutions into covering grid (initial guess = 0) */
    /* (already zeroed by calloc in mg_level_init) */

    /* Run single-grid FAS — proven code, no inter-block exchange */
    mg_fmg(levels, n_mg_levels, four_field);

    double residual = mg_residual_norm(&levels[0], four_field);
    if (verbose > 1)
        printf("      [FAS] FMG done: residual = %.6e\n", residual);

    for (int cycle = 0; cycle < max_iter && residual > tol; cycle++) {
        mg_vcycle(levels, n_mg_levels, 0, four_field);
        residual = mg_residual_norm(&levels[0], four_field);
        if (verbose > 1)
            printf("      [FAS] V-cycle %d: residual = %.6e\n",
                   cycle + 1, residual);
    }

    if (verbose > 1)
        printf("      [FAS] Final: residual = %.6e\n", residual);

    /* Scatter solution from covering grid back to blocks */
    mg_level_t *finest = &levels[0];
    int gw = finest->ghost;
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || blk->loc.level != amr_level) continue;
        grid_t *g = blk->grid;
        (void)0;

        for (int k = 0; k < g->Ntotal; k++)
            for (int j = 0; j < g->Ntotal; j++)
                for (int i = 0; i < g->Ntotal; i++) {
                    /* Map block grid point to covering grid point */
                    double x = BLOCK_COORD(blk, 0, i);
                    double y = BLOCK_COORD(blk, 1, j);
                    double z = BLOCK_COORD(blk, 2, k);
                    int ci = gw + (int)round((x - (center[0] - Lcov * 0.5)) / dx - 0.5);
                    int cj = gw + (int)round((y - (center[1] - Lcov * 0.5)) / dx - 0.5);
                    int ck = gw + (int)round((z - (center[2] - Lcov * 0.5)) / dx - 0.5);

                    if (ci >= 0 && ci < finest->Ntotal &&
                        cj >= 0 && cj < finest->Ntotal &&
                        ck >= 0 && ck < finest->Ntotal) {
                        int cov_idx = ck * finest->Ntotal * finest->Ntotal
                                    + cj * finest->Ntotal + ci;
                        int blk_idx = IDX(g, i, j, k);
                        g->fields[SOL_PSI][blk_idx] = finest->psi[cov_idx];
                        if (four_field)
                            for (int d = 0; d < 3; d++)
                                g->fields[SOL_V1 + d][blk_idx] = finest->V[d][cov_idx];
                    }
                }
    }

    /* Clean up */
    for (int l = 0; l < n_mg_levels; l++)
        mg_level_free(&levels[l]);
    free(levels);

    return residual;
}

/* ================================================================
 * Level-by-level driver
 * ================================================================ */

/* Prolongate solution from parent blocks to newly created fine blocks.
 * Trilinear interpolation of solver fields from parent -> child. */
static void prolongate_initial_guess(mesh_t *m, int fine_level, int n_comp)
{
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *fblk = m->blocks[b];
        if (!fblk || fblk->loc.level != fine_level) continue;
        if (fblk->parent_id < 0) continue;
        block_t *pblk = m->blocks[fblk->parent_id];
        if (!pblk || !pblk->grid) continue;

        grid_t *fg = fblk->grid;
        grid_t *pg = pblk->grid;
        int f_ghost = fg->ghost;
        int f_N = fg->N;
        int p_ghost = pg->ghost;
        int p_N = pg->N;
        int half_N = p_N / 2;

        int cx = fblk->loc.lx1 % 2;
        int cy = fblk->loc.lx2 % 2;
        int cz = fblk->loc.lx3 % 2;
        int c_off_i = cx * half_N;
        int c_off_j = cy * half_N;
        int c_off_k = cz * half_N;

        for (int s = 0; s < n_comp; s++) {
            double *csol = pg->fields[s];
            double *fsol = fg->fields[s];

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
}

/* Compute global residual norm across all leaf blocks */
static double global_residual_norm(mesh_t *m, int four_field)
{
    int n_comp = four_field ? 4 : 1;

    solver_ghost_exchange_all(m, four_field);

    /* Compute operator on all leaf blocks */
    for (int L = 0; L <= m->max_level; L++)
        compute_operator_level(m, L, four_field);

    double sum = 0.0;
    int count = 0;
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || !blk->is_leaf) continue;
        grid_t *g = blk->grid;
        int gw = g->ghost, N = g->N, Nt = g->Ntotal;
        for (int k = gw; k < gw + N; k++)
            for (int j = gw; j < gw + N; j++)
                for (int i = gw; i < gw + N; i++) {
                    int idx = k * Nt * Nt + j * Nt + i;
                    for (int c = 0; c < n_comp; c++) {
                        double val = g->accum[c][idx];
                        sum += val * val;
                    }
                    count++;
                }
    }
    count *= n_comp;
    return (count > 0) ? sqrt(sum / count) : 0.0;
}

static double fas_solve_level_by_level(mesh_t *m, int n_bh,
                                       const puncture_data_t *bhs,
                                       double tol, int max_iter,
                                       int verbose, int four_field)
{
    int n_comp = four_field ? 4 : 1;

    /* Precompute backgrounds on all blocks */
    if (verbose)
        printf("[FAS] Precomputing %s backgrounds...\n",
               four_field ? "4-field" : "1-field");

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk) continue;
        if (four_field) amr_precompute_bg_4field_block(blk, n_bh, bhs);
        else            amr_precompute_bg_1field_block(blk, n_bh, bhs);
    }

    /* Zero solver solution on all blocks */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk) continue;
        for (int s = 0; s < n_comp; s++) {
            memset(blk->grid->fields[s], 0,
                   blk->grid->npoints * sizeof(double));
        }
    }

    solver_ghost_exchange_all(m, four_field);

    /* Solve level by level, coarse to fine.
     * Solve on ALL blocks at each level (not just leaves). Non-leaf
     * blocks provide the coarse correction u that finer levels read
     * via CF ghost boundary interpolation. */
    for (int L = 0; L <= m->max_level; L++) {
        /* Count all blocks at this level */
        int n_at_level = 0;
        for (int b = 0; b < m->num_blocks; b++) {
            block_t *blk = m->blocks[b];
            if (blk && blk->loc.level == L)
                n_at_level++;
        }
        if (n_at_level == 0) continue;

        if (verbose)
            printf("[FAS] Level %d: %d blocks\n", L, n_at_level);

        /* Prolongate initial guess from parent (converged coarser level) */
        if (L > 0) {
            prolongate_initial_guess(m, L, n_comp);
            solver_full_exchange(m, L, four_field);
        }

        /* FAS multigrid solve on this level */
        fas_solve_level(m, L, four_field, tol, max_iter, verbose,
                        n_bh, bhs);

        /* Ghost exchange on converged solution */
        solver_same_level_exchange(m, L, four_field);
    }

    /* Compute global residual */
    double residual = global_residual_norm(m, four_field);
    if (verbose)
        printf("[FAS] Final residual = %.6e\n", residual);

    return residual;
}

/* ================================================================
 * Mesh refinement near punctures (reused from relaxation_amr.c)
 * ================================================================ */

static const double REFINE_BETA = 1.5157165665103982;  /* pow(2.0, 0.6) */
static const double REFINE_C    = 4.0;

void refine_mesh_near_punctures(mesh_t *m, int n_amr_levels,
                                int n_bh, const puncture_data_t *bhs)
{
    int base_level = m->max_level;
    double S = m->L / 2.0;

    double m_min = bhs[0].mass;
    for (int p = 1; p < n_bh; p++)
        if (bhs[p].mass < m_min) m_min = bhs[p].mass;
    double r_finest = REFINE_C * m_min;
    if (r_finest < S) {
        int n_useful = (int)(log(S / r_finest) / log(REFINE_BETA)) + 1;
        if (n_amr_levels > n_useful) {
            printf("[AMR-refine] Capping levels from %d to %d "
                   "(domain S=%.1f, r_finest=%.2f)\n",
                   n_amr_levels, n_useful, S, r_finest);
            n_amr_levels = n_useful;
        }
    }

    for (int level = base_level; level < base_level + n_amr_levels; level++) {
        int k = level - base_level;
        int levels_from_finest = n_amr_levels - 1 - k;
        double dx_level = m->L / (double)(m->N_block * (1 << level));

        double m_max = 0.0;
        for (int p = 0; p < n_bh; p++)
            if (bhs[p].mass > m_max) m_max = bhs[p].mass;
        double r_log = REFINE_C * m_max * pow(REFINE_BETA, levels_from_finest);
        if (r_log > S) r_log = S;

        printf("[AMR-refine] Level %d/%d: dx=%.4f, r_refine=%.2f, blocks=%d\n",
               level + 1, base_level + n_amr_levels, dx_level, r_log,
               m->num_blocks);
        fflush(stdout);

        int n_to_refine = 0;
        int *refine_ids = calloc(m->num_blocks, sizeof(int));

        for (int b = 0; b < m->num_blocks; b++) {
            block_t *blk = m->blocks[b];
            if (!blk || !blk->is_leaf || blk->loc.level != level) continue;

            double block_dx = blk->grid->dx;
            int N_blk = blk->grid->N;
            double bx_min = blk->origin[0];
            double by_min = blk->origin[1];
            double bz_min = blk->origin[2];
            double bx_max = bx_min + N_blk * block_dx;
            double by_max = by_min + N_blk * block_dx;
            double bz_max = bz_min + N_blk * block_dx;

            for (int p = 0; p < n_bh; p++) {
                double r_refine = REFINE_C * bhs[p].mass
                                * pow(REFINE_BETA, levels_from_finest);
                if (r_refine > S) r_refine = S;

                double pcx = bhs[p].center[0];
                double pcy = bhs[p].center[1];
                double pcz = bhs[p].center[2];
                double nx = (pcx < bx_min) ? bx_min : (pcx > bx_max) ? bx_max : pcx;
                double ny = (pcy < by_min) ? by_min : (pcy > by_max) ? by_max : pcy;
                double nz = (pcz < bz_min) ? bz_min : (pcz > bz_max) ? bz_max : pcz;
                double dist = sqrt((nx-pcx)*(nx-pcx) + (ny-pcy)*(ny-pcy) + (nz-pcz)*(nz-pcz));
                if (dist < r_refine) {
                    refine_ids[n_to_refine++] = blk->id;
                    break;
                }
            }
        }

        for (int r = 0; r < n_to_refine; r++) {
            block_t *blk = m->blocks[refine_ids[r]];
            if (blk && blk->is_leaf)
                mesh_refine_block(m, refine_ids[r]);
        }

        free(refine_ids);
        mesh_compact(m);
        mesh_rebuild_neighbors(m);
    }
}

/* ================================================================
 * Create solver mesh with refinement near punctures
 * ================================================================ */
static mesh_t *create_solver_mesh(int N_base, double L, int n_amr_levels,
                                   int n_bh, const puncture_data_t *bhs)
{
    mesh_t *m = mesh_create_ex(N_base, L, RK_CLASSIC, JFNK_N_FIELDS);
    refine_mesh_near_punctures(m, n_amr_levels, n_bh, bhs);
    return m;
}

/* ================================================================
 * Transfer functions: copy solver data to uniform output grid
 * ================================================================ */

static void transfer_1field_to_grid(mesh_t *m, grid_t *g, double *psi_full)
{
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
}

static void transfer_4field_to_grid(mesh_t *m, grid_t *g,
                                     double *psi_full, double *V_full[3])
{
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

    /* Overwrite with restricted data from finer leaf blocks */
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
}

/* ================================================================
 * Public API: mesh-based solvers
 * ================================================================ */

double jfnk_solve_mesh(mesh_t *m, int n_bh, const puncture_data_t *bhs,
                        double tol, int max_iter, int verbose,
                        int n_amr_levels)
{
    if (n_amr_levels > MAX_AMR_LEVELS) {
        fprintf(stderr, "[FAS-mesh] ERROR: n_amr_levels=%d exceeds "
                "MAX_AMR_LEVELS=%d\n", n_amr_levels, MAX_AMR_LEVELS);
        return -1.0;
    }

    /* Validate accum arrays */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || !blk->is_leaf) continue;
        if (!blk->grid->accum_block) {
            fprintf(stderr, "[FAS-mesh] ERROR: accum arrays not allocated. "
                    "Use RK_CLASSIC for evolution mesh solver.\n");
            return -1.0;
        }
        break;
    }

    if (verbose)
        printf("[FAS-mesh] Starting 1-field solver, %d AMR levels, tol=%.2e\n",
               n_amr_levels, tol);

    if (n_amr_levels > 0)
        refine_mesh_near_punctures(m, n_amr_levels, n_bh, bhs);
    if (verbose)
        printf("[FAS-mesh] Mesh: %d blocks, %d leaves, max_level=%d\n",
               m->num_blocks, mesh_num_leaves(m), m->max_level);

    return fas_solve_level_by_level(m, n_bh, bhs, tol, max_iter,
                                    verbose, /*four_field=*/0);
}

double jfnk_solve_mesh_coupled(mesh_t *m, int n_bh, const puncture_data_t *bhs,
                                double tol, int max_iter, int verbose,
                                int n_amr_levels)
{
    /* Validate accum arrays */
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || !blk->is_leaf) continue;
        if (!blk->grid->accum_block) {
            fprintf(stderr, "[FAS-mesh] ERROR: accum arrays not allocated. "
                    "Use RK_CLASSIC for evolution mesh solver.\n");
            return -1.0;
        }
        break;
    }

    if (verbose)
        printf("[FAS-mesh] Starting 4-field solver, %d AMR levels, tol=%.2e\n",
               n_amr_levels, tol);

    if (n_amr_levels > 0)
        refine_mesh_near_punctures(m, n_amr_levels, n_bh, bhs);
    if (verbose)
        printf("[FAS-mesh] Mesh: %d blocks, %d leaves, max_level=%d\n",
               m->num_blocks, mesh_num_leaves(m), m->max_level);

    return fas_solve_level_by_level(m, n_bh, bhs, tol, max_iter,
                                    verbose, /*four_field=*/1);
}

/* ================================================================
 * Public API: grid-based solvers (create internal mesh)
 * ================================================================ */

double jfnk_solve(grid_t *g, int n_bh, const puncture_data_t *bhs,
                   double tol, int max_iter, int verbose)
{
    return jfnk_solve_amr(g, n_bh, bhs, tol, max_iter, verbose, 0);
}

double jfnk_solve_coupled(grid_t *g, int n_bh, const puncture_data_t *bhs,
                            double tol, int max_iter, int verbose)
{
    return jfnk_solve_coupled_amr(g, n_bh, bhs, tol, max_iter, verbose, 0);
}

double jfnk_solve_amr(grid_t *g, int n_bh, const puncture_data_t *bhs,
                       double tol, int max_iter, int verbose,
                       int n_amr_levels)
{
    if (verbose)
        printf("[FAS] Starting 1-field solver: N=%d, %d AMR levels, tol=%.2e\n",
               g->N, n_amr_levels, tol);

    mesh_t *m = create_solver_mesh(g->N, g->L, n_amr_levels, n_bh, bhs);
    if (verbose)
        printf("[FAS] Solver mesh: %d blocks, %d leaves, max_level=%d\n",
               m->num_blocks, mesh_num_leaves(m), m->max_level);

    double residual = fas_solve_level_by_level(m, n_bh, bhs, tol, max_iter,
                                               verbose, /*four_field=*/0);

    double *psi_full = calloc(g->npoints, sizeof(double));
    transfer_1field_to_grid(m, g, psi_full);
    set_ccz4_from_psi(g, psi_full, n_bh, bhs);

    free(psi_full);
    mesh_free(m);
    return residual;
}

double jfnk_solve_coupled_amr(grid_t *g, int n_bh, const puncture_data_t *bhs,
                               double tol, int max_iter, int verbose,
                               int n_amr_levels)
{
    if (verbose)
        printf("[FAS] Starting coupled solver: N=%d, %d AMR levels\n",
               g->N, n_amr_levels);

    mesh_t *m = create_solver_mesh(g->N, g->L, n_amr_levels, n_bh, bhs);
    if (verbose)
        printf("[FAS] Solver mesh: %d blocks, %d leaves, max_level=%d\n",
               m->num_blocks, mesh_num_leaves(m), m->max_level);

    double residual = fas_solve_level_by_level(m, n_bh, bhs, tol, max_iter,
                                               verbose, /*four_field=*/1);

    double *psi_full = calloc(g->npoints, sizeof(double));
    double *V_full[3];
    for (int d = 0; d < 3; d++)
        V_full[d] = calloc(g->npoints, sizeof(double));
    transfer_4field_to_grid(m, g, psi_full, V_full);
    set_ccz4_from_hispid(g, psi_full, V_full, n_bh, bhs);

    free(psi_full);
    for (int d = 0; d < 3; d++)
        free(V_full[d]);
    mesh_free(m);
    return residual;
}
