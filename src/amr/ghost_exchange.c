/*
 * Lattice — 3D Numerical Relativity
 * 26-neighbor ghost zone exchange (same-level + multi-level).
 *
 * Same-level exchange: for each block and each of its 26 neighbors,
 * copy the appropriate region from the neighbor's interior into the
 * block's ghost zone. Since we only read from neighbors' interior cells
 * (which are not modified by ghost exchange), iteration order is arbitrary.
 *
 * Multi-level exchange (Stage 4.1): AthenaK coarse-buffer architecture.
 * Each leaf block at level > 0 carries its own coarse_buf grid at half
 * resolution. All operations are block-local — no cross-block writes.
 *
 * Phases (all embarrassingly parallel per block):
 *   Phase 0+1: Same-level exchange at each level (fine siblings + root)
 *   Phase 2:   Restrict fine → own coarse_buf (4th-order, block-local)
 *   Phase 3:   Fill coarse_buf ghost zones (from sibling bufs + coarse nbrs)
 *   Phase 4:   Prolongate own coarse_buf → fine ghost (skip same-level dirs)
 *
 * Ref: AthenaK src/bvals/ (GPU boundary exchange, coarse-buffer)
 * Ref: Athena++ src/bvals/bvals_cc.cpp (CellCenteredBoundaryVariable)
 * Ref: GRChombo GRAMRLevel.cpp:1029-1043 (FillPatch pattern)
 */

#include "ghost_exchange.h"
#include "prolongation.h"
#include "restriction.h"
#include "../core/fields.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Compute source and destination index ranges for one direction.
 *   offset: neighbor offset in this direction (-1, 0, +1)
 *   ghost:  ghost zone width (GHOST_WIDTH)
 *   N:      interior cells per side
 *   Nt:     total cells per side (N + 2*ghost)
 *
 * Returns: dst_lo, dst_hi, src_lo, src_hi (half-open ranges)
 *
 * Matches AthenaK buffs_cc.cpp InitSendIndices/InitRecvIndices:
 *   Send ox>0: [ie-ng+1, ie]    Recv ox>0: [ie+1, ie+ng]
 *   Send ox<0: [is, is+ng-1]    Recv ox<0: [is-ng, is-1]
 *   Send ox=0: [is, ie]         Recv ox=0: [is, ie]
 */
static inline void ghost_range(int offset, int ghost, int N, int Nt,
                               int *dst_lo, int *dst_hi,
                               int *src_lo, int *src_hi)
{
    int hi = ghost + N;

    if (offset == -1) {
        /* Low ghost zone ← neighbor's high interior */
        *dst_lo = 0;
        *dst_hi = ghost;
        *src_lo = hi - ghost;   /* = N */
        *src_hi = hi;           /* = ghost + N */
    } else if (offset == 0) {
        /* Interior range ← same interior range */
        *dst_lo = ghost;
        *dst_hi = hi;
        *src_lo = ghost;
        *src_hi = hi;
    } else {
        /* High ghost zone ← neighbor's low interior */
        *dst_lo = hi;
        *dst_hi = Nt;
        *src_lo = ghost;
        *src_hi = ghost + (Nt - hi);  /* = 2*ghost */
    }
}

/* ========================================================================
 * Same-level exchange helpers
 * ======================================================================== */

/*
 * Exchange a single neighbor pair: copy from nbr's interior to b's ghost.
 * field_arrays: 0 = fields, 1 = scratch (for use in RK intermediate states).
 */
static void exchange_neighbor(block_t *b, const block_t *nbr,
                              int ox, int oy, int oz, int field_arrays)
{
    grid_t *dg = b->grid;
    const grid_t *sg = nbr->grid;
    int ghost = dg->ghost;
    int N     = dg->N;
    int Nt    = dg->Ntotal;

    /* Compute ranges for each direction */
    int dx_lo, dx_hi, sx_lo, sx_hi;
    int dy_lo, dy_hi, sy_lo, sy_hi;
    int dz_lo, dz_hi, sz_lo, sz_hi;

    ghost_range(ox, ghost, N, Nt, &dx_lo, &dx_hi, &sx_lo, &sx_hi);
    ghost_range(oy, ghost, N, Nt, &dy_lo, &dy_hi, &sy_lo, &sy_hi);
    ghost_range(oz, ghost, N, Nt, &dz_lo, &dz_hi, &sz_lo, &sz_hi);

    /* Select which arrays to read from / write to */
    double *const      *dst_arr = (field_arrays == 0) ? dg->fields : dg->scratch;
    const double *const *src_arr = (field_arrays == 0)
                                   ? (const double *const *)sg->fields
                                   : (const double *const *)sg->scratch;

    int nx = dx_hi - dx_lo;

    for (int f = 0; f < dg->n_fields; f++) {
        for (int k = 0; k < (dz_hi - dz_lo); k++) {
            int dk = dz_lo + k;
            int sk = sz_lo + k;
            for (int j = 0; j < (dy_hi - dy_lo); j++) {
                int dj = dy_lo + j;
                int sj = sy_lo + j;

                /* memcpy the contiguous x-strip (unit stride) */
                int dst_idx = IDX(dg, dx_lo, dj, dk);
                int src_idx = IDX(sg, sx_lo, sj, sk);
                memcpy(&dst_arr[f][dst_idx], &src_arr[f][src_idx],
                       nx * sizeof(double));
            }
        }
    }
}

/*
 * Exchange between two grids of the same dimensions (N, ghost).
 * Used for coarse_buf ↔ coarse_buf exchange between same-level siblings.
 */
static void exchange_grid_pair(grid_t *dg, const grid_t *sg,
                               int ox, int oy, int oz)
{
    int ghost = dg->ghost;
    int N     = dg->N;
    int Nt    = dg->Ntotal;

    int dx_lo, dx_hi, sx_lo, sx_hi;
    int dy_lo, dy_hi, sy_lo, sy_hi;
    int dz_lo, dz_hi, sz_lo, sz_hi;

    ghost_range(ox, ghost, N, Nt, &dx_lo, &dx_hi, &sx_lo, &sx_hi);
    ghost_range(oy, ghost, N, Nt, &dy_lo, &dy_hi, &sy_lo, &sy_hi);
    ghost_range(oz, ghost, N, Nt, &dz_lo, &dz_hi, &sz_lo, &sz_hi);

    int nx = dx_hi - dx_lo;

    for (int f = 0; f < dg->n_fields; f++) {
        for (int k = 0; k < (dz_hi - dz_lo); k++) {
            int dk = dz_lo + k;
            int sk = sz_lo + k;
            for (int j = 0; j < (dy_hi - dy_lo); j++) {
                int dj = dy_lo + j;
                int sj = sy_lo + j;

                int dst_idx = IDX(dg, dx_lo, dj, dk);
                int src_idx = IDX(sg, sx_lo, sj, sk);
                memcpy(&dg->fields[f][dst_idx], &sg->fields[f][src_idx],
                       nx * sizeof(double));
            }
        }
    }
}

/*
 * Same-level exchange at a specific level.
 * Only exchanges between leaf blocks at the same level.
 */
static void exchange_same_level(mesh_t *m, int level)
{
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf || b->loc.level != level) continue;

        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int nbr_id = b->neighbor_ids[n];
            if (nbr_id < 0) continue;

            block_t *nbr = m->blocks[nbr_id];
            if (!nbr || !nbr->is_leaf || nbr->loc.level != level) continue;

            exchange_neighbor(b, nbr,
                              nbr_offset[n][0],
                              nbr_offset[n][1],
                              nbr_offset[n][2],
                              0);
        }
    }
}

/* ========================================================================
 * Phase 3: Fill coarse_buf ghost zones
 * ======================================================================== */

/*
 * Copy from a coarse neighbor's main grid into this block's coarse_buf
 * ghost zone. Both grids have the same dx but different N.
 *
 * Uses index-offset mapping:
 *   src_i = dst_i + round((dst_origin - src_origin) / dx)
 *
 * The coarse neighbor's grid has N = N_block, while coarse_buf has N = N_block/2.
 * Both have ghost = GHOST_WIDTH and the same dx.
 */
static void copy_from_coarse_grid(grid_t *dst, const double *dst_origin,
                                   const grid_t *src, const double *src_origin,
                                   int ox, int oy, int oz)
{
    const double dx = dst->dx;
    const int ghost_d = dst->ghost;
    const int N_d = dst->N;
    const int Nt_d = dst->Ntotal;

    /* Origin offset in grid cells (integer since both grids share dx) */
    int off_i = (int)round((dst_origin[0] - src_origin[0]) / dx);
    int off_j = (int)round((dst_origin[1] - src_origin[1]) / dx);
    int off_k = (int)round((dst_origin[2] - src_origin[2]) / dx);

    /* Compute ghost zone range on destination (coarse_buf) */
    int dx_lo, dx_hi, dummy_s1, dummy_s2;
    int dy_lo, dy_hi, dummy_s3, dummy_s4;
    int dz_lo, dz_hi, dummy_s5, dummy_s6;

    ghost_range(ox, ghost_d, N_d, Nt_d, &dx_lo, &dx_hi, &dummy_s1, &dummy_s2);
    ghost_range(oy, ghost_d, N_d, Nt_d, &dy_lo, &dy_hi, &dummy_s3, &dummy_s4);
    ghost_range(oz, ghost_d, N_d, Nt_d, &dz_lo, &dz_hi, &dummy_s5, &dummy_s6);

    for (int f = 0; f < dst->n_fields; f++) {
        for (int k = dz_lo; k < dz_hi; k++) {
            int sk = k + off_k;
            if (sk < 0 || sk >= src->Ntotal) continue;
            for (int j = dy_lo; j < dy_hi; j++) {
                int sj = j + off_j;
                if (sj < 0 || sj >= src->Ntotal) continue;
                for (int i = dx_lo; i < dx_hi; i++) {
                    int si = i + off_i;
                    if (si < 0 || si >= src->Ntotal) continue;
                    dst->fields[f][IDX(dst, i, j, k)] =
                        src->fields[f][IDX(src, si, sj, sk)];
                }
            }
        }
    }
}

/*
 * Fill coarse_buf ghost zones for all fine leaf blocks.
 *
 * For each leaf at level > 0, for each of 26 neighbor directions:
 * - Same-level neighbor: copy from neighbor's coarse_buf interior
 *   (both coarse_bufs have same N and ghost → use exchange_grid_pair)
 * - Coarser neighbor: copy from coarse neighbor's main grid interior
 *   (different N, same dx → use index-offset mapping)
 */
static void fill_coarse_buf_ghosts(mesh_t *m)
{
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf || b->loc.level == 0) continue;
        if (!b->coarse_buf) continue;

        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int ox = nbr_offset[n][0];
            int oy = nbr_offset[n][1];
            int oz = nbr_offset[n][2];
            int nlev = b->nblevel[oz + 1][oy + 1][ox + 1];

            if (nlev == b->loc.level) {
                /* Same-level neighbor: exchange coarse_buf ↔ coarse_buf */
                int nbr_id = b->neighbor_ids[n];
                if (nbr_id < 0) continue;
                block_t *nbr = m->blocks[nbr_id];
                if (!nbr || !nbr->coarse_buf) continue;

                exchange_grid_pair(b->coarse_buf, nbr->coarse_buf,
                                   ox, oy, oz);

            } else if (nlev >= 0 && nlev == b->loc.level - 1) {
                /* Coarser neighbor: copy from coarse neighbor's grid */
                int nbr_id = b->neighbor_ids[n];
                if (nbr_id < 0) continue;
                block_t *nbr = m->blocks[nbr_id];
                if (!nbr || !nbr->grid) continue;

                copy_from_coarse_grid(b->coarse_buf, b->origin,
                                      nbr->grid, nbr->origin,
                                      ox, oy, oz);
            }
            /* nlev < 0: domain boundary — skip (Sommerfeld handles it) */
        }
    }
}

/* ========================================================================
 * Phase 3.5: Fill coarse_buf boundary ghost cells by extrapolation
 * ======================================================================== */

/*
 * Fill domain-boundary ghost cells of coarse_buf by quadratic extrapolation
 * from interior data. This ensures the prolongation stencil (Phase 4) always
 * reads valid data, even when it extends into boundary ghost zones.
 *
 * Uses 3-point Lagrange extrapolation (exact for degree ≤ 2):
 *   ghost[d] = 3*interior[0] - 3*interior[1] + interior[2]   (d=1 from bdry)
 *   ghost[d] = 6*interior[0] - 8*interior[1] + 3*interior[2] (d=2 from bdry)
 *   ...general: sum_j C(d,j) * interior[j], j=0..2
 *
 * Processes faces sequentially (x, then y, then z) with FULL array ranges
 * in the non-ghost directions, matching AthenaK's BCHelper dimension sweep.
 * Each later sweep reads data written by earlier sweeps, so edge and corner
 * ghost cells are filled automatically without special edge/corner logic:
 *   X sweep: fills X ghosts for ALL j,k (y,z ghost zones may have stale data)
 *   Y sweep: fills Y ghosts for ALL i,k (X ghosts already valid from X sweep)
 *   Z sweep: fills Z ghosts for ALL i,j (X,Y ghosts already valid)
 *
 * Ref: AthenaK src/bvals/physics/z4c_bcs.cpp BCHelper — dimension-by-dimension
 *      sweep with full-width loops (n1, n2, n3 include ghost zones).
 */
static void fill_coarse_buf_boundary(block_t *b)
{
    if (!b || !b->coarse_buf) return;

    grid_t *cg = b->coarse_buf;
    const int gh = cg->ghost;
    const int N = cg->N;
    const int Nt = cg->Ntotal;

    /* 3-point extrapolation coefficients for distance d from boundary.
     * extrap(d) = sum_j coeff[d][j] * f(interior_start + j)
     * Derived from Lagrange basis through points {0, 1, 2} evaluated at -d.
     * d=1: L(-1) = {3, -3, 1},  d=2: L(-2) = {6, -8, 3}, etc. */
    double c[4][3];
    for (int d = 0; d < (int)gh; d++) {
        double t = -(d + 1);  /* evaluate at distance -(d+1) from first interior */
        c[d][0] = (t - 1.0) * (t - 2.0) / 2.0;
        c[d][1] = -t * (t - 2.0);
        c[d][2] = t * (t - 1.0) / 2.0;
    }

    for (int f = 0; f < cg->n_fields; f++) {
        double *data = cg->fields[f];

        /* X-faces: extrapolate in i for ALL j, k (full array width).
         * Non-interior j,k may read stale data, but the Y and Z sweeps
         * below will overwrite those cells with correct values. */
        if (b->nblevel[1][1][0] < 0) {  /* x- boundary */
            for (int k = 0; k < Nt; k++) {
                for (int j = 0; j < Nt; j++) {
                    for (int d = 0; d < gh; d++) {
                        int gi = gh - 1 - d;  /* ghost index */
                        data[IDX(cg, gi, j, k)] =
                            c[d][0] * data[IDX(cg, gh,     j, k)] +
                            c[d][1] * data[IDX(cg, gh + 1, j, k)] +
                            c[d][2] * data[IDX(cg, gh + 2, j, k)];
                    }
                }
            }
        }
        if (b->nblevel[1][1][2] < 0) {  /* x+ boundary */
            for (int k = 0; k < Nt; k++) {
                for (int j = 0; j < Nt; j++) {
                    for (int d = 0; d < gh; d++) {
                        int gi = gh + N + d;
                        data[IDX(cg, gi, j, k)] =
                            c[d][0] * data[IDX(cg, gh + N - 1, j, k)] +
                            c[d][1] * data[IDX(cg, gh + N - 2, j, k)] +
                            c[d][2] * data[IDX(cg, gh + N - 3, j, k)];
                    }
                }
            }
        }

        /* Y-faces: extrapolate in j for ALL i, ALL k (full array width).
         * X-ghost cells already filled by X sweep above; Z-ghost cells
         * may be stale but the Z sweep below will overwrite them. */
        if (b->nblevel[1][0][1] < 0) {  /* y- boundary */
            for (int k = 0; k < Nt; k++) {
                for (int i = 0; i < Nt; i++) {
                    for (int d = 0; d < gh; d++) {
                        int gj = gh - 1 - d;
                        data[IDX(cg, i, gj, k)] =
                            c[d][0] * data[IDX(cg, i, gh,     k)] +
                            c[d][1] * data[IDX(cg, i, gh + 1, k)] +
                            c[d][2] * data[IDX(cg, i, gh + 2, k)];
                    }
                }
            }
        }
        if (b->nblevel[1][2][1] < 0) {  /* y+ boundary */
            for (int k = 0; k < Nt; k++) {
                for (int i = 0; i < Nt; i++) {
                    for (int d = 0; d < gh; d++) {
                        int gj = gh + N + d;
                        data[IDX(cg, i, gj, k)] =
                            c[d][0] * data[IDX(cg, i, gh + N - 1, k)] +
                            c[d][1] * data[IDX(cg, i, gh + N - 2, k)] +
                            c[d][2] * data[IDX(cg, i, gh + N - 3, k)];
                    }
                }
            }
        }

        /* Z-faces: extrapolate in k, for ALL i and ALL j. */
        if (b->nblevel[0][1][1] < 0) {  /* z- boundary */
            for (int j = 0; j < Nt; j++) {
                for (int i = 0; i < Nt; i++) {
                    for (int d = 0; d < gh; d++) {
                        int gk = gh - 1 - d;
                        data[IDX(cg, i, j, gk)] =
                            c[d][0] * data[IDX(cg, i, j, gh)] +
                            c[d][1] * data[IDX(cg, i, j, gh + 1)] +
                            c[d][2] * data[IDX(cg, i, j, gh + 2)];
                    }
                }
            }
        }
        if (b->nblevel[2][1][1] < 0) {  /* z+ boundary */
            for (int j = 0; j < Nt; j++) {
                for (int i = 0; i < Nt; i++) {
                    for (int d = 0; d < gh; d++) {
                        int gk = gh + N + d;
                        data[IDX(cg, i, j, gk)] =
                            c[d][0] * data[IDX(cg, i, j, gh + N - 1)] +
                            c[d][1] * data[IDX(cg, i, j, gh + N - 2)] +
                            c[d][2] * data[IDX(cg, i, j, gh + N - 3)];
                    }
                }
            }
        }
    }
}

/* ========================================================================
 * Phase 4: Prolongate from own coarse_buf into fine ghost zones
 * ======================================================================== */

/*
 * For each fine ghost cell in block b, determine which neighbor direction
 * it belongs to. If that direction has a same-level neighbor, the cell was
 * already filled by Phase 1 → skip. Otherwise, 5-point Lagrange
 * interpolation from b->coarse_buf.
 *
 * Mapping from fine index to coarse_buf continuous index:
 *   ci_cont = (fi - ghost_f + 0.5) / 2.0 + ghost_c - 0.5
 * Since coarse_buf covers the same physical domain at 2× spacing.
 */
static void prolongate_from_own_coarse_buf(block_t *b)
{
    if (!b || !b->coarse_buf) return;

    grid_t *fg = b->grid;
    const grid_t *cg = b->coarse_buf;
    const int ghost_f = fg->ghost;
    const int N_f = fg->N;
    const int Nt_f = fg->Ntotal;
    const int ghost_c = cg->ghost;
    const int half = PROLONG_STENCIL / 2;

    for (int f = 0; f < fg->n_fields; f++) {
        const double *src = cg->fields[f];

        for (int fk = 0; fk < Nt_f; fk++) {
            for (int fj = 0; fj < Nt_f; fj++) {
                for (int fi = 0; fi < Nt_f; fi++) {
                    /* Skip interior cells — only fill ghost zones */
                    if (fi >= ghost_f && fi < ghost_f + N_f &&
                        fj >= ghost_f && fj < ghost_f + N_f &&
                        fk >= ghost_f && fk < ghost_f + N_f)
                        continue;

                    /* Determine ghost zone direction */
                    int ox = (fi < ghost_f) ? -1 :
                             (fi >= ghost_f + N_f) ? 1 : 0;
                    int oy = (fj < ghost_f) ? -1 :
                             (fj >= ghost_f + N_f) ? 1 : 0;
                    int oz = (fk < ghost_f) ? -1 :
                             (fk >= ghost_f + N_f) ? 1 : 0;

                    /* Check nblevel: skip if same-level neighbor filled this
                     * direction in Phase 1. Domain boundary ghost cells
                     * (nlev < 0) ARE filled here — Phase 3.5 extrapolated
                     * the coarse_buf boundary, so the prolongation stencil
                     * has valid data. Without this, fine boundary ghosts
                     * stay zero → NaN in the RHS from chi=0, h_ij=0. */
                    int nlev = b->nblevel[oz + 1][oy + 1][ox + 1];
                    if (nlev == b->loc.level) continue;  /* same-level */

                    /* Map fine index to coarse_buf continuous index */
                    double ci_cont = (fi - ghost_f + 0.5) / 2.0
                                     + ghost_c - 0.5;
                    double cj_cont = (fj - ghost_f + 0.5) / 2.0
                                     + ghost_c - 0.5;
                    double ck_cont = (fk - ghost_f + 0.5) / 2.0
                                     + ghost_c - 0.5;

                    /* Nearest coarse cell center */
                    int ci0 = (int)(ci_cont + 0.5);
                    int cj0 = (int)(cj_cont + 0.5);
                    int ck0 = (int)(ck_cont + 0.5);

                    /* Skip if stencil extends outside coarse_buf */
                    if (ci0 < half || ci0 >= cg->Ntotal - half ||
                        cj0 < half || cj0 >= cg->Ntotal - half ||
                        ck0 < half || ck0 >= cg->Ntotal - half)
                        continue;

                    /* Which child of coarse cell: left (0) or right (1) */
                    int oi = (ci_cont >= ci0) ? 1 : 0;
                    int oj = (cj_cont >= cj0) ? 1 : 0;
                    int ok = (ck_cont >= ck0) ? 1 : 0;

                    /* 5×5×5 tensor product Lagrange interpolation
                     * Ref: AthenaK prolongation.hpp ProlongInterpolation */
                    double val = 0.0;
                    for (int sk = 0; sk < PROLONG_STENCIL; sk++) {
                        int wk = ok ? (PROLONG_STENCIL - 1 - sk) : sk;
                        for (int sj = 0; sj < PROLONG_STENCIL; sj++) {
                            int wj = oj ? (PROLONG_STENCIL - 1 - sj) : sj;
                            double wkj = prolong_w[wk] * prolong_w[wj];
                            for (int si = 0; si < PROLONG_STENCIL; si++) {
                                int wi = oi ?
                                    (PROLONG_STENCIL - 1 - si) : si;
                                int src_idx = IDX(cg,
                                    ci0 - half + si,
                                    cj0 - half + sj,
                                    ck0 - half + sk);
                                val += wkj * prolong_w[wi] * src[src_idx];
                            }
                        }
                    }

                    fg->fields[f][IDX(fg, fi, fj, fk)] = val;
                }
            }
        }
    }
}

/* ========================================================================
 * Temporal interpolation for subcycling (Berger-Oliger)
 * ======================================================================== */

/*
 * Fill fine-level ghost zones from time-interpolated coarse neighbors.
 *
 * For each fine leaf block at fine_level:
 *   1. For each of 26 neighbor directions where nblevel = fine_level - 1:
 *      - Time-interpolate coarser neighbor (frac between old and new states)
 *      - Copy interpolated data into fine block's coarse_buf ghost zones
 *   2. Fill coarse_buf boundary ghost cells by extrapolation
 *   3. Prolongate from coarse_buf into fine ghost zones
 *
 * The interpolation fraction frac is relative to the coarse step:
 *   frac = 0.0 → use coarse old state
 *   frac = 1.0 → use coarse new state
 *   frac = 0.5 → midpoint interpolation
 *
 * Ref: Athena++ MeshRefinement::ProlongateBoundaries() temporal interpolation.
 * Ref: Chombo AMR::timeStep() coarse-fine boundary fill.
 */
void ghost_fill_from_coarser(mesh_t *m, int fine_level, double frac)
{
    if (fine_level <= 0) return;

    /* Pass 1: Restrict ALL fine blocks' interiors into their coarse_bufs.
     * After this pass, every coarse_buf at fine_level has valid interior data.
     * Must complete before Pass 2 so same-level neighbor exchanges read
     * restricted (not stale) coarse_buf data. */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf || b->loc.level != fine_level) continue;
        if (!b->coarse_buf) continue;
        restrict_to_coarse_buf(b);
    }

    /* Pass 2: Exchange coarse_buf ghosts + boundary extrapolate + prolongate.
     * Safe to read neighbors' coarse_bufs — they were all restricted in Pass 1. */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf || b->loc.level != fine_level) continue;
        if (!b->coarse_buf) continue;

        /* Fill coarse_buf ghosts from same-level siblings and coarser neighbors */
        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int ox = nbr_offset[n][0];
            int oy = nbr_offset[n][1];
            int oz = nbr_offset[n][2];
            int nlev = b->nblevel[oz + 1][oy + 1][ox + 1];

            if (nlev == b->loc.level) {
                /* Same-level neighbor: exchange coarse_buf ↔ coarse_buf */
                int nbr_id = b->neighbor_ids[n];
                if (nbr_id < 0) continue;
                block_t *nbr = m->blocks[nbr_id];
                if (!nbr || !nbr->coarse_buf) continue;
                exchange_grid_pair(b->coarse_buf, nbr->coarse_buf,
                                   ox, oy, oz);

            } else if (nlev >= 0 && nlev == b->loc.level - 1) {
                /* Coarser neighbor: need temporal interpolation.
                 * If neighbor has fields_old, interpolate; otherwise use current. */
                int nbr_id = b->neighbor_ids[n];
                if (nbr_id < 0) continue;
                block_t *nbr = m->blocks[nbr_id];
                if (!nbr || !nbr->grid) continue;

                if (nbr->fields_old_block) {
                    /* Time-interpolate coarse neighbor into a temporary,
                     * then copy from that temporary into coarse_buf ghosts.
                     * We do this in-place: temporarily modify nbr->grid->fields,
                     * do the copy, then restore. Instead, we use a simpler approach:
                     * interpolate directly into the coarse_buf ghost cells. */

                    /* Use copy_from_coarse_grid approach but with interpolated data.
                     * We temporarily point nbr's grid fields to interpolated data. */
                    grid_t *cg = nbr->grid;
                    size_t npts = cg->npoints;

                    /* Allocate temporary for interpolated fields */
                    double *interp_ptrs[NUM_FIELDS];
                    double *interp_block = malloc((size_t)cg->n_fields * npts
                                                   * sizeof(double));
                    if (!interp_block) {
                        fprintf(stderr, "ghost_fill_from_coarser: malloc failed\n");
                        exit(1);
                    }
                    for (int f = 0; f < cg->n_fields; f++)
                        interp_ptrs[f] = interp_block + f * npts;

                    block_time_interp(nbr, frac, interp_ptrs, npts);

                    /* Temporarily swap fields to interpolated data for copy */
                    double *saved_fields[NUM_FIELDS];
                    for (int f = 0; f < cg->n_fields; f++) {
                        saved_fields[f] = cg->fields[f];
                        cg->fields[f] = interp_ptrs[f];
                    }

                    copy_from_coarse_grid(b->coarse_buf, b->origin,
                                          cg, nbr->origin, ox, oy, oz);

                    /* Restore original fields */
                    for (int f = 0; f < cg->n_fields; f++)
                        cg->fields[f] = saved_fields[f];

                    free(interp_block);
                } else {
                    /* No old state: use current coarse data as-is
                     * (first step or uniform mesh) */
                    copy_from_coarse_grid(b->coarse_buf, b->origin,
                                          nbr->grid, nbr->origin,
                                          ox, oy, oz);
                }
            }
        }

        /* Phase 3.5: Extrapolate boundary ghost cells of coarse_buf */
        fill_coarse_buf_boundary(b);

        /* Phase 4: Prolongate coarse_buf → fine ghost zones */
        prolongate_from_own_coarse_buf(b);
    }
}

/* ========================================================================
 * Public API
 * ======================================================================== */

void ghost_exchange(mesh_t *m)
{
    /* Loop over all leaf blocks */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;

        /* Exchange with each of 26 neighbors */
        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int nbr_id = b->neighbor_ids[n];
            if (nbr_id < 0) continue;  /* physical boundary — skip */

            block_t *nbr = m->blocks[nbr_id];
            if (!nbr) continue;
            exchange_neighbor(b, nbr,
                              nbr_offset[n][0],
                              nbr_offset[n][1],
                              nbr_offset[n][2],
                              0);  /* exchange fields[] */
        }
    }
}

void ghost_exchange_array(mesh_t *m, int src_field)
{
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;

        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int nbr_id = b->neighbor_ids[n];
            if (nbr_id < 0) continue;

            block_t *nbr = m->blocks[nbr_id];
            if (!nbr) continue;
            exchange_neighbor(b, nbr,
                              nbr_offset[n][0],
                              nbr_offset[n][1],
                              nbr_offset[n][2],
                              src_field);
        }
    }
}

/*
 * Multi-level ghost exchange (Stage 4.1): coarse-buffer architecture.
 *
 * Phase 0+1: Same-level exchange at each level (coarsest first)
 *            — fills fine sibling ghost zones and root block ghosts
 * Phase 2:   Restrict fine → own coarse_buf (4th-order, block-local)
 * Phase 3:   Fill coarse_buf ghost zones from sibling bufs + coarse nbrs
 * Phase 3.5: Fill coarse_buf boundary ghost cells by extrapolation
 * Phase 4:   Prolongate own coarse_buf → fine ghost zones
 *            — only cells where same-level neighbor doesn't exist
 *
 * No cross-block memory writes. No parent dependency.
 * Each phase operates on disjoint cell sets.
 *
 * Ref: AthenaK coarse-buffer architecture
 * Ref: GRChombo GRAMRLevel.cpp:1029-1043 (inter-level first, then intra)
 */
void ghost_exchange_multilevel(mesh_t *m)
{
    /* Fast path for uniform grids */
    if (m->max_level == 0) {
        ghost_exchange(m);
        return;
    }

    /* Phase 0+1: Same-level exchange at each level */
    for (int L = 0; L <= m->max_level; L++)
        exchange_same_level(m, L);

    /* Phase 2: Restrict fine → own coarse_buf */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf || b->loc.level == 0) continue;
        restrict_to_coarse_buf(b);
    }

    /* Phase 3: Fill coarse_buf ghost zones */
    fill_coarse_buf_ghosts(m);

    /* Phase 3.5: Fill coarse_buf boundary ghost cells by extrapolation.
     * Ref: AthenaK ApplyPhysicalBoundariesOnCoarseLevel */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf || b->loc.level == 0) continue;
        fill_coarse_buf_boundary(b);
    }

    /* Phase 4: Prolongate coarse_buf → fine ghost zones */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf || b->loc.level == 0) continue;
        prolongate_from_own_coarse_buf(b);
    }
}
