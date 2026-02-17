/*
 * Lattice — 3D Numerical Relativity
 * 26-neighbor ghost zone exchange (same-level, Stage 2).
 *
 * For each block and each of its 26 neighbors, copy the appropriate
 * region from the neighbor's interior into the block's ghost zone.
 *
 * The source/destination ranges are computed uniformly for all 26
 * neighbor types. For direction d with offset o in {-1, 0, +1}:
 *
 *   o == -1: ghost zone at low side  ← neighbor's high interior
 *   o ==  0: full interior range     ← same interior range
 *   o == +1: ghost zone at high side ← neighbor's low interior
 *
 * This produces: face = slab (GW × N × N), edge = column (GW × GW × N),
 * corner = cube (GW × GW × GW), matching Athena++ bvals patterns.
 *
 * All 25 fields are exchanged. Since we only read from neighbors'
 * interior cells (which are not modified by ghost exchange), the
 * iteration order over blocks does not matter.
 *
 * Ref: Athena++ src/bvals/bvals_cc.cpp — CellCenteredBoundaryVariable
 * Ref: AthenaK  src/bvals/ — GPU boundary exchange
 */

#include "ghost_exchange.h"
#include "prolongation.h"
#include "../core/fields.h"
#include <string.h>

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

    /* Copy all fields.
     * For faces (1 direction has GW width, 2 have N width), the inner
     * x loop is unit-stride, matching our SoA layout convention.
     * For edges and corners, the copy volumes are small (GW^2*N or GW^3). */
    int nx = dx_hi - dx_lo;

    for (int f = 0; f < NUM_FIELDS; f++) {
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

/* ========================================================================
 * Multi-level ghost exchange (Stage 4)
 *
 * Phase 1: Restriction — fine leaf data → non-leaf parents
 * Phase 2: Same-level exchange at each level (including non-leaf)
 * Phase 3: Prolongation — coarse parent data → fine ghost zones
 *
 * Ref: GRChombo GRAMRLevel.cpp:1029-1043 (FillPatch pattern)
 * ======================================================================== */

/*
 * Restrict all children's interior data into a non-leaf parent.
 * Each child covers one octant of the parent's interior.
 */
static void restrict_children_into_parent(const mesh_t *m, block_t *parent)
{
    grid_t *pg = parent->grid;
    const int ghost = pg->ghost;
    const int N = pg->N;
    const int half_N = N / 2;

    for (int cz = 0; cz < 2; cz++) {
        for (int cy = 0; cy < 2; cy++) {
            for (int cx = 0; cx < 2; cx++) {
                int octant = cx + (cy << 1) + (cz << 2);
                int cid = parent->child_ids[octant];
                if (cid < 0 || !m->blocks[cid]) continue;

                const block_t *child = m->blocks[cid];
                const grid_t *cg = child->grid;

                int p_off_i = cx * half_N;
                int p_off_j = cy * half_N;
                int p_off_k = cz * half_N;

                for (int f = 0; f < NUM_FIELDS; f++) {
                    for (int pk = 0; pk < half_N; pk++) {
                        for (int pj = 0; pj < half_N; pj++) {
                            for (int pi = 0; pi < half_N; pi++) {
                                double sum = 0.0;
                                for (int ok = 0; ok < 2; ok++) {
                                    for (int oj = 0; oj < 2; oj++) {
                                        for (int oi = 0; oi < 2; oi++) {
                                            int fi = cg->ghost + 2 * pi + oi;
                                            int fj = cg->ghost + 2 * pj + oj;
                                            int fk = cg->ghost + 2 * pk + ok;
                                            sum += cg->fields[f][IDX(cg, fi, fj, fk)];
                                        }
                                    }
                                }
                                int pii = ghost + p_off_i + pi;
                                int pjj = ghost + p_off_j + pj;
                                int pkk = ghost + p_off_k + pk;
                                pg->fields[f][IDX(pg, pii, pjj, pkk)] = sum * 0.125;
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 * Prolongate parent block's data into a fine child's ghost zones.
 * The parent has valid data in both interior (from restriction) and ghost
 * zones (from same-level exchange at the coarser level). This covers all
 * ghost cells of the child uniformly — faces, edges, and corners — without
 * needing per-direction neighbor lookups.
 *
 * For each fine ghost cell:
 *   1. Compute physical coordinate
 *   2. Map to parent grid index space
 *   3. If within parent's valid stencil range, apply 5-point interpolation
 *   4. Skip cells outside parent's range (domain boundary)
 */
static void prolongate_from_parent(const block_t *parent, block_t *child)
{
    grid_t *fg = child->grid;
    const grid_t *cg = parent->grid;
    const int ghost = fg->ghost;
    const int N = fg->N;
    const int Nt = fg->Ntotal;
    const double fdx = fg->dx;
    const double cdx = cg->dx;
    const int half = PROLONG_STENCIL / 2;  /* = 2 */

    for (int f = 0; f < NUM_FIELDS; f++) {
        for (int fk = 0; fk < Nt; fk++) {
            for (int fj = 0; fj < Nt; fj++) {
                for (int fi = 0; fi < Nt; fi++) {
                    /* Skip interior cells — only fill ghost zones */
                    if (fi >= ghost && fi < ghost + N &&
                        fj >= ghost && fj < ghost + N &&
                        fk >= ghost && fk < ghost + N)
                        continue;

                    /* Physical coordinate of this fine ghost cell */
                    double px = child->origin[0] + (fi - ghost + 0.5) * fdx;
                    double py = child->origin[1] + (fj - ghost + 0.5) * fdx;
                    double pz = child->origin[2] + (fk - ghost + 0.5) * fdx;

                    /* Map to parent grid index space (continuous).
                     * parent cell i has center at:
                     *   origin + (i - ghost + 0.5) * cdx
                     * So: i = (px - origin) / cdx + ghost - 0.5 */
                    double ci_cont = (px - parent->origin[0]) / cdx + cg->ghost - 0.5;
                    double cj_cont = (py - parent->origin[1]) / cdx + cg->ghost - 0.5;
                    double ck_cont = (pz - parent->origin[2]) / cdx + cg->ghost - 0.5;

                    /* Nearest coarse cell center */
                    int ci0 = (int)(ci_cont + 0.5);
                    int cj0 = (int)(cj_cont + 0.5);
                    int ck0 = (int)(ck_cont + 0.5);

                    /* Skip cells outside parent's valid stencil range.
                     * These are at the domain boundary (no data). */
                    if (ci0 < half || ci0 >= cg->Ntotal - half ||
                        cj0 < half || cj0 >= cg->Ntotal - half ||
                        ck0 < half || ck0 >= cg->Ntotal - half)
                        continue;

                    /* Which child of the coarse cell: left (0) or right (1) */
                    int oi = (ci_cont >= ci0) ? 1 : 0;
                    int oj = (cj_cont >= cj0) ? 1 : 0;
                    int ok = (ck_cont >= ck0) ? 1 : 0;

                    /* 5×5×5 tensor product interpolation */
                    double val = 0.0;
                    for (int sk = 0; sk < PROLONG_STENCIL; sk++) {
                        int wk = ok ? (PROLONG_STENCIL - 1 - sk) : sk;
                        for (int sj = 0; sj < PROLONG_STENCIL; sj++) {
                            int wj = oj ? (PROLONG_STENCIL - 1 - sj) : sj;
                            double wkj = prolong_w[wk] * prolong_w[wj];
                            for (int si = 0; si < PROLONG_STENCIL; si++) {
                                int wi = oi ? (PROLONG_STENCIL - 1 - si) : si;
                                int src_idx = IDX(cg,
                                                  ci0 - half + si,
                                                  cj0 - half + sj,
                                                  ck0 - half + sk);
                                val += wkj * prolong_w[wi] *
                                       cg->fields[f][src_idx];
                            }
                        }
                    }

                    fg->fields[f][IDX(fg, fi, fj, fk)] = val;
                }
            }
        }
    }
}

void ghost_exchange_multilevel(mesh_t *m)
{
    /* Fast path for uniform grids */
    if (m->max_level == 0) {
        ghost_exchange(m);
        return;
    }

    /* Phase 1: Restriction — fine leaf data → non-leaf parents.
     * Process from finest to coarsest so parents at each level have
     * up-to-date data from their children. */
    for (int L = m->max_level; L >= 1; L--) {
        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *b = m->blocks[bid];
            if (!b || b->loc.level != L - 1 || b->is_leaf) continue;
            /* b is non-leaf at level L-1: restrict children into b */
            restrict_children_into_parent(m, b);
        }
    }

    /* Phase 2: Same-level exchange at each level (coarsest-first).
     * Exchange between all blocks at the same level, including non-leaf
     * blocks (which now have valid data from restriction). */
    for (int L = 0; L <= m->max_level; L++) {
        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *b = m->blocks[bid];
            if (!b || b->loc.level != L) continue;

            for (int n = 0; n < NUM_NEIGHBORS; n++) {
                int nbr_id = b->neighbor_ids[n];
                if (nbr_id < 0) continue;

                block_t *nbr = m->blocks[nbr_id];
                if (!nbr || nbr->loc.level != L) continue;

                exchange_neighbor(b, nbr,
                                  nbr_offset[n][0],
                                  nbr_offset[n][1],
                                  nbr_offset[n][2],
                                  0);
            }
        }
    }

    /* Phase 3: Prolongation — parent data → fine ghost zones.
     * For each fine leaf block with a parent, prolongate from the parent's
     * grid (which has valid interior from restriction + valid ghost from
     * same-level exchange) to fill all ghost cells of the child.
     * This handles faces, edges, and corners uniformly.
     *
     * Ref: GRChombo FillPatch — coarse data fills fine ghost zones */
    for (int L = 1; L <= m->max_level; L++) {
        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *b = m->blocks[bid];
            if (!b || !b->is_leaf || b->loc.level != L) continue;
            if (b->parent_id < 0) continue;

            block_t *parent = m->blocks[b->parent_id];
            if (!parent || !parent->grid) continue;

            prolongate_from_parent(parent, b);
        }
    }

    /* Phase 4: Same-level exchange again for fine blocks.
     * After prolongation filled ghost zones from coarse data, exchange
     * between same-level siblings to get high-resolution ghost data where
     * available (overwriting the prolongated data with exact same-level data). */
    for (int L = 1; L <= m->max_level; L++) {
        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *b = m->blocks[bid];
            if (!b || !b->is_leaf || b->loc.level != L) continue;

            for (int n = 0; n < NUM_NEIGHBORS; n++) {
                int nbr_id = b->neighbor_ids[n];
                if (nbr_id < 0) continue;

                block_t *nbr = m->blocks[nbr_id];
                if (!nbr || nbr->loc.level != L) continue;

                exchange_neighbor(b, nbr,
                                  nbr_offset[n][0],
                                  nbr_offset[n][1],
                                  nbr_offset[n][2],
                                  0);
            }
        }
    }
}
