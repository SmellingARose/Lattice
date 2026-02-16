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
        if (!b->is_leaf) continue;

        /* Exchange with each of 26 neighbors */
        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int nbr_id = b->neighbor_ids[n];
            if (nbr_id < 0) continue;  /* physical boundary — skip */

            block_t *nbr = m->blocks[nbr_id];
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
        if (!b->is_leaf) continue;

        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int nbr_id = b->neighbor_ids[n];
            if (nbr_id < 0) continue;

            block_t *nbr = m->blocks[nbr_id];
            exchange_neighbor(b, nbr,
                              nbr_offset[n][0],
                              nbr_offset[n][1],
                              nbr_offset[n][2],
                              src_field);
        }
    }
}
