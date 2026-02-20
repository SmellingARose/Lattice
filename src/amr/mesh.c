/*
 * Lattice — 3D Numerical Relativity
 * AMR mesh creation and management.
 *
 * Stage 1: uniform level-0 decomposition.
 *   1. Allocate N_root^3 blocks in Morton (Z-order)
 *   2. Compute 26 neighbors from logical coordinates
 *   3. Set nblevel[3][3][3] tables and boundary flags
 *
 * Neighbor finding follows Athena++ bvals_base.cpp:
 *   SearchAndSetNeighbors — for uniform grids, this reduces
 *   to simple coordinate offset + bounds check.
 *
 * Ref: Athena++ src/mesh/mesh.cpp (Mesh constructor, root grid setup)
 * Ref: Athena++ src/bvals/bvals_base.cpp (SearchAndSetNeighbors)
 */

#include "mesh.h"
#include "morton.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Helper: find block ID at logical coordinates (lx1, lx2, lx3)
 * in a uniform N_root^3 root grid.
 *
 * Uses the same ordering as the creation loop:
 *   ID = lookup[morton_encode3d(lx1, lx2, lx3)]
 *
 * For Stage 1, we use a direct lookup table built during creation.
 * For later stages with non-uniform grids, this becomes a tree walk
 * (Athena++ MeshBlockTree::FindNeighbor).
 */

/* Comparison function for sorting blocks by Morton code (Z-order).
 * Ref: Athena++ MeshBlockTree::GetMeshBlockList — assigns GIDs in Z-order. */
static int compare_morton(const void *a, const void *b)
{
    const int64_t ma = (*(const block_t **)a)->loc.level >= 0
                       ? morton_encode3d((*(const block_t **)a)->loc.lx1,
                                         (*(const block_t **)a)->loc.lx2,
                                         (*(const block_t **)a)->loc.lx3)
                       : 0;
    const int64_t mb = (*(const block_t **)b)->loc.level >= 0
                       ? morton_encode3d((*(const block_t **)b)->loc.lx1,
                                         (*(const block_t **)b)->loc.lx2,
                                         (*(const block_t **)b)->loc.lx3)
                       : 0;
    return (ma > mb) - (ma < mb);
}

mesh_t *mesh_create(int N_root, int N_block, double L, rk_method_t method)
{
    mesh_t *m = calloc(1, sizeof(mesh_t));
    if (!m) {
        fprintf(stderr, "mesh_create: calloc failed\n");
        exit(1);
    }

    m->N_root    = N_root;
    m->N_block   = N_block;
    m->L         = L;
    m->rk_method = method;
    m->dx_base   = L / ((double)N_root * N_block);
    m->max_level = 0;

    int total = N_root * N_root * N_root;
    m->num_blocks = total;
    m->max_blocks = total * 2;  /* headroom for future refinement */
    m->blocks = calloc(m->max_blocks, sizeof(block_t *));
    if (!m->blocks) {
        fprintf(stderr, "mesh_create: calloc blocks failed\n");
        exit(1);
    }

    printf("[AMR] Creating mesh: N_root=%d, N_block=%d, L=%.1f, dx=%.6f\n",
           N_root, N_block, L, m->dx_base);
    printf("[AMR] Total blocks: %d, effective N=%d per side\n",
           total, N_root * N_block);

    /* --- Phase 1: Allocate all root blocks --- */
    /* Create blocks in simple (iz, iy, ix) order first, then sort by Morton */
    block_t **tmp_blocks = calloc(total, sizeof(block_t *));
    if (!tmp_blocks) {
        fprintf(stderr, "mesh_create: calloc tmp_blocks failed\n");
        exit(1);
    }

    double block_L = N_block * m->dx_base;  /* physical size of one block */

    for (int iz = 0; iz < N_root; iz++) {
        for (int iy = 0; iy < N_root; iy++) {
            for (int ix = 0; ix < N_root; ix++) {
                int idx = (iz * N_root + iy) * N_root + ix;
                double origin[3] = {
                    ix * block_L - L / 2.0,
                    iy * block_L - L / 2.0,
                    iz * block_L - L / 2.0
                };

                tmp_blocks[idx] = block_alloc(idx, 0, N_block, m->dx_base,
                                               origin, method);
                tmp_blocks[idx]->loc.lx1  = ix;
                tmp_blocks[idx]->loc.lx2  = iy;
                tmp_blocks[idx]->loc.lx3  = iz;
                tmp_blocks[idx]->loc.level = 0;
            }
        }
    }

    /* Sort by Morton code (Z-order) for cache-friendly traversal.
     * Ref: Athena++ MeshBlockTree::GetMeshBlockList — GIDs assigned in Z-order */
    qsort(tmp_blocks, total, sizeof(block_t *), compare_morton);

    /* Assign final IDs after Morton sort */
    for (int i = 0; i < total; i++) {
        tmp_blocks[i]->id = i;
        m->blocks[i] = tmp_blocks[i];
    }
    free(tmp_blocks);

    /* --- Phase 2: Build coordinate -> ID lookup for neighbor finding ---
     * For uniform level-0, this is a simple 3D array.
     * Athena++ uses tree walk (FindNeighbor); we use direct lookup at level 0. */
    int *coord_to_id = calloc(total, sizeof(int));
    if (!coord_to_id) {
        fprintf(stderr, "mesh_create: calloc coord_to_id failed\n");
        exit(1);
    }
    /* Initialize to -1 */
    memset(coord_to_id, -1, total * sizeof(int));

    for (int i = 0; i < total; i++) {
        block_t *b = m->blocks[i];
        int flat = (b->loc.lx3 * N_root + b->loc.lx2) * N_root + b->loc.lx1;
        coord_to_id[flat] = b->id;
    }

    /* --- Phase 3: Set 26 neighbors, nblevel, boundary flags ---
     * Ref: Athena++ bvals_base.cpp SearchAndSetNeighbors
     * For uniform grid: neighbor at (lx1+ox1, lx2+ox2, lx3+ox3),
     * boundary if out of [0, N_root). */
    for (int i = 0; i < total; i++) {
        block_t *b = m->blocks[i];
        int ix = b->loc.lx1;
        int iy = b->loc.lx2;
        int iz = b->loc.lx3;

        /* 26 neighbors */
        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int nx = ix + nbr_offset[n][0];
            int ny = iy + nbr_offset[n][1];
            int nz = iz + nbr_offset[n][2];

            if (nx >= 0 && nx < N_root &&
                ny >= 0 && ny < N_root &&
                nz >= 0 && nz < N_root) {
                int flat = (nz * N_root + ny) * N_root + nx;
                b->neighbor_ids[n] = coord_to_id[flat];
            } else {
                b->neighbor_ids[n] = -1;  /* physical boundary */
            }
        }

        /* nblevel[3][3][3] table (Athena++ pattern).
         * [oz+1][oy+1][ox+1] = level of neighbor at offset (ox,oy,oz).
         * -1 = physical boundary, 0 = level 0 (all same in Stage 1). */
        for (int oz = -1; oz <= 1; oz++) {
            for (int oy = -1; oy <= 1; oy++) {
                for (int ox = -1; ox <= 1; ox++) {
                    if (ox == 0 && oy == 0 && oz == 0) {
                        b->nblevel[oz + 1][oy + 1][ox + 1] = b->loc.level;
                        continue;
                    }
                    int nx = ix + ox;
                    int ny = iy + oy;
                    int nz = iz + oz;
                    if (nx >= 0 && nx < N_root &&
                        ny >= 0 && ny < N_root &&
                        nz >= 0 && nz < N_root) {
                        b->nblevel[oz + 1][oy + 1][ox + 1] = 0;  /* level 0 */
                    } else {
                        b->nblevel[oz + 1][oy + 1][ox + 1] = -1;
                    }
                }
            }
        }

        /* Boundary flags: 6 faces.
         * [0]=x-, [1]=x+, [2]=y-, [3]=y+, [4]=z-, [5]=z+ */
        b->on_boundary[0] = (ix == 0)          ? 1 : 0;
        b->on_boundary[1] = (ix == N_root - 1) ? 1 : 0;
        b->on_boundary[2] = (iy == 0)          ? 1 : 0;
        b->on_boundary[3] = (iy == N_root - 1) ? 1 : 0;
        b->on_boundary[4] = (iz == 0)          ? 1 : 0;
        b->on_boundary[5] = (iz == N_root - 1) ? 1 : 0;
    }

    free(coord_to_id);

    printf("[AMR] Mesh created: %d blocks, Morton-sorted, neighbors set\n",
           total);

    return m;
}

void mesh_free(mesh_t *m)
{
    if (!m) return;
    for (int i = 0; i < m->num_blocks; i++) {
        if (m->blocks[i])
            block_free(m->blocks[i]);
    }
    free(m->blocks);
    free(m);
}

int mesh_num_leaves(const mesh_t *m)
{
    int count = 0;
    for (int i = 0; i < m->num_blocks; i++) {
        if (m->blocks[i] && m->blocks[i]->is_leaf)
            count++;
    }
    return count;
}

block_t *mesh_find_block(const mesh_t *m, int level, int lx1, int lx2, int lx3)
{
    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (!b) continue;
        if (b->loc.level == level &&
            b->loc.lx1 == lx1 &&
            b->loc.lx2 == lx2 &&
            b->loc.lx3 == lx3)
            return b;
    }
    return NULL;
}

block_t *mesh_find_block_at(const mesh_t *m, double x, double y, double z)
{
    block_t *best = NULL;
    int best_level = -1;

    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (!b || !b->is_leaf) continue;

        double dx = b->grid->dx;
        int N = b->grid->N;

        /* Check if (x,y,z) falls within this block's interior cell range */
        int inside = 1;
        for (int d = 0; d < 3; d++) {
            double lo = b->origin[d];
            double hi = b->origin[d] + N * dx;
            double coord = (d == 0) ? x : (d == 1) ? y : z;
            if (coord < lo || coord >= hi) { inside = 0; break; }
        }

        if (inside && b->loc.level > best_level) {
            best = b;
            best_level = b->loc.level;
        }
    }

    return best;
}

int mesh_add_block(mesh_t *m, block_t *b)
{
    /* Try to find an empty (NULL) slot first */
    for (int i = 0; i < m->num_blocks; i++) {
        if (!m->blocks[i]) {
            b->id = i;
            m->blocks[i] = b;
            return i;
        }
    }

    /* No empty slot — append */
    if (m->num_blocks >= m->max_blocks) {
        m->max_blocks *= 2;
        m->blocks = realloc(m->blocks, m->max_blocks * sizeof(block_t *));
        if (!m->blocks) {
            fprintf(stderr, "mesh_add_block: realloc failed\n");
            exit(1);
        }
        /* Zero new slots */
        for (int i = m->num_blocks; i < m->max_blocks; i++)
            m->blocks[i] = NULL;
    }

    int slot = m->num_blocks;
    b->id = slot;
    m->blocks[slot] = b;
    m->num_blocks++;
    return slot;
}

void mesh_remove_block(mesh_t *m, int id)
{
    if (id < 0 || id >= m->num_blocks) return;
    m->blocks[id] = NULL;  /* caller frees */
}

void mesh_compact(mesh_t *m)
{
    /* Build old_id -> new_id mapping */
    int *id_map = calloc(m->num_blocks, sizeof(int));
    if (!id_map) {
        fprintf(stderr, "mesh_compact: calloc failed\n");
        exit(1);
    }
    for (int i = 0; i < m->num_blocks; i++)
        id_map[i] = -1;

    /* Compact: move non-NULL blocks to front */
    int write = 0;
    for (int read = 0; read < m->num_blocks; read++) {
        if (m->blocks[read]) {
            id_map[read] = write;
            if (write != read) {
                m->blocks[write] = m->blocks[read];
                m->blocks[read] = NULL;
            }
            m->blocks[write]->id = write;
            write++;
        }
    }
    m->num_blocks = write;

    /* Update all internal ID references */
    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (!b) continue;

        if (b->parent_id >= 0)
            b->parent_id = id_map[b->parent_id];

        for (int c = 0; c < 8; c++) {
            if (b->child_ids[c] >= 0)
                b->child_ids[c] = id_map[b->child_ids[c]];
        }

        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            if (b->neighbor_ids[n] >= 0)
                b->neighbor_ids[n] = id_map[b->neighbor_ids[n]];
        }
    }

    free(id_map);
}

/*
 * Helper: compute the number of blocks per side at a given level.
 * At level L with N_root root blocks, there are N_root * 2^L blocks per side.
 */
static int blocks_per_side(const mesh_t *m, int level)
{
    return m->N_root * (1 << level);
}

void mesh_rebuild_neighbors(mesh_t *m)
{
    /* First pass: update max_level */
    m->max_level = 0;
    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (!b) continue;
        if (b->loc.level > m->max_level)
            m->max_level = b->loc.level;
    }

    /* Second pass: rebuild neighbors for every block */
    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (!b) continue;

        int level = b->loc.level;
        int bps = blocks_per_side(m, level);

        /* Reset neighbor arrays */
        for (int n = 0; n < NUM_NEIGHBORS; n++)
            b->neighbor_ids[n] = -1;
        memset(b->nblevel, -1, sizeof(b->nblevel));
        b->nblevel[1][1][1] = level;

        /* Boundary flags */
        b->on_boundary[0] = (b->loc.lx1 == 0)       ? 1 : 0;
        b->on_boundary[1] = (b->loc.lx1 == bps - 1) ? 1 : 0;
        b->on_boundary[2] = (b->loc.lx2 == 0)       ? 1 : 0;
        b->on_boundary[3] = (b->loc.lx2 == bps - 1) ? 1 : 0;
        b->on_boundary[4] = (b->loc.lx3 == 0)       ? 1 : 0;
        b->on_boundary[5] = (b->loc.lx3 == bps - 1) ? 1 : 0;

        /* 26 neighbors.
         * For each direction, first try same-level. If not found (refined
         * away or boundary), walk up to coarser levels by mapping the
         * NEIGHBOR'S fine-level coordinates to coarser levels via floor
         * division. This is the key difference from the naive approach
         * of dividing the block's own coords then adding the offset —
         * that fails for edge/corner directions at coarse-fine boundaries.
         *
         * Ref: Athena++ MeshBlockTree::FindNeighbor (tree walk) */
        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int ox = nbr_offset[n][0];
            int oy = nbr_offset[n][1];
            int oz = nbr_offset[n][2];

            /* Neighbor's fine-level logical coordinates */
            int nx = b->loc.lx1 + ox;
            int ny = b->loc.lx2 + oy;
            int nz = b->loc.lx3 + oz;

            /* Try same-level neighbor (must be within bounds) */
            if (nx >= 0 && nx < bps &&
                ny >= 0 && ny < bps &&
                nz >= 0 && nz < bps) {
                block_t *nbr = mesh_find_block(m, level, nx, ny, nz);
                if (nbr) {
                    b->neighbor_ids[n] = nbr->id;
                    b->nblevel[oz + 1][oy + 1][ox + 1] = level;
                    continue;
                }
            }

            /* Try coarser levels: map neighbor's fine-level coords to
             * coarser levels using floor division.
             * Floor div for negative x: ~(~x >> shift).
             * For non-negative x: simple right shift. */
            {
                int cur_level = level;
                int found = 0;

                while (cur_level > 0 && !found) {
                    cur_level--;
                    int cbps = blocks_per_side(m, cur_level);
                    int shift = level - cur_level;

                    /* Floor division by 2^shift (handles negative coords) */
                    int cnx = (nx >= 0) ? (nx >> shift) : ~(~nx >> shift);
                    int cny = (ny >= 0) ? (ny >> shift) : ~(~ny >> shift);
                    int cnz = (nz >= 0) ? (nz >> shift) : ~(~nz >> shift);

                    if (cnx < 0 || cnx >= cbps ||
                        cny < 0 || cny >= cbps ||
                        cnz < 0 || cnz >= cbps)
                        continue;

                    block_t *nbr = mesh_find_block(m, cur_level, cnx, cny, cnz);
                    if (nbr) {
                        b->neighbor_ids[n] = nbr->id;
                        b->nblevel[oz + 1][oy + 1][ox + 1] = cur_level;
                        found = 1;
                    }
                }
            }
            /* If not found at any level, remains -1 (physical boundary) */
        }
    }
}

