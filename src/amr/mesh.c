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
        block_free(m->blocks[i]);
    }
    free(m->blocks);
    free(m);
}

int mesh_num_leaves(const mesh_t *m)
{
    int count = 0;
    for (int i = 0; i < m->num_blocks; i++) {
        if (m->blocks[i]->is_leaf)
            count++;
    }
    return count;
}
