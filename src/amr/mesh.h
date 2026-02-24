/*
 * Lattice — 3D Numerical Relativity
 * AMR mesh: manages collection of blocks in an oct-tree.
 *
 * Stage 1: uniform level-0 decomposition (N_root^3 blocks).
 * Later stages add refinement, multi-level, subcycling.
 *
 * Follows Athena++ Mesh pattern (Stone et al. 2020, ApJS 249, 4):
 *   - Root grid of N_root^3 blocks at level 0
 *   - Blocks sorted by Morton (Z-order) for cache locality
 *   - 26 neighbors computed from logical coordinates
 *   - nblevel[3][3][3] table set per block
 *
 * Ref: Athena++ src/mesh/mesh.hpp
 * Ref: AthenaK  src/mesh/mesh.hpp
 */

#ifndef LATTICE_MESH_H
#define LATTICE_MESH_H

#include "block.h"
#include "../core/params.h"

typedef struct mesh_s {
    block_t   **blocks;       /* array of all blocks (indexed by ID)       */
    int         num_blocks;   /* total blocks (all are leaves in Stage 1)  */
    int         max_blocks;   /* allocated capacity                        */
    int         max_level;    /* deepest refinement level present          */

    int         N_block;      /* interior cells per block side (e.g. 32)   */
    double      L;            /* physical domain size                      */
    double      dx_base;      /* coarsest dx = L / (N_root * N_block)      */
    int         N_root;       /* root blocks per side                      */
    rk_method_t rk_method;    /* time integrator for block allocation      */
    int         n_fields;     /* active fields per block (<= NUM_FIELDS)   */
} mesh_t;

/*
 * Create a uniform mesh with explicit field count.
 *   N_root:   root blocks per side (e.g. 1 for single block, 4 for 64 blocks)
 *   N_block:  interior cells per block side (e.g. 32)
 *   L:        physical domain size
 *   method:   RK method (determines memory allocation per block)
 *   n_fields: active field count (<= NUM_FIELDS)
 *
 * Blocks are created in Morton (Z-order) and assigned sequential IDs.
 * All 26 neighbors + nblevel tables are computed.
 * Boundary faces flagged for Sommerfeld BCs.
 *
 * Effective resolution: N_eff = N_root * N_block per side.
 */
mesh_t *mesh_create_ex(int N_root, int N_block, double L, rk_method_t method,
                       int n_fields);

/* Backward-compatible wrapper: creates mesh with all NUM_FIELDS fields. */
static inline mesh_t *mesh_create(int N_root, int N_block, double L,
                                  rk_method_t method)
{
    return mesh_create_ex(N_root, N_block, L, method, NUM_FIELDS);
}

/* Free mesh and all its blocks */
void mesh_free(mesh_t *m);

/* Get block by ID (bounds-checked in debug builds) */
static inline block_t *mesh_get_block(const mesh_t *m, int id)
{
#ifdef DEBUG
    if (id < 0 || id >= m->num_blocks) {
        fprintf(stderr, "mesh_get_block: id=%d out of range [0,%d)\n",
                id, m->num_blocks);
        exit(1);
    }
#endif
    return m->blocks[id];
}

/* Number of leaf (active) blocks */
int mesh_num_leaves(const mesh_t *m);

/*
 * Find a block by logical location. Linear scan of blocks[].
 * Returns NULL if not found. Fine for <1000 blocks.
 * Ref: Athena++ MeshBlockTree::FindNeighbor (tree walk variant)
 */
block_t *mesh_find_block(const mesh_t *m, int level, int lx1, int lx2, int lx3);

/*
 * Append a block to the mesh. Grows blocks[] if needed.
 * Returns the slot index where the block was placed.
 * The block's id is set to the slot index.
 */
int mesh_add_block(mesh_t *m, block_t *b);

/*
 * Remove a block by slot index: sets blocks[id] to NULL.
 * Does not free the block — caller must do that.
 */
void mesh_remove_block(mesh_t *m, int id);

/*
 * Compact the blocks array: remove NULL slots, reassign IDs.
 * Updates parent_id, child_ids, neighbor_ids to reflect new IDs.
 * Sorts remaining blocks by Morton code within each level.
 */
void mesh_compact(mesh_t *m);

/*
 * Find the finest-level leaf block whose interior contains point (x,y,z).
 * Linear scan of leaf blocks. Returns NULL if outside all blocks.
 * O(n_blocks) per call — adequate for AH finder (~1000 calls × ~100 blocks).
 */
block_t *mesh_find_block_at(const mesh_t *m, double x, double y, double z);

/*
 * Rebuild neighbor_ids and nblevel tables for all blocks.
 * Handles multi-level meshes: for each block, searches all 26 directions
 * for same-level neighbors, falls back to coarser-level blocks.
 * Also recomputes on_boundary flags and updates max_level.
 *
 * Ref: Athena++ bvals_base.cpp SearchAndSetNeighbors
 */
void mesh_rebuild_neighbors(mesh_t *m);

/*
 * Find the leaf block containing physical coordinates (x, y, z).
 * Linear scan of leaf blocks; returns finest-level block when multiple
 * blocks (coarse overlapping fine) contain the point.
 * Returns NULL if point is outside the domain.
 *
 * Fine for <1000 blocks (same as mesh_find_block).
 */
block_t *mesh_find_block_at(const mesh_t *m, double x, double y, double z);

#endif /* LATTICE_MESH_H */
