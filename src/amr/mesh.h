/*
 * Lattice — 3D Numerical Relativity
 * AMR mesh: manages collection of blocks in an oct-tree.
 *
 * Single root block at level 0, refined by oct-tree AMR.
 *
 * Follows Athena++ Mesh pattern (Stone et al. 2020, ApJS 249, 4):
 *   - Single root block at level 0
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
#include "meshblock_pack.h"
#include "../core/params.h"
#include "../core/device.h"

EXTERN_C_BEGIN

#define MAX_AMR_LEVELS 16

typedef struct mesh_s {
    block_t   **blocks;       /* array of all blocks (indexed by ID)       */
    int         num_blocks;   /* total blocks (all are leaves in Stage 1)  */
    int         max_blocks;   /* allocated capacity                        */
    int         max_level;    /* deepest refinement level present          */

    int         N_block;      /* interior cells per block side (e.g. 32)   */
    double      L;            /* physical domain size                      */
    double      dx_base;      /* coarsest dx = L / N_block                 */
    rk_method_t rk_method;    /* time integrator for block allocation      */
    int         n_fields;     /* active fields per block (<= NUM_FIELDS)   */

    /* Pre-allocated scratch for ghost_fill_from_coarser temporal interp.
     * Size = n_fields * block_npoints. Eliminates malloc/free per exchange. */
    double     *ghost_scratch;
    size_t      ghost_scratch_size;

    /* Persistent packs: cached across time steps to eliminate per-step
     * malloc/free/metadata overhead. Only the data buffer is synced in/out.
     * leaf_pack: uniform mesh (max_level == 0). One pack for all leaves.
     * level_packs[L]: AMR subcycling. One pack per level L.
     * packs_dirty: set to 1 on regrid/init to force pack rebuild. */
    meshblock_pack_t *leaf_pack;
    meshblock_pack_t *level_packs[MAX_AMR_LEVELS];
    int         packs_dirty;
} mesh_t;

/*
 * Create a mesh with a single root block.
 *   N_block:  interior cells per block side (e.g. 32)
 *   L:        physical domain size
 *   method:   RK method (determines memory allocation per block)
 *   n_fields: active field count (<= NUM_FIELDS)
 *
 * The root block is assigned ID 0 with all 6 faces on the boundary.
 * AMR refinement adds child blocks via oct-tree splitting.
 */
mesh_t *mesh_create_ex(int N_block, double L, rk_method_t method,
                       int n_fields);

/* Wrapper: creates mesh with all NUM_FIELDS fields. */
static inline mesh_t *mesh_create(int N_block, double L,
                                  rk_method_t method)
{
    return mesh_create_ex(N_block, L, method, NUM_FIELDS);
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

EXTERN_C_END

#endif /* LATTICE_MESH_H */
