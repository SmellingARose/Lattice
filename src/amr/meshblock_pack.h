/*
 * Lattice — 3D Numerical Relativity
 * MeshBlockPack: contiguous buffer for GPU-batched kernel launches.
 *
 * Groups blocks into a single contiguous allocation so one GPU kernel
 * covers all blocks. This is the AthenaK pattern (Grete et al. 2024,
 * arXiv:2409.16053) adapted to our SoA layout.
 *
 * Memory layout (field-major, block-minor, point-innermost):
 *   data[f * n_blocks * npts + b * npts + idx]
 *
 * This gives unit-stride access for adjacent spatial points within
 * the same (field, block), matching our x-innermost convention.
 *
 * AthenaK uses (nmb, nvar, nz, ny, nx) ordering with block outermost.
 * We use field outermost because our OpenMP target loop is:
 *   for (bidx = 0; bidx < n_blocks * npts; bidx++)
 * and we want all fields for one point to be computed together.
 *
 * Ref: AthenaK src/mesh/meshblock_pack.hpp
 * Ref: Parthenon-VIBE arXiv:2509.19701 (block size GPU benchmarks)
 */

#ifndef LATTICE_MESHBLOCK_PACK_H
#define LATTICE_MESHBLOCK_PACK_H

#include "block.h"
#include "../core/fields.h"
#include <stddef.h>

/*
 * Buffer index constants for backend_zero_packed / backend_copy_packed.
 *   0 = data (evolved fields U)
 *   1 = rhs  (right-hand-side F)
 *   2 = scratch (CK45 dU or classic RK4 backup U^0)
 *   3 = accum (classic RK4 accumulator, NULL if CK45)
 */
#define PACK_BUF_DATA    0
#define PACK_BUF_RHS     1
#define PACK_BUF_SCRATCH 2
#define PACK_BUF_ACCUM   3

typedef struct {
    /* ---- core field buffers (existing) ----
     * Layout: [n_fields * n_blocks * npts], field-major, block-minor,
     * point-innermost (x unit-stride). Matches AthenaK convention. */
    double  *data;         /* evolved fields U                              */
    double  *rhs;          /* right-hand-side F                             */
    double  *scratch;      /* CK45 dU register, or classic RK4 backup U^0  */
    double  *accum;        /* classic RK4 accumulator (NULL if CK45)        */

    int      n_blocks;     /* number of blocks in this pack                */
    size_t   npts;         /* grid points per block = Ntotal^3             */
    int      n_fields;     /* active fields per block (<= NUM_FIELDS)      */

    int     *block_ids;    /* which block IDs are packed [n_blocks]         */
    int      level;        /* level of blocks in this pack (-1 = mixed)    */

    /* ---- grid dimensions (same for all blocks in pack) ----
     * Set by meshblock_pack_load_meta from the first block's grid. */
    int      N;            /* interior cells per side                       */
    int      ghost;        /* = GHOST_WIDTH = 4                             */
    int      Ntotal;       /* N + 2*ghost (may be padded per grid_alloc)    */

    /* ---- per-block metadata (flat GPU-mappable arrays) ----
     * Filled by meshblock_pack_load_meta and meshblock_pack_build_neighbors.
     * These arrays are mapped to GPU memory by backend_map_pack. */
    double  *origins;       /* [n_blocks * 3] — block origin (x,y,z)        */
    double  *dx_per_block;  /* [n_blocks] — dx varies by refinement level   */
    int     *on_boundary;   /* [n_blocks * 6] — domain boundary flags       */
                            /* [b*6+0..5] = {x-,x+,y-,y+,z-,z+}            */
    int     *levels;        /* [n_blocks] — refinement level per block      */
    int     *neighbor_table;/* [n_blocks * 26] — pack-local neighbor index   */
                            /* -1 = no neighbor in pack (boundary or absent) */

    /* ---- coarse_buf data for multilevel ghost exchange (Commit 2) ----
     * Each leaf block at level > 0 has a half-resolution coarse_buf used
     * by the 5-phase multilevel ghost exchange. These arrays pack all
     * coarse_bufs contiguously for GPU kernels.
     * Ref: AthenaK coarse-buffer architecture */
    double  *coarse_data;   /* [n_refined * n_fields * coarse_npts]          */
    int      n_refined;     /* count of level > 0 blocks (have coarse_bufs) */
    size_t   coarse_npts;   /* points per coarse_buf = coarse_Ntotal^3       */
    int      coarse_Ntotal; /* N/2 + 2*ghost                                */
    int      coarse_N;      /* N/2 (half the fine block interior)            */
    int     *refined_map;   /* [n_blocks] — pack index → coarse_data index   */
                            /* -1 if level == 0 (no coarse_buf for this blk) */
    int     *coarse_neighbor_table; /* [n_refined * 26] — coarse_buf nbrs    */

    /* ---- neighbor level table for ghost exchange phases 3.5 + 4 ----
     * Flattened nblevel[3][3][3] per block. Index:
     *   nblevel_table[b * 27 + (oz+1)*9 + (oy+1)*3 + (ox+1)]
     * Values: neighbor level, or -1 = domain boundary.
     * nblevel_table[b*27 + 13] = self level (center element). */
    int     *nblevel_table;   /* [n_blocks * 27]                           */
} meshblock_pack_t;

/* Pack indexing macro:
 *   PACK_IDX(pack, f, b, idx) -> flat offset into pack->data/rhs/scratch
 * where f = field index, b = block index within pack, idx = point index */
#define PACK_IDX(pack, f, b, idx) \
    ((size_t)(f) * (pack)->n_blocks * (pack)->npts \
     + (size_t)(b) * (pack)->npts + (size_t)(idx))

/*
 * Allocate a MeshBlockPack for n_blocks blocks, each with npts grid points.
 * Allocates page-aligned contiguous buffers for data, rhs, scratch, and
 * accum (if rk_method == RK_CLASSIC). Also allocates per-block metadata
 * arrays (origins, dx_per_block, etc.) — caller fills via load_meta.
 *
 * block_ids: array of mesh-level block IDs to include [n_blocks].
 * level: refinement level (-1 = mixed levels, normal for AMR).
 * rk_method: determines whether to allocate accum buffer.
 * n_fields: active field count (<= NUM_FIELDS).
 */
meshblock_pack_t *meshblock_pack_create(int n_blocks, size_t npts,
                                         const int *block_ids, int level,
                                         rk_method_t rk_method, int n_fields);

/* Free pack and all its buffers (data, rhs, scratch, accum, metadata) */
void meshblock_pack_free(meshblock_pack_t *pack);

/*
 * Copy field data from individual blocks into the pack's contiguous buffer.
 * blocks: array of all blocks (indexed by block_ids[b]).
 * Copies: fields→data, rhs→rhs, scratch→scratch, accum→accum (if non-NULL).
 */
void meshblock_pack_load(meshblock_pack_t *pack, block_t **blocks);

/*
 * Copy field data from pack back into individual blocks.
 * Inverse of meshblock_pack_load.
 */
void meshblock_pack_store(const meshblock_pack_t *pack, block_t **blocks);

/*
 * Fill per-block metadata from block_t structs into flat GPU-mappable arrays.
 * Sets: origins, dx_per_block, on_boundary, levels, grid dimensions (N,
 * ghost, Ntotal). Also counts refined blocks and allocates coarse_data
 * if any blocks are at level > 0.
 *
 * Must be called after meshblock_pack_create and before build_neighbors.
 */
void meshblock_pack_load_meta(meshblock_pack_t *pack, block_t **blocks);

/*
 * Build pack-local neighbor table from block_t neighbor_ids.
 * For each block b and each of 26 directions, maps the mesh-level
 * neighbor_id to a pack-local index (0..n_blocks-1) or -1 if the
 * neighbor isn't in the pack (domain boundary or different level).
 *
 * Also builds coarse_neighbor_table for refined blocks if n_refined > 0.
 * Must be called after meshblock_pack_load_meta.
 */
void meshblock_pack_build_neighbors(meshblock_pack_t *pack, block_t **blocks);

/*
 * Copy coarse_buf data from individual blocks into pack->coarse_data.
 * Only copies for refined blocks (level > 0, refined_map[b] >= 0).
 * Used to initialize pack coarse data before GPU ghost exchange (Commit 2).
 */
void meshblock_pack_load_coarse(meshblock_pack_t *pack, block_t **blocks);

/*
 * Copy pack->coarse_data back into individual blocks' coarse_bufs.
 * Inverse of meshblock_pack_load_coarse.
 */
void meshblock_pack_store_coarse(const meshblock_pack_t *pack, block_t **blocks);

#endif /* LATTICE_MESHBLOCK_PACK_H */
