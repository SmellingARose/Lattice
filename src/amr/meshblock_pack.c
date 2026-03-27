/*
 * Lattice — 3D Numerical Relativity
 * MeshBlockPack: contiguous buffer allocation, metadata, and data transfer.
 *
 * Page-aligned allocations for zero-copy GPU mapping,
 * matching the pattern in src/core/grid.c.
 *
 * Extended for full AMR support: per-block metadata (origins, dx, boundary
 * flags, levels), pack-local neighbor tables, and coarse_buf data for
 * multilevel ghost exchange.
 *
 * Memory layout (field-major, block-minor, point-innermost):
 *   data[f * n_blocks * npts + b * npts + IDX(g, i, j, k)]
 *
 * Ref: AthenaK src/mesh/meshblock_pack.hpp (pack layout)
 * Ref: AthenaK src/mesh/meshblock.cpp (pack initialization)
 */

#include "meshblock_pack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define PAGE_ALIGN 4096

/*
 * Page-aligned allocation for GPU-mappable buffers.
 * Returns zeroed memory. Fatal on failure.
 */
static double *pack_alloc_block(size_t total_bytes)
{
    if (total_bytes == 0) return NULL;
    void *ptr = NULL;
    if (posix_memalign(&ptr, PAGE_ALIGN, total_bytes) != 0) {
        fprintf(stderr, "meshblock_pack: posix_memalign failed for %zu bytes\n",
                total_bytes);
        exit(1);
    }
    memset(ptr, 0, total_bytes);
    return (double *)ptr;
}

/*
 * Integer array allocation (not page-aligned, just calloc).
 * Used for metadata arrays that don't need GPU-aligned allocation.
 */
static int *pack_alloc_int(size_t count)
{
    if (count == 0) return NULL;
    int *ptr = calloc(count, sizeof(int));
    if (!ptr) {
        fprintf(stderr, "meshblock_pack: calloc failed for %zu ints\n", count);
        exit(1);
    }
    return ptr;
}

/*
 * Double array allocation (not page-aligned, just calloc).
 * Used for small metadata arrays (origins, dx).
 */
static double *pack_alloc_double(size_t count)
{
    if (count == 0) return NULL;
    double *ptr = calloc(count, sizeof(double));
    if (!ptr) {
        fprintf(stderr, "meshblock_pack: calloc failed for %zu doubles\n", count);
        exit(1);
    }
    return ptr;
}

/* ========================================================================
 * Creation and destruction
 * ======================================================================== */

/*
 * Allocate a MeshBlockPack for n_blocks blocks, each with npts grid points.
 *
 * Allocates:
 *   - Core field buffers: data, rhs, scratch (always), accum (classic RK4 only)
 *   - Per-block metadata arrays: origins, dx_per_block, on_boundary, levels,
 *     neighbor_table, refined_map
 *   - Block ID list (copied from caller)
 *
 * Coarse_data is NOT allocated here — it's allocated by load_meta once we
 * know how many blocks are refined (level > 0).
 *
 * rk_method: RK_CK45 → accum=NULL (3 memory blocks),
 *            RK_CLASSIC → accum allocated (4 memory blocks).
 */
meshblock_pack_t *meshblock_pack_create(int n_blocks, size_t npts,
                                         const int *block_ids, int level,
                                         rk_method_t rk_method, int n_fields)
{
    meshblock_pack_t *pack = calloc(1, sizeof(meshblock_pack_t));
    if (!pack) {
        fprintf(stderr, "meshblock_pack_create: calloc failed\n");
        exit(1);
    }

    pack->n_blocks = n_blocks;
    pack->n_evolve = n_blocks;  /* default: all blocks evolved (caller overrides for buffer) */
    pack->npts     = npts;
    pack->n_fields = n_fields;
    pack->level    = level;

    /* Core field buffers: n_fields * n_blocks * npts doubles each.
     * Page-aligned for zero-copy GPU mapping. */
    size_t total_doubles = (size_t)pack->n_fields * n_blocks * npts;
    size_t total_bytes   = total_doubles * sizeof(double);

    pack->data    = pack_alloc_block(total_bytes);
    pack->rhs     = pack_alloc_block(total_bytes);
    pack->scratch = pack_alloc_block(total_bytes);

    /* Classic RK4 needs a 4th buffer for the weighted accumulator.
     * CK45 only needs 3 buffers (dU stored in scratch). */
    if (rk_method == RK_CLASSIC)
        pack->accum = pack_alloc_block(total_bytes);
    else
        pack->accum = NULL;

    /* Copy block ID list */
    pack->block_ids = pack_alloc_int(n_blocks);
    memcpy(pack->block_ids, block_ids, n_blocks * sizeof(int));

    /* Per-block metadata arrays (GPU-mappable).
     * Contents filled by meshblock_pack_load_meta. */
    pack->origins       = pack_alloc_double(n_blocks * 3);
    pack->dx_per_block  = pack_alloc_double(n_blocks);
    pack->on_boundary   = pack_alloc_int(n_blocks * 6);
    pack->levels        = pack_alloc_int(n_blocks);
    pack->neighbor_table = pack_alloc_int(n_blocks * NUM_NEIGHBORS);
    pack->refined_map   = pack_alloc_int(n_blocks);

    pack->nblevel_table = pack_alloc_int(n_blocks * 27);

    /* Initialize refined_map to -1 (no coarse_buf by default) */
    for (int b = 0; b < n_blocks; b++)
        pack->refined_map[b] = -1;

    /* Coarse_buf data: not allocated yet — meshblock_pack_load_meta
     * counts refined blocks and allocates if needed. */
    pack->coarse_data = NULL;
    pack->n_refined   = 0;
    pack->coarse_npts = 0;
    pack->coarse_Ntotal = 0;
    pack->coarse_N    = 0;
    pack->coarse_neighbor_table = NULL;

    /* Grid dimensions: set by meshblock_pack_load_meta */
    pack->N      = 0;
    pack->ghost  = 0;
    pack->Ntotal = 0;

    /* Device memory handle: NULL until backend_map_pack allocates */
    pack->device_handle = NULL;

    return pack;
}

/* Forward declaration: free persistent GPU device memory */
extern void backend_free_pack_device(meshblock_pack_t *pack);

/*
 * Free all pack buffers and the pack struct itself.
 */
void meshblock_pack_free(meshblock_pack_t *pack)
{
    if (!pack) return;

    /* Free persistent GPU device memory before host buffers */
    if (pack->device_handle)
        backend_free_pack_device(pack);

    /* Core field buffers */
    free(pack->data);
    free(pack->rhs);
    free(pack->scratch);
    free(pack->accum);

    /* Block ID list */
    free(pack->block_ids);

    /* Per-block metadata */
    free(pack->origins);
    free(pack->dx_per_block);
    free(pack->on_boundary);
    free(pack->levels);
    free(pack->neighbor_table);
    free(pack->refined_map);

    /* Coarse_buf data */
    free(pack->coarse_data);
    free(pack->coarse_neighbor_table);

    /* Neighbor level table */
    free(pack->nblevel_table);

    free(pack);
}

/* ========================================================================
 * Data transfer: pack ↔ blocks
 * ======================================================================== */

/*
 * Copy field data from per-block grid_t arrays into pack's contiguous buffer.
 *
 * Layout: pack->data[f * n_blocks * npts + b * npts + idx]
 *         = blocks[block_ids[b]]->grid->fields[f][idx]
 *
 * Copies: fields→data, rhs→rhs, scratch→scratch, accum→accum (if non-NULL).
 */
void meshblock_pack_load(meshblock_pack_t *pack, block_t **blocks)
{
    for (int b = 0; b < pack->n_blocks; b++) {
        block_t *blk = blocks[pack->block_ids[b]];
        size_t npts = pack->npts;

        for (int f = 0; f < pack->n_fields; f++) {
            size_t dst_off = (size_t)f * pack->n_blocks * npts
                           + (size_t)b * npts;
            memcpy(pack->data    + dst_off, blk->grid->fields[f],
                   npts * sizeof(double));
            memcpy(pack->rhs     + dst_off, blk->grid->rhs[f],
                   npts * sizeof(double));
            memcpy(pack->scratch + dst_off, blk->grid->scratch[f],
                   npts * sizeof(double));
            /* Classic RK4: also copy accumulator */
            if (pack->accum && blk->grid->accum[f])
                memcpy(pack->accum + dst_off, blk->grid->accum[f],
                       npts * sizeof(double));
        }
    }
}

/*
 * Copy field data from pack's contiguous buffer back into per-block grid_t.
 * Inverse of meshblock_pack_load.
 */
void meshblock_pack_store(const meshblock_pack_t *pack, block_t **blocks)
{
    for (int b = 0; b < pack->n_blocks; b++) {
        block_t *blk = blocks[pack->block_ids[b]];
        size_t npts = pack->npts;

        for (int f = 0; f < pack->n_fields; f++) {
            size_t src_off = (size_t)f * pack->n_blocks * npts
                           + (size_t)b * npts;
            memcpy(blk->grid->fields[f],  pack->data    + src_off,
                   npts * sizeof(double));
            memcpy(blk->grid->rhs[f],     pack->rhs     + src_off,
                   npts * sizeof(double));
            memcpy(blk->grid->scratch[f], pack->scratch + src_off,
                   npts * sizeof(double));
            /* Classic RK4: also copy accumulator back */
            if (pack->accum && blk->grid->accum[f])
                memcpy(blk->grid->accum[f], pack->accum + src_off,
                       npts * sizeof(double));
        }
    }
}

/* ========================================================================
 * Metadata loading
 * ======================================================================== */

/*
 * Fill per-block metadata from block_t structs into flat GPU-mappable arrays.
 *
 * Extracts from each block:
 *   - origin[3] → origins[b*3 .. b*3+2]
 *   - grid->dx  → dx_per_block[b]
 *   - on_boundary[6] → on_boundary[b*6 .. b*6+5]
 *   - loc.level → levels[b]
 *
 * Sets grid dimensions (N, ghost, Ntotal) from the first block.
 * All blocks must have the same N_block (required by AMR design).
 *
 * Counts refined blocks (level > 0), builds refined_map, and allocates
 * coarse_data buffer if any blocks are refined.
 */
void meshblock_pack_load_meta(meshblock_pack_t *pack, block_t **blocks)
{
    /* Grid dimensions from first block — all blocks share same N_block */
    block_t *first = blocks[pack->block_ids[0]];
    pack->N      = first->grid->N;
    pack->ghost  = first->grid->ghost;
    pack->Ntotal = first->grid->Ntotal;

    /* Count refined blocks for coarse_data allocation */
    int n_refined = 0;

    for (int b = 0; b < pack->n_blocks; b++) {
        block_t *blk = blocks[pack->block_ids[b]];

        /* Origin coordinates (physical position of block's low corner) */
        pack->origins[b * 3 + 0] = blk->origin[0];
        pack->origins[b * 3 + 1] = blk->origin[1];
        pack->origins[b * 3 + 2] = blk->origin[2];

        /* Grid spacing for this block (varies by refinement level) */
        pack->dx_per_block[b] = blk->grid->dx;

        /* Domain boundary flags: 6 faces per block */
        for (int face = 0; face < 6; face++)
            pack->on_boundary[b * 6 + face] = blk->on_boundary[face];

        /* Refinement level */
        pack->levels[b] = blk->loc.level;

        /* Flatten nblevel[3][3][3] → nblevel_table[b*27 + (oz+1)*9+(oy+1)*3+(ox+1)] */
        for (int oz = -1; oz <= 1; oz++)
            for (int oy = -1; oy <= 1; oy++)
                for (int ox = -1; ox <= 1; ox++)
                    pack->nblevel_table[b * 27 + (oz+1)*9 + (oy+1)*3 + (ox+1)]
                        = blk->nblevel[oz+1][oy+1][ox+1];

        /* Track refined blocks and build pack→coarse_data index mapping.
         * refined_map[b] gives the coarse_data index for this block,
         * or -1 if the block is at level 0 (no coarse_buf). */
        if (blk->loc.level > 0 && blk->coarse_buf) {
            pack->refined_map[b] = n_refined;
            n_refined++;
        } else {
            pack->refined_map[b] = -1;
        }
    }

    pack->n_refined = n_refined;

    /* Allocate coarse_data if any blocks are refined.
     * Coarse_buf dimensions: N_c = N_block/2, same ghost width.
     * Not padded (see coarse_buf_alloc in block.c). */
    if (n_refined > 0) {
        /* Get coarse_buf dimensions from first refined block */
        for (int b = 0; b < pack->n_blocks; b++) {
            block_t *blk = blocks[pack->block_ids[b]];
            if (blk->coarse_buf) {
                pack->coarse_N      = blk->coarse_buf->N;
                pack->coarse_Ntotal = blk->coarse_buf->Ntotal;
                pack->coarse_npts   = blk->coarse_buf->npoints;
                break;
            }
        }

        /* Contiguous buffer for all refined blocks' coarse_bufs.
         * Layout: coarse_data[r * n_fields * coarse_npts + f * coarse_npts + idx]
         * where r = refined_map[b] is the coarse_data slot for block b. */
        size_t coarse_total = (size_t)n_refined * pack->n_fields * pack->coarse_npts;
        pack->coarse_data = pack_alloc_block(coarse_total * sizeof(double));

        /* Neighbor table for coarse_bufs (filled by build_neighbors) */
        pack->coarse_neighbor_table = pack_alloc_int(n_refined * NUM_NEIGHBORS);
    }
}

/* ========================================================================
 * Neighbor table construction
 * ======================================================================== */

/*
 * Build pack-local neighbor table from block_t neighbor_ids.
 *
 * For each block b in the pack and each of 26 neighbor directions:
 *   1. Look up the mesh-level neighbor_id from the block's neighbor_ids[]
 *   2. Search the reverse-lookup table (mesh_id → pack_index) to find
 *      the pack-local index of that neighbor
 *   3. Store in neighbor_table[b * 26 + n] (or -1 if not in pack)
 *
 * This allows GPU kernels to find neighbors by pack-local index without
 * needing the full mesh data structure on the device.
 *
 * Also builds coarse_neighbor_table for refined blocks:
 *   - Maps each refined block's 26 neighbor directions to coarse_data indices
 *   - Same-level sibling → sibling's coarse_data index
 *   - Other → -1 (handled by cross-level copy from main grid)
 */
void meshblock_pack_build_neighbors(meshblock_pack_t *pack, block_t **blocks)
{
    /* Build reverse-lookup: mesh block ID → pack index.
     * Find max block_id to size the lookup array. */
    int max_id = 0;
    for (int b = 0; b < pack->n_blocks; b++) {
        if (pack->block_ids[b] > max_id)
            max_id = pack->block_ids[b];
    }

    /* Allocate reverse-lookup, initialized to -1 (not in pack) */
    int *reverse = malloc((max_id + 1) * sizeof(int));
    if (!reverse) {
        fprintf(stderr, "meshblock_pack_build_neighbors: malloc failed\n");
        exit(1);
    }
    for (int i = 0; i <= max_id; i++)
        reverse[i] = -1;

    for (int b = 0; b < pack->n_blocks; b++)
        reverse[pack->block_ids[b]] = b;

    /* Build main neighbor table: pack-local indices for 26 directions.
     * neighbor_table[b * 26 + n] = pack index of neighbor, or -1. */
    for (int b = 0; b < pack->n_blocks; b++) {
        block_t *blk = blocks[pack->block_ids[b]];

        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int mesh_nbr_id = blk->neighbor_ids[n];

            if (mesh_nbr_id < 0 || mesh_nbr_id > max_id) {
                /* No neighbor (domain boundary) */
                pack->neighbor_table[b * NUM_NEIGHBORS + n] = -1;
            } else {
                /* Map mesh-level neighbor ID to pack-local index */
                pack->neighbor_table[b * NUM_NEIGHBORS + n] = reverse[mesh_nbr_id];
            }
        }
    }

    /* Build coarse_neighbor_table for refined blocks.
     * For each refined block's 26 directions, find same-level siblings
     * that are also refined (have coarse_data entries). */
    if (pack->n_refined > 0 && pack->coarse_neighbor_table) {
        for (int b = 0; b < pack->n_blocks; b++) {
            int r = pack->refined_map[b];
            if (r < 0) continue;  /* skip level-0 blocks */

            block_t *blk = blocks[pack->block_ids[b]];

            for (int n = 0; n < NUM_NEIGHBORS; n++) {
                int mesh_nbr_id = blk->neighbor_ids[n];
                int coarse_nbr = -1;

                if (mesh_nbr_id >= 0 && mesh_nbr_id <= max_id) {
                    int pack_nbr = reverse[mesh_nbr_id];
                    /* Same-level sibling with a coarse_buf → use its
                     * coarse_data slot for coarse_buf ghost exchange */
                    if (pack_nbr >= 0 && pack->refined_map[pack_nbr] >= 0)
                        coarse_nbr = pack->refined_map[pack_nbr];
                }

                pack->coarse_neighbor_table[r * NUM_NEIGHBORS + n] = coarse_nbr;
            }
        }
    }

    free(reverse);
}

/* ========================================================================
 * Coarse_buf data transfer
 * ======================================================================== */

/*
 * Copy coarse_buf data from individual blocks into pack->coarse_data.
 *
 * For each refined block (level > 0), copies all field data from the
 * block's coarse_buf->fields into the pack's contiguous coarse_data buffer.
 *
 * Layout: coarse_data[r * NF * coarse_npts + f * coarse_npts + idx]
 * where r = refined_map[b].
 */
void meshblock_pack_load_coarse(meshblock_pack_t *pack, block_t **blocks)
{
    if (!pack->coarse_data || pack->n_refined == 0) return;

    size_t cnpts = pack->coarse_npts;

    for (int b = 0; b < pack->n_blocks; b++) {
        int r = pack->refined_map[b];
        if (r < 0) continue;  /* skip level-0 blocks */

        block_t *blk = blocks[pack->block_ids[b]];
        if (!blk->coarse_buf) continue;

        for (int f = 0; f < pack->n_fields; f++) {
            size_t dst_off = (size_t)r * pack->n_fields * cnpts
                           + (size_t)f * cnpts;
            memcpy(pack->coarse_data + dst_off,
                   blk->coarse_buf->fields[f],
                   cnpts * sizeof(double));
        }
    }
}

/*
 * Copy pack->coarse_data back into individual blocks' coarse_bufs.
 * Inverse of meshblock_pack_load_coarse.
 */
void meshblock_pack_store_coarse(const meshblock_pack_t *pack, block_t **blocks)
{
    if (!pack->coarse_data || pack->n_refined == 0) return;

    size_t cnpts = pack->coarse_npts;

    for (int b = 0; b < pack->n_blocks; b++) {
        int r = pack->refined_map[b];
        if (r < 0) continue;  /* skip level-0 blocks */

        block_t *blk = blocks[pack->block_ids[b]];
        if (!blk->coarse_buf) continue;

        for (int f = 0; f < pack->n_fields; f++) {
            size_t src_off = (size_t)r * pack->n_fields * cnpts
                           + (size_t)f * cnpts;
            memcpy(blk->coarse_buf->fields[f],
                   pack->coarse_data + src_off,
                   cnpts * sizeof(double));
        }
    }
}

/* ========================================================================
 * Lightweight sync: data buffer only (persistent pack path)
 * ======================================================================== */

/*
 * Sync only pack->data back into block fields.
 * Skips rhs, scratch, accum — those are temporary per-step buffers.
 * Saves 75% of memcpy vs full meshblock_pack_store.
 */
void meshblock_pack_sync_to_blocks(const meshblock_pack_t *pack, block_t **blocks)
{
    for (int b = 0; b < pack->n_blocks; b++) {
        block_t *blk = blocks[pack->block_ids[b]];
        size_t npts = pack->npts;

        for (int f = 0; f < pack->n_fields; f++) {
            size_t src_off = (size_t)f * pack->n_blocks * npts
                           + (size_t)b * npts;
            memcpy(blk->grid->fields[f], pack->data + src_off,
                   npts * sizeof(double));
        }
    }
}

/*
 * Sync block field data into pack->data.
 * Skips rhs, scratch, accum — those get overwritten each step.
 * Saves 75% of memcpy vs full meshblock_pack_load.
 */
void meshblock_pack_sync_from_blocks(meshblock_pack_t *pack, block_t **blocks)
{
    for (int b = 0; b < pack->n_blocks; b++) {
        block_t *blk = blocks[pack->block_ids[b]];
        size_t npts = pack->npts;

        for (int f = 0; f < pack->n_fields; f++) {
            size_t dst_off = (size_t)f * pack->n_blocks * npts
                           + (size_t)b * npts;
            memcpy(pack->data + dst_off, blk->grid->fields[f],
                   npts * sizeof(double));
        }
    }
}
