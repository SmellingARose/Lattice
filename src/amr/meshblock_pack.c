/*
 * Lattice — 3D Numerical Relativity
 * MeshBlockPack: contiguous buffer allocation and data transfer.
 *
 * Page-aligned allocations for zero-copy GPU mapping,
 * matching the pattern in src/core/grid.c.
 *
 * Ref: AthenaK src/mesh/meshblock_pack.hpp (pack layout)
 * Ref: AthenaK src/mesh/meshblock.cpp (pack initialization)
 */

#include "meshblock_pack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PAGE_ALIGN 4096

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

meshblock_pack_t *meshblock_pack_create(int n_blocks, size_t npts,
                                         const int *block_ids, int level)
{
    meshblock_pack_t *pack = calloc(1, sizeof(meshblock_pack_t));
    if (!pack) {
        fprintf(stderr, "meshblock_pack_create: calloc failed\n");
        exit(1);
    }

    pack->n_blocks = n_blocks;
    pack->npts     = npts;
    pack->n_fields = NUM_FIELDS;
    pack->level    = level;

    /* Contiguous buffers: n_fields * n_blocks * npts doubles each */
    size_t total_doubles = (size_t)NUM_FIELDS * n_blocks * npts;
    size_t total_bytes   = total_doubles * sizeof(double);

    pack->data    = pack_alloc_block(total_bytes);
    pack->rhs     = pack_alloc_block(total_bytes);
    pack->scratch = pack_alloc_block(total_bytes);

    /* Copy block ID list */
    pack->block_ids = calloc(n_blocks, sizeof(int));
    if (!pack->block_ids) {
        fprintf(stderr, "meshblock_pack_create: calloc block_ids failed\n");
        exit(1);
    }
    memcpy(pack->block_ids, block_ids, n_blocks * sizeof(int));

    return pack;
}

void meshblock_pack_free(meshblock_pack_t *pack)
{
    if (!pack) return;
    free(pack->data);
    free(pack->rhs);
    free(pack->scratch);
    free(pack->block_ids);
    free(pack);
}

/*
 * Copy from per-block grid_t arrays into pack's contiguous buffer.
 *
 * Layout: pack->data[f * n_blocks * npts + b * npts + idx]
 *         = blocks[block_ids[b]]->grid->fields[f][idx]
 */
void meshblock_pack_load(meshblock_pack_t *pack, block_t **blocks)
{
    for (int b = 0; b < pack->n_blocks; b++) {
        block_t *blk = blocks[pack->block_ids[b]];
        size_t npts = pack->npts;

        for (int f = 0; f < NUM_FIELDS; f++) {
            size_t dst_off = (size_t)f * pack->n_blocks * npts
                           + (size_t)b * npts;
            memcpy(pack->data    + dst_off, blk->grid->fields[f],
                   npts * sizeof(double));
            memcpy(pack->rhs     + dst_off, blk->grid->rhs[f],
                   npts * sizeof(double));
            memcpy(pack->scratch + dst_off, blk->grid->scratch[f],
                   npts * sizeof(double));
        }
    }
}

/*
 * Copy from pack's contiguous buffer back into per-block grid_t arrays.
 * Inverse of meshblock_pack_load.
 */
void meshblock_pack_store(const meshblock_pack_t *pack, block_t **blocks)
{
    for (int b = 0; b < pack->n_blocks; b++) {
        block_t *blk = blocks[pack->block_ids[b]];
        size_t npts = pack->npts;

        for (int f = 0; f < NUM_FIELDS; f++) {
            size_t src_off = (size_t)f * pack->n_blocks * npts
                           + (size_t)b * npts;
            memcpy(blk->grid->fields[f],  pack->data    + src_off,
                   npts * sizeof(double));
            memcpy(blk->grid->rhs[f],     pack->rhs     + src_off,
                   npts * sizeof(double));
            memcpy(blk->grid->scratch[f], pack->scratch + src_off,
                   npts * sizeof(double));
        }
    }
}
