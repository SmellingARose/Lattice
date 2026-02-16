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

typedef struct {
    double  *data;         /* fields: [n_fields * n_blocks * npts]         */
    double  *rhs;          /* RHS:    same layout                          */
    double  *scratch;      /* scratch/dU: same layout                      */

    int      n_blocks;     /* number of blocks in this pack                */
    size_t   npts;         /* grid points per block = Ntotal^3             */
    int      n_fields;     /* NUM_FIELDS (25)                              */

    int     *block_ids;    /* which block IDs are packed [n_blocks]         */
    int      level;        /* level of blocks in this pack (-1 = mixed)    */
} meshblock_pack_t;

/* Pack indexing macro:
 *   PACK_IDX(pack, f, b, idx) -> flat offset into pack->data/rhs/scratch
 * where f = field index, b = block index within pack, idx = point index */
#define PACK_IDX(pack, f, b, idx) \
    ((size_t)(f) * (pack)->n_blocks * (pack)->npts \
     + (size_t)(b) * (pack)->npts + (size_t)(idx))

/*
 * Allocate a MeshBlockPack for n_blocks blocks, each with npts grid points.
 * Allocates page-aligned contiguous buffers for data, rhs, scratch.
 * block_ids is caller-allocated and copied in.
 */
meshblock_pack_t *meshblock_pack_create(int n_blocks, size_t npts,
                                         const int *block_ids, int level);

/* Free pack and all its buffers */
void meshblock_pack_free(meshblock_pack_t *pack);

/*
 * Copy field data from individual blocks into the pack's contiguous buffer.
 * blocks: array of all blocks (indexed by block_ids[b]).
 * Copies fields -> data, rhs -> rhs, scratch -> scratch.
 */
void meshblock_pack_load(meshblock_pack_t *pack, block_t **blocks);

/*
 * Copy field data from pack back into individual blocks.
 * Inverse of meshblock_pack_load.
 */
void meshblock_pack_store(const meshblock_pack_t *pack, block_t **blocks);

#endif /* LATTICE_MESHBLOCK_PACK_H */
