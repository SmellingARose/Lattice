/*
 * Lattice — 3D Numerical Relativity
 * Grid allocation and deallocation.
 *
 * All arrays page-aligned (4096 bytes) for zero-copy GPU buffers.
 * Each group (fields, rhs, scratch, accum) is allocated as one
 * contiguous block for efficient GPU mapping.
 * Ntotal padded so interior N is a multiple of 16 for cache alignment.
 */

#include "grid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PAGE_ALIGN 4096

static double *alloc_block(size_t total_bytes)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, PAGE_ALIGN, total_bytes) != 0) {
        fprintf(stderr, "grid: posix_memalign failed for %zu bytes\n", total_bytes);
        exit(1);
    }
    memset(ptr, 0, total_bytes);
    return (double *)ptr;
}

grid_t *grid_alloc(int N, double L)
{
    grid_t *g = calloc(1, sizeof(grid_t));
    if (!g) {
        fprintf(stderr, "grid: calloc failed\n");
        exit(1);
    }

    /* Pad N to next multiple of 16 */
    int N_padded = ((N + 15) / 16) * 16;
    g->N      = N_padded;
    g->ghost  = GHOST_WIDTH;
    g->Ntotal = N_padded + 2 * GHOST_WIDTH;
    g->L      = L;
    g->dx     = L / N_padded;
    g->npoints = (size_t)g->Ntotal * g->Ntotal * g->Ntotal;

    /* Allocate contiguous blocks: NUM_FIELDS * npoints each */
    size_t block_bytes = (size_t)NUM_FIELDS * g->npoints * sizeof(double);
    g->fields_block  = alloc_block(block_bytes);
    g->rhs_block     = alloc_block(block_bytes);
    g->scratch_block = alloc_block(block_bytes);
    g->accum_block   = alloc_block(block_bytes);

    /* Point per-field pointers into the contiguous blocks */
    for (int f = 0; f < NUM_FIELDS; f++) {
        g->fields[f]  = g->fields_block  + f * g->npoints;
        g->rhs[f]     = g->rhs_block     + f * g->npoints;
        g->scratch[f] = g->scratch_block + f * g->npoints;
        g->accum[f]   = g->accum_block   + f * g->npoints;
    }

    return g;
}

void grid_free(grid_t *g)
{
    if (!g) return;
    free(g->fields_block);
    free(g->rhs_block);
    free(g->scratch_block);
    free(g->accum_block);
    free(g);
}
