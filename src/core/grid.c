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

grid_t *grid_alloc_ex(int N, double L, rk_method_t method, int n_fields)
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
    g->n_fields = n_fields;

    /* Allocate contiguous blocks: n_fields * npoints each.
     * CK45 needs only 3 blocks (fields, rhs, scratch=dU); classic needs 4. */
    size_t block_bytes = (size_t)n_fields * g->npoints * sizeof(double);
    g->fields_block  = alloc_block(block_bytes);
    g->rhs_block     = alloc_block(block_bytes);
    g->scratch_block = alloc_block(block_bytes);

    if (method == RK_CLASSIC) {
        g->accum_block = alloc_block(block_bytes);
    } else {
        g->accum_block = NULL;
    }

    /* Point per-field pointers into the contiguous blocks.
     * Pointers for f >= n_fields are set to NULL. */
    for (int f = 0; f < n_fields; f++) {
        g->fields[f]  = g->fields_block  + f * g->npoints;
        g->rhs[f]     = g->rhs_block     + f * g->npoints;
        g->scratch[f] = g->scratch_block + f * g->npoints;
        g->accum[f]   = g->accum_block ? g->accum_block + f * g->npoints : NULL;
    }
    for (int f = n_fields; f < NUM_FIELDS; f++) {
        g->fields[f]  = NULL;
        g->rhs[f]     = NULL;
        g->scratch[f] = NULL;
        g->accum[f]   = NULL;
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
