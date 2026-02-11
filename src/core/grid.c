/*
 * Lattice — 3D Numerical Relativity
 * Grid allocation and deallocation.
 *
 * All arrays page-aligned (4096 bytes) for zero-copy GPU buffers.
 * Ntotal padded so interior N is a multiple of 16 for cache alignment.
 */

#include "grid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PAGE_ALIGN 4096

static double *alloc_field(size_t npoints)
{
    void *ptr = NULL;
    size_t bytes = npoints * sizeof(double);
    if (posix_memalign(&ptr, PAGE_ALIGN, bytes) != 0) {
        fprintf(stderr, "grid: posix_memalign failed for %zu bytes\n", bytes);
        exit(1);
    }
    memset(ptr, 0, bytes);
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

    for (int f = 0; f < NUM_FIELDS; f++) {
        g->fields[f]  = alloc_field(g->npoints);
        g->rhs[f]     = alloc_field(g->npoints);
        g->scratch[f] = alloc_field(g->npoints);
        g->accum[f]   = alloc_field(g->npoints);
    }

    return g;
}

void grid_free(grid_t *g)
{
    if (!g) return;
    for (int f = 0; f < NUM_FIELDS; f++) {
        free(g->fields[f]);
        free(g->rhs[f]);
        free(g->scratch[f]);
        free(g->accum[f]);
    }
    free(g);
}
