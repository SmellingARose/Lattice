/*
 * grid.c — Grid allocation and management
 *
 * All field arrays are page-aligned (4096 bytes) via posix_memalign
 * for zero-copy GPU buffer compatibility.
 */

#include "grid.h"

#include <stdlib.h>
#include <string.h>

#define PAGE_ALIGN 4096

static double *alloc_field(int n)
{
    double *ptr = NULL;
    if (posix_memalign((void **)&ptr, PAGE_ALIGN, (size_t)n * sizeof(double)) != 0) {
        return NULL;
    }
    memset(ptr, 0, (size_t)n * sizeof(double));
    return ptr;
}

int grid_alloc(grid_t *g)
{
    int n = grid_total_points(g);
    int nf = g->params.num_fields;

    for (int f = 0; f < nf; f++) {
        g->fields[f] = alloc_field(n);
        if (!g->fields[f]) return -1;

        g->rhs[f] = alloc_field(n);
        if (!g->rhs[f]) return -1;

        g->rk_scratch[f] = alloc_field(n);
        if (!g->rk_scratch[f]) return -1;

        for (int s = 0; s < 4; s++) {
            g->rk_k[s][f] = alloc_field(n);
            if (!g->rk_k[s][f]) return -1;
        }
    }

    /* Zero out unused field slots */
    for (int f = nf; f < NUM_TOTAL_FIELDS; f++) {
        g->fields[f] = NULL;
        g->rhs[f] = NULL;
        g->rk_scratch[f] = NULL;
        for (int s = 0; s < 4; s++) {
            g->rk_k[s][f] = NULL;
        }
    }

    g->time = 0.0;
    g->step = 0;
    return 0;
}

void grid_free(grid_t *g)
{
    for (int f = 0; f < NUM_TOTAL_FIELDS; f++) {
        free(g->fields[f]);
        g->fields[f] = NULL;

        free(g->rhs[f]);
        g->rhs[f] = NULL;

        free(g->rk_scratch[f]);
        g->rk_scratch[f] = NULL;

        for (int s = 0; s < 4; s++) {
            free(g->rk_k[s][f]);
            g->rk_k[s][f] = NULL;
        }
    }
}

void grid_zero_fields(grid_t *g)
{
    int n = grid_total_points(g);
    for (int f = 0; f < g->params.num_fields; f++) {
        memset(g->fields[f], 0, (size_t)n * sizeof(double));
    }
}

void grid_zero_rhs(grid_t *g)
{
    int n = grid_total_points(g);
    for (int f = 0; f < g->params.num_fields; f++) {
        memset(g->rhs[f], 0, (size_t)n * sizeof(double));
    }
}
