/*
 * Lattice — 3D Numerical Relativity
 * AMR block allocation and deallocation.
 *
 * Ref: Athena++ src/mesh/meshblock.cpp (MeshBlock constructor)
 * Ref: AthenaK  src/mesh/meshblock.cpp
 */

#include "block.h"
#include "../core/fields.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PAGE_ALIGN 4096

/*
 * Allocate a fields-only grid with exact N (no padding to multiple of 16).
 * Used for coarse_buf which may have N < 16 (e.g. N=8 for N_block=16).
 * Only allocates fields_block; rhs/scratch/accum are NULL.
 *
 * Unlike grid_alloc(), this does not pad N and does not allocate
 * RK scratch arrays since coarse_buf is never time-evolved.
 */
static grid_t *coarse_buf_alloc(int N, double dx)
{
    grid_t *g = calloc(1, sizeof(grid_t));
    if (!g) {
        fprintf(stderr, "coarse_buf_alloc: calloc failed\n");
        exit(1);
    }

    g->N      = N;
    g->ghost  = GHOST_WIDTH;
    g->Ntotal = N + 2 * GHOST_WIDTH;
    g->L      = N * dx;
    g->dx     = dx;
    g->npoints = (size_t)g->Ntotal * g->Ntotal * g->Ntotal;

    /* Allocate only fields block (page-aligned for potential GPU use) */
    size_t block_bytes = (size_t)NUM_FIELDS * g->npoints * sizeof(double);
    void *ptr = NULL;
    if (posix_memalign(&ptr, PAGE_ALIGN, block_bytes) != 0) {
        fprintf(stderr, "coarse_buf_alloc: posix_memalign failed\n");
        exit(1);
    }
    memset(ptr, 0, block_bytes);
    g->fields_block = (double *)ptr;

    for (int f = 0; f < NUM_FIELDS; f++)
        g->fields[f] = g->fields_block + f * g->npoints;

    /* No RK arrays needed for coarse_buf */
    g->rhs_block = NULL;
    g->scratch_block = NULL;
    g->accum_block = NULL;
    for (int f = 0; f < NUM_FIELDS; f++) {
        g->rhs[f] = NULL;
        g->scratch[f] = NULL;
        g->accum[f] = NULL;
    }

    return g;
}

/*
 * 26-neighbor offset table: (ox1, ox2, ox3) for each neighbor.
 * Order: 6 faces, 12 edges, 8 corners.
 * Ref: Athena++ bvals_base.cpp SearchAndSetNeighbors enumeration.
 */
const int nbr_offset[NUM_NEIGHBORS][3] = {
    /* 6 faces */
    {-1,  0,  0}, { 1,  0,  0},   /* x-minus, x-plus */
    { 0, -1,  0}, { 0,  1,  0},   /* y-minus, y-plus */
    { 0,  0, -1}, { 0,  0,  1},   /* z-minus, z-plus */
    /* 12 edges */
    {-1, -1,  0}, { 1, -1,  0}, {-1,  1,  0}, { 1,  1,  0},  /* xy edges */
    {-1,  0, -1}, { 1,  0, -1}, {-1,  0,  1}, { 1,  0,  1},  /* xz edges */
    { 0, -1, -1}, { 0,  1, -1}, { 0, -1,  1}, { 0,  1,  1},  /* yz edges */
    /* 8 corners */
    {-1, -1, -1}, { 1, -1, -1}, {-1,  1, -1}, { 1,  1, -1},
    {-1, -1,  1}, { 1, -1,  1}, {-1,  1,  1}, { 1,  1,  1}
};

block_t *block_alloc(int id, int level, int N_block, double dx,
                     const double origin[3], rk_method_t method)
{
    block_t *b = calloc(1, sizeof(block_t));
    if (!b) {
        fprintf(stderr, "block_alloc: calloc failed\n");
        exit(1);
    }

    /* Allocate grid: L_block = N_block * dx gives correct dx after padding.
     * N_block should be a multiple of 16 so grid_alloc doesn't change it. */
    double L_block = N_block * dx;
    b->grid = grid_alloc(N_block, L_block, method);

    /* Verify grid_alloc didn't pad N (it shouldn't if N_block is multiple of 16) */
    if (b->grid->N != N_block) {
        fprintf(stderr, "block_alloc: grid_alloc padded N=%d to %d. "
                "Use N_block as a multiple of 16.\n", N_block, b->grid->N);
        exit(1);
    }

    /* Allocate coarse buffer at half resolution for level > 0 blocks.
     * N_c = N_block / 2, dx_c = 2 * dx (same resolution as coarse-level).
     * Uses coarse_buf_alloc (exact N, no padding, fields-only).
     * Ref: AthenaK coarse-buffer architecture (block-local AMR ghost fill) */
    if (level > 0 && N_block >= 2) {
        int N_c = N_block / 2;
        double dx_c = 2.0 * dx;
        b->coarse_buf = coarse_buf_alloc(N_c, dx_c);
    } else {
        b->coarse_buf = NULL;
    }

    b->id = id;
    b->loc.lx1   = 0;  /* caller sets via mesh_create */
    b->loc.lx2   = 0;
    b->loc.lx3   = 0;
    b->loc.level  = level;
    b->origin[0] = origin[0];
    b->origin[1] = origin[1];
    b->origin[2] = origin[2];

    /* Tree links: root block, no children (leaf) */
    b->parent_id = -1;
    for (int c = 0; c < 8; c++)
        b->child_ids[c] = -1;
    b->is_leaf = 1;

    /* Neighbors: default to no neighbor (physical boundary).
     * nblevel[3][3][3]: -1 = no neighbor, except self = own level.
     * Ref: Athena++ bvals_base.cpp initialization */
    for (int n = 0; n < NUM_NEIGHBORS; n++)
        b->neighbor_ids[n] = -1;

    memset(b->nblevel, -1, sizeof(b->nblevel));
    b->nblevel[1][1][1] = level;  /* self */

    memset(b->on_boundary, 0, sizeof(b->on_boundary));

    return b;
}

void block_free(block_t *b)
{
    if (!b) return;
    grid_free(b->coarse_buf);
    grid_free(b->grid);
    free(b);
}
