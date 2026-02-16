/*
 * Lattice — 3D Numerical Relativity
 * AMR block allocation and deallocation.
 *
 * Ref: Athena++ src/mesh/meshblock.cpp (MeshBlock constructor)
 * Ref: AthenaK  src/mesh/meshblock.cpp
 */

#include "block.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    grid_free(b->grid);
    free(b);
}
