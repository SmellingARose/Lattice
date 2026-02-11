/*
 * Lattice — 3D Numerical Relativity
 * Grid struct, allocation, and indexing.
 *
 * SoA layout: each field is a contiguous double* array.
 * x is the innermost (unit-stride) index.
 * Ghost zone width = 4 (4th-order stencils + 6th-order KO dissipation).
 */

#ifndef LATTICE_GRID_H
#define LATTICE_GRID_H

#include "fields.h"
#include <stddef.h>

#define GHOST_WIDTH 4

typedef struct {
    int    N;          /* interior grid points per side               */
    int    ghost;      /* ghost zone width = GHOST_WIDTH              */
    int    Ntotal;     /* N + 2*ghost, padded so N is multiple of 16  */
    double dx;         /* grid spacing                                */
    double L;          /* physical domain size                        */
    size_t npoints;    /* Ntotal^3, total points including ghosts     */

    double *fields[NUM_FIELDS];   /* evolved field arrays       */
    double *rhs[NUM_FIELDS];      /* RHS scratch (for RK stages)*/
    double *scratch[NUM_FIELDS];  /* scratch state (RK stages)  */
    double *accum[NUM_FIELDS];    /* RK4 accumulator            */
} grid_t;

/*
 * Flat 3D indexing: x innermost (unit stride), z outermost.
 * IDX(g, i, j, k) = k*Ntotal*Ntotal + j*Ntotal + i
 */
#define IDX(g, i, j, k) \
    ((k) * (g)->Ntotal * (g)->Ntotal + (j) * (g)->Ntotal + (i))

/* Strides for finite difference access by direction */
#define STRIDE_X          1
#define STRIDE_Y(g)       ((g)->Ntotal)
#define STRIDE_Z(g)       ((g)->Ntotal * (g)->Ntotal)

/* Physical coordinate of grid point i (cell-centered) */
#define COORD(g, i) (((i) - (g)->ghost + 0.5) * (g)->dx - (g)->L * 0.5)

/* Allocate and initialize grid (all arrays zeroed) */
grid_t *grid_alloc(int N, double L);

/* Free all grid arrays */
void grid_free(grid_t *g);

#endif /* LATTICE_GRID_H */
