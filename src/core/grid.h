/*
 * Lattice — 3D Numerical Relativity
 * Grid struct, allocation, and indexing.
 *
 * SoA layout: each field is a contiguous double* array.
 * x is the innermost (unit-stride) index.
 * Ghost zone width = 4 (6th-order FD needs 3, 6th-order KO needs 3, advection needs 4).
 */

#ifndef LATTICE_GRID_H
#define LATTICE_GRID_H

#include "device.h"
#include "fields.h"
#include "params.h"
#include <stddef.h>

EXTERN_C_BEGIN

#define GHOST_WIDTH 4

typedef struct {
    int    N;          /* interior grid points per side               */
    int    ghost;      /* ghost zone width = GHOST_WIDTH              */
    int    Ntotal;     /* N + 2*ghost, padded so N is multiple of 16  */
    int    level;      /* AMR level (0 = base, higher = finer)        */
    double dx;         /* grid spacing                                */
    double inv_dx;     /* 1.0 / dx (precomputed for FD stencils)      */
    double L;          /* physical domain size                        */
    size_t npoints;    /* Ntotal^3, total points including ghosts     */
    int    n_fields;   /* active fields (<= NUM_FIELDS)               */

    double *fields[NUM_FIELDS];   /* evolved field arrays       */
    double *rhs[NUM_FIELDS];      /* RHS scratch (for RK stages)*/
    double *scratch[NUM_FIELDS];  /* scratch state / dU for CK45  */
    double *accum[NUM_FIELDS];    /* RK4 accumulator (NULL if CK45) */

    /* Contiguous backing blocks (fields[f] points into these) */
    double *fields_block;
    double *rhs_block;
    double *scratch_block;
    double *accum_block;
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

/* Allocate grid with explicit field count.
 * n_fields <= NUM_FIELDS. Pointers for f >= n_fields set to NULL.
 * CK45 skips accum_block allocation (3 blocks instead of 4). */
grid_t *grid_alloc_ex(int N, double L, rk_method_t method, int n_fields);

/* Allocate grid with all NUM_FIELDS fields (backward compatible). */
static inline grid_t *grid_alloc(int N, double L, rk_method_t method)
{
    return grid_alloc_ex(N, L, method, NUM_FIELDS);
}

/* Free all grid arrays */
void grid_free(grid_t *g);

EXTERN_C_END

#endif /* LATTICE_GRID_H */
