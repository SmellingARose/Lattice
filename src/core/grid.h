/*
 * grid.h — Grid structure and access functions
 *
 * Memory layout: struct-of-arrays (SoA). Each field is a contiguous double*
 * array. x is the innermost (unit-stride) index. Loops: z-outer, y-mid, x-inner.
 *
 * Allocations are page-aligned (4096 bytes) for zero-copy GPU buffers.
 */

#ifndef LATTICE_GRID_H
#define LATTICE_GRID_H

#include "fields.h"
#include "params.h"

typedef struct {
    /* Field data — one array per field */
    double *fields[NUM_TOTAL_FIELDS];

    /* RK4 intermediate stages: k[stage][field] */
    double *rk_k[4][NUM_TOTAL_FIELDS];

    /* RK4 scratch space for intermediate state */
    double *rk_scratch[NUM_TOTAL_FIELDS];

    /* RHS storage */
    double *rhs[NUM_TOTAL_FIELDS];

    /* Simulation parameters */
    sim_params_t params;

    /* Current simulation time and step */
    double time;
    int step;
} grid_t;

/*
 * Allocate all field arrays. Returns 0 on success, -1 on failure.
 */
int grid_alloc(grid_t *g);

/*
 * Free all field arrays.
 */
void grid_free(grid_t *g);

/*
 * Zero all field arrays.
 */
void grid_zero_fields(grid_t *g);

/*
 * Zero all RHS arrays.
 */
void grid_zero_rhs(grid_t *g);

/*
 * Total number of grid points (padded).
 */
static inline int grid_total_points(const grid_t *g)
{
    return g->params.nx_pad * g->params.ny_pad * g->params.nz_pad;
}

/*
 * Strides for each dimension.
 * x is unit stride, y stride = nx_pad, z stride = nx_pad * ny_pad.
 */
static inline int grid_stride_x(const grid_t *g)
{
    (void)g;
    return 1;
}

static inline int grid_stride_y(const grid_t *g)
{
    return g->params.nx_pad;
}

static inline int grid_stride_z(const grid_t *g)
{
    return g->params.nx_pad * g->params.ny_pad;
}

/*
 * Linear index from (i, j, k) grid coordinates.
 */
static inline int grid_idx(const grid_t *g, int i, int j, int k)
{
    return i + j * g->params.nx_pad + k * g->params.nx_pad * g->params.ny_pad;
}

/*
 * Physical coordinates at grid point (i, j, k).
 * Origin at center of domain.
 */
static inline double grid_x(const grid_t *g, int i)
{
    return (i - g->params.ghost_width) * g->params.dx - g->params.lx * 0.5;
}

static inline double grid_y(const grid_t *g, int j)
{
    return (j - g->params.ghost_width) * g->params.dy - g->params.ly * 0.5;
}

static inline double grid_z(const grid_t *g, int k)
{
    return (k - g->params.ghost_width) * g->params.dz - g->params.lz * 0.5;
}

/*
 * Interior loop bounds (excludes ghost zones).
 * Usage:
 *   GRID_LOOP_INTERIOR(g, i, j, k) {
 *       int idx = grid_idx(g, i, j, k);
 *       ...
 *   }
 *
 * Loop order: z-outer, y-mid, x-inner (unit stride on x).
 */
#define GRID_LOOP_INTERIOR(g, i, j, k)                                        \
    for (int (k) = (g)->params.ghost_width;                                   \
         (k) < (g)->params.nz - (g)->params.ghost_width; ++(k))              \
        for (int (j) = (g)->params.ghost_width;                               \
             (j) < (g)->params.ny - (g)->params.ghost_width; ++(j))           \
            for (int (i) = (g)->params.ghost_width;                           \
                 (i) < (g)->params.nx - (g)->params.ghost_width; ++(i))

/*
 * Full loop including ghost zones.
 */
#define GRID_LOOP_ALL(g, i, j, k)                                             \
    for (int (k) = 0; (k) < (g)->params.nz; ++(k))                           \
        for (int (j) = 0; (j) < (g)->params.ny; ++(j))                       \
            for (int (i) = 0; (i) < (g)->params.nx; ++(i))

#endif /* LATTICE_GRID_H */
