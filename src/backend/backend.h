/*
 * Lattice — 3D Numerical Relativity
 * Backend abstraction layer.
 *
 * CPU backend = OpenMP triple loop calling the C RHS function.
 * GPU backends (Metal/CUDA/HIP) are stubs for now.
 */

#ifndef LATTICE_BACKEND_H
#define LATTICE_BACKEND_H

#include "../core/grid.h"
#include "../core/params.h"

/* Point-wise RHS function signature.
 * Called for each interior grid point (i,j,k).
 * Reads from src arrays, writes to rhs arrays. */
typedef void (*rhs_point_func_t)(double **rhs, const double *const *src,
                                 const grid_t *g, const sim_params_t *p,
                                 int i, int j, int k);

/* Compute RHS over the entire interior grid.
 * Dispatches to the selected backend (OpenMP, Metal, CUDA, HIP). */
void backend_compute_rhs(double **rhs, const double *const *src,
                         const grid_t *g, const sim_params_t *p,
                         rhs_point_func_t func);

/* Backend lifecycle (no-op for CPU) */
void backend_init(void);
void backend_cleanup(void);

#endif /* LATTICE_BACKEND_H */
