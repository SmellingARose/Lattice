/*
 * Lattice — 3D Numerical Relativity
 * RK4 time integrator (memory-efficient 5b variant).
 *
 * Uses 1 scratch + 1 accumulator instead of 4 k-arrays.
 * Saves ~60% scratch memory.
 */

#ifndef LATTICE_RK4_H
#define LATTICE_RK4_H

#include "../core/grid.h"
#include "../core/params.h"
#include "../backend/backend.h"

/* Function pointer types for RHS and boundary conditions */
typedef rhs_point_func_t rk4_rhs_func_t;
typedef void (*rk4_bc_func_t)(double **rhs, const double *const *src,
                               const grid_t *g);

/* Advance one full RK4 step.
 * After the step, enforces algebraic constraints:
 *   - det(gambar) = 1
 *   - tr(Abar) = 0 */
void rk4_step(grid_t *g, const sim_params_t *p,
              rk4_rhs_func_t rhs_func, rk4_bc_func_t bc_func,
              double dt);

#endif /* LATTICE_RK4_H */
