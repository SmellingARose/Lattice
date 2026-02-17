/*
 * Lattice — 3D Numerical Relativity
 * Time integration: Classic RK4 and CK45 (Carpenter-Kennedy 2N low-storage).
 *
 * Classic RK4: 4 stages, 4 memory blocks (fields, rhs, scratch, accum).
 * CK45:        5 stages, 3 memory blocks (fields=U, rhs=F, scratch=dU).
 *              25% more compute per step, 25% less memory.
 *
 * Selected at runtime via p->rk_method (RK_CLASSIC or RK_CK45).
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

/* Advance one full RK step (classic or CK45 per p->rk_method).
 * After the step, enforces algebraic constraints:
 *   - det(gambar) = 1
 *   - tr(Abar) = 0 */
void rk4_step(grid_t *g, const sim_params_t *p,
              rk4_rhs_func_t rhs_func, rk4_bc_func_t bc_func,
              double dt);

/* Advance all blocks in a mesh by one full RK step (global dt).
 * Uses the packed batch kernel path: all leaf blocks are packed into
 * a contiguous meshblock_pack_t, kernels operate on the full pack,
 * ghost exchange uses CPU fallback (Commit 1) or device kernels (Commit 2).
 * Ghost exchange before each RHS, Sommerfeld on domain boundaries only.
 * Same integrator (classic or CK45) as rk4_step. */
struct mesh_s;  /* forward declaration to avoid circular include */
void rk4_step_mesh(struct mesh_s *m, const sim_params_t *p,
                   rk4_rhs_func_t rhs_func, double dt);

/* Per-block fallback for debug/comparison.
 * Same algorithm as rk4_step_mesh but launches one kernel per block.
 * Kept for validation: packed should produce identical results. */
void rk4_step_mesh_perblock(struct mesh_s *m, const sim_params_t *p,
                             rk4_rhs_func_t rhs_func, double dt);

#endif /* LATTICE_RK4_H */
