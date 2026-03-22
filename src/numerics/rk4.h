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

/* Function pointer type for RHS */
typedef rhs_point_func_t rk4_rhs_func_t;

/* Advance all blocks in a mesh by one full RK step (global dt).
 * Uses the packed batch kernel path: all leaf blocks are packed into
 * a contiguous meshblock_pack_t, kernels operate on the full pack,
 * ghost exchange uses CPU fallback (Commit 1) or device kernels (Commit 2).
 * Ghost exchange before each RHS, Sommerfeld on domain boundaries only.
 * Same integrator (classic or CK45) as single-grid rk4_step. */
struct mesh_s;  /* forward declaration to avoid circular include */
void rk4_step_mesh(struct mesh_s *m, const sim_params_t *p,
                   rk4_rhs_func_t rhs_func, double dt);

/*
 * Sync all GPU-resident level packs back to host blocks.
 * Called from main.c before CPU-only operations (checkpoint, output,
 * AH finder, regrid). No-op if data is already on host.
 * After this call, block->grid->fields contain current data.
 */
void gpu_sync_all_to_host(struct mesh_s *m);

#endif /* LATTICE_RK4_H */
