/*
 * Lattice — 3D Numerical Relativity
 * Backend abstraction layer.
 *
 * Packed batch kernels (AMR production path):
 *     backend_*_packed() — one kernel per operation across ALL blocks.
 *     All data stays on device; ghost exchange is device-to-device.
 *     CPU backend: OpenMP parallel for mirrors of all packed functions.
 *     GPU backend: HIP kernels on AMD/NVIDIA GPUs (via backend_hip.cpp).
 *
 * CPU backend = OpenMP parallel loops calling the C RHS function.
 * GPU backend = HIP kernels (same C physics code with LATTICE_DEVICE annotations).
 *
 * Ref: AthenaK meshblock_pack pattern (Grete et al. 2024, arXiv:2409.16053)
 */

#ifndef LATTICE_BACKEND_H
#define LATTICE_BACKEND_H

#include "../core/device.h"
#include "../core/grid.h"
#include "../core/params.h"
#include "../amr/meshblock_pack.h"

EXTERN_C_BEGIN

/* Point-wise RHS function signature.
 * Called for each interior grid point (i,j,k).
 * Reads from src arrays, writes to rhs arrays. */
typedef void (*rhs_point_func_t)(double ** restrict rhs,
                                 const double *const * restrict src,
                                 const grid_t *g, const sim_params_t *p,
                                 int i, int j, int k);

/* Backend lifecycle (no-op for CPU) */
void backend_init(void);
void backend_cleanup(void);

/* ========================================================================
 * Packed batch kernel API (AMR production path)
 *
 * These functions operate on meshblock_pack_t — all leaf blocks packed
 * into contiguous buffers. One kernel launch covers all blocks.
 *
 * GPU backend: data mapped to device by backend_map_pack, all kernels
 * execute on device, backend_unmap_pack syncs back to host.
 *
 * CPU backend: map/unmap are no-ops, kernels use OpenMP parallel for.
 * ======================================================================== */

/*
 * Map pack data to GPU device memory. CPU backend: no-op.
 * Maps: data, rhs, scratch, accum (if non-NULL), all metadata arrays,
 * coarse_data (if non-NULL). Params mapped as read-only.
 *
 * Must be called before any packed kernel and after all pack loading
 * (load, load_meta, build_neighbors, load_coarse).
 */
void backend_map_pack(meshblock_pack_t *pack, const sim_params_t *p);

/*
 * Unmap pack data from GPU device memory. CPU backend: no-op.
 * Frees device memory WITHOUT copying data back to host.
 * Use when host data is not needed (future optimization).
 */
void backend_unmap_pack(meshblock_pack_t *pack);

/*
 * Unmap pack data from GPU device memory WITH sync. CPU backend: no-op.
 * Syncs data, rhs, scratch, accum, coarse_data back to host, then frees.
 * Must be called before meshblock_pack_store.
 */
void backend_unmap_pack_sync(meshblock_pack_t *pack);

/*
 * Zero a pack buffer. Used to initialize dU (scratch) for CK45
 * and accum for classic RK4.
 *
 * which: PACK_BUF_DATA (0), PACK_BUF_RHS (1),
 *        PACK_BUF_SCRATCH (2), PACK_BUF_ACCUM (3)
 */
void backend_zero_packed(meshblock_pack_t *pack, int which);

/*
 * Batched RHS evaluation for all interior cells of all blocks in one
 * kernel launch. Dispatches to ccz4_maxwell_rhs_point when p->em_enabled,
 * otherwise ccz4_rhs_point (both are LATTICE_DEVICE annotated).
 *
 * For each block b, constructs per-field pointer arrays into the pack
 * layout and a minimal grid_t (N, ghost, Ntotal, dx) for the RHS
 * function. Evaluates over [ghost, ghost+N)^3.
 *
 * GPU: collapse(4) over (block, k, j, i), ~5.3 KB stack per thread.
 * CPU: outer loop over blocks, OpenMP collapse(2) over (k, j) per block.
 */
void backend_compute_rhs_packed(meshblock_pack_t *pack, const sim_params_t *p);

/*
 * Batched Sommerfeld boundary conditions: apply outgoing-wave BCs to RHS
 * for all blocks in one kernel launch.
 *
 * For each block, iterates over ghost zone points, skips interior and
 * non-boundary ghosts (filled by ghost exchange). For boundary ghost
 * points, computes: rhs = -x^i/r * d_i(f) + (f_asymptotic - f) / r
 *
 * Uses pack metadata (origins, dx_per_block, on_boundary) for physical
 * coordinates and boundary detection. Calls asymptotic_value() and
 * boundary_d1() from sommerfeld.h (declared with omp declare target).
 */
void backend_sommerfeld_packed(meshblock_pack_t *pack, const sim_params_t *p);

/*
 * Fused CK45 update: dU = A*dU + dt*F;  U += B*dU
 * Operates on the entire flat buffer (all fields, all blocks, all points).
 *
 * This is the inner loop of the Carpenter-Kennedy 2N low-storage scheme.
 * A_s, B_s are the CK45 coefficients for stage s.
 *
 * Ref: Carpenter & Kennedy, NASA TM-109112 (1994), Solution 3.
 */
void backend_update_ck45_packed(meshblock_pack_t *pack,
                                 double A_s, double B_s, double dt);

/* ---- Classic RK4 packed operations ----
 * Each operates on the entire flat buffer (NF * n_blocks * npts). */

/*
 * Copy between pack buffers: dst_arr[i] = src_arr[i].
 * dst/src select which buffer: PACK_BUF_DATA (0), PACK_BUF_RHS (1),
 *   PACK_BUF_SCRATCH (2), PACK_BUF_ACCUM (3).
 */
void backend_copy_packed(meshblock_pack_t *pack, int dst, int src);

/*
 * Accumulate weighted RHS: accum[i] += weight * dt * rhs[i].
 * Used by classic RK4 to build the weighted sum of k-values.
 */
void backend_accum_add_packed(meshblock_pack_t *pack, double weight, double dt);

/*
 * Linear combination: data[i] = scratch[i] + alpha * dt * rhs[i].
 * Used by classic RK4 to compute intermediate states from backup U^0.
 */
void backend_axpy_packed(meshblock_pack_t *pack, double alpha, double dt);

/*
 * Apply accumulator: data[i] += accum[i].
 * Final step of classic RK4: U = U^0 + accum.
 */
void backend_apply_accum_packed(meshblock_pack_t *pack);

/*
 * Fused RK4 stage update (stages 1-3):
 *   accum[i] += weight * dt * rhs[i]
 *   data[i]   = scratch[i] + alpha * dt * rhs[i]
 *
 * Replaces separate backend_accum_add_packed + backend_axpy_packed.
 * Saves 1 kernel launch + 1 rhs buffer read per stage.
 */
void backend_rk4_stage_packed(meshblock_pack_t *pack,
                               double weight, double alpha, double dt);

/*
 * Fused RK4 final update (stage 4):
 *   data[i] = scratch[i] + accum[i] + weight * dt * rhs[i]
 *
 * Replaces separate backend_accum_add_packed + backend_copy_packed +
 * backend_apply_accum_packed. Saves 2 kernel launches.
 */
void backend_rk4_final_packed(meshblock_pack_t *pack, double weight, double dt);

/*
 * Ghost exchange on pack buffers.
 * All 5 phases of multilevel ghost exchange on pack->data/coarse_data.
 *
 * CPU: OpenMP parallel for on host memory.
 * GPU: 7 device-side kernel launches, zero PCIe DMA.
 *
 * Ref: ghost_exchange_multilevel() in ghost_exchange.c (per-block version)
 */
void backend_ghost_exchange_packed(meshblock_pack_t *pack);

/*
 * Enforce algebraic constraints on packed data: det(h)=1, tr(A)=0,
 * chi > 0, lapse > 0. Operates on ALL points (interior + ghost).
 *
 * CPU: OpenMP parallel for over all blocks and points.
 * GPU: single kernel launch, collapse(4) over (block, k, j, i).
 *
 * Called after the final RK4 field update, before ghost exchange.
 * Ref: enforce_algebraic() in rk4.c (per-grid version)
 */
void backend_enforce_algebraic_packed(meshblock_pack_t *pack);

/* ========================================================================
 * Runtime backend detection
 * ======================================================================== */

/* Returns 1 if the GPU backend is active, 0 for CPU.
 * Used to select GPU solver path at runtime. */
int backend_is_gpu(void);

/* ========================================================================
 * Multigrid solver packed kernel API
 *
 * GPU-accelerated FAS multigrid constraint solver. Separate device state
 * from evolution kernels (d_solver[] vs d_ptrs). Operates on meshblock
 * packs with solver field layout (10 fields: 4 solution + 6 background).
 *
 * CPU backend: OpenMP loops calling shared point functions from
 *              mg_smooth_point.h.
 * GPU backend: HIP kernels — one per operation, 8 launches per GS sweep.
 *
 * Ref: arXiv:2510.11152 (GPU FAS multigrid, 8-color MCGS)
 * ======================================================================== */

#define MAX_SOLVER_SLOTS 8  /* one per AMR level */

/* Solver field slot indices: see MGP_* in mg_smooth_point.h */

/*
 * Map/unmap solver pack to/from GPU device memory.
 * Uses separate d_solver[] slots from evolution d_ptrs.
 * CPU backend: all no-ops.
 */
void backend_map_solver_pack(meshblock_pack_t *pack, int slot);
void backend_unmap_solver_pack_sync(meshblock_pack_t *pack, int slot);

/*
 * Sync solver pack data buffer between host and device.
 * Used for level-0 transfer to/from uniform MG solver on CPU.
 */
void backend_sync_solver_data_to_host(meshblock_pack_t *pack, int slot);
void backend_sync_solver_data_to_device(meshblock_pack_t *pack, int slot);

/*
 * 8-color Newton-Gauss-Seidel smoother (one color per launch).
 * All points of the given color are independent → GPU parallel.
 *
 * four_field: 0 = 1-field (psi only), 1 = 4-field (psi + V^i)
 */
void backend_mg_smooth_packed(meshblock_pack_t *pack, int slot, int color,
                               int four_field);

/*
 * Same-level ghost exchange for solver fields.
 * Copies ghost zones between same-level neighbors in the pack.
 * n_sol: number of solution fields to exchange (1 or 4).
 */
void backend_mg_ghost_same_level_packed(meshblock_pack_t *pack, int slot, int n_sol);

/*
 * Apply zero-Dirichlet BCs on domain-boundary ghost zones.
 * Zeros solution field ghost zones on boundary faces.
 */
void backend_mg_bc_packed(meshblock_pack_t *pack, int slot, int four_field);

/*
 * Evaluate nonlinear operator L(u) on all interior points.
 * Writes result to accum buffer (slots 0..3).
 */
void backend_mg_operator_packed(meshblock_pack_t *pack, int slot,
                                 int four_field);

/*
 * Compute residual: accum[i] = rhs[i] - accum[i] on interior points.
 * Must be called after backend_mg_operator_packed.
 */
void backend_mg_residual_packed(meshblock_pack_t *pack, int slot,
                                 int four_field);

/*
 * Save solution: scratch[s] = data[s] for solution fields.
 * Used for FAS correction computation.
 */
void backend_mg_save_packed(meshblock_pack_t *pack, int slot, int four_field);

/*
 * Tau correction: rhs[s] += accum[s] on interior points.
 * Adds L(restricted_solution) to restricted residual.
 */
void backend_mg_tau_packed(meshblock_pack_t *pack, int slot, int four_field);

/*
 * Zero solution fields in data buffer.
 */
void backend_mg_zero_solution_packed(meshblock_pack_t *pack, int slot,
                                      int four_field);

/*
 * Zero RHS fields in rhs buffer.
 */
void backend_mg_zero_rhs_packed(meshblock_pack_t *pack, int slot, int four_field);

/*
 * Cross-slot restriction: fine → coarse.
 * Restricts solution (data) and residual (accum) from fine_slot to
 * coarse_slot using 8-child volume average (0.125 weight).
 *
 * child_map: [n_parents * 8] — for parent p, child_map[p*8+octant] is
 *            the fine-slot block index of that child (-1 if absent).
 * parent_ids: [n_parents] — coarse-slot block index of each parent.
 */
void backend_mg_restrict_packed(meshblock_pack_t *fine_pack, int fine_slot,
                                 meshblock_pack_t *coarse_pack, int coarse_slot,
                                 int four_field,
                                 const int *child_map, const int *parent_ids,
                                 int n_parents);

/*
 * Cross-slot prolongation: coarse correction → fine (add).
 * Computes correction = coarse_data - coarse_scratch, trilinear interp,
 * adds to fine_data.
 */
void backend_mg_prolong_add_packed(meshblock_pack_t *coarse_pack, int coarse_slot,
                                    meshblock_pack_t *fine_pack, int fine_slot,
                                    int four_field,
                                    const int *child_map, const int *parent_ids,
                                    int n_parents);

/*
 * FMG prolongation: coarse solution → fine (overwrite).
 * Trilinear interpolation from coarse parent to fine child.
 */
void backend_mg_prolong_fmg_packed(meshblock_pack_t *coarse_pack, int coarse_slot,
                                    meshblock_pack_t *fine_pack, int fine_slot,
                                    int four_field,
                                    const int *child_map, const int *parent_ids,
                                    int n_parents);

/*
 * L2 norm of residual on interior points (reduction).
 * Returns sqrt(sum((rhs - L(u))^2) / count).
 * Requires operator already evaluated in accum.
 */
double backend_mg_l2_norm_packed(meshblock_pack_t *pack, int slot,
                                  int four_field);

EXTERN_C_END

#endif /* LATTICE_BACKEND_H */
