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
 * Used to select GPU-resident evolution (Berger-Oliger subcycling). */
int backend_is_gpu(void);

/* ========================================================================
 * Persistent per-pack device memory API
 *
 * Eliminates hipMalloc/hipFree per sub-step. Device memory lives with
 * the pack (via pack->device_handle). First backend_map_pack allocates;
 * subsequent calls only sync data. backend_free_pack_device releases
 * device memory (called from meshblock_pack_free or explicitly).
 *
 * backend_activate_pack sets d_ptrs from a pack's device handle without
 * any memcpy — used by GPU-resident subcycling where data is already
 * on device.
 *
 * backend_save_old_packed does a device→device copy of the data buffer
 * to a pre-step snapshot (fields_old) for temporal interpolation.
 *
 * backend_cross_level_ghost_fill_packed performs the full 5-phase
 * cross-level ghost exchange entirely on device, reading the coarser
 * pack's data (with temporal interpolation) to fill the finer pack's
 * coarse buffer.
 * ======================================================================== */

/*
 * Free persistent device memory associated with a pack.
 * Called from meshblock_pack_free or explicitly on regrid.
 * CPU backend: no-op.
 */
void backend_free_pack_device(meshblock_pack_t *pack);

/*
 * Activate a pack's persistent device state as the current d_ptrs
 * without any host↔device memcpy. Used by GPU-resident subcycling
 * to switch between level packs that are already on device.
 * CPU backend: no-op.
 */
void backend_activate_pack(meshblock_pack_t *pack);

/*
 * Save pre-step field data on device: fields_old = data (D2D copy).
 * Used for temporal interpolation by finer levels during subcycling.
 * CPU backend: no-op (CPU path uses block-level fields_old).
 */
void backend_save_old_packed(meshblock_pack_t *pack);

/*
 * Cross-level ghost fill entirely on device. Fills fine_pack's
 * coarse buffer from coarse_pack's data with temporal interpolation.
 * Executes the full 5-phase multilevel ghost exchange:
 *   1. Restrict fine interior → fine's coarse_buf
 *   2. Same-level coarse_buf exchange
 *   3. Cross-level fill (temporal interp from coarse pack)
 *   4. Boundary extrapolation
 *   5. Prolongation into fine ghosts
 * CPU backend: no-op (CPU path uses ghost_fill_from_coarser).
 */
void backend_cross_level_ghost_fill_packed(
    meshblock_pack_t *fine_pack,
    meshblock_pack_t *coarse_pack,
    double frac);

/*
 * Upload cross-level neighbor map to device for a pack.
 * Called after build_cross_level_map in rk4.c.
 * Stores the map in the pack's device handle.
 * CPU backend: no-op.
 */
void backend_upload_cross_level_map(meshblock_pack_t *pack,
                                     const int *map, int count);

/*
 * Trilinear restriction (cell averaging): fine pack → buffer blocks in coarse pack.
 * Buffer blocks are non-leaf parents at [n_evolve, n_blocks) in coarse pack.
 * Each buffer block's 8 children are in the fine pack.
 * Averages 2×2×2 = 8 fine cells per coarse cell (all positive weights).
 * Requires fine ghost zones to be valid (call ghost_exchange + cross_level_fill first).
 * CPU backend: no-op (CPU path uses restrict_level_to_parents on mesh blocks).
 * Ref: AthenaK device-resident restriction.
 */
void backend_restrict_to_buffers_packed(meshblock_pack_t *fine_pack,
                                         meshblock_pack_t *coarse_pack);

/*
 * Upload restriction child map to device for a coarse pack.
 * Map format: [n_buffers * 8] — for each buffer block, 8 fine pack indices (children).
 * -1 = child not in fine pack (shouldn't happen in valid mesh).
 */
void backend_upload_restrict_map(meshblock_pack_t *coarse_pack,
                                   const int *map, int n_buffers);

/* ========================================================================
 * GPU diagnostic kernels
 *
 * Lightweight on-device diagnostics: constraints, lapse, separation, NaN.
 * Data stays on device; only tiny scalar results (~100 bytes) return to host.
 *
 * GPU: map pack data with backend_map_pack_diag (data + metadata only, no
 * rhs/scratch/accum), run ghost exchange, then call diagnostic kernels.
 * CPU: map/unmap are no-ops; diagnostic functions call existing mesh-level
 * CPU code directly.
 * ======================================================================== */

/*
 * Lightweight diagnostic-only pack mapping (data + metadata, no rhs/scratch).
 * Saves ~75% transfer vs full backend_map_pack.
 * CPU backend: no-op.
 */
void backend_map_pack_diag(meshblock_pack_t *pack);

/*
 * Free diagnostic pack device memory WITHOUT syncing data back (read-only).
 * CPU backend: no-op.
 */
void backend_unmap_pack_diag(meshblock_pack_t *pack);

/*
 * Hamiltonian constraint L2 norm over all interior points on device.
 * Returns sqrt(sum(H^2 * dV) / sum(dV)).
 */
double backend_constraint_l2_packed(meshblock_pack_t *pack);

/*
 * Raw Hamiltonian constraint: returns sum(H^2 * dV) and sum(dV) separately.
 * Use for combining results across multiple level packs in AMR:
 *   total = sqrt((sum1 + sum2 + ...) / (vol1 + vol2 + ...))
 */
void backend_constraint_l2_raw_packed(meshblock_pack_t *pack,
                                       double *out_sum, double *out_vol);

/*
 * Momentum constraint L2 norm over all interior points on device.
 * Returns sqrt(sum(|M_i|^2 * dV) / sum(dV)).
 */
double backend_momentum_l2_packed(meshblock_pack_t *pack);

/*
 * Raw momentum constraint: returns sum(|M_i|^2 * dV) and sum(dV) separately.
 */
void backend_momentum_l2_raw_packed(meshblock_pack_t *pack,
                                      double *out_sum, double *out_vol);

/*
 * Minimum lapse over all interior points on device.
 * Returns min(alpha) and writes the position to out_x/y/z.
 */
double backend_min_lapse_packed(meshblock_pack_t *pack,
                                 double *out_x, double *out_y, double *out_z);

/*
 * BH separation from two deepest lapse minima on device.
 * excl_radius: exclusion zone around BH#1 for finding BH#2 (typically 2.0).
 * Returns distance and writes both positions to x1/y1/z1, x2/y2/z2.
 */
double backend_bh_separation_packed(meshblock_pack_t *pack, double excl_radius,
                                      double *x1, double *y1, double *z1,
                                      double *x2, double *y2, double *z2);

/*
 * Minimum lapse with exclusion zones on device.
 * Skips points within any exclusion sphere. Used by BH tracker for
 * successive lapse-minimum searches (one call per BH).
 * n_excl=0 is equivalent to backend_min_lapse_packed.
 */
double backend_min_lapse_excl_packed(meshblock_pack_t *pack,
                                       int n_excl,
                                       const double excl_centers[][3],
                                       const double *excl_radii,
                                       double *out_x, double *out_y,
                                       double *out_z);

/*
 * Check all evolved fields for NaN/Inf on device. Returns 1 if all finite.
 */
int backend_check_finite_packed(meshblock_pack_t *pack);

/*
 * GPU Psi4 extraction: compute r*Psi4 at pre-mapped angular points on device,
 * then decompose into spin-weighted spherical harmonic modes on host.
 *
 * Fills ws->re_psi4, ws->im_psi4 (r*Psi4 on sphere) and
 * ws->mode_re, ws->mode_im (mode coefficients).
 *
 * The host pre-computes angular-point-to-block mapping using mesh_find_block_at,
 * uploads the mapping, and launches a GPU kernel calling psi4_compute for each
 * angular point. Mode decomposition runs on host (trivial arithmetic on 512 values).
 *
 * GPU: requires prior backend_map_pack_diag + backend_ghost_exchange_packed.
 * CPU: calls psi4_extract directly (pack unused).
 */
struct psi4_workspace_s;  /* forward declaration */
struct mesh_s;             /* forward declaration */
void backend_psi4_extract_packed(meshblock_pack_t *pack,
                                   struct psi4_workspace_s *ws,
                                   const struct mesh_s *m);

EXTERN_C_END

#endif /* LATTICE_BACKEND_H */
