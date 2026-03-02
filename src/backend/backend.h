/*
 * Lattice — 3D Numerical Relativity
 * Backend abstraction layer.
 *
 * Packed batch kernels (AMR production path):
 *     backend_*_packed() — one kernel per operation across ALL blocks.
 *     All data stays on device; ghost exchange is device-to-device.
 *     CPU backend: OpenMP parallel for mirrors of all packed functions.
 *     GPU backend: OpenMP target teams distribute parallel for.
 *
 * CPU backend = OpenMP parallel loops calling the C RHS function.
 * GPU backend = OpenMP target offloading (same C kernels on device).
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
 * Syncs data, rhs, scratch, accum, coarse_data back to host.
 * Must be called before meshblock_pack_store.
 */
void backend_unmap_pack(meshblock_pack_t *pack);

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
 * otherwise ccz4_rhs_point (both are omp declare target).
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

EXTERN_C_END

#endif /* LATTICE_BACKEND_H */
