/*
 * Lattice — 3D Numerical Relativity
 * GPU backend: OpenMP target offloading for both per-grid and packed kernels.
 *
 * Per-grid API: same triple loop as CPU, but with target teams distribute.
 * Calls ccz4_rhs_point directly — GPU can't do function pointers.
 *
 * Packed API: batched kernels across all blocks in one launch.
 * Data mapped to device by backend_map_pack, kernels execute on device,
 * backend_unmap_pack syncs back to host.
 *
 * GPU stack budget per thread: ~5.3 KB for batched RHS
 *   (400B field pointers + 872B grid_t + 4096B RHS locals)
 * Requires GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE=16384.
 *
 * Build with: clang -fopenmp -fopenmp-targets=nvptx64 (NVIDIA)
 *          or gcc -fopenmp -foffload=nvptx-none        (GCC + NVIDIA)
 */

#include "backend.h"
#include "../evolution/ccz4_rhs.h"
#include "../boundary/sommerfeld.h"
#include "../core/fields.h"
#include <string.h>
#include <math.h>

/* ========================================================================
 * Legacy per-grid API
 * ======================================================================== */

void backend_compute_rhs(double **rhs, const double *const *src,
                         const grid_t *g, const sim_params_t *p,
                         rhs_point_func_t func)
{
    (void)func; /* GPU calls ccz4_rhs_point directly */

    int lo = g->ghost;
    int hi = g->ghost + g->N;

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = lo; k < hi; k++) {
        for (int j = lo; j < hi; j++) {
            for (int i = lo; i < hi; i++) {
                ccz4_rhs_point(rhs, src, g, p, i, j, k);
            }
        }
    }
}

void backend_init(void) { /* OpenMP runtime handles GPU init */ }
void backend_cleanup(void) { /* OpenMP runtime handles GPU cleanup */ }

/* ========================================================================
 * Packed batch kernel API — GPU implementations
 *
 * All kernels use #pragma omp target teams distribute parallel for.
 * Data is mapped to device by backend_map_pack before any kernel call.
 * ======================================================================== */

/* ---- Data management: map/unmap pack to GPU ---- */

/*
 * Map all pack buffers and metadata to GPU device memory.
 * Uses omp target enter data map(to:...) for initial transfer.
 *
 * Maps:
 *   - Core buffers: data, rhs, scratch, accum (read-write)
 *   - Metadata: origins, dx_per_block, on_boundary, levels,
 *     neighbor_table, refined_map (read-only after initial transfer)
 *   - Coarse data: coarse_data (read-write, if present)
 *   - Params: sim_params_t (read-only)
 */
void backend_map_pack(meshblock_pack_t *pack, const sim_params_t *p)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    int nb = pack->n_blocks;

    /* Core field buffers */
    #pragma omp target enter data map(to: pack->data[0:total])
    #pragma omp target enter data map(to: pack->rhs[0:total])
    #pragma omp target enter data map(to: pack->scratch[0:total])
    if (pack->accum) {
        #pragma omp target enter data map(to: pack->accum[0:total])
    }

    /* Per-block metadata (read-only on device) */
    #pragma omp target enter data map(to: pack->origins[0:nb*3])
    #pragma omp target enter data map(to: pack->dx_per_block[0:nb])
    #pragma omp target enter data map(to: pack->on_boundary[0:nb*6])
    #pragma omp target enter data map(to: pack->levels[0:nb])
    #pragma omp target enter data map(to: pack->neighbor_table[0:nb*NUM_NEIGHBORS])

    /* Coarse_buf data (if present) */
    if (pack->coarse_data && pack->n_refined > 0) {
        size_t coarse_total = (size_t)pack->n_refined * pack->n_fields
                            * pack->coarse_npts;
        #pragma omp target enter data map(to: pack->coarse_data[0:coarse_total])
    }

    /* Simulation parameters (read-only) */
    #pragma omp target enter data map(to: p[0:1])
}

/*
 * Unmap pack buffers from GPU device memory.
 * Syncs modified data back to host with map(from:...).
 */
void backend_unmap_pack(meshblock_pack_t *pack)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;

    /* Sync field buffers back to host */
    #pragma omp target exit data map(from: pack->data[0:total])
    #pragma omp target exit data map(from: pack->rhs[0:total])
    #pragma omp target exit data map(from: pack->scratch[0:total])
    if (pack->accum) {
        #pragma omp target exit data map(from: pack->accum[0:total])
    }

    /* Release metadata (read-only, no sync needed) */
    int nb = pack->n_blocks;
    #pragma omp target exit data map(release: pack->origins[0:nb*3])
    #pragma omp target exit data map(release: pack->dx_per_block[0:nb])
    #pragma omp target exit data map(release: pack->on_boundary[0:nb*6])
    #pragma omp target exit data map(release: pack->levels[0:nb])
    #pragma omp target exit data map(release: pack->neighbor_table[0:nb*NUM_NEIGHBORS])

    /* Coarse_buf data */
    if (pack->coarse_data && pack->n_refined > 0) {
        size_t coarse_total = (size_t)pack->n_refined * pack->n_fields
                            * pack->coarse_npts;
        #pragma omp target exit data map(from: pack->coarse_data[0:coarse_total])
    }
}

/* ---- Zero buffer ---- */

/*
 * Zero the selected pack buffer on the GPU.
 * Flat loop over total = n_fields * n_blocks * npts.
 */
void backend_zero_packed(meshblock_pack_t *pack, int which)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *buf;

    switch (which) {
        case PACK_BUF_DATA:    buf = pack->data;    break;
        case PACK_BUF_RHS:     buf = pack->rhs;     break;
        case PACK_BUF_SCRATCH: buf = pack->scratch;  break;
        case PACK_BUF_ACCUM:   buf = pack->accum;   break;
        default: return;
    }
    if (!buf) return;

    #pragma omp target teams distribute parallel for
    for (size_t i = 0; i < total; i++)
        buf[i] = 0.0;
}

/* ---- Batched RHS ---- */

/*
 * Compute CCZ4 RHS for all interior cells of all blocks in one GPU launch.
 *
 * collapse(4) over (block, k, j, i) to maximize GPU occupancy.
 * Each GPU thread:
 *   1. Builds per-field pointer arrays on the stack (~400 bytes)
 *   2. Constructs a minimal grid_t on the stack (~872 bytes)
 *   3. Calls ccz4_rhs_point (~4096 bytes for RHS locals)
 * Total stack: ~5.3 KB per thread < 16 KB GPU stack limit.
 *
 * Ref: AthenaK task_list/calculate_fluxes.cpp (batched kernel pattern)
 */
void backend_compute_rhs_packed(meshblock_pack_t *pack, const sim_params_t *p)
{
    int lo = pack->ghost;
    int hi = pack->ghost + pack->N;
    int N  = pack->N;
    int ghost = pack->ghost;
    int Nt = pack->Ntotal;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    double *data = pack->data;
    double *rhs_data = pack->rhs;
    double *dx_arr = pack->dx_per_block;

    #pragma omp target teams distribute parallel for collapse(4)
    for (int b = 0; b < nb; b++) {
        for (int k = lo; k < hi; k++) {
            for (int j = lo; j < hi; j++) {
                for (int i = lo; i < hi; i++) {
                    /* Per-field pointers from pack layout (on GPU stack).
                     * src[f] points to field f of block b within the pack. */
                    double *rhs_ptrs[NUM_FIELDS];
                    const double *src_ptrs[NUM_FIELDS];
                    for (int f = 0; f < NUM_FIELDS; f++) {
                        size_t base = (size_t)f * nb * npts + (size_t)b * npts;
                        src_ptrs[f] = data + base;
                        rhs_ptrs[f] = rhs_data + base;
                    }

                    /* Minimal grid_t on stack for IDX/dx access.
                     * Only fields used by ccz4_rhs_point are set. */
                    grid_t g_local;
                    memset(&g_local, 0, sizeof(grid_t));
                    g_local.N       = N;
                    g_local.ghost   = ghost;
                    g_local.Ntotal  = Nt;
                    g_local.dx      = dx_arr[b];
                    g_local.npoints = npts;

                    ccz4_rhs_point(rhs_ptrs,
                                   (const double *const *)src_ptrs,
                                   &g_local, p, i, j, k);
                }
            }
        }
    }
}

/* ---- Batched Sommerfeld BCs ---- */

/*
 * Apply Sommerfeld BCs to RHS for all blocks in one GPU launch.
 *
 * collapse(4) over (block, k, j, i). Each thread checks if its point
 * is a boundary ghost cell and applies the Sommerfeld condition.
 * Interior and inter-block ghost cells are skipped.
 *
 * Calls asymptotic_value() and boundary_d1() which are declared with
 * omp declare target in sommerfeld.h/c.
 */
void backend_sommerfeld_packed(meshblock_pack_t *pack, const sim_params_t *p)
{
    (void)p;
    int lo = pack->ghost;
    int hi = pack->ghost + pack->N;
    int Nt = pack->Ntotal;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int ghost = pack->ghost;
    double *data = pack->data;
    double *rhs_data = pack->rhs;
    double *origins = pack->origins;
    double *dx_arr = pack->dx_per_block;
    int *ob_all = pack->on_boundary;

    #pragma omp target teams distribute parallel for collapse(4)
    for (int b = 0; b < nb; b++) {
        for (int k = 0; k < Nt; k++) {
            for (int j = 0; j < Nt; j++) {
                for (int i = 0; i < Nt; i++) {
                    /* Skip interior points */
                    if (i >= lo && i < hi &&
                        j >= lo && j < hi &&
                        k >= lo && k < hi)
                        continue;

                    /* Check domain boundary adjacency */
                    const int *ob = ob_all + b * 6;
                    int near_boundary = 0;
                    if (i < lo  && ob[0]) near_boundary = 1;
                    if (i >= hi && ob[1]) near_boundary = 1;
                    if (j < lo  && ob[2]) near_boundary = 1;
                    if (j >= hi && ob[3]) near_boundary = 1;
                    if (k < lo  && ob[4]) near_boundary = 1;
                    if (k >= hi && ob[5]) near_boundary = 1;

                    if (!near_boundary) continue;

                    /* Flat index within block grid */
                    int idx = k * Nt * Nt + j * Nt + i;

                    /* Physical coordinates via block origin */
                    double dx = dx_arr[b];
                    double x = origins[b*3+0] + (i - ghost + 0.5) * dx;
                    double y = origins[b*3+1] + (j - ghost + 0.5) * dx;
                    double z = origins[b*3+2] + (k - ghost + 0.5) * dx;
                    double r = sqrt(x*x + y*y + z*z);
                    if (r < 1.0e-10) r = 1.0e-10;

                    int lo_off[3] = { i, j, k };
                    int hi_off[3] = { Nt-1-i, Nt-1-j, Nt-1-k };
                    int strides[3] = { 1, Nt, Nt*Nt };
                    double loc[3] = { x, y, z };

                    /* Apply Sommerfeld to each field */
                    for (int field = 0; field < NUM_FIELDS; field++) {
                        size_t base = (size_t)field * nb * npts
                                    + (size_t)b * npts;
                        const double *src_f = data + base;
                        double *rhs_f = rhs_data + base;

                        double sommerfeld = 0.0;
                        for (int dir = 0; dir < 3; dir++) {
                            double d1 = boundary_d1(
                                src_f, idx, strides[dir],
                                lo_off[dir], hi_off[dir], dx);
                            sommerfeld += -d1 * loc[dir] / r;
                        }

                        double f_asym = asymptotic_value(field);
                        sommerfeld += (f_asym - src_f[idx]) / r;
                        rhs_f[idx] = sommerfeld;
                    }
                }
            }
        }
    }
}

/* ---- CK45 fused update ---- */

/*
 * Fused CK45 update over all fields, blocks, and points:
 *   scratch[i] = A_s * scratch[i] + dt * rhs[i]
 *   data[i]   += B_s * scratch[i]
 *
 * Single flat loop — highest bandwidth utilization on GPU.
 */
void backend_update_ck45_packed(meshblock_pack_t *pack,
                                 double A_s, double B_s, double dt)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *data_buf    = pack->data;
    double *scratch_buf = pack->scratch;
    const double *rhs_buf = pack->rhs;

    #pragma omp target teams distribute parallel for
    for (size_t i = 0; i < total; i++) {
        scratch_buf[i] = A_s * scratch_buf[i] + dt * rhs_buf[i];
        data_buf[i]   += B_s * scratch_buf[i];
    }
}

/* ---- Classic RK4 packed operations ---- */

/*
 * Copy between pack buffers on GPU: dst[i] = src[i].
 */
void backend_copy_packed(meshblock_pack_t *pack, int dst, int src)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *dst_buf = NULL, *src_buf = NULL;

    switch (dst) {
        case PACK_BUF_DATA:    dst_buf = pack->data;    break;
        case PACK_BUF_RHS:     dst_buf = pack->rhs;     break;
        case PACK_BUF_SCRATCH: dst_buf = pack->scratch;  break;
        case PACK_BUF_ACCUM:   dst_buf = pack->accum;   break;
    }
    switch (src) {
        case PACK_BUF_DATA:    src_buf = pack->data;    break;
        case PACK_BUF_RHS:     src_buf = pack->rhs;     break;
        case PACK_BUF_SCRATCH: src_buf = pack->scratch;  break;
        case PACK_BUF_ACCUM:   src_buf = pack->accum;   break;
    }
    if (!dst_buf || !src_buf) return;

    #pragma omp target teams distribute parallel for
    for (size_t i = 0; i < total; i++)
        dst_buf[i] = src_buf[i];
}

/*
 * Accumulate weighted RHS on GPU: accum[i] += weight * dt * rhs[i].
 */
void backend_accum_add_packed(meshblock_pack_t *pack, double weight, double dt)
{
    if (!pack->accum) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *accum_buf   = pack->accum;
    const double *rhs_buf = pack->rhs;
    double coeff = weight * dt;

    #pragma omp target teams distribute parallel for
    for (size_t i = 0; i < total; i++)
        accum_buf[i] += coeff * rhs_buf[i];
}

/*
 * Linear combination on GPU: data[i] = scratch[i] + alpha * dt * rhs[i].
 */
void backend_axpy_packed(meshblock_pack_t *pack, double alpha, double dt)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *data_buf    = pack->data;
    const double *scratch_buf = pack->scratch;
    const double *rhs_buf = pack->rhs;
    double coeff = alpha * dt;

    #pragma omp target teams distribute parallel for
    for (size_t i = 0; i < total; i++)
        data_buf[i] = scratch_buf[i] + coeff * rhs_buf[i];
}

/*
 * Apply accumulator on GPU: data[i] += accum[i].
 */
void backend_apply_accum_packed(meshblock_pack_t *pack)
{
    if (!pack->accum) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *data_buf  = pack->data;
    const double *accum_buf = pack->accum;

    #pragma omp target teams distribute parallel for
    for (size_t i = 0; i < total; i++)
        data_buf[i] += accum_buf[i];
}

/* ---- Ghost exchange (Commit 2 placeholder) ---- */

/*
 * GPU ghost exchange — not yet implemented (Commit 2).
 * See backend_cpu.c for the CPU fallback documentation.
 */
void backend_ghost_exchange_packed(meshblock_pack_t *pack)
{
    (void)pack;
    /* Commit 2: implement 5-phase ghost exchange as GPU kernels.
     * Phase 1: same-level face/edge/corner copy
     * Phase 2: restrict fine → coarse_buf
     * Phase 3: fill coarse_buf ghosts from siblings + coarser neighbors
     * Phase 3.5: boundary extrapolation on coarse_buf
     * Phase 4: prolongate coarse_buf → fine ghosts */
}
