/*
 * Lattice — 3D Numerical Relativity
 * HIP GPU backend: native AMD/NVIDIA GPU kernels.
 *
 * Replaces OpenMP target offloading with native HIP kernels.
 * Works on both AMD GPUs (native) and NVIDIA GPUs (HIP CUDA backend).
 *
 * Packed API: batched kernels across all blocks in one launch.
 * Data mapped to device by backend_map_pack (hipMalloc + hipMemcpy),
 * kernels execute on device, backend_unmap_pack syncs back to host.
 *
 * GPU stack budget per thread: ~5.3 KB for batched RHS
 *   (400B field pointers + 872B grid_t + 4096B RHS locals)
 * Stack size set to 16384 in backend_init().
 *
 * Build with: make BACKEND=gpu (requires ROCm hipcc or CUDA hipcc)
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

extern "C" {
#include "backend.h"
#include "../evolution/ccz4_rhs.h"
#include "../evolution/maxwell_rhs.h"
#include "../boundary/sommerfeld.h"
#include "../boundary/constraint_preserving.h"
#include "../core/fields.h"
#include "../geometry/tensor_utils.h"
#include "../amr/block.h"
#include "../amr/restriction.h"
#include "../amr/prolongation.h"
#include "../initial_data/mg_smooth_point.h"
}

/* ========================================================================
 * Error checking macro
 * ======================================================================== */

#define HIP_CHECK(call) do {                                      \
    hipError_t err = (call);                                      \
    if (err != hipSuccess) {                                      \
        fprintf(stderr, "HIP error at %s:%d: %s\n",              \
                __FILE__, __LINE__, hipGetErrorString(err));      \
        exit(1);                                                  \
    }                                                             \
} while (0)

/* ========================================================================
 * Constant memory (~2.7 KB total)
 * Loaded once in backend_init() via hipMemcpyToSymbol.
 * ======================================================================== */

__constant__ int d_nbr_offset[26][3];                    /* 624 B  */
__constant__ double d_restrict_w[6];                     /* 48 B   */
__constant__ double d_restrict_wkj[6][6];                /* 288 B  */
__constant__ double d_prolong_w[7];                      /* 56 B   */
__constant__ double d_prolong_wkj[4][7][7];              /* 1568 B */
__constant__ double d_extrap_c[4][3];                    /* 96 B   */

/* Extrapolation coefficients for ghost boundary fill (Phase 3.5).
 * Quadratic extrapolation: p(x) through 3 interior points at x=0,1,2,
 * evaluated at x=-(d+1) for ghost depth d=0..3. */
static const double h_extrap_c[4][3] = {
    {  3.0,  -3.0,  1.0 },
    {  6.0,  -8.0,  3.0 },
    { 10.0, -15.0,  6.0 },
    { 15.0, -24.0, 10.0 },
};

/* ========================================================================
 * Persistent HIP stream (2E optimization)
 * Non-blocking launches, potential host/device overlap.
 * ======================================================================== */

static hipStream_t gpu_stream;

/* ========================================================================
 * Device pointer tracking
 * Stores device pointers for the currently mapped pack.
 * ======================================================================== */

typedef struct {
    double *data, *rhs, *scratch, *accum;
    double *origins, *dx_per_block;
    int *on_boundary, *levels, *neighbor_table, *refined_map, *nblevel_table;
    double *coarse_data;
    int *coarse_neighbor_table;
    sim_params_t *params;
    /* Compact Sommerfeld: only boundary blocks (2A optimization) */
    int *boundary_block_ids;
    int n_boundary;
    /* Sizes for cleanup */
    size_t total;
    int nb;
    int n_refined;
    size_t coarse_total;
    int n_fields;
} hip_device_ptrs_t;

static hip_device_ptrs_t d_ptrs;
static int d_ptrs_valid = 0;

/* ========================================================================
 * Backend lifecycle
 * ======================================================================== */

extern "C"
void backend_init(void)
{
    /* Set stack size for RHS kernel (~5.3 KB per thread) */
    HIP_CHECK(hipDeviceSetLimit(hipLimitStackSize, 16384));

    /* Create persistent stream for non-blocking kernel launches */
    HIP_CHECK(hipStreamCreate(&gpu_stream));

    /* Load constant memory */
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_nbr_offset), nbr_offset,
              sizeof(nbr_offset)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_restrict_w), restrict_w,
              sizeof(double) * RESTRICT_STENCIL));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_restrict_wkj), restrict_wkj,
              sizeof(double) * RESTRICT_STENCIL * RESTRICT_STENCIL));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_prolong_w), prolong_w,
              sizeof(double) * PROLONG_STENCIL));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_prolong_wkj), prolong_wkj,
              sizeof(double) * 4 * PROLONG_STENCIL * PROLONG_STENCIL));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_extrap_c), h_extrap_c,
              sizeof(h_extrap_c)));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("HIP backend: %s (%.0f MB, compute %d.%d)\n",
           prop.name, prop.totalGlobalMem / 1048576.0,
           prop.major, prop.minor);
}

extern "C"
void backend_cleanup(void)
{
    hipStreamDestroy(gpu_stream);
}

/* ========================================================================
 * Helper: allocate + copy to device
 * ======================================================================== */

static void *hip_alloc_copy(const void *host, size_t bytes)
{
    void *dev;
    HIP_CHECK(hipMalloc(&dev, bytes));
    HIP_CHECK(hipMemcpy(dev, host, bytes, hipMemcpyHostToDevice));
    return dev;
}

/* ========================================================================
 * Data management: map/unmap pack to GPU
 * ======================================================================== */

extern "C"
void backend_map_pack(meshblock_pack_t *pack, const sim_params_t *p)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    int nb = pack->n_blocks;
    size_t total_bytes = total * sizeof(double);

    memset(&d_ptrs, 0, sizeof(d_ptrs));
    d_ptrs.total = total;
    d_ptrs.nb = nb;
    d_ptrs.n_fields = pack->n_fields;

    /* Core field buffers */
    d_ptrs.data    = (double *)hip_alloc_copy(pack->data,    total_bytes);
    d_ptrs.rhs     = (double *)hip_alloc_copy(pack->rhs,     total_bytes);
    d_ptrs.scratch = (double *)hip_alloc_copy(pack->scratch,  total_bytes);
    if (pack->accum) {
        d_ptrs.accum = (double *)hip_alloc_copy(pack->accum, total_bytes);
    }

    /* Per-block metadata */
    d_ptrs.origins         = (double *)hip_alloc_copy(pack->origins,
                              nb * 3 * sizeof(double));
    d_ptrs.dx_per_block    = (double *)hip_alloc_copy(pack->dx_per_block,
                              nb * sizeof(double));
    d_ptrs.on_boundary     = (int *)hip_alloc_copy(pack->on_boundary,
                              nb * 6 * sizeof(int));
    d_ptrs.levels          = (int *)hip_alloc_copy(pack->levels,
                              nb * sizeof(int));
    d_ptrs.neighbor_table  = (int *)hip_alloc_copy(pack->neighbor_table,
                              nb * NUM_NEIGHBORS * sizeof(int));
    d_ptrs.refined_map     = (int *)hip_alloc_copy(pack->refined_map,
                              nb * sizeof(int));
    d_ptrs.nblevel_table   = (int *)hip_alloc_copy(pack->nblevel_table,
                              nb * 27 * sizeof(int));

    /* Coarse data (if present) */
    d_ptrs.n_refined = pack->n_refined;
    if (pack->coarse_data && pack->n_refined > 0) {
        size_t coarse_total = (size_t)pack->n_refined * pack->n_fields
                            * pack->coarse_npts;
        d_ptrs.coarse_total = coarse_total;
        d_ptrs.coarse_data = (double *)hip_alloc_copy(pack->coarse_data,
                              coarse_total * sizeof(double));
        d_ptrs.coarse_neighbor_table = (int *)hip_alloc_copy(
            pack->coarse_neighbor_table,
            pack->n_refined * NUM_NEIGHBORS * sizeof(int));
    }

    /* Simulation parameters (read-only) */
    d_ptrs.params = (sim_params_t *)hip_alloc_copy(p, sizeof(sim_params_t));

    /* Build compact boundary block ID list (2A optimization).
     * Only blocks with at least one on_boundary face need Sommerfeld. */
    {
        int *h_bids = (int *)malloc(nb * sizeof(int));
        int n_bdy = 0;
        for (int b = 0; b < nb; b++) {
            int has = 0;
            for (int face = 0; face < 6; face++)
                has |= pack->on_boundary[b * 6 + face];
            if (has) h_bids[n_bdy++] = b;
        }
        d_ptrs.n_boundary = n_bdy;
        d_ptrs.boundary_block_ids = NULL;
        if (n_bdy > 0) {
            d_ptrs.boundary_block_ids = (int *)hip_alloc_copy(
                h_bids, n_bdy * sizeof(int));
        }
        free(h_bids);
    }

    d_ptrs_valid = 1;
}

/* Free all device memory (shared by unmap and unmap_sync) */
static void hip_free_device_ptrs(void)
{
    hipFree(d_ptrs.data);
    hipFree(d_ptrs.rhs);
    hipFree(d_ptrs.scratch);
    if (d_ptrs.accum) hipFree(d_ptrs.accum);
    hipFree(d_ptrs.origins);
    hipFree(d_ptrs.dx_per_block);
    hipFree(d_ptrs.on_boundary);
    hipFree(d_ptrs.levels);
    hipFree(d_ptrs.neighbor_table);
    hipFree(d_ptrs.refined_map);
    hipFree(d_ptrs.nblevel_table);
    if (d_ptrs.coarse_data) hipFree(d_ptrs.coarse_data);
    if (d_ptrs.coarse_neighbor_table) hipFree(d_ptrs.coarse_neighbor_table);
    if (d_ptrs.boundary_block_ids) hipFree(d_ptrs.boundary_block_ids);
    hipFree(d_ptrs.params);
    d_ptrs_valid = 0;
}

extern "C"
void backend_unmap_pack(meshblock_pack_t *pack)
{
    (void)pack;
    if (!d_ptrs_valid) return;
    HIP_CHECK(hipStreamSynchronize(gpu_stream));
    hip_free_device_ptrs();
}

extern "C"
void backend_unmap_pack_sync(meshblock_pack_t *pack)
{
    if (!d_ptrs_valid) return;

    HIP_CHECK(hipStreamSynchronize(gpu_stream));

    size_t total_bytes = d_ptrs.total * sizeof(double);

    /* Sync field buffers back to host */
    HIP_CHECK(hipMemcpy(pack->data,    d_ptrs.data,    total_bytes,
              hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(pack->rhs,     d_ptrs.rhs,     total_bytes,
              hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(pack->scratch,  d_ptrs.scratch, total_bytes,
              hipMemcpyDeviceToHost));
    if (d_ptrs.accum && pack->accum) {
        HIP_CHECK(hipMemcpy(pack->accum, d_ptrs.accum, total_bytes,
                  hipMemcpyDeviceToHost));
    }

    /* Sync coarse data */
    if (d_ptrs.coarse_data && pack->coarse_data) {
        HIP_CHECK(hipMemcpy(pack->coarse_data, d_ptrs.coarse_data,
                  d_ptrs.coarse_total * sizeof(double),
                  hipMemcpyDeviceToHost));
    }

    hip_free_device_ptrs();
}

/* ========================================================================
 * Kernel 1: Zero buffer
 * ======================================================================== */

__global__ void hip_zero_packed(double *buf, size_t total)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) buf[tid] = 0.0;
}

extern "C"
void backend_zero_packed(meshblock_pack_t *pack, int which)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *d_buf;

    switch (which) {
        case PACK_BUF_DATA:    d_buf = d_ptrs.data;    break;
        case PACK_BUF_RHS:     d_buf = d_ptrs.rhs;     break;
        case PACK_BUF_SCRATCH: d_buf = d_ptrs.scratch;  break;
        case PACK_BUF_ACCUM:   d_buf = d_ptrs.accum;   break;
        default: return;
    }
    if (!d_buf) return;

    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    hipLaunchKernelGGL(hip_zero_packed, grid_size, block_size, 0, gpu_stream,
                       d_buf, total);
}

/* ========================================================================
 * Kernel 2: Batched RHS
 * ======================================================================== */

__global__ void hip_compute_rhs(double *data, double *rhs_data,
                                 double *dx_arr, sim_params_t *params,
                                 int nb, size_t npts, int nf,
                                 int N, int ghost, int Ntotal,
                                 int total_threads, int inner,
                                 int lo)
{
    /* Shared memory: load dx once per GPU block (2C optimization).
     * All 64 threads in a GPU block share the same mesh block b,
     * so only thread 0 loads dx_arr[b]. Saves 63 global loads. */
    __shared__ double s_dx;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    int b = tid / (inner * inner * inner);
    if (threadIdx.x == 0) s_dx = dx_arr[b];
    __syncthreads();

    int i = lo + (tid % inner);
    int j = lo + ((tid / inner) % inner);
    int k = lo + ((tid / (inner * inner)) % inner);

    /* Build per-field pointer arrays on stack */
    double *rhs_ptrs[NUM_FIELDS];
    const double *src_ptrs[NUM_FIELDS];
    for (int f = 0; f < nf; f++) {
        size_t base = (size_t)f * nb * npts + (size_t)b * npts;
        src_ptrs[f] = data + base;
        rhs_ptrs[f] = rhs_data + base;
    }

    /* Minimal grid_t on stack */
    grid_t g_local;
    memset(&g_local, 0, sizeof(grid_t));
    g_local.N        = N;
    g_local.ghost    = ghost;
    g_local.Ntotal   = Ntotal;
    g_local.npoints  = npts;
    g_local.n_fields = nf;
    g_local.dx       = s_dx;
    g_local.inv_dx   = 1.0 / s_dx;

#ifdef LATTICE_EM_ENABLED
    if (params->em_enabled)
        ccz4_maxwell_rhs_point(rhs_ptrs,
                               (const double *const *)src_ptrs,
                               &g_local, params, i, j, k);
    else
#endif
        ccz4_rhs_point(rhs_ptrs,
                       (const double *const *)src_ptrs,
                       &g_local, params, i, j, k);
}

extern "C"
void backend_compute_rhs_packed(meshblock_pack_t *pack, const sim_params_t *p)
{
    int lo = pack->ghost;
    int inner = pack->N;
    int nb = pack->n_blocks;
    int total_threads = nb * inner * inner * inner;

    int block_size = 64;  /* Smaller block for high-register RHS kernel */
    int grid_size = (total_threads + block_size - 1) / block_size;

    hipLaunchKernelGGL(hip_compute_rhs, grid_size, block_size, 0, gpu_stream,
                       d_ptrs.data, d_ptrs.rhs, d_ptrs.dx_per_block,
                       d_ptrs.params,
                       nb, pack->npts, pack->n_fields,
                       pack->N, pack->ghost, pack->Ntotal,
                       total_threads, inner, lo);
}

/* ========================================================================
 * Kernel 3: Batched Sommerfeld / CP-BC
 * ======================================================================== */

__global__ void hip_sommerfeld(double *data, double *rhs_data,
                                double *origins, double *dx_arr,
                                int *ob_all, sim_params_t *params,
                                int *boundary_block_ids,
                                int nb, size_t npts, int nf,
                                int Nt, int ghost, int lo, int hi,
                                int total_threads, int use_cp)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    int i = tid % Nt;
    int j = (tid / Nt) % Nt;
    int k = (tid / (Nt * Nt)) % Nt;
    int b = boundary_block_ids[tid / (Nt * Nt * Nt)];

    /* Skip interior points */
    if (i >= lo && i < hi && j >= lo && j < hi && k >= lo && k < hi)
        return;

    /* Check domain boundary adjacency */
    const int *ob = ob_all + b * 6;
    int near_boundary = 0;
    if (i < lo  && ob[0]) near_boundary = 1;
    if (i >= hi && ob[1]) near_boundary = 1;
    if (j < lo  && ob[2]) near_boundary = 1;
    if (j >= hi && ob[3]) near_boundary = 1;
    if (k < lo  && ob[4]) near_boundary = 1;
    if (k >= hi && ob[5]) near_boundary = 1;

    if (!near_boundary) return;

    /* Dominant face direction and outward sign */
    int face_dir = 0, s_sign = 0;
    if      (i < lo  && ob[0]) { face_dir = 0; s_sign = -1; }
    else if (i >= hi && ob[1]) { face_dir = 0; s_sign = +1; }
    else if (j < lo  && ob[2]) { face_dir = 1; s_sign = -1; }
    else if (j >= hi && ob[3]) { face_dir = 1; s_sign = +1; }
    else if (k < lo  && ob[4]) { face_dir = 2; s_sign = -1; }
    else if (k >= hi && ob[5]) { face_dir = 2; s_sign = +1; }

    int idx = k * Nt * Nt + j * Nt + i;
    double dx = dx_arr[b];
    double inv_dx = 1.0 / dx;
    double x = origins[b*3+0] + (i - ghost + 0.5) * dx;
    double y = origins[b*3+1] + (j - ghost + 0.5) * dx;
    double z = origins[b*3+2] + (k - ghost + 0.5) * dx;
    double r = sqrt(x*x + y*y + z*z);
    if (r < 1.0e-10) r = 1.0e-10;

    int lo_off[3] = { i, j, k };
    int hi_off[3] = { Nt-1-i, Nt-1-j, Nt-1-k };
    int strides[3] = { 1, Nt, Nt*Nt };
    double loc[3] = { x, y, z };

    /* Read lapse for CP speed computation */
    size_t lapse_base = (size_t)FIELD_LAPSE * nb * npts + (size_t)b * npts;
    double alpha = data[lapse_base + idx];

    for (int field = 0; field < nf; field++) {
        size_t base = (size_t)field * nb * npts + (size_t)b * npts;
        const double *src_f = data + base;
        double *rhs_f = rhs_data + base;

        double speed = use_cp ? cp_char_speed(field, face_dir, alpha) : 0.0;

        if (speed > 0.0) {
            double df_ds = boundary_d1(src_f, idx, strides[face_dir],
                                        lo_off[face_dir], hi_off[face_dir], inv_dx);
            double f_asym = asymptotic_value(field);
            rhs_f[idx] = cp_rhs(alpha, speed, s_sign, df_ds,
                                src_f[idx], f_asym, r);
        } else {
            double sommerfeld = 0.0;
            for (int dir = 0; dir < 3; dir++) {
                double d1 = boundary_d1(src_f, idx, strides[dir],
                                         lo_off[dir], hi_off[dir], inv_dx);
                sommerfeld += -d1 * loc[dir] / r;
            }
            double f_asym = asymptotic_value(field);
            sommerfeld += (f_asym - src_f[idx]) / r;
            rhs_f[idx] = sommerfeld;
        }
    }
}

extern "C"
void backend_sommerfeld_packed(meshblock_pack_t *pack, const sim_params_t *p)
{
    /* Compact Sommerfeld (2A optimization): launch only boundary blocks.
     * Typical AMR meshes: 70-90% of blocks are interior (no boundary faces).
     * Launching n_boundary instead of nb eliminates idle threads. */
    if (d_ptrs.n_boundary == 0) return;

    int lo = pack->ghost;
    int hi = pack->ghost + pack->N;
    int Nt = pack->Ntotal;
    int nb = pack->n_blocks;
    int total_threads = d_ptrs.n_boundary * Nt * Nt * Nt;
    int use_cp = (p->bc_type == BC_CONSTRAINT_PRESERVING);

    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    hipLaunchKernelGGL(hip_sommerfeld, grid_size, block_size, 0, gpu_stream,
                       d_ptrs.data, d_ptrs.rhs, d_ptrs.origins,
                       d_ptrs.dx_per_block, d_ptrs.on_boundary,
                       d_ptrs.params, d_ptrs.boundary_block_ids,
                       nb, pack->npts, pack->n_fields,
                       Nt, pack->ghost, lo, hi,
                       total_threads, use_cp);
}

/* ========================================================================
 * Kernel 4: CK45 fused update
 * ======================================================================== */

__global__ void hip_update_ck45(double *data_buf, double *scratch_buf,
                                 const double *rhs_buf,
                                 double A_s, double B_s, double dt,
                                 size_t total)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    scratch_buf[tid] = A_s * scratch_buf[tid] + dt * rhs_buf[tid];
    data_buf[tid]   += B_s * scratch_buf[tid];
}

extern "C"
void backend_update_ck45_packed(meshblock_pack_t *pack,
                                 double A_s, double B_s, double dt)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);

    hipLaunchKernelGGL(hip_update_ck45, grid_size, block_size, 0, gpu_stream,
                       d_ptrs.data, d_ptrs.scratch, d_ptrs.rhs,
                       A_s, B_s, dt, total);
}

/* ========================================================================
 * Kernel 5: Copy between pack buffers
 * ======================================================================== */

__global__ void hip_copy(double *dst, const double *src, size_t total)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) dst[tid] = src[tid];
}

extern "C"
void backend_copy_packed(meshblock_pack_t *pack, int dst, int src)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *dst_buf = NULL, *src_buf = NULL;

    switch (dst) {
        case PACK_BUF_DATA:    dst_buf = d_ptrs.data;    break;
        case PACK_BUF_RHS:     dst_buf = d_ptrs.rhs;     break;
        case PACK_BUF_SCRATCH: dst_buf = d_ptrs.scratch;  break;
        case PACK_BUF_ACCUM:   dst_buf = d_ptrs.accum;   break;
    }
    switch (src) {
        case PACK_BUF_DATA:    src_buf = d_ptrs.data;    break;
        case PACK_BUF_RHS:     src_buf = d_ptrs.rhs;     break;
        case PACK_BUF_SCRATCH: src_buf = d_ptrs.scratch;  break;
        case PACK_BUF_ACCUM:   src_buf = d_ptrs.accum;   break;
    }
    if (!dst_buf || !src_buf) return;

    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    hipLaunchKernelGGL(hip_copy, grid_size, block_size, 0, gpu_stream,
                       dst_buf, src_buf, total);
}

/* ========================================================================
 * Kernel 6: Accumulate weighted RHS
 * ======================================================================== */

__global__ void hip_accum_add(double *accum_buf, const double *rhs_buf,
                               double coeff, size_t total)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) accum_buf[tid] += coeff * rhs_buf[tid];
}

extern "C"
void backend_accum_add_packed(meshblock_pack_t *pack, double weight, double dt)
{
    if (!d_ptrs.accum) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double coeff = weight * dt;

    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    hipLaunchKernelGGL(hip_accum_add, grid_size, block_size, 0, gpu_stream,
                       d_ptrs.accum, d_ptrs.rhs, coeff, total);
}

/* ========================================================================
 * Kernel 7: AXPY (data = scratch + alpha*dt*rhs)
 * ======================================================================== */

__global__ void hip_axpy(double *data_buf, const double *scratch_buf,
                          const double *rhs_buf, double coeff, size_t total)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) data_buf[tid] = scratch_buf[tid] + coeff * rhs_buf[tid];
}

extern "C"
void backend_axpy_packed(meshblock_pack_t *pack, double alpha, double dt)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double coeff = alpha * dt;

    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    hipLaunchKernelGGL(hip_axpy, grid_size, block_size, 0, gpu_stream,
                       d_ptrs.data, d_ptrs.scratch, d_ptrs.rhs, coeff, total);
}

/* ========================================================================
 * Kernel 8: Apply accumulator (data += accum)
 * ======================================================================== */

__global__ void hip_apply_accum(double *data_buf, const double *accum_buf,
                                 size_t total)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) data_buf[tid] += accum_buf[tid];
}

extern "C"
void backend_apply_accum_packed(meshblock_pack_t *pack)
{
    if (!d_ptrs.accum) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;

    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    hipLaunchKernelGGL(hip_apply_accum, grid_size, block_size, 0, gpu_stream,
                       d_ptrs.data, d_ptrs.accum, total);
}

/* ========================================================================
 * Kernel 8a: Fused RK4 stage update (2B optimization)
 * accum += w*dt*rhs;  data = scratch + a*dt*rhs
 * Replaces separate accum_add + axpy, saves 1 kernel launch + 1 rhs read.
 * ======================================================================== */

__global__ void hip_rk4_stage(double *data_buf, const double *scratch_buf,
                               double *accum_buf, const double *rhs_buf,
                               double w_dt, double a_dt, size_t total)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    double r = rhs_buf[tid];
    accum_buf[tid] += w_dt * r;
    data_buf[tid]   = scratch_buf[tid] + a_dt * r;
}

extern "C"
void backend_rk4_stage_packed(meshblock_pack_t *pack,
                               double weight, double alpha, double dt)
{
    if (!d_ptrs.accum) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double w_dt = weight * dt;
    double a_dt = alpha * dt;

    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    hipLaunchKernelGGL(hip_rk4_stage, grid_size, block_size, 0, gpu_stream,
                       d_ptrs.data, d_ptrs.scratch, d_ptrs.accum, d_ptrs.rhs,
                       w_dt, a_dt, total);
}

/* ========================================================================
 * Kernel 8b: Fused RK4 final update (2B optimization)
 * data = scratch + accum + w*dt*rhs
 * Replaces accum_add + copy + apply_accum, saves 2 kernel launches.
 * ======================================================================== */

__global__ void hip_rk4_final(double *data_buf, const double *scratch_buf,
                               const double *accum_buf, const double *rhs_buf,
                               double w_dt, size_t total)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    data_buf[tid] = scratch_buf[tid] + accum_buf[tid] + w_dt * rhs_buf[tid];
}

extern "C"
void backend_rk4_final_packed(meshblock_pack_t *pack, double weight, double dt)
{
    if (!d_ptrs.accum) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double w_dt = weight * dt;

    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    hipLaunchKernelGGL(hip_rk4_final, grid_size, block_size, 0, gpu_stream,
                       d_ptrs.data, d_ptrs.scratch, d_ptrs.accum, d_ptrs.rhs,
                       w_dt, total);
}

/* ========================================================================
 * Ghost exchange kernels (9-13) + orchestrator (14)
 *
 * Device-side multi-level ghost exchange: zero PCIe DMA.
 * 7 kernel launches total (1+1+1+3+1). Uniform meshes: 1 launch.
 * ======================================================================== */

/* Ghost range helper */
__device__ static inline void ghost_range_pack(int offset, int ghost, int N,
                                                int Nt, int *dst_lo,
                                                int *dst_hi, int *src_lo,
                                                int *src_hi)
{
    int hi = ghost + N;
    if (offset == -1) {
        *dst_lo = 0;       *dst_hi = ghost;
        *src_lo = hi - ghost; *src_hi = hi;
    } else if (offset == 0) {
        *dst_lo = ghost;   *dst_hi = hi;
        *src_lo = ghost;   *src_hi = hi;
    } else {
        *dst_lo = hi;      *dst_hi = Nt;
        *src_lo = ghost;   *src_hi = ghost + (Nt - hi);
    }
}

/* Kernel 9: Same-level ghost exchange */
__global__ void hip_ghost_same_level(double *data, int *neighbors, int *levels,
                                      int nb, int ghost, int N, int Nt,
                                      size_t npts, int nf)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nb * NUM_NEIGHBORS;
    if (tid >= total) return;

    int b = tid / NUM_NEIGHBORS;
    int n = tid % NUM_NEIGHBORS;

    int nbr = neighbors[b * NUM_NEIGHBORS + n];
    if (nbr < 0) return;
    if (levels[b] != levels[nbr]) return;

    int ox = d_nbr_offset[n][0];
    int oy = d_nbr_offset[n][1];
    int oz = d_nbr_offset[n][2];

    int dx_lo, dx_hi, sx_lo, sx_hi;
    int dy_lo, dy_hi, sy_lo, sy_hi;
    int dz_lo, dz_hi, sz_lo, sz_hi;
    ghost_range_pack(ox, ghost, N, Nt, &dx_lo, &dx_hi, &sx_lo, &sx_hi);
    ghost_range_pack(oy, ghost, N, Nt, &dy_lo, &dy_hi, &sy_lo, &sy_hi);
    ghost_range_pack(oz, ghost, N, Nt, &dz_lo, &dz_hi, &sz_lo, &sz_hi);

    for (int f = 0; f < nf; f++) {
        size_t dst_base = (size_t)f * nb * npts + (size_t)b * npts;
        size_t src_base = (size_t)f * nb * npts + (size_t)nbr * npts;

        for (int kk = 0; kk < (dz_hi - dz_lo); kk++) {
            for (int jj = 0; jj < (dy_hi - dy_lo); jj++) {
                for (int ii = 0; ii < (dx_hi - dx_lo); ii++) {
                    size_t d = dst_base
                        + (size_t)(dz_lo + kk) * Nt * Nt
                        + (size_t)(dy_lo + jj) * Nt + (dx_lo + ii);
                    size_t s = src_base
                        + (size_t)(sz_lo + kk) * Nt * Nt
                        + (size_t)(sy_lo + jj) * Nt + (sx_lo + ii);
                    data[d] = data[s];
                }
            }
        }
    }
}

/* Kernel 10: Restrict fine → coarse_data (6th-order) */
__global__ void hip_ghost_restrict(double *pk_data, double *coarse_data,
                                    int *refined_map,
                                    int nb, int ghost_f, int Nt_f,
                                    int ghost_c, int N_c, int Nt_c,
                                    size_t npts, size_t cnpts, int nf)
{
    int ck_hi = ghost_c + N_c;
    int total = nb * nf * N_c * N_c;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int b  = tid / (nf * N_c * N_c);
    int rem = tid % (nf * N_c * N_c);
    int f  = rem / (N_c * N_c);
    int rem2 = rem % (N_c * N_c);
    int ck_off = rem2 / N_c;
    int cj_off = rem2 % N_c;
    int ck = ghost_c + ck_off;
    int cj = ghost_c + cj_off;

    int r = refined_map[b];
    if (r < 0) return;

    const double *src = pk_data + (size_t)f * nb * npts + (size_t)b * npts;
    double *dst = coarse_data + (size_t)r * nf * cnpts + (size_t)f * cnpts;

    int fk_base = 2 * (ck - ghost_c) + ghost_f;
    int fj_base = 2 * (cj - ghost_c) + ghost_f;

    for (int ci = ghost_c; ci < ck_hi; ci++) {
        int fi_base = 2 * (ci - ghost_c) + ghost_f;

        double val = 0.0;
        for (int sk = 0; sk < 6; sk++) {
            int fk = fk_base - 2 + sk;
            for (int sj = 0; sj < 6; sj++) {
                double wkj = d_restrict_wkj[sk][sj];
                int fj = fj_base - 2 + sj;
                for (int si = 0; si < 6; si++) {
                    int fi = fi_base - 2 + si;
                    val += wkj * d_restrict_w[si]
                        * src[fi + fj*Nt_f + fk*Nt_f*Nt_f];
                }
            }
        }
        dst[ci + cj*Nt_c + ck*Nt_c*Nt_c] = val;
    }
}

/* Kernel 11: Fill coarse_buf ghosts */
__global__ void hip_ghost_coarse_fill(double *pk_data, double *coarse_data,
                                       int *refined_map, int *levels,
                                       int *nblevel_table, int *neighbor_table,
                                       int *coarse_nbr_table,
                                       double *origins, double *dx_arr,
                                       int nb, int ghost, int N_c, int Nt_c,
                                       int Nt_f, size_t npts, size_t cnpts,
                                       int nf)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nb * NUM_NEIGHBORS;
    if (tid >= total) return;

    int b = tid / NUM_NEIGHBORS;
    int n = tid % NUM_NEIGHBORS;

    int r = refined_map[b];
    if (r < 0) return;

    int blk_level = levels[b];
    int ox = d_nbr_offset[n][0];
    int oy = d_nbr_offset[n][1];
    int oz = d_nbr_offset[n][2];
    int nlev = nblevel_table[b*27 + (oz+1)*9 + (oy+1)*3 + (ox+1)];

    if (nlev == blk_level) {
        int coarse_nbr = coarse_nbr_table[r * NUM_NEIGHBORS + n];
        if (coarse_nbr < 0) return;

        int dx_lo, dx_hi, sx_lo, sx_hi;
        int dy_lo, dy_hi, sy_lo, sy_hi;
        int dz_lo, dz_hi, sz_lo, sz_hi;
        ghost_range_pack(ox, ghost, N_c, Nt_c, &dx_lo, &dx_hi, &sx_lo, &sx_hi);
        ghost_range_pack(oy, ghost, N_c, Nt_c, &dy_lo, &dy_hi, &sy_lo, &sy_hi);
        ghost_range_pack(oz, ghost, N_c, Nt_c, &dz_lo, &dz_hi, &sz_lo, &sz_hi);

        for (int f = 0; f < nf; f++) {
            size_t dst_off = (size_t)r * nf * cnpts + (size_t)f * cnpts;
            size_t src_off = (size_t)coarse_nbr * nf * cnpts + (size_t)f * cnpts;

            for (int kk = 0; kk < (dz_hi - dz_lo); kk++) {
                for (int jj = 0; jj < (dy_hi - dy_lo); jj++) {
                    for (int ii = 0; ii < (dx_hi - dx_lo); ii++) {
                        size_t dd = dst_off
                            + (size_t)(dz_lo+kk) * Nt_c * Nt_c
                            + (size_t)(dy_lo+jj) * Nt_c + (dx_lo+ii);
                        size_t ss = src_off
                            + (size_t)(sz_lo+kk) * Nt_c * Nt_c
                            + (size_t)(sy_lo+jj) * Nt_c + (sx_lo+ii);
                        coarse_data[dd] = coarse_data[ss];
                    }
                }
            }
        }
    } else if (nlev >= 0 && nlev == blk_level - 1) {
        int pack_nbr = neighbor_table[b * NUM_NEIGHBORS + n];
        if (pack_nbr < 0) return;

        double dx_c = dx_arr[pack_nbr];
        int off_i = (int)round((origins[b*3+0] - origins[pack_nbr*3+0]) / dx_c);
        int off_j = (int)round((origins[b*3+1] - origins[pack_nbr*3+1]) / dx_c);
        int off_k = (int)round((origins[b*3+2] - origins[pack_nbr*3+2]) / dx_c);

        int dx_lo, dx_hi, dummy1, dummy2;
        int dy_lo, dy_hi, dummy3, dummy4;
        int dz_lo, dz_hi, dummy5, dummy6;
        ghost_range_pack(ox, ghost, N_c, Nt_c, &dx_lo, &dx_hi, &dummy1, &dummy2);
        ghost_range_pack(oy, ghost, N_c, Nt_c, &dy_lo, &dy_hi, &dummy3, &dummy4);
        ghost_range_pack(oz, ghost, N_c, Nt_c, &dz_lo, &dz_hi, &dummy5, &dummy6);

        for (int f = 0; f < nf; f++) {
            size_t dst_off = (size_t)r * nf * cnpts + (size_t)f * cnpts;
            size_t src_off = (size_t)f * nb * npts + (size_t)pack_nbr * npts;

            for (int kk = dz_lo; kk < dz_hi; kk++) {
                int sk = kk + off_k;
                if (sk < 0 || sk >= Nt_f) continue;
                for (int jj = dy_lo; jj < dy_hi; jj++) {
                    int sj = jj + off_j;
                    if (sj < 0 || sj >= Nt_f) continue;
                    for (int ii = dx_lo; ii < dx_hi; ii++) {
                        int si = ii + off_i;
                        if (si < 0 || si >= Nt_f) continue;
                        coarse_data[dst_off + kk*Nt_c*Nt_c + jj*Nt_c + ii] =
                            pk_data[src_off + sk*Nt_f*Nt_f + sj*Nt_f + si];
                    }
                }
            }
        }
    }
}

/* Kernel 12: Boundary extrapolation on coarse_data (single dimension) */
__global__ void hip_ghost_extrap(double *coarse_data, int *refined_map,
                                  int *nblevel_table,
                                  int nb, int gh, int N_c, int Nt_c,
                                  size_t cnpts, int nf,
                                  int dim)
{
    /* dim: 0=X, 1=Y, 2=Z */
    int total = nb * nf * Nt_c * Nt_c;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int b   = tid / (nf * Nt_c * Nt_c);
    int rem = tid % (nf * Nt_c * Nt_c);
    int f   = rem / (Nt_c * Nt_c);
    int rem2 = rem % (Nt_c * Nt_c);
    /* outer_idx and inner_idx depend on dimension */
    int outer = rem2 / Nt_c;  /* k for dim=0, k for dim=1, j for dim=2 */
    int inner = rem2 % Nt_c;  /* j for dim=0, i for dim=1, i for dim=2 */

    int r = refined_map[b];
    if (r < 0) return;

    double *fdata = coarse_data + (size_t)r * nf * cnpts + (size_t)f * cnpts;

    /* Neighbor level lookup indices for the dimension */
    int nm_idx, np_idx;
    if (dim == 0) {
        nm_idx = 1*9 + 1*3 + 0;  /* x- */
        np_idx = 1*9 + 1*3 + 2;  /* x+ */
    } else if (dim == 1) {
        nm_idx = 1*9 + 0*3 + 1;  /* y- */
        np_idx = 1*9 + 2*3 + 1;  /* y+ */
    } else {
        nm_idx = 0*9 + 1*3 + 1;  /* z- */
        np_idx = 2*9 + 1*3 + 1;  /* z+ */
    }

    int nm = nblevel_table[b*27 + nm_idx] < 0;
    int np = nblevel_table[b*27 + np_idx] < 0;

    /* Index computation differs by dimension.
     * For X: iterate over (k=outer, j=inner), ghost varies i
     * For Y: iterate over (k=outer, i=inner), ghost varies j
     * For Z: iterate over (j=outer, i=inner), ghost varies k */
    if (nm) {
        for (int d = 0; d < gh; d++) {
            int gi = gh - 1 - d;
            size_t dst, s0, s1, s2;
            if (dim == 0) {
                dst = gi + inner*Nt_c + outer*Nt_c*Nt_c;
                s0  = gh     + inner*Nt_c + outer*Nt_c*Nt_c;
                s1  = (gh+1) + inner*Nt_c + outer*Nt_c*Nt_c;
                s2  = (gh+2) + inner*Nt_c + outer*Nt_c*Nt_c;
            } else if (dim == 1) {
                dst = inner + gi*Nt_c + outer*Nt_c*Nt_c;
                s0  = inner + gh*Nt_c     + outer*Nt_c*Nt_c;
                s1  = inner + (gh+1)*Nt_c + outer*Nt_c*Nt_c;
                s2  = inner + (gh+2)*Nt_c + outer*Nt_c*Nt_c;
            } else {
                dst = inner + outer*Nt_c + gi*Nt_c*Nt_c;
                s0  = inner + outer*Nt_c + gh*Nt_c*Nt_c;
                s1  = inner + outer*Nt_c + (gh+1)*Nt_c*Nt_c;
                s2  = inner + outer*Nt_c + (gh+2)*Nt_c*Nt_c;
            }
            fdata[dst] = d_extrap_c[d][0]*fdata[s0]
                       + d_extrap_c[d][1]*fdata[s1]
                       + d_extrap_c[d][2]*fdata[s2];
        }
    }
    if (np) {
        for (int d = 0; d < gh; d++) {
            int gi = gh + N_c + d;
            size_t dst, s0, s1, s2;
            if (dim == 0) {
                dst = gi + inner*Nt_c + outer*Nt_c*Nt_c;
                s0  = (gh+N_c-1) + inner*Nt_c + outer*Nt_c*Nt_c;
                s1  = (gh+N_c-2) + inner*Nt_c + outer*Nt_c*Nt_c;
                s2  = (gh+N_c-3) + inner*Nt_c + outer*Nt_c*Nt_c;
            } else if (dim == 1) {
                dst = inner + gi*Nt_c + outer*Nt_c*Nt_c;
                s0  = inner + (gh+N_c-1)*Nt_c + outer*Nt_c*Nt_c;
                s1  = inner + (gh+N_c-2)*Nt_c + outer*Nt_c*Nt_c;
                s2  = inner + (gh+N_c-3)*Nt_c + outer*Nt_c*Nt_c;
            } else {
                dst = inner + outer*Nt_c + gi*Nt_c*Nt_c;
                s0  = inner + outer*Nt_c + (gh+N_c-1)*Nt_c*Nt_c;
                s1  = inner + outer*Nt_c + (gh+N_c-2)*Nt_c*Nt_c;
                s2  = inner + outer*Nt_c + (gh+N_c-3)*Nt_c*Nt_c;
            }
            fdata[dst] = d_extrap_c[d][0]*fdata[s0]
                       + d_extrap_c[d][1]*fdata[s1]
                       + d_extrap_c[d][2]*fdata[s2];
        }
    }
}

/* Kernel 13: Prolongate coarse_data → fine ghosts */
__global__ void hip_ghost_prolong(double *pk_data, double *coarse_data,
                                   int *refined_map, int *levels,
                                   int *nblevel_table,
                                   int nb, int ghost_f, int N_f, int Nt_f,
                                   int ghost_c, int Nt_c,
                                   size_t npts, size_t cnpts, int nf)
{
    int total = nb * Nt_f * Nt_f * Nt_f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int b  = tid / (Nt_f * Nt_f * Nt_f);
    int rem = tid % (Nt_f * Nt_f * Nt_f);
    int fk = rem / (Nt_f * Nt_f);
    int fj = (rem / Nt_f) % Nt_f;
    int fi = rem % Nt_f;

    int r = refined_map[b];
    if (r < 0) return;

    /* Skip interior cells */
    if (fi >= ghost_f && fi < ghost_f + N_f &&
        fj >= ghost_f && fj < ghost_f + N_f &&
        fk >= ghost_f && fk < ghost_f + N_f)
        return;

    /* Determine neighbor direction */
    int ox = (fi < ghost_f) ? -1 : (fi >= ghost_f + N_f) ? 1 : 0;
    int oy = (fj < ghost_f) ? -1 : (fj >= ghost_f + N_f) ? 1 : 0;
    int oz = (fk < ghost_f) ? -1 : (fk >= ghost_f + N_f) ? 1 : 0;

    /* Skip ghosts filled by same-level neighbors */
    int blk_level = levels[b];
    int nlev = nblevel_table[b*27 + (oz+1)*9 + (oy+1)*3 + (ox+1)];
    if (nlev == blk_level) return;

    int half = 7 / 2;  /* PROLONG_STENCIL / 2 */

    /* Coarse cell mapping */
    double ci_cont = (fi - ghost_f + 0.5) / 2.0 + ghost_c - 0.5;
    double cj_cont = (fj - ghost_f + 0.5) / 2.0 + ghost_c - 0.5;
    double ck_cont = (fk - ghost_f + 0.5) / 2.0 + ghost_c - 0.5;

    int ci0 = (int)(ci_cont + 0.5);
    int cj0 = (int)(cj_cont + 0.5);
    int ck0 = (int)(ck_cont + 0.5);

    if (ci0 < half || ci0 >= Nt_c - half ||
        cj0 < half || cj0 >= Nt_c - half ||
        ck0 < half || ck0 >= Nt_c - half)
        return;

    int oi = (ci_cont >= ci0) ? 1 : 0;
    int oj = (cj_cont >= cj0) ? 1 : 0;
    int ok = (ck_cont >= ck0) ? 1 : 0;
    int combo = ok * 2 + oj;

    for (int f = 0; f < nf; f++) {
        const double *csrc = coarse_data
            + (size_t)r * nf * cnpts + (size_t)f * cnpts;
        double *fdata = pk_data
            + (size_t)f * nb * npts + (size_t)b * npts;

        double val = 0.0;
        for (int sk = 0; sk < 7; sk++) {
            for (int sj = 0; sj < 7; sj++) {
                double wkj = d_prolong_wkj[combo][sk][sj];
                for (int si = 0; si < 7; si++) {
                    int wi = oi ? (6-si) : si;
                    int idx = (ci0-half+si) + (cj0-half+sj) * Nt_c
                            + (ck0-half+sk) * Nt_c * Nt_c;
                    val += wkj * d_prolong_w[wi] * csrc[idx];
                }
            }
        }
        fdata[fi + fj*Nt_f + fk*Nt_f*Nt_f] = val;
    }
}

/* Kernel 14: Ghost exchange orchestrator */
extern "C"
void backend_ghost_exchange_packed(meshblock_pack_t *pack)
{
    int nb = pack->n_blocks;
    int ghost = pack->ghost;
    int N = pack->N;
    int Nt = pack->Ntotal;
    size_t npts = pack->npts;
    int nf = pack->n_fields;

    /* Phase 0+1: Same-level exchange */
    {
        int total = nb * NUM_NEIGHBORS;
        int bs = 256;
        int gs = (total + bs - 1) / bs;
        hipLaunchKernelGGL(hip_ghost_same_level, gs, bs, 0, gpu_stream,
                           d_ptrs.data, d_ptrs.neighbor_table, d_ptrs.levels,
                           nb, ghost, N, Nt, npts, nf);
    }

    if (pack->n_refined == 0) return;

    int ghost_c = pack->ghost;
    int N_c = pack->coarse_N;
    int Nt_c = pack->coarse_Ntotal;
    size_t cnpts = pack->coarse_npts;

    /* Phase 2: Restrict fine → coarse_buf */
    {
        int total = nb * nf * N_c * N_c;
        int bs = 256;
        int gs = (total + bs - 1) / bs;
        hipLaunchKernelGGL(hip_ghost_restrict, gs, bs, 0, gpu_stream,
                           d_ptrs.data, d_ptrs.coarse_data,
                           d_ptrs.refined_map,
                           nb, ghost, Nt, ghost_c, N_c, Nt_c,
                           npts, cnpts, nf);
    }

    /* Phase 3: Fill coarse_buf ghosts */
    {
        int total = nb * NUM_NEIGHBORS;
        int bs = 256;
        int gs = (total + bs - 1) / bs;
        hipLaunchKernelGGL(hip_ghost_coarse_fill, gs, bs, 0, gpu_stream,
                           d_ptrs.data, d_ptrs.coarse_data,
                           d_ptrs.refined_map, d_ptrs.levels,
                           d_ptrs.nblevel_table, d_ptrs.neighbor_table,
                           d_ptrs.coarse_neighbor_table,
                           d_ptrs.origins, d_ptrs.dx_per_block,
                           nb, ghost, N_c, Nt_c, Nt,
                           npts, cnpts, nf);
    }

    /* Phase 3.5: Boundary extrapolation (3 sub-kernels: X, Y, Z) */
    {
        int total = nb * nf * Nt_c * Nt_c;
        int bs = 256;
        int gs = (total + bs - 1) / bs;
        for (int dim = 0; dim < 3; dim++) {
            hipLaunchKernelGGL(hip_ghost_extrap, gs, bs, 0, gpu_stream,
                               d_ptrs.coarse_data, d_ptrs.refined_map,
                               d_ptrs.nblevel_table,
                               nb, ghost, N_c, Nt_c, cnpts, nf, dim);
        }
    }

    /* Phase 4: Prolongate → fine ghosts */
    {
        int total = nb * Nt * Nt * Nt;
        int bs = 256;
        int gs = (total + bs - 1) / bs;
        hipLaunchKernelGGL(hip_ghost_prolong, gs, bs, 0, gpu_stream,
                           d_ptrs.data, d_ptrs.coarse_data,
                           d_ptrs.refined_map, d_ptrs.levels,
                           d_ptrs.nblevel_table,
                           nb, ghost, N, Nt, ghost_c, Nt_c,
                           npts, cnpts, nf);
    }
}

/* ========================================================================
 * Kernel 15: Enforce algebraic constraints
 * ======================================================================== */

__global__ void hip_enforce_algebraic(double *data, int Nt, int nb,
                                       size_t npts, int total_threads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    int pts_per_block = Nt * Nt * Nt;
    int b = tid / pts_per_block;
    int rem = tid % pts_per_block;
    int k = rem / (Nt * Nt);
    int j = (rem / Nt) % Nt;
    int i = rem % Nt;
    int idx = k * Nt * Nt + j * Nt + i;

    #define FP(fld) (data + (size_t)(fld) * nb * npts + (size_t)b * npts)

    /* Load h_ij */
    double h_loc[3][3];
    h_loc[0][0] = FP(FIELD_H11)[idx];
    h_loc[0][1] = FP(FIELD_H12)[idx];
    h_loc[0][2] = FP(FIELD_H13)[idx];
    h_loc[1][0] = h_loc[0][1];
    h_loc[1][1] = FP(FIELD_H22)[idx];
    h_loc[1][2] = FP(FIELD_H23)[idx];
    h_loc[2][0] = h_loc[0][2];
    h_loc[2][1] = h_loc[1][2];
    h_loc[2][2] = FP(FIELD_H33)[idx];

    /* Enforce det(h) = 1 */
    double det = compute_det_sym(h_loc);
    double scale = fast_inv_cbrt(det);
    for (int a = 0; a < 3; a++)
        for (int bb = 0; bb < 3; bb++)
            h_loc[a][bb] *= scale;

    FP(FIELD_H11)[idx] = h_loc[0][0];
    FP(FIELD_H12)[idx] = h_loc[0][1];
    FP(FIELD_H13)[idx] = h_loc[0][2];
    FP(FIELD_H22)[idx] = h_loc[1][1];
    FP(FIELD_H23)[idx] = h_loc[1][2];
    FP(FIELD_H33)[idx] = h_loc[2][2];

    /* Enforce tr(A) = 0 */
    double h_UU[3][3];
    compute_inverse_sym_unit_det(h_loc, h_UU);

    double A_loc[3][3];
    A_loc[0][0] = FP(FIELD_A11)[idx];
    A_loc[0][1] = FP(FIELD_A12)[idx];
    A_loc[0][2] = FP(FIELD_A13)[idx];
    A_loc[1][0] = A_loc[0][1];
    A_loc[1][1] = FP(FIELD_A22)[idx];
    A_loc[1][2] = FP(FIELD_A23)[idx];
    A_loc[2][0] = A_loc[0][2];
    A_loc[2][1] = A_loc[1][2];
    A_loc[2][2] = FP(FIELD_A33)[idx];

    make_trace_free(A_loc, h_loc, h_UU);

    FP(FIELD_A11)[idx] = A_loc[0][0];
    FP(FIELD_A12)[idx] = A_loc[0][1];
    FP(FIELD_A13)[idx] = A_loc[0][2];
    FP(FIELD_A22)[idx] = A_loc[1][1];
    FP(FIELD_A23)[idx] = A_loc[1][2];
    FP(FIELD_A33)[idx] = A_loc[2][2];

    /* Ensure chi > 0, lapse > 0 */
    if (FP(FIELD_CHI)[idx] < 1.0e-12)
        FP(FIELD_CHI)[idx] = 1.0e-12;
    if (FP(FIELD_LAPSE)[idx] < 1.0e-12)
        FP(FIELD_LAPSE)[idx] = 1.0e-12;

    #undef FP
}

extern "C"
void backend_enforce_algebraic_packed(meshblock_pack_t *pack)
{
    int Nt = pack->Ntotal;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int total_threads = nb * Nt * Nt * Nt;

    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    hipLaunchKernelGGL(hip_enforce_algebraic, grid_size, block_size, 0, gpu_stream,
                       d_ptrs.data, Nt, nb, npts, total_threads);
}

/* ========================================================================
 * Runtime backend detection
 * ======================================================================== */

extern "C"
int backend_is_gpu(void) { return 1; }

/* ========================================================================
 * Multigrid solver — separate device pointer slots
 *
 * The solver uses its own device pointers (d_solver[]) to avoid
 * interfering with evolution d_ptrs. One slot per AMR level.
 *
 * Ref: arXiv:2510.11152 (GPU FAS multigrid, 8-color MCGS)
 * ======================================================================== */

typedef struct {
    double *data, *rhs, *scratch, *accum;
    double *dx_per_block, *origins;
    int *on_boundary, *levels, *neighbor_table, *nblevel_table;
    size_t total;
    int nb;
    int n_fields;
    int valid;
} hip_solver_ptrs_t;

static hip_solver_ptrs_t d_solver[MAX_SOLVER_SLOTS];

extern "C"
void backend_map_solver_pack(meshblock_pack_t *pack, int slot)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS) {
        fprintf(stderr, "backend_map_solver_pack: slot %d out of range\n", slot);
        return;
    }

    hip_solver_ptrs_t *sp = &d_solver[slot];
    memset(sp, 0, sizeof(*sp));

    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int nf = pack->n_fields;
    size_t total = (size_t)nf * nb * npts;
    size_t total_bytes = total * sizeof(double);

    sp->total = total;
    sp->nb = nb;
    sp->n_fields = nf;

    /* Field buffers */
    sp->data    = (double *)hip_alloc_copy(pack->data,    total_bytes);
    sp->rhs     = (double *)hip_alloc_copy(pack->rhs,     total_bytes);
    sp->scratch = (double *)hip_alloc_copy(pack->scratch,  total_bytes);
    if (pack->accum)
        sp->accum = (double *)hip_alloc_copy(pack->accum, total_bytes);

    /* Metadata */
    sp->dx_per_block   = (double *)hip_alloc_copy(pack->dx_per_block,
                          nb * sizeof(double));
    sp->origins        = (double *)hip_alloc_copy(pack->origins,
                          nb * 3 * sizeof(double));
    sp->on_boundary    = (int *)hip_alloc_copy(pack->on_boundary,
                          nb * 6 * sizeof(int));
    sp->levels         = (int *)hip_alloc_copy(pack->levels,
                          nb * sizeof(int));
    sp->neighbor_table = (int *)hip_alloc_copy(pack->neighbor_table,
                          nb * NUM_NEIGHBORS * sizeof(int));
    if (pack->nblevel_table)
        sp->nblevel_table = (int *)hip_alloc_copy(pack->nblevel_table,
                              nb * 27 * sizeof(int));

    sp->valid = 1;
}

static void hip_free_solver_ptrs(hip_solver_ptrs_t *sp)
{
    if (!sp->valid) return;
    hipFree(sp->data);
    hipFree(sp->rhs);
    hipFree(sp->scratch);
    if (sp->accum) hipFree(sp->accum);
    hipFree(sp->dx_per_block);
    hipFree(sp->origins);
    hipFree(sp->on_boundary);
    hipFree(sp->levels);
    hipFree(sp->neighbor_table);
    if (sp->nblevel_table) hipFree(sp->nblevel_table);
    sp->valid = 0;
}

extern "C"
void backend_unmap_solver_pack_sync(meshblock_pack_t *pack, int slot)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];
    if (!sp->valid) return;

    HIP_CHECK(hipStreamSynchronize(gpu_stream));

    size_t total_bytes = sp->total * sizeof(double);
    HIP_CHECK(hipMemcpy(pack->data,    sp->data,    total_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(pack->rhs,     sp->rhs,     total_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(pack->scratch,  sp->scratch, total_bytes, hipMemcpyDeviceToHost));
    if (sp->accum && pack->accum)
        HIP_CHECK(hipMemcpy(pack->accum, sp->accum, total_bytes, hipMemcpyDeviceToHost));

    hip_free_solver_ptrs(sp);
}

extern "C"
void backend_sync_solver_data_to_host(meshblock_pack_t *pack, int slot)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];
    if (!sp->valid) return;

    HIP_CHECK(hipStreamSynchronize(gpu_stream));
    size_t total_bytes = sp->total * sizeof(double);
    HIP_CHECK(hipMemcpy(pack->data, sp->data, total_bytes, hipMemcpyDeviceToHost));
    /* Also sync rhs (needed for level-0 transfer) */
    HIP_CHECK(hipMemcpy(pack->rhs, sp->rhs, total_bytes, hipMemcpyDeviceToHost));
}

extern "C"
void backend_sync_solver_data_to_device(meshblock_pack_t *pack, int slot)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];
    if (!sp->valid) return;

    size_t total_bytes = sp->total * sizeof(double);
    HIP_CHECK(hipMemcpy(sp->data, pack->data, total_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(sp->rhs,  pack->rhs,  total_bytes, hipMemcpyHostToDevice));
}

/* ========================================================================
 * MG Kernel: 1-field Newton-GS smoother (one color)
 *
 * Thread mapping: tid → (block, colored_point).
 * Each thread calls mg_smooth_1field_point from mg_smooth_point.h.
 * ======================================================================== */

__global__ void hip_mg_smooth_1field(
    double *data, const double *rhs, const double *dx_arr,
    int nb, size_t npts, int N, int ghost, int Nt,
    int c0, int c1, int c2, int total_threads)
{
    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total_threads) return;

    /* Points per color per block: (N/2)^3 */
    int half_N = N / 2;
    int pts_per_block = half_N * half_N * half_N;
    int b = tid / pts_per_block;
    int pt = tid % pts_per_block;
    if (b >= nb) return;

    int pi = pt % half_N;
    int pj = (pt / half_N) % half_N;
    int pk = pt / (half_N * half_N);
    int i = ghost + c0 + 2 * pi;
    int j = ghost + c1 + 2 * pj;
    int k = ghost + c2 + 2 * pk;

    int sx = 1, sy = Nt, sz = Nt * Nt;
    int idx = k * sz + j * sy + i;

    double dx = dx_arr[b];
    double dx2 = dx * dx;
    double inv_dx = 1.0 / dx;

    double *psi     = data + (size_t)MGP_SOL_PSI * nb * npts + (size_t)b * npts;
    const double *psi_BL = data + (size_t)MGP_BG_PSI_BL * nb * npts + (size_t)b * npts;
    const double *A2     = data + (size_t)MGP_BG_A2 * nb * npts + (size_t)b * npts;
    const double *f_psi  = rhs + (size_t)MGP_SOL_PSI * nb * npts + (size_t)b * npts;

    mg_smooth_1field_point(psi, psi_BL, A2, f_psi,
                           idx, sx, sy, sz, inv_dx, dx2);
}

/* ========================================================================
 * MG Kernel: 4-field Newton-GS smoother (one color)
 * ======================================================================== */

__global__ void hip_mg_smooth_4field(
    double *data, const double *rhs, const double *dx_arr,
    int nb, size_t npts, int N, int ghost, int Nt,
    int c0, int c1, int c2, int total_threads)
{
    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total_threads) return;

    int half_N = N / 2;
    int pts_per_block = half_N * half_N * half_N;
    int b = tid / pts_per_block;
    int pt = tid % pts_per_block;
    if (b >= nb) return;

    int pi = pt % half_N;
    int pj = (pt / half_N) % half_N;
    int pk = pt / (half_N * half_N);
    int i = ghost + c0 + 2 * pi;
    int j = ghost + c1 + 2 * pj;
    int k = ghost + c2 + 2 * pk;

    int sx = 1, sy = Nt, sz = Nt * Nt;
    int idx = k * sz + j * sy + i;

    double dx = dx_arr[b];
    double dx2 = dx * dx;
    double inv_dx = 1.0 / dx;

    size_t base = (size_t)b * npts;
    size_t stride = (size_t)nb * npts;

    double *psi = data + (size_t)MGP_SOL_PSI * stride + base;
    double *V0  = data + (size_t)MGP_SOL_V1  * stride + base;
    double *V1  = data + (size_t)MGP_SOL_V2  * stride + base;
    double *V2  = data + (size_t)MGP_SOL_V3  * stride + base;

    const double *psi_BL  = data + (size_t)MGP_BG_PSI_BL * stride + base;
    const double *A2      = data + (size_t)MGP_BG_A2      * stride + base;
    const double *R_tilde = data + (size_t)MGP_BG_RTILDE  * stride + base;
    const double *SM0     = data + (size_t)MGP_BG_SM1     * stride + base;
    const double *SM1     = data + (size_t)MGP_BG_SM2     * stride + base;
    const double *SM2     = data + (size_t)MGP_BG_SM3     * stride + base;

    const double *f_psi = rhs + (size_t)MGP_SOL_PSI * stride + base;
    const double *f_V0  = rhs + (size_t)MGP_SOL_V1  * stride + base;
    const double *f_V1  = rhs + (size_t)MGP_SOL_V2  * stride + base;
    const double *f_V2  = rhs + (size_t)MGP_SOL_V3  * stride + base;

    mg_smooth_4field_point(psi, V0, V1, V2, psi_BL, A2,
                           R_tilde, SM0, SM1, SM2,
                           f_psi, f_V0, f_V1, f_V2,
                           idx, sx, sy, sz, inv_dx, dx2);
}

extern "C"
void backend_mg_smooth_packed(meshblock_pack_t *pack, int slot, int color,
                               int four_field)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS || !d_solver[slot].valid) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];

    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;
    int c0 = color & 1, c1 = (color >> 1) & 1, c2 = (color >> 2) & 1;

    int half_N = N / 2;
    int pts_per_block = half_N * half_N * half_N;
    int total_threads = nb * pts_per_block;

    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    if (four_field) {
        hipLaunchKernelGGL(hip_mg_smooth_4field, grid_size, block_size,
                           0, gpu_stream,
                           sp->data, sp->rhs, sp->dx_per_block,
                           nb, npts, N, ghost, Nt, c0, c1, c2, total_threads);
    } else {
        hipLaunchKernelGGL(hip_mg_smooth_1field, grid_size, block_size,
                           0, gpu_stream,
                           sp->data, sp->rhs, sp->dx_per_block,
                           nb, npts, N, ghost, Nt, c0, c1, c2, total_threads);
    }
}

/* ========================================================================
 * MG Kernel: Same-level ghost exchange
 *
 * Each thread copies one point of one ghost zone slab.
 * Thread mapping: (block, neighbor_dir, slab_point).
 * ======================================================================== */

/* Solver same-level ghost exchange kernel.
 * Reuses ghost_range_pack helper and d_nbr_offset from evolution kernels.
 * One thread per (block, neighbor_direction) pair — copies the ghost slab. */
__global__ void hip_mg_ghost_same_level(
    double *data, const int *neighbor_table,
    int nb, size_t npts, int N, int ghost, int Nt,
    int n_sol)
{
    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    int total = nb * 26;
    if (tid >= total) return;

    int b = tid / 26;
    int n = tid % 26;

    int nbr = neighbor_table[b * 26 + n];
    if (nbr < 0) return;

    int ox = d_nbr_offset[n][0];
    int oy = d_nbr_offset[n][1];
    int oz = d_nbr_offset[n][2];

    int dx_lo, dx_hi, sx_lo, sx_hi;
    int dy_lo, dy_hi, sy_lo, sy_hi;
    int dz_lo, dz_hi, sz_lo, sz_hi;
    ghost_range_pack(ox, ghost, N, Nt, &dx_lo, &dx_hi, &sx_lo, &sx_hi);
    ghost_range_pack(oy, ghost, N, Nt, &dy_lo, &dy_hi, &sy_lo, &sy_hi);
    ghost_range_pack(oz, ghost, N, Nt, &dz_lo, &dz_hi, &sz_lo, &sz_hi);

    for (int f = 0; f < n_sol; f++) {
        size_t dst_base = (size_t)f * nb * npts + (size_t)b * npts;
        size_t src_base = (size_t)f * nb * npts + (size_t)nbr * npts;

        for (int kk = 0; kk < (dz_hi - dz_lo); kk++) {
            for (int jj = 0; jj < (dy_hi - dy_lo); jj++) {
                for (int ii = 0; ii < (dx_hi - dx_lo); ii++) {
                    size_t d = dst_base
                        + (size_t)(dz_lo + kk) * Nt * Nt
                        + (size_t)(dy_lo + jj) * Nt + (dx_lo + ii);
                    size_t s = src_base
                        + (size_t)(sz_lo + kk) * Nt * Nt
                        + (size_t)(sy_lo + jj) * Nt + (sx_lo + ii);
                    data[d] = data[s];
                }
            }
        }
    }
}

extern "C"
void backend_mg_ghost_same_level_packed(meshblock_pack_t *pack, int slot, int n_sol)
{
    /* Device-side same-level ghost exchange. One thread per (block, direction).
     * No PCIe transfers — all data stays on device. */
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS || !d_solver[slot].valid) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];

    int nb = pack->n_blocks;
    int total = nb * 26;
    int bs = 256;
    int gs = (total + bs - 1) / bs;

    hipLaunchKernelGGL(hip_mg_ghost_same_level, gs, bs, 0, gpu_stream,
                       sp->data, sp->neighbor_table,
                       nb, (size_t)pack->npts, pack->N, pack->ghost,
                       pack->Ntotal, n_sol);
}

/* ========================================================================
 * MG Kernel: Zero-Dirichlet BCs on boundary ghost zones
 * ======================================================================== */

__global__ void hip_mg_bc(
    double *data, const int *on_boundary,
    int nb, size_t npts, int N, int ghost, int Nt,
    int n_sol, int total_threads)
{
    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total_threads) return;

    int npts_per_block = Nt * Nt * Nt;
    int b = tid / npts_per_block;
    int pt = tid % npts_per_block;
    if (b >= nb) return;

    int i = pt % Nt;
    int j = (pt / Nt) % Nt;
    int k = pt / (Nt * Nt);

    /* Check if this point is in any boundary ghost zone */
    int is_ghost = 0;
    if (on_boundary[b * 6 + 0] && i < ghost) is_ghost = 1;
    if (on_boundary[b * 6 + 1] && i >= ghost + N) is_ghost = 1;
    if (on_boundary[b * 6 + 2] && j < ghost) is_ghost = 1;
    if (on_boundary[b * 6 + 3] && j >= ghost + N) is_ghost = 1;
    if (on_boundary[b * 6 + 4] && k < ghost) is_ghost = 1;
    if (on_boundary[b * 6 + 5] && k >= ghost + N) is_ghost = 1;

    if (!is_ghost) return;

    size_t stride = (size_t)nb * npts;
    size_t base = (size_t)b * npts;
    int idx = k * Nt * Nt + j * Nt + i;

    for (int s = 0; s < n_sol; s++)
        data[s * stride + base + idx] = 0.0;
}

extern "C"
void backend_mg_bc_packed(meshblock_pack_t *pack, int slot, int four_field)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS || !d_solver[slot].valid) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];

    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;
    int n_sol = four_field ? 4 : 1;

    int total_threads = nb * Nt * Nt * Nt;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    hipLaunchKernelGGL(hip_mg_bc, grid_size, block_size, 0, gpu_stream,
                       sp->data, sp->on_boundary,
                       nb, npts, N, ghost, Nt, n_sol, total_threads);
}

/* ========================================================================
 * MG Kernel: Operator evaluation L(u)
 * ======================================================================== */

__global__ void hip_mg_operator_1field(
    double *data, double *accum, const double *dx_arr,
    int nb, size_t npts, int N, int ghost, int Nt, int total_threads)
{
    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total_threads) return;

    int interior_pts = N * N * N;
    int b = tid / interior_pts;
    int pt = tid % interior_pts;
    if (b >= nb) return;

    int i = ghost + pt % N;
    int j = ghost + (pt / N) % N;
    int k = ghost + pt / (N * N);

    int sx = 1, sy = Nt, sz = Nt * Nt;
    int idx = k * sz + j * sy + i;
    double inv_dx = 1.0 / dx_arr[b];

    size_t stride = (size_t)nb * npts;
    size_t base = (size_t)b * npts;

    const double *psi    = data + (size_t)MGP_SOL_PSI   * stride + base;
    const double *psi_BL = data + (size_t)MGP_BG_PSI_BL * stride + base;
    const double *A2     = data + (size_t)MGP_BG_A2     * stride + base;
    double *L_psi = accum + (size_t)MGP_SOL_PSI * stride + base;

    mg_operator_1field_point(L_psi, psi, psi_BL, A2,
                             idx, sx, sy, sz, inv_dx);
}

__global__ void hip_mg_operator_4field(
    double *data, double *accum, const double *dx_arr,
    int nb, size_t npts, int N, int ghost, int Nt, int total_threads)
{
    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total_threads) return;

    int interior_pts = N * N * N;
    int b = tid / interior_pts;
    int pt = tid % interior_pts;
    if (b >= nb) return;

    int i = ghost + pt % N;
    int j = ghost + (pt / N) % N;
    int k = ghost + pt / (N * N);

    int sx = 1, sy = Nt, sz = Nt * Nt;
    int idx = k * sz + j * sy + i;
    double inv_dx = 1.0 / dx_arr[b];

    size_t stride = (size_t)nb * npts;
    size_t base = (size_t)b * npts;

    const double *psi     = data + (size_t)MGP_SOL_PSI    * stride + base;
    const double *V0      = data + (size_t)MGP_SOL_V1     * stride + base;
    const double *V1      = data + (size_t)MGP_SOL_V2     * stride + base;
    const double *V2      = data + (size_t)MGP_SOL_V3     * stride + base;
    const double *psi_BL  = data + (size_t)MGP_BG_PSI_BL  * stride + base;
    const double *A2      = data + (size_t)MGP_BG_A2      * stride + base;
    const double *R_tilde = data + (size_t)MGP_BG_RTILDE  * stride + base;
    const double *SM0     = data + (size_t)MGP_BG_SM1     * stride + base;
    const double *SM1     = data + (size_t)MGP_BG_SM2     * stride + base;
    const double *SM2     = data + (size_t)MGP_BG_SM3     * stride + base;

    double *L_psi = accum + (size_t)MGP_SOL_PSI * stride + base;
    double *L_V0  = accum + (size_t)MGP_SOL_V1  * stride + base;
    double *L_V1  = accum + (size_t)MGP_SOL_V2  * stride + base;
    double *L_V2  = accum + (size_t)MGP_SOL_V3  * stride + base;

    mg_operator_4field_point(L_psi, L_V0, L_V1, L_V2,
                             psi, V0, V1, V2,
                             psi_BL, A2, R_tilde, SM0, SM1, SM2,
                             idx, sx, sy, sz, inv_dx);
}

extern "C"
void backend_mg_operator_packed(meshblock_pack_t *pack, int slot,
                                 int four_field)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS || !d_solver[slot].valid) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];

    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;

    int interior_pts = N * N * N;
    int total_threads = nb * interior_pts;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    if (four_field) {
        hipLaunchKernelGGL(hip_mg_operator_4field, grid_size, block_size,
                           0, gpu_stream,
                           sp->data, sp->accum, sp->dx_per_block,
                           nb, npts, N, ghost, Nt, total_threads);
    } else {
        hipLaunchKernelGGL(hip_mg_operator_1field, grid_size, block_size,
                           0, gpu_stream,
                           sp->data, sp->accum, sp->dx_per_block,
                           nb, npts, N, ghost, Nt, total_threads);
    }
}

/* ========================================================================
 * MG Kernel: Residual r = f - L(u)
 * ======================================================================== */

__global__ void hip_mg_residual(
    double *accum, const double *rhs,
    int nb, size_t npts, int N, int ghost, int Nt,
    int n_sol, int total_threads)
{
    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total_threads) return;

    int interior_pts = N * N * N;
    int b = tid / interior_pts;
    int pt = tid % interior_pts;
    if (b >= nb) return;

    int i = ghost + pt % N;
    int j = ghost + (pt / N) % N;
    int k = ghost + pt / (N * N);
    int idx = k * Nt * Nt + j * Nt + i;

    size_t stride = (size_t)nb * npts;
    size_t base = (size_t)b * npts;

    for (int s = 0; s < n_sol; s++)
        accum[s * stride + base + idx] = rhs[s * stride + base + idx]
                                        - accum[s * stride + base + idx];
}

extern "C"
void backend_mg_residual_packed(meshblock_pack_t *pack, int slot,
                                 int four_field)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS || !d_solver[slot].valid) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];

    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;
    int n_sol = four_field ? 4 : 1;

    int total_threads = nb * N * N * N;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    hipLaunchKernelGGL(hip_mg_residual, grid_size, block_size, 0, gpu_stream,
                       sp->accum, sp->rhs,
                       nb, npts, N, ghost, Nt, n_sol, total_threads);
}

/* ========================================================================
 * MG Kernel: Save solution scratch = data
 * ======================================================================== */

__global__ void hip_mg_save(
    double *scratch, const double *data,
    int nb, size_t npts, int n_sol, size_t total)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    /* Only copy solution fields (0..n_sol-1) */
    size_t field_stride = (size_t)nb * npts;
    size_t field_idx = tid / field_stride;
    if ((int)field_idx >= n_sol) return;

    scratch[tid] = data[tid];
}

extern "C"
void backend_mg_save_packed(meshblock_pack_t *pack, int slot, int four_field)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS || !d_solver[slot].valid) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];

    int n_sol = four_field ? 4 : 1;
    size_t total = (size_t)n_sol * pack->n_blocks * pack->npts;

    int block_size = 256;
    int grid_size = ((int)total + block_size - 1) / block_size;

    hipLaunchKernelGGL(hip_mg_save, grid_size, block_size, 0, gpu_stream,
                       sp->scratch, sp->data,
                       pack->n_blocks, pack->npts, n_sol, total);
}

/* ========================================================================
 * MG Kernel: Tau correction rhs += accum
 * ======================================================================== */

__global__ void hip_mg_tau(
    double *rhs_buf, const double *accum,
    int nb, size_t npts, int N, int ghost, int Nt,
    int n_sol, int total_threads)
{
    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= total_threads) return;

    int interior_pts = N * N * N;
    int b = tid / interior_pts;
    int pt = tid % interior_pts;
    if (b >= nb) return;

    int i = ghost + pt % N;
    int j = ghost + (pt / N) % N;
    int k = ghost + pt / (N * N);
    int idx = k * Nt * Nt + j * Nt + i;

    size_t stride = (size_t)nb * npts;
    size_t base = (size_t)b * npts;

    for (int s = 0; s < n_sol; s++)
        rhs_buf[s * stride + base + idx] += accum[s * stride + base + idx];
}

extern "C"
void backend_mg_tau_packed(meshblock_pack_t *pack, int slot, int four_field)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS || !d_solver[slot].valid) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];

    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;
    int n_sol = four_field ? 4 : 1;

    int total_threads = nb * N * N * N;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    hipLaunchKernelGGL(hip_mg_tau, grid_size, block_size, 0, gpu_stream,
                       sp->rhs, sp->accum,
                       nb, npts, N, ghost, Nt, n_sol, total_threads);
}

/* ========================================================================
 * MG Kernel: Zero solution / RHS
 * ======================================================================== */

extern "C"
void backend_mg_zero_solution_packed(meshblock_pack_t *pack, int slot,
                                      int four_field)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS || !d_solver[slot].valid) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];

    int n_sol = four_field ? 4 : 1;
    size_t bytes = (size_t)n_sol * pack->n_blocks * pack->npts * sizeof(double);
    HIP_CHECK(hipMemsetAsync(sp->data, 0, bytes, gpu_stream));
}

extern "C"
void backend_mg_zero_rhs_packed(meshblock_pack_t *pack, int slot, int four_field)
{
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS || !d_solver[slot].valid) return;
    hip_solver_ptrs_t *sp = &d_solver[slot];

    int n_sol = four_field ? 4 : 1;
    size_t bytes = (size_t)n_sol * pack->n_blocks * pack->npts * sizeof(double);
    HIP_CHECK(hipMemsetAsync(sp->rhs, 0, bytes, gpu_stream));
}

/* ========================================================================
 * MG: Restriction, prolongation, L2 norm — host round-trip for v1
 *
 * These operations involve cross-level data movement (different packs).
 * V1 strategy: sync both levels to host, run CPU code, sync back.
 * The smoother dominates total time; these are called O(1) per V-cycle.
 * ======================================================================== */

extern "C"
void backend_mg_restrict_packed(meshblock_pack_t *fine_pack, int fine_slot,
                                 meshblock_pack_t *coarse_pack, int coarse_slot,
                                 int four_field,
                                 const int *child_map, const int *parent_ids,
                                 int n_parents)
{
    /* Sync fine data+accum to host */
    if (fine_slot >= 0 && fine_slot < MAX_SOLVER_SLOTS && d_solver[fine_slot].valid) {
        HIP_CHECK(hipStreamSynchronize(gpu_stream));
        hip_solver_ptrs_t *fsp = &d_solver[fine_slot];
        size_t bytes = fsp->total * sizeof(double);
        HIP_CHECK(hipMemcpy(fine_pack->data, fsp->data, bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(fine_pack->accum, fsp->accum, bytes, hipMemcpyDeviceToHost));
    }

    /* Sync coarse data+rhs to host (will be overwritten) */
    if (coarse_slot >= 0 && coarse_slot < MAX_SOLVER_SLOTS && d_solver[coarse_slot].valid) {
        hip_solver_ptrs_t *csp = &d_solver[coarse_slot];
        size_t bytes = csp->total * sizeof(double);
        HIP_CHECK(hipMemcpy(coarse_pack->data, csp->data, bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(coarse_pack->rhs, csp->rhs, bytes, hipMemcpyDeviceToHost));
    }

    /* Run CPU restriction logic */
    int n_sol = four_field ? 4 : 1;
    int f_nb = fine_pack->n_blocks;
    size_t f_npts = fine_pack->npts;
    int c_nb = coarse_pack->n_blocks;
    size_t c_npts = coarse_pack->npts;
    int ghost = fine_pack->ghost;
    int half_N = coarse_pack->N / 2;
    int f_Nt = fine_pack->Ntotal;
    int c_Nt = coarse_pack->Ntotal;

    for (int p = 0; p < n_parents; p++) {
        int c_b = parent_ids[p];
        for (int cz = 0; cz < 2; cz++)
            for (int cy = 0; cy < 2; cy++)
                for (int cx = 0; cx < 2; cx++) {
                    int octant = cx + (cy << 1) + (cz << 2);
                    int f_b = child_map[p * 8 + octant];
                    if (f_b < 0) continue;

                    int p_off_i = cx * half_N;
                    int p_off_j = cy * half_N;
                    int p_off_k = cz * half_N;

                    for (int s = 0; s < n_sol; s++) {
                        const double *f_sol = fine_pack->data
                            + (size_t)s * f_nb * f_npts + (size_t)f_b * f_npts;
                        double *c_sol = coarse_pack->data
                            + (size_t)s * c_nb * c_npts + (size_t)c_b * c_npts;
                        const double *f_res = fine_pack->accum
                            + (size_t)s * f_nb * f_npts + (size_t)f_b * f_npts;
                        double *c_rhs = coarse_pack->rhs
                            + (size_t)s * c_nb * c_npts + (size_t)c_b * c_npts;

                        for (int pk = 0; pk < half_N; pk++)
                            for (int pj = 0; pj < half_N; pj++)
                                for (int pi = 0; pi < half_N; pi++) {
                                    int fi = ghost + 2 * pi;
                                    int fj = ghost + 2 * pj;
                                    int fk = ghost + 2 * pk;
                                    int f000 = fk * f_Nt * f_Nt + fj * f_Nt + fi;
                                    int sxf = 1, syf = f_Nt, szf = f_Nt * f_Nt;

                                    double sol = 0.125 * (
                                        f_sol[f000] + f_sol[f000+sxf]
                                      + f_sol[f000+syf] + f_sol[f000+syf+sxf]
                                      + f_sol[f000+szf] + f_sol[f000+szf+sxf]
                                      + f_sol[f000+syf+szf] + f_sol[f000+syf+szf+sxf]);

                                    double res = 0.125 * (
                                        f_res[f000] + f_res[f000+sxf]
                                      + f_res[f000+syf] + f_res[f000+syf+sxf]
                                      + f_res[f000+szf] + f_res[f000+szf+sxf]
                                      + f_res[f000+syf+szf] + f_res[f000+syf+szf+sxf]);

                                    int ci = ghost + p_off_i + pi;
                                    int cj = ghost + p_off_j + pj;
                                    int ck = ghost + p_off_k + pk;
                                    int cidx = ck * c_Nt * c_Nt + cj * c_Nt + ci;
                                    c_sol[cidx] = sol;
                                    c_rhs[cidx] = res;
                                }
                    }
                }
    }

    /* Sync coarse back to device */
    if (coarse_slot >= 0 && coarse_slot < MAX_SOLVER_SLOTS && d_solver[coarse_slot].valid) {
        hip_solver_ptrs_t *csp = &d_solver[coarse_slot];
        size_t bytes = csp->total * sizeof(double);
        HIP_CHECK(hipMemcpy(csp->data, coarse_pack->data, bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(csp->rhs,  coarse_pack->rhs,  bytes, hipMemcpyHostToDevice));
    }
}

extern "C"
void backend_mg_prolong_add_packed(meshblock_pack_t *coarse_pack, int coarse_slot,
                                    meshblock_pack_t *fine_pack, int fine_slot,
                                    int four_field,
                                    const int *child_map, const int *parent_ids,
                                    int n_parents)
{
    /* Sync to host */
    if (coarse_slot >= 0 && coarse_slot < MAX_SOLVER_SLOTS && d_solver[coarse_slot].valid) {
        HIP_CHECK(hipStreamSynchronize(gpu_stream));
        hip_solver_ptrs_t *csp = &d_solver[coarse_slot];
        size_t bytes = csp->total * sizeof(double);
        HIP_CHECK(hipMemcpy(coarse_pack->data, csp->data, bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(coarse_pack->scratch, csp->scratch, bytes, hipMemcpyDeviceToHost));
    }
    if (fine_slot >= 0 && fine_slot < MAX_SOLVER_SLOTS && d_solver[fine_slot].valid) {
        hip_solver_ptrs_t *fsp = &d_solver[fine_slot];
        size_t bytes = fsp->total * sizeof(double);
        HIP_CHECK(hipMemcpy(fine_pack->data, fsp->data, bytes, hipMemcpyDeviceToHost));
    }

    /* Run CPU prolongation logic */
    int n_sol = four_field ? 4 : 1;
    int f_nb = fine_pack->n_blocks;
    size_t f_npts = fine_pack->npts;
    int c_nb = coarse_pack->n_blocks;
    size_t c_npts = coarse_pack->npts;
    int ghost = coarse_pack->ghost;
    int half_N = coarse_pack->N / 2;
    int f_N = fine_pack->N;
    int f_ghost = fine_pack->ghost;
    int f_Nt = fine_pack->Ntotal;
    int c_Nt = coarse_pack->Ntotal;

    for (int p = 0; p < n_parents; p++) {
        int c_b = parent_ids[p];

        /* Compute correction: scratch = data - scratch */
        for (int s = 0; s < n_sol; s++) {
            double *d = coarse_pack->data + (size_t)s * c_nb * c_npts + (size_t)c_b * c_npts;
            double *sc = coarse_pack->scratch + (size_t)s * c_nb * c_npts + (size_t)c_b * c_npts;
            for (size_t idx = 0; idx < c_npts; idx++)
                sc[idx] = d[idx] - sc[idx];
        }

        for (int cz = 0; cz < 2; cz++)
            for (int cy = 0; cy < 2; cy++)
                for (int cx = 0; cx < 2; cx++) {
                    int octant = cx + (cy << 1) + (cz << 2);
                    int f_b = child_map[p * 8 + octant];
                    if (f_b < 0) continue;

                    int c_off_i = cx * half_N;
                    int c_off_j = cy * half_N;
                    int c_off_k = cz * half_N;

                    for (int s = 0; s < n_sol; s++) {
                        const double *corr = coarse_pack->scratch
                            + (size_t)s * c_nb * c_npts + (size_t)c_b * c_npts;
                        double *fsol = fine_pack->data
                            + (size_t)s * f_nb * f_npts + (size_t)f_b * f_npts;

                        for (int fk = 0; fk < f_N; fk++) {
                            int Kc = ghost + c_off_k + fk / 2;
                            int dk = (fk % 2) ? 1 : -1;
                            for (int fj = 0; fj < f_N; fj++) {
                                int Jc = ghost + c_off_j + fj / 2;
                                int dj = (fj % 2) ? 1 : -1;
                                for (int fi = 0; fi < f_N; fi++) {
                                    int Ic = ghost + c_off_i + fi / 2;
                                    int di = (fi % 2) ? 1 : -1;

                                    double val = 0.0;
                                    for (int ck = 0; ck < 2; ck++) {
                                        int CK = ck ? Kc + dk : Kc;
                                        double wk = ck ? 0.25 : 0.75;
                                        for (int cj = 0; cj < 2; cj++) {
                                            int CJ = cj ? Jc + dj : Jc;
                                            double wkj = wk * (cj ? 0.25 : 0.75);
                                            for (int ci = 0; ci < 2; ci++) {
                                                int CI = ci ? Ic + di : Ic;
                                                val += wkj * (ci ? 0.25 : 0.75)
                                                    * corr[CK * c_Nt * c_Nt + CJ * c_Nt + CI];
                                            }
                                        }
                                    }
                                    fsol[(f_ghost + fk) * f_Nt * f_Nt
                                       + (f_ghost + fj) * f_Nt
                                       + (f_ghost + fi)] += val;
                                }
                            }
                        }
                    }
                }
    }

    /* Sync fine back to device */
    if (fine_slot >= 0 && fine_slot < MAX_SOLVER_SLOTS && d_solver[fine_slot].valid) {
        hip_solver_ptrs_t *fsp = &d_solver[fine_slot];
        size_t bytes = fsp->total * sizeof(double);
        HIP_CHECK(hipMemcpy(fsp->data, fine_pack->data, bytes, hipMemcpyHostToDevice));
    }
}

extern "C"
void backend_mg_prolong_fmg_packed(meshblock_pack_t *coarse_pack, int coarse_slot,
                                    meshblock_pack_t *fine_pack, int fine_slot,
                                    int four_field,
                                    const int *child_map, const int *parent_ids,
                                    int n_parents)
{
    /* Sync coarse to host */
    if (coarse_slot >= 0 && coarse_slot < MAX_SOLVER_SLOTS && d_solver[coarse_slot].valid) {
        HIP_CHECK(hipStreamSynchronize(gpu_stream));
        hip_solver_ptrs_t *csp = &d_solver[coarse_slot];
        size_t bytes = csp->total * sizeof(double);
        HIP_CHECK(hipMemcpy(coarse_pack->data, csp->data, bytes, hipMemcpyDeviceToHost));
    }

    /* Run CPU FMG prolongation */
    int n_sol = four_field ? 4 : 1;
    int f_nb = fine_pack->n_blocks;
    size_t f_npts = fine_pack->npts;
    int c_nb = coarse_pack->n_blocks;
    size_t c_npts = coarse_pack->npts;
    int ghost = coarse_pack->ghost;
    int half_N = coarse_pack->N / 2;
    int f_N = fine_pack->N;
    int f_ghost = fine_pack->ghost;
    int f_Nt = fine_pack->Ntotal;
    int c_Nt = coarse_pack->Ntotal;

    for (int p = 0; p < n_parents; p++) {
        int c_b = parent_ids[p];
        for (int cz = 0; cz < 2; cz++)
            for (int cy = 0; cy < 2; cy++)
                for (int cx = 0; cx < 2; cx++) {
                    int octant = cx + (cy << 1) + (cz << 2);
                    int f_b = child_map[p * 8 + octant];
                    if (f_b < 0) continue;

                    int c_off_i = cx * half_N;
                    int c_off_j = cy * half_N;
                    int c_off_k = cz * half_N;

                    for (int s = 0; s < n_sol; s++) {
                        const double *csol = coarse_pack->data
                            + (size_t)s * c_nb * c_npts + (size_t)c_b * c_npts;
                        double *fsol = fine_pack->data
                            + (size_t)s * f_nb * f_npts + (size_t)f_b * f_npts;

                        for (int fk = 0; fk < f_N; fk++) {
                            int Kc = ghost + c_off_k + fk / 2;
                            int dk = (fk % 2) ? 1 : -1;
                            for (int fj = 0; fj < f_N; fj++) {
                                int Jc = ghost + c_off_j + fj / 2;
                                int dj = (fj % 2) ? 1 : -1;
                                for (int fi = 0; fi < f_N; fi++) {
                                    int Ic = ghost + c_off_i + fi / 2;
                                    int di = (fi % 2) ? 1 : -1;

                                    double val = 0.0;
                                    for (int ck = 0; ck < 2; ck++) {
                                        int CK = ck ? Kc + dk : Kc;
                                        double wk = ck ? 0.25 : 0.75;
                                        for (int cj = 0; cj < 2; cj++) {
                                            int CJ = cj ? Jc + dj : Jc;
                                            double wkj = wk * (cj ? 0.25 : 0.75);
                                            for (int ci = 0; ci < 2; ci++) {
                                                int CI = ci ? Ic + di : Ic;
                                                val += wkj * (ci ? 0.25 : 0.75)
                                                    * csol[CK * c_Nt * c_Nt + CJ * c_Nt + CI];
                                            }
                                        }
                                    }
                                    fsol[(f_ghost + fk) * f_Nt * f_Nt
                                       + (f_ghost + fj) * f_Nt
                                       + (f_ghost + fi)] = val;
                                }
                            }
                        }
                    }
                }
    }

    /* Sync fine to device */
    if (fine_slot >= 0 && fine_slot < MAX_SOLVER_SLOTS && d_solver[fine_slot].valid) {
        hip_solver_ptrs_t *fsp = &d_solver[fine_slot];
        size_t bytes = fsp->total * sizeof(double);
        HIP_CHECK(hipMemcpy(fsp->data, fine_pack->data, bytes, hipMemcpyHostToDevice));
    }
}

extern "C"
double backend_mg_l2_norm_packed(meshblock_pack_t *pack, int slot,
                                  int four_field)
{
    /* Sync accum + rhs to host, compute L2 on CPU */
    if (slot < 0 || slot >= MAX_SOLVER_SLOTS || !d_solver[slot].valid) return 0.0;
    hip_solver_ptrs_t *sp = &d_solver[slot];

    HIP_CHECK(hipStreamSynchronize(gpu_stream));
    size_t bytes = sp->total * sizeof(double);
    HIP_CHECK(hipMemcpy(pack->accum, sp->accum, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(pack->rhs,   sp->rhs,   bytes, hipMemcpyDeviceToHost));

    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;
    int n_sol = four_field ? 4 : 1;

    double sum = 0.0;
    int count = 0;
    for (int b = 0; b < nb; b++) {
        int sy = Nt, sz = Nt * Nt;
        for (int k = ghost; k < ghost + N; k++)
            for (int j = ghost; j < ghost + N; j++)
                for (int i = ghost; i < ghost + N; i++) {
                    int idx = k * sz + j * sy + i;
                    for (int s = 0; s < n_sol; s++) {
                        size_t off = (size_t)s * nb * npts + (size_t)b * npts + idx;
                        double d = pack->rhs[off] - pack->accum[off];
                        sum += d * d;
                    }
                    count++;
                }
    }
    if (count == 0) return 0.0;
    return sqrt(sum / ((double)n_sol * count));
}
