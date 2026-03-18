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
#include "../diagnostics/constraints.h"
#include "../diagnostics/psi4.h"
#include "../amr/mesh.h"
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
    /* GPU-resident subcycling (Part B): temporal interp + cross-level map */
    double *fields_old;          /* [total] pre-step data for temporal interp  */
    int *cross_level_map;        /* [n_cross * 3] (fine_blk, dir, coarse_blk) */
    int n_cross_entries;
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

    hip_device_ptrs_t *dp = (hip_device_ptrs_t *)pack->device_handle;

    if (dp) {
        /* Re-map: device memory persists. Sync data + coarse_data only. */
        HIP_CHECK(hipMemcpyAsync(dp->data, pack->data, total_bytes,
                                  hipMemcpyHostToDevice, gpu_stream));
        if (dp->coarse_data && pack->coarse_data) {
            HIP_CHECK(hipMemcpyAsync(dp->coarse_data, pack->coarse_data,
                        dp->coarse_total * sizeof(double),
                        hipMemcpyHostToDevice, gpu_stream));
        }
        /* Update params in case they changed between steps */
        HIP_CHECK(hipMemcpyAsync(dp->params, p, sizeof(sim_params_t),
                                  hipMemcpyHostToDevice, gpu_stream));
        d_ptrs = *dp;
        d_ptrs_valid = 1;
        return;
    }

    /* First map: allocate everything */
    dp = (hip_device_ptrs_t *)calloc(1, sizeof(*dp));
    dp->total = total;
    dp->nb = nb;
    dp->n_fields = pack->n_fields;

    /* Core field buffers */
    dp->data    = (double *)hip_alloc_copy(pack->data,    total_bytes);
    dp->rhs     = (double *)hip_alloc_copy(pack->rhs,     total_bytes);
    dp->scratch = (double *)hip_alloc_copy(pack->scratch,  total_bytes);
    if (pack->accum) {
        dp->accum = (double *)hip_alloc_copy(pack->accum, total_bytes);
    }

    /* Allocate fields_old for temporal interpolation (same size as data) */
    HIP_CHECK(hipMalloc(&dp->fields_old, total_bytes));
    HIP_CHECK(hipMemset(dp->fields_old, 0, total_bytes));

    /* Per-block metadata */
    dp->origins         = (double *)hip_alloc_copy(pack->origins,
                              nb * 3 * sizeof(double));
    dp->dx_per_block    = (double *)hip_alloc_copy(pack->dx_per_block,
                              nb * sizeof(double));
    dp->on_boundary     = (int *)hip_alloc_copy(pack->on_boundary,
                              nb * 6 * sizeof(int));
    dp->levels          = (int *)hip_alloc_copy(pack->levels,
                              nb * sizeof(int));
    dp->neighbor_table  = (int *)hip_alloc_copy(pack->neighbor_table,
                              nb * NUM_NEIGHBORS * sizeof(int));
    dp->refined_map     = (int *)hip_alloc_copy(pack->refined_map,
                              nb * sizeof(int));
    dp->nblevel_table   = (int *)hip_alloc_copy(pack->nblevel_table,
                              nb * 27 * sizeof(int));

    /* Coarse data (if present) */
    dp->n_refined = pack->n_refined;
    if (pack->coarse_data && pack->n_refined > 0) {
        size_t coarse_total = (size_t)pack->n_refined * pack->n_fields
                            * pack->coarse_npts;
        dp->coarse_total = coarse_total;
        dp->coarse_data = (double *)hip_alloc_copy(pack->coarse_data,
                              coarse_total * sizeof(double));
        dp->coarse_neighbor_table = (int *)hip_alloc_copy(
            pack->coarse_neighbor_table,
            pack->n_refined * NUM_NEIGHBORS * sizeof(int));
    }

    /* Simulation parameters (read-only) */
    dp->params = (sim_params_t *)hip_alloc_copy(p, sizeof(sim_params_t));

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
        dp->n_boundary = n_bdy;
        dp->boundary_block_ids = NULL;
        if (n_bdy > 0) {
            dp->boundary_block_ids = (int *)hip_alloc_copy(
                h_bids, n_bdy * sizeof(int));
        }
        free(h_bids);
    }

    /* Cross-level map: not yet built (Part B builds it separately) */
    dp->cross_level_map = NULL;
    dp->n_cross_entries = 0;

    /* Save handle to pack for persistence */
    pack->device_handle = dp;
    d_ptrs = *dp;
    d_ptrs_valid = 1;
}

/* Free all device allocations in a hip_device_ptrs_t */
static void hip_free_device_ptrs_struct(hip_device_ptrs_t *dp)
{
    if (!dp) return;
    (void)hipFree(dp->data);
    (void)hipFree(dp->rhs);
    (void)hipFree(dp->scratch);
    if (dp->accum) (void)hipFree(dp->accum);
    if (dp->fields_old) (void)hipFree(dp->fields_old);
    (void)hipFree(dp->origins);
    (void)hipFree(dp->dx_per_block);
    (void)hipFree(dp->on_boundary);
    (void)hipFree(dp->levels);
    (void)hipFree(dp->neighbor_table);
    (void)hipFree(dp->refined_map);
    (void)hipFree(dp->nblevel_table);
    if (dp->coarse_data) (void)hipFree(dp->coarse_data);
    if (dp->coarse_neighbor_table) (void)hipFree(dp->coarse_neighbor_table);
    if (dp->boundary_block_ids) (void)hipFree(dp->boundary_block_ids);
    (void)hipFree(dp->params);
    if (dp->cross_level_map) (void)hipFree(dp->cross_level_map);
}

/*
 * Free persistent device memory for a pack and clear its handle.
 * Called from meshblock_pack_free or explicitly on regrid.
 */
extern "C"
void backend_free_pack_device(meshblock_pack_t *pack)
{
    hip_device_ptrs_t *dp = (hip_device_ptrs_t *)pack->device_handle;
    if (!dp) return;
    /* If this was the active pack, invalidate d_ptrs before freeing */
    if (d_ptrs_valid && d_ptrs.data == dp->data)
        d_ptrs_valid = 0;
    hip_free_device_ptrs_struct(dp);
    free(dp);
    pack->device_handle = NULL;
}

/*
 * Activate a pack's persistent device state as the current d_ptrs
 * without any host↔device memcpy. Zero-cost pack switching for
 * GPU-resident subcycling.
 */
extern "C"
void backend_activate_pack(meshblock_pack_t *pack)
{
    hip_device_ptrs_t *dp = (hip_device_ptrs_t *)pack->device_handle;
    if (!dp) return;
    d_ptrs = *dp;
    d_ptrs_valid = 1;
}

/*
 * Save pre-step field data on device: fields_old = data (D2D copy).
 * Used for temporal interpolation by finer levels during subcycling.
 */
extern "C"
void backend_save_old_packed(meshblock_pack_t *pack)
{
    hip_device_ptrs_t *dp = (hip_device_ptrs_t *)pack->device_handle;
    if (!dp || !dp->fields_old) return;
    size_t bytes = dp->total * sizeof(double);
    HIP_CHECK(hipMemcpyAsync(dp->fields_old, dp->data, bytes,
                              hipMemcpyDeviceToDevice, gpu_stream));
}

/* Forward declarations: ghost exchange kernels used by cross-level and solver
 * orchestrator functions before their actual definitions. */
__global__ void hip_ghost_restrict(double *pk_data, double *coarse_data,
                                    int *refined_map,
                                    int nb, int ghost_f, int Nt_f,
                                    int ghost_c, int N_c, int Nt_c,
                                    size_t npts, size_t cnpts, int nf);
__global__ void hip_ghost_coarse_fill(double *pk_data, double *coarse_data,
                                       int *refined_map, int *levels,
                                       int *nblevel_table, int *neighbor_table,
                                       int *coarse_nbr_table,
                                       double *origins, double *dx_arr,
                                       int nb, int ghost, int N_c, int Nt_c,
                                       int Nt_f, size_t npts, size_t cnpts,
                                       int nf);
__global__ void hip_ghost_extrap(double *coarse_data, int *refined_map,
                                  int *nblevel_table,
                                  int nb, int gh, int N_c, int Nt_c,
                                  size_t cnpts, int nf,
                                  int dim);
__global__ void hip_ghost_prolong(double *pk_data, double *coarse_data,
                                   int *refined_map, int *levels,
                                   int *nblevel_table,
                                   int nb, int ghost_f, int N_f, int Nt_f,
                                   int ghost_c, int Nt_c,
                                   size_t npts, size_t cnpts, int nf);
__global__ void hip_cross_level_ghost_fill(
    double *fine_coarse_data,
    const double *coarse_data_new,
    const double *coarse_data_old,
    const int *cross_map,
    int n_entries,
    const int *fine_refined_map,
    const double *fine_origins,
    const double *coarse_origins,
    const double *coarse_dx,
    double frac,
    int fine_nb, int coarse_nb,
    int ghost_c, int N_c, int Nt_c,
    int Nt_coarse_full,
    size_t fine_cnpts, size_t coarse_npts,
    int nf);

/*
 * Cross-level ghost fill entirely on device.
 * Fills fine_pack's coarse buffer and ghosts from coarse_pack's data with
 * temporal interpolation. Full 5-phase multilevel ghost exchange on device.
 *
 * Both packs must have persistent device handles (device_handle != NULL).
 * Uses fine pack's existing restrict/prolong kernels with explicit pointers
 * from both packs' device handles.
 */
extern "C"
void backend_cross_level_ghost_fill_packed(
    meshblock_pack_t *fine_pack,
    meshblock_pack_t *coarse_pack,
    double frac)
{
    hip_device_ptrs_t *fdp = (hip_device_ptrs_t *)fine_pack->device_handle;
    hip_device_ptrs_t *cdp = (hip_device_ptrs_t *)coarse_pack->device_handle;
    if (!fdp || !cdp) return;
    if (fine_pack->n_refined == 0) return;

    int nb = fine_pack->n_blocks;
    int ghost = fine_pack->ghost;
    int N_f = fine_pack->N;
    int Nt_f = fine_pack->Ntotal;
    size_t npts = fine_pack->npts;
    int nf = fine_pack->n_fields;
    int ghost_c = fine_pack->ghost;
    int N_c = fine_pack->coarse_N;
    int Nt_c = fine_pack->coarse_Ntotal;
    size_t cnpts = fine_pack->coarse_npts;
    int bs = 256;

    /* Phase 2: Restrict fine interior → fine's coarse_buf.
     * Uses fine pack's data and coarse_data on device. */
    {
        /* Activate fine pack's device state */
        d_ptrs = *fdp;
        d_ptrs_valid = 1;

        int total = nb * nf * N_c * N_c;
        int gs = (total + bs - 1) / bs;
        hipLaunchKernelGGL(hip_ghost_restrict, gs, bs, 0, gpu_stream,
                           fdp->data, fdp->coarse_data, fdp->refined_map,
                           nb, ghost, Nt_f, ghost_c, N_c, Nt_c,
                           npts, cnpts, nf);
    }

    /* Phase 3a: Same-level coarse_buf exchange among siblings.
     * Re-launches hip_ghost_coarse_fill with fine pack's pointers only
     * for the same-level branch (nlev == blk_level). */
    {
        int total = nb * NUM_NEIGHBORS;
        int gs = (total + bs - 1) / bs;
        hipLaunchKernelGGL(hip_ghost_coarse_fill, gs, bs, 0, gpu_stream,
                           fdp->data, fdp->coarse_data,
                           fdp->refined_map, fdp->levels,
                           fdp->nblevel_table, fdp->neighbor_table,
                           fdp->coarse_neighbor_table,
                           fdp->origins, fdp->dx_per_block,
                           nb, ghost, N_c, Nt_c, Nt_f,
                           npts, cnpts, nf);
    }

    /* Phase 3b: Cross-level fill from coarser pack.
     * For fine blocks with coarser-level neighbors, copy from coarse pack's
     * data (or temporally interpolate with fields_old) into fine's coarse_buf.
     *
     * This reuses hip_ghost_coarse_fill's coarser-neighbor branch (nlev ==
     * blk_level - 1) but reads from the coarse pack's data. Since that
     * branch reads from pk_data at pack_nbr, and the coarser-level neighbors
     * are NOT in the fine pack, this branch doesn't fire. Instead we need
     * a dedicated cross-pack kernel.
     *
     * For now, the cross-level fill entries are recorded in the cross_level_map.
     * If no map is built, skip (the CPU path handles this case). */
    if (fdp->cross_level_map && fdp->n_cross_entries > 0) {
        /* Launch cross-level ghost fill kernel.
         * Each entry: (fine_block, direction, coarse_block).
         * We copy a slab from coarse_pack's data (temporally interpolated)
         * into fine_pack's coarse_data. */
        int n_entries = fdp->n_cross_entries;
        int gs = (n_entries + bs - 1) / bs;

        /* Linear temporal interpolation: val = (1-frac)*old + frac*new.
         * frac = 0.0 for first sub-step, 0.5 for second. */
        hipLaunchKernelGGL(hip_cross_level_ghost_fill, gs, bs, 0, gpu_stream,
                           fdp->coarse_data,
                           cdp->data,
                           cdp->fields_old,
                           fdp->cross_level_map,
                           n_entries,
                           fdp->refined_map,
                           fdp->origins,
                           cdp->origins,
                           cdp->dx_per_block,
                           frac,
                           nb, cdp->nb,
                           ghost_c, N_c, Nt_c,
                           coarse_pack->Ntotal,
                           cnpts, coarse_pack->npts,
                           nf);
    }

    /* Phase 3.5: Boundary extrapolation on fine's coarse_data (X, Y, Z) */
    {
        int total = nb * nf * Nt_c * Nt_c;
        int gs = (total + bs - 1) / bs;
        for (int dim = 0; dim < 3; dim++) {
            hipLaunchKernelGGL(hip_ghost_extrap, gs, bs, 0, gpu_stream,
                               fdp->coarse_data, fdp->refined_map,
                               fdp->nblevel_table,
                               nb, ghost_c, N_c, Nt_c, cnpts, nf, dim);
        }
    }

    /* Phase 4: Prolongate fine's coarse_data → fine ghosts */
    {
        int total = nb * Nt_f * Nt_f * Nt_f;
        int gs = (total + bs - 1) / bs;
        hipLaunchKernelGGL(hip_ghost_prolong, gs, bs, 0, gpu_stream,
                           fdp->data, fdp->coarse_data,
                           fdp->refined_map, fdp->levels,
                           fdp->nblevel_table,
                           nb, ghost, N_f, Nt_f, ghost_c, Nt_c,
                           npts, cnpts, nf);
    }
}

/*
 * Upload cross-level neighbor map to device. Stores in the pack's
 * persistent device handle. If a previous map exists, frees it first.
 */
extern "C"
void backend_upload_cross_level_map(meshblock_pack_t *pack,
                                     const int *map, int count)
{
    hip_device_ptrs_t *dp = (hip_device_ptrs_t *)pack->device_handle;
    if (!dp) return;

    /* Free previous map if any */
    if (dp->cross_level_map) {
        (void)hipFree(dp->cross_level_map);
        dp->cross_level_map = NULL;
    }

    dp->n_cross_entries = count;
    if (count > 0) {
        dp->cross_level_map = (int *)hip_alloc_copy(map, count * 3 * sizeof(int));
    }
}

extern "C"
void backend_unmap_pack(meshblock_pack_t *pack)
{
    (void)pack;
    if (!d_ptrs_valid) return;
    HIP_CHECK(hipStreamSynchronize(gpu_stream));
    /* Do NOT free — device memory persists in pack->device_handle */
    d_ptrs_valid = 0;
}

extern "C"
void backend_unmap_pack_sync(meshblock_pack_t *pack)
{
    if (!d_ptrs_valid) return;

    HIP_CHECK(hipStreamSynchronize(gpu_stream));

    size_t total_bytes = d_ptrs.total * sizeof(double);

    /* Sync data back to host (not rhs/scratch/accum — temporary) */
    HIP_CHECK(hipMemcpy(pack->data, d_ptrs.data, total_bytes,
              hipMemcpyDeviceToHost));

    /* Sync coarse data */
    if (d_ptrs.coarse_data && pack->coarse_data) {
        HIP_CHECK(hipMemcpy(pack->coarse_data, d_ptrs.coarse_data,
                  d_ptrs.coarse_total * sizeof(double),
                  hipMemcpyDeviceToHost));
    }

    /* Do NOT free — device memory persists in pack->device_handle */
    d_ptrs_valid = 0;
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

/* Kernel 13b: Cross-level ghost fill (cross-pack, temporal interpolation).
 * Each entry in cross_map encodes (fine_block, direction, coarse_block).
 * One thread per entry. Reads from coarse pack's data (temporally
 * interpolated), writes to fine pack's coarse_data buffer.
 *
 * Used by backend_cross_level_ghost_fill_packed for GPU-resident subcycling.
 * Linear temporal interpolation: val = (1-frac)*old + frac*new.
 */
__global__ void hip_cross_level_ghost_fill(
    double *fine_coarse_data,       /* fine pack's coarse buffer (dst) */
    const double *coarse_data_new,  /* coarse pack's current data */
    const double *coarse_data_old,  /* coarse pack's pre-step data */
    const int *cross_map,           /* [n_entries * 3] */
    int n_entries,
    const int *fine_refined_map,    /* fine pack's refined_map */
    const double *fine_origins,     /* fine pack's origins */
    const double *coarse_origins,   /* coarse pack's origins */
    const double *coarse_dx,        /* coarse pack's dx_per_block */
    double frac,                    /* temporal interp fraction (0.0 or 0.5) */
    int fine_nb, int coarse_nb,
    int ghost_c, int N_c, int Nt_c,
    int Nt_coarse_full,             /* coarse pack's Ntotal */
    size_t fine_cnpts, size_t coarse_npts,
    int nf)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_entries) return;

    int fb = cross_map[e * 3 + 0];    /* fine pack block index */
    int dir = cross_map[e * 3 + 1];   /* neighbor direction 0-25 */
    int cb = cross_map[e * 3 + 2];    /* coarse pack block index */

    int r = fine_refined_map[fb];
    if (r < 0) return;

    double dx_c = coarse_dx[cb];

    /* Compute origin offset: fine block origin relative to coarse block origin,
     * in coarse grid units. This maps fine's coarse_buf coordinates to coarse
     * pack's data coordinates. */
    int off_i = (int)round((fine_origins[fb*3+0] - coarse_origins[cb*3+0]) / dx_c);
    int off_j = (int)round((fine_origins[fb*3+1] - coarse_origins[cb*3+1]) / dx_c);
    int off_k = (int)round((fine_origins[fb*3+2] - coarse_origins[cb*3+2]) / dx_c);

    int ox = d_nbr_offset[dir][0];
    int oy = d_nbr_offset[dir][1];
    int oz = d_nbr_offset[dir][2];

    /* Compute destination range in fine's coarse_buf */
    int dx_lo, dx_hi, dummy1, dummy2;
    int dy_lo, dy_hi, dummy3, dummy4;
    int dz_lo, dz_hi, dummy5, dummy6;
    ghost_range_pack(ox, ghost_c, N_c, Nt_c, &dx_lo, &dx_hi, &dummy1, &dummy2);
    ghost_range_pack(oy, ghost_c, N_c, Nt_c, &dy_lo, &dy_hi, &dummy3, &dummy4);
    ghost_range_pack(oz, ghost_c, N_c, Nt_c, &dz_lo, &dz_hi, &dummy5, &dummy6);

    double one_minus_frac = 1.0 - frac;

    for (int f = 0; f < nf; f++) {
        size_t dst_off = (size_t)r * nf * fine_cnpts + (size_t)f * fine_cnpts;
        size_t src_off_new = (size_t)f * coarse_nb * coarse_npts
                           + (size_t)cb * coarse_npts;
        size_t src_off_old = src_off_new;  /* same layout for fields_old */

        for (int kk = dz_lo; kk < dz_hi; kk++) {
            int sk = kk + off_k;
            if (sk < 0 || sk >= Nt_coarse_full) continue;
            for (int jj = dy_lo; jj < dy_hi; jj++) {
                int sj = jj + off_j;
                if (sj < 0 || sj >= Nt_coarse_full) continue;
                for (int ii = dx_lo; ii < dx_hi; ii++) {
                    int si = ii + off_i;
                    if (si < 0 || si >= Nt_coarse_full) continue;

                    size_t src_idx = (size_t)sk * Nt_coarse_full * Nt_coarse_full
                                   + (size_t)sj * Nt_coarse_full + si;
                    double val_new = coarse_data_new[src_off_new + src_idx];
                    double val_old = coarse_data_old[src_off_old + src_idx];
                    double val = one_minus_frac * val_old + frac * val_new;

                    size_t dst_idx = (size_t)kk * Nt_c * Nt_c
                                   + (size_t)jj * Nt_c + ii;
                    fine_coarse_data[dst_off + dst_idx] = val;
                }
            }
        }
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

    /* Ensure chi >= 1e-4, lapse >= 1e-4 (GRChombo convention).
     * Ref: GRChombo PositiveChiAndAlpha.hpp */
    if (FP(FIELD_CHI)[idx] < 1.0e-4)
        FP(FIELD_CHI)[idx] = 1.0e-4;
    if (FP(FIELD_LAPSE)[idx] < 1.0e-4)
        FP(FIELD_LAPSE)[idx] = 1.0e-4;

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
 * GPU diagnostic kernels
 *
 * Lightweight on-device diagnostics: constraints, lapse, separation, NaN.
 * Uses the same d_ptrs state as evolution (one mapping at a time).
 * Data stays on device; only tiny scalar results return to host.
 * ======================================================================== */

/* ---- Diagnostic-only pack mapping (data + metadata, no rhs/scratch) ---- */

extern "C"
void backend_map_pack_diag(meshblock_pack_t *pack)
{
    hip_device_ptrs_t *dp = (hip_device_ptrs_t *)pack->device_handle;

    if (dp) {
        /* Pack has persistent device memory — just sync data and activate.
         * Data only (read-only diagnostics), metadata already on device. */
        size_t total_bytes = dp->total * sizeof(double);
        HIP_CHECK(hipMemcpyAsync(dp->data, pack->data, total_bytes,
                                  hipMemcpyHostToDevice, gpu_stream));
        if (dp->coarse_data && pack->coarse_data) {
            HIP_CHECK(hipMemcpyAsync(dp->coarse_data, pack->coarse_data,
                        dp->coarse_total * sizeof(double),
                        hipMemcpyHostToDevice, gpu_stream));
        }
        d_ptrs = *dp;
        d_ptrs_valid = 1;
        return;
    }

    /* No persistent handle — allocate temporary diagnostic-only device memory.
     * This path is used for leaf packs that haven't gone through
     * backend_map_pack (e.g., diagnostics on a fresh pack). */
    int nb = pack->n_blocks;
    size_t total = (size_t)pack->n_fields * nb * pack->npts;
    size_t total_bytes = total * sizeof(double);

    memset(&d_ptrs, 0, sizeof(d_ptrs));
    d_ptrs.total = total;
    d_ptrs.nb = nb;
    d_ptrs.n_fields = pack->n_fields;

    /* Data buffer (read-only for diagnostics) */
    d_ptrs.data = (double *)hip_alloc_copy(pack->data, total_bytes);

    /* Per-block metadata needed by constraint stencils and position */
    d_ptrs.dx_per_block = (double *)hip_alloc_copy(pack->dx_per_block,
                           nb * sizeof(double));
    d_ptrs.origins      = (double *)hip_alloc_copy(pack->origins,
                           nb * 3 * sizeof(double));
    d_ptrs.on_boundary  = (int *)hip_alloc_copy(pack->on_boundary,
                           nb * 6 * sizeof(int));
    d_ptrs.levels       = (int *)hip_alloc_copy(pack->levels,
                           nb * sizeof(int));
    d_ptrs.neighbor_table = (int *)hip_alloc_copy(pack->neighbor_table,
                             nb * NUM_NEIGHBORS * sizeof(int));
    d_ptrs.refined_map  = (int *)hip_alloc_copy(pack->refined_map,
                           nb * sizeof(int));
    d_ptrs.nblevel_table = (int *)hip_alloc_copy(pack->nblevel_table,
                            nb * 27 * sizeof(int));

    /* Coarse data for multi-level ghost exchange */
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

    d_ptrs_valid = 1;
}

extern "C"
void backend_unmap_pack_diag(meshblock_pack_t *pack)
{
    (void)pack;
    if (!d_ptrs_valid) return;
    HIP_CHECK(hipStreamSynchronize(gpu_stream));

    if (pack->device_handle) {
        /* Persistent: just deactivate d_ptrs, don't free */
        d_ptrs_valid = 0;
        return;
    }

    /* Temporary allocation: free all device memory — no sync back (read-only) */
    (void)hipFree(d_ptrs.data);
    (void)hipFree(d_ptrs.origins);
    (void)hipFree(d_ptrs.dx_per_block);
    (void)hipFree(d_ptrs.on_boundary);
    (void)hipFree(d_ptrs.levels);
    (void)hipFree(d_ptrs.neighbor_table);
    (void)hipFree(d_ptrs.refined_map);
    (void)hipFree(d_ptrs.nblevel_table);
    if (d_ptrs.coarse_data) (void)hipFree(d_ptrs.coarse_data);
    if (d_ptrs.coarse_neighbor_table) (void)hipFree(d_ptrs.coarse_neighbor_table);
    if (d_ptrs.boundary_block_ids) (void)hipFree(d_ptrs.boundary_block_ids);

    d_ptrs_valid = 0;
}

/* ---- Kernel: Hamiltonian + Momentum constraint L2 partial reduction ---- */

__global__ void hip_constraint_l2_partial(
    double *data, double *dx_arr,
    int nb, size_t npts, int nf, int N, int ghost, int Nt,
    double *partial_ham, double *partial_mom, double *partial_vol,
    int total_points)
{
    extern __shared__ double sdata[];
    /* Layout: [blockDim.x] ham sums, [blockDim.x] mom sums, [blockDim.x] vol sums */
    double *s_ham = sdata;
    double *s_mom = &sdata[blockDim.x];
    double *s_vol = &s_mom[blockDim.x];

    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    int ltid = threadIdx.x;

    double my_ham = 0.0, my_mom = 0.0, my_vol = 0.0;

    if (tid < total_points) {
        int interior_per_block = N * N * N;
        int b = tid / interior_per_block;
        int pt = tid % interior_per_block;

        int i = ghost + pt % N;
        int j = ghost + (pt / N) % N;
        int k = ghost + pt / (N * N);

        /* Build per-field pointer array */
        const double *src_ptrs[NUM_FIELDS];
        for (int f = 0; f < nf; f++)
            src_ptrs[f] = data + (size_t)f * nb * npts + (size_t)b * npts;

        /* Minimal grid_t */
        grid_t g_local;
        memset(&g_local, 0, sizeof(grid_t));
        g_local.N = N;
        g_local.ghost = ghost;
        g_local.Ntotal = Nt;
        g_local.npoints = npts;
        g_local.n_fields = nf;
        g_local.dx = dx_arr[b];
        g_local.inv_dx = 1.0 / dx_arr[b];

        double dV = g_local.dx * g_local.dx * g_local.dx;

        double H = compute_hamiltonian_at(
            (const double *const *)src_ptrs, &g_local, i, j, k);
        my_ham = H * H * dV;

        double mom[3];
        compute_momentum_at(
            (const double *const *)src_ptrs, &g_local, i, j, k, mom);
        my_mom = (mom[0]*mom[0] + mom[1]*mom[1] + mom[2]*mom[2]) * dV;

        my_vol = dV;
    }

    s_ham[ltid] = my_ham;
    s_mom[ltid] = my_mom;
    s_vol[ltid] = my_vol;
    __syncthreads();

    /* Block-level reduction */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (ltid < stride) {
            s_ham[ltid] += s_ham[ltid + stride];
            s_mom[ltid] += s_mom[ltid + stride];
            s_vol[ltid] += s_vol[ltid + stride];
        }
        __syncthreads();
    }

    if (ltid == 0) {
        partial_ham[blockIdx.x] = s_ham[0];
        partial_mom[blockIdx.x] = s_mom[0];
        partial_vol[blockIdx.x] = s_vol[0];
    }
}

extern "C"
double backend_constraint_l2_packed(meshblock_pack_t *pack)
{
    if (!d_ptrs_valid) return 0.0;

    int nb = pack->n_blocks;
    int N = pack->N;
    int total_points = nb * N * N * N;

    int bs = 64;  /* Smaller block for high-register constraint kernel */
    int gs = (total_points + bs - 1) / bs;

    double *d_ham, *d_mom, *d_vol;
    HIP_CHECK(hipMalloc(&d_ham, gs * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_mom, gs * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vol, gs * sizeof(double)));

    size_t shared_bytes = bs * 3 * sizeof(double);
    hipLaunchKernelGGL(hip_constraint_l2_partial, gs, bs, shared_bytes,
                       gpu_stream,
                       d_ptrs.data, d_ptrs.dx_per_block,
                       nb, (size_t)pack->npts, pack->n_fields,
                       N, pack->ghost, pack->Ntotal,
                       d_ham, d_mom, d_vol, total_points);

    /* Copy partial results to host and finalize */
    double *h_ham = (double *)malloc(gs * sizeof(double));
    double *h_vol = (double *)malloc(gs * sizeof(double));
    HIP_CHECK(hipStreamSynchronize(gpu_stream));
    HIP_CHECK(hipMemcpy(h_ham, d_ham, gs * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_vol, d_vol, gs * sizeof(double), hipMemcpyDeviceToHost));

    double sum = 0.0;
    double vol = 0.0;
    for (int i = 0; i < gs; i++) {
        sum += h_ham[i];
        vol += h_vol[i];
    }

    free(h_ham);
    free(h_vol);
    (void)hipFree(d_ham);
    (void)hipFree(d_mom);
    (void)hipFree(d_vol);

    return (vol > 0.0) ? sqrt(sum / vol) : 0.0;
}

extern "C"
double backend_momentum_l2_packed(meshblock_pack_t *pack)
{
    if (!d_ptrs_valid) return 0.0;

    int nb = pack->n_blocks;
    int N = pack->N;
    int total_points = nb * N * N * N;

    int bs = 64;
    int gs = (total_points + bs - 1) / bs;

    double *d_ham, *d_mom, *d_vol;
    HIP_CHECK(hipMalloc(&d_ham, gs * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_mom, gs * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vol, gs * sizeof(double)));

    size_t shared_bytes = bs * 3 * sizeof(double);
    hipLaunchKernelGGL(hip_constraint_l2_partial, gs, bs, shared_bytes,
                       gpu_stream,
                       d_ptrs.data, d_ptrs.dx_per_block,
                       nb, (size_t)pack->npts, pack->n_fields,
                       N, pack->ghost, pack->Ntotal,
                       d_ham, d_mom, d_vol, total_points);

    /* Copy momentum partial results */
    double *h_mom = (double *)malloc(gs * sizeof(double));
    double *h_vol = (double *)malloc(gs * sizeof(double));
    HIP_CHECK(hipStreamSynchronize(gpu_stream));
    HIP_CHECK(hipMemcpy(h_mom, d_mom, gs * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_vol, d_vol, gs * sizeof(double), hipMemcpyDeviceToHost));

    double sum = 0.0;
    double vol = 0.0;
    for (int i = 0; i < gs; i++) {
        sum += h_mom[i];
        vol += h_vol[i];
    }

    free(h_mom);
    free(h_vol);
    (void)hipFree(d_ham);
    (void)hipFree(d_mom);
    (void)hipFree(d_vol);

    return (vol > 0.0) ? sqrt(sum / (3.0 * vol)) : 0.0;
}

/* ---- Kernel: Min lapse with position tracking ---- */

__global__ void hip_min_lapse_partial(
    double *data, double *dx_arr, double *origins,
    int nb, size_t npts, int nf, int N, int ghost, int Nt,
    double *partial_min, double *partial_pos, /* [3] per GPU block */
    int total_points)
{
    extern __shared__ double sdata[];
    /* Layout: [blockDim.x] val, [blockDim.x * 3] pos */
    double *s_val = sdata;
    double *s_pos = &sdata[blockDim.x];

    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    int ltid = threadIdx.x;

    double my_val = 1.0e30;
    double my_pos[3] = {0.0, 0.0, 0.0};

    if (tid < total_points) {
        int interior_per_block = N * N * N;
        int b = tid / interior_per_block;
        int pt = tid % interior_per_block;

        int i = ghost + pt % N;
        int j = ghost + (pt / N) % N;
        int k = ghost + pt / (N * N);

        int idx = k * Nt * Nt + j * Nt + i;
        size_t off = (size_t)FIELD_LAPSE * nb * npts + (size_t)b * npts + idx;
        my_val = data[off];

        double dx = dx_arr[b];
        my_pos[0] = origins[b * 3 + 0] + (i - ghost + 0.5) * dx;
        my_pos[1] = origins[b * 3 + 1] + (j - ghost + 0.5) * dx;
        my_pos[2] = origins[b * 3 + 2] + (k - ghost + 0.5) * dx;
    }

    s_val[ltid] = my_val;
    s_pos[ltid * 3 + 0] = my_pos[0];
    s_pos[ltid * 3 + 1] = my_pos[1];
    s_pos[ltid * 3 + 2] = my_pos[2];
    __syncthreads();

    /* Block-level min reduction with position tracking */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (ltid < stride) {
            if (s_val[ltid + stride] < s_val[ltid]) {
                s_val[ltid] = s_val[ltid + stride];
                s_pos[ltid * 3 + 0] = s_pos[(ltid + stride) * 3 + 0];
                s_pos[ltid * 3 + 1] = s_pos[(ltid + stride) * 3 + 1];
                s_pos[ltid * 3 + 2] = s_pos[(ltid + stride) * 3 + 2];
            }
        }
        __syncthreads();
    }

    if (ltid == 0) {
        partial_min[blockIdx.x] = s_val[0];
        partial_pos[blockIdx.x * 3 + 0] = s_pos[0];
        partial_pos[blockIdx.x * 3 + 1] = s_pos[1];
        partial_pos[blockIdx.x * 3 + 2] = s_pos[2];
    }
}

/* Host helper: launch min-lapse kernel and finalize on host */
static double hip_find_min_lapse(meshblock_pack_t *pack,
                                  double *out_x, double *out_y, double *out_z)
{
    int nb = pack->n_blocks;
    int N = pack->N;
    int total_points = nb * N * N * N;

    int bs = 256;
    int gs = (total_points + bs - 1) / bs;

    double *d_min, *d_pos;
    HIP_CHECK(hipMalloc(&d_min, gs * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_pos, gs * 3 * sizeof(double)));

    size_t shared_bytes = bs * (1 + 3) * sizeof(double);
    hipLaunchKernelGGL(hip_min_lapse_partial, gs, bs, shared_bytes,
                       gpu_stream,
                       d_ptrs.data, d_ptrs.dx_per_block, d_ptrs.origins,
                       nb, (size_t)pack->npts, pack->n_fields,
                       N, pack->ghost, pack->Ntotal,
                       d_min, d_pos, total_points);

    double *h_min = (double *)malloc(gs * sizeof(double));
    double *h_pos = (double *)malloc(gs * 3 * sizeof(double));
    HIP_CHECK(hipStreamSynchronize(gpu_stream));
    HIP_CHECK(hipMemcpy(h_min, d_min, gs * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_pos, d_pos, gs * 3 * sizeof(double), hipMemcpyDeviceToHost));

    double best = 1.0e30;
    *out_x = *out_y = *out_z = 0.0;
    for (int i = 0; i < gs; i++) {
        if (h_min[i] < best) {
            best = h_min[i];
            *out_x = h_pos[i * 3 + 0];
            *out_y = h_pos[i * 3 + 1];
            *out_z = h_pos[i * 3 + 2];
        }
    }

    free(h_min);
    free(h_pos);
    (void)hipFree(d_min);
    (void)hipFree(d_pos);

    return best;
}

extern "C"
double backend_min_lapse_packed(meshblock_pack_t *pack,
                                 double *out_x, double *out_y, double *out_z)
{
    if (!d_ptrs_valid) { *out_x = *out_y = *out_z = 0.0; return 1.0; }
    return hip_find_min_lapse(pack, out_x, out_y, out_z);
}

/* ---- Kernel: Min lapse with exclusion zone (for BH #2) ---- */

__global__ void hip_min_lapse_excl_partial(
    double *data, double *dx_arr, double *origins,
    int nb, size_t npts, int nf, int N, int ghost, int Nt,
    double ex, double ey, double ez, double excl_r2,
    double *partial_min, double *partial_pos,
    int total_points)
{
    extern __shared__ double sdata[];
    double *s_val = sdata;
    double *s_pos = &sdata[blockDim.x];

    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    int ltid = threadIdx.x;

    double my_val = 1.0e30;
    double my_pos[3] = {0.0, 0.0, 0.0};

    if (tid < total_points) {
        int interior_per_block = N * N * N;
        int b = tid / interior_per_block;
        int pt = tid % interior_per_block;

        int i = ghost + pt % N;
        int j = ghost + (pt / N) % N;
        int k = ghost + pt / (N * N);

        int idx = k * Nt * Nt + j * Nt + i;
        size_t off = (size_t)FIELD_LAPSE * nb * npts + (size_t)b * npts + idx;
        double a = data[off];

        double dx = dx_arr[b];
        double cx = origins[b * 3 + 0] + (i - ghost + 0.5) * dx;
        double cy = origins[b * 3 + 1] + (j - ghost + 0.5) * dx;
        double cz = origins[b * 3 + 2] + (k - ghost + 0.5) * dx;

        double dr2 = (cx - ex) * (cx - ex)
                    + (cy - ey) * (cy - ey)
                    + (cz - ez) * (cz - ez);

        if (dr2 > excl_r2) {
            my_val = a;
            my_pos[0] = cx; my_pos[1] = cy; my_pos[2] = cz;
        }
    }

    s_val[ltid] = my_val;
    s_pos[ltid * 3 + 0] = my_pos[0];
    s_pos[ltid * 3 + 1] = my_pos[1];
    s_pos[ltid * 3 + 2] = my_pos[2];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (ltid < stride) {
            if (s_val[ltid + stride] < s_val[ltid]) {
                s_val[ltid] = s_val[ltid + stride];
                s_pos[ltid * 3 + 0] = s_pos[(ltid + stride) * 3 + 0];
                s_pos[ltid * 3 + 1] = s_pos[(ltid + stride) * 3 + 1];
                s_pos[ltid * 3 + 2] = s_pos[(ltid + stride) * 3 + 2];
            }
        }
        __syncthreads();
    }

    if (ltid == 0) {
        partial_min[blockIdx.x] = s_val[0];
        partial_pos[blockIdx.x * 3 + 0] = s_pos[0];
        partial_pos[blockIdx.x * 3 + 1] = s_pos[1];
        partial_pos[blockIdx.x * 3 + 2] = s_pos[2];
    }
}

extern "C"
double backend_bh_separation_packed(meshblock_pack_t *pack, double excl_radius,
                                      double *x1, double *y1, double *z1,
                                      double *x2, double *y2, double *z2)
{
    if (!d_ptrs_valid) {
        *x1 = *y1 = *z1 = *x2 = *y2 = *z2 = 0.0;
        return 0.0;
    }

    /* Pass 1: global lapse minimum (BH #1) */
    double px1, py1, pz1;
    hip_find_min_lapse(pack, &px1, &py1, &pz1);
    *x1 = px1; *y1 = py1; *z1 = pz1;

    /* Pass 2: deepest minimum outside exclusion zone (BH #2) */
    int nb = pack->n_blocks;
    int N = pack->N;
    int total_points = nb * N * N * N;

    int bs = 256;
    int gs = (total_points + bs - 1) / bs;

    double *d_min, *d_pos;
    HIP_CHECK(hipMalloc(&d_min, gs * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_pos, gs * 3 * sizeof(double)));

    size_t shared_bytes = bs * (1 + 3) * sizeof(double);
    hipLaunchKernelGGL(hip_min_lapse_excl_partial, gs, bs, shared_bytes,
                       gpu_stream,
                       d_ptrs.data, d_ptrs.dx_per_block, d_ptrs.origins,
                       nb, (size_t)pack->npts, pack->n_fields,
                       N, pack->ghost, pack->Ntotal,
                       px1, py1, pz1, excl_radius * excl_radius,
                       d_min, d_pos, total_points);

    double *h_min = (double *)malloc(gs * sizeof(double));
    double *h_pos = (double *)malloc(gs * 3 * sizeof(double));
    HIP_CHECK(hipStreamSynchronize(gpu_stream));
    HIP_CHECK(hipMemcpy(h_min, d_min, gs * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_pos, d_pos, gs * 3 * sizeof(double), hipMemcpyDeviceToHost));

    double best = 1.0e30;
    *x2 = *y2 = *z2 = 0.0;
    for (int i = 0; i < gs; i++) {
        if (h_min[i] < best) {
            best = h_min[i];
            *x2 = h_pos[i * 3 + 0];
            *y2 = h_pos[i * 3 + 1];
            *z2 = h_pos[i * 3 + 2];
        }
    }

    free(h_min);
    free(h_pos);
    (void)hipFree(d_min);
    (void)hipFree(d_pos);

    if (best > 0.99) return 0.0;
    return sqrt((px1 - *x2) * (px1 - *x2) +
                (py1 - *y2) * (py1 - *y2) +
                (pz1 - *z2) * (pz1 - *z2));
}

/* ---- Kernel: NaN/Inf check ---- */

__global__ void hip_check_finite(double *data, size_t total_elements, int *any_bad)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    if (!isfinite(data[tid]))
        atomicOr(any_bad, 1);
}

extern "C"
int backend_check_finite_packed(meshblock_pack_t *pack)
{
    if (!d_ptrs_valid) return 1;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;

    int *d_bad;
    HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
    HIP_CHECK(hipMemsetAsync(d_bad, 0, sizeof(int), gpu_stream));

    int bs = 256;
    int gs = (int)((total + bs - 1) / bs);
    hipLaunchKernelGGL(hip_check_finite, gs, bs, 0, gpu_stream,
                       d_ptrs.data, total, d_bad);

    int h_bad = 0;
    HIP_CHECK(hipStreamSynchronize(gpu_stream));
    HIP_CHECK(hipMemcpy(&h_bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
    (void)hipFree(d_bad);

    return (h_bad == 0) ? 1 : 0;
}

/* ---- Kernel: Psi4 extraction at pre-mapped angular points ---- */

__global__ void hip_psi4_sphere(
    double *data, double *dx_arr, double *origins,
    int nb, size_t npts, int nf, int N, int ghost, int Nt,
    const int *point_block, const int *point_ijk, const double *point_pos,
    double center_x, double center_y, double center_z, double radius,
    double *out_re, double *out_im,
    int n_angular)
{
    int tid = (int)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n_angular) return;

    int b = point_block[tid];
    if (b < 0) {
        out_re[tid] = 0.0;
        out_im[tid] = 0.0;
        return;
    }

    int gi = point_ijk[tid * 3 + 0];
    int gj = point_ijk[tid * 3 + 1];
    int gk = point_ijk[tid * 3 + 2];

    double gpos[3] = { point_pos[tid * 3 + 0],
                        point_pos[tid * 3 + 1],
                        point_pos[tid * 3 + 2] };
    double center[3] = { center_x, center_y, center_z };

    /* Build per-field pointer array into pack data */
    const double *src_ptrs[NUM_FIELDS];
    for (int f = 0; f < nf; f++)
        src_ptrs[f] = data + (size_t)f * nb * npts + (size_t)b * npts;

    /* Minimal grid_t for FD stencils */
    grid_t g_local;
    memset(&g_local, 0, sizeof(grid_t));
    g_local.N = N;
    g_local.ghost = ghost;
    g_local.Ntotal = Nt;
    g_local.npoints = npts;
    g_local.n_fields = nf;
    g_local.dx = dx_arr[b];
    g_local.inv_dx = 1.0 / dx_arr[b];

    double psi4_val[2];
    psi4_at_point((const double *const *)src_ptrs, &g_local,
                  gi, gj, gk, center, psi4_val);

    out_re[tid] = radius * psi4_val[0];
    out_im[tid] = radius * psi4_val[1];
}

extern "C"
void backend_psi4_extract_packed(meshblock_pack_t *pack,
                                   psi4_workspace_t *ws,
                                   const struct mesh_s *m)
{
    if (!d_ptrs_valid) {
        /* Fallback to CPU */
        psi4_extract(ws, m);
        return;
    }

    int nb = pack->n_blocks;
    int n_angular = ws->n_theta * ws->n_phi;
    double r = ws->radius;
    double dphi = 2.0 * M_PI / ws->n_phi;

    /* --- Host: pre-compute angular-point-to-block mapping --- */

    /* Build reverse map: block_id → pack_index */
    int max_bid = 0;
    for (int b = 0; b < nb; b++)
        if (pack->block_ids[b] > max_bid)
            max_bid = pack->block_ids[b];
    int *bid_to_pack = (int *)calloc(max_bid + 1, sizeof(int));
    for (int i = 0; i <= max_bid; i++) bid_to_pack[i] = -1;
    for (int b = 0; b < nb; b++)
        bid_to_pack[pack->block_ids[b]] = b;

    int *h_block = (int *)malloc(n_angular * sizeof(int));
    int *h_ijk   = (int *)malloc(n_angular * 3 * sizeof(int));
    double *h_pos = (double *)malloc(n_angular * 3 * sizeof(double));

    for (int ith = 0; ith < ws->n_theta; ith++) {
        double th = ws->theta[ith];
        double st = sin(th), ct = cos(th);

        for (int iph = 0; iph < ws->n_phi; iph++) {
            int aidx = ith * ws->n_phi + iph;
            double ph = dphi * iph;
            double sp = sin(ph), cp = cos(ph);

            double x = ws->center[0] + r * st * cp;
            double y = ws->center[1] + r * st * sp;
            double z = ws->center[2] + r * ct;

            block_t *blk = mesh_find_block_at(m, x, y, z);
            if (!blk || blk->id > max_bid || bid_to_pack[blk->id] < 0) {
                h_block[aidx] = -1;
                continue;
            }

            h_block[aidx] = bid_to_pack[blk->id];

            grid_t *g = blk->grid;
            int ghost = g->ghost;
            double cix = (x - blk->origin[0]) / g->dx - 0.5 + ghost;
            double ciy = (y - blk->origin[1]) / g->dx - 0.5 + ghost;
            double ciz = (z - blk->origin[2]) / g->dx - 0.5 + ghost;

            int gi = (int)round(cix);
            int gj = (int)round(ciy);
            int gk = (int)round(ciz);

            int lo = ghost, hi_bound = ghost + g->N - 1;
            if (gi < lo) gi = lo; if (gi > hi_bound) gi = hi_bound;
            if (gj < lo) gj = lo; if (gj > hi_bound) gj = hi_bound;
            if (gk < lo) gk = lo; if (gk > hi_bound) gk = hi_bound;

            h_ijk[aidx * 3 + 0] = gi;
            h_ijk[aidx * 3 + 1] = gj;
            h_ijk[aidx * 3 + 2] = gk;

            h_pos[aidx * 3 + 0] = blk->origin[0] + (gi - ghost + 0.5) * g->dx;
            h_pos[aidx * 3 + 1] = blk->origin[1] + (gj - ghost + 0.5) * g->dx;
            h_pos[aidx * 3 + 2] = blk->origin[2] + (gk - ghost + 0.5) * g->dx;
        }
    }

    free(bid_to_pack);

    /* --- Upload mapping to device --- */
    int *d_block, *d_ijk;
    double *d_pos, *d_re, *d_im;
    HIP_CHECK(hipMalloc(&d_block, n_angular * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_ijk,   n_angular * 3 * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_pos,   n_angular * 3 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_re,    n_angular * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_im,    n_angular * sizeof(double)));

    HIP_CHECK(hipMemcpyAsync(d_block, h_block, n_angular * sizeof(int),
                              hipMemcpyHostToDevice, gpu_stream));
    HIP_CHECK(hipMemcpyAsync(d_ijk, h_ijk, n_angular * 3 * sizeof(int),
                              hipMemcpyHostToDevice, gpu_stream));
    HIP_CHECK(hipMemcpyAsync(d_pos, h_pos, n_angular * 3 * sizeof(double),
                              hipMemcpyHostToDevice, gpu_stream));

    /* --- Launch kernel --- */
    int bs = 64;  /* small block for high-register Psi4 kernel */
    int gs = (n_angular + bs - 1) / bs;
    hipLaunchKernelGGL(hip_psi4_sphere, gs, bs, 0, gpu_stream,
                       d_ptrs.data, d_ptrs.dx_per_block, d_ptrs.origins,
                       nb, (size_t)pack->npts, pack->n_fields,
                       pack->N, pack->ghost, pack->Ntotal,
                       d_block, d_ijk, d_pos,
                       ws->center[0], ws->center[1], ws->center[2], r,
                       d_re, d_im, n_angular);

    /* --- Download results --- */
    HIP_CHECK(hipStreamSynchronize(gpu_stream));
    HIP_CHECK(hipMemcpy(ws->re_psi4, d_re, n_angular * sizeof(double),
                         hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ws->im_psi4, d_im, n_angular * sizeof(double),
                         hipMemcpyDeviceToHost));

    /* --- Cleanup device buffers --- */
    (void)hipFree(d_block);
    (void)hipFree(d_ijk);
    (void)hipFree(d_pos);
    (void)hipFree(d_re);
    (void)hipFree(d_im);

    free(h_block);
    free(h_ijk);
    free(h_pos);

    /* --- Mode decomposition on host --- */
    psi4_decompose_modes(ws);
}
