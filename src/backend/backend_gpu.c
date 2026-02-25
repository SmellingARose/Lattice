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
#include "../evolution/maxwell_rhs.h"
#include "../boundary/sommerfeld.h"
#include "../core/fields.h"
#include "../geometry/tensor_utils.h"
#include "../amr/block.h"
#include "../amr/restriction.h"
#include "../amr/prolongation.h"
#include <string.h>
#include <math.h>

/* ========================================================================
 * Legacy per-grid API
 * ======================================================================== */

void backend_compute_rhs(double ** restrict rhs,
                         const double *const * restrict src,
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
    #pragma omp target enter data map(to: pack->refined_map[0:nb])
    #pragma omp target enter data map(to: pack->nblevel_table[0:nb*27])

    /* Coarse_buf data (if present) */
    if (pack->coarse_data && pack->n_refined > 0) {
        size_t coarse_total = (size_t)pack->n_refined * pack->n_fields
                            * pack->coarse_npts;
        #pragma omp target enter data map(to: pack->coarse_data[0:coarse_total])
        #pragma omp target enter data map(to: pack->coarse_neighbor_table[0:pack->n_refined*NUM_NEIGHBORS])
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
    #pragma omp target exit data map(release: pack->refined_map[0:nb])
    #pragma omp target exit data map(release: pack->nblevel_table[0:nb*27])

    /* Coarse_buf data */
    if (pack->coarse_data && pack->n_refined > 0) {
        size_t coarse_total = (size_t)pack->n_refined * pack->n_fields
                            * pack->coarse_npts;
        #pragma omp target exit data map(from: pack->coarse_data[0:coarse_total])
        #pragma omp target exit data map(release: pack->coarse_neighbor_table[0:pack->n_refined*NUM_NEIGHBORS])
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
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    double *data = pack->data;
    double *rhs_data = pack->rhs;
    double *dx_arr = pack->dx_per_block;
    int nf = pack->n_fields;

    /* Build grid_t template once before kernel — eliminates 1064-byte
     * memset + 5 assignments per GPU thread. Only dx varies per block. */
    grid_t g_template;
    memset(&g_template, 0, sizeof(grid_t));
    g_template.N        = pack->N;
    g_template.ghost    = pack->ghost;
    g_template.Ntotal   = pack->Ntotal;
    g_template.npoints  = npts;
    g_template.n_fields = nf;

    #pragma omp target teams distribute parallel for collapse(4)
    for (int b = 0; b < nb; b++) {
        for (int k = lo; k < hi; k++) {
            for (int j = lo; j < hi; j++) {
                for (int i = lo; i < hi; i++) {
                    /* Per-field pointers from pack layout (on GPU stack) */
                    double *rhs_ptrs[NUM_FIELDS];
                    const double *src_ptrs[NUM_FIELDS];
                    for (int f = 0; f < nf; f++) {
                        size_t base = (size_t)f * nb * npts + (size_t)b * npts;
                        src_ptrs[f] = data + base;
                        rhs_ptrs[f] = rhs_data + base;
                    }

                    /* Copy pre-built template, set only per-block dx */
                    grid_t g_local = g_template;
                    g_local.dx = dx_arr[b];

                    if (p->em_enabled)
                        ccz4_maxwell_rhs_point(rhs_ptrs,
                                               (const double *const *)src_ptrs,
                                               &g_local, p, i, j, k);
                    else
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
    int nf = pack->n_fields;

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

                    /* Apply Sommerfeld to each field.
                     * Skip EM fields when disabled (saves 6/31 iterations). */
                    for (int field = 0; field < nf; field++) {
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

/* ========================================================================
 * Packed ghost exchange — GPU implementation.
 *
 * Syncs pack data to host, runs all 5 phases on host memory, syncs back.
 * This eliminates the Commit 1 unpack/repack overhead while keeping the
 * 5-phase logic on host (where complex control flow is natural).
 *
 * True device-side kernels for each phase would avoid the sync, but
 * the host-side exchange is correct and ~3x faster than Commit 1 fallback
 * (no unpack→exchange→repack, just sync→exchange→sync).
 *
 * Same algorithm as backend_cpu.c — see that file for phase documentation.
 * ======================================================================== */

/* Ghost range helper (same as CPU backend) */
static inline void ghost_range_pack(int offset, int ghost, int N, int Nt,
                                     int *dst_lo, int *dst_hi,
                                     int *src_lo, int *src_hi)
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

/* Phase 0+1: Same-level exchange on pack->data */
static void packed_exchange_same_level(meshblock_pack_t *pack)
{
    int nb = pack->n_blocks;
    int ghost = pack->ghost;
    int N = pack->N;
    int Nt = pack->Ntotal;
    size_t npts = pack->npts;

    for (int b = 0; b < nb; b++) {
        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int nbr = pack->neighbor_table[b * NUM_NEIGHBORS + n];
            if (nbr < 0) continue;
            if (pack->levels[b] != pack->levels[nbr]) continue;

            int ox = nbr_offset[n][0];
            int oy = nbr_offset[n][1];
            int oz = nbr_offset[n][2];

            int dx_lo, dx_hi, sx_lo, sx_hi;
            int dy_lo, dy_hi, sy_lo, sy_hi;
            int dz_lo, dz_hi, sz_lo, sz_hi;
            ghost_range_pack(ox, ghost, N, Nt, &dx_lo, &dx_hi, &sx_lo, &sx_hi);
            ghost_range_pack(oy, ghost, N, Nt, &dy_lo, &dy_hi, &sy_lo, &sy_hi);
            ghost_range_pack(oz, ghost, N, Nt, &dz_lo, &dz_hi, &sz_lo, &sz_hi);

            int nx = dx_hi - dx_lo;

            for (int f = 0; f < pack->n_fields; f++) {
                size_t dst_base = (size_t)f * nb * npts + (size_t)b * npts;
                size_t src_base = (size_t)f * nb * npts + (size_t)nbr * npts;

                for (int k = 0; k < (dz_hi - dz_lo); k++) {
                    for (int j = 0; j < (dy_hi - dy_lo); j++) {
                        size_t d = dst_base
                            + (size_t)(dz_lo + k) * Nt * Nt
                            + (size_t)(dy_lo + j) * Nt + dx_lo;
                        size_t s = src_base
                            + (size_t)(sz_lo + k) * Nt * Nt
                            + (size_t)(sy_lo + j) * Nt + sx_lo;
                        memcpy(&pack->data[d], &pack->data[s],
                               nx * sizeof(double));
                    }
                }
            }
        }
    }
}

/* Phase 2: Restrict fine → coarse_data (6th-order, no fallback) */
static void packed_restrict_to_coarse(meshblock_pack_t *pack)
{
    if (pack->n_refined == 0) return;

    int nb = pack->n_blocks;
    int ghost_f = pack->ghost;
    int Nt_f = pack->Ntotal;
    int ghost_c = pack->ghost;
    int N_c = pack->coarse_N;
    int Nt_c = pack->coarse_Ntotal;
    size_t npts = pack->npts;
    size_t cnpts = pack->coarse_npts;

    for (int b = 0; b < nb; b++) {
        int r = pack->refined_map[b];
        if (r < 0) continue;

        for (int f = 0; f < pack->n_fields; f++) {
            const double *src = pack->data
                + (size_t)f * nb * npts + (size_t)b * npts;
            double *dst = pack->coarse_data
                + (size_t)r * pack->n_fields * cnpts + (size_t)f * cnpts;

            for (int ck = ghost_c; ck < ghost_c + N_c; ck++) {
                int fk_base = 2 * (ck - ghost_c) + ghost_f;
                for (int cj = ghost_c; cj < ghost_c + N_c; cj++) {
                    int fj_base = 2 * (cj - ghost_c) + ghost_f;
                    for (int ci = ghost_c; ci < ghost_c + N_c; ci++) {
                        int fi_base = 2 * (ci - ghost_c) + ghost_f;

                        double val = 0.0;
                        for (int sk = 0; sk < RESTRICT_STENCIL; sk++) {
                            int fk = fk_base - 2 + sk;
                            for (int sj = 0; sj < RESTRICT_STENCIL; sj++) {
                                double wkj = restrict_wkj[sk][sj];
                                int fj = fj_base - 2 + sj;
                                for (int si = 0; si < RESTRICT_STENCIL; si++) {
                                    int fi = fi_base - 2 + si;
                                    val += wkj * restrict_w[si]
                                        * src[fi + fj*Nt_f + fk*Nt_f*Nt_f];
                                }
                            }
                        }
                        dst[ci + cj*Nt_c + ck*Nt_c*Nt_c] = val;
                    }
                }
            }
        }
    }
}

/* Phase 3: Fill coarse_buf ghosts */
static void packed_fill_coarse_buf_ghosts(meshblock_pack_t *pack)
{
    if (pack->n_refined == 0) return;

    int nb = pack->n_blocks;
    int ghost = pack->ghost;
    int N_c = pack->coarse_N;
    int Nt_c = pack->coarse_Ntotal;
    int Nt_f = pack->Ntotal;
    size_t npts = pack->npts;
    size_t cnpts = pack->coarse_npts;

    for (int b = 0; b < nb; b++) {
        int r = pack->refined_map[b];
        if (r < 0) continue;

        int blk_level = pack->levels[b];

        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int ox = nbr_offset[n][0];
            int oy = nbr_offset[n][1];
            int oz = nbr_offset[n][2];
            int nlev = pack->nblevel_table[b*27 + (oz+1)*9 + (oy+1)*3 + (ox+1)];

            if (nlev == blk_level) {
                int coarse_nbr = pack->coarse_neighbor_table[r * NUM_NEIGHBORS + n];
                if (coarse_nbr < 0) continue;

                int dx_lo, dx_hi, sx_lo, sx_hi;
                int dy_lo, dy_hi, sy_lo, sy_hi;
                int dz_lo, dz_hi, sz_lo, sz_hi;
                ghost_range_pack(ox, ghost, N_c, Nt_c,
                                 &dx_lo, &dx_hi, &sx_lo, &sx_hi);
                ghost_range_pack(oy, ghost, N_c, Nt_c,
                                 &dy_lo, &dy_hi, &sy_lo, &sy_hi);
                ghost_range_pack(oz, ghost, N_c, Nt_c,
                                 &dz_lo, &dz_hi, &sz_lo, &sz_hi);

                int nx = dx_hi - dx_lo;

                for (int f = 0; f < pack->n_fields; f++) {
                    size_t dst_off = (size_t)r * pack->n_fields * cnpts
                                   + (size_t)f * cnpts;
                    size_t src_off = (size_t)coarse_nbr * pack->n_fields * cnpts
                                   + (size_t)f * cnpts;

                    for (int k = 0; k < (dz_hi - dz_lo); k++) {
                        for (int j = 0; j < (dy_hi - dy_lo); j++) {
                            size_t dd = dst_off
                                + (size_t)(dz_lo+k) * Nt_c * Nt_c
                                + (size_t)(dy_lo+j) * Nt_c + dx_lo;
                            size_t ss = src_off
                                + (size_t)(sz_lo+k) * Nt_c * Nt_c
                                + (size_t)(sy_lo+j) * Nt_c + sx_lo;
                            memcpy(&pack->coarse_data[dd],
                                   &pack->coarse_data[ss],
                                   nx * sizeof(double));
                        }
                    }
                }

            } else if (nlev >= 0 && nlev == blk_level - 1) {
                int pack_nbr = pack->neighbor_table[b * NUM_NEIGHBORS + n];
                if (pack_nbr < 0) continue;

                double dx_c = pack->dx_per_block[pack_nbr];
                int off_i = (int)round(
                    (pack->origins[b*3+0] - pack->origins[pack_nbr*3+0]) / dx_c);
                int off_j = (int)round(
                    (pack->origins[b*3+1] - pack->origins[pack_nbr*3+1]) / dx_c);
                int off_k = (int)round(
                    (pack->origins[b*3+2] - pack->origins[pack_nbr*3+2]) / dx_c);

                int dx_lo, dx_hi, dummy1, dummy2;
                int dy_lo, dy_hi, dummy3, dummy4;
                int dz_lo, dz_hi, dummy5, dummy6;
                ghost_range_pack(ox, ghost, N_c, Nt_c,
                                 &dx_lo, &dx_hi, &dummy1, &dummy2);
                ghost_range_pack(oy, ghost, N_c, Nt_c,
                                 &dy_lo, &dy_hi, &dummy3, &dummy4);
                ghost_range_pack(oz, ghost, N_c, Nt_c,
                                 &dz_lo, &dz_hi, &dummy5, &dummy6);

                for (int f = 0; f < pack->n_fields; f++) {
                    size_t dst_off = (size_t)r * pack->n_fields * cnpts
                                   + (size_t)f * cnpts;
                    size_t src_off = (size_t)f * nb * npts
                                   + (size_t)pack_nbr * npts;

                    for (int k = dz_lo; k < dz_hi; k++) {
                        int sk = k + off_k;
                        if (sk < 0 || sk >= Nt_f) continue;
                        for (int j = dy_lo; j < dy_hi; j++) {
                            int sj = j + off_j;
                            if (sj < 0 || sj >= Nt_f) continue;
                            for (int i = dx_lo; i < dx_hi; i++) {
                                int si = i + off_i;
                                if (si < 0 || si >= Nt_f) continue;
                                pack->coarse_data[dst_off
                                    + k*Nt_c*Nt_c + j*Nt_c + i] =
                                    pack->data[src_off
                                    + sk*Nt_f*Nt_f + sj*Nt_f + si];
                            }
                        }
                    }
                }
            }
        }
    }
}

/* Phase 3.5: Boundary extrapolation on coarse_data */
static void packed_fill_coarse_boundary(meshblock_pack_t *pack)
{
    if (pack->n_refined == 0) return;

    int nb = pack->n_blocks;
    int gh = pack->ghost;
    int N_c = pack->coarse_N;
    int Nt_c = pack->coarse_Ntotal;
    size_t cnpts = pack->coarse_npts;

    double c[4][3];
    for (int d = 0; d < gh && d < 4; d++) {
        double t = -(d + 1);
        c[d][0] = (t - 1.0) * (t - 2.0) / 2.0;
        c[d][1] = -t * (t - 2.0);
        c[d][2] = t * (t - 1.0) / 2.0;
    }

    for (int b = 0; b < nb; b++) {
        int r = pack->refined_map[b];
        if (r < 0) continue;

        int xm = pack->nblevel_table[b*27 + 1*9 + 1*3 + 0] < 0;
        int xp = pack->nblevel_table[b*27 + 1*9 + 1*3 + 2] < 0;
        int ym = pack->nblevel_table[b*27 + 1*9 + 0*3 + 1] < 0;
        int yp = pack->nblevel_table[b*27 + 1*9 + 2*3 + 1] < 0;
        int zm = pack->nblevel_table[b*27 + 0*9 + 1*3 + 1] < 0;
        int zp = pack->nblevel_table[b*27 + 2*9 + 1*3 + 1] < 0;

        for (int f = 0; f < pack->n_fields; f++) {
            double *data = pack->coarse_data
                + (size_t)r * pack->n_fields * cnpts + (size_t)f * cnpts;

            if (xm) {
                for (int k = 0; k < Nt_c; k++)
                    for (int j = 0; j < Nt_c; j++)
                        for (int d = 0; d < gh; d++) {
                            int gi = gh - 1 - d;
                            data[gi + j*Nt_c + k*Nt_c*Nt_c] =
                                c[d][0]*data[gh     + j*Nt_c + k*Nt_c*Nt_c] +
                                c[d][1]*data[(gh+1) + j*Nt_c + k*Nt_c*Nt_c] +
                                c[d][2]*data[(gh+2) + j*Nt_c + k*Nt_c*Nt_c];
                        }
            }
            if (xp) {
                for (int k = 0; k < Nt_c; k++)
                    for (int j = 0; j < Nt_c; j++)
                        for (int d = 0; d < gh; d++) {
                            int gi = gh + N_c + d;
                            data[gi + j*Nt_c + k*Nt_c*Nt_c] =
                                c[d][0]*data[(gh+N_c-1) + j*Nt_c + k*Nt_c*Nt_c] +
                                c[d][1]*data[(gh+N_c-2) + j*Nt_c + k*Nt_c*Nt_c] +
                                c[d][2]*data[(gh+N_c-3) + j*Nt_c + k*Nt_c*Nt_c];
                        }
            }
            if (ym) {
                for (int k = 0; k < Nt_c; k++)
                    for (int i = 0; i < Nt_c; i++)
                        for (int d = 0; d < gh; d++) {
                            int gj = gh - 1 - d;
                            data[i + gj*Nt_c + k*Nt_c*Nt_c] =
                                c[d][0]*data[i + gh*Nt_c     + k*Nt_c*Nt_c] +
                                c[d][1]*data[i + (gh+1)*Nt_c + k*Nt_c*Nt_c] +
                                c[d][2]*data[i + (gh+2)*Nt_c + k*Nt_c*Nt_c];
                        }
            }
            if (yp) {
                for (int k = 0; k < Nt_c; k++)
                    for (int i = 0; i < Nt_c; i++)
                        for (int d = 0; d < gh; d++) {
                            int gj = gh + N_c + d;
                            data[i + gj*Nt_c + k*Nt_c*Nt_c] =
                                c[d][0]*data[i + (gh+N_c-1)*Nt_c + k*Nt_c*Nt_c] +
                                c[d][1]*data[i + (gh+N_c-2)*Nt_c + k*Nt_c*Nt_c] +
                                c[d][2]*data[i + (gh+N_c-3)*Nt_c + k*Nt_c*Nt_c];
                        }
            }
            if (zm) {
                for (int j = 0; j < Nt_c; j++)
                    for (int i = 0; i < Nt_c; i++)
                        for (int d = 0; d < gh; d++) {
                            int gk = gh - 1 - d;
                            data[i + j*Nt_c + gk*Nt_c*Nt_c] =
                                c[d][0]*data[i + j*Nt_c + gh*Nt_c*Nt_c] +
                                c[d][1]*data[i + j*Nt_c + (gh+1)*Nt_c*Nt_c] +
                                c[d][2]*data[i + j*Nt_c + (gh+2)*Nt_c*Nt_c];
                        }
            }
            if (zp) {
                for (int j = 0; j < Nt_c; j++)
                    for (int i = 0; i < Nt_c; i++)
                        for (int d = 0; d < gh; d++) {
                            int gk = gh + N_c + d;
                            data[i + j*Nt_c + gk*Nt_c*Nt_c] =
                                c[d][0]*data[i + j*Nt_c + (gh+N_c-1)*Nt_c*Nt_c] +
                                c[d][1]*data[i + j*Nt_c + (gh+N_c-2)*Nt_c*Nt_c] +
                                c[d][2]*data[i + j*Nt_c + (gh+N_c-3)*Nt_c*Nt_c];
                        }
            }
        }
    }
}

/* Phase 4: Prolongate coarse_data → fine ghosts */
static void packed_prolongate_fine_ghosts(meshblock_pack_t *pack)
{
    if (pack->n_refined == 0) return;

    int nb = pack->n_blocks;
    int ghost_f = pack->ghost;
    int N_f = pack->N;
    int Nt_f = pack->Ntotal;
    int ghost_c = pack->ghost;
    int Nt_c = pack->coarse_Ntotal;
    size_t npts = pack->npts;
    size_t cnpts = pack->coarse_npts;
    int half = PROLONG_STENCIL / 2;

    for (int b = 0; b < nb; b++) {
        int r = pack->refined_map[b];
        if (r < 0) continue;

        int blk_level = pack->levels[b];

        for (int f = 0; f < pack->n_fields; f++) {
            const double *csrc = pack->coarse_data
                + (size_t)r * pack->n_fields * cnpts + (size_t)f * cnpts;
            double *fdata = pack->data
                + (size_t)f * nb * npts + (size_t)b * npts;

            for (int fk = 0; fk < Nt_f; fk++) {
                for (int fj = 0; fj < Nt_f; fj++) {
                    for (int fi = 0; fi < Nt_f; fi++) {
                        if (fi >= ghost_f && fi < ghost_f + N_f &&
                            fj >= ghost_f && fj < ghost_f + N_f &&
                            fk >= ghost_f && fk < ghost_f + N_f)
                            continue;

                        int ox = (fi < ghost_f) ? -1 :
                                 (fi >= ghost_f + N_f) ? 1 : 0;
                        int oy = (fj < ghost_f) ? -1 :
                                 (fj >= ghost_f + N_f) ? 1 : 0;
                        int oz = (fk < ghost_f) ? -1 :
                                 (fk >= ghost_f + N_f) ? 1 : 0;

                        int nlev = pack->nblevel_table[
                            b*27 + (oz+1)*9 + (oy+1)*3 + (ox+1)];
                        if (nlev == blk_level) continue;

                        double ci_cont = (fi - ghost_f + 0.5) / 2.0
                                         + ghost_c - 0.5;
                        double cj_cont = (fj - ghost_f + 0.5) / 2.0
                                         + ghost_c - 0.5;
                        double ck_cont = (fk - ghost_f + 0.5) / 2.0
                                         + ghost_c - 0.5;

                        int ci0 = (int)(ci_cont + 0.5);
                        int cj0 = (int)(cj_cont + 0.5);
                        int ck0 = (int)(ck_cont + 0.5);

                        if (ci0 < half || ci0 >= Nt_c - half ||
                            cj0 < half || cj0 >= Nt_c - half ||
                            ck0 < half || ck0 >= Nt_c - half)
                            continue;

                        int oi = (ci_cont >= ci0) ? 1 : 0;
                        int oj = (cj_cont >= cj0) ? 1 : 0;
                        int ok = (ck_cont >= ck0) ? 1 : 0;

                        int combo = ok * 2 + oj;
                        double val = 0.0;
                        for (int sk = 0; sk < PROLONG_STENCIL; sk++) {
                            for (int sj = 0; sj < PROLONG_STENCIL; sj++) {
                                double wkj = prolong_wkj[combo][sk][sj];
                                for (int si = 0; si < PROLONG_STENCIL; si++) {
                                    int wi = oi ?
                                        (PROLONG_STENCIL-1-si) : si;
                                    int idx = (ci0-half+si)
                                        + (cj0-half+sj) * Nt_c
                                        + (ck0-half+sk) * Nt_c * Nt_c;
                                    val += wkj * prolong_w[wi] * csrc[idx];
                                }
                            }
                        }

                        fdata[fi + fj*Nt_f + fk*Nt_f*Nt_f] = val;
                    }
                }
            }
        }
    }
}

/*
 * GPU ghost exchange: sync to host, run 5-phase exchange, sync back.
 * Eliminates Commit 1 unpack/repack overhead by operating directly
 * on pack buffers. True device-side kernels are a future optimization.
 */
void backend_ghost_exchange_packed(meshblock_pack_t *pack)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;

    /* Sync field data from device to host */
    #pragma omp target update from(pack->data[0:total])
    if (pack->coarse_data && pack->n_refined > 0) {
        size_t ct = (size_t)pack->n_refined * pack->n_fields * pack->coarse_npts;
        #pragma omp target update from(pack->coarse_data[0:ct])
    }

    /* Phase 0+1: Same-level exchange */
    packed_exchange_same_level(pack);

    if (pack->n_refined > 0) {
        /* Phase 2: Restrict fine → coarse_buf */
        packed_restrict_to_coarse(pack);
        /* Phase 3: Fill coarse_buf ghosts */
        packed_fill_coarse_buf_ghosts(pack);
        /* Phase 3.5: Boundary extrapolation */
        packed_fill_coarse_boundary(pack);
        /* Phase 4: Prolongate → fine ghosts */
        packed_prolongate_fine_ghosts(pack);
    }

    /* Sync updated data back to device */
    #pragma omp target update to(pack->data[0:total])
    if (pack->coarse_data && pack->n_refined > 0) {
        size_t ct = (size_t)pack->n_refined * pack->n_fields * pack->coarse_npts;
        #pragma omp target update to(pack->coarse_data[0:ct])
    }
}

/*
 * Enforce algebraic constraints on GPU: det(h)=1, tr(A)=0, chi>0, lapse>0.
 * Single kernel launch, collapse(4) over (block, k, j, i).
 * Stack per thread: ~288 bytes (h_loc, h_UU, A_loc, det, scale).
 *
 * All called functions (compute_det_sym, fast_inv_cbrt, compute_inverse_sym,
 * make_trace_free) are declared with #pragma omp declare target in
 * tensor_utils.h.
 *
 * Ref: GRChombo CCZ4/TraceARemoval.hpp, CCZ4/PositiveChiAndAlpha.hpp
 */
void backend_enforce_algebraic_packed(meshblock_pack_t *pack)
{
    int Nt = pack->Ntotal;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    double *data = pack->data;

    #pragma omp target teams distribute parallel for collapse(4)
    for (int b = 0; b < nb; b++) {
        for (int k = 0; k < Nt; k++) {
            for (int j = 0; j < Nt; j++) {
                for (int i = 0; i < Nt; i++) {
                    int idx = k * Nt * Nt + j * Nt + i;

                    /* Per-field pointers for this block */
                    #define FP(fld) (data + (size_t)(fld) * nb * npts \
                                      + (size_t)b * npts)

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
                    FOR2(a, bb) h_loc[a][bb] *= scale;

                    FP(FIELD_H11)[idx] = h_loc[0][0];
                    FP(FIELD_H12)[idx] = h_loc[0][1];
                    FP(FIELD_H13)[idx] = h_loc[0][2];
                    FP(FIELD_H22)[idx] = h_loc[1][1];
                    FP(FIELD_H23)[idx] = h_loc[1][2];
                    FP(FIELD_H33)[idx] = h_loc[2][2];

                    /* Enforce tr(A) = 0 */
                    double h_UU[3][3];
                    compute_inverse_sym(h_loc, h_UU);

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
            }
        }
    }
}
