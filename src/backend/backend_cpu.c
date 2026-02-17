/*
 * Lattice — 3D Numerical Relativity
 * CPU backend: OpenMP parallel loops for both per-grid and packed kernels.
 *
 * Per-grid API: z (outer, parallelized) -> y -> x (inner, unit stride)
 *
 * Packed API: all leaf blocks batched into meshblock_pack_t.
 * Outer loop over blocks, OpenMP parallel for over (k,j) per block.
 * GPU-analogous structure allows validation on macOS before GPU hardware.
 *
 * Map/unmap operations are no-ops on CPU since data is already in host memory.
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
    int lo = g->ghost;
    int hi = g->ghost + g->N;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = lo; k < hi; k++) {
        for (int j = lo; j < hi; j++) {
            for (int i = lo; i < hi; i++) {
                func(rhs, src, g, p, i, j, k);
            }
        }
    }
}

void backend_init(void) { /* no-op for CPU */ }
void backend_cleanup(void) { /* no-op for CPU */ }

/* ========================================================================
 * Packed batch kernel API — CPU implementations
 *
 * Each function mirrors the GPU version but uses OpenMP parallel for
 * instead of omp target teams distribute. This allows full testing
 * on macOS before GPU hardware is available.
 * ======================================================================== */

/* ---- Data management (no-ops on CPU) ---- */

void backend_map_pack(meshblock_pack_t *pack, const sim_params_t *p)
{
    (void)pack; (void)p;
    /* CPU: data already in host memory, nothing to map */
}

void backend_unmap_pack(meshblock_pack_t *pack)
{
    (void)pack;
    /* CPU: data already in host memory, nothing to unmap */
}

/* ---- Helper: select pack buffer by index ---- */

/*
 * Returns pointer to the requested pack buffer.
 *   PACK_BUF_DATA (0)    → pack->data
 *   PACK_BUF_RHS (1)     → pack->rhs
 *   PACK_BUF_SCRATCH (2) → pack->scratch
 *   PACK_BUF_ACCUM (3)   → pack->accum
 */
static double *pack_buffer(meshblock_pack_t *pack, int which)
{
    switch (which) {
        case PACK_BUF_DATA:    return pack->data;
        case PACK_BUF_RHS:     return pack->rhs;
        case PACK_BUF_SCRATCH: return pack->scratch;
        case PACK_BUF_ACCUM:   return pack->accum;
        default: return NULL;
    }
}

/* ---- Zero buffer ---- */

/*
 * Zero the selected pack buffer (all fields, all blocks, all points).
 * Used to initialize dU (scratch) for CK45 and accum for classic RK4.
 */
void backend_zero_packed(meshblock_pack_t *pack, int which)
{
    double *buf = pack_buffer(pack, which);
    if (!buf) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    memset(buf, 0, total * sizeof(double));
}

/* ---- Batched RHS ---- */

/*
 * Compute CCZ4 RHS for all interior cells of all blocks.
 *
 * For each block b in the pack:
 *   1. Build per-field pointer arrays (src_ptrs, rhs_ptrs) that point
 *      into the pack's contiguous buffer at the correct offset for block b
 *   2. Construct a stack-local grid_t with the block's dimensions and dx
 *   3. Call ccz4_rhs_point for each interior cell (i,j,k)
 *
 * The per-field pointers satisfy:
 *   src_ptrs[f] = pack->data + f*n_blocks*npts + b*npts
 * so src_ptrs[f][IDX(g, i, j, k)] accesses the correct packed location.
 *
 * CPU: outer loop over blocks, OpenMP collapse(2) over (k,j) per block.
 * This matches the GPU's collapse(4) over (b,k,j,i) in structure.
 */
void backend_compute_rhs_packed(meshblock_pack_t *pack, const sim_params_t *p)
{
    int lo = pack->ghost;
    int hi = pack->ghost + pack->N;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;

    for (int b = 0; b < nb; b++) {
        /* Build per-field pointer arrays into pack layout for this block.
         * These point into the contiguous pack buffers at the correct
         * offset, so ccz4_rhs_point can use IDX(g, i, j, k) normally. */
        double *rhs_ptrs[NUM_FIELDS];
        const double *src_ptrs[NUM_FIELDS];
        for (int f = 0; f < NUM_FIELDS; f++) {
            size_t base = (size_t)f * nb * npts + (size_t)b * npts;
            src_ptrs[f] = pack->data + base;
            rhs_ptrs[f] = pack->rhs  + base;
        }

        /* Stack-local grid_t with this block's dimensions.
         * ccz4_rhs_point only reads: N, ghost, Ntotal, dx, npoints
         * (for IDX macro and finite difference stencils). */
        grid_t g_local;
        memset(&g_local, 0, sizeof(grid_t));
        g_local.N       = pack->N;
        g_local.ghost   = pack->ghost;
        g_local.Ntotal  = pack->Ntotal;
        g_local.dx      = pack->dx_per_block[b];
        g_local.npoints = npts;

        /* Evaluate RHS over interior cells [ghost, ghost+N)^3 */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int k = lo; k < hi; k++) {
            for (int j = lo; j < hi; j++) {
                for (int i = lo; i < hi; i++) {
                    ccz4_rhs_point(rhs_ptrs, (const double *const *)src_ptrs,
                                   &g_local, p, i, j, k);
                }
            }
        }
    }
}

/* ---- Batched Sommerfeld BCs ---- */

/*
 * Apply Sommerfeld radiative BCs to RHS for all blocks in the pack.
 *
 * For each block, iterates over ALL grid points. Skips interior cells
 * and ghost cells not adjacent to a domain boundary (those were filled
 * by ghost exchange). For remaining boundary ghost cells, applies:
 *   rhs = -(x^i/r) * d_i(f) + (f_asymptotic - f) / r
 *
 * Uses pack metadata for:
 *   - origins[b*3..b*3+2]: block origin for physical coordinates
 *   - dx_per_block[b]: grid spacing for boundary derivative
 *   - on_boundary[b*6..b*6+5]: which faces touch domain boundary
 *
 * Calls asymptotic_value() and boundary_d1() from sommerfeld.h.
 *
 * CPU: sequential over blocks, no inner OMP (boundary cells are sparse).
 */
void backend_sommerfeld_packed(meshblock_pack_t *pack, const sim_params_t *p)
{
    (void)p;
    int lo = pack->ghost;
    int hi = pack->ghost + pack->N;
    int Nt = pack->Ntotal;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;

    for (int b = 0; b < nb; b++) {
        /* Per-field pointer arrays into pack layout for this block */
        double *rhs_ptrs[NUM_FIELDS];
        const double *src_ptrs[NUM_FIELDS];
        for (int f = 0; f < NUM_FIELDS; f++) {
            size_t base = (size_t)f * nb * npts + (size_t)b * npts;
            src_ptrs[f] = pack->data + base;
            rhs_ptrs[f] = pack->rhs  + base;
        }

        /* Block-local metadata */
        double block_ox = pack->origins[b * 3 + 0];
        double block_oy = pack->origins[b * 3 + 1];
        double block_oz = pack->origins[b * 3 + 2];
        double dx = pack->dx_per_block[b];
        const int *ob = pack->on_boundary + b * 6;

        /* Stack-local grid_t for IDX, STRIDE macros */
        grid_t g_local;
        memset(&g_local, 0, sizeof(grid_t));
        g_local.N       = pack->N;
        g_local.ghost   = pack->ghost;
        g_local.Ntotal  = pack->Ntotal;
        g_local.dx      = dx;
        g_local.npoints = npts;

        /* Iterate over all points, apply Sommerfeld to boundary ghosts */
        for (int k = 0; k < Nt; k++) {
            for (int j = 0; j < Nt; j++) {
                for (int i = 0; i < Nt; i++) {
                    /* Skip interior points — no BCs needed */
                    if (i >= lo && i < hi &&
                        j >= lo && j < hi &&
                        k >= lo && k < hi)
                        continue;

                    /* Check if this ghost point borders a domain boundary.
                     * A point is near boundary face F if:
                     *   - It's in the ghost zone for direction F
                     *   - on_boundary[F] == 1 for this block
                     * Ref: apply_sommerfeld_block in sommerfeld.c */
                    int near_boundary = 0;
                    if (i < lo  && ob[0]) near_boundary = 1;
                    if (i >= hi && ob[1]) near_boundary = 1;
                    if (j < lo  && ob[2]) near_boundary = 1;
                    if (j >= hi && ob[3]) near_boundary = 1;
                    if (k < lo  && ob[4]) near_boundary = 1;
                    if (k >= hi && ob[5]) near_boundary = 1;

                    if (!near_boundary) continue;

                    int idx = IDX(&g_local, i, j, k);

                    /* Physical coordinates via block origin.
                     * Matches BLOCK_COORD(blk, dir, i) from block.h:
                     *   origin[dir] + (i - GHOST_WIDTH + 0.5) * dx */
                    double x = block_ox + (i - GHOST_WIDTH + 0.5) * dx;
                    double y = block_oy + (j - GHOST_WIDTH + 0.5) * dx;
                    double z = block_oz + (k - GHOST_WIDTH + 0.5) * dx;
                    double r = sqrt(x*x + y*y + z*z);
                    if (r < 1.0e-10) r = 1.0e-10;

                    /* Distance from each boundary edge (for stencil choice) */
                    int lo_off[3] = { i, j, k };
                    int hi_off[3] = { Nt - 1 - i, Nt - 1 - j, Nt - 1 - k };

                    int strides[3] = {
                        STRIDE_X,
                        STRIDE_Y(&g_local),
                        STRIDE_Z(&g_local)
                    };
                    double loc[3] = { x, y, z };

                    /* Apply Sommerfeld to each field:
                     * rhs = -sum_dir(d_dir(f) * x_dir / r)
                     *      + (f_asymptotic - f) / r
                     * Ref: GRChombo BoundaryConditions.cpp:593-661 */
                    for (int field = 0; field < NUM_FIELDS; field++) {
                        double sommerfeld = 0.0;

                        for (int dir = 0; dir < 3; dir++) {
                            double d1 = boundary_d1(
                                src_ptrs[field], idx,
                                strides[dir],
                                lo_off[dir], hi_off[dir],
                                dx);
                            sommerfeld += -d1 * loc[dir] / r;
                        }

                        double f_asym = asymptotic_value(field);
                        sommerfeld += (f_asym - src_ptrs[field][idx]) / r;

                        rhs_ptrs[field][idx] = sommerfeld;
                    }
                }
            }
        }
    }
}

/* ---- CK45 fused update ---- */

/*
 * Fused CK45 update over all fields, blocks, and points:
 *   scratch[i] = A_s * scratch[i] + dt * rhs[i]   (dU = A*dU + dt*F)
 *   data[i]   += B_s * scratch[i]                  (U += B*dU)
 *
 * Flat loop over the entire buffer: total = n_fields * n_blocks * npts.
 * This is the memory-bandwidth-limited inner kernel of CK45.
 */
void backend_update_ck45_packed(meshblock_pack_t *pack,
                                 double A_s, double B_s, double dt)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *data    = pack->data;
    double *scratch = pack->scratch;
    const double *rhs_data = pack->rhs;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; i++) {
        scratch[i] = A_s * scratch[i] + dt * rhs_data[i];
        data[i]   += B_s * scratch[i];
    }
}

/* ---- Classic RK4 packed operations ---- */

/*
 * Copy between pack buffers: dst_arr[i] = src_arr[i].
 * Used by classic RK4 to save initial state (scratch = data)
 * and restore it (data = scratch) at the end.
 */
void backend_copy_packed(meshblock_pack_t *pack, int dst, int src)
{
    double *dst_buf = pack_buffer(pack, dst);
    const double *src_buf = pack_buffer(pack, src);
    if (!dst_buf || !src_buf) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    memcpy(dst_buf, src_buf, total * sizeof(double));
}

/*
 * Accumulate weighted RHS into accumulator:
 *   accum[i] += weight * dt * rhs[i]
 *
 * Classic RK4 weights: 1/6, 1/3, 1/3, 1/6 for stages 1-4.
 */
void backend_accum_add_packed(meshblock_pack_t *pack, double weight, double dt)
{
    if (!pack->accum) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *accum   = pack->accum;
    const double *rhs_data = pack->rhs;
    double coeff = weight * dt;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; i++)
        accum[i] += coeff * rhs_data[i];
}

/*
 * Linear combination: data[i] = scratch[i] + alpha * dt * rhs[i].
 * Used by classic RK4 to compute intermediate states:
 *   U = U^0 + (dt/2)*F  or  U = U^0 + dt*F
 */
void backend_axpy_packed(meshblock_pack_t *pack, double alpha, double dt)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *data    = pack->data;
    const double *scratch = pack->scratch;
    const double *rhs_data = pack->rhs;
    double coeff = alpha * dt;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; i++)
        data[i] = scratch[i] + coeff * rhs_data[i];
}

/*
 * Apply accumulator: data[i] += accum[i].
 * Final step of classic RK4: U = U^0 + (dt/6)(F1 + 2*F2 + 2*F3 + F4).
 */
void backend_apply_accum_packed(meshblock_pack_t *pack)
{
    if (!pack->accum) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *data  = pack->data;
    const double *accum = pack->accum;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; i++)
        data[i] += accum[i];
}

/* ---- Ghost exchange (Commit 2 placeholder) ---- */

/*
 * GPU ghost exchange — not yet implemented (Commit 2).
 *
 * For Commit 1, the packed stepper in rk4.c uses a CPU fallback:
 *   1. meshblock_pack_store() — unpack fields to individual blocks
 *   2. ghost_exchange_multilevel() — run existing 5-phase exchange
 *   3. meshblock_pack_load() — repack fields into pack
 *
 * Commit 2 will replace this with device-side kernels for all 5 phases.
 * On CPU, it will call the equivalent OpenMP loops.
 *
 * This function should NOT be called in Commit 1 — it's a stub.
 */
void backend_ghost_exchange_packed(meshblock_pack_t *pack)
{
    (void)pack;
    /* Commit 2: implement 5-phase ghost exchange as CPU parallel loops.
     * For now, the stepper uses the manual fallback path. */
}
