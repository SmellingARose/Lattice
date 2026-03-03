/*
 * Lattice — 3D Numerical Relativity
 * CPU backend: OpenMP parallel loops for packed kernels.
 *
 * Packed API: all leaf blocks batched into meshblock_pack_t.
 * Outer loop over blocks, OpenMP parallel for over (k,j) per block.
 * GPU-analogous structure allows validation on macOS before GPU hardware.
 *
 * Map/unmap operations are no-ops on CPU since data is already in host memory.
 */

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
#include <string.h>
#include <math.h>

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

void backend_unmap_pack_sync(meshblock_pack_t *pack)
{
    (void)pack;
    /* CPU: data already in host memory, nothing to sync */
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
/*
 * Flattened packed RHS: single omp parallel region across all blocks.
 *
 * Instead of creating/destroying a parallel region per block (serial outer
 * block loop with inner omp parallel for), flatten (block, k, j) into one
 * index. Each thread computes which block and (k,j) it handles. Block
 * pointer arrays are cached and rebuilt only when the block index changes.
 *
 * This eliminates fork/join overhead for multi-block meshes (~10-15%).
 * Results are bit-identical to the per-block version.
 */
void backend_compute_rhs_packed(meshblock_pack_t *pack, const sim_params_t *p)
{
    int lo = pack->ghost;
    int hi = pack->ghost + pack->N;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int n_kj = (hi - lo) * (hi - lo);
    int total_work = nb * n_kj;

    #pragma omp parallel
    {
        /* Per-thread cached state: rebuild pointers only when block changes */
        int last_b = -1;
        double *rhs_ptrs[NUM_FIELDS];
        const double *src_ptrs[NUM_FIELDS];
        grid_t g_local;
        memset(&g_local, 0, sizeof(grid_t));
        g_local.N        = pack->N;
        g_local.ghost    = pack->ghost;
        g_local.Ntotal   = pack->Ntotal;
        g_local.npoints  = npts;
        g_local.n_fields = pack->n_fields;

        #pragma omp for schedule(static)
        for (int bkj = 0; bkj < total_work; bkj++) {
            int b = bkj / n_kj;
            int rem = bkj % n_kj;
            int k = lo + rem / (hi - lo);
            int j = lo + rem % (hi - lo);

            /* Rebuild per-field pointers when block changes */
            if (b != last_b) {
                for (int f = 0; f < pack->n_fields; f++) {
                    size_t base = (size_t)f * nb * npts + (size_t)b * npts;
                    src_ptrs[f] = pack->data + base;
                    rhs_ptrs[f] = pack->rhs  + base;
                }
                g_local.dx = pack->dx_per_block[b];
                g_local.inv_dx = 1.0 / pack->dx_per_block[b];
                last_b = b;
            }

            for (int i = lo; i < hi; i++) {
#ifdef LATTICE_EM_ENABLED
                if (p->em_enabled)
                    ccz4_maxwell_rhs_point(rhs_ptrs,
                                           (const double *const *)src_ptrs,
                                           &g_local, p, i, j, k);
                else
#endif
                    ccz4_rhs_point(rhs_ptrs,
                                   (const double *const *)src_ptrs,
                                   &g_local, p, i, j, k);
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
/*
 * Apply Sommerfeld/CP-BC RHS at a single ghost point for packed data.
 * Extracted to avoid duplicating the formula in each face loop.
 *
 * face_dir: dominant boundary direction (0=x, 1=y, 2=z), used for
 *           Gamma normal/tangential speed selection in CP mode.
 * s_sign:   outward normal sign (+1 or -1) along face_dir.
 * bc_type:  BC_SOMMERFELD or BC_CONSTRAINT_PRESERVING.
 *
 * For CP BCs, constraint fields (Theta, K, A_ij, Gamma^i) use the
 * characteristic-speed formula; all others use standard Sommerfeld.
 * Ref: arXiv:1212.2901 (Hilditch et al., BAM)
 */
static inline void packed_sommerfeld_point(
    double *const *rhs_ptrs, const double *const *src_ptrs,
    const grid_t *g, int nf, int i, int j, int k,
    double x, double y, double z,
    int face_dir, int s_sign, bc_type_t bc_type)
{
    int idx = IDX(g, i, j, k);
    int Nt = g->Ntotal;

    double r = sqrt(x*x + y*y + z*z);
    if (r < 1.0e-10) r = 1.0e-10;

    int lo_off[3] = { i, j, k };
    int hi_off[3] = { Nt - 1 - i, Nt - 1 - j, Nt - 1 - k };
    int strides[3] = { STRIDE_X, STRIDE_Y(g), STRIDE_Z(g) };
    double loc[3] = { x, y, z };

    /* Read lapse for CP speed computation */
    double alpha = src_ptrs[FIELD_LAPSE][idx];

    for (int field = 0; field < nf; field++) {
        double speed = (bc_type == BC_CONSTRAINT_PRESERVING)
                       ? cp_char_speed(field, face_dir, alpha) : 0.0;

        if (speed > 0.0) {
            /* CP-BC: one-sided derivative in the face normal direction */
            double df_ds = boundary_d1(src_ptrs[field], idx,
                                       strides[face_dir],
                                       lo_off[face_dir],
                                       hi_off[face_dir], g->inv_dx);
            double f_asym = asymptotic_value(field);
            rhs_ptrs[field][idx] = cp_rhs(alpha, speed, s_sign, df_ds,
                                          src_ptrs[field][idx], f_asym, r);
        } else {
            /* Standard Sommerfeld */
            double sommerfeld = 0.0;
            for (int dir = 0; dir < 3; dir++) {
                double d1 = boundary_d1(src_ptrs[field], idx, strides[dir],
                                        lo_off[dir], hi_off[dir], g->inv_dx);
                sommerfeld += -d1 * loc[dir] / r;
            }
            double f_asym = asymptotic_value(field);
            sommerfeld += (f_asym - src_ptrs[field][idx]) / r;
            rhs_ptrs[field][idx] = sommerfeld;
        }
    }
}

/*
 * Packed Sommerfeld BCs: iterate only over boundary-face ghost slabs.
 * For each block, only faces with on_boundary[face] == 1 are processed.
 * Points at face intersections (edges/corners) may be visited multiple
 * times but the Sommerfeld formula is idempotent.
 *
 * Replaces the original Nt^3 loop with interior-skip + near_boundary
 * check, eliminating 80-97% of wasted iterations per block.
 */
void backend_sommerfeld_packed(meshblock_pack_t *pack, const sim_params_t *p)
{
    int lo = pack->ghost;
    int hi = pack->ghost + pack->N;
    int Nt = pack->Ntotal;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    bc_type_t bc_type = p->bc_type;

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < nb; b++) {
        const int *ob = pack->on_boundary + b * 6;

        /* Skip blocks with no boundary faces */
        if (!ob[0] && !ob[1] && !ob[2] && !ob[3] && !ob[4] && !ob[5])
            continue;

        /* Per-field pointer arrays into pack layout for this block */
        double *rhs_ptrs[NUM_FIELDS];
        const double *src_ptrs[NUM_FIELDS];
        for (int f = 0; f < pack->n_fields; f++) {
            size_t base = (size_t)f * nb * npts + (size_t)b * npts;
            src_ptrs[f] = pack->data + base;
            rhs_ptrs[f] = pack->rhs  + base;
        }

        double block_ox = pack->origins[b * 3 + 0];
        double block_oy = pack->origins[b * 3 + 1];
        double block_oz = pack->origins[b * 3 + 2];
        double dx = pack->dx_per_block[b];

        grid_t g_local;
        memset(&g_local, 0, sizeof(grid_t));
        g_local.N       = pack->N;
        g_local.ghost   = pack->ghost;
        g_local.Ntotal  = pack->Ntotal;
        g_local.dx      = dx;
        g_local.inv_dx  = 1.0 / dx;
        g_local.npoints = npts;

        int nf = pack->n_fields;

        /* Macro: physical coord from block origin + local index */
        #define BX(ii) (block_ox + ((ii) - GHOST_WIDTH + 0.5) * dx)
        #define BY(jj) (block_oy + ((jj) - GHOST_WIDTH + 0.5) * dx)
        #define BZ(kk) (block_oz + ((kk) - GHOST_WIDTH + 0.5) * dx)

        /* X- face: face_dir=0, s_sign=-1 (outward is -x) */
        if (ob[0])
            for (int k = 0; k < Nt; k++)
                for (int j = 0; j < Nt; j++)
                    for (int i = 0; i < lo; i++)
                        packed_sommerfeld_point(rhs_ptrs,
                            (const double *const *)src_ptrs, &g_local,
                            nf, i, j, k, BX(i), BY(j), BZ(k),
                            0, -1, bc_type);

        /* X+ face: face_dir=0, s_sign=+1 (outward is +x) */
        if (ob[1])
            for (int k = 0; k < Nt; k++)
                for (int j = 0; j < Nt; j++)
                    for (int i = hi; i < Nt; i++)
                        packed_sommerfeld_point(rhs_ptrs,
                            (const double *const *)src_ptrs, &g_local,
                            nf, i, j, k, BX(i), BY(j), BZ(k),
                            0, +1, bc_type);

        /* Y- face: face_dir=1, s_sign=-1 */
        if (ob[2])
            for (int k = 0; k < Nt; k++)
                for (int j = 0; j < lo; j++)
                    for (int i = 0; i < Nt; i++)
                        packed_sommerfeld_point(rhs_ptrs,
                            (const double *const *)src_ptrs, &g_local,
                            nf, i, j, k, BX(i), BY(j), BZ(k),
                            1, -1, bc_type);

        /* Y+ face: face_dir=1, s_sign=+1 */
        if (ob[3])
            for (int k = 0; k < Nt; k++)
                for (int j = hi; j < Nt; j++)
                    for (int i = 0; i < Nt; i++)
                        packed_sommerfeld_point(rhs_ptrs,
                            (const double *const *)src_ptrs, &g_local,
                            nf, i, j, k, BX(i), BY(j), BZ(k),
                            1, +1, bc_type);

        /* Z- face: face_dir=2, s_sign=-1 */
        if (ob[4])
            for (int k = 0; k < lo; k++)
                for (int j = 0; j < Nt; j++)
                    for (int i = 0; i < Nt; i++)
                        packed_sommerfeld_point(rhs_ptrs,
                            (const double *const *)src_ptrs, &g_local,
                            nf, i, j, k, BX(i), BY(j), BZ(k),
                            2, -1, bc_type);

        /* Z+ face: face_dir=2, s_sign=+1 */
        if (ob[5])
            for (int k = hi; k < Nt; k++)
                for (int j = 0; j < Nt; j++)
                    for (int i = 0; i < Nt; i++)
                        packed_sommerfeld_point(rhs_ptrs,
                            (const double *const *)src_ptrs, &g_local,
                            nf, i, j, k, BX(i), BY(j), BZ(k),
                            2, +1, bc_type);

        #undef BX
        #undef BY
        #undef BZ
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

/*
 * Fused RK4 stage update (stages 1-3):
 *   accum[i] += weight * dt * rhs[i]
 *   data[i]   = scratch[i] + alpha * dt * rhs[i]
 * Saves 1 pass over rhs[] compared to separate accum_add + axpy.
 */
void backend_rk4_stage_packed(meshblock_pack_t *pack,
                               double weight, double alpha, double dt)
{
    if (!pack->accum) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *data    = pack->data;
    double *accum   = pack->accum;
    const double *scratch = pack->scratch;
    const double *rhs     = pack->rhs;
    double w_dt = weight * dt;
    double a_dt = alpha * dt;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; i++) {
        double r = rhs[i];
        accum[i] += w_dt * r;
        data[i]   = scratch[i] + a_dt * r;
    }
}

/*
 * Fused RK4 final update (stage 4):
 *   data[i] = scratch[i] + accum[i] + weight * dt * rhs[i]
 * Replaces accum_add + copy + apply_accum (3 passes → 1).
 */
void backend_rk4_final_packed(meshblock_pack_t *pack, double weight, double dt)
{
    if (!pack->accum) return;

    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    double *data    = pack->data;
    const double *accum   = pack->accum;
    const double *scratch = pack->scratch;
    const double *rhs     = pack->rhs;
    double w_dt = weight * dt;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; i++)
        data[i] = scratch[i] + accum[i] + w_dt * rhs[i];
}

/* ========================================================================
 * Packed ghost exchange — all 5 phases on contiguous pack buffers.
 *
 * Replaces the Commit 1 fallback (unpack → exchange → repack) with
 * direct operations on pack->data and pack->coarse_data.
 *
 * CPU backend: OpenMP parallel for where beneficial.
 * GPU backend: identical algorithm with omp target teams distribute.
 *
 * Ref: ghost_exchange_multilevel() in ghost_exchange.c (per-block version)
 * Ref: AthenaK src/bvals/ (GPU boundary exchange, coarse-buffer)
 * ======================================================================== */

/*
 * Compute source and destination index ranges for one direction.
 * Same logic as ghost_range() in ghost_exchange.c.
 */
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

/*
 * Phase 0+1: Same-level exchange on pack->data.
 * For each block b and each of 26 neighbors, if the neighbor is in the
 * pack and at the same level, copy from neighbor's interior to b's ghost.
 * Safe to parallelize over blocks: ghost exchange only READS interiors
 * and WRITES ghost zones (disjoint sets).
 */
static void packed_exchange_same_level(meshblock_pack_t *pack)
{
    int nb = pack->n_blocks;
    int ghost = pack->ghost;
    int N = pack->N;
    int Nt = pack->Ntotal;
    size_t npts = pack->npts;

    /* Safe to parallelize: each block's ghost writes are disjoint */
    #pragma omp parallel for schedule(dynamic)
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

/*
 * Phase 2: Restrict fine block interior → own coarse_buf in pack.
 * 6th-order cell-average restriction (6×6×6 Lagrange stencil).
 * Stencil [base-2, base+3] fits within [0, Nt_f) for all coarse cells
 * since ghost width (4) exceeds stencil reach (2).
 *
 * Reads: pack->data (fine block)
 * Writes: pack->coarse_data (coarse_buf interior)
 *
 * Ref: restrict_to_coarse_buf() in restriction.c
 */
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

    #pragma omp parallel for schedule(dynamic)
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

                        /* 6th-order: 6-point stencil */
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

/*
 * Phase 3: Fill coarse_buf ghost zones from sibling coarse_bufs and
 * coarser neighbors' main data.
 *
 * For each refined block b, each of 26 neighbor directions:
 *   - Same-level sibling → copy from sibling's coarse_data interior
 *   - Coarser neighbor → copy from coarser block's pack->data (index offset)
 *   - Domain boundary → skip (Phase 3.5 handles it)
 *
 * Ref: fill_coarse_buf_ghosts() in ghost_exchange.c
 */
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

    #pragma omp parallel for schedule(dynamic)
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
                /* Same-level sibling: exchange coarse_bufs */
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
                            size_t d = dst_off
                                + (size_t)(dz_lo+k) * Nt_c * Nt_c
                                + (size_t)(dy_lo+j) * Nt_c + dx_lo;
                            size_t s = src_off
                                + (size_t)(sz_lo+k) * Nt_c * Nt_c
                                + (size_t)(sy_lo+j) * Nt_c + sx_lo;
                            memcpy(&pack->coarse_data[d],
                                   &pack->coarse_data[s],
                                   nx * sizeof(double));
                        }
                    }
                }

            } else if (nlev >= 0 && nlev == blk_level - 1) {
                /* Coarser neighbor: copy from its main grid in pack->data.
                 * Both the coarser block's grid and this block's coarse_buf
                 * share the same dx, so use integer index offset. */
                int pack_nbr = pack->neighbor_table[b * NUM_NEIGHBORS + n];
                if (pack_nbr < 0) continue;

                double dx_c = pack->dx_per_block[pack_nbr];
                int off_i = (int)round(
                    (pack->origins[b*3+0] - pack->origins[pack_nbr*3+0]) / dx_c);
                int off_j = (int)round(
                    (pack->origins[b*3+1] - pack->origins[pack_nbr*3+1]) / dx_c);
                int off_k = (int)round(
                    (pack->origins[b*3+2] - pack->origins[pack_nbr*3+2]) / dx_c);

                /* Ghost range on coarse_buf (destination) */
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
            /* nlev < 0: domain boundary — skip, Phase 3.5 handles */
        }
    }
}

/*
 * Phase 3.5: Quadratic extrapolation for domain-boundary ghost cells of
 * coarse_bufs. Dimension sweep: X faces, then Y, then Z.
 * Each later sweep reads data written by earlier sweeps (handles edges/corners).
 *
 * Ref: fill_coarse_buf_boundary() in ghost_exchange.c
 * Ref: AthenaK src/bvals/physics/z4c_bcs.cpp BCHelper
 */
static void packed_fill_coarse_boundary(meshblock_pack_t *pack)
{
    if (pack->n_refined == 0) return;

    int nb = pack->n_blocks;
    int gh = pack->ghost;
    int N_c = pack->coarse_N;
    int Nt_c = pack->coarse_Ntotal;
    size_t cnpts = pack->coarse_npts;

    /* 3-point Lagrange extrapolation coefficients */
    double c[4][3];
    for (int d = 0; d < gh && d < 4; d++) {
        double t = -(d + 1);
        c[d][0] = (t - 1.0) * (t - 2.0) / 2.0;
        c[d][1] = -t * (t - 2.0);
        c[d][2] = t * (t - 1.0) / 2.0;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < nb; b++) {
        int r = pack->refined_map[b];
        if (r < 0) continue;

        /* Face boundary flags from nblevel_table */
        int xm = pack->nblevel_table[b*27 + 1*9 + 1*3 + 0] < 0;
        int xp = pack->nblevel_table[b*27 + 1*9 + 1*3 + 2] < 0;
        int ym = pack->nblevel_table[b*27 + 1*9 + 0*3 + 1] < 0;
        int yp = pack->nblevel_table[b*27 + 1*9 + 2*3 + 1] < 0;
        int zm = pack->nblevel_table[b*27 + 0*9 + 1*3 + 1] < 0;
        int zp = pack->nblevel_table[b*27 + 2*9 + 1*3 + 1] < 0;

        for (int f = 0; f < pack->n_fields; f++) {
            double *data = pack->coarse_data
                + (size_t)r * pack->n_fields * cnpts + (size_t)f * cnpts;

            /* X-faces */
            if (xm) {
                for (int k = 0; k < Nt_c; k++)
                    for (int j = 0; j < Nt_c; j++)
                        for (int d = 0; d < gh; d++) {
                            int gi = gh - 1 - d;
                            data[gi + j*Nt_c + k*Nt_c*Nt_c] =
                                c[d][0] * data[gh     + j*Nt_c + k*Nt_c*Nt_c] +
                                c[d][1] * data[(gh+1) + j*Nt_c + k*Nt_c*Nt_c] +
                                c[d][2] * data[(gh+2) + j*Nt_c + k*Nt_c*Nt_c];
                        }
            }
            if (xp) {
                for (int k = 0; k < Nt_c; k++)
                    for (int j = 0; j < Nt_c; j++)
                        for (int d = 0; d < gh; d++) {
                            int gi = gh + N_c + d;
                            data[gi + j*Nt_c + k*Nt_c*Nt_c] =
                                c[d][0] * data[(gh+N_c-1) + j*Nt_c + k*Nt_c*Nt_c] +
                                c[d][1] * data[(gh+N_c-2) + j*Nt_c + k*Nt_c*Nt_c] +
                                c[d][2] * data[(gh+N_c-3) + j*Nt_c + k*Nt_c*Nt_c];
                        }
            }
            /* Y-faces (X ghosts already filled by X sweep) */
            if (ym) {
                for (int k = 0; k < Nt_c; k++)
                    for (int i = 0; i < Nt_c; i++)
                        for (int d = 0; d < gh; d++) {
                            int gj = gh - 1 - d;
                            data[i + gj*Nt_c + k*Nt_c*Nt_c] =
                                c[d][0] * data[i + gh*Nt_c     + k*Nt_c*Nt_c] +
                                c[d][1] * data[i + (gh+1)*Nt_c + k*Nt_c*Nt_c] +
                                c[d][2] * data[i + (gh+2)*Nt_c + k*Nt_c*Nt_c];
                        }
            }
            if (yp) {
                for (int k = 0; k < Nt_c; k++)
                    for (int i = 0; i < Nt_c; i++)
                        for (int d = 0; d < gh; d++) {
                            int gj = gh + N_c + d;
                            data[i + gj*Nt_c + k*Nt_c*Nt_c] =
                                c[d][0] * data[i + (gh+N_c-1)*Nt_c + k*Nt_c*Nt_c] +
                                c[d][1] * data[i + (gh+N_c-2)*Nt_c + k*Nt_c*Nt_c] +
                                c[d][2] * data[i + (gh+N_c-3)*Nt_c + k*Nt_c*Nt_c];
                        }
            }
            /* Z-faces (X+Y ghosts already filled) */
            if (zm) {
                for (int j = 0; j < Nt_c; j++)
                    for (int i = 0; i < Nt_c; i++)
                        for (int d = 0; d < gh; d++) {
                            int gk = gh - 1 - d;
                            data[i + j*Nt_c + gk*Nt_c*Nt_c] =
                                c[d][0] * data[i + j*Nt_c + gh*Nt_c*Nt_c] +
                                c[d][1] * data[i + j*Nt_c + (gh+1)*Nt_c*Nt_c] +
                                c[d][2] * data[i + j*Nt_c + (gh+2)*Nt_c*Nt_c];
                        }
            }
            if (zp) {
                for (int j = 0; j < Nt_c; j++)
                    for (int i = 0; i < Nt_c; i++)
                        for (int d = 0; d < gh; d++) {
                            int gk = gh + N_c + d;
                            data[i + j*Nt_c + gk*Nt_c*Nt_c] =
                                c[d][0] * data[i + j*Nt_c + (gh+N_c-1)*Nt_c*Nt_c] +
                                c[d][1] * data[i + j*Nt_c + (gh+N_c-2)*Nt_c*Nt_c] +
                                c[d][2] * data[i + j*Nt_c + (gh+N_c-3)*Nt_c*Nt_c];
                        }
            }
        }
    }
}

/*
 * Phase 4: Prolongate from own coarse_buf → fine ghost zones.
 * For each fine ghost cell, if the neighbor direction was NOT filled by
 * Phase 1 (same-level), interpolate from coarse_data using PROLONG_STENCIL^3 Lagrange.
 *
 * Reads: pack->coarse_data
 * Writes: pack->data (ghost zones only)
 *
 * Ref: prolongate_from_own_coarse_buf() in ghost_exchange.c
 * Ref: AthenaK prolongation.hpp HighOrderProlongCC
 */
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

    #pragma omp parallel for schedule(dynamic)
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
                        /* Skip interior cells */
                        if (fi >= ghost_f && fi < ghost_f + N_f &&
                            fj >= ghost_f && fj < ghost_f + N_f &&
                            fk >= ghost_f && fk < ghost_f + N_f)
                            continue;

                        /* Ghost zone direction */
                        int ox = (fi < ghost_f) ? -1 :
                                 (fi >= ghost_f + N_f) ? 1 : 0;
                        int oy = (fj < ghost_f) ? -1 :
                                 (fj >= ghost_f + N_f) ? 1 : 0;
                        int oz = (fk < ghost_f) ? -1 :
                                 (fk >= ghost_f + N_f) ? 1 : 0;

                        /* Skip if same-level neighbor filled this in Phase 1 */
                        int nlev = pack->nblevel_table[
                            b*27 + (oz+1)*9 + (oy+1)*3 + (ox+1)];
                        if (nlev == blk_level) continue;

                        /* Map fine index to coarse_buf continuous coordinate */
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

                        /* Left (0) or right (1) child */
                        int oi = (ci_cont >= ci0) ? 1 : 0;
                        int oj = (cj_cont >= cj0) ? 1 : 0;
                        int ok = (ck_cont >= ck0) ? 1 : 0;

                        /* 7×7×7 tensor product Lagrange interpolation */
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
 * Device-side ghost exchange: all 5 phases on pack buffers.
 *
 * CPU backend: OpenMP parallel for where beneficial.
 * Eliminates the Commit 1 fallback (unpack → exchange → repack)
 * by operating directly on pack->data and pack->coarse_data.
 *
 * For uniform meshes (n_refined == 0), only Phase 0+1 runs.
 *
 * Ref: ghost_exchange_multilevel() in ghost_exchange.c
 */
void backend_ghost_exchange_packed(meshblock_pack_t *pack)
{
    /* Phase 0+1: Same-level exchange at all levels */
    packed_exchange_same_level(pack);

    if (pack->n_refined == 0) return;

    /* Phase 2: Restrict fine → own coarse_buf */
    packed_restrict_to_coarse(pack);

    /* Phase 3: Fill coarse_buf ghosts from siblings + coarser neighbors */
    packed_fill_coarse_buf_ghosts(pack);

    /* Phase 3.5: Boundary extrapolation on coarse_buf */
    packed_fill_coarse_boundary(pack);

    /* Phase 4: Prolongate coarse_buf → fine ghosts */
    packed_prolongate_fine_ghosts(pack);
}

/*
 * Enforce algebraic constraints on packed data: det(h)=1, tr(A)=0.
 * Flattened (block, k, j) outer loop with single OMP parallel region.
 * Same physics as enforce_algebraic() in rk4.c, applied to pack buffers.
 *
 * Ref: GRChombo CCZ4/TraceARemoval.hpp, CCZ4/PositiveChiAndAlpha.hpp
 */
void backend_enforce_algebraic_packed(meshblock_pack_t *pack)
{
    int Nt = pack->Ntotal;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int n_kj = Nt * Nt;
    int total_work = nb * n_kj;

    #pragma omp parallel for schedule(static)
    for (int bkj = 0; bkj < total_work; bkj++) {
        int b = bkj / n_kj;
        int rem = bkj % n_kj;
        int k = rem / Nt;
        int j = rem % Nt;

        for (int i = 0; i < Nt; i++) {
            int idx = k * Nt * Nt + j * Nt + i;

            /* Per-field pointers for this block */
            #define FIELD_PTR(fld) (pack->data + (size_t)(fld) * nb * npts \
                                    + (size_t)b * npts)

            /* Load h_ij */
            double h_loc[3][3];
            h_loc[0][0] = FIELD_PTR(FIELD_H11)[idx];
            h_loc[0][1] = FIELD_PTR(FIELD_H12)[idx];
            h_loc[0][2] = FIELD_PTR(FIELD_H13)[idx];
            h_loc[1][0] = h_loc[0][1];
            h_loc[1][1] = FIELD_PTR(FIELD_H22)[idx];
            h_loc[1][2] = FIELD_PTR(FIELD_H23)[idx];
            h_loc[2][0] = h_loc[0][2];
            h_loc[2][1] = h_loc[1][2];
            h_loc[2][2] = FIELD_PTR(FIELD_H33)[idx];

            /* Enforce det(h) = 1 by rescaling */
            double det = compute_det_sym(h_loc);
            double scale = fast_inv_cbrt(det);
            FOR2(a, bb) h_loc[a][bb] *= scale;

            FIELD_PTR(FIELD_H11)[idx] = h_loc[0][0];
            FIELD_PTR(FIELD_H12)[idx] = h_loc[0][1];
            FIELD_PTR(FIELD_H13)[idx] = h_loc[0][2];
            FIELD_PTR(FIELD_H22)[idx] = h_loc[1][1];
            FIELD_PTR(FIELD_H23)[idx] = h_loc[1][2];
            FIELD_PTR(FIELD_H33)[idx] = h_loc[2][2];

            /* Enforce tr(A) = 0. Use unit-det inverse since det(h)=1
             * was just enforced above — saves det computation + division. */
            double h_UU[3][3];
            compute_inverse_sym_unit_det(h_loc, h_UU);

            double A_loc[3][3];
            A_loc[0][0] = FIELD_PTR(FIELD_A11)[idx];
            A_loc[0][1] = FIELD_PTR(FIELD_A12)[idx];
            A_loc[0][2] = FIELD_PTR(FIELD_A13)[idx];
            A_loc[1][0] = A_loc[0][1];
            A_loc[1][1] = FIELD_PTR(FIELD_A22)[idx];
            A_loc[1][2] = FIELD_PTR(FIELD_A23)[idx];
            A_loc[2][0] = A_loc[0][2];
            A_loc[2][1] = A_loc[1][2];
            A_loc[2][2] = FIELD_PTR(FIELD_A33)[idx];

            make_trace_free(A_loc, h_loc, h_UU);

            FIELD_PTR(FIELD_A11)[idx] = A_loc[0][0];
            FIELD_PTR(FIELD_A12)[idx] = A_loc[0][1];
            FIELD_PTR(FIELD_A13)[idx] = A_loc[0][2];
            FIELD_PTR(FIELD_A22)[idx] = A_loc[1][1];
            FIELD_PTR(FIELD_A23)[idx] = A_loc[1][2];
            FIELD_PTR(FIELD_A33)[idx] = A_loc[2][2];

            /* Ensure chi > 0, lapse > 0 */
            if (FIELD_PTR(FIELD_CHI)[idx] < 1.0e-12)
                FIELD_PTR(FIELD_CHI)[idx] = 1.0e-12;
            if (FIELD_PTR(FIELD_LAPSE)[idx] < 1.0e-12)
                FIELD_PTR(FIELD_LAPSE)[idx] = 1.0e-12;

            #undef FIELD_PTR
        }
    }
}

/* ========================================================================
 * Runtime backend detection
 * ======================================================================== */

int backend_is_gpu(void) { return 0; }

/* ========================================================================
 * GPU diagnostic kernels — CPU implementations
 *
 * Map/unmap are no-ops. Diagnostic functions reuse the pack's host data
 * directly, mirroring the GPU kernel logic with OpenMP parallel loops.
 * ======================================================================== */

void backend_map_pack_diag(meshblock_pack_t *pack) { (void)pack; }
void backend_unmap_pack_diag(meshblock_pack_t *pack) { (void)pack; }

#include "../diagnostics/constraints.h"

double backend_constraint_l2_packed(meshblock_pack_t *pack)
{
    int nb = pack->n_blocks;
    int N = pack->N;
    int ghost = pack->ghost;
    int Nt = pack->Ntotal;
    size_t npts = pack->npts;
    int nf = pack->n_fields;

    double sum = 0.0;
    int count = 0;

    #pragma omp parallel for schedule(static) reduction(+:sum,count)
    for (int bpt = 0; bpt < nb * N * N * N; bpt++) {
        int b = bpt / (N * N * N);
        int pt = bpt % (N * N * N);
        int i = ghost + pt % N;
        int j = ghost + (pt / N) % N;
        int k = ghost + pt / (N * N);

        const double *src_ptrs[NUM_FIELDS];
        for (int f = 0; f < nf; f++)
            src_ptrs[f] = pack->data + (size_t)f * nb * npts + (size_t)b * npts;

        grid_t g_local;
        memset(&g_local, 0, sizeof(grid_t));
        g_local.N = N;
        g_local.ghost = ghost;
        g_local.Ntotal = Nt;
        g_local.npoints = npts;
        g_local.n_fields = nf;
        g_local.dx = pack->dx_per_block[b];
        g_local.inv_dx = 1.0 / pack->dx_per_block[b];

        double H = compute_hamiltonian_at(
            (const double *const *)src_ptrs, &g_local, i, j, k);
        sum += H * H;
        count++;
    }

    return (count > 0) ? sqrt(sum / count) : 0.0;
}

double backend_momentum_l2_packed(meshblock_pack_t *pack)
{
    int nb = pack->n_blocks;
    int N = pack->N;
    int ghost = pack->ghost;
    int Nt = pack->Ntotal;
    size_t npts = pack->npts;
    int nf = pack->n_fields;

    double sum = 0.0;
    int count = 0;

    #pragma omp parallel for schedule(static) reduction(+:sum,count)
    for (int bpt = 0; bpt < nb * N * N * N; bpt++) {
        int b = bpt / (N * N * N);
        int pt = bpt % (N * N * N);
        int i = ghost + pt % N;
        int j = ghost + (pt / N) % N;
        int k = ghost + pt / (N * N);

        const double *src_ptrs[NUM_FIELDS];
        for (int f = 0; f < nf; f++)
            src_ptrs[f] = pack->data + (size_t)f * nb * npts + (size_t)b * npts;

        grid_t g_local;
        memset(&g_local, 0, sizeof(grid_t));
        g_local.N = N;
        g_local.ghost = ghost;
        g_local.Ntotal = Nt;
        g_local.npoints = npts;
        g_local.n_fields = nf;
        g_local.dx = pack->dx_per_block[b];
        g_local.inv_dx = 1.0 / pack->dx_per_block[b];

        double mom[3];
        compute_momentum_at(
            (const double *const *)src_ptrs, &g_local, i, j, k, mom);
        sum += mom[0]*mom[0] + mom[1]*mom[1] + mom[2]*mom[2];
        count++;
    }

    return (count > 0) ? sqrt(sum / (3 * count)) : 0.0;
}

double backend_min_lapse_packed(meshblock_pack_t *pack,
                                 double *out_x, double *out_y, double *out_z)
{
    int nb = pack->n_blocks;
    int N = pack->N;
    int ghost = pack->ghost;
    int Nt = pack->Ntotal;
    size_t npts = pack->npts;

    double global_min = 1.0e30;
    double gx = 0.0, gy = 0.0, gz = 0.0;

    #pragma omp parallel
    {
        double local_min = 1.0e30;
        double lx = 0.0, ly = 0.0, lz = 0.0;

        #pragma omp for schedule(static)
        for (int bpt = 0; bpt < nb * N * N * N; bpt++) {
            int b = bpt / (N * N * N);
            int pt = bpt % (N * N * N);
            int i = ghost + pt % N;
            int j = ghost + (pt / N) % N;
            int k = ghost + pt / (N * N);

            int idx = k * Nt * Nt + j * Nt + i;
            size_t off = (size_t)FIELD_LAPSE * nb * npts + (size_t)b * npts + idx;
            double a = pack->data[off];

            if (a < local_min) {
                local_min = a;
                double dx = pack->dx_per_block[b];
                lx = pack->origins[b * 3 + 0] + (i - ghost + 0.5) * dx;
                ly = pack->origins[b * 3 + 1] + (j - ghost + 0.5) * dx;
                lz = pack->origins[b * 3 + 2] + (k - ghost + 0.5) * dx;
            }
        }

        #pragma omp critical
        {
            if (local_min < global_min) {
                global_min = local_min;
                gx = lx; gy = ly; gz = lz;
            }
        }
    }

    *out_x = gx; *out_y = gy; *out_z = gz;
    return global_min;
}

double backend_bh_separation_packed(meshblock_pack_t *pack, double excl_radius,
                                      double *x1, double *y1, double *z1,
                                      double *x2, double *y2, double *z2)
{
    /* Pass 1: find global lapse minimum (BH #1) */
    double px1, py1, pz1;
    (void)backend_min_lapse_packed(pack, &px1, &py1, &pz1);
    *x1 = px1; *y1 = py1; *z1 = pz1;

    /* Pass 2: find deepest minimum at least excl_radius from BH #1 */
    int nb = pack->n_blocks;
    int N = pack->N;
    int ghost = pack->ghost;
    int Nt = pack->Ntotal;
    size_t npts = pack->npts;

    double best2 = 1.0e30;
    double px2 = 0.0, py2 = 0.0, pz2 = 0.0;

    #pragma omp parallel
    {
        double local_min = 1.0e30;
        double lx = 0.0, ly = 0.0, lz = 0.0;

        #pragma omp for schedule(static)
        for (int bpt = 0; bpt < nb * N * N * N; bpt++) {
            int b = bpt / (N * N * N);
            int pt = bpt % (N * N * N);
            int i = ghost + pt % N;
            int j = ghost + (pt / N) % N;
            int k = ghost + pt / (N * N);

            int idx = k * Nt * Nt + j * Nt + i;
            size_t off = (size_t)FIELD_LAPSE * nb * npts + (size_t)b * npts + idx;
            double a = pack->data[off];

            if (a < local_min) {
                double dx = pack->dx_per_block[b];
                double cx = pack->origins[b * 3 + 0] + (i - ghost + 0.5) * dx;
                double cy = pack->origins[b * 3 + 1] + (j - ghost + 0.5) * dx;
                double cz = pack->origins[b * 3 + 2] + (k - ghost + 0.5) * dx;
                double dr = sqrt((cx - px1) * (cx - px1) +
                                 (cy - py1) * (cy - py1) +
                                 (cz - pz1) * (cz - pz1));
                if (dr > excl_radius) {
                    local_min = a;
                    lx = cx; ly = cy; lz = cz;
                }
            }
        }

        #pragma omp critical
        {
            if (local_min < best2) {
                best2 = local_min;
                px2 = lx; py2 = ly; pz2 = lz;
            }
        }
    }

    *x2 = px2; *y2 = py2; *z2 = pz2;

    if (best2 > 0.99) return 0.0;
    return sqrt((px1 - px2) * (px1 - px2) +
                (py1 - py2) * (py1 - py2) +
                (pz1 - pz2) * (pz1 - pz2));
}

int backend_check_finite_packed(meshblock_pack_t *pack)
{
    size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
    int ok = 1;

    #pragma omp parallel for schedule(static) reduction(min:ok)
    for (size_t i = 0; i < total; i++) {
        if (!isfinite(pack->data[i]))
            ok = 0;
    }

    return ok;
}

#include "../diagnostics/psi4.h"

void backend_psi4_extract_packed(meshblock_pack_t *pack,
                                   psi4_workspace_t *ws,
                                   const struct mesh_s *m)
{
    (void)pack;  /* CPU: pack unused, operate on mesh blocks directly */
    psi4_extract(ws, m);
}

/* ========================================================================
 * Multigrid solver packed kernel API — CPU implementations
 *
 * All operations use OpenMP parallel loops calling the shared point
 * functions from mg_smooth_point.h. Map/unmap are no-ops on CPU.
 *
 * Pack buffer layout: data[f * nb * npts + b * npts + idx]
 * Solver uses 10 fields: 4 solution + 6 background.
 * ======================================================================== */

void backend_map_solver_pack(meshblock_pack_t *pack, int slot)
{
    (void)pack; (void)slot; /* no-op for CPU */
}

void backend_unmap_solver_pack_sync(meshblock_pack_t *pack, int slot)
{
    (void)pack; (void)slot; /* no-op for CPU */
}

void backend_sync_solver_data_to_host(meshblock_pack_t *pack, int slot)
{
    (void)pack; (void)slot; /* no-op for CPU */
}

void backend_sync_solver_data_to_device(meshblock_pack_t *pack, int slot)
{
    (void)pack; (void)slot; /* no-op for CPU */
}

/*
 * 8-color Newton-GS smoother — CPU implementation.
 * For each block in the pack, iterates over colored interior points
 * and applies the point-wise Newton-GS update.
 *
 * Ref: relaxation_amr.c:smooth_block_1field/smooth_block_4field
 */
void backend_mg_smooth_packed(meshblock_pack_t *pack, int slot, int color,
                               int four_field)
{
    (void)slot;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;
    int c0 = color & 1, c1 = (color >> 1) & 1, c2 = (color >> 2) & 1;

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < nb; b++) {
        double dx = pack->dx_per_block[b];
        double dx2 = dx * dx;
        double inv_dx = 1.0 / dx;
        int sx = 1, sy = Nt, sz = Nt * Nt;

        double *psi    = mg_pack_field(pack->data, MGP_SOL_PSI, b, nb, npts);
        const double *psi_BL = mg_pack_field_c(pack->data, MGP_BG_PSI_BL, b, nb, npts);
        const double *A2     = mg_pack_field_c(pack->data, MGP_BG_A2, b, nb, npts);
        const double *f_psi  = mg_pack_field_c(pack->rhs, MGP_SOL_PSI, b, nb, npts);

        if (four_field) {
            double *V0 = mg_pack_field(pack->data, MGP_SOL_V1, b, nb, npts);
            double *V1 = mg_pack_field(pack->data, MGP_SOL_V2, b, nb, npts);
            double *V2 = mg_pack_field(pack->data, MGP_SOL_V3, b, nb, npts);
            const double *R_tilde = mg_pack_field_c(pack->data, MGP_BG_RTILDE, b, nb, npts);
            const double *SM0 = mg_pack_field_c(pack->data, MGP_BG_SM1, b, nb, npts);
            const double *SM1 = mg_pack_field_c(pack->data, MGP_BG_SM2, b, nb, npts);
            const double *SM2 = mg_pack_field_c(pack->data, MGP_BG_SM3, b, nb, npts);
            const double *f_V0 = mg_pack_field_c(pack->rhs, MGP_SOL_V1, b, nb, npts);
            const double *f_V1 = mg_pack_field_c(pack->rhs, MGP_SOL_V2, b, nb, npts);
            const double *f_V2 = mg_pack_field_c(pack->rhs, MGP_SOL_V3, b, nb, npts);

            for (int k = ghost + c2; k < ghost + N; k += 2)
                for (int j = ghost + c1; j < ghost + N; j += 2)
                    for (int i = ghost + c0; i < ghost + N; i += 2) {
                        int idx = k * sz + j * sy + i;
                        mg_smooth_4field_point(
                            psi, V0, V1, V2, psi_BL, A2,
                            R_tilde, SM0, SM1, SM2,
                            f_psi, f_V0, f_V1, f_V2,
                            idx, sx, sy, sz, inv_dx, dx2);
                    }
        } else {
            for (int k = ghost + c2; k < ghost + N; k += 2)
                for (int j = ghost + c1; j < ghost + N; j += 2)
                    for (int i = ghost + c0; i < ghost + N; i += 2) {
                        int idx = k * sz + j * sy + i;
                        mg_smooth_1field_point(
                            psi, psi_BL, A2, f_psi,
                            idx, sx, sy, sz, inv_dx, dx2);
                    }
        }
    }
}

/*
 * Same-level ghost exchange for solver — CPU implementation.
 * Copies ghost zones between same-level neighbors using pack metadata.
 */
void backend_mg_ghost_same_level_packed(meshblock_pack_t *pack, int slot, int n_sol)
{
    (void)slot;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;

    /* 26-neighbor directions */
    static const int dir_offset[26][3] = {
        {-1,-1,-1},{0,-1,-1},{1,-1,-1},{-1,0,-1},{0,0,-1},{1,0,-1},
        {-1,1,-1},{0,1,-1},{1,1,-1},{-1,-1,0},{0,-1,0},{1,-1,0},
        {-1,0,0},{1,0,0},{-1,1,0},{0,1,0},{1,1,0},{-1,-1,1},{0,-1,1},
        {1,-1,1},{-1,0,1},{0,0,1},{1,0,1},{-1,1,1},{0,1,1},{1,1,1}
    };

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < nb; b++) {
        for (int d = 0; d < 26; d++) {
            int nbr = pack->neighbor_table[b * 26 + d];
            if (nbr < 0) continue;

            int ox = dir_offset[d][0];
            int oy = dir_offset[d][1];
            int oz = dir_offset[d][2];

            /* Determine ghost slab range: receive zone from neighbor */
            int i_start, i_end, j_start, j_end, k_start, k_end;
            if (ox == -1) { i_start = 0;        i_end = ghost; }
            else if (ox == 1) { i_start = ghost + N; i_end = Nt; }
            else { i_start = ghost;  i_end = ghost + N; }

            if (oy == -1) { j_start = 0;        j_end = ghost; }
            else if (oy == 1) { j_start = ghost + N; j_end = Nt; }
            else { j_start = ghost;  j_end = ghost + N; }

            if (oz == -1) { k_start = 0;        k_end = ghost; }
            else if (oz == 1) { k_start = ghost + N; k_end = Nt; }
            else { k_start = ghost;  k_end = ghost + N; }

            /* Source offset: interior region of neighbor corresponding
             * to our ghost zone */
            int si_off = -ox * N;
            int sj_off = -oy * N;
            int sk_off = -oz * N;

            for (int s = 0; s < n_sol; s++) {
                double *dst = mg_pack_field(pack->data, s, b, nb, npts);
                const double *src = mg_pack_field_c(pack->data, s, nbr, nb, npts);

                for (int k = k_start; k < k_end; k++)
                    for (int j = j_start; j < j_end; j++)
                        for (int i = i_start; i < i_end; i++) {
                            int didx = k * Nt * Nt + j * Nt + i;
                            int sidx = (k + sk_off) * Nt * Nt
                                     + (j + sj_off) * Nt + (i + si_off);
                            dst[didx] = src[sidx];
                        }
            }
        }
    }
}

/*
 * Zero-Dirichlet BCs on domain-boundary ghost zones — CPU implementation.
 * Ref: relaxation_amr.c:714 amr_apply_bc_block
 */
void backend_mg_bc_packed(meshblock_pack_t *pack, int slot, int four_field)
{
    (void)slot;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;
    int n_sol = four_field ? 4 : 1;

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < nb; b++) {
        for (int face = 0; face < 6; face++) {
            if (!pack->on_boundary[b * 6 + face]) continue;

            for (int s = 0; s < n_sol; s++) {
                double *u = mg_pack_field(pack->data, s, b, nb, npts);

                for (int k = 0; k < Nt; k++)
                    for (int j = 0; j < Nt; j++)
                        for (int i = 0; i < Nt; i++) {
                            int is_ghost = 0;
                            switch (face) {
                            case 0: is_ghost = (i < ghost); break;
                            case 1: is_ghost = (i >= ghost + N); break;
                            case 2: is_ghost = (j < ghost); break;
                            case 3: is_ghost = (j >= ghost + N); break;
                            case 4: is_ghost = (k < ghost); break;
                            case 5: is_ghost = (k >= ghost + N); break;
                            }
                            if (is_ghost)
                                u[k * Nt * Nt + j * Nt + i] = 0.0;
                        }
            }
        }
    }
}

/*
 * Evaluate L(u) on interior points — CPU implementation.
 * Writes to accum buffer (slots 0..n_sol-1).
 */
void backend_mg_operator_packed(meshblock_pack_t *pack, int slot,
                                 int four_field)
{
    (void)slot;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < nb; b++) {
        double inv_dx = 1.0 / pack->dx_per_block[b];
        int sx = 1, sy = Nt, sz = Nt * Nt;

        const double *psi    = mg_pack_field_c(pack->data, MGP_SOL_PSI, b, nb, npts);
        const double *psi_BL = mg_pack_field_c(pack->data, MGP_BG_PSI_BL, b, nb, npts);
        const double *A2     = mg_pack_field_c(pack->data, MGP_BG_A2, b, nb, npts);
        double *L_psi = mg_pack_field(pack->accum, MGP_SOL_PSI, b, nb, npts);

        if (four_field) {
            const double *V0 = mg_pack_field_c(pack->data, MGP_SOL_V1, b, nb, npts);
            const double *V1 = mg_pack_field_c(pack->data, MGP_SOL_V2, b, nb, npts);
            const double *V2 = mg_pack_field_c(pack->data, MGP_SOL_V3, b, nb, npts);
            const double *R_tilde = mg_pack_field_c(pack->data, MGP_BG_RTILDE, b, nb, npts);
            const double *SM0 = mg_pack_field_c(pack->data, MGP_BG_SM1, b, nb, npts);
            const double *SM1 = mg_pack_field_c(pack->data, MGP_BG_SM2, b, nb, npts);
            const double *SM2 = mg_pack_field_c(pack->data, MGP_BG_SM3, b, nb, npts);
            double *L_V0 = mg_pack_field(pack->accum, MGP_SOL_V1, b, nb, npts);
            double *L_V1 = mg_pack_field(pack->accum, MGP_SOL_V2, b, nb, npts);
            double *L_V2 = mg_pack_field(pack->accum, MGP_SOL_V3, b, nb, npts);

            for (int k = ghost; k < ghost + N; k++)
                for (int j = ghost; j < ghost + N; j++)
                    for (int i = ghost; i < ghost + N; i++) {
                        int idx = k * sz + j * sy + i;
                        mg_operator_4field_point(
                            L_psi, L_V0, L_V1, L_V2,
                            psi, V0, V1, V2,
                            psi_BL, A2, R_tilde, SM0, SM1, SM2,
                            idx, sx, sy, sz, inv_dx);
                    }
        } else {
            for (int k = ghost; k < ghost + N; k++)
                for (int j = ghost; j < ghost + N; j++)
                    for (int i = ghost; i < ghost + N; i++) {
                        int idx = k * sz + j * sy + i;
                        mg_operator_1field_point(
                            L_psi, psi, psi_BL, A2,
                            idx, sx, sy, sz, inv_dx);
                    }
        }
    }
}

/*
 * Residual: accum[i] = rhs[i] - accum[i] on interior points.
 */
void backend_mg_residual_packed(meshblock_pack_t *pack, int slot,
                                 int four_field)
{
    (void)slot;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;
    int n_sol = four_field ? 4 : 1;

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < nb; b++) {
        int sy = Nt, sz = Nt * Nt;
        for (int k = ghost; k < ghost + N; k++)
            for (int j = ghost; j < ghost + N; j++)
                for (int i = ghost; i < ghost + N; i++) {
                    int idx = k * sz + j * sy + i;
                    for (int s = 0; s < n_sol; s++) {
                        double *a = mg_pack_field(pack->accum, s, b, nb, npts);
                        const double *r = mg_pack_field_c(pack->rhs, s, b, nb, npts);
                        a[idx] = r[idx] - a[idx];
                    }
                }
    }
}

/*
 * Save solution: scratch[s] = data[s].
 */
void backend_mg_save_packed(meshblock_pack_t *pack, int slot, int four_field)
{
    (void)slot;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int n_sol = four_field ? 4 : 1;

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < nb; b++) {
        for (int s = 0; s < n_sol; s++) {
            const double *src = mg_pack_field_c(pack->data, s, b, nb, npts);
            double *dst = mg_pack_field(pack->scratch, s, b, nb, npts);
            memcpy(dst, src, npts * sizeof(double));
        }
    }
}

/*
 * Tau correction: rhs[s] += accum[s] on interior points.
 */
void backend_mg_tau_packed(meshblock_pack_t *pack, int slot, int four_field)
{
    (void)slot;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;
    int n_sol = four_field ? 4 : 1;

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < nb; b++) {
        int sy = Nt, sz = Nt * Nt;
        for (int k = ghost; k < ghost + N; k++)
            for (int j = ghost; j < ghost + N; j++)
                for (int i = ghost; i < ghost + N; i++) {
                    int idx = k * sz + j * sy + i;
                    for (int s = 0; s < n_sol; s++) {
                        double *r = mg_pack_field(pack->rhs, s, b, nb, npts);
                        const double *a = mg_pack_field_c(pack->accum, s, b, nb, npts);
                        r[idx] += a[idx];
                    }
                }
    }
}

/*
 * Zero solution fields in data buffer.
 */
void backend_mg_zero_solution_packed(meshblock_pack_t *pack, int slot,
                                      int four_field)
{
    (void)slot;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int n_sol = four_field ? 4 : 1;

    for (int b = 0; b < nb; b++)
        for (int s = 0; s < n_sol; s++)
            memset(mg_pack_field(pack->data, s, b, nb, npts), 0,
                   npts * sizeof(double));
}

/*
 * Zero RHS fields in rhs buffer.
 */
void backend_mg_zero_rhs_packed(meshblock_pack_t *pack, int slot, int four_field)
{
    (void)slot;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int n_sol = four_field ? 4 : 1;

    for (int b = 0; b < nb; b++)
        for (int s = 0; s < n_sol; s++)
            memset(mg_pack_field(pack->rhs, s, b, nb, npts), 0,
                   npts * sizeof(double));
}

/*
 * Cross-slot restriction: fine → coarse — CPU implementation.
 * 8-child volume average (0.125 weight per fine cell).
 *
 * child_map: [n_parents * 8] — for parent p, child_map[p*8 + octant]
 *            is the fine-pack block index (-1 if absent).
 * parent_ids: [n_parents] — coarse-pack block index of each parent.
 *
 * Ref: relaxation_amr.c:1166 restrict_to_coarser_amr
 */
void backend_mg_restrict_packed(meshblock_pack_t *fine_pack, int fine_slot,
                                 meshblock_pack_t *coarse_pack, int coarse_slot,
                                 int four_field,
                                 const int *child_map, const int *parent_ids,
                                 int n_parents)
{
    (void)fine_slot; (void)coarse_slot;
    int n_sol = four_field ? 4 : 1;
    int f_nb = fine_pack->n_blocks;
    size_t f_npts = fine_pack->npts;
    int c_nb = coarse_pack->n_blocks;
    size_t c_npts = coarse_pack->npts;
    int ghost = fine_pack->ghost;
    int c_N = coarse_pack->N;
    int half_N = c_N / 2;
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
                        /* Restrict solution (data) */
                        const double *f_sol = mg_pack_field_c(fine_pack->data,
                            s, f_b, f_nb, f_npts);
                        double *c_sol = mg_pack_field(coarse_pack->data,
                            s, c_b, c_nb, c_npts);
                        /* Restrict residual (accum) */
                        const double *f_res = mg_pack_field_c(fine_pack->accum,
                            s, f_b, f_nb, f_npts);
                        double *c_rhs = mg_pack_field(coarse_pack->rhs,
                            s, c_b, c_nb, c_npts);

                        for (int pk = 0; pk < half_N; pk++)
                            for (int pj = 0; pj < half_N; pj++)
                                for (int pi = 0; pi < half_N; pi++) {
                                    int fi = ghost + 2 * pi;
                                    int fj = ghost + 2 * pj;
                                    int fk = ghost + 2 * pk;
                                    int f000 = fk * f_Nt * f_Nt + fj * f_Nt + fi;
                                    int sxf = 1;
                                    int syf = f_Nt;
                                    int szf = f_Nt * f_Nt;

                                    double sol = 0.125 * (
                                        f_sol[f000]
                                      + f_sol[f000 + sxf]
                                      + f_sol[f000 + syf]
                                      + f_sol[f000 + syf + sxf]
                                      + f_sol[f000 + szf]
                                      + f_sol[f000 + szf + sxf]
                                      + f_sol[f000 + syf + szf]
                                      + f_sol[f000 + syf + szf + sxf]);

                                    double res = 0.125 * (
                                        f_res[f000]
                                      + f_res[f000 + sxf]
                                      + f_res[f000 + syf]
                                      + f_res[f000 + syf + sxf]
                                      + f_res[f000 + szf]
                                      + f_res[f000 + szf + sxf]
                                      + f_res[f000 + syf + szf]
                                      + f_res[f000 + syf + szf + sxf]);

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
}

/*
 * Cross-slot prolongation (correction): coarse → fine — CPU implementation.
 * Computes correction = data - scratch on coarse, trilinear interp, adds
 * to fine data.
 *
 * Ref: relaxation_amr.c:1272 prolongate_correction_amr
 */
void backend_mg_prolong_add_packed(meshblock_pack_t *coarse_pack, int coarse_slot,
                                    meshblock_pack_t *fine_pack, int fine_slot,
                                    int four_field,
                                    const int *child_map, const int *parent_ids,
                                    int n_parents)
{
    (void)fine_slot; (void)coarse_slot;
    int n_sol = four_field ? 4 : 1;
    int f_nb = fine_pack->n_blocks;
    size_t f_npts = fine_pack->npts;
    int c_nb = coarse_pack->n_blocks;
    size_t c_npts = coarse_pack->npts;
    int ghost = coarse_pack->ghost;
    int c_N = coarse_pack->N;
    int half_N = c_N / 2;
    int f_N = fine_pack->N;
    int f_ghost = fine_pack->ghost;
    int f_Nt = fine_pack->Ntotal;
    int c_Nt = coarse_pack->Ntotal;

    for (int p = 0; p < n_parents; p++) {
        int c_b = parent_ids[p];

        /* Compute correction on coarse: scratch = data - scratch */
        for (int s = 0; s < n_sol; s++) {
            double *data = mg_pack_field(coarse_pack->data, s, c_b, c_nb, c_npts);
            double *scratch = mg_pack_field(coarse_pack->scratch, s, c_b, c_nb, c_npts);
            for (size_t idx = 0; idx < c_npts; idx++)
                scratch[idx] = data[idx] - scratch[idx];
        }

        /* Prolongate to each child */
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
                        const double *correction = mg_pack_field_c(
                            coarse_pack->scratch, s, c_b, c_nb, c_npts);
                        double *fine_sol = mg_pack_field(
                            fine_pack->data, s, f_b, f_nb, f_npts);

                        for (int fk = 0; fk < f_N; fk++) {
                            int Kc = ghost + c_off_k + fk / 2;
                            int ok = fk % 2;
                            int dk = ok ? 1 : -1;

                            for (int fj = 0; fj < f_N; fj++) {
                                int Jc = ghost + c_off_j + fj / 2;
                                int oj = fj % 2;
                                int dj = oj ? 1 : -1;

                                for (int fi = 0; fi < f_N; fi++) {
                                    int Ic = ghost + c_off_i + fi / 2;
                                    int oi = fi % 2;
                                    int di = oi ? 1 : -1;

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
                                                    * correction[CK * c_Nt * c_Nt
                                                                + CJ * c_Nt + CI];
                                            }
                                        }
                                    }
                                    fine_sol[(f_ghost + fk) * f_Nt * f_Nt
                                           + (f_ghost + fj) * f_Nt
                                           + (f_ghost + fi)] += val;
                                }
                            }
                        }
                    }
                }
    }
}

/*
 * FMG prolongation: coarse → fine (overwrite) — CPU implementation.
 * Trilinear interpolation from coarse parent to fine child.
 *
 * Ref: relaxation_amr.c:1568 composite_fmg
 */
void backend_mg_prolong_fmg_packed(meshblock_pack_t *coarse_pack, int coarse_slot,
                                    meshblock_pack_t *fine_pack, int fine_slot,
                                    int four_field,
                                    const int *child_map, const int *parent_ids,
                                    int n_parents)
{
    (void)fine_slot; (void)coarse_slot;
    int n_sol = four_field ? 4 : 1;
    int f_nb = fine_pack->n_blocks;
    size_t f_npts = fine_pack->npts;
    int c_nb = coarse_pack->n_blocks;
    size_t c_npts = coarse_pack->npts;
    int ghost = coarse_pack->ghost;
    int c_N = coarse_pack->N;
    int half_N = c_N / 2;
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
                        const double *csol = mg_pack_field_c(
                            coarse_pack->data, s, c_b, c_nb, c_npts);
                        double *fsol = mg_pack_field(
                            fine_pack->data, s, f_b, f_nb, f_npts);

                        for (int fk = 0; fk < f_N; fk++) {
                            int Kc = ghost + c_off_k + fk / 2;
                            int ok = fk % 2;
                            int dk = ok ? 1 : -1;

                            for (int fj = 0; fj < f_N; fj++) {
                                int Jc = ghost + c_off_j + fj / 2;
                                int oj = fj % 2;
                                int dj = oj ? 1 : -1;

                                for (int fi = 0; fi < f_N; fi++) {
                                    int Ic = ghost + c_off_i + fi / 2;
                                    int oi = fi % 2;
                                    int di = oi ? 1 : -1;

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
                                                    * csol[CK * c_Nt * c_Nt
                                                          + CJ * c_Nt + CI];
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
}

/*
 * L2 norm of residual: sqrt(sum((rhs - accum)^2) / count) — CPU.
 * Requires operator already evaluated in accum.
 */
double backend_mg_l2_norm_packed(meshblock_pack_t *pack, int slot,
                                  int four_field)
{
    (void)slot;
    int nb = pack->n_blocks;
    size_t npts = pack->npts;
    int N = pack->N, ghost = pack->ghost, Nt = pack->Ntotal;
    int n_sol = four_field ? 4 : 1;

    double sum = 0.0;
    int count = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:sum,count)
    for (int b = 0; b < nb; b++) {
        int sy = Nt, sz = Nt * Nt;
        for (int k = ghost; k < ghost + N; k++)
            for (int j = ghost; j < ghost + N; j++)
                for (int i = ghost; i < ghost + N; i++) {
                    int idx = k * sz + j * sy + i;
                    for (int s = 0; s < n_sol; s++) {
                        const double *r = mg_pack_field_c(pack->rhs, s, b, nb, npts);
                        const double *a = mg_pack_field_c(pack->accum, s, b, nb, npts);
                        double d = r[idx] - a[idx];
                        sum += d * d;
                    }
                    count++;
                }
    }

    if (count == 0) return 0.0;
    return sqrt(sum / ((double)n_sol * count));
}
