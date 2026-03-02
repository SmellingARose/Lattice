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
