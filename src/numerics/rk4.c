/*
 * Lattice — 3D Numerical Relativity
 * Time integrators: Classic RK4 and Carpenter-Kennedy 2N low-storage RK4.
 *
 * Classic RK4 (4 stages, 4 blocks):
 *   scratch = fields (backup)
 *   Stage 1: RHS(rhs, fields); accum  = (dt/6)*rhs; fields = scratch + (dt/2)*rhs; BCs
 *   Stage 2: RHS(rhs, fields); accum += (dt/3)*rhs; fields = scratch + (dt/2)*rhs; BCs
 *   Stage 3: RHS(rhs, fields); accum += (dt/3)*rhs; fields = scratch + dt*rhs; BCs
 *   Stage 4: RHS(rhs, fields); accum += (dt/6)*rhs
 *   fields = scratch + accum
 *   Enforce: det(gambar)=1, tr(Abar)=0
 *
 * CK45 — Carpenter-Kennedy 2N low-storage RK4 (5 stages, 3 blocks):
 *   Ref: Carpenter & Kennedy, NASA TM-109112 (1994), Solution 3.
 *   dU = 0
 *   For s = 0..4:
 *     F = RHS(U); BCs(F)
 *     dU = A[s]*dU + dt*F
 *     U += B[s]*dU
 *   Enforce: det(gambar)=1, tr(Abar)=0
 *
 *   Uses scratch[] as dU register. accum[] not allocated.
 *   5 RHS evaluations per step (25% more compute, 25% less memory).
 */

#include "rk4.h"
#include "../amr/mesh.h"
#include "../amr/ghost_exchange.h"
#include "../amr/meshblock_pack.h"
#include "../amr/restriction.h"
#include "../boundary/sommerfeld.h"
#include "../core/fields.h"
#include "../geometry/tensor_utils.h"
#include "../backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Shared helpers
 * ======================================================================== */

/* Copy all field arrays: dst = src.
 * Backing blocks are contiguous (fields[f] = block + f*n), so use single memcpy. */
static void copy_fields(double **dst, const double *const *src, size_t n)
{
    memcpy(dst[0], src[0], (size_t)NUM_FIELDS * n * sizeof(double));
}

/* Linear combination: dst[i] = a[i] + coeff * b[i] for all fields.
 * Flattened to single OMP region over all fields (eliminates 31 fork/joins). */
static void axpy_fields(double **dst, const double *const *a,
                        const double *const *b, double coeff, size_t n)
{
    double *restrict d = dst[0];
    const double *restrict ap = a[0];
    const double *restrict bp = b[0];
    size_t total = (size_t)NUM_FIELDS * n;
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < total; i++)
        d[i] = ap[i] + coeff * bp[i];
}

/* Accumulate: accum[i] += coeff * rhs[i].
 * Flattened to single OMP region over all fields. */
static void accum_add(double **accum, const double *const *rhs_arr,
                      double coeff, size_t n)
{
    double *restrict acc = accum[0];
    const double *restrict rp = rhs_arr[0];
    size_t total = (size_t)NUM_FIELDS * n;
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < total; i++)
        acc[i] += coeff * rp[i];
}

/* Apply: fields[i] += accum[i].
 * Flattened to single OMP region over all fields. */
static void apply_accum(double **fields, const double *const *accum, size_t n)
{
    double *restrict fp = fields[0];
    const double *restrict ap = accum[0];
    size_t total = (size_t)NUM_FIELDS * n;
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < total; i++)
        fp[i] += ap[i];
}

/* Zero all field arrays.
 * Backing blocks are contiguous, so use single memset. */
static void zero_fields(double **arr, size_t n)
{
    memset(arr[0], 0, (size_t)NUM_FIELDS * n * sizeof(double));
}

/*
 * Enforce algebraic constraints after a full RK step.
 *   1. det(gambar) = 1: rescale h_ij so det(h) = 1
 *   2. tr(Abar) = 0: remove trace from A_ij
 *
 * Ref: GRChombo CCZ4/TraceARemoval.hpp, CCZ4/PositiveChiAndAlpha.hpp
 */
static void enforce_algebraic(grid_t *g)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < g->Ntotal; k++) {
        for (int j = 0; j < g->Ntotal; j++) {
            for (int i = 0; i < g->Ntotal; i++) {
                int idx = IDX(g, i, j, k);

                /* Load h_ij */
                double h_loc[3][3];
                h_loc[0][0] = g->fields[FIELD_H11][idx];
                h_loc[0][1] = g->fields[FIELD_H12][idx];
                h_loc[0][2] = g->fields[FIELD_H13][idx];
                h_loc[1][0] = h_loc[0][1];
                h_loc[1][1] = g->fields[FIELD_H22][idx];
                h_loc[1][2] = g->fields[FIELD_H23][idx];
                h_loc[2][0] = h_loc[0][2];
                h_loc[2][1] = h_loc[1][2];
                h_loc[2][2] = g->fields[FIELD_H33][idx];

                /* Enforce det(h) = 1 by rescaling */
                double det = compute_det_sym(h_loc);
                double scale = fast_inv_cbrt(det);
                FOR2(a, b) h_loc[a][b] *= scale;

                g->fields[FIELD_H11][idx] = h_loc[0][0];
                g->fields[FIELD_H12][idx] = h_loc[0][1];
                g->fields[FIELD_H13][idx] = h_loc[0][2];
                g->fields[FIELD_H22][idx] = h_loc[1][1];
                g->fields[FIELD_H23][idx] = h_loc[1][2];
                g->fields[FIELD_H33][idx] = h_loc[2][2];

                /* Enforce tr(A) = 0 */
                double h_UU[3][3];
                compute_inverse_sym(h_loc, h_UU);

                double A_loc[3][3];
                A_loc[0][0] = g->fields[FIELD_A11][idx];
                A_loc[0][1] = g->fields[FIELD_A12][idx];
                A_loc[0][2] = g->fields[FIELD_A13][idx];
                A_loc[1][0] = A_loc[0][1];
                A_loc[1][1] = g->fields[FIELD_A22][idx];
                A_loc[1][2] = g->fields[FIELD_A23][idx];
                A_loc[2][0] = A_loc[0][2];
                A_loc[2][1] = A_loc[1][2];
                A_loc[2][2] = g->fields[FIELD_A33][idx];

                make_trace_free(A_loc, h_loc, h_UU);

                g->fields[FIELD_A11][idx] = A_loc[0][0];
                g->fields[FIELD_A12][idx] = A_loc[0][1];
                g->fields[FIELD_A13][idx] = A_loc[0][2];
                g->fields[FIELD_A22][idx] = A_loc[1][1];
                g->fields[FIELD_A23][idx] = A_loc[1][2];
                g->fields[FIELD_A33][idx] = A_loc[2][2];

                /* Ensure chi > 0 */
                if (g->fields[FIELD_CHI][idx] < 1.0e-12)
                    g->fields[FIELD_CHI][idx] = 1.0e-12;
                /* Ensure lapse > 0 */
                if (g->fields[FIELD_LAPSE][idx] < 1.0e-12)
                    g->fields[FIELD_LAPSE][idx] = 1.0e-12;
            }
        }
    }
}

/* ========================================================================
 * Classic RK4 (4 stages, 4 memory blocks)
 * ======================================================================== */

static void classic_rk4_step(grid_t *g, const sim_params_t *p,
                              rk4_rhs_func_t rhs_func, rk4_bc_func_t bc_func,
                              double dt)
{
    size_t n = g->npoints;

    /* Save initial state */
    copy_fields(g->scratch, (const double *const *)g->fields, n);
    zero_fields(g->accum, n);

    /* --- Stage 1: rhs from fields --- */
    backend_compute_rhs(g->rhs, (const double *const *)g->fields, g, p, rhs_func);
    bc_func(g->rhs, (const double *const *)g->fields, g);
    accum_add(g->accum, (const double *const *)g->rhs, dt / 6.0, n);
    axpy_fields(g->fields, (const double *const *)g->scratch,
                (const double *const *)g->rhs, dt / 2.0, n);

    /* --- Stage 2: rhs from fields (= scratch + dt/2 * k1) --- */
    backend_compute_rhs(g->rhs, (const double *const *)g->fields, g, p, rhs_func);
    bc_func(g->rhs, (const double *const *)g->fields, g);
    accum_add(g->accum, (const double *const *)g->rhs, dt / 3.0, n);
    axpy_fields(g->fields, (const double *const *)g->scratch,
                (const double *const *)g->rhs, dt / 2.0, n);

    /* --- Stage 3: rhs from fields (= scratch + dt/2 * k2) --- */
    backend_compute_rhs(g->rhs, (const double *const *)g->fields, g, p, rhs_func);
    bc_func(g->rhs, (const double *const *)g->fields, g);
    accum_add(g->accum, (const double *const *)g->rhs, dt / 3.0, n);
    axpy_fields(g->fields, (const double *const *)g->scratch,
                (const double *const *)g->rhs, dt, n);

    /* --- Stage 4: rhs from fields (= scratch + dt * k3) --- */
    backend_compute_rhs(g->rhs, (const double *const *)g->fields, g, p, rhs_func);
    bc_func(g->rhs, (const double *const *)g->fields, g);
    accum_add(g->accum, (const double *const *)g->rhs, dt / 6.0, n);

    /* Apply: fields = scratch + accum */
    copy_fields(g->fields, (const double *const *)g->scratch, n);
    apply_accum(g->fields, (const double *const *)g->accum, n);

    enforce_algebraic(g);
}

/* ========================================================================
 * CK45 — Carpenter-Kennedy 2N low-storage RK4
 * Ref: Carpenter & Kennedy, NASA TM-109112 (1994), Solution 3.
 * 5 stages, 3 memory blocks (fields=U, scratch=dU, rhs=F).
 * ======================================================================== */

/* CK45 coefficients (exact rational fractions as doubles) */
static const double CK_A[5] = {
    0.0,
    -567301805773.0  / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0
};

static const double CK_B[5] = {
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0
};

/* Fused CK45 update: dU = A*dU + dt*F; U += B*dU.
 * Flattened to single OMP region over all fields (eliminates 31 fork/joins). */
static void ck45_update(double **U, double **dU, const double *const *F,
                        double A_s, double B_s, double dt, size_t n)
{
    double *restrict u = U[0];
    double *restrict du = dU[0];
    const double *restrict fp = F[0];
    size_t total = (size_t)NUM_FIELDS * n;
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < total; i++) {
        du[i] = A_s * du[i] + dt * fp[i];
        u[i] += B_s * du[i];
    }
}

static void ck45_step(grid_t *g, const sim_params_t *p,
                      rk4_rhs_func_t rhs_func, rk4_bc_func_t bc_func,
                      double dt)
{
    size_t n = g->npoints;

    /* Zero dU (stored in scratch) */
    zero_fields(g->scratch, n);

    /* 5 stages */
    for (int s = 0; s < 5; s++) {
        backend_compute_rhs(g->rhs, (const double *const *)g->fields, g, p, rhs_func);
        bc_func(g->rhs, (const double *const *)g->fields, g);
        ck45_update(g->fields, g->scratch, (const double *const *)g->rhs,
                    CK_A[s], CK_B[s], dt, n);
    }

    enforce_algebraic(g);
}

/* ========================================================================
 * Public interface — dispatches on p->rk_method
 * ======================================================================== */

void rk4_step(grid_t *g, const sim_params_t *p,
              rk4_rhs_func_t rhs_func, rk4_bc_func_t bc_func,
              double dt)
{
    if (p->rk_method == RK_CK45)
        ck45_step(g, p, rhs_func, bc_func, dt);
    else
        classic_rk4_step(g, p, rhs_func, bc_func, dt);
}

/* ========================================================================
 * Mesh-level helpers (shared by per-block and packed paths)
 * ======================================================================== */

/*
 * Helper: perform ghost exchange, choosing multilevel variant when
 * the mesh has refinement (max_level > 0).
 */
static void mesh_ghost_exchange(mesh_t *m)
{
    if (m->max_level > 0)
        ghost_exchange_multilevel(m);
    else
        ghost_exchange(m);
}

/*
 * Restrict fine leaf data into non-leaf parents after a full RK step.
 * Keeps parent data synchronized for cross-level ghost exchange.
 */
static void restrict_level_to_parents(mesh_t *m, int level)
{
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || b->loc.level != level - 1 || b->is_leaf) continue;

        /* b is non-leaf parent at level-1: restrict children */
        grid_t *pg = b->grid;
        const int ghost_w = pg->ghost;
        const int N = pg->N;
        const int half_N = N / 2;

        for (int cz = 0; cz < 2; cz++) {
            for (int cy = 0; cy < 2; cy++) {
                for (int cx = 0; cx < 2; cx++) {
                    int octant = cx + (cy << 1) + (cz << 2);
                    int cid = b->child_ids[octant];
                    if (cid < 0 || !m->blocks[cid]) continue;

                    const block_t *child = m->blocks[cid];
                    const grid_t *cg = child->grid;

                    int p_off_i = cx * half_N;
                    int p_off_j = cy * half_N;
                    int p_off_k = cz * half_N;

                    for (int f = 0; f < NUM_FIELDS; f++) {
                        for (int pk = 0; pk < half_N; pk++) {
                            for (int pj = 0; pj < half_N; pj++) {
                                for (int pi = 0; pi < half_N; pi++) {
                                    int fi_base = cg->ghost + 2 * pi;
                                    int fj_base = cg->ghost + 2 * pj;
                                    int fk_base = cg->ghost + 2 * pk;
                                    int pii = ghost_w + p_off_i + pi;
                                    int pjj = ghost_w + p_off_j + pj;
                                    int pkk = ghost_w + p_off_k + pk;
                                    pg->fields[f][IDX(pg, pii, pjj, pkk)] =
                                        restrict_cell(cg->fields[f], cg,
                                                      fi_base, fj_base, fk_base);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 * Compute RHS + apply Sommerfeld BCs for all blocks in the mesh.
 * Ghost exchange must have been called before this.
 */
static void mesh_compute_rhs_and_bc(mesh_t *m, const sim_params_t *p,
                                    rk4_rhs_func_t rhs_func)
{
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;

        grid_t *g = b->grid;
        backend_compute_rhs(g->rhs, (const double *const *)g->fields,
                            g, p, rhs_func);
        apply_sommerfeld_block(g->rhs, (const double *const *)g->fields, b);
    }
}

/* Enforce algebraic constraints on all leaf blocks */
static void mesh_enforce_algebraic(mesh_t *m)
{
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        enforce_algebraic(b->grid);
    }
}

/* Synchronize non-leaf parent data after a full RK step */
static void mesh_restrict_to_parents(mesh_t *m)
{
    if (m->max_level == 0) return;
    for (int L = m->max_level; L >= 1; L--)
        restrict_level_to_parents(m, L);
}

/* ========================================================================
 * Per-block mesh steppers (legacy — one kernel launch per block per stage)
 *
 * Kept for validation: the packed stepper should produce identical results.
 * Used by rk4_step_mesh_perblock() for debug/comparison.
 * ======================================================================== */

/*
 * Classic RK4 per-block: loops over leaf blocks, one kernel per block.
 */
static void classic_rk4_step_mesh_perblock(mesh_t *m, const sim_params_t *p,
                                            rk4_rhs_func_t rhs_func, double dt)
{
    /* Save initial state and zero accumulators */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        grid_t *g = b->grid;
        size_t n = g->npoints;
        copy_fields(g->scratch, (const double *const *)g->fields, n);
        zero_fields(g->accum, n);
    }

    /* Stage 1 */
    mesh_ghost_exchange(m);
    mesh_compute_rhs_and_bc(m, p, rhs_func);
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        grid_t *g = b->grid;
        size_t n = g->npoints;
        accum_add(g->accum, (const double *const *)g->rhs, dt / 6.0, n);
        axpy_fields(g->fields, (const double *const *)g->scratch,
                    (const double *const *)g->rhs, dt / 2.0, n);
    }

    /* Stage 2 */
    mesh_ghost_exchange(m);
    mesh_compute_rhs_and_bc(m, p, rhs_func);
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        grid_t *g = b->grid;
        size_t n = g->npoints;
        accum_add(g->accum, (const double *const *)g->rhs, dt / 3.0, n);
        axpy_fields(g->fields, (const double *const *)g->scratch,
                    (const double *const *)g->rhs, dt / 2.0, n);
    }

    /* Stage 3 */
    mesh_ghost_exchange(m);
    mesh_compute_rhs_and_bc(m, p, rhs_func);
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        grid_t *g = b->grid;
        size_t n = g->npoints;
        accum_add(g->accum, (const double *const *)g->rhs, dt / 3.0, n);
        axpy_fields(g->fields, (const double *const *)g->scratch,
                    (const double *const *)g->rhs, dt, n);
    }

    /* Stage 4 */
    mesh_ghost_exchange(m);
    mesh_compute_rhs_and_bc(m, p, rhs_func);
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        grid_t *g = b->grid;
        size_t n = g->npoints;
        accum_add(g->accum, (const double *const *)g->rhs, dt / 6.0, n);
        copy_fields(g->fields, (const double *const *)g->scratch, n);
        apply_accum(g->fields, (const double *const *)g->accum, n);
    }

    mesh_enforce_algebraic(m);
    mesh_restrict_to_parents(m);
}

/*
 * CK45 per-block: loops over leaf blocks, one kernel per block.
 */
static void ck45_step_mesh_perblock(mesh_t *m, const sim_params_t *p,
                                     rk4_rhs_func_t rhs_func, double dt)
{
    /* Zero dU (stored in scratch) for all blocks */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        zero_fields(b->grid->scratch, b->grid->npoints);
    }

    /* 5 CK45 stages */
    for (int s = 0; s < 5; s++) {
        mesh_ghost_exchange(m);
        mesh_compute_rhs_and_bc(m, p, rhs_func);

        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *b = m->blocks[bid];
            if (!b || !b->is_leaf) continue;
            grid_t *g = b->grid;
            ck45_update(g->fields, g->scratch,
                        (const double *const *)g->rhs,
                        CK_A[s], CK_B[s], dt, g->npoints);
        }
    }

    mesh_enforce_algebraic(m);
    mesh_restrict_to_parents(m);
}

/* ========================================================================
 * Packed batch mesh steppers (production path)
 *
 * All leaf blocks packed into a single meshblock_pack_t. One kernel
 * launch per operation (RHS, Sommerfeld, update) covers all blocks.
 *
 * Ghost exchange: Commit 1 uses CPU fallback (unpack → exchange → repack).
 * Commit 2 replaces this with device-side ghost exchange kernels.
 *
 * Ref: AthenaK task_list/ pattern (meshblock_pack batched kernels)
 * ======================================================================== */

/*
 * Build a meshblock_pack_t from the mesh's leaf blocks.
 *
 * Collects all leaf block IDs, creates the pack with the appropriate
 * RK method, loads field data and metadata, and builds neighbor tables.
 *
 * The returned pack is ready for backend_map_pack + kernel launches.
 * Caller must free with meshblock_pack_free.
 */
static meshblock_pack_t *mesh_build_leaf_pack(mesh_t *m,
                                               rk_method_t rk_method)
{
    /* Count leaf blocks and collect their IDs */
    int n_leaves = mesh_num_leaves(m);
    int *ids = malloc(n_leaves * sizeof(int));
    if (!ids) {
        fprintf(stderr, "mesh_build_leaf_pack: malloc failed\n");
        exit(1);
    }

    int idx = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (b && b->is_leaf)
            ids[idx++] = bid;
    }

    /* All blocks share the same N_block → same npts */
    size_t npts = m->blocks[ids[0]]->grid->npoints;

    /* Create pack: allocates field buffers + metadata arrays.
     * level = -1 (mixed levels, normal for AMR meshes). */
    meshblock_pack_t *pack = meshblock_pack_create(
        n_leaves, npts, ids, -1, rk_method);

    /* Load field data from blocks into pack's contiguous buffers */
    meshblock_pack_load(pack, m->blocks);

    /* Fill per-block metadata (origins, dx, boundary flags, levels).
     * Also allocates coarse_data if any blocks are refined. */
    meshblock_pack_load_meta(pack, m->blocks);

    /* Build pack-local neighbor table (mesh block ID → pack index).
     * Required by Commit 2's device ghost exchange; also useful for
     * diagnostics and debugging. */
    meshblock_pack_build_neighbors(pack, m->blocks);

    /* Load coarse_buf data for multilevel ghost exchange */
    if (m->max_level > 0)
        meshblock_pack_load_coarse(pack, m->blocks);

    free(ids);
    return pack;
}

/* Commit 1 fallback (packed_ghost_exchange_fallback) removed.
 * Ghost exchange now runs directly on pack buffers via
 * backend_ghost_exchange_packed() — no unpack/repack needed. */

/*
 * CK45 packed mesh stepper: all leaf blocks in one pack, one kernel per op.
 *
 * Algorithm (same CK45 as per-block, different execution model):
 *   dU = 0   (zero scratch)
 *   For s = 0..4:
 *     Ghost exchange (Commit 2: direct on pack buffers)
 *     F = RHS(U)       (batched: all blocks in one kernel)
 *     Sommerfeld(F)    (batched: boundary BCs in one kernel)
 *     dU = A[s]*dU + dt*F;  U += B[s]*dU  (fused update, one kernel)
 *   Enforce algebraic constraints (CPU, once per step)
 *   Restrict to parents (CPU, once per step)
 *
 * Ref: Carpenter & Kennedy, NASA TM-109112 (1994), Solution 3.
 * Ref: AthenaK task_list/ batched kernel pattern.
 */
static void ck45_step_mesh_packed(mesh_t *m, const sim_params_t *p, double dt)
{
    /* Build pack from all leaf blocks */
    meshblock_pack_t *pack = mesh_build_leaf_pack(m, RK_CK45);

    /* Map to GPU (no-op on CPU) */
    backend_map_pack(pack, p);

    /* Zero dU register (stored in scratch buffer) */
    backend_zero_packed(pack, PACK_BUF_SCRATCH);

    /* 5 CK45 stages */
    for (int s = 0; s < 5; s++) {
        /* Ghost exchange: all 5 phases on pack buffers (Commit 2) */
        backend_ghost_exchange_packed(pack);

        /* Batched RHS + Sommerfeld BCs: one kernel each for all blocks */
        backend_compute_rhs_packed(pack, p);
        backend_sommerfeld_packed(pack, p);

        /* Fused CK45 update: dU = A*dU + dt*F; U += B*dU */
        backend_update_ck45_packed(pack, CK_A[s], CK_B[s], dt);
    }

    /* Unmap from GPU (syncs data back to host; no-op on CPU) */
    backend_unmap_pack(pack);

    /* Store final state from pack back to individual blocks */
    meshblock_pack_store(pack, m->blocks);

    /* Enforce algebraic constraints: det(gambar)=1, tr(Abar)=0.
     * Done on CPU per block, once per step (not per stage). */
    mesh_enforce_algebraic(m);

    /* Restrict leaf data into non-leaf parents for next step's
     * cross-level ghost exchange. */
    mesh_restrict_to_parents(m);

    meshblock_pack_free(pack);
}

/*
 * Classic RK4 packed mesh stepper: all leaf blocks in one pack.
 *
 * Algorithm (same classic RK4 as per-block, different execution model):
 *   scratch = data (backup U^0)
 *   accum = 0
 *   Stage 1: F1 = RHS(U); accum += dt/6 * F1; data = scratch + dt/2 * F1
 *   Stage 2: F2 = RHS(U); accum += dt/3 * F2; data = scratch + dt/2 * F2
 *   Stage 3: F3 = RHS(U); accum += dt/3 * F3; data = scratch + dt * F3
 *   Stage 4: F4 = RHS(U); accum += dt/6 * F4; data = scratch; data += accum
 *   Enforce algebraic; restrict to parents.
 *
 * Uses 4 memory blocks per pack (data, rhs, scratch, accum).
 */
static void classic_rk4_step_mesh_packed(mesh_t *m, const sim_params_t *p,
                                          double dt)
{
    /* Build pack with accum buffer for classic RK4 */
    meshblock_pack_t *pack = mesh_build_leaf_pack(m, RK_CLASSIC);

    /* Map to GPU (no-op on CPU) */
    backend_map_pack(pack, p);

    /* Save initial state: scratch = data (backup U^0) */
    backend_copy_packed(pack, PACK_BUF_SCRATCH, PACK_BUF_DATA);
    /* Zero accumulator */
    backend_zero_packed(pack, PACK_BUF_ACCUM);

    /* Stage 1: F1 = RHS(U^0) */
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_accum_add_packed(pack, 1.0/6.0, dt);   /* accum += dt/6 * F1 */
    backend_axpy_packed(pack, 0.5, dt);             /* data = scratch + dt/2*rhs */

    /* Stage 2: F2 = RHS(U^0 + dt/2 * F1) */
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_accum_add_packed(pack, 1.0/3.0, dt);   /* accum += dt/3 * F2 */
    backend_axpy_packed(pack, 0.5, dt);             /* data = scratch + dt/2*rhs */

    /* Stage 3: F3 = RHS(U^0 + dt/2 * F2) */
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_accum_add_packed(pack, 1.0/3.0, dt);   /* accum += dt/3 * F3 */
    backend_axpy_packed(pack, 1.0, dt);             /* data = scratch + dt*rhs */

    /* Stage 4: F4 = RHS(U^0 + dt * F3) */
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_accum_add_packed(pack, 1.0/6.0, dt);   /* accum += dt/6 * F4 */
    backend_copy_packed(pack, PACK_BUF_DATA, PACK_BUF_SCRATCH); /* data = U^0 */
    backend_apply_accum_packed(pack);               /* data += accum */

    /* Unmap from GPU (no-op on CPU) */
    backend_unmap_pack(pack);

    /* Store final state back to blocks */
    meshblock_pack_store(pack, m->blocks);

    /* Post-step: algebraic constraints + parent restriction */
    mesh_enforce_algebraic(m);
    mesh_restrict_to_parents(m);

    meshblock_pack_free(pack);
}

/* ========================================================================
 * Berger-Oliger subcycling (per-level packing)
 *
 * Each AMR level advances at dt_L = dt_0 / 2^L. Fine levels take 2x
 * more sub-steps per coarse step. Cross-level ghost zones are filled
 * via temporal interpolation from the coarser level's saved old state.
 *
 * Ref: Berger & Oliger (1984), JCP 53:484.
 * Ref: Athena++ src/mesh/mesh.cpp Mesh::Step().
 * Ref: GRChombo GRAMRLevel::advance() + Chombo AMR::timeStep().
 * ======================================================================== */

/*
 * Save k1 (Stage 1 RHS) from pack into blocks' rhs_old for quartic
 * temporal interpolation. Called after Stage 1 RHS + Sommerfeld,
 * before Stage 2 overwrites the pack's RHS buffer.
 *
 * Only copies for blocks that have rhs_old allocated (level < max_level).
 * Ref: Quartic temporal prolongation design (tools/compute_amr_weights.py).
 */
static void save_k1_from_pack(const meshblock_pack_t *pack, block_t **blocks)
{
    for (int b = 0; b < pack->n_blocks; b++) {
        block_t *blk = blocks[pack->block_ids[b]];
        if (!blk->rhs_old_block) continue;

        size_t npts = pack->npts;
        for (int f = 0; f < NUM_FIELDS; f++) {
            size_t src_off = (size_t)f * pack->n_blocks * npts
                           + (size_t)b * npts;
            memcpy(blk->rhs_old[f], pack->rhs + src_off,
                   npts * sizeof(double));
        }
    }
}

/*
 * Build a meshblock_pack_t containing only leaf blocks at the specified level.
 * Same as mesh_build_leaf_pack but filtered by level.
 *
 * Returns NULL if no leaves exist at this level (caller must handle).
 *
 * Ref: AthenaK MeshBlockPack per-level construction.
 */
static meshblock_pack_t *mesh_build_level_pack(mesh_t *m, int level,
                                                rk_method_t rk_method)
{
    /* Count leaves at this level */
    int n_leaves = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (b && b->is_leaf && b->loc.level == level)
            n_leaves++;
    }

    if (n_leaves == 0) return NULL;

    int *ids = malloc(n_leaves * sizeof(int));
    if (!ids) {
        fprintf(stderr, "mesh_build_level_pack: malloc failed\n");
        exit(1);
    }

    int idx = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (b && b->is_leaf && b->loc.level == level)
            ids[idx++] = bid;
    }

    size_t npts = m->blocks[ids[0]]->grid->npoints;

    meshblock_pack_t *pack = meshblock_pack_create(
        n_leaves, npts, ids, level, rk_method);

    meshblock_pack_load(pack, m->blocks);
    meshblock_pack_load_meta(pack, m->blocks);
    meshblock_pack_build_neighbors(pack, m->blocks);

    /* Per-level packs at level > 0 may have coarse_bufs, but cross-level
     * ghost fill is done BEFORE packing (via ghost_fill_from_coarser),
     * so we don't need coarse_data in the pack. The pack's ghost exchange
     * only handles same-level neighbors. For blocks that do have coarse_bufs,
     * load coarse data so pack phases 2-4 can run if n_refined > 0. */
    if (pack->n_refined > 0)
        meshblock_pack_load_coarse(pack, m->blocks);

    free(ids);
    return pack;
}

/*
 * Save fields_old for all leaf blocks at the given level.
 * Called before advancing a level so we have the pre-step state
 * for temporal interpolation during finer-level subcycles.
 *
 * Ref: Chombo AMRLevel::m_old_data save pattern.
 */
static void mesh_save_old_level(mesh_t *m, int level)
{
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf || b->loc.level != level) continue;

        if (!b->fields_old_block)
            block_alloc_fields_old(b);
        block_save_old(b);
    }
}

/*
 * Advance all leaf blocks at `level` by one step of size dt.
 * Fills cross-level ghosts (time-interpolated from coarser),
 * then does same-level ghost exchange within a per-level pack,
 * then RHS + Sommerfeld + update.
 *
 * frac: temporal interpolation fraction for cross-level ghosts.
 *   frac = (current_time - coarse_old_time) / dt_coarse
 *   For level 0: frac is unused (no coarser level).
 *
 * Ref: Berger & Oliger (1984), Athena++ Mesh::Step().
 */
static void step_level(mesh_t *m, const sim_params_t *p,
                        int level, double dt_level, double frac)
{
    /* Cross-level ghost fill (time-interpolated from coarser level).
     * Done on the mesh BEFORE building the per-level pack. */
    if (level > 0)
        ghost_fill_from_coarser(m, level, frac);

    /* Build per-level pack (only leaves at this level) */
    meshblock_pack_t *pack = mesh_build_level_pack(m, level, p->rk_method);
    if (!pack) return;  /* no blocks at this level */

    backend_map_pack(pack, p);

    /* CK45 or classic RK4 — same kernel sequence as global packed stepper,
     * but pack contains only one level's blocks */
    if (p->rk_method == RK_CK45) {
        backend_zero_packed(pack, PACK_BUF_SCRATCH);
        for (int s = 0; s < 5; s++) {
            backend_ghost_exchange_packed(pack);
            backend_compute_rhs_packed(pack, p);
            backend_sommerfeld_packed(pack, p);
            /* Save k1 (Stage 0 RHS) for quartic temporal interpolation.
             * Must happen before Stage 1 overwrites the pack's RHS buffer.
             * Only saves for blocks with rhs_old allocated (level < max_level). */
            if (s == 0) save_k1_from_pack(pack, m->blocks);
            backend_update_ck45_packed(pack, CK_A[s], CK_B[s], dt_level);
        }
    } else {
        /* Classic RK4: 4 stages */
        backend_copy_packed(pack, PACK_BUF_SCRATCH, PACK_BUF_DATA);
        backend_zero_packed(pack, PACK_BUF_ACCUM);

        /* Stage 1 */
        backend_ghost_exchange_packed(pack);
        backend_compute_rhs_packed(pack, p);
        backend_sommerfeld_packed(pack, p);
        /* Save k1 (beginning-of-step RHS) for quartic temporal interpolation.
         * Must happen before Stage 2 overwrites the pack's RHS buffer.
         * Only saves for blocks with rhs_old allocated (level < max_level). */
        save_k1_from_pack(pack, m->blocks);
        backend_accum_add_packed(pack, 1.0/6.0, dt_level);
        backend_axpy_packed(pack, 0.5, dt_level);

        /* Stage 2 */
        backend_ghost_exchange_packed(pack);
        backend_compute_rhs_packed(pack, p);
        backend_sommerfeld_packed(pack, p);
        backend_accum_add_packed(pack, 1.0/3.0, dt_level);
        backend_axpy_packed(pack, 0.5, dt_level);

        /* Stage 3 */
        backend_ghost_exchange_packed(pack);
        backend_compute_rhs_packed(pack, p);
        backend_sommerfeld_packed(pack, p);
        backend_accum_add_packed(pack, 1.0/3.0, dt_level);
        backend_axpy_packed(pack, 1.0, dt_level);

        /* Stage 4 */
        backend_ghost_exchange_packed(pack);
        backend_compute_rhs_packed(pack, p);
        backend_sommerfeld_packed(pack, p);
        backend_accum_add_packed(pack, 1.0/6.0, dt_level);
        backend_copy_packed(pack, PACK_BUF_DATA, PACK_BUF_SCRATCH);
        backend_apply_accum_packed(pack);
    }

    backend_unmap_pack(pack);
    meshblock_pack_store(pack, m->blocks);
    meshblock_pack_free(pack);
}

/*
 * Berger-Oliger recursive subcycling.
 * Advances level L by dt_L, then recursively subcycles finer levels.
 *
 * Algorithm:
 *   1. Save fields_old at level L (for temporal interpolation by finer levels)
 *   2. Advance level L by dt_L
 *   3. If L < max_level:
 *        subcycle(L+1, dt_L/2, t_start)           — first half
 *        subcycle(L+1, dt_L/2, t_start + dt_L/2)  — second half
 *   4. Restrict level L+1 → level L parents
 *   5. Enforce algebraic constraints at level L
 *
 * Ref: Berger & Oliger (1984), JCP 53:484.
 * Ref: Athena++ src/mesh/mesh.cpp Mesh::Step().
 * Ref: GRChombo GRAMRLevel::advance().
 */
static void subcycle_level(mesh_t *m, const sim_params_t *p,
                            int level, double dt_level, double t_start)
{
    /* Save pre-step state for temporal interpolation by finer levels */
    if (level < m->max_level)
        mesh_save_old_level(m, level);

    /* Compute temporal interpolation fraction for cross-level ghosts.
     * frac = how far into the coarser level's step we are at t_start.
     * At the start of a coarse step, frac = 0.
     * At the midpoint (second fine sub-step), frac = 0.5.
     *
     * For level 0: no coarser level, frac unused.
     * For level > 0: the coarse level's step runs from t_coarse_old to
     *   t_coarse_old + dt_coarse, where dt_coarse = 2 * dt_level.
     *   frac = (t_start + dt_level - t_coarse_old) / dt_coarse
     *        = fraction of the way through the coarse step AFTER this step.
     *   But for the ghost fill at the START of the step, we need:
     *   frac_start = (t_start - t_coarse_old) / dt_coarse
     *   The coarser level was already advanced, so its fields = t_coarse_new.
     *   Its fields_old = t_coarse_old.
     *
     * We pass the frac for the beginning of this level's step.
     * The coarse level's old time is when it saved fields_old. For the
     * first sub-step of a pair, frac = 0 (we're at coarse old time).
     * For the second sub-step, frac = 0.5 (midway through coarse step).
     *
     * We compute this from the block times: after the coarse step,
     * coarse blocks are at t_start (same as our t_start for first sub-step).
     * Actually, the coarse level advanced by dt_coarse = 2*dt_level before
     * we do the fine sub-steps. So:
     *   - coarse old time = t_start (saved before coarse step)
     *   - coarse new time = t_start + 2*dt_level
     *   - first fine step starts at t_start: frac = 0
     *   - second fine step starts at t_start + dt_level: frac = 0.5
     *
     * Wait, that's not right. Let me re-derive. The call pattern is:
     *   subcycle(L, dt_L, t):
     *     save_old(L)            // saves state at time t
     *     step(L, dt_L)          // advances L from t to t+dt_L
     *     subcycle(L+1, dt_L/2, t)          // first fine sub-step
     *     subcycle(L+1, dt_L/2, t+dt_L/2)  // second fine sub-step
     *
     * When subcycle(L+1, ..., t) is called:
     *   - L's fields_old = state at time t (before L stepped)
     *   - L's fields     = state at time t + dt_L
     *   - First fine sub-step (t_start = t): needs ghosts at time t → frac = 0
     *   - Second fine sub-step (t_start = t + dt_L/2): needs ghosts at t + dt_L/2
     *     → frac = (t + dt_L/2 - t) / dt_L = 0.5
     *
     * So frac = (t_start_of_fine - t_start_of_coarse) / dt_coarse.
     * The coarse level's t_start is our parent's t_start, passed to us.
     * For level > 0, we need to track the coarse level's step start time.
     *
     * Simplification: since subcycling is recursive and symmetric,
     * the frac at level L+1 is determined by which sub-step we're on:
     *   - Called with t_start = parent_t_start: frac = 0
     *   - Called with t_start = parent_t_start + dt_L/2: frac = 0.5
     *
     * We compute frac from t_start and the coarse level's step parameters.
     * The coarse level was called with subcycle(L, dt_L, parent_t_start),
     * so coarse old time = parent_t_start, dt_coarse = dt_level.
     *
     * Actually, we ARE level L. The finer level (L+1) will need frac.
     * For this level's own step, frac is passed to us from the caller.
     * Let me restructure: frac is computed by the CALLER (parent) for us. */

    /* For level 0, frac = 0 (no coarser level to interpolate from). */
    double frac = 0.0;
    if (level > 0) {
        /* Our coarser level (level-1) saved its old state at some time t_old.
         * After stepping, its fields are at t_old + dt_coarse.
         * dt_coarse = 2 * dt_level (our dt is half of parent's dt).
         * We were called at t_start. The coarse old time is t_start
         * rounded down to the coarse step boundary.
         *
         * Instead of tracking absolute times, we use a simpler approach:
         * The frac is computed from where t_start falls within the coarse
         * step interval. We check the coarse block's saved time.
         * But coarse blocks might not all have the same time.
         *
         * Simplest correct approach: compute frac from block times.
         * Find any coarse neighbor and use its time field. */
        double dt_coarse = 2.0 * dt_level;
        /* Find the coarse step start by looking at fields_old timing.
         * Coarse block time = t_coarse_new. Coarse old = t_coarse_new - dt_coarse.
         * frac = (t_start - (t_coarse_new - dt_coarse)) / dt_coarse
         *      = (t_start - t_coarse_new + dt_coarse) / dt_coarse
         *      = 1 + (t_start - t_coarse_new) / dt_coarse
         * But we don't track t_coarse_new explicitly...
         *
         * Better: since the recursion structure is rigid (binary subdivision),
         * we can compute frac from t_start and dt_coarse alone.
         * The coarse step covers [t_coarse_start, t_coarse_start + dt_coarse].
         * t_coarse_start = t_start rounded down to nearest multiple of dt_coarse.
         * frac = (t_start - t_coarse_start) / dt_coarse. */
        double t_coarse_start = floor(t_start / dt_coarse) * dt_coarse;
        /* Handle floating-point rounding at t_start near step boundaries */
        double offset = t_start - t_coarse_start;
        if (offset < 0.0) offset = 0.0;
        frac = offset / dt_coarse;
        /* Clamp frac to [0, 1) */
        if (frac < 0.0) frac = 0.0;
        if (frac >= 1.0) frac = 1.0 - 1.0e-14;
    }

    step_level(m, p, level, dt_level, frac);

    /* Update block times after stepping */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (b && b->is_leaf && b->loc.level == level)
            b->time = t_start + dt_level;
    }

    /* Subcycle finer levels: 2 sub-steps at dt/2 */
    if (level < m->max_level) {
        subcycle_level(m, p, level + 1, dt_level / 2.0, t_start);
        subcycle_level(m, p, level + 1, dt_level / 2.0,
                       t_start + dt_level / 2.0);
    }

    /* Restrict fine data into coarse parents at this level boundary */
    if (level < m->max_level)
        restrict_level_to_parents(m, level + 1);

    /* Enforce algebraic constraints at this level */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (b && b->is_leaf && b->loc.level == level)
            enforce_algebraic(b->grid);
    }
}

/* ========================================================================
 * Public interface — mesh-level stepping
 * ======================================================================== */

/*
 * Production path: packed batch kernels.
 * Uniform mesh (max_level == 0): single pack, no subcycling.
 * AMR mesh (max_level > 0): Berger-Oliger subcycling, per-level packs.
 *
 * Ghost exchange via device kernels (Commit 2: direct on pack buffers).
 *
 * Ref: Berger & Oliger (1984), JCP 53:484.
 */
void rk4_step_mesh(mesh_t *m, const sim_params_t *p,
                   rk4_rhs_func_t rhs_func, double dt)
{
    (void)rhs_func;  /* packed kernels call ccz4_rhs_point directly */

    if (m->max_level == 0) {
        /* Uniform mesh: single pack, no subcycling needed */
        if (p->rk_method == RK_CK45)
            ck45_step_mesh_packed(m, p, dt);
        else
            classic_rk4_step_mesh_packed(m, p, dt);
    } else {
        /* AMR with Berger-Oliger subcycling.
         * dt is the coarsest-level time step (CFL * dx_coarse).
         * Each finer level takes 2x more sub-steps at half the dt. */
        subcycle_level(m, p, 0, dt, p->time);
    }
}

/*
 * Per-block fallback for debug/comparison.
 * Same algorithm, one kernel launch per block per stage.
 * Should produce identical results to rk4_step_mesh — any difference
 * indicates a bug in the packed infrastructure.
 */
void rk4_step_mesh_perblock(mesh_t *m, const sim_params_t *p,
                             rk4_rhs_func_t rhs_func, double dt)
{
    if (p->rk_method == RK_CK45)
        ck45_step_mesh_perblock(m, p, rhs_func, dt);
    else
        classic_rk4_step_mesh_perblock(m, p, rhs_func, dt);
}
