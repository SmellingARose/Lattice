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
#include "../backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* ========================================================================
 * Mesh-level helpers
 * ======================================================================== */

/*
 * Restrict fine leaf data into non-leaf parents after a full RK step.
 * Keeps parent data synchronized for cross-level ghost exchange.
 */
static void restrict_level_to_parents(mesh_t *m, int level)
{
    #pragma omp parallel for schedule(dynamic)
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

                    for (int f = 0; f < child->grid->n_fields; f++) {
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

/* Synchronize non-leaf parent data after a full RK step */
static void mesh_restrict_to_parents(mesh_t *m)
{
    if (m->max_level == 0) return;
    for (int L = m->max_level; L >= 1; L--)
        restrict_level_to_parents(m, L);
}

/* ========================================================================
 * Packed batch mesh steppers (production path)
 *
 * All leaf blocks packed into a single meshblock_pack_t. One kernel
 * launch per operation (RHS, Sommerfeld, update) covers all blocks.
 *
 * Persistent packs: packs are cached in mesh_t across time steps.
 * Only the data buffer is synced in/out each step (via sync_to/from_blocks),
 * saving malloc/free and metadata rebuild overhead.
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
        n_leaves, npts, ids, -1, rk_method, m->n_fields);

    /* Load field data from blocks into pack's contiguous buffers */
    meshblock_pack_load(pack, m->blocks);

    /* Fill per-block metadata (origins, dx, boundary flags, levels).
     * Also allocates coarse_data if any blocks are refined. */
    meshblock_pack_load_meta(pack, m->blocks);

    /* Build pack-local neighbor table (mesh block ID → pack index).
     * Required by device ghost exchange and diagnostics. */
    meshblock_pack_build_neighbors(pack, m->blocks);

    /* Load coarse_buf data for multilevel ghost exchange */
    if (m->max_level > 0)
        meshblock_pack_load_coarse(pack, m->blocks);

    free(ids);
    return pack;
}


/*
 * CK45 packed mesh stepper: all leaf blocks in one pack, one kernel per op.
 *
 * Algorithm (same CK45 as per-block, different execution model):
 *   dU = 0   (zero scratch)
 *   For s = 0..4:
 *     Ghost exchange (direct on pack buffers)
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
    /* Get or rebuild cached leaf pack */
    if (!m->leaf_pack || m->packs_dirty) {
        if (m->leaf_pack) meshblock_pack_free(m->leaf_pack);
        m->leaf_pack = mesh_build_leaf_pack(m, RK_CK45);
        m->packs_dirty = 0;
    } else {
        meshblock_pack_sync_from_blocks(m->leaf_pack, m->blocks);
    }
    meshblock_pack_t *pack = m->leaf_pack;

    backend_map_pack(pack, p);

    backend_zero_packed(pack, PACK_BUF_SCRATCH);

    for (int s = 0; s < 5; s++) {
        backend_enforce_algebraic_packed(pack);  /* chi/lapse >= 1e-4 before RHS */
        backend_ghost_exchange_packed(pack);
        backend_compute_rhs_packed(pack, p);
        backend_sommerfeld_packed(pack, p);
        backend_update_ck45_packed(pack, CK_A[s], CK_B[s], dt);
    }

    backend_enforce_algebraic_packed(pack);  /* final enforcement */
    backend_unmap_pack_sync(pack);

    meshblock_pack_sync_to_blocks(pack, m->blocks);
    mesh_restrict_to_parents(m);
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
    /* Get or rebuild cached leaf pack */
    if (!m->leaf_pack || m->packs_dirty) {
        if (m->leaf_pack) meshblock_pack_free(m->leaf_pack);
        m->leaf_pack = mesh_build_leaf_pack(m, RK_CLASSIC);
        m->packs_dirty = 0;
    } else {
        meshblock_pack_sync_from_blocks(m->leaf_pack, m->blocks);
    }
    meshblock_pack_t *pack = m->leaf_pack;

    backend_map_pack(pack, p);

    backend_copy_packed(pack, PACK_BUF_SCRATCH, PACK_BUF_DATA);
    backend_zero_packed(pack, PACK_BUF_ACCUM);

    /* Stage 1: accum += dt/6 * rhs; data = scratch + dt/2 * rhs */
    backend_enforce_algebraic_packed(pack);  /* chi/lapse >= 1e-4 before RHS */
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_rk4_stage_packed(pack, 1.0/6.0, 0.5, dt);

    /* Stage 2: accum += dt/3 * rhs; data = scratch + dt/2 * rhs */
    backend_enforce_algebraic_packed(pack);
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_rk4_stage_packed(pack, 1.0/3.0, 0.5, dt);

    /* Stage 3: accum += dt/3 * rhs; data = scratch + dt * rhs */
    backend_enforce_algebraic_packed(pack);
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_rk4_stage_packed(pack, 1.0/3.0, 1.0, dt);

    /* Stage 4: data = scratch + accum + dt/6 * rhs */
    backend_enforce_algebraic_packed(pack);
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_rk4_final_packed(pack, 1.0/6.0, dt);

    backend_enforce_algebraic_packed(pack);  /* final enforcement */
    backend_unmap_pack_sync(pack);

    meshblock_pack_sync_to_blocks(pack, m->blocks);

    /* Fix inter-block ghost zones. Non-boundary ghost RHS stays zero, so
     * the final data = scratch + accum leaves ghosts at U^n. A same-level
     * exchange restores correct ghost values from neighbor interiors. */
    ghost_exchange(m);

    mesh_restrict_to_parents(m);
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
 * Accumulate RK4 stage RHS from pack into per-block Taylor coefficient
 * buffers for cubic temporal interpolation. Called after each stage's
 * RHS + BCs are applied.
 *
 * Only accumulates for blocks that have Taylor buffers allocated
 * (level < max_level). Reads from the pack's RHS buffer which has
 * the current stage's right-hand side.
 *
 * Ref: Chombo TimeInterpolatorRK4::saveRHS coefficient accumulation.
 */
static void accumulate_taylor_from_pack(const meshblock_pack_t *pack,
                                         block_t **blocks,
                                         int stage, double dt)
{
    for (int b = 0; b < pack->n_blocks; b++) {
        block_t *blk = blocks[pack->block_ids[b]];
        if (!blk->taylor_block) continue;

        size_t npts = pack->npts;
        int nf = pack->n_fields;

        /* Build per-field RHS pointer array from pack layout */
        const double *rhs_ptrs[NUM_FIELDS];
        for (int f = 0; f < nf; f++)
            rhs_ptrs[f] = pack->rhs + (size_t)f * pack->n_blocks * npts
                         + (size_t)b * npts;

        block_accumulate_taylor(blk, rhs_ptrs, stage, dt, npts);
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
                                                rk_method_t rk_method,
                                                int with_buffers)
{
    /* Count leaf blocks (evolved) and non-leaf buffer blocks at this level.
     * Buffer blocks (AthenaK pattern): non-leaf parents stay in the pack
     * as data sources for ghost exchange. RK4 kernels skip them.
     * Ref: AthenaK src/mesh/meshblock_pack.hpp */
    int n_leaves = 0, n_buffers = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || b->loc.level != level) continue;
        if (b->is_leaf) n_leaves++;
        else if (with_buffers && b->grid) n_buffers++;
    }

    if (n_leaves == 0) return NULL;

    int n_total = n_leaves + n_buffers;
    int *ids = malloc(n_total * sizeof(int));
    if (!ids) {
        fprintf(stderr, "mesh_build_level_pack: malloc failed\n");
        exit(1);
    }

    /* Leaf blocks first [0, n_leaves), buffers after [n_leaves, n_total) */
    int idx = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (b && b->is_leaf && b->loc.level == level)
            ids[idx++] = bid;
    }
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (b && !b->is_leaf && b->loc.level == level && b->grid)
            ids[idx++] = bid;
    }

    size_t npts = m->blocks[ids[0]]->grid->npoints;

    meshblock_pack_t *pack = meshblock_pack_create(
        n_total, npts, ids, level, rk_method, m->n_fields);
    pack->n_evolve = n_leaves;

    meshblock_pack_load(pack, m->blocks);
    meshblock_pack_load_meta(pack, m->blocks);
    meshblock_pack_build_neighbors(pack, m->blocks);

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
    /* Get or rebuild cached level pack */
    if (level < MAX_AMR_LEVELS &&
        m->level_packs[level] && !m->packs_dirty) {
        /* Sync block fields (including filled ghost zones) into pack data */
        meshblock_pack_sync_from_blocks(m->level_packs[level], m->blocks);
        if (m->level_packs[level]->n_refined > 0)
            meshblock_pack_load_coarse(m->level_packs[level], m->blocks);
    } else {
        /* Build fresh level pack */
        if (level < MAX_AMR_LEVELS && m->level_packs[level])
            meshblock_pack_free(m->level_packs[level]);
        meshblock_pack_t *new_pack = mesh_build_level_pack(m, level, p->rk_method, 0);
        if (!new_pack) return;  /* no blocks at this level */
        if (level < MAX_AMR_LEVELS)
            m->level_packs[level] = new_pack;
    }

    meshblock_pack_t *pack = (level < MAX_AMR_LEVELS) ?
        m->level_packs[level] : NULL;
    if (!pack) return;

    backend_map_pack(pack, p);

    /* RK4 sub-stage time offsets for per-stage cross-level ghost fill.
     * stage_frac = frac + c_s / refine_ratio.
     * Ref: GRChombo evalRHS → fillInterp at each sub-stage. */
    const double c_s[4] = {0.0, 0.5, 0.5, 1.0};
    const double inv_ratio = 0.5;  /* 1/refine_ratio, refine_ratio=2 */

    if (p->rk_method == RK_CK45) {
        backend_zero_packed(pack, PACK_BUF_SCRATCH);
        for (int s = 0; s < 5; s++) {
            backend_enforce_algebraic_packed(pack);
            backend_ghost_exchange_packed(pack);
            backend_compute_rhs_packed(pack, p);
            backend_sommerfeld_packed(pack, p);
            backend_update_ck45_packed(pack, CK_A[s], CK_B[s], dt_level);
        }
    } else {
        backend_copy_packed(pack, PACK_BUF_SCRATCH, PACK_BUF_DATA);
        backend_zero_packed(pack, PACK_BUF_ACCUM);
        backend_zero_packed(pack, PACK_BUF_RHS);

        /* Stage 1: evaluate at t (c=0) */
        if (level > 0) {
            meshblock_pack_sync_to_blocks(pack, m->blocks);
            ghost_fill_from_coarser(m, level, frac + c_s[0] * inv_ratio);
            meshblock_pack_sync_from_blocks(pack, m->blocks);
            if (pack->n_refined > 0)
                meshblock_pack_load_coarse(pack, m->blocks);
        }
        backend_enforce_algebraic_packed(pack);
        backend_ghost_exchange_packed(pack);
        backend_compute_rhs_packed(pack, p);
        backend_sommerfeld_packed(pack, p);
        accumulate_taylor_from_pack(pack, m->blocks, 0, dt_level);
        backend_rk4_stage_packed(pack, 1.0/6.0, 0.5, dt_level);

        /* Stage 2: evaluate at t + dt/2 (c=0.5) */
        if (level > 0) {
            meshblock_pack_sync_to_blocks(pack, m->blocks);
            ghost_fill_from_coarser(m, level, frac + c_s[1] * inv_ratio);
            meshblock_pack_sync_from_blocks(pack, m->blocks);
            if (pack->n_refined > 0)
                meshblock_pack_load_coarse(pack, m->blocks);
        }
        backend_enforce_algebraic_packed(pack);
        backend_ghost_exchange_packed(pack);
        backend_compute_rhs_packed(pack, p);
        backend_sommerfeld_packed(pack, p);
        accumulate_taylor_from_pack(pack, m->blocks, 1, dt_level);
        backend_rk4_stage_packed(pack, 1.0/3.0, 0.5, dt_level);

        /* Stage 3: evaluate at t + dt/2 (c=0.5) */
        if (level > 0) {
            meshblock_pack_sync_to_blocks(pack, m->blocks);
            ghost_fill_from_coarser(m, level, frac + c_s[2] * inv_ratio);
            meshblock_pack_sync_from_blocks(pack, m->blocks);
            if (pack->n_refined > 0)
                meshblock_pack_load_coarse(pack, m->blocks);
        }
        backend_enforce_algebraic_packed(pack);
        backend_ghost_exchange_packed(pack);
        backend_compute_rhs_packed(pack, p);
        backend_sommerfeld_packed(pack, p);
        accumulate_taylor_from_pack(pack, m->blocks, 2, dt_level);
        backend_rk4_stage_packed(pack, 1.0/3.0, 1.0, dt_level);

        /* Stage 4: evaluate at t + dt (c=1.0) */
        if (level > 0) {
            meshblock_pack_sync_to_blocks(pack, m->blocks);
            ghost_fill_from_coarser(m, level, frac + c_s[3] * inv_ratio);
            meshblock_pack_sync_from_blocks(pack, m->blocks);
            if (pack->n_refined > 0)
                meshblock_pack_load_coarse(pack, m->blocks);
        }
        backend_enforce_algebraic_packed(pack);
        backend_ghost_exchange_packed(pack);
        backend_compute_rhs_packed(pack, p);
        backend_sommerfeld_packed(pack, p);
        accumulate_taylor_from_pack(pack, m->blocks, 3, dt_level);
        backend_rk4_final_packed(pack, 1.0/6.0, dt_level);
    }

    backend_enforce_algebraic_packed(pack);
    backend_unmap_pack_sync(pack);

    /* Sync only data buffer back to blocks (not rhs/scratch/accum) */
    meshblock_pack_sync_to_blocks(pack, m->blocks);
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
                            int level, double dt_level, double t_start,
                            int sub_step)
{
    /* Save pre-step state for temporal interpolation by finer levels */
    if (level < m->max_level)
        mesh_save_old_level(m, level);

    /* Temporal interpolation fraction for cross-level ghosts.
     *
     * sub_step is an integer (0 or 1) passed by the parent level:
     *   0 = first fine sub-step  → frac = 0.0 (use coarse old state)
     *   1 = second fine sub-step → frac = 0.5 (midpoint interpolation)
     *
     * Using integer sub_step eliminates floating-point drift that occurs
     * when computing frac via floor(t_start / dt_coarse) * dt_coarse.
     * At deep recursion (many levels), t_start accumulates rounding errors,
     * but sub_step is always exact.
     *
     * Level 0 has no coarser level, so frac is unused (sub_step = 0).
     *
     * Ref: Berger & Oliger (1984), Athena++ Mesh::Step(). */
    double frac = (level > 0) ? sub_step * 0.5 : 0.0;

    step_level(m, p, level, dt_level, frac);

    /* Update block times after stepping */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (b && b->is_leaf && b->loc.level == level)
            b->time = t_start + dt_level;
    }

    /* Subcycle diagnostic callback: fire after each step at diag_level.
     * Enables fine-cadence BH tracking and Psi4 extraction.
     * Ref: GRChombo specificPostTimeStep at extraction_level. */
    if (p->subcycle_diag && level == p->diag_level)
        p->subcycle_diag(m, t_start + dt_level, level, p->diag_ctx);

    /* Subcycle finer levels: 2 sub-steps at dt/2 */
    if (level < m->max_level) {
        subcycle_level(m, p, level + 1, dt_level / 2.0, t_start, 0);
        subcycle_level(m, p, level + 1, dt_level / 2.0,
                       t_start + dt_level / 2.0, 1);
    }

    /* Restrict fine data into coarse parents at this level boundary.
     * Ghost exchange first: the 6th-order restriction stencil reads 2 cells
     * into ghost zones. After step_level(), ghosts may be stale (classic RK4
     * overwrites them with pre-step values in the copy+apply stage, and even
     * CK45 ghosts are one stage behind). A same-level exchange ensures the
     * stencil reads valid post-step data.
     *
     * Ref: Berger & Oliger (1984) — restrict after subcycling at level boundary.
     * Ref: Athena++ MeshRefinement::RestrictCellCenteredValues() post-exchange. */
    if (level < m->max_level) {
        /* Post-subcycle restriction: fine → coarse parents.
         *
         * ghost_exchange: fill same-level ghost zones (stale after RK4).
         * ghost_fill_from_coarser: fill CROSS-LEVEL ghost zones for fine
         *   blocks at AMR boundaries (facing coarser-level blocks). These
         *   were filled at the START of step_level but are stale after RK4
         *   stages overwrote them. The 6th-order restriction stencil reads
         *   2 cells into ghost zones — if cross-level ghosts are stale,
         *   restriction produces garbage → NaN.
         * restrict_level_to_parents: restrict with valid ghosts.
         *
         * frac=1.0: use coarse post-step data (coarse level already stepped).
         *
         * Ref: Berger & Oliger (1984) — restrict at level boundary.
         * Ref: Athena++ MeshRefinement::RestrictCellCenteredValues. */
        ghost_exchange(m);
        ghost_fill_from_coarser(m, level + 1, 1.0);
        restrict_level_to_parents(m, level + 1);
    }

    /* Algebraic constraints (det(h)=1, tr(A)=0) enforced inside step_level
     * via backend_enforce_algebraic_packed — runs on pack data before unmap,
     * so GPU can execute it on device. No per-block pass needed here. */
}

/* ========================================================================
 * GPU-resident Berger-Oliger subcycling
 *
 * Zero PCIe transfers during evolution. Data uploaded once to device,
 * stays resident across all sub-steps, downloaded once for output.
 *
 * Key differences from CPU subcycling:
 *   - Cross-level ghosts filled on device (backend_cross_level_ghost_fill_packed)
 *   - Pack switching via backend_activate_pack (no memcpy)
 *   - Temporal interp state saved on device (backend_save_old_packed)
 *   - Single host sync after all levels done (gpu_sync_all_to_host)
 *
 * Ref: Berger & Oliger (1984), JCP 53:484.
 * ======================================================================== */

/*
 * Build cross-level neighbor map for GPU-resident ghost fill.
 * For each fine block with a coarser-level neighbor, records
 * (fine_pack_idx, direction, coarse_pack_idx) for the cross-level
 * ghost fill kernel.
 */
static void build_cross_level_map(mesh_t *m, meshblock_pack_t *fine_pack,
                                   meshblock_pack_t *coarse_pack)
{
    /* Build reverse map: mesh_block_id → coarse pack index */
    int max_id = 0;
    for (int i = 0; i < coarse_pack->n_blocks; i++)
        if (coarse_pack->block_ids[i] > max_id)
            max_id = coarse_pack->block_ids[i];
    for (int i = 0; i < fine_pack->n_blocks; i++)
        if (fine_pack->block_ids[i] > max_id)
            max_id = fine_pack->block_ids[i];

    int *coarse_rev = calloc(max_id + 1, sizeof(int));
    for (int i = 0; i <= max_id; i++) coarse_rev[i] = -1;
    for (int b = 0; b < coarse_pack->n_blocks; b++)
        coarse_rev[coarse_pack->block_ids[b]] = b;

    /* Scan fine pack for cross-level neighbors */
    int max_entries = fine_pack->n_blocks * NUM_NEIGHBORS;
    int *map = malloc(max_entries * 3 * sizeof(int));
    int count = 0;

    for (int fb = 0; fb < fine_pack->n_blocks; fb++) {
        int bid = fine_pack->block_ids[fb];
        block_t *blk = m->blocks[bid];
        int blk_level = blk->loc.level;

        for (int d = 0; d < NUM_NEIGHBORS; d++) {
            int ox = nbr_offset[d][0];
            int oy = nbr_offset[d][1];
            int oz = nbr_offset[d][2];
            int nlev = blk->nblevel[oz+1][oy+1][ox+1];

            if (nlev >= 0 && nlev == blk_level - 1) {
                /* This fine block has a coarser-level neighbor in direction d */
                int nbr_id = blk->neighbor_ids[d];
                if (nbr_id >= 0 && nbr_id <= max_id) {
                    int ci = coarse_rev[nbr_id];
                    if (ci >= 0) {
                        map[count*3+0] = fb;
                        map[count*3+1] = d;
                        map[count*3+2] = ci;
                        count++;
                    }
                }
            }
        }
    }

    /* Upload to device (stored in fine pack's device handle) via backend */
    if (count > 0) {
        /* The device handle's cross_level_map is set by the backend.
         * We store the host map temporarily and let backend upload it. */
        extern void backend_upload_cross_level_map(meshblock_pack_t *pack,
                                                    const int *map, int count);
        backend_upload_cross_level_map(fine_pack, map, count);
    }

    free(map);
    free(coarse_rev);
}

/*
 * Ensure all level packs exist and are mapped to device.
 * Called once at start of GPU-resident subcycling (or after regrid).
 */
static void gpu_ensure_level_packs(mesh_t *m, const sim_params_t *p)
{
    for (int L = 0; L <= m->max_level; L++) {
        if (!m->level_packs[L] || m->packs_dirty) {
            if (m->level_packs[L])
                meshblock_pack_free(m->level_packs[L]);
            m->level_packs[L] = mesh_build_level_pack(m, L, p->rk_method, 1);
            if (!m->level_packs[L]) continue;

            /* Map to device (first map: allocates everything) */
            backend_map_pack(m->level_packs[L], p);
            backend_unmap_pack(m->level_packs[L]);
        } else if (!m->level_packs[L]->device_handle) {
            /* Pack exists but no device memory yet — sync and map */
            meshblock_pack_sync_from_blocks(m->level_packs[L], m->blocks);
            if (m->level_packs[L]->n_refined > 0)
                meshblock_pack_load_coarse(m->level_packs[L], m->blocks);
            backend_map_pack(m->level_packs[L], p);
            backend_unmap_pack(m->level_packs[L]);
        } else {
            /* Pack exists and has device memory — sync data H→D */
            meshblock_pack_sync_from_blocks(m->level_packs[L], m->blocks);
            if (m->level_packs[L]->n_refined > 0)
                meshblock_pack_load_coarse(m->level_packs[L], m->blocks);
            backend_map_pack(m->level_packs[L], p);
            backend_unmap_pack(m->level_packs[L]);
        }
    }

    /* Set has_children flag for Taylor buffer allocation on GPU */
    for (int L = 0; L <= m->max_level; L++) {
        if (m->level_packs[L])
            m->level_packs[L]->has_children = (L < m->max_level) ? 1 : 0;
    }

    /* Build cross-level maps for levels > 0 */
    for (int L = 1; L <= m->max_level; L++) {
        if (m->level_packs[L] && m->level_packs[L-1] &&
            m->level_packs[L]->n_refined > 0) {
            build_cross_level_map(m, m->level_packs[L], m->level_packs[L-1]);
        }
    }

    /* Build restriction maps: for each coarse pack with buffer blocks,
     * map each buffer block's 8 children to fine pack block indices.
     * Used by backend_restrict_to_buffers_packed. */
    for (int L = 0; L < m->max_level; L++) {
        meshblock_pack_t *cpk = m->level_packs[L];
        meshblock_pack_t *fpk = m->level_packs[L + 1];
        if (!cpk || !fpk) continue;
        int n_buf = cpk->n_blocks - cpk->n_evolve;
        if (n_buf == 0) continue;

        /* Build reverse lookup: mesh block ID → fine pack index */
        int max_id = 0;
        for (int b = 0; b < fpk->n_blocks; b++)
            if (fpk->block_ids[b] > max_id) max_id = fpk->block_ids[b];
        int *fine_rev = calloc(max_id + 1, sizeof(int));
        for (int i = 0; i <= max_id; i++) fine_rev[i] = -1;
        for (int b = 0; b < fpk->n_blocks; b++)
            fine_rev[fpk->block_ids[b]] = b;

        int *rmap = malloc(n_buf * 8 * sizeof(int));
        for (int i = 0; i < n_buf * 8; i++) rmap[i] = -1;

        for (int bi = 0; bi < n_buf; bi++) {
            int buf_pack_idx = cpk->n_evolve + bi;
            int mesh_id = cpk->block_ids[buf_pack_idx];
            block_t *parent = m->blocks[mesh_id];
            if (!parent) continue;
            for (int c = 0; c < 8; c++) {
                int child_id = parent->child_ids[c];
                if (child_id >= 0 && child_id <= max_id)
                    rmap[bi * 8 + c] = fine_rev[child_id];
            }
        }

        backend_upload_restrict_map(cpk, rmap, n_buf);
        free(rmap);
        free(fine_rev);
    }
}

/*
 * Sync all level packs' data from device back to host blocks.
 * Called once after the full GPU-resident subcycling completes.
 * Also runs ghost_exchange and restrict_to_parents on host for consistency.
 */
void gpu_sync_all_to_host(mesh_t *m)
{
    for (int L = 0; L <= m->max_level; L++) {
        meshblock_pack_t *pack = m->level_packs[L];
        if (!pack) continue;

        /* Activate this pack's device state and sync data D→H */
        backend_activate_pack(pack);
        backend_unmap_pack_sync(pack);

        /* Copy device-synced data back to block structs */
        meshblock_pack_sync_to_blocks(pack, m->blocks);
    }

    /* Restore inter-block consistency on host */
    ghost_exchange(m);
    mesh_restrict_to_parents(m);
}

/*
 * GPU-resident level step: all kernels on device, no PCIe transfers.
 * Classic RK4 only (GPU-resident path requires RK_CLASSIC).
 */
static void step_level_gpu(mesh_t *m, const sim_params_t *p,
                            int level, double dt_level, double frac)
{
    meshblock_pack_t *pack = m->level_packs[level];
    if (!pack) return;

    /* Coarser level pack for per-stage cross-level ghost fill.
     * GRChombo fills coarse-fine boundary ghosts at every RK4 sub-stage with
     * temporally interpolated data at the sub-stage time (Chombo fillInterp).
     * This prevents stale cross-level ghost data from accumulating O(dt) error
     * across stages, which compounds exponentially through deep AMR hierarchies.
     * Ref: arXiv:2112.10567 (GRChombo AMR lessons), Chombo TimeInterpolatorRK4. */
    meshblock_pack_t *coarser = (level > 0) ? m->level_packs[level - 1] : NULL;

    /* RK4 sub-stage time offsets within a single step: c_s = {0, 0.5, 0.5, 1.0}.
     * Convert to coarse-time fraction: stage_frac = frac + c_s / refine_ratio.
     * refine_ratio = 2 for oct-tree AMR (each level halves dx and dt). */
    const int refine_ratio = 2;
    const double inv_ratio = 1.0 / refine_ratio;

    /* Set global d_ptrs from this pack's device handle (no memcpy) */
    backend_activate_pack(pack);

    /* Classic RK4 — identical kernel sequence as existing step_level.
     * Zero RHS buffer: compute_rhs only writes interior cells [ghost,ghost+N).
     * Ghost zone RHS would be garbage (from previous step/level), corrupting
     * ghost data via rk4_stage: data = scratch + c*dt*garbage_rhs.
     * Ref: AthenaK zeros scratch arrays before each kernel phase. */
    backend_copy_packed(pack, PACK_BUF_SCRATCH, PACK_BUF_DATA);
    backend_zero_packed(pack, PACK_BUF_ACCUM);
    backend_zero_packed(pack, PACK_BUF_RHS);

    /* Stage 1: evaluate at t (c=0) */
    if (coarser)
        backend_cross_level_ghost_fill_packed(pack, coarser, frac);
    backend_enforce_algebraic_packed(pack);
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_taylor_accumulate_packed(pack, 0, dt_level);
    backend_rk4_stage_packed(pack, 1.0/6.0, 0.5, dt_level);

    /* Stage 2: evaluate at t + dt/2 (c=0.5) */
    if (coarser)
        backend_cross_level_ghost_fill_packed(pack, coarser,
                                              frac + 0.5 * inv_ratio);
    backend_enforce_algebraic_packed(pack);
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_taylor_accumulate_packed(pack, 1, dt_level);
    backend_rk4_stage_packed(pack, 1.0/3.0, 0.5, dt_level);

    /* Stage 3: evaluate at t + dt/2 (c=0.5) */
    if (coarser)
        backend_cross_level_ghost_fill_packed(pack, coarser,
                                              frac + 0.5 * inv_ratio);
    backend_enforce_algebraic_packed(pack);
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_taylor_accumulate_packed(pack, 2, dt_level);
    backend_rk4_stage_packed(pack, 1.0/3.0, 1.0, dt_level);

    /* Stage 4: evaluate at t + dt (c=1.0) */
    if (coarser)
        backend_cross_level_ghost_fill_packed(pack, coarser,
                                              frac + 1.0 * inv_ratio);
    backend_enforce_algebraic_packed(pack);
    backend_ghost_exchange_packed(pack);
    backend_compute_rhs_packed(pack, p);
    backend_sommerfeld_packed(pack, p);
    backend_taylor_accumulate_packed(pack, 3, dt_level);
    backend_rk4_final_packed(pack, 1.0/6.0, dt_level);

    backend_enforce_algebraic_packed(pack);
    /* NO unmap — data stays on device */
}

/*
 * GPU-resident recursive subcycling.
 * Same algorithm as subcycle_level() but operates entirely on device.
 * No PCIe transfers between sub-steps.
 *
 * Ref: Berger & Oliger (1984), JCP 53:484.
 */
static void subcycle_level_gpu(mesh_t *m, const sim_params_t *p,
                                int level, double dt_level, double t_start,
                                int sub_step)
{
    /* Save pre-step state on device for temporal interp by finer levels */
    if (level < m->max_level && m->level_packs[level])
        backend_save_old_packed(m->level_packs[level]);

    /* Temporal interpolation fraction (same as CPU path) */
    double frac = (level > 0) ? sub_step * 0.5 : 0.0;

    step_level_gpu(m, p, level, dt_level, frac);

    /* Subcycle diagnostic callback.
     * GPU path: call GPU callback directly on device-resident data.
     * CPU path (fallback): sync to host, call CPU callback.
     * Ref: AthenaK keeps data on device, uses Kokkos reduction for scalars. */
    if (level == p->diag_level) {
        if (p->subcycle_diag_gpu) {
            p->subcycle_diag_gpu(m, t_start + dt_level, level, p->diag_ctx);
        } else if (p->subcycle_diag) {
            gpu_sync_all_to_host(m);
            p->subcycle_diag(m, t_start + dt_level, level, p->diag_ctx);
        }
    }

    /* Subcycle finer levels: 2 sub-steps at dt/2 */
    if (level < m->max_level) {
        subcycle_level_gpu(m, p, level + 1, dt_level / 2.0, t_start, 0);
        subcycle_level_gpu(m, p, level + 1, dt_level / 2.0,
                           t_start + dt_level / 2.0, 1);

        /* Post-subcycle restriction: GPU-native, zero PCIe.
         *
         * Every production AMR code restricts after fine subcycling:
         * GRChombo (coarseAverage), Athena++ (RestrictCellCenteredValues),
         * CarpetX (average_down), AthenaK, BAM. No exceptions.
         *
         * GPU sequence (3 kernel launches, no host transfer):
         * 1. ghost_exchange on fine pack — valid same-level ghosts
         * 2. cross_level_ghost_fill — valid cross-level ghosts from coarse
         * 3. restrict fine interior+ghost → buffer blocks in coarse pack
         *
         * Buffer blocks (AthenaK pattern) are non-leaf parents packed at
         * [n_evolve, n_blocks). Ghost exchange reads their interiors to
         * fill leaf ghost zones at refined boundaries.
         *
         * Ref: Berger & Oliger (1984), JCP 53:484.
         * Ref: AthenaK device-resident restriction. */
        meshblock_pack_t *fpk = m->level_packs[level + 1];
        meshblock_pack_t *cpk = m->level_packs[level];

        if (fpk && cpk && cpk->n_evolve < cpk->n_blocks) {
            /* 6th-order restriction with floor clamp: fine → buffer blocks.
             * Ghost exchange fills fine same-level ghosts (stencil reads
             * 2 cells into ghost zone). Ghost data is from last RK4 stage —
             * stale by O(dt), but much better than O(dx²) from 0th-order.
             * Ref: ExaHyPE arXiv:2504.15814 — restriction order must match
             * FD order to avoid Ham violations at AMR boundaries. */
            backend_activate_pack(fpk);
            backend_ghost_exchange_packed(fpk);
            backend_cross_level_ghost_fill_packed(fpk, cpk, 1.0);
            backend_restrict_to_buffers_packed(fpk, cpk);

            /* Post-restriction coarse ghost exchange.
             * Restriction just wrote into coarse buffer blocks. Coarse leaf
             * blocks bordering those buffers need updated ghost zones.
             * Without this, ghosts contain pre-restriction data → O(dt) error
             * that compounds through deep AMR hierarchies.
             *
             * Ref: GRChombo postTimeStep → averageToCoarse → fillBdyGhosts. */
            backend_activate_pack(cpk);
            backend_ghost_exchange_packed(cpk);
            backend_enforce_algebraic_packed(cpk);
        }
    }
}

/* ========================================================================
 * Public interface — mesh-level stepping
 * ======================================================================== */

/*
 * Production path: packed batch kernels.
 * Uniform mesh (max_level == 0): single pack, no subcycling.
 * AMR mesh (max_level > 0): Berger-Oliger subcycling, per-level packs.
 *   GPU: device-resident subcycling (zero PCIe during sub-steps).
 *   CPU: host-side subcycling with per-level packs.
 *
 * Ghost exchange via device kernels (direct on pack buffers).
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
    } else if (backend_is_gpu() && p->rk_method == RK_CLASSIC) {
        /* GPU AMR subcycling with host-orchestrated restriction.
         * Level packs stay on device during evolution (GPU compute).
         * Post-subcycle restriction + ghost exchange done on host via
         * D→H sync, CPU restrict_level_to_parents, H→D re-sync.
         * Matches CarpetX/AMReX pattern (host-orchestrated restriction).
         *
         * Ref: Berger & Oliger (1984), JCP 53:484. */
        gpu_ensure_level_packs(m, p);
        subcycle_level_gpu(m, p, 0, dt, p->time, 0);
        /* Data stays on device — caller syncs to host only when needed
         * (checkpoint, output, AH finder, regrid). GPU diagnostics
         * (constraints, Psi4, BH separation) run directly on device packs. */
        if (m->packs_dirty) m->packs_dirty = 0;
    } else {
        /* CPU AMR subcycling (existing path).
         * Also used for GPU + CK45 (quartic temporal interp not yet on device). */
        subcycle_level(m, p, 0, dt, p->time, 0);
        if (m->packs_dirty) m->packs_dirty = 0;
    }
}
