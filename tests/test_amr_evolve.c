/*
 * Lattice — 3D Numerical Relativity
 * AMR integration test: mesh evolution with dynamic regridding.
 *
 * Tests:
 *   1. Refined mesh vs single-block BH evolution
 *   2. Dynamic regridding around a BH (blocks increase)
 *   3. Flat spacetime with regridding (should not refine)
 *
 * Pass criteria:
 *   - Test 1: Ham L2 finite and bounded for both
 *   - Test 2: No crash, Ham L2 bounded, block count increased
 *   - Test 3: Block count unchanged
 */

#include "../src/amr/mesh.h"
#include "../src/amr/refine.h"
#include "../src/amr/block.h"
#include "../src/core/params.h"
#include "../src/initial_data/puncture.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int test_count = 0;
static int pass_count = 0;

static void check(int cond, const char *name)
{
    test_count++;
    if (cond) {
        pass_count++;
        printf("  [PASS] %s\n", name);
    } else {
        printf("  [FAIL] %s\n", name);
    }
}

/* ======================================================================
 * Test 1: Refined mesh vs single-block BH evolution
 *
 * Single BH (M=1) at origin, evolve 20 steps.
 * Compare: single root block N=32 vs refined mesh (root N=32, refine once
 * near BH → 8 child blocks at level 1 plus root).
 * Uses global timestepping with CFL on finest level.
 * Pass: Ham L2 finite and bounded for both.
 * ====================================================================== */
static void test_refined_vs_single_block(void)
{
    printf("\n--- Test 1: Refined mesh vs single-block BH evolution ---\n");

    sim_params_t p = default_params();
    p.L = 64.0;
    p.CFL = 0.25;
    int nsteps = 20;

    double mass = 1.0;
    double center[1][3] = {{0.0, 0.0, 0.0}};

    /* (a) Single-block reference: N=32, L=64 */
    mesh_t *mref = mesh_create_ex(32, p.L, p.rk_method, NUM_FIELDS);
    p.dx = mref->dx_base;
    p.dt = p.CFL * p.dx;

    set_brill_lindquist(mref->blocks[0]->grid, 1, &mass,
                        (const double(*)[3])center);
    p.time = 0.0;
    for (int step = 0; step < nsteps; step++) {
        rk4_step_mesh(mref, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;
    }

    double ham_ref = mesh_constraint_l2(mref);
    printf("  Single-block: Ham L2 = %.6e (N=32, %d steps)\n",
           ham_ref, nsteps);

    /* (b) Refined mesh: N=32 root, refine once → multi-block */
    mesh_t *m = mesh_create(32, p.L, p.rk_method);
    p.amr.chi_refine  = 0.05;
    p.amr.chi_coarsen = 0.0;
    p.amr.max_level   = 1;

    /* Set initial data, then refine */
    set_brill_lindquist_global(m->blocks[0]->grid, m->blocks[0]->origin,
                               1, &mass, (const double(*)[3])center);
    mesh_regrid(m, &p.amr);

    /* Re-set initial data on all leaf blocks after regrid */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        set_brill_lindquist_global(b->grid, b->origin, 1, &mass,
                                   (const double(*)[3])center);
    }

    int leaves = mesh_num_leaves(m);
    double dx_fine = m->dx_base / (1 << m->max_level);
    p.dx = dx_fine;
    p.dt = p.CFL * dx_fine;
    printf("  Refined mesh: %d leaves, max_level=%d, dx_fine=%.4f\n",
           leaves, m->max_level, dx_fine);

    p.time = 0.0;
    for (int step = 0; step < nsteps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;
    }

    double ham_mesh = mesh_constraint_l2(m);
    printf("  Refined mesh: Ham L2 = %.6e (%d leaves, %d steps)\n",
           ham_mesh, leaves, nsteps);

    /* Both should be finite and bounded */
    check(isfinite(ham_ref), "Single-block Ham L2 finite");
    check(isfinite(ham_mesh), "Refined mesh Ham L2 finite");
    check(ham_mesh < 1.0, "Refined mesh Ham L2 bounded (< 1.0)");

    mesh_free(mref);
    mesh_free(m);
}

/* ======================================================================
 * Test 2: Dynamic regridding around a BH
 *
 * Single BH (M=1), single root block N_block=16, L=64.
 * Set chi_refine=0.05 (triggers near BH), regrid_every=5.
 * Evolve 20 steps with regridding.
 * After regridding, dt is recalculated from the finest-level dx to
 * satisfy CFL at all levels (no subcycling yet).
 * Pass: No crash, Ham L2 bounded, refinement triggered.
 * ====================================================================== */
static void test_dynamic_regridding(void)
{
    printf("\n--- Test 2: Dynamic regridding around a BH ---\n");

    sim_params_t p = default_params();
    p.L = 64.0;
    p.CFL = 0.25;
    int nsteps = 20;

    mesh_t *m = mesh_create(16, p.L, p.rk_method);
    p.dx = m->dx_base;
    p.dt = p.CFL * p.dx;
    p.amr.chi_refine  = 0.05;
    p.amr.chi_coarsen = 0.0;   /* disable coarsening — focus on refine + evolve */
    p.amr.max_level   = 1;     /* single level of refinement */
    p.amr.regrid_every = 5;

    int initial_blocks = mesh_num_leaves(m);
    int max_blocks_seen = initial_blocks;
    printf("  Initial leaf blocks: %d, dx_base=%.4f, dt=%.6f\n",
           initial_blocks, m->dx_base, p.dt);

    /* Set BH initial data */
    double mass = 1.0;
    double center[1][3] = {{0.0, 0.0, 0.0}};
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        set_brill_lindquist_global(b->grid, b->origin, 1, &mass,
                                   (const double(*)[3])center);
    }

    /* Evolve with regridding */
    p.time = 0.0;
    for (int step = 1; step <= nsteps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;

        if (step % p.amr.regrid_every == 0) {
            int delta = mesh_regrid(m, &p.amr);
            /* Update dt for finest-level CFL (global timestepping) */
            double dx_fine = m->dx_base / (1 << m->max_level);
            p.dt = p.CFL * dx_fine;
            int leaves = mesh_num_leaves(m);
            if (leaves > max_blocks_seen) max_blocks_seen = leaves;
            printf("  step %d: regrid delta=%d, leaves=%d, max_level=%d, dt=%.6f\n",
                   step, delta, leaves, m->max_level, p.dt);
        }
    }

    int final_blocks = mesh_num_leaves(m);
    double ham = mesh_constraint_l2(m);
    printf("  Final leaf blocks: %d (was %d, peak %d)\n",
           final_blocks, initial_blocks, max_blocks_seen);
    printf("  Final Ham L2 = %.6e\n", ham);

    check(isfinite(ham), "Ham L2 finite after regridding");
    check(ham < 1.0, "Ham L2 < 1.0 (bounded)");
    check(max_blocks_seen > initial_blocks,
          "Refinement triggered (peak blocks > initial)");

    mesh_free(m);
}

/* ======================================================================
 * Test 3: Flat spacetime with regridding (should not refine)
 *
 * Flat spacetime, single root block N_block=16, L=10, regrid_every=5.
 * Pass: Block count unchanged (chi-gradient = 0 -> no refinement).
 * ====================================================================== */
static void test_flat_no_refinement(void)
{
    printf("\n--- Test 3: Flat spacetime with regridding (no refinement) ---\n");

    sim_params_t p = default_params();
    p.L = 10.0;
    p.CFL = 0.25;
    int nsteps = 20;

    mesh_t *m = mesh_create(16, p.L, p.rk_method);
    p.dx = m->dx_base;
    p.dt = p.CFL * p.dx;
    p.amr.chi_refine  = 0.1;
    p.amr.chi_coarsen = 0.01;
    p.amr.max_level   = 2;
    p.amr.regrid_every = 5;

    int initial_blocks = mesh_num_leaves(m);

    /* Set flat spacetime on all blocks */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        set_flat_spacetime(b->grid);
    }

    /* Evolve with regridding */
    p.time = 0.0;
    for (int step = 1; step <= nsteps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;
        if (step % p.amr.regrid_every == 0)
            mesh_regrid(m, &p.amr);
    }

    int final_blocks = mesh_num_leaves(m);
    double ham = mesh_constraint_l2(m);
    printf("  Blocks: initial=%d, final=%d\n", initial_blocks, final_blocks);
    printf("  Final Ham L2 = %.6e\n", ham);

    check(final_blocks == initial_blocks,
          "Block count unchanged (flat -> no refinement)");
    check(ham < 1.0e-10, "Flat spacetime Ham L2 < 1e-10");

    mesh_free(m);
}

/* ====================================================================== */

int main(void)
{
    setbuf(stdout, NULL);

    printf("=== AMR Evolution Integration Test ===\n");

    backend_init();

    test_refined_vs_single_block();
    test_dynamic_regridding();
    test_flat_no_refinement();

    backend_cleanup();

    printf("\n=== Results: %d/%d passed ===\n", pass_count, test_count);
    printf("%s\n", (pass_count == test_count) ? "ALL PASSED" : "SOME FAILED");

    return (pass_count == test_count) ? 0 : 1;
}
