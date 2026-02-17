/*
 * Lattice — 3D Numerical Relativity
 * AMR integration test: mesh evolution with dynamic regridding.
 *
 * Tests:
 *   1. Uniform mesh vs single-grid BH evolution (same N_eff)
 *   2. Dynamic regridding around a BH (blocks increase)
 *   3. Flat spacetime with regridding (should not refine)
 *
 * Pass criteria:
 *   - Test 1: Ham L2 ratio within 2x
 *   - Test 2: No crash, Ham L2 bounded, block count increased
 *   - Test 3: Block count unchanged
 */

#include "../src/amr/mesh.h"
#include "../src/amr/refine.h"
#include "../src/amr/block.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/puncture.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/boundary/sommerfeld.h"
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
 * Test 1: Uniform mesh vs single-grid BH evolution
 *
 * Single BH (M=1) at origin, evolve 20 steps.
 * Compare: single grid N=32 vs 2x2x2 mesh of N_block=16 (same N_eff=32).
 * Pass: Ham L2 ratio within 2x.
 * ====================================================================== */
static void test_uniform_vs_single_grid(void)
{
    printf("\n--- Test 1: Uniform mesh vs single-grid BH evolution ---\n");

    sim_params_t p = default_params();
    p.L = 64.0;
    p.CFL = 0.25;
    int nsteps = 20;

    double mass = 1.0;
    double center[1][3] = {{0.0, 0.0, 0.0}};

    /* (a) Single-grid reference: N=32, L=64 */
    grid_t *gref = grid_alloc(32, p.L, RK_CK45);
    p.N  = gref->N;
    p.dx = gref->dx;
    p.dt = p.CFL * p.dx;

    set_brill_lindquist(gref, 1, &mass, (const double(*)[3])center);
    for (int step = 0; step < nsteps; step++)
        rk4_step(gref, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);

    double ham_ref = compute_constraint_l2(gref);
    printf("  Single-grid:  Ham L2 = %.6e (N=32, %d steps)\n",
           ham_ref, nsteps);

    /* (b) Multi-block mesh: 2x2x2 x 16^3 = same N_eff=32 */
    mesh_t *m = mesh_create(2, 16, p.L, RK_CK45);

    /* dx/dt from mesh (should match single-grid) */
    p.dx = m->dx_base;
    p.dt = p.CFL * p.dx;

    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        set_brill_lindquist_global(b->grid, b->origin, 1, &mass,
                                   (const double(*)[3])center);
    }

    for (int step = 0; step < nsteps; step++)
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);

    double ham_mesh = mesh_constraint_l2(m);
    printf("  Multi-block:  Ham L2 = %.6e (2x2x2 x 16^3, %d steps)\n",
           ham_mesh, nsteps);

    /* Both should be finite and bounded */
    check(isfinite(ham_ref), "Single-grid Ham L2 finite");
    check(isfinite(ham_mesh), "Multi-block Ham L2 finite");

    /* Ratio should be within 2x */
    double ratio = (ham_ref > 0) ? ham_mesh / ham_ref : 0.0;
    printf("  Ratio mesh/ref = %.4f\n", ratio);
    check(ratio > 0.5 && ratio < 2.0,
          "Multi-block Ham L2 within 2x of single-grid");

    grid_free(gref);
    mesh_free(m);
}

/* ======================================================================
 * Test 2: Dynamic regridding around a BH
 *
 * Single BH (M=1), 2x2x2 mesh of N_block=16, L=64.
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

    mesh_t *m = mesh_create(2, 16, p.L, RK_CK45);
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
    for (int step = 1; step <= nsteps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
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
 * Flat spacetime, 2x2x2 mesh of N_block=16, L=10, regrid_every=5.
 * Pass: Block count unchanged (chi-gradient = 0 -> no refinement).
 * ====================================================================== */
static void test_flat_no_refinement(void)
{
    printf("\n--- Test 3: Flat spacetime with regridding (no refinement) ---\n");

    sim_params_t p = default_params();
    p.L = 10.0;
    p.CFL = 0.25;
    int nsteps = 20;

    mesh_t *m = mesh_create(2, 16, p.L, RK_CK45);
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
    for (int step = 1; step <= nsteps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
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

    test_uniform_vs_single_grid();
    test_dynamic_regridding();
    test_flat_no_refinement();

    backend_cleanup();

    printf("\n=== Results: %d/%d passed ===\n", pass_count, test_count);
    printf("%s\n", (pass_count == test_count) ? "ALL PASSED" : "SOME FAILED");

    return (pass_count == test_count) ? 0 : 1;
}
