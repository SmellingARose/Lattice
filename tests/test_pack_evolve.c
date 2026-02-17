/*
 * Lattice — 3D Numerical Relativity
 * Packed batch kernel validation test.
 *
 * Verifies that the packed mesh stepper (all leaf blocks batched into
 * a single meshblock_pack_t with batched kernels) produces identical
 * results to the per-block stepper (one kernel per block per stage).
 *
 * Tests:
 *   1. Packed vs per-block: identical field values after 10 steps (BH)
 *   2. Packed with multilevel AMR: regridding + evolution works
 *   3. Packed flat spacetime: constraint stability < 1e-10
 *
 * Pass criteria:
 *   - Test 1: max |diff| < 1e-12 between packed and per-block fields
 *   - Test 2: Ham L2 finite and bounded, refinement triggered
 *   - Test 3: Ham L2 < 1e-10 after 100 steps
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
 * Test 1: Packed vs per-block identical results
 *
 * Single BH (M=1) at origin, 2x2x2 mesh, N_block=16, 10 steps.
 * Run the packed stepper on one mesh and the per-block stepper on
 * an identically-initialized second mesh. Compare all field values.
 *
 * The packed path packs all blocks into a contiguous buffer and uses
 * batched kernels; the per-block path launches one kernel per block.
 * Both should produce bit-identical results (same floating-point ops).
 *
 * Pass: max |diff| < 1e-12 between all field values.
 * ====================================================================== */
static void test_packed_vs_perblock(void)
{
    printf("\n--- Test 1: Packed vs per-block identical results ---\n");

    sim_params_t p = default_params();
    p.L = 64.0;
    p.CFL = 0.25;
    int nsteps = 10;

    double mass = 1.0;
    double center[1][3] = {{0.0, 0.0, 0.0}};

    /* Create two identical meshes: 2x2x2 blocks of N_block=16 */
    mesh_t *m_packed  = mesh_create(2, 16, p.L, RK_CK45);
    mesh_t *m_perblk  = mesh_create(2, 16, p.L, RK_CK45);

    /* Both share the same dx and dt */
    p.dx = m_packed->dx_base;
    p.dt = p.CFL * p.dx;

    /* Set identical BH initial data on both meshes */
    for (int bid = 0; bid < m_packed->num_blocks; bid++) {
        block_t *bp = m_packed->blocks[bid];
        block_t *bb = m_perblk->blocks[bid];
        if (!bp || !bp->is_leaf) continue;

        set_brill_lindquist_global(bp->grid, bp->origin, 1, &mass,
                                   (const double(*)[3])center);
        set_brill_lindquist_global(bb->grid, bb->origin, 1, &mass,
                                   (const double(*)[3])center);
    }

    /* Evolve: packed stepper on m_packed, per-block on m_perblk */
    for (int step = 0; step < nsteps; step++) {
        rk4_step_mesh(m_packed, &p, ccz4_rhs_point, p.dt);
        rk4_step_mesh_perblock(m_perblk, &p, ccz4_rhs_point, p.dt);
    }

    /* Compare all field values across all leaf blocks.
     * Both meshes have the same block structure (no regridding),
     * so blocks[bid] corresponds 1:1. */
    double max_diff = 0.0;
    int max_field = 0, max_block = 0;
    for (int bid = 0; bid < m_packed->num_blocks; bid++) {
        block_t *bp = m_packed->blocks[bid];
        block_t *bb = m_perblk->blocks[bid];
        if (!bp || !bp->is_leaf) continue;

        for (int f = 0; f < NUM_FIELDS; f++) {
            for (size_t idx = 0; idx < bp->grid->npoints; idx++) {
                double diff = fabs(bp->grid->fields[f][idx]
                                 - bb->grid->fields[f][idx]);
                if (diff > max_diff) {
                    max_diff = diff;
                    max_field = f;
                    max_block = bid;
                }
            }
        }
    }

    printf("  Max |diff| = %.6e (field %d, block %d)\n",
           max_diff, max_field, max_block);

    /* Both should be finite */
    double ham_packed = mesh_constraint_l2(m_packed);
    double ham_perblk = mesh_constraint_l2(m_perblk);
    printf("  Packed Ham L2 = %.6e, Per-block Ham L2 = %.6e\n",
           ham_packed, ham_perblk);

    check(isfinite(ham_packed), "Packed Ham L2 finite");
    check(isfinite(ham_perblk), "Per-block Ham L2 finite");
    check(max_diff < 1.0e-12, "Packed vs per-block max diff < 1e-12");

    mesh_free(m_packed);
    mesh_free(m_perblk);
}

/* ======================================================================
 * Test 2: Packed with multilevel (AMR regridding)
 *
 * Single BH (M=1), 2x2x2 mesh, N_block=16, chi_refine=0.05.
 * Evolve 20 steps with regridding every 5 steps.
 * The packed stepper handles multi-level blocks (fine + coarse ghost
 * exchange via CPU fallback in Commit 1).
 *
 * Pass: No crash, Ham L2 finite and bounded, refinement triggered.
 * ====================================================================== */
static void test_packed_multilevel(void)
{
    printf("\n--- Test 2: Packed with multilevel (AMR regridding) ---\n");

    sim_params_t p = default_params();
    p.L = 64.0;
    p.CFL = 0.25;
    int nsteps = 20;

    mesh_t *m = mesh_create(2, 16, p.L, RK_CK45);
    p.dx = m->dx_base;
    p.dt = p.CFL * p.dx;
    p.amr.chi_refine  = 0.05;
    p.amr.chi_coarsen = 0.0;   /* disable coarsening */
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

    /* Evolve with regridding — uses packed stepper via rk4_step_mesh */
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
 * Test 3: Packed flat spacetime stability
 *
 * Flat Minkowski spacetime, 2x2x2 mesh, N_block=16, 100 steps.
 * The packed stepper should maintain constraint violation < 1e-10,
 * identical to the per-block stepper on flat spacetime.
 *
 * Pass: Ham L2 < 1e-10 after 100 steps.
 * ====================================================================== */
static void test_packed_flat_stability(void)
{
    printf("\n--- Test 3: Packed flat spacetime stability ---\n");

    sim_params_t p = default_params();
    p.L = 10.0;
    p.CFL = 0.25;
    int nsteps = 100;

    mesh_t *m = mesh_create(2, 16, p.L, RK_CK45);
    p.dx = m->dx_base;
    p.dt = p.CFL * p.dx;

    /* Set flat spacetime on all blocks */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        set_flat_spacetime(b->grid);
    }

    /* Evolve 100 steps with packed stepper */
    for (int step = 0; step < nsteps; step++)
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);

    double ham = mesh_constraint_l2(m);
    printf("  Ham L2 after %d steps = %.6e\n", nsteps, ham);

    check(isfinite(ham), "Ham L2 finite");
    check(ham < 1.0e-10, "Ham L2 < 1e-10 (flat spacetime stable)");

    mesh_free(m);
}

/* ====================================================================== */

int main(void)
{
    setbuf(stdout, NULL);

    printf("=== Packed Batch Kernel Validation Test ===\n");

    backend_init();

    test_packed_vs_perblock();
    test_packed_multilevel();
    test_packed_flat_stability();

    backend_cleanup();

    printf("\n=== Results: %d/%d passed ===\n", pass_count, test_count);
    printf("%s\n", (pass_count == test_count) ? "ALL PASSED" : "SOME FAILED");

    return (pass_count == test_count) ? 0 : 1;
}
