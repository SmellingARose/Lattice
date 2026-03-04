/*
 * Lattice — 3D Numerical Relativity
 * Checkpoint/restart validation test.
 *
 * Tests:
 *   1. Single BH on uniform grid: evolve 20 steps, checkpoint at step 10,
 *      restart from checkpoint, verify fields match continuous evolution.
 *   2. Single BH with AMR: evolve 20 steps, checkpoint at step 10,
 *      restart, verify constraints and lapse match.
 *
 * The checkpoint format follows the pattern of Cactus/CarpetIOHDF5 and
 * Einstein Toolkit checkpoint/restart (binary state dump at user-specified
 * intervals). Ref: Loeffler et al. 2012 (Einstein Toolkit).
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/puncture.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/amr/mesh.h"
#include "../src/amr/refine.h"
#include "../src/amr/ghost_exchange.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/io/checkpoint.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int passed = 0, failed = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while(0)

/* ----------------------------------------------------------------
 * Test 1: Uniform grid checkpoint/restart
 * ---------------------------------------------------------------- */
static void test_uniform_checkpoint(void)
{
    printf("\n--- Test 1: Uniform Grid Checkpoint/Restart ---\n");

    backend_init();

    sim_params_t p = default_params();
    p.L = 16.0;
    p.rk_method = RK_CLASSIC;

    /* Create mesh and set single BH initial data */
    mesh_t *m = mesh_create(32, p.L, p.rk_method);
    p.dx = m->dx_base;
    p.dt = p.CFL * p.dx;

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    set_bowen_york_mesh(m, 1, &bh, 0);

    double ham0 = mesh_constraint_l2(m);
    printf("  Initial Ham L2 = %.6e\n", ham0);

    /* Evolve 10 steps */
    p.time = 0.0;
    for (int step = 1; step <= 10; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;
    }

    double ham10 = mesh_constraint_l2(m);
    printf("  Step 10 Ham L2 = %.6e\n", ham10);

    /* Save checkpoint */
    const char *ckpt_path = "build/test_checkpoint_uniform.lat";
    int rc = checkpoint_write(m, &p, 10, ckpt_path);
    CHECK(rc == 0, "Checkpoint write succeeds");

    /* Continue evolving to step 20 (reference) */
    for (int step = 11; step <= 20; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;
    }

    double ham20_ref = mesh_constraint_l2(m);
    double lapse20_ref = m->blocks[0]->grid->fields[FIELD_LAPSE]
        [IDX(m->blocks[0]->grid,
             m->blocks[0]->grid->ghost + m->blocks[0]->grid->N / 2,
             m->blocks[0]->grid->ghost + m->blocks[0]->grid->N / 2,
             m->blocks[0]->grid->ghost + m->blocks[0]->grid->N / 2)];
    printf("  Step 20 (ref) Ham L2 = %.6e, center lapse = %.6f\n",
           ham20_ref, lapse20_ref);

    mesh_free(m);

    /* Restart from checkpoint */
    mesh_t *m2 = NULL;
    sim_params_t p2;
    int restart_step = 0;

    rc = checkpoint_read(ckpt_path, &m2, &p2, &restart_step);
    CHECK(rc == 0, "Checkpoint read succeeds");
    CHECK(restart_step == 10, "Restart step correct (10)");
    CHECK(fabs(p2.time - 10 * p.dt / 2.0) < 1e-10 ||
          fabs(p2.time - p.dt * 10.0 / p.dt * p.dt) < 1e-6,
          "Restart time reasonable");

    /* Recompute dx/dt from restored params */
    p2.dx = m2->dx_base;
    p2.dt = p2.CFL * p2.dx;

    double ham10_restart = mesh_constraint_l2(m2);
    printf("  Step 10 (restart) Ham L2 = %.6e\n", ham10_restart);
    CHECK(fabs(ham10_restart - ham10) / (ham10 + 1e-15) < 1e-10,
          "Restart Ham L2 matches checkpoint");

    /* Evolve restarted mesh to step 20 */
    for (int step = restart_step + 1; step <= 20; step++) {
        rk4_step_mesh(m2, &p2, ccz4_rhs_point, p2.dt);
        p2.time += p2.dt;
    }

    double ham20_restart = mesh_constraint_l2(m2);
    double lapse20_restart = m2->blocks[0]->grid->fields[FIELD_LAPSE]
        [IDX(m2->blocks[0]->grid,
             m2->blocks[0]->grid->ghost + m2->blocks[0]->grid->N / 2,
             m2->blocks[0]->grid->ghost + m2->blocks[0]->grid->N / 2,
             m2->blocks[0]->grid->ghost + m2->blocks[0]->grid->N / 2)];
    printf("  Step 20 (restart) Ham L2 = %.6e, center lapse = %.6f\n",
           ham20_restart, lapse20_restart);

    /* Verify match: bitwise identical since same FP operations */
    CHECK(ham20_restart == ham20_ref,
          "Restart Ham L2 matches reference exactly");
    CHECK(lapse20_restart == lapse20_ref,
          "Restart center lapse matches reference exactly");

    mesh_free(m2);
}

/* ----------------------------------------------------------------
 * Test 2: AMR checkpoint/restart
 * ---------------------------------------------------------------- */
static void test_amr_checkpoint(void)
{
    printf("\n--- Test 2: AMR Checkpoint/Restart ---\n");

    sim_params_t p = default_params();
    p.L = 32.0;
    p.rk_method = RK_CLASSIC;

    /* Create mesh and refine near BH */
    mesh_t *m = mesh_create(16, p.L, p.rk_method);
    p.dx = m->dx_base;
    p.dt = p.CFL * p.dx;

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    set_bowen_york_mesh(m, 1, &bh, 2);  /* 2 AMR levels */

    int leaves0 = mesh_num_leaves(m);
    double ham0 = mesh_constraint_l2(m);
    printf("  Initial: %d leaves, Ham L2 = %.6e\n", leaves0, ham0);

    CHECK(leaves0 > 1, "AMR mesh has refinement");

    /* Evolve 10 steps */
    p.time = 0.0;
    for (int step = 1; step <= 10; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;
    }

    double ham10 = mesh_constraint_l2(m);
    int leaves10 = mesh_num_leaves(m);
    printf("  Step 10: %d leaves, Ham L2 = %.6e\n", leaves10, ham10);

    /* Checkpoint */
    const char *ckpt_path = "build/test_checkpoint_amr.lat";
    int rc = checkpoint_write(m, &p, 10, ckpt_path);
    CHECK(rc == 0, "AMR checkpoint write succeeds");

    /* Continue to step 20 (reference) */
    for (int step = 11; step <= 20; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;
    }

    double ham20_ref = mesh_constraint_l2(m);
    printf("  Step 20 (ref) Ham L2 = %.6e\n", ham20_ref);

    mesh_free(m);

    /* Restart */
    mesh_t *m2 = NULL;
    sim_params_t p2;
    int restart_step = 0;

    rc = checkpoint_read(ckpt_path, &m2, &p2, &restart_step);
    CHECK(rc == 0, "AMR checkpoint read succeeds");
    CHECK(restart_step == 10, "AMR restart step correct");

    int leaves_restart = mesh_num_leaves(m2);
    CHECK(leaves_restart == leaves10,
          "AMR leaf count matches after restart");

    printf("  Restart: %d leaves, max_level=%d\n",
           leaves_restart, m2->max_level);

    /* Recompute derived params */
    p2.dx = m2->dx_base;
    p2.dt = p2.CFL * p2.dx;

    double ham10_restart = mesh_constraint_l2(m2);
    printf("  Step 10 (restart) Ham L2 = %.6e\n", ham10_restart);
    CHECK(fabs(ham10_restart - ham10) / (ham10 + 1e-15) < 1e-10,
          "AMR restart Ham L2 matches checkpoint");

    /* Evolve to step 20 */
    for (int step = restart_step + 1; step <= 20; step++) {
        rk4_step_mesh(m2, &p2, ccz4_rhs_point, p2.dt);
        p2.time += p2.dt;
    }

    double ham20_restart = mesh_constraint_l2(m2);
    printf("  Step 20 (restart) Ham L2 = %.6e\n", ham20_restart);

    /* AMR restart may have sub-epsilon differences from ghost exchange
     * reconstruction (restriction/prolongation round-trip). Use relative
     * tolerance instead of bitwise equality. */
    double rel_err = fabs(ham20_restart - ham20_ref) / (ham20_ref + 1e-15);
    printf("  Relative error = %.6e\n", rel_err);
    CHECK(rel_err < 1e-10,
          "AMR restart Ham L2 matches reference (rel < 1e-10)");

    mesh_free(m2);
}

int main(void)
{
    printf("=== Checkpoint/Restart Test Suite ===\n");

    test_uniform_checkpoint();
    test_amr_checkpoint();

    /* Clean up checkpoint files */
    remove("build/test_checkpoint_uniform.lat");
    remove("build/test_checkpoint_amr.lat");

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
