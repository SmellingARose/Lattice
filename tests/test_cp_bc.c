/*
 * Lattice — 3D Numerical Relativity
 * Constraint-preserving boundary condition tests.
 *
 * Verifies:
 *   1. cp_char_speed() returns correct speeds for each field type
 *   2. Gamma normal/tangential speed selection by face_dir
 *   3. cp_rhs() formula against hand-computed values
 *   4. Flat spacetime stability with CP BCs (1000 steps, Ham < 1e-10)
 *   5. Single BH constraint comparison: CP <= Sommerfeld at t=50M
 *   6. Lapse clamping: small alpha doesn't blow up
 *
 * Ref: arXiv:1212.2901 (Hilditch et al., BAM)
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/boundary/constraint_preserving.h"
#include "../src/initial_data/puncture.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/amr/mesh.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, fmt, ...) do {                                        \
    if (cond) { printf("  PASS: " fmt "\n", ##__VA_ARGS__); n_pass++; }   \
    else      { printf("  FAIL: " fmt "\n", ##__VA_ARGS__); n_fail++; }   \
} while(0)

#define NEAR(a, b, tol) (fabs((a) - (b)) < (tol))

/* ---- Test 1: Characteristic speed values ---- */
static void test_char_speeds(void)
{
    printf("\n--- Test 1: Characteristic speed values ---\n");
    double alpha = 1.0;  /* flat-space lapse */

    /* Theta → 1.0 */
    CHECK(NEAR(cp_char_speed(FIELD_THETA, 0, alpha), 1.0, 1e-14),
          "Theta speed = 1.0");

    /* K → sqrt(2/alpha) = sqrt(2) for alpha=1 */
    CHECK(NEAR(cp_char_speed(FIELD_K, 0, alpha), sqrt(2.0), 1e-14),
          "K speed = sqrt(2) at alpha=1");

    /* K at alpha=0.5 → sqrt(2/0.5) = 2.0 */
    CHECK(NEAR(cp_char_speed(FIELD_K, 0, 0.5), 2.0, 1e-14),
          "K speed = 2.0 at alpha=0.5");

    /* A_ij → 1.0 */
    CHECK(NEAR(cp_char_speed(FIELD_A11, 0, alpha), 1.0, 1e-14),
          "A11 speed = 1.0");
    CHECK(NEAR(cp_char_speed(FIELD_A33, 0, alpha), 1.0, 1e-14),
          "A33 speed = 1.0");

    /* Gamma^1 with face_dir=0 (normal) → sqrt(3/4) */
    CHECK(NEAR(cp_char_speed(FIELD_GAMMA1, 0, alpha), sqrt(0.75), 1e-14),
          "Gamma1 normal (face_dir=0) = sqrt(3/4)");

    /* Gamma^1 with face_dir=1 (tangential) → 1.0 */
    CHECK(NEAR(cp_char_speed(FIELD_GAMMA1, 1, alpha), 1.0, 1e-14),
          "Gamma1 tangential (face_dir=1) = 1.0");

    /* Non-CP fields → 0.0 */
    CHECK(NEAR(cp_char_speed(FIELD_CHI, 0, alpha), 0.0, 1e-14),
          "chi speed = 0 (non-CP)");
    CHECK(NEAR(cp_char_speed(FIELD_LAPSE, 0, alpha), 0.0, 1e-14),
          "lapse speed = 0 (non-CP)");
    CHECK(NEAR(cp_char_speed(FIELD_H11, 0, alpha), 0.0, 1e-14),
          "h11 speed = 0 (non-CP)");
    CHECK(NEAR(cp_char_speed(FIELD_SHIFT1, 0, alpha), 0.0, 1e-14),
          "shift1 speed = 0 (non-CP)");
    CHECK(NEAR(cp_char_speed(FIELD_B1, 0, alpha), 0.0, 1e-14),
          "B1 speed = 0 (non-CP)");
}

/* ---- Test 2: Gamma normal/tangential for each face ---- */
static void test_gamma_face_dirs(void)
{
    printf("\n--- Test 2: Gamma normal/tangential by face ---\n");
    double alpha = 1.0;
    double v_n = sqrt(0.75);

    /* face_dir=0 (X face): Gamma1=normal, Gamma2,3=tangential */
    CHECK(NEAR(cp_char_speed(FIELD_GAMMA1, 0, alpha), v_n, 1e-14),
          "X-face: Gamma1 normal");
    CHECK(NEAR(cp_char_speed(FIELD_GAMMA2, 0, alpha), 1.0, 1e-14),
          "X-face: Gamma2 tangential");
    CHECK(NEAR(cp_char_speed(FIELD_GAMMA3, 0, alpha), 1.0, 1e-14),
          "X-face: Gamma3 tangential");

    /* face_dir=1 (Y face): Gamma2=normal, Gamma1,3=tangential */
    CHECK(NEAR(cp_char_speed(FIELD_GAMMA1, 1, alpha), 1.0, 1e-14),
          "Y-face: Gamma1 tangential");
    CHECK(NEAR(cp_char_speed(FIELD_GAMMA2, 1, alpha), v_n, 1e-14),
          "Y-face: Gamma2 normal");
    CHECK(NEAR(cp_char_speed(FIELD_GAMMA3, 1, alpha), 1.0, 1e-14),
          "Y-face: Gamma3 tangential");

    /* face_dir=2 (Z face): Gamma3=normal, Gamma1,2=tangential */
    CHECK(NEAR(cp_char_speed(FIELD_GAMMA1, 2, alpha), 1.0, 1e-14),
          "Z-face: Gamma1 tangential");
    CHECK(NEAR(cp_char_speed(FIELD_GAMMA2, 2, alpha), 1.0, 1e-14),
          "Z-face: Gamma2 tangential");
    CHECK(NEAR(cp_char_speed(FIELD_GAMMA3, 2, alpha), v_n, 1e-14),
          "Z-face: Gamma3 normal");
}

/* ---- Test 3: CP RHS formula ---- */
static void test_cp_rhs_formula(void)
{
    printf("\n--- Test 3: CP RHS formula ---\n");

    /* Known inputs:
     *   alpha=1.0, speed=1.0, s_sign=+1, df_ds=0.5,
     *   f_val=0.1, f_asymp=0.0, r=5.0
     *
     *   rhs = -1.0 * 1.0 * (+1) * 0.5 - 1.0 * (0.1 - 0.0) / 5.0
     *       = -0.5 - 0.02 = -0.52 */
    double val = cp_rhs(1.0, 1.0, +1, 0.5, 0.1, 0.0, 5.0);
    CHECK(NEAR(val, -0.52, 1e-14), "CP RHS: alpha=1, speed=1, s=+1 → -0.52 (got %.6f)", val);

    /* s_sign=-1:
     *   rhs = -1.0 * 1.0 * (-1) * 0.5 - 1.0 * (0.1) / 5.0
     *       = +0.5 - 0.02 = +0.48 */
    val = cp_rhs(1.0, 1.0, -1, 0.5, 0.1, 0.0, 5.0);
    CHECK(NEAR(val, 0.48, 1e-14), "CP RHS: s_sign=-1 → +0.48 (got %.6f)", val);

    /* speed=sqrt(2), alpha=0.5:
     *   rhs = -0.5 * sqrt(2) * (+1) * 0.5 - 0.5 * (0.1) / 5.0
     *       = -0.5*sqrt(2)*0.5 - 0.01 = -0.25*sqrt(2) - 0.01 */
    double expected = -0.25 * sqrt(2.0) - 0.01;
    val = cp_rhs(0.5, sqrt(2.0), +1, 0.5, 0.1, 0.0, 5.0);
    CHECK(NEAR(val, expected, 1e-14), "CP RHS: alpha=0.5, speed=sqrt(2) → %.6f (got %.6f)",
          expected, val);

    /* Asymptotic value test: f_val = f_asymp → falloff term = 0 */
    val = cp_rhs(1.0, 1.0, +1, 0.3, 0.0, 0.0, 10.0);
    CHECK(NEAR(val, -0.3, 1e-14), "CP RHS: f_val=f_asymp → only wave term (got %.6f)", val);
}

/* ---- Test 4: Flat spacetime stability with CP BCs ---- */
static void test_flat_stability_cp(void)
{
    printf("\n--- Test 4: Flat spacetime stability (CP BCs) ---\n");

    sim_params_t p = default_params();
    p.N         = 32;
    p.L         = 10.0;
    p.CFL       = 0.25;
    p.num_steps = 1000;
    p.sigma     = 0.3;
    p.bc_type   = BC_CONSTRAINT_PRESERVING;
    p.dx = p.L / p.N;
    p.dt = p.CFL * p.dx;

    mesh_t *m = mesh_create_ex(1, p.N, p.L, p.rk_method, NUM_CCZ4_FIELDS);
    grid_t *g = m->blocks[0]->grid;
    p.N  = g->N;
    p.dx = g->dx;
    p.dt = p.CFL * p.dx;

    set_flat_spacetime(g);

    p.time = 0.0;
    for (int step = 1; step <= p.num_steps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;
    }

    double ham = mesh_constraint_l2(m);
    printf("  Final Ham L2 = %.6e (after %d steps)\n", ham, p.num_steps);
    CHECK(ham < 1.0e-10, "Flat spacetime CP BCs: Ham < 1e-10 (%.2e)", ham);

    mesh_free(m);
}

/* ---- Test 5: Single BH constraint comparison ---- */
static void test_single_bh_comparison(void)
{
    printf("\n--- Test 5: Single BH constraint comparison ---\n");

    /* Common setup */
    int N = 32;
    double L = 16.0;
    double CFL = 0.25;
    int num_steps = 200;  /* ~50M at this resolution */

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.center[0] = 0.0; bh.center[1] = 0.0; bh.center[2] = 0.0;

    /* Run with Sommerfeld BCs */
    sim_params_t p_som = default_params();
    p_som.N = N; p_som.L = L; p_som.CFL = CFL; p_som.num_steps = num_steps;
    p_som.bc_type = BC_SOMMERFELD;
    p_som.dx = L / N; p_som.dt = CFL * p_som.dx;

    mesh_t *m_som = mesh_create_ex(1, N, L, p_som.rk_method, NUM_CCZ4_FIELDS);
    p_som.N = m_som->blocks[0]->grid->N;
    p_som.dx = m_som->blocks[0]->grid->dx;
    p_som.dt = CFL * p_som.dx;

    set_bowen_york_mesh(m_som, 1, &bh, 0);

    p_som.time = 0.0;
    for (int step = 1; step <= num_steps; step++) {
        rk4_step_mesh(m_som, &p_som, ccz4_rhs_point, p_som.dt);
        p_som.time += p_som.dt;
    }
    double ham_som = mesh_constraint_l2(m_som);
    printf("  Sommerfeld Ham L2 = %.6e (t=%.2f)\n", ham_som, p_som.time);
    mesh_free(m_som);

    /* Run with CP BCs */
    sim_params_t p_cp = default_params();
    p_cp.N = N; p_cp.L = L; p_cp.CFL = CFL; p_cp.num_steps = num_steps;
    p_cp.bc_type = BC_CONSTRAINT_PRESERVING;
    p_cp.dx = L / N; p_cp.dt = CFL * p_cp.dx;

    mesh_t *m_cp = mesh_create_ex(1, N, L, p_cp.rk_method, NUM_CCZ4_FIELDS);
    p_cp.N = m_cp->blocks[0]->grid->N;
    p_cp.dx = m_cp->blocks[0]->grid->dx;
    p_cp.dt = CFL * p_cp.dx;

    set_bowen_york_mesh(m_cp, 1, &bh, 0);

    p_cp.time = 0.0;
    for (int step = 1; step <= num_steps; step++) {
        rk4_step_mesh(m_cp, &p_cp, ccz4_rhs_point, p_cp.dt);
        p_cp.time += p_cp.dt;
    }
    double ham_cp = mesh_constraint_l2(m_cp);
    printf("  CP Ham L2 = %.6e (t=%.2f)\n", ham_cp, p_cp.time);
    mesh_free(m_cp);

    /* CP should preserve constraints at least as well as Sommerfeld */
    CHECK(ham_cp <= ham_som * 1.05,
          "CP constraint <= Sommerfeld (%.2e <= %.2e)", ham_cp, ham_som);
}

/* ---- Test 6: Lapse clamping ---- */
static void test_lapse_clamping(void)
{
    printf("\n--- Test 6: Lapse clamping ---\n");

    /* Very small alpha should be clamped to 0.01, giving sqrt(2/0.01) = sqrt(200) */
    double speed_tiny = cp_char_speed(FIELD_K, 0, 1.0e-6);
    double expected = sqrt(2.0 / 0.01);
    CHECK(NEAR(speed_tiny, expected, 1e-10),
          "K speed at alpha=1e-6 clamped to sqrt(200) = %.4f (got %.4f)",
          expected, speed_tiny);

    /* alpha=0 should also be clamped */
    double speed_zero = cp_char_speed(FIELD_K, 0, 0.0);
    CHECK(NEAR(speed_zero, expected, 1e-10),
          "K speed at alpha=0 clamped (got %.4f)", speed_zero);

    /* Normal alpha should not be clamped */
    double speed_normal = cp_char_speed(FIELD_K, 0, 1.0);
    CHECK(NEAR(speed_normal, sqrt(2.0), 1e-14),
          "K speed at alpha=1.0 not clamped (got %.6f)", speed_normal);
}

int main(void)
{
    printf("=== Constraint-Preserving BC Tests ===\n");
    backend_init();

    test_char_speeds();
    test_gamma_face_dirs();
    test_cp_rhs_formula();
    test_flat_stability_cp();
    test_single_bh_comparison();
    test_lapse_clamping();

    backend_cleanup();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
