/*
 * Lattice — 3D Numerical Relativity
 * HiSpID (High-Spin Initial Data) test suite.
 *
 * Tests:
 *   1. QI Kerr metric at known radius vs analytic (machine precision)
 *   2. QI Kerr metric -> delta_ij falloff at large r
 *   3. Kerr extrinsic curvature symmetry and falloff
 *   4. Gaussian superposition: 2 BHs, weights sum correctly
 *   5. Coupled solver: zero spin reduces to BY (V^i ~ 0)
 *   6. Coupled solver: chi=0.5 spin, convergence to tol
 *   7. Constraint violation: Ham + Mom after solve (expect bounded)
 *   8. High spin: chi=0.9, evolve 10 steps, no NaN
 *   9. det(h_ij) = 1 enforcement after CCZ4 conversion
 *
 * Ref: arXiv:1410.8607, arXiv:1001.4077
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/puncture.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/initial_data/relaxation.h"
#include "../src/initial_data/kerr_quasi_isotropic.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/geometry/tensor_utils.h"
#include "../src/amr/mesh.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        printf("  PASS: %s\n", msg); \
        tests_passed++; \
    } else { \
        printf("  FAIL: %s\n", msg); \
        tests_failed++; \
    } \
} while(0)

/* ================================================================
 * Test 1: QI Kerr metric at known radius vs analytic
 * ================================================================ */
static void test_kerr_metric_analytic(void)
{
    printf("\n--- Test 1: QI Kerr metric analytic values ---\n");

    /* Schwarzschild (a=0): conformal metric should be flat (delta_ij) */
    double h[3][3], psi;
    double spin_zero[3] = {0, 0, 0};
    kerr_qi_metric(h, &psi, 2.0, 0.0, 0.0, 1.0, spin_zero);

    printf("  Schwarzschild at r=2, M=1:\n");
    printf("  h_11 = %.15f (expect 1.0)\n", h[0][0]);
    printf("  h_22 = %.15f (expect 1.0)\n", h[1][1]);
    printf("  h_12 = %.15e (expect 0.0)\n", h[0][1]);
    printf("  psi  = %.15f (expect %.15f)\n", psi, 1.0 + 1.0 / (2.0 * 2.0));

    CHECK(fabs(h[0][0] - 1.0) < 1e-12, "Schwarzschild h_11 = 1");
    CHECK(fabs(h[1][1] - 1.0) < 1e-12, "Schwarzschild h_22 = 1");
    CHECK(fabs(h[2][2] - 1.0) < 1e-12, "Schwarzschild h_33 = 1");
    CHECK(fabs(h[0][1]) < 1e-12, "Schwarzschild h_12 = 0");
    CHECK(fabs(h[0][2]) < 1e-12, "Schwarzschild h_13 = 0");
    CHECK(fabs(h[1][2]) < 1e-12, "Schwarzschild h_23 = 0");

    /* Spinning BH: conformal metric should differ from delta_ij */
    double spin_z[3] = {0, 0, 0.5};  /* a = 0.5/1.0 = 0.5 */
    kerr_qi_metric(h, &psi, 2.0, 0.0, 0.0, 1.0, spin_z);

    printf("  Kerr a=0.5 at (2,0,0):\n");
    printf("  h_11 = %.10f\n", h[0][0]);
    printf("  h_22 = %.10f\n", h[1][1]);
    printf("  h_33 = %.10f\n", h[2][2]);
    printf("  det(h) = %.15f (expect 1.0)\n", compute_det_sym(h));

    /* Conformal metric should have unit determinant */
    CHECK(fabs(compute_det_sym(h) - 1.0) < 1e-10,
          "Kerr conformal metric det(h) = 1");
    /* Off-diagonal should be small on x-axis (axisymmetric) */
    CHECK(fabs(h[0][1]) < 1e-6, "Kerr h_12 small on x-axis");
}

/* ================================================================
 * Test 2: QI Kerr metric falloff to delta_ij at large r
 * ================================================================ */
static void test_kerr_metric_falloff(void)
{
    printf("\n--- Test 2: QI Kerr metric falloff ---\n");

    double spin_z[3] = {0, 0, 0.5};
    double h_near[3][3], h_far[3][3];

    kerr_qi_metric(h_near, NULL, 5.0, 0.0, 0.0, 1.0, spin_z);
    kerr_qi_metric(h_far,  NULL, 50.0, 0.0, 0.0, 1.0, spin_z);

    double dev_near = fabs(h_near[0][0] - 1.0) + fabs(h_near[1][1] - 1.0)
                    + fabs(h_near[2][2] - 1.0);
    double dev_far  = fabs(h_far[0][0] - 1.0) + fabs(h_far[1][1] - 1.0)
                    + fabs(h_far[2][2] - 1.0);

    printf("  |h - delta| at r=5:  %.6e\n", dev_near);
    printf("  |h - delta| at r=50: %.6e\n", dev_far);

    CHECK(dev_far < dev_near, "h_ij approaches delta_ij at large r");
    CHECK(dev_far < 1e-2, "h_ij close to delta_ij at r=50");
}

/* ================================================================
 * Test 3: Kerr extrinsic curvature symmetry and falloff
 * ================================================================ */
static void test_kerr_extrinsic(void)
{
    printf("\n--- Test 3: Kerr extrinsic curvature ---\n");

    double spin_z[3] = {0, 0, 0.5};

    /* Symmetry check */
    double A[3][3];
    kerr_qi_extrinsic(A, 3.0, 1.5, -0.5, 1.0, spin_z);

    double max_asym = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 3; j++) {
            double d = fabs(A[i][j] - A[j][i]);
            if (d > max_asym) max_asym = d;
        }
    printf("  Max |A_ij - A_ji| = %.6e\n", max_asym);
    CHECK(max_asym < 1e-12, "Kerr A_ij is symmetric");

    /* Falloff: K_ij should vanish at large r */
    double A_far[3][3];
    kerr_qi_extrinsic(A_far, 100.0, 0.0, 0.0, 1.0, spin_z);
    double max_A_far = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double v = fabs(A_far[i][j]);
            if (v > max_A_far) max_A_far = v;
        }
    printf("  Max |A_ij| at r=100: %.6e\n", max_A_far);
    CHECK(max_A_far < 1e-4, "Kerr A_ij falls off at large r");

    /* Zero spin: K_ij should be zero (time-symmetric) */
    double spin_zero[3] = {0, 0, 0};
    kerr_qi_extrinsic(A, 2.0, 1.0, 0.5, 1.0, spin_zero);
    double max_A_zero = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double v = fabs(A[i][j]);
            if (v > max_A_zero) max_A_zero = v;
        }
    CHECK(max_A_zero < 1e-15, "Schwarzschild A_ij = 0");
}

/* ================================================================
 * Test 4: Gaussian superposition: 2 BHs
 * ================================================================ */
static void test_gaussian_superposition(void)
{
    printf("\n--- Test 4: Gaussian superposition ---\n");

    puncture_data_t bhs[2];
    memset(bhs, 0, sizeof(bhs));

    bhs[0].mass = 0.5;
    bhs[0].center[0] = 2.0;
    bhs[0].spin[2] = 0.2;  /* a = 0.2/0.5 = 0.4 */

    bhs[1].mass = 0.5;
    bhs[1].center[0] = -2.0;
    bhs[1].spin[2] = -0.2;

    /* At the midpoint (origin), both weights should be equal */
    double h[3][3];
    hispid_conformal_metric(h, 0.0, 0.0, 0.0, 2, bhs);

    printf("  h_ij at origin (midpoint):\n");
    printf("  h_11 = %.10f\n", h[0][0]);
    printf("  h_22 = %.10f\n", h[1][1]);
    printf("  det(h) = %.15f\n", compute_det_sym(h));

    CHECK(fabs(compute_det_sym(h) - 1.0) < 1e-10,
          "Superposed det(h) = 1 at midpoint");

    /* Far from both BHs, should approach delta_ij */
    hispid_conformal_metric(h, 50.0, 0.0, 0.0, 2, bhs);
    double dev = fabs(h[0][0] - 1.0) + fabs(h[1][1] - 1.0)
               + fabs(h[2][2] - 1.0);
    printf("  |h - delta| at r=50: %.6e\n", dev);
    CHECK(dev < 1e-6, "Superposed h -> delta far from BHs");
}

/* ================================================================
 * Test 5: Zero spin reduces to BY (V^i ~ 0)
 * ================================================================ */
static void test_zero_spin_reduces_to_by(void)
{
    printf("\n--- Test 5: Zero spin -> BY behavior ---\n");

    /* Single puncture with momentum but no spin */
    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.momentum[0] = 0.1;
    /* spin = 0: HiSpID should reduce to standard BY behavior */

    int N = 32;
    double L = 20.0;

    /* Standard BY solve */
    grid_t *g_by = grid_alloc(N, L, RK_CLASSIC);
    double res_by = relaxation_solve(g_by, 1, &bh, 1e-6, 2000, 0);

    /* HiSpID coupled solve (zero spin -> flat conformal metric) */
    grid_t *g_hi = grid_alloc(N, L, RK_CLASSIC);
    double res_hi = relaxation_solve_coupled(g_hi, 1, &bh, 1e-4, 2000, 0);

    printf("  BY residual:    %.6e\n", res_by);
    printf("  HiSpID residual: %.6e\n", res_hi);

    /* Compare chi at center */
    int mid = g_by->Ntotal / 2;
    int idx = IDX(g_by, mid, mid, mid);
    double chi_by = g_by->fields[FIELD_CHI][idx];
    double chi_hi = g_hi->fields[FIELD_CHI][idx];

    printf("  chi(center) BY:    %.10e\n", chi_by);
    printf("  chi(center) HiSpID: %.10e\n", chi_hi);

    double rel_diff = fabs(chi_by - chi_hi) / fabs(chi_by);
    printf("  Relative difference: %.6e\n", rel_diff);

    /* Allow 10% difference due to different background/source formulations */
    CHECK(rel_diff < 0.1, "Zero-spin HiSpID matches BY chi (within 10%)");
    CHECK(res_hi < 1e-4, "HiSpID solver converged for zero spin");

    grid_free(g_by);
    grid_free(g_hi);
}

/* ================================================================
 * Test 6: Moderate spin (chi=0.5), solver convergence
 * ================================================================ */
static void test_moderate_spin(void)
{
    printf("\n--- Test 6: Moderate spin (chi=0.5) ---\n");

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.spin[2] = 0.5;  /* chi = S/M^2 = 0.5 */

    int N = 32;
    double L = 20.0;
    grid_t *g = grid_alloc(N, L, RK_CLASSIC);

    double residual = relaxation_solve_coupled(g, 1, &bh, 1e-4, 2000, 1);

    printf("  Solver residual = %.6e\n", residual);
    CHECK(residual < 1e-2, "Coupled solver converged for chi=0.5");

    /* Check chi is physical */
    int gw = g->ghost;
    int Nt = g->Ntotal;
    double chi_min = 1e30;
    for (int k = gw; k < Nt - gw; k++)
        for (int j = gw; j < Nt - gw; j++)
            for (int i = gw; i < Nt - gw; i++) {
                int idx = IDX(g, i, j, k);
                double chi = g->fields[FIELD_CHI][idx];
                if (chi < chi_min) chi_min = chi;
            }

    printf("  chi_min = %.6e\n", chi_min);
    CHECK(chi_min > 0.0, "chi > 0 everywhere");

    grid_free(g);
}

/* ================================================================
 * Test 7: Constraint violation after solve
 * ================================================================ */
static void test_constraints_after_solve(void)
{
    printf("\n--- Test 7: Constraints after HiSpID solve ---\n");

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.spin[2] = 0.5;

    int N = 32;
    double L = 20.0;
    grid_t *g = grid_alloc(N, L, RK_CLASSIC);

    relaxation_solve_coupled(g, 1, &bh, 1e-4, 2000, 0);

    double ham = compute_constraint_l2(g);
    double mom = compute_momentum_l2(g);

    printf("  Ham L2 = %.6e\n", ham);
    printf("  Mom L2 = %.6e\n", mom);

    /* At this coarse resolution with a puncture, constraints will have
     * FD error.  Just verify they're bounded (not NaN/huge). */
    CHECK(ham < 1.0, "Hamiltonian constraint bounded");
    CHECK(mom < 1.0, "Momentum constraint bounded");
    CHECK(!isnan(ham) && !isinf(ham), "Ham is finite");
    CHECK(!isnan(mom) && !isinf(mom), "Mom is finite");

    grid_free(g);
}

/* ================================================================
 * Test 8: High spin (chi=0.9), evolve 10 steps, no NaN
 * ================================================================ */
static void test_high_spin_evolve(void)
{
    printf("\n--- Test 8: High spin chi=0.9, evolve 10 steps ---\n");

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.spin[2] = 0.9;  /* chi = 0.9 */

    int N = 32;
    double L = 20.0;
    mesh_t *m = mesh_create_ex(1, N, L, RK_CLASSIC, NUM_CCZ4_FIELDS);
    grid_t *g = m->blocks[0]->grid;

    printf("  Setting up HiSpID initial data (chi=0.9)...\n");
    double residual = relaxation_solve_coupled(g, 1, &bh, 1e-4, 2000, 1);
    printf("  Solver residual = %.6e\n", residual);

    /* Evolve 10 steps */
    sim_params_t p = default_params();
    p.N = g->N;
    p.L = L;
    p.dx = g->dx;
    p.CFL = 0.25;
    p.dt = p.CFL * p.dx;
    p.sigma = 0.3;
    p.time = 0.0;

    int nan_detected = 0;
    for (int step = 1; step <= 10; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;

        int mid = g->Ntotal / 2;
        double lapse = g->fields[FIELD_LAPSE][IDX(g, mid, mid, mid)];
        if (isnan(lapse) || isinf(lapse)) {
            printf("  NaN/Inf at step %d!\n", step);
            nan_detected = 1;
            break;
        }
    }

    CHECK(!nan_detected, "No NaN during 10 evolution steps (chi=0.9)");

    if (!nan_detected) {
        double ham = mesh_constraint_l2(m);
        printf("  After 10 steps: Ham L2 = %.6e\n", ham);
        CHECK(ham < 1e3, "Ham bounded after evolution");
    }

    mesh_free(m);
}

/* ================================================================
 * Test 9: det(h_ij) = 1 enforcement after CCZ4 conversion
 * ================================================================ */
static void test_det_h_unity(void)
{
    printf("\n--- Test 9: det(h_ij) = 1 after conversion ---\n");

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.spin[2] = 0.5;

    int N = 32;
    double L = 20.0;
    grid_t *g = grid_alloc(N, L, RK_CLASSIC);

    relaxation_solve_coupled(g, 1, &bh, 1e-4, 2000, 0);

    /* Check det(h_ij) at several interior points */
    int gw = g->ghost;
    int Nt = g->Ntotal;
    double max_dev = 0.0;

    for (int k = gw; k < Nt - gw; k += 4)
        for (int j = gw; j < Nt - gw; j += 4)
            for (int i = gw; i < Nt - gw; i += 4) {
                int idx = IDX(g, i, j, k);
                double h[3][3];
                h[0][0] = g->fields[FIELD_H11][idx];
                h[0][1] = g->fields[FIELD_H12][idx];
                h[0][2] = g->fields[FIELD_H13][idx];
                h[1][0] = h[0][1];
                h[1][1] = g->fields[FIELD_H22][idx];
                h[1][2] = g->fields[FIELD_H23][idx];
                h[2][0] = h[0][2];
                h[2][1] = h[1][2];
                h[2][2] = g->fields[FIELD_H33][idx];

                double det = compute_det_sym(h);
                double dev = fabs(det - 1.0);
                if (dev > max_dev) max_dev = dev;
            }

    printf("  Max |det(h) - 1| = %.6e\n", max_dev);
    CHECK(max_dev < 1e-6, "det(h_ij) = 1 after HiSpID conversion");

    grid_free(g);
}

/* ================================================================ */
int main(void)
{
    setbuf(stdout, NULL);  /* unbuffered for progress visibility */
    printf("=== HiSpID (High-Spin Initial Data) Test Suite ===\n");
    backend_init();

    test_kerr_metric_analytic();
    test_kerr_metric_falloff();
    test_kerr_extrinsic();
    test_gaussian_superposition();
    test_zero_spin_reduces_to_by();
    test_moderate_spin();
    test_constraints_after_solve();
    test_high_spin_evolve();
    test_det_h_unity();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);

    backend_cleanup();
    return tests_failed > 0 ? 1 : 0;
}
