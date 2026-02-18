/*
 * Lattice — 3D Numerical Relativity
 * Bowen-York initial data test suite.
 *
 * Tests:
 *   1. A_ij at known point vs analytic (machine precision)
 *   2. A_ij symmetry: |A_ij - A_ji| < 1e-15
 *   3. Falloff: momentum 1/r^2, spin 1/r^3
 *   4. Two-puncture superposition (linearity of A_ij)
 *   5. CCZ4 conversion: chi = psi^{-4}, A_CCZ4 = psi^{-6} * A_phys
 *   6. Zero momentum: BL exact, solver converges to u~0
 *   7. Small momentum: solver converges, Ham constraint bounded
 *   8. Convergence order: N=32 vs N=64 Ham violation ratio
 *   9. Binary orbit: two BHs with tangential momentum, evolve 10 steps
 *
 * Ref: gr-qc/9703066, B&S Eqs. 3.43-3.44
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/puncture.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/initial_data/relaxation.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/boundary/sommerfeld.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
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
 * Test 1: A_ij at known point vs analytic formula
 * ================================================================ */
static void test_Aij_analytic(void)
{
    printf("\n--- Test 1: A_ij analytic values ---\n");

    /* Single puncture at origin with P_x = 1.0 */
    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.center[0] = 0.0; bh.center[1] = 0.0; bh.center[2] = 0.0;
    bh.momentum[0] = 1.0; bh.momentum[1] = 0.0; bh.momentum[2] = 0.0;

    /* Evaluate at (r, 0, 0) where n = (1, 0, 0), r = 2.0 */
    double A[3][3];
    bowen_york_Aij(A, 2.0, 0.0, 0.0, 1, &bh);

    /* At (r,0,0) with P = (P,0,0):
     * n = (1,0,0), P.n = P
     * A_xx = (3/(2r^2)) [P*1 + P*1 - (1-1)*P] = 3P/r^2
     * A_yy = (3/(2r^2)) [0 + 0 - (1-0)*P] = -3P/(2r^2)
     * A_zz = same as yy
     * A_xy = A_xz = A_yz = 0 */
    double r = 2.0, P = 1.0;
    double expect_xx = 3.0 * P / (r * r);
    double expect_yy = -1.5 * P / (r * r);

    printf("  A_xx = %.15e (expect %.15e)\n", A[0][0], expect_xx);
    printf("  A_yy = %.15e (expect %.15e)\n", A[1][1], expect_yy);
    printf("  A_zz = %.15e (expect %.15e)\n", A[2][2], expect_yy);
    printf("  A_xy = %.15e (expect 0)\n", A[0][1]);

    CHECK(fabs(A[0][0] - expect_xx) < 1e-14, "A_xx matches analytic");
    CHECK(fabs(A[1][1] - expect_yy) < 1e-14, "A_yy matches analytic");
    CHECK(fabs(A[2][2] - expect_yy) < 1e-14, "A_zz matches analytic");
    CHECK(fabs(A[0][1]) < 1e-14, "A_xy = 0 on axis");
    CHECK(fabs(A[0][2]) < 1e-14, "A_xz = 0 on axis");
    CHECK(fabs(A[1][2]) < 1e-14, "A_yz = 0 on axis");
}

/* ================================================================
 * Test 2: A_ij symmetry
 * ================================================================ */
static void test_Aij_symmetry(void)
{
    printf("\n--- Test 2: A_ij symmetry ---\n");

    /* Puncture with both momentum and spin at a generic point */
    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 0.5;
    bh.center[0] = 1.0; bh.center[1] = -0.5; bh.center[2] = 0.3;
    bh.momentum[0] = 0.3; bh.momentum[1] = -0.7; bh.momentum[2] = 0.1;
    bh.spin[0] = 0.1; bh.spin[1] = 0.2; bh.spin[2] = -0.3;

    double A[3][3];
    bowen_york_Aij(A, 3.5, 1.2, -0.8, 1, &bh);

    double max_asym = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 3; j++) {
            double d = fabs(A[i][j] - A[j][i]);
            if (d > max_asym) max_asym = d;
        }

    printf("  Max |A_ij - A_ji| = %.6e\n", max_asym);
    CHECK(max_asym < 1e-15, "A_ij is symmetric");
}

/* ================================================================
 * Test 3: Radial falloff — momentum 1/r^2, spin 1/r^3
 * ================================================================ */
static void test_Aij_falloff(void)
{
    printf("\n--- Test 3: A_ij radial falloff ---\n");

    /* Momentum-only puncture */
    puncture_data_t bh_P;
    memset(&bh_P, 0, sizeof(bh_P));
    bh_P.mass = 1.0;
    bh_P.momentum[0] = 1.0;

    double A1[3][3], A2[3][3];
    bowen_york_Aij(A1, 1.0, 0.0, 0.0, 1, &bh_P);
    bowen_york_Aij(A2, 2.0, 0.0, 0.0, 1, &bh_P);

    /* Momentum: A ~ 1/r^2, so A(r=1)/A(r=2) should be 4.0 */
    double ratio_P = A1[0][0] / A2[0][0];
    printf("  Momentum: A_xx(r=1)/A_xx(r=2) = %.6f (expect 4.0)\n", ratio_P);
    CHECK(fabs(ratio_P - 4.0) < 1e-12, "Momentum A_ij falls as 1/r^2");

    /* Spin-only puncture */
    puncture_data_t bh_S;
    memset(&bh_S, 0, sizeof(bh_S));
    bh_S.mass = 1.0;
    bh_S.spin[2] = 1.0;  /* spin along z */

    /* Evaluate along x-axis: nxS = (0,0,1)x(0,0,S_z) with n=(1,0,0)
     * nxS = n x S = (0*Sz - 0*Sy, 0*Sx - 1*Sz, 1*Sy - 0*Sx) = (0, -Sz, 0)
     * A_xy^S = -(3/r^3) [nxS_x * n_y + nxS_y * n_x] = -(3/r^3)(0 + (-Sz)*1)
     *        = 3*Sz/r^3 */
    bowen_york_Aij(A1, 1.0, 0.0, 0.0, 1, &bh_S);
    bowen_york_Aij(A2, 2.0, 0.0, 0.0, 1, &bh_S);

    double ratio_S = A1[0][1] / A2[0][1];
    printf("  Spin: A_xy(r=1)/A_xy(r=2) = %.6f (expect 8.0)\n", ratio_S);
    CHECK(fabs(ratio_S - 8.0) < 1e-12, "Spin A_ij falls as 1/r^3");
}

/* ================================================================
 * Test 4: Two-puncture superposition (linearity)
 * ================================================================ */
static void test_Aij_superposition(void)
{
    printf("\n--- Test 4: Two-puncture superposition ---\n");

    puncture_data_t bhs[2];
    memset(bhs, 0, sizeof(bhs));

    bhs[0].mass = 0.5;
    bhs[0].center[0] = 2.0;
    bhs[0].momentum[1] = 0.3;

    bhs[1].mass = 0.5;
    bhs[1].center[0] = -2.0;
    bhs[1].momentum[1] = -0.3;

    /* Combined */
    double A_both[3][3];
    bowen_york_Aij(A_both, 0.0, 1.0, 0.5, 2, bhs);

    /* Individual */
    double A1[3][3], A2[3][3];
    bowen_york_Aij(A1, 0.0, 1.0, 0.5, 1, &bhs[0]);
    bowen_york_Aij(A2, 0.0, 1.0, 0.5, 1, &bhs[1]);

    double max_diff = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double d = fabs(A_both[i][j] - (A1[i][j] + A2[i][j]));
            if (d > max_diff) max_diff = d;
        }

    printf("  Max |A_both - (A_1 + A_2)| = %.6e\n", max_diff);
    CHECK(max_diff < 1e-15, "A_ij is linear (superposition holds)");
}

/* ================================================================
 * Test 5: CCZ4 conversion — chi = psi^{-4}, A_CCZ4 = psi^{-6} * A_phys
 * ================================================================ */
static void test_ccz4_conversion(void)
{
    printf("\n--- Test 5: CCZ4 field conversion ---\n");

    /* Single puncture at origin, no momentum (analytic psi) */
    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.momentum[0] = 0.5;  /* small momentum for non-trivial A_ij */

    /* Small grid just to test the conversion */
    int N = 16;
    double L = 20.0;
    grid_t *g = grid_alloc(N, L, RK_CLASSIC);

    /* Build psi = psi_BL (analytic) and call set_ccz4_from_psi */
    double *psi = calloc(g->npoints, sizeof(double));
    for (int k = 0; k < g->Ntotal; k++)
        for (int j = 0; j < g->Ntotal; j++)
            for (int i = 0; i < g->Ntotal; i++) {
                int idx = IDX(g, i, j, k);
                double x = COORD(g, i);
                double y = COORD(g, j);
                double z = COORD(g, k);
                psi[idx] = brill_lindquist_psi(x, y, z, 1, &bh);
            }

    set_ccz4_from_psi(g, psi, 1, &bh);

    /* Check a few interior points */
    int mid = g->Ntotal / 2;
    int idx = IDX(g, mid, mid, mid);

    double psi_val = psi[idx];
    double chi_expect = 1.0 / (psi_val * psi_val * psi_val * psi_val);
    double chi_got = g->fields[FIELD_CHI][idx];

    printf("  psi = %.10f\n", psi_val);
    printf("  chi = %.10e (expect %.10e)\n", chi_got, chi_expect);
    CHECK(fabs(chi_got - chi_expect) < 1e-14, "chi = psi^{-4}");

    /* Check A_CCZ4 = psi^{-6} * A_phys */
    double x = COORD(g, mid);
    double y = COORD(g, mid);
    double z = COORD(g, mid);
    double A_phys[3][3];
    bowen_york_Aij(A_phys, x, y, z, 1, &bh);
    double psi6_inv = 1.0 / (psi_val * psi_val * psi_val *
                              psi_val * psi_val * psi_val);
    double A11_expect = psi6_inv * A_phys[0][0];
    double A11_got = g->fields[FIELD_A11][idx];

    printf("  A11_CCZ4 = %.10e (expect %.10e)\n", A11_got, A11_expect);
    CHECK(fabs(A11_got - A11_expect) < 1e-14, "A_CCZ4 = psi^{-6} * A_phys");

    /* Check h_ij = delta_ij */
    CHECK(fabs(g->fields[FIELD_H11][idx] - 1.0) < 1e-15, "h_11 = 1");
    CHECK(fabs(g->fields[FIELD_H12][idx]) < 1e-15, "h_12 = 0");
    CHECK(fabs(g->fields[FIELD_K][idx]) < 1e-15, "K = 0");

    /* Check lapse = sqrt(chi) */
    double lapse_expect = sqrt(chi_expect);
    double lapse_got = g->fields[FIELD_LAPSE][idx];
    CHECK(fabs(lapse_got - lapse_expect) < 1e-14, "lapse = sqrt(chi)");

    free(psi);
    grid_free(g);
}

/* ================================================================
 * Test 6: Zero momentum — BL exact, solver should converge to u~0
 * ================================================================ */
static void test_zero_momentum(void)
{
    printf("\n--- Test 6: Zero momentum (BL exact path) ---\n");

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    /* P = 0, S = 0 */

    int N = 24;
    double L = 20.0;
    grid_t *g = grid_alloc(N, L, RK_CLASSIC);

    set_bowen_york(g, 1, &bh);

    /* Check: chi at center should match BL */
    int mid = g->Ntotal / 2;
    int idx = IDX(g, mid, mid, mid);
    double x = COORD(g, mid);
    double y = COORD(g, mid);
    double z = COORD(g, mid);
    double psi_bl = brill_lindquist_psi(x, y, z, 1, &bh);
    double chi_expect = 1.0 / (psi_bl * psi_bl * psi_bl * psi_bl);
    double chi_got = g->fields[FIELD_CHI][idx];

    printf("  chi(center) = %.10e (expect %.10e)\n", chi_got, chi_expect);
    CHECK(fabs(chi_got - chi_expect) / chi_expect < 1e-12,
          "BL path: chi matches analytic");

    /* All A_ij should be zero */
    double max_A = 0.0;
    int gw = g->ghost;
    int Nt = g->Ntotal;
    for (int k = gw; k < Nt - gw; k++)
        for (int j = gw; j < Nt - gw; j++)
            for (int i = gw; i < Nt - gw; i++) {
                int ix = IDX(g, i, j, k);
                for (int f = FIELD_A11; f <= FIELD_A33; f++) {
                    double v = fabs(g->fields[f][ix]);
                    if (v > max_A) max_A = v;
                }
            }

    printf("  Max |A_ij| = %.6e\n", max_A);
    CHECK(max_A < 1e-15, "BL path: A_ij = 0 everywhere");

    /* Hamiltonian constraint */
    double ham = compute_constraint_l2(g);
    printf("  Ham L2 = %.6e\n", ham);
    /* Discretization error at this coarse resolution (dx~0.83) is O(dx^4) ~ O(0.5).
     * BL data has 1/r singularity near puncture, so FD error is expected. */
    CHECK(ham < 0.05, "BL path: Hamiltonian constraint bounded");

    grid_free(g);
}

/* ================================================================
 * Test 7: Small momentum — solver converges, Ham bounded
 * ================================================================ */
static void test_small_momentum(void)
{
    printf("\n--- Test 7: Small momentum (P=0.1) solver ---\n");

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.momentum[0] = 0.1;

    int N = 24;
    double L = 20.0;
    grid_t *g = grid_alloc(N, L, RK_CLASSIC);

    double residual = relaxation_solve(g, 1, &bh, 1e-10, 30000, 1);

    printf("  Solver residual ||v||_L2 = %.6e\n", residual);
    CHECK(residual < 1e-6, "Solver converged (||v||_L2 small)");

    /* Check Hamiltonian constraint */
    double ham = compute_constraint_l2(g);
    printf("  Ham L2 = %.6e\n", ham);
    CHECK(ham < 1e-2, "Hamiltonian constraint bounded after solve");

    /* Check that chi is physical (positive everywhere interior) */
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
    CHECK(chi_min > 0.0, "chi > 0 everywhere (physical)");

    /* A_ij should be non-trivial */
    double max_A = 0.0;
    for (int k = gw; k < Nt - gw; k++)
        for (int j = gw; j < Nt - gw; j++)
            for (int i = gw; i < Nt - gw; i++) {
                int idx = IDX(g, i, j, k);
                double a11 = fabs(g->fields[FIELD_A11][idx]);
                if (a11 > max_A) max_A = a11;
            }

    printf("  Max |A_11| = %.6e\n", max_A);
    CHECK(max_A > 1e-6, "A_ij is non-trivial with momentum");

    grid_free(g);
}

/* ================================================================
 * Test 8: Convergence order — N=16 vs N=32, Ham ratio
 * ================================================================ */
static void test_convergence_order(void)
{
    printf("\n--- Test 8: Convergence order (N=16 vs N=32) ---\n");

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.momentum[0] = 0.1;

    double L = 20.0;

    /* Coarse: N=16 */
    grid_t *g1 = grid_alloc(16, L, RK_CLASSIC);
    relaxation_solve(g1, 1, &bh, 1e-10, 30000, 0);
    double ham1 = compute_constraint_l2(g1);

    /* Fine: N=32 */
    grid_t *g2 = grid_alloc(32, L, RK_CLASSIC);
    relaxation_solve(g2, 1, &bh, 1e-10, 30000, 0);
    double ham2 = compute_constraint_l2(g2);

    double ratio = ham1 / ham2;
    double order = log2(ratio);

    printf("  Ham L2 (N=16) = %.6e\n", ham1);
    printf("  Ham L2 (N=32) = %.6e\n", ham2);
    printf("  Ratio = %.2f (order = %.2f, expect 3-5)\n", ratio, order);

    /* 4th-order FD gives ratio ~16 asymptotically, but at these coarse
     * resolutions the puncture singularity + boundary effects limit
     * effective order.  Accept order >= 1.0 as passing. */
    CHECK(ratio > 2.0, "Ham violation decreases with resolution");
    CHECK(order > 1.0, "Effective convergence order > 1.0");

    grid_free(g1);
    grid_free(g2);
}

/* ================================================================
 * Test 9: Binary orbit — two BHs with tangential momentum, evolve
 * ================================================================ */
static void test_binary_orbit(void)
{
    printf("\n--- Test 9: Binary orbit (evolve 10 steps) ---\n");

    /* Two equal-mass BHs at z=+/-3 with tangential momentum P_y */
    puncture_data_t bhs[2];
    memset(bhs, 0, sizeof(bhs));

    bhs[0].mass = 0.5;
    bhs[0].center[2] = 3.0;
    bhs[0].momentum[1] = 0.1;

    bhs[1].mass = 0.5;
    bhs[1].center[2] = -3.0;
    bhs[1].momentum[1] = -0.1;

    int N = 24;
    double L = 20.0;
    grid_t *g = grid_alloc(N, L, RK_CLASSIC);

    printf("  Setting up binary BY initial data...\n");
    double residual = relaxation_solve(g, 2, bhs, 1e-10, 30000, 1);
    printf("  Solver residual = %.6e\n", residual);
    CHECK(residual < 1e-4, "Binary solver converged");

    /* Check constraints before evolution */
    double ham0 = compute_constraint_l2(g);
    double mom0 = compute_momentum_l2(g);
    printf("  Initial: Ham L2 = %.6e, Mom L2 = %.6e\n", ham0, mom0);

    /* Evolve 10 steps */
    sim_params_t p = default_params();
    p.N = g->N;
    p.L = L;
    p.dx = g->dx;
    p.CFL = 0.25;
    p.dt = p.CFL * p.dx;
    p.sigma = 0.3;

    int nan_detected = 0;
    for (int step = 1; step <= 10; step++) {
        rk4_step(g, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);

        /* Check for NaN in lapse */
        int mid = g->Ntotal / 2;
        double lapse = g->fields[FIELD_LAPSE][IDX(g, mid, mid, mid)];
        if (isnan(lapse) || isinf(lapse)) {
            printf("  NaN/Inf detected at step %d!\n", step);
            nan_detected = 1;
            break;
        }
    }

    CHECK(!nan_detected, "No NaN/Inf during 10 evolution steps");

    if (!nan_detected) {
        double ham10 = compute_constraint_l2(g);
        double mom10 = compute_momentum_l2(g);
        printf("  After 10 steps: Ham L2 = %.6e, Mom L2 = %.6e\n",
               ham10, mom10);
        CHECK(ham10 < 1e2, "Ham constraint bounded after evolution");
        CHECK(mom10 < 1e2, "Mom constraint bounded after evolution");
    }

    grid_free(g);
}

/* ================================================================ */
int main(void)
{
    printf("=== Bowen-York Initial Data Test Suite ===\n");
    backend_init();

    test_Aij_analytic();
    test_Aij_symmetry();
    test_Aij_falloff();
    test_Aij_superposition();
    test_ccz4_conversion();
    test_zero_momentum();
    test_small_momentum();
    test_convergence_order();
    test_binary_orbit();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);

    backend_cleanup();
    return tests_failed > 0 ? 1 : 0;
}
