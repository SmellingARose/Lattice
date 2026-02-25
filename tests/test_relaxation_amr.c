/*
 * Lattice — 3D Numerical Relativity
 * AMR FAS Multigrid constraint solver test suite.
 *
 * Tests:
 *   1. AMR vs uniform agreement: single puncture, compare psi in overlap
 *   2. Convergence: N=16,32 base with 2 AMR levels, order > 1
 *   3. N=3 punctures with momentum: solver converges
 *   4. Fallback: n_amr_levels=0 matches uniform solver
 *
 * Ref: arXiv:0705.1486 (Natchu & Matzner, 4th-order MG for BH data)
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/initial_data/relaxation.h"
#include "../src/initial_data/relaxation_amr.h"
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
 * Test 1: AMR solver converges and matches uniform at base resolution
 * ================================================================ */
static void test_amr_vs_uniform(void)
{
    printf("\n--- Test 1: AMR vs uniform agreement ---\n");

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.momentum[0] = 0.1;

    int N = 32;
    double L = 20.0;

    /* Uniform solve */
    grid_t *g_uni = grid_alloc(N, L, RK_CLASSIC);
    double res_uni = relaxation_solve(g_uni, 1, &bh, 1e-10, 30000, 1);
    printf("  Uniform residual = %.6e\n", res_uni);

    /* AMR solve (2 levels) */
    grid_t *g_amr = grid_alloc(N, L, RK_CLASSIC);
    double res_amr = relaxation_solve_amr(g_amr, 1, &bh, 1e-10, 30000, 1, 2);
    printf("  AMR residual = %.6e\n", res_amr);

    CHECK(res_uni < 1e-4, "Uniform solver converged");
    CHECK(res_amr < 1e-4, "AMR solver converged");

    /* Compare chi at interior points far from puncture
     * (away from the puncture singularity where resolution matters) */
    int gw = g_uni->ghost;
    int Nt = g_uni->Ntotal;
    double max_diff = 0.0;
    int n_compared = 0;
    for (int k = gw + 2; k < Nt - gw - 2; k++)
        for (int j = gw + 2; j < Nt - gw - 2; j++)
            for (int i = gw + 2; i < Nt - gw - 2; i++) {
                double x = COORD(g_uni, i);
                double y = COORD(g_uni, j);
                double z = COORD(g_uni, k);
                double r = sqrt(x*x + y*y + z*z);
                if (r < 2.0) continue;  /* skip near puncture */

                int idx = IDX(g_uni, i, j, k);
                double chi_uni = g_uni->fields[FIELD_CHI][idx];
                double chi_amr = g_amr->fields[FIELD_CHI][idx];
                double diff = fabs(chi_uni - chi_amr);
                if (chi_uni > 1e-10) {
                    double rel = diff / chi_uni;
                    if (rel > max_diff) max_diff = rel;
                }
                n_compared++;
            }

    printf("  Max relative chi difference (r>2M) = %.6e (%d pts)\n",
           max_diff, n_compared);
    /* AMR solver with coarse base but finer near puncture should give
     * comparable results at the base resolution far from the BH.
     * Allow 5% difference since the AMR solve path is different. */
    CHECK(max_diff < 0.05, "AMR and uniform chi agree to 5% (far field)");

    /* Both should have bounded Hamiltonian constraint */
    double ham_uni = compute_constraint_l2(g_uni);
    double ham_amr = compute_constraint_l2(g_amr);
    printf("  Ham L2: uniform = %.6e, AMR = %.6e\n", ham_uni, ham_amr);
    CHECK(ham_uni < 0.5, "Uniform Ham bounded");
    CHECK(ham_amr < 0.5, "AMR Ham bounded");

    grid_free(g_uni);
    grid_free(g_amr);
}

/* ================================================================
 * Test 2: Convergence order with AMR
 * ================================================================ */
static void test_convergence(void)
{
    printf("\n--- Test 2: Convergence with AMR ---\n");

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.momentum[0] = 0.1;

    double L = 20.0;

    /* Coarse: N=16 with 2 AMR levels */
    grid_t *g1 = grid_alloc(16, L, RK_CLASSIC);
    relaxation_solve_amr(g1, 1, &bh, 1e-10, 30000, 0, 2);
    double ham1 = compute_constraint_l2(g1);

    /* Fine: N=32 with 2 AMR levels */
    grid_t *g2 = grid_alloc(32, L, RK_CLASSIC);
    relaxation_solve_amr(g2, 1, &bh, 1e-10, 30000, 0, 2);
    double ham2 = compute_constraint_l2(g2);

    double ratio = ham1 / ham2;
    double order = log2(ratio);

    printf("  Ham L2 (N=16, 2 AMR) = %.6e\n", ham1);
    printf("  Ham L2 (N=32, 2 AMR) = %.6e\n", ham2);
    printf("  Ratio = %.2f (order = %.2f)\n", ratio, order);

    CHECK(ratio > 2.0, "Ham decreases with resolution (AMR)");
    CHECK(order > 1.0, "Convergence order > 1 (AMR)");

    grid_free(g1);
    grid_free(g2);
}

/* ================================================================
 * Test 3: N=3 punctures with momentum
 * ================================================================ */
static void test_three_punctures(void)
{
    printf("\n--- Test 3: 3-puncture AMR solve ---\n");

    puncture_data_t bhs[3];
    memset(bhs, 0, sizeof(bhs));

    bhs[0].mass = 0.5;
    bhs[0].center[0] = 3.0;
    bhs[0].momentum[1] = 0.1;

    bhs[1].mass = 0.5;
    bhs[1].center[0] = -3.0;
    bhs[1].momentum[1] = -0.1;

    bhs[2].mass = 0.3;
    bhs[2].center[1] = 4.0;
    bhs[2].momentum[0] = -0.05;

    int N = 32;
    double L = 24.0;
    grid_t *g = grid_alloc(N, L, RK_CLASSIC);

    double residual = relaxation_solve_amr(g, 3, bhs, 1e-8, 30000, 1, 2);
    printf("  Solver residual = %.6e\n", residual);
    CHECK(residual < 1e-3, "3-puncture AMR solver converged");

    double ham = compute_constraint_l2(g);
    printf("  Ham L2 = %.6e\n", ham);
    CHECK(ham < 1.0, "3-puncture Ham constraint bounded");

    /* chi should be positive everywhere */
    int gw = g->ghost;
    int Nt = g->Ntotal;
    double chi_min = 1e30;
    for (int k = gw; k < Nt - gw; k++)
        for (int j = gw; j < Nt - gw; j++)
            for (int i = gw; i < Nt - gw; i++) {
                double chi = g->fields[FIELD_CHI][IDX(g, i, j, k)];
                if (chi < chi_min) chi_min = chi;
            }
    printf("  chi_min = %.6e\n", chi_min);
    CHECK(chi_min > 0.0, "chi > 0 everywhere");

    grid_free(g);
}

/* ================================================================
 * Test 4: Fallback — n_amr_levels=0 matches uniform solver
 * ================================================================ */
static void test_fallback(void)
{
    printf("\n--- Test 4: Fallback (0 AMR levels = uniform) ---\n");

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.momentum[0] = 0.1;

    int N = 24;
    double L = 20.0;

    /* Uniform */
    grid_t *g1 = grid_alloc(N, L, RK_CLASSIC);
    double res1 = relaxation_solve(g1, 1, &bh, 1e-10, 30000, 0);

    /* AMR with 0 levels (should fall back) */
    grid_t *g2 = grid_alloc(N, L, RK_CLASSIC);
    double res2 = relaxation_solve_amr(g2, 1, &bh, 1e-10, 30000, 0, 0);

    printf("  Uniform residual = %.6e\n", res1);
    printf("  AMR(0) residual  = %.6e\n", res2);

    /* Results should be identical (same code path) */
    CHECK(fabs(res1 - res2) < 1e-15, "Fallback gives identical residual");

    /* Compare chi at center */
    int mid = g1->Ntotal / 2;
    double chi1 = g1->fields[FIELD_CHI][IDX(g1, mid, mid, mid)];
    double chi2 = g2->fields[FIELD_CHI][IDX(g2, mid, mid, mid)];
    printf("  chi(center): uniform=%.15e, AMR(0)=%.15e\n", chi1, chi2);
    CHECK(fabs(chi1 - chi2) < 1e-14, "Fallback gives identical chi");

    grid_free(g1);
    grid_free(g2);
}

/* ================================================================ */
int main(void)
{
    setbuf(stdout, NULL);
    printf("=== AMR FAS Multigrid Constraint Solver Test Suite ===\n");
    backend_init();

    test_amr_vs_uniform();
    test_convergence();
    test_three_punctures();
    test_fallback();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);

    backend_cleanup();
    return tests_failed > 0 ? 1 : 0;
}
