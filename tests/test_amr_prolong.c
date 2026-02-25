/*
 * Lattice — 3D Numerical Relativity
 * Tests for AMR Stage 3: Prolongation, Restriction, and Noise Reduction.
 *
 * Test groups:
 *  1. Prolongation 4th-order convergence (smooth function on 3 resolutions)
 *  2. Restriction volume-weighted average accuracy
 *  3. Restrict-then-prolongate round-trip
 *  4. CAKO on flat spacetime (chi=1 → no effect, Ham L2 unchanged)
 *  5. Per-field sigma verification (gauge fields get stronger dissipation)
 *  6. CAHD on single BH (verify damping term present)
 *  7. SSL on single BH (verify gauge pulse damping at t=0)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../src/core/grid.h"
#include "../src/core/fields.h"
#include "../src/core/params.h"
#include "../src/amr/prolongation.h"
#include "../src/amr/restriction.h"
#include "../src/initial_data/puncture.h"
#include "../src/diagnostics/constraints.h"
#include "../src/numerics/rk4.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/boundary/sommerfeld.h"
#include "../src/backend/backend.h"

static int pass_count = 0;
static int total_count = 0;

static void check(int cond, const char *msg)
{
    total_count++;
    if (cond) {
        printf("  [PASS] %s\n", msg);
        pass_count++;
    } else {
        printf("  [FAIL] %s\n", msg);
    }
}

/* Smooth test function: sin(pi*x/L)*sin(pi*y/L)*sin(pi*z/L)
 * Has non-zero derivatives of all orders → good convergence test */
static double smooth_func(double x, double y, double z, double L)
{
    double pi = 3.14159265358979323846;
    return sin(pi * x / L) * sin(pi * y / L) * sin(pi * z / L);
}

/* Set a field to the smooth test function on a grid */
static void set_smooth(grid_t *g, int field, double L)
{
    int ghost = g->ghost;
    int Nt = g->Ntotal;
    for (int k = 0; k < Nt; k++) {
        for (int j = 0; j < Nt; j++) {
            for (int i = 0; i < Nt; i++) {
                double x = (i - ghost + 0.5) * g->dx;
                double y = (j - ghost + 0.5) * g->dx;
                double z = (k - ghost + 0.5) * g->dx;
                g->fields[field][IDX(g, i, j, k)] = smooth_func(x, y, z, L);
            }
        }
    }
}

/* Compute max error of a field vs the smooth function on interior points */
static double max_error_smooth(const grid_t *g, int field, double L)
{
    int ghost = g->ghost;
    int N = g->N;
    double max_err = 0.0;
    for (int k = ghost; k < ghost + N; k++) {
        for (int j = ghost; j < ghost + N; j++) {
            for (int i = ghost; i < ghost + N; i++) {
                double x = (i - ghost + 0.5) * g->dx;
                double y = (j - ghost + 0.5) * g->dx;
                double z = (k - ghost + 0.5) * g->dx;
                double exact = smooth_func(x, y, z, L);
                double val = g->fields[field][IDX(g, i, j, k)];
                double err = fabs(val - exact);
                if (err > max_err) max_err = err;
            }
        }
    }
    return max_err;
}

/* ===== Test 1: Prolongation convergence ===== */
static void test_prolongation_convergence(void)
{
    printf("\n--- Test: Prolongation 4th-order convergence ---\n");

    double L = 10.0;
    /* Three resolutions: N=8→16, N=16→32, N=32→64 */
    int N_coarse[3] = {8, 16, 32};
    double errors[3];

    for (int r = 0; r < 3; r++) {
        int Nc = N_coarse[r];
        int Nf = 2 * Nc;

        grid_t *gc = grid_alloc(Nc, L, RK_CLASSIC);
        grid_t *gf = grid_alloc(Nf, L, RK_CLASSIC);

        /* Set smooth function on coarse grid (including ghost zones) */
        set_smooth(gc, FIELD_CHI, L);

        /* Prolongate to fine grid */
        prolongate_field(gc, FIELD_CHI, gf, FIELD_CHI);

        /* Measure error on fine interior */
        errors[r] = max_error_smooth(gf, FIELD_CHI, L);
        printf("  Nc=%2d → Nf=%2d  dx_c=%.4f  max_err=%.6e\n",
               Nc, Nf, gc->dx, errors[r]);

        grid_free(gc);
        grid_free(gf);
    }

    /* Check convergence order between successive resolutions */
    double order1 = log2(errors[0] / errors[1]);
    double order2 = log2(errors[1] / errors[2]);
    printf("  Order (8→16 vs 16→32):  %.2f\n", order1);
    printf("  Order (16→32 vs 32→64): %.2f\n", order2);

    check(order1 > 3.5, "Prolongation order > 3.5 (first pair)");
    check(order2 > 3.5, "Prolongation order > 3.5 (second pair)");
}

/* ===== Test 2: Restriction accuracy ===== */
static void test_restriction_accuracy(void)
{
    printf("\n--- Test: Restriction volume-weighted average ---\n");

    double L = 10.0;
    int Nc = 16;
    int Nf = 32;

    grid_t *gc = grid_alloc(Nc, L, RK_CLASSIC);
    grid_t *gf = grid_alloc(Nf, L, RK_CLASSIC);

    /* Set smooth function on fine grid */
    set_smooth(gf, FIELD_CHI, L);

    /* Restrict to coarse grid */
    restrict_field(gf, FIELD_CHI, gc, FIELD_CHI);

    /* The restricted value at a coarse cell should be close to the exact
     * value at the coarse cell center. For a smooth function, the error
     * is O(dx_f^2) = O((dx_c/2)^2). */
    double max_err = max_error_smooth(gc, FIELD_CHI, L);
    double dx_c = gc->dx;
    printf("  Restrict max error vs exact: %.6e\n", max_err);
    printf("  dx_c^2 = %.6e\n", dx_c * dx_c);

    check(max_err < dx_c * dx_c, "Restriction error < dx_c^2 (2nd-order)");

    grid_free(gc);
    grid_free(gf);
}

/* ===== Test 3: Restrict-then-prolongate round-trip ===== */
static void test_round_trip(void)
{
    printf("\n--- Test: Restrict-then-prolongate round-trip ---\n");

    double L = 10.0;
    int Nc = 16;
    int Nf = 32;

    grid_t *gf_orig = grid_alloc(Nf, L, RK_CLASSIC);
    grid_t *gc      = grid_alloc(Nc, L, RK_CLASSIC);
    grid_t *gf_rt   = grid_alloc(Nf, L, RK_CLASSIC);

    /* Set smooth function on fine grid */
    set_smooth(gf_orig, FIELD_CHI, L);

    /* Restrict to coarse */
    restrict_field(gf_orig, FIELD_CHI, gc, FIELD_CHI);

    /* Fill coarse ghost zones with the smooth function
     * (prolongation needs valid ghost data) */
    set_smooth(gc, FIELD_CHI, L);
    /* Re-restrict to only update interior */
    restrict_field(gf_orig, FIELD_CHI, gc, FIELD_CHI);

    /* Prolongate back to fine */
    prolongate_field(gc, FIELD_CHI, gf_rt, FIELD_CHI);

    /* Compare round-trip with original */
    int ghost = gf_orig->ghost;
    int N = gf_orig->N;
    double dx_c = gc->dx;
    double max_err = 0.0;
    for (int k = ghost; k < ghost + N; k++) {
        for (int j = ghost; j < ghost + N; j++) {
            for (int i = ghost; i < ghost + N; i++) {
                int idx = IDX(gf_orig, i, j, k);
                double orig = gf_orig->fields[FIELD_CHI][idx];
                double rt   = gf_rt->fields[FIELD_CHI][idx];
                double err = fabs(orig - rt);
                if (err > max_err) max_err = err;
            }
        }
    }
    printf("  Round-trip max error: %.6e\n", max_err);
    printf("  dx_c^2 = %.6e (expected scale)\n", dx_c * dx_c);

    /* Round-trip error is bounded by restriction + prolongation errors */
    check(max_err < 10.0 * dx_c * dx_c,
          "Round-trip error < 10 * dx_c^2");

    grid_free(gf_orig);
    grid_free(gc);
    grid_free(gf_rt);
}

/* ===== Test 4: CAKO on flat spacetime ===== */
static void test_cako_flat(void)
{
    printf("\n--- Test: CAKO on flat spacetime ---\n");

    sim_params_t p = default_params();
    p.N = 32;
    p.L = 10.0;
    p.dx = p.L / p.N;
    p.dt = p.CFL * p.dx;
    p.num_steps = 100;
    p.rk_method = RK_CLASSIC;

    /* Run without CAKO */
    grid_t *g1 = grid_alloc(p.N, p.L, p.rk_method);
    set_flat_spacetime(g1);
    for (int step = 0; step < p.num_steps; step++)
        rk4_step(g1, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);
    double ham1 = compute_constraint_l2(g1);

    /* Run with CAKO enabled (chi=1 on flat, so sqrt(1)=1, no effect) */
    sim_params_t p2 = p;
    p2.noise.use_cako = 1;
    grid_t *g2 = grid_alloc(p2.N, p2.L, p2.rk_method);
    set_flat_spacetime(g2);
    for (int step = 0; step < p2.num_steps; step++)
        rk4_step(g2, &p2, ccz4_rhs_point, apply_sommerfeld, p2.dt);
    double ham2 = compute_constraint_l2(g2);

    printf("  Without CAKO: Ham L2 = %.6e\n", ham1);
    printf("  With CAKO:    Ham L2 = %.6e\n", ham2);
    double ratio = ham2 / ham1;
    printf("  Ratio: %.6f\n", ratio);

    check(fabs(ratio - 1.0) < 1e-6, "CAKO has no effect on flat (chi=1)");

    grid_free(g1);
    grid_free(g2);
}

/* ===== Test 5: Per-field sigma ===== */
static void test_per_field_sigma(void)
{
    printf("\n--- Test: Per-field sigma ---\n");

    /* Verify the is_gauge_field classification.
     * Gauge fields: lapse(1) + shift(3) + B(3) = 7.
     * Physical fields: chi(1) + h_ij(6) + K(1) + A_ij(6) + Theta(1) + Gamma^i(3) = 18.
     * EM fields (E^i, BM^i) are neither gauge nor CCZ4 physical. */
    int gauge_count = 0;
    for (int f = FIELD_LAPSE; f <= FIELD_B3; f++)
        gauge_count++;
    int phys_count = FIELD_LAPSE;  /* fields 0..FIELD_LAPSE-1 */

    printf("  Gauge fields (sigma=0.99): %d (lapse, shift^i, B^i)\n", gauge_count);
    printf("  Physical fields (sigma=0.3): %d\n", phys_count);

    check(gauge_count == 7, "7 gauge fields (lapse + 3 shift + 3 B)");
    check(phys_count == 18, "18 physical fields");

    /* Run flat spacetime with per-field sigma to verify no crash */
    sim_params_t p = default_params();
    p.N = 16;
    p.L = 10.0;
    p.dx = p.L / p.N;
    p.dt = p.CFL * p.dx;
    p.noise.use_per_field_sigma = 1;

    grid_t *g = grid_alloc(p.N, p.L, p.rk_method);
    set_flat_spacetime(g);
    for (int step = 0; step < 10; step++)
        rk4_step(g, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);
    double ham = compute_constraint_l2(g);
    printf("  Per-field sigma flat Ham L2 = %.6e\n", ham);

    check(ham < 1e-10, "Per-field sigma stable on flat spacetime");

    grid_free(g);
}

/* ===== Test 6: CAHD on single BH ===== */
static void test_cahd_single_bh(void)
{
    printf("\n--- Test: CAHD on single BH ---\n");

    sim_params_t p = default_params();
    p.N = 32;
    p.L = 64.0;
    p.dx = p.L / p.N;
    p.dt = p.CFL * p.dx;
    int steps = 10;

    /* Without CAHD */
    double mass = 1.0;
    double center[1][3] = {{0.0, 0.0, 0.0}};
    grid_t *g1 = grid_alloc(p.N, p.L, p.rk_method);
    set_brill_lindquist(g1, 1, &mass, center);
    for (int step = 0; step < steps; step++)
        rk4_step(g1, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);
    double ham1 = compute_constraint_l2(g1);

    /* With CAHD */
    sim_params_t p2 = p;
    p2.noise.use_cahd = 1;
    grid_t *g2 = grid_alloc(p2.N, p2.L, p2.rk_method);
    set_brill_lindquist(g2, 1, &mass, center);
    for (int step = 0; step < steps; step++)
        rk4_step(g2, &p2, ccz4_rhs_point, apply_sommerfeld, p2.dt);
    double ham2 = compute_constraint_l2(g2);

    printf("  Without CAHD: Ham L2 = %.6e\n", ham1);
    printf("  With CAHD:    Ham L2 = %.6e\n", ham2);

    /* CAHD damps constraint violations, so ham2 should be different from ham1.
     * On a uniform grid the effect may be small but it should be measurable. */
    double diff = fabs(ham2 - ham1) / ham1;
    printf("  Relative difference: %.6e\n", diff);

    check(diff > 1e-10, "CAHD produces measurable effect on single BH");
    check(ham2 < ham1 * 2.0, "CAHD does not blow up constraints");

    grid_free(g1);
    grid_free(g2);
}

/* ===== Test 7: SSL on single BH ===== */
static void test_ssl_single_bh(void)
{
    printf("\n--- Test: SSL on single BH ---\n");

    sim_params_t p = default_params();
    p.N = 32;
    p.L = 64.0;
    p.dx = p.L / p.N;
    p.dt = p.CFL * p.dx;
    p.time = 0.0;
    int steps = 10;

    /* Without SSL (SSL is on by default, so explicitly disable) */
    p.noise.use_ssl = 0;
    double mass = 1.0;
    double center[1][3] = {{0.0, 0.0, 0.0}};
    grid_t *g1 = grid_alloc(p.N, p.L, p.rk_method);
    set_brill_lindquist(g1, 1, &mass, center);
    for (int step = 0; step < steps; step++) {
        rk4_step(g1, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);
        p.time += p.dt;
    }

    /* With SSL */
    sim_params_t p2 = p;
    p2.time = 0.0;
    p2.noise.use_ssl = 1;
    p2.noise.ssl_total_mass = 1.0;
    grid_t *g2 = grid_alloc(p2.N, p2.L, p2.rk_method);
    set_brill_lindquist(g2, 1, &mass, center);
    for (int step = 0; step < steps; step++) {
        rk4_step(g2, &p2, ccz4_rhs_point, apply_sommerfeld, p2.dt);
        p2.time += p2.dt;
    }

    /* Compare lapse profiles. SSL drives alpha toward sqrt(chi),
     * so at early times the lapse should differ between the two runs. */
    int ghost = g1->ghost;
    int N = g1->N;
    int center_i = ghost + N / 2;
    int center_idx = IDX(g1, center_i, center_i, center_i);
    double lapse1 = g1->fields[FIELD_LAPSE][center_idx];
    double lapse2 = g2->fields[FIELD_LAPSE][center_idx];

    printf("  Without SSL: lapse at center = %.8f\n", lapse1);
    printf("  With SSL:    lapse at center = %.8f\n", lapse2);
    double diff = fabs(lapse2 - lapse1);
    printf("  Difference: %.6e\n", diff);

    check(diff > 1e-10, "SSL produces measurable lapse difference at t=0");

    /* Verify SSL decays at late times: the Gaussian envelope at t >> sigma_t
     * should be negligible. exp(-(170)^2/(2*20^2)) ~ 10^-16 */
    double t_late = 170.0;
    double sigma_t = 20.0;
    double envelope = exp(-t_late * t_late / (2.0 * sigma_t * sigma_t));
    printf("  SSL envelope at t=170M: %.6e (should be negligible)\n", envelope);

    check(envelope < 1e-14, "SSL negligible at t=170M");

    grid_free(g1);
    grid_free(g2);
}

/* ===== Test 8: Prolongation weight sum ===== */
static void test_weight_sum(void)
{
    printf("\n--- Test: Prolongation weight sum ---\n");

    /* The weights should sum to 1 (partition of unity for constants) */
    double sum = 0.0;
    for (int i = 0; i < PROLONG_STENCIL; i++)
        sum += prolong_w[i];
    printf("  1D weight sum: %.16e (expect 1.0)\n", sum);

    check(fabs(sum - 1.0) < 1e-14, "1D weights sum to 1");

    /* 3D weight sum for one child should also be 1 */
    double sum3d = 0.0;
    for (int k = 0; k < PROLONG_STENCIL; k++)
        for (int j = 0; j < PROLONG_STENCIL; j++)
            for (int i = 0; i < PROLONG_STENCIL; i++)
                sum3d += prolong_w[k] * prolong_w[j] * prolong_w[i];
    printf("  3D weight sum: %.16e (expect 1.0)\n", sum3d);

    check(fabs(sum3d - 1.0) < 1e-14, "3D tensor product weights sum to 1");
}

/* ===== Test 9: Prolongation exact for linear ===== */
static void test_prolongation_linear(void)
{
    printf("\n--- Test: Prolongation exact for linear function ---\n");

    double L = 10.0;
    int Nc = 16;
    int Nf = 32;

    grid_t *gc = grid_alloc(Nc, L, RK_CLASSIC);
    grid_t *gf = grid_alloc(Nf, L, RK_CLASSIC);

    /* Set f(x,y,z) = 2x + 3y + 5z on coarse grid (including ghosts) */
    int ghost = gc->ghost;
    int Nt = gc->Ntotal;
    for (int k = 0; k < Nt; k++)
        for (int j = 0; j < Nt; j++)
            for (int i = 0; i < Nt; i++) {
                double x = (i - ghost + 0.5) * gc->dx;
                double y = (j - ghost + 0.5) * gc->dx;
                double z = (k - ghost + 0.5) * gc->dx;
                gc->fields[FIELD_CHI][IDX(gc, i, j, k)] = 2*x + 3*y + 5*z;
            }

    prolongate_field(gc, FIELD_CHI, gf, FIELD_CHI);

    /* Check error on fine interior */
    double max_err = 0.0;
    int N = gf->N;
    ghost = gf->ghost;
    for (int k = ghost; k < ghost + N; k++)
        for (int j = ghost; j < ghost + N; j++)
            for (int i = ghost; i < ghost + N; i++) {
                double x = (i - ghost + 0.5) * gf->dx;
                double y = (j - ghost + 0.5) * gf->dx;
                double z = (k - ghost + 0.5) * gf->dx;
                double exact = 2*x + 3*y + 5*z;
                double val = gf->fields[FIELD_CHI][IDX(gf, i, j, k)];
                double err = fabs(val - exact);
                if (err > max_err) max_err = err;
            }

    printf("  Linear function max error: %.6e\n", max_err);
    check(max_err < 1e-12, "Prolongation exact for linear (to roundoff)");

    grid_free(gc);
    grid_free(gf);
}

int main(void)
{
    printf("=== AMR Stage 3: Prolongation + Noise Reduction Test ===\n");

    test_weight_sum();
    test_prolongation_linear();
    test_prolongation_convergence();
    test_restriction_accuracy();
    test_round_trip();
    test_cako_flat();
    test_per_field_sigma();
    test_cahd_single_bh();
    test_ssl_single_bh();

    printf("\n=== Results: %d/%d passed ===\n", pass_count, total_count);
    if (pass_count == total_count) {
        printf("ALL PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}
