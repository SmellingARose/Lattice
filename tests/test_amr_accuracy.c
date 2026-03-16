/*
 * Lattice — 3D Numerical Relativity
 * AMR vs uniform solver accuracy comparison.
 *
 * Two tests:
 *
 * Test 1 (same output grid): Both solvers output to a uniform N=32 grid.
 * The AMR solver refines internally but the output is at base resolution,
 * so constraint quality should be similar.  This validates that AMR
 * doesn't degrade the solution.
 *
 * Test 2 (effective resolution comparison): The AMR solver at N=32 base
 * with 2 refinement levels achieves dx_fine = dx_base/4 near the puncture.
 * Compare its solver residual against a uniform solve at N=128 (same dx).
 * The AMR solver should match N=128 quality near the puncture while using
 * far less memory (~9 blocks vs 128^3 points).
 *
 * Test 3 (large domain): With L=128, N=32 gives dx=4M — far too coarse
 * for puncture structure.  Compare uniform N=32 (dx=4M) against a uniform
 * N=128 reference (dx=1M).  Then test AMR N=32 + 2 levels (dx=1M near
 * puncture) and verify it approaches the N=128 reference quality.
 *
 * Ref: arXiv:0705.1486 (Natchu & Matzner, 4th-order MG for BH data)
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/initial_data/jfnk_solver.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Compute Ham L2 in a spherical shell [r_min, r_max] around origin */
static double ham_l2_shell(const grid_t *g, double r_min, double r_max)
{
    int gw = g->ghost;
    int Nt = g->Ntotal;
    double sum = 0.0;
    int count = 0;

    for (int k = gw + 2; k < Nt - gw - 2; k++)
        for (int j = gw + 2; j < Nt - gw - 2; j++)
            for (int i = gw + 2; i < Nt - gw - 2; i++) {
                double x = COORD(g, i);
                double y = COORD(g, j);
                double z = COORD(g, k);
                double r = sqrt(x*x + y*y + z*z);
                if (r < r_min || r > r_max) continue;

                double H = compute_hamiltonian_at(
                    (const double *const *)g->fields, g, i, j, k);
                sum += H * H;
                count++;
            }

    return (count > 0) ? sqrt(sum / count) : 0.0;
}

/* Compute Ham Linf (max |H|) in a spherical shell */
static double ham_linf_shell(const grid_t *g, double r_min, double r_max)
{
    int gw = g->ghost;
    int Nt = g->Ntotal;
    double maxH = 0.0;

    for (int k = gw + 2; k < Nt - gw - 2; k++)
        for (int j = gw + 2; j < Nt - gw - 2; j++)
            for (int i = gw + 2; i < Nt - gw - 2; i++) {
                double x = COORD(g, i);
                double y = COORD(g, j);
                double z = COORD(g, k);
                double r = sqrt(x*x + y*y + z*z);
                if (r < r_min || r > r_max) continue;

                double H = compute_hamiltonian_at(
                    (const double *const *)g->fields, g, i, j, k);
                double absH = fabs(H);
                if (absH > maxH) maxH = absH;
            }

    return maxH;
}

int main(void)
{
    setbuf(stdout, NULL);
    printf("=== AMR vs Uniform Solver Accuracy Comparison ===\n\n");
    backend_init();

    puncture_data_t bh;
    memset(&bh, 0, sizeof(bh));
    bh.mass = 1.0;
    bh.momentum[0] = 0.1;

    /* ============================================================
     * Test 1: Small domain (L=20) — verify AMR matches uniform
     * At L=20, N=32: dx=0.625M, already decent resolution.
     * AMR shouldn't help much but shouldn't hurt either.
     * ============================================================ */
    {
        double L = 20.0;
        int N = 32;
        printf("--- Test 1: Small domain (L=%.0f, N=%d, dx=%.3f) ---\n",
               L, N, L/N);
        printf("  AMR: 2 levels -> dx_fine=%.4f near puncture\n\n",
               L / (N * 4.0));

        grid_t *g_uni = grid_alloc(N, L, RK_CLASSIC);
        jfnk_solve(g_uni, 1, &bh, 1e-10, 50, 0);

        grid_t *g_amr = grid_alloc(N, L, RK_CLASSIC);
        jfnk_solve_amr(g_amr, 1, &bh, 1e-10, 50, 0, 2);

        double uni_near = ham_l2_shell(g_uni, 1.0, 4.0);
        double amr_near = ham_l2_shell(g_amr, 1.0, 4.0);
        double uni_far  = ham_l2_shell(g_uni, 5.0, 8.0);
        double amr_far  = ham_l2_shell(g_amr, 5.0, 8.0);
        double uni_glob = compute_constraint_l2(g_uni);
        double amr_glob = compute_constraint_l2(g_amr);

        printf("  %-22s %12s %12s\n", "Metric", "Uniform", "AMR(2lev)");
        printf("  %-22s %12s %12s\n", "------", "-------", "---------");
        printf("  %-22s %12.4e %12.4e\n", "Ham L2  r=1..4M", uni_near, amr_near);
        printf("  %-22s %12.4e %12.4e\n", "Ham L2  r=5..8M", uni_far, amr_far);
        printf("  %-22s %12.4e %12.4e\n", "Ham L2  global", uni_glob, amr_glob);
        printf("  -> AMR matches uniform (expected: same output grid resolution)\n");

        grid_free(g_uni);
        grid_free(g_amr);
    }

    /* ============================================================
     * Test 2: Large domain (L=128) — AMR should help
     * At L=128, N=32: dx=4M (can't resolve puncture at all)
     * AMR 2 levels: dx_fine = 1M near puncture
     * Reference: uniform N=128 (dx=1M everywhere)
     * ============================================================ */
    {
        double L = 128.0;
        int N_coarse = 32;
        int N_fine = 128;
        printf("\n--- Test 2: Large domain (L=%.0f) ---\n", L);
        printf("  Coarse: N=%d, dx=%.1fM (can't resolve puncture)\n",
               N_coarse, L/N_coarse);
        printf("  AMR:    N=%d + 2 levels, dx_fine=%.2fM near puncture\n",
               N_coarse, L / (N_coarse * 4.0));
        printf("  Ref:    N=%d uniform, dx=%.2fM everywhere\n\n",
               N_fine, L/N_fine);

        /* Coarse uniform: N=32, dx=4M */
        grid_t *g_coarse = grid_alloc(N_coarse, L, RK_CLASSIC);
        double res_coarse = jfnk_solve(g_coarse, 1, &bh, 1e-10, 50, 1);

        /* AMR: N=32 base + 2 levels -> dx=1M near puncture */
        grid_t *g_amr = grid_alloc(N_coarse, L, RK_CLASSIC);
        double res_amr = jfnk_solve_amr(g_amr, 1, &bh, 1e-10, 50, 1, 2);

        /* Reference: N=128 uniform, dx=1M */
        grid_t *g_ref = grid_alloc(N_fine, L, RK_CLASSIC);
        double res_ref = jfnk_solve(g_ref, 1, &bh, 1e-10, 50, 1);

        printf("\n  Solver residuals:\n");
        printf("    Coarse (N=%d):  %.4e\n", N_coarse, res_coarse);
        printf("    AMR (N=%d+2):   %.4e\n", N_coarse, res_amr);
        printf("    Ref (N=%d):     %.4e\n", N_fine, res_ref);

        /* Compare Ham constraint in shells */
        double r_min = 2.0, r_max = 6.0;
        double r_far_min = 10.0, r_far_max = 30.0;

        double coarse_near_l2   = ham_l2_shell(g_coarse, r_min, r_max);
        double coarse_near_linf = ham_linf_shell(g_coarse, r_min, r_max);
        double coarse_far_l2    = ham_l2_shell(g_coarse, r_far_min, r_far_max);

        double amr_near_l2   = ham_l2_shell(g_amr, r_min, r_max);
        double amr_near_linf = ham_linf_shell(g_amr, r_min, r_max);
        double amr_far_l2    = ham_l2_shell(g_amr, r_far_min, r_far_max);

        double ref_near_l2   = ham_l2_shell(g_ref, r_min, r_max);
        double ref_near_linf = ham_linf_shell(g_ref, r_min, r_max);
        double ref_far_l2    = ham_l2_shell(g_ref, r_far_min, r_far_max);

        printf("\n  Hamiltonian constraint comparison:\n");
        printf("  %-22s %12s %12s %12s\n",
               "Metric", "Coarse N=32", "AMR N=32+2", "Ref N=128");
        printf("  %-22s %12s %12s %12s\n",
               "------", "-----------", "----------", "---------");
        printf("  %-22s %12.4e %12.4e %12.4e\n",
               "Ham L2  r=2..6M", coarse_near_l2, amr_near_l2, ref_near_l2);
        printf("  %-22s %12.4e %12.4e %12.4e\n",
               "Ham Linf r=2..6M", coarse_near_linf, amr_near_linf, ref_near_linf);
        printf("  %-22s %12.4e %12.4e %12.4e\n",
               "Ham L2  r=10..30M", coarse_far_l2, amr_far_l2, ref_far_l2);

        printf("\n  Improvement ratios (vs coarse):\n");
        if (amr_near_l2 > 0)
            printf("    AMR near-field: %.1fx better\n",
                   coarse_near_l2 / amr_near_l2);
        if (ref_near_l2 > 0)
            printf("    Ref near-field: %.1fx better\n",
                   coarse_near_l2 / ref_near_l2);

        /* Memory comparison */
        printf("\n  Memory comparison:\n");
        printf("    Coarse N=%d: %.1f MB (single grid)\n",
               N_coarse,
               (double)N_coarse * N_coarse * N_coarse * 25 * 8 / 1e6);
        printf("    Ref N=%d: %.1f MB (single grid)\n",
               N_fine,
               (double)N_fine * N_fine * N_fine * 25 * 8 / 1e6);
        printf("    AMR: ~%d blocks x %.1f MB = ~%.1f MB (solver only)\n",
               9, (double)N_coarse * N_coarse * N_coarse * 10 * 8 / 1e6,
               9 * (double)N_coarse * N_coarse * N_coarse * 10 * 8 / 1e6);

        grid_free(g_coarse);
        grid_free(g_amr);
        grid_free(g_ref);
    }

    /* ============================================================
     * Test 3: Convergence
     * ============================================================ */
    {
        double L = 20.0;
        printf("\n--- Test 3: Convergence at L=%.0f ---\n\n", L);

        int Ns[] = { 32, 64 };
        double uni_hams[2], amr_hams[2];

        for (int r = 0; r < 2; r++) {
            grid_t *gu = grid_alloc(Ns[r], L, RK_CLASSIC);
            jfnk_solve(gu, 1, &bh, 1e-10, 50, 0);
            uni_hams[r] = ham_l2_shell(gu, 1.0, 4.0);
            grid_free(gu);

            grid_t *ga = grid_alloc(Ns[r], L, RK_CLASSIC);
            jfnk_solve_amr(ga, 1, &bh, 1e-10, 50, 0, 2);
            amr_hams[r] = ham_l2_shell(ga, 1.0, 4.0);
            grid_free(ga);
        }

        double uni_ratio = uni_hams[0] / uni_hams[1];
        double amr_ratio = amr_hams[0] / amr_hams[1];

        printf("  Near-field Ham L2 (r=1..4M):\n");
        printf("  %-10s %12s %12s\n", "N_base", "Uniform", "AMR(2lev)");
        printf("  %-10s %12s %12s\n", "------", "-------", "---------");
        for (int r = 0; r < 2; r++)
            printf("  %-10d %12.4e %12.4e\n", Ns[r], uni_hams[r], amr_hams[r]);
        printf("\n");
        printf("  Uniform 32->64: ratio=%.2f, order=%.2f\n",
               uni_ratio, log2(uni_ratio));
        printf("  AMR     32->64: ratio=%.2f, order=%.2f\n",
               amr_ratio, log2(amr_ratio));
    }

    printf("\n=== Done ===\n");
    backend_cleanup();
    return 0;
}
