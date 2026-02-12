/*
 * Lattice — 3D Numerical Relativity
 * Single Schwarzschild puncture evolution test.
 *
 * Brill-Lindquist puncture: M=1.0 at origin.
 * Grid: N=128, L=64, dx=0.5, CFL=0.25.
 * Evolve for T=10M.
 *
 * Pass criteria:
 *   - No crash (NaN/Inf)
 *   - Final min lapse < 0.5 (trumpet collapse)
 *   - Hamiltonian constraint bounded (peak < 1.0)
 *
 * Domain: boundary at 32M from puncture, clean evolution to ~10M.
 * Memory: ~1.3 GB with CK45 (fits easily on 16 GB).
 *
 * This is the second test in the validation ladder:
 *   flat spacetime -> single BH -> binary
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/puncture.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/boundary/sommerfeld.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <math.h>

/* Find minimum lapse over interior domain */
static double min_lapse(const grid_t *g)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;
    double min_val = 1.0e30;

    for (int k = lo; k < hi; k++) {
        for (int j = lo; j < hi; j++) {
            for (int i = lo; i < hi; i++) {
                int idx = IDX(g, i, j, k);
                double a = g->fields[FIELD_LAPSE][idx];
                if (a < min_val) min_val = a;
            }
        }
    }
    return min_val;
}

/* Check for NaN/Inf in all fields */
static int check_finite(const grid_t *g)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;

    for (int f = 0; f < NUM_FIELDS; f++) {
        for (int k = lo; k < hi; k++) {
            for (int j = lo; j < hi; j++) {
                for (int i = lo; i < hi; i++) {
                    int idx = IDX(g, i, j, k);
                    double v = g->fields[f][idx];
                    if (!isfinite(v)) return 0;
                }
            }
        }
    }
    return 1;
}

int main(void)
{
    printf("=== Single BH Evolution Test ===\n");
    fflush(stdout);

    /* N=128, L=64, dx=0.5, T=10M.
     * Boundary at 32M — no reflections reach puncture by T=10M. */
    sim_params_t p = default_params();
    p.N     = 128;
    p.L     = 64.0;
    p.CFL   = 0.25;
    p.sigma = 0.3;
    p.dx    = p.L / p.N;
    p.dt    = p.CFL * p.dx;

    double T_final = 10.0;
    p.num_steps = (int)(T_final / p.dt + 0.5);

    printf("  Initializing backend...\n");
    fflush(stdout);
    backend_init();

    printf("  Allocating grid (N=%d, L=%.1f)...\n", p.N, p.L);
    fflush(stdout);
    grid_t *g = grid_alloc(p.N, p.L, p.rk_method);

    /* Recompute after possible padding */
    p.N  = g->N;
    p.dx = g->dx;
    p.dt = p.CFL * p.dx;
    p.num_steps = (int)(T_final / p.dt + 0.5);

    printf("  N = %d, dx = %.4f, dt = %.6f, steps = %d\n",
           g->N, g->dx, p.dt, p.num_steps);
    fflush(stdout);

    /* Brill-Lindquist: single puncture M=1 at origin */
    printf("  Setting Brill-Lindquist initial data (M=1.0)...\n");
    fflush(stdout);
    double mass = 1.0;
    double center[1][3] = {{0.0, 0.0, 0.0}};
    set_brill_lindquist(g, 1, &mass, center);

    double ham0 = compute_constraint_l2(g);
    double ml0  = min_lapse(g);
    printf("  Initial: Ham L2 = %.6e, min lapse = %.6f\n", ham0, ml0);
    printf("  Starting evolution...\n");
    fflush(stdout);

    /* Evolve */
    int diag_every = p.num_steps / 50;
    if (diag_every < 1) diag_every = 1;

    double ham_peak = 0.0;
    int crashed = 0;

    for (int step = 1; step <= p.num_steps; step++) {
        rk4_step(g, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);

        double pct = 100.0 * step / p.num_steps;

        if (step % diag_every == 0 || step == p.num_steps) {
            if (!check_finite(g)) {
                printf("  CRASH: NaN/Inf at step %d (t = %.2f M)\n",
                       step, step * p.dt);
                fflush(stdout);
                crashed = 1;
                break;
            }

            double ham = compute_constraint_l2(g);
            double ml  = min_lapse(g);
            double t   = step * p.dt;

            if (ham > ham_peak) ham_peak = ham;

            printf("  step %4d/%d  [%5.1f%%]  t = %6.2f M  Ham L2 = %.4e  min(alpha) = %.4f\n",
                   step, p.num_steps, pct, t, ham, ml);
            fflush(stdout);
        } else {
            printf("  step %4d/%d  [%5.1f%%]\n", step, p.num_steps, pct);
            fflush(stdout);
        }
    }

    if (!crashed) {
        double ham_final = compute_constraint_l2(g);
        double ml_final  = min_lapse(g);

        printf("\n  Final:    Ham L2 = %.6e, min lapse = %.6f\n", ham_final, ml_final);
        printf("  Peak Ham: %.6e\n", ham_peak);

        /* Pass criteria */
        int lapse_ok      = (ml_final < 0.5);
        int finite_ok     = check_finite(g);
        int constraint_ok = (ham_peak < 1.0);

        printf("\n  Lapse collapsed:     %s (final=%.4f, want < 0.5)\n",
               lapse_ok ? "YES" : "NO", ml_final);
        printf("  Fields finite:       %s\n",
               finite_ok ? "YES" : "NO");
        printf("  Constraints bounded: %s (peak=%.4e, want < 1.0)\n",
               constraint_ok ? "YES" : "NO", ham_peak);

        int passed = lapse_ok && finite_ok && constraint_ok;
        printf("\n  %s\n", passed ? "PASSED" : "FAILED");

        grid_free(g);
        backend_cleanup();
        return passed ? 0 : 1;
    }

    grid_free(g);
    backend_cleanup();
    return 1;
}
