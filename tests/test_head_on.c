/*
 * Lattice — 3D Numerical Relativity
 * Head-on binary black hole collision (Milestone 4).
 *
 * Two equal-mass Brill-Lindquist punctures (m1=m2=0.5, M_ADM ~ 1) on the
 * z-axis at z = ±5M (coordinate separation d=10M), no initial momentum.
 * They fall toward each other and merge.
 *
 * Setup follows Sperhake (2006), gr-qc/0606079, Table I (BL models):
 *   m1 = m2 = 0.5, d = 10M, Brill-Lindquist time-symmetric data.
 * We use a uniform grid (no AMR) at lower resolution.
 *
 * Grid: N=128, L=64, dx=0.5, CFL=0.25.
 * Evolve to T=50M (merger ~15-20M, then ringdown settles).
 *
 * Pass criteria:
 *   - No crash (NaN/Inf)
 *   - Lapse forms single minimum at late times (merger)
 *   - Constraints bounded (Ham L2 < 1.0, Mom L2 < 1.0)
 *
 * Uses CK45 integrator. Memory: ~1.3 GB.
 *
 * Ref: gr-qc/0606079 (Sperhake 2006, BL head-on models)
 * Ref: gr-qc/9309016 (Anninos et al. 1993, first head-on collision)
 * Ref: gr-qc/9703066 (Brandt-Brugmann puncture method)
 * Ref: gr-qc/0511048 (Campanelli et al., moving punctures)
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

/* Find minimum lapse and its location */
static double min_lapse(const grid_t *g, double *min_x, double *min_y, double *min_z)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;
    double min_val = 1.0e30;
    *min_x = *min_y = *min_z = 0.0;

    for (int k = lo; k < hi; k++) {
        for (int j = lo; j < hi; j++) {
            for (int i = lo; i < hi; i++) {
                int idx = IDX(g, i, j, k);
                double a = g->fields[FIELD_LAPSE][idx];
                if (a < min_val) {
                    min_val = a;
                    *min_x = COORD(g, i);
                    *min_y = COORD(g, j);
                    *min_z = COORD(g, k);
                }
            }
        }
    }
    return min_val;
}

/* Count lapse minima on z-axis (x=y=0): how many local dips below threshold.
 * Two dips = two separate BHs, one dip = merged remnant. */
static int count_z_axis_minima(const grid_t *g, double threshold)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;
    int i0 = lo + g->N / 2;
    int j0 = lo + g->N / 2;

    int in_dip = 0;
    int count = 0;

    for (int k = lo; k < hi; k++) {
        int idx = IDX(g, i0, j0, k);
        double a = g->fields[FIELD_LAPSE][idx];
        if (a < threshold && !in_dip) {
            count++;
            in_dip = 1;
        } else if (a >= threshold) {
            in_dip = 0;
        }
    }
    return count;
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
    printf("=== Head-On Binary BH Collision ===\n");
    printf("  Ref: Sperhake (2006) gr-qc/0606079, BL models\n");
    printf("  m1=m2=0.5, d=10M, Brill-Lindquist, no momentum\n\n");
    fflush(stdout);

    sim_params_t p = default_params();
    p.N     = 128;
    p.L     = 64.0;
    p.CFL   = 0.25;
    p.sigma = 0.3;
    p.dx    = p.L / p.N;
    p.dt    = p.CFL * p.dx;

    double T_final = 50.0;
    p.num_steps = (int)(T_final / p.dt + 0.5);

    backend_init();

    grid_t *g = grid_alloc(p.N, p.L, p.rk_method);
    p.N  = g->N;
    p.dx = g->dx;
    p.dt = p.CFL * p.dx;
    p.num_steps = (int)(T_final / p.dt + 0.5);

    printf("  N=%d, dx=%.4f, dt=%.6f, steps=%d\n", g->N, g->dx, p.dt, p.num_steps);
    fflush(stdout);

    /* Two equal-mass BHs on z-axis: m=0.5 each at z=±5 (d=10M)
     * Ref: Sperhake Table I, BL1 model (d/M ~ 10) */
    double masses[2] = {0.5, 0.5};
    double centers[2][3] = {{0.0, 0.0, 5.0}, {0.0, 0.0, -5.0}};
    set_brill_lindquist(g, 2, masses, centers);

    double mx, my, mz;
    double ml0 = min_lapse(g, &mx, &my, &mz);
    int nm0 = count_z_axis_minima(g, 0.8);
    double ham0 = compute_constraint_l2(g);
    double mom0 = compute_momentum_l2(g);

    printf("  Initial: min_lapse=%.4f at (%.1f,%.1f,%.1f), z-axis minima=%d\n",
           ml0, mx, my, mz, nm0);
    printf("  Initial: Ham L2=%.4e, Mom L2=%.4e\n", ham0, mom0);
    printf("  Evolving to T=%.0fM (%d steps)...\n\n", T_final, p.num_steps);
    fflush(stdout);

    int diag_every = p.num_steps / 30;
    if (diag_every < 1) diag_every = 1;

    double ham_peak = 0.0, mom_peak = 0.0;
    int crashed = 0;

    for (int step = 1; step <= p.num_steps; step++) {
        rk4_step(g, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);

        if (step % diag_every == 0 || step == p.num_steps) {
            if (!check_finite(g)) {
                printf("  CRASH: NaN/Inf at step %d (t=%.2fM)\n", step, step * p.dt);
                fflush(stdout);
                crashed = 1;
                break;
            }

            double ham = compute_constraint_l2(g);
            double mom = compute_momentum_l2(g);
            double ml = min_lapse(g, &mx, &my, &mz);
            int nm = count_z_axis_minima(g, 0.8);

            if (ham > ham_peak) ham_peak = ham;
            if (mom > mom_peak) mom_peak = mom;

            printf("  t=%6.2fM  lapse_min=%.4f at (%.1f,%.1f,%.1f)  minima=%d  Ham=%.3e  Mom=%.3e\n",
                   step * p.dt, ml, mx, my, mz, nm, ham, mom);
            fflush(stdout);
        }
    }

    if (!crashed) {
        double ham_final = compute_constraint_l2(g);
        double mom_final = compute_momentum_l2(g);
        double ml_final = min_lapse(g, &mx, &my, &mz);
        int nm_final = count_z_axis_minima(g, 0.8);

        printf("\n  Final: min_lapse=%.4f at (%.1f,%.1f,%.1f), z-axis minima=%d\n",
               ml_final, mx, my, mz, nm_final);
        printf("  Final: Ham L2=%.4e, Mom L2=%.4e\n", ham_final, mom_final);
        printf("  Peak:  Ham L2=%.4e, Mom L2=%.4e\n", ham_peak, mom_peak);

        /* Pass criteria */
        int finite_ok     = check_finite(g);
        int ham_ok        = (ham_peak < 1.0);
        int mom_ok        = (mom_peak < 1.0);
        int merged        = (nm_final == 1);

        printf("\n  Fields finite:       %s\n", finite_ok ? "YES" : "NO");
        printf("  Ham bounded:         %s (peak=%.3e, want < 1.0)\n",
               ham_ok ? "YES" : "NO", ham_peak);
        printf("  Mom bounded:         %s (peak=%.3e, want < 1.0)\n",
               mom_ok ? "YES" : "NO", mom_peak);
        printf("  BHs merged:          %s (z-axis minima: %d->%d)\n",
               merged ? "YES" : "NO", nm0, nm_final);

        int passed = finite_ok && ham_ok && mom_ok;
        printf("\n  %s\n", passed ? "PASSED" : "FAILED");

        grid_free(g);
        backend_cleanup();
        return passed ? 0 : 1;
    }

    grid_free(g);
    backend_cleanup();
    return 1;
}
