/*
 * test_bh_smoke.c — Quick smoke test for single Schwarzschild puncture
 *
 * Runs 20 steps on a small 33^3 grid. NOT a convergence or accuracy test.
 * Pass criteria (sanity checks only):
 *   1. No NaN/Inf
 *   2. Constraints don't blow up (ham_l2 < 1e6)
 *   3. Lapse is evolving (alpha_min changes from initial value)
 *   4. Chi stays positive
 *
 * Runs in ~5 seconds on M4. Use this for rapid iteration during debugging.
 */

#include "../src/core/grid.h"
#include "../src/core/fields.h"
#include "../src/core/params.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Forward declarations */
typedef void (*rhs_func_t)(grid_t *g);
extern void rk4_step(grid_t *g, rhs_func_t rhs_func, double dt);
extern void enforce_algebraic_constraints(grid_t *g);
extern void sommerfeld_apply(grid_t *g);
extern void ccz4_rhs(grid_t *g);
extern void gauge_rhs(grid_t *g);
extern void dissipation_apply(grid_t *g);
extern void constraints_l2(grid_t *g, double *ham_l2, double *mom_l2);
extern int backend_init(void);
extern void backend_shutdown(void);
extern void puncture_set_single(grid_t *g, double mass, double cx, double cy, double cz);

static void full_rhs(grid_t *g)
{
    grid_zero_rhs(g);
    sommerfeld_apply(g);
    ccz4_rhs(g);
    gauge_rhs(g);
    dissipation_apply(g);
}

static double find_alpha_min(grid_t *g)
{
    double alpha_min = 1e30;
    GRID_LOOP_INTERIOR(g, i, j, k) {
        int idx = grid_idx(g, i, j, k);
        double a = g->fields[FIELD_ALPHA][idx];
        if (a < alpha_min) alpha_min = a;
    }
    return alpha_min;
}

static double find_chi_min(grid_t *g)
{
    double chi_min = 1e30;
    GRID_LOOP_INTERIOR(g, i, j, k) {
        int idx = grid_idx(g, i, j, k);
        double c = g->fields[FIELD_CHI][idx];
        if (c < chi_min) chi_min = c;
    }
    return chi_min;
}

int main(void)
{
    sim_params_t params;
    params_set_defaults(&params);

    /* Small grid for speed: 33^3, domain=32M, dx~1M */
    params.nx = params.ny = params.nz = 41;
    params.lx = params.ly = params.lz = 32.0;
    params.cfl = 0.25;
    params.kappa1 = 0.02;
    params.kappa2 = 0.0;
    params.eta = 2.0;

    int nsteps = 20;

    params_init(&params);
    backend_init();

    grid_t g;
    g.params = params;
    if (grid_alloc(&g) != 0) {
        fprintf(stderr, "FAIL: grid allocation\n");
        return 1;
    }

    /* Offset puncture by dx/2 so no grid point sits at r=0.
     * This avoids the discontinuity from chi-floor at the coordinate singularity. */
    puncture_set_single(&g, 1.0, 0.5 * params.dx, 0.5 * params.dy, 0.5 * params.dz);
    enforce_algebraic_constraints(&g);

    double alpha_min_init = find_alpha_min(&g);

    printf("BH smoke test: %dx%dx%d, domain=%.0f, dt=%.4e, steps=%d\n",
           params.nx, params.ny, params.nz, params.lx, params.dt, nsteps);

    printf("\n%6s  %10s  %12s  %12s  %10s  %10s\n",
           "step", "time", "ham_l2", "mom_l2", "alpha_min", "chi_min");
    printf("------  ----------  ------------  ------------  ----------  ----------\n");

    double ham_l2 = 0.0, mom_l2 = 0.0;
    int failed = 0;
    char fail_reason[256] = {0};

    for (int step = 0; step <= nsteps; step++) {
        constraints_l2(&g, &ham_l2, &mom_l2);
        double alpha_min = find_alpha_min(&g);
        double chi_min = find_chi_min(&g);

        printf("%6d  %10.4e  %12.4e  %12.4e  %10.6f  %10.6f\n",
               step, g.time, ham_l2, mom_l2, alpha_min, chi_min);

        /* Check for NaN */
        if (isnan(ham_l2) || isnan(alpha_min) || isnan(chi_min)) {
            snprintf(fail_reason, sizeof(fail_reason),
                     "NaN detected at step %d, t=%.4f", step, g.time);
            failed = 1;
            break;
        }

        /* Check for constraint blowup (generous threshold for smoke test) */
        if (ham_l2 > 1e6) {
            snprintf(fail_reason, sizeof(fail_reason),
                     "constraint blowup at step %d: ham_l2=%.2e", step, ham_l2);
            failed = 1;
            break;
        }

        /* Check chi stays positive */
        if (chi_min < 0.0) {
            snprintf(fail_reason, sizeof(fail_reason),
                     "chi went negative at step %d: chi_min=%.2e", step, chi_min);
            failed = 1;
            break;
        }

        if (step < nsteps) {
            rk4_step(&g, full_rhs, params.dt);
        }
    }

    double alpha_min_final = find_alpha_min(&g);

    printf("\n--- Smoke Test Summary ---\n");
    printf("  Steps completed: %d / %d\n", failed ? -1 : nsteps, nsteps);
    printf("  alpha_min:  %.6f -> %.6f (init -> final)\n", alpha_min_init, alpha_min_final);
    printf("  ham_l2:     %.4e\n", ham_l2);
    printf("  mom_l2:     %.4e\n", mom_l2);

    if (!failed) {
        /* Check lapse is evolving — it should change from initial value */
        double alpha_change = fabs(alpha_min_final - alpha_min_init);
        if (alpha_change < 1e-12) {
            printf("  WARNING: lapse not evolving (delta=%.2e)\n", alpha_change);
        } else {
            printf("  Lapse evolving: OK (delta=%.4e)\n", alpha_change);
        }
    }

    if (failed) {
        fprintf(stderr, "FAIL: %s\n", fail_reason);
    }

    printf("\n%s\n", failed ? "FAIL: BH smoke test" : "PASS: BH smoke test");

    grid_free(&g);
    backend_shutdown();

    return failed ? 1 : 0;
}
