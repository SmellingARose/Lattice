/*
 * main.c — Lattice NR simulator driver
 *
 * Main evolution loop:
 *   1. Set initial data (flat or puncture)
 *   2. Time-step with RK4
 *   3. Each RK stage: boundaries -> CCZ4 RHS -> gauge RHS -> dissipation
 *   4. Monitor constraints and output
 */

#include "core/grid.h"
#include "core/fields.h"
#include "core/params.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Forward declarations */
typedef void (*rhs_func_t)(grid_t *g);
extern void rk4_step(grid_t *g, rhs_func_t rhs_func, double dt);
extern void enforce_algebraic_constraints(grid_t *g);
extern void sommerfeld_apply(grid_t *g);
extern void ccz4_rhs(grid_t *g);
extern void gauge_rhs(grid_t *g);
extern void dissipation_apply(grid_t *g);
extern void constraints_l2(grid_t *g, double *ham_l2, double *mom_l2);
extern void output_scalars(grid_t *g, double ham_l2, double mom_l2, double alpha_min);
extern void output_slice(grid_t *g, int field_id, const char *filename);
extern int backend_init(void);
extern void backend_shutdown(void);

/*
 * Set flat spacetime initial data.
 * alpha = 1, chi = 1, gt = delta_{ij}, everything else = 0.
 */
static void set_flat_initial_data(grid_t *g)
{
    GRID_LOOP_ALL(g, i, j, k) {
        int idx = grid_idx(g, i, j, k);

        g->fields[FIELD_CHI][idx] = 1.0;
        g->fields[FIELD_GT11][idx] = 1.0;
        g->fields[FIELD_GT12][idx] = 0.0;
        g->fields[FIELD_GT13][idx] = 0.0;
        g->fields[FIELD_GT22][idx] = 1.0;
        g->fields[FIELD_GT23][idx] = 0.0;
        g->fields[FIELD_GT33][idx] = 1.0;
        g->fields[FIELD_TRKA][idx] = 0.0;
        for (int a = 0; a < 6; a++) {
            g->fields[FIELD_AT_BASE + a][idx] = 0.0;
        }
        g->fields[FIELD_GHAT1][idx] = 0.0;
        g->fields[FIELD_GHAT2][idx] = 0.0;
        g->fields[FIELD_GHAT3][idx] = 0.0;
        g->fields[FIELD_THETA][idx] = 0.0;
        g->fields[FIELD_ALPHA][idx] = 1.0;
        g->fields[FIELD_BETA1][idx] = 0.0;
        g->fields[FIELD_BETA2][idx] = 0.0;
        g->fields[FIELD_BETA3][idx] = 0.0;
        g->fields[FIELD_GBAUX1][idx] = 0.0;
        g->fields[FIELD_GBAUX2][idx] = 0.0;
        g->fields[FIELD_GBAUX3][idx] = 0.0;
    }
}

/*
 * Full RHS computation: called once per RK stage.
 *
 * Order matters:
 *   1. Zero RHS
 *   2. Apply boundary conditions (fills ghost zones)
 *   3. CCZ4 physics RHS
 *   4. Gauge RHS (must be after CCZ4 — reads dt Ghat^i for B^i eq)
 *   5. KO dissipation
 */
static void full_rhs(grid_t *g)
{
    grid_zero_rhs(g);
    sommerfeld_apply(g);
    ccz4_rhs(g);
    gauge_rhs(g);
    dissipation_apply(g);
}

/*
 * Find minimum of alpha (lapse) over the interior.
 */
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

int main(int argc, char *argv[])
{
    /* Initialize parameters */
    sim_params_t params;
    params_set_defaults(&params);

    /* Parse command-line overrides */
    int max_steps = 1000;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nx") == 0 && i + 1 < argc) {
            params.nx = params.ny = params.nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--domain") == 0 && i + 1 < argc) {
            params.lx = params.ly = params.lz = atof(argv[++i]);
        } else if (strcmp(argv[i], "--cfl") == 0 && i + 1 < argc) {
            params.cfl = atof(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output-every") == 0 && i + 1 < argc) {
            params.output_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--kappa1") == 0 && i + 1 < argc) {
            params.kappa1 = atof(argv[++i]);
        } else if (strcmp(argv[i], "--eta") == 0 && i + 1 < argc) {
            params.eta = atof(argv[++i]);
        }
    }

    params_init(&params);

    printf("Lattice NR Simulator\n");
    printf("Grid: %d x %d x %d (padded: %d)\n",
           params.nx, params.ny, params.nz, params.nx_pad);
    printf("Domain: %.1f x %.1f x %.1f\n", params.lx, params.ly, params.lz);
    printf("dx = %.4e, dt = %.4e, CFL = %.3f\n", params.dx, params.dt, params.cfl);
    printf("kappa1 = %.4f, kappa2 = %.4f, eta = %.2f\n",
           params.kappa1, params.kappa2, params.eta);
    printf("Fields: %d, Steps: %d\n\n", params.num_fields, max_steps);

    /* Initialize backend */
    if (backend_init() != 0) {
        fprintf(stderr, "Error: backend init failed\n");
        return 1;
    }

    /* Allocate grid */
    grid_t g;
    g.params = params;
    if (grid_alloc(&g) != 0) {
        fprintf(stderr, "Error: grid allocation failed\n");
        return 1;
    }

    /* Set initial data */
    set_flat_initial_data(&g);

    printf("%6s  %10s  %12s  %12s  %10s\n",
           "step", "time", "ham_l2", "mom_l2", "alpha_min");
    printf("------  ----------  ------------  ------------  ----------\n");

    /* Main evolution loop */
    for (int step = 0; step <= max_steps; step++) {
        /* Output diagnostics */
        if (step % params.output_every == 0) {
            double ham_l2 = 0.0, mom_l2 = 0.0;
            constraints_l2(&g, &ham_l2, &mom_l2);
            double alpha_min = find_alpha_min(&g);

            printf("%6d  %10.4e  %12.4e  %12.4e  %10.6f\n",
                   step, g.time, ham_l2, mom_l2, alpha_min);

            output_scalars(&g, ham_l2, mom_l2, alpha_min);
        }

        if (step < max_steps) {
            rk4_step(&g, full_rhs, params.dt);
        }
    }

    printf("\nSimulation complete.\n");

    /* Cleanup */
    grid_free(&g);
    backend_shutdown();

    return 0;
}
