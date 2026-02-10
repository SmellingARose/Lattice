/*
 * test_flat.c — Tier 1 validation: flat spacetime stability
 *
 * Tests that flat Minkowski spacetime remains stable under evolution.
 * Pass criteria:
 *   - ham_l2 < 1e-10 after 1000 steps
 *   - mom_l2 < 1e-10 after 1000 steps
 *   - No NaN or crash
 *
 * Usage: ./build/test_flat [--nx N] [--steps S]
 */

#include "../src/core/grid.h"
#include "../src/core/fields.h"
#include "../src/core/params.h"

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
extern int backend_init(void);
extern void backend_shutdown(void);

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
        for (int a = 0; a < 6; a++)
            g->fields[FIELD_AT_BASE + a][idx] = 0.0;
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

static void full_rhs(grid_t *g)
{
    grid_zero_rhs(g);
    sommerfeld_apply(g);
    ccz4_rhs(g);
    gauge_rhs(g);
    dissipation_apply(g);
}

int main(int argc, char *argv[])
{
    sim_params_t params;
    params_set_defaults(&params);
    params.nx = params.ny = params.nz = 32;
    params.lx = params.ly = params.lz = 10.0;
    params.cfl = 0.25;

    int max_steps = 1000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nx") == 0 && i + 1 < argc)
            params.nx = params.ny = params.nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            max_steps = atoi(argv[++i]);
    }

    params_init(&params);
    backend_init();

    grid_t g;
    g.params = params;
    if (grid_alloc(&g) != 0) {
        fprintf(stderr, "FAIL: grid allocation\n");
        return 1;
    }

    set_flat_initial_data(&g);

    printf("test_flat: %dx%dx%d, %d steps, CFL=%.3f\n",
           params.nx, params.ny, params.nz, max_steps, params.cfl);

    /* Evolve */
    for (int step = 0; step < max_steps; step++) {
        rk4_step(&g, full_rhs, params.dt);

        /* Check for NaN every 100 steps */
        if (step % 100 == 0) {
            double alpha = g.fields[FIELD_ALPHA][grid_idx(&g, params.nx/2, params.ny/2, params.nz/2)];
            if (isnan(alpha)) {
                fprintf(stderr, "FAIL: NaN detected at step %d\n", step);
                grid_free(&g);
                backend_shutdown();
                return 1;
            }
        }
    }

    /* Check constraints */
    double ham_l2, mom_l2;
    constraints_l2(&g, &ham_l2, &mom_l2);

    printf("  ham_l2 = %.6e\n", ham_l2);
    printf("  mom_l2 = %.6e\n", mom_l2);

    int pass = (ham_l2 < 1e-10) && (mom_l2 < 1e-10);

    if (pass) {
        printf("PASS: flat spacetime stable\n");
    } else {
        printf("FAIL: constraint violation too large\n");
        if (ham_l2 >= 1e-10) printf("  ham_l2 = %.6e >= 1e-10\n", ham_l2);
        if (mom_l2 >= 1e-10) printf("  mom_l2 = %.6e >= 1e-10\n", mom_l2);
    }

    grid_free(&g);
    backend_shutdown();

    return pass ? 0 : 1;
}
