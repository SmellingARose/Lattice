/*
 * Lattice — 3D Numerical Relativity Simulator
 *
 * CLI interface: allocate grid, set initial data, evolve with RK4.
 *
 * Usage:
 *   ./lattice --N 32 --steps 1000 --CFL 0.25 --output_every 100
 *   ./lattice --N 64 --steps 100 --puncture 1.0,0,0,0
 */

#include "core/grid.h"
#include "core/params.h"
#include "core/fields.h"
#include "initial_data/puncture.h"
#include "evolution/ccz4_rhs.h"
#include "boundary/sommerfeld.h"
#include "numerics/rk4.h"
#include "diagnostics/constraints.h"
#include "backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declaration for output */
extern void output_1d_slice(const grid_t *g, int step, double time);

static void print_usage(void)
{
    fprintf(stderr, "Usage: lattice [options]\n");
    fprintf(stderr, "  --N <int>           Grid points per side (default 32)\n");
    fprintf(stderr, "  --L <float>         Domain size (default 10)\n");
    fprintf(stderr, "  --steps <int>       Evolution steps (default 1000)\n");
    fprintf(stderr, "  --CFL <float>       CFL factor (default 0.25)\n");
    fprintf(stderr, "  --sigma <float>     KO dissipation (default 0.3)\n");
    fprintf(stderr, "  --output_every <int> Output interval (default 0=off)\n");
    fprintf(stderr, "  --puncture M,x,y,z  Add a puncture BH\n");
    fprintf(stderr, "  --rk classic|ck45   Time integrator (default ck45)\n");
}

int main(int argc, char **argv)
{
    sim_params_t p = default_params();

    /* Puncture storage */
    double bh_masses[16];
    double bh_centers[16][3];
    int n_bh = 0;

    /* Parse CLI args */
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--N") == 0 && a + 1 < argc) {
            p.N = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--L") == 0 && a + 1 < argc) {
            p.L = atof(argv[++a]);
        } else if (strcmp(argv[a], "--steps") == 0 && a + 1 < argc) {
            p.num_steps = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--CFL") == 0 && a + 1 < argc) {
            p.CFL = atof(argv[++a]);
        } else if (strcmp(argv[a], "--sigma") == 0 && a + 1 < argc) {
            p.sigma = atof(argv[++a]);
        } else if (strcmp(argv[a], "--output_every") == 0 && a + 1 < argc) {
            p.output_every = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--puncture") == 0 && a + 1 < argc) {
            if (n_bh >= 16) {
                fprintf(stderr, "Error: max 16 punctures\n");
                return 1;
            }
            char *s = argv[++a];
            if (sscanf(s, "%lf,%lf,%lf,%lf",
                       &bh_masses[n_bh],
                       &bh_centers[n_bh][0],
                       &bh_centers[n_bh][1],
                       &bh_centers[n_bh][2]) == 4) {
                n_bh++;
            } else {
                fprintf(stderr, "Error: --puncture expects M,x,y,z\n");
                return 1;
            }
        } else if (strcmp(argv[a], "--rk") == 0 && a + 1 < argc) {
            a++;
            if (strcmp(argv[a], "classic") == 0) {
                p.rk_method = RK_CLASSIC;
            } else if (strcmp(argv[a], "ck45") == 0) {
                p.rk_method = RK_CK45;
            } else {
                fprintf(stderr, "Error: --rk expects 'classic' or 'ck45'\n");
                return 1;
            }
        } else if (strcmp(argv[a], "--help") == 0 || strcmp(argv[a], "-h") == 0) {
            print_usage();
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[a]);
            print_usage();
            return 1;
        }
    }

    /* Recompute derived parameters */
    p.dx = p.L / p.N;
    p.dt = p.CFL * p.dx;

    printf("Lattice — 3D Numerical Relativity\n");
    printf("  N = %d, L = %.1f, dx = %.6f, dt = %.6f\n", p.N, p.L, p.dx, p.dt);
    printf("  steps = %d, sigma = %.2f, CFL = %.2f, rk = %s\n",
           p.num_steps, p.sigma, p.CFL,
           p.rk_method == RK_CK45 ? "ck45" : "classic");

    /* Allocate grid */
    backend_init();
    grid_t *g = grid_alloc(p.N, p.L, p.rk_method);

    /* Note: grid_alloc may pad N to a multiple of 16 */
    p.N  = g->N;
    p.dx = g->dx;
    p.dt = p.CFL * p.dx;

    printf("  Ntotal = %d (N=%d + 2*ghost=%d)\n", g->Ntotal, g->N, g->ghost);

    /* Set initial data */
    if (n_bh > 0) {
        printf("  Initial data: Brill-Lindquist, %d puncture(s)\n", n_bh);
        set_brill_lindquist(g, n_bh, bh_masses, (const double(*)[3])bh_centers);
    } else {
        printf("  Initial data: flat spacetime\n");
        set_flat_spacetime(g);
    }

    /* Initial constraint */
    double ham0 = compute_constraint_l2(g);
    printf("  Initial Ham L2 = %.6e\n", ham0);

    /* Time evolution */
    for (int step = 1; step <= p.num_steps; step++) {
        rk4_step(g, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);

        if (step % 100 == 0 || step == p.num_steps) {
            double ham = compute_constraint_l2(g);
            printf("  step %5d  t = %.4f  Ham L2 = %.6e\n",
                   step, step * p.dt, ham);
        }

        if (p.output_every > 0 && step % p.output_every == 0) {
            output_1d_slice(g, step, step * p.dt);
        }
    }

    /* Cleanup */
    grid_free(g);
    backend_cleanup();

    return 0;
}
