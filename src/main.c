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
#include "initial_data/bowen_york.h"
#include "evolution/ccz4_rhs.h"
#include "boundary/sommerfeld.h"
#include "numerics/rk4.h"
#include "diagnostics/constraints.h"
#include "amr/mesh.h"
#include "amr/refine.h"
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
    fprintf(stderr, "  --puncture M,x,y,z[,Px,Py,Pz[,Sx,Sy,Sz]]  Add a puncture BH\n");
    fprintf(stderr, "  --rk classic|ck45   Time integrator (default ck45)\n");
    fprintf(stderr, "\nAMR options:\n");
    fprintf(stderr, "  --amr               Enable adaptive mesh refinement\n");
    fprintf(stderr, "  --N_root <int>      Root blocks per side (default 4)\n");
    fprintf(stderr, "  --N_block <int>     Cells per block side (default 32)\n");
    fprintf(stderr, "  --max_level <int>   Max refinement depth (default 6)\n");
    fprintf(stderr, "  --chi_refine <float>  Refinement threshold (default 0.1)\n");
    fprintf(stderr, "  --chi_coarsen <float> Coarsening threshold (default 0.01)\n");
    fprintf(stderr, "  --regrid_every <int>  Regrid check interval (default 10)\n");
    fprintf(stderr, "\nInitial data options:\n");
    fprintf(stderr, "  --hispid            Force HiSpID (high-spin) initial data\n");
}

int main(int argc, char **argv)
{
    sim_params_t p = default_params();

    /* Puncture storage */
    puncture_data_t bhs[MAX_PUNCTURES];
    memset(bhs, 0, sizeof(bhs));
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
            if (n_bh >= MAX_PUNCTURES) {
                fprintf(stderr, "Error: max %d punctures\n", MAX_PUNCTURES);
                return 1;
            }
            char *s = argv[++a];
            puncture_data_t *bh = &bhs[n_bh];
            /* Accept 4, 7, or 10 comma-separated values:
             *   M,x,y,z                    (BL at rest)
             *   M,x,y,z,Px,Py,Pz          (with momentum)
             *   M,x,y,z,Px,Py,Pz,Sx,Sy,Sz (with momentum + spin) */
            int nread = sscanf(s, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                               &bh->mass,
                               &bh->center[0], &bh->center[1], &bh->center[2],
                               &bh->momentum[0], &bh->momentum[1], &bh->momentum[2],
                               &bh->spin[0], &bh->spin[1], &bh->spin[2]);
            if (nread == 4 || nread == 7 || nread == 10) {
                n_bh++;
            } else {
                fprintf(stderr, "Error: --puncture expects M,x,y,z[,Px,Py,Pz[,Sx,Sy,Sz]]\n");
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
        } else if (strcmp(argv[a], "--amr") == 0) {
            p.amr.enabled = 1;
        } else if (strcmp(argv[a], "--N_root") == 0 && a + 1 < argc) {
            p.amr.N_root = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--N_block") == 0 && a + 1 < argc) {
            p.amr.N_block = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--max_level") == 0 && a + 1 < argc) {
            p.amr.max_level = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--chi_refine") == 0 && a + 1 < argc) {
            p.amr.chi_refine = atof(argv[++a]);
        } else if (strcmp(argv[a], "--chi_coarsen") == 0 && a + 1 < argc) {
            p.amr.chi_coarsen = atof(argv[++a]);
        } else if (strcmp(argv[a], "--regrid_every") == 0 && a + 1 < argc) {
            p.amr.regrid_every = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--hispid") == 0) {
            set_hispid_override(1);
        } else if (strcmp(argv[a], "--help") == 0 || strcmp(argv[a], "-h") == 0) {
            print_usage();
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[a]);
            print_usage();
            return 1;
        }
    }

    backend_init();

    printf("Lattice — 3D Numerical Relativity\n");

    if (p.amr.enabled) {
        /* === AMR path === */
        mesh_t *m = mesh_create(p.amr.N_root, p.amr.N_block, p.L, p.rk_method);
        p.dx = m->dx_base;
        p.dt = p.CFL * p.dx;
        int N_eff = p.amr.N_root * p.amr.N_block;

        printf("  AMR mode: N_root=%d, N_block=%d, N_eff=%d, max_level=%d\n",
               p.amr.N_root, p.amr.N_block, N_eff, p.amr.max_level);
        printf("  L = %.1f, dx = %.6f, dt = %.6f\n", p.L, p.dx, p.dt);
        printf("  steps = %d, sigma = %.2f, CFL = %.2f, rk = %s\n",
               p.num_steps, p.sigma, p.CFL,
               p.rk_method == RK_CK45 ? "ck45" : "classic");
        printf("  chi_refine = %.4f, chi_coarsen = %.4f, regrid_every = %d\n",
               p.amr.chi_refine, p.amr.chi_coarsen, p.amr.regrid_every);
        printf("  blocks = %d (leaves = %d)\n",
               m->num_blocks, mesh_num_leaves(m));

        /* Set initial data on all leaf blocks (AMR uses BL for now) */
        double amr_masses[MAX_PUNCTURES];
        double amr_centers[MAX_PUNCTURES][3];
        for (int n = 0; n < n_bh; n++) {
            amr_masses[n] = bhs[n].mass;
            for (int d = 0; d < 3; d++)
                amr_centers[n][d] = bhs[n].center[d];
        }
        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *b = m->blocks[bid];
            if (!b || !b->is_leaf) continue;
            if (n_bh > 0)
                set_brill_lindquist_global(b->grid, b->origin, n_bh,
                                           amr_masses,
                                           (const double(*)[3])amr_centers);
            else
                set_flat_spacetime(b->grid);
        }

        printf("  Initial data: %s\n",
               n_bh > 0 ? "Brill-Lindquist (AMR)" : "flat spacetime (AMR)");

        double ham0 = mesh_constraint_l2(m);
        printf("  Initial Ham L2 = %.6e\n", ham0);

        /* Evolution loop.
         * p.time tracks the simulation time for subcycling's temporal
         * interpolation (Berger-Oliger). dt is the coarsest-level step. */
        p.time = 0.0;
        for (int step = 1; step <= p.num_steps; step++) {
            rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
            p.time += p.dt;

            if (p.amr.regrid_every > 0 && step % p.amr.regrid_every == 0)
                mesh_regrid(m, &p.amr);

            if (step % 100 == 0 || step == p.num_steps) {
                double ham = mesh_constraint_l2(m);
                printf("  step %5d  t=%.4f  Ham L2=%.6e  blocks=%d\n",
                       step, p.time, ham, mesh_num_leaves(m));
            }
        }

        mesh_free(m);
    } else {
        /* === Single-grid path (unchanged) === */
        p.dx = p.L / p.N;
        p.dt = p.CFL * p.dx;

        printf("  N = %d, L = %.1f, dx = %.6f, dt = %.6f\n",
               p.N, p.L, p.dx, p.dt);
        printf("  steps = %d, sigma = %.2f, CFL = %.2f, rk = %s\n",
               p.num_steps, p.sigma, p.CFL,
               p.rk_method == RK_CK45 ? "ck45" : "classic");

        grid_t *g = grid_alloc(p.N, p.L, p.rk_method);

        /* Note: grid_alloc may pad N to a multiple of 16 */
        p.N  = g->N;
        p.dx = g->dx;
        p.dt = p.CFL * p.dx;

        printf("  Ntotal = %d (N=%d + 2*ghost=%d)\n",
               g->Ntotal, g->N, g->ghost);

        /* Set initial data */
        if (n_bh > 0) {
            printf("  Initial data: Bowen-York, %d puncture(s)\n", n_bh);
            set_bowen_york(g, n_bh, bhs);
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

        grid_free(g);
    }

    backend_cleanup();

    return 0;
}
