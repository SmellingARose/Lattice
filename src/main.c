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
#include "evolution/maxwell_rhs.h"
#include "boundary/sommerfeld.h"
#include "numerics/rk4.h"
#include "diagnostics/constraints.h"
#include "diagnostics/ah_finder.h"
#include "amr/mesh.h"
#include "amr/refine.h"
#include "backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declarations for output */
extern void output_1d_slice(const grid_t *g, int step, double time);
extern void output_mesh_1d_slice(const mesh_t *m, int step, double time);

static void print_usage(void)
{
    fprintf(stderr, "Usage: lattice [options]\n");
    fprintf(stderr, "\nSimulation:\n");
    fprintf(stderr, "  --N <int>              Grid points per side (default 32)\n");
    fprintf(stderr, "  --L <float>            Domain size (default 10)\n");
    fprintf(stderr, "  --steps <int>          Evolution steps (default 1000)\n");
    fprintf(stderr, "  --CFL <float>          CFL factor (default 0.25)\n");
    fprintf(stderr, "  --sigma <float>        KO dissipation (default 0.3)\n");
    fprintf(stderr, "  --output_every <int>   Output interval (default 0=off)\n");
    fprintf(stderr, "  --rk classic|ck45      Time integrator (default ck45)\n");
    fprintf(stderr, "  --puncture M,x,y,z[,Px,Py,Pz[,Sx,Sy,Sz[,Q]]]  Add puncture\n");
    fprintf(stderr, "\nCCZ4 constraint damping:\n");
    fprintf(stderr, "  --kappa1 <float>       Theta+Z_i damping (default 0.1)\n");
    fprintf(stderr, "  --kappa2 <float>       Theta mix in K eq (default 0)\n");
    fprintf(stderr, "  --kappa3 <float>       Z in Gamma eq (default 1)\n");
    fprintf(stderr, "  --covariant_z4         Use covariant Z4 (default on)\n");
    fprintf(stderr, "  --no_covariant_z4      Use non-covariant Z4\n");
    fprintf(stderr, "\nGauge:\n");
    fprintf(stderr, "  --lapse_coeff <float>  1+log coefficient c (default 2.0)\n");
    fprintf(stderr, "  --lapse_power <float>  Bona-Masso power p (default 1.0)\n");
    fprintf(stderr, "  --shift_Gamma_coeff <float>  Gamma-driver F (default 0.75)\n");
    fprintf(stderr, "  --eta <float>          Gamma-driver damping (default 1.0)\n");
    fprintf(stderr, "  --lapse_advec <float>  Lapse advection coeff (default 0)\n");
    fprintf(stderr, "  --shift_advec <float>  Shift advection coeff (default 0)\n");
    fprintf(stderr, "\nNoise reduction (arXiv:2404.01137):\n");
    fprintf(stderr, "  --cako / --no_cako     Chi-adjusted KO dissipation (default on)\n");
    fprintf(stderr, "  --per_field_sigma / --no_per_field_sigma  Per-field KO (default on)\n");
    fprintf(stderr, "  --ssl / --no_ssl       Slow-start lapse (default on)\n");
    fprintf(stderr, "  --cahd / --no_cahd     Constraint-adjusted H damping (default off)\n");
    fprintf(stderr, "  --sigma_gauge <float>  Gauge field KO sigma (default 0.99)\n");
    fprintf(stderr, "  --sigma_phys <float>   Physical field KO sigma (default 0.3)\n");
    fprintf(stderr, "  --cahd_coeff <float>   CAHD coefficient C (default 0.15)\n");
    fprintf(stderr, "  --ssl_h <float>        SSL Gaussian height (default 0.6)\n");
    fprintf(stderr, "  --ssl_sigma_t <float>  SSL Gaussian width (default 20.0)\n");
    fprintf(stderr, "  --ssl_total_mass <float>  SSL total mass M (default 1.0)\n");
    fprintf(stderr, "\nAMR:\n");
    fprintf(stderr, "  --amr                  Enable adaptive mesh refinement\n");
    fprintf(stderr, "  --N_root <int>         Root blocks per side (default 4)\n");
    fprintf(stderr, "  --N_block <int>        Cells per block side (default 32)\n");
    fprintf(stderr, "  --max_level <int>      Max refinement depth (default 6)\n");
    fprintf(stderr, "  --chi_refine <float>   Refinement threshold (default 0.1)\n");
    fprintf(stderr, "  --chi_coarsen <float>  Coarsening threshold (default 0.01)\n");
    fprintf(stderr, "  --regrid_every <int>   Regrid check interval (default 1)\n");
    fprintf(stderr, "\nInitial data:\n");
    fprintf(stderr, "  --hispid               Force HiSpID (high-spin) initial data\n");
    fprintf(stderr, "\nEinstein-Maxwell:\n");
    fprintf(stderr, "  --em                   Enable Einstein-Maxwell coupling\n");
    fprintf(stderr, "  --kappa_em <float>     EM constraint damping (default 0.1)\n");
    fprintf(stderr, "\nApparent horizon finder:\n");
    fprintf(stderr, "  --ah                   Enable AH finder\n");
    fprintf(stderr, "  --ah_every <int>       AH finder interval (default 100)\n");
    fprintf(stderr, "  --ah_guess <float>     Initial radius guess (default M/2)\n");
    fprintf(stderr, "  --ah_eta <float>       AH flow speed (default 10.0)\n");
    fprintf(stderr, "  --ah_n_theta <int>     AH polar resolution (default 16)\n");
    fprintf(stderr, "  --ah_n_phi <int>       AH azimuthal resolution (default 32)\n");
    fprintf(stderr, "  --ah_tol <float>       AH convergence tolerance (default 1e-6)\n");
    fprintf(stderr, "  --ah_max_iter <int>    AH max iterations (default 500)\n");
}

int main(int argc, char **argv)
{
    sim_params_t p = default_params();

    /* Puncture storage */
    puncture_data_t bhs[MAX_PUNCTURES];
    memset(bhs, 0, sizeof(bhs));
    int n_bh = 0;

    /* AH finder options */
    int ah_enabled = 0;
    int ah_every = 100;
    double ah_guess = -1.0; /* negative = auto from puncture mass */
    double ah_eta = 10.0;
    int ah_n_theta = 16;
    int ah_n_phi = 32;
    double ah_tol = 1.0e-6;
    int ah_max_iter = 500;

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
            /* Accept 4, 7, 10, or 11 comma-separated values:
             *   M,x,y,z                         (BL at rest)
             *   M,x,y,z,Px,Py,Pz               (with momentum)
             *   M,x,y,z,Px,Py,Pz,Sx,Sy,Sz      (with momentum + spin)
             *   M,x,y,z,Px,Py,Pz,Sx,Sy,Sz,Q    (with charge) */
            int nread = sscanf(s, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                               &bh->mass,
                               &bh->center[0], &bh->center[1], &bh->center[2],
                               &bh->momentum[0], &bh->momentum[1], &bh->momentum[2],
                               &bh->spin[0], &bh->spin[1], &bh->spin[2],
                               &bh->charge);
            if (nread == 4 || nread == 7 || nread == 10 || nread == 11) {
                n_bh++;
            } else {
                fprintf(stderr, "Error: --puncture expects M,x,y,z[,Px,Py,Pz[,Sx,Sy,Sz[,Q]]]\n");
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
        /* CCZ4 constraint damping */
        } else if (strcmp(argv[a], "--kappa1") == 0 && a + 1 < argc) {
            p.ccz4.kappa1 = atof(argv[++a]);
        } else if (strcmp(argv[a], "--kappa2") == 0 && a + 1 < argc) {
            p.ccz4.kappa2 = atof(argv[++a]);
        } else if (strcmp(argv[a], "--kappa3") == 0 && a + 1 < argc) {
            p.ccz4.kappa3 = atof(argv[++a]);
        } else if (strcmp(argv[a], "--covariant_z4") == 0) {
            p.ccz4.covariant_Z4 = true;
        } else if (strcmp(argv[a], "--no_covariant_z4") == 0) {
            p.ccz4.covariant_Z4 = false;
        /* Gauge */
        } else if (strcmp(argv[a], "--lapse_coeff") == 0 && a + 1 < argc) {
            p.gauge.lapse_coeff = atof(argv[++a]);
        } else if (strcmp(argv[a], "--lapse_power") == 0 && a + 1 < argc) {
            p.gauge.lapse_power = atof(argv[++a]);
        } else if (strcmp(argv[a], "--shift_Gamma_coeff") == 0 && a + 1 < argc) {
            p.gauge.shift_Gamma_coeff = atof(argv[++a]);
        } else if (strcmp(argv[a], "--eta") == 0 && a + 1 < argc) {
            p.gauge.eta = atof(argv[++a]);
        } else if (strcmp(argv[a], "--lapse_advec") == 0 && a + 1 < argc) {
            p.gauge.lapse_advec_coeff = atof(argv[++a]);
        } else if (strcmp(argv[a], "--shift_advec") == 0 && a + 1 < argc) {
            p.gauge.shift_advec_coeff = atof(argv[++a]);
        /* Noise reduction */
        } else if (strcmp(argv[a], "--cako") == 0) {
            p.noise.use_cako = 1;
        } else if (strcmp(argv[a], "--no_cako") == 0) {
            p.noise.use_cako = 0;
        } else if (strcmp(argv[a], "--per_field_sigma") == 0) {
            p.noise.use_per_field_sigma = 1;
        } else if (strcmp(argv[a], "--no_per_field_sigma") == 0) {
            p.noise.use_per_field_sigma = 0;
        } else if (strcmp(argv[a], "--ssl") == 0) {
            p.noise.use_ssl = 1;
        } else if (strcmp(argv[a], "--no_ssl") == 0) {
            p.noise.use_ssl = 0;
        } else if (strcmp(argv[a], "--cahd") == 0) {
            p.noise.use_cahd = 1;
        } else if (strcmp(argv[a], "--no_cahd") == 0) {
            p.noise.use_cahd = 0;
        } else if (strcmp(argv[a], "--sigma_gauge") == 0 && a + 1 < argc) {
            p.noise.sigma_gauge = atof(argv[++a]);
        } else if (strcmp(argv[a], "--sigma_phys") == 0 && a + 1 < argc) {
            p.noise.sigma_phys = atof(argv[++a]);
        } else if (strcmp(argv[a], "--cahd_coeff") == 0 && a + 1 < argc) {
            p.noise.cahd_coeff = atof(argv[++a]);
        } else if (strcmp(argv[a], "--ssl_h") == 0 && a + 1 < argc) {
            p.noise.ssl_h = atof(argv[++a]);
        } else if (strcmp(argv[a], "--ssl_sigma_t") == 0 && a + 1 < argc) {
            p.noise.ssl_sigma_t = atof(argv[++a]);
        } else if (strcmp(argv[a], "--ssl_total_mass") == 0 && a + 1 < argc) {
            p.noise.ssl_total_mass = atof(argv[++a]);
        /* Einstein-Maxwell */
        } else if (strcmp(argv[a], "--em") == 0) {
            p.em_enabled = 1;
        } else if (strcmp(argv[a], "--kappa_em") == 0 && a + 1 < argc) {
            p.kappa_em = atof(argv[++a]);
        /* Apparent horizon finder */
        } else if (strcmp(argv[a], "--ah") == 0) {
            ah_enabled = 1;
        } else if (strcmp(argv[a], "--ah_every") == 0 && a + 1 < argc) {
            ah_every = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--ah_guess") == 0 && a + 1 < argc) {
            ah_guess = atof(argv[++a]);
        } else if (strcmp(argv[a], "--ah_eta") == 0 && a + 1 < argc) {
            ah_eta = atof(argv[++a]);
        } else if (strcmp(argv[a], "--ah_n_theta") == 0 && a + 1 < argc) {
            ah_n_theta = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--ah_n_phi") == 0 && a + 1 < argc) {
            ah_n_phi = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--ah_tol") == 0 && a + 1 < argc) {
            ah_tol = atof(argv[++a]);
        } else if (strcmp(argv[a], "--ah_max_iter") == 0 && a + 1 < argc) {
            ah_max_iter = atoi(argv[++a]);
        /* Initial data */
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

    setbuf(stdout, NULL);
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

        /* Set initial data: solve on temporary uniform grid, copy to blocks.
         * The constraint solve (FAS multigrid) needs the full domain, so we
         * solve globally at base AMR resolution, then distribute to blocks. */
        if (n_bh > 0) {
            printf("  Initial data: Bowen-York, %d puncture(s)\n", n_bh);

            /* Solve initial data on temporary uniform grid at N_eff */
            grid_t *tmp = grid_alloc(N_eff, p.L, RK_CK45);
            set_bowen_york(tmp, n_bh, bhs);

            /* Copy solved fields from temp grid to each leaf block.
             * Both grids have the same dx at level 0. Block origin gives
             * the physical offset: i_temp = ghost + (i_block - ghost) + offset
             * where offset = round((origin - (-L/2)) / dx). */
            int ghost = GHOST_WIDTH;
            for (int bid = 0; bid < m->num_blocks; bid++) {
                block_t *b = m->blocks[bid];
                if (!b || !b->is_leaf) continue;

                /* Compute cell offset of this block in the global grid */
                int off[3];
                for (int d = 0; d < 3; d++)
                    off[d] = (int)((b->origin[d] + p.L * 0.5) / tmp->dx + 0.5);

                int Nt_b = b->grid->Ntotal;
                int Nt_g = tmp->Ntotal;

                for (int f = 0; f < NUM_FIELDS; f++) {
                    double *dst = b->grid->fields[f];
                    const double *src = tmp->fields[f];
                    /* Copy interior + ghost zones (clamp to temp grid bounds) */
                    for (int k = 0; k < Nt_b; k++) {
                        int kg = k - ghost + off[2] + ghost;
                        if (kg < 0) kg = 0;
                        if (kg >= Nt_g) kg = Nt_g - 1;
                        for (int j = 0; j < Nt_b; j++) {
                            int jg = j - ghost + off[1] + ghost;
                            if (jg < 0) jg = 0;
                            if (jg >= Nt_g) jg = Nt_g - 1;
                            for (int i = 0; i < Nt_b; i++) {
                                int ig = i - ghost + off[0] + ghost;
                                if (ig < 0) ig = 0;
                                if (ig >= Nt_g) ig = Nt_g - 1;
                                dst[k * Nt_b * Nt_b + j * Nt_b + i] =
                                    src[kg * Nt_g * Nt_g + jg * Nt_g + ig];
                            }
                        }
                    }
                }
            }
            grid_free(tmp);
        } else {
            printf("  Initial data: flat spacetime (AMR)\n");
            for (int bid = 0; bid < m->num_blocks; bid++) {
                block_t *b = m->blocks[bid];
                if (!b || !b->is_leaf) continue;
                set_flat_spacetime(b->grid);
            }
        }

        double ham0 = mesh_constraint_l2(m);
        printf("  Initial Ham L2 = %.6e\n", ham0);

        /* Select RHS function: combined CCZ4+Maxwell if EM enabled */
        rk4_rhs_func_t rhs_func = p.em_enabled
            ? ccz4_maxwell_rhs_point : ccz4_rhs_point;

        /* AH finder setup (AMR path) */
        ah_workspace_t *ah_ws = NULL;
        if (ah_enabled && n_bh > 0) {
            double r0 = ah_guess > 0 ? ah_guess : bhs[0].mass / 2.0;
            ah_ws = ah_alloc(ah_n_theta, ah_n_phi, bhs[0].center, r0);
            ah_ws->eta = ah_eta;
            printf("  AH finder: enabled, every %d steps, r_guess=%.4f\n",
                   ah_every, r0);
        }

        /* Evolution loop.
         * p.time tracks the simulation time for subcycling's temporal
         * interpolation (Berger-Oliger). dt is the coarsest-level step. */
        p.time = 0.0;
        for (int step = 1; step <= p.num_steps; step++) {
            rk4_step_mesh(m, &p, rhs_func, p.dt);
            p.time += p.dt;

            if (p.amr.regrid_every > 0 && step % p.amr.regrid_every == 0)
                mesh_regrid(m, &p.amr);

            if (p.output_every > 0 && step % p.output_every == 0)
                output_mesh_1d_slice(m, step, p.time);

            if (step % 100 == 0 || step == p.num_steps) {
                double ham = mesh_constraint_l2(m);
                printf("  step %5d  t=%.4f  Ham L2=%.6e  blocks=%d\n",
                       step, p.time, ham, mesh_num_leaves(m));
            }

            /* AH finder (AMR path) */
            if (ah_ws && ah_every > 0 && step % ah_every == 0) {
                int conv = ah_find_amr(ah_ws, m, ah_tol, ah_max_iter, 0);
                if (conv) {
                    ah_result_t ahr = ah_compute_diagnostics_amr(ah_ws, m);
                    printf("  AH step %5d: A=%.4f M_irr=%.4f |J|=%.4e r=%.4f\n",
                           step, ahr.area, ahr.mass_irr, ahr.spin_mag,
                           ahr.mean_radius);
                }
            }
        }

        ah_free(ah_ws);
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

        /* Select RHS function: combined CCZ4+Maxwell if EM enabled */
        rk4_rhs_func_t rhs_func = p.em_enabled
            ? ccz4_maxwell_rhs_point : ccz4_rhs_point;

        /* AH finder setup */
        ah_workspace_t *ah_ws = NULL;
        if (ah_enabled && n_bh > 0) {
            double r0 = ah_guess > 0 ? ah_guess : bhs[0].mass / 2.0;
            ah_ws = ah_alloc(ah_n_theta, ah_n_phi, bhs[0].center, r0);
            ah_ws->eta = ah_eta;
            printf("  AH finder: enabled, every %d steps, r_guess=%.4f\n",
                   ah_every, r0);
        }

        /* Time evolution */
        for (int step = 1; step <= p.num_steps; step++) {
            rk4_step(g, &p, rhs_func, apply_sommerfeld, p.dt);

            if (step % 100 == 0 || step == p.num_steps) {
                double ham = compute_constraint_l2(g);
                printf("  step %5d  t = %.4f  Ham L2 = %.6e\n",
                       step, step * p.dt, ham);
            }

            if (p.output_every > 0 && step % p.output_every == 0) {
                output_1d_slice(g, step, step * p.dt);
            }

            /* AH finder */
            if (ah_ws && ah_every > 0 && step % ah_every == 0) {
                int conv = ah_find(ah_ws, g, ah_tol, ah_max_iter, 0);
                if (conv) {
                    ah_result_t ahr = ah_compute_diagnostics(ah_ws, g);
                    printf("  AH step %5d: A=%.4f M_irr=%.4f |J|=%.4e r=%.4f\n",
                           step, ahr.area, ahr.mass_irr, ahr.spin_mag,
                           ahr.mean_radius);
                }
            }
        }

        ah_free(ah_ws);
        grid_free(g);
    }

    backend_cleanup();

    return 0;
}
