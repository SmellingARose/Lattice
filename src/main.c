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
#include "numerics/rk4.h"
#include "diagnostics/constraints.h"
#include "diagnostics/ah_finder.h"
#include "diagnostics/psi4.h"
#include "diagnostics/bh_tracker.h"
#ifdef LATTICE_HDF5
#include "diagnostics/cce_worldtube.h"
#endif
#include "io/checkpoint.h"
#include "amr/mesh.h"
#include "amr/refine.h"
#include "backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Forward declaration for output */
extern void output_mesh_1d_slice(const mesh_t *m, int step, double time);

/*
 * GPU-accelerated constraint/momentum L2 norms for AMR meshes.
 * Loops over device-resident level packs, accumulates raw (sum, vol)
 * from each, combines into a single volume-weighted L2 norm.
 * Falls back to CPU mesh_constraint_l2 if not on GPU or no level packs.
 */
static double gpu_constraint_l2(mesh_t *m)
{
    if (!backend_is_gpu() || m->max_level == 0 || !m->level_packs[0])
        return mesh_constraint_l2(m);

    double total_sum = 0.0, total_vol = 0.0;
    for (int L = 0; L <= m->max_level; L++) {
        meshblock_pack_t *pack = m->level_packs[L];
        if (!pack || pack->n_blocks == 0) continue;
        backend_activate_pack(pack);
        double s, v;
        backend_constraint_l2_raw_packed(pack, &s, &v);
        total_sum += s;
        total_vol += v;
    }
    return (total_vol > 0.0) ? sqrt(total_sum / total_vol) : 0.0;
}

static double gpu_momentum_l2(mesh_t *m)
{
    if (!backend_is_gpu() || m->max_level == 0 || !m->level_packs[0])
        return mesh_momentum_l2(m);

    double total_sum = 0.0, total_vol = 0.0;
    for (int L = 0; L <= m->max_level; L++) {
        meshblock_pack_t *pack = m->level_packs[L];
        if (!pack || pack->n_blocks == 0) continue;
        backend_activate_pack(pack);
        double s, v;
        backend_momentum_l2_raw_packed(pack, &s, &v);
        total_sum += s;
        total_vol += v;
    }
    return (total_vol > 0.0) ? sqrt(total_sum / (3.0 * total_vol)) : 0.0;
}

/*
 * GPU-accelerated Psi4 extraction on device-resident level packs.
 * Uses the coarsest level pack (level 0) which covers the extraction sphere.
 * Falls back to CPU psi4_extract if not on GPU.
 */
static void gpu_psi4_extract(psi4_workspace_t *ws, mesh_t *m)
{
    if (!backend_is_gpu() || !m->level_packs[0]) {
        psi4_extract(ws, m);
        return;
    }
    meshblock_pack_t *pack = m->level_packs[0];
    backend_activate_pack(pack);
    backend_ghost_exchange_packed(pack);
    backend_psi4_extract_packed(pack, ws, m);
}

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
    fprintf(stderr, "  --rk classic|ck45      Time integrator (default classic)\n");
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
    fprintf(stderr, "  --N_block <int>        Cells per block side (default 32)\n");
    fprintf(stderr, "  --block-size <int>     Alias for --N_block\n");
    fprintf(stderr, "  --max_level <int>      Max refinement depth (default 6)\n");
    fprintf(stderr, "  --chi_refine <float>   Refinement threshold (default 0.1)\n");
    fprintf(stderr, "  --chi_coarsen <float>  Coarsening threshold (default 0.01)\n");
    fprintf(stderr, "  --regrid_every <int>   Regrid check interval (default 1)\n");
    fprintf(stderr, "  --amr-levels <int>    Initial data solver levels (default: max-level)\n");
    fprintf(stderr, "  --refine-c <float>     Finest box radius = C * M (default 4.0)\n");
    fprintf(stderr, "  --refine-beta <float>  Level growth ratio (default 1.516)\n");
    fprintf(stderr, "  --diag_level <int>     Fire diagnostics at this AMR level's dt (-1=off)\n");
    fprintf(stderr, "\nInitial data:\n");
    fprintf(stderr, "  --hispid               Force HiSpID (high-spin) initial data\n");
    fprintf(stderr, "\nBoundary conditions:\n");
    fprintf(stderr, "  --bc sommerfeld|cp     Boundary type (default cp)\n");
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
    fprintf(stderr, "\nPsi4 wave extraction:\n");
    fprintf(stderr, "  --psi4                 Enable Psi4 extraction\n");
    fprintf(stderr, "  --psi4_every <int>     Extraction interval (default 10)\n");
    fprintf(stderr, "  --psi4_radius <float>  Extraction radius (default 50)\n");
    fprintf(stderr, "  --psi4_l_max <int>     Max l for modes (default 4)\n");
    fprintf(stderr, "  --psi4_n_theta <int>   Polar resolution (default 32)\n");
    fprintf(stderr, "  --psi4_n_phi <int>     Azimuthal resolution (default 64)\n");
#ifdef LATTICE_HDF5
    fprintf(stderr, "\nCCE worldtube output (requires HDF5):\n");
    fprintf(stderr, "  --cce                  Enable CCE worldtube output\n");
    fprintf(stderr, "  --cce_every <int>      Extraction interval (default 1)\n");
    fprintf(stderr, "  --cce_radius <float>   Extraction radius (default 100)\n");
    fprintf(stderr, "  --cce_lmax <int>       Angular resolution (default 16)\n");
#endif
    fprintf(stderr, "\nN-body BH tracker:\n");
    fprintf(stderr, "  --tracker              Enable multi-BH tracker\n");
    fprintf(stderr, "  --tracker_every <int>  Tracker interval (default: ah_every or 10)\n");
    fprintf(stderr, "\nCheckpoint/restart:\n");
    fprintf(stderr, "  --checkpoint-every <int>  Save checkpoint every N steps (0=off)\n");
    fprintf(stderr, "  --restart <file>          Resume from checkpoint file\n");
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

    /* Psi4 extraction options */
    int psi4_enabled = 0;
    int psi4_every = 10;
    double psi4_radius = 50.0;
    int psi4_l_max = 4;
    int psi4_n_theta = 32;
    int psi4_n_phi = 64;

#ifdef LATTICE_HDF5
    /* CCE worldtube output options */
    int cce_enabled = 0;
    int cce_every = 1;
    double cce_radius = 100.0;
    int cce_lmax = 16;
#endif

    /* N-body BH tracker */
    int tracker_enabled = 0;
    int tracker_every = -1;  /* -1 = use ah_every or default 10 */

    /* NaN/Inf checking */
    int nan_check_every = 0;  /* 0 = disabled; >0 = check every N steps */

    /* Checkpoint/restart */
    int checkpoint_every = 0;
    const char *restart_file = NULL;

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
        } else if ((strcmp(argv[a], "--N_block") == 0 ||
                    strcmp(argv[a], "--block-size") == 0) && a + 1 < argc) {
            p.amr.N_block = atoi(argv[++a]);
            if (p.amr.N_block < 8 || p.amr.N_block % 2 != 0) {
                fprintf(stderr, "Error: --block-size must be even and >= 8\n");
                return 1;
            }
        } else if (strcmp(argv[a], "--max_level") == 0 && a + 1 < argc) {
            p.amr.max_level = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--chi_refine") == 0 && a + 1 < argc) {
            p.amr.chi_refine = atof(argv[++a]);
        } else if (strcmp(argv[a], "--chi_coarsen") == 0 && a + 1 < argc) {
            p.amr.chi_coarsen = atof(argv[++a]);
        } else if (strcmp(argv[a], "--regrid_every") == 0 && a + 1 < argc) {
            p.amr.regrid_every = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--amr-levels") == 0 && a + 1 < argc) {
            p.amr.solver_levels = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--refine-c") == 0 && a + 1 < argc) {
            p.amr.refine_c = atof(argv[++a]);
        } else if (strcmp(argv[a], "--refine-beta") == 0 && a + 1 < argc) {
            p.amr.refine_beta = atof(argv[++a]);
        } else if (strcmp(argv[a], "--diag_level") == 0 && a + 1 < argc) {
            p.diag_level = atoi(argv[++a]);
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
        /* Boundary conditions */
        } else if (strcmp(argv[a], "--bc") == 0 && a + 1 < argc) {
            a++;
            if (strcmp(argv[a], "sommerfeld") == 0) {
                p.bc_type = BC_SOMMERFELD;
            } else if (strcmp(argv[a], "cp") == 0) {
                p.bc_type = BC_CONSTRAINT_PRESERVING;
            } else {
                fprintf(stderr, "Error: --bc expects 'sommerfeld' or 'cp'\n");
                return 1;
            }
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
        /* Psi4 wave extraction */
        } else if (strcmp(argv[a], "--psi4") == 0) {
            psi4_enabled = 1;
        } else if (strcmp(argv[a], "--psi4_every") == 0 && a + 1 < argc) {
            psi4_every = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--psi4_radius") == 0 && a + 1 < argc) {
            psi4_radius = atof(argv[++a]);
        } else if (strcmp(argv[a], "--psi4_l_max") == 0 && a + 1 < argc) {
            psi4_l_max = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--psi4_n_theta") == 0 && a + 1 < argc) {
            psi4_n_theta = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--psi4_n_phi") == 0 && a + 1 < argc) {
            psi4_n_phi = atoi(argv[++a]);
#ifdef LATTICE_HDF5
        /* CCE worldtube output */
        } else if (strcmp(argv[a], "--cce") == 0) {
            cce_enabled = 1;
        } else if (strcmp(argv[a], "--cce_every") == 0 && a + 1 < argc) {
            cce_every = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--cce_radius") == 0 && a + 1 < argc) {
            cce_radius = atof(argv[++a]);
        } else if (strcmp(argv[a], "--cce_lmax") == 0 && a + 1 < argc) {
            cce_lmax = atoi(argv[++a]);
#endif
        /* N-body BH tracker */
        } else if (strcmp(argv[a], "--tracker") == 0) {
            tracker_enabled = 1;
        } else if (strcmp(argv[a], "--tracker_every") == 0 && a + 1 < argc) {
            tracker_every = atoi(argv[++a]);
            tracker_enabled = 1;
        /* Checkpoint/restart */
        } else if (strcmp(argv[a], "--nan-check-every") == 0 && a + 1 < argc) {
            nan_check_every = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--checkpoint-every") == 0 && a + 1 < argc) {
            checkpoint_every = atoi(argv[++a]);
        } else if (strcmp(argv[a], "--restart") == 0 && a + 1 < argc) {
            restart_file = argv[++a];
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

    /* Resolve tracker interval */
    if (tracker_every < 0)
        tracker_every = ah_enabled ? ah_every : 10;

    /* Auto-enable tracker for multi-BH simulations (N>=2) */
    if (n_bh >= 2 && !tracker_enabled)
        tracker_enabled = 1;

    setbuf(stdout, NULL);
    backend_init();

    printf("Lattice — 3D Numerical Relativity\n");

    if (restart_file) {
        /* === Restart path === */
        mesh_t *m = NULL;
        int start_step = 0;

        if (checkpoint_read(restart_file, &m, &p, &start_step) != 0) {
            fprintf(stderr, "Error: failed to read checkpoint %s\n",
                    restart_file);
            return 1;
        }

        /* Recompute derived quantities from restored params */
        p.dx = m->dx_base;
        p.dt = p.CFL * p.dx;

        printf("  Resuming from step %d (t=%.4f)\n", start_step, p.time);
        printf("  AMR mode: N_block=%d, leaves=%d, max_level=%d\n",
               m->N_block, mesh_num_leaves(m), m->max_level);
        printf("  L = %.1f, dx = %.6f, dt = %.6f\n", p.L, p.dx, p.dt);

        double ham0 = mesh_constraint_l2(m);
        printf("  Ham L2 at restart = %.6e\n", ham0);

        /* Select RHS function */
        rk4_rhs_func_t rhs_func = p.em_enabled
            ? ccz4_maxwell_rhs_point : ccz4_rhs_point;

        /* Evolution loop (continues from start_step) */
        for (int step = start_step + 1; step <= p.num_steps; step++) {
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

            /* Checkpoint */
            if (checkpoint_every > 0 && step % checkpoint_every == 0) {
                char ckpt_path[256];
                snprintf(ckpt_path, sizeof(ckpt_path),
                         "build/checkpoint_%06d.lat", step);
                checkpoint_write(m, &p, step, ckpt_path);
            }
        }

        mesh_free(m);
    } else if (p.amr.enabled) {
        /* === AMR path (fresh start) === */
        /* Resolve solver_levels: default to max_level if not explicitly set */
        if (p.amr.solver_levels < 0)
            p.amr.solver_levels = p.amr.max_level;

        mesh_t *m = mesh_create_ex(p.amr.N_block, p.L, p.rk_method,
                                    p.em_enabled ? NUM_FIELDS : NUM_CCZ4_FIELDS);
        p.dx = m->dx_base;
        p.dt = p.CFL * p.dx;

        printf("  AMR mode: N_block=%d, max_level=%d\n",
               p.amr.N_block, p.amr.max_level);
        printf("  L = %.1f, dx = %.6f, dt = %.6f\n", p.L, p.dx, p.dt);
        printf("  steps = %d, sigma = %.2f, CFL = %.2f, rk = %s\n",
               p.num_steps, p.sigma, p.CFL,
               p.rk_method == RK_CK45 ? "ck45" : "classic");
        printf("  chi_refine = %.4f, chi_coarsen = %.4f, regrid_every = %d\n",
               p.amr.chi_refine, p.amr.chi_coarsen, p.amr.regrid_every);
        printf("  blocks = %d (leaves = %d)\n",
               m->num_blocks, mesh_num_leaves(m));

        /* Set initial data: solve directly on the evolution mesh.
         * The constraint solver operates on the mesh blocks in-place,
         * then converts solver data → CCZ4 fields. Zero interpolation error.
         * Ref: Tomida & Stone 2023 (Athena++ MG on evolution mesh) */
        if (n_bh > 0) {
            printf("  Initial data: Bowen-York, %d puncture(s), "
                   "solver_levels=%d\n", n_bh, p.amr.solver_levels);
            set_bowen_york_mesh_ex(m, n_bh, bhs, p.amr.solver_levels,
                                    p.amr.refine_c, p.amr.refine_beta);
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

        /* N-body BH tracker setup (AMR path) */
        bh_tracker_t *tracker = NULL;
        FILE *tracker_csv = NULL;
        if (tracker_enabled && n_bh > 0) {
            tracker = bh_tracker_alloc(n_bh, bhs, ah_n_theta, ah_n_phi);
            tracker_csv = fopen("build/nbody_diagnostics.csv", "w");
            if (tracker_csv) bh_tracker_write_csv_header(tracker, tracker_csv);
            printf("  BH tracker: %d BHs, every %d steps\n",
                   n_bh, tracker_every);
        }

        /* Psi4 extraction setup (AMR path) */
        psi4_workspace_t *psi4_ws = NULL;
        if (psi4_enabled) {
            double psi4_center[3] = {0, 0, 0};
            psi4_ws = psi4_alloc(psi4_n_theta, psi4_n_phi, psi4_l_max,
                                  psi4_radius, psi4_center);
            printf("  Psi4: enabled, every %d steps, r=%.1f, l_max=%d\n",
                   psi4_every, psi4_radius, psi4_l_max);
        }

#ifdef LATTICE_HDF5
        /* CCE worldtube setup (AMR path) */
        cce_ws_t *cce_ws = NULL;
        if (cce_enabled) {
            double cce_center[3] = {0, 0, 0};
            char cce_fname[64];
            snprintf(cce_fname, sizeof(cce_fname),
                     "build/CceR%04d.h5", (int)cce_radius);
            cce_ws = cce_alloc(cce_lmax, cce_radius, cce_center, cce_fname);
            printf("  CCE: enabled, every %d steps, r=%.0f, l_max=%d → %s\n",
                   cce_every, cce_radius, cce_lmax, cce_fname);
        }
#endif

        /* Evolution loop.
         * p.time tracks the simulation time for subcycling's temporal
         * interpolation (Berger-Oliger). dt is the coarsest-level step.
         *
         * GPU path: after rk4_step_mesh, data stays on device. We only
         * sync to host when CPU-only work is needed (regrid, checkpoint,
         * output, AH finder). GPU diagnostic kernels (constraints, Psi4,
         * BH separation) run directly on device packs. */
        p.time = 0.0;
        for (int step = 1; step <= p.num_steps; step++) {
            rk4_step_mesh(m, &p, rhs_func, p.dt);
            p.time += p.dt;

            /* Determine what work is needed this step */
            int need_regrid = (p.amr.regrid_every > 0 &&
                               step % p.amr.regrid_every == 0);
            int need_output = (p.output_every > 0 &&
                               step % p.output_every == 0);
            int need_ham100 = (step % 100 == 0 || step == p.num_steps);
            int need_tracker = (tracker && tracker_every > 0 &&
                                step % tracker_every == 0);
            int need_ah = (ah_ws && ah_every > 0 &&
                           step % ah_every == 0);
            int need_psi4 = (psi4_ws && psi4_every > 0 &&
                             step % psi4_every == 0);
            int need_checkpoint = (checkpoint_every > 0 &&
                                   step % checkpoint_every == 0);
            int need_nan_check = (nan_check_every > 0 &&
                                  step % nan_check_every == 0);
#ifdef LATTICE_HDF5
            int need_cce = (cce_ws && cce_every > 0 &&
                            step % cce_every == 0);
#else
            int need_cce = 0;
#endif

            /* GPU diagnostic path: constraints, Psi4, and BH separation
             * run directly on device-resident level packs — no host sync.
             * Only CPU-only operations (regrid, output, checkpoint, AH
             * finder, BH tracker, CCE) force a gpu_sync_all_to_host. */
            int gpu_resident = backend_is_gpu() && m->max_level > 0
                               && m->level_packs[0];

            /* --- GPU-native diagnostics (no host sync needed) --- */
            if (need_ham100 && gpu_resident) {
                double ham = gpu_constraint_l2(m);
                printf("  step %5d  t=%.4f  Ham L2=%.6e  blocks=%d\n",
                       step, p.time, ham, mesh_num_leaves(m));
            }

            if (need_psi4 && gpu_resident) {
                gpu_psi4_extract(psi4_ws, m);
                psi4_write_modes(psi4_ws, p.time, "build/psi4_modes.csv");
                int mi_22 = 4 + 2 + 2 - 4; /* (2,2) mode index */
                double re22 = psi4_ws->mode_re[mi_22];
                double im22 = psi4_ws->mode_im[mi_22];
                printf("  Psi4 step %5d: r*Psi4(2,2) = %.6e + %.6ei\n",
                       step, re22, im22);
            }

            /* --- GPU-native NaN/Inf check (no sync needed) --- */
            if (need_nan_check && gpu_resident) {
                meshblock_pack_t *lpk = m->level_packs[0];
                backend_activate_pack(lpk);
                if (!backend_check_finite_packed(lpk)) {
                    fprintf(stderr,
                        "\n*** NaN/Inf detected at step %d (t=%.4f) ***\n",
                        step, p.time);
                    break;
                }
            }

            /* --- GPU-native BH tracker position update (no sync) --- */
            if (need_tracker && gpu_resident) {
                bh_tracker_update_positions_packed(tracker, m);
                bh_tracker_check_mergers(tracker, p.time);
            }

            /* --- CPU-only operations: sync to host first --- */
            int need_host_sync = need_regrid || need_output ||
                                 need_checkpoint || need_ah || need_cce;
            /* Tracker needs sync only for find_horizons (AH per BH) */
            if (need_tracker)
                need_host_sync = 1;  /* find_horizons is CPU-only */
            /* Also sync for diagnostics when NOT on GPU path */
            if (!gpu_resident)
                need_host_sync |= need_ham100 || need_psi4 || need_tracker;

            if (need_host_sync && backend_is_gpu())
                gpu_sync_all_to_host(m);

            if (need_regrid)
                mesh_regrid(m, &p.amr);

            if (need_output)
                output_mesh_1d_slice(m, step, p.time);

            /* CPU NaN/Inf check (non-GPU path) */
            if (need_nan_check && !gpu_resident) {
                if (!mesh_check_finite(m)) {
                    fprintf(stderr,
                        "\n*** NaN/Inf detected at step %d (t=%.4f) ***\n",
                        step, p.time);
                    break;
                }
            }

            /* CPU constraint check (non-GPU path only) */
            if (need_ham100 && !gpu_resident) {
                double ham = mesh_constraint_l2(m);
                printf("  step %5d  t=%.4f  Ham L2=%.6e  blocks=%d\n",
                       step, p.time, ham, mesh_num_leaves(m));
            }

            /* N-body BH tracker */
            if (need_tracker) {
                if (!gpu_resident)
                    bh_tracker_update_positions(tracker, m);
                /* GPU path already did position update + merger check above;
                 * but find_horizons needs host data (AH finder is CPU-only) */
                bh_tracker_find_horizons(tracker, m, ah_tol, ah_max_iter);
                if (!gpu_resident)
                    bh_tracker_check_mergers(tracker, p.time);
                double ham_tr = gpu_resident ? gpu_constraint_l2(m)
                                             : mesh_constraint_l2(m);
                double mom_tr = gpu_resident ? gpu_momentum_l2(m)
                                             : mesh_momentum_l2(m);
                bh_tracker_write_csv(tracker, tracker_csv, p.time,
                                      ham_tr, mom_tr, mesh_num_leaves(m));
            }

            /* AH finder (CPU-only: pseudo-time PDE solve) */
            if (need_ah) {
                int conv = ah_find_amr(ah_ws, m, ah_tol, ah_max_iter, 0);
                if (conv) {
                    ah_result_t ahr = ah_compute_diagnostics_amr(ah_ws, m);
                    printf("  AH step %5d: A=%.4f M_irr=%.4f |J|=%.4e r=%.4f\n",
                           step, ahr.area, ahr.mass_irr, ahr.spin_mag,
                           ahr.mean_radius);
                }
            }

            /* Psi4 extraction (CPU path when not GPU-resident) */
            if (need_psi4 && !gpu_resident) {
                psi4_extract(psi4_ws, m);
                psi4_write_modes(psi4_ws, p.time, "build/psi4_modes.csv");
                int mi_22 = 4 + 2 + 2 - 4;
                double re22 = psi4_ws->mode_re[mi_22];
                double im22 = psi4_ws->mode_im[mi_22];
                printf("  Psi4 step %5d: r*Psi4(2,2) = %.6e + %.6ei\n",
                       step, re22, im22);
            }

#ifdef LATTICE_HDF5
            /* CCE worldtube (CPU-only: interpolation + HDF5 I/O) */
            if (need_cce)
                cce_extract(cce_ws, m, p.time);
#endif

            /* Checkpoint (after diagnostics, before next step) */
            if (need_checkpoint) {
                char ckpt_path[256];
                snprintf(ckpt_path, sizeof(ckpt_path),
                         "build/checkpoint_%06d.lat", step);
                checkpoint_write(m, &p, step, ckpt_path);
            }
        }

        /* Tracker cleanup + merger log */
        if (tracker) {
            bh_tracker_write_mergers(tracker, "build/merger_events.log");
            bh_tracker_free(tracker);
        }
        if (tracker_csv) fclose(tracker_csv);

#ifdef LATTICE_HDF5
        cce_free(cce_ws);
#endif
        psi4_free(psi4_ws);
        ah_free(ah_ws);
        mesh_free(m);
    } else {
        /* === Single-grid path (via 1-block mesh) === */
        int n_fields = p.em_enabled ? NUM_FIELDS : NUM_CCZ4_FIELDS;
        mesh_t *m = mesh_create_ex(p.N, p.L, p.rk_method, n_fields);
        grid_t *g = m->blocks[0]->grid;

        /* mesh_create_ex may pad N — refresh derived quantities */
        p.N  = g->N;
        p.dx = g->dx;
        p.dt = p.CFL * p.dx;

        printf("  N = %d, L = %.1f, dx = %.6f, dt = %.6f\n",
               p.N, p.L, p.dx, p.dt);
        printf("  steps = %d, sigma = %.2f, CFL = %.2f, rk = %s\n",
               p.num_steps, p.sigma, p.CFL,
               p.rk_method == RK_CK45 ? "ck45" : "classic");
        printf("  Ntotal = %d (N=%d + 2*ghost=%d)\n",
               g->Ntotal, g->N, g->ghost);

        /* Set initial data */
        if (n_bh > 0) {
            printf("  Initial data: Bowen-York, %d puncture(s)\n", n_bh);
            set_bowen_york_mesh(m, n_bh, bhs, 0);
        } else {
            printf("  Initial data: flat spacetime\n");
            set_flat_spacetime(g);
        }

        /* Initial constraint */
        double ham0 = mesh_constraint_l2(m);
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

        /* N-body BH tracker setup (single-grid path) */
        bh_tracker_t *tracker = NULL;
        FILE *tracker_csv = NULL;
        if (tracker_enabled && n_bh > 0) {
            tracker = bh_tracker_alloc(n_bh, bhs, ah_n_theta, ah_n_phi);
            tracker_csv = fopen("build/nbody_diagnostics.csv", "w");
            if (tracker_csv) bh_tracker_write_csv_header(tracker, tracker_csv);
            printf("  BH tracker: %d BHs, every %d steps\n",
                   n_bh, tracker_every);
        }

        /* Psi4 extraction setup (single-grid path) */
        psi4_workspace_t *psi4_ws = NULL;
        if (psi4_enabled) {
            double psi4_center[3] = {0, 0, 0};
            psi4_ws = psi4_alloc(psi4_n_theta, psi4_n_phi, psi4_l_max,
                                  psi4_radius, psi4_center);
            printf("  Psi4: enabled, every %d steps, r=%.1f, l_max=%d\n",
                   psi4_every, psi4_radius, psi4_l_max);
        }

#ifdef LATTICE_HDF5
        /* CCE worldtube setup (single-grid path) */
        cce_ws_t *cce_ws = NULL;
        if (cce_enabled) {
            double cce_center[3] = {0, 0, 0};
            char cce_fname[64];
            snprintf(cce_fname, sizeof(cce_fname),
                     "build/CceR%04d.h5", (int)cce_radius);
            cce_ws = cce_alloc(cce_lmax, cce_radius, cce_center, cce_fname);
            printf("  CCE: enabled, every %d steps, r=%.0f, l_max=%d → %s\n",
                   cce_every, cce_radius, cce_lmax, cce_fname);
        }
#endif

        /* Time evolution */
        p.time = 0.0;
        for (int step = 1; step <= p.num_steps; step++) {
            rk4_step_mesh(m, &p, rhs_func, p.dt);
            p.time += p.dt;

            if (step % 100 == 0 || step == p.num_steps) {
                double ham = mesh_constraint_l2(m);
                printf("  step %5d  t = %.4f  Ham L2 = %.6e\n",
                       step, p.time, ham);
            }

            if (p.output_every > 0 && step % p.output_every == 0) {
                output_mesh_1d_slice(m, step, p.time);
            }

            /* N-body BH tracker (single-grid path) */
            if (tracker && tracker_every > 0 && step % tracker_every == 0) {
                bh_tracker_update_positions(tracker, m);
                bh_tracker_find_horizons(tracker, m, ah_tol, ah_max_iter);
                bh_tracker_check_mergers(tracker, p.time);
                double ham_tr = mesh_constraint_l2(m);
                double mom_tr = mesh_momentum_l2(m);
                bh_tracker_write_csv(tracker, tracker_csv, p.time,
                                      ham_tr, mom_tr, mesh_num_leaves(m));
            }

            /* AH finder */
            if (ah_ws && ah_every > 0 && step % ah_every == 0) {
                int conv = ah_find_amr(ah_ws, m, ah_tol, ah_max_iter, 0);
                if (conv) {
                    ah_result_t ahr = ah_compute_diagnostics_amr(ah_ws, m);
                    printf("  AH step %5d: A=%.4f M_irr=%.4f |J|=%.4e r=%.4f\n",
                           step, ahr.area, ahr.mass_irr, ahr.spin_mag,
                           ahr.mean_radius);
                }
            }

            /* Psi4 extraction (single-grid path) */
            if (psi4_ws && psi4_every > 0 && step % psi4_every == 0) {
                psi4_extract(psi4_ws, m);
                psi4_write_modes(psi4_ws, p.time, "build/psi4_modes.csv");
                int mi_22 = 4 + 2 + 2 - 4;
                double re22 = psi4_ws->mode_re[mi_22];
                double im22 = psi4_ws->mode_im[mi_22];
                printf("  Psi4 step %5d: r*Psi4(2,2) = %.6e + %.6ei\n",
                       step, re22, im22);
            }

#ifdef LATTICE_HDF5
            /* CCE worldtube extraction (single-grid path) */
            if (cce_ws && cce_every > 0 && step % cce_every == 0)
                cce_extract(cce_ws, m, p.time);
#endif

            /* Checkpoint */
            if (checkpoint_every > 0 && step % checkpoint_every == 0) {
                char ckpt_path[256];
                snprintf(ckpt_path, sizeof(ckpt_path),
                         "build/checkpoint_%06d.lat", step);
                checkpoint_write(m, &p, step, ckpt_path);
            }
        }

        /* Tracker cleanup + merger log */
        if (tracker) {
            bh_tracker_write_mergers(tracker, "build/merger_events.log");
            bh_tracker_free(tracker);
        }
        if (tracker_csv) fclose(tracker_csv);

#ifdef LATTICE_HDF5
        cce_free(cce_ws);
#endif
        psi4_free(psi4_ws);
        ah_free(ah_ws);
        mesh_free(m);
    }

    backend_cleanup();

    return 0;
}
