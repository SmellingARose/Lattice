/*
 * ====================================================================
 * Lattice — 3D Numerical Relativity
 * Binary Black Hole Inspiral: D10 Benchmark Validation
 * ====================================================================
 *
 * Evolves the canonical D10 equal-mass nonspinning quasi-circular binary
 * through inspiral, merger, and ringdown.  Compares remnant properties
 * against the Samurai cross-code consensus (5 independent NR codes).
 *
 * This test exercises every major subsystem simultaneously:
 *
 *   1. Bowen-York initial data with FAS multigrid constraint solver
 *   2. AMR mesh with chi-gradient refinement and periodic regridding
 *   3. CCZ4 evolution with constraint-preserving boundary conditions
 *   4. Classic RK4 time integration
 *   5. Kreiss-Oliger dissipation (6th order)
 *   6. Hamiltonian and momentum constraint monitoring
 *   7. Psi4 gravitational wave extraction (spin-weighted harmonics)
 *   8. Apparent horizon finder (hyperbolic flow method)
 *   9. Lapse profile and BH separation tracking
 *  10. 1D slice CSV output for visualization
 *
 * Physical setup (Bode et al. 2009, arXiv:0902.1127, Table I):
 *
 *   Puncture masses:    m1 = m2 = 0.48595  (constraint-satisfying QC)
 *   Separation:         d = 10 M  (z-axis, z = +/-5)
 *   Tangential momentum: P_y = +/-0.09543  (quasi-circular)
 *   E_ADM = 0.9895, J_ADM = 0.9530
 *
 * Grid configuration (matching BAM D10, gr-qc/0610128):
 *
 *   Domain:     [-768, 768]^3 M  (outer boundary at ~770M, BAM uses 773M)
 *   Root mesh:  1 block x 32^3 cells, dx_base = 48 M
 *   AMR:        max_level = 11  ->  dx_fine = M/43 near punctures
 *               (BAM medium: M/44.8, Samurai: arXiv:0901.2437)
 *   CFL:        0.25
 *   Integrator: classic RK4
 *   BCs:        constraint-preserving (BAM-style, arXiv:1212.2901)
 *
 * Gravitational wave extraction:
 *
 *   Psi4 on sphere at r = 90 M, decomposed into _{-2}Y_{lm} up to l = 4.
 *   The dominant (2,2) mode encodes the orbital frequency and amplitude.
 *   GW phase tracked via atan2(Im, Re) with 2pi unwrapping.
 *
 * Apparent horizons:
 *
 *   Individual BH AH finders (12x24) track each puncture during inspiral.
 *   After merger + 50M settling, a remnant AH finder (16x32) centered at
 *   the origin attempts to find the common horizon and extract M_chr, chi.
 *
 * Samurai consensus (arXiv:0901.2437, 5-code comparison):
 *
 *   M_final / M_ADM = 0.9516
 *   chi_final       = 0.6865
 *   E_radiated      = 0.048  (fraction of M_ADM)
 *
 * Pass criteria:
 *
 *   Tier 1 (hard, fail build):
 *     1. No NaN/Inf in any evolved field
 *     2. Hamiltonian constraint L2 < 0.1
 *     3. Momentum constraint L2 < 0.1
 *     4. GW radiation present: max |rPsi4_22| > 0.01
 *     5. GW amplitude sane: max |rPsi4_22| < 0.20
 *     6. Trumpet lapse: min alpha < 0.4
 *     7. Inspiral motion: final sep < initial sep
 *     8. Merger occurred: sep < 3M at some point
 *
 *   Tier 2 (advisory, logged but don't fail):
 *     9. Remnant mass: M_chr/M_ADM ~ 0.9516  (+/-0.05)
 *    10. Remnant spin: chi ~ 0.6865  (+/-0.10)
 *    11. Orbital dynamics: >= 0.5 orbits in GW phase
 *    12. Peak Psi4 timing: 50M < t_peak < 500M
 *
 * References:
 *   [1] arXiv:0902.1127  Bode et al. 2009 (D10 QC parameters)
 *   [2] arXiv:0901.2437  Samurai project (cross-code consensus)
 *   [3] gr-qc/0610128    Brugmann et al. 2008 (BAM calibration)
 *   [4] arXiv:1106.2254  Alic et al. 2012 (CCZ4 formulation)
 *   [5] arXiv:1212.2901  Hilditch et al. 2013 (constraint-preserving BCs)
 *   [6] arXiv:2505.15912 Rashti et al. 2025 (BHaHAHA AH finder)
 *   [7] B&S 8.3          Baumgarte & Shapiro (Weyl scalars, Psi4)
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/amr/mesh.h"
#include "../src/amr/refine.h"
#include "../src/amr/ghost_exchange.h"
#include "../src/amr/meshblock_pack.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/diagnostics/psi4.h"
#include "../src/diagnostics/ah_finder.h"
#include "../src/diagnostics/bh_tracker.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Forward declaration — defined in src/io/output.c, no header. */
extern void output_mesh_1d_slice(const mesh_t *m, int step, double time);

/* ====================================================================
 * Physical parameters
 * ====================================================================
 * D10 equal-mass non-spinning quasi-circular binary.
 * Constraint-satisfying QC values from Table I of arXiv:0902.1127.
 *
 * Ref: arXiv:0902.1127 (Bode et al. 2009)
 * ==================================================================== */
#define M_BARE     0.48595
#define D_SEP      10.0
#define P_Y        0.09543

/* Published ADM quantities (Bode et al. 2009, Table I) */
#define M_ADM      0.9895
#define J_ADM      0.9530

/* ====================================================================
 * Samurai consensus remnant values
 * ====================================================================
 * 5-code cross-comparison for D10 equal-mass nonspinning binary.
 * Ref: arXiv:0901.2437 (Hannam et al. 2009)
 * ==================================================================== */
#define M_FINAL_OVER_M   0.9516
#define CHI_FINAL         0.6865
#define E_RADIATED_FRAC   0.048

/* ====================================================================
 * Grid and evolution parameters
 * ==================================================================== */
#define L_DOMAIN    1536.0
#define N_BLOCK     32
#define MAX_LEVEL   11
#define CFL_FACTOR  0.25
#define T_FINAL     700.0

/* ====================================================================
 * Diagnostic schedule
 * ==================================================================== */
#define DIAG_EVERY    1
#define REGRID_EVERY  1
#define SLICE_EVERY   200
#define PSI4_EVERY    1
#define AH_EVERY      1

/* ====================================================================
 * Wave extraction parameters
 * ==================================================================== */
#define PSI4_RADIUS   90.0
#define PSI4_LMAX     4
#define PSI4_NTHETA   16
#define PSI4_NPHI     32

/* ====================================================================
 * Apparent horizon parameters (individual BHs)
 * ==================================================================== */
#define AH_NTHETA     12
#define AH_NPHI       24
#define AH_RGUESS     0.5
#define AH_TOL        1.0e-2
#define AH_MAXITER    200

/* ====================================================================
 * Remnant apparent horizon parameters (post-merger)
 * ==================================================================== */
#define AH_REMNANT_NTHETA  16
#define AH_REMNANT_NPHI    32
#define AH_REMNANT_RGUESS  1.5


/* ====================================================================
 * Mesh-level diagnostics
 * ==================================================================== */

/*
 * Find the global minimum of the lapse function across all leaf blocks.
 * The lapse minimum tracks the BH location — in the moving puncture
 * gauge, the lapse collapses to the "trumpet" value (~0.3) at the
 * puncture and recovers to ~1 far away.
 */
static double mesh_min_lapse(const mesh_t *m,
                              double *out_x, double *out_y, double *out_z)
{
    double min_val = 1.0e30;
    *out_x = *out_y = *out_z = 0.0;

    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        grid_t *g = b->grid;
        int lo = g->ghost, hi = g->ghost + g->N;
        for (int k = lo; k < hi; k++)
            for (int j = lo; j < hi; j++)
                for (int i = lo; i < hi; i++) {
                    double a = g->fields[FIELD_LAPSE][IDX(g, i, j, k)];
                    if (a < min_val) {
                        min_val = a;
                        *out_x = b->origin[0] + (i - g->ghost + 0.5) * g->dx;
                        *out_y = b->origin[1] + (j - g->ghost + 0.5) * g->dx;
                        *out_z = b->origin[2] + (k - g->ghost + 0.5) * g->dx;
                    }
                }
    }
    return min_val;
}

/*
 * Compute BH separation from tracker positions.
 * Returns 0.0 if fewer than 2 active BHs remain (merged).
 */
static double tracker_separation(const bh_tracker_t *tr)
{
    int a = -1, b = -1;
    for (int i = 0; i < tr->n_bh; i++) {
        if (tr->bh[i].status != BH_STATUS_ACTIVE) continue;
        if (a < 0) a = i;
        else if (b < 0) { b = i; break; }
    }
    if (a < 0 || b < 0) return 0.0;
    double dx = tr->bh[a].center[0] - tr->bh[b].center[0];
    double dy = tr->bh[a].center[1] - tr->bh[b].center[1];
    double dz = tr->bh[a].center[2] - tr->bh[b].center[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

/*
 * Scan all evolved fields for NaN/Inf.  Returns 1 if every interior
 * grid point in every leaf block is finite.  A single non-finite value
 * means the simulation has crashed.
 */
static int mesh_check_finite(const mesh_t *m)
{
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        grid_t *g = b->grid;
        int lo = g->ghost, hi = g->ghost + g->N;
        for (int f = 0; f < g->n_fields; f++)
            for (int k = lo; k < hi; k++)
                for (int j = lo; j < hi; j++)
                    for (int i = lo; i < hi; i++)
                        if (!isfinite(g->fields[f][IDX(g, i, j, k)]))
                            return 0;
    }
    return 1;
}

/*
 * Build a temporary meshblock_pack_t containing all leaf blocks for
 * GPU diagnostics.  Same logic as mesh_build_leaf_pack() in rk4.c.
 * Allocates data + rhs + scratch + accum (only data is used).
 */
static meshblock_pack_t *build_diag_pack(const mesh_t *m)
{
    int n_leaves = mesh_num_leaves(m);
    int *ids = malloc(n_leaves * sizeof(int));
    int idx = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (b && b->is_leaf)
            ids[idx++] = bid;
    }

    size_t npts = m->blocks[ids[0]]->grid->npoints;
    meshblock_pack_t *pack = meshblock_pack_create(
        n_leaves, npts, ids, -1, RK_CLASSIC, m->n_fields);
    meshblock_pack_load(pack, m->blocks);
    meshblock_pack_load_meta(pack, m->blocks);
    meshblock_pack_build_neighbors(pack, m->blocks);
    if (m->max_level > 0)
        meshblock_pack_load_coarse(pack, m->blocks);

    free(ids);
    return pack;
}


/* ====================================================================
 * Main
 * ==================================================================== */
int main(void)
{
    setbuf(stdout, NULL);

    /* -- Banner -------------------------------------------------------- */
    printf("\n");
    printf("==================================================================\n");
    printf("  Lattice -- Binary Black Hole Inspiral\n");
    printf("  D10 Benchmark Validation (Samurai cross-code comparison)\n");
    printf("==================================================================\n");
    printf("\n");
    printf("  Physical setup (arXiv:0902.1127, Table I):\n");
    printf("    m1 = m2 = %.5f  (E_ADM = %.4f, J_ADM = %.4f)\n",
           M_BARE, M_ADM, J_ADM);
    printf("    d = %.0f M  (z-axis)\n", D_SEP);
    printf("    P_y = +/-%.5f  (quasi-circular)\n", P_Y);
    printf("\n");
    printf("  Samurai consensus (arXiv:0901.2437):\n");
    printf("    M_final/M_ADM = %.4f, chi_final = %.4f\n",
           M_FINAL_OVER_M, CHI_FINAL);
    printf("\n");
    printf("  Grid:\n");
    printf("    Domain:     [-%.0f, %.0f]^3 M\n", L_DOMAIN / 2.0, L_DOMAIN / 2.0);
    printf("    Root mesh:  %d^3\n", N_BLOCK);
    printf("    AMR levels: %d  (dx_base = %.2f, dx_fine = %.4f M)\n",
           MAX_LEVEL, L_DOMAIN / N_BLOCK,
           L_DOMAIN / N_BLOCK / (1 << MAX_LEVEL));
    printf("    BCs:        constraint-preserving (arXiv:1212.2901)\n");
    printf("\n");
    printf("  Diagnostics:\n");
    printf("    Psi4:       r = %.0f M, l_max = %d, every %d steps\n",
           PSI4_RADIUS, PSI4_LMAX, PSI4_EVERY);
    printf("    AH finder:  %dx%d (individual), %dx%d (remnant)\n",
           AH_NTHETA, AH_NPHI, AH_REMNANT_NTHETA, AH_REMNANT_NPHI);
    printf("    Constraints: every %d steps\n", DIAG_EVERY);
    printf("    1D slices:  every %d steps\n", SLICE_EVERY);
    printf("\n");

    time_t wall_start = time(NULL);

    /* -- Initialize ---------------------------------------------------- */
    backend_init();
    mesh_t *m = mesh_create(N_BLOCK, L_DOMAIN, RK_CLASSIC);

    sim_params_t p = default_params();
    p.L         = L_DOMAIN;
    p.rk_method = RK_CLASSIC;
    p.CFL       = CFL_FACTOR;
    p.dx        = m->dx_base;
    p.dt        = p.CFL * p.dx;
    p.bc_type   = BC_CONSTRAINT_PRESERVING;

    /* ---- Match BAM parameters (arXiv:1212.2901, gr-qc/0610128) ---- */

    /* CCZ4 constraint damping: BAM Z4c uses kappa1=0.02, kappa2=0.
     * Ref: arXiv:1212.2901 (Hilditch et al. 2013) */
    p.ccz4.kappa1 = 0.02;

    /* Kreiss-Oliger dissipation: BAM uses sigma=0.1 on inner (near-BH) levels.
     * Ref: gr-qc/0610128 Table I */
    p.sigma = 0.1;

    /* Gauge: BAM "000" variant — full advection on lapse, shift, and B^i.
     * 1+log lapse (c=2) + Gamma-driver shift (F=3/4) are already default.
     * Ref: gr-qc/0610128 Sec. II.B, gr-qc/0605030 (van Meter et al.) */
    p.gauge.lapse_advec_coeff = 1.0;
    p.gauge.shift_advec_coeff = 1.0;
    p.gauge.eta = 2.0;                  /* BAM: eta = 2/M_ADM, M_ADM ~ 1 */
    p.gauge.position_dependent_eta = 0; /* BAM uses constant eta */

    /* Disable Lattice-specific noise reduction features (not in BAM) */
    p.noise.use_ssl  = 0;   /* no slow-start lapse */
    p.noise.use_cako = 0;   /* no chi-adjusted KO */
    p.noise.use_per_field_sigma = 0;  /* BAM uses zone-dep, not per-field */

    amr_params_t ap;
    ap.max_level   = MAX_LEVEL;
    ap.chi_refine  = 0.5;
    ap.chi_coarsen = 0.01;

    int num_steps = (int)(T_FINAL / p.dt + 0.5);

    printf("  dx = %.4f M, dt = %.6f M, CFL = %.2f\n", p.dx, p.dt, p.CFL);
    printf("  Steps: %d  (T_final = %.0f M)\n", num_steps, T_FINAL);
    printf("  Leaf blocks: %d (initial)\n\n", mesh_num_leaves(m));

    /* -- Initial data -------------------------------------------------- */
    printf("  [1/4] Solving Bowen-York initial data (FAS multigrid)...\n");
    fflush(stdout);
    time_t id_start = time(NULL);

    puncture_data_t bhs[2];
    memset(bhs, 0, sizeof(bhs));
    bhs[0].mass        = M_BARE;
    bhs[0].center[2]   = +D_SEP / 2.0;
    bhs[0].momentum[1] = +P_Y;
    bhs[1].mass        = M_BARE;
    bhs[1].center[2]   = -D_SEP / 2.0;
    bhs[1].momentum[1] = -P_Y;

    set_bowen_york_mesh(m, 2, bhs, MAX_LEVEL);

    printf("        Done (%.0f sec)\n\n", difftime(time(NULL), id_start));

    /* -- Post-solver mesh state ---------------------------------------- */
    printf("  [2/4] Mesh after initial data...\n");
    p.dx = m->dx_base;
    p.dt = p.CFL * p.dx;
    num_steps = (int)(T_FINAL / p.dt + 0.5);
    printf("        Leaves: %d, max_level = %d\n\n",
           mesh_num_leaves(m), m->max_level);

    /* -- Allocate diagnostic workspaces -------------------------------- */
    printf("  [3/4] Allocating diagnostics...\n");

    /* Psi4 extraction sphere */
    double psi4_center[3] = {0, 0, 0};
    psi4_workspace_t *psi4_ws = psi4_alloc(PSI4_NTHETA, PSI4_NPHI,
                                            PSI4_LMAX, PSI4_RADIUS,
                                            psi4_center);
    printf("        Psi4: %d modes, r = %.0f M\n",
           psi4_ws->n_modes, PSI4_RADIUS);

    /* N-body BH tracker: position tracking + per-BH AH finding */
    int n_bh = 2;
    bh_tracker_t *tracker = bh_tracker_alloc(n_bh, bhs, AH_NTHETA, AH_NPHI);
    bh_tracker_update_positions(tracker, m);
    printf("        BH tracker: %d BHs (%dx%d AH per BH)\n",
           n_bh, AH_NTHETA, AH_NPHI);

    /* Remnant AH finder -- post-merger, centered at origin */
    ah_workspace_t *ah_remnant = ah_alloc(AH_REMNANT_NTHETA, AH_REMNANT_NPHI,
                                           psi4_center, AH_REMNANT_RGUESS);
    printf("        AH finder: 1 remnant (%dx%d, r_guess = %.1f M)\n\n",
           AH_REMNANT_NTHETA, AH_REMNANT_NPHI, AH_REMNANT_RGUESS);

    /* -- Initial diagnostics ------------------------------------------- */
    double lx, ly, lz;
    double ml      = mesh_min_lapse(m, &lx, &ly, &lz);
    double sep0    = tracker_separation(tracker);
    double ham     = mesh_constraint_l2(m);
    double mom     = mesh_momentum_l2(m);

    printf("  [4/4] Evolution: %d steps, T = %.0f M\n\n", num_steps, T_FINAL);

    /* -- Evolution log header ------------------------------------------ */
    printf("  step    t/M    alpha_min  sep/M  leaves  Ham_L2      Mom_L2      "
           "|rPsi4_22|  AH1_M_irr  AH2_M_irr  wall(s)\n");
    printf("  --------------------------------------------------------------------------"
           "----------------------------------------------\n");

    printf("  %5d  %6.1f    %6.4f  %6.2f  %5d   %10.3e  %10.3e"
           "          -          -          -     0.0\n",
           0, 0.0, ml, sep0, mesh_num_leaves(m), ham, mom);
    fflush(stdout);

    /* -- Tracking ------------------------------------------------------ */
    double ham_peak     = ham;
    double mom_peak     = mom;
    double psi4_22_max  = 0.0;
    double psi4_22_peak_time = 0.0;
    double ml_min       = ml;
    double ah1_mass_last = -1.0, ah2_mass_last = -1.0;
    int    ah1_found = 0, ah2_found = 0;
    int    crashed = 0;

    /* Phase tracking (GW cycle counting via 2pi unwrapping) */
    double prev_phase22  = 0.0;
    double cumul_phase22 = 0.0;
    int    phase_initialized = 0;

    /* Merger tracking */
    int    merger_detected = 0;
    double merger_time     = 0.0;
    double min_sep         = sep0;
    double prev_sep        = sep0;

    /* Remnant AH results */
    double remnant_m_chr = -1.0;
    double remnant_chi   = -1.0;
    double remnant_m_irr = -1.0;

    /* -- Diagnostics CSV file ------------------------------------------ */
    FILE *diag_fp = fopen("build/inspiral_diagnostics.csv", "w");
    if (diag_fp) {
        fprintf(diag_fp, "time,ham_l2,mom_l2,alpha_min,separation,leaves");
        for (int i = 0; i < n_bh; i++)
            fprintf(diag_fp, ",bh%d_x,bh%d_y,bh%d_z,bh%d_mass,bh%d_spin,bh%d_lapse",
                    i, i, i, i, i, i);
        fprintf(diag_fp, ",psi4_22_re,psi4_22_im,psi4_22_amp,"
                "psi4_22_phase,gw_cycles,merger_flag,"
                "remnant_m_irr,remnant_m_chr,remnant_chi\n");
        /* Initial row */
        fprintf(diag_fp, "%.6f,%.6e,%.6e,%.6f,%.6f,%d",
                0.0, ham, mom, ml, sep0, mesh_num_leaves(m));
        for (int i = 0; i < n_bh; i++)
            fprintf(diag_fp, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    tracker->bh[i].center[0], tracker->bh[i].center[1],
                    tracker->bh[i].center[2], 0.0, 0.0, tracker->bh[i].lapse_min);
        fprintf(diag_fp, ",0,0,0,0,0,0,-1,-1,-1\n");
        fflush(diag_fp);
    }

    /* -- Evolution loop ------------------------------------------------ */
    p.time = 0.0;

    struct timespec ts_start, ts_end;
    for (int step = 1; step <= num_steps; step++) {
        /* -- Advance one time step ------------------------------------- */
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        double step_wall = (ts_end.tv_sec - ts_start.tv_sec)
                         + 1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec);
        p.time += p.dt;

        /* -- Periodic regridding --------------------------------------- */
        if (step % REGRID_EVERY == 0)
            mesh_regrid(m, &ap);

        /* -- Diagnostics ----------------------------------------------- */
        if (step % DIAG_EVERY == 0 || step == num_steps) {
            /* BH tracker: positions + AH + mergers (works on host mesh) */
            bh_tracker_update_positions(tracker, m);
            double sep = tracker_separation(tracker);

            if (backend_is_gpu()) {
                meshblock_pack_t *dp = build_diag_pack(m);
                backend_map_pack_diag(dp);
                backend_ghost_exchange_packed(dp);

                if (!backend_check_finite_packed(dp)) {
                    backend_unmap_pack_diag(dp);
                    meshblock_pack_free(dp);
                    printf("\n  *** CRASH: NaN/Inf detected at step %d "
                           "(t = %.2f M) ***\n\n", step, p.time);
                    crashed = 1;
                    break;
                }

                ham = backend_constraint_l2_packed(dp);
                mom = backend_momentum_l2_packed(dp);
                ml  = backend_min_lapse_packed(dp, &lx, &ly, &lz);

                if (step % PSI4_EVERY == 0)
                    backend_psi4_extract_packed(dp, psi4_ws, m);

                backend_unmap_pack_diag(dp);
                meshblock_pack_free(dp);
            } else {
                if (!mesh_check_finite(m)) {
                    printf("\n  *** CRASH: NaN/Inf detected at step %d "
                           "(t = %.2f M) ***\n\n", step, p.time);
                    crashed = 1;
                    break;
                }

                ham = mesh_constraint_l2(m);
                mom = mesh_momentum_l2(m);
                ml  = mesh_min_lapse(m, &lx, &ly, &lz);

                if (step % PSI4_EVERY == 0)
                    psi4_extract(psi4_ws, m);
            }

            if (ham > ham_peak) ham_peak = ham;
            if (mom > mom_peak) mom_peak = mom;
            if (ml < ml_min) ml_min = ml;

            /* Psi4 mode analysis + phase tracking */
            double psi4_22_amp = 0.0;
            double current_phase = cumul_phase22;
            double current_gw_cycles = fabs(cumul_phase22) / (2.0 * M_PI);

            if (step % PSI4_EVERY == 0) {
                psi4_write_modes(psi4_ws, p.time,
                                 "build/inspiral_psi4.csv");

                /* (l=2, m=2) mode index = l*l + l + m - 4 = 4 */
                int mi22 = 4;
                if (mi22 < psi4_ws->n_modes) {
                    double re = psi4_ws->mode_re[mi22];
                    double im = psi4_ws->mode_im[mi22];
                    psi4_22_amp = sqrt(re * re + im * im);

                    /* Phase tracking with 2pi unwrapping */
                    if (psi4_22_amp > 1e-10) {
                        double raw_phase = atan2(im, re);
                        if (phase_initialized) {
                            double delta = raw_phase - prev_phase22;
                            if (delta > M_PI) delta -= 2.0 * M_PI;
                            if (delta < -M_PI) delta += 2.0 * M_PI;
                            cumul_phase22 += delta;
                        }
                        prev_phase22 = raw_phase;
                        phase_initialized = 1;
                    }

                    current_phase = cumul_phase22;
                    current_gw_cycles = fabs(cumul_phase22) / (2.0 * M_PI);

                    if (psi4_22_amp > psi4_22_max) {
                        psi4_22_max = psi4_22_amp;
                        psi4_22_peak_time = p.time;
                    }
                }
            }

            /* Merger detection via tracker */
            if (sep > 0.0 && sep < min_sep) min_sep = sep;
            bh_tracker_check_mergers(tracker, p.time);
            if (!merger_detected && tracker->n_mergers > 0) {
                merger_detected = 1;
                merger_time = tracker->mergers[0].time;
            }
            /* Also detect via separation threshold (backup) */
            if (!merger_detected) {
                if ((sep > 0.0 && sep < 3.0) ||
                    (sep == 0.0 && prev_sep > 0.0 && step > 10)) {
                    merger_detected = 1;
                    merger_time = p.time;
                }
            }
            prev_sep = sep;

            /* Apparent horizons via tracker (per active BH) */
            double ah1_m = -1.0, ah2_m = -1.0;
            double ah1_spin = -1.0, ah2_spin = -1.0;
            if (step % AH_EVERY == 0) {
                bh_tracker_find_horizons(tracker, m, AH_TOL, AH_MAXITER);

                if (tracker->bh[0].mass_irr > 0) {
                    ah1_m = tracker->bh[0].mass_irr;
                    ah1_spin = tracker->bh[0].chi_spin;
                    ah1_mass_last = ah1_m;
                    ah1_found = 1;
                }
                if (n_bh > 1 && tracker->bh[1].mass_irr > 0) {
                    ah2_m = tracker->bh[1].mass_irr;
                    ah2_spin = tracker->bh[1].chi_spin;
                    ah2_mass_last = ah2_m;
                    ah2_found = 1;
                }

                /* Remnant AH search (after merger + 50M settling) */
                if (merger_detected && p.time >= merger_time + 50.0) {
                    ah_remnant->center[0] = 0.0;
                    ah_remnant->center[1] = 0.0;
                    ah_remnant->center[2] = 0.0;
                    if (ah_find_amr(ah_remnant, m, AH_TOL, AH_MAXITER, 0)) {
                        ah_result_t rr = ah_compute_diagnostics_amr(
                            ah_remnant, m);
                        if (rr.converged) {
                            remnant_m_chr = rr.mass_christodoulou;
                            remnant_chi   = rr.chi_spin;
                            remnant_m_irr = rr.mass_irr;
                        }
                    }
                }
            }

            /* Log line */
            printf("  %5d  %6.1f    %6.4f  %6.2f  %5d   %10.3e  %10.3e  %10.3e",
                   step, p.time, ml, sep, mesh_num_leaves(m), ham, mom,
                   psi4_22_amp);
            if (ah1_m > 0) printf("  %10.4f", ah1_m);
            else           printf("          -");
            if (ah2_m > 0) printf("  %10.4f", ah2_m);
            else           printf("          -");
            printf("  %7.1f\n", step_wall);
            fflush(stdout);

            /* Write diagnostics CSV row */
            if (diag_fp) {
                double p22_re = 0.0, p22_im = 0.0;
                int mi22_csv = 4;
                if (mi22_csv < psi4_ws->n_modes) {
                    p22_re = psi4_ws->mode_re[mi22_csv];
                    p22_im = psi4_ws->mode_im[mi22_csv];
                }
                fprintf(diag_fp,
                    "%.6f,%.6e,%.6e,%.6f,%.6f,%d",
                    p.time, ham, mom, ml, sep, mesh_num_leaves(m));
                /* Per-BH columns (scales with N) */
                for (int i = 0; i < n_bh; i++) {
                    const bh_state_t *bh = &tracker->bh[i];
                    if (bh->status == BH_STATUS_MERGED)
                        fprintf(diag_fp, ",nan,nan,nan,nan,nan,nan");
                    else
                        fprintf(diag_fp, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                                bh->center[0], bh->center[1], bh->center[2],
                                bh->mass_chr, bh->chi_spin, bh->lapse_min);
                }
                fprintf(diag_fp,
                    ",%.6e,%.6e,%.6e,"
                    "%.6f,%.6f,%d,%.6f,%.6f,%.6f\n",
                    p22_re, p22_im, psi4_22_amp,
                    current_phase, current_gw_cycles, merger_detected,
                    remnant_m_irr, remnant_m_chr, remnant_chi);
                fflush(diag_fp);
            }
        }

        /* -- 1D slice output ------------------------------------------- */
        if (step % SLICE_EVERY == 0)
            output_mesh_1d_slice(m, step, p.time);
    }

    /* -- Final state --------------------------------------------------- */
    bh_tracker_update_positions(tracker, m);
    double final_sep = tracker_separation(tracker);
    if (!crashed) {
        if (backend_is_gpu()) {
            meshblock_pack_t *dp = build_diag_pack(m);
            backend_map_pack_diag(dp);
            backend_ghost_exchange_packed(dp);
            backend_psi4_extract_packed(dp, psi4_ws, m);
            backend_unmap_pack_diag(dp);
            meshblock_pack_free(dp);
        } else {
            psi4_extract(psi4_ws, m);
        }
    }
    double wall_sec  = difftime(time(NULL), wall_start);

    /* Update min_sep and merger from final state */
    if (final_sep > 0.0 && final_sep < min_sep) min_sep = final_sep;
    if (!merger_detected && final_sep == 0.0 && prev_sep > 0.0)
        merger_detected = 1;

    /* Final Psi4 mode check */
    if (!crashed) {
        int mi22 = 4;
        if (mi22 < psi4_ws->n_modes) {
            double re = psi4_ws->mode_re[mi22];
            double im = psi4_ws->mode_im[mi22];
            double amp = sqrt(re * re + im * im);
            if (amp > psi4_22_max) {
                psi4_22_max = amp;
                psi4_22_peak_time = p.time;
            }
        }
    }

    /* -- Results ------------------------------------------------------- */
    printf("\n");
    printf("==================================================================\n");
    printf("  Results\n");
    printf("==================================================================\n");

    /* ---- Tier 1: Hard tests (fail the build) ---- */
    printf("\n  Tier 1 -- Hard tests (must pass):\n\n");

    int t1 = !crashed && mesh_check_finite(m);
    int t2 = ham_peak < 0.1;
    int t3 = mom_peak < 0.1;
    int t4 = psi4_22_max > 0.01;
    int t5 = psi4_22_max < 0.20;
    int t6 = ml_min < 0.4;
    int t7 = final_sep < sep0 * 0.99;
    int t8 = merger_detected;

    printf("  1. Stability (no NaN/Inf):        %s\n",
           t1 ? "PASS" : "FAIL");
    printf("  2. Ham L2 bounded:                %s"
           "  (peak = %.3e, limit 0.1)\n",
           t2 ? "PASS" : "FAIL", ham_peak);
    printf("  3. Mom L2 bounded:                %s"
           "  (peak = %.3e, limit 0.1)\n",
           t3 ? "PASS" : "FAIL", mom_peak);
    printf("  4. GW radiation present:          %s"
           "  (max |rPsi4_22| = %.4f, need > 0.01)\n",
           t4 ? "PASS" : "FAIL", psi4_22_max);
    printf("  5. GW amplitude sane:             %s"
           "  (max |rPsi4_22| = %.4f, need < 0.20)\n",
           t5 ? "PASS" : "FAIL", psi4_22_max);
    printf("  6. Trumpet lapse formed:          %s"
           "  (min alpha = %.4f, need < 0.4)\n",
           t6 ? "PASS" : "FAIL", ml_min);
    printf("  7. Inspiral motion:               %s"
           "  (sep: %.2f -> %.2f M)\n",
           t7 ? "PASS" : "FAIL", sep0, final_sep);
    printf("  8. Merger occurred:               %s"
           "  (min sep = %.2f M, need < 3.0)\n",
           t8 ? "PASS" : "FAIL", min_sep);

    int tier1_passed = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;
    int all_hard_pass = (tier1_passed == 8);

    /* ---- Tier 2: Advisory checks (logged, don't fail build) ---- */
    printf("\n  Tier 2 -- Advisory checks (informational):\n\n");

    int a1 = 0, a2 = 0, a3 = 0, a4 = 0;

    if (remnant_m_chr > 0) {
        double mf_over_m = remnant_m_chr / M_ADM;
        a1 = fabs(mf_over_m - M_FINAL_OVER_M) < 0.05;
        printf("  9. Remnant mass:    M_chr/M_ADM = %.4f"
               "  (ref %.4f +/- 0.05)  %s\n",
               mf_over_m, M_FINAL_OVER_M, a1 ? "MATCH" : "MISMATCH");
    } else {
        printf("  9. Remnant mass:    AH not found\n");
    }

    if (remnant_chi >= 0) {
        a2 = fabs(remnant_chi - CHI_FINAL) < 0.10;
        printf(" 10. Remnant spin:    chi = %.4f"
               "  (ref %.4f +/- 0.10)  %s\n",
               remnant_chi, CHI_FINAL, a2 ? "MATCH" : "MISMATCH");
    } else {
        printf(" 10. Remnant spin:    AH not found\n");
    }

    double n_orbits = fabs(cumul_phase22) / (2.0 * M_PI) / 2.0;
    double gw_cycles = fabs(cumul_phase22) / (2.0 * M_PI);
    a3 = n_orbits >= 0.5;
    printf(" 11. Orbital dynamics: %.1f orbits (%.1f GW cycles)  %s\n",
           n_orbits, gw_cycles, a3 ? "OK" : "LOW");

    a4 = psi4_22_peak_time > 50.0 && psi4_22_peak_time < 500.0;
    printf(" 12. Peak Psi4 time:  t = %.1f M  %s\n",
           psi4_22_peak_time,
           a4 ? "(50-500M, OK)" : "(outside expected range)");

    int tier2_passed = a1 + a2 + a3 + a4;

    /* ---- Remnant properties summary ---- */
    if (remnant_m_chr > 0) {
        printf("\n  Remnant properties:\n");
        printf("    M_irr = %.4f, M_chr = %.4f, chi = %.4f\n",
               remnant_m_irr, remnant_m_chr, remnant_chi);
        printf("    Samurai: M_f/M = %.4f, chi = %.4f\n",
               M_FINAL_OVER_M, CHI_FINAL);
    }

    /* ---- GW summary ---- */
    printf("\n  Gravitational waves:\n");
    printf("    Max |rPsi4_22| = %.4f at t = %.1f M\n",
           psi4_22_max, psi4_22_peak_time);
    printf("    GW cycles = %.1f, est. orbits = %.1f\n",
           gw_cycles, n_orbits);

    if (merger_detected)
        printf("\n  Merger at t = %.1f M (sep < 3M)\n", merger_time);

    /* ---- Apparent horizons ---- */
    printf("\n  Apparent horizons (individual):\n");
    if (ah1_found)
        printf("    BH 1: M_irr = %.4f  (last found)\n", ah1_mass_last);
    else
        printf("    BH 1: not found\n");
    if (ah2_found)
        printf("    BH 2: M_irr = %.4f  (last found)\n", ah2_mass_last);
    else
        printf("    BH 2: not found\n");

    /* ---- Performance ---- */
    printf("\n  Performance:\n");
    printf("    Wall time:    %.0f sec (%.1f min)\n", wall_sec, wall_sec / 60.0);
    printf("    Final leaves: %d, max_level: %d\n",
           mesh_num_leaves(m), m->max_level);
    if (num_steps > 0)
        printf("    Avg sec/step: %.2f\n", wall_sec / num_steps);

    printf("\n  Output files:\n");
    printf("    build/inspiral_diagnostics.csv -- time series\n");
    printf("    build/inspiral_psi4.csv        -- Psi4 mode coefficients\n");
    printf("    build/slice_*.csv              -- 1D profiles\n");

    printf("\n  ==================================================\n");
    printf("  Tier 1: %d/8 hard tests %s\n",
           tier1_passed, all_hard_pass ? "PASSED" : "FAILED");
    printf("  Tier 2: %d/4 advisory checks\n", tier2_passed);
    printf("  %s\n", all_hard_pass ? "ALL PASSED" : "FAILED");
    printf("  ==================================================\n\n");

    /* -- Cleanup ------------------------------------------------------- */
    if (diag_fp) fclose(diag_fp);
    bh_tracker_write_mergers(tracker, "build/inspiral_mergers.log");
    bh_tracker_free(tracker);
    psi4_free(psi4_ws);
    ah_free(ah_remnant);
    mesh_free(m);
    backend_cleanup();

    return all_hard_pass ? 0 : 1;
}
