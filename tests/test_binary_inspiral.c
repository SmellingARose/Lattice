/*
 * ====================================================================
 * Lattice — 3D Numerical Relativity
 * Binary Black Hole Inspiral: Full System Validation
 * ====================================================================
 *
 * This test evolves an equal-mass, non-spinning, quasi-circular binary
 * black hole system through inspiral, merger, and ringdown.  It exercises
 * every major subsystem simultaneously:
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
 * Physical setup (Brugmann et al. 2008, arXiv:0709.0838, Table I):
 *
 *   Puncture masses:    m1 = m2 = 0.4824  (M_ADM ≈ 1.0)
 *   Separation:         d = 10 M  (z-axis, z = ±5)
 *   Tangential momentum: P_y = ±0.0939  (3PN quasi-circular)
 *   Orbital period:     T_orbit ≈ 200 M
 *
 * Grid configuration:
 *
 *   Domain:     [-32, 32]^3 M
 *   Root mesh:  1 block x 32^3 cells, dx_base = 2.0 M
 *   AMR:        max_level = 4  →  dx_fine = 0.125 M near punctures
 *   CFL:        0.25
 *   Integrator: classic RK4
 *   BCs:        constraint-preserving (BAM-style, arXiv:1212.2901)
 *
 * Gravitational wave extraction:
 *
 *   Psi4 on sphere at r = 20 M, decomposed into _{-2}Y_{lm} up to l = 4.
 *   The dominant (2,2) mode encodes the orbital frequency and amplitude.
 *   For a quasi-circular orbit, |r Psi4_{22}| ≈ 0.01 at r = 20 M.
 *
 * Apparent horizons:
 *
 *   Hyperbolic flow finder attempts to locate each BH's AH every 20
 *   steps, extracting irreducible mass M_irr and dimensionless spin chi.
 *   With AMR refinement (dx_fine = 0.125 M), the AH radius (~0.12 M)
 *   spans ~1 cell — marginal but detectable.
 *
 * Pass criteria:
 *
 *   1. No NaN/Inf in any evolved field (stability)
 *   2. Hamiltonian constraint L2 < 1.0 at all times (accuracy)
 *   3. Psi4 (2,2) mode amplitude > 1e-6 (radiation present)
 *   4. Minimum lapse < 0.5 (trumpet slice forming near puncture)
 *   5. Final separation < initial separation (inspiral, not escape)
 *
 * References:
 *   [1] arXiv:0709.0838  Brugmann et al. 2008 (BAM calibration binary)
 *   [2] arXiv:1106.2254  Alic et al. 2012 (CCZ4 formulation)
 *   [3] gr-qc/0511048    Campanelli et al. 2006 (moving punctures)
 *   [4] arXiv:1212.2901  Hilditch et al. 2013 (constraint-preserving BCs)
 *   [5] arXiv:2505.15912 Rashti et al. 2025 (BHaHAHA AH finder)
 *   [6] B&S §8.3         Baumgarte & Shapiro (Weyl scalars, Psi4)
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/amr/mesh.h"
#include "../src/amr/refine.h"
#include "../src/amr/ghost_exchange.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/diagnostics/psi4.h"
#include "../src/diagnostics/ah_finder.h"
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
 * Standard equal-mass non-spinning quasi-circular binary.
 * Bare masses chosen so that M_ADM ≈ 1.0 after binding energy.
 * Tangential momentum from 3PN approximation for quasi-circular orbit.
 *
 * Ref: arXiv:0709.0838, Table I
 * ==================================================================== */
#define M_BARE     0.4824
#define D_SEP      10.0
#define P_Y        0.0939

/* ====================================================================
 * Grid and evolution parameters
 * ==================================================================== */
#define L_DOMAIN    64.0
#define N_BLOCK     32
#define MAX_LEVEL   4
#define CFL_FACTOR  0.25
#define T_FINAL     700.0

/* ====================================================================
 * Diagnostic schedule
 * ==================================================================== */
#define DIAG_EVERY    1
#define REGRID_EVERY  50
#define SLICE_EVERY   200
#define PSI4_EVERY    1
#define AH_EVERY      1

/* ====================================================================
 * Wave extraction parameters
 * ==================================================================== */
#define PSI4_RADIUS   20.0
#define PSI4_LMAX     4
#define PSI4_NTHETA   16
#define PSI4_NPHI     32

/* ====================================================================
 * Apparent horizon parameters
 * ==================================================================== */
#define AH_NTHETA     12
#define AH_NPHI       24
#define AH_RGUESS     0.5
#define AH_TOL        1.0e-2
#define AH_MAXITER    200


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
 * Estimate BH coordinate separation from the two deepest lapse minima.
 *
 * Algorithm: scan all leaf blocks for the global lapse minimum (BH #1),
 * then find the deepest minimum at least 2M away (BH #2).  This is
 * more robust than axis-only scans for inspiraling binaries where the
 * orbital plane rotates.
 *
 * Returns 0.0 if only one minimum found (merged or pre-formation).
 */
static double mesh_bh_separation(const mesh_t *m,
                                  double pos1[3], double pos2[3])
{
    /* Pass 1: find global lapse minimum (BH #1) */
    double best1 = 1.0e30;
    pos1[0] = pos1[1] = pos1[2] = 0.0;

    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        grid_t *g = b->grid;
        int lo = g->ghost, hi = g->ghost + g->N;
        for (int k = lo; k < hi; k++)
            for (int j = lo; j < hi; j++)
                for (int i = lo; i < hi; i++) {
                    double a = g->fields[FIELD_LAPSE][IDX(g, i, j, k)];
                    if (a < best1) {
                        best1 = a;
                        pos1[0] = b->origin[0] + (i - g->ghost + 0.5) * g->dx;
                        pos1[1] = b->origin[1] + (j - g->ghost + 0.5) * g->dx;
                        pos1[2] = b->origin[2] + (k - g->ghost + 0.5) * g->dx;
                    }
                }
    }

    /* Pass 2: find deepest minimum at least 2M from BH #1 (BH #2) */
    double best2 = 1.0e30;
    pos2[0] = pos2[1] = pos2[2] = 0.0;

    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        grid_t *g = b->grid;
        int lo = g->ghost, hi = g->ghost + g->N;
        for (int k = lo; k < hi; k++)
            for (int j = lo; j < hi; j++)
                for (int i = lo; i < hi; i++) {
                    double a = g->fields[FIELD_LAPSE][IDX(g, i, j, k)];
                    if (a < best2) {
                        double x = b->origin[0] + (i - g->ghost + 0.5) * g->dx;
                        double y = b->origin[1] + (j - g->ghost + 0.5) * g->dx;
                        double z = b->origin[2] + (k - g->ghost + 0.5) * g->dx;
                        double dr = sqrt((x - pos1[0]) * (x - pos1[0]) +
                                         (y - pos1[1]) * (y - pos1[1]) +
                                         (z - pos1[2]) * (z - pos1[2]));
                        if (dr > 2.0) {
                            best2 = a;
                            pos2[0] = x; pos2[1] = y; pos2[2] = z;
                        }
                    }
                }
    }

    if (best2 > 0.99) return 0.0;

    return sqrt((pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) +
                (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]) +
                (pos1[2] - pos2[2]) * (pos1[2] - pos2[2]));
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


/* ====================================================================
 * Main
 * ==================================================================== */
int main(void)
{
    setbuf(stdout, NULL);

    /* ── Banner ─────────────────────────────────────────────────────── */
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Lattice — Binary Black Hole Inspiral                      ║\n");
    printf("║  Full System Validation Test                               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("  Physical setup:\n");
    printf("    m1 = m2 = %.4f  (M_ADM ~ 1.0)\n", M_BARE);
    printf("    d = %.0f M  (z-axis)\n", D_SEP);
    printf("    P_y = +/-%.4f  (3PN quasi-circular)\n", P_Y);
    printf("    T_orbit ~ 200 M\n");
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
    printf("    AH finder:  %dx%d angular grid, every %d steps\n",
           AH_NTHETA, AH_NPHI, AH_EVERY);
    printf("    Constraints: every %d steps\n", DIAG_EVERY);
    printf("    1D slices:  every %d steps\n", SLICE_EVERY);
    printf("\n");
    printf("  Ref: arXiv:0709.0838 (Brugmann et al. 2008)\n");
    printf("\n");

    time_t wall_start = time(NULL);

    /* ── Initialize ────────────────────────────────────────────────── */
    backend_init();
    mesh_t *m = mesh_create(N_BLOCK, L_DOMAIN, RK_CLASSIC);

    sim_params_t p = default_params();
    p.L         = L_DOMAIN;
    p.rk_method = RK_CLASSIC;
    p.CFL       = CFL_FACTOR;
    p.dx        = m->dx_base;
    p.dt        = p.CFL * p.dx;
    p.sigma     = 0.3;
    p.bc_type   = BC_CONSTRAINT_PRESERVING;
    p.noise.use_ssl = 0;  /* SSL delays gauge formation at fine AMR levels */

    amr_params_t ap;
    ap.max_level   = MAX_LEVEL;
    ap.chi_refine  = 0.5;
    ap.chi_coarsen = 0.01;

    int num_steps = (int)(T_FINAL / p.dt + 0.5);

    printf("  dx = %.4f M, dt = %.6f M, CFL = %.2f\n", p.dx, p.dt, p.CFL);
    printf("  Steps: %d  (T_final = %.0f M)\n", num_steps, T_FINAL);
    printf("  Leaf blocks: %d (initial)\n\n", mesh_num_leaves(m));

    /* ── Initial data ──────────────────────────────────────────────── */
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

    /* Use set_bowen_york_mesh() which handles everything:
     *   1. Refine near punctures (n_amr_levels deep)
     *   2. Solve constraint equation (FAS composite multigrid)
     *   3. Convert solver fields → CCZ4 on each leaf block
     *   4. Ghost exchange for CCZ4 fields across blocks
     *
     * The composite solver uses ghost_exchange_all_blocks() to fill
     * ghost zones for non-leaf blocks in multi-root meshes, fixing
     * the divergence bug that previously required the workaround. */
    set_bowen_york_mesh(m, 2, bhs, MAX_LEVEL);

    printf("        Done (%.0f sec)\n\n", difftime(time(NULL), id_start));

    /* ── Post-solver mesh state ───────────────────────────────────── */
    printf("  [2/4] Mesh after initial data...\n");
    p.dx = m->dx_base;
    p.dt = p.CFL * p.dx;
    num_steps = (int)(T_FINAL / p.dt + 0.5);
    printf("        Leaves: %d, max_level = %d\n\n",
           mesh_num_leaves(m), m->max_level);

    /* ── Allocate diagnostic workspaces ────────────────────────────── */
    printf("  [3/4] Allocating diagnostics...\n");

    /* Psi4 extraction sphere */
    double psi4_center[3] = {0, 0, 0};
    psi4_workspace_t *psi4_ws = psi4_alloc(PSI4_NTHETA, PSI4_NPHI,
                                            PSI4_LMAX, PSI4_RADIUS,
                                            psi4_center);
    printf("        Psi4: %d modes, r = %.0f M\n",
           psi4_ws->n_modes, PSI4_RADIUS);

    /* Apparent horizon finders — one per BH, centered on lapse minima */
    double bh1_pos[3], bh2_pos[3];
    mesh_bh_separation(m, bh1_pos, bh2_pos);

    ah_workspace_t *ah1 = ah_alloc(AH_NTHETA, AH_NPHI, bh1_pos, AH_RGUESS);
    ah_workspace_t *ah2 = ah_alloc(AH_NTHETA, AH_NPHI, bh2_pos, AH_RGUESS);
    printf("        AH finder: 2 horizons, %dx%d angular grid\n\n",
           AH_NTHETA, AH_NPHI);

    /* ── Initial diagnostics ───────────────────────────────────────── */
    double lx, ly, lz;
    double ml      = mesh_min_lapse(m, &lx, &ly, &lz);
    double sep0    = mesh_bh_separation(m, bh1_pos, bh2_pos);
    double ham     = mesh_constraint_l2(m);
    double mom     = mesh_momentum_l2(m);

    printf("  [4/4] Evolution: %d steps, T = %.0f M\n\n", num_steps, T_FINAL);

    /* ── Evolution log header ──────────────────────────────────────── */
    printf("  step    t/M    alpha_min  sep/M  leaves  Ham_L2      Mom_L2      "
           "|rPsi4_22|  AH1_M_irr  AH2_M_irr  wall(s)\n");
    printf("  ─────────────────────────────────────────────────────────────────"
           "──────────────────────────────────────────\n");

    printf("  %5d  %6.1f    %6.4f  %6.2f  %5d   %10.3e  %10.3e"
           "          -          -          -     0.0\n",
           0, 0.0, ml, sep0, mesh_num_leaves(m), ham, mom);
    fflush(stdout);

    /* ── Tracking ──────────────────────────────────────────────────── */
    double ham_peak     = ham;
    double mom_peak     = mom;
    double psi4_22_max  = 0.0;
    double ml_min       = ml;
    double ah1_mass_last = -1.0, ah2_mass_last = -1.0;
    int    ah1_found = 0, ah2_found = 0;
    int    crashed = 0;

    /* ── Diagnostics CSV file ─────────────────────────────────────── */
    FILE *diag_fp = fopen("build/inspiral_diagnostics.csv", "w");
    if (diag_fp) {
        fprintf(diag_fp, "time,ham_l2,mom_l2,alpha_min,separation,"
                "leaves,psi4_22_re,psi4_22_im,psi4_22_amp,"
                "ah1_mass,ah1_spin,ah2_mass,ah2_spin\n");
        fprintf(diag_fp, "%.6f,%.6e,%.6e,%.6f,%.6f,%d,0,0,0,-1,-1,-1,-1\n",
                0.0, ham, mom, ml, sep0, mesh_num_leaves(m));
        fflush(diag_fp);
    }

    /* ── Evolution loop ────────────────────────────────────────────── */
    p.time = 0.0;

    struct timespec ts_start, ts_end;
    for (int step = 1; step <= num_steps; step++) {
        /* ── Advance one time step ─────────────────────────────────── */
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        double step_wall = (ts_end.tv_sec - ts_start.tv_sec)
                         + 1e-9 * (ts_end.tv_nsec - ts_start.tv_nsec);
        p.time += p.dt;

        /* ── Periodic regridding ───────────────────────────────────── */
        if (step % REGRID_EVERY == 0)
            mesh_regrid(m, &ap);

        /* ── Diagnostics ───────────────────────────────────────────── */
        if (step % DIAG_EVERY == 0 || step == num_steps) {

            /* Stability check */
            if (!mesh_check_finite(m)) {
                printf("\n  *** CRASH: NaN/Inf detected at step %d "
                       "(t = %.2f M) ***\n\n", step, p.time);
                crashed = 1;
                break;
            }

            /* Constraint norms */
            ham = mesh_constraint_l2(m);
            mom = mesh_momentum_l2(m);
            if (ham > ham_peak) ham_peak = ham;
            if (mom > mom_peak) mom_peak = mom;

            /* Lapse and separation */
            ml = mesh_min_lapse(m, &lx, &ly, &lz);
            if (ml < ml_min) ml_min = ml;
            double sep = mesh_bh_separation(m, bh1_pos, bh2_pos);

            /* Psi4 extraction */
            double psi4_22_amp = 0.0;
            if (step % PSI4_EVERY == 0) {
                psi4_extract(psi4_ws, m);
                psi4_write_modes(psi4_ws, p.time,
                                 "build/inspiral_psi4.csv");

                /* (l=2, m=2) mode index = l*l + l + m - 4 = 4 */
                int mi22 = 4;
                if (mi22 < psi4_ws->n_modes) {
                    double re = psi4_ws->mode_re[mi22];
                    double im = psi4_ws->mode_im[mi22];
                    psi4_22_amp = sqrt(re * re + im * im);
                    if (psi4_22_amp > psi4_22_max)
                        psi4_22_max = psi4_22_amp;
                }
            }

            /* Apparent horizons */
            double ah1_m = -1.0, ah2_m = -1.0;
            double ah1_spin = -1.0, ah2_spin = -1.0;
            if (step % AH_EVERY == 0) {
                /* Update AH centers to track moving punctures */
                ah1->center[0] = bh1_pos[0];
                ah1->center[1] = bh1_pos[1];
                ah1->center[2] = bh1_pos[2];
                ah2->center[0] = bh2_pos[0];
                ah2->center[1] = bh2_pos[1];
                ah2->center[2] = bh2_pos[2];

                if (ah_find_amr(ah1, m, AH_TOL, AH_MAXITER, 0)) {
                    ah_result_t r1 = ah_compute_diagnostics_amr(ah1, m);
                    if (r1.converged) {
                        ah1_m = r1.mass_irr;
                        ah1_spin = r1.chi_spin;
                        ah1_mass_last = ah1_m;
                        ah1_found = 1;
                    }
                }
                if (ah_find_amr(ah2, m, AH_TOL, AH_MAXITER, 0)) {
                    ah_result_t r2 = ah_compute_diagnostics_amr(ah2, m);
                    if (r2.converged) {
                        ah2_m = r2.mass_irr;
                        ah2_spin = r2.chi_spin;
                        ah2_mass_last = ah2_m;
                        ah2_found = 1;
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
                    "%.6f,%.6e,%.6e,%.6f,%.6f,%d,%.6e,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f\n",
                    p.time, ham, mom, ml, sep, mesh_num_leaves(m),
                    p22_re, p22_im, psi4_22_amp,
                    ah1_m, ah1_spin, ah2_m, ah2_spin);
                fflush(diag_fp);
            }
        }

        /* ── 1D slice output ───────────────────────────────────────── */
        if (step % SLICE_EVERY == 0)
            output_mesh_1d_slice(m, step, p.time);
    }

    /* ── Final state ───────────────────────────────────────────────── */
    double final_sep = mesh_bh_separation(m, bh1_pos, bh2_pos);
    double wall_sec  = difftime(time(NULL), wall_start);

    /* Final Psi4 extraction */
    if (!crashed) {
        psi4_extract(psi4_ws, m);
        int mi22 = 4;
        if (mi22 < psi4_ws->n_modes) {
            double re = psi4_ws->mode_re[mi22];
            double im = psi4_ws->mode_im[mi22];
            double amp = sqrt(re * re + im * im);
            if (amp > psi4_22_max) psi4_22_max = amp;
        }
    }

    /* ── Results ───────────────────────────────────────────────────── */
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Results                                                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    int t1 = !crashed && mesh_check_finite(m);
    int t2 = ham_peak < 1.0;
    int t3 = psi4_22_max > 1.0e-6;
    int t4 = ml_min < 0.5;
    int t5 = final_sep < sep0 * 0.99;

    printf("  Test 1 — Stability (no NaN/Inf):        %s\n",
           t1 ? "PASS" : "FAIL");
    printf("  Test 2 — Constraints bounded:           %s"
           "  (Ham peak = %.3e, limit 1.0)\n",
           t2 ? "PASS" : "FAIL", ham_peak);
    printf("  Test 3 — Gravitational radiation:       %s"
           "  (max |rPsi4_22| = %.3e)\n",
           t3 ? "PASS" : "FAIL", psi4_22_max);
    printf("  Test 4 — Trumpet lapse formed:          %s"
           "  (min alpha = %.4f, need < 0.5)\n",
           t4 ? "PASS" : "FAIL", ml_min);
    printf("  Test 5 — Inspiral motion:               %s"
           "  (sep: %.2f -> %.2f M)\n",
           t5 ? "PASS" : "FAIL", sep0, final_sep);

    printf("\n  Apparent horizons:\n");
    if (ah1_found)
        printf("    BH 1: M_irr = %.4f  (last found)\n", ah1_mass_last);
    else
        printf("    BH 1: not found\n");
    if (ah2_found)
        printf("    BH 2: M_irr = %.4f  (last found)\n", ah2_mass_last);
    else
        printf("    BH 2: not found\n");

    printf("\n  Performance:\n");
    printf("    Wall time:    %.0f sec (%.1f min)\n", wall_sec, wall_sec / 60.0);
    printf("    Final leaves: %d, max_level: %d\n",
           mesh_num_leaves(m), m->max_level);
    if (num_steps > 0)
        printf("    Avg sec/step: %.2f\n", wall_sec / num_steps);

    printf("\n  Output files:\n");
    printf("    build/inspiral_diagnostics.csv — time series (Ham, Mom, sep, lapse, Psi4, AH)\n");
    printf("    build/inspiral_psi4.csv        — Psi4 mode coefficients vs time\n");
    printf("    build/slice_*.csv              — 1D lapse/chi/K profiles\n");

    int all_passed = t1 && t2 && t3 && t4 && t5;
    printf("\n  ════════════════════════════════════════════\n");
    printf("  %s (%d/5 tests passed)\n",
           all_passed ? "ALL PASSED" : "FAILED", t1 + t2 + t3 + t4 + t5);
    printf("  ════════════════════════════════════════════\n\n");

    /* ── Cleanup ───────────────────────────────────────────────────── */
    if (diag_fp) fclose(diag_fp);
    psi4_free(psi4_ws);
    ah_free(ah1);
    ah_free(ah2);
    mesh_free(m);
    backend_cleanup();

    return all_passed ? 0 : 1;
}
