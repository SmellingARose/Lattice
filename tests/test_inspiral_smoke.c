/*
 * ====================================================================
 * Lattice -- 3D Numerical Relativity
 * Binary Inspiral Smoke Test: Fast Diagnostic Subsystem Validation
 * ====================================================================
 *
 * Scaled-down version of the full D10 inspiral benchmark.  Same physics
 * (Bowen-York initial data, CCZ4 evolution, CAKO + SSL + per-field sigma,
 * CP BCs, position-dependent eta) on a coarse uniform grid for T=700M.
 * Runs in ~7 minutes on CPU (single block, no AMR).
 *
 * Purpose: exercise ALL diagnostic subsystems through a full inspiral
 * duration without requiring the multi-hour AMR run on GPU.  Catches
 * integration bugs, API breakage, long-term stability, and NaN crashes.
 *
 * Physical setup (identical to full test):
 *   m1 = m2 = 0.48595, d = 10M (z-axis), P_y = +/-0.09543
 *
 * Grid (reduced resolution, full duration):
 *   L = 20M, N = 32 (dx = 0.625M), 0 AMR levels (single block)
 *   T = 10M (~32 steps at CFL = 0.25)
 *
 * Diagnostic subsystems exercised:
 *   1. Constraint monitoring (Ham L2, Mom L2)
 *   2. BH tracker (positions, separation, merger check)
 *   3. Psi4 extraction (r = 15M, l_max = 2, 8x16)
 *   4. AH finder (per-BH, 8x16)
 *   5. CSV diagnostic output
 *   6. Lapse minimum tracking
 *
 * Pass criteria (relaxed for dx = 1.25M):
 *   1. No NaN/Inf
 *   2. Ham L2 bounded (< 1.0)
 *   3. Lapse collapsed (min < 0.9)
 *   4. BH tracker found both BHs (separation > 0)
 *   5. Psi4 extraction ran without crash
 *   6. AH finder ran without crash
 *   7. CSV written with correct column count
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/amr/mesh.h"
#include "../src/amr/ghost_exchange.h"
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

/* ====================================================================
 * CHECK macro (same pattern as test_bowen_york.c)
 * ==================================================================== */
static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        printf("  PASS: %s\n", msg); \
        tests_passed++; \
    } else { \
        printf("  FAIL: %s\n", msg); \
        tests_failed++; \
    } \
} while(0)

/* ====================================================================
 * Physical parameters (identical to full test)
 * ==================================================================== */
#define M_BARE     0.48595
#define D_SEP      10.0
#define P_Y        0.09543

/* ====================================================================
 * Reduced grid and evolution parameters
 * ==================================================================== */
#define L_DOMAIN    20.0
#define N_BLOCK     32
#define CFL_FACTOR  0.25
#define T_FINAL     700.0

/* ====================================================================
 * Diagnostic schedule
 * ==================================================================== */
#define DIAG_EVERY    100
#define AH_STEP_1     10

/* ====================================================================
 * Wave extraction (single radius, small angular grid)
 * ==================================================================== */
#define PSI4_RADIUS   8.0
#define PSI4_LMAX     2
#define PSI4_NTHETA   8
#define PSI4_NPHI     16

/* ====================================================================
 * Apparent horizon parameters (small angular grid)
 * ==================================================================== */
#define AH_NTHETA     8
#define AH_NPHI       16
#define AH_RGUESS     0.5
#define AH_TOL        1.0e-2
#define AH_MAXITER    100

/* ====================================================================
 * CSV output path
 * ==================================================================== */
#define CSV_PATH  "build/smoke_diagnostics.csv"


/* ====================================================================
 * Mesh-level diagnostics (same as full test)
 * ==================================================================== */

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

    /* -- Banner -------------------------------------------------------- */
    printf("\n");
    printf("==================================================================\n");
    printf("  Lattice -- Binary Inspiral Smoke Test\n");
    printf("  D10 setup, coarse grid, all diagnostic subsystems\n");
    printf("==================================================================\n\n");

    printf("  Physical setup: m1 = m2 = %.5f, d = %.0f M, P_y = +/-%.5f\n",
           M_BARE, D_SEP, P_Y);
    printf("  Grid: L = %.0f M, N = %d (dx = %.4f M), 0 AMR levels\n",
           L_DOMAIN, N_BLOCK, L_DOMAIN / N_BLOCK);
    printf("  Evolution: T = %.0f M, CFL = %.2f\n", T_FINAL, CFL_FACTOR);
    printf("  Diagnostics: every %d steps, Psi4 r = %.0f M, AH %dx%d\n\n",
           DIAG_EVERY, PSI4_RADIUS, AH_NTHETA, AH_NPHI);

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

    /* Physics parameters (identical to full test) */
    p.ccz4.kappa1 = 0.02;

    p.noise.use_cako = 1;
    p.noise.use_per_field_sigma = 1;
    p.noise.sigma_gauge = 0.99;
    p.noise.sigma_phys  = 0.3;
    p.noise.use_ssl = 1;

    p.gauge.lapse_advec_coeff = 1.0;
    p.gauge.shift_advec_coeff = 1.0;
    p.gauge.eta = 2.0;
    p.gauge.position_dependent_eta = 1;

    int num_steps = (int)(T_FINAL / p.dt + 0.5);

    printf("  dx = %.4f M, dt = %.6f M, steps = %d\n\n", p.dx, p.dt, num_steps);

    /* -- Initial data -------------------------------------------------- */
    printf("  [1/4] Solving Bowen-York initial data...\n");
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

    set_bowen_york_mesh(m, 2, bhs, 0);  /* 0 AMR levels */

    printf("        Done (%.0f sec)\n\n", difftime(time(NULL), id_start));

    /* -- Allocate diagnostic workspaces -------------------------------- */
    printf("  [2/4] Allocating diagnostics...\n");

    /* Psi4 extraction sphere */
    double psi4_center[3] = {0, 0, 0};
    psi4_workspace_t *psi4_ws = psi4_alloc(PSI4_NTHETA, PSI4_NPHI,
                                            PSI4_LMAX, PSI4_RADIUS,
                                            psi4_center);
    printf("        Psi4: %d modes, r = %.0f M\n", psi4_ws->n_modes, PSI4_RADIUS);

    /* BH tracker */
    int n_bh = 2;
    bh_tracker_t *tracker = bh_tracker_alloc(n_bh, bhs, AH_NTHETA, AH_NPHI);
    bh_tracker_update_positions(tracker, m);
    printf("        BH tracker: %d BHs (%dx%d AH per BH)\n",
           n_bh, AH_NTHETA, AH_NPHI);
    printf("\n");

    /* -- Initial diagnostics ------------------------------------------- */
    double lx, ly, lz;
    double ml      = mesh_min_lapse(m, &lx, &ly, &lz);
    double sep0    = tracker_separation(tracker);
    double ham     = mesh_constraint_l2(m);
    double mom     = mesh_momentum_l2(m);

    printf("  [3/4] Initial state: alpha_min = %.4f, sep = %.2f M, "
           "Ham = %.3e, Mom = %.3e\n\n", ml, sep0, ham, mom);

    /* -- Evolution log header ------------------------------------------ */
    printf("  [4/4] Evolution: %d steps, T = %.0f M\n\n", num_steps, T_FINAL);

    printf("  step    t/M    alpha_min  sep/M    Ham_L2      Mom_L2      "
           "|rPsi4_22|\n");
    printf("  -----------------------------------------------------------"
           "-----------\n");

    printf("  %5d  %6.2f    %6.4f  %6.2f   %10.3e  %10.3e          -\n",
           0, 0.0, ml, sep0, ham, mom);

    /* -- Tracking variables -------------------------------------------- */
    double ham_peak    = ham;
    double ml_min      = ml;
    int    crashed     = 0;
    int    psi4_ran    = 0;
    int    ah_ran      = 0;
    int    tracker_found_both = (sep0 > 0.0) ? 1 : 0;

    /* -- CSV file ------------------------------------------------------ */
    int csv_columns = 0;
    FILE *diag_fp = fopen(CSV_PATH, "w");
    if (diag_fp) {
        fprintf(diag_fp, "time,ham_l2,mom_l2,alpha_min,separation");
        for (int i = 0; i < n_bh; i++)
            fprintf(diag_fp, ",bh%d_x,bh%d_y,bh%d_z,bh%d_mass,bh%d_spin,bh%d_lapse",
                    i, i, i, i, i, i);
        fprintf(diag_fp, ",psi4_22_re,psi4_22_im,psi4_22_amp\n");
        /* Count columns: 5 base + 6*n_bh + 3 psi4 */
        csv_columns = 5 + 6 * n_bh + 3;

        /* Initial row */
        fprintf(diag_fp, "%.6f,%.6e,%.6e,%.6f,%.6f",
                0.0, ham, mom, ml, sep0);
        for (int i = 0; i < n_bh; i++)
            fprintf(diag_fp, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    tracker->bh[i].center[0], tracker->bh[i].center[1],
                    tracker->bh[i].center[2], 0.0, 0.0, tracker->bh[i].lapse_min);
        fprintf(diag_fp, ",0,0,0\n");
        fflush(diag_fp);
    }

    /* -- Evolution loop ------------------------------------------------ */
    p.time = 0.0;

    struct timespec ts_start, ts_end;
    for (int step = 1; step <= num_steps; step++) {
        /* Advance one step */
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        (void)ts_start; (void)ts_end;
        p.time += p.dt;

        /* Periodic diagnostics */
        if (step % DIAG_EVERY == 0 || step == num_steps) {
            /* NaN check */
            if (!mesh_check_finite(m)) {
                printf("\n  *** CRASH: NaN/Inf at step %d (t = %.2f M) ***\n\n",
                       step, p.time);
                crashed = 1;
                break;
            }

            /* Constraints */
            ham = mesh_constraint_l2(m);
            mom = mesh_momentum_l2(m);
            ml  = mesh_min_lapse(m, &lx, &ly, &lz);
            if (ham > ham_peak) ham_peak = ham;
            if (ml < ml_min) ml_min = ml;

            /* BH tracker */
            bh_tracker_update_positions(tracker, m);
            double sep = tracker_separation(tracker);
            bh_tracker_check_mergers(tracker, p.time);
            if (sep > 0.0) tracker_found_both = 1;

            /* Psi4 extraction */
            double psi4_22_amp = 0.0;
            double p22_re = 0.0, p22_im = 0.0;
            psi4_extract(psi4_ws, m);
            psi4_ran = 1;
            int mi22 = 4;  /* (l=2,m=2) mode index */
            if (mi22 < psi4_ws->n_modes) {
                p22_re = psi4_ws->mode_re[mi22];
                p22_im = psi4_ws->mode_im[mi22];
                psi4_22_amp = sqrt(p22_re * p22_re + p22_im * p22_im);
            }

            /* AH finder at designated steps */
            if (step == AH_STEP_1 || step == num_steps) {
                bh_tracker_find_horizons(tracker, m, AH_TOL, AH_MAXITER);
                ah_ran = 1;
            }

            /* Log line */
            printf("  %5d  %6.2f    %6.4f  %6.2f   %10.3e  %10.3e  %10.3e\n",
                   step, p.time, ml, sep, ham, mom, psi4_22_amp);

            /* CSV row */
            if (diag_fp) {
                fprintf(diag_fp, "%.6f,%.6e,%.6e,%.6f,%.6f",
                        p.time, ham, mom, ml, sep);
                for (int i = 0; i < n_bh; i++) {
                    const bh_state_t *bh = &tracker->bh[i];
                    fprintf(diag_fp, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                            bh->center[0], bh->center[1], bh->center[2],
                            bh->mass_chr, bh->chi_spin, bh->lapse_min);
                }
                fprintf(diag_fp, ",%.6e,%.6e,%.6e\n", p22_re, p22_im, psi4_22_amp);
                fflush(diag_fp);
            }
        }
    }

    if (diag_fp) fclose(diag_fp);
    double wall_sec = difftime(time(NULL), wall_start);

    /* -- Verify CSV column count --------------------------------------- */
    int csv_ok = 0;
    {
        FILE *fp = fopen(CSV_PATH, "r");
        if (fp) {
            char line[4096];
            if (fgets(line, sizeof(line), fp)) {
                /* Count commas in header to get column count */
                int commas = 0;
                for (char *c = line; *c; c++)
                    if (*c == ',') commas++;
                csv_ok = (commas + 1 == csv_columns);
                if (!csv_ok)
                    printf("  CSV columns: got %d, expected %d\n",
                           commas + 1, csv_columns);
            }
            fclose(fp);
        }
    }

    /* -- Final checks -------------------------------------------------- */
    printf("\n");
    printf("==================================================================\n");
    printf("  Results (%.0f sec wall time)\n", wall_sec);
    printf("==================================================================\n\n");

    CHECK(!crashed && mesh_check_finite(m),
          "No NaN/Inf in any evolved field");

    CHECK(ham_peak < 1.0,
          "Ham L2 bounded (< 1.0)");

    CHECK(ml_min < 0.9,
          "Lapse collapsed (min < 0.9, trumpet forming)");

    CHECK(tracker_found_both,
          "BH tracker found both BHs (separation > 0)");

    CHECK(psi4_ran,
          "Psi4 extraction ran without crash");

    CHECK(ah_ran,
          "AH finder ran without crash");

    CHECK(csv_ok,
          "CSV file written with correct column count");

    printf("\n  Summary: %d/%d passed\n\n",
           tests_passed, tests_passed + tests_failed);

    /* -- Cleanup ------------------------------------------------------- */
    bh_tracker_free(tracker);
    psi4_free(psi4_ws);
    mesh_free(m);
    backend_cleanup();

    return tests_failed > 0 ? 1 : 0;
}
