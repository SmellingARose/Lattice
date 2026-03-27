/*
 * Lattice — 3D Numerical Relativity
 * Schwarzschild quasi-normal mode ringdown test (AMR).
 *
 * A single M=1 puncture BH starts from conformally flat initial data.
 * The gauge (1+log slicing + Gamma-driver shift) rings down to the
 * trumpet state, exciting the fundamental l=2 QNM.
 *
 * Known Schwarzschild l=2 QNM (Leaver 1985):
 *   ω_R = 0.37367 / M   (oscillation frequency)
 *   ω_I = 0.08896 / M   (damping rate)
 *
 * AMR grid: N_block=32, L=64, 3 refinement levels, CK45 integrator.
 *   Level 0: dx = 2.0M   (boundary at 32M)
 *   Level 3: dx = 0.25M  (resolves trumpet gauge)
 * Extraction at r=15M on the base level.
 * Memory: ~12 GB (CK45). Needs ≥16 GB RAM or GPU.
 *
 * Ref: Leaver (1985), Berti et al. (2009, arXiv:0905.2975)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/puncture.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/amr/mesh.h"
#include "../src/amr/refine.h"
#include "../src/amr/ghost_exchange.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/diagnostics/psi4.h"
#include "../src/backend/backend.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int n_pass = 0, n_fail = 0;

static void check(int cond, const char *msg)
{
    if (cond) {
        printf("  [PASS] %s\n", msg);
        n_pass++;
    } else {
        printf("  [FAIL] %s\n", msg);
        n_fail++;
    }
}

static inline int mode_index(int l, int m) { return l * l + l + m - 4; }

/* Find min lapse across all leaf blocks */
static double mesh_min_lapse(const mesh_t *m)
{
    double best = 1e30;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *blk = m->blocks[bid];
        if (!blk || !blk->is_leaf) continue;
        grid_t *g = blk->grid;
        int lo = g->ghost, hi = g->ghost + g->N;
        for (int k = lo; k < hi; k++)
            for (int j = lo; j < hi; j++)
                for (int i = lo; i < hi; i++) {
                    double a = g->fields[FIELD_LAPSE][IDX(g,i,j,k)];
                    if (a < best) best = a;
                }
    }
    return best;
}

int main(void)
{
    setbuf(stdout, NULL);
    printf("=== Schwarzschild QNM Ringdown Test (AMR) ===\n\n");
    backend_init();

    /* AMR grid: N_block=32, L=64, 3 levels → dx_fine = 0.25M
     * Level 0: 1 block (32³), dx=2M.  Boundary at 32M.
     * Level 3: ~64 blocks (32³), dx=0.25M near puncture.
     * dx=0.5M crashes at t≈20M (gauge instability). dx=0.25M is stable.
     * CK45: 3 memory blocks (25% less than RK4 classic).
     * Memory: ~120 blocks × 40³ × 25 × 3 × 8 ≈ 5.8 GB. */
    int N_block = 32;
    double L = 64.0;
    double M_bh = 1.0;
    int max_level = 3;

    mesh_t *m = mesh_create_ex(N_block, L, RK_CLASSIC, NUM_CCZ4_FIELDS);
    mesh_rebuild_neighbors(m);

    sim_params_t p = default_params();
    p.rk_method = RK_CLASSIC;
    p.N = N_block;
    p.L = L;
    p.dx = m->dx_base;
    p.dt = 0.25 * p.dx;  /* CFL on base level */
    p.amr.max_level = max_level;
    p.amr.chi_refine = 0.1;
    p.amr.regrid_every = 0;  /* static mesh after initial regrid */

    printf("  N_block=%d, L=%.0f, dx_base=%.3f, dt=%.4f\n",
           N_block, L, p.dx, p.dt);
    printf("  AMR levels=%d, dx_fine=%.4f\n",
           max_level, p.dx / (1 << max_level));

    /* Initial data + AMR refinement */
    puncture_data_t bh = {.mass = M_bh, .center = {0, 0, 0}};
    set_bowen_york_mesh(m, 1, &bh, max_level);
    ghost_exchange(m);

    printf("  Mesh: %d blocks (%d leaves), max_level=%d\n",
           m->num_blocks, mesh_num_leaves(m), m->max_level);

    double ham0 = mesh_constraint_l2(m);
    double ml0 = mesh_min_lapse(m);
    printf("  Initial: Ham L2=%.4e, min_lapse=%.4f\n\n", ham0, ml0);

    /* Psi4 at r=15M (outside refinement region, inside boundary at 32M).
     * Boundary reflections reach r=15M at t ≈ 2*(32-15)/1 = 34M.
     * Start analysis at t=40M, so we have a clean window from ~10-34M
     * of uncontaminated QNM signal (2+ cycles). */
    double psi4_radius = 15.0;
    double psi4_center[3] = {0, 0, 0};
    psi4_workspace_t *ws = psi4_alloc(16, 32, 4, psi4_radius, psi4_center);

    /* Time series */
    double T_final = 100.0;  /* QNM decays by e^-9 after 100M — plenty */
    int total_steps = (int)(T_final / p.dt + 0.5);
    int psi4_every = 2;
    int max_samples = total_steps / psi4_every + 2;
    double *t_arr  = malloc(max_samples * sizeof(double));
    double *re_arr = malloc(max_samples * sizeof(double));
    double *im_arr = malloc(max_samples * sizeof(double));
    int n_samples = 0;

    int mi_20 = mode_index(2, 0);

    printf("  Evolving T=%.0fM (%d base steps), Psi4 every %d steps at r=%.0fM\n\n",
           T_final, total_steps, psi4_every, psi4_radius);

    /* --- Evolution --- */
    p.time = 0.0;
    for (int step = 1; step <= total_steps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;

        if (step % psi4_every == 0) {
            psi4_extract(ws, m);
            t_arr[n_samples]  = p.time;
            re_arr[n_samples] = ws->mode_re[mi_20];
            im_arr[n_samples] = ws->mode_im[mi_20];
            n_samples++;
        }

        if (step % 10 == 0 || step == total_steps) {
            double ham = mesh_constraint_l2(m);
            double ml = mesh_min_lapse(m);
            printf("  step %4d  t=%6.1fM  Ham=%.4e  lapse_min=%.4f",
                   step, p.time, ham, ml);
            if (n_samples > 0) {
                double amp = sqrt(re_arr[n_samples-1]*re_arr[n_samples-1] +
                                  im_arr[n_samples-1]*im_arr[n_samples-1]);
                printf("  |Psi4(2,0)|=%.4e", amp);
            }
            printf("\n");
        }
    }

    printf("\n  Collected %d Psi4 samples over T=%.1fM\n\n", n_samples, p.time);

    /* --- Frequency from zero crossings --- */
    printf("--- QNM Frequency Analysis ---\n");
    double t_start = 30.0;  /* past initial transient + junk radiation */
    int i_start = 0;
    for (int i = 0; i < n_samples; i++) {
        if (t_arr[i] >= t_start) { i_start = i; break; }
    }

    int n_crossings = 0;
    double crossing_times[500];
    for (int i = i_start; i < n_samples - 1; i++) {
        if (re_arr[i] * re_arr[i+1] < 0.0) {
            double frac = fabs(re_arr[i]) / (fabs(re_arr[i]) + fabs(re_arr[i+1]));
            crossing_times[n_crossings] = t_arr[i] + frac * (t_arr[i+1] - t_arr[i]);
            n_crossings++;
            if (n_crossings >= 500) break;
        }
    }

    double omega_R_measured = 0.0;
    if (n_crossings >= 4) {
        double sum_hp = 0.0;
        for (int i = 0; i < n_crossings - 1; i++)
            sum_hp += crossing_times[i+1] - crossing_times[i];
        double avg_hp = sum_hp / (n_crossings - 1);
        omega_R_measured = M_PI / avg_hp;
        printf("  Zero crossings: %d (t=%.1f to %.1fM)\n",
               n_crossings, crossing_times[0], crossing_times[n_crossings-1]);
        printf("  Average half-period: %.3fM\n", avg_hp);
        printf("  Measured ω_R = %.4f/M  (expected 0.3737)\n", omega_R_measured);
        printf("  Error: %.1f%%\n", 100.0*fabs(omega_R_measured-0.3737)/0.3737);
    } else {
        printf("  Only %d zero crossings (need >= 4)\n", n_crossings);
    }

    /* --- Damping from peak amplitudes --- */
    printf("\n--- QNM Damping Analysis ---\n");
    int n_peaks = 0;
    double peak_t[200], peak_a[200];
    for (int i = i_start + 1; i < n_samples - 1; i++) {
        double a_p = fabs(re_arr[i-1]), a_c = fabs(re_arr[i]), a_n = fabs(re_arr[i+1]);
        if (a_c > a_p && a_c > a_n && a_c > 1e-15) {
            peak_t[n_peaks] = t_arr[i];
            peak_a[n_peaks] = a_c;
            n_peaks++;
            if (n_peaks >= 200) break;
        }
    }

    double omega_I_measured = 0.0;
    if (n_peaks >= 3) {
        double st=0, slA=0, st2=0, stlA=0;
        for (int i = 0; i < n_peaks; i++) {
            double t = peak_t[i], lA = log(peak_a[i]);
            st += t; slA += lA; st2 += t*t; stlA += t*lA;
        }
        double slope = (n_peaks*stlA - st*slA) / (n_peaks*st2 - st*st);
        omega_I_measured = -slope;
        printf("  Peaks: %d\n", n_peaks);
        for (int i = 0; i < n_peaks && i < 6; i++)
            printf("    t=%.1fM  |Psi4|=%.4e\n", peak_t[i], peak_a[i]);
        if (n_peaks > 6) printf("    ...\n");
        printf("  Measured ω_I = %.4f/M  (expected 0.0890)\n", omega_I_measured);
        printf("  Error: %.1f%%\n", 100.0*fabs(omega_I_measured-0.0890)/0.0890);
    } else {
        printf("  Only %d peaks (need >= 3)\n", n_peaks);
    }

    /* --- Gauge check --- */
    printf("\n--- Gauge ---\n");
    double min_alpha = mesh_min_lapse(m);
    double ham_final = mesh_constraint_l2(m);
    printf("  Final min lapse: %.4f (trumpet ≈ 0.3)\n", min_alpha);
    printf("  Final Ham L2: %.4e\n", ham_final);

    /* --- Pass/fail --- */
    printf("\n--- Results ---\n");
    check(n_crossings >= 4, "Enough zero crossings");
    check(n_peaks >= 3, "Enough peaks for damping");

    if (n_crossings >= 4) {
        double err = fabs(omega_R_measured - 0.3737) / 0.3737;
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "QNM freq ω_R=%.4f within 25%% of 0.3737 (err=%.1f%%)",
                 omega_R_measured, 100*err);
        check(err < 0.25, msg);
    }

    if (n_peaks >= 3) {
        double err = fabs(omega_I_measured - 0.0890) / 0.0890;
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "QNM damp ω_I=%.4f within 50%% of 0.0890 (err=%.1f%%)",
                 omega_I_measured, 100*err);
        check(err < 0.50, msg);
    }

    check(min_alpha < 0.5, "Lapse collapsed (trumpet gauge)");
    check(min_alpha > 0.05, "Lapse stable (no crash)");
    check(ham_final < 1.0, "Ham L2 bounded");
    check(isfinite(ham_final), "No NaN/Inf");

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    /* Write CSV */
    FILE *fp = fopen("build/qnm_psi4.csv", "w");
    if (fp) {
        fprintf(fp, "t,re_psi4_20,im_psi4_20,amp_psi4_20\n");
        for (int i = 0; i < n_samples; i++) {
            double amp = sqrt(re_arr[i]*re_arr[i] + im_arr[i]*im_arr[i]);
            fprintf(fp, "%.6f,%.10e,%.10e,%.10e\n",
                    t_arr[i], re_arr[i], im_arr[i], amp);
        }
        fclose(fp);
        printf("  CSV: build/qnm_psi4.csv\n");
    }

    free(t_arr); free(re_arr); free(im_arr);
    psi4_free(ws);
    mesh_free(m);
    backend_cleanup();
    return n_fail > 0 ? 1 : 0;
}
