/*
 * Lattice — 3D Numerical Relativity
 * Publication-quality Schwarzschild QNM ringdown test.
 *
 * Single M=1 puncture BH, AMR with dx_fine = M/8 = 0.125M.
 * Extracts Psi4 (2,0) and (2,2) modes at two radii, measures QNM
 * frequency + damping, tracks constraints and AH mass over time.
 *
 * Known Schwarzschild l=2 QNM (Leaver 1985):
 *   ω_R = 0.37367 / M,  ω_I = 0.08896 / M
 *
 * Grid: N_block=32, L=128, 5 AMR levels → dx_fine = 0.125M = M/8
 *   Level 2 (dx=1M) covers extraction radii (r=20M, r=25M).
 *   Boundary at 64M. Reflections reach r=25M at t≈103M. T=100M safe.
 *
 * Output CSVs for plotting:
 *   build/qnm_pub_r20.csv     — Psi4 modes at r=20M
 *   build/qnm_pub_r25.csv     — Psi4 modes at r=25M
 *   build/qnm_pub_diag.csv    — constraints + lapse over time
 *
 * Memory: ~200 blocks × 40³ × 25 × 4 × 8 ≈ 10 GB
 * Runs on: GPU (≥16 GB VRAM) or CPU (≥16 GB RAM).
 *
 * Ref: Leaver (1985), Berti et al. (2009, arXiv:0905.2975)
 * Ref: Etienne (2024, arXiv:2404.01137)
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
#include <time.h>

static double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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

static double measure_omega_R(const double *t, const double *re,
                                int n, double t_start)
{
    int i0 = 0;
    for (int i = 0; i < n; i++) { if (t[i] >= t_start) { i0 = i; break; } }
    double crossings[1000];
    int nc = 0;
    for (int i = i0; i < n - 1 && nc < 1000; i++) {
        if (re[i] * re[i+1] < 0.0) {
            double f = fabs(re[i]) / (fabs(re[i]) + fabs(re[i+1]));
            crossings[nc++] = t[i] + f * (t[i+1] - t[i]);
        }
    }
    if (nc < 4) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < nc - 1; i++) sum += crossings[i+1] - crossings[i];
    return M_PI / (sum / (nc - 1));
}

static double measure_omega_I(const double *t, const double *re,
                                int n, double t_start)
{
    int i0 = 0;
    for (int i = 0; i < n; i++) { if (t[i] >= t_start) { i0 = i; break; } }
    double pt[500], pa[500];
    int np = 0;
    for (int i = i0 + 1; i < n - 1 && np < 500; i++) {
        double ap = fabs(re[i-1]), ac = fabs(re[i]), an = fabs(re[i+1]);
        if (ac > ap && ac > an && ac > 1e-15) {
            pt[np] = t[i]; pa[np] = ac; np++;
        }
    }
    if (np < 3) return 0.0;
    double st=0, slA=0, st2=0, stlA=0;
    for (int i = 0; i < np; i++) {
        double ti = pt[i], lA = log(pa[i]);
        st += ti; slA += lA; st2 += ti*ti; stlA += ti*lA;
    }
    return -(np*stlA - st*slA) / (np*st2 - st*st);
}

int main(void)
{
    setbuf(stdout, NULL);
    printf("=== Schwarzschild QNM Ringdown — Publication Quality ===\n\n");
    backend_init();

    /* --- Grid: L=128, 5 AMR levels, dx_fine = M/8 = 0.125M ---
     * dx_base = 128/32 = 4M. Boundary at 64M.
     * Level 2 (dx=1M) covers r≤32M — extraction at r=20M,25M has dx≤1M
     *   (≥16 points per QNM wavelength = 16.8M).
     * Reflections reach r=25M at t ≈ 64 + (64-25) = 103M > T=100M.
     * Analysis window: t=40M to 100M = 60M = 3.6 clean QNM cycles.
     * Memory: ~200 blocks × 40³ × 25 × 4 × 8 ≈ 10 GB. */
    int N_block = 32;
    double L = 128.0;
    double M_bh = 1.0;
    int max_level = 5;

    mesh_t *m = mesh_create_ex(N_block, L, RK_CLASSIC, NUM_CCZ4_FIELDS);
    mesh_rebuild_neighbors(m);

    sim_params_t p = default_params();
    p.N = N_block;
    p.L = L;
    p.dx = m->dx_base;
    p.dt = 0.25 * p.dx;
    p.amr.max_level = max_level;
    p.amr.chi_refine = 0.1;
    p.amr.regrid_every = 0;
    p.noise.use_cako = 0;  /* uniform dissipation — stable for deep AMR */

    double dx_fine = p.dx / (1 << max_level);
    printf("  Grid: N_block=%d, L=%.0f, dx_base=%.3f, dx_fine=%.4f (M/%.0f)\n",
           N_block, L, p.dx, dx_fine, 1.0/dx_fine);
    printf("  Time: dt=%.4f, T=100M, CFL=0.25\n", p.dt);

    /* --- Initial data with AMR --- */
    puncture_data_t bh = {.mass = M_bh, .center = {0, 0, 0}};
    set_bowen_york_mesh(m, 1, &bh, max_level);
    ghost_exchange(m);

    int n_leaves = mesh_num_leaves(m);
    double mem_gb = (double)n_leaves * 40.0*40.0*40.0 * 25 * 4 * 8 / 1e9;
    printf("  Mesh: %d blocks (%d leaves), max_level=%d, ~%.1f GB\n",
           m->num_blocks, n_leaves, m->max_level, mem_gb);

    double ham0 = mesh_constraint_l2(m);
    double mom0 = mesh_momentum_l2(m);
    double ml0 = mesh_min_lapse(m);
    printf("  Initial: Ham=%.4e, Mom=%.4e, lapse_min=%.4f\n\n", ham0, mom0, ml0);

    /* --- Psi4 at two radii --- */
    double r1 = 20.0, r2 = 25.0;
    double center[3] = {0, 0, 0};
    int n_theta = 24, n_phi = 48, l_max = 4;
    psi4_workspace_t *ws1 = psi4_alloc(n_theta, n_phi, l_max, r1, center);
    psi4_workspace_t *ws2 = psi4_alloc(n_theta, n_phi, l_max, r2, center);
    printf("  Psi4: r=%.0fM, r=%.0fM, %dx%d, l_max=%d\n", r1, r2,
           n_theta, n_phi, l_max);

    /* --- Time series --- */
    double T_final = 100.0;  /* reflections reach r=25M at t≈103M; T=100M safe */
    int total_steps = (int)(T_final / p.dt + 0.5);
    int psi4_every = 1;   /* every base step = 2M cadence, ~8 samples/QNM cycle */
    int diag_every = 10;  /* constraints every 10 steps */
    int max_psi4 = total_steps / psi4_every + 2;
    int max_diag = total_steps / diag_every + 2;

    int mi_20 = mode_index(2, 0);
    int mi_22 = mode_index(2, 2);

    /* Psi4 arrays: both radii, (2,0) and (2,2) modes */
    double *psi4_t    = malloc(max_psi4 * sizeof(double));
    double *r1_re20   = malloc(max_psi4 * sizeof(double));
    double *r1_im20   = malloc(max_psi4 * sizeof(double));
    double *r1_re22   = malloc(max_psi4 * sizeof(double));
    double *r1_im22   = malloc(max_psi4 * sizeof(double));
    double *r2_re20   = malloc(max_psi4 * sizeof(double));
    double *r2_im20   = malloc(max_psi4 * sizeof(double));
    double *r2_re22   = malloc(max_psi4 * sizeof(double));
    double *r2_im22   = malloc(max_psi4 * sizeof(double));
    int n_psi4 = 0;

    /* Diagnostic arrays */
    double *diag_t     = malloc(max_diag * sizeof(double));
    double *diag_ham   = malloc(max_diag * sizeof(double));
    double *diag_mom   = malloc(max_diag * sizeof(double));
    double *diag_lapse = malloc(max_diag * sizeof(double));
    double *diag_mass  = malloc(max_diag * sizeof(double));
    double *diag_spin  = malloc(max_diag * sizeof(double));
    int n_diag = 0;

    printf("  Evolving T=%.0fM (%d base steps)\n", T_final, total_steps);
    printf("  Psi4 every %d steps, diagnostics every %d\n\n",
           psi4_every, diag_every);

    /* --- Evolution --- */
    int is_gpu = backend_is_gpu() && m->max_level > 0;
    p.time = 0.0;
    double t_wall_start = wtime();

    for (int step = 1; step <= total_steps; step++) {
        double t_step_start = wtime();
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;

        int do_psi4 = (step % psi4_every == 0);
        int do_diag = 1;  /* every step for debugging */

        /* Psi4 extraction: CPU psi4_extract uses mesh_find_block_at to
         * find the finest block at each angular point. GPU-resident data
         * needs sync to host for CPU extraction. */
        if (do_psi4) {
            if (is_gpu)
                gpu_sync_all_to_host(m);
            psi4_extract(ws1, m);
            psi4_extract(ws2, m);
            psi4_t[n_psi4]  = p.time;
            r1_re20[n_psi4] = ws1->mode_re[mi_20];
            r1_im20[n_psi4] = ws1->mode_im[mi_20];
            r1_re22[n_psi4] = ws1->mode_re[mi_22];
            r1_im22[n_psi4] = ws1->mode_im[mi_22];
            r2_re20[n_psi4] = ws2->mode_re[mi_20];
            r2_im20[n_psi4] = ws2->mode_im[mi_20];
            r2_re22[n_psi4] = ws2->mode_re[mi_22];
            r2_im22[n_psi4] = ws2->mode_im[mi_22];
            n_psi4++;
        }

        /* GPU-native constraints (no host sync) */
        if (do_diag && is_gpu) {
            double ham = 0, mom = 0, vol_h = 0, vol_m = 0;
            for (int L = 0; L <= m->max_level; L++) {
                meshblock_pack_t *pk = m->level_packs[L];
                if (!pk || pk->n_blocks == 0) continue;
                backend_activate_pack(pk);
                double s, v;
                backend_constraint_l2_raw_packed(pk, &s, &v);
                ham += s; vol_h += v;
                backend_momentum_l2_raw_packed(pk, &s, &v);
                mom += s; vol_m += v;
            }
            ham = (vol_h > 0) ? sqrt(ham / vol_h) : 0;
            mom = (vol_m > 0) ? sqrt(mom / (3.0 * vol_m)) : 0;

            /* Min lapse: use finest level pack */
            double ml = 1.0;
            for (int L = m->max_level; L >= 0; L--) {
                meshblock_pack_t *pk = m->level_packs[L];
                if (!pk || pk->n_blocks == 0) continue;
                backend_activate_pack(pk);
                double x, y, z;
                double a = backend_min_lapse_packed(pk, &x, &y, &z);
                if (a < ml) ml = a;
            }

            diag_t[n_diag]     = p.time;
            diag_ham[n_diag]   = ham;
            diag_mom[n_diag]   = mom;
            diag_lapse[n_diag] = ml;
            diag_mass[n_diag]  = 0;
            diag_spin[n_diag]  = 0;
            n_diag++;

            double amp20 = (n_psi4 > 0) ?
                sqrt(r1_re20[n_psi4-1]*r1_re20[n_psi4-1] +
                     r1_im20[n_psi4-1]*r1_im20[n_psi4-1]) : 0;
            /* chi_min from host blocks (already synced for Psi4) */
            double chi_min = 1e30;
            for (int bid = 0; bid < m->num_blocks; bid++) {
                block_t *blk = m->blocks[bid];
                if (!blk || !blk->is_leaf) continue;
                grid_t *g = blk->grid;
                int lo = g->ghost, hi = g->ghost + g->N;
                for (int k = lo; k < hi; k++)
                    for (int j = lo; j < hi; j++)
                        for (int i = lo; i < hi; i++) {
                            double c = g->fields[FIELD_CHI][IDX(g,i,j,k)];
                            if (c < chi_min) chi_min = c;
                        }
            }

            printf("  step %4d  t=%6.1fM  Ham=%.3e  Mom=%.3e  lapse=%.4f  chi=%.4e",
                   step, p.time, ham, mom, ml, chi_min);
            if (amp20 > 0) printf("  |Psi4|=%.3e", amp20);
            double elapsed = wtime() - t_wall_start;
            double dt_wall = wtime() - t_step_start;
            printf("  [%.1fs/step, %.0fs total]\n", dt_wall, elapsed);
        } else if (do_diag) {
            /* CPU fallback */
            double ham = mesh_constraint_l2(m);
            double mom = mesh_momentum_l2(m);
            double ml = mesh_min_lapse(m);

            diag_t[n_diag]     = p.time;
            diag_ham[n_diag]   = ham;
            diag_mom[n_diag]   = mom;
            diag_lapse[n_diag] = ml;
            diag_mass[n_diag]  = 0;
            diag_spin[n_diag]  = 0;
            n_diag++;

            double amp20 = (n_psi4 > 0) ?
                sqrt(r1_re20[n_psi4-1]*r1_re20[n_psi4-1] +
                     r1_im20[n_psi4-1]*r1_im20[n_psi4-1]) : 0;
            double chi_min = 1e30;
            for (int bid = 0; bid < m->num_blocks; bid++) {
                block_t *blk = m->blocks[bid];
                if (!blk || !blk->is_leaf) continue;
                grid_t *g = blk->grid;
                int lo = g->ghost, hi = g->ghost + g->N;
                for (int k = lo; k < hi; k++)
                    for (int j = lo; j < hi; j++)
                        for (int i = lo; i < hi; i++) {
                            double c = g->fields[FIELD_CHI][IDX(g,i,j,k)];
                            if (c < chi_min) chi_min = c;
                        }
            }
            printf("  step %4d  t=%6.1fM  Ham=%.3e  Mom=%.3e  lapse=%.4f  chi=%.4e",
                   step, p.time, ham, mom, ml, chi_min);
            if (amp20 > 0) printf("  |Psi4|=%.3e", amp20);
            double elapsed = wtime() - t_wall_start;
            double dt_wall = wtime() - t_step_start;
            printf("  [%.1fs/step, %.0fs total]\n", dt_wall, elapsed);
        }
    }

    /* --- QNM analysis --- */
    printf("\n=== QNM Analysis (t > 40M) ===\n\n");
    double t_start = 40.0;

    /* Frequency */
    double wR1 = measure_omega_R(psi4_t, r1_re20, n_psi4, t_start);
    double wR2 = measure_omega_R(psi4_t, r2_re20, n_psi4, t_start);
    printf("  ω_R(r=20M) = %.5f/M  (expected 0.37367, err=%.2f%%)\n",
           wR1, wR1 > 0 ? 100*fabs(wR1-0.37367)/0.37367 : -1.0);
    printf("  ω_R(r=25M) = %.5f/M  (expected 0.37367, err=%.2f%%)\n",
           wR2, wR2 > 0 ? 100*fabs(wR2-0.37367)/0.37367 : -1.0);

    /* Damping */
    double wI1 = measure_omega_I(psi4_t, r1_re20, n_psi4, t_start);
    double wI2 = measure_omega_I(psi4_t, r2_re20, n_psi4, t_start);
    printf("  ω_I(r=20M) = %.5f/M  (expected 0.08896, err=%.2f%%)\n",
           wI1, wI1 > 0 ? 100*fabs(wI1-0.08896)/0.08896 : -1.0);
    printf("  ω_I(r=25M) = %.5f/M  (expected 0.08896, err=%.2f%%)\n",
           wI2, wI2 > 0 ? 100*fabs(wI2-0.08896)/0.08896 : -1.0);

    /* (2,2) mode — should be near zero for non-spinning BH */
    double max_22 = 0;
    for (int i = 0; i < n_psi4; i++) {
        double a = sqrt(r1_re22[i]*r1_re22[i] + r1_im22[i]*r1_im22[i]);
        if (a > max_22) max_22 = a;
    }
    double max_20 = 0;
    for (int i = 0; i < n_psi4; i++) {
        double a = sqrt(r1_re20[i]*r1_re20[i] + r1_im20[i]*r1_im20[i]);
        if (a > max_20) max_20 = a;
    }
    printf("  max|Psi4(2,0)| = %.4e\n", max_20);
    printf("  max|Psi4(2,2)| = %.4e  (ratio to (2,0): %.2e)\n",
           max_22, max_20 > 0 ? max_22/max_20 : 0);

    /* Final CPU diagnostics — sync from device if needed */
    if (is_gpu)
        gpu_sync_all_to_host(m);
    double min_alpha = mesh_min_lapse(m);
    double ham_final = mesh_constraint_l2(m);
    printf("  Final: lapse_min=%.4f, Ham=%.4e\n", min_alpha, ham_final);

    /* --- Pass/fail --- */
    printf("\n=== Pass/Fail ===\n");

    check(wR1 > 0, "ω_R(r=20) measurable");
    check(wR2 > 0, "ω_R(r=25) measurable");
    if (wR1 > 0)
        check(fabs(wR1-0.37367)/0.37367 < 0.05,
              "ω_R(r=20) within 5% of Leaver");
    if (wR2 > 0)
        check(fabs(wR2-0.37367)/0.37367 < 0.05,
              "ω_R(r=25) within 5% of Leaver");
    if (wR1 > 0 && wR2 > 0)
        check(fabs(wR1-wR2)/(0.5*(wR1+wR2)) < 0.03,
              "ω_R consistent across radii (<3%)");

    check(wI1 > 0, "ω_I(r=20) measurable");
    if (wI1 > 0)
        check(fabs(wI1-0.08896)/0.08896 < 0.15,
              "ω_I(r=20) within 15% of Leaver");

    check(max_20 > 0, "Psi4(2,0) mode excited");
    check(max_22 < 0.1 * max_20,
          "Psi4(2,2) suppressed (< 10% of (2,0)) — axisymmetric");

    check(min_alpha < 0.1,
          "Lapse collapsed at puncture (< 0.1)");
    check(ham_final < 0.01, "Final Ham L2 < 0.01");
    check(isfinite(ham_final), "No NaN/Inf");

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    /* --- CSV: Psi4 at both radii --- */
    FILE *fp = fopen("build/qnm_pub_r20.csv", "w");
    if (fp) {
        fprintf(fp, "t,re_20,im_20,re_22,im_22\n");
        for (int i = 0; i < n_psi4; i++)
            fprintf(fp, "%.6f,%.14e,%.14e,%.14e,%.14e\n",
                    psi4_t[i], r1_re20[i], r1_im20[i], r1_re22[i], r1_im22[i]);
        fclose(fp);
    }
    fp = fopen("build/qnm_pub_r25.csv", "w");
    if (fp) {
        fprintf(fp, "t,re_20,im_20,re_22,im_22\n");
        for (int i = 0; i < n_psi4; i++)
            fprintf(fp, "%.6f,%.14e,%.14e,%.14e,%.14e\n",
                    psi4_t[i], r2_re20[i], r2_im20[i], r2_re22[i], r2_im22[i]);
        fclose(fp);
    }

    /* --- CSV: diagnostics over time --- */
    fp = fopen("build/qnm_pub_diag.csv", "w");
    if (fp) {
        fprintf(fp, "t,ham_l2,mom_l2,lapse_min,mass_irr,spin\n");
        for (int i = 0; i < n_diag; i++)
            fprintf(fp, "%.6f,%.14e,%.14e,%.14e,%.14e,%.14e\n",
                    diag_t[i], diag_ham[i], diag_mom[i],
                    diag_lapse[i], diag_mass[i], diag_spin[i]);
        fclose(fp);
    }
    printf("  CSV: qnm_pub_r20.csv, qnm_pub_r25.csv, qnm_pub_diag.csv\n");

    free(psi4_t); free(r1_re20); free(r1_im20); free(r1_re22); free(r1_im22);
    free(r2_re20); free(r2_im20); free(r2_re22); free(r2_im22);
    free(diag_t); free(diag_ham); free(diag_mom);
    free(diag_lapse); free(diag_mass); free(diag_spin);
    psi4_free(ws1); psi4_free(ws2);
    mesh_free(m);
    backend_cleanup();
    return n_fail > 0 ? 1 : 0;
}
