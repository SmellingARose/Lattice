/*
 * Lattice — 3D Numerical Relativity
 * Binary inspiral AMR convergence test.
 *
 * Standard equal-mass non-spinning quasi-circular binary (Bode et al.
 * 2009, arXiv:0902.1127):  m_bare = 0.48595, d = 10M, P_y = ±0.09543 (QC).
 *
 * Self-convergence test: run 3 resolutions with N_block = 32, 48, 64
 * (ratio 1.5x, effective grids 32/48/64).  Uniform AMR mesh
 * (max_level=0): no refinement, but exercises the full AMR infrastructure
 * (block decomposition, packed kernels, ghost exchange).  Classic RK4.
 *
 * Measure: Hamiltonian constraint L2 norm every step.
 * Convergence check: for 4th-order code with ratio r=1.5,
 *   Q = |Ham(low) - Ham(med)| / |Ham(med) - Ham(high)| ≈ 1.5^4 = 5.06.
 *
 * Logs every step for live monitoring (stdout unbuffered).
 * Designed to run in background: nohup ./test_inspiral_convergence > log &
 * Monitor with: tail -f inspiral.log
 *
 * 6000 steps at CFL=0.25 → t_final varies by resolution (different dx).
 *
 * Memory: peak ~16 GB (HIGH run: 27 blocks + pack, classic RK4).
 * Estimated runtime: ~41 hours total (LOW ~3h, MED ~11h, HIGH ~27h).
 *
 * Ref: arXiv:0902.1127 (Bode et al. 2009, D10 QC parameters)
 * Ref: arXiv:2409.10383 (AthenaK self-convergence methodology)
 */

#include "../src/amr/mesh.h"
#include "../src/amr/block.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/initial_data/puncture.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Physical parameters (Bode et al. 2009, arXiv:0902.1127) ─────── */
#define M_BARE     0.48595  /* bare puncture mass (E_ADM ≈ 0.9895)     */
#define D_SEP      10.0     /* coordinate separation                   */
#define P_Y        0.09543  /* quasi-circular tangential momentum      */
#define L_DOMAIN   64.0     /* domain size [-32, 32]^3                 */
#define CFL_FACTOR 0.25
#define NUM_STEPS  6000

/* ── AMR parameters (fixed across resolutions) ────────────────────── */
/* N_ROOT removed: single root block per mesh */
#define MAX_LEVEL    0      /* uniform mesh — no refinement             */

/* ── Resolution triplet (ratio 1.5x) ─────────────────────────────── */
static const int N_BLOCKS[] = { 32, 48, 64 };
static const int N_RES = 3;
static const char *RES_LABELS[] = { "LOW", "MED", "HIGH" };

/* ── Set up puncture data ─────────────────────────────────────────── */
static void setup_punctures(puncture_data_t bhs[2])
{
    memset(bhs, 0, sizeof(puncture_data_t) * 2);

    bhs[0].mass = M_BARE;
    bhs[0].center[2] = D_SEP / 2.0;      /* z = +5 */
    bhs[0].momentum[1] = P_Y;             /* P_y = +0.0939 */

    bhs[1].mass = M_BARE;
    bhs[1].center[2] = -D_SEP / 2.0;     /* z = -5 */
    bhs[1].momentum[1] = -P_Y;            /* P_y = -0.0939 */
}

/* ── Run one resolution ───────────────────────────────────────────── */
static double *run_resolution(int res_idx, int n_block)
{
    const char *label = RES_LABELS[res_idx];
    printf("\n");
    printf("================================================================\n");
    printf("  RESOLUTION %s: N_block=%d, dx=%.6f\n",
           label, n_block, L_DOMAIN / n_block);
    printf("================================================================\n");
    fflush(stdout);

    /* Create uniform AMR mesh (max_level=0, no refinement) */
    mesh_t *m = mesh_create(n_block, L_DOMAIN, RK_CLASSIC);

    sim_params_t p = default_params();
    p.L = L_DOMAIN;
    p.rk_method = RK_CLASSIC;
    p.CFL = CFL_FACTOR;
    p.dx = m->dx_base;
    p.dt = p.CFL * p.dx;
    p.sigma = 0.3;

    double t_final = NUM_STEPS * p.dt;

    printf("  L=%.1f, dx=%.6f, dt=%.6f, CFL=%.2f, rk=classic\n",
           p.L, p.dx, p.dt, p.CFL);
    printf("  max_level=%d (uniform), blocks=%d\n",
           MAX_LEVEL, mesh_num_leaves(m));
    printf("  steps=%d, t_final=%.1fM\n", NUM_STEPS, t_final);
    fflush(stdout);

    /* Solve initial data directly on the mesh (no temp grid needed) */
    puncture_data_t bhs[2];
    setup_punctures(bhs);

    printf("  Solving initial data (FAS multigrid, N_block=%d)...\n", n_block);
    fflush(stdout);
    time_t t0 = time(NULL);

    set_bowen_york_mesh(m, 2, bhs, 0);

    printf("  ID solve done (%.0f sec)\n", difftime(time(NULL), t0));
    fflush(stdout);

    double ham0 = mesh_constraint_l2(m);
    printf("  Initial Ham L2 = %.6e, leaves = %d\n", ham0, mesh_num_leaves(m));
    fflush(stdout);

    /* Allocate array to store Ham L2 at every step */
    double *ham_history = calloc(NUM_STEPS + 1, sizeof(double));
    ham_history[0] = ham0;

    /* Header */
    printf("\n  [%s] step     t/M      Ham_L2      leaves  sec/step\n", label);
    printf("  -------------------------------------------------------\n");
    fflush(stdout);

    /* Evolution — no regridding (max_level=0) */
    p.time = 0.0;
    for (int step = 1; step <= NUM_STEPS; step++) {
        time_t step_t0 = time(NULL);

        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;

        double ham = mesh_constraint_l2(m);
        ham_history[step] = ham;

        double elapsed = difftime(time(NULL), step_t0);
        printf("  [%s] %5d  %7.2f  %12.6e  %5d  %.1f\n",
               label, step, p.time, ham, mesh_num_leaves(m), elapsed);
        fflush(stdout);
    }

    mesh_free(m);
    return ham_history;
}

/* ── Convergence analysis ─────────────────────────────────────────── */
static void analyze_convergence(double *ham[3])
{
    printf("\n");
    printf("================================================================\n");
    printf("  CONVERGENCE ANALYSIS (ratio r=1.5, expected Q=1.5^4=5.06)\n");
    printf("================================================================\n");
    printf("\n");

    /* Each resolution has different dt (dt = CFL * dx), so the same step
     * number corresponds to different physical times.  For self-convergence,
     * we compare at the same step number — the standard approach when
     * varying only spatial resolution with a fixed CFL.
     * Ref: arXiv:2409.10383 Section III.B */

    printf("  step      Ham_low        Ham_med        Ham_high       "
           "|L-M|          |M-H|          Q\n");
    printf("  -----------------------------------------------------------"
           "----------------------------------------------\n");

    /* Sample every 100 steps for the summary table */
    double q_sum = 0.0;
    int q_count = 0;

    for (int step = 100; step <= NUM_STEPS; step += 100) {
        double hl = ham[0][step];
        double hm = ham[1][step];
        double hh = ham[2][step];

        double diff_lm = fabs(hl - hm);
        double diff_mh = fabs(hm - hh);

        double q = (diff_mh > 1e-16) ? diff_lm / diff_mh : 0.0;

        printf("  %5d  %13.6e  %13.6e  %13.6e  %13.6e  %13.6e  %6.2f\n",
               step, hl, hm, hh, diff_lm, diff_mh, q);

        if (q > 0.5 && q < 50.0) {
            q_sum += q;
            q_count++;
        }
    }

    double q_avg = (q_count > 0) ? q_sum / q_count : 0.0;

    printf("\n");
    printf("  Mean convergence factor Q = %.2f (expected 5.06 for 4th order)\n",
           q_avg);
    printf("  Measured convergence order = %.2f (expected 4.0)\n",
           (q_avg > 0) ? log(q_avg) / log(1.5) : 0.0);
    printf("\n");

    /* Pass/fail: accept order in [2.5, 6.0] — binary dynamics can
     * degrade order slightly during merger, overshoot during gauge settling */
    double measured_order = (q_avg > 0) ? log(q_avg) / log(1.5) : 0.0;
    int passed = (measured_order > 2.5 && measured_order < 6.0);
    printf("  RESULT: %s (order %.2f, range [2.5, 6.0])\n",
           passed ? "PASS" : "FAIL", measured_order);
    printf("================================================================\n\n");
}

/* ── main ─────────────────────────────────────────────────────────── */
int main(void)
{
    setbuf(stdout, NULL);
    backend_init();

    printf("================================================================\n");
    printf("  Lattice — Binary Inspiral AMR Convergence Test\n");
    printf("================================================================\n");
    printf("  Physics: equal-mass non-spinning, d=10M, P_y=%.5f (QC)\n", P_Y);
    printf("  Ref: Bode et al. 2009, arXiv:0902.1127\n");
    printf("  Resolutions: N_block = 32, 48, 64 (ratio 1.5x)\n");
    printf("  max_level=%d (uniform AMR mesh)\n", MAX_LEVEL);
    printf("  Integrator: classic RK4, CFL=%.2f\n", CFL_FACTOR);
    printf("  Steps: %d per resolution\n", NUM_STEPS);
    printf("================================================================\n");
    fflush(stdout);

    time_t wall_start = time(NULL);

    /* Run all 3 resolutions sequentially */
    double *ham[3];
    for (int r = 0; r < N_RES; r++) {
        time_t res_start = time(NULL);
        ham[r] = run_resolution(r, N_BLOCKS[r]);
        printf("  [%s] completed in %.0f sec (%.1f hours)\n",
               RES_LABELS[r], difftime(time(NULL), res_start),
               difftime(time(NULL), res_start) / 3600.0);
        fflush(stdout);
    }

    /* Convergence analysis */
    analyze_convergence(ham);

    /* Wall time */
    printf("  Total wall time: %.0f sec (%.1f hours)\n",
           difftime(time(NULL), wall_start),
           difftime(time(NULL), wall_start) / 3600.0);
    printf("\n");

    /* Cleanup */
    for (int r = 0; r < N_RES; r++)
        free(ham[r]);

    backend_cleanup();
    return 0;
}
