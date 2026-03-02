/*
 * Lattice — 3D Numerical Relativity
 * Head-on binary black hole collision (Milestone 4).
 *
 * Two equal-mass Brill-Lindquist punctures (m1=m2=0.5, M_ADM ~ 1) on the
 * z-axis at z = ±5M (coordinate separation d=10M), no initial momentum.
 * They fall toward each other and merge.
 *
 * Setup follows Sperhake (2006), gr-qc/0606079, Table I (BL models):
 *   m1 = m2 = 0.5, d = 10M, Brill-Lindquist time-symmetric data.
 * We use a uniform grid (no AMR) at lower resolution.
 *
 * Grid: N=128, L=64, dx=0.5, CFL=0.25.
 * Evolve to T=50M (merger ~15-20M, then ringdown settles).
 *
 * Pass criteria:
 *   1. No crash (NaN/Inf)
 *   2. Ham L2 < 1.0
 *   3. Mom L2 < 1.0
 *
 * Uses CK45 integrator. Memory: ~1.3 GB.
 *
 * Ref: gr-qc/0606079 (Sperhake 2006, BL head-on models)
 * Ref: gr-qc/9309016 (Anninos et al. 1993, first head-on collision)
 * Ref: gr-qc/9703066 (Brandt-Brugmann puncture method)
 * Ref: gr-qc/0511048 (Campanelli et al., moving punctures)
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/puncture.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/amr/mesh.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <math.h>

/* ── helpers (all static, test-local) ──────────────────────────────── */

/* Global minimum of lapse and its (x,y,z) location. */
static double min_lapse(const grid_t *g,
                        double *min_x, double *min_y, double *min_z)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;
    double min_val = 1.0e30;
    *min_x = *min_y = *min_z = 0.0;

    for (int k = lo; k < hi; k++) {
        for (int j = lo; j < hi; j++) {
            for (int i = lo; i < hi; i++) {
                int idx = IDX(g, i, j, k);
                double a = g->fields[FIELD_LAPSE][idx];
                if (a < min_val) {
                    min_val = a;
                    *min_x = COORD(g, i);
                    *min_y = COORD(g, j);
                    *min_z = COORD(g, k);
                }
            }
        }
    }
    return min_val;
}

/*
 * Count lapse minima on the z-axis (x=y=0): how many local dips below
 * the given threshold.  Two dips = two separate BHs, one dip = merged.
 */
static int count_z_axis_minima(const grid_t *g, double threshold)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;
    int i0 = lo + g->N / 2;
    int j0 = lo + g->N / 2;

    int in_dip = 0;
    int count  = 0;

    for (int k = lo; k < hi; k++) {
        int idx = IDX(g, i0, j0, k);
        double a = g->fields[FIELD_LAPSE][idx];
        if (a < threshold && !in_dip) {
            count++;
            in_dip = 1;
        } else if (a >= threshold) {
            in_dip = 0;
        }
    }
    return count;
}

/*
 * BH separation: walk the z-axis (x=y=0) and find the two deepest local
 * minima of the lapse.  Return the z-distance between them.
 * When only one minimum exists, return 0.0 (merged).
 */
static double bh_separation(const grid_t *g)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;
    int i0 = lo + g->N / 2;
    int j0 = lo + g->N / 2;

    /* Collect z-axis lapse profile */
    int nz = hi - lo;
    double *alpha_z = (double *)__builtin_alloca(sizeof(double) * (size_t)nz);
    for (int k = lo; k < hi; k++)
        alpha_z[k - lo] = g->fields[FIELD_LAPSE][IDX(g, i0, j0, k)];

    /* Find the two deepest local minima (interior points only) */
    double best1 = 1.0e30, best2 = 1.0e30;   /* lapse values */
    int    loc1  = -1,     loc2  = -1;         /* k-indices    */

    for (int n = 1; n < nz - 1; n++) {
        if (alpha_z[n] <= alpha_z[n - 1] && alpha_z[n] <= alpha_z[n + 1]) {
            if (alpha_z[n] < best1) {
                best2 = best1; loc2 = loc1;
                best1 = alpha_z[n]; loc1 = n;
            } else if (alpha_z[n] < best2) {
                best2 = alpha_z[n]; loc2 = n;
            }
        }
    }

    if (loc1 < 0 || loc2 < 0)
        return 0.0;   /* zero or one minimum — merged */

    double z1 = COORD(g, lo + loc1);
    double z2 = COORD(g, lo + loc2);
    return fabs(z1 - z2);
}

/* NaN/Inf scan over all interior field values. Returns 1 if all finite. */
static int check_finite(const grid_t *g)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;

    for (int f = 0; f < NUM_FIELDS; f++) {
        for (int k = lo; k < hi; k++) {
            for (int j = lo; j < hi; j++) {
                for (int i = lo; i < hi; i++) {
                    int idx = IDX(g, i, j, k);
                    double v = g->fields[f][idx];
                    if (!isfinite(v)) return 0;
                }
            }
        }
    }
    return 1;
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(void)
{
    /* ── parameters ─────────────────────────────────────────────── */
    sim_params_t p = default_params();
    p.N     = 128;
    p.L     = 64.0;
    p.CFL   = 0.25;
    p.sigma = 0.3;
    p.dx    = p.L / p.N;
    p.dt    = p.CFL * p.dx;

    double T_final = 50.0;
    p.num_steps = (int)(T_final / p.dt + 0.5);

    /* ── mesh ──────────────────────────────────────────────────── */
    backend_init();
    mesh_t *m = mesh_create_ex(p.N, p.L, p.rk_method, NUM_CCZ4_FIELDS);
    grid_t *g = m->blocks[0]->grid;

    /* mesh_create_ex may pad N — refresh derived quantities */
    p.N  = g->N;
    p.dx = g->dx;
    p.dt = p.CFL * p.dx;
    p.num_steps = (int)(T_final / p.dt + 0.5);

    /* ── initial data ───────────────────────────────────────────── */
    double masses[2]     = {0.5, 0.5};
    double centers[2][3] = {{0.0, 0.0, 5.0}, {0.0, 0.0, -5.0}};
    set_brill_lindquist(g, 2, masses, centers);

    /* ── banner ─────────────────────────────────────────────────── */
    printf("\n");
    printf("=== Head-On Binary BH Collision ===\n");
    printf("  m1=m2=0.5, d=10M, Brill-Lindquist (gr-qc/0606079)\n");
    printf("  N=%d, dx=%.3f, dt=%.3f, T=%.0fM (%d steps)\n\n",
           g->N, g->dx, p.dt, T_final, p.num_steps);
    printf(" step     t/M   lapse_min   z_min    sep/M  #dip    Ham_L2     Mom_L2\n");
    printf("--------------------------------------------------------------------------\n");
    fflush(stdout);

    /* ── initial diagnostics (step 0) ───────────────────────────── */
    double mx, my, mz;
    double ml     = min_lapse(g, &mx, &my, &mz);
    int    ndip   = count_z_axis_minima(g, 0.8);
    double sep    = (ndip >= 2) ? bh_separation(g) : 0.0;
    double ham    = compute_constraint_l2(g);
    double mom    = compute_momentum_l2(g);

    printf("%5d  %7.2f    %7.4f  %+6.2f  %7.2f    %d   %10.3e  %10.3e\n",
           0, 0.0, ml, mz, sep, ndip, ham, mom);
    fflush(stdout);

    int    ndip_init = ndip;
    double ham_peak  = ham;
    double mom_peak  = mom;
    int    crashed   = 0;

    /* ── evolution loop ─────────────────────────────────────────── */
    p.time = 0.0;
    for (int step = 1; step <= p.num_steps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;

        /* diagnostics every step */
        if (!check_finite(g)) {
            printf("  *** CRASH: NaN/Inf detected at step %d (t=%.2fM) ***\n",
                   step, step * p.dt);
            fflush(stdout);
            crashed = 1;
            break;
        }

        ml   = min_lapse(g, &mx, &my, &mz);
        ndip = count_z_axis_minima(g, 0.8);
        sep  = (ndip >= 2) ? bh_separation(g) : 0.0;
        ham  = mesh_constraint_l2(m);
        mom  = mesh_momentum_l2(m);

        if (ham > ham_peak) ham_peak = ham;
        if (mom > mom_peak) mom_peak = mom;

        double t = step * p.dt;
        printf("%5d  %7.2f    %7.4f  %+6.2f  %7.2f    %d   %10.3e  %10.3e\n",
               step, t, ml, mz, sep, ndip, ham, mom);
        fflush(stdout);
    }

    /* ── summary ────────────────────────────────────────────────── */
    printf("==========================================================================\n");

    if (crashed) {
        printf("  Fields finite:  NO\n");
        printf("\n  FAILED\n");
        printf("==========================================================================\n\n");
        mesh_free(m);
        backend_cleanup();
        return 1;
    }

    int finite_ok = check_finite(g);
    int ham_ok    = (ham_peak < 1.0);
    int mom_ok    = (mom_peak < 1.0);
    int merged    = (ndip <= 1);

    printf("  Fields finite:  %s\n", finite_ok ? "YES" : "NO");
    printf("  Ham bounded:    %s  (peak %.3e, limit 1.0)\n",
           ham_ok ? "YES" : "NO", ham_peak);
    printf("  Mom bounded:    %s  (peak %.3e, limit 1.0)\n",
           mom_ok ? "YES" : "NO", mom_peak);
    printf("  BHs merged:     %s  (z-axis minima: %d -> %d)\n",
           merged ? "YES" : "NO", ndip_init, ndip);

    int passed = finite_ok && ham_ok && mom_ok;
    printf("\n  %s\n", passed ? "PASSED" : "FAILED");
    printf("==========================================================================\n\n");

    mesh_free(m);
    backend_cleanup();
    return passed ? 0 : 1;
}
