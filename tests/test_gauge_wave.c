/*
 * Lattice — Gauge Wave Test (Apples with Apples)
 *
 * Exact analytical solution: ds^2 = -H dt^2 + H dx^2 + dy^2 + dz^2
 *   H = 1 - A sin(2pi(x-t)/d)
 *
 * Validates: FD accuracy, time integrator, convergence order, constraint
 * damping, long-term stability. Measures pointwise L2 error vs exact.
 *
 * Gauge: harmonic slicing (f(alpha)=1), zero shift.
 * CCZ4 with kappa1=1/d, kappa3=0.5.
 *
 * Ref: gr-qc/0305023 (Alcubierre — original AwA definition)
 * Ref: arXiv:1106.2254 (Alic — CCZ4 gauge wave, 1000+ crossing times)
 * Ref: arXiv:1810.12346 (Hilditch — CCZ4 conformal decomposition)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/amr/mesh.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int n_pass = 0, n_fail = 0;

static void check(int cond, const char *msg)
{
    if (cond) { printf("  [PASS] %s\n", msg); n_pass++; }
    else      { printf("  [FAIL] %s\n", msg); n_fail++; }
}

/*
 * Set gauge wave initial data on a grid.
 * H = 1 - A sin(2pi(x-t)/d), wave along x-axis.
 *
 * CCZ4 variables from Hilditch et al. 2018 (arXiv:1810.12346):
 *   chi    = H^(-1/3)
 *   h_xx   = H^(2/3),  h_yy = h_zz = H^(-1/3)
 *   K      = -H_c / H^(3/2)
 *   A_xx   = -(2/3) H_c H^(-5/6)
 *   A_yy   = A_zz = (1/3) H_c H^(-11/6)
 *   Gamma^x = -(4/3) H_c H^(-5/3)
 *   lapse  = sqrt(H)
 *   Theta = shift = B = 0
 *
 * where H_c = A pi/d cos(2pi(x-t)/d).
 */
static void set_gauge_wave(grid_t *g, double A, double d, double t, double L)
{
    int lo = g->ghost, hi = g->ghost + g->N;
    int Nt = g->Ntotal;

    for (int k = lo; k < hi; k++)
        for (int j = lo; j < hi; j++)
            for (int i = lo; i < hi; i++) {
                int idx = i + j * Nt + k * Nt * Nt;
                double x = (i - g->ghost + 0.5) * g->dx - L / 2.0;

                double u = 2.0 * M_PI * (x - t) / d;
                double H_s = A * sin(u);
                double H_c = A * M_PI / d * cos(u);
                double H = 1.0 - H_s;

                /* Conformal factor */
                g->fields[FIELD_CHI][idx] = pow(H, -1.0/3.0);

                /* Conformal metric (det = 1) */
                g->fields[FIELD_H11][idx] = pow(H, 2.0/3.0);
                g->fields[FIELD_H12][idx] = 0.0;
                g->fields[FIELD_H13][idx] = 0.0;
                g->fields[FIELD_H22][idx] = pow(H, -1.0/3.0);
                g->fields[FIELD_H23][idx] = 0.0;
                g->fields[FIELD_H33][idx] = pow(H, -1.0/3.0);

                /* Trace of extrinsic curvature */
                g->fields[FIELD_K][idx] = -H_c / (H * sqrt(H));

                /* Traceless conformal extrinsic curvature */
                g->fields[FIELD_A11][idx] = -(2.0/3.0) * H_c * pow(H, -5.0/6.0);
                g->fields[FIELD_A12][idx] = 0.0;
                g->fields[FIELD_A13][idx] = 0.0;
                g->fields[FIELD_A22][idx] = (1.0/3.0) * H_c * pow(H, -11.0/6.0);
                g->fields[FIELD_A23][idx] = 0.0;
                g->fields[FIELD_A33][idx] = (1.0/3.0) * H_c * pow(H, -11.0/6.0);

                /* Conformal connections */
                g->fields[FIELD_GAMMA1][idx] = -(4.0/3.0) * H_c * pow(H, -5.0/3.0);
                g->fields[FIELD_GAMMA2][idx] = 0.0;
                g->fields[FIELD_GAMMA3][idx] = 0.0;

                /* CCZ4 constraint scalar */
                g->fields[FIELD_THETA][idx] = 0.0;

                /* Gauge */
                g->fields[FIELD_LAPSE][idx] = sqrt(H);
                g->fields[FIELD_SHIFT1][idx] = 0.0;
                g->fields[FIELD_SHIFT2][idx] = 0.0;
                g->fields[FIELD_SHIFT3][idx] = 0.0;
                g->fields[FIELD_B1][idx] = 0.0;
                g->fields[FIELD_B2][idx] = 0.0;
                g->fields[FIELD_B3][idx] = 0.0;
            }

    /* Fill ghost zones by copying nearest interior value */
    for (int f = 0; f < g->n_fields; f++) {
        for (int k = 0; k < Nt; k++)
            for (int j = 0; j < Nt; j++) {
                for (int ig = 0; ig < g->ghost; ig++) {
                    g->fields[f][ig + j*Nt + k*Nt*Nt] =
                        g->fields[f][g->ghost + j*Nt + k*Nt*Nt];
                    g->fields[f][(g->ghost+g->N+ig) + j*Nt + k*Nt*Nt] =
                        g->fields[f][(g->ghost+g->N-1) + j*Nt + k*Nt*Nt];
                }
            }
    }
}

/*
 * Compute L2 error of chi against exact gauge wave solution.
 */
static double chi_l2_error(const grid_t *g, double A, double d, double t, double L)
{
    int lo = g->ghost, hi = g->ghost + g->N;
    int Nt = g->Ntotal;
    double err2 = 0.0;
    int count = 0;

    for (int k = lo; k < hi; k++)
        for (int j = lo; j < hi; j++)
            for (int i = lo; i < hi; i++) {
                int idx = i + j * Nt + k * Nt * Nt;
                double x = (i - g->ghost + 0.5) * g->dx - L / 2.0;
                double u = 2.0 * M_PI * (x - t) / d;
                double H = 1.0 - A * sin(u);
                double chi_exact = pow(H, -1.0/3.0);
                double chi_num = g->fields[FIELD_CHI][idx];
                double e = chi_num - chi_exact;
                err2 += e * e;
                count++;
            }

    return sqrt(err2 / count);
}

int main(void)
{
    setbuf(stdout, NULL);
    printf("=== Gauge Wave Test (Apples with Apples) ===\n\n");
    backend_init();

    double A = 0.1;      /* amplitude */
    double d = 1.0;      /* wavelength */
    double L = 10.0;     /* domain size (10 wavelengths, Sommerfeld absorbs edges) */
    double T_cross = 5.0; /* evolve for 5 crossing times */

    printf("  A=%.2f, d=%.1f, L=%.1f\n", A, d, L);

    /* --- Three resolutions for convergence (must be multiples of 16) --- */
    int resolutions[] = {32, 64, 128};
    double errors[3];

    for (int r = 0; r < 3; r++) {
        int N = resolutions[r];
        mesh_t *m = mesh_create_ex(N, L, RK_CLASSIC, NUM_CCZ4_FIELDS);
        grid_t *g = m->blocks[0]->grid;

        sim_params_t p = default_params();
        p.N = g->N;
        p.L = L;
        p.dx = g->dx;
        p.dt = 0.25 * g->dx;

        /* Harmonic slicing: f(alpha) = 1 => lapse_coeff=1, lapse_power=2 */
        p.gauge.lapse_coeff = 1.0;
        p.gauge.lapse_power = 2.0;

        /* CCZ4 damping */
        p.ccz4.kappa1 = 1.0;   /* = 1/d */
        p.ccz4.kappa2 = 0.0;
        p.ccz4.kappa3 = 0.5;

        /* Disable puncture-specific techniques */
        p.noise.use_cako = 0;
        p.noise.use_cahd = 0;
        p.noise.use_ssl  = 0;
        p.noise.use_per_field_sigma = 0;
        p.sigma = 0.3;

        /* Set initial data */
        set_gauge_wave(g, A, d, 0.0, L);

        double err0 = chi_l2_error(g, A, d, 0.0, L);
        printf("  N=%3d: initial L2(chi)=%.4e", N, err0);

        /* Evolve */
        int total_steps = (int)(T_cross * d / p.dt + 0.5);
        p.time = 0.0;
        for (int step = 0; step < total_steps; step++) {
            rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
            p.time += p.dt;
        }

        double err = chi_l2_error(g, A, d, p.time, L);
        errors[r] = err;
        double ham = compute_constraint_l2(g);
        printf(", final L2(chi)=%.4e, Ham=%.4e (t=%.2f)\n", err, ham, p.time);

        mesh_free(m);
    }

    /* --- Convergence order --- */
    printf("\n  Convergence:\n");
    if (errors[0] > 0 && errors[1] > 0) {
        double order_12 = log(errors[0] / errors[1]) / log(2.0);
        printf("    N=%d->%d: order = %.2f\n", resolutions[0], resolutions[1], order_12);
        check(order_12 > 3.5, "Convergence order > 3.5 (coarse->mid)");
    }
    if (errors[1] > 0 && errors[2] > 0) {
        double order_23 = log(errors[1] / errors[2]) / log(2.0);
        printf("    N=%d->%d: order = %.2f\n", resolutions[1], resolutions[2], order_23);
        check(order_23 > 3.5, "Convergence order > 3.5 (mid->fine)");
    }
    check(errors[2] < 1e-6, "Fine resolution L2 error < 1e-6");
    check(errors[2] < errors[0], "Error decreases with resolution");

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    backend_cleanup();
    return n_fail > 0 ? 1 : 0;
}
