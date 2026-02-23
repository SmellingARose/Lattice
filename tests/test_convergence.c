/*
 * Lattice — 3D Numerical Relativity
 * 4th-order convergence test.
 *
 * Run single Schwarzschild puncture (M=1) at 3 resolutions with the same
 * domain (L=64). Evolve to T=1M. Measure Hamiltonian constraint L2 in an
 * annular region 5M < r < 25M (away from puncture singularity and boundary).
 *
 * The constraint should be zero for an exact solution, so the measured L2
 * is pure truncation error. For 4th-order FD + 4th-order RK4, constraint
 * error should scale as dx^4, giving a ratio of 2^4 = 16 between successive
 * resolution doublings.
 *
 * Pass criteria: convergence order > 3.5 for both refinement steps.
 *
 * Uses CK45 low-storage integrator (3 memory blocks).
 * Peak memory: ~1.5 GB (N=128 run).
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/puncture.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/boundary/sommerfeld.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <math.h>

/* Hamiltonian constraint L2 in annular region r_min < r < r_max.
 * Excludes puncture singularity and boundary ghost effects. */
static double constraint_l2_annular(const grid_t *g,
                                     double r_min, double r_max)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;
    double sum = 0.0;
    int count = 0;

    for (int k = lo; k < hi; k++) {
        for (int j = lo; j < hi; j++) {
            for (int i = lo; i < hi; i++) {
                double x = COORD(g, i);
                double y = COORD(g, j);
                double z = COORD(g, k);
                double r = sqrt(x*x + y*y + z*z);

                if (r >= r_min && r <= r_max) {
                    double H = compute_hamiltonian_at(
                        (const double *const *)g->fields, g, i, j, k);
                    sum += H * H;
                    count++;
                }
            }
        }
    }

    return (count > 0) ? sqrt(sum / count) : 0.0;
}

int main(void)
{
    printf("=== 4th-Order Convergence Test ===\n");
    printf("  Single BH (M=1), L=64, T=1M, CK45 integrator\n");
    printf("  Constraint L2 measured in 5M < r < 25M\n\n");
    fflush(stdout);

    backend_init();

    int resolutions[] = {32, 64, 128};
    int n_res = 3;
    double L = 64.0;
    double T_final = 1.0;
    double r_min = 5.0;
    double r_max = 25.0;

    double ham_l2[3];
    double dx_vals[3];

    for (int res = 0; res < n_res; res++) {
        int N = resolutions[res];

        sim_params_t p = default_params();
        p.N     = N;
        p.L     = L;
        p.CFL   = 0.25;
        p.sigma = 0.3;
        p.dx    = p.L / p.N;
        p.dt    = p.CFL * p.dx;
        p.num_steps = (int)(T_final / p.dt + 0.5);

        grid_t *g = grid_alloc(p.N, p.L, p.rk_method);

        /* Recompute after possible N padding */
        p.N  = g->N;
        p.dx = g->dx;
        p.dt = p.CFL * p.dx;
        p.num_steps = (int)(T_final / p.dt + 0.5);
        dx_vals[res] = p.dx;

        printf("  N=%3d  dx=%.4f  steps=%d  ", g->N, g->dx, p.num_steps);
        fflush(stdout);

        /* Brill-Lindquist single puncture M=1 at origin */
        double mass = 1.0;
        double center[1][3] = {{0.0, 0.0, 0.0}};
        set_brill_lindquist(g, 1, &mass, center);

        /* Evolve to T_final */
        for (int step = 1; step <= p.num_steps; step++)
            rk4_step(g, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);

        ham_l2[res] = constraint_l2_annular(g, r_min, r_max);
        printf("Ham_L2 = %.6e\n", ham_l2[res]);
        fflush(stdout);

        grid_free(g);
    }

    /* Convergence analysis */
    printf("\n=== Convergence Analysis ===\n\n");

    double ratio_1 = ham_l2[0] / ham_l2[1];
    double ratio_2 = ham_l2[1] / ham_l2[2];

    /* order = log(ratio) / log(dx_coarse / dx_fine) */
    double order_1 = log(ratio_1) / log(dx_vals[0] / dx_vals[1]);
    double order_2 = log(ratio_2) / log(dx_vals[1] / dx_vals[2]);

    printf("  N=%d -> N=%d:  ratio = %.2f  order = %.2f  (expect 16.0 / 4.0)\n",
           resolutions[0], resolutions[1], ratio_1, order_1);
    printf("  N=%d -> N=%d:  ratio = %.2f  order = %.2f  (expect 16.0 / 4.0)\n",
           resolutions[1], resolutions[2], ratio_2, order_2);

    int passed = (order_1 > 3.5) && (order_2 > 3.5);
    printf("\n  %s (threshold: order > 3.5)\n", passed ? "PASSED" : "FAILED");

    backend_cleanup();
    return passed ? 0 : 1;
}
