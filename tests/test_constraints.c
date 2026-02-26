/*
 * Lattice — 3D Numerical Relativity
 * Test Hamiltonian and momentum constraints.
 *
 * 1. Flat spacetime: both H and M_i should be zero (machine precision)
 * 2. Single BH (evolved): both should be small and bounded
 * 3. Single BH (evolved): momentum should converge at same order as Hamiltonian
 *
 * Ref: GRChombo NewConstraints.impl.hpp:55-100
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

/* Constraint L2 in annular region, for both Ham and Mom */
static void constraints_annular(const grid_t *g, double r_min, double r_max,
                                 double *ham_l2, double *mom_l2)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;
    double ham_sum = 0.0, mom_sum = 0.0;
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
                    double mom[3];
                    compute_momentum_at(
                        (const double *const *)g->fields, g, i, j, k, mom);
                    ham_sum += H * H;
                    mom_sum += mom[0]*mom[0] + mom[1]*mom[1] + mom[2]*mom[2];
                    count++;
                }
            }
        }
    }

    *ham_l2 = (count > 0) ? sqrt(ham_sum / count) : 0.0;
    *mom_l2 = (count > 0) ? sqrt(mom_sum / (3 * count)) : 0.0;
}

int main(void)
{
    printf("=== Constraint Tests (Hamiltonian + Momentum) ===\n\n");
    fflush(stdout);

    backend_init();
    int all_passed = 1;

    /* ------ Test 1: Flat spacetime ------ */
    {
        printf("--- Test 1: Flat spacetime (N=32) ---\n");
        fflush(stdout);

        sim_params_t p = default_params();
        p.N = 32; p.L = 10.0;
        p.dx = p.L / p.N; p.dt = 0.25 * p.dx;

        grid_t *g = grid_alloc(p.N, p.L, p.rk_method);
        set_flat_spacetime(g);

        double ham = compute_constraint_l2(g);
        double mom = compute_momentum_l2(g);

        printf("  Ham L2 = %.4e  Mom L2 = %.4e\n", ham, mom);

        int pass = (ham < 1e-10) && (mom < 1e-10);
        printf("  %s (threshold 1e-10)\n\n", pass ? "PASSED" : "FAILED");
        if (!pass) all_passed = 0;

        grid_free(g);
    }

    /* ------ Test 2: Single BH evolved, constraints bounded ------ */
    {
        printf("--- Test 2: Single BH (N=64, T=2M) ---\n");
        fflush(stdout);

        sim_params_t p = default_params();
        p.N = 64; p.L = 64.0; p.CFL = 0.25; p.sigma = 0.3;
        p.dx = p.L / p.N; p.dt = p.CFL * p.dx;
        int steps = (int)(2.0 / p.dt + 0.5);

        mesh_t *m = mesh_create_ex(1, p.N, p.L, p.rk_method, NUM_CCZ4_FIELDS);
        grid_t *g = m->blocks[0]->grid;
        p.N = g->N; p.dx = g->dx; p.dt = p.CFL * p.dx;
        steps = (int)(2.0 / p.dt + 0.5);

        double mass = 1.0;
        double center[1][3] = {{0.0, 0.0, 0.0}};
        set_brill_lindquist(g, 1, &mass, center);

        printf("  Evolving %d steps (T=2M)...\n", steps);
        fflush(stdout);
        p.time = 0.0;
        for (int s = 1; s <= steps; s++) {
            rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
            p.time += p.dt;
        }

        double ham, mom;
        constraints_annular(g, 5.0, 25.0, &ham, &mom);

        printf("  Ham L2 = %.4e  Mom L2 = %.4e  (5M < r < 25M)\n", ham, mom);

        int pass = (ham < 0.1) && (mom < 0.1);
        printf("  %s (threshold 0.1)\n\n", pass ? "PASSED" : "FAILED");
        if (!pass) all_passed = 0;

        mesh_free(m);
    }

    /* ------ Test 3: Momentum convergence ------ */
    {
        printf("--- Test 3: Momentum convergence (N=32,64) ---\n");
        fflush(stdout);

        int resolutions[] = {32, 64};
        double mom_l2[2], ham_l2[2], dx_vals[2];

        for (int res = 0; res < 2; res++) {
            sim_params_t p = default_params();
            p.N = resolutions[res]; p.L = 64.0; p.CFL = 0.25; p.sigma = 0.3;
            p.dx = p.L / p.N; p.dt = p.CFL * p.dx;
            int steps = (int)(2.0 / p.dt + 0.5);

            mesh_t *m = mesh_create_ex(1, p.N, p.L, p.rk_method, NUM_CCZ4_FIELDS);
            grid_t *g = m->blocks[0]->grid;
            p.N = g->N; p.dx = g->dx; p.dt = p.CFL * p.dx;
            steps = (int)(2.0 / p.dt + 0.5);
            dx_vals[res] = p.dx;

            double mass = 1.0;
            double center[1][3] = {{0.0, 0.0, 0.0}};
            set_brill_lindquist(g, 1, &mass, center);

            p.time = 0.0;
            for (int s = 1; s <= steps; s++) {
                rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
                p.time += p.dt;
            }

            constraints_annular(g, 5.0, 25.0, &ham_l2[res], &mom_l2[res]);
            printf("  N=%d  Ham=%.4e  Mom=%.4e\n",
                   resolutions[res], ham_l2[res], mom_l2[res]);
            fflush(stdout);

            mesh_free(m);
        }

        double ham_order = log(ham_l2[0] / ham_l2[1]) / log(dx_vals[0] / dx_vals[1]);
        double mom_order = log(mom_l2[0] / mom_l2[1]) / log(dx_vals[0] / dx_vals[1]);

        printf("  Ham order = %.2f  Mom order = %.2f\n", ham_order, mom_order);

        int pass = (mom_order > 3.0);
        printf("  %s (momentum order > 3.0)\n\n", pass ? "PASSED" : "FAILED");
        if (!pass) all_passed = 0;
    }

    printf("=== Overall: %s ===\n", all_passed ? "ALL PASSED" : "SOME FAILED");

    backend_cleanup();
    return all_passed ? 0 : 1;
}
