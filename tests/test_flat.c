/*
 * Lattice — 3D Numerical Relativity
 * Flat spacetime stability test.
 *
 * Minkowski initial data evolved for 500 steps.
 * Pass criterion: Hamiltonian constraint L2 norm < 1e-10.
 *
 * This is the first test in the validation ladder:
 *   flat spacetime -> single BH -> binary
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

int main(void)
{
    printf("=== Flat Spacetime Stability Test ===\n");

    /* Setup: N=32, L=10, CFL=0.25, 500 steps */
    sim_params_t p = default_params();
    p.N         = 32;
    p.L         = 10.0;
    p.CFL       = 0.25;
    p.num_steps = 500;
    p.sigma     = 0.3;
    p.dx = p.L / p.N;
    p.dt = p.CFL * p.dx;

    backend_init();
    mesh_t *m = mesh_create_ex(1, p.N, p.L, p.rk_method, NUM_CCZ4_FIELDS);
    grid_t *g = m->blocks[0]->grid;

    /* Recompute after possible padding */
    p.N  = g->N;
    p.dx = g->dx;
    p.dt = p.CFL * p.dx;

    printf("  N = %d, Ntotal = %d, dx = %.6f, dt = %.6f\n",
           g->N, g->Ntotal, g->dx, p.dt);

    /* Set flat initial data */
    set_flat_spacetime(g);

    double ham0 = compute_constraint_l2(g);
    printf("  Initial Ham L2 = %.6e\n", ham0);

    /* Evolve */
    int diag_every = p.num_steps / 5;  /* full diagnostic 5 times */
    if (diag_every < 1) diag_every = 1;

    p.time = 0.0;
    for (int step = 1; step <= p.num_steps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;

        double pct = 100.0 * step / p.num_steps;
        if (step % diag_every == 0 || step == p.num_steps) {
            double ham = mesh_constraint_l2(m);
            printf("\r  step %4d/%d  [%5.1f%%]  Ham L2 = %.6e",
                   step, p.num_steps, pct, ham);
            fflush(stdout);
        } else if (step % 10 == 0) {
            printf("\r  step %4d/%d  [%5.1f%%]", step, p.num_steps, pct);
            fflush(stdout);
        }
    }
    printf("\n");

    /* Final check */
    double ham_final = mesh_constraint_l2(m);
    printf("  Final Ham L2 = %.6e\n", ham_final);

    int passed = (ham_final < 1.0e-10);
    printf("\n  %s (threshold = 1e-10)\n", passed ? "PASSED" : "FAILED");

    mesh_free(m);
    backend_cleanup();

    return passed ? 0 : 1;
}
