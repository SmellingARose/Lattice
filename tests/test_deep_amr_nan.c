/*
 * Lattice — Deep AMR NaN isolation test.
 * Small mesh + deep AMR + per-step NaN check.
 * Runs fast on GPU (~2-5s/step) to isolate subcycling bugs.
 *
 * Tests: N=16, L=16, max_level=4, single BH, 80 steps.
 * Reports exact step/block/level of first NaN.
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
#include "../src/backend/backend.h"

int main(void)
{
    setbuf(stdout, NULL);
    backend_init();

    int N = 16;
    double L = 16.0;
    int max_level = 4;

    mesh_t *m = mesh_create_ex(N, L, RK_CLASSIC, NUM_CCZ4_FIELDS);
    mesh_rebuild_neighbors(m);

    sim_params_t p = default_params();
    p.N = N; p.L = L; p.dx = m->dx_base; p.dt = 0.25 * p.dx;
    p.amr.max_level = max_level;
    p.amr.chi_refine = 0.1;
    p.amr.regrid_every = 0;
    p.noise.use_cako = 0;

    printf("dx_base=%.3f dt=%.4f dx_fine=%.4f levels=%d\n",
           p.dx, p.dt, p.dx / (1 << max_level), max_level);

    puncture_data_t bh = {.mass = 1.0, .center = {0, 0, 0}};
    set_bowen_york_mesh(m, 1, &bh, max_level);
    ghost_exchange(m);

    printf("%d blocks (%d leaves)\n", m->num_blocks, mesh_num_leaves(m));

    p.time = 0.0;
    int crashed = 0;

    for (int step = 1; step <= 80; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;

        /* Sync to host for NaN scan */
        if (backend_is_gpu())
            gpu_sync_all_to_host(m);

        for (int bid = 0; bid < m->num_blocks && !crashed; bid++) {
            block_t *blk = m->blocks[bid];
            if (!blk || !blk->is_leaf) continue;
            grid_t *g = blk->grid;
            int lo = g->ghost, hi = g->ghost + g->N;
            for (int k = lo; k < hi && !crashed; k++)
                for (int j = lo; j < hi && !crashed; j++)
                    for (int i = lo; i < hi && !crashed; i++) {
                        int idx = i + j * g->Ntotal + k * g->Ntotal * g->Ntotal;
                        if (!isfinite(g->fields[FIELD_CHI][idx]) ||
                            !isfinite(g->fields[FIELD_LAPSE][idx])) {
                            printf("*** NaN step %d t=%.2f block %d level %d "
                                   "cell (%d,%d,%d) chi=%.4e lapse=%.4e\n",
                                   step, p.time, bid, blk->loc.level,
                                   i - lo, j - lo, k - lo,
                                   g->fields[FIELD_CHI][idx],
                                   g->fields[FIELD_LAPSE][idx]);
                            crashed = 1;
                        }
                    }
        }
        if (crashed) break;

        if (step % 10 == 0) {
            double ham = mesh_constraint_l2(m);
            printf("step %2d t=%5.1f Ham=%.3e %s\n",
                   step, p.time, ham, isfinite(ham) ? "OK" : "NaN!");
        }
    }

    if (!crashed)
        printf("=== PASSED — survived 80 steps (t=%.1f) ===\n", p.time);
    else
        printf("=== CRASHED ===\n");

    mesh_free(m);
    backend_cleanup();
    return crashed ? 1 : 0;
}
