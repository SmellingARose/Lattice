/*
 * Test: GPU vs CPU BH tracker position update.
 * Creates a small uniform mesh with a single BH, evolves a few steps,
 * then verifies both bh_tracker_update_positions (mesh/CPU path) and
 * bh_tracker_update_positions_packed (packed/GPU path) find the same
 * lapse minimum position.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/amr/mesh.h"
#include "../src/amr/meshblock_pack.h"
#include "../src/amr/ghost_exchange.h"
#include "../src/diagnostics/bh_tracker.h"
#include "../src/initial_data/puncture.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/numerics/rk4.h"
#include "../src/backend/backend.h"
#include "../src/evolution/ccz4_rhs.h"

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

int main(void)
{
    backend_init();

    printf("=== Test: GPU vs CPU BH tracker position update ===\n\n");

    /* --- Test 1: Single BH, uniform grid --- */
    printf("--- Test 1: Single BH at origin ---\n");
    {
        int N = 32;
        double L = 16.0;
        mesh_t *m = mesh_create(N, L, RK_CLASSIC);
        mesh_rebuild_neighbors(m);

        /* Set up single BH at origin */
        puncture_data_t bh = {.mass = 1.0, .center = {0, 0, 0}};
        set_bowen_york_mesh(m, 1, &bh, 0);
        ghost_exchange(m);

        /* Evolve 10 steps to let lapse settle */
        sim_params_t p = default_params();
        p.N = N; p.L = L; p.dx = m->dx_base;
        p.dt = 0.25 * p.dx;
        for (int s = 0; s < 10; s++) {
            rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
            p.time += p.dt;
        }

        /* CPU path: tracker update on mesh blocks */
        bh_tracker_t *tr_cpu = bh_tracker_alloc(1, &bh, 12, 24);
        bh_tracker_update_positions(tr_cpu, m);
        double cpu_pos[3] = {tr_cpu->bh[0].center[0],
                             tr_cpu->bh[0].center[1],
                             tr_cpu->bh[0].center[2]};
        double cpu_lapse = tr_cpu->bh[0].lapse_min;
        printf("  CPU: pos=(%.4f, %.4f, %.4f) lapse=%.6f\n",
               cpu_pos[0], cpu_pos[1], cpu_pos[2], cpu_lapse);

        /* GPU/packed path: build a level pack and use packed tracker */
        int ids[1] = {0};
        meshblock_pack_t *pack = meshblock_pack_create(
            1, m->blocks[0]->grid->npoints, ids, 0,
            RK_CLASSIC, m->n_fields);
        meshblock_pack_load(pack, m->blocks);
        meshblock_pack_load_meta(pack, m->blocks);
        m->level_packs[0] = pack;

        bh_tracker_t *tr_gpu = bh_tracker_alloc(1, &bh, 12, 24);
        bh_tracker_update_positions_packed(tr_gpu, m);
        double gpu_pos[3] = {tr_gpu->bh[0].center[0],
                             tr_gpu->bh[0].center[1],
                             tr_gpu->bh[0].center[2]};
        double gpu_lapse = tr_gpu->bh[0].lapse_min;
        printf("  GPU: pos=(%.4f, %.4f, %.4f) lapse=%.6f\n",
               gpu_pos[0], gpu_pos[1], gpu_pos[2], gpu_lapse);

        double dist = sqrt((cpu_pos[0]-gpu_pos[0])*(cpu_pos[0]-gpu_pos[0]) +
                           (cpu_pos[1]-gpu_pos[1])*(cpu_pos[1]-gpu_pos[1]) +
                           (cpu_pos[2]-gpu_pos[2])*(cpu_pos[2]-gpu_pos[2]));
        printf("  Position difference: %.6e\n", dist);
        printf("  Lapse difference: %.6e\n", fabs(cpu_lapse - gpu_lapse));

        /* Position may differ by up to dx due to symmetry tie-breaking:
         * multiple cells at the lapse minimum have identical values.
         * CPU and GPU iterate in different orders, pick different ties. */
        double dx = m->dx_base;
        check(dist < dx + 1e-10, "CPU and GPU find same position (within dx)");
        check(fabs(cpu_lapse - gpu_lapse) < 1e-10, "CPU and GPU find same lapse");

        m->level_packs[0] = NULL;  /* don't double-free */
        meshblock_pack_free(pack);
        bh_tracker_free(tr_cpu);
        bh_tracker_free(tr_gpu);
        mesh_free(m);
    }

    /* --- Test 2: Two BHs with exclusion zones --- */
    printf("\n--- Test 2: Two BHs with exclusion ---\n");
    {
        int N = 32;
        double L = 20.0;
        mesh_t *m = mesh_create(N, L, RK_CLASSIC);
        mesh_rebuild_neighbors(m);

        puncture_data_t bhs[2] = {
            {.mass = 1.0, .center = {-3, 0, 0}},
            {.mass = 1.0, .center = { 3, 0, 0}}
        };
        set_bowen_york_mesh(m, 2, bhs, 0);
        ghost_exchange(m);

        /* Evolve a few steps */
        sim_params_t p = default_params();
        p.N = N; p.L = L; p.dx = m->dx_base;
        p.dt = 0.25 * p.dx;
        for (int s = 0; s < 5; s++) {
            rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
            p.time += p.dt;
        }

        /* CPU path */
        bh_tracker_t *tr_cpu = bh_tracker_alloc(2, bhs, 12, 24);
        bh_tracker_update_positions(tr_cpu, m);
        printf("  CPU BH0: (%.4f, %.4f, %.4f) lapse=%.6f\n",
               tr_cpu->bh[0].center[0], tr_cpu->bh[0].center[1],
               tr_cpu->bh[0].center[2], tr_cpu->bh[0].lapse_min);
        printf("  CPU BH1: (%.4f, %.4f, %.4f) lapse=%.6f\n",
               tr_cpu->bh[1].center[0], tr_cpu->bh[1].center[1],
               tr_cpu->bh[1].center[2], tr_cpu->bh[1].lapse_min);

        /* GPU/packed path */
        int ids[1] = {0};
        meshblock_pack_t *pack = meshblock_pack_create(
            1, m->blocks[0]->grid->npoints, ids, 0,
            RK_CLASSIC, m->n_fields);
        meshblock_pack_load(pack, m->blocks);
        meshblock_pack_load_meta(pack, m->blocks);
        m->level_packs[0] = pack;

        bh_tracker_t *tr_gpu = bh_tracker_alloc(2, bhs, 12, 24);
        bh_tracker_update_positions_packed(tr_gpu, m);
        printf("  GPU BH0: (%.4f, %.4f, %.4f) lapse=%.6f\n",
               tr_gpu->bh[0].center[0], tr_gpu->bh[0].center[1],
               tr_gpu->bh[0].center[2], tr_gpu->bh[0].lapse_min);
        printf("  GPU BH1: (%.4f, %.4f, %.4f) lapse=%.6f\n",
               tr_gpu->bh[1].center[0], tr_gpu->bh[1].center[1],
               tr_gpu->bh[1].center[2], tr_gpu->bh[1].lapse_min);

        /* Check both BHs match */
        for (int b = 0; b < 2; b++) {
            double dist = sqrt(
                (tr_cpu->bh[b].center[0]-tr_gpu->bh[b].center[0])*
                (tr_cpu->bh[b].center[0]-tr_gpu->bh[b].center[0]) +
                (tr_cpu->bh[b].center[1]-tr_gpu->bh[b].center[1])*
                (tr_cpu->bh[b].center[1]-tr_gpu->bh[b].center[1]) +
                (tr_cpu->bh[b].center[2]-tr_gpu->bh[b].center[2])*
                (tr_cpu->bh[b].center[2]-tr_gpu->bh[b].center[2]));
            char msg[64];
            snprintf(msg, sizeof(msg), "BH%d position matches (dist=%.2e)", b, dist);
            check(dist < 1e-10, msg);
            snprintf(msg, sizeof(msg), "BH%d lapse matches", b);
            check(fabs(tr_cpu->bh[b].lapse_min - tr_gpu->bh[b].lapse_min) < 1e-10, msg);
        }

        m->level_packs[0] = NULL;
        meshblock_pack_free(pack);
        bh_tracker_free(tr_cpu);
        bh_tracker_free(tr_gpu);
        mesh_free(m);
    }

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    backend_cleanup();
    return n_fail > 0 ? 1 : 0;
}
