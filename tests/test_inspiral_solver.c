/*
 * Lattice — 3D Numerical Relativity
 * Inspiral solver smoke test: constraint solver at inspiral-scale AMR depth.
 *
 * Validates the CPU composite FAS multigrid solver with inspiral-like
 * parameters. Tests run sequentially — each mesh is freed before the
 * next is allocated, so peak memory is per-test, not cumulative.
 *
 * Memory budget: ~51 MB/block (4 RK banks × 25 fields × 64K points × 8B).
 *
 * Tests:
 *   1. L=1536, 7 levels, 2 punctures (D10) — ~329 blocks, ~17 GB
 *   2. L=1536, 6 levels, 4 punctures       — ~265 blocks, ~14 GB
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/amr/mesh.h"
#include "../src/amr/refine.h"
#include "../src/amr/ghost_exchange.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* D10 physical params (same as inspiral test) */
#define M_BARE  0.48595
#define D_SEP   10.0
#define P_Y     0.09543

/* Helper: run solver, check chi > 0 and Ham bounded.
 * Returns number of failures (0 = all pass). */
static int run_solver_test(const char *label, double L, int max_level,
                           int n_punctures, puncture_data_t *bhs,
                           int *passed, int *failed)
{
    int local_fail = 0;

    printf("--- %s ---\n", label);
    int N = 32;

    mesh_t *m = mesh_create(N, L, RK_CLASSIC);
    printf("  Domain: [-%.0f, %.0f]^3, N=%d, dx_base=%.2f\n",
           L/2, L/2, N, m->dx_base);

    time_t t0 = time(NULL);
    set_bowen_york_mesh(m, n_punctures, bhs, max_level);
    double dt_sec = difftime(time(NULL), t0);

    printf("  Solver time: %.0f sec\n", dt_sec);
    printf("  Blocks: %d total, %d leaves, max_level=%d\n",
           m->num_blocks, mesh_num_leaves(m), m->max_level);

    /* Check chi > 0 */
    double chi_min = 1e30;
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || !blk->is_leaf) continue;
        grid_t *g = blk->grid;
        int gw = g->ghost;
        for (int k = gw; k < gw + g->N; k++)
            for (int j = gw; j < gw + g->N; j++)
                for (int i = gw; i < gw + g->N; i++) {
                    double c = g->fields[FIELD_CHI][IDX(g, i, j, k)];
                    if (c < chi_min) chi_min = c;
                }
    }
    printf("  chi_min = %.6e\n", chi_min);

    if (chi_min > 0.0) {
        printf("  PASS: chi > 0\n"); (*passed)++;
    } else {
        printf("  FAIL: chi <= 0\n"); (*failed)++; local_fail++;
    }

    /* Check Hamiltonian constraint */
    ghost_exchange_multilevel(m);
    double ham_l2 = mesh_constraint_l2(m);
    printf("  Ham L2 = %.6e\n", ham_l2);

    if (ham_l2 < 1.0) {
        printf("  PASS: Ham bounded\n"); (*passed)++;
    } else {
        printf("  FAIL: Ham unbounded\n"); (*failed)++; local_fail++;
    }

    mesh_free(m);
    return local_fail;
}

int main(void)
{
    setbuf(stdout, NULL);

    printf("\n=== Inspiral Solver Smoke Test ===\n\n");

    backend_init();

    int passed = 0, failed = 0;

    /* ---- Test 1: D10 binary, 7 AMR levels ----
     * Same physics as the full inspiral but 7 levels instead of 11.
     * Exercises composite V-cycle with cross-level ghost exchange,
     * restriction, prolongation, and FAS correction.
     * ~329 blocks = ~17 GB. */
    {
        puncture_data_t bhs[2];
        memset(bhs, 0, sizeof(bhs));
        bhs[0].mass        = M_BARE;
        bhs[0].center[2]   = +D_SEP / 2.0;
        bhs[0].momentum[1] = +P_Y;
        bhs[1].mass        = M_BARE;
        bhs[1].center[2]   = -D_SEP / 2.0;
        bhs[1].momentum[1] = -P_Y;

        run_solver_test("Test 1: L=1536, 7 levels, 2 punctures (D10 binary)",
                        1536.0, 7, 2, bhs, &passed, &failed);
    }

    /* ---- Test 2: 4 BH square, 6 AMR levels ----
     * Four equal-mass punctures in a square at D=10M separation.
     * Tests N-body solver path with multiple refinement centers.
     * ~265 blocks = ~14 GB. */
    {
        printf("\n");
        double r = D_SEP / sqrt(2.0);  /* half-diagonal of square */
        puncture_data_t bhs[4];
        memset(bhs, 0, sizeof(bhs));
        /* Square in x-z plane, each BH orbiting */
        bhs[0].mass = M_BARE;
        bhs[0].center[0] = +r;  bhs[0].center[2] = +r;
        bhs[0].momentum[1] = +P_Y;

        bhs[1].mass = M_BARE;
        bhs[1].center[0] = -r;  bhs[1].center[2] = +r;
        bhs[1].momentum[1] = -P_Y;

        bhs[2].mass = M_BARE;
        bhs[2].center[0] = -r;  bhs[2].center[2] = -r;
        bhs[2].momentum[1] = +P_Y;

        bhs[3].mass = M_BARE;
        bhs[3].center[0] = +r;  bhs[3].center[2] = -r;
        bhs[3].momentum[1] = -P_Y;

        run_solver_test("Test 2: L=1536, 6 levels, 4 punctures (N-body)",
                        1536.0, 6, 4, bhs, &passed, &failed);
    }

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
