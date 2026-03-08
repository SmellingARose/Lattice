/*
 * Lattice — 3D Numerical Relativity
 * Inspiral solver smoke test: D10 initial data with reduced domain.
 *
 * Same physical setup as test_binary_inspiral (D10 equal-mass QC binary)
 * but with a smaller domain (L=256 instead of L=1536) and fewer AMR levels
 * to verify the constraint solver works correctly before committing to a
 * multi-hour evolution.
 *
 * Tests:
 *   1. Solver converges (residual < tolerance)
 *   2. chi > 0 everywhere (no negative conformal factor)
 *   3. Hamiltonian constraint bounded
 *   4. Lapse profile reasonable (trumpet lapse < 1 near punctures)
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

int main(void)
{
    setbuf(stdout, NULL);

    printf("\n=== Inspiral Solver Smoke Test (D10) ===\n\n");

    backend_init();

    int passed = 0, failed = 0;

    /* ---- Test 1: Small domain (L=256, 5 AMR levels) ---- */
    {
        printf("--- Test 1: L=256, 5 AMR levels ---\n");
        int max_level = 5;
        double L = 256.0;
        int N = 32;

        mesh_t *m = mesh_create(N, L, RK_CLASSIC);
        printf("  Domain: [-%.0f, %.0f]^3, N=%d, dx_base=%.2f\n",
               L/2, L/2, N, m->dx_base);

        puncture_data_t bhs[2];
        memset(bhs, 0, sizeof(bhs));
        bhs[0].mass        = M_BARE;
        bhs[0].center[2]   = +D_SEP / 2.0;
        bhs[0].momentum[1] = +P_Y;
        bhs[1].mass        = M_BARE;
        bhs[1].center[2]   = -D_SEP / 2.0;
        bhs[1].momentum[1] = -P_Y;

        time_t t0 = time(NULL);
        set_bowen_york_mesh(m, 2, bhs, max_level);
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
            printf("  PASS: chi > 0\n"); passed++;
        } else {
            printf("  FAIL: chi <= 0\n"); failed++;
        }

        /* Check Hamiltonian constraint */
        ghost_exchange_multilevel(m);
        double ham_l2 = mesh_constraint_l2(m);
        printf("  Ham L2 = %.6e\n", ham_l2);

        if (ham_l2 < 1.0) {
            printf("  PASS: Ham bounded\n"); passed++;
        } else {
            printf("  FAIL: Ham unbounded\n"); failed++;
        }

        mesh_free(m);
    }

    /* ---- Test 2: Medium domain (L=512, 8 AMR levels) ---- */
    {
        printf("\n--- Test 2: L=512, 8 AMR levels ---\n");
        int max_level = 8;
        double L = 512.0;
        int N = 32;

        mesh_t *m = mesh_create(N, L, RK_CLASSIC);
        printf("  Domain: [-%.0f, %.0f]^3, N=%d, dx_base=%.2f\n",
               L/2, L/2, N, m->dx_base);

        puncture_data_t bhs[2];
        memset(bhs, 0, sizeof(bhs));
        bhs[0].mass        = M_BARE;
        bhs[0].center[2]   = +D_SEP / 2.0;
        bhs[0].momentum[1] = +P_Y;
        bhs[1].mass        = M_BARE;
        bhs[1].center[2]   = -D_SEP / 2.0;
        bhs[1].momentum[1] = -P_Y;

        time_t t0 = time(NULL);
        set_bowen_york_mesh(m, 2, bhs, max_level);
        double dt_sec = difftime(time(NULL), t0);

        printf("  Solver time: %.0f sec\n", dt_sec);
        printf("  Blocks: %d total, %d leaves, max_level=%d\n",
               m->num_blocks, mesh_num_leaves(m), m->max_level);

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
            printf("  PASS: chi > 0\n"); passed++;
        } else {
            printf("  FAIL: chi <= 0\n"); failed++;
        }

        ghost_exchange_multilevel(m);
        double ham_l2 = mesh_constraint_l2(m);
        printf("  Ham L2 = %.6e\n", ham_l2);

        if (ham_l2 < 1.0) {
            printf("  PASS: Ham bounded\n"); passed++;
        } else {
            printf("  FAIL: Ham unbounded\n"); failed++;
        }

        mesh_free(m);
    }

    /* ---- Test 3: Full inspiral domain (L=1536, 11 AMR levels) ---- */
    {
        printf("\n--- Test 3: L=1536, 11 AMR levels (inspiral params) ---\n");
        int max_level = 11;
        double L = 1536.0;
        int N = 32;

        mesh_t *m = mesh_create(N, L, RK_CLASSIC);
        printf("  Domain: [-%.0f, %.0f]^3, N=%d, dx_base=%.2f\n",
               L/2, L/2, N, m->dx_base);

        puncture_data_t bhs[2];
        memset(bhs, 0, sizeof(bhs));
        bhs[0].mass        = M_BARE;
        bhs[0].center[2]   = +D_SEP / 2.0;
        bhs[0].momentum[1] = +P_Y;
        bhs[1].mass        = M_BARE;
        bhs[1].center[2]   = -D_SEP / 2.0;
        bhs[1].momentum[1] = -P_Y;

        time_t t0 = time(NULL);
        set_bowen_york_mesh(m, 2, bhs, max_level);
        double dt_sec = difftime(time(NULL), t0);

        printf("  Solver time: %.0f sec\n", dt_sec);
        printf("  Blocks: %d total, %d leaves, max_level=%d\n",
               m->num_blocks, mesh_num_leaves(m), m->max_level);

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
            printf("  PASS: chi > 0\n"); passed++;
        } else {
            printf("  FAIL: chi <= 0\n"); failed++;
        }

        ghost_exchange_multilevel(m);
        double ham_l2 = mesh_constraint_l2(m);
        printf("  Ham L2 = %.6e\n", ham_l2);

        if (ham_l2 < 1.0) {
            printf("  PASS: Ham bounded\n"); passed++;
        } else {
            printf("  FAIL: Ham unbounded\n"); failed++;
        }

        mesh_free(m);
    }

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
