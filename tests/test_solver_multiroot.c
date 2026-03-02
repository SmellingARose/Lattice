/*
 * Lattice — Quick AMR multigrid solver validation.
 *
 * Tests the constraint solver on a single-root AMR mesh with
 * refinement near punctures — same config as the binary inspiral
 * but solver-only (no evolution). Runs in seconds.
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/amr/mesh.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Same physical setup as binary inspiral (arXiv:0709.0838, Table I) */
#define M_BARE     0.4824
#define D_SEP      10.0
#define P_Y        0.0939

#define L_DOMAIN   40.0
#define N_BLOCK    16
#define MAX_LEVEL  2

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); pass_count++; } \
    else      { printf("  FAIL: %s\n", msg); fail_count++; } \
} while(0)

int main(void)
{
    printf("=== AMR Solver Quick Test ===\n");
    printf("  N_BLOCK=%d, L=%.0f, MAX_LEVEL=%d\n",
           N_BLOCK, L_DOMAIN, MAX_LEVEL);
    printf("  dx_base = %.4f, dx_fine = %.4f (at level %d)\n\n",
           L_DOMAIN / N_BLOCK,
           L_DOMAIN / N_BLOCK / (1 << MAX_LEVEL), MAX_LEVEL);

    backend_init();
    mesh_t *m = mesh_create(N_BLOCK, L_DOMAIN, RK_CLASSIC);

    printf("  Root blocks: %d, leaf blocks: %d\n\n",
           m->num_blocks, mesh_num_leaves(m));

    /* ── Test 1: Binary puncture (same as inspiral) ─────────────── */
    printf("--- Test 1: Binary puncture (inspiral config) ---\n");
    fflush(stdout);

    puncture_data_t bhs[2];
    memset(bhs, 0, sizeof(bhs));
    bhs[0].mass        = M_BARE;
    bhs[0].center[2]   = +D_SEP / 2.0;
    bhs[0].momentum[1] = +P_Y;
    bhs[1].mass        = M_BARE;
    bhs[1].center[2]   = -D_SEP / 2.0;
    bhs[1].momentum[1] = -P_Y;

    time_t t0 = time(NULL);
    set_bowen_york_mesh(m, 2, bhs, MAX_LEVEL);
    double solve_sec = difftime(time(NULL), t0);

    printf("  Solve time: %.0f sec\n", solve_sec);

    /* Check constraint quality */
    double ham = mesh_constraint_l2(m);
    printf("  Ham L2 = %.6e\n", ham);

    CHECK(ham < 1.0, "Hamiltonian constraint bounded (< 1.0)");
    CHECK(!isnan(ham) && !isinf(ham), "No NaN/Inf in constraints");

    /* Check chi > 0 everywhere */
    double chi_min = 1e30;
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || !blk->is_leaf) continue;
        grid_t *g = blk->grid;
        int gw = g->ghost, Nt = g->Ntotal;
        for (int k = gw; k < Nt - gw; k++)
            for (int j = gw; j < Nt - gw; j++)
                for (int i = gw; i < Nt - gw; i++) {
                    int idx = k * Nt * Nt + j * Nt + i;
                    double c = g->fields[FIELD_CHI][idx];
                    if (c < chi_min) chi_min = c;
                }
    }
    printf("  chi_min = %.6e\n", chi_min);
    CHECK(chi_min > 0.0, "chi > 0 everywhere");

    /* Check lapse profile (trumpet slice) */
    double lapse_min = 1e30;
    for (int b = 0; b < m->num_blocks; b++) {
        block_t *blk = m->blocks[b];
        if (!blk || !blk->is_leaf) continue;
        grid_t *g = blk->grid;
        int gw = g->ghost, Nt = g->Ntotal;
        for (int k = gw; k < Nt - gw; k++)
            for (int j = gw; j < Nt - gw; j++)
                for (int i = gw; i < Nt - gw; i++) {
                    int idx = k * Nt * Nt + j * Nt + i;
                    double a = g->fields[FIELD_LAPSE][idx];
                    if (a < lapse_min) lapse_min = a;
                }
    }
    printf("  lapse_min = %.6e\n", lapse_min);
    CHECK(lapse_min > 0.0 && lapse_min < 1.0, "Lapse in (0, 1)");

    mesh_free(m);

    /* ── Test 2: Single puncture with AMR ───────────────────────── */
    printf("\n--- Test 2: Single puncture with AMR ---\n");
    fflush(stdout);

    mesh_t *m2 = mesh_create(N_BLOCK, L_DOMAIN, RK_CLASSIC);

    puncture_data_t bh1[1];
    memset(bh1, 0, sizeof(bh1));
    bh1[0].mass = 1.0;
    bh1[0].momentum[1] = 0.1;

    t0 = time(NULL);
    set_bowen_york_mesh(m2, 1, bh1, MAX_LEVEL);
    solve_sec = difftime(time(NULL), t0);

    printf("  Solve time: %.0f sec\n", solve_sec);

    double ham2 = mesh_constraint_l2(m2);
    printf("  Ham L2 = %.6e\n", ham2);

    CHECK(ham2 < 1.0, "Single BH: Hamiltonian bounded (< 1.0)");
    CHECK(!isnan(ham2) && !isinf(ham2), "Single BH: No NaN/Inf");

    mesh_free(m2);

    /* ── Summary ────────────────────────────────────────────────── */
    printf("\n=== Results: %d passed, %d failed ===\n",
           pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
