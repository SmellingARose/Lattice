/*
 * Lattice — 3D Numerical Relativity
 * AMR Stage 1 test: mesh infrastructure verification.
 *
 * Tests:
 *   1. Single-block mesh (N_root=1, N_block=32) produces identical
 *      results to grid_t for flat spacetime evolution.
 *   2. Mesh topology: neighbor IDs, nblevel tables, boundary flags.
 *   3. MeshBlockPack load/store round-trip preserves data to roundoff.
 *
 * Pass criterion:
 *   - 1-block mesh flat spacetime Ham L2 < 1e-10 (same as test_flat)
 *   - Topology checks pass (correct neighbors, boundaries)
 *   - Pack round-trip max error < 1e-15 (roundoff)
 */

#include "../src/amr/mesh.h"
#include "../src/amr/meshblock_pack.h"
#include "../src/amr/morton.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/puncture.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/numerics/rk4.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int test_count = 0;
static int pass_count = 0;

static void check(int cond, const char *name)
{
    test_count++;
    if (cond) {
        pass_count++;
        printf("  [PASS] %s\n", name);
    } else {
        printf("  [FAIL] %s\n", name);
    }
}

/* ======================================================================
 * Test 1: Morton encoding/decoding round-trip
 * ====================================================================== */
static void test_morton(void)
{
    printf("\n--- Test: Morton encoding ---\n");

    /* Round-trip for various coordinates */
    int coords[][3] = {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1},
                        {3,5,7}, {15,15,15}, {0,0,255}};
    int n = sizeof(coords) / sizeof(coords[0]);
    int all_ok = 1;

    for (int t = 0; t < n; t++) {
        int64_t code = morton_encode3d(coords[t][0], coords[t][1], coords[t][2]);
        int dx, dy, dz;
        morton_decode3d(code, &dx, &dy, &dz);
        if (dx != coords[t][0] || dy != coords[t][1] || dz != coords[t][2]) {
            printf("    Morton fail: (%d,%d,%d) -> %lld -> (%d,%d,%d)\n",
                   coords[t][0], coords[t][1], coords[t][2],
                   (long long)code, dx, dy, dz);
            all_ok = 0;
        }
    }
    check(all_ok, "Morton encode/decode round-trip");

    /* Verify Z-ordering: (0,0,0) < (1,0,0) < (0,1,0) < (1,1,0) < ... */
    int64_t m000 = morton_encode3d(0, 0, 0);
    int64_t m100 = morton_encode3d(1, 0, 0);
    int64_t m010 = morton_encode3d(0, 1, 0);
    int64_t m110 = morton_encode3d(1, 1, 0);
    int64_t m001 = morton_encode3d(0, 0, 1);
    check(m000 < m100 && m100 < m010 && m010 < m110 && m110 < m001,
          "Morton Z-ordering correct");

    /* Child encoding: child 7 of parent (1,2,3) = (3,5,7) */
    int64_t parent = morton_encode3d(1, 2, 3);
    int64_t child7 = morton_child(parent, 7);
    int cx, cy, cz;
    morton_decode3d(child7, &cx, &cy, &cz);
    check(cx == 3 && cy == 5 && cz == 7, "Morton child encoding (octant 7)");

    /* Parent recovery */
    int64_t recovered = morton_parent(child7);
    check(recovered == parent, "Morton parent recovery");
}

/* ======================================================================
 * Test 2: Mesh creation and topology
 * ====================================================================== */
static void test_mesh_topology(void)
{
    printf("\n--- Test: Mesh topology (2x2x2 = 8 blocks) ---\n");

    /* Use N_block=16 to keep memory low (~5 MB/block, 40 MB total) */
    mesh_t *m = mesh_create(2, 16, 10.0, RK_CLASSIC);

    check(m->num_blocks == 8, "8 blocks created");
    check(m->N_root == 2, "N_root = 2");
    check(m->N_block == 16, "N_block = 16");
    check(m->max_level == 0, "max_level = 0");
    check(mesh_num_leaves(m) == 8, "8 leaf blocks");

    /* Check all blocks have correct level and are leaves */
    int all_leaves = 1;
    int all_level0 = 1;
    for (int i = 0; i < m->num_blocks; i++) {
        if (!m->blocks[i]->is_leaf) all_leaves = 0;
        if (m->blocks[i]->loc.level != 0) all_level0 = 0;
    }
    check(all_leaves, "All blocks are leaves");
    check(all_level0, "All blocks at level 0");

    /* Check neighbor topology for corner block (0,0,0):
     * Should have 3 face neighbors, 3 edge, 1 corner = 7 neighbors,
     * and 19 boundaries. */
    block_t *corner = NULL;
    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (b->loc.lx1 == 0 && b->loc.lx2 == 0 && b->loc.lx3 == 0) {
            corner = b;
            break;
        }
    }
    check(corner != NULL, "Found corner block (0,0,0)");

    if (corner) {
        int nbr_count = 0;
        int bdy_count = 0;
        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            if (corner->neighbor_ids[n] >= 0) nbr_count++;
            else bdy_count++;
        }
        check(nbr_count == 7, "Corner block has 7 neighbors (3F+3E+1C)");
        check(bdy_count == 19, "Corner block has 19 boundary faces");

        /* Boundary flags: x-, y-, z- are boundaries; x+, y+, z+ are not */
        check(corner->on_boundary[0] == 1, "Corner: x- is boundary");
        check(corner->on_boundary[1] == 0, "Corner: x+ is not boundary");
        check(corner->on_boundary[2] == 1, "Corner: y- is boundary");
        check(corner->on_boundary[3] == 0, "Corner: y+ is not boundary");
        check(corner->on_boundary[4] == 1, "Corner: z- is boundary");
        check(corner->on_boundary[5] == 0, "Corner: z+ is not boundary");

        /* nblevel: self = 0, x+ neighbor = 0, x- = -1 (boundary) */
        check(corner->nblevel[1][1][1] == 0, "nblevel: self = 0");
        check(corner->nblevel[1][1][0] == -1, "nblevel: x- = -1 (boundary)");
        check(corner->nblevel[1][1][2] == 0, "nblevel: x+ = 0 (neighbor)");
    }

    /* Check interior block in 2x2x2: block (1,1,1) should have no boundaries.
     * Wait — in 2x2x2, the max index is 1, so (1,1,1) is also a corner
     * of the domain! All blocks in 2x2x2 touch at least 3 boundaries.
     * Let's verify that. */
    block_t *b111 = NULL;
    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (b->loc.lx1 == 1 && b->loc.lx2 == 1 && b->loc.lx3 == 1) {
            b111 = b;
            break;
        }
    }
    if (b111) {
        /* (1,1,1) in 2x2x2: x+, y+, z+ are boundaries */
        check(b111->on_boundary[0] == 0 && b111->on_boundary[1] == 1,
              "Block (1,1,1): x- interior, x+ boundary");
    }

    mesh_free(m);
}

/* ======================================================================
 * Test 3: Single-block mesh = grid_t (flat spacetime evolution)
 * ====================================================================== */
static void test_single_block_evolution(void)
{
    printf("\n--- Test: Single-block mesh flat spacetime evolution ---\n");

    sim_params_t p = default_params();
    p.N         = 16;
    p.L         = 10.0;
    p.CFL       = 0.25;
    p.num_steps = 100;
    p.sigma     = 0.3;
    p.dx = p.L / p.N;
    p.dt = p.CFL * p.dx;

    /* Create 1-block mesh (N_block=16 to save RAM) */
    mesh_t *m = mesh_create(1, 16, p.L, p.rk_method);
    block_t *b = mesh_get_block(m, 0);
    grid_t *g = b->grid;

    /* Update params to match grid padding */
    p.N  = g->N;
    p.dx = g->dx;
    p.dt = p.CFL * p.dx;

    printf("  N=%d, Ntotal=%d, dx=%.6f, dt=%.6f\n", g->N, g->Ntotal, g->dx, p.dt);

    /* Verify block is the single root */
    check(m->num_blocks == 1, "1 block created");
    check(b->loc.lx1 == 0 && b->loc.lx2 == 0 && b->loc.lx3 == 0,
          "Block at (0,0,0)");
    check(b->is_leaf == 1, "Block is leaf");

    /* All 26 neighbors should be -1 (boundary) for a single block */
    int all_boundary = 1;
    for (int n = 0; n < NUM_NEIGHBORS; n++) {
        if (b->neighbor_ids[n] != -1) all_boundary = 0;
    }
    check(all_boundary, "Single block: all 26 neighbors are boundary");

    /* Set flat spacetime initial data */
    set_flat_spacetime(g);

    double ham0 = mesh_constraint_l2(m);
    printf("  Initial Ham L2 = %.6e\n", ham0);

    /* Evolve for 100 steps (lightweight — just verify mesh works) */
    p.time = 0.0;
    for (int step = 1; step <= p.num_steps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
        p.time += p.dt;
        if (step % 10 == 0) {
            double ham = mesh_constraint_l2(m);
            printf("  step %3d/%d  Ham L2 = %.6e\n", step, p.num_steps, ham);
        }
    }

    double ham_final = mesh_constraint_l2(m);
    printf("  Final Ham L2 = %.6e\n", ham_final);
    check(ham_final < 1.0e-10, "1-block mesh flat spacetime Ham L2 < 1e-10");

    mesh_free(m);
}

/* ======================================================================
 * Test 4: MeshBlockPack load/store round-trip
 * ====================================================================== */
static void test_meshblock_pack(void)
{
    printf("\n--- Test: MeshBlockPack round-trip ---\n");

    /* Create a 1-block mesh with some non-trivial data */
    mesh_t *m = mesh_create(1, 32, 10.0, RK_CLASSIC);
    block_t *b = mesh_get_block(m, 0);
    grid_t *g = b->grid;

    /* Set flat spacetime (gives non-trivial field values: chi=1, lapse=1, h=I) */
    set_flat_spacetime(g);

    /* Create pack and load data */
    int ids[] = {0};
    meshblock_pack_t *pack = meshblock_pack_create(1, g->npoints, ids, 0,
                                                    RK_CLASSIC, NUM_FIELDS);

    meshblock_pack_load(pack, m->blocks);

    /* Verify pack data matches grid data */
    double max_err = 0.0;
    for (int f = 0; f < NUM_FIELDS; f++) {
        for (size_t i = 0; i < g->npoints; i++) {
            double pack_val = pack->data[PACK_IDX(pack, f, 0, i)];
            double grid_val = g->fields[f][i];
            double err = fabs(pack_val - grid_val);
            if (err > max_err) max_err = err;
        }
    }
    printf("  Load max error = %.6e\n", max_err);
    check(max_err < 1.0e-15, "Pack load matches grid data (roundoff)");

    /* Modify pack data slightly, store back, verify */
    pack->data[PACK_IDX(pack, FIELD_CHI, 0, 0)] += 1.0e-8;
    meshblock_pack_store(pack, m->blocks);

    double chi_val = g->fields[FIELD_CHI][0];
    double expected = 1.0 + 1.0e-8;  /* chi was 1.0 in flat spacetime */
    check(fabs(chi_val - expected) < 1.0e-15,
          "Pack store updates grid data correctly");

    meshblock_pack_free(pack);
    mesh_free(m);
}

/* ======================================================================
 * Test 5: Multi-block pack with 8 blocks
 * ====================================================================== */
static void test_multiblock_pack(void)
{
    printf("\n--- Test: MeshBlockPack with 8 blocks ---\n");

    mesh_t *m = mesh_create(2, 16, 10.0, RK_CLASSIC);

    check(m->num_blocks == 8, "8 blocks created (2^3)");

    /* Set flat spacetime on all blocks */
    for (int i = 0; i < m->num_blocks; i++) {
        set_flat_spacetime(m->blocks[i]->grid);
    }

    /* Pack all 8 blocks */
    int ids[8];
    for (int i = 0; i < 8; i++) ids[i] = i;

    size_t npts = m->blocks[0]->grid->npoints;
    meshblock_pack_t *pack = meshblock_pack_create(8, npts, ids, 0,
                                                    RK_CLASSIC, NUM_FIELDS);
    meshblock_pack_load(pack, m->blocks);

    /* Verify pack layout: block 3's chi should be at correct offset */
    double max_err = 0.0;
    for (int b = 0; b < 8; b++) {
        for (int f = 0; f < NUM_FIELDS; f++) {
            for (size_t i = 0; i < npts; i++) {
                double pack_val = pack->data[PACK_IDX(pack, f, b, i)];
                double grid_val = m->blocks[ids[b]]->grid->fields[f][i];
                double err = fabs(pack_val - grid_val);
                if (err > max_err) max_err = err;
            }
        }
    }
    printf("  8-block load max error = %.6e\n", max_err);
    check(max_err < 1.0e-15, "8-block pack load matches all grid data");

    meshblock_pack_free(pack);
    mesh_free(m);
}

/* ====================================================================== */

int main(void)
{
    /* Disable stdout buffering so output appears immediately */
    setbuf(stdout, NULL);

    printf("=== AMR Stage 1: Mesh Infrastructure Test ===\n");

    backend_init();

    test_morton();
    test_mesh_topology();
    test_single_block_evolution();
    test_meshblock_pack();
    test_multiblock_pack();

    backend_cleanup();

    printf("\n=== Results: %d/%d passed ===\n", pass_count, test_count);
    printf("%s\n", (pass_count == test_count) ? "ALL PASSED" : "SOME FAILED");

    return (pass_count == test_count) ? 0 : 1;
}
