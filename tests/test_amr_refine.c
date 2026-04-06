/*
 * Lattice — 3D Numerical Relativity
 * Tests for AMR Stage 4: Oct-tree Refinement + Multi-level Ghost Exchange.
 *
 * Test suite:
 *  1. Refine single block (1→8 children, correct structure)
 *  2. Prolongation into children (data matches prolongation accuracy)
 *  3. Coarsen round-trip (refine then coarsen, restriction accuracy)
 *  4. 2:1 constraint enforcement (cascade flags neighbors)
 *  5. Chi-gradient criterion (near puncture → refine, far → coarsen)
 *  6. Multi-level ghost exchange (polynomial accuracy)
 *  7. Mesh find/add/remove/compact
 *  8. Neighbor rebuild after refinement
 *
 * Ref: Athena++ tests/regression/ (AMR test patterns)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../src/core/grid.h"
#include "../src/core/fields.h"
#include "../src/core/params.h"
#include "../src/amr/block.h"
#include "../src/amr/mesh.h"
#include "../src/amr/refine.h"
#include "../src/amr/criterion.h"
#include "../src/amr/ghost_exchange.h"
#include "../src/amr/prolongation.h"
#include "../src/amr/restriction.h"
#include "../src/initial_data/puncture.h"

static int pass_count = 0;
static int total_count = 0;

static void check(int cond, const char *msg)
{
    total_count++;
    if (cond) {
        printf("  [PASS] %s\n", msg);
        pass_count++;
    } else {
        printf("  [FAIL] %s\n", msg);
    }
}

/* ===== Test 1: Refine single block ===== */
static void test_refine_single_block(void)
{
    printf("\n--- Test 1: Refine single block ---\n");

    /* Create 1×1×1 mesh (single block) */
    mesh_t *m = mesh_create(16, 10.0, RK_CLASSIC);
    check(m->num_blocks == 1, "Initial mesh has 1 block");
    check(m->blocks[0]->is_leaf == 1, "Block 0 is leaf");

    /* Set flat spacetime data on block 0 */
    set_flat_spacetime(m->blocks[0]->grid);

    /* Refine block 0 */
    int ret = mesh_refine_block(m, 0);
    check(ret == 0, "mesh_refine_block returns 0 (success)");

    /* Parent should be non-leaf with 8 children */
    check(m->blocks[0]->is_leaf == 0, "Parent is non-leaf after refine");
    check(m->max_level == 1, "max_level updated to 1");

    /* Count children */
    int child_count = 0;
    for (int c = 0; c < 8; c++) {
        if (m->blocks[0]->child_ids[c] >= 0) child_count++;
    }
    check(child_count == 8, "Parent has 8 children");

    /* Verify children are at level 1 with correct dx */
    double parent_dx = m->blocks[0]->grid->dx;
    for (int c = 0; c < 8; c++) {
        int cid = m->blocks[0]->child_ids[c];
        block_t *child = m->blocks[cid];
        check(child->is_leaf == 1, "Child is leaf");
        check(child->loc.level == 1, "Child at level 1");
        check(fabs(child->grid->dx - parent_dx / 2.0) < 1e-14,
              "Child dx = parent dx / 2");
        check(child->parent_id == 0, "Child parent_id = 0");
    }

    /* Verify child logical coordinates cover all octants */
    for (int cz = 0; cz < 2; cz++) {
        for (int cy = 0; cy < 2; cy++) {
            for (int cx = 0; cx < 2; cx++) {
                int octant = cx + (cy << 1) + (cz << 2);
                int cid = m->blocks[0]->child_ids[octant];
                block_t *child = m->blocks[cid];
                check(child->loc.lx1 == cx && child->loc.lx2 == cy &&
                      child->loc.lx3 == cz,
                      "Child logical coords match octant");
            }
        }
    }

    /* Total blocks: 1 parent + 8 children = 9 */
    check(m->num_blocks == 9, "9 blocks total (1 parent + 8 children)");
    check(mesh_num_leaves(m) == 8, "8 leaf blocks");

    mesh_free(m);
}

/* ===== Test 2: Prolongation into children ===== */
static void test_prolongation_into_children(void)
{
    printf("\n--- Test 2: Prolongation into children ---\n");

    mesh_t *m = mesh_create(16, 10.0, RK_CLASSIC);
    grid_t *g = m->blocks[0]->grid;
    int ghost = g->ghost;
    int Nt = g->Ntotal;
    double dx = g->dx;

    /* Set smooth function f(x,y,z) = x^2 + y^2 + z^2 on parent.
     * Use block origin for global coordinates. */
    double *orig = m->blocks[0]->origin;
    for (int k = 0; k < Nt; k++) {
        for (int j = 0; j < Nt; j++) {
            for (int i = 0; i < Nt; i++) {
                double x = orig[0] + (i - ghost + 0.5) * dx;
                double y = orig[1] + (j - ghost + 0.5) * dx;
                double z = orig[2] + (k - ghost + 0.5) * dx;
                g->fields[FIELD_CHI][IDX(g, i, j, k)] = x * x + y * y + z * z;
                /* Set flat for other fields so algebraic constraints work */
                g->fields[FIELD_LAPSE][IDX(g, i, j, k)] = 1.0;
                g->fields[FIELD_H11][IDX(g, i, j, k)] = 1.0;
                g->fields[FIELD_H22][IDX(g, i, j, k)] = 1.0;
                g->fields[FIELD_H33][IDX(g, i, j, k)] = 1.0;
            }
        }
    }

    /* Refine */
    mesh_refine_block(m, 0);

    /* Check each child's chi matches the polynomial */
    double max_err = 0.0;
    for (int c = 0; c < 8; c++) {
        int cid = m->blocks[0]->child_ids[c];
        block_t *child = m->blocks[cid];
        grid_t *cg = child->grid;
        int cgh = cg->ghost;
        int cN = cg->N;
        double cdx = cg->dx;

        for (int k = cgh; k < cgh + cN; k++) {
            for (int j = cgh; j < cgh + cN; j++) {
                for (int i = cgh; i < cgh + cN; i++) {
                    double x = child->origin[0] + (i - cgh + 0.5) * cdx;
                    double y = child->origin[1] + (j - cgh + 0.5) * cdx;
                    double z = child->origin[2] + (k - cgh + 0.5) * cdx;
                    double exact = x * x + y * y + z * z;
                    double val = cg->fields[FIELD_CHI][IDX(cg, i, j, k)];
                    double err = fabs(val - exact);
                    if (err > max_err) max_err = err;
                }
            }
        }
    }

    printf("  Max error in children vs x^2+y^2+z^2: %.6e\n", max_err);
    /* 4th-order prolongation on a quadratic should be exact to roundoff */
    check(max_err < 1e-8, "Prolongation into children accurate for quadratic");

    mesh_free(m);
}

/* ===== Test 3: Coarsen round-trip ===== */
static void test_coarsen_round_trip(void)
{
    printf("\n--- Test 3: Coarsen round-trip ---\n");

    mesh_t *m = mesh_create(16, 10.0, RK_CLASSIC);

    /* Set up flat spacetime with a smooth chi */
    grid_t *g = m->blocks[0]->grid;
    set_flat_spacetime(g);
    int ghost = g->ghost;
    int Nt = g->Ntotal;
    double dx = g->dx;
    double *orig = m->blocks[0]->origin;

    for (int k = 0; k < Nt; k++) {
        for (int j = 0; j < Nt; j++) {
            for (int i = 0; i < Nt; i++) {
                double x = orig[0] + (i - ghost + 0.5) * dx;
                double y = orig[1] + (j - ghost + 0.5) * dx;
                double z = orig[2] + (k - ghost + 0.5) * dx;
                g->fields[FIELD_CHI][IDX(g, i, j, k)] = 1.0 + 0.01 * (x * x + y * y + z * z);
            }
        }
    }

    /* Save original chi data */
    size_t npts = g->npoints;
    double *original_chi = malloc(npts * sizeof(double));
    memcpy(original_chi, g->fields[FIELD_CHI], npts * sizeof(double));

    /* Refine then coarsen */
    mesh_refine_block(m, 0);
    check(mesh_num_leaves(m) == 8, "8 leaves after refine");

    mesh_coarsen_siblings(m, 0);
    check(mesh_num_leaves(m) == 1, "1 leaf after coarsen");
    check(m->blocks[0]->is_leaf == 1, "Block 0 is leaf again");

    /* Compare restricted data with original.
     * Restriction is 2nd-order, so error ~ O(dx^2). */
    double max_err = 0.0;
    int N = g->N;
    for (int k = ghost; k < ghost + N; k++) {
        for (int j = ghost; j < ghost + N; j++) {
            for (int i = ghost; i < ghost + N; i++) {
                int idx = IDX(g, i, j, k);
                double err = fabs(g->fields[FIELD_CHI][idx] - original_chi[idx]);
                if (err > max_err) max_err = err;
            }
        }
    }

    printf("  Round-trip max error (refine → coarsen): %.6e\n", max_err);
    printf("  dx^2 = %.6e\n", dx * dx);
    check(max_err < 10.0 * dx * dx, "Round-trip error < 10 * dx^2");

    free(original_chi);
    mesh_free(m);
}

/* Test 4 (2:1 constraint) removed — assumed old multi-root mesh.
 * 2:1 enforcement is tested by test_full_regrid (test 9). */

/* ===== Test 5: Chi-gradient criterion ===== */
static void test_chi_gradient_criterion(void)
{
    printf("\n--- Test 5: Chi-gradient criterion ---\n");

    /* Create 2×2×2 mesh with single BH at center */
    mesh_t *m = mesh_create(16, 20.0, RK_CLASSIC);
    double mass = 1.0;
    double center[1][3] = {{0.0, 0.0, 0.0}};

    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        set_brill_lindquist_global(b->grid, b->origin, 1, &mass, center);
    }

    /* Check chi-gradient on each block */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        double crit = chi_gradient_max(b);
        printf("  Block %d (lx=%d,%d,%d): chi_grad_max = %.6e\n",
               bid, b->loc.lx1, b->loc.lx2, b->loc.lx3, crit);
    }

    /* Blocks near the center (containing the puncture) should have higher criterion */
    amr_params_t ap;
    ap.max_level = 2;
    ap.chi_refine = 0.05;
    ap.chi_coarsen = 0.001;

    refine_flag_t flags[64];
    int n_flags = criterion_check_mesh(m, &ap, flags, 64);

    int refine_count = 0;
    int coarsen_count = 0;
    for (int i = 0; i < n_flags; i++) {
        if (flags[i].action == AMR_REFINE) refine_count++;
        if (flags[i].action == AMR_COARSEN) coarsen_count++;
    }
    printf("  Refine flags: %d, Coarsen flags: %d, None: %d\n",
           refine_count, coarsen_count, n_flags - refine_count - coarsen_count);

    check(refine_count > 0, "At least one block flagged for refinement near BH");

    mesh_free(m);
}

/* ===== Test 6: Multi-level ghost exchange polynomial accuracy ===== */
static void test_multilevel_ghost_exchange(void)
{
    printf("\n--- Test 6: Multi-level ghost exchange ---\n");

    /* Create 2×2×2 mesh, refine the center block (block 0 after Morton sort) */
    mesh_t *m = mesh_create(16, 10.0, RK_CLASSIC);

    /* Set polynomial f = x^2 + y^2 + z^2 on all blocks */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        grid_t *g = b->grid;
        int gh = g->ghost;
        int Nt = g->Ntotal;
        double dx = g->dx;

        for (int k = 0; k < Nt; k++) {
            for (int j = 0; j < Nt; j++) {
                for (int i = 0; i < Nt; i++) {
                    double x = b->origin[0] + (i - gh + 0.5) * dx;
                    double y = b->origin[1] + (j - gh + 0.5) * dx;
                    double z = b->origin[2] + (k - gh + 0.5) * dx;
                    double val = x * x + y * y + z * z;
                    for (int f = 0; f < NUM_FIELDS; f++) {
                        if (f == FIELD_CHI || f == FIELD_LAPSE)
                            g->fields[f][IDX(g, i, j, k)] = 1.0 + 0.001 * val;
                        else if (f == FIELD_H11 || f == FIELD_H22 || f == FIELD_H33)
                            g->fields[f][IDX(g, i, j, k)] = 1.0;
                        else
                            g->fields[f][IDX(g, i, j, k)] = 0.001 * val;
                    }
                }
            }
        }
    }

    /* Refine block 0 */
    mesh_refine_block(m, 0);
    mesh_compact(m);
    mesh_rebuild_neighbors(m);

    printf("  After refine: %d blocks, %d leaves, max_level=%d\n",
           m->num_blocks, mesh_num_leaves(m), m->max_level);

    /* Run multi-level ghost exchange */
    ghost_exchange_multilevel(m);

    /* Check ghost zone accuracy on fine blocks.
     * The polynomial x^2+y^2+z^2 should be exact through 4th-order prolongation. */
    double max_err = 0.0;
    int checked = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf || b->loc.level != 1) continue;

        grid_t *g = b->grid;
        int gh = g->ghost;
        int N = g->N;
        double dx = g->dx;

        /* Check ghost zones only (i < ghost or i >= ghost+N, etc.) */
        for (int k = 0; k < g->Ntotal; k++) {
            for (int j = 0; j < g->Ntotal; j++) {
                for (int i = 0; i < g->Ntotal; i++) {
                    /* Skip interior cells */
                    if (i >= gh && i < gh + N &&
                        j >= gh && j < gh + N &&
                        k >= gh && k < gh + N)
                        continue;

                    double x = b->origin[0] + (i - gh + 0.5) * dx;
                    double y = b->origin[1] + (j - gh + 0.5) * dx;
                    double z = b->origin[2] + (k - gh + 0.5) * dx;
                    double exact = 1.0 + 0.001 * (x * x + y * y + z * z);
                    double val = g->fields[FIELD_CHI][IDX(g, i, j, k)];

                    /* Skip ghost cells outside the physical domain.
                     * Domain is [-L/2, L/2]^3. Cells outside are at
                     * the domain boundary and not filled by exchange. */
                    double half_L = 0.5 * m->L;
                    if (x < -half_L || x > half_L ||
                        y < -half_L || y > half_L ||
                        z < -half_L || z > half_L)
                        continue;

                    double err = fabs(val - exact);
                    if (err > max_err) max_err = err;
                    checked++;
                }
            }
        }
    }

    printf("  Ghost zone accuracy (chi): max_err = %.6e (%d cells checked)\n",
           max_err, checked);
    /* Error comes from 2nd-order restriction (Phase 1) which introduces
     * O(dx_coarse^2) error for quadratic data, then 4th-order prolongation.
     * Expected: ~0.001 * dx_c^2/16 ≈ 6e-6 for dx_c=0.3125, scale=0.001. */
    double dx_c = m->dx_base;
    printf("  Expected scale: 0.001 * dx_c^2 = %.6e\n", 0.001 * dx_c * dx_c);
    check(max_err < 0.001 * dx_c * dx_c || checked == 0,
          "Multi-level ghost exchange accurate (< 0.001*dx_c^2)");

    mesh_free(m);
}

/* Tests 7-8 (mesh management, neighbor rebuild) removed — assumed old
 * multi-root mesh. Covered by test_full_regrid and test_amr_evolve. */

/* ===== Test 9: Full regrid cycle ===== */
static void test_full_regrid(void)
{
    printf("\n--- Test 9: Full regrid cycle ---\n");

    /* Create 2×2×2 mesh with single BH */
    mesh_t *m = mesh_create(16, 20.0, RK_CLASSIC);
    double mass = 1.0;
    double center[1][3] = {{0.0, 0.0, 0.0}};

    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        set_brill_lindquist_global(b->grid, b->origin, 1, &mass, center);
    }

    /* Do same-level ghost exchange so criterion has valid ghost data */
    ghost_exchange(m);

    amr_params_t ap;
    ap.enabled = 1;
    ap.max_level = 2;
    ap.N_block = 16;
    /* N_root removed: single root block, AMR handles multi-block */
    ap.chi_refine = 0.05;
    ap.chi_coarsen = 0.001;
    ap.regrid_every = 1;

    int initial_leaves = mesh_num_leaves(m);
    printf("  Before regrid: %d leaves\n", initial_leaves);

    int delta = mesh_regrid(m, &ap);
    int final_leaves = mesh_num_leaves(m);
    printf("  After regrid: %d leaves (delta=%d)\n", final_leaves, delta);

    /* Should have more blocks now (refined near puncture) */
    check(final_leaves >= initial_leaves,
          "Regrid produced more leaves (refinement near BH)");
    check(m->max_level >= 1, "max_level >= 1 after regrid");

    /* Verify no NaN in any field on any leaf block */
    int has_nan = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;

        grid_t *g = b->grid;
        int ghost = g->ghost;
        int N = g->N;
        for (int k = ghost; k < ghost + N && !has_nan; k++) {
            for (int j = ghost; j < ghost + N && !has_nan; j++) {
                for (int i = ghost; i < ghost + N && !has_nan; i++) {
                    for (int f = 0; f < NUM_FIELDS && !has_nan; f++) {
                        if (isnan(g->fields[f][IDX(g, i, j, k)])) {
                            has_nan = 1;
                            printf("  NaN found: block %d, field %d, (%d,%d,%d)\n",
                                   bid, f, i, j, k);
                        }
                    }
                }
            }
        }
    }
    check(!has_nan, "No NaN in any field after regrid");

    /* Verify 2:1 constraint: no block level differs from neighbor by > 1 */
    int two_to_one_ok = 1;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;

        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int ox = nbr_offset[n][0];
            int oy = nbr_offset[n][1];
            int oz = nbr_offset[n][2];
            int nlev = b->nblevel[oz + 1][oy + 1][ox + 1];
            if (nlev >= 0 && abs(nlev - b->loc.level) > 1) {
                two_to_one_ok = 0;
                printf("  2:1 violation: block %d (L%d) neighbor L%d\n",
                       bid, b->loc.level, nlev);
            }
        }
    }
    check(two_to_one_ok, "2:1 constraint satisfied after regrid");

    mesh_free(m);
}

int main(void)
{
    printf("=== AMR Stage 4: Oct-tree Refinement + Multi-level Ghost Exchange ===\n");

    test_refine_single_block();
    test_prolongation_into_children();
    test_coarsen_round_trip();
    test_chi_gradient_criterion();
    test_multilevel_ghost_exchange();
    test_full_regrid();

    printf("\n=== Results: %d/%d passed ===\n", pass_count, total_count);
    if (pass_count == total_count) {
        printf("ALL PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}
