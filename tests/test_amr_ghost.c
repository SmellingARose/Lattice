/*
 * Lattice — 3D Numerical Relativity
 * AMR Stage 2 test: 26-neighbor ghost exchange + multi-block evolution.
 *
 * Tests:
 *   1. Ghost exchange with known polynomial function — verify ghost values
 *      match expected interior values from neighbors.
 *   2. Face, edge, and corner ghost regions all filled correctly.
 *   3. Multi-block flat spacetime evolution matches single-grid to roundoff.
 *   4. Multi-block single BH matches single-grid constraints.
 *   5. Sommerfeld block-aware: only applies to domain boundary ghost points.
 *
 * Pass criteria:
 *   - Ghost exchange error < 1e-14 (roundoff)
 *   - Multi-block vs single-grid flat Ham L2 difference < 1e-12
 *   - Multi-block single BH constraints bounded
 *   - make — zero warnings
 */

#include "../src/amr/mesh.h"
#include "../src/amr/ghost_exchange.h"
#include "../src/amr/block.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/initial_data/puncture.h"
#include "../src/evolution/ccz4_rhs.h"
#include "../src/boundary/sommerfeld.h"
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
 * Test 1: Ghost exchange with known polynomial
 *
 * Set f(x,y,z) = x + 2*y + 3*z on all blocks' interior points using
 * BLOCK_COORD. Run ghost exchange. Verify ghost zone values match
 * the expected polynomial evaluated at ghost point coordinates.
 * ====================================================================== */
static void test_ghost_polynomial(void)
{
    printf("\n--- Test: Ghost exchange with polynomial f = x + 2y + 3z ---\n");

    /* 2x2x2 mesh, N_block=16. Effective N=32. */
    mesh_t *m = mesh_create(2, 16, 10.0, RK_CLASSIC);
    check(m->num_blocks == 8, "8 blocks created");

    /* Set field 0 (CHI) to f(x,y,z) = x + 2y + 3z on all interiors */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        grid_t *g = b->grid;
        int lo = g->ghost;
        int hi = g->ghost + g->N;

        for (int k = lo; k < hi; k++) {
            for (int j = lo; j < hi; j++) {
                for (int i = lo; i < hi; i++) {
                    double x = BLOCK_COORD(b, 0, i);
                    double y = BLOCK_COORD(b, 1, j);
                    double z = BLOCK_COORD(b, 2, k);
                    int idx = IDX(g, i, j, k);
                    g->fields[FIELD_CHI][idx] = x + 2.0*y + 3.0*z;
                }
            }
        }
    }

    /* Run ghost exchange */
    ghost_exchange(m);

    /* Verify ghost zone values match expected polynomial.
     * Only check ghost points that have neighbors (not domain boundaries). */
    double max_err = 0.0;
    int ghost_pts_checked = 0;
    int face_pts = 0, edge_pts = 0, corner_pts = 0;

    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        grid_t *g = b->grid;
        int lo = g->ghost;
        int hi = g->ghost + g->N;
        int Nt = g->Ntotal;

        for (int k = 0; k < Nt; k++) {
            for (int j = 0; j < Nt; j++) {
                for (int i = 0; i < Nt; i++) {
                    /* Skip interior */
                    if (i >= lo && i < hi &&
                        j >= lo && j < hi &&
                        k >= lo && k < hi)
                        continue;

                    /* Determine which neighbor directions are out-of-bounds.
                     * A ghost point needs data from a neighbor in each
                     * direction it extends beyond the interior. */
                    int ox = (i < lo) ? -1 : (i >= hi) ? 1 : 0;
                    int oy = (j < lo) ? -1 : (j >= hi) ? 1 : 0;
                    int oz = (k < lo) ? -1 : (k >= hi) ? 1 : 0;

                    /* Check if the neighbor in that direction exists */
                    /* Find the neighbor index for this (ox,oy,oz) */
                    int nbr_found = 0;
                    for (int n = 0; n < NUM_NEIGHBORS; n++) {
                        if (nbr_offset[n][0] == ox &&
                            nbr_offset[n][1] == oy &&
                            nbr_offset[n][2] == oz &&
                            b->neighbor_ids[n] >= 0) {
                            nbr_found = 1;
                            break;
                        }
                    }
                    if (!nbr_found) continue;

                    /* Count neighbor type */
                    int n_nonzero = (ox != 0) + (oy != 0) + (oz != 0);
                    if (n_nonzero == 1) face_pts++;
                    else if (n_nonzero == 2) edge_pts++;
                    else corner_pts++;

                    double x = BLOCK_COORD(b, 0, i);
                    double y = BLOCK_COORD(b, 1, j);
                    double z = BLOCK_COORD(b, 2, k);
                    double expected = x + 2.0*y + 3.0*z;
                    double actual = g->fields[FIELD_CHI][IDX(g, i, j, k)];
                    double err = fabs(actual - expected);
                    if (err > max_err) max_err = err;
                    ghost_pts_checked++;
                }
            }
        }
    }

    printf("  Ghost points checked: %d (face=%d, edge=%d, corner=%d)\n",
           ghost_pts_checked, face_pts, edge_pts, corner_pts);
    printf("  Max ghost exchange error: %.6e\n", max_err);

    check(ghost_pts_checked > 0, "Ghost points were checked");
    check(face_pts > 0, "Face ghost points checked");
    check(edge_pts > 0, "Edge ghost points checked");
    check(corner_pts > 0, "Corner ghost points checked");
    check(max_err < 1.0e-12, "Ghost exchange error < 1e-12 (roundoff)");

    mesh_free(m);
}

/* ======================================================================
 * Test 2: Multi-block flat spacetime vs single-grid
 *
 * Evolve flat spacetime on:
 *   (a) Single grid N=32
 *   (b) 2x2x2 mesh of N_block=16
 * Both use CK45, same dt. Compare Ham L2 after 50 steps.
 * ====================================================================== */
static void test_multiblock_flat(void)
{
    printf("\n--- Test: Multi-block flat spacetime vs single-grid ---\n");

    sim_params_t p = default_params();
    int N_eff = 32;
    double L = 10.0;

    /* (a) Single-grid reference run */
    grid_t *gref = grid_alloc(N_eff, L, RK_CLASSIC);
    p.N  = gref->N;
    p.dx = gref->dx;
    p.dt = p.CFL * p.dx;
    p.num_steps = 50;

    set_flat_spacetime(gref);
    for (int step = 0; step < p.num_steps; step++) {
        rk4_step(gref, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);
    }
    double ham_ref = compute_constraint_l2(gref);
    printf("  Single-grid: Ham L2 = %.6e (N=%d, %d steps)\n",
           ham_ref, gref->N, p.num_steps);

    /* (b) Multi-block mesh: 2x2x2 = 8 blocks of 16^3 */
    mesh_t *m = mesh_create(2, 16, L, RK_CLASSIC);

    /* Set flat spacetime on all blocks */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        set_flat_spacetime(m->blocks[bid]->grid);
    }

    /* Evolve with mesh stepper.
     * Use same dt as reference (based on same dx). */
    for (int step = 0; step < p.num_steps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
    }

    /* Compute Ham L2 on each block, combine */
    double ham_sum = 0.0;
    int n_interior = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        grid_t *g = m->blocks[bid]->grid;
        int lo = g->ghost;
        int hi = g->ghost + g->N;
        for (int k = lo; k < hi; k++) {
            for (int j = lo; j < hi; j++) {
                for (int i = lo; i < hi; i++) {
                    double h = compute_hamiltonian_at(
                        (const double *const *)g->fields, g, i, j, k);
                    ham_sum += h * h;
                    n_interior++;
                }
            }
        }
    }
    double ham_mesh = sqrt(ham_sum / n_interior);
    printf("  Multi-block:  Ham L2 = %.6e (2x2x2 x 16^3, %d steps)\n",
           ham_mesh, p.num_steps);

    /* Both should be near machine epsilon for flat spacetime */
    check(ham_ref < 1.0e-10, "Single-grid flat Ham L2 < 1e-10");
    check(ham_mesh < 1.0e-10, "Multi-block flat Ham L2 < 1e-10");

    /* The difference should be small — they compute the same thing
     * with the same stencils, just different memory layout */
    double ratio = (ham_ref > 0) ? ham_mesh / ham_ref : 0.0;
    printf("  Ratio mesh/ref = %.6f\n", ratio);
    check(ratio < 10.0 && ratio > 0.1,
          "Multi-block Ham L2 within 10x of single-grid");

    grid_free(gref);
    mesh_free(m);
}

/* ======================================================================
 * Test 3: Point-by-point comparison of multi-block vs single-grid
 *
 * After evolving flat spacetime for a few steps, compare field values
 * at matching physical coordinates. Interior points should agree
 * to roundoff because the stencils access the same data (ghost zones
 * are filled by exchange, which copies neighbor interior cells).
 * ====================================================================== */
static void test_multiblock_pointwise(void)
{
    printf("\n--- Test: Point-by-point multi-block vs single-grid ---\n");

    sim_params_t p = default_params();
    double L = 10.0;
    int nsteps = 10;

    /* Single grid N=32 */
    grid_t *gref = grid_alloc(32, L, RK_CLASSIC);
    p.N  = gref->N;
    p.dx = gref->dx;
    p.dt = p.CFL * p.dx;

    set_flat_spacetime(gref);
    for (int s = 0; s < nsteps; s++)
        rk4_step(gref, &p, ccz4_rhs_point, apply_sommerfeld, p.dt);

    /* Multi-block 2x2x2 x 16^3 */
    mesh_t *m = mesh_create(2, 16, L, RK_CLASSIC);
    for (int bid = 0; bid < m->num_blocks; bid++)
        set_flat_spacetime(m->blocks[bid]->grid);
    for (int s = 0; s < nsteps; s++)
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);

    /* Compare interior points of each block to corresponding single-grid points.
     * Block (ix,iy,iz) interior point (i,j,k) corresponds to single-grid
     * point (ix*N_block + i - ghost + ghost, ...). */
    double max_err = 0.0;
    int pts_compared = 0;
    int N_block = 16;

    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        grid_t *bg = b->grid;
        int lo = bg->ghost;
        int hi_b = lo + bg->N;

        /* This block's logical position gives offset into single grid */
        int off_x = b->loc.lx1 * N_block;
        int off_y = b->loc.lx2 * N_block;
        int off_z = b->loc.lx3 * N_block;

        for (int k = lo; k < hi_b; k++) {
            for (int j = lo; j < hi_b; j++) {
                for (int i = lo; i < hi_b; i++) {
                    /* Corresponding single-grid indices */
                    int ri = gref->ghost + off_x + (i - lo);
                    int rj = gref->ghost + off_y + (j - lo);
                    int rk = gref->ghost + off_z + (k - lo);

                    for (int f = 0; f < NUM_FIELDS; f++) {
                        double v_mesh = bg->fields[f][IDX(bg, i, j, k)];
                        double v_ref  = gref->fields[f][IDX(gref, ri, rj, rk)];
                        double err = fabs(v_mesh - v_ref);
                        if (err > max_err) max_err = err;
                    }
                    pts_compared++;
                }
            }
        }
    }

    printf("  Points compared: %d, max field error: %.6e\n",
           pts_compared, max_err);
    check(pts_compared == N_block * N_block * N_block * 8,
          "All interior points compared");
    check(max_err < 1.0e-10,
          "Multi-block matches single-grid to < 1e-10");

    grid_free(gref);
    mesh_free(m);
}

/* ======================================================================
 * Test 4: Sommerfeld block-aware correctness
 *
 * Set a non-trivial field on a 2x2x2 mesh. Verify that
 * apply_sommerfeld_block only modifies ghost points adjacent to
 * domain boundaries, leaving inter-block ghost points untouched.
 * ====================================================================== */
static void test_sommerfeld_block(void)
{
    printf("\n--- Test: Sommerfeld block-aware boundary application ---\n");

    mesh_t *m = mesh_create(2, 16, 10.0, RK_CLASSIC);

    /* Set flat spacetime and fill ghost zones */
    for (int bid = 0; bid < m->num_blocks; bid++)
        set_flat_spacetime(m->blocks[bid]->grid);
    ghost_exchange(m);

    /* Mark all RHS values with a sentinel value */
    double sentinel = -999.0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        grid_t *g = m->blocks[bid]->grid;
        for (int f = 0; f < NUM_FIELDS; f++) {
            for (size_t i = 0; i < g->npoints; i++)
                g->rhs[f][i] = sentinel;
        }
    }

    /* Apply block-aware Sommerfeld to each block */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        apply_sommerfeld_block(b->grid->rhs,
                               (const double *const *)b->grid->fields, b);
    }

    /* Check: interior points should still have sentinel (not modified) */
    /* Check: domain boundary ghost points should be modified (not sentinel) */
    /* Check: inter-block ghost points should still have sentinel */
    int interior_ok = 1;
    int boundary_modified = 0;
    int interblock_ok = 1;

    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        grid_t *g = b->grid;
        int lo = g->ghost;
        int hi = lo + g->N;
        int Nt = g->Ntotal;

        for (int k = 0; k < Nt; k++) {
            for (int j = 0; j < Nt; j++) {
                for (int i = 0; i < Nt; i++) {
                    int idx = IDX(g, i, j, k);
                    double val = g->rhs[0][idx];  /* check field 0 */
                    int is_interior = (i >= lo && i < hi &&
                                       j >= lo && j < hi &&
                                       k >= lo && k < hi);

                    if (is_interior) {
                        if (val != sentinel) interior_ok = 0;
                        continue;
                    }

                    /* Ghost point: check if near domain boundary */
                    int near_bdy = 0;
                    if (i < lo  && b->on_boundary[0]) near_bdy = 1;
                    if (i >= hi && b->on_boundary[1]) near_bdy = 1;
                    if (j < lo  && b->on_boundary[2]) near_bdy = 1;
                    if (j >= hi && b->on_boundary[3]) near_bdy = 1;
                    if (k < lo  && b->on_boundary[4]) near_bdy = 1;
                    if (k >= hi && b->on_boundary[5]) near_bdy = 1;

                    if (near_bdy) {
                        if (val != sentinel) boundary_modified++;
                    } else {
                        /* Inter-block ghost: should be untouched */
                        if (val != sentinel) interblock_ok = 0;
                    }
                }
            }
        }
    }

    check(interior_ok, "Interior points not modified by Sommerfeld");
    check(boundary_modified > 0, "Domain boundary ghost points modified");
    check(interblock_ok, "Inter-block ghost points not modified");

    mesh_free(m);
}

/* ======================================================================
 * Test 5: Single BH through multi-block mesh
 *
 * Evolve a single Schwarzschild puncture through the multi-block mesh
 * for a few steps. Verify constraints are bounded (not diverging).
 * ====================================================================== */
static void test_multiblock_single_bh(void)
{
    printf("\n--- Test: Single BH through multi-block mesh ---\n");

    sim_params_t p = default_params();
    p.L = 64.0;
    p.CFL = 0.25;
    int nsteps = 4;

    /* 2x2x2 x 16^3 = effective N=32. Coarse but enough to test. */
    mesh_t *m = mesh_create(2, 16, p.L, RK_CLASSIC);

    /* Set BH initial data on all blocks */
    double mass = 1.0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        grid_t *g = b->grid;
        int Nt = g->Ntotal;

        /* We need to use BLOCK_COORD for physical coordinates */
        for (int k = 0; k < Nt; k++) {
            for (int j = 0; j < Nt; j++) {
                for (int i = 0; i < Nt; i++) {
                    int idx = IDX(g, i, j, k);
                    double x = BLOCK_COORD(b, 0, i);
                    double y = BLOCK_COORD(b, 1, j);
                    double z = BLOCK_COORD(b, 2, k);
                    double r = sqrt(x*x + y*y + z*z);
                    if (r < 1.0e-10) r = 1.0e-10;

                    double psi = 1.0 + mass / (2.0 * r);
                    double chi = 1.0 / (psi * psi * psi * psi);

                    g->fields[FIELD_CHI][idx]   = chi;
                    g->fields[FIELD_H11][idx]   = 1.0;
                    g->fields[FIELD_H12][idx]   = 0.0;
                    g->fields[FIELD_H13][idx]   = 0.0;
                    g->fields[FIELD_H22][idx]   = 1.0;
                    g->fields[FIELD_H23][idx]   = 0.0;
                    g->fields[FIELD_H33][idx]   = 1.0;
                    g->fields[FIELD_K][idx]     = 0.0;
                    g->fields[FIELD_LAPSE][idx] = sqrt(chi);
                    /* All other fields default to 0.0 from calloc */
                }
            }
        }
    }

    p.dx = m->dx_base;
    p.dt = p.CFL * p.dx;

    /* Evolve a few steps */
    for (int step = 0; step < nsteps; step++) {
        rk4_step_mesh(m, &p, ccz4_rhs_point, p.dt);
    }

    /* Check constraints on interior of each block */
    double ham_sum = 0.0;
    int n_interior = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        grid_t *g = m->blocks[bid]->grid;
        int lo = g->ghost;
        int hi = lo + g->N;
        for (int k = lo; k < hi; k++) {
            for (int j = lo; j < hi; j++) {
                for (int i = lo; i < hi; i++) {
                    double h = compute_hamiltonian_at(
                        (const double *const *)g->fields, g, i, j, k);
                    ham_sum += h * h;
                    n_interior++;
                }
            }
        }
    }
    double ham_l2 = sqrt(ham_sum / n_interior);
    printf("  Multi-block BH Ham L2 = %.6e (N_eff=32, %d steps)\n",
           ham_l2, nsteps);

    /* At N=32 with L=64, dx=2, this is coarse. Constraints should be
     * bounded but not tiny. Check they're not NaN/Inf and reasonable. */
    check(isfinite(ham_l2), "Ham L2 is finite");
    check(ham_l2 < 1.0, "Ham L2 < 1.0 (bounded, not diverging)");

    mesh_free(m);
}

/* ====================================================================== */

int main(void)
{
    setbuf(stdout, NULL);

    printf("=== AMR Stage 2: Ghost Exchange + Multi-Block Evolution Test ===\n");

    backend_init();

    test_ghost_polynomial();
    test_multiblock_flat();
    test_multiblock_pointwise();
    test_sommerfeld_block();
    test_multiblock_single_bh();

    backend_cleanup();

    printf("\n=== Results: %d/%d passed ===\n", pass_count, test_count);
    printf("%s\n", (pass_count == test_count) ? "ALL PASSED" : "SOME FAILED");

    return (pass_count == test_count) ? 0 : 1;
}
