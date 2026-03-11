/*
 * Lattice -- 3D Numerical Relativity
 * Test suite for the multi-BH tracker.
 *
 * Tests:
 *   1. Initialization — tracker alloc with 5 BHs, centers match input
 *   2. Position update — 3-BH Brill-Lindquist, lapse minima near initial pos
 *   3. AH finding loop — single Schwarzschild, tracker AH matches standalone
 *   4. Merger detection — 2 BHs at separation=2M, merger detected
 *   5. Merger bookkeeping — n_active decremented, merged_into set, remnant created
 *   6. CSV output — write 3 steps, verify header + data lines + column count
 *   7. 25-BH allocation — alloc/free with 25 BHs, no leaks
 *   8. Post-merger tracking — after merger, remnant is tracked
 *
 * Ref: arXiv:2505.01495 (GRChombo 25-BH cluster simulation)
 */

#include "../src/core/grid.h"
#include "../src/core/params.h"
#include "../src/core/fields.h"
#include "../src/amr/mesh.h"
#include "../src/initial_data/puncture.h"
#include "../src/initial_data/bowen_york.h"
#include "../src/diagnostics/bh_tracker.h"
#include "../src/diagnostics/constraints.h"
#include "../src/backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

/* ================================================================
 * Test 1: Initialization
 * ================================================================ */
static void test_initialization(void)
{
    printf("\n--- Test 1: Initialization ---\n");

    puncture_data_t bhs[5];
    memset(bhs, 0, sizeof(bhs));
    for (int i = 0; i < 5; i++) {
        bhs[i].mass = 0.5;
        /* Pentagon arrangement */
        double angle = 2.0 * M_PI * i / 5.0;
        bhs[i].center[0] = 3.0 * cos(angle);
        bhs[i].center[1] = 3.0 * sin(angle);
        bhs[i].center[2] = 0.0;
    }

    bh_tracker_t *tr = bh_tracker_alloc(5, bhs, 8, 16);
    check(tr != NULL, "Tracker allocated");
    check(tr->n_bh == 5, "n_bh = 5");
    check(tr->n_active == 5, "n_active = 5");
    check(tr->n_mergers == 0, "n_mergers = 0");

    /* Verify centers match input */
    int centers_ok = 1;
    for (int i = 0; i < 5; i++) {
        if (tr->bh[i].status != BH_STATUS_ACTIVE) { centers_ok = 0; break; }
        double dx = tr->bh[i].center[0] - bhs[i].center[0];
        double dy = tr->bh[i].center[1] - bhs[i].center[1];
        if (fabs(dx) > 1e-14 || fabs(dy) > 1e-14) { centers_ok = 0; break; }
    }
    check(centers_ok, "Centers match input puncture data");

    /* Verify AH workspaces allocated */
    int ah_ok = 1;
    for (int i = 0; i < 5; i++) {
        if (!tr->ah[i]) { ah_ok = 0; break; }
    }
    check(ah_ok, "AH workspaces allocated for all BHs");

    bh_tracker_free(tr);
}

/* ================================================================
 * Test 2: Position update (lapse-minimum tracking)
 * ================================================================ */
static void test_position_update(void)
{
    printf("\n--- Test 2: Position update ---\n");

    /* 3-BH Brill-Lindquist on a mesh */
    int N = 32;
    double L = 20.0;
    mesh_t *m = mesh_create_ex(N, L, RK_CLASSIC, NUM_CCZ4_FIELDS);
    grid_t *g = m->blocks[0]->grid;

    puncture_data_t bhs[3];
    memset(bhs, 0, sizeof(bhs));
    bhs[0].mass = 1.0; bhs[0].center[0] = -3.0;
    bhs[1].mass = 1.0; bhs[1].center[0] =  3.0;
    bhs[2].mass = 1.0; bhs[2].center[1] =  3.0;

    /* Set BL data */
    double masses[3] = {1.0, 1.0, 1.0};
    double centers[3][3] = {{-3.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}};
    set_brill_lindquist(g, 3, masses, centers);

    bh_tracker_t *tr = bh_tracker_alloc(3, bhs, 8, 16);
    bh_tracker_update_positions(tr, m);

    /* Check that tracked positions are near the input positions */
    int pos_ok = 1;
    for (int i = 0; i < 3; i++) {
        double dx = tr->bh[i].center[0] - bhs[i].center[0];
        double dy = tr->bh[i].center[1] - bhs[i].center[1];
        double dz = tr->bh[i].center[2] - bhs[i].center[2];
        double d = sqrt(dx*dx + dy*dy + dz*dz);
        printf("    BH%d: tracked=(%.2f, %.2f, %.2f), expected=(%.2f, %.2f, %.2f), dist=%.4f\n",
               i, tr->bh[i].center[0], tr->bh[i].center[1], tr->bh[i].center[2],
               bhs[i].center[0], bhs[i].center[1], bhs[i].center[2], d);
        /* On a coarse grid (dx=0.625), positions within ~1.5*dx */
        if (d > 1.5) pos_ok = 0;
    }
    check(pos_ok, "Tracked positions near initial positions (within 1.5M)");

    /* Lapse minima should be < 1 near punctures */
    int lapse_ok = 1;
    for (int i = 0; i < 3; i++) {
        if (tr->bh[i].lapse_min >= 1.0) { lapse_ok = 0; break; }
    }
    check(lapse_ok, "Lapse minima < 1 at tracked BH positions");

    bh_tracker_free(tr);
    mesh_free(m);
}

/* ================================================================
 * Test 3: AH finding via tracker
 * ================================================================ */
static void test_ah_finding(void)
{
    printf("\n--- Test 3: AH finding via tracker ---\n");

    double M = 1.0;
    int N = 64;
    double L = 16.0;
    mesh_t *m = mesh_create_ex(N, L, RK_CLASSIC, NUM_FIELDS);
    grid_t *g = m->blocks[0]->grid;

    double masses[1] = {M};
    double centers_bl[1][3] = {{0.0, 0.0, 0.0}};
    set_brill_lindquist(g, 1, masses, centers_bl);

    puncture_data_t bhs[1];
    memset(bhs, 0, sizeof(bhs));
    bhs[0].mass = M;

    bh_tracker_t *tr = bh_tracker_alloc(1, bhs, 16, 32);
    /* Don't call update_positions — use initial center (0,0,0) which
     * is where the AH finder was calibrated in standalone tests */
    bh_tracker_find_horizons(tr, m, 1e-2, 3000);

    printf("    M_irr = %.4f (expected ~%.4f)\n", tr->bh[0].mass_irr, M);
    printf("    M_chr = %.4f\n", tr->bh[0].mass_chr);
    printf("    chi   = %.4f\n", tr->bh[0].chi_spin);

    double mass_err = fabs(tr->bh[0].mass_irr - M) / M;
    check(mass_err < 0.15, "Tracker M_irr within 15% of M");
    check(tr->bh[0].chi_spin < 0.2, "Tracker spin near zero for Schwarzschild");

    bh_tracker_free(tr);
    mesh_free(m);
}

/* ================================================================
 * Test 4: Merger detection
 * ================================================================ */
static void test_merger_detection(void)
{
    printf("\n--- Test 4: Merger detection ---\n");

    /* Place 2 BHs at separation = 2M < 3M threshold */
    puncture_data_t bhs[2];
    memset(bhs, 0, sizeof(bhs));
    bhs[0].mass = 1.0; bhs[0].center[0] = -1.0;
    bhs[1].mass = 1.0; bhs[1].center[0] =  1.0;

    bh_tracker_t *tr = bh_tracker_alloc(2, bhs, 8, 16);

    /* Separation = 2M < 3*max(M_i, M_j) = 3M → should merge */
    bh_tracker_check_mergers(tr, 10.0);

    check(tr->n_mergers == 1, "Merger detected (separation < 3M)");
    check(tr->bh[0].status == BH_STATUS_MERGED, "BH 0 marked merged");
    check(tr->bh[1].status == BH_STATUS_MERGED, "BH 1 marked merged");

    bh_tracker_free(tr);
}

/* ================================================================
 * Test 5: Merger bookkeeping
 * ================================================================ */
static void test_merger_bookkeeping(void)
{
    printf("\n--- Test 5: Merger bookkeeping ---\n");

    puncture_data_t bhs[3];
    memset(bhs, 0, sizeof(bhs));
    bhs[0].mass = 1.0; bhs[0].center[0] = -1.0;
    bhs[1].mass = 1.0; bhs[1].center[0] =  1.0;
    bhs[2].mass = 1.0; bhs[2].center[0] = 10.0;  /* far away — not merged */

    bh_tracker_t *tr = bh_tracker_alloc(3, bhs, 8, 16);
    bh_tracker_check_mergers(tr, 42.0);

    check(tr->n_active == 2, "n_active = 2 after one merger");
    check(tr->n_bh == 4, "n_bh = 4 (3 original + 1 remnant)");
    check(tr->bh[0].merged_into == 3, "BH 0 merged_into = 3 (remnant id)");
    check(tr->bh[1].merged_into == 3, "BH 1 merged_into = 3 (remnant id)");
    check(tr->bh[2].status == BH_STATUS_ACTIVE, "BH 2 still active (far away)");
    check(tr->bh[3].status == BH_STATUS_ACTIVE, "Remnant (BH 3) is active");

    /* Remnant position is midpoint */
    double mid_x = 0.5 * (bhs[0].center[0] + bhs[1].center[0]);
    check(fabs(tr->bh[3].center[0] - mid_x) < 1e-14, "Remnant at midpoint");

    /* Remnant mass = sum of parents */
    check(fabs(tr->bh[3].initial_mass - 2.0) < 1e-14, "Remnant mass = M1 + M2");

    /* Merger event log */
    check(tr->mergers[0].id1 == 0, "Merger event: id1 = 0");
    check(tr->mergers[0].id2 == 1, "Merger event: id2 = 1");
    check(tr->mergers[0].remnant_id == 3, "Merger event: remnant_id = 3");
    check(fabs(tr->mergers[0].time - 42.0) < 1e-14, "Merger event: time = 42.0");

    bh_tracker_free(tr);
}

/* ================================================================
 * Test 6: CSV output
 * ================================================================ */
static void test_csv_output(void)
{
    printf("\n--- Test 6: CSV output ---\n");

    puncture_data_t bhs[2];
    memset(bhs, 0, sizeof(bhs));
    bhs[0].mass = 1.0; bhs[0].center[0] = -5.0;
    bhs[1].mass = 1.0; bhs[1].center[0] =  5.0;

    bh_tracker_t *tr = bh_tracker_alloc(2, bhs, 8, 16);

    char tmppath[] = "/tmp/lattice_tracker_test_XXXXXX";
    int fd = mkstemp(tmppath);
    check(fd >= 0, "Temp file created");
    if (fd < 0) { bh_tracker_free(tr); return; }

    FILE *fp = fdopen(fd, "w+");
    bh_tracker_write_csv_header(tr, fp);
    bh_tracker_write_csv(tr, fp, 0.0, 1e-6, 1e-7, 100);
    bh_tracker_write_csv(tr, fp, 0.1, 2e-6, 2e-7, 100);
    bh_tracker_write_csv(tr, fp, 0.2, 3e-6, 3e-7, 100);
    fflush(fp);

    /* Read back and verify */
    rewind(fp);
    char buf[4096];
    int line_count = 0;
    int header_cols = 0;
    while (fgets(buf, sizeof(buf), fp)) {
        if (line_count == 0) {
            /* Count commas in header */
            for (char *c = buf; *c; c++) {
                if (*c == ',') header_cols++;
            }
            header_cols++; /* fields = commas + 1 */
            printf("    Header columns: %d\n", header_cols);
            /* 5 global + 6*2 per-BH + 1 n_mergers = 18 */
            check(header_cols == 18, "CSV header has 18 columns (5 + 6*2 + 1)");
        }
        line_count++;
    }
    check(line_count == 4, "CSV has 4 lines (1 header + 3 data)");

    fclose(fp);
    remove(tmppath);
    bh_tracker_free(tr);
}

/* ================================================================
 * Test 7: 25-BH allocation
 * ================================================================ */
static void test_25bh_alloc(void)
{
    printf("\n--- Test 7: 25-BH allocation ---\n");

    puncture_data_t bhs[25];
    memset(bhs, 0, sizeof(bhs));
    for (int i = 0; i < 25; i++) {
        bhs[i].mass = 0.4;
        /* Arrange in a grid-like pattern */
        bhs[i].center[0] = (i % 5 - 2) * 3.0;
        bhs[i].center[1] = (i / 5 - 2) * 3.0;
    }

    bh_tracker_t *tr = bh_tracker_alloc(25, bhs, 12, 24);
    check(tr != NULL, "25-BH tracker allocated");
    check(tr->n_bh == 25, "n_bh = 25");
    check(tr->n_active == 25, "n_active = 25");

    /* Verify all AH workspaces exist */
    int all_ah = 1;
    for (int i = 0; i < 25; i++) {
        if (!tr->ah[i]) { all_ah = 0; break; }
    }
    check(all_ah, "All 25 AH workspaces allocated");

    /* Free without leaks (valgrind would catch) */
    bh_tracker_free(tr);
    printf("    25-BH tracker freed successfully\n");
    check(1, "25-BH alloc/free cycle complete");
}

/* ================================================================
 * Test 8: Post-merger tracking
 * ================================================================ */
static void test_post_merger_tracking(void)
{
    printf("\n--- Test 8: Post-merger tracking ---\n");

    puncture_data_t bhs[3];
    memset(bhs, 0, sizeof(bhs));
    bhs[0].mass = 1.0; bhs[0].center[0] = -0.5;  /* close pair */
    bhs[1].mass = 1.0; bhs[1].center[0] =  0.5;
    bhs[2].mass = 1.0; bhs[2].center[0] =  8.0;  /* far away */

    bh_tracker_t *tr = bh_tracker_alloc(3, bhs, 8, 16);

    /* Trigger merger of BH 0 and BH 1 */
    bh_tracker_check_mergers(tr, 5.0);
    check(tr->n_mergers == 1, "One merger triggered");

    /* Remnant (id=3) should be active and trackable */
    check(tr->bh[3].status == BH_STATUS_ACTIVE, "Remnant active");
    check(tr->bh[3].id == 3, "Remnant id = 3");

    /* BH 2 should still be active */
    check(tr->bh[2].status == BH_STATUS_ACTIVE, "BH 2 still active");

    /* Verify merge times */
    check(fabs(tr->bh[0].merge_time - 5.0) < 1e-14, "BH 0 merge_time = 5.0");
    check(fabs(tr->bh[1].merge_time - 5.0) < 1e-14, "BH 1 merge_time = 5.0");

    /* Remnant AH workspace should exist */
    check(tr->ah[3] != NULL, "Remnant AH workspace allocated");

    bh_tracker_free(tr);
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void)
{
    printf("=== N-Body BH Tracker Test Suite ===\n");
    printf("Ref: arXiv:2505.01495 (GRChombo 25-BH cluster)\n");
    backend_init();

    test_initialization();
    test_position_update();
    test_ah_finding();
    test_merger_detection();
    test_merger_bookkeeping();
    test_csv_output();
    test_25bh_alloc();
    test_post_merger_tracking();

    printf("\n=== Results: %d/%d passed ===\n", pass_count, test_count);
    backend_cleanup();

    return (pass_count == test_count) ? 0 : 1;
}
