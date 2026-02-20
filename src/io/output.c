/*
 * Lattice — 3D Numerical Relativity
 * 1D slice CSV output along x-axis through domain center.
 *
 * Single-grid: output_1d_slice() — direct interior traversal.
 * AMR mesh:    output_mesh_1d_slice() — iterate leaf blocks, find
 *              cells on the y=0,z=0 line, sort by x, keep finest level.
 */

#include "../core/grid.h"
#include "../core/fields.h"
#include "../diagnostics/constraints.h"
#include "../amr/mesh.h"
#include "../amr/block.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void output_1d_slice(const grid_t *g, int step, double time)
{
    (void)time;
    char filename[256];
    snprintf(filename, sizeof(filename), "build/slice_%06d.csv", step);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "output: cannot open %s\n", filename);
        return;
    }

    fprintf(fp, "x,chi,lapse,K,Hamiltonian\n");

    int jc = g->ghost + g->N / 2;
    int kc = g->ghost + g->N / 2;

    for (int i = g->ghost; i < g->ghost + g->N; i++) {
        double x = COORD(g, i);
        int idx = IDX(g, i, jc, kc);

        double chi   = g->fields[FIELD_CHI][idx];
        double alpha = g->fields[FIELD_LAPSE][idx];
        double K_val = g->fields[FIELD_K][idx];
        double Ham   = compute_hamiltonian_at(
            (const double *const *)g->fields, g, i, jc, kc);

        fprintf(fp, "%.8e,%.8e,%.8e,%.8e,%.8e\n", x, chi, alpha, K_val, Ham);
    }

    fclose(fp);
}

/* ========================================================================
 * AMR mesh slice output
 *
 * Extract a 1D slice along the x-axis through (y=0, z=0) from a mesh
 * of potentially overlapping blocks at different refinement levels.
 *
 * Algorithm:
 *   1. For each leaf block, check if the x-axis passes through it
 *      (block must contain y=0 and z=0 in its interior).
 *   2. Extract interior cells along that line: (x, chi, lapse, K, Ham, level, dx).
 *   3. Sort all collected points by x coordinate.
 *   4. De-duplicate overlapping coarse/fine data: keep finest level at each x.
 *   5. Write CSV in same format as single-grid.
 *
 * Ref: GRChombo PlotFile output pattern (finest-available data)
 * ======================================================================== */

/* Temporary struct for collecting slice data points */
typedef struct {
    double x;
    double chi;
    double lapse;
    double K;
    double Ham;
    int    level;   /* refinement level (higher = finer) */
    double dx;      /* grid spacing at this level        */
} slice_point_t;

/* Comparison for qsort: sort by x coordinate */
static int compare_slice_x(const void *a, const void *b)
{
    const slice_point_t *pa = a;
    const slice_point_t *pb = b;
    if (pa->x < pb->x) return -1;
    if (pa->x > pb->x) return  1;
    return 0;
}

void output_mesh_1d_slice(const mesh_t *m, int step, double time)
{
    (void)time;

    /* Estimate max points: sum of N_block across all leaf blocks */
    int max_pts = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (b && b->is_leaf)
            max_pts += b->grid->N;
    }
    if (max_pts == 0) return;

    slice_point_t *pts = malloc((size_t)max_pts * sizeof(slice_point_t));
    if (!pts) return;
    int npts = 0;

    /* Collect points from all leaf blocks whose interior contains y=0, z=0 */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;

        grid_t *g = b->grid;
        double dx = g->dx;
        int N = g->N;
        int ghost = g->ghost;

        /* Block interior spans [origin, origin + N*dx) in each direction.
         * BLOCK_COORD(b, dir, ghost) = origin + 0.5*dx  (first interior cell center)
         * BLOCK_COORD(b, dir, ghost+N-1) = origin + (N-0.5)*dx  (last cell center)
         * Check if y=0 and z=0 fall within the interior cell range. */
        double y_lo = b->origin[1];
        double y_hi = b->origin[1] + N * dx;
        double z_lo = b->origin[2];
        double z_hi = b->origin[2] + N * dx;

        if (0.0 < y_lo || 0.0 >= y_hi) continue;
        if (0.0 < z_lo || 0.0 >= z_hi) continue;

        /* Find the j,k indices closest to y=0, z=0 */
        int jc = ghost + (int)((0.0 - b->origin[1]) / dx);
        int kc = ghost + (int)((0.0 - b->origin[2]) / dx);
        if (jc < ghost) jc = ghost;
        if (jc >= ghost + N) jc = ghost + N - 1;
        if (kc < ghost) kc = ghost;
        if (kc >= ghost + N) kc = ghost + N - 1;

        /* Extract interior cells along x */
        for (int i = ghost; i < ghost + N; i++) {
            double x = BLOCK_COORD(b, 0, i);
            int idx = IDX(g, i, jc, kc);

            if (npts >= max_pts) break;  /* safety */
            pts[npts].x     = x;
            pts[npts].chi   = g->fields[FIELD_CHI][idx];
            pts[npts].lapse = g->fields[FIELD_LAPSE][idx];
            pts[npts].K     = g->fields[FIELD_K][idx];
            pts[npts].Ham   = compute_hamiltonian_at(
                (const double *const *)g->fields, g, i, jc, kc);
            pts[npts].level = b->loc.level;
            pts[npts].dx    = dx;
            npts++;
        }
    }

    if (npts == 0) {
        free(pts);
        return;
    }

    /* Sort by x coordinate */
    qsort(pts, npts, sizeof(slice_point_t), compare_slice_x);

    /* De-duplicate: when coarse and fine points overlap (within 0.5*dx_coarse),
     * keep only the finest-level data.
     * Simple approach: mark coarse points for removal when a fine point is nearby. */
    int *keep = malloc((size_t)npts * sizeof(int));
    if (!keep) { free(pts); return; }
    for (int i = 0; i < npts; i++) keep[i] = 1;

    for (int i = 0; i < npts; i++) {
        if (!keep[i]) continue;
        for (int j = i + 1; j < npts; j++) {
            if (pts[j].x - pts[i].x > pts[i].dx * 0.6) break;
            /* Points at nearly same x: keep the finer one */
            if (fabs(pts[j].x - pts[i].x) < fmin(pts[i].dx, pts[j].dx) * 0.6) {
                if (pts[i].level < pts[j].level)
                    keep[i] = 0;
                else if (pts[j].level < pts[i].level)
                    keep[j] = 0;
            }
        }
    }

    /* Write CSV */
    char filename[256];
    snprintf(filename, sizeof(filename), "build/slice_%06d.csv", step);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        free(pts);
        free(keep);
        return;
    }

    fprintf(fp, "x,chi,lapse,K,Hamiltonian\n");
    for (int i = 0; i < npts; i++) {
        if (!keep[i]) continue;
        fprintf(fp, "%.8e,%.8e,%.8e,%.8e,%.8e\n",
                pts[i].x, pts[i].chi, pts[i].lapse,
                pts[i].K, pts[i].Ham);
    }

    fclose(fp);
    free(pts);
    free(keep);
}
