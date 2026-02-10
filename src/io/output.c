/*
 * output.c — I/O routines for field dumps and scalar output
 *
 * Binary dumps: header + raw double array
 * Slice output: 2D tab-separated text at z = nz/2
 * Scalar output: append to scalars.dat
 */

#include "../core/grid.h"
#include "../core/fields.h"

#include <stdio.h>
#include <string.h>

/*
 * Write binary dump of all fields: header + raw data.
 * Header: nx, ny, nz (int), time (double), then field data.
 */
void output_fields(grid_t *g)
{
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/fields_%06d.dat",
             g->params.output_dir, g->step);

    FILE *fp = fopen(filename, "wb");
    if (!fp) return;

    int dims[3] = { g->params.nx, g->params.ny, g->params.nz };
    fwrite(dims, sizeof(int), 3, fp);
    fwrite(&g->time, sizeof(double), 1, fp);

    int n = grid_total_points(g);
    for (int f = 0; f < g->params.num_fields; f++) {
        fwrite(g->fields[f], sizeof(double), (size_t)n, fp);
    }

    fclose(fp);
}

/*
 * Write a 2D slice (z = nz/2) of a single field as tab-separated text.
 * Format: x\ty\tf(x,y)\n
 */
void output_slice(grid_t *g, int field_id, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) return;

    int k_mid = g->params.nz / 2;
    double *field = g->fields[field_id];

    for (int j = g->params.ghost_width; j < g->params.ny - g->params.ghost_width; j++) {
        for (int i = g->params.ghost_width; i < g->params.nx - g->params.ghost_width; i++) {
            int idx = grid_idx(g, i, j, k_mid);
            fprintf(fp, "%.6e\t%.6e\t%.15e\n",
                    grid_x(g, i), grid_y(g, j), field[idx]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

/*
 * Append scalar diagnostics to scalars.dat.
 * Format: step  time  ham_l2  mom_l2  alpha_min
 */
void output_scalars(grid_t *g, double ham_l2, double mom_l2, double alpha_min)
{
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/scalars.dat", g->params.output_dir);

    FILE *fp = fopen(filename, "a");
    if (!fp) return;

    fprintf(fp, "%6d  %.6e  %.6e  %.6e  %.6e\n",
            g->step, g->time, ham_l2, mom_l2, alpha_min);

    fclose(fp);
}
