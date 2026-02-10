/*
 * sommerfeld.c — Radiative (Sommerfeld) outflow boundary conditions
 *
 * For each ghost point on all 6 faces, apply the outgoing-wave condition:
 *   (d/dt + d/dr) (r * (f - f0)) = 0
 *   => f = f0 + (f_int - f0) * r_int/r + (r_int/r) * df_int * (r - r_int)
 *
 * where f0 is the asymptotic background value and f_int is the nearest
 * interior point. This allows outgoing waves to leave the domain cleanly.
 *
 * Uses field_background_value() and field_falloff_power() from fields.h.
 * Ref: B&S Ch. 12.1.3
 */

#include "../core/grid.h"
#include "../core/fields.h"

#include <math.h>

/*
 * Apply Sommerfeld boundary conditions to ghost zones of all evolved fields.
 * Uses simple radial extrapolation: f(ghost) = f0 + (f(interior) - f0) * r_int / r_ghost
 *
 * This implements a first-order radiative condition that matches the 1/r falloff.
 */
void sommerfeld_apply(grid_t *g)
{
    const int nx = g->params.nx;
    const int ny = g->params.ny;
    const int nz = g->params.nz;
    const int gw = g->params.ghost_width;
    const int nf = g->params.num_fields;

    for (int f = 0; f < nf; f++) {
        double f0 = field_background_value(f);
        double *field = g->rk_scratch[f];

        /* x-faces */
        /* OMP: boundary fill on x-faces, independent per (k,j) slice.
         * Toggle: make PARALLEL=0/1 */
#ifdef LATTICE_USE_OMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                /* x-lower face */
                for (int gi = 0; gi < gw; gi++) {
                    int idx_ghost = grid_idx(g, gi, j, k);
                    int idx_int = grid_idx(g, gw, j, k);

                    double xg = grid_x(g, gi);
                    double yg = grid_y(g, j);
                    double zg = grid_z(g, k);
                    double rg = sqrt(xg * xg + yg * yg + zg * zg);
                    if (rg < 1e-10) rg = 1e-10;

                    double xi = grid_x(g, gw);
                    double ri = sqrt(xi * xi + yg * yg + zg * zg);
                    if (ri < 1e-10) ri = 1e-10;

                    field[idx_ghost] = f0 + (field[idx_int] - f0) * ri / rg;
                }

                /* x-upper face */
                for (int gi = nx - gw; gi < nx; gi++) {
                    int idx_ghost = grid_idx(g, gi, j, k);
                    int idx_int = grid_idx(g, nx - gw - 1, j, k);

                    double xg = grid_x(g, gi);
                    double yg = grid_y(g, j);
                    double zg = grid_z(g, k);
                    double rg = sqrt(xg * xg + yg * yg + zg * zg);
                    if (rg < 1e-10) rg = 1e-10;

                    double xi = grid_x(g, nx - gw - 1);
                    double ri = sqrt(xi * xi + yg * yg + zg * zg);
                    if (ri < 1e-10) ri = 1e-10;

                    field[idx_ghost] = f0 + (field[idx_int] - f0) * ri / rg;
                }
            }
        }

        /* y-faces (skip corners already done by x) */
        /* OMP: boundary fill on y-faces, independent per (k,i) slice.
         * Toggle: make PARALLEL=0/1 */
#ifdef LATTICE_USE_OMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int k = 0; k < nz; k++) {
            for (int i = gw; i < nx - gw; i++) {
                /* y-lower */
                for (int gj = 0; gj < gw; gj++) {
                    int idx_ghost = grid_idx(g, i, gj, k);
                    int idx_int = grid_idx(g, i, gw, k);

                    double xg = grid_x(g, i);
                    double yg = grid_y(g, gj);
                    double zg = grid_z(g, k);
                    double rg = sqrt(xg * xg + yg * yg + zg * zg);
                    if (rg < 1e-10) rg = 1e-10;

                    double yi = grid_y(g, gw);
                    double ri = sqrt(xg * xg + yi * yi + zg * zg);
                    if (ri < 1e-10) ri = 1e-10;

                    field[idx_ghost] = f0 + (field[idx_int] - f0) * ri / rg;
                }

                /* y-upper */
                for (int gj = ny - gw; gj < ny; gj++) {
                    int idx_ghost = grid_idx(g, i, gj, k);
                    int idx_int = grid_idx(g, i, ny - gw - 1, k);

                    double xg = grid_x(g, i);
                    double yg = grid_y(g, gj);
                    double zg = grid_z(g, k);
                    double rg = sqrt(xg * xg + yg * yg + zg * zg);
                    if (rg < 1e-10) rg = 1e-10;

                    double yi = grid_y(g, ny - gw - 1);
                    double ri = sqrt(xg * xg + yi * yi + zg * zg);
                    if (ri < 1e-10) ri = 1e-10;

                    field[idx_ghost] = f0 + (field[idx_int] - f0) * ri / rg;
                }
            }
        }

        /* z-faces (skip edges already done) */
        /* OMP: boundary fill on z-faces, independent per (j,i) slice.
         * Toggle: make PARALLEL=0/1 */
#ifdef LATTICE_USE_OMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int j = gw; j < ny - gw; j++) {
            for (int i = gw; i < nx - gw; i++) {
                /* z-lower */
                for (int gk = 0; gk < gw; gk++) {
                    int idx_ghost = grid_idx(g, i, j, gk);
                    int idx_int = grid_idx(g, i, j, gw);

                    double xg = grid_x(g, i);
                    double yg = grid_y(g, j);
                    double zg = grid_z(g, gk);
                    double rg = sqrt(xg * xg + yg * yg + zg * zg);
                    if (rg < 1e-10) rg = 1e-10;

                    double zi = grid_z(g, gw);
                    double ri = sqrt(xg * xg + yg * yg + zi * zi);
                    if (ri < 1e-10) ri = 1e-10;

                    field[idx_ghost] = f0 + (field[idx_int] - f0) * ri / rg;
                }

                /* z-upper */
                for (int gk = nz - gw; gk < nz; gk++) {
                    int idx_ghost = grid_idx(g, i, j, gk);
                    int idx_int = grid_idx(g, i, j, nz - gw - 1);

                    double xg = grid_x(g, i);
                    double yg = grid_y(g, j);
                    double zg = grid_z(g, gk);
                    double rg = sqrt(xg * xg + yg * yg + zg * zg);
                    if (rg < 1e-10) rg = 1e-10;

                    double zi = grid_z(g, nz - gw - 1);
                    double ri = sqrt(xg * xg + yg * yg + zi * zi);
                    if (ri < 1e-10) ri = 1e-10;

                    field[idx_ghost] = f0 + (field[idx_int] - f0) * ri / rg;
                }
            }
        }
    }
}
