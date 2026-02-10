/*
 * backend_cpu.c — CPU backend with OpenMP parallelism
 *
 * Field update loops use collapse(2) on outer two dimensions (z, y)
 * with x as the innermost (unit-stride) loop for cache efficiency.
 */

#include "backend.h"

int backend_init(void)
{
    return 0;
}

void backend_shutdown(void)
{
}

void backend_field_update(grid_t *g, double *dst, const double *src,
                          const double *rhs, double dt_factor, int field_id)
{
    (void)field_id;
    const int nx = g->params.nx;
    const int ny = g->params.ny;
    const int nz = g->params.nz;
    const int sx = grid_stride_x(g);
    const int sy = grid_stride_y(g);
    const int sz = grid_stride_z(g);

#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = i * sx + j * sy + k * sz;
                dst[idx] = src[idx] + dt_factor * rhs[idx];
            }
        }
    }
}

void backend_sync_to_device(grid_t *g)
{
    (void)g;
}

void backend_sync_to_host(grid_t *g)
{
    (void)g;
}
