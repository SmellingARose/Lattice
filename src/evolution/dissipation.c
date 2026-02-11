/*
 * Lattice — 3D Numerical Relativity
 * Kreiss-Oliger dissipation.
 *
 * Adds sigma * sum_dir(fd_ko) to each field's RHS at point (i,j,k).
 * 6th-order operator using 7-point stencil.
 *
 * Ref: GRChombo FourthOrderDerivatives.hpp:361-415
 */

#include "../core/fields.h"
#include "../core/grid.h"
#include "../numerics/finite_diff.h"

#ifdef LATTICE_GPU
#pragma omp declare target
#endif
void add_ko_dissipation(double **rhs, const double *const *src,
                        const grid_t *g, double sigma,
                        int i, int j, int k)
{
    int idx = IDX(g, i, j, k);
    int sx = STRIDE_X;
    int sy = STRIDE_Y(g);
    int sz = STRIDE_Z(g);
    double dx = g->dx;

    for (int f = 0; f < NUM_FIELDS; f++) {
        rhs[f][idx] += sigma * (fd_ko(src[f], idx, sx, dx)
                              + fd_ko(src[f], idx, sy, dx)
                              + fd_ko(src[f], idx, sz, dx));
    }
}
#ifdef LATTICE_GPU
#pragma omp end declare target
#endif
