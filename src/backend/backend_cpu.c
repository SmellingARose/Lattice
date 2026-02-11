/*
 * Lattice — 3D Numerical Relativity
 * CPU backend: OpenMP parallel triple loop.
 *
 * z (outer, parallelized) -> y -> x (inner, unit stride)
 */

#include "backend.h"

void backend_compute_rhs(double **rhs, const double *const *src,
                         const grid_t *g, const sim_params_t *p,
                         rhs_point_func_t func)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = lo; k < hi; k++) {
        for (int j = lo; j < hi; j++) {
            for (int i = lo; i < hi; i++) {
                func(rhs, src, g, p, i, j, k);
            }
        }
    }
}

void backend_init(void) { /* no-op for CPU */ }
void backend_cleanup(void) { /* no-op for CPU */ }
