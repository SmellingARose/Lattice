/*
 * Lattice — 3D Numerical Relativity
 * GPU backend: OpenMP target offloading.
 *
 * Same triple loop as CPU, but with target teams distribute parallel for.
 * Calls ccz4_rhs_point directly — GPU offloading can't do function pointers.
 *
 * Build with: clang -fopenmp -fopenmp-targets=nvptx64 (NVIDIA)
 *          or gcc -fopenmp -foffload=nvptx-none        (GCC + NVIDIA)
 */

#include "backend.h"
#include "../evolution/ccz4_rhs.h"

void backend_compute_rhs(double **rhs, const double *const *src,
                         const grid_t *g, const sim_params_t *p,
                         rhs_point_func_t func)
{
    (void)func; /* GPU calls ccz4_rhs_point directly */

    int lo = g->ghost;
    int hi = g->ghost + g->N;

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = lo; k < hi; k++) {
        for (int j = lo; j < hi; j++) {
            for (int i = lo; i < hi; i++) {
                ccz4_rhs_point(rhs, src, g, p, i, j, k);
            }
        }
    }
}

void backend_init(void) { /* OpenMP runtime handles GPU init */ }
void backend_cleanup(void) { /* OpenMP runtime handles GPU cleanup */ }
