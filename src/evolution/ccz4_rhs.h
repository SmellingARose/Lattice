/*
 * Lattice — 3D Numerical Relativity
 * CCZ4 right-hand-side prototype.
 *
 * Ref: arXiv:1106.2254 (CCZ4 equations)
 * Ref: GRChombo Source/CCZ4/CCZ4RHS.impl.hpp
 */

#ifndef LATTICE_CCZ4_RHS_H
#define LATTICE_CCZ4_RHS_H

#include "../core/grid.h"
#include "../core/params.h"

/* Compute the full CCZ4 RHS at a single grid point (i,j,k).
 * Includes: CCZ4 evolution, moving puncture gauge, KO dissipation.
 * Writes results into rhs arrays. */
#ifdef LATTICE_GPU
#pragma omp declare target
#endif
void ccz4_rhs_point(double ** restrict rhs,
                    const double *const * restrict src,
                    const grid_t *g, const sim_params_t *p,
                    int i, int j, int k);
#ifdef LATTICE_GPU
#pragma omp end declare target
#endif

#endif /* LATTICE_CCZ4_RHS_H */
