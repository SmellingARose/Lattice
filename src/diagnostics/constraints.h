/*
 * Lattice — 3D Numerical Relativity
 * Hamiltonian and momentum constraint diagnostics.
 *
 * Ref: GRChombo Source/CCZ4/NewConstraints.impl.hpp
 */

#ifndef LATTICE_CONSTRAINTS_H
#define LATTICE_CONSTRAINTS_H

#include "../core/grid.h"

/* Hamiltonian constraint at one point:
 * H = R + (2/3)*K^2 - A^{ij}*A_{ij}
 * Ref: GRChombo NewConstraints.impl.hpp:55-61 */
double compute_hamiltonian_at(const double *const *fields, const grid_t *g,
                              int i, int j, int k);

/* Momentum constraint at one point: returns M_i (3 components).
 * M_i = -(2/3) d_i(K) + h^{jk} covd_k(A_{ji}) - 3/(2 chi) h^{jk} A_{ij} d_k(chi)
 * Ref: GRChombo NewConstraints.impl.hpp:72-99 */
void compute_momentum_at(const double *const *fields, const grid_t *g,
                          int i, int j, int k, double mom[3]);

/* L2 norm of Hamiltonian constraint over the interior domain */
double compute_constraint_l2(const grid_t *g);

/* L2 norm of momentum constraint (rms of |M_i|) over interior */
double compute_momentum_l2(const grid_t *g);

#endif /* LATTICE_CONSTRAINTS_H */
