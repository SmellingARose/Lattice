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

/* L2 norm of Hamiltonian constraint over the interior domain */
double compute_constraint_l2(const grid_t *g);

#endif /* LATTICE_CONSTRAINTS_H */
