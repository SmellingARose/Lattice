/*
 * Lattice — 3D Numerical Relativity
 * Sommerfeld radiative boundary conditions.
 *
 * Ref: GRChombo Source/GRChomboCore/BoundaryConditions.cpp:593-661
 */

#ifndef LATTICE_SOMMERFELD_H
#define LATTICE_SOMMERFELD_H

#include "../core/grid.h"
#include "../core/params.h"

/* Apply Sommerfeld boundary conditions to RHS arrays.
 * Sets RHS in ghost zones to enforce 1/r falloff toward asymptotic values.
 * src: the current state arrays (fields or scratch) that the RHS was computed from. */
void apply_sommerfeld(double **rhs, const double *const *src, const grid_t *g);

#endif /* LATTICE_SOMMERFELD_H */
