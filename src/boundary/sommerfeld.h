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

/* Block-aware Sommerfeld: only applies to ghost points adjacent to domain
 * boundaries. Ghost points filled by inter-block exchange are left untouched.
 * Uses block origin for physical coordinates (correct for multi-block).
 *
 * For each ghost point, checks which directions are out-of-interior and
 * whether the corresponding on_boundary[] flag is set. If none are domain
 * boundaries, the point is skipped (filled by ghost exchange instead). */
struct block_s;  /* forward declaration */
void apply_sommerfeld_block(double **rhs, const double *const *src,
                            const struct block_s *b);

#endif /* LATTICE_SOMMERFELD_H */
