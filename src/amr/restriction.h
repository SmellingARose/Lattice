/*
 * Lattice — 3D Numerical Relativity
 * Volume-weighted restriction (fine → coarse).
 *
 * U_coarse = (1/8) * sum of 8 fine children overlapping the coarse cell.
 * 2nd-order accurate, conservative by construction.
 *
 * Ref: GRChombo CoarseAverage (Chombo library)
 * Ref: AthenaK also supports Lagrange restriction, but simple averaging
 *      is standard for NR codes (GRChombo, BAM, Carpet).
 */

#ifndef LATTICE_RESTRICTION_H
#define LATTICE_RESTRICTION_H

#include "../core/grid.h"

/* Restrict a single field from fine grid to coarse grid.
 * Fine grid has 2x the resolution (N_fine = 2 * N_coarse).
 * Same physical domain. Fills coarse interior points. */
void restrict_field(const grid_t *fine_g, int fine_field,
                    grid_t *coarse_g, int coarse_field);

/* Restrict all NUM_FIELDS from fine grid to coarse grid. */
void restrict_all(const grid_t *fine_g, grid_t *coarse_g);

#endif /* LATTICE_RESTRICTION_H */
