/*
 * Lattice — 3D Numerical Relativity
 * 4th-order cell-centered Lagrange prolongation (coarse → fine).
 *
 * Each coarse cell maps to 8 fine children via tensor-product interpolation
 * using a 5-point 1D stencil. Right-child weights are left-child reversed.
 *
 * Ref: AthenaK src/mesh/prolongation.hpp (HighOrderProlongCC)
 * Ref: AthenaK src/mesh/mesh_refinement.cpp InitInterpWghts() (weight values)
 */

#ifndef LATTICE_PROLONGATION_H
#define LATTICE_PROLONGATION_H

#include "../core/grid.h"

/* 5-point 1D stencil for 4th-order cell-centered Lagrange interpolation.
 * Left child at x = -1/4 from coarse cell center.
 * Right child weights = reversed left child weights.
 *
 * Values from AthenaK mesh_refinement.cpp:
 *   w[0] = -45/2048    (coarse cell at -2)
 *   w[1] =  105/512    (coarse cell at -1)
 *   w[2] =  945/1024   (coarse cell at  0, center)
 *   w[3] =  -63/512    (coarse cell at +1)
 *   w[4] =   35/2048   (coarse cell at +2)
 */
#define PROLONG_STENCIL 5

extern const double prolong_w[PROLONG_STENCIL];

/* Prolongate a single field from coarse grid to fine grid.
 * Fine grid has 2x the resolution (N_fine = 2 * N_coarse, dx_fine = dx_coarse/2).
 * Same physical domain. Fills fine interior points from coarse data.
 * Requires ghost >= PROLONG_STENCIL/2 = 2 on coarse grid. */
void prolongate_field(const grid_t *coarse_g, int coarse_field,
                      grid_t *fine_g, int fine_field);

/* Prolongate all NUM_FIELDS from coarse grid to fine grid. */
void prolongate_all(const grid_t *coarse_g, grid_t *fine_g);

#endif /* LATTICE_PROLONGATION_H */
