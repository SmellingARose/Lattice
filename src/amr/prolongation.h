/*
 * Lattice — 3D Numerical Relativity
 * 6th-order cell-centered Lagrange prolongation (coarse → fine).
 *
 * Each coarse cell maps to 8 fine children via tensor-product interpolation
 * using a 7-point 1D stencil. Right-child weights are left-child reversed.
 *
 * Ref: AthenaK src/mesh/prolongation.hpp (HighOrderProlongCC)
 * Ref: Fornberg, SIAM Review 40 (1998) — Lagrange interpolation weights
 */

#ifndef LATTICE_PROLONGATION_H
#define LATTICE_PROLONGATION_H

#include "../core/device.h"
#include "../core/grid.h"

EXTERN_C_BEGIN

/* 7-point 1D stencil for 6th-order cell-centered Lagrange interpolation.
 * Left child at x = -1/4 from coarse cell center, nodes at {-3..+3}.
 * Right child weights = reversed left child weights.
 *
 * Weights derived via SymPy (tools/compute_amr_weights.py):
 *   w[0] =   273/65536   (coarse cell at -3)
 *   w[1] = -1287/32768   (coarse cell at -2)
 *   w[2] = 15015/65536   (coarse cell at -1)
 *   w[3] = 15015/16384   (coarse cell at  0, center)
 *   w[4] = -9009/65536   (coarse cell at +1)
 *   w[5] =  1001/32768   (coarse cell at +2)
 *   w[6] =  -231/65536   (coarse cell at +3)
 */
#define PROLONG_STENCIL 7

/* Ghost width for the coarse buffer: ensures the 7-point prolongation stencil
 * can fill ALL fine ghost cells, including the outermost (fi=0).
 * Derived from: ghost_c >= PROLONG_STENCIL/2 + ceil(GHOST_WIDTH/2).
 * With GHOST_WIDTH=4, PROLONG_STENCIL=7: ghost_c >= 3 + 2 = 5.
 * Ref: AthenaK coarse-buffer ghost width (must exceed fine ghost / ratio + stencil half). */
#define COARSE_BUF_GHOST (PROLONG_STENCIL / 2 + (GHOST_WIDTH + 1) / 2)

extern const double prolong_w[PROLONG_STENCIL];
extern const double prolong_wkj[4][PROLONG_STENCIL][PROLONG_STENCIL];

/* Prolongate a single field from coarse grid to fine grid.
 * Fine grid has 2x the resolution (N_fine = 2 * N_coarse, dx_fine = dx_coarse/2).
 * Same physical domain. Fills fine interior points from coarse data.
 * Requires ghost >= PROLONG_STENCIL/2 = 3 on coarse grid (have 4). */
void prolongate_field(const grid_t *coarse_g, int coarse_field,
                      grid_t *fine_g, int fine_field);

/* Prolongate all NUM_FIELDS from coarse grid to fine grid. */
void prolongate_all(const grid_t *coarse_g, grid_t *fine_g);

EXTERN_C_END

#endif /* LATTICE_PROLONGATION_H */
