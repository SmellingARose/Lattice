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

#include "../core/grid.h"

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

extern const double prolong_w[PROLONG_STENCIL];

/* Pre-computed prolong_w[wk]*prolong_w[wj] for 4 octant combos.
 * combo = ok*2 + oj, where wk = ok ? (6-sk) : sk, wj = oj ? (6-sj) : sj.
 * Eliminates one multiply + two conditional index computations per middle loop.
 * Verified bit-exact against runtime computation (tools/verify_weights.c). */
extern const double prolong_wkj[4][PROLONG_STENCIL][PROLONG_STENCIL];

/* Prolongate a single field from coarse grid to fine grid.
 * Fine grid has 2x the resolution (N_fine = 2 * N_coarse, dx_fine = dx_coarse/2).
 * Same physical domain. Fills fine interior points from coarse data.
 * Requires ghost >= PROLONG_STENCIL/2 = 3 on coarse grid (have 4). */
void prolongate_field(const grid_t *coarse_g, int coarse_field,
                      grid_t *fine_g, int fine_field);

/* Prolongate all NUM_FIELDS from coarse grid to fine grid. */
void prolongate_all(const grid_t *coarse_g, grid_t *fine_g);

#endif /* LATTICE_PROLONGATION_H */
