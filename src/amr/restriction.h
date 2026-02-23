/*
 * Lattice — 3D Numerical Relativity
 * 6th-order cell-average restriction (fine → coarse).
 *
 * Symmetric 6-point Lagrange stencil (exact for degree ≤ 5):
 *   w = { -17/11520, 97/3840, 2743/5760, 2743/5760, 97/3840, -17/11520 }
 * 3D tensor product: 6³ = 216 fine cells per coarse cell.
 *
 * The 6-point stencil reaches at most 2 cells into ghost zones, which always
 * contain valid data (ghost width = 4). No fallback needed.
 *
 * Ref: ExaHyPE (arXiv:2504.15814) — upgrading restriction to match
 *      prolongation order eliminates Hamiltonian violations at AMR boundaries.
 * Ref: GRChombo CoarseAverage (Chombo library) — 0th-order baseline.
 * Ref: Fornberg, SIAM Review 40 (1998) — FD weight generation algorithm.
 */

#ifndef LATTICE_RESTRICTION_H
#define LATTICE_RESTRICTION_H

#include "../core/grid.h"

/* Forward declaration for restrict_to_coarse_buf */
struct block_s;

/* 6-point symmetric Lagrange restriction stencil.
 * Fine cell centers at {-5δ/2, -3δ/2, -δ/2, +δ/2, +3δ/2, +5δ/2}
 * where δ = dx_fine = dx_coarse/2.
 *
 * Weights are cell-average integrals of degree-5 Lagrange basis polynomials
 * over the coarse cell: (1/Δx_c) ∫_{-Δx_c/2}^{+Δx_c/2} L_j(x) dx.
 * Small negative outer weights (necessary for 6th-order accuracy).
 *
 * Derived via SymPy (tools/compute_amr_weights.py).
 * Ref: Fornberg, SIAM Review 40 (1998) */
#define RESTRICT_STENCIL 6

extern const double restrict_w[RESTRICT_STENCIL];

/* Restrict a single field from fine grid to coarse grid.
 * Fine grid has 2x the resolution (N_fine = 2 * N_coarse).
 * Same physical domain. Fills coarse interior points.
 * Uses 6th-order stencil for all cells (stencil fits within ghost zones). */
void restrict_field(const grid_t *fine_g, int fine_field,
                    grid_t *coarse_g, int coarse_field);

/* Restrict all NUM_FIELDS from fine grid to coarse grid. */
void restrict_all(const grid_t *fine_g, grid_t *coarse_g);

/* Restrict fine block interior → block's own coarse_buf interior (6th-order).
 * Block-local operation: no cross-block memory access.
 * Ref: AthenaK coarse-buffer architecture */
void restrict_to_coarse_buf(struct block_s *b);

#endif /* LATTICE_RESTRICTION_H */
