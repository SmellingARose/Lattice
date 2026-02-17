/*
 * Lattice — 3D Numerical Relativity
 * 4th-order cell-average restriction (fine → coarse).
 *
 * Symmetric 4-point Lagrange stencil (exact for degree ≤ 3):
 *   w = { 1/48, 23/48, 23/48, 1/48 }
 * 3D tensor product: 4³ = 64 fine cells per coarse cell.
 *
 * Falls back to 2nd-order (8-cell average) if stencil exits fine grid.
 *
 * Ref: ExaHyPE (arXiv:2504.15814) — upgrading restriction to match
 *      prolongation order eliminates Hamiltonian violations at AMR boundaries.
 * Ref: GRChombo CoarseAverage (Chombo library) — 2nd-order baseline.
 * Ref: AthenaK Lagrange restriction matching FD order.
 */

#ifndef LATTICE_RESTRICTION_H
#define LATTICE_RESTRICTION_H

#include "../core/grid.h"

/* Forward declaration for restrict_to_coarse_buf */
struct block_s;

/* 4-point symmetric Lagrange restriction stencil.
 * Fine positions relative to coarse center: {-3δ/2, -δ/2, +δ/2, +3δ/2}.
 * Weights are cell-average integrals of Lagrange basis polynomials
 * over the coarse cell: (1/Δx_c) ∫ L_j(x) dx.
 *
 * Ref: Fornberg's FD weight algorithm (SIAM Review 40, 1998) */
#define RESTRICT_STENCIL 4

extern const double restrict_w[RESTRICT_STENCIL];

/* Restrict a single field from fine grid to coarse grid.
 * Fine grid has 2x the resolution (N_fine = 2 * N_coarse).
 * Same physical domain. Fills coarse interior points.
 * Uses 4th-order stencil with 2nd-order fallback at boundaries. */
void restrict_field(const grid_t *fine_g, int fine_field,
                    grid_t *coarse_g, int coarse_field);

/* Restrict all NUM_FIELDS from fine grid to coarse grid. */
void restrict_all(const grid_t *fine_g, grid_t *coarse_g);

/* Restrict fine block interior → block's own coarse_buf interior (4th-order).
 * Block-local operation: no cross-block memory access.
 * Ref: AthenaK coarse-buffer architecture */
void restrict_to_coarse_buf(struct block_s *b);

#endif /* LATTICE_RESTRICTION_H */
