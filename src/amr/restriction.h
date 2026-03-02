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

#include "../core/device.h"
#include "../core/grid.h"

EXTERN_C_BEGIN

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
extern const double restrict_wkj[RESTRICT_STENCIL][RESTRICT_STENCIL];

/*
 * Restrict a single coarse cell from fine data.
 * fi_base, fj_base, fk_base: fine grid indices of the first direct child.
 *
 * 6th-order: 3D tensor product of 6-point stencil (base-2 to base+3).
 * Stencil reaches at most 2 cells into ghost zones; with ghost width = 4
 * the bounds [0, Ntotal) always contain valid data. No fallback needed.
 */
static inline double restrict_cell(const double *src, const grid_t *fg,
                                    int fi_base, int fj_base, int fk_base)
{
    double val = 0.0;
    for (int sk = 0; sk < RESTRICT_STENCIL; sk++) {
        int fk = fk_base - 2 + sk;
        for (int sj = 0; sj < RESTRICT_STENCIL; sj++) {
            double wkj = restrict_wkj[sk][sj];
            int fj = fj_base - 2 + sj;
            for (int si = 0; si < RESTRICT_STENCIL; si++) {
                int fi = fi_base - 2 + si;
                val += wkj * restrict_w[si] * src[IDX(fg, fi, fj, fk)];
            }
        }
    }
    return val;
}

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

EXTERN_C_END

#endif /* LATTICE_RESTRICTION_H */
