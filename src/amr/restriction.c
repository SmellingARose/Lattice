/*
 * Lattice — 3D Numerical Relativity
 * 6th-order cell-average restriction (fine → coarse).
 *
 * 1D symmetric 6-point Lagrange stencil (exact for degree ≤ 5):
 *   w = { -17/11520, 97/3840, 2743/5760, 2743/5760, 97/3840, -17/11520 }
 * 3D: tensor product of 1D weights, 6³ = 216 fine cells per coarse cell.
 *
 * Stencil reach: 2 fine cells beyond the 2 direct children per direction.
 * The stencil accesses [base-2, base+3], which stays within [0, Ntotal)
 * for all coarse cells since ghost width (4) exceeds stencil reach (2).
 *
 * Ref: ExaHyPE (arXiv:2504.15814) — upgrading restriction to match
 *      prolongation order eliminates Hamiltonian violations at AMR boundaries.
 * Ref: GRChombo CoarseAverage (Chombo library) — 0th-order baseline.
 * Ref: AthenaK Lagrange restriction matching FD order.
 * Ref: Fornberg, SIAM Review 40 (1998) — FD weight generation algorithm.
 */

#include "restriction.h"
#include "block.h"
#include "../core/fields.h"

/*
 * 6th-order cell-average restriction weights.
 * For coarse cell at position x_c, fine cells at:
 *   -5δ/2, -3δ/2, -δ/2, +δ/2, +3δ/2, +5δ/2 (δ = dx_fine).
 *
 * Derived by integrating degree-5 Lagrange basis polynomials over the coarse cell:
 *   w_j = (1/Δx_c) ∫_{-Δx_c/2}^{+Δx_c/2} L_j(x) dx
 *
 * Sum = 1, symmetric: w[j] = w[5-j]. Outer weights slightly negative.
 * Derived via SymPy (tools/compute_amr_weights.py).
 */
const double restrict_w[RESTRICT_STENCIL] = {
    -17.0 / 11520.0,     /* -0.00147569... (outermost) */
     97.0 /  3840.0,     /*  0.02526041... */
   2743.0 /  5760.0,     /*  0.47621527... (innermost) */
   2743.0 /  5760.0,     /*  0.47621527... (innermost) */
     97.0 /  3840.0,     /*  0.02526041... */
    -17.0 / 11520.0      /* -0.00147569... (outermost) */
};

/* Pre-computed w[sk]*w[sj] products for 6×6 tensor restriction.
 * Eliminates one multiply per inner loop iteration (216 per coarse cell). */
const double restrict_wkj[RESTRICT_STENCIL][RESTRICT_STENCIL] = {
    { (-17.0/11520.0)*(-17.0/11520.0), (-17.0/11520.0)*(97.0/3840.0),
      (-17.0/11520.0)*(2743.0/5760.0), (-17.0/11520.0)*(2743.0/5760.0),
      (-17.0/11520.0)*(97.0/3840.0),   (-17.0/11520.0)*(-17.0/11520.0) },
    { (97.0/3840.0)*(-17.0/11520.0), (97.0/3840.0)*(97.0/3840.0),
      (97.0/3840.0)*(2743.0/5760.0), (97.0/3840.0)*(2743.0/5760.0),
      (97.0/3840.0)*(97.0/3840.0),   (97.0/3840.0)*(-17.0/11520.0) },
    { (2743.0/5760.0)*(-17.0/11520.0), (2743.0/5760.0)*(97.0/3840.0),
      (2743.0/5760.0)*(2743.0/5760.0), (2743.0/5760.0)*(2743.0/5760.0),
      (2743.0/5760.0)*(97.0/3840.0),   (2743.0/5760.0)*(-17.0/11520.0) },
    { (2743.0/5760.0)*(-17.0/11520.0), (2743.0/5760.0)*(97.0/3840.0),
      (2743.0/5760.0)*(2743.0/5760.0), (2743.0/5760.0)*(2743.0/5760.0),
      (2743.0/5760.0)*(97.0/3840.0),   (2743.0/5760.0)*(-17.0/11520.0) },
    { (97.0/3840.0)*(-17.0/11520.0), (97.0/3840.0)*(97.0/3840.0),
      (97.0/3840.0)*(2743.0/5760.0), (97.0/3840.0)*(2743.0/5760.0),
      (97.0/3840.0)*(97.0/3840.0),   (97.0/3840.0)*(-17.0/11520.0) },
    { (-17.0/11520.0)*(-17.0/11520.0), (-17.0/11520.0)*(97.0/3840.0),
      (-17.0/11520.0)*(2743.0/5760.0), (-17.0/11520.0)*(2743.0/5760.0),
      (-17.0/11520.0)*(97.0/3840.0),   (-17.0/11520.0)*(-17.0/11520.0) },
};

void restrict_field(const grid_t *fine_g, int ff,
                    grid_t *coarse_g, int cf)
{
    const int ghost_c = coarse_g->ghost;
    const int N_c = coarse_g->N;
    const int ghost_f = fine_g->ghost;
    const double *src = fine_g->fields[ff];

    for (int ck = ghost_c; ck < ghost_c + N_c; ck++) {
        for (int cj = ghost_c; cj < ghost_c + N_c; cj++) {
            for (int ci = ghost_c; ci < ghost_c + N_c; ci++) {
                int fi_base = 2 * (ci - ghost_c) + ghost_f;
                int fj_base = 2 * (cj - ghost_c) + ghost_f;
                int fk_base = 2 * (ck - ghost_c) + ghost_f;

                coarse_g->fields[cf][IDX(coarse_g, ci, cj, ck)] =
                    restrict_cell(src, fine_g, fi_base, fj_base, fk_base);
            }
        }
    }
}

void restrict_all(const grid_t *fine_g, grid_t *coarse_g)
{
    for (int f = 0; f < NUM_FIELDS; f++) {
        restrict_field(fine_g, f, coarse_g, f);
    }
}

void restrict_to_coarse_buf(struct block_s *b)
{
    if (!b || !b->coarse_buf) return;

    const grid_t *fg = b->grid;
    grid_t *cg = b->coarse_buf;
    const int ghost_f = fg->ghost;
    const int ghost_c = cg->ghost;
    const int N_c = cg->N;

    for (int f = 0; f < NUM_FIELDS; f++) {
        const double *src = fg->fields[f];

        for (int ck = ghost_c; ck < ghost_c + N_c; ck++) {
            for (int cj = ghost_c; cj < ghost_c + N_c; cj++) {
                for (int ci = ghost_c; ci < ghost_c + N_c; ci++) {
                    int fi_base = 2 * (ci - ghost_c) + ghost_f;
                    int fj_base = 2 * (cj - ghost_c) + ghost_f;
                    int fk_base = 2 * (ck - ghost_c) + ghost_f;

                    cg->fields[f][IDX(cg, ci, cj, ck)] =
                        restrict_cell(src, fg, fi_base, fj_base, fk_base);
                }
            }
        }
    }
}
