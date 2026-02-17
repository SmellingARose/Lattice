/*
 * Lattice — 3D Numerical Relativity
 * 4th-order cell-average restriction (fine → coarse).
 *
 * 1D symmetric 4-point Lagrange stencil (exact for degree ≤ 3):
 *   w = { 1/48, 23/48, 23/48, 1/48 }
 * 3D: tensor product of 1D weights, 4³ = 64 fine cells per coarse cell.
 *
 * Stencil reach: 1 fine cell beyond the 2 direct children per direction.
 * Falls back to 2nd-order (2-point average per direction) if stencil
 * extends outside the fine grid array bounds.
 *
 * Ref: ExaHyPE (arXiv:2504.15814) — upgrading restriction to match
 *      prolongation order eliminates Hamiltonian violations at AMR boundaries.
 * Ref: GRChombo CoarseAverage (Chombo library) — 2nd-order baseline.
 * Ref: AthenaK Lagrange restriction matching FD order.
 * Ref: Fornberg, SIAM Review 40 (1998) — FD weight generation algorithm.
 */

#include "restriction.h"
#include "block.h"
#include "../core/fields.h"

/*
 * 4th-order cell-average restriction weights.
 * For coarse cell at position x_c, fine cells at -3δ/2, -δ/2, +δ/2, +3δ/2
 * (where δ = dx_fine = dx_coarse/2).
 *
 * Derived by integrating Lagrange basis polynomials over the coarse cell:
 *   w_j = (1/Δx_c) ∫_{-Δx_c/2}^{+Δx_c/2} L_j(x) dx
 * where L_j are the 4-point Lagrange polynomials through the fine cell centers.
 *
 * Sum = 1 (partition of unity), all positive (no oscillations).
 */
const double restrict_w[RESTRICT_STENCIL] = {
     1.0 / 48.0,    /*  0.02083333... (outer left)  */
    23.0 / 48.0,    /*  0.47916666... (inner left)  */
    23.0 / 48.0,    /*  0.47916666... (inner right) */
     1.0 / 48.0     /*  0.02083333... (outer right) */
};

/*
 * Restrict a single coarse cell from fine data using 4th-order stencil.
 * fi_base, fj_base, fk_base: fine grid indices of the first direct child.
 * Returns the restricted value.
 *
 * If the 4-point stencil extends outside [0, Ntotal), falls back to
 * 2nd-order 8-cell average (just the 2 direct children per direction).
 */
static inline double restrict_cell(const double *src, const grid_t *fg,
                                    int fi_base, int fj_base, int fk_base)
{
    const int lo = fg->ghost;
    const int hi = fg->ghost + fg->N;

    /* Check if 4-point stencil stays within fine grid INTERIOR [ghost, ghost+N).
     * The stencil extends 1 cell beyond the 2 direct children per direction.
     * At coarse-fine boundaries, fine ghost zones may not yet be filled, so we
     * must restrict only from valid interior data.
     * Ref: AthenaK keeps restriction stencil within active region. */
    if (fi_base - 1 >= lo && fi_base + 2 < hi &&
        fj_base - 1 >= lo && fj_base + 2 < hi &&
        fk_base - 1 >= lo && fk_base + 2 < hi) {

        /* 4th-order: 3D tensor product of 4-point stencil */
        double val = 0.0;
        for (int sk = 0; sk < RESTRICT_STENCIL; sk++) {
            int fk = fk_base - 1 + sk;
            for (int sj = 0; sj < RESTRICT_STENCIL; sj++) {
                double wkj = restrict_w[sk] * restrict_w[sj];
                int fj = fj_base - 1 + sj;
                for (int si = 0; si < RESTRICT_STENCIL; si++) {
                    int fi = fi_base - 1 + si;
                    val += wkj * restrict_w[si] * src[IDX(fg, fi, fj, fk)];
                }
            }
        }
        return val;

    } else {
        /* 2nd-order fallback: average of 8 direct children */
        double sum = 0.0;
        for (int ok = 0; ok < 2; ok++)
            for (int oj = 0; oj < 2; oj++)
                for (int oi = 0; oi < 2; oi++)
                    sum += src[IDX(fg, fi_base + oi,
                                       fj_base + oj,
                                       fk_base + ok)];
        return sum * 0.125;
    }
}

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
                    /* Coarse cell (ci,cj,ck) maps to fine cells starting at
                     * fi_base = 2*(ci - ghost_c) + ghost_f */
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
