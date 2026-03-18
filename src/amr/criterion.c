/*
 * Lattice — 3D Numerical Relativity
 * AMR refinement criterion: chi-gradient based.
 *
 * The chi-gradient criterion measures how steep the conformal factor is:
 *   criterion = (dx / chi^2) * |grad(chi)|
 *
 * Near punctures, chi → 0 and |grad(chi)| is large, producing large
 * criterion values that trigger refinement. Far from punctures, chi → 1
 * and gradients are small, allowing coarsening.
 *
 * Ref: GRChombo Source/TaggingCriteria/ChiTaggingCriterion.hpp:31
 * Ref: arXiv:2312.05438 (chi-gradient vs truncation error)
 */

#include "criterion.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"
#include <math.h>

double chi_gradient_max(const block_t *b)
{
    const grid_t *g = b->grid;
    const double *chi = g->fields[FIELD_CHI];
    const int ghost = g->ghost;
    const int N = g->N;
    const double dx = g->dx;
    const double inv_dx = g->inv_dx;
    const int sx = STRIDE_X;
    const int sy = STRIDE_Y(g);
    const int sz = STRIDE_Z(g);

    double max_crit = 0.0;

    /* Loop over interior cells only (exclude ghost zones) */
    for (int k = ghost; k < ghost + N; k++) {
        for (int j = ghost; j < ghost + N; j++) {
            for (int i = ghost; i < ghost + N; i++) {
                int idx = IDX(g, i, j, k);
                double chi_val = chi[idx];

                /* Protect against chi → 0 (near puncture) */
                if (chi_val < 1.0e-4) chi_val = 1.0e-4;

                double chi2 = chi_val * chi_val;

                /* 4th-order first derivatives of chi */
                double d1x = fd_d1(chi, idx, sx, inv_dx);
                double d1y = fd_d1(chi, idx, sy, inv_dx);
                double d1z = fd_d1(chi, idx, sz, inv_dx);

                /* criterion = (dx / chi^2) * |grad(chi)| */
                double grad_mag = sqrt(d1x * d1x + d1y * d1y + d1z * d1z);
                double crit = dx * grad_mag / chi2;

                if (crit > max_crit) max_crit = crit;
            }
        }
    }

    return max_crit;
}

int criterion_check_block(const block_t *b, double chi_refine, double chi_coarsen)
{
    double crit = chi_gradient_max(b);

    if (crit > chi_refine)
        return AMR_REFINE;
    if (crit < chi_coarsen)
        return AMR_COARSEN;
    return AMR_NONE;
}

int criterion_check_mesh(const mesh_t *m, const amr_params_t *ap,
                         refine_flag_t *flags, int max_flags)
{
    int count = 0;

    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        if (count >= max_flags) break;

        int action = criterion_check_block(b, ap->chi_refine, ap->chi_coarsen);

        /* Don't refine beyond max_level */
        if (action == AMR_REFINE && b->loc.level >= ap->max_level)
            action = AMR_NONE;

        /* Don't coarsen root level */
        if (action == AMR_COARSEN && b->loc.level <= 0)
            action = AMR_NONE;

        flags[count].block_id = b->id;
        flags[count].action = action;
        count++;
    }

    return count;
}
