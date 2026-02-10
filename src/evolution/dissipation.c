/*
 * dissipation.c — Kreiss-Oliger numerical dissipation
 *
 * Applies 6th-order KO dissipation to all evolved fields:
 *   rhs[f][idx] -= eps(x) * (KO6_x + KO6_y + KO6_z)
 *
 * Spatially varying dissipation coefficient (arXiv:2404.01137):
 *   eps(x) = W(x) * eps_CA
 * where W = chi^{1/2} (conformal factor, -> 0 at punctures, -> 1 in weak field).
 *
 * Gauge variables (alpha, beta, B): eps_CA = 0.99
 * Other CCZ4 variables:              eps_CA = 0.3
 */

#include "../core/grid.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"

#include <math.h>

static int is_gauge_field(int f)
{
    return (f == FIELD_ALPHA
         || f == FIELD_BETA1 || f == FIELD_BETA2 || f == FIELD_BETA3
         || f == FIELD_GBAUX1 || f == FIELD_GBAUX2 || f == FIELD_GBAUX3);
}

void dissipation_apply(grid_t *g)
{
    const double dx = g->params.dx;
    const double dy = g->params.dy;
    const double dz = g->params.dz;
    const int sx = grid_stride_x(g);
    const int sy = grid_stride_y(g);
    const int sz = grid_stride_z(g);
    const int nf = g->params.num_fields;
    const double eps_gauge = g->params.ko_eps_gauge;
    const double eps_other = g->params.ko_eps_other;

    /* OMP: per-point dissipation, no cross-point dependencies.
     * Toggle: make PARALLEL=0/1 */
    GRID_LOOP_INTERIOR_OMP(g, i, j, k) {
        int idx = grid_idx(g, i, j, k);

        /* W = chi^{1/2} — spatially varying multiplier */
        double chi = g->rk_scratch[FIELD_CHI][idx];
        double W = sqrt(fmax(chi, 1e-16));

        for (int f = 0; f < nf; f++) {
            double eps_ca = is_gauge_field(f) ? eps_gauge : eps_other;
            double eps = W * eps_ca;

            double ko = FD_KO6(g->rk_scratch[f], idx, sx, dx)
                      + FD_KO6(g->rk_scratch[f], idx, sy, dy)
                      + FD_KO6(g->rk_scratch[f], idx, sz, dz);

            g->rhs[f][idx] -= eps * ko;
        }
    }
}
