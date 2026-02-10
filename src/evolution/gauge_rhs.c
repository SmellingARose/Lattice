/*
 * gauge_rhs.c — Gauge evolution equations (lapse + shift)
 *
 * 1+log lapse with CCZ4 modification:
 *   dt alpha = -2 alpha (K - 2 Theta) + beta^k d_k alpha
 *
 * Gamma-driver shift:
 *   dt beta^i  = f * B^i + beta^k d_k beta^i
 *   dt B^i     = dt Ghat^i - beta^k d_k Ghat^i + beta^k d_k B^i - eta B^i
 *
 * Position-dependent eta:
 *   eta(x) = eta_0 / W(x) = eta_0 / chi(x)^{1/2}
 *
 * IMPORTANT: gauge_rhs must be called AFTER ccz4_rhs because the B^i
 * equation reads g->rhs[FIELD_GHAT*] (= dt Ghat^i computed by ccz4_rhs).
 *
 * Ref: B&S Ch. 4.2; arXiv:gr-qc/0206072 (Gamma driver)
 */

#include "../core/grid.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"

#include <math.h>

void gauge_rhs(grid_t *g)
{
    const int sx = grid_stride_x(g);
    const int sy = grid_stride_y(g);
    const int sz = grid_stride_z(g);
    const double dx = g->params.dx;
    const double dy = g->params.dy;
    const double dz = g->params.dz;
    const int strides[3] = { sx, sy, sz };
    const double dxs[3] = { dx, dy, dz };
    const double f_driver = g->params.gamma_driver_f;
    const double eta0 = g->params.eta;

    /* OMP: per-point gauge computation, no cross-point dependencies.
     * Toggle: make PARALLEL=0/1 */
    GRID_LOOP_INTERIOR_OMP(g, i, j, k) {
        int idx = grid_idx(g, i, j, k);

        double alpha = g->rk_scratch[FIELD_ALPHA][idx];
        double K = g->rk_scratch[FIELD_TRKA][idx];
        double Theta = g->rk_scratch[FIELD_THETA][idx];
        double chi = g->rk_scratch[FIELD_CHI][idx];

        double beta[3], B[3], Ghat[3];
        for (int d = 0; d < 3; d++) {
            beta[d] = g->rk_scratch[FIELD_BETA1 + d][idx];
            B[d] = g->rk_scratch[FIELD_GBAUX1 + d][idx];
            Ghat[d] = g->rk_scratch[FIELD_GHAT1 + d][idx];
        }

        /* Position-dependent eta: eta(x) = eta0 / W = eta0 / sqrt(chi) */
        double W = sqrt(fmax(chi, 1e-16));
        double eta = eta0 / fmax(W, 1e-8);

        /* === Lapse: 1+log with CCZ4 === */
        /* dt alpha = -2 alpha (K - 2 Theta) + advection */
        double adv_alpha = 0.0;
        for (int d = 0; d < 3; d++) {
            adv_alpha += FD_ADV(g->rk_scratch[FIELD_ALPHA], idx, strides[d], dxs[d], beta[d]);
        }

        g->rhs[FIELD_ALPHA][idx] = -2.0 * alpha * (K - 2.0 * Theta) + adv_alpha;

        /* === Shift: Gamma driver === */
        for (int a = 0; a < 3; a++) {
            /* dt beta^i = f * B^i + advection */
            double adv_beta = 0.0;
            for (int d = 0; d < 3; d++) {
                adv_beta += FD_ADV(g->rk_scratch[FIELD_BETA1 + a], idx, strides[d], dxs[d], beta[d]);
            }

            g->rhs[FIELD_BETA1 + a][idx] = f_driver * B[a] + adv_beta;

            /* dt B^i = dt Ghat^i - beta^k d_k Ghat^i + beta^k d_k B^i - eta B^i
             *
             * dt Ghat^i was already computed by ccz4_rhs and stored in g->rhs[FIELD_GHAT1 + a].
             * We read it here.
             */
            double dt_Ghat_i = g->rhs[FIELD_GHAT1 + a][idx];

            /* beta^k d_k Ghat^i (upwind advection of Ghat) */
            double adv_Ghat = 0.0;
            for (int d = 0; d < 3; d++) {
                adv_Ghat += FD_ADV(g->rk_scratch[FIELD_GHAT1 + a], idx, strides[d], dxs[d], beta[d]);
            }

            /* beta^k d_k B^i (upwind advection of B) */
            double adv_B = 0.0;
            for (int d = 0; d < 3; d++) {
                adv_B += FD_ADV(g->rk_scratch[FIELD_GBAUX1 + a], idx, strides[d], dxs[d], beta[d]);
            }

            g->rhs[FIELD_GBAUX1 + a][idx] = dt_Ghat_i - adv_Ghat + adv_B - eta * B[a];
        }
    }
}
