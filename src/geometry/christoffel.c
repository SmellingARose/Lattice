/*
 * christoffel.c — Christoffel symbols of the conformal metric
 *
 * Computes Gamma_tilde^i_{jk} from first derivatives of gamma_tilde_{ij}.
 *
 * Gamma^i_{jk} = (1/2) g^{il} (d_j g_{lk} + d_k g_{lj} - d_l g_{jk})
 *
 * All quantities are computed locally per point (stack variables).
 * B&S eq 2.108
 */

#include "../core/grid.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"

/*
 * Compute Christoffel symbols Gamma_tilde^i_{jk} at a single grid point.
 *
 * Inputs:
 *   g     — grid (for strides/spacing)
 *   idx   — linear index of the point
 *   gt[6] — conformal metric gamma_tilde_{ij} at this point
 *   gtu[6] — inverse conformal metric gamma_tilde^{ij} at this point
 *
 * Output:
 *   chris[3][6] — Gamma^i_{jk} stored as chris[i][SYM(j,k)]
 *
 * Uses FD_D1 to compute derivatives of the conformal metric.
 */
void christoffel_at_point(grid_t *g, int idx,
                          const double gt[6], const double gtu[6],
                          double chris[3][6])
{
    (void)gt; /* metric values not needed — only derivatives and inverse used */
    const int sx = grid_stride_x(g);
    const int sy = grid_stride_y(g);
    const int sz = grid_stride_z(g);
    const double dx = g->params.dx;
    const double dy = g->params.dy;
    const double dz = g->params.dz;

    /* First derivatives of conformal metric: d1_gt[dir][component] */
    double d1_gt[3][6];
    const int strides[3] = { sx, sy, sz };
    const double dxs[3] = { dx, dy, dz };

    for (int a = 0; a < 6; a++) {
        const double *f = g->rk_scratch[FIELD_GT_BASE + a];
        for (int d = 0; d < 3; d++) {
            d1_gt[d][a] = FD_D1(f, idx, strides[d], dxs[d]);
        }
    }

    /* Gamma^i_{jk} = (1/2) g^{il} (d_j g_{lk} + d_k g_{lj} - d_l g_{jk}) */
    for (int jk = 0; jk < 6; jk++) {
        /* Decode symmetric index jk back to (j, k) */
        int jj, kk;
        switch (jk) {
        case SYM_XX: jj = 0; kk = 0; break;
        case SYM_XY: jj = 0; kk = 1; break;
        case SYM_XZ: jj = 0; kk = 2; break;
        case SYM_YY: jj = 1; kk = 1; break;
        case SYM_YZ: jj = 1; kk = 2; break;
        case SYM_ZZ: jj = 2; kk = 2; break;
        default: jj = 0; kk = 0; break;
        }

        for (int ii = 0; ii < 3; ii++) {
            double val = 0.0;
            for (int ll = 0; ll < 3; ll++) {
                /* g^{il} * (d_j g_{lk} + d_k g_{lj} - d_l g_{jk}) */
                double gtu_il;
                switch (SYM(ii, ll)) {
                case SYM_XX: gtu_il = gtu[SYM_XX]; break;
                case SYM_XY: gtu_il = gtu[SYM_XY]; break;
                case SYM_XZ: gtu_il = gtu[SYM_XZ]; break;
                case SYM_YY: gtu_il = gtu[SYM_YY]; break;
                case SYM_YZ: gtu_il = gtu[SYM_YZ]; break;
                case SYM_ZZ: gtu_il = gtu[SYM_ZZ]; break;
                default: gtu_il = 0.0; break;
                }

                double d_j_g_lk = d1_gt[jj][SYM(ll, kk)];
                double d_k_g_lj = d1_gt[kk][SYM(ll, jj)];
                double d_l_g_jk = d1_gt[ll][SYM(jj, kk)];

                val += gtu_il * (d_j_g_lk + d_k_g_lj - d_l_g_jk);
            }
            chris[ii][jk] = 0.5 * val;
        }
    }
}

/*
 * Contracted Christoffel symbols: Gamma^i = g^{jk} Gamma^i_{jk}
 *
 * Input:
 *   gtu[6]      — inverse conformal metric
 *   chris[3][6] — Christoffel symbols from christoffel_at_point()
 *
 * Output:
 *   gamma_contracted[3] — Gamma^i = g^{jk} Gamma^i_{jk}
 */
void christoffel_contracted(const double gtu[6], const double chris[3][6],
                            double gamma_contracted[3])
{
    for (int ii = 0; ii < 3; ii++) {
        gamma_contracted[ii] =
            gtu[SYM_XX] * chris[ii][SYM_XX]
          + gtu[SYM_YY] * chris[ii][SYM_YY]
          + gtu[SYM_ZZ] * chris[ii][SYM_ZZ]
          + 2.0 * (gtu[SYM_XY] * chris[ii][SYM_XY]
                 + gtu[SYM_XZ] * chris[ii][SYM_XZ]
                 + gtu[SYM_YZ] * chris[ii][SYM_YZ]);
    }
}
