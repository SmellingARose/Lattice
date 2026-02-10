/*
 * ricci.c — Conformal Ricci tensor computation
 *
 * Computes the conformal Ricci tensor R_tilde_{ij} from second derivatives
 * of the conformal metric and Christoffel symbols.
 *
 * The efficient form (arXiv:1106.2254 eq 3.10, B&S eq 3.69):
 *
 *   R_tilde_{ij} = -(1/2) g_tilde^{lm} d_l d_m g_tilde_{ij}
 *                  + g_tilde_{k(i} d_{j)} Gamma_hat^k
 *                  + Gamma_hat^k Gamma_tilde_{(ij)k}
 *                  + g_tilde^{lm} (2 Gamma^k_{l(i} Gamma_{j)km}
 *                                  + Gamma^k_{im} Gamma_{klj})
 *
 * This uses Gamma_hat^i (the evolved connection) rather than computing
 * Gamma_tilde^i from derivatives, which improves stability.
 */

#include "../core/grid.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"

/*
 * Compute conformal Ricci tensor R_tilde_{ij} at a single grid point.
 *
 * Inputs:
 *   g          — grid
 *   idx        — linear index
 *
 * Output:
 *   ricci_dd[6] — R_tilde_{ij} in symmetric storage
 *
 * Internally computes: gt, gtu, Christoffels, d1_gt, d2_gt, d1_Ghat, Ghat
 */
void ricci_conformal(grid_t *g, int idx, double ricci_dd[6])
{
    const int sx = grid_stride_x(g);
    const int sy = grid_stride_y(g);
    const int sz = grid_stride_z(g);
    const double dx = g->params.dx;
    const double dy = g->params.dy;
    const double dz = g->params.dz;
    const int strides[3] = { sx, sy, sz };
    const double dxs[3] = { dx, dy, dz };

    /* Read conformal metric at this point */
    double gt[6], gtu[6];
    for (int a = 0; a < 6; a++) {
        gt[a] = g->rk_scratch[FIELD_GT_BASE + a][idx];
    }
    double det = gt[SYM_XX] * (gt[SYM_YY] * gt[SYM_ZZ] - gt[SYM_YZ] * gt[SYM_YZ])
               - gt[SYM_XY] * (gt[SYM_XY] * gt[SYM_ZZ] - gt[SYM_XZ] * gt[SYM_YZ])
               + gt[SYM_XZ] * (gt[SYM_XY] * gt[SYM_YZ] - gt[SYM_XZ] * gt[SYM_YY]);
    double inv_det = 1.0 / det;

    gtu[SYM_XX] = (gt[SYM_YY] * gt[SYM_ZZ] - gt[SYM_YZ] * gt[SYM_YZ]) * inv_det;
    gtu[SYM_XY] = (gt[SYM_XZ] * gt[SYM_YZ] - gt[SYM_XY] * gt[SYM_ZZ]) * inv_det;
    gtu[SYM_XZ] = (gt[SYM_XY] * gt[SYM_YZ] - gt[SYM_XZ] * gt[SYM_YY]) * inv_det;
    gtu[SYM_YY] = (gt[SYM_XX] * gt[SYM_ZZ] - gt[SYM_XZ] * gt[SYM_XZ]) * inv_det;
    gtu[SYM_YZ] = (gt[SYM_XY] * gt[SYM_XZ] - gt[SYM_XX] * gt[SYM_YZ]) * inv_det;
    gtu[SYM_ZZ] = (gt[SYM_XX] * gt[SYM_YY] - gt[SYM_XY] * gt[SYM_XY]) * inv_det;

    /* Read Gamma_hat^i */
    double Ghat[3];
    Ghat[0] = g->rk_scratch[FIELD_GHAT1][idx];
    Ghat[1] = g->rk_scratch[FIELD_GHAT2][idx];
    Ghat[2] = g->rk_scratch[FIELD_GHAT3][idx];

    /* First derivatives of conformal metric: d1_gt[dir][comp] */
    double d1_gt[3][6];
    for (int a = 0; a < 6; a++) {
        const double *f = g->rk_scratch[FIELD_GT_BASE + a];
        for (int d = 0; d < 3; d++) {
            d1_gt[d][a] = FD_D1(f, idx, strides[d], dxs[d]);
        }
    }

    /* Second derivatives of conformal metric: d2_gt[dir1][dir2][comp]
     * Only compute unique combinations (dir1 <= dir2 for diagonal, all mixed) */
    double d2_gt[3][3][6];
    for (int a = 0; a < 6; a++) {
        const double *f = g->rk_scratch[FIELD_GT_BASE + a];
        /* Diagonal second derivatives */
        for (int d = 0; d < 3; d++) {
            d2_gt[d][d][a] = FD_D2(f, idx, strides[d], dxs[d]);
        }
        /* Mixed second derivatives */
        d2_gt[0][1][a] = FD_D1D1(f, idx, sx, sy, dx, dy);
        d2_gt[1][0][a] = d2_gt[0][1][a];
        d2_gt[0][2][a] = FD_D1D1(f, idx, sx, sz, dx, dz);
        d2_gt[2][0][a] = d2_gt[0][2][a];
        d2_gt[1][2][a] = FD_D1D1(f, idx, sy, sz, dy, dz);
        d2_gt[2][1][a] = d2_gt[1][2][a];
    }

    /* First derivatives of Gamma_hat^i: d1_Ghat[dir][i] */
    double d1_Ghat[3][3];
    for (int ii = 0; ii < 3; ii++) {
        const double *f = g->rk_scratch[FIELD_GHAT1 + ii];
        for (int d = 0; d < 3; d++) {
            d1_Ghat[d][ii] = FD_D1(f, idx, strides[d], dxs[d]);
        }
    }

    /* Christoffel symbols: Gamma^i_{jk} */
    double chris[3][6];
    for (int jk = 0; jk < 6; jk++) {
        int jj, kk;
        switch (jk) {
        case SYM_XX: jj = 0; kk = 0; break;
        case SYM_XY: jj = 0; kk = 1; break;
        case SYM_XZ: jj = 0; kk = 2; break;
        case SYM_YY: jj = 1; kk = 1; break;
        case SYM_YZ: jj = 1; kk = 2; break;
        default:     jj = 2; kk = 2; break; /* SYM_ZZ */
        }

        for (int ii = 0; ii < 3; ii++) {
            double val = 0.0;
            for (int ll = 0; ll < 3; ll++) {
                double gtu_il = gtu[SYM(ii, ll)];
                val += gtu_il * (d1_gt[jj][SYM(ll, kk)]
                               + d1_gt[kk][SYM(ll, jj)]
                               - d1_gt[ll][SYM(jj, kk)]);
            }
            chris[ii][jk] = 0.5 * val;
        }
    }

    /* Lower-index Christoffels: Gamma_{ijk} = g_{il} Gamma^l_{jk} */
    double chris_d[3][6]; /* chris_d[i][SYM(j,k)] = Gamma_{ijk} */
    for (int jk = 0; jk < 6; jk++) {
        for (int ii = 0; ii < 3; ii++) {
            chris_d[ii][jk] = 0.0;
            for (int ll = 0; ll < 3; ll++) {
                chris_d[ii][jk] += gt[SYM(ii, ll)] * chris[ll][jk];
            }
        }
    }

    /*
     * Assemble R_tilde_{ij}:
     *
     * R_{ij} = -(1/2) g^{lm} d_l d_m g_{ij}
     *        + g_{k(i} d_{j)} Ghat^k
     *        + Ghat^k Gamma_{(ij)k}
     *        + g^{lm} (2 Gamma^k_{l(i} Gamma_{j)km} + Gamma^k_{im} Gamma_{klj})
     */
    for (int ij = 0; ij < 6; ij++) {
        int ii, jj;
        switch (ij) {
        case SYM_XX: ii = 0; jj = 0; break;
        case SYM_XY: ii = 0; jj = 1; break;
        case SYM_XZ: ii = 0; jj = 2; break;
        case SYM_YY: ii = 1; jj = 1; break;
        case SYM_YZ: ii = 1; jj = 2; break;
        default:     ii = 2; jj = 2; break;
        }

        /* Term 1: -(1/2) g^{lm} d_l d_m g_{ij} */
        double term1 = 0.0;
        for (int ll = 0; ll < 3; ll++) {
            for (int mm = 0; mm < 3; mm++) {
                term1 += gtu[SYM(ll, mm)] * d2_gt[ll][mm][ij];
            }
        }
        term1 *= -0.5;

        /* Term 2: g_{k(i} d_{j)} Ghat^k = (1/2)(g_{ki} d_j Ghat^k + g_{kj} d_i Ghat^k) */
        double term2 = 0.0;
        for (int kk = 0; kk < 3; kk++) {
            term2 += gt[SYM(kk, ii)] * d1_Ghat[jj][kk]
                   + gt[SYM(kk, jj)] * d1_Ghat[ii][kk];
        }
        term2 *= 0.5;

        /* Term 3: Ghat^k Gamma_{(ij)k} = (1/2) Ghat^k (Gamma_{ijk} + Gamma_{jik}) */
        /* Note: Gamma_{ijk} = chris_d[i][SYM(j,k)] which is already symmetric in j,k.
         * So Gamma_{(ij)k} = (1/2)(Gamma_{ijk} + Gamma_{jik}) = (1/2)(chris_d[i][SYM(j,k)] + chris_d[j][SYM(i,k)])
         * But we want symmetry in (ij), so:
         * Ghat^k * Gamma_{(ij)k} = Ghat^k * (1/2)(chris_d[i][SYM(j,k)] + chris_d[j][SYM(i,k)]) */
        double term3 = 0.0;
        for (int kk = 0; kk < 3; kk++) {
            term3 += Ghat[kk] * 0.5 * (chris_d[ii][SYM(jj, kk)]
                                       + chris_d[jj][SYM(ii, kk)]);
        }

        /* Term 4: g^{lm} (2 Gamma^k_{l(i} Gamma_{j)km} + Gamma^k_{im} Gamma_{klj}) */
        double term4 = 0.0;
        for (int ll = 0; ll < 3; ll++) {
            for (int mm = 0; mm < 3; mm++) {
                double gtu_lm = gtu[SYM(ll, mm)];
                for (int kk = 0; kk < 3; kk++) {
                    /* 2 * Gamma^k_{l(i} * Gamma_{j)km}
                     * = Gamma^k_{li} * Gamma_{jkm} + Gamma^k_{lj} * Gamma_{ikm} */
                    term4 += gtu_lm * (chris[kk][SYM(ll, ii)] * chris_d[jj][SYM(kk, mm)]
                                     + chris[kk][SYM(ll, jj)] * chris_d[ii][SYM(kk, mm)]);
                    /* Gamma^k_{im} * Gamma_{klj} */
                    term4 += gtu_lm * chris[kk][SYM(ii, mm)] * chris_d[kk][SYM(ll, jj)];
                }
            }
        }

        ricci_dd[ij] = term1 + term2 + term3 + term4;
    }
}

/*
 * Full physical Ricci scalar including conformal factor terms.
 * R = chi * R_tilde + chi * R_chi
 * where R_chi contains terms from the conformal factor.
 *
 * R_chi = g_tilde^{ij} (-2 D_i D_j chi / chi
 *                        - g_tilde_{ij} g_tilde^{kl} D_k D_l chi / chi
 *                        + 4 D_i chi D_j chi / chi^2
 *                        + g_tilde_{ij} g_tilde^{kl} D_k chi D_l chi / chi^2)
 * Wait — the full formula is (B&S eq 3.57):
 * R = chi * R_tilde
 *   + 1/(2 chi) * [-3 g^{ij} D_i D_j chi + 3/(2 chi) g^{ij} D_i chi D_j chi]
 * Not needed for CCZ4 RHS (uses separate conformal + Ricci contributions).
 * Kept for diagnostic purposes.
 */
double ricci_scalar(grid_t *g, int idx)
{
    double ricci_dd[6];
    ricci_conformal(g, idx, ricci_dd);

    /* Contract with inverse conformal metric */
    double gt[6], gtu[6];
    for (int a = 0; a < 6; a++) {
        gt[a] = g->rk_scratch[FIELD_GT_BASE + a][idx];
    }
    double det = gt[SYM_XX] * (gt[SYM_YY] * gt[SYM_ZZ] - gt[SYM_YZ] * gt[SYM_YZ])
               - gt[SYM_XY] * (gt[SYM_XY] * gt[SYM_ZZ] - gt[SYM_XZ] * gt[SYM_YZ])
               + gt[SYM_XZ] * (gt[SYM_XY] * gt[SYM_YZ] - gt[SYM_XZ] * gt[SYM_YY]);
    double inv_det = 1.0 / det;
    gtu[SYM_XX] = (gt[SYM_YY] * gt[SYM_ZZ] - gt[SYM_YZ] * gt[SYM_YZ]) * inv_det;
    gtu[SYM_XY] = (gt[SYM_XZ] * gt[SYM_YZ] - gt[SYM_XY] * gt[SYM_ZZ]) * inv_det;
    gtu[SYM_XZ] = (gt[SYM_XY] * gt[SYM_YZ] - gt[SYM_XZ] * gt[SYM_YY]) * inv_det;
    gtu[SYM_YY] = (gt[SYM_XX] * gt[SYM_ZZ] - gt[SYM_XZ] * gt[SYM_XZ]) * inv_det;
    gtu[SYM_YZ] = (gt[SYM_XY] * gt[SYM_XZ] - gt[SYM_XX] * gt[SYM_YZ]) * inv_det;
    gtu[SYM_ZZ] = (gt[SYM_XX] * gt[SYM_YY] - gt[SYM_XY] * gt[SYM_XY]) * inv_det;

    /* R_tilde = g_tilde^{ij} R_tilde_{ij} */
    double Rt = gtu[SYM_XX] * ricci_dd[SYM_XX]
              + gtu[SYM_YY] * ricci_dd[SYM_YY]
              + gtu[SYM_ZZ] * ricci_dd[SYM_ZZ]
              + 2.0 * (gtu[SYM_XY] * ricci_dd[SYM_XY]
                     + gtu[SYM_XZ] * ricci_dd[SYM_XZ]
                     + gtu[SYM_YZ] * ricci_dd[SYM_YZ]);

    /* Add conformal factor contribution:
     * R = chi * Rt + chi-derivative terms
     * For a full expression see ccz4_rhs.c; here we just return chi * Rt
     * as a leading-order approximation for diagnostics. */
    double chi = g->rk_scratch[FIELD_CHI][idx];
    return chi * Rt;
}
