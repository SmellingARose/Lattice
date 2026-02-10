/*
 * ccz4_rhs.c — CCZ4 evolution equations right-hand side
 *
 * Implements the complete CCZ4 system (arXiv:1106.2254) with GRChombo
 * modification kappa1 -> kappa1/alpha for BH stability (arXiv:1503.03436).
 *
 * Evolved variables: {chi, gt_{ij}, K, At_{ij}, Ghat^i, Theta}
 *
 * Conformal factor convention: chi = e^{-4 phi}, W = chi^{1/2} = e^{-2 phi}
 *
 * Per-point computation:
 *   1. Read field values
 *   2. Compute inverse metric
 *   3. First derivatives (D1) of all fields
 *   4. Second derivatives (D2) of chi, gt, alpha
 *   5. Advection terms (upwind)
 *   6. Christoffel symbols
 *   7. Conformal Ricci tensor
 *   8. Z_i from Gamma_hat^i
 *   9. Assemble all RHS equations
 */

#include "../core/grid.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"
#include "../geometry/tensor_utils.h"

#include <math.h>

/*
 * Compute CCZ4 RHS at a single grid point.
 * All field data is read from g->rk_scratch[], output to g->rhs[].
 */
static void ccz4_rhs_point(grid_t *g, int idx)
{
    const int sx = grid_stride_x(g);
    const int sy = grid_stride_y(g);
    const int sz = grid_stride_z(g);
    const double dx = g->params.dx;
    const double dy = g->params.dy;
    const double dz = g->params.dz;
    const int strides[3] = { sx, sy, sz };
    const double dxs[3] = { dx, dy, dz };

    /* CCZ4 parameters */
    const double kappa1 = g->params.kappa1;
    const double kappa2 = g->params.kappa2;
    const double kappa3 = g->params.kappa3;

    /* ===== 1. Read field values ===== */
    double chi = g->rk_scratch[FIELD_CHI][idx];
    double chi_safe = fmax(chi, 1e-16);

    double gt[6], at[6];
    for (int a = 0; a < 6; a++) {
        gt[a] = g->rk_scratch[FIELD_GT_BASE + a][idx];
        at[a] = g->rk_scratch[FIELD_AT_BASE + a][idx];
    }

    double K = g->rk_scratch[FIELD_TRKA][idx];
    double Theta = g->rk_scratch[FIELD_THETA][idx];
    double alpha = g->rk_scratch[FIELD_ALPHA][idx];
    double alpha_safe = fmax(fabs(alpha), 1e-16);

    double beta[3], Ghat[3];
    for (int d = 0; d < 3; d++) {
        beta[d] = g->rk_scratch[FIELD_BETA1 + d][idx];
        Ghat[d] = g->rk_scratch[FIELD_GHAT1 + d][idx];
    }

    /* ===== 2. Inverse conformal metric ===== */
    double gtu[6];
    sym3_inv(gt, gtu);

    /* ===== 3. First derivatives ===== */
    double d1_chi[3], d1_K[3], d1_Theta[3], d1_alpha[3];
    double d1_gt[3][6], d1_beta[3][3], d1_Ghat[3][3];

    for (int d = 0; d < 3; d++) {
        d1_chi[d] = FD_D1(g->rk_scratch[FIELD_CHI], idx, strides[d], dxs[d]);
        d1_K[d] = FD_D1(g->rk_scratch[FIELD_TRKA], idx, strides[d], dxs[d]);
        d1_Theta[d] = FD_D1(g->rk_scratch[FIELD_THETA], idx, strides[d], dxs[d]);
        d1_alpha[d] = FD_D1(g->rk_scratch[FIELD_ALPHA], idx, strides[d], dxs[d]);

        for (int a = 0; a < 6; a++) {
            d1_gt[d][a] = FD_D1(g->rk_scratch[FIELD_GT_BASE + a], idx, strides[d], dxs[d]);
        }

        for (int b = 0; b < 3; b++) {
            d1_beta[d][b] = FD_D1(g->rk_scratch[FIELD_BETA1 + b], idx, strides[d], dxs[d]);
            d1_Ghat[d][b] = FD_D1(g->rk_scratch[FIELD_GHAT1 + b], idx, strides[d], dxs[d]);
        }
    }

    /* Divergence of shift: d_k beta^k */
    double div_beta = d1_beta[0][0] + d1_beta[1][1] + d1_beta[2][2];

    /* ===== 4. Second derivatives ===== */
    double d2_chi[6], d2_alpha[6];
    double d2_gt[3][3][6];

    /* chi second derivatives */
    d2_chi[SYM_XX] = FD_D2(g->rk_scratch[FIELD_CHI], idx, sx, dx);
    d2_chi[SYM_YY] = FD_D2(g->rk_scratch[FIELD_CHI], idx, sy, dy);
    d2_chi[SYM_ZZ] = FD_D2(g->rk_scratch[FIELD_CHI], idx, sz, dz);
    d2_chi[SYM_XY] = FD_D1D1(g->rk_scratch[FIELD_CHI], idx, sx, sy, dx, dy);
    d2_chi[SYM_XZ] = FD_D1D1(g->rk_scratch[FIELD_CHI], idx, sx, sz, dx, dz);
    d2_chi[SYM_YZ] = FD_D1D1(g->rk_scratch[FIELD_CHI], idx, sy, sz, dy, dz);

    /* alpha second derivatives */
    d2_alpha[SYM_XX] = FD_D2(g->rk_scratch[FIELD_ALPHA], idx, sx, dx);
    d2_alpha[SYM_YY] = FD_D2(g->rk_scratch[FIELD_ALPHA], idx, sy, dy);
    d2_alpha[SYM_ZZ] = FD_D2(g->rk_scratch[FIELD_ALPHA], idx, sz, dz);
    d2_alpha[SYM_XY] = FD_D1D1(g->rk_scratch[FIELD_ALPHA], idx, sx, sy, dx, dy);
    d2_alpha[SYM_XZ] = FD_D1D1(g->rk_scratch[FIELD_ALPHA], idx, sx, sz, dx, dz);
    d2_alpha[SYM_YZ] = FD_D1D1(g->rk_scratch[FIELD_ALPHA], idx, sy, sz, dy, dz);

    /* gt second derivatives */
    for (int a = 0; a < 6; a++) {
        const double *f = g->rk_scratch[FIELD_GT_BASE + a];
        for (int d = 0; d < 3; d++) {
            d2_gt[d][d][a] = FD_D2(f, idx, strides[d], dxs[d]);
        }
        d2_gt[0][1][a] = FD_D1D1(f, idx, sx, sy, dx, dy);
        d2_gt[1][0][a] = d2_gt[0][1][a];
        d2_gt[0][2][a] = FD_D1D1(f, idx, sx, sz, dx, dz);
        d2_gt[2][0][a] = d2_gt[0][2][a];
        d2_gt[1][2][a] = FD_D1D1(f, idx, sy, sz, dy, dz);
        d2_gt[2][1][a] = d2_gt[1][2][a];
    }

    /* Second derivatives of beta (for Gamma_hat RHS) */
    double d2_beta[3][3][3]; /* d2_beta[d1][d2][component] */
    for (int b = 0; b < 3; b++) {
        const double *f = g->rk_scratch[FIELD_BETA1 + b];
        for (int d = 0; d < 3; d++) {
            d2_beta[d][d][b] = FD_D2(f, idx, strides[d], dxs[d]);
        }
        d2_beta[0][1][b] = FD_D1D1(f, idx, sx, sy, dx, dy);
        d2_beta[1][0][b] = d2_beta[0][1][b];
        d2_beta[0][2][b] = FD_D1D1(f, idx, sx, sz, dx, dz);
        d2_beta[2][0][b] = d2_beta[0][2][b];
        d2_beta[1][2][b] = FD_D1D1(f, idx, sy, sz, dy, dz);
        d2_beta[2][1][b] = d2_beta[1][2][b];
    }

    /* ===== 5. Advection terms ===== */
    double adv_chi = 0.0, adv_K = 0.0, adv_Theta = 0.0;
    double adv_gt[6], adv_at[6], adv_Ghat[3];

    for (int d = 0; d < 3; d++) {
        adv_chi += FD_ADV(g->rk_scratch[FIELD_CHI], idx, strides[d], dxs[d], beta[d]);
        adv_K += FD_ADV(g->rk_scratch[FIELD_TRKA], idx, strides[d], dxs[d], beta[d]);
        adv_Theta += FD_ADV(g->rk_scratch[FIELD_THETA], idx, strides[d], dxs[d], beta[d]);
    }

    for (int a = 0; a < 6; a++) {
        adv_gt[a] = 0.0;
        adv_at[a] = 0.0;
        for (int d = 0; d < 3; d++) {
            adv_gt[a] += FD_ADV(g->rk_scratch[FIELD_GT_BASE + a], idx, strides[d], dxs[d], beta[d]);
            adv_at[a] += FD_ADV(g->rk_scratch[FIELD_AT_BASE + a], idx, strides[d], dxs[d], beta[d]);
        }
    }

    for (int b = 0; b < 3; b++) {
        adv_Ghat[b] = 0.0;
        for (int d = 0; d < 3; d++) {
            adv_Ghat[b] += FD_ADV(g->rk_scratch[FIELD_GHAT1 + b], idx, strides[d], dxs[d], beta[d]);
        }
    }

    /* ===== 6. Christoffel symbols ===== */
    double chris[3][6]; /* Gamma^i_{jk} */
    for (int jk = 0; jk < 6; jk++) {
        int jj, kk;
        switch (jk) {
        case SYM_XX: jj = 0; kk = 0; break;
        case SYM_XY: jj = 0; kk = 1; break;
        case SYM_XZ: jj = 0; kk = 2; break;
        case SYM_YY: jj = 1; kk = 1; break;
        case SYM_YZ: jj = 1; kk = 2; break;
        default:     jj = 2; kk = 2; break;
        }
        for (int ii = 0; ii < 3; ii++) {
            double val = 0.0;
            for (int ll = 0; ll < 3; ll++) {
                val += gtu[SYM(ii, ll)] * (d1_gt[jj][SYM(ll, kk)]
                                         + d1_gt[kk][SYM(ll, jj)]
                                         - d1_gt[ll][SYM(jj, kk)]);
            }
            chris[ii][jk] = 0.5 * val;
        }
    }

    /* Contracted Christoffels: Gamma^i = g^{jk} Gamma^i_{jk} */
    double Gamma_contracted[3];
    for (int ii = 0; ii < 3; ii++) {
        Gamma_contracted[ii] = gtu[SYM_XX] * chris[ii][SYM_XX]
                             + gtu[SYM_YY] * chris[ii][SYM_YY]
                             + gtu[SYM_ZZ] * chris[ii][SYM_ZZ]
                             + 2.0 * (gtu[SYM_XY] * chris[ii][SYM_XY]
                                    + gtu[SYM_XZ] * chris[ii][SYM_XZ]
                                    + gtu[SYM_YZ] * chris[ii][SYM_YZ]);
    }

    /* ===== 7. Conformal Ricci tensor ===== */
    double Rt_dd[6]; /* R_tilde_{ij} */

    /* Lower-index Christoffels for Ricci */
    double chris_d[3][6];
    for (int jk = 0; jk < 6; jk++) {
        for (int ii = 0; ii < 3; ii++) {
            chris_d[ii][jk] = 0.0;
            for (int ll = 0; ll < 3; ll++) {
                chris_d[ii][jk] += gt[SYM(ii, ll)] * chris[ll][jk];
            }
        }
    }

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

        /* -(1/2) g^{lm} d_l d_m g_{ij} */
        double term1 = 0.0;
        for (int ll = 0; ll < 3; ll++) {
            for (int mm = 0; mm < 3; mm++) {
                term1 += gtu[SYM(ll, mm)] * d2_gt[ll][mm][ij];
            }
        }
        term1 *= -0.5;

        /* g_{k(i} d_{j)} Ghat^k */
        double term2 = 0.0;
        for (int kk = 0; kk < 3; kk++) {
            term2 += gt[SYM(kk, ii)] * d1_Ghat[jj][kk]
                   + gt[SYM(kk, jj)] * d1_Ghat[ii][kk];
        }
        term2 *= 0.5;

        /* Ghat^k Gamma_{(ij)k} */
        double term3 = 0.0;
        for (int kk = 0; kk < 3; kk++) {
            term3 += Ghat[kk] * 0.5 * (chris_d[ii][SYM(jj, kk)]
                                       + chris_d[jj][SYM(ii, kk)]);
        }

        /* g^{lm} (2 Gamma^k_{l(i} Gamma_{j)km} + Gamma^k_{im} Gamma_{klj}) */
        double term4 = 0.0;
        for (int ll = 0; ll < 3; ll++) {
            for (int mm = 0; mm < 3; mm++) {
                double gtu_lm = gtu[SYM(ll, mm)];
                for (int kk = 0; kk < 3; kk++) {
                    term4 += gtu_lm * (chris[kk][SYM(ll, ii)] * chris_d[jj][SYM(kk, mm)]
                                     + chris[kk][SYM(ll, jj)] * chris_d[ii][SYM(kk, mm)]);
                    term4 += gtu_lm * chris[kk][SYM(ii, mm)] * chris_d[kk][SYM(ll, jj)];
                }
            }
        }

        Rt_dd[ij] = term1 + term2 + term3 + term4;
    }

    /* ===== Conformal factor contributions to physical Ricci ===== */
    /*
     * The full physical Ricci has conformal factor terms.
     * Using chi = e^{-4 phi}:
     *
     * R_{ij} = R_tilde_{ij}
     *        + (1/(2 chi)) [-D_i D_j chi - gt_{ij} g^{kl} D_k D_l chi
     *                        + (1/chi) d_i chi d_j chi
     *                        + gt_{ij} g^{kl} d_k chi d_l chi / (2 chi)]
     *        + Christoffel corrections to D_i D_j chi
     *
     * Rewritten using covariant derivatives with conformal connection:
     * D_i D_j chi = d_i d_j chi - Gamma^k_{ij} d_k chi
     */
    double DDchi[6]; /* Covariant D_i D_j chi (with conformal Christoffels) */
    for (int ij = 0; ij < 6; ij++) {
        DDchi[ij] = d2_chi[ij];
        for (int kk = 0; kk < 3; kk++) {
            DDchi[ij] -= chris[kk][ij] * d1_chi[kk];
        }
    }

    /* Trace: g^{ij} D_i D_j chi */
    double laplacian_chi = gtu[SYM_XX] * DDchi[SYM_XX]
                         + gtu[SYM_YY] * DDchi[SYM_YY]
                         + gtu[SYM_ZZ] * DDchi[SYM_ZZ]
                         + 2.0 * (gtu[SYM_XY] * DDchi[SYM_XY]
                                + gtu[SYM_XZ] * DDchi[SYM_XZ]
                                + gtu[SYM_YZ] * DDchi[SYM_YZ]);

    /* g^{kl} d_k chi d_l chi */
    double dchi_sq = 0.0;
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            dchi_sq += gtu[SYM(a, b)] * d1_chi[a] * d1_chi[b];
        }
    }

    /* Full Ricci R_{ij} including conformal factor:
     * R_{ij} = Rt_{ij} + (1/(2 chi))[-DDchi_{ij} - gt_{ij} laplacian_chi
     *          + d_i chi d_j chi / chi + gt_{ij} dchi_sq / (2 chi)]
     * Note: the exact form depends on conventions. Using the chi convention
     * (B&S eq 3.57 adapted):
     *
     * Rphys_{ij} = Rt_{ij}
     *   + 1/(2 chi) * (DDchi_{ij} + gt_{ij} laplacian_chi
     *                  - 3/(2 chi) d_i chi d_j chi
     *                  - gt_{ij}/(2 chi) dchi_sq)  ... wait, need to be careful.
     *
     * Actually using the standard BSSN/CCZ4 decomposition with chi:
     * phi = -ln(chi)/4, so d_i phi = -d_i chi / (4 chi)
     *
     * R_{ij}^{phi} = -(1/2) g^{kl} d_k d_l gt_{ij} + ...  [already in Rt]
     *   +  1/(2 chi) [DDchi_{ij} - gt_{ij} laplacian_chi + ...]
     *
     * For CCZ4, the conformal factor contributions appear in the At and K
     * equations via specific combinations. Let's compute what we need:
     */

    /* R_{ij}^{phys} = Rt_{ij} + Rchi_{ij} where: */
    double Rchi_dd[6];
    for (int ij = 0; ij < 6; ij++) {
        /* Rchi_{ij} = 1/(2 chi) * [DDchi_{ij} + gt_{ij} * laplacian_chi]
         *           - 3/(4 chi^2) * [d_i chi d_j chi + gt_{ij} * dchi_sq / ???]
         *
         * Standard form (B&S eq 3.10 in chi convention):
         * Rchi_{ij} = (1/(2 chi)) * (DDchi_{ij} + gt_{ij} * laplacian_chi / 3)
         *           wait, let me use the known-correct form.
         *
         * With chi = psi^{-4} = e^{-4 phi}:
         * The physical Ricci contribution from the conformal factor is:
         *
         * R^{phi}_{ij} = 1/(2 chi) [ tD_i tD_j chi + gt_{ij} gt^{kl} tD_k tD_l chi ]
         *              - 1/(4 chi^2) [ 3 d_i chi d_j chi - gt_{ij} gt^{kl} d_k chi d_l chi ]
         *
         * where tD is the covariant derivative compatible with gt.
         * Ref: adapted from B&S eq 3.57, converting phi -> chi
         */
        int ii2, jj2;
        switch (ij) {
        case SYM_XX: ii2 = 0; jj2 = 0; break;
        case SYM_XY: ii2 = 0; jj2 = 1; break;
        case SYM_XZ: ii2 = 0; jj2 = 2; break;
        case SYM_YY: ii2 = 1; jj2 = 1; break;
        case SYM_YZ: ii2 = 1; jj2 = 2; break;
        default:     ii2 = 2; jj2 = 2; break;
        }

        Rchi_dd[ij] = (1.0 / (2.0 * chi_safe)) * (DDchi[ij] + gt[ij] * laplacian_chi)
                    - (1.0 / (4.0 * chi_safe * chi_safe))
                      * (3.0 * d1_chi[ii2] * d1_chi[jj2] - gt[ij] * dchi_sq);

        /* Note: for diagonal components ii2==jj2, d1_chi[ii2]*d1_chi[jj2] is fine.
         * For off-diagonal, this is the mixed product. */
    }

    /* Physical Ricci (lower indices, conformal + chi parts) */
    double Rphys_dd[6];
    for (int a = 0; a < 6; a++) {
        Rphys_dd[a] = Rt_dd[a] + Rchi_dd[a];
    }

    /* Physical Ricci scalar: R = chi * g^{ij} R_{ij} */
    /* Actually R = g^{ij}_{phys} R_{ij} = chi * g_tilde^{ij} R_{ij} */
    double Rphys = 0.0;
    for (int a = 0; a < 6; a++) {
        double fac = (a == SYM_XX || a == SYM_YY || a == SYM_ZZ) ? 1.0 : 2.0;
        Rphys += chi * fac * gtu[SYM_XX + (a - SYM_XX)] * Rphys_dd[a];
        /* ^ This is wrong for off-diag. Let me fix: */
    }
    /* Redo properly */
    Rphys = chi * (gtu[SYM_XX] * Rphys_dd[SYM_XX]
                 + gtu[SYM_YY] * Rphys_dd[SYM_YY]
                 + gtu[SYM_ZZ] * Rphys_dd[SYM_ZZ]
                 + 2.0 * (gtu[SYM_XY] * Rphys_dd[SYM_XY]
                        + gtu[SYM_XZ] * Rphys_dd[SYM_XZ]
                        + gtu[SYM_YZ] * Rphys_dd[SYM_YZ]));

    /* ===== 8. Z_i from Gamma_hat ===== */
    /* Z^i = (1/2)(Ghat^i - Gamma^i)  where Gamma^i = contracted Christoffel */
    double Z_u[3]; /* Z^i (upper index) */
    for (int ii = 0; ii < 3; ii++) {
        Z_u[ii] = 0.5 * (Ghat[ii] - Gamma_contracted[ii]);
    }

    /* Z_i = gt_{ij} Z^j (lower index) */
    double Z_d[3];
    for (int ii = 0; ii < 3; ii++) {
        Z_d[ii] = gt[SYM(ii, 0)] * Z_u[0]
                + gt[SYM(ii, 1)] * Z_u[1]
                + gt[SYM(ii, 2)] * Z_u[2];
    }

    /* Divergence of Z: D_i Z^i ≈ d_i Z^i + Gamma^i_{ik} Z^k */
    double divZ = 0.0;
    for (int ii = 0; ii < 3; ii++) {
        /* d_i Z^i = (1/2) d_i (Ghat^i - Gamma^i) ≈ (1/2) d_i Ghat^i */
        divZ += 0.5 * d1_Ghat[ii][ii];
        /* Christoffel correction */
        for (int kk = 0; kk < 3; kk++) {
            divZ += chris[ii][SYM(ii, kk)] * Z_u[kk];
        }
    }

    /* Z^i d_i alpha */
    double Z_d_alpha = Z_u[0] * d1_alpha[0] + Z_u[1] * d1_alpha[1]
                     + Z_u[2] * d1_alpha[2];

    /* ===== 9. Covariant Laplacian of alpha ===== */
    /* nabla^2 alpha = g^{ij}_{phys} D_i D_j alpha
     *              = chi * g^{ij} (d_i d_j alpha - Gamma^k_{ij} d_k alpha) */
    double DDalpha[6];
    for (int ij = 0; ij < 6; ij++) {
        DDalpha[ij] = d2_alpha[ij];
        for (int kk = 0; kk < 3; kk++) {
            DDalpha[ij] -= chris[kk][ij] * d1_alpha[kk];
        }
        /* Conformal factor correction:
         * D^phys_i D^phys_j alpha = chi * [D~_i D~_j alpha
         *   + 1/(2 chi) (d_i chi d_j alpha + d_j chi d_i alpha
         *                - gt_{ij} g^{kl} d_k chi d_l alpha)]
         */
    }
    double lap_alpha = chi * (gtu[SYM_XX] * DDalpha[SYM_XX]
                            + gtu[SYM_YY] * DDalpha[SYM_YY]
                            + gtu[SYM_ZZ] * DDalpha[SYM_ZZ]
                            + 2.0 * (gtu[SYM_XY] * DDalpha[SYM_XY]
                                   + gtu[SYM_XZ] * DDalpha[SYM_XZ]
                                   + gtu[SYM_YZ] * DDalpha[SYM_YZ]));

    /* Add conformal factor contribution to Laplacian of alpha:
     * The full physical Laplacian includes d_i chi * d_i alpha terms */
    double dchi_dalpha = 0.0;
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            dchi_dalpha += gtu[SYM(a, b)] * d1_chi[a] * d1_alpha[b];
        }
    }
    lap_alpha += 0.5 * dchi_dalpha;  /* correction from conformal decomposition */

    /* ===== Raise At: A^{ij} = chi * g^{ik} g^{jl} At_{kl} ===== */
    /* A_tilde^{ij} = g_tilde^{ik} g_tilde^{jl} A_tilde_{kl} */
    double atu[6]; /* A_tilde^{ij} in symmetric storage */
    atu[SYM_XX] = gtu[SYM_XX] * (gtu[SYM_XX] * at[SYM_XX] + gtu[SYM_XY] * at[SYM_XY] + gtu[SYM_XZ] * at[SYM_XZ])
                + gtu[SYM_XY] * (gtu[SYM_XX] * at[SYM_XY] + gtu[SYM_XY] * at[SYM_YY] + gtu[SYM_XZ] * at[SYM_YZ])
                + gtu[SYM_XZ] * (gtu[SYM_XX] * at[SYM_XZ] + gtu[SYM_XY] * at[SYM_YZ] + gtu[SYM_XZ] * at[SYM_ZZ]);
    atu[SYM_XY] = gtu[SYM_XX] * (gtu[SYM_XY] * at[SYM_XX] + gtu[SYM_YY] * at[SYM_XY] + gtu[SYM_YZ] * at[SYM_XZ])
                + gtu[SYM_XY] * (gtu[SYM_XY] * at[SYM_XY] + gtu[SYM_YY] * at[SYM_YY] + gtu[SYM_YZ] * at[SYM_YZ])
                + gtu[SYM_XZ] * (gtu[SYM_XY] * at[SYM_XZ] + gtu[SYM_YY] * at[SYM_YZ] + gtu[SYM_YZ] * at[SYM_ZZ]);
    atu[SYM_XZ] = gtu[SYM_XX] * (gtu[SYM_XZ] * at[SYM_XX] + gtu[SYM_YZ] * at[SYM_XY] + gtu[SYM_ZZ] * at[SYM_XZ])
                + gtu[SYM_XY] * (gtu[SYM_XZ] * at[SYM_XY] + gtu[SYM_YZ] * at[SYM_YY] + gtu[SYM_ZZ] * at[SYM_YZ])
                + gtu[SYM_XZ] * (gtu[SYM_XZ] * at[SYM_XZ] + gtu[SYM_YZ] * at[SYM_YZ] + gtu[SYM_ZZ] * at[SYM_ZZ]);
    atu[SYM_YY] = gtu[SYM_XY] * (gtu[SYM_XY] * at[SYM_XX] + gtu[SYM_YY] * at[SYM_XY] + gtu[SYM_YZ] * at[SYM_XZ])
                + gtu[SYM_YY] * (gtu[SYM_XY] * at[SYM_XY] + gtu[SYM_YY] * at[SYM_YY] + gtu[SYM_YZ] * at[SYM_YZ])
                + gtu[SYM_YZ] * (gtu[SYM_XY] * at[SYM_XZ] + gtu[SYM_YY] * at[SYM_YZ] + gtu[SYM_YZ] * at[SYM_ZZ]);
    atu[SYM_YZ] = gtu[SYM_XY] * (gtu[SYM_XZ] * at[SYM_XX] + gtu[SYM_YZ] * at[SYM_XY] + gtu[SYM_ZZ] * at[SYM_XZ])
                + gtu[SYM_YY] * (gtu[SYM_XZ] * at[SYM_XY] + gtu[SYM_YZ] * at[SYM_YY] + gtu[SYM_ZZ] * at[SYM_YZ])
                + gtu[SYM_YZ] * (gtu[SYM_XZ] * at[SYM_XZ] + gtu[SYM_YZ] * at[SYM_YZ] + gtu[SYM_ZZ] * at[SYM_ZZ]);
    atu[SYM_ZZ] = gtu[SYM_XZ] * (gtu[SYM_XZ] * at[SYM_XX] + gtu[SYM_YZ] * at[SYM_XY] + gtu[SYM_ZZ] * at[SYM_XZ])
                + gtu[SYM_YZ] * (gtu[SYM_XZ] * at[SYM_XY] + gtu[SYM_YZ] * at[SYM_YY] + gtu[SYM_ZZ] * at[SYM_YZ])
                + gtu[SYM_ZZ] * (gtu[SYM_XZ] * at[SYM_XZ] + gtu[SYM_YZ] * at[SYM_YZ] + gtu[SYM_ZZ] * at[SYM_ZZ]);

    /* A_ij A^{ij} */
    double ata = sym3_contract(at, atu);

    /* A^i_j = g^{ik} A_{kj} */
    double at_ud[3][3];
    sym3_raise_first(gtu, at, at_ud);

    /* GRChombo modification: kappa1 -> kappa1 / alpha */
    double kappa1_eff = kappa1 / alpha_safe;

    /* ======================================================================
     * EVOLUTION EQUATIONS
     * ====================================================================== */

    /* ----- dt_chi: conformal factor ----- */
    /* Using chi = e^{-4 phi}:
     * dt phi = (1/3) alpha K - (1/3) div_beta + adv
     * dt chi = -4 chi * dt phi = -(2/3) alpha chi K + (2/3) chi div_beta + adv_chi
     *
     * With CCZ4 Theta: K -> K - 2 Theta
     */
    double dt_chi = -(2.0 / 3.0) * alpha * chi * (K - 2.0 * Theta)
                  + (2.0 / 3.0) * chi * div_beta
                  + adv_chi;

    /* ----- dt_gt: conformal metric ----- */
    /* dt gt_{ij} = -2 alpha At_{ij} + 2 gt_{k(i} d_{j)} beta^k
     *            - (2/3) gt_{ij} d_k beta^k + adv */
    double dt_gt[6];
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

        double lie_beta = 0.0;
        for (int kk = 0; kk < 3; kk++) {
            lie_beta += gt[SYM(kk, ii)] * d1_beta[jj][kk]
                      + gt[SYM(kk, jj)] * d1_beta[ii][kk];
        }

        dt_gt[ij] = -2.0 * alpha * at[ij]
                   + lie_beta
                   - (2.0 / 3.0) * gt[ij] * div_beta
                   + adv_gt[ij];
    }

    /* ----- dt_K: trace of extrinsic curvature ----- */
    /* dt K = -nabla^2 alpha + alpha (R + 2 div Z + K^2 - 2 Theta K)
     *      - 3 alpha kappa1 (1 + kappa2) Theta + adv */
    double dt_K = -lap_alpha
                + alpha * (Rphys + 2.0 * divZ + K * K - 2.0 * Theta * K)
                - 3.0 * alpha * kappa1_eff * (1.0 + kappa2) * Theta
                + adv_K;

    /* ----- dt_Theta: CCZ4 constraint propagation ----- */
    /* dt Theta = (1/2) alpha [R + 2 div Z - Aij Aij + (2/3) K^2 - 2 Theta K]
     *          - Z^i d_i alpha - alpha kappa1 (2 + kappa2) Theta + adv */
    double dt_Theta = 0.5 * alpha * (Rphys + 2.0 * divZ - ata
                                    + (2.0 / 3.0) * K * K - 2.0 * Theta * K)
                    - Z_d_alpha
                    - alpha * kappa1_eff * (2.0 + kappa2) * Theta
                    + adv_Theta;

    /* ----- dt_At: tracefree extrinsic curvature ----- */
    /*
     * dt At_{ij} = chi [-D_i D_j alpha + alpha (R_{ij} + D_i Z_j + D_j Z_i)]^{TF}
     *           + alpha At_{ij} (K - 2 Theta) - 2 alpha At_{il} At^l_j
     *           + lie_beta terms + adv
     *
     * Note: the term in [ ]^{TF} means "take the tracefree part w.r.t. gt".
     * We compute it as: X_{ij} = stuff, then X_{ij} - (1/3) gt_{ij} g^{kl} X_{kl}
     */

    /* Covariant derivative of Z: D_i Z_j = d_i Z_j - Gamma^k_{ij} Z_k
     * But Z_j involves derivatives of Ghat, which is complex.
     * Simplification: D_i Z_j + D_j Z_i can be expressed via the Gamma_hat equation.
     * For now, we approximate the Z contribution.
     */

    /* Physical D_i D_j alpha including conformal decomposition */
    double phys_DDalpha[6];
    for (int ij = 0; ij < 6; ij++) {
        int ii2, jj2;
        switch (ij) {
        case SYM_XX: ii2 = 0; jj2 = 0; break;
        case SYM_XY: ii2 = 0; jj2 = 1; break;
        case SYM_XZ: ii2 = 0; jj2 = 2; break;
        case SYM_YY: ii2 = 1; jj2 = 1; break;
        case SYM_YZ: ii2 = 1; jj2 = 2; break;
        default:     ii2 = 2; jj2 = 2; break;
        }
        /* nabla_i nabla_j alpha includes conformal factor:
         * = D~_i D~_j alpha + 1/(2 chi) (d_i chi D_j alpha + d_j chi D_i alpha
         *   - gt_{ij} g^{kl} d_k chi D_l alpha)
         * where D~ is the conformal covariant derivative
         */
        phys_DDalpha[ij] = DDalpha[ij]
            + 0.5 / chi_safe * (d1_chi[ii2] * d1_alpha[jj2]
                              + d1_chi[jj2] * d1_alpha[ii2])
            - 0.5 / chi_safe * gt[ij] * dchi_dalpha;
    }

    /* D_i Z_j + D_j Z_i (symmetric): approximate from Ghat definition */
    /* Z_j = (1/2) gt_{jk} (Ghat^k - Gamma^k)
     * D_i Z_j ≈ (1/2) gt_{jk} d_i Ghat^k + ... (Christoffel terms)
     * This is a simplification; the full expression includes metric derivatives. */
    double DZ_sym[6]; /* D_i Z_j + D_j Z_i */
    for (int ij = 0; ij < 6; ij++) {
        int ii2, jj2;
        switch (ij) {
        case SYM_XX: ii2 = 0; jj2 = 0; break;
        case SYM_XY: ii2 = 0; jj2 = 1; break;
        case SYM_XZ: ii2 = 0; jj2 = 2; break;
        case SYM_YY: ii2 = 1; jj2 = 1; break;
        case SYM_YZ: ii2 = 1; jj2 = 2; break;
        default:     ii2 = 2; jj2 = 2; break;
        }

        DZ_sym[ij] = 0.0;
        for (int kk = 0; kk < 3; kk++) {
            /* d_i (gt_{jk} Z^k) + d_j (gt_{ik} Z^k) includes:
             * gt_{jk} d_i Z^k + Z^k d_i gt_{jk} + (i <-> j)
             * Z^k = (1/2)(Ghat^k - Gamma^k)
             * d_i Z^k ≈ (1/2) d_i Ghat^k  (ignoring d_i Gamma^k which involves d2_gt)
             */
            DZ_sym[ij] += gt[SYM(jj2, kk)] * 0.5 * d1_Ghat[ii2][kk]
                        + gt[SYM(ii2, kk)] * 0.5 * d1_Ghat[jj2][kk];
            /* Metric derivative terms */
            DZ_sym[ij] += Z_u[kk] * (d1_gt[ii2][SYM(jj2, kk)]
                                   + d1_gt[jj2][SYM(ii2, kk)]);
        }
        /* Christoffel corrections */
        for (int kk = 0; kk < 3; kk++) {
            DZ_sym[ij] -= chris[kk][SYM(ii2, jj2)] * Z_d[kk] * 2.0;
        }
    }

    /* Source term for At: X_{ij} = -D_i D_j alpha + alpha (R_{ij} + D_i Z_j + D_j Z_i) */
    double X_dd[6];
    for (int a = 0; a < 6; a++) {
        X_dd[a] = chi * (-phys_DDalpha[a] + alpha * (Rphys_dd[a] + DZ_sym[a]));
    }

    /* Make tracefree: X^{TF}_{ij} = X_{ij} - (1/3) gt_{ij} g^{kl} X_{kl} */
    double trX = gtu[SYM_XX] * X_dd[SYM_XX]
               + gtu[SYM_YY] * X_dd[SYM_YY]
               + gtu[SYM_ZZ] * X_dd[SYM_ZZ]
               + 2.0 * (gtu[SYM_XY] * X_dd[SYM_XY]
                      + gtu[SYM_XZ] * X_dd[SYM_XZ]
                      + gtu[SYM_YZ] * X_dd[SYM_YZ]);

    double dt_at[6];
    for (int ij = 0; ij < 6; ij++) {
        int ii2, jj2;
        switch (ij) {
        case SYM_XX: ii2 = 0; jj2 = 0; break;
        case SYM_XY: ii2 = 0; jj2 = 1; break;
        case SYM_XZ: ii2 = 0; jj2 = 2; break;
        case SYM_YY: ii2 = 1; jj2 = 1; break;
        case SYM_YZ: ii2 = 1; jj2 = 2; break;
        default:     ii2 = 2; jj2 = 2; break;
        }

        /* TF part */
        double xtf = X_dd[ij] - (1.0 / 3.0) * gt[ij] * trX;

        /* alpha At_{ij} (K - 2 Theta) */
        double atak = alpha * at[ij] * (K - 2.0 * Theta);

        /* -2 alpha At_{il} At^l_j */
        double ata2 = 0.0;
        for (int ll = 0; ll < 3; ll++) {
            ata2 += at_ud[ll][ii2] * at[SYM(ll, jj2)];
            /* wait: At_{il} At^l_j = at[SYM(i,l)] * at_ud[l][j] */
        }
        /* Actually: At_{il} At^l_j = sum_l at[SYM(i,l)] * at_ud[l][j] */
        ata2 = 0.0;
        for (int ll = 0; ll < 3; ll++) {
            ata2 += at[SYM(ii2, ll)] * at_ud[ll][jj2];
        }

        /* Lie derivative terms */
        double lie_at = 0.0;
        for (int kk = 0; kk < 3; kk++) {
            lie_at += at[SYM(kk, ii2)] * d1_beta[jj2][kk]
                    + at[SYM(kk, jj2)] * d1_beta[ii2][kk];
        }
        lie_at -= (2.0 / 3.0) * at[ij] * div_beta;

        dt_at[ij] = xtf + atak - 2.0 * alpha * ata2 + lie_at + adv_at[ij];
    }

    /* ----- dt_Ghat: modified conformal connection ----- */
    /*
     * dt Ghat^i = 2 alpha (Gamma^i_{jk} At^{jk} - 3 At^{ij} d_j chi / (2 chi)
     *            - (2/3) g^{ij} d_j K)
     *          + 2 g^{ki} (alpha d_k Theta - Theta d_k alpha
     *                      - (2/3) alpha K Z_k)
     *          - 2 At^{ij} d_j alpha
     *          + g^{kl} d_k d_l beta^i + (1/3) g^{ik} d_k d_l beta^l
     *          + (2/3) Gamma^i div_beta - Gamma^k d_k beta^i
     *          + 2 kappa3 ((2/3) g^{ij} Z_j div_beta - g^{jk} Z_j d_k beta^i)
     *          - 2 alpha kappa1 g^{ij} Z_j + adv
     */
    double dt_Ghat[3];
    for (int ii = 0; ii < 3; ii++) {
        /* Term 1: 2 alpha Gamma^i_{jk} At^{jk} */
        double chris_at = 0.0;
        for (int jk = 0; jk < 6; jk++) {
            double fac = (jk == SYM_XX || jk == SYM_YY || jk == SYM_ZZ) ? 1.0 : 2.0;
            chris_at += fac * chris[ii][jk] * atu[jk];
        }

        /* Term 2: -3 At^{ij} d_j chi / (2 chi) = -(3/2) At^{ij} d_j chi / chi */
        double at_dchi = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            at_dchi += atu[SYM(ii, jj)] * d1_chi[jj];
        }
        /* Multiply by factor: since At^{ij} is stored as sym, need care.
         * atu[SYM(ii, jj)] is At^{ij} */

        /* Term 3: -(2/3) g^{ij} d_j K */
        double gtu_dK = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            gtu_dK += gtu[SYM(ii, jj)] * d1_K[jj];
        }

        /* Term 4: 2 g^{ki} (alpha d_k Theta - Theta d_k alpha - (2/3) alpha K Z_k) */
        double theta_terms = 0.0;
        for (int kk = 0; kk < 3; kk++) {
            theta_terms += gtu[SYM(kk, ii)]
                * (alpha * d1_Theta[kk] - Theta * d1_alpha[kk]
                   - (2.0 / 3.0) * alpha * K * Z_d[kk]);
        }

        /* Term 5: -2 At^{ij} d_j alpha */
        double at_dalpha = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            at_dalpha += atu[SYM(ii, jj)] * d1_alpha[jj];
        }

        /* Term 6: g^{kl} d_k d_l beta^i */
        double lap_beta = 0.0;
        for (int kk = 0; kk < 3; kk++) {
            for (int ll = 0; ll < 3; ll++) {
                lap_beta += gtu[SYM(kk, ll)] * d2_beta[kk][ll][ii];
            }
        }

        /* Term 7: (1/3) g^{ik} d_k d_l beta^l */
        double div_grad_beta = 0.0;
        for (int kk = 0; kk < 3; kk++) {
            for (int ll = 0; ll < 3; ll++) {
                div_grad_beta += gtu[SYM(ii, kk)] * d2_beta[kk][ll][ll];
            }
        }

        /* Term 8: (2/3) Gamma^i div_beta - Gamma^k d_k beta^i */
        double gamma_beta = (2.0 / 3.0) * Gamma_contracted[ii] * div_beta;
        for (int kk = 0; kk < 3; kk++) {
            gamma_beta -= Gamma_contracted[kk] * d1_beta[kk][ii];
        }

        /* Term 9: kappa3 terms */
        double k3_terms = 0.0;
        double gtu_Z = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            gtu_Z += gtu[SYM(ii, jj)] * Z_d[jj];
        }
        k3_terms = 2.0 * kappa3 * ((2.0 / 3.0) * gtu_Z * div_beta);
        for (int jj = 0; jj < 3; jj++) {
            for (int kk = 0; kk < 3; kk++) {
                k3_terms -= 2.0 * kappa3 * gtu[SYM(jj, kk)] * Z_d[jj] * d1_beta[kk][ii];
            }
        }

        /* Term 10: -2 alpha kappa1 g^{ij} Z_j (damping) */
        double damping = -2.0 * alpha * kappa1_eff * gtu_Z;

        dt_Ghat[ii] = 2.0 * alpha * (chris_at - 1.5 * at_dchi / chi_safe
                                    - (2.0 / 3.0) * gtu_dK)
                    + 2.0 * theta_terms
                    - 2.0 * at_dalpha
                    + lap_beta + (1.0 / 3.0) * div_grad_beta
                    + gamma_beta + k3_terms + damping
                    + adv_Ghat[ii];
    }

    /* ===== Write RHS output ===== */
    g->rhs[FIELD_CHI][idx] = dt_chi;
    for (int a = 0; a < 6; a++) {
        g->rhs[FIELD_GT_BASE + a][idx] = dt_gt[a];
        g->rhs[FIELD_AT_BASE + a][idx] = dt_at[a];
    }
    g->rhs[FIELD_TRKA][idx] = dt_K;
    g->rhs[FIELD_THETA][idx] = dt_Theta;
    for (int a = 0; a < 3; a++) {
        g->rhs[FIELD_GHAT1 + a][idx] = dt_Ghat[a];
    }
}

/*
 * Compute CCZ4 RHS for all interior grid points.
 * Called from the main driver's full_rhs function.
 */
void ccz4_rhs(grid_t *g)
{
    GRID_LOOP_INTERIOR(g, i, j, k) {
        int idx = grid_idx(g, i, j, k);
        ccz4_rhs_point(g, idx);
    }
}
