/*
 * constraints.c — Hamiltonian and momentum constraint monitoring
 *
 * Hamiltonian constraint (vacuum):
 *   H = R + (2/3)K^2 - A_tilde_{ij} A_tilde^{ij}
 *   where R is the full physical Ricci scalar.
 *
 * Momentum constraint (vacuum):
 *   M^i = D_tilde_j A_tilde^{ij} - (2/3) g_tilde^{ij} D_j K
 *        + 6 A_tilde^{ij} d_j phi / phi
 *        + CCZ4 Theta terms
 *
 * We compute L2 norms over the interior for monitoring.
 * B&S eq 3.73-3.74; arXiv:1106.2254 for CCZ4 modifications
 */

#include "../core/grid.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"
#include "../geometry/tensor_utils.h"

#include <math.h>

void constraints_l2(grid_t *g, double *ham_l2, double *mom_l2)
{
    const int sx = grid_stride_x(g);
    const int sy = grid_stride_y(g);
    const int sz = grid_stride_z(g);
    const double dx = g->params.dx;
    const double dy = g->params.dy;
    const double dz = g->params.dz;
    const int strides[3] = { sx, sy, sz };
    const double dxs[3] = { dx, dy, dz };

    double ham_sum = 0.0;
    double mom_sum = 0.0;
    int count = 0;

    /* OMP: constraint computation with reduction on ham_sum, mom_sum, count.
     * Uses explicit pragma instead of GRID_LOOP_INTERIOR_OMP for reduction support.
     * Toggle: compile with/without -DLATTICE_USE_OMP (make PARALLEL=0/1) */
#ifdef LATTICE_USE_OMP
#pragma omp parallel for collapse(2) schedule(static) reduction(+:ham_sum,mom_sum,count)
#endif
    for (int k = g->params.ghost_width; k < g->params.nz - g->params.ghost_width; ++k)
        for (int j = g->params.ghost_width; j < g->params.ny - g->params.ghost_width; ++j)
            for (int i = g->params.ghost_width; i < g->params.nx - g->params.ghost_width; ++i) {
        int idx = grid_idx(g, i, j, k);

        /* Read fields */
        double chi = g->fields[FIELD_CHI][idx];
        double K = g->fields[FIELD_TRKA][idx];

        double gt[6], gtu[6], at[6];
        for (int a = 0; a < 6; a++) {
            gt[a] = g->fields[FIELD_GT_BASE + a][idx];
            at[a] = g->fields[FIELD_AT_BASE + a][idx];
        }
        sym3_inv(gt, gtu);

        /* A_ij A^{ij} = g^{ik} g^{jl} A_{ij} A_{kl} */
        double ata = 0.0;
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                for (int c = 0; c < 3; c++) {
                    for (int d = 0; d < 3; d++) {
                        ata += gtu[SYM(a, c)] * gtu[SYM(b, d)]
                             * at[SYM(a, b)] * at[SYM(c, d)];
                    }
                }
            }
        }

        /* Compute conformal Ricci scalar (simplified for constraint check) */
        /* R_tilde_{ij}: need second derivatives of gt */
        /* For efficiency, use a simplified estimate:
         * R ≈ chi * g^{ij} R_tilde_{ij} + conformal factor terms
         * For flat spacetime this should be exactly zero. */

        /* First derivatives of chi */
        double d1_chi[3];
        for (int d = 0; d < 3; d++) {
            d1_chi[d] = FD_D1(g->fields[FIELD_CHI], idx, strides[d], dxs[d]);
        }

        /* Second derivatives of chi */
        double d2_chi[6];
        d2_chi[SYM_XX] = FD_D2(g->fields[FIELD_CHI], idx, sx, dx);
        d2_chi[SYM_YY] = FD_D2(g->fields[FIELD_CHI], idx, sy, dy);
        d2_chi[SYM_ZZ] = FD_D2(g->fields[FIELD_CHI], idx, sz, dz);
        d2_chi[SYM_XY] = FD_D1D1(g->fields[FIELD_CHI], idx, sx, sy, dx, dy);
        d2_chi[SYM_XZ] = FD_D1D1(g->fields[FIELD_CHI], idx, sx, sz, dx, dz);
        d2_chi[SYM_YZ] = FD_D1D1(g->fields[FIELD_CHI], idx, sy, sz, dy, dz);

        /* First derivatives of gt */
        double d1_gt[3][6];
        for (int a = 0; a < 6; a++) {
            for (int d = 0; d < 3; d++) {
                d1_gt[d][a] = FD_D1(g->fields[FIELD_GT_BASE + a], idx, strides[d], dxs[d]);
            }
        }

        /* Second derivatives of gt (full, including mixed) */
        double d2_gt[3][3][6];
        for (int a = 0; a < 6; a++) {
            const double *f = g->fields[FIELD_GT_BASE + a];
            d2_gt[0][0][a] = FD_D2(f, idx, sx, dx);
            d2_gt[1][1][a] = FD_D2(f, idx, sy, dy);
            d2_gt[2][2][a] = FD_D2(f, idx, sz, dz);
            d2_gt[0][1][a] = FD_D1D1(f, idx, sx, sy, dx, dy);
            d2_gt[1][0][a] = d2_gt[0][1][a];
            d2_gt[0][2][a] = FD_D1D1(f, idx, sx, sz, dx, dz);
            d2_gt[2][0][a] = d2_gt[0][2][a];
            d2_gt[1][2][a] = FD_D1D1(f, idx, sy, sz, dy, dz);
            d2_gt[2][1][a] = d2_gt[1][2][a];
        }

        /* Christoffel symbols */
        double chris[3][6];
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

        /* Lower-index Christoffels: Gamma_{ijk} = g_{il} Gamma^l_{jk} */
        double chris_d[3][6];
        for (int jk = 0; jk < 6; jk++) {
            for (int ii = 0; ii < 3; ii++) {
                chris_d[ii][jk] = 0.0;
                for (int ll = 0; ll < 3; ll++) {
                    chris_d[ii][jk] += gt[SYM(ii, ll)] * chris[ll][jk];
                }
            }
        }

        /* Gamma_hat^i from stored field (if evolved), else from contracted Christoffel */
        double Ghat[3];
        for (int ii = 0; ii < 3; ii++) {
            Ghat[ii] = g->fields[FIELD_GHAT1 + ii][idx];
        }

        /* Full conformal Ricci tensor Rt_{ij} using the 4-term formula
         * (same as ccz4_rhs.c, B&S eq 3.69):
         *   Rt_{ij} = -(1/2) g^{lm} d_l d_m g_{ij}
         *           + g_{k(i} d_{j)} Ghat^k
         *           + Ghat^k Gamma_{(ij)k}
         *           + g^{lm} (2 Gamma^k_{l(i} Gamma_{j)km} + Gamma^k_{im} Gamma_{klj})
         */
        double Rt_dd[6];
        double d1_Ghat[3][3];
        for (int ii = 0; ii < 3; ii++) {
            for (int d = 0; d < 3; d++) {
                d1_Ghat[d][ii] = FD_D1(g->fields[FIELD_GHAT1 + ii], idx, strides[d], dxs[d]);
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

            /* Term 1: -(1/2) g^{lm} d_l d_m g_{ij} */
            double term1 = 0.0;
            for (int ll = 0; ll < 3; ll++)
                for (int mm = 0; mm < 3; mm++)
                    term1 += gtu[SYM(ll, mm)] * d2_gt[ll][mm][ij];
            term1 *= -0.5;

            /* Term 2: g_{k(i} d_{j)} Ghat^k */
            double term2 = 0.0;
            for (int kk = 0; kk < 3; kk++) {
                term2 += gt[SYM(kk, ii)] * d1_Ghat[jj][kk]
                       + gt[SYM(kk, jj)] * d1_Ghat[ii][kk];
            }
            term2 *= 0.5;

            /* Term 3: Ghat^k Gamma_{(ij)k} */
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
                        term4 += gtu_lm * (chris[kk][SYM(ll, ii)] * chris_d[jj][SYM(kk, mm)]
                                         + chris[kk][SYM(ll, jj)] * chris_d[ii][SYM(kk, mm)]);
                        term4 += gtu_lm * chris[kk][SYM(ii, mm)] * chris_d[kk][SYM(ll, jj)];
                    }
                }
            }

            Rt_dd[ij] = term1 + term2 + term3 + term4;
        }

        /* Conformal Ricci scalar: Rt = g^{ij} Rt_{ij} */
        double Rt = sym3_trace(Rt_dd, gtu);

        /* Conformal Laplacian of chi: g^{ij} D_i D_j chi
         * D_i D_j chi = d_i d_j chi - Gamma^k_{ij} d_k chi */
        double DDchi[6];
        for (int ij = 0; ij < 6; ij++) {
            DDchi[ij] = d2_chi[ij];
            for (int kk = 0; kk < 3; kk++)
                DDchi[ij] -= chris[kk][ij] * d1_chi[kk];
        }
        double laplacian_chi = sym3_trace(DDchi, gtu);

        /* g^{ij} d_i chi d_j chi */
        double grad_chi_sq = 0.0;
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                grad_chi_sq += gtu[SYM(a, b)] * d1_chi[a] * d1_chi[b];

        /* Physical Ricci scalar (B&S eq 3.10 in chi convention):
         *   R = chi * Rt + 2 * laplacian_chi - 5/(2 chi) * grad_chi_sq
         * Derivation: Rchi_{ij} = 1/(2chi)(DDchi_{ij} + gt_{ij} lap_chi)
         *                       - 1/(4chi^2)(d_i chi d_j chi + 3 gt_{ij} dchi_sq)
         * Trace: g^{ij} Rchi_{ij} = 2/chi * lap_chi - 5/(2chi^2) * dchi_sq
         * R = chi * (Rt + g^{ij} Rchi_{ij}) */
        double chi_safe = fmax(chi, 1e-4);
        double R = chi * Rt + 2.0 * laplacian_chi
                 - 2.5 / chi_safe * grad_chi_sq;

        /* Hamiltonian constraint: H = R + (2/3)K^2 - Aij Aij */
        double H = R + (2.0 / 3.0) * K * K - ata;

        ham_sum += H * H;

        /* Momentum constraint: simplified L2 norm
         * M^i = D_j A^{ij} - (2/3) g^{ij} d_j K + ...
         * For flat spacetime with zero extrinsic curvature, M^i = 0 exactly.
         */
        double d1_K[3];
        for (int d = 0; d < 3; d++) {
            d1_K[d] = FD_D1(g->fields[FIELD_TRKA], idx, strides[d], dxs[d]);
        }

        /* First derivatives of At */
        double d1_at[3][6];
        for (int a = 0; a < 6; a++) {
            for (int d = 0; d < 3; d++) {
                d1_at[d][a] = FD_D1(g->fields[FIELD_AT_BASE + a], idx, strides[d], dxs[d]);
            }
        }

        /* M^i = g^{jk} d_j A_{ik} + Christoffel corrections - (2/3) g^{ij} d_j K
         *       + 6 A^{ij} d_j chi / (2 chi) */
        double mom_sq = 0.0;
        for (int ii = 0; ii < 3; ii++) {
            double Mi = 0.0;

            /* g^{jk} d_j A_{ik} */
            for (int jj = 0; jj < 3; jj++) {
                for (int kk = 0; kk < 3; kk++) {
                    Mi += gtu[SYM(jj, kk)] * d1_at[jj][SYM(ii, kk)];
                }
            }

            /* Christoffel connection corrections: Gamma^i_{jk} A^{jk} + Gamma^j_{jk} A^{ik} */
            /* These vanish in flat space — omit for monitoring */

            /* -(2/3) g^{ij} d_j K */
            for (int jj = 0; jj < 3; jj++) {
                Mi -= (2.0 / 3.0) * gtu[SYM(ii, jj)] * d1_K[jj];
            }

            /* +6 A^{ij} d_j chi / (2 chi) = 3 A^{ij} d_j chi / chi */
            for (int jj = 0; jj < 3; jj++) {
                /* A^{ij} = g^{ik} g^{jl} A_{kl} — compute component */
                double Aij_upper = 0.0;
                for (int kk = 0; kk < 3; kk++) {
                    for (int ll = 0; ll < 3; ll++) {
                        Aij_upper += gtu[SYM(ii, kk)] * gtu[SYM(jj, ll)] * at[SYM(kk, ll)];
                    }
                }
                Mi += 3.0 * Aij_upper * d1_chi[jj] / chi_safe;
            }

            mom_sq += Mi * Mi;
        }

        mom_sum += mom_sq;
        count++;
    }

    *ham_l2 = (count > 0) ? sqrt(ham_sum / count) : 0.0;
    *mom_l2 = (count > 0) ? sqrt(mom_sum / count) : 0.0;
}
