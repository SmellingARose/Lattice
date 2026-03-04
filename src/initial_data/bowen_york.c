/*
 * Lattice — 3D Numerical Relativity
 * Bowen-York initial data: A_ij with momentum and spin.
 *
 * Momentum term (Eq. 3.43 of B&S, 1/r^2 falloff):
 *   A_ij^P = (3/(2r^2)) [P_i n_j + P_j n_i - (delta_ij - n_i n_j)(P.n)]
 *
 * Spin term (Eq. 3.44 of B&S, 1/r^3 falloff):
 *   A_ij^S = -(3/r^3) [eps_{kil} S^l n^k n_j + eps_{kjl} S^l n^k n_i]
 *          = -(3/r^3) [(nxS)_i n_j + (nxS)_j n_i]
 *
 * Ref: gr-qc/9703066 (Brandt-Brugmann)
 * Ref: GRChombo BoostedBH.impl.hpp:47-50
 * Ref: TwoPunctures Equations.cc:BY_Aijofxyz()
 */

#include "bowen_york.h"
#include "relaxation.h"
#include "relaxation_amr.h"
#include "kerr_quasi_isotropic.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"
#include "../geometry/tensor_utils.h"
#include "../amr/ghost_exchange.h"
#include "../amr/restriction.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Solver field slot indices (must match relaxation_amr.c) */
#define SOL_PSI_BY   0
#define BG_PSI_BL_BY 4

void bowen_york_Aij(double A_tilde[3][3], double x, double y, double z,
                    int n_bh, const puncture_data_t *bhs)
{
    memset(A_tilde, 0, sizeof(double) * 9);

    for (int n = 0; n < n_bh; n++) {
        double rx = x - bhs[n].center[0];
        double ry = y - bhs[n].center[1];
        double rz = z - bhs[n].center[2];
        double r2 = rx*rx + ry*ry + rz*rz;
        double r  = sqrt(r2);
        if (r < 1.0e-10) r = 1.0e-10;

        double s[3] = { rx/r, ry/r, rz/r };  /* unit normal n_i */

        const double *P = bhs[n].momentum;
        const double *S = bhs[n].spin;

        /* Momentum contribution: A_ij^P = (3/(2r^2)) [P_i n_j + P_j n_i
         *   - (delta_ij - n_i n_j)(P.n)]
         * Ref: B&S Eq. 3.43, GRChombo BoostedBH.impl.hpp:47-50 */
        double Pdotn = P[0]*s[0] + P[1]*s[1] + P[2]*s[2];
        double fac_P = 1.5 / (r * r);  /* 3/(2r^2) */

        for (int i = 0; i < 3; i++) {
            for (int j = i; j < 3; j++) {
                double dij = (i == j) ? 1.0 : 0.0;
                double A_P = fac_P * (P[i]*s[j] + P[j]*s[i]
                             - (dij - s[i]*s[j]) * Pdotn);
                A_tilde[i][j] += A_P;
                if (j != i) A_tilde[j][i] += A_P;
            }
        }

        /* Spin contribution: A_ij^S = -(3/r^3) [(nxS)_i n_j + (nxS)_j n_i]
         * Ref: B&S Eq. 3.44, TwoPunctures Equations.cc:BY_Aijofxyz() */
        double nxS[3];
        nxS[0] = s[1]*S[2] - s[2]*S[1];
        nxS[1] = s[2]*S[0] - s[0]*S[2];
        nxS[2] = s[0]*S[1] - s[1]*S[0];

        double fac_S = -3.0 / (r * r * r);

        for (int i = 0; i < 3; i++) {
            for (int j = i; j < 3; j++) {
                double A_S = fac_S * (nxS[i]*s[j] + nxS[j]*s[i]);
                A_tilde[i][j] += A_S;
                if (j != i) A_tilde[j][i] += A_S;
            }
        }
    }
}

double bowen_york_A2(const double A_tilde[3][3])
{
    /* Flat metric contraction: A^ij A_ij = sum_{i,j} A_ij^2 */
    double A2 = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            A2 += A_tilde[i][j] * A_tilde[i][j];
    return A2;
}

double hispid_A2(const double A[3][3], const double h[3][3])
{
    /* Conformal metric contraction: A^ij A_ij = h^{ik} h^{jl} A_{kl} A_{ij} */
    double h_UU[3][3];
    compute_inverse_sym(h, h_UU);
    double A2 = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++)
                    A2 += h_UU[i][k] * h_UU[j][l] * A[k][l] * A[i][j];
    return A2;
}

double brill_lindquist_psi(double x, double y, double z,
                           int n_bh, const puncture_data_t *bhs)
{
    double psi = 1.0;
    for (int n = 0; n < n_bh; n++) {
        double rx = x - bhs[n].center[0];
        double ry = y - bhs[n].center[1];
        double rz = z - bhs[n].center[2];
        double r  = sqrt(rx*rx + ry*ry + rz*rz);
        if (r < 1.0e-10) r = 1.0e-10;
        psi += bhs[n].mass / (2.0 * r);
    }
    return psi;
}

void set_ccz4_from_psi(grid_t *g, const double *psi_arr,
                        int n_bh, const puncture_data_t *bhs)
{
    /* Convert solved psi + BY A_ij to CCZ4 variables:
     *   chi      = psi^{-4}
     *   h_ij     = delta_ij  (conformally flat)
     *   K        = 0         (maximal slicing initial data)
     *   A_ij^CCZ4 = psi^{-6} * A_ij^phys  (conformal rescaling)
     *   Theta    = 0
     *   Gamma^i  = 0
     *   lapse    = sqrt(chi) = psi^{-2}
     *   shift    = 0
     *   B^i      = 0
     *   E^i      = Coulomb field (if charged), else 0
     *   B^i_mag  = 0
     *
     * Ref: GRChombo BinaryBH.impl.hpp:53-68
     * Ref: B&S Eq. 3.10 (conformal decomposition)
     * Ref: arXiv:1903.01036 (Bozzola & Paschalidis) Eq. (12) (Coulomb field)
     */

    /* Check if any BH has charge */
    int has_charge = 0;
    for (int n = 0; n < n_bh; n++) {
        if (fabs(bhs[n].charge) > 1.0e-15) has_charge = 1;
    }

    for (int k = 0; k < g->Ntotal; k++) {
        for (int j = 0; j < g->Ntotal; j++) {
            for (int i = 0; i < g->Ntotal; i++) {
                int idx = IDX(g, i, j, k);
                double x = COORD(g, i);
                double y = COORD(g, j);
                double z = COORD(g, k);

                double psi = psi_arr[idx];
                double psi4 = psi * psi * psi * psi;
                double chi  = 1.0 / psi4;

                /* Conformal metric: flat */
                g->fields[FIELD_CHI][idx]  = chi;
                g->fields[FIELD_H11][idx]  = 1.0;
                g->fields[FIELD_H12][idx]  = 0.0;
                g->fields[FIELD_H13][idx]  = 0.0;
                g->fields[FIELD_H22][idx]  = 1.0;
                g->fields[FIELD_H23][idx]  = 0.0;
                g->fields[FIELD_H33][idx]  = 1.0;

                /* K=0 (maximal slicing) */
                g->fields[FIELD_K][idx] = 0.0;

                /* A_bar_ij = psi^{-6} * A_tilde_ij (York -> CCZ4 conformal weight)
                 * A_tilde is the Bowen-York extrinsic curvature (conformal weight +2),
                 * A_bar is the CCZ4 convention (conformal weight -4): A_bar = chi * K_phys.
                 * Since K_phys = psi^{-2} * A_tilde, A_bar = psi^{-4} * psi^{-2} * A_tilde.
                 * Ref: B&S Eq. 3.18, GRChombo BinaryBH.impl.hpp:60 */
                double A_tilde[3][3];
                bowen_york_Aij(A_tilde, x, y, z, n_bh, bhs);

                double psi6 = psi4 * psi * psi;
                double psi6_inv = 1.0 / psi6;

                g->fields[FIELD_A11][idx] = psi6_inv * A_tilde[0][0];
                g->fields[FIELD_A12][idx] = psi6_inv * A_tilde[0][1];
                g->fields[FIELD_A13][idx] = psi6_inv * A_tilde[0][2];
                g->fields[FIELD_A22][idx] = psi6_inv * A_tilde[1][1];
                g->fields[FIELD_A23][idx] = psi6_inv * A_tilde[1][2];
                g->fields[FIELD_A33][idx] = psi6_inv * A_tilde[2][2];

                /* Theta, Gamma, shift, B = 0 */
                g->fields[FIELD_THETA][idx]  = 0.0;
                g->fields[FIELD_GAMMA1][idx] = 0.0;
                g->fields[FIELD_GAMMA2][idx] = 0.0;
                g->fields[FIELD_GAMMA3][idx] = 0.0;

                /* Pre-collapsed lapse: alpha = psi^{-2} = sqrt(chi) */
                g->fields[FIELD_LAPSE][idx]  = sqrt(chi);

                g->fields[FIELD_SHIFT1][idx] = 0.0;
                g->fields[FIELD_SHIFT2][idx] = 0.0;
                g->fields[FIELD_SHIFT3][idx] = 0.0;
                g->fields[FIELD_B1][idx]     = 0.0;
                g->fields[FIELD_B2][idx]     = 0.0;
                g->fields[FIELD_B3][idx]     = 0.0;

                /* EM fields: Coulomb E^i for charged BHs, B^i = 0.
                 * Physical E^i = Q / (4 pi r^2) * n^i  (Coulomb)
                 * Conformal E^i = chi^{3/2} * E^i_phys = psi^{-6} * E^i_phys
                 *
                 * In isotropic coordinates, the Coulomb field is:
                 *   E^r_phys = Q / (4 pi r^2)  (coordinate r)
                 * Conformal: E^i_conf = psi^{-6} * (Q / (4 pi r^2)) * n^i
                 *
                 * Ref: arXiv:1903.01036 Eq. (12) */
                double Ex = 0.0, Ey = 0.0, Ez = 0.0;
                if (has_charge) {
                    for (int n = 0; n < n_bh; n++) {
                        if (fabs(bhs[n].charge) < 1.0e-15) continue;
                        double rx = x - bhs[n].center[0];
                        double ry = y - bhs[n].center[1];
                        double rz = z - bhs[n].center[2];
                        double r2 = rx*rx + ry*ry + rz*rz;
                        double r  = sqrt(r2);
                        if (r < 1.0e-10) r = 1.0e-10;
                        double Q = bhs[n].charge;
                        double fac = Q / (4.0 * M_PI * r2 * r);  /* Q/(4pi r^3) */
                        /* E^i_phys = Q/(4 pi r^2) * n^i = Q/(4 pi r^3) * r_i */
                        /* E^i_conf = psi^{-6} * E^i_phys */
                        Ex += psi6_inv * fac * rx;
                        Ey += psi6_inv * fac * ry;
                        Ez += psi6_inv * fac * rz;
                    }
                }
                if (g->n_fields > FIELD_E1) {
                    g->fields[FIELD_E1][idx]  = Ex;
                    g->fields[FIELD_E2][idx]  = Ey;
                    g->fields[FIELD_E3][idx]  = Ez;
                    g->fields[FIELD_BM1][idx] = 0.0;
                    g->fields[FIELD_BM2][idx] = 0.0;
                    g->fields[FIELD_BM3][idx] = 0.0;
                }
            }
        }
    }
}

/* Global flag for --hispid CLI override */
static int hispid_force = 0;

void set_hispid_override(int val) { hispid_force = val; }

void set_ccz4_from_hispid(grid_t *g, const double *psi_arr,
                           double *const *V_arr,
                           int n_bh, const puncture_data_t *bhs)
{
    /* Convert solved psi + non-flat h_ij to CCZ4 variables:
     *   chi      = psi^{-4}
     *   h_ij     = h_ij^QI  (quasi-isotropic Kerr, NOT delta_ij)
     *   K        = 0        (maximal slicing)
     *   A_ij^CCZ4 = psi^{-6} * A_ij^phys
     *   Theta    = 0
     *   Gamma^i  = computed from d_j h^{ij} (NON-ZERO)
     *   lapse    = sqrt(chi)
     *   shift    = 0
     *   B^i      = 0
     *
     * Key difference from set_ccz4_from_psi: h_ij != delta_ij.
     * Gamma^i must be computed from the conformal metric via FD.
     *
     * Ref: arXiv:1410.8607, GRChombo KerrBH.impl.hpp:86-93 */

    int N = g->Ntotal;

    /* First pass: set chi, h_ij, K, A_ij, Theta, lapse, shift, B */
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int idx = IDX(g, i, j, k);
                double x = COORD(g, i);
                double y = COORD(g, j);
                double z = COORD(g, k);

                double psi = psi_arr[idx];
                double psi4 = psi * psi * psi * psi;
                double chi  = 1.0 / psi4;

                g->fields[FIELD_CHI][idx] = chi;

                /* Non-flat conformal metric from superposed QI Kerr */
                double h[3][3];
                hispid_conformal_metric(h, x, y, z, n_bh, bhs);

                g->fields[FIELD_H11][idx] = h[0][0];
                g->fields[FIELD_H12][idx] = h[0][1];
                g->fields[FIELD_H13][idx] = h[0][2];
                g->fields[FIELD_H22][idx] = h[1][1];
                g->fields[FIELD_H23][idx] = h[1][2];
                g->fields[FIELD_H33][idx] = h[2][2];

                /* K = 0 (maximal slicing) */
                g->fields[FIELD_K][idx] = 0.0;

                /* A_bar = psi^{-6} * A_tilde_total.
                 * Both A_kerr and A_by are York weight +2 (A_tilde convention).
                 * Ref: arXiv:1410.8607 Eq. (16), B&S Eq. 3.18 */
                double A_kerr[3][3];
                hispid_extrinsic(A_kerr, x, y, z, n_bh, bhs);
                double A_by[3][3];
                bowen_york_Aij(A_by, x, y, z, n_bh, bhs);

                double psi6 = psi4 * psi * psi;
                double psi6_inv = 1.0 / psi6;

                g->fields[FIELD_A11][idx] = psi6_inv * (A_kerr[0][0] + A_by[0][0]);
                g->fields[FIELD_A12][idx] = psi6_inv * (A_kerr[0][1] + A_by[0][1]);
                g->fields[FIELD_A13][idx] = psi6_inv * (A_kerr[0][2] + A_by[0][2]);
                g->fields[FIELD_A22][idx] = psi6_inv * (A_kerr[1][1] + A_by[1][1]);
                g->fields[FIELD_A23][idx] = psi6_inv * (A_kerr[1][2] + A_by[1][2]);
                g->fields[FIELD_A33][idx] = psi6_inv * (A_kerr[2][2] + A_by[2][2]);

                g->fields[FIELD_THETA][idx] = 0.0;
                g->fields[FIELD_LAPSE][idx] = sqrt(chi);

                g->fields[FIELD_SHIFT1][idx] = 0.0;
                g->fields[FIELD_SHIFT2][idx] = 0.0;
                g->fields[FIELD_SHIFT3][idx] = 0.0;
                g->fields[FIELD_B1][idx]     = 0.0;
                g->fields[FIELD_B2][idx]     = 0.0;
                g->fields[FIELD_B3][idx]     = 0.0;

                /* EM fields: zero for HiSpID path (charge not yet supported) */
                if (g->n_fields > FIELD_E1) {
                    g->fields[FIELD_E1][idx]  = 0.0;
                    g->fields[FIELD_E2][idx]  = 0.0;
                    g->fields[FIELD_E3][idx]  = 0.0;
                    g->fields[FIELD_BM1][idx] = 0.0;
                    g->fields[FIELD_BM2][idx] = 0.0;
                    g->fields[FIELD_BM3][idx] = 0.0;
                }
            }
        }
    }

    /* Second pass: compute Gamma^i = -d_j h^{ij} from the conformal metric.
     * For non-flat h_ij, Gamma^i != 0.
     * Uses the Chris contracted formula: Gamma^i = h^{jk} Gamma^i_{jk}
     * computed from 4th-order FD of h_ij.
     *
     * Ref: GRChombo KerrBH.impl.hpp:90-93 (notes Gamma^i is NON ZERO) */
    int gw = g->ghost;
    double inv_dx = g->inv_dx;
    int strides[3] = { STRIDE_X, STRIDE_Y(g), STRIDE_Z(g) };

    static const int h_field_idx[3][3] = {
        {FIELD_H11, FIELD_H12, FIELD_H13},
        {FIELD_H12, FIELD_H22, FIELD_H23},
        {FIELD_H13, FIELD_H23, FIELD_H33}
    };

    for (int k = gw; k < N - gw; k++) {
        for (int j = gw; j < N - gw; j++) {
            for (int i = gw; i < N - gw; i++) {
                int idx = IDX(g, i, j, k);

                /* Load h_ij */
                double h[3][3];
                h[0][0] = g->fields[FIELD_H11][idx];
                h[0][1] = g->fields[FIELD_H12][idx];
                h[0][2] = g->fields[FIELD_H13][idx];
                h[1][0] = h[0][1];
                h[1][1] = g->fields[FIELD_H22][idx];
                h[1][2] = g->fields[FIELD_H23][idx];
                h[2][0] = h[0][2];
                h[2][1] = h[1][2];
                h[2][2] = g->fields[FIELD_H33][idx];

                double h_UU[3][3];
                compute_inverse_sym(h, h_UU);

                /* First derivatives of h_ij */
                double d1_h[3][3][3];
                for (int dir = 0; dir < 3; dir++) {
                    int s = strides[dir];
                    for (int a = 0; a < 3; a++)
                        for (int b = a; b < 3; b++) {
                            double val = fd_d1(
                                g->fields[h_field_idx[a][b]], idx, s, inv_dx);
                            d1_h[a][b][dir] = val;
                            d1_h[b][a][dir] = val;
                        }
                }

                /* Christoffel symbols and contracted Gamma^i */
                chris_t chris;
                compute_christoffel(d1_h, h_UU, &chris);

                g->fields[FIELD_GAMMA1][idx] = chris.contracted[0];
                g->fields[FIELD_GAMMA2][idx] = chris.contracted[1];
                g->fields[FIELD_GAMMA3][idx] = chris.contracted[2];
            }
        }
    }

    /* Ghost zones: Gamma^i = 0 (Sommerfeld will handle BCs during evolution) */
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) {
                if (i < gw || i >= N - gw ||
                    j < gw || j >= N - gw ||
                    k < gw || k >= N - gw) {
                    int idx = IDX(g, i, j, k);
                    g->fields[FIELD_GAMMA1][idx] = 0.0;
                    g->fields[FIELD_GAMMA2][idx] = 0.0;
                    g->fields[FIELD_GAMMA3][idx] = 0.0;
                }
            }

    (void)V_arr; /* V^i correction applied in future refinement */
}

/* ================================================================
 * Block-aware CCZ4 conversion: solve-on-evolution-mesh path
 * ================================================================ */

void set_ccz4_from_psi_block(block_t *blk, int n_bh, const puncture_data_t *bhs,
                              int n_fields)
{
    /* Convert solver data (fields[SOL_PSI], fields[BG_PSI_BL]) to CCZ4.
     * Read solver slots first at each point, then overwrite with CCZ4.
     * This avoids aliasing since solver slots 0-9 overlap with CCZ4 slots 0-24.
     *
     * Ref: GRChombo BinaryBH.impl.hpp:53-68, B&S Eq. 3.10 */
    grid_t *g = blk->grid;
    int Nt = g->Ntotal;

    /* Check if any BH has charge */
    int has_charge = 0;
    for (int n = 0; n < n_bh; n++)
        if (fabs(bhs[n].charge) > 1.0e-15) has_charge = 1;

    for (int k = 0; k < Nt; k++) {
        for (int j = 0; j < Nt; j++) {
            for (int i = 0; i < Nt; i++) {
                int idx = IDX(g, i, j, k);
                double x = BLOCK_COORD(blk, 0, i);
                double y = BLOCK_COORD(blk, 1, j);
                double z = BLOCK_COORD(blk, 2, k);

                /* Read solver data BEFORE overwriting with CCZ4 */
                double psi = g->fields[BG_PSI_BL_BY][idx]
                           + g->fields[SOL_PSI_BY][idx];
                double psi4 = psi * psi * psi * psi;
                double chi  = 1.0 / psi4;

                /* Conformal rescaling */
                double A_tilde[3][3];
                bowen_york_Aij(A_tilde, x, y, z, n_bh, bhs);
                double psi6 = psi4 * psi * psi;
                double psi6_inv = 1.0 / psi6;

                /* Now write CCZ4 fields */
                g->fields[FIELD_CHI][idx]  = chi;
                g->fields[FIELD_H11][idx]  = 1.0;
                g->fields[FIELD_H12][idx]  = 0.0;
                g->fields[FIELD_H13][idx]  = 0.0;
                g->fields[FIELD_H22][idx]  = 1.0;
                g->fields[FIELD_H23][idx]  = 0.0;
                g->fields[FIELD_H33][idx]  = 1.0;
                g->fields[FIELD_K][idx]    = 0.0;

                g->fields[FIELD_A11][idx] = psi6_inv * A_tilde[0][0];
                g->fields[FIELD_A12][idx] = psi6_inv * A_tilde[0][1];
                g->fields[FIELD_A13][idx] = psi6_inv * A_tilde[0][2];
                g->fields[FIELD_A22][idx] = psi6_inv * A_tilde[1][1];
                g->fields[FIELD_A23][idx] = psi6_inv * A_tilde[1][2];
                g->fields[FIELD_A33][idx] = psi6_inv * A_tilde[2][2];

                g->fields[FIELD_THETA][idx]  = 0.0;
                g->fields[FIELD_GAMMA1][idx] = 0.0;
                g->fields[FIELD_GAMMA2][idx] = 0.0;
                g->fields[FIELD_GAMMA3][idx] = 0.0;
                g->fields[FIELD_LAPSE][idx]  = sqrt(chi);
                g->fields[FIELD_SHIFT1][idx] = 0.0;
                g->fields[FIELD_SHIFT2][idx] = 0.0;
                g->fields[FIELD_SHIFT3][idx] = 0.0;
                g->fields[FIELD_B1][idx]     = 0.0;
                g->fields[FIELD_B2][idx]     = 0.0;
                g->fields[FIELD_B3][idx]     = 0.0;

                /* EM fields: Coulomb E^i for charged BHs, B^i = 0 */
                if (n_fields >= NUM_FIELDS) {
                    double Ex = 0.0, Ey = 0.0, Ez = 0.0;
                    if (has_charge) {
                        for (int n = 0; n < n_bh; n++) {
                            if (fabs(bhs[n].charge) < 1.0e-15) continue;
                            double rx = x - bhs[n].center[0];
                            double ry = y - bhs[n].center[1];
                            double rz = z - bhs[n].center[2];
                            double r2 = rx*rx + ry*ry + rz*rz;
                            double r  = sqrt(r2);
                            if (r < 1.0e-10) r = 1.0e-10;
                            double Q = bhs[n].charge;
                            double fac = Q / (4.0 * M_PI * r2 * r);
                            Ex += psi6_inv * fac * rx;
                            Ey += psi6_inv * fac * ry;
                            Ez += psi6_inv * fac * rz;
                        }
                    }
                    g->fields[FIELD_E1][idx]  = Ex;
                    g->fields[FIELD_E2][idx]  = Ey;
                    g->fields[FIELD_E3][idx]  = Ez;
                    g->fields[FIELD_BM1][idx] = 0.0;
                    g->fields[FIELD_BM2][idx] = 0.0;
                    g->fields[FIELD_BM3][idx] = 0.0;
                }
            }
        }
    }
}

void set_ccz4_from_hispid_block(block_t *blk, int n_bh, const puncture_data_t *bhs,
                                 int n_fields)
{
    /* Convert solver data + non-flat h_ij to CCZ4.
     * Two passes: (1) all fields, (2) Gamma^i from FD of h_ij.
     *
     * Ref: arXiv:1410.8607, GRChombo KerrBH.impl.hpp:86-93 */
    grid_t *g = blk->grid;
    int Nt = g->Ntotal;

    /* Pass 1: set chi, h_ij, K, A_ij, gauge, EM */
    for (int k = 0; k < Nt; k++) {
        for (int j = 0; j < Nt; j++) {
            for (int i = 0; i < Nt; i++) {
                int idx = IDX(g, i, j, k);
                double x = BLOCK_COORD(blk, 0, i);
                double y = BLOCK_COORD(blk, 1, j);
                double z = BLOCK_COORD(blk, 2, k);

                /* Read solver data BEFORE overwriting */
                double psi = g->fields[BG_PSI_BL_BY][idx]
                           + g->fields[SOL_PSI_BY][idx];
                double psi4 = psi * psi * psi * psi;
                double chi  = 1.0 / psi4;

                /* Non-flat conformal metric from superposed QI Kerr */
                double h[3][3];
                hispid_conformal_metric(h, x, y, z, n_bh, bhs);

                /* A_bar = psi^{-6} * (A_kerr + A_by), both York weight +2 */
                double A_kerr[3][3];
                hispid_extrinsic(A_kerr, x, y, z, n_bh, bhs);
                double A_by[3][3];
                bowen_york_Aij(A_by, x, y, z, n_bh, bhs);
                double psi6 = psi4 * psi * psi;
                double psi6_inv = 1.0 / psi6;

                /* Now write CCZ4 fields */
                g->fields[FIELD_CHI][idx] = chi;
                g->fields[FIELD_H11][idx] = h[0][0];
                g->fields[FIELD_H12][idx] = h[0][1];
                g->fields[FIELD_H13][idx] = h[0][2];
                g->fields[FIELD_H22][idx] = h[1][1];
                g->fields[FIELD_H23][idx] = h[1][2];
                g->fields[FIELD_H33][idx] = h[2][2];
                g->fields[FIELD_K][idx] = 0.0;

                g->fields[FIELD_A11][idx] = psi6_inv * (A_kerr[0][0] + A_by[0][0]);
                g->fields[FIELD_A12][idx] = psi6_inv * (A_kerr[0][1] + A_by[0][1]);
                g->fields[FIELD_A13][idx] = psi6_inv * (A_kerr[0][2] + A_by[0][2]);
                g->fields[FIELD_A22][idx] = psi6_inv * (A_kerr[1][1] + A_by[1][1]);
                g->fields[FIELD_A23][idx] = psi6_inv * (A_kerr[1][2] + A_by[1][2]);
                g->fields[FIELD_A33][idx] = psi6_inv * (A_kerr[2][2] + A_by[2][2]);

                g->fields[FIELD_THETA][idx] = 0.0;
                g->fields[FIELD_LAPSE][idx] = sqrt(chi);
                g->fields[FIELD_SHIFT1][idx] = 0.0;
                g->fields[FIELD_SHIFT2][idx] = 0.0;
                g->fields[FIELD_SHIFT3][idx] = 0.0;
                g->fields[FIELD_B1][idx]     = 0.0;
                g->fields[FIELD_B2][idx]     = 0.0;
                g->fields[FIELD_B3][idx]     = 0.0;

                if (n_fields >= NUM_FIELDS) {
                    g->fields[FIELD_E1][idx]  = 0.0;
                    g->fields[FIELD_E2][idx]  = 0.0;
                    g->fields[FIELD_E3][idx]  = 0.0;
                    g->fields[FIELD_BM1][idx] = 0.0;
                    g->fields[FIELD_BM2][idx] = 0.0;
                    g->fields[FIELD_BM3][idx] = 0.0;
                }
            }
        }
    }

    /* Pass 2: compute Gamma^i from FD of h_ij (interior only).
     * Ghost zone h_ij was set analytically in pass 1, so FD stencils
     * at block boundaries are correct. */
    int gw = g->ghost;
    double inv_dx = g->inv_dx;
    int strides[3] = { STRIDE_X, STRIDE_Y(g), STRIDE_Z(g) };

    static const int h_field_idx[3][3] = {
        {FIELD_H11, FIELD_H12, FIELD_H13},
        {FIELD_H12, FIELD_H22, FIELD_H23},
        {FIELD_H13, FIELD_H23, FIELD_H33}
    };

    for (int k = gw; k < Nt - gw; k++) {
        for (int j = gw; j < Nt - gw; j++) {
            for (int i = gw; i < Nt - gw; i++) {
                int idx = IDX(g, i, j, k);

                /* Load h_ij */
                double h[3][3];
                h[0][0] = g->fields[FIELD_H11][idx];
                h[0][1] = g->fields[FIELD_H12][idx];
                h[0][2] = g->fields[FIELD_H13][idx];
                h[1][0] = h[0][1];
                h[1][1] = g->fields[FIELD_H22][idx];
                h[1][2] = g->fields[FIELD_H23][idx];
                h[2][0] = h[0][2];
                h[2][1] = h[1][2];
                h[2][2] = g->fields[FIELD_H33][idx];

                double h_UU[3][3];
                compute_inverse_sym(h, h_UU);

                /* First derivatives of h_ij */
                double d1_h[3][3][3];
                for (int dir = 0; dir < 3; dir++) {
                    int s = strides[dir];
                    for (int a = 0; a < 3; a++)
                        for (int b = a; b < 3; b++) {
                            double val = fd_d1(
                                g->fields[h_field_idx[a][b]], idx, s, inv_dx);
                            d1_h[a][b][dir] = val;
                            d1_h[b][a][dir] = val;
                        }
                }

                /* Christoffel symbols and contracted Gamma^i */
                chris_t chris;
                compute_christoffel(d1_h, h_UU, &chris);

                g->fields[FIELD_GAMMA1][idx] = chris.contracted[0];
                g->fields[FIELD_GAMMA2][idx] = chris.contracted[1];
                g->fields[FIELD_GAMMA3][idx] = chris.contracted[2];
            }
        }
    }

    /* Ghost zones: Gamma^i = 0 (Sommerfeld will handle BCs during evolution) */
    for (int k = 0; k < Nt; k++)
        for (int j = 0; j < Nt; j++)
            for (int i = 0; i < Nt; i++) {
                if (i < gw || i >= Nt - gw ||
                    j < gw || j >= Nt - gw ||
                    k < gw || k >= Nt - gw) {
                    int idx = IDX(g, i, j, k);
                    g->fields[FIELD_GAMMA1][idx] = 0.0;
                    g->fields[FIELD_GAMMA2][idx] = 0.0;
                    g->fields[FIELD_GAMMA3][idx] = 0.0;
                }
            }
}

/* ================================================================
 * set_bowen_york_mesh: initial data on an AMR evolution mesh
 * ================================================================ */
void set_bowen_york_mesh(mesh_t *m, int n_bh, const puncture_data_t *bhs,
                          int n_amr_levels)
{
    /* Check if all momenta and spins are zero — use fast BL path */
    int need_solver = 0;
    int high_spin = 0;
    for (int n = 0; n < n_bh; n++) {
        double S_mag = sqrt(bhs[n].spin[0] * bhs[n].spin[0]
                          + bhs[n].spin[1] * bhs[n].spin[1]
                          + bhs[n].spin[2] * bhs[n].spin[2]);
        double chi_spin = S_mag / (bhs[n].mass * bhs[n].mass);
        if (chi_spin > 0.9) high_spin = 1;

        for (int d = 0; d < 3; d++) {
            if (fabs(bhs[n].momentum[d]) > 1.0e-15 ||
                fabs(bhs[n].spin[d]) > 1.0e-15) {
                need_solver = 1;
            }
        }
    }

    int nf = m->n_fields;

    if (!need_solver) {
        /* Pure Brill-Lindquist: psi is analytic, A_ij = 0 */
        printf("  Bowen-York (mesh): P=0, S=0 — analytic BL path\n");

        /* Refine near punctures if requested */
        if (n_amr_levels > 0)
            refine_mesh_near_punctures(m, n_amr_levels, n_bh, bhs);

        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *blk = m->blocks[bid];
            if (!blk || !blk->is_leaf) continue;
            grid_t *g = blk->grid;
            int Nt = g->Ntotal;

            /* Write BL psi into solver slot, zero correction */
            for (int k = 0; k < Nt; k++)
                for (int j = 0; j < Nt; j++)
                    for (int i = 0; i < Nt; i++) {
                        int idx = IDX(g, i, j, k);
                        double x = BLOCK_COORD(blk, 0, i);
                        double y = BLOCK_COORD(blk, 1, j);
                        double z = BLOCK_COORD(blk, 2, k);
                        g->fields[BG_PSI_BL_BY][idx] =
                            brill_lindquist_psi(x, y, z, n_bh, bhs);
                        g->fields[SOL_PSI_BY][idx] = 0.0;
                    }

            set_ccz4_from_psi_block(blk, n_bh, bhs, nf);
        }
    } else if (high_spin || hispid_force) {
        /* HiSpID: coupled 4-field solver on evolution mesh */
        printf("  Bowen-York (mesh): HiSpID path, %d AMR levels\n", n_amr_levels);
        double residual = relaxation_solve_coupled_amr_mesh(
            m, n_bh, bhs, 1.0e-10, 50000, 1, n_amr_levels);
        printf("  HiSpID (mesh): residual = %.6e\n", residual);

        /* Convert solver data → CCZ4 on each leaf block */
        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *blk = m->blocks[bid];
            if (!blk || !blk->is_leaf) continue;
            set_ccz4_from_hispid_block(blk, n_bh, bhs, nf);
        }
    } else {
        /* Standard BY: 1-field solver on evolution mesh */
        printf("  Bowen-York (mesh): 1-field path, %d AMR levels\n", n_amr_levels);
        double residual = relaxation_solve_amr_mesh(
            m, n_bh, bhs, 1.0e-12, 50000, 1, n_amr_levels);
        printf("  Bowen-York (mesh): residual = %.6e\n", residual);

        /* Convert solver data → CCZ4 on each leaf block */
        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *blk = m->blocks[bid];
            if (!blk || !blk->is_leaf) continue;
            set_ccz4_from_psi_block(blk, n_bh, bhs, nf);
        }
    }

    /* Restrict leaf CCZ4 data into non-leaf parent blocks.
     * After CCZ4 conversion, only leaf blocks have valid data.  Non-leaf
     * parents still contain solver residuals / junk.  During subcycled
     * evolution, ghost_fill_from_coarser() reads non-leaf block data to
     * fill fine-level ghost zones — if that data is garbage, the evolution
     * blows up immediately.  Restricting from fine → coarse ensures ALL
     * blocks (leaf and non-leaf) have valid CCZ4 data.
     *
     * Restrict from finest level down, same order as subcycle post-step.
     * Ref: Berger & Oliger (1984) — restrict at level boundaries. */
    for (int L = m->max_level; L >= 1; L--) {
        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *blk = m->blocks[bid];
            if (!blk || blk->loc.level != L - 1 || blk->is_leaf) continue;

            /* blk is a non-leaf parent at level L-1: restrict children */
            grid_t *pg = blk->grid;
            int ghost_w = pg->ghost;
            int half_N  = pg->N / 2;

            for (int cz = 0; cz < 2; cz++)
            for (int cy = 0; cy < 2; cy++)
            for (int cx = 0; cx < 2; cx++) {
                int octant = cx + (cy << 1) + (cz << 2);
                int cid = blk->child_ids[octant];
                if (cid < 0 || !m->blocks[cid]) continue;

                const block_t *child = m->blocks[cid];
                const grid_t  *cg    = child->grid;

                int p_off_i = cx * half_N;
                int p_off_j = cy * half_N;
                int p_off_k = cz * half_N;

                for (int f = 0; f < nf; f++)
                for (int pk = 0; pk < half_N; pk++)
                for (int pj = 0; pj < half_N; pj++)
                for (int pi = 0; pi < half_N; pi++) {
                    int fi = cg->ghost + 2 * pi;
                    int fj = cg->ghost + 2 * pj;
                    int fk = cg->ghost + 2 * pk;
                    int pii = ghost_w + p_off_i + pi;
                    int pjj = ghost_w + p_off_j + pj;
                    int pkk = ghost_w + p_off_k + pk;
                    pg->fields[f][IDX(pg, pii, pjj, pkk)] =
                        restrict_cell(cg->fields[f], cg, fi, fj, fk);
                }
            }
        }
    }

    /* Ghost exchange for CCZ4 fields across blocks.
     * Use ghost_exchange_all_blocks so non-leaf parents also get valid
     * ghost zones (needed by ghost_fill_from_coarser during subcycling). */
    ghost_exchange_all_blocks(m);

    /* Prolongation phase: fill fine ghost zones from coarser data.
     * ghost_exchange_multilevel handles restrict→coarse_buf→prolongate. */
    ghost_exchange_multilevel(m);
}
