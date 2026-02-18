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
#include "../core/fields.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void bowen_york_Aij(double A_phys[3][3], double x, double y, double z,
                    int n_bh, const puncture_data_t *bhs)
{
    memset(A_phys, 0, sizeof(double) * 9);

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
                A_phys[i][j] += A_P;
                if (j != i) A_phys[j][i] += A_P;
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
                A_phys[i][j] += A_S;
                if (j != i) A_phys[j][i] += A_S;
            }
        }
    }
}

double bowen_york_A2(const double A_phys[3][3])
{
    /* Flat metric contraction: A^ij A_ij = sum_{i,j} A_ij^2 */
    double A2 = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            A2 += A_phys[i][j] * A_phys[i][j];
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
     *
     * Ref: GRChombo BinaryBH.impl.hpp:53-68
     * Ref: B&S Eq. 3.10 (conformal decomposition)
     */
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

                /* A_ij^CCZ4 = psi^{-6} * A_ij^phys = chi^{3/2} * A_ij^phys
                 * Ref: B&S Eq. 3.18, GRChombo BinaryBH.impl.hpp:60 */
                double A_phys[3][3];
                bowen_york_Aij(A_phys, x, y, z, n_bh, bhs);

                double psi6 = psi4 * psi * psi;
                double psi6_inv = 1.0 / psi6;

                g->fields[FIELD_A11][idx] = psi6_inv * A_phys[0][0];
                g->fields[FIELD_A12][idx] = psi6_inv * A_phys[0][1];
                g->fields[FIELD_A13][idx] = psi6_inv * A_phys[0][2];
                g->fields[FIELD_A22][idx] = psi6_inv * A_phys[1][1];
                g->fields[FIELD_A23][idx] = psi6_inv * A_phys[1][2];
                g->fields[FIELD_A33][idx] = psi6_inv * A_phys[2][2];

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
            }
        }
    }
}

void set_bowen_york(grid_t *g, int n_bh, const puncture_data_t *bhs)
{
    /* Check if all momenta and spins are zero — use fast BL path */
    int need_solver = 0;
    for (int n = 0; n < n_bh; n++) {
        for (int d = 0; d < 3; d++) {
            if (fabs(bhs[n].momentum[d]) > 1.0e-15 ||
                fabs(bhs[n].spin[d]) > 1.0e-15) {
                need_solver = 1;
                break;
            }
        }
        if (need_solver) break;
    }

    if (!need_solver) {
        /* Pure Brill-Lindquist: psi is analytic, A_ij = 0 */
        printf("  Bowen-York: P=0, S=0 — using analytic BL path\n");
        double *psi_arr = (double *)calloc(g->npoints, sizeof(double));
        for (int k = 0; k < g->Ntotal; k++) {
            for (int j = 0; j < g->Ntotal; j++) {
                for (int i = 0; i < g->Ntotal; i++) {
                    int idx = IDX(g, i, j, k);
                    double x = COORD(g, i);
                    double y = COORD(g, j);
                    double z = COORD(g, k);
                    psi_arr[idx] = brill_lindquist_psi(x, y, z, n_bh, bhs);
                }
            }
        }
        set_ccz4_from_psi(g, psi_arr, n_bh, bhs);
        free(psi_arr);
    } else {
        /* Non-trivial momentum/spin: solve Hamiltonian constraint */
        printf("  Bowen-York: solving Hamiltonian constraint...\n");
        double residual = relaxation_solve(g, n_bh, bhs,
                                           1.0e-12, 50000, 1);
        printf("  Bowen-York: solver converged, ||v||_L2 = %.6e\n", residual);
    }
}
