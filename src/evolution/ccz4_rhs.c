/*
 * Lattice — 3D Numerical Relativity
 * Full CCZ4 right-hand-side at a single grid point.
 *
 * Steps:
 *   1. Load fields into locals
 *   2. Compute first derivatives (d1) for all needed fields
 *   3. Compute second derivatives (d2) for chi, h, lapse, shift
 *   4. Compute advection derivatives (upwind with shift)
 *   5. Inverse metric, Christoffel symbols
 *   6. Z vector from Gamma - chris_contracted
 *   7. Ricci tensor with Z terms
 *   8. Covariant derivatives of lapse
 *   9. CCZ4 RHS: chi, h, K, A, Theta, Gamma
 *  10. Moving puncture gauge: lapse, shift, B
 *  11. Kreiss-Oliger dissipation
 *
 * Ref: arXiv:1106.2254 (CCZ4 equations)
 * Ref: GRChombo CCZ4RHS.impl.hpp:60-227
 * Ref: GRChombo CCZ4Geometry.hpp (Ricci with Z)
 * Ref: GRChombo MovingPunctureGauge.hpp (gauge)
 */

#include "ccz4_rhs.h"
#include "maxwell_rhs.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"
#include "../geometry/tensor_utils.h"
#include <math.h>

/* Forward declaration */
extern void add_ko_dissipation(double ** restrict rhs,
                               const double *const * restrict src,
                               const grid_t *g, const sim_params_t *p,
                               int i, int j, int k);

void ccz4_rhs_point(double ** restrict rhs,
                    const double *const * restrict src,
                    const grid_t *g, const sim_params_t *p,
                    int i, int j, int k)
{
    const int idx = IDX(g, i, j, k);
    const int sx  = STRIDE_X;
    const int sy  = STRIDE_Y(g);
    const int sz  = STRIDE_Z(g);
    const double dx = g->dx;
    const int strides[3] = { sx, sy, sz };

    /* ========== 1. Load fields into locals ========== */
    double chi   = src[FIELD_CHI][idx];
    double h[3][3];
    h[0][0] = src[FIELD_H11][idx]; h[0][1] = src[FIELD_H12][idx]; h[0][2] = src[FIELD_H13][idx];
    h[1][0] = h[0][1];             h[1][1] = src[FIELD_H22][idx]; h[1][2] = src[FIELD_H23][idx];
    h[2][0] = h[0][2];             h[2][1] = h[1][2];             h[2][2] = src[FIELD_H33][idx];

    double K     = src[FIELD_K][idx];
    double A[3][3];
    A[0][0] = src[FIELD_A11][idx]; A[0][1] = src[FIELD_A12][idx]; A[0][2] = src[FIELD_A13][idx];
    A[1][0] = A[0][1];             A[1][1] = src[FIELD_A22][idx]; A[1][2] = src[FIELD_A23][idx];
    A[2][0] = A[0][2];             A[2][1] = A[1][2];             A[2][2] = src[FIELD_A33][idx];

    double Theta  = src[FIELD_THETA][idx];
    double Gamma[3] = { src[FIELD_GAMMA1][idx], src[FIELD_GAMMA2][idx], src[FIELD_GAMMA3][idx] };
    double lapse  = src[FIELD_LAPSE][idx];
    double shift[3] = { src[FIELD_SHIFT1][idx], src[FIELD_SHIFT2][idx], src[FIELD_SHIFT3][idx] };
    double B[3]     = { src[FIELD_B1][idx], src[FIELD_B2][idx], src[FIELD_B3][idx] };

    /* ========== 2. First derivatives ========== */
    double d1_chi[3], d1_K[3], d1_Theta[3], d1_lapse[3];
    double d1_h[3][3][3];     /* d1_h[i][j][dir] = d_{dir} h_{ij} */
    double d1_A[3][3][3];     /* d1_A[i][j][dir] = d_{dir} A_{ij} */
    double d1_Gamma[3][3];    /* d1_Gamma[i][dir] = d_{dir} Gamma^i */
    double d1_shift[3][3];    /* d1_shift[i][dir] = d_{dir} shift^i */

    /* Field indices for symmetric tensor components */
    static const int h_idx[3][3] = {
        {FIELD_H11, FIELD_H12, FIELD_H13},
        {FIELD_H12, FIELD_H22, FIELD_H23},
        {FIELD_H13, FIELD_H23, FIELD_H33}
    };
    static const int A_idx[3][3] = {
        {FIELD_A11, FIELD_A12, FIELD_A13},
        {FIELD_A12, FIELD_A22, FIELD_A23},
        {FIELD_A13, FIELD_A23, FIELD_A33}
    };

    /* ========== 2+3. Fused first + diagonal second derivatives ========== */
    /* Fields needing both d1 and d2: chi, lapse, h_ij(6), shift^i(3).
     * Load stencil points once via fd_d1_d2(), saving ~40% memory loads.
     * Fields needing d1 only: K, Theta, A_ij(6), Gamma^i(3). */
    double d2_chi[3][3], d2_lapse[3][3];
    double d2_h[3][3][3][3];    /* d2_h[i][j][dir1][dir2] */
    double d2_shift[3][3][3];   /* d2_shift[i][dir1][dir2] */

    FOR1(dir) {
        int s = strides[dir];

        /* Fused d1+d2 for fields needing both */
        fd_d1_d2(src[FIELD_CHI],   idx, s, dx, &d1_chi[dir],   &d2_chi[dir][dir]);
        fd_d1_d2(src[FIELD_LAPSE], idx, s, dx, &d1_lapse[dir], &d2_lapse[dir][dir]);
        FOR2(a, b) {
            fd_d1_d2(src[h_idx[a][b]], idx, s, dx,
                     &d1_h[a][b][dir], &d2_h[a][b][dir][dir]);
        }
        FOR1(a) {
            fd_d1_d2(src[FIELD_SHIFT1 + a], idx, s, dx,
                     &d1_shift[a][dir], &d2_shift[a][dir][dir]);
        }

        /* d1-only fields */
        d1_K[dir]      = fd_d1(src[FIELD_K],     idx, s, dx);
        d1_Theta[dir]  = fd_d1(src[FIELD_THETA], idx, s, dx);
        FOR2(a, b) {
            d1_A[a][b][dir] = fd_d1(src[A_idx[a][b]], idx, s, dx);
        }
        FOR1(a) {
            d1_Gamma[a][dir] = fd_d1(src[FIELD_GAMMA1 + a], idx, s, dx);
        }
    }

    /* Mixed second derivatives */
    for (int dir1 = 0; dir1 < 3; dir1++) {
        for (int dir2 = 0; dir2 < dir1; dir2++) {
            int s1 = strides[dir1], s2 = strides[dir2];

            d2_chi[dir1][dir2]   = fd_d2_mixed(src[FIELD_CHI],   idx, s1, s2, dx);
            d2_chi[dir2][dir1]   = d2_chi[dir1][dir2];
            d2_lapse[dir1][dir2] = fd_d2_mixed(src[FIELD_LAPSE], idx, s1, s2, dx);
            d2_lapse[dir2][dir1] = d2_lapse[dir1][dir2];

            FOR2(a, b) {
                d2_h[a][b][dir1][dir2] = fd_d2_mixed(src[h_idx[a][b]], idx, s1, s2, dx);
                d2_h[a][b][dir2][dir1] = d2_h[a][b][dir1][dir2];
            }
            FOR1(a) {
                d2_shift[a][dir1][dir2] = fd_d2_mixed(src[FIELD_SHIFT1 + a], idx, s1, s2, dx);
                d2_shift[a][dir2][dir1] = d2_shift[a][dir1][dir2];
            }
        }
    }

    /* ========== 4. Advection derivatives ========== */
    double advec_chi = 0.0, advec_K = 0.0, advec_Theta = 0.0, advec_lapse = 0.0;
    double advec_h[3][3] = {{0}}, advec_A[3][3] = {{0}};
    double advec_Gamma[3] = {0}, advec_shift[3] = {0}, advec_B[3] = {0};

    FOR1(dir) {
        int s = strides[dir];
        double beta = shift[dir];
        /* Hoist sign check: all fields share the same beta per direction.
         * Eliminates 22 branches per direction (66 total) on GPU. */
        double (*fd)(const double *, int, int, double) =
            (beta > 0.0) ? fd_adv_up : fd_adv_down;
        advec_chi    += beta * fd(src[FIELD_CHI],   idx, s, dx);
        advec_K      += beta * fd(src[FIELD_K],     idx, s, dx);
        advec_Theta  += beta * fd(src[FIELD_THETA], idx, s, dx);
        advec_lapse  += beta * fd(src[FIELD_LAPSE], idx, s, dx);

        FOR2(a, b) {
            advec_h[a][b] += beta * fd(src[h_idx[a][b]], idx, s, dx);
            advec_A[a][b] += beta * fd(src[A_idx[a][b]], idx, s, dx);
        }
        FOR1(a) {
            advec_Gamma[a] += beta * fd(src[FIELD_GAMMA1 + a], idx, s, dx);
            advec_shift[a] += beta * fd(src[FIELD_SHIFT1 + a], idx, s, dx);
            advec_B[a]     += beta * fd(src[FIELD_B1 + a],     idx, s, dx);
        }
    }

    /* ========== 5. Inverse metric, Christoffel ========== */
    double h_UU[3][3];
    compute_inverse_sym(h, h_UU);

    chris_t chris;
    compute_christoffel(d1_h, h_UU, &chris);

    /* ========== 6. Z vector ========== */
    /* Z_over_chi[i] = 0.5*(Gamma[i] - chris_contracted[i])
     * Z[i] = chi * Z_over_chi[i]
     * Ref: GRChombo CCZ4RHS.impl.hpp:71-82 */
    double Z_over_chi[3], Z[3];
    FOR1(i) {
        Z_over_chi[i] = 0.5 * (Gamma[i] - chris.contracted[i]);
        Z[i] = chi * Z_over_chi[i];
    }

    /* ========== 7. Ricci tensor with Z terms ========== */
    /* Ref: GRChombo CCZ4Geometry.hpp:56-112 */

    /* Covariant derivative of chi: covdtilde2chi[k][l] = d2_chi[k][l] - Gamma^m_{kl} d1_chi[m] */
    double covdtilde2chi[3][3];
    FOR2(kk, ll) {
        covdtilde2chi[kk][ll] = d2_chi[kk][ll];
        FOR1(m) covdtilde2chi[kk][ll] -= chris.ULL[m][kk][ll] * d1_chi[m];
    }

    /* chris_LLU[i][j][k] = h^{kl} chris_LLL[i][j][l] */
    double chris_LLU[3][3][3] = {{{0}}};
    double boxtildechi = 0.0;
    double dchi_dot_dchi = 0.0;
    FOR2(ii, jj) {
        boxtildechi += covdtilde2chi[ii][jj] * h_UU[ii][jj];
        dchi_dot_dchi += d1_chi[ii] * d1_chi[jj] * h_UU[ii][jj];
        FOR2(kk, ll) chris_LLU[ii][jj][kk] += h_UU[kk][ll] * chris.LLL[ii][jj][ll];
    }

    /* Exploit Ricci symmetry: R_{ij} = R_{ji}. Compute upper triangle
     * (ii <= jj) and mirror, saving 3 of 9 expensive iterations (~120 FMAs).
     * Ref: GRChombo CCZ4Geometry.hpp:78-101, arXiv:1106.2254 Eq. (A1)-(A3) */
    ricci_t ricci;
    for (int ii = 0; ii < 3; ii++) {
        for (int jj = ii; jj < 3; jj++) {
        /* ricci_hat: conformal Ricci using hat-Gamma trick
         * Ref: GRChombo CCZ4Geometry.hpp:78-93 */
        double ricci_hat = 0.0;
        FOR1(kk) {
            ricci_hat += 0.5 * (h[kk][ii] * d1_Gamma[kk][jj]
                              + h[kk][jj] * d1_Gamma[kk][ii]);
            ricci_hat += 0.5 * Gamma[kk] * d1_h[ii][jj][kk];
            FOR1(ll) {
                ricci_hat += -0.5 * h_UU[kk][ll] * d2_h[ii][jj][kk][ll]
                           + (chris.ULL[kk][ll][ii] * chris_LLU[jj][kk][ll]
                            + chris.ULL[kk][ll][jj] * chris_LLU[ii][kk][ll]
                            + chris.ULL[kk][ii][ll] * chris_LLU[kk][jj][ll]);
            }
        }

        /* ricci_chi: chi contribution to Ricci
         * Ref: GRChombo CCZ4Geometry.hpp:96-101 */
        double ricci_chi = 0.5 * (
            (GR_SPACEDIM - 2) * covdtilde2chi[ii][jj]
            + h[ii][jj] * boxtildechi
            - ((GR_SPACEDIM - 2) * d1_chi[ii] * d1_chi[jj]
               + GR_SPACEDIM * h[ii][jj] * dchi_dot_dchi) / (2.0 * chi)
        );

        /* Z terms
         * Ref: GRChombo CCZ4Geometry.hpp:34-45 */
        double z_terms = 0.0;
        FOR1(kk) {
            z_terms += Z_over_chi[kk] * (h[ii][kk] * d1_chi[jj]
                                        + h[jj][kk] * d1_chi[ii]
                                        - h[ii][jj] * d1_chi[kk]);
        }

        ricci.LL[ii][jj] = (ricci_chi + chi * ricci_hat + z_terms) / chi;
        ricci.LL[jj][ii] = ricci.LL[ii][jj];
        }
    }

    /* Ricci scalar: R = chi * h^{ij} R_{ij} */
    ricci.scalar = chi * compute_trace(ricci.LL, h_UU);

    /* ========== 8. Covariant derivatives of lapse ========== */
    /* Ref: GRChombo CCZ4RHS.impl.hpp:87-112 */

    double divshift = 0.0;
    FOR1(ii) divshift += d1_shift[ii][ii];

    double Z_dot_d1lapse = compute_dot_product(Z, d1_lapse);
    double dlapse_dot_dchi = compute_dot_product_metric(d1_lapse, d1_chi, h_UU);

    double covdtilde2lapse[3][3];
    double covd2lapse[3][3];
    FOR2(kk, ll) {
        covdtilde2lapse[kk][ll] = d2_lapse[kk][ll];
        FOR1(m) covdtilde2lapse[kk][ll] -= chris.ULL[m][kk][ll] * d1_lapse[m];

        /* Physical covariant derivative
         * Ref: GRChombo CCZ4RHS.impl.hpp:97-101 */
        covd2lapse[kk][ll] = chi * covdtilde2lapse[kk][ll]
            + 0.5 * (d1_lapse[kk] * d1_chi[ll] + d1_chi[kk] * d1_lapse[ll]
                    - h[kk][ll] * dlapse_dot_dchi);
    }

    /* Trace of covd2lapse
     * Ref: GRChombo CCZ4RHS.impl.hpp:103-112 */
    double tr_covd2lapse = -(GR_SPACEDIM / 2.0) * dlapse_dot_dchi;
    FOR1(ii) {
        tr_covd2lapse -= chi * chris.contracted[ii] * d1_lapse[ii];
        FOR1(jj) {
            tr_covd2lapse += h_UU[ii][jj] * (chi * d2_lapse[ii][jj]
                                             + d1_lapse[ii] * d1_chi[jj]);
        }
    }

    /* A^{ij} = h^{ik} h^{jl} A_{kl} */
    double A_UU[3][3];
    raise_all_2(A, h_UU, A_UU);

    /* A^{ij} A_{ij} (trace of A^2)
     * Ref: GRChombo CCZ4RHS.impl.hpp:117 */
    double tr_A2 = compute_trace(A, A_UU);

    /* ========== 9. CCZ4 RHS ========== */

    /* --- chi --- */
    /* dt(chi) = advec + (2/3)*chi*(alpha*K - divshift)
     * Ref: GRChombo CCZ4RHS.impl.hpp:118-119 */
    double rhs_chi = advec_chi
        + (2.0 / GR_SPACEDIM) * chi * (lapse * K - divshift);

    /* CAHD: Coarse-grid-Adjusted Hamiltonian-constraint Damping.
     * Adds damping to chi evolution proportional to the Hamiltonian constraint.
     * dt(phi) += -C * CFL * dx * H_minus  →  dt(chi) += +4*chi*C*CFL*dx*H
     * On a uniform grid the level scaling factor = 1; for AMR (Stage 4+),
     * scale by dx_level / dx_finest.
     * Ref: arXiv:2404.01137, Eq. (26) */
    if (p->noise.use_cahd) {
        double H = ricci.scalar
                 + ((GR_SPACEDIM - 1.0) / (double)GR_SPACEDIM) * K * K
                 - tr_A2;
        rhs_chi += 4.0 * chi * p->noise.cahd_coeff * p->CFL * dx * H;
    }

    /* --- h_ij --- */
    /* dt(h_ij) = advec - 2*alpha*A_ij - (2/3)*h_ij*divshift + h_{ki}*d_j(shift^k) + h_{kj}*d_i(shift^k)
     * Ref: GRChombo CCZ4RHS.impl.hpp:120-129 */
    double rhs_h[3][3];
    FOR2(ii, jj) {
        rhs_h[ii][jj] = advec_h[ii][jj]
            - 2.0 * lapse * A[ii][jj]
            - (2.0 / GR_SPACEDIM) * h[ii][jj] * divshift;
        FOR1(kk) {
            rhs_h[ii][jj] += h[kk][ii] * d1_shift[kk][jj]
                            + h[kk][jj] * d1_shift[kk][ii];
        }
    }

    /* --- A_ij --- */
    /* Trace-free part of -D_i D_j alpha + chi * alpha * R_ij
     * Ref: GRChombo CCZ4RHS.impl.hpp:131-154 */
    double Adot_TF[3][3];
    FOR2(ii, jj) {
        Adot_TF[ii][jj] = -covd2lapse[ii][jj]
                         + chi * lapse * ricci.LL[ii][jj];
    }
    make_trace_free(Adot_TF, h, h_UU);

    double rhs_A[3][3];
    FOR2(ii, jj) {
        rhs_A[ii][jj] = advec_A[ii][jj] + Adot_TF[ii][jj]
            + A[ii][jj] * (lapse * (K - 2.0 * Theta)
                          - (2.0 / GR_SPACEDIM) * divshift);
        FOR1(kk) {
            rhs_A[ii][jj] += A[kk][ii] * d1_shift[kk][jj]
                            + A[kk][jj] * d1_shift[kk][ii];
            FOR1(ll) {
                rhs_A[ii][jj] -= 2.0 * lapse * h_UU[kk][ll] * A[ii][kk] * A[ll][jj];
            }
        }
    }

    /* --- Theta --- */
    /* Ref: GRChombo CCZ4RHS.impl.hpp:172-180 */
    double kappa1_times_lapse;
    if (p->ccz4.covariant_Z4)
        kappa1_times_lapse = p->ccz4.kappa1;
    else
        kappa1_times_lapse = p->ccz4.kappa1 * lapse;

    double rhs_Theta = advec_Theta
        + 0.5 * lapse * (ricci.scalar - tr_A2
            + ((GR_SPACEDIM - 1.0) / (double)GR_SPACEDIM) * K * K
            - 2.0 * Theta * K)
        - 0.5 * Theta * kappa1_times_lapse
            * ((GR_SPACEDIM + 1) + p->ccz4.kappa2 * (GR_SPACEDIM - 1))
        - Z_dot_d1lapse;

    /* --- K --- */
    /* Ref: GRChombo CCZ4RHS.impl.hpp:183-191 */
    double rhs_K = advec_K
        + lapse * (ricci.scalar + K * (K - 2.0 * Theta))
        - kappa1_times_lapse * GR_SPACEDIM * (1.0 + p->ccz4.kappa2) * Theta
        - tr_covd2lapse;

    /* --- Gamma^i --- */
    /* Ref: GRChombo CCZ4RHS.impl.hpp:193-222 */
    double Gammadot[3];
    FOR1(ii) {
        Gammadot[ii] = (2.0 / GR_SPACEDIM) *
            (divshift * (chris.contracted[ii] + 2.0 * p->ccz4.kappa3 * Z_over_chi[ii])
             - 2.0 * lapse * K * Z_over_chi[ii])
            - 2.0 * kappa1_times_lapse * Z_over_chi[ii];

        FOR1(jj) {
            Gammadot[ii] +=
                2.0 * h_UU[ii][jj] * (lapse * d1_Theta[jj] - Theta * d1_lapse[jj])
                - 2.0 * A_UU[ii][jj] * d1_lapse[jj]
                - lapse * ((2.0 * (GR_SPACEDIM - 1.0) / (double)GR_SPACEDIM)
                           * h_UU[ii][jj] * d1_K[jj]
                         + GR_SPACEDIM * A_UU[ii][jj] * d1_chi[jj] / chi)
                - (chris.contracted[jj] + 2.0 * p->ccz4.kappa3 * Z_over_chi[jj])
                  * d1_shift[ii][jj];

            FOR1(kk) {
                Gammadot[ii] +=
                    2.0 * lapse * chris.ULL[ii][jj][kk] * A_UU[jj][kk]
                    + h_UU[jj][kk] * d2_shift[ii][jj][kk]
                    + ((GR_SPACEDIM - 2.0) / (double)GR_SPACEDIM)
                      * h_UU[ii][jj] * d2_shift[kk][jj][kk];
            }
        }
    }

    double rhs_Gamma[3];
    FOR1(ii) rhs_Gamma[ii] = advec_Gamma[ii] + Gammadot[ii];

    /* ========== 9b. EM stress-energy source terms ========== */
    /* Gate behind em_enabled: zero overhead when EM is off.
     * T^{mu nu}_{EM} enters the CCZ4 RHS via:
     *   Theta   += -8 pi rho_EM
     *   K       += -8 pi alpha (rho_EM + S_EM) / 2    (actually -4 pi alpha (rho + S))
     *   A_ij    += -8 pi alpha chi [S_ij - (1/3) gamma_ij S]^TF
     *   Gamma^i += -16 pi alpha j^i_EM
     *
     * Note: K equation in CCZ4 contains a factor from the full Einstein equations.
     * The standard 3+1 ADM K equation has:  dt(K) += -4 pi alpha (rho + S)
     * The A_ij gets the TF part of -8 pi alpha chi S_ij.
     *
     * Ref: B&S Eq. (2.106)-(2.112) (matter terms in 3+1 evolution)
     * Ref: arXiv:0907.1151 (Alcubierre et al.) Sec. III */
    if (p->em_enabled) {
        double rho_em, S_em_trace;
        double j_em[3], S_em_dd[3][3];
        em_stress_energy(src, g, idx, chi, h_UU, h,
                         &rho_em, j_em, S_em_dd, &S_em_trace);

        /* Theta equation: += -8*pi*rho_EM (Hamiltonian constraint source)
         * Ref: B&S Eq. (2.107): Theta source = -8*pi*rho */
        rhs_Theta += -8.0 * M_PI * rho_em;

        /* K equation: += -4*pi*alpha*(rho + S)
         * For EM: S = rho, so this is -8*pi*alpha*rho
         * Ref: B&S Eq. (2.106) */
        rhs_K += -4.0 * M_PI * lapse * (rho_em + S_em_trace);

        /* A_ij equation: += -8*pi*alpha*chi * [S_ij - (1/3)*gamma_ij*S]^TF
         * gamma_ij = h_ij/chi, so chi*gamma_ij = h_ij.
         * The TF part is taken with respect to h_ij (conformal metric).
         * Ref: B&S Eq. (2.109) */
        double matter_A[3][3];
        FOR2(ii2, jj2) {
            /* chi * S_ij - (1/3) * h_ij * S_em_trace  (since chi*gamma = h) */
            matter_A[ii2][jj2] = chi * S_em_dd[ii2][jj2]
                                 - (1.0 / 3.0) * h[ii2][jj2] * S_em_trace;
        }
        make_trace_free(matter_A, h, h_UU);
        FOR2(ii2, jj2) {
            rhs_A[ii2][jj2] += -8.0 * M_PI * lapse * matter_A[ii2][jj2];
        }

        /* Gamma^i equation: += -16*pi*alpha*j^i_EM
         * Ref: B&S Eq. (2.112) */
        FOR1(ii2) {
            rhs_Gamma[ii2] += -16.0 * M_PI * lapse * j_em[ii2];
        }
    }

    /* ========== 10. Moving puncture gauge ========== */
    /* Ref: GRChombo MovingPunctureGauge.hpp:54-65 */
    /* Fast path: default lapse_power = 1.0 avoids 50-100 cycle pow() call.
     * Ref: GRChombo MovingPunctureGauge.hpp:54-65 */
    double lapse_pow = (p->gauge.lapse_power == 1.0) ? lapse
                       : pow(lapse, p->gauge.lapse_power);
    double rhs_lapse = p->gauge.lapse_advec_coeff * advec_lapse
        - p->gauge.lapse_coeff * lapse_pow * (K - 2.0 * Theta);

    /* SSL: Slow-Start Lapse — temporary Gaussian damping of initial gauge pulse.
     * Drives lapse toward trumpet solution (alpha → W = sqrt(chi)) during
     * early evolution (t < ~100M), then decays to zero.
     * Ref: arXiv:2404.01137, Eq. (27) */
    if (p->noise.use_ssl) {
        double W = sqrt(fmax(chi, 1.0e-10));
        double M = p->noise.ssl_total_mass;
        double h_ssl = p->noise.ssl_h * M;
        double sigma_t = p->noise.ssl_sigma_t * M;
        double t = p->time;
        double ssl_damp = W * h_ssl * exp(-t * t / (2.0 * sigma_t * sigma_t));
        rhs_lapse += -ssl_damp * (lapse - W);
    }

    double rhs_shift[3], rhs_B[3];
    FOR1(ii) {
        rhs_shift[ii] = p->gauge.shift_advec_coeff * advec_shift[ii]
                       + p->gauge.shift_Gamma_coeff * B[ii];

        /* Position-dependent eta: eta(x) = eta_0 / W(x), W = sqrt(chi).
         * Increases damping near punctures (chi→0).
         * Ref: arXiv:1003.0859 (Muller & Brugmann) */
        double eta_eff = p->gauge.eta;
        if (p->gauge.position_dependent_eta) {
            double W = sqrt(fmax(chi, 1.0e-6));
            eta_eff /= fmax(W, 1.0e-6);
        }
        rhs_B[ii] = p->gauge.shift_advec_coeff * advec_B[ii]
                   - p->gauge.shift_advec_coeff * advec_Gamma[ii]
                   + rhs_Gamma[ii] - eta_eff * B[ii];
    }

    /* ========== 11. Store RHS ========== */
    rhs[FIELD_CHI][idx]   = rhs_chi;
    rhs[FIELD_H11][idx]   = rhs_h[0][0];
    rhs[FIELD_H12][idx]   = rhs_h[0][1];
    rhs[FIELD_H13][idx]   = rhs_h[0][2];
    rhs[FIELD_H22][idx]   = rhs_h[1][1];
    rhs[FIELD_H23][idx]   = rhs_h[1][2];
    rhs[FIELD_H33][idx]   = rhs_h[2][2];
    rhs[FIELD_K][idx]     = rhs_K;
    rhs[FIELD_A11][idx]   = rhs_A[0][0];
    rhs[FIELD_A12][idx]   = rhs_A[0][1];
    rhs[FIELD_A13][idx]   = rhs_A[0][2];
    rhs[FIELD_A22][idx]   = rhs_A[1][1];
    rhs[FIELD_A23][idx]   = rhs_A[1][2];
    rhs[FIELD_A33][idx]   = rhs_A[2][2];
    rhs[FIELD_THETA][idx] = rhs_Theta;
    rhs[FIELD_GAMMA1][idx] = rhs_Gamma[0];
    rhs[FIELD_GAMMA2][idx] = rhs_Gamma[1];
    rhs[FIELD_GAMMA3][idx] = rhs_Gamma[2];
    rhs[FIELD_LAPSE][idx]  = rhs_lapse;
    rhs[FIELD_SHIFT1][idx] = rhs_shift[0];
    rhs[FIELD_SHIFT2][idx] = rhs_shift[1];
    rhs[FIELD_SHIFT3][idx] = rhs_shift[2];
    rhs[FIELD_B1][idx]     = rhs_B[0];
    rhs[FIELD_B2][idx]     = rhs_B[1];
    rhs[FIELD_B3][idx]     = rhs_B[2];

    /* ========== 12. Kreiss-Oliger dissipation ========== */
    add_ko_dissipation(rhs, src, g, p, i, j, k);
}
