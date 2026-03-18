/*
 * Lattice — 3D Numerical Relativity
 * Maxwell evolution equations in 3+1 conformal decomposition.
 *
 * Evolves 6 fields: conformal electric E^i and magnetic B^i (stored as
 * FIELD_E1..E3 and FIELD_BM1..BM3).
 *
 * The physical (densitized) EM fields are:
 *   E^i_phys = chi^{-3/2} E^i_conf  (E^i_conf stored on grid)
 *   B^i_phys = chi^{-3/2} B^i_conf
 *
 * We evolve the conformal fields directly. The evolution equations in
 * the 3+1 Valencia formulation are:
 *
 *   dt E^i = beta^j dj E^i - E^j dj beta^i + alpha K E^i
 *          + alpha chi^{-1/2} epsilon^{ijk} dj B_k
 *          + (3/2) (d_t chi / chi) E^i
 *          + kappa_em * alpha * d^i (div E)
 *
 *   dt B^i = beta^j dj B^i - B^j dj beta^i + alpha K B^i
 *          - alpha chi^{-1/2} epsilon^{ijk} dj E_k
 *          + (3/2) (d_t chi / chi) B^i
 *          + kappa_em * alpha * d^i (div B)
 *
 * where epsilon^{ijk} = epsilon_{ijk} / det(gamma) = epsilon_{ijk} * chi^{3/2}
 * (Levi-Civita symbol divided by sqrt of physical metric determinant).
 *
 * The chi time-derivative terms cancel with the conformal rescaling,
 * leaving the evolution of the conformal fields as written above without
 * explicit dt(chi) terms when using the conformal curl formulation.
 *
 * Ref: arXiv:0907.1151 (Alcubierre et al. 2010), Eqs. (23)-(24)
 * Ref: arXiv:1903.01036 (Bozzola & Paschalidis), Eq. (6)-(8)
 */

#include "maxwell_rhs.h"
#include "ccz4_rhs.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"
#include "../geometry/tensor_utils.h"
#include <math.h>

/* Levi-Civita symbol epsilon_{ijk} (fully antisymmetric, values +1/-1/0) */
LATTICE_DEVICE static const int levi_civita[3][3][3] = {
    {{ 0, 0, 0}, { 0, 0, 1}, { 0,-1, 0}},
    {{ 0, 0,-1}, { 0, 0, 0}, { 1, 0, 0}},
    {{ 0, 1, 0}, {-1, 0, 0}, { 0, 0, 0}}
};

/*
 * EM stress-energy tensor components at a single point.
 *
 * Physical E^i = chi^{-3/2} E^i_conf, but E_i = gamma_{ij} E^j.
 * gamma_{ij} = h_{ij} / chi, so E_i = h_{ij} E^j / chi.
 * E^2 = gamma_{ij} E^i E^j = h_{ij} E^i E^j / chi.
 *
 * For conformal fields:
 *   rho_EM   = (E^2 + B^2) / 2
 *   j^i_EM   = epsilon^{ijk} E_j B_k / sqrt(gamma)
 *            = chi^{3/2} epsilon_{ijk} E_j B_k  (using physical E_j, B_j)
 *   S_{ij}   = E_i E_j + B_i B_j - (1/2) gamma_{ij} (E^2 + B^2)
 *   S = gamma^{ij} S_{ij} = (E^2 + B^2) / 2 = rho_EM  (for EM: S = rho)
 *
 * All computed in terms of conformal fields and metric.
 *
 * Ref: arXiv:0907.1151 Eq. (5)
 */
LATTICE_DEVICE
void em_stress_energy(const double *const * restrict src, const grid_t *g,
                      int idx,
                      double chi, const double h_UU[3][3],
                      const double h[3][3],
                      double *rho_em, double j_em[3],
                      double S_em_dd[3][3], double *S_em_trace)
{
    (void)g;
    (void)h_UU;  /* stress-energy uses h (lower) only; h_UU kept in API for future use */

    /* Load conformal EM fields */
    double Eu[3] = { src[FIELD_E1][idx], src[FIELD_E2][idx], src[FIELD_E3][idx] };
    double Bu[3] = { src[FIELD_BM1][idx], src[FIELD_BM2][idx], src[FIELD_BM3][idx] };

    /* Lower indices using physical metric gamma_ij = h_ij / chi.
     * E_i = gamma_{ij} E^j = (h_{ij} / chi) E^j_conf * chi^{-3/2}
     *
     * Since we store conformal E^i and the metric coupling uses physical
     * quantities, we need to work with the physical fields.
     * Physical E^i_phys = chi^{-3/2} * E^i_conf.
     *
     * For the stress-energy, what matters is:
     *   E^2 = gamma_{ij} E^i E^j (physical)
     *       = (h_{ij}/chi) * (chi^{-3/2} Eu_i) * (chi^{-3/2} Eu_j)
     *       = chi^{-4} * h_{ij} * Eu_i * Eu_j
     *
     * We compute everything in terms of conformal quantities and chi.
     */
    double chi_inv = 1.0 / fmax(chi, 1.0e-4);

    /* h_{ij} E^i E^j (conformal contraction) */
    double hEE = 0.0, hBB = 0.0;
    FOR2(i, j) {
        hEE += h[i][j] * Eu[i] * Eu[j];
        hBB += h[i][j] * Bu[i] * Bu[j];
    }

    /* Physical E^2 = chi^{-4} * hEE, B^2 = chi^{-4} * hBB */
    double chi4_inv = chi_inv * chi_inv * chi_inv * chi_inv;
    double E2_phys = chi4_inv * hEE;
    double B2_phys = chi4_inv * hBB;

    /* rho_EM = (E^2 + B^2) / 2 */
    *rho_em = 0.5 * (E2_phys + B2_phys);

    /* S_EM = rho_EM (for EM radiation, trace of spatial stress = energy density) */
    *S_em_trace = *rho_em;

    /* Physical E_i = gamma_{ij} E^j_phys = (h_{ij}/chi) * chi^{-3/2} * E^j_conf
     *              = chi^{-5/2} * h_{ij} * E^j_conf */
    double chi52_inv = chi_inv * chi_inv * sqrt(chi_inv);
    double E_d[3] = {0}, B_d[3] = {0};
    FOR1(i) FOR1(j) {
        E_d[i] += chi52_inv * h[i][j] * Eu[j];
        B_d[i] += chi52_inv * h[i][j] * Bu[j];
    }

    /* j^i_EM = epsilon^{ijk} E_j B_k (physical Poynting vector, contravariant)
     * epsilon^{ijk} = epsilon_{ijk} / sqrt(gamma)
     *               = epsilon_{ijk} * chi^{3/2}  (since det(gamma) = chi^{-3})
     * Ref: arXiv:0907.1151 Eq. (5) */
    double chi32 = chi * sqrt(fmax(chi, 1.0e-4));
    FOR1(i) {
        j_em[i] = 0.0;
        FOR2(j, k) {
            j_em[i] += chi32 * levi_civita[i][j][k] * E_d[j] * B_d[k];
        }
    }

    /* S_{ij} = E_i E_j + B_i B_j - (1/2) gamma_{ij} (E^2 + B^2)
     * where gamma_{ij} = h_{ij} / chi
     * Ref: arXiv:0907.1151 Eq. (5) */
    double EB2_half = 0.5 * (E2_phys + B2_phys);
    FOR2(i, j) {
        S_em_dd[i][j] = E_d[i] * E_d[j] + B_d[i] * B_d[j]
                        - (h[i][j] * chi_inv) * EB2_half;
    }
}

/*
 * Maxwell RHS at a single grid point.
 *
 * Evolution equations for conformal E^i and B^i:
 *
 *   dt E^i = beta^j dj E^i  (advection)
 *          - E^j dj beta^i   (shift coupling)
 *          + alpha K E^i     (trace of extrinsic curvature)
 *          + alpha * curl(B)^i  (Maxwell curl in curved space)
 *          + kappa_em * alpha * grad(div E)^i  (constraint damping)
 *
 *   dt B^i = beta^j dj B^i  (advection)
 *          - B^j dj beta^i   (shift coupling)
 *          + alpha K B^i     (trace of extrinsic curvature)
 *          - alpha * curl(E)^i  (Maxwell curl in curved space)
 *          + kappa_em * alpha * grad(div B)^i  (constraint damping)
 *
 * The conformal curl uses:
 *   curl(F)^i = h^{ij} h^{kl} epsilon_{jkm} d_l F^m / chi
 *
 * But more directly, for conformal fields evolving on conformal metric:
 *   curl(F)^i = epsilon^{ijk} d_j F_k / sqrt(gamma)
 *
 * We implement the curl in a simpler form using flat-space partial
 * derivatives of the conformal EM fields, which is correct for the
 * conformally flat initial data case and a good approximation otherwise.
 *
 * Ref: arXiv:0907.1151 Eqs. (23)-(24)
 * Ref: arXiv:1903.01036 Eq. (6)-(8)
 */
LATTICE_DEVICE
void maxwell_rhs_point(double ** restrict rhs,
                       const double *const * restrict src,
                       const grid_t *g, const sim_params_t *p,
                       int i, int j, int k)
{
    const int idx = IDX(g, i, j, k);
    const int sx  = STRIDE_X;
    const int sy  = STRIDE_Y(g);
    const int sz  = STRIDE_Z(g);
    const double inv_dx = g->inv_dx;
    const int strides[3] = { sx, sy, sz };

    /* Load metric and gauge fields */
    double chi   = src[FIELD_CHI][idx];
    double K     = src[FIELD_K][idx];
    double lapse = src[FIELD_LAPSE][idx];
    double shift[3] = {
        src[FIELD_SHIFT1][idx],
        src[FIELD_SHIFT2][idx],
        src[FIELD_SHIFT3][idx]
    };

    /* Load conformal metric and compute inverse */
    double h[3][3];
    h[0][0] = src[FIELD_H11][idx]; h[0][1] = src[FIELD_H12][idx]; h[0][2] = src[FIELD_H13][idx];
    h[1][0] = h[0][1];             h[1][1] = src[FIELD_H22][idx]; h[1][2] = src[FIELD_H23][idx];
    h[2][0] = h[0][2];             h[2][1] = h[1][2];             h[2][2] = src[FIELD_H33][idx];

    double h_UU[3][3];
    compute_inverse_sym(h, h_UU);

    /* Load EM fields */
    double Eu[3] = { src[FIELD_E1][idx], src[FIELD_E2][idx], src[FIELD_E3][idx] };
    double Bu[3] = { src[FIELD_BM1][idx], src[FIELD_BM2][idx], src[FIELD_BM3][idx] };

    /* EM first derivatives: d1_E[comp][dir], d1_B[comp][dir] */
    double d1_E[3][3], d1_B[3][3];
    double d1_shift[3][3];
    FOR1(dir) {
        int s = strides[dir];
        FOR1(a) {
            d1_E[a][dir] = fd_d1(src[FIELD_E1 + a], idx, s, inv_dx);
            d1_B[a][dir] = fd_d1(src[FIELD_BM1 + a], idx, s, inv_dx);
            d1_shift[a][dir] = fd_d1(src[FIELD_SHIFT1 + a], idx, s, inv_dx);
        }
    }

    /* Advection terms: beta^j d_j E^i, beta^j d_j B^i
     * Using upwind derivatives for numerical stability */
    double advec_E[3] = {0}, advec_B[3] = {0};
    FOR1(dir) {
        int s = strides[dir];
        double beta = shift[dir];
        double (*fd)(const double *, int, int, double) =
            (beta > 0.0) ? fd_adv_up : fd_adv_down;
        FOR1(a) {
            advec_E[a] += beta * fd(src[FIELD_E1 + a], idx, s, inv_dx);
            advec_B[a] += beta * fd(src[FIELD_BM1 + a], idx, s, inv_dx);
        }
    }

    /* Lower EM indices for curl computation.
     * F_i = h_{ij} F^j / chi (conformal lowering with physical metric).
     * For the curl term, we need d_j F_k where F_k is the covariant component.
     *
     * Actually, for the 3+1 Maxwell equations in Valencia formulation,
     * the curl is computed from partial derivatives of the contravariant
     * components combined with the metric. In conformal decomposition:
     *
     *   curl(B)^i = chi * h^{ia} h^{bc} epsilon_{abd} d_c B^d
     *
     * where the chi factor comes from the conformal rescaling.
     * We compute this explicitly below. */

    /* Compute conformal curl: curl_B^i = chi * h^{ia} h^{bc} eps_{abd} d_c B^d
     * and similarly curl_E^i.
     *
     * This gives the curl of the physical field components in terms of
     * derivatives of the conformal field components.
     *
     * Ref: arXiv:0907.1151 Eq. (23)-(24), adapted for conformal variables */
    /* Unrolled Levi-Civita curl: eps_{abd} nonzero for 6 of 27 (a,b,d) triplets.
     * Eliminates 63 of 81 branch+lookup iterations per component.
     * Nonzero: (0,1,2)=+1, (1,2,0)=+1, (2,0,1)=+1,
     *          (0,2,1)=-1, (2,1,0)=-1, (1,0,2)=-1. */
    double curl_B[3] = {0}, curl_E[3] = {0};
    FOR1(i_idx) {
        FOR1(c) {
            /* eps_{012}=+1: a=0,b=1,d=2 */
            curl_B[i_idx] += chi * h_UU[i_idx][0] * h_UU[1][c] * d1_B[2][c];
            curl_E[i_idx] += chi * h_UU[i_idx][0] * h_UU[1][c] * d1_E[2][c];
            /* eps_{120}=+1: a=1,b=2,d=0 */
            curl_B[i_idx] += chi * h_UU[i_idx][1] * h_UU[2][c] * d1_B[0][c];
            curl_E[i_idx] += chi * h_UU[i_idx][1] * h_UU[2][c] * d1_E[0][c];
            /* eps_{201}=+1: a=2,b=0,d=1 */
            curl_B[i_idx] += chi * h_UU[i_idx][2] * h_UU[0][c] * d1_B[1][c];
            curl_E[i_idx] += chi * h_UU[i_idx][2] * h_UU[0][c] * d1_E[1][c];
            /* eps_{021}=-1: a=0,b=2,d=1 */
            curl_B[i_idx] -= chi * h_UU[i_idx][0] * h_UU[2][c] * d1_B[1][c];
            curl_E[i_idx] -= chi * h_UU[i_idx][0] * h_UU[2][c] * d1_E[1][c];
            /* eps_{210}=-1: a=2,b=1,d=0 */
            curl_B[i_idx] -= chi * h_UU[i_idx][2] * h_UU[1][c] * d1_B[0][c];
            curl_E[i_idx] -= chi * h_UU[i_idx][2] * h_UU[1][c] * d1_E[0][c];
            /* eps_{102}=-1: a=1,b=0,d=2 */
            curl_B[i_idx] -= chi * h_UU[i_idx][1] * h_UU[0][c] * d1_B[2][c];
            curl_E[i_idx] -= chi * h_UU[i_idx][1] * h_UU[0][c] * d1_E[2][c];
        }
    }

    /* Constraint damping: div(E) and div(B) and their gradients.
     *
     * div(E) = d_i E^i (coordinate divergence of conformal E).
     * For constraint damping, we add kappa_em * alpha * d^i(divE) to dt E^i.
     *
     * div(E) = sum_i d1_E[i][i]  (partial derivative, coordinate divergence)
     * grad(divE)^i = h^{ij} d_j(divE)
     *
     * For computational efficiency, we compute divE from centered FD,
     * then compute its gradient also from centered FD. This requires
     * computing divE at neighboring points. Instead, we use the simpler
     * approach of computing d^2 E^i / dx_i dx_j terms directly. */
    double divE = 0.0, divB = 0.0;
    FOR1(a) {
        divE += d1_E[a][a];
        divB += d1_B[a][a];
    }

    /* Second derivatives for constraint damping gradient.
     * grad(divE)^i approx = h^{ij} * d_j(divE).
     * We approximate d_j(divE) = sum_a d2_E[a][a][j].
     * This requires second derivatives of E along direction j. */
    double grad_divE[3] = {0}, grad_divB[3] = {0};
    if (fabs(p->kappa_em) > 1.0e-15) {
        FOR1(dir) {
            int s = strides[dir];
            double d_divE = 0.0, d_divB = 0.0;
            FOR1(a) {
                /* d_dir(d_a E^a) via mixed second derivative if dir != a,
                 * or second derivative if dir == a */
                int sa = strides[a];
                if (dir == a) {
                    d_divE += fd_d2(src[FIELD_E1 + a], idx, s, inv_dx);
                    d_divB += fd_d2(src[FIELD_BM1 + a], idx, s, inv_dx);
                } else {
                    d_divE += fd_d2_mixed(src[FIELD_E1 + a], idx, sa, s, inv_dx);
                    d_divB += fd_d2_mixed(src[FIELD_BM1 + a], idx, sa, s, inv_dx);
                }
            }
            FOR1(m) {
                grad_divE[m] += h_UU[m][dir] * d_divE;
                grad_divB[m] += h_UU[m][dir] * d_divB;
            }
        }
    }

    /* Assemble Maxwell RHS */
    double rhs_E[3], rhs_B[3];
    FOR1(ii) {
        /* Advection */
        rhs_E[ii] = advec_E[ii];
        rhs_B[ii] = advec_B[ii];

        /* Shift coupling: -E^j d_j beta^i */
        FOR1(jj) {
            rhs_E[ii] -= Eu[jj] * d1_shift[ii][jj];
            rhs_B[ii] -= Bu[jj] * d1_shift[ii][jj];
        }

        /* Trace K coupling: +alpha * K * E^i */
        rhs_E[ii] += lapse * K * Eu[ii];
        rhs_B[ii] += lapse * K * Bu[ii];

        /* Maxwell curl terms:
         *   dt E^i += +alpha * curl(B)^i
         *   dt B^i += -alpha * curl(E)^i
         * Ref: arXiv:0907.1151 Eq. (23)-(24) */
        rhs_E[ii] += lapse * curl_B[ii];
        rhs_B[ii] -= lapse * curl_E[ii];

        /* Constraint damping: kappa_em * alpha * d^i(div F) */
        rhs_E[ii] += p->kappa_em * lapse * grad_divE[ii];
        rhs_B[ii] += p->kappa_em * lapse * grad_divB[ii];
    }

    /* Store RHS */
    rhs[FIELD_E1][idx]  = rhs_E[0];
    rhs[FIELD_E2][idx]  = rhs_E[1];
    rhs[FIELD_E3][idx]  = rhs_E[2];
    rhs[FIELD_BM1][idx] = rhs_B[0];
    rhs[FIELD_BM2][idx] = rhs_B[1];
    rhs[FIELD_BM3][idx] = rhs_B[2];
}

/*
 * Combined CCZ4 + Maxwell RHS at a single grid point.
 * Calls ccz4_rhs_point (which includes EM source terms via p->em_enabled),
 * then maxwell_rhs_point for the 6 EM field equations.
 */
LATTICE_DEVICE
void ccz4_maxwell_rhs_point(double ** restrict rhs,
                             const double *const * restrict src,
                             const grid_t *g, const sim_params_t *p,
                             int i, int j, int k)
{
    /* CCZ4 evolution (includes EM stress-energy coupling if p->em_enabled) */
    ccz4_rhs_point(rhs, src, g, p, i, j, k);

    /* Maxwell evolution equations */
    maxwell_rhs_point(rhs, src, g, p, i, j, k);
}
