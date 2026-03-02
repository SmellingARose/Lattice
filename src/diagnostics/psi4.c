/*
 * Lattice — 3D Numerical Relativity
 * Psi4 gravitational wave extraction.
 *
 * Newman-Penrose scalar Psi4 from the 3+1 decomposition of the Weyl tensor.
 * Electric Weyl: E_ij = R_ij + K K_ij - K_ik K^k_j  (trace-free)
 * Magnetic Weyl: B_ij = epsilon_{(i}^{kl} D_k K_{j)l}  (trace-free)
 * Projection:    Re(Psi4) = (E_vv - E_ww)/2 + B_vw
 *                Im(Psi4) = (B_vv - B_ww)/2 - E_vw
 *
 * Ref: B&S "Numerical Relativity" §8.3
 * Ref: GRChombo Source/CCZ4/Weyl4.impl.hpp
 * Ref: arXiv:1106.2254 (CCZ4 formulation)
 */

#include "psi4.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"
#include "../geometry/tensor_utils.h"
#include "../amr/mesh.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * Field index tables (file-scope, same as constraints.c)
 * ================================================================ */

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

/* Levi-Civita antisymmetric symbol: e_{ijk} = (i-j)(j-k)(k-i)/2 */
static inline int levi_civita(int i, int j, int k)
{
    return (i - j) * (j - k) * (k - i) / 2;
}

/* Mode index: l ranges from 2 to l_max, m from -l to l.
 * idx = l^2 + l + m - 4 */
static inline int mode_index(int l, int m)
{
    return l * l + l + m - 4;
}

/* ================================================================
 * 1. Gauss-Legendre quadrature nodes and weights on [-1, 1]
 * ================================================================ */

static void gauss_legendre_nodes(int n, double *x, double *w)
{
    for (int i = 0; i < n; i++) {
        /* Initial guess: Chebyshev-type */
        double xi = cos(M_PI * (4 * i + 3) / (4.0 * n + 2));

        /* Newton iteration on Legendre polynomial P_n(x) */
        for (int iter = 0; iter < 100; iter++) {
            double p0 = 1.0, p1 = xi;
            for (int k = 2; k <= n; k++) {
                double p2 = ((2 * k - 1) * xi * p1 - (k - 1) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            /* P_n(x) = p1, P'_n(x) = n*(x*P_n - P_{n-1}) / (x^2 - 1) */
            double dp = n * (xi * p1 - p0) / (xi * xi - 1.0);
            double dx_val = p1 / dp;
            xi -= dx_val;
            if (fabs(dx_val) < 1e-15) break;
        }
        x[i] = xi;
        /* Weight: w_i = 2 / ((1-x_i^2) * [P'_n(x_i)]^2) */
        double p0 = 1.0, p1 = xi;
        for (int k = 2; k <= n; k++) {
            double p2 = ((2 * k - 1) * xi * p1 - (k - 1) * p0) / k;
            p0 = p1;
            p1 = p2;
        }
        double dp = n * (xi * p1 - p0) / (xi * xi - 1.0);
        w[i] = 2.0 / ((1.0 - xi * xi) * dp * dp);
    }
}

/* ================================================================
 * 2. Spin-weighted spherical harmonics _{-2}Y_{lm}
 *
 * _{s}Y_{lm}(theta, phi) = (-1)^s sqrt((2l+1)/(4pi))
 *                           * d^l_{m,-s}(theta) * e^{im*phi}
 *
 * For s = -2: _{-2}Y_{lm} = sqrt((2l+1)/(4pi)) * d^l_{m,2}(theta) * e^{im*phi}
 *
 * Wigner d-matrix computed via the sum formula.
 * Ref: Goldberg et al., J. Math. Phys. 8, 2155 (1967)
 * ================================================================ */

static double factorial_table[25];
static int factorial_initialized = 0;

static void init_factorial(void)
{
    if (factorial_initialized) return;
    factorial_table[0] = 1.0;
    for (int i = 1; i < 25; i++)
        factorial_table[i] = factorial_table[i - 1] * i;
    factorial_initialized = 1;
}

static inline double factorial(int n)
{
    if (n < 0) return 0.0;
    if (n < 25) return factorial_table[n];
    double f = factorial_table[24];
    for (int i = 25; i <= n; i++) f *= i;
    return f;
}

/* Wigner small d-matrix: d^l_{m,mp}(theta) */
static double wigner_d(int l, int m, int mp, double theta)
{
    double c = cos(theta * 0.5);
    double s = sin(theta * 0.5);
    double sum = 0.0;

    /* Sum over k where all factorial arguments are non-negative:
     * k >= 0, k >= m - mp, k <= l + m, k <= l - mp */
    int kmin = (m - mp > 0) ? m - mp : 0;
    int kmax = l + m;
    if (l - mp < kmax) kmax = l - mp;

    for (int k = kmin; k <= kmax; k++) {
        int a = l + m - k;      /* >= 0 by kmax */
        int b = l - mp - k;     /* >= 0 by kmax */
        int c_fac = k - m + mp; /* >= 0 by kmin */

        double sign = ((k & 1) ? -1.0 : 1.0);
        double coeff = sign / (factorial(a) * factorial(k) *
                                factorial(c_fac) * factorial(b));

        int power_c = 2 * l - 2 * k + m - mp;
        int power_s = 2 * k - m + mp;

        sum += coeff * pow(c, power_c) * pow(s, power_s);
    }

    double norm = sqrt(factorial(l + m) * factorial(l - m) *
                       factorial(l + mp) * factorial(l - mp));
    return norm * sum;
}

/* Compute Re and Im of _{-2}Y_{lm}(theta, phi).
 * For s = -2: _{-2}Y_{lm} = sqrt((2l+1)/(4pi)) * d^l_{m,2}(theta) * e^{im*phi} */
static void spin_weighted_Ylm(int l, int m, double theta, double phi,
                               double *re, double *im)
{
    double d = wigner_d(l, m, 2, theta);
    double norm = sqrt((2 * l + 1) / (4.0 * M_PI));
    double val = norm * d;
    *re = val * cos(m * phi);
    *im = val * sin(m * phi);
}

/* ================================================================
 * 3. Workspace allocation / free
 * ================================================================ */

psi4_workspace_t *psi4_alloc(int n_theta, int n_phi, int l_max,
                              double radius, const double center[3])
{
    init_factorial();

    psi4_workspace_t *ws = calloc(1, sizeof(psi4_workspace_t));
    ws->n_theta = n_theta;
    ws->n_phi   = n_phi;
    ws->l_max   = l_max;
    ws->radius  = radius;
    ws->center[0] = center[0];
    ws->center[1] = center[1];
    ws->center[2] = center[2];

    /* Gauss-Legendre nodes on [-1, 1] → theta = arccos(x) */
    double *gl_x = malloc((size_t)n_theta * sizeof(double));
    ws->theta      = malloc((size_t)n_theta * sizeof(double));
    ws->gl_weights = malloc((size_t)n_theta * sizeof(double));
    gauss_legendre_nodes(n_theta, gl_x, ws->gl_weights);
    for (int i = 0; i < n_theta; i++)
        ws->theta[i] = acos(gl_x[i]);
    free(gl_x);

    /* Sphere data */
    int np = n_theta * n_phi;
    ws->re_psi4 = calloc((size_t)np, sizeof(double));
    ws->im_psi4 = calloc((size_t)np, sizeof(double));

    /* Mode coefficients: l from 2 to l_max, m from -l to l
     * n_modes = (l_max+1)^2 - 4 */
    ws->n_modes = (l_max + 1) * (l_max + 1) - 4;
    ws->mode_re = calloc((size_t)ws->n_modes, sizeof(double));
    ws->mode_im = calloc((size_t)ws->n_modes, sizeof(double));

    return ws;
}

void psi4_free(psi4_workspace_t *ws)
{
    if (!ws) return;
    free(ws->theta);
    free(ws->gl_weights);
    free(ws->re_psi4);
    free(ws->im_psi4);
    free(ws->mode_re);
    free(ws->mode_im);
    free(ws);
}

/* ================================================================
 * 4. Psi4 at a single grid point (core physics kernel)
 *
 * Same derivative pattern as compute_hamiltonian_at() in constraints.c,
 * extended with A_ij and K first derivatives for the magnetic Weyl.
 *
 * Electric Weyl: E_ij = R_ij + K K_ij - K_ik K^k_j  (trace-free)
 * Magnetic Weyl: B_ij = epsilon_{(i}^{kl} D_k K_{j)l}
 *   with epsilon_i^{kl} = sqrt(chi) * h_{im} * e^{mkl}
 *   and D = physical covariant derivative (B&S Eq. 3.30 Christoffel)
 *
 * Tetrad: Gram-Schmidt orthonormalization of {r_hat, theta_hat, phi_hat}
 * against the physical metric gamma_ij = h_ij / chi.
 *
 * Ref: B&S Eqs. (8.53)-(8.55), (3.30)
 * Ref: GRChombo Weyl4.impl.hpp (equation cross-reference)
 * ================================================================ */

/* Internal version taking explicit physical position for AMR support */
static void psi4_compute(const double *const *fields, const grid_t *g,
                          int ii, int jj, int kk,
                          const double pos[3], const double center[3],
                          double out[2])
{
    const int idx = IDX(g, ii, jj, kk);
    const int strides[3] = { STRIDE_X, STRIDE_Y(g), STRIDE_Z(g) };
    const double dx = g->dx;

    /* --- Load fields --- */
    double chi = fields[FIELD_CHI][idx];
    if (chi < 1e-12) chi = 1e-12;

    double h[3][3];
    h[0][0] = fields[FIELD_H11][idx]; h[0][1] = fields[FIELD_H12][idx]; h[0][2] = fields[FIELD_H13][idx];
    h[1][0] = h[0][1];                h[1][1] = fields[FIELD_H22][idx]; h[1][2] = fields[FIELD_H23][idx];
    h[2][0] = h[0][2];                h[2][1] = h[1][2];                h[2][2] = fields[FIELD_H33][idx];

    double K_val = fields[FIELD_K][idx];
    double Theta = fields[FIELD_THETA][idx];

    double A_loc[3][3];
    A_loc[0][0] = fields[FIELD_A11][idx]; A_loc[0][1] = fields[FIELD_A12][idx]; A_loc[0][2] = fields[FIELD_A13][idx];
    A_loc[1][0] = A_loc[0][1];           A_loc[1][1] = fields[FIELD_A22][idx]; A_loc[1][2] = fields[FIELD_A23][idx];
    A_loc[2][0] = A_loc[0][2];           A_loc[2][1] = A_loc[1][2];           A_loc[2][2] = fields[FIELD_A33][idx];

    /* --- First derivatives (chi, h, K, A) using fused d1+d2 where possible --- */
    double d1_chi[3], d2_chi_diag[3];
    double d1_h[3][3][3], d2_h_diag[3][3][3]; /* d2_h_diag[a][b][dir] = d^2 h_{ab}/dx_dir^2 */
    double d1_K[3];
    double d1_A[3][3][3]; /* d1_A[a][b][dir] */

    FOR1(dir) {
        int s = strides[dir];
        fd_d1_d2(fields[FIELD_CHI], idx, s, dx, &d1_chi[dir], &d2_chi_diag[dir]);
        fd_d1_d2(fields[FIELD_K], idx, s, dx, &d1_K[dir], &(double){0}); /* d2_K not needed */
        FOR2(a, b) {
            if (b < a) {
                d1_h[a][b][dir] = d1_h[b][a][dir];
                d2_h_diag[a][b][dir] = d2_h_diag[b][a][dir];
                d1_A[a][b][dir] = d1_A[b][a][dir];
            } else {
                fd_d1_d2(fields[h_idx[a][b]], idx, s, dx,
                         &d1_h[a][b][dir], &d2_h_diag[a][b][dir]);
                d1_A[a][b][dir] = fd_d1(fields[A_idx[a][b]], idx, s, dx);
            }
        }
    }

    /* Second derivatives: mixed partials */
    double d2_chi[3][3];
    double d2_h[3][3][3][3];
    FOR1(dir) {
        d2_chi[dir][dir] = d2_chi_diag[dir];
        FOR2(a, b) d2_h[a][b][dir][dir] = d2_h_diag[a][b][dir];
    }
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = d1 + 1; d2 < 3; d2++) {
            int s1 = strides[d1], s2 = strides[d2];
            d2_chi[d1][d2] = fd_d2_mixed(fields[FIELD_CHI], idx, s1, s2, dx);
            d2_chi[d2][d1] = d2_chi[d1][d2];
            for (int a = 0; a < 3; a++) {
                for (int b = a; b < 3; b++) {
                    d2_h[a][b][d1][d2] = fd_d2_mixed(fields[h_idx[a][b]], idx, s1, s2, dx);
                    d2_h[a][b][d2][d1] = d2_h[a][b][d1][d2];
                    d2_h[b][a][d1][d2] = d2_h[a][b][d1][d2];
                    d2_h[b][a][d2][d1] = d2_h[a][b][d1][d2];
                }
            }
        }
    }

    /* d1_Gamma for Ricci computation */
    double d1_Gamma[3][3];
    double Gamma[3] = { fields[FIELD_GAMMA1][idx], fields[FIELD_GAMMA2][idx],
                        fields[FIELD_GAMMA3][idx] };
    FOR1(dir) {
        int s = strides[dir];
        FOR1(a) d1_Gamma[a][dir] = fd_d1(fields[FIELD_GAMMA1 + a], idx, s, dx);
    }

    /* --- Conformal geometry --- */
    double h_UU[3][3];
    compute_inverse_sym(h, h_UU);

    chris_t chris;
    compute_christoffel(d1_h, h_UU, &chris);

    /* --- Conformal Ricci tensor (same as constraints.c:86-178) --- */
    double covdtilde2chi[3][3];
    FOR2(a, b) {
        covdtilde2chi[a][b] = d2_chi[a][b];
        FOR1(m) covdtilde2chi[a][b] -= chris.ULL[m][a][b] * d1_chi[m];
    }

    double chris_LLU[3][3][3] = {{{0}}};
    double boxtildechi = 0.0;
    double dchi_dot_dchi = 0.0;
    FOR2(a, b) {
        boxtildechi += covdtilde2chi[a][b] * h_UU[a][b];
        dchi_dot_dchi += d1_chi[a] * d1_chi[b] * h_UU[a][b];
        FOR2(c, d) chris_LLU[a][b][c] += h_UU[c][d] * chris.LLL[a][b][d];
    }

    ricci_t ricci;
    FOR2(a, b) {
        double ricci_hat = 0.0;
        FOR1(c) {
            ricci_hat += 0.5 * (h[c][a] * d1_Gamma[c][b]
                              + h[c][b] * d1_Gamma[c][a]);
            ricci_hat += 0.5 * Gamma[c] * d1_h[a][b][c];
            FOR1(d) {
                ricci_hat += -0.5 * h_UU[c][d] * d2_h[a][b][c][d]
                           + (chris.ULL[c][d][a] * chris_LLU[b][c][d]
                            + chris.ULL[c][d][b] * chris_LLU[a][c][d]
                            + chris.ULL[c][a][d] * chris_LLU[c][b][d]);
            }
        }

        double ricci_chi = 0.5 * (
            (GR_SPACEDIM - 2) * covdtilde2chi[a][b]
            + h[a][b] * boxtildechi
            - ((GR_SPACEDIM - 2) * d1_chi[a] * d1_chi[b]
               + GR_SPACEDIM * h[a][b] * dchi_dot_dchi) / (2.0 * chi)
        );

        /* Physical Ricci: R_ij = (ricci_chi + chi * ricci_hat) / chi */
        ricci.LL[a][b] = (ricci_chi + chi * ricci_hat) / chi;
    }

    /* --- Physical extrinsic curvature --- */
    double K_phys[3][3];
    double K3 = K_val / 3.0;
    FOR2(a, b) K_phys[a][b] = (A_loc[a][b] + K3 * h[a][b]) / chi;

    /* K^i_j = gamma^{ik} K_{kj} = chi * h^{ik} * K_{kj} */
    double K_mixed[3][3]; /* K^i_j */
    FOR2(i, j) {
        K_mixed[i][j] = 0.0;
        FOR1(k) K_mixed[i][j] += chi * h_UU[i][k] * K_phys[k][j];
    }

    /* --- Electric Weyl tensor ---
     * E_ij = R_ij + (K - Theta) * K_ij - K_ik * K^k_j
     * Then make trace-free w.r.t. physical metric gamma_ij = h_ij/chi.
     *
     * In CCZ4, the evolved K contains a Theta contribution. The Gauss-Codazzi
     * relation for the Weyl tensor requires (K - Theta), NOT (K - 2*Theta).
     * GRChombo Weyl4.impl.hpp uses K_minus_theta with explicit comment:
     *   "Not a mistake, this is not to confuse with the typical K-2*Theta"
     * Ref: B&S Eq. (8.54), GRChombo Weyl4.impl.hpp */
    double K_minus_Theta = K_val - Theta;
    double E[3][3];
    FOR2(i, j) {
        double KK_term = 0.0;
        FOR1(k) KK_term += K_phys[i][k] * K_mixed[k][j];
        E[i][j] = ricci.LL[i][j] + K_minus_Theta * K_phys[i][j] - KK_term;
    }

    /* Make trace-free: E_ij -= (1/3) gamma_ij gamma^{kl} E_{kl}
     * gamma^{kl} E_{kl} = chi * h^{kl} * E_{kl} */
    double trE = 0.0;
    FOR2(k, l) trE += chi * h_UU[k][l] * E[k][l];
    double gamma_ij;
    FOR2(i, j) {
        gamma_ij = h[i][j] / chi;
        E[i][j] -= (1.0 / 3.0) * gamma_ij * trE;
    }

    /* --- Magnetic Weyl tensor ---
     * B_ij = (1/2)[epsilon_i^{kl} D_k K_{lj} + epsilon_j^{kl} D_k K_{li}]
     *
     * epsilon_i^{kl} = sqrt(chi) * h_{im} * e^{mkl}
     * (since gamma = h/chi, det(gamma) = 1/chi^3, epsilon^{mkl} = chi^{3/2} e^{mkl},
     *  and gamma_{im} = h_{im}/chi, so epsilon_i^{kl} = sqrt(chi) h_{im} e^{mkl})
     *
     * D_k K_{lj} = partial_k K_{lj} - Gamma^p_{kl} K_{pj} - Gamma^p_{kj} K_{lp}
     * (physical Christoffel from B&S Eq. 3.30)
     * Ref: B&S Eq. (8.55) */

    /* Physical Christoffel: B&S Eq. (3.30)
     * Gamma^i_{jk}(phys) = Gamma^i_{jk}(conf)
     *   - (1/(2 chi)) [delta^i_j d_k chi + delta^i_k d_j chi - h_{jk} h^{il} d_l chi]
     * Same as ah_finder.c:194-203 */
    double phys_chris[3][3][3]; /* Gamma^i_{jk} physical */
    double inv2chi = 0.5 / chi;
    FOR3(i, j, k) {
        double h_grad_i = 0.0;
        FOR1(l) h_grad_i += h_UU[i][l] * d1_chi[l];
        phys_chris[i][j][k] = chris.ULL[i][j][k]
            - inv2chi * (DELTA(i, j) * d1_chi[k]
                       + DELTA(i, k) * d1_chi[j]
                       - h[j][k] * h_grad_i);
    }

    /* Partial derivative of physical K_{lj}:
     * K_{lj} = (A_{lj} + K/3 * h_{lj}) / chi
     * d_k K_{lj} = [d_k A_{lj} + (d_k K/3) h_{lj} + (K/3) d_k h_{lj}] / chi
     *              - K_{lj} * d_k chi / chi */
    double dK_phys[3][3][3]; /* dK_phys[l][j][k] = partial_k K_{lj} */
    FOR3(l, j, k) {
        int la = l, jb = j;
        if (la > jb) { la = j; jb = l; } /* exploit symmetry for d1_A, d1_h */
        dK_phys[l][j][k] = (d1_A[la][jb][k]
                           + (d1_K[k] / 3.0) * h[l][j]
                           + K3 * d1_h[l][j][k]) / chi
                          - K_phys[l][j] * d1_chi[k] / chi;
    }

    /* Covariant derivative: D_k K_{lj} = d_k K_{lj} - Gamma^p_{kl} K_{pj} - Gamma^p_{kj} K_{lp} */
    double covd_K[3][3][3]; /* covd_K[k][l][j] = D_k K_{lj} */
    FOR3(k, l, j) {
        covd_K[k][l][j] = dK_phys[l][j][k];
        FOR1(p) {
            covd_K[k][l][j] -= phys_chris[p][k][l] * K_phys[p][j]
                              + phys_chris[p][k][j] * K_phys[l][p];
        }
    }

    /* B_ij = (1/2) [epsilon_i^{kl} D_k K_{lj} + epsilon_j^{kl} D_k K_{li}]
     * epsilon_i^{kl} = sqrt(chi) * sum_m h_{im} e_{mkl} */
    double sqrt_chi = sqrt(chi);
    double B[3][3];
    FOR2(i, j) {
        double sum_i = 0.0, sum_j = 0.0;
        FOR3(m, k, l) {
            int e_mkl = levi_civita(m, k, l);
            if (e_mkl == 0) continue;
            sum_i += h[i][m] * e_mkl * covd_K[k][l][j];
            sum_j += h[j][m] * e_mkl * covd_K[k][l][i];
        }
        B[i][j] = 0.5 * sqrt_chi * (sum_i + sum_j);
    }

    /* Make trace-free (should be automatic if constraints satisfied,
     * but enforce numerically for robustness) */
    double trB = 0.0;
    FOR2(k, l) trB += chi * h_UU[k][l] * B[k][l];
    FOR2(i, j) {
        gamma_ij = h[i][j] / chi;
        B[i][j] -= (1.0 / 3.0) * gamma_ij * trB;
    }

    /* --- Null tetrad construction ---
     * Coordinate vectors from extraction center to grid point.
     * Gram-Schmidt orthonormalization w.r.t. physical metric gamma_ij = h_ij/chi. */

    double x_rel = pos[0] - center[0];
    double y_rel = pos[1] - center[1];
    double z_rel = pos[2] - center[2];
    double r = sqrt(x_rel * x_rel + y_rel * y_rel + z_rel * z_rel);
    if (r < 1e-12) { out[0] = 0.0; out[1] = 0.0; return; }

    double rho = sqrt(x_rel * x_rel + y_rel * y_rel);

    /* Coordinate basis vectors */
    double r_hat[3] = { x_rel / r, y_rel / r, z_rel / r };
    double theta_hat[3], phi_hat[3];

    if (rho > 1e-10 * r) {
        theta_hat[0] = x_rel * z_rel / (r * rho);
        theta_hat[1] = y_rel * z_rel / (r * rho);
        theta_hat[2] = -rho / r;
        phi_hat[0] = -y_rel / rho;
        phi_hat[1] =  x_rel / rho;
        phi_hat[2] =  0.0;
    } else {
        /* On the z-axis: arbitrary choice of transverse directions */
        if (z_rel > 0) {
            theta_hat[0] = 1.0; theta_hat[1] = 0.0; theta_hat[2] = 0.0;
        } else {
            theta_hat[0] = -1.0; theta_hat[1] = 0.0; theta_hat[2] = 0.0;
        }
        phi_hat[0] = 0.0; phi_hat[1] = 1.0; phi_hat[2] = 0.0;
    }

    /* Inner product w.r.t. physical metric: <u, v>_gamma = gamma_{ij} u^i v^j = h_{ij} u^i v^j / chi */
    #define DOT_PHYS(u, v) ({ \
        double _d = 0.0; \
        for (int _i = 0; _i < 3; _i++) \
            for (int _j = 0; _j < 3; _j++) \
                _d += h[_i][_j] * (u)[_i] * (v)[_j] / chi; \
        _d; })

    /* Gram-Schmidt: orthonormalize {r_hat, theta_hat, phi_hat} */
    /* Step 1: normalize r_hat */
    double nr = sqrt(DOT_PHYS(r_hat, r_hat));
    double e_r[3];
    FOR1(i) e_r[i] = r_hat[i] / nr;

    /* Step 2: theta_hat → v = theta_hat - <theta_hat, e_r>*e_r, normalize */
    double proj = DOT_PHYS(theta_hat, e_r);
    double v[3];
    FOR1(i) v[i] = theta_hat[i] - proj * e_r[i];
    double nv = sqrt(DOT_PHYS(v, v));
    if (nv < 1e-12) { out[0] = 0.0; out[1] = 0.0; return; }
    FOR1(i) v[i] /= nv;

    /* Step 3: phi_hat → w = phi_hat - <phi_hat, e_r>*e_r - <phi_hat, v>*v, normalize */
    double proj_r = DOT_PHYS(phi_hat, e_r);
    double proj_v = DOT_PHYS(phi_hat, v);
    double w[3];
    FOR1(i) w[i] = phi_hat[i] - proj_r * e_r[i] - proj_v * v[i];
    double nw = sqrt(DOT_PHYS(w, w));
    if (nw < 1e-12) { out[0] = 0.0; out[1] = 0.0; return; }
    FOR1(i) w[i] /= nw;

    #undef DOT_PHYS

    /* --- Project Weyl tensors onto tetrad ---
     * Re(Psi4) = (E_vv - E_ww)/2 + B_vw
     * Im(Psi4) = (B_vv - B_ww)/2 - E_vw
     * Ref: GRChombo Weyl4.impl.hpp (same sign convention) */
    double E_vv = 0.0, E_ww = 0.0, E_vw = 0.0;
    double B_vv = 0.0, B_ww = 0.0, B_vw = 0.0;
    FOR2(i, j) {
        E_vv += E[i][j] * v[i] * v[j];
        E_ww += E[i][j] * w[i] * w[j];
        E_vw += E[i][j] * v[i] * w[j];
        B_vv += B[i][j] * v[i] * v[j];
        B_ww += B[i][j] * w[i] * w[j];
        B_vw += B[i][j] * v[i] * w[j];
    }

    out[0] = 0.5 * (E_vv - E_ww) + B_vw;
    out[1] = 0.5 * (B_vv - B_ww) - E_vw;
}

void psi4_at_point(const double *const *fields, const grid_t *g,
                   int i, int j, int k,
                   const double center[3], double out[2])
{
    double pos[3] = { COORD(g, i), COORD(g, j), COORD(g, k) };
    psi4_compute(fields, g, i, j, k, pos, center, out);
}

/* ================================================================
 * 5. Sphere extraction (mesh-aware)
 *
 * For each (theta, phi) on the GL×trapezoidal sphere:
 *   1. Compute Cartesian position from (theta, phi, radius)
 *   2. Find containing block via mesh_find_block_at()
 *   3. Snap to nearest interior grid point
 *   4. Call psi4_compute() at that grid point
 *   5. Multiply by r to get r*Psi4
 * ================================================================ */

void psi4_extract(psi4_workspace_t *ws, const struct mesh_s *m)
{
    double r = ws->radius;
    double dphi = 2.0 * M_PI / ws->n_phi;
    block_t *cached = NULL;

    for (int ith = 0; ith < ws->n_theta; ith++) {
        double th = ws->theta[ith];
        double st = sin(th), ct = cos(th);

        for (int iph = 0; iph < ws->n_phi; iph++) {
            int aidx = ith * ws->n_phi + iph;
            double ph = dphi * iph;
            double sp = sin(ph), cp = cos(ph);

            /* Cartesian position on sphere */
            double x = ws->center[0] + r * st * cp;
            double y = ws->center[1] + r * st * sp;
            double z = ws->center[2] + r * ct;

            /* Find block containing this point (check cache first) */
            block_t *b = cached;
            if (b) {
                double bx = b->grid->dx;
                int N = b->grid->N;
                int inside = 1;
                for (int d = 0; d < 3; d++) {
                    double coord = (d == 0) ? x : (d == 1) ? y : z;
                    if (coord < b->origin[d] || coord >= b->origin[d] + N * bx)
                        { inside = 0; break; }
                }
                if (!inside) b = NULL;
            }
            if (!b) b = mesh_find_block_at(m, x, y, z);
            if (!b) {
                ws->re_psi4[aidx] = 0.0;
                ws->im_psi4[aidx] = 0.0;
                continue;
            }
            cached = b;

            grid_t *g = b->grid;
            int ghost = g->ghost;

            /* Convert to grid indices (block-local) */
            double cix = (x - b->origin[0]) / g->dx - 0.5 + ghost;
            double ciy = (y - b->origin[1]) / g->dx - 0.5 + ghost;
            double ciz = (z - b->origin[2]) / g->dx - 0.5 + ghost;

            /* Snap to nearest interior grid point */
            int gi = (int)round(cix);
            int gj = (int)round(ciy);
            int gk = (int)round(ciz);

            /* Clamp to interior with margin for FD stencil (need ±3) */
            int lo = ghost, hi_bound = ghost + g->N - 1;
            if (gi < lo) gi = lo;
            if (gi > hi_bound) gi = hi_bound;
            if (gj < lo) gj = lo;
            if (gj > hi_bound) gj = hi_bound;
            if (gk < lo) gk = lo;
            if (gk > hi_bound) gk = hi_bound;

            /* Physical position of this grid point */
            double gx = b->origin[0] + (gi - ghost + 0.5) * g->dx;
            double gy = b->origin[1] + (gj - ghost + 0.5) * g->dx;
            double gz = b->origin[2] + (gk - ghost + 0.5) * g->dx;
            double gpos[3] = { gx, gy, gz };

            double psi4_val[2];
            psi4_compute((const double *const *)g->fields, g,
                         gi, gj, gk, gpos, ws->center, psi4_val);

            /* r * Psi4 */
            ws->re_psi4[aidx] = r * psi4_val[0];
            ws->im_psi4[aidx] = r * psi4_val[1];
        }
    }

    /* --- Mode decomposition ---
     * a_{lm} = integral of Psi4 * conj(_{-2}Y_{lm}) sin(theta) dtheta dphi
     *
     * GL nodes handle sin(theta) dtheta → dx (x = cos(theta)).
     * Trapezoidal rule for phi (spectrally accurate for periodic functions).
     */
    double dphi_w = 2.0 * M_PI / ws->n_phi;

    for (int mi = 0; mi < ws->n_modes; mi++) {
        ws->mode_re[mi] = 0.0;
        ws->mode_im[mi] = 0.0;
    }

    for (int l = 2; l <= ws->l_max; l++) {
        for (int mm = -l; mm <= l; mm++) {
            int mi = mode_index(l, mm);
            double a_re = 0.0, a_im = 0.0;

            for (int ith = 0; ith < ws->n_theta; ith++) {
                double th = ws->theta[ith];
                double wth = ws->gl_weights[ith];

                for (int iph = 0; iph < ws->n_phi; iph++) {
                    int aidx = ith * ws->n_phi + iph;
                    double ph = dphi_w * iph;

                    /* conj(_{-2}Y_{lm}) */
                    double ylm_re, ylm_im;
                    spin_weighted_Ylm(l, mm, th, ph, &ylm_re, &ylm_im);
                    double ylm_conj_re = ylm_re;
                    double ylm_conj_im = -ylm_im;

                    /* Psi4 * conj(Y_{lm}): (a+bi)(c+di) = (ac-bd) + (ad+bc)i */
                    double p_re = ws->re_psi4[aidx];
                    double p_im = ws->im_psi4[aidx];
                    a_re += wth * dphi_w * (p_re * ylm_conj_re - p_im * ylm_conj_im);
                    a_im += wth * dphi_w * (p_re * ylm_conj_im + p_im * ylm_conj_re);
                }
            }

            ws->mode_re[mi] = a_re;
            ws->mode_im[mi] = a_im;
        }
    }
}

/* ================================================================
 * 6. CSV output
 * ================================================================ */

void psi4_write_modes(const psi4_workspace_t *ws, double time,
                      const char *filename)
{
    int new_file = 0;
    FILE *f = fopen(filename, "r");
    if (!f) new_file = 1;
    else fclose(f);

    f = fopen(filename, "a");
    if (!f) return;

    if (new_file)
        fprintf(f, "# t, l, m, Re(rPsi4), Im(rPsi4), |rPsi4|, phase\n");

    for (int l = 2; l <= ws->l_max; l++) {
        for (int mm = -l; mm <= l; mm++) {
            int mi = mode_index(l, mm);
            double re = ws->mode_re[mi];
            double im = ws->mode_im[mi];
            double amp = sqrt(re * re + im * im);
            double phase = atan2(im, re);
            fprintf(f, "%.6f, %d, %d, %.12e, %.12e, %.12e, %.8f\n",
                    time, l, mm, re, im, amp, phase);
        }
    }

    fclose(f);
}
