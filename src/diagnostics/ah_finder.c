/*
 * Lattice -- 3D Numerical Relativity
 * Apparent horizon finder via hyperbolic flow method.
 *
 * Trial surface r = h(theta, phi) evolved via damped wave equation
 * in pseudo-time tau until the null expansion Theta vanishes.
 *
 * The expansion is computed from interpolated CCZ4 fields at each
 * angular grid point, using 4th-order Lagrange interpolation.
 *
 * Ref: Thornburg, CQG 4 (1987) 1119 (AH finder formulation)
 * Ref: Thornburg, PRD 54 (1996) 4899 (flow method)
 * Ref: Baumgarte & Shapiro, "Numerical Relativity" Eq. (6.1) (expansion)
 */

#include "ah_finder.h"
#include "../core/fields.h"
#include "../numerics/interpolate.h"
#include "../geometry/tensor_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Field index tables (file-scope for reuse) */
static const int h_field_idx[3][3] = {
    {FIELD_H11, FIELD_H12, FIELD_H13},
    {FIELD_H12, FIELD_H22, FIELD_H23},
    {FIELD_H13, FIELD_H23, FIELD_H33}
};
static const int A_field_idx[3][3] = {
    {FIELD_A11, FIELD_A12, FIELD_A13},
    {FIELD_A12, FIELD_A22, FIELD_A23},
    {FIELD_A13, FIELD_A23, FIELD_A33}
};

/* ================================================================
 * 1. Workspace allocation / free
 * ================================================================ */

ah_workspace_t *ah_alloc(int n_theta, int n_phi,
                          const double center[3], double r_guess)
{
    ah_workspace_t *ws = calloc(1, sizeof(ah_workspace_t));
    ws->n_theta = n_theta;
    ws->n_phi   = n_phi;
    ws->eta     = 5.0;
    ws->c_wave  = 1.0;
    ws->center[0] = center[0];
    ws->center[1] = center[1];
    ws->center[2] = center[2];

    int np = n_theta * n_phi;
    ws->h         = malloc((size_t)np * sizeof(double));
    ws->v         = calloc((size_t)np, sizeof(double));
    ws->rhs_h     = malloc((size_t)np * sizeof(double));
    ws->rhs_v     = malloc((size_t)np * sizeof(double));
    ws->scratch_h = malloc((size_t)np * sizeof(double));
    ws->scratch_v = malloc((size_t)np * sizeof(double));
    ws->accum_h   = malloc((size_t)np * sizeof(double));
    ws->accum_v   = malloc((size_t)np * sizeof(double));
    ws->theta_arr = malloc((size_t)np * sizeof(double));

    for (int i = 0; i < np; i++)
        ws->h[i] = r_guess;

    return ws;
}

void ah_free(ah_workspace_t *ws)
{
    if (!ws) return;
    free(ws->h);
    free(ws->v);
    free(ws->rhs_h);
    free(ws->rhs_v);
    free(ws->scratch_h);
    free(ws->scratch_v);
    free(ws->accum_h);
    free(ws->accum_v);
    free(ws->theta_arr);
    free(ws);
}

/* ================================================================
 * 2. Angular grid helpers
 * ================================================================ */

static inline double ah_theta(const ah_workspace_t *ws, int i)
{
    return M_PI * i / (ws->n_theta - 1);
}

static inline double ah_phi(const ah_workspace_t *ws, int j)
{
    return 2.0 * M_PI * j / ws->n_phi;
}

static inline int ah_idx(const ah_workspace_t *ws, int i, int j)
{
    return i * ws->n_phi + j;
}

static inline void ah_position(const ah_workspace_t *ws, int i, int j,
                                double h_val, double pos[3])
{
    double th = ah_theta(ws, i);
    double ph = ah_phi(ws, j);
    pos[0] = ws->center[0] + h_val * sin(th) * cos(ph);
    pos[1] = ws->center[1] + h_val * sin(th) * sin(ph);
    pos[2] = ws->center[2] + h_val * cos(th);
}

/* Coordinate radial unit vector at angular point (i, j) */
static inline void ah_normal(const ah_workspace_t *ws, int i, int j,
                              double n[3])
{
    double th = ah_theta(ws, i);
    double ph = ah_phi(ws, j);
    n[0] = sin(th) * cos(ph);
    n[1] = sin(th) * sin(ph);
    n[2] = cos(th);
}

/* ================================================================
 * 3. Expansion computation
 * ================================================================ */

/*
 * Public API: compute expansion at arbitrary (x,y,z) with unit normal s[3].
 * This version computes the expansion for a coordinate sphere of radius r_coord,
 * centered such that the surface point is at (x,y,z) with outward direction s[3].
 *
 * Theta = D_i s^i + K_{ij} s^i s^j - K
 *
 * CCZ4 -> physical conversion:
 *   gamma_{ij} = h_{ij} / chi
 *   K_{ij} = (A_{ij} + (K/3) h_{ij}) / chi
 *   Physical Christoffel from conformal + chi gradient corrections.
 *
 * Ref: B&S "Numerical Relativity" Eq. (6.1), (3.30)
 */
double compute_expansion(const grid_t *g, double x, double y, double z,
                          const double s[3])
{
    const double *const *f = (const double *const *)g->fields;

    /* Interpolate chi and derivatives */
    double chi_vd[4];
    interp_field_deriv_at(f[FIELD_CHI], g, x, y, z, chi_vd);
    double chi = chi_vd[0];
    double d_chi[3] = { chi_vd[1], chi_vd[2], chi_vd[3] };
    if (chi < 1e-12) chi = 1e-12;

    double K_val = interp_field_at(f[FIELD_K], g, x, y, z);

    /* Conformal metric and derivatives */
    double h_met[3][3], d_h[3][3][3];
    for (int a = 0; a < 3; a++) {
        for (int b = a; b < 3; b++) {
            double vd[4];
            interp_field_deriv_at(f[h_field_idx[a][b]], g, x, y, z, vd);
            h_met[a][b] = vd[0]; h_met[b][a] = vd[0];
            for (int d = 0; d < 3; d++) {
                d_h[a][b][d] = vd[1 + d];
                d_h[b][a][d] = vd[1 + d];
            }
        }
    }

    /* Conformal A_ij */
    double A_conf[3][3];
    for (int a = 0; a < 3; a++) {
        for (int b = a; b < 3; b++) {
            A_conf[a][b] = interp_field_at(f[A_field_idx[a][b]], g, x, y, z);
            A_conf[b][a] = A_conf[a][b];
        }
    }

    /* Inverse conformal metric and Christoffels */
    double h_UU[3][3];
    compute_inverse_sym(h_met, h_UU);
    chris_t chris_conf;
    compute_christoffel(d_h, h_UU, &chris_conf);

    /* Physical Christoffel:
     * Gamma^i_{jk}(phys) = Gamma^i_{jk}(conf)
     *   - (1/(2 chi)) (delta^i_j d_k chi + delta^i_k d_j chi - h_{jk} h^{il} d_l chi)
     * Ref: B&S Eq. (3.30) */
    double phys_chris[3][3][3];
    double inv2chi = 0.5 / chi;
    FOR3(i, j, k) {
        double h_grad_chi_i = 0.0;
        FOR1(l) h_grad_chi_i += h_UU[i][l] * d_chi[l];
        phys_chris[i][j][k] = chris_conf.ULL[i][j][k]
            - inv2chi * (DELTA(i, j) * d_chi[k]
                       + DELTA(i, k) * d_chi[j]
                       - h_met[j][k] * h_grad_chi_i);
    }

    /* Physical K_ij = (A_ij + (K/3) h_ij) / chi */
    double K_phys[3][3];
    double K3 = K_val / 3.0;
    FOR2(i, j) K_phys[i][j] = (A_conf[i][j] + K3 * h_met[i][j]) / chi;

    /* Normalize s^i w.r.t. physical metric gamma_{ij} = h_{ij}/chi */
    double s_norm_sq = 0.0;
    FOR2(i, j) s_norm_sq += h_met[i][j] * s[i] * s[j] / chi;
    double alpha_s = sqrt(s_norm_sq);
    if (alpha_s < 1e-12) alpha_s = 1e-12;
    double s_hat[3];
    FOR1(i) s_hat[i] = s[i] / alpha_s;

    /* Contracted physical Christoffel: Gamma^i_{ij}(phys) s^j_hat */
    double chris_contract = 0.0;
    FOR2(i, j) chris_contract += phys_chris[i][i][j] * s_hat[j];

    /* For a coordinate sphere of radius r with normal n^i = s * alpha_s:
     * partial_i n^i = 2/r (flat-space result)
     * partial_i s^i involves metric corrections to alpha_s.
     *
     * D_i s^i = partial_i s^i + Gamma^i_{ij}(phys) s^j
     *
     * We compute partial_i(s^i) from the chain rule:
     * s^i = n^i / alpha_s, so partial_j s^i needs d_j n^i and d_j alpha_s.
     *
     * For this standalone function without surface context, we compute
     * the projection-based expansion:
     * Theta = (gamma^{ij} - s^i s^j)(nabla_i s_j) + K_{ij} s^i s^j - K
     *
     * Using the identity that for unit s^i:
     * gamma^{ij} nabla_i s_j = nabla_i s^i
     * s^i s^j nabla_i s_j = 0  (since |s|=1)
     * So Theta = nabla_i s^i + K_{ij} s^i s^j - K
     *
     * For a coordinate sphere, nabla_i s^i = partial_i s^i + Gamma^i_{ij} s^j.
     *
     * We estimate partial_i s^i ≈ 2/(r * alpha_s) as the leading term.
     * This is exact when the metric is conformally flat (h_ij = delta_ij).
     */

    /* Infer r from the normalization: if n^i n_i(flat) = 1, then
     * the surface point is at distance r from center.
     * We don't have r explicitly, so we estimate from the input normal
     * and any coordinate information we can infer.
     *
     * WORKAROUND: For the public API, assume the caller normalizes s[3]
     * as a coordinate unit vector (|s|_flat = 1), and we can estimate r
     * from the grid spacing. But this is imprecise.
     *
     * For the AH finder itself, we use ah_compute_expansion_at which has
     * full context. This function is primarily for testing (expansion sign checks). */

    /* Use contracted Christoffel as the divergence estimate when r is unknown.
     * Gamma^i_{ij} s^j captures the curvature contribution to D_i s^i.
     * The flat-space 2/r part vanishes in the limit of a flat metric far from sources,
     * while the Christoffel terms capture the strong-field effects near the horizon.
     *
     * For a proper computation we need r. Since s[3] is a unit coordinate vector
     * and position is (x,y,z), we can try to infer center from s and position:
     * center ≈ (x,y,z) - r * s, but we still need r.
     *
     * DECISION: Accept r_coord as optional context. If |s|_flat = 1 (coordinate
     * unit normal), then we can compute things. But we need the radius.
     * This API function is for testing only. The real computation is internal. */

    /* K_{ij} s^i s^j */
    double K_ss = 0.0;
    FOR2(i, j) K_ss += K_phys[i][j] * s_hat[i] * s_hat[j];

    /* Return partial expansion without the 2/r divergence term.
     * The caller can add it if r is known:
     * Theta_full = 2.0/(r * alpha_s) + chris_contract + K_ss - K_val
     *
     * For the test suite, we check the sign of the full expansion
     * using ah_compute_expansion_at which has full context. */
    return chris_contract + K_ss - K_val;
}

/* ================================================================
 * 3b. Internal expansion computation with full surface context
 * ================================================================ */

/*
 * Compute expansion at angular point (ith, iph) of the AH workspace.
 *
 * Theta = D_i s^i + K_{ij} s^i s^j - K
 *
 * D_i s^i is computed with the full chain rule for a coordinate sphere
 * of radius h centered at ws->center, in the physical metric gamma_{ij}.
 *
 * Ref: B&S "Numerical Relativity" Eq. (6.1), (3.30)
 */
static double ah_compute_expansion_at(const ah_workspace_t *ws,
                                       const grid_t *g,
                                       int ith, int iph,
                                       const double *h_arr)
{
    const double *const *f = (const double *const *)g->fields;
    int aidx = ah_idx(ws, ith, iph);
    double r = h_arr[aidx];
    if (r < 1e-10) return 1.0;

    double pos[3];
    ah_position(ws, ith, iph, r, pos);

    double n[3];
    ah_normal(ws, ith, iph, n);

    /* Interpolate chi + derivatives */
    double chi_vd[4];
    interp_field_deriv_at(f[FIELD_CHI], g, pos[0], pos[1], pos[2], chi_vd);
    double chi = chi_vd[0];
    double d_chi[3] = { chi_vd[1], chi_vd[2], chi_vd[3] };
    if (chi < 1e-12) chi = 1e-12;

    double K_val = interp_field_at(f[FIELD_K], g, pos[0], pos[1], pos[2]);

    /* Conformal metric + derivatives */
    double h_met[3][3], d_h[3][3][3];
    for (int a = 0; a < 3; a++) {
        for (int b = a; b < 3; b++) {
            double vd[4];
            interp_field_deriv_at(f[h_field_idx[a][b]], g,
                                  pos[0], pos[1], pos[2], vd);
            h_met[a][b] = vd[0]; h_met[b][a] = vd[0];
            for (int dd = 0; dd < 3; dd++) {
                d_h[a][b][dd] = vd[1 + dd];
                d_h[b][a][dd] = vd[1 + dd];
            }
        }
    }

    /* Conformal A_ij */
    double A_conf[3][3];
    for (int a = 0; a < 3; a++) {
        for (int b = a; b < 3; b++) {
            A_conf[a][b] = interp_field_at(f[A_field_idx[a][b]], g,
                                            pos[0], pos[1], pos[2]);
            A_conf[b][a] = A_conf[a][b];
        }
    }

    /* Inverse conformal metric and Christoffels */
    double h_UU[3][3];
    compute_inverse_sym(h_met, h_UU);
    chris_t chris_conf;
    compute_christoffel(d_h, h_UU, &chris_conf);

    /* Physical Christoffel (conformal + chi corrections):
     * Gamma^i_{jk}(phys) = Gamma^i_{jk}(conf)
     *   - (1/(2 chi)) (delta^i_j d_k chi + delta^i_k d_j chi - h_{jk} h^{il} d_l chi)
     * Ref: B&S Eq. (3.30) */
    double phys_chris[3][3][3];
    double inv2chi = 0.5 / chi;
    FOR3(i, j, k) {
        double h_grad_i = 0.0;
        FOR1(l) h_grad_i += h_UU[i][l] * d_chi[l];
        phys_chris[i][j][k] = chris_conf.ULL[i][j][k]
            - inv2chi * (DELTA(i, j) * d_chi[k]
                       + DELTA(i, k) * d_chi[j]
                       - h_met[j][k] * h_grad_i);
    }

    /* Physical K_ij = (A_ij + (K/3) h_ij) / chi */
    double K_phys[3][3];
    double K3 = K_val / 3.0;
    FOR2(i, j) K_phys[i][j] = (A_conf[i][j] + K3 * h_met[i][j]) / chi;

    /* Physical metric: gamma_{ij} = h_{ij} / chi */

    /* Normalize n^i w.r.t. physical metric:
     * |n|^2_phys = gamma_{ij} n^i n^j = h_{ij} n^i n^j / chi */
    double n_sq = 0.0;
    FOR2(i, j) n_sq += h_met[i][j] * n[i] * n[j] / chi;
    double alpha_s = sqrt(n_sq);
    if (alpha_s < 1e-12) alpha_s = 1e-12;

    /* Physical unit normal: s^i = n^i / alpha_s */
    double s_hat[3];
    FOR1(i) s_hat[i] = n[i] / alpha_s;

    /* === D_i s^i computation ===
     *
     * s^i = n^i / alpha_s
     * where n^i = (x^i - c^i)/r is the coordinate radial unit vector
     * and alpha_s normalizes w.r.t. physical metric.
     *
     * partial_j(n^i) = (delta^i_j - n^i n^j) / r
     * partial_j(alpha_s) = [d_j(gamma_{kl}) n^k n^l + 2 gamma_{kl} d_j(n^k) n^l] / (2 alpha_s)
     *
     * d_j(gamma_{kl}) = (d_j(h_{kl}) - h_{kl} d_j(chi)/chi) / chi
     * d_j(n^k) = (delta^k_j - n^k n^j) / r
     *
     * partial_j(s^i) = partial_j(n^i) / alpha_s - n^i partial_j(alpha_s) / alpha_s^2
     * trace: partial_i(s^i) = (2/r) / alpha_s - n^i d_i(alpha_s) / alpha_s^2
     */

    /* partial_i alpha_s for each direction */
    double d_alpha[3];
    FOR1(j) {
        /* Term from d_j(gamma_{kl}) n^k n^l */
        double term1 = 0.0;
        FOR2(k, l) {
            double d_gamma = (d_h[k][l][j] - h_met[k][l] * d_chi[j] / chi) / chi;
            term1 += d_gamma * n[k] * n[l];
        }

        /* Term from 2 gamma_{kl} d_j(n^k) n^l
         * d_j(n^k) = (delta^k_j - n^k n^j) / r
         * gamma_{kl} d_j(n^k) n^l = (1/r) * (gamma_{jl} n^l - n_sq * n^j)
         * (using gamma_{kl} delta^k_j = gamma_{jl}, gamma_{kl} n^k n^l = n_sq) */
        double term2 = 0.0;
        FOR1(l) term2 += h_met[j][l] / chi * n[l];
        term2 = (term2 - n_sq * n[j]) / r;

        d_alpha[j] = (term1 + 2.0 * term2) / (2.0 * alpha_s);
    }

    /* partial_i s^i = (2/r) / alpha_s - n^i d_i(alpha_s) / alpha_s^2 */
    double n_dot_dalpha = 0.0;
    FOR1(i) n_dot_dalpha += n[i] * d_alpha[i];
    double partial_div_s = 2.0 / (r * alpha_s) - n_dot_dalpha / (alpha_s * alpha_s);

    /* Christoffel contraction: Gamma^i_{ij}(phys) s^j */
    double chris_contract = 0.0;
    FOR2(i, j) chris_contract += phys_chris[i][i][j] * s_hat[j];

    double div_s = partial_div_s + chris_contract;

    /* K_{ij} s^i s^j */
    double K_ss = 0.0;
    FOR2(i, j) K_ss += K_phys[i][j] * s_hat[i] * s_hat[j];

    /* Expansion: Theta = D_i s^i + K_{ij} s^i s^j - K
     * Ref: B&S Eq. (6.1) */
    return div_s + K_ss - K_val;
}

/* ================================================================
 * 4. RHS computation for hyperbolic flow
 * ================================================================ */

/*
 * Compute RHS for the damped wave flow:
 *   dh/dtau = v
 *   dv/dtau = -eta * v - c^2 * Theta(h)
 *
 * Ref: Thornburg, PRD 54 (1996) 4899
 */
static void ah_compute_rhs(const ah_workspace_t *ws, const grid_t *g,
                            const double *h_in, const double *v_in,
                            double *rhs_h_out, double *rhs_v_out,
                            double *theta_out)
{
    for (int ith = 0; ith < ws->n_theta; ith++) {
        for (int iph = 0; iph < ws->n_phi; iph++) {
            int idx = ah_idx(ws, ith, iph);

            /* Pole handling: at theta=0 and theta=pi, h is independent of phi.
             * Compute once at phi=0, replicate to all phi. */
            if ((ith == 0 || ith == ws->n_theta - 1) && iph > 0) {
                int idx0 = ah_idx(ws, ith, 0);
                rhs_h_out[idx] = rhs_h_out[idx0];
                rhs_v_out[idx] = rhs_v_out[idx0];
                if (theta_out) theta_out[idx] = theta_out[idx0];
                continue;
            }

            double Theta = ah_compute_expansion_at(ws, g, ith, iph, h_in);

            rhs_h_out[idx] = v_in[idx];
            rhs_v_out[idx] = -ws->eta * v_in[idx]
                             - ws->c_wave * ws->c_wave * Theta;

            if (theta_out) theta_out[idx] = Theta;
        }
    }
}

/* ================================================================
 * 5. RK4 time stepping for AH flow
 * ================================================================ */

static void ah_enforce_poles(ah_workspace_t *ws, double *arr)
{
    if (ws->n_phi <= 1) return;

    /* North pole (ith=0): average over all phi */
    double avg = 0.0;
    for (int j = 0; j < ws->n_phi; j++) avg += arr[ah_idx(ws, 0, j)];
    avg /= ws->n_phi;
    for (int j = 0; j < ws->n_phi; j++) arr[ah_idx(ws, 0, j)] = avg;

    /* South pole (ith=n_theta-1) */
    int last = ws->n_theta - 1;
    avg = 0.0;
    for (int j = 0; j < ws->n_phi; j++) avg += arr[ah_idx(ws, last, j)];
    avg /= ws->n_phi;
    for (int j = 0; j < ws->n_phi; j++) arr[ah_idx(ws, last, j)] = avg;
}

static void ah_rk4_step(ah_workspace_t *ws, const grid_t *g, double dt_ah)
{
    int np = ws->n_theta * ws->n_phi;

    /* Stage 1: k1 = f(y_n) */
    ah_compute_rhs(ws, g, ws->h, ws->v, ws->rhs_h, ws->rhs_v, ws->theta_arr);
    for (int i = 0; i < np; i++) {
        ws->accum_h[i] = ws->rhs_h[i];
        ws->accum_v[i] = ws->rhs_v[i];
        ws->scratch_h[i] = ws->h[i] + 0.5 * dt_ah * ws->rhs_h[i];
        ws->scratch_v[i] = ws->v[i] + 0.5 * dt_ah * ws->rhs_v[i];
    }

    /* Stage 2: k2 = f(y_n + 0.5*dt*k1) */
    ah_compute_rhs(ws, g, ws->scratch_h, ws->scratch_v,
                   ws->rhs_h, ws->rhs_v, NULL);
    for (int i = 0; i < np; i++) {
        ws->accum_h[i] += 2.0 * ws->rhs_h[i];
        ws->accum_v[i] += 2.0 * ws->rhs_v[i];
        ws->scratch_h[i] = ws->h[i] + 0.5 * dt_ah * ws->rhs_h[i];
        ws->scratch_v[i] = ws->v[i] + 0.5 * dt_ah * ws->rhs_v[i];
    }

    /* Stage 3: k3 = f(y_n + 0.5*dt*k2) */
    ah_compute_rhs(ws, g, ws->scratch_h, ws->scratch_v,
                   ws->rhs_h, ws->rhs_v, NULL);
    for (int i = 0; i < np; i++) {
        ws->accum_h[i] += 2.0 * ws->rhs_h[i];
        ws->accum_v[i] += 2.0 * ws->rhs_v[i];
        ws->scratch_h[i] = ws->h[i] + dt_ah * ws->rhs_h[i];
        ws->scratch_v[i] = ws->v[i] + dt_ah * ws->rhs_v[i];
    }

    /* Stage 4: k4 = f(y_n + dt*k3) */
    ah_compute_rhs(ws, g, ws->scratch_h, ws->scratch_v,
                   ws->rhs_h, ws->rhs_v, NULL);
    for (int i = 0; i < np; i++) {
        ws->accum_h[i] += ws->rhs_h[i];
        ws->accum_v[i] += ws->rhs_v[i];
    }

    /* Update: y_{n+1} = y_n + (dt/6)(k1 + 2k2 + 2k3 + k4) */
    double dt6 = dt_ah / 6.0;
    for (int i = 0; i < np; i++) {
        ws->h[i] += dt6 * ws->accum_h[i];
        ws->v[i] += dt6 * ws->accum_v[i];
    }

    /* Floor radius and enforce pole consistency */
    for (int i = 0; i < np; i++) {
        if (ws->h[i] < 1e-6) ws->h[i] = 1e-6;
    }
    ah_enforce_poles(ws, ws->h);
    ah_enforce_poles(ws, ws->v);
}

/* ================================================================
 * 6. Main AH finder driver
 * ================================================================ */

int ah_find(ah_workspace_t *ws, const grid_t *g,
            double tol, int max_iter, int verbose)
{
    double dtheta = M_PI / (ws->n_theta - 1);
    double dt_ah = 0.3 * dtheta / ws->c_wave;

    double max_theta = 1e30;

    for (int iter = 0; iter < max_iter; iter++) {
        ah_rk4_step(ws, g, dt_ah);

        /* Check convergence every 10 iterations */
        if (iter % 10 == 0 || iter == max_iter - 1) {
            ah_compute_rhs(ws, g, ws->h, ws->v,
                           ws->rhs_h, ws->rhs_v, ws->theta_arr);

            max_theta = 0.0;
            int np = ws->n_theta * ws->n_phi;
            for (int i = 0; i < np; i++) {
                double absT = fabs(ws->theta_arr[i]);
                if (absT > max_theta) max_theta = absT;
            }

            if (verbose && iter % 100 == 0) {
                double mean_r = 0.0;
                for (int i = 0; i < np; i++) mean_r += ws->h[i];
                mean_r /= np;
                printf("  AH iter %5d: max|Theta| = %.6e, mean_r = %.6f\n",
                       iter, max_theta, mean_r);
            }

            if (max_theta < tol) {
                if (verbose)
                    printf("  AH converged at iter %d: max|Theta| = %.6e\n",
                           iter, max_theta);
                return 1;
            }
        }
    }

    if (verbose)
        printf("  AH did not converge: max|Theta| = %.6e (tol = %.6e)\n",
               max_theta, tol);
    return 0;
}

/* ================================================================
 * 7. Diagnostics from found horizon
 * ================================================================ */

/*
 * Compute area, mass, and spin from a converged AH surface.
 *
 * Area: A = integral sqrt(det(q_{ab})) dtheta dphi  (trapezoidal rule)
 *   where q_{ab} = gamma_{ij} e_a^i e_b^j is the induced 2-metric.
 *
 * Spin: J_z = (1/8pi) integral K_{ij} s^i phi^j dA
 *   where phi^i = (-y, x, 0) is the rotational Killing vector.
 *
 * Ref: Dreyer et al., PRD 67 (2003) 024018 (spin from AH)
 */
ah_result_t ah_compute_diagnostics(const ah_workspace_t *ws, const grid_t *g)
{
    ah_result_t res;
    memset(&res, 0, sizeof(res));
    res.center[0] = ws->center[0];
    res.center[1] = ws->center[1];
    res.center[2] = ws->center[2];

    const double *const *f = (const double *const *)g->fields;

    double dtheta = M_PI / (ws->n_theta - 1);
    double dphi = 2.0 * M_PI / ws->n_phi;

    double area = 0.0;
    double mean_r = 0.0;
    double spin_z = 0.0;
    int np = ws->n_theta * ws->n_phi;

    for (int ith = 0; ith < ws->n_theta; ith++) {
        double th = ah_theta(ws, ith);
        double st = sin(th), ct = cos(th);
        (void)ct;

        /* Trapezoidal weight: half at endpoints */
        double wth = dtheta;
        if (ith == 0 || ith == ws->n_theta - 1) wth *= 0.5;

        for (int iph = 0; iph < ws->n_phi; iph++) {
            double ph = ah_phi(ws, iph);
            double sp = sin(ph), cp = cos(ph);
            (void)sp; (void)cp;

            int idx = ah_idx(ws, ith, iph);
            double h_val = ws->h[idx];
            mean_r += h_val;

            double pos[3];
            ah_position(ws, ith, iph, h_val, pos);

            /* Interpolate chi and conformal metric */
            double chi_val = interp_field_at(f[FIELD_CHI], g,
                                              pos[0], pos[1], pos[2]);
            if (chi_val < 1e-12) chi_val = 1e-12;

            double h_met[3][3];
            for (int a = 0; a < 3; a++) {
                for (int b = a; b < 3; b++) {
                    h_met[a][b] = interp_field_at(f[h_field_idx[a][b]], g,
                                                   pos[0], pos[1], pos[2]);
                    h_met[b][a] = h_met[a][b];
                }
            }

            /* Physical metric gamma_{ij} = h_{ij} / chi */
            double gamma[3][3];
            FOR2(a, b) gamma[a][b] = h_met[a][b] / chi_val;

            /* Tangent vectors for sphere of radius h:
             * e_theta = dpos/dtheta, e_phi = dpos/dphi */
            double e_th[3] = {
                h_val * cos(th) * cos(ph),
                h_val * cos(th) * sin(ph),
                -h_val * st
            };
            double e_ph[3] = {
                -h_val * st * sin(ph),
                h_val * st * cos(ph),
                0.0
            };

            /* Induced 2-metric */
            double q_tt = 0.0, q_tp = 0.0, q_pp = 0.0;
            FOR2(a, b) {
                q_tt += gamma[a][b] * e_th[a] * e_th[b];
                q_tp += gamma[a][b] * e_th[a] * e_ph[b];
                q_pp += gamma[a][b] * e_ph[a] * e_ph[b];
            }

            double det_q = q_tt * q_pp - q_tp * q_tp;
            if (det_q < 0.0) det_q = 0.0;
            double sqrt_det_q = sqrt(det_q);

            area += sqrt_det_q * wth * dphi;

            /* Spin: J_z = (1/8pi) integral K_{ij} s^i phi^j dA
             * Ref: Dreyer et al., PRD 67 (2003) 024018, Eq. (9) */
            double n_vec[3];
            ah_normal(ws, ith, iph, n_vec);
            double n_sq_phys = 0.0;
            FOR2(a, b) n_sq_phys += gamma[a][b] * n_vec[a] * n_vec[b];
            double as_loc = sqrt(n_sq_phys > 0 ? n_sq_phys : 1e-24);
            double s_phys[3];
            FOR1(a) s_phys[a] = n_vec[a] / as_loc;

            /* Physical K_ij */
            double K_val = interp_field_at(f[FIELD_K], g,
                                            pos[0], pos[1], pos[2]);
            double A_conf[3][3];
            for (int a = 0; a < 3; a++) {
                for (int b = a; b < 3; b++) {
                    A_conf[a][b] = interp_field_at(f[A_field_idx[a][b]], g,
                                                    pos[0], pos[1], pos[2]);
                    A_conf[b][a] = A_conf[a][b];
                }
            }
            double K_phys[3][3];
            double K3 = K_val / 3.0;
            FOR2(a, b) K_phys[a][b] = (A_conf[a][b] + K3 * h_met[a][b]) / chi_val;

            /* Rotational Killing vector phi^i for J_z */
            double dx_loc = pos[0] - ws->center[0];
            double dy_loc = pos[1] - ws->center[1];
            double phi_vec[3] = { -dy_loc, dx_loc, 0.0 };

            double K_s_phi = 0.0;
            FOR2(a, b) K_s_phi += K_phys[a][b] * s_phys[a] * phi_vec[b];

            spin_z += K_s_phi * sqrt_det_q * wth * dphi;
        }
    }

    mean_r /= np;
    spin_z /= (8.0 * M_PI);

    res.area = area;
    res.mass_irr = sqrt(fabs(area) / (16.0 * M_PI));
    res.spin_mag = fabs(spin_z);
    res.mean_radius = mean_r;

    if (res.mass_irr > 1e-20) {
        res.chi_spin = res.spin_mag / (res.mass_irr * res.mass_irr);
        res.mass_christodoulou = sqrt(res.mass_irr * res.mass_irr
                                      + spin_z * spin_z
                                        / (4.0 * res.mass_irr * res.mass_irr));
    }

    /* Convergence from last theta_arr */
    double max_th = 0.0;
    for (int i = 0; i < np; i++) {
        double absT = fabs(ws->theta_arr[i]);
        if (absT > max_th) max_th = absT;
    }
    res.residual = max_th;
    res.converged = (max_th < 1e-4);

    return res;
}

/* ================================================================
 * 8. Evaluate expansion at all angular points (for testing)
 * ================================================================ */

void ah_eval_expansion(ah_workspace_t *ws, const grid_t *g)
{
    for (int ith = 0; ith < ws->n_theta; ith++) {
        for (int iph = 0; iph < ws->n_phi; iph++) {
            int idx = ah_idx(ws, ith, iph);

            if ((ith == 0 || ith == ws->n_theta - 1) && iph > 0) {
                ws->theta_arr[idx] = ws->theta_arr[ah_idx(ws, ith, 0)];
                continue;
            }

            ws->theta_arr[idx] = ah_compute_expansion_at(ws, g, ith, iph, ws->h);
        }
    }
}

/* ================================================================
 * 9. AMR mesh variants — block-aware interpolation
 *
 * Each function mirrors its single-grid counterpart but uses
 * mesh_find_block_at() + block-local coordinate transform to
 * interpolate from the finest-level block containing each point.
 *
 * Key insight: ghost width (4) > interpolation half-width (2), so
 * the stencil for any interior point fits entirely within one block.
 * No cross-block stencil needed.
 *
 * Coordinate transform for block-local interpolation:
 *   interp_field_at expects: x_local = phys - block_origin - L_block/2
 *   where L_block = b->grid->L = N_block * dx
 * ================================================================ */

#include "../amr/mesh.h"

/* Block-local interpolation: value at physical (x,y,z) from block b */
static inline double interp_at_block(const double *field, const block_t *b,
                                      double x, double y, double z)
{
    double half_L = b->grid->L * 0.5;
    return interp_field_at(field, b->grid,
                           x - b->origin[0] - half_L,
                           y - b->origin[1] - half_L,
                           z - b->origin[2] - half_L);
}

/* Block-local interpolation: value + gradient at physical (x,y,z) */
static inline void interp_deriv_at_block(const double *field, const block_t *b,
                                          double x, double y, double z,
                                          double val[4])
{
    double half_L = b->grid->L * 0.5;
    interp_field_deriv_at(field, b->grid,
                          x - b->origin[0] - half_L,
                          y - b->origin[1] - half_L,
                          z - b->origin[2] - half_L,
                          val);
}

/*
 * AMR expansion at angular point (ith, iph).
 * Same physics as ah_compute_expansion_at but uses block-aware interpolation.
 * cached_block is checked first before scanning the mesh (optimization).
 */
static double ah_compute_expansion_at_amr(const ah_workspace_t *ws,
                                           const mesh_t *m,
                                           int ith, int iph,
                                           const double *h_arr,
                                           block_t **cached_block)
{
    int aidx = ah_idx(ws, ith, iph);
    double r = h_arr[aidx];
    if (r < 1e-10) return 1.0;

    double pos[3];
    ah_position(ws, ith, iph, r, pos);

    /* Find block containing this point (check cache first) */
    block_t *b = *cached_block;
    if (b) {
        double dx = b->grid->dx;
        int N = b->grid->N;
        int inside = 1;
        for (int d = 0; d < 3; d++) {
            double coord = pos[d];
            if (coord < b->origin[d] || coord >= b->origin[d] + N * dx)
                { inside = 0; break; }
        }
        if (!inside) b = NULL;
    }
    if (!b) b = mesh_find_block_at(m, pos[0], pos[1], pos[2]);
    if (!b) return 1.0;  /* outside domain */
    *cached_block = b;

    const double *const *f = (const double *const *)b->grid->fields;

    double n[3];
    ah_normal(ws, ith, iph, n);

    /* Interpolate chi + derivatives */
    double chi_vd[4];
    interp_deriv_at_block(f[FIELD_CHI], b, pos[0], pos[1], pos[2], chi_vd);
    double chi = chi_vd[0];
    double d_chi[3] = { chi_vd[1], chi_vd[2], chi_vd[3] };
    if (chi < 1e-12) chi = 1e-12;

    double K_val = interp_at_block(f[FIELD_K], b, pos[0], pos[1], pos[2]);

    /* Conformal metric + derivatives */
    double h_met[3][3], d_h[3][3][3];
    for (int a = 0; a < 3; a++) {
        for (int bb = a; bb < 3; bb++) {
            double vd[4];
            interp_deriv_at_block(f[h_field_idx[a][bb]], b,
                                  pos[0], pos[1], pos[2], vd);
            h_met[a][bb] = vd[0]; h_met[bb][a] = vd[0];
            for (int dd = 0; dd < 3; dd++) {
                d_h[a][bb][dd] = vd[1 + dd];
                d_h[bb][a][dd] = vd[1 + dd];
            }
        }
    }

    /* Conformal A_ij */
    double A_conf[3][3];
    for (int a = 0; a < 3; a++) {
        for (int bb = a; bb < 3; bb++) {
            A_conf[a][bb] = interp_at_block(f[A_field_idx[a][bb]], b,
                                             pos[0], pos[1], pos[2]);
            A_conf[bb][a] = A_conf[a][bb];
        }
    }

    /* Inverse conformal metric and Christoffels */
    double h_UU[3][3];
    compute_inverse_sym(h_met, h_UU);
    chris_t chris_conf;
    compute_christoffel(d_h, h_UU, &chris_conf);

    /* Physical Christoffel: conformal + chi corrections.
     * Ref: B&S Eq. (3.30) */
    double phys_chris[3][3][3];
    double inv2chi = 0.5 / chi;
    FOR3(i, j, k) {
        double h_grad_i = 0.0;
        FOR1(l) h_grad_i += h_UU[i][l] * d_chi[l];
        phys_chris[i][j][k] = chris_conf.ULL[i][j][k]
            - inv2chi * (DELTA(i, j) * d_chi[k]
                       + DELTA(i, k) * d_chi[j]
                       - h_met[j][k] * h_grad_i);
    }

    /* Physical K_ij = (A_ij + (K/3) h_ij) / chi */
    double K_phys[3][3];
    double K3 = K_val / 3.0;
    FOR2(i, j) K_phys[i][j] = (A_conf[i][j] + K3 * h_met[i][j]) / chi;

    /* Normalize n^i w.r.t. physical metric */
    double n_sq = 0.0;
    FOR2(i, j) n_sq += h_met[i][j] * n[i] * n[j] / chi;
    double alpha_s = sqrt(n_sq);
    if (alpha_s < 1e-12) alpha_s = 1e-12;

    double s_hat[3];
    FOR1(i) s_hat[i] = n[i] / alpha_s;

    /* D_i s^i computation (same as single-grid version) */
    double d_alpha[3];
    FOR1(j) {
        double term1 = 0.0;
        FOR2(kk, l) {
            double d_gamma = (d_h[kk][l][j] - h_met[kk][l] * d_chi[j] / chi) / chi;
            term1 += d_gamma * n[kk] * n[l];
        }
        double term2 = 0.0;
        FOR1(l) term2 += h_met[j][l] / chi * n[l];
        term2 = (term2 - n_sq * n[j]) / r;
        d_alpha[j] = (term1 + 2.0 * term2) / (2.0 * alpha_s);
    }

    double n_dot_dalpha = 0.0;
    FOR1(i) n_dot_dalpha += n[i] * d_alpha[i];
    double partial_div_s = 2.0 / (r * alpha_s) - n_dot_dalpha / (alpha_s * alpha_s);

    double chris_contract = 0.0;
    FOR2(i, j) chris_contract += phys_chris[i][i][j] * s_hat[j];

    double div_s = partial_div_s + chris_contract;

    double K_ss = 0.0;
    FOR2(i, j) K_ss += K_phys[i][j] * s_hat[i] * s_hat[j];

    return div_s + K_ss - K_val;
}

/* AMR RHS for hyperbolic flow */
static void ah_compute_rhs_amr(const ah_workspace_t *ws, const mesh_t *m,
                                const double *h_in, const double *v_in,
                                double *rhs_h_out, double *rhs_v_out,
                                double *theta_out)
{
    block_t *cached = NULL;
    for (int ith = 0; ith < ws->n_theta; ith++) {
        for (int iph = 0; iph < ws->n_phi; iph++) {
            int idx = ah_idx(ws, ith, iph);

            if ((ith == 0 || ith == ws->n_theta - 1) && iph > 0) {
                int idx0 = ah_idx(ws, ith, 0);
                rhs_h_out[idx] = rhs_h_out[idx0];
                rhs_v_out[idx] = rhs_v_out[idx0];
                if (theta_out) theta_out[idx] = theta_out[idx0];
                continue;
            }

            double Theta = ah_compute_expansion_at_amr(ws, m, ith, iph,
                                                        h_in, &cached);

            rhs_h_out[idx] = v_in[idx];
            rhs_v_out[idx] = -ws->eta * v_in[idx]
                             - ws->c_wave * ws->c_wave * Theta;

            if (theta_out) theta_out[idx] = Theta;
        }
    }
}

/* AMR RK4 step for AH flow */
static void ah_rk4_step_amr(ah_workspace_t *ws, const mesh_t *m, double dt_ah)
{
    int np = ws->n_theta * ws->n_phi;

    /* Stage 1 */
    ah_compute_rhs_amr(ws, m, ws->h, ws->v, ws->rhs_h, ws->rhs_v, ws->theta_arr);
    for (int i = 0; i < np; i++) {
        ws->accum_h[i] = ws->rhs_h[i];
        ws->accum_v[i] = ws->rhs_v[i];
        ws->scratch_h[i] = ws->h[i] + 0.5 * dt_ah * ws->rhs_h[i];
        ws->scratch_v[i] = ws->v[i] + 0.5 * dt_ah * ws->rhs_v[i];
    }

    /* Stage 2 */
    ah_compute_rhs_amr(ws, m, ws->scratch_h, ws->scratch_v,
                       ws->rhs_h, ws->rhs_v, NULL);
    for (int i = 0; i < np; i++) {
        ws->accum_h[i] += 2.0 * ws->rhs_h[i];
        ws->accum_v[i] += 2.0 * ws->rhs_v[i];
        ws->scratch_h[i] = ws->h[i] + 0.5 * dt_ah * ws->rhs_h[i];
        ws->scratch_v[i] = ws->v[i] + 0.5 * dt_ah * ws->rhs_v[i];
    }

    /* Stage 3 */
    ah_compute_rhs_amr(ws, m, ws->scratch_h, ws->scratch_v,
                       ws->rhs_h, ws->rhs_v, NULL);
    for (int i = 0; i < np; i++) {
        ws->accum_h[i] += 2.0 * ws->rhs_h[i];
        ws->accum_v[i] += 2.0 * ws->rhs_v[i];
        ws->scratch_h[i] = ws->h[i] + dt_ah * ws->rhs_h[i];
        ws->scratch_v[i] = ws->v[i] + dt_ah * ws->rhs_v[i];
    }

    /* Stage 4 */
    ah_compute_rhs_amr(ws, m, ws->scratch_h, ws->scratch_v,
                       ws->rhs_h, ws->rhs_v, NULL);
    for (int i = 0; i < np; i++) {
        ws->accum_h[i] += ws->rhs_h[i];
        ws->accum_v[i] += ws->rhs_v[i];
    }

    /* Update */
    double dt6 = dt_ah / 6.0;
    for (int i = 0; i < np; i++) {
        ws->h[i] += dt6 * ws->accum_h[i];
        ws->v[i] += dt6 * ws->accum_v[i];
    }

    for (int i = 0; i < np; i++) {
        if (ws->h[i] < 1e-6) ws->h[i] = 1e-6;
    }
    ah_enforce_poles(ws, ws->h);
    ah_enforce_poles(ws, ws->v);
}

/* Public: AMR AH finder driver */
int ah_find_amr(ah_workspace_t *ws, const mesh_t *m,
                double tol, int max_iter, int verbose)
{
    double dtheta = M_PI / (ws->n_theta - 1);
    double dt_ah = 0.3 * dtheta / ws->c_wave;

    double max_theta = 1e30;

    for (int iter = 0; iter < max_iter; iter++) {
        ah_rk4_step_amr(ws, m, dt_ah);

        if (iter % 10 == 0 || iter == max_iter - 1) {
            ah_compute_rhs_amr(ws, m, ws->h, ws->v,
                               ws->rhs_h, ws->rhs_v, ws->theta_arr);

            max_theta = 0.0;
            int np = ws->n_theta * ws->n_phi;
            for (int i = 0; i < np; i++) {
                double absT = fabs(ws->theta_arr[i]);
                if (absT > max_theta) max_theta = absT;
            }

            if (verbose && iter % 100 == 0) {
                double mean_r = 0.0;
                for (int i = 0; i < np; i++) mean_r += ws->h[i];
                mean_r /= np;
                printf("  AH iter %5d: max|Theta| = %.6e, mean_r = %.6f\n",
                       iter, max_theta, mean_r);
            }

            if (max_theta < tol) {
                if (verbose)
                    printf("  AH converged at iter %d: max|Theta| = %.6e\n",
                           iter, max_theta);
                return 1;
            }
        }
    }

    if (verbose)
        printf("  AH did not converge: max|Theta| = %.6e (tol = %.6e)\n",
               max_theta, tol);
    return 0;
}

/* Public: AMR diagnostics from found horizon */
ah_result_t ah_compute_diagnostics_amr(const ah_workspace_t *ws,
                                        const mesh_t *m)
{
    ah_result_t res;
    memset(&res, 0, sizeof(res));
    res.center[0] = ws->center[0];
    res.center[1] = ws->center[1];
    res.center[2] = ws->center[2];

    double dtheta = M_PI / (ws->n_theta - 1);
    double dphi = 2.0 * M_PI / ws->n_phi;

    double area = 0.0;
    double mean_r = 0.0;
    double spin_z = 0.0;
    int np = ws->n_theta * ws->n_phi;
    block_t *cached = NULL;

    for (int ith = 0; ith < ws->n_theta; ith++) {
        double th = ah_theta(ws, ith);
        double st = sin(th);

        double wth = dtheta;
        if (ith == 0 || ith == ws->n_theta - 1) wth *= 0.5;

        for (int iph = 0; iph < ws->n_phi; iph++) {
            int idx = ah_idx(ws, ith, iph);
            double h_val = ws->h[idx];
            mean_r += h_val;

            double pos[3];
            ah_position(ws, ith, iph, h_val, pos);

            /* Find block for this point */
            block_t *b = cached;
            if (b) {
                double dx = b->grid->dx;
                int N = b->grid->N;
                int inside = 1;
                for (int d = 0; d < 3; d++) {
                    double coord = pos[d];
                    if (coord < b->origin[d] || coord >= b->origin[d] + N * dx)
                        { inside = 0; break; }
                }
                if (!inside) b = NULL;
            }
            if (!b) b = mesh_find_block_at(m, pos[0], pos[1], pos[2]);
            if (!b) continue;
            cached = b;

            const double *const *f = (const double *const *)b->grid->fields;

            double chi_val = interp_at_block(f[FIELD_CHI], b,
                                              pos[0], pos[1], pos[2]);
            if (chi_val < 1e-12) chi_val = 1e-12;

            double h_met[3][3];
            for (int a = 0; a < 3; a++) {
                for (int bb = a; bb < 3; bb++) {
                    h_met[a][bb] = interp_at_block(f[h_field_idx[a][bb]], b,
                                                    pos[0], pos[1], pos[2]);
                    h_met[bb][a] = h_met[a][bb];
                }
            }

            double gamma[3][3];
            FOR2(a, bb) gamma[a][bb] = h_met[a][bb] / chi_val;

            double e_th[3] = {
                h_val * cos(th) * cos(ah_phi(ws, iph)),
                h_val * cos(th) * sin(ah_phi(ws, iph)),
                -h_val * st
            };
            double e_ph[3] = {
                -h_val * st * sin(ah_phi(ws, iph)),
                h_val * st * cos(ah_phi(ws, iph)),
                0.0
            };

            double q_tt = 0.0, q_tp = 0.0, q_pp = 0.0;
            FOR2(a, bb) {
                q_tt += gamma[a][bb] * e_th[a] * e_th[bb];
                q_tp += gamma[a][bb] * e_th[a] * e_ph[bb];
                q_pp += gamma[a][bb] * e_ph[a] * e_ph[bb];
            }

            double det_q = q_tt * q_pp - q_tp * q_tp;
            if (det_q < 0.0) det_q = 0.0;
            double sqrt_det_q = sqrt(det_q);

            area += sqrt_det_q * wth * dphi;

            /* Spin */
            double n_vec[3];
            ah_normal(ws, ith, iph, n_vec);
            double n_sq_phys = 0.0;
            FOR2(a, bb) n_sq_phys += gamma[a][bb] * n_vec[a] * n_vec[bb];
            double as_loc = sqrt(n_sq_phys > 0 ? n_sq_phys : 1e-24);
            double s_phys[3];
            FOR1(a) s_phys[a] = n_vec[a] / as_loc;

            double K_val = interp_at_block(f[FIELD_K], b,
                                            pos[0], pos[1], pos[2]);
            double A_conf[3][3];
            for (int a = 0; a < 3; a++) {
                for (int bb = a; bb < 3; bb++) {
                    A_conf[a][bb] = interp_at_block(f[A_field_idx[a][bb]], b,
                                                     pos[0], pos[1], pos[2]);
                    A_conf[bb][a] = A_conf[a][bb];
                }
            }
            double K_phys[3][3];
            double K3 = K_val / 3.0;
            FOR2(a, bb) K_phys[a][bb] = (A_conf[a][bb] + K3 * h_met[a][bb]) / chi_val;

            double dx_loc = pos[0] - ws->center[0];
            double dy_loc = pos[1] - ws->center[1];
            double phi_vec[3] = { -dy_loc, dx_loc, 0.0 };

            double K_s_phi = 0.0;
            FOR2(a, bb) K_s_phi += K_phys[a][bb] * s_phys[a] * phi_vec[bb];

            spin_z += K_s_phi * sqrt_det_q * wth * dphi;
        }
    }

    mean_r /= np;
    spin_z /= (8.0 * M_PI);

    res.area = area;
    res.mass_irr = sqrt(fabs(area) / (16.0 * M_PI));
    res.spin_mag = fabs(spin_z);
    res.mean_radius = mean_r;

    if (res.mass_irr > 1e-20) {
        res.chi_spin = res.spin_mag / (res.mass_irr * res.mass_irr);
        res.mass_christodoulou = sqrt(res.mass_irr * res.mass_irr
                                      + spin_z * spin_z
                                        / (4.0 * res.mass_irr * res.mass_irr));
    }

    double max_th = 0.0;
    for (int i = 0; i < np; i++) {
        double absT = fabs(ws->theta_arr[i]);
        if (absT > max_th) max_th = absT;
    }
    res.residual = max_th;
    res.converged = (max_th < 1e-4);

    return res;
}

/* Public: AMR expansion evaluation */
void ah_eval_expansion_amr(ah_workspace_t *ws, const mesh_t *m)
{
    block_t *cached = NULL;
    for (int ith = 0; ith < ws->n_theta; ith++) {
        for (int iph = 0; iph < ws->n_phi; iph++) {
            int idx = ah_idx(ws, ith, iph);

            if ((ith == 0 || ith == ws->n_theta - 1) && iph > 0) {
                ws->theta_arr[idx] = ws->theta_arr[ah_idx(ws, ith, 0)];
                continue;
            }

            ws->theta_arr[idx] = ah_compute_expansion_at_amr(
                ws, m, ith, iph, ws->h, &cached);
        }
    }
}
