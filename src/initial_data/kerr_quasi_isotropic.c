/*
 * Lattice — 3D Numerical Relativity
 * Quasi-isotropic Kerr metric for HiSpID (High-Spin Initial Data).
 *
 * Computes the Kerr BH spatial metric and extrinsic curvature in
 * semi-isotropic coordinates (Liu, Etienne, Shapiro 2010), then
 * converts to Cartesian via Jacobian transformation and rotation
 * for arbitrary spin direction.
 *
 * For N BHs, uses Gaussian-weighted superposition of individual
 * Kerr metrics, ensuring smooth falloff to flat space far from BHs.
 *
 * Ref: arXiv:1410.8607 (Ruchlin et al., HiSpID)
 * Ref: arXiv:1001.4077 (Liu, Etienne, Shapiro — QI Kerr)
 * Ref: GRChombo KerrBH.impl.hpp (reference implementation, BSD 3-Clause)
 */

#include "kerr_quasi_isotropic.h"
#include "../geometry/tensor_utils.h"
#include <math.h>
#include <string.h>

/* ================================================================
 * Internal helpers
 * ================================================================ */

/*
 * Rotation matrix R that maps spin_dir to z_dir = (0,0,1).
 * Uses Rodrigues' formula.
 * Ref: GRChombo CoordinateTransformations.hpp:235-295
 */
static void rotation_matrix_to_z(const double spin_dir[3], double R[3][3])
{
    /* axis = spin_dir x z_dir */
    double ax = spin_dir[1] * 1.0 - spin_dir[2] * 0.0;  /* spin[1]*z[2] - spin[2]*z[1] */
    double ay = spin_dir[2] * 0.0 - spin_dir[0] * 1.0;  /* spin[2]*z[0] - spin[0]*z[2] */
    double az = spin_dir[0] * 0.0 - spin_dir[1] * 0.0;  /* spin[0]*z[1] - spin[1]*z[0] */

    double axis_norm = sqrt(ax * ax + ay * ay + az * az);

    if (axis_norm < 1e-13) {
        /* spin_dir is (anti-)parallel to z */
        if (spin_dir[2] > 0.0) {
            /* Identity */
            R[0][0] = 1; R[0][1] = 0; R[0][2] = 0;
            R[1][0] = 0; R[1][1] = 1; R[1][2] = 0;
            R[2][0] = 0; R[2][1] = 0; R[2][2] = 1;
        } else {
            /* 180 deg about x-axis */
            R[0][0] = 1;  R[0][1] = 0;  R[0][2] = 0;
            R[1][0] = 0;  R[1][1] = -1; R[1][2] = 0;
            R[2][0] = 0;  R[2][1] = 0;  R[2][2] = -1;
        }
        return;
    }

    /* Normalize axis */
    ax /= axis_norm;
    ay /= axis_norm;
    az /= axis_norm;

    /* cos(angle) = spin_dir . z_dir = spin_dir[2] */
    double c = spin_dir[2];
    double s = sqrt(1.0 - c * c);
    double omc = 1.0 - c;

    /* Rodrigues' formula
     * Ref: GRChombo CoordinateTransformations.hpp:247-258 */
    R[0][0] = ax * ax * omc + c;
    R[0][1] = ay * ax * omc - az * s;
    R[0][2] = az * ax * omc + ay * s;
    R[1][0] = ax * ay * omc + az * s;
    R[1][1] = ay * ay * omc + c;
    R[1][2] = az * ay * omc - ax * s;
    R[2][0] = ax * az * omc - ay * s;
    R[2][1] = ay * az * omc + ax * s;
    R[2][2] = az * az * omc + c;
}

/*
 * Transform a covariant 2-tensor from frame A to frame B:
 *   out[i][j] = sum_{a,b} J[a][i] * J[b][j] * T[a][b]
 * where J is the Jacobian of the transformation (or rotation matrix).
 *
 * Ref: GRChombo CoordinateTransformations.hpp:128-142
 */
static void transform_tensor_LL(const double T[3][3], const double J[3][3],
                                double out[3][3])
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            out[i][j] = 0.0;
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++)
                    out[i][j] += J[a][i] * J[b][j] * T[a][b];
        }
}

/*
 * Spherical Jacobian: jac[sph_idx][cart_idx] = d(sph)/d(cart)
 * sph = (r, theta, phi), cart = (x, y, z)
 *
 * Ref: GRChombo CoordinateTransformations.hpp:19-44
 */
static void spherical_jacobian(double jac[3][3],
                               double x, double y, double z)
{
    double rho2 = x * x + y * y;
    if (rho2 < 1e-12) rho2 = 1e-12;
    double rho = sqrt(rho2);
    double r2 = x * x + y * y + z * z;
    if (r2 < 1e-12) r2 = 1e-12;
    double r = sqrt(r2);

    double cos_phi = x / rho;
    double sin_phi = y / rho;

    jac[0][0] = x / r;               /* dr/dx */
    jac[0][1] = y / r;               /* dr/dy */
    jac[0][2] = z / r;               /* dr/dz */
    jac[1][0] = cos_phi * z / r2;    /* dtheta/dx */
    jac[1][1] = sin_phi * z / r2;    /* dtheta/dy */
    jac[1][2] = -rho / r2;           /* dtheta/dz */
    jac[2][0] = -y / rho2;           /* dphi/dx */
    jac[2][1] = x / rho2;            /* dphi/dy */
    jac[2][2] = 0.0;                 /* dphi/dz */
}

/*
 * Core Kerr metric computation in spherical quasi-isotropic coordinates.
 * Assumes spin is aligned with z-axis.  Coordinates (x, y, z) are
 * already in the rotated frame.
 *
 * Outputs:
 *   g_sph[3][3] — spatial metric in spherical coords (r, theta, phi)
 *   K_sph[3][3] — extrinsic curvature in spherical coords
 *   *psi_K      — Kerr conformal factor (det(g_sph)^{1/12} / r)
 *
 * Ref: arXiv:1001.4077 Eqs. (1)-(13)
 * Ref: GRChombo KerrBH.impl.hpp:97-174
 */
static void compute_kerr_spherical(double g_sph[3][3], double K_sph[3][3],
                                   double x, double y, double z,
                                   double M, double a)
{
    double r = sqrt(x * x + y * y + z * z);
    if (r < 1e-6) r = 1e-6;
    double r2 = r * r;

    double rho2 = x * x + y * y;
    if (rho2 < 1e-12) rho2 = 1e-12;
    double rho = sqrt(rho2);

    double cos_theta = z / r;
    double sin_theta = rho / r;
    double cos_theta2 = cos_theta * cos_theta;
    double sin_theta2 = sin_theta * sin_theta;

    /* Boyer-Lindquist coordinate from quasi-isotropic r.
     * Ref: GRChombo KerrBH.impl.hpp:134 */
    double r_plus = M + sqrt(M * M - a * a);
    double r_minus = M - sqrt(M * M - a * a);
    double r_BL = r * pow(1.0 + 0.25 * r_plus / r, 2.0);

    /* Kerr metric quantities.
     * Ref: arXiv:1001.4077 Eqs. (3)-(6), GRChombo lines 137-143 */
    double Sigma = r_BL * r_BL + a * a * cos_theta2;
    double Delta = r_BL * r_BL - 2.0 * M * r_BL + a * a;
    double AA = (r_BL * r_BL + a * a) * (r_BL * r_BL + a * a)
              - Delta * a * a * sin_theta2;

    /* Spatial metric in spherical QI coords.
     * Ref: GRChombo KerrBH.impl.hpp:147-150 */
    memset(g_sph, 0, sizeof(double) * 9);
    double rp4 = r + 0.25 * r_plus;
    g_sph[0][0] = Sigma * rp4 * rp4 / (r * r2 * (r_BL - r_minus)); /* gamma_rr */
    g_sph[1][1] = Sigma;                                             /* gamma_tt */
    g_sph[2][2] = AA / Sigma * sin_theta2;                           /* gamma_pp */

    /* Extrinsic curvature (only off-diagonal K_rp and K_tp are non-zero).
     * Ref: GRChombo KerrBH.impl.hpp:152-165 */
    memset(K_sph, 0, sizeof(double) * 9);

    double sqrt_AA_Sigma = sqrt(AA * Sigma);
    double r_BL2 = r_BL * r_BL;
    double a2 = a * a;
    double a4 = a2 * a2;

    /* K_rp: Ref GRChombo line 156-160 */
    K_sph[0][2] = a * M * sin_theta2 / (Sigma * sqrt_AA_Sigma)
                * (3.0 * r_BL2 * r_BL2 + 2.0 * a2 * r_BL2 - a4
                   - a2 * (r_BL2 - a2) * sin_theta2)
                * (1.0 + 0.25 * r_plus / r)
                / sqrt(r * r_BL - r * r_minus);
    K_sph[2][0] = K_sph[0][2];

    /* K_tp: Ref GRChombo line 162-164 */
    K_sph[2][1] = -2.0 * a2 * a * M * r_BL * cos_theta * sin_theta
                * sin_theta2 / (Sigma * sqrt_AA_Sigma)
                * (r - 0.25 * r_plus)
                * sqrt(r_BL / r - r_minus / r);
    K_sph[1][2] = K_sph[2][1];
}

/* ================================================================
 * Public API
 * ================================================================ */

void kerr_qi_metric(double h[3][3], double *psi_out,
                    double x, double y, double z,
                    double M, const double spin[3])
{
    double S_mag = sqrt(spin[0] * spin[0] + spin[1] * spin[1]
                      + spin[2] * spin[2]);
    double a = S_mag / M;

    /* Cap spin near extremal to avoid singularity at horizon */
    if (a > 0.998 * M) a = 0.998 * M;

    /* Zero spin: Schwarzschild in isotropic coords.
     * Conformal metric is flat (all curvature is in the conformal factor). */
    if (a < 1e-12) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                h[i][j] = (i == j) ? 1.0 : 0.0;
        if (psi_out) {
            double r = sqrt(x * x + y * y + z * z);
            if (r < 1e-6) r = 1e-6;
            *psi_out = 1.0 + M / (2.0 * r);
        }
        return;
    }

    /* 1. Rotation matrix: spin_dir -> z_axis */
    double spin_dir[3] = { spin[0] / S_mag, spin[1] / S_mag, spin[2] / S_mag };
    double R[3][3];
    rotation_matrix_to_z(spin_dir, R);

    /* 2. Rotate coordinates to spin-aligned frame */
    double xr = R[0][0] * x + R[0][1] * y + R[0][2] * z;
    double yr = R[1][0] * x + R[1][1] * y + R[1][2] * z;
    double zr = R[2][0] * x + R[2][1] * y + R[2][2] * z;

    /* 3. Compute Kerr metric in spherical QI coords */
    double g_sph[3][3], K_sph[3][3];
    compute_kerr_spherical(g_sph, K_sph, xr, yr, zr, M, a);

    /* 4. Spherical -> Cartesian via Jacobian (in rotated frame)
     * g_cart[i][j] = sum_{a,b} jac[a][i] * jac[b][j] * g_sph[a][b]
     * Ref: GRChombo CoordinateTransformations.hpp:144-152 */
    double jac[3][3];
    spherical_jacobian(jac, xr, yr, zr);

    double g_cart[3][3];
    transform_tensor_LL(g_sph, jac, g_cart);

    /* 5. Rotate metric back to original frame.
     * h_orig[i][j] = sum_{a,b} R[a][i] * R[b][j] * g_cart[a][b]
     * Ref: GRChombo KerrBH.impl.hpp:64 */
    double g_orig[3][3];
    transform_tensor_LL(g_cart, R, g_orig);

    /* 6. Extract conformal metric: h_ij = det(g)^{-1/3} * g_ij
     * Ref: GRChombo KerrBH.impl.hpp:69-83 */
    double det = compute_det_sym(g_orig);
    double chi_factor = pow(fabs(det), -1.0 / 3.0);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            h[i][j] = chi_factor * g_orig[i][j];

    /* Conformal factor: psi = det(g)^{1/12} (since gamma = psi^4 * h, det(h)=1
     * => det(gamma) = psi^12 => psi = det(gamma)^{1/12}) */
    if (psi_out)
        *psi_out = pow(fabs(det), 1.0 / 12.0);
}

void kerr_qi_extrinsic(double A[3][3],
                       double x, double y, double z,
                       double M, const double spin[3])
{
    double S_mag = sqrt(spin[0] * spin[0] + spin[1] * spin[1]
                      + spin[2] * spin[2]);
    double a = S_mag / M;
    if (a > 0.998 * M) a = 0.998 * M;

    /* Zero spin: Schwarzschild is time-symmetric, K_ij = 0 on isotropic slice */
    if (a < 1e-12) {
        memset(A, 0, sizeof(double) * 9);
        return;
    }

    /* 1. Rotation */
    double spin_dir[3] = { spin[0] / S_mag, spin[1] / S_mag, spin[2] / S_mag };
    double R[3][3];
    rotation_matrix_to_z(spin_dir, R);

    double xr = R[0][0] * x + R[0][1] * y + R[0][2] * z;
    double yr = R[1][0] * x + R[1][1] * y + R[1][2] * z;
    double zr = R[2][0] * x + R[2][1] * y + R[2][2] * z;

    /* 2. Kerr in spherical */
    double g_sph[3][3], K_sph[3][3];
    compute_kerr_spherical(g_sph, K_sph, xr, yr, zr, M, a);

    /* 3. K_sph -> Cartesian */
    double jac[3][3];
    spherical_jacobian(jac, xr, yr, zr);

    double K_cart[3][3];
    transform_tensor_LL(K_sph, jac, K_cart);

    /* 4. Rotate back */
    double K_orig[3][3];
    transform_tensor_LL(K_cart, R, K_orig);

    /* 5. Extract trace and make trace-free.
     * Need the full metric to compute trace and conformal rescaling.
     * Ref: GRChombo KerrBH.impl.hpp:69-83 */
    double g_cart_tmp[3][3], g_orig[3][3];
    transform_tensor_LL(g_sph, jac, g_cart_tmp);
    transform_tensor_LL(g_cart_tmp, R, g_orig);

    double g_UU[3][3];
    compute_inverse_sym(g_orig, g_UU);

    /* Trace K = g^{ij} K_ij */
    double trK = compute_trace(K_orig, g_UU);

    /* A_ij = K_ij - (1/3) gamma_ij K */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            A[i][j] = K_orig[i][j] - (1.0 / 3.0) * g_orig[i][j] * trK;

    /* 6. York conformal rescaling: A_tilde_ij = psi^2 * A_phys_ij
     * where psi = det(g)^{1/12}.  York weight +2 matches bowen_york_Aij().
     * (Previous code used chi = psi^{-4}, giving CCZ4 weight -4.)
     * Ref: arXiv:1410.8607 Eq. (16), B&S Eq. 3.18 */
    double det = compute_det_sym(g_orig);
    double psi_kerr = pow(fabs(det), 1.0 / 12.0);
    double psi2 = psi_kerr * psi_kerr;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            A[i][j] *= psi2;
}

void hispid_conformal_metric(double h[3][3],
                             double x, double y, double z,
                             int n_bh, const puncture_data_t *bhs)
{
    /* Start with flat metric */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            h[i][j] = (i == j) ? 1.0 : 0.0;

    /* Gaussian-weighted superposition:
     * h_ij(x) = delta_ij + sum_n w_n(x) * [h_ij^Kerr_n(x) - delta_ij]
     * w_n(x) = exp(-r_n^2 / sigma_n^2), sigma_n = 1.5 * M_n
     *
     * Ref: arXiv:1410.8607 Eq. (15) */
    for (int n = 0; n < n_bh; n++) {
        double rx = x - bhs[n].center[0];
        double ry = y - bhs[n].center[1];
        double rz = z - bhs[n].center[2];
        double r2 = rx * rx + ry * ry + rz * rz;

        double sigma = 1.5 * bhs[n].mass;
        double w = exp(-r2 / (sigma * sigma));

        /* Compute single-BH Kerr conformal metric centered at this BH */
        double h_kerr[3][3];
        kerr_qi_metric(h_kerr, NULL, rx, ry, rz,
                       bhs[n].mass, bhs[n].spin);

        /* Blend: h += w * (h_kerr - delta) */
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double delta = (i == j) ? 1.0 : 0.0;
                h[i][j] += w * (h_kerr[i][j] - delta);
            }
    }

    /* Enforce unit determinant: h_ij -> det(h)^{-1/3} * h_ij */
    double det = compute_det_sym(h);
    if (fabs(det) > 1e-30) {
        double scale = pow(fabs(det), -1.0 / 3.0);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                h[i][j] *= scale;
    }
}

void hispid_extrinsic(double A[3][3],
                      double x, double y, double z,
                      int n_bh, const puncture_data_t *bhs)
{
    /* Gaussian-weighted superposition of Kerr extrinsic curvatures.
     * A_ij(x) = sum_n w_n(x) * A_ij^Kerr_n(x)
     *
     * Far from all BHs (w_n ~ 0), A_ij -> 0.
     * This is supplemented by the Bowen-York A_ij for momentum,
     * handled at a higher level in set_ccz4_from_hispid.
     *
     * Ref: arXiv:1410.8607 Eq. (16) */
    memset(A, 0, sizeof(double) * 9);

    for (int n = 0; n < n_bh; n++) {
        double rx = x - bhs[n].center[0];
        double ry = y - bhs[n].center[1];
        double rz = z - bhs[n].center[2];
        double r2 = rx * rx + ry * ry + rz * rz;

        double sigma = 1.5 * bhs[n].mass;
        double w = exp(-r2 / (sigma * sigma));

        double A_kerr[3][3];
        kerr_qi_extrinsic(A_kerr, rx, ry, rz,
                          bhs[n].mass, bhs[n].spin);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                A[i][j] += w * A_kerr[i][j];
    }
}
