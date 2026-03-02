/*
 * Lattice — 3D Numerical Relativity
 * Inline tensor algebra utilities.
 *
 * Ref: GRChombo Source/utils/TensorAlgebra.hpp
 * All functions operate on double[3][3] or double[3] arrays.
 */

#ifndef LATTICE_TENSOR_UTILS_H
#define LATTICE_TENSOR_UTILS_H

#include "../core/device.h"
#include "../core/fields.h"
#include <math.h>

/*
 * Christoffel symbol data: Gamma^i_{jk}, Gamma_{ijk}, Gamma^i (contracted)
 * Ref: GRChombo TensorAlgebra.hpp chris_t struct
 */
typedef struct {
    double ULL[3][3][3];     /* Gamma^i_{jk}           */
    double LLL[3][3][3];     /* Gamma_{ijk}            */
    double contracted[3];    /* Gamma^i = h^{jk} Gamma^i_{jk} */
} chris_t;

/* Ricci tensor data */
typedef struct {
    double LL[3][3];   /* R_{ij}  */
    double scalar;     /* R = h^{ij} R_{ij} (conformal weighted) */
} ricci_t;

/*
 * Determinant of symmetric 3x3 matrix.
 * Ref: GRChombo TensorAlgebra.hpp:55-63
 */
LATTICE_DEVICE
static inline double compute_det_sym(const double h[3][3])
{
    return h[0][0] * h[1][1] * h[2][2]
         + 2.0 * h[0][1] * h[0][2] * h[1][2]
         - h[0][0] * h[1][2] * h[1][2]
         - h[1][1] * h[0][2] * h[0][2]
         - h[2][2] * h[0][1] * h[0][1];
}

/*
 * Inverse of symmetric 3x3 matrix.
 * Ref: GRChombo TensorAlgebra.hpp:77-99
 */
LATTICE_DEVICE
static inline void compute_inverse_sym(const double h[3][3],
                                       double h_UU[3][3])
{
    double det = compute_det_sym(h);
    double inv_det = 1.0 / det;

    h_UU[0][0] = (h[1][1] * h[2][2] - h[1][2] * h[1][2]) * inv_det;
    h_UU[0][1] = (h[0][2] * h[1][2] - h[0][1] * h[2][2]) * inv_det;
    h_UU[0][2] = (h[0][1] * h[1][2] - h[0][2] * h[1][1]) * inv_det;
    h_UU[1][1] = (h[0][0] * h[2][2] - h[0][2] * h[0][2]) * inv_det;
    h_UU[1][2] = (h[0][1] * h[0][2] - h[0][0] * h[1][2]) * inv_det;
    h_UU[2][2] = (h[0][0] * h[1][1] - h[0][1] * h[0][1]) * inv_det;

    h_UU[1][0] = h_UU[0][1];
    h_UU[2][0] = h_UU[0][2];
    h_UU[2][1] = h_UU[1][2];
}

/*
 * Trace of A_{ij} with inverse metric: tr = h^{ij} A_{ij}
 * Ref: GRChombo TensorAlgebra.hpp:165-171
 */
LATTICE_DEVICE
static inline double compute_trace(const double A[3][3],
                                   const double h_UU[3][3])
{
    double tr = 0.0;
    FOR2(i, j) tr += h_UU[i][j] * A[i][j];
    return tr;
}

/*
 * Trace of a mixed tensor A^i_j (diagonal sum).
 * Ref: GRChombo TensorAlgebra.hpp:175-179
 */
LATTICE_DEVICE
static inline double compute_trace_diag(const double A[3][3])
{
    return A[0][0] + A[1][1] + A[2][2];
}

/*
 * Dot product of two vectors: v^i w_i (no metric).
 * Ref: GRChombo TensorAlgebra.hpp:193-199
 */
LATTICE_DEVICE
static inline double compute_dot_product(const double v[3], const double w[3])
{
    return v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
}

/*
 * Dot product with inverse metric: h^{ij} v_i w_j
 * Ref: GRChombo TensorAlgebra.hpp:204-214
 */
LATTICE_DEVICE
static inline double compute_dot_product_metric(const double v[3],
                                                const double w[3],
                                                const double h_UU[3][3])
{
    double dp = 0.0;
    FOR2(m, n) dp += h_UU[m][n] * v[m] * w[n];
    return dp;
}

/*
 * Compute Christoffel symbols from d1_h[i][j][k] = d_k h_{ij}
 * and the inverse metric h_UU.
 *
 * Gamma_{ijk} = 0.5*(d_k h_{ji} + d_j h_{ki} - d_i h_{jk})
 * Gamma^i_{jk} = h^{il} Gamma_{ljk}
 * Gamma^i = h^{jk} Gamma^i_{jk}
 *
 * Ref: GRChombo TensorAlgebra.hpp:344-367
 */
LATTICE_DEVICE
static inline void compute_christoffel(const double d1_h[3][3][3],
                                       const double h_UU[3][3],
                                       chris_t *chris)
{
    /* Gamma_{ijk} = 0.5*(d1_h[j][i][k] + d1_h[k][i][j] - d1_h[j][k][i])
     * where d1_h[a][b][dir] = d_{dir} h_{ab} */
    FOR3(i, j, k) {
        chris->LLL[i][j][k] = 0.5 * (d1_h[j][i][k] + d1_h[k][i][j]
                                    - d1_h[j][k][i]);
    }

    FOR3(i, j, k) {
        chris->ULL[i][j][k] = 0.0;
        FOR1(l) chris->ULL[i][j][k] += h_UU[i][l] * chris->LLL[l][j][k];
    }

    FOR1(i) {
        chris->contracted[i] = 0.0;
        FOR2(j, k) chris->contracted[i] += h_UU[j][k] * chris->ULL[i][j][k];
    }
}

/*
 * Raise both indices of a symmetric 2-tensor: A^{ij} = h^{ik} h^{jl} A_{kl}
 * Ref: GRChombo TensorAlgebra.hpp:259-270
 */
LATTICE_DEVICE
static inline void raise_all_2(const double A_dd[3][3],
                                const double h_UU[3][3],
                                double A_uu[3][3])
{
    /* Exploit symmetry: A_uu[i][j] == A_uu[j][i] since A_dd is symmetric.
     * Compute upper triangle (i <= j), mirror lower. Saves 27 FMAs/call.
     * Ref: same pattern as compute_inverse_sym() above. */
    for (int i = 0; i < 3; i++) {
        for (int j = i; j < 3; j++) {
            A_uu[i][j] = 0.0;
            FOR2(k, l) A_uu[i][j] += h_UU[i][k] * h_UU[j][l] * A_dd[k][l];
            A_uu[j][i] = A_uu[i][j];
        }
    }
}

/*
 * Remove trace from symmetric tensor: A_{ij} -= (1/3) h_{ij} h^{kl} A_{kl}
 * Ref: GRChombo TensorAlgebra.hpp:220-230
 */
LATTICE_DEVICE
static inline void make_trace_free(double A[3][3],
                                   const double h[3][3],
                                   const double h_UU[3][3])
{
    double tr = compute_trace(A, h_UU);
    double one_over_dim = 1.0 / (double)GR_SPACEDIM;
    FOR2(i, j) A[i][j] -= one_over_dim * h[i][j] * tr;
}

/*
 * Fast inverse cube root for det ≈ 1: two Newton-Raphson iterations
 * starting from linear approximation 1/cbrt(1+e) ≈ 1 - e/3.
 * Falls back to cbrt() for det outside [0.5, 2.0].
 * ~3-5x faster than 1/cbrt(det) for the common case.
 */
LATTICE_DEVICE
static inline double fast_inv_cbrt(double det)
{
    if (det < 0.5 || det > 2.0)
        return 1.0 / cbrt(det);

    /* Linear seed: 1/cbrt(1+e) ≈ 1 - e/3 */
    double s = 1.0 + (1.0 - det) / 3.0;

    /* Newton for f(s) = s^3 * det - 1 = 0 → s = s * (4 - det*s^3) / 3 */
    double s3 = s * s * s;
    s = s * (4.0 - det * s3) / 3.0;
    s3 = s * s * s;
    s = s * (4.0 - det * s3) / 3.0;

    return s;
}

#endif /* LATTICE_TENSOR_UTILS_H */
