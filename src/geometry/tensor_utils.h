/*
 * tensor_utils.h — Symmetric 3x3 tensor utilities (header-only)
 *
 * All symmetric tensors are stored as flat arrays in order:
 *   [xx, xy, xz, yy, yz, zz] (indices 0-5)
 *
 * Convention: _dd suffix = covariant (lower), _uu suffix = contravariant (upper)
 */

#ifndef LATTICE_TENSOR_UTILS_H
#define LATTICE_TENSOR_UTILS_H

#include "../core/fields.h"
#include <math.h>

/*
 * Determinant of a symmetric 3x3 matrix.
 * det = a00(a11*a22 - a12^2) - a01(a01*a22 - a02*a12) + a02(a01*a12 - a02*a11)
 */
static inline double sym3_det(const double s[6])
{
    return s[SYM_XX] * (s[SYM_YY] * s[SYM_ZZ] - s[SYM_YZ] * s[SYM_YZ])
         - s[SYM_XY] * (s[SYM_XY] * s[SYM_ZZ] - s[SYM_XZ] * s[SYM_YZ])
         + s[SYM_XZ] * (s[SYM_XY] * s[SYM_YZ] - s[SYM_XZ] * s[SYM_YY]);
}

/*
 * Inverse of a symmetric 3x3 matrix: inv[6] = s^{-1}.
 * Returns determinant. Caller should check det != 0.
 */
static inline double sym3_inv(const double s[6], double inv[6])
{
    double det = sym3_det(s);
    double inv_det = 1.0 / det;

    inv[SYM_XX] = (s[SYM_YY] * s[SYM_ZZ] - s[SYM_YZ] * s[SYM_YZ]) * inv_det;
    inv[SYM_XY] = (s[SYM_XZ] * s[SYM_YZ] - s[SYM_XY] * s[SYM_ZZ]) * inv_det;
    inv[SYM_XZ] = (s[SYM_XY] * s[SYM_YZ] - s[SYM_XZ] * s[SYM_YY]) * inv_det;
    inv[SYM_YY] = (s[SYM_XX] * s[SYM_ZZ] - s[SYM_XZ] * s[SYM_XZ]) * inv_det;
    inv[SYM_YZ] = (s[SYM_XY] * s[SYM_XZ] - s[SYM_XX] * s[SYM_YZ]) * inv_det;
    inv[SYM_ZZ] = (s[SYM_XX] * s[SYM_YY] - s[SYM_XY] * s[SYM_XY]) * inv_det;

    return det;
}

/*
 * Trace of a symmetric tensor with respect to an inverse metric:
 *   tr = g^{ij} s_{ij} = gtu[0]*s[0] + gtu[3]*s[3] + gtu[5]*s[5]
 *        + 2*(gtu[1]*s[1] + gtu[2]*s[2] + gtu[4]*s[4])
 */
static inline double sym3_trace(const double s[6], const double gtu[6])
{
    return gtu[SYM_XX] * s[SYM_XX] + gtu[SYM_YY] * s[SYM_YY] + gtu[SYM_ZZ] * s[SYM_ZZ]
         + 2.0 * (gtu[SYM_XY] * s[SYM_XY] + gtu[SYM_XZ] * s[SYM_XZ]
                  + gtu[SYM_YZ] * s[SYM_YZ]);
}

/*
 * Make a symmetric tensor tracefree with respect to a metric:
 *   s_ij -> s_ij - (1/3) g_ij g^{kl} s_{kl}
 */
static inline void sym3_make_tracefree(double s[6], const double g_dd[6],
                                       const double g_uu[6])
{
    double tr = sym3_trace(s, g_uu);
    double third_tr = tr / 3.0;
    for (int a = 0; a < 6; a++) {
        s[a] -= third_tr * g_dd[a];
    }
}

/*
 * Raise a covector v_i with inverse metric: v^i = g^{ij} v_j
 */
static inline void sym3_raise_vector(const double gtu[6], const double v_d[3],
                                     double v_u[3])
{
    v_u[0] = gtu[SYM_XX] * v_d[0] + gtu[SYM_XY] * v_d[1] + gtu[SYM_XZ] * v_d[2];
    v_u[1] = gtu[SYM_XY] * v_d[0] + gtu[SYM_YY] * v_d[1] + gtu[SYM_YZ] * v_d[2];
    v_u[2] = gtu[SYM_XZ] * v_d[0] + gtu[SYM_YZ] * v_d[1] + gtu[SYM_ZZ] * v_d[2];
}

/*
 * Contract two symmetric tensors: result = a^{ij} b_{ij}
 * (same as trace of product when one index is raised)
 */
static inline double sym3_contract(const double a[6], const double b[6])
{
    return a[SYM_XX] * b[SYM_XX] + a[SYM_YY] * b[SYM_YY] + a[SYM_ZZ] * b[SYM_ZZ]
         + 2.0 * (a[SYM_XY] * b[SYM_XY] + a[SYM_XZ] * b[SYM_XZ]
                  + a[SYM_YZ] * b[SYM_YZ]);
}

/*
 * Raise first index of a symmetric tensor: A^i_j = g^{ik} A_{kj}
 * Result is NOT symmetric — stored as [3][3] = result[i][j]
 */
static inline void sym3_raise_first(const double gtu[6], const double a_dd[6],
                                    double a_ud[3][3])
{
    /* i=0: A^x_j = g^{xk} A_{kj} */
    a_ud[0][0] = gtu[SYM_XX] * a_dd[SYM_XX] + gtu[SYM_XY] * a_dd[SYM_XY]
               + gtu[SYM_XZ] * a_dd[SYM_XZ];
    a_ud[0][1] = gtu[SYM_XX] * a_dd[SYM_XY] + gtu[SYM_XY] * a_dd[SYM_YY]
               + gtu[SYM_XZ] * a_dd[SYM_YZ];
    a_ud[0][2] = gtu[SYM_XX] * a_dd[SYM_XZ] + gtu[SYM_XY] * a_dd[SYM_YZ]
               + gtu[SYM_XZ] * a_dd[SYM_ZZ];

    /* i=1: A^y_j = g^{yk} A_{kj} */
    a_ud[1][0] = gtu[SYM_XY] * a_dd[SYM_XX] + gtu[SYM_YY] * a_dd[SYM_XY]
               + gtu[SYM_YZ] * a_dd[SYM_XZ];
    a_ud[1][1] = gtu[SYM_XY] * a_dd[SYM_XY] + gtu[SYM_YY] * a_dd[SYM_YY]
               + gtu[SYM_YZ] * a_dd[SYM_YZ];
    a_ud[1][2] = gtu[SYM_XY] * a_dd[SYM_XZ] + gtu[SYM_YY] * a_dd[SYM_YZ]
               + gtu[SYM_YZ] * a_dd[SYM_ZZ];

    /* i=2: A^z_j = g^{zk} A_{kj} */
    a_ud[2][0] = gtu[SYM_XZ] * a_dd[SYM_XX] + gtu[SYM_YZ] * a_dd[SYM_XY]
               + gtu[SYM_ZZ] * a_dd[SYM_XZ];
    a_ud[2][1] = gtu[SYM_XZ] * a_dd[SYM_XY] + gtu[SYM_YZ] * a_dd[SYM_YY]
               + gtu[SYM_ZZ] * a_dd[SYM_YZ];
    a_ud[2][2] = gtu[SYM_XZ] * a_dd[SYM_XZ] + gtu[SYM_YZ] * a_dd[SYM_YZ]
               + gtu[SYM_ZZ] * a_dd[SYM_ZZ];
}

/*
 * Enforce unit determinant: rescale g_ij -> g_ij / cbrt(det(g))
 * Used after every RK4 step to maintain det(gamma_tilde) = 1.
 */
static inline void sym3_enforce_unit_det(double s[6])
{
    double det = sym3_det(s);
    double scale = 1.0 / cbrt(det);
    for (int a = 0; a < 6; a++) {
        s[a] *= scale;
    }
}

#endif /* LATTICE_TENSOR_UTILS_H */
