/*
 * Lattice — 3D Numerical Relativity
 * Sommerfeld radiative boundary conditions.
 *
 * For each ghost zone point, the RHS is set to enforce:
 *   d_t f = -x^i/r * d_i f + (f_asymptotic - f) / r
 *
 * This permits outgoing waves to leave the domain with minimal reflection.
 *
 * Ref: GRChombo BoundaryConditions.cpp:593-661
 */

#include "sommerfeld.h"
#include "../amr/block.h"
#include "../core/fields.h"
#include <math.h>

/* Asymptotic values for each field.
 * Non-static: exposed in sommerfeld.h for packed GPU kernels.
 * Ref: GRChombo BoundaryConditions.cpp:593-598 */
/* Array lookup: branchless, eliminates GPU warp divergence.
 * Fields with flat-space value 1: chi, h_11, h_22, h_33, lapse.
 * All others (K, A_ij, Theta, Gamma^i, shift, B^i, E^i, BM^i) → 0.
 *
 * C++ (hipcc) doesn't support C99 designated initializers, so we use
 * positional initialization for C++ builds. Both produce the same array. */
#ifdef __cplusplus
static const double asym_values[NUM_FIELDS] = {
    /* chi */ 1.0,
    /* h11 */ 1.0, /* h12 */ 0.0, /* h13 */ 0.0,
    /* h22 */ 1.0, /* h23 */ 0.0, /* h33 */ 1.0,
    /* K   */ 0.0,
    /* A11 */ 0.0, /* A12 */ 0.0, /* A13 */ 0.0,
    /* A22 */ 0.0, /* A23 */ 0.0, /* A33 */ 0.0,
    /* Theta */ 0.0,
    /* Gamma1 */ 0.0, /* Gamma2 */ 0.0, /* Gamma3 */ 0.0,
    /* lapse */ 1.0,
    /* shift1 */ 0.0, /* shift2 */ 0.0, /* shift3 */ 0.0,
    /* B1 */ 0.0, /* B2 */ 0.0, /* B3 */ 0.0,
    /* E1 */ 0.0, /* E2 */ 0.0, /* E3 */ 0.0,
    /* BM1 */ 0.0, /* BM2 */ 0.0, /* BM3 */ 0.0,
};
#else
static const double asym_values[NUM_FIELDS] = {
    [FIELD_CHI]   = 1.0,
    [FIELD_H11]   = 1.0, [FIELD_H22] = 1.0, [FIELD_H33] = 1.0,
    [FIELD_LAPSE] = 1.0
};
#endif

LATTICE_DEVICE
double asymptotic_value(int field)
{
    return asym_values[field];
}

/*
 * 4th-order derivative at boundary using adaptive stencils.
 * Non-static: exposed in sommerfeld.h for packed GPU kernels.
 *
 * Five stencil variants covering all offset combinations:
 *   Centered:        [1/12, -2/3, 0, 2/3, -1/12] / dx       (lo>=2, hi>=2)
 *   Forward:         [-25/12, 4, -3, 4/3, -1/4] / dx         (lo<1, hi>=4)
 *   Backward:        [25/12, -4, 3, -4/3, 1/4] / dx          (hi<1, lo>=4)
 *   Forward-biased:  [-1/4, -5/6, 3/2, -1/2, 1/12] / dx     (lo>=1, hi>=3)
 *   Backward-biased: mirror of above                          (hi>=1, lo>=3)
 *
 * With ghost width = 4, lo_offset + hi_offset = Ntotal - 1 >= 8, so all
 * boundary ghost cells are covered by one of the five 4th-order stencils.
 *
 * Ref: Fornberg, SIAM Review 40 (1998) — FD weight generation algorithm
 * Ref: GRChombo BoundaryConditions.cpp:617-649 (original 2nd-order)
 */
LATTICE_DEVICE
double boundary_d1(const double *f, int idx, int stride,
                   int lo_offset, int hi_offset, double dx)
{
    if (lo_offset >= 2 && hi_offset >= 2) {
        /* Centered — 4th-order (5-point): same as fd_d1 at 4th order */
        return (  (1.0 / 12.0) * f[idx - 2*stride]
                - (2.0 /  3.0) * f[idx -   stride]
                + (2.0 /  3.0) * f[idx +   stride]
                - (1.0 / 12.0) * f[idx + 2*stride] ) / dx;
    } else if (lo_offset < 1 && hi_offset >= 4) {
        /* Near low boundary — 4th-order forward stencil (needs 4 points ahead) */
        return (-25.0/12.0 * f[idx]
                + 4.0      * f[idx +   stride]
                - 3.0      * f[idx + 2*stride]
                + 4.0/3.0  * f[idx + 3*stride]
                - 1.0/4.0  * f[idx + 4*stride]) / dx;
    } else if (hi_offset < 1 && lo_offset >= 4) {
        /* Near high boundary — 4th-order backward stencil */
        return ( 25.0/12.0 * f[idx]
                - 4.0      * f[idx -   stride]
                + 3.0      * f[idx - 2*stride]
                - 4.0/3.0  * f[idx - 3*stride]
                + 1.0/4.0  * f[idx - 4*stride]) / dx;
    } else if (lo_offset >= 1 && hi_offset >= 3) {
        /* Near low boundary — 4th-order forward-biased (nodes -1,0,+1,+2,+3).
         * Ref: Fornberg, SIAM Review 40 (1998) */
        return (-1.0/4.0  * f[idx - stride]
                - 5.0/6.0  * f[idx]
                + 3.0/2.0  * f[idx + stride]
                - 1.0/2.0  * f[idx + 2*stride]
                + 1.0/12.0 * f[idx + 3*stride]) / dx;
    } else if (hi_offset >= 1 && lo_offset >= 3) {
        /* Near high boundary — 4th-order backward-biased (nodes -3,-2,-1,0,+1).
         * Ref: Fornberg, SIAM Review 40 (1998) */
        return ( 1.0/4.0  * f[idx + stride]
                + 5.0/6.0  * f[idx]
                - 3.0/2.0  * f[idx - stride]
                + 1.0/2.0  * f[idx - 2*stride]
                - 1.0/12.0 * f[idx - 3*stride]) / dx;
    } else {
        /* Fallback: centered 2nd-order (should not be reached with ghost >= 4) */
        return 0.5 * (f[idx + stride] - f[idx - stride]) / dx;
    }
}

