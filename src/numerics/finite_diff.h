/*
 * Lattice — 3D Numerical Relativity
 * 4th-order finite difference stencils (all static inline).
 *
 * Stencil coefficients from GRChombo FourthOrderDerivatives.hpp.
 * All derivatives go through these functions — no hand-coded stencils.
 *
 * Ref: GRChombo Source/BoxUtils/FourthOrderDerivatives.hpp
 */

#ifndef LATTICE_FINITE_DIFF_H
#define LATTICE_FINITE_DIFF_H

#include <math.h>

#ifdef LATTICE_GPU
#pragma omp declare target
#endif

/*
 * 4th-order first derivative: d f / d x
 * f: field array, idx: flat index, s: stride for direction, dx: spacing
 */
static inline double fd_d1(const double *f, int idx, int s, double dx)
{
    /* coefficients: 1/12, 2/3 — GRChombo lines 36-46 */
    return (  (1.0 / 12.0) * f[idx - 2*s]
            - (2.0 /  3.0) * f[idx -   s]
            + (2.0 /  3.0) * f[idx +   s]
            - (1.0 / 12.0) * f[idx + 2*s] ) / dx;
}

/*
 * 4th-order second derivative: d^2 f / d x^2
 * Ref: GRChombo lines 114-128
 */
static inline double fd_d2(const double *f, int idx, int s, double dx)
{
    double dx2 = dx * dx;
    return ( -(1.0 / 12.0) * f[idx - 2*s]
             + (4.0 /  3.0) * f[idx -   s]
             - (5.0 /  2.0) * f[idx]
             + (4.0 /  3.0) * f[idx +   s]
             - (1.0 / 12.0) * f[idx + 2*s] ) / dx2;
}

/*
 * 4th-order mixed second derivative: d^2 f / (d x_a d x_b)
 * s1, s2: strides for the two directions
 * Ref: GRChombo mixed_diff2, lines 163-192
 */
static inline double fd_d2_mixed(const double *f, int idx, int s1, int s2,
                                 double dx)
{
    double dx2 = dx * dx;

    double wff = 1.0 / 144.0;   /* 6.944e-3 = (1/12)^2           */
    double wnf =  1.0 / 18.0;   /* 5.556e-2 = (1/12)*(2/3)       */
    double wnn =  4.0 /  9.0;   /* 4.444e-1 = (2/3)^2            */

    return ( wff * f[idx - 2*s1 - 2*s2]
           - wnf * f[idx - 2*s1 -   s2]
           + wnf * f[idx - 2*s1 +   s2]
           - wff * f[idx - 2*s1 + 2*s2]

           - wnf * f[idx -   s1 - 2*s2]
           + wnn * f[idx -   s1 -   s2]
           - wnn * f[idx -   s1 +   s2]
           + wnf * f[idx -   s1 + 2*s2]

           + wnf * f[idx +   s1 - 2*s2]
           - wnn * f[idx +   s1 -   s2]
           + wnn * f[idx +   s1 +   s2]
           - wnf * f[idx +   s1 + 2*s2]

           - wff * f[idx + 2*s1 - 2*s2]
           + wnf * f[idx + 2*s1 -   s2]
           - wnf * f[idx + 2*s1 +   s2]
           + wff * f[idx + 2*s1 + 2*s2] ) / dx2;
}

/*
 * 4th-order upwind advection derivative: beta^a d_a f
 * vel: velocity component in this direction (determines upwind side)
 * Ref: GRChombo advection_term, lines 269-301
 *
 * Upwind stencil (vel > 0):   w0*f[i-1] + w1*f[i] + w2*f[i+1] + w3*f[i+2] + w4*f[i+3]
 * Downwind stencil (vel < 0): mirror of upwind
 */
static inline double fd_adv(const double *f, int idx, int s, double vel,
                            double dx)
{
    double w0 = -1.0 / 4.0;
    double w1 = -5.0 / 6.0;
    double w2 = +3.0 / 2.0;
    double w3 = -1.0 / 2.0;
    double w4 = +1.0 / 12.0;

    if (vel > 0.0) {
        return vel * ( w0 * f[idx -   s]
                     + w1 * f[idx]
                     + w2 * f[idx +   s]
                     + w3 * f[idx + 2*s]
                     + w4 * f[idx + 3*s] ) / dx;
    } else {
        return vel * (-w4 * f[idx - 3*s]
                     - w3 * f[idx - 2*s]
                     - w2 * f[idx -   s]
                     - w1 * f[idx]
                     - w0 * f[idx +   s] ) / dx;
    }
}

/*
 * 6th-order Kreiss-Oliger dissipation operator.
 * Returns the dissipation term for one direction; caller sums over directions
 * and multiplies by sigma.
 *
 * Stencil: 7 points, divided by dx (not dx^6) following GRChombo convention.
 * Ref: GRChombo dissipation_term, lines 361-378
 */
static inline double fd_ko(const double *f, int idx, int s, double dx)
{
    double wvf = 1.0 / 64.0;     /* 1.5625e-2 */
    double wf  = 6.0 / 64.0;     /* 9.375e-2  */
    double wn  = 15.0 / 64.0;    /* 2.34375e-1 */
    double wl  = 20.0 / 64.0;    /* 3.125e-1  */

    return ( wvf * f[idx - 3*s]
           - wf  * f[idx - 2*s]
           + wn  * f[idx -   s]
           - wl  * f[idx]
           + wn  * f[idx +   s]
           - wf  * f[idx + 2*s]
           + wvf * f[idx + 3*s] ) / dx;
}

#ifdef LATTICE_GPU
#pragma omp end declare target
#endif

#endif /* LATTICE_FINITE_DIFF_H */
