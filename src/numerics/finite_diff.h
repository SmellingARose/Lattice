/*
 * Lattice — 3D Numerical Relativity
 * Finite difference stencils (all static inline).
 *
 * FD order controlled by FD_ORDER macro:
 *   FD_ORDER=4: 4th-order FD (5-point stencils, ghost >= 2)
 *   FD_ORDER=6: 6th-order FD (7-point stencils, ghost >= 3)  [default]
 *
 * KO dissipation order controlled by KO_ORDER macro (decoupled from FD):
 *   KO_ORDER=6: 6th-order KO (7-point, half-width 3, ghost margin=1)  [default]
 *   KO_ORDER=8: 8th-order KO (9-point, half-width 4, ghost margin=0)
 *
 * All derivatives go through these functions — no hand-coded stencils.
 * Ghost width = 4 supports all combinations (6th FD needs 3, advection needs 4).
 *
 * Ref: GRChombo Source/BoxUtils/FourthOrderDerivatives.hpp (4th-order)
 * Ref: GRChombo Source/BoxUtils/SixthOrderDerivatives.hpp (6th-order)
 * Ref: Fornberg, SIAM Review 40 (1998) — finite difference weight tables
 */

#ifndef LATTICE_FINITE_DIFF_H
#define LATTICE_FINITE_DIFF_H

#include "../core/device.h"
#include <math.h>

/* Default to 6th-order if not specified */
#ifndef FD_ORDER
#define FD_ORDER 6
#endif

#if FD_ORDER == 6

/* ========================================================================
 * 6th-order finite differences (7-point stencils, ghost >= 3)
 * Ref: GRChombo SixthOrderDerivatives.hpp
 * ======================================================================== */

/*
 * 6th-order first derivative: d f / d x
 * 7-point stencil: [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60] * inv_dx
 * Ref: GRChombo SixthOrderDerivatives.hpp lines 36-49
 * Ref: Fornberg table, 6th-order centered d1
 */
LATTICE_DEVICE
static inline double fd_d1(const double *f, int idx, int s, double inv_dx)
{
    double wvf = 1.0 / 60.0;    /* 1.6667e-2 */
    double wf  = 3.0 / 20.0;    /* 1.5000e-1 */
    double wn  = 3.0 /  4.0;    /* 7.5000e-1 */

    return (-wvf * f[idx - 3*s]
            + wf * f[idx - 2*s]
            - wn * f[idx -   s]
            + wn * f[idx +   s]
            - wf * f[idx + 2*s]
            +wvf * f[idx + 3*s]) * inv_dx;
}

/*
 * 6th-order second derivative: d^2 f / d x^2
 * 7-point stencil: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90] * inv_dx^2
 * Ref: GRChombo SixthOrderDerivatives.hpp lines 117-133
 * Ref: Fornberg table, 6th-order centered d2
 */
LATTICE_DEVICE
static inline double fd_d2(const double *f, int idx, int s, double inv_dx)
{
    double inv_dx2 = inv_dx * inv_dx;
    double wvf = 1.0 / 90.0;         /* 1.1111e-2 */
    double wf  = 3.0 / 20.0;         /* 1.5000e-1 */
    double wn  = 3.0 /  2.0;         /* 1.5000e+0 */
    double wl  = 49.0 / 18.0;        /* 2.7222e+0 */

    return ( wvf * f[idx - 3*s]
           - wf  * f[idx - 2*s]
           + wn  * f[idx -   s]
           - wl  * f[idx]
           + wn  * f[idx +   s]
           - wf  * f[idx + 2*s]
           + wvf * f[idx + 3*s]) * inv_dx2;
}

/*
 * 6th-order fused first + second derivative (diagonal d2 only).
 * Loads 7 stencil points once, computes both d1 and d2, saving ~40%
 * memory loads for fields needing both derivatives.
 * Ref: Fornberg table, 6th-order centered d1 + d2
 */
LATTICE_DEVICE
static inline void fd_d1_d2(const double *f, int idx, int s, double inv_dx,
                            double *out_d1, double *out_d2)
{
    double fm3 = f[idx - 3*s], fm2 = f[idx - 2*s], fm1 = f[idx - s];
    double f0  = f[idx];
    double fp1 = f[idx + s],   fp2 = f[idx + 2*s], fp3 = f[idx + 3*s];

    /* d1: [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60] * inv_dx */
    *out_d1 = (-(1.0/60.0)*fm3 + (3.0/20.0)*fm2 - (3.0/4.0)*fm1
               + (3.0/4.0)*fp1 - (3.0/20.0)*fp2 + (1.0/60.0)*fp3) * inv_dx;

    /* d2: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90] * inv_dx^2 */
    double inv_dx2 = inv_dx * inv_dx;
    *out_d2 = ((1.0/90.0)*fm3 - (3.0/20.0)*fm2 + (3.0/2.0)*fm1
              - (49.0/18.0)*f0
              + (3.0/2.0)*fp1 - (3.0/20.0)*fp2 + (1.0/90.0)*fp3) * inv_dx2;
}

/*
 * 6th-order mixed second derivative: d^2 f / (d x_a d x_b)
 * 7×7 tensor product of 6th-order d1 weights.
 * 6 distinct weight magnitudes from products of {1/60, 3/20, 3/4}.
 * Ref: GRChombo SixthOrderDerivatives.hpp lines 169-223
 */
LATTICE_DEVICE
static inline double fd_d2_mixed(const double *f, int idx, int s1, int s2,
                                 double inv_dx)
{
    double inv_dx2 = inv_dx * inv_dx;

    /* Products of d1 weights: (1/60)^2, (1/60)(3/20), (1/60)(3/4),
     *                         (3/20)^2, (3/20)(3/4), (3/4)^2 */
    double wvv = 1.0 / 3600.0;       /* 2.7778e-4 = (1/60)^2       */
    double wvf = 1.0 /  400.0;       /* 2.5000e-3 = (1/60)(3/20)   */
    double wvn = 1.0 /   80.0;       /* 1.2500e-2 = (1/60)(3/4)    */
    double wff = 9.0 /  400.0;       /* 2.2500e-2 = (3/20)^2       */
    double wfn = 9.0 /   80.0;       /* 1.1250e-1 = (3/20)(3/4)    */
    double wnn = 9.0 /   16.0;       /* 5.6250e-1 = (3/4)^2        */

    return (
        /* row s1 = -3 */
          wvv * f[idx - 3*s1 - 3*s2]
        - wvf * f[idx - 3*s1 - 2*s2]
        + wvn * f[idx - 3*s1 -   s2]
        - wvn * f[idx - 3*s1 +   s2]
        + wvf * f[idx - 3*s1 + 2*s2]
        - wvv * f[idx - 3*s1 + 3*s2]

        /* row s1 = -2 */
        - wvf * f[idx - 2*s1 - 3*s2]
        + wff * f[idx - 2*s1 - 2*s2]
        - wfn * f[idx - 2*s1 -   s2]
        + wfn * f[idx - 2*s1 +   s2]
        - wff * f[idx - 2*s1 + 2*s2]
        + wvf * f[idx - 2*s1 + 3*s2]

        /* row s1 = -1 */
        + wvn * f[idx -   s1 - 3*s2]
        - wfn * f[idx -   s1 - 2*s2]
        + wnn * f[idx -   s1 -   s2]
        - wnn * f[idx -   s1 +   s2]
        + wfn * f[idx -   s1 + 2*s2]
        - wvn * f[idx -   s1 + 3*s2]

        /* row s1 = +1 */
        - wvn * f[idx +   s1 - 3*s2]
        + wfn * f[idx +   s1 - 2*s2]
        - wnn * f[idx +   s1 -   s2]
        + wnn * f[idx +   s1 +   s2]
        - wfn * f[idx +   s1 + 2*s2]
        + wvn * f[idx +   s1 + 3*s2]

        /* row s1 = +2 */
        + wvf * f[idx + 2*s1 - 3*s2]
        - wff * f[idx + 2*s1 - 2*s2]
        + wfn * f[idx + 2*s1 -   s2]
        - wfn * f[idx + 2*s1 +   s2]
        + wff * f[idx + 2*s1 + 2*s2]
        - wvf * f[idx + 2*s1 + 3*s2]

        /* row s1 = +3 */
        - wvv * f[idx + 3*s1 - 3*s2]
        + wvf * f[idx + 3*s1 - 2*s2]
        - wvn * f[idx + 3*s1 -   s2]
        + wvn * f[idx + 3*s1 +   s2]
        - wvf * f[idx + 3*s1 + 2*s2]
        + wvv * f[idx + 3*s1 + 3*s2]
    ) * inv_dx2;
}

/*
 * 6th-order upwind advection stencil (branchless variants).
 * 7-point upwind stencil biased in flow direction.
 * fd_adv_up: vel > 0 (forward-biased), fd_adv_down: vel <= 0 (backward-biased).
 * fd_adv: branching wrapper (legacy, use hoisted variants in hot loops).
 * Requires ghost >= 4. Ref: GRChombo SixthOrderDerivatives.hpp lines 300-337
 */
LATTICE_DEVICE
static inline double fd_adv_up(const double *f, int idx, int s, double inv_dx)
{
    return ( ( 1.0/30.0) * f[idx - 2*s]
           + (-2.0/ 5.0) * f[idx -   s]
           + (-7.0/12.0) * f[idx]
           + ( 4.0/ 3.0) * f[idx +   s]
           + (-1.0/ 2.0) * f[idx + 2*s]
           + ( 2.0/15.0) * f[idx + 3*s]
           + (-1.0/60.0) * f[idx + 4*s] ) * inv_dx;
}

LATTICE_DEVICE
static inline double fd_adv_down(const double *f, int idx, int s, double inv_dx)
{
    return ( ( 1.0/60.0) * f[idx - 4*s]
           + (-2.0/15.0) * f[idx - 3*s]
           + ( 1.0/ 2.0) * f[idx - 2*s]
           + (-4.0/ 3.0) * f[idx -   s]
           + ( 7.0/12.0) * f[idx]
           + ( 2.0/ 5.0) * f[idx +   s]
           + (-1.0/30.0) * f[idx + 2*s] ) * inv_dx;
}

LATTICE_DEVICE
static inline double fd_adv(const double *f, int idx, int s, double vel,
                            double inv_dx)
{
    if (vel > 0.0)
        return vel * fd_adv_up(f, idx, s, inv_dx);
    else
        return vel * fd_adv_down(f, idx, s, inv_dx);
}

#elif FD_ORDER == 4

/* ========================================================================
 * 4th-order finite differences (5-point stencils, ghost >= 2)
 * Ref: GRChombo FourthOrderDerivatives.hpp
 * ======================================================================== */

/*
 * 4th-order first derivative: d f / d x
 * Ref: GRChombo FourthOrderDerivatives.hpp lines 36-46
 */
LATTICE_DEVICE
static inline double fd_d1(const double *f, int idx, int s, double inv_dx)
{
    return (  (1.0 / 12.0) * f[idx - 2*s]
            - (2.0 /  3.0) * f[idx -   s]
            + (2.0 /  3.0) * f[idx +   s]
            - (1.0 / 12.0) * f[idx + 2*s] ) * inv_dx;
}

/*
 * 4th-order second derivative: d^2 f / d x^2
 * Ref: GRChombo FourthOrderDerivatives.hpp lines 114-128
 */
LATTICE_DEVICE
static inline double fd_d2(const double *f, int idx, int s, double inv_dx)
{
    double inv_dx2 = inv_dx * inv_dx;
    return ( -(1.0 / 12.0) * f[idx - 2*s]
             + (4.0 /  3.0) * f[idx -   s]
             - (5.0 /  2.0) * f[idx]
             + (4.0 /  3.0) * f[idx +   s]
             - (1.0 / 12.0) * f[idx + 2*s] ) * inv_dx2;
}

/*
 * 4th-order fused first + second derivative (diagonal d2 only).
 * Loads 5 stencil points once, computes both d1 and d2.
 * Ref: Fornberg table, 4th-order centered d1 + d2
 */
LATTICE_DEVICE
static inline void fd_d1_d2(const double *f, int idx, int s, double inv_dx,
                            double *out_d1, double *out_d2)
{
    double fm2 = f[idx - 2*s], fm1 = f[idx - s];
    double f0  = f[idx];
    double fp1 = f[idx + s],   fp2 = f[idx + 2*s];

    /* d1: [1/12, -2/3, 0, 2/3, -1/12] * inv_dx */
    *out_d1 = ((1.0/12.0)*fm2 - (2.0/3.0)*fm1
               + (2.0/3.0)*fp1 - (1.0/12.0)*fp2) * inv_dx;

    /* d2: [-1/12, 4/3, -5/2, 4/3, -1/12] * inv_dx^2 */
    double inv_dx2 = inv_dx * inv_dx;
    *out_d2 = (-(1.0/12.0)*fm2 + (4.0/3.0)*fm1
              - (5.0/2.0)*f0
              + (4.0/3.0)*fp1 - (1.0/12.0)*fp2) * inv_dx2;
}

/*
 * 4th-order mixed second derivative: d^2 f / (d x_a d x_b)
 * Ref: GRChombo mixed_diff2, FourthOrderDerivatives.hpp lines 163-192
 */
LATTICE_DEVICE
static inline double fd_d2_mixed(const double *f, int idx, int s1, int s2,
                                 double inv_dx)
{
    double inv_dx2 = inv_dx * inv_dx;

    double wff = 1.0 / 144.0;   /* (1/12)^2   */
    double wnf =  1.0 / 18.0;   /* (1/12)(2/3) */
    double wnn =  4.0 /  9.0;   /* (2/3)^2     */

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
           + wff * f[idx + 2*s1 + 2*s2] ) * inv_dx2;
}

/*
 * 4th-order upwind advection derivative: beta^a d_a f
 * Ref: GRChombo advection_term, FourthOrderDerivatives.hpp lines 269-301
 */
LATTICE_DEVICE
static inline double fd_adv(const double *f, int idx, int s, double vel,
                            double inv_dx)
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
                     + w4 * f[idx + 3*s] ) * inv_dx;
    } else {
        return vel * (-w4 * f[idx - 3*s]
                     - w3 * f[idx - 2*s]
                     - w2 * f[idx -   s]
                     - w1 * f[idx]
                     - w0 * f[idx +   s] ) * inv_dx;
    }
}

#else
#error "FD_ORDER must be 4 or 6"
#endif

/* ========================================================================
 * Kreiss-Oliger dissipation — decoupled from FD order.
 *
 * KO_ORDER=6: 7-point stencil, half-width 3. With GHOST=4, margin=1.
 *   GRChombo/AthenaK default. Safe for AMR (avoids worst ghost cell).
 * KO_ORDER=8: 9-point stencil, half-width 4. With GHOST=4, margin=0.
 *   Zero margin — first interior cell reads outermost ghost cell (g0).
 *
 * Sign convention: negative central weight (dissipative when added with
 * positive sigma). Both orders use this convention.
 *
 * Ref: GRChombo FourthOrderDerivatives.hpp lines 361-378 (6th-order)
 * Ref: GRChombo SixthOrderDerivatives.hpp lines 398-422 (8th-order)
 * Ref: arXiv:2404.01137 — higher-order KO paired with higher-order FD
 * ======================================================================== */

#ifndef KO_ORDER
#define KO_ORDER 6
#endif

#if KO_ORDER == 6

/*
 * 6th-order Kreiss-Oliger dissipation operator.
 * 7-point stencil (half-width 3), divided by dx.
 * Ref: GRChombo FourthOrderDerivatives.hpp lines 361-378
 */
LATTICE_DEVICE
static inline double fd_ko(const double *f, int idx, int s, double inv_dx)
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
           + wvf * f[idx + 3*s] ) * inv_dx;
}

#elif KO_ORDER == 8

/*
 * 8th-order Kreiss-Oliger dissipation operator.
 * 9-point stencil (half-width 4), divided by dx.
 * WARNING: With GHOST=4 this gives zero ghost margin — the stencil at the
 * first interior cell reads the outermost ghost cell (lowest quality data).
 * Ref: GRChombo SixthOrderDerivatives.hpp lines 398-422 (commented-out 8th)
 */
LATTICE_DEVICE
static inline double fd_ko(const double *f, int idx, int s, double inv_dx)
{
    double wvvf =  1.0 / 256.0;   /* 3.9063e-3 */
    double wvf  =  8.0 / 256.0;   /* 3.1250e-2 */
    double wf   = 28.0 / 256.0;   /* 1.0938e-1 */
    double wn   = 56.0 / 256.0;   /* 2.1875e-1 */
    double wl   = 70.0 / 256.0;   /* 2.7344e-1 */

    return (-wvvf * f[idx - 4*s]
            + wvf * f[idx - 3*s]
            - wf  * f[idx - 2*s]
            + wn  * f[idx -   s]
            - wl  * f[idx]
            + wn  * f[idx +   s]
            - wf  * f[idx + 2*s]
            + wvf * f[idx + 3*s]
            -wvvf * f[idx + 4*s]) * inv_dx;
}

#else
#error "KO_ORDER must be 6 or 8"
#endif

#endif /* LATTICE_FINITE_DIFF_H */
