/*
 * Lattice — 3D Numerical Relativity
 * Constraint-preserving boundary conditions (BAM approach).
 *
 * Replaces the RHS of constraint-carrying fields at boundary points with
 * outgoing-wave equations at their correct characteristic speeds, while
 * keeping standard Sommerfeld for metric/gauge fields.
 *
 * Ref: arXiv:1212.2901 (Hilditch et al., BAM)
 */

#ifndef LATTICE_CONSTRAINT_PRESERVING_H
#define LATTICE_CONSTRAINT_PRESERVING_H

#include "../core/device.h"
#include "../core/fields.h"
#include <math.h>

/*
 * Characteristic speed for constraint field `f` at given lapse.
 * face_dir: 0=x, 1=y, 2=z — selects normal vs tangential Gamma speed.
 * Returns 0.0 for non-CP fields (caller uses standard Sommerfeld).
 *
 * Speeds from BAM (arXiv:1212.2901):
 *   Theta       → 1.0
 *   K           → sqrt(2/alpha)  (gauge-constraint mix)
 *   A_ij        → 1.0
 *   Gamma^s (normal to face)      → sqrt(3/4)
 *   Gamma^A (tangential to face)  → 1.0
 */
LATTICE_DEVICE
static inline double cp_char_speed(int f, int face_dir, double alpha)
{
    if (f == FIELD_THETA)
        return 1.0;

    if (f == FIELD_K) {
        double a = alpha > 0.01 ? alpha : 0.01;
        return sqrt(2.0 / a);
    }

    if (f >= FIELD_A11 && f <= FIELD_A33)
        return 1.0;

    if (f >= FIELD_GAMMA1 && f <= FIELD_GAMMA3) {
        /* Normal component: Gamma^{face_dir} → sqrt(3/4) */
        if (f == FIELD_GAMMA1 + face_dir)
            return sqrt(0.75);
        /* Tangential components → 1.0 */
        return 1.0;
    }

    return 0.0;  /* Non-CP field: use standard Sommerfeld */
}

/*
 * CP-BC RHS formula for a single field at a boundary point.
 *   rhs(f) = -alpha * v_char * s_sign * d_s(f) - alpha * (f - f_asymp) / r
 *
 * alpha:  lapse at this point
 * speed:  characteristic speed from cp_char_speed()
 * s_sign: +1 or -1 (outward normal direction along the face axis)
 * df_ds:  one-sided derivative in the normal direction
 * f_val:  field value at this point
 * f_asymp: asymptotic value of the field
 * r:      distance from origin
 */
LATTICE_DEVICE
static inline double cp_rhs(double alpha, double speed, double s_sign,
                             double df_ds, double f_val, double f_asymp,
                             double r)
{
    return -alpha * speed * s_sign * df_ds - alpha * (f_val - f_asymp) / r;
}

#endif /* LATTICE_CONSTRAINT_PRESERVING_H */
