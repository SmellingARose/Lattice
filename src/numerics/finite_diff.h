/*
 * finite_diff.h — Finite difference macros (header-only)
 *
 * All FD stencils are 4th-order centered. KO dissipation is 6th-order.
 * Ghost width of 4 is required for all stencils.
 *
 * Arguments:
 *   f     — pointer to field array (double*)
 *   idx   — linear index of center point
 *   s     — stride in the differentiation direction
 *   dx    — grid spacing in the differentiation direction
 *   vel   — advection velocity (for upwind stencil)
 *
 * Coefficients from standard 4th-order centered differences:
 *   D1: (-1, 8, 0, -8, 1) / (12 dx)
 *   D2: (-1, 16, -30, 16, -1) / (12 dx^2)
 *   KO6: (1, -6, 15, -20, 15, -6, 1) / 64 / dx  [7-point, 6th-order]
 *
 * B&S Ch. 8 (finite differencing); arXiv:gr-qc/0206072 (advection terms)
 */

#ifndef LATTICE_FINITE_DIFF_H
#define LATTICE_FINITE_DIFF_H

/*
 * 4th-order centered first derivative: df/dx
 * Stencil: (-1, 8, 0, -8, 1) / (12 dx)
 */
#define FD_D1(f, idx, s, dx) \
    (((f)[(idx) - 2*(s)] \
      - 8.0 * (f)[(idx) - (s)] \
      + 8.0 * (f)[(idx) + (s)] \
      - (f)[(idx) + 2*(s)]) / (12.0 * (dx)))

/*
 * 4th-order centered second derivative: d^2f/dx^2
 * Stencil: (-1, 16, -30, 16, -1) / (12 dx^2)
 */
#define FD_D2(f, idx, s, dx) \
    ((-(f)[(idx) - 2*(s)] \
      + 16.0 * (f)[(idx) - (s)] \
      - 30.0 * (f)[(idx)] \
      + 16.0 * (f)[(idx) + (s)] \
      - (f)[(idx) + 2*(s)]) / (12.0 * (dx) * (dx)))

/*
 * Mixed partial: d^2f/(dx dy)
 * Composed from two 4th-order first derivatives.
 * Applies D1_x to D1_y (or vice versa — symmetric).
 */
#define FD_D1D1(f, idx, sx, sy, dx, dy) \
    ((FD_D1(f, (idx) - 2*(sx), sy, dy) \
      - 8.0 * FD_D1(f, (idx) - (sx), sy, dy) \
      + 8.0 * FD_D1(f, (idx) + (sx), sy, dy) \
      - FD_D1(f, (idx) + 2*(sx), sy, dy)) / (12.0 * (dx)))

/*
 * 4th-order upwind advection: vel * df/dx
 *
 * Decomposes into centered derivative + sign-dependent dissipation:
 *   FD_ADV = vel * D1_centered + |vel| * h^4 * D5 / 12
 * where D5 is the 5th undivided difference (matching direction).
 *
 * For vel > 0 (left-biased stencil, points -3 to +2):
 *   vel * (-f[-3] + 6f[-2] - 18f[-1] + 10f[0] + 3f[+1]) / (-12 dx)
 *   = vel * (f[-3] - 6f[-2] + 18f[-1] - 10f[0] - 3f[+1]) / (12 dx)
 *
 * For vel < 0 (right-biased stencil, points -2 to +3):
 *   vel * (3f[-1] + 10f[0] - 18f[+1] + 6f[+2] - f[+3]) / (12 dx)
 *
 * Ref: Zlochower et al., gr-qc/0505055
 */
#define FD_ADV_LEFT(f, idx, s, dx, vel) \
    ((vel) * ((f)[(idx) - 3*(s)] \
              - 6.0 * (f)[(idx) - 2*(s)] \
              + 18.0 * (f)[(idx) - (s)] \
              - 10.0 * (f)[(idx)] \
              - 3.0 * (f)[(idx) + (s)]) / (12.0 * (dx)))

#define FD_ADV_RIGHT(f, idx, s, dx, vel) \
    ((vel) * (3.0 * (f)[(idx) - (s)] \
              + 10.0 * (f)[(idx)] \
              - 18.0 * (f)[(idx) + (s)] \
              + 6.0 * (f)[(idx) + 2*(s)] \
              - (f)[(idx) + 3*(s)]) / (12.0 * (dx)))

#define FD_ADV(f, idx, s, dx, vel) \
    ((vel) >= 0.0 \
     ? FD_ADV_LEFT(f, idx, s, dx, vel) \
     : FD_ADV_RIGHT(f, idx, s, dx, vel))

/*
 * 6th-order Kreiss-Oliger dissipation operator
 * Stencil: (1, -6, 15, -20, 15, -6, 1) / 64
 * Applied as: rhs -= eps * (KO6_x + KO6_y + KO6_z) / dx
 *
 * Note: requires 3 ghost points on each side (fits in ghost_width=4).
 * Ref: Kreiss & Oliger (1973); Babiuc et al., arXiv:0709.3559
 */
#define FD_KO6(f, idx, s, dx) \
    (((f)[(idx) - 3*(s)] \
      - 6.0 * (f)[(idx) - 2*(s)] \
      + 15.0 * (f)[(idx) - (s)] \
      - 20.0 * (f)[(idx)] \
      + 15.0 * (f)[(idx) + (s)] \
      - 6.0 * (f)[(idx) + 2*(s)] \
      + (f)[(idx) + 3*(s)]) / (64.0 * (dx)))

#endif /* LATTICE_FINITE_DIFF_H */
