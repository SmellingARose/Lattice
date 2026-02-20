/*
 * Lattice -- 3D Numerical Relativity
 * 4th-order Lagrange interpolation at arbitrary (x,y,z) from a grid_t.
 *
 * Uses 5-point stencil (same Lagrange basis as prolongation.h).
 * Tensor-product evaluation: 5^3 = 125 source points per interpolation.
 *
 * Also provides derivative interpolation using the analytically-known
 * derivative of the Lagrange basis polynomials.
 *
 * Ref: AthenaK src/mesh/prolongation.hpp (Lagrange basis)
 */

#ifndef LATTICE_INTERPOLATE_H
#define LATTICE_INTERPOLATE_H

#include "../core/grid.h"
#include <math.h>

#define INTERP_STENCIL 5
#define INTERP_HALF    2  /* stencil half-width = INTERP_STENCIL / 2 */

/*
 * 4th-order Lagrange basis values at offset delta from center node.
 * Nodes at positions {-2, -1, 0, +1, +2}.
 * L_k(delta) = prod_{j!=k} (delta - j) / (k - j)
 */
static inline void lagrange_basis_5(double delta, double w[INTERP_STENCIL])
{
    /* Node positions: n[k] = k - 2 for k = 0..4 */
    double d = delta;
    /* L_0(d) at node -2: prod over j=-1,0,1,2 of (d-j)/((-2)-j) */
    w[0] = (d + 1.0) * d * (d - 1.0) * (d - 2.0) / 24.0;
    /* L_1(d) at node -1: prod over j=-2,0,1,2 of (d-j)/((-1)-j) */
    w[1] = -(d + 2.0) * d * (d - 1.0) * (d - 2.0) / 6.0;
    /* L_2(d) at node  0: prod over j=-2,-1,1,2 of (d-j)/(0-j) */
    w[2] = (d + 2.0) * (d + 1.0) * (d - 1.0) * (d - 2.0) / 4.0;
    /* L_3(d) at node +1: prod over j=-2,-1,0,2 of (d-j)/(1-j) */
    w[3] = -(d + 2.0) * (d + 1.0) * d * (d - 2.0) / 6.0;
    /* L_4(d) at node +2: prod over j=-2,-1,0,1 of (d-j)/(2-j) */
    w[4] = (d + 2.0) * (d + 1.0) * d * (d - 1.0) / 24.0;
}

/*
 * Derivative of 4th-order Lagrange basis at offset delta.
 * dL_k/d(delta), analytically computed.
 * L_k(d) = prod_{j!=k} (d - n_j) / (n_k - n_j)
 * dL_k/dd = sum_{m!=k} [ prod_{j!=k,j!=m} (d - n_j) / (n_k - n_j) ]
 */
static inline void lagrange_basis_deriv_5(double delta,
                                           double dw[INTERP_STENCIL])
{
    double d = delta;
    /* Derivative of L_0 at node -2 */
    dw[0] = ((d)*(d - 1.0)*(d - 2.0)
           + (d + 1.0)*(d - 1.0)*(d - 2.0)
           + (d + 1.0)*(d)*(d - 2.0)
           + (d + 1.0)*(d)*(d - 1.0)) / 24.0;
    /* Derivative of L_1 at node -1 */
    dw[1] = -((d)*(d - 1.0)*(d - 2.0)
            + (d + 2.0)*(d - 1.0)*(d - 2.0)
            + (d + 2.0)*(d)*(d - 2.0)
            + (d + 2.0)*(d)*(d - 1.0)) / 6.0;
    /* Derivative of L_2 at node 0 */
    dw[2] = ((d + 1.0)*(d - 1.0)*(d - 2.0)
           + (d + 2.0)*(d - 1.0)*(d - 2.0)
           + (d + 2.0)*(d + 1.0)*(d - 2.0)
           + (d + 2.0)*(d + 1.0)*(d - 1.0)) / 4.0;
    /* Derivative of L_3 at node +1 */
    dw[3] = -((d + 1.0)*(d)*(d - 2.0)
            + (d + 2.0)*(d)*(d - 2.0)
            + (d + 2.0)*(d + 1.0)*(d - 2.0)
            + (d + 2.0)*(d + 1.0)*(d)) / 6.0;
    /* Derivative of L_4 at node +2 */
    dw[4] = ((d + 1.0)*(d)*(d - 1.0)
           + (d + 2.0)*(d)*(d - 1.0)
           + (d + 2.0)*(d + 1.0)*(d - 1.0)
           + (d + 2.0)*(d + 1.0)*(d)) / 24.0;
}

/*
 * Interpolate a field value at arbitrary physical coordinates (x,y,z).
 *
 * Algorithm:
 * 1. Map (x,y,z) to continuous grid index:
 *    ci = (x - origin) / dx + ghost  where origin = -L/2 + 0.5*dx
 *    (cell-centered: COORD(g, i) = (i - ghost + 0.5)*dx - L/2)
 * 2. Nearest grid point = round(ci), fractional offset delta in [-0.5, 0.5]
 * 3. 5-point Lagrange basis at delta in each direction
 * 4. 3D tensor product sum over 5^3 = 125 points
 *
 * Returns NAN if the point is too close to the grid boundary for the stencil.
 */
static inline double interp_field_at(const double *field, const grid_t *g,
                                      double x, double y, double z)
{
    double dx = g->dx;
    double origin = -g->L * 0.5 + 0.5 * dx;

    /* Continuous grid indices (with ghost offset) */
    double cix = (x - origin) / dx + g->ghost;
    double ciy = (y - origin) / dx + g->ghost;
    double ciz = (z - origin) / dx + g->ghost;

    /* Nearest grid point */
    int ix = (int)round(cix);
    int iy = (int)round(ciy);
    int iz = (int)round(ciz);

    /* Fractional offset */
    double dx_frac = cix - ix;
    double dy_frac = ciy - iy;
    double dz_frac = ciz - iz;

    /* Bounds check: stencil needs [i-2, i+2] within [0, Ntotal-1] */
    if (ix - INTERP_HALF < 0 || ix + INTERP_HALF >= g->Ntotal ||
        iy - INTERP_HALF < 0 || iy + INTERP_HALF >= g->Ntotal ||
        iz - INTERP_HALF < 0 || iz + INTERP_HALF >= g->Ntotal) {
        return 0.0 / 0.0;  /* NAN */
    }

    /* Compute 1D Lagrange weights */
    double wx[INTERP_STENCIL], wy[INTERP_STENCIL], wz[INTERP_STENCIL];
    lagrange_basis_5(dx_frac, wx);
    lagrange_basis_5(dy_frac, wy);
    lagrange_basis_5(dz_frac, wz);

    /* 3D tensor product */
    double val = 0.0;
    for (int sk = 0; sk < INTERP_STENCIL; sk++) {
        int kk = iz - INTERP_HALF + sk;
        for (int sj = 0; sj < INTERP_STENCIL; sj++) {
            int jj = iy - INTERP_HALF + sj;
            double wkj = wz[sk] * wy[sj];
            for (int si = 0; si < INTERP_STENCIL; si++) {
                int ii = ix - INTERP_HALF + si;
                val += wkj * wx[si] * field[IDX(g, ii, jj, kk)];
            }
        }
    }

    return val;
}

/*
 * Interpolate a field and its first spatial derivatives at (x,y,z).
 * val[0] = field value
 * val[1] = df/dx
 * val[2] = df/dy
 * val[3] = df/dz
 *
 * Derivatives computed via the derivative of the Lagrange basis
 * (analytically exact for the interpolant, no finite differencing).
 * The chain rule gives df/dx = (1/dx) * sum_i dL_i/d(delta_x) * ...
 */
static inline void interp_field_deriv_at(const double *field, const grid_t *g,
                                          double x, double y, double z,
                                          double val[4])
{
    double dx = g->dx;
    double origin = -g->L * 0.5 + 0.5 * dx;

    double cix = (x - origin) / dx + g->ghost;
    double ciy = (y - origin) / dx + g->ghost;
    double ciz = (z - origin) / dx + g->ghost;

    int ix = (int)round(cix);
    int iy = (int)round(ciy);
    int iz = (int)round(ciz);

    double dx_frac = cix - ix;
    double dy_frac = ciy - iy;
    double dz_frac = ciz - iz;

    if (ix - INTERP_HALF < 0 || ix + INTERP_HALF >= g->Ntotal ||
        iy - INTERP_HALF < 0 || iy + INTERP_HALF >= g->Ntotal ||
        iz - INTERP_HALF < 0 || iz + INTERP_HALF >= g->Ntotal) {
        val[0] = val[1] = val[2] = val[3] = 0.0 / 0.0;
        return;
    }

    /* Lagrange basis and derivatives */
    double wx[INTERP_STENCIL], wy[INTERP_STENCIL], wz[INTERP_STENCIL];
    double dwx[INTERP_STENCIL], dwy[INTERP_STENCIL], dwz[INTERP_STENCIL];
    lagrange_basis_5(dx_frac, wx);
    lagrange_basis_5(dy_frac, wy);
    lagrange_basis_5(dz_frac, wz);
    lagrange_basis_deriv_5(dx_frac, dwx);
    lagrange_basis_deriv_5(dy_frac, dwy);
    lagrange_basis_deriv_5(dz_frac, dwz);

    double f = 0.0, dfdx = 0.0, dfdy = 0.0, dfdz = 0.0;

    for (int sk = 0; sk < INTERP_STENCIL; sk++) {
        int kk = iz - INTERP_HALF + sk;
        for (int sj = 0; sj < INTERP_STENCIL; sj++) {
            int jj = iy - INTERP_HALF + sj;
            for (int si = 0; si < INTERP_STENCIL; si++) {
                int ii = ix - INTERP_HALF + si;
                double fv = field[IDX(g, ii, jj, kk)];
                f    += wz[sk]  * wy[sj]  * wx[si]  * fv;
                dfdx += wz[sk]  * wy[sj]  * dwx[si] * fv;
                dfdy += wz[sk]  * dwy[sj] * wx[si]  * fv;
                dfdz += dwz[sk] * wy[sj]  * wx[si]  * fv;
            }
        }
    }

    val[0] = f;
    val[1] = dfdx / dx;  /* chain rule: d/dx = (1/dx) * d/d(delta) */
    val[2] = dfdy / dx;
    val[3] = dfdz / dx;
}

/* ========================================================================
 * Block-aware interpolation for AMR meshes.
 *
 * Same 4th-order Lagrange interpolation as above, but uses an explicit
 * block origin instead of the global -L/2 convention. The coordinate
 * mapping for AMR blocks is:
 *   BLOCK_COORD(blk, dir, i) = origin[dir] + (i - ghost + 0.5) * dx
 *   ci = (x - origin[dir]) / dx - 0.5 + ghost
 *
 * Ref: block.h BLOCK_COORD macro.
 * ======================================================================== */

/*
 * Interpolate a field at physical (x,y,z) within a block whose grid origin
 * is given by origin[3].
 *
 * Coordinate mapping:
 *   cell center i has coordinate origin[d] + (i - ghost + 0.5) * dx
 *   → continuous index ci = (x - origin[d]) / dx - 0.5 + ghost
 */
static inline double interp_field_at_block(const double *field,
                                            const grid_t *g,
                                            const double origin[3],
                                            double x, double y, double z)
{
    double dx = g->dx;
    int ghost = g->ghost;

    double cix = (x - origin[0]) / dx - 0.5 + ghost;
    double ciy = (y - origin[1]) / dx - 0.5 + ghost;
    double ciz = (z - origin[2]) / dx - 0.5 + ghost;

    int ix = (int)round(cix);
    int iy = (int)round(ciy);
    int iz = (int)round(ciz);

    double dx_frac = cix - ix;
    double dy_frac = ciy - iy;
    double dz_frac = ciz - iz;

    if (ix - INTERP_HALF < 0 || ix + INTERP_HALF >= g->Ntotal ||
        iy - INTERP_HALF < 0 || iy + INTERP_HALF >= g->Ntotal ||
        iz - INTERP_HALF < 0 || iz + INTERP_HALF >= g->Ntotal) {
        return 0.0 / 0.0;  /* NAN */
    }

    double wx[INTERP_STENCIL], wy[INTERP_STENCIL], wz[INTERP_STENCIL];
    lagrange_basis_5(dx_frac, wx);
    lagrange_basis_5(dy_frac, wy);
    lagrange_basis_5(dz_frac, wz);

    double val = 0.0;
    for (int sk = 0; sk < INTERP_STENCIL; sk++) {
        int kk = iz - INTERP_HALF + sk;
        for (int sj = 0; sj < INTERP_STENCIL; sj++) {
            int jj = iy - INTERP_HALF + sj;
            double wkj = wz[sk] * wy[sj];
            for (int si = 0; si < INTERP_STENCIL; si++) {
                int ii = ix - INTERP_HALF + si;
                val += wkj * wx[si] * field[IDX(g, ii, jj, kk)];
            }
        }
    }

    return val;
}

/*
 * Interpolate a field and its derivatives at (x,y,z) within a block.
 * val[0] = field value, val[1..3] = df/dx, df/dy, df/dz.
 */
static inline void interp_field_deriv_at_block(const double *field,
                                                const grid_t *g,
                                                const double origin[3],
                                                double x, double y, double z,
                                                double val[4])
{
    double dx = g->dx;
    int ghost = g->ghost;

    double cix = (x - origin[0]) / dx - 0.5 + ghost;
    double ciy = (y - origin[1]) / dx - 0.5 + ghost;
    double ciz = (z - origin[2]) / dx - 0.5 + ghost;

    int ix = (int)round(cix);
    int iy = (int)round(ciy);
    int iz = (int)round(ciz);

    double dx_frac = cix - ix;
    double dy_frac = ciy - iy;
    double dz_frac = ciz - iz;

    if (ix - INTERP_HALF < 0 || ix + INTERP_HALF >= g->Ntotal ||
        iy - INTERP_HALF < 0 || iy + INTERP_HALF >= g->Ntotal ||
        iz - INTERP_HALF < 0 || iz + INTERP_HALF >= g->Ntotal) {
        val[0] = val[1] = val[2] = val[3] = 0.0 / 0.0;
        return;
    }

    double wx[INTERP_STENCIL], wy[INTERP_STENCIL], wz[INTERP_STENCIL];
    double dwx[INTERP_STENCIL], dwy[INTERP_STENCIL], dwz[INTERP_STENCIL];
    lagrange_basis_5(dx_frac, wx);
    lagrange_basis_5(dy_frac, wy);
    lagrange_basis_5(dz_frac, wz);
    lagrange_basis_deriv_5(dx_frac, dwx);
    lagrange_basis_deriv_5(dy_frac, dwy);
    lagrange_basis_deriv_5(dz_frac, dwz);

    double f = 0.0, dfdx = 0.0, dfdy = 0.0, dfdz = 0.0;

    for (int sk = 0; sk < INTERP_STENCIL; sk++) {
        int kk = iz - INTERP_HALF + sk;
        for (int sj = 0; sj < INTERP_STENCIL; sj++) {
            int jj = iy - INTERP_HALF + sj;
            for (int si = 0; si < INTERP_STENCIL; si++) {
                int ii = ix - INTERP_HALF + si;
                double fv = field[IDX(g, ii, jj, kk)];
                f    += wz[sk]  * wy[sj]  * wx[si]  * fv;
                dfdx += wz[sk]  * wy[sj]  * dwx[si] * fv;
                dfdy += wz[sk]  * dwy[sj] * wx[si]  * fv;
                dfdz += dwz[sk] * wy[sj]  * wx[si]  * fv;
            }
        }
    }

    val[0] = f;
    val[1] = dfdx / dx;
    val[2] = dfdy / dx;
    val[3] = dfdz / dx;
}

#endif /* LATTICE_INTERPOLATE_H */
