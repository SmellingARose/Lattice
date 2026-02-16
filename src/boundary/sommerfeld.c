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

/* Asymptotic values for each field */
static double asymptotic_value(int field)
{
    switch (field) {
        case FIELD_CHI:   return 1.0;
        case FIELD_H11:   return 1.0;
        case FIELD_H22:   return 1.0;
        case FIELD_H33:   return 1.0;
        case FIELD_LAPSE: return 1.0;
        default:          return 0.0;
    }
}

/*
 * 2nd-order derivative at boundary using one-sided stencils.
 * Ref: GRChombo BoundaryConditions.cpp:617-649
 */
static double boundary_d1(const double *f, int idx, int stride,
                          int lo_offset, int hi_offset, double dx)
{
    if (lo_offset < 1) {
        /* Near low boundary — forward stencil */
        return (-1.5 * f[idx] + 2.0 * f[idx + stride]
                - 0.5 * f[idx + 2*stride]) / dx;
    } else if (hi_offset < 1) {
        /* Near high boundary — backward stencil */
        return (1.5 * f[idx] - 2.0 * f[idx - stride]
                + 0.5 * f[idx - 2*stride]) / dx;
    } else {
        /* Interior — centered 2nd order */
        return 0.5 * (f[idx + stride] - f[idx - stride]) / dx;
    }
}

void apply_sommerfeld(double **rhs, const double *const *src, const grid_t *g)
{
    int lo = g->ghost;
    int hi = g->ghost + g->N;
    int Nt = g->Ntotal;

    for (int k = 0; k < Nt; k++) {
        for (int j = 0; j < Nt; j++) {
            for (int i = 0; i < Nt; i++) {
                /* Skip interior points */
                if (i >= lo && i < hi &&
                    j >= lo && j < hi &&
                    k >= lo && k < hi)
                    continue;

                int idx = IDX(g, i, j, k);

                double x = COORD(g, i);
                double y = COORD(g, j);
                double z = COORD(g, k);
                double r = sqrt(x*x + y*y + z*z);
                if (r < 1.0e-10) r = 1.0e-10;

                /* Distance from each boundary edge */
                int lo_off[3] = { i, j, k };
                int hi_off[3] = { Nt - 1 - i, Nt - 1 - j, Nt - 1 - k };

                int strides[3] = { STRIDE_X, STRIDE_Y(g), STRIDE_Z(g) };
                double loc[3] = { x, y, z };

                for (int field = 0; field < NUM_FIELDS; field++) {
                    double sommerfeld = 0.0;

                    /* Sum: -d_i f * x^i / r */
                    for (int dir = 0; dir < 3; dir++) {
                        double d1 = boundary_d1(src[field], idx,
                                                strides[dir],
                                                lo_off[dir], hi_off[dir],
                                                g->dx);
                        sommerfeld += -d1 * loc[dir] / r;
                    }

                    /* Add decay: (f_asymptotic - f) / r */
                    double f_asym = asymptotic_value(field);
                    sommerfeld += (f_asym - src[field][idx]) / r;

                    rhs[field][idx] = sommerfeld;
                }
            }
        }
    }
}

/*
 * Block-aware Sommerfeld: only applies to ghost points adjacent to
 * domain boundaries (on_boundary[face] == 1). Ghost zones filled by
 * inter-block exchange are left untouched.
 *
 * Physical coordinates use BLOCK_COORD (block origin + local index)
 * to give correct global coordinates for multi-block meshes.
 *
 * For each ghost point, we check which directions place it outside the
 * interior and whether on_boundary is set for that direction. If no
 * direction qualifies, the point was filled by ghost exchange.
 */
void apply_sommerfeld_block(double **rhs, const double *const *src,
                            const block_t *b)
{
    const grid_t *g = b->grid;
    int lo = g->ghost;
    int hi = g->ghost + g->N;
    int Nt = g->Ntotal;

    for (int k = 0; k < Nt; k++) {
        for (int j = 0; j < Nt; j++) {
            for (int i = 0; i < Nt; i++) {
                /* Skip interior points */
                if (i >= lo && i < hi &&
                    j >= lo && j < hi &&
                    k >= lo && k < hi)
                    continue;

                /* Check if this ghost point borders a domain boundary.
                 * A point is near boundary face F if:
                 *   - It's in the ghost zone for direction F
                 *   - on_boundary[F] == 1 */
                int near_boundary = 0;
                if (i < lo  && b->on_boundary[0]) near_boundary = 1;
                if (i >= hi && b->on_boundary[1]) near_boundary = 1;
                if (j < lo  && b->on_boundary[2]) near_boundary = 1;
                if (j >= hi && b->on_boundary[3]) near_boundary = 1;
                if (k < lo  && b->on_boundary[4]) near_boundary = 1;
                if (k >= hi && b->on_boundary[5]) near_boundary = 1;

                if (!near_boundary) continue;

                int idx = IDX(g, i, j, k);

                /* Physical coordinates via block origin */
                double x = BLOCK_COORD(b, 0, i);
                double y = BLOCK_COORD(b, 1, j);
                double z = BLOCK_COORD(b, 2, k);
                double r = sqrt(x*x + y*y + z*z);
                if (r < 1.0e-10) r = 1.0e-10;

                /* Distance from each boundary edge (used for stencil choice) */
                int lo_off[3] = { i, j, k };
                int hi_off[3] = { Nt - 1 - i, Nt - 1 - j, Nt - 1 - k };

                int strides[3] = { STRIDE_X, STRIDE_Y(g), STRIDE_Z(g) };
                double loc[3] = { x, y, z };

                for (int field = 0; field < NUM_FIELDS; field++) {
                    double sommerfeld = 0.0;

                    for (int dir = 0; dir < 3; dir++) {
                        double d1 = boundary_d1(src[field], idx,
                                                strides[dir],
                                                lo_off[dir], hi_off[dir],
                                                g->dx);
                        sommerfeld += -d1 * loc[dir] / r;
                    }

                    double f_asym = asymptotic_value(field);
                    sommerfeld += (f_asym - src[field][idx]) / r;

                    rhs[field][idx] = sommerfeld;
                }
            }
        }
    }
}
