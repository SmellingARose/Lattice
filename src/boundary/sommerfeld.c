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
#ifdef LATTICE_GPU
#pragma omp declare target
#endif
double asymptotic_value(int field)
{
    /* Array lookup: branchless, eliminates GPU warp divergence.
     * Fields with flat-space value 1: chi, h_11, h_22, h_33, lapse.
     * All others (K, A_ij, Theta, Gamma^i, shift, B^i, E^i, BM^i) → 0. */
    static const double asym[NUM_FIELDS] = {
        [FIELD_CHI]   = 1.0,
        [FIELD_H11]   = 1.0, [FIELD_H22] = 1.0, [FIELD_H33] = 1.0,
        [FIELD_LAPSE] = 1.0
    };
    return asym[field];
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
#ifdef LATTICE_GPU
#pragma omp end declare target
#endif

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

                for (int field = 0; field < g->n_fields; field++) {
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

                for (int field = 0; field < g->n_fields; field++) {
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
