/*
 * Lattice — 3D Numerical Relativity
 * Volume-weighted restriction (fine → coarse).
 *
 * Each coarse cell is the average of the 8 fine cells that overlap it.
 * This is 2nd-order accurate and conservative.
 *
 * Ref: GRChombo uses Chombo's CoarseAverage class for this operation.
 */

#include "restriction.h"
#include "../core/fields.h"

void restrict_field(const grid_t *fine_g, int ff,
                    grid_t *coarse_g, int cf)
{
    const int ghost = coarse_g->ghost;
    const int N_c = coarse_g->N;
    const double *src = fine_g->fields[ff];

    /* Loop over coarse interior cells */
    for (int ck = ghost; ck < ghost + N_c; ck++) {
        for (int cj = ghost; cj < ghost + N_c; cj++) {
            for (int ci = ghost; ci < ghost + N_c; ci++) {

                /* Sum the 8 fine children overlapping this coarse cell */
                double sum = 0.0;
                for (int ok = 0; ok < 2; ok++) {
                    for (int oj = 0; oj < 2; oj++) {
                        for (int oi = 0; oi < 2; oi++) {
                            int fi = 2 * (ci - ghost) + ghost + oi;
                            int fj = 2 * (cj - ghost) + ghost + oj;
                            int fk = 2 * (ck - ghost) + ghost + ok;
                            sum += src[IDX(fine_g, fi, fj, fk)];
                        }
                    }
                }

                coarse_g->fields[cf][IDX(coarse_g, ci, cj, ck)] = sum * 0.125;
            }
        }
    }
}

void restrict_all(const grid_t *fine_g, grid_t *coarse_g)
{
    for (int f = 0; f < NUM_FIELDS; f++) {
        restrict_field(fine_g, f, coarse_g, f);
    }
}
