/*
 * Lattice — 3D Numerical Relativity
 * Puncture initial data.
 *
 * Flat spacetime: trivial Minkowski.
 * Brill-Lindquist: conformal factor psi = 1 + sum(M_n / 2r_n)
 *
 * Ref: gr-qc/9703066 (Brandt-Brugmann)
 * Ref: GRChombo does not include BL data — this is original code
 */

#include "puncture.h"
#include "../core/fields.h"
#include <math.h>

/* Helper: compute BL data at a single point given global (x,y,z). */
static void set_bl_point(grid_t *g, int idx, double x, double y, double z,
                         int n_bh, const double *masses,
                         const double centers[][3])
{
    double psi = 1.0;
    for (int n = 0; n < n_bh; n++) {
        double rx = x - centers[n][0];
        double ry = y - centers[n][1];
        double rz = z - centers[n][2];
        double r  = sqrt(rx*rx + ry*ry + rz*rz);
        if (r < 1.0e-10) r = 1.0e-10;
        psi += masses[n] / (2.0 * r);
    }

    double chi = 1.0 / (psi * psi * psi * psi);

    g->fields[FIELD_CHI][idx]    = chi;
    g->fields[FIELD_H11][idx]    = 1.0;
    g->fields[FIELD_H12][idx]    = 0.0;
    g->fields[FIELD_H13][idx]    = 0.0;
    g->fields[FIELD_H22][idx]    = 1.0;
    g->fields[FIELD_H23][idx]    = 0.0;
    g->fields[FIELD_H33][idx]    = 1.0;
    g->fields[FIELD_K][idx]      = 0.0;
    g->fields[FIELD_A11][idx]    = 0.0;
    g->fields[FIELD_A12][idx]    = 0.0;
    g->fields[FIELD_A13][idx]    = 0.0;
    g->fields[FIELD_A22][idx]    = 0.0;
    g->fields[FIELD_A23][idx]    = 0.0;
    g->fields[FIELD_A33][idx]    = 0.0;
    g->fields[FIELD_THETA][idx]  = 0.0;
    g->fields[FIELD_GAMMA1][idx] = 0.0;
    g->fields[FIELD_GAMMA2][idx] = 0.0;
    g->fields[FIELD_GAMMA3][idx] = 0.0;
    g->fields[FIELD_LAPSE][idx]  = sqrt(chi);
    g->fields[FIELD_SHIFT1][idx] = 0.0;
    g->fields[FIELD_SHIFT2][idx] = 0.0;
    g->fields[FIELD_SHIFT3][idx] = 0.0;
    g->fields[FIELD_B1][idx]     = 0.0;
    g->fields[FIELD_B2][idx]     = 0.0;
    g->fields[FIELD_B3][idx]     = 0.0;
    g->fields[FIELD_E1][idx]     = 0.0;
    g->fields[FIELD_E2][idx]     = 0.0;
    g->fields[FIELD_E3][idx]     = 0.0;
    g->fields[FIELD_BM1][idx]    = 0.0;
    g->fields[FIELD_BM2][idx]    = 0.0;
    g->fields[FIELD_BM3][idx]    = 0.0;
}

void set_flat_spacetime(grid_t *g)
{
    for (int k = 0; k < g->Ntotal; k++) {
        for (int j = 0; j < g->Ntotal; j++) {
            for (int i = 0; i < g->Ntotal; i++) {
                int idx = IDX(g, i, j, k);

                g->fields[FIELD_CHI][idx]   = 1.0;
                g->fields[FIELD_H11][idx]   = 1.0;
                g->fields[FIELD_H12][idx]   = 0.0;
                g->fields[FIELD_H13][idx]   = 0.0;
                g->fields[FIELD_H22][idx]   = 1.0;
                g->fields[FIELD_H23][idx]   = 0.0;
                g->fields[FIELD_H33][idx]   = 1.0;
                g->fields[FIELD_K][idx]     = 0.0;
                g->fields[FIELD_A11][idx]   = 0.0;
                g->fields[FIELD_A12][idx]   = 0.0;
                g->fields[FIELD_A13][idx]   = 0.0;
                g->fields[FIELD_A22][idx]   = 0.0;
                g->fields[FIELD_A23][idx]   = 0.0;
                g->fields[FIELD_A33][idx]   = 0.0;
                g->fields[FIELD_THETA][idx] = 0.0;
                g->fields[FIELD_GAMMA1][idx] = 0.0;
                g->fields[FIELD_GAMMA2][idx] = 0.0;
                g->fields[FIELD_GAMMA3][idx] = 0.0;
                g->fields[FIELD_LAPSE][idx]  = 1.0;
                g->fields[FIELD_SHIFT1][idx] = 0.0;
                g->fields[FIELD_SHIFT2][idx] = 0.0;
                g->fields[FIELD_SHIFT3][idx] = 0.0;
                g->fields[FIELD_B1][idx]     = 0.0;
                g->fields[FIELD_B2][idx]     = 0.0;
                g->fields[FIELD_B3][idx]     = 0.0;
                g->fields[FIELD_E1][idx]     = 0.0;
                g->fields[FIELD_E2][idx]     = 0.0;
                g->fields[FIELD_E3][idx]     = 0.0;
                g->fields[FIELD_BM1][idx]    = 0.0;
                g->fields[FIELD_BM2][idx]    = 0.0;
                g->fields[FIELD_BM3][idx]    = 0.0;
            }
        }
    }
}

void set_brill_lindquist(grid_t *g, int n_bh,
                         const double *masses,
                         const double centers[][3])
{
    for (int k = 0; k < g->Ntotal; k++) {
        for (int j = 0; j < g->Ntotal; j++) {
            for (int i = 0; i < g->Ntotal; i++) {
                int idx = IDX(g, i, j, k);
                double x = COORD(g, i);
                double y = COORD(g, j);
                double z = COORD(g, k);
                set_bl_point(g, idx, x, y, z, n_bh, masses, centers);
            }
        }
    }
}

void set_brill_lindquist_global(grid_t *g, const double origin[3],
                                int n_bh, const double *masses,
                                const double centers[][3])
{
    for (int k = 0; k < g->Ntotal; k++) {
        for (int j = 0; j < g->Ntotal; j++) {
            for (int i = 0; i < g->Ntotal; i++) {
                int idx = IDX(g, i, j, k);
                double x = origin[0] + (i - GHOST_WIDTH + 0.5) * g->dx;
                double y = origin[1] + (j - GHOST_WIDTH + 0.5) * g->dx;
                double z = origin[2] + (k - GHOST_WIDTH + 0.5) * g->dx;
                set_bl_point(g, idx, x, y, z, n_bh, masses, centers);
            }
        }
    }
}
