/*
 * puncture.c — Brill-Lindquist puncture initial data
 *
 * Single BH (Schwarzschild in isotropic coordinates):
 *   psi = 1 + M / (2r)
 *   chi = psi^{-4}
 *   alpha = chi^{1/2}  (pre-collapsed lapse)
 *   gt_{ij} = delta_{ij}
 *   K = 0, At = 0, Ghat = 0, Theta = 0, beta = 0, B = 0
 *
 * Multiple BHs (superposition):
 *   psi = 1 + sum_n M_n / (2 |x - x_n|)
 *
 * Ref: B&S Ch. 3.4 (Brill-Lindquist); gr-qc/9703066 (punctures)
 */

#include "../core/grid.h"
#include "../core/fields.h"

#include <math.h>

typedef struct {
    double x, y, z;   /* Position */
    double mass;       /* Bare mass parameter */
} puncture_params_t;

/*
 * Set single Schwarzschild puncture initial data.
 * BH centered at (cx, cy, cz) with bare mass M.
 */
void puncture_set_single(grid_t *g, double mass, double cx, double cy, double cz)
{
    GRID_LOOP_ALL(g, i, j, k) {
        int idx = grid_idx(g, i, j, k);

        double x = grid_x(g, i) - cx;
        double y = grid_y(g, j) - cy;
        double z = grid_z(g, k) - cz;
        double r = sqrt(x * x + y * y + z * z);

        /* Regularize: avoid 1/0 at the puncture.
         * The puncture is a coordinate singularity, not a physical one.
         * Use a small floor for r. */
        if (r < 1e-6) r = 1e-6;

        /* Conformal factor: psi = 1 + M/(2r) */
        double psi = 1.0 + mass / (2.0 * r);
        double psi4 = psi * psi * psi * psi;

        /* chi = psi^{-4} = e^{-4 phi} */
        double chi = 1.0 / psi4;

        /* Pre-collapsed lapse: alpha = chi^{1/2} = psi^{-2} */
        double alpha = 1.0 / (psi * psi);

        /* Conformal metric = flat */
        g->fields[FIELD_CHI][idx] = chi;
        g->fields[FIELD_GT11][idx] = 1.0;
        g->fields[FIELD_GT12][idx] = 0.0;
        g->fields[FIELD_GT13][idx] = 0.0;
        g->fields[FIELD_GT22][idx] = 1.0;
        g->fields[FIELD_GT23][idx] = 0.0;
        g->fields[FIELD_GT33][idx] = 1.0;

        /* Extrinsic curvature = 0 (time-symmetric) */
        g->fields[FIELD_TRKA][idx] = 0.0;
        for (int a = 0; a < 6; a++) {
            g->fields[FIELD_AT_BASE + a][idx] = 0.0;
        }

        /* Connections, Theta, shift, B = 0 */
        g->fields[FIELD_GHAT1][idx] = 0.0;
        g->fields[FIELD_GHAT2][idx] = 0.0;
        g->fields[FIELD_GHAT3][idx] = 0.0;
        g->fields[FIELD_THETA][idx] = 0.0;
        g->fields[FIELD_ALPHA][idx] = alpha;
        g->fields[FIELD_BETA1][idx] = 0.0;
        g->fields[FIELD_BETA2][idx] = 0.0;
        g->fields[FIELD_BETA3][idx] = 0.0;
        g->fields[FIELD_GBAUX1][idx] = 0.0;
        g->fields[FIELD_GBAUX2][idx] = 0.0;
        g->fields[FIELD_GBAUX3][idx] = 0.0;
    }
}

/*
 * Set multiple Brill-Lindquist punctures (superposition of conformal factors).
 * Each puncture adds M_n / (2 |x - x_n|) to the conformal factor.
 */
void puncture_set_multiple(grid_t *g, const puncture_params_t params[], int n_punctures)
{
    GRID_LOOP_ALL(g, i, j, k) {
        int idx = grid_idx(g, i, j, k);

        double psi = 1.0;
        for (int n = 0; n < n_punctures; n++) {
            double dx = grid_x(g, i) - params[n].x;
            double dy = grid_y(g, j) - params[n].y;
            double dz = grid_z(g, k) - params[n].z;
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-6) r = 1e-6;
            psi += params[n].mass / (2.0 * r);
        }

        double psi4 = psi * psi * psi * psi;
        double chi = 1.0 / psi4;
        double alpha = 1.0 / (psi * psi);

        g->fields[FIELD_CHI][idx] = chi;
        g->fields[FIELD_GT11][idx] = 1.0;
        g->fields[FIELD_GT12][idx] = 0.0;
        g->fields[FIELD_GT13][idx] = 0.0;
        g->fields[FIELD_GT22][idx] = 1.0;
        g->fields[FIELD_GT23][idx] = 0.0;
        g->fields[FIELD_GT33][idx] = 1.0;
        g->fields[FIELD_TRKA][idx] = 0.0;
        for (int a = 0; a < 6; a++)
            g->fields[FIELD_AT_BASE + a][idx] = 0.0;
        g->fields[FIELD_GHAT1][idx] = 0.0;
        g->fields[FIELD_GHAT2][idx] = 0.0;
        g->fields[FIELD_GHAT3][idx] = 0.0;
        g->fields[FIELD_THETA][idx] = 0.0;
        g->fields[FIELD_ALPHA][idx] = alpha;
        g->fields[FIELD_BETA1][idx] = 0.0;
        g->fields[FIELD_BETA2][idx] = 0.0;
        g->fields[FIELD_BETA3][idx] = 0.0;
        g->fields[FIELD_GBAUX1][idx] = 0.0;
        g->fields[FIELD_GBAUX2][idx] = 0.0;
        g->fields[FIELD_GBAUX3][idx] = 0.0;
    }
}
