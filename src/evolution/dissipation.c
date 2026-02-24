/*
 * Lattice — 3D Numerical Relativity
 * Kreiss-Oliger dissipation with optional CAKO and per-field sigma.
 *
 * Adds sigma_eff * sum_dir(fd_ko) to each field's RHS at point (i,j,k).
 * 6th-order operator using 7-point stencil.
 *
 * CAKO (Chi-Adjusted KO): sigma_eff = W * sigma_base where W = sqrt(chi).
 *   Suppresses dissipation near punctures where chi → 0.
 *   Ref: arXiv:2404.01137, Eq. (20)
 *
 * Per-field sigma: sigma = 0.99 for gauge fields, 0.3 for physical fields.
 *   Ref: arXiv:2404.01137, text near Eq. (20)
 *
 * Ref: GRChombo FourthOrderDerivatives.hpp:361-415
 */

#include "../core/fields.h"
#include "../core/grid.h"
#include "../core/params.h"
#include "../numerics/finite_diff.h"
#include <math.h>

/* Gauge fields: lapse, shift^i, B^i (indices FIELD_LAPSE through FIELD_B3).
 * All other fields are physical.
 * Ref: arXiv:2404.01137 — "epsilon_KO,CA = 0.99 for gauge, 0.3 for physical" */
static inline int is_gauge_field(int f)
{
    return f >= FIELD_LAPSE && f <= FIELD_B3;
}

#ifdef LATTICE_GPU
#pragma omp declare target
#endif
void add_ko_dissipation(double ** restrict rhs,
                        const double *const * restrict src,
                        const grid_t *g, const sim_params_t *p,
                        int i, int j, int k)
{
    int idx = IDX(g, i, j, k);
    int sx = STRIDE_X;
    int sy = STRIDE_Y(g);
    int sz = STRIDE_Z(g);
    double dx = g->dx;

    /* CAKO scaling factor: W = sqrt(chi), suppresses near punctures.
     * When disabled, W = 1 (no effect).
     * Ref: arXiv:2404.01137, Eq. (20) */
    double W = 1.0;
    if (p->noise.use_cako) {
        double chi = src[FIELD_CHI][idx];
        W = sqrt(fmax(chi, 1.0e-10));
    }

    /* Skip EM fields when EM is disabled (saves 6/31 field iterations) */
    int nf = g->n_fields;
    for (int f = 0; f < nf; f++) {
        /* Per-field sigma: gauge fields get stronger dissipation.
         * When disabled, all fields use p->sigma (existing behavior). */
        double sigma_f;
        if (p->noise.use_per_field_sigma) {
            sigma_f = is_gauge_field(f) ? p->noise.sigma_gauge
                                        : p->noise.sigma_phys;
        } else {
            sigma_f = p->sigma;
        }

        double sigma_eff = W * sigma_f;

        rhs[f][idx] += sigma_eff * (fd_ko(src[f], idx, sx, dx)
                                  + fd_ko(src[f], idx, sy, dx)
                                  + fd_ko(src[f], idx, sz, dx));
    }
}
#ifdef LATTICE_GPU
#pragma omp end declare target
#endif
