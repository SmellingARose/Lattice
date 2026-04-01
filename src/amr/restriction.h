/*
 * Lattice — 3D Numerical Relativity
 * Trilinear restriction (fine → coarse).
 *
 * For 2:1 refinement, each coarse cell overlaps 2×2×2 = 8 fine cells.
 * Trilinear interpolation at the coarse cell center gives equal weights
 * of 1/8 to each — equivalent to simple cell averaging. All weights
 * are positive, eliminating Gibbs oscillations near non-smooth features
 * (puncture singularities, C^0 fields).
 *
 * Generalizes to arbitrary refinement ratios: for ratio r, the coarse
 * cell center sits at a general position in the fine grid. Trilinear
 * interpolation of the 2×2×2 surrounding fine cells gives fractional
 * weights that are always in [0,1] and sum to 1.
 *
 * Ref: GRChombo CoarseAverage (Chombo library) — uses simple averaging.
 * Ref: arXiv:2112.10567 (GRChombo AMR lessons) — averaging is standard.
 */

#ifndef LATTICE_RESTRICTION_H
#define LATTICE_RESTRICTION_H

#include "../core/device.h"
#include "../core/grid.h"

EXTERN_C_BEGIN

/* Forward declaration for restrict_to_coarse_buf */
struct block_s;

/*
 * Restrict a single coarse cell from fine data (trilinear / cell averaging).
 * fi_base, fj_base, fk_base: fine grid indices of the first direct child.
 *
 * For 2:1 refinement: averages the 2×2×2 = 8 fine cells at
 * [fi_base, fi_base+1] × [fj_base, fj_base+1] × [fk_base, fk_base+1].
 * No ghost zone access beyond direct children — stencil stays within
 * the fine block interior. All weights positive (1/8 each).
 */
static inline double restrict_cell(const double *src, const grid_t *fg,
                                    int fi_base, int fj_base, int fk_base)
{
    double val = 0.0;
    for (int dk = 0; dk < 2; dk++)
        for (int dj = 0; dj < 2; dj++)
            for (int di = 0; di < 2; di++)
                val += src[IDX(fg, fi_base + di, fj_base + dj, fk_base + dk)];
    return val * 0.125;
}

/* Restrict a single field from fine grid to coarse grid.
 * Fine grid has 2x the resolution (N_fine = 2 * N_coarse).
 * Same physical domain. Fills coarse interior points. */
void restrict_field(const grid_t *fine_g, int fine_field,
                    grid_t *coarse_g, int coarse_field);

/* Restrict fine block interior → block's own coarse_buf interior.
 * Block-local operation: no cross-block memory access.
 * Ref: AthenaK coarse-buffer architecture */
void restrict_to_coarse_buf(struct block_s *b);

EXTERN_C_END

#endif /* LATTICE_RESTRICTION_H */
