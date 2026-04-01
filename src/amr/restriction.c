/*
 * Lattice — 3D Numerical Relativity
 * Trilinear restriction (fine → coarse).
 *
 * For 2:1 refinement: simple cell averaging of 2×2×2 = 8 fine cells.
 * All weights positive (1/8 each), no Gibbs oscillations near punctures.
 *
 * Ref: GRChombo CoarseAverage (Chombo library) — uses simple averaging.
 */

#include "restriction.h"
#include "block.h"
#include "../core/fields.h"

void restrict_field(const grid_t *fine_g, int ff,
                    grid_t *coarse_g, int cf)
{
    const int ghost_c = coarse_g->ghost;
    const int N_c = coarse_g->N;
    const int ghost_f = fine_g->ghost;
    const double *src = fine_g->fields[ff];

    for (int ck = ghost_c; ck < ghost_c + N_c; ck++) {
        for (int cj = ghost_c; cj < ghost_c + N_c; cj++) {
            for (int ci = ghost_c; ci < ghost_c + N_c; ci++) {
                int fi_base = 2 * (ci - ghost_c) + ghost_f;
                int fj_base = 2 * (cj - ghost_c) + ghost_f;
                int fk_base = 2 * (ck - ghost_c) + ghost_f;

                coarse_g->fields[cf][IDX(coarse_g, ci, cj, ck)] =
                    restrict_cell(src, fine_g, fi_base, fj_base, fk_base);
            }
        }
    }
}

void restrict_to_coarse_buf(struct block_s *b)
{
    if (!b || !b->coarse_buf) return;

    const grid_t *fg = b->grid;
    grid_t *cg = b->coarse_buf;
    const int ghost_f = fg->ghost;
    const int ghost_c = cg->ghost;
    const int N_c = cg->N;

    for (int f = 0; f < fg->n_fields; f++) {
        const double *src = fg->fields[f];

        for (int ck = ghost_c; ck < ghost_c + N_c; ck++) {
            for (int cj = ghost_c; cj < ghost_c + N_c; cj++) {
                for (int ci = ghost_c; ci < ghost_c + N_c; ci++) {
                    int fi_base = 2 * (ci - ghost_c) + ghost_f;
                    int fj_base = 2 * (cj - ghost_c) + ghost_f;
                    int fk_base = 2 * (ck - ghost_c) + ghost_f;

                    cg->fields[f][IDX(cg, ci, cj, ck)] =
                        restrict_cell(src, fg, fi_base, fj_base, fk_base);
                }
            }
        }
    }
}
