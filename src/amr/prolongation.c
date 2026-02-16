/*
 * Lattice — 3D Numerical Relativity
 * 4th-order cell-centered Lagrange prolongation (coarse → fine).
 *
 * Tensor-product interpolation: for each coarse interior cell, compute
 * 8 fine children using a 5×5×5 stencil of coarse values weighted by
 * the 1D Lagrange coefficients.
 *
 * Ref: AthenaK src/mesh/prolongation.hpp (HighOrderProlongCC)
 * Ref: AthenaK src/mesh/mesh_refinement.cpp InitInterpWghts()
 */

#include "prolongation.h"
#include "../core/fields.h"

/*
 * 4th-order cell-centered Lagrange weights for left child (x = -1/4).
 * Coarse cells at positions {-2, -1, 0, +1, +2} relative to parent.
 *
 * Ref: AthenaK mesh_refinement.cpp lines ~1300-1320
 *   pro_4th[0] = -45./2048.
 *   pro_4th[1] = 105./512.
 *   pro_4th[2] = 945./1024.
 *   pro_4th[3] = -63./512.
 *   pro_4th[4] = 35./2048.
 */
const double prolong_w[PROLONG_STENCIL] = {
    -45.0 / 2048.0,     /* -0.02197265625 */
     105.0 / 512.0,     /*  0.20507812500 */
     945.0 / 1024.0,    /*  0.92285156250 */
     -63.0 / 512.0,     /* -0.12304687500 */
      35.0 / 2048.0     /*  0.01708984375 */
};

void prolongate_field(const grid_t *coarse_g, int cf,
                      grid_t *fine_g, int ff)
{
    const int ghost = coarse_g->ghost;
    const int N_c = coarse_g->N;
    const int half = PROLONG_STENCIL / 2;  /* = 2 */
    const double *src = coarse_g->fields[cf];

    /* Loop over coarse interior cells */
    for (int ck = ghost; ck < ghost + N_c; ck++) {
        for (int cj = ghost; cj < ghost + N_c; cj++) {
            for (int ci = ghost; ci < ghost + N_c; ci++) {

                /* 8 fine children: offset (oi, oj, ok) ∈ {0,1}^3 */
                for (int ok = 0; ok < 2; ok++) {
                    for (int oj = 0; oj < 2; oj++) {
                        for (int oi = 0; oi < 2; oi++) {

                            /* Fine cell index:
                             * fi = 2*(ci - ghost) + ghost + oi */
                            int fi = 2 * (ci - ghost) + ghost + oi;
                            int fj = 2 * (cj - ghost) + ghost + oj;
                            int fk = 2 * (ck - ghost) + ghost + ok;

                            /* 3D tensor product of 5-point stencils.
                             * Ref: AthenaK prolongation.hpp ProlongInterpolation */
                            double val = 0.0;
                            for (int sk = 0; sk < PROLONG_STENCIL; sk++) {
                                int wk = ok ? (PROLONG_STENCIL - 1 - sk) : sk;
                                for (int sj = 0; sj < PROLONG_STENCIL; sj++) {
                                    int wj = oj ? (PROLONG_STENCIL - 1 - sj) : sj;
                                    double wkj = prolong_w[wk] * prolong_w[wj];
                                    for (int si = 0; si < PROLONG_STENCIL; si++) {
                                        int wi = oi ? (PROLONG_STENCIL - 1 - si) : si;
                                        int src_idx = IDX(coarse_g,
                                                          ci - half + si,
                                                          cj - half + sj,
                                                          ck - half + sk);
                                        val += wkj * prolong_w[wi] * src[src_idx];
                                    }
                                }
                            }

                            fine_g->fields[ff][IDX(fine_g, fi, fj, fk)] = val;
                        }
                    }
                }
            }
        }
    }
}

void prolongate_all(const grid_t *coarse_g, grid_t *fine_g)
{
    for (int f = 0; f < NUM_FIELDS; f++) {
        prolongate_field(coarse_g, f, fine_g, f);
    }
}
