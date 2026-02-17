/*
 * Lattice — 3D Numerical Relativity
 * AMR refinement criterion: chi-gradient based.
 *
 * Flags blocks for refinement or coarsening based on the gradient of the
 * conformal factor chi, following GRChombo's ChiTaggingCriterion.
 *
 * criterion = (dx / chi^2) * sqrt(d1_chi_x^2 + d1_chi_y^2 + d1_chi_z^2)
 *
 * Ref: GRChombo Source/TaggingCriteria/ChiTaggingCriterion.hpp:31
 * Ref: arXiv:2312.05438 (chi-gradient vs truncation error comparison)
 */

#ifndef LATTICE_CRITERION_H
#define LATTICE_CRITERION_H

#include "block.h"
#include "mesh.h"
#include "../core/params.h"

/* Refinement action flags */
#define AMR_COARSEN  (-1)
#define AMR_NONE       0
#define AMR_REFINE     1

/* Per-block refinement flag */
typedef struct {
    int block_id;   /* mesh block ID */
    int action;     /* AMR_REFINE, AMR_COARSEN, or AMR_NONE */
} refine_flag_t;

/*
 * Compute the maximum chi-gradient criterion over a block's interior.
 * Uses 4th-order FD_D1 for derivatives of chi.
 * Returns max over all interior cells of:
 *   (dx / chi^2) * sqrt(d1x^2 + d1y^2 + d1z^2)
 */
double chi_gradient_max(const block_t *b);

/*
 * Check a single block against refinement/coarsening thresholds.
 * Returns AMR_REFINE, AMR_COARSEN, or AMR_NONE.
 */
int criterion_check_block(const block_t *b, double chi_refine, double chi_coarsen);

/*
 * Check all leaf blocks in the mesh and fill flags array.
 * Returns the number of flags written (one per leaf block).
 * max_flags: capacity of flags array.
 */
int criterion_check_mesh(const mesh_t *m, const amr_params_t *ap,
                         refine_flag_t *flags, int max_flags);

#endif /* LATTICE_CRITERION_H */
