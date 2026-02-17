/*
 * Lattice — 3D Numerical Relativity
 * AMR block refinement and coarsening (oct-tree split/merge).
 *
 * mesh_refine_block:     split 1 block → 8 children (prolongate data)
 * mesh_coarsen_siblings: merge 8 children → 1 parent (restrict data)
 * mesh_enforce_2to1:     cascade 2:1 level constraint
 * mesh_regrid:           full regrid cycle (criterion + refine/coarsen)
 *
 * Ref: Athena++ src/mesh/mesh.cpp AdaptiveMeshRefinement()
 * Ref: Athena++ src/mesh/meshblock_tree.cpp (tree split/merge)
 */

#ifndef LATTICE_REFINE_H
#define LATTICE_REFINE_H

#include "mesh.h"
#include "criterion.h"
#include "../core/params.h"

/*
 * Refine a single block: split 1 → 8 children.
 * 1. Creates 8 child blocks at level+1 with dx/2
 * 2. Prolongates parent data into children
 * 3. Marks parent as non-leaf (retains grid for restriction)
 * 4. Sets parent child_ids and children parent_id
 *
 * Returns 0 on success, -1 on failure.
 */
int mesh_refine_block(mesh_t *m, int block_id);

/*
 * Coarsen siblings: merge 8 children → 1 parent.
 * 1. Verifies all 8 children are leaves
 * 2. Restricts children data into parent
 * 3. Frees children and marks parent as leaf
 *
 * parent_id: ID of the non-leaf parent whose children are merged.
 * Returns 0 on success, -1 on failure.
 */
int mesh_coarsen_siblings(mesh_t *m, int parent_id);

/*
 * Enforce the 2:1 level constraint.
 * For each block flagged for refinement, cascade to ensure no neighbor
 * differs by more than 1 level. Converges iteratively.
 * For coarsening, blocks are unflagged if they would violate 2:1.
 *
 * Modifies flags[] in-place and updates n_flags.
 */
void mesh_enforce_2to1(mesh_t *m, refine_flag_t *flags, int *n_flags);

/*
 * Full regrid cycle:
 * 1. Evaluate chi-gradient criterion on all leaf blocks
 * 2. Enforce 2:1 constraint
 * 3. Refine flagged blocks
 * 4. Coarsen flagged blocks (where all siblings agree)
 * 5. Compact mesh and rebuild neighbors
 *
 * Returns the number of blocks added/removed (positive = net refinement).
 */
int mesh_regrid(mesh_t *m, const amr_params_t *ap);

#endif /* LATTICE_REFINE_H */
