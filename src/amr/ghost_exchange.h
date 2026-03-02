/*
 * Lattice — 3D Numerical Relativity
 * AMR ghost zone exchange: 26-neighbor (6 face + 12 edge + 8 corner).
 *
 * Single parallel pass over all neighbor pairs. For each of the 26
 * neighbor types, copy the appropriate slab/column/cube from the
 * neighbor's interior into the block's ghost zone. All 25 fields
 * x GHOST_WIDTH layers.
 *
 * For each direction d with neighbor offset o:
 *   o == -1: dst = [0, ghost),          src = [ghost+N-ghost, ghost+N)
 *   o ==  0: dst = [ghost, ghost+N),    src = [ghost, ghost+N)
 *   o == +1: dst = [ghost+N, Ntotal),   src = [ghost, ghost+ghost)
 *
 * This handles all 26 cases uniformly: faces copy slabs (GW x N x N),
 * edges copy columns (GW x GW x N), corners copy cubes (GW x GW x GW).
 *
 * Stage 2: same-level exchange only. Stage 4 adds cross-level
 * (prolongation/restriction into ghost zones).
 *
 * Ref: Athena++ src/bvals/bvals_cc.cpp (CellCenteredBoundaryVariable)
 * Ref: AthenaK  src/bvals/ (GPU ghost exchange patterns)
 */

#ifndef LATTICE_GHOST_EXCHANGE_H
#define LATTICE_GHOST_EXCHANGE_H

#include "mesh.h"

/*
 * Fill ghost zones for all leaf blocks via same-level neighbor exchange.
 * Must be called before each RHS evaluation in the RK step.
 *
 * Boundary ghost zones (where neighbor_id == -1) are NOT touched;
 * those are handled by apply_sommerfeld_block() after the RHS.
 */
void ghost_exchange(mesh_t *m);

/*
 * Like ghost_exchange() but includes non-leaf blocks.
 * Required for composite multigrid solvers where the V-cycle operates
 * on ALL blocks at each level (not just leaves).  Without this, non-leaf
 * blocks in multi-root meshes have unfilled ghost zones, causing the FD
 * operator to read stale data and the solver to diverge.
 */
void ghost_exchange_all_blocks(mesh_t *m);

/*
 * Multi-level ghost exchange for AMR meshes with refinement.
 * Three phases:
 *   Phase 1: Restrict fine leaf data → non-leaf parents (top-down)
 *   Phase 2: Same-level exchange at each level
 *   Phase 3: Prolongate coarse data → fine ghost zones (cross-level fill)
 *
 * Falls back to ghost_exchange() for uniform meshes (max_level == 0).
 */
void ghost_exchange_multilevel(mesh_t *m);

/*
 * Like ghost_exchange_multilevel() but processes ALL blocks (not just leaves).
 * Required for composite multigrid solvers where the V-cycle operates on
 * both leaf and non-leaf blocks. Uses the coarse-buffer protocol to correctly
 * interpolate ghost zones at coarse-fine refinement boundaries.
 */
void ghost_exchange_multilevel_all(mesh_t *m);

/*
 * Same-level ghost exchange for ALL blocks at a specific level.
 * Only exchanges between blocks at the same level — no cross-level
 * interpolation. For use between Gauss-Seidel colors in multigrid
 * smoothing, where CF boundary values are held fixed as Dirichlet BCs.
 * Ref: AMReX MLMG, Chombo AMRMultiGrid — same-level only between sweeps.
 */
void ghost_exchange_same_level_all(mesh_t *m, int level);

/*
 * Fill coarse-fine boundary ghost zones on blocks at a specific level
 * by interpolation from level-1 data (coarse-buffer protocol).
 * Called ONCE before smoothing begins at a level, not between colors.
 * Ref: AMReX MLMG — CF boundary = fixed Dirichlet from coarse level.
 */
void ghost_fill_cf_boundary(mesh_t *m, int level);

/*
 * Fill fine-level ghost zones from time-interpolated coarse neighbors.
 * For each fine leaf at `fine_level`, finds coarser neighbors, interpolates
 * their fields between t_old and t_new using frac = (t_fine - t_old)/dt_coarse,
 * restricts to coarse_buf, then prolongates into fine ghosts.
 *
 * Must be called BEFORE same-level ghost exchange for the fine level.
 *
 * Ref: Athena++ MeshRefinement::ProlongateBoundaries() temporal interpolation.
 * Ref: Chombo AMR::timeStep() coarse-fine boundary fill.
 */
void ghost_fill_from_coarser(mesh_t *m, int fine_level, double frac);

#endif /* LATTICE_GHOST_EXCHANGE_H */
