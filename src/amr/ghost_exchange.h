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
 * Fill ghost zones for a specific field array (not necessarily fields[]).
 * Used during RK stages when operating on intermediate states.
 * src_field selects which array to exchange: 0 = fields, 1 = scratch.
 */
void ghost_exchange_array(mesh_t *m, int src_field);

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

#endif /* LATTICE_GHOST_EXCHANGE_H */
