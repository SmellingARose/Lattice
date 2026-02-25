/*
 * Lattice — 3D Numerical Relativity
 * AMR FAS Multigrid constraint solver.
 *
 * Two-tier hierarchy: composite multigrid on AMR blocks above, existing
 * uniform FAS multigrid below.  The base AMR level (level 0) is a full
 * uniform grid at the same resolution as the current solver — finer AMR
 * levels add resolution near punctures.
 *
 * Ref: arXiv:0705.1486 (Natchu & Matzner, 4th-order MG for BH data)
 * Ref: arXiv:2510.11152 (GPU FAS multigrid, 8-color MCGS)
 * Ref: Trottenberg et al., Multigrid Methods (composite AMR MG)
 */

#ifndef LATTICE_RELAXATION_AMR_H
#define LATTICE_RELAXATION_AMR_H

#include "../core/grid.h"
#include "../core/params.h"
#include "../amr/mesh.h"

/*
 * Solve the Hamiltonian constraint on grid g using AMR multigrid.
 * n_amr_levels additional levels above the base resolution.
 * Returns the final residual.  Sets all 25 CCZ4 fields on g.
 *
 * Falls back to relaxation_solve() if n_amr_levels == 0.
 */
double relaxation_solve_amr(grid_t *g, int n_bh, const puncture_data_t *bhs,
                             double tol, int max_iter, int verbose,
                             int n_amr_levels);

/*
 * Solve the coupled Hamiltonian + 3 momentum constraints (HiSpID)
 * using AMR multigrid.  4-field system: (psi, V^1, V^2, V^3).
 * Returns the final max(||r_psi||, ||r_V||) residual.
 *
 * Falls back to relaxation_solve_coupled() if n_amr_levels == 0.
 */
double relaxation_solve_coupled_amr(grid_t *g, int n_bh,
                                     const puncture_data_t *bhs,
                                     double tol, int max_iter, int verbose,
                                     int n_amr_levels);

/*
 * Refine an existing mesh near punctures.
 * Adds n_amr_levels of refinement around each BH position.
 * Works on any mesh (evolution or solver-owned).
 *
 * Ref: Athena++ MeshRefinement pattern.
 */
void refine_mesh_near_punctures(mesh_t *m, int n_amr_levels,
                                int n_bh, const puncture_data_t *bhs);

/*
 * Solve the 1-field Hamiltonian constraint directly on an external mesh.
 * The mesh is refined in-place (caller owns it, solver borrows it).
 * Solver field data is stored in the mesh's block arrays (fields/rhs/scratch/accum).
 * Returns the final residual.  Solver data left in fields[SOL_PSI], fields[BG_PSI_BL].
 *
 * Requires RK_CLASSIC (accum arrays must be allocated).
 */
double relaxation_solve_amr_mesh(mesh_t *m, int n_bh, const puncture_data_t *bhs,
                                  double tol, int max_iter, int verbose,
                                  int n_amr_levels);

/*
 * Solve the 4-field coupled constraints (HiSpID) directly on an external mesh.
 * Same semantics as relaxation_solve_amr_mesh but for the coupled system.
 * Returns the final max(||r_psi||, ||r_V||) residual.
 */
double relaxation_solve_coupled_amr_mesh(mesh_t *m, int n_bh,
                                          const puncture_data_t *bhs,
                                          double tol, int max_iter, int verbose,
                                          int n_amr_levels);

#endif /* LATTICE_RELAXATION_AMR_H */
