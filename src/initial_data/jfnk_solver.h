/*
 * Lattice — 3D Numerical Relativity
 * Covering grid FAS multigrid constraint solver.
 *
 * Level-by-level solve on AMR mesh: each level solved independently
 * from coarse to fine, with converged coarser levels providing fixed
 * Dirichlet BCs.  Each level's blocks are gathered into a single
 * temporary uniform grid, solved by proven single-grid FAS (FMG +
 * V-cycles + 8-color Newton-GS smoother), then scattered back.
 * No inter-block ghost exchange during MG — zero risk of cross-level
 * corruption.  FMG converges in 1 pass per level.
 *
 * File retains jfnk_solver name for API compatibility (replaces the
 * old composite FAS and JFNK+BiCGSTAB approaches).
 *
 * Ref: arXiv:0705.1486 (Natchu & Matzner, 4th-order MG for BH data)
 * Ref: arXiv:2510.11152 (GPU FAS multigrid, 8-color MCGS, 61x speedup)
 */

#ifndef LATTICE_JFNK_SOLVER_H
#define LATTICE_JFNK_SOLVER_H

#include "../core/grid.h"
#include "../core/params.h"
#include "../amr/mesh.h"

/*
 * Solve the 1-field Hamiltonian constraint directly on an external mesh.
 * The mesh is refined in-place (caller owns it, solver borrows it).
 * Returns the final residual.
 *
 * Requires RK_CLASSIC (accum arrays must be allocated).
 */
double jfnk_solve_mesh(mesh_t *m, int n_bh, const puncture_data_t *bhs,
                        double tol, int max_iter, int verbose,
                        int n_amr_levels);

/*
 * Solve the 4-field coupled constraints (HiSpID) directly on an external mesh.
 * Same semantics as jfnk_solve_mesh but for the coupled system.
 * Returns the final max(||r_psi||, ||r_V||) residual.
 */
double jfnk_solve_mesh_coupled(mesh_t *m, int n_bh, const puncture_data_t *bhs,
                                double tol, int max_iter, int verbose,
                                int n_amr_levels);

/*
 * Grid-based wrappers (for unit tests — creates 1-block mesh internally).
 * Solve the Hamiltonian constraint on grid g for n_bh Bowen-York punctures.
 * Returns the final residual.  Sets all 25 CCZ4 fields on g.
 */
double jfnk_solve(grid_t *g, int n_bh, const puncture_data_t *bhs,
                   double tol, int max_iter, int verbose);

/*
 * Grid-based wrapper for 4-field coupled (HiSpID).
 * Returns the final max(||r_psi||, ||r_V||) residual.
 */
double jfnk_solve_coupled(grid_t *g, int n_bh, const puncture_data_t *bhs,
                            double tol, int max_iter, int verbose);

/*
 * AMR variants with explicit AMR levels (grid-based).
 */
double jfnk_solve_amr(grid_t *g, int n_bh, const puncture_data_t *bhs,
                       double tol, int max_iter, int verbose,
                       int n_amr_levels);

double jfnk_solve_coupled_amr(grid_t *g, int n_bh, const puncture_data_t *bhs,
                               double tol, int max_iter, int verbose,
                               int n_amr_levels);

/*
 * Refine an existing mesh near punctures.
 * Adds n_amr_levels of refinement around each BH position.
 *
 * Ref: Athena++ MeshRefinement pattern.
 */
void refine_mesh_near_punctures(mesh_t *m, int n_amr_levels,
                                int n_bh, const puncture_data_t *bhs);

#endif /* LATTICE_JFNK_SOLVER_H */
