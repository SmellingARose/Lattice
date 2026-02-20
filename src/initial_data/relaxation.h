/*
 * Lattice — 3D Numerical Relativity
 * FAS Multigrid constraint solver (FMG + Newton-Gauss-Seidel).
 *
 * Solves: nabla^2 u + S(u) = 0  via Full Multigrid with FAS V-cycles.
 * Newton-GS smoother with 8-color ordering for GPU compatibility.
 * Achieves discretization accuracy in O(N^3) total work.
 *
 * Ref: arXiv:0705.1486 (Natchu & Matzner, 4th-order MG for BH initial data)
 * Ref: arXiv:2510.11152 (GPU FAS multigrid, 8-color MCGS)
 */

#ifndef LATTICE_RELAXATION_H
#define LATTICE_RELAXATION_H

#include "../core/grid.h"
#include "../core/params.h"

/* Solve the Hamiltonian constraint on grid g for n_bh Bowen-York punctures.
 * Returns the final ||f - L(u)||_L2 residual (should be < tol on convergence).
 * Sets all 25 CCZ4 fields on g via set_ccz4_from_psi().
 *
 * Tuning (both solvers):
 *   tol      — target residual.  Discretization floor is O(dx^4):
 *              ~1e-3 at N=24, ~1e-5 at N=64, ~1e-7 at N=128.
 *   max_iter — max post-FMG V-cycles (usually 0-3 needed). */
double relaxation_solve(grid_t *g, int n_bh, const puncture_data_t *bhs,
                        double tol, int max_iter, int verbose);

/* Solve the coupled Hamiltonian + 3 momentum constraints for HiSpID data.
 * 4-field system: (psi, V^1, V^2, V^3) via FAS multigrid.
 *
 * The conformal metric h_ij is non-flat (quasi-isotropic Kerr), so:
 *   - Hamiltonian source includes the conformal Ricci scalar R_tilde
 *   - Momentum constraint gives a source for the vector correction V^i
 *
 * Sets all 25 CCZ4 fields on g via set_ccz4_from_hispid().
 * Returns the final max(||r_psi||, ||r_V||) residual.
 *
 * Ref: arXiv:1410.8607 (HiSpID), arXiv:0705.1486 (FAS multigrid)
 */
double relaxation_solve_coupled(grid_t *g, int n_bh, const puncture_data_t *bhs,
                                double tol, int max_iter, int verbose);

#endif /* LATTICE_RELAXATION_H */
