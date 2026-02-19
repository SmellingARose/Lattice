/*
 * Lattice — 3D Numerical Relativity
 * Hyperbolic relaxation solver for the Hamiltonian constraint.
 *
 * Solves: nabla^2 u + (1/8) A^2 (psi_BL + u)^{-7} = 0
 * via a damped wave equation in pseudo-time tau:
 *   d_tau u = v
 *   d_tau v = -eta * v + c^2 * [nabla^2(u) + (1/8) * A^2 * (psi_BL + u)^{-7}]
 *
 * Ref: Ruter et al. arXiv:1708.07358 (hyperbolic relaxation)
 * Ref: NRPyElliptic arXiv:2111.02424
 */

#ifndef LATTICE_RELAXATION_H
#define LATTICE_RELAXATION_H

#include "../core/grid.h"
#include "../core/params.h"

/* Solve the Hamiltonian constraint on grid g for n_bh Bowen-York punctures.
 * Returns the final ||v||_L2 residual (should be < tol on convergence).
 * Sets all 25 CCZ4 fields on g via set_ccz4_from_psi().
 *
 * Tuning (both solvers):
 *   tol      — target residual.  Discretization floor is O(dx^4):
 *              ~1e-3 at N=24, ~1e-5 at N=64, ~1e-7 at N=128.
 *              No benefit going below the floor (stagnation detection stops).
 *   max_iter — hard cap.  Stagnation detection exits early when
 *              residual improves <5% over 1000 iterations. */
double relaxation_solve(grid_t *g, int n_bh, const puncture_data_t *bhs,
                        double tol, int max_iter, int verbose);

/* Solve the coupled Hamiltonian + 3 momentum constraints for HiSpID data.
 * 4-field system: (psi, V^1, V^2, V^3) via damped wave relaxation.
 *
 * The conformal metric h_ij is non-flat (quasi-isotropic Kerr), so:
 *   - Hamiltonian source includes the conformal Ricci scalar R_tilde
 *   - Momentum constraint gives a source for the vector correction V^i
 *
 * Sets all 25 CCZ4 fields on g via set_ccz4_from_hispid().
 * Returns the final max(||v_psi||, ||v_V||) residual.
 * Same tol/max_iter tuning as relaxation_solve (see above).
 *
 * Ref: arXiv:1410.8607 (HiSpID), arXiv:1708.07358 (hyperbolic relaxation)
 */
double relaxation_solve_coupled(grid_t *g, int n_bh, const puncture_data_t *bhs,
                                double tol, int max_iter, int verbose);

#endif /* LATTICE_RELAXATION_H */
