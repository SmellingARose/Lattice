/*
 * Lattice — 3D Numerical Relativity
 * Bowen-York initial data with momentum and spin.
 *
 * Computes the Bowen-York extrinsic curvature A_tilde_ij (York conformal
 * weight +2) for punctures with linear momentum P_i and spin S_i, then solves the
 * Hamiltonian constraint for the conformal factor via FAS multigrid.
 *
 * Ref: gr-qc/9703066 (Brandt-Brugmann puncture method)
 * Ref: GRChombo BoostedBH.impl.hpp:47-50 (BY A_ij formula)
 * Ref: TwoPunctures Equations.cc:BY_Aijofxyz()
 */

#ifndef LATTICE_BOWEN_YORK_H
#define LATTICE_BOWEN_YORK_H

#include "../core/grid.h"
#include "../core/params.h"
#include "../amr/block.h"
#include "../amr/mesh.h"

/* Compute Bowen-York A_tilde_ij at point (x,y,z) by summing over
 * all punctures.  A_tilde[3][3] is the output (symmetric, York weight +2). */
void bowen_york_Aij(double A_tilde[3][3], double x, double y, double z,
                    int n_bh, const puncture_data_t *bhs);

/* Trace A_ij A^ij with flat metric (A raised/lowered by delta_ij).
 * Correct for conformally flat BY data where h_ij = delta_ij. */
double bowen_york_A2(const double A_tilde[3][3]);

/* Trace A_ij A^ij with conformal metric: h^{ik} h^{jl} A_{kl} A_{ij}.
 * Required for HiSpID where h_ij != delta_ij. */
double hispid_A2(const double A[3][3], const double h[3][3]);

/* Brill-Lindquist conformal factor: psi = 1 + sum(M/(2r)). */
double brill_lindquist_psi(double x, double y, double z,
                           int n_bh, const puncture_data_t *bhs);

/* Convert solved psi + BY A_ij to all 25 CCZ4 fields on the grid.
 * psi_arr[npoints] is the full conformal factor (BL + correction u).
 * Ref: GRChombo BinaryBH.impl.hpp:53-68 */
void set_ccz4_from_psi(grid_t *g, const double *psi_arr,
                        int n_bh, const puncture_data_t *bhs);

/* Convert solved psi + V^i + non-flat h_ij to all 25 CCZ4 fields.
 * psi_arr[npoints]: full conformal factor (BL + correction u).
 * V_arr[3]: momentum correction arrays (may be NULL for V=0).
 * h_ij from hispid_conformal_metric(), Gamma^i computed via FD.
 *
 * Key difference from set_ccz4_from_psi: h_ij != delta_ij, Gamma^i != 0.
 * Ref: arXiv:1410.8607, GRChombo KerrBH.impl.hpp:86-93 */
void set_ccz4_from_hispid(grid_t *g, const double *psi_arr,
                           double *const *V_arr,
                           int n_bh, const puncture_data_t *bhs);

/* Set --hispid CLI override (forces HiSpID even for low spin) */
void set_hispid_override(int val);

/*
 * Block-aware CCZ4 conversion: convert solver data in a block's arrays
 * to full CCZ4 fields.  Reads solver slots (0-9), writes CCZ4 slots (0-30).
 * Read-before-write at each point avoids aliasing.
 * n_fields controls whether EM fields (slots 25-30) are written.
 *
 * Ref: GRChombo BinaryBH.impl.hpp:53-68
 */
void set_ccz4_from_psi_block(block_t *blk, int n_bh, const puncture_data_t *bhs,
                              int n_fields);

/*
 * Block-aware HiSpID CCZ4 conversion.
 * Two passes: (1) set chi, h_ij, K, A_ij, gauge, EM
 *             (2) compute Gamma^i from FD of h_ij
 * n_fields controls whether EM fields are written.
 *
 * Ref: arXiv:1410.8607, GRChombo KerrBH.impl.hpp:86-93
 */
void set_ccz4_from_hispid_block(block_t *blk, int n_bh, const puncture_data_t *bhs,
                                 int n_fields);

/*
 * Set initial data on an AMR evolution mesh.
 * Dispatches: analytic BL, BY solver, or HiSpID solver.
 * Solves constraints directly on the mesh blocks, then converts
 * solver data → CCZ4 in-place.  Zero interpolation error.
 *
 * Ref: Tomida & Stone 2023 (Athena++ MG self-gravity on evolution mesh)
 * Ref: arXiv:0912.2920 (Alic et al., FD constraint violation vs spectral)
 */
void set_bowen_york_mesh(mesh_t *m, int n_bh, const puncture_data_t *bhs,
                          int n_amr_levels);
/* Extended version with custom refinement geometry */
void set_bowen_york_mesh_ex(mesh_t *m, int n_bh, const puncture_data_t *bhs,
                             int n_amr_levels,
                             double refine_c, double refine_beta);

#endif /* LATTICE_BOWEN_YORK_H */
