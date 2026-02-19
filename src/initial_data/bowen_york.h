/*
 * Lattice — 3D Numerical Relativity
 * Bowen-York initial data with momentum and spin.
 *
 * Computes the physical traceless extrinsic curvature A_ij^phys for
 * punctures with linear momentum P_i and spin S_i, then solves the
 * Hamiltonian constraint for the conformal factor via hyperbolic relaxation.
 *
 * Ref: gr-qc/9703066 (Brandt-Brugmann puncture method)
 * Ref: GRChombo BoostedBH.impl.hpp:47-50 (BY A_ij formula)
 * Ref: TwoPunctures Equations.cc:BY_Aijofxyz()
 */

#ifndef LATTICE_BOWEN_YORK_H
#define LATTICE_BOWEN_YORK_H

#include "../core/grid.h"
#include "../core/params.h"

/* Compute physical Bowen-York A_ij at point (x,y,z) by summing over
 * all punctures.  A_phys[3][3] is the output (symmetric). */
void bowen_york_Aij(double A_phys[3][3], double x, double y, double z,
                    int n_bh, const puncture_data_t *bhs);

/* Trace A_ij A^ij with flat metric (A raised/lowered by delta_ij). */
double bowen_york_A2(const double A_phys[3][3]);

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

/* Top-level dispatch: if all P=0 and S=0, uses fast BL path.
 * Otherwise runs hyperbolic relaxation solver.
 * If hispid_override is set (via --hispid), forces coupled solver. */
void set_bowen_york(grid_t *g, int n_bh, const puncture_data_t *bhs);

/* Force HiSpID path (coupled solver with non-flat conformal metric).
 * Called by set_bowen_york when high spin detected or --hispid flag set. */
void set_hispid(grid_t *g, int n_bh, const puncture_data_t *bhs);

/* Set --hispid CLI override (forces HiSpID even for low spin) */
void set_hispid_override(int val);

#endif /* LATTICE_BOWEN_YORK_H */
