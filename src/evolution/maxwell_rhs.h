/*
 * Lattice — 3D Numerical Relativity
 * Maxwell evolution equations in 3+1 form.
 *
 * Evolves conformal E^i and B^i fields coupled to CCZ4 spacetime.
 * Ref: arXiv:0907.1151 (Alcubierre et al.), Eqs. (23)-(24)
 */

#ifndef LATTICE_MAXWELL_RHS_H
#define LATTICE_MAXWELL_RHS_H

#include "../core/device.h"
#include "../core/grid.h"
#include "../core/params.h"

EXTERN_C_BEGIN

/* Compute Maxwell RHS at a single grid point.
 * Writes RHS for FIELD_E1..E3 and FIELD_BM1..BM3.
 * Ref: arXiv:0907.1151 Eqs. (23)-(24) */
LATTICE_DEVICE
void maxwell_rhs_point(double ** restrict rhs,
                       const double *const * restrict src,
                       const grid_t *g, const sim_params_t *p,
                       int i, int j, int k);

/* Combined CCZ4 + Maxwell RHS at a single grid point.
 * Calls ccz4_rhs_point (with EM source terms if em_enabled),
 * then maxwell_rhs_point for the 6 EM fields. */
LATTICE_DEVICE
void ccz4_maxwell_rhs_point(double ** restrict rhs,
                             const double *const * restrict src,
                             const grid_t *g, const sim_params_t *p,
                             int i, int j, int k);

/* EM stress-energy quantities at a single point.
 * Computes energy density rho, momentum j^i, stress tensor S_ij,
 * and stress trace S from the EM fields.
 * Uses conformal metric h_ij, chi to raise/lower indices.
 *
 * Ref: arXiv:0907.1151 Eq. (5) (EM stress-energy tensor) */
LATTICE_DEVICE
void em_stress_energy(const double *const * restrict src, const grid_t *g,
                      int idx,
                      double chi, const double h_UU[3][3],
                      const double h[3][3],
                      double *rho_em, double j_em[3],
                      double S_em_dd[3][3], double *S_em_trace);

EXTERN_C_END

#endif /* LATTICE_MAXWELL_RHS_H */
