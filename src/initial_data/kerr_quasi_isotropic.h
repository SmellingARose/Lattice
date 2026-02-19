/*
 * Lattice — 3D Numerical Relativity
 * Quasi-isotropic Kerr metric for HiSpID (High-Spin Initial Data).
 *
 * Computes the Kerr BH spatial metric and extrinsic curvature in
 * quasi-isotropic (semi-isotropic) coordinates, converted to Cartesian.
 *
 * The conformal metric for N BHs is a Gaussian-weighted superposition:
 *   h_ij(x) = delta_ij + sum_n w_n(x) * [h_ij^Kerr_n(x) - delta_ij]
 * where w_n = exp(-r_n^2 / sigma_n^2), sigma_n = 1.5 * M_n.
 *
 * Ref: arXiv:1410.8607 (Ruchlin et al., HiSpID method)
 * Ref: arXiv:1001.4077 (Liu, Etienne, Shapiro — QI Kerr coords)
 * Ref: GRChombo KerrBH.impl.hpp (reference implementation)
 */

#ifndef LATTICE_KERR_QI_H
#define LATTICE_KERR_QI_H

#include "../core/params.h"

/* Compute the quasi-isotropic Kerr CONFORMAL 3-metric at point (x,y,z)
 * for a single Kerr BH at the origin with mass M and spin vector spin[3].
 * Returns h_ij with unit determinant (conformal metric).
 * Also returns the Kerr conformal factor psi_K via *psi_out (if non-NULL).
 *
 * spin[3] = angular momentum J_i.  Kerr parameter a = |J|/M.
 *
 * Ref: arXiv:1001.4077 Eqs. (1)-(11)
 * Ref: GRChombo KerrBH.impl.hpp:97-174 (compute_kerr) */
void kerr_qi_metric(double h[3][3], double *psi_out,
                    double x, double y, double z,
                    double M, const double spin[3]);

/* Compute the Kerr traceless conformal extrinsic curvature A_ij at (x,y,z).
 * Returns the conformal A_ij (trace-free, raised/lowered with h_ij).
 *
 * Ref: arXiv:1001.4077 Eq. (12)-(13)
 * Ref: GRChombo KerrBH.impl.hpp:152-165 */
void kerr_qi_extrinsic(double A[3][3],
                       double x, double y, double z,
                       double M, const double spin[3]);

/* Gaussian-weighted superposition of N Kerr conformal metrics:
 *   h_ij(x) = delta_ij + sum_n w_n(x) * [h_ij^Kerr_n(x) - delta_ij]
 * where w_n(x) = exp(-r_n^2 / sigma_n^2), sigma_n = 1.5 * M_n.
 *
 * Returns the superposed conformal metric (unit determinant enforced).
 *
 * Ref: arXiv:1410.8607 Eq. (15) */
void hispid_conformal_metric(double h[3][3],
                             double x, double y, double z,
                             int n_bh, const puncture_data_t *bhs);

/* Gaussian-weighted superposition of N Kerr extrinsic curvatures.
 * Uses same Gaussian weights as hispid_conformal_metric.
 *
 * Ref: arXiv:1410.8607 Eq. (16) */
void hispid_extrinsic(double A[3][3],
                      double x, double y, double z,
                      int n_bh, const puncture_data_t *bhs);

#endif /* LATTICE_KERR_QI_H */
