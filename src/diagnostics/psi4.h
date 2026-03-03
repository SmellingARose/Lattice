/*
 * Lattice — 3D Numerical Relativity
 * Psi4 gravitational wave extraction.
 *
 * Newman-Penrose scalar Psi4 = C_{abcd} n^a mbar^b n^c mbar^d
 * encodes outgoing gravitational radiation. r*Psi4 → d²h/dt² at infinity.
 *
 * Computed from the electric (E_ij) and magnetic (B_ij) parts of the
 * Weyl tensor in the 3+1 decomposition, projected onto a null tetrad.
 *
 * Mode decomposition into spin-weighted spherical harmonics _{-2}Y_{lm}
 * using Gauss-Legendre × trapezoidal quadrature.
 *
 * Ref: B&S "Numerical Relativity" §8.3 (Weyl scalars)
 * Ref: GRChombo Source/CCZ4/Weyl4.impl.hpp
 */

#ifndef LATTICE_PSI4_H
#define LATTICE_PSI4_H

#include "../core/grid.h"
#include "../core/device.h"

EXTERN_C_BEGIN

struct mesh_s;  /* forward declaration */

typedef struct psi4_workspace_s {
    int    n_theta, n_phi, l_max;
    double radius, center[3];
    double *theta, *gl_weights;     /* Gauss-Legendre nodes/weights [n_theta] */
    double *re_psi4, *im_psi4;     /* r*Psi4 on sphere [n_theta * n_phi]     */
    int    n_modes;
    double *mode_re, *mode_im;     /* (l,m) modes [n_modes]                  */
} psi4_workspace_t;

/* Allocate workspace for Psi4 extraction.
 *   n_theta, n_phi: angular resolution of the extraction sphere
 *   l_max: maximum l for mode decomposition (>= 2)
 *   radius: extraction sphere radius
 *   center[3]: coordinate center of the sphere */
psi4_workspace_t *psi4_alloc(int n_theta, int n_phi, int l_max,
                              double radius, const double center[3]);

/* Free workspace */
void psi4_free(psi4_workspace_t *ws);

/* Compute Psi4 at a grid point using FD stencils.
 * out[0] = Re(Psi4), out[1] = Im(Psi4).
 * center[3] = coordinate center for the null tetrad construction.
 * Ref: B&S Eq. (8.53)-(8.55) */
LATTICE_DEVICE
void psi4_at_point(const double *const *fields, const grid_t *g,
                   int i, int j, int k,
                   const double center[3], double out[2]);

/* Extract Psi4 on the sphere and decompose into _{-2}Y_{lm} modes.
 * Fills ws->re_psi4, ws->im_psi4 (r*Psi4 on sphere),
 * and ws->mode_re, ws->mode_im (mode coefficients). */
void psi4_extract(psi4_workspace_t *ws, const struct mesh_s *m);

/* Decompose r*Psi4 on sphere into spin-weighted spherical harmonic modes.
 * Reads ws->re_psi4, ws->im_psi4; fills ws->mode_re, ws->mode_im.
 * Separated from psi4_extract for GPU path (sphere computed on device,
 * mode decomposition on host). */
void psi4_decompose_modes(psi4_workspace_t *ws);

/* Append mode data to CSV file.
 * Format: t, l, m, Re(rPsi4), Im(rPsi4), |rPsi4|, phase */
void psi4_write_modes(const psi4_workspace_t *ws, double time,
                      const char *filename);

EXTERN_C_END

#endif /* LATTICE_PSI4_H */
