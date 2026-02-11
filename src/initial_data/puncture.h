/*
 * Lattice — 3D Numerical Relativity
 * Puncture initial data: flat spacetime and Brill-Lindquist.
 *
 * Ref: gr-qc/9703066 (Brandt-Brugmann puncture method)
 */

#ifndef LATTICE_PUNCTURE_H
#define LATTICE_PUNCTURE_H

#include "../core/grid.h"

/* Set Minkowski (flat) initial data: chi=1, h=I, lapse=1, rest=0 */
void set_flat_spacetime(grid_t *g);

/* Brill-Lindquist multi-puncture data (at rest).
 * psi = 1 + sum_n( M_n / (2*r_n) )
 * chi = psi^{-4}, h_{ij} = delta_{ij}, lapse = sqrt(chi)
 *
 * n_bh:      number of punctures
 * masses:    array of bare masses [n_bh]
 * centers:   array of positions [n_bh][3]
 */
void set_brill_lindquist(grid_t *g, int n_bh,
                         const double *masses,
                         const double centers[][3]);

#endif /* LATTICE_PUNCTURE_H */
