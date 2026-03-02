/*
 * Lattice — 3D Numerical Relativity
 * Sommerfeld radiative boundary conditions.
 *
 * Ref: GRChombo Source/GRChomboCore/BoundaryConditions.cpp:593-661
 */

#ifndef LATTICE_SOMMERFELD_H
#define LATTICE_SOMMERFELD_H

#include "../core/device.h"
#include "../core/grid.h"
#include "../core/params.h"

EXTERN_C_BEGIN

/* ---- Helpers exposed for packed GPU kernels ----
 * These functions are used by backend_compute_sommerfeld_packed to apply
 * Sommerfeld BCs on batched pack data. Annotated with LATTICE_DEVICE
 * so they're available on both CPU and GPU.
 *
 * asymptotic_value: returns the falloff target for each field
 *   (1.0 for chi, h_ii, lapse; 0.0 for everything else)
 * boundary_d1: one-sided or centered finite difference at boundary
 *   (forward/backward stencil near edges, centered in interior)
 */
LATTICE_DEVICE
double asymptotic_value(int field);

LATTICE_DEVICE
double boundary_d1(const double *f, int idx, int stride,
                   int lo_offset, int hi_offset, double inv_dx);

EXTERN_C_END

#endif /* LATTICE_SOMMERFELD_H */
