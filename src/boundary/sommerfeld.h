/*
 * Lattice — 3D Numerical Relativity
 * Sommerfeld radiative boundary conditions.
 *
 * Ref: GRChombo Source/GRChomboCore/BoundaryConditions.cpp:593-661
 */

#ifndef LATTICE_SOMMERFELD_H
#define LATTICE_SOMMERFELD_H

#include "../core/grid.h"
#include "../core/params.h"

/* ---- Helpers exposed for packed GPU kernels ----
 * These functions are used by backend_compute_sommerfeld_packed to apply
 * Sommerfeld BCs on batched pack data. Declared with omp declare target
 * so they're available on both CPU and GPU.
 *
 * asymptotic_value: returns the falloff target for each field
 *   (1.0 for chi, h_ii, lapse; 0.0 for everything else)
 * boundary_d1: one-sided or centered finite difference at boundary
 *   (forward/backward stencil near edges, centered in interior)
 */
#ifdef LATTICE_GPU
#pragma omp declare target
#endif
double asymptotic_value(int field);
double boundary_d1(const double *f, int idx, int stride,
                   int lo_offset, int hi_offset, double dx);
#ifdef LATTICE_GPU
#pragma omp end declare target
#endif

#endif /* LATTICE_SOMMERFELD_H */
