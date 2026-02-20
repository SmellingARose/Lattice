/*
 * Lattice -- 3D Numerical Relativity
 * Apparent horizon finder via hyperbolic flow method.
 *
 * Finds the apparent horizon (AH) as the outermost marginally trapped surface
 * where the outgoing null expansion vanishes: Theta = 0.
 *
 * The trial surface r = h(theta, phi) is evolved via a damped wave equation
 * in pseudo-time tau until max|Theta| < tol.
 *
 * Ref: Thornburg, CQG 4 (1987) 1119 (original AH finder formulation)
 * Ref: Thornburg, PRD 54 (1996) 4899 (flow method)
 * Ref: Gundlach, PRD 57 (1998) 863 (fast flow)
 */

#ifndef LATTICE_AH_FINDER_H
#define LATTICE_AH_FINDER_H

#include "../core/grid.h"
#include "../amr/mesh.h"

/* Workspace for the AH finder's angular grid and flow evolution.
 * The trial surface is r = h(theta_i, phi_j) on a 2D angular grid.
 * Evolved via damped wave: dh/dtau = v, dv/dtau = -eta*v - c^2*Theta(h). */
typedef struct {
    int n_theta;          /* number of theta points (including poles) */
    int n_phi;            /* number of phi points                     */
    double *h;            /* surface radius h[i*n_phi + j]            */
    double *v;            /* velocity field v[i*n_phi + j]            */
    double *rhs_h;        /* RK4 RHS for h                            */
    double *rhs_v;        /* RK4 RHS for v                            */
    double *scratch_h;    /* RK4 scratch for h                        */
    double *scratch_v;    /* RK4 scratch for v                        */
    double *accum_h;      /* RK4 accumulator for h                    */
    double *accum_v;      /* RK4 accumulator for v                    */
    double *theta_arr;    /* expansion Theta at each angular point     */
    double center[3];     /* center of the trial surface               */
    double eta;           /* damping coefficient (default 5.0)         */
    double c_wave;        /* wave speed coefficient (default 1.0)      */
} ah_workspace_t;

/* Results from a found apparent horizon */
typedef struct {
    double area;                /* proper area of the horizon           */
    double mass_irr;            /* irreducible mass: M_irr = sqrt(A/16pi) */
    double spin_mag;            /* spin magnitude |J|                    */
    double chi_spin;            /* dimensionless spin: chi = J / M_irr^2 */
    double center[3];          /* centroid of the horizon               */
    double mean_radius;        /* mean coordinate radius                */
    double mass_christodoulou;  /* M_chr = sqrt(M_irr^2 + J^2/(4 M_irr^2)) */
    int    converged;          /* 1 if finder converged, 0 otherwise    */
    double residual;           /* final max|Theta|                      */
} ah_result_t;

/* Allocate an AH finder workspace with angular resolution (n_theta, n_phi).
 * center[3] = center of the trial surface.
 * r_guess = initial constant radius guess. */
ah_workspace_t *ah_alloc(int n_theta, int n_phi,
                          const double center[3], double r_guess);

/* Free the workspace */
void ah_free(ah_workspace_t *ws);

/*
 * Compute the expansion Theta at a point (x,y,z) with outward unit normal s[3].
 *
 * Theta = D_i s^i + K_ij s^i s^j - K
 * where D_i is the covariant derivative, K_ij the physical extrinsic curvature.
 *
 * CCZ4 -> physical conversion:
 *   gamma_ij = h_ij / chi       (physical metric)
 *   K_ij = A_ij/chi + (K/3) * gamma_ij
 *
 * Ref: Baumgarte & Shapiro, "Numerical Relativity" Eq. (6.1)
 */
double compute_expansion(const grid_t *g, double x, double y, double z,
                          const double s[3]);

/*
 * Run the AH finder: evolve the trial surface h(theta, phi) via damped
 * wave flow until max|Theta| < tol or max_iter reached.
 *
 * Returns 1 if converged, 0 otherwise.
 * After convergence, call ah_compute_diagnostics() for area/mass/spin.
 */
int ah_find(ah_workspace_t *ws, const grid_t *g,
            double tol, int max_iter, int verbose);

/*
 * Compute horizon diagnostics from a converged surface.
 *
 * Area:  A = integral sqrt(det(g_2d)) dtheta dphi  (trapezoidal rule)
 * Mass:  M_irr = sqrt(A / 16 pi)
 * Spin:  J = (1/8pi) integral epsilon_{cab} x^a s^b K_{cd} x^d dA
 * M_chr: sqrt(M_irr^2 + J^2 / (4 M_irr^2))
 *
 * Ref: Dreyer et al., PRD 67 (2003) 024018 (isolated horizon spin)
 */
ah_result_t ah_compute_diagnostics(const ah_workspace_t *ws, const grid_t *g);

/*
 * Evaluate expansion Theta at all angular points of the current surface.
 * Fills ws->theta_arr with expansion values. Useful for testing expansion sign.
 */
void ah_eval_expansion(ah_workspace_t *ws, const grid_t *g);

/* ========================================================================
 * AMR mesh variants — same algorithms, block-aware interpolation.
 * Each mirrors its single-grid counterpart but takes const mesh_t *m
 * instead of const grid_t *g. Uses mesh_find_block_at() to locate
 * the finest-level block containing each surface point.
 * ======================================================================== */

struct mesh_s;  /* forward declaration */

int ah_find_amr(ah_workspace_t *ws, const struct mesh_s *m,
                double tol, int max_iter, int verbose);

ah_result_t ah_compute_diagnostics_amr(const ah_workspace_t *ws,
                                        const struct mesh_s *m);

void ah_eval_expansion_amr(ah_workspace_t *ws, const struct mesh_s *m);

#endif /* LATTICE_AH_FINDER_H */
