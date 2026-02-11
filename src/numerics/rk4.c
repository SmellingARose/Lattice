/*
 * Lattice — 3D Numerical Relativity
 * RK4 time integrator (memory-efficient variant).
 *
 * Stages:
 *   scratch = fields (backup)
 *   Stage 1: RHS(rhs, fields); accum  = (dt/6)*rhs; scratch = fields + (dt/2)*rhs; BCs
 *   Stage 2: RHS(rhs, scratch); accum += (dt/3)*rhs; scratch = fields + (dt/2)*rhs; BCs
 *   Stage 3: RHS(rhs, scratch); accum += (dt/3)*rhs; scratch = fields + dt*rhs; BCs
 *   Stage 4: RHS(rhs, scratch); accum += (dt/6)*rhs
 *   fields += accum
 *   Enforce: det(gambar)=1, tr(Abar)=0
 */

#include "rk4.h"
#include "../core/fields.h"
#include "../geometry/tensor_utils.h"
#include "../backend/backend.h"
#include <string.h>
#include <math.h>

/* Copy all field arrays: dst = src */
static void copy_fields(double **dst, const double *const *src, size_t n)
{
    for (int f = 0; f < NUM_FIELDS; f++)
        memcpy(dst[f], src[f], n * sizeof(double));
}

/* Linear combination: dst[i] = a[i] + coeff * b[i] for all fields */
static void axpy_fields(double **dst, const double *const *a,
                        const double *const *b, double coeff, size_t n)
{
    for (int f = 0; f < NUM_FIELDS; f++)
        for (size_t i = 0; i < n; i++)
            dst[f][i] = a[f][i] + coeff * b[f][i];
}

/* Accumulate: accum[i] += coeff * rhs[i] */
static void accum_add(double **accum, const double *const *rhs_arr,
                      double coeff, size_t n)
{
    for (int f = 0; f < NUM_FIELDS; f++)
        for (size_t i = 0; i < n; i++)
            accum[f][i] += coeff * rhs_arr[f][i];
}

/* Apply: fields[i] += accum[i] */
static void apply_accum(double **fields, const double *const *accum, size_t n)
{
    for (int f = 0; f < NUM_FIELDS; f++)
        for (size_t i = 0; i < n; i++)
            fields[f][i] += accum[f][i];
}

/* Zero all accumulator arrays */
static void zero_fields(double **arr, size_t n)
{
    for (int f = 0; f < NUM_FIELDS; f++)
        memset(arr[f], 0, n * sizeof(double));
}

/*
 * Enforce algebraic constraints after a full RK4 step.
 *   1. det(gambar) = 1: rescale h_ij so det(h) = 1
 *   2. tr(Abar) = 0: remove trace from A_ij
 *
 * Ref: GRChombo CCZ4/TraceARemoval.hpp, CCZ4/PositiveChiAndAlpha.hpp
 */
static void enforce_algebraic(grid_t *g)
{
    for (int k = 0; k < g->Ntotal; k++) {
        for (int j = 0; j < g->Ntotal; j++) {
            for (int i = 0; i < g->Ntotal; i++) {
                int idx = IDX(g, i, j, k);

                /* Load h_ij */
                double h_loc[3][3];
                h_loc[0][0] = g->fields[FIELD_H11][idx];
                h_loc[0][1] = g->fields[FIELD_H12][idx];
                h_loc[0][2] = g->fields[FIELD_H13][idx];
                h_loc[1][0] = h_loc[0][1];
                h_loc[1][1] = g->fields[FIELD_H22][idx];
                h_loc[1][2] = g->fields[FIELD_H23][idx];
                h_loc[2][0] = h_loc[0][2];
                h_loc[2][1] = h_loc[1][2];
                h_loc[2][2] = g->fields[FIELD_H33][idx];

                /* Enforce det(h) = 1 by rescaling */
                double det = compute_det_sym(h_loc);
                double scale = 1.0 / cbrt(det);
                FOR2(a, b) h_loc[a][b] *= scale;

                g->fields[FIELD_H11][idx] = h_loc[0][0];
                g->fields[FIELD_H12][idx] = h_loc[0][1];
                g->fields[FIELD_H13][idx] = h_loc[0][2];
                g->fields[FIELD_H22][idx] = h_loc[1][1];
                g->fields[FIELD_H23][idx] = h_loc[1][2];
                g->fields[FIELD_H33][idx] = h_loc[2][2];

                /* Enforce tr(A) = 0 */
                double h_UU[3][3];
                compute_inverse_sym(h_loc, h_UU);

                double A_loc[3][3];
                A_loc[0][0] = g->fields[FIELD_A11][idx];
                A_loc[0][1] = g->fields[FIELD_A12][idx];
                A_loc[0][2] = g->fields[FIELD_A13][idx];
                A_loc[1][0] = A_loc[0][1];
                A_loc[1][1] = g->fields[FIELD_A22][idx];
                A_loc[1][2] = g->fields[FIELD_A23][idx];
                A_loc[2][0] = A_loc[0][2];
                A_loc[2][1] = A_loc[1][2];
                A_loc[2][2] = g->fields[FIELD_A33][idx];

                make_trace_free(A_loc, h_loc, h_UU);

                g->fields[FIELD_A11][idx] = A_loc[0][0];
                g->fields[FIELD_A12][idx] = A_loc[0][1];
                g->fields[FIELD_A13][idx] = A_loc[0][2];
                g->fields[FIELD_A22][idx] = A_loc[1][1];
                g->fields[FIELD_A23][idx] = A_loc[1][2];
                g->fields[FIELD_A33][idx] = A_loc[2][2];

                /* Ensure chi > 0 */
                if (g->fields[FIELD_CHI][idx] < 1.0e-12)
                    g->fields[FIELD_CHI][idx] = 1.0e-12;
                /* Ensure lapse > 0 */
                if (g->fields[FIELD_LAPSE][idx] < 1.0e-12)
                    g->fields[FIELD_LAPSE][idx] = 1.0e-12;
            }
        }
    }
}

void rk4_step(grid_t *g, const sim_params_t *p,
              rk4_rhs_func_t rhs_func, rk4_bc_func_t bc_func,
              double dt)
{
    size_t n = g->npoints;

    /* Save initial state */
    copy_fields(g->scratch, (const double *const *)g->fields, n);
    zero_fields(g->accum, n);

    /* --- Stage 1: rhs from fields --- */
    backend_compute_rhs(g->rhs, (const double *const *)g->fields, g, p, rhs_func);
    bc_func(g->rhs, (const double *const *)g->fields, g);
    accum_add(g->accum, (const double *const *)g->rhs, dt / 6.0, n);
    /* scratch2 = fields + (dt/2)*rhs -> store in fields temporarily via scratch */
    axpy_fields(g->fields, (const double *const *)g->scratch,
                (const double *const *)g->rhs, dt / 2.0, n);

    /* --- Stage 2: rhs from fields (= scratch + dt/2 * k1) --- */
    backend_compute_rhs(g->rhs, (const double *const *)g->fields, g, p, rhs_func);
    bc_func(g->rhs, (const double *const *)g->fields, g);
    accum_add(g->accum, (const double *const *)g->rhs, dt / 3.0, n);
    axpy_fields(g->fields, (const double *const *)g->scratch,
                (const double *const *)g->rhs, dt / 2.0, n);

    /* --- Stage 3: rhs from fields (= scratch + dt/2 * k2) --- */
    backend_compute_rhs(g->rhs, (const double *const *)g->fields, g, p, rhs_func);
    bc_func(g->rhs, (const double *const *)g->fields, g);
    accum_add(g->accum, (const double *const *)g->rhs, dt / 3.0, n);
    axpy_fields(g->fields, (const double *const *)g->scratch,
                (const double *const *)g->rhs, dt, n);

    /* --- Stage 4: rhs from fields (= scratch + dt * k3) --- */
    backend_compute_rhs(g->rhs, (const double *const *)g->fields, g, p, rhs_func);
    bc_func(g->rhs, (const double *const *)g->fields, g);
    accum_add(g->accum, (const double *const *)g->rhs, dt / 6.0, n);

    /* Apply: fields = scratch + accum */
    copy_fields(g->fields, (const double *const *)g->scratch, n);
    apply_accum(g->fields, (const double *const *)g->accum, n);

    /* Enforce algebraic constraints */
    enforce_algebraic(g);
}
