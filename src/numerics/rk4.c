/*
 * rk4.c — Classical 4th-order Runge-Kutta time integrator
 *
 * 4 stages with coefficients:
 *   k1 = dt * f(t,       y)
 *   k2 = dt * f(t + dt/2, y + k1/2)
 *   k3 = dt * f(t + dt/2, y + k2/2)
 *   k4 = dt * f(t + dt,   y + k3)
 *   y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
 *
 * After each full step, algebraic constraints are enforced:
 *   - det(gamma_tilde) = 1
 *   - tr(A_tilde) = 0
 *
 * B&S Ch. 9.1; standard RK4 method
 */

#include "../core/grid.h"
#include "../core/fields.h"
#include "../geometry/tensor_utils.h"

#include <string.h>

/* RHS function signature: compute rhs from current state */
typedef void (*rhs_func_t)(grid_t *g);

/*
 * Copy fields to scratch, then add dt_factor * k_stage to scratch.
 * scratch = fields + dt_factor * k_stage
 */
static void prepare_stage(grid_t *g, int stage, double dt_factor)
{
    int n = grid_total_points(g);
    int nf = g->params.num_fields;

    if (stage == 0) {
        /* Stage 0: scratch = fields (compute RHS at current state) */
        for (int f = 0; f < nf; f++) {
            memcpy(g->rk_scratch[f], g->fields[f], (size_t)n * sizeof(double));
        }
    } else {
        /* Subsequent stages: scratch = fields + dt_factor * k[stage-1] */
        for (int f = 0; f < nf; f++) {
            const double *src = g->fields[f];
            const double *k = g->rk_k[stage - 1][f];
            double *dst = g->rk_scratch[f];
            for (int idx = 0; idx < n; idx++) {
                dst[idx] = src[idx] + dt_factor * k[idx];
            }
        }
    }
}

/*
 * Store RHS into k[stage]: k[stage] = dt * rhs
 */
static void store_k(grid_t *g, int stage, double dt)
{
    int n = grid_total_points(g);
    int nf = g->params.num_fields;

    for (int f = 0; f < nf; f++) {
        const double *r = g->rhs[f];
        double *k = g->rk_k[stage][f];
        for (int idx = 0; idx < n; idx++) {
            k[idx] = dt * r[idx];
        }
    }
}

/*
 * Final RK4 combination:
 *   fields = fields + (k1 + 2*k2 + 2*k3 + k4) / 6
 */
static void rk4_combine(grid_t *g)
{
    int n = grid_total_points(g);
    int nf = g->params.num_fields;

    for (int f = 0; f < nf; f++) {
        double *u = g->fields[f];
        const double *k1 = g->rk_k[0][f];
        const double *k2 = g->rk_k[1][f];
        const double *k3 = g->rk_k[2][f];
        const double *k4 = g->rk_k[3][f];
        for (int idx = 0; idx < n; idx++) {
            u[idx] += (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]) / 6.0;
        }
    }
}

/*
 * Enforce algebraic constraints after a full RK4 step:
 *   1. det(gamma_tilde) = 1   (rescale conformal metric)
 *   2. tr(A_tilde) = 0        (remove trace from tracefree extrinsic curvature)
 */
void enforce_algebraic_constraints(grid_t *g)
{
    GRID_LOOP_ALL(g, i, j, k) {
        int idx = grid_idx(g, i, j, k);

        /* Enforce det(gamma_tilde) = 1 */
        double gt[6];
        for (int a = 0; a < 6; a++) {
            gt[a] = g->fields[FIELD_GT_BASE + a][idx];
        }
        sym3_enforce_unit_det(gt);
        for (int a = 0; a < 6; a++) {
            g->fields[FIELD_GT_BASE + a][idx] = gt[a];
        }

        /* Enforce tr(A_tilde) = 0: A_ij -= (1/3) g_ij g^{kl} A_{kl} */
        double gtu[6];
        sym3_inv(gt, gtu);

        double at[6];
        for (int a = 0; a < 6; a++) {
            at[a] = g->fields[FIELD_AT_BASE + a][idx];
        }
        sym3_make_tracefree(at, gt, gtu);
        for (int a = 0; a < 6; a++) {
            g->fields[FIELD_AT_BASE + a][idx] = at[a];
        }
    }
}

/*
 * Perform one full RK4 time step.
 *
 * The rhs_func computes the right-hand side using g->rk_scratch[] as input
 * and writes to g->rhs[].
 */
void rk4_step(grid_t *g, rhs_func_t rhs_func, double dt)
{
    /* Stage 1: k1 = dt * f(t, y) */
    prepare_stage(g, 0, 0.0);
    rhs_func(g);
    store_k(g, 0, dt);

    /* Stage 2: k2 = dt * f(t + dt/2, y + k1/2) */
    prepare_stage(g, 1, 0.5);
    rhs_func(g);
    store_k(g, 1, dt);

    /* Stage 3: k3 = dt * f(t + dt/2, y + k2/2) */
    prepare_stage(g, 2, 0.5);
    rhs_func(g);
    store_k(g, 2, dt);

    /* Stage 4: k4 = dt * f(t + dt, y + k3) */
    prepare_stage(g, 3, 1.0);
    rhs_func(g);
    store_k(g, 3, dt);

    /* Combine: y += (k1 + 2k2 + 2k3 + k4) / 6 */
    rk4_combine(g);

    /* Enforce algebraic constraints */
    enforce_algebraic_constraints(g);

    /* Advance time */
    g->time += dt;
    g->step++;
}
