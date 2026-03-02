/*
 * Lattice — 3D Numerical Relativity
 * Full CCZ4 right-hand-side at a single grid point.
 *
 * Restructured into 5 static inline sub-functions for better register
 * allocation on GPU targets. Variable lifetimes are scoped to each phase,
 * allowing the compiler to reuse registers (e.g., the 81-entry d2_h array
 * is freed after the Ricci computation in phase 2).
 *
 * Phases:
 *   1. ccz4_load_and_differentiate — load fields, d1/d2/advection
 *   2. ccz4_compute_geometry       — inverse metric, Christoffel, Ricci, Z
 *   3. ccz4_compute_covariant      — covariant d2 lapse, A_UU, tr_A2
 *   4. ccz4_compute_evolution      — CCZ4 RHS: chi, h, K, A, Theta, Gamma
 *   5. ccz4_compute_gauge          — gauge RHS: lapse, shift, B
 *
 * Top-level ccz4_rhs_point() dispatches these 5 + add_ko_dissipation().
 * All sub-functions are static inline with omp declare target: compiler
 * inlines them into a single GPU kernel with zero call overhead.
 *
 * Ref: arXiv:1106.2254 (CCZ4 equations)
 * Ref: GRChombo CCZ4RHS.impl.hpp:60-227
 * Ref: GRChombo CCZ4Geometry.hpp (Ricci with Z)
 * Ref: GRChombo MovingPunctureGauge.hpp (gauge)
 */

#include "ccz4_rhs.h"
#include "maxwell_rhs.h"
#include "../core/fields.h"
#include "../numerics/finite_diff.h"
#include "../geometry/tensor_utils.h"
#include <math.h>

/* ============================================================
 * Output structs for sub-function scoping.
 * Typed outputs scope variable lifetimes so the compiler can
 * reuse registers after each phase completes.
 * ============================================================ */

typedef struct {
    double chi, K, Theta, lapse;
    double h[3][3], A[3][3];
    double Gamma[3], shift[3], B[3];
} ccz4_fields_t;

typedef struct {
    double d1_chi[3], d1_K[3], d1_Theta[3], d1_lapse[3];
    double d1_h[3][3][3], d1_A[3][3][3];
    double d1_Gamma[3][3], d1_shift[3][3];
    double d2_chi[3][3], d2_lapse[3][3];
    double d2_h[3][3][3][3];
    double d2_shift[3][3][3];
    double advec_chi, advec_K, advec_Theta, advec_lapse;
    double advec_h[3][3], advec_A[3][3];
    double advec_Gamma[3], advec_shift[3], advec_B[3];
} ccz4_derivs_t;

typedef struct {
    double h_UU[3][3];
    chris_t chris;
    ricci_t ricci;
    double Z_over_chi[3], Z[3];
} ccz4_geom_t;

typedef struct {
    double covd2lapse[3][3];
    double tr_covd2lapse;
    double A_UU[3][3];
    double tr_A2;
    double divshift;
    double Z_dot_d1lapse;
    double K_minus_2Theta;
} ccz4_covd_t;

/* RHS output structs — sub-functions write computed values here,
 * ccz4_rhs_point stores them to rhs[] arrays. Avoids passing double**
 * through sub-function boundaries (GCC nvptx codegen issue). */
typedef struct {
    double chi, K, Theta;
    double h[3][3], A[3][3];
    double Gamma[3];
} ccz4_evo_rhs_t;

typedef struct {
    double lapse;
    double shift[3], B[3];
} ccz4_gauge_rhs_t;

/* Field indices for symmetric tensor components (file-scope for sub-functions) */
static const int h_idx[3][3] = {
    {FIELD_H11, FIELD_H12, FIELD_H13},
    {FIELD_H12, FIELD_H22, FIELD_H23},
    {FIELD_H13, FIELD_H23, FIELD_H33}
};
static const int A_idx[3][3] = {
    {FIELD_A11, FIELD_A12, FIELD_A13},
    {FIELD_A12, FIELD_A22, FIELD_A23},
    {FIELD_A13, FIELD_A23, FIELD_A33}
};

/* ============================================================
 * Phase 1: Load fields + compute all derivatives.
 * ============================================================ */
LATTICE_DEVICE
static inline void ccz4_load_and_differentiate(
    const double *const * restrict src,
    const grid_t *g, int idx,
    ccz4_fields_t *f, ccz4_derivs_t *d)
{
    const int sx = STRIDE_X;
    const int sy = STRIDE_Y(g);
    const int sz = STRIDE_Z(g);
    const double dx = g->dx;
    const int strides[3] = { sx, sy, sz };

    /* ---- Load fields ---- */
    f->chi = src[FIELD_CHI][idx];
    f->h[0][0] = src[FIELD_H11][idx]; f->h[0][1] = src[FIELD_H12][idx]; f->h[0][2] = src[FIELD_H13][idx];
    f->h[1][0] = f->h[0][1];          f->h[1][1] = src[FIELD_H22][idx]; f->h[1][2] = src[FIELD_H23][idx];
    f->h[2][0] = f->h[0][2];          f->h[2][1] = f->h[1][2];          f->h[2][2] = src[FIELD_H33][idx];

    f->K = src[FIELD_K][idx];
    f->A[0][0] = src[FIELD_A11][idx]; f->A[0][1] = src[FIELD_A12][idx]; f->A[0][2] = src[FIELD_A13][idx];
    f->A[1][0] = f->A[0][1];          f->A[1][1] = src[FIELD_A22][idx]; f->A[1][2] = src[FIELD_A23][idx];
    f->A[2][0] = f->A[0][2];          f->A[2][1] = f->A[1][2];          f->A[2][2] = src[FIELD_A33][idx];

    f->Theta = src[FIELD_THETA][idx];
    f->Gamma[0] = src[FIELD_GAMMA1][idx]; f->Gamma[1] = src[FIELD_GAMMA2][idx]; f->Gamma[2] = src[FIELD_GAMMA3][idx];
    f->lapse = src[FIELD_LAPSE][idx];
    f->shift[0] = src[FIELD_SHIFT1][idx]; f->shift[1] = src[FIELD_SHIFT2][idx]; f->shift[2] = src[FIELD_SHIFT3][idx];
    f->B[0] = src[FIELD_B1][idx]; f->B[1] = src[FIELD_B2][idx]; f->B[2] = src[FIELD_B3][idx];

    /* ---- Fused d1 + diagonal d2 derivatives ---- */
    FOR1(dir) {
        int s = strides[dir];
        fd_d1_d2(src[FIELD_CHI],   idx, s, dx, &d->d1_chi[dir],   &d->d2_chi[dir][dir]);
        fd_d1_d2(src[FIELD_LAPSE], idx, s, dx, &d->d1_lapse[dir], &d->d2_lapse[dir][dir]);
        FOR2(a, b) {
            fd_d1_d2(src[h_idx[a][b]], idx, s, dx,
                     &d->d1_h[a][b][dir], &d->d2_h[a][b][dir][dir]);
        }
        FOR1(a) {
            fd_d1_d2(src[FIELD_SHIFT1 + a], idx, s, dx,
                     &d->d1_shift[a][dir], &d->d2_shift[a][dir][dir]);
        }
        /* d1-only fields */
        d->d1_K[dir]     = fd_d1(src[FIELD_K],     idx, s, dx);
        d->d1_Theta[dir] = fd_d1(src[FIELD_THETA], idx, s, dx);
        FOR2(a, b) {
            d->d1_A[a][b][dir] = fd_d1(src[A_idx[a][b]], idx, s, dx);
        }
        FOR1(a) {
            d->d1_Gamma[a][dir] = fd_d1(src[FIELD_GAMMA1 + a], idx, s, dx);
        }
    }

    /* ---- Mixed second derivatives ---- */
    for (int dir1 = 0; dir1 < 3; dir1++) {
        for (int dir2 = 0; dir2 < dir1; dir2++) {
            int s1 = strides[dir1], s2 = strides[dir2];
            d->d2_chi[dir1][dir2]   = fd_d2_mixed(src[FIELD_CHI],   idx, s1, s2, dx);
            d->d2_chi[dir2][dir1]   = d->d2_chi[dir1][dir2];
            d->d2_lapse[dir1][dir2] = fd_d2_mixed(src[FIELD_LAPSE], idx, s1, s2, dx);
            d->d2_lapse[dir2][dir1] = d->d2_lapse[dir1][dir2];
            FOR2(a, b) {
                d->d2_h[a][b][dir1][dir2] = fd_d2_mixed(src[h_idx[a][b]], idx, s1, s2, dx);
                d->d2_h[a][b][dir2][dir1] = d->d2_h[a][b][dir1][dir2];
            }
            FOR1(a) {
                d->d2_shift[a][dir1][dir2] = fd_d2_mixed(src[FIELD_SHIFT1 + a], idx, s1, s2, dx);
                d->d2_shift[a][dir2][dir1] = d->d2_shift[a][dir1][dir2];
            }
        }
    }

    /* ---- Advection derivatives ---- */
    d->advec_chi = 0.0; d->advec_K = 0.0; d->advec_Theta = 0.0; d->advec_lapse = 0.0;
    FOR2(a, b) { d->advec_h[a][b] = 0.0; d->advec_A[a][b] = 0.0; }
    FOR1(a) { d->advec_Gamma[a] = 0.0; d->advec_shift[a] = 0.0; d->advec_B[a] = 0.0; }

    FOR1(dir) {
        int s = strides[dir];
        double beta = f->shift[dir];
        /* Hoist sign check: all fields share the same beta per direction. */
        double (*fd)(const double *, int, int, double) =
            (beta > 0.0) ? fd_adv_up : fd_adv_down;
        d->advec_chi   += beta * fd(src[FIELD_CHI],   idx, s, dx);
        d->advec_K     += beta * fd(src[FIELD_K],     idx, s, dx);
        d->advec_Theta += beta * fd(src[FIELD_THETA], idx, s, dx);
        d->advec_lapse += beta * fd(src[FIELD_LAPSE], idx, s, dx);
        FOR2(a, b) {
            d->advec_h[a][b] += beta * fd(src[h_idx[a][b]], idx, s, dx);
            d->advec_A[a][b] += beta * fd(src[A_idx[a][b]], idx, s, dx);
        }
        FOR1(a) {
            d->advec_Gamma[a] += beta * fd(src[FIELD_GAMMA1 + a], idx, s, dx);
            d->advec_shift[a] += beta * fd(src[FIELD_SHIFT1 + a], idx, s, dx);
            d->advec_B[a]     += beta * fd(src[FIELD_B1 + a],     idx, s, dx);
        }
    }
}

/* ============================================================
 * Phase 2: Inverse metric, Christoffel symbols, Ricci tensor, Z vector.
 * After return, d2_h (81 doubles) and d2_chi (9 doubles) are no longer
 * needed — the compiler can reuse those registers.
 * Ref: GRChombo CCZ4Geometry.hpp:56-112
 * ============================================================ */
LATTICE_DEVICE
static inline void ccz4_compute_geometry(
    const ccz4_fields_t *f, const ccz4_derivs_t *d,
    ccz4_geom_t *geom)
{
    compute_inverse_sym(f->h, geom->h_UU);
    compute_christoffel(d->d1_h, geom->h_UU, &geom->chris);

    /* Z vector: Z_over_chi[i] = 0.5*(Gamma[i] - chris_contracted[i])
     * Ref: GRChombo CCZ4RHS.impl.hpp:71-82 */
    FOR1(i) {
        geom->Z_over_chi[i] = 0.5 * (f->Gamma[i] - geom->chris.contracted[i]);
        geom->Z[i] = f->chi * geom->Z_over_chi[i];
    }

    /* Covariant derivative of chi */
    double covdtilde2chi[3][3];
    FOR2(kk, ll) {
        covdtilde2chi[kk][ll] = d->d2_chi[kk][ll];
        FOR1(m) covdtilde2chi[kk][ll] -= geom->chris.ULL[m][kk][ll] * d->d1_chi[m];
    }

    double chris_LLU[3][3][3] = {{{0}}};
    double boxtildechi = 0.0;
    double dchi_dot_dchi = 0.0;
    FOR2(ii, jj) {
        boxtildechi += covdtilde2chi[ii][jj] * geom->h_UU[ii][jj];
        dchi_dot_dchi += d->d1_chi[ii] * d->d1_chi[jj] * geom->h_UU[ii][jj];
        FOR2(kk, ll) chris_LLU[ii][jj][kk] += geom->h_UU[kk][ll] * geom->chris.LLL[ii][jj][ll];
    }

    /* Exploit Ricci symmetry: compute upper triangle and mirror.
     * Ref: GRChombo CCZ4Geometry.hpp:78-101, arXiv:1106.2254 Eq. (A1)-(A3) */
    for (int ii = 0; ii < 3; ii++) {
        for (int jj = ii; jj < 3; jj++) {
        double ricci_hat = 0.0;
        FOR1(kk) {
            ricci_hat += 0.5 * (f->h[kk][ii] * d->d1_Gamma[kk][jj]
                              + f->h[kk][jj] * d->d1_Gamma[kk][ii]);
            ricci_hat += 0.5 * f->Gamma[kk] * d->d1_h[ii][jj][kk];
            FOR1(ll) {
                ricci_hat += -0.5 * geom->h_UU[kk][ll] * d->d2_h[ii][jj][kk][ll]
                           + (geom->chris.ULL[kk][ll][ii] * chris_LLU[jj][kk][ll]
                            + geom->chris.ULL[kk][ll][jj] * chris_LLU[ii][kk][ll]
                            + geom->chris.ULL[kk][ii][ll] * chris_LLU[kk][jj][ll]);
            }
        }

        double ricci_chi = 0.5 * (
            (GR_SPACEDIM - 2) * covdtilde2chi[ii][jj]
            + f->h[ii][jj] * boxtildechi
            - ((GR_SPACEDIM - 2) * d->d1_chi[ii] * d->d1_chi[jj]
               + GR_SPACEDIM * f->h[ii][jj] * dchi_dot_dchi) / (2.0 * f->chi)
        );

        double z_terms = 0.0;
        FOR1(kk) {
            z_terms += geom->Z_over_chi[kk] * (f->h[ii][kk] * d->d1_chi[jj]
                                        + f->h[jj][kk] * d->d1_chi[ii]
                                        - f->h[ii][jj] * d->d1_chi[kk]);
        }

        geom->ricci.LL[ii][jj] = (ricci_chi + f->chi * ricci_hat + z_terms) / f->chi;
        geom->ricci.LL[jj][ii] = geom->ricci.LL[ii][jj];
        }
    }

    geom->ricci.scalar = f->chi * compute_trace(geom->ricci.LL, geom->h_UU);
}

/* ============================================================
 * Phase 3: Covariant derivatives of lapse, A_UU, tr_A2, divshift.
 * After return, d2_lapse (9 doubles) is no longer needed.
 * Ref: GRChombo CCZ4RHS.impl.hpp:87-112
 * ============================================================ */
LATTICE_DEVICE
static inline void ccz4_compute_covariant(
    const ccz4_fields_t *f, const ccz4_derivs_t *d,
    const ccz4_geom_t *geom, ccz4_covd_t *covd)
{
    covd->divshift = 0.0;
    FOR1(ii) covd->divshift += d->d1_shift[ii][ii];

    covd->Z_dot_d1lapse = compute_dot_product(geom->Z, d->d1_lapse);
    double dlapse_dot_dchi = compute_dot_product_metric(d->d1_lapse, d->d1_chi, geom->h_UU);

    double covdtilde2lapse[3][3];
    FOR2(kk, ll) {
        covdtilde2lapse[kk][ll] = d->d2_lapse[kk][ll];
        FOR1(m) covdtilde2lapse[kk][ll] -= geom->chris.ULL[m][kk][ll] * d->d1_lapse[m];

        covd->covd2lapse[kk][ll] = f->chi * covdtilde2lapse[kk][ll]
            + 0.5 * (d->d1_lapse[kk] * d->d1_chi[ll] + d->d1_chi[kk] * d->d1_lapse[ll]
                    - f->h[kk][ll] * dlapse_dot_dchi);
    }

    /* Trace of covd2lapse
     * Ref: GRChombo CCZ4RHS.impl.hpp:103-112 */
    covd->tr_covd2lapse = -(GR_SPACEDIM / 2.0) * dlapse_dot_dchi;
    FOR1(ii) {
        covd->tr_covd2lapse -= f->chi * geom->chris.contracted[ii] * d->d1_lapse[ii];
        FOR1(jj) {
            covd->tr_covd2lapse += geom->h_UU[ii][jj] * (f->chi * d->d2_lapse[ii][jj]
                                                 + d->d1_lapse[ii] * d->d1_chi[jj]);
        }
    }

    raise_all_2(f->A, geom->h_UU, covd->A_UU);
    covd->tr_A2 = compute_trace(f->A, covd->A_UU);
    covd->K_minus_2Theta = f->K - 2.0 * f->Theta;
}

/* ============================================================
 * Phase 4: CCZ4 evolution RHS — chi, h_ij, K, A_ij, Theta, Gamma^i.
 * Also applies EM source terms if enabled.
 * Writes results to rhs arrays. Outputs rhs_Gamma for the gauge phase.
 * ============================================================ */
LATTICE_DEVICE
static inline void ccz4_compute_evolution(
    const double *const * restrict src,
    const grid_t *g, int idx,
    const ccz4_fields_t *f, const ccz4_derivs_t *d,
    const ccz4_geom_t *geom, const ccz4_covd_t *covd,
    const sim_params_t *p, ccz4_evo_rhs_t *out)
{
    /* --- chi ---
     * dt(chi) = advec + (2/3)*chi*(alpha*K - divshift)
     * Ref: GRChombo CCZ4RHS.impl.hpp:118-119 */
    out->chi = d->advec_chi
        + (2.0 / GR_SPACEDIM) * f->chi * (f->lapse * f->K - covd->divshift);

    /* CAHD: Coarse-grid-Adjusted Hamiltonian-constraint Damping.
     * Ref: arXiv:2404.01137, Eq. (26) */
    if (p->noise.use_cahd) {
        double H = geom->ricci.scalar
                 + ((GR_SPACEDIM - 1.0) / (double)GR_SPACEDIM) * f->K * f->K
                 - covd->tr_A2;
        out->chi += 4.0 * f->chi * p->noise.cahd_coeff * p->CFL * g->dx * H;
    }

    /* --- h_ij ---
     * Ref: GRChombo CCZ4RHS.impl.hpp:120-129 */
    FOR2(ii, jj) {
        out->h[ii][jj] = d->advec_h[ii][jj]
            - 2.0 * f->lapse * f->A[ii][jj]
            - (2.0 / GR_SPACEDIM) * f->h[ii][jj] * covd->divshift;
        FOR1(kk) {
            out->h[ii][jj] += f->h[kk][ii] * d->d1_shift[kk][jj]
                             + f->h[kk][jj] * d->d1_shift[kk][ii];
        }
    }

    /* --- A_ij ---
     * Ref: GRChombo CCZ4RHS.impl.hpp:131-154 */
    double Adot_TF[3][3];
    FOR2(ii, jj) {
        Adot_TF[ii][jj] = -covd->covd2lapse[ii][jj]
                         + f->chi * f->lapse * geom->ricci.LL[ii][jj];
    }
    make_trace_free(Adot_TF, f->h, geom->h_UU);

    /* A_mixed[k][j] = h^{kl} A_{lj}: first index raised.
     * Eliminates the inner ll-loop in the A_ij contraction. */
    double A_mixed[3][3] = {{0}};
    FOR2(kk, jj) {
        FOR1(ll) A_mixed[kk][jj] += geom->h_UU[kk][ll] * f->A[ll][jj];
    }

    FOR2(ii, jj) {
        out->A[ii][jj] = d->advec_A[ii][jj] + Adot_TF[ii][jj]
            + f->A[ii][jj] * (f->lapse * covd->K_minus_2Theta
                          - (2.0 / GR_SPACEDIM) * covd->divshift);
        FOR1(kk) {
            out->A[ii][jj] += f->A[kk][ii] * d->d1_shift[kk][jj]
                             + f->A[kk][jj] * d->d1_shift[kk][ii]
                             - 2.0 * f->lapse * f->A[ii][kk] * A_mixed[kk][jj];
        }
    }

    /* --- Theta ---
     * Ref: GRChombo CCZ4RHS.impl.hpp:172-180 */
    double kappa1_times_lapse;
    if (p->ccz4.covariant_Z4)
        kappa1_times_lapse = p->ccz4.kappa1;
    else
        kappa1_times_lapse = p->ccz4.kappa1 * f->lapse;

    out->Theta = d->advec_Theta
        + 0.5 * f->lapse * (geom->ricci.scalar - covd->tr_A2
            + ((GR_SPACEDIM - 1.0) / (double)GR_SPACEDIM) * f->K * f->K
            - 2.0 * f->Theta * f->K)
        - 0.5 * f->Theta * kappa1_times_lapse
            * ((GR_SPACEDIM + 1) + p->ccz4.kappa2 * (GR_SPACEDIM - 1))
        - covd->Z_dot_d1lapse;

    /* --- K ---
     * Ref: GRChombo CCZ4RHS.impl.hpp:183-191 */
    out->K = d->advec_K
        + f->lapse * (geom->ricci.scalar + f->K * covd->K_minus_2Theta)
        - kappa1_times_lapse * GR_SPACEDIM * (1.0 + p->ccz4.kappa2) * f->Theta
        - covd->tr_covd2lapse;

    /* --- Gamma^i ---
     * Ref: GRChombo CCZ4RHS.impl.hpp:193-222 */
    double Gammadot[3];
    FOR1(ii) {
        Gammadot[ii] = (2.0 / GR_SPACEDIM) *
            (covd->divshift * (geom->chris.contracted[ii] + 2.0 * p->ccz4.kappa3 * geom->Z_over_chi[ii])
             - 2.0 * f->lapse * f->K * geom->Z_over_chi[ii])
            - 2.0 * kappa1_times_lapse * geom->Z_over_chi[ii];

        FOR1(jj) {
            Gammadot[ii] +=
                2.0 * geom->h_UU[ii][jj] * (f->lapse * d->d1_Theta[jj] - f->Theta * d->d1_lapse[jj])
                - 2.0 * covd->A_UU[ii][jj] * d->d1_lapse[jj]
                - f->lapse * ((2.0 * (GR_SPACEDIM - 1.0) / (double)GR_SPACEDIM)
                           * geom->h_UU[ii][jj] * d->d1_K[jj]
                         + GR_SPACEDIM * covd->A_UU[ii][jj] * d->d1_chi[jj] / f->chi)
                - (geom->chris.contracted[jj] + 2.0 * p->ccz4.kappa3 * geom->Z_over_chi[jj])
                  * d->d1_shift[ii][jj];

            FOR1(kk) {
                Gammadot[ii] +=
                    2.0 * f->lapse * geom->chris.ULL[ii][jj][kk] * covd->A_UU[jj][kk]
                    + geom->h_UU[jj][kk] * d->d2_shift[ii][jj][kk]
                    + ((GR_SPACEDIM - 2.0) / (double)GR_SPACEDIM)
                      * geom->h_UU[ii][jj] * d->d2_shift[kk][jj][kk];
            }
        }
    }

    FOR1(ii) out->Gamma[ii] = d->advec_Gamma[ii] + Gammadot[ii];

    /* --- EM stress-energy source terms ---
     * Ref: B&S Eq. (2.106)-(2.112), arXiv:0907.1151 Sec. III
     *
     * Physical stress-energy diverges as chi -> 0 (near the puncture)
     * because rho_EM ~ chi^{-4} * (conformal E)^2 and the conformal
     * fields only cancel 3 powers of chi, leaving a chi^{-1} divergence.
     * We suppress the EM source by chi^2, ensuring smooth coupling that
     * vanishes inside the horizon (chi << 1) and is full-strength in the
     * wave zone (chi ~ 1). This is standard practice for puncture evolutions
     * with matter coupling.
     * Ref: arXiv:2104.06978 (Liebling & Palenzuela, charged binary inspiral) */
    if (p->em_enabled) {
        double rho_em, S_em_trace;
        double j_em[3], S_em_dd[3][3];
        em_stress_energy(src, g, idx, f->chi, geom->h_UU, f->h,
                         &rho_em, j_em, S_em_dd, &S_em_trace);

        double em_damp = f->chi * f->chi;

        out->Theta += -8.0 * M_PI * em_damp * rho_em;
        out->K += -4.0 * M_PI * em_damp * f->lapse * (rho_em + S_em_trace);

        double matter_A[3][3];
        FOR2(ii2, jj2) {
            matter_A[ii2][jj2] = f->chi * S_em_dd[ii2][jj2]
                                 - (1.0 / 3.0) * f->h[ii2][jj2] * S_em_trace;
        }
        make_trace_free(matter_A, f->h, geom->h_UU);
        FOR2(ii2, jj2) {
            out->A[ii2][jj2] += -8.0 * M_PI * em_damp * f->lapse * matter_A[ii2][jj2];
        }
        FOR1(ii2) {
            out->Gamma[ii2] += -16.0 * M_PI * em_damp * f->lapse * j_em[ii2];
        }
    }
}

/* ============================================================
 * Phase 5: Moving puncture gauge RHS — lapse, shift, B.
 * Ref: GRChombo MovingPunctureGauge.hpp:54-65
 * ============================================================ */
LATTICE_DEVICE
static inline void ccz4_compute_gauge(
    const ccz4_fields_t *f, const ccz4_derivs_t *d,
    const ccz4_covd_t *covd, const double rhs_Gamma[3],
    const sim_params_t *p, ccz4_gauge_rhs_t *out)
{
    /* Lapse: dt(alpha) = -c * alpha^p * (K - 2*Theta) + advec
     * Fast path: default lapse_power=1.0 avoids pow() call. */
    double lapse_pow = (p->gauge.lapse_power == 1.0) ? f->lapse
                       : pow(f->lapse, p->gauge.lapse_power);
    out->lapse = p->gauge.lapse_advec_coeff * d->advec_lapse
        - p->gauge.lapse_coeff * lapse_pow * covd->K_minus_2Theta;

    /* SSL: Slow-Start Lapse — Gaussian damping toward trumpet solution.
     * Ref: arXiv:2404.01137, Eq. (27) */
    if (p->noise.use_ssl) {
        double W = sqrt(fmax(f->chi, 1.0e-10));
        double M = p->noise.ssl_total_mass;
        double h_ssl = p->noise.ssl_h * M;
        double sigma_t = p->noise.ssl_sigma_t * M;
        double t = p->time;
        double ssl_damp = W * h_ssl * exp(-t * t / (2.0 * sigma_t * sigma_t));
        out->lapse += -ssl_damp * (f->lapse - W);
    }

    /* Shift + B: Gamma-driver with position-dependent eta */
    FOR1(ii) {
        out->shift[ii] = p->gauge.shift_advec_coeff * d->advec_shift[ii]
                        + p->gauge.shift_Gamma_coeff * f->B[ii];

        /* Position-dependent eta: eta(x) = eta_0 / W(x), W = sqrt(chi).
         * Ref: arXiv:1003.0859 (Muller & Brugmann) */
        double eta_eff = p->gauge.eta;
        if (p->gauge.position_dependent_eta) {
            double W = sqrt(fmax(f->chi, 1.0e-6));
            eta_eff /= fmax(W, 1.0e-6);
        }
        out->B[ii] = p->gauge.shift_advec_coeff * d->advec_B[ii]
                    - p->gauge.shift_advec_coeff * d->advec_Gamma[ii]
                    + rhs_Gamma[ii] - eta_eff * f->B[ii];
    }
}

/* Forward declaration — add_ko_dissipation is defined in dissipation.c. */
LATTICE_DEVICE
extern void add_ko_dissipation(double ** restrict rhs,
                               const double *const * restrict src,
                               const grid_t *g, const sim_params_t *p,
                               int i, int j, int k);

/* ============================================================
 * Top-level CCZ4 RHS dispatcher.
 * Calls 5 phases + Kreiss-Oliger dissipation.
 *
 * All rhs[] writes are done here — sub-functions write to output
 * structs on the stack. This avoids passing double** through
 * sub-function boundaries (GCC nvptx codegen issue).
 * ============================================================ */
LATTICE_DEVICE
void ccz4_rhs_point(double ** restrict rhs,
                    const double *const * restrict src,
                    const grid_t *g, const sim_params_t *p,
                    int i, int j, int k)
{
    const int idx = IDX(g, i, j, k);

    ccz4_fields_t f;
    ccz4_derivs_t d;
    ccz4_load_and_differentiate(src, g, idx, &f, &d);

    ccz4_geom_t geom;
    ccz4_compute_geometry(&f, &d, &geom);

    ccz4_covd_t covd;
    ccz4_compute_covariant(&f, &d, &geom, &covd);

    ccz4_evo_rhs_t evo;
    ccz4_compute_evolution(src, g, idx, &f, &d, &geom, &covd, p, &evo);

    ccz4_gauge_rhs_t gauge;
    ccz4_compute_gauge(&f, &d, &covd, evo.Gamma, p, &gauge);

    /* ---- Store all RHS values ---- */
    rhs[FIELD_CHI][idx]    = evo.chi;
    rhs[FIELD_H11][idx]    = evo.h[0][0];
    rhs[FIELD_H12][idx]    = evo.h[0][1];
    rhs[FIELD_H13][idx]    = evo.h[0][2];
    rhs[FIELD_H22][idx]    = evo.h[1][1];
    rhs[FIELD_H23][idx]    = evo.h[1][2];
    rhs[FIELD_H33][idx]    = evo.h[2][2];
    rhs[FIELD_K][idx]      = evo.K;
    rhs[FIELD_A11][idx]    = evo.A[0][0];
    rhs[FIELD_A12][idx]    = evo.A[0][1];
    rhs[FIELD_A13][idx]    = evo.A[0][2];
    rhs[FIELD_A22][idx]    = evo.A[1][1];
    rhs[FIELD_A23][idx]    = evo.A[1][2];
    rhs[FIELD_A33][idx]    = evo.A[2][2];
    rhs[FIELD_THETA][idx]  = evo.Theta;
    rhs[FIELD_GAMMA1][idx] = evo.Gamma[0];
    rhs[FIELD_GAMMA2][idx] = evo.Gamma[1];
    rhs[FIELD_GAMMA3][idx] = evo.Gamma[2];
    rhs[FIELD_LAPSE][idx]  = gauge.lapse;
    rhs[FIELD_SHIFT1][idx] = gauge.shift[0];
    rhs[FIELD_SHIFT2][idx] = gauge.shift[1];
    rhs[FIELD_SHIFT3][idx] = gauge.shift[2];
    rhs[FIELD_B1][idx]     = gauge.B[0];
    rhs[FIELD_B2][idx]     = gauge.B[1];
    rhs[FIELD_B3][idx]     = gauge.B[2];

    add_ko_dissipation(rhs, src, g, p, i, j, k);
}
