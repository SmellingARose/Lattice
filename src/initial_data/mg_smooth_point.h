/*
 * Lattice — 3D Numerical Relativity
 * Point-wise multigrid smoother and operator functions.
 *
 * LATTICE_DEVICE-annotated functions shared by CPU (OpenMP) and GPU (HIP)
 * backends. All point functions operate on flat pack buffers using the
 * standard PACK_IDX layout: data[f * nb * npts + b * npts + idx].
 *
 * Solver field layout (10 fields per block):
 *   0: psi (solution)       4: psi_BL (background)
 *   1: V^1 (4-field)        5: A^2 (background)
 *   2: V^2 (4-field)        6: R_tilde (4-field bg)
 *   3: V^3 (4-field)        7-9: S_M^1..3 (4-field bg)
 *
 * Ref: relaxation_amr.c smooth_block_1field / smooth_block_4field
 * Ref: arXiv:0705.1486 (Natchu & Matzner), arXiv:2510.11152 (GPU FAS MG)
 */

#ifndef LATTICE_MG_SMOOTH_POINT_H
#define LATTICE_MG_SMOOTH_POINT_H

#include "../core/device.h"
#include "../numerics/finite_diff.h"

/* Solver field slot indices */
#define MGP_SOL_PSI    0
#define MGP_SOL_V1     1
#define MGP_SOL_V2     2
#define MGP_SOL_V3     3
#define MGP_BG_PSI_BL  4
#define MGP_BG_A2      5
#define MGP_BG_RTILDE  6
#define MGP_BG_SM1     7
#define MGP_BG_SM2     8
#define MGP_BG_SM3     9
#define MGP_N_FIELDS   10

/* FD_D2 center weight for Newton-GS Jacobian diagonal.
 * Ref: relaxation_amr.c line 66 */
#if FD_ORDER == 6
#define MGP_FD_D2_CENTER (-49.0 / 18.0)
#else
#define MGP_FD_D2_CENTER (-5.0 / 2.0)
#endif

/* ================================================================
 * Pack buffer pointer helper
 *
 * Given a flat pack buffer, field index f, block index b, and
 * per-block point count npts, returns pointer to field f of block b.
 * ================================================================ */
LATTICE_DEVICE
static inline double *mg_pack_field(double *buf, int f, int b,
                                     int nb, size_t npts)
{
    return buf + (size_t)f * nb * npts + (size_t)b * npts;
}

LATTICE_DEVICE
static inline const double *mg_pack_field_c(const double *buf, int f, int b,
                                              int nb, size_t npts)
{
    return buf + (size_t)f * nb * npts + (size_t)b * npts;
}

/* ================================================================
 * 1-field Newton-GS smoother (single point)
 *
 * Hamiltonian constraint: Lap(psi) + (1/8) * A^2 * psi_tot^{-7} = f_psi
 * Newton update: psi -= (Lap + source - f) / (J_lap + dS)
 *
 * Ref: relaxation_amr.c:898 smooth_block_1field
 * Ref: arXiv:0705.1486 eq. (8)
 * ================================================================ */
LATTICE_DEVICE
static inline void mg_smooth_1field_point(
    double *psi, const double *psi_BL, const double *A2,
    const double *f_psi,
    int idx, int sx, int sy, int sz, double inv_dx, double dx2)
{
    double J_lap = 3.0 * MGP_FD_D2_CENTER / dx2;

    double psi_tot = psi_BL[idx] + psi[idx];
    if (psi_tot < 0.1) psi_tot = 0.1;

    double lap = fd_d2(psi, idx, sx, inv_dx)
               + fd_d2(psi, idx, sy, inv_dx)
               + fd_d2(psi, idx, sz, inv_dx);

    double p2 = psi_tot * psi_tot;
    double p4 = p2 * p2;
    double p7 = p4 * p2 * psi_tot;
    double p8 = p7 * psi_tot;

    double source = 0.125 * A2[idx] / p7;
    double residual = lap + source - f_psi[idx];
    double dS = -0.875 * A2[idx] / p8;

    psi[idx] -= residual / (J_lap + dS);
}

/* ================================================================
 * 4-field Newton-GS smoother (single point)
 *
 * Hamiltonian: Lap(psi) + R_tilde/8 * psi_tot + A^2/8 * psi_tot^{-7} = f_psi
 * Momentum:   Lap(V^d) + (1/3) d_d(div V) + S_M^d = f_V^d
 *
 * Ref: relaxation_amr.c:934 smooth_block_4field
 * Ref: arXiv:0705.1486 eq. (8)+(10)
 * ================================================================ */
LATTICE_DEVICE
static inline void mg_smooth_4field_point(
    double *psi, double *V0, double *V1, double *V2,
    const double *psi_BL, const double *A2,
    const double *R_tilde, const double *SM0, const double *SM1,
    const double *SM2,
    const double *f_psi, const double *f_V0, const double *f_V1,
    const double *f_V2,
    int idx, int sx, int sy, int sz, double inv_dx, double dx2)
{
    double J_lap = 3.0 * MGP_FD_D2_CENTER / dx2;
    double J_V_diag = J_lap + (1.0 / 3.0) * MGP_FD_D2_CENTER / dx2;

    /* Psi update */
    double psi_tot = psi_BL[idx] + psi[idx];
    if (psi_tot < 0.1) psi_tot = 0.1;

    double lap_psi = fd_d2(psi, idx, sx, inv_dx)
                   + fd_d2(psi, idx, sy, inv_dx)
                   + fd_d2(psi, idx, sz, inv_dx);

    double p2 = psi_tot * psi_tot;
    double p4 = p2 * p2;
    double p7 = p4 * p2 * psi_tot;
    double p8 = p7 * psi_tot;

    double src_H = R_tilde[idx] * 0.125 * psi_tot
                 + A2[idx] * 0.125 / p7;
    double res_psi = lap_psi + src_H - f_psi[idx];
    double dS_psi = 0.125 * R_tilde[idx]
                  - 0.875 * A2[idx] / p8;
    psi[idx] -= res_psi / (J_lap + dS_psi);

    /* V^d updates */
    double *V[3] = { V0, V1, V2 };
    const double *f_V[3] = { f_V0, f_V1, f_V2 };
    const double *SM[3] = { SM0, SM1, SM2 };
    int strides[3] = { sx, sy, sz };

    for (int d = 0; d < 3; d++) {
        double lap_V = fd_d2(V[d], idx, sx, inv_dx)
                     + fd_d2(V[d], idx, sy, inv_dx)
                     + fd_d2(V[d], idx, sz, inv_dx);
        double d_divV = 0.0;
        for (int e = 0; e < 3; e++) {
            if (e == d)
                d_divV += fd_d2(V[e], idx, strides[e], inv_dx);
            else
                d_divV += fd_d2_mixed(V[e], idx,
                                      strides[d], strides[e], inv_dx);
        }
        double res_V = lap_V + d_divV / 3.0
                     + SM[d][idx] - f_V[d][idx];
        V[d][idx] -= res_V / J_V_diag;
    }
}

/* ================================================================
 * 1-field Newton-GS delta (single point) — GPU two-pass variant
 *
 * Same math as mg_smooth_1field_point, but computes the delta
 * without writing to psi. Returns delta via output pointer.
 * Pass 1 of two-pass GPU smoother: all threads read psi (frozen),
 * no writes → no race even with 6th-order radius-3 stencil.
 *
 * Ref: HPGMG out-of-place GSRB (Sec. 3.2, Adams et al. 2014)
 * ================================================================ */
LATTICE_DEVICE
static inline void mg_smooth_1field_delta(
    const double *psi, const double *psi_BL, const double *A2,
    const double *f_psi, double *delta_psi,
    int idx, int sx, int sy, int sz, double inv_dx, double dx2)
{
    double J_lap = 3.0 * MGP_FD_D2_CENTER / dx2;

    double psi_tot = psi_BL[idx] + psi[idx];
    if (psi_tot < 0.1) psi_tot = 0.1;

    double lap = fd_d2(psi, idx, sx, inv_dx)
               + fd_d2(psi, idx, sy, inv_dx)
               + fd_d2(psi, idx, sz, inv_dx);

    double p2 = psi_tot * psi_tot;
    double p4 = p2 * p2;
    double p7 = p4 * p2 * psi_tot;
    double p8 = p7 * psi_tot;

    double source = 0.125 * A2[idx] / p7;
    double residual = lap + source - f_psi[idx];
    double dS = -0.875 * A2[idx] / p8;

    delta_psi[idx] = -residual / (J_lap + dS);
}

/* ================================================================
 * 4-field Newton-GS delta (single point) — GPU two-pass variant
 *
 * Same math as mg_smooth_4field_point, but writes deltas to
 * separate output buffers instead of updating solution in-place.
 *
 * Ref: HPGMG out-of-place GSRB (Sec. 3.2, Adams et al. 2014)
 * ================================================================ */
LATTICE_DEVICE
static inline void mg_smooth_4field_delta(
    const double *psi, const double *V0, const double *V1,
    const double *V2,
    const double *psi_BL, const double *A2,
    const double *R_tilde, const double *SM0, const double *SM1,
    const double *SM2,
    const double *f_psi, const double *f_V0, const double *f_V1,
    const double *f_V2,
    double *delta_psi, double *delta_V0, double *delta_V1,
    double *delta_V2,
    int idx, int sx, int sy, int sz, double inv_dx, double dx2)
{
    double J_lap = 3.0 * MGP_FD_D2_CENTER / dx2;
    double J_V_diag = J_lap + (1.0 / 3.0) * MGP_FD_D2_CENTER / dx2;

    /* Psi delta */
    double psi_tot = psi_BL[idx] + psi[idx];
    if (psi_tot < 0.1) psi_tot = 0.1;

    double lap_psi = fd_d2(psi, idx, sx, inv_dx)
                   + fd_d2(psi, idx, sy, inv_dx)
                   + fd_d2(psi, idx, sz, inv_dx);

    double p2 = psi_tot * psi_tot;
    double p4 = p2 * p2;
    double p7 = p4 * p2 * psi_tot;
    double p8 = p7 * psi_tot;

    double src_H = R_tilde[idx] * 0.125 * psi_tot
                 + A2[idx] * 0.125 / p7;
    double res_psi = lap_psi + src_H - f_psi[idx];
    double dS_psi = 0.125 * R_tilde[idx]
                  - 0.875 * A2[idx] / p8;
    delta_psi[idx] = -res_psi / (J_lap + dS_psi);

    /* V^d deltas */
    const double *V[3] = { V0, V1, V2 };
    const double *f_V[3] = { f_V0, f_V1, f_V2 };
    const double *SM[3] = { SM0, SM1, SM2 };
    double *delta_V[3] = { delta_V0, delta_V1, delta_V2 };
    int strides[3] = { sx, sy, sz };

    for (int d = 0; d < 3; d++) {
        double lap_V = fd_d2(V[d], idx, sx, inv_dx)
                     + fd_d2(V[d], idx, sy, inv_dx)
                     + fd_d2(V[d], idx, sz, inv_dx);
        double d_divV = 0.0;
        for (int e = 0; e < 3; e++) {
            if (e == d)
                d_divV += fd_d2(V[e], idx, strides[e], inv_dx);
            else
                d_divV += fd_d2_mixed(V[e], idx,
                                      strides[d], strides[e], inv_dx);
        }
        double res_V = lap_V + d_divV / 3.0
                     + SM[d][idx] - f_V[d][idx];
        delta_V[d][idx] = -res_V / J_V_diag;
    }
}

/* ================================================================
 * 1-field operator L(u) evaluation (single point)
 *
 * L(psi) = Lap(psi) + (1/8) * A^2 * psi_tot^{-7}
 * Writes result to L_psi.
 *
 * Ref: relaxation_amr.c:1062 compute_operator_level
 * ================================================================ */
LATTICE_DEVICE
static inline void mg_operator_1field_point(
    double *L_psi, const double *psi, const double *psi_BL,
    const double *A2,
    int idx, int sx, int sy, int sz, double inv_dx)
{
    double psi_tot = psi_BL[idx] + psi[idx];
    if (psi_tot < 0.1) psi_tot = 0.1;

    double lap = fd_d2(psi, idx, sx, inv_dx)
               + fd_d2(psi, idx, sy, inv_dx)
               + fd_d2(psi, idx, sz, inv_dx);

    double p2 = psi_tot * psi_tot;
    double p4 = p2 * p2;
    double p7 = p4 * p2 * psi_tot;

    L_psi[idx] = lap + 0.125 * A2[idx] / p7;
}

/* ================================================================
 * 4-field operator L(u) evaluation (single point)
 *
 * L_psi = Lap(psi) + R_tilde/8 * psi_tot + A^2/8 * psi_tot^{-7}
 * L_V^d = Lap(V^d) + (1/3) d_d(div V) + S_M^d
 *
 * Ref: relaxation_amr.c:1062 compute_operator_level
 * ================================================================ */
LATTICE_DEVICE
static inline void mg_operator_4field_point(
    double *L_psi, double *L_V0, double *L_V1, double *L_V2,
    const double *psi, const double *V0, const double *V1, const double *V2,
    const double *psi_BL, const double *A2, const double *R_tilde,
    const double *SM0, const double *SM1, const double *SM2,
    int idx, int sx, int sy, int sz, double inv_dx)
{
    double psi_tot = psi_BL[idx] + psi[idx];
    if (psi_tot < 0.1) psi_tot = 0.1;

    double lap = fd_d2(psi, idx, sx, inv_dx)
               + fd_d2(psi, idx, sy, inv_dx)
               + fd_d2(psi, idx, sz, inv_dx);

    double p2 = psi_tot * psi_tot;
    double p4 = p2 * p2;
    double p7 = p4 * p2 * psi_tot;

    L_psi[idx] = lap + R_tilde[idx] * 0.125 * psi_tot
               + A2[idx] * 0.125 / p7;

    const double *V[3] = { V0, V1, V2 };
    double *L_V[3] = { L_V0, L_V1, L_V2 };
    const double *SM[3] = { SM0, SM1, SM2 };
    int strides[3] = { sx, sy, sz };

    for (int d = 0; d < 3; d++) {
        double lap_V = fd_d2(V[d], idx, sx, inv_dx)
                     + fd_d2(V[d], idx, sy, inv_dx)
                     + fd_d2(V[d], idx, sz, inv_dx);
        double d_divV = 0.0;
        for (int e = 0; e < 3; e++) {
            if (e == d)
                d_divV += fd_d2(V[e], idx, strides[e], inv_dx);
            else
                d_divV += fd_d2_mixed(V[e], idx,
                                      strides[d], strides[e], inv_dx);
        }
        L_V[d][idx] = lap_V + d_divV / 3.0 + SM[d][idx];
    }
}

#endif /* LATTICE_MG_SMOOTH_POINT_H */
