/*
 * Lattice — 3D Numerical Relativity
 * CCE worldtube output for SpECTRE.
 *
 * Interpolates conformal CCZ4 fields on an extraction sphere,
 * reconstructs physical ADM quantities, and writes HDF5 in SpECTRE's
 * AdmMetricNodal format (49 datasets with GL×uniform angular grid).
 *
 * Conformal → physical reconstruction:
 *   gamma_ij = h_ij / chi
 *   K_ij = (A_ij + K/3 h_ij) / chi
 *   d_k gamma_ij = (d_k h_ij) / chi - h_ij d_k(chi) / chi^2
 *
 * Ref: SpECTRE CCE documentation (AdmMetricNodal format)
 * Ref: arXiv:1106.2254 (CCZ4 conformal variables)
 */

#ifdef LATTICE_HDF5

#include "cce_worldtube.h"
#include "../core/fields.h"
#include "../numerics/interpolate.h"
#include "../amr/mesh.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * Dataset names (SpECTRE AdmMetricNodal format, 49 total)
 * ================================================================ */

const char *cce_dataset_names[CCE_NUM_DATASETS] = {
    /* Spatial metric (6) */
    "gxx.dat", "gxy.dat", "gxz.dat", "gyy.dat", "gyz.dat", "gzz.dat",
    /* d_x metric (6) */
    "Dxgxx.dat", "Dxgxy.dat", "Dxgxz.dat", "Dxgyy.dat", "Dxgyz.dat", "Dxgzz.dat",
    /* d_y metric (6) */
    "Dygxx.dat", "Dygxy.dat", "Dygxz.dat", "Dygyy.dat", "Dygyz.dat", "Dygzz.dat",
    /* d_z metric (6) */
    "Dzgxx.dat", "Dzgxy.dat", "Dzgxz.dat", "Dzgyy.dat", "Dzgyz.dat", "Dzgzz.dat",
    /* Lapse + derivatives (4) */
    "Lapse.dat", "DxLapse.dat", "DyLapse.dat", "DzLapse.dat",
    /* Shift (3) */
    "Shiftx.dat", "Shifty.dat", "Shiftz.dat",
    /* d_x shift (3) */
    "DxShiftx.dat", "DxShifty.dat", "DxShiftz.dat",
    /* d_y shift (3) */
    "DyShiftx.dat", "DyShifty.dat", "DyShiftz.dat",
    /* d_z shift (3) */
    "DzShiftx.dat", "DzShifty.dat", "DzShiftz.dat",
    /* Extrinsic curvature (6) */
    "Kxx.dat", "Kxy.dat", "Kxz.dat", "Kyy.dat", "Kyz.dat", "Kzz.dat",
    /* Gauge auxiliary (3) */
    "AuxiliaryShiftx.dat", "AuxiliaryShifty.dat", "AuxiliaryShiftz.dat"
};

/* Symmetric tensor flat index: (i,j) → 0..5 matching xx,xy,xz,yy,yz,zz */
static const int sym_flat[3][3] = {
    {0, 1, 2},
    {1, 3, 4},
    {2, 4, 5}
};

/* Field indices for h_ij and A_ij */
static const int h_field[3][3] = {
    {FIELD_H11, FIELD_H12, FIELD_H13},
    {FIELD_H12, FIELD_H22, FIELD_H23},
    {FIELD_H13, FIELD_H23, FIELD_H33}
};

static const int A_field[3][3] = {
    {FIELD_A11, FIELD_A12, FIELD_A13},
    {FIELD_A12, FIELD_A22, FIELD_A23},
    {FIELD_A13, FIELD_A23, FIELD_A33}
};

/* ================================================================
 * Gauss-Legendre quadrature nodes/weights on [-1, 1]
 * (Same algorithm as psi4.c:63-94)
 * ================================================================ */

static void gauss_legendre_nodes(int n, double *x, double *w)
{
    for (int i = 0; i < n; i++) {
        /* Initial guess: Chebyshev-type */
        double xi = cos(M_PI * (4 * i + 3) / (4.0 * n + 2));

        /* Newton iteration on Legendre polynomial P_n(x) */
        for (int iter = 0; iter < 100; iter++) {
            double p0 = 1.0, p1 = xi;
            for (int k = 2; k <= n; k++) {
                double p2 = ((2 * k - 1) * xi * p1 - (k - 1) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            double dp = n * (xi * p1 - p0) / (xi * xi - 1.0);
            double dx_val = p1 / dp;
            xi -= dx_val;
            if (fabs(dx_val) < 1e-15) break;
        }
        x[i] = xi;
        double p0 = 1.0, p1 = xi;
        for (int k = 2; k <= n; k++) {
            double p2 = ((2 * k - 1) * xi * p1 - (k - 1) * p0) / k;
            p0 = p1;
            p1 = p2;
        }
        double dp = n * (xi * p1 - p0) / (xi * xi - 1.0);
        w[i] = 2.0 / ((1.0 - xi * xi) * dp * dp);
    }
}

/* ================================================================
 * HDF5 helpers
 * ================================================================ */

/* Create one extensible dataset with Legend attribute */
static hid_t create_cce_dataset(hid_t file, const char *name, int n_cols)
{
    hsize_t dims[2] = {0, (hsize_t)n_cols};
    hsize_t maxdims[2] = {H5S_UNLIMITED, (hsize_t)n_cols};
    hid_t space = H5Screate_simple(2, dims, maxdims);

    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk[2] = {1, (hsize_t)n_cols};
    H5Pset_chunk(plist, 2, chunk);

    hid_t ds = H5Dcreate2(file, name, H5T_IEEE_F64LE, space,
                           H5P_DEFAULT, plist, H5P_DEFAULT);

    /* Write Legend attribute: "time,Node0,Node1,..." */
    size_t legend_len = 16 + (size_t)n_cols * 12;
    char *legend = malloc(legend_len);
    int pos = sprintf(legend, "time");
    for (int i = 0; i < n_cols - 1; i++)
        pos += sprintf(legend + pos, ",Node%d", i);

    hid_t atype = H5Tcopy(H5T_C_S1);
    H5Tset_size(atype, strlen(legend) + 1);
    H5Tset_strpad(atype, H5T_STR_NULLTERM);
    hid_t aspace = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(ds, "Legend", atype, aspace,
                             H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, atype, legend);

    H5Aclose(attr);
    H5Sclose(aspace);
    H5Tclose(atype);
    free(legend);

    H5Pclose(plist);
    H5Sclose(space);

    return ds;
}

/* Append one row of data to an extensible dataset */
static void write_cce_row(hid_t ds, int row, const double *data, int n_cols)
{
    /* Extend dataset to accommodate new row */
    hsize_t new_dims[2] = {(hsize_t)(row + 1), (hsize_t)n_cols};
    H5Dset_extent(ds, new_dims);

    /* Select hyperslab for the new row */
    hid_t fspace = H5Dget_space(ds);
    hsize_t offset[2] = {(hsize_t)row, 0};
    hsize_t count[2] = {1, (hsize_t)n_cols};
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);

    /* Memory space: one row */
    hsize_t mdims[2] = {1, (hsize_t)n_cols};
    hid_t mspace = H5Screate_simple(2, mdims, NULL);

    H5Dwrite(ds, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, data);

    H5Sclose(mspace);
    H5Sclose(fspace);
}

/* ================================================================
 * Workspace allocation / free
 * ================================================================ */

cce_ws_t *cce_alloc(int l_max, double radius, const double center[3],
                     const char *filename)
{
    cce_ws_t *ws = calloc(1, sizeof(cce_ws_t));
    ws->l_max   = l_max;
    ws->n_theta = l_max + 1;
    ws->n_phi   = 2 * l_max + 1;
    ws->radius  = radius;
    ws->center[0] = center[0];
    ws->center[1] = center[1];
    ws->center[2] = center[2];

    int n_angular = ws->n_theta * ws->n_phi;
    ws->n_cols = 1 + n_angular;

    /* Gauss-Legendre nodes → theta = arccos(x) */
    double *gl_x = malloc((size_t)ws->n_theta * sizeof(double));
    ws->theta      = malloc((size_t)ws->n_theta * sizeof(double));
    ws->gl_weights = malloc((size_t)ws->n_theta * sizeof(double));
    gauss_legendre_nodes(ws->n_theta, gl_x, ws->gl_weights);
    for (int i = 0; i < ws->n_theta; i++)
        ws->theta[i] = acos(gl_x[i]);
    free(gl_x);

    /* Uniform phi grid */
    ws->phi = malloc((size_t)ws->n_phi * sizeof(double));
    for (int j = 0; j < ws->n_phi; j++)
        ws->phi[j] = 2.0 * M_PI * j / ws->n_phi;

    /* Scratch buffers */
    ws->row_buf = malloc((size_t)ws->n_cols * sizeof(double));
    ws->angular_buf = malloc((size_t)CCE_NUM_DATASETS * (size_t)n_angular
                             * sizeof(double));

    /* Create HDF5 file */
    ws->file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    for (int d = 0; d < CCE_NUM_DATASETS; d++)
        ws->dataset_ids[d] = create_cce_dataset(ws->file_id,
                                                 cce_dataset_names[d],
                                                 ws->n_cols);

    ws->n_rows = 0;

    return ws;
}

void cce_free(cce_ws_t *ws)
{
    if (!ws) return;

    for (int d = 0; d < CCE_NUM_DATASETS; d++)
        H5Dclose(ws->dataset_ids[d]);
    H5Fclose(ws->file_id);

    free(ws->theta);
    free(ws->gl_weights);
    free(ws->phi);
    free(ws->row_buf);
    free(ws->angular_buf);
    free(ws);
}

/* ================================================================
 * Conformal → physical reconstruction
 *
 * gamma_ij = h_ij / chi
 * K_ij = (A_ij + K/3 h_ij) / chi
 * d_k gamma_ij = d_k(h_ij)/chi - h_ij * d_k(chi)/chi^2
 *
 * Lapse, shift, and their derivatives are physical (no conformal factor).
 * B^i (Gamma-driver auxiliary) maps directly to AuxiliaryShift.
 *
 * Ref: arXiv:1106.2254 Eq. (1)-(3), B&S §3.5
 * Ref: GRChombo Source/CCZ4/CCZ4Vars.hpp (conformal decomposition)
 * ================================================================ */

static void conformal_to_physical(
    double chi, const double h[3][3],
    double K_val, const double A[3][3],
    const double d_chi[3], const double d_h[3][3][3],
    /* outputs: */
    double gamma[3][3],
    double K_phys[3][3],
    double d_gamma[3][3][3])   /* d_gamma[k][i][j] = d_k gamma_ij */
{
    if (chi < 1e-12) chi = 1e-12;
    double inv_chi = 1.0 / chi;
    double inv_chi2 = inv_chi * inv_chi;
    double K3 = K_val / 3.0;

    for (int i = 0; i < 3; i++) {
        for (int j = i; j < 3; j++) {
            gamma[i][j] = h[i][j] * inv_chi;
            gamma[j][i] = gamma[i][j];

            K_phys[i][j] = (A[i][j] + K3 * h[i][j]) * inv_chi;
            K_phys[j][i] = K_phys[i][j];

            for (int k = 0; k < 3; k++) {
                d_gamma[k][i][j] = d_h[k][i][j] * inv_chi
                                 - h[i][j] * d_chi[k] * inv_chi2;
                d_gamma[k][j][i] = d_gamma[k][i][j];
            }
        }
    }
}

/* ================================================================
 * Main extraction
 *
 * For each (theta, phi) on the GL×uniform sphere:
 *   1. Compute Cartesian position
 *   2. Find containing block (cached)
 *   3. Interpolate conformal fields + derivatives
 *   4. Reconstruct physical ADM quantities
 *   5. Store in theta-varies-fastest order
 *
 * Same block-caching pattern as psi4_extract (psi4.c:593-670).
 * ================================================================ */

void cce_extract(cce_ws_t *ws, const mesh_t *m, double time)
{
    double r = ws->radius;
    int n_angular = ws->n_theta * ws->n_phi;
    double *abuf = ws->angular_buf;
    block_t *cached = NULL;

    /* Zero angular buffer */
    memset(abuf, 0, (size_t)CCE_NUM_DATASETS * (size_t)n_angular
                     * sizeof(double));

    for (int iph = 0; iph < ws->n_phi; iph++) {
        double ph = ws->phi[iph];
        double sp = sin(ph), cp = cos(ph);

        for (int ith = 0; ith < ws->n_theta; ith++) {
            /* Theta-varies-fastest: index = phi_j * n_theta + theta_i */
            int aidx = iph * ws->n_theta + ith;

            double th = ws->theta[ith];
            double st = sin(th), ct = cos(th);

            /* Cartesian position on sphere */
            double x = ws->center[0] + r * st * cp;
            double y = ws->center[1] + r * st * sp;
            double z = ws->center[2] + r * ct;

            /* Find block (cached lookup, same as psi4_extract) */
            block_t *b = cached;
            if (b) {
                double bx = b->grid->dx;
                int N = b->grid->N;
                int inside = 1;
                for (int d = 0; d < 3; d++) {
                    double coord = (d == 0) ? x : (d == 1) ? y : z;
                    if (coord < b->origin[d] ||
                        coord >= b->origin[d] + N * bx)
                        { inside = 0; break; }
                }
                if (!inside) b = NULL;
            }
            if (!b) b = mesh_find_block_at(m, x, y, z);
            if (!b) continue;  /* outside domain — leave as zero */
            cached = b;

            grid_t *g = b->grid;
            const double *origin = b->origin;

            /* --- Interpolate conformal fields with derivatives --- */

            /* chi + derivatives */
            double chi_v[4];
            interp_field_deriv_at_block(g->fields[FIELD_CHI], g, origin,
                                        x, y, z, chi_v);
            double chi = chi_v[0];
            double d_chi[3] = {chi_v[1], chi_v[2], chi_v[3]};

            /* h_ij + derivatives (6 independent components) */
            double h[3][3], d_h[3][3][3]; /* d_h[dir][i][j] */
            for (int i = 0; i < 3; i++) {
                for (int j = i; j < 3; j++) {
                    double hv[4];
                    interp_field_deriv_at_block(g->fields[h_field[i][j]], g,
                                                origin, x, y, z, hv);
                    h[i][j] = hv[0]; h[j][i] = hv[0];
                    d_h[0][i][j] = hv[1]; d_h[0][j][i] = hv[1];
                    d_h[1][i][j] = hv[2]; d_h[1][j][i] = hv[2];
                    d_h[2][i][j] = hv[3]; d_h[2][j][i] = hv[3];
                }
            }

            /* lapse + derivatives */
            double lapse_v[4];
            interp_field_deriv_at_block(g->fields[FIELD_LAPSE], g, origin,
                                        x, y, z, lapse_v);

            /* shift + derivatives (3 components) */
            double shift[3], d_shift[3][3]; /* d_shift[dir][comp] */
            static const int shift_fields[3] =
                {FIELD_SHIFT1, FIELD_SHIFT2, FIELD_SHIFT3};
            for (int i = 0; i < 3; i++) {
                double sv[4];
                interp_field_deriv_at_block(g->fields[shift_fields[i]], g,
                                            origin, x, y, z, sv);
                shift[i] = sv[0];
                d_shift[0][i] = sv[1];
                d_shift[1][i] = sv[2];
                d_shift[2][i] = sv[3];
            }

            /* K (value only) */
            double K_val = interp_field_at_block(g->fields[FIELD_K], g,
                                                  origin, x, y, z);

            /* A_ij (values only, 6 independent) */
            double A[3][3];
            for (int i = 0; i < 3; i++) {
                for (int j = i; j < 3; j++) {
                    A[i][j] = interp_field_at_block(
                        g->fields[A_field[i][j]], g, origin, x, y, z);
                    A[j][i] = A[i][j];
                }
            }

            /* B^i (values only) — maps to AuxiliaryShift */
            double B[3];
            B[0] = interp_field_at_block(g->fields[FIELD_B1], g,
                                          origin, x, y, z);
            B[1] = interp_field_at_block(g->fields[FIELD_B2], g,
                                          origin, x, y, z);
            B[2] = interp_field_at_block(g->fields[FIELD_B3], g,
                                          origin, x, y, z);

            /* --- Conformal → physical reconstruction --- */
            double gamma[3][3], K_phys[3][3], d_gamma[3][3][3];
            conformal_to_physical(chi, h, K_val, A, d_chi, d_h,
                                  gamma, K_phys, d_gamma);

            /* --- Store into angular buffer --- */

            /* Spatial metric (6) */
            for (int i = 0; i < 3; i++)
                for (int j = i; j < 3; j++)
                    abuf[(CCE_GXX + sym_flat[i][j]) * n_angular + aidx]
                        = gamma[i][j];

            /* d_k metric (18 = 3 dirs × 6 components) */
            for (int k = 0; k < 3; k++)
                for (int i = 0; i < 3; i++)
                    for (int j = i; j < 3; j++)
                        abuf[(CCE_DXGXX + k * 6 + sym_flat[i][j])
                             * n_angular + aidx] = d_gamma[k][i][j];

            /* Lapse + derivatives */
            abuf[CCE_LAPSE   * n_angular + aidx] = lapse_v[0];
            abuf[CCE_DXLAPSE * n_angular + aidx] = lapse_v[1];
            abuf[CCE_DYLAPSE * n_angular + aidx] = lapse_v[2];
            abuf[CCE_DZLAPSE * n_angular + aidx] = lapse_v[3];

            /* Shift (3) */
            for (int i = 0; i < 3; i++)
                abuf[(CCE_SHIFTX + i) * n_angular + aidx] = shift[i];

            /* d_k shift (9 = 3 dirs × 3 components) */
            for (int k = 0; k < 3; k++)
                for (int i = 0; i < 3; i++)
                    abuf[(CCE_DXSHIFTX + k * 3 + i) * n_angular + aidx]
                        = d_shift[k][i];

            /* Extrinsic curvature (6) */
            for (int i = 0; i < 3; i++)
                for (int j = i; j < 3; j++)
                    abuf[(CCE_KXX + sym_flat[i][j]) * n_angular + aidx]
                        = K_phys[i][j];

            /* Auxiliary shift = B^i (3) */
            for (int i = 0; i < 3; i++)
                abuf[(CCE_AUXSHIFTX + i) * n_angular + aidx] = B[i];
        }
    }

    /* --- Write one row per dataset to HDF5 --- */
    for (int d = 0; d < CCE_NUM_DATASETS; d++) {
        ws->row_buf[0] = time;
        memcpy(ws->row_buf + 1, abuf + d * n_angular,
               (size_t)n_angular * sizeof(double));
        write_cce_row(ws->dataset_ids[d], ws->n_rows,
                      ws->row_buf, ws->n_cols);
    }

    ws->n_rows++;
}

#endif /* LATTICE_HDF5 */
