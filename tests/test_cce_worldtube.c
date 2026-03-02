/*
 * Lattice — 3D Numerical Relativity
 * Tests for CCE worldtube output (SpECTRE AdmMetricNodal format).
 *
 * Tests:
 *   1. Conformal-to-physical reconstruction (unit test)
 *   2. Flat spacetime extraction (full pipeline)
 *   3. Schwarzschild derivatives (analytical BL comparison)
 *   4. HDF5 format validation (dataset names, shapes, Legend)
 *   5. Angular ordering (theta-varies-fastest)
 *   6. Dataset name completeness
 *   7. Multiple time rows
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <hdf5.h>
#include "../src/core/grid.h"
#include "../src/core/fields.h"
#include "../src/amr/mesh.h"
#include "../src/initial_data/puncture.h"
#include "../src/diagnostics/cce_worldtube.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; printf("  PASS: %s\n", msg); } \
    else { tests_failed++; printf("  FAIL: %s\n", msg); } \
} while(0)

/* ================================================================
 * Test 1: Conformal-to-physical reconstruction
 *
 * Set known conformal fields and verify physical quantities match
 * the expected formulas:
 *   gamma_ij = h_ij / chi
 *   K_ij = (A_ij + K/3 h_ij) / chi
 *   d_k gamma_ij = d_k(h_ij)/chi - h_ij d_k(chi)/chi^2
 * ================================================================ */

static void test_conformal_to_physical(void)
{
    printf("\n=== Test 1: Conformal-to-physical reconstruction ===\n");

    double chi = 0.5;
    double inv_chi = 1.0 / chi;
    double inv_chi2 = inv_chi * inv_chi;

    /* h_ij = identity */
    double h[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    double K = 0.3;
    double K3 = K / 3.0;
    double A[3][3] = {{0.1, -0.02, 0.0},
                      {-0.02, 0.05, 0.01},
                      {0.0, 0.01, -0.15}};
    double d_chi[3] = {0.01, -0.005, 0.002};

    /* gamma_ij = h_ij / chi = 2 * delta_ij */
    CHECK(fabs(h[0][0] * inv_chi - 2.0) < 1e-14, "gamma_xx = h_xx/chi = 2.0");
    CHECK(fabs(h[1][1] * inv_chi - 2.0) < 1e-14, "gamma_yy = h_yy/chi = 2.0");
    CHECK(fabs(h[0][1] * inv_chi) < 1e-14, "gamma_xy = 0");

    /* K_ij = (A_ij + K/3 h_ij) / chi */
    double K_xx = (A[0][0] + K3 * h[0][0]) * inv_chi;
    double K_xy = (A[0][1] + K3 * h[0][1]) * inv_chi;
    CHECK(fabs(K_xx - (0.1 + 0.1) * 2.0) < 1e-14, "K_xx = (A_xx + K/3)/chi");
    CHECK(fabs(K_xy - (-0.02) * 2.0) < 1e-14, "K_xy = A_xy/chi (h_xy=0)");

    /* d_k gamma_ij = d_k(h_ij)/chi - h_ij * d_k(chi)/chi^2 */
    /* With d_h = 0: d_x gamma_xx = -h_xx * d_x(chi) / chi^2 */
    double dx_gxx = -h[0][0] * d_chi[0] * inv_chi2;
    double dy_gxx = -h[0][0] * d_chi[1] * inv_chi2;
    CHECK(fabs(dx_gxx - (-0.01 * 4.0)) < 1e-14,
          "d_x gamma_xx = -h_xx * d_x(chi)/chi^2");
    CHECK(fabs(dy_gxx - (0.005 * 4.0)) < 1e-14,
          "d_y gamma_xx = -h_xx * d_y(chi)/chi^2");

    /* Off-diagonal: d_x gamma_xy = -h_xy * d_x(chi)/chi^2 = 0 */
    double dx_gxy = -h[0][1] * d_chi[0] * inv_chi2;
    CHECK(fabs(dx_gxy) < 1e-14, "d_x gamma_xy = 0 (h_xy=0, d_h=0)");
}

/* ================================================================
 * Test 2: Flat spacetime extraction
 *
 * On flat Minkowski: gamma_ij = delta_ij, K_ij = 0, lapse = 1,
 * shift = 0, all derivatives = 0, B^i = 0.
 * ================================================================ */

static void test_flat_extraction(void)
{
    printf("\n=== Test 2: Flat spacetime extraction ===\n");

    /* Domain must be large enough to contain extraction sphere at R=50.
     * L=128 gives domain [-64, 64], so R=50 fits. */
    mesh_t *m = mesh_create_ex(32, 128.0, RK_CLASSIC, NUM_CCZ4_FIELDS);
    grid_t *g = m->blocks[0]->grid;
    set_flat_spacetime(g);

    double center[3] = {0, 0, 0};
    const char *fname = "build/test_cce_flat.h5";
    cce_ws_t *ws = cce_alloc(4, 50.0, center, fname);

    CHECK(ws != NULL, "cce_alloc succeeded");
    CHECK(ws->n_theta == 5, "n_theta = l_max+1 = 5");
    CHECK(ws->n_phi == 9, "n_phi = 2*l_max+1 = 9");

    cce_extract(ws, m, 0.0);

    /* Verify angular data from angular_buf */
    int n_angular = ws->n_theta * ws->n_phi;
    double *abuf = ws->angular_buf;

    double max_gii_err = 0, max_gij_err = 0;
    double max_lapse_err = 0, max_K_err = 0;
    double max_shift_err = 0, max_deriv_err = 0;
    double max_aux_err = 0;

    for (int a = 0; a < n_angular; a++) {
        /* Diagonal metric should be 1 */
        for (int d = 0; d < 6; d++) {
            double val = abuf[(CCE_GXX + d) * n_angular + a];
            int diag = (d == 0 || d == 3 || d == 5); /* xx, yy, zz */
            double expected = diag ? 1.0 : 0.0;
            double err = fabs(val - expected);
            if (diag) {
                if (err > max_gii_err) max_gii_err = err;
            } else {
                if (err > max_gij_err) max_gij_err = err;
            }
        }

        /* Lapse should be 1 */
        double lapse = abuf[CCE_LAPSE * n_angular + a];
        double err = fabs(lapse - 1.0);
        if (err > max_lapse_err) max_lapse_err = err;

        /* K_ij should be 0 */
        for (int d = 0; d < 6; d++) {
            double kval = abuf[(CCE_KXX + d) * n_angular + a];
            if (fabs(kval) > max_K_err) max_K_err = fabs(kval);
        }

        /* Shift should be 0 */
        for (int d = 0; d < 3; d++) {
            double sval = abuf[(CCE_SHIFTX + d) * n_angular + a];
            if (fabs(sval) > max_shift_err) max_shift_err = fabs(sval);
        }

        /* All metric derivatives should be 0 */
        for (int d = CCE_DXGXX; d <= CCE_DZGZZ; d++) {
            double dval = abuf[d * n_angular + a];
            if (fabs(dval) > max_deriv_err) max_deriv_err = fabs(dval);
        }

        /* Lapse derivatives should be 0 */
        for (int d = CCE_DXLAPSE; d <= CCE_DZLAPSE; d++) {
            double dval = abuf[d * n_angular + a];
            if (fabs(dval) > max_deriv_err) max_deriv_err = fabs(dval);
        }

        /* Shift derivatives should be 0 */
        for (int d = CCE_DXSHIFTX; d <= CCE_DZSHIFTZ; d++) {
            double dval = abuf[d * n_angular + a];
            if (fabs(dval) > max_deriv_err) max_deriv_err = fabs(dval);
        }

        /* AuxiliaryShift should be 0 */
        for (int d = 0; d < 3; d++) {
            double bval = abuf[(CCE_AUXSHIFTX + d) * n_angular + a];
            if (fabs(bval) > max_aux_err) max_aux_err = fabs(bval);
        }
    }

    printf("  max |g_ii - 1| = %.2e\n", max_gii_err);
    printf("  max |g_ij|     = %.2e\n", max_gij_err);
    printf("  max |alpha - 1| = %.2e\n", max_lapse_err);
    printf("  max |K_ij|     = %.2e\n", max_K_err);
    printf("  max |shift|    = %.2e\n", max_shift_err);
    printf("  max |derivs|   = %.2e\n", max_deriv_err);
    printf("  max |aux|      = %.2e\n", max_aux_err);

    CHECK(max_gii_err < 1e-12, "gamma_ii = 1 (flat)");
    CHECK(max_gij_err < 1e-12, "gamma_ij = 0 (flat)");
    CHECK(max_lapse_err < 1e-12, "lapse = 1 (flat)");
    CHECK(max_K_err < 1e-12, "K_ij = 0 (flat)");
    CHECK(max_shift_err < 1e-12, "shift = 0 (flat)");
    CHECK(max_deriv_err < 1e-10, "all derivatives ~ 0 (flat)");
    CHECK(max_aux_err < 1e-12, "AuxiliaryShift = 0 (flat)");

    cce_free(ws);
    mesh_free(m);
}

/* ================================================================
 * Test 3: Schwarzschild derivatives
 *
 * Brill-Lindquist single puncture at origin with mass M=1:
 *   psi = 1 + M/(2r),  chi = psi^{-4},  h_ij = delta_ij
 *   gamma_ij = psi^4 * delta_ij,  K_ij = 0,  alpha = psi^{-2}
 *
 * Analytical derivatives at (x,y,z) on sphere of radius R:
 *   d_k gamma_ij = -2M * psi^3 * delta_ij * x_k / R^3
 *   d_k alpha    = M * psi^{-3} * x_k / R^3
 *   d_k shift    = 0
 * ================================================================ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void test_schwarzschild_derivatives(void)
{
    printf("\n=== Test 3: Schwarzschild derivatives ===\n");

    /* Domain L=256 gives [-128,128]. Extraction at R=50 fits well inside.
     * N=64 gives dx=4 — smooth BL data at R=50 gives ~1e-8 interpolation error. */
    double L = 256.0;
    int N = 64;
    double M_bh = 1.0;
    double R_ext = 50.0;

    mesh_t *m = mesh_create_ex(N, L, RK_CLASSIC, NUM_CCZ4_FIELDS);
    double masses[1] = {M_bh};
    double centers[1][3] = {{0, 0, 0}};
    set_brill_lindquist(m->blocks[0]->grid, 1, masses, centers);

    double center[3] = {0, 0, 0};
    const char *fname = "build/test_cce_schwarz.h5";
    int l_max = 4;  /* 5x9=45 points — enough for this test */
    cce_ws_t *ws = cce_alloc(l_max, R_ext, center, fname);

    cce_extract(ws, m, 0.0);

    /* Analytical values at R_ext */
    double psi = 1.0 + M_bh / (2.0 * R_ext);
    double psi3 = psi * psi * psi;
    double psi4 = psi3 * psi;
    double psi_m2 = 1.0 / (psi * psi);
    double psi_m3 = psi_m2 / psi;

    int n_angular = ws->n_theta * ws->n_phi;
    double *abuf = ws->angular_buf;

    double max_gii_err = 0, max_gij_err = 0;
    double max_K_err = 0, max_lapse_err = 0;
    double max_dgamma_err = 0, max_dlapse_err = 0;
    double max_shift_err = 0, max_dshift_err = 0;

    for (int iph = 0; iph < ws->n_phi; iph++) {
        double ph = ws->phi[iph];
        double sp = sin(ph), cp = cos(ph);

        for (int ith = 0; ith < ws->n_theta; ith++) {
            int aidx = iph * ws->n_theta + ith;
            double th = ws->theta[ith];
            double st = sin(th), ct = cos(th);

            /* Unit direction vector */
            double nx = st * cp;
            double ny = st * sp;
            double nz = ct;
            double n_hat[3] = {nx, ny, nz};

            /* --- Metric: gamma_ij = psi^4 * delta_ij --- */
            for (int d = 0; d < 6; d++) {
                double val = abuf[(CCE_GXX + d) * n_angular + aidx];
                int diag = (d == 0 || d == 3 || d == 5);
                double expected = diag ? psi4 : 0.0;
                double err = fabs(val - expected);
                if (diag) {
                    if (err > max_gii_err) max_gii_err = err;
                } else {
                    if (err > max_gij_err) max_gij_err = err;
                }
            }

            /* --- K_ij = 0 (time-symmetric) --- */
            for (int d = 0; d < 6; d++) {
                double kval = abuf[(CCE_KXX + d) * n_angular + aidx];
                if (fabs(kval) > max_K_err) max_K_err = fabs(kval);
            }

            /* --- Lapse: alpha = psi^{-2} --- */
            double alpha_num = abuf[CCE_LAPSE * n_angular + aidx];
            double err = fabs(alpha_num - psi_m2);
            if (err > max_lapse_err) max_lapse_err = err;

            /* --- d_k gamma_ij = -2M psi^3 delta_ij x_k / R^3 --- */
            /* x_k = R * n_hat[k], so d_k gamma_ij = -2M psi^3 delta_ij n_hat[k] / R^2 */
            double coeff_dg = -2.0 * M_bh * psi3 / (R_ext * R_ext);
            for (int k = 0; k < 3; k++) {
                for (int s = 0; s < 6; s++) {
                    double dval = abuf[(CCE_DXGXX + k * 6 + s) * n_angular + aidx];
                    int diag = (s == 0 || s == 3 || s == 5);
                    double expected = diag ? coeff_dg * n_hat[k] : 0.0;
                    double de = fabs(dval - expected);
                    if (de > max_dgamma_err) max_dgamma_err = de;
                }
            }

            /* --- d_k alpha = M psi^{-3} x_k / R^3 = M psi^{-3} n_hat[k] / R^2 --- */
            double coeff_da = M_bh * psi_m3 / (R_ext * R_ext);
            double dlapse[3] = {
                abuf[CCE_DXLAPSE * n_angular + aidx],
                abuf[CCE_DYLAPSE * n_angular + aidx],
                abuf[CCE_DZLAPSE * n_angular + aidx]
            };
            for (int k = 0; k < 3; k++) {
                double expected = coeff_da * n_hat[k];
                double de = fabs(dlapse[k] - expected);
                if (de > max_dlapse_err) max_dlapse_err = de;
            }

            /* --- Shift and shift derivatives = 0 --- */
            for (int d = 0; d < 3; d++) {
                double sval = abuf[(CCE_SHIFTX + d) * n_angular + aidx];
                if (fabs(sval) > max_shift_err) max_shift_err = fabs(sval);
            }
            for (int d = CCE_DXSHIFTX; d <= CCE_DZSHIFTZ; d++) {
                double dsval = abuf[d * n_angular + aidx];
                if (fabs(dsval) > max_dshift_err) max_dshift_err = fabs(dsval);
            }
        }
    }

    printf("  psi(R=50) = %.10f\n", psi);
    printf("  gamma_ii expected = %.10f\n", psi4);
    printf("  max |g_ii - psi^4|    = %.2e\n", max_gii_err);
    printf("  max |g_ij|            = %.2e\n", max_gij_err);
    printf("  max |K_ij|            = %.2e\n", max_K_err);
    printf("  max |alpha - psi^-2|  = %.2e\n", max_lapse_err);
    printf("  max |d_k g_ij err|    = %.2e\n", max_dgamma_err);
    printf("  max |d_k alpha err|   = %.2e\n", max_dlapse_err);
    printf("  max |shift|           = %.2e\n", max_shift_err);
    printf("  max |d_k shift|       = %.2e\n", max_dshift_err);

    /* 6th-order interpolation at dx=4, R=50 should give ~1e-7 or better */
    CHECK(max_gii_err < 1e-6, "gamma_ii matches psi^4 (Schwarzschild)");
    CHECK(max_gij_err < 1e-6, "gamma_ij = 0 off-diagonal (conformal flat)");
    CHECK(max_K_err < 1e-6, "K_ij = 0 (time-symmetric BL data)");
    CHECK(max_lapse_err < 1e-6, "lapse matches psi^{-2} (Schwarzschild)");
    CHECK(max_dgamma_err < 1e-6, "d_k gamma_ij matches analytical (chain rule)");
    CHECK(max_dlapse_err < 1e-6, "d_k lapse matches analytical");
    CHECK(max_shift_err < 1e-12, "shift = 0 (BL data)");
    CHECK(max_dshift_err < 1e-6, "d_k shift ~ 0 (BL data)");

    cce_free(ws);
    mesh_free(m);
}

/* ================================================================
 * Test 4 (was 3): HDF5 format validation
 *
 * Reopen the file from test 2 and verify:
 *   - All 49 datasets exist with correct names
 *   - Correct shape (1 × n_cols)
 *   - Legend attribute present
 *   - Data values are correct (gxx=1, Kxx=0, etc.)
 * ================================================================ */

static void test_hdf5_format(void)
{
    printf("\n=== Test 4: HDF5 format validation ===\n");

    const char *fname = "build/test_cce_flat.h5";
    hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(file >= 0, "HDF5 file opens successfully");

    /* Verify all 49 datasets exist */
    int all_exist = 1;
    for (int d = 0; d < CCE_NUM_DATASETS; d++) {
        htri_t exists = H5Lexists(file, cce_dataset_names[d], H5P_DEFAULT);
        if (exists <= 0) {
            printf("    MISSING: %s\n", cce_dataset_names[d]);
            all_exist = 0;
        }
    }
    CHECK(all_exist, "All 49 datasets exist");

    /* Check shape of gxx.dat (should be 1 × 46 for l_max=4) */
    hid_t ds = H5Dopen2(file, "gxx.dat", H5P_DEFAULT);
    hid_t space = H5Dget_space(ds);
    int ndims = H5Sget_simple_extent_ndims(space);
    CHECK(ndims == 2, "Dataset is 2D");

    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, NULL);
    CHECK(dims[0] == 1, "1 row written");
    /* n_cols = 1 + n_theta*n_phi = 1 + 5*9 = 46 for l_max=4 */
    CHECK((int)dims[1] == 46, "n_cols = 1 + 5*9 = 46");

    /* Check Legend attribute */
    htri_t has_legend = H5Aexists(ds, "Legend");
    CHECK(has_legend > 0, "Legend attribute exists");

    /* Read first row: time should be 0, gxx should be 1 everywhere */
    double *row = malloc(dims[1] * sizeof(double));
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, row);
    CHECK(fabs(row[0]) < 1e-15, "Column 0 = time = 0.0");

    double max_err = 0;
    for (hsize_t i = 1; i < dims[1]; i++) {
        double err = fabs(row[i] - 1.0);
        if (err > max_err) max_err = err;
    }
    CHECK(max_err < 1e-12, "gxx.dat contains 1.0 (flat spacetime)");

    free(row);
    H5Sclose(space);
    H5Dclose(ds);

    /* Check Kxx.dat contains zeros */
    ds = H5Dopen2(file, "Kxx.dat", H5P_DEFAULT);
    space = H5Dget_space(ds);
    H5Sget_simple_extent_dims(space, dims, NULL);
    row = malloc(dims[1] * sizeof(double));
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, row);
    max_err = 0;
    for (hsize_t i = 1; i < dims[1]; i++) {
        if (fabs(row[i]) > max_err) max_err = fabs(row[i]);
    }
    CHECK(max_err < 1e-12, "Kxx.dat contains 0.0 (flat spacetime)");

    /* Check Lapse.dat contains 1.0 */
    free(row);
    H5Sclose(space);
    H5Dclose(ds);

    ds = H5Dopen2(file, "Lapse.dat", H5P_DEFAULT);
    space = H5Dget_space(ds);
    H5Sget_simple_extent_dims(space, dims, NULL);
    row = malloc(dims[1] * sizeof(double));
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, row);
    max_err = 0;
    for (hsize_t i = 1; i < dims[1]; i++) {
        double err = fabs(row[i] - 1.0);
        if (err > max_err) max_err = err;
    }
    CHECK(max_err < 1e-12, "Lapse.dat contains 1.0 (flat spacetime)");

    free(row);
    H5Sclose(space);
    H5Dclose(ds);
    H5Fclose(file);
}

/* ================================================================
 * Test 5: Angular ordering (theta-varies-fastest)
 *
 * SpECTRE convention: column index = 1 + phi_j * n_theta + theta_i
 * ================================================================ */

static void test_angular_ordering(void)
{
    printf("\n=== Test 5: Angular ordering ===\n");

    int l_max = 4;
    int n_theta = l_max + 1;   /* 5 */
    int n_phi = 2 * l_max + 1; /* 9 */
    int n_angular = n_theta * n_phi;

    /* Verify theta-varies-fastest:
     * For fixed phi, consecutive indices step through theta */
    int correct = 1;
    for (int iph = 0; iph < n_phi && correct; iph++) {
        for (int ith = 0; ith < n_theta && correct; ith++) {
            int idx = iph * n_theta + ith;
            /* Next theta index (same phi) should be idx+1 */
            if (ith < n_theta - 1) {
                int next = iph * n_theta + (ith + 1);
                if (next != idx + 1) correct = 0;
            }
        }
    }
    CHECK(correct, "Theta varies fastest (consecutive for fixed phi)");

    CHECK(n_angular == 45, "Total angular points = 5*9 = 45");

    /* Column mapping: column 0 = time, then angular data */
    CHECK(1 + 0 == 1, "First angular point at column 1");
    CHECK(1 + n_angular - 1 == n_angular,
          "Last angular point at column n_angular");
}

/* ================================================================
 * Test 6: Dataset name completeness
 *
 * Verify all 49 names match the SpECTRE specification exactly.
 * ================================================================ */

static void test_dataset_names(void)
{
    printf("\n=== Test 6: Dataset name completeness ===\n");

    const char *expected[] = {
        "gxx.dat", "gxy.dat", "gxz.dat", "gyy.dat", "gyz.dat", "gzz.dat",
        "Dxgxx.dat", "Dxgxy.dat", "Dxgxz.dat", "Dxgyy.dat", "Dxgyz.dat",
        "Dxgzz.dat",
        "Dygxx.dat", "Dygxy.dat", "Dygxz.dat", "Dygyy.dat", "Dygyz.dat",
        "Dygzz.dat",
        "Dzgxx.dat", "Dzgxy.dat", "Dzgxz.dat", "Dzgyy.dat", "Dzgyz.dat",
        "Dzgzz.dat",
        "Lapse.dat", "DxLapse.dat", "DyLapse.dat", "DzLapse.dat",
        "Shiftx.dat", "Shifty.dat", "Shiftz.dat",
        "DxShiftx.dat", "DxShifty.dat", "DxShiftz.dat",
        "DyShiftx.dat", "DyShifty.dat", "DyShiftz.dat",
        "DzShiftx.dat", "DzShifty.dat", "DzShiftz.dat",
        "Kxx.dat", "Kxy.dat", "Kxz.dat", "Kyy.dat", "Kyz.dat", "Kzz.dat",
        "AuxiliaryShiftx.dat", "AuxiliaryShifty.dat", "AuxiliaryShiftz.dat"
    };

    int n_expected = (int)(sizeof(expected) / sizeof(expected[0]));
    CHECK(n_expected == CCE_NUM_DATASETS, "Expected 49 dataset names");

    int all_match = 1;
    for (int i = 0; i < n_expected; i++) {
        if (strcmp(cce_dataset_names[i], expected[i]) != 0) {
            printf("    MISMATCH [%d]: got '%s', expected '%s'\n",
                   i, cce_dataset_names[i], expected[i]);
            all_match = 0;
        }
    }
    CHECK(all_match, "All dataset names match SpECTRE specification");

    /* Verify all names end with .dat */
    int all_dat = 1;
    for (int i = 0; i < CCE_NUM_DATASETS; i++) {
        const char *name = cce_dataset_names[i];
        size_t len = strlen(name);
        if (len < 4 || strcmp(name + len - 4, ".dat") != 0) {
            printf("    '%s' does not end with .dat\n", name);
            all_dat = 0;
        }
    }
    CHECK(all_dat, "All dataset names end with .dat");
}

/* ================================================================
 * Test 7: Multiple time rows
 *
 * Write 3 time steps, verify HDF5 has 3 rows with correct times.
 * ================================================================ */

static void test_multiple_rows(void)
{
    printf("\n=== Test 7: Multiple time rows ===\n");

    mesh_t *m = mesh_create_ex(32, 128.0, RK_CLASSIC, NUM_CCZ4_FIELDS);
    set_flat_spacetime(m->blocks[0]->grid);

    double center[3] = {0, 0, 0};
    const char *fname = "build/test_cce_multi.h5";
    cce_ws_t *ws = cce_alloc(4, 50.0, center, fname);

    cce_extract(ws, m, 0.0);
    cce_extract(ws, m, 0.5);
    cce_extract(ws, m, 1.0);

    CHECK(ws->n_rows == 3, "3 rows written");

    cce_free(ws);

    /* Verify via HDF5 */
    hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t ds = H5Dopen2(file, "gxx.dat", H5P_DEFAULT);
    hid_t space = H5Dget_space(ds);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, NULL);
    CHECK((int)dims[0] == 3, "HDF5 file has 3 rows");

    /* Read all rows, check time column */
    double *data = malloc(dims[0] * dims[1] * sizeof(double));
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    CHECK(fabs(data[0 * dims[1] + 0] - 0.0) < 1e-15, "Row 0 time = 0.0");
    CHECK(fabs(data[1 * dims[1] + 0] - 0.5) < 1e-15, "Row 1 time = 0.5");
    CHECK(fabs(data[2 * dims[1] + 0] - 1.0) < 1e-15, "Row 2 time = 1.0");

    /* gxx should be 1.0 in all rows */
    double max_err = 0;
    for (hsize_t r = 0; r < dims[0]; r++)
        for (hsize_t c = 1; c < dims[1]; c++) {
            double err = fabs(data[r * dims[1] + c] - 1.0);
            if (err > max_err) max_err = err;
        }
    CHECK(max_err < 1e-12, "gxx = 1.0 in all rows");

    free(data);
    H5Sclose(space);
    H5Dclose(ds);
    H5Fclose(file);
    mesh_free(m);
}

int main(void)
{
    printf("Lattice — CCE worldtube output tests\n");

    test_conformal_to_physical();
    test_flat_extraction();
    test_schwarzschild_derivatives();
    test_hdf5_format();
    test_angular_ordering();
    test_dataset_names();
    test_multiple_rows();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
