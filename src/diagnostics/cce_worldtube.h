/*
 * Lattice — 3D Numerical Relativity
 * CCE worldtube output for SpECTRE.
 *
 * Writes AdmMetricNodal HDF5 files compatible with SpECTRE's
 * Cauchy-Characteristic Evolution (CCE) pipeline. The worldtube data
 * consists of 49 datasets (spatial metric, derivatives, lapse, shift,
 * extrinsic curvature) on a 2-sphere at finite extraction radius.
 *
 * Pipeline: Lattice → CceR####.h5 → SpECTRE PreprocessCceWorldtube
 *           → SpECTRE CharacteristicExtract → strain h at scri+
 *
 * Ref: SpECTRE CCE documentation (AdmMetricNodal format)
 */

#ifndef LATTICE_CCE_WORLDTUBE_H
#define LATTICE_CCE_WORLDTUBE_H

#ifdef LATTICE_HDF5

#include <hdf5.h>
#include "../core/grid.h"

struct mesh_s;  /* forward declaration */

#define CCE_NUM_DATASETS 49

/* Dataset indices for structured access (SpECTRE AdmMetricNodal ordering) */
enum {
    CCE_GXX = 0, CCE_GXY, CCE_GXZ, CCE_GYY, CCE_GYZ, CCE_GZZ,
    CCE_DXGXX, CCE_DXGXY, CCE_DXGXZ, CCE_DXGYY, CCE_DXGYZ, CCE_DXGZZ,
    CCE_DYGXX, CCE_DYGXY, CCE_DYGXZ, CCE_DYGYY, CCE_DYGYZ, CCE_DYGZZ,
    CCE_DZGXX, CCE_DZGXY, CCE_DZGXZ, CCE_DZGYY, CCE_DZGYZ, CCE_DZGZZ,
    CCE_LAPSE, CCE_DXLAPSE, CCE_DYLAPSE, CCE_DZLAPSE,
    CCE_SHIFTX, CCE_SHIFTY, CCE_SHIFTZ,
    CCE_DXSHIFTX, CCE_DXSHIFTY, CCE_DXSHIFTZ,
    CCE_DYSHIFTX, CCE_DYSHIFTY, CCE_DYSHIFTZ,
    CCE_DZSHIFTX, CCE_DZSHIFTY, CCE_DZSHIFTZ,
    CCE_KXX, CCE_KXY, CCE_KXZ, CCE_KYY, CCE_KYZ, CCE_KZZ,
    CCE_AUXSHIFTX, CCE_AUXSHIFTY, CCE_AUXSHIFTZ
};

/* Dataset names (SpECTRE AdmMetricNodal format) */
extern const char *cce_dataset_names[CCE_NUM_DATASETS];

typedef struct {
    int    n_theta, n_phi, l_max;
    double radius, center[3];
    double *theta, *gl_weights;     /* GL nodes/weights [n_theta] */
    double *phi;                    /* uniform phi [n_phi]        */
    /* HDF5 handles */
    hid_t  file_id;
    hid_t  dataset_ids[CCE_NUM_DATASETS];
    int    n_rows;                  /* rows written so far */
    int    n_cols;                  /* 1 + n_theta * n_phi */
    double *row_buf;               /* scratch for one row [n_cols] */
    double *angular_buf;           /* scratch [CCE_NUM_DATASETS * n_angular] */
} cce_ws_t;

/*
 * Allocate workspace and create HDF5 file for CCE worldtube output.
 *   l_max: angular resolution (n_theta = l_max+1, n_phi = 2*l_max+1)
 *   radius: extraction sphere radius
 *   center[3]: coordinate center of sphere
 *   filename: output HDF5 file path (e.g. "CceR0100.h5")
 */
cce_ws_t *cce_alloc(int l_max, double radius, const double center[3],
                     const char *filename);

/* Free workspace and close HDF5 file */
void cce_free(cce_ws_t *ws);

/*
 * Extract ADM quantities on the worldtube sphere and write one time row
 * to each of the 49 HDF5 datasets.
 *
 * For each angular point on the GL×uniform sphere:
 *   1. Interpolate conformal CCZ4 fields + spatial derivatives
 *   2. Reconstruct physical ADM quantities (gamma_ij, K_ij, lapse, shift)
 *   3. Write in theta-varies-fastest order
 */
void cce_extract(cce_ws_t *ws, const struct mesh_s *m, double time);

#endif /* LATTICE_HDF5 */
#endif /* LATTICE_CCE_WORLDTUBE_H */
