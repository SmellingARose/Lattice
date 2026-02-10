/*
 * fields.h — Field enumeration and metadata for CCZ4 + Einstein-Maxwell
 *
 * Conformal factor convention:
 *   chi = e^{-4 phi}
 *   W   = chi^{1/2} = e^{-2 phi}
 *
 * Near punctures: chi -> 0, W -> 0.
 * In flat space:  chi = 1, W = 1.
 *
 * Field enum is APPEND-ONLY. Never reorder existing entries.
 *
 * Symmetric tensors use flat array [xx, xy, xz, yy, yz, zz] (indices 0-5).
 * Access via SYM(i,j) macro or SYM_XX..SYM_ZZ constants.
 */

#ifndef LATTICE_FIELDS_H
#define LATTICE_FIELDS_H

/* Symmetric tensor index: maps (i,j) in {0,1,2} to flat index 0-5 */
#define SYM(i, j) ((i) <= (j) ? (i) * (5 - (i)) / 2 + (j) : (j) * (5 - (j)) / 2 + (i))

enum {
    SYM_XX = 0,
    SYM_XY = 1,
    SYM_XZ = 2,
    SYM_YY = 3,
    SYM_YZ = 4,
    SYM_ZZ = 5
};

/* Base offsets for symmetric tensor field groups */
#define FIELD_GT_BASE  FIELD_GT11
#define FIELD_AT_BASE  FIELD_AT11

typedef enum {
    /* Conformal factor: chi = e^{-4 phi} */
    FIELD_CHI = 0,

    /* Conformal metric gamma_tilde_{ij} (6 components) */
    FIELD_GT11, /* xx */
    FIELD_GT12, /* xy */
    FIELD_GT13, /* xz */
    FIELD_GT22, /* yy */
    FIELD_GT23, /* yz */
    FIELD_GT33, /* zz */

    /* Trace of extrinsic curvature K */
    FIELD_TRKA,

    /* Tracefree extrinsic curvature A_tilde_{ij} (6 components) */
    FIELD_AT11, /* xx */
    FIELD_AT12, /* xy */
    FIELD_AT13, /* xz */
    FIELD_AT22, /* yy */
    FIELD_AT23, /* yz */
    FIELD_AT33, /* zz */

    /* Modified conformal connection Gamma_hat^i */
    FIELD_GHAT1,
    FIELD_GHAT2,
    FIELD_GHAT3,

    /* CCZ4 constraint damping scalar Theta */
    FIELD_THETA,

    /* Lapse */
    FIELD_ALPHA,

    /* Shift beta^i */
    FIELD_BETA1,
    FIELD_BETA2,
    FIELD_BETA3,

    /* Gamma-driver auxiliary B^i */
    FIELD_GBAUX1,
    FIELD_GBAUX2,
    FIELD_GBAUX3,

    NUM_VACUUM_FIELDS, /* = 25 */

    /* Einstein-Maxwell: electric field E^i */
    FIELD_EX = NUM_VACUUM_FIELDS,
    FIELD_EY,
    FIELD_EZ,

    /* Einstein-Maxwell: magnetic field B^i */
    FIELD_BX,
    FIELD_BY,
    FIELD_BZ,

    NUM_TOTAL_FIELDS /* = 31 */
} field_id_t;

/*
 * Background (asymptotic) value for Sommerfeld boundary conditions.
 * f ~ f0 + u(t - r) / r  as  r -> infinity.
 */
static inline double field_background_value(int field_id)
{
    switch (field_id) {
    case FIELD_CHI:   return 1.0;
    case FIELD_GT11:  return 1.0;
    case FIELD_GT22:  return 1.0;
    case FIELD_GT33:  return 1.0;
    case FIELD_ALPHA: return 1.0;
    default:          return 0.0;
    }
}

/*
 * Falloff power for Sommerfeld boundary conditions.
 * All fields fall off as 1/r.
 */
static inline double field_falloff_power(int field_id)
{
    (void)field_id;
    return 1.0;
}

#endif /* LATTICE_FIELDS_H */
