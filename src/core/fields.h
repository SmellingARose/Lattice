/*
 * Lattice — 3D Numerical Relativity
 * Field enum and loop macros for CCZ4 evolution.
 *
 * 25 evolved fields for vacuum CCZ4 (Phase 1).
 * Ref: arXiv:1106.2254, GRChombo CCZ4Vars.hpp
 */

#ifndef LATTICE_FIELDS_H
#define LATTICE_FIELDS_H

#define GR_SPACEDIM 3

/* Evolved field indices — append-only, never reorder */
enum {
    FIELD_CHI = 0,   /* conformal factor chi                   */

    FIELD_H11,       /* conformal metric h_ij (symmetric 3x3)  */
    FIELD_H12,
    FIELD_H13,
    FIELD_H22,
    FIELD_H23,
    FIELD_H33,

    FIELD_K,         /* trace of extrinsic curvature K          */

    FIELD_A11,       /* traceless conformal ext. curvature A_ij */
    FIELD_A12,
    FIELD_A13,
    FIELD_A22,
    FIELD_A23,
    FIELD_A33,

    FIELD_THETA,     /* CCZ4 constraint damping scalar          */

    FIELD_GAMMA1,    /* conformal connection functions Gamma^i   */
    FIELD_GAMMA2,
    FIELD_GAMMA3,

    FIELD_LAPSE,     /* lapse alpha                             */

    FIELD_SHIFT1,    /* shift beta^i                            */
    FIELD_SHIFT2,
    FIELD_SHIFT3,

    FIELD_B1,        /* Gamma-driver auxiliary B^i              */
    FIELD_B2,
    FIELD_B3,

    NUM_FIELDS       /* = 25                                    */
};

/* First field index for symmetric tensor components */
#define FIELD_H_START FIELD_H11
#define FIELD_A_START FIELD_A11

/*
 * Symmetric 3x3 index mapping: (i,j) -> flat index 0..5
 * Ordering: 00,01,02,11,12,22 matching GRChombo convention.
 */
static const int sym_table[3][3] = {
    {0, 1, 2},
    {1, 3, 4},
    {2, 4, 5}
};
#define SYM_INDEX(i, j) (sym_table[(i)][(j)])

/* Loop macros matching GRChombo's FOR() convention */
#define FOR1(i) for (int (i) = 0; (i) < GR_SPACEDIM; ++(i))
#define FOR2(i, j) FOR1(i) FOR1(j)
#define FOR3(i, j, k) FOR1(i) FOR1(j) FOR1(k)
#define FOR4(i, j, k, l) FOR1(i) FOR1(j) FOR1(k) FOR1(l)

/* Kronecker delta */
#define DELTA(i, j) ((i) == (j) ? 1.0 : 0.0)

#endif /* LATTICE_FIELDS_H */
