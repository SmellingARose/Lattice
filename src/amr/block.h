/*
 * Lattice — 3D Numerical Relativity
 * AMR block: grid_t wrapped with oct-tree metadata.
 *
 * Follows Athena++ MeshBlock pattern (Stone et al. 2020, ApJS 249, 4):
 *   - LogicalLocation for tree addressing
 *   - nblevel[3][3][3] neighbor level table
 *   - 26 same-level neighbors (expands to 56 for cross-level in Stage 4)
 *   - Child octant indexing: n = ox1 + (ox2<<1) + (ox3<<2)
 *
 * Ref: Athena++ src/mesh/meshblock.hpp
 * Ref: AthenaK  src/mesh/meshblock.hpp
 */

#ifndef LATTICE_BLOCK_H
#define LATTICE_BLOCK_H

#include "../core/device.h"
#include "../core/grid.h"
#include "../core/params.h"

EXTERN_C_BEGIN

/* Number of same-level neighbors in 3D (6 face + 12 edge + 8 corner).
 * Stage 4 will expand to 56 for cross-level (4 sub-faces + 2 sub-edges). */
#define NUM_NEIGHBORS 26

/*
 * Logical location in the oct-tree (Athena++ pattern).
 * (lx1, lx2, lx3) are integer coordinates at the given level.
 * Child coords = parent*2 + {0,1} in each direction.
 * Ref: Athena++ athena.hpp, AthenaK mesh.hpp
 */
typedef struct {
    int lx1, lx2, lx3;  /* logical x,y,z at this level          */
    int level;           /* refinement level (0 = coarsest root)  */
} logical_location_t;

/*
 * 26-neighbor offset table.
 * Indexed 0..25: 6 faces, then 12 edges, then 8 corners.
 * Each entry is (ox1, ox2, ox3) in {-1, 0, +1}.
 *
 * The nblevel[3][3][3] table uses (ox3+1, ox2+1, ox1+1) indexing
 * following Athena++ convention, so nblevel[1][1][1] = self level.
 */
extern const int nbr_offset[NUM_NEIGHBORS][3];

/* Neighbor type classification */
typedef enum {
    NBR_FACE = 0,    /* 6 face neighbors   */
    NBR_EDGE = 1,    /* 12 edge neighbors  */
    NBR_CORNER = 2   /* 8 corner neighbors */
} neighbor_type_t;

/* Returns the type of neighbor at index n (0..25) */
static inline neighbor_type_t nbr_type(int n)
{
    if (n < 6)  return NBR_FACE;
    if (n < 18) return NBR_EDGE;
    return NBR_CORNER;
}

/*
 * AMR block: one grid_t plus tree metadata.
 * Leaf blocks hold field data; non-leaf blocks are tree-only (grid == NULL).
 */
typedef struct block_s {
    grid_t  *grid;                 /* field data (NULL if non-leaf)           */
    grid_t  *coarse_buf;           /* half-res buffer for AMR (NULL if L==0)  */
    int      id;                   /* unique block ID (Z-order within level)  */
    logical_location_t loc;        /* position in oct-tree                    */
    double   origin[3];            /* physical coords of block's low corner   */

    /* Tree links (Athena++ MeshBlockTree pattern) */
    int      parent_id;            /* -1 if root level                        */
    int      child_ids[8];         /* -1 if leaf. Octant: ox1+(ox2<<1)+(ox3<<2) */
    int      is_leaf;              /* 1 if active (holds field data)           */

    /* Neighbor info (Athena++ bvals pattern) */
    int      neighbor_ids[NUM_NEIGHBORS];  /* -1 = physical boundary          */
    int      nblevel[3][3][3];     /* neighbor level table. [oz+1][oy+1][ox+1] */
                                   /* nblevel[1][1][1] = self level            */
                                   /* -1 = no neighbor (physical boundary)     */
    int      on_boundary[6];       /* 1 if face touches domain boundary        */
                                   /* [0]=x-, [1]=x+, [2]=y-, [3]=y+, [4]=z-, [5]=z+ */

    /* Subcycling (Berger-Oliger) state for cubic Taylor temporal interpolation.
     *
     * Chombo TimeInterpolatorRK4 pattern: during each RK4 stage on the coarse
     * level, accumulate 3 Taylor coefficient buffers from the stage's RHS.
     * Finer levels evaluate: U(θ) = U_n + θ*(a1 + θ*(a2 + θ*a3))
     *
     * Coefficient matrix (coeffs[term][stage], each × dt × k_s):
     *   a1:  { 1,      0,      0,      0     }
     *   a2:  {-3/2,    1,      1,     -1/2   }
     *   a3:  { 2/3,   -2/3,   -2/3,    2/3   }
     *
     * At θ=0: U_n. At θ=1: U_n + dt/6*(k1+2k2+2k3+k4) = U_{n+1}.
     * 3rd-degree polynomial, O(dt⁴) accurate.
     *
     * Only allocated for leaf blocks at level < max_level.
     *
     * Ref: Chombo TimeInterpolatorRK4.cpp, GRChombo GRAMRLevel.cpp. */
    double   time;                 /* simulation time this block is at          */
    double   time_old;             /* time at t_n (before current step)         */

    double  *fields_old[NUM_FIELDS]; /* state U_n at start of step             */
    double  *fields_old_block;     /* contiguous backing allocation             */
    double  *taylor_a1[NUM_FIELDS];  /* cubic Taylor coefficient a1            */
    double  *taylor_a2[NUM_FIELDS];  /* cubic Taylor coefficient a2            */
    double  *taylor_a3[NUM_FIELDS];  /* cubic Taylor coefficient a3            */
    double  *taylor_block;         /* contiguous backing for a1+a2+a3          */
} block_t;

/* Physical coordinate of grid point i in direction dir within a block.
 * Ref: Athena++ Coordinates::CellCenterX1() pattern */
#define BLOCK_COORD(blk, dir, i) \
    ((blk)->origin[(dir)] + ((i) - GHOST_WIDTH + 0.5) * (blk)->grid->dx)

/*
 * Allocate a block with field data.
 *   id:       unique block ID
 *   level:    refinement level (0 = coarsest)
 *   N_block:  interior cells per side (e.g. 32)
 *   dx:       grid spacing at this level
 *   origin:   physical coordinates of block's low corner [3]
 *   method:   RK method (determines memory allocation)
 *   n_fields: active field count (<= NUM_FIELDS)
 *
 * Internally calls grid_alloc_ex(N_block, N_block*dx, method, n_fields).
 * All tree links initialized to -1, is_leaf = 1.
 */
block_t *block_alloc(int id, int level, int N_block, double dx,
                     const double origin[3], rk_method_t method,
                     int n_fields);

/* Free a block and its grid data */
void block_free(block_t *b);

/*
 * Allocate fields_old + Taylor coefficient arrays for cubic temporal
 * interpolation (subcycling). Only needed for leaf blocks at level < max_level.
 * No-op if already allocated.
 *
 * Ref: Chombo TimeInterpolatorRK4.
 */
void block_alloc_fields_old(block_t *b);

/* Free all temporal interpolation arrays. Safe to call if already NULL. */
void block_free_fields_old(block_t *b);

/*
 * Save current fields → fields_old and zero Taylor coefficient buffers.
 * Called before advancing a level so finer levels can time-interpolate.
 *
 * Ref: Chombo TimeInterpolatorRK4::setDt + saveInitialSoln.
 */
void block_save_old(block_t *b);

/*
 * Accumulate RK4 stage RHS into Taylor coefficient buffers.
 * Called after each stage's RHS + BC evaluation.
 *
 * Chombo coefficient matrix (coeffs[term][stage] × dt):
 *   a1:  { 1,      0,      0,      0     }
 *   a2:  {-3/2,    1,      1,     -1/2   }
 *   a3:  { 2/3,   -2/3,   -2/3,    2/3   }
 *
 * Ref: Chombo TimeInterpolatorRK4::saveRHS.
 */
void block_accumulate_taylor(block_t *b, const double *const *rhs_src,
                              int stage, double dt, size_t npoints);

/*
 * Cubic Taylor temporal interpolation (Horner form).
 * U(θ) = U_n + θ*(a1 + θ*(a2 + θ*a3))
 *
 * frac in [0,1]: 0 = old state (t_n), 1 = new state (t_{n+1}).
 * When Taylor buffers are zero (first step), gives U_n (0th-order copy).
 * Writes into the provided out[] array (caller-owned, same layout as fields[]).
 *
 * Ref: Chombo TimeInterpolatorRK4::intermediate (Horner evaluation).
 */
void block_time_interp(const block_t *b, double frac,
                        double *out[], size_t npoints);

EXTERN_C_END

#endif /* LATTICE_BLOCK_H */
