/*
 * Lattice — 3D Numerical Relativity
 * AMR block allocation and deallocation.
 *
 * Ref: Athena++ src/mesh/meshblock.cpp (MeshBlock constructor)
 * Ref: AthenaK  src/mesh/meshblock.cpp
 */

#include "block.h"
#include "../core/fields.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PAGE_ALIGN 4096

/*
 * Allocate a fields-only grid with exact N (no padding to multiple of 16).
 * Used for coarse_buf which may have N < 16 (e.g. N=8 for N_block=16).
 * Only allocates fields_block; rhs/scratch/accum are NULL.
 *
 * Unlike grid_alloc(), this does not pad N and does not allocate
 * RK scratch arrays since coarse_buf is never time-evolved.
 */
static grid_t *coarse_buf_alloc(int N, double dx, int n_fields)
{
    grid_t *g = calloc(1, sizeof(grid_t));
    if (!g) {
        fprintf(stderr, "coarse_buf_alloc: calloc failed\n");
        exit(1);
    }

    g->N      = N;
    g->ghost  = GHOST_WIDTH;
    g->Ntotal = N + 2 * GHOST_WIDTH;
    g->L      = N * dx;
    g->dx     = dx;
    g->npoints = (size_t)g->Ntotal * g->Ntotal * g->Ntotal;
    g->n_fields = n_fields;

    /* Allocate only fields block (page-aligned for potential GPU use) */
    size_t block_bytes = (size_t)n_fields * g->npoints * sizeof(double);
    void *ptr = NULL;
    if (posix_memalign(&ptr, PAGE_ALIGN, block_bytes) != 0) {
        fprintf(stderr, "coarse_buf_alloc: posix_memalign failed\n");
        exit(1);
    }
    memset(ptr, 0, block_bytes);
    g->fields_block = (double *)ptr;

    for (int f = 0; f < n_fields; f++)
        g->fields[f] = g->fields_block + f * g->npoints;
    for (int f = n_fields; f < NUM_FIELDS; f++)
        g->fields[f] = NULL;

    /* No RK arrays needed for coarse_buf */
    g->rhs_block = NULL;
    g->scratch_block = NULL;
    g->accum_block = NULL;
    for (int f = 0; f < NUM_FIELDS; f++) {
        g->rhs[f] = NULL;
        g->scratch[f] = NULL;
        g->accum[f] = NULL;
    }

    return g;
}

/*
 * 26-neighbor offset table: (ox1, ox2, ox3) for each neighbor.
 * Order: 6 faces, 12 edges, 8 corners.
 * Ref: Athena++ bvals_base.cpp SearchAndSetNeighbors enumeration.
 */
#ifdef LATTICE_GPU
#pragma omp declare target
#endif
const int nbr_offset[NUM_NEIGHBORS][3] = {
    /* 6 faces */
    {-1,  0,  0}, { 1,  0,  0},   /* x-minus, x-plus */
    { 0, -1,  0}, { 0,  1,  0},   /* y-minus, y-plus */
    { 0,  0, -1}, { 0,  0,  1},   /* z-minus, z-plus */
    /* 12 edges */
    {-1, -1,  0}, { 1, -1,  0}, {-1,  1,  0}, { 1,  1,  0},  /* xy edges */
    {-1,  0, -1}, { 1,  0, -1}, {-1,  0,  1}, { 1,  0,  1},  /* xz edges */
    { 0, -1, -1}, { 0,  1, -1}, { 0, -1,  1}, { 0,  1,  1},  /* yz edges */
    /* 8 corners */
    {-1, -1, -1}, { 1, -1, -1}, {-1,  1, -1}, { 1,  1, -1},
    {-1, -1,  1}, { 1, -1,  1}, {-1,  1,  1}, { 1,  1,  1}
};
#ifdef LATTICE_GPU
#pragma omp end declare target
#endif

block_t *block_alloc(int id, int level, int N_block, double dx,
                     const double origin[3], rk_method_t method,
                     int n_fields)
{
    block_t *b = calloc(1, sizeof(block_t));
    if (!b) {
        fprintf(stderr, "block_alloc: calloc failed\n");
        exit(1);
    }

    /* Allocate grid: L_block = N_block * dx gives correct dx after padding.
     * N_block should be a multiple of 16 so grid_alloc doesn't change it. */
    double L_block = N_block * dx;
    b->grid = grid_alloc_ex(N_block, L_block, method, n_fields);

    /* Verify grid_alloc didn't pad N (it shouldn't if N_block is multiple of 16) */
    if (b->grid->N != N_block) {
        fprintf(stderr, "block_alloc: grid_alloc padded N=%d to %d. "
                "Use N_block as a multiple of 16.\n", N_block, b->grid->N);
        exit(1);
    }

    /* Allocate coarse buffer at half resolution for level > 0 blocks.
     * N_c = N_block / 2, dx_c = 2 * dx (same resolution as coarse-level).
     * Uses coarse_buf_alloc (exact N, no padding, fields-only).
     * Ref: AthenaK coarse-buffer architecture (block-local AMR ghost fill) */
    if (level > 0 && N_block >= 2) {
        int N_c = N_block / 2;
        double dx_c = 2.0 * dx;
        b->coarse_buf = coarse_buf_alloc(N_c, dx_c, n_fields);
    } else {
        b->coarse_buf = NULL;
    }

    b->id = id;
    b->loc.lx1   = 0;  /* caller sets via mesh_create */
    b->loc.lx2   = 0;
    b->loc.lx3   = 0;
    b->loc.level  = level;
    b->origin[0] = origin[0];
    b->origin[1] = origin[1];
    b->origin[2] = origin[2];

    /* Tree links: root block, no children (leaf) */
    b->parent_id = -1;
    for (int c = 0; c < 8; c++)
        b->child_ids[c] = -1;
    b->is_leaf = 1;

    /* Neighbors: default to no neighbor (physical boundary).
     * nblevel[3][3][3]: -1 = no neighbor, except self = own level.
     * Ref: Athena++ bvals_base.cpp initialization */
    for (int n = 0; n < NUM_NEIGHBORS; n++)
        b->neighbor_ids[n] = -1;

    memset(b->nblevel, -1, sizeof(b->nblevel));
    b->nblevel[1][1][1] = level;  /* self */

    memset(b->on_boundary, 0, sizeof(b->on_boundary));

    /* Subcycling state: initialized to NULL, allocated on demand */
    b->time = 0.0;
    b->interp_order = 0;
    b->time_old = 0.0;
    b->dt_old = 0.0;
    for (int f = 0; f < NUM_FIELDS; f++) {
        b->fields_old[f] = NULL;
        b->fields_older[f] = NULL;
        b->rhs_old[f] = NULL;
        b->rhs_older[f] = NULL;
    }
    b->fields_old_block = NULL;
    b->fields_older_block = NULL;
    b->rhs_old_block = NULL;
    b->rhs_older_block = NULL;

    return b;
}

void block_free(block_t *b)
{
    if (!b) return;
    block_free_fields_old(b);
    grid_free(b->coarse_buf);
    grid_free(b->grid);
    free(b);
}

void block_alloc_fields_old(block_t *b)
{
    if (!b || !b->grid || b->fields_old_block) return;

    size_t npts = b->grid->npoints;
    int nf = b->grid->n_fields;
    size_t block_bytes = (size_t)nf * npts * sizeof(double);

    /* Allocate 4 contiguous blocks for quartic temporal interpolation:
     *   fields_old (U_n), fields_older (U_{n-1}),
     *   rhs_old (F_n), rhs_older (F_{n-1}).
     * Ref: Chombo AMRLevel old/new data pattern. */
    double **backing[4] = { &b->fields_old_block, &b->fields_older_block,
                            &b->rhs_old_block, &b->rhs_older_block };
    double *ptrs[4][NUM_FIELDS];

    for (int a = 0; a < 4; a++) {
        void *ptr = NULL;
        if (posix_memalign(&ptr, PAGE_ALIGN, block_bytes) != 0) {
            fprintf(stderr, "block_alloc_fields_old: posix_memalign failed\n");
            exit(1);
        }
        memset(ptr, 0, block_bytes);
        *backing[a] = (double *)ptr;
        for (int f = 0; f < nf; f++)
            ptrs[a][f] = *backing[a] + f * npts;
    }

    for (int f = 0; f < nf; f++) {
        b->fields_old[f]   = ptrs[0][f];
        b->fields_older[f] = ptrs[1][f];
        b->rhs_old[f]      = ptrs[2][f];
        b->rhs_older[f]    = ptrs[3][f];
    }
}

void block_free_fields_old(block_t *b)
{
    if (!b) return;

    free(b->fields_old_block);
    free(b->fields_older_block);
    free(b->rhs_old_block);
    free(b->rhs_older_block);

    b->fields_old_block = NULL;
    b->fields_older_block = NULL;
    b->rhs_old_block = NULL;
    b->rhs_older_block = NULL;

    for (int f = 0; f < NUM_FIELDS; f++) {
        b->fields_old[f] = NULL;
        b->fields_older[f] = NULL;
        b->rhs_old[f] = NULL;
        b->rhs_older[f] = NULL;
    }
}

void block_save_old(block_t *b)
{
    if (!b || !b->grid || !b->fields_old_block) return;

    size_t npts = b->grid->npoints;

    /* Save dt of the step that brought us to the current time.
     * dt_old = t_current - t_old (from previous save). */
    b->dt_old = b->time - b->time_old;

    /* Rotate time history via pointer swaps (zero-copy).
     * fields_older ← fields_old, rhs_older ← rhs_old.
     * Ref: Chombo AMRLevel m_old_data save pattern. */

    int nf = b->grid->n_fields;

    /* Swap fields_older ↔ fields_old (block + per-field pointers) */
    double *tmp_block = b->fields_older_block;
    b->fields_older_block = b->fields_old_block;
    b->fields_old_block = tmp_block;
    for (int f = 0; f < nf; f++) {
        double *tmp = b->fields_older[f];
        b->fields_older[f] = b->fields_old[f];
        b->fields_old[f] = tmp;
    }

    /* Swap rhs_older ↔ rhs_old */
    tmp_block = b->rhs_older_block;
    b->rhs_older_block = b->rhs_old_block;
    b->rhs_old_block = tmp_block;
    for (int f = 0; f < nf; f++) {
        double *tmp = b->rhs_older[f];
        b->rhs_older[f] = b->rhs_old[f];
        b->rhs_old[f] = tmp;
    }

    /* Save current fields → fields_old (memcpy into the swapped buffer) */
    for (int f = 0; f < nf; f++)
        memcpy(b->fields_old[f], b->grid->fields[f], npts * sizeof(double));

    /* Save current time as "old" */
    b->time_old = b->time;

    /* Ramp interpolation order: 0 → 1 → 4.
     * Step 0: no history → copy only.
     * Step 1: have U_n and U_{n+1} → linear.
     * Step 2+: have U_{n-1}, U_n, U_{n+1}, F_n, F_{n-1} → quartic. */
    if (b->interp_order == 0)
        b->interp_order = 1;
    else if (b->interp_order == 1)
        b->interp_order = 4;
}

void block_reset_interp(block_t *b)
{
    if (!b) return;
    b->interp_order = 0;
}

void block_save_rhs_old(block_t *b, const double *const *rhs_src, size_t npoints)
{
    if (!b || !b->rhs_old_block) return;
    int nf = b->grid->n_fields;
    for (int f = 0; f < nf; f++)
        memcpy(b->rhs_old[f], rhs_src[f], npoints * sizeof(double));
}

void block_time_interp(const block_t *b, double frac,
                        double *out[], size_t npoints)
{
    if (!b || !b->fields_old_block || !b->grid) return;
    int nf = b->grid->n_fields;

    if (b->interp_order == 0) {
        /* No history: copy fields_old (pre-step state) */
        for (int f = 0; f < nf; f++)
            memcpy(out[f], b->fields_old[f], npoints * sizeof(double));

    } else if (b->interp_order <= 1 || !b->fields_older_block) {
        /* 1st-order: linear interpolation p(s) = (1-s)*U_old + s*U_new */
        double one_minus_frac = 1.0 - frac;
        for (int f = 0; f < nf; f++) {
            const double *old_f = b->fields_old[f];
            const double *new_f = b->grid->fields[f];
            double *dst = out[f];
            for (size_t i = 0; i < npoints; i++)
                dst[i] = one_minus_frac * old_f[i] + frac * new_f[i];
        }

    } else {
        /* 4th-order quartic interpolation through 5 constraints:
         *   p(0) = U_n, p(1) = U_{n+1}, p(-1) = U_{n-1},
         *   p'(0) = dt*F_n, p'(-1) = dt*F_{n-1}
         *
         * s ∈ [0,1]: 0 = old (U_n), 1 = new (U_{n+1}).
         * Assumes uniform time steps (dt_old = dt), always true in subcycling.
         *
         * Verified via SymPy (tools/compute_amr_weights.py).
         * Ref: Berger & Oliger (1984), Athena++ temporal prolongation. */
        double s = frac;
        double s2 = s * s, s3 = s2 * s, s4 = s3 * s;

        double w_Un   = 1.0 - 2.0*s2 + s4;
        double w_Un1  = 0.25*s2 + 0.5*s3 + 0.25*s4;
        double w_Unm1 = 1.75*s2 - 0.5*s3 - 1.25*s4;
        double w_Fn   = s + s2 - s3 - s4;
        double w_Fnm1 = 0.5*s2 - 0.5*s4;

        /* dt for scaling RHS terms: time between U_n and U_{n+1} */
        double dt = b->time - b->time_old;
        if (dt <= 0.0) dt = b->dt_old;  /* fallback */

        double w_Fn_dt   = w_Fn * dt;
        double w_Fnm1_dt = w_Fnm1 * dt;

        for (int f = 0; f < nf; f++) {
            const double *u_n   = b->fields_old[f];
            const double *u_n1  = b->grid->fields[f];
            const double *u_nm1 = b->fields_older[f];
            const double *f_n   = b->rhs_old[f];
            const double *f_nm1 = b->rhs_older[f];
            double *dst = out[f];

            for (size_t i = 0; i < npoints; i++)
                dst[i] = w_Un * u_n[i] + w_Un1 * u_n1[i]
                        + w_Unm1 * u_nm1[i]
                        + w_Fn_dt * f_n[i] + w_Fnm1_dt * f_nm1[i];
        }
    }
}
