/*
 * Lattice — 3D Numerical Relativity
 * AMR block allocation and deallocation.
 *
 * Ref: Athena++ src/mesh/meshblock.cpp (MeshBlock constructor)
 * Ref: AthenaK  src/mesh/meshblock.cpp
 */

#include "block.h"
#include "prolongation.h"
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

    /* Use COARSE_BUF_GHOST (= 5 with current stencil parameters) instead of
     * GHOST_WIDTH (= 4). The wider ghost zone ensures the 7-point prolongation
     * stencil can fill ALL fine ghost cells including the outermost (fi=0).
     * Without this, the outermost 2 fine ghost cells at coarse-fine boundaries
     * are skipped by prolongation, leaving stale/zero data that the KO
     * dissipation stencil reads — causing NaN from chi=0 division. */
    g->N      = N;
    g->ghost  = COARSE_BUF_GHOST;
    g->Ntotal = N + 2 * COARSE_BUF_GHOST;
    g->L      = N * dx;
    g->dx     = dx;
    g->inv_dx = 1.0 / dx;
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
    b->time_old = 0.0;
    for (int f = 0; f < NUM_FIELDS; f++) {
        b->fields_old[f] = NULL;
        b->taylor_a1[f] = NULL;
        b->taylor_a2[f] = NULL;
        b->taylor_a3[f] = NULL;
    }
    b->fields_old_block = NULL;
    b->taylor_block = NULL;

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

    /* Allocate fields_old (U_n snapshot) */
    void *ptr = NULL;
    if (posix_memalign(&ptr, PAGE_ALIGN, block_bytes) != 0) {
        fprintf(stderr, "block_alloc_fields_old: posix_memalign failed\n");
        exit(1);
    }
    memset(ptr, 0, block_bytes);
    b->fields_old_block = (double *)ptr;
    for (int f = 0; f < nf; f++)
        b->fields_old[f] = b->fields_old_block + f * npts;

    /* Allocate contiguous Taylor coefficient buffers (a1 + a2 + a3).
     * Ref: Chombo TimeInterpolatorRK4 — 3 coefficient arrays. */
    size_t taylor_bytes = 3 * block_bytes;
    ptr = NULL;
    if (posix_memalign(&ptr, PAGE_ALIGN, taylor_bytes) != 0) {
        fprintf(stderr, "block_alloc_fields_old: taylor posix_memalign failed\n");
        exit(1);
    }
    memset(ptr, 0, taylor_bytes);
    b->taylor_block = (double *)ptr;
    for (int f = 0; f < nf; f++) {
        b->taylor_a1[f] = b->taylor_block + 0 * nf * npts + f * npts;
        b->taylor_a2[f] = b->taylor_block + 1 * nf * npts + f * npts;
        b->taylor_a3[f] = b->taylor_block + 2 * nf * npts + f * npts;
    }
}

void block_free_fields_old(block_t *b)
{
    if (!b) return;

    free(b->fields_old_block);
    free(b->taylor_block);

    b->fields_old_block = NULL;
    b->taylor_block = NULL;

    for (int f = 0; f < NUM_FIELDS; f++) {
        b->fields_old[f] = NULL;
        b->taylor_a1[f] = NULL;
        b->taylor_a2[f] = NULL;
        b->taylor_a3[f] = NULL;
    }
}

void block_save_old(block_t *b)
{
    if (!b || !b->grid || !b->fields_old_block) return;

    size_t npts = b->grid->npoints;
    int nf = b->grid->n_fields;

    /* Save current fields → fields_old */
    for (int f = 0; f < nf; f++)
        memcpy(b->fields_old[f], b->grid->fields[f], npts * sizeof(double));

    /* Zero Taylor coefficient buffers for the upcoming step's accumulation.
     * Ref: Chombo TimeInterpolatorRK4::setDt zeroes coefficients. */
    if (b->taylor_block) {
        size_t taylor_bytes = 3 * (size_t)nf * npts * sizeof(double);
        memset(b->taylor_block, 0, taylor_bytes);
    }

    b->time_old = b->time;
}

/* Chombo TimeInterpolatorRK4 coefficient matrix.
 * coeffs[term][stage] — multiply by dt * rhs to accumulate.
 * Ref: Chombo/lib/src/AMRTimeDependent/TimeInterpolatorRK4.cpp */
static const double taylor_coeffs[3][4] = {
    {  1.0,          0.0,          0.0,          0.0         },  /* a1 */
    { -3.0/2.0,      1.0,          1.0,         -1.0/2.0     },  /* a2 */
    {  2.0/3.0,     -2.0/3.0,     -2.0/3.0,      2.0/3.0     }   /* a3 */
};

void block_accumulate_taylor(block_t *b, const double *const *rhs_src,
                              int stage, double dt, size_t npoints)
{
    if (!b || !b->taylor_block) return;
    int nf = b->grid->n_fields;

    double c1 = taylor_coeffs[0][stage] * dt;
    double c2 = taylor_coeffs[1][stage] * dt;
    double c3 = taylor_coeffs[2][stage] * dt;

    for (int f = 0; f < nf; f++) {
        const double *rhs = rhs_src[f];
        double *a1 = b->taylor_a1[f];
        double *a2 = b->taylor_a2[f];
        double *a3 = b->taylor_a3[f];
        for (size_t i = 0; i < npoints; i++) {
            double r = rhs[i];
            a1[i] += c1 * r;
            a2[i] += c2 * r;
            a3[i] += c3 * r;
        }
    }
}

void block_time_interp(const block_t *b, double frac,
                        double *out[], size_t npoints)
{
    if (!b || !b->fields_old_block || !b->grid) return;
    int nf = b->grid->n_fields;

    /* Cubic Taylor Horner evaluation:
     *   U(θ) = U_n + θ*(a1 + θ*(a2 + θ*a3))
     *
     * When Taylor buffers are zero (first step before any accumulation),
     * this gives U(θ) = U_n (0th-order copy), which is correct.
     *
     * Ref: Chombo TimeInterpolatorRK4::intermediate (Horner form). */
    for (int f = 0; f < nf; f++) {
        const double *u_n = b->fields_old[f];
        const double *a1  = b->taylor_a1[f];
        const double *a2  = b->taylor_a2[f];
        const double *a3  = b->taylor_a3[f];
        double *dst = out[f];

        if (a1) {
            for (size_t i = 0; i < npoints; i++)
                dst[i] = u_n[i] + frac * (a1[i] + frac * (a2[i] + frac * a3[i]));
        } else {
            /* Taylor not allocated — copy U_n */
            memcpy(dst, u_n, npoints * sizeof(double));
        }
    }
}
