/*
 * Benchmark: mesh_find_block_at with hash vs linear scan.
 * Creates a multi-level AMR mesh and times repeated point lookups.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/amr/mesh.h"
#include "../src/amr/refine.h"
#include "../src/amr/ghost_exchange.h"
#include "../src/initial_data/puncture.h"

static double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    int N = 16;
    double L = 64.0;
    mesh_t *m = mesh_create(N, L, RK_CK45);

    /* Set up puncture at origin for chi-gradient refinement */
    double mass = 1.0;
    double center[1][3] = {{0.0, 0.0, 0.0}};
    set_brill_lindquist(m->blocks[0]->grid, 1, &mass, center);

    /* Refine 3 levels — enough blocks to show speedup, fits in memory */
    amr_params_t ap = {0};
    ap.chi_refine = 0.1;
    ap.max_level = 3;
    for (int iter = 0; iter < 3; iter++) {
        mesh_regrid(m, &ap);
        mesh_rebuild_neighbors(m);
        ghost_exchange_all_blocks(m);
        for (int i = 0; i < m->num_blocks; i++) {
            if (m->blocks[i] && m->blocks[i]->is_leaf)
                set_brill_lindquist_global(m->blocks[i]->grid,
                    m->blocks[i]->origin, 1, &mass, center);
        }
    }

    int n_blocks = m->num_blocks;
    int n_leaves = 0;
    for (int i = 0; i < n_blocks; i++)
        if (m->blocks[i] && m->blocks[i]->is_leaf) n_leaves++;

    printf("Mesh: %d blocks (%d leaves), max_level=%d\n",
           n_blocks, n_leaves, m->max_level);

    /* Generate random test points inside the domain */
    int N_LOOKUPS = 50000;
    double *xs = malloc(N_LOOKUPS * sizeof(double));
    double *ys = malloc(N_LOOKUPS * sizeof(double));
    double *zs = malloc(N_LOOKUPS * sizeof(double));
    srand(42);
    for (int i = 0; i < N_LOOKUPS; i++) {
        xs[i] = (rand() / (double)RAND_MAX - 0.5) * L * 0.99;
        ys[i] = (rand() / (double)RAND_MAX - 0.5) * L * 0.99;
        zs[i] = (rand() / (double)RAND_MAX - 0.5) * L * 0.99;
    }

    /* --- Benchmark WITH hash (current code) --- */
    int found_hash = 0;
    double t0 = wtime();
    for (int i = 0; i < N_LOOKUPS; i++) {
        block_t *b = mesh_find_block_at(m, xs[i], ys[i], zs[i]);
        if (b) found_hash++;
    }
    double t_hash = wtime() - t0;

    /* --- Benchmark WITHOUT hash (linear scan) --- */
    void *saved_hash = m->block_hash;
    m->block_hash = NULL;

    int found_linear = 0;
    t0 = wtime();
    for (int i = 0; i < N_LOOKUPS; i++) {
        block_t *b = mesh_find_block_at(m, xs[i], ys[i], zs[i]);
        if (b) found_linear++;
    }
    double t_linear = wtime() - t0;

    m->block_hash = saved_hash;

    /* Results */
    printf("\n%d lookups:\n", N_LOOKUPS);
    printf("  Hash:   %.4f s  (found %d)\n", t_hash, found_hash);
    printf("  Linear: %.4f s  (found %d)\n", t_linear, found_linear);
    printf("  Speedup: %.1fx\n", t_linear / t_hash);
    printf("  Results match: %s\n",
           found_hash == found_linear ? "YES" : "NO *** MISMATCH ***");

    free(xs); free(ys); free(zs);
    mesh_free(m);
    return (found_hash == found_linear) ? 0 : 1;
}
