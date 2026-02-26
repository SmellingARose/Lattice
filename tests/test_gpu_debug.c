/*
 * GPU kernel isolation test.
 * Exercises each backend kernel individually to pinpoint which one crashes.
 * Build: make BACKEND=gpu CC=gcc-13 test-gpu-debug
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "core/grid.h"
#include "core/fields.h"
#include "core/params.h"
#include "amr/mesh.h"
#include "amr/meshblock_pack.h"
#include "backend/backend.h"
#include "initial_data/puncture.h"
#include "initial_data/bowen_york.h"
#include "numerics/rk4.h"

/* Force a GPU sync point and check for errors.
 * We do this by running a trivial 1-element kernel and checking the return. */
static void gpu_sync_check(const char *label)
{
    /* Flush stdout before potential crash */
    fflush(stdout);

    /* Use a tiny target region to force sync */
    int dummy = 0;
    #pragma omp target map(tofrom: dummy)
    { dummy = 1; }

    if (dummy == 1) {
        printf("  [OK] %s\n", label);
    } else {
        printf("  [FAIL] %s — sync returned wrong value\n", label);
    }
    fflush(stdout);
}

/* Build a pack manually since mesh_build_leaf_pack is static in rk4.c */
static meshblock_pack_t *build_test_pack(mesh_t *m)
{
    /* Collect leaf block IDs */
    int n_leaves = mesh_num_leaves(m);
    int *ids = malloc(n_leaves * sizeof(int));
    int idx = 0;
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (b && b->is_leaf)
            ids[idx++] = bid;
    }

    size_t npts = m->blocks[ids[0]]->grid->npoints;

    meshblock_pack_t *pack = meshblock_pack_create(
        n_leaves, npts, ids, -1, RK_CLASSIC, m->n_fields);

    meshblock_pack_load(pack, m->blocks);
    meshblock_pack_load_meta(pack, m->blocks);
    meshblock_pack_build_neighbors(pack, m->blocks);

    free(ids);
    return pack;
}

int main(void)
{
    printf("=== GPU Kernel Debug Test ===\n");
    fflush(stdout);

    /* Basic GPU availability check */
    gpu_sync_check("GPU available");

    /* Create a minimal mesh (1 block, N=16) */
    int N = 16;
    double L = 10.0;
    sim_params_t p = default_params();
    p.N  = N;
    p.L  = L;
    p.dx = L / N;
    p.dt = 0.25 * p.dx;
    int nf = NUM_CCZ4_FIELDS;
    p.em_enabled = 0;

    printf("\nCreating mesh: N=%d, L=%.1f, nf=%d\n", N, L, nf);
    fflush(stdout);
    mesh_t *m = mesh_create_ex(1, N, L, RK_CLASSIC, nf);
    grid_t *g = m->blocks[0]->grid;

    /* Set flat spacetime initial data (simplest possible) */
    for (size_t n = 0; n < g->npoints; n++) {
        g->fields[FIELD_CHI][n]   = 1.0;
        g->fields[FIELD_H11][n]   = 1.0;
        g->fields[FIELD_H22][n]   = 1.0;
        g->fields[FIELD_H33][n]   = 1.0;
        g->fields[FIELD_LAPSE][n] = 1.0;
    }

    printf("Building leaf pack...\n");
    fflush(stdout);
    meshblock_pack_t *pack = build_test_pack(m);
    printf("  Pack: n_blocks=%d, N=%d, Ntotal=%d, npts=%zu, nf=%d\n",
           pack->n_blocks, pack->N, pack->Ntotal, pack->npts, pack->n_fields);
    fflush(stdout);

    /* ---- Test 1: Map pack to GPU ---- */
    printf("\n--- Test 1: backend_map_pack ---\n");
    fflush(stdout);
    backend_map_pack(pack, &p);
    gpu_sync_check("backend_map_pack");

    /* ---- Test 2: Zero buffer ---- */
    printf("\n--- Test 2: backend_zero_packed (ACCUM) ---\n");
    fflush(stdout);
    backend_zero_packed(pack, PACK_BUF_ACCUM);
    gpu_sync_check("backend_zero_packed(ACCUM)");

    printf("--- Test 2b: backend_zero_packed (RHS) ---\n");
    fflush(stdout);
    backend_zero_packed(pack, PACK_BUF_RHS);
    gpu_sync_check("backend_zero_packed(RHS)");

    /* ---- Test 3: Copy buffer ---- */
    printf("\n--- Test 3: backend_copy_packed (SCRATCH <- DATA) ---\n");
    fflush(stdout);
    backend_copy_packed(pack, PACK_BUF_SCRATCH, PACK_BUF_DATA);
    gpu_sync_check("backend_copy_packed");

    /* ---- Test 4: Ghost exchange ---- */
    printf("\n--- Test 4: backend_ghost_exchange_packed ---\n");
    fflush(stdout);
    backend_ghost_exchange_packed(pack);
    gpu_sync_check("backend_ghost_exchange_packed");

    /* ---- Test 5a: Minimal target region reading pack data ---- */
    printf("\n--- Test 5a: Read pack data on GPU ---\n");
    fflush(stdout);
    {
        double *data = pack->data;
        size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
        double sum = 0.0;
        #pragma omp target teams distribute parallel for reduction(+:sum)
        for (size_t i = 0; i < total; i++)
            sum += data[i];
        printf("  sum = %.6e\n", sum);
    }
    gpu_sync_check("read pack data");

    /* ---- Test 5b: Write to rhs on GPU ---- */
    printf("\n--- Test 5b: Write to pack rhs on GPU ---\n");
    fflush(stdout);
    {
        double *rhs_data = pack->rhs;
        size_t total = (size_t)pack->n_fields * pack->n_blocks * pack->npts;
        #pragma omp target teams distribute parallel for
        for (size_t i = 0; i < total; i++)
            rhs_data[i] = 1.0;
    }
    gpu_sync_check("write pack rhs");

    /* ---- Test 5c: collapse(4) with direct offset (no pointer array) ---- */
    printf("\n--- Test 5c: collapse(4) with direct offset ---\n");
    fflush(stdout);
    {
        int lo = pack->ghost;
        int hi = pack->ghost + pack->N;
        int nb = pack->n_blocks;
        size_t npts = pack->npts;
        double *data = pack->data;
        int Nt = pack->Ntotal;

        double result = 0.0;
        #pragma omp target teams distribute parallel for collapse(4) reduction(+:result)
        for (int b = 0; b < nb; b++) {
            for (int k = lo; k < hi; k++) {
                for (int j = lo; j < hi; j++) {
                    for (int i = lo; i < hi; i++) {
                        /* Direct offset into pack — no pointer array */
                        size_t base = (size_t)0 * nb * npts + (size_t)b * npts;
                        int idx = k * Nt * Nt + j * Nt + i;
                        result += data[base + idx];
                    }
                }
            }
        }
        printf("  result = %.6e\n", result);
    }
    gpu_sync_check("collapse(4) direct offset");

    /* ---- Test 5d: collapse(4) with small pointer array on stack ---- */
    printf("\n--- Test 5d: collapse(4) with pointer array[4] ---\n");
    fflush(stdout);
    {
        int lo = pack->ghost;
        int hi = pack->ghost + pack->N;
        int nb = pack->n_blocks;
        size_t npts = pack->npts;
        double *data = pack->data;
        int Nt = pack->Ntotal;

        double result = 0.0;
        #pragma omp target teams distribute parallel for collapse(4) reduction(+:result)
        for (int b = 0; b < nb; b++) {
            for (int k = lo; k < hi; k++) {
                for (int j = lo; j < hi; j++) {
                    for (int i = lo; i < hi; i++) {
                        /* Small pointer array (4 ptrs = 32 bytes) */
                        const double *ptrs[4];
                        for (int f = 0; f < 4; f++) {
                            size_t base = (size_t)f * nb * npts + (size_t)b * npts;
                            ptrs[f] = data + base;
                        }
                        int idx = k * Nt * Nt + j * Nt + i;
                        result += ptrs[0][idx] + ptrs[1][idx]
                                + ptrs[2][idx] + ptrs[3][idx];
                    }
                }
            }
        }
        printf("  result = %.6e\n", result);
    }
    gpu_sync_check("collapse(4) ptr array[4]");

    /* ---- Test 5e0: Access sim_params_t *p on GPU ---- */
    printf("\n--- Test 5e0: Access sim_params_t *p on GPU ---\n");
    fflush(stdout);
    {
        double result = 0.0;
        #pragma omp target map(tofrom: result)
        {
            result = p.sigma + p.ccz4.kappa1 + p.gauge.eta;
        }
        printf("  p.sigma=%.2f, result=%.6e (expected ~1.4)\n", p.sigma, result);
    }
    gpu_sync_check("sim_params_t by value");

    /* ---- Test 5e1: Access sim_params_t *p via pointer on GPU ---- */
    printf("\n--- Test 5e1: Access mapped sim_params_t *p on GPU ---\n");
    fflush(stdout);
    {
        const sim_params_t *pp = &p;
        double result = 0.0;
        #pragma omp target enter data map(to: pp[0:1])
        #pragma omp target map(tofrom: result)
        {
            result = pp->sigma + pp->ccz4.kappa1 + pp->gauge.eta;
        }
        #pragma omp target exit data map(release: pp[0:1])
        printf("  result=%.6e (expected ~1.4)\n", result);
    }
    gpu_sync_check("sim_params_t via mapped pointer");

    /* ---- Test 5e2: Access already-mapped p inside pack kernel ---- */
    printf("\n--- Test 5e2: Access p inside collapse(4) kernel ---\n");
    fflush(stdout);
    {
        int lo = pack->ghost;
        int hi = pack->ghost + pack->N;
        int nb = pack->n_blocks;
        const sim_params_t *pp = &p; /* same address mapped by backend_map_pack */

        double result = 0.0;
        #pragma omp target teams distribute parallel for collapse(4) reduction(+:result)
        for (int b = 0; b < nb; b++) {
            for (int k = lo; k < hi; k++) {
                for (int j = lo; j < hi; j++) {
                    for (int i = lo; i < hi; i++) {
                        (void)b; (void)k; (void)j; (void)i;
                        result += pp->sigma;
                    }
                }
            }
        }
        printf("  result=%.6e (expected %.1f)\n", result, 0.3 * 16.0*16.0*16.0);
    }
    gpu_sync_check("p in collapse(4) kernel");

    /* ---- Test 5e: collapse(4) with full NUM_FIELDS pointer array ---- */
    printf("\n--- Test 5e: collapse(4) with pointer array[NUM_FIELDS=%d] ---\n",
           NUM_FIELDS);
    fflush(stdout);
    {
        int lo = pack->ghost;
        int hi = pack->ghost + pack->N;
        int nb = pack->n_blocks;
        size_t npts = pack->npts;
        double *data = pack->data;
        double *rhs_data = pack->rhs;
        double *dx_arr = pack->dx_per_block;
        int nf = pack->n_fields;
        int Nt = pack->Ntotal;

        double result = 0.0;
        #pragma omp target teams distribute parallel for collapse(4) reduction(+:result)
        for (int b = 0; b < nb; b++) {
            for (int k = lo; k < hi; k++) {
                for (int j = lo; j < hi; j++) {
                    for (int i = lo; i < hi; i++) {
                        double *rhs_ptrs[NUM_FIELDS];
                        const double *src_ptrs[NUM_FIELDS];
                        for (int f = 0; f < nf; f++) {
                            size_t base = (size_t)f * nb * npts + (size_t)b * npts;
                            src_ptrs[f] = data + base;
                            rhs_ptrs[f] = rhs_data + base;
                        }
                        int idx = k * Nt * Nt + j * Nt + i;
                        result += src_ptrs[0][idx];
                        rhs_ptrs[0][idx] = dx_arr[b];
                    }
                }
            }
        }
        printf("  result = %.6e\n", result);
    }
    gpu_sync_check("collapse(4) ptr array[NUM_FIELDS]");

    /* ---- Test 5d: Build grid_t on GPU and access fields ---- */
    printf("\n--- Test 5d: Build grid_t template + access on GPU ---\n");
    fflush(stdout);
    {
        int lo = pack->ghost;
        int hi = pack->ghost + pack->N;
        int nb = pack->n_blocks;
        size_t npts = pack->npts;
        double *data = pack->data;
        double *dx_arr = pack->dx_per_block;
        int nf = pack->n_fields;

        grid_t g_template;
        memset(&g_template, 0, sizeof(grid_t));
        g_template.N        = pack->N;
        g_template.ghost    = pack->ghost;
        g_template.Ntotal   = pack->Ntotal;
        g_template.npoints  = npts;
        g_template.n_fields = nf;

        double result = 0.0;
        #pragma omp target teams distribute parallel for collapse(4) reduction(+:result)
        for (int b = 0; b < nb; b++) {
            for (int k = lo; k < hi; k++) {
                for (int j = lo; j < hi; j++) {
                    for (int i = lo; i < hi; i++) {
                        const double *src_ptrs[NUM_FIELDS];
                        for (int f = 0; f < nf; f++) {
                            size_t base = (size_t)f * nb * npts + (size_t)b * npts;
                            src_ptrs[f] = data + base;
                        }
                        grid_t g_local = g_template;
                        g_local.dx = dx_arr[b];

                        /* Use grid_t macros to compute index */
                        int idx = k * g_local.Ntotal * g_local.Ntotal
                                + j * g_local.Ntotal + i;
                        int sx = 1;
                        int sy = g_local.Ntotal;

                        /* Read a stencil-like pattern (3 points) */
                        result += src_ptrs[FIELD_CHI][idx];
                        result += src_ptrs[FIELD_CHI][idx + sx];
                        result += src_ptrs[FIELD_CHI][idx + sy];
                    }
                }
            }
        }
        printf("  result = %.6e\n", result);
    }
    gpu_sync_check("grid_t template on GPU");

    /* ---- Test 5e: Call ccz4_rhs_point on GPU ---- */
    printf("\n--- Test 5e: Call ccz4_rhs_point on GPU ---\n");
    fflush(stdout);
    backend_compute_rhs_packed(pack, &p);
    gpu_sync_check("backend_compute_rhs_packed");

    /* ---- Test 6: Sommerfeld BCs ---- */
    printf("\n--- Test 6: backend_sommerfeld_packed ---\n");
    fflush(stdout);
    backend_sommerfeld_packed(pack, &p);
    gpu_sync_check("backend_sommerfeld_packed");

    /* ---- Test 7: Enforce algebraic ---- */
    printf("\n--- Test 7: backend_enforce_algebraic_packed ---\n");
    fflush(stdout);
    backend_enforce_algebraic_packed(pack);
    gpu_sync_check("backend_enforce_algebraic_packed");

    /* ---- Unmap ---- */
    printf("\n--- Unmap ---\n");
    fflush(stdout);
    backend_unmap_pack(pack);
    gpu_sync_check("backend_unmap_pack");

    /* Cleanup */
    meshblock_pack_free(pack);
    mesh_free(m);

    printf("\n=== All GPU kernel tests passed ===\n");
    return 0;
}
