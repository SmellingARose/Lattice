/*
 * Lattice — 3D Numerical Relativity
 * Checkpoint/restart implementation.
 *
 * File format (all little-endian, 64-bit aligned):
 *
 *   [Header — 1024 bytes]
 *     uint64_t  magic          "LATC0001"
 *     uint32_t  version        1
 *     uint32_t  step           evolution step number
 *     double    time           simulation time
 *     uint32_t  num_leaves     number of leaf blocks saved
 *     uint32_t  N_block        interior cells per block side
 *     uint32_t  n_fields       active field count
 *     uint32_t  rk_method      0=classic, 1=ck45
 *     double    L              domain size
 *     sim_params_t params      full parameter struct
 *     ... padding to 1024 bytes ...
 *
 *   [Per leaf block — repeated num_leaves times]
 *     int32_t   level
 *     int32_t   lx1, lx2, lx3  logical location
 *     double    origin[3]       physical corner
 *     double    field_data[n_fields * npoints]  all field arrays
 */

#include "checkpoint.h"
#include "../amr/block.h"
#include "../amr/mesh.h"
#include "../amr/refine.h"
#include "../amr/ghost_exchange.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Magic: 8 bytes "LATCKPT\0" */
static const char CHECKPOINT_MAGIC[8] = "LATCKPT";
#define CHECKPOINT_VERSION 1
#define HEADER_SIZE 1024

/* Header layout (packed into HEADER_SIZE bytes) */
typedef struct {
    char          magic[8];
    int           version;
    int           step;
    double        time;
    int           num_leaves;
    int           N_block;
    int           n_fields;
    int           rk_method;
    double        L;
    sim_params_t  params;
    /* Implicit padding to HEADER_SIZE */
} checkpoint_header_t;

/* Per-block metadata (written before field data) */
typedef struct {
    int    level;
    int    lx1, lx2, lx3;
    double origin[3];
} block_meta_t;

int checkpoint_write(const mesh_t *m, const sim_params_t *p,
                     int step, const char *path)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "checkpoint_write: cannot open %s\n", path);
        return -1;
    }

    /* Count leaves */
    int num_leaves = mesh_num_leaves(m);

    /* Build header */
    char header_buf[HEADER_SIZE];
    memset(header_buf, 0, HEADER_SIZE);

    checkpoint_header_t *hdr = (checkpoint_header_t *)header_buf;
    memcpy(hdr->magic, CHECKPOINT_MAGIC, 8);
    hdr->version    = CHECKPOINT_VERSION;
    hdr->step       = step;
    hdr->time       = p->time;
    hdr->num_leaves = num_leaves;
    hdr->N_block    = m->N_block;
    hdr->n_fields   = m->n_fields;
    hdr->rk_method  = (int)m->rk_method;
    hdr->L          = m->L;
    hdr->params     = *p;

    /* Verify header fits */
    _Static_assert(sizeof(checkpoint_header_t) <= HEADER_SIZE,
                   "checkpoint header exceeds 1024 bytes");

    if (fwrite(header_buf, 1, HEADER_SIZE, fp) != HEADER_SIZE) {
        fprintf(stderr, "checkpoint_write: header write failed\n");
        fclose(fp);
        return -1;
    }

    /* Write each leaf block: metadata + field data */
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;

        grid_t *g = b->grid;

        /* Block metadata */
        block_meta_t meta;
        meta.level = b->loc.level;
        meta.lx1   = b->loc.lx1;
        meta.lx2   = b->loc.lx2;
        meta.lx3   = b->loc.lx3;
        meta.origin[0] = b->origin[0];
        meta.origin[1] = b->origin[1];
        meta.origin[2] = b->origin[2];

        if (fwrite(&meta, sizeof(block_meta_t), 1, fp) != 1) {
            fprintf(stderr, "checkpoint_write: meta write failed\n");
            fclose(fp);
            return -1;
        }

        /* Field data: write all active fields contiguously */
        for (int f = 0; f < m->n_fields; f++) {
            if (fwrite(g->fields[f], sizeof(double), g->npoints, fp)
                != g->npoints) {
                fprintf(stderr, "checkpoint_write: field %d write failed\n", f);
                fclose(fp);
                return -1;
            }
        }
    }

    fclose(fp);
    printf("  [Checkpoint] Saved step %d (t=%.4f) to %s"
           " (%d leaves, %.1f MB)\n",
           step, p->time, path, num_leaves,
           (double)(HEADER_SIZE + num_leaves *
            (sizeof(block_meta_t) +
             m->n_fields * m->blocks[0]->grid->npoints * sizeof(double)))
           / (1024.0 * 1024.0));

    return 0;
}

/*
 * Reconstruct AMR tree to match saved leaf locations.
 *
 * Strategy: start from root (level 0). For each saved leaf at level L > 0,
 * ensure its ancestor at each level exists by refining from root down.
 * We collect unique parent locations at each level and refine them.
 */
static int reconstruct_tree(mesh_t *m, const block_meta_t *leaves,
                            int num_leaves, int max_level)
{
    /* Refine level by level from 0 to max_level-1 */
    for (int lev = 0; lev < max_level; lev++) {
        /* Find which blocks at this level need to be refined:
         * any block that is an ancestor of a saved leaf at level > lev */
        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *b = m->blocks[bid];
            if (!b || !b->is_leaf || b->loc.level != lev) continue;

            /* Check if any saved leaf is a descendant of this block */
            int needs_refine = 0;
            for (int li = 0; li < num_leaves; li++) {
                if (leaves[li].level <= lev) continue;

                /* Check ancestry: at each level above, the logical coords
                 * must match when shifted down */
                int ancestor_lx1 = leaves[li].lx1 >> (leaves[li].level - lev);
                int ancestor_lx2 = leaves[li].lx2 >> (leaves[li].level - lev);
                int ancestor_lx3 = leaves[li].lx3 >> (leaves[li].level - lev);

                if (ancestor_lx1 == b->loc.lx1 &&
                    ancestor_lx2 == b->loc.lx2 &&
                    ancestor_lx3 == b->loc.lx3) {
                    needs_refine = 1;
                    break;
                }
            }

            if (needs_refine) {
                if (mesh_refine_block(m, bid) != 0) {
                    fprintf(stderr, "checkpoint_read: refine failed at "
                            "level %d block %d\n", lev, bid);
                    return -1;
                }
            }
        }

        /* Rebuild neighbors after each level of refinement */
        mesh_rebuild_neighbors(m);
    }

    return 0;
}

int checkpoint_read(const char *path, mesh_t **m_out,
                    sim_params_t *p_out, int *step_out)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "checkpoint_read: cannot open %s\n", path);
        return -1;
    }

    /* Read header */
    char header_buf[HEADER_SIZE];
    if (fread(header_buf, 1, HEADER_SIZE, fp) != HEADER_SIZE) {
        fprintf(stderr, "checkpoint_read: header read failed\n");
        fclose(fp);
        return -1;
    }

    checkpoint_header_t *hdr = (checkpoint_header_t *)header_buf;

    /* Validate */
    if (memcmp(hdr->magic, CHECKPOINT_MAGIC, 8) != 0) {
        fprintf(stderr, "checkpoint_read: bad magic (not a Lattice checkpoint)\n");
        fclose(fp);
        return -1;
    }
    if (hdr->version != CHECKPOINT_VERSION) {
        fprintf(stderr, "checkpoint_read: version %u (expected %u)\n",
                hdr->version, CHECKPOINT_VERSION);
        fclose(fp);
        return -1;
    }

    int num_leaves = hdr->num_leaves;
    int N_block    = hdr->N_block;
    int n_fields   = hdr->n_fields;
    double L       = hdr->L;
    rk_method_t rk = (rk_method_t)hdr->rk_method;

    printf("  [Checkpoint] Reading %s\n", path);
    printf("    step=%u, t=%.4f, leaves=%d, N_block=%d, n_fields=%d, L=%.1f\n",
           hdr->step, hdr->time, num_leaves, N_block, n_fields, L);

    /* Read all block metadata first */
    block_meta_t *metas = calloc(num_leaves, sizeof(block_meta_t));
    if (!metas) {
        fprintf(stderr, "checkpoint_read: calloc metas failed\n");
        fclose(fp);
        return -1;
    }

    /* We need to read metadata and then seek back to read field data,
     * or read metadata in a first pass. Let's save file positions. */
    long *data_offsets = calloc(num_leaves, sizeof(long));
    if (!data_offsets) {
        fprintf(stderr, "checkpoint_read: calloc offsets failed\n");
        free(metas);
        fclose(fp);
        return -1;
    }

    /* Compute npoints for each block (all blocks have same N_block) */
    int Ntotal = N_block + 2 * GHOST_WIDTH;
    size_t npoints = (size_t)Ntotal * Ntotal * Ntotal;
    size_t field_bytes = n_fields * npoints * sizeof(double);

    /* Read metadata + skip field data to collect all block locations */
    for (int li = 0; li < num_leaves; li++) {
        if (fread(&metas[li], sizeof(block_meta_t), 1, fp) != 1) {
            fprintf(stderr, "checkpoint_read: meta read failed at leaf %d\n", li);
            free(metas);
            free(data_offsets);
            fclose(fp);
            return -1;
        }
        data_offsets[li] = ftell(fp);
        if (fseek(fp, (long)field_bytes, SEEK_CUR) != 0) {
            fprintf(stderr, "checkpoint_read: seek failed at leaf %d\n", li);
            free(metas);
            free(data_offsets);
            fclose(fp);
            return -1;
        }
    }

    /* Find max level */
    int max_level = 0;
    for (int li = 0; li < num_leaves; li++) {
        if (metas[li].level > max_level)
            max_level = metas[li].level;
    }

    /* Create mesh with single root block */
    mesh_t *m = mesh_create_ex(N_block, L, rk, n_fields);

    /* Reconstruct AMR tree structure */
    if (max_level > 0) {
        if (reconstruct_tree(m, metas, num_leaves, max_level) != 0) {
            fprintf(stderr, "checkpoint_read: tree reconstruction failed\n");
            free(metas);
            free(data_offsets);
            fclose(fp);
            mesh_free(m);
            return -1;
        }
    }

    /* Now populate field data for each leaf block */
    for (int li = 0; li < num_leaves; li++) {
        /* Find the matching leaf block in the mesh */
        block_t *b = mesh_find_block(m, metas[li].level,
                                     metas[li].lx1, metas[li].lx2,
                                     metas[li].lx3);
        if (!b || !b->is_leaf) {
            fprintf(stderr, "checkpoint_read: cannot find leaf at "
                    "level=%d (%d,%d,%d)\n",
                    metas[li].level, metas[li].lx1, metas[li].lx2,
                    metas[li].lx3);
            free(metas);
            free(data_offsets);
            fclose(fp);
            mesh_free(m);
            return -1;
        }

        /* Seek to field data position and read */
        if (fseek(fp, data_offsets[li], SEEK_SET) != 0) {
            fprintf(stderr, "checkpoint_read: seek to data failed\n");
            free(metas);
            free(data_offsets);
            fclose(fp);
            mesh_free(m);
            return -1;
        }

        grid_t *g = b->grid;
        for (int f = 0; f < n_fields; f++) {
            if (fread(g->fields[f], sizeof(double), g->npoints, fp)
                != g->npoints) {
                fprintf(stderr, "checkpoint_read: field %d read failed\n", f);
                free(metas);
                free(data_offsets);
                fclose(fp);
                mesh_free(m);
                return -1;
            }
        }
    }

    free(metas);
    free(data_offsets);
    fclose(fp);

    /* Rebuild neighbor tables. Ghost zone data was saved in the checkpoint,
     * so no ghost exchange needed — the saved values are exact. */
    mesh_rebuild_neighbors(m);

    /* Restore outputs */
    *m_out    = m;
    *p_out    = hdr->params;
    *step_out = hdr->step;

    printf("    Mesh restored: %d blocks (%d leaves), max_level=%d\n",
           m->num_blocks, mesh_num_leaves(m), m->max_level);

    return 0;
}
