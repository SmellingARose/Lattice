/*
 * Lattice — 3D Numerical Relativity
 * AMR mesh creation and management.
 *
 * Single root block at level 0, refined by oct-tree AMR.
 *   1. Allocate 1 root block covering the full domain
 *   2. All 6 faces are physical boundaries (no inter-block neighbors at level 0)
 *   3. AMR refinement adds child blocks via oct-tree splitting
 *
 * Ref: Athena++ src/mesh/mesh.cpp (Mesh constructor, root grid setup)
 */

#include "mesh.h"
#include "morton.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/*
 * Hash table for O(1) block lookup by (level, lx1, lx2, lx3).
 * Open addressing with linear probing. Key includes level to
 * distinguish blocks at different refinement levels with the
 * same logical coordinates.
 *
 * Persisted on mesh_t for use by mesh_find_block[_at].
 * Rebuilt in mesh_rebuild_neighbors after every regrid.
 */
typedef struct {
    uint64_t *keys;     /* hash keys (0 = empty slot) */
    int      *values;   /* block IDs */
    int       capacity; /* table size (power of 2) */
    int       mask;     /* capacity - 1 for fast modulo */
} block_hash_t;

/* Encode (level, lx1, lx2, lx3) into a non-zero 64-bit key */
static inline uint64_t block_hash_key(int level, int lx1, int lx2, int lx3)
{
    return ((uint64_t)(level + 1) << 48)
         | ((uint64_t)(lx1 & 0xFFFF) << 32)
         | ((uint64_t)(lx2 & 0xFFFF) << 16)
         | (uint64_t)(lx3 & 0xFFFF);
}

static void block_hash_init(block_hash_t *ht, int n_entries)
{
    int cap = 16;
    while (cap < 2 * n_entries) cap *= 2;
    ht->capacity = cap;
    ht->mask = cap - 1;
    ht->keys = calloc(cap, sizeof(uint64_t));
    ht->values = calloc(cap, sizeof(int));
}

static void block_hash_free(block_hash_t *ht)
{
    free(ht->keys);
    free(ht->values);
}

static void block_hash_insert(block_hash_t *ht, int level,
                                int lx1, int lx2, int lx3, int block_id)
{
    uint64_t key = block_hash_key(level, lx1, lx2, lx3);
    int slot = (int)(key & ht->mask);
    while (ht->keys[slot] != 0) {
        if (ht->keys[slot] == key) { ht->values[slot] = block_id; return; }
        slot = (slot + 1) & ht->mask;
    }
    ht->keys[slot] = key;
    ht->values[slot] = block_id;
}

/* Returns block ID or -1 if not found */
static int block_hash_find(const block_hash_t *ht, int level,
                             int lx1, int lx2, int lx3)
{
    uint64_t key = block_hash_key(level, lx1, lx2, lx3);
    int slot = (int)(key & ht->mask);
    while (ht->keys[slot] != 0) {
        if (ht->keys[slot] == key) return ht->values[slot];
        slot = (slot + 1) & ht->mask;
    }
    return -1;
}

mesh_t *mesh_create_ex(int N_block, double L, rk_method_t method,
                       int n_fields)
{
    mesh_t *m = calloc(1, sizeof(mesh_t));
    if (!m) {
        fprintf(stderr, "mesh_create: calloc failed\n");
        exit(1);
    }

    m->N_block   = N_block;
    m->L         = L;
    m->rk_method = method;
    m->n_fields  = n_fields;
    m->dx_base   = L / (double)N_block;
    m->max_level = 0;
    m->packs_dirty = 1;  /* force pack creation on first use */

    m->num_blocks = 1;
    m->max_blocks = 2;  /* headroom for future refinement */
    m->blocks = calloc(m->max_blocks, sizeof(block_t *));
    if (!m->blocks) {
        fprintf(stderr, "mesh_create: calloc blocks failed\n");
        exit(1);
    }

    printf("[AMR] Creating mesh: N_block=%d, L=%.1f, dx=%.6f\n",
           N_block, L, m->dx_base);

    /* Single root block covering full domain */
    double origin[3] = { -L / 2.0, -L / 2.0, -L / 2.0 };
    m->blocks[0] = block_alloc(0, 0, N_block, m->dx_base,
                                origin, method, n_fields);
    m->blocks[0]->loc.lx1   = 0;
    m->blocks[0]->loc.lx2   = 0;
    m->blocks[0]->loc.lx3   = 0;
    m->blocks[0]->loc.level = 0;

    /* No neighbors at level 0 — all 26 directions are physical boundary */
    for (int n = 0; n < NUM_NEIGHBORS; n++)
        m->blocks[0]->neighbor_ids[n] = -1;

    /* nblevel: self = 0, all neighbors = -1 (boundary) */
    for (int oz = -1; oz <= 1; oz++)
        for (int oy = -1; oy <= 1; oy++)
            for (int ox = -1; ox <= 1; ox++) {
                if (ox == 0 && oy == 0 && oz == 0)
                    m->blocks[0]->nblevel[oz+1][oy+1][ox+1] = 0;
                else
                    m->blocks[0]->nblevel[oz+1][oy+1][ox+1] = -1;
            }

    /* All 6 faces on physical boundary */
    for (int f = 0; f < 6; f++)
        m->blocks[0]->on_boundary[f] = 1;

    /* Pre-allocate scratch buffer for ghost_fill_from_coarser temporal
     * interpolation. Size = n_fields * block_npoints (one block's worth).
     * Eliminates per-call malloc/free (~50+ per ghost exchange). */
    {
        int Nt_block = N_block + 2 * GHOST_WIDTH;
        size_t block_npts = (size_t)Nt_block * Nt_block * Nt_block;
        m->ghost_scratch_size = (size_t)n_fields * block_npts;
        m->ghost_scratch = malloc(m->ghost_scratch_size * sizeof(double));
        if (!m->ghost_scratch) {
            fprintf(stderr, "mesh_create: ghost_scratch malloc failed\n");
            exit(1);
        }
    }

    printf("[AMR] Mesh created: 1 root block, N=%d, dx=%.6f\n",
           N_block, m->dx_base);

    return m;
}

void mesh_free(mesh_t *m)
{
    if (!m) return;

    /* Free cached persistent packs */
    if (m->leaf_pack) {
        meshblock_pack_free(m->leaf_pack);
        m->leaf_pack = NULL;
    }
    for (int L = 0; L < MAX_AMR_LEVELS; L++) {
        if (m->level_packs[L]) {
            meshblock_pack_free(m->level_packs[L]);
            m->level_packs[L] = NULL;
        }
    }

    for (int i = 0; i < m->num_blocks; i++) {
        if (m->blocks[i])
            block_free(m->blocks[i]);
    }
    free(m->blocks);
    free(m->ghost_scratch);
    if (m->block_hash) {
        block_hash_free((block_hash_t *)m->block_hash);
        free(m->block_hash);
    }
    free(m);
}

int mesh_num_leaves(const mesh_t *m)
{
    int count = 0;
    for (int i = 0; i < m->num_blocks; i++) {
        if (m->blocks[i] && m->blocks[i]->is_leaf)
            count++;
    }
    return count;
}

int mesh_check_finite(const mesh_t *m)
{
    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *b = m->blocks[bid];
        if (!b || !b->is_leaf) continue;
        grid_t *g = b->grid;
        int lo = g->ghost, hi = g->ghost + g->N;
        for (int f = 0; f < g->n_fields; f++)
            for (int k = lo; k < hi; k++)
                for (int j = lo; j < hi; j++)
                    for (int i = lo; i < hi; i++)
                        if (!isfinite(g->fields[f][IDX(g, i, j, k)]))
                            return 0;
    }
    return 1;
}

block_t *mesh_find_block(const mesh_t *m, int level, int lx1, int lx2, int lx3)
{
    /* O(1) hash lookup if hash table is built */
    if (m->block_hash) {
        int id = block_hash_find((const block_hash_t *)m->block_hash,
                                 level, lx1, lx2, lx3);
        return (id >= 0) ? m->blocks[id] : NULL;
    }

    /* Fallback: linear scan (before first mesh_rebuild_neighbors call) */
    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (!b) continue;
        if (b->loc.level == level &&
            b->loc.lx1 == lx1 &&
            b->loc.lx2 == lx2 &&
            b->loc.lx3 == lx3)
            return b;
    }
    return NULL;
}

block_t *mesh_find_block_at(const mesh_t *m, double x, double y, double z)
{
    /* O(max_level) hash-based lookup: convert physical (x,y,z) to logical
     * block coordinates at each level, starting from finest. First leaf
     * block found is the answer (finest level wins).
     *
     * Domain: [-L/2, +L/2] in each dimension.
     * At level l: block_size = L / 2^l, blocks_per_side = 2^l.
     * Logical index: lx = floor((x + L/2) / block_size). */
    if (m->block_hash) {
        const block_hash_t *ht = (const block_hash_t *)m->block_hash;
        double half_L = m->L * 0.5;
        double px = x + half_L;
        double py = y + half_L;
        double pz = z + half_L;

        /* Check out of domain */
        if (px < 0.0 || px >= m->L ||
            py < 0.0 || py >= m->L ||
            pz < 0.0 || pz >= m->L)
            return NULL;

        /* Search from finest to coarsest — first leaf hit is the answer */
        for (int level = m->max_level; level >= 0; level--) {
            double block_size = m->L / (double)(1 << level);
            int lx = (int)(px / block_size);
            int ly = (int)(py / block_size);
            int lz = (int)(pz / block_size);

            /* Clamp to valid range (handles edge case at x = +L/2) */
            int bps = 1 << level;
            if (lx >= bps) lx = bps - 1;
            if (ly >= bps) ly = bps - 1;
            if (lz >= bps) lz = bps - 1;

            int id = block_hash_find(ht, level, lx, ly, lz);
            if (id >= 0 && m->blocks[id] && m->blocks[id]->is_leaf)
                return m->blocks[id];
        }
        return NULL;
    }

    /* Fallback: linear scan (before first mesh_rebuild_neighbors call) */
    block_t *best = NULL;
    int best_level = -1;

    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (!b || !b->is_leaf) continue;

        double dx = b->grid->dx;
        int N = b->grid->N;

        int inside = 1;
        for (int d = 0; d < 3; d++) {
            double lo = b->origin[d];
            double hi = b->origin[d] + N * dx;
            double coord = (d == 0) ? x : (d == 1) ? y : z;
            if (coord < lo || coord >= hi) { inside = 0; break; }
        }

        if (inside && b->loc.level > best_level) {
            best = b;
            best_level = b->loc.level;
        }
    }

    return best;
}

int mesh_add_block(mesh_t *m, block_t *b)
{
    /* Try to find an empty (NULL) slot first */
    for (int i = 0; i < m->num_blocks; i++) {
        if (!m->blocks[i]) {
            b->id = i;
            m->blocks[i] = b;
            return i;
        }
    }

    /* No empty slot — append */
    if (m->num_blocks >= m->max_blocks) {
        m->max_blocks *= 2;
        m->blocks = realloc(m->blocks, m->max_blocks * sizeof(block_t *));
        if (!m->blocks) {
            fprintf(stderr, "mesh_add_block: realloc failed\n");
            exit(1);
        }
        /* Zero new slots */
        for (int i = m->num_blocks; i < m->max_blocks; i++)
            m->blocks[i] = NULL;
    }

    int slot = m->num_blocks;
    b->id = slot;
    m->blocks[slot] = b;
    m->num_blocks++;
    return slot;
}

void mesh_remove_block(mesh_t *m, int id)
{
    if (id < 0 || id >= m->num_blocks) return;
    m->blocks[id] = NULL;  /* caller frees */
}

void mesh_compact(mesh_t *m)
{
    /* Build old_id -> new_id mapping */
    int *id_map = calloc(m->num_blocks, sizeof(int));
    if (!id_map) {
        fprintf(stderr, "mesh_compact: calloc failed\n");
        exit(1);
    }
    for (int i = 0; i < m->num_blocks; i++)
        id_map[i] = -1;

    /* Compact: move non-NULL blocks to front */
    int write = 0;
    for (int read = 0; read < m->num_blocks; read++) {
        if (m->blocks[read]) {
            id_map[read] = write;
            if (write != read) {
                m->blocks[write] = m->blocks[read];
                m->blocks[read] = NULL;
            }
            m->blocks[write]->id = write;
            write++;
        }
    }
    m->num_blocks = write;

    /* Update all internal ID references */
    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (!b) continue;

        if (b->parent_id >= 0)
            b->parent_id = id_map[b->parent_id];

        for (int c = 0; c < 8; c++) {
            if (b->child_ids[c] >= 0)
                b->child_ids[c] = id_map[b->child_ids[c]];
        }

        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            if (b->neighbor_ids[n] >= 0)
                b->neighbor_ids[n] = id_map[b->neighbor_ids[n]];
        }
    }

    free(id_map);
}

/*
 * Helper: compute the number of blocks per side at a given level.
 * At level L, there are 2^L blocks per side (single root block).
 */
static int blocks_per_side(const mesh_t *m, int level)
{
    (void)m;
    return (1 << level);
}

void mesh_rebuild_neighbors(mesh_t *m)
{
    /* First pass: update max_level */
    m->max_level = 0;
    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (!b) continue;
        if (b->loc.level > m->max_level)
            m->max_level = b->loc.level;
    }

    /* Build/rebuild persistent hash table for O(1) block lookup.
     * Used by mesh_find_block, mesh_find_block_at, and neighbor search. */
    if (m->block_hash) {
        block_hash_free((block_hash_t *)m->block_hash);
        free(m->block_hash);
    }
    block_hash_t *ht_p = malloc(sizeof(block_hash_t));
    block_hash_init(ht_p, m->num_blocks);
    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (!b) continue;
        block_hash_insert(ht_p, b->loc.level,
                          b->loc.lx1, b->loc.lx2, b->loc.lx3, b->id);
    }
    m->block_hash = ht_p;
    block_hash_t ht = *ht_p;  /* local copy for neighbor search below */

    /* Second pass: rebuild neighbors for every block */
    for (int i = 0; i < m->num_blocks; i++) {
        block_t *b = m->blocks[i];
        if (!b) continue;

        int level = b->loc.level;
        int bps = blocks_per_side(m, level);

        /* Reset neighbor arrays */
        for (int n = 0; n < NUM_NEIGHBORS; n++)
            b->neighbor_ids[n] = -1;
        memset(b->nblevel, -1, sizeof(b->nblevel));
        b->nblevel[1][1][1] = level;

        /* Boundary flags */
        b->on_boundary[0] = (b->loc.lx1 == 0)       ? 1 : 0;
        b->on_boundary[1] = (b->loc.lx1 == bps - 1) ? 1 : 0;
        b->on_boundary[2] = (b->loc.lx2 == 0)       ? 1 : 0;
        b->on_boundary[3] = (b->loc.lx2 == bps - 1) ? 1 : 0;
        b->on_boundary[4] = (b->loc.lx3 == 0)       ? 1 : 0;
        b->on_boundary[5] = (b->loc.lx3 == bps - 1) ? 1 : 0;

        /* 26 neighbors via hash table lookup (O(1) per direction).
         * For each direction, first try same-level. If not found (refined
         * away or boundary), walk up to coarser levels by mapping the
         * NEIGHBOR'S fine-level coordinates to coarser levels via floor
         * division. This is the key difference from the naive approach
         * of dividing the block's own coords then adding the offset —
         * that fails for edge/corner directions at coarse-fine boundaries.
         *
         * Ref: Athena++ MeshBlockTree::FindNeighbor (tree walk) */
        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int ox = nbr_offset[n][0];
            int oy = nbr_offset[n][1];
            int oz = nbr_offset[n][2];

            /* Neighbor's fine-level logical coordinates */
            int nx = b->loc.lx1 + ox;
            int ny = b->loc.lx2 + oy;
            int nz = b->loc.lx3 + oz;

            /* Try same-level neighbor (must be within bounds) */
            if (nx >= 0 && nx < bps &&
                ny >= 0 && ny < bps &&
                nz >= 0 && nz < bps) {
                int nbr_id = block_hash_find(&ht, level, nx, ny, nz);
                if (nbr_id >= 0) {
                    b->neighbor_ids[n] = nbr_id;
                    b->nblevel[oz + 1][oy + 1][ox + 1] = level;
                    continue;
                }
            }

            /* Try coarser levels: map neighbor's fine-level coords to
             * coarser levels using floor division.
             * Floor div for negative x: ~(~x >> shift).
             * For non-negative x: simple right shift. */
            {
                int cur_level = level;
                int found = 0;

                while (cur_level > 0 && !found) {
                    cur_level--;
                    int cbps = blocks_per_side(m, cur_level);
                    int shift = level - cur_level;

                    /* Floor division by 2^shift (handles negative coords) */
                    int cnx = (nx >= 0) ? (nx >> shift) : ~(~nx >> shift);
                    int cny = (ny >= 0) ? (ny >> shift) : ~(~ny >> shift);
                    int cnz = (nz >= 0) ? (nz >> shift) : ~(~nz >> shift);

                    if (cnx < 0 || cnx >= cbps ||
                        cny < 0 || cny >= cbps ||
                        cnz < 0 || cnz >= cbps)
                        continue;

                    int nbr_id = block_hash_find(&ht, cur_level,
                                                  cnx, cny, cnz);
                    if (nbr_id >= 0) {
                        b->neighbor_ids[n] = nbr_id;
                        b->nblevel[oz + 1][oy + 1][ox + 1] = cur_level;
                        found = 1;
                    }
                }
            }
            /* If not found at any level, remains -1 (physical boundary) */
        }
    }

    /* Hash persists on m->block_hash for mesh_find_block[_at] */
}

