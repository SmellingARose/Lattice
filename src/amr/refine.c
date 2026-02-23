/*
 * Lattice — 3D Numerical Relativity
 * AMR block refinement and coarsening.
 *
 * Split: parent → 8 children with prolongated data.
 * Merge: 8 children → parent with restricted data.
 * 2:1 constraint: no block differs by more than 1 level from neighbors.
 * Regrid: criterion + enforce + split/merge + rebuild neighbors.
 *
 * Ref: Athena++ src/mesh/mesh.cpp AdaptiveMeshRefinement()
 * Ref: Athena++ src/mesh/meshblock_tree.cpp (CreateNewNode, RemoveNode)
 */

#include "refine.h"
#include "prolongation.h"
#include "restriction.h"
#include "ghost_exchange.h"
#include "../core/fields.h"
#include "../geometry/tensor_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Prolongate data from a parent block into one child block.
 * The child covers one octant of the parent's domain: octant (cx,cy,cz) in {0,1}^3.
 * We create a temporary full-resolution fine grid covering the parent's entire
 * domain, prolongate into it, then copy the octant's data into the child.
 *
 * This reuses the existing prolongate_field() which expects coarse and fine grids
 * covering the same physical domain with N_fine = 2 * N_coarse.
 */
static void prolongate_into_child(const block_t *parent, block_t *child,
                                  int cx, int cy, int cz)
{
    const grid_t *pg = parent->grid;
    const int N = pg->N;
    const int ghost = pg->ghost;

    /* Create temporary fine grid covering parent's entire domain */
    grid_t *fine_tmp = grid_alloc(2 * N, pg->L, RK_CK45);

    /* Prolongate all fields from parent into fine_tmp */
    prolongate_all(pg, fine_tmp);

    /* Copy the relevant octant from fine_tmp into child's grid.
     * Child octant (cx,cy,cz) covers fine cells [cx*N, (cx+1)*N) in each direction,
     * offset by ghost zones. */
    grid_t *cg = child->grid;
    int fg = fine_tmp->ghost;

    for (int f = 0; f < NUM_FIELDS; f++) {
        /* Copy interior */
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                    int fi = fg + cx * N + i;
                    int fj = fg + cy * N + j;
                    int fk = fg + cz * N + k;
                    int ci = ghost + i;
                    int cj = ghost + j;
                    int ck = ghost + k;
                    cg->fields[f][IDX(cg, ci, cj, ck)] =
                        fine_tmp->fields[f][IDX(fine_tmp, fi, fj, fk)];
                }
            }
        }
    }

    grid_free(fine_tmp);
}

/*
 * Enforce algebraic constraints on a single block (det(gambar)=1, tr(Abar)=0).
 * Duplicate of the logic in rk4.c enforce_algebraic, applied to one block.
 */
static void enforce_algebraic_block(grid_t *g)
{
    for (int k = 0; k < g->Ntotal; k++) {
        for (int j = 0; j < g->Ntotal; j++) {
            for (int i = 0; i < g->Ntotal; i++) {
                int idx = IDX(g, i, j, k);

                double h_loc[3][3];
                h_loc[0][0] = g->fields[FIELD_H11][idx];
                h_loc[0][1] = g->fields[FIELD_H12][idx];
                h_loc[0][2] = g->fields[FIELD_H13][idx];
                h_loc[1][0] = h_loc[0][1];
                h_loc[1][1] = g->fields[FIELD_H22][idx];
                h_loc[1][2] = g->fields[FIELD_H23][idx];
                h_loc[2][0] = h_loc[0][2];
                h_loc[2][1] = h_loc[1][2];
                h_loc[2][2] = g->fields[FIELD_H33][idx];

                double det = compute_det_sym(h_loc);
                if (det > 0.0) {
                    double scale = 1.0 / cbrt(det);
                    FOR2(a, b) h_loc[a][b] *= scale;

                    g->fields[FIELD_H11][idx] = h_loc[0][0];
                    g->fields[FIELD_H12][idx] = h_loc[0][1];
                    g->fields[FIELD_H13][idx] = h_loc[0][2];
                    g->fields[FIELD_H22][idx] = h_loc[1][1];
                    g->fields[FIELD_H23][idx] = h_loc[1][2];
                    g->fields[FIELD_H33][idx] = h_loc[2][2];
                }

                double h_UU[3][3];
                compute_inverse_sym(h_loc, h_UU);

                double A_loc[3][3];
                A_loc[0][0] = g->fields[FIELD_A11][idx];
                A_loc[0][1] = g->fields[FIELD_A12][idx];
                A_loc[0][2] = g->fields[FIELD_A13][idx];
                A_loc[1][0] = A_loc[0][1];
                A_loc[1][1] = g->fields[FIELD_A22][idx];
                A_loc[1][2] = g->fields[FIELD_A23][idx];
                A_loc[2][0] = A_loc[0][2];
                A_loc[2][1] = A_loc[1][2];
                A_loc[2][2] = g->fields[FIELD_A33][idx];

                make_trace_free(A_loc, h_loc, h_UU);

                g->fields[FIELD_A11][idx] = A_loc[0][0];
                g->fields[FIELD_A12][idx] = A_loc[0][1];
                g->fields[FIELD_A13][idx] = A_loc[0][2];
                g->fields[FIELD_A22][idx] = A_loc[1][1];
                g->fields[FIELD_A23][idx] = A_loc[1][2];
                g->fields[FIELD_A33][idx] = A_loc[2][2];

                if (g->fields[FIELD_CHI][idx] < 1.0e-12)
                    g->fields[FIELD_CHI][idx] = 1.0e-12;
                if (g->fields[FIELD_LAPSE][idx] < 1.0e-12)
                    g->fields[FIELD_LAPSE][idx] = 1.0e-12;
            }
        }
    }
}

int mesh_refine_block(mesh_t *m, int block_id)
{
    block_t *parent = m->blocks[block_id];
    if (!parent || !parent->is_leaf) {
        fprintf(stderr, "mesh_refine_block: block %d not a valid leaf\n", block_id);
        return -1;
    }

    int level = parent->loc.level;
    int N = m->N_block;
    double child_dx = parent->grid->dx / 2.0;

    /* Create 8 children: octant = cx + (cy<<1) + (cz<<2) */
    for (int cz = 0; cz < 2; cz++) {
        for (int cy = 0; cy < 2; cy++) {
            for (int cx = 0; cx < 2; cx++) {
                int octant = cx + (cy << 1) + (cz << 2);

                /* Child logical coordinates */
                int lx1 = 2 * parent->loc.lx1 + cx;
                int lx2 = 2 * parent->loc.lx2 + cy;
                int lx3 = 2 * parent->loc.lx3 + cz;

                /* Child origin */
                double origin[3] = {
                    parent->origin[0] + cx * N * child_dx,
                    parent->origin[1] + cy * N * child_dx,
                    parent->origin[2] + cz * N * child_dx
                };

                block_t *child = block_alloc(0, level + 1, N, child_dx,
                                             origin, m->rk_method);
                child->loc.lx1 = lx1;
                child->loc.lx2 = lx2;
                child->loc.lx3 = lx3;
                child->loc.level = level + 1;
                child->parent_id = parent->id;

                /* Add to mesh (assigns child->id) */
                int child_id = mesh_add_block(m, child);

                /* Prolongate parent data into child */
                prolongate_into_child(parent, child, cx, cy, cz);

                /* Enforce algebraic constraints on child */
                enforce_algebraic_block(child->grid);

                parent->child_ids[octant] = child_id;
            }
        }
    }

    /* Mark parent as non-leaf (retains grid data for restriction) */
    parent->is_leaf = 0;

    /* Update max_level */
    if (level + 1 > m->max_level)
        m->max_level = level + 1;

    return 0;
}

/*
 * Restrict one child's interior into the overlapping region of the parent.
 * Child at octant (cx,cy,cz) covers parent interior cells [cx*N/2, (cx+1)*N/2).
 *
 * 6th-order cell-average Lagrange restriction (restrict_w[] weights).
 * Stencil [base-2, base+3] always fits within [0, child_Nt) since
 * fi_base ∈ [ghost, ghost+N-2] and ghost=4 gives margin of 2 on each side.
 *
 * Ref: ExaHyPE (arXiv:2504.15814) — matching restriction to prolongation order
 */
static void restrict_child_into_parent(const block_t *child, block_t *parent,
                                       int cx, int cy, int cz)
{
    const grid_t *cg = child->grid;
    grid_t *pg = parent->grid;
    const int ghost = pg->ghost;
    const int N = pg->N;
    const int half_N = N / 2;
    const int child_ghost = cg->ghost;

    int p_off_i = cx * half_N;
    int p_off_j = cy * half_N;
    int p_off_k = cz * half_N;

    for (int f = 0; f < NUM_FIELDS; f++) {
        const double *src = cg->fields[f];

        for (int pk = 0; pk < half_N; pk++) {
            for (int pj = 0; pj < half_N; pj++) {
                for (int pi = 0; pi < half_N; pi++) {
                    /* Fine base index: first direct child cell */
                    int fi_base = child_ghost + 2 * pi;
                    int fj_base = child_ghost + 2 * pj;
                    int fk_base = child_ghost + 2 * pk;

                    /* 6th-order: 3D tensor product of 6-point stencil */
                    double val = 0.0;
                    for (int sk = 0; sk < RESTRICT_STENCIL; sk++) {
                        int fk = fk_base - 2 + sk;
                        for (int sj = 0; sj < RESTRICT_STENCIL; sj++) {
                            double wkj = restrict_w[sk] * restrict_w[sj];
                            int fj = fj_base - 2 + sj;
                            for (int si = 0; si < RESTRICT_STENCIL; si++) {
                                int fi = fi_base - 2 + si;
                                val += wkj * restrict_w[si] *
                                       src[IDX(cg, fi, fj, fk)];
                            }
                        }
                    }

                    int pii = ghost + p_off_i + pi;
                    int pjj = ghost + p_off_j + pj;
                    int pkk = ghost + p_off_k + pk;
                    pg->fields[f][IDX(pg, pii, pjj, pkk)] = val;
                }
            }
        }
    }
}

int mesh_coarsen_siblings(mesh_t *m, int parent_id)
{
    block_t *parent = m->blocks[parent_id];
    if (!parent || parent->is_leaf) {
        fprintf(stderr, "mesh_coarsen_siblings: block %d not a valid non-leaf\n",
                parent_id);
        return -1;
    }

    /* Verify all 8 children exist and are leaves */
    for (int c = 0; c < 8; c++) {
        int cid = parent->child_ids[c];
        if (cid < 0 || !m->blocks[cid] || !m->blocks[cid]->is_leaf) {
            fprintf(stderr, "mesh_coarsen_siblings: child %d of block %d "
                    "not a valid leaf\n", c, parent_id);
            return -1;
        }
    }

    /* Restrict each child's data into parent */
    for (int cz = 0; cz < 2; cz++) {
        for (int cy = 0; cy < 2; cy++) {
            for (int cx = 0; cx < 2; cx++) {
                int octant = cx + (cy << 1) + (cz << 2);
                int cid = parent->child_ids[octant];
                block_t *child = m->blocks[cid];
                restrict_child_into_parent(child, parent, cx, cy, cz);
            }
        }
    }

    /* Free children and remove from mesh */
    for (int c = 0; c < 8; c++) {
        int cid = parent->child_ids[c];
        block_t *child = m->blocks[cid];
        mesh_remove_block(m, cid);
        block_free(child);
        parent->child_ids[c] = -1;
    }

    /* Mark parent as leaf again */
    parent->is_leaf = 1;

    return 0;
}

void mesh_enforce_2to1(mesh_t *m, refine_flag_t *flags, int *n_flags)
{
    int changed = 1;

    /* Build lookup: block_id -> flag index for quick access */
    int max_id = 0;
    for (int i = 0; i < m->num_blocks; i++) {
        if (m->blocks[i] && m->blocks[i]->id > max_id)
            max_id = m->blocks[i]->id;
    }

    int *action_map = calloc(max_id + 1, sizeof(int));
    if (!action_map) {
        fprintf(stderr, "mesh_enforce_2to1: calloc failed\n");
        exit(1);
    }
    for (int i = 0; i < *n_flags; i++)
        action_map[flags[i].block_id] = flags[i].action;

    /* Iterate until convergence: cascade refinement flags */
    while (changed) {
        changed = 0;

        for (int i = 0; i < *n_flags; i++) {
            if (flags[i].action != AMR_REFINE) continue;

            block_t *b = m->blocks[flags[i].block_id];
            if (!b) continue;

            /* Check all 26 neighbors */
            for (int n = 0; n < NUM_NEIGHBORS; n++) {
                int nbr_id = b->neighbor_ids[n];
                if (nbr_id < 0) continue;

                block_t *nbr = m->blocks[nbr_id];
                if (!nbr || !nbr->is_leaf) continue;

                /* If neighbor is at same or lower level and not already
                 * flagged for refinement, flag it */
                if (nbr->loc.level <= b->loc.level &&
                    action_map[nbr_id] != AMR_REFINE) {
                    action_map[nbr_id] = AMR_REFINE;
                    changed = 1;

                    /* Add flag or update existing */
                    int found = 0;
                    for (int j = 0; j < *n_flags; j++) {
                        if (flags[j].block_id == nbr_id) {
                            flags[j].action = AMR_REFINE;
                            found = 1;
                            break;
                        }
                    }
                    if (!found) {
                        flags[*n_flags].block_id = nbr_id;
                        flags[*n_flags].action = AMR_REFINE;
                        (*n_flags)++;
                    }
                }
            }
        }
    }

    /* Remove coarsening flags that would violate 2:1 */
    for (int i = 0; i < *n_flags; i++) {
        if (flags[i].action != AMR_COARSEN) continue;

        block_t *b = m->blocks[flags[i].block_id];
        if (!b) continue;

        /* Check: after coarsening, would any neighbor be > 1 level above? */
        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int ox = nbr_offset[n][0];
            int oy = nbr_offset[n][1];
            int oz = nbr_offset[n][2];
            int nbr_level = b->nblevel[oz + 1][oy + 1][ox + 1];

            if (nbr_level >= 0 && nbr_level > b->loc.level) {
                /* Neighbor is finer — can't coarsen */
                flags[i].action = AMR_NONE;
                break;
            }
        }
    }

    free(action_map);
}

int mesh_regrid(mesh_t *m, const amr_params_t *ap)
{
    int initial_blocks = mesh_num_leaves(m);

    /* Allocate flags array (generous upper bound) */
    int max_flags = m->num_blocks * 2;
    refine_flag_t *flags = calloc(max_flags, sizeof(refine_flag_t));
    if (!flags) {
        fprintf(stderr, "mesh_regrid: calloc failed\n");
        exit(1);
    }

    /* Step 1: Evaluate criterion on all leaf blocks */
    int n_flags = criterion_check_mesh(m, ap, flags, max_flags);

    /* Step 2: Enforce 2:1 constraint */
    mesh_enforce_2to1(m, flags, &n_flags);

    /* Step 3: Refine flagged blocks (process in order) */
    int refine_count = 0;
    for (int i = 0; i < n_flags; i++) {
        if (flags[i].action != AMR_REFINE) continue;

        block_t *b = m->blocks[flags[i].block_id];
        if (!b || !b->is_leaf) continue;
        if (b->loc.level >= ap->max_level) continue;

        if (mesh_refine_block(m, flags[i].block_id) == 0)
            refine_count++;
    }

    /* Step 4: Coarsen flagged blocks (only if all 8 siblings agree) */
    int coarsen_count = 0;
    for (int i = 0; i < n_flags; i++) {
        if (flags[i].action != AMR_COARSEN) continue;

        block_t *b = m->blocks[flags[i].block_id];
        if (!b || !b->is_leaf) continue;
        if (b->parent_id < 0) continue;  /* can't coarsen root */

        block_t *parent = m->blocks[b->parent_id];
        if (!parent) continue;

        /* Check all 8 siblings are flagged for coarsening */
        int all_coarsen = 1;
        for (int c = 0; c < 8; c++) {
            int cid = parent->child_ids[c];
            if (cid < 0 || !m->blocks[cid] || !m->blocks[cid]->is_leaf) {
                all_coarsen = 0;
                break;
            }
            /* Check if this sibling is flagged for coarsening */
            int flagged = 0;
            for (int j = 0; j < n_flags; j++) {
                if (flags[j].block_id == cid && flags[j].action == AMR_COARSEN) {
                    flagged = 1;
                    break;
                }
            }
            if (!flagged) {
                all_coarsen = 0;
                break;
            }
        }

        if (all_coarsen) {
            if (mesh_coarsen_siblings(m, b->parent_id) == 0)
                coarsen_count++;
        }
    }

    free(flags);

    /* Step 5: Compact and rebuild if anything changed */
    if (refine_count > 0 || coarsen_count > 0) {
        mesh_compact(m);
        mesh_rebuild_neighbors(m);

        /* Step 6: Fill ghost zones of newly created blocks.
         * Prolongation only fills interior cells; ghost zones are zero.
         * Ghost exchange fills from same-level neighbors. Then extrapolate
         * boundary ghost cells to prevent chi=0 → NaN in the CCZ4 RHS.
         * Ref: Athena++ AMR post-regrid boundary fill. */
        if (m->max_level > 0)
            ghost_exchange_multilevel(m);
        else
            ghost_exchange(m);

        /* Extrapolate boundary ghost cells from interior data.
         * Domain-boundary ghosts have no neighbor to exchange with, so they
         * remain zero after ghost exchange. Copy nearest interior plane
         * into all ghost layers to prevent chi=0 blowup. */
        for (int bid = 0; bid < m->num_blocks; bid++) {
            block_t *b = m->blocks[bid];
            if (!b || !b->is_leaf) continue;
            grid_t *g = b->grid;
            int gw = g->ghost;
            int N = g->N;

            for (int face = 0; face < 6; face++) {
                if (!b->on_boundary[face]) continue;

                /* Fill boundary ghost cells by copying nearest interior plane */
                for (int f = 0; f < NUM_FIELDS; f++) {
                    for (int k = 0; k < g->Ntotal; k++) {
                        for (int j = 0; j < g->Ntotal; j++) {
                            for (int i = 0; i < g->Ntotal; i++) {
                                int target = -1;
                                switch (face) {
                                case 0: /* x- */
                                    if (i < gw) target = IDX(g, gw, j, k);
                                    break;
                                case 1: /* x+ */
                                    if (i >= gw + N) target = IDX(g, gw + N - 1, j, k);
                                    break;
                                case 2: /* y- */
                                    if (j < gw) target = IDX(g, i, gw, k);
                                    break;
                                case 3: /* y+ */
                                    if (j >= gw + N) target = IDX(g, i, gw + N - 1, k);
                                    break;
                                case 4: /* z- */
                                    if (k < gw) target = IDX(g, i, j, gw);
                                    break;
                                case 5: /* z+ */
                                    if (k >= gw + N) target = IDX(g, i, j, gw + N - 1);
                                    break;
                                }
                                if (target >= 0)
                                    g->fields[f][IDX(g, i, j, k)] = g->fields[f][target];
                            }
                        }
                    }
                }
            }
        }

        printf("[AMR] Regrid: +%d refined, -%d coarsened, %d total leaves "
               "(was %d)\n", refine_count, coarsen_count,
               mesh_num_leaves(m), initial_blocks);
    }

    return mesh_num_leaves(m) - initial_blocks;
}
