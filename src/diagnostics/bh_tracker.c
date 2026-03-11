/*
 * Lattice -- 3D Numerical Relativity
 * Multi-BH tracker: position tracking, AH diagnostics, merger detection.
 *
 * Tracks N black holes via successive lapse-minimum searches with exclusion
 * zones (GRChombo PunctureTracker pattern). Each BH gets an independent AH
 * finder for mass/spin extraction. Mergers detected by proximity threshold.
 *
 * Ref: arXiv:2505.01495 (GRChombo 25-BH cluster simulation)
 * Ref: GRChombo Source/Diagnostics/PunctureTracker
 */

#include "bh_tracker.h"
#include "../amr/meshblock_pack.h"
#include "../backend/backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * Allocation / Free
 * ================================================================ */

bh_tracker_t *bh_tracker_alloc(int n_bh, const puncture_data_t bhs[],
                                int ah_n_theta, int ah_n_phi)
{
    if (n_bh <= 0 || n_bh > MAX_PUNCTURES) return NULL;

    bh_tracker_t *tr = calloc(1, sizeof(bh_tracker_t));
    if (!tr) return NULL;

    tr->n_bh = n_bh;
    tr->n_bh_initial = n_bh;
    tr->n_active = n_bh;
    tr->ah_n_theta = ah_n_theta;
    tr->ah_n_phi = ah_n_phi;
    tr->n_mergers = 0;

    for (int i = 0; i < n_bh; i++) {
        bh_state_t *b = &tr->bh[i];
        b->id = i;
        b->status = BH_STATUS_ACTIVE;
        b->center[0] = bhs[i].center[0];
        b->center[1] = bhs[i].center[1];
        b->center[2] = bhs[i].center[2];
        b->initial_mass = bhs[i].mass;
        b->mass_irr = 0.0;
        b->mass_chr = 0.0;
        b->chi_spin = 0.0;
        b->lapse_min = 1.0;
        b->merged_into = -1;
        b->merge_time = -1.0;

        /* AH workspace: initial radius guess = M/2 (isotropic Schwarzschild) */
        double r_guess = bhs[i].mass / 2.0;
        if (r_guess < 0.1) r_guess = 0.5;
        tr->ah[i] = ah_alloc(ah_n_theta, ah_n_phi, bhs[i].center, r_guess);
        if (tr->ah[i]) tr->ah[i]->eta = 10.0;
    }

    return tr;
}

void bh_tracker_free(bh_tracker_t *tr)
{
    if (!tr) return;
    for (int i = 0; i < tr->n_bh; i++) {
        ah_free(tr->ah[i]);
    }
    free(tr);
}

/* ================================================================
 * Position update via N successive lapse-minimum searches
 *
 * Algorithm (GRChombo PunctureTracker pattern):
 *   1. Scan mesh for global lapse minimum -> assign to nearest active BH
 *   2. Mask exclusion zone (R = 2*M around that BH)
 *   3. Repeat for remaining active BHs
 *
 * O(N * blocks) per step. Fine for N <= 32.
 * ================================================================ */

/* Find lapse minimum over a mesh, excluding spheres */
static void mesh_find_lapse_min_excl(const mesh_t *m,
                                      int n_excl, const double excl_centers[][3],
                                      const double *excl_radii,
                                      double *out_min, double out_pos[3])
{
    double best = 1e30;
    double bx = 0, by = 0, bz = 0;

    for (int bid = 0; bid < m->num_blocks; bid++) {
        block_t *blk = m->blocks[bid];
        if (!blk || !blk->is_leaf) continue;
        grid_t *g = blk->grid;

        for (int kk = g->ghost; kk < g->ghost + g->N; kk++) {
            for (int jj = g->ghost; jj < g->ghost + g->N; jj++) {
                for (int ii = g->ghost; ii < g->ghost + g->N; ii++) {
                    int idx = kk * g->Ntotal * g->Ntotal + jj * g->Ntotal + ii;
                    double a = g->fields[FIELD_LAPSE][idx];
                    if (a >= best) continue;

                    double x = blk->origin[0] + (ii - g->ghost + 0.5) * g->dx;
                    double y = blk->origin[1] + (jj - g->ghost + 0.5) * g->dx;
                    double z = blk->origin[2] + (kk - g->ghost + 0.5) * g->dx;

                    /* Check exclusion zones */
                    int excluded = 0;
                    for (int e = 0; e < n_excl; e++) {
                        double dx = x - excl_centers[e][0];
                        double dy = y - excl_centers[e][1];
                        double dz = z - excl_centers[e][2];
                        if (dx*dx + dy*dy + dz*dz < excl_radii[e] * excl_radii[e]) {
                            excluded = 1;
                            break;
                        }
                    }
                    if (excluded) continue;

                    best = a;
                    bx = x; by = y; bz = z;
                }
            }
        }
    }

    *out_min = best;
    out_pos[0] = bx; out_pos[1] = by; out_pos[2] = bz;
}

void bh_tracker_update_positions(bh_tracker_t *tr, const mesh_t *m)
{
    /* Exclusion zone storage for successive searches */
    double excl_centers[MAX_PUNCTURES][3];
    double excl_radii[MAX_PUNCTURES];
    int n_excl = 0;

    /* Track which active BHs still need assignment */
    int assigned[MAX_PUNCTURES];
    memset(assigned, 0, sizeof(assigned));

    for (int pass = 0; pass < tr->n_active; pass++) {
        double min_lapse;
        double pos[3];
        mesh_find_lapse_min_excl(m, n_excl, (const double (*)[3])excl_centers,
                                  excl_radii, &min_lapse, pos);

        if (min_lapse >= 1e30) break;  /* no more minima found */

        /* Assign to nearest unassigned active BH */
        int best_id = -1;
        double best_dist = 1e30;
        for (int i = 0; i < tr->n_bh; i++) {
            if (tr->bh[i].status != BH_STATUS_ACTIVE) continue;
            if (assigned[i]) continue;
            double dx = pos[0] - tr->bh[i].center[0];
            double dy = pos[1] - tr->bh[i].center[1];
            double dz = pos[2] - tr->bh[i].center[2];
            double d = sqrt(dx*dx + dy*dy + dz*dz);
            if (d < best_dist) {
                best_dist = d;
                best_id = i;
            }
        }

        if (best_id < 0) break;

        assigned[best_id] = 1;
        tr->bh[best_id].center[0] = pos[0];
        tr->bh[best_id].center[1] = pos[1];
        tr->bh[best_id].center[2] = pos[2];
        tr->bh[best_id].lapse_min = min_lapse;

        /* Add exclusion zone: R = 2 * M of this BH */
        excl_centers[n_excl][0] = pos[0];
        excl_centers[n_excl][1] = pos[1];
        excl_centers[n_excl][2] = pos[2];
        excl_radii[n_excl] = 2.0 * tr->bh[best_id].initial_mass;
        if (excl_radii[n_excl] < 1.0) excl_radii[n_excl] = 1.0;
        n_excl++;
    }
}

/* ================================================================
 * AH finding per active BH
 * ================================================================ */

void bh_tracker_find_horizons(bh_tracker_t *tr, const mesh_t *m,
                               double tol, int max_iter)
{
    for (int i = 0; i < tr->n_bh; i++) {
        if (tr->bh[i].status != BH_STATUS_ACTIVE) continue;
        ah_workspace_t *ws = tr->ah[i];
        if (!ws) continue;

        /* Update AH center to current BH position */
        ws->center[0] = tr->bh[i].center[0];
        ws->center[1] = tr->bh[i].center[1];
        ws->center[2] = tr->bh[i].center[2];

        int conv = ah_find_amr(ws, m, tol, max_iter, 0);
        if (conv) {
            ah_result_t res = ah_compute_diagnostics_amr(ws, m);
            tr->bh[i].mass_irr = res.mass_irr;
            tr->bh[i].mass_chr = res.mass_christodoulou;
            tr->bh[i].chi_spin = res.chi_spin;
        }
    }
}

/* ================================================================
 * Merger detection
 *
 * Pairwise distance check among active BHs: O(N^2) but N <= 32.
 * When separation < 3 * max(M_i, M_j), merge the pair.
 * Remnant placed at midpoint, assigned next available id slot.
 *
 * Ref: arXiv:2505.01495 Sec. III (25-BH cluster, merger sequence)
 * ================================================================ */

void bh_tracker_check_mergers(bh_tracker_t *tr, double time)
{
    for (int i = 0; i < tr->n_bh; i++) {
        if (tr->bh[i].status != BH_STATUS_ACTIVE) continue;

        for (int j = i + 1; j < tr->n_bh; j++) {
            if (tr->bh[j].status != BH_STATUS_ACTIVE) continue;

            double dx = tr->bh[i].center[0] - tr->bh[j].center[0];
            double dy = tr->bh[i].center[1] - tr->bh[j].center[1];
            double dz = tr->bh[i].center[2] - tr->bh[j].center[2];
            double sep = sqrt(dx*dx + dy*dy + dz*dz);

            double M_i = tr->bh[i].initial_mass;
            double M_j = tr->bh[j].initial_mass;
            double M_max = M_i > M_j ? M_i : M_j;
            double merge_threshold = 3.0 * M_max;

            if (sep < merge_threshold) {
                /* Create remnant */
                int rid = tr->n_bh;
                if (rid >= MAX_PUNCTURES) continue;  /* can't add more */

                bh_state_t *rem = &tr->bh[rid];
                rem->id = rid;
                rem->status = BH_STATUS_ACTIVE;
                rem->center[0] = 0.5 * (tr->bh[i].center[0] + tr->bh[j].center[0]);
                rem->center[1] = 0.5 * (tr->bh[i].center[1] + tr->bh[j].center[1]);
                rem->center[2] = 0.5 * (tr->bh[i].center[2] + tr->bh[j].center[2]);
                rem->initial_mass = M_i + M_j;
                rem->mass_irr = 0.0;
                rem->mass_chr = 0.0;
                rem->chi_spin = 0.0;
                rem->lapse_min = 0.5 * (tr->bh[i].lapse_min + tr->bh[j].lapse_min);
                rem->merged_into = -1;
                rem->merge_time = -1.0;

                /* Allocate AH workspace for remnant */
                double r_guess = rem->initial_mass / 2.0;
                tr->ah[rid] = ah_alloc(tr->ah_n_theta, tr->ah_n_phi,
                                        rem->center, r_guess);
                if (tr->ah[rid]) tr->ah[rid]->eta = 10.0;

                /* Mark parents as merged */
                tr->bh[i].status = BH_STATUS_MERGED;
                tr->bh[i].merged_into = rid;
                tr->bh[i].merge_time = time;

                tr->bh[j].status = BH_STATUS_MERGED;
                tr->bh[j].merged_into = rid;
                tr->bh[j].merge_time = time;

                /* Record merger event */
                if (tr->n_mergers < MAX_PUNCTURES) {
                    tr->mergers[tr->n_mergers].id1 = i;
                    tr->mergers[tr->n_mergers].id2 = j;
                    tr->mergers[tr->n_mergers].remnant_id = rid;
                    tr->mergers[tr->n_mergers].time = time;
                    tr->n_mergers++;
                }

                tr->n_bh++;
                tr->n_active--;  /* lost 2, gained 1 => net -1 */

                /* Don't check j any further — it's merged */
                break;
            }
        }
    }
}

/* ================================================================
 * CSV output
 *
 * Dynamic columns: per-BH diagnostics + global quantities.
 * Merged BHs output NaN for all fields.
 *
 * Format matches GRChombo diagnostic output pattern:
 * time-series of per-horizon quantities.
 * Ref: arXiv:2505.01495 Fig. 2 (merger tree + mass/spin evolution)
 * ================================================================ */

void bh_tracker_write_csv_header(const bh_tracker_t *tr, FILE *fp)
{
    if (!fp) return;
    fprintf(fp, "time,ham_l2,mom_l2,n_active,n_leaves");
    for (int i = 0; i < tr->n_bh_initial; i++) {
        fprintf(fp, ",bh%d_x,bh%d_y,bh%d_z,bh%d_mass,bh%d_spin,bh%d_lapse",
                i, i, i, i, i, i);
    }
    fprintf(fp, ",n_mergers\n");
}

void bh_tracker_write_csv(const bh_tracker_t *tr, FILE *fp, double time,
                           double ham_l2, double mom_l2, int n_leaves)
{
    if (!fp) return;
    fprintf(fp, "%.6f,%.6e,%.6e,%d,%d", time, ham_l2, mom_l2,
            tr->n_active, n_leaves);
    /* Write only the initial BH columns for consistent CSV width */
    for (int i = 0; i < tr->n_bh_initial; i++) {
        const bh_state_t *b = &tr->bh[i];
        if (b->status == BH_STATUS_MERGED) {
            fprintf(fp, ",nan,nan,nan,nan,nan,nan");
        } else {
            fprintf(fp, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    b->center[0], b->center[1], b->center[2],
                    b->mass_chr, b->chi_spin, b->lapse_min);
        }
    }
    fprintf(fp, ",%d\n", tr->n_mergers);
}

void bh_tracker_write_mergers(const bh_tracker_t *tr, const char *path)
{
    if (!tr || !path || tr->n_mergers == 0) return;
    FILE *fp = fopen(path, "w");
    if (!fp) return;

    fprintf(fp, "# Merger events (Lattice BH tracker)\n");
    fprintf(fp, "# time  id1  id2  remnant_id\n");
    for (int i = 0; i < tr->n_mergers; i++) {
        fprintf(fp, "%.6f  %d  %d  %d\n",
                tr->mergers[i].time,
                tr->mergers[i].id1,
                tr->mergers[i].id2,
                tr->mergers[i].remnant_id);
    }
    fclose(fp);
}
