/*
 * Lattice -- 3D Numerical Relativity
 * Multi-BH tracker: position tracking, AH diagnostics, merger detection.
 *
 * Tracks N black holes via successive lapse-minimum searches with exclusion
 * zones, runs per-BH apparent horizon finding for mass/spin extraction,
 * and detects mergers via proximity threshold.
 *
 * Ref: arXiv:2505.01495 (GRChombo 25-BH cluster simulation)
 * Ref: arXiv:2505.15912 (BHaHAHA apparent horizon finder)
 * Ref: GRChombo Source/Diagnostics/PunctureTracker (lapse-minimum tracking)
 */

#ifndef LATTICE_BH_TRACKER_H
#define LATTICE_BH_TRACKER_H

#include "../core/params.h"
#include "../diagnostics/ah_finder.h"
#include "../amr/mesh.h"
#include <stdio.h>

#define BH_STATUS_ACTIVE  0
#define BH_STATUS_MERGED  1

typedef struct {
    int    id;                  /* persistent BH label (0..N-1)             */
    int    status;              /* BH_STATUS_ACTIVE or BH_STATUS_MERGED     */
    double center[3];           /* current position (from lapse minimum)    */
    double mass_irr;            /* irreducible mass from AH (0 if not found)*/
    double mass_chr;            /* Christodoulou mass                       */
    double chi_spin;            /* dimensionless spin chi = J / M_irr^2    */
    double lapse_min;           /* deepest lapse value at this BH          */
    int    merged_into;         /* id of remnant BH (-1 if active)         */
    double merge_time;          /* time of merger (-1 if active)           */
    double initial_mass;        /* bare mass from initial data             */
} bh_state_t;

typedef struct {
    int    n_bh;                /* total BHs tracked (including merged)     */
    int    n_bh_initial;        /* original BH count (for fixed CSV cols)   */
    int    n_active;            /* currently active BHs                     */
    bh_state_t bh[MAX_PUNCTURES];
    ah_workspace_t *ah[MAX_PUNCTURES];  /* one AH workspace per BH         */
    int    ah_n_theta;          /* angular resolution for AH finder         */
    int    ah_n_phi;

    int    n_mergers;
    struct {
        int id1, id2;           /* parent BH ids                            */
        int remnant_id;         /* id of remnant BH                         */
        double time;            /* merger time                              */
    } mergers[MAX_PUNCTURES];
} bh_tracker_t;

/*
 * Allocate tracker for n_bh black holes.
 * Initializes centers from puncture data, allocates AH workspaces.
 * AH finder uses (ah_n_theta, ah_n_phi) angular resolution per BH.
 */
bh_tracker_t *bh_tracker_alloc(int n_bh, const puncture_data_t bhs[],
                                int ah_n_theta, int ah_n_phi);

/* Free tracker and all AH workspaces */
void bh_tracker_free(bh_tracker_t *tr);

/*
 * Update BH positions from lapse minima.
 * Uses successive exclusion-zone algorithm: find deepest lapse minimum,
 * assign to nearest active BH, mask exclusion zone (R = 2*M), repeat.
 *
 * Ref: GRChombo PunctureTracker (lapse-minimum search)
 */
void bh_tracker_update_positions(bh_tracker_t *tr, const mesh_t *m);

/*
 * Run AH finder on each active BH, extract mass/spin/area.
 * Updates ah workspace center to current BH position before finding.
 */
void bh_tracker_find_horizons(bh_tracker_t *tr, const mesh_t *m,
                               double tol, int max_iter);

/*
 * Check for mergers: if separation(i,j) < 3 * max(M_i, M_j),
 * mark both as merged and create remnant at midpoint.
 *
 * Ref: arXiv:2505.01495 Sec. III (merger detection via proximity)
 */
void bh_tracker_check_mergers(bh_tracker_t *tr, double time);

/* Write CSV header line (dynamic columns based on n_bh) */
void bh_tracker_write_csv_header(const bh_tracker_t *tr, FILE *fp);

/*
 * Write one CSV data line per step.
 * Columns: time, ham_l2, mom_l2, n_active, n_leaves,
 *          [bh0_x, bh0_y, bh0_z, bh0_mass, bh0_spin, bh0_lapse] x N,
 *          n_mergers
 */
void bh_tracker_write_csv(const bh_tracker_t *tr, FILE *fp, double time,
                           double ham_l2, double mom_l2, int n_leaves);

/* Write merger event log to file */
void bh_tracker_write_mergers(const bh_tracker_t *tr, const char *path);

#endif /* LATTICE_BH_TRACKER_H */
