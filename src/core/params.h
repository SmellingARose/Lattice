/*
 * params.h — Simulation parameters
 *
 * All tunable parameters live here. Computed quantities (dx, dt, padded dims)
 * are derived in params_init().
 */

#ifndef LATTICE_PARAMS_H
#define LATTICE_PARAMS_H

#include "fields.h"

typedef struct {
    /* Grid dimensions (user-specified, before padding) */
    int nx, ny, nz;

    /* Padded dimensions (nx rounded up to next multiple of 16) */
    int nx_pad, ny_pad, nz_pad;

    /* Domain physical size */
    double lx, ly, lz;

    /* Grid spacing (computed) */
    double dx, dy, dz;

    /* Time step (computed from CFL) */
    double dt;

    /* Ghost zone width — must be >= 4 for 4th-order stencils + KO6 */
    int ghost_width;

    /* CFL factor */
    double cfl;

    /* CCZ4 constraint damping */
    double kappa1;
    double kappa2;
    double kappa3;

    /* Kreiss-Oliger dissipation coefficients */
    double ko_eps_gauge;  /* for alpha, beta, B (gauge variables) */
    double ko_eps_other;  /* for chi, gt, K, At, Ghat, Theta */

    /* Gamma-driver parameters */
    double eta;             /* damping in B^i equation */
    double gamma_driver_f;  /* coefficient f in dt beta^i = f * B^i */

    /* Number of fields to evolve */
    int num_fields;

    /* Enable Einstein-Maxwell fields */
    int enable_em;

    /* Output control */
    int output_every;       /* steps between output */
    char output_dir[256];   /* output directory path */
} sim_params_t;

/*
 * Set default parameter values.
 */
static inline void params_set_defaults(sim_params_t *p)
{
    p->nx = 32;
    p->ny = 32;
    p->nz = 32;

    p->lx = 10.0;
    p->ly = 10.0;
    p->lz = 10.0;

    p->ghost_width = 4;

    p->cfl = 0.25;

    p->kappa1 = 0.02;
    p->kappa2 = 0.0;
    p->kappa3 = 0.5;

    p->ko_eps_gauge = 0.99;
    p->ko_eps_other = 0.3;

    p->eta = 2.0;
    p->gamma_driver_f = 0.75;

    p->enable_em = 0;
    p->num_fields = NUM_VACUUM_FIELDS;

    p->output_every = 10;
    p->output_dir[0] = '.';
    p->output_dir[1] = '\0';
}

/*
 * Compute derived quantities: dx, dt, padded dimensions.
 * Call after setting nx/ny/nz and lx/ly/lz.
 */
static inline void params_init(sim_params_t *p)
{
    /* Pad nx to next multiple of 16 for cache alignment */
    p->nx_pad = ((p->nx + 15) / 16) * 16;
    p->ny_pad = p->ny;
    p->nz_pad = p->nz;

    /* Grid spacing: domain covers [0, L] with nx points including ghosts */
    p->dx = p->lx / (p->nx - 2 * p->ghost_width - 1);
    p->dy = p->ly / (p->ny - 2 * p->ghost_width - 1);
    p->dz = p->lz / (p->nz - 2 * p->ghost_width - 1);

    /* Time step from CFL condition */
    double dx_min = p->dx;
    if (p->dy < dx_min) dx_min = p->dy;
    if (p->dz < dx_min) dx_min = p->dz;
    p->dt = p->cfl * dx_min;

    /* Field count */
    if (p->enable_em) {
        p->num_fields = NUM_TOTAL_FIELDS;
    } else {
        p->num_fields = NUM_VACUUM_FIELDS;
    }
}

#endif /* LATTICE_PARAMS_H */
