/*
 * Lattice — 3D Numerical Relativity
 * Parameter structs for CCZ4 evolution.
 *
 * Ref: arXiv:1106.2254 (CCZ4 params)
 * Ref: GRChombo MovingPunctureGauge.hpp (gauge params)
 */

#ifndef LATTICE_PARAMS_H
#define LATTICE_PARAMS_H

#include <stdbool.h>

/* Time integration method */
typedef enum {
    RK_CLASSIC,  /* Classic 4-stage RK4 (4 blocks: fields, rhs, scratch, accum) */
    RK_CK45      /* Carpenter-Kennedy 2N low-storage RK4 (3 blocks, 5 RHS evals) */
} rk_method_t;

/* CCZ4 constraint damping parameters */
typedef struct {
    double kappa1;       /* constraint damping (Theta + Z_i), default 0.1   */
    double kappa2;       /* Theta damping mix in K equation, default 0       */
    double kappa3;       /* Z contribution in Gamma equation, default 1      */
    bool   covariant_Z4; /* if true, use kappa1 (covariant); else kappa1*alpha */
} ccz4_params_t;

/* Moving puncture gauge parameters */
typedef struct {
    double lapse_coeff;        /* c in 1+log: dt(alpha) = -c*alpha^p*(K-2*Theta) */
    double lapse_power;        /* p in Bona-Masso, default 1                      */
    double shift_Gamma_coeff;  /* F in dt(beta^i) = F * B^i, default 0.75         */
    double eta;                /* damping in Gamma-driver: dt(B^i) -= eta*B^i     */
    double lapse_advec_coeff;  /* advection coefficient for lapse, default 0       */
    double shift_advec_coeff;  /* advection coefficient for shift, default 0       */
} gauge_params_t;

/*
 * AMR parameters.
 * Ref: plan1.md Stage 4, Athena++ ChomboParameters pattern.
 * Ref: arXiv:2312.05438 (chi-gradient vs truncation error comparison)
 */
typedef struct {
    int    enabled;        /* 0 = uniform grid (default), 1 = AMR             */
    int    max_level;      /* max refinement levels (default 6)               */
    int    N_block;        /* interior cells per block side (default 32)      */
    int    N_root;         /* root blocks per side (default 4)                */
    double chi_refine;     /* refine threshold for |grad(chi)|*dx/chi^2       */
    double chi_coarsen;    /* coarsen threshold (hysteresis)                  */
    int    regrid_every;   /* check interval: 1=every step, 0=never (static)  */
} amr_params_t;

/* Simulation parameters */
typedef struct {
    int    N;              /* grid points per side (interior)        */
    double L;              /* physical domain size                   */
    double dx;             /* grid spacing = L / N                   */
    double dt;             /* time step = CFL * dx                   */
    double CFL;            /* CFL factor, default 0.25               */
    double sigma;          /* Kreiss-Oliger dissipation, default 0.3 */
    int    num_steps;      /* total evolution steps                  */
    int    output_every;   /* output interval (0 = never)            */

    rk_method_t rk_method;  /* RK_CLASSIC or RK_CK45              */

    ccz4_params_t  ccz4;
    gauge_params_t gauge;
    amr_params_t   amr;
} sim_params_t;

/* Default parameter initialization */
static inline sim_params_t default_params(void)
{
    sim_params_t p;
    p.N            = 32;
    p.L            = 10.0;
    p.CFL          = 0.25;
    p.num_steps    = 1000;
    p.output_every = 0;
    p.sigma        = 0.3;
    p.rk_method    = RK_CK45;

    p.dx = p.L / p.N;
    p.dt = p.CFL * p.dx;

    p.ccz4.kappa1       = 0.1;
    p.ccz4.kappa2       = 0.0;
    p.ccz4.kappa3       = 1.0;
    p.ccz4.covariant_Z4 = true;

    p.gauge.lapse_coeff       = 2.0;
    p.gauge.lapse_power       = 1.0;
    p.gauge.shift_Gamma_coeff = 0.75;
    p.gauge.eta               = 1.0;
    p.gauge.lapse_advec_coeff = 0.0;
    p.gauge.shift_advec_coeff = 0.0;

    /* AMR defaults (disabled by default — uniform grid) */
    p.amr.enabled       = 0;
    p.amr.max_level     = 6;
    p.amr.N_block       = 32;
    p.amr.N_root        = 4;
    p.amr.chi_refine    = 0.1;
    p.amr.chi_coarsen   = 0.01;
    p.amr.regrid_every  = 1;

    return p;
}

#endif /* LATTICE_PARAMS_H */
