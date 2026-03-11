/*
 * Lattice — 3D Numerical Relativity
 * Parameter structs for CCZ4 evolution.
 *
 * Ref: arXiv:1106.2254 (CCZ4 params)
 * Ref: GRChombo MovingPunctureGauge.hpp (gauge params)
 */

#ifndef LATTICE_PARAMS_H
#define LATTICE_PARAMS_H

#include "device.h"
#include <stdbool.h>

EXTERN_C_BEGIN

/* Time integration method */
typedef enum {
    RK_CLASSIC,  /* Classic 4-stage RK4 (4 blocks: fields, rhs, scratch, accum) */
    RK_CK45      /* Carpenter-Kennedy 2N low-storage RK4 (3 blocks, 5 RHS evals) */
} rk_method_t;

/* Boundary condition type */
typedef enum {
    BC_SOMMERFELD = 0,            /* Standard Sommerfeld radiative BCs for all fields */
    BC_CONSTRAINT_PRESERVING = 1  /* CP BCs for constraint fields, Sommerfeld for rest.
                                   * Ref: arXiv:1212.2901 (Hilditch et al., BAM) */
} bc_type_t;

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
    int    position_dependent_eta; /* eta(x) = eta/W(x), W=sqrt(chi). Ref: arXiv:1003.0859 */
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
    double chi_refine;     /* refine threshold for |grad(chi)|*dx/chi^2       */
    double chi_coarsen;    /* coarsen threshold (hysteresis)                  */
    int    regrid_every;   /* check interval: 1=every step, 0=never (static)  */
    int    solver_levels;  /* AMR levels for initial data solver (-1 = use max_level) */
} amr_params_t;

/*
 * Noise reduction parameters.
 * CAKO, per-field sigma, and SSL on by default; CAHD off.
 * Ref: arXiv:2404.01137 (Etienne 2024)
 */
typedef struct {
    int    use_cako;           /* CAKO: chi-adjusted KO dissipation        */
    int    use_cahd;           /* CAHD: constraint-adjusted Hamiltonian    */
    int    use_ssl;            /* SSL: slow-start lapse                    */
    int    use_per_field_sigma; /* per-field dissipation strengths         */
    double sigma_gauge;        /* KO sigma for gauge fields (default 0.99) */
    double sigma_phys;         /* KO sigma for physical fields (def 0.3)   */
    double cahd_coeff;         /* CAHD C coefficient (default 0.15)        */
    double ssl_h;              /* SSL Gaussian height h/(M) (default 0.6)  */
    double ssl_sigma_t;        /* SSL Gaussian width σ_t/(M) (default 20)  */
    double ssl_total_mass;     /* total puncture mass M (default 1.0)       */
} noise_params_t;

/*
 * Puncture data: mass, position, linear momentum, spin.
 * Used by Bowen-York initial data (gr-qc/9703066).
 * 4 values = BL at rest, 7 = with momentum, 10 = with spin.
 */
#define MAX_PUNCTURES 32
typedef struct {
    double mass;
    double center[3];
    double momentum[3];
    double spin[3];
    double charge;        /* electric charge Q (default 0) */
} puncture_data_t;

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
    double time;           /* current simulation time (updated by caller) */

    rk_method_t rk_method;  /* RK_CLASSIC or RK_CK45              */

    ccz4_params_t  ccz4;
    gauge_params_t gauge;
    amr_params_t   amr;
    noise_params_t noise;

    int    em_enabled;    /* Einstein-Maxwell coupling (default 0)  */
    double kappa_em;      /* EM constraint damping (default 0.1)    */

    bc_type_t bc_type;    /* BC_SOMMERFELD or BC_CONSTRAINT_PRESERVING */
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
    p.rk_method    = RK_CLASSIC;

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
    p.gauge.position_dependent_eta = 1;
    p.gauge.lapse_advec_coeff = 1.0;  /* required for AMR gauge stability */
    p.gauge.shift_advec_coeff = 1.0;  /* Ref: gr-qc/0610128 (Brugmann et al.) */

    p.time = 0.0;

    /* AMR defaults (disabled by default — uniform grid) */
    p.amr.enabled       = 0;
    p.amr.max_level     = 6;
    p.amr.N_block       = 32;
    p.amr.chi_refine    = 0.1;
    p.amr.chi_coarsen   = 0.01;
    p.amr.regrid_every  = 1;
    p.amr.solver_levels = -1;  /* default: use max_level */

    /* Noise reduction defaults.
     * CAKO, per-field sigma, and SSL enabled by default — these are
     * standard production techniques (arXiv:2404.01137, Etienne 2024).
     * CAHD disabled by default (experimental, Phase 3). */
    p.noise.use_cako           = 1;
    p.noise.use_cahd           = 0;
    p.noise.use_ssl            = 1;
    p.noise.use_per_field_sigma = 1;
    p.noise.sigma_gauge        = 0.99;
    p.noise.sigma_phys         = 0.3;
    p.noise.cahd_coeff         = 0.15;
    p.noise.ssl_h              = 0.6;
    p.noise.ssl_sigma_t        = 20.0;
    p.noise.ssl_total_mass     = 1.0;

    /* Einstein-Maxwell defaults (disabled — vacuum CCZ4 by default) */
    p.em_enabled = 0;
    p.kappa_em   = 0.1;

    /* Boundary conditions: CP by default (better constraint preservation,
     * negligible overhead). Ref: arXiv:1212.2901 */
    p.bc_type = BC_CONSTRAINT_PRESERVING;

    return p;
}

EXTERN_C_END

#endif /* LATTICE_PARAMS_H */
