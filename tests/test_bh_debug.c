/* Quick diagnostic: run BH test with verbose output near puncture */
#include "../src/core/grid.h"
#include "../src/core/fields.h"
#include "../src/core/params.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef void (*rhs_func_t)(grid_t *g);
extern void rk4_step(grid_t *g, rhs_func_t rhs_func, double dt);
extern void enforce_algebraic_constraints(grid_t *g);
extern void sommerfeld_apply(grid_t *g);
extern void ccz4_rhs(grid_t *g);
extern void gauge_rhs(grid_t *g);
extern void dissipation_apply(grid_t *g);
extern void constraints_l2(grid_t *g, double *ham_l2, double *mom_l2);
extern int backend_init(void);
extern void backend_shutdown(void);
extern void puncture_set_single(grid_t *g, double mass, double cx, double cy, double cz);

static void full_rhs(grid_t *g)
{
    grid_zero_rhs(g);
    sommerfeld_apply(g);
    ccz4_rhs(g);
    gauge_rhs(g);
    dissipation_apply(g);
}

int main(void)
{
    sim_params_t params;
    params_set_defaults(&params);
    params.nx = params.ny = params.nz = 64;
    params.lx = params.ly = params.lz = 256.0;
    params.cfl = 0.25;
    params.kappa1 = 0.02;
    params.kappa2 = 0.0;
    params.eta = 2.0;
    params_init(&params);
    backend_init();

    grid_t g;
    g.params = params;
    grid_alloc(&g);
    /* Offset puncture by dx/2 to avoid grid point at coordinate singularity */
    puncture_set_single(&g, 1.0, 0.5 * params.dx, 0.5 * params.dy, 0.5 * params.dz);

    /* Find the grid point nearest to origin */
    int ci = params.ghost_width + (params.nx - 2*params.ghost_width) / 2;
    int cj = params.ghost_width + (params.ny - 2*params.ghost_width) / 2;
    int ck = params.ghost_width + (params.nz - 2*params.ghost_width) / 2;
    int center = grid_idx(&g, ci, cj, ck);
    
    printf("Center point: (%d,%d,%d), x=%.2f y=%.2f z=%.2f\n",
           ci, cj, ck, grid_x(&g, ci), grid_y(&g, cj), grid_z(&g, ck));
    printf("dx = %.4f, dt = %.6f\n", params.dx, params.dt);
    printf("\nInitial at center: chi=%.6e, alpha=%.6e, K=%.6e\n",
           g.fields[FIELD_CHI][center], g.fields[FIELD_ALPHA][center],
           g.fields[FIELD_TRKA][center]);
    
    /* Check min chi at init */
    double chi_min = 1e30, alpha_min = 1e30;
    int gw = params.ghost_width;
    for (int k = gw; k < params.nz - gw; k++)
      for (int j = gw; j < params.ny - gw; j++)
        for (int i = gw; i < params.nx - gw; i++) {
            int idx = grid_idx(&g, i, j, k);
            if (g.fields[FIELD_CHI][idx] < chi_min) chi_min = g.fields[FIELD_CHI][idx];
            if (g.fields[FIELD_ALPHA][idx] < alpha_min) alpha_min = g.fields[FIELD_ALPHA][idx];
        }
    printf("Init chi_min=%.6e, alpha_min=%.6e\n\n", chi_min, alpha_min);

    printf("%4s %10s %12s %12s %12s %12s %12s %12s\n",
           "step", "time", "alpha_ctr", "chi_ctr", "K_ctr", "Theta_ctr", "alpha_min", "chi_min");

    for (int step = 0; step <= 15; step++) {
        /* Find min alpha and chi */
        chi_min = 1e30; alpha_min = 1e30;
        double chi_max = -1e30;
        for (int k = gw; k < params.nz - gw; k++)
          for (int j = gw; j < params.ny - gw; j++)
            for (int i = gw; i < params.nx - gw; i++) {
                int idx = grid_idx(&g, i, j, k);
                double c = g.fields[FIELD_CHI][idx];
                double a = g.fields[FIELD_ALPHA][idx];
                if (c < chi_min) chi_min = c;
                if (c > chi_max) chi_max = c;
                if (a < alpha_min) alpha_min = a;
            }
        
        printf("%4d %10.4f %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e\n",
               step, g.time, 
               g.fields[FIELD_ALPHA][center],
               g.fields[FIELD_CHI][center],
               g.fields[FIELD_TRKA][center],
               g.fields[FIELD_THETA][center],
               alpha_min, chi_min);
        
        if (isnan(g.fields[FIELD_ALPHA][center]) || isnan(chi_min)) {
            printf("NaN detected!\n");
            break;
        }
        
        if (step < 15)
            rk4_step(&g, full_rhs, params.dt);
    }

    grid_free(&g);
    backend_shutdown();
    return 0;
}
