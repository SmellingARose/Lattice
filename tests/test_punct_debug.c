/* Quick debug: check alpha at puncture */
#include "../src/core/grid.h"
#include "../src/core/fields.h"
#include "../src/core/params.h"
#include <stdio.h>
#include <math.h>

extern int backend_init(void);
extern void backend_shutdown(void);
extern void puncture_set_single(grid_t *g, double mass, double cx, double cy, double cz);
extern void enforce_algebraic_constraints(grid_t *g);

int main(void)
{
    sim_params_t params;
    params_set_defaults(&params);
    params.nx = params.ny = params.nz = 41;
    params.lx = params.ly = params.lz = 32.0;
    params.cfl = 0.25;
    params.kappa1 = 0.02;
    params.kappa2 = 0.0;
    params.eta = 2.0;
    params_init(&params);
    backend_init();

    grid_t g;
    g.params = params;
    if (grid_alloc(&g) != 0) return 1;

    int idx20 = grid_idx(&g, 20, 20, 20);
    printf("Before puncture: alpha[20,20,20]=%.6e chi=%.6e\n",
           g.fields[FIELD_ALPHA][idx20], g.fields[FIELD_CHI][idx20]);

    puncture_set_single(&g, 1.0, 0.0, 0.0, 0.0);
    printf("After puncture:  alpha[20,20,20]=%.6e chi=%.6e\n",
           g.fields[FIELD_ALPHA][idx20], g.fields[FIELD_CHI][idx20]);

    enforce_algebraic_constraints(&g);
    printf("After enforce:   alpha[20,20,20]=%.6e chi=%.6e\n",
           g.fields[FIELD_ALPHA][idx20], g.fields[FIELD_CHI][idx20]);

    /* Same loop as smoke test */
    double alpha_min = 1e30;
    double chi_min = 1e30;
    GRID_LOOP_INTERIOR(&g, i, j, k) {
        int idx = grid_idx(&g, i, j, k);
        double a = g.fields[FIELD_ALPHA][idx];
        double c = g.fields[FIELD_CHI][idx];
        if (a < alpha_min) alpha_min = a;
        if (c < chi_min) chi_min = c;
    }
    printf("alpha_min=%.6e chi_min=%.6e\n", alpha_min, chi_min);

    grid_free(&g);
    backend_shutdown();
    return 0;
}
