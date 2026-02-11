/*
 * Lattice — 3D Numerical Relativity
 * 1D slice CSV output along x-axis through domain center.
 */

#include "../core/grid.h"
#include "../core/fields.h"
#include "../diagnostics/constraints.h"
#include <stdio.h>

void output_1d_slice(const grid_t *g, int step, double time)
{
    (void)time;
    char filename[256];
    snprintf(filename, sizeof(filename), "build/slice_%06d.csv", step);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "output: cannot open %s\n", filename);
        return;
    }

    fprintf(fp, "x,chi,lapse,K,Hamiltonian\n");

    int jc = g->ghost + g->N / 2;
    int kc = g->ghost + g->N / 2;

    for (int i = g->ghost; i < g->ghost + g->N; i++) {
        double x = COORD(g, i);
        int idx = IDX(g, i, jc, kc);

        double chi   = g->fields[FIELD_CHI][idx];
        double alpha = g->fields[FIELD_LAPSE][idx];
        double K_val = g->fields[FIELD_K][idx];
        double Ham   = compute_hamiltonian_at(
            (const double *const *)g->fields, g, i, jc, kc);

        fprintf(fp, "%.8e,%.8e,%.8e,%.8e,%.8e\n", x, chi, alpha, K_val, Ham);
    }

    fclose(fp);
}
