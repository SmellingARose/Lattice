/*
 * backend.h — Backend abstraction for CPU/GPU dispatch
 *
 * Physics kernels never include platform headers directly.
 * All platform interaction goes through this interface.
 *
 * Backends: cpu (OpenMP), metal, cuda, hip
 */

#ifndef LATTICE_BACKEND_H
#define LATTICE_BACKEND_H

#include "../core/grid.h"

/*
 * Initialize backend. Returns 0 on success.
 */
int backend_init(void);

/*
 * Shutdown backend and release resources.
 */
void backend_shutdown(void);

/*
 * Update a single field for one RK4 sub-step:
 *   dst[idx] = src[idx] + dt_factor * rhs[idx]
 * for all interior points.
 */
void backend_field_update(grid_t *g, double *dst, const double *src,
                          const double *rhs, double dt_factor, int field_id);

/*
 * Sync field data to device (GPU). No-op for CPU backend.
 */
void backend_sync_to_device(grid_t *g);

/*
 * Sync field data from device to host. No-op for CPU backend.
 */
void backend_sync_to_host(grid_t *g);

#endif /* LATTICE_BACKEND_H */
