/*
 * Lattice — 3D Numerical Relativity
 * Checkpoint/restart: save and restore full simulation state.
 *
 * Binary format:
 *   - 1024-byte header (magic, version, params, step, time, mesh metadata)
 *   - Per leaf block: level, logical location, origin, field data
 *
 * Usage:
 *   checkpoint_write(m, &p, step, "build/checkpoint_0100.lat");
 *   checkpoint_read("build/checkpoint_0100.lat", &m, &p, &step);
 */

#ifndef LATTICE_CHECKPOINT_H
#define LATTICE_CHECKPOINT_H

#include "../amr/mesh.h"
#include "../core/params.h"

/*
 * Write a checkpoint file.
 *   m:    mesh with all leaf blocks
 *   p:    simulation parameters (includes time)
 *   step: current evolution step
 *   path: output file path
 *
 * Returns 0 on success, -1 on error.
 */
int checkpoint_write(const mesh_t *m, const sim_params_t *p,
                     int step, const char *path);

/*
 * Read a checkpoint file and reconstruct the mesh.
 *   path:   input file path
 *   m_out:  receives reconstructed mesh (caller must mesh_free)
 *   p_out:  receives restored sim_params_t (including time)
 *   step_out: receives the step number at checkpoint
 *
 * The mesh is fully reconstructed with all leaf blocks populated,
 * neighbors rebuilt, and ghost zones exchanged. Ready for evolution.
 *
 * Returns 0 on success, -1 on error.
 */
int checkpoint_read(const char *path, mesh_t **m_out,
                    sim_params_t *p_out, int *step_out);

#endif /* LATTICE_CHECKPOINT_H */
