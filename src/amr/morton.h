/*
 * Lattice — 3D Numerical Relativity
 * Morton (Z-order) encoding/decoding for oct-tree AMR.
 *
 * Header-only. Standard bit-interleaving approach.
 * Handles coordinates up to 21 bits (2^21 blocks per side per level).
 *
 * Ref: Athena++ src/mesh/mesh.cpp (Morton-based block ordering)
 * Ref: Burstedde et al. 2011 (p4est — scalable oct-tree library)
 */

#ifndef LATTICE_MORTON_H
#define LATTICE_MORTON_H

#include <stdint.h>

/*
 * Encode 3D integer coordinates (ix, iy, iz) into a single Morton code.
 * Bits are interleaved: z2 y2 x2 z1 y1 x1 z0 y0 x0
 * This gives spatial locality: nearby blocks have nearby Morton codes.
 */
static inline int64_t morton_encode3d(int ix, int iy, int iz)
{
    int64_t code = 0;
    for (int b = 0; b < 21; b++) {
        code |= ((int64_t)((ix >> b) & 1) << (3 * b))
              | ((int64_t)((iy >> b) & 1) << (3 * b + 1))
              | ((int64_t)((iz >> b) & 1) << (3 * b + 2));
    }
    return code;
}

/*
 * Decode a Morton code back into 3D integer coordinates.
 */
static inline void morton_decode3d(int64_t code, int *ix, int *iy, int *iz)
{
    *ix = 0;
    *iy = 0;
    *iz = 0;
    for (int b = 0; b < 21; b++) {
        *ix |= (int)((code >> (3 * b))     & 1) << b;
        *iy |= (int)((code >> (3 * b + 1)) & 1) << b;
        *iz |= (int)((code >> (3 * b + 2)) & 1) << b;
    }
}

/*
 * Compute the Morton code for a child at octant `child` (0..7)
 * of a parent with Morton code `parent_morton`.
 *
 * Octant numbering: child = cz*4 + cy*2 + cx where cx,cy,cz in {0,1}.
 * Child coordinates = 2*parent + (cx, cy, cz).
 */
static inline int64_t morton_child(int64_t parent_morton, int child)
{
    int px, py, pz;
    morton_decode3d(parent_morton, &px, &py, &pz);
    int cx = child & 1;
    int cy = (child >> 1) & 1;
    int cz = (child >> 2) & 1;
    return morton_encode3d(2 * px + cx, 2 * py + cy, 2 * pz + cz);
}

/*
 * Compute the Morton code of the parent (halve coordinates).
 */
static inline int64_t morton_parent(int64_t child_morton)
{
    int cx, cy, cz;
    morton_decode3d(child_morton, &cx, &cy, &cz);
    return morton_encode3d(cx / 2, cy / 2, cz / 2);
}

#endif /* LATTICE_MORTON_H */
