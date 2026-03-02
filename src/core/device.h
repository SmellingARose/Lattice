/*
 * Lattice — 3D Numerical Relativity
 * Device portability macros for HIP GPU backend.
 *
 * LATTICE_DEVICE: annotates functions callable from both host and device.
 *   - HIP build: expands to __host__ __device__
 *   - CPU build: expands to nothing (zero impact)
 *
 * EXTERN_C_BEGIN/END: wraps declarations with extern "C" when compiled as C++.
 *   Ensures C linkage for functions shared between gcc-compiled host code
 *   and hipcc-compiled device code.
 *
 * Include this header before any function that needs device annotation.
 */

#ifndef LATTICE_DEVICE_H
#define LATTICE_DEVICE_H

#if defined(LATTICE_HIP) && (defined(__HIPCC__) || defined(__cplusplus))
  /* Device/mixed compilation (hipcc or C++) — full HIP annotations */
  #include <hip/hip_runtime.h>
  #define LATTICE_DEVICE __host__ __device__
#else
  /* Host-only C compilation — no HIP header needed */
  #define LATTICE_DEVICE
#endif

#ifdef __cplusplus
  #define EXTERN_C_BEGIN extern "C" {
  #define EXTERN_C_END   }
#else
  #define EXTERN_C_BEGIN
  #define EXTERN_C_END
#endif

/* C++ doesn't have 'restrict' keyword; use compiler extension.
 * hipcc (clang-based) and gcc both support __restrict__. */
#ifdef __cplusplus
  #ifndef restrict
    #define restrict __restrict__
  #endif
#endif

#endif /* LATTICE_DEVICE_H */
