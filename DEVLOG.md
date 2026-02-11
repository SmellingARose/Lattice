# Lattice Development Log

## 2026-02-10: Fresh Start — Phase 1, Milestones 1+2

### Decision: Start from scratch with GRChombo reference approach

Previous implementation had accumulated bugs in tensor indexing, advection,
and CCZ4 equations. Rather than continuing to patch, starting clean with a
systematic approach:

1. Write each module from the published equations (arXiv:1106.2254)
2. Cross-reference every tensor expression against GRChombo's C++ code
3. Build bottom-up: core -> numerics -> geometry -> evolution -> integration
4. Test at each level: flat spacetime must be stable before moving on

### Architecture decisions

- **SoA memory layout**: Each field is a contiguous `double*` array.
  At each grid point the RHS loads needed values into local variables,
  computes, stores back. This is cache-friendly for the triple-loop
  backend and maps directly to GPU kernels later.

- **RK4 memory-efficient (5b)**: 1 scratch + 1 accumulator instead of 4
  k-arrays. Saves ~60% scratch memory. Important for fitting 128^3 grids
  on the M4's 16 GB.

- **Single RHS function per point**: `ccz4_rhs_point(rhs, src, g, p, i, j, k)`
  matches GRChombo's `rhs_equation()` structure. All tensor algebra is
  inline in `tensor_utils.h`. This makes the physics transparent and
  the function self-contained for GPU dispatch.

- **Backend abstraction**: `backend_compute_rhs()` owns the loop structure.
  CPU = OpenMP collapse(2) on z,y. GPU backends get their own kernel code.
  Physics code never includes platform headers.

### Files created

25 files implementing full CCZ4 vacuum evolution:
- Core: fields.h, params.h, grid.h/c
- Numerics: finite_diff.h, rk4.h/c
- Geometry: tensor_utils.h
- Evolution: ccz4_rhs.h/c, dissipation.c
- Initial data: puncture.h/c
- Diagnostics: constraints.h/c
- Boundary: sommerfeld.h/c
- IO: output.c
- Backend: backend.h, backend_cpu.c, backend_stubs.c
- Main: main.c
- Tests: test_flat.c
- Build: Makefile

### Equation references

- CCZ4 RHS: arXiv:1106.2254, GRChombo CCZ4RHS.impl.hpp:60-227
- Ricci with Z: GRChombo CCZ4Geometry.hpp:56-112
- Christoffel: GRChombo TensorAlgebra.hpp:344-367
- Moving puncture gauge: GRChombo MovingPunctureGauge.hpp:54-65
- FD stencils: GRChombo FourthOrderDerivatives.hpp
- KO dissipation: GRChombo FourthOrderDerivatives.hpp:361-378
- Sommerfeld BCs: GRChombo BoundaryConditions.cpp:593-661
- Constraints: GRChombo NewConstraints.impl.hpp:55-61
- Brill-Lindquist: gr-qc/9703066

### Test target

Flat spacetime: N=32, L=10, CFL=0.25, 1000 steps.
Hamiltonian constraint L2 < 1e-10.

## 2026-02-10: GPU Backend + Single BH Test

### GPU backend: OpenMP target offloading

Replaced the three stub backends (Metal/CUDA/HIP) with a single OpenMP target
offloading backend (`backend_gpu.c`). The GPU backend calls `ccz4_rhs_point`
directly instead of through a function pointer — GPU offloading can't do
callbacks. Physics kernels marked with `#pragma omp declare target` (guarded
by `LATTICE_GPU` define) in: `ccz4_rhs.h`, `tensor_utils.h`, `finite_diff.h`,
`dissipation.c`.

Build: `make BACKEND=gpu` (requires clang with `-fopenmp-targets=nvptx64` or
GCC with `-foffload=nvptx-none`). CPU backend unchanged.

Deleted `backend_stubs.c`. Makefile now accepts `BACKEND=cpu` (default) or
`BACKEND=gpu` only.

### Single BH test: test_single_bh.c

Single Brill-Lindquist puncture (M=1.0 at origin), N=256, L=64, dx=0.25,
CFL=0.25, evolve to T=30M (480 steps). Boundary at 32M from puncture —
no reflections corrupt the solution within the evolution window.

Requires ~15 GB RAM (run on 32GB+ machine).

**Pass criteria (final-time checks):**
- `min_lapse(final) < 0.5` (trumpet collapse)
- `check_finite` (no NaN/Inf)
- `ham_peak < 1.0` (constraints bounded)
