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

## 2026-02-11: Contiguous Memory Allocation

Changed grid allocation from per-field `posix_memalign` calls to contiguous
block allocation. Each group (fields, rhs, scratch, accum) is now a single
contiguous allocation of `NUM_FIELDS * npoints` doubles. Per-field pointer
arrays (`g->fields[f]`, etc.) index into these blocks.

Benefits:
- Slightly better CPU cache utilization (adjacent fields contiguous)
- Required for efficient GPU mapping (one `omp target enter data` per block
  instead of per-field)
- Simpler allocation/deallocation code

No functional change to CPU performance. Tests pass unchanged.

## 2026-02-11: GPU Backend Investigation (GCC 13 + Tesla P40)

### Setup

- **GPU**: NVIDIA Tesla P40 (24 GB GDDR5X, compute capability 6.1)
- **Secondary GPU**: GT 1030 (2 GB, device 1 — not used)
- **Compiler**: GCC 13.3 with `-foffload=nvptx-none`
- **Driver**: CUDA 570.211.01
- **Build flags**: `-fopenmp -foffload=nvptx-none -fcf-protection=none
  -fno-stack-protector -foffload-options="-lm -fno-stack-protector" -DLATTICE_GPU`

### What worked

All basic GPU operations verified working:
1. Device detection (2 devices found)
2. Simple `#pragma omp target` regions
3. `omp target teams distribute parallel for` with collapse(3)
4. `omp target enter data` for persistent GPU memory residency
5. Multiple target regions reusing mapped data via `map(tofrom:)`
6. Large allocations (~3.5 GB, 4 blocks of ~2.6 MB each for N=16)
7. Pointer arithmetic inside target regions (reconstructing per-field
   pointer arrays from contiguous blocks on-device)
8. Math functions: `pow()`, `cbrt()`, `sqrt()` all work correctly
9. Multi-field stencil access patterns (reading all 25 fields with
   finite-difference-like neighbor access)
10. Full RK4 pipeline with stubbed kernel (zero RHS) — mapping, RK4 stages,
    Sommerfeld BCs, algebraic enforcement all work end-to-end

### What didn't work

`ccz4_rhs_point` crashes with "illegal memory access" when called on GPU,
even with a single thread (`#pragma omp target` without teams/parallel).

### Root cause: GPU thread stack overflow

Systematic bisection (GPU_BISECT preprocessor levels 0-10) with dead-code
elimination controls revealed that the crash occurs when the compiler must
preserve all ~4 KB of local variables in `ccz4_rhs_point`:

| Bisect level | What runs | Result |
|---|---|---|
| 0 | Zero RHS and return (stub) | PASS |
| 1-9 | Compute through gauge RHS, write zeros | PASS (DCE removes dead computation) |
| 9 (fixed) | Compute + write 4 real values | PASS (DCE still removes most) |
| 9 (fixed) | Compute + write all 25 real values | **CRASH** |
| 10 | Full store, skip KO dissipation | **CRASH** |
| Full | Complete function | **CRASH** |

The function's ~4 KB+ of stack locals (3x3x3x3 derivative tensors, Christoffel
symbols, Ricci tensor, etc.) exceeds the GCC nvptx soft-stack default. When
`-O3` eliminates dead code, the effective stack usage drops below the limit
and the function appears to work.

### Attempted fixes (none resolved the crash)

1. `-msoft-stack-reserve-local=1024` and `=4096` — no effect
2. `-O1` optimization for nvptx offloaded code — no effect
3. `GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE` env var — GCC 15+ only, not
   available in GCC 13
4. Removing `static` from `const` array declarations — no effect
5. Wrapping all declarations and definitions in `#pragma omp declare target` —
   correct but didn't fix the stack issue

### Lessons learned

- GCC 13's nvptx soft-stack default is insufficient for large numerical kernels
- `-O3` dead-code elimination can mask stack overflow during bisection
- `map(present:)` (OpenMP 5.1) not supported by GCC 13
- `is_device_ptr` doesn't work with `omp target enter data`-mapped pointers
  in GCC 13; must use `map(tofrom:)` to reference already-mapped data
- GCC 15 adds `GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE` for runtime control
  of native GPU thread stack — this would likely fix the issue

### Recommendation for future GPU work

Options to resolve the stack overflow:
1. **Upgrade to GCC 15** — adds env var for native GPU stack control
2. **Use CUDA/nvcc directly** — full control over launch configuration
   and `cudaDeviceSetLimit(cudaLimitStackSize, N)`
3. **Split `ccz4_rhs_point`** — break into smaller functions to reduce
   peak stack usage (significant refactor, may hurt CPU performance)
4. **Use scratch memory** — move large derivative arrays to pre-allocated
   per-thread GPU scratch buffers instead of stack

GPU backend code reverted; CPU backend unchanged and fully functional.

## 2026-02-11: CK45 Low-Storage RK4 Integrator

### Problem

Single BH test (N=256) allocates 4 blocks of ~3.4 GB each = 13.7 GB, which
thrashes on the 16 GB M4. Classic RK4 needs 4 memory blocks: fields, rhs,
scratch (initial state backup), and accum (weighted sum).

### Solution

Added Carpenter-Kennedy 2N low-storage RK4 (CK45) as an alternative integrator.
Ref: Carpenter & Kennedy, NASA TM-109112 (1994), Solution 3.

CK45 uses only 3 memory blocks (fields=U, rhs=F, scratch=dU), saving 25% memory
at the cost of 5 RHS evaluations per step instead of 4. Both methods are
4th-order accurate.

Algorithm per step:
```
dU = 0
for s = 0..4:
    F = RHS(U); BCs(F)
    dU = A[s]*dU + dt*F
    U += B[s]*dU
enforce algebraic constraints
```

### Implementation

- `params.h`: Added `rk_method_t` enum (`RK_CLASSIC`, `RK_CK45`), default CK45
- `grid.h/c`: `grid_alloc()` takes `rk_method_t`, skips accum_block for CK45
- `rk4.c`: Classic RK4 renamed to `classic_rk4_step()`, CK45 added as
  `ck45_step()` with fused `ck45_update()` kernel. `rk4_step()` dispatches
  on `p->rk_method`
- `main.c`: Added `--rk classic|ck45` CLI flag
- All callers updated: main.c, test_flat.c, test_single_bh.c

### Memory savings

N=256: 3 blocks x 3.42 GB = 10.3 GB (fits in 16 GB) vs 4 blocks = 13.7 GB.

### OpenMP parallelization

Added `#pragma omp parallel for` to all field update loops in rk4.c:
`ck45_update`, `enforce_algebraic` (collapse(2) on k,j), `axpy_fields`,
`accum_add`, `apply_accum`. Previously only `backend_compute_rhs` was
parallelized, causing CPU utilization to oscillate between RHS (all cores)
and updates (single core).

### Test results

Flat spacetime (N=32, 1000 steps, CK45): Ham L2 = 5.3e-14 — PASSED (< 1e-10).
Classic RK4 unchanged and also passing.
Single BH (N=256) allocates successfully with CK45 (10.3 GB) — evolution running.
