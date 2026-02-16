# Lattice — 3D Numerical Relativity Simulator

## About

C codebase implementing the CCZ4 (conformal covariant Z4) formulation of
Einstein's field equations for evolving black hole spacetimes through inspiral,
merger, and ringdown. GPU acceleration via OpenMP target offloading — physics
kernels are pure C, a thin backend abstraction swaps CPU (OpenMP threads) and
GPU (OpenMP target teams) with no platform-specific code.

Primary dev target: Apple M4 (16 GB unified memory) for CPU, NVIDIA HPC GPUs
(V100/A100/H100) for production runs. Requires GCC 15+ for GPU offloading.

## Approach

CCZ4 evolution equations are implemented from the published mathematics
(arXiv:1106.2254) using GRChombo's open-source implementation (BSD 3-Clause)
as a reference for correctness. We write our own C code — our architecture,
memory layout, function design, and GPU backends are entirely original.
GRChombo's C++ is only consulted to verify that tensor expressions are correct.

The reference repository is stored at `grchombo-ref/` (read-only, not compiled).
Key equation files:

| GRChombo file | What it contains |
|---|---|
| `Source/CCZ4/CCZ4RHS.impl.hpp` | CCZ4 right-hand-side equations |
| `Source/CCZ4/CCZ4Geometry.hpp` | Ricci tensor with Z terms |
| `Source/utils/TensorAlgebra.hpp` | Christoffel symbols, metric inverse, traces |
| `Source/CCZ4/MovingPunctureGauge.hpp` | Bona-Masso lapse + Gamma-driver shift |
| `Source/CCZ4/NewConstraints.impl.hpp` | Hamiltonian + momentum constraints |
| `Source/CCZ4/Weyl4.impl.hpp` | Psi4 gravitational wave extraction |

When citing in code comments, reference both the original paper equation and
the GRChombo file/line for cross-checking.

## Development Phases

### Phase 1: Vacuum CCZ4 (replicate GRChombo's core)

Get a working vacuum CCZ4 evolution code that passes the same test ladder
GRChombo can handle. No charge, no spin, no Maxwell, no AMR. Uniform grid only.

**Milestones:**

1. **Infrastructure** — grid, fields, memory layout, all 4 backends compiling
2. **Flat spacetime stable** — constraint violation < 1e-10 after 1000 steps
3. **Single Schwarzschild puncture stable 50M** — trumpet lapse ~0.3, 4th-order convergence
4. **Head-on binary collision** — merger visible in lapse, constraints bounded
5. **Binary inspiral (1+ orbits)** — orbital dynamics, psi4 waveform extraction

**Phase 1 scope:**
- CCZ4 RHS (chi, h_ij, K, A_ij, Theta, Gamma^i)
- Moving puncture gauge (Bona-Masso lapse, Gamma-driver shift with B^i)
- Christoffel symbols, Ricci tensor with Z terms
- Hamiltonian + momentum constraints
- 4th-order finite differences
- RK4 time integration
- 6th-order Kreiss-Oliger dissipation
- Sommerfeld radiative boundary conditions
- Puncture initial data (Brill-Lindquist for 1-2 BHs)
- Psi4 / Weyl4 extraction
- Algebraic enforcement: det(gambar)=1, tr(Abar)=0

### Phase 2: Novel Extensions

Once Phase 1 passes all tests at 4th-order convergence, add original features.
Priority order:

1. **N-body initial data** — multigrid elliptic solver for N punctures (N=3+),
   extending Brandt-Brugmann beyond the 2-body TwoPunctures limit
2. **Einstein-Maxwell (charge)** — couple Maxwell's equations to CCZ4,
   charged puncture initial data, evolve E_i and B_i fields
3. **Spin** — Bowen-York spinning puncture initial data, spin diagnostics
4. **Apparent horizons** — AH finder for remnant properties
5. **AMR** — block-structured (Berger-Oliger) adaptive mesh refinement

### Phase 3: Production

- N=10+ simultaneous black holes with arbitrary mass, spin, charge
- Position-dependent eta for unequal mass: eta(x) = eta_0 / W(x)
- Spatially varying KO dissipation
- Full waveform catalog capability

**Current status: Phase 1 — infrastructure + flat spacetime + single BH stable. AMR Stage 3 complete (prolongation + noise reduction).** Update as milestones are reached.

## Project Structure

```
lattice/
├── CLAUDE.md
├── DEVLOG.md
├── Makefile
├── grchombo-ref/           # read-only reference (not compiled)
├── src/
│   ├── core/
│   │   ├── grid.h / grid.c     # grid allocation, indexing, ghost zones
│   │   ├── fields.h            # field enum, count
│   │   └── params.h            # simulation parameters
│   ├── backend/
│   │   ├── backend.h           # abstract interface
│   │   ├── backend_cpu.c       # OpenMP threads (CPU)
│   │   └── backend_gpu.c       # OpenMP target offloading (GPU)
│   ├── evolution/
│   │   ├── ccz4_rhs.c          # CCZ4 right-hand-side
│   │   └── dissipation.c       # Kreiss-Oliger dissipation
│   ├── geometry/
│   │   └── tensor_utils.h      # inline tensor operations
│   ├── numerics/
│   │   ├── finite_diff.h       # FD_D1, FD_D2 macros (4th-order)
│   │   └── rk4.c               # RK4 time integrator (+mesh stepping)
│   ├── initial_data/
│   │   └── puncture.c          # Brill-Lindquist puncture data
│   ├── diagnostics/
│   │   ├── constraints.c       # Hamiltonian + momentum constraints
│   │   └── psi4.c              # Weyl4 scalar extraction
│   ├── boundary/
│   │   └── sommerfeld.c        # radiative BCs (+block-aware variant)
│   ├── amr/
│   │   ├── morton.h             # Morton (Z-order) encoding for SFC
│   │   ├── block.h / block.c   # block_t: single mesh block with metadata
│   │   ├── mesh.h / mesh.c     # mesh_t: collection of blocks forming domain
│   │   ├── ghost_exchange.h/c  # 26-neighbor ghost zone exchange
│   │   ├── prolongation.h/c    # 4th-order Lagrange coarse→fine (AthenaK)
│   │   ├── restriction.h/c     # volume-weighted fine→coarse averaging
│   │   └── meshblock_pack.h/c  # GPU batch packing (AthenaK-style)
│   └── io/
│       └── output.c            # data output
├── tests/
│   ├── test_flat.c             # flat spacetime stability
│   ├── test_single_bh.c        # single puncture evolution
│   ├── test_gauge_wave.c       # gauge wave propagation
│   ├── test_amr_mesh.c         # AMR mesh creation + Morton ordering
│   ├── test_amr_ghost.c        # ghost exchange + multi-block evolution
│   ├── test_amr_prolong.c     # prolongation + noise reduction
│   └── convergence.sh          # 3-resolution convergence check
├── docs/
│   ├── physics.md              # variable-to-math mapping
│   └── architecture.html       # interactive codebase map
└── tools/
    └── plot_convergence.py
```

## Build & Test

```bash
make                    # CPU build (-O3 -ffast-math -march=native)
make BACKEND=gpu        # GPU build (OpenMP target offloading, requires GCC 15+)
make debug              # debug build (-O0 -g -fsanitize=address,undefined)
make test               # all tests
make test-convergence   # 3-resolution convergence verification
make test-amr-mesh      # AMR mesh creation + Morton ordering
make test-amr-ghost     # ghost exchange + multi-block evolution
make test-amr-prolong   # prolongation + noise reduction (CAKO/CAHD/SSL)
make clean
```

Backend flag: `BACKEND=cpu|gpu`. Default is `cpu` (OpenMP threads).
Time integrator: `--rk classic|ck45`. Default is `ck45` (Carpenter-Kennedy 2N
low-storage, 3 memory blocks). Use `--rk classic` for standard 4-stage RK4
(4 memory blocks).
Compiler: `clang` on macOS (CPU only), `gcc-15` on Linux (CPU + GPU).
No external dependencies beyond standard C and libomp.
Debug builds enable NaN/Inf checking — any floating-point trap is a bug.

### GPU backend requirements

GPU offloading uses OpenMP target (`#pragma omp target teams distribute
parallel for`). No CUDA, HIP, or Metal code — the same C physics kernels run
on both CPU and GPU.

**Requirements:**
- GCC 15+ with `-foffload=nvptx-none` (NVIDIA) or `-foffload=amdgcn-amdhsa` (AMD)
- `GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE=16384` env var (required — the CCZ4
  RHS kernel uses ~4 KB stack per thread for derivative tensors, Christoffel
  symbols, and Ricci tensor locals; GCC 13's default stack is too small)
- HPC-class GPU with 1:2 FP64:FP32 ratio (V100, A100, H100, MI250X, MI300X).
  Consumer GPUs (1:32 ratio) have negligible FP64 throughput.

```bash
# Example: GPU build on Linux with GCC 15 targeting NVIDIA
make BACKEND=gpu CC=gcc-15

# Run with stack size env var
export GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE=16384
./build/lattice --N 256 --steps 400 --puncture 1.0,0,0,0
```

## Code Style

- **C17. No C++.**
- `snake_case` everywhere. `_t` suffix for typedefs. `UPPER_SNAKE_CASE` for constants.
- Field enum uses `FIELD_` prefix (e.g. `FIELD_CHI`, `FIELD_H11`).
- **Comment the physics, not the syntax.** Every function header cites:
  (1) the equation from arXiv:1106.2254 or B&S,
  (2) the corresponding GRChombo file:line for cross-reference.

## Evolved Variables (Phase 1: 25 fields)

| Variable | Count | Math | Description |
|---|---|---|---|
| `chi` | 1 | chi | conformal factor |
| `h[6]` | 6 | h_ij | conformal metric (symmetric) |
| `K` | 1 | K | trace of extrinsic curvature |
| `A[6]` | 6 | A_ij | traceless conformal extrinsic curvature |
| `Gamma[3]` | 3 | Gamma^i | conformal connection functions |
| `Theta` | 1 | Theta | CCZ4 constraint scalar |
| `lapse` | 1 | alpha | lapse function |
| `shift[3]` | 3 | beta^i | shift vector |
| `B[3]` | 3 | B^i | Gamma-driver auxiliary variable |

Total: 25 evolved fields. Phase 2 adds E_i, B_i (6 more for Maxwell).

## Physics Conventions

**Naming suffixes** encode tensor character:
- `_dd` = covariant (lower indices), `_uu` = contravariant (upper indices)
- `d1_` = first derivative, `d2_` = second derivative

**Symmetric 3x3 tensors** stored as `[3][3]` arrays (matching GRChombo's
convention for clarity). Use direct `[i][j]` indexing in loops.

**Index ordering:** `FOR(i,j)` loops run i=0..2, j=0..2. Spatial dimension
constant `GR_SPACEDIM = 3`.

## Memory Layout

- **Struct-of-arrays (SoA).** Each field is a contiguous `double*` array.
- x is the innermost (unit-stride) index. Loops: z (outer) -> y -> x (inner).
- Ghost zone width = 4 (4th-order stencils + 6th-order KO dissipation).
- Allocations page-aligned (4096 bytes) for zero-copy GPU buffers.
- NX padded to next multiple of 16 for cache alignment.

## Critical Invariants — DO NOT VIOLATE

1. **All physics arrays are `double` (64-bit).** Metal shaders may use `float`.
2. **SoA layout only.** Never convert to AoS.
3. **All finite differences go through `FD_D1()` / `FD_D2()` macros.** No hand-coded stencils.
4. **Innermost loop is always x.** No exceptions.
5. **Field enum ordering is append-only.** Never reorder existing entries.
6. **det(gambar) = 1 enforced algebraically** after every full RK4 step.
7. **Abar trace-free enforced algebraically** after every full RK4 step.
8. **Convergence order = 4** for all CCZ4 variables. Any change that breaks this is a bug.
9. **Physics kernels must never `#include` platform headers.** GPU interaction goes through `backend.h`.

## CCZ4 Parameters

| Parameter | Default | Description |
|---|---|---|
| `kappa1` | 0.1 | Constraint damping (Theta + Z_i). GRChombo uses kappa1/alpha (covariant). |
| `kappa2` | 0.0 | Controls mix of Theta damping in K equation |
| `kappa3` | 1.0 | Controls Z contribution in Gamma equation |
| `sigma` | 0.3 | Kreiss-Oliger dissipation strength |
| `lapse_coeff` | 2.0 | Coefficient c in 1+log slicing: dt(alpha) = -c * alpha * (K - 2*Theta) |
| `lapse_power` | 1.0 | Power p in Bona-Masso: f(alpha) = c * alpha^(p-2) |
| `shift_Gamma_coeff` | 0.75 | F in dt(beta^i) = F * B^i |
| `eta` | 1.0 | Damping in Gamma-driver: dt(B^i) = dt(Gamma^i) - eta * B^i |
| `rk_method` | `RK_CK45` | Time integrator: `RK_CLASSIC` (4 stages, 4 blocks) or `RK_CK45` (5 stages, 3 blocks) |

## Workflow Rules

**Do autonomously:** Bug fixes, style cleanup, comments, tests, single-file
refactors, updating DEVLOG.md.

**Ask first:** Adding/removing files, changing field enum, modifying RHS
equations, changing memory layout, altering Makefile targets, anything in
`numerics/` affecting convergence order.

**After any code change:**
1. `make` must succeed with zero warnings (`-Wall -Wextra -Werror`)
2. `make test` must pass
3. If touching `evolution/` or `numerics/`: `make test-convergence` must confirm 4th-order
4. If adding/removing/renaming files or functions, or changing the call graph:
   update `docs/architecture.html` to reflect the new structure

**When writing new code:** Start from the equation, cite both the paper and the
GRChombo reference line. Write the test first when possible. One function = one
physical operation. Log decisions in DEVLOG.md.

**When debugging:** Check convergence order first — wrong order = code bug,
right order but wrong magnitude = physics issue. Walk the test ladder:
flat spacetime -> single BH -> binary.

## Hardware Constraints

**CPU dev target:** Apple M4 (16 GB). Practical grid limit: 128^3 with CK45
(~5 GB for 25 fields x 3 blocks). N=256 barely fits (10.3 GB CK45).
Measured performance: ~4.5 sec/step at N=128 (~23 GFLOPS effective FP64).

**GPU production targets:** NVIDIA V100/A100/H100 via OpenMP target offloading.
Estimated 20-100x speedup over M4 CPU (0.05-0.2 sec/step at N=128).
N=256 requires 40+ GB GPU memory. N=512 requires H200 (141 GB) or multi-GPU.

No platform-specific code outside `src/backend/`. Physics kernels are pure C
with `#pragma omp declare target` guards for GPU compilation.

## DEVLOG.md

Maintain a running log of decisions, test results, equation references, and
design rationale. Every code addition should have a corresponding entry.

## Key References

### Equation Sources
- **arXiv:1106.2254**: Original CCZ4 paper (Alic et al. 2012) — primary equation source
- **B&S**: Baumgarte & Shapiro, *Numerical Relativity* (2010)

### Reference Implementation
- **GRChombo**: Andrade et al. 2021 (JOSS, doi:10.21105/joss.03703) — equation cross-reference
- **arXiv:1503.03436**: GRChombo methods paper

### Formulation & Methods
- **gr-qc/9810065**: BSSN formulation
- **gr-qc/9703066**: Brandt-Brugmann puncture method
- **gr-qc/0206072**: Gamma-driver shift condition
- **gr-qc/0511048**: Moving punctures (Campanelli et al.)
- **arXiv:2501.01055**: CCZ4 variants — CCZ3 stable to 10^5 M

### Phase 2 References (for later)
- **arXiv:0907.1151**: Einstein-Maxwell 3+1 (Alcubierre et al.)
- **arXiv:1903.01036**: Charged puncture initial data (Bozzola & Paschalidis)
- **arXiv:2104.06978**: Charged binary inspiral
- **arXiv:1004.1353**: N-puncture multigrid solver (Lousto et al.)
- **arXiv:1003.0859**: Position-dependent eta (Muller & Brugmann)
- **arXiv:2404.01137**: Improved KO dissipation + constraint damping
- **arXiv:2505.01495**: GRChombo 25-BH cluster simulation
- **arXiv:2312.05438**: AMR refinement strategy comparison
