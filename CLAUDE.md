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
- 6th-order finite differences
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

**AMR parity gaps:** All closed. Both packed kernels (CPU + GPU) branch on
`p->em_enabled` to call `ccz4_maxwell_rhs_point`. Initial data solved
directly on evolution mesh (no interpolation). AH finder and output slices
work on AMR meshes.

**Current status: Phase 2 complete, Phase 3 in progress.**

**Phase 3 progress:**
- **6th-order operators:** FD stencils, KO dissipation, AMR prolongation (7-point)
  and restriction (6-point) all upgraded from 4th to 6th order. 4th-order Sommerfeld
  BCs. Quartic temporal interpolation for subcycling.
- **Tier 1 optimizations (all complete):** LTO for CPU builds, fast-path `pow(lapse,1)`,
  `restrict` qualifiers on RHS pointers, skip EM fields in dissipation/Sommerfeld,
  OMP-parallelized packed ghost exchange, flattened RK4 update loops (single OMP
  region over all fields), hoisted GPU grid_t construction, pre-computed
  restriction/prolongation weight product tables (232 entries, bit-exact verified),
  fused d1/d2 stencil (`fd_d1_d2()` loads 7 points once for both derivatives),
  conditional EM allocation (`grid_alloc_ex` with `n_fields` threaded through all
  subsystems — 25 fields when EM disabled, 19% memory savings).
- **Position-dependent eta:** `eta(x) = eta_0 / W(x)` where `W = sqrt(chi)` for
  stable unequal-mass binary evolution. Gated behind `position_dependent_eta` flag
  (default 1). Ref: arXiv:1003.0859 (Muller & Brugmann).
- **N-body initial data:** FAS multigrid constraint solver (FMG + Newton-Gauss-Seidel, 8-color GPU-compatible), O(N³) solve to discretization accuracy, arbitrary puncture count. BY 1-field + HiSpID 4-field coupled solvers.
- **Einstein-Maxwell:** 6 new evolved fields (E^i, B^i), conformal Maxwell evolution with constraint damping, EM stress-energy coupling to CCZ4 (gated by `--em` flag), charged puncture initial data via `--puncture M,x,y,z,Px,Py,Pz,Sx,Sy,Sz,Q`.
- **Spin:** Bowen-York spinning punctures + HiSpID high-spin initial data (quasi-isotropic Kerr conformal metric, coupled 4-field relaxation).
- **Apparent horizons:** Hyperbolic flow method (BHaHAHA-inspired) with 4th-order off-grid interpolation, mass/spin/area extraction, `--ah` CLI flag. Works on both single-grid and AMR meshes.
- **AMR:** Block-structured Berger-Oliger with subcycling, Morton-ordered mesh, 6th-order prolongation/restriction, multi-level ghost exchange. AMR-aware 1D output slices and AH finder.
- **Solve on evolution mesh:** AMR initial data constraint solver operates directly
  on evolution blocks (`set_bowen_york_mesh()`), eliminating interpolation error
  and ensuring exact discrete operator consistency. Solver reuses idle evolution
  arrays at t=0 (22 of 100 slots). Refinement depth defaults to `--max-level` so
  initial data and evolution use the same AMR depth (override with `--amr-levels`).
  Each level halves dx near punctures. Measured 218,000x better near-field constraint
  quality vs the old copy approach on refined meshes.
  Ref: Athena++ MG (Tomida & Stone 2023), arXiv:0912.2920 (Alic et al.).
- **Tier 0 bug fixes:** `enforce_algebraic_block()` in refine.c fixed (was using slow
  `1.0/cbrt(det)` with divergent `if (det > 0.0)` guard; now uses `fast_inv_cbrt(det)`
  unconditionally, matching `rk4.c`). Packed RHS kernel `g_local.n_fields` was 0
  (memset default), disabling KO dissipation in all AMR runs; fixed in both backends.
- **Default integrator:** Changed from CK45 to classic RK4 (`RK_CLASSIC`). Classic is
  faster (4 stages vs 5) but uses 25% more memory. All test allocations updated.
- **Tests:** Flat spacetime, convergence (order 6.5), Bowen-York (29/29), HiSpID (26/26),
  AH finder (13/13), Maxwell (15/15), pack_evolve (8/8), amr_prolong (15/15).
  Total: 31 evolved fields (25 CCZ4 + 6 EM).

Update as milestones are reached.

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
│   │   ├── params.h            # simulation parameters
│   │   └── timer.h             # TIMER_START/STOP macros (clock_gettime)
│   ├── backend/
│   │   ├── backend.h           # abstract interface
│   │   ├── backend_cpu.c       # OpenMP threads (CPU)
│   │   └── backend_gpu.c       # OpenMP target offloading (GPU)
│   ├── evolution/
│   │   ├── ccz4_rhs.c          # CCZ4 right-hand-side (+EM source terms)
│   │   ├── maxwell_rhs.h/c     # Maxwell evolution equations (E^i, B^i)
│   │   └── dissipation.c       # Kreiss-Oliger dissipation
│   ├── geometry/
│   │   └── tensor_utils.h      # inline tensor operations
│   ├── numerics/
│   │   ├── finite_diff.h       # FD_D1, FD_D2 macros (6th-order)
│   │   ├── interpolate.h       # 4th-order off-grid Lagrange interpolation
│   │   └── rk4.c               # RK4 time integrator (+mesh stepping)
│   ├── initial_data/
│   │   ├── puncture.c          # Brill-Lindquist puncture data
│   │   ├── bowen_york.h/c      # BY A_ij (momentum+spin) + CCZ4 conversion (+mesh-level API)
│   │   ├── relaxation.h/c      # FAS multigrid constraint solver (1-field + 4-field coupled)
│   │   ├── relaxation_amr.h/c  # AMR composite multigrid (FAS + uniform MG hierarchy)
│   │   └── kerr_quasi_isotropic.h/c  # QI Kerr metric for HiSpID (high-spin data)
│   ├── diagnostics/
│   │   ├── constraints.c       # Hamiltonian + momentum constraints
│   │   ├── ah_finder.h/c       # Apparent horizon finder (hyperbolic flow)
│   │   └── psi4.c              # Weyl4 scalar extraction
│   ├── boundary/
│   │   └── sommerfeld.c        # radiative BCs (+block-aware variant)
│   ├── amr/
│   │   ├── morton.h             # Morton (Z-order) encoding for SFC
│   │   ├── block.h / block.c   # block_t: single mesh block with metadata
│   │   ├── mesh.h / mesh.c     # mesh_t: collection of blocks forming domain
│   │   ├── ghost_exchange.h/c  # 26-neighbor + multi-level ghost exchange
│   │   ├── prolongation.h/c    # 6th-order Lagrange coarse→fine (AthenaK)
│   │   ├── restriction.h/c     # 6th-order Lagrange fine→coarse restriction
│   │   ├── criterion.h/c       # chi-gradient refinement criterion
│   │   ├── refine.h/c          # oct-tree split/merge/regrid
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
│   ├── test_head_on.c          # head-on binary collision
│   ├── test_head_on_output.txt # saved test output (merger diagnostics)
│   ├── test_amr_refine.c     # oct-tree refinement + multi-level ghost
│   ├── test_pack_evolve.c    # packed batch kernel validation
│   ├── test_subcycle.c       # Berger-Oliger subcycling validation
│   ├── test_bowen_york.c    # Bowen-York initial data (A_ij + solver + evolution)
│   ├── test_hispid.c        # HiSpID high-spin initial data (QI Kerr + coupled solver)
│   ├── test_ah_finder.c     # Apparent horizon finder tests (13/13)
│   ├── test_maxwell.c       # Einstein-Maxwell tests (15/15)
│   ├── test_inspiral_convergence.c  # AMR binary inspiral convergence (3 resolutions)
│   └── convergence.sh          # 3-resolution convergence check
├── docs/
│   ├── architecture.html       # consolidated architecture & design reference
│   └── archive/                # older deep-dive guides (preserved, not primary)
└── tools/
    ├── compute_amr_weights.py  # SymPy derivation of AMR stencil weights
    ├── verify_weights.c        # bit-exact verification of pre-computed weight tables
    └── plot_convergence.py
```

## Build & Test

```bash
make                    # CPU build (-O3 -ffast-math -march=native -flto)
make BACKEND=gpu        # GPU build (OpenMP target offloading, requires GCC 15+)
make debug              # debug build (-O0 -g -fsanitize=address,undefined)
make test               # all tests
make test-convergence   # 3-resolution convergence verification
make test-amr-mesh      # AMR mesh creation + Morton ordering
make test-amr-ghost     # ghost exchange + multi-block evolution
make test-amr-prolong   # prolongation + noise reduction (CAKO/CAHD/SSL)
make test-amr-refine    # oct-tree refinement + multi-level ghost exchange
make test-subcycle      # Berger-Oliger subcycling validation
make test-bowen-york   # Bowen-York initial data (A_ij, solver, evolution)
make test-hispid       # HiSpID high-spin initial data (QI Kerr + coupled solver)
make test-ah           # Apparent horizon finder (interpolation, Schwarzschild, diagnostics)
make test-maxwell      # Einstein-Maxwell (flat EM, plane wave, charged BH, constraints)
make test-inspiral-convergence  # AMR binary inspiral convergence (long run, ~hours)
make clean
```

Backend flag: `BACKEND=cpu|gpu`. Default is `cpu` (OpenMP threads).
Time integrator: `--rk classic|ck45`. Default is `classic` (standard 4-stage
RK4, 4 memory blocks). Use `--rk ck45` for Carpenter-Kennedy 2N low-storage
(5 stages, 3 memory blocks, 25% less memory).
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

## Evolved Variables (31 fields)

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
| `E[3]` | 3 | E^i | conformal electric field |
| `BM[3]` | 3 | B^i_mag | conformal magnetic field (BM prefix avoids clash with shift B^i) |

Total: 31 evolved fields (25 CCZ4 + 6 EM). EM fields enabled with `--em` flag.

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
- Ghost zone width = 4 (6th-order stencils + 6th-order KO dissipation).
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
8. **Convergence order >= 4** for all CCZ4 variables (measured ~6.5 with 6th-order FD). Any change that breaks this is a bug.
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
| `rk_method` | `RK_CLASSIC` | Time integrator: `RK_CLASSIC` (4 stages, 4 blocks) or `RK_CK45` (5 stages, 3 blocks) |
| `amr_levels` | `max_level` | Initial data solver refinement levels (`--amr-levels`). Defaults to `--max-level` so initial data and evolution use the same depth. Each level halves dx near punctures. Override for rare cases where you want finer initial data than evolution can afford. Requires `--rk classic`. |

### FAS Multigrid Solver Tuning

Both `relaxation_solve` (1-field BY) and `relaxation_solve_coupled` (4-field HiSpID)
use FMG (Full Multigrid) with FAS V-cycles and Newton-Gauss-Seidel smoothing.
`tol` is the target residual; `max_iter` is the max post-FMG V-cycles (usually 0-9).

The discretization floor is O(dx^4):

| Grid N | ~Floor residual | Recommended `tol` |
|--------|----------------|-------------------|
| 24     | 1e-3           | 1e-4              |
| 64     | 1e-7           | 1e-8              |
| 128    | 1e-9           | 1e-10             |
| 256    | 1e-11          | 1e-12             |

FMG achieves discretization accuracy in a single pass (~1.14 V-cycles of work).
Post-FMG V-cycles polish below the FMG residual; typically 0-9 needed.
Multigrid hierarchy: N, N/2, N/4, ... down to N_min=16.
For production, `tol=1e-12, max_iter=50000` is the default in `set_bowen_york()`.

**AMR solver resolution:** With `--amr-levels L`, the solver adds L refinement
levels near each puncture. Each level halves the grid spacing:
`dx_fine = dx_base / 2^L`. For example, a base grid at N=32, L=64 (dx=2M) with
`--amr-levels 3` achieves dx=0.25M near the BH. The solver does real work at the
fine resolution, so constraint quality scales with the finest dx, not the base dx.
Higher `--amr-levels` is more accurate but costs more compute (each level adds
up to 8x fine blocks that must be solved).

Ref: arXiv:0705.1486 (Natchu & Matzner), arXiv:2510.11152 (GPU FAS multigrid).

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

**CPU dev target:** Apple M4 (16 GB). Practical grid limit: 128^3 (~5 GB with
CK45, ~7 GB with classic RK4). N=256 barely fits (10.3 GB CK45, 13.7 GB classic).
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

### Phase 2 References (implemented)
- **arXiv:0907.1151**: Einstein-Maxwell 3+1 (Alcubierre et al.) — **implemented**
- **arXiv:1903.01036**: Charged puncture initial data (Bozzola & Paschalidis) — **implemented**
- **arXiv:2505.15912**: BHaHAHA hyperbolic AH flow algorithm — **implemented**
- **gr-qc/0512169**: AH finder review, expansion formula — **implemented**

### Phase 2 References (planned)
- **arXiv:2104.06978**: Charged binary inspiral
- **arXiv:1004.1353**: N-puncture multigrid solver (Lousto et al.)
- **arXiv:1003.0859**: Position-dependent eta (Muller & Brugmann)
- **arXiv:2404.01137**: Improved KO dissipation + constraint damping
- **arXiv:2505.01495**: GRChombo 25-BH cluster simulation
- **arXiv:2312.05438**: AMR refinement strategy comparison
