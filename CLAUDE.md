# Lattice — 3D Numerical Relativity Simulator

## About

C codebase implementing the CCZ4 (conformal covariant Z4) formulation of
Einstein's field equations coupled with Maxwell's equations (Einstein-Maxwell)
for evolving N black holes through inspiral, merger, and ringdown. Target N=10+
with arbitrary mass, spin, charge, position, and velocity. Extracts
gravitational waveforms (psi4), finds apparent horizons, and measures remnant
properties.

Multi-platform GPU acceleration via compile-time backend dispatch: physics
kernels are pure C, a thin abstraction layer swaps CPU (OpenMP) / Metal / CUDA
/ HIP. Primary dev target: Apple M4 (16 GB unified memory, 10 GPU cores).

## Tier Milestones

1. **Flat spacetime stable** — constraint violation < 1e-10 after 1000 steps
2. **Single Schwarzschild puncture stable 50M** — trumpet lapse ~0.3, 4th-order convergence
3. **Head-on binary collision** — merger in lapse collapse, constraints bounded
4. **Binary inspiral (1+ orbits) + N-body initial data** — orbital trajectory, N>=3 punctures
5. **psi4 extraction + apparent horizons + spin + charge** — waveforms, AH surfaces, demos

Current status: **Pre-coding (theory reading phase).** Update this line as tiers are reached.

## Project Structure

```
lattice/
├── CLAUDE.md
├── Makefile
├── DEVLOG.md
├── src/
│   ├── core/
│   │   ├── grid.h / grid.c
│   │   ├── fields.h
│   │   └── params.h
│   ├── backend/
│   │   ├── backend.h
│   │   ├── backend_cpu.c
│   │   ├── backend_metal.c
│   │   ├── backend_cuda.c
│   │   └── backend_hip.c
│   ├── evolution/
│   │   ├── ccz4_rhs.c
│   │   ├── gauge_rhs.c
│   │   ├── maxwell_rhs.c
│   │   └── dissipation.c
│   ├── geometry/
│   │   ├── christoffel.c
│   │   ├── ricci.c
│   │   └── tensor_utils.h
│   ├── numerics/
│   │   ├── finite_diff.h
│   │   ├── rk4.c
│   │   ├── elliptic.c
│   │   └── amr.c
│   ├── initial_data/
│   │   └── puncture.c
│   ├── diagnostics/
│   │   ├── constraints.c
│   │   ├── psi4.c
│   │   └── horizon.c
│   ├── boundary/
│   │   └── sommerfeld.c
│   └── io/
│       └── output.c
├── metal/
│   ├── ccz4_rhs.metal
│   └── rk4_update.metal
├── tests/
│   ├── test_flat.c
│   ├── test_gauge_wave.c
│   ├── test_single_bh.c
│   └── convergence.sh
├── docs/
│   ├── physics.md
│   └── methods.md
└── tools/
    └── plot_convergence.py
```

## Build & Test

```bash
make                    # optimized build (-O3 -ffast-math -march=native)
make BACKEND=metal      # build with Metal backend (default: cpu/OpenMP)
make BACKEND=cuda       # build with CUDA backend
make BACKEND=hip        # build with HIP/ROCm backend
make debug              # debug build (-O0 -g -fsanitize=address,undefined)
make test               # all tests
make test-convergence   # 3-resolution convergence verification
make clean
```

Backend flag: `BACKEND=cpu|metal|cuda|hip`. Default is `cpu` (OpenMP parallelism).
Compiler: `clang` on macOS, `gcc`/`nvcc` on Linux. No external dependencies beyond standard C and Accelerate (macOS).
Debug builds enable NaN/Inf checking — any floating-point trap is a bug, not a warning.

## Code Style

- **C17. No C++.**
- `snake_case` everywhere. `_t` suffix for typedefs. `UPPER_SNAKE_CASE` for constants.
- Field enum uses `FIELD_` prefix (e.g. `FIELD_CHI`, `FIELD_H11`).
- **Comment the physics, not the syntax.** Every function header cites the equation it implements (B&S chapter.equation or arXiv ID). Inline comments explain physics meaning.

## Physics Conventions

See `docs/physics.md` for the full variable-to-math mapping.

CCZ4 extends the standard BSSN variable set with constraint-damping fields:
Theta (scalar) and Z_i (vector). These promote constraint violations to damped
propagating modes that leave the grid instead of accumulating. One tunable
parameter kappa controls damping strength.

**Naming suffixes** encode tensor character:
- `_dd` = covariant (lower indices), `_uu` = contravariant (upper indices), `_u` = vector
- `d1_` = first derivative prefix, `d2_` = second derivative prefix

**Symmetric tensors** are flat arrays in order `[xx, xy, xz, yy, yz, zz]` (indices 0-5).
Use `SYM(i,j)` macro to convert index pairs to flat index.

## Memory Layout

- **Struct-of-arrays (SoA).** Each field is a contiguous `double*` array.
- x is the innermost (unit-stride) index. Loops are always z-y-x from outermost to innermost.
- Ghost zone width = 4 (supports 4th-order stencils + 6th-order Kreiss-Oliger dissipation).
- Allocations are page-aligned (4096 bytes) for zero-copy GPU buffers.
- NX padded to next multiple of 16 for cache alignment.
- Field count: ~29 vacuum (BSSN core + Theta, Z_i), ~35 with Einstein-Maxwell (+ E_i, B_i).

## Critical Invariants — DO NOT VIOLATE

1. **All physics arrays are `double` (64-bit).** Metal shaders may use `float` — validated separately.
2. **SoA layout only.** Never convert to AoS.
3. **All finite differences go through `FD_D1()` / `FD_D2()` macros.** No hand-coded stencils.
4. **Innermost loop is always x.** No exceptions.
5. **Field enum ordering is append-only.** Never reorder existing entries in `fields.h`.
6. **det(gambar) = 1 enforced algebraically** after every full RK4 step.
7. **Abar trace-free enforced algebraically** after every full RK4 step.
8. **Convergence order = 4** for all CCZ4 variables. Any change that breaks this is a bug.
9. **Physics kernels must never `#include` platform headers.** All GPU/platform interaction goes through `backend.h`.
10. **Berger-Oliger subcycling** for AMR levels — finer levels take proportionally smaller timesteps. Without this, the finest level's CFL forces all levels to tiny timesteps.

## Workflow Rules

**Do autonomously:** Bug fixes, style cleanup, comments, tests, single-file refactors, updating DEVLOG.md.

**Ask first:** Adding/removing files, changing field enum, modifying RHS equations, changing memory layout, altering Makefile targets, anything in `numerics/` affecting convergence order.

**After any code change:**
1. `make` must succeed with zero warnings (`-Wall -Wextra -Werror`)
2. `make test` must pass
3. If touching `evolution/` or `numerics/`: `make test-convergence` must confirm 4th-order

**When writing new code:** Start from the equation and cite it. Write the test first when possible. One function = one physical operation. Log decisions in DEVLOG.md.

**When debugging:** Check convergence order first — wrong order means code bug, right order but wrong magnitude means physics issue. Reduce to simplest failing case. Walk the test-problem ladder (flat spacetime -> single BH -> binary).

## Hardware Constraints

Apple M4 (16 GB) is the primary dev target. Code must compile and run on any platform via the backend abstraction — no platform-specific code outside `src/backend/` and `metal/`. Practical grid limit on M4: ~128 cubed in FP64 (~8 GB for 35 fields x 4 RK stages). GPU has no native FP64 — Metal shaders use FP32.

## Research Items

Resolved. See `docs/research_notes.md` for full findings, equations, and citations.

- [x] **AMR scheme**: **Block-structured (Berger-Oliger) with L₂ (sphere-in-sphere) refinement.** Only approach proven at N>10 (GRChombo N=25, arXiv:2505.01495). L₂ strategy is 60% cheaper than L∞ with equal or better accuracy (arXiv:2312.05438). Remaining: design level structure and regridding frequency for our hardware.
- [x] **CCZ4 variant**: **Start with standard CCZ4** (κ₁→κ₁/α modification per GRChombo). Fallback to CCZ3 (Θ=0, κ₁=0) if instabilities appear beyond ~20,000 M — CCZ3 is stable to 10⁵ M (arXiv:2501.01055). Implement spatially varying KO dissipation from day one: ε_KO = W·ε_{KO,CA} (arXiv:2404.01137). Full CCZ4 equations documented in research_notes.md.
- [x] **Initial data solver**: **Multigrid elliptic solver for N punctures** (not spectral TwoPunctures, which is limited to N=2 by bispherical coords). For charge: superpose conformal electric field Ē^i = Σ Q_n/R_n² R̂_n^i (satisfies Gauss constraint automatically), add EM source term 2πψ⁻³ε̄ to Hamiltonian constraint (arXiv:1903.01036). Proven to N=4 uncharged (arXiv:1004.1353).
- [x] **Position-dependent eta**: **Conformal-factor-based** η(x) = η₀/W(x). Avoids explicit puncture tracking, reuses W from KO dissipation, adapts to all N punctures automatically. Fallback: explicit reciprocal superposition 1/η = Σ 1/η_n(r_n) per Müller & Brügmann (arXiv:1003.0859). No published work exists for N>4 — we will be first at N=10+.

## DEVLOG.md

Maintain a running log of daily decisions, test results, equation references, and design rationale. Every code addition should have a corresponding DEVLOG entry.

## Key References

- **B&S**: Baumgarte & Shapiro, *Numerical Relativity* (2010) — primary reference
- **B&S Ch. 11.5 + Box 11.1**: BSSN/CCZ4 evolution system
- **B&S Ch. 5.2.4**: Einstein-Maxwell in 3+1
- **B&S Ch. 9**: Gravitational wave extraction
- **B&S App. H + I**: Trumpet benchmark + binary BH recipe
- **gr-qc/9810065**: BSSN formulation paper
- **gr-qc/9703066**: Brandt-Brugmann puncture method
- **gr-qc/0206072**: Gamma-driver shift condition
- **gr-qc/0511048**: Moving punctures (Campanelli et al.)
- **arXiv:1106.2254**: Original CCZ4 paper — complete evolution equations
- **arXiv:2501.01055**: CCZ4 variants (CCZ4'/CCZ3) — CCZ3 stable to 10⁵ M
- **arXiv:2404.01137**: Improved KO dissipation (CAKO) + constraint damping (CAℋD) + slow-start lapse
- **arXiv:1503.03436**: GRChombo — AMR + CCZ4 reference implementation
- **arXiv:2505.01495**: GRChombo 25-BH cluster — first N>10 full GR simulation
- **arXiv:2312.05438**: AMR refinement strategy comparison (L₂ > L∞ > TE)
- **arXiv:0907.1151**: Alcubierre et al. — Einstein-Maxwell 3+1 form, charged BH initial data
- **arXiv:1903.01036**: Bozzola & Paschalidis — charged puncture initial data (TwoChargedPunctures)
- **arXiv:2104.06978**: Bozzola & Paschalidis — charged binary BH inspiral, λ ≤ 0.3
- **arXiv:1004.1353**: Lousto et al. — N-puncture multigrid solver, 3-BH and 4-BH evolutions
- **arXiv:1003.0859**: Müller & Brügmann — position-dependent η for unequal mass
