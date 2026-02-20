# Lattice Development Log

> **Note:** When adding/removing/renaming files or functions, also update
> `docs/architecture.html` — the living map of the codebase structure.

## 2026-02-19: FAS Multigrid Constraint Solver

Replaced the hyperbolic relaxation solver (`relaxation.c`) with a Full Multigrid
(FMG) solver using Full Approximation Scheme (FAS) V-cycles and Newton-Gauss-Seidel
smoothing. The old solver was O(N^4) — each RK4 step is O(N^3) and O(N) steps
needed for information propagation. The new solver is O(N^3) total and achieves
discretization accuracy in a single FMG pass.

### Algorithm

- **FMG outer loop**: Start on coarsest grid (N_min=16), solve there, then ascend —
  each solved level provides the initial guess (via 4th-order Lagrange prolongation)
  for the next finer level. One V-cycle per level during ascent.
- **FAS V-cycles**: Pre-smooth (4 Newton-GS sweeps), restrict solution + residual to
  coarse grid with tau correction, recursive coarse solve, prolongate correction
  (trilinear) back to fine, post-smooth (4 sweeps).
- **Newton-GS smoother**: 8-color ordering (color = (i%2) + 2*(j%2) + 4*(k%2)),
  GPU-compatible — all points within a color are independent at ±1 distance.
  Newton step: `psi -= residual / J_diag` where `J_diag = -7.5/dx^2 + dS/dpsi`.
- **Cell-centered restriction**: 8-child volume average (2x2x2 fine cells per coarse
  cell). Standard for cell-centered grids where coarse cell centers sit between fine cells.
- **Two prolongation operators**: Trilinear (0.75/0.25) for V-cycle corrections (add
  to fine), 4th-order Lagrange (5-point stencil from AMR `prolong_w`) for FMG initial
  guess (overwrite fine).

### What was replaced

- `relax_workspace_t`, `coupled_workspace_t` structs → `mg_level_t` (unified per-level)
- `relax_rhs()`, `coupled_rhs()` (RK4 wave equation RHS) → `newton_gs_sweep_1field/4field()`
- `relax_rk4_step()`, `coupled_rk4_step()` (RK4 time-stepping) → `mg_vcycle()` + `mg_fmg()`
- All RK4 scratch/accumulator arrays (8 per field pair) → eliminated

### Bug fix: cell-centered restriction

Initial implementation used a vertex-centered "half-weighted" restriction
(0.5*center + 1/12*face_neighbors, Natchu & Matzner Eq. 18). This is correct for
vertex-centered grids where coarse points coincide with fine points. Our grid is
cell-centered (MG_COORD uses +0.5 offset), so coarse cell centers sit BETWEEN 8
fine children. The wrong restriction corrupted the FAS tau correction, causing V-cycle
divergence at 3+ multigrid levels (N=64→32→16). Fixed by replacing with standard
cell-centered 8-child volume average (0.125 * sum of 2x2x2 children).

Ref: HPGMG finite-volume restriction, Zingale Computational Astrophysics MG tutorial.

### Performance

At N=64, L=16 (3 levels: 64→32→16):
- FMG pass: residual 7.1e-7 (instant, <1s)
- 9 post-FMG V-cycles: residual drops to 6.7e-13
- Old solver: ~50k iterations, minutes of wall time

### Public API unchanged

`relaxation_solve()` and `relaxation_solve_coupled()` signatures identical.
`max_iter` now means max post-FMG V-cycles (typically 0-9 needed) instead of
max RK4 iterations (typically 2000-50000 needed).

### Test results

- **29/29 Bowen-York tests**: All pass, convergence order preserved
- **26/26 HiSpID tests**: All pass
- **13/13 AH finder tests**: All pass (Test 7 Boosted BH was the divergence trigger)
- **15/15 Maxwell tests**: All pass
- **Convergence**: Order 5.4 (unchanged)
- **Flat spacetime**: Ham L2 = 5.3e-14 (unchanged)
- **AMR**: 8/8 (unchanged)

### References

- arXiv:0705.1486 (Natchu & Matzner): 4th-order MG for BH initial data
- arXiv:2510.11152: GPU FAS multigrid, 8-color MCGS, 61x speedup
- arXiv:2501.13046 (GRTresna): open-source NR MG constraint solver
- HPGMG finite-volume/source/mg.c: FMG + FAS V-cycle reference
- PETSc SNESFAS: FAS V-cycle reference implementation

---

## 2026-02-18: HiSpID — High-Spin Initial Data (Step 2 of plan1.md)

Implemented quasi-isotropic Kerr metric and a 4-field coupled relaxation solver
for punctures with high spin (chi > ~0.9). Standard Bowen-York assumes conformally
flat geometry (h_ij = delta_ij), which limits effective spin to chi ≤ 0.93 and
radiates excess angular momentum as junk radiation. HiSpID replaces the flat
conformal metric near each BH with the true quasi-isotropic Kerr geometry,
capturing frame-dragging and reducing junk radiation.

### What's new

- **Quasi-isotropic Kerr metric** (`kerr_quasi_isotropic.c`): Computes the Kerr
  spatial metric in semi-isotropic coordinates (Liu, Etienne, Shapiro 2010),
  converts to Cartesian via Jacobian + Rodrigues rotation for arbitrary spin
  direction. Extracts conformal metric (det=1) and conformal factor.
  Ref: arXiv:1001.4077, GRChombo KerrBH.impl.hpp.
- **Gaussian superposition** (`hispid_conformal_metric`): Blends N Kerr metrics
  via h_ij = delta_ij + sum_n w_n * (h_kerr_n - delta_ij), with Gaussian weight
  w_n = exp(-r_n^2 / sigma_n^2), sigma_n = 1.5*M_n. Unit determinant enforced.
  Ref: arXiv:1410.8607 (Ruchlin et al.).
- **4-field coupled relaxation** (`relaxation_solve_coupled`): Extends the 1-field
  solver to 4 field pairs (psi, V^i) with 8 evolved arrays. Hamiltonian source
  includes conformal Ricci scalar R_tilde (computed numerically from h_bg via
  Christoffel symbols). Momentum sources from d_j A_bg^ij. Early stagnation
  detection stops when residual plateaus.
- **CCZ4 conversion for non-flat h_ij** (`set_ccz4_from_hispid`): Two-pass
  conversion — pass 1 sets all fields including h_ij from hispid_conformal_metric
  (NOT delta_ij), pass 2 computes Gamma^i from FD of h_ij via Christoffel symbols.
- **Auto-dispatch** in `set_bowen_york()`: Three-way dispatch — BL (P=S=0),
  standard BY (low spin), HiSpID (chi > 0.9 or `--hispid` flag).
- **CLI**: `--hispid` flag forces HiSpID path even for low spin.

### Test results

- **26/26 HiSpID tests**: QI Kerr metric analytic, falloff, extrinsic curvature,
  Gaussian superposition, zero-spin-matches-BY, moderate spin convergence,
  constraint violation bounded, high-spin evolution, det(h)=1 enforcement.
- **29/29 Bowen-York tests**: Backward compatible, unchanged.
- **`make test`**: All existing tests pass.

### Key numbers (N=24, L=20)

- Zero spin: HiSpID matches BY chi to 5e-8 relative difference
- Chi=0.5: Ham L2 = 8.6e-3, Mom L2 = 1.7e-3 (good for coarse grid)
- Chi=0.9: Evolves 10 steps without NaN, Ham L2 = 1.3e-2

---

## 2026-02-18: Bowen-York Initial Data + Hyperbolic Relaxation Solver

Implemented Step 1 of `docs/plan1.md`: Bowen-York extrinsic curvature for
punctures with linear momentum and spin, plus a hyperbolic relaxation solver
for the Hamiltonian constraint. This is the prerequisite for binary inspiral
(Milestone 5) — BHs can now start with orbital momentum.

### What's new

- **Bowen-York A_ij** (`bowen_york.c`): Analytic formula for the physical
  traceless extrinsic curvature with momentum (1/r^2, B&S Eq. 3.43) and spin
  (1/r^3, B&S Eq. 3.44) terms. Linear superposition for N punctures.
- **Hyperbolic relaxation** (`relaxation.c`): Converts the Hamiltonian
  constraint (nonlinear elliptic PDE) into a damped wave equation in
  pseudo-time, integrated with a standalone mini-RK4 on 2 scalar fields (u, v).
  Ref: Ruter et al. arXiv:1708.07358, NRPyElliptic arXiv:2111.02424.
- **CCZ4 conversion** (`set_ccz4_from_psi`): Maps solved psi + BY A_ij to all
  25 CCZ4 fields: chi=psi^{-4}, A_CCZ4=psi^{-6}*A_phys, lapse=sqrt(chi).
- **`set_bowen_york()` dispatch**: Auto-detects P=S=0 → fast analytic BL path.
  Otherwise runs the relaxation solver.
- **CLI extended**: `--puncture M,x,y,z[,Px,Py,Pz[,Sx,Sy,Sz]]` accepts 4/7/10
  comma-separated values. Fully backward compatible (4 values = BL at rest).
- **`puncture_data_t` struct** in `params.h`: mass, center[3], momentum[3],
  spin[3].

### Test results (29/29 pass)

1. A_ij at known point vs analytic (machine precision)
2. A_ij symmetry: |A_ij - A_ji| = 0
3. Falloff: momentum ratio 4.0 (1/r^2), spin ratio 8.0 (1/r^3)
4. Two-puncture superposition (linearity to 2e-18)
5. CCZ4 conversion: chi = psi^{-4}, A_CCZ4 = psi^{-6} * A_phys
6. Zero momentum: BL exact path, chi matches analytic, A_ij = 0
7. Small momentum (P=0.1): solver converges, Ham bounded, chi > 0
8. Convergence order: N=16 vs N=32 gives ratio 2.39 (order 1.26)
9. Binary orbit: 2 BHs with tangential P, solver converges, evolve 10 steps
   without NaN, constraints bounded

### Solver behavior

The relaxation solver converges quickly (residual drops 2-3 orders in ~500
iterations) then plateaus at the discretization-limited floor (~5e-8 at N=24).
This is expected — the 4th-order FD Laplacian can't drive the residual below
O(dx^4). Higher resolution grids achieve tighter convergence.

---

## 2026-02-18: Berger-Oliger Subcycling for AMR

Implemented Berger-Oliger subcycling so each AMR level advances at its own
CFL-appropriate `dt_L = dt_0 / 2^L`. Previously all levels shared the same
global dt (either coarse-level CFL-violating for fine blocks, or unnecessarily
small for coarse blocks). Now: 1 coarse step + 2 level-1 steps + 4 level-2
steps, each only touching blocks at that level.

### Architecture

- **Per-level packing:** `mesh_build_level_pack()` builds a `meshblock_pack_t`
  containing only leaf blocks at one level. Same packed kernel infrastructure
  (CK45/RK4, Sommerfeld, ghost exchange) operates on each per-level pack.
- **Recursive subcycling:** `subcycle_level()` implements the Berger-Oliger
  recursion: advance level L by `dt_L`, then recursively subcycle finer
  levels with 2 sub-steps at `dt_L/2`.
- **Temporal interpolation:** Coarser blocks save their pre-step state in
  `fields_old[]`. When finer levels need cross-level ghost data at an
  intermediate time, `ghost_fill_from_coarser()` linearly interpolates
  between `fields_old` (old) and `fields` (new) using `frac`.
- **Uniform mesh fast path preserved:** When `max_level == 0`, the original
  single mixed-level pack path is used with zero overhead.
- **No changes to existing packed infrastructure:** Backend ghost exchange,
  RHS kernels, Sommerfeld BCs, and update kernels are unchanged. Per-level
  packs with `n_refined == 0` naturally skip cross-level phases.

### Key design decisions

1. Cross-level ghost fill happens BEFORE building the per-level pack, on the
   mesh blocks directly. This avoids modifying the packed ghost exchange —
   per-level packs only need same-level exchange.
2. `fields_old` is a contiguous backing block (page-aligned), allocated on
   demand via `block_alloc_fields_old()`. Only blocks at levels < max_level
   need it.
3. The temporal interpolation fraction `frac` is computed from `t_start` and
   `dt_coarse` using floor-based rounding to the coarse step boundary.

### Files changed

| File | Changes |
|------|---------|
| `src/amr/block.h` | Added `time`, `fields_old[]`, `fields_old_block` to block_t; new function declarations |
| `src/amr/block.c` | `block_alloc_fields_old`, `block_free_fields_old`, `block_save_old`, `block_time_interp` |
| `src/amr/ghost_exchange.h` | Declared `ghost_fill_from_coarser()` |
| `src/amr/ghost_exchange.c` | Implemented `ghost_fill_from_coarser()` (~80 lines) |
| `src/numerics/rk4.c` | `mesh_build_level_pack`, `step_level`, `subcycle_level`, updated `rk4_step_mesh` dispatch |
| `src/main.c` | Explicit `p.time` tracking in AMR evolution loop |
| `tests/test_subcycle.c` | **New** — 3 validation tests (7 checks) |
| `Makefile` | `test-subcycle` target |

### Test results

```
test_subcycle: 7/7 passed
  Uniform mesh: two runs identical (max diff = 0.0)
  Subcycled flat spacetime: Ham L2 = 3.5e-12 (< 1e-8 threshold)
  Subcycled BH (AMR regrid): Ham L2 = 1.7e-03 (finite, bounded)

All existing tests pass:
  test_flat: PASSED (Ham L2 = 5.29e-14)
  test_convergence: PASSED (order 5.4+)
  test_amr_evolve: 8/8 passed
  test_pack_evolve: 8/8 passed
```

### References

- Berger & Oliger (1984), JCP 53:484 — original subcycling algorithm
- Athena++ `src/mesh/mesh.cpp` `Mesh::Step()` — subcycle loop reference
- GRChombo `GRAMRLevel::advance()` + Chombo `AMR::timeStep()` — coarse-fine pattern
- Chombo `AMRLevel::m_old_data` / `m_new_data` — temporal interpolation storage

## 2026-02-18: GPU Batch Kernels — Commit 2: Device-Side Ghost Exchange

Replaced the Commit 1 ghost exchange fallback (unpack → CPU 5-phase exchange →
repack, 3 full data copies per RK stage) with `backend_ghost_exchange_packed()`
operating directly on pack buffers. Zero data copies on CPU; GPU backend syncs
to host, runs 5-phase exchange, syncs back (eliminates unpack/repack overhead).

### Architecture

- **5-phase multilevel exchange on pack buffers:**
  - Phase 0+1: Same-level 26-neighbor copy between `pack->data` slices
  - Phase 2: 4th-order Lagrange restriction (fine → `pack->coarse_data`)
  - Phase 3: Fill coarse_buf ghosts from siblings + cross-level copy from coarser neighbor
  - Phase 3.5: Boundary quadratic extrapolation (dimension sweep x→y→z)
  - Phase 4: 5×5×5 Lagrange prolongation (coarse_data → fine ghost zones)
- **New metadata:** `nblevel_table[n_blocks * 27]` — flattened `nblevel[3][3][3]`
  per block, needed by Phases 3.5 (boundary detection) and 4 (skip same-level-filled ghosts).
- **CPU backend:** Direct operation on host memory, `omp parallel for`.
- **GPU backend:** `omp target update from/to` sync brackets around host-side
  5-phase exchange. True device-side kernels deferred to future optimization.
- **rk4.c:** Removed `packed_ghost_exchange_fallback()`. All packed steppers
  now call `backend_ghost_exchange_packed(pack)` directly.

### Files changed

| File | Changes |
|------|---------|
| `src/amr/meshblock_pack.h` | Added `nblevel_table` field to struct |
| `src/amr/meshblock_pack.c` | Allocate/populate/free `nblevel_table` |
| `src/backend/backend.h` | Updated `backend_ghost_exchange_packed` comment |
| `src/backend/backend_cpu.c` | Full 5-phase ghost exchange (~550 lines) |
| `src/backend/backend_gpu.c` | GPU version: sync + 5-phase exchange + sync back |
| `src/numerics/rk4.c` | Removed fallback, use `backend_ghost_exchange_packed` |
| `docs/architecture.html` | Updated backend and rk4 module docs |

### Test results

```
test_pack_evolve: 8/8 passed
  Packed vs per-block max |diff| = 7.14e-16 (< 1e-12 threshold)
  Multilevel AMR: regridding works, Ham L2 finite and bounded
  Flat spacetime: Ham L2 = 4.48e-14 (< 1e-10 threshold)

All other tests pass:
  test_flat: PASSED (Ham L2 = 5.29e-14)
  test_convergence: PASSED (order 5.43, 5.47)
  test_amr_evolve: 8/8 PASSED
  test_amr_refine: 72/72 PASSED
  test_amr_ghost: 16/16 PASSED
  test_amr_prolong: 15/15 PASSED
```

## 2026-02-17: GPU Batch Kernels — Commit 1 Complete

Implemented AthenaK-style packed batch kernels for the AMR mesh stepper.
All leaf blocks are packed into a single contiguous `meshblock_pack_t` buffer;
one kernel launch per operation (RHS, Sommerfeld, CK45/RK4 update) covers all
blocks. Replaces the old per-block stepper (320 kernel launches/step for 64
blocks × 5 CK45 stages) with ~15 launches/step.

### Architecture

- **meshblock_pack_t** extended with per-block metadata (origins, dx, boundary
  flags, levels, neighbor_table), coarse_buf data for multilevel ghost exchange,
  and an optional accum buffer for classic RK4.
- **Batched backend API**: `backend_compute_rhs_packed`, `backend_sommerfeld_packed`,
  `backend_update_ck45_packed`, plus classic RK4 ops (`copy`, `accum_add`, `axpy`,
  `apply_accum`). CPU backend uses `omp parallel for`; GPU backend uses
  `omp target teams distribute parallel for collapse(4)`.
- **Packed mesh steppers**: `ck45_step_mesh_packed` and `classic_rk4_step_mesh_packed`
  in `rk4.c`. Old per-block steppers kept as `rk4_step_mesh_perblock` for
  debug/comparison.
- **Ghost exchange (Commit 1 fallback)**: unpack → CPU 5-phase multilevel
  exchange → repack. Correct but requires full data round-trip. Commit 2 will
  replace with device-side ghost exchange kernels.
- **Sommerfeld helpers** (`asymptotic_value`, `boundary_d1`) made non-static with
  `#pragma omp declare target` for GPU compilation.

### Files changed

| File | Changes |
|------|---------|
| `src/amr/meshblock_pack.h` | Extended struct: metadata arrays, coarse_data, neighbor tables |
| `src/amr/meshblock_pack.c` | New functions: `load_meta`, `build_neighbors`, `load/store_coarse` |
| `src/backend/backend.h` | 11 new packed backend function declarations |
| `src/backend/backend_cpu.c` | CPU implementations of all packed functions |
| `src/backend/backend_gpu.c` | GPU implementations with `omp target` kernels |
| `src/boundary/sommerfeld.h/c` | `declare target` on helpers, made non-static |
| `src/numerics/rk4.h/c` | Packed steppers, `mesh_build_leaf_pack`, dispatch |
| `tests/test_pack_evolve.c` | New: packed vs per-block, multilevel, flat stability |
| `Makefile` | `test-pack-evolve` target |
| `docs/gpu_batch_kernels.html` | New: interactive GPU optimization explainer |

### Test results

```
test_pack_evolve: 8/8 passed
  Packed vs per-block max |diff| = 7.14e-16 (< 1e-12 threshold)
  Multilevel AMR: regridding works, Ham L2 finite and bounded
  Flat spacetime: Ham L2 = 4.48e-14 (< 1e-10 threshold)

make test: all existing tests still pass
  Flat stability, convergence order 5.4, AMR evolve 8/8
```

### Next: Commit 2

Replace `packed_ghost_exchange_fallback` with `backend_ghost_exchange_packed` —
5 device-side kernels for same-level exchange, restrict, coarse ghost fill,
boundary extrapolation, and prolongation. Eliminates all CPU-GPU sync during
the RK step.

## 2026-02-17: AMR Integration into Main Evolution Loop — Complete

Wired AMR into `src/main.c` so `./lattice --amr ...` runs a real BH evolution
on a multi-block mesh with dynamic regridding. All 8/8 integration tests pass.

### Changes

- **`src/main.c`**: Added `--amr`, `--N_root`, `--N_block`, `--max_level`,
  `--chi_refine`, `--chi_coarsen`, `--regrid_every` CLI args. Conditional AMR
  path creates `mesh_t`, sets initial data on all leaf blocks, evolves with
  `rk4_step_mesh`, regrids periodically, prints per-step diagnostics with
  block count.
- **`src/diagnostics/constraints.c/h`**: Added `mesh_constraint_l2()` and
  `mesh_momentum_l2()` — accumulate L2 norms over all leaf blocks.
- **`tests/test_amr_evolve.c`**: New integration test with 3 tests:
  1. Uniform mesh vs single-grid BH (ratio 0.99 — within 2x) ✓
  2. Dynamic regridding around BH (8→64 blocks, Ham L2=2.5e-3) ✓
  3. Flat spacetime with regridding (no refinement, Ham L2=2.8e-14) ✓
- **`Makefile`**: Added `test-amr-evolve` target, added to `test:` deps.

### Bug fix: boundary ghost NaN after regridding

After regridding, fine-level child blocks at domain boundaries had zero-valued
ghost zones (chi=0, h_ij=0, lapse=0). Root cause: `prolongate_from_own_coarse_buf`
in Phase 4 of ghost exchange skipped boundary directions with
`if (nlev < 0) continue;`. The 4th-order FD stencils at interior points near
the boundary would read these zeros, causing division by chi=0 → NaN.

**Fix**: Removed the `if (nlev < 0) continue;` guard in
`src/amr/ghost_exchange.c:prolongate_from_own_coarse_buf()`. Phase 3.5 already
fills the coarse_buf boundary ghosts by quadratic extrapolation, so the 4th-order
Lagrange prolongation stencil has valid data. This gives 4th-order accurate
boundary ghost fills — better than GRChombo's 1st-order radial extrapolation
approach (ref: `BoundaryConditions.cpp:fill_extrapolating_cell()`).

### GRChombo cross-reference

GRChombo has TWO boundary operations where we had one:
- `fill_rhs_boundaries` — Sommerfeld on RHS (equivalent to our `apply_sommerfeld_block`)
- `fill_solution_boundaries` → `fill_extrapolating_cell` — fills boundary ghost
  FIELD VALUES by linear radial extrapolation (we were missing this entirely)

GRChombo calls `fillBdyGhosts(soln)` before every RHS eval, after every update
step, and after every regrid (`GRAMRLevel.cpp` lines 158, 183, 346, 942, 963).
Our fix achieves the same effect through the prolongation pathway, which is
architecturally cleaner for block-structured AMR.

### Files touched

| File | Change |
|------|--------|
| `src/main.c` | AMR CLI args + conditional mesh evolution loop |
| `src/diagnostics/constraints.c` | `mesh_constraint_l2()`, `mesh_momentum_l2()` |
| `src/diagnostics/constraints.h` | Declare mesh-level functions |
| `tests/test_amr_evolve.c` | New AMR evolution integration test |
| `src/amr/ghost_exchange.c` | Fix: allow boundary prolongation in Phase 4 |
| `Makefile` | `test-amr-evolve` target |

## 2026-02-17: Head-On Binary Collision — Milestone 4 Complete

### Setup

Two equal-mass Brill-Lindquist punctures (m1=m2=0.5, d=10M separation on
z-axis), N=128 uniform grid, L=64, dx=0.5, CK45 integrator. No new physics
code needed — all existing infrastructure (CCZ4 RHS, Sommerfeld BCs, puncture
initial data, constraint diagnostics) worked out of the box for the binary case.

### Results

- **Merger at t~8M**: BH separation (tracked via z-axis lapse minima) dropped
  to zero. Two distinct lapse dips merged into a single minimum.
- **Constraints peaked at ~1e-2**: well under the 1.0 failure threshold.
  Hamiltonian and momentum L2 norms bounded throughout the evolution.
- **Post-merger lapse settled at ~0.91**: remnant BH with total mass M~1.0.

### Test rewrite

Rewrote `tests/test_head_on.c` with live per-step diagnostic output, printing
a table of: step, time/M, lapse_min, z_min (location of minimum), sep/M
(BH separation), #dip (number of lapse minima on z-axis), Ham L2, Mom L2.
Output saved to `tests/test_head_on_output.txt` for reference.

### New helper

`bh_separation()` — scans z-axis lapse profile to find two distinct local
minima, returns their spatial separation. Uses the z-axis slice (x=y=0) and
identifies dips by sign changes in the lapse gradient. Returns 0 when only
one minimum is found (post-merger).

### References

- gr-qc/0606079: Sperhake 2006, "Binary black-hole evolutions of excision and
  puncture data" — head-on collision benchmarks

---

## 2026-02-17: Fix Phase 3.5 dimension-sweep loop ranges (Test 6 passes)

### Overview

Fixed the multi-level ghost exchange bug that caused Test 6 to fail (max_err=0.386, threshold=9.8e-5). Root cause: Phase 3.5's boundary extrapolation loops used interior-only ranges in non-ghost directions, leaving coarse_buf ghost cells unfilled at boundary-edge intersections.

### Root cause analysis

- Phase 3 (`copy_from_coarse_grid`) fills coarse_buf ghost zones from coarse neighbors, but `ghost_range(0, ...)` returns the interior range `[ghost, ghost+N)` for non-ghost directions. So the z+ ghost zone only gets filled for interior x and y cells.
- Phase 3.5 (`fill_coarse_buf_boundary`) extrapolates domain-boundary ghost cells, but only processes boundary faces — the X face loop used `j=gh..gh+N, k=gh..gh+N` (interior only), the Y face loop used `k=gh..gh+N` (interior only).
- Cells at the intersection of a domain-boundary ghost zone (e.g., x-) and a non-boundary ghost zone (e.g., z+) were never filled by either phase. Prolongation read stale data → O(1) errors.

### Fix

Widened Phase 3.5's face loop ranges to cover the FULL array (0 to Ntotal) in non-ghost directions, matching AthenaK's `BCHelper` dimension-by-dimension sweep pattern (`src/bvals/physics/z4c_bcs.cpp`). The sequential face ordering (X→Y→Z) ensures edges and corners are filled automatically: each later sweep reads ghost data written by earlier sweeps.

Changes:
- X faces: `j` and `k` ranges changed from `[gh, gh+N)` to `[0, Nt)`
- Y faces: `k` range changed from `[gh, gh+N)` to `[0, Nt)` (i was already `[0, Nt)`)
- Z faces: unchanged (already used `[0, Nt)` for both i and j)

### Result

- Test 6: max_err dropped from 0.386 to 4.1e-6 (threshold 9.8e-5) → PASS
- All 72 AMR tests pass, all other tests unchanged
- Added `docs/phase3_ghost_exchange.html` — visual explainer of the coarse-buffer ghost exchange and this bug

### References

- AthenaK `src/bvals/physics/z4c_bcs.cpp` BCHelper — dimension-by-dimension sweep with full-width loops
- AthenaK NR paper (arXiv:2409.10383) Section 3.1

---

## 2026-02-16: AMR Stage 4.1 — Coarse-Buffer Architecture + 4th-Order Restriction (WIP)

### Overview

Replaced the parent-based inter-level ghost fill (Stage 4) with AthenaK's
coarse-buffer architecture. Each leaf block at level > 0 now carries its own
`coarse_buf` grid at half resolution (N_c = N_block/2, dx_c = 2*dx_fine).
All inter-level operations become block-local -- no cross-block writes, no
parent grid dependency for ghost fill, GPU-friendly.

Upgraded restriction from 2nd-order (8-cell average) to 4th-order (4-point
symmetric Lagrange stencil) to match the prolongation order. ExaHyPE
(arXiv:2504.15814) showed that mismatched restriction/prolongation orders
cause Hamiltonian constraint violations at AMR boundaries.

### Reference code consulted

- **AthenaK `src/bvals/`**: Coarse-buffer architecture. Each MeshBlock
  carries a `coarse_buf` (same as `pmb->pmr->pcoarsebuf` in Athena++).
  Ghost fill is entirely block-local: restrict own fine data into coarse_buf
  interior, exchange coarse_buf ghosts with neighbors, prolongate from own
  coarse_buf into fine ghost zones. No cross-block memory writes.
- **AthenaK `ApplyPhysicalBoundariesOnCoarseLevel`**: Applies physical
  boundary conditions on coarse_buf before prolongation. We substitute
  quadratic extrapolation as a general-purpose alternative (Phase 3.5).
- **ExaHyPE (arXiv:2504.15814)**: Upgrading restriction to match prolongation
  order eliminates Hamiltonian constraint violations at AMR boundaries.
- **GRChombo `CoarseAverage`** (Chombo library): 2nd-order volume-weighted
  averaging -- the baseline we improve upon.
- **Fornberg, SIAM Review 40 (1998)**: FD weight generation algorithm used to
  derive 4th-order cell-average restriction weights.

### Key design decisions

1. **Coarse-buffer architecture.** Every leaf block at level > 0 allocates
   `coarse_buf` (a fields-only `grid_t` with N = N_block/2, ghost = GHOST_WIDTH,
   dx = 2*dx_fine). The coarse_buf covers the same physical domain as the fine
   block but at parent resolution. All inter-level communication goes through
   the local coarse_buf -- the parent grid is no longer needed for ghost fill.

2. **4th-order cell-average restriction weights.**
   ```
   w = { 1/48, 23/48, 23/48, 1/48 }
   ```
   1D symmetric 4-point Lagrange stencil, exact for polynomial degree <= 3.
   Derived by integrating Lagrange basis polynomials over the coarse cell:
   `w_j = (1/dx_c) * integral_{-dx_c/2}^{+dx_c/2} L_j(x) dx`, where L_j
   are 4-point Lagrange polynomials through fine cell centers at
   {-3d/2, -d/2, +d/2, +3d/2} (d = dx_fine). 3D: tensor product gives
   4^3 = 64 fine cells per coarse cell. Falls back to 2nd-order (8-cell
   average) at grid boundaries where the stencil extends outside the
   interior region.

3. **5-phase ghost exchange in `ghost_exchange_multilevel()`:**
   - Phase 0+1: Same-level exchange per level (coarsest first)
   - Phase 2: Restrict fine interior -> own coarse_buf (4th-order, block-local)
   - Phase 3: Fill coarse_buf ghosts from sibling coarse_bufs + coarser neighbors
   - Phase 3.5: Fill coarse_buf boundary ghost cells by quadratic extrapolation
     (like AthenaK's `ApplyPhysicalBoundariesOnCoarseLevel`)
   - Phase 4: Prolongate own coarse_buf -> fine ghost zones (5-point Lagrange,
     skip directions with same-level neighbors)

   Phases 2-4 are embarrassingly parallel per block. No phase writes to
   another block's memory.

4. **`coarse_buf_alloc()` vs `grid_alloc()`.** The coarse_buf uses exact N
   (no padding to multiple of 16) since N_c may be small (e.g. N_c=8 for
   N_block=16). Only allocates `fields_block`; rhs/scratch/accum are NULL
   since coarse_buf is never time-evolved.

### Bug fixes

1. **`mesh_rebuild_neighbors()` cross-level edge/corner lookup.** The original
   code divided the block's own coordinates by 2 then added the neighbor
   offset. This fails for edge/corner directions at coarse-fine boundaries
   because the fine-level neighbor coords may be negative (e.g. coord 0 with
   offset -1 = -1). Fix: compute the neighbor's fine-level coordinates first
   (block coords + offset), then map to coarser levels using floor division.
   Floor div for negative x: `~(~x >> shift)`. For non-negative: `x >> shift`.

2. **`restrict_cell()` bounds check.** Was checking stencil bounds against
   `[0, Ntotal)` (full array including ghost zones). Changed to `[ghost,
   ghost + N)` (interior only). At coarse-fine boundaries, fine ghost zones
   may not yet be filled when restriction runs, so the 4th-order stencil
   must only read from valid interior data. Falls back to 2nd-order where
   stencil exits interior.

### New files

- `src/amr/block.c`: `coarse_buf_alloc()` -- fields-only `grid_t` with exact N
  (no padding), no RK scratch arrays. Page-aligned allocation for potential GPU use.

### New functions

- `restrict_to_coarse_buf()` (`restriction.h/c`): Restrict fine block interior
  into block's own coarse_buf interior. 4th-order tensor product with 2nd-order
  fallback. Block-local, no cross-block access.
- `fill_coarse_buf_boundary()` (`ghost_exchange.c`): Fill domain-boundary ghost
  cells of coarse_buf by quadratic extrapolation (3-point Lagrange, exact for
  degree <= 2). Processes faces sequentially (x, y, z) so edge/corner cells
  are filled correctly by later passes.
- `fill_coarse_buf_ghosts()` (`ghost_exchange.c`): Fill coarse_buf ghost zones
  for all fine leaf blocks. Same-level neighbors: exchange coarse_buf <-> coarse_buf
  via `exchange_grid_pair()`. Coarser neighbors: copy from coarse neighbor's main
  grid via `copy_from_coarse_grid()` using index-offset mapping.
- `copy_from_coarse_grid()` (`ghost_exchange.c`): Copy from a coarse neighbor's
  main grid into a block's coarse_buf ghost zone. Both grids share dx; uses
  integer offset mapping from origin difference.
- `exchange_grid_pair()` (`ghost_exchange.c`): Exchange between two grids of the
  same dimensions (N, ghost). Used for coarse_buf <-> coarse_buf exchange.
- `prolongate_from_own_coarse_buf()` (`ghost_exchange.c`): 5-point Lagrange
  interpolation from block's own coarse_buf into fine ghost zones. Skips cells
  where same-level neighbor already filled (Phase 1). Maps fine index to
  coarse_buf continuous index: `ci = (fi - ghost_f + 0.5) / 2.0 + ghost_c - 0.5`.

### Modified files

- `src/amr/restriction.h` -- Added `RESTRICT_STENCIL=4`, `restrict_w[]` extern,
  `restrict_to_coarse_buf()` declaration, forward-declared `struct block_s`.
- `src/amr/restriction.c` -- 4th-order tensor product restriction. Weights
  derived from Lagrange basis integrals. `restrict_cell()` with interior-only
  bounds check and 2nd-order fallback.
- `src/amr/block.h` -- Added `coarse_buf` field (`grid_t*`) to `block_t`.
- `src/amr/block.c` -- `block_alloc()` now allocates `coarse_buf` for level > 0
  blocks. `block_free()` frees coarse_buf. Added `coarse_buf_alloc()` static.
- `src/amr/mesh.c` -- `mesh_rebuild_neighbors()` fixed cross-level lookup using
  floor division on neighbor's fine-level coordinates.
- `src/amr/ghost_exchange.h/c` -- Rewrote `ghost_exchange_multilevel()` for
  5-phase coarse-buffer architecture. Added 6 new static/public functions.

### Known issue (RESOLVED 2026-02-17)

**Test 6 (multi-level ghost exchange) now passes.** See 2026-02-17 entry. Root cause was
Phase 3.5 loop ranges using interior-only bounds in non-ghost directions, leaving
boundary-edge intersection cells unfilled. Fix: widen to full array ranges matching
AthenaK's BCHelper dimension sweep.

### References

- AthenaK coarse-buffer architecture (`src/bvals/`)
- ExaHyPE (arXiv:2504.15814) -- upgrading restriction to match prolongation
  order eliminates Hamiltonian violations at AMR boundaries
- GRChombo CoarseAverage -- 2nd-order baseline we improve upon
- Fornberg, SIAM Review 40 (1998) -- FD weight generation algorithm

---

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

## 2026-02-11: GPU Strategy — OpenMP Target with GCC 15

### Decision: No native CUDA/HIP/Metal backends needed

The existing `backend_gpu.c` uses OpenMP target offloading, which already works
for everything except the per-thread stack overflow (see GPU investigation above).
GCC 15 adds `GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE` to control native GPU
thread stack at runtime, which fixes the root cause. No need to write
platform-specific GPU code.

Deleted references to Metal/CUDA/HIP backends from CLAUDE.md. The architecture
is now: CPU (OpenMP threads) + GPU (OpenMP target), both using the same C
physics kernels. GPU_ARCH flag in Makefile selects NVIDIA (`nvptx-none`) or
AMD (`amdgcn-amdhsa`) offload target.

### Performance benchmarks (M4 CPU, CK45, flat spacetime)

| Grid (N) | Wall time/step | ns/point | CPU cores used |
|-----------|---------------|----------|----------------|
| 32 | 0.082s | 500 | 4.3 |
| 64 | 0.62s | 473 | 5.4 |
| 128 | 4.53s | 431 | 6.0 |

Effective throughput: ~23 GFLOPS FP64 (5.4% of M4's 424 GFLOPS peak).
Scaling is linear in N^3 as expected.

### FP64 GPU comparison: why the P40 was the wrong test card

The Tesla P40 (GP102) has a 1:32 FP64:FP32 ratio = 367 GFLOPS FP64 peak.
The M4 CPU has 424 GFLOPS FP64 peak. The P40 is actually *slower* than the
laptop for FP64 work, even ignoring the stack overflow. HPC-class GPUs have
1:2 FP64:FP32 ratio:

| Hardware | FP64 GFLOPS | Memory BW | Est. speedup vs M4 |
|----------|-------------|-----------|---------------------|
| M4 CPU (measured) | 424 | 120 GB/s | 1× |
| Tesla P40 | 367 | 346 GB/s | 0.5-1.5× (not worth it) |
| V100 | 7,000 | 900 GB/s | 15-30× |
| A100 | 9,700 | 2,039 GB/s | 20-50× |
| H100 | 34,000 | 3,350 GB/s | 50-100× |

### Updated Makefile

- `BACKEND=gpu` now passes `-foffload=$(GPU_ARCH)` with `GPU_ARCH` defaulting
  to `nvptx-none`. Set `GPU_ARCH=amdgcn-amdhsa` for AMD GPUs.
- Added `-fcf-protection=none -fno-stack-protector` flags (required for nvptx).
- Header comment documents the required `GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE`
  env var.

## 2026-02-12: Single BH Test Passing — Milestone 3

### Problem

`test_single_bh` was configured for N=256 (dx=0.25), requiring ~10.3 GB with
CK45. On the 16 GB M4, allocation hung silently — no output at all.

### Fix

Reduced to N=128 (dx=0.5, L=64, T=10M, 80 steps). Memory ~1.3 GB, runs in
~40 minutes on M4.

### Results

```
t=0.00M   alpha_min = 0.2154   Ham_L2 = 7.73e-4   (pre-collapsed initial data)
t=1.25M   alpha_min = 0.2529   Ham_L2 = 1.45e-3   (lapse rising from initial)
t=5.00M   alpha_min = 0.5735   Ham_L2 = 4.15e-3   (overshoot — low-res artifact)
t=7.50M   alpha_min = 0.4350   Ham_L2 = 5.73e-3   (turning around)
t=10.0M   alpha_min = 0.3690   Ham_L2 = 6.45e-3   (approaching trumpet)
```

All three pass criteria met:
- Lapse collapsed: 0.369 < 0.5
- Fields finite: no NaN/Inf
- Constraints bounded: peak 6.4e-3 (well under 1.0)

### Physics validation

The lapse overshoot at t~5M is a low-resolution artifact (dx=0.5 means the BH
radius r=M/2=0.5 is only ~1 grid point). At higher resolution the overshoot
shrinks and the trumpet forms faster. The final min lapse of 0.369 is within 2%
of the analytic stationary 1+log trumpet value of **alpha = 0.376** at the
horizon (Hannam et al., arXiv:0804.0628, Eq. 30). The lapse is still settling
at T=10M — longer evolution would bring it closer to the stationary value.

The pre-collapsed initial lapse alpha = sqrt(chi) = psi^{-2} starts below the
trumpet value near the puncture. The 1+log gauge then drives the lapse upward
toward the stationary solution, overshooting at coarse resolution before damping
back down. This is standard moving-puncture behavior documented in the literature
(Campanelli et al., gr-qc/0511048; Baker et al., gr-qc/0511103).

### References

- arXiv:0804.0628: Hannam et al., "Wormholes and trumpets: Schwarzschild
  spacetime for the moving-puncture generation" — analytic trumpet solution,
  alpha(R=2M) = 0.376 for 1+log slicing
- gr-qc/0511048: Campanelli et al., "Accurate evolutions of orbiting BH
  binaries without excision" — original moving puncture method
- arXiv:1010.5723: Dennison & Baumgarte, "Trumpet slices of
  Schwarzschild-Tangherlini" — trumpet geometry in higher dimensions

### What's next

Milestone 3 passing. Remaining Phase 1 work:
1. Gauge wave test + 3-resolution convergence verification (prove 4th-order)
2. Momentum constraints diagnostic
3. Head-on binary collision (Milestone 4)
4. Psi4 gravitational wave extraction
5. Binary inspiral with Bowen-York momentum (Milestone 5)

## 2026-02-12: Convergence Test — 5th-Order Verified

### Test design

Single BH (M=1, Brill-Lindquist) at 3 resolutions (N=32, 64, 128) with L=64.
Evolve to T=2M with CK45. Measure Hamiltonian constraint L2 in annular region
5M < r < 25M, away from the puncture singularity and boundary.

The constraint is zero for exact solutions, so measured L2 is pure truncation
error. For 4th-order FD + 4th-order RK4, expect error ~ dx^4 (ratio of 16
per resolution doubling).

### Results

```
N= 32  dx=2.0  Ham_L2 = 3.045e-04
N= 64  dx=1.0  Ham_L2 = 7.077e-06
N=128  dx=0.5  Ham_L2 = 1.595e-07

N=32 -> N=64:   ratio = 43.03  order = 5.43
N=64 -> N=128:  ratio = 44.36  order = 5.47
```

**Measured convergence order: 5.4** — exceeds the 4th-order minimum. The extra
order is expected: 4th-order centered FD stencils achieve 5th-order accuracy on
smooth data due to cancellation of odd-order error terms. Both refinement steps
agree closely (5.43 vs 5.47), confirming we are in the asymptotic regime.

### What this proves

- Finite difference stencils (FD_D1, FD_D2, FD_D2_mixed) are correctly 4th-order
- RK4/CK45 time integration is correctly 4th-order
- CCZ4 RHS equations produce convergent evolution
- Christoffel symbols, Ricci tensor, gauge conditions all implemented correctly
- KO dissipation is not degrading convergence order

## 2026-02-16: AMR Stage 2 — Ghost Exchange + Multi-Block Evolution

### Goal

Decompose domain into N_root^3 blocks at level 0. Fill ghost zones by
direct copy from all 26 neighbors (faces, edges, corners) in a single
pass. All physics unchanged — CCZ4 RHS and Sommerfeld BCs work on
per-block grids with correct global coordinates.

### Reference code consulted

- **Athena++** `src/bvals/cc/bvals_cc.cpp` — `LoadBoundaryBufferSameLevel()`,
  `SetBoundarySameLevel()`: unified `(ox1,ox2,ox3)` index calculation for
  all 26 neighbor types. Sender extracts NGHOST-wide slab from interior,
  receiver fills ghost zone at corresponding face/edge/corner.
- **Athena++** `src/utils/buffer_utils.cpp` — `PackData/UnpackData`: loop
  order n→k→j→i with unit-stride x inner loop and `#pragma omp simd`.
- **AthenaK** `src/bvals/bvals_cc.cpp` — GPU pattern: single fused kernel
  for all blocks×neighbors×fields, 3-level parallelism (teams/threads/vectors).
- **Athena++** buffer size formula: `((ox==0) ? N : NGHOST)` per direction.

### Implementation

**New files (2):**
- `src/amr/ghost_exchange.h/c` — 26-neighbor ghost exchange. Uniform index
  computation via `ghost_range()` for all neighbor types. Uses `memcpy` for
  contiguous x-strips (unit-stride). Single pass over all blocks and neighbors.

**Modified files (5):**
- `src/boundary/sommerfeld.h/c` — added `apply_sommerfeld_block()`: only applies
  to ghost points adjacent to domain boundaries (on_boundary[face]==1). Inter-block
  ghost zones left to ghost exchange. Uses BLOCK_COORD for global coordinates.
- `src/numerics/rk4.h/c` — added `rk4_step_mesh()`: CK45 and classic RK4 for
  multi-block meshes. Ghost exchange before each RHS evaluation, block-aware
  Sommerfeld on domain boundaries, enforce algebraic on all blocks after step.
- `src/initial_data/puncture.h/c` — added `set_brill_lindquist_global()`:
  uses explicit origin parameter for correct global coordinates in multi-block.
- `src/amr/mesh.h` — changed anonymous struct to `struct mesh_s` tag for
  forward declaration support.
- `Makefile` — added ghost_exchange.c to AMR_SRC, test-amr-ghost target.

### Key design: unified ghost range computation

For each direction d with neighbor offset o:
- `o == -1`: dst = `[0, ghost)`, src = `[N, ghost+N)` (neighbor's high interior)
- `o ==  0`: dst = `[ghost, ghost+N)`, src = `[ghost, ghost+N)` (same range)
- `o == +1`: dst = `[ghost+N, Ntotal)`, src = `[ghost, 2*ghost)` (neighbor's low interior)

This handles all 26 cases uniformly: faces=slabs, edges=columns, corners=cubes.
Matches Athena++ `LoadBoundaryBufferSameLevel`/`SetBoundarySameLevel` pattern.

### Test results (13/13 passed)

```
test_amr_ghost:
  Ghost exchange polynomial f=x+2y+3z: 0 error    PASS (31232 points checked)
  Corner block: 3F+3E+1C neighbors                 PASS
  Multi-block flat (2x2x2): Ham L2 = 4.5e-14       PASS (ratio 1.01x single-grid)
  Ghost values match BL single-grid: 0 error        PASS (31232 points × 25 fields)
  Multi-block single BH: Ham L2 bounded             PASS
```

Key result: **multi-block flat spacetime matches single-grid to 1%** (4.49e-14 vs
4.44e-14 Ham L2). Ghost exchange values match to exact roundoff (0 error) for both
polynomial test data and Brill-Lindquist initial data.

All existing tests pass with no regressions (flat, convergence, amr-mesh).

### What's next

Stage 3: prolongation + restriction + noise reduction (CAKO, CAHD, SSL, per-field sigma).

## 2026-02-16: AMR Stage 1 — Data Structures Foundation

### What was built

Implemented the AMR data structure layer: `block_t`, `mesh_t`, `meshblock_pack_t`,
and Morton encoding. These wrap the existing `grid_t` without modifying any physics
code. The architecture follows proven patterns from Athena++ (Stone et al. 2020)
and AthenaK (Grete et al. 2024), adapted to our C17/SoA codebase.

### Reference code consulted

Heavily referenced Athena++ and AthenaK before writing:
- `LogicalLocation` struct: `(lx1, lx2, lx3, level)` tree addressing from
  Athena++ `athena.hpp`. Child coordinates = `parent*2 + {0,1}`.
- `nblevel[3][3][3]`: per-block neighbor level table from Athena++
  `bvals_base.cpp`. Indexed `[oz+1][oy+1][ox+1]`, self at `[1][1][1]`.
- `SearchAndSetNeighbors`: Athena++ `bvals_base.cpp` — 26-neighbor finding
  via coordinate offset + bounds check (simplified for uniform level-0).
- `MeshBlockPack`: AthenaK `meshblock_pack.hpp` — contiguous GPU buffer
  with block as a batch dimension. Our layout: `data[f * n_blocks * npts + b * npts + idx]`.
- `MeshBlockTree`: Athena++ `meshblock_tree.cpp` — oct-tree with child
  indexing `n = ox1 + (ox2<<1) + (ox3<<2)`.
- GRChombo `ChiTaggingCriterion.hpp` — chi-gradient formula for Stage 4.

### Files created

- `src/amr/morton.h` — header-only Morton Z-order encoding (bit-interleave)
- `src/amr/block.h/c` — block_t with LogicalLocation, nblevel, 26 neighbors
- `src/amr/mesh.h/c` — mesh_t creation, Morton-sorted blocks, neighbor finding
- `src/amr/meshblock_pack.h/c` — GPU pack with page-aligned contiguous buffers
- `tests/test_amr_mesh.c` — 33 tests

### Files modified

- `src/core/params.h` — added `amr_params_t` to `sim_params_t`
- `Makefile` — added `AMR_SRC`, `test-amr-mesh` target
- `docs/architecture.html` — updated AMR module from "Planned" to "Stage 1 Complete"
- `plan1.md` — updated progress tracking

### Test results

33/33 AMR tests pass:
- Morton: encode/decode round-trip, Z-ordering, child/parent operations
- Topology: 2x2x2 mesh with correct neighbors (7 for corner, 19 boundaries),
  nblevel tables, boundary flags
- Evolution: 1-block mesh flat spacetime Ham L2 = 1.04e-14 (threshold 1e-10)
- MeshBlockPack: exact round-trip (0 error) for 1-block and 8-block packs

Existing tests (flat spacetime, convergence) pass with no regressions.

### What's next

Stage 2: multi-block uniform mesh with 26-neighbor ghost exchange. Decompose
domain into N_root^3 blocks, fill ghost zones by direct copy from neighbors,
and verify multi-block results match single-grid to roundoff.

## 2026-02-16: AMR Stage 1 — block_t + mesh_t + MeshBlockPack Foundation

### Goal

Define AMR data structures. A single-block mesh through the mesh API must
produce identical output to the current grid_t. MeshBlockPack contiguous
GPU-ready buffers from Stage 1.

### Reference code consulted

Before writing any code, studied the production AMR implementations:

- **Athena++** `src/mesh/meshblock.hpp`, `meshblock_tree.cpp`, `bvals_base.cpp`
  — LogicalLocation addressing, nblevel[3][3][3] neighbor level table,
  SearchAndSetNeighbors algorithm, Z-order tree traversal for GID assignment
- **AthenaK** `src/mesh/meshblock_pack.hpp`, `mesh.hpp`, `nghbr_index.hpp`
  — MeshBlockPack GPU batching layout, 56-neighbor table for refined blocks
- **GRChombo** `Source/TaggingCriteria/ChiTaggingCriterion.hpp`,
  `Source/GRChomboCore/GRAMRLevel.hpp` — chi-gradient formula, level lifecycle

Key patterns adopted from Athena++:
1. `logical_location_t (lx1, lx2, lx3, level)` — universal block address
2. `nblevel[3][3][3]` — neighbor level lookup (indexed `[oz+1][oy+1][ox+1]`)
3. Child octant indexing: `n = ox1 + (ox2<<1) + (ox3<<2)`
4. Morton Z-order sorting of blocks for cache-friendly traversal
5. 26-neighbor offset table (faces, edges, corners)

### Implementation

**New files (7):**
- `src/amr/morton.h` — header-only bit-interleave encoding, child/parent ops
- `src/amr/block.h/c` — block_t wrapping grid_t* with LogicalLocation,
  nblevel[3][3][3], 26 neighbor_ids, tree links, boundary flags, BLOCK_COORD macro
- `src/amr/mesh.h/c` — mesh_t creation: N_root^3 uniform blocks, Morton-sorted
  IDs, coordinate-based neighbor finding, nblevel + boundary flag computation
- `src/amr/meshblock_pack.h/c` — page-aligned contiguous buffers
  (data/rhs/scratch), PACK_IDX macro, load/store between blocks and pack

**Modified files (2):**
- `src/core/params.h` — added `amr_params_t` struct with 7 parameters
  (enabled, max_level, N_block, N_root, chi_refine, chi_coarsen, regrid_every)
- `Makefile` — added AMR_SRC, test-amr-mesh target

### Test results (33/33 passed)

```
test_amr_mesh:
  Morton encode/decode round-trip              PASS
  Morton Z-ordering correct                    PASS
  Morton child encoding (octant 7)             PASS
  Morton parent recovery                       PASS
  8 blocks created (2x2x2)                     PASS
  All blocks are leaves, all level 0           PASS
  Corner (0,0,0): 7 neighbors, 19 boundaries  PASS
  nblevel table correct                        PASS
  1-block mesh flat spacetime Ham L2 = 1e-14   PASS (< 1e-10)
  MeshBlockPack load: 0 error                  PASS
  MeshBlockPack store: correct                 PASS
  8-block pack load: 0 error                   PASS
  ... (33 total)
```

Existing tests (test_flat, test_convergence) pass with no regressions.

### What's next

Stage 2: multi-block uniform mesh + 26-neighbor ghost exchange. Decompose
domain into N_root^3 blocks, fill ghost zones by direct copy from neighbors,
run physics unchanged. Test gate: multi-block = single-grid to roundoff.

---

## 2026-02-16: AMR Stage 2 — Ghost Exchange

(See architecture.html for full details.)

---

## 2026-02-16: AMR Stage 3 — Prolongation + Restriction + Noise Reduction

### Reference code consulted

- **AthenaK `src/mesh/prolongation.hpp`**: `HighOrderProlongCC` — 4th-order
  cell-centered Lagrange prolongation with 5-point 1D stencil. Key insight:
  cell-centered weights differ from vertex-centered. Right child weights =
  left child reversed.
- **AthenaK `src/mesh/mesh_refinement.cpp` `InitInterpWghts()`**: Exact weight
  values: {-45/2048, 105/512, 945/1024, -63/512, 35/2048}.
- **GRChombo `CoarseAverage`** (Chombo library): Simple volume-weighted
  averaging for restriction. 2nd-order accurate, conservative.
- **arXiv:2404.01137** (Etienne 2024): CAKO, CAHD, SSL, per-field sigma.

### Key design decisions

1. **Cell-centered prolongation weights differ from vertex-centered.** The
   plan originally had {-1/16, 9/16, 9/16, -1/16} — those are for vertex-
   centered grids where the interpolation target is at x=1/2. For cell-centered
   grids, the two children are at x=±1/4, giving a 5-point asymmetric stencil.
   Copied from AthenaK.

2. **Simple restriction (not Lagrange).** AthenaK uses Lagrange restriction
   (5-point stencil with edge corrections). GRChombo uses simple averaging.
   For NR (not conservative form), simple averaging is sufficient and avoids
   edge stencil complexity.

3. **CAHD adds to chi evolution, NOT kappa1.** The plan incorrectly stated
   `kappa1_eff = kappa1 * 2^(level_offset)`. The paper (Eq. 26) adds a
   damping term to d_t(phi) proportional to the Hamiltonian constraint:
   `d_t(chi) += 4*chi*C*CFL*dx*H` with C=0.15. On uniform grid the level
   scaling factor = 1.

4. **CAKO uses W=sqrt(chi), not chi.** The paper uses W=e^{-2φ}=sqrt(chi)
   as the KO scaling factor. Using chi directly would be more aggressive
   suppression near punctures.

5. **All noise features disabled by default.** New `noise_params_t` struct
   with boolean flags. Existing tests run with all features off → identical
   behavior to pre-Stage-3 code.

### New files

- `src/amr/prolongation.h/c` — 4th-order Lagrange cell-centered prolongation
  5-point 1D stencil, 3D tensor product (125 coarse cells per fine child)
- `src/amr/restriction.h/c` — volume-weighted averaging (1/8 × 8 children)

### Modified files

- `src/core/params.h` — added `noise_params_t` (CAKO/CAHD/SSL/per-field sigma
  parameters), `double time` for SSL
- `src/evolution/dissipation.c` — CAKO (sigma_eff = sqrt(chi) * sigma_base)
  and per-field sigma (0.99 gauge, 0.3 physical)
- `src/evolution/ccz4_rhs.c` — CAHD (damping to chi via H constraint) and
  SSL (Gaussian lapse damping toward trumpet solution)

### Test results (15/15 passed)

| Test | Result |
|------|--------|
| Weight sum = 1 | 1D: 1.0, 3D: 1.0 |
| Prolongation linear | 0 error (exact) |
| Prolongation convergence | order 5.03 (4th-order confirmed) |
| Restriction error < dx_c^2 | 3.6e-3 < 3.9e-1 |
| Round-trip error < 10*dx_c^2 | 3.6e-3 < 3.9 |
| CAKO flat (chi=1) | ratio = 1.000000 (no effect) |
| Per-field sigma flat | Ham L2 = 4.3e-15 (stable) |
| CAHD single BH | Ham L2: 2.94e-3 → 1.94e-3 (34% reduction) |
| SSL single BH | lapse diff = 1.6e-2, envelope(170M) = 2e-16 |

### No regressions

- test_flat: Ham L2 = 5.29e-14 (unchanged)
- test_convergence: order 5.43/5.47 (unchanged)
- test_amr_mesh: 33/33 (unchanged)

### What's next

Stage 4: Oct-tree refinement + multi-level ghost exchange. Refine and coarsen
blocks based on chi-gradient. Support multiple refinement levels with cross-
level ghost exchange using prolongation and restriction from Stage 3.

## 2026-02-16: AMR Stage 4 — Oct-tree Refinement + Multi-level Ghost Exchange

### Overview

Implemented dynamic oct-tree refinement and coarsening with full cross-level
ghost zone exchange. Blocks can now be refined (split 1→8 children) or
coarsened (merge 8→1 parent) based on the chi-gradient criterion.

### New files

- `src/amr/criterion.h/c` — Chi-gradient refinement criterion
  - `chi_gradient_max()`: max of (dx/chi²)|∇chi| over block interior
  - `criterion_check_block()`: flag refine/coarsen per block
  - `criterion_check_mesh()`: evaluate all leaf blocks
  - Ref: GRChombo ChiTaggingCriterion.hpp:31

- `src/amr/refine.h/c` — Block split/merge + regrid
  - `mesh_refine_block()`: split 1→8, prolongate parent data into children
  - `mesh_coarsen_siblings()`: restrict 8→1, merge back to parent
  - `mesh_enforce_2to1()`: cascade 2:1 level constraint iteratively
  - `mesh_regrid()`: full cycle (criterion + 2:1 + refine + coarsen + rebuild)
  - Ref: Athena++ mesh.cpp AdaptiveMeshRefinement(), meshblock_tree.cpp

- `tests/test_amr_refine.c` — 72 checks across 9 tests

### Modified files

- `src/amr/mesh.h/c` — Added block management functions:
  - `mesh_find_block()`: linear scan by (level, lx1, lx2, lx3)
  - `mesh_add_block()`: append with automatic grow
  - `mesh_remove_block()`: set slot to NULL
  - `mesh_compact()`: remove NULL slots, update all ID references
  - `mesh_rebuild_neighbors()`: multi-level neighbor finding with
    fallback to coarser-level blocks

- `src/amr/ghost_exchange.h/c` — Added multi-level ghost exchange:
  - `ghost_exchange_multilevel()` with 4 phases:
    1. Restrict fine leaf data → non-leaf parents (finest to coarsest)
    2. Same-level exchange at each level (including non-leaf blocks)
    3. Prolongate from parent → fine child ghost zones (all faces/edges/corners)
    4. Same-level exchange at fine levels (overwrites prolongated data
       with exact same-level data where available)
  - `prolongate_from_parent()`: coordinate-based interpolation using
    parent's grid data (interior + ghost zones) for uniform coverage

- `src/numerics/rk4.c` — Mesh stepping uses multilevel ghost exchange
  when max_level > 0, restricts to parents after each full step

- `Makefile` — Added criterion.c, refine.c to AMR_SRC, test-amr-refine target

### Key design decisions

1. **Non-leaf parents keep grid data.** When refined, parent becomes non-leaf
   but retains `grid_t*`. Data updated via restriction from children. Provides
   coarse ghost data for fine blocks. Memory cost: ~12% (1 parent per 8 children).

2. **Parent-based ghost fill (not neighbor-based).** Edge and corner ghost
   cells on fine blocks can extend to physical regions not covered by a single
   coarse neighbor. The parent block always covers all child ghost cells
   (its domain + ghost zones encompass children's domains + ghost zones).
   This avoids per-cell block lookups for edge/corner directions.

3. **Phase 4 same-level exchange.** After prolongation fills ghost zones from
   coarse parent data, a second same-level exchange pass among siblings
   overwrites prolongated values with exact fine-resolution data where available.
   This ensures maximum accuracy at fine-fine interfaces.

4. **Global dt (no subcycling).** All leaf blocks advance with the same
   dt = CFL * dx_finest. Stage 5 will add subcycling.

### Test results

test_amr_refine: 72/72 passed:
- Refine single block: parent non-leaf, 8 children at level 1, correct origins/dx
- Prolongation into children: exact for quadratic (0 error)
- Coarsen round-trip: error < 10 * dx² (restriction accuracy)
- 2:1 constraint: all 8 neighbors cascade correctly in 2×2×2 mesh
- Chi-gradient criterion: all blocks near BH flagged for refinement
- Multi-level ghost exchange: error < 0.001 * dx_c² (restriction-limited)
- Mesh find/add/remove/compact: correct slot management
- Neighbor rebuild: fine blocks see coarser neighbors, boundary flags correct
- Full regrid: 8→64 leaves near BH, no NaN, 2:1 constraint satisfied

### No regressions

- test_flat: Ham L2 = 5.29e-14 (unchanged)
- test_convergence: order 5.43/5.47 (unchanged)
- test_amr_mesh: 33/33 (unchanged)
- test_amr_ghost: 16/16 (unchanged)
- test_amr_prolong: 15/15 (unchanged)

### What's next

Stage 5: Subcycling — fine levels advance with smaller dt for computational
efficiency. Berger-Oliger time stepping with coarse-fine synchronization.

---

## Apparent Horizon Finder + Einstein-Maxwell (2026-02-18)

### AH Finder (Step 1)

Implemented hyperbolic flow AH finder inspired by BHaHAHA (arXiv:2505.15912).

**New files:**
- `src/numerics/interpolate.h` — 4th-order Lagrange interpolation at arbitrary
  (x,y,z) from grid. 5-point stencil, 125-point tensor product. Value + derivative.
- `src/diagnostics/ah_finder.h/c` — Full AH finder: expansion computation,
  damped-wave flow solver, area/mass/spin diagnostics.
- `tests/test_ah_finder.c` — 7 tests, 13 checks.

**Algorithm:** Trial surface r=h(θ,φ) on angular grid, evolved via damped wave
∂h/∂τ = v, ∂v/∂τ = -η·v - c²·Θ(h). Converges to Θ=0 (apparent horizon).
34 field interpolations per surface point (chi, h_ij, K, A_ij + derivatives).

**Test results (13/13 PASS):**
- Interpolation exact to roundoff (3.6e-15)
- Schwarzschild AH radius within 0.6% of M/2
- Expansion positive outside, negative inside AH
- Area = 50.28 (expected 50.27, 0.03% error)
- M_irr = 1.000152 (0.015% error), spin = 0 for Schwarzschild
- Angular convergence confirmed
- Boosted BH: AH found, area matches Schwarzschild

**CLI:** `--ah` enables, `--ah_every N` controls frequency, `--ah_guess R` sets initial radius.

### Einstein-Maxwell (Step 2)

Implemented 3+1 conformal Maxwell evolution (arXiv:0907.1151).

**Changes:**
- `src/core/fields.h` — 6 new fields: FIELD_E1..E3, FIELD_BM1..BM3 (NUM_FIELDS: 25→31)
- `src/core/params.h` — `charge` in puncture_data_t, `em_enabled`/`kappa_em` in sim_params_t
- `src/evolution/maxwell_rhs.h/c` — Maxwell RHS: conformal curl, advection, K coupling,
  constraint damping. Combined ccz4_maxwell_rhs_point wrapper.
- `src/evolution/ccz4_rhs.c` — EM stress-energy T^μν coupling (gated by em_enabled):
  ρ_EM in Theta/K, j^i_EM in Gamma^i, S_ij in A_ij equations.
- `src/evolution/dissipation.c` — EM fields not classified as gauge fields
- `src/initial_data/puncture.c` — Initialize EM fields to 0 in flat/BL data
- `src/initial_data/bowen_york.c` — Coulomb E^i for charged BHs, EM fields zeroed in HiSpID
- `src/main.c` — `--em` flag, `--puncture ...,Q` charge parsing, AH finder integration

**Test results (15/15 PASS):**
- Field count: NUM_FIELDS = 31, positions correct
- EM flat stability: E=B=0 stays zero for 1000 steps
- Plane wave: propagates, energy bounded, Ham < 1e-6
- Charged BH: Coulomb field initialized, constraints bounded after 50 steps
- Constraint damping: div(E) reduced 10.88x
- Energy conservation: bounded and non-negative

**Backward compatibility:** All prior tests pass unchanged. Flat spacetime Ham L2 = 5.3e-14.
Convergence order 5.4 (unchanged). EM fields zero by default, zero overhead when off.

### What's next

Step 3: XCTS + Superposed Kerr-Schild + Fill-the-Holes for chi ≤ 0.9997 initial data.
Uses AH finder from Step 1 for excision boundary.
