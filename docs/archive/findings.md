# Performance & Optimization Findings

New opportunities identified through deep analysis of the Lattice codebase, comparison
against production NR codes (AthenaK, GR-Athena++, SpECTRE, CarpetX, ExaHyPE, NRPy+),
and 2024-2026 mathematical methods literature. February 2026.

Items already documented in `docs/stage3options.html` (LTO, `__restrict__`, fused d1/d2,
conditional EM allocation, hoist em_enabled, CSE, symmetric tensor raise, mixed d2
precomputation, interior/boundary split, persistent pack, device-side ghost exchange,
per-thread GPU overhead, kernel splitting, optimal block size, d2 reorder, dense output,
CCZ3, optimized-stability RK, P-ERK, mixed-order stencils, KO stage reduction) are
**not repeated here**. This document covers only genuinely new findings.

---

## Correctness Bugs

### 1. 0th-order restriction in parent sync limits AMR convergence

**File:** `src/numerics/rk4.c:334-387` (`restrict_level_to_parents`)

The function uses simple 2x2x2 averaging (0th-order) to populate non-leaf parent data:
```c
sum += cg->fields[f][IDX(cg, fi, fj, fk)];
pg->fields[f][IDX(pg, pii, pjj, pkk)] = sum * 0.125;
```
Meanwhile `restrict_to_coarse_buf` and `restrict_child_into_parent` both use 6th-order
Lagrange stencils (`restrict_w[]`). The 0th-order parent data feeds directly into the
ghost exchange critical path: parent grid -> Phase 3 `copy_from_coarse_grid`
(`ghost_exchange.c:278-287`) -> `coarse_buf` -> Phase 4 prolongation -> fine ghost
zones -> fine RHS. Fine-level RHS sees O(h^2) ghost data at coarse-fine boundaries.

Currently masked in tests because convergence is measured on uniform-level regions.
Would manifest in long-duration AMR runs measuring constraints at coarse-fine boundaries.

**Fix:** Replace 2x2x2 average with `restrict_cell()` from `restriction.c`. ~20 LOC.
**Impact:** Correctness -- maintains 6th-order convergence consistency across AMR levels.

### 2. CK45 subcycling produces degraded temporal interpolation

**File:** `src/numerics/rk4.c:872-879` (CK45 subcycling path)

The CK45 path never calls `save_k1_from_pack` (compare classic RK4 at line 892 which
does). Without it, `rhs_old` stays all-zeros (from `block_alloc_fields_old` memset at
`block.c:193`). The `interp_order` still ramps to 4 (`block.c:274-277`), but quartic
interpolation in `block_time_interp` (`block.c:314-352`) reads zero RHS terms:
```
dst[i] = w_Un * u_n[i] + w_Un1 * u_n1[i] + w_Unm1 * u_nm1[i]
       + w_Fn_dt * 0 + w_Fnm1_dt * 0
```
This produces a degraded 3-point cubic with wrong zero-derivative constraints --
potentially worse than true linear interpolation.

The first CK45 stage evaluates RHS from the beginning-of-step state, which IS k1.
**Fix:** Add `save_k1_from_pack(pack, m->blocks)` after stage 0, before stage 1
overwrites the RHS buffer. ~5 LOC.
**Impact:** Correctness -- restores quartic temporal accuracy at AMR boundaries.
**Ref:** arXiv:2503.09629 (CarpetX dense output)

### 3. Duplicate enforce_algebraic uses slower code path

**File:** `src/amr/refine.c:78-139` (`enforce_algebraic_block`)

Copy-paste of `enforce_algebraic` from `rk4.c:127-192`, but uses `1.0 / cbrt(det)`
instead of `fast_inv_cbrt(det)` (3-5x slower) and has a different `if (det > 0.0)` guard.
Should be factored into a shared function using the fast path.

**Fix:** Call `enforce_algebraic()` from `rk4.c` directly or extract shared function. ~10 LOC.
**Impact:** Code hygiene + 3-5x speedup for the refinement enforce pass.

---

## High-Impact Optimizations

### 4. Persistent per-level packs for AMR subcycling

**Files:** `src/numerics/rk4.c:856-922` (`step_level`), `rk4.c:557-601` (`mesh_build_leaf_pack`)

The single largest performance issue for AMR runs. With L-level subcycling, packs are
rebuilt 2^L - 1 times per coarse step (15 for 4 levels). Each rebuild cycle: `posix_memalign`
x4 + `calloc` x7 + `memcpy` load (all fields) + metadata + neighbors + coarse_buf +
GPU map (H2D) + 5 CK45 stages + GPU unmap (D2H) + `memcpy` store + `free`.

Quantification for 100 blocks at N_block=32, 4-level AMR:
- CPU: ~48 GB of memcpy per coarse step (3-7% overhead from malloc/free alone)
- GPU: ~142 GB of PCIe traffic per coarse step (dominates total runtime)

This is architecturally different from the "persistent pack" in `stage3options.html`,
which describes keeping data on device between RK stages within a single step. Subcycling
creates entirely new packs per level per substep, abandoning device-resident data.

**Fix:** One pack per AMR level, allocated once, surviving across subcycle steps. Only
rebuild metadata after regridding. Reuse data/rhs/scratch buffers; only reload fields
via `meshblock_pack_load`/`meshblock_pack_store`. ~200 LOC.
**Impact:** 15-30% CPU AMR speedup. Eliminates ~95% of GPU PCIe traffic from pack rebuild.
**Ref:** AthenaK persistent data residency model (arXiv:2409.16053)

### 5. OpenMP-parallelize ghost exchange and Sommerfeld on CPU

**Files:** `src/backend/backend_cpu.c:463-500` (`packed_exchange_same_level`), `backend_cpu.c:208-315` (`backend_sommerfeld_packed`)

Both packed ghost exchange and Sommerfeld BCs run with zero OpenMP parallelization.
Ghost exchange reads interiors and writes ghost zones (disjoint sets -- confirmed safe
for parallel execution, no data race). The source code at line 454 even contains a comment
noting it is safe to parallelize, but no pragma was added.

For 100 blocks on 8-core M4: ghost exchange runs 500 times per step (5 CK45 stages x
100 blocks x 26 directions). Sommerfeld runs once per stage for ~20 boundary blocks.

**Fix:** Add `#pragma omp parallel for schedule(dynamic)` on outer block loops. ~5 LOC total.
**Impact:** Ghost exchange is 10-15% of step time; parallelizing gives ~6-7x speedup
for that phase = 8-12% total. Sommerfeld adds ~3-4% for boundary-heavy AMR. Combined: ~10-15%.

### 6. Prolongation/ghost-fill loops iterate over entire grid including interior

**Files:** `src/amr/ghost_exchange.c:462-536` (`prolongate_from_own_coarse_buf`), `src/backend/backend_cpu.c:847-914` (`packed_prolongate_fine_ghosts`), `src/amr/refine.c:496-531`

All three iterate over all Nt^3 points and skip interiors via a branch. For N_block=32
(Nt=40): 51% of iterations are skipped. The prolongation stencil is 7x7x7 = 343 points
per ghost cell, so wasted iterations are expensive.

Similarly, `backend_sommerfeld_packed` at `backend_gpu.c:253-330` launches threads for
all blocks and all points but 97.5% do nothing (only boundary ghost cells on
boundary-touching blocks need processing).

**Fix:** Replace with direct ghost-region loops (6 faces + 12 edges + 8 corners) that
iterate only over ghost cells. For Sommerfeld on GPU, build a boundary-cell index list
and launch a 1D kernel over just those cells.
- Prolongation: ~80 LOC, ~2x speedup for Phase 4.
- Sommerfeld GPU: ~50 LOC, 40x fewer threads launched.
- Sommerfeld CPU: ~30 LOC, eliminates 30-51% no-op iterations.

### 7. enforce_algebraic and restrict_to_parents not on GPU

**Files:** `src/numerics/rk4.c:649-659`, `rk4.c:127-192` (enforce_algebraic), `rk4.c:334-387` (restrict_to_parents)

After `backend_unmap_pack()`, both functions run on CPU, requiring a full D2H transfer.
`enforce_algebraic` is a per-point operation (no neighbor access) -- trivially
parallelizable as a GPU kernel.

Quantification for 100 blocks at N=32: D2H = ~1.58 GB / 12 GB/s PCIe = ~132 ms.
On GPU: ~1 ms at 900 GB/s HBM. Moving enforce_algebraic to GPU saves ~130 ms/step.

**Fix:** Add `backend_enforce_algebraic_packed()` GPU kernel. Requires porting
`fast_inv_cbrt`, `compute_det_sym`, `compute_inverse_sym`, `make_trace_free` to
`omp declare target`. ~60 LOC.
**Impact:** ~130 ms/step saved on GPU. Enables keeping data on device longer.

---

## CPU Optimizations

### 8. Skip EM fields in dissipation when EM is off

**File:** `src/evolution/dissipation.c:54` -- `for (int f = 0; f < NUM_FIELDS; f++)`

Dissipation iterates over all 31 fields even when `em_enabled=false`. The 6 EM fields
(FIELD_E1..FIELD_BM3) are zeroed but still load 27 doubles per field (9 stencil points
x 3 directions) and compute weighted sums. 6/31 = 19.4% of dissipation work is wasted.

**Fix:** `int nf = p->em_enabled ? NUM_FIELDS : FIELD_E1;` as loop bound. 1 line.
**Impact:** ~3-4% total (dissipation is ~15-20% of RHS cost; saving 19% = 3-4%).

### 9. Compile-time EM dispatch via separate build targets

**Files:** `src/backend/backend_cpu.c:176`, `src/evolution/dissipation.c:54`, Makefile

Instead of runtime `if (p->em_enabled)` checks, build two object files:
`backend_cpu_ccz4.o` (NUM_ACTIVE_FIELDS=25, calls `ccz4_rhs_point` directly) and
`backend_cpu_em.o` (NUM_ACTIVE_FIELDS=31, calls `ccz4_maxwell_rhs_point`). Select via
`make` vs `make EM=1`. Eliminates all runtime EM overhead: no RHS dispatch branch, no
wasted dissipation, no wasted allocation, tighter loops, better instruction cache.

AthenaK achieves the same via C++ templates for compile-time algorithm selection.

**Fix:** Makefile change + `#ifndef NUM_ACTIVE_FIELDS` fallback. ~20 LOC.
**Impact:** Subsumes finding #8 and the stage3options "hoist em_enabled" item. ~4-5% CPU, ~5% GPU.

### 10. Hoist advection sign check outside per-field loop

**Files:** `src/numerics/finite_diff.h:169` (`fd_adv`), `src/evolution/ccz4_rhs.c:146-163`

`fd_adv()` branches on `vel > 0.0` at every call. In the RHS, `shift[dir]` is the same
for all ~22 field advection terms per direction = 66 redundant branches per point. On
CPU the branch predictor handles this (0.5-1% benefit). On GPU, each branch wastes
instruction slots and causes warp divergence (5-8% benefit).

**Fix:** Add branchless `fd_adv_pos()` / `fd_adv_neg()` to `finite_diff.h`, select once
per direction. ~25 LOC.
**Impact:** 0.5-1% CPU, 5-8% GPU for advection portion of RHS.

### 11. Fast-path pow(lapse, 1.0) in gauge equation

**File:** `src/evolution/ccz4_rhs.c:452` -- `pow(lapse, p->gauge.lapse_power)`

Default `lapse_power = 1.0` makes this `pow(lapse, 1.0)` = identity, but the compiler
cannot optimize away the generic `pow()` call (50-100 cycles) since `lapse_power` is
a runtime parameter.

**Fix:** `double lapse_factor = (p->gauge.lapse_power == 1.0) ? lapse : pow(lapse, p->gauge.lapse_power);` 3 lines.
**Impact:** ~0.5-1% total.

### 12. Flatten per-grid CK45 update loop

**File:** `src/numerics/rk4.c:268-277` (`ck45_update`)

Loops `for (int f = 0; f < NUM_FIELDS; f++)` with inner `for (size_t i = 0; i < n; i++)`.
This means 31 separate OpenMP dispatch cycles + 31 prefetch restarts. The packed version
(`backend_cpu.c:336-339`) correctly fuses into a single flat loop.

**Fix:** Flatten to `total = NUM_FIELDS * npoints`, single loop. 5 lines.
**Impact:** ~5-10% for single-grid runs (test codes, single-BH).

### 13. Exploit Ricci symmetry (upper triangle only)

**File:** `src/evolution/ccz4_rhs.c:203-238`

Ricci is symmetric (R_ij = R_ji) but `FOR2(ii,jj)` computes all 9 entries. Inner loops
do 243 FMAs for Christoffel products; only 162 are unique. Hoist
`chris.ULL[kk][ll][ii]` into a temporary for the inner jj loop.

**Fix:** Compute upper triangle (ii <= jj) and copy. ~15 lines.
**Impact:** ~0.6-1% total (Ricci is ~30% of RHS).

### 14. Triple-fused d1/d2/advection stencil for scalar fields

**File:** `src/numerics/finite_diff.h`, `src/evolution/ccz4_rhs.c:85-163`

Extension of the d1/d2 fusion in stage3options.html. The 4 scalar fields needing all
three derivatives per direction (chi, K, Theta, lapse) share overlapping stencil points.
The union of offsets {-4..+4} = 9 points covers d1 (centered), d2 (centered), and
advection (upwind). Currently: 21 loads per field per direction. Fused: 9 loads.

**Fix:** Create `fd_d1_d2_adv()`. ~40 LOC.
**Impact:** ~2-3% additional on top of d1/d2 fusion. 4 fields x 3 dirs x 12 eliminated loads = 144 saved loads/point.

### 15. Fuse Hamiltonian + momentum constraint evaluation

**File:** `src/diagnostics/constraints.c:82-84` (Hamiltonian), `constraints.c:254-256` (momentum)

Both independently compute derivatives, inverse metric, and full Christoffel symbols.
When both are evaluated at the same point (diagnostic output), all work is duplicated.

**Fix:** Create `compute_constraints_at()` sharing derivatives, metric inverse, and
Christoffels. ~80 LOC.
**Impact:** ~40-45% speedup for combined constraint evaluation (diagnostics only, not RHS hot path).

### 16. Unroll Levi-Civita curl in Maxwell RHS

**File:** `src/evolution/maxwell_rhs.c:265-273`

`FOR4(a,b,c,d_idx)` = 81 iterations per component, but Levi-Civita is nonzero for only
6 of 27 (a,b,d) triplets. 78% of loop iterations are wasted. On GPU, 92.6% of warp
iterations skip (warp divergence).

**Fix:** Unroll to the 6 nonzero terms explicitly. ~30 LOC.
**Impact:** 5-8% of Maxwell RHS (EM-only). Eliminates warp divergence on GPU.

### 17. Precompute per-field sigma array for dissipation

**File:** `src/evolution/dissipation.c:58-63`

`if (p->noise.use_per_field_sigma)` and `is_gauge_field(f)` execute inside the per-field
loop at every grid point. Both are loop-invariant.

**Fix:** Precompute `double sigma_array[NUM_FIELDS]` before the point loop. ~10 LOC.
**Impact:** ~1% (branch removal + vectorization potential).

### 18. Deduplicate d1_shift in CCZ4 + Maxwell RHS

**File:** `src/evolution/maxwell_rhs.c:222-229`, `src/evolution/ccz4_rhs.c:96-99`

When `em_enabled=true`, both functions independently compute `d1_shift[3][3]` (9 fd_d1
calls duplicated). Each 6th-order fd_d1 loads 7 doubles.

**Fix:** Pass precomputed `d1_shift` from ccz4_rhs to maxwell_rhs. ~20 LOC.
**Impact:** ~1-2% of EM-enabled RHS.

---

## GPU Optimizations

### 19. GPU kernel occupancy tuning via num_teams/thread_limit

**File:** `src/backend/backend_gpu.c:44, 198, 268, 349`

All GPU kernels lack `num_teams` / `thread_limit` clauses. The RHS kernel uses ~400
registers/thread; on A100 this limits to 64 threads/SM but the runtime may try 128-256
(causing register spill to local memory). Bandwidth-limited kernels (CK45 update, zero)
should use `thread_limit(256)`.

**Fix:** Add per-kernel `thread_limit` clauses. ~5 LOC per kernel, requires profiling.
**Impact:** 5-15% GPU from avoiding register spill on RHS kernel.

### 20. Sommerfeld asymptotic_value switch/case causes warp divergence

**File:** `src/boundary/sommerfeld.c:24-34`

`asymptotic_value()` uses a switch with 31 evaluations per boundary point in the GPU
kernel. Replace with constant array lookup.

**Fix:** `static const double asym_values[NUM_FIELDS] = {1.0, 1.0, 0.0, ...};` ~5 LOC.
**Impact:** Eliminates warp divergence in Sommerfeld GPU kernel.

### 21. Coalesce separate DMA map pragmas

**File:** `src/backend/backend_gpu.c:83-88`

Three separate `#pragma omp target enter data map(to:...)` for data/rhs/scratch could
be combined into a single pragma, allowing GCC libgomp to coalesce DMA transfers.

**Fix:** Combine into one pragma. ~3 LOC.
**Impact:** Potentially faster H2D transfer (depends on runtime coalescing).

### 22. save_k1_from_pack forces D2H sync in classic RK4 subcycling

**File:** `src/numerics/rk4.c:754-768`

After Stage 1 RHS on GPU, `save_k1_from_pack()` copies the entire RHS buffer (~1.58 GB
for 100 blocks at N=32) to host. Forces a mid-step D2H sync.

**Fix:** Store k1 on device in a dedicated pack buffer for on-device temporal interpolation.
~30 LOC + adding `k1_data` to `meshblock_pack_t`.
**Impact:** Eliminates ~130 ms mid-step D2H transfer.

### 23. enforce_algebraic runs sequentially per-block after unmap

**File:** `src/numerics/rk4.c:655` (`mesh_enforce_algebraic`), `rk4.c:408-415`

After unmap, per-block `enforce_algebraic()` calls have internal OpenMP (collapse(2)) but
fork/join per block. For 100+ blocks at N_block=16 (only 4096 interior points), each
parallel region is dominated by thread startup overhead.

**Fix:** Batch into a single flat loop across all blocks. ~30 LOC.
**Impact:** ~1-2% for multi-block AMR (eliminates 100+ fork/join cycles per step).

---

## AMR Optimizations

### 24. Temporary full-resolution grid during refinement causes 8x memory spike

**File:** `src/amr/refine.c:34-72` (`prolongate_into_child`)

Allocates `grid_alloc(2 * N, ...)` -- a temporary fine grid at 2x parent resolution with
all RK buffers. For N_block=32: 3 blocks x 31 fields x 72^3 = ~282 MB per child. Called
8 times per refinement = ~2.3 GB peak spike.

**Fix:** Prolongate directly into each child using octant-aware stencil evaluation. ~50 LOC.
**Impact:** Eliminates ~2.3 GB peak memory spike during refinement.

### 25. ghost_fill_from_coarser allocates/frees per neighbor

**File:** `src/amr/ghost_exchange.c:611-638`

For every coarse neighbor of every fine block, allocates a temporary interpolation buffer
(~15.9 MB at N_block=32). With 26 directions and multiple fine blocks: 50+ malloc/free
cycles per ghost fill.

**Fix:** Pre-allocate a single thread-local scratch buffer per level. ~15 LOC.
**Impact:** Eliminates hundreds of MB of malloc/free per step in deep AMR.

### 26. mesh_rebuild_neighbors uses O(n^2) linear scan

**File:** `src/amr/mesh.c:367-465`

For each block x 26 neighbors x max_level, `mesh_find_block` does a linear scan of all
blocks. For 1000 blocks with 4 levels: ~100M comparisons per regrid.

**Fix:** Hash table (level, lx1, lx2, lx3) -> block_id during compaction. ~40 LOC.
**Impact:** Regrid O(n^2) -> O(n). Matters for N>500 blocks.

### 27. Subcycling temporal fraction uses floor() division

**File:** `src/numerics/rk4.c:1037`

`double t_coarse_start = floor(t_start / dt_coarse) * dt_coarse;` can accumulate
rounding errors over 10^4+ steps.

**Fix:** Pass `frac` directly from recursive call site (first sub-step = 0.0, second = 0.5).
~10 LOC.
**Impact:** Exact temporal fractions, eliminates floating-point drift.

### 28. Pre-compute restriction/prolongation weight products

**Files:** `src/amr/restriction.c:52-67`, `src/amr/prolongation.c:64`, `src/backend/backend_cpu.c:546-558, 897`

Inner loops compute `wkj = restrict_w[sk] * restrict_w[sj]` for every cell, but the 6x6
(restriction) and 7x7 (prolongation) product tables are constant.

**Fix:** Pre-compute `static const double wkj[6][6]` and `prolong_wkj[7][7]`. ~10 LOC.
**Impact:** Eliminates ~4.6 billion redundant multiplies per restriction pass (100 blocks, N=32).

### 29. Sommerfeld BC iterates over all 31 fields when EM is off

**File:** `src/backend/backend_cpu.c:294` -- `for (int field = 0; field < NUM_FIELDS; field++)`

Processes all 31 fields even when only 25 are active. 24% overhead for boundary-heavy meshes.

**Fix:** Use `n_active_fields` as loop bound. ~3 LOC.
**Impact:** 24% Sommerfeld speedup when EM is off.

---

## Mathematical Methods

### 30. Adaptive constraint damping (dynamic kappa1)

Instead of fixed `kappa1=0.1`, adapt based on local constraint magnitude:
`kappa1_local = kappa1_base * f(Ham, Mom)` where f decays as constraints decrease.
Offers a middle ground between fixed CCZ4 (unstable at 10^3 M) and CCZ3 (no damping).

Formulation hierarchy for a `--formulation` flag:
- `ccz4`: fixed kappa1 (current, stable ~10^3 M)
- `ccz4` + adaptive kappa1: strong early damping, weak late (stable ~10^4-10^5 M)
- `ccz4p` (= Z4c, used by AthenaK): no momentum damping (stable ~10^5 M)
- `ccz3`: no Theta, no damping (stable ~10^5 M, 24 fields)

**Fix:** ~10-20 LOC in `ccz4_rhs.c`. Needs tuning.
**Impact:** Better early-time constraint control without late-time instability.
**Ref:** arXiv:2501.01055 (CCZ3), arXiv:2409.10383 (AthenaK Z4c)

### 31. Taylor-based AMR prolongation (ExaHyPE approach)

Instead of tensor-product Lagrange interpolation, use Taylor expansion around nearest
cell center: compute local derivatives on source grid, evaluate Taylor polynomial at
target cells. ExaHyPE demonstrates elimination of accumulated errors near AMR boundaries
in black hole simulations.

**Fix:** ~200 LOC to replace tensor-product kernel (keep Lagrange as fallback).
**Impact:** Faster prolongation, reduced AMR boundary artifacts.
**Ref:** arXiv:2504.15814 (ExaHyPE, 2025)

### 32. Richardson extrapolation for AMR refinement criteria

Instead of chi-gradient heuristic (`criterion.c`), estimate local truncation error via
Richardson extrapolation: compare the current leaf solution against the coarser parent
level. Refine where the difference exceeds a threshold. Physics-agnostic, catches
truncation error in any field (A_ij, Gamma^i) not just chi.

**Fix:** ~100-150 LOC. Coarse solution already available from AMR hierarchy.
**Impact:** 10-30% fewer blocks for the same accuracy.
**Ref:** arXiv:2404.16648 (AMR refinement comparison, 2024)

### 33. IMEX time integration for constraint damping terms

Treat stiff damping source terms (kappa1*Theta, kappa1*Z_i) implicitly, spatial
derivatives explicitly. The implicit part is purely local (no spatial derivatives),
so the "solve" is just point-wise algebraic inversion -- no linear system needed.
Allows larger kappa1 without reducing dt.

Becomes unnecessary if using CCZ3 (no damping terms). Only relevant for CCZ4 with
strong kappa1 during junk radiation phase.

**Fix:** ~200-300 LOC. IMEX RK tableau management is nontrivial.
**Impact:** Larger dt with strong initial damping.
**Ref:** arXiv:2411.23015 (structure-preserving IMEX, 2024)

### 34. Hyperbolic relaxation for initial data (GPU-friendly alternative to multigrid)

Transform the elliptic Hamiltonian constraint into a damped hyperbolic PDE and evolve to
steady state. Maps naturally onto GPU kernels (same FD infrastructure as evolution). No
tridiagonal solves, no colored sweeps.

**Fix:** ~300-500 LOC. Requires damped wave equation + compactified boundary.
**Impact:** 2-4x speedup for initial data solve on GPU vs CPU multigrid.
**Ref:** arXiv:2501.14030 (NRPyEllipticGPU, 2025), arXiv:2111.02424 (Assumpcao et al., 2022)

### 35. AMR-enabled constraint solver (GRTresna)

Use AMR during the constraint solve itself, concentrating resolution near punctures.
For N=10+ widely-separated punctures, most of the domain is flat spacetime.

**Fix:** ~500+ LOC. Integrate AMR block structure into multigrid hierarchy.
**Impact:** 5-10x for N-body initial data with large separation.
**Ref:** arXiv:2501.13046 (GRTresna, 2025)

### 36. Per-puncture / AH-feedback refinement criterion for N-body

GRChombo's 25-BH simulation (arXiv:2505.01495) uses per-horizon refinement: minimum 32
grid points across each apparent horizon, with buffer zones extending 2-3 block widths
beyond. Refinement thresholds scale per-BH based on local mass, so small remnants after
hierarchical mergers are not missed by a threshold tuned for the largest BH.

Lattice's chi-gradient criterion works for binaries but may under-resolve small BHs in
N-body configurations. Coupling the AH finder output to the refinement criterion enables
mass-adaptive tagging.

**Fix:** ~40 LOC in `criterion.c` + coupling to AH finder output.
**Impact:** Correct resolution for unequal-mass N-body. Required for N=10+ production.
**Ref:** arXiv:2505.01495 (GRChombo 25-BH, 2025)

### 37. Automatic per-block FD order selection via smoothness indicator

SpECTRE uses a troubled-cell indicator (TCI) to switch between DG and FD per cell.
The same concept applies to Lattice's mixed-order stencils: monitor solution regularity
per block (e.g., `max(|d4_chi|)`) and automatically select 4th-order FD (already in
`finite_diff.h`) for non-smooth blocks near punctures, 6th-order elsewhere.

This automates the "mixed-order stencils near punctures" idea from stage3options.html
without requiring manual per-level configuration.

**Fix:** Per-block `fd_order` flag in `block_t`, check in `criterion.c` during regrid,
two function pointers for RHS dispatch. ~30 LOC.
**Impact:** 10-20% RHS speedup on fine levels (same as mixed-order), with automatic selection.
**Ref:** SpECTRE DG-FD hybrid (sxs-collaboration/spectre)

### 38. Wavelet-based error indicator for AMR refinement

Dendro-GR uses wavelet coefficients as error indicators: compare the solution at
resolution h with a coarsened version at 2h. The difference gives a pointwise error
estimate without separate convergence runs. More general than chi-gradient (catches
truncation error in any field) and cheaper than Richardson extrapolation (no extra RHS
evaluation).

**Fix:** ~80 LOC. New criterion function comparing block data with its restriction.
**Impact:** More accurate mesh placement than chi-gradient; complements finding #32.
**Ref:** Dendro-GR (paralab.github.io/Dendro-GR)

---

## Production Infrastructure

### 39. Asynchronous output via device-host overlap

AthenaK writes output asynchronously: copy needed data to a host buffer, then write in
a background thread while the simulation continues on the device. Lattice's `output.c`
writes synchronously, blocking the simulation.

For production runs with waveform extraction every few steps, synchronous output is a
bottleneck. The same fork()/pthread mechanism planned for checkpoint (stage3options.html
Section 3) works for slice and waveform output.

**Fix:** Double-buffer output: host-side buffer + background pthread writer. ~50-100 LOC.
**Impact:** Overlaps I/O with computation. Significant for frequent-output production runs.
**Ref:** arXiv:2409.16053 (AthenaK async I/O)

### 40. Raw binary + Python post-processing for 3D output

AthenaK uses raw binary output with Python conversion to HDF5, avoiding runtime
serialization overhead and external library dependencies. This matches Lattice's
"no external dependencies" philosophy better than generating XDMF directly in C.

**Fix:** Write raw binary per block (same layout as checkpoint), lightweight JSON manifest
for block positions/levels/field offsets. Python converter to XDMF/HDF5 for ParaView.
~100 LOC C + ~50 LOC Python.
**Impact:** Simpler and faster than in-C XDMF generation. Zero runtime dependencies.
**Ref:** arXiv:2409.16053 (AthenaK output format)

---

## Not Recommended (New Evaluations)

| Technique | Why not |
|-----------|---------|
| Structure-preserving exponential integrators | Very high effort; requires Hamiltonian reformulation of CCZ4 + matrix exponentials per step. Theoretical advantage unclear for practical runtimes (arXiv:2408.06613) |
| Spectral deferred corrections (SDC) | High effort (~500+ LOC); requires new integrator + quadrature + AMR integration. Adaptive dt benefit (20-50%) achievable more simply via error-estimating embedded RK |
| Compact finite differences | Requires tridiagonal solve per line; explicit 6th-order stencils already work well |
| PML boundary conditions | CCZ4 is not purely hyperbolic; Sommerfeld + constraint damping works fine |
| Parallel-in-time (Parareal) | Convergence requires many iterations; explicit RK4 is already efficient |
| WENO/ENO for AMR operators | Designed for shocks; NR solutions are smooth (C2-C4 near punctures, C-infinity elsewhere) |
| Multigrid AH finder | High effort (~500+ LOC), completely different from current hyperbolic flow. Current method works well (arXiv:2404.16511) |
| Block-major pack layout | Better GPU coalescing but breaks field-major layout invariant across entire codebase. Future consideration only (arXiv:2409.16053) |
| Cache-oblivious stencil algorithms | x-innermost loop already provides optimal cache behavior; adds complexity for ~2-3% gain |
| Apple M4 SME tensor intrinsics | Clang SME support immature; no stable intrinsics API. Monitor for future (arXiv:2502.05317) |
| Charm++ task-based distribution | Only relevant when MPI is implemented (Tier 4). OpenMP tasks sufficient for near-term |

---

## Priority Roadmap

### Tier 0: Fix Now (correctness)
1. 6th-order restriction in parent sync -- `rk4.c:334` (~20 LOC)
2. CK45 subcycling k1 save -- `rk4.c:877` (~5 LOC)
3. Factor out enforce_algebraic, use fast_inv_cbrt -- `refine.c:78` (~10 LOC)

### Tier 1: Quick Wins (days, 15-25% combined CPU improvement)
4. OpenMP-parallelize ghost exchange + Sommerfeld -- `backend_cpu.c:463,208` (~5 LOC)
5. Skip EM fields in dissipation / compile-time EM dispatch -- `dissipation.c:54`, Makefile (~20 LOC)
6. Triple-fused d1/d2/advection stencil -- `finite_diff.h` (~40 LOC)
7. Flatten per-grid CK45 update -- `rk4.c:268` (5 LOC)
8. Fast-path pow(lapse, 1.0) -- `ccz4_rhs.c:452` (3 LOC)
9. Hoist advection sign check -- `finite_diff.h:169` (~25 LOC)
10. Pre-compute restriction/prolongation weight products -- `restriction.c`, `prolongation.c` (~10 LOC)

### Tier 2: Medium Effort (weeks, 20-40% additional)
11. Persistent per-level packs -- `rk4.c:856` (~200 LOC)
12. Direct ghost-region iteration (prolongation + Sommerfeld) -- `ghost_exchange.c`, `backend_gpu.c` (~160 LOC)
13. enforce_algebraic on GPU -- `rk4.c:127` (~60 LOC)
14. Pre-allocate ghost exchange scratch buffers -- `ghost_exchange.c:611` (~15 LOC)
15. Hash table for neighbor lookup -- `mesh.c:367` (~40 LOC)
16. Adaptive kappa1 / formulation hierarchy -- `ccz4_rhs.c` (~20 LOC)
17. Eliminate refinement memory spike -- `refine.c:34` (~50 LOC)
18. Taylor-based AMR prolongation -- `prolongation.c` (~200 LOC)
19. Richardson extrapolation / wavelet refinement criteria -- `criterion.c` (~80-150 LOC)
20. Per-puncture AH-feedback refinement -- `criterion.c` (~40 LOC)
21. Automatic per-block FD order via smoothness indicator -- `criterion.c`, `block.h` (~30 LOC)
22. Async output with double buffering -- `output.c` (~50-100 LOC)
23. Raw binary + Python 3D output -- `output.c` (~100 LOC C + ~50 LOC Python)

### Tier 3: Major Work (weeks-months)
24. GPU kernel occupancy tuning -- `backend_gpu.c` (~25 LOC, needs profiling)
25. save_k1 on device for GPU subcycling -- `rk4.c:754` (~30 LOC)
26. IMEX for constraint damping -- `rk4.c`, `ccz4_rhs.c` (~300 LOC)
27. Hyperbolic relaxation for initial data -- `relaxation.c` (~500 LOC)

### Tier 4: Future
28. AMR-enabled constraint solver (GRTresna approach)
29. Code generation from SymPy for full CSE
30. Block-major pack layout for GPU coalescing

---

## References

- arXiv:2409.10383 -- AthenaK NR (Zhu et al. 2024)
- arXiv:2409.16053 -- AthenaK framework (Stone et al. 2024)
- arXiv:2101.08289 -- GR-Athena++ (Daszuta et al. 2021)
- arXiv:2503.09629 -- CarpetX subcycling with dense output (2025)
- arXiv:2504.15814 -- ExaHyPE Taylor-based AMR interpolation (2025)
- arXiv:2501.14030 -- NRPyEllipticGPU hyperbolic relaxation (2025)
- arXiv:2501.13046 -- GRTresna AMR constraint solver (2025)
- arXiv:2501.01055 -- CCZ3 stable to 10^5 M (Bezares et al. 2025)
- arXiv:2509.19701 -- Parthenon-VIBE block size benchmarks (2025)
- arXiv:2404.16648 -- AMR refinement technique comparison (2024)
- arXiv:2404.16511 -- Multigrid AH finder (Ramos-Buades et al. 2024)
- arXiv:2411.23015 -- Structure-preserving IMEX methods (2024)
- arXiv:2408.06613 -- Exponential integrators for damped Hamiltonian systems (2024)
- arXiv:2111.02424 -- Hyperbolic relaxation for NR initial data (Assumpcao et al. 2022)
- arXiv:2409.01939 -- GRChombo cosmology with adaptive damping (2024)
- arXiv:2505.01495 -- GRChombo 25-BH cluster (Aurrekoetxea et al. 2025)
- arXiv:2505.00097 -- superB/NRPy task-based NR (Etienne et al. 2025)
- arXiv:2502.05317 -- Apple M-Series HPC evaluation (2025)
- Dendro-GR -- wavelet error indicators (paralab.github.io/Dendro-GR)
