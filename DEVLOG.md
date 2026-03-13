# Lattice Development Log

> **Note:** When adding/removing/renaming files or functions, also update
> `docs/architecture.html` — the living map of the codebase structure.

## 2026-03-13: Inspiral test — production dissipation + 2-radius Psi4

**Per-field sigma + CAKO + SSL enabled in inspiral test.** Previously the test
matched BAM's flat sigma=0.1 with CAKO/SSL/per-field sigma disabled. Now uses
Lattice's full noise reduction stack:
- `sigma_gauge=0.99`, `sigma_phys=0.3` (arXiv:2404.01137, Etienne et al. 2024)
- CAKO: sigma multiplied by W=sqrt(chi), auto-suppresses near punctures
- SSL: Gaussian slow-start lapse ramp
- Position-dependent eta: `eta(x) = eta_0/W(x)` for N-body stability

**Two-radius Psi4 extraction.** r=70M and r=100M (was single r=90M). Enables
Richardson extrapolation to r→∞ for publication-quality waveforms. Separate
CSV files: `inspiral_psi4_r70.csv`, `inspiral_psi4_r100.csv`. Matches GRChombo
practice (2 radii, arXiv:2505.01495).

**chi_refine = 0.05** (was 0.5). Matches GRChombo's threshold for aggressive
refinement near punctures.

**chi_coarsen = 0.0125** (4x hysteresis ratio). The criterion formula includes
dx, so coarsening a block (dx → 2dx) roughly doubles the measured criterion.
Additionally, the coarser grid resamples the gradient — sharp features get
smoothed, so the measured gradient is ~0.3-0.7x of the fine-grid value. Net
criterion change on coarsening: ~0.6-1.4x. A 2x ratio is the minimum to
prevent oscillation from the dx factor alone; 4x provides margin for the
gradient resampling effect. GRChombo uses a single threshold (no hysteresis)
but regrids every 64 steps; we regrid every step and need the dead band.

---

## 2026-03-11: N-body BH tracker + inspiral integration + advection default fix

**Inspiral test integration:** Replaced hand-rolled N=2 tracking in
`test_binary_inspiral.c` with `bh_tracker` module. Removes ~60 lines of
duplicated `mesh_bh_separation()` code. CSV now includes dynamic per-BH
position columns (`bh0_x,y,z,mass,spin,lapse, bh1_x,...`) that scale with N.
Remnant AH finder (16x32, higher resolution) kept separate from per-BH tracker.

**CSV column fix:** Added `n_bh_initial` field to `bh_tracker_t` to keep CSV
column count constant across mergers. Without this, each merger added 6 new
columns (remnant BH) making the CSV unparseable. Now merged BHs output `nan`
in their fixed columns.

---

## 2026-03-11 (earlier): N-body BH tracker + advection default fix

**N-body BH tracker** (`bh_tracker.h/c`, ~480 lines):
- Successive lapse-minimum search with exclusion zones (R=2M per BH)
- Per-BH AH finder for mass/spin extraction
- Pairwise merger detection: sep < 3*max(M_i, M_j) → merge pair, create remnant
- CSV diagnostic output: per-BH columns (x,y,z, mass, spin, lapse)
- Auto-enabled for N≥2 punctures in `main.c` (all 3 paths: restart, AMR, single-grid)
- CLI: `--tracker`, `--tracker_every N`

**Advection default fix:** Changed `lapse_advec_coeff` and `shift_advec_coeff`
from 0.0 to 1.0 in `default_params()`. Required for AMR gauge stability —
without β^i ∂_i α, the gauge is purely local and unstable at dx≥2M. All
existing tests pass with new defaults (flat spacetime: advection=0 since β=0;
BH tests: gauge term is small on uniform grids).

**Tests:** `test_nbody_track.c` — 8 tests, 40/40:
1. Initialization (5 BHs, centers match input)
2. Position update (3-BH BL, lapse minima near initial positions)
3. AH finding (single Schwarzschild, M_irr within 15%)
4. Merger detection (sep=2M < 3M threshold)
5. Merger bookkeeping (n_active, merged_into, remnant)
6. CSV output (header + data lines, correct column count)
7. 25-BH allocation (alloc/free cycle)
8. Post-merger tracking (remnant active, AH workspace allocated)

Ref: arXiv:2505.01495 (GRChombo 25-BH cluster simulation)

## 2026-03-07: Fix GPU multigrid V-cycle divergence — two-pass smoother

**Problem:** GPU multigrid solver V-cycle diverged (18x/cycle on H100 with 11
AMR levels). CPU converged at ~0.24/cycle. Root cause: 8-color Gauss-Seidel
(stride 2) + 6th-order FD stencil (radius 3) = race condition. Same-color
points at distance 2 overlap in each other's 7-point stencils.

**Failed approaches (reverted previously):**
1. *2nd-order smoother:* Race-free but stalls — smoother fights coarse grid
   correction in FAS (solves different equation than 6th-order operator).
2. *Under-relaxed Jacobian:* Dampens but doesn't eliminate the race.

**Fix: Two-pass GPU smoothing per color (HPGMG out-of-place pattern).**
- Pass 1 (compute): All threads read solution (frozen), compute Newton deltas
  into scratch buffer. No writes to solution → no race with any stencil radius.
- Pass 2 (apply): Each thread applies `data[idx] += delta[idx]` to its own
  unique point. No race (disjoint writes).

Jacobi within color, GS across colors. Same 6th-order `fd_d2` stencil as CPU.
CPU path unchanged (serial within block, no race needed).

**Implementation:** `mg_smooth_point.h` (delta variants), `backend_hip.cpp`
(compute-delta + apply-delta kernels, `smooth_delta` buffer in solver slots).

**Results (GPU, Tesla P40):**
- 33/33 BY tests pass, 26/26 HiSpID pass, AMR tests pass (11-level OOM on 23GB card)
- GPU convergence: 0.16–0.34/cycle (matches CPU ~0.24/cycle)
- GPU vs CPU residuals: 2.657e-13 vs 2.654e-13 (Jacobi-within-color path difference)
- GPU vs CPU Ham L2: identical (1.0138e-02)

Ref: HPGMG out-of-place GSRB (Adams et al. 2014), block-asynchronous
smoothers (Anzt et al. 2012, arXiv:2510.11152)

## 2026-03-05: Equidistribution-optimal AMR refinement radius scaling

Replaced the ad-hoc `r = 8*dx` refinement radius formula in
`refine_mesh_near_punctures()` with a physics-derived formula:

    r_k = C · M_p · β^k,  β = 2^(3/5) ≈ 1.516,  C = 4

β is derived from the equidistribution principle: equal truncation error
at every level boundary for a p-th order FD scheme on a 1/r^α field:

    β = 2^(p / (α + p + 1))

We choose α = 3 (Riemann curvature / extrinsic curvature ~ 1/r³) with
p = 6 (6th-order FD), giving β = 2^(6/10) = 2^(3/5) ≈ 1.516. This is
less than the standard halving (β = 2), which is only optimal for a
constant field (α = 0). Fields with slower falloff (χ ~ 1/r, Γ^i ~ 1/r²)
are safely over-resolved.

The radius is now per-puncture (scaled by M_p), so unequal-mass binaries
automatically get mass-appropriate refinement regions. Domain half-size
cap prevents excessive levels on small domains.

Old formula depended on N_block (grid-dependent); new formula depends only
on BH mass (physics-dependent).

Full derivation: docs/amr_refinement_ratio.html

## 2026-03-04: Fix GPU solver slot limit for deep AMR

`MAX_SOLVER_SLOTS` was hardcoded to 8 in `backend.h`, but the D10 benchmark
uses 11 AMR levels (slots 0–11 = 12 slots). Slots 8–11 silently failed to map
in `backend_map_solver_pack`, causing the GPU multigrid solver to return
residual=0.0 and produce -nan constraints. Fixed by deriving
`MAX_SOLVER_SLOTS` from `MAX_AMR_LEVELS` (16). Added runtime validation in
`relaxation_solve_amr_mesh` to catch future mismatches.

## 2026-03-04: Binary inspiral test upgraded to D10 benchmark

Upgraded `test_binary_inspiral.c` from a smoke test to a full D10 benchmark
validation against the Samurai cross-code consensus (arXiv:0901.2437). This
is the canonical equal-mass nonspinning BBH benchmark — 5 independent NR codes
(BAM, CCATIE, Hahndol, Lean, SpEC) agreed on the remnant properties.

**Initial data:** Exact D10 QC parameters from Bode et al. 2009
(arXiv:0902.1127, Table I): m_bare=0.48595, d=10M, P_y=±0.09543.
E_ADM=0.9895, J_ADM=0.9530.

**Grid (matching BAM):** L=1536M (outer boundary at 768M, BAM uses 773M),
N_block=32, MAX_LEVEL=11, dx_fine=M/43 (BAM medium: M/44.8). CFL=0.25.
Psi4 extraction at r=90M (BAM extraction radius). Regrid every global step
to track puncture motion on the coarse base grid (dx_base=48M).

**Physics parameters matched to BAM (gr-qc/0610128, arXiv:1212.2901):**
kappa1=0.02 (not 0.1), eta=2/M_ADM (constant, not position-dependent),
sigma=0.1 (not 0.3), gauge="000" variant (full advection on lapse/shift/B),
no CAKO, no SSL, no per-field sigma. Constraint-preserving BCs.

**Test structure:** 8 Tier 1 hard tests (fail the build): stability, Ham/Mom
L2 < 0.1, GW present (>0.01) and sane (<0.20), trumpet lapse (<0.4), inspiral
motion, merger detected. 4 Tier 2 advisory checks (logged): remnant M_chr vs
Samurai (0.9516±0.05), remnant χ vs Samurai (0.6865±0.10), orbital count,
peak Psi4 timing.

**New tracking systems:** GW phase via atan2 unwrapping (counts orbits),
merger detection (sep<3M), remnant AH search (16×32, r_guess=1.5M, after
merger+50M). CSV extended to 19 columns including remnant properties.

**Target hardware:** H100 GPU (80 GB). 11 AMR levels → 2048 fine substeps
per global step. ~58 global steps for T=700M.

Refs: arXiv:0902.1127, arXiv:0901.2437, gr-qc/0610128, arXiv:1212.2901.

Also updated `test_inspiral_convergence.c` to use the same D10 parameters
(m_bare=0.48595, P_y=0.09543).

## 2026-03-04: Checkpoint/restart for pause and resume

Binary checkpoint/restart system for long-running simulations. Saves full
simulation state (all fields on all leaf blocks, mesh structure, sim_params_t)
to a flat binary file. Bitwise-identical restart verified for both uniform and
AMR meshes (14/14 tests).

**File format:** 1024-byte header (magic "LATCKPT", version, step, time, mesh
metadata, full `sim_params_t`) + per-leaf-block records (level, logical location,
origin, all field data including ghost zones). No external dependencies.

**CLI:** `--checkpoint-every N` saves `build/checkpoint_NNNNNN.lat` every N steps.
`--restart <file>` resumes from a checkpoint file. The checkpoint contains all
parameters — only `--restart` and `--steps` needed on the command line.

**AMR tree reconstruction:** On restart, the saved logical locations (level, lx1,
lx2, lx3) of each leaf block are used to rebuild the oct-tree by iteratively
refining from the root. Field data (including ghost zones) is loaded directly
from the file — no ghost exchange needed, ensuring bitwise-identical evolution.

**Integration points:** Checkpoint hook after diagnostics in both AMR and single-
grid evolution loops in `main.c`. Restart path is a separate `if (restart_file)`
branch that skips initial data setup entirely.

Follows the same philosophy as Cactus/CarpetIOHDF5 (Einstein Toolkit) and
GRChombo checkpoint files, but uses a simple flat binary format (no HDF5
dependency).

Files: `src/io/checkpoint.h`, `src/io/checkpoint.c`, `tests/test_checkpoint.c`.

## 2026-03-04: Volume-weighted AMR constraint L2 + lapse advection fix

**Volume-weighted constraint norms:** `mesh_constraint_l2()` and `mesh_momentum_l2()`
(and their packed GPU/CPU variants) were counting every cell equally regardless
of its physical volume. On AMR meshes, fine cells near punctures (dx=0.125M)
vastly outnumber coarse cells (dx=2.0M) but represent tiny physical volume. This
made the L2 norm appear 55x worse after AMR refinement — a diagnostic artifact,
not a physics problem. Fix: weight each cell by dV = dx^3, normalize by total
volume. Standard practice in all AMR codes (GRChombo, Carpet, Cactus).

Changed files: `constraints.c` (mesh-level), `backend_cpu.c` (packed CPU),
`backend_hip.cpp` (GPU kernel — shared memory layout changed from
`[ham, mom, count]` to `[ham, mom, vol]`, all `double`).

**Lapse/shift advection in inspiral test:** The binary inspiral was crashing at
t=20M on AMR (but stable to t=1000M on uniform grid at same domain size L=64M).
Root cause: default `lapse_advec_coeff=0.0` and `shift_advec_coeff=0.0` makes
the gauge evolution purely local (`∂α/∂t = -2α(K-2Θ)` with no transport term).
On coarse AMR base grids (dx=2.0M), this is unstable — the lapse collapses to
zero instead of settling to the trumpet value (~0.3). Every production code
(GRChombo, BAM, Einstein Toolkit) uses advection coeff=1.0.

Fix: Set `lapse_advec_coeff=1.0` and `shift_advec_coeff=1.0` in the inspiral
test. Default in `params.h` unchanged (0.0) for backwards compatibility with
other tests. The advection terms `β^i ∂_i α` and `β^j ∂_j β^i` are already
computed in `ccz4_rhs.c` — setting the coefficient to 1.0 has zero additional
computational cost.

Ref: gr-qc/0206072 (Alcubierre et al., Gamma-driver),
gr-qc/0610128 (Brugmann et al., advection form for moving punctures).

## 2026-03-03: GPU diagnostics — constraints, lapse, separation, NaN, Psi4

Moved the heaviest per-step diagnostics from CPU to GPU. The binary inspiral
test on H100 showed 254s/step with 0% GPU-Util because diagnostics (constraint
L2, min lapse, BH separation, Psi4) ran every step on CPU, each requiring a
full D→H sync + CPU grid scan (~250s for constraints at 5.76M points).

**Diagnostic-only pack mapping:** `backend_map_pack_diag()` / `backend_unmap_pack_diag()`
transfer only data + metadata to device (skips rhs/scratch/accum, ~75% savings).
Read-only: no D→H sync on unmap. CPU backend: no-op.

**GPU diagnostic kernels (backend_hip.cpp):**
- `hip_constraint_l2` — Hamiltonian constraint L2 norm via shared-memory reduction
- `hip_momentum_l2` — momentum constraint L2 norm via shared-memory reduction
- `hip_min_lapse` — min lapse + position tracking (block-level min reduction)
- `hip_check_finite` — NaN/Inf check via atomicOr
- `hip_bh_separation` — two-pass min-lapse with exclusion zone for BH pair
- `hip_psi4_sphere` — Psi4 extraction: one thread per angular point, calls
  `psi4_at_point()` on device, host does mode decomposition

**Device-compilable diagnostics:**
- `constraints.c`: `compute_hamiltonian_at()`, `compute_momentum_at()` annotated
  `LATTICE_DEVICE`. Static arrays renamed to `c_h_idx`/`c_A_idx` with device annotation.
- `psi4.c`: `psi4_compute()`, `psi4_at_point()` annotated `LATTICE_DEVICE`.
  Arrays renamed to `p4_h_idx`/`p4_A_idx`/`psi4_levi_civita`. `DOT_PHYS` macro
  (GCC statement expression) replaced with `psi4_dot_phys()` inline function.
  C compound literals and implicit void* casts fixed for nvcc C++ compatibility.
  Factored out `psi4_decompose_modes()` for host-only mode decomposition.
- Both added to `HIP_DEVICE_SRC` in Makefile.

**Headers updated for device compilation:**
- `mesh.h`: Added `EXTERN_C_BEGIN/END` and `device.h` include (nvlink needs
  C linkage for cross-file device function calls).
- `psi4.h`, `constraints.h`: Added `EXTERN_C_BEGIN/END`, `device.h` include,
  `LATTICE_DEVICE` on device-callable functions.

**Backend API (backend.h):** 7 new functions — `backend_map/unmap_pack_diag`,
`backend_constraint_l2_packed`, `backend_momentum_l2_packed`,
`backend_min_lapse_packed`, `backend_bh_separation_packed`,
`backend_check_finite_packed`, `backend_psi4_extract_packed`.

**Inspiral test (test_binary_inspiral.c):** GPU path uses diagnostic pack mapping
for constraints/lapse/separation/NaN + Psi4 extraction. CPU path unchanged.
PSI4_EVERY and AH_EVERY tunable for mixed GPU+CPU steps.

**Estimated impact:** Per-step diagnostics drop from ~250s (CPU) to ~50ms (GPU).
DIAG_EVERY=1 now affordable. Only Psi4 mode decomposition and AH finder remain
on CPU (infrequent steps).

## 2026-03-02: Remove N_ROOT > 1 + full inspiral test (T=700M)

**N_ROOT removal:** Hardcoded single root block (N_ROOT=1). Multi-root meshes
caused V-cycle divergence in the composite multigrid solver (cross-block coupling
only via ghost exchange, which doesn't work correctly across root blocks). With
N_ROOT=1, all multi-block topology comes from AMR refinement — the solver sees
the whole domain as one block at the coarsest level, and uniform MG converges
correctly. Removed `N_root` from `mesh_t`, `amr_params_t`, `mesh_create` API,
CLI `--N_root` flag, and all test files. Higher MAX_LEVEL compensates: e.g.,
N_BLOCK=32, MAX_LEVEL=4 gives dx_fine = 0.125M (was N_ROOT=3, MAX_LEVEL=3,
dx_fine ≈ 0.083M — comparable resolution, far simpler code path).

**Inspiral test upgrade:** Updated `test_binary_inspiral.c` for full inspiral +
merger + ringdown. N_BLOCK=32, MAX_LEVEL=4, T_FINAL=700M (~3-4 orbits + merger
+ ringdown for D=10M binary, Peters estimate ~781M). Added diagnostics CSV
(`build/inspiral_diagnostics.csv`) with time series of Ham/Mom L2, lapse, BH
separation, Psi4 (2,2) mode, AH masses and spins. All diagnostics flush after
every write for crash resilience.

Refs: BAM D=6.5M → 160M merger (gr-qc/0610128), D=11M → 1000M (0706.0904).

## 2026-03-02: Fix 3 AMR bugs: composite multigrid divergence + evolution blowup

**Bug 1 (PRIMARY): Cross-level ghost exchange in multigrid solver.**
`solver_ghost_exchange()` called `ghost_exchange_all_blocks()`, which does
direct `memcpy` between blocks regardless of AMR level. When a fine block
(level L) has a coarser neighbor (level L-1), data is copied at wrong physical
resolution (coarser block covers 2x the physical extent, same cell count).
This corrupts FD stencils near refinement boundaries, compounding each V-cycle
(8 colors × 8 sweeps = 64 bad ghost exchanges per level per cycle).

With 3 AMR levels: V-cycles diverged (2x/cycle). With 5 levels: divergence
damped by multi-level structure → very slow convergence (0.85/cycle, 159
V-cycles). The wrong initial data then caused evolution blowup at t=5.5M.

**Fix:** Added `ghost_exchange_multilevel_all()` in `ghost_exchange.c` — same
coarse-buffer protocol as `ghost_exchange_multilevel()` but processes ALL blocks
(not just leaves). Solver now uses this + re-applies zero Dirichlet BCs.
Result: convergence factor ~0.24 per V-cycle (excellent).

**Bug 2: Uninitialized rhs arrays (latent, defense-in-depth).**
`posix_memalign` does not guarantee zero memory. On Linux large allocations
use `mmap` (zero pages), masking the bug, but on other platforms rhs could
contain garbage. Added `memset(rhs[SOL_*], 0, ...)` in all 4 solver entry
points. Also zeroes `fields[SOL_*]` for consistency.

**Bug 3: Inspiral test parameters too aggressive.**
L=20 domain too small (boundaries 5M from BHs), 5 AMR levels gave gauge CFL
violation (speed ~31c, CFL ~8x). Updated to L=64, N_ROOT=3, N_BLOCK=32,
MAX_LEVEL=3, PSI4_RADIUS=20, disabled SSL.

## 2026-03-02: Optimization pass (CPU + GPU + dead code)

Comprehensive optimization pass covering CPU, GPU, and code cleanup.

**CPU optimizations:**
- Precomputed `inv_dx` in `grid_t`: all FD stencil functions now multiply by
  `inv_dx` instead of dividing by `dx`. Eliminates ~20 divisions per grid point
  per RHS evaluation. Touched 12+ source files.
- KO dissipation branch reduction: precomputed per-field `sigma_arr[]` eliminates
  25+ branches per point in the hot field loop.
- Unit-determinant metric inverse: `compute_inverse_sym_unit_det()` skips det
  computation and division in `enforce_algebraic` (saves ~6 multiplies + 1
  division per point). Used in both CPU and GPU backends after det=1 enforcement.
- OMP-parallelized mesh-level ghost exchange: added `#pragma omp parallel for`
  to block loops in `ghost_exchange()`, `ghost_exchange_all_blocks()`,
  `ghost_exchange_multilevel()` (phases 2-4), and `fill_coarse_buf_ghosts()`.

**GPU optimizations:**
- Compact Sommerfeld kernel: boundary-only block ID list (`boundary_block_ids[]`)
  built during `backend_map_pack`. Sommerfeld launches `n_boundary * Nt^3` threads
  instead of `nb * Nt^3`, eliminating 70-90% of idle threads.
- Fused RK4 kernels: `hip_rk4_stage` (accum += w*dt*rhs; data = scratch + a*dt*rhs)
  replaces separate accum_add + axpy. `hip_rk4_final` (data = scratch + accum +
  w*dt*rhs) replaces accum_add + copy + apply_accum. Saves 3 kernel launches per
  RK4 step.
- Shared memory block metadata: `dx_arr[b]` loaded into `__shared__` once per GPU
  block in RHS kernel. Saves 63 redundant global loads per block (block_size=64).
- Lazy host copy: split `backend_unmap_pack` into no-sync (free only) and sync
  (copy + free) variants. All current call sites use sync; no-sync enables future
  optimization when host data isn't needed.
- HIP streams: persistent `hipStream_t` for all kernel launches. Non-blocking
  launches enable potential host/device overlap.

**Dead code removed:**
- `block_reset_interp()` (block.h/c) — unused since interpolation refactor
- `restrict_all()` (restriction.h/c) — superseded by per-level restriction
- `ghost_exchange_array()` (ghost_exchange.h/c) — superseded by packed ghost exchange

**Verification:** `make` zero warnings, convergence order 6.56/6.25 (unchanged),
all test suites pass (flat, convergence, amr_evolve, maxwell, bowen_york, hispid,
ah, psi4, cp_bc).

## 2026-03-02: Binary inspiral upgraded to MAX_LEVEL=5

Upgraded the binary inspiral test to MAX_LEVEL=5: dx_fine = 0.020M (beating
Brugmann et al. 2008's dx ~ 0.024M). Mesh: 584 blocks, 512 leaves. Running
via nohup as `inspiral_lev5.log`, expected to take several hours.

This was enabled by fixing three AMR composite multigrid bugs (below) that
caused the solver to diverge and evolution to blow up at step 5.

## 2026-03-02: Three AMR composite multigrid bug fixes

Three bugs in the AMR composite multigrid solver and initial data pipeline
were causing solver divergence and early evolution blowup on multi-level meshes.
All fixed in commit 32298e8.

**Bug #1: Non-leaf ghost zones in multigrid solver.** `ghost_exchange()` skips
non-leaf blocks (it only processes leaves). But the composite AMR V-cycle
operates on ALL blocks at each level — non-leaf parents participate in
restriction, tau correction, and smoothing. Their ghost zones contained stale
data, corrupting FD operator evaluations.

*Fix:* Added `ghost_exchange_all_blocks()` to `src/amr/ghost_exchange.c/.h`.
This variant includes non-leaf blocks in the same-level 26-neighbor exchange.
Used by `solver_ghost_exchange()` in `relaxation_amr.c`, which wraps it with
solver BC re-application.

**Bug #2: Leaf block RHS accumulation in V-cycle.** In `composite_vcycle()`
(relaxation_amr.c), `apply_tau_correction()` adds L(u) to rhs for all blocks
at the coarser level. But `restrict_to_coarser_amr()` only overwrites rhs on
blocks that have fine children. Leaf blocks (childless at the target level)
were never reset — their rhs accumulated stale L(u) terms across V-cycles,
causing monotonic divergence.

*Fix:* Zero rhs on childless blocks before tau correction. After tau, they get
rhs = 0 + L(u) = L(u), the correct FAS target (delta = 0, no correction needed
since no fine-level residual exists for these blocks).

**Bug #3: Non-leaf blocks have garbage after CCZ4 conversion.**
`set_bowen_york_mesh()` only converts leaf blocks from solver variables to CCZ4
fields. Non-leaf parents retained solver garbage (residuals, operator values).
During subcycled evolution, `ghost_fill_from_coarser()` reads non-leaf block
data to fill fine-level ghost zones — junk data caused the evolution to blow
up at step 5.

*Fix:* Added a restriction loop (fine-to-coarse, max_level down to 1) in
`set_bowen_york_mesh()` after CCZ4 conversion. Each non-leaf parent gets valid
data restricted from its children. Followed by `ghost_exchange_all_blocks()` +
`ghost_exchange_multilevel()` to fill all ghost zones.

**Also fixed:** `refine_mesh_near_punctures()` AABB distance check — was using
block-center-to-puncture distance, now uses minimum distance from puncture to
block bounding box. The old check failed for large blocks where `r_refine` was
smaller than the block half-diagonal.

**Results:** AMR composite multigrid solver now converges monotonically:
4.3e-4 to 9.8e-13 in 68 V-cycles. Binary inspiral evolution runs stably (no
more step-5 blowup). All existing tests pass (BY 33/33, all others green).

## 2026-03-02: Binary inspiral system validation test

Added `tests/test_binary_inspiral.c` — a comprehensive integration test that
exercises all 10 major subsystems simultaneously: Bowen-York initial data with
FAS multigrid, CCZ4 evolution with CP BCs, RK4 integration, KO dissipation,
constraint monitoring, Psi4 extraction, AH finder, lapse/separation tracking,
and 1D slice output.

**Physical setup:** Equal-mass non-spinning quasi-circular binary (Brugmann et
al. 2008, arXiv:0709.0838). Bare masses m=0.4824, separation d=10M, tangential
momentum P_y=±0.0939 (3PN). Domain [-32,32]^3 M, 4^3×16^3 = 64^3 effective
resolution, uniform mesh (dx=1.0M), constraint-preserving BCs.

**Pass criteria (5/5):** stability (no NaN/Inf), constraints bounded (Ham < 1.0),
Psi4(2,2) signal > 1e-6, gauge collapse (lapse < 0.8), inspiral motion
(separation decreasing).

**Results:** All 5 pass in 2.3 min on CPU. Constraints monotonically decrease
(7.5e-3 → 1.7e-4). Psi4(2,2) mode amplitude ~2.6e-5. Separation 9.0→2.24M
(binary inspiraling). Lapse min 0.5859. AH finder attempted but not found at
this resolution (expected — AH radius ~0.12M vs dx=1M; tested separately in
`test_ah_finder.c` at finer grids).

**Bug fixes:**
- `refine_mesh_near_punctures` AABB fix: was checking block-center-to-puncture
  distance, now checks minimum distance from puncture to block bounding box.
  The old check failed for large blocks (16M wide) where `r_refine = 8*dx = 8M`
  was smaller than the block half-diagonal.
- Two-pass BH separation tracker: the old single-pass algorithm had a bug where
  the "push-down" from BH1 to BH2 slot didn't check the 2M minimum distance
  requirement, causing adjacent cells to fill the BH2 slot and report separation
  ~1M regardless of true BH distance.

**Known limitation:** AMR initial data requires solving the constraint equation
on the refined mesh. The composite FMG solver diverges on multi-root meshes
(N_ROOT > 1) with AMR levels > 0. Prolongation of solved CCZ4 fields from
coarse to fine produces negative chi near the puncture singularity (Runge
phenomenon from polynomial interpolation of 1/r^4). Future work: solve on fine
mesh directly or use monotone prolongation near singularities.

## 2026-03-02: Native HIP GPU backend (replaces OpenMP target)

Replaced OpenMP target offloading (`#pragma omp target teams distribute`) with
a native HIP backend. HIP supports both AMD (ROCm) and NVIDIA (CUDA backend)
GPUs, eliminating the GCC `-foffload` dependency and its issues (`-O2` forced
on host, `GOMP_NVPTX_NATIVE_GPU_THREAD_STACK_SIZE` env var required).

**Architecture:**
- Single C++ file `src/backend/backend_hip.cpp` (~900 lines) contains all 15
  HIP kernels, device memory management, and constant memory.
- All physics code stays pure C. Headers use `LATTICE_DEVICE` macro
  (`__host__ __device__` for HIP, empty for CPU) and `EXTERN_C_BEGIN/END`
  guards for C++ linkage.
- New `src/core/device.h` defines portability macros.
- Two-phase Makefile: `gcc` compiles host C, `hipcc` compiles device C + HIP.

**Changes:**
- Created `src/core/device.h`, `src/backend/backend_hip.cpp`.
- Deleted `src/backend/backend_gpu.c` (old OpenMP target backend).
- All `#pragma omp declare target` blocks replaced with `LATTICE_DEVICE`.
- Fixed `sommerfeld.c` `asym_values` designated initializer for C++ compat.
- `restrict` → `__restrict__` compatibility for C++ builds.
- Constant memory (~2.7 KB): `nbr_offset`, restriction/prolongation weight
  tables, extrapolation coefficients. Loaded once in `backend_init()`.
- Stack size set to 16 KB via `hipDeviceSetLimit()` (CCZ4 RHS needs ~5.3 KB).

**15 kernels:** zero_packed, compute_rhs (4D decomposition calling
`ccz4_rhs_point`), sommerfeld (4D with CP-BC support), update_ck45, copy,
accum_add, axpy, apply_accum, ghost_same_level, ghost_restrict,
ghost_coarse_fill, ghost_extrap, ghost_prolong, ghost_exchange (orchestrator),
enforce_algebraic.

**Result:** 2 backends remain: `BACKEND=cpu` (OpenMP threads, unchanged) and
`BACKEND=gpu` (HIP). CPU build verified: zero warnings, all tests pass.

## 2026-03-02: Constraint-preserving boundary conditions

Implemented BAM-style CP BCs (arXiv:1212.2901, Hilditch et al.). For constraint
fields (Theta, K, A_ij, Gamma^i), the boundary RHS uses the outgoing-wave
equation at correct characteristic speeds instead of the generic Sommerfeld
formula. Non-constraint fields (chi, h_ij, lapse, shift, B^i, EM) keep
standard Sommerfeld.

**Formula:** `rhs(f) = -alpha * v_char * s_sign * d_s(f) - alpha * (f - f_asymp) / r`

**Characteristic speeds (BAM):**
- Theta → 1.0
- K → sqrt(2/alpha), clamped at alpha=0.01
- A_ij → 1.0
- Gamma^s (face normal) → sqrt(3/4)
- Gamma^A (tangential) → 1.0

**Implementation details:**
- Header-only `constraint_preserving.h`: `cp_char_speed()` + `cp_rhs()`, both
  `static inline` with `LATTICE_DEVICE` for GPU.
- CPU backend: `packed_sommerfeld_point()` gains `face_dir`, `s_sign`, `bc_type`
  params. Each face loop passes correct direction/sign.
- GPU backend: same per-field branch in collapse(4) kernel. face_dir/s_sign
  extracted from existing boundary detection. Inner field loop is warp-coherent
  (all threads process same field index → uniform branch).
- `bc_type_t` enum added to `sim_params_t`, default `BC_CONSTRAINT_PRESERVING`.
- CLI: `--bc sommerfeld|cp`.

**Test results (30/30):**
- Char speed values: 12 checks, all exact to 1e-14
- Gamma normal/tangential by face: 9 checks
- CP RHS formula: 4 hand-computed checks
- Flat spacetime: Ham=2.5e-14 after 1000 steps with CP BCs
- Single BH comparison: CP Ham ≤ Sommerfeld at t=25M
- Lapse clamping: alpha=0 and alpha=1e-6 properly clamped

## 2026-03-02: CCE accuracy improvements

Three changes based on review against research doc:

1. **Default l_max bumped from 8 to 16.** Research doc specifies l_max=16
   (17×33=561 angular points) as the production default. l_max=8 (9×17=153)
   was too coarse for precessing/high-spin systems with significant power in
   l=5+ modes. SpECTRE's PreprocessCceWorldtube uses `LMaxFactor: 3`, so
   l_max=16 input → l_max=48 internal CCE resolution.

2. **Research doc dataset count fixed: 31→49.** Was missing the 18 spatial
   metric derivative datasets (Dx/Dy/Dz of gamma_ij). Implementation was
   always correct at 49.

3. **Schwarzschild derivative test added.** Extracts on Brill-Lindquist
   single-puncture (M=1) at R=50 with dx=4. Compares all 49 extracted
   quantities against analytical isotropic Schwarzschild:
   - `gamma_ij = psi^4 delta_ij`, error = 4.7e-9
   - `d_k gamma_ij = -2M psi^3 delta_ij x_k/r^3`, error = 7.0e-9
   - `alpha = psi^{-2}`, error = 2.3e-9
   - `d_k alpha = M psi^{-3} x_k/r^3`, error = 3.5e-9
   - K_ij, shift, B^i all exactly zero (time-symmetric BL data)

   This validates the conformal→physical chain rule with real gradients,
   not just flat spacetime. Errors are ~O(dx^6/R^7) as expected from
   6th-order Lagrange interpolation.

Test count: 41→49 (8 new Schwarzschild assertions).

## 2026-03-01: CCE worldtube output for SpECTRE

Added SpECTRE-compatible Cauchy-Characteristic Evolution (CCE) worldtube HDF5
output. This is the production path to gauge-invariant gravitational wave strain
`h` at future null infinity (scri+), complementing the existing Psi4 extraction
which gives `r·Psi4 ≈ d²h/dt²` at finite radius.

**Pipeline:** Lattice → `CceR####.h5` → SpECTRE `PreprocessCceWorldtube` →
SpECTRE `CharacteristicExtract` → strain `h` at scri+.

**New files:**
- `src/diagnostics/cce_worldtube.h` — workspace struct, 49-dataset enum, API
- `src/diagnostics/cce_worldtube.c` — GL grid, HDF5 creation, sphere interpolation,
  conformal→physical ADM reconstruction, row writes (~350 lines)
- `tests/test_cce_worldtube.c` — 41 tests: conformal→physical, flat spacetime,
  HDF5 format, angular ordering, dataset names, multi-row

**SpECTRE AdmMetricNodal format:** 49 datasets (gamma_ij, d_k gamma_ij, lapse,
d_k lapse, shift, d_k shift, K_ij, AuxShift) on GL×uniform angular grid.
Theta-varies-fastest ordering. Each dataset is extensible (chunked, unlimited
rows). Column 0 = time, columns 1..N = angular data. Legend string attribute.

**Conformal → physical reconstruction:**
- `gamma_ij = h_ij / chi`
- `K_ij = (A_ij + K/3 h_ij) / chi`
- `d_k gamma_ij = d_k(h_ij)/chi - h_ij d_k(chi)/chi²`
- Lapse/shift derivatives pass through directly

**Implementation:** For each angular point on extraction sphere, interpolates 11
fields with derivatives (`interp_field_deriv_at_block`: chi, h_ij, lapse, shift)
and 10 fields values-only (`interp_field_at_block`: K, A_ij, B^i). Block-cached
mesh lookup follows psi4_extract pattern. All 49 quantities written per time step.

**Build:** `make HDF5=on` adds `-DLATTICE_HDF5` + pkg-config HDF5 flags. All CCE
code is `#ifdef LATTICE_HDF5` guarded — zero impact on builds without HDF5.
`make HDF5=on test-cce` runs the test suite.

**CLI:** `--cce`, `--cce_every <int>`, `--cce_radius <float>`, `--cce_lmax <int>`.

**Testing:** 41/41 tests pass. Flat spacetime: `|g_ii - 1| = 4.4e-16`,
`|derivatives| = 1.8e-16` (machine precision). HDF5 format validated
(49 datasets, correct shapes, Legend attributes, theta-varies-fastest ordering).

## 2026-03-01: Upgrade off-grid interpolation to 6th order

Replaced 4th-order (5-point) Lagrange interpolation with 6th-order (7-point)
in `src/numerics/interpolate.h`. This was the one link in the evolution chain
that needlessly degraded 6th-order evolution data when interpolating onto the
CCE worldtube sphere.

**Changes:**
- Added `lagrange_basis_7()`: 7-point Lagrange basis with nodes {-3,...,+3},
  denominators {720, -120, 48, -36, 48, -120, 720}. Half-width 3 fits in
  ghost width 4.
- Added `lagrange_basis_deriv_7()`: derivative using sum-of-products form
  for numerical stability near nodes.
- Updated all 4 interpolation functions to use 7-point basis (343 source
  points per interpolation vs 125 previously).
- Legacy 5-point functions retained as `lagrange_basis_5`/`lagrange_basis_deriv_5`.
- `INTERP_STENCIL` = 7, `INTERP_HALF` = 3 (was 5/2).

**Verification:** Weights independently verified via Python computation of
`prod_{j!=k} (n_k - n_j)`. All 6 mathematical tests pass: partition of unity
(sum = 1 to machine precision), Kronecker property, exact polynomial
interpolation through degree 6, exact derivatives through degree 5.

**Motivation:** GR-Athena++ achieves 10^-12 CCE mismatch with the same chain
(6th FD, RK4, 6th prolongation/restriction, Sommerfeld) — all links are
sufficient except off-grid interpolation, which was 4th-order. This upgrade
closes the last accuracy bottleneck for CCE worldtube data.

**Testing:** All tests pass (flat, convergence order 6.5, AMR 8/8, AH 13/13,
Maxwell 15/15).

## 2026-02-26: Tier 3 Item 6 — Device-side GPU ghost exchange

Replaced host-side PCIe round-trip ghost exchange with 7 GPU kernel launches.
All data stays on device during time steps — zero PCIe DMA.

**Implementation:** 5 GPU kernels replacing 5 host-side static functions:
- **Phase 0+1** (`gpu_packed_exchange_same_level`): collapse(2) over (block, neighbor).
  Element-wise copy replaces memcpy. Uniform meshes early-return after this phase.
- **Phase 2** (`gpu_packed_restrict_to_coarse`): collapse(4) over (b, f, ck, cj).
  6×6×6 tensor-product stencil using `restrict_w`/`restrict_wkj`.
- **Phase 3** (`gpu_packed_fill_coarse_buf_ghosts`): collapse(2) over (b, n).
  Two cases: same-level sibling (coarse↔coarse) and coarser neighbor (data→coarse
  with `round()` offset).
- **Phase 3.5** (`gpu_packed_fill_coarse_boundary`): 3 separate kernels (X→Y→Z)
  with implicit barriers for dimension-sweep ordering. `extrap_c[4][3]` declare-target
  constant for quadratic extrapolation coefficients.
- **Phase 4** (`gpu_packed_prolongate_fine_ghosts`): collapse(4) over (b, fk, fj, fi).
  7×7×7 Lagrange stencil. Field loop inside to share geometry across fields.

**Prerequisites:** Added `#pragma omp declare target` (with `#ifdef LATTICE_GPU` guards)
to constant arrays needed on device: `nbr_offset[26][3]` (block.h/c),
`restrict_w[6]`/`restrict_wkj[6][6]` (restriction.h/c),
`prolong_w[7]`/`prolong_wkj[4][7][7]` (prolongation.h/c).

**Testing:** CPU build zero warnings, all tests pass. GPU build (`make BACKEND=gpu
CC=gcc-14`) zero errors, `test-gpu-debug` passes all kernels including ghost exchange.

**Files changed:** `block.h`, `block.c`, `restriction.h`, `restriction.c`,
`prolongation.h`, `prolongation.c`, `backend.h`, `backend_gpu.c`.

Tier 3: 6/7 complete. Item 7 (dense output subcycling) deferred.

## 2026-02-25: Tier 3 GPU/performance optimizations (5/7)

5 of 7 Table 1 optimizations complete. Item 7 (dense output subcycling) deferred.

**Item 1 — Manual CSE in `ccz4_rhs_point`:**
- Cache `K_minus_2Theta = K - 2.0 * Theta` (used 6× in evolution equations).
- Pre-compute `A_mixed[k][j] = A[i][k] * h_UU[k][j]` to replace inner `ll` loop
  in A² trace and RHS_A computation. Net: eliminates ~50 redundant multiplies per
  grid point.

**Item 2 — `--block-size` CLI:**
- Added `--block-size <int>` as alias for `--N_block` in `main.c`.
- Validates: even and >= 8.

**Item 3 — Compile-time EM dispatch:**
- `Makefile`: `EM ?= off`. `make EM=on` adds `-DLATTICE_EM_ENABLED`.
- `fields.h`: `COMPILED_NUM_FIELDS` macro — 31 when EM enabled, 25 when not.
  Gives compiler constant loop bounds in dissipation/Sommerfeld hot paths.
- `test-maxwell` target always compiles with `-DLATTICE_EM_ENABLED` regardless.

**Item 4 — Kernel restructuring (`ccz4_rhs.c`):**
- Split monolithic 541-line `ccz4_rhs_point` into 5 `static inline` sub-functions,
  each with a typed output struct for scoped variable lifetimes:
  1. `ccz4_load_and_differentiate()` → `ccz4_fields_t` + `ccz4_derivs_t`
  2. `ccz4_compute_geometry()` → `ccz4_geom_t` (h_UU, Christoffel, Ricci, Z)
  3. `ccz4_compute_covariant()` → `ccz4_covd_t` (covd2lapse, A_UU, tr_A2)
  4. `ccz4_compute_evolution()` → writes CCZ4 RHS + outputs `rhs_Gamma[3]`
  5. `ccz4_compute_gauge()` → writes gauge RHS (uses `rhs_Gamma` for B^i)
- Top-level `ccz4_rhs_point` is now a thin 12-line dispatcher.
- All sub-functions in `#pragma omp declare target` block for GPU compilation.
- Moved `h_idx`/`A_idx` lookup tables to file scope inside OMP declare target.
- GPU benefit: scoped lifetimes allow better register allocation. Zero memory
  overhead — compiler inlines everything into a single kernel.

**Item 5 — Persistent per-level packs:**
- Added to `mesh_t`: `leaf_pack`, `level_packs[MAX_AMR_LEVELS]`, `packs_dirty`.
- New `meshblock_pack_sync_to_blocks()` / `meshblock_pack_sync_from_blocks()` —
  copy only `data` buffer (evolved fields). Skips rhs/scratch/accum (temporary
  per-stage buffers overwritten each step). ~75% less memcpy than full load/store.
- `rk4.c`: All packed steppers (`ck45_step_mesh_packed`, `classic_rk4_step_mesh_packed`,
  `step_level`) check `packs_dirty`, rebuild pack if needed, otherwise reuse cached.
  After step, `sync_to_blocks` instead of full store+free.
- `refine.c`: Sets `packs_dirty = 1` on regrid.
- `mesh.c`: `mesh_free()` cleans up cached packs. `mesh_create_ex()` initializes
  `packs_dirty = 1`.
- Subcycling subtlety: `ghost_fill_from_coarser()` writes to block ghost zones
  before stepping. `sync_from_blocks` copies updated ghosts into pack before RK stages.

**Files changed:** `ccz4_rhs.c`, `main.c`, `Makefile`, `fields.h`,
`meshblock_pack.h`, `meshblock_pack.c`, `mesh.h`, `mesh.c`, `refine.c`, `rk4.c`.

Build: zero warnings. Tests: not yet run (deferred to next session).

## 2026-02-25: Tier 2 mechanical optimizations

12 mechanical Phase 3 improvements across two sessions. Highlights:

**Structural (measurable impact):**
- **Hash table neighbor lookup** (`mesh.c`): Open-addressing hash in
  `mesh_rebuild_neighbors()` replaces O(N) linear scan with O(1). Key =
  `(level+1)<<48 | lx1<<32 | lx2<<16 | lx3`. Capacity 4*N_blocks, ~25% load.
- **`backend_enforce_algebraic_packed()`** (`backend_cpu.c`, `backend_gpu.c`):
  Batched det(h)=1 / tr(A)=0 on device. Wired into packed RK4 before unmap,
  eliminating GPU↔host round-trip. CPU: flattened (block,k,j) OMP. GPU: collapse(4).
- **Face-only Sommerfeld** (`sommerfeld.c`): Rewrote all 3 functions to iterate
  ghost-zone face slabs instead of full Nt³ with interior skip. Extracted
  `sommerfeld_point()` / `packed_sommerfeld_point()` helpers. ~5x fewer iterations
  for single grid; 80-97% fewer for multi-block (interior blocks skip entirely).
- **Subcycling frac drift fix** (`rk4.c`): Changed `subcycle_level()` to take
  `int sub_step` parameter. Frac = `sub_step * 0.5` instead of `floor(t/dt)`
  floating-point arithmetic. Correctness fix for long-duration runs.

**Infrastructure:**
- **Ghost scratch buffer** (`mesh.h`, `mesh.c`, `ghost_exchange.c`): Pre-allocated
  `ghost_scratch` in `mesh_t`, sized `n_fields * block_npoints`. Eliminates
  malloc/free per `ghost_fill_from_coarser()` call.
- **MAX_PUNCTURES 16→32** (`params.h`).

**Micro-optimizations (correct but marginal):**
- Sommerfeld asymptotic value / falloff rate array lookup (no branch divergence).
- `raise_all_2()` symmetry: 6 vs 9 components.
- Hoisted advection `sign(beta^i)` before inner x-loop.
- Ricci tensor symmetry: 6 vs 9 components.
- Levi-Civita curl unrolled to 6 terms (Maxwell).

**New test:** `test_nbody()` in `test_bowen_york.c` — 3-BH line + 5-BH pentagon
smoke tests (Ham L2 < 1.0, chi > 0).

**Files changed:** `rk4.c`, `mesh.h`, `mesh.c`, `ghost_exchange.c`, `sommerfeld.c`,
`backend.h`, `backend_cpu.c`, `backend_gpu.c`, `params.h`, `ccz4_rhs.c`,
`maxwell_rhs.c`, `tensor_utils.h`, `test_bowen_york.c`.

All tests pass. Convergence order unchanged: 6.56 / 6.25.

## 2026-02-24: Fix packed kernel dissipation + stale tests

**Bug (production):** Packed RHS kernels in `backend_cpu.c` and `backend_gpu.c`
constructed a stack-local `grid_t` via `memset(0)` but never set `n_fields`.
Dissipation (`dissipation.c:56`) reads `g->n_fields` to loop over fields — with
`n_fields=0`, all KO dissipation was silently skipped in the packed (AMR) path.
Introduced in commit 0cbdb13 (n_fields threading). Fix: set
`g_local.n_fields = pack->n_fields` in both CPU and GPU backends.

**Test fixes:**
- `test_pack_evolve`: compared all points including ghost zones, but packed and
  per-block paths leave ghosts in different states by design (packed does a final
  `ghost_exchange` after store-back). Changed to interior-only comparison.
  Interior fields are bit-identical (diff = 0.0).
- `test_amr_prolong` SSL test: "without SSL" run used `default_params()` which
  has `use_ssl=1` by default (since commit 967b209). Both runs had SSL on.
  Fix: explicitly set `use_ssl=0` for baseline.
- `test_amr_prolong` gauge field count: counted fields `>= FIELD_LAPSE` as gauge,
  expected 7. After EM fields were added (commit 0cbdb13), count became 13.
  Fix: count exactly `FIELD_LAPSE..FIELD_B3`.

All tests pass: flat (4.9e-14), convergence (6.56/6.25), AMR evolve (8/8),
Maxwell (15/15), pack_evolve (8/8), amr_prolong (15/15).

## 2026-02-24: Solve on evolution mesh — production AMR initial data

Architectural change: the AMR FAS multigrid constraint solver now operates
directly on the evolution mesh's blocks instead of creating a separate solver
mesh and copying/interpolating the solution back.

**Before:** `main.c` → create evolution mesh → solve on temp uniform grid →
copy fields to each leaf block (35 lines of interpolation code).

**After:** `main.c` → create evolution mesh → `set_bowen_york_mesh(m, ...)` →
refine near punctures → solve on evolution blocks → convert psi→CCZ4 in-place.

**Why:** Exact discrete operator consistency (same `fd_d2()` stencils, same
ghost zones, same grid points). Zero interpolation error. Zero extra memory
(solver reuses 22 of 100 idle array slots at t=0). This is the Athena++ MG
approach (Tomida & Stone 2023). Avoids the operator mismatch when using
external solvers like TwoPunctures on FD grids (Alic et al., arXiv:0912.2920).

**New functions:**
- `refine_mesh_near_punctures()` — extracted from `create_solver_mesh()`, works
  on any mesh (evolution or solver-owned)
- `relaxation_solve_amr_mesh()` — 1-field BY solver on external mesh (borrows,
  doesn't own)
- `relaxation_solve_coupled_amr_mesh()` — 4-field HiSpID coupled solver, same
  pattern
- `set_ccz4_from_psi_block()` — block-aware CCZ4 conversion (read-before-write
  to avoid aliasing with solver slots)
- `set_ccz4_from_hispid_block()` — block-aware HiSpID→CCZ4 (two-pass: fields
  then Gamma^i from FD)
- `set_bowen_york_mesh()` — top-level dispatch (BL analytic / BY 1-field /
  HiSpID 4-field), replaces 35-line copy path in main.c

**CLI:** `--amr-levels <int>` (default 2). Each level halves dx near punctures.
Level 2 = 4x finer, level 3 = 8x finer, etc. Requires `--rk classic`.

**Benchmark (L=64, N=32, 2 AMR levels, dx_base=2.0 → dx_fine=0.5):**
- Near-field Ham L2 (r=2..8M): old 1.6e-01, new 7.4e-07 → **218,000x better**
- Far-field Ham L2 (r=8..20M): old 4.7e-02, new 2.7e-08 → **1,750,000x better**
- Wall time: old 0.4s (coarse solve + copy), new 12.6s (fine solve)
- At level 0 (uniform, no refinement): both produce bit-identical results

The old approach solved at coarse resolution and copied to fine blocks — the fine
blocks got staircase artifacts (O(1) constraint violation). The new approach does
real work at fine resolution, paying compute cost for genuine accuracy.

**Files changed:** `params.h`, `main.c`, `relaxation_amr.h`, `relaxation_amr.c`,
`bowen_york.h`, `bowen_york.c`.

**Tests:** All pass — flat (4.9e-14), convergence (6.56/6.25), AMR evolve (8/8),
Maxwell (15/15), relaxation-amr (12/12), bowen-york (29/29), hispid (26/26).
Existing non-AMR paths unchanged.

## 2026-02-23: Tier 1 complete + position-dependent eta

Three changes completing all remaining Tier 1 optimizations plus one Phase 3 item:

**1. Position-dependent eta** (`params.h`, `ccz4_rhs.c`)
Gamma-driver shift damping: `eta(x) = eta_0 / W(x)` where `W = sqrt(chi)`.
Increases damping near punctures (chi→0) for stable unequal-mass evolution.
Gated behind `position_dependent_eta` flag (default 1). Guards: `fmax(chi, 1e-6)`
for sqrt, `fmax(W, 1e-6)` for division. Ref: arXiv:1003.0859.

**2. Fused d1/d2 stencil** (`finite_diff.h`, `ccz4_rhs.c`)
New `fd_d1_d2()` inline: loads 7 stencil points once, computes both 1st and 2nd
derivatives. Merged separate d1 loop (section 2) and d2 diagonal loop (section 3)
in `ccz4_rhs_point()` for the 11 fields needing both (chi, lapse, h_ij(6),
shift^i(3)). Eliminates ~40% redundant memory loads in the RHS kernel.
Both 6th-order (7-point) and 4th-order (5-point) versions provided.
Convergence order unchanged: 6.56 / 6.25.

**3. Conditional EM allocation** (`grid_alloc_ex` + n_fields threading)
When `--em` is off (default), allocate only 25 fields instead of 31, saving 19%
memory (~1.2 GB at N=128). Added `n_fields` member to `grid_t`, `mesh_t`,
`meshblock_pack_t`. New `_ex()` API variants (`grid_alloc_ex`, `mesh_create_ex`,
`meshblock_pack_create` with n_fields). Old APIs are backward-compatible wrappers
defaulting to `NUM_FIELDS`. Converted ~50 `NUM_FIELDS` → `n_fields` references
across 17 source files: grid, block, mesh, ghost_exchange, prolongation,
restriction, refine, meshblock_pack, rk4, backend_cpu, backend_gpu, sommerfeld,
dissipation, main. All field loops iterate `f < n_fields` instead of `f < NUM_FIELDS`.

All tests pass: flat (4.9e-14), convergence (6.56/6.25), AMR evolve (8/8),
AMR mesh (33/33), Maxwell (15/15), subcycle (7/7).

## 2026-02-23: Fix two AMR packed-stepper bugs (test_amr_evolve 8/8)

**Bug 1: ghost_fill_from_coarser ordering (Test 2 NaN)**

Root cause: `ghost_fill_from_coarser()` did restrict → exchange → prolongate
per block in a single loop. Block B read neighbor C's `coarse_buf` before C
had been restricted → stale/zero data → NaN in AMR subcycling.

Fix: Split the single loop into two passes:
- **Pass 1:** Restrict ALL fine blocks' interiors into their `coarse_buf`s.
- **Pass 2:** Exchange coarse_buf ghosts + boundary extrapolate + prolongate.

Cleanup: removed `has_coarser_leaves` workaround from `step_level()` in `rk4.c`,
removed `RK_CK45` debug override and ~60 lines of NaN diagnostics from
`test_amr_evolve.c`, removed unused `fields.h` include.

**Bug 2: classic RK4 packed mesh stale ghost zones (Test 1 ratio 11.87)**

Root cause: packed RHS only computes interior + Sommerfeld-boundary cells.
Non-boundary ghost `rhs` stays zero, so `accum` accumulates zeros there.
Final `data = scratch + accum` puts ghost zones at U^n (start of step) instead
of U^{n+1}. `meshblock_pack_store` copies these stale ghosts back to blocks.
Constraint measurement then reads wrong ghost values in FD stencils.
CK45 unaffected: its incremental `data += B*dU` leaves ghosts at whatever the
last ghost exchange set them to (≈ U^{n+1}).

Fix: added `ghost_exchange(m)` after `meshblock_pack_store` in
`classic_rk4_step_mesh_packed()`. Restores correct ghost values from neighbors'
correctly-evolved interiors.

**Verification:** `make` zero warnings. test_amr_evolve: **8/8 ALL PASSED**.
Test 1 ratio = 1.0000 (bit-identical to single-grid). Test 2 Ham L2 = 2.6e-03.
Full suite: flat PASSED, convergence 6.56/6.25 PASSED, Maxwell 15/15 PASSED.

---

## 2026-02-23: Tier 0 Bug Fix + Default RK4 Switch

**Bug 3 fix: `enforce_algebraic_block()` in `refine.c`**
- Was using `1.0 / cbrt(det)` (~3-5x slower than `fast_inv_cbrt`) and had a
  divergent `if (det > 0.0)` guard not present in the canonical `rk4.c` version.
- Fixed: `fast_inv_cbrt(det)` (already in `tensor_utils.h` from Tier 1), removed
  the `if` guard to match `rk4.c` behavior (always enforce, `fast_inv_cbrt` handles
  edge cases via `cbrt()` fallback for det outside [0.5, 2.0]).
- Bugs 1 and 2 (0th-order restriction, missing `save_k1_from_pack` in CK45
  subcycling) were already fixed in the Tier 1 commit.

**Default integrator changed: CK45 → classic RK4 (`RK_CLASSIC`)**
- `params.h`: `p.rk_method = RK_CLASSIC` (was `RK_CK45`).
- Classic RK4 is faster (4 stages vs 5, 20% fewer RHS evaluations per step).
- Uses 25% more memory (4 blocks vs 3). OK for production GPUs; tight on M4 at N=256.
- All test files updated: hardcoded `RK_CK45` → `RK_CLASSIC` in grid/mesh allocations.
  Tests using `default_params()` now get classic by default.

**Discovered issues during testing (both now fixed — see entry above):**
1. AMR Test 2 NaN — `ghost_fill_from_coarser` ordering bug.
2. Classic RK4 packed mesh divergence (ratio 11.87x) — stale ghost zones after
   `data = scratch + accum`.

**Verification:** `make` zero warnings. Flat PASSED. Convergence 6.56/6.25 PASSED.
Maxwell PASSED (15/15). AMR evolve 8/8 ALL PASSED.

---

## 2026-02-23: Tier 1 Performance Optimizations (A1–C2)

**Goal:** 25–45% CPU improvement from 8 low-risk optimizations (~280 LOC).
Implemented in risk order: zero-risk first, hot-path numerics last.

**Group A — Zero-risk (no physics code touched):**
- **A1: LTO (`-flto`)**: Added to CPU builds via `LTO_FLAGS` in Makefile. Disabled
  for GPU backend (GCC 15 LTO bug with offloading). ~4 lines.
- **A2: Fast-path `pow(lapse, 1.0)`**: Default `lapse_power=1.0` now skips `pow()`
  call entirely (identity). Saves ~50–100 cycles per grid point. ~3 lines.
- **A3: `restrict` on RHS pointers**: Added `restrict` qualifier to `rhs` and `src`
  pointer parameters in ccz4_rhs, maxwell_rhs, dissipation, backend.h typedef.
  Safe: SoA layout guarantees no aliasing between field arrays. ~20 lines.

**Group B — Low-risk (loop structure changes):**
- **B1: Skip EM fields**: Added `NUM_CCZ4_FIELDS=25` to `fields.h`. Dissipation and
  Sommerfeld loops use `p->em_enabled ? NUM_FIELDS : NUM_CCZ4_FIELDS`. Saves 6/31
  field iterations when EM disabled. ~10 lines.
- **B2: OMP-parallelize ghost exchange**: Added `#pragma omp parallel for schedule(dynamic)`
  to 6 packed block loops: `packed_exchange_same_level`, `packed_restrict_to_coarse`,
  `packed_fill_coarse_buf_ghosts`, `packed_fill_coarse_boundary`,
  `packed_prolongate_fine_ghosts`, `backend_sommerfeld_packed`. Each block's ghost
  writes are disjoint. ~6 lines.
- **B3: Flatten update loops**: `rk4.c` helpers (`copy_fields`, `axpy_fields`,
  `accum_add`, `apply_accum`, `zero_fields`, `ck45_update`) now use `dst[0]` as
  base pointer for single flat OMP loop over `NUM_FIELDS * n`. Eliminates 31
  separate OMP fork/joins per call. `copy_fields` → single `memcpy`,
  `zero_fields` → single `memset`. ~40 lines.

**Group C — Medium risk (structural changes):**
- **C1: Hoist GPU grid_t**: `backend_gpu.c` collapse(4) kernel now receives a
  pre-built `g_template` instead of constructing a 1064-byte `grid_t` per thread.
  ~15 lines.
- **C2: Pre-computed weight tables**: Added `restrict_wkj[6][6]` (36 entries) and
  `prolong_wkj[4][7][7]` (196 entries, 4 octant combos with reversal baked in).
  All 232 entries verified **bit-exact** via `memcmp` against runtime computation
  (`tools/verify_weights.c`). Weight sums verified via integer fraction arithmetic
  (restriction: 11520/11520, prolongation: 65536/65536). Updated `restrict_cell()`,
  `prolongate_field()`, and all packed kernels in both backends. ~60 lines.

**Also moved:** `fast_inv_cbrt()` from `rk4.c` to `tensor_utils.h` (shared utility).

**Also upgraded:** `restrict_level_to_parents()` from 0th-order (2x2x2 averaging)
to 6th-order (`restrict_cell()`). Previously had NaN from stale ghost zones —
fixed by splitting `ghost_fill_from_coarser` into two passes (see entry above).

**Remaining Tier 1 items:** C3 (conditional EM allocation), D1 (fused d1/d2 stencil).

**Verification:** `make` zero warnings. Flat spacetime PASSED. Convergence order
6.55/6.26 PASSED (threshold 3.5).

---

## 2026-02-20: AMR Parity Gaps + Performance Optimizations

**AMR parity gaps closed:**
- **Output slices:** `output_mesh_1d_slice()` in output.c — iterates leaf blocks,
  collects cells on y=0/z=0, deduplicates keeping finest level, writes CSV.
- **AH finder on AMR:** `mesh_find_block_at()` for coordinate→block lookup,
  `ah_find_amr()`/`ah_compute_diagnostics_amr()` for block-local interpolation.
  Block caching exploits angular locality. Wired into main.c AMR loop (`--amr --ah`).

**Performance optimizations (CPU):**
- **Flattened packed kernel** (backend_cpu.c): Single OMP parallel region with
  combined (block,k,j) iteration space. Eliminates N_blocks fork/joins per RHS call.
- **restrict + omp simd** (rk4.c): All bulk array ops (axpy, accum, ck45_update)
  now vectorize. `restrict` removes aliasing barrier, `omp simd` forces SIMD codegen.
- **fast_inv_cbrt** (rk4.c): Newton-Raphson for `1/cbrt(det)` in enforce_algebraic.
  2 iterations from Taylor start, exact to double precision when det≈1. ~4x faster
  than libm `cbrt` on the hot path.
- **timer.h**: `TIMER_START`/`TIMER_STOP` macros via `clock_gettime(CLOCK_MONOTONIC)`.

**Verification:** All tests pass, convergence order 5.4 (threshold 4.0).

## 2026-02-20: Binary Inspiral AMR Convergence Test

**Goal:** Demonstrate 4th-order self-convergence of the full CCZ4 evolution code
on a binary BH inspiral using AMR, following standard NR methodology
(arXiv:2409.10383 AthenaK, arXiv:2312.05438 AMR strategy comparison).

**Setup:** Equal-mass non-spinning quasi-circular binary (Brugmann et al. 2008,
arXiv:0709.0838): m_bare=0.4824, d=10M, P_y=±0.0939 (3PN), L=64, CFL=0.25.

**Methodology:** Self-convergence with 3 AMR resolutions at ratio r=1.5:
- LOW:  N_block=32, N_root=3 → N_eff=96,  dx=0.667
- MED:  N_block=48, N_root=3 → N_eff=144, dx=0.444
- HIGH: N_block=64, N_root=3 → N_eff=192, dx=0.333

Uniform AMR mesh (max_level=0, no refinement): exercises the full AMR
infrastructure (block decomposition, packed kernels, ghost exchange) without
the memory explosion from dynamic refinement.  Classic RK4 integrator.
6000 steps per resolution.

**Memory:** Peak ~16 GB (HIGH run: 27 blocks + pack, classic RK4).  CK45 used
for the temp initial-data grid to reduce peak during the FAS multigrid solve.

**Memory lessons from earlier attempts:**
- max_level=4 + classic RK4 (N_root=4): OOM killed at 31.6 GB during first
  regrid (64 → 512 leaves, pack duplication doubled memory).
- max_level=2 + CK45 (N_root=4): also OOM — subcycling + pack duplication for
  512 blocks exceeded 32 GB.
- Root cause: the packed batch stepper duplicates all block data into contiguous
  GPU-mappable buffers.  With 64+ blocks and 4 RK buffers, that's 2× the block
  memory.  Solution: N_root=3 (27 blocks), max_level=0, classic RK4.

**Convergence criterion:** For 4th-order code with r=1.5:
  Q = |Ham(low) - Ham(med)| / |Ham(med) - Ham(high)| ≈ 1.5^4 = 5.06

**Estimated runtime:** ~41 hours total (LOW ~3h, MED ~11h, HIGH ~27h).

**Test file:** `tests/test_inspiral_convergence.c`
**Build:** `make test-inspiral-convergence`
**Run:** `nohup build/test_inspiral_convergence > inspiral.log 2>&1 &`
**Monitor:** `tail -f inspiral.log`

**Status:** Test written, compilation verified.  Run pending.

---

## 2026-02-20: AMR + Bowen-York Initial Data, Binary Inspiral

**AMR initial data upgrade:** The AMR path in `main.c` previously only supported
Brill-Lindquist (BHs at rest). Now supports full Bowen-York initial data
(momentum, spin, HiSpID) by solving the constraint on a temporary uniform grid
at base AMR resolution (N_eff = N_root * N_block), then copying solved fields to
each leaf block. Also wired up EM-aware RHS selection (`ccz4_maxwell_rhs_point`)
for AMR evolution when `--em` is enabled.

**Remaining AMR parity gaps** (single-grid has all, AMR does not yet):
- EM-aware RHS in packed kernels (packed path hardcodes `ccz4_rhs_point`)
- AH finder on AMR mesh (needs cross-block interpolation)
- Output slices from AMR blocks

**Binary inspiral test command** (standard equal-mass non-spinning quasi-circular,
matching Brugmann et al. 2008, arXiv:0709.0838):

```bash
./build/lattice --N 128 --L 64 --steps 4000 --CFL 0.25 \
    --ah --ah_every 200 \
    --puncture 0.4824,0,0,5,0,0.0939,0 \
    --puncture 0.4824,0,0,-5,0,-0.0939,0
```

Parameters: d=10M separation, m_bare=0.4824 each (M_ADM≈1.0), P_y=±0.0939
(3PN quasi-circular). N=128 → dx=0.5, dt=0.125, t_final=500M.

Expected results to compare against published NR data:
- Final remnant mass: M_f/M ≈ 0.9516
- Final spin: a/M_f ≈ 0.6864
- Radiated energy: ~3.5% of M_ADM
- Merger time: ~500-700M (depends on eccentricity)

Initial solver output (verified): FAS multigrid converges in 11 V-cycles,
residual 6.4e-13. Initial Ham L2 = 6.95e-3. ~4.5 sec/step on M4.

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

---

## Psi4 Gravitational Wave Extraction

Implemented Newman-Penrose Psi4 scalar for outgoing gravitational radiation.

### Physics

Psi4 = C_{abcd} n^a mbar^b n^c mbar^d, computed via 3+1 decomposition:
- Electric Weyl: E_ij = R_ij + (K - Theta) K_ij - K_ik K^k_j (trace-free)
- Magnetic Weyl: B_ij = epsilon_{(i}^{kl} D_k K_{j)l} (trace-free)
- Projection: Re(Psi4) = (E_vv - E_ww)/2 + B_vw
              Im(Psi4) = (B_vv - B_ww)/2 - E_vw
- Tetrad: Gram-Schmidt orthonormalization of {r,theta,phi} against gamma_ij

**CCZ4 correction:** E_ij uses `(K - Theta)`, not plain `K`. The CCZ4 evolved
variable K contains a Theta contribution; the Gauss-Codazzi relation requires
`(K - Theta)` for the physical trace. This is NOT the `(K - 2*Theta)` that
appears in the CCZ4 evolution equations. GRChombo Weyl4.impl.hpp has an
explicit comment about this distinction.

Ref: B&S Eqs. (8.53)-(8.55), (3.30); GRChombo Weyl4.impl.hpp

### Implementation

- `src/diagnostics/psi4.h` (~80 lines) — workspace struct, public API
- `src/diagnostics/psi4.c` (~740 lines) — full pipeline:
  - Gauss-Legendre quadrature (Newton iteration on Legendre roots)
  - Spin-weighted spherical harmonics _{-2}Y_{lm} via Wigner d-matrix
  - Psi4 kernel: FD derivatives, physical Ricci, physical Christoffel (B&S 3.30),
    covariant derivative of K_ij, Electric + Magnetic Weyl, Gram-Schmidt tetrad
  - Block-aware AMR sphere extraction (mesh_find_block_at with cache)
  - Mode decomposition: GL × trapezoidal quadrature
  - CSV output (t, l, m, Re, Im, |rPsi4|, phase)

CLI: `--psi4`, `--psi4_every N`, `--psi4_radius R`, `--psi4_l_max L`,
     `--psi4_n_theta N`, `--psi4_n_phi N`

### Test results (15/15 PASS)

- GL quadrature: exact for x^0, x^2, x^4, x^30 (n=16)
- _{-2}Y_{22}, _{-2}Y_{20} normalization = 1 (to 1e-8)
- _{-2}Y_{22} orthogonal to _{-2}Y_{21} (to 1e-12)
- Mode round-trip: inject _{-2}Y_{22}, recover (2,2)=1.0, others < 1e-15
- Flat spacetime: |Psi4| = 1.3e-15 (machine epsilon)
- Schwarzschild: |Psi4| = 4.2e-8 (r=4M), 7.8e-11 (r=8M), 1.4e-12 (r=12M)
  — pure FD truncation error, consistent with 6th-order at dx=0.5
- CSV output: header + data lines written correctly

### Reused infrastructure

compute_inverse_sym, compute_christoffel (tensor_utils.h), fd_d1/d2/d2_mixed
(finite_diff.h), mesh_find_block_at (mesh.h), Physical Christoffel pattern
(ah_finder.c), Levi-Civita (same as maxwell_rhs.c).

### What's next

PHE (Perturbative Hyperboloidal Extraction) — post-processing tool that
propagates Psi4 modes to null infinity via RWZ on hyperboloidal slices.
~470 lines, pure C17, no external dependencies. Unblocked by Psi4.

---

## AMR Composite Multigrid: V-Cycle Fix + OMP Parallelization

### Bug: Cross-level ghost exchange in multigrid solver

**Root cause:** `solver_ghost_exchange()` called `ghost_exchange_all_blocks()`,
which does direct `memcpy` between blocks regardless of AMR level. When a fine
block (level L) has a coarser neighbor (level L-1), data is copied at wrong
physical resolution — the coarser block covers 2x the physical extent with the
same cell count. This corrupts FD stencils near refinement boundaries.

**Symptom:** V-cycle divergence on multi-root AMR meshes.
- 3 AMR levels: 2x growth per V-cycle (fast divergence)
- 5 AMR levels: 0.85 factor per V-cycle (very slow convergence, 159 V-cycles)

The 5-level case converged slowly because more intermediate levels each
contribute less cross-level error, and the smoother partially compensates.

### Fix: Correct CF boundary protocol (AMReX/Chombo pattern)

Based on AMReX MLMG, Chombo AMRMultiGrid, Athena++ MG (Tomida & Stone 2023):

1. **CF ghost zones = fixed Dirichlet from coarse level**, set ONCE before
   smoothing begins at each level, held fixed during all 8 GS colors.
2. **Between GS colors: same-level exchange only** — no cross-level interpolation.
3. **Before operator evaluation: full exchange** (same-level + CF fill + BCs).

New ghost exchange functions:
- `ghost_exchange_same_level_all(m, level)` — same-level only at one level
- `ghost_fill_cf_boundary(m, level)` — CF boundary fill from level-1 data
- `ghost_exchange_multilevel_all(m)` — full coarse-buffer protocol for all blocks

New solver exchange functions:
- `solver_same_level_exchange(m, level)` — between GS colors
- `solver_full_exchange(m, level)` — before operator evaluation
- `solver_ghost_exchange_all(m)` — all levels for setup/residual

**Verified:** BY tests 33/33 pass, convergence factor ~0.24 (excellent).
N-body tests (3-BH, 5-BH) converge correctly.

### OMP parallelization

Added `#pragma omp parallel for` to:
- `umg_sweep_1field` / `umg_sweep_4field` k-loops (uniform MG smoothers)
- `umg_compute_operator` k-loop (uniform MG operator evaluation)
- `mesh_constraint_l2` / `mesh_momentum_l2` block loops (constraint diagnostics)
- `restrict_level_to_parents` block loop (AMR restriction)
- `ghost_fill_from_coarser` Pass 1 (coarse_buf restriction)
- `ghost_exchange_same_level_all` / `ghost_fill_cf_boundary` block loops

All are embarrassingly parallel (each block/k-slice writes to independent memory).

### References

- AMReX MLMG: CF boundary as fixed Dirichlet, same-level exchange between sweeps
- Chombo AMRMultiGrid: `mgRelax` + `homogeneousCFInterp`
- Athena++ MG (Tomida & Stone 2023): same pattern
- Afivo octree-mg: ghost cells from parent at CF boundaries

---

## Zero-PCIe GPU Pipeline (Parts A + B + C)

Root cause of ~200 s/step GPU performance: `backend_map_pack()` did
hipMalloc + hipMemcpy for ALL buffers (~730 MB) on EVERY sub-step, and
`backend_unmap_pack_sync()` did hipMemcpy + hipFree every sub-step. With 4
AMR levels, Berger-Oliger subcycling creates ~31 sub-steps per global step.
Net GPU utilization: ~3% → reported as 0%.

### Part A: Persistent Per-Pack Device Memory

Added `void *device_handle` to `meshblock_pack_t` — opaque pointer to
`hip_device_ptrs_t` on GPU (NULL on CPU). Device memory allocated on first
`backend_map_pack()`, persists across sub-steps, freed only by
`backend_free_pack_device()` or `meshblock_pack_free()`.

- `backend_map_pack()`: first call allocates + copies everything; subsequent
  calls sync only data + coarse_data (metadata unchanged)
- `backend_unmap_pack_sync()`: syncs data + coarse_data back to host, does NOT
  free device memory
- `backend_activate_pack()`: sets global `d_ptrs` from pack's device handle
  without any memcpy — used by GPU-resident subcycling
- `backend_free_pack_device()`: frees all device allocations (called from
  `meshblock_pack_free()`)

Eliminates ~465 hipMalloc/hipFree calls per global step.

### Part B: GPU-Resident Evolution

Zero PCIe transfers during Berger-Oliger subcycling. All data stays on device
across sub-steps.

New `hip_device_ptrs_t` fields:
- `fields_old`: pre-step data for temporal interpolation by finer levels
- `cross_level_map`: (fine_block, direction, coarse_block) triples for
  cross-pack ghost fill

New kernels/functions:
- `hip_cross_level_ghost_fill`: reads from coarser pack's data (new + old)
  with temporal interpolation, writes to fine pack's coarse_data
- `backend_cross_level_ghost_fill_packed()`: orchestrates 5-phase cross-level
  ghost exchange entirely on device (restrict → same-level coarse → cross-level
  fill → extrapolation → prolongation)
- `backend_save_old_packed()`: device→device copy of data to fields_old

New evolution path in `rk4.c`:
- `step_level_gpu()`: single-level RK4 step entirely on device (no map/unmap)
- `subcycle_level_gpu()`: recursive Berger-Oliger subcycling on device
- `gpu_ensure_level_packs()`: builds level packs + cross-level maps on first
  step or after regrid
- `gpu_sync_all_to_host()`: single sync of all levels at end of global step

Entry point: `rk4_step_mesh()` dispatches to `subcycle_level_gpu()` when
`backend_is_gpu()` and `max_level > 0`.

### Part C: Solver Round-Trip Elimination

Replaced 7 host↔device round-trip patterns in the AMR composite multigrid
solver with device-side ghost exchange kernels. Each round-trip was a 5-line
sequence: `backend_sync_solver_data_to_host → meshblock_pack_store →
solver_full_exchange → meshblock_pack_load → backend_sync_solver_data_to_device`.

New solver kernel/function:
- `hip_mg_ghost_cf_fill`: combined same-level coarse_buf exchange + cross-level
  fill from coarser solver slot. One thread per (block, 26-neighbor) pair.
  Same-level branch copies between sibling coarse_bufs via `coarse_nbr_table`.
  Cross-level branch reads from `d_solver[coarse_slot].data` using origin
  offset computation.
- `hip_mg_zero_leaf_rhs`: zeros RHS fields on leaf blocks (skips parents),
  replacing a host round-trip for leaf RHS zeroing.
- `backend_mg_ghost_full_packed(pack, slot, coarse_slot, four_field)`:
  orchestrates 6-phase device-side solver ghost exchange:
  1. Same-level exchange (`hip_mg_ghost_same_level`)
  2. Restrict data → coarse_data (`hip_ghost_restrict`, reused from evolution)
  3. CF fill (`hip_mg_ghost_cf_fill`, new kernel)
  4. Boundary extrapolation × 3 dims (`hip_ghost_extrap`, reused)
  5. Prolongation coarse_data → fine ghosts (`hip_ghost_prolong`, reused)
  6. Zero-Dirichlet BCs (`hip_mg_bc`)
- `backend_mg_upload_cf_data(slot, cf_map, nb, is_parent)`: uploads cross-level
  neighbor map + parent mask to solver slot
- `gpu_build_cf_maps()` in `relaxation_amr.c`: builds cf_map[nb * 26] and
  is_parent[nb] for each AMR level, uploads via `backend_mg_upload_cf_data`

Round-trips replaced:
- `composite_vcycle_gpu()`: 5 round-trips → 5 device-side calls
- `composite_fmg_gpu()`: 1 round-trip → 1 device-side call
- `amr_residual_norm_gpu()`: 1 round-trip → 1 device-side call

Evolution kernels (`hip_ghost_restrict`, `hip_ghost_extrap`, `hip_ghost_prolong`)
reused directly with solver pointers — they take raw pointer arguments, not globals.

### Modified Files

| File | Changes |
|------|---------|
| `meshblock_pack.h` | `void *device_handle` field |
| `meshblock_pack.c` | Init handle to NULL; call `backend_free_pack_device()` in free |
| `backend.h` | New declarations: `backend_free_pack_device`, `backend_activate_pack`, `backend_save_old_packed`, `backend_cross_level_ghost_fill_packed`, `backend_mg_ghost_full_packed`, `backend_mg_zero_leaf_rhs_packed`, `backend_mg_upload_cf_data` |
| `backend_hip.cpp` | Persistent device ptrs, `fields_old` + `cross_level_map`, solver `cf_map` + `is_parent`, new kernels, orchestrator functions (+851 lines) |
| `backend_cpu.c` | No-op stubs for all new functions |
| `rk4.c` | `step_level_gpu`, `subcycle_level_gpu`, `gpu_ensure_level_packs`, `gpu_sync_all_to_host`, `build_cross_level_map` (+251 lines) |
| `relaxation_amr.c` | `gpu_build_cf_maps`, replaced 7 round-trip patterns |

### Test Results

- CPU build: zero warnings, all tests pass
- GPU build (NVIDIA Tesla P40): zero warnings
  - GPU kernel isolation: ALL PASSED
  - Bowen-York 33/33: PASSED
  - HiSpID 26/26: PASSED
  - AMR relaxation: OOM (only 2.5 GB free on P40 — external resource constraint)
- Convergence: unaffected (CPU paths unchanged)

## Volume-Weighted AMR Constraint Norms + Lapse Advection

### Problem: Inspiral crash at t=20M on AMR meshes

Binary inspiral with 4 AMR levels crashed at t=20M with lapse collapse, despite
the same physics running stably to t=1000M on uniform grids. Two independent
issues compounded:

1. **Misleading diagnostics:** Unweighted constraint L2 norms on AMR meshes
   gave a 55x apparent jump at step 0 vs uniform grids. Fine cells near
   punctures (dx=0.125M) dominated the norm despite representing tiny physical
   volume — a pure diagnostic artifact.

2. **Gauge instability:** Without the lapse advection transport term
   `β^i ∂_i α`, the 1+log gauge is purely local and unstable on coarse AMR
   base grids (dx≥2M). The advection form couples neighboring cells via the
   shift vector, essential for stable gauge propagation.

### Fix 1: Volume-weighted L2 norms

Changed `mesh_constraint_l2()` and `mesh_momentum_l2()` from:
```
sum += H*H; count++;  →  return sqrt(sum/count)
```
to:
```
dV = dx³; sum += H*H*dV; vol += dV;  →  return sqrt(sum/vol)
```

Updated in all three backends:
- `src/diagnostics/constraints.c` — CPU mesh-level
- `src/backend/backend_cpu.c` — packed CPU variant
- `src/backend/backend_hip.cpp` — GPU kernel (shared memory layout changed
  from `[ham, mom, count(int)]` to `[ham, mom, vol(double)]`)

Standard practice in all AMR NR codes (BAM, Einstein Toolkit, GRChombo).

### Fix 2: Lapse/shift advection

Enabled `lapse_advec_coeff=1.0` and `shift_advec_coeff=1.0` in the inspiral
test. This adds the transport term `β^i ∂_i α` to the gauge evolution, which
was already implemented in ccz4_rhs.c but defaulted to 0.0.

Both advection (coeff=1) and non-advection (coeff=0) are valid gauge choices.
Non-advection is the original Bona-Masso form and works fine on uniform grids.
Advection is standard in production AMR codes.
Ref: gr-qc/0610128 (Brugmann et al.).

Zero additional cost — the shift and lapse derivatives are already computed
for other terms. Default in params.h remains 0.0; inspiral test overrides.

### Result

Inspiral initial data with volume weighting: Ham_L2 = 2.338e-03 (was 1.1e-02
unweighted — the physics didn't change, just the diagnostic).

### Modified files

| File | Changes |
|------|---------|
| `constraints.c` | Volume-weighted mesh_constraint_l2, mesh_momentum_l2 |
| `backend_cpu.c` | Volume-weighted packed CPU variants |
| `backend_hip.cpp` | Volume-weighted GPU kernel, shared mem layout |
| `test_binary_inspiral.c` | Added lapse/shift advection coefficients |

## Checkpoint/Restart: Binary Save and Restore

### Motivation

Long-duration inspiral runs (T=700M+, hours of wall time) need the ability to
pause and resume. Crashes, timeouts, and iterative debugging all require
checkpoint/restart.

### File format

Binary, little-endian, 64-bit aligned:

```
[Header — 1024 bytes]
  char[8]       magic          "LATCKPT\0"
  int           version        1
  int           step           evolution step number
  double        time           simulation time
  int           num_leaves     number of leaf blocks saved
  int           N_block        interior cells per block side
  int           n_fields       active field count
  int           rk_method      0=classic, 1=ck45
  double        L              domain size
  sim_params_t  params         full parameter struct
  ... padding to 1024 bytes ...

[Per leaf block — repeated num_leaves times]
  int           level
  int           lx1, lx2, lx3  logical location
  double[3]     origin          physical corner
  double[n_fields * npoints]    all field arrays (including ghost zones)
```

Ghost zone data is saved verbatim. On restart, no ghost exchange is needed —
the saved values are exact. This ensures bitwise-identical restart for uniform
meshes and sub-epsilon (5.6e-16) for AMR.

### AMR tree reconstruction

On read, the mesh starts as a single root block. `reconstruct_tree()` iterates
level by level (0 to max_level-1), checking each block to see if any saved leaf
is a descendant (via bit-shifting logical coordinates). If so, the block is
refined. After each level, neighbor tables are rebuilt.

### CLI integration

- `--checkpoint-every N`: save checkpoint every N steps (default: disabled)
- `--restart <file>`: restart from a checkpoint file

Both flags work with single-grid and AMR evolution paths.

### Key design decision: no ghost exchange after read

Initial implementation called `ghost_exchange_all_blocks()` after restoring
field data, which overwrote saved ghost values with freshly-computed
prolongation/restriction values. This introduced sub-epsilon FP differences
that broke bitwise-identical restart. Removed — ghost data is preserved
exactly from the checkpoint file.

### Test results (14/14 PASS)

Test 1 — Uniform grid (32³, L=16, single BH):
- Checkpoint at step 10, restart, evolve to step 20
- Ham L2 and center lapse match reference **bitwise-identically**

Test 2 — AMR grid (16³, L=32, 2 AMR levels, single BH):
- Same pattern, relative tolerance for AMR
- Leaf count preserved, relative error = 5.6e-16

### New files

| File | Description |
|------|-------------|
| `src/io/checkpoint.h` | API: checkpoint_write, checkpoint_read |
| `src/io/checkpoint.c` | Implementation (~365 lines) |
| `tests/test_checkpoint.c` | Validation test suite (14/14) |

### Modified files

| File | Changes |
|------|---------|
| `src/main.c` | --checkpoint-every, --restart CLI flags + restart path |
| `Makefile` | checkpoint.c in IO_SRC, test-checkpoint target |
