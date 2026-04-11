# Lattice — 3D Numerical Relativity Simulator

## About

C codebase implementing the CCZ4 (conformal covariant Z4) formulation of
Einstein's field equations for evolving black hole spacetimes through inspiral,
merger, and ringdown. GPU acceleration via HIP (AMD + NVIDIA) — physics
kernels are pure C with `LATTICE_DEVICE` annotations, a thin backend
abstraction swaps CPU (OpenMP threads) and GPU (HIP kernels).

Primary dev target: Apple M4 (16 GB unified memory) for CPU, AMD (MI250X/MI300X)
and NVIDIA (V100/A100/H100) HPC GPUs for production runs.

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
- Algebraic enforcement: det(gambar)=1, tr(Abar)=0, chi≥1e-4, lapse≥1e-4

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

- N=10+ simultaneous black holes with arbitrary mass, spin, charge (MAX_PUNCTURES=32)
- Position-dependent eta for unequal mass: eta(x) = eta_0 / W(x)
- Spatially varying KO dissipation
- Full waveform catalog capability

**AMR parity gaps:** All closed. Both packed kernels (CPU + GPU) branch on
`p->em_enabled` to call `ccz4_maxwell_rhs_point`. Initial data solved
directly on evolution mesh (no interpolation). AH finder and output slices
work on AMR meshes.

**Current status: Phase 2 complete, Phase 3 in progress.**

**Phase 3 progress:**
- **6th-order operators:** FD stencils, AMR prolongation (7-point)
  all upgraded from 4th to 6th order. KO dissipation decoupled via `KO_ORDER`
  macro (default 6th-order, 7-point; optional 8th-order, 9-point). Restriction changed from 6th-order Lagrange
  to trilinear cell averaging (GRChombo match — positive weights only, no Gibbs
  oscillations near puncture singularities). 6th-order off-grid
  Lagrange interpolation (7-point stencil, half-width 3). 4th-order Sommerfeld
  BCs. Cubic Taylor temporal interpolation for subcycling (Chombo
  TimeInterpolatorRK4 match — 3rd-degree polynomial from all 4 RK4
  stages, O(dt⁴) accurate, Horner evaluation).
- **Tier 1 optimizations (all complete):** LTO for CPU builds, fast-path `pow(lapse,1)`,
  `restrict` qualifiers on RHS pointers, skip EM fields in dissipation/Sommerfeld,
  OMP-parallelized packed ghost exchange, flattened RK4 update loops (single OMP
  region over all fields), hoisted GPU grid_t construction, pre-computed
  restriction/prolongation weight product tables (232 entries, bit-exact verified),
  fused d1/d2 stencil (`fd_d1_d2()` loads 7 points once for both derivatives),
  conditional EM allocation (`grid_alloc_ex` with `n_fields` threaded through all
  subsystems — 25 fields when EM disabled, 19% memory savings).
- **Tier 2 optimizations (all complete):** Face-only Sommerfeld BC iteration
  (~5x fewer iterations than full-grid scan), pre-allocated ghost exchange scratch
  buffer (eliminates malloc/free in hot loop), persistent spatial hash table on
  `mesh_t` for O(1) block lookup — used by `mesh_find_block`, `mesh_find_block_at`,
  and `mesh_rebuild_neighbors` (built on regrid, ~100x speedup for AH finder/Psi4
  point-to-block lookups on large AMR meshes), `backend_enforce_algebraic_packed`
  (batched det(h)=1 / tr(A)=0 on device — eliminates GPU↔host round-trip),
  integer sub_step in subcycling (fixes floating-point frac drift), Ricci/raise_all
  symmetry exploitation (6 vs 9 components), Sommerfeld asymptotic array lookup,
  Levi-Civita curl unrolling, hoisted advection sign.
- **Tier 3 optimizations (all complete, 7/7):**
  - *Manual CSE in `ccz4_rhs_point`:* Pre-compute `K - 2*Theta` (used 6×),
    `A_mixed[k][j] = A[i][k] * h_UU[k][j]` (eliminates inner `ll` loop in A²
    trace and RHS_A). Pure arithmetic — same math, fewer FLOPs.
  - *`--block-size` CLI:* Alias for `--N_block`, validates even and >= 8.
  - *Compile-time EM dispatch:* `make EM=on` adds `-DLATTICE_EM_ENABLED`.
    `COMPILED_NUM_FIELDS` macro (25 or 31) gives compiler constant loop bounds
    in dissipation/Sommerfeld hot paths. Default `EM=off` eliminates EM overhead.
  - *Kernel restructuring:* Split `ccz4_rhs_point` into 5 `static inline`
    sub-functions (`ccz4_load_and_differentiate`, `ccz4_compute_geometry`,
    `ccz4_compute_covariant`, `ccz4_compute_evolution`, `ccz4_compute_gauge`).
    Scoped lifetimes via typed structs (`ccz4_fields_t`, `ccz4_derivs_t`,
    `ccz4_geom_t`, `ccz4_covd_t`). Evolution and gauge sub-functions write to
    output structs (`ccz4_evo_rhs_t`, `ccz4_gauge_rhs_t`); `ccz4_rhs_point`
    does all `rhs[]` stores. This avoids passing `double**` through sub-function
    boundaries, which triggers a GCC nvptx codegen bug. GPU benefit: better
    register allocation from scoped lifetimes. Zero memory overhead — compiler
    inlines into single kernel.
  - *Persistent per-level packs:* `mesh_t` caches `leaf_pack` and
    `level_packs[MAX_AMR_LEVELS]` across time steps. New `sync_to_blocks` /
    `sync_from_blocks` copy only the data buffer (not rhs/scratch/accum).
    `packs_dirty` flag triggers rebuild on regrid. Eliminates per-step
    malloc/free/memcpy cycle (~1 GB allocation per step for large meshes).
  - *Device-side ghost exchange:* Replaced host-side PCIe round-trip with
    7 GPU kernel launches (same-level, restriction, coarse ghost fill, 3 boundary
    extrapolation sweeps, prolongation). Zero PCIe DMA during time steps. Uniform
    meshes: 1 kernel launch. Constants (`nbr_offset`, `restrict_w/wkj`,
    `prolong_w/wkj`) in HIP `__constant__` memory for device access.
  - *Precomputed `inv_dx`:* `grid_t` stores `inv_dx = 1.0/dx`. All FD stencil
    functions (`fd_d1`, `fd_d2`, `fd_d1_d2`, `fd_d2_mixed`, `fd_adv`, `fd_ko`)
    take `inv_dx` and multiply instead of dividing by `dx`. Eliminates ~20
    divisions per grid point per RHS evaluation.
  - *Decoupled KO order (`KO_ORDER` macro):* KO dissipation order independent of
    FD order. Default `KO_ORDER=6` (7-point stencil, half-width 3) gives 1-cell
    ghost margin with GHOST=4, matching GRChombo/AthenaK. `make KO_ORDER=8` for
    legacy 8th-order (9-point, half-width 4, zero margin — unstable on AMR).
  - Dense output subcycling deferred.
- **Tier 4 optimizations (GPU, all complete):**
  - *Compact Sommerfeld kernel:* Boundary-only block ID list eliminates 70-90%
    of idle threads in typical AMR meshes.
  - *Fused RK4 kernels:* `backend_rk4_stage_packed` (accum + axpy in 1 kernel)
    and `backend_rk4_final_packed` (accum + copy + apply in 1 kernel). Saves
    3 kernel launches + 3 rhs buffer reads per RK4 step.
  - *Shared memory block metadata:* `dx_arr[b]` loaded into `__shared__` once
    per GPU block in RHS kernel. Saves 63 redundant global loads per block.
  - *Lazy host copy:* `backend_unmap_pack()` (free only) vs
    `backend_unmap_pack_sync()` (sync + free). Future: skip sync when host
    data not needed.
  - *HIP streams:* Persistent `hipStream_t` for all kernel launches. Enables
    non-blocking launches and potential host/device overlap.
  - *KO dissipation branch reduction:* Precomputed per-field sigma array
    eliminates 25+ branches per point in hot field loop.
  - *Unit-determinant inverse:* `compute_inverse_sym_unit_det()` skips det
    computation and division in `enforce_algebraic` (both CPU and GPU).
  - *OMP-parallelized mesh ghost exchange:* `#pragma omp parallel for` on
    block loops in `ghost_exchange()`, `ghost_exchange_all_blocks()`,
    `ghost_exchange_multilevel()` (all 5 phases), `fill_coarse_buf_ghosts()`,
    `ghost_exchange_same_level_all()`, `ghost_fill_cf_boundary()`,
    `ghost_fill_from_coarser()` Pass 1, `restrict_level_to_parents()`,
    `mesh_constraint_l2()`, `mesh_momentum_l2()`, `umg_sweep_1field()`,
    `umg_sweep_4field()`, `umg_compute_operator()`.
- **Tier 5 optimizations (GPU, zero-PCIe pipeline, all complete):**
  - *Persistent per-pack device memory:* `void *device_handle` on
    `meshblock_pack_t` stores opaque `hip_device_ptrs_t`. Device memory
    allocated on first `backend_map_pack()`, persists across sub-steps.
    `backend_activate_pack()` sets global pointers without memcpy.
    `backend_free_pack_device()` frees on pack destruction. Eliminates
    ~465 hipMalloc/hipFree calls per global step.
  - *GPU AMR subcycling with device-resident restriction:*
    `step_level_gpu()` / `subcycle_level_gpu()` in `rk4.c` run RK4
    evolution on device. Cubic Taylor temporal interpolation (Chombo
    TimeInterpolatorRK4 match): 3 Taylor coefficient buffers per parent
    pack, accumulated from each RK4 stage's RHS, Horner evaluation in
    `hip_cross_level_ghost_fill` kernel. `gpu_ensure_level_packs()` builds packs + cross-level maps on
    first step or regrid. Post-subcycle restriction via GPU kernel:
    Trilinear restriction (cell averaging, GRChombo match)
    with floor clamp (chi≥1e-4, lapse≥1e-4) to prevent sub-floor values
    from negative stencil weights near puncture. Ghost exchange on fine
    pack before restriction fills same-level ghosts for stencil reach.
    Buffer blocks (AthenaK pattern): non-leaf parents packed at
    `[n_evolve, n_blocks)`, participate in ghost exchange as data sources
    but are not evolved by RK4. Every production AMR NR code restricts
    after fine subcycling (GRChombo, Athena++, CarpetX, BAM).
    Without restriction, momentum constraint grows exponentially at AMR
    boundaries → NaN. RHS buffer zeroed before RK4 stages to prevent
    stale ghost-zone RHS from corrupting data.
  - *Solver:* GPU solver removed (produced inf residuals at inspiral scale).
    CPU covering grid FAS always used — FMG converges in 1 pass per level,
    no inter-block ghost exchange during MG. ~2800 lines of GPU solver backend
    code deleted (kernels, device state, packed API).
- **GPU diagnostics:** On-device kernels exist for constraint L2, momentum L2,
  min lapse with position, BH separation (two-pass lapse minimum), NaN/Inf
  check, and Psi4 extraction. Diagnostic-only pack mapping
  (`backend_map_pack_diag`) transfers data + metadata without rhs/scratch/accum
  (~75% savings). Constraint/momentum kernels use shared-memory block-level
  reduction. Psi4 kernel: host pre-computes angular-point-to-block mapping
  via `mesh_find_block_at`, GPU calls `psi4_compute` per point (512 threads),
  mode decomposition on host.
  **Conditional host sync:** `gpu_sync_all_to_host()` (now public in `rk4.h`)
  removed from unconditional post-step path in `rk4_step_mesh`. main.c
  computes per-step need flags and syncs ONCE only when CPU-only work is
  required (regrid, checkpoint, output, AH finder, BH tracker, CCE).
  Steps with no diagnostics skip the ~1.2 GB transfer entirely.
  **GPU-native diagnostics in main.c:** Constraint L2 and Psi4 extraction
  run directly on device-resident level packs when GPU AMR is active.
  `gpu_constraint_l2()` / `gpu_momentum_l2()` loop over level packs using
  `backend_constraint_l2_raw_packed` (returns sum+vol separately for correct
  multi-level volume-weighted combination). `gpu_psi4_extract()` activates
  level-0 pack and calls `backend_psi4_extract_packed`. These diagnostics
  need zero host sync — only ~100 bytes of scalar results return to host.
  **GPU BH tracker:** `bh_tracker_update_positions_packed()` uses
  `backend_min_lapse_excl_packed` (N exclusion zones) on device-resident
  level packs. Same successive lapse-min algorithm as CPU, but each pass
  is a single GPU kernel launch. Only ~100 bytes transfer per BH.
  HIP kernel: `hip_min_lapse_multi_excl_partial` with device-side
  exclusion arrays. `find_horizons` (AH per BH) remains CPU-only.
  **Missing GPU kernels:** AH finder (fully CPU, biggest bottleneck at 5-50s),
  CCE worldtube interpolation (similar to Psi4 infrastructure).
- **Volume-weighted AMR constraint L2:** `mesh_constraint_l2()`, `mesh_momentum_l2()`,
  and packed variants (CPU + GPU) now weight each cell by dV=dx^3 and normalize by
  total volume. Eliminates diagnostic artifacts on AMR meshes where fine cells near
  punctures dominated the unweighted norm. Standard practice in all AMR codes.
- **Lapse/shift advection:** `lapse_advec_coeff=1.0` and
  `shift_advec_coeff=1.0` are the default in `params.h` (required for AMR gauge
  stability). Without the transport term `β^i ∂_i α`, the gauge is purely local
  and unstable at dx≥2M. Zero additional cost (derivatives already computed).
  Ref: gr-qc/0610128 (Brugmann et al.).
- **Constraint-preserving BCs:** BAM-style CP BCs replace the RHS of constraint
  fields (Theta, K, A_ij, Gamma^i) at boundary points with outgoing-wave equations
  at correct characteristic speeds, while keeping Sommerfeld for metric/gauge fields.
  Per-field speeds: Theta=1, K=sqrt(2/alpha), A_ij=1, Gamma^s(normal)=sqrt(3/4),
  Gamma^A(tangential)=1. GPU-optimized: warp-coherent field loop, zero divergence.
  CLI: `--bc sommerfeld|cp` (default `cp`). Ref: arXiv:1212.2901 (Hilditch et al.).
- **Position-dependent eta:** `eta(x) = eta_0 / W(x)` where `W = sqrt(chi)` for
  stable unequal-mass binary evolution. Gated behind `position_dependent_eta` flag
  (default 1). Ref: arXiv:1003.0859 (Muller & Brugmann).
- **N-body initial data:** Covering grid FAS multigrid constraint solver. Level-by-level from coarse to fine, each AMR level solved by creating a single temporary uniform grid spanning all blocks, running proven single-grid FAS (FMG + V-cycles + 8-color Newton-GS smoother). BY 1-field + HiSpID 4-field coupled solvers. No inter-block ghost exchange during MG — zero risk of cross-level corruption. FMG converges in 1 pass per level. Replaces old composite FAS multigrid and JFNK+BiCGSTAB. Inspiral solver benchmarks: D10 binary Ham L2 = 9.76e-5 in 172s, 4-BH Ham L2 = 1.01e-4 in 145s. Solver runs once at t=0 on CPU; evolution uses GPU for time-stepping. File is `jfnk_solver.c/h` for API compatibility.
- **Einstein-Maxwell:** 6 new evolved fields (E^i, B^i), conformal Maxwell evolution with constraint damping, EM stress-energy coupling to CCZ4 (gated by `--em` flag), charged puncture initial data via `--puncture M,x,y,z,Px,Py,Pz,Sx,Sy,Sz,Q`.
- **Spin:** Bowen-York spinning punctures + HiSpID high-spin initial data (quasi-isotropic Kerr conformal metric, coupled 4-field relaxation).
- **Apparent horizons:** Hyperbolic flow method (BHaHAHA-inspired) with 6th-order off-grid
  interpolation, mass/spin/area extraction, `--ah` CLI flag. Works on both single-grid and
  AMR meshes. AH finder requires tracker offset < 20% of AH radius (verified at dx=0.031M).
  Production inspiral (24×48 angular grid, tol=1e-4, 500 max iterations) uses AH-radius
  excision for constraint norms: `mesh_constraint_l2_ex(m, tracker)` excludes spheres of
  1.5×r_AH around each tracked BH. Falls back to lapse < 0.3 when AH not found.
  Ref: BHaHAHA arXiv:2505.15912, Einstein Toolkit CarpetMask.
- **Psi4 extraction:** Newman-Penrose Psi4 via 3+1 Weyl decomposition (Electric + Magnetic Weyl tensors), Gram-Schmidt tetrad, spin-weighted spherical harmonic decomposition via Wigner d-matrix, Gauss-Legendre × trapezoidal quadrature, block-aware AMR extraction. CLI: `--psi4`, `--psi4_every`, `--psi4_radius`, `--psi4_l_max`, `--psi4_n_theta`, `--psi4_n_phi`. CSV mode output.
- **CCE worldtube output:** SpECTRE-compatible AdmMetricNodal HDF5 format for
  Cauchy-Characteristic Evolution. Interpolates conformal CCZ4 fields on extraction
  sphere, reconstructs physical ADM quantities (gamma_ij, K_ij, lapse, shift + all
  spatial derivatives), writes 49 datasets in GL×uniform angular grid. Pipeline:
  Lattice → `CceR####.h5` → SpECTRE `PreprocessCceWorldtube` → gauge-invariant
  strain at scri+. Optional dependency on libhdf5 (`make HDF5=on`). CLI: `--cce`,
  `--cce_every`, `--cce_radius`, `--cce_lmax`.
- **AMR:** Block-structured Berger-Oliger with subcycling, Morton-ordered mesh, 6th-order prolongation, trilinear restriction (cell averaging), multi-level ghost exchange. AMR-aware 1D output slices and AH finder.
  Refinement radius per level uses equidistribution-optimal scaling:
  `r_k = C · M_p · β^k` where β = 2^(3/5) ≈ 1.516 (derived from equal
  truncation error at every level boundary for 6th-order FD on 1/r³ Riemann
  curvature fields), C = 1.5 (BAM-like, finest level covers ±1.5M),
  M_p = puncture mass. Both C and β are CLI-configurable (`--refine-c`,
  `--refine-beta`). Number of levels auto-capped at domain half-size.
  Per-puncture radii scale with mass — heavier BHs get larger refinement regions.
  Ref: docs/amr_refinement_ratio.html (full derivation).
- **Solve on evolution mesh:** AMR initial data constraint solver operates directly
  on evolution blocks (`set_bowen_york_mesh()`), eliminating interpolation error
  and ensuring exact discrete operator consistency. Solver reuses idle evolution
  arrays at t=0 (22 of 100 slots). Refinement depth defaults to `--max_level` so
  initial data and evolution use the same AMR depth (override with `--amr-levels`).
  Each level halves dx near punctures. Each AMR level is solved on a covering grid
  (single temporary uniform grid), then scattered back to blocks.
  Ref: Athena++ MG (Tomida & Stone 2023), arXiv:0912.2920 (Alic et al.).
- **N-body BH tracker:** Multi-BH position tracking via successive lapse-minimum
  searches with exclusion zones (GRChombo PunctureTracker pattern), per-BH AH
  finding for mass/spin extraction, and pairwise merger detection (sep < 3M).
  CSV diagnostic output with per-BH columns (position, mass, spin, lapse).
  Auto-enabled for N>=2 punctures. CLI: `--tracker`, `--tracker_every`.
  Ref: arXiv:2505.01495 (GRChombo 25-BH cluster simulation).
- **Single root block (N_ROOT=1):** Multi-root meshes removed. All multi-block topology
  comes from AMR refinement only. The covering grid solver requires whole-domain
  visibility at the coarsest level — multi-root meshes broke cross-block coupling.
  With single root, the solver achieves convergence factor ~0.13/cycle (textbook
  multigrid). Higher MAX_LEVEL compensates for the coarser base grid.
  `mesh_create(N_block, L, method)` creates one root block; no N_root parameter.
- **Tier 0 bug fixes:** `enforce_algebraic_block()` in refine.c fixed (was using slow
  `1.0/cbrt(det)` with divergent `if (det > 0.0)` guard; now uses `fast_inv_cbrt(det)`
  unconditionally, matching `rk4.c`). Packed RHS kernel `g_local.n_fields` was 0
  (memset default), disabling KO dissipation in all AMR runs; fixed in both backends.
  Subcycling frac drift: replaced floating-point `floor(t/dt)` computation with
  integer `sub_step` parameter (latent bug in long-duration runs). Old AMR composite
  multigrid cross-level ghost corruption bug eliminated by design — the covering grid
  solver has no inter-block ghost exchange during MG (each level solved on a single
  contiguous grid). CPU `step_level` missing `backend_zero_packed(RHS)` — stale
  ghost-zone RHS corrupted cross-level boundaries during RK4 stages. GPU
  `subcycle_level_gpu` missing cross-level ghost fill before post-subcycle
  restriction — 6th-order restriction stencil read stale ghost data at
  refinement boundaries. GPU `step_level_gpu` enforce_algebraic ordering:
  was BEFORE `ghost_exchange_cross_level_packed` — prolongated ghost zones with
  chi<0 (from 6th-order negative Lagrange weights near puncture) never floored
  before RHS. Fixed: enforce AFTER ghost exchange (matches CPU path). CPU
  `subcycle_level` missing post-restriction enforcement — added to match GPU.
  Phase 3a coarser-neighbor branch removed from both CPU and GPU
  (`hip_ghost_coarse_fill`, `packed_fill_coarse_buf_ghosts`) — was direct-copying
  buffer block data without temporal interpolation. AthenaK pattern: Phase 3a =
  same-level only; Phase 3b handles ALL coarser via Taylor interpolation.
  Cross-level ghost fill 1-cell offset bug: `hip_cross_level_ghost_fill` and CPU
  `copy_from_coarse_grid` mapped fine coarse_buf indices to coarser pack indices
  via `si = ii + off_i` without correcting for the ghost width difference. Fine
  coarse_buf has ghost = `COARSE_BUF_GHOST` = 5 (for 7-point prolongation
  stencil); coarser pack has ghost = `GHOST_WIDTH` = 4. Fixed via
  `ghost_correction = ghost_src - ghost_dst = -1`. Without this, every
  cross-level ghost cell read from a 1-cell-shifted location in the coarser
  pack — corrupting temporal interpolation at every refinement boundary on
  every RK4 substep.
- **Per-stage cross-level ghost fill (GRChombo match):** Both CPU `step_level` and
  GPU `step_level_gpu` fill coarse-fine boundary ghosts at RK4 sub-stages 1, 2,
  and 4 with temporally interpolated data at the correct sub-stage time
  (`frac + c_s / refine_ratio`, where c_s = {0, 0.5, 0.5, 1.0} for classic RK4).
  Stage 3 cross-level fill skipped: shares frac=0.5 with stage 2, and interior-only
  RK4 stage writes preserve coarse_data ghost cells between stages (25% fewer
  cross-level fills, ~5,200 kernel launches saved per base step at 7 AMR levels).
  Uses cubic Taylor temporal interpolation (Chombo TimeInterpolatorRK4): Taylor
  coefficient buffers (a1, a2, a3) accumulated from each RK4 stage's RHS,
  evaluated via Horner form `U(θ) = U_n + θ*(a1 + θ*(a2 + θ*a3))`.
  Post-subcycle restriction followed by coarse ghost re-exchange (GRChombo
  `postTimeStep` → `averageToCoarse` → `fillBdyGhosts` pattern).
  Ref: arXiv:2112.10567 (GRChombo AMR lessons), Chombo TimeInterpolatorRK4.
- **Interior-only RK4 stage/final kernels:** `hip_rk4_stage` and `hip_rk4_final`
  write only interior cells `[ghost,ghost+N)³` of evolve blocks (leaves).
  Ghost zones and buffer blocks untouched — preserves cross-level fill data
  between stages, enabling the stage 3 skip. CPU backend uses same pattern
  with `#pragma omp parallel for collapse(2)`. Reduces RK4 data traffic by ~55%.
- **Parallelized cross-level ghost fill kernel:** `hip_cross_level_ghost_fill`
  parallelized over `(entry, field)` instead of just `entry` — 25x more GPU
  threads per launch, eliminating the serial field loop bottleneck.
- **Phase-3b-only cross-level ghost fill:** `backend_cross_level_ghost_fill_packed`
  stripped from 7 kernel launches (restrict, same-level fill, cross-level fill,
  extrap×3, prolong) down to 1 (cross-level fill only). Phases 2, 3a, 3.5, 4
  are redundant with the subsequent `backend_ghost_exchange_packed` which already
  does the full 5-phase exchange. Eliminates 381 redundant prolong launches per
  base step (the most expensive kernel: 7³=343-point stencil per ghost cell).
  QNM publication benchmark: **400s → 100s/step on H100** (7 AMR levels, 344 blocks).
- **HiSpID solver bug fixes:** (1) `mg_level_init` received domain length `Lcov`
  instead of grid spacing `Lcov/N_l` — all FD stencils off by N², solver never
  actually solved on covering grid. (2) Hamiltonian constraint R_tilde term had
  wrong sign (+R/8 instead of -R/8) at 3 locations — caused Newton divergence
  for any nonzero conformal curvature. Ref: arXiv:1410.8607 Eq. (13).
- **Production NaN/Inf check:** `mesh_check_finite()` (CPU) and
  `backend_check_finite_packed()` (GPU) wired into main.c via
  `--nan-check-every N` CLI flag. Catches blown simulations immediately.
- **Default integrator:** Changed from CK45 to classic RK4 (`RK_CLASSIC`). Classic is
  faster (4 stages vs 5) but uses 25% more memory. All test allocations updated.
- **Tests:** Flat spacetime, convergence (order 6.5), Bowen-York (33/33 + N-body),
  HiSpID (26/26), AH finder (13/13), Maxwell (15/15), Psi4 (15/15), CCE (49/49),
  CP-BC (30/30), pack_evolve (5/5), amr_prolong (15/15), checkpoint (14/14),
  binary inspiral D10 benchmark (T=700M, CAKO + per-field sigma + SSL + position-dep
  eta, 2-radius Psi4 (r=70+100M), chi_refine=0.05, Samurai consensus validation,
  8 hard + 4 advisory tests, per-BH position/mass/spin CSV via bh_tracker),
  inspiral solver smoke (7-level D10 binary + 6-level 4-BH square, 4/4).
  N-body tracker (40/40: init, position update, AH loop, merger detection,
  bookkeeping, CSV output, 25-BH alloc, post-merger tracking).
  GPU tracker (9/9: single BH, two-BH with exclusion, boosted BH tracking,
  CPU vs packed path equivalence).
  N-body smoke tests:
  3-BH line, 5-BH pentagon. Total: 31 evolved fields (25 CCZ4 + 6 EM).

Update as milestones are reached.

## Project Structure

```
lattice/
├── CLAUDE.md
├── DEVLOG.md
├── novel.md               # novel contributions vs existing NR codes
├── Makefile
├── setup_rocm.sh           # GPU setup script (CUDA + ROCm + env vars)
├── grchombo-ref/           # read-only reference (not compiled)
├── src/
│   ├── core/
│   │   ├── grid.h / grid.c     # grid allocation, indexing, ghost zones
│   │   ├── fields.h            # field enum, count
│   │   ├── params.h            # simulation parameters
│   │   ├── device.h            # LATTICE_DEVICE / EXTERN_C macros (HIP portability)
│   │   └── timer.h             # TIMER_START/STOP macros (clock_gettime)
│   ├── backend/
│   │   ├── backend.h           # abstract interface
│   │   ├── backend_cpu.c       # OpenMP threads (CPU)
│   │   └── backend_hip.cpp     # HIP GPU kernels (AMD + NVIDIA)
│   ├── evolution/
│   │   ├── ccz4_rhs.h/c        # CCZ4 right-hand-side (+EM source terms)
│   │   ├── maxwell_rhs.h/c     # Maxwell evolution equations (E^i, B^i)
│   │   └── dissipation.c       # Kreiss-Oliger dissipation
│   ├── geometry/
│   │   └── tensor_utils.h      # inline tensor operations
│   ├── numerics/
│   │   ├── finite_diff.h       # FD_D1, FD_D2 macros (6th-order)
│   │   ├── interpolate.h       # 6th-order off-grid Lagrange interpolation
│   │   └── rk4.h/c             # RK4 time integrator (+mesh stepping)
│   ├── initial_data/
│   │   ├── puncture.h/c        # Brill-Lindquist puncture data
│   │   ├── bowen_york.h/c      # BY A_ij (momentum+spin) + CCZ4 conversion (+mesh-level API)
│   │   ├── jfnk_solver.h/c     # Covering grid FAS multigrid solver (replaces relaxation.c/h + relaxation_amr.c/h)
│   │   └── kerr_quasi_isotropic.h/c  # QI Kerr metric for HiSpID (high-spin data)
│   ├── diagnostics/
│   │   ├── constraints.h/c     # Hamiltonian + momentum constraints
│   │   ├── ah_finder.h/c       # Apparent horizon finder (hyperbolic flow)
│   │   ├── psi4.h/c            # Psi4 gravitational wave extraction
│   │   ├── bh_tracker.h/c     # Multi-BH tracker (N-body position/AH/merger)
│   │   └── cce_worldtube.h/c   # CCE worldtube HDF5 output (optional, HDF5=on)
│   ├── boundary/
│   │   ├── sommerfeld.h/c      # radiative BCs (+block-aware variant)
│   │   └── constraint_preserving.h  # CP BCs: characteristic speeds + CP RHS formula
│   ├── amr/
│   │   ├── morton.h             # Morton (Z-order) encoding for SFC
│   │   ├── block.h / block.c   # block_t: single mesh block with metadata
│   │   ├── mesh.h / mesh.c     # mesh_t: collection of blocks forming domain
│   │   ├── ghost_exchange.h/c  # 26-neighbor + multi-level ghost exchange
│   │   ├── prolongation.h/c    # 6th-order Lagrange coarse→fine (AthenaK)
│   │   ├── restriction.h/c     # trilinear (cell averaging) fine→coarse restriction
│   │   ├── criterion.h/c       # chi-gradient refinement criterion
│   │   ├── refine.h/c          # oct-tree split/merge/regrid
│   │   └── meshblock_pack.h/c  # GPU batch packing (AthenaK-style)
│   └── io/
│       ├── output.c            # data output
│       ├── checkpoint.h        # checkpoint/restart API
│       └── checkpoint.c        # binary checkpoint save/restore (AMR-aware)
├── tests/
│   ├── test_flat.c             # flat spacetime stability
│   ├── test_single_bh.c        # single puncture evolution
│   ├── test_convergence.c      # 3-resolution convergence (order 6.5)
│   ├── test_constraints.c      # Hamiltonian + momentum constraint tests
│   ├── test_amr_mesh.c         # AMR mesh creation + Morton ordering
│   ├── test_amr_ghost.c        # ghost exchange + multi-block evolution
│   ├── test_amr_prolong.c     # prolongation + noise reduction
│   ├── test_amr_refine.c     # oct-tree refinement + multi-level ghost
│   ├── test_amr_evolve.c     # AMR evolution integration (8/8)
│   ├── test_amr_accuracy.c   # AMR accuracy validation
│   ├── test_head_on.c          # head-on binary collision
│   ├── test_head_on_output.txt # saved test output (merger diagnostics)
│   ├── test_pack_evolve.c    # packed batch kernel validation
│   ├── test_subcycle.c       # Berger-Oliger subcycling validation
│   ├── test_bowen_york.c    # Bowen-York initial data (A_ij + solver + evolution)
│   ├── test_jfnk.c          # Covering grid FAS solver tests
│   ├── test_hispid.c        # HiSpID high-spin initial data (QI Kerr + coupled solver)
│   ├── test_ah_finder.c     # Apparent horizon finder tests (13/13)
│   ├── test_maxwell.c       # Einstein-Maxwell tests (15/15)
│   ├── test_psi4.c          # Psi4 gravitational wave extraction tests (15/15)
│   ├── test_cce_worldtube.c # CCE worldtube HDF5 output tests (49/49, requires HDF5)
│   ├── test_cp_bc.c         # Constraint-preserving BC tests (30/30)
│   ├── test_binary_inspiral.c  # D10 benchmark (Samurai consensus, T=700M, CAKO+SSL+per-field sigma, 2-radius Psi4, 8+4 tests)
│   ├── test_inspiral_solver.c  # Inspiral solver smoke test (7-level D10 binary + 6-level 4-BH, 4/4)
│   ├── test_nbody_track.c      # N-body BH tracker (init, position, AH, merger, CSV, 40/40)
│   ├── test_inspiral_convergence.c  # AMR binary inspiral convergence (3 resolutions)
│   ├── test_checkpoint.c       # Checkpoint/restart validation (uniform + AMR, 14/14)
│   ├── test_gpu_tracker.c     # GPU vs CPU BH tracker (packed lapse-min, boosted BH, 9/9)
│   ├── test_gpu_debug.c       # GPU kernel isolation test (per-kernel sync barriers)
│   ├── test_qnm_ringdown.c   # Schwarzschild QNM ringdown (AMR, dx=0.5M, quick)
│   ├── test_qnm_publication.c  # Publication QNM (L=256, 7 levels, C=1.5, β=2, dx=M/16, Psi4 r=30+50M)
│   └── test_gauge_wave.c       # Gauge wave (WIP — needs periodic BCs)
├── docs/
│   ├── architecture.html       # consolidated architecture & design reference
│   ├── nr_code_comparison_full.html  # line-by-line comparison vs 8 NR codes (13 sections)
│   ├── amr_refinement_ratio.html  # equidistribution-optimal β=1.516 derivation
│   ├── qnm_ringdown.html      # Schwarzschild QNM validation (physics + results)
│   └── archive/                # older deep-dive guides (preserved, not primary)
└── tools/
    ├── compute_amr_weights.py  # SymPy derivation of AMR stencil weights
    └── verify_weights.c        # bit-exact verification of pre-computed weight tables
```

## Build & Test

```bash
make                    # CPU build (-O3 -ffast-math -march=native -flto)
make BACKEND=gpu        # GPU build (HIP — AMD + NVIDIA, requires ROCm)
make debug              # debug build (-O0 -g -fsanitize=address,undefined)
make test               # all tests
make test-single-bh     # single puncture evolution
make test-convergence   # 3-resolution convergence verification
make test-constraints   # Hamiltonian + momentum constraint tests
make test-head-on       # head-on binary collision
make test-amr-mesh      # AMR mesh creation + Morton ordering
make test-amr-ghost     # ghost exchange + multi-block evolution
make test-amr-prolong   # prolongation + noise reduction (CAKO/CAHD/SSL)
make test-amr-refine    # oct-tree refinement + multi-level ghost exchange
make test-amr-evolve    # AMR evolution integration (8/8)
make test-amr-accuracy  # AMR accuracy validation
make test-pack-evolve   # packed batch kernel validation (8/8)
make test-subcycle      # Berger-Oliger subcycling validation
make test-bowen-york   # Bowen-York initial data (A_ij, solver, evolution)
make test-jfnk         # Covering grid FAS solver tests
make test-hispid       # HiSpID high-spin initial data (QI Kerr + coupled solver)
make test-ah           # Apparent horizon finder (interpolation, Schwarzschild, diagnostics)
make test-maxwell      # Einstein-Maxwell (flat EM, plane wave, charged BH, constraints)
make test-psi4         # Psi4 extraction (GL quadrature, harmonics, modes, flat, Schwarzschild)
make HDF5=on test-cce  # CCE worldtube HDF5 output (requires libhdf5-dev)
make test-cp-bc        # Constraint-preserving BCs (speeds, formula, flat, single BH)
make test-inspiral     # D10 benchmark: CAKO+SSL+per-field sigma, 2-radius Psi4, Samurai consensus (H100)
make test-inspiral-solver  # Inspiral solver smoke test (8-level binary + 7-level 4-BH)
make test-nbody-track  # N-body BH tracker (init, position, AH, merger, CSV, 40/40)
make test-inspiral-convergence  # AMR binary inspiral convergence (long run, ~hours)
make test-checkpoint   # Checkpoint/restart (uniform + AMR, bitwise-identical)
make test-gpu-tracker  # GPU vs CPU BH tracker (packed lapse-min, boosted BH, 9/9)
make test-gpu-debug    # GPU kernel isolation test (requires BACKEND=gpu)
make test-qnm          # Schwarzschild QNM ringdown (AMR, dx=0.5M, ~1 hr CPU)
make test-qnm-pub      # Publication QNM (L=256, 7 levels, β=2, dx=M/16, Psi4 r=30+50M, ~2 hrs GPU)
make test-gauge-wave   # Gauge wave (WIP — needs periodic BCs)
make clean
```

Backend flag: `BACKEND=cpu|gpu`. Default is `cpu` (OpenMP threads).
HDF5 flag: `HDF5=on` enables CCE worldtube output (requires `libhdf5-dev` + `pkg-config`).
Time integrator: `--rk classic|ck45`. Default is `classic` (standard 4-stage
RK4, 4 memory blocks). Use `--rk ck45` for Carpenter-Kennedy 2N low-storage
(5 stages, 3 memory blocks, 25% less memory).
Compiler: `clang` on macOS (CPU only), `gcc` on Linux (CPU), `nvcc` (NVIDIA GPU),
`hipcc` (AMD GPU). The Makefile auto-detects the platform.
No external dependencies beyond standard C and libomp. GPU requires ROCm HIP
headers (for the portable HIP API). NVIDIA builds also require `nvcc` (CUDA toolkit).
Optional: `libhdf5-dev` for CCE worldtube output (`make HDF5=on`).
Debug builds enable NaN/Inf checking — any floating-point trap is a bug.
Setup helper: `setup_rocm.sh` automates GPU dependency installation (CUDA toolkit,
ROCm HIP headers, environment variables) on Ubuntu with NVIDIA or AMD GPUs.

### GPU backend requirements

GPU acceleration uses HIP (Heterogeneous-compute Interface for Portability).
Physics kernels are pure C with `LATTICE_DEVICE` annotations — the same code
compiles for both CPU and GPU. Only `src/backend/backend_hip.cpp` is C++.

**Requirements:**
- ROCm HIP headers (`/opt/rocm/include/hip/hip_runtime.h`) for the portable API
- AMD GPUs: `hipcc` compiler (native HIP, `HIP_PLATFORM=amd` default)
- NVIDIA GPUs: `nvcc` compiler (CUDA toolkit). The Makefile sets `HIPCC := nvcc`
  and uses `-x cu` to compile device code. HIP API calls (hipMalloc, hipMemcpy,
  etc.) are thin wrappers around CUDA, provided by the ROCm HIP headers.
- HPC-class GPU with 1:2 FP64:FP32 ratio (MI250X, MI300X, V100, A100, H100).
  Consumer GPUs (1:32 ratio) have negligible FP64 throughput.

**Two-phase compilation:** Host-only C files compiled with `gcc`, device-callable
C files and `backend_hip.cpp` compiled with the device compiler (`nvcc` on NVIDIA,
`hipcc` on AMD). The Makefile auto-detects the platform via `HIP_PLATFORM`.
Stack size set to 16 KB per thread via `hipDeviceSetLimit()` in `backend_init()`
(CCZ4 RHS kernel uses ~5.3 KB stack per thread).

**`device.h` portability:** `LATTICE_DEVICE` expands to `__host__ __device__`
only when compiled by hipcc/nvcc (detected via `__HIPCC__` or `__cplusplus`
with `LATTICE_HIP` defined). Host-only C files never see HIP headers.
`LATTICE_DEVICE` is also required on `static const` arrays at file scope
(e.g., `h_idx`, `A_idx`, `levi_civita`, `asym_values`) because `nvcc` does
not auto-promote static const arrays to device-visible — without the annotation,
device code gets zeroed-out values.

```bash
# AMD GPU build (hipcc native)
make BACKEND=gpu

# NVIDIA GPU build (nvcc + HIP headers)
export HIP_PLATFORM=nvidia CUDA_PATH=/usr
make BACKEND=gpu

# Run
./build/lattice --N 256 --steps 400 --puncture 1.0,0,0,0
```

## Code Style

- **C17 for physics code.** Only `backend_hip.cpp` is C++ (required by HIP kernel syntax).
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
| `kappa1` | 0.1 | Constraint damping (Theta + Z_i). BAM uses 0.1, AthenaK uses 0.02. |
| `kappa2` | 0.0 | Controls mix of Theta damping in K equation |
| `kappa3` | 1.0 | Z contribution in Gamma equation. GRChombo covariant CCZ4 (kappa3=1 + constant kappa1). BAM/AthenaK use 0.5 with non-covariant (kappa1×alpha). Ref: arXiv:1106.2254 (Alic 2012). |
| `sigma` | 0.5 | KO dissipation strength (used when `use_per_field_sigma=0`) |
| `sigma_gauge` | 1.0 | Per-field KO sigma for gauge fields (lapse/shift/B^i). Aggressive damping absorbs the gauge wave that causes AMR boundary reflections. |
| `sigma_phys` | 0.15 | Per-field KO sigma for physical fields (chi/h/K/A/Theta/Gamma). Gentle damping preserves waveform accuracy. Sweep-informed. |
| `use_per_field_sigma` | 1 | Enable per-field sigma by default (compensates for 6th-order FD's lower implicit diffusion) |
| `lapse_coeff` | 2.0 | Coefficient c in 1+log slicing: dt(alpha) = -c * alpha * (K - 2*Theta) |
| `lapse_power` | 1.0 | Power p in Bona-Masso: f(alpha) = c * alpha^(p-2) |
| `shift_Gamma_coeff` | 0.75 | F in dt(beta^i) = F * B^i |
| `eta` | 1.0 | Damping in Gamma-driver: dt(B^i) = dt(Gamma^i) - eta * B^i |
| `rk_method` | `RK_CLASSIC` | Time integrator: `RK_CLASSIC` (4 stages, 4 blocks) or `RK_CK45` (5 stages, 3 blocks) |
| `bc_type` | `BC_CONSTRAINT_PRESERVING` | Boundary conditions: `BC_SOMMERFELD` (standard radiative) or `BC_CONSTRAINT_PRESERVING` (BAM-style, arXiv:1212.2901) |
| `amr_levels` | `max_level` | Initial data solver refinement levels (`--amr-levels`). Defaults to `--max_level` so initial data and evolution use the same depth. Each level halves dx near punctures. Override for rare cases where you want finer initial data than evolution can afford. Requires `--rk classic`. |
| `refine_c` | 1.5 | Finest refinement box radius = C × M_puncture (`--refine-c`). BAM uses ~1.25. |
| `refine_beta` | 1.516 | Level growth ratio β = 2^(3/5) (`--refine-beta`). Optimal for 6th-order FD. |

### FAS Multigrid Solver Tuning

The covering grid FAS solver (`jfnk_solver.c`) uses FMG (Full Multigrid) with
FAS V-cycles and 8-color Newton-Gauss-Seidel smoothing. `tol` is the target
residual; `max_iter` is the max post-FMG V-cycles (usually 0-9). Both 1-field
(BY Hamiltonian) and 4-field (HiSpID coupled) modes are supported.

The discretization floor is O(dx^4):

| Grid N | ~Floor residual | Recommended `tol` |
|--------|----------------|-------------------|
| 24     | 1e-3           | 1e-4              |
| 64     | 1e-7           | 1e-8              |
| 128    | 1e-9           | 1e-10             |
| 256    | 1e-11          | 1e-12             |

FMG achieves discretization accuracy in a single pass (~1.14 V-cycles of work).
Post-FMG V-cycles polish below the FMG residual; typically 0-9 needed.
Multigrid hierarchy: N, N/2, N/4, ... down to N_min=4.
For production, `tol=1e-10, max_iter=50000` is the default in `set_bowen_york()`.
Both BY (1-field) and HiSpID (4-field) use 1e-10 — evolution truncation errors
are orders of magnitude larger, so polishing below 1e-10 is wasted work.

**AMR solver resolution:** With `--amr-levels L`, the solver adds L refinement
levels near each puncture. Each level halves the grid spacing:
`dx_fine = dx_base / 2^L`. For example, a base grid at N=32, L=64 (dx=2M) with
`--amr-levels 3` achieves dx=0.25M near the BH. Each AMR level is solved on a
covering grid (single temporary uniform grid spanning all blocks at that level).
No inter-block ghost exchange during MG — the covering grid approach eliminates
the cross-level corruption bug that plagued the old composite FAS solver.
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

**GPU production targets:** AMD MI250X/MI300X and NVIDIA V100/A100/H100 via HIP.
Estimated 20-100x speedup over M4 CPU (0.05-0.2 sec/step at N=128).
N=256 requires 40+ GB GPU memory. N=512 requires MI300X (192 GB) or H200 (141 GB).

No platform-specific code outside `src/backend/`. Physics kernels are pure C
with `LATTICE_DEVICE` annotations for GPU compilation.

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
- **arXiv:1212.2901**: Constraint-preserving BCs (Hilditch et al., BAM) — **implemented**

### Phase 2 References (planned)
- **arXiv:2104.06978**: Charged binary inspiral
- **arXiv:1004.1353**: N-puncture multigrid solver (Lousto et al.)
- **arXiv:1003.0859**: Position-dependent eta (Muller & Brugmann)
- **arXiv:2404.01137**: Improved KO dissipation + constraint damping
- **arXiv:2505.01495**: GRChombo 25-BH cluster simulation
- **arXiv:2312.05438**: AMR refinement strategy comparison
