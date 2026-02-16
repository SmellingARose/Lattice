# AMR Implementation Plan (Revised)

## Design Decisions (from decision guide discussion)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Block size | **Configurable (default 32^3, option 16^3)** | 32^3 has 1.95x ghost overhead vs 3.4x for 16^3. GPU benchmarks show 90x perf difference (Parthenon-VIBE arXiv:2509.19701). Make N_BLOCK a compile-time constant so both sizes work. |
| Ghost exchange | **26-neighbor (direct, 1-pass)** from the start | Single pass, 0 sync barriers, GPU-optimal. More code (~600 lines vs ~300) but no refactoring later. |
| Prolongation | **4th-order Lagrange** from the start | Chaotic N-body needs high-order. Copy weights from GRChombo `QuadCFInterp.hpp`. CAKO mitigates ringing near punctures. |
| Restriction | **Volume-weighted average** (8 fine → 1 coarse) | Standard, conservative, trivial. |
| Refinement criterion | **Chi-gradient** with every-step check | Check is cheap (~0.1 ms max-reduction). Only regrid when triggered. Hysteresis: chi_refine=0.1, chi_coarsen=0.01. |
| Time stepping | **Global dt first**, then **level-by-level subcycling (Option C)** | Global dt to get AMR working + tested. Then add subcycling as a separate stage. |
| GPU strategy | **MeshBlockPack batching** from Stage 1 | Contiguous buffer, single kernel launch. GPU-first design throughout. |
| Noise reduction | **CAKO + CAHD + SSL + per-field sigma** all from Stage 3 | One-line changes each. No buffer zones (CAKO/CAHD/SSL replace them). |
| Ghost approach | **26-neighbor** for faces, edges, corners | 6 face + 12 edge + 8 corner neighbors in 1 parallel pass. |

### What we're skipping (and why)

| Skipped | Why |
|---------|-----|
| Dense output | Only relevant with subcycling, and spatial error dominates temporal. BAM/GRChombo skip it too. |
| Truncation error criterion | Chi-gradient gives identical refinement for vacuum BHs at half cost (arXiv:2312.05438). Add |∇E|+|∇B| in Phase 2 for charge. |
| Vertex-centered | Architectural rewrite. CAKO/CAHD/SSL solve the same noise problem. Cell-centered is standard. |
| Flux correction | CCZ4 isn't conservative form. Maybe add for E_i/B_i only in Phase 2 Einstein-Maxwell. |
| Buffer zones | CAKO/CAHD/SSL achieve 5 OOM more noise reduction at zero memory cost. GRChombo uses 3-block buffers and still has noise. |
| Static refinement (manual) | Chi-gradient with regrid_every=0 (evaluate once) gives the same thing. |
| Box-in-box | Oct-tree subsumes it. Dead-end architecture for N≥3. |
| Per-block GPU kernels | Anti-pattern. MeshBlockPack batching is strictly better. |

---

## Reference Code to Copy From

**Critical rule: copy first, write second.** Every component should start by
reading the reference implementation, understanding the math, then adapting
to our C/SoA architecture. This prevents sign flips and index errors that
take days to debug.

| Component | Primary reference | File(s) to read | What to copy |
|-----------|------------------|-----------------|-------------|
| Prolongation (4th-order Lagrange) | GRChombo | `Source/AMR/QuadCFInterp.hpp` | Interpolation weights/coefficients |
| Restriction | GRChombo | `Source/AMR/FourthOrderFillPatch.hpp` | Averaging pattern |
| Morton encoding | Athena++ | `src/mesh/mesh.cpp`, `src/utils/morton.hpp` | Bit-interleave functions |
| Ghost exchange | Athena++ / AthenaK | `src/bvals/bvals.cpp` | Neighbor finding, slab copy logic |
| Refinement criterion | GRChombo | `Source/AMR/TaggingCriterion.hpp` | Chi-gradient formula |
| 2:1 constraint | Athena++ | `src/mesh/mesh.cpp` (`AdaptiveMeshRefinement`) | Cascade propagation |
| CAKO | arXiv:2404.01137 | Eq. (1)–(3), Table 1 | sigma_CAKO = chi * sigma_base |
| CAHD | arXiv:2404.01137 | Eq. (4)–(5) | kappa1_CAHD = kappa1 * 2^level_offset |
| SSL | arXiv:2404.01137 | Eq. (6)–(8) | Slow-start lapse damping term |
| Per-field sigma | arXiv:2404.01137 | Table 1 | sigma=0.99 gauge, 0.3 physical |
| MeshBlockPack | AthenaK | `src/mesh/meshblock_pack.hpp` | Pack layout, kernel dispatch |
| Subcycling (Option C) | Carpet / CarpetX | `src/driver/driver.cxx` | Level-by-level scheduling |

---

## Debugging & Logging Strategy

Every stage has:
1. **Verbose logging mode** (`--verbose-amr`) that prints per-block diagnostics
2. **NaN/Inf checking** after every ghost exchange and prolongation (debug builds)
3. **Constraint monitoring** — Hamiltonian + momentum L2 norms per level, per step
4. **Ghost exchange validation** — compare multi-block to single-grid (Stage 2 gate)
5. **Convergence testing** — 3-resolution test at every stage that touches numerics

### Log format
```
[AMR] step=100 level=0 blocks=64 dt=0.025 Ham_L2=3.2e-08 Mom_L2=1.1e-07
[AMR] step=100 level=1 blocks=128 dt=0.025 Ham_L2=8.7e-10 Mom_L2=2.3e-09
[AMR] step=100 regrid: checked=yes triggered=no (max_chi_grad=0.042 < threshold=0.1)
[AMR] step=110 regrid: checked=yes triggered=yes (max_chi_grad=0.132 > threshold=0.1)
[AMR]   refined: 4 blocks, coarsened: 0 blocks, total: 196 -> 228 blocks
```

---

## Implementation Stages

### Stage 1: block_t + mesh_t + MeshBlockPack foundation

**Goal:** Define data structures. Single-block mesh through mesh API produces
identical output to current grid_t. MeshBlockPack layout from the start.

**New files:**
```
src/amr/block.h          — block_t struct (grid_t + AMR metadata)
src/amr/block.c          — block allocation/deallocation
src/amr/mesh.h           — mesh_t struct, mesh lifecycle
src/amr/mesh.c           — mesh creation, block management, level iteration
src/amr/morton.h          — Morton encoding/decoding (header-only, copy from Athena++)
src/amr/meshblock_pack.h — MeshBlockPack struct + pack/unpack API
src/amr/meshblock_pack.c — contiguous buffer allocation, field indexing
```

**Data structures:**
```c
typedef struct block_s {
    grid_t  grid;              // existing grid (N=N_BLOCK, fields/rhs/scratch)
    int     level;             // refinement level (0 = coarsest)
    int     id;                // unique block ID
    int64_t morton;            // Morton/Z-order index within level
    double  origin[3];         // physical coords of block's low corner

    // Tree links
    int     parent_id;         // -1 if root
    int     child_ids[8];      // -1 if leaf
    int     neighbor_ids[26];  // 6 face + 12 edge + 8 corner (-1 = boundary)

    // Flags
    int     is_leaf;           // 1 if active (holds data)
    int     on_boundary[6];    // 1 if face touches domain boundary
} block_t;

typedef struct {
    block_t **blocks;          // array of all blocks (indexed by ID)
    int      num_blocks;       // active leaf blocks
    int      max_blocks;       // allocated capacity
    int      max_level;        // deepest level present
    int      N_block;          // interior cells per side (32 default)
    double   L;                // domain size
    double   dx_base;          // coarsest dx
    int      N_root;           // root blocks per side
    rk_method_t rk_method;
    sim_params_t params;
} mesh_t;

// MeshBlockPack: contiguous buffer for GPU batching
typedef struct {
    double *data;              // pack[field * n_blocks * Ntotal + blk * Ntotal + idx]
    double *rhs;               // same layout for RHS
    double *scratch;           // same layout for RK scratch
    int     n_blocks;          // number of packed blocks
    int     n_total;           // points per block (N_block + 2*GHOST_WIDTH)^3
    int     n_fields;          // NUM_FIELDS (25)
    int    *block_ids;         // which block IDs are in this pack
    int     level;             // level of blocks in this pack (-1 if mixed)
} meshblock_pack_t;
```

**Changes to existing files:**
- `Makefile`: add `AMR_SRC`, new test targets
- `src/core/params.h`: add AMR parameters to `sim_params_t`

**Test gate:**
- [ ] Create mesh with 1 block, run flat spacetime test through mesh API
- [ ] Results identical to current grid_t path (Ham L2 < 1e-10)
- [ ] MeshBlockPack with 1 block matches direct block access to roundoff
- [ ] `make` — zero warnings

**Reference code to consult:**
- Athena++ `src/mesh/meshblock.hpp` for block_t design
- AthenaK `src/mesh/meshblock_pack.hpp` for pack layout

---

### Stage 2: Multi-block uniform mesh + 26-neighbor ghost exchange

**Goal:** Decompose domain into N_root^3 blocks at level 0. Ghost zones
filled by direct copy from all 26 neighbors. All physics unchanged.

**New files:**
```
src/amr/ghost_exchange.h  — ghost zone fill API
src/amr/ghost_exchange.c  — 26-neighbor ghost exchange (faces + edges + corners)
```

**Key operations:**

1. **Morton encoding** (copy from Athena++): `(ix, iy, iz) → morton_id` via
   bit-interleaving. All 26 neighbors found by bit arithmetic.

2. **26-neighbor ghost exchange:** Single parallel pass over all neighbor pairs.
   For each of the 26 neighbor types (6 face, 12 edge, 8 corner), copy the
   appropriate slab/column/cube from neighbor's interior into block's ghost zone.
   All 25 fields × GHOST_WIDTH layers.

3. **Boundary-aware Sommerfeld:** Only apply on faces where
   `block->on_boundary[face] == 1`. Inter-block ghost zones filled by exchange.

**Changes to existing files:**
- `src/boundary/sommerfeld.c`: add `apply_sommerfeld_block()` variant
- `src/numerics/rk4.c`: add `rk_step_mesh()` that loops over blocks:
  ```
  ghost_exchange_26(mesh)
  for each leaf block: compute_rhs(block) + sommerfeld_block(block)
  for each leaf block: rk_update(block)
  enforce_algebraic_all(mesh)
  ```

**Test gate:**
- [ ] Decompose N=128 into uniform blocks (4^3=64 blocks of 32^3). Run flat spacetime.
- [ ] Results match single-grid to roundoff (~1e-14 relative error)
- [ ] Single BH through multi-block mesh matches single-grid
- [ ] Edge and corner ghost values verified against expected interior values
- [ ] `make` — zero warnings

**Reference code to consult:**
- Athena++ `src/bvals/bvals.cpp` for neighbor identification
- Athena++ `src/bvals/bvals_cc.cpp` for cell-centered ghost fill
- AthenaK `src/bvals/` for GPU ghost exchange patterns

---

### Stage 3: Prolongation + Restriction + Noise reduction (CAKO/CAHD/SSL/per-field sigma)

**Goal:** Interpolation operators for cross-level data transfer, plus all
noise reduction techniques integrated from day one.

**New files:**
```
src/amr/prolongation.h    — prolongation API
src/amr/prolongation.c    — 4th-order Lagrange coarse→fine interpolation
src/amr/restriction.h     — restriction API
src/amr/restriction.c     — volume-weighted fine→coarse averaging
```

**4th-order Lagrange prolongation** (copy weights from GRChombo):
- 4-point stencil per direction (tensor product in 3D)
- Needs 2 coarse cells on each side of the interpolation point
- Fills ghost zones of fine blocks bordering coarse blocks
- Also used when refining a block (creating child initial data)
- Weights are symmetric: w = {-1/16, 9/16, 9/16, -1/16} for cell midpoints

**Restriction** (volume-weighted average):
- U_coarse = (1/8) × sum of 8 fine cells overlapping this coarse cell
- Conservative by construction
- ~100 lines

**CAKO** (modify `src/evolution/dissipation.c`):
```c
// One-line change: scale sigma by chi
// Before: sigma_eff = sigma
// After:  sigma_eff = chi * sigma_base
double sigma_eff = src[FIELD_CHI][idx] * sigma_for_field[f];
```
arXiv:2404.01137, Eq. (1)–(3). 2 OOM strong-field, 5 OOM wave-zone improvement.

**CAHD** (modify `src/evolution/ccz4_rhs.c`):
```c
// One-line change: scale kappa1 by refinement level
double kappa1_eff = p->ccz4.kappa1 * (1 << (mesh_max_level - block_level));
```
arXiv:2404.01137, Eq. (4)–(5). 3 OOM momentum constraint improvement.

**SSL** (modify gauge section of `src/evolution/ccz4_rhs.c`):
```c
// ~5 lines: temporary damping of initial gauge pulse
double ssl_h = 0.6;       // (3/5)M — from arXiv:2404.01137 Eq. (6)
double ssl_sigma_t = 20.0; // M
double ssl_damp = chi * ssl_h * exp(-t*t / (2.0 * ssl_sigma_t * ssl_sigma_t));
rhs_lapse += -ssl_damp * (lapse - chi);  // drives alpha toward sqrt(chi)
// Negligible after t ~ 170M
```

**Per-field sigma** (modify `src/evolution/dissipation.c`):
```c
// sigma_for_field[f]: 0.99 for gauge (lapse, shift, B), 0.3 for physical
static const double sigma_gauge = 0.99;
static const double sigma_phys  = 0.30;
```
arXiv:2404.01137, Table 1.

**Test gate:**
- [ ] Prolongate smooth Gaussian, verify error O(dx^4) (4th-order Lagrange)
- [ ] Restrict then prolongate, verify conservation (round-trip)
- [ ] CAKO on flat spacetime: Ham L2 unchanged (chi=1 everywhere, no effect)
- [ ] CAHD: single BH with 2 levels, compare constraint L2 with/without
- [ ] SSL: single BH, verify gauge pulse amplitude reduced during early evolution
- [ ] Per-field sigma: verify gauge variables get stronger dissipation in output
- [ ] `make` — zero warnings

**Reference code to consult:**
- GRChombo `Source/AMR/QuadCFInterp.hpp` for Lagrange weights
- GRChombo `Source/AMR/FourthOrderFillPatch.hpp` for restriction
- arXiv:2404.01137 equations (1)–(8), Table 1 for all noise techniques

---

### Stage 4: Oct-tree refinement + multi-level ghost exchange

**Goal:** Refine and coarsen blocks based on chi-gradient. Support multiple
refinement levels with proper cross-level ghost exchange using prolongation
and restriction.

**New files:**
```
src/amr/refine.h          — refine/coarsen API
src/amr/refine.c          — split (1→8) and merge (8→1) operations
src/amr/criterion.h       — refinement criterion API
src/amr/criterion.c       — chi-gradient with every-step check + hysteresis
```

**Every-step check (cheap):**
```c
// ~0.1 ms on GPU: max-reduction over all blocks
bool needs_regrid = false;
for each leaf block b:
    double max_chi_grad = max(|grad(chi)| * dx) over interior points
    if (max_chi_grad > chi_refine && b->level < max_level):
        needs_regrid = true
    // Also check if all 8 siblings can coarsen
```

**Regrid (only when triggered):**
1. Flag blocks for refine/coarsen based on chi-gradient
2. Enforce 2:1 level constraint (cascade: copy from Athena++ `AdaptiveMeshRefinement`)
3. Execute splits: allocate 8 children, prolongate parent data, update tree
4. Execute merges: restrict children data into parent, deallocate, update tree
5. Rebuild MeshBlockPack (repack contiguous buffers)

**Multi-level ghost exchange:**
- Process levels coarsest-first (ensures coarse data available for prolongation)
- Same-level neighbors: direct copy (Stage 2 code)
- Fine block bordering coarse: prolongation fills fine's ghost zones
- Coarse block bordering fine: restriction fills coarse's ghost zones
- 26-neighbor pattern applies at every level

**Changes to existing files:**
- `src/core/params.h`: add AMR parameters:
  ```c
  int    amr_enabled;        // 0 = uniform (default), 1 = AMR
  int    amr_max_level;      // max refinement levels (default 6)
  int    amr_N_block;        // interior cells per side (default 32)
  int    amr_N_root;         // root blocks per side (default 4)
  double amr_chi_refine;     // refine threshold (default 0.1)
  double amr_chi_coarsen;    // coarsen threshold (default 0.01)
  int    amr_check_regrid;   // 1 = every step (default), 0 = fixed interval
  int    amr_regrid_interval;// fixed interval if check disabled (default 10)
  ```

**Test gate:**
- [ ] Single BH at center with 2 refinement levels
- [ ] Blocks near puncture refined to level 2 (dx = dx_base/4)
- [ ] Constraints bounded and convergent across refinement boundaries
- [ ] Results improve vs uniform grid at same memory budget
- [ ] 2:1 constraint: no block >1 level different from any neighbor
- [ ] Regrid check fires correctly: refines when BH approaches block boundary
- [ ] 3-resolution convergence test on uniform-block mesh: still 4th+ order
- [ ] `make` — zero warnings

**Reference code to consult:**
- Athena++ `src/mesh/mesh.cpp` — `AdaptiveMeshRefinement()` for 2:1 cascade
- GRChombo `Source/AMR/TaggingCriterion.hpp` for chi-gradient
- Athena++ `src/mesh/meshblock_tree.cpp` for tree split/merge operations

---

### Stage 5: Level-by-level subcycling (Option C)

**Goal:** Finer levels take smaller timesteps. All blocks at the same level
advance together (GPU-friendly batching by level).

**New files:**
```
src/amr/subcycling.h      — subcycling scheduler API
src/amr/subcycling.c      — level-by-level time advance with restriction
```

**Algorithm (Option C: level-by-level):**
```
advance_mesh(mesh, dt_coarse):
    for level = 0 to max_level:
        n_substeps = 2^(level)          // level 0: 1 step, level 1: 2, etc.
        dt_level = dt_coarse / n_substeps
        for step = 0 to n_substeps-1:
            ghost_exchange_level(mesh, level)   // same-level + cross-level
            for each block at level:
                compute_rhs(block)
                apply_sommerfeld(block)         // only domain boundaries
            for each block at level:
                rk_update(block, dt_level)
            enforce_algebraic_level(mesh, level)
            restrict_to_coarser(mesh, level)    // update coarse data
```

**Cross-level ghost data at intermediate times:**
Without dense output (which we're skipping), use **linear interpolation in time**
between coarse-level snapshots. This gives 2nd-order temporal accuracy at
refinement boundaries. BAM and most NR codes operate this way — spatial
prolongation error (4th-order Lagrange) dominates anyway.

```c
// For fine block needing coarse ghost at time t_fine:
// coarse has data at t_n and t_{n+1}
double theta = (t_fine - t_n) / (t_n1 - t_n);  // 0 to 1
U_coarse_interp = (1 - theta) * U_n + theta * U_n1;
// Then prolongate U_coarse_interp into fine ghost zones
```

This requires storing the coarse solution at 2 time levels (current + previous).
Memory cost: 1 extra field set per coarse block. For typical runs where coarse
blocks are a small fraction of total, this is ~5-10% memory overhead.

**Changes to existing files:**
- `src/numerics/rk4.c`: add `rk_step_mesh_subcycled()` that calls subcycling scheduler
- `src/amr/mesh.c`: add `mesh_store_previous_level()` for temporal interpolation

**Test gate:**
- [ ] Single BH: subcycled result matches global-dt result to time integration error
- [ ] Constraint norms don't grow faster with subcycling than without
- [ ] Measure speedup vs global dt (expect 2-4x for 4+ levels)
- [ ] Verify restriction after each fine substep keeps coarse grid consistent
- [ ] `make` — zero warnings

**Reference code to consult:**
- Carpet `src/driver/driver.cxx` for level-by-level scheduling
- CarpetX for temporal interpolation at refinement boundaries
- Athena++ `src/mesh/mesh.cpp` — `MeshBlock::Step()` for per-level stepping

---

### Stage 6: GPU optimization + MeshBlockPack tuning

**Goal:** Full GPU performance. All blocks packed, single kernel launches,
on-device ghost exchange.

**Changes to existing files:**
```
src/amr/meshblock_pack.c  — add on-device ghost exchange
src/backend/backend_gpu.c — add backend_compute_rhs_packed()
```

**On-device ghost exchange:**
- Pack boundary slabs into send buffers on device
- Device-to-device copy for neighbor data
- Unpack into ghost zones — all without host round-trip

**Kernel dispatch:**
```c
// Single kernel over all blocks at this level
#pragma omp target teams distribute parallel for
for (int bidx = 0; bidx < n_blocks * Ntotal; bidx++) {
    int b = bidx / Ntotal;
    int idx = bidx % Ntotal;
    ccz4_rhs_point_packed(rhs_pack, src_pack, b, idx, params);
}
```

**Performance optimizations (apply all "free" tricks):**
- Morton-sorted block arrays for cache-friendly ghost exchange
- Page-aligned allocations (4096 bytes) for zero-copy GPU mapping
- Fused RK update: all 25 fields in single memory pass
- Pre-load chi once per point for CAKO (don't re-fetch 25 times)
- CAHD kappa1_eff computed once per block (constant per block)
- Sommerfeld only on boundary faces (not entire block volume)
- Precomputed Lagrange prolongation weights (lookup table)
- Async I/O in separate thread
- BMI2 `_pdep_u64`/`_pext_u64` for Morton encode/decode on x86
- Inner x-loop unit-stride aligned for auto-vectorization

**Test gate:**
- [ ] GPU batch results match CPU sequential to roundoff
- [ ] Performance: <1 sec/step at N_eff=256 on A100
- [ ] Ghost exchange on-device: no host-device transfer for same-level exchange
- [ ] Profile: kernel occupancy, memory bandwidth, launch overhead
- [ ] `make BACKEND=gpu` — zero warnings

**Reference code to consult:**
- AthenaK `src/mesh/meshblock_pack.hpp` for pack dispatch
- AthenaK `src/bvals/` for on-device ghost exchange
- Parthenon docs for kernel fusion patterns

---

## File Summary

### New files (18):
```
src/amr/block.h              — block_t struct
src/amr/block.c              — block allocation/free
src/amr/mesh.h               — mesh_t struct, lifecycle
src/amr/mesh.c               — mesh creation, block management
src/amr/morton.h              — Morton encoding (header-only, copy from Athena++)
src/amr/ghost_exchange.h     — ghost zone fill API
src/amr/ghost_exchange.c     — 26-neighbor ghost exchange
src/amr/prolongation.h       — prolongation API
src/amr/prolongation.c       — 4th-order Lagrange interpolation
src/amr/restriction.h        — restriction API
src/amr/restriction.c        — volume-weighted averaging
src/amr/refine.h             — refine/coarsen API
src/amr/refine.c             — oct-tree split/merge + 2:1 constraint
src/amr/criterion.h          — refinement criterion API
src/amr/criterion.c          — chi-gradient with every-step check
src/amr/subcycling.h         — subcycling scheduler API
src/amr/subcycling.c         — level-by-level time advance
src/amr/meshblock_pack.h     — MeshBlockPack struct + pack/unpack
src/amr/meshblock_pack.c     — contiguous buffer, GPU dispatch
```

### Modified files (8):
```
Makefile                      — add AMR_SRC, new test targets
src/core/params.h             — add AMR parameters
src/boundary/sommerfeld.c     — add block-aware variant
src/numerics/rk4.c            — add rk_step_mesh(), rk_step_mesh_subcycled()
src/evolution/dissipation.c   — CAKO + per-field sigma
src/evolution/ccz4_rhs.c      — CAHD + SSL
src/backend/backend_gpu.c     — add packed batch dispatch
docs/amr_decision_guide.html  — update as decisions change
```

### New test files (5):
```
tests/test_amr_mesh.c         — single-block mesh = grid_t
tests/test_amr_ghost.c        — 26-neighbor ghost exchange correctness
tests/test_amr_prolongation.c — 4th-order Lagrange accuracy
tests/test_amr_single_bh.c    — single BH with multi-level AMR
tests/test_amr_subcycling.c   — subcycled vs global dt comparison
```

---

## Testing Strategy

Each stage must pass its test gate before proceeding. Tests are cumulative —
later stages must not break earlier tests.

| Stage | Test | Pass criterion |
|-------|------|----------------|
| 1 | Flat spacetime via mesh (1 block) | Ham L2 < 1e-10, identical to grid_t |
| 1 | MeshBlockPack 1-block = direct block | Match to roundoff |
| 2 | 64 uniform blocks, flat spacetime | Match single-grid to ~1e-14 |
| 2 | 64 uniform blocks, single BH | Match single-grid to ~1e-14 |
| 2 | Edge/corner ghost values | Match expected interior values |
| 3 | Prolongate smooth function | Error O(dx^4) |
| 3 | Restrict + prolongate round-trip | Conservation verified |
| 3 | CAKO flat spacetime | Ham L2 unchanged |
| 3 | CAHD single BH 2 levels | Constraint L2 improved vs no-CAHD |
| 3 | SSL single BH | Gauge pulse amplitude reduced early |
| 4 | Single BH, 2 levels | Constraints bounded, better than uniform |
| 4 | 2:1 constraint enforcement | No block >1 level diff from neighbor |
| 4 | Regrid check triggers correctly | Refines when BH moves near boundary |
| 4 | 3-resolution convergence | 4th+ order on uniform blocks |
| 5 | Subcycled = global dt | Match to time integration error |
| 5 | Constraint growth rate | No worse with subcycling |
| 6 | GPU batch = CPU sequential | Match to roundoff |
| 6 | Performance target | <1 sec/step N_eff=256 on A100 |

---

## Progress Tracking

### Status: STAGE 2 COMPLETE

- [x] **Stage 1:** block_t + mesh_t + MeshBlockPack foundation
- [x] **Stage 2:** Multi-block uniform mesh + 26-neighbor ghost exchange
- [ ] **Stage 3:** Prolongation + Restriction + CAKO/CAHD/SSL/per-field sigma
- [ ] **Stage 4:** Oct-tree refinement + multi-level ghost exchange
- [ ] **Stage 5:** Level-by-level subcycling
- [ ] **Stage 6:** GPU optimization + MeshBlockPack tuning

### Blockers / Open questions
- None currently

### Completed milestones
- **Stage 2** (2026-02-16): 13/13 tests pass. Ghost exchange + multi-block evolution.
  - `src/amr/ghost_exchange.h/c` — 26-neighbor ghost exchange (unified index computation)
  - `src/boundary/sommerfeld.h/c` — added `apply_sommerfeld_block()` (domain boundaries only)
  - `src/numerics/rk4.h/c` — added `rk4_step_mesh()` (CK45 + classic RK4 for multi-block)
  - `src/initial_data/puncture.h/c` — added `set_brill_lindquist_global()` (block-aware coords)
  - `tests/test_amr_ghost.c` — polynomial exchange, BL ghost match, multi-block flat + BH evolution
  - Multi-block flat: Ham L2 = 4.5e-14 (ratio 1.01x vs single-grid 4.4e-14)
  - Ghost values match single-grid to 0 error (31232 points × 25 fields)
  - Multi-block single BH: constraints bounded at 2e-02

- **Stage 1** (2026-02-16): 33/33 tests pass. All data structures implemented.
  - `src/amr/block.h/c` — block_t with LogicalLocation, nblevel[3][3][3] (Athena++ pattern)
  - `src/amr/mesh.h/c` — mesh_t with Morton-sorted blocks, 26-neighbor finding
  - `src/amr/meshblock_pack.h/c` — MeshBlockPack with page-aligned contiguous buffers
  - `src/amr/morton.h` — header-only Morton encoding/decoding
  - `src/core/params.h` — amr_params_t added to sim_params_t
  - `tests/test_amr_mesh.c` — Morton, topology, evolution, pack round-trip tests
  - 1-block mesh Ham L2 = 5.3e-14 (identical to grid_t path)
  - 8-block pack load/store: zero error
  - Existing tests (flat, convergence) pass with no regressions

---

## References

### AMR Architecture
- **Athena++:** Stone et al. 2020, ApJS 249, 4 — oct-tree AMR, MeshBlock
- **GR-Athena++:** Daszuta et al. 2021, ApJS 257, 25 — BBH with oct-tree AMR
- **AthenaK:** Grete et al. 2024, arXiv:2409.16053 — GPU batching, MeshBlockPack
- **Parthenon-VIBE:** arXiv:2509.19701 — block size GPU benchmarks (90x penalty for 16^3 vs 32^3)
- **GRChombo:** Andrade et al. 2021, JOSS 6(68), 3703 — block-structured AMR with CCZ4

### Noise Reduction
- **arXiv:2404.01137** (Etienne 2024) — CAKO + CAHD + SSL + per-field sigma

### Refinement Strategy
- **arXiv:2312.05438** — chi-gradient vs truncation error comparison

### Other
- **Berger & Oliger 1984** — original AMR paper
- **p4est:** Burstedde et al. 2011 — scalable oct-tree library
- **Carpet:** Schnetter et al. 2004 — Berger-Oliger for Einstein Toolkit
