# Oct-Tree AMR Implementation Plan

## Context

The head-on binary BH test (Milestone 4) showed that uniform grids can't resolve
both the near-puncture region (needs dx ~ 0.1) and the far-field wave zone
(r ~ 50M) within 16 GB RAM. At N=128, dx=0.5 — the BH horizon (r=M/2=0.5) is
barely 1 grid point. AMR is needed to proceed to binary inspiral (Milestone 5).

After researching all AMR approaches (see `docs/amr_options.html`), **oct-tree
block AMR** was chosen over block-structured (Berger-Oliger) because:
- Uniform 16-32^3 blocks batch perfectly for GPU (vs variable-sized boxes)
- Memory scales linearly with N_BH (no bounding-box waste)
- Proven for NR: GR-Athena++ (Daszuta et al. 2021) runs BBH with this approach
- Simpler tree operations than Berger-Rigoutsos clustering

Design follows Athena++ (Stone et al. 2020) and GR-Athena++ patterns, adapted
for our C/OpenMP architecture.

---

## Refinement Boundary Noise: The Central Problem

Every AMR code in numerical relativity suffers from **constraint violations at
refinement boundaries**. When data is interpolated between coarse and fine grids,
truncation error introduces high-frequency noise that accumulates over thousands
of steps. This is well-documented across all major NR codes:

- **GRChombo** docs call refinement boundaries a "significant source of inaccuracy"
- **BAM** has conservation violations when matter crosses boundaries
- **Carpet/Einstein Toolkit** requires careful per-level dissipation tuning
- **GR-Athena++** mitigates via vertex-centered discretization

### Who uses what

| Code | AMR type | Boundary noise strategy |
|------|----------|------------------------|
| GRChombo | Block-structured (Chombo) | Buffer zones, CCZ4 constraint damping, conservative refluxing |
| Einstein Toolkit (Carpet) | Block-structured (Berger-Oliger) | Per-level dissipation `epsdis_for_level`, subcycling |
| Einstein Toolkit (CarpetX) | Block-structured (AMReX) | Same as Carpet + AMReX refluxing |
| BAM | Block-structured | BSSN/Z4c, flux correction (partial — source terms not matched) |
| GR-Athena++ | Oct-tree (Athena++) | Vertex-centered discretization, minmod prolongation |
| Dendro-GR | Oct-tree + wavelets | Wavelet-based fine-grain adaptivity |
| SpEC | Pseudospectral multi-domain | No FD grid — no interpolation noise |
| NRPy+/BlackHoles@Home | Multi-patch (no traditional AMR) | **CAKO + CAHD + SSL** (2-5 OOM improvement) |

### Three key techniques from arXiv:2404.01137 (Etienne 2024)

This paper introduces three techniques that achieve **2-5 orders of magnitude**
reduction in constraint violations at refinement boundaries. Validated across
100+ BBH configurations (q=1-6, spin up to 0.5). Available in Einstein Toolkit's
BaikalVacuum thorn but **not yet widely adopted** by other codes.

**1. CAKO — Curvature-Adjusted Kreiss-Oliger Dissipation**

Standard KO dissipation uses a constant sigma everywhere. CAKO scales sigma by
the conformal factor W = chi, so dissipation is strong in weak-field regions
(where interpolation noise lives) and vanishes near punctures (where fields are
physically sharp):
```
sigma_CAKO = W * sigma_base
```
Uses 9th-order KO on an 11-point stencil (vs our current 6th-order / 7-point).
sigma_base = 0.99 for gauge variables, 0.3 for others.

**Result:** 2 OOM reduction in strong-field Hamiltonian, 5 OOM in wave zone.

**2. CAHD — Coarse-Grid-Adjusted Hamiltonian Damping**

Standard CCZ4 kappa1 is constant across all levels. CAHD increases constraint
damping on coarser grids where violations accumulate most:
```
kappa1_CAHD = kappa1_base * 2^level_offset
```
On the coarsest grid (5 levels up from finest): 32x stronger damping.

**Result:** 3 OOM reduction in momentum constraint violations.

**3. SSL — Slow-Start Lapse**

During early evolution, the 1+log gauge generates a sharp superluminal gauge
pulse that propagates at sqrt(2)c from each puncture. When this pulse crosses
AMR refinement boundaries, it reflects and generates substantial noise. SSL
temporarily damps this pulse during startup:
```
dt(alpha) = [standard 1+log] - W * h * exp(-t^2 / (2*sigma_t^2)) * (alpha - W)
```
h = (3/5)M, sigma_t = 20M. Effect negligible after t ~ 170M.

**Result:** 4.3x noise reduction in dominant GW mode, higher modes (l >= 4)
now detectable above noise floor.

### Our strategy: implement CAKO + CAHD + SSL from day 1

Most codes implement "standard" AMR, then spend months fighting constraint
noise. We will implement these three techniques alongside AMR from the start:
- CAKO is a one-line change to dissipation.c (multiply sigma by chi)
- CAHD is a one-line change to ccz4_rhs.c (scale kappa1 by level)
- SSL is ~5 lines in the gauge condition (temporary exponential damping)
All three are simple modifications — no architectural changes needed.

**Open question:** Do GR-Athena++ or Dendro-GR use any equivalent of CAKO/CAHD?
Their papers don't mention these specific techniques. The vertex-centered
approach in GR-Athena++ may partially mitigate the same issues. Worth
investigating during implementation to see if our oct-tree approach needs all
three or if the uniform block structure already helps.

---

## Architecture Overview

### Core Concept

`ccz4_rhs_point(rhs, src, g, p, i, j, k)` is already per-point. It only needs
`g->Ntotal`, `g->dx`, and `g->fields[]` to compute derivatives via FD macros.
**If each AMR block has the same `grid_t` interface, all physics code works
unchanged.** This is the key insight — AMR is purely a mesh management layer
on top of the existing physics.

### Data Flow (per RK stage)

```
  ghost_exchange(mesh)           <-- NEW: fill ghost zones from neighbors
  for each block b:
    backend_compute_rhs(b)       <-- UNCHANGED: per-point RHS
    if is_outer_boundary(b):
      apply_sommerfeld(b)        <-- MINOR CHANGE: only on domain boundary faces
  for each block b:
    rk_update(b)                 <-- UNCHANGED: field updates
```

### Block Size Decision

Use **N_block = 16** (interior cells per side). With GHOST_WIDTH=4:
- Total: 24^3 = 13,824 points per block
- Memory: 25 fields x 3 blocks (CK45) x 13,824 x 8 bytes = **8.3 MB/block**
- Ghost overhead: 3.4x (24^3/16^3) — significant but acceptable at this size
- Matches GR-Athena++ practice for BBH simulations
- Trade-off: 32^3 blocks have 2x less ghost overhead but 8x fewer blocks per
  level, reducing adaptivity. Start with 16^3, make configurable.

### Stencil Requirements for Ghost Width

| Stencil | Points | Ghost needed |
|---------|--------|-------------|
| fd_d1 (4th-order centered) | +-2 | 2 |
| fd_d2 (4th-order centered) | +-2 | 2 |
| fd_d2_mixed | +-2 each dir | 2 |
| fd_adv (4th-order upwind) | +-3 | 3 |
| fd_ko (6th-order KO) | +-3 | 3 |
| **GHOST_WIDTH = 4** | | **4 (safe margin)** |

Athena++ requires NG=4 for 4th-order MHD and NG must be even for AMR
(restriction reduces by factor 2). Our GHOST_WIDTH=4 satisfies both.

---

## Implementation Stages

### Stage 1: Block + Mesh Data Structures

**Goal:** Define `block_t` and `mesh_t`. Single-block mesh behaves identically
to current `grid_t`. All existing tests pass unchanged.

**New files:**
- `src/amr/block.h` — `block_t` struct definition
- `src/amr/block.c` — block allocation/deallocation
- `src/amr/mesh.h` — `mesh_t` struct definition, mesh lifecycle
- `src/amr/mesh.c` — mesh creation, block management

**Data structures:**

```c
// block_t wraps grid_t + AMR metadata
typedef struct block_s {
    grid_t  grid;           // existing grid (N=N_block, fields/rhs/scratch)
    int     level;          // refinement level (0 = coarsest)
    int     id;             // unique block ID
    int64_t morton;         // Morton/Z-order index within level
    double  origin[3];      // physical coords of block's low corner

    // Tree links
    int     parent_id;      // -1 if root
    int     child_ids[8];   // -1 if leaf (not refined)
    int     neighbor_ids[6];// face-neighbor IDs (-1 = domain boundary)

    // Flags
    int     is_leaf;        // 1 if active (holds data), 0 if refined
    int     on_boundary[6]; // 1 if face touches domain boundary
} block_t;

// mesh_t manages the oct-tree
typedef struct {
    block_t **blocks;       // array of all blocks (sparse, indexed by ID)
    int      num_blocks;    // number of active (leaf) blocks
    int      max_blocks;    // allocated capacity
    int      max_level;     // deepest refinement level present
    int      N_block;       // interior cells per block side (16)
    double   L;             // domain size
    double   dx_base;       // coarsest dx = L / (N_block * N_root)
    int      N_root;        // root blocks per side (e.g. 4 = 4^3=64 root blocks)
    rk_method_t rk_method;
    sim_params_t params;
} mesh_t;
```

**Changes to existing files:**
- `Makefile`: add `AMR_SRC = src/amr/block.c src/amr/mesh.c` to ALL_SRC
- `src/core/grid.h`: no changes (grid_t used as-is inside block_t)

**Test:** Create a mesh with 1 block, run flat spacetime test through mesh API.
Verify results identical to current grid_t path.

---

### Stage 2: Multi-Block Uniform Mesh + Ghost Exchange

**Goal:** Decompose domain into N_root^3 blocks at level 0. Ghost zones filled
by copying from neighbor blocks. All physics unchanged.

**New files:**
- `src/amr/ghost_exchange.h` — ghost zone fill API
- `src/amr/ghost_exchange.c` — same-level ghost copy implementation
- `src/amr/morton.h` — Morton encoding/decoding, neighbor finding

**Key operations:**

1. **Morton encoding:** `(ix, iy, iz) -> morton_id` using bit-interleaving.
   Neighbors found by incrementing/decrementing Morton coords.

2. **Ghost exchange (same level):** For each face between blocks A and B,
   copy A's interior cells adjacent to the face into B's ghost zone (and vice
   versa). 4 layers deep (GHOST_WIDTH). Edge and corner ghosts filled by
   successive face exchanges (simpler than direct corner-to-corner copy).

3. **Boundary-aware Sommerfeld:** `apply_sommerfeld` modified to only apply on
   ghost zone faces that touch the domain boundary (where `on_boundary[face]=1`).
   Inter-block ghost zones are filled by ghost exchange, not Sommerfeld.

**Changes to existing files:**
- `src/boundary/sommerfeld.c`: add `apply_sommerfeld_block()` variant that only
  applies Sommerfeld on outer-boundary faces (check `block->on_boundary[]`)
- `src/numerics/rk4.c`: add `rk4_step_mesh()` that loops over blocks:
  ```
  ghost_exchange_same_level(mesh)
  for each block:
    backend_compute_rhs(block->grid)
    apply_sommerfeld_block(block)  // only on outer boundaries
  for each block:
    ck45_update(block->grid)
  enforce_algebraic_all(mesh)
  ```

**Test:** Decompose N=64 into 4^3=64 blocks of 16^3. Run flat spacetime and
single BH tests. Results must match single-grid within roundoff (~1e-14).
This validates ghost exchange correctness.

---

### Stage 3: Prolongation + Restriction + Noise Reduction

**Goal:** Interpolation operators for moving data between refinement levels,
with CAKO/CAHD/SSL integrated from the start to prevent refinement boundary
noise.

**New files:**
- `src/amr/prolongation.c` — coarse-to-fine interpolation
- `src/amr/restriction.c` — fine-to-coarse averaging
- `src/amr/interp.h` — shared declarations

**Prolongation (coarse -> fine):**
- Slope-limited linear interpolation with **minmod limiter**
  (following Athena++ — sharper limiters cause artifacts at refinement boundaries)
- For each fine cell, compute value from coarse cell + limited gradient
- Fills ghost zones of fine blocks that border coarse blocks
- Also used when refining a block (creating initial data for children)

**Restriction (fine -> coarse):**
- Volume-weighted average: U_coarse = mean(U_fine) over the 2^3=8 fine cells
  that map to each coarse cell
- Used after fine-level time steps to update coarse-level data
- Conservative by construction

**CAKO integration** (modify `src/evolution/dissipation.c`):
```c
// Before: rhs[f][idx] += sigma * (fd_ko(...) + fd_ko(...) + fd_ko(...));
// After:  rhs[f][idx] += sigma * chi * (fd_ko(...) + fd_ko(...) + fd_ko(...));
```
Scale KO dissipation by chi (conformal factor). Near punctures chi -> 0 so
dissipation vanishes (preserving physical gradients). In weak-field regions
chi ~ 1 so full dissipation applied (suppressing interpolation noise).

**CAHD integration** (modify `src/evolution/ccz4_rhs.c`):
```c
// Scale kappa1 by 2^(level_max - level) for this block's level
double kappa1_eff = p->ccz4.kappa1 * (1 << (mesh_max_level - block_level));
```
Block level passed via `sim_params_t` or a per-block parameter. Coarser grids
get exponentially stronger constraint damping.

**SSL integration** (modify gauge section of `src/evolution/ccz4_rhs.c`):
```c
// Temporary damping of gauge pulse during early evolution
double ssl_h = 0.6;  // (3/5)M
double ssl_sigma = 20.0;  // M
double ssl_damp = chi * ssl_h * exp(-t*t / (2.0*ssl_sigma*ssl_sigma));
rhs_lapse += -ssl_damp * (lapse - chi);  // drives alpha toward sqrt(chi)
```
Negligible after t ~ 170M. Prevents superluminal gauge pulse from generating
noise at refinement boundaries.

**Test:**
- Prolongate a smooth function (e.g. Gaussian), restrict back, verify error
  is O(dx^2) (limited by minmod's 2nd-order accuracy)
- Prolongate single BH initial data onto a refined block, verify constraints
  remain bounded after prolongation
- CAKO: flat spacetime with uniform blocks, verify Ham L2 unchanged (chi=1)
- CAHD: single BH with 2 levels, compare constraint L2 with and without CAHD

---

### Stage 4: Oct-Tree Refinement + Multi-Level Ghost Exchange

**Goal:** Refine and coarsen blocks based on chi gradient. Support multiple
refinement levels with proper ghost exchange across levels.

**New files:**
- `src/amr/refine.c` — refinement/coarsening logic
- `src/amr/criterion.c` — refinement criterion (chi gradient)

**Refinement criterion:**
- Compute `|grad(chi)| * dx` at each interior point of each block
- If max exceeds threshold `chi_refine` (e.g. 0.1): flag block for refinement
- If max below `chi_coarsen` (e.g. 0.01) for all 8 siblings: flag for coarsening
- Enforce 2:1 level constraint: no block may be >1 level coarser than neighbor

**Refine operation (split 1 block -> 8 children):**
1. Allocate 8 new child blocks at level+1
2. Set child dx = parent dx / 2
3. Prolongate parent field data into children (minmod)
4. Update tree links (parent, children, neighbors)
5. Deallocate parent's field data (keep tree node for potential coarsening)

**Coarsen operation (merge 8 children -> 1 parent):**
1. Restrict children's field data into parent
2. Deallocate children
3. Update tree links

**Multi-level ghost exchange:**
- Same-level neighbors: direct copy (Stage 2)
- Fine block bordering coarse: prolongation fills fine's ghost zones
- Coarse block bordering fine: restriction fills coarse's ghost zones
- Order: coarsest level first, then finer levels (ensures coarse data
  available for prolongation)

**Regridding schedule:** Every N_regrid steps (e.g. 10), re-evaluate criterion
on all blocks. This amortizes the regridding cost.

**Changes to existing files:**
- `src/core/params.h`: add AMR parameters to `sim_params_t`:
  ```c
  int    amr_enabled;       // 0 = uniform grid (default), 1 = AMR
  int    amr_max_level;     // max refinement levels (default 6)
  int    amr_N_block;       // interior cells per block side (default 16)
  int    amr_N_root;        // root blocks per side (default 4)
  double amr_chi_refine;    // chi gradient threshold for refinement
  double amr_chi_coarsen;   // chi gradient threshold for coarsening
  int    amr_regrid_every;  // re-check refinement every N steps
  ```

**Test:** Single BH at center with 2 refinement levels. Verify:
- Blocks near puncture refined to level 2 (dx = dx_base/4)
- Constraints bounded and convergent across refinement boundaries
- Results improve compared to uniform grid at same memory budget

---

### Stage 5: Time Subcycling (Optional — can start with global dt)

**Goal:** Finer levels take smaller time steps, reducing total work.

**Strategy:**
- **Start with global timestep** (all levels share dt = CFL * dx_finest).
  Simpler, no inter-level synchronization. Athena++ uses this approach.
- **Add subcycling later** if performance requires it: each level L takes
  2^(L_max - L) steps per coarse step, with restriction after each fine step.

**Implementation (global timestep first):**
- dt = CFL * dx_min where dx_min is the finest level's spacing
- All blocks advance together — simple loop, no scheduling complexity
- Cost: coarse blocks do more steps than needed, but they're cheap (few blocks)

**Implementation (subcycling, later):**
- Recursive: advance level L by one coarse step = 2 fine steps at level L+1
- After each fine step: restrict to coarse, fill coarse ghost zones
- Flux correction at refinement boundaries

**Test:** Compare subcycled vs global dt for single BH. Verify identical results
to within time integration error. Measure speedup from subcycling.

---

### Stage 6: GPU Batching + Optimization

**Goal:** Efficient GPU execution by batching blocks.

**New files:**
- `src/amr/meshblock_pack.h` — pack/unpack blocks for GPU
- `src/amr/meshblock_pack.c` — implementation

**MeshBlockPack (following AthenaK pattern):**
- Pack all leaf blocks at same level into one contiguous buffer
- Layout: `pack[field][block_idx][k][j][i]` — block index as outer loop
- Single `omp target teams distribute` kernel over all blocks x all points
- Unpack results back into individual block arrays

**Ghost exchange on GPU:**
- Pack boundary slabs into send buffers on device
- Copy to neighbor's receive buffers
- Unpack into ghost zones
- All on-device — no host round-trip for same-level exchange

**Changes to existing files:**
- `src/backend/backend_gpu.c`: add `backend_compute_rhs_packed()` that operates
  on a MeshBlockPack instead of a single grid

**Performance target:** At N_eff = 256 (equivalent resolution with AMR), achieve
< 1 sec/step on A100 GPU.

---

## File Summary

### New files (14):
```
src/amr/block.h          — block_t struct
src/amr/block.c          — block allocation/free
src/amr/mesh.h           — mesh_t struct, lifecycle
src/amr/mesh.c           — mesh creation, block management
src/amr/ghost_exchange.h — ghost zone fill API
src/amr/ghost_exchange.c — same-level + cross-level ghost exchange
src/amr/morton.h         — Morton encoding, neighbor finding
src/amr/prolongation.c   — coarse-to-fine interpolation (minmod)
src/amr/restriction.c    — fine-to-coarse volume-weighted averaging
src/amr/interp.h         — prolongation/restriction declarations
src/amr/refine.c         — refine/coarsen operations
src/amr/criterion.c      — chi-gradient refinement criterion
src/amr/meshblock_pack.h — GPU batch packing (Stage 6)
src/amr/meshblock_pack.c — GPU batch implementation (Stage 6)
```

### Modified files (7):
```
src/core/params.h         — add AMR parameters to sim_params_t
src/boundary/sommerfeld.c — add boundary-aware variant for blocks
src/numerics/rk4.c        — add rk4_step_mesh() for multi-block stepping
src/evolution/dissipation.c — CAKO: scale KO dissipation by chi
src/evolution/ccz4_rhs.c  — CAHD: level-scaled kappa1; SSL: slow-start lapse
src/backend/backend_gpu.c — add packed batch dispatch (Stage 6)
Makefile                  — add AMR_SRC, new test targets
```

### New test files (4):
```
tests/test_amr_ghost.c        — ghost exchange correctness
tests/test_amr_prolongation.c — interpolation accuracy
tests/test_amr_single_bh.c    — single BH with refinement
tests/test_amr_head_on.c      — head-on binary with AMR
```

---

## Testing Strategy

Each stage has its own validation before proceeding:

| Stage | Test | Pass criterion |
|-------|------|----------------|
| 1 | Flat spacetime via mesh API (1 block) | Ham L2 < 1e-10 (identical to grid_t) |
| 2 | Flat + single BH decomposed into 64 blocks | Results match single-grid to roundoff |
| 2 | Ghost exchange correctness | Interior values at block boundaries match |
| 3 | Prolongation of smooth function | Error O(dx^2) |
| 3 | Restriction round-trip | Conservation verified |
| 3 | CAKO on flat spacetime | Ham L2 unchanged (chi=1 everywhere) |
| 4 | Single BH with 2 levels | Constraints bounded, better than uniform |
| 4 | 2:1 constraint enforcement | No block > 1 level diff from neighbor |
| 5 | Global dt = subcycled dt | Results match to time integration error |
| 6 | GPU batch = CPU sequential | Results match to roundoff |

**Convergence preservation:** After Stage 4, run 3-resolution convergence test
on the uniform-block mesh (no refinement). Must still show 4th+ order.
Refinement boundaries may reduce local order to 2 (due to minmod) but global
convergence should remain > 3.

---

## CLAUDE.md Updates Needed

After implementation begins, update CLAUDE.md to reflect:
1. Move AMR from Phase 2 item 5 to **current active work** (between Milestones 4 and 5)
2. Change AMR description from "block-structured (Berger-Oliger)" to "oct-tree block"
3. Add `src/amr/` to project structure
4. Add AMR parameters to the CCZ4 parameters table
5. Add AMR-specific invariants (2:1 level constraint, minmod prolongation)
6. Add AMR references (Athena++, GR-Athena++, AthenaK, arXiv:2404.01137)
7. Update Phase 1 current status to reflect Milestones 1-4 done, AMR in progress

---

## References

### AMR Architecture
- **Athena++:** Stone et al. 2020 (ApJS 249, 4) — oct-tree AMR design
- **GR-Athena++:** Daszuta et al. 2021 (ApJS 257, 25) — BBH with oct-tree AMR
- **AthenaK:** Grete et al. 2024 (arXiv:2409.16053) — GPU batching with MeshBlockPack
- **arXiv:2312.05438** — AMR refinement strategy comparison (box-in-box vs sphere vs truncation error)

### Refinement Boundary Noise Reduction
- **arXiv:2404.01137** (Etienne 2024) — CAKO + CAHD + SSL, 2-5 OOM improvement
- **arXiv:2112.10567** — Lessons for AMR in numerical relativity
- **arXiv:1504.01266** — Conservative mesh refinement for NS mergers
- **arXiv:1003.0859** — Position-dependent eta (Muller & Brugmann)

### Existing Code Approaches
- **GRChombo:** arXiv:1503.03436 — block-structured AMR with CCZ4
- **Einstein Toolkit:** arXiv:1111.3344 — Carpet AMR with per-level dissipation
- **Dendro-GR:** wavelet-based oct-tree AMR
- **NRPy+/BlackHoles@Home:** multi-patch, home of CAKO/CAHD/SSL development

---

## Verification

After all stages complete:
1. `make` — zero warnings
2. `make test` — all existing tests pass (backward compatible)
3. `make test-amr` — all AMR-specific tests pass
4. `make test-convergence` — 4th-order convergence on uniform blocks
5. Head-on binary with AMR: constraints bounded, merger detected, memory < 4 GB
6. Single BH with 3 levels: trumpet lapse matches analytic value closer than
   uniform N=128
7. CAKO/CAHD comparison: measure constraint L2 with and without, verify
   improvement consistent with Etienne 2024 results
