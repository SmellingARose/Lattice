# GPU Multigrid Solver V-Cycle Divergence — Investigation

**Date:** March 5, 2026
**Platform:** NVIDIA H100 80GB
**Config:** N=32, L=6144, 11 AMR levels, ~777 blocks, BY 1-field initial data
**Symptom:** FAS V-cycle residual diverges exponentially after FMG

```
[AMR-MG-GPU] FMG done: residual = 1.850788e-01
[AMR-MG-GPU] V-cycle 1: residual = 3.396290e+00    ← 18x increase
[AMR-MG-GPU] V-cycle 2: residual = 5.026468e+01    ← 15x increase
```

CPU solver converges at ~0.24/cycle on identical mesh. All unit tests pass (2-5 AMR levels).

---

## Previously Fixed Bugs

Two bugs from the earlier investigation (commit 3d73914) have already been fixed:

### Bug #1: MAX_SOLVER_SLOTS Overflow (FIXED — commit fb7b3d9)

`MAX_SOLVER_SLOTS` was hardcoded to 8 but 11 AMR levels need 12 slots. Levels 8-10 were
complete no-ops on GPU. Fixed by changing to `MAX_SOLVER_SLOTS = MAX_AMR_LEVELS` (16).

### Bug #2: Coarse-Data Stride Mismatch (FIXED — in backend_hip.cpp:3786)

Ghost exchange kernels were using `n_sol` (1 or 4) as the coarse_data field stride, but
the buffer was allocated with `MG_AMR_N_FIELDS = 10`. Fixed by using `sp->n_fields`
instead of `n_sol`. Comment on lines 3782-3785 documents the fix.

**Despite both fixes, V-cycles still diverge.** The remaining issue(s) are described below.

---

## Remaining Bug: 8-Color Gauss-Seidel + 6th-Order Stencil Race Condition

### The Problem

The GPU smoother uses 8-color (stride-2) checkerboard ordering. All same-color points
are updated simultaneously by GPU threads. The Newton-GS update at each point calls
`fd_d2()`, a 6th-order 7-point stencil (radius 3):

```
fd_d2(psi, idx, sx=1, inv_dx) reads: psi[idx-3], psi[idx-2], psi[idx-1],
                                       psi[idx], psi[idx+1], psi[idx+2], psi[idx+3]
```

Two same-color points at positions `i` and `i+2` in the x-direction:
- Thread at `i` reads `psi[i-3..i+3]` and writes `psi[i]`
- Thread at `i+2` reads `psi[i-1..i+5]` and writes `psi[i+2]`
- **Overlap:** Thread at `i+2` reads `psi[i]` while thread at `i` is writing it

For 8-color to be race-free, the stencil radius must be ≤ 1 (3-point, 2nd-order).
A 6th-order stencil (radius 3) requires stride ≥ 7, i.e., 343 colors (7^3) — impractical.

### Why It Matters on GPU but Not CPU

| Platform | Execution | Effect |
|----------|-----------|--------|
| CPU serial (relaxation.c) | Sequential loop, deterministic order | "Gauss-Seidel with leakage" — converges deterministically |
| CPU OMP (relaxation_amr.c) | Parallel across blocks, serial within each block | Race only across block boundaries (ghost zone overlap), small impact |
| GPU (backend_hip.cpp) | All same-color points in all blocks execute simultaneously | Full race within every block — non-deterministic read-modify-write |

On GPU, the smoother becomes a non-deterministic hybrid between Jacobi and Gauss-Seidel.
At small scale (2-5 AMR levels, few blocks), the corrections are small and the race is
tolerable. At 11 levels with 777 blocks, the error compounds across level interfaces.

### Why It Causes Divergence (Not Just Slow Convergence)

A standard Jacobi smoother has spectral radius < 1 and always converges (slowly). The
race-corrupted smoother is worse than Jacobi because:

1. **Non-deterministic partial updates:** Each sweep produces a different result depending
   on warp scheduling. The V-cycle cannot make consistent progress.

2. **FAS amplification:** In the FAS composite V-cycle, the coarse correction depends on
   accurate smoothing at every level. If the fine-level smoother fails to reduce
   high-frequency error, the restricted residual is wrong, the tau correction is wrong,
   and the prolongated correction makes things worse. With 11 levels of recursion, each
   level amplifies the error from the level above.

3. **Ghost exchange timing:** Between GS colors, a same-level ghost exchange propagates
   the race-corrupted values to neighboring blocks, which then use them in subsequent
   colors. Over 4 pre-smooth sweeps × 8 colors × 11 levels, corruption accumulates.

### Evidence

- CPU converges at 0.24/cycle (correct algorithm, serial/deterministic smoothing)
- GPU diverges at 18x/cycle (same algorithm, non-deterministic smoothing)
- All other V-cycle operations (restriction, prolongation, tau, save, operator, residual,
  ghost exchange, BCs) have been verified correct by line-by-line comparison

---

## Proposed Fixes

### Fix A: Defect Correction (Recommended)

The standard approach used by HPGMG, AMReX, Dendro, and GPU multigrid codes
(arXiv:2510.11152):

1. **Smoother:** Use 2nd-order Laplacian (3-point, radius 1) in the Newton-GS update.
   Compatible with 8-color ordering. The smoother only needs to damp high-frequency
   error — it does NOT need to be high-order accurate.

2. **Residual/operator:** Keep the full 6th-order stencil. The residual computation
   reads from a frozen solution (no in-place updates), so there is no race.

3. **Jacobian diagonal:** Change `MGP_FD_D2_CENTER` in the smoother from `-49/18`
   (6th-order) to `-2.0` (2nd-order).

**Implementation:**

```c
// mg_smooth_point.h — add 2nd-order fd_d2 for smoother only:
LATTICE_DEVICE
static inline double fd_d2_2nd(const double *f, int idx, int stride, double inv_dx) {
    return (f[idx - stride] - 2.0 * f[idx] + f[idx + stride]) * inv_dx * inv_dx;
}

// In mg_smooth_1field_point: replace fd_d2 calls with fd_d2_2nd
// Change J_lap = 3.0 * (-2.0) / dx2  (instead of 3.0 * (-49/18) / dx2)
```

**Effort:** ~20 lines changed in `mg_smooth_point.h`. No changes to operator or
residual kernels. Smoother convergence rate decreases slightly (more V-cycles needed)
but each V-cycle actually converges instead of diverging.

### Fix B: Weighted Jacobi Smoother (Alternative)

Replace in-place Gauss-Seidel with explicit weighted Jacobi:

1. Compute update for all points (read from `data`, write to `scratch`)
2. Apply: `data = data + omega * (scratch - data)` with `omega ≈ 0.8`

**Pros:** No coloring needed, trivially parallel, no races.
**Cons:** Needs extra buffer, slower convergence per sweep (need more sweeps),
requires omega tuning.

### Fix C: 64-Color Ordering (Overkill but Exact)

Use stride-4 coloring (4^3 = 64 colors) to guarantee radius-3 stencil safety.
Each color has (N/4)^3 points per block.

**Pros:** True Gauss-Seidel, best convergence rate.
**Cons:** 64 kernel launches per sweep × 4 sweeps = 256 launches per smoothing pass.
Ghost exchange needed between every color. Impractical overhead.

---

## Other Observations

### 1. Same-Level Ghost Exchange is Serialized per Thread

`hip_mg_ghost_same_level` assigns **one thread per (block, direction)**. Each thread
loops over the entire ghost slab (up to `4 × N × N = 4096` doubles per field).
With 777 blocks × 26 directions = ~20K threads, each doing ~4K iterations, this is
extremely poor GPU utilization.

**Impact:** Performance only (10-100x slower than optimal). Does not cause divergence.

**Fix:** One thread per ghost point (not per direction). Grid = nb × 26 × ghost_pts.

### 2. Memory Waste: NUM_FIELDS vs COMPILED_NUM_FIELDS

`mesh_create` uses `NUM_FIELDS = 31` for block allocation even when EM is disabled
(`COMPILED_NUM_FIELDS = 25`). Wastes 24% memory. With 777 blocks × 4 arrays × 31 fields
× 40^3 × 8 bytes = ~39 GB. At 25 fields: ~31 GB.

**Impact:** Memory only. The solver packs correctly use `MG_AMR_N_FIELDS = 10`.

### 3. Level-0 CPU Fallback Creates PCIe Stall

Every V-cycle, level 0 does GPU→host sync, CPU uniform MG solve, host→GPU sync.
This creates a pipeline bubble but doesn't affect correctness. Can be fixed by running
the level-0 smoother on GPU (50 sweeps × 8 colors directly on device) after the
main divergence bug is fixed.

---

## Verified Correct (No Issues Found)

| Component | Status |
|-----------|--------|
| FAS tau correction formula (rhs += L(u_restricted)) | Correct |
| V-cycle recursion structure (GPU matches CPU) | Correct |
| Restriction kernel (8-child average, fine→coarse data+residual) | Correct |
| Prolongation kernel (trilinear correction, coarse→fine) | Correct |
| Save kernel (scratch = data, solution fields only) | Correct |
| Residual kernel (accum = rhs - accum) | Correct |
| Operator kernel (L(u) using correct backgrounds) | Correct |
| Zero-leaf-rhs kernel (is_parent guard) | Correct |
| Ghost exchange field count (nf = sp->n_fields = 10) | Correct |
| Boundary conditions (zero-Dirichlet on boundary faces) | Correct |
| cf_map construction (coarser-level pack index per direction) | Correct |
| is_parent mask construction and upload | Correct |
| Solver slot assignment (slot = level index, MAX=16) | Correct |
| Field slot consistency (0-3 solution, 4-9 background) | Correct |
| FMG ascending loop (prolongate + V-cycle per level) | Correct |
| Background precomputation (all levels, before GPU upload) | Correct |
| HIP stream ordering (single stream, in-order guaranteed) | Correct |
| Child map construction for cross-level restriction/prolongation | Correct |
| L2 norm computation (rhs - accum, block-level reduction) | Correct |

---

## Recommended Action Plan

1. **Implement Fix A (defect correction)** — 2nd-order smoother + 6th-order operator/residual.
   This is the standard practice in every production GPU multigrid code.

2. **Test on H100** — V-cycles should converge at ~0.3-0.5/cycle (2nd-order smoother is
   slightly slower than exact GS but correct).

3. **Optimize ghost exchange** (Observation #1) — one thread per ghost point for proper
   GPU utilization. Low priority vs correctness.

4. **Add level-0 GPU path** (Observation #3) — eliminate PCIe stall. Low priority.

---

*Investigation by Claude, March 5, 2026. Code audited: `relaxation_amr.c`,
`backend_hip.cpp`, `mg_smooth_point.h`, `meshblock_pack.c`, `backend.h`.*
