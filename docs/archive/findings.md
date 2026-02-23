# Binary Inspiral Convergence Test: Failure Analysis & AMR Plan

**Date:** 2026-02-22
**Test:** `test_inspiral_convergence.c` — AMR binary inspiral convergence (3 resolutions)
**Result:** FAIL — measured order 0.30, expected 4.0
**Wall time:** 38.1 hours

---

## Test Parameters

| Resolution | N_block | N_eff | dx (M) | dt (M) | t_final (M) |
|------------|---------|-------|--------|--------|-------------|
| LOW        | 32      | 96    | 0.6667 | 0.1667 | 1000.0      |
| MED        | 48      | 144   | 0.4444 | 0.1111 | 666.7       |
| HIGH       | 64      | 192   | 0.3333 | 0.0833 | 500.0       |

Physics: equal-mass non-spinning, d=10M, P_y=0.0939 (3PN), L=64M domain,
uniform AMR mesh (max_level=0), classic RK4, CFL=0.25, 6000 steps.

---

## What the Log Shows

### Anti-convergence in the Hamiltonian constraint

At the same physical time, the higher-resolution runs have **larger** constraint
violations — the opposite of convergence:

| t (M)  | Ham_LOW  | Ham_MED  | Ham_HIGH | Ratio HIGH/LOW |
|--------|----------|----------|----------|----------------|
| 0      | 8.06e-03 | 6.21e-03 | 4.21e-03 | 0.52 (good)    |
| 16.7   | 1.43e-02 | 2.37e-02 | 3.31e-02 | 2.31 (bad)     |
| 100.0  | 1.04e-01 | 1.88e-01 | 2.87e-01 | 2.76 (bad)     |
| 250.0  | 1.51e-01 | 2.74e-01 | 4.19e-01 | 2.77 (bad)     |

For 4th-order convergence with ratio 1.5x, Ham_HIGH should be 5.06x **smaller**
than Ham_LOW. Instead it is 2.8x **larger**.

### Constraint growth (stable but large)

All three runs are numerically stable — no NaN, no blowup. But constraints grow
monotonically from O(1e-3) to O(1e-1), a factor of 20–110x, then plateau:

- LOW saturates at ~0.172 by t ~ 500M
- MED saturates at ~0.310
- HIGH saturates at ~0.468

### Convergence factor Q

Q was never good. Mean Q = 1.13 (expected 5.06). It oscillated wildly early
(0.32 to 6.20 at steps 200–300) then settled to ~0.9–1.0 at late times.

---

## Root Causes (ranked by severity)

### 1. CRITICAL BUG: Same-step comparison instead of same-time comparison

The test compares Hamiltonian constraint values at the **same step number** across
resolutions. But since dt = CFL × dx differs per resolution, the same step number
corresponds to completely different physical times:

- Step 1000: LOW at t=166.7M, MED at t=111.1M, HIGH at t=83.3M
- Step 6000: LOW at t=1000.0M, MED at t=666.7M, HIGH at t=500.0M

The convergence ratio Q = |H_low − H_med| / |H_med − H_high| is only meaningful
when all three solutions are evaluated at the **same physical time**. Here the
differences are dominated by different evolutionary states (different numbers of
orbits completed), not by truncation error.

**The passing single-BH test (`test_convergence.c`) does it correctly:** it
computes a different number of steps per resolution to reach the same T_final=2M,
then compares final constraint values.

### 2. CRITICAL: All resolutions are below the convergent regime

Production NR codes use AMR with finest grid spacing of **M/42 to M/160** near
punctures. Our uniform grid has dx = 0.33–0.67M, which is **20–50x too coarse**.

| Code               | Finest dx near puncture | Our dx    | Ratio   |
|--------------------|------------------------|-----------|---------|
| BAM                | M/80 typical           | M/1.5     | 53x     |
| AthenaK            | M/64 to M/128          | M/1.5     | 43–85x  |
| GR-Athena++        | M/51 to M/153          | M/1.5     | 34–102x |
| Einstein Toolkit   | M/42 to M/72           | M/1.5     | 28–48x  |

Points across the gravitational radius (~0.5M) at each resolution:

| Resolution | dx (M) | Points across r_g |
|------------|--------|-------------------|
| LOW        | 0.6667 | 0.75              |
| MED        | 0.4444 | 1.12              |
| HIGH       | 0.3333 | 1.50              |

At LOW resolution, the gravitational radius is **smaller than a single grid cell**.
The 4th-order convergence formula e ~ C·dx⁴ requires smooth solutions on the scale
of dx. The puncture has unbounded derivatives (chi ~ r⁴ as r→0), so the error does
not follow a power law at these resolutions.

### 3. SEVERE: No exclusion of puncture region from L2 norm

The global L2 norm `mesh_constraint_l2()` integrates over the entire domain
**including the puncture neighborhood** where constraint violations are O(1) and
non-convergent. The passing single-BH test uses `constraint_l2_annular()` with
5M < r < 25M, which excludes the puncture and boundary regions.

Higher resolution resolves **more** of the puncture's steep gradients, producing
**larger** finite-difference errors in absolute terms. This is why the constraints
get worse with resolution — classic pre-convergent behavior.

### 4. SEVERE: Domain too small (L=64M, boundary at 32M)

The outer boundary is only 27M from the nearest puncture. Boundary reflections
reach the BHs at t ~ 54M and make 9–18 round-trips during the simulation.

| Code               | Outer boundary | Our boundary | Ratio   |
|--------------------|---------------|--------------|---------|
| BAM                | ~240M         | 32M          | 7.5x    |
| AthenaK            | ~1024M        | 32M          | 32x     |
| Einstein Toolkit   | ~1365M        | 32M          | 43x     |
| MAYA catalog       | >400M         | 32M          | 12.5x   |

The Sommerfeld BCs use 2nd-order one-sided stencils, so reflected wave amplitude
scales as O(dx²). After 54M of evolution, boundary errors dominate interior errors
and degrade measured convergence from 4th to ~2nd order.

### 5. MODERATE: Evolution time far too long

For a convergence test at these resolutions, t_final should be short enough that
all three resolutions are still accurate. The passing single-BH test uses
T_final=2M. This test runs to 500–1000M, by which time the constraints have
saturated at O(0.1) — any convergence information is lost.

### 6. MODERATE: Gauge advection disabled

Both `lapse_advec_coeff` and `shift_advec_coeff` default to 0.0 (no advection
terms in the gauge equations). GRChombo and most production codes advect both
lapse and shift. The non-advecting variant produces stronger gauge transients
that are harder to control at coarse resolution.

### 7. MINOR: CAKO and per-field dissipation disabled

Without chi-adjusted KO dissipation (CAKO), the uniform sigma=0.3 does not
suppress near punctures where it should. Per-field sigma (0.99 for gauge, 0.3
for physical) would help damp gauge transients (ref: arXiv:2404.01137).

---

## What Is NOT Wrong

- **Finite difference stencils:** Correctly 4th-order (verified by inspection).
- **RK4 integrator:** Standard textbook implementation, correct.
- **CCZ4 equations:** Match GRChombo reference line-by-line.
- **Initial data:** Bowen-York with pre-collapsed lapse (alpha = psi⁻²), standard.
- **KO dissipation:** 6th-order (preserves 4th-order convergence). Correct.
- **Numerical stability:** All runs complete without NaN or blowup.

---

## What Production Codes Actually Do

### No code has ever demonstrated binary inspiral convergence on a uniform grid

AMR is effectively mandatory because the problem spans multiple scales:

| Scale              | Requirement     |
|--------------------|----------------|
| Puncture structure | dx ~ M/50–M/100 |
| Orbital dynamics   | ~10M            |
| Wave zone          | ~100M           |
| Outer boundary     | 200–1000M       |

On a uniform grid, resolving the puncture at M/50 with boundary at 200M would
require N = 20,000 per side — a 20,000³ grid, which is impossible. AMR with
10–14 levels of factor-2 refinement provides the necessary 1024–16384x dynamic
range.

### Published convergence results

| Code         | Measured order | Finest dx | Outer boundary | AMR levels | Source                    |
|--------------|---------------|-----------|----------------|------------|---------------------------|
| AthenaK      | 4th (clean)   | M/64–M/128| ±1024M         | 10–11      | arXiv:2409.10383          |
| GR-Athena++  | 4th (q=1,2)   | M/51–M/153| large          | 12–14      | arXiv:2411.11989          |
| BAM          | 4th           | M/57–M/95 | ~240M          | 10         | arXiv:0709.2160           |
| MAYA catalog | 2.1–2.9       | ~M/48     | >400M          | many       | arXiv:2309.00262          |
| Etienne 2024 | improving     | M/42–M/72 | ~1365M         | 11         | arXiv:2404.01137          |

Key finding from Etienne (2024): M/42 "lies slightly outside the convergent
regime" for a GW150914-like binary. Our finest resolution of M/3 is nowhere
close.

Even mature production codes (MAYA catalog) measure order 2.1–2.9 with 6th-order
spatial FD, attributed to AMR interpolation effects. Clean 4th order requires
being well inside the convergent regime.

### Comparison with our test vs production standards

| Parameter | Our failing test | Production standard | Source |
|-----------|-----------------|---------------------|--------|
| Finest dx | M/1.5 – M/3 | M/64 – M/128 | AthenaK, GR-Athena++ |
| Domain half-width | 32M | 400 – 2048M | All codes |
| Refinement levels | 0 | 10 – 14 | All codes |
| Block size | 32–64 | 16 (CPU), packed 16 or 32 (GPU) | AthenaK, GRChombo |
| Refinement criterion | none (uniform) | L2 distance from punctures (primary) + chi-gradient (secondary) | Rashti et al. 2023 |
| Regrid interval | n/a | 32–64 coarse steps (inner levels), static (outer) | GRChombo, ET |
| Subcycling | n/a | Yes (2.5–4.5x speedup) | CarpetX 2025 |
| Time interp at boundaries | Linear (in code) | Dense output / 4th-order (CarpetX 2025) | Key for convergence |
| KO dissipation | Uniform sigma=0.3 | CAKO: sigma(x) = chi(x) * sigma_base | Etienne 2024 |

---

## Current State of the AMR System

Our AMR infrastructure is **fully implemented and tested**:
- Berger-Oliger subcycling (per-level packing, recursive)
- Multi-level ghost exchange (5-phase, coarse-buffer architecture)
- 4th-order prolongation (5-point Lagrange, AthenaK weights)
- 4th-order restriction (4-point symmetric Lagrange)
- Chi-gradient refinement criterion
- 2:1 level constraint enforcement
- Oct-tree block hierarchy with Morton ordering
- Temporal interpolation for subcycling (`ghost_fill_from_coarser`)
- Block-aware Sommerfeld BCs
- AH finder on AMR meshes
- AMR-aware 1D output slices

**Test coverage:**
- `test_amr_mesh.c` (8/8): block creation, Morton ordering, neighbor finding
- `test_amr_ghost.c` (5/5): same-level and multi-level ghost exchange
- `test_amr_prolong.c` (9/9): prolongation, restriction, CAKO/CAHD/SSL
- `test_amr_refine.c` (9/9): split, merge, 2:1 constraint, full regrid
- `test_subcycle.c` (3/3): uniform identity, flat stability, BH evolution
- `test_pack_evolve.c` (3/3): packed vs per-block validation

**What has NOT been tested:**
- max_level > 2 (deep refinement hierarchies)
- Binary inspiral WITH active AMR refinement
- Many regrid cycles on a moving binary over hundreds of M
- Time interpolation accuracy at coarse-fine boundaries during long subcycling runs

---

## Key Findings from Literature Review

### 1. AMR is mandatory for binary inspiral convergence

No published code has ever demonstrated binary inspiral convergence on a uniform
grid. The dynamic range (M/100 near puncture to 2000M outer boundary = 200,000:1)
requires 10–14 levels of factor-2 refinement.

### 2. MAYA catalog only gets order 2.1–2.9

Even with mature AMR (Einstein Toolkit + Carpet), the MAYA waveform catalog
(arXiv:2309.00262) measures convergence order 2.1–2.9, not the theoretical 4th.
Root cause: **2nd-order time interpolation** at coarse-fine boundaries during
subcycling. CarpetX fixed this in 2025 using RK4 dense output. Our code uses
linear temporal interpolation in `ghost_fill_from_coarser()` — same limitation.

### 3. Prolongation should be 5th-order

Rule: prolongation order >= FD convergence order + 1. For 4th-order FD, need at
least 5th-order prolongation. Our 5-point Lagrange stencil is formally 4th-order
(degree-4 polynomial). Upgrading to 6-point (5th-order) would use one more stencil
point; our 4-ghost-zone width supports this.

### 4. Most codes use puncture-tracking, not pure chi-gradient

Production codes use **L2 distance from puncture positions** as the primary
criterion, with chi-gradient as secondary. The L2 (spherical) method from Rashti
et al. 2023 (arXiv:2312.05438) creates **60% fewer blocks** than box-in-box and
produces smoother convergence with less regridding noise. Chi-gradient alone
creates and destroys blocks too aggressively.

### 5. Block size of 16³ is universal standard

AthenaK, GR-Athena++, and GRChombo all use N_block=16 for CPU runs. For GPU,
AthenaK packs multiple 16³ MeshBlocks into MeshBlockPacks (same pattern as our
`meshblock_pack.h`). Larger blocks (32–64) waste memory in refined regions and
reduce load balancing flexibility.

### 6. CAKO dissipation reduces constraint violations by ~2 orders of magnitude

Chi-adjusted KO: `sigma_eff(x) = chi(x) * sigma_base`. Near punctures where
chi → 0, this gives near-zero dissipation (preventing over-dissipation of the
steep fields). In the wave zone where chi ≈ 1, full dissipation strength.
Combined with per-field sigma (0.99 for gauge, 0.3 for physical), this is
the state of the art (arXiv:2404.01137, Etienne 2024).

---

## Memory Estimates for AMR Inspiral

### Memory per block

Formula: `31 fields × (N_block + 8)³ × rk_blocks × 8 bytes`

| N_block | CK45 (3 blocks) | Classic RK4 (4 blocks) |
|---------|-----------------|------------------------|
| 16      | 9.8 MB          | 13.1 MB                |
| 32      | 45.4 MB         | 60.6 MB                |

### AMR hierarchy estimate (N_root=8, N_block=16, L=128M, max_level=6, CK45)

| Level | dx (M) | What it covers | Est. leaf blocks | Memory |
|-------|--------|----------------|-----------------|--------|
| 0 | 1.0 | Whole domain (±64M) | ~490 | 4.8 GB |
| 1 | 0.5 | Around binary (~30M) | ~40 | 0.4 GB |
| 2 | 0.25 | Near each BH (~15M) | ~24 | 0.2 GB |
| 3 | 0.125 | BH vicinity (~7M) | ~16 | 0.2 GB |
| 4 | 0.0625 | Horizon region (~3M) | ~12 | 0.1 GB |
| 5 | 0.031 | Puncture (~1.5M) | ~10 | 0.1 GB |
| 6 | 0.016 (M/64) | Puncture core | ~8 | 0.1 GB |
| **Total** | | | **~600** | **~6 GB** |

### Domain size vs memory trade-off

| N_root | Domain L | Boundary | Root blocks | Root memory (CK45) |
|--------|----------|----------|-------------|-------------------|
| 4 | 64M | 32M | 64 | 0.6 GB |
| 6 | 96M | 48M | 216 | 2.1 GB |
| 8 | 128M | 64M | 512 | 5.0 GB |
| 12 | 192M | 96M | 1728 | 16.9 GB |
| 16 | 256M | 128M | 4096 | 40.1 GB |

**33 GB machine limit:** N_root=8 (L=128M) fits easily. N_root=12 (L=192M)
is tight. N_root=16 (L=256M) exceeds memory.

---

## Runtime Estimates for AMR Inspiral

### Calibration from the uniform test

From the inspiral.log timing data:
- LOW (27 blocks of 32³, classic RK4): ~2.3 sec/step
- MED (27 blocks of 48³): ~6.5 sec/step
- HIGH (27 blocks of 64³): ~14 sec/step
- Derived: ~1.3 μs per point per classic RK4 step (4 stages)
- CK45 (5 stages): ~1.6 μs per point per step

### Subcycling cost per coarse step

With Berger-Oliger, level L takes 2^L sub-steps per coarse step. For
N_root=8, N_block=16, max_level=6, CK45:

| Level | Leaf blocks | Sub-steps | Block-steps/coarse step | % of work |
|-------|-------------|-----------|------------------------|-----------|
| 0 | ~490 | 1 | 490 | 29% |
| 1 | ~40 | 2 | 80 | 5% |
| 2 | ~24 | 4 | 96 | 6% |
| 3 | ~16 | 8 | 128 | 8% |
| 4 | ~12 | 16 | 192 | 11% |
| 5 | ~10 | 32 | 320 | 19% |
| 6 | ~8 | 64 | 512 | **30%** |
| **Total** | | | **~1818** | |

Cost per block-step (16³ block, CK45): (16+8)³ × 1.6μs ≈ 0.022 sec
Cost per coarse step: ~1818 × 0.022 ≈ **40 sec**

### 3-resolution convergence test runtime

Varying N_block at 1.5x ratio (16 → 24 → 32), t_final=200M (~1 orbit):

| Resolution | N_block | Finest dx | Coarse dt | Steps | sec/step | Wall time |
|------------|---------|-----------|-----------|-------|----------|-----------|
| LOW | 16 | M/64 | 0.25M | 800 | ~40 | **~9 hours** |
| MED | 24 | M/96 | 0.167M | 1200 | ~95 | **~32 hours** |
| HIGH | 32 | M/128 | 0.125M | 1600 | ~180 | **~80 hours** |
| **Total** | | | | | | **~121 hours (~5 days)** |

### Comparison with the failed uniform test

| | Uniform test (38h) | AMR test (~121h) |
|-|-------------------|-----------------|
| Finest dx | M/3 (useless) | M/64 to M/128 (production) |
| Boundary | 32M (too small) | 64M (marginal) |
| t_final | 500–1000M | 200M |
| Convergence? | No (order 0.3) | Expected yes (order ~3–4) |

The AMR test is ~3x longer but actually resolves the physics. The uniform test
was faster but produced no useful convergence information.

### Budget-constrained option

For a faster (~45h) test with slightly lower resolution:

```
N_root=6, N_block=16, L=96M, max_level=5, CK45, t_final=100M
3 resolutions: N_block = 16, 24, 32
```

| Resolution | Finest dx | Wall time |
|------------|-----------|-----------|
| LOW (16) | M/32 | ~3h |
| MED (24) | M/48 | ~12h |
| HIGH (32) | M/64 | ~30h |
| **Total** | | **~45 hours** |

Memory peak: ~8 GB. M/32 is marginal (Etienne says M/42 is "slightly outside
the convergent regime"), but M/48 and M/64 should be in or near the regime.

### Why not skip subcycling? (global timestep)

AthenaK uses a global timestep for GPU efficiency. But:
- dt_global = CFL × dx_finest = 0.25 × 0.016 = 0.004M
- Steps for t=200M: 50,000
- Cost per step: ~600 blocks × 0.022 sec ≈ 13 sec
- Total: 50,000 × 13 ≈ **180 hours** (50% worse than subcycling)

Subcycling wins by ~1.5x because most blocks are coarse (level 0 at dt=0.25M
vs global dt=0.004M).

---

## Recommended Fixes (Implementation Plan)

### Fix 1: Fix convergence comparison methodology

Compare at the same physical time, not the same step number. Either:
- Run different step counts per resolution (as `test_convergence.c` does)
- Interpolate the Hamiltonian history to common physical times

### Fix 2: Use annular exclusion region for constraint norms

For binary: exclude r < 5M from each puncture AND r > 0.8 × L_boundary.
This requires tracking puncture positions (already needed for L2 refinement
criterion).

### Fix 3: Add L2-distance refinement criterion

Primary: refine blocks within spherical shells around each puncture at
prescribed radii per level. Secondary: chi-gradient for any features missed.
Ref: Rashti et al. 2023 (arXiv:2312.05438) — 60% fewer blocks than box-in-box.

### Fix 4: Enable AMR with sufficient refinement levels

Target configuration:
```
N_root=6, N_block=16, L=96M, max_level=5, CK45
chi_refine=0.08, chi_coarsen=0.01, regrid_every=32
```

### Fix 5: Enable gauge advection

Set `lapse_advec_coeff = 1.0` and `shift_advec_coeff = 1.0` for binary
evolutions, matching GRChombo and most production codes.

### Fix 6: Enable CAKO + per-field dissipation

CAKO: `sigma_eff(x) = chi(x) * sigma_base`
Per-field: sigma=0.99 for gauge (lapse, shift, B^i), sigma=0.3 for physical.
Ref: arXiv:2404.01137 (Etienne 2024).

### Fix 7: Upgrade prolongation to 5th-order

Add one stencil point to the Lagrange prolongation (6-point, degree-5).
Rule: prolongation order >= FD order + 1. 4-ghost-zone width supports this.

### Fix 8 (future): Upgrade temporal interpolation to 4th-order

Replace linear time interpolation in `ghost_fill_from_coarser()` with RK4
dense output (polynomial from stage vectors). Without this, temporal convergence
at coarse-fine boundaries is limited to 2nd order — the MAYA catalog limitation.
Ref: CarpetX arXiv:2503.09629.

---

## Summary

The convergence test failure has **two classes of cause**:

1. **Test methodology bugs** (fixes 1–2): comparing at wrong times, including
   the puncture singularity in the norm. These would cause failure even with
   a perfect evolution code.

2. **Inadequate resolution for the problem** (fixes 3–7): binary inspiral on a
   uniform grid with dx ~ M/1.5–M/3 and boundary at 32M is 20–50x too coarse
   and 8–40x too small compared to production standards. AMR is not optional
   for this problem — it is a prerequisite.

The evolution code itself (CCZ4 RHS, finite differences, time integration,
dissipation) is correctly implemented and verified by the passing single-BH
convergence test at order 5.4. The issue is entirely in the test setup and
resolution requirements for binary inspiral.

---

## References

- arXiv:2409.10383 — AthenaK performance-portable NR (2024)
- arXiv:2411.11989 — GR-Athena++ BBH waveforms (2024)
- arXiv:2312.05438 — AMR refinement criterion comparison (Rashti et al. 2023)
- arXiv:2404.01137 — Improved moving-puncture techniques (Etienne 2024)
- arXiv:2503.09629 — CarpetX GPU subcycling with dense output (2025)
- arXiv:2309.00262 — Second MAYA waveform catalog (2023)
- arXiv:0709.2160 — BAM high-spin BBH mergers (Marronetti et al.)
- arXiv:2505.01495 — GRChombo 25-BH cluster simulation (2025)
- arXiv:2406.09139 — Cell-centered vs vertex-centered AMR (2024)
- arXiv:2406.11626 — ExaGRyPE NR solvers (2024)
- arXiv:2506.06838 — AthenaK GW150914 simulations (2025)
