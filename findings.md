# Binary Inspiral Convergence Test: Failure Analysis

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

---

## Recommended Fixes

### Fix 1: Fix the convergence comparison methodology

Compare at the same physical time, not the same step number. Either:
- Run different step counts per resolution (as `test_convergence.c` does)
- Interpolate the Hamiltonian history to common physical times

### Fix 2: Use an annular exclusion region

Replace `mesh_constraint_l2()` with an annular norm that excludes r < 5M from
each puncture and r > 0.8·L_boundary, matching `constraint_l2_annular()`.

### Fix 3: Enable AMR refinement for inspiral

Use the existing AMR infrastructure with sufficient refinement levels (6–8+)
to achieve dx ~ M/40–M/80 near the punctures while extending the boundary to
200M+. This is the standard approach in every production NR code.

### Fix 4: Reduce evolution time for convergence test

If testing on a uniform grid (without AMR), use T_final ~ 10–20M — short enough
that all resolutions are still in the convergent regime and boundary reflections
have not yet arrived.

### Fix 5: Increase domain size

Target outer boundary at 200M minimum (standard in literature).

### Fix 6: Enable gauge advection

Set `lapse_advec_coeff = 1.0` and `shift_advec_coeff = 1.0` for binary
evolutions, matching GRChombo and most production codes.

### Fix 7: Enable CAKO + per-field dissipation

Use chi-adjusted KO (CAKO) with per-field sigma (0.99 gauge, 0.3 physical)
as recommended by arXiv:2404.01137.

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
