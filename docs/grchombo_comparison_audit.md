# SUPERSEDED — See docs/nr_code_comparison_full.html (2026-04-05)

# Lattice vs GRChombo: Comprehensive Code Audit

**Date:** 2026-03-26
**Purpose:** Systematic term-by-term comparison of every subsystem to identify deviations that could cause NaN or instability in QNM ringdown tests.

**Legend:**
- **MATCH** — our code agrees with GRChombo
- **INTENTIONAL DIFFERENCE** — documented improvement (6th-order, GPU, etc.)
- **SUSPICIOUS DIFFERENCE** — deviation that could cause bugs (flagged for investigation)
- **MISSING** — feature GRChombo has that we lack

---

## 1. CCZ4 Physics — RHS, Ricci, Gauge, Constraints, Dissipation

### 1.1 Ricci Tensor (chi corrections, Z terms, conformal Christoffel symbols)

**GRChombo:** `CCZ4Geometry.hpp:56-112` | **Lattice:** `ccz4_rhs.c:220-288`

**Christoffel symbols:**
- **MATCH.** `tensor_utils.h:155-158` computes `LLL[i][j][k] = 0.5*(d1_h[j][i][k] + d1_h[k][i][j] - d1_h[j][k][i])`, matching GRChombo `TensorAlgebra.hpp:350-354` exactly. Raising to ULL and contraction also match.

**covdtilde2chi:**
- **MATCH.** `ccz4_rhs.c:237-240`: `covdtilde2chi[kk][ll] = d2_chi[kk][ll] - chris.ULL[m][kk][ll]*d1_chi[m]`, identical to GRChombo `CCZ4Geometry.hpp:60-63`.

**chris_LLU, boxtildechi, dchi_dot_dchi:**
- **MATCH.** `ccz4_rhs.c:242-249` matches GRChombo `CCZ4Geometry.hpp:66-74` term by term.

**ricci_hat (conformal Ricci without chi or Z):**
- **MATCH.** `ccz4_rhs.c:256-265` reproduces GRChombo `CCZ4Geometry.hpp:78-93` exactly — hat-Gamma trick, all 4 terms verified.

**ricci_chi (conformal factor contribution):**
- **MATCH.** `ccz4_rhs.c:268-273` matches GRChombo `CCZ4Geometry.hpp:96-101`.

**Z terms:**
- **MATCH.** `ccz4_rhs.c:276-280` matches GRChombo `CCZ4Geometry.hpp:33-44`.

**Final assembly + Ricci scalar:**
- **MATCH.** `ricci.LL[i][j] = (ricci_chi + chi*ricci_hat + z_terms)/chi` and `ricci.scalar = chi * trace(ricci.LL, h_UU)`.

### 1.2 CCZ4 RHS for Each Field

#### dt(chi)
- **MATCH.** `ccz4_rhs.c:348-349`: `advec_chi + (2/3)*chi*(lapse*K - divshift)` matches GRChombo line 118-119.

#### dt(h_ij)
- **MATCH.** `ccz4_rhs.c:362-369` matches GRChombo lines 120-129.

#### dt(A_ij)
- **MATCH.** `ccz4_rhs.c:375-397` matches GRChombo lines 131-154. Adot_TF trace-free part, A-squared term (Lattice pre-computes A_mixed for efficiency — algebraically identical).

#### dt(Theta)
- **MATCH.** `ccz4_rhs.c:407-413` matches GRChombo lines 172-180 term by term.
- **INTENTIONAL DIFFERENCE:** No cosmological constant term (vacuum BH spacetimes).

#### dt(K)
- **MATCH.** `ccz4_rhs.c:417-420` matches GRChombo lines 183-188.
- **INTENTIONAL DIFFERENCE:** No cosmological constant.

#### dt(Gamma^i)
- **MATCH.** `ccz4_rhs.c:425-449` matches GRChombo lines 193-222. All 8 terms verified individually.

#### Z vector
- **MATCH.** `Z_over_chi[i] = 0.5*(Gamma[i] - chris.contracted[i])` identical to GRChombo `CCZ4RHS.impl.hpp:80-82`.

#### Covariant second derivative of lapse
- **MATCH.** `ccz4_rhs.c:306-325` matches GRChombo lines 91-112.

### 1.3 Kappa1/2/3 Damping — Covariant vs Non-covariant

- **MATCH.** `ccz4_rhs.c:401-405`: `if (covariant_Z4) kappa1_times_lapse = kappa1; else kappa1_times_lapse = kappa1 * lapse;` — identical to GRChombo `CCZ4RHS.impl.hpp:156-160`.
- **INTENTIONAL DIFFERENCE:** Lattice defaults to covariant formulation (`covariant_Z4 = true`), recommended for strong-field BH evolutions.

### 1.4 Gauge Conditions

**Lapse:** `ccz4_rhs.c:502-505` — **MATCH.** Fast-path for `lapse_power=1.0` is optimization only.

**Shift:** `ccz4_rhs.c:521-522` — **MATCH.** `shift_advec_coeff*advec_shift[i] + shift_Gamma_coeff*B[i]`.

**B (Gamma-driver):** `ccz4_rhs.c:531-533` — **MATCH.** Standard formula.
- **INTENTIONAL DIFFERENCE:** Position-dependent eta supported: `eta_eff = eta / max(sqrt(max(chi, 1e-4)), 1e-2)` (arXiv:1003.0859). GRChombo uses constant eta.

| Parameter | Lattice | GRChombo |
|---|---|---|
| `lapse_power` | 1.0 | 1.0 |
| `lapse_coeff` | 2.0 | 2.0 |
| `shift_Gamma_coeff` | 0.75 | 0.75 |
| `eta` | 1.0 | 1.0 |

### 1.5 Chi/Lapse Floor Values and Where Applied

- **MATCH.** Floor value `1e-4` for both chi and lapse, identical to GRChombo's `PositiveChiAndAlpha.hpp` default.
- Applied in Lattice at: RHS entry (`ccz4_rhs.c:127-128`), algebraic enforcement (`backend_cpu.c:1146-1149`), Psi4 (`psi4.c:270`).

### 1.6 det(h)=1 and tr(A)=0 Enforcement

**tr(A)=0:** **MATCH.** Both use `make_trace_free(A, h, h_UU)`.

**det(h)=1:** **INTENTIONAL DIFFERENCE.** Lattice actively rescales `h_ij *= det(h)^{-1/3}` via `fast_inv_cbrt` (`backend_cpu.c:1105-1140`). GRChombo relies on evolution equations preserving it. Active enforcement is more robust for long evolutions.

### 1.7 KO Dissipation

**Stencil order:** **INTENTIONAL DIFFERENCE.** Lattice uses **8th-order** KO with 6th-order FD (9-point stencil, `finite_diff.h:231-252`). GRChombo uses **6th-order** KO (7-point). GRChombo has 8th-order commented out with matching weights.

**Sign convention:** **MATCH.** Both: positive sigma gives dissipation.

**Sigma handling:** **INTENTIONAL DIFFERENCE.** Lattice supports per-field sigma and CAKO (Chi-Adjusted KO, `sigma_eff = sqrt(chi) * sigma`). GRChombo uses uniform sigma.

### 1.8 Constraint Computation

**Hamiltonian:** **SUSPICIOUS DIFFERENCE.** Lattice's Hamiltonian constraint (`constraints.c:140-186`) uses the CCZ4 Ricci (with Z terms via field Gamma) instead of the pure geometric Ricci (with Z=0). GRChombo explicitly removes Z by calling `compute_ricci_Z_general(..., dZ_coeff=0.0)`. When Z is small, the difference is negligible. **Not a NaN source** — affects constraint monitoring only, not evolution.

**Momentum:** **MATCH.** `constraints.c:259-285` matches GRChombo lines 72-99.

### 1.9 Psi4 Gravitational Wave Extraction

**Physical K_ij, derivatives, Christoffel:** **MATCH.** All tensor algebra verified.

**Electric Weyl tensor:** **MATCH.** Uses `K - Theta` (not `K - 2*Theta`), matching GRChombo.
- **SUSPICIOUS DIFFERENCE (minor):** Lattice uses full CCZ4 Ricci (coefficient 2 on Z terms) vs GRChombo's half-Z Ricci (`dZ_coeff=1.0`). Effect is O(constraint violation) — negligible for well-resolved runs.

**Magnetic Weyl tensor:** **MATCH.** Shift-dependent epsilon terms omitted (standard at extraction radius).

**Tetrad:** **INTENTIONAL DIFFERENCE.** Different Gram-Schmidt ordering (spherical basis vs Cartesian). Produces a sign flip on Re(Psi4) documented at `psi4.c:565`. |Psi4| is unchanged.

**Mode decomposition:** **INTENTIONAL DIFFERENCE.** Lattice implements spin-weighted spherical harmonics in-code. GRChombo does this in post-processing.

### 1.10 Finite Difference Stencil Coefficients

| Stencil | Status |
|---|---|
| 1st derivative (d1) | **MATCH** — `{1/60, 3/20, 3/4}` |
| 2nd derivative (d2) | **MATCH** — `{1/90, 3/20, 3/2, 49/18}` |
| Mixed 2nd derivative | **MATCH** — all 6 weight magnitudes |
| Advection (upwind) | **MATCH** — `{1/30, 2/5, 7/12, 4/3, 1/2, 2/15, 1/60}` |
| KO dissipation | **INTENTIONAL DIFFERENCE** — 8th-order vs 6th-order |

### 1.11 Physics Section Summary

| Item | Status |
|---|---|
| Christoffel symbols | MATCH |
| Conformal Ricci (all terms) | MATCH |
| dt(chi), dt(h_ij), dt(K), dt(A_ij), dt(Theta), dt(Gamma^i) | MATCH |
| Kappa1 covariant/non-covariant | MATCH |
| Gauge: lapse, shift, B | MATCH |
| Chi/lapse floor (1e-4) | MATCH |
| det(h)=1 enforcement | INTENTIONAL DIFFERENCE (active rescaling) |
| tr(A)=0 enforcement | MATCH |
| KO stencil | INTENTIONAL DIFFERENCE (8th-order) |
| Hamiltonian constraint Ricci | SUSPICIOUS (Z contamination, monitoring only) |
| Psi4 Ricci Z coefficient | SUSPICIOUS (minor, O(constraint violation)) |
| All FD coefficients | MATCH |

---

## 2. AMR Flow — Ghost Exchange, Prolongation, Restriction, Subcycling

### 2.1 Ghost Exchange Phase Ordering

**GRChombo** (`GRAMRLevel.cpp:1029-1043`): 2-phase FillPatch: (1) prolongate from coarser level, (2) same-level exchange overwrites where neighbors exist.

**Lattice** (`ghost_exchange.c:734-773`): 5-phase coarse-buffer architecture:
- Phase 0+1: Same-level exchange at each level
- Phase 2: Restrict fine interior into coarse_buf
- Phase 3: Fill coarse_buf ghosts from siblings + coarse neighbors
- Phase 3.5: Extrapolate coarse_buf boundary ghost cells
- Phase 4: Prolongate coarse_buf into fine ghost zones (skip same-level dirs)

**INTENTIONAL DIFFERENCE.** AthenaK-style coarse-buffer approach vs Chombo FillPatch. Ordering is equivalent (same-level takes priority). More GPU-friendly.

### 2.2 Prolongation Order and Stencil

**GRChombo:** 4th-order (`FourthOrderFillPatch`, `FourthOrderFineInterp`).

**Lattice:** 6th-order 7-point Lagrange (`prolongation.h:33`, `PROLONG_STENCIL = 7`). Half-width = 3, requires ghost >= 3 on coarse grid (have 4).

**INTENTIONAL DIFFERENCE.** Matches FD order per ExaHyPE recommendation.

### 2.3 Restriction Method

**GRChombo:** 0th-order simple averaging (`CoarseAverage` — mean of 2x2x2 children).

**Lattice:** 6th-order cell-average restriction with 6-point 1D stencil (`restriction.h:39-40`). Stencil reach: 2 fine cells beyond direct children.

**INTENTIONAL DIFFERENCE.** Eliminates Hamiltonian violations at AMR boundaries.

### 2.4 Temporal Interpolation for Subcycling

**GRChombo:** Linear (1st-order) temporal interpolation between old/new coarse states. Alpha snapped to {0, 0.25, 0.5, 0.75, 1.0}.

**Lattice** (`block.c:296-358`): Quartic (4th-order, 5-point) temporal interpolation using fields_old, fields_older, rhs_old, rhs_older. Ramp: step 0 = copy, step 1 = linear, step 2+ = quartic.

**INTENTIONAL DIFFERENCE.** Significantly more accurate temporal interpolation at coarse-fine boundaries.

### 2.5 Subcycling Algorithm

**MATCH.** Both use standard Berger-Oliger 2:1 subcycling. Lattice implements recursion explicitly (`rk4.c:541-604`), GRChombo delegates to Chombo.

### 2.6 Ghost Exchange Per RK Stage

**GRChombo** (`GRAMRLevel.cpp:911-939`): Same-level AND cross-level ghost fill every RK stage (FillPatch in `evalRHS`).

**Lattice** (`rk4.c:476-515`): Same-level ghost exchange every RK stage. Cross-level fill only ONCE before packing (`rk4.c:447-448`).

**SUSPICIOUS DIFFERENCE.** During RK4 stages 2-4, fine-level blocks near coarse-fine boundaries have cross-level ghost data from the beginning of the step, not from the intermediate RK state. Introduces O(dt^2) error at coarse-fine boundaries. Likely acceptable (CFL ~ 0.25) but a genuine difference.

### 2.7 Cross-Level Ghost Fill Timing

**GRChombo:** Per-stage cross-level fill with time-interpolation alpha matching RK sub-step time.

**Lattice** (`rk4.c:443-448`): Single pre-step cross-level fill. Pack built from pre-filled blocks.

**SUSPICIOUS DIFFERENCE (same as 2.6).** RK stages 2-4 use stale cross-level ghost data. Error is O(dt * dx_coarse).

### 2.8 Algebraic Enforcement Timing

**MATCH.** Both enforce before each RHS and after full step. Lattice additionally enforces det(h)=1. GRChombo's post-sub-step tr(A)=0 is not replicated in Lattice but is effectively redundant.

### 2.9 Chi-Gradient Refinement Criterion

**MATCH.** Both use `(dx / chi^2) * |grad(chi)|`. Lattice uses 6th-order derivatives vs GRChombo's 4th-order.

### 2.10 Stale Ghost Data at Coarse-Fine Boundaries

**SUSPICIOUS DIFFERENCE.** Lattice fills cross-level ghosts once per fine step. GRChombo refreshes every RK stage with updated time-interpolation alpha. This is the most significant algorithmic difference in the AMR infrastructure. Since Lattice passes convergence tests at expected order, this is acceptable but may produce elevated constraint violations at refinement boundaries.

### 2.11 CRITICAL: Prolongation Stencil Bounds at coarse_buf Edge

The 7-point prolongation stencil needs `half = 3` cells on each side. Mapping from fine ghost index to coarse_buf index:

```
fi=0 → ci0=2 → stencil needs ci=-1 → OUT OF BOUNDS → SKIPPED
fi=1 → ci0=3 → stencil needs ci=0  → borderline valid → FILLED
fi=2 → ci0=4 → FILLED
fi=3 → ci0=5 → FILLED
```

**Protection** (`ghost_exchange.c:503-507`): `if (ci0 < half) continue;` — correctly prevents out-of-bounds access but **leaves the outermost fine ghost cell (fi=0) unfilled**.

**Impact analysis:**
- 6th-order FD stencils (FD_D1/FD_D2) access at most 3 ghost cells — they do NOT read fi=0.
- KO dissipation (8th-order, 9-point stencil) needs 4 ghost cells — it DOES read fi=0.
- If fi=0 was filled by same-level exchange (Phase 1), no problem.
- If fi=0 has no same-level neighbor AND no domain boundary Sommerfeld BC: the cell retains stale/zero data.

**GRChombo comparison:** Uses 4th-order FillPatch with half=2, which fits in ghost=3-4 without skipping.

**SUSPICIOUS DIFFERENCE — potential NaN source.** The unfilled outermost ghost cell is read by the KO dissipation stencil at the first interior cell near coarse-fine boundaries. If the stale data is extreme, the dissipation term injects noise. The `continue` guard prevents crash but not data corruption. Most dangerous for isolated fine blocks at the edge of the refined region.

### 2.12 AMR Section Summary

| Item | Status | Risk |
|---|---|---|
| Ghost exchange ordering | INTENTIONAL DIFFERENCE | None |
| Prolongation (6th-order) | INTENTIONAL DIFFERENCE | None |
| Restriction (6th-order) | INTENTIONAL DIFFERENCE | None |
| Temporal interpolation (quartic) | INTENTIONAL DIFFERENCE | None |
| Subcycling (Berger-Oliger) | MATCH | None |
| Cross-level ghost per RK stage | SUSPICIOUS | Low — O(dt^2) error |
| Cross-level ghost fill timing | SUSPICIOUS | Low — O(dt * dx) error |
| Algebraic enforcement timing | MATCH | None |
| Chi-gradient criterion | MATCH | None |
| Stale ghost data in subcycling | SUSPICIOUS | Low-Medium |
| **Prolongation stencil skip (fi=0)** | **SUSPICIOUS** | **Medium — KO reads unfilled data** |

---

## 3. Boundary Conditions, Initial Data, Algebraic Enforcement

### 3.1 Sommerfeld BC Formula

**GRChombo** (`BoundaryConditions.cpp:593-661`): `rhs(f) = -sum_i[d_i(f) * x^i/r] + (f_asymp - f)/r`. Uses **2nd-order** stencils at boundaries.

**Lattice** (`sommerfeld.c:57-118`): Same formula, **4th-order** adaptive stencils (Fornberg 1998).

**INTENTIONAL DIFFERENCE.** Same physics, higher accuracy.

### 3.2 Sommerfeld Asymptotic Values

| Field | Lattice | GRChombo |
|---|---|---|
| chi | 1.0 | 1.0 |
| h_11, h_22, h_33 | 1.0 | 1.0 |
| lapse | 1.0 | 1.0 |
| all others | 0.0 | 0.0 |

**MATCH.**

### 3.3 Fields Receiving Sommerfeld BCs

**MATCH.** Both apply to all evolved fields uniformly.

### 3.4 Parity Enforcement at Boundaries

**MISSING.** GRChombo has a full parity system (EVEN, ODD_X, ODD_Y, etc.) for reflective boundaries. Lattice does not implement reflective boundaries. **Not a concern** for full-domain simulations (all faces are Sommerfeld/CP-BC).

### 3.5 Constraint-Preserving BCs

**INTENTIONAL DIFFERENCE.** Lattice adds BAM-style CP-BCs (arXiv:1212.2901) as an option beyond GRChombo's repertoire.

### 3.6 Chi/Lapse Floor Value and Placement

**MATCH.** Both use 1e-4. Both apply before every RHS and after full step. Lattice adds defense-in-depth floor inside the RHS itself (`ccz4_rhs.c:127-128`).

### 3.7 det(h)=1 Enforcement

**INTENTIONAL DIFFERENCE.** Lattice explicitly rescales `h_ij *= det(h)^{-1/3}`. GRChombo does not explicitly enforce (relies on evolution equations). Lattice is more robust.

### 3.8 tr(A)=0 Enforcement

**MATCH.** Same `make_trace_free(A, h, h_UU)` operation.

### 3.9 Algebraic Enforcement Timing in RK4

**GRChombo (BinaryBH):**
- Before each RHS: TraceARemoval + PositiveChiAndAlpha (4x)
- After each sub-step update: TraceARemoval only (4x)
- After full step: TraceARemoval + PositiveChiAndAlpha (1x)
- Total: 9 enforcement calls per step

**Lattice:**
- Before each RHS: det(h)=1 + tr(A)=0 + chi/lapse floor (4-5x)
- After full step: same (1x)
- After prolongation in AMR regrid
- Total: 5-6 enforcement calls per step

**INTENTIONAL DIFFERENCE.** Lattice omits post-sub-step tr(A)=0 (redundant since enforcement runs before next RHS). No stability risk.

### 3.10 Brill-Lindquist Initial Data

**MATCH.** `chi = psi^{-4}`, `h_ij = delta_ij`, `K = 0`, `A_bar = psi^{-6} * A_tilde`, `lapse = sqrt(chi)`.

### 3.11 Bowen-York A_ij

**INTENTIONAL DIFFERENCE.** Lattice includes spin contributions (`bowen_york.c:34-84`). GRChombo's BoostedBH is momentum-only. Puncture regularization: 1e-10 (Lattice) vs 1e-6 (GRChombo) — both inside the AH, inconsequential.

### 3.12 Post-Prolongation Algebraic Enforcement

**INTENTIONAL DIFFERENCE (improvement).** Lattice enforces det(h)=1, tr(A)=0, chi/lapse floor immediately after prolongation (`refine.c:186-187`). GRChombo does not. 6th-order Lagrange negative weights can push chi below zero — immediate enforcement prevents artifacts.

### 3.13 NaN Checking

**GRChombo** (`NanCheck.hpp`): Production-mode check every step. Checks all evolved variables for NaN and values > 1e20. Prints detailed diagnostics on detection.

**Lattice:** GPU has `backend_check_finite_packed()`. CPU has no equivalent. Debug builds use `-fsanitize=address,undefined`.

**SUSPICIOUS DIFFERENCE.** No automatic CPU NaN detection in production. NaN propagates silently until output is obviously wrong. The chi/lapse floor provides significant protection, but NaN from other sources (bad ghost data, overflow) goes undetected.

### 3.14 Boundary/ID Section Summary

| Item | Status |
|---|---|
| Sommerfeld formula | INTENTIONAL DIFFERENCE (4th-order stencils) |
| Asymptotic values | MATCH |
| Fields receiving BCs | MATCH |
| Parity enforcement | MISSING (not needed) |
| CP-BCs | INTENTIONAL DIFFERENCE (added feature) |
| Chi/lapse floor | MATCH (1e-4, same placement) |
| det(h)=1 enforcement | INTENTIONAL DIFFERENCE (explicit) |
| tr(A)=0 enforcement | MATCH |
| Enforcement timing | INTENTIONAL DIFFERENCE (fewer but equivalent) |
| Brill-Lindquist ID | MATCH |
| Post-prolongation enforcement | INTENTIONAL DIFFERENCE (improvement) |
| NaN checking | SUSPICIOUS (no CPU production check) |

---

## 4. Consolidated Findings

### All Suspicious Differences (ranked by NaN risk)

| # | Item | Risk | Could Cause QNM NaN? |
|---|---|---|---|
| S1 | Prolongation stencil skip at fi=0 (§2.11) | **Medium** | **Yes** — KO dissipation reads unfilled ghost data at coarse-fine boundaries |
| S2 | Cross-level ghosts filled once per step, not per RK stage (§2.6/2.7) | Low-Medium | Unlikely — O(dt^2) error, passes convergence tests |
| S3 | Stale ghost data at coarse-fine during subcycling (§2.10) | Low-Medium | Unlikely — same mechanism as S2 |
| S4 | No CPU NaN detection in production (§3.13) | Low | No (effect, not cause) — but masks the true source |
| S5 | Hamiltonian constraint uses CCZ4 Ricci with Z terms (§1.8) | None | No — monitoring only, not evolution |
| S6 | Psi4 Ricci Z coefficient differs (§1.9) | None | No — diagnostics only |

### Recommended Actions

1. **S1 — Investigate prolongation stencil skip.** Determine if the QNM test has isolated fine blocks where fi=0 ghost cells at coarse-fine boundaries lack same-level neighbors. If so, the KO stencil reads stale/zero data. Options:
   - Widen coarse_buf by 1 ghost cell (GHOST_WIDTH+1 on the coarse buffer)
   - Fall back to lower-order prolongation for outermost ghost cells
   - Skip KO dissipation at the first interior cell near coarse-fine boundaries

2. **S4 — Add periodic CPU NaN check.** Even a check every 100 steps would catch NaN propagation early and identify the source cell/field.

3. **S2/S3 — Monitor but don't fix yet.** Cross-level ghost staleness is a known approximation. If constraint violations at AMR boundaries are elevated after fixing S1, revisit per-stage cross-level fill.

### Verification Protocol

After fixing S1:
1. `make` — zero warnings
2. `make test` — all pass
3. `make test-qnm` — verify no NaN, check constraint convergence
4. If QNM still fails, add the CPU NaN check (S4) to identify the failing field/location
