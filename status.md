# Lattice — Current Status

**Date:** 2026-02-10
**Branch:** main
**Tier:** 2 (single Schwarzschild puncture)

## What Works

- **Flat spacetime test**: PASS — constraint violation < 1e-10 after 1000 steps
- **Convergence test (flat)**: PASS — 4th-order convergence confirmed
- **BH smoke test** (`make test-bh-smoke`): PASS — 41^3 grid, 20 steps, ~5 seconds
  - alpha_min: 0.40 -> 0.35 (correct trumpet lapse dynamics, heading toward ~0.3)
  - ham_l2 bounded at ~3e-2 (no blowup)
  - chi_min evolving correctly
  - No NaN, no crash

## What Needs Testing (requires compute power)

- **Full BH test** (`make test-bh`): 137^3 grid, 50M evolution
  - Expected runtime: 30-60 min on powerful server
  - Pass criteria: trumpet lapse ~0.3, constraints bounded, no NaN
  - Too slow for M4 laptop (~60+ min)

- **BH convergence test**: Not yet implemented for BH data (only flat)
  - Need 3 resolutions (e.g. 33^3, 65^3, 129^3) to confirm 4th-order

## Bugs Fixed This Session

### Critical (equation bugs)
1. **SYM macro** (`src/core/fields.h:21`): Formula `i*(3-i)/2+j` gave wrong indices for yy/yz/zz tensor components. Fixed to `i*(5-i)/2+j`. Corrupted all symmetric tensor operations.

2. **Upwind advection stencils** (`src/numerics/finite_diff.h:73-85`): FD_ADV_LEFT/RIGHT had negated coefficients, reversing all advection terms.

3. **dt_chi signs** (`src/evolution/ccz4_rhs.c`): Conformal factor RHS had flipped signs on K and div_beta terms. Also incorrectly included Theta coupling (should be just K, not K-2Theta per arXiv:1106.2254).

4. **Rchi_{ij} prefactors** (`src/evolution/ccz4_rhs.c`): The d_i chi d_j chi term had factor 3 instead of 1, and gt_{ij}*dchi_sq had factor 1 instead of 3 (swapped).

5. **phys_DDalpha signs** (`src/evolution/ccz4_rhs.c`): Connection correction for conformal-to-physical Hessian of lapse had wrong signs (B&S eq 3.30).

6. **Momentum constraint** (`src/diagnostics/constraints.c`): Wrong coefficient in diagnostic (doesn't affect evolution).

### Root cause of BH blowup
7. **Puncture at grid point**: Placing r=0 exactly on a grid point created a discontinuity that FD stencils amplified. Fixed by offsetting puncture by dx/2 (standard practice — GRChombo uses cell-centered grids).

### Stability improvements
- Chi/alpha floors (1e-4) in `enforce_algebraic_constraints` and RK4 substeps (matches GRChombo PositiveChiAndAlpha)
- NaN/blowup detector in RK4 (immune to -ffast-math)
- Spatially-varying KO dissipation with W floor at punctures
- Debug tracing for RHS blowup detection

## Build & Test Commands

```bash
make                    # optimized build
make test-bh-smoke      # quick BH test (~5 sec) — USE THIS FOR ITERATION
make test-bh            # full BH test (30-60 min on server)
make test-flat           # flat spacetime test
make test-convergence   # convergence verification (flat)
```

## Remaining Work (lower priority)

1. Missing Z*dchi cross terms in conformal Ricci (GRChombo adds z_terms). Low impact since Z starts at 0 for Brill-Lindquist data.
2. DZ_sym approximation in At equation — missing d_i Gamma^k terms. Preserves convergence order but reduces accuracy.
3. BH convergence test at 3 resolutions.
4. Long-term stability test (t > 100M).

## References Used for Verification

- GRChombo CCZ4RHS.impl.hpp, CCZ4Geometry.hpp, PositiveChiAndAlpha.hpp
- B&S (Baumgarte & Shapiro) equations 3.10, 3.30, 3.57, 3.69, 11.13
- arXiv:1106.2254 (original CCZ4 paper)
- arXiv:2404.01137 (improved KO dissipation)
- https://20k.github.io/ (NR implementation notes)
