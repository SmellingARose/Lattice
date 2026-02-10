# Lattice — Session Handoff (2026-02-10)

> **Purpose:** This file is context for Claude Code to resume work on a new machine.
> Read this file, then CLAUDE.md, then the source files referenced below.

## Current State

**Branch:** main
**Tier:** 2 (single Schwarzschild puncture)
**Last commit:** `c593cc5` — all bugs fixed, smoke test passing

### Test Results

| Test | Status | Command | Runtime |
|------|--------|---------|---------|
| Flat spacetime | PASS | `make test-flat` | ~3 min |
| Flat convergence (4th-order) | PASS | `make test-convergence` | ~10 min |
| BH smoke (41^3, 20 steps) | PASS | `make test-bh-smoke` | ~5 sec |
| **Full BH (137^3, 50M)** | **NOT RUN** | `make test-bh` | ~30-60 min |

The full BH test is the next milestone. It needs server compute.

### Smoke Test Output (last known good)
```
alpha_min: 0.40 -> 0.35 (correct trumpet lapse dynamics, heading toward ~0.3)
ham_l2: bounded at ~3e-2 (no blowup)
chi_min: evolving correctly
No NaN, no crash
```

## What To Do Next

### Immediate (on server)
1. `git clone` and `make`
2. Run `make test-bh-smoke` to verify build works
3. Run `make test-bh` — this is the real test: 137^3 grid, evolve to t=50M
   - **Pass criteria:** trumpet lapse alpha_min ~ 0.3, constraints bounded (ham_l2 < 1.0), no NaN
4. If it passes, update CLAUDE.md tier status to "Tier 2 complete"

### If full BH test fails
Likely failure modes and what to investigate:
- **Blowup after ~10-20M**: Probably boundary condition issue (Sommerfeld). Check `src/boundary/sommerfeld.c`. Domain=128M means boundary at 64M — outgoing waves reach it by t~64M.
- **Slow constraint growth**: Check KO dissipation strength (`eps_ko` in params). May need tuning. Also check if Z*dchi cross terms in Ricci matter (see "Remaining equation work" below).
- **Lapse doesn't settle to ~0.3**: Gauge parameter issue. Check eta value (currently 2.0) and 1+log condition in `src/evolution/gauge_rhs.c`.
- **NaN at puncture**: Chi floor too small or too large. Currently 1e-4, matches GRChombo.

### After full BH passes
1. BH convergence test: run at 3 resolutions (65^3, 97^3, 129^3) and confirm 4th-order convergence
2. Move to Tier 3: head-on binary collision
3. Consider removing debug fprintf in ccz4_rhs.c (lines 749-786) for production runs

## Bugs Fixed This Session

Seven bugs were found and fixed by a team of 4 agents (debugger, researcher, test-runner, coordinator) cross-referencing against GRChombo source, B&S textbook, and arXiv:1106.2254.

### Critical equation bugs
1. **SYM macro** (`src/core/fields.h:21`): `i*(3-i)/2+j` → `i*(5-i)/2+j`. Old formula gave wrong flat indices for yy/yz/zz symmetric tensor components. Corrupted ALL tensor operations.

2. **Upwind advection stencils** (`src/numerics/finite_diff.h:73-85`): FD_ADV_LEFT/RIGHT coefficients were negated — all advection terms had wrong sign.

3. **dt_chi** (`src/evolution/ccz4_rhs.c:475`): Two issues:
   - Signs flipped on K and div_beta terms (chi evolved backwards)
   - Incorrectly included Theta coupling `(K - 2*Theta)` → should be just `K` (per arXiv:1106.2254 eq 13; Theta coupling is only in lapse/At/K equations, not chi)

4. **Rchi_{ij}** (`src/evolution/ccz4_rhs.c:352`): Prefactors swapped — `d_i chi d_j chi` had factor 3 (should be 1), `gt_{ij}*dchi_sq` had factor 1 (should be 3). Ref: B&S eq 3.10 in chi convention.

5. **phys_DDalpha** (`src/evolution/ccz4_rhs.c:554-557`): Connection correction signs wrong for conformal→physical Hessian of lapse. Ref: B&S eq 3.30.

6. **Momentum constraint diagnostic** (`src/diagnostics/constraints.c`): Wrong coefficient. Doesn't affect evolution, only monitoring accuracy.

### Root cause of BH blowup
7. **Puncture at grid point**: r=0 sitting on a grid point created discontinuity from chi floor. FD stencils amplified it. Fixed by offsetting puncture by dx/2. Standard practice (GRChombo uses cell-centered grids).

### Stability improvements added
- Chi/alpha floors (1e-4) in `enforce_algebraic_constraints` and RK4 substeps (matches GRChombo PositiveChiAndAlpha.hpp)
- NaN/blowup detector in RK4 (uses integer bit-pattern check, immune to -ffast-math)
- KO dissipation W floor at punctures (W = max(sqrt(chi), 0.1))
- Debug tracing for RHS blowup location (ccz4_rhs.c lines 749-786)

## Remaining Equation Work (lower priority)

These were identified but deferred since they don't break convergence order:

1. **Z*dchi cross terms in conformal Ricci**: GRChombo adds `z_over_chi_terms` to Rchi_{ij}. Low impact since Z_i starts at 0 for Brill-Lindquist puncture data and stays small with constraint damping.

2. **DZ_sym approximation** (`ccz4_rhs.c:567-596`): The D_i Z_j + D_j Z_i computation uses `d_i Ghat^k` but drops `d_i Gamma^k` terms (which need d2_gt). Preserves convergence order but reduces accuracy.

## Verified-Correct Equations

All major evolution equations have been cross-referenced against arXiv:1106.2254 and GRChombo:
- dt_chi, dt_gt, dt_At, dt_K, dt_Theta, dt_Ghat — all verified
- lap_alpha, phys_DDalpha, Rchi_dd, Rt_dd — all verified
- 1+log lapse, Gamma-driver shift — verified
- Hamiltonian and momentum constraints — verified

## Key Source Files

| File | What it does |
|------|-------------|
| `src/evolution/ccz4_rhs.c` | Main CCZ4 RHS — the core physics (~800 lines) |
| `src/evolution/gauge_rhs.c` | 1+log lapse + Gamma-driver shift |
| `src/evolution/dissipation.c` | Kreiss-Oliger dissipation (6th-order) |
| `src/numerics/rk4.c` | RK4 integrator + algebraic constraint enforcement |
| `src/numerics/finite_diff.h` | FD stencil macros (D1, D2, ADV, KO) |
| `src/core/fields.h` | Field enum + SYM macro |
| `src/diagnostics/constraints.c` | Hamiltonian + momentum constraint monitor |
| `src/initial_data/puncture.c` | Brill-Lindquist puncture initial data |
| `src/boundary/sommerfeld.c` | Sommerfeld outgoing-wave boundary conditions |
| `tests/test_bh_smoke.c` | Quick BH sanity test (use for iteration) |
| `tests/test_single_bh.c` | Full BH validation test (use on server) |

## Build & Test

```bash
make                    # optimized build (-O3 -ffast-math -march=native + OpenMP)
make test-bh-smoke      # quick BH test (~5 sec) — USE THIS FOR ITERATION
make test-bh            # full BH test (30-60 min on server)
make test-flat          # flat spacetime test (~3 min)
make test-convergence   # 3-resolution convergence check (~10 min)
make clean              # remove build artifacts
```

On Linux servers, OpenMP works out of the box with GCC. No external dependencies.

## References

- GRChombo: CCZ4RHS.impl.hpp, CCZ4Geometry.hpp, PositiveChiAndAlpha.hpp
- B&S: Baumgarte & Shapiro, *Numerical Relativity* — equations 3.10, 3.30, 3.57, 3.69, 11.13
- arXiv:1106.2254 — original CCZ4 paper (complete evolution equations)
- arXiv:2404.01137 — improved KO dissipation (CAKO method)
- arXiv:2501.01055 — CCZ4 variants and long-term stability
- https://20k.github.io/ — NR implementation notes

## Team Setup (for Claude Code)

To spin up a debugging/testing team:
```
Create team "convergence-debug" with 4 agents:
- test-runner: builds and runs tests, reports results (use make test-bh-smoke for iteration)
- debugger: reads source code, cross-references equations against papers/GRChombo
- researcher: searches web for reference implementations and paper equations
- coordinator: routes info between agents, tracks progress, enforces process

IMPORTANT RULES FOR THE TEAM:
- Use ONLY `make test-bh-smoke` for iteration testing (5 sec)
- Only run `make test-bh` for final validation (30+ min)
- Every fix must be cross-verified by debugger AND researcher before applying
- Coordinator sends regular status updates to team-lead
```
