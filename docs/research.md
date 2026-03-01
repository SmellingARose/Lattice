# Lattice — Remaining Implementation Research

Compiled research for all remaining Phase 3 features. Each section covers the
math, simplest implementation path, estimated effort, and key references.

---

## Executive Summary

| Item | Effort | Lines | Difficulty | Blocks | Competitive Edge |
|------|--------|-------|------------|--------|-----------------|
| **Psi4 extraction** | 3-5 days | ~580 | Medium | PHE, CCE | Table stakes — every NR code has this |
| **PHE** | 1-2 weeks | ~470 | Medium | — | **No public standalone PHE exists** — first-of-kind tool |
| **CCE worldtube** | 2-3 weeks | ~930 | Medium | Needs HDF5 | Matches GR-Athena++, SpECTRE interop |
| **HIP backend** | 1-2 weeks | ~900 | Medium | — | **AMD + NVIDIA from one source** — only AthenaK does this (via Kokkos) |
| **Constraint-preserving BC** | 2-3 weeks | ~500 | Hard | — | Matches BAM; most CCZ4 codes lack this |
| **Larger domain (AMR)** | 0 days | 0 | Trivial | — | Already possible, just use `--L 4000` |

**Fast path to publishable waveforms:** Psi4 (5 days) + PHE (2 weeks) = ~3 weeks.
No external dependencies, pure C17.

### How Lattice Compares After Implementation

| Capability | Lattice (current) | Lattice (after) | GRChombo | SpECTRE | GR-Athena++ | ET |
|------------|-------------------|-----------------|----------|---------|-------------|-----|
| Spatial FD order | 6th | 6th | 4th | Spectral | 6th | 4th-8th |
| Off-grid interpolation | 6th | 6th | 4th | Spectral | 4th | 4th |
| Waveform extraction | None | Psi4 + PHE + CCE | Psi4 only | Full CCE | CCE | Psi4 + CCE |
| GPU backend | OpenMP target | OpenMP + HIP | None | Charm++ | Kokkos | None |
| AMD GPU support | No | **Yes (HIP)** | No | No | Yes (Kokkos) | No |
| N-body (N>2) | Yes (32) | Yes (32) | No | No | No | Yes |
| Einstein-Maxwell | Yes | Yes | No | No | No | Yes |
| High-spin initial data | Yes (HiSpID) | Yes | Yes | Yes | No | Yes |
| Boundary conditions | Sommerfeld | CP-Sommerfeld | Sommerfeld | CP + Bjorhus | Sommerfeld | Sommerfeld |
| AMR | Yes (block) | Yes | Yes (Chombo) | Yes (AMR) | Yes (block) | Yes (Carpet) |
| Formulation | CCZ4 | CCZ4 | CCZ4 | Gen. Harmonic | Z4c | BSSN |

**Unique advantages after completion:**
- **Only code with N-body + EM + GPU + CCE pipeline** (no other code combines all four)
- **PHE tool** would be first publicly available standalone implementation
- **HIP backend** gives AMD GPU support — only AthenaK has this, via heavier Kokkos abstraction
- **6th-order interpolation everywhere** — most codes use 4th-order off-grid

---

## Table of Contents

1. [Psi4 Extraction](#1-psi4-extraction)
2. [PHE (Perturbative Hyperboloidal Extraction)](#2-phe)
3. [CCE Worldtube Output](#3-cce-worldtube-output)
4. [HIP Backend](#4-hip-backend)
5. [HOBC (Higher-Order Absorbing BCs)](#5-hobc)
6. [Priority & Dependencies](#6-priority--dependencies)

---

## 1. Psi4 Extraction

**Status:** Not implemented. Blocks PHE and CCE.
**Effort:** 3-5 days, ~580 lines across `psi4.h` + `psi4.c`.
**Dependencies:** None — reuses existing infrastructure heavily.

### 1.1 Core Math

Psi4 encodes outgoing gravitational radiation via the electric and magnetic
parts of the Weyl tensor projected onto a null tetrad.

**Electric part E_ij** (B&S Eq. 3.28, GRChombo `Weyl4.impl.hpp`):

```
E_ij = R_ij + (K - Theta) * K_ij - K_ik * K^k_j
```

- `R_ij` = spatial Ricci with CCZ4 Z terms (`dZ_coeff=1`)
- `K_ij = (A_ij + (K/3) h_ij) / chi` (physical extrinsic curvature)
- Index raising uses physical inverse metric `gamma^{ij} = chi * h^{ij}`
- Make trace-free after: `E_ij -= (1/3) gamma_ij * tr(E)`

**Magnetic part B_ij** (B&S Eq. 3.29):

```
B_ij = epsilon^{ikl} nabla_k K_{lj}
```

- `epsilon^{ikl} = levi_civita[i][k][l] * alpha / chi^{3/2}`
- `nabla_k K_{ij}` uses physical Christoffels (already in `ah_finder.c:196-203`)
- Symmetrize: `B_ij = (B_ij + B_ji) / 2`

**Covariant derivative of K_ij** (chain rule from CCZ4 variables):

```
d_k K_{ij} = d_k(A_ij)/chi - d_k(chi)/chi * K_ij
           + (1/3) d_k(h_ij) * K / chi + (1/3) h_ij * d_k(K) / chi
```

**Newman-Penrose tetrad** — Gram-Schmidt orthonormalization:

1. `u^i = (x - x_center, y - y_center, z - z_center)` (radial)
2. `v^i = (-y + y_center, x - x_center, 0)` (azimuthal)
3. `w^i = epsilon^{ijk} v_j u_k / sqrt(det(gamma))` (cross product)
4. Orthonormalize against physical metric `gamma_{ij} = h_{ij}/chi`

**Psi4 contraction** (gr-qc/0104063):

```
Re(Psi4) = (1/2) * [E_ij (w^i w^j - v^i v^j) - 2 B_ij w^i v^j]
Im(Psi4) = (1/2) * [B_ij (-w^i w^j + v^i v^j) - 2 E_ij w^i v^j]
```

### 1.2 Simplest Implementation Path

**Strategy: compute on grid, interpolate scalars.** This is what GRChombo does.

1. `psi4_at_point()` — E_ij + B_ij + tetrad + contraction at one grid point.
   Uses existing `fd_d1`, `fd_d2`, `fd_d2_mixed` for derivatives, same Ricci
   pattern as `constraints.c`, physical Christoffels from `ah_finder.c`. ~200 lines.

2. `psi4_compute_grid()` — loop over grid, store Re/Im in two scratch arrays. ~30 lines.

3. `psi4_extract_spheres()` — interpolate Re/Im onto extraction spheres at
   r=50,75,100M using existing `interp_field_at()`. Follow `ah_finder.c`
   pattern for AMR. ~80 lines.

4. `psi4_decompose_modes()` — integrate Psi4 * conj(_{-2}Y_{lm}). Start with
   l_max=4 (25 modes). Hardcode l=2 harmonics, general Goldberg formula for
   l >= 3. ~130 lines.

**No new FD code. No changes to existing files.** Two scratch arrays (same size
as one field each, ~40 MB at N=128).

### 1.3 Spin-Weighted Spherical Harmonics (s = -2)

Dominant l=2 modes — hardcode closed forms:

```
_{-2}Y_{20} = sqrt(15/(32*pi)) * sin^2(theta)
_{-2}Y_{22} = sqrt(5/(64*pi)) * cos^4(theta/2) * e^{2i*phi}
_{-2}Y_{21} = sqrt(5/(16*pi)) * cos^3(theta/2) * sin(theta/2) * e^{i*phi}
```

Typical l_max: 4 for equal-mass BBH, 8 for high-spin/precessing.

### 1.4 Reusable Code

| Existing code | What it provides | Location |
|---------------|-----------------|----------|
| `compute_inverse_sym()` | 3x3 inverse | `tensor_utils.h:52` |
| `compute_christoffel()` | Conformal Christoffels | `tensor_utils.h:123` |
| `make_trace_free()` | Remove trace | `tensor_utils.h:169` |
| `fd_d1/d2/d2_mixed()` | 6th-order derivatives | `finite_diff.h` |
| `interp_field_at()` | 6th-order interpolation | `interpolate.h:170` |
| Physical Christoffel pattern | Conformal→physical | `ah_finder.c:196-203` |
| Block cache + lookup | AMR sphere sampling | `ah_finder.c:366-378` |
| `levi_civita[3][3][3]` | Levi-Civita symbol | `maxwell_rhs.c:47` |

### 1.5 New Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/diagnostics/psi4.h` | ~80 | API declarations, extraction sphere struct |
| `src/diagnostics/psi4.c` | ~500 | Core physics + extraction + mode decomposition |

### 1.6 Key References

- B&S Eqs. 3.28-3.29 (E_ij, B_ij definitions)
- GRChombo `Source/CCZ4/Weyl4.impl.hpp` (full implementation to cross-check)
- gr-qc/0104063 (NP tetrad construction for Psi4)

---

## 2. PHE

**Status:** Not implemented. **No public standalone PHE tool exists** — this
would be a genuine first-of-kind contribution.
**Effort:** 1-2 weeks, ~470 lines in `tools/phe_extract.c`.
**Dependencies:** Requires Psi4 extraction (Section 1).

### 2.1 What PHE Does

Post-processing tool that takes Psi4 multipoles extracted at finite radius and
propagates them to null infinity via the Regge-Wheeler-Zerilli (RWZ) equations
on hyperboloidal slices. Removes finite-radius extraction error without
modifying the evolution code.

**Published accuracy (arXiv:2508.05743):** PHE vs CCE phase difference < 0.05
rad for BBH mergers. Sufficient for LIGO A+, ET, and CE detectors.

### 2.2 The RWZ Master Equation

Perturbations of Schwarzschild satisfy a 1+1 wave equation per (l,m) mode:

```
d²Psi/dt² - d²Psi/dr*² + V_l(r) * Psi = 0
```

**Tortoise coordinate:** `r* = r + 2M ln(r/(2M) - 1)`

**Potentials:**

```
V_RW(r) = (1 - 2M/r) * [l(l+1)/r² - 6M/r³]               (odd parity)
V_Z(r)  = (1 - 2M/r) * [2λ²(λ+1)r³ + 6λ²Mr² + 18λM²r + 18M³]
                       / [r³(λr + 3M)²]                     (even parity)
```

where `λ = (l-1)(l+2)/2`.

### 2.3 Hyperboloidal Compactification

Transform (t, r*) → (tau, rho) via height function:

```
tau = t - h(r*)           (time transformation)
r*  = rho / Omega(rho)    (spatial compactification)
```

**Domain:**

```
rho_min = r*(r_ext)        inner boundary (extraction radius)
rho_*   = 1.5 * rho_min    start of hyperboloidal layer
rho_S   = 2.0 * rho_min    null infinity (Omega = 0)
```

At null infinity the equation becomes purely ingoing — **no outer boundary
condition needed**. This is the key advantage.

**Inner boundary:** NR data provides ingoing characteristic variable
`g(t) = Psi_t - Psi_{r*}` at r_ext, smoothed by activation function
ramping 0→1 over ~100M.

### 2.4 Implementation

Standalone post-processing — zero coupling to evolution code:

```
tools/phe_extract.c
  Input:  psi4_lm_rXXX.csv (Psi4 multipole time series)
  Output: strain_lm_scri.csv (h at null infinity)
  Build:  make phe-extract (pure C17 + libm, no Lattice dependency)
```

Steps per (l,m) mode:
1. Set up 1D radial grid (N = 401-1601 points)
2. Compute compactification functions Omega(rho), H(rho), V_l(rho)
3. Read NR time series, construct inner BC with activation
4. Time-step hyperboloidal RWZ: RK4 in time, 4th-order FD in space
5. Record waveform at rho_S (null infinity)

### 2.5 Resolution & Accuracy

| N (radial) | Phase accuracy | Use case |
|------------|---------------|----------|
| 401 | ~0.01 rad | Development |
| 801 | ~10⁻³ rad | Production |
| 1601 | ~10⁻⁴ rad | High accuracy |

CFL is modest: a 5000M run at N=401 takes ~25,000 steps — seconds on a laptop.

### 2.6 Line Count

| Component | Lines |
|-----------|-------|
| Grid setup + compactification | ~120 |
| RWZ RHS (4th-order FD + potential) | ~80 |
| RK4 time stepper | ~60 |
| Inner BC (characteristic + activation) | ~50 |
| I/O (CSV read/write) | ~100 |
| Main + CLI | ~60 |
| **Total** | **~470** |

### 2.7 Key References

- arXiv:2508.05743 (Bernuzzi et al. 2025) — the PHE paper
- arXiv:1008.3809 (Zenginoglu 2011) — hyperboloidal layers
- arXiv:1107.5402 (Bernuzzi, Nagar, Zenginoglu 2012) — BBH with hyperboloidal
- gr-qc/0502028 (Martel & Poisson 2005) — gauge-invariant RWZ formalism

---

## 3. CCE Worldtube Output

**Status:** Not implemented.
**Effort:** 2-3 weeks, ~930 lines in `cce_worldtube.h` + `cce_worldtube.c`.
**Dependencies:** Requires Psi4 infrastructure (shares interpolation). Requires
HDF5 library (conditional: `make HDF5=on`).

### 3.1 Overview

Output worldtube boundary data on a coordinate sphere for SpECTRE's CCE solver.
SpECTRE does the characteristic evolution to null infinity — gold standard for
waveform accuracy.

**Workflow:**
1. Lattice outputs **ADM Metric Nodal** format (simplest from CCZ4 variables)
2. Run SpECTRE `PreprocessCceWorldtube` to convert to Bondi-Sachs modal
3. Run SpECTRE CCE on the converted file

### 3.2 Required Data (31 HDF5 Datasets)

| Category | Datasets | Count |
|----------|----------|-------|
| Spatial metric | `gxx.dat` ... `gzz.dat` | 6 |
| Extrinsic curvature | `Kxx.dat` ... `Kzz.dat` | 6 |
| Lapse + derivatives | `Lapse.dat`, `D{x,y,z}Lapse.dat` | 4 |
| Shift + derivatives | `Shift{x,y,z}.dat`, `D{x,y,z}Shift{x,y,z}.dat` | 12 |
| Gauge auxiliary | `AuxiliaryShift{x,y,z}.dat` (= B^i) | 3 |

### 3.3 Angular Grid

- theta: `l_max + 1` Gauss-Legendre points in cos(theta)
- phi: `2 * l_max + 1` equally spaced
- Default l_max = 16 → 17 × 33 = 561 points per timestep per dataset
- Row format: `[time, val(phi_0,theta_0), val(phi_0,theta_1), ...]`
- `Legend` attribute required on every dataset
- File naming: `CceR{XXXX}.h5` (zero-padded radius)

### 3.4 Conformal-to-Physical Reconstruction

```c
gamma_ij = h_ij / chi                                    // physical metric
K_ij     = (A_ij + (K/3) * h_ij) / chi                  // physical ext. curvature
d_k gamma_ij = d_k(h_ij)/chi - h_ij * d_k(chi) / chi²  // chain rule
```

Cartesian derivatives via `interp_field_deriv_at()` — no FD on the sphere.

### 3.5 PreprocessCceWorldtube Config

```yaml
InputDataFormat:
  AdmMetricNodal:
    Lapse: { Advective: True }         # 1+log slicing
    Shift: { Advective: True }         # Gamma-driver
    FirstOrderDriverFactor: 0.75       # shift_Gamma_coeff
    SecondOrderDriverEta: 1.0          # eta
ExtractionRadius: 100
LMaxFactor: 3
```

### 3.6 HDF5 Dependency

Required — SpECTRE only reads HDF5. Gated behind `make HDF5=on`:
```makefile
ifeq ($(HDF5),on)
    HDF5_FLAGS = -DLATTICE_HDF5 $(shell pkg-config --cflags hdf5)
    HDF5_LIBS  = $(shell pkg-config --libs hdf5)
endif
```

Default `HDF5=off` — zero impact on normal builds.

**Alternative:** Python postprocessor (`tools/write_cce_worldtube.py`, ~150
lines) reads raw binary and writes HDF5 via h5py. Avoids C dependency.

### 3.7 CLI Flags

```
--cce-radius 100     # extraction radius (M)
--cce-every 20       # output cadence in steps
--cce-lmax 16        # angular resolution
```

### 3.8 Line Count

| Component | Lines |
|-----------|-------|
| Header | ~60 |
| Gauss-Legendre nodes/weights | ~80 |
| Sphere coordinate setup | ~40 |
| HDF5 file creation + Legend attrs | ~200 |
| Per-step interpolation + reconstruction | ~250 |
| Per-step HDF5 write | ~120 |
| Cartesian derivatives (chain rule) | ~150 |
| Cleanup | ~30 |
| **Total** | **~930** |

### 3.9 Key References

- arXiv:2110.08635 (Moxon et al. 2023) — SpECTRE CCE system
- spectre-code.org/tutorial_cce.html — exact HDF5 format spec
- arXiv:2411.11989 (GR-Athena++ waveforms) — CCE from non-SpECTRE code

---

## 4. HIP Backend

**Status:** Not implemented.
**Effort:** 1-2 weeks, ~900 lines in `backend_hip.cpp` + Makefile.
**Dependencies:** None (independent of other work).

### 4.1 What HIP Is

AMD's GPU programming model — near-identical CUDA syntax, compiles for both:

- `HIP_PLATFORM=amd` → AMD GPUs (MI250X, MI300X) via ROCm
- `HIP_PLATFORM=nvidia` → NVIDIA GPUs (V100, A100, H100) via nvcc

20-50% faster than OpenMP target offloading. Avoids GCC nvptx codegen bugs.

### 4.2 Translation Pattern

| OpenMP Target | HIP |
|---------------|-----|
| `#pragma omp target teams distribute parallel for` | `__global__` kernel + `<<<grid,block>>>` |
| `#pragma omp target enter data map(to:)` | `hipMalloc` + `hipMemcpy` |
| `#pragma omp target exit data map(from:)` | `hipMemcpy` + `hipFree` |
| `#pragma omp declare target` on function | `__device__` attribute |
| `#pragma omp declare target` on constant | `__constant__` memory |

### 4.3 Architecture

**Single new file:** `src/backend/backend_hip.cpp` (.cpp required for hipcc).

**Compile everything with hipcc.** All physics code is valid C++ — hipcc handles
it. Required because `ccz4_rhs_point` and everything it calls must be
`__device__`-annotated.

**Device macro:** Add `LATTICE_DEVICE` to existing headers:

```c
#ifdef LATTICE_HIP
  #define LATTICE_DEVICE __device__
#else
  #define LATTICE_DEVICE
#endif
```

### 4.4 Required Kernels (~17)

| Function | Kernels | Lines | Notes |
|----------|---------|-------|-------|
| `backend_map/unmap_pack` | 0 | ~110 | hipMalloc/hipMemcpy for ~15 buffers |
| `backend_zero/copy/axpy/accum/apply` | 5 | ~75 | Trivial flat loops |
| `backend_compute_rhs_packed` | 1 | ~50 | Calls ccz4_rhs_point |
| `backend_sommerfeld_packed` | 1 | ~60 | Face iteration |
| `backend_update_ck45_packed` | 1 | ~15 | Flat loop |
| `backend_ghost_exchange_packed` | 7 | ~300 | 7 phases (same as GPU backend) |
| `backend_enforce_algebraic_packed` | 1 | ~60 | det/inverse/trace-free |
| `backend_init/cleanup` | 0 | ~20 | hipMemcpyToSymbol |

### 4.5 Performance Notes

- **No shared memory for RHS.** 25+ fields × 7-point stencils = ~690 KB needed,
  far exceeding 48-64 KB limits. Global memory + cache is correct (same as
  AthenaK).
- **Block size:** 128-256 for RHS (5.3 KB stack/thread), 256-512 for simple
  kernels.

### 4.6 Makefile Addition

```makefile
else ifeq ($(BACKEND),hip)
    HIPCC ?= hipcc
    HIP_PLATFORM ?= $(shell hipconfig --platform 2>/dev/null || echo amd)
    ifeq ($(HIP_PLATFORM),nvidia)
        GPU_ARCH_HIP ?= sm_80
    else
        GPU_ARCH_HIP ?= gfx90a
    endif
    CC = $(HIPCC)
    CFLAGS_BASE = -std=c++17 -DLATTICE_HIP --offload-arch=$(GPU_ARCH_HIP) ...
endif
```

### 4.7 Implementation Order

1. Add `LATTICE_DEVICE` macro to ~17 headers (~30 min)
2. Simple kernels: zero, copy, axpy, etc. (~1 hr)
3. Map/unmap: mechanical hipMalloc/hipMemcpy (~1 hr)
4. enforce_algebraic (~30 min)
5. compute_rhs — the big one (~2 hr)
6. sommerfeld (~30 min)
7. Ghost exchange (7 kernels) (~3 hr)
8. Makefile + test (~2 hr)

### 4.8 Key References

- ROCm HIP Programming Guide: rocm.docs.amd.com
- AthenaK (arXiv:2409.10383) — GPU NR with Kokkos targeting HIP
- HIP Porting Guide — CUDA→HIP API mapping tables

---

## 5. HOBC

**Status:** Not implemented. Current Sommerfeld is the "weakest link."
**Effort:** Depends on approach (0 to 3 weeks).

### 5.1 Three Paths (Pick One)

#### Path A: Larger Domain — Zero Code Changes (Recommended First)

Push outer boundary from R~500M to R~1500-2000M using AMR coarse levels.

- **Effort:** 0 lines. Just use `--L 4000`.
- **Cost:** ~10-20% more compute (only adds coarse blocks).
- **Benefit:** Reflections reduced 4-16x. Round-trip time exceeds run duration.
- **This is what every BSSN/CCZ4 production code does.**

#### Path B: Constraint-Preserving Sommerfeld (~500 lines, 2-3 weeks)

Decompose into characteristic variables at boundary, zero incoming constraint
modes. This is what BAM uses.

**CCZ4 characteristic speeds at boundary (far field, alpha→1, beta→0):**

| Speed | Variables | Count |
|-------|-----------|-------|
| 0 | chi, h_ij, Theta | 8 |
| ±1 | K, A_ij, Gamma^i | 20 (10 in, 10 out) |
| ±1 | GW modes | 4 (2 in, 2 out) |

**Implementation:** Project onto characteristic variables, Sommerfeld on
outgoing modes, zero incoming constraint modes (Theta, Gamma^i).

**New files:** `src/boundary/cpbc.h` (~80), `src/boundary/cpbc.c` (~400)
**CLI:** `--bc sommerfeld|cpbc`

#### Path C: Full RWZ-Based HOBC (~1200 lines, 2-3 months)

Bayliss-Turkel hierarchy on gauge-invariant RWZ scalars. **Never done for
CCZ4** — only SpEC (generalized harmonic) has this. Research contribution.

Not recommended unless needed for sub-dominant mode accuracy.

### 5.2 What Production Codes Use

| Code | Formulation | BCs |
|------|------------|-----|
| SpEC | Gen. Harmonic | HOBC (only code with full implementation) |
| SpECTRE | Gen. Harmonic | Constraint-preserving + Bjorhus |
| BAM | Z4c/BSSN | Sommerfeld + constraint-preserving |
| GRChombo | CCZ4 | Sommerfeld |
| Einstein Toolkit | BSSN | Sommerfeld |

**No BSSN/CCZ4 code has full HOBC.** All rely on Sommerfeld + large domains.

### 5.3 Recommendation

1. **Now:** Larger domain with AMR. Zero effort, solves the problem.
2. **Phase 3:** Constraint-preserving Sommerfeld (Path B) for long runs (>10⁴M).
3. **Only if needed:** Full HOBC (Path C) — research project.

### 5.4 Key References

- arXiv:0811.3593 (Buchman, Rinne, Sarbach 2009) — HOBC implementation
- arXiv:1010.0523 (Ruiz, Hilditch, Bernuzzi 2011) — CP BCs for Z4c
- arXiv:1707.09910 (Dumbser et al. 2018) — CCZ4 eigenstructure

---

## 6. Priority & Dependencies

### Dependency Chain

```
Psi4 (3-5 days)
  ├─→ PHE (1-2 weeks) ───→ production waveforms at scri+
  └─→ CCE worldtube (2-3 weeks) ───→ gold-standard via SpECTRE

HIP backend (1-2 weeks) ─── independent, no blockers

Larger domain (0 days) ─── just change --L flag
CP-Sommerfeld (2-3 weeks) ─── independent, nice-to-have
```

### Recommended Order

| # | Item | Effort | Why first? |
|---|------|--------|------------|
| 1 | **Psi4 extraction** | 3-5 days | Blocks everything else |
| 2 | **PHE** | 1-2 weeks | Fast path to publishable waveforms |
| 3 | **HIP backend** | 1-2 weeks | Can run in parallel with PHE |
| 4 | **Larger domain** | 0 days | Just use `--L 4000` |
| 5 | **CCE worldtube** | 2-3 weeks | Gold standard, needs HDF5 |
| 6 | **CP-Sommerfeld** | 2-3 weeks | Nice-to-have for very long runs |

### Fast Path to First Publishable Waveforms

**Psi4 (5 days) + PHE (2 weeks) = ~3 weeks total.**

No HDF5, no external tools, pure C17. Produces waveforms at null infinity with
< 0.05 rad phase accuracy — sufficient for current and next-gen detectors.
