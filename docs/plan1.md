# Milestone 5: Binary Inspiral — Initial Data Research & Plan

> **Note (2026-02-19):** The hyperbolic relaxation solver described in Section 5
> was superseded by an FAS multigrid solver (FMG + FAS V-cycles +
> Newton-Gauss-Seidel smoother). See DEVLOG.md entry for 2026-02-19 and the
> updated `docs/bowen_york.html` for the current algorithm. The Bowen-York
> initial data formulation (Sections 3-4) and the implementation roadmap
> (Section 8) remain accurate.

## Status: Research complete, implementation pending

---

## 1. The Problem

Our code currently uses Brill-Lindquist initial data: `psi = 1 + sum(M_n/2r_n)`,
with `K = 0` and `A_ij = 0`. Black holes start at rest — no momentum, no spin.
Binary inspiral requires orbital momentum (tangential velocity) and eventually spin.

The challenge: once `A_ij != 0`, the conformal factor `psi` is no longer analytic.
We must solve the Hamiltonian constraint as a nonlinear elliptic PDE.

---

## 2. Method Survey (Exhaustive)

### 2.1 Initial Data Formulations

| Formulation | Equations Solved | Spin Limit | N-body | Puncture Compatible |
|-------------|-----------------|------------|--------|---------------------|
| **CTT + Hamiltonian only** | 1 scalar PDE (nabla^2 psi = S) | chi <= 0.93 | Any N | Yes (native) |
| **CTS (non-extended)** | 4 PDEs (psi + shift) | chi <= 0.93 | Any N | Partial (Tichy CTSP) |
| **XCTS** | 5 coupled PDEs (psi + shift + lapse) | chi <= 0.93 (CF) | Any N | **Requires excision** |
| **XCTS + Superposed Kerr-Schild** | 5 PDEs on Kerr background | chi <= 0.9997 | Any N | **Requires excision** |
| **HiSpID** (quasi-isotropic Kerr) | Coupled H + momentum constraints | chi <= 0.99 | N >= 2 | Yes (native) |

**Key finding — XCTS requires excision for binary BH:**
The Tichy-Brugmann obstruction (PRD 68, 064003, 2003) shows that the XCTS lapse
equation is ill-posed at punctures for quasiequilibrium binary data. The lapse must
pass through zero at the puncture, which makes the equation singular. All production
XCTS implementations (SpECTRE, SpEC, KADATH) use excision.

**The "fill the holes" workaround:**
Etienne et al. (arXiv:0707.2083, 2007) proved you can solve XCTS with excision,
fill the excised interiors with arbitrary smooth data, and evolve with moving
punctures. The exterior solution and waveforms are unchanged. Production-proven
(Varma et al. 2022, arXiv:2202.12133).

### 2.2 Elliptic Solver Methods

| Solver | Type | Reuses Our Infra | New LOC | Convergence | GPU Fit |
|--------|------|------------------|---------|-------------|---------|
| **Hyperbolic relaxation** | Damped wave -> steady state | RK4, FD, Sommerfeld, AMR, GPU | ~400-600 | O(N^4) in 3D | Excellent |
| Multigrid (FAS) | V-cycle iteration | Prolong/restrict only | ~3000-5000 | O(N log N) | Poor |
| TwoPunctures | Spectral (Chebyshev) | None | ~3000 (external) | Exponential | No |
| Newton-CG | Krylov iteration | FD, ghost exchange | ~500 | O(N^{3/2}) | Good |
| Parabolic relaxation | Heat equation | RK4, FD | ~150 | O(N^5) in 3D | Good |
| SOR | Over-relaxation | FD only | ~200 | O(N^{5/3}) | Poor |

**Winner: Hyperbolic relaxation.** Reuses 90%+ of our existing infrastructure
(RK4/CK45, 4th-order FD, Sommerfeld BCs, ghost exchange, AMR with subcycling,
GPU batch kernels). Our AMR subcycling provides wave-speed acceleration equivalent
to NRPyElliptic's curvilinear coordinates.

**Ref:** Ruter et al. arXiv:1708.07358 (method), Assumpcao et al. arXiv:2111.02424
(NRPyElliptic implementation), Tootle et al. arXiv:2501.14030 (GPU version).

### 2.3 What Accuracy Do We Actually Need?

For waveform overlap > 0.999 with spins up to chi = 0.8:

- Conformal flatness phase error: ~0.01 radians (budget is ~0.045 rad) — within budget
- Junk radiation mismatch at chi < 0.8: ~1e-5 to 1e-4 — well below threshold
- Solver accuracy beyond FD truncation error (~dx^4) is wasted
- **CTT + Hamiltonian solve is sufficient for chi < 0.8**

Full XCTS solver NOT needed until chi > 0.93.
Superposed Kerr-Schild NOT needed until chi > 0.93.

**Ref:** Varma, Scheel, Pfeiffer arXiv:1808.08228 (comparison of BBH initial data sets).

---

## 3. Evolution Improvements (Already Implemented)

Three single-line improvements from Etienne arXiv:2404.01137 are already in our code:

- **CAKO** (Curvature-Adjusted KO dissipation): `dissipation.c:45-49` — multiplies
  sigma by W = sqrt(chi) near punctures, reducing artificial dissipation in strong field
- **CAHD** (Coarse-grid Adjusted Hamiltonian Damping): `ccz4_rhs.c:291-301` — adds
  constraint-damping term to chi evolution, reduces Ham violations by ~2 orders of magnitude
- **SSL** (Slow-Start Lapse): `ccz4_rhs.c:403-414` — Gaussian ramp on initial gauge
  pulse, reduces junk radiation

These reduce GW noise by 4.3x with zero overhead. Enabled via `p.noise.use_cako`,
`p.noise.use_cahd`, `p.noise.use_ssl` flags in `params.h:59-68`.

**Ref:** arXiv:2501.01055 — CCZ3/CCZ4' (separate kappa_Theta from kappa_Gamma)
enables stable evolution to 10^5 M. Not yet implemented but straightforward.

---

## 4. The Bowen-York Extrinsic Curvature

### 4.1 The Formula (Triple-Verified)

For puncture n at position C_n with momentum P_i and spin S_i:

```
A_ij^(n) = A_ij^P + A_ij^S
```

**Momentum term:**
```
              3
A_ij^P = --------- [ P_i n_j + P_j n_i - (delta_ij - n_i n_j)(P . n) ]
           2 r^2
```

Equivalently (TwoPunctures form):
```
              3
A_ij^P = --------- [ P_i n_j + P_j n_i + (n.P) n_i n_j ] - delta_ij * 3(n.P) / (2r^2)
           2 r^2
```

**Spin term:**
```
              3
A_ij^S = - ------- [ (n x S)_i n_j + (n x S)_j n_i ]
            r^3
```

Where:
- n_i = (x_i - C_i) / r is the unit radial vector from puncture
- r = |x - C| with floor at r = 1e-6 to regularize
- (n x S)_i = epsilon_{ijk} n_j S_k (standard cross product)
- Momentum terms fall off as 1/r^2, spin terms as 1/r^3

**For N punctures:** Linear superposition: `A_ij = sum_n A_ij^(n)`.

**The trace A_ij A^ij** (needed for Hamiltonian constraint source):
```
A_ij A^ij = sum_{i,j} (A_ij)^2       (flat conformal metric: trivial raising)
```

### 4.2 Cross-Reference Sources

| Source | File | Momentum | Spin |
|--------|------|----------|------|
| TwoPunctures | `Equations.cc:BY_Aijofxyz()` | Yes | Yes |
| GRChombo | `BoostedBH.impl.hpp:47-50` | Yes | **No** (momentum only) |
| NRPyElliptic | arXiv:2111.02424, Eq. (4) | Yes | Yes |
| B&S textbook | Chapter 3, pp. 73-74 | Yes | Yes |
| Brandt-Brugmann | gr-qc/9703066 | Yes | Yes |

### 4.3 Conversion to CCZ4 Variables

After computing A_ij^phys (Bowen-York) and solving for psi:

```
chi        = psi^{-4}
h_ij       = delta_ij                    (conformal metric is flat)
K          = 0                           (maximal slicing, Bowen-York property)
A_ij^CCZ4  = chi^{3/2} * sum_n(A_ij^phys_n)    [= psi^{-6} * A_ij^phys]
Theta      = 0
Gamma^i    = 0                           (flat conformal metric)
lapse      = sqrt(chi)                   (pre-collapsed, same as current)
shift^i    = 0
B^i        = 0
```

**Ref:** GRChombo `BinaryBH.impl.hpp:53-68`.

### 4.4 Validity Limits

- The superposed conformal factor approximation (`psi = 1 + sum(psi0 + P^2/M^2 * psi2)`)
  is only valid for |P| < 0.3M (GRChombo SimulationParameters.hpp:278).
- For larger boosts, the elliptic solver correction u is essential.
- Spin: Bowen-York conformally flat data limits effective spin to chi <= 0.93 after
  junk radiation dissipates. Excess angular momentum radiates away.

---

## 5. The Hyperbolic Relaxation Solver

### 5.1 The Algorithm

Convert the elliptic Hamiltonian constraint into a damped wave equation in
pseudo-time tau:

```
d_tau u  = v - eta * u
d_tau v  = c^2 * [ nabla^2(u) + S(u) ]
```

where:
- u is the correction to the conformal factor (unknown)
- v is the pseudo-velocity (auxiliary)
- eta is the damping parameter
- c is the wave speed
- S(u) = -(1/8) * A_ij A^ij * (psi_BL + u)^{-7} is the nonlinear source

As tau -> infinity, d_tau u = d_tau v = 0, so u satisfies:
```
nabla^2(u) + S(u) = 0
```
which is the original Hamiltonian constraint.

**Ref:** Ruter et al. arXiv:1708.07358 (method), NRPyElliptic arXiv:2111.02424.

### 5.2 Parameters

- **Wave speed c:** Constant on uniform grid, `c = 1` in code units.
  CFL: `dt_relax = CFL_relax * dx / c` with CFL_relax ~ 0.5-0.7.
  On AMR, our subcycling handles variable dt per level automatically.
- **Damping eta:** Optimal eta ~ 1-20/L where L is domain size.
  Found by linear sweep with 0.25 increments. Too large = overdamped (slow),
  too small = underdamped (oscillatory).
- **Convergence criterion:** L2 norm of v drops below threshold (1e-12 to 1e-14).
  Monitor ||v||_L2 = sqrt(sum(v_i^2) / N).

### 5.3 Infrastructure Reuse

| Component | How Reused |
|-----------|-----------|
| RK4/CK45 time integrator | Evolve the 2-field damped wave system in pseudo-time |
| FD_D1 / FD_D2 macros | Compute nabla^2(u) = d2x(u) + d2y(u) + d2z(u) |
| Sommerfeld BCs | Robin BC: u -> 0 at infinity (set f_asymptotic = 0) |
| Ghost exchange | Same 26-neighbor exchange for u, v arrays |
| AMR subcycling | Wave-speed acceleration: coarse levels take larger dt |
| GPU batch kernels | Relaxation RHS is simpler than CCZ4 RHS — better GPU perf |
| KO dissipation | Optional on u and v to damp high-frequency noise |

### 5.4 Performance Estimates

- **Uniform grid N=128:** ~1500 relaxation steps, ~0.1 sec/step on M4 = ~150 sec total
- **AMR grid (3-4 levels):** 2-5x speedup from subcycling = ~30-75 sec
- **GPU (V100/A100):** 20-100x speedup = ~1-5 sec
- **Fraction of total evolution:** < 1% (negligible)

---

## 6. Charge Extension (Phase 2b)

The Einstein-Maxwell coupling is **additive** to the Hamiltonian constraint:

```
nabla^2 psi + (1/8) A_ij A^ij psi^{-7} + pi (E_bar^2 + B_bar^2) psi^{-7} = 0
```

The EM source has the **same psi^{-7} structure** — no solver changes needed.
The electric field has an analytic Coulomb solution (decouples from psi):

```
E_bar^i = Q / (4 pi r^2) * n^i
```

**Full Einstein-Maxwell evolution:** ~530 lines (6 new fields E^i, B^i, evolution
equations, stress-energy coupling to CCZ4, constraint damping).

**Ref:** Alcubierre et al. arXiv:0907.1151, Bozzola & Paschalidis arXiv:1903.01036,
Zilhao et al. arXiv:1205.1063 (Q/M up to 0.99 demonstrated).

---

## 7. High-Spin Extension (Phase 2a)

### 7.1 HiSpID (Puncture-Native, chi <= 0.99)

Ruchlin et al. arXiv:1410.8607 — uses quasi-isotropic Kerr coordinates (NOT
Kerr-Schild) which are puncture-compatible:

- Non-conformally-flat metric near each BH (captures true Kerr geometry)
- Gaussian-weighted superposition (same idea as SKS but in puncture coords)
- Solves coupled Hamiltonian + momentum constraints (4 PDEs, not 5)
- Our hyperbolic relaxation solver extends naturally (just more fields)
- ~1200 lines total (including Step 1 infrastructure)
- Order of magnitude less junk radiation than standard Bowen-York

### 7.2 XCTS + SKS + "Fill the Holes" (Ultimate, chi <= 0.9997)

For the absolute best initial data quality:
1. Build our own excision-based XCTS solver
2. Superposed Kerr-Schild background
3. Fill excised interiors with smooth data
4. Evolve with our existing puncture code

This is a bigger effort (excision handling, apparent horizon finder) but proven
in production by SXS collaboration.

---

## 8. Implementation Roadmap

### Step 1: CTT + Hyperbolic Relaxation (~600 lines) — Milestone 5

| File | Changes | LOC |
|------|---------|-----|
| `src/core/params.h` | Add `puncture_data_t` struct (mass, center, momentum, spin) | +25 |
| `src/main.c` | Extend `--puncture` CLI to 10 values, dispatch to BY solver | +40 |
| `src/initial_data/puncture.c` | Refactor to use `puncture_data_t` | ~20 changed |
| **`src/initial_data/bowen_york.c`** (new) | BY A_ij formula + CCZ4 field setup | +150 |
| **`src/initial_data/bowen_york.h`** (new) | Header | +20 |
| **`src/initial_data/relaxation.c`** (new) | Hyperbolic relaxation solver | +250 |
| **`src/initial_data/relaxation.h`** (new) | Header | +15 |
| `Makefile` | Add new .c files | +3 |
| `tests/test_bowen_york.c` (new) | Validation tests | +200 |

**Enables:** Orbiting binaries with momentum, spin up to chi ~ 0.93, N-body.

### Step 2: HiSpID (~600 more lines) — Phase 2a

- Quasi-isotropic Kerr metric computation
- Coupled H + momentum constraint relaxation (4 fields instead of 1)
- Spin up to chi ~ 0.99

### Step 3: Einstein-Maxwell (~530 lines) — Phase 2b

- 6 new EM fields (E^i, B^i)
- Maxwell evolution equations with constraint damping
- EM stress-energy coupling to CCZ4
- Charged puncture initial data (add EM source to relaxation)

### Step 4: Excision-Based XCTS Solver — Phase 3

- Full 5-equation XCTS with superposed Kerr-Schild
- Excision + apparent horizon boundary conditions
- "Fill the holes" for puncture evolution
- Spin up to chi ~ 0.9997

---

## 9. Equation References

### Primary Papers
- **gr-qc/9703066** — Brandt & Brugmann: puncture method, Bowen-York decomposition
- **arXiv:2111.02424** — NRPyElliptic: hyperbolic relaxation for Hamiltonian constraint
- **arXiv:1708.07358** — Ruter et al.: hyperbolic relaxation method (mathematical foundation)
- **arXiv:2501.14030** — NRPyEllipticGPU: GPU-accelerated relaxation
- **gr-qc/0404056** — Ansorg et al.: TwoPunctures spectral solver

### Reference Implementations (Code Cross-Check)
- **TwoPunctures** `Equations.cc:BY_Aijofxyz()` — Bowen-York A_ij (momentum + spin)
- **TwoPunctures** `Equations.cc:NonLinEquations()` — Hamiltonian constraint source
- **GRChombo** `BoostedBH.impl.hpp:47-50` — Bowen-York A_ij (momentum only)
- **GRChombo** `BinaryBH.impl.hpp:53-68` — Superposition + CCZ4 conversion
- **GRChombo** `KerrBH.impl.hpp` — Exact Kerr solution (spin validation)

### High-Spin and Charge
- **arXiv:1410.8607** — HiSpID: quasi-isotropic Kerr puncture data (chi <= 0.99)
- **arXiv:0805.4192** — Lovelace et al.: superposed Kerr-Schild (chi <= 0.9997)
- **arXiv:0707.2083** — Etienne et al.: "fill the holes" (excision data + puncture evolution)
- **arXiv:2202.12133** — Varma et al.: conformally curved charged data + puncture evolution
- **arXiv:0907.1151** — Alcubierre et al.: Einstein-Maxwell 3+1 formulation
- **arXiv:1903.01036** — Bozzola & Paschalidis: charged puncture initial data
- **arXiv:2104.06978** — Bozzola & Paschalidis: first charged binary inspiral

### Evolution Improvements
- **arXiv:2404.01137** — Etienne: CAKO/CAHD/SSL (already implemented)
- **arXiv:2501.01055** — CCZ3/CCZ4' stable to 10^5 M
- **arXiv:2410.05531** — Habib et al.: eccentricity reduction for quasi-circular orbits

### Feasibility Research
- **Tichy & Brugmann PRD 68, 064003 (2003)** — XCTS + puncture obstruction
- **Pfeiffer & York gr-qc/0504142** — XCTS lapse equation non-uniqueness
- **Baumgarte et al. gr-qc/0610120** — XCTS wrong-sign analysis
- **East, Pretorius, Stephens arXiv:1208.3473** — CTS solver without excision
- **Baumgarte arXiv:1202.4639** — Solve for W = 1/psi (finite at punctures)
- **arXiv:2501.13046** — GRTresna: multigrid solver (XCTS forthcoming)

### N-Body and Waveform Catalogs
- **arXiv:1004.1353** — Lousto et al.: N-body accurate initial data (N=3,4)
- **arXiv:2505.01495** — GRChombo 25-BH cluster simulation
- **arXiv:2505.13378** — SXS third catalog (3,756 BBH simulations)
- **arXiv:1808.08228** — Varma et al.: comparison of BBH initial data sets

### Novel Methods (Surveyed, Not Selected)
- **arXiv:2207.03125** — CTTK method (algebraic Hamiltonian, not for vacuum BBH)
- **arXiv:2509.11144** — Analytical perturbative corrections (warm-start for solver)
- **arXiv:2307.08867** — Racz parabolic-hyperbolic (requires spherical harmonics)
- **arXiv:2111.06767** — SpECTRE DG elliptic solver (incompatible architecture)
