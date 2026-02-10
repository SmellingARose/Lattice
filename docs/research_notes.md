# Research Notes — Open Items from Architecture Doc

Compiled Feb 2026. Each section covers one of the four research items identified in
`NR_Code_Architecture.txt`. Findings are based on reading the referenced papers; key
equations are included for implementation reference.

---

## 1. AMR Scheme Selection

### Candidates

| Property | Block-structured (Berger-Oliger) | Wavelet (Dendro-GR) | Octree |
|---|---|---|---|
| Used by | GRChombo, Carpet/Einstein Toolkit, GR-Athena++ | Dendro-GR | AMReX (not pure NR) |
| Max N demonstrated | **N=25** (arXiv:2505.01495) | N=2 | N=1 |
| Refinement granularity | Rectangular boxes | Cell-level (wavelet coefficients) | Cell-level (octree leaves) |
| Regridding cost | 6–60% depending on strategy | Unknown for fragmented meshes | Unknown at scale |
| Scalability | 1000+ cores (GRChombo), proven MPI+OpenMP | 229K cores, multi-GPU | Unproven for NR |
| Subcycling | Berger-Oliger standard | Supported | Possible but undemonstrated |

### Refinement Strategy Comparison (arXiv:2312.05438)

GR-Athena++ compared three refinement strategies for binary BH:

| Strategy | Mesh blocks vs L∞ | Convergence order | Mismatch at h=9.8e-3 M | Cost vs L₂ |
|---|---|---|---|---|
| Box-in-box (L∞ norm) | baseline | 4th order | ~10⁻⁵ | ~60% more |
| **Sphere-in-sphere (L₂ norm)** | **~60% fewer** | **5th order** | **~10⁻⁵** | **baseline** |
| Truncation error (TE) | ~70% fewer (but...) | 3rd order | >10⁻² | more than L₂ |

Key finding: L∞ generates/destroys mesh blocks at rates **400× higher** than L₂ per timestep,
causing far more restriction/prolongation operator calls. L₂ (sphere-in-sphere) has the
smallest oscillations and fewest interpolation calls.

For next-gen detector accuracy (mismatch ≲ 10⁻⁷): L₂ needs h/M = 6.57×10⁻³ (feasible),
TE needs h/M = 0.51×10⁻³ (impractical).

### GRChombo Implementation Details (arXiv:1503.03436)

- Block-structured AMR via Berger-Rigoutsos algorithm
- Up to 8+ refinement levels, 2:1 ratio: dx_l = dx_0 · 2^{-l}
- Tagging criterion: L₂ norm of field gradient across cell exceeds threshold
- Regridding at user-preset intervals per level
- Kreiss-Oliger dissipation with N=3 (6th order), σ ~ 0.01
- Load balancing: bin-packing/knapsack algorithm, runs every few timesteps
- Scalability: excellent to ~200 cores, useful to ~1000 cores
- CCZ4 with modification κ₁ → κ₁/α for BH stability

### GRChombo 25-BH Result (arXiv:2505.01495)

- 25 equal-mass, nonspinning BHs in compact cluster (R/M ≈ 10)
- Minimum grid spacing: **dx_min ≲ M/(16N) = M/400** (≥32 points across each horizon)
- Initial data: collisionless relativistic cluster equilibrium (not Bowen-York superposition)
  - Sample positions/velocities from density ρ₀(r) and velocity dispersion v²(r)
  - Parameter y_c = 0.819 (central-to-surface redshift)
  - Iterative routine ensures all BHs have same initial horizon mass
- Confirmed gravitationally bound: E_e/M = (M₀ - M)/M > 0

### Recommendation

**Block-structured AMR (Berger-Oliger) with sphere-in-sphere (L₂) refinement strategy.**

Rationale:
1. Only approach demonstrated at N>10 (GRChombo N=25)
2. L₂ strategy is cheapest and most accurate per the GR-Athena++ comparison
3. Proven load balancing and scalability
4. Berger-Oliger subcycling is well-established

### What We Still Don't Know

- Optimal number of levels and box sizes for our N=10+ target on M4 hardware (16 GB limit)
- Whether fixed-box tracking or gradient-based tagging is better for chaotic N-body orbits
- Regridding frequency tuning: too often = expensive, too seldom = loss of resolution
- How L₂ strategy generalizes from binary to N>2 (the comparison was done for N=2 only)
- Memory overhead of AMR metadata for many small boxes around N=10+ punctures

---

## 2. CCZ4 Variant Selection

### The Z4 Framework

The covariant Z4 system introduces a four-vector Z_μ measuring deviation from Einstein's
equations. The constraint Z_μ = 0 recovers standard GR. The damped field equations are
(arXiv:1106.2254):

```
R_μν + ∇_μ Z_ν + ∇_ν Z_μ + κ₁[n_μ Z_ν + n_ν Z_μ - (1+κ₂)g_μν n_σ Z^σ]
    = 8π(T_μν - ½ g_μν T)
```

The conformal decomposition (CCZ4) decomposes this into evolved variables:
{φ, γ̃_ij, K, Ã_ij, Γ̂^i, Θ} where Θ = n_μ Z^μ and Γ̂^i = Γ̃^i + 2γ̃^{ij} Z_j.

### Complete CCZ4 Evolution Equations (arXiv:1106.2254)

**Conformal factor:**
```
∂_t φ = (1/3)αφK - (1/3)φ ∂_k β^k + β^k ∂_k φ
```

**Conformal metric:**
```
∂_t γ̃_ij = -2α Ã^{TF}_ij + 2γ̃_{k(i} ∂_{j)} β^k - (2/3)γ̃_ij ∂_k β^k + β^k ∂_k γ̃_ij
```

**Trace of extrinsic curvature (K):**
```
∂_t K = -∇^i∇_i α + α(R + 2∇_i Z^i + K² - 2ΘK)
        + β^j ∂_j K - 3ακ₁(1+κ₂)Θ + 4πα(S - 3τ)
```

**Tracefree extrinsic curvature:**
```
∂_t Ã_ij = φ²[-∇_i∇_j α + α(R_ij + ∇_i Z_j + ∇_j Z_i - 8πS_ij)]^{TF}
           + α Ã_ij(K - 2Θ) - 2α Ã_{il} Ã^l_j
           + 2Ã_{k(i} ∂_{j)} β^k - (2/3)Ã_ij ∂_k β^k + β^k ∂_k Ã_ij
```

**Theta (Hamiltonian constraint propagation):**
```
∂_t Θ = (1/2)α[R + 2∇_i Z^i - Ã_ij Ã^{ij} + (2/3)K² - 2ΘK]
        - Z^i ∂_i α + β^k ∂_k Θ - ακ₁(2+κ₂)Θ - 8παν
```

**Modified conformal connection (encodes Z_i):**
```
∂_t Γ̂^i = 2α(Γ̃^i_{jk} Ã^{jk} - 3Ã^{ij} ∂_j φ/φ - (2/3)γ̃^{ij} ∂_j K)
           + 2γ̃^{ki}(α ∂_k Θ - Θ ∂_k α - (2/3)αK Z_k)
           - 2Ã^{ij} ∂_j α
           + γ̃^{kl} ∂_k∂_l β^i + (1/3)γ̃^{ik} ∂_k∂_l β^l
           + (2/3)Γ̃^i ∂_k β^k - Γ̃^k ∂_k β^i
           + 2κ₃((2/3)γ̃^{ij} Z_j ∂_k β^k - γ̃^{jk} Z_j ∂_k β^i)
           + β^k ∂_k Γ̂^i - 2ακ₁ γ̃^{ij} Z_j - 16πα γ̃^{ij} S_j
```

where Γ̂^i ≡ Γ̃^i + 2γ̃^{ij} Z_j.

**Lapse (1+log with CCZ4 modification):**
```
∂_t α = -2α(K - 2Θ) + β^k ∂_k α
```

**Shift (Gamma-driver):**
```
∂_t β^i = f B^i + β^k ∂_k β^i
∂_t B^i = ∂_t Γ̂^i - β^k ∂_k Γ̂^i + β^k ∂_k B^i - η B^i
```

### How Z_i Modifies BSSN

The extra terms beyond standard BSSN are:
1. **Γ̂^i definition:** Γ̂^i = Γ̃^i + 2γ̃^{ij} Z_j (Z_i absorbed into connection)
2. **Ã_ij equation:** +∇_i Z_j + ∇_j Z_i terms alongside Ricci tensor
3. **K equation:** +2∇_i Z^i and -2ΘK and -3ακ₁(1+κ₂)Θ damping
4. **Θ equation:** entirely new (Hamiltonian constraint propagation)
5. **Γ̂^i equation:** terms involving Z_k and κ₁ damping of Z_i
6. **Lapse:** K → K - 2Θ (constraint-aware slicing)

### Constraint Damping Parameters

| Parameter | Role | Stability requirement | Typical value |
|---|---|---|---|
| κ₁ | Overall damping strength | κ₁ > 0 | ~0.1/M (or ~0.02) |
| κ₂ | Ratio between H and M damping | κ₂ > -1 | 0 |
| κ₃ | Coupling of Z_i to shift terms | — | 0 or 1 |

GRChombo uses the modification κ₁ → κ₁/α to allow stable BH evolution while retaining
covariance (arXiv:1503.03436).

### CCZ4 Variant Comparison (arXiv:2501.01055)

| Property | CCZ4 (standard) | CCZ4' | CCZ3 |
|---|---|---|---|
| Θ evolved? | Yes | Yes | **No (Θ=0)** |
| Z_i evolved? | Yes, damped by κ₁ | Yes, **undamped** (κ_Γ=0) | Yes, **undamped** (κ₁=0) |
| Damping params | κ₁ (unified) | κ_Θ, κ_Γ (split) | κ₁ = 0 |
| Best κ values | κ₁ = 0.02 | κ_Θ = 0.02, κ_Γ = 0 | N/A |
| Stability (Schwarzschild) | Fails ~20,000 M | Stable to ~100,000 M | **Stable to 10⁵ M** |
| Extra fields vs BSSN | +1 (Θ) | +1 (Θ) | +0 |
| Complexity | Moderate | Moderate | **Simplest** |

**The key insight:** The late-time instability in standard CCZ4 comes from **damping momentum
constraint violations** (the -2ακ₁γ̃^{ij}Z_j term in the Γ̂^i equation). Accumulated
momentum constraint violations, when damped, feed back into the evolution and grow
nonlinearly. Setting κ_Γ = 0 (CCZ4') or removing Θ entirely plus κ₁ = 0 (CCZ3) fixes this.

**CCZ4' modifications:** In the Θ equation, replace κ₁ → κ_Θ. In the Γ̂^i equation,
replace κ₁ → κ_Γ. Set κ_Γ = 0 to prevent momentum damping instability while keeping
Hamiltonian constraint damping via κ_Θ.

**CCZ3:** Set Θ = 0 everywhere (remove from all equations), set κ₁ = 0. Z_i still
propagates (through Γ̂^i) but is never damped. This is the simplest and most robust variant.

### Spatially Varying KO Dissipation (arXiv:2404.01137)

Three improvements to standard BSSN/CCZ4 evolution:

**1. Curvature-Adjusted KO Dissipation (CAKO):**
```
ε_KO(x) = W(x) · ε_{KO,CA}
```
where W = e^{-2φ} is the conformal factor (W → 0 at punctures, W ≈ 1 in weak field).

Coefficient values:
- Gauge quantities (α, β^i): ε_{KO,CA} = 0.99
- Other BSSN/CCZ4 variables: ε_{KO,CA} = 0.3

This naturally suppresses dissipation near punctures (where W → 0) and applies full
strength in weak-field regions.

**2. Constraint-Adjusted Hamiltonian Damping (CAℋD):**
```
∂_t φ += −C · Δt_n · (CFL₀ · Δs_n / Δt_n) · ℋ₋
```
where C = 0.15, CFL₀ = 0.9 × CFL_max, and ℋ₋ = min(ℋ, 0) damps only negative
Hamiltonian constraint violations.

**3. Slow-Start Lapse (SSL):**
```
∂_t α += −W · [h · exp(−t²/(2σ²))] · (α − W)
```
where h = (3/5)M, σ = 20.0 M. This Gaussian term is large at t=0 and decays to zero,
smoothing the initial gauge transient.

Result: 2–3 orders of magnitude reduction in constraint violations in strong-field regions.

### Recommendation

**Start with standard CCZ4** (most established, used by GRChombo, best documented).
Implement the GRChombo modification κ₁ → κ₁/α from day one. Include the arXiv:2404.01137
spatially varying KO dissipation (ε_KO = W · ε_{KO,CA}) — it's a simple multiplication.

**Fallback path:** If late-time instabilities appear (expected beyond ~20,000 M), switch to
CCZ3 by setting Θ = 0 and κ₁ = 0 in all equations. This is a parameter change, not a
rewrite.

### What We Still Don't Know

- Whether CCZ4 instability timescale changes for N>2 (all tests were single BH)
- Whether CCZ3's lack of Hamiltonian damping causes problems for charged BH (untested)
- Optimal κ₁ value for N-body: single-BH optimal may differ from N=10+
- Whether the arXiv:2404.01137 improvements have been tested with CCZ4 (paper uses BSSN)
- Interaction between spatially varying KO dissipation and AMR refinement boundaries

---

## 3. Initial Data: N-Puncture Solver with Charge

### Standard Puncture Method (Bowen-York)

For N uncharged, nonspinning BHs with linear momenta P_i, the conformal factor ψ satisfies
the Hamiltonian constraint on a conformally flat background (arXiv:1004.1353):

```
∇²ψ + (1/8) ψ⁻⁷ Ā_{ij} Ā^{ij} = 0
```

where the Bowen-York extrinsic curvature superposes linearly for N punctures:

```
Ā^{ij} = Σ_n [3/(2R_n²)](P_n^i l_n^j + P_n^j l_n^i - (δ^{ij} - l_n^i l_n^j) P_n^k l_{n,k})
         + Σ_n 3/(R_n³) ε^{kl(i} S_n^{j)} l_{n,k} l_{n,l}]
```

(l_n^i = (x^i - x_n^i)/R_n is the unit vector from puncture n, S_n is spin.)

The conformal factor is split as: ψ = 1 + u + Σ_n M_n/(2R_n), where u is a smooth
correction solved numerically.

### Extension to Charged BHs (arXiv:1903.01036, Bozzola & Paschalidis)

For N charged BHs, the conformal decomposition includes electromagnetic fields:

**Conformal electric field (superposition):**
```
Ē^i = Σ_n (Q_n / R_n²) R̂_n^i
```
This automatically satisfies the Gauss constraint ∂_i Ē^i = 0 (away from punctures).

**Conformal magnetic field:** B̄^i = 0 (for initially non-moving charges; moving charges
generate B fields via the momentum constraint coupling).

**Modified Hamiltonian constraint:**
```
∇²ψ + (1/8) ψ⁻⁷ Ā_{ij} Ā^{ij} + 2π ψ⁻³ ε̄ = 0
```
where the electromagnetic energy density is:
```
4π ε̄ = (1/2)(Ē_i Ē^i + B̄_i B̄^i)
```

**Conformal factor ansatz:**
```
ψ = √(κ² − φ_EM²)
```
where:
```
κ = 1 + u + Σ_n M_n/(2R_n)      (gravitational)
φ_EM = Σ_n Q_n/(2R_n)            (electromagnetic)
```

The correction u satisfies:
```
κ ∇²u + ∂_a κ ∂^a κ − ∂_a φ_EM ∂^a φ_EM − ∂_a ψ ∂^a ψ
    + (1/8) ψ⁻⁶ Ā_{ij} Ā^{ij} + 2π ψ⁻² ε̄ = 0
```

**Momentum constraint (Bowen-York + EM Poynting vector):**
```
V^i = V^i_{0,GR} + V^i_{EM}
```
For Reissner-Nordström (no magnetic field), the Poynting vector S̄^i = 0, so V^i_{EM} = 0.

The gravitational part:
```
V^i_{0,GR} = Σ_n [-(7/4)(P_n^i/R_n) − (1/4)δ_{jk} x_n^j P_n^k (x_n^i/R_n³)
              + ε̄^{ijk} x_{n,j} S_{n,k} / R_n³]
```

### 3+1 Maxwell Evolution Equations (arXiv:0907.1151, Alcubierre et al.)

Once initial data is constructed, the electromagnetic fields evolve via:

**Electric field:**
```
∂_t E^i + L_N E^i = (D × NB)^i + NKE^i − 4πN j^i
```

**Magnetic field:**
```
∂_t B^i + L_N B^i = −(D × NE)^i + NKB^i
```

where L_N is the Lie derivative along the shift, D is the spatial covariant derivative,
and N is the lapse.

**EM energy-momentum tensor components (source terms for CCZ4):**
```
ε = (1/8π)(E² + B²)                           (energy density)
J^a = (1/4π) ε^{acd} E_c B_d                  (Poynting vector)
S_{ab} = (1/8π)[h_{ab}(E² + B²) − 2(E_a E_b + B_a B_b)]  (stress tensor)
```

### Charged Binary Results (arXiv:2104.06978, Bozzola & Paschalidis)

- Stable evolution of charged binary BH inspiral with λ = Q/M ≤ 0.3
- Used modified TwoPunctures code ("TwoChargedPunctures")
- Spectral solver with Chebyshev polynomials in bispherical coordinates
- Newton-Raphson iteration for the nonlinear conformal factor equation
- Quasi-local charge measurement: Q_S = (1/4π) ∮_S E^c dA on horizon surface

### N>2 Puncture Solver (arXiv:1004.1353)

- Used **high-order multigrid elliptic solver** (not spectral TwoPunctures)
- Demonstrated 3-BH and 4-BH evolutions with **6th-order convergence**
- Key finding: approximate analytic solution (used in prior 3-BH work) gives
  **different dynamics and waveforms** — full numerical solve is necessary
- Multigrid naturally handles N>2 without the bispherical coordinate limitations of
  TwoPunctures (which is inherently a 2-center decomposition)

### Solver Approach Decision

| Solver type | N=2 | N>2 | Charge | Proven? |
|---|---|---|---|---|
| TwoPunctures (spectral) | Excellent | Hard (bispherical coords) | Yes (modified) | N=2 only |
| Multigrid | Good | **Natural extension** | Straightforward | N=3,4 demonstrated |
| GRTresna (multigrid, C++) | Good | N=2 only currently | No | New code, limited |

### Recommendation

**Multigrid elliptic solver for N punctures.** The spectral TwoPunctures approach is
inherently limited to 2 centers by its bispherical coordinate system. Multigrid scales
naturally to N>2 and is proven at N=3,4 (arXiv:1004.1353).

For charge: follow Bozzola & Paschalidis's conformal decomposition. The electric field
superposes analytically (Ē^i = Σ Q_n/R_n² R̂_n^i) and automatically satisfies the Gauss
constraint. The Hamiltonian constraint gains one extra EM source term (2πψ⁻³ε̄). This is
a modest modification to the standard N-puncture solver.

### What We Still Don't Know

- Whether to write our own multigrid solver or adapt existing code (GRTresna is C++, we need C)
- Convergence rate of multigrid for N=10+ with charge (tested only to N=4 uncharged)
- How to set bare masses M_n to achieve desired horizon masses (requires iteration)
- Whether spectral methods could work for N>2 with a different coordinate system
- Spin + charge + boost simultaneously: the Bowen-York superposition is proven for each
  separately, but the combined nonlinear solve may have convergence issues at high spin+charge

---

## 4. Position-Dependent η for N=10+

### The Problem

In the Gamma-driver shift condition:
```
∂₀ B^i = ∂₀ Γ̃^i − η B^i
```

Constant η = 2/M works for comparable-mass binaries but fails for:
- Unequal masses (different BHs need different damping)
- N>2 configurations (single constant can't adapt to all punctures)

### Approach 1: Explicit Position-Dependent Forms (arXiv:1003.0859, Müller & Brügmann)

**Smooth form:**
```
η(r) = η* · R² / (r² + R²)
```
where R is a transition radius and η* is the peak value.

**Piecewise form (implemented in McLachlan):**
```
η(r) = η* · { 1         for r ≤ R
             { R/r       for r > R
```

**Multi-BH generalization (reciprocal superposition):**
```
1/η(x) = Σ_n 1/η_n(|x − x_n|)
```
where η_n is the single-BH η profile centered on puncture n.

This requires **explicit tracking of puncture positions** — each η_n depends on distance
to puncture n. For N=10+, this means tracking 10+ positions every timestep and evaluating
N distance functions at every grid point.

### Approach 2: Conformal-Factor-Based η

```
η(x) = η₀ / W(x)     or     η(x) = η₀ · f(W(x))
```
where W = e^{-2φ} is the conformal factor already computed for KO dissipation.

**Advantages:**
- No explicit puncture tracking needed
- Automatically adapts to all punctures simultaneously (W → 0 near each one)
- Uses the same W field as the spatially varying KO dissipation
- Smooth and differentiable everywhere

**Disadvantages:**
- Less control per-puncture (can't set different η for different masses independently)
- W doesn't directly encode mass — a 5M BH and a 1M BH both have W → 0 at the puncture
- For very unequal masses, the W profile near each puncture differs, which partially
  compensates, but may not be sufficient

### Approach 3: Metric-Determinant-Based η

Müller & Brügmann also mention using the determinant of the 3-metric as an automatic
proxy for proximity to punctures, avoiding explicit distance calculations. This is
conceptually similar to the conformal factor approach.

### Parameter Values

- Standard constant: η = 2/M (M = total ADM mass)
- Per-puncture: η_n ~ 1/M_n (scales inversely with individual BH mass)
- Transition radius R: typically ~ few × M_n (horizon scale of each BH)

### Recommendation

**Start with conformal-factor-based η:** η(x) = η₀ / W(x) or η(x) = η₀ · (1 + c/W(x)).
This avoids puncture tracking, reuses the W field from KO dissipation, and automatically
adapts to all N punctures.

**Fallback:** If conformal-factor η is insufficient for very unequal masses (q > 4:1),
switch to explicit reciprocal superposition with puncture tracking.

### What We Still Don't Know

- No published work on η for N>4 configurations — we will be the first to test at N=10+
- Whether conformal-factor-based η provides sufficient mass-dependent adaptation
- CFL stability interaction: η affects the shift evolution, which affects the CFL condition
  on each AMR level. Spatially varying η may require smaller timesteps.
- Optimal functional form f(W) — linear, inverse, square root?
- Whether the 25-BH GRChombo run (arXiv:2505.01495) used position-dependent η (paper
  doesn't specify their gauge parameter choices in detail)

---

## References

1. arXiv:2505.01495 — Clough et al., "Evolution of a Black Hole Cluster in Full General Relativity" (2025). GRChombo N=25 BH simulation.
2. arXiv:2312.05438 — Rashti et al., AMR refinement strategy comparison in GR-Athena++ (2024). Published: Class. Quantum Grav. 41 095001.
3. arXiv:2501.01055 — CCZ4 variant comparison: CCZ4 vs CCZ4' vs CCZ3 (2025). Published: Phys. Rev. D 111, 084018.
4. arXiv:1106.2254 — Alic et al., "Conformal and covariant formulation of the Z4 system" (2012). Published: Phys. Rev. D 85, 064040.  Original CCZ4 paper.
5. arXiv:2404.01137 — Improved moving-puncture techniques: CAKO dissipation, CAℋD, SSL (2024).
6. arXiv:2104.06978 — Bozzola & Paschalidis, charged binary BH inspiral with λ ≤ 0.3 (2021). Published: Phys. Rev. D 104, 044004.
7. arXiv:0907.1151 — Alcubierre, Degollado & Salgado, "Einstein-Maxwell system in 3+1 form and initial data for multiple charged black holes" (2009). Published: Phys. Rev. D 80, 104022.
8. arXiv:1903.01036 — Bozzola & Paschalidis, "Initial data for charged BHs with linear and angular momenta" (2019). Published: Phys. Rev. D 99, 104044.
9. arXiv:1004.1353 — Lousto et al., "Numerical evolution of multiple black holes with accurate initial data" (2010). Published: Phys. Rev. D 82, 024005.
10. arXiv:1003.0859 — Müller & Brügmann, "Dynamical shift condition for unequal mass BH binaries" (2010). Published: Phys. Rev. D 82, 064004.
11. arXiv:1503.03436 — Clough et al., "GRChombo: Numerical relativity with adaptive mesh refinement" (2015).
12. arXiv:2501.13046 — GRTresna: open-source initial data solver (2025). C++, multigrid, currently N≤2.

### Reference Implementations

| Code | Language | Open source | Relevant for |
|---|---|---|---|
| [GRChombo](https://github.com/GRChombo) | C++ | Yes | AMR, CCZ4, N-body |
| [Einstein Toolkit](https://einsteintoolkit.org) | C/C++/Fortran | Yes | TwoPunctures, Carpet AMR |
| [GR-Athena++](https://github.com/FMhyp/GRAthena) | C++ | Partial | AMR strategies, BSSN |
| [Dendro-GR](https://github.com/paralab/Dendro-GR) | C++ | Yes | Wavelet AMR, BSSN |
| [GRTresna](https://github.com/GRTLCollaboration/GRTresna) | C++ | Yes | Multigrid initial data |
| TwoChargedPunctures | C | In Einstein Toolkit | Charged BH initial data (N=2) |
