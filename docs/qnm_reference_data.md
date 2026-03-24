# Schwarzschild QNM Reference Data

Comprehensive reference for validating numerical relativity QNM ringdown results
against known analytic and published numerical values.

Last updated: 2026-03-24

---

## 1. Exact QNM Frequencies (Gravitational, s = -2)

All values are Mw where w = w_R - i w_I (convention: w_I > 0 for damped modes).
Units: 1/M (geometric, G = c = 1).

### Primary source: Leaver (1985), tabulated by Berti, Cardoso & Starinets (2009)

Leaver's continued-fraction method gives effectively exact results (limited only
by machine precision). Values below are from the Kokkotas & Schmidt Living Review
(Liv. Rev. Rel. 2, 2, 1999, Table 2) cross-checked against
Berti et al. (Class. Quant. Grav. 26, 163001, 2009, arXiv:0905.2975).

#### l = 2 (quadrupole -- dominant mode in ringdown)

| n | M w_R | M w_I | Period T/M | Damping time tau/M | Quality Q |
|---|---------|---------|------------|--------------------|-----------|
| 0 | 0.37367 | 0.08896 | 16.82 | 11.24 | 2.10 |
| 1 | 0.34671 | 0.27391 | 18.13 | 3.65 | 0.63 |
| 2 | 0.30105 | 0.47828 | 20.87 | 2.09 | 0.31 |
| 3 | 0.25150 | 0.70514 | 24.98 | 1.42 | 0.18 |

#### l = 3 (octupole)

| n | M w_R | M w_I |
|---|---------|---------|
| 0 | 0.59944 | 0.09270 |
| 1 | 0.58264 | 0.28130 |
| 2 | 0.55168 | 0.47909 |
| 3 | 0.51196 | 0.69034 |

#### l = 4 (hexadecapole)

| n | M w_R | M w_I |
|---|---------|---------|
| 0 | 0.80918 | 0.09416 |
| 1 | 0.79663 | 0.28443 |
| 2 | 0.77271 | 0.47991 |
| 3 | 0.73984 | 0.68392 |

### Derived physical quantities for (l=2, n=0)

| Quantity | Formula | Value (M=1) |
|----------|---------|-------------|
| Oscillation frequency | w_R | 0.37367 / M |
| Damping rate | w_I | 0.08896 / M |
| Period | T = 2 pi / w_R | 16.82 M |
| Damping time | tau = 1 / w_I | 11.24 M |
| Quality factor | Q = w_R / (2 w_I) | 2.10 |
| e-folding cycles | w_R / (2 pi w_I) | 0.669 |
| GW frequency | f = w_R / (2 pi) | 0.05947 / M |
| Half-period | T/2 = pi / w_R | 8.41 M |

### Key observations

- For all l, the real part w_R decreases with overtone number n.
- The imaginary part w_I increases roughly linearly: w_I ~ (n + 1/2) * 0.19 for l=2.
- The fundamental mode (n=0) is the longest-lived and dominates late-time ringdown.
- Higher l modes have higher w_R (faster oscillation) but similar w_I at each n.
- The n >= 1 overtones decay much faster than the fundamental; they are only visible
  in the first ~10-20M after merger/perturbation onset.

### Higher precision values for (l=2, n=0)

From Leaver's method at extended precision (Berti et al. 2009 data tables,
Stein's `qnm` Python package):

```
M w = 0.37367168 - 0.08896232 i
```

This is the standard benchmark value used across all NR codes. Any code claiming
QNM validation should reproduce these digits to at least 3 significant figures
at the finest available resolution.

---

## 2. Published Numerical Results from NR Codes

### 2.1 BAM (Jena/UIB)

- **Measured**: w_R = 0.374 +/- 0.003, w_I = 0.089 +/- 0.005
- **Resolution**: AMR, dx_fine ~ 0.1M, 6th-order FD
- **Formulation**: Z4c (CCZ4 variant)
- **Extraction**: Psi4 at r = 50-100M, (2,2) mode from BBH merger
- **Reference**: Husa et al., Phys. Rev. D 77, 044037 (2008), arXiv:0706.0740
- **Notes**: BAM is the primary reference for Z4c/CCZ4 validation.
  Hilditch et al. (arXiv:1212.2901) demonstrate convergence with
  constraint-preserving BCs.

### 2.2 SpEC (Caltech-Cornell)

- **Measured**: w_R = 0.3737, w_I = 0.0890 (spectral accuracy)
- **Resolution**: Pseudo-spectral, exponential convergence
- **Formulation**: Generalized harmonic
- **Extraction**: Cauchy-characteristic extraction (CCE) for gauge-invariant strain
- **Reference**: Scheel et al., Phys. Rev. D 79, 024003 (2009), arXiv:0810.1767
- **Notes**: Highest accuracy QNM extraction in the literature. Achieves ~6 digits
  of agreement with perturbation theory after CCE.

### 2.3 GRChombo (GRTL Collaboration)

- **Measured**: w_R = 0.374 +/- 0.004, w_I = 0.089 +/- 0.008
- **Resolution**: AMR with Chombo, dx_fine ~ 0.06M, 4th-order FD
- **Formulation**: CCZ4
- **Extraction**: Psi4 on coordinate spheres
- **Reference**: Clough et al., Class. Quant. Grav. 32, 245011 (2015), arXiv:1503.03436
- **Notes**: Direct comparison code for Lattice (same CCZ4 formulation).
  GRChombo uses 4th-order FD vs our 6th-order.

### 2.4 GR-Athena++ (Daszuta et al.)

- **Measured**: Not published for single BH; validated via cross-code comparison
  with BAM for BBH waveforms
- **Resolution**: AMR oct-tree, 6th-order FD
- **Formulation**: Z4c
- **Extraction**: Psi4 at multiple radii
- **Reference**: Daszuta et al., ApJS 257, 25 (2021), arXiv:2101.08289
- **Notes**: Closest architecture to Lattice (vertex-centered AMR, 6th-order FD).
  Strong scaling to 1.2e4 CPUs, weak scaling to 1e5 CPUs.

### 2.5 Einstein Toolkit / Lean / McLachlan

- **Measured**: Typically w_R within 1-2% of exact at dx ~ 0.2-0.4M
- **Resolution**: Carpet AMR, 4th or 8th order FD
- **Formulation**: BSSN or CCZ4 (McLachlan thorn)
- **Extraction**: WeylScal4 + Multipole thorns
- **Reference**: Loffler et al., Class. Quant. Grav. 29, 115001 (2012), arXiv:1111.3344
- **Notes**: Community standard. Extensive regression test suite includes
  single BH ringdown.

### Summary comparison

| Code | w_R | w_I | dx_fine | FD order | Formulation |
|------|-----|-----|---------|----------|-------------|
| Exact | 0.37367 | 0.08896 | -- | -- | perturbation theory |
| BAM | 0.374(3) | 0.089(5) | 0.1M | 6th | Z4c |
| SpEC | 0.3737 | 0.0890 | spectral | spectral | gen. harmonic |
| GRChombo | 0.374(4) | 0.089(8) | 0.06M | 4th | CCZ4 |
| GR-Athena++ | (cross-validated) | -- | varies | 6th | Z4c |
| ET/Lean | ~0.374 | ~0.089 | 0.2-0.4M | 4th-8th | BSSN/CCZ4 |

---

## 3. Trumpet Gauge Solution

### Stationary lapse at the puncture

In moving-puncture gauge (1+log slicing with coefficient c=2, Gamma-driver shift
with eta=1), the Schwarzschild spacetime settles to a "trumpet" geometry.

**Key values for the 1+log trumpet** (Hannam et al., Phys. Rev. D 78, 064020,
2008, arXiv:0804.0628; Brugmann, Gen. Rel. Grav. 41, 2131, 2009, arXiv:0904.4418):

| Quantity | Value | Notes |
|----------|-------|-------|
| Lapse at puncture (alpha_min) | ~0.30 | Stationary value; exact depends on resolution |
| Trumpet areal radius R_0 | ~1.31M | Cylinder radius inside horizon |
| Apparent horizon radius | ~0.85M | In isotropic coordinates |
| Time to reach trumpet | ~30-50M | From Brill-Lindquist initial data |
| Conformal factor at puncture | psi ~ 3M/(2r) as r->0 | Diverges at puncture |
| chi at puncture | ~0 | chi = 1/psi^4 -> 0 at puncture |

The Dennison & Baumgarte family of analytical trumpet slices
(Phys. Rev. D 90, 044028, 2014, arXiv:1403.5484) parameterizes stationary slices
by R_0 (0 < R_0 <= M). The standard 1+log result with c=2 gives R_0 ~ 1.31M.

**Practical test criterion**: After evolving a single Schwarzschild puncture for
T >= 50M, the minimum lapse should satisfy:
- 0.1 < alpha_min < 0.5 (gauge has collapsed but is stable)
- Best resolved value: alpha_min ~ 0.30 +/- 0.05 depending on resolution
- If alpha_min -> 0: gauge singularity, likely a bug
- If alpha_min -> 1: gauge has not evolved, likely lapse equation not being solved

### Gauge dynamics timeline

| Time | Event |
|------|-------|
| t = 0 | Brill-Lindquist: alpha = 1 everywhere, shift = 0 |
| t ~ 5-10M | Lapse collapse begins; alpha drops toward ~0.3 near puncture |
| t ~ 10-20M | Shift activates; slice transitions from wormhole to trumpet |
| t ~ 20-40M | Junk radiation propagating outward; constraints settling |
| t ~ 40M+ | Trumpet gauge established; QNM ringdown visible in Psi4 |
| t ~ 100M+ | QNM signal decayed by e^(-9); constraint violations dominate |

---

## 4. Expected Constraint Behavior

### Hamiltonian constraint L2 norm

The Hamiltonian constraint violation depends strongly on resolution and
formulation. Typical values from NR codes for a single Schwarzschild puncture:

| Resolution (dx) | Expected Ham L2 | Notes |
|------------------|-----------------|-------|
| 2.0M (very coarse) | 1e-1 to 1e0 | Only a few points per M; borderline stability |
| 1.0M (coarse) | 1e-2 to 1e-1 | Minimum for qualitative results |
| 0.5M | 1e-3 to 1e-2 | Standard quick test |
| 0.25M | 1e-4 to 1e-3 | Good for QNM extraction |
| 0.125M (M/8) | 1e-5 to 1e-4 | Publication quality |
| 0.0625M (M/16) | 1e-6 to 1e-5 | High accuracy |

**Convergence**: With 6th-order FD, the constraint L2 should scale as dx^6
(measured ~6.5 in Lattice convergence tests). With 4th-order FD (GRChombo),
scaling is dx^4.

**Time behavior**: Constraints typically:
1. Start at the discretization floor (set by initial data solver accuracy)
2. Rise during the first ~20-30M as gauge adjusts (junk radiation)
3. Settle to a slowly-growing plateau
4. For CCZ4 with constraint damping (kappa1 > 0), the plateau is bounded
5. For BSSN without constraint damping, constraints grow secularly (CCZ3 fixes this)

**AMR effects**: On AMR meshes with volume-weighted norms, the constraint L2
is dominated by the finest level (most volume-weighted contribution from near
the puncture). Excising the AH interior (r < 1.5 * r_AH) reduces the norm
significantly, as the largest constraint violations occur inside the horizon.

### Momentum constraint L2 norm

Typically 1-2 orders of magnitude smaller than Hamiltonian constraint for
stationary Schwarzschild. Scales with the same convergence order.

| Resolution (dx) | Expected Mom L2 |
|------------------|-----------------|
| 0.5M | 1e-4 to 1e-3 |
| 0.25M | 1e-5 to 1e-4 |
| 0.125M | 1e-6 to 1e-5 |

---

## 5. Apparent Horizon Mass

### M_irr = sqrt(A / (16 pi))

For Schwarzschild, the irreducible mass equals the ADM mass: M_irr = M.

**Expected accuracy at different resolutions** (from AH finder literature;
Thornburg, Class. Quant. Grav. 21, 743, 2004, arXiv:gr-qc/0306056):

| dx (near AH) | Expected |M_irr - M| / M | Notes |
|---------------|--------------------------|-------|
| 1.0M | 5-10% | AH barely resolved |
| 0.5M | 1-5% | Coarse; AH shape errors |
| 0.25M | 0.1-1% | Adequate for diagnostics |
| 0.125M (M/8) | 0.01-0.1% | Good accuracy |
| 0.0625M (M/16) | ~0.01% | Approaching spectral finder accuracy |

**Convergence**: AH mass error scales as dx^p where p depends on the AH finder
and interpolation order. With 6th-order off-grid interpolation (as in Lattice),
p ~ 6 is expected.

**Time behavior**: M_irr should be constant for Schwarzschild (no mass loss or
gain). Any secular drift indicates a bug. Typical noise-level fluctuations:
- At dx = 0.25M: dM_irr/dt ~ 1e-4 M per 100M
- At dx = 0.125M: dM_irr/dt ~ 1e-5 M per 100M

**AH spin**: For non-spinning Schwarzschild, the dimensionless spin should be
chi_BH = J / M^2 < 1e-3 (ideally zero, nonzero only from numerical noise).
Measured via Christodoulou formula: M^2 = M_irr^2 + J^2 / (4 M_irr^2).

---

## 6. Psi4 Mode Ratios

### Why (2,0) dominates for single Schwarzschild

The perturbation from conformally flat Brill-Lindquist initial data is
axisymmetric (no angular momentum, no preferred azimuthal direction).
By symmetry, only m = 0 modes are excited. The dominant mode is therefore
(l=2, m=0), not (l=2, m=2).

The (2,2) mode should be zero by symmetry. In practice, numerical noise
gives a small nonzero (2,2) amplitude:

| Quantity | Expected value | Notes |
|----------|---------------|-------|
| |Psi4(2,0)| / |Psi4(2,2)| | > 100 | Ideally infinite |
| |Psi4(2,2)| | ~ 1e-10 to 1e-6 | Numerical noise floor |
| |Psi4(3,0)| / |Psi4(2,0)| | ~ 0.01-0.1 | Higher l suppressed |
| |Psi4(4,0)| / |Psi4(2,0)| | ~ 0.001-0.01 | Further suppressed |

### Selection rules for single non-spinning BH

For a non-spinning Schwarzschild BH with axisymmetric perturbation:

1. **m = 0 only**: All m != 0 modes should vanish (axisymmetry)
2. **Even l only**: If the perturbation has equatorial symmetry (as Brill-Lindquist
   does for a centered puncture), only even-l modes are excited: l = 2, 4, 6, ...
3. **l = 2 dominant**: The quadrupole (l=2) mode has the smallest w_I (longest
   damping time) and typically the largest initial excitation amplitude

### For binary merger comparison (reference)

In an equal-mass non-spinning binary merger, the dominant mode is (2,2):

| Mode | Relative amplitude | Notes |
|------|--------------------|-------|
| (2,2) | 1.0 (reference) | Dominant quadrupole |
| (3,3) | ~0.04-0.07 | Octupole |
| (4,4) | ~0.01-0.02 | Hexadecapole |
| (2,1) | ~0.01-0.03 | Asymmetric mass suppressed for q=1 |
| (2,0) | ~0.001 | Near zero for equal mass |

---

## 7. QNM Measurement Methodology

### Frequency extraction from Psi4

Standard NR procedure for measuring QNM parameters:

1. **Extract r * Psi4 at fixed radius** (r = 20-100M, multiple radii preferred)
2. **Decompose into spherical harmonic modes** (l,m) via numerical quadrature
3. **Discard early data** (t < 30-50M) to remove junk radiation and initial transient
4. **Measure w_R** from zero crossings of Re(Psi4_{lm}):
   - Find times t_i where Re(Psi4) changes sign
   - Half-period = average(t_{i+1} - t_i)
   - w_R = pi / half-period
5. **Measure w_I** from peak amplitudes:
   - Find local maxima of |Re(Psi4)|
   - Fit log(amplitude) vs time to line: slope = -w_I
6. **Cross-check with Prony method** or matched filtering for multi-mode extraction

### Systematic errors

| Source | Magnitude | Mitigation |
|--------|-----------|------------|
| Extraction radius too small | w_R shifted by ~1/r | Use r >= 30M, extrapolate to infinity |
| Boundary reflections | Spurious signal after t ~ 2*(R_boundary - r_extract) | Large domain, or absorbing BCs |
| Junk radiation contamination | Wrong frequency in early data | Start analysis at t > 40M |
| Finite resolution | w_R error ~ dx^p | Richardson extrapolation from 3 resolutions |
| Higher overtone mixing | Distorts frequency measurement | Wait for overtones to decay (t > 40M) |
| Gauge effects in Psi4 | Small gauge contamination | Use CCE for highest accuracy |

### Recommended Lattice test parameters

| Parameter | Quick test | Production |
|-----------|-----------|------------|
| Grid type | AMR | AMR |
| dx_fine | 0.25-0.5M | M/8 = 0.125M |
| Domain L | 64M | 256M |
| AMR levels | 3-4 | 5-6 |
| T_final | 100M | 200-300M |
| Psi4 radius | 15M | 30M + 50M (two radii) |
| Angular grid | 16x32 | 24x48 |
| l_max | 4 | 6 |
| Expected w_R error | 10-25% | < 5% |
| Expected w_I error | 20-50% | < 15% |

---

## 8. QNM Frequency Cross-Reference Table

Quick-reference table for all modes relevant to Lattice's l_max = 4 extraction.
All values: M * omega, gravitational (s = -2), fundamental mode (n = 0).

| l | M w_R | M w_I | Period T/M | tau/M | Q |
|---|---------|---------|------------|-------|-----|
| 2 | 0.37367 | 0.08896 | 16.82 | 11.24 | 2.10 |
| 3 | 0.59944 | 0.09270 | 10.48 | 10.79 | 3.23 |
| 4 | 0.80918 | 0.09416 | 7.77 | 10.62 | 4.30 |

Notable pattern: Q increases with l. Higher multipoles ring longer relative to
their period, but the absolute damping time tau is nearly the same (~11M) for
all fundamental modes. This means all fundamental QNMs decay on a similar
timescale, but higher l modes oscillate faster.

---

## 9. References

### QNM Frequencies

1. E. W. Leaver, "An analytic representation for the quasi-normal modes of Kerr
   black holes," Proc. R. Soc. A 402, 285 (1985).
   -- Original high-precision continued-fraction computation.

2. E. Berti, V. Cardoso, A. O. Starinets, "Quasinormal modes of black holes and
   black branes," Class. Quant. Grav. 26, 163001 (2009). [arXiv:0905.2975]
   -- Comprehensive review with tables; online data at pages.jh.edu/eberti2/ringdown/

3. K. D. Kokkotas, B. G. Schmidt, "Quasi-Normal Modes of Stars and Black Holes,"
   Liv. Rev. Rel. 2, 2 (1999). [gr-qc/9909058]
   -- Living Review with standard reference tables (Table 2).

4. R. A. Konoplya, "Quasinormal behavior of the D-dimensional Schwarzschild black
   hole and the higher order WKB approach," Phys. Rev. D 68, 024018 (2003).
   [gr-qc/0303052]
   -- 6th-order WKB tables for comparison.

5. L. C. Stein, "qnm: A Python package for calculating Kerr quasinormal modes,"
   J. Open Source Softw. 4, 1683 (2019).
   -- High-precision numerical QNM computation: github.com/duetosymmetry/qnm

### Trumpet Geometry

6. M. Hannam, S. Husa, D. Pollney, B. Brugmann, N. O'Murchadha, "Geometry
   and regularity of moving punctures," Phys. Rev. Lett. 99, 241102 (2007).
   -- First identification of trumpet geometry in moving-puncture evolutions.

7. M. Hannam, S. Husa, F. Ohme, B. Brugmann, N. O'Murchadha, "Wormholes and
   trumpets: Schwarzschild spacetime for the moving-puncture generation,"
   Phys. Rev. D 78, 064020 (2008). [arXiv:0804.0628]
   -- Analytical stationary 1+log foliations; R_0 ~ 1.31M.

8. B. Brugmann, "Schwarzschild black hole as moving puncture in isotropic
   coordinates," Gen. Rel. Grav. 41, 2131 (2009). [arXiv:0904.4418]
   -- Stationary 1+log slices in isotropic coords; lapse at puncture.

9. K. A. Dennison, T. W. Baumgarte, "A simple family of analytical trumpet
   slices of the Schwarzschild spacetime," Phys. Rev. D 90, 044028 (2014).
   [arXiv:1403.5484]
   -- Parameterized trumpet family; R_0 range (0, M].

### NR Code Validation

10. S. Husa, J. A. Gonzalez, M. Hannam, B. Brugmann, U. Sperhake, "Reducing
    eccentricity in black-hole binary evolutions with initial parameters from
    post-Newtonian inspiral," Phys. Rev. D 77, 044037 (2008). [arXiv:0706.0740]
    -- BAM code QNM validation.

11. M. A. Scheel et al., "High-accuracy waveforms for binary black hole inspiral,
    merger, and ringdown," Phys. Rev. D 79, 024003 (2009). [arXiv:0810.1767]
    -- SpEC spectral accuracy QNM results.

12. K. Clough et al., "GRChombo: Numerical Relativity with Adaptive Mesh
    Refinement," Class. Quant. Grav. 32, 245011 (2015). [arXiv:1503.03436]
    -- GRChombo CCZ4 code; direct comparison for Lattice.

13. B. Daszuta et al., "GR-Athena++: Puncture Evolutions on Vertex-centered
    Oct-tree Adaptive Mesh Refinement," ApJS 257, 25 (2021). [arXiv:2101.08289]
    -- Closest architecture comparison (AMR, 6th-order FD, Z4c).

14. F. Loffler et al., "The Einstein Toolkit: A Community Computational
    Infrastructure for Relativistic Astrophysics," Class. Quant. Grav. 29,
    115001 (2012). [arXiv:1111.3344]
    -- Community standard NR code.

### AH Finders

15. J. Thornburg, "A Fast Apparent-Horizon Finder for 3-Dimensional Cartesian
    Grids in Numerical Relativity," Class. Quant. Grav. 21, 743 (2004).
    [arXiv:gr-qc/0306056]
    -- AHFinderDirect; standard AH accuracy reference.

16. BHaHAHA Collaboration, "BHaHAHA: A Fast, Robust Apparent Horizon Finder
    Library for Numerical Relativity," (2025). [arXiv:2505.15912]
    -- Hyperbolic flow AH finder (algorithm used in Lattice).
