# Lattice

A 3D numerical relativity code for simulating black hole spacetimes through
inspiral, merger, and ringdown. Lattice implements the CCZ4 (conformal
covariant Z4) formulation of Einstein's field equations in pure C17 with GPU
acceleration via HIP, targeting both AMD and NVIDIA hardware. The code supports
N-body black hole systems with arbitrary mass, spin, and charge; solves the
initial data constraints directly on the evolution mesh via FAS multigrid; and
extracts gravitational waveforms through Psi4 decomposition and
SpECTRE-compatible CCE worldtube output. Block-structured adaptive mesh
refinement with Berger-Oliger subcycling provides the dynamic resolution
needed to resolve black hole horizons while maintaining a large wave-zone
domain. The long-term goal is the first simulation of 100 interacting black
holes -- the current record is 25 (GRChombo, arXiv:2505.01495, 2025).

## Features

**Evolution**
- CCZ4 formulation with moving puncture gauge (1+log slicing, Gamma-driver shift)
- 6th-order finite differences, 6th-order Kreiss-Oliger dissipation
- RK4 time integration (classic 4-stage or Carpenter-Kennedy low-storage)
- Constraint-preserving boundary conditions (BAM-style, arXiv:1212.2901)
- Einstein-Maxwell coupling for charged black holes (6 additional evolved fields)
- Bowen-York and HiSpID (high-spin isotropic) initial data for arbitrary spin

**Mesh and GPU**
- Block-structured octree AMR with Berger-Oliger subcycling
- GPU-resident evolution: zero host-device data transfer during subcycling
- HIP backend portable across AMD (MI250X/MI300X) and NVIDIA (V100/A100/H100)
- Physics kernels are pure C with `LATTICE_DEVICE` annotations; only the backend file (`backend_hip.cpp`) is C++

**N-Body**
- Up to 32 punctures with arbitrary mass, momentum, spin, and charge
- FAS multigrid constraint solver on the evolution AMR mesh (no interpolation)
- Multi-BH tracker with exclusion-zone lapse-minimum searches, per-BH apparent horizon finding, and merger detection

**Diagnostics**
- Newman-Penrose Psi4 extraction with spin-weighted spherical harmonic decomposition
- Apparent horizon finder (hyperbolic flow, mass/spin/area extraction)
- SpECTRE-compatible CCE worldtube HDF5 output for next-generation detector waveforms
- Hamiltonian and momentum constraint monitoring (volume-weighted on AMR)

## Build

```bash
# CPU (default) -- requires gcc/clang + OpenMP
make
make test

# GPU (HIP) -- requires hipcc (AMD) or nvcc + ROCm HIP headers (NVIDIA)
make BACKEND=gpu
make test

# CCE worldtube output -- requires libhdf5-dev
make HDF5=on

# Debug build with sanitizers
make debug
```

GPU setup on a fresh machine is handled by `setup_rocm.sh`, which auto-detects
the GPU vendor and installs the CUDA toolkit and/or ROCm HIP headers.

No external dependencies beyond a C17 compiler and OpenMP. GPU builds require
HIP headers from ROCm. Optional: `libhdf5-dev` for CCE output.

## Quick Start

```bash
# Single Schwarzschild black hole
./build/lattice --N 64 --L 64 --steps 100 --puncture 1.0,0,0,0

# Binary inspiral (D=10M, T=700M) with AMR and full diagnostics
./build/lattice --amr --N_block 32 --max_level 4 --L 64 \
  --steps 1400 --CFL 0.25 \
  --puncture 0.4824,0,0,5,0,0.0939,0 \
  --puncture 0.4824,0,0,-5,0,-0.0939,0 \
  --psi4 --psi4_radius 20 --ah

# Run the full inspiral benchmark
./build/test_binary_inspiral
```

## Evolved Variables (31 fields)

| Variable | Count | Symbol | Description |
|----------|-------|--------|-------------|
| chi | 1 | chi | Conformal factor |
| h[6] | 6 | h_ij | Conformal metric (symmetric) |
| K | 1 | K | Trace of extrinsic curvature |
| A[6] | 6 | A_ij | Traceless conformal extrinsic curvature |
| Gamma[3] | 3 | Gamma^i | Conformal connection functions |
| Theta | 1 | Theta | CCZ4 constraint scalar |
| lapse | 1 | alpha | Lapse function |
| shift[3] | 3 | beta^i | Shift vector |
| B[3] | 3 | B^i | Gamma-driver auxiliary variable |
| E[3] | 3 | E^i | Conformal electric field (EM) |
| BM[3] | 3 | B^i_mag | Conformal magnetic field (EM) |

25 CCZ4 + 6 Einstein-Maxwell. EM fields enabled with `--em`.

## Verification

**Schwarzschild QNM ringdown.** Evolved a perturbed Schwarzschild black hole on
an AMR mesh and extracted Psi4 at multiple radii. The dominant (l=2, m=2) mode
frequency and damping rate match the analytically known quasi-normal mode
values. Two-radius extraction confirms convergence. See `docs/qnm_ringdown.html`.

**Convergence order.** Three-resolution convergence test measures order 6.5,
consistent with the 6th-order finite difference stencils.

**Binary inspiral D10 benchmark.** Equal-mass non-spinning binary at initial
separation D=10M evolved through T=700M (inspiral, merger, ringdown). Validated
against the Samurai consensus waveform (8 hard tests + 4 advisory). Two-radius
Psi4 extraction (r=70M, r=100M), per-BH position/mass/spin tracking via apparent
horizons.

**Test suite.** 300+ automated tests covering: flat spacetime stability,
single BH evolution, convergence, Bowen-York initial data (33), HiSpID
high-spin data (26), apparent horizons (13), Einstein-Maxwell (15), Psi4
extraction (15), CCE worldtube output (49), constraint-preserving BCs (30),
AMR prolongation (15), checkpoint/restart (14), packed evolution (8), N-body
tracker (40), GPU tracker (9), binary inspiral, and N-body solver smoke tests.

## Novel Contributions

**Equidistribution-optimal AMR refinement ratio.** Standard NR codes halve the
refinement radius at each AMR level (beta = 2). Lattice uses beta = 2^(3/5) =
1.516, derived by equating truncation error at adjacent level boundaries for
6th-order finite differences on 1/r^3 Riemann curvature. This provides 5x
better coverage of the puncture region and fewer wasted coarse blocks. Full
derivation: `docs/amr_refinement_ratio.html`.

**Zero-PCIe GPU-resident subcycling.** The entire Berger-Oliger recursive
subcycle executes on the GPU with no host-device data transfers. Cross-level
ghost exchange, temporal interpolation, restriction, and prolongation all run
as device kernels.

**Initial data solved on evolution mesh.** The FAS multigrid constraint solver
operates directly on the AMR evolution blocks, eliminating interpolation error
and ensuring exact discrete operator consistency. Measured 218,000x improvement
in near-field constraint quality versus post-solve interpolation.

See `novel.md` for the complete list with comparisons to existing codes.

## CLI Reference

### Simulation

| Flag | Default | Description |
|------|---------|-------------|
| `--N` | 32 | Grid points per side (single-grid mode) |
| `--L` | 10.0 | Domain half-size (domain is [-L/2, L/2]^3) |
| `--steps` | 1000 | Evolution steps |
| `--CFL` | 0.25 | CFL factor (dt = CFL * dx) |
| `--sigma` | 0.3 | Kreiss-Oliger dissipation strength |
| `--rk` | `classic` | Time integrator: `classic` (RK4) or `ck45` (low-storage) |
| `--bc` | `cp` | Boundary conditions: `sommerfeld` or `cp` (constraint-preserving) |
| `--output_every` | 0 | Output interval (0 = off) |

### AMR

| Flag | Default | Description |
|------|---------|-------------|
| `--amr` | off | Enable adaptive mesh refinement |
| `--N_block` / `--block-size` | 32 | Interior cells per block side |
| `--max_level` | 6 | Maximum refinement depth |
| `--amr-levels` | max_level | Solver refinement levels for initial data |
| `--chi_refine` | 0.1 | Refinement threshold |
| `--chi_coarsen` | 0.01 | Coarsening threshold |
| `--regrid_every` | 1 | Regrid interval (0 = never) |

### CCZ4 Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--kappa1` | 0.1 | Constraint damping (Theta + Z_i) |
| `--kappa2` | 0.0 | Theta mix in K equation |
| `--kappa3` | 1.0 | Z contribution in Gamma equation |
| `--lapse_coeff` | 2.0 | 1+log slicing coefficient |
| `--lapse_power` | 1.0 | Bona-Masso power |
| `--shift_Gamma_coeff` | 0.75 | Gamma-driver coefficient |
| `--eta` | 1.0 | Gamma-driver damping |
| `--lapse_advec` | 1.0 | Lapse advection coefficient |
| `--shift_advec` | 1.0 | Shift advection coefficient |

### Initial Data

| Flag | Description |
|------|-------------|
| `--puncture M,x,y,z[,Px,Py,Pz[,Sx,Sy,Sz[,Q]]]` | Add puncture (up to 32) |
| `--hispid` | Force HiSpID high-spin initial data |
| `--em` | Enable Einstein-Maxwell coupling (charged BHs) |

### Diagnostics

| Flag | Default | Description |
|------|---------|-------------|
| `--psi4` | off | Psi4 gravitational wave extraction |
| `--psi4_every` | 10 | Extraction interval (steps) |
| `--psi4_radius` | 50.0 | Extraction sphere radius |
| `--psi4_l_max` | 4 | Maximum spherical harmonic l |
| `--ah` | off | Apparent horizon finder |
| `--ah_every` | 100 | AH finder interval (steps) |
| `--cce` | off | CCE worldtube output (requires HDF5) |
| `--cce_radius` | 100.0 | CCE extraction radius |
| `--cce_lmax` | 16 | CCE angular resolution |

### Noise Reduction

| Flag | Default | Description |
|------|---------|-------------|
| `--cako` / `--no_cako` | on | Chi-adjusted Kreiss-Oliger dissipation |
| `--per_field_sigma` / `--no_per_field_sigma` | on | Per-field KO strengths |
| `--ssl` / `--no_ssl` | on | Slow-start lapse |
| `--sigma_gauge` | 0.99 | KO sigma for gauge fields |
| `--sigma_phys` | 0.3 | KO sigma for physical fields |

## Project Structure

```
src/
  core/           grid, fields, parameters, device portability macros
  backend/        CPU (OpenMP) and GPU (HIP) backends
  evolution/      CCZ4 RHS, Maxwell RHS, Kreiss-Oliger dissipation
  geometry/       tensor utilities
  numerics/       finite differences, interpolation, RK4 integrator
  initial_data/   puncture data, Bowen-York, HiSpID, FAS multigrid solver
  diagnostics/    constraints, AH finder, Psi4, BH tracker, CCE worldtube
  boundary/       Sommerfeld and constraint-preserving BCs
  amr/            block/mesh management, ghost exchange, prolongation/restriction
  io/             output and checkpoint/restart
tests/            automated test suite
docs/             architecture reference, QNM validation, AMR derivation
```

## References

| Paper | Topic |
|-------|-------|
| [arXiv:1106.2254](https://arxiv.org/abs/1106.2254) | CCZ4 formulation (Alic et al. 2012) |
| [gr-qc/0511048](https://arxiv.org/abs/gr-qc/0511048) | Moving punctures (Campanelli et al. 2005) |
| [gr-qc/9703066](https://arxiv.org/abs/gr-qc/9703066) | Brandt-Brugmann puncture method |
| [arXiv:1212.2901](https://arxiv.org/abs/1212.2901) | Constraint-preserving BCs (Hilditch et al.) |
| [arXiv:0907.1151](https://arxiv.org/abs/0907.1151) | Einstein-Maxwell 3+1 (Alcubierre et al.) |
| [arXiv:2505.15912](https://arxiv.org/abs/2505.15912) | BHaHAHA hyperbolic AH finder |
| [arXiv:1503.03436](https://arxiv.org/abs/1503.03436) | GRChombo methods paper |
| [gr-qc/0610128](https://arxiv.org/abs/gr-qc/0610128) | BAM gauge calibration (Brugmann et al.) |
| [arXiv:1003.0859](https://arxiv.org/abs/1003.0859) | Position-dependent eta (Muller & Brugmann) |
| [arXiv:2505.01495](https://arxiv.org/abs/2505.01495) | 25-BH cluster simulation (GRChombo) |

## License

MIT
