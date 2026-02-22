# Lattice

3D numerical relativity simulator for black hole spacetimes.

- Pure C17, no external dependencies beyond libomp
- CCZ4 formulation with moving punctures ([arXiv:1106.2254](https://arxiv.org/abs/1106.2254))
- GPU acceleration via OpenMP target offloading -- same physics kernels for CPU and GPU
- Block-structured AMR with Berger-Oliger subcycling
- Einstein-Maxwell coupling for charged black holes
- N-body initial data via FAS multigrid (arbitrary puncture count)
- Apparent horizon finder (hyperbolic flow method)
- Noise reduction from [arXiv:2404.01137](https://arxiv.org/abs/2404.01137) (CAKO, per-field KO, slow-start lapse)

## Quick Start

```bash
# Build (CPU)
make

# Build (GPU, requires GCC 15+)
make BACKEND=gpu

# Single black hole
./build/lattice --N 64 --L 64 --steps 100 --puncture 1.0,0,0,0

# Binary inspiral (AMR)
./build/lattice --amr --N_root 6 --N_block 16 --max_level 5 --L 96 \
  --steps 1000 \
  --puncture 0.4824,0,0,5,0,0.0939,0 \
  --puncture 0.4824,0,0,-5,0,-0.0939,0

# Run all tests
make test
```

## CLI Reference

### Simulation

| Flag | Default | Description |
|------|---------|-------------|
| `--N` | 32 | Grid points per side (interior) |
| `--L` | 10.0 | Physical domain size |
| `--steps` | 1000 | Number of evolution steps |
| `--CFL` | 0.25 | CFL factor (dt = CFL * dx) |
| `--sigma` | 0.3 | Kreiss-Oliger dissipation strength |
| `--output_every` | 0 | Output interval (0 = off) |
| `--rk` | ck45 | Time integrator: `classic` or `ck45` |

### CCZ4 Constraint Damping

| Flag | Default | Description |
|------|---------|-------------|
| `--kappa1` | 0.1 | Theta + Z_i damping coefficient |
| `--kappa2` | 0.0 | Theta mix in K equation |
| `--kappa3` | 1.0 | Z contribution in Gamma equation |
| `--covariant_z4` / `--no_covariant_z4` | on | Covariant Z4 (kappa1 vs kappa1*alpha) |

### Gauge

| Flag | Default | Description |
|------|---------|-------------|
| `--lapse_coeff` | 2.0 | 1+log slicing coefficient c |
| `--lapse_power` | 1.0 | Bona-Masso power p |
| `--shift_Gamma_coeff` | 0.75 | Gamma-driver coefficient F |
| `--eta` | 1.0 | Gamma-driver damping |
| `--lapse_advec` | 0.0 | Lapse advection coefficient |
| `--shift_advec` | 0.0 | Shift advection coefficient |

### Noise Reduction

| Flag | Default | Description |
|------|---------|-------------|
| `--cako` / `--no_cako` | on | Chi-adjusted KO dissipation |
| `--per_field_sigma` / `--no_per_field_sigma` | on | Per-field KO dissipation strengths |
| `--ssl` / `--no_ssl` | on | Slow-start lapse |
| `--cahd` / `--no_cahd` | off | Constraint-adjusted Hamiltonian damping |
| `--sigma_gauge` | 0.99 | KO sigma for gauge fields |
| `--sigma_phys` | 0.3 | KO sigma for physical fields |
| `--cahd_coeff` | 0.15 | CAHD coefficient C |
| `--ssl_h` | 0.6 | SSL Gaussian height h/M |
| `--ssl_sigma_t` | 20.0 | SSL Gaussian width sigma_t/M |
| `--ssl_total_mass` | 1.0 | SSL total puncture mass M |

### AMR

| Flag | Default | Description |
|------|---------|-------------|
| `--amr` | off | Enable adaptive mesh refinement |
| `--N_root` | 4 | Root blocks per side |
| `--N_block` | 32 | Interior cells per block side |
| `--max_level` | 6 | Maximum refinement depth |
| `--chi_refine` | 0.1 | Refinement threshold |
| `--chi_coarsen` | 0.01 | Coarsening threshold |
| `--regrid_every` | 1 | Regrid check interval (0 = never) |

### Einstein-Maxwell

| Flag | Default | Description |
|------|---------|-------------|
| `--em` | off | Enable Einstein-Maxwell coupling |
| `--kappa_em` | 0.1 | EM constraint damping coefficient |

### Apparent Horizon Finder

| Flag | Default | Description |
|------|---------|-------------|
| `--ah` | off | Enable AH finder |
| `--ah_every` | 100 | AH finder interval (steps) |
| `--ah_guess` | M/2 | Initial radius guess |
| `--ah_eta` | 10.0 | AH flow speed |
| `--ah_n_theta` | 16 | AH polar resolution |
| `--ah_n_phi` | 32 | AH azimuthal resolution |
| `--ah_tol` | 1e-6 | AH convergence tolerance |
| `--ah_max_iter` | 500 | AH maximum iterations |

### Initial Data

| Flag | Default | Description |
|------|---------|-------------|
| `--puncture M,x,y,z[,Px,Py,Pz[,Sx,Sy,Sz[,Q]]]` | -- | Add puncture (up to 16) |
| `--hispid` | off | Force HiSpID high-spin initial data |

## Build Targets

| Target | Description |
|--------|-------------|
| `make` | CPU build (-O3 -ffast-math -march=native) |
| `make BACKEND=gpu` | GPU build (OpenMP target, GCC 15+) |
| `make debug` | Debug build (-O0 -g -fsanitize=address,undefined) |
| `make test` | Run all tests |
| `make test-convergence` | 3-resolution convergence verification |
| `make test-amr-mesh` | AMR mesh creation + Morton ordering |
| `make test-amr-ghost` | Ghost exchange + multi-block evolution |
| `make test-amr-prolong` | Prolongation + noise reduction |
| `make test-amr-refine` | Oct-tree refinement + multi-level ghost |
| `make test-subcycle` | Berger-Oliger subcycling validation |
| `make test-bowen-york` | Bowen-York initial data (29 tests) |
| `make test-hispid` | HiSpID high-spin initial data (26 tests) |
| `make test-ah` | Apparent horizon finder (13 tests) |
| `make test-maxwell` | Einstein-Maxwell (15 tests) |
| `make test-inspiral-convergence` | AMR binary inspiral convergence |
| `make clean` | Remove build artifacts |

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

25 CCZ4 + 6 EM. EM fields enabled with `--em`.

## References

| Paper | Topic |
|-------|-------|
| [arXiv:1106.2254](https://arxiv.org/abs/1106.2254) | CCZ4 formulation (Alic et al. 2012) |
| [gr-qc/0511048](https://arxiv.org/abs/gr-qc/0511048) | Moving punctures (Campanelli et al.) |
| [gr-qc/9703066](https://arxiv.org/abs/gr-qc/9703066) | Brandt-Brugmann puncture method |
| [arXiv:2404.01137](https://arxiv.org/abs/2404.01137) | Noise reduction: CAKO, per-field KO, SSL |
| [arXiv:0907.1151](https://arxiv.org/abs/0907.1151) | Einstein-Maxwell 3+1 decomposition |
| [arXiv:2505.15912](https://arxiv.org/abs/2505.15912) | BHaHAHA hyperbolic AH flow |
| [arXiv:2312.05438](https://arxiv.org/abs/2312.05438) | AMR refinement strategy comparison |
| [arXiv:1503.03436](https://arxiv.org/abs/1503.03436) | GRChombo methods paper |
