# Lattice

3D numerical relativity simulator implementing the CCZ4 formulation of
Einstein's field equations for evolving black hole spacetimes through
inspiral, merger, and ringdown.

- Pure C17 physics kernels with `LATTICE_DEVICE` annotations for GPU portability
- CPU backend: OpenMP threads
- GPU backend: HIP (AMD ROCm + NVIDIA CUDA)
- Block-structured AMR with Berger-Oliger subcycling
- 6th-order finite differences, constraint-preserving boundary conditions
- Einstein-Maxwell coupling for charged black holes
- N-body initial data via FAS multigrid (arbitrary puncture count)
- Apparent horizon finder, Psi4 waveform extraction, CCE worldtube output

## Setup Guide

### Prerequisites

**CPU build (any platform):**
- C17 compiler (`gcc` on Linux, `clang` on macOS)
- OpenMP (`libomp` / `libgomp`)
- `make`

**GPU build (HPC):**
- HIP compiler (`hipcc`) in PATH
- AMD: ROCm 5.0+ (`sudo apt install rocm-dev` on Ubuntu)
- NVIDIA: HIP CUDA backend via ROCm or direct ROCm-on-NVIDIA installation

**Optional:**
- `libhdf5-dev` + `pkg-config` for CCE worldtube HDF5 output (`make HDF5=on`)

### Ubuntu / Debian (CPU)

```bash
sudo apt update
sudo apt install build-essential libomp-dev
git clone https://github.com/SmellingARose/Lattice.git
cd Lattice
make
make test
```

### Ubuntu / Debian (GPU — NVIDIA H100/H200/A100/V100)

```bash
# 1. Install ROCm (provides hipcc with NVIDIA backend)
#    Follow: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
#    Or use the quick install script:
sudo apt install wget gnupg2
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/latest/ ubuntu main' | \
  sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dev hip-runtime-nvidia hip-dev

# 2. Verify hipcc is available
hipcc --version

# 3. Build and run
git clone https://github.com/SmellingARose/Lattice.git
cd Lattice
make clean
make BACKEND=gpu
make test

# 4. Run binary inspiral (full merger, ~2-3 hours on H200)
nohup build/test_binary_inspiral > inspiral.log 2>&1 &
tail -f inspiral.log
```

**Note:** If ROCm's NVIDIA backend is not available on your cloud provider,
an alternative is to install AMD's HIP-on-CUDA directly. The key requirement
is that `hipcc` can compile `.cpp` files targeting your GPU. Verify with:

```bash
hipcc --version
hipconfig --platform   # should show "nvidia" for NVIDIA GPUs
```

### Ubuntu / Debian (GPU — AMD MI250X/MI300X)

```bash
# ROCm is native for AMD GPUs
sudo apt install rocm-dev
git clone https://github.com/SmellingARose/Lattice.git
cd Lattice
make clean
make BACKEND=gpu
make test
```

### macOS (CPU only)

```bash
# Install Xcode command line tools + OpenMP
xcode-select --install
brew install libomp

git clone https://github.com/SmellingARose/Lattice.git
cd Lattice
make
make test
```

### Cloud Quick Setup (copy-paste)

For a fresh Ubuntu cloud instance with an NVIDIA GPU:

```bash
# All-in-one: install deps, clone, build GPU, run inspiral
sudo apt update && sudo apt install -y build-essential libomp-dev git

# Install ROCm for NVIDIA HIP backend (see ROCm docs for your distro)
# ... or if hipcc is already available in your cloud image, skip this step

git clone https://github.com/SmellingARose/Lattice.git
cd Lattice
make clean && make BACKEND=gpu
nohup build/test_binary_inspiral > inspiral.log 2>&1 &
tail -f inspiral.log
```

Output files (all flush every write, safe if disconnected):
- `inspiral.log` — evolution log (step, time, constraints, separation, lapse)
- `build/inspiral_diagnostics.csv` — full time series for plotting
- `build/inspiral_psi4.csv` — gravitational wave mode coefficients

### GPU Memory Requirements

| Grid config | Leaf blocks | Memory | Suitable GPUs |
|---|---|---|---|
| N=32, MAX_LEVEL=4 | ~176 | ~13 GB | V100 (16 GB), A100 (40/80 GB), H100/H200 |
| N=64, MAX_LEVEL=3 | ~70 | ~10 GB | V100, A100, H100/H200 |
| N=64, MAX_LEVEL=5 | ~1000+ | ~80 GB | A100 (80 GB), H200 (141 GB), MI300X (192 GB) |

**Important:** Only HPC-class GPUs with 1:2 FP64:FP32 ratio are useful
(V100, A100, H100, H200, MI250X, MI300X). Consumer GPUs (RTX 4090 etc.)
have 1:32 FP64 ratio — negligible double-precision throughput.

## Quick Start

```bash
# Single black hole (CPU)
./build/lattice --N 64 --L 64 --steps 100 --puncture 1.0,0,0,0

# Binary inspiral with AMR
./build/lattice --amr --N_block 32 --max-level 4 --L 64 \
  --steps 1400 --CFL 0.25 \
  --puncture 0.4824,0,0,5,0,0.0939,0 \
  --puncture 0.4824,0,0,-5,0,-0.0939,0 \
  --psi4 --psi4_radius 20 --ah

# Run the full inspiral test (all diagnostics included)
build/test_binary_inspiral
```

## Build Targets

| Target | Description |
|--------|-------------|
| `make` | CPU build (`-O3 -ffast-math -march=native -flto`) |
| `make BACKEND=gpu` | GPU build (HIP — AMD + NVIDIA) |
| `make HDF5=on` | Enable CCE worldtube HDF5 output |
| `make debug` | Debug build (`-O0 -g -fsanitize=address,undefined`) |
| `make test` | Run all tests |
| `make test-convergence` | 3-resolution convergence (order 6.5) |
| `make test-bowen-york` | Bowen-York initial data (33 tests) |
| `make test-hispid` | HiSpID high-spin initial data (26 tests) |
| `make test-ah` | Apparent horizon finder (13 tests) |
| `make test-maxwell` | Einstein-Maxwell (15 tests) |
| `make test-psi4` | Psi4 gravitational wave extraction (15 tests) |
| `make test-cp-bc` | Constraint-preserving BCs (30 tests) |
| `make test-inspiral` | Binary inspiral full system validation |
| `make HDF5=on test-cce` | CCE worldtube HDF5 output (49 tests) |
| `make clean` | Remove build artifacts |

## CLI Reference

### Simulation

| Flag | Default | Description |
|------|---------|-------------|
| `--N` | 32 | Grid points per side (single-grid mode) |
| `--L` | 10.0 | Physical domain half-size (domain is [-L/2, L/2]^3) |
| `--steps` | 1000 | Number of evolution steps |
| `--CFL` | 0.25 | CFL factor (dt = CFL * dx) |
| `--sigma` | 0.3 | Kreiss-Oliger dissipation strength |
| `--output_every` | 0 | Output interval (0 = off) |
| `--rk` | `classic` | Time integrator: `classic` (RK4) or `ck45` (low-storage) |
| `--bc` | `cp` | Boundary conditions: `sommerfeld` or `cp` (constraint-preserving) |

### AMR

| Flag | Default | Description |
|------|---------|-------------|
| `--amr` | off | Enable adaptive mesh refinement |
| `--N_block` / `--block-size` | 32 | Interior cells per block side |
| `--max-level` | 6 | Maximum refinement depth |
| `--amr-levels` | max-level | Initial data solver refinement levels |
| `--chi_refine` | 0.1 | Refinement threshold |
| `--chi_coarsen` | 0.01 | Coarsening threshold |
| `--regrid_every` | 1 | Regrid check interval (0 = never) |

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

### Initial Data

| Flag | Description |
|------|-------------|
| `--puncture M,x,y,z[,Px,Py,Pz[,Sx,Sy,Sz[,Q]]]` | Add puncture (up to 32) |
| `--hispid` | Force HiSpID high-spin initial data |

### Diagnostics

| Flag | Default | Description |
|------|---------|-------------|
| `--psi4` | off | Enable Psi4 gravitational wave extraction |
| `--psi4_every` | 10 | Extraction interval (steps) |
| `--psi4_radius` | 20.0 | Extraction sphere radius |
| `--psi4_l_max` | 4 | Maximum spherical harmonic l |
| `--psi4_n_theta` | 16 | Polar resolution |
| `--psi4_n_phi` | 32 | Azimuthal resolution |
| `--ah` | off | Enable apparent horizon finder |
| `--ah_every` | 100 | AH finder interval (steps) |
| `--ah_n_theta` | 16 | AH polar resolution |
| `--ah_n_phi` | 32 | AH azimuthal resolution |
| `--em` | off | Enable Einstein-Maxwell coupling |
| `--cce` | off | Enable CCE worldtube output (requires HDF5) |

### Einstein-Maxwell

| Flag | Default | Description |
|------|---------|-------------|
| `--em` | off | Enable Einstein-Maxwell coupling |
| `--kappa_em` | 0.1 | EM constraint damping coefficient |

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
| [arXiv:1212.2901](https://arxiv.org/abs/1212.2901) | Constraint-preserving BCs (Hilditch et al.) |
| [arXiv:0907.1151](https://arxiv.org/abs/0907.1151) | Einstein-Maxwell 3+1 (Alcubierre et al.) |
| [arXiv:2505.15912](https://arxiv.org/abs/2505.15912) | BHaHAHA hyperbolic AH flow |
| [arXiv:1503.03436](https://arxiv.org/abs/1503.03436) | GRChombo methods paper |
| [gr-qc/0610128](https://arxiv.org/abs/gr-qc/0610128) | BAM calibration (Brügmann et al.) |
