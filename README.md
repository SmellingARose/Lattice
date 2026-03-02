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
- AMD: ROCm 5.0+ with `hipcc` (`sudo apt install rocm-dev hip-dev`)
- NVIDIA: `nvcc` (CUDA toolkit) + ROCm HIP headers (NOT `hipcc`). Requires `export HIP_PLATFORM=nvidia` and `export CUDA_PATH=/usr`
- Or just run `bash setup_rocm.sh` to auto-detect and install everything

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

### Ubuntu / Debian (GPU — NVIDIA H100/H200/A100/V100/P40)

The GPU backend uses HIP headers from ROCm + NVIDIA's `nvcc` compiler (NOT
`hipcc` on NVIDIA). On AMD, `hipcc` works natively.

**Automated setup:** The `setup_rocm.sh` script auto-detects your GPU vendor
and installs everything:

```bash
git clone https://github.com/SmellingARose/Lattice.git
cd Lattice
bash setup_rocm.sh       # detects NVIDIA/AMD, installs deps, sets env vars
make clean && make BACKEND=gpu
make test
```

**Manual setup:**

```bash
# 1. Install CUDA toolkit + ROCm HIP headers
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
# Install ROCm (for HIP headers — the Makefile uses nvcc directly)
wget -q https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb -O /tmp/amdgpu-install.deb
sudo apt install -y /tmp/amdgpu-install.deb
sudo apt install -y hip-dev rocm-hip-sdk

# 2. Set environment variables (add to ~/.bashrc for persistence)
export HIP_PLATFORM=nvidia
export CUDA_PATH=/usr   # or /usr/local/cuda if using NVIDIA's CUDA installer

# 3. Verify
nvcc --version           # should show CUDA compiler
ls /opt/rocm/include/hip/hip_runtime.h  # HIP headers present

# 4. Build and run
git clone https://github.com/SmellingARose/Lattice.git
cd Lattice
make clean
make BACKEND=gpu         # uses nvcc + HIP headers automatically
make test

# 5. Run binary inspiral (full merger, ~2-3 hours on H200)
nohup build/test_binary_inspiral > inspiral.log 2>&1 &
tail -f inspiral.log
```

**How it works:** On NVIDIA, `make BACKEND=gpu` detects `HIP_PLATFORM=nvidia`
and uses `nvcc` directly with `-x cu` to compile device code. HIP API calls
(hipMalloc, hipMemcpy, etc.) are thin wrappers around CUDA, provided by the
ROCm HIP headers at `/opt/rocm/include`. Host C files are compiled with gcc
as usual -- they never see HIP headers.

### Ubuntu / Debian (GPU -- AMD MI250X/MI300X)

```bash
# ROCm is native for AMD GPUs — hipcc compiles directly
sudo apt install rocm-dev hip-dev
git clone https://github.com/SmellingARose/Lattice.git
cd Lattice
make clean
make BACKEND=gpu    # uses hipcc natively
make test
```

No environment variables needed on AMD -- `hipcc` detects the GPU automatically.

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

### Cloud Quick Setup (copy-paste for NVIDIA GPU)

For a fresh Ubuntu 24.04 cloud instance with an NVIDIA GPU (H100/H200/A100/V100/P40):

```bash
sudo apt update && sudo apt install -y build-essential libomp-dev git
git clone https://github.com/SmellingARose/Lattice.git
cd Lattice
bash setup_rocm.sh                    # auto-detects GPU, installs CUDA + ROCm
source ~/.bashrc                      # pick up HIP_PLATFORM and CUDA_PATH
make clean && make BACKEND=gpu
build/test_binary_inspiral            # full inspiral+merger+ringdown
```

Or manually:

```bash
# 1. System deps
sudo apt update && sudo apt install -y build-essential libomp-dev git nvidia-cuda-toolkit

# 2. ROCm HIP headers (for HIP API portability layer)
wget -q https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb -O /tmp/amdgpu-install.deb
sudo apt install -y /tmp/amdgpu-install.deb
sudo apt install -y hip-dev rocm-hip-sdk

# 3. Environment (add to ~/.bashrc)
export HIP_PLATFORM=nvidia
export CUDA_PATH=/usr

# 4. Build + run
git clone https://github.com/SmellingARose/Lattice.git
cd Lattice
make clean && make BACKEND=gpu
build/test_binary_inspiral
```

Output files (all flush every write, safe if disconnected):
- `inspiral.log` — evolution log (step, time, constraints, separation, lapse)
- `build/inspiral_diagnostics.csv` — full time series for plotting
- `build/inspiral_psi4.csv` — gravitational wave mode coefficients

### GPU Memory Requirements

| Grid config | Leaf blocks | Memory | Suitable GPUs |
|---|---|---|---|
| N=32, MAX_LEVEL=4 | ~176 | ~13 GB | P40 (24 GB), V100 (16 GB), A100, H100/H200 |
| N=64, MAX_LEVEL=3 | ~70 | ~10 GB | P40, V100, A100, H100/H200 |
| N=64, MAX_LEVEL=5 | ~1000+ | ~80 GB | A100 (80 GB), H200 (141 GB), MI300X (192 GB) |

**Verified:** Tesla P40 (24 GB) passed flat spacetime, binary BH with AMR
solver, and 5 evolution steps. Any HPC GPU with 1:2 FP64:FP32 ratio works
(V100, A100, H100, H200, MI250X, MI300X). Consumer GPUs (RTX 4090 etc.)
have 1:32 FP64 ratio and are too slow for production runs.

## Quick Start

```bash
# Single black hole (CPU)
./build/lattice --N 64 --L 64 --steps 100 --puncture 1.0,0,0,0

# Binary inspiral with AMR
./build/lattice --amr --N_block 32 --max_level 4 --L 64 \
  --steps 1400 --CFL 0.25 \
  --puncture 0.4824,0,0,5,0,0.0939,0 \
  --puncture 0.4824,0,0,-5,0,-0.0939,0 \
  --psi4 --psi4_radius 20 --ah

# Run the full inspiral test (T=700M inspiral+merger+ringdown, all diagnostics)
build/test_binary_inspiral
# Output:
#   build/inspiral_diagnostics.csv  — time series (constraints, separation, lapse)
#   build/inspiral_psi4.csv         — gravitational wave mode coefficients
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
| `--max_level` | 6 | Maximum refinement depth |
| `--amr-levels` | max_level | Initial data solver refinement levels |
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
| `--lapse_advec` | 0.0 | Lapse advection coefficient |
| `--shift_advec` | 0.0 | Shift advection coefficient |
| `--covariant_z4` / `--no_covariant_z4` | on | Covariant Z4 (kappa1 vs kappa1*alpha) |

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
| `--psi4_radius` | 50.0 | Extraction sphere radius |
| `--psi4_l_max` | 4 | Maximum spherical harmonic l |
| `--psi4_n_theta` | 32 | Polar resolution |
| `--psi4_n_phi` | 64 | Azimuthal resolution |
| `--ah` | off | Enable apparent horizon finder |
| `--ah_every` | 100 | AH finder interval (steps) |
| `--ah_guess` | M/2 | Initial radius guess |
| `--ah_eta` | 10.0 | AH flow speed |
| `--ah_n_theta` | 16 | AH polar resolution |
| `--ah_n_phi` | 32 | AH azimuthal resolution |
| `--ah_tol` | 1e-6 | AH convergence tolerance |
| `--ah_max_iter` | 500 | AH max iterations |
| `--em` | off | Enable Einstein-Maxwell coupling |
| `--cce` | off | Enable CCE worldtube output (requires HDF5) |
| `--cce_every` | 1 | CCE extraction interval |
| `--cce_radius` | 100.0 | CCE extraction radius |
| `--cce_lmax` | 16 | CCE angular resolution |

### Einstein-Maxwell

| Flag | Default | Description |
|------|---------|-------------|
| `--em` | off | Enable Einstein-Maxwell coupling |
| `--kappa_em` | 0.1 | EM constraint damping coefficient |

### Noise Reduction

| Flag | Default | Description |
|------|---------|-------------|
| `--cako` / `--no_cako` | on | Chi-adjusted KO dissipation |
| `--per_field_sigma` / `--no_per_field_sigma` | on | Per-field KO strengths |
| `--ssl` / `--no_ssl` | on | Slow-start lapse |
| `--cahd` / `--no_cahd` | off | Constraint-adjusted Hamiltonian damping |
| `--sigma_gauge` | 0.99 | KO sigma for gauge fields |
| `--sigma_phys` | 0.3 | KO sigma for physical fields |

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
