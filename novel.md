# Novel Contributions

Things Lattice does differently from existing NR codes (BAM, GRChombo,
Einstein Toolkit, SpECTRE, AthenaK, Dendro-GR, NRPy+, etc.).

Only genuinely novel items are listed here — techniques that no other
published NR code implements, or where Lattice's approach is architecturally
distinct. Standard techniques (moving punctures, Berger-Oliger subcycling,
block-structured AMR, etc.) are not listed even if our implementation is good.

---

## 1. Equidistribution-Optimal AMR Refinement Ratio (β = 1.516)

**What every other code does:** Halve the refinement radius at each level
(β = 2). This is universal in BAM, GRChombo, Einstein Toolkit, AthenaK, and
every AMR NR code. Nobody derives it — it's just "the obvious choice."

**What we do:** Use β = 2^(3/5) ≈ 1.516 instead of 2.

**Why it's better:** β = 2 is only optimal for resolving a *constant* field.
Near a black hole, Riemann curvature falls off as 1/r³. The truncation error
of a p-th order FD scheme on a 1/r^α field at a level boundary is:

    ε ~ dx^p / r^(α+p+1)

Setting equal error at adjacent boundaries and solving:

    β = 2^(p/(α+p+1))

For 6th-order FD (p=6) on 1/r³ curvature (α=3): β = 2^(6/10) = 2^(3/5) ≈ 1.516.

**Concrete effect:** Compared to standard halving for D10 inspiral:
- Finest level covers 1.94M (vs 0.375M) — 5x better puncture coverage
- Coarsest level covers 124M (vs 384M) — fewer wasted blocks
- Per-puncture mass scaling: r_k = 4·M·β^k (heavier BHs get larger regions)

Full derivation: [docs/amr_refinement_ratio.html](docs/amr_refinement_ratio.html)
How it works: [docs/amr_refinement_howto.html](docs/amr_refinement_howto.html)
Code: `src/initial_data/relaxation_amr.c`, `refine_mesh_near_punctures()`

---

## 2. Zero-PCIe GPU-Resident Berger-Oliger Subcycling

**What other codes do:** AthenaK/Parthenon keeps data device-resident and
launches kernels from the host with minimal PCIe data copies — architecturally
similar. CarpetX/AMReX offloads individual phases but maintains host-side
control flow. The key distinction is what happens for *cross-level* operations
during subcycling: ghost exchange, temporal interpolation, restriction.

**What we do:** The entire recursive Berger-Oliger subcycle runs on GPU with
zero host-device data copies. `subcycle_level_gpu()` / `step_level_gpu()` in
`rk4.c` execute all RK stages, ghost exchanges, and cross-level temporal
interpolation entirely via device kernels. A single `gpu_sync_all_to_host()`
at the end of the global step brings data back for diagnostics.

**What's distinct from AthenaK:** Our device-side 5-phase ghost exchange
includes cross-level temporal interpolation (`hip_cross_level_ghost_fill`
kernel), restriction, and prolongation — all as GPU kernels with pre-uploaded
neighbor maps. AthenaK's Parthenon does device-resident data but the
cross-level ghost fill protocol is less documented. Our explicit guarantee
is: zero `hipMemcpy(D2H)` or `hipMemcpy(H2D)` between `gpu_sync_all_to_host`
calls.

**Why it matters:** For the D10 inspiral with 11 AMR levels, one global step
has 2048 fine substeps. Eliminating PCIe round-trips during cross-level
operations avoids latency stalling in deep refinement hierarchies.

**Key enablers:**
- Device-side 5-phase ghost exchange (7 kernel launches, zero host copies)
- `hip_cross_level_ghost_fill` kernel reads coarser pack with temporal interpolation
- Persistent per-pack device memory (`device_handle` on `meshblock_pack_t`)
- `backend_activate_pack()` switches between level packs without memcpy
- Cross-level neighbor maps uploaded once, reused across all substeps

Code: `src/numerics/rk4.c` (step_level_gpu, subcycle_level_gpu),
`src/backend/backend_hip.cpp` (device-side ghost kernels)

---

## 3. Initial Data Solved Directly on Evolution AMR Mesh

**What other codes do:** Most codes create a separate solver grid, solve the
constraints there, then interpolate back onto the evolution mesh. GRChombo uses
GRTresna (standalone solver). Einstein Toolkit uses TwoPunctures (spectral,
2-body only). AthenaK uses external TwoPunctures. Athena++ (Tomida & Stone 2023)
demonstrated multigrid on the evolution mesh for MHD, and Alic et al.
(arXiv:0912.2920) proposed the same idea for GR. The concept is known but
rarely implemented in production NR codes.

**What we do:** The FAS multigrid constraint solver operates directly on the
evolution AMR blocks at t=0. Solver fields are stored in the evolution grid's
pre-allocated array slots (22 of 100). After the solve, the solution is already
at the exact discrete points where evolution will happen — no interpolation.

**Why it matters:** Measured 218,000x better near-field constraint quality
vs the old interpolation approach on refined meshes. The solver uses the same
finite-difference operators as the evolution code, ensuring exact discrete
operator consistency. Zero interpolation error by construction. While the idea
has been proposed before, we believe this is the first GPU-accelerated
implementation for NR constraint solving on the actual evolution AMR hierarchy.

Code: `src/initial_data/bowen_york.c` (set_bowen_york_mesh),
`src/initial_data/relaxation_amr.c` (AMR composite multigrid)

---

## 4. N-Body Initial Data with GPU-Accelerated AMR Multigrid (N > 2)

**What other codes have done:** Lousto et al. (arXiv:1004.1353, 2010)
demonstrated N-puncture multigrid constraint solving for up to 100 BHs on
CPU using the Cactus framework. TwoPunctures (spectral) is limited to N=2.
GRChombo evolved a 25-BH cluster (arXiv:2505.01495) but used superposition
of conformally flat data — not a constraint-satisfying solve.

**What we do:** FAS multigrid with Newton-Gauss-Seidel smoothing, 8-color
GPU-compatible checkerboard, on the actual AMR evolution mesh. Supports
arbitrary N (tested up to 5, designed for 32). Both 1-field (Bowen-York)
and 4-field coupled (HiSpID high-spin) solvers. GPU-accelerated via the
same device-side ghost exchange as evolution.

**What's distinct:** Lousto et al. proved N-body multigrid works but used
CPU-only uniform grids with the Cactus framework. Our solver combines three
elements no prior code has together: (1) GPU acceleration with 8-color
checkerboard compatible with warp-based execution, (2) solving directly on
the evolution AMR mesh (#3), and (3) both Bowen-York and HiSpID (4-field
coupled) systems. The combination enables constraint-satisfying N-body
initial data at GPU speed on the exact mesh used for evolution.

Code: `src/initial_data/relaxation.c` (base FAS solver),
`src/initial_data/relaxation_amr.c` (AMR composite multigrid),
`src/initial_data/mg_smooth_point.h` (GPU-annotated smoothing kernels)

---

## 5. Pure C Physics with Single-File GPU Backend (No Framework)

**What every other code does:** Depends on a framework:
- GRChombo → Chombo/AMReX (C++ class hierarchy)
- Einstein Toolkit → Cactus (thorns, Fortran+C, huge dependency tree)
- AthenaK → Kokkos + Parthenon (C++ templates)
- Dendro-GR → SympyGR code generation (Python → C/CUDA)
- SpECTRE → Charm++ (task-based C++)

**What we do:** All physics is pure C17. The `LATTICE_DEVICE` macro expands
to `__host__ __device__` when compiled by hipcc/nvcc, and to nothing on CPU.
The same C function compiles for both targets. Only one file is C++:
`backend_hip.cpp` (~900 lines). No framework, no code generation, no templates.

**Why it matters:**
- Zero external dependencies (no MPI, no framework, no pkg-config)
- Build time: ~5 sec CPU, ~30 sec GPU
- Physics equations map 1:1 to C — readable by physicists, not just C++ experts
- True HIP portability: same code on AMD (MI250X/MI300X) and NVIDIA (V100/A100/H100)
- Two-phase compilation: gcc for host, hipcc/nvcc for device — physics files never
  see HIP headers

Code: `src/core/device.h` (LATTICE_DEVICE macro),
`src/backend/backend_hip.cpp` (single GPU backend file)

---

## 6. Einstein-Maxwell Coupled to CCZ4 on GPU

**What other codes do:** Bozzola & Paschalidis (arXiv:2104.06978) evolved
charged BH binaries using the Einstein Toolkit — CPU-only, BSSN formulation.
GRaM-X (arXiv:2510.18968) implements resistive MHD coupled to Z4c on GPU
via the Einstein Toolkit/AMReX, which is a related but different physical
system (fluid EM, not vacuum Maxwell). No other code couples vacuum Maxwell equations to CCZ4 on GPU.

**What we do:** 6 additional evolved fields (E^i, B^i) with conformal Maxwell
evolution, constraint damping, and EM stress-energy coupling to CCZ4. Charged
puncture initial data via the same N-body multigrid solver. Compile-time
dispatch: `make EM=on` sets `COMPILED_NUM_FIELDS=31` so hot loops have
constant bounds; `make EM=off` (default) uses 25 fields with zero EM overhead.

**Why it matters:** Enables charged black hole simulations on GPU with the
full CCZ4+Maxwell system. The distinction from GRaM-X: we solve vacuum
Einstein-Maxwell (no fluid, pure E^i/B^i evolution with constraint damping),
they solve resistive MHD (fluid + EM). The compile-time dispatch means
pure-vacuum runs pay zero cost for the EM capability.

Code: `src/evolution/maxwell_rhs.c`, `src/evolution/ccz4_rhs.c` (EM source terms)

---

## 7. Fused d1/d2 Finite Difference Stencil

**What every other code does:** Compute first and second derivatives
separately, loading the stencil points twice.

**What we do:** `fd_d1_d2()` loads the 7-point stencil once and computes
both d1 and d2 from the same data. For fields needing both derivatives
(chi, lapse, h_ij, shift — most of them), this eliminates ~40% of redundant
memory loads in the RHS kernel.

**Why it matters:** The CCZ4 RHS is memory-bandwidth-bound on GPU. Halving
the stencil loads for the dominant operation (mixed d1+d2) directly improves
throughput. This is a micro-optimization but it applies to every grid point
at every RK stage.

Code: `src/numerics/finite_diff.h`, `fd_d1_d2()`

---

## 8. CCZ4 RHS Kernel Decomposition for GPU Register Pressure

**What every other code does:** One monolithic RHS function (GRChombo, BAM,
Einstein Toolkit). The compiler sees all ~200 local variables at once and
struggles with register allocation on GPU (leading to register spilling to
slow local memory).

**What we do:** Split `ccz4_rhs_point` into 5 `static inline` sub-functions
with typed output structs (`ccz4_fields_t`, `ccz4_derivs_t`, `ccz4_geom_t`,
`ccz4_covd_t`). Each struct scopes variable lifetimes — e.g., the 81-entry
`d2_h` array is freed after Ricci computation, letting the compiler reuse
those registers. All 5 functions inline into a single kernel (zero call
overhead), but the scoped lifetimes give the register allocator explicit
lifetime information.

**Why it matters:** GPU kernels have limited registers per thread (255 on
NVIDIA). The monolithic approach spills to local memory; scoped lifetimes
reduce spilling. Also works around a GCC nvptx codegen bug with `double**`
across function boundaries.

Code: `src/evolution/ccz4_rhs.c` (ccz4_load_and_differentiate,
ccz4_compute_geometry, ccz4_compute_covariant, ccz4_compute_evolution,
ccz4_compute_gauge)

---

## 9. Combined Feature Set in One Codebase

No other single NR code has all of these:

| Feature | Lattice | GRChombo | Einstein Toolkit | AthenaK | BAM |
|---------|---------|----------|-----------------|---------|-----|
| CCZ4 evolution | yes | yes | Z4c | Z4c | yes |
| GPU acceleration | HIP | no | AMReX | Kokkos | no |
| Einstein-Maxwell | yes | no | BSSN only | no | no |
| N-body solver (N>2) | yes | no | CPU only† | no | no |
| Solve on evolution mesh | yes | no | no | no | no |
| AH finder | yes | yes | yes | post‡ | yes |
| Psi4 extraction | yes | yes | yes | post‡ | yes |
| CCE worldtube output | yes | no | yes§ | post‡ | no |
| Constraint-preserving BCs | yes | no | no | no | yes |
| Zero external dependencies | yes | no | no | no | no |

† Lousto et al. (arXiv:1004.1353) demonstrated N-body multigrid in the Cactus
framework (CPU-only). ‡ AthenaK provides Psi4, AH, and CCE via external
post-processing tools rather than inline during evolution. § Einstein Toolkit
has `CCE_Export` thorn for worldtube output.

GRChombo has the broadest physics (scalar fields, modified gravity) but no
GPU. AthenaK has the best GPU performance but diagnostics require external
post-processing. Einstein Toolkit has the most thorns but requires the Cactus
framework. Lattice is the only code that combines GPU-accelerated CCZ4 +
Maxwell + N-body solver + AMR + integrated inline diagnostics in a single
dependency-free C codebase.

---

## What We Do NOT Claim as Novel

These are good implementations of known techniques, not firsts:

- 6th-order FD stencils (AthenaK also uses 6th-order)
- Block-structured AMR with subcycling (standard)
- GPU acceleration for NR (AthenaK, CarpetX, Dendro-GR all do this)
- Meshblock packing for GPU (originated by AthenaK/Parthenon)
- Constraint-preserving BCs (originated by BAM, arXiv:1212.2901)
- Position-dependent eta (originated by Muller & Brugmann, arXiv:1003.0859)
- Hyperbolic AH finder (BHaHAHA, arXiv:2505.15912)
- HiSpID high-spin initial data (published technique)
- Moving puncture gauge (standard since 2005)
- Volume-weighted constraint norms (standard in AMR codes)
