# Lattice Performance Optimization Research

> Compiled 2026-02-20 from four parallel research agents analyzing parallelization,
> algorithms, kernel-level optimization, and memory/IO strategies.

---

## Top Recommendations (ranked by impact/effort)

| # | Optimization | Speedup | Effort | Notes |
|---|-------------|---------|--------|-------|
| 1 | **6th-order FD stencils** | 3-10x (coarser grids, same accuracy) | Low-Med | Ghost=4 already sufficient. Add to finite_diff.h |
| 2 | **Enable LTO** (`-flto`) | 5-15% CPU | Trivial | One Makefile line. Allows cross-TU inlining |
| 3 | **Fused d1/d2 stencil** | 10-15% CPU, 5-10% GPU | Low | Load 5 points once, return both derivatives |
| 4 | **Checkpoint/restart** | N/A (prevents data loss) | Low | Raw binary + CRC32. ~300 lines. Critical for 40h runs |
| 5 | **Persistent pack** (no per-step alloc) | 10-20% GPU | Med | Eliminates 4 GB/step memcpy |
| 6 | **Selective GPU transfer** (ghost only) | 3-8x transfer phase | Low-Med | Transfer 40 MB instead of 342 MB per sync |
| 7 | **GPU-native ghost exchange** | 2-5x overall on GPU | Med-High | Phase 1 (same-level) is straightforward as GPU kernel |
| 8 | **Fused RHS+CK45 update** | 15-25% CPU, 10-20% GPU | Med | Eliminate rhs buffer round-trip |
| 9 | **Larger AMR blocks** (32^3) | 1.7x (ghost overhead) | Low | Ghost drops from 70% to 45%. Better GPU occupancy |
| 10 | **GPU kernel splitting** | 20-40% GPU | Med-High | Split RHS into 2 stages to double occupancy (15%→30%) |
| 11 | **CCZ3 formulation** | Stability to 10^5 M | Low-Med | Theta=0, kappa1=0. Ref: arXiv:2501.01055 |
| 12 | **3D output** (raw binary + XDMF) | N/A (enables science) | Low | VisIt/ParaView compatible. ~200 lines |
| 13 | **Multi-GPU** (split pack per device) | 1.5-3.5x on 4 GPUs | Med-High | Per-device meshblock_pack |
| 14 | **Fields-only block grids** | 33% memory reduction | Med | AMR blocks only need fields array; RK scratch in pack |
| 15 | **MPI domain decomposition** | Linear with node count | High | 2-3 months. Morton-partitioned blocks across ranks |

---

## 1. Parallelization & Multi-GPU

### GPU ghost exchange is the #1 GPU bottleneck

Current flow per RK stage:
```
GPU: compute RHS         (fast)
GPU→CPU: sync ALL data   (342 MB — slow!)
CPU: 5-phase exchange    (slow)
CPU→GPU: sync ALL back   (342 MB — slow!)
```

Fix 1 (easy): Transfer only ghost slabs, not full buffer. Cuts transfer 5-10x.

Fix 2 (medium): Run Phase 1 (same-level exchange) as a GPU kernel — it's just
device-to-device copies within the pack. Pre-compute a work list on host, map
to device, run a copy kernel.

Fix 3 (hard): All 5 phases on GPU. Phases 2-4 (restriction, coarse_buf fill,
prolongation) are compute-per-point operations that parallelize well.

### Multi-GPU (one node)

Split meshblock_pack across devices. Each GPU gets a subset of blocks with its
own pack. Inter-device ghost exchange via host staging buffers or CUDA-aware MPI.

**Warning from Parthenon-VIBE (arXiv:2509.19701):** 16^3 blocks are at the
lower boundary of GPU efficiency (only 4096 interior points per block). 32^3+
recommended for production GPU runs.

### MPI (multi-node)

Partition Morton-sorted leaf blocks across ranks. Ghost exchange becomes:
- Intra-rank: direct memory copy (same as today)
- Inter-rank: MPI_Isend/Irecv with packed ghost buffers

Subcycling complication: fine blocks needing coarse neighbor data across ranks
require temporal interpolation data to be communicated after each coarse step.

Estimated: 85-90% efficiency at 8 nodes, 70-80% at 64 nodes.

### Not recommended: task-based parallelism

5-15% benefit, 2-4 month effort, breaks the simple bulk-synchronous model.
Subcycling already provides inter-level asynchrony.

---

## 2. Algorithmic Improvements

### 6th-order FD stencils (highest impact)

Ghost width is already 4 — sufficient for 6th-order FD + 8th-order KO.
No memory layout change needed.

For equivalent accuracy, 6th order needs N = N_4th^(2/3):
- N=128 at 4th order → N=40 at 6th order (33x fewer points)
- Realistic speedup after accounting for wider stencil compute: **3-10x**

AthenaK uses 6th-order as their standard NR configuration.

Implementation: add 7-point stencil coefficients to finite_diff.h. Compile-time
or runtime flag (FD_ORDER=4 or FD_ORDER=6).

### CCZ3 formulation

arXiv:2501.01055 shows that setting Theta=0 and kappa1=0 in all CCZ4 equations
gives stable evolutions to 10^5 M. Eliminates momentum-constraint-damping
instabilities seen in standard CCZ4 at late times.

Direct speedup is small (remove 1 field, ~4% memory savings), but enables
much longer stable inspiral runs — the real bottleneck for science.

### Classic RK4 over CK45

Classic RK4 = 4 RHS evaluations. CK45 = 5 RHS evaluations. **1.25x speedup**
by switching from CK45 to classic when memory allows. Already available via
`--rk classic`.

### Paired Explicit RK (P-ERK) — future

Different blocks use different numbers of RK stages based on local stiffness.
Stiff blocks (near punctures) get many stages; non-stiff blocks get few.
2-5x speedup on AMR meshes. High implementation effort.

### Not recommended

- IMEX for gauge terms: too complex for 1.2x gain
- Operator splitting: breaks 4th-order convergence
- Local timestepping (per-block dt): DAG scheduling complexity, 1.5-2x gain

---

## 3. Kernel-Level Optimization

### Fused d1/d2 stencil

`fd_d1` and `fd_d2` load the same 5 stencil points. Fusing them halves the
loads for the ~14 fields needing both derivatives:

```c
static inline void fd_d1_d2(const double *f, int idx, int s, double dx,
                             double *d1, double *d2) {
    double fm2 = f[idx-2*s], fm1 = f[idx-s], f0 = f[idx];
    double fp1 = f[idx+s],   fp2 = f[idx+2*s];
    *d1 = ((1.0/12)*fm2 - (2.0/3)*fm1 + (2.0/3)*fp1 - (1.0/12)*fp2) / dx;
    *d2 = (-(1.0/12)*fm2 + (4.0/3)*fm1 - (5.0/2)*f0 + (4.0/3)*fp1 - (1.0/12)*fp2) / (dx*dx);
}
```

Saves ~210 loads per point. 10-15% CPU speedup.

### LTO (link-time optimization)

Adding `-flto` to CFLAGS allows GCC to inline `ccz4_rhs_point` into the
backend loop. Currently the inner i-loop calls it as a function pointer,
preventing all cross-TU optimization. **Trivial change, 5-15% CPU speedup.**

### Fused RHS + CK45 update

Instead of: RHS writes to `rhs[]` buffer → CK45 reads `rhs[]` buffer, fuse
them so RHS values go directly into the CK45 update:
```c
// Inside ccz4_rhs_point, instead of: rhs[f][idx] = rhs_val
scratch[f][idx] = A_s * scratch[f][idx] + dt * rhs_val;
data[f][idx] += B_s * scratch[f][idx];
```

Eliminates the rhs buffer entirely (saves 25% memory for CK45) and removes
one full memory pass (~500 MB at N=128). **15-25% CPU, 10-20% GPU.**

Sommerfeld BCs need special handling for boundary points.

### GPU occupancy

ccz4_rhs_point uses ~400 registers per thread → ~15% occupancy on A100.
Splitting into two stages (derivatives → algebra+RHS) halves register pressure,
doubling occupancy to ~30%. **20-40% GPU speedup.**

The `memset(&g_local, 0, sizeof(grid_t))` in backend_gpu.c zeroes 872 bytes
per thread unnecessarily. Replace with targeted initialization of the 5 fields
actually used. **Trivial, 2-5% GPU.**

### Mixed precision: NOT recommended

Float32 Christoffel symbols accumulate ~1e-5 roundoff near punctures,
comparable to N=256 truncation error. Would break convergence. FP64 is
non-negotiable for all physics arrays.

### Precomputed inverse metric: NOT recommended

The inverse metric computation is 30 FLOPs (~0.6% of RHS). Storing as 6 extra
fields adds 48 bytes/point of bandwidth for negligible compute savings.
Slightly negative net effect (kernel is bandwidth-limited on GPU).

---

## 4. Memory & I/O

### Checkpoint/restart (critical infrastructure)

**Format:** Raw binary + CRC32 checksum. Zero dependencies.
```
Header: magic, version, N, L, ghost, num_fields, step, time, params
Per block: metadata (level, origin) + field data
Footer: CRC32
```

Checkpoint every 500 steps: 0.2% overhead, max 37 min lost on crash.
Ring buffer of 2 files. ~300 lines of C.

### Persistent pack

Currently the pack is allocated, loaded, and freed every timestep. For static
meshes (no regridding), create once and reuse. Eliminates ~4 GB of memcpy per
step. Invalidate on regrid.

### Fields-only block grids

AMR blocks currently allocate 3 arrays (fields + rhs + scratch) even though
the packed path does all computation in the pack. Allocating only `fields`
saves 66% of per-block memory. For 200 blocks: 1.37 GB saved.

Combined with persistent pack: peak memory drops 33%.

### 3D output

Raw binary data files + XDMF descriptors. VisIt/ParaView load XDMF natively.
Zero dependencies (~200 lines of C for binary writer + XML emitter).

Async I/O thread for high-frequency output (pthread + staging buffer, ~100 extra lines).

### CK45 is already optimal

CK45 uses 3 memory blocks (the minimum for 4th-order stencil PDEs). Cannot
reduce further: stencils read from `U` at neighboring points, so RHS output
must go to a separate buffer. 2N refers to solution registers, not counting
the mandatory RHS scratch.

### Conditional EM allocation

When `--em` is not used, the 6 EM fields waste 19% of memory at N=256.
Compile-time `#define LATTICE_VACUUM` setting NUM_FIELDS=25 saves 2.7 GB.

### Not recommended

- Unified GPU memory: page fault overhead, GCC immaturity
- Perturbation storage (h_ij = delta + epsilon): no savings at FP64
- Ghost width reduction: would break convergence
- Shared sibling coarse_bufs: high effort for ~140 MB savings

---

## Implementation Roadmap

### Immediate (days)

1. `-flto` in Makefile (1 line)
2. Fused `fd_d1_d2()` in finite_diff.h
3. Replace GPU memset with targeted init
4. Checkpoint/restart (raw binary)

### Short-term (1-2 weeks)

5. 6th-order FD stencils (+ 8th-order KO)
6. Selective GPU transfer (ghost slabs only)
7. Persistent pack for AMR
8. 3D output (raw binary + XDMF)

### Medium-term (weeks to months)

9. Fused RHS+CK45 compute-and-accumulate
10. GPU-native ghost exchange (Phase 1 on device)
11. GPU kernel splitting for occupancy
12. CCZ3 formulation mode
13. 32^3 block size option for GPU production
14. Fields-only block grids for AMR

### Long-term (months)

15. Multi-GPU support (per-device packs)
16. MPI domain decomposition
17. P-ERK multi-rate time integration

---

## Key References

### Parallelization
- arXiv:2409.16053 — AthenaK: MPI + Kokkos, 80% scaling to 65K GPUs
- arXiv:2509.19701 — Parthenon-VIBE: block size vs GPU performance
- arXiv:2505.00097 — superB/NRPy: Charm++ task-based NR
- arXiv:2509.21527 — GROMACS NVSHMEM halo exchange

### Algorithms
- arXiv:2501.01055 — CCZ3: stable to 10^5 M with kappa1=0
- arXiv:2409.10383 — AthenaK NR: 6th-order FD, GPU benchmarks
- arXiv:2312.05438 — AMR refinement strategy comparison
- arXiv:2404.01137 — CAKO/CAHD/SSL dissipation techniques
- Carpenter & Kennedy 1994 — CK45 coefficients (NASA TM-109112)
- Vermeire 2019 — Paired Explicit Runge-Kutta

### Kernel optimization
- Williams et al. 2009 — Roofline model (CACM)
- NVIDIA A100 whitepaper — register file, shared memory specs

### Memory/IO
- Ketcheson 2010 — Minimum storage RK methods
- GRChombo wiki — HDF5 checkpoint/visualization
- ADIOS2 docs — high-performance I/O engine
