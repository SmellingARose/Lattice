# Performance & Optimization Findings

Compiled from deep analysis of the Lattice codebase, comparison against 6 production NR codes (AthenaK, GR-Athena++, SpECTRE, CarpetX, ExaGRyPE, NRPy+), and mathematical methods research. February 2026.

---

## Quick Wins

### 1. LTO (-flto)
Add `-flto` to CFLAGS_OPT in the Makefile (CPU only; GPU LTO broken in GCC 15). Enables cross-TU inlining of `ccz4_rhs_point` through the function pointer in `backend_cpu.c`. One line change.

**Benefit:** 5-15% CPU speedup.

### 2. `__restrict__` on RHS pointers
The compiler cannot assume `rhs[f]` and `src[FIELD_CHI]` don't alias, so it re-loads from memory after every write. Add `__restrict__` to `ccz4_rhs_point` signature and backend dispatch functions.

**Benefit:** 3-8% (enables more aggressive CSE and loop-invariant code motion). GPU-safe.

### 3. Conditional EM field allocation
The 6 EM fields (E^i, B^i_mag) are always allocated even when `--em` is not set. `NUM_FIELDS=31` is baked into every allocation. Check `em_enabled` in `grid_alloc()` and skip EM fields when disabled.

**Benefit:** 19% memory reduction when EM is off (~970 MB per single grid at N=256, ~4.5 GB for 200-block AMR).

### 4. Move `em_enabled` branch outside inner loop
In `backend_cpu.c:176` and `backend_gpu.c:227`, the EM dispatch check is inside the innermost loop. Hoist it outside, use two separate code paths.

**Benefit:** Marginal on CPU (branch predictor handles it), ~5% on GPU (eliminates warp divergence).

---

## CPU Optimizations

### 5. Fused d1/d2 stencil (HIGH IMPACT)
11 fields need both d1 and d2 per direction (chi, lapse, 6x h_ij, 3x shift). Currently loads the same 7 stencil points twice. A fused `fd_d1_d2()` halves the load count for these fields: 33x3 = 99 redundant load sequences eliminated.

```c
static inline void fd_d1_d2(const double *f, int idx, int s, double dx,
                            double *d1, double *d2) {
    double fm3=f[idx-3*s], fm2=f[idx-2*s], fm1=f[idx-s], f0=f[idx];
    double fp1=f[idx+s], fp2=f[idx+2*s], fp3=f[idx+3*s];
    *d1 = (-1.0/60*fm3 + 3.0/20*fm2 - 3.0/4*fm1
           + 3.0/4*fp1 - 3.0/20*fp2 + 1.0/60*fp3) / dx;
    *d2 = (1.0/90*fm3 - 3.0/20*fm2 + 3.0/2*fm1
           - 49.0/18*f0
           + 3.0/2*fp1 - 3.0/20*fp2 + 1.0/90*fp3) / (dx*dx);
}
```

**Benefit:** 15-20% CPU, 5-10% GPU. ~50 lines changed.

### 6. Symmetric tensor raise
`raise_all_2()` in `tensor_utils.h` computes all 9 entries of A^{ij} but only 6 are unique. Compute diagonal (3) + upper triangle (3) and copy: 54 FMAs down to 36 (33% reduction).

**Benefit:** 2-3%.

### 7. Common subexpression elimination in RHS
`chris.ULL[m][kk][ll]` is used repeatedly across Ricci, Gamma RHS, and A_ij RHS. Products like `lapse * K` and `h_UU[i][j]` contracted with various tensors appear multiple times. The compiler's CSE pass is limited by C aliasing rules and `-ffast-math` interference. Manual extraction of repeated sub-expressions can reduce FLOP count.

NRPy+ (arXiv:2501.14030) uses SymPy's CSE pass on the entire RHS expression tree and reports ~21% instruction reduction. Manual CSE on the obvious cases (Christoffel products, metric contractions) is low effort.

**Benefit:** 10-20% FLOP reduction. Low-medium effort for manual CSE.

### 8. Derivative precomputation for mixed d2
`fd_d2_mixed()` uses a 7x7=49-point stencil (6th-order). If first derivatives are stored in scratch arrays, mixed second derivatives can be computed as 1D stencils on the stored d1 values: 7 loads instead of 49 per mixed pair. With 11 fields needing mixed d2 and 3 mixed direction pairs each, this is significant.

**Benefit:** 30-50% fewer loads for mixed second derivatives. Requires scratch storage for d1 arrays.

### 9. Interior/boundary RHS split for async overlap
Compute interior cells' RHS (no ghost dependency) while ghost exchange runs concurrently. Interior cells vastly outnumber boundary cells. Used by GR-Athena++, SpECTRE, ExaGRyPE.

**Benefit:** 10-20% on CPU for multi-block AMR meshes. Medium effort.

---

## GPU Optimizations

### 10. Persistent pack allocation (HIGH IMPACT)
Lattice rebuilds the entire `meshblock_pack_t` every time step: malloc + field copy + `omp target enter data` + compute + `omp target exit data` + store + free. AthenaK keeps all field data on GPU permanently, only rebuilding after regridding.

**Benefit:** Eliminates two full-buffer host-device transfers per step. 5-10% overhead reduction on CPU, much larger on GPU.

### 11. Device-side ghost exchange (HIGH IMPACT)
`backend_gpu.c:879-910` syncs the entire pack to host (`omp target update from`), runs all 5 ghost exchange phases on CPU, then syncs back (`omp target update to`). For 100 blocks at 64^3: ~1.6 GB round-trip per CK45 stage, 5 stages = ~8 GB/step of PCIe traffic. AthenaK does all ghost exchange on-device.

Phase 0+1 (same-level copy) is trivially parallelizable as a GPU kernel. Phases 2-4 (restriction/prolongation) are more complex but have regular control flow.

**Benefit:** 2-5x GPU speedup. The single largest GPU bottleneck. ~300 lines.

### 12. Eliminate per-thread overhead on GPU
Every GPU thread in `backend_gpu.c:205-221` constructs a `grid_t` on stack (~872 bytes) via `memset` and builds two 31-element pointer arrays (~496 bytes). Total: ~1.4 KB per thread. Grid dimensions are identical for all threads in a block. Pass as kernel parameters instead.

**Benefit:** 15-30% GPU occupancy improvement. Low effort.

### 13. Kernel splitting for occupancy
The CCZ4 RHS kernel uses ~400 registers per thread = ~25% occupancy on A100. Split into Phase A (all derivatives + Christoffels, ~2 KB stack, ~200 registers) and Phase B (Ricci + RHS assembly, uses cached derivatives from scratch buffer, ~200 registers). Each phase achieves ~75-80% occupancy.

**Benefit:** 20-40% GPU speedup. Medium-high effort (~400 lines).

### 14. Optimal block size
Parthenon-VIBE benchmarks (arXiv:2509.19701) show N_block=16 makes GPU slower than 96-core CPU. N_block=32 is minimum for GPU; 64 is optimal for weak scaling. Already a runtime parameter in Lattice.

**Benefit:** 2-30x GPU depending on current block size. Zero code changes.

### 15. Reorder d2 computations for instruction cache
The NR101 blog found that reordering second derivative computations to maximize instruction reuse (sort `d2[min(i,j)][max(i,j)]`) yielded ~20% GPU speedup by fitting within the 32 KB instruction cache.

**Benefit:** 10-20% GPU. Low effort.

---

## Memory Savings

### 16. CK45 is already optimal
CK45 uses 3 memory blocks (fields, rhs, scratch) vs Classic RK4's 4 (adds accum). At N=256, 31 fields: CK45 = 14.9 GB, Classic = 19.9 GB. Already the default.

### 17. Ghost zone overhead is unavoidable
Ghost=4 is required for 6th-order FD + 8th-order KO. No reduction possible. At N_block=16: 4.1x overhead; at N_block=32: 1.95x; at N_block=64: 1.51x.

### 18. No malloc in hot loops
Verified: no malloc/free inside time-stepping loops. Pack creation happens once per step (or once per level per step for AMR). Allocation strategy is sound.

### 19. Page alignment waste is negligible
4096-byte alignment wastes <4 KB per block. Essential for GPU zero-copy mapping.

---

## Mathematical Methods

### 20. Dense output for temporal interpolation (HIGH IMPACT)
Current quartic interpolation at coarse-fine boundaries stores 5 quantities per point (U_n, U_{n+1}, U_{n-1}, F_n, F_{n-1}). RK4 dense output achieves 4th-order temporal accuracy using the 4 stage vectors + pre-step state:

```
y(t_n + theta*h) = y_n + h * [b1(theta)*k1 + b2(theta)*k2 + b3(theta)*k3 + b4(theta)*k4]
b1 = theta - 3*theta^2/2 + 2*theta^3/3
b2 = b3 = theta^2 - 2*theta^3/3
b4 = -theta^2/2 + 2*theta^3/3
```

CarpetX (arXiv:2503.09629) uses this to achieve 4th-order convergence at AMR boundaries with fewer prolongation operations ("prolongates 5/8 as many points"). Simpler than the current fields_old/fields_older/rhs_old/rhs_older scheme.

**Caveat:** Works naturally with classic RK4 (k1-k4 explicit). CK45's 2N storage overwrites stage vectors, so subcycling steps would need to use classic RK4.

**Benefit:** Cleaner code, 4th-order temporal convergence at refinement boundaries.

### 21. CCZ3 formulation
Already in stage3options.html. Removes momentum constraint damping, stable to 10^5 M. One-line change + flag. arXiv:2501.01055.

### 22. Optimized-stability 2N RK methods
Niegemann et al. (2012) constructed 2N methods with 8-14 stages optimized for maximum CFL rather than higher order. 1.5-2x larger stable timestep at the cost of more stages. Net savings if CFL-limited. Swapping in new coefficients is trivial (same `CK_A[s]`, `CK_B[s]` infrastructure).

Ketcheson (2010) found 2S methods achieving 4th order in only 4 stages (vs CK45's 5), saving 20% RHS evaluations. Different update formula but similar memory.

**Benefit:** 20-50% fewer total steps if CFL is the bottleneck.

### 23. P-ERK (Paired Explicit Runge-Kutta)
Different AMR blocks use different RK stage counts based on local stiffness. Coarse blocks (80% of mesh) get fewer stages. Reported 1.5-5x speedup for 4-6 AMR levels (arXiv:2403.05144). Fourth-order P-ERK now exists (arXiv:2408.05470).

**Caveat:** Very high implementation complexity. Requires spectral radius estimation per block. No existing NR implementation. Future work.

### 24. Mixed-order stencils near punctures
Use 4th-order FD (5-point) on finest AMR levels wrapping the puncture (where solution is only C2-C4), 6th-order elsewhere. Both are already implemented in `finite_diff.h` (compile-time `FD_ORDER`). Making it per-block at runtime requires two versions of the RHS or function pointer dispatch.

**Benefit:** 10-20% RHS speedup on fine levels. Medium effort.

### 25. KO dissipation stage reduction
Apply Kreiss-Oliger dissipation at 1 RK stage instead of all 5. Saves ~80% of KO computation. Risk: reduced stability near punctures.

**Benefit:** Few percent overall. Low effort but stability risk.

---

## What Other Codes Do Differently

| Technique | Used by | Our status | Applicable? |
|-----------|---------|------------|-------------|
| Persistent GPU data residency | AthenaK | Rebuild every step | Yes (item 10) |
| Device-side ghost exchange | AthenaK | Host round-trip | Yes (item 11) |
| Dense output for subcycling | CarpetX | Quartic interp | Yes (item 20) |
| Code generation from SymPy | NRPy+ | Hand-written C | Future |
| Task-based parallelism | SpECTRE, GR-Athena++ | Bulk-synchronous | Partial (item 9) |
| Enclave tasking for GPU | ExaGRyPE | Full-step offload | Not worth it |
| Taylor-based AMR interpolation | ExaHyPE | Lagrange tensor-product | Worth investigating |
| Vertex-centered grids | GR-Athena++ | Cell-centered | Not applicable |
| Kokkos portability layer | AthenaK | OpenMP target | Not worth switching |
| Block size 32-64 for GPU | AthenaK, Parthenon-VIBE | N_block configurable | Yes (item 14) |

---

## Not Worth It

| Idea | Why not |
|------|---------|
| Mixed precision (float32) | ~1e-5 roundoff near punctures breaks convergence |
| Precomputed inverse metric | 30 FLOPs saved vs 48 bytes/point extra bandwidth |
| Operator splitting | CCZ4 is tightly coupled; Strang splitting caps at 2nd order |
| Unified GPU memory | Page faults serialize stencil code |
| SBP operators | Very high effort, constraint damping + KO works fine |
| Constraint-preserving BCs | High effort, Sommerfeld sufficient for current goals |
| FCCZ4 / Z4c formulation | Same cost as CCZ4, no meaningful advantage |
| Full shared memory tiling on GPU | 6th-order 3D stencil footprint (14^3 = 22 KB) exceeds shared memory budget; OpenMP target has limited control |
| Loop tiling for cache (CPU) | x-innermost already optimal; ~2-3% gain for high complexity |
| Reducing ghost width | Would break 6th-order FD + 8th-order KO |
| AoS field grouping | Violates SoA invariant; minimal cache benefit (~5%) |

---

## Priority Roadmap

### Tier 1: Do Now (days, 30-50% combined improvement)
1. LTO in Makefile (1 line)
2. `__restrict__` on RHS pointers (5 lines)
3. Fused d1/d2 stencil (~50 lines)
4. Conditional EM allocation (~20 lines)
5. Eliminate per-thread GPU overhead (~30 lines)

### Tier 2: Do Soon (weeks, 20-40% additional)
6. Persistent pack allocation (~100 lines)
7. Manual CSE in ccz4_rhs.c (~50 lines)
8. Symmetric tensor raise (~20 lines)
9. Dense output for subcycling (~200 lines)

### Tier 3: Major GPU Work (weeks-months, 2-5x GPU)
10. Device-side ghost exchange (~300 lines)
11. Kernel splitting for occupancy (~400 lines)
12. Interior/boundary RHS split for async overlap (~150 lines)

### Tier 4: Future
13. P-ERK multirate time integration
14. Code generation from SymPy for full CSE
15. Optimized-stability RK methods

---

## References

- arXiv:2409.10383 -- AthenaK NR (Zhu et al. 2024)
- arXiv:2409.16053 -- AthenaK framework (Stone et al. 2024)
- arXiv:2101.08289 -- GR-Athena++ (Daszuta et al. 2021)
- arXiv:2503.09629 -- CarpetX subcycling with dense output (2025)
- arXiv:2406.11626 -- ExaGRyPE (2024)
- arXiv:2504.15814 -- ExaHyPE higher-order AMR interpolation (2025)
- arXiv:2404.01137 -- NRPy improved moving puncture (Etienne 2024)
- arXiv:2501.14030 -- NRPy GPU code generation (2025)
- arXiv:2505.00097 -- superB/NRPy task-based NR (2025)
- arXiv:2509.19701 -- Parthenon-VIBE block size benchmarks (2025)
- arXiv:2501.01055 -- CCZ3 stable to 10^5 M (Bezares et al. 2025)
- arXiv:2408.05470 -- 4th-order P-ERK (Doehring et al. 2024)
- arXiv:2403.05144 -- P-ERK with AMR (Doehring et al. 2024)
- arXiv:1909.12256 -- Original P-ERK (Vermeire 2019)
- Ketcheson 2010 -- Low-storage RK methods
- Niegemann et al. 2012 -- Optimized-stability 2N RK
- NR101/NR102 blog -- GPU NR optimization (James Brown 2024-2025)
