# MPI Implementation Plan for Lattice

## Summary

~1,350 lines total (780 new + 565 modified). No restructuring needed.
Physics kernels, GPU backend, pack structure, and RK4 integrator untouched.

Our code already follows the AthenaK pattern: blocks are Morton-sorted,
packs are self-contained, ghost exchange separates local vs boundary.
MPI adds a second phase to ghost exchange for off-rank neighbors.

## What Changes

| Component | Lines | Complexity |
|-----------|-------|-----------|
| MPI wrapper (`comm.h/c`) | 280 new | Easy |
| MPI ghost exchange (`ghost_mpi.h/c`) | 500 new | Medium |
| Block partitioning (Morton curve) | 130 modified | Easy |
| Diagnostic allreduce | 100 modified | Easy |
| Regrid coordination (allgather flags) | 100 modified | Medium |
| I/O awareness (rank-0 output) | 100 modified | Easy |

## What Does NOT Change

- `ccz4_rhs.c` / `maxwell_rhs.c` — pure point-wise physics
- `finite_diff.h` — stencil macros
- `backend_cpu.c` / `backend_hip.cpp` — GPU kernels operate on rank-local packs
- `sommerfeld.c` / `constraint_preserving.h` — boundary conditions
- `prolongation.c` / `restriction.c` — block-local operations
- `ah_finder.c` — runs on rank-local data
- `jfnk_solver.c` — initial data, runs once at t=0
- `interpolate.h` / `tensor_utils.h` — pure math

## Communication Approach

**Start with staged host buffers** (works everywhere, zero dependencies).
GPU-aware MPI as optional compile-time upgrade later.

| Approach | Latency | Bandwidth | Complexity | Portability |
|---|---|---|---|---|
| Staged host buffers | ~50 us + memcpy | PCIe-limited | Low | All platforms |
| GPU-aware MPI | ~5-20 us (RDMA) | NVLink/IB | Medium | Needs GPU-aware MPI |
| NCCL/RCCL | ~3-10 us | Optimal collectives | High | NVIDIA/AMD only |

Ghost zone data is small (surface area, not volume). For N=32, 4 ghost cells,
25 fields: one face = 800 KB. Total MPI traffic per step: tens of MB, not GB.

## Block Partitioning

Blocks already Morton-sorted by `mesh_compact()`. Partition is trivial:

```c
void mesh_partition(mesh_t *m, int n_ranks) {
    int n_leaves = mesh_num_leaves(m);
    int base = n_leaves / n_ranks;
    int extra = n_leaves % n_ranks;
    int idx = 0;
    for (int r = 0; r < n_ranks; r++) {
        int count = base + (r < extra ? 1 : 0);
        for (int i = 0; i < count; i++) {
            while (!m->blocks[idx] || !m->blocks[idx]->is_leaf) idx++;
            m->blocks[idx]->rank_owner = r;
            idx++;
        }
    }
}
```

For weighted load balancing, weight each block by `2^level` (subcycling cost).

## Ghost Exchange with MPI

### Current (single-node):
```
ghost_exchange(m):
  for each leaf block b:
    for each of 26 neighbors:
      if neighbor_id >= 0:
        memcpy slab from neighbor->interior to b->ghost_zone
```

### With MPI:
```
ghost_exchange(m):
  // Phase 1: local (unchanged)
  for each local leaf block b:
    for each same-rank neighbor: memcpy (unchanged)

  // Phase 2: MPI (new)
  ghost_exchange_mpi(m):
    pack interior slabs into send_buf[dest_rank]
    MPI_Isend / MPI_Irecv (non-blocking)
    MPI_Waitall
    unpack recv_buf into ghost zones
```

Pack/unpack reuses existing `ghost_range()` index computation.

### GPU-resident path:
```
backend_ghost_exchange_packed(pack):
  hip_ghost_exchange_same_level(...)  // for pack-internal neighbors (existing)
  // MPI for off-pack neighbors (new):
  // Option A (staged): D2H pack -> MPI -> H2D unpack
  // Option B (GPU-aware MPI): direct from device buffer
```

## Diagnostics with MPI

| Diagnostic | Pattern |
|---|---|
| Constraint L2 | Local (sum, vol) -> `MPI_Allreduce(SUM)` -> sqrt(sum/vol) |
| Momentum L2 | Same |
| Min lapse | Local min -> `MPI_Allreduce(MINLOC)` -> broadcast position |
| BH separation | Two successive min-lapse + allreduce |
| Psi4 extraction | Each rank computes local sphere points -> allreduce mode coefficients |
| NaN check | `MPI_Allreduce(LAND)` — logical AND |

## Regrid with MPI

All ranks maintain full mesh metadata (tree structure, block locations).
Only field data is distributed. On regrid:

1. All ranks evaluate refinement criterion on local blocks
2. `MPI_Allgather` flag array (~1 int per block, small)
3. All ranks independently compute same 2:1 enforcement (deterministic)
4. All ranks update tree structure identically
5. Redistribute blocks to rebalance (send/recv field data)

This is the AthenaK approach. Tree metadata is tiny (~100 bytes/block).

## Implementation Order

1. **Phase 0:** MPI wrapper (`comm.h/c`) with `MPI=on` compile flag. When off, all functions are no-ops. Zero impact on existing code.
2. **Phase 1:** Block partitioning. Add `rank_owner` to `block_t`, `mesh_partition()`. Test: N=1 produces identical results.
3. **Phase 2:** Diagnostic allreduce. Test: 2-rank run reports correct global diagnostics.
4. **Phase 3:** MPI ghost exchange (`ghost_mpi.c`). Test: 2-rank flat spacetime + single BH stable.
5. **Phase 4:** Regrid coordination. Test: 2-rank AMR binary inspiral matches single-rank.
6. **Phase 5 (optional):** GPU-aware MPI. Compile flag `GPU_AWARE_MPI=on`.

## Scaling

Near-linear to 8-16 GPUs for typical binary inspiral (hundreds of AMR blocks).
Beyond 16 GPUs, load imbalance from subcycling becomes the bottleneck.
Production NR codes typically run on 4-64 GPUs.

## References

- AthenaK: Grete et al. 2024, arXiv:2409.16053
- AMReX: Zhang et al. 2019, JOSS 4:1370
- CarpetX: Schnetter et al. (uses AMReX MPI layer)
- GRChombo: Andrade et al. 2021, JOSS (uses Chombo MPI)
- Athena++: Stone et al. 2020, ApJS 249:4
