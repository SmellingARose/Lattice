# QNM Publication Test — Step 50 NaN Root Cause Analysis

## Summary

The QNM publication test (`test_qnm_publication.c`) crashes at step 50
(t=50M) on H100 GPU regardless of dissipation settings (CAKO on/off) or
ghost data fixes. Two agents independently traced the same root cause.

## Root Cause: `gpu_sync_all_to_host` Corrupts Temporal Interpolation

### The Bug

With `psi4_every=1`, `gpu_sync_all_to_host(m)` is called **every step** for
Psi4 extraction. This triggers a D→H→D round-trip that destroys the
`fields_old` / `data` distinction needed for temporal interpolation:

1. **End of step N:** `gpu_sync_all_to_host(m)` syncs device→host, then
   runs `ghost_exchange(m)` + `mesh_restrict_to_parents(m)` on host,
   **modifying** host block data.

2. **Start of step N+1:** `gpu_ensure_level_packs` calls
   `meshblock_pack_sync_from_blocks` (copies modified host data → pack)
   then `backend_map_pack` (uploads pack → device).

3. **`backend_save_old_packed`** copies `data → fields_old` on device.
   But `data` was just uploaded from the modified host blocks.

4. **Result:** `fields_old = data = same modified data`. Temporal
   interpolation `(1-frac) * fields_old + frac * data` returns the same
   value regardless of `frac`. **Cross-level ghost fill is broken.**

### Why Step 50

The crash isn't specific to step 50 — the error accumulates from step 1.
By ~step 45-49, the corrupted temporal interpolation at AMR level boundaries
has injected enough error into the fine-level ghost zones to destabilize the
evolution. The diagnostic only prints every 10 steps, so we see healthy step
40 and NaN step 50.

### Evidence

- 4 runs with different dissipation/ghost settings ALL crash at step 50
- Constraints decrease normally (3.6e-2 → 1.8e-2) — the bulk evolution works
- Lapse jumps to 1.0 at crash (initial value, suggesting data corruption)
- The crash location is always at/near AMR level boundaries

## Secondary Issue: Stale RHS in Ghost Zones

`backend_compute_rhs_packed` only computes RHS for interior cells `[ghost, ghost+N)`.
Ghost zone RHS values are never zeroed or computed. But `backend_rk4_stage_packed`
applies RHS to ALL cells:

```
data[i] = scratch[i] + c * dt * rhs[i]   // i includes ghost cells
```

For ghost cells, `rhs[i]` contains stale values from the previous RK4 step or
uninitialized memory. This corrupts ghost zone field values every RK4 stage.
`backend_ghost_exchange_packed` fixes same-level-facing ghost zones, but ghost
zones facing non-leaf blocks (`neighbor_table = -1`) remain corrupted.

## Fixes

### Fix 1: Don't sync every step (immediate)

Set `psi4_every = 10` (or higher) so `gpu_sync_all_to_host` is called less
frequently. This reduces the temporal interpolation corruption but doesn't
eliminate it.

### Fix 2: Re-save fields_old after H→D re-upload (proper)

In `gpu_ensure_level_packs`, after `backend_map_pack` re-uploads data to
device, call `backend_save_old_packed` to snapshot the uploaded data as
`fields_old`. Then `subcycle_level_gpu`'s `backend_save_old_packed` at the
start of each level correctly saves the pre-step state.

Actually, this would make `fields_old = data` at the start of the step, which
is the same problem. The real fix: **don't modify host data between D→H sync
and H→D re-upload.** Either:
- Skip `ghost_exchange` + `restrict_to_parents` in `gpu_sync_all_to_host`
  (defer to subcycle_level_gpu's post-subcycle restriction), OR
- Don't re-upload modified data — let the device keep its own evolved state

### Fix 3: Zero RHS buffer (complementary)

Add `backend_zero_packed(pack, PACK_BUF_RHS)` in `step_level_gpu` before
Stage 1. This ensures ghost zone RHS values are 0 instead of garbage.
`rk4_stage` then gives `data[ghost] = scratch[ghost] + 0 = scratch[ghost]`
(preserved). Fixes the secondary issue.

### Fix 4: Separate Psi4 sync from evolution state (best)

Instead of `gpu_sync_all_to_host` (which syncs ALL packs + runs host-side
ghost exchange), add a lightweight sync that only copies the blocks needed
for Psi4 extraction without modifying the evolution state:

```c
if (do_psi4) {
    // Sync only data D→H, no ghost_exchange, no restriction
    for (int L = 0; L <= m->max_level; L++) {
        if (!m->level_packs[L]) continue;
        backend_activate_pack(m->level_packs[L]);
        backend_unmap_pack_sync(m->level_packs[L]);
        meshblock_pack_sync_to_blocks(m->level_packs[L], m->blocks);
    }
    // Extract Psi4 on host (reads from mesh blocks)
    psi4_extract(ws1, m);
    psi4_extract(ws2, m);
    // DON'T re-upload — device state is unchanged, still valid
}
```

Then `gpu_ensure_level_packs` at the start of the next step detects that
device data is still valid and skips the re-upload. The ghost_exchange +
restriction only happen in the post-subcycle path inside `subcycle_level_gpu`.

## References

- `src/numerics/rk4.c` — gpu_sync_all_to_host (line 766), gpu_ensure_level_packs (line 693), subcycle_level_gpu (line 816)
- `src/backend/backend_hip.cpp` — backend_map_pack re-map path (line 177), backend_save_old_packed (line 333), hip_rk4_stage (line 940)
- `tests/test_qnm_publication.c` — psi4_every=1 (line 193), gpu_sync_all_to_host call (line 244)
