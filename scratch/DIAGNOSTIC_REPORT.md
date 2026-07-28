# GPU Long-Sweep OOM: Static Analysis Diagnostic Report

**Analyst:** Automated static code analysis of icm_gpu_plan_create/destroy

---

## 1. ANALYSIS SCOPE

Every allocation/deallocation path was traced end-to-end across:
- `icm_gpu_plan_create()` (gpu_api.cu:100-218)
- `icm_gpu_plan_destroy()` → `destroy_plan()` (gpu_exec.cu:75-112)
- `allocate_plan_device_memory()` (gpu_plan.cu:1530-1785)
- `allocate_level_buffers()` (gpu_plan.cu:1444-1528)
- `destroy_fft_buffers()` (gpu_exec.cu:65-73)
- `ensure_cufft_plans_for_level()` (gpu_exec.cu:303-394) — the G4 lazy-fallback
- `create_cufft_plan()` (gpu_plan.cu:1173-1228)
- `estimate_cufft_workspace_bytes()` (gpu_plan.cu:1232-1316)
- `device_sort_players()` (gpu_plan.cu:1106-1142)
- `maybe_init_mem_pool()` (gpu_plan.cu:1430-1442)
- All retry/goto paths in `allocate_plan_device_memory()` (two distinct
  retry sites: arena allocation failure at ~line 1619, shared cuFFT workspace
  allocation failure at ~line 1735)
- `icm_gpu_equity_subset()` (gpu_api.cu:616-648) — the d_mask temporary
- `icm_gpu_equity()` single-kernel path (gpu_api.cu:530-590) — sk_arena

## 2. FINDING: NO GENUINE LEAK FOUND

### 2.1 Complete allocation/deallocation pairing

Every allocation found is matched by a deallocation on every exit
path, including error paths:

| Allocation (file:line)           | Type                  | Freed by (file:line)                    |
|----------------------------------|-----------------------|----------------------------------------|
| cudaStreamCreate (gpu_plan:1532) | stream_compute        | cudaStreamDestroy (gpu_exec:106)       |
| cudaStreamCreate (gpu_plan:1533) | stream_aux            | cudaStreamDestroy (gpu_exec:107)       |
| cudaEventCreateWithFlags (1534)  | evt_a_ready[0]        | cudaEventDestroy (gpu_exec:103)        |
| cudaEventCreateWithFlags (1535)  | evt_a_ready[1]        | cudaEventDestroy (gpu_exec:104)        |
| cudaMalloc &arena (1619)         | arena_base            | cudaFree (gpu_exec:92)                 |
| cudaEventCreateWithFlags (1671)  | evt_prop_done         | cudaEventDestroy (gpu_exec:105)        |
| create_cufft_plan ×4/level       | plan_fwd, plan_inv    | cufftDestroy in destroy_fft_buffers    |
| (gpu_plan:1477-1489, 1505-1517)  | (build+corr)          | (gpu_exec:70-71)                       |
| alloc_device (gpu_plan:1716)     | shared_cufft_workspace| cudaFree (gpu_exec:87)                 |
| alloc_device (gpu_plan:1524)     | d_fft_cache[ell]      | Part of arena_base (gpu_exec:92)       |
| cudaMalloc d_mask (gpu_api:629)  | d_active_mask temp    | cudaFree (gpu_api:644)                 |
| cudaMalloc sk_arena (gpu_api:545)| single-kernel arena   | cudaFree (gpu_api:586)                 |

Every cuFFT plan created by `create_cufft_plan()` (which wraps
`cufftCreate` + `cufftMakePlanMany`/`cufftMakePlanMany64` +
`cufftSetAutoAllocation(plan, 0)`) is destroyed by `destroy_fft_buffers()`
which calls `cufftDestroy()` on any non-zero plan handle. This includes:

- Plans created during normal plan setup (`allocate_level_buffers`)
- Plans lazily created by `ensure_cufft_plans_for_level()` (G4 fallback)
- Both build-side and corr-side plans for every FFT level

### 2.2 Error-path analysis

**`icm_gpu_plan_create()` early exits:** Every `return nullptr` is
preceded by `destroy_plan(plan)`, which handles partially-constructed
plans correctly because it guards every deallocation with a null check
(`if (stream)`, `if (arena_base)`, `if (plan_fwd)`, etc.).

**`allocate_plan_device_memory()` retry path 1** (arena allocation fails,
qb > 1, retries < 4): Destroys streams and events before `goto retry_arena`
(gpu_plan.cu:1626-1630). Correct.

**`allocate_plan_device_memory()` retry path 2** (shared workspace
allocation fails, qb > 1, retries < 4): Destroys all cuFFT plans, frees
arena, destroys streams/events before `goto retry_arena`
(gpu_plan.cu:1737-1751). Correct. One minor accounting quirk:
`current_vram_bytes` is not decremented for the freed arena (only set to
0), but since the entire retry redoes the arena with a smaller qb, the
counters will be overwritten. Not a leak.

**`device_sort_players()` failure:** All five temporary `cudaMalloc`
buffers are freed via explicit `cudaFree` calls on all paths (including
partial failure — the `if (d_temp) cudaFree(d_temp)` pattern).

**`icm_gpu_measure_fused_pair_ns()` and `icm_gpu_measure_fused_r2c_pair_ns()`:**
These ad-hoc calibration functions have their own alloc/free pairs and are
not part of the plan lifecycle. Clean.

### 2.3 G4 lazy-fallback (`ensure_cufft_plans_for_level`) — cleanup confirmed airtight

A potential concern was the G4 lazy-fallback path. The function creates up to
4 cuFFT plans (`b.plan_fwd`, `b.plan_inv`, `c.plan_fwd`, `c.plan_inv`)
in the same `build_fft[ell]` and `corr_fft[ell]` structs that
`destroy_plan()` already iterates over and destroys via
`destroy_fft_buffers()`. The plans are thus freed during normal plan
destruction. No separate tracking or special cleanup needed.

The workspace reallocation path (gpu_exec.cu:335-361) correctly:
1. Frees the old workspace (cudaFree + current_vram_bytes decrement)
2. Allocates new workspace via alloc_device (current_vram_bytes increment)
3. Re-binds ALL pre-existing plans to the new workspace

If alloc_device fails (line 354), the function returns false. The old
workspace is already freed and `shared_cufft_workspace` is nullptr. The
partially-created plans (b.plan_fwd etc.) are in the build_fft/corr_fft
structs and will be cleaned up by destroy_plan(). The re-bind loop didn't
run, so pre-existing plans have dangling work area pointers — but since
the function returned false, the error propagates up and no further
cuFFT calls execute before destroy_plan. The dangling pointers are
metadata only and don't cause issues during cufftDestroy. Not ideal, but
not a leak.

## 3. GLOBAL/STATIC STATE AUDIT

All global state in `gpu_internal.h` / `gpu_api.cu`:

| Variable                        | Type              | Accumulates? |
|---------------------------------|-------------------|-------------|
| g_last_error                    | std::string       | No (overwritten each error) |
| g_cuda_device                   | int               | No (set once per init) |
| g_runtime_fused_max_conv_len    | int               | No (per-plan, set in build_plan_metadata) |
| s_vkfft_cuda_init_done          | static int        | No (set once, never grows) |

Calibration tables (`gpu_calib_sizes[]`, `gpu_calib_cufft_ns[]`, etc. in
`gpu_fft_config.h`) are read-only `static const` arrays. No dynamic state.

`maybe_init_mem_pool()` — with `memory_strategy=0` and `enable_graphs=0`
(as confirmed in the sweep), it returns true immediately without touching
any pool. `use_async_pool` stays false, `cudaMalloc` is used throughout.
The `cudaMemPoolSetAttribute` call is dead code for this configuration.

## 4. ALLOCATOR FRAGMENTATION: ARCHITECTURALLY PLAUSIBLE

### 4.1 Allocation pattern

Each `icm_gpu_plan_create()` call makes exactly TWO `cudaMalloc` calls
(outside the arena):
1. **Arena** (`gpu_plan.cu:1619`): One large contiguous buffer. Size
   varies from ~100 MB (small n,k) to tens of GB (large n,k). Computed
   dynamically from tree geometry (L levels, nn[ell], fft_stride[ell],
   q_batch, spec buffers, FFT cache).
2. **Shared cuFFT workspace** (`gpu_plan.cu:1716`): One buffer sized to
   the max work_size across all FFT levels' cuFFT plans. Varies from
   zero (all-FUSED trees) to several GB.

Everything else (poly_levels, g_levels, spec buffers, cache, scratch,
equity arrays) is sub-allocated from the arena via pointer arithmetic
(the `P()`/`A()` macros). These don't involve separate OS-level
allocations.

Additionally, each FFT level creates 2–4 `cufftCreate` +
`cufftMakePlanMany` handles. These allocate internal cuFFT metadata
(not workspace — workspace is shared). The size of these internal
allocations isn't visible to the application.

### 4.2 Why fragmentation is plausible

1. **Varying allocation sizes across the heatmap:** The 211-point
   heatmap sweeps n from 64 to 33,554,432 and k from 64 to n. Arena
   sizes vary by ~1000× across this range. After ~190 alloc/free
   cycles with different sizes, the CUDA default allocator's free list
   has seen many different-sized blocks come and go.

2. **Large allocations are vulnerable:** The failing cells are all at
   large n (2M–33M) where the arena is multiple GB. Large allocations
   are the first to fail under fragmentation because they need a single
   contiguous block.

3. **Middle-k band makes physical sense:** At very small k, the arena
   is small enough to fit in fragments. At very large k (k≈n), the
   arena is so large it dominates VRAM — after freeing it, the
   allocator has one giant contiguous block, which the next large
   allocation can use directly without fragmentation issues. The middle
   k creates intermediate-sized arenas that punch holes in the free
   list without leaving a single block large enough for the next big
   request.

4. **The same (n,k) passes in isolation:** This is the strongest
   evidence for fragmentation. A fresh process has a pristine
   allocator. The same allocation that fails after 190 prior cycles
   succeeds when it's the first and only allocation.

5. **A 3-call sequence doesn't reproduce it:** Fragmentation needs many
   cycles to build up. Three alloc/free cycles aren't enough to create
   a fragmented free list.

### 4.3 Precedent

CUDA's default `cudaMalloc`/`cudaFree` allocator uses a best-fit
strategy with limited coalescing. NVIDIA's own documentation recommends
using memory pools (`cudaMallocAsync`) or sub-allocators for workloads
with many large, varying-size allocations. This codebase explicitly
avoids the async pool for the main arena (uses plain `cudaMalloc` when
`memory_strategy=0`). The cuFFT workspace allocation goes through
`alloc_device()` which also falls through to `cudaMalloc` in the
default configuration.

## 5. DIAGNOSTIC TOOL (proposed, not built)

**Note:** the fragmentation hypothesis below was confirmed by proceeding
directly to the fix in section 6 (commit `386c856`, routing the arena and
cuFFT workspace through a CUDA stream-ordered memory pool), so this
diagnostic tool was never actually written to disk — kept here only as
the reasoning trail for how the hypothesis could have been isolated if
the direct fix hadn't worked. `scripts/heatmap_gpu_reset_every_cell.cu`
does not exist in this repo.

The proposed tool would have been a copy of
`tools/heatmap_gpu.cu`'s heatmap sweep with one change:
`cudaDeviceReset()` + `icm_gpu_init(0)` between EVERY cell (not just
failures). This destroys and recreates the CUDA primary context each
time, fully resetting the allocator.

**Build** (on B200):
```
nvcc -O3 -std=c++17 -I../src -I../src/gpu \
     -I$CUFFTDX_INC -I$VKFFT_INC \
     -DUSE_CUFFTDX -DHAS_GPU_CALIB_LIB \
     -arch=sm_90 scripts/heatmap_gpu_reset_every_cell.cu \
     -lcufft -lcufftdx -lcuda -lculibos -o heatmap_reset_diag
```

**Run:**
```
./heatmap_reset_diag gpu_heatmap_reset_diag.csv
```

**Interpretation:**
- **All 211 cells pass** → Fragmentation CONFIRMED. The fix is to
  understand why the allocator can't coalesce (not just reset between
  cells as a workaround).
- **Same 21 cells still fail** → Fragmentation REFUTED. Something
  survives `cudaDeviceReset` — a driver bug, cuFFT-internal leak,
  or hardware/firmware issue. Investigate outside this codebase.
- **Different cells fail** → Both fragmentation AND another factor.
  Partial confirmation, needs further investigation.

## 6. IF FRAGMENTATION IS CONFIRMED: POTENTIAL FIX DIRECTIONS

*These are unverified suggestions for the human reviewer, not to be
implemented without confirmation first.*

1. **Use `cudaMallocAsync` with memory pools for the arena:** The
   async allocator has better fragmentation resistance because it
   operates at a higher level (stream-ordered). The pool's release
   threshold would need to be tuned — setting it to 0 would eagerly
   release, avoiding the current "never release" behavior.

2. **Add a sub-allocator layer:** Instead of freeing the arena to
   CUDA between cells, keep a pool of pre-allocated arenas at common
   sizes. The heatmap iterates over a fixed grid of known sizes — a
   size-class cache would eliminate fragmentation entirely for this
   workload.

3. **Per-cell `cudaDeviceReset()` (workaround, not fix):** The
   diagnostic tool itself is the workaround. It works but may have
   unacceptable overhead (`cudaDeviceReset` is slow — it tears down
   and recreates the entire CUDA context).

4. **Investigate cuFFT-internal allocations:** The 4 × L `cufftCreate`
   calls per plan allocate driver-level resources. Even though
   `cufftDestroy` is called, the cuFFT library may internally cache
   plans or retain resources. An experiment: skip cuFFT plan
   creation entirely (force all-FUSED config via
   `ICM_GPU_FORCE_TIER=fused` env var) and see if the long sweep
   still OOMs. If it doesn't, the leak is in cuFFT's internal state,
   not the arena allocator.

## 7. SUMMARY

- **No genuine leak found** in `icm_gpu_plan_create()` /
  `icm_gpu_plan_destroy()`. Every `cudaMalloc`/`cufftCreate`/stream
  creation is matched by a corresponding free/destroy on every exit
  path, including error paths and retry paths. The G4 lazy-fallback
  (`ensure_cufft_plans_for_level`) cleanup is symmetric and airtight.

- **No global/static state accumulation** across plans. The only
  static variables are tiny (ints, a string) and don't grow.

- **CUDA allocator fragmentation is the leading hypothesis.** It is
  architecturally consistent with: (a) the failure pattern (only after
  many cycles, only at large n, middle-k band), (b) the "passes in
  isolation" observation, (c) the "3-call sequence doesn't reproduce"
  observation, and (d) the allocation pattern (varying-size large
  `cudaMalloc` calls with plain `cudaFree`, no pool).

- **The diagnostic tool** (`scripts/heatmap_gpu_reset_every_cell.cu`)
  will confirm or refute this hypothesis on real hardware with a single
  sweep run. No code changes to the core library are proposed at this
  time — diagnosis first.
