# GPU Debug Plan — E1_GPU_PREP (Wave W0b)

**Board:** SPRINT_FINISH_LINE_DAG.md  
**Target GPU:** B200, ~178.4 GiB VRAM (191,505,498,112 bytes), 148 SMs  
**Local analysis only** — no GPU available; findings for later E2 (B200 session) and E3 (tier ablation).

---

## 1. OOM at n=1,048,576, k=n — VRAM Accounting

### 1.1 Tree Geometry (B=128 assumed; planner may select B≈128–144)

| Parameter | Value |
|-----------|-------|
| n | 1,048,576 |
| k (k_pad) | 1,048,576 (power of 2, no pad needed) |
| B (estimated) | 128 |
| nblocks | 8,192 |
| N_tree | 8,192 (power of 2) |
| L (levels) | 14 |

**Per-level geometry (nn = node count, psz = poly size):**

| ell | nn | psz | nn×psz | conv_build | below_sat | tier | fft_n | cn |
|-----|-------|------|-----------|------------|-----------|------|-------|------|
| 0 | 8192 | 256 | 2,097,152 | — | — | — | — | — |
| 1 | 4096 | 512 | 2,097,152 | 256 | 1 | FUSED | 256 | 129 |
| 2 | 2048 | 1024 | 2,097,152 | 512 | 1 | FUSED | 512 | 257 |
| 3 | 1024 | 2048 | 2,097,152 | 1024 | 1 | FUSED | 1024 | 513 |
| 4 | 512 | 4096 | 2,097,152 | 2048 | 1 | FUSED | 2048 | 1025 |
| 5 | 256 | 8192 | 2,097,152 | 4096 | 1 | FUSED | 4096 | 2049 |
| 6 | 128 | 16384 | 2,097,152 | 8192 | 1 | FUSED | 8192 | 4097 |
| 7 | 64 | 32768 | 2,097,152 | 16384 | 1 | CUFFT | 16384 | 8193 |
| 8 | 32 | 65536 | 2,097,152 | 32768 | 1 | CUFFT | 32768 | 16385 |
| 9 | 16 | 131072 | 2,097,152 | 65536 | 1 | CUFFT | 65536 | 32769 |
| 10 | 8 | 262144 | 2,097,152 | 131072 | 1 | CUFFT | 131072 | 65537 |
| 11 | 4 | 524288 | 2,097,152 | 262144 | 1 | CUFFT | 262144 | 131073 |
| 12 | 2 | 1048576 | 2,097,152 | 524288 | 1 | CUFFT | 524288 | 262145 |
| 13 | 1 | 1048576 | 1,048,576 | 2097151 | 0 | CUFFT | 2097152 | 1048577 |

Notes:
- Ell 1–6: conv ≤ GPU_FUSED_MAX_CONV_LEN(8192) → FUSED (cuFFTDx). Fused kernels do NOT use shared spec buffers — they use cuFFTDx's own SMEM-based execution.
- Ell 7–13: conv > 8192 → CUFFT. Schoolbook would cost ~(conv² × 0.000168 ns) ≈ 45–4400 ms per parent, while cuFFT costs ~0.005–10 µs. cuFFT dominates at all these levels.

### 1.2 q_batch Computation (from `gpu_api.cu`)

The planner computes `per_q_bytes` to determine how many q-points can be batched.
The CURRENT (pre-patch) code counts ONLY these three items:

```
per_q_bytes (current code, qb=1) =
  poly+g arrays:  2 × Σ(nn×psz) × 8  =  432.0 MB   (NOTE: actual alloc uses
                                                     fft_stride ≥ psz → ~480 MB)
  block_prods:     N×(B+1)×8            =    8.1 MB
  a_sorted:        2×n×8                =   16.0 MB
  TOTAL budgeted per_q                  ≈  456.2 MB
```

Items actually allocated per-q but NOT budgeted (the bug):

```
  a/inner/block_prods qbatch:            ~24.9 MB
  spec buffers (4, max over levels;
    ell 13 cn=1,048,577 dominates):     ~100.7 MB   (not 50.4 — mc_sm/mb_si are
                                                     2×pb·cn and cb·cn at ell 13)
  fft caches:      Σ(cb×cn×16), 6 lvls  ~100.7 MB
  fft_stride inflation of poly+g:        ~48.0 MB   (ell 12/13 stride = 2,097,152)
  UNBUDGETED per_q                      ≈  274 MB
```

(There is NO separate gather/scatter "fft scratch" allocation in the current
code — the fft_stride layout eliminated it. The patch still budgets
max(cb×fft_n×8) ≈ 33.5 MB as conservative headroom.)

```
qb = ⌊0.60 × 191,505,498,112 / 456,216,192⌋ ≈ 240   (budget fraction is 0.60,
                                                      Q_BATCH_MAX = 256)
```

### 1.3 Actual VRAM Allocation (qb=256)

All items below are from `allocate_plan_device_memory()` in `gpu_plan.cu`:

| # | Allocation | Formula | Size (MB) | Notes |
|---|-----------|---------|-----------|-------|
| 1 | S_sorted | n×8 | 8.0 | base |
| 2 | sort_perm | n×4 | 4.0 | base |
| 3 | inv_perm | n×4 | 4.0 | base |
| 4 | a_sorted[2] | 2×n×8 | 16.0 | base |
| 5 | graph_logv/scale | tiny | 0.0 | base |
| 6 | inner_sorted | n×8 | 8.0 | base |
| 7 | equity | n×8 | 8.0 | base |
| 8 | payout | k×8 | 8.0 | base |
| 9 | block_prods | N×(B+1)×8 | 8.1 | base |
| 10 | **a_qbatch[256]** | 256×n×8 | **2,048.0** | qb>1 extra — **NOT in per_q** |
| 11 | **inner_qbatch** | 256×n×8 | **2,048.0** | qb>1 extra — **NOT in per_q** |
| 12 | **block_prods_qbatch** | 256×8.1 MB | **2,073.6** | qb>1 extra — **NOT in per_q** |
| 13 | qb_a_ptrs/weights/inv_vs | tiny | 0.0 | qb>1 extra |
| 14 | poly+g arrays | qb×432 MB | 110,592.0 | qb-scaled in arena |
| 15 | FFT caches (×6) | qb×100.7 MB | 25,804.8 | qb-scaled, 6 cuFFT levels |
| 16 | spec: mb_si | qb×16.8 MB | 4,300.8 | shared build spec_in |
| 17 | spec: mb_sm | qb×8.4 MB | 2,150.4 | shared build spec_mid |
| 18 | spec: mc_si | qb×8.4 MB | 2,150.4 | shared corr spec_in |
| 19 | spec: mc_sm | qb×16.8 MB | 4,300.8 | shared corr spec_mid |
| 20 | fft scratch | qb×16 MB | 4,096.0 | gather/scatter temporary |
| | **Arena subtotal** | | **~159,625 MB** | **≈ 155.9 GB** |
| 21 | cuFFT shared workspace | cufftGetSize(max plan) | **2,000–8,000** | allocated separately |
| 22 | cuFFT plan descriptors | 28 plans total | ~100–500 | internal cuFFT state |
| | **TOTAL VRAM** | | **~162–168 GB** | against 191.5 GB phys |

### 1.4 Root Cause Identified

**The VRAM budget math is plausible on its face (~162–168 GB needed vs ~191 GB available). However, three factors combine to cause the OOM:**

1. **Missing qb>1 overhead in `per_q_bytes`**: The per_q_bytes computation in `gpu_api.cu` omits `a_qbatch` (n×8 per q), `inner_qbatch` (n×8 per q), and `block_prods_qbatch` — together **~24 MB per q-point**. For qb=256 this is **~6 GB** of unaccounted VRAM. The `per_q_bytes` estimate of 623 MB should be ~647 MB.

2. **cuFFT shared workspace is unbudgeted**: The shared cuFFT workspace (item 21) is allocated AFTER the arena and is NOT included in the per_q_bytes budget calculation. For a 1M-point R2C plan with batch=1024, `cufftGetSize` can return **2–8 GB**. This comes on top of the already-committed arena.

3. **Contiguous allocation failure**: `cudaMalloc(156 GB)` for the arena requires a single contiguous virtual address range. On a GPU with 191 GB total, the driver may not be able to satisfy a 156 GB contiguous allocation after reserving memory for context, page tables, and other overhead. Even if total free VRAM is sufficient, fragmentation or alignment constraints can cause `cudaMalloc` to fail.

**Combined worst case**: arena 156 GB + workspace 8 GB + plan overhead 0.5 GB + driver 3 GB = **167.5 GB**. With only ~191 GB physical and the arena needing a single contiguous block, the probability of OOM is high.

> **W0R review correction (independent re-derivation):** the table above
> understates the arena. With qb=240 (the value the 0.60-budget code actually
> picks), the true qb-scaled footprint is ≈730 MB/q — poly+g at fft_stride
> ≈503 MB (not 432), spec buffers ≈100.7 MB (not 50.4; ell 13 cn=1,048,577
> dominates), caches 100.7 MB, qbatch arrays 24.9 MB — giving an arena of
> ≈175 GB, plus a multi-GB cuFFT workspace, against 191.5 GB physical. The
> allocation simply exceeds free VRAM; no contiguity/fragmentation hypothesis
> is needed. Unbudgeted total ≈274 MB/q × 240 ≈ 66 GB (larger than the ~48 GB
> figure above). Conclusion unchanged: OOM is real and the accounting fix
> (patched per_q ≈738 MB/q → qb≈150, arena ≈112 GB) prevents it.

### 1.5 Proposed Fix — `patches/oom_fix.patch`

Three independent changes, each addressing a layer of the problem:

**(A)** Add missing qb>1 allocations to `per_q_bytes` in `gpu_api.cu`:
- `a_qbatch`: n×8 per q-point
- `inner_qbatch`: n×8 per q-point
- `block_prods_qbatch`: N×(B+1)×8 per q-point

**(B)** Reserve VRAM for the shared cuFFT workspace: estimate `ws_est` (floored
at 1% of VRAM) and subtract it from the existing 0.60 budget fraction before
computing qb. (Note: `ws_est` does not scale with qb — the true workspace for
the largest plan at qb≈150 can reach several GB, larger than the 1.9 GB floor.
The retry fallback in (C) is the backstop for this residual underestimate.)

**(C)** Retry-on-failure fallback in `allocate_plan_device_memory()`
(gpu_plan.cu): if the arena `cudaMalloc` fails, halve `plan->q_batch`
(destroying and re-creating streams/events) and retry, up to 4 times.

### 1.6 Instance-Minute Estimate for OOM Fix

| Step | Action | Est. minutes |
|------|--------|-------------|
| Apply patch, rebuild | `git apply patches/oom_fix.patch && make` | 2 |
| Repro: n=1048576 k=n | single run | 3 |
| Verify fix | `./bench_gpu_fused verify` | 2 |
| **Total** | | **~7 min** |

---

## 2. cuFFT Plan Failure (Code 5) at n=524,288, k=n

### 2.1 Call Path

`create_cufft_plan()` in `gpu_plan.cu` → `cufftMakePlanMany()` with `int` typed parameters:

```c
int rank = 1;
int n_arr[1] = {n};       // fft_n up to 524,288
int ie[1] = {real_dist};  // = fft_n
int oe[1] = {cn};         // = fft_n/2+1
// ...
cufftMakePlanMany(*plan, rank, n_arr,
                  ie, 1, real_dist,   // idist, istride = 1, fft_n
                  oe, 1, cn,          // odist, ostride = 1, cn
                  CUFFT_D2Z, batch, &work_size);
```

### 2.2 Overflow Diagnosis

For n=524,288, k=n with B≈96 (planner estimate for this size) or B≈112:

Tree for n=524,288, B=96:
- nblocks = 5,462, N=8,192, L=14
- qb likely ~256 (similar per_q ratio)

Critical level: ell≈10 with fft_n=524,288:
- child_batch = nn[9] = 16, parent_batch = nn[10] = 8
- build_fft batch = qb × child_batch = 256 × 16 = **4,096**
- corr_fft batch_fwd = qb × parent_batch = 256 × 8 = **2,048**
- corr_fft batch_inv = qb × 2 × parent_batch = 256 × 16 = **4,096**

**The overflow:**

```
batch × fft_n = 4,096 × 524,288 = 2,147,483,648
```

This is **INT_MAX + 1** (INT_MAX = 2,147,483,647). cuFFT's `cufftMakePlanMany` takes `int batch` and internally computes 32-bit products like `batch × n` for index calculations. A product of 2,147,483,648 wraps to **−2,147,483,648** in signed 32-bit, causing `CUFFT_INTERNAL_ERROR` (code 5).

**Confirmation:** For the build plan at ell=10: batch=4096, n=524288 → exact overflow. For ell=9 with fft_n=262144 and batch=256×32=8192: 8192×262144=2,147,483,648 — same overflow at the same product boundary (2^31).

The overflow threshold `batch × n ≥ 2^31` is hit whenever:
- `qb × nn[ell-1] × fft_n ≥ 2,147,483,648`
- With qb=256, this happens when `nn[ell-1] × fft_n ≥ 8,388,608`

At n=524,288, the product `nn[ell] × fft_n` is roughly constant at ~2,097,152 per level (nn halves while fft_n doubles). With qb=256: 256 × 2,097,152 = 536,870,912 — below overflow for most levels. But the worst level is where nn[ell-1] × fft_n peaks, which is at the level where nn is highest while fft_n is large enough — around nn=16, fft_n=524288: 16×524288=8,388,608. With qb=256: 256×8,388,608=2,147,483,648 — exact overflow boundary!

**Also:** The `batch` parameter itself can overflow `int` since batch = qb × child_batch can exceed INT_MAX for very large qb and child_batch combinations.

### 2.3 Proposed Fix — `patches/cufft_524k_fix.patch`

Two complementary fixes:

**(A) Switch to `cufftMakePlanMany64`** (available since CUDA 11.0) which uses `long long` for all dimension parameters, preventing the 32-bit overflow entirely:

```c
long long n_arr[1] = {n};
long long ie[1] = {real_dist};
long long oe[1] = {cn};
cufftMakePlanMany64(*plan, rank, n_arr,
                    ie, 1LL, (long long)real_dist,
                    oe, 1LL, (long long)cn,
                    CUFFT_D2Z, (long long)batch, &work_size);
```

**(B) Fallback: batch splitting** — if the 64-bit API is not available (older CUDA), split the large batch into multiple cuFFT plans with `batch ≤ INT_MAX / fft_n`, executed sequentially. This is a compile-time `#if` guard.

The patch prefers the 64-bit API and falls back to the 32-bit path when
`CUDART_VERSION < 11000`.

> **W0R review refinement:** the overflow predicate now uses
> `batch × max(fft_n, real_dist) ≥ 2^31`, not `batch × fft_n`. The plans pass
> `real_dist = fft_stride`, and fft_stride can be up to 2× fft_n (a level's
> parent stride is padded to the NEXT level's fft_n), so the indexing span
> `batch × idist` can overflow while `batch × fft_n` is still below 2^31
> (e.g. corr-forward at qb=256, pb=8, stride=1,048,576).

### 2.4 Instance-Minute Estimate for cuFFT Fix

| Step | Action | Est. minutes |
|------|--------|-------------|
| Apply patch, rebuild | `git apply patches/cufft_524k_fix.patch && make` | 2 |
| Repro: n=524288 k=n | single run | 2 |
| **Total** | | **~4 min** |

---

## 3. `tools/b200_session.sh` — Pre-Scripted E2 Session

### 3.1 Design

Complete script for a supervisor to run on a fresh vast.ai B200 instance (CUDA 13, repo rsynced to ~/ICM). Every command wrapped in `timeout` with sensible caps; `set -e` with clear failure messages; total session ≤ 45 instance-minutes.

### 3.2 Steps

1. **Environment check** (nvidia-smi, nvcc version, VRAM) — 1 min
2. **Build** (`make bench_gpu_fused CUDA_ARCH=sm_100`) — 3 min
3. **Repro OOM** (n=1048576 k=n, timeout 5 min) — 3 min (expected: OOM)
4. **Repro cuFFT** (n=524288 k=n, timeout 5 min) — 3 min (expected: cufft error)
5. **Apply patches** (`git apply --check` then `git apply`) — 1 min
6. **Rebuild** — 3 min
7. **Verify fixes** (re-run the two repros) — 5 min
8. **Verify correctness** (`./bench_gpu_fused verify`) — 5 min
9. **Frontier confirmation** (n=1441792, n=1572864, k=n, 3 reps each) — 15 min
10. **Write outputs** to `results/` — 1 min
11. **Print rsync-back** — 0 min

Total: ~40 instance-minutes (with buffer: ~45 min).

See file: `tools/b200_session.sh`

### 3.3 Instance-Minute Estimate

| Phase | Est. minutes |
|-------|-------------|
| Env check + build | 4 |
| Bug repros (expected failures) | 6 |
| Patch apply + rebuild | 4 |
| Fix verification | 10 |
| Frontier confirmation | 15 |
| Output + wrap-up | 2 |
| **Total** | **~41 min** |

---

## 4. `tools/tier_ablation.cu` — Tier Ablation Runner

### 4.1 Design

Per `B200_RESUME_CHECKPOINT.md` step 4: for a set of representative tree levels / convolution sizes, directly time:
- Schoolbook kernel (no FFT)
- cuFFTDx fused kernel (R2C build + C2R propagate pair)
- Batched cuFFT (D2Z + Z2D pair)

Outputs a CSV: `size, batch, t_schoolbook_ms, t_fused_ms, t_cufft_ms, winner`

### 4.2 Representative Sizes

Convolution sizes spanning the full range:
- Small (within fused range): 64, 128, 256, 512, 1024, 2048, 4096, 8192
- Crossover region: 16384, 32768, 65536
- Large (cuFFT-only): 131072, 262144, 524288

Batch sizes spanning realistic tree parent counts: 1, 4, 16, 64, 256

### 4.3 Compilation Notes

- Mirrors includes from `tools/gpu_phase_profile.cu`: uses `<icm.h>`, `<icm_gpu.h>`, `"gpu/gpu_internal.h"`
- Uses `icm_gpu_detail` namespace for kernel dispatch functions
- Accesses `is_cufftdx_supported_fft_n`, `launch_cufftdx_build_r2c_dispatch`, `launch_cufftdx_corr_r2c_dispatch`
- **Untested status**: cannot compile or run locally (no GPU); noted in both the file header and this plan

See file: `tools/tier_ablation.cu`

### 4.4 Instance-Minute Estimate

| Step | Action | Est. minutes |
|------|--------|-------------|
| Build | `make tier_ablation CUDA_ARCH=sm_100` | 2 |
| Run | `./tier_ablation` | 10–20 |
| **Total** (E3 node) | | **~12–22 min** |

---

## 5. Summary of All Instance-Minute Estimates

| Deliverable | Node | Est. minutes |
|------------|------|-------------|
| OOM fix (patch + verify) | E2 | 7 |
| cuFFT fix (patch + verify) | E2 | 4 |
| Full E2 session (b200_session.sh) | E2 | 41 (includes above) |
| Tier ablation (build + run) | E3 | 12–22 |
| **Grand total (E2+E3)** | | **~53–63 min** |

---

## 6. Files Created

| File | Status |
|------|--------|
| `GPU_DEBUG_PLAN.md` | This file |
| `patches/oom_fix.patch` | Unified diff against src/gpu |
| `patches/cufft_524k_fix.patch` | Unified diff against src/gpu |
| `tools/b200_session.sh` | bash -n clean |
| `tools/tier_ablation.cu` | Draft, untested |
