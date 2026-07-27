# GPU Ragged Tree Fix Plan

## Motivation

When the number of leaf blocks `nblocks = ceil(n/B)` is not a power of two,
the GPU hybrid engine's merge tree pads up to `N = next_pow2(nblocks)` slots
at every level.  The plan correctly computes `n_real[]` (real block count per
level, `n_real[0] = nblocks`, `n_real[ell] = ceil(n_real[ell-1]/2)`) and stores
it in `plan->n_real[]`, but every kernel-launch site in `gpu_exec.cu` uses
`plan->nn[ell]` (the padded power-of-two width) as its work-size, **never**
`plan->n_real[ell]`.  There is no equivalent anywhere in the GPU kernels of
the CPU's `if (2*j+1 >= nr_child)` padding-skip guard.

This was measured on real B200 hardware this session: `n=1,000,000` (ragged,
~4.6% padding) took 632ms vs `n=1,048,576` (exact power of two, zero padding)
at 511.9ms on the `k=n` curve — a 23.5% slowdown despite n=1,048,576 being
the strictly harder problem (more players, more leaf blocks, identical tree
depth `L`, identical `nn[]`).  That is a real, reproducible, non-monotonic
inversion.

**Caveat — k-pad confound**: because both runs use `k=n`, the two k values
differ (1,000,000 vs 1,048,576), so `best_k_pad_gpu()` may choose different
`k_pad` values, changing `psz[]` (polynomial sizes) and therefore per-level FFT
sizes and tier choices.  Any measurement **must** control for this: either fix
`k` to the same value for both `n` (e.g. `k=256` so k_pad is identical) or
force the same `B` and `k_pad` via env vars, to isolate the padded-tree effect.

---

## 1. Exact Call Sites

All five `nparents = plan->nn[ell]` sites in `src/gpu/gpu_exec.cu` and the
analogous q-batched variants that multiply by `qb`:

### Site 1: `run_build_level_schoolbook()` — line ~115

```c
int nparents = plan->nn[ell];
```

Launches one of three schoolbook-build kernels:
- `k_schoolbook_build_warp_batch` — `blocks = (nparents + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK`, one warp per parent.
- `k_schoolbook_build_smem_parent` — `blocks = nparents`, one block per parent.
- `k_schoolbook_build` (fallback) — flat grid over `nparents * pps` elements.

**What gets wasted**: each padding parent (index ≥ `n_real[ell]`) computes a
full convolution of two zero-polynomial children into a zero-polynomial parent.
The smem and warp-batch variants waste entire blocks; the fallback wastes
`pps` thread-elements per padding parent.

### Site 2: `run_build_level_fft()` — lines ~167–170

```c
int child_batch  = plan->nn[ell - 1];
int parent_batch = plan->nn[ell];
```

These feed:
- `k_gather_to_fft` — `batch = total_child = qb * child_batch`
- `cufftExecD2Z` — batch = `child_batch` (via `b.batch_fwd`)
- `k_pairwise_mul` — `total = total_parent = qb * parent_batch`
- `cufftExecZ2D` — batch = `parent_batch` (via `b.batch_inv`)
- `k_scatter_from_fft` — `batch = total_parent`
- `k_wrap_build` — `blocks = parent_batch`

**What gets wasted**: cuFFT plans were created with batch = `qb * nn[ell-1]`
and `qb * nn[ell]` at plan-creation time (`allocate_level_buffers()`,
`src/gpu/gpu_plan.cu` line ~1522).  The forward transform FFTs `nn[ell-1]`
child signals per Q-point; for padding children (index ≥ `n_real[ell-1]`),
the input is zero data.  cuFFT executes the full butterfly network regardless
of data values — zero inputs do NOT short-circuit.  The pointwise multiply and
inverse FFT likewise process `nn[ell]` parents, with padding parents (index ≥
`n_real[ell]`) producing zero output.  Wrap correction (`k_wrap_build`)
launches `nn[ell]` blocks; padding blocks find only zero data and compute
trivial wrap corrections (zero minus zero).

### Site 3: `run_build_level_fused()` — line ~377

```c
int nparents = plan->nn[ell];
```

Feeds:
- `launch_cufftdx_build_r2c_dispatch(…, nparents, …)` or `launch_cufftdx_build_dispatch(…, nparents, …)`
- `k_wrap_build` — `blocks = nparents`

**What gets wasted**: cuFFTDx dispatch performs the fused FFT + pointwise-mul
+ IFFT pipeline for `nn[ell]` parents.  Padding parents produce zero output.
Wrap correction same as Site 2.

### Site 4: `run_prop_level_schoolbook()` — line ~416

```c
int nparents = plan->nn[ell];
```

Launches one of three schoolbook-corr kernels:
- `k_schoolbook_corr_pair_warp_batch`
- `k_schoolbook_corr_pair_smem_parent`
- `k_schoolbook_corr_pair` (fallback)

**What gets wasted**: padding parents (index ≥ `n_real[ell]`) compute
correlation of a zero `g_parent` vector with zero child polynomials, producing
zero `g_child` output.  Entire blocks wasted for the smem/warp variants.

### Site 5: `run_prop_level_fft()` — line ~472

```c
int nparents = plan->nn[ell];
```

Feeds:
- `k_gather_to_fft` — `batch = nparents` (VkFFT path) or `nparents` (cuFFT)
- `cufftExecD2Z` — batch = `nparents` (via `c.batch_fwd`)
- `k_paired_corr_freq` — `total = nparents * cn`
- `cufftExecZ2D` + scatter — batch = `2 * nparents` (via `c.batch_inv`)
- `k_wrap_corr_pair` — `blocks = nparents`

Same waste pattern as build FFT: zero data through the full FFT pipeline.

### Site 6: `run_prop_level_fused()` — line ~666

```c
int nparents = plan->nn[ell];
```

Analogous to Site 3 but for the correlate direction.

### Q-batched variants

`run_build_level_schoolbook_qb`, `run_build_level_fft_qb`, etc. all multiply
`plan->nn[ell]` by `qb`; same waste scaled by Q-batch.

### Plan-time use (non-execution)

`build_plan_metadata()` line ~887:
```c
int nparents = plan->nn[ell];
…
double wrap_scale = wrap_serial_penalty_gpu(nparents);
```

This uses `nn[ell]` instead of `n_real[ell]` in the wrap-serial-penalty
computation, which feeds into `best_fft_config_gpu()` /
`best_fft_config_joint_gpu()` as the `correction_scale` parameter.  Since
`nn ≥ n_real`, the penalty is slightly **underestimated** (more parallelism
→ smaller penalty).  However, for `n=1,000,000` at B=32, `nn[ell] ≥ 16384` at
every level with padding, so `wrap_serial_penalty_gpu()` returns 1.0
regardless — the distortion has zero effect on FFT-size selection for this
problem size.  It could matter at very small `n` where `nn[ell]` is comparable
to `GPU_SM_COUNT` (~160 on B200), but at those sizes padding is already zero.

---

## 2. GPU-Appropriate Shape

### The CPU pattern (do not copy naively)

The CPU fix (`src/icm.c` lines 1435–1468 for build, 1515–1566 for propagate)
is a per-node scalar branch:

```c
if (2*j+1 >= nr_child) {
    memcpy(out, Lc, cp * sizeof(double));  // identity: lone child → parent
} else {
    polymul_fft_wrap(Lc, cps, Rc, cps, out, pps, …);  // full convolution
}
```

This is **correct** for a serial CPU loop but **wrong for GPU** if translated
literally into a per-thread branch inside a CUDA kernel.

### Why per-thread branching is bad here

Consider `k_schoolbook_build` (line 792 of `gpu_kernels.cu`), the flat-mapped
fallback kernel.  It maps `(p, m)` → `idx = p * parent_stride + m`, with one
thread per output coefficient.  If we insert a branch `if (p >= n_real)`
(or `if (2*p+1 >= n_real_child)`), threads within the same warp will have
different `p` values and diverge:

- Threads hitting a real parent execute the full convolution loop (tens of
  FMAs).
- Threads hitting a padding parent take the fast path (zero or copy).

Warp divergence on B200 means both paths execute serially within the warp —
the warp's runtime is max(path_A, path_B), not sum, but you lose the
throughput benefit of having all 32 lanes active simultaneously.  For this
kernel, the real-vs-padding split is ~95.4%/4.6%, so ~1–2 lanes per warp
diverge, costing ~5–10% throughput on affected warps.  This is modest but
adds up across all levels.

However, for the **one-block-per-parent** kernels (`k_schoolbook_build_smem_parent`,
`k_schoolbook_build_warp_batch`, `k_wrap_build`, `k_wrap_corr_pair`,
`k_schoolbook_corr_pair_smem_parent`, `k_schoolbook_corr_pair_warp_batch`),
there is **no intra-warp divergence** from a padding check, because every
thread in the block/warp works on the same parent `p`.  A guard like
`if (p >= nparents_real) return;` at the top of these kernels causes the
entire block/warp to exit early — zero divergence, zero wasted work.

### Recommended shape: launch-config shrink + boundary guard

**Change the launch configuration** to use `n_real[ell]` (or its Q-batched
equivalent) instead of `nn[ell]` **for the grid dimensions** of
one-block-per-parent kernels.  Padding slots are **never scheduled** as GPU
work — they don't even launch a block.  A safety guard `if (p >= n_real) return;`
is still added as defense-in-depth (costs one SASS instruction, no divergence
since all threads in the block see the same `p`).

For **cuFFT/cuFFTDx/VkFFT batched calls**, the situation is more complex
because:

1. **cuFFT plans are created with batch = `nn[ell]`** at plan time
   (`allocate_level_buffers()`).  Changing the batch at runtime would require
   plan recreation, which is expensive (~milliseconds per level).  The fix
   would need to change the **plan-time** batch to `n_real[ell]`.

2. **cuFFT operates on contiguous or regularly-strided data**.  The data
   layout has `nn[ell]` slots per level.  If we shrink the cuFFT batch to
   `n_real[ell]`, the forward FFT operates on children 0..n_real[ell-1]-1
   (all real), and the inverse FFT produces `n_real[ell]` parent outputs.
   But the `k_pairwise_mul` kernel accesses children at indices `(2p, 2p+1)`
   for `p = 0..n_real[ell]-1`, and the last parent may have only one real
   child (`2*n_real[ell]-1 >= n_real[ell-1]` when `n_real[ell-1]` is odd).
   This single-parent boundary case must be handled.

3. **cuFFTDx dispatch** takes `nparents` as a dynamic parameter.  Shrinking
   this to `n_real[ell]` is straightforward, with the same boundary-case
   consideration.

**Recommendation**: **Option B below (launch-config shrink everywhere)** is
the right shape.  It avoids warp divergence by not scheduling padding work
in the first place, handles the single boundary parent explicitly, and changes
cuFFT plan batch sizes at plan-creation time (not runtime).

---

## 3. Implementation Options

### Option A: In-kernel guard only (lowest risk, partial benefit)

**What changes**:
- In every kernel that receives `nparents` as a parameter
  (`k_schoolbook_build*`, `k_schoolbook_corr_pair*`, `k_wrap_build`,
  `k_wrap_corr_pair`, `k_pairwise_mul`, `k_paired_corr_freq`,
  `k_gather_to_fft`, `k_scatter_from_fft`), add a new parameter
  `int nparents_real` (or compute from existing `nparents` + new `nr_child`
  parameter), and add an early-exit guard:
  - For per-parent kernels: `if (p >= nparents_real) return;`
  - For flat-mapped kernels: compute `p = idx / stride; if (p >= nparents_real) return;`
  - For `k_pairwise_mul`/`k_paired_corr_freq`: same, plus handle the boundary
    parent — if `2*p+1 >= nr_child`, use identity spectrum `[1,0,0,…]` for
    the right child instead of reading actual (garbage) data.

- The launch configurations **stay unchanged** (still launch `nn[ell]` blocks).
  Padding blocks launch but immediately return — they consume a block slot
  on the SM for ~1µs (the cost of the guard + return) rather than running
  the full computation.

- The cuFFT batch sizes **stay unchanged**.  cuFFT still FFTs padding data
  (zeros).  This is the largest remaining waste.

**Risk**: Very low.  The GPU kernels already have `if (p >= nparents) return;`
guards at many call sites (e.g. `k_wrap_build` line 942, `k_schoolbook_build_smem_parent`
line 818, `k_schoolbook_corr_pair_smem_parent` line 1064).  Adding a tighter
bound is a one-line change per kernel.  The boundary-parent logic in
`k_pairwise_mul`/`k_paired_corr_freq` is correctness-critical and needs
careful review (one wrong spectral identity and equity values are wrong).

**Benefit**: Modest.  Eliminates schoolbook and wrap computation on padding
slots (~4.6% of levels with padding).  Does NOT eliminate cuFFT waste (the
bulk of FFT-tier compute).  Estimated wall-clock reduction: 10–15% of the
23.5% gap, i.e. maybe 2–3% overall speedup for ragged trees.

### Option B: Launch-config shrink + cuFFT plan rebatch (high benefit, medium risk)

**What changes**:

1. **Plan-time** (`build_plan_metadata()` in `gpu_plan.cu`):
   - Use `n_real[ell]` instead of `nn[ell]` when computing `wrap_scale`
     (minor — only matters for small `n`).
   - Store `n_real[ell]` in `GpuLevelPlan` (already available via
     `plan->n_real[ell]`, but having it in the level plan makes the
     execution code cleaner).

2. **Plan-time** (`allocate_level_buffers()` in `gpu_plan.cu`):
   - Change cuFFT plan batch sizes from `qb * plan->nn[ell-1]` to
     `qb * plan->n_real[ell-1]` (forward) and from `qb * plan->nn[ell]` to
     `qb * plan->n_real[ell]` (inverse).  This means cuFFT plans are created
     for the exact real batch size.
   - Same for VkFFT plan creation.
   - The cuFFT workspace calibration table (`estimate_cufft_workspace_bytes`)
     and workspace allocation must also use the real batch sizes.
   - FFT cache allocation (`d_fft_cache[ell]`) must be sized for
     `qb * n_real[ell-1] * cn` not `qb * nn[ell-1] * cn`.

3. **Execution-time** (all `run_*_level_*` functions in `gpu_exec.cu`):
   - Replace `plan->nn[ell]` with `plan->n_real[ell]` and
     `plan->nn[ell-1]` with `plan->n_real[ell-1]` everywhere they control
     work size.
   - For `k_pairwise_mul`: the launch is sized for `n_real[ell]` parents.
     The kernel already accesses children at `(2p, 2p+1)`.  Add a boundary
     check: when `2*p+1 >= n_real_child`, substitute the right child's
     spectrum with `{1.0, 0.0, 0.0, …}` (the FFT of a delta at position 0,
     i.e. the identity polynomial).  This makes the multiply produce the
     left child's spectrum unchanged (convolution with identity), which is
     the correct result for a lone-child parent.
   - Same for `k_paired_corr_freq`: when the right child doesn't exist,
     substitute its cached spectrum with `{1.0, 0.0, …}` so the correlation
     correctly passes the left child through.
   - Add a guard `if (p >= n_real) return;` in every kernel as defense.

4. **Boundary-parent helper**: Rather than add branching inside the hot
   multiply/correlate kernels, a cleaner approach is to **pre-fill** the
   boundary right-child spectrum slot with the identity spectrum.  This can
   be done with a tiny kernel launch (one block, or even `cudaMemset` for
   the real part + zero for imag) after the forward FFT and before the
   multiply.  This avoids any branch in the throughput-critical multiply
   kernel.

**Risk**: Medium.

- **cuFFT plan recreation**: cuFFT plans are created once per plan.  Changing
  batch sizes means different plan parameters; this is not a runtime cost
  issue, but the plan validation (sizes, strides, batch counts) must remain
  consistent.  The `batch_fwd`/`batch_inv` fields in `GpuFftBuffers` track
  the batch used at creation; these must match the actual call-site batch.

- **Boundary-parent correctness**: The identity-spectrum trick is
  mathematically exact (FFT of `[1, 0, 0, …]` is `[1+0i, 1+0i, 1+0i, …]`),
  but it relies on the real-to-complex FFT producing the expected DC component.
  For cuFFT R2C of length `fft_n`, a real input `[1, 0, 0, …, 0]` produces
  complex output `[1+0i, 1+0i, …, 1+0i]` (all ones).  This is verified by the
  cuFFT specification but should be explicitly tested.

- **Data layout**: Currently, data is laid out with `nn[ell]` slots per level,
  each of stride `psz[ell]`.  Shrinking the batch means the tail slots
  (indices `n_real[ell]` through `nn[ell]-1`) are never read or written by
  FFT operations.  They remain allocated in VRAM but become dead space.  This
  is acceptable — the memory wastage is ~4.6% at worst and the alternative
  (compacting the array) would require expensive data movement.

- **VkFFT path**: VkFFT plans are also created with specific batch sizes.
  Same rebatching needed.

- **cuFFTDx path**: The dispatch functions (`launch_cufftdx_build_*`,
  `launch_cufftdx_corr_*`) accept `nparents` as a runtime parameter.  Simply
  pass `n_real[ell]`.  Inside the dispatch, the batch loop uses `nparents`,
  so no further changes needed.

**Benefit**: High.  Eliminates all wasted GPU work — no cuFFT on padding data,
no schoolbook on padding parents, no wrap correction on padding, no gather/
scatter on padding.  The only remaining cost is the VRAM footprint of the
unused tail slots.  Expected to recover most of the 23.5% gap (estimated
15–20% of the gap, i.e. ~4% overall speedup), with the remainder possibly
due to the k-pad confound.

### Option C: Data compaction (highest benefit, highest risk)

Instead of leaving dead tail slots, repack the data arrays to have exactly
`n_real[ell]` entries per level.  This eliminates VRAM waste AND all padding
work, at the cost of non-power-of-two array dimensions throughout the codebase.

**Risk**: High.  Every array access, every stride computation, every offset
calculation in `gpu_exec.cu`, `gpu_kernels.cu`, `gpu_api.cu`, and `gpu_plan.cu`
must be audited.  The current code heavily relies on `nn[ell]` and power-of-two
sizes for indexing (e.g. `2*p` and `2*p+1` child indices are naturally
aligned when `nn` is a power of two).  Changing to ragged arrays introduces
per-level boundary conditions at every access point.  **Not recommended** for
the current sprint — the benefit over Option B is marginal (~4.6% VRAM savings
at the leaf level, less at higher levels) and the verification burden is large.

---

## 4. Magnitude Check

### Per-level padding fractions for n=1,000,000, B=32

```
nblocks = ceil(1,000,000 / 32) = 31,250
N = next_pow2(31,250) = 32,768 = 2^15   →  L = 16 (levels 0..15)
```

| Level ell | nn[ell] | n_real[ell] | Padding slots | Padding % |
|-----------|---------|-------------|---------------|-----------|
| 0 (leaf)  | 32,768  | 31,250      | 1,518         | 4.63%     |
| 1         | 16,384  | 15,625      | 759           | 4.63%     |
| 2         | 8,192   | 7,813       | 379           | 4.63%     |
| 3         | 4,096   | 3,907       | 189           | 4.61%     |
| 4         | 2,048   | 1,954       | 94            | 4.59%     |
| 5         | 1,024   | 977         | 47            | 4.59%     |
| 6         | 512     | 489         | 23            | 4.49%     |
| 7         | 256     | 245         | 11            | 4.30%     |
| 8         | 128     | 123         | 5             | 3.91%     |
| 9         | 64      | 62          | 2             | 3.13%     |
| 10        | 32      | 31          | 1             | 3.13%     |
| 11        | 16      | 16          | 0             | 0%        |
| 12        | 8       | 8           | 0             | 0%        |
| 13        | 4       | 4           | 0             | 0%        |
| 14        | 2       | 2           | 0             | 0%        |
| 15 (root) | 1       | 1           | 0             | 0%        |

Computed using the exact formulas from `build_tree_geometry()` in
`src/gpu/gpu_plan.cu` lines 330–356:

```c
n_real[0] = n_leaves;   // = nblocks
for (int ell = 1; ell < L; ++ell)
    n_real[ell] = (n_real[ell - 1] + 1) / 2;   // integer ceil division
nn[ell] = N >> ell;
```

**Key observation**: Padding fraction stays ~4.6% through level 5 (~97% of
the padding slots are concentrated in levels 0–5, which are also the levels
with the largest polynomial sizes and the most FFT work).  It drops to zero
from level 11 onward (where `n_real` naturally aligns with the power-of-two
sequence).  The weighted-average padding fraction, weighted by per-level
compute, is approximately 4.5%.

### Does 4.6% wasted compute explain a 23.5% wall-clock slowdown?

**Almost certainly not by itself.**  A 4.6% increase in FFT work should produce
at most a ~5% wall-clock increase (and likely less, since cuFFT batched calls
have sub-linear scaling with batch size — overhead amortization means an extra
4.6% batch costs less than 4.6% more time).  The 23.5% gap is ~5× larger than
the raw waste fraction.

Three possible explanations for the gap (not mutually exclusive):

1. **k-pad confound (most likely dominant factor)**.  `k=1,000,000` vs
   `k=1,048,576` produce different `k_pad` values via `best_k_pad_gpu()`.
   Different `k_pad` → different `psz[]` → different `conv_build`, `conv_corr`
   per level → potentially different FFT sizes and tier choices.  The
   `n=1,048,576` case (k is a power of two) likely hits a "sweet spot" where
   FFT sizes align naturally with cuFFT-calibrated sizes, while `n=1,000,000`
   may trigger suboptimal FFT-size choices — the same class of bug as the two
   already-found-and-fixed planning bugs this session.

2. **Schoolbook-tier interaction**.  If `n=1,000,000`'s k_pad pushes one or
   more levels from FFT-tier into schoolbook-tier (or vice versa), the cost
   model may mispredict, causing a level to use the wrong tier.  Schoolbook
   kernels at large conv lengths are dramatically more expensive than FFT.

3. **Padding waste amplified by FFT-size choice**.  The wasted compute itself
   may be at particularly expensive FFT sizes — if the ragged tree's per-level
   `conv_build`/`conv_corr` values happen to land on poorly-calibrated FFT
   sizes, the cost per wasted slot is higher than average.

**To disambiguate**, the next hardware measurement must control for k_pad
(see Section 5).

### Does padding perturb FFT-size selection through wrap_scale?

As analyzed in Section 1, `wrap_serial_penalty_gpu(nn[ell])` is used for the
`correction_scale` in `best_fft_config_gpu()` / `best_fft_config_joint_gpu()`.
For `n=1,000,000` at B=32, `nn[ell] ≥ 32` at every level with non-zero padding
(ell ≤ 10), and `GPU_SM_COUNT ≈ 160` on B200, so `GPU_SM_COUNT / nn[ell] ≤ 5`,
but the function clamps to `max(1.0, …)`.  At `nn[ell]=32`, penalty would be
`160/32 = 5.0`.  At `nn[ell]=64`, penalty is `160/64 = 2.5`.  At higher levels
(ell ≤ 5, nn ≥ 1024), penalty is 1.0.

The real-vs-padded difference: at ell=10, `nn[10]=32`, `n_real[10]=31`.  The
penalty difference is `160/32=5.0` vs `160/31≈5.16` — a 3% difference.  This
could theoretically shift the FFT-size choice at the highest tree levels, but
those levels contribute negligible compute (nn is tiny).  At the compute-heavy
levels (ell 1–5), penalty = 1.0 for both nn and n_real.  **Conclusion: the
wrap_scale distortion from using nn vs n_real is negligible for this problem
size.**

---

## 5. What to Measure Next

Before implementing any fix, disambiguate the k-pad confound from the padding
effect.  On the next B200 rental:

### Measurement A: Isolate the padded-tree effect

Pick a `k` value that is small enough that `k_pad` is identical for both `n`
values (or force `k_pad` explicitly):

```
# Fixed k, compare ragged vs power-of-two nblocks
ICM_GPU_FORCE_B=32 ICM_GPU_DEBUG_PLAN=1 ./bench_gpu n=1000000 k=256
ICM_GPU_FORCE_B=32 ICM_GPU_DEBUG_PLAN=1 ./bench_gpu n=1048576 k=256
```

With `k=256`, both runs will likely get the same `k_pad` (256 is small; pad
might be 256 or 288).  If the 23.5% gap persists, padding is the dominant
cause.  If it shrinks dramatically (to ~5%), the k-pad confound was dominant.

**Also capture with debug output enabled** (`ICM_GPU_DEBUG_PLAN=1`): the
per-level tier, fft_n, conv_build, conv_corr for both runs.  Compare
side-by-side.  If any level has a different tier or FFT size between the two
runs, that's a planning issue, not a padding-execution issue.

### Measurement B: Directly measure the padding waste

Add a one-line diagnostic to `run_build_level_fft()` and `run_prop_level_fft()`:

```c
if (plan->nn[ell] != plan->n_real[ell]) {
    fprintf(stderr, "  level %d: padding %d/%d (%.1f%%)\n",
            ell, plan->nn[ell] - plan->n_real[ell],
            plan->nn[ell],
            100.0 * (plan->nn[ell] - plan->n_real[ell]) / plan->nn[ell]);
}
```

Confirm the per-level padding fractions match the computed table above.

### Measurement C: Quick prototype of Option B

For ONE level (e.g. ell=1, the heaviest level), manually shrink the launch
config in `run_build_level_fft()` and `run_prop_level_fft()` to use
`n_real[ell]` instead of `nn[ell]`, with the identity-spectrum boundary fix
described in Option B.  Compare timing for `n=1,000,000` vs `n=1,048,576`
(both with fixed k).  If the gap closes substantially, Option B is validated.

### Expected outcome if the diagnosis is correct

- Measurement A with fixed k: gap shrinks from 23.5% to ~5–8% (the raw
  padding waste + some secondary effects).
- Measurement B: confirms per-level padding fractions match the computed
  table.
- Measurement C: gap shrinks to near zero (same fixed k, same B, one level
  fixed — if the fix works for the heaviest level, the remaining gap should
  be tiny).

If Measurement A shows the gap persists at ~23.5% even with identical k_pad,
then there is a THIRD mechanism beyond (a) the two already-fixed planning bugs
and (b) the padding-execution waste diagnosed here.  In that case, capture
full debug output (`ICM_GPU_DEBUG_PLAN=1`) at both n values and compare every
per-level decision (tier, fft_n, build_wrap_m, corr_wrap_m, cache_fft) to
find the discrepancy.

---

## Summary

- **5 execution call sites** + q-batched variants all use `plan->nn[ell]`
  (padded width) instead of `plan->n_real[ell]` (real block count) to size
  GPU work.
- **Recommended fix**: Option B — change launch configurations and cuFFT plan
  batch sizes to `n_real[ell]`, with an identity-spectrum boundary fix for
  the single case per level where a parent has only one real child.
- **Primary risk**: the k-pad confound may be the dominant cause of the 23.5%
  gap; isolate it first (Measurement A above) before committing to a full
  implementation.
- **Secondary risk**: the identity-spectrum boundary fix in `k_pairwise_mul`
  and `k_paired_corr_freq` must be mathematically verified against cuFFT's
  R2C output convention for the `[1, 0, 0, …]` input.
