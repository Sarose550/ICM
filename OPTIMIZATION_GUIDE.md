# ICM Equity Computation — Optimization Guide

## What We Did

### Problem
Compute `equity[i] = <payout, Q_i mod x^k>` for all `i = 1..n` players across
`Q = 256` quadrature points. This is the core of the Independent Chip Model
(ICM) for poker tournament equity using generating-function quadrature.

### Starting Point
Three naive algorithms: tree (schoolbook), naive (build + bidirectional divide),
linear (forward-backward). All used malloc/free per quadrature point, no SIMD,
no FFT. Baseline: n=1024 k=n took 689ms.

### Final Architecture
Three engines with dispatch:

```c
int B = select_engine(n, k);  // cost-based: compares linear vs hybrid
if (B > 0)
    use hybrid(B);        // block build + FFT tree + bidirectional divide
else
    use linear_batched;   // BQ=8 quad points, interleaved layout, fused backward pass
```

### Final Performance

> **Data provenance:** Calibrated 2026-07-20 with FFTW PATIENT wisdom on Apple M3 Pro
> (6P+6E, 12 logical cores). Cost-model constants refit from real M3 Pro hardware measurements
> (see `devices/m3_pro/fft_config.h` and [RESULTS.md](RESULTS.md#cost-model-constants)).

Single-threaded (ms, Q=256, Apple M3 Pro, median of 5):
```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
256      0.413  1.45   3.32   1.77   3.38   5.21
1024     1.71   7.16  13.2   24.4   28.9   34.8
4096     8.18  28.5   52.6  159    214    232
8192    16.2   56.6  104    438    483    516
16384   32.5  113    208   1020   1090   1480
32768   64.9  226    418   2330   3090   4680
65536  130    453    836   6580   9710  10000
```

## Optimizations — What Worked (ordered by impact)

### 1. FFTW Tree Engine with Calibrated Per-Level Decisions (5-97x for large k)
**The single biggest win.** Replaced schoolbook polynomial multiplication in the
subproduct tree with FFTW3's r2c/c2r transforms, with per-level decisions driven
by offline-calibrated data.

#### Offline calibration pipeline
1. **PATIENT wisdom**: generate FFTW_PATIENT plans for all 749 smooth (7-smooth:
   2^a·3^b·5^c·7^d) sizes up to 131072. Stored in `devices/<machine>/fftw_wisdom.dat`.
2. **Per-size benchmarking**: time the full r2c + pointwise + c2r pipeline at each
   smooth size (10K-10M reps). Stored in `devices/<machine>/fft_config.h` as
   `calib_sizes[]` and `calib_times_ns[]`.
3. **`fastest_fft_ge(n)`**: for non-cyclic operations (standard multiply, correlate),
   search all smooth sizes from n up to next_pow2(n) and return the one with the
   lowest calibrated time. Critical: the *smallest* smooth ≥ n is often not the
   *fastest* (e.g., 245 at 1144ns vs 256 at 740ns = 1.55x slower).

#### Per-level FFT vs schoolbook decision
At each tree level, `tree_ctx_create_ex2` compares:
```
fft_cost = calib_time[best_fft_size] + overhead(40ns) + correction(m)
school_cost = (d_eff + 1)² × FMA_NS
```
Where `d_eff = cps/2` at below-saturation levels (half the coefficients are zero)
and `d_eff = cps - 1` at saturated levels. Using `cps²` instead of `(d_eff+1)²`
overestimates schoolbook by 4x at below-sat levels — this was a bug that caused
FFT at tiny sizes where schoolbook was faster. The 40ns overhead was measured via
`./bench_grid profile` (plan lookup + buffer copies not in the calibration).

`FMA_NS` is device-specific: **M3 Pro = 0.0839 ns**, Zen 4 = 0.0500 ns
(AVX-512 scalar FMA).

Note: `school_cost_flops` must be `long long` — at cps=65536, `(65536)²` overflows
a 32-bit int, which caused the root level to use schoolbook (4.3 billion FMAs)
instead of FFT (1ms). This was the n=65536 blowup bug.

#### Cyclic multiply with multi-coefficient wrapping (m-wrap)
At below-saturation tree levels, polynomials have actual degree d stored in 2d
slots. The product has degree 2d. A standard FFT would need size 2d+1 ≈ 2d.
The cyclic multiply exploits the structure:

- Use an FFT of size N < 2d. The cyclic convolution wraps terms C[N..2d] back
  to positions C[0..2d-N], producing aliased values.
- The wrap count m = 2d - N. The correction is a schoolbook product of the top
  (m+1) coefficients of each input: cost = (m+1)² FMAs.
- **This lets us use a smaller, faster FFT by paying a tiny correction.**

`best_fft_config(L)` finds the optimal (fft_size, m) pair for convolution length L:
```
for each smooth S from L/2+1 to 2L:
    m = (S >= L) ? 0 : L - S
    cost = calib_time[S] + (m+1)² × WRAP_FMA_NS
    pick the S with minimum cost
```

Example: for L=256 on M3 Pro, the FFT at size 240 (7-smooth) costs 676ns +
17²×10.0 = 3576ns total, vs 256 (pow2) at 740ns. At WRAP_FMA_NS=10.0,
the m=16 wrap is far worse — the optimizer correctly picks 256 here.
At smaller m values (m≤4), wrapping is typically profitable.

For m=0 (no wrapping), this reduces to a standard cyclic convolution with a single
subtraction to undo the one aliased term (the z^{2d} coefficient).

m-wrap applies to ALL FFT operations — builds (both below-sat and saturated) and
correlates. The general function `polymul_fft_wrap` handles any input sizes.

#### Joint build+correlate optimization with automatic caching decisions

At each non-root FFT level, the code compares two strategies:

**Joint (cached + paired):** build and correlate share one FFT size S. FFT(P) is
cached from the build and reused in correlates. The paired correlate also shares
FFT(g) across both siblings.
```
joint_cost(S) = calib[S]                                // build (full pipeline)
              + (m_build+1)² × FMA_NS                   // build correction
              + calib[S] × PAIRED_CACHED_CORR_RATIO      // paired cached correlate
              + 2 × (m_corr+1)² × FMA_NS                // 2 correlate corrections
```

**Independent (uncached):** build and correlate each pick their own optimal FFT size.
The pair shares FFT(g) but computes FFT(P_reversed) fresh.
```
indep_cost = calib[build_size]                          // build (full pipeline)
           + (m_build+1)² × FMA_NS                     // build correction
           + INDEP_PAIR_RATIO × calib[corr_size]        // correlate_fft_pair
           + 2 × (m_corr+1)² × FMA_NS                  // 2 correlate corrections
```

These ratios are derived from the measured FFT phase split (see RESULTS.md) and
refit via `tools/fit_cost_model.py` against 200 sampled (n,k,B) plans on real hardware
(2026-07-20):

- `PAIRED_CACHED_CORR_RATIO`: cost of paired cached correlate / full FFT pipeline.
  **M3 Pro: 1.8205**, Zen 4: 2.9709.
- `INDEP_PAIR_RATIO`: cost of correlate_fft_pair (shared g, fresh P FFTs) / full
  FFT pipeline. **M3 Pro: 1.8205**, Zen 4: 2.9709 (fit_cost_model.py's single R,
  applied to both ratios — the two ratios converged to the same value on both devices).

All constants live as `#define`s in `fft_config.h` for per-device tuning.

The code picks whichever is cheaper. In practice:
- **Saturated levels** (build and correlate have similar conv_len): joint wins.
- **Below-sat levels** (correlate conv_len ≈ 2× build): independent wins, because
  padding the build to the correlate's larger size costs more than the caching saves.

#### k-padding to fastest smooth size
`best_k_pad(k)` searches smooth numbers near k and picks the one whose
saturated-level FFT cost (for conv_len = 2k-1) is minimized. Powers of 2 are
consistently fastest for FFTW (e.g., 1024 at 3585ns vs 1000 at 3941ns = 9% faster),
so the function naturally gravitates toward them.

### 2. Fused Backward Pass in Linear Engine (24% for linear, shifts crossover)
The linear engine's backward pass does two things per player: update the suffix
polynomial R, and compute the inner product ⟨P_{j-1}, R⟩. Previously these were
separate loops. Fusing them into one loop halves memory passes:
```c
for (int m = k-1; m >= 1; m--) {
    eq += gb[m] * R[m];           // dot product BEFORE update
    R[m] = aj * R[m] + bj * R[m-1]; // suffix update (R[m-1] still pristine)
}
```
This works because the suffix update runs backward, so R[m] is read for the dot
product before being overwritten. The fused version reduced linear times by 24%
(e.g., n=8192 k=50: 98ms → 74ms), pushing the linear→hybrid crossover from
k≈60 up to k≈95 for batched linear.

### 3. OpenMP Parallelism (~8-10x on 16 threads)
The Q=256 quadrature loop is embarrassingly parallel. Each thread gets its own
engine context (cloned workspace). Thread-local equity arrays avoid false sharing.
FFTW plan creation before parallel region (not thread-safe). Context cloning uses
`FFTW_MEASURE | FFTW_WISDOM_ONLY` with ESTIMATE fallback — this gives cloned
contexts the same PATIENT-quality plans as the original (critical for parallel
performance; using bare ESTIMATE produces significantly slower plans).
~9.5x on M3 Pro's P+E topology (n=8192 k=n: serial→parallel speedup, 16-thread).
HybridCtx pre-allocates permutation buffers (`a_sorted`, `inner_sorted`) to avoid
per-call malloc under parallel allocator contention — without this, tree beats hybrid
in parallel despite hybrid winning in serial.

### 4. Truncated Correlate (35-40% at below-saturation levels)
Exploit sparsity of P (only d+1 non-zero terms) AND limited range of g (only need
g[0..3d-1]) to **halve the correlate FFT size** from ~8d to ~4d.

### 5. Shared g Forward FFT (8-16% across all FFT levels)
At each propagation node, both child correlates FFT the same g_parent. The forward
FFT of g is shared across both siblings in ALL code paths:
- **Cached levels**: `correlate_fft_cached_pair_wrap` shares FFT(g) AND reuses
  cached FFT(P) from the build phase. Saves 1 fwd(g) per parent.
- **Non-cached levels**: `correlate_fft_pair` shares FFT(g) across both fresh-P
  correlates. Saves 1 fwd(g) per parent.
The paired cached variant was measured at 11-16% improvement at moderate k (where
cached levels dominate) and 3% at large k (fewer cached levels relative to total).

### 6. Quad-Point Batching for Linear Engine (2x at small k, large n)
Interleave BQ=8 quadrature points in the data layout (`a_batch[j*BQ+qi]`). The inner
loop vectorizes across quad points instead of across the tiny k dimension. The
interleaved layout is cache-friendly (contiguous access) and eliminates L1 misses that
occurred with the old strided layout. BQ=8 maps to 4 NEON FMA ports on Apple Silicon
and native AVX-512 width on Zen 4. Only helps for n ≥ 2048 (interleave overhead
exceeds savings at smaller n). Template in `src/linear_batched_impl.inc`.

### 7. Hybrid Block-Divide Engine (8-12%)
Replace the tree's bottom log₂(B) levels with: sequential block build (tight loop,
no tree overhead) + bidirectional divide (stable on the complete block product).
B is selected by cost model (`select_best_B`): typically B=16 on M3 Pro, B=32 on Zen 4.
Players sorted by stack size for branch-prediction-friendly divide direction.

### 8. Truncated Propagation (2-5%)
Only compute g_needed[ell-1] output terms at each level instead of full cgsz.
g_needed propagates upward from 1 (pure tree) or B (hybrid) at the leaves.

### 9. FFT Coefficient Caching (3-8% at saturated levels)
Cache FFT(P_child) during the build phase; reuse via conjugation during propagation.
Pad the build FFT to the correlate size so caching works at all saturated levels.

### 10. Skip Root Multiply (1-2%)
The root product of the subproduct tree is never read — propagation only uses
children's polynomials. Skipping saves the most expensive FFT multiply in the tree.

### 11. Ragged Tree (34% at non-pow2 n)
Track `n_real[ell]` per level instead of padding to power-of-2. Skip padding nodes
entirely in build and propagation. Lone children (odd n_real) are copied up without
multiplication. Only O(log n) boundary nodes are affected — the savings come from
skipping ~39% of nodes at non-pow2 n.

### 12. Workspace Pre-allocation (1.4-2.5x from original baseline)
Allocate engine workspace once, reuse across all 256 quadrature points.

### 13. FFTW Wisdom Persistence (320x startup speedup)
Save FFTW_MEASURE results to `fftw_wisdom.dat`. Subsequent runs skip benchmarking.

## What Did NOT Work

| Approach | Result | Why |
|---|---|---|
| Karatsuba | 1.5x slower at all sizes | FMA hardware: schoolbook `c[i+j] += a[i]*b[j]` is one FMA; Karatsuba trades multiplies for adds but both cost 1 cycle |
| CRT cyclic+negacyclic | 17% more expensive | r2c(2k) ≈ c2c(k) via Hermitian symmetry; explicit CRT adds overhead for negacyclic twiddle |
| Composite FFT sizes (ESTIMATE) | Slower | FFTW_ESTIMATE for non-power-of-2 produces suboptimal plans. Fixed: PATIENT wisdom + MEASURE plans make composites 10-30% faster than next power-of-2 |
| Checkpointed linear | 4% gain | Streaming access already prefetcher-friendly on M3 Pro/Max (400 GB/s) |
| Apple vDSP for inner loops | 2x slower | Per-call overhead at small k exceeds vectorization benefit |
| PGO / -Ofast | Hurt or neutral | PGO profile mismatch; -Ofast breaks FP64 precision |
| Batched divide on M3 Pro | No improvement | OoO execution already pipelines independent serial chains |
| 4-way tree | Worse than binary | LOO products create larger polynomials; binary minimizes correlate FFT sizes |
| FFTW NEON for FP64 | 3.4x slower | FFTW's NEON codelets suboptimal on Apple Silicon for doubles |
| Inlined schoolbook size 4/8/16 | No improvement | Compiler with -O3 already inlines small functions |
| Blocking the linear algorithm | Not profitable | Working set fits L2; streaming at 400 GB/s is faster than recomputation overhead |

## Why Binary Tree is Optimal

Binary tree minimizes total FFT work for both build and propagation:

- **Build**: n-1 multiplies regardless of tree shape. Balanced binary minimizes FFT
  sizes (inputs are equal-sized at each level).
- **Propagation**: each correlate uses the sibling polynomial. In a binary tree, the
  sibling is a single child (size p). In a k-ary tree, the sibling is a leave-one-out
  product of k-1 children (size (k-1)p), requiring larger FFTs.
- **Concrete**: for 4 leaves, binary does 6 correlates at sizes ~3p and ~6p;
  4-ary does 4 correlates at size ~7p plus 4 LOO multiplies. Binary wins by 2.3x.

## Why the Root Product is Wasted

The propagation starts with g_root = payout and correlates with the root's TWO
CHILDREN. It never reads the root polynomial itself. Skipping the root multiply
saves the single most expensive FFT in the entire tree (the largest polynomial size).

## Why Blocking the Linear Engine Doesn't Help (on M3 Pro/Max)

The linear forward-backward stores all n rows (O(nk) memory). The forward pass
writes sequentially; the backward pass reads in reverse. Both are streaming access
patterns that the hardware prefetcher handles perfectly at 400 GB/s.

Checkpointing reduces memory to O(√n·k) but adds 33% recomputation. On M3 Pro/Max
(unified memory at ~400 GB/s),
the recomputation costs more than the cache miss savings because streaming is
already fast. On bandwidth-limited hardware (Zen 4 at 80 GB/s), checkpointing
becomes essential.

The hybrid engine IS the blocked linear: block build = forward within a block,
divide = backward within a block, tree = inter-block structure. The block's working
set (B=16 → ~144 bytes) fits trivially in registers.

## OpenMP Parallelism

The Q=256 quadrature loop is embarrassingly parallel. Each thread gets its own
engine context (TreeCtx/HybridCtx with independent workspace). Thread-local
equity arrays are accumulated after the parallel region.

```
OMP_NUM_THREADS=16 ./bench_grid
```

### Implementation notes
- `fftw_make_planner_thread_safe()` before any plan creation (requires -lfftw3_threads)
- Context cloning: `tree_ctx_clone`, `hybrid_ctx_clone` create independent workspaces
  with `FFTW_MEASURE | FFTW_WISDOM_ONLY` plans (PATIENT-quality from wisdom, instant)
- Batched linear parallelized over Q/2 pairs with `schedule(dynamic, 4)`
- ~10x speedup on M3 Pro (limited by mixed P/E core topology and thread overhead)

## Porting to a New Device (General)

The codebase is designed for easy porting. All device-specific tuning lives in
`devices/<DEVICE>/fft_config.h` — no changes to `src/icm.c` needed. The engines,
cost models, and dispatch logic are fully parameterized by the constants in that header.

## Porting to AMD Zen 4 (Ryzen 7950X)

### Key Architectural Differences from M3 Pro/Max
- **SIMD**: AVX-512 (8 FP64/vector) vs NEON (2 FP64/vector)
- **Memory BW**: ~60 GB/s DDR5 vs 400 GB/s unified
- **L1 cache**: 32KB vs 192KB
- **L2 cache**: 1MB/core vs 32MB cluster
- **Cores**: 16P (no E-cores) vs M3 Pro's P+E topology
- **FMA throughput**: 2× 512-bit FMA/cycle = 16 FP64 FMA/cycle vs ~4
- **No vDSP**: Apple-only FFT dispatch auto-disabled via `#ifdef __APPLE__`

### Complete step-by-step porting guide

**Step 1: Generate calibration data.**

This is the most important step — all cost models depend on accurate per-size FFT
timings. Run on the target machine with minimal background load.

```bash
# Build the calibration tool
gcc -O3 -march=znver4 -o calibrate tools/calibrate.c -lfftw3 -lm

# Run calibration (pin to one core, high priority). Takes 10-30 min.
taskset -c 0 nice -20 ./calibrate

# Copy outputs to device directory
mkdir -p devices/zen4
cp fft_config.h devices/zen4/fft_config.h
cp fftw_wisdom.dat devices/zen4/fftw_wisdom.dat
```

This generates:
- FFTW PATIENT wisdom for all 749 smooth sizes (2^a·3^b·5^c·7^d up to 131072)
- `calib_sizes[]` and `calib_times_ns[]` arrays with per-size FFT pipeline costs
- Skeleton `#define`s for platform constants (need manual update in Step 3)

**Step 2: Build and profile.**

```bash
make DEVICE=zen4
./bench_grid profile
```

The profile output has three measurement sections. Record these values:

1. **FFT overhead table** — The "overhead" column is the per-call constant cost
   not captured in calibration (plan lookup, buffer copies, result extraction).
   → `FFT_OVERHEAD_NS`

2. **Schoolbook row** in the overhead table — `school_ns / cps²` at the largest
   schoolbook size gives the scalar FMA cost.
   → `FMA_NS` (measured: 0.0500 on Zen4 with AVX-512)

3. **Phase split table** — `f_fwd`, `f_pw`, `f_ifft` fractions at each FFT size.
   Compute the paired/independent correlate ratios:
   - `PAIRED_CACHED_CORR_RATIO`: shares FFT(g) and reuses cached FFT(P).
     Cost = fwd(g) + 2×(pw + ifft). Ratio = this / full_pipeline_calib.
     M3 Pro measured 1.6806 (fit_cost_model.py), Zen 4 measured 2.9709.
   - `INDEP_PAIR_RATIO`: shares FFT(g), computes FFT(P) fresh.
     Cost = fwd(g) + 2×(fwd(P) + pw + ifft). Ratio = this / full_pipeline_calib.
     M3 Pro measured 1.6806 (fit_cost_model.py), Zen 4 measured 2.9709.

**Step 3: Update platform constants in `devices/zen4/fft_config.h`.**

Edit the `#define`s at the top of the file with measured values:

```c
#define FMA_NS             0.0500  /* scalar FMA cost from profile schoolbook row */
#define FFT_OVERHEAD_NS    0.0     /* per-call FFT overhead, converged to 0 in fit */
#define PAIRED_CACHED_CORR_RATIO 2.9709  /* from phase split + fit_cost_model.py */
#define INDEP_PAIR_RATIO   2.9709  /* from phase split + fit_cost_model.py */
#define L2_CACHE_SIZE      (1 * 1024 * 1024)  /* 1MB per-core L2 */
```

**Important**: `L2_CACHE_SIZE` controls the checkpointing interval in the batched
linear engine (`ckpt_interval_batched`). Zen 4's 1MB L2 vs M3 Pro/Max's 32MB means
checkpointing activates much earlier, which is critical — without it, the linear
engine would stream through DRAM at 60 GB/s instead of L2 at ~1 TB/s.

**Step 4: Rebuild and verify.**

```bash
make DEVICE=zen4
./bench_grid verify     # ALL TESTS PASSED required — do not proceed without this
```

**Step 5: Verify dispatch decisions.**

```bash
./bench_grid crossover    # sweep k=40-150 at n=512-8192
```

This runs both linear and hybrid at each (n, k) and shows which wins. The
`select_engine()` cost model should match the empirical crossover. If it doesn't,
check that `FMA_NS` and `FFT_OVERHEAD_NS` are correct — the
dispatch is fully derived from these constants and the calibration table.

No manual `K_CROSS` tuning is needed — dispatch is cost-based.

**Step 6: Run the full benchmark grid.**

```bash
./bench_grid              # full grid, single-threaded
OMP_NUM_THREADS=16 ./bench_grid   # parallel scaling
```

Record results in RESULTS.md under the Zen 4 section.

### What auto-adapts (no manual tuning)

These features automatically adapt to Zen 4 via the calibration data and constants:

| Feature | How it adapts |
|---|---|
| `select_engine(n,k)` | Compares linear roofline cost vs hybrid cost (using `calib_times_ns[]`). Zen 4's faster schoolbook shifts crossover to higher k |
| `select_best_B(n,k)` | Derives optimal block size from calibration data. Typically B=32 on Zen 4 (vs B=16 on M3 Pro) because wider schoolbook regime |
| `ckpt_interval_batched` | Sized to fit working set in `L2_CACHE_SIZE`. Activates much earlier on Zen 4 (1MB vs 32MB) |
| BQ=8 batched linear | Same interleaved `a_batch[j*BQ+qi]` layout. AVX-512 processes 8 doubles natively per instruction |
| Per-level FFT vs schoolbook | Each tree level uses `calib_times_ns[]` to decide. Zen 4's faster schoolbook (AVX-512) means more levels use schoolbook |
| `best_fft_config()` | Picks optimal FFT size + wrap correction from Zen 4-specific calibration table |
| vDSP dispatch | Auto-disabled (not Apple Silicon). All FFTs use FFTW |

### What does NOT auto-adapt (needs measurement)

| Constant | Why manual | How to measure |
|---|---|---|
| `FMA_NS` | Hardware-specific scalar FMA cost | `./bench_grid profile` schoolbook row |
| `FFT_OVERHEAD_NS` | Plan lookup + buffer copy cost | `./bench_grid profile` overhead column |
| `PAIRED_CACHED_CORR_RATIO` | Depends on FFT phase balance | `./bench_grid profile` phase split |
| `INDEP_PAIR_RATIO` | Depends on FFT phase balance | `./bench_grid profile` phase split |
| `L2_CACHE_SIZE` | Hardware spec | CPU datasheet |

### Optional: Karatsuba at intermediate sizes

AVX-512 makes schoolbook ~4x faster than on NEON, potentially opening a gap between
schoolbook and FFT where Karatsuba (O(n^1.585)) could win. Prior from M3 Pro testing:
Karatsuba was 1.5x slower at all sizes (FMA hardware makes schoolbook's n² FMAs cheap).
On Zen 4, wider SIMD inflates the schoolbook regime further, making Karatsuba even less
likely to help — but measure to be sure.

### Optional: Intel MKL as alternative FFT backend

See "FFT library choice: FFTW vs Intel MKL" section below for dual-library dispatch.

### FFT library choice: FFTW vs Intel MKL

Both are supported — the code uses only the standard FFTW3 API.

**FFTW with AVX-512** (`--enable-avx512` at configure time, or distro package):
PATIENT wisdom is still essential — AVX-512 adds more codelet variants to the
search space, making the gap between ESTIMATE and PATIENT even larger (20-50%).
The calibration tool's wisdom generation step is critical.

**Intel MKL** (oneMKL, free, `apt install intel-mkl` or `conda install mkl`):
MKL's FFTW wrapper ignores all planning flags — `FFTW_PATIENT/MEASURE/ESTIMATE`
are accepted but have no effect. MKL always uses pre-tuned algorithms internally.
Wisdom import/export are no-ops. This means:
- Wisdom generation is unnecessary (skip `--wisdom-only` step)
- Plan creation is always fast and always optimal
- The clone code's `FFTW_MEASURE | FFTW_WISDOM_ONLY` works fine (MKL always succeeds)
- **Per-size calibration is still essential** — MKL still has size-dependent costs

MKL is typically 10-30% faster than FFTW on x86 for power-of-2 sizes, but FFTW
sometimes wins at composite (non-power-of-2) smooth sizes due to its specialized
codelets. For maximum performance, use **both** — dispatch to whichever is faster
per FFT size.

#### Dual-library dispatch (per-size best-of-both)

FFTW and MKL export the same symbols (`fftw_plan_*`, `fftw_execute`, etc.), so
they can't be linked simultaneously via normal linking. Use `dlopen` to load one
(or both) at runtime:

```c
/* At init time: load both libraries */
void *fftw_lib = dlopen("libfftw3.so", RTLD_NOW);
void *mkl_lib  = dlopen("libmkl_rt.so", RTLD_NOW);

/* Resolve function pointers */
typedef fftw_plan (*plan_r2c_fn)(int, double*, fftw_complex*, unsigned);
plan_r2c_fn fftw_plan_r2c = dlsym(fftw_lib, "fftw_plan_dft_r2c_1d");
plan_r2c_fn mkl_plan_r2c  = dlsym(mkl_lib,  "fftw_plan_dft_r2c_1d");
/* ... same for fftw_execute, fftw_plan_dft_c2r_1d, etc. */
```

**Calibration**: `tools/calibrate.c` should be extended to:
1. `dlopen` both libraries
2. For each smooth size, benchmark the full pipeline with both
3. Record `calib_times_ns[i] = min(fftw_time, mkl_time)` and
   `calib_lib[i] = FFTW or MKL` (which library won at that size)
4. Write both arrays into `fft_config.h`

**Runtime**: In `fft_cache_create_sizes()`, create each plan using the library
that won at calibration time. The `FFTPlan` struct gets an extra field indicating
which library's `fftw_execute` to call:

```c
typedef struct {
    int fft_n;
    fftw_plan fwd_plan, inv_plan;
    double *rbuf;
    fftw_complex *cbuf;
    int lib;  /* 0=FFTW, 1=MKL */
} FFTPlan;

/* Execute dispatch */
static execute_fn exec_fns[2];  /* exec_fns[0]=FFTW, exec_fns[1]=MKL */

static inline void fft_execute(FFTPlan *p) {
    exec_fns[p->lib](p->fwd_plan);
}
```

All downstream code (`polymul_fft_wrap`, `correlate_fft_cached_pair_wrap`, etc.)
calls `fft_execute(plan)` instead of `fftw_execute(plan->fwd_plan)` — one
indirection, zero overhead (branch predictor learns the per-size pattern instantly).

**Expected payoff**: 5-15% on sizes where one library significantly beats the other.
The composite smooth sizes (where m-wrap saves FFT size) are where the libraries
diverge most — FFTW's specialized codelets vs MKL's radix-2/3/5 focus.

### Compile
```bash
# Serial (FFTW only)
gcc -O3 -march=znver4 -Wall -Wno-unused-variable -Wno-unused-function \
    -Isrc -Idevices/zen4 -o bench_grid bench/bench.c -lfftw3 -lm

# Parallel (FFTW only)
gcc -O3 -march=znver4 -Wall -Wno-unused-variable -Wno-unused-function \
    -fopenmp -Isrc -Idevices/zen4 -o bench_grid bench/bench.c \
    -lfftw3 -lfftw3_threads -lm

# Parallel (dual-library: dlopen both at runtime, no direct link dependency)
gcc -O3 -march=znver4 -Wall -Wno-unused-variable -Wno-unused-function \
    -fopenmp -Isrc -Idevices/zen4 -o bench_grid bench/bench.c -ldl -lm
```

## Porting to NVIDIA H200 GPU

### Fundamental Architecture Change
The H200 is a throughput processor: 16896 CUDA cores, 4.8 TB/s HBM3e bandwidth,
~34 TFLOPS FP64 (CUDA cores), ~67 TFLOPS FP64 (tensor cores). This is a full
rewrite (~2000 lines CUDA), not a port. The algorithmic core (tree structure,
truncated propagation) stays the same — all infrastructure changes.

**Key physics:** FFTs are bandwidth-bound, not compute-bound. Each element is
read/written ~3 times (forward FFT, pointwise, inverse FFT). The H200's advantage
over H100 is primarily 4.8 vs 3.35 TB/s bandwidth (~1.4x), not TFLOPS. FP64
penalty vs FP32 is only 2x on CUDA cores but largely hidden by bandwidth limits.

The linear engine is inherently sequential (forward-backward pass) — only the
tree-based engines apply on GPU.

### Step 1: Implement tree engine in CUDA (level-parallel)

The tree's level-by-level structure maps to GPU: all node operations at each
level are independent. Each level is one batched cuFFT call + pointwise kernel,
with synchronization between levels.

- **Build (bottom-up):** one batched operation per level.
- **Propagation (top-down):** same structure, batched correlates.

### Step 2: Replace FFTW with cuFFT batched plans

At each tree level, batch `Q × nn[ell]` independent FFTs into one cuFFT call.

```c
int N = padded_poly_size;  // must be power of 2 on GPU
int batch = Q * nn[ell];   // e.g., 256 * 4096 = 1M FFTs

cufftHandle plan;
cufftCreate(&plan);
size_t workSize;
int n_arr[] = {N};
cufftMakePlanMany(plan, 1, n_arr,
    NULL, 1, N,           // input: contiguous, idist=N
    NULL, 1, N/2+1,       // output: contiguous, odist=N/2+1
    CUFFT_D2Z, batch, &workSize);
cufftExecD2Z(plan, d_input, d_output);
```

**FFT size selection.** cuFFT has optimized radix kernels for 7-smooth sizes
(2^a·3^b·5^c·7^d), with power-of-2 fastest, then 3, 5, 7. Non-smooth sizes
(large prime factors) fall back to Bluestein's algorithm (~3x overhead).
The performance gradient between pow2 and nearby smooth composites determines
whether m-wrap and composite size selection are worthwhile on GPU — if the
penalty is only 10-30% (like FFTW on CPU), the cyclic m-wrap trick still helps;
if it's 2x+, stick to pow2-only. **Benchmark:** compare cuFFT at pow2 vs nearby
7-smooth sizes (e.g., 1024 vs 960, 2048 vs 1920) on the target GPU.

**Shared work areas** across tree levels to reduce memory:
```c
cufftSetAutoAllocation(plan, 0);  // disable per-plan allocation
cufftSetWorkArea(plan, d_shared_workspace);  // reuse one buffer
```

### Step 3: Schoolbook vs FFT crossover on GPU

Research (CUMODP library) suggests the schoolbook→FFT crossover on GPU is at
**degree ~4096**, far higher than CPU's ~32. This is because:
- Schoolbook maps well to shared-memory CUDA kernels (coalesced access, no
  global memory traffic beyond initial load)
- cuFFT has per-call overhead and memory access patterns less suited to small sizes
- At the bottom tree levels, there are millions of tiny independent multiplies
  (Q × nn[ell]) giving abundant parallelism for schoolbook kernels

**Benchmark:** time shared-memory schoolbook kernel vs cuFFT batched at sizes
16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 with realistic batch counts.

### Step 4: Hybrid block size on GPU

The much higher schoolbook→FFT crossover fundamentally changes the hybrid strategy.
On CPU, B=16-32 (cost-model selected). On GPU, B=64 to B=256 may be optimal:

- B=64: eliminates 6 bottom tree levels, each needing a kernel launch + sync
- B=256: eliminates 8 levels. Block products are degree-256 polynomials,
  still well within the GPU schoolbook crossover
- The block build (n/B × Q independent chains of B schoolbook multiplies)
  and block divide (same structure) both have ample GPU parallelism
- Fewer tree levels = fewer `cudaDeviceSynchronize` barriers

**Benchmark:** sweep B = 8, 16, 32, 64, 128, 256. Measure total time including
kernel launch overhead. At small B, launch overhead dominates. At large B,
block products get expensive.

### Step 5: Reducing kernel launch overhead

Each tree level requires at least 3 kernel launches (forward FFT, pointwise
multiply, inverse FFT) plus synchronization. For a 15-level tree: ~45 launches.

**Option A: CUDA Graphs.** Capture the entire tree as a graph, replay with ~2.5μs
total launch overhead (vs ~135μs without). cuFFT supports CUDA Graphs on
single-GPU plans. Tree structure is fixed per (n, k), so graphs can be reused
across quadrature points.
```c
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int level = 0; level < num_levels; level++) {
    cufftSetStream(fwd_plan[level], stream);
    cufftExecD2Z(fwd_plan[level], ...);
    pointwise_multiply<<<...>>>(...);
    cufftSetStream(inv_plan[level], stream);
    cufftExecZ2D(inv_plan[level], ...);
}
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec_graph, graph, NULL, NULL, 0);
// Replay with O(1) overhead:
cudaGraphLaunch(exec_graph, stream);
```

**Option B: cuFFT LTO callbacks.** Fuse pointwise multiply into the inverse FFT's
load callback, eliminating one full memory pass (~20% of the FFT-multiply-IFFT
pipeline):
```c
// Device callback: multiply during IFFT load
__device__ cufftDoubleComplex cufftJITCallbackLoadDoubleComplex(
    void *dataIn, size_t offset, void *callerInfo, void *sharedPointer) {
    cufftDoubleComplex a = ((cufftDoubleComplex *)dataIn)[offset];
    cufftDoubleComplex b = ((cufftDoubleComplex *)callerInfo)[offset];
    cufftDoubleComplex r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    return r;
}
```
Compile to LTO-IR with `nvcc -dlto -dc`, embed with `bin2c`.

**Note:** CUDA Graphs and cuFFT callbacks currently conflict for out-of-place
transforms. Choose one based on benchmarking: graphs save launch overhead,
callbacks save memory bandwidth.

### Step 6: Quadrature loop on-device

Process all Q=256 quadrature points simultaneously. Keep everything on-device:
- Compute `a[j] = exp(S[j] * logv[q])` for all (j, q) in one kernel
  (n×Q independent exp calls, use CUDA math library).
- Never transfer per-quad-point data to/from host.
- Each quad point needs its own copy of the polynomial arrays.
  Batch dimension is folded into the cuFFT batch count.

### Step 7: Memory layout

cuFFT requires contiguous unit-stride input for optimal coalescing.

**Recommended:** `poly[level][batch][coeff]` where `batch = q * nn[ell] + node`.
All polynomials at a given level packed contiguously, each of length N (padded
to pow2). cuFFT parameters: `istride=1, idist=N, batch=Q*nn[ell]`.

Out-of-place transforms (separate real input and complex output buffers) are
cleaner than in-place for a polynomial tree — no padding worries.

Memory footprint for n=8192 k=8192 Q=256: tree storage ~14MB/quad × 256 ≈ 3.5GB.
Fits in 141GB HBM. At n=131072 k=131072: ~200MB/quad × 256 ≈ 50GB. Still fits.
Larger problems: process Q in chunks of 32-64.

### Step 8: Tune and validate

| Parameter | CPU value | GPU: benchmark |
|-----------|-----------|---------------|
| Schoolbook→FFT crossover | cps ≈ 32 | Expect ~4096; benchmark 16-8192 |
| Hybrid B | 8 | Sweep 8, 32, 64, 128, 256 |
| FFT sizes | 7-smooth | Benchmark pow2 vs 7-smooth composites; determines if m-wrap applies |
| Launch strategy | N/A | CUDA Graphs vs LTO callbacks — benchmark both |
| Q batch size | 256 (all) | Largest Q fitting HBM; benchmark chunk sizes |
| Work area sharing | N/A | `cufftSetAutoAllocation(0)` + shared buffer |

### Reference implementations
- **CUMODP** (cumodp.org): GPU subproduct tree for modular polynomial arithmetic.
  Achieves ~43% of peak bandwidth. Key paper: "On the Parallelization of Subproduct
  Tree Techniques Targeting Many-Core Architectures" (Springer, 2014).

Run correctness verification against CPU reference at n=256, 1024, 8192
before benchmarking.
