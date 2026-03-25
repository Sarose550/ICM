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
int k_cross = (n >= 2048) ? 95 : 70;
if (k >= k_cross && n >= 256)
    use hybrid(B=8);    // block build + FFT tree + bidirectional divide
else if (n >= 2048)
    use linear_batched;  // 2 quad points interleaved, fused backward pass
else
    use linear;          // plain forward-backward, fused backward pass
```

### Final Performance

Single-threaded (ms, Q=256, Apple M3 Max):
```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
1024     3     11     18     21     25     27
4096     9     37     72    126    142    153
8192    19     73    143    297    343    360
```

16-thread parallel (ms):
```
n        k=n     ratio_to_8192
8192     46      1.0x
16384    110     2.4x
32768    249     5.4x
65536    569     12.3x
131072   1216    26.2x
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
school_cost = (d_eff + 1)² × 0.25 ns/FMA
```
Where `d_eff = cps/2` at below-saturation levels (half the coefficients are zero)
and `d_eff = cps - 1` at saturated levels. Using `cps²` instead of `(d_eff+1)²`
overestimates schoolbook by 4x at below-sat levels — this was a bug that caused
FFT at tiny sizes where schoolbook was faster. The 40ns overhead was measured via
`./bench_grid profile` (plan lookup + buffer copies not in the calibration).

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
    cost = calib_time[S] + (m+1)² × 0.25 ns/FMA
    pick the S with minimum cost
```

Example: for L=256, the FFT at size 240 (7-smooth) costs 676ns + 17²×0.25 = 748ns
total, vs 256 (pow2) at 740ns. At 0.25ns/FMA, the m=16 wrap is slightly worse —
the optimizer correctly picks 256 here. At smaller m values (m≤4), wrapping is
typically profitable.

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

These ratios are derived from the measured FFT phase split (see RESULTS.md):
- `PAIRED_CACHED_CORR_RATIO`: fwd(g) + 2×(pw + ifft) relative to full pipeline.
  M3 Max: (0.30 + 2×0.365) = 1.03×calib.
- `INDEP_PAIR_RATIO`: fwd(g) + 2×fwd(P_rev) + 2×pw + 2×ifft relative to full pipeline.
  M3 Max: ~1.25×calib (empirical; theoretical 1.63 overestimates due to pipeline overlap).

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
~8-10x on M3 Max's 12P+4E topology.

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
Interleave 2 quadrature points in the data layout. The inner loop vectorizes across
quad points (NEON width 2) instead of across the tiny k dimension. Only helps for
n ≥ 2048 (interleave overhead exceeds savings at smaller n).

### 7. Hybrid Block-Divide Engine (8-12%)
Replace the tree's bottom log₂(B) levels with: sequential block build (tight loop,
no tree overhead) + bidirectional divide (stable on the complete block product).
B=8 is optimal on ARM. Players sorted by stack size for branch-prediction-friendly
divide direction.

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
| Checkpointed linear | 4% gain | Streaming access already prefetcher-friendly on M3 Max (400 GB/s) |
| Apple vDSP for inner loops | 2x slower | Per-call overhead at small k exceeds vectorization benefit |
| PGO / -Ofast | Hurt or neutral | PGO profile mismatch; -Ofast breaks FP64 precision |
| Batched divide on M3 Max | No improvement | OoO execution already pipelines independent serial chains |
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

## Why Blocking the Linear Engine Doesn't Help (on M3 Max)

The linear forward-backward stores all n rows (O(nk) memory). The forward pass
writes sequentially; the backward pass reads in reverse. Both are streaming access
patterns that the hardware prefetcher handles perfectly at 400 GB/s.

Checkpointing reduces memory to O(√n·k) but adds 33% recomputation. On M3 Max,
the recomputation costs more than the cache miss savings because streaming is
already fast. On bandwidth-limited hardware (Zen 4 at 80 GB/s), checkpointing
becomes essential.

The hybrid engine IS the blocked linear: block build = forward within a block,
divide = backward within a block, tree = inter-block structure. The block's working
set (B=8 → 72 bytes) fits trivially in registers.

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
- ~10x speedup on M3 Max (limited by mixed P/E core topology and thread overhead)

## Fine-Tuning for AMD Zen 4 (Ryzen 7950X)

### Key Architectural Differences
- **SIMD**: AVX-512 (8 FP64/vector) vs M3 Max's NEON (2 FP64/vector)
- **Memory BW**: ~60 GB/s DDR5 vs M3 Max's 400 GB/s unified
- **L1 cache**: 32KB vs M3 Max's 192KB
- **L2 cache**: 1MB/core vs M3 Max's 32MB cluster
- **Cores**: 16P (no E-cores) vs M3 Max's 12P + 4E
- **FMA throughput**: 2× 512-bit FMA/cycle = 16 FP64 FMA/cycle vs M3 Max's ~4

### Step-by-step calibration

**Step 1: Calibrate FFT sizes (generates fft_config.h + fftw_wisdom.dat).**
```bash
gcc -O3 -march=znver4 -o calibrate tools/calibrate.c -lfftw3 -lm
taskset -c 0 nice -20 ./calibrate     # 10-30 min on quiet machine
cp fft_config.h devices/zen4/fft_config.h
cp fftw_wisdom.dat devices/zen4/fftw_wisdom.dat
```
This generates FFTW PATIENT wisdom for all 749 smooth sizes, benchmarks the full
r2c + pointwise + c2r pipeline at each size, and writes the calibration header.

**Step 2: Build and measure platform constants.**
```bash
make DEVICE=zen4    # or: gcc -O3 -march=znver4 -Isrc -Idevices/zen4 ...
./bench_grid profile
```
The profile output has three measurement tables:
- **FFT overhead**: the "overhead" column gives `FFT_OVERHEAD_NS`.
- **Phase split**: `f_fwd`, `f_pw`, `f_ifft` fractions. Compute:
  - `PAIRED_CACHED_CORR_RATIO = f_fwd + 2×(f_pw + f_ifft)` (relative to fwd+pw+ifft sum,
    then scale by sum/calib to get ratio relative to calib).
  - `INDEP_PAIR_RATIO = (3×fwd + 2×pw + 2×ifft) / calib` at representative sizes.
- **Schoolbook row** in FFT overhead table: derive `FMA_NS` from
  `school_ns / cps²` at the largest schoolbook size.

Update the `#define`s at the top of `devices/zen4/fft_config.h`:
```c
#define FMA_NS 0.06               /* expected: ~0.06-0.08 with AVX-512 */
#define FFT_OVERHEAD_NS 30.0      /* measure from profile output */
#define PAIRED_CACHED_CORR_RATIO 1.03  /* re-derive from phase split */
#define INDEP_PAIR_RATIO 1.25     /* re-derive from phase split */
```

**Step 3: Rebuild and tune dispatch.**
```bash
make DEVICE=zen4
./bench_grid crossover    # sweep k=40-150 at n=512-8192
```
Find the k where hybrid first beats linear at each n. Update `k_cross` in
`icm_equity()` and `compute_equity_subset()` if the crossover shifted.

**Step 4: Tune Zen 4-specific parameters in icm.c.**

| Parameter | M3 Max | Expected Zen 4 | How to tune |
|-----------|--------|----------------|-------------|
| `BQ` | 2 | 4 or 8 | AVX-512 is 8-wide; test BQ=4 and BQ=8 in `run_linear_batched` |
| `CKPT_THRESHOLD` | 4194304 (32MB) | ~250K-500K | 1MB L2; sweep with `./bench_grid quick` |
| `B` (hybrid) | 8 | 8 | Benchmarked: B=8 is optimal (B=16/32 help <3%, B=64 regresses). Keep 8 unless Zen 4 data disagrees — sweep B=4,8,16,32 to confirm |
| `k_cross` | 95/70 | Measure | From crossover sweep |
| `OMP_NUM_THREADS_DEFAULT` | 16 | 16 | Match physical core count |

**Step 5: Test Karatsuba for intermediate multiply sizes.**
AVX-512 makes schoolbook ~4x faster than on NEON, potentially opening a gap between
schoolbook and FFT where Karatsuba (O(n^1.585)) could win. Prior from M3 Max testing:
Karatsuba was 1.5x slower at all sizes (FMA hardware makes schoolbook's n² FMAs cheap).
On Zen 4, wider SIMD inflates the schoolbook regime further, making Karatsuba even less
likely to help — but measure to be sure.

Test approach: implement Karatsuba multiply for sizes 64, 128, 256, 512, 1024.
Use the standard recursive formulation with a schoolbook base case at size 16-32.
Search the internet for optimized AVX-512 Karatsuba implementations as reference.
Compare against both schoolbook (`polymul_modk`) and FFT (`polymul_fft_wrap`) at each
size. If Karatsuba beats both at any size, add it as a per-level option in the tree.
If not (expected), document the result and move on.

**Step 6: Checkpointing tuning.**
With ~5x less memory bandwidth than M3 Max, streaming through large g_store arrays
becomes bandwidth-bound. Lower `CKPT_THRESHOLD` to fit the linear engine's working set
in L2. Sweep values:
```c
#define CKPT_THRESHOLD 262144   /* try 2MB (256K doubles) */
#define CKPT_THRESHOLD 524288   /* try 4MB (512K doubles) */
#define CKPT_THRESHOLD 1048576  /* try 8MB (1M doubles) */
```
Measure at n=4096 k=100 (linear-dominated) and n=8192 k=50 (linear-dominated, large).

**Step 7: Verify and benchmark.**
```bash
./bench_grid verify     # ALL TESTS PASSED required
./bench_grid            # full grid
OMP_NUM_THREADS=16 ./bench_grid   # parallel scaling
```

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
On CPU, B=8 (schoolbook up to degree 8). On GPU, B=64 to B=256 may be optimal:

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
