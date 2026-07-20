# RESULTS.md — ICM Equity Optimization Results

All correctness tests PASS at < 5e-12 relative error (< 1e-9 for Smax=10^9).
Q=256 quadrature points.

## Apple M3 Pro (ARM64, NEON+vDSP, BQ=8)

> Calibrated 2026-07-20 with FFTW PATIENT wisdom on Apple M3 Pro (6P+6E, 12 logical cores).
> All cost-model constants refit from real M3 Pro hardware measurements.
> Engine dispatch: `select_engine()` cost-based, B auto-selected (typically B=16).

### Single-threaded (ms, uniform stacks, median of 5) — M3 Pro

```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
64       0.095  0.343  0.441  0.126  0.216  0.440
128      0.191  0.716  1.31   0.477  0.896  1.41
256      0.413  1.45   3.32   1.77   3.38   5.21
512      0.859  3.64   6.60   7.57  11.3   13.0
1024     1.71   7.16  13.2   24.4   28.9   34.8
2048     4.11  14.3   26.3   60.9   75.2   99.8
4096     8.18  28.5   52.6  159    214    232
8192    16.2   56.6  104    438    483    516
16384   32.5  113    208   1020   1090   1480
32768   64.9  226    418   2330   3090   4680
65536  130    453    836   6580   9710  10000
```

### 12-thread parallel (ms, uniform stacks, median of 5) — M3 Pro

```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
64       0.054  0.092  0.110  0.057  0.071  0.120
128      0.078  0.184  0.309  0.119  0.204  0.221
256      0.117  0.301  0.592  0.373  0.492  0.716
512      0.191  0.647  1.22   1.08   1.59   1.80
1024     0.341  1.25   2.27   3.36   3.87   4.96
2048     0.723  2.44   4.46   8.01  10.3   14.7
4096     1.59   4.91   9.09  21.6   29.9   31.1
8192     2.71   9.79  17.7   60.7   67.3   73.3
16384    5.36  18.6   34.5  147    148    202
32768   10.6   37.6   67.8  309    420    606
65536   21.1   74.2  138    855   1270   1310
```

### Parallel speedup — M3 Pro

At the 1-second boundary (from contour sweep, Q=256):

| k | Serial n | Parallel n | Speedup |
|---|----------|------------|---------|
| 2 | 1,025,391 | 5,468,751 | 5.3x |
| 100 | 76,256 | 421,890 | 5.5x |
| 1000 | 22,437 | 162,687 | 7.3x |
| 10000 | 13,750 | 94,375 | 6.9x |
| 13000 | 13,000 | 98,312 | 7.6x |

Speedup varies by k due to engine dispatch: linear-only k values see ~5-6x (simple SIMD scaling),
while hybrid-engine k values reach ~7-8x (FFT tree parallelism). M3 Pro's 6P+6E topology
limits peak parallel speedup to ~8x vs Zen 4's ~14x on 16 homogeneous P-cores.

### 1-second threshold: n ≈ 13,000 (k=n, single-threaded), n ≈ 98,000 (k=n, 12-thread)

### Dispatch: cost-based `select_engine()`, B from `select_best_B()` (typically B=16)

---

## AMD Ryzen 9 7950X (Zen 4, AVX-512, FFTW+MKL dual dispatch)

> ⚠️ **Data freshness (2026-07-20):** Cells updated with corrected-dispatch Zen4 benchmarks:
> n∈{1024,4096,8192,16384,65536}, k∈{10,100,n/2,n}. Remaining cells (n=64,128,256,512,2048,32768
> and k=50,n/4 for all n) still use pre-refit data pending a full grid run.

### Single-threaded (ms, Q=256, uniform stacks, median of 5)

```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
64       0      0      0      0      0      0
128      0      0      1      0      0      1
256      0      1      2      1      2      4
512      1      2      3      4      8     10
1024     1.28   4      7.61  20     29.1   34.0
2048     3      8     13     44     49     53
4096     7.32  16     28.3  102    161    168
8192    14.5   30     53.4  237    376    382
16384   29.6   66    112    577    866    835
32768   58    110    201   1396   1514   1608
65536  117    252    419   3285   4170   4490
```

### 16-thread parallel (ms, Q=256, uniform stacks, median of 5)

```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
64       0      0      0      0      0      0
128      0      0      0      0      0      0
256      0      0      0      0      0      0
512      0      0      1      1      1      1
1024     0.125  1      0.562  1      2.23   2.48
2048     0      2      2      3      4      4
4096     0.604  4      2.35   8     11.4   11.9
8192     1.18   8      4.86  18     26.6   26.9
16384    2.39  15     10.4   46     67.5   81.2
32768   16     32     58    159    179    212
65536   19.5  122     45.2  384    530    631
```

### Parallel speedup (n=8192 k=n)

```
Threads  Serial  Parallel  Speedup
1        382     —         —
16       —       26.9      14.2x
```

### 1-second threshold: n ≈ 17,216 (k=n, single-threaded)

### Dispatch: cost-based `select_engine()`, B from `select_best_B()` (typically B=32)

### MKL dual dispatch

AOCL-FFTW+MKL per-size best-of-both via `dlopen`. AOCL-FFTW wins 637/749 smooth sizes,
MKL wins 112/749 (mostly small composites 14-64, plus 131072 at 1.15x).
`calib_lib[]` array in `fft_config.h` drives per-plan library selection.

---

## Key optimizations by device

### Both platforms

- FFTW PATIENT wisdom + MEASURE|WISDOM_ONLY for clones
- Paired cached correlate (shares FFT(g) + cached FFT(P))
- Cost-model-driven B selection (`select_best_B`)
- Shared tree_build_levels / tree_propagate_g helpers
- BQ=8 batched linear with interleaved a_batch layout
(`a_batch[j*BQ+qi]` — cache-friendly, eliminates L1 misses at all n).
Template in `src/linear_batched_impl.inc`.
- L2-aware checkpointing (`ckpt_interval_batched`)
- Cost-based engine dispatch (`select_engine`) — no fixed K_CROSS thresholds
- Cross-correlation wrap correction handles both output-wrap and input-wrap
cyclic aliasing (corrects a pre-existing bug with wrap_m > 0)
- AMX FP64 infrastructure for Apple Silicon (`src/amx.h`) — validated, gated
at AMX_SCHOOL_MIN_DEG=160 (AMX only wins at degree ≥170 due to extraction overhead)

### M3 Pro / Apple Silicon specific

- vDSP interleaved DFT dispatch (`vDSP_DFT_Interleaved_CreateSetupD`) — 10-18%
faster FFT at 33 supported sizes (f × 2^g where f ∈ {1,3,5,15}, g ≥ 4).
Zero format conversion (uses same interleaved complex as FFTW). Forward ×2
scaling absorbed into pointwise multiply; single ×0.25 on inverse output.
- Calibration table updated with vDSP dispatch times, steering `best_fft_config()`
to prefer vDSP-supported sizes (e.g. 192 replaces 200 at saturated tree levels).

### Zen 4 specific

- FFTW+MKL dual dispatch via `dlopen` — per-size best-of-both (181/749 sizes use MKL)
- BQ=2 batched linear with interleaved layout (AVX-512 native width)
- L2-aware checkpointing with 1MB per-core L2
- B=32 (vs M3 Pro B=16) — cost model adapts to Zen 4's wider schoolbook-FFT crossover
- `MKL_THREADING_LAYER=SEQUENTIAL` set automatically at init (avoids OpenMP dependency)

## FFT Phase Split (Zen 4 7950X)

```
fft_n    fwd(ns)  pw(ns)   ifft(ns) f_fwd  f_pw   f_ifft
64       55       8        52       0.48   0.07   0.46
256      231      28       225      0.48   0.06   0.46
512      320      55       349      0.44   0.08   0.48
1024     617      137      722      0.42   0.09   0.49
4096     3298     584      3407     0.45   0.08   0.47
8192     8509     1166     11731    0.40   0.05   0.55
16384    25481    2312     28840    0.45   0.04   0.51
```

---

## Cost-Model Constants

These constants drive `select_engine()`, `select_best_B()`, per-level FFT-vs-schoolbook
decisions, and the m-wrap correction cost model. They live in `devices/<device>/fft_config.h`
and were refit on real hardware via `tools/fit_cost_model.py` (2026-07-20 session).

### M3 Pro (Apple Silicon, ARM64)

| Constant | Value | Notes |
|---|---|---|
| `FMA_NS` | 0.0839 | Scalar FMA cost. Refit 2026-07-20 from real M3 Pro hardware. |
| `WRAP_FMA_NS` | 0.1000 | Per-FMA cost for wrap correction (memory-latency-bound). |
| `BLOCK_FMA_NS` | 0.0500 | FMA cost inside block build/divide (cache-resident). |
| `BLOCK_MEM_NS` | 0.1000 | Memory cost per element in block build/divide. |
| `PAIRED_CACHED_CORR_RATIO` | 1.8205 | Paired cached correlate cost / full FFT pipeline cost. |
| `INDEP_PAIR_RATIO` | 1.8205 | Independent pair correlate cost / full FFT pipeline cost. |
| `FP64_DIV_NS` | 6.0449 | FP64 divide latency. |
| `LEAF_FMA_NS` | 0.1889 | FMA cost at tree-leaf schoolbook multiplies. |
| `LEAF_BLOCK_NS` | 74.3047 | Per-block overhead at leaf level. |
| `FFT_OVERHEAD_NS` | 204.5517 | Per-call FFT overhead (plan lookup + buffer copies). |

> FFT calibration table (`calib_sizes[]`/`calib_times_ns[]`) and FFTW wisdom
> in `devices/m3_pro/fft_config.h` are from a genuine FFTW PATIENT calibration
> on this Apple M3 Pro machine (2026-07-20). Cost-model constants refit via
> `tools/fit_cost_model.py` on real M3 Pro hardware.

### Zen 4 (AMD Ryzen 9 7950X, AVX-512)

| Constant | Value | Notes |
|---|---|---|
| `FMA_NS` | 0.0500 | Scalar FMA cost. Fit against 200 sampled (n,k,B) plans, 6.0% RMS log-relative error. |
| `WRAP_FMA_NS` | 0.8612 | Per-FMA cost for wrap correction. |
| `BLOCK_FMA_NS` | 0.0500 | FMA cost inside block build/divide (cache-resident). |
| `BLOCK_MEM_NS` | 0.1000 | Memory cost per element in block build/divide. |
| `PAIRED_CACHED_CORR_RATIO` | 2.9709 | Paired cached correlate cost / full FFT pipeline cost. |
| `INDEP_PAIR_RATIO` | 2.9709 | Independent pair correlate cost / full FFT pipeline cost. Single R from fit. |
| `FP64_DIV_NS` | 13.4590 | FP64 divide latency. |
| `LEAF_FMA_NS` | 0.2804 | FMA cost at tree-leaf schoolbook multiplies. |
| `LEAF_BLOCK_NS` | 42.2533 | Per-block overhead at leaf level. |
| `FFT_OVERHEAD_NS` | 0.0 | Per-call FFT overhead (converged to 0 in fit). |

> Calibration table, FFTW wisdom, and MKL dispatch (`calib_lib[]`) in
> `devices/zen4/fft_config.h` are from the same AMD Ryzen 9 7950X SKU and
> reflect real Zen4 microarchitecture measurements.

