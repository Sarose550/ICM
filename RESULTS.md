# RESULTS.md — ICM Equity Optimization Results

All correctness tests PASS at < 5e-12 relative error (< 1e-9 for Smax=10^9).
Q=256 quadrature points.

## Apple M3 Max (ARM64, NEON+vDSP, BQ=8)

### Single-threaded (ms, uniform stacks, median of 5)

```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
64       0      1      1      0      0      1
128      0      1      2      1      2      2
256      1      3      4      3      4      4
512      2      6      8      8     10     11
1024     3     11     16     21     24     27
2048     4     12     23     51     58     63
4096     7     24     46    121    137    147
8192    14     50    115    287    318    350
16384   28    122    230    660    709    752
32768   55    240    457   1511   1710   1808
65536  135    460    937   3480   4017   4392
```

Improvements vs original (BQ=2, FFTW-only, threshold dispatch):
- Linear engine paths (k≤100): **25-32% faster** (BQ=8 with interleaved a_batch layout)
- Hybrid/tree paths (k=n): **3-6% faster** (vDSP FFT dispatch + calibrated size selection)
- Cost-based engine dispatch replaces fixed K_CROSS thresholds

### 16-thread parallel (ms, uniform stacks, median of 5)

```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
64       0      0      0      0      0      0
128      0      0      0      0      0      0
256      0      0      0      0      0      1
512      0      1      1      1      1      1
1024     0      1      2      2      2      3
2048     1      3      3      6      6      7
4096     2      6      7     12     14     15
8192     4     12     14     28     33     37
16384    9     23     28     68     81     90
32768   17     48     55    170    199    217
65536   39    102    114    437    501    594
```

### Parallel speedup (n=8192 k=n)

```
Threads  Serial  Parallel  Speedup
1        350     —         —
16       —       37        9.5x
```

### 1-second threshold: n ≈ 18,368 (k=n, single-threaded, binary search)

### Dispatch: cost-based `select_engine()`, B=16 (cost-model selected)

---

## AMD Ryzen 9 7950X (Zen 4, AVX-512, BQ=8)

### Single-threaded (ms, uniform stacks)

```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
256      1      2      4      3      4      4
1024     5      9     16     21     23     24
4096     9     19     30    111    120    130
8192    17     36     62    256    283    294
```

### Parallel scaling (k=n)

```
Threads  n=8192  Speedup
1        287     1.0x
2        148     1.9x
4         76     3.8x
8         43     6.7x
16        34     8.4x
```

### 1-second threshold: n ≈ 19,136

### Dispatch: cost-based `select_engine()`, B=32 (cost-model selected)
Note: Zen 4 numbers are pre-BQ=8-interleaved and pre-vDSP. With the interleaved
a_batch layout, Zen 4 should see similar ~30% gains on the linear engine.
Re-benchmark on Zen 4 hardware to get updated numbers.

---

## Head-to-head (single-threaded, n=8192, median of 5)

| k | M3 Max (current) | M3 Max (old) | Zen 4 (old) |
|---|---|---|---|
| k=10 | L:14 | L:20 | L:17 |
| k=50 | L:50 | L:74 | L:36 |
| k=100 | L:115 | H:130 | L:62 |
| k=n/4 | H:287 | H:291 | H:256 |
| k=n/2 | H:318 | H:337 | H:283 |
| k=n | H:350 | H:361 | H:294 |

M3 Max linear engine improved 25-32% (BQ=8 interleaved + vDSP).
Zen 4 numbers are pre-optimization — expect similar linear engine gains
from the interleaved layout (shared template).

## Head-to-head (16-thread, n=8192 k=n)

| Machine | Time | Speedup vs M3 Max |
|---|---|---|
| M3 Max (12P+4E) | 37ms | 1.0x |
| Zen 4 7950X (16P) | 34ms | 1.09x |

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

### M3 Max specific
- vDSP interleaved DFT dispatch (`vDSP_DFT_Interleaved_CreateSetupD`) — 10-18%
  faster FFT at 33 supported sizes (f × 2^g where f ∈ {1,3,5,15}, g ≥ 4).
  Zero format conversion (uses same interleaved complex as FFTW). Forward ×2
  scaling absorbed into pointwise multiply; single ×0.25 on inverse output.
- Calibration table updated with vDSP dispatch times, steering `best_fft_config()`
  to prefer vDSP-supported sizes (e.g. 192 replaces 200 at saturated tree levels).

### Zen 4 specific
- BQ=8 (AVX-512) with same interleaved layout — expects similar linear engine gains
- L2-aware checkpointing with 1MB L2
- B=32 (vs M3 Max B=16) — cost model adapts to Zen 4's wider schoolbook-FFT crossover

## FFT Phase Split (M3 Max)

```
fft_n    fwd(ns)  pw(ns)   ifft(ns) f_fwd  f_pw   f_ifft
64       39       16       46       0.38   0.16   0.46
256      214      61       233      0.42   0.12   0.46
1024     1039     246      1113     0.43   0.10   0.46
4096     5833     942      5338     0.48   0.08   0.44
8192     13183    1963     13960    0.45   0.07   0.48
16384    31920    3851     31846    0.47   0.06   0.47
```
