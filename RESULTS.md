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

## AMD Ryzen 9 7950X (Zen 4, AVX-512, FFTW+MKL dual dispatch)

### Single-threaded (ms, Q=256, uniform stacks, median of 5)

```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
64       0      0      0      0      0      0
128      0      0      1      0      0      1
256      0      1      2      1      2      4
512      1      2      3      4      8     10
1024     1      4      6     20     21     23
2048     3      8     13     44     49     53
4096     7     16     24    102    113    125
8192    14     30     49    237    271    298
16384   26     66    109    577    661    702
32768   58    110    201   1396   1514   1608
65536  114    252    395   3285   3573   3937
```

### 16-thread parallel (ms, Q=256, uniform stacks, median of 5)

```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
64       0      0      0      0      0      0
128      0      0      0      0      0      0
256      0      0      0      0      0      0
512      0      0      1      1      1      1
1024     0      1      1      1      2      2
2048     0      2      2      3      4      4
4096     2      4      4      8      9      9
8192     4      8      8     18     21     24
16384    8     15     17     46     59     67
32768   16     32     58    159    179    212
65536   36    122    197    384    423    551
```

### Parallel speedup (n=8192 k=n)

```
Threads  Serial  Parallel  Speedup
1        298     —         —
16       —       24        12.4x
```

### 1-second threshold: n ≈ 19,904 (k=n, single-threaded)

### Dispatch: cost-based `select_engine()`, B from `select_best_B()` (typically B=32)

### MKL dual dispatch

AOCL-FFTW+MKL per-size best-of-both via `dlopen`. AOCL-FFTW wins 637/749 smooth sizes,
MKL wins 112/749 (mostly small composites 14-64, plus 131072 at 1.15x).
`calib_lib[]` array in `fft_config.h` drives per-plan library selection.

---

## Head-to-head (single-threaded, n=8192, median of 5)


| k     | M3 Max | Zen 4 |
| ----- | ------ | ----- |
| k=10  | L:14   | L:14  |
| k=50  | L:50   | L:30  |
| k=100 | L:115  | L:49  |
| k=n/4 | H:287  | H:237 |
| k=n/2 | H:318  | H:271 |
| k=n   | H:350  | T:298 |


Zen 4 wins at all k values: ~2x faster at small k (AVX-512 linear engine),
12-16% faster at large k (calibrated FFT tree + FFTW PATIENT wisdom).

## Head-to-head (16-thread, n=8192 k=n)


| Machine           | Time | Speedup vs M3 Max |
| ----------------- | ---- | ----------------- |
| M3 Max (12P+4E)   | 37ms | 1.0x              |
| Zen 4 7950X (16P) | 24ms | 1.54x             |


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

- FFTW+MKL dual dispatch via `dlopen` — per-size best-of-both (181/749 sizes use MKL)
- BQ=2 batched linear with interleaved layout (AVX-512 native width)
- L2-aware checkpointing with 1MB per-core L2
- B=32 (vs M3 Max B=16) — cost model adapts to Zen 4's wider schoolbook-FFT crossover
- `MKL_THREADING_LAYER=SEQUENTIAL` set automatically at init (avoids OpenMP dependency)

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

