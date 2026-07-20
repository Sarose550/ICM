# RESULTS.md — ICM Equity Optimization Results

All correctness tests PASS at < 5e-12 relative error (< 1e-9 for Smax=10^9).
Q=256 quadrature points.

## Apple M3 Pro (ARM64, NEON+vDSP, BQ=8)

> Calibrated 2026-07-20 with FFTW PATIENT wisdom on Apple M3 Pro (6P+6E, 12 logical cores).
> All cost-model constants refit from real M3 Pro hardware measurements.
> Engine dispatch: `select_engine()` cost-based, B auto-selected (typically B=16).

### Performance (ms, uniform stacks, median of 5) — M3 Pro

Single-threaded vs 12-thread parallel, per (n, k) cell:

| n | k | serial (ms) | parallel (ms) | speedup |
|---|---|---|---|---|
| 64 | k=10 | 0.0950 | 0.0540 | 1.8x |
| 64 | k=50 | 0.343 | 0.0920 | 3.7x |
| 64 | k=100 | 0.441 | 0.110 | 4.0x |
| 64 | k=n/4 | 0.126 | 0.0570 | 2.2x |
| 64 | k=n/2 | 0.216 | 0.0710 | 3.0x |
| 64 | k=n | 0.440 | 0.120 | 3.7x |
| 128 | k=10 | 0.191 | 0.0780 | 2.4x |
| 128 | k=50 | 0.716 | 0.184 | 3.9x |
| 128 | k=100 | 1.31 | 0.309 | 4.2x |
| 128 | k=n/4 | 0.477 | 0.119 | 4.0x |
| 128 | k=n/2 | 0.896 | 0.204 | 4.4x |
| 128 | k=n | 1.41 | 0.221 | 6.4x |
| 256 | k=10 | 0.413 | 0.117 | 3.5x |
| 256 | k=50 | 1.45 | 0.301 | 4.8x |
| 256 | k=100 | 3.32 | 0.592 | 5.6x |
| 256 | k=n/4 | 1.77 | 0.373 | 4.7x |
| 256 | k=n/2 | 3.38 | 0.492 | 6.9x |
| 256 | k=n | 5.21 | 0.716 | 7.3x |
| 512 | k=10 | 0.859 | 0.191 | 4.5x |
| 512 | k=50 | 3.64 | 0.647 | 5.6x |
| 512 | k=100 | 6.60 | 1.22 | 5.4x |
| 512 | k=n/4 | 7.57 | 1.08 | 7.0x |
| 512 | k=n/2 | 11.3 | 1.59 | 7.1x |
| 512 | k=n | 13.0 | 1.80 | 7.2x |
| 1024 | k=10 | 1.71 | 0.341 | 5.0x |
| 1024 | k=50 | 7.16 | 1.25 | 5.7x |
| 1024 | k=100 | 13.2 | 2.27 | 5.8x |
| 1024 | k=n/4 | 24.4 | 3.36 | 7.3x |
| 1024 | k=n/2 | 28.9 | 3.87 | 7.5x |
| 1024 | k=n | 34.8 | 4.96 | 7.0x |
| 2048 | k=10 | 4.11 | 0.723 | 5.7x |
| 2048 | k=50 | 14.3 | 2.44 | 5.9x |
| 2048 | k=100 | 26.3 | 4.46 | 5.9x |
| 2048 | k=n/4 | 60.9 | 8.01 | 7.6x |
| 2048 | k=n/2 | 75.2 | 10.3 | 7.3x |
| 2048 | k=n | 99.8 | 14.7 | 6.8x |
| 4096 | k=10 | 8.18 | 1.59 | 5.1x |
| 4096 | k=50 | 28.5 | 4.91 | 5.8x |
| 4096 | k=100 | 52.6 | 9.09 | 5.8x |
| 4096 | k=n/4 | 159 | 21.6 | 7.4x |
| 4096 | k=n/2 | 214 | 29.9 | 7.2x |
| 4096 | k=n | 232 | 31.1 | 7.5x |
| 8192 | k=10 | 16.2 | 2.71 | 6.0x |
| 8192 | k=50 | 56.6 | 9.79 | 5.8x |
| 8192 | k=100 | 104 | 17.7 | 5.9x |
| 8192 | k=n/4 | 438 | 60.7 | 7.2x |
| 8192 | k=n/2 | 483 | 67.3 | 7.2x |
| 8192 | k=n | 516 | 73.3 | 7.0x |
| 16384 | k=10 | 32.5 | 5.36 | 6.1x |
| 16384 | k=50 | 113 | 18.6 | 6.1x |
| 16384 | k=100 | 208 | 34.5 | 6.0x |
| 16384 | k=n/4 | 1020 | 147 | 6.9x |
| 16384 | k=n/2 | 1090 | 148 | 7.4x |
| 16384 | k=n | 1480 | 202 | 7.3x |
| 32768 | k=10 | 64.9 | 10.6 | 6.1x |
| 32768 | k=50 | 226 | 37.6 | 6.0x |
| 32768 | k=100 | 418 | 67.8 | 6.2x |
| 32768 | k=n/4 | 2330 | 309 | 7.5x |
| 32768 | k=n/2 | 3090 | 420 | 7.4x |
| 32768 | k=n | 4680 | 606 | 7.7x |
| 65536 | k=10 | 130 | 21.1 | 6.2x |
| 65536 | k=50 | 453 | 74.2 | 6.1x |
| 65536 | k=100 | 836 | 138 | 6.1x |
| 65536 | k=n/4 | 6580 | 855 | 7.7x |
| 65536 | k=n/2 | 9710 | 1270 | 7.6x |
| 65536 | k=n | 10000 | 1310 | 7.6x |

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

> Full serial + parallel grids collected 2026-07-20 with the corrected engine
> dispatch (see `results/bench_grid_zen4_{serial,parallel}_2026-07-20.txt`).

### Performance (ms, uniform stacks, median of 5) — Zen 4

Single-threaded vs 16-thread parallel, per (n, k) cell:

| n | k | serial (ms) | parallel (ms) | speedup |
|---|---|---|---|---|
| 64 | k=10 | 0.0909 | 0.0133 | 6.8x |
| 64 | k=50 | 0.163 | 0.0201 | 8.1x |
| 64 | k=100 | 0.246 | 0.0240 | 10.3x |
| 64 | k=n/4 | 0.0930 | 0.0137 | 6.8x |
| 64 | k=n/2 | 0.125 | 0.0163 | 7.7x |
| 64 | k=n | 0.246 | 0.0248 | 9.9x |
| 128 | k=10 | 0.163 | 0.0188 | 8.7x |
| 128 | k=50 | 0.387 | 0.0399 | 9.7x |
| 128 | k=100 | 0.571 | 0.0593 | 9.6x |
| 128 | k=n/4 | 0.291 | 0.0310 | 9.4x |
| 128 | k=n/2 | 0.393 | 0.0452 | 8.7x |
| 128 | k=n | 0.812 | 0.0763 | 10.6x |
| 256 | k=10 | 0.322 | 0.0357 | 9.0x |
| 256 | k=50 | 0.657 | 0.0755 | 8.7x |
| 256 | k=100 | 1.70 | 0.144 | 11.8x |
| 256 | k=n/4 | 0.822 | 0.0901 | 9.1x |
| 256 | k=n/2 | 2.20 | 0.181 | 12.2x |
| 256 | k=n | 4.82 | 0.356 | 13.5x |
| 512 | k=10 | 0.727 | 0.0629 | 11.6x |
| 512 | k=50 | 1.73 | 0.174 | 9.9x |
| 512 | k=100 | 3.51 | 0.291 | 12.1x |
| 512 | k=n/4 | 4.25 | 0.373 | 11.4x |
| 512 | k=n/2 | 11.0 | 0.732 | 15.0x |
| 512 | k=n | 13.4 | 0.987 | 13.6x |
| 1024 | k=10 | 1.28 | 0.125 | 10.2x |
| 1024 | k=50 | 3.95 | 0.349 | 11.3x |
| 1024 | k=100 | 7.61 | 0.562 | 13.5x |
| 1024 | k=n/4 | 26.5 | 1.94 | 13.7x |
| 1024 | k=n/2 | 29.1 | 2.23 | 13.0x |
| 1024 | k=n | 34.0 | 2.48 | 13.7x |
| 2048 | k=10 | 3.18 | 0.308 | 10.3x |
| 2048 | k=50 | 6.83 | 0.694 | 9.8x |
| 2048 | k=100 | 13.9 | 1.25 | 11.1x |
| 2048 | k=n/4 | 63.0 | 4.51 | 14.0x |
| 2048 | k=n/2 | 73.7 | 5.23 | 14.1x |
| 2048 | k=n | 75.1 | 5.34 | 14.1x |
| 4096 | k=10 | 7.32 | 0.604 | 12.1x |
| 4096 | k=50 | 14.1 | 1.45 | 9.7x |
| 4096 | k=100 | 28.3 | 2.35 | 12.0x |
| 4096 | k=n/4 | 153 | 10.8 | 14.2x |
| 4096 | k=n/2 | 161 | 11.4 | 14.1x |
| 4096 | k=n | 168 | 11.9 | 14.1x |
| 8192 | k=10 | 14.5 | 1.18 | 12.3x |
| 8192 | k=50 | 28.6 | 2.76 | 10.4x |
| 8192 | k=100 | 53.4 | 4.86 | 11.0x |
| 8192 | k=n/4 | 343 | 23.8 | 14.4x |
| 8192 | k=n/2 | 376 | 26.6 | 14.1x |
| 8192 | k=n | 382 | 26.9 | 14.2x |
| 16384 | k=10 | 29.6 | 2.39 | 12.4x |
| 16384 | k=50 | 62.7 | 5.56 | 11.3x |
| 16384 | k=100 | 112 | 10.4 | 10.8x |
| 16384 | k=n/4 | 805 | 57.5 | 14.0x |
| 16384 | k=n/2 | 866 | 67.5 | 12.8x |
| 16384 | k=n | 835 | 81.2 | 10.3x |
| 32768 | k=10 | 63.4 | 5.59 | 11.3x |
| 32768 | k=50 | 116 | 12.4 | 9.4x |
| 32768 | k=100 | 217 | 21.0 | 10.3x |
| 32768 | k=n/4 | 1880 | 194 | 9.7x |
| 32768 | k=n/2 | 1810 | 207 | 8.7x |
| 32768 | k=n | 1840 | 258 | 7.1x |
| 65536 | k=10 | 117 | 19.5 | 6.0x |
| 65536 | k=50 | 244 | 30.7 | 7.9x |
| 65536 | k=100 | 419 | 45.2 | 9.3x |
| 65536 | k=n/4 | 3920 | 431 | 9.1x |
| 65536 | k=n/2 | 4170 | 530 | 7.9x |
| 65536 | k=n | 4490 | 631 | 7.1x |

### Parallel speedup — Zen 4

At the 1-second boundary (from contour sweep, Q=256):

| k | Serial n | Parallel n | Speedup |
|---|----------|------------|---------|
| 2 | 415,040 | 1,611,329 | 3.9x |
| 100 | 136,790 | 1,062,546 | 7.8x |
| 1000 | 27,031 | 181,343 | 6.7x |
| 10000 | 18,750 | 127,187 | 6.8x |
| 13000 | 16,250 | 122,687 | 7.5x |

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
| `FFT_OVERHEAD_NS` | 0.0000 | Per-call FFT overhead — baked into `calib_times_ns[]` (full pipeline), not double-counted. |

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

