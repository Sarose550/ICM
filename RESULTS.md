# RESULTS.md - ICM Equity Optimization Results

All correctness tests PASS at < 5e-12 relative error (< 1e-9 for Smax=10^9).
Q=256 quadrature points.

## Apple M3 Pro (ARM64, NEON+vDSP, BQ=8)

> FFTW PATIENT calibration on Apple M3 Pro (6P+6E, 12 logical cores). `WRAP_FMA_NS`
> and `FP64_DIV_NS` are directly measured via isolated microbenchmarks
> (`tools/bench_wrap_fma.c`, `tools/bench_div_chain.c`) rather than recovered
> from aggregate regression — see [Calibration methodology](#calibration-methodology) below.
> Engine dispatch: `select_engine()` cost-based, B auto-selected (typically B=16).

### Performance (ms, uniform stacks, median of 5) - M3 Pro

Single-threaded vs 12-thread parallel, per (n, k) cell:

| n | k | serial (ms) | parallel (ms) | speedup |
|---|---|---|---|---|
| 64 | k=10 | 0.0980 | 0.0510 | 1.9x |
| 64 | k=50 | 0.341 | 0.0970 | 3.5x |
| 64 | k=100 | 0.440 | 0.115 | 3.8x |
| 64 | k=n/4 | 0.125 | 0.0550 | 2.3x |
| 64 | k=n/2 | 0.219 | 0.0730 | 3.0x |
| 64 | k=n | 0.439 | 0.113 | 3.9x |
| 128 | k=10 | 0.189 | 0.0740 | 2.6x |
| 128 | k=50 | 0.704 | 0.169 | 4.2x |
| 128 | k=100 | 1.31 | 0.275 | 4.8x |
| 128 | k=n/4 | 0.476 | 0.138 | 3.4x |
| 128 | k=n/2 | 0.883 | 0.202 | 4.4x |
| 128 | k=n | 1.65 | 0.348 | 4.7x |
| 256 | k=10 | 0.410 | 0.120 | 3.4x |
| 256 | k=50 | 1.41 | 0.319 | 4.4x |
| 256 | k=100 | 3.27 | 0.619 | 5.3x |
| 256 | k=n/4 | 1.78 | 0.357 | 5.0x |
| 256 | k=n/2 | 4.14 | 0.777 | 5.3x |
| 256 | k=n | 4.94 | 0.689 | 7.2x |
| 512 | k=10 | 0.865 | 0.189 | 4.6x |
| 512 | k=50 | 3.54 | 0.674 | 5.3x |
| 512 | k=100 | 6.55 | 1.19 | 5.5x |
| 512 | k=n/4 | 8.23 | 1.49 | 5.5x |
| 512 | k=n/2 | 11.0 | 1.49 | 7.4x |
| 512 | k=n | 21.6 | 2.90 | 7.4x |
| 1024 | k=10 | 1.71 | 0.329 | 5.2x |
| 1024 | k=50 | 7.07 | 1.28 | 5.5x |
| 1024 | k=100 | 13.1 | 2.26 | 5.8x |
| 1024 | k=n/4 | 23.3 | 3.09 | 7.5x |
| 1024 | k=n/2 | 45.9 | 6.07 | 7.6x |
| 1024 | k=n | 46.0 | 6.03 | 7.6x |
| 2048 | k=10 | 4.07 | 0.703 | 5.8x |
| 2048 | k=50 | 14.2 | 2.48 | 5.7x |
| 2048 | k=100 | 26.2 | 4.46 | 5.9x |
| 2048 | k=n/4 | 94.6 | 14.1 | 6.7x |
| 2048 | k=n/2 | 97.1 | 13.1 | 7.4x |
| 2048 | k=n | 103 | 13.5 | 7.6x |
| 4096 | k=10 | 8.13 | 1.38 | 5.9x |
| 4096 | k=50 | 28.2 | 4.85 | 5.8x |
| 4096 | k=100 | 52.4 | 8.84 | 5.9x |
| 4096 | k=n/4 | 201 | 27.4 | 7.3x |
| 4096 | k=n/2 | 217 | 29.2 | 7.4x |
| 4096 | k=n | 238 | 31.5 | 7.6x |
| 8192 | k=10 | 16.2 | 2.71 | 6.0x |
| 8192 | k=50 | 56.5 | 9.73 | 5.8x |
| 8192 | k=100 | 105 | 17.6 | 6.0x |
| 8192 | k=n/4 | 447 | 60.2 | 7.4x |
| 8192 | k=n/2 | 516 | 68.4 | 7.5x |
| 8192 | k=n | 549 | 73.3 | 7.5x |
| 16384 | k=10 | 32.3 | 5.29 | 6.1x |
| 16384 | k=50 | 113 | 19.5 | 5.8x |
| 16384 | k=100 | 209 | 35.3 | 5.9x |
| 16384 | k=n/4 | 1070 | 142 | 7.5x |
| 16384 | k=n/2 | 1180 | 158 | 7.5x |
| 16384 | k=n | 1260 | 168 | 7.5x |
| 32768 | k=10 | 65.1 | 10.7 | 6.1x |
| 32768 | k=50 | 226 | 38.2 | 5.9x |
| 32768 | k=100 | 418 | 70.1 | 6.0x |
| 32768 | k=n/4 | 2450 | 331 | 7.4x |
| 32768 | k=n/2 | 2700 | 360 | 7.5x |
| 32768 | k=n | 2890 | 388 | 7.4x |
| 65536 | k=10 | 129 | 21.2 | 6.1x |
| 65536 | k=50 | 450 | 78.8 | 5.7x |
| 65536 | k=100 | 836 | 139 | 6.0x |
| 65536 | k=n/4 | 5600 | 752 | 7.4x |
| 65536 | k=n/2 | 6240 | 852 | 7.3x |
| 65536 | k=n | 7180 | 993 | 7.2x |

### Parallel speedup - M3 Pro

At the 1-second boundary (from contour sweep, Q=256; earlier sweep, not re-run this session):

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

> Contour data from an earlier sweep — may shift slightly with the corrected
> calibration. Grid numbers above are authoritative.

### Dispatch: cost-based `select_engine()`, B from `select_best_B()` (typically B=16). Linear→hybrid crossover at k≈140.

---

## AMD Ryzen 9 7950X (Zen 4, AVX-512, AOCL-FFTW)

> AOCL-FFTW (AMD's official znver4-tuned build, tag 5.3) is the sole FFT backend.
> A direct A/B test confirmed AOCL is cleanly faster than plain system FFTW at every
> calibrated size — no dual dispatch. All numbers below are from a box running under
> the `performance` cpufreq governor (16 physical cores, SMT off for benchmarking,
> `OMP_NUM_THREADS=16` for parallel). `WRAP_FMA_NS=0.40` is directly measured via
> `tools/bench_wrap_fma.c` — see [Calibration methodology](#calibration-methodology).

### Performance (ms, uniform stacks, median of 5) - Zen 4

Single-threaded vs 16-thread parallel, per (n, k) cell:

| n | k | serial (ms) | parallel (ms) | speedup |
|---|---|---|---|---|
| 64 | k=10 | 0.0945 | 0.0132 | 7.2x |
| 64 | k=50 | 0.202 | 0.0223 | 9.1x |
| 64 | k=100 | 0.190 | 0.0273 | 7.0x |
| 64 | k=n/4 | 0.105 | 0.0149 | 7.0x |
| 64 | k=n/2 | 0.143 | 0.0183 | 7.8x |
| 64 | k=n | 0.190 | 0.0237 | 8.0x |
| 128 | k=10 | 0.181 | 0.0197 | 9.2x |
| 128 | k=50 | 0.358 | 0.0359 | 10.0x |
| 128 | k=100 | 0.617 | 0.0617 | 10.0x |
| 128 | k=n/4 | 0.271 | 0.0286 | 9.5x |
| 128 | k=n/2 | 0.407 | 0.0448 | 9.1x |
| 128 | k=n | 0.795 | 0.0808 | 9.8x |
| 256 | k=10 | 0.369 | 0.0358 | 10.3x |
| 256 | k=50 | 0.726 | 0.0792 | 9.2x |
| 256 | k=100 | 1.69 | 0.143 | 11.8x |
| 256 | k=n/4 | 0.845 | 0.0916 | 9.2x |
| 256 | k=n/2 | 2.05 | 0.183 | 11.2x |
| 256 | k=n | 4.42 | 0.341 | 13.0x |
| 512 | k=10 | 0.719 | 0.0651 | 11.0x |
| 512 | k=50 | 1.97 | 0.175 | 11.3x |
| 512 | k=100 | 3.69 | 0.289 | 12.8x |
| 512 | k=n/4 | 4.76 | 0.386 | 12.3x |
| 512 | k=n/2 | 9.47 | 0.740 | 12.8x |
| 512 | k=n | 10.6 | 0.807 | 13.1x |
| 1024 | k=10 | 1.33 | 0.131 | 10.2x |
| 1024 | k=50 | 3.53 | 0.346 | 10.2x |
| 1024 | k=100 | 7.07 | 0.578 | 12.2x |
| 1024 | k=n/4 | 20.8 | 1.56 | 13.3x |
| 1024 | k=n/2 | 24.2 | 1.80 | 13.4x |
| 1024 | k=n | 25.9 | 1.90 | 13.6x |
| 2048 | k=10 | 3.92 | 0.298 | 13.2x |
| 2048 | k=50 | 7.10 | 0.706 | 10.1x |
| 2048 | k=100 | 13.4 | 1.18 | 11.4x |
| 2048 | k=n/4 | 52.1 | 3.81 | 13.7x |
| 2048 | k=n/2 | 56.7 | 4.18 | 13.6x |
| 2048 | k=n | 60.4 | 4.42 | 13.7x |
| 4096 | k=10 | 8.00 | 0.607 | 13.2x |
| 4096 | k=50 | 16.8 | 1.48 | 11.4x |
| 4096 | k=100 | 28.6 | 2.34 | 12.2x |
| 4096 | k=n/4 | 121 | 8.85 | 13.7x |
| 4096 | k=n/2 | 133 | 9.81 | 13.6x |
| 4096 | k=n | 150 | 10.9 | 13.8x |
| 8192 | k=10 | 16.2 | 1.24 | 13.1x |
| 8192 | k=50 | 28.8 | 2.90 | 9.9x |
| 8192 | k=100 | 49.6 | 5.11 | 9.7x |
| 8192 | k=n/4 | 285 | 20.4 | 14.0x |
| 8192 | k=n/2 | 341 | 24.9 | 13.7x |
| 8192 | k=n | 353 | 25.4 | 13.9x |
| 16384 | k=10 | 26.2 | 3.14 | 8.3x |
| 16384 | k=50 | 56.7 | 5.82 | 9.7x |
| 16384 | k=100 | 111 | 9.85 | 11.3x |
| 16384 | k=n/4 | 730 | 64.0 | 11.4x |
| 16384 | k=n/2 | 771 | 60.2 | 12.8x |
| 16384 | k=n | 843 | 78.1 | 10.8x |
| 32768 | k=10 | 52.2 | 5.51 | 9.5x |
| 32768 | k=50 | 126 | 11.7 | 10.8x |
| 32768 | k=100 | 236 | 19.3 | 12.2x |
| 32768 | k=n/4 | 1640 | 155 | 10.6x |
| 32768 | k=n/2 | 1840 | 179 | 10.3x |
| 32768 | k=n | 1970 | 208 | 9.5x |
| 65536 | k=10 | 116 | 19.6 | 5.9x |
| 65536 | k=50 | 237 | 29.2 | 8.1x |
| 65536 | k=100 | 475 | 52.3 | 9.1x |
| 65536 | k=n/4 | 3890 | 408 | 9.5x |
| 65536 | k=n/2 | 4360 | 450 | 9.7x |
| 65536 | k=n | 4610 | 551 | 8.4x |

### Parallel speedup - Zen 4

At the 1-second boundary (from contour sweep, Q=256; earlier sweep, not re-run this session):

| k | Serial n | Parallel n | Speedup |
|---|----------|------------|---------|
| 2 | 415,040 | 1,611,329 | 3.9x |
| 100 | 136,790 | 1,062,546 | 7.8x |
| 1000 | 27,031 | 181,343 | 6.7x |
| 10000 | 18,750 | 127,187 | 6.8x |
| 13000 | 16,250 | 122,687 | 7.5x |

### 1-second threshold: n ≈ 17,216 (k=n, single-threaded)

> Contour data from an earlier sweep — may shift slightly with the corrected
> calibration. Grid numbers above are authoritative.

### Dispatch: cost-based `select_engine()`, B from `select_best_B()` (typically B=32). Linear→hybrid crossover at k≈275.

### AOCL-FFTW: sole backend, no dual dispatch

AOCL-FFTW (AMD's official znver4-tuned build) is the only FFT backend for Zen 4.
A direct A/B test at n=32768,k=n confirmed AOCL is 20–25% faster than plain system
FFTW at the raw kernel level, reproducible across repeated runs. Per-level FFT-size
selection uses `best_fft_config()` driven by `calib_times_ns[]` (749 calibrated sizes,
AOCL PATIENT wisdom). No `calib_lib[]` array exists — the earlier claim of
"AOCL-FFTW+MKL dual dispatch, 637 vs 112 sizes" in prior versions of this document
traced to measurements on a different box that never had AOCL-FFTW installed.

---

## Key optimizations by device

### Both platforms

- FFTW PATIENT wisdom + MEASURE|WISDOM_ONLY for clones
- Paired cached correlate (shares FFT(g) + cached FFT(P))
- Cost-model-driven B selection (`select_best_B`)
- Shared tree_build_levels / tree_propagate_g helpers
- BQ=8 batched linear with interleaved a_batch layout
(`a_batch[j*BQ+qi]` - cache-friendly, eliminates L1 misses at all n).
Template in `src/linear_batched_impl.inc`.
- L2-aware checkpointing (`ckpt_interval_batched`)
- Cost-based engine dispatch (`select_engine`) - no fixed K_CROSS thresholds
- Cross-correlation wrap correction handles both output-wrap and input-wrap
cyclic aliasing (corrects a pre-existing bug with wrap_m > 0)

### M3 Pro / Apple Silicon specific

- vDSP interleaved DFT dispatch (`vDSP_DFT_Interleaved_CreateSetupD`) - 10-18%
faster FFT at 33 supported sizes (f × 2^g where f ∈ {1,3,5,15}, g ≥ 4).
Zero format conversion (uses same interleaved complex as FFTW). Forward ×2
scaling absorbed into pointwise multiply; single ×0.25 on inverse output.
- Calibration table updated with vDSP dispatch times, steering `best_fft_config()`
to prefer vDSP-supported sizes (e.g. 192 replaces 200 at saturated tree levels).

### Zen 4 specific

- AOCL-FFTW (znver4-tuned, tag 5.3) — sole FFT backend, 20–25% faster than plain FFTW
- BQ=8 batched linear with interleaved layout (native AVX-512 width)
- L2-aware checkpointing with 1MB per-core L2
- B=32 (vs M3 Pro B=16) — cost model adapts to Zen 4's wider schoolbook-FFT crossover.
  Validated empirically: B=32 is optimal in 98.9% of tested cells up to n=16384
  (single mismatch at n=8192,k=350 is noise-level, 1.1%) — see `results/b_optimal_report_zen4.md`

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
and were refit on real hardware via `tools/fit_cost_model.py`.

### M3 Pro (Apple Silicon, ARM64)

| Constant | Value | Notes |
|---|---|---|
| `FMA_NS` | 0.0500 | Scalar FMA cost. Fit lower bound — hit its limit when `WRAP_FMA_NS` and `FP64_DIV_NS` were pinned; see caveat below. |
| `WRAP_FMA_NS` | 0.4942 | Per-FMA cost for wrap correction. **Directly measured** via `tools/bench_wrap_fma.c`. |
| `FP64_DIV_NS` | 3.4890 | FP64 divide latency. **Directly measured** via `tools/bench_div_chain.c` (dependency-chained, not throughput). |
| `BLOCK_FMA_NS` | 0.4027 | FMA cost inside block build/divide. 7-param fit (both pins active). |
| `BLOCK_MEM_NS` | 0.1000 | Memory cost per element in block build/divide. |
| `PAIRED_CACHED_CORR_RATIO` | 1.9080 | Paired cached correlate cost / full FFT pipeline cost. |
| `INDEP_PAIR_RATIO` | 1.9080 | Independent pair correlate cost / full FFT pipeline cost. Equal to PAIRED — likely a fitting artifact (solver couldn't separate them). |
| `LEAF_FMA_NS` | 0.0727 | FMA cost at tree-leaf schoolbook multiplies. 7-param fit. |
| `LEAF_BLOCK_NS` | 48.1032 | Per-block overhead at leaf level. |
| `FFT_OVERHEAD_NS` | 631.0974 | Per-call FFT overhead. Physically odd value — pushed here to compensate when both pins are active; see caveat. |

> FFT calibration table (`calib_sizes[]`/`calib_times_ns[]`) and FFTW wisdom
> in `devices/m3_pro/fft_config.h` are from a genuine FFTW PATIENT calibration
> on this Apple M3 Pro machine (July 2026). `WRAP_FMA_NS` and `FP64_DIV_NS` are
> direct microbenchmark measurements, not recovered from aggregate regression —
> both were unidentifiable from the indirect fit alone (the regression converged
> to physically implausible values: 0.1ns and 0.5ns respectively, both hitting
> their fit lower bounds). Pinning both raises the fit's RMS log-relative error
> to 10.2% and pushes `FFT_OVERHEAD_NS`/`FMA_NS` to compensate — a collinearity
> limitation in the current `sample_plans` training data, not a correctness
> issue. `./bench_grid verify` passes ALL TESTS and `./bench_grid crossover`
> shows a clean, monotonic linear→hybrid transition at k≈140.

### Zen 4 (AMD Ryzen 9 7950X, AVX-512, AOCL-FFTW)

| Constant | Value | Notes |
|---|---|---|
| `FMA_NS` | 0.0793 | Scalar FMA cost. 8-param fit (only `WRAP_FMA_NS` pinned). |
| `WRAP_FMA_NS` | 0.40 | Per-FMA cost for wrap correction. **Directly measured** via `tools/bench_wrap_fma.c` — extracted as least-squares slope over the decision-relevant range `wrap_m ∈ [64,384]`. |
| `FP64_DIV_NS` | 12.5287 | FP64 divide latency. From the unpinned 8-param fit — not independently cross-checked against a direct measurement this session. |
| `BLOCK_FMA_NS` | 0.6833 | FMA cost inside block build/divide (sequential dependency chain, latency- not throughput-bound). |
| `BLOCK_MEM_NS` | 0.1 | Memory cost per element in block build/divide. |
| `PAIRED_CACHED_CORR_RATIO` | 1.8287 | Paired cached correlate cost / full FFT pipeline cost. |
| `INDEP_PAIR_RATIO` | 1.8287 | Independent pair correlate cost / full FFT pipeline cost. |
| `LEAF_FMA_NS` | 0.1610 | FMA cost at tree-leaf schoolbook multiplies. 7-param fit. |
| `LEAF_BLOCK_NS` | 61.3029 | Per-block overhead at leaf level. |
| `FFT_OVERHEAD_NS` | 0.0 | Per-call FFT overhead (baked into `calib_times_ns[]`, not double-counted). |

> Calibration table (`calib_sizes[]`/`calib_times_ns[]`, 749 entries) and
> AOCL-FFTW PATIENT wisdom in `devices/zen4/fft_config.h` are from an AMD
> Ryzen 9 7950X (same SKU as the benchmark machine). `WRAP_FMA_NS` was
> directly measured after the indirect fit proved it unidentifiable from
> aggregate `sample_plans` data (the old fit value 0.8612 was arbitrary —
> wrap-correction cost never exceeds 1.5% of any sampled plan's total time,
> a "persistency of excitation" failure). Fixing this constant (and unifying
> the code-level `FMA_NS`/`WRAP_FMA_NS` mismatch in the planner) produced a
> 2.35× speedup on the previously-regressed n=32768,k=n cell with no
> regressions across spot-checks. `./bench_grid verify`: ALL TESTS PASSED.

### Zen 4 bandwidth constants — known measurement bug

`devices/zen4/fft_config.h` contains `L2_BW_GBS=341868.5` and `L3_BW_GBS=3233.3`,
both physically impossible (hundreds of TB/s for L2). This bug is pre-existing
(confirmed in the commit before this sprint started) and was **not fixed this
session**. These constants feed `blended_bandwidth()` in `src/cost_model.h`,
which affects `select_engine()` dispatch cost for the linear engine — this
could bias dispatch toward linear for L2-resident working sets, independent of
everything else addressed this sprint. Does not affect correctness
(`./bench_grid verify` is unaffected). Real follow-up work for a future session.

---

## Calibration methodology

This sprint established a direct-microbenchmark calibration pipeline that
replaces the previous indirect-aggregate-regression approach for two constants
that proved unidentifiable from aggregate timing data alone.

### The problem: persistency of excitation

The cost model has 9 free parameters fitted against per-plan measured times
from `tools/sample_plans.c`. Two of them — `WRAP_FMA_NS` (wrap-correction FMA
cost) and `FP64_DIV_NS` (dependency-chained FP64 division latency) — each
contribute at most ~1.5% of any single sampled plan's total predicted time.
In control-theory / system-identification terms, the training signal doesn't
vary these parameters' effects enough to be recoverable — a "persistency of
excitation" failure. The regression converges to arbitrary values within a wide
flat basin, not to physically meaningful ones.

This is the same class of problem FFTW solves by timing plans directly (PATIENT
mode) rather than fitting a global model, ATLAS/AEOS solves by per-kernel
empirical timing, and the roofline model solves with dedicated bandwidth/FLOP
microbenchmarks — all cite direct isolated measurement over indirect aggregate
regression for exactly this reason.

### The fix: direct isolated microbenchmarks

- **`WRAP_FMA_NS`**: measured via `tools/bench_wrap_fma.c` — a verbatim copy of
  the wrap-correction loop body run in isolation, sweeping `wrap_m` over a wide
  range so the correction dominates measured time by construction. The value is
  extracted as a least-squares **slope** of time vs. FMA count over the
  decision-relevant range (cancels fixed per-call overhead). The measured curve
  shows a real, physically-explicable cache-hierarchy transition: marginal cost
  rises smoothly from near-FMA-throughput at small working sets to
  memory-latency-bound at large ones. R²=0.9998 on Zen4.
- **`FP64_DIV_NS`**: measured via `tools/bench_div_chain.c` — a
  dependency-chained microbenchmark that reproduces the actual usage pattern
  (leaf extraction's synthetic-division recurrence). Critically, this is NOT an
  independent/vectorizable division loop — that would measure throughput, a
  very different and wrong number for this sequential-dependency-chain usage.

Both tools are wired into `tools/calibrate_full.sh` as standard pipeline steps
for all future device ports.

### Known limitation

Pinning both constants can raise the fit's RMS log-relative error if the
`sample_plans` training data doesn't cleanly separate their effects from
other parameters. Observed on M3 Pro (10.2% RMS error with both pinned, vs.
6.57% unpinned; `FFT_OVERHEAD_NS` pushed to a physically odd 631ns to
compensate). This is a collinearity limitation in the current training-data
coverage, not a correctness issue — `./bench_grid verify` still passes ALL
TESTS and dispatch decisions remain sound. Improving `sample_plans.c`'s B/n
coverage to break this collinearity is flagged as real, open follow-up work.

---

## NVIDIA B200 GPU (sm_100, cuFFTDx fused kernels, CUDA graph capture)

> The linear engine is CPU-only (sequential player-by-player structure can't
> saturate GPU parallelism). Only the tree-based engines map to the GPU; the
> planner assigns each subproduct-tree level to one of three kernel tiers
> (schoolbook, cuFFTDx fused, batched cuFFT) based on polynomial degree.

### Performance (ms, Q=256, FP64): systematic (n, k) grid

| n | k=64 | k=1024 | k=n/2 | k=n |
|---|------|--------|-------|-----|
| 4,096 | 0.37 | 0.75 | 0.82 | 0.86 |
| 16,384 | 1.19 | 2.86 | 4.07 | 4.37 |
| 65,536 | 4.40 | 10.83 | 19.85 | 20.64 |
| 262,144 | 17.14 | 42.21 | 97.60 | 101.3 |
| 1,048,576 | 68.09 | 167.34 | 683.06 | 687.67 |
| 4,194,304 | 273.28 | 873.28 | 2475.64 | 2500.45 |

Sampled from the 211-point calibration heatmap (`results/gpu_heatmap_b200.csv`).

### Frontier probes (dedicated max-n / max-field search, `tools/push_limit_gpu.cu`)

These are not part of the systematic grid above -- they're the specific `n`
values a binary search landed on to pin down the 1-second and 626ms
boundaries.

| n | k | Time (ms) |
|---|---|-----------|
| 1,441,792 | n | 866 |
| 1,572,864 | n | 1,148 |
| 6,291,456 | 100 | 626.3 |
| 8,388,608 | 100 | 1,235 |
| 16,777,216 | 10 | 2,592 |

### 1-second threshold: n ≈ 1,441,792 (k=n), n ≈ 6,291,456 (k=100)

### Dispatch: three-tier kernel planner (schoolbook / cuFFTDx fused / batched cuFFT), cost-based per tree level

GPU cost-model constants (`C_wrap`, `C_school`, `R`, `C_gap`) are fit
separately from the CPU model via `tools/fit_gpu_cost_model.py` against
empirical kernel benchmarks in `devices/b200/gpu_fft_config.h` -- see
"GPU Cost Model (B200)" in `OPTIMIZATION_GUIDE.md` for the full pipeline.

> **Diagnostic pass (July 2026):** The GPU planner was confirmed NOT to have the
> CPU's wrap-correction cost-model bug. `src/gpu/gpu_plan.cu` uses one constant
> (`GPU_SCHOOL_FMA_NS`) uniformly in both joint and independent paths — no
> code-level asymmetry. Additionally, the GPU's fitted `C_wrap` is
> diagnostic-only (`fit_gpu_cost_model.py` never writes it to any config
> header), so even if under-identified it has zero effect on real planning.
> No GPU numbers changed this session.

