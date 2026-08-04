# RESULTS.md: ICM Equity Optimization Results

All correctness tests pass at < 1.6e-10 relative error.
Q=256 quadrature points.

## Apple M3 Pro (ARM64, NEON+vDSP, BQ=8)

> FFTW PATIENT calibration on Apple M3 Pro (6P+6E, 12 logical cores). `WRAP_FMA_NS`
> and `FP64_DIV_NS` are directly measured via isolated microbenchmarks
> (`tools/bench_wrap_fma.c`, `tools/bench_div_chain.c`) rather than recovered
> from aggregate regression, see [Calibration methodology](#calibration-methodology) below.
> Engine dispatch: `select_engine()` cost-based, B auto-selected (typically B=32).

### Performance (ms, uniform stacks, median of 5): M3 Pro

Single-threaded vs 12-thread parallel, per (n, k) cell:

| n | k | serial (ms) | parallel (ms) | speedup |
|---|---|---|---|---|
| 64 | k=10 | 0.0950 | 0.0600 | 1.6x |
| 64 | k=50 | 0.340 | 0.104 | 3.3x |
| 64 | k=100 | 0.438 | 0.109 | 4.0x |
| 64 | k=n/4 | 0.125 | 0.0560 | 2.2x |
| 64 | k=n/2 | 0.226 | 0.0720 | 3.1x |
| 64 | k=n | 0.447 | 0.107 | 4.2x |
| 128 | k=10 | 0.189 | 0.0640 | 3.0x |
| 128 | k=50 | 0.703 | 0.157 | 4.5x |
| 128 | k=100 | 1.29 | 0.252 | 5.1x |
| 128 | k=n/4 | 0.485 | 0.124 | 3.9x |
| 128 | k=n/2 | 0.885 | 0.184 | 4.8x |
| 128 | k=n | 1.42 | 0.224 | 6.3x |
| 256 | k=10 | 0.410 | 0.118 | 3.5x |
| 256 | k=50 | 1.42 | 0.288 | 4.9x |
| 256 | k=100 | 3.29 | 0.527 | 6.2x |
| 256 | k=n/4 | 1.76 | 0.341 | 5.2x |
| 256 | k=n/2 | 3.23 | 0.470 | 6.9x |
| 256 | k=n | 3.65 | 0.521 | 7.0x |
| 512 | k=10 | 0.850 | 0.182 | 4.7x |
| 512 | k=50 | 3.54 | 0.645 | 5.5x |
| 512 | k=100 | 6.51 | 1.01 | 6.4x |
| 512 | k=n/4 | 6.94 | 0.945 | 7.3x |
| 512 | k=n/2 | 8.35 | 1.13 | 7.4x |
| 512 | k=n | 11.2 | 1.57 | 7.1x |
| 1024 | k=10 | 1.71 | 0.319 | 5.4x |
| 1024 | k=50 | 7.05 | 1.25 | 5.6x |
| 1024 | k=100 | 13.0 | 2.03 | 6.4x |
| 1024 | k=n/4 | 17.8 | 2.35 | 7.6x |
| 1024 | k=n/2 | 20.9 | 2.79 | 7.5x |
| 1024 | k=n | 29.5 | 3.73 | 7.9x |
| 2048 | k=10 | 4.80 | 0.703 | 6.8x |
| 2048 | k=50 | 14.5 | 2.45 | 5.9x |
| 2048 | k=100 | 26.1 | 4.00 | 6.5x |
| 2048 | k=n/4 | 45.3 | 5.79 | 7.8x |
| 2048 | k=n/2 | 51.4 | 6.77 | 7.6x |
| 2048 | k=n | 55.8 | 7.29 | 7.7x |
| 4096 | k=10 | 8.14 | 1.36 | 6.0x |
| 4096 | k=50 | 28.1 | 4.93 | 5.7x |
| 4096 | k=100 | 51.9 | 8.65 | 6.0x |
| 4096 | k=n/4 | 108 | 14.4 | 7.5x |
| 4096 | k=n/2 | 122 | 16.6 | 7.3x |
| 4096 | k=n | 135 | 18.4 | 7.3x |
| 8192 | k=10 | 16.2 | 2.72 | 6.0x |
| 8192 | k=50 | 56.2 | 9.62 | 5.8x |
| 8192 | k=100 | 105 | 16.4 | 6.4x |
| 8192 | k=n/4 | 280 | 38.0 | 7.4x |
| 8192 | k=n/2 | 298 | 41.0 | 7.3x |
| 8192 | k=n | 320 | 47.2 | 6.8x |
| 16384 | k=10 | 32.3 | 5.40 | 6.0x |
| 16384 | k=50 | 116 | 19.0 | 6.1x |
| 16384 | k=100 | 208 | 32.7 | 6.4x |
| 16384 | k=n/4 | 625 | 86.6 | 7.2x |
| 16384 | k=n/2 | 701 | 97.1 | 7.2x |
| 16384 | k=n | 751 | 105 | 7.2x |
| 32768 | k=10 | 64.9 | 10.8 | 6.0x |
| 32768 | k=50 | 227 | 38.2 | 5.9x |
| 32768 | k=100 | 415 | 68.9 | 6.0x |
| 32768 | k=n/4 | 1470 | 208 | 7.1x |
| 32768 | k=n/2 | 1650 | 238 | 6.9x |
| 32768 | k=n | 1760 | 261 | 6.7x |
| 65536 | k=10 | 130 | 21.9 | 5.9x |
| 65536 | k=50 | 451 | 79.2 | 5.7x |
| 65536 | k=100 | 830 | 140 | 5.9x |
| 65536 | k=n/4 | 3460 | 512 | 6.8x |
| 65536 | k=n/2 | 3820 | 575 | 6.6x |
| 65536 | k=n | 4060 | 627 | 6.5x |

### Parallel speedup: M3 Pro

At the 1-second boundary (from regenerated contour sweep, Q=256):

| k | Serial n | Parallel n | Speedup |
|---|----------|------------|---------|
| 2 | 1,025,391 | 5,273,438 | 5.1x |
| 100 | 78,209 | 468,756 | 6.0x |
| 1000 | 36,218 | 219,202 | 6.1x |
| 10000 | 20,312 | 117,812 | 5.8x |
| 13000 | 18,687 | 119,640 | 6.4x |

Speedup varies by k due to engine dispatch: linear-only k values see ~5-6x (simple SIMD scaling),
while hybrid-engine k values reach ~5.8-6.4x at this 1-second boundary (FFT tree parallelism); the
full performance grid above shows hybrid cells peaking around 7.5-7.9x at more moderate n. M3 Pro's
6P+6E topology limits peak parallel speedup well below Zen 4's ~14x on 16 homogeneous P-cores.

### 1-second threshold: n = 18,368 (k=n, single-threaded), n = 88,064 (k=n, 12-thread)

Real binary search (`bench_grid threshold`; `results/threshold_m3pro_serial.txt`,
`results/threshold_m3pro_parallel.txt`), not an interpolation from grid points.
Past sessions repeatedly substituted an interpolated estimate here, which shipped
a materially wrong number more than once (see HANDOFF.md). This run post-dates
the M3 Pro calibration extension to 262,144 (905 sizes) and the wrap-safety-margin
fix, so it reflects the corrected dispatch behavior, not the pre-fix
catastrophic-wrap regime that affected n≈89,600-90,112 before the 2026-08-03 fixes.

These specific numbers were measured with a single sample per candidate (the
`threshold` binary search's original methodology). A later check found that
measuring with real repetition (median of 5, matching this project's
usual convention) shifts the M3 Pro serial figure down to roughly n=16,256:
directly confirmed to be sustained thermal throttling under back-to-back
near-1-second computations (not a bug: 8 consecutive reps at one fixed n show a
clear upward time drift), not a change in what's actually being measured. The
`threshold` subcommand itself has since been fixed to take a real median of 5
per candidate (matching every other benchmark in this project) for future runs;
the headline numbers above are kept as originally measured, single-sample,
cold-start figures.

### Dispatch: cost-based `select_engine()`, B from `select_best_B()` (typically B=32). Linear→hybrid crossover at k≈122-124 (empirical crossover table in `devices/m3_pro/fft_config.h`).

---

## AMD Ryzen 9 7950X (Zen 4, AVX-512, AOCL-FFTW)

> **Reference hardware**: bare-metal Ryzen 9 7950X, 16 physical / 32 logical cores,
> 128 GB (4 x 32 GB, so 2 DIMMs per channel). AOCL-FFTW (AMD's official
> znver4-tuned build, tag 5.3) is the sole FFT backend, built from source with the
> documented AM5 flag set. A direct A/B test confirmed AOCL is cleanly faster than
> plain system FFTW at every calibrated size, no dual dispatch. All numbers below
> are under the `performance` cpufreq governor (verified by reading every
> `scaling_governor` back, not just issuing the command), with
> `OMP_NUM_THREADS=16`, one thread per **physical** core.
> `WRAP_FMA_NS=0.4360` is directly measured via `tools/bench_wrap_fma.c`, see
> [Calibration methodology](#calibration-methodology).
>
> **This box's RAM runs at 3600 MT/s against a 4800 MT/s DIMM rating**, the AMD
> AM5 2-DIMMs-per-channel (2DPC) electrical limit, confirmed via `dmidecode`
> (4x 32GB, Configured Memory Speed 3600 MT/s) and not fixable at the OS/BIOS
> level. This is a permanent characteristic of the reference hardware, not a
> temporary anomaly. Measured streaming DRAM bandwidth (`tools/calibrate.c`'s
> own bandwidth phase) is 32.7 GB/s, consistent with genuine 3600 MT/s 2DPC
> operation.
>
> This is the project's standing Zen4 reference; all numbers below are measured
> directly on it. (A prior box used earlier in this project ran the same
> nominal configuration but measured slower on hybrid/FFT-heavy cells; that
> discrepancy was never resolved and isn't relevant to the numbers below.
> See `VERDICTS.md` if the history is relevant.)

### Performance (ms, uniform stacks, median of 5): Zen 4 (current reference)

Single-threaded vs 16-thread parallel, per (n, k) cell. Source:
`results/bench_grid_zen4_serial.txt`, `results/bench_grid_zen4_parallel.txt`
(undated = current, per this repo's convention).

| n | k | serial (ms) | parallel (ms) | speedup |
|---|---|---|---|---|
| 64 | k=10 | 0.0946 | 0.0120 | 7.9x |
| 64 | k=50 | 0.190 | 0.0225 | 8.4x |
| 64 | k=100 | 0.229 | 0.0237 | 9.7x |
| 64 | k=n/4 | 0.0950 | 0.0137 | 6.9x |
| 64 | k=n/2 | 0.127 | 0.0174 | 7.3x |
| 64 | k=n | 0.242 | 0.0234 | 10.3x |
| 128 | k=10 | 0.167 | 0.0203 | 8.2x |
| 128 | k=50 | 0.327 | 0.0403 | 8.1x |
| 128 | k=100 | 0.573 | 0.0635 | 9.0x |
| 128 | k=n/4 | 0.266 | 0.0312 | 8.5x |
| 128 | k=n/2 | 0.391 | 0.0445 | 8.8x |
| 128 | k=n | 0.740 | 0.0962 | 7.7x |
| 256 | k=10 | 0.329 | 0.0416 | 7.9x |
| 256 | k=50 | 0.672 | 0.0833 | 8.1x |
| 256 | k=100 | 1.60 | 0.172 | 9.3x |
| 256 | k=n/4 | 0.889 | 0.0938 | 9.5x |
| 256 | k=n/2 | 2.14 | 0.213 | 10.0x |
| 256 | k=n | 3.36 | 0.265 | 12.7x |
| 512 | k=10 | 0.646 | 0.0656 | 9.8x |
| 512 | k=50 | 1.99 | 0.179 | 11.1x |
| 512 | k=100 | 3.55 | 0.300 | 11.8x |
| 512 | k=n/4 | 4.17 | 0.383 | 10.9x |
| 512 | k=n/2 | 7.37 | 0.549 | 13.4x |
| 512 | k=n | 7.91 | 0.579 | 13.7x |
| 1024 | k=10 | 1.33 | 0.129 | 10.3x |
| 1024 | k=50 | 3.53 | 0.350 | 10.1x |
| 1024 | k=100 | 7.54 | 0.610 | 12.4x |
| 1024 | k=n/4 | 15.8 | 1.11 | 14.2x |
| 1024 | k=n/2 | 17.3 | 1.25 | 13.8x |
| 1024 | k=n | 19.7 | 1.49 | 13.2x |
| 2048 | k=10 | 3.11 | 0.305 | 10.2x |
| 2048 | k=50 | 6.99 | 0.689 | 10.1x |
| 2048 | k=100 | 13.9 | 1.19 | 11.7x |
| 2048 | k=n/4 | 36.3 | 2.59 | 14.0x |
| 2048 | k=n/2 | 39.3 | 2.86 | 13.7x |
| 2048 | k=n | 45.0 | 3.18 | 14.2x |
| 4096 | k=10 | 6.21 | 0.615 | 10.1x |
| 4096 | k=50 | 15.0 | 1.40 | 10.7x |
| 4096 | k=100 | 28.5 | 2.34 | 12.2x |
| 4096 | k=n/4 | 82.6 | 5.92 | 14.0x |
| 4096 | k=n/2 | 95.0 | 6.69 | 14.2x |
| 4096 | k=n | 103 | 7.40 | 13.9x |
| 8192 | k=10 | 12.6 | 1.25 | 10.1x |
| 8192 | k=50 | 31.5 | 3.37 | 9.3x |
| 8192 | k=100 | 57.9 | 5.02 | 11.5x |
| 8192 | k=n/4 | 194 | 13.8 | 14.1x |
| 8192 | k=n/2 | 206 | 15.7 | 13.1x |
| 8192 | k=n | 232 | 26.9 | 8.6x |
| 16384 | k=10 | 25.5 | 2.47 | 10.3x |
| 16384 | k=50 | 74.0 | 5.49 | 13.5x |
| 16384 | k=100 | 127 | 9.73 | 13.1x |
| 16384 | k=n/4 | 442 | 85.8 | 5.2x |
| 16384 | k=n/2 | 484 | 116 | 4.2x |
| 16384 | k=n | 525 | 157 | 3.3x |
| 32768 | k=10 | 52.9 | 5.61 | 9.4x |
| 32768 | k=50 | 118 | 11.8 | 10.0x |
| 32768 | k=100 | 252 | 18.9 | 13.3x |
| 32768 | k=n/4 | 991 | 247 | 4.0x |
| 32768 | k=n/2 | 1060 | 275 | 3.9x |
| 32768 | k=n | 1240 | 389 | 3.2x |
| 65536 | k=10 | 107 | 20.2 | 5.3x |
| 65536 | k=50 | 252 | 30.1 | 8.4x |
| 65536 | k=100 | 727 | 45.7 | 15.9x |
| 65536 | k=n/4 | 2570 | 606 | 4.2x |
| 65536 | k=n/2 | 2850 | 701 | 4.1x |
| 65536 | k=n | 3160 | 862 | 3.7x |

### Parallel speedup: Zen 4 (current reference)

At the 1-second boundary (from contour sweep, Q=256, `OMP_NUM_THREADS=16`).
Source: `results/contour_zen4_serial_q256.csv`,
`results/contour_zen4_parallel_q256.csv`.

| k | Serial n | Parallel n | Speedup |
|---|----------|------------|---------|
| 2 | 415,040 | 1,562,501 | 3.8x |
| 100 | 121,169 | 968,753 | 8.0x |
| 1000 | 46,937 | 131,593 | 2.8x |
| 10000 | 32,500 | 103,750 | 3.2x |
| 13000 | 30,062 | 95,468 | 3.2x |

These boundary speedups are much lower than the 10-14x the performance grid
shows at moderate n, and lower than the previous box's 4.4-5.4x at the same k.
That is the expected shape, not an anomaly: at the one-second boundary the
parallel run is at a far larger n than the serial run it is compared against,
deep into the memory-bandwidth-bound regime documented below, so it is not a
same-n speedup. Read the grid for same-n scaling and this table only for
"how much further does 16 threads push the frontier".

### Known scaling limit: parallel speedup degrades at n ≥ 16,384 (same mechanism, different absolute numbers)

Parallel speedup on the 16-physical-core 7950X is a healthy 10-14x below
n=16,384 (e.g. n=8,192, k=n: 12.1x) but falls to ~5.7x at n=16,384 and stays
in the 5-7x range through n=65,536 (k=n). This is a genuine memory-bandwidth/
cache-capacity wall (confirmed via `perf stat`): the 3600 MT/s bandwidth
ceiling limits how much aggregate bandwidth 16 threads can extract once the
working set outgrows cache, so parallel efficiency drops at exactly the n
where the hybrid/FFT engine becomes memory-bandwidth-bound rather than
compute-bound.

### 1-second threshold: n = 26,816 (k=n, single-threaded), n = 65,536 (k=n, 16-thread)

**Both numbers are real `bench_grid threshold` binary searches on this box**
(`results/threshold_zen4_serial.txt`, `results/threshold_zen4_parallel.txt`).
Neither is interpolated or extrapolated. Like the M3 Pro figures above, these
were measured single-sample per candidate (the `threshold` subcommand's
original methodology, since fixed to a real median of 5 for future runs;
see the M3 Pro section for why: single-sample measurements are vulnerable to
sustained-load thermal drift, confirmed directly 2026-08-03). Kept as
originally measured for consistency with the rest of that day's data.

Two changes worth stating explicitly rather than silently swapping:

- **Parallel: the previously published `n ≈ 72,200` was never measured.**
  RESULTS.md disclosed it as extrapolated from contour data, and the real
  binary search now puts the boundary at **n = 65,536** (n=65,792 -> 1143 ms,
  n=66,560 -> 1093 ms, both over budget). The extrapolation was therefore
  about **10% optimistic**, in the direction HANDOFF.md predicted it might be.
  This is the first real parallel threshold measurement Zen4 has ever had.
- **Serial: 26,816 here vs the 17,984 previously published.** Both are real
  binary searches; they are simply different physical machines. See the
  reference-hardware note above, which quantifies the same 35-40% machine gap
  across the whole hybrid region.

### Dispatch: cost-based `select_engine()`, B from `select_best_B()` (typically B=32). Linear→hybrid crossover at k≈249-281 (empirical `crossover_k[]` in `devices/zen4/fft_config.h`).

Validated on the current box in **both** thread modes
(`results/crossover_zen4_20260803_serial.txt`, `..._parallel.txt`):

- **Serial** agrees with the table: linear wins through k=200 and hybrid takes
  over by k=240-260 across n=512-8192, bracketing the calibrated 249-281.
- **Parallel (16 threads) does not.** The real transition moves *down*, to
  k≈200 at n=4096-8192 and k≈40-80 at n=512-1024, so the serially-calibrated
  threshold keeps choosing linear over a band of k where hybrid is already
  faster at 16 threads. This is the known, deliberate scoping decision in
  VERDICTS.md V15 (one table, calibrated serial, reused for parallel) meeting
  its first counterexample: the same check on M3 Pro shows *no* shift at all.
  Disclosed in the paper; a thread-count-aware table is not implemented.

### AOCL-FFTW: sole backend, no dual dispatch

AOCL-FFTW (AMD's official znver4-tuned build) is the only FFT backend for Zen 4.
A direct A/B test at n=32768,k=n confirmed AOCL is 20-25% faster than plain system
FFTW at the raw kernel level, reproducible across repeated runs. Per-level FFT-size
selection uses `best_fft_config()` driven by `calib_times_ns[]` (749 calibrated sizes,
AOCL PATIENT wisdom). No `calib_lib[]` array exists, the earlier claim of
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
(`a_batch[j*BQ+qi]`, cache-friendly, eliminates L1 misses at all n).
Template in `src/cpu/linear_batched_impl.inc`.
- L2-aware checkpointing (`ckpt_interval_batched`)
- Cost-based engine dispatch (`select_engine`): no fixed K_CROSS thresholds
- Cross-correlation wrap correction handles both output-wrap and input-wrap
cyclic aliasing (corrects a pre-existing bug with wrap_m > 0)

### M3 Pro / Apple Silicon specific

- vDSP interleaved DFT dispatch (`vDSP_DFT_Interleaved_CreateSetupD`): 10-18%
faster FFT at 33 supported sizes (f × 2^g where f ∈ {1,3,5,15}, g ≥ 4).
Zero format conversion (uses same interleaved complex as FFTW). Forward ×2
scaling absorbed into pointwise multiply; single ×0.25 on inverse output.
- Calibration table updated with vDSP dispatch times, steering `best_fft_config()`
to prefer vDSP-supported sizes (e.g. 192 replaces 200 at saturated tree levels).

### Zen 4 specific

- AOCL-FFTW (znver4-tuned, tag 5.3), sole FFT backend, 20-25% faster than plain FFTW
- BQ=8 batched linear with interleaved layout (native AVX-512 width)
- L2-aware checkpointing with 1MB per-core L2
- B=24/32, cost model adapts to Zen 4's wider schoolbook-FFT crossover.
  Empirical B-selection table in `devices/zen4/fft_config.h`, currently 1944
  points. See `VERDICTS.md` V11 for the history (an earlier report validated
  the analytical `select_best_B` that was replaced by this empirical table
  one commit later; that report is no longer kept in the repo).

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

They live in `devices/<device>/fft_config.h`.

> **Superseded framing, corrected 2026-07-30.** The tables in this section, and
> the "Calibration methodology" section below, describe a nonlinear-least-squares
> fit with an RMS-error caveat. **That regression no longer exists.**
> `tools/fit_cost_model.py` skips scipy entirely when all six scalar pins are
> supplied ("a 0-parameter degenerate fit is meaningless") and simply writes the
> measured values through. Every constant is now a direct microbenchmark, a
> measured per-size or per-B table, or a hardware fact.
>
> Also note `select_engine()` and `select_best_B()` are **not** driven by these
> constants at all any more; both are empirical lookup tables. The constants'
> surviving role is pricing wrap correction, plus two scalars multiplying
> measured FFT times.
>
> Several constants below (`FMA_NS`, `FP64_DIV_NS`, `LEAF_*`, `BLOCK_*`,
> `FFT_OVERHEAD_NS`, the `*_BW_GBS` trio) were **read by nothing in `libicm.a`**
> and have since been deleted from `devices/*/fft_config.h` along with the dead
> `src/cpu/cost_model.h` that consumed them. `CLAUDE.md`'s constants table has
> the audited live/dead split. Values below are retained as a historical
> record of the fitting process, not as a description of any currently-shipped
> `#define`.

### M3 Pro (Apple Silicon, ARM64)

| Constant | Value | Notes |
|---|---|---|
| `FMA_NS` | 0.0500 | Scalar FMA cost. Fit lower bound, hit its limit when `WRAP_FMA_NS` and `FP64_DIV_NS` were pinned; see caveat below. |
| `WRAP_FMA_NS` | 0.5160 | Per-FMA cost for wrap correction. **Directly measured** via `tools/bench_wrap_fma.c`. |
| `FP64_DIV_NS` | 3.4890 | FP64 divide latency. **Directly measured** via `tools/bench_div_chain.c` (dependency-chained, not throughput). |
| `BLOCK_FMA_NS` | 0.4027 | FMA cost inside block build/divide. Superseded by the per-B `block_build_ns_per_player[]` table; dead in `libicm.a`. |
| `BLOCK_MEM_NS` | 0.1000 | Memory cost per element in block build/divide. |
| `PAIRED_CACHED_CORR_RATIO` | 1.9080 | Paired cached correlate cost / full FFT pipeline cost. |
| `INDEP_PAIR_RATIO` | 1.9080 | Independent pair correlate cost / full FFT pipeline cost. Equal to PAIRED, likely a fitting artifact (solver couldn't separate them). |
| `LEAF_FMA_NS` | 0.0727 | FMA cost at tree-leaf schoolbook multiplies. Superseded by the per-B `leaf_fma_ns_per_player[]` table; dead in `libicm.a`. |
| `LEAF_BLOCK_NS` | 48.1032 | Per-block overhead at leaf level. |
| `FFT_OVERHEAD_NS` | 631.0974 | Per-call FFT overhead. Physically odd value, pushed here to compensate when both pins are active; see caveat. |

> FFT calibration table (`calib_sizes[]`/`calib_times_ns[]`) and FFTW wisdom
> in `devices/m3_pro/fft_config.h` are from a genuine FFTW PATIENT calibration
> on this Apple M3 Pro machine (July 2026). `WRAP_FMA_NS` and `FP64_DIV_NS` are
> direct microbenchmark measurements, not recovered from aggregate regression;
> both were unidentifiable from the indirect fit alone (the regression converged
> to physically implausible values: 0.1ns and 0.5ns respectively, both hitting
> their fit lower bounds). Pinning both raises the fit's RMS log-relative error
> to 10.2% and pushes `FFT_OVERHEAD_NS`/`FMA_NS` to compensate, a collinearity
> limitation in the current `sample_plans` training data, not a correctness
> issue. `./bench_grid verify` passes ALL TESTS and `./bench_grid crossover`
> shows a clean, monotonic linear→hybrid transition at k≈122-124 (empirical
> crossover table in `devices/m3_pro/fft_config.h`).

### Zen 4 (AMD Ryzen 9 7950X, AVX-512, AOCL-FFTW)

| Constant | Value | Notes |
|---|---|---|
| `FMA_NS` | 0.0793 | Scalar FMA cost. 8-param fit (only `WRAP_FMA_NS` pinned). |
| `WRAP_FMA_NS` | 0.4360 | Per-FMA cost for wrap correction. **Directly measured** via `tools/bench_wrap_fma.c`, extracted as least-squares slope over the decision-relevant range `wrap_m ∈ [64,384]`. |
| `FP64_DIV_NS` | 12.5287 | FP64 divide latency. From the unpinned 8-param fit, never independently cross-checked against a direct measurement (unlike M3 Pro's). |
| `BLOCK_FMA_NS` | 0.6833 | FMA cost inside block build/divide (sequential dependency chain, latency- not throughput-bound). |
| `BLOCK_MEM_NS` | 0.1 | Memory cost per element in block build/divide. |
| `PAIRED_CACHED_CORR_RATIO` | 1.8287 | Paired cached correlate cost / full FFT pipeline cost. |
| `INDEP_PAIR_RATIO` | 1.8287 | Independent pair correlate cost / full FFT pipeline cost. |
| `LEAF_FMA_NS` | 0.1610 | FMA cost at tree-leaf schoolbook multiplies. Superseded by the per-B `leaf_fma_ns_per_player[]` table; dead in `libicm.a`. |
| `LEAF_BLOCK_NS` | 61.3029 | Per-block overhead at leaf level. |
| `FFT_OVERHEAD_NS` | 0.0 | Per-call FFT overhead (baked into `calib_times_ns[]`, not double-counted). |

> Calibration table (`calib_sizes[]`/`calib_times_ns[]`, 776 entries,
> ceiling 150,384 after the 2026-08-03 extension) and
> AOCL-FFTW PATIENT wisdom in `devices/zen4/fft_config.h` are from an AMD
> Ryzen 9 7950X (same SKU as the benchmark machine). `WRAP_FMA_NS` was
> directly measured after the indirect fit proved it unidentifiable from
> aggregate `sample_plans` data (the old fit value 0.8612 was arbitrary;
> wrap-correction cost never exceeds 1.5% of any sampled plan's total time,
> a "persistency of excitation" failure). Fixing this constant (and unifying
> the code-level `FMA_NS`/`WRAP_FMA_NS` mismatch in the planner) produced a
> 2.35× speedup on the previously-regressed n=32768,k=n cell with no
> regressions across spot-checks. `./bench_grid verify`: ALL TESTS PASSED.

### Zen 4 bandwidth constants, root cause diagnosed and fixed, re-verified

`devices/zen4/fft_config.h` once contained `L2_BW_GBS=341868.5` and
`L3_BW_GBS=3233.3`, both physically impossible (hundreds of TB/s for L2).
Pre-existing bug (confirmed present in the commit before this sprint started).
Root cause: `tools/calibrate.c`'s `measure_bw()` runs its streaming loop `reps`
times, but the loop body (`a[i] = b[i]*s + c[i]`) doesn't depend on the
repetition index, an optimizing compiler can prove the repeated stores are
redundant and collapse the whole `reps` loop to a single real pass, while the
byte-count computation still charges for every nominal repetition, inflating
the reported bandwidth by ~`reps`x.

**Fixed** with a standard compiler memory barrier (`asm volatile` with a
memory clobber) after each repetition, forcing the compiler to treat memory as
externally observed. Verified in isolation not to regress M3 Pro's
already-correct values (it tightens them: 83-114 GB/s scattered → a consistent
~115 GB/s across all three cache levels).

**Re-verified with a fresh calibration run on Zen4 hardware** (commit `18bf1c3`,
2026-07-22). `devices/zen4/fft_config.h` at the time contained sane values
(`L2_BW_GBS=131.5`, `L3_BW_GBS=56.0`, `DRAM_BW_GBS=33.0`, all in GB/s), directly
measured on an AMD Ryzen 9 7950X with the compiler-barrier fix applied.
`src/cpu/cost_model.h` (the only consumer of these constants, via
`blended_bw()`/`linear_roofline_cost()`, neither ever called live) and the
`*_BW_GBS` macros themselves have since been deleted as dead code; this
historical bug affected no shipped dispatch decision and `./bench_grid verify`
was unaffected throughout.

---

## Calibration methodology

This sprint established a direct-microbenchmark calibration pipeline that
replaces the previous indirect-aggregate-regression approach for two constants
that proved unidentifiable from aggregate timing data alone.

### The problem: persistency of excitation

The cost model has 9 free parameters fitted against per-plan measured times
from `tools/sample_plans.c`. Two of them, `WRAP_FMA_NS` (wrap-correction FMA
cost) and `FP64_DIV_NS` (dependency-chained FP64 division latency), each
contribute at most ~1.5% of any single sampled plan's total predicted time.
In control-theory / system-identification terms, the training signal doesn't
vary these parameters' effects enough to be recoverable, a "persistency of
excitation" failure. The regression converges to arbitrary values within a wide
flat basin, not to physically meaningful ones.

This is the same class of problem FFTW solves by timing plans directly (PATIENT
mode) rather than fitting a global model, ATLAS/AEOS solves by per-kernel
empirical timing, and the roofline model solves with dedicated bandwidth/FLOP
microbenchmarks, all cite direct isolated measurement over indirect aggregate
regression for exactly this reason.

### The fix: direct isolated microbenchmarks

- **`WRAP_FMA_NS`**: measured via `tools/bench_wrap_fma.c`, a verbatim copy of
  the wrap-correction loop body run in isolation, sweeping `wrap_m` over a wide
  range so the correction dominates measured time by construction. The value is
  extracted as a least-squares **slope** of time vs. FMA count over the
  decision-relevant range (cancels fixed per-call overhead). The measured curve
  shows a real, physically-explicable cache-hierarchy transition: marginal cost
  rises smoothly from near-FMA-throughput at small working sets to
  memory-latency-bound at large ones. R²=0.9998 on Zen4.
- **`FP64_DIV_NS`**: measured via `tools/bench_div_chain.c`, a
  dependency-chained microbenchmark that reproduces the actual usage pattern
  (leaf extraction's synthetic-division recurrence). Critically, this is NOT an
  independent/vectorizable division loop, that would measure throughput, a
  very different and wrong number for this sequential-dependency-chain usage.

Both tools are wired into `tools/calibrate_full.sh` as standard pipeline steps
for all future device ports.

### Known limitation (resolved; retained as history)

Pinning constants used to raise the fit's RMS log-relative error where the
`sample_plans` training data didn't cleanly separate their effects. Observed on
M3 Pro: 10.2% RMS with both pinned vs. 6.57% unpinned, with `FFT_OVERHEAD_NS`
pushed to a physically odd 631 ns to compensate.

**This limitation no longer applies.** The regression it describes was removed;
all six scalar constants are now pinned from direct microbenchmarks and the
optimizer is skipped, so there is no fit whose RMS error could degrade.
`FFT_OVERHEAD_NS` (which this fit pushed to the physically odd value above)
has since been deleted entirely as confirmed dead code, redundant by
construction with `calib_times_ns[]` already measuring the full pipeline. The
identifiability failure recorded here is the *motivation* for that migration,
not a standing caveat.

---

## NVIDIA B200 GPU (sm_100, cuFFTDx fused kernels, CUDA graph capture)

> The linear engine is CPU-only (sequential player-by-player structure can't
> saturate GPU parallelism). Only the tree-based engines map to the GPU; the
> planner assigns each subproduct-tree level to one of three kernel tiers
> (schoolbook, cuFFTDx fused, batched cuFFT) based on polynomial degree.

### Performance (ms, Q=256, FP64): systematic (n, k) grid

| n | k=64 | k=1024 | k=n/2 | k=n |
|---|------|--------|-------|-----|
| 4,096 | 0.37 | 0.76 | 0.87 | 0.89 |
| 16,384 | 1.18 | 2.81 | 3.93 | 4.17 |
| 65,536 | 4.29 | 10.62 | 19.43 | 20.26 |
| 262,144 | 16.58 | 41.01 | 95.95 | 100.09 |
| 1,048,576 | 65.68 | 178.26 | 501.84 | 507.90 |
| 4,194,304 | 272.47 | 753.65 | 2,352.43 | 2,320.45 |
| 16,777,216 | 1,215.01 | 2,493.21 | 10,582.00 | 10,719.01 |
| 33,554,432 | 2,506.71 | 5,059.26 | 22,321.49 | 22,865.76 |

> **Provenance note.** A cost-model bug (`VERDICTS.md` V20a/V20b, found
> 2026-08-04, same class as V20 but a third, GPU-specific mechanism) let
> the planner pick an FFT size smaller than the operand at 5 of these 32
> cells: k=1,024 for n=65,536/262,144/4,194,304, and k=n/2 and k=n for
> n=1,048,576. Those 5 cells were re-measured 2026-08-04 on a fresh B200
> with the fix compiled in, after `test_gpu_wrap_feasibility` passed on
> hardware and an extended `bench_gpu_fused verify` grid (now including
> the k=1,024-2,048 band the bug lived in) passed 42/42 against the CPU
> reference; the table above shows the re-measured values. The k=1,024
> column got slower on its now-feasible plans (e.g. 671.06 → 753.65ms at
> n=4,194,304); the two n=1,048,576 large-k cells moved <0.5%. The other
> 27 cells were confirmed unaffected by replaying the pre-fix planner over
> every published cell (`VERDICTS.md` V20b) and keep their 2026-07-30
> timings.

Full 211-point calibration heatmap
(`results/gpu_heatmap_b200.csv`), regenerated 2026-07-30 on
top of two rounds of B-selection anchor fixes (`2620583`, `71db180`; see
`VERDICTS.md` V7). All 210 cells **ran to completion** with cv ≤ 0.036
everywhere (most 0.000). **"zero errors" here means no OOM/execution
failure, not a numerical accuracy check**: `tools/heatmap_gpu.cu` makes no
comparison against a reference value at all, so this line was previously
worded to imply a correctness guarantee it never provided. (Numerical
accuracy is gated separately by `bench_gpu_fused verify`, whose grid now
includes the k=1,024-2,048 band.) 21 of these 210 cells were affected by
the V20a/V20b timing bug and were re-measured 2026-08-04 on a fresh B200
with the fix in place (`./heatmap_gpu --cells`, identical measurement path
as the full run); the CSV contains the re-measured rows, with the
pre-splice file kept as
`results/gpu_heatmap_b200_20260804_pre_v20b_resplice.csv`. The remaining
189 cells were confirmed unaffected by planner replay (`VERDICTS.md` V20b)
and keep their 2026-07-30 values; re-measured neighbouring cells that
merely changed run (not plan) moved ~1%, so the two runs are directly
comparable.

> **Known limitation, disclosed rather than hidden: 1 cell out of 210 is a
> genuine, understood regression.** n=65,536, k=2,048 dispatches B=48 and
> ran 14.13ms versus 12.13ms before the 2026-07-30 anchor fixes (+16.5%).
> (Both of those timings predate the V20a feasibility fix; the cell's
> current post-fix, re-measured value is 15.28ms, but the B-selection
> regression mechanism described here is independent of that fix and
> stands.)
> Root cause: a low-k anchor added at n=131,072 to fix a *different* cell
> became this cell's nearest neighbour in the joint (n,k) lookup, pulling
> it to the wrong B. Two rounds of targeted anchor fixes each fixed their
> target cells and introduced exactly one new problem elsewhere (16 broken
> cells in round 1, reduced to this single cell in round 2); chasing the
> last one is not currently judged worth another full B200 rental cycle
> against an aggregate signal (below) that is already unambiguous. See
> `VERDICTS.md` V7 for the complete history and the exact next step if
> this gets revisited.

**This session's B-selection anchor fixes are a large, broad win overall,
not just a correctness fix, this one cell aside.** Diffing the new heatmap
cell-by-cell against the original pre-session baseline
(`results/gpu_heatmap_b200_20260728.csv`) at the same 210 `(n,k)` points:
61 cells changed B, 54 improved (some substantially: the 12 cells above
n=1,572,864 that motivated this fix were up to 78.7% slower before it),
1 regressed (above), 6 within noise. Total grid time drops from 451.7s to
441.1s (-2.34% aggregate). This is a separate, later fix from the
`b53dd17`/`b06379e` pair described just above (both from an earlier
session, 2026-07-27/28); that comparison and its 1.61x aggregate figure
are historical record and still accurate for what they measured at the
time.

### 1-second threshold: real binary search (median of 5 reps/candidate)

| Query shape | Largest n ≤ 1000ms | Smallest n > 1000ms | Bracket width |
|---|---|---|---|
| k = n | 1,490,944 (943.9ms) | 1,506,304 (1,150.5ms) | 15,360 (1.0% of lo) |
| k = 100 | 7,975,936 (998.9ms) | 7,991,296 (1,001.1ms) | 15,360 (0.2% of lo) |

The `k = n` search was fully re-run 2026-08-04 with the V20a/V20b
feasibility fix in place (trace:
`results/gpu_threshold_search_kn_20260804.txt`), because both endpoints of
the original bracket had been timed on infeasible plans. **The bracket
came back identical**: the largest full-field n under one second is still
1,490,944, now measured on a feasible plan (943.9ms vs the pre-fix
918.8ms). The `k = 100` bracket was never affected (verified by planner
replay, `VERDICTS.md` V20b) and keeps its 2026-07-28 numbers (trace:
`results/gpu_threshold_search_20260728.txt`;
`tools/threshold_search_gpu.cu`, plan-based API). These searches supersede
the old 2026-07-25 frontier-probe-derived estimates (n≈1,441,792 /
n≈6,291,456), which predated the ragged-tree fix and were
reinterpretations of 5 fixed sample points, not an actual search.

### Dispatch: three-tier kernel planner (schoolbook / cuFFTDx fused / batched cuFFT), cost-based per tree level

The shipped GPU planner is driven by measured calibration data
(`devices/b200/gpu_fft_config.h`: per-pipeline FFT timing tables, directly
measured scalar constants such as `GPU_SCHOOL_FMA_NS`, and the empirical
`gbselect_*[]` B table), not by fitted parameters.
`tools/fit_gpu_cost_model.py` does fit four constants (`C_wrap`,
`C_school`, `R`, `C_gap`) for offline analysis, but **writes nothing into
the build**: no corresponding macros exist in the config header or
`src/gpu/` (verified by grep, 2026-07-30). See "GPU Cost Model (B200)" in
`OPTIMIZATION_GUIDE.md` for the decision-path details.

> **Diagnostic pass (July 2026):** The GPU planner was confirmed NOT to have the
> CPU's wrap-correction cost-model bug. `src/gpu/gpu_plan.cu` uses one constant
> (`GPU_SCHOOL_FMA_NS`) uniformly in both joint and independent paths, no
> code-level asymmetry. Additionally, the GPU's fitted `C_wrap` is
> diagnostic-only (`fit_gpu_cost_model.py` never writes it to any config
> header), so even if under-identified it has zero effect on real planning.
> No GPU numbers changed in that pass.

