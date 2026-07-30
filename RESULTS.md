# RESULTS.md - ICM Equity Optimization Results

All correctness tests PASS at < 5e-12 relative error (< 1e-9 for Smax=10^9).
Q=256 quadrature points.

## Apple M3 Pro (ARM64, NEON+vDSP, BQ=8)

> FFTW PATIENT calibration on Apple M3 Pro (6P+6E, 12 logical cores). `WRAP_FMA_NS`
> and `FP64_DIV_NS` are directly measured via isolated microbenchmarks
> (`tools/bench_wrap_fma.c`, `tools/bench_div_chain.c`) rather than recovered
> from aggregate regression, see [Calibration methodology](#calibration-methodology) below.
> Engine dispatch: `select_engine()` cost-based, B auto-selected (typically B=32).

### Performance (ms, uniform stacks, median of 5) - M3 Pro

Single-threaded vs 12-thread parallel, per (n, k) cell:

| n | k | serial (ms) | parallel (ms) | speedup |
|---|---|---|---|---|
| 64 | k=10 | 0.09 | 0.049 | 1.8x |
| 64 | k=50 | 0.32 | 0.088 | 3.6x |
| 64 | k=100 | 0.43 | 0.106 | 4.1x |
| 64 | k=n/4 | 0.119 | 0.05 | 2.4x |
| 64 | k=n/2 | 0.201 | 0.07 | 2.9x |
| 64 | k=n | 0.421 | 0.116 | 3.6x |
| 128 | k=10 | 0.177 | 0.055 | 3.2x |
| 128 | k=50 | 0.68 | 0.156 | 4.4x |
| 128 | k=100 | 1.3 | 0.253 | 5.1x |
| 128 | k=n/4 | 0.475 | 0.123 | 3.9x |
| 128 | k=n/2 | 0.869 | 0.18 | 4.8x |
| 128 | k=n | 1.31 | 0.229 | 5.7x |
| 256 | k=10 | 0.412 | 0.109 | 3.8x |
| 256 | k=50 | 1.36 | 0.281 | 4.8x |
| 256 | k=100 | 3.21 | 0.515 | 6.2x |
| 256 | k=n/4 | 1.8 | 0.344 | 5.2x |
| 256 | k=n/2 | 3.36 | 0.471 | 7.1x |
| 256 | k=n | 3.65 | 0.522 | 7.0x |
| 512 | k=10 | 0.839 | 0.183 | 4.6x |
| 512 | k=50 | 3.58 | 0.64 | 5.6x |
| 512 | k=100 | 6.56 | 1.12 | 5.9x |
| 512 | k=n/4 | 7.58 | 1.03 | 7.4x |
| 512 | k=n/2 | 9.16 | 1.28 | 7.2x |
| 512 | k=n | 10.1 | 1.39 | 7.3x |
| 1024 | k=10 | 1.79 | 0.329 | 5.4x |
| 1024 | k=50 | 7.28 | 1.23 | 5.9x |
| 1024 | k=100 | 12.8 | 2.19 | 5.8x |
| 1024 | k=n/4 | 18.4 | 2.64 | 7.0x |
| 1024 | k=n/2 | 22.2 | 2.99 | 7.4x |
| 1024 | k=n | 24.0 | 3.27 | 7.3x |
| 2048 | k=10 | 4.01 | 0.703 | 5.7x |
| 2048 | k=50 | 14.1 | 2.4 | 5.9x |
| 2048 | k=100 | 26.2 | 4.41 | 5.9x |
| 2048 | k=n/4 | 47.9 | 6.27 | 7.6x |
| 2048 | k=n/2 | 54.8 | 7.18 | 7.6x |
| 2048 | k=n | 57.3 | 7.77 | 7.4x |
| 4096 | k=10 | 8.1 | 1.36 | 6.0x |
| 4096 | k=50 | 28.2 | 4.79 | 5.9x |
| 4096 | k=100 | 52.4 | 8.11 | 6.5x |
| 4096 | k=n/4 | 115 | 16.1 | 7.1x |
| 4096 | k=n/2 | 135 | 18.5 | 7.3x |
| 4096 | k=n | 155 | 21.5 | 7.2x |
| 8192 | k=10 | 16.1 | 2.67 | 6.0x |
| 8192 | k=50 | 56.4 | 9.44 | 6.0x |
| 8192 | k=100 | 104 | 16.4 | 6.3x |
| 8192 | k=n/4 | 273 | 37.9 | 7.2x |
| 8192 | k=n/2 | 325 | 46.9 | 6.9x |
| 8192 | k=n | 359 | 50.4 | 7.1x |
| 16384 | k=10 | 32.3 | 5.4 | 6.0x |
| 16384 | k=50 | 113 | 18.7 | 6.0x |
| 16384 | k=100 | 210 | 35.1 | 6.0x |
| 16384 | k=n/4 | 650 | 91.9 | 7.1x |
| 16384 | k=n/2 | 725 | 101 | 7.2x |
| 16384 | k=n | 778 | 111 | 7.0x |
| 32768 | k=10 | 65.0 | 10.7 | 6.1x |
| 32768 | k=50 | 227 | 37.9 | 6.0x |
| 32768 | k=100 | 420 | 67.7 | 6.2x |
| 32768 | k=n/4 | 1590 | 229 | 6.9x |
| 32768 | k=n/2 | 1800 | 265 | 6.8x |
| 32768 | k=n | 1970 | 296 | 6.7x |
| 65536 | k=10 | 131 | 21.6 | 6.1x |
| 65536 | k=50 | 456 | 77.4 | 5.9x |
| 65536 | k=100 | 842 | 142 | 5.9x |
| 65536 | k=n/4 | 3460 | 515 | 6.7x |
| 65536 | k=n/2 | 3880 | 583 | 6.7x |
| 65536 | k=n | 4120 | 649 | 6.3x |

### Parallel speedup - M3 Pro

At the 1-second boundary (from regenerated contour sweep, Q=256):

| k | Serial n | Parallel n | Speedup |
|---|----------|------------|---------|
| 2 | 1,025,391 | 5,468,751 | 5.3x |
| 100 | 78,209 | 453,134 | 5.8x |
| 1000 | 33,156 | 244,140 | 7.4x |
| 10000 | 20,312 | 131,875 | 6.5x |
| 13000 | 19,500 | 122,687 | 6.3x |

Speedup varies by k due to engine dispatch: linear-only k values see ~5-6x (simple SIMD scaling),
while hybrid-engine k values reach ~7-8x (FFT tree parallelism). M3 Pro's 6P+6E topology
limits peak parallel speedup to ~8x vs Zen 4's ~14x on 16 homogeneous P-cores.

### 1-second threshold: n ≈ 19,400 (k=n, single-threaded), n ≈ 98,000 (k=n, 12-thread)

Serial: interpolated from bench_grid (n=16,384 at 778 ms, n=32,768 at 1,970 ms). Parallel: extrapolated from bench_grid (n=32,768 at 296 ms, n=65,536 at 649 ms); regenerated contour and grid (July 2026) using the post-recalibration B-selection tables.

### Dispatch: cost-based `select_engine()`, B from `select_best_B()` (typically B=32). Linear→hybrid crossover at k≈122-124 (empirical crossover table in `devices/m3_pro/fft_config.h`).

---

## AMD Ryzen 9 7950X (Zen 4, AVX-512, AOCL-FFTW), 2026-07-27 reference data

> **Reference hardware**: dedicated Zen4 box, fresh redeployment 2026-07-27.
> AOCL-FFTW (AMD's official znver4-tuned build, tag 5.3) is the sole FFT backend,
> built from source with the documented AM5 flag set. A direct A/B test confirmed
> AOCL is cleanly faster than plain system FFTW at every calibrated size, no dual
> dispatch. All numbers below are under the `performance` cpufreq governor (16 physical
> cores, SMT off for benchmarking, `OMP_NUM_THREADS=16` for parallel).
> `WRAP_FMA_NS=0.40` is directly measured via `tools/bench_wrap_fma.c`, see
> [Calibration methodology](#calibration-methodology).
>
> **Important: this box's RAM runs at 3600 MT/s vs. its 5600 MT/s DIMM rating**,
> an AMD AM5 2-DIMMs-per-channel (2DPC) platform electrical limit, confirmed not
> fixable at the OS/BIOS level. A 1DPC replacement box was not available. Per
> explicit user decision, **this box (at 3600 MT/s) is now the standing Zen4
> reference** for this project; this is a permanent characteristic of the reference
> hardware, not a temporary anomaly or a bug to fix.
>
> The RAM ceiling's effect is **not a flat percentage**: linear/schoolbook-engine
> timings are nearly identical to our prior (higher-bandwidth) Zen4 box (~0.97-1.0×),
> while hybrid/FFT-heavy timings are 40-65% slower at larger k, because FFT is
> memory-bandwidth-bound and schoolbook isn't. See the [Prior hardware comparison
> (2026-07-22/24)](#prior-hardware-comparison-2026-07-2224) subsection below for the
> full old-vs-new breakdown.

### Performance (ms, uniform stacks, median of 5) - Zen 4 (current reference, 2026-07-27)

Single-threaded vs 16-thread parallel, per (n, k) cell. Source:
`results/bench_grid_zen4_serial_20260727.txt`, `results/bench_grid_zen4_parallel_20260727.txt`.

| n | k | serial (ms) | parallel (ms) | speedup |
|---|---|---|---|---|
| 64 | k=10 | 0.103 | 0.0165 | 6.2x |
| 64 | k=50 | 0.189 | 0.0263 | 7.2x |
| 64 | k=100 | 0.223 | 0.0263 | 8.5x |
| 64 | k=n/4 | 0.108 | 0.0166 | 6.5x |
| 64 | k=n/2 | 0.145 | 0.0194 | 7.5x |
| 64 | k=n | 0.225 | 0.0267 | 8.4x |
| 128 | k=10 | 0.197 | 0.0223 | 8.8x |
| 128 | k=50 | 0.329 | 0.0384 | 8.6x |
| 128 | k=100 | 0.621 | 0.0615 | 10.1x |
| 128 | k=n/4 | 0.283 | 0.0332 | 8.5x |
| 128 | k=n/2 | 0.397 | 0.0457 | 8.7x |
| 128 | k=n | 0.792 | 0.0737 | 10.7x |
| 256 | k=10 | 0.393 | 0.0402 | 9.8x |
| 256 | k=50 | 0.673 | 0.0801 | 8.4x |
| 256 | k=100 | 1.56 | 0.130 | 12.0x |
| 256 | k=n/4 | 0.818 | 0.0883 | 9.3x |
| 256 | k=n/2 | 2.03 | 0.159 | 12.8x |
| 256 | k=n | 4.11 | 0.230 | 17.9x |
| 512 | k=10 | 0.724 | 0.0612 | 11.8x |
| 512 | k=50 | 2.08 | 0.548 | 3.8x |
| 512 | k=100 | 3.74 | 0.688 | 5.4x |
| 512 | k=n/4 | 4.67 | 0.596 | 7.8x |
| 512 | k=n/2 | 9.11 | 0.670 | 13.6x |
| 512 | k=n | 9.95 | 0.756 | 13.2x |
| 1024 | k=10 | 1.46 | 0.121 | 12.1x |
| 1024 | k=50 | 3.48 | 0.359 | 9.7x |
| 1024 | k=100 | 7.32 | 1.21 | 6.0x |
| 1024 | k=n/4 | 19.1 | 1.41 | 13.5x |
| 1024 | k=n/2 | 22.3 | 1.66 | 13.4x |
| 1024 | k=n | 24.1 | 1.78 | 13.5x |
| 2048 | k=10 | 3.67 | 0.294 | 12.5x |
| 2048 | k=50 | 8.21 | 0.681 | 12.1x |
| 2048 | k=100 | 15.0 | 1.16 | 12.9x |
| 2048 | k=n/4 | 48.4 | 3.50 | 13.8x |
| 2048 | k=n/2 | 53.7 | 3.94 | 13.6x |
| 2048 | k=n | 57.2 | 4.23 | 13.5x |
| 4096 | k=10 | 7.39 | 0.651 | 11.4x |
| 4096 | k=50 | 17.8 | 1.48 | 12.0x |
| 4096 | k=100 | 25.8 | 2.35 | 11.0x |
| 4096 | k=n/4 | 115 | 8.40 | 13.7x |
| 4096 | k=n/2 | 128 | 9.36 | 13.7x |
| 4096 | k=n | 139 | 10.1 | 13.8x |
| 8192 | k=10 | 13.4 | 1.23 | 10.9x |
| 8192 | k=50 | 34.3 | 2.84 | 12.1x |
| 8192 | k=100 | 53.6 | 4.63 | 11.6x |
| 8192 | k=n/4 | 277 | 20.4 | 13.6x |
| 8192 | k=n/2 | 315 | 24.2 | 13.0x |
| 8192 | k=n | 341 | 28.1 | 12.1x |
| 16384 | k=10 | 29.6 | 2.37 | 12.5x |
| 16384 | k=50 | 56.9 | 6.43 | 8.8x |
| 16384 | k=100 | 117 | 9.69 | 12.1x |
| 16384 | k=n/4 | 667 | 88.1 | 7.6x |
| 16384 | k=n/2 | 743 | 112 | 6.6x |
| 16384 | k=n | 810 | 142 | 5.7x |
| 32768 | k=10 | 62.3 | 6.04 | 10.3x |
| 32768 | k=50 | 131 | 13.6 | 9.6x |
| 32768 | k=100 | 218 | 23.6 | 9.2x |
| 32768 | k=n/4 | 1570 | 280 | 5.6x |
| 32768 | k=n/2 | 1790 | 314 | 5.7x |
| 32768 | k=n | 1950 | 363 | 5.4x |
| 65536 | k=10 | 118 | 21.0 | 5.6x |
| 65536 | k=50 | 266 | 30.4 | 8.8x |
| 65536 | k=100 | 411 | 48.4 | 8.5x |
| 65536 | k=n/4 | 4110 | 627 | 6.6x |
| 65536 | k=n/2 | 4750 | 729 | 6.5x |
| 65536 | k=n | 5110 | 901 | 5.7x |

### Parallel speedup - Zen 4 (current reference, 2026-07-27)

At the 1-second boundary (from contour sweep, Q=256). Source:
`results/contour_zen4_serial_q256_20260727.csv`, `results/contour_zen4_parallel_q256_20260727.csv`.

| k | Serial n | Parallel n | Speedup |
|---|----------|------------|---------|
| 2 | 402,833 | 1,513,672 | 3.8x |
| 100 | 117,264 | 906,259 | 7.7x |
| 1000 | 31,625 | 137,812 | 4.4x |
| 10000 | 17,500 | 94,375 | 5.4x |
| 13000 | 16,250 | 86,937 | 5.4x |

### Known scaling limit: parallel speedup degrades at n ≥ 16,384 (same mechanism, different absolute numbers)

Parallel speedup on the 16-physical-core 7950X is a healthy 10-14x below
n=16,384 (e.g. n=8,192, k=n: 12.1x) but falls to ~5.7x at n=16,384 and stays
in the 5-7x range through n=65,536 (k=n). The underlying mechanism is the same
genuine memory-bandwidth/cache-capacity wall documented in the prior-hardware
section below (confirmed via `perf stat` on the earlier box), but the absolute
speedup numbers differ because the same 3600 MT/s bandwidth ceiling that
hurts serial also limits how much aggregate bandwidth 16 threads can extract;
the parallel numbers are closer to the old box's parallel numbers than the
serial numbers are, consistent with 16 threads already saturating the bus on
both boxes. The full old-vs-new comparison is in the prior-hardware section
below.

### 1-second threshold: n = 17,984 (k=n, single-threaded), n ≈ 72,200 (k=n, 16-thread)

Serial: **n=17,984**, the first real `bench_grid threshold` binary search this
project has run on Zen4 hardware (prior RESULTS.md's "n≈29,000" was an interpolation
between two grid points, never a real binary search). Parallel: n≈72,200,
extrapolated from the contour data above (n=70,000 at 1718ms floor; this is the same
floor the old box hit, consistent with both boxes being bandwidth-saturated in parallel).

### Dispatch: cost-based `select_engine()`, B from `select_best_B()` (typically B=32). Linear→hybrid crossover at k≈249-281 (empirical crossover table in `devices/zen4/fft_config.h`, recalibrated for this box this session, commits `eb40e2d`/`2aff562`).

### AOCL-FFTW: sole backend, no dual dispatch

AOCL-FFTW (AMD's official znver4-tuned build) is the only FFT backend for Zen 4.
A direct A/B test at n=32768,k=n confirmed AOCL is 20-25% faster than plain system
FFTW at the raw kernel level, reproducible across repeated runs. Per-level FFT-size
selection uses `best_fft_config()` driven by `calib_times_ns[]` (749 calibrated sizes,
AOCL PATIENT wisdom). No `calib_lib[]` array exists, the earlier claim of
"AOCL-FFTW+MKL dual dispatch, 637 vs 112 sizes" in prior versions of this document
traced to measurements on a different box that never had AOCL-FFTW installed.

### Prior hardware comparison (2026-07-22/24): "when we had more memory bandwidth"

The numbers below are from our earlier Zen4 box, which ran its DIMMs at full rated
speed (1DPC configuration, ~5600 MT/s effective). They are preserved here as a
labeled historical reference; the user's own framing: "when you had more memory
bandwidth, you did better on the Zen4." The gap is **not a flat percentage**: at
small k (linear/schoolbook engine, compute-bound), the two boxes are nearly identical
(ratio ~0.97-1.0×). At large k (hybrid/FFT engine, memory-bandwidth-bound), the
current box is ~40-65% slower. A single "divide everything by ~1.6" extrapolation
would be wrong; it overcorrects the linear-dominated region. If a 1DPC-equivalent
number is needed for commentary, scale by regime: ~1.0× for k < 100 (linear-bound),
~1.6-1.8× for k ≥ 1000 (hybrid/FFT-bound). **These are rough estimates, not
measurements.**

#### Old performance grid (2026-07-22/24, higher-bandwidth Zen4)

Single-threaded vs 16-thread parallel, per (n, k) cell. All numbers from the prior
bench_grid runs (median of 5, Q=256, uniform stacks).

| n | k | serial (ms) | parallel (ms) | speedup |
|---|---|---|---|---|
| 64 | k=10 | 0.0891 | 0.014 | 6.4x |
| 64 | k=50 | 0.159 | 0.023 | 6.9x |
| 64 | k=100 | 0.189 | 0.026 | 7.3x |
| 64 | k=n/4 | 0.105 | 0.0138 | 7.6x |
| 64 | k=n/2 | 0.14 | 0.0162 | 8.6x |
| 64 | k=n | 0.19 | 0.0241 | 7.9x |
| 128 | k=10 | 0.181 | 0.0203 | 8.9x |
| 128 | k=50 | 0.317 | 0.0386 | 8.2x |
| 128 | k=100 | 0.613 | 0.0685 | 8.9x |
| 128 | k=n/4 | 0.244 | 0.0307 | 7.9x |
| 128 | k=n/2 | 0.382 | 0.0461 | 8.3x |
| 128 | k=n | 0.724 | 0.0814 | 8.9x |
| 256 | k=10 | 0.318 | 0.035 | 9.1x |
| 256 | k=50 | 0.658 | 0.0781 | 8.4x |
| 256 | k=100 | 1.74 | 0.151 | 11.5x |
| 256 | k=n/4 | 0.872 | 0.0896 | 9.7x |
| 256 | k=n/2 | 2.11 | 0.195 | 10.8x |
| 256 | k=n | 3.36 | 0.254 | 13.2x |
| 512 | k=10 | 0.633 | 0.0646 | 9.8x |
| 512 | k=50 | 1.7 | 0.175 | 9.7x |
| 512 | k=100 | 3.12 | 0.296 | 10.5x |
| 512 | k=n/4 | 4.29 | 0.404 | 10.6x |
| 512 | k=n/2 | 7.31 | 0.545 | 13.4x |
| 512 | k=n | 7.76 | 0.593 | 13.1x |
| 1024 | k=10 | 1.32 | 0.13 | 10.2x |
| 1024 | k=50 | 3.46 | 0.351 | 9.9x |
| 1024 | k=100 | 6.99 | 0.602 | 11.6x |
| 1024 | k=n/4 | 15.4 | 1.14 | 13.5x |
| 1024 | k=n/2 | 17.0 | 1.26 | 13.5x |
| 1024 | k=n | 17.7 | 1.33 | 13.3x |
| 2048 | k=10 | 3.2 | 0.313 | 10.2x |
| 2048 | k=50 | 7.09 | 0.719 | 9.9x |
| 2048 | k=100 | 14.3 | 1.19 | 12.0x |
| 2048 | k=n/4 | 35.6 | 2.62 | 13.6x |
| 2048 | k=n/2 | 38.5 | 2.85 | 13.5x |
| 2048 | k=n | 40.7 | 3.01 | 13.5x |
| 4096 | k=10 | 7.27 | 0.631 | 11.5x |
| 4096 | k=50 | 15.2 | 1.43 | 10.6x |
| 4096 | k=100 | 27.5 | 2.34 | 11.8x |
| 4096 | k=n/4 | 81.5 | 5.99 | 13.6x |
| 4096 | k=n/2 | 89.3 | 6.59 | 13.6x |
| 4096 | k=n | 93.7 | 6.96 | 13.5x |
| 8192 | k=10 | 14.6 | 1.21 | 12.1x |
| 8192 | k=50 | 27.7 | 2.79 | 9.9x |
| 8192 | k=100 | 50.7 | 4.76 | 10.7x |
| 8192 | k=n/4 | 185 | 17.8 | 10.4x |
| 8192 | k=n/2 | 205 | 16.9 | 12.1x |
| 8192 | k=n | 220 | 21.9 | 10.0x |
| 16384 | k=10 | 29.4 | 2.58 | 11.4x |
| 16384 | k=50 | 57.6 | 5.59 | 10.3x |
| 16384 | k=100 | 121 | 9.55 | 12.7x |
| 16384 | k=n/4 | 433 | 84.6 | 5.1x |
| 16384 | k=n/2 | 472 | 110 | 4.3x |
| 16384 | k=n | 491 | 149 | 3.3x |
| 32768 | k=10 | 58.3 | 5.96 | 9.8x |
| 32768 | k=50 | 123 | 11.8 | 10.4x |
| 32768 | k=100 | 235 | 20.0 | 11.8x |
| 32768 | k=n/4 | 1030 | 280 | 3.7x |
| 32768 | k=n/2 | 1070 | 311 | 3.4x |
| 32768 | k=n | 1140 | 359 | 3.2x |
| 65536 | k=10 | 125 | 20.4 | 6.1x |
| 65536 | k=50 | 255 | 32.5 | 7.8x |
| 65536 | k=100 | 443 | 45.8 | 9.7x |
| 65536 | k=n/4 | 2620 | 644 | 4.1x |
| 65536 | k=n/2 | 2940 | 747 | 3.9x |
| 65536 | k=n | 3300 | 928 | 3.6x |

#### Old parallel speedup (2026-07-22/24, higher-bandwidth Zen4)

| k | Serial n | Parallel n | Speedup |
|---|----------|------------|---------|
| 2 | 402,833 | 1,513,672 | 3.8x |
| 100 | 128,980 | 1,000,050 | 7.8x |
| 1000 | 48,468 | 131,593 | 2.7x |
| 10000 | 30,625 | 99,062 | 3.2x |
| 13000 | 28,843 | 95,468 | 3.3x |

#### Old 1-second threshold (2026-07-22/24): n ≈ 29,000 (k=n, serial), n ≈ 70,000 (k=n, parallel)

Serial: interpolated from bench_grid (n=16,384 at 491 ms, n=32,768 at 1,140 ms);
**never a real binary search**; the current box's n=17,984 is the first real
`bench_grid threshold` result for Zen4. Parallel: interpolated from bench_grid
(n=32,768 at 359 ms, n=65,536 at 928 ms).

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

- AOCL-FFTW (znver4-tuned, tag 5.3), sole FFT backend, 20-25% faster than plain FFTW
- BQ=8 batched linear with interleaved layout (native AVX-512 width)
- L2-aware checkpointing with 1MB per-core L2
- B=24/32, cost model adapts to Zen 4's wider schoolbook-FFT crossover.
  Empirical B-selection table in `devices/zen4/fft_config.h`. Note
  `results/b_optimal_report_zen4.md` is a **superseded historical artifact**: it
  validated the analytical `select_best_B` that was replaced by the empirical
  table one commit later, and its 21/34 split refers to that earlier, much
  smaller grid. The current table holds 1944 points. See the header of that
  report and `VERDICTS.md` V11.

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
> `FFT_OVERHEAD_NS`, the `*_BW_GBS` trio) are **read by nothing in `libicm.a`**;
> they are emitted leftovers awaiting deletion. `CLAUDE.md`'s constants table has
> the audited live/dead split; `HANDOFF.md` has the deletion plan. Values below
> are retained as a historical record, not as a description of live behaviour.

### M3 Pro (Apple Silicon, ARM64)

| Constant | Value | Notes |
|---|---|---|
| `FMA_NS` | 0.0500 | Scalar FMA cost. Fit lower bound, hit its limit when `WRAP_FMA_NS` and `FP64_DIV_NS` were pinned; see caveat below. |
| `WRAP_FMA_NS` | 0.4942 | Per-FMA cost for wrap correction. **Directly measured** via `tools/bench_wrap_fma.c`. |
| `FP64_DIV_NS` | 3.4890 | FP64 divide latency. **Directly measured** via `tools/bench_div_chain.c` (dependency-chained, not throughput). |
| `BLOCK_FMA_NS` | 0.4027 | FMA cost inside block build/divide. 7-param fit (both pins active). |
| `BLOCK_MEM_NS` | 0.1000 | Memory cost per element in block build/divide. |
| `PAIRED_CACHED_CORR_RATIO` | 1.9080 | Paired cached correlate cost / full FFT pipeline cost. |
| `INDEP_PAIR_RATIO` | 1.9080 | Independent pair correlate cost / full FFT pipeline cost. Equal to PAIRED, likely a fitting artifact (solver couldn't separate them). |
| `LEAF_FMA_NS` | 0.0727 | FMA cost at tree-leaf schoolbook multiplies. 7-param fit. |
| `LEAF_BLOCK_NS` | 48.1032 | Per-block overhead at leaf level. |
| `FFT_OVERHEAD_NS` | 631.0974 | Per-call FFT overhead. Physically odd value, pushed here to compensate when both pins are active; see caveat. |

> FFT calibration table (`calib_sizes[]`/`calib_times_ns[]`) and FFTW wisdom
> in `devices/m3_pro/fft_config.h` are from a genuine FFTW PATIENT calibration
> on this Apple M3 Pro machine (July 2026). `WRAP_FMA_NS` and `FP64_DIV_NS` are
> direct microbenchmark measurements, not recovered from aggregate regression ,
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
| `WRAP_FMA_NS` | 0.40 | Per-FMA cost for wrap correction. **Directly measured** via `tools/bench_wrap_fma.c`, extracted as least-squares slope over the decision-relevant range `wrap_m ∈ [64,384]`. |
| `FP64_DIV_NS` | 12.5287 | FP64 divide latency. From the unpinned 8-param fit, not independently cross-checked against a direct measurement this session. |
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
> aggregate `sample_plans` data (the old fit value 0.8612 was arbitrary ,
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
2026-07-22). `devices/zen4/fft_config.h` now contains sane values
(`L2_BW_GBS=131.5`, `L3_BW_GBS=56.0`, `DRAM_BW_GBS=33.0`, all in GB/s), directly
measured on an AMD Ryzen 9 7950X with the compiler-barrier fix applied. These
constants feed `blended_bandwidth()` in `src/cost_model.h`, affecting
`select_engine()` dispatch cost for the linear engine, not correctness
(`./bench_grid verify` is unaffected regardless).

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
`FFT_OVERHEAD_NS` is 0.0 on every device. The identifiability failure recorded
here is the *motivation* for that migration, not a standing caveat.

---

## NVIDIA B200 GPU (sm_100, cuFFTDx fused kernels, CUDA graph capture)

> The linear engine is CPU-only (sequential player-by-player structure can't
> saturate GPU parallelism). Only the tree-based engines map to the GPU; the
> planner assigns each subproduct-tree level to one of three kernel tiers
> (schoolbook, cuFFTDx fused, batched cuFFT) based on polynomial degree.

### Performance (ms, Q=256, FP64): systematic (n, k) grid

| n | k=64 | k=1024 | k=n/2 | k=n |
|---|------|--------|-------|-----|
| 4,096 | 0.35 | 0.74 | 0.82 | 0.84 |
| 16,384 | 1.16 | 2.79 | 3.85 | 4.10 |
| 65,536 | 4.35 | 10.94 | 19.36 | 20.22 |
| 262,144 | 17.09 | 43.37 | 96.31 | 100.49 |
| 1,048,576 | 77.45 | 187.72 | 508.24 | 513.41 |
| 4,194,304 | 272.98 | 771.48 | 2,383.66 | 2,340.89 |
| 16,777,216 | 1,280.52 | 3,127.15 | 10,716.87 | 10,825.24 |
| 33,554,432 | 2,639.85 | 6,141.28 | 22,584.75 | 23,116.73 |

Full 211-point calibration heatmap (`results/gpu_heatmap_b200_20260728.csv`),
regenerated 2026-07-27/28 on top of the ragged-tree fix (`b53dd17`) and the
extended B-selection table above n=1,572,864 (`b06379e`) -- all 210 cells
pass with zero errors, cv ≤ 0.036 everywhere (most 0.000), and exactly one
negligible non-monotonicity (n=4,194,304, k=2048→4096: 967.9→939.0ms, a
~3% single-sample wobble with identical B=128 and identical tier
composition on both sides, not a structural issue like the earlier
FFT-calibration-gap non-monotonicity).

**This session's two GPU fixes combined are a large, broad win, not just a
correctness fix.** Diffing this heatmap cell-by-cell against the previous
(2026-07-26, pre-fix) run at the same 210 `(n,k)` points: every cell with
n ≥ 2,097,152 -- previously dispatched to the stale `B=112` anchor beyond
the old B-selection table's range -- now dispatches `B=128` and is
**1.84x-2.34x faster**, e.g. the largest cell (n=k=33,554,432) drops from
45,046ms to 23,117ms. Zero cells regressed (none slower by more than 1%);
smaller cells already inside the pre-existing calibration range (n ≤
1,048,576) are unchanged (within ~1% noise), as expected. Summed over all
210 common cells, total grid time drops from 726.8s to 451.7s (1.61x
aggregate). The two fixes landed together (`b53dd17` then `b06379e`), so
this comparison cannot cleanly separate the register-pressure fix's
contribution from the B-selection table's -- both changed between the old
and new run -- but the combined effect at large n is unambiguous and
substantial.

### 1-second threshold: real binary search (median of 5 reps/candidate)

| Query shape | Largest n ≤ 1000ms | Smallest n > 1000ms | Bracket width |
|---|---|---|---|
| k = n | 1,490,944 (918.8ms) | 1,506,304 (1,124.2ms) | 15,360 (1.0% of lo) |
| k = 100 | 7,975,936 (998.9ms) | 7,991,296 (1,001.1ms) | 15,360 (0.2% of lo) |

Full search trace in `results/gpu_threshold_search_20260728.txt`
(`tools/threshold_search_gpu.cu`, plan-based API). Supersedes the old
2026-07-25 frontier-probe-derived estimates (n≈1,441,792 / n≈6,291,456),
which predated the ragged-tree fix and were reinterpretations of 5 fixed
sample points, not an actual search.

### Dispatch: three-tier kernel planner (schoolbook / cuFFTDx fused / batched cuFFT), cost-based per tree level

GPU cost-model constants (`C_wrap`, `C_school`, `R`, `C_gap`) are fit
separately from the CPU model via `tools/fit_gpu_cost_model.py` against
empirical kernel benchmarks in `devices/b200/gpu_fft_config.h` -- see
"GPU Cost Model (B200)" in `OPTIMIZATION_GUIDE.md` for the full pipeline.

> **Diagnostic pass (July 2026):** The GPU planner was confirmed NOT to have the
> CPU's wrap-correction cost-model bug. `src/gpu/gpu_plan.cu` uses one constant
> (`GPU_SCHOOL_FMA_NS`) uniformly in both joint and independent paths, no
> code-level asymmetry. Additionally, the GPU's fitted `C_wrap` is
> diagnostic-only (`fit_gpu_cost_model.py` never writes it to any config
> header), so even if under-identified it has zero effect on real planning.
> No GPU numbers changed this session.

