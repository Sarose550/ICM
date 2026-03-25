# CLAUDE.md

## Project

ICM (Independent Chip Model) equity computation for poker tournaments.
High-performance C library computing tournament placement equities using
generating function quadrature.

## Build

```bash
# Serial (macOS / Apple Silicon)
make

# Parallel (macOS with libomp — requires `brew install libomp`)
make parallel

# Different device (uses devices/<DEVICE>/fft_config.h)
make DEVICE=zen4

# Calibrate a new device (generates fft_config.h + fftw_wisdom.dat)
gcc -O3 -march=native -o calibrate tools/calibrate.c -lfftw3 -lm
./calibrate
```

## Test

```bash
./bench_grid verify    # correctness (V1 + V2 + cross-check, up to n=65536)
./bench_grid quick     # fast grid subset
./bench_grid           # full grid (n up to 65536)
./bench_grid crossover # linear vs hybrid crossover sweep
./bench_grid cliff     # power-of-2 scaling test
./bench_grid threshold # binary search for 1-second boundary
./bench_grid profile   # FFT overhead + phase split + per-engine profiling
OMP_NUM_THREADS=16 ./bench_grid   # parallel
```

## Architecture

Three engines with automatic dispatch:
- **Linear** (batched): O(nk), best for small k. Quad-point batched for n >= 2048.
  Fused backward pass (dot product + suffix update in one loop).
- **Hybrid** (B=8): Block build + FFT tree + bidirectional divide. Best for large k.
  Players sorted by stack size. Paired cached correlate shares both FFT(g) across
  siblings and FFT(P) from the build phase. Per-level FFT decisions use calibrated
  data from fft_config.h (no global crossover). k padded to fastest smooth size
  via `best_k_pad()`.
- **Tree** (pure): FFT-accelerated subproduct tree. Slightly slower than hybrid
  in serial, but wins in parallel (simpler clone, same PATIENT-quality plans).

Dispatch rule: `k >= 95 && n >= 2048` or `k >= 70 && n >= 256` → hybrid, else → linear.
(Batched linear's fused backward pass + NEON raises the crossover to k≈95 for n ≥ 2048.)

Shared helpers: `tree_build_levels()` and `tree_propagate_g()` are used by both the
tree and hybrid engines to avoid code duplication.

## Public API (icm.h)

```c
void   icm_init(const char *wisdom_path);
double icm_equity(int n, const double *S, int Q, const double *payout, int k, double *equity);
double icm_equity_subset(int n, const double *S, int Q, const double *payout, int k,
                         double *equity, const int *targets, int n_targets);
```

## Key constraint

**DO NOT use polynomial division for top-k (k < n).** Synthetic division
is numerically unstable when a_i < 0.5. The hybrid engine uses division
only within B=8 blocks on the COMPLETE (non-truncated) block product,
where amplification is bounded by |c|^B ≈ 10^7 (safe for FP64).

## Device-specific constants

All tuning constants live in `devices/<DEVICE>/fft_config.h` as `#define`s:

| Constant | M3 Max | What | How to measure |
|---|---|---|---|
| `FMA_NS` | 0.25 | ns per scalar FMA | `./bench_grid profile` (schoolbook row) |
| `FFT_OVERHEAD_NS` | 40.0 | Per-call FFT overhead | `./bench_grid profile` (overhead table) |
| `PAIRED_CACHED_CORR_RATIO` | 1.03 | Paired correlate / full pipeline | `./bench_grid profile` (phase split) |
| `INDEP_PAIR_RATIO` | 1.25 | correlate_fft_pair / full pipeline | `./bench_grid profile` (phase split) |
| `calib_sizes[]` / `calib_times_ns[]` | 749 entries | Per-size FFT costs | `tools/calibrate` |

Additional constants hardcoded in `icm.c` that may need per-device tuning:

| Constant | M3 Max | What |
|---|---|---|
| `BQ` | 2 | Quad-point batch width (NEON=2, AVX2=4, AVX-512=8) |
| `CKPT_THRESHOLD` | 4194304 | Linear checkpointing threshold (doubles; 32MB on M3 Max) |
| `B=8` in `icm_equity()` | 8 | Hybrid block size |
| `k_cross` (95/70) | 95/70 | Linear→hybrid crossover |
| `OMP_NUM_THREADS_DEFAULT` | 16 | Default thread count |

## Directory Structure

```
src/
  icm.h                — public API header
  icm.c                — library implementation (all engines, FFT infrastructure)
bench/
  bench.c              — benchmark harness, verification, tuning tools
tools/
  calibrate.c          — FFT calibration (generates fft_config.h + wisdom)
devices/
  m3_max/              — Apple M3 Max calibration
    fft_config.h       — calibrated FFT times + cost model constants
    fftw_wisdom.dat    — FFTW PATIENT plans
  zen4/                — AMD Zen 4 (placeholder — needs calibration)
  h200/                — NVIDIA H200 GPU (CUDA port — not yet implemented)
Makefile               — build system
RESULTS.md             — performance grid and optimization history
OPTIMIZATION_GUIDE.md  — detailed optimization notes + porting guide
archive/               — historical prototypes
```

## Correctness invariant

After ANY change, run `./bench_grid quick` and confirm ALL TESTS PASSED.

## Porting to a new device

1. Run `tools/calibrate` on the target machine → produces `fft_config.h` + `fftw_wisdom.dat`
2. Copy to `devices/<DEVICE>/`
3. `make DEVICE=<DEVICE>` and `./bench_grid profile` to measure platform constants
4. Update `#define`s in `fft_config.h` (FMA_NS, FFT_OVERHEAD_NS, etc.)
5. `./bench_grid crossover` to find optimal dispatch thresholds
6. Update `icm.c` dispatch constants if needed (k_cross, BQ, CKPT_THRESHOLD, B)
7. `./bench_grid verify` then `./bench_grid` for final numbers
8. Test Karatsuba: implement and benchmark a Karatsuba multiply at sizes 64-512
   to check if there's a regime between schoolbook and FFT. Prior: unlikely on
   AVX-512 (wide SIMD inflates schoolbook regime) but worth measuring.
   Search the internet for optimized AVX-512 Karatsuba implementations as reference.
9. FFT library: use BOTH FFTW and Intel MKL via dlopen, dispatch per-size to
   whichever is faster. Calibrate both, store per-size winner in fft_config.h.
   MKL often wins at power-of-2, FFTW at composites. See OPTIMIZATION_GUIDE.md
   for the dual-library dispatch implementation pattern.
