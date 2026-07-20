[![CI](https://github.com/Sarose550/ICM/actions/workflows/ci.yml/badge.svg)](https://github.com/Sarose550/ICM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# ICM -- Independent Chip Model Equity Computation

High-performance C library for computing tournament placement equities using generating-function quadrature. Computes exact ICM equities for poker tournaments with up to ~17,216 players / payouts in 1 second*.

## Quick Start

```bash
# Build (requires FFTW3)
make

# Verify correctness
./bench_grid verify

# Full benchmark grid
./bench_grid
```

## API

```c
#include "icm.h"

// Initialize (call once -- loads FFTW wisdom, builds lookup tables)
icm_init("fftw_wisdom.dat");

// Compute equities for all n players
//   S[n]       -- chip stacks
//   Q          -- quadrature points (typically 256)
//   payout[k]  -- payout coefficients
//   equity[n]  -- output (caller-allocated)
double ns = icm_equity(n, S, Q, payout, k, equity);

// Compute equities for a subset of players
double ns = icm_equity_subset(n, S, Q, payout, k, equity, targets, n_targets);
```

Returns wall-clock time in nanoseconds. All correctness tests pass at < 5e-12 relative error.

## How It Works

**1. The problem.** A tournament has `n` players with chip stacks
`S_1, ..., S_n` and a payout structure `(π_1, ..., π_k)` where `k ≤ n`
positions receive nonzero prizes. ICM computes each player's expected
payout - the sum over finishing positions of the prize for that position
times the probability the player finishes there. The naive answer enumerates
all `n!` elimination orderings, weights each by its probability under the
Malmuth–Harville model (Harville, 1973; Malmuth, 2001), and sums the
resulting payouts.

**2. Why naive enumeration is dead almost immediately.** `n!` grows
catastrophically fast: 15! ≈ 1.3 × 10^12, 20! ≈ 2.4 × 10^18. As the
HoldemResources.net blog post ["High Accuracy ICM Calculations for Large
Fields"](https://www.holdemresources.net/blog/high-accuracy-mtt-icm/)
notes, "Naive implementations of ICM can handle about 15 players, and even
optimized versions can't calculate exact Malmuth-Harville values beyond
25-30 players." The naive enumeration wall is around `n ≈ 15` - any attempt
to enumerate all orderings for 16+ players runs into years of compute time.

**3. The industry-standard exact method: bitmask dynamic programming.**
Rather than enumerating orderings, one can track the set of players who
have busted so far. Let `dp[mask]` be the probability that exactly the
players in the bitmask `mask` have been eliminated. From each state, for
each surviving player `j`, the transition adds `dp[mask] · (S_j / total
remaining stack)` to `dp[mask | (1<<j)]`. There are `2^n` states and up to
`n` candidate transitions per state, giving `O(n · 2^n)` total work - the
per-state cost comes from looping over surviving players, not from the
payout structure. This is the method used by real poker tools; see GTO
Wizard's ["Theoretical Breakthroughs in
ICM"](https://blog.gtowizard.com/theoretical-breakthroughs-in-icm/) post
for a practical discussion and Helmuth Melcher's 2015 TU Wien diploma
thesis ["Evaluation of Equity Models for Tournament
Poker"](https://repositum.tuwien.at/handle/20.500.12708/79991) for the
academic writeup. The practical wall is roughly 25–30 players - at
`n = 30`, `2^30` states already pushes into gigabytes of memory.

**4. How real tools scale past 30 players: Monte Carlo via the "exponential
clock" framing.** The Malmuth–Harville elimination rule ("at each step, the
probability any surviving player busts next is proportional to their
stack") has an equivalent continuous-time formulation. Assign each player
`j` an independent exponential random variable `T_j` with rate equal to
their chip stack `S_j` (an "elimination clock"), and eliminate players in
order of increasing `T_j`. The memoryless property of the exponential
distribution guarantees this recovers exactly the same stack-proportional
elimination rule at every step. This gives a simple, unbiased way to
*sample* a full elimination order in one shot - draw `n` exponentials, sort
- instead of simulating step-by-step. Tysen Streib introduced this
technique in a TwoPlusTwo forum thread, ["New Algorithm: Calculate ICM
Large
Tournaments"](https://forumserver.twoplustwo.com/15/poker-theory-amp-gto/new-algorithm-calculate-icm-large-tournaments-1098489/).
Error shrinks as `O(1/√N)` in the number of sampled tournaments `N`, so
high precision gets expensive. (A refinement uses [Quasi-Monte Carlo
sampling](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method) -
deterministic low-discrepancy point sequences instead of independent random
draws, giving closer to `O(1/N)` convergence for smooth integrands - but
this repo does not build on that approach.)

**5. This repo's approach: make the Monte Carlo estimate exact.** The
exponential-clock model above yields an integral representation of "player
`i` finishes in position `r`." Substituting `v = e^(-t)` turns the
combinatorial sum inside that integral into the coefficient of a generating
function. For player `i`, define:

```
a_j(v) = v^(S_j),    b_j(v) = 1 - v^(S_j)
Q_i(x; v) = Π_{j ≠ i} (a_j(v) + b_j(v) · x)
```

The coefficient of `x^m` in `Q_i(x; v)` captures exactly the combinatorial
term that Monte Carlo would otherwise have to sample - the sum over all
subsets of `m` other players of the product of their elimination
probabilities times the remaining players' survival probabilities. So
instead of drawing `N` random samples of the exponential race and
averaging, this repo evaluates the exact 1-D integral over `v` via
quadrature (after a change of variables `v = Φ(y)` using the standard
normal CDF to make the integrand decay rapidly). With `Q = 256`
Gauss-Legendre nodes, this yields deterministic double-precision accuracy
(relative error < 5 × 10^(-12); see Accuracy section below).

The remaining computational challenge is evaluating that degree-`k`
polynomial product for all `n` players efficiently, which is what the three
CPU engines and the FFT-accelerated subproduct tree exist to do. FFT-based
convolution multiplies two degree-`d` polynomials in `O(d log d)` time
instead of `O(d²)`, and the subproduct tree uses this to compute all
leave-one-out products simultaneously. (See Wikipedia's article on the
[Fast Fourier
Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) for
background on why FFT convolution beats the naive approach.)

**Three CPU engines with cost-based dispatch.** The library picks the
fastest engine per `(n, k)` pair via `select_engine()`, which compares a
roofline linear-cost estimate against a calibrated hybrid-cost model - no
hand-tuned crossover thresholds:

1. **Linear (batched):** `O(nk)` forward-backward pass. Interleaves BQ=8
   quadrature points for SIMD (NEON on Apple Silicon, AVX-512 on Zen 4).
   Best for small `k`.
2. **Hybrid (block + tree):** Partitions `n` players into cost-model-selected
   blocks, builds block products sequentially, then runs an FFT-accelerated
   binary subproduct tree over the blocks. Best for large `k`.
3. **Tree (pure FFT):** FFT-accelerated subproduct tree without blocking.
   Slightly slower than hybrid in serial but wins in parallel.

**The FFT infrastructure.** All tree operations use offline-calibrated
per-size FFT costs across the 7-smooth (`2^a · 3^b · 5^c · 7^d`) size
family. `best_fft_config()` searches for the optimal FFT size including
wrap-correction tradeoffs: a smaller FFT plus a schoolbook correction for
aliased terms often beats padding to the next power of two. The propagation
phase shares the forward FFT of the parent g-vector across both children
and reuses cached FFT(P) from the build phase.

**GPU path (cuFFTDx fused-kernel).** On NVIDIA B200/H200, `src/gpu/`
implements a planner, execution engine, and API. The planner assigns each
tree level to one of three tiers - schoolbook (small degrees), cuFFTDx
fused kernels (medium), or batched cuFFT (large) - and executes via CUDA
graph capture for near-zero launch overhead. See the Performance tables
below for current throughput numbers.

## Accuracy

The library is validated against exact closed-form reference values for two
special payout structures, not against a slow general-purpose reference
(which would cap validation at ~20–30 players). These closed forms are exact
for *any* `n` because they follow from linearity of expectation over pairs
and triples of players, not from enumerating elimination orderings:

- **V1 (linear payout, `payout[m] = n - m`):** The exact equity for player
  `i` is `1 + Σ_{j ≠ i} S_i / (S_i + S_j)` - a pairwise sum computable in
  `O(n²)` total. Each term is the probability that `i` outlasts `j` in a
  head-to-head race under the MH model, and linearity of expectation sums
  them.

- **V2 (quadratic payout, `payout[m] = C(n-1-m, 2)`):** The exact equity is
  a sum over all pairs `{j, k}` (both ≠ `i`) of `S_i / (S_i + S_j + S_k)` -
  the same idea one order up, computable in `O(n³)`, exact for the same
  reason.

These are implemented as `v1_exact()` and `v2_exact()` in `src/icm.c`
(publicly exposed as `icm_v1_exact()` / `icm_v2_exact()` in `icm.h`).
The tool `tools/accuracy_bench.c` sweeps the quadrature node count `Q`
and reports convergence against both closed forms across four stack
distributions: uniform (all stacks equal), adversarial (100:1 ratio),
geometric, and an extreme 1e9:1 adversarial case.

**Headline result:** Gauss-Legendre quadrature converges to ~5 × 10^(-13)
relative error by `Q = 1024` against both V1 and V2 closed forms across all
tested distributions. The convergence is rapid - here are representative
rows from `results/accuracy_m3max_20260718.csv` for the `gauss` scheme on
uniform stacks (V1 payout):

| Q | max_rel_err (n=4, uniform, V1) |
|---|-------------------------------|
| 4 | 4.10 × 10^0 |
| 8 | 4.36 × 10^(-1) |
| 16 | 1.32 × 10^(-1) |
| 64 | 3.08 × 10^(-8) |
| 128 | 6.79 × 10^(-13) |
| 256 | 8.87 × 10^(-13) |
| 1024 | 1.07 × 10^(-12) |

At `Q = 1024`, the maximum relative error across *all* tested configurations
(`n` up to 20, all four stack distributions, both V1 and V2) stays below
~2 × 10^(-12) for uniform stacks and below ~6 × 10^(-13) for the adversarial
and 1e9:1 cases. The production default is `Q = 256`, which already delivers
sub-2 × 10^(-12) relative error - sufficient for any practical poker
application.

![Accuracy convergence](accuracy_convergence.png)

## Performance

Three engines with cost-based automatic dispatch:

| Engine | Strategy | Best for |
|--------|----------|----------|
| **Linear** (batched) | O(nk), BQ=8 quad-point batching, interleaved layout, L2-aware checkpointing | Small k |
| **Hybrid** (B=auto) | Block build + FFT tree + bidirectional divide, calibrated block size | Large k |
| **Tree** (pure FFT) | FFT-accelerated subproduct tree | Parallel workloads |

`select_engine(n, k)` chooses the optimal engine for each (n, k) pair based on calibrated FFT costs and hardware parameters. No manual tuning required.

### Single-threaded (ms, Q=256)

| n | k=10 | k=100 | k=n/2 | k=n | | k=10 | k=100 | k=n/2 | k=n |
|---|------|-------|-------|-----|-|------|-------|-------|-----|
| | **M3 Pro (recalibrating)** |||| | **Zen 4 7950X** ||||
| 1024 | - | - | - | - | | 1.28 | 7.61 | 29.1 | 34.0 |
| 4096 | - | - | - | - | | 7.32 | 28.3 | 161 | 168 |
| 8192 | - | - | - | - | | 14.5 | 53.4 | 376 | 382 |
| 16384 | - | - | - | - | | 29.6 | 112 | 866 | 835 |
| 65536 | - | - | - | - | | 117 | 419 | 4170 | 4490 |

*M3 Pro numbers are being recalibrated after a hardware migration - this table will be refreshed once that run completes.*

### 16-thread parallel (ms, Q=256)

| n | k=10 | k=100 | k=n/2 | k=n | | k=10 | k=100 | k=n/2 | k=n |
|---|------|-------|-------|-----|-|------|-------|-------|-----|
| | **M3 Pro (recalibrating)** |||| | **Zen 4 7950X** ||||
| 1024 | - | - | - | - | | 0.125 | 0.562 | 2.23 | 2.48 |
| 4096 | - | - | - | - | | 0.604 | 2.35 | 11.4 | 11.9 |
| 8192 | - | - | - | - | | 1.18 | 4.86 | 26.6 | 26.9 |
| 16384 | - | - | - | - | | 2.39 | 10.4 | 67.5 | 81.2 |
| 65536 | - | - | - | - | | 19.5 | 45.2 | 530 | 631 |

*M3 Pro numbers are being recalibrated after a hardware migration - this table will be refreshed once that run completes.*

See [RESULTS.md](RESULTS.md) for complete performance tables across platforms.

## Building

### macOS (Apple Silicon)

```bash
# Serial
make

# Parallel (requires: brew install libomp)
make parallel
```

Uses Accelerate framework (vDSP) for FFT dispatch at supported sizes.

### Linux

```bash
# Install FFTW3
sudo apt-get install libfftw3-dev    # Debian/Ubuntu
sudo dnf install fftw-devel          # Fedora/RHEL

# Serial
make

# Parallel
make parallel
```

Automatically detects MKL (via dlopen) for dual-dispatch FFT when available.

### Linux with AOCL-FFTW (AMD)

```bash
# Install AOCL-FFTW to /usr/local/aocl-fftw
make DEVICE=zen4
make DEVICE=zen4 parallel
```

Auto-detected if installed at `/usr/local/aocl-fftw`.

### GPU (NVIDIA)

```bash
make bench_gpu_fused CUDA_ARCH=sm_100    # B200/B100
make bench_gpu_fused CUDA_ARCH=sm_90     # H100/H200
```

Requires CUDA toolkit and cuFFTDx.

**B200 performance** (Q=256, fused cuFFTDx kernels):

| n | k=n | Time |
|---|-----|------|
| 65,536 | 65,536 | 24.75 ms |
| 262,144 | 262,144 | 117.90 ms |
| 1,441,792 | 1,441,792 | 866 ms |
| 1,572,864 | 1,572,864 | 1,148 ms |

See `devices/b200/gpu_fft_config.h` for calibration data.

## Calibrating for a New Device

If your hardware matches an already-calibrated device (`devices/m3_pro`, `devices/zen4`), you don't need to run `./calibrate` at all - build straight against the shipped wisdom and config:

```bash
make DEVICE=m3_pro   # or zen4 - whichever matches your machine
./bench_grid verify
./bench_grid crossover   # confirm dispatch decisions match measured winners on YOUR unit
```

`fftw_wisdom.dat` and the `calib_times_ns[]` table are measured on one specific physical machine. FFTW will happily load wisdom from a different unit of the same CPU model - it just isn't guaranteed to have picked the fastest codelet for *your* silicon, and the nanosecond timings the cost model reads for FFT-vs-schoolbook and engine-dispatch decisions won't necessarily match your machine's actual behavior (different DIMM speed, microcode revision, thermal/boost profile, or memory bandwidth can all shift these numbers). `./bench_grid crossover` is the check that catches this: if every cell's dispatch decision agrees with the measured winner, the shipped calibration is good enough and you're done. Only recalibrate from scratch (below) if it disagrees - and definitely recalibrate if you're on hardware unlike anything already in `devices/`.

```bash
# Generate calibration data
gcc -O3 -march=native -o calibrate tools/calibrate.c -lfftw3 -lm
./calibrate

# Copy to device directory
mkdir -p devices/mydevice
cp fft_config.h fftw_wisdom.dat devices/mydevice/

# Build and verify
make DEVICE=mydevice
./bench_grid verify
./bench_grid profile    # measure FMA_NS, FFT_OVERHEAD_NS, etc.
```

Update the `#define` constants in `fft_config.h` with measured values from `./bench_grid profile`. See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) for details on each constant.

## Python Bindings

Python bindings are in `python/`. Build the shared library first:

```bash
make libicm.a
```

## Project Structure

```
src/icm.h                    -- public CPU API
src/icm.c                    -- all CPU engines + FFT infrastructure
src/linear_batched_impl.inc  -- batched linear engine template
src/icm_gpu.h                -- GPU API header
src/gpu/                     -- GPU implementation (split modules)
  gpu_internal.h             -- shared GPU types and helpers
  gpu_kernels.cu             -- CUDA kernels
  gpu_plan.cu                -- GPU planner and cost model
  gpu_exec.cu                -- GPU execution engine
  gpu_api.cu                 -- GPU public API
bench/bench.c                -- CPU benchmark + verification harness
bench/bench_gpu.cu           -- GPU benchmark + verification harness
tools/calibrate.c            -- FFTW calibration tool
tools/calibrate_gpu.cu       -- GPU FFT calibration tool
devices/                     -- per-device calibration data
python/                      -- Python ctypes bindings
```

## Documentation

- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) -- detailed optimization notes, porting guide, and algorithm descriptions
- [RESULTS.md](RESULTS.md) -- complete performance tables, head-to-head comparisons, and phase-split analysis

## License

MIT. See [LICENSE](LICENSE).

---
\* Single-threaded, AMD Ryzen 9 7950X.
