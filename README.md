[![CI](https://github.com/Sarose550/ICM/actions/workflows/ci.yml/badge.svg)](https://github.com/Sarose550/ICM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# ICM -- Independent Chip Model Equity Computation

High-performance C library for computing tournament placement equities using generating-function quadrature. Computes exact ICM equities for poker tournaments with up to ~17,216 players / payouts in 1 second*. A CUDA backend extends this to over 1.5 million players in about a second on an NVIDIA B200. Python bindings (ctypes, calling straight into the compiled shared library) are included for the CPU library.

> 📄 **Paper:** [Fast Tournament Equity Computation via Generating-Function Quadrature and FFT-Accelerated Subproduct Trees](paper/icm_paper.pdf) — full derivation, proofs, and performance evaluation.

## What is ICM?

The Independent Chip Model (ICM) is a tournament equity model that converts
chip stacks into real-money expected payouts by accounting for the payout
structure. In a poker tournament, chips do not have a fixed dollar value - your last chip is worth far less than your first - and ICM computes each
player's fair expected share of the prize pool. For a general introduction,
see the [ICM Wikipedia page](https://en.wikipedia.org/wiki/Independent_Chip_Model).

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
icm_equity(n, S, Q, payout, k, equity);

// Compute equities for a subset of players
icm_equity_subset(n, S, Q, payout, k, equity, targets, n_targets);
```

All correctness tests pass at < 2e-10 relative error.

**Subset equity.** `icm_equity_subset()` computes equities for only a chosen
subset of players (`targets`) instead of all `n`. It prunes the hybrid
engine's propagate pass with a per-level hot/cold bitmask marking which
tree branches can contain a target player, skipping cold branches entirely - the sort order used by the rest of the engine is untouched, so this is
purely a pruning optimization, not a different algorithm. Worthwhile when
you only need a handful of players' equities out of a large field; the
speedup is workload-dependent (larger `n`, smaller target fraction helps
most).

**Python bindings.** `python/` provides a ctypes wrapper (`icm.equity(stacks, payouts)`)
that calls straight into the same compiled shared library the C API uses.
See [python/README.md](python/README.md) for setup (`make libicm`, then
`import icm`). These bindings cover the CPU library only -- no Python
wrapper exists for the CUDA API below.

## CUDA API

```c
#include "icm_gpu.h"

// Initialize (call once -- selects the CUDA device)
icm_gpu_init(/* device_id */ 0);

// Compute equities for all n players; opts=NULL uses defaults.
// Returns 0 on success, -1 on failure (check icm_gpu_last_error()).
// Timing is opt-in: pass a non-NULL stats to read stats.total_ns
// afterward, or NULL to skip it.
IcmGpuRunStats stats;
int status = icm_gpu_equity(n, S, Q, payout, k, equity, /* opts */ NULL, &stats);

icm_gpu_shutdown();
```

All correctness tests pass at < 1e-8 relative error against the CPU reference
(`bench_gpu verify`). See [src/icm_gpu.h](src/icm_gpu.h) for the full API,
including the reusable `IcmGpuPlan` (amortizes planning cost across repeated
calls at the same `n`/`k`) and calibration/diagnostics helpers.

## How It Works

The algorithm reformulates ICM equity as a one-dimensional integral over
generating-function coefficients, evaluated by Gaussian quadrature
($Q = 256$ nodes, relative error $< 5 \times 10^{-12}$). The central
challenge---computing leave-one-out polynomial products for all $n$ players
simultaneously---is solved by an FFT-accelerated binary subproduct tree
whose propagation phase is the adjoint of its build phase, reducing cost
from $O(nk)$ to $O(n \log^2 k)$ per quadrature point.

A roofline cost model dispatches automatically between a batched linear
engine (optimal for small $k$) and a hybrid block-tree engine (optimal for
large $k$). The GPU path (NVIDIA B200) uses cuFFTDx fused device-side
kernels with CUDA graph capture, computing 6.3 million player equities
($k = 100$) in 626 ms.

**For the full derivation, complexity analysis, correctness proofs, and
performance evaluation, see the paper:**
[**paper/icm_paper.pdf**](paper/icm_paper.pdf)

## Accuracy

Validated against exact closed-form reference equities (`v1_exact()`,
`v2_exact()` in `src/icm.c`) for two payout structures -- linear and
quadratic -- that are exact for *any* $n$ via linearity of expectation over
player pairs/triples, not by enumerating elimination orderings. This avoids
capping validation at the ~20-30 players a slow general-purpose reference
would allow.

`tools/accuracy_bench.c` sweeps the quadrature node count `Q` against both
closed forms across four stack distributions (uniform, 100:1 adversarial,
geometric, and an extreme 1e9:1 case). Gauss-Legendre quadrature (the
production choice) converges to $\sim 5 \times 10^{-13}$ relative error by
`Q = 1024` on all of them; tanh-sinh (double-exponential) quadrature converges
faster on easy distributions but stalls around $10^{-7}$-$10^{-8}$ on the
1e9:1 case and doesn't improve from `Q = 512` to `Q = 1024`, which is why
Gauss-Legendre is used in production rather than tanh-sinh. The production
default `Q = 256` already delivers sub-$2 \times 10^{-12}$ relative error on
uniform stacks and sub-$1.5 \times 10^{-10}$ at the 1e9:1 bound.

Full derivation (the V1/V2 closed forms, the exponential-clock argument they
rely on, and the complete Gauss-Legendre vs. tanh-sinh convergence tables)
is in the paper; raw sweep data is in `results/accuracy_convergence.csv`.

![Accuracy convergence](results/accuracy_convergence.png)

## Performance

Three engines with cost-based automatic dispatch:

| Engine | Strategy | Best for |
|--------|----------|----------|
| **Linear** (batched) | $O(nk)$, BQ=8 quad-point batching, interleaved layout, L2-aware checkpointing | Small k |
| **Hybrid** (B=auto) | Block build + FFT tree + bidirectional divide, calibrated block size | Large k |
| **Tree** (pure FFT) | FFT-accelerated subproduct tree | Parallel workloads |

`select_engine(n, k)` chooses the optimal engine for each (n, k) pair based on calibrated FFT costs and hardware parameters. No manual tuning required.

**Headline numbers.** Single-threaded, the library handles up to ~17,216
players (k=n) in one second on Zen 4 (see footnote). The GPU path pushes well
past that ceiling: the B200 computes 1,572,864 players (k=n) in 1,148 ms -
roughly 1.5 million players in about a second, a ~90x increase in problem
size at comparable latency.

![Runtime vs n at fixed k, NVIDIA B200](results/runtime_vs_n_gpu.png)

Full benchmark grids (all n/k combinations, both CPU platforms, serial and
parallel), 1-second contour plots, engine-dispatch visualizations, and the
GPU heatmap: see [RESULTS.md](RESULTS.md) for the raw tables, or the paper
for the full evaluation and figures.

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

**B200 performance** (Q=256, fused cuFFTDx kernels, n=k):

| n | Time |
|---|------|
| 65,536 | 20.64 ms |
| 262,144 | 101.30 ms |
| 1,441,792 | 866 ms |
| 1,572,864 | 1,148 ms |

See `devices/b200/gpu_fft_config.h` for calibration data.

## Calibrating for a New Device

If your hardware matches an already-calibrated device (`devices/m3_pro`, `devices/zen4`), you don't need to run `./calibrate` at all - build straight against the shipped wisdom and config:

```bash
make DEVICE=m3_pro   # or zen4 - whichever matches your machine
./bench_grid verify
./bench_grid crossover   # confirm dispatch decisions match measured winners on YOUR unit
```

`fftw_wisdom.dat` and the `calib_times_ns[]` table are measured on one specific physical machine. FFTW will happily load wisdom from a different unit of the same CPU model - it just isn't guaranteed to have picked the fastest codelet for *your* silicon, and the nanosecond timings the cost model reads for FFT-vs-schoolbook and engine-dispatch decisions won't necessarily match your machine's actual behavior (different DIMM speed, microcode revision, thermal/boost profile, or memory bandwidth can all shift these numbers). `./bench_grid crossover` is the check that catches this: if every cell's dispatch decision agrees with the measured winner, the shipped calibration is good enough and you're done. Only recalibrate from scratch (below) if it disagrees - and definitely recalibrate if you're on hardware unlike anything already in `devices/`.

One command runs the whole pipeline (FFTW calibration, hybrid-engine timing,
and cost-model constant fitting) and finishes with a `verify` + `crossover`
check:

```bash
./tools/calibrate_full.sh mydevice   # add --quick for a faster, less precise FFTW pass
```

If you want to see (or run) each step by hand
instead:

```bash
# Generate calibration data
# macOS: add -I/opt/homebrew/include -L/opt/homebrew/lib (Homebrew FFTW)
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
