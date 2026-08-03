[![CI](https://github.com/Sarose550/ICM/actions/workflows/ci.yml/badge.svg)](https://github.com/Sarose550/ICM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# ICM: Independent Chip Model Equity Computation

High-performance C library for computing tournament placement equities using generating-function quadrature. Computes exact ICM equities for poker tournaments with up to 17,984 players / payouts in 1 second single-threaded, or ~72,200 across 16 threads (AMD Zen 4; see [RESULTS.md](RESULTS.md) for per-device figures). A CUDA backend extends this to up to 1.49 million players in under a second on an NVIDIA B200 (full field, `k=n`). Python bindings (ctypes, calling straight into the compiled shared library) are included for the CPU library.

> 📄 **Paper:** [Fast Tournament Equity Computation via Generating-Function Quadrature and FFT-Accelerated Subproduct Trees](paper/icm_paper.pdf) - full derivation, proofs, and performance evaluation.
>
> **Status:** arXiv submission pending.

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

// Initialize (call once; loads FFTW wisdom, builds lookup tables)
icm_init("fftw_wisdom.dat");

// Compute equities for all n players
//   S[n]       : chip stacks
//   Q          : quadrature points (typically 256)
//   payout[k]  : payout coefficients
//   equity[n]  : output (caller-allocated)
icm_equity(n, S, Q, payout, k, equity);

// Compute equities for a subset of players
icm_equity_subset(n, S, Q, payout, k, equity, targets, n_targets);
```

All correctness tests pass at < 1.6e-10 relative error.

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
`import icm`). These bindings cover the CPU library only; no Python
wrapper exists for the CUDA API below.

## CUDA API

```c
#include "icm_gpu.h"

// Initialize (call once; selects the CUDA device)
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
(`bench_gpu verify`). See [src/gpu/icm_gpu.h](src/gpu/icm_gpu.h) for the full API,
including the reusable `IcmGpuPlan` (amortizes planning cost across repeated
calls at the same `n`/`k`) and calibration/diagnostics helpers.

## Accuracy

Validated against exact closed-form reference equities for two payout
structures (linear and quadratic), exact for *any* `n` rather than by
enumerating elimination orderings, so validation isn't capped at the small
`n` a slow general-purpose reference would allow. The production default
(`Q = 256`, Gauss-Legendre quadrature) delivers under 1.6e-10 relative
error, including at a 1e9:1 stack-ratio extreme. See the paper for the full
quadrature-convergence study (Gauss-Legendre vs. tanh-sinh, four stack
distributions) and `results/accuracy_convergence.csv` for the raw sweep.

![Accuracy convergence](results/accuracy_convergence.png)

## Performance

**CPU, single-threaded (ms, Q=256, uniform stacks, median of 5):**

| n | k=10 | k=50 | k=100 | k=n/4 | k=n/2 | k=n | | k=10 | k=50 | k=100 | k=n/4 | k=n/2 | k=n |
|---|------|------|-------|-------|-------|-----|-|------|------|-------|-------|-------|-----|
| | **M3 Pro** |||||| | **Zen 4 7950X** (AOCL-FFTW) |||||
| 1024  | 1.70 | 7.17 | 13.1 | 18.0 | 21.2 | 28.0 | | 1.44 | 4.04 | 7.90 | 15.7 | 16.7 | 17.6 |
| 2048  | 4.12 | 14.4 | 26.3 | 44.9 | 52.1 | 56.4 | | 3.21 | 6.87 | 13.7 | 36.2 | 38.6 | 40.9 |
| 4096  | 8.24 | 28.7 | 52.5 | 109  | 124  | 137  | | 6.58 | 14.1 | 29.3 | 83.4 | 92.5 | 93.6 |
| 8192  | 16.4 | 57.2 | 105  | 284  | 302  | 321  | | 13.1 | 28.2 | 53.4 | 188  | 203  | 213  |
| 16384 | 32.7 | 114  | 210  | 636  | 712  | 753  | | 26.4 | 66.3 | 106  | 433  | 479  | 508  |
| 32768 | 64.8 | 232  | 417  | 1490 | 1660 | 1790 | | 52.3 | 127  | 228  | 980  | 1080 | 1230 |
| 65536 | 133  | 460  | 834  | 3500 | 3910 | 4150 | | 115  | 225  | 414  | 2580 | 2970 | 3330 |

**GPU, NVIDIA B200 (ms, Q=256):**

| n | k=64 | k=1024 | k=n/2 | k=n |
|---|------|--------|-------|-----|
| 4,096 | 0.37 | 0.76 | 0.87 | 0.89 |
| 16,384 | 1.18 | 2.81 | 3.93 | 4.17 |
| 65,536 | 4.29 | 9.46 | 19.43 | 20.26 |
| 262,144 | 16.58 | 36.38 | 95.95 | 100.09 |
| 1,048,576 | 65.68 | 178.26 | 504.29 | 509.51 |
| 4,194,304 | 272.47 | 671.06 | 2,352.43 | 2,320.45 |
| 33,554,432 | 2,506.71 | 5,059.26 | 22,321.49 | 22,865.76 |

See the paper for the full grids, contour plots, and dispatch analysis.

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

Uses system FFTW3. For AMD platforms, AOCL-FFTW is recommended, see below.

### Linux with AOCL-FFTW (AMD Zen 4)

```bash
# Install AOCL-FFTW to /usr/local/aocl-fftw
make DEVICE=zen4
make DEVICE=zen4 parallel
```

AOCL-FFTW is the sole FFT backend for Zen 4, a direct A/B test confirmed it is
cleanly faster than plain FFTW at every calibrated size. Auto-detected if
installed at `/usr/local/aocl-fftw`.

### GPU (NVIDIA)

```bash
make bench_gpu_fused CUDA_ARCH=sm_100    # B200/B100
make bench_gpu_fused CUDA_ARCH=sm_90     # H100/H200
```

Requires CUDA toolkit and cuFFTDx. See the [Performance](#performance) section
above for B200 timings, and `devices/b200/gpu_fft_config.h` for calibration
data.

## Platform Support

| Device | Architecture | FFT Backend | Status |
|--------|-------------|-------------|--------|
| Apple M3 Pro | ARM (Apple Silicon) | FFTW + vDSP dispatch at supported sizes | Calibrated, verified |
| AMD Ryzen 9 7950X (Zen 4) | x86-64 (AVX-512) | AOCL-FFTW | Calibrated, verified |
| NVIDIA B200 | Blackwell GPU (sm_100) | cuFFT + cuFFTDx | Calibrated, verified |

An uncalibrated device falls back to `devices/generic/` and still produces
correct results; every tree level uses FFT with `FFTW_ESTIMATE` plans, the
engine always dispatches hybrid with B=32, and a build-time warning is
printed. Run `./tools/calibrate_full.sh <DEVICE>` to add real calibration for
your hardware.

## Calibrating for a New Device

If your hardware matches an already-calibrated device, build straight
against the shipped wisdom and config, then confirm dispatch is still
correct on your specific unit:

```bash
make DEVICE=m3_pro   # or zen4 - whichever matches your machine
./bench_grid verify
./bench_grid crossover   # confirm dispatch agrees with measured winners on YOUR unit
```

If `crossover` disagrees, or your hardware isn't already in `devices/`, one
command runs the full calibration pipeline (FFTW calibration, hybrid-engine
timing, cost-model fitting) and finishes with a `verify` + `crossover`
check:

```bash
./tools/calibrate_full.sh mydevice   # add --quick for a faster, less precise pass
```

Expect 10-30+ minutes on an otherwise-idle machine (FFTW's PATIENT planner
dominates the time). See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md#porting-to-a-new-device-general)
for the manual step-by-step version and what each calibrated constant means.

**GPU (NVIDIA)** devices calibrate separately from the CPU pipeline; see
["GPU Cost Model (B200)"](OPTIMIZATION_GUIDE.md#gpu-cost-model-b200) in
`OPTIMIZATION_GUIDE.md`, or run `./tools/run_b200_campaign.sh` for the
whole pipeline in one shot.

## Python Bindings

Python bindings are in `python/`. Build the shared library first:

```bash
make libicm.a
```

## Project Structure

```
src/cpu/     : CPU engines (linear/hybrid/tree) + FFT infrastructure
src/gpu/     : CUDA implementation (kernels, planner, execution, API)
bench/       : benchmark + verification harnesses (CPU and GPU)
tools/       : calibration, validation, and diagnostic tools
devices/     : per-device calibration data (m3_pro, zen4, b200, generic)
python/      : Python ctypes bindings
results/     : benchmark results, CSVs, and plots
paper/       : paper PDF
```

## Documentation

- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md): detailed optimization notes, porting guide, and algorithm descriptions
- [RESULTS.md](RESULTS.md): complete performance tables, head-to-head comparisons, and phase-split analysis

## How It Works

The algorithm reformulates ICM equity as a one-dimensional integral over
generating-function coefficients, evaluated by Gaussian quadrature, and
computes the per-player leave-one-out products via an FFT-accelerated
subproduct tree. Three engines (linear, hybrid, tree) cover different
`(n, k)` regimes; which one runs is decided automatically at every level,
from offline-calibrated lookup tables rather than analytical cost formulas
(the same empirical-measurement-over-modeling approach FFTW's `PATIENT`
planner and LAPACK's `ILAENV` use). The GPU path (NVIDIA B200) uses
cuFFTDx fused device-side kernels with CUDA graph capture.

**For the full derivation, complexity analysis, correctness proofs, and
performance evaluation, see the paper:**
[**paper/icm_paper.pdf**](paper/icm_paper.pdf) (the dispatch mechanism is
in the Algorithm section). For an implementation-level walkthrough, see
[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md).

## Getting Help / Reporting Issues

Open an issue on the [GitHub repository](https://github.com/Sarose550/ICM/issues).
Include:

- Your device (CPU model or GPU model)
- The exact `make` line you ran (including `DEVICE=`)
- Output of `./bench_grid verify`

## Citation

There is no arXiv ID or published paper yet. Until then, cite the GitHub
repository and the in-repo PDF:

```bibtex
@misc{icm_2026,
  author       = {Sam Rosenstrauch},
  title        = {{ICM}: Independent Chip Model Equity Computation},
  howpublished = {\url{https://github.com/Sarose550/ICM}},
  note         = {Paper: \texttt{paper/icm\_paper.pdf}},
  year         = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
