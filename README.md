[![CI](https://github.com/samrosenstrauch/icm/actions/workflows/ci.yml/badge.svg)](https://github.com/samrosenstrauch/icm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# ICM -- Independent Chip Model Equity Computation

High-performance C library for computing tournament placement equities using generating-function quadrature. Computes exact ICM equities for poker tournaments with up to 65,536 players.

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
| | **M3 Max** |||| | **Zen 4 7950X** ||||
| 1024 | 3 | 16 | 24 | 27 | | 5 | 15 | 22 | 24 |
| 4096 | 7 | 46 | 137 | 147 | | 7 | 28 | 119 | 130 |
| 8192 | 14 | 115 | 318 | 350 | | 15 | 56 | 287 | 296 |
| 16384 | 28 | 230 | 709 | 752 | | 29 | 114 | 640 | 690 |
| 65536 | 135 | 937 | 4017 | 4392 | | 142 | 510 | 3473 | 3861 |

### 16-thread parallel (ms, Q=256)

| n | k=10 | k=100 | k=n/2 | k=n | | k=10 | k=100 | k=n/2 | k=n |
|---|------|-------|-------|-----|-|------|-------|-------|-----|
| | **M3 Max** |||| | **Zen 4 7950X** ||||
| 1024 | 0 | 2 | 2 | 3 | | 0 | 1 | 2 | 2 |
| 4096 | 2 | 7 | 14 | 15 | | 2 | 4 | 9 | 9 |
| 8192 | 4 | 14 | 33 | 37 | | 4 | 8 | 21 | 24 |
| 16384 | 9 | 28 | 81 | 90 | | 8 | 17 | 59 | 67 |
| 65536 | 39 | 114 | 501 | 594 | | 36 | 197 | 423 | 551 |

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

# Optional: VkFFT dual-dispatch (cuFFT + VkFFT per-size selection)
make bench_gpu_fused CUDA_ARCH=sm_100 VKFFT_INC="-I/path/to/vkfft"
```

Requires CUDA toolkit and cuFFTDx. VkFFT is optional for dual-dispatch FFT
(per-size best-of-both selection, similar to CPU FFTW+MKL dual dispatch).

**B200 performance** (Q=256, fused cuFFTDx kernels):

| n | k=n | Time |
|---|-----|------|
| 65,536 | 65,536 | 25 ms |
| 262,144 | 262,144 | 118 ms |
| 1,572,864 | 1,572,864 | 981 ms |

See `devices/b200/gpu_fft_config.h` for calibration data.

## Calibrating for a New Device

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
src/amx.h                    -- Apple AMX FP64 outer-product primitives
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
tools/bench_vkfft.cu         -- VkFFT benchmark (optional)
devices/                     -- per-device calibration data
python/                      -- Python ctypes bindings
```

## Documentation

- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) -- detailed optimization notes, porting guide, and algorithm descriptions
- [RESULTS.md](RESULTS.md) -- complete performance tables, head-to-head comparisons, and phase-split analysis

## License

MIT. See [LICENSE](LICENSE).
