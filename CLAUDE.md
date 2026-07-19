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

# Dual-library calibration (FFTW vs MKL, Linux only — run after calibrate)
gcc -O3 -march=native -o calibrate_dual tools/calibrate_dual.c -ldl -lm
MKL_THREADING_LAYER=SEQUENTIAL ./calibrate_dual

# GPU (NVIDIA, requires CUDA toolkit + cuFFTDx)
make bench_gpu_fused CUDA_ARCH=sm_100    # B200/B100
make bench_gpu_fused CUDA_ARCH=sm_90     # H100/H200

# GPU with VkFFT dual-dispatch (optional — cuFFT + VkFFT per-size selection)
make bench_gpu_fused CUDA_ARCH=sm_100 VKFFT_INC="-I/path/to/vkfft"
```

Note: The GPU code lives in `src/gpu/` (split into 4 modules + internal header).
The old monolithic `src/icm_gpu.cu` is kept for reference but the Makefile
compiles from the split files via separate compilation + device linking.

## Test

```bash
./bench_grid verify    # correctness (V1 + V2 + cross-check, up to n=65536)
./bench_grid quick     # fast grid subset
./bench_grid           # full grid (n up to 65536)
./bench_grid crossover # linear vs hybrid crossover sweep
./bench_grid cliff     # power-of-2 scaling test
./bench_grid threshold # binary search for 1-second boundary
./bench_grid profile   # FFT overhead + phase split + per-engine profiling
./bench_grid bench 8192 8192      # single (n,k) benchmark, 5 reps
./bench_grid bench 8192 100 10    # with custom rep count
OMP_NUM_THREADS=16 ./bench_grid   # parallel
```

## Architecture

Three engines with automatic dispatch:
- **Linear** (batched): O(nk), best for small k. Quad-point batched at BQ=8 on all
  platforms (AVX-512 native, Apple Silicon via 4 NEON FMA ports). Interleaved a_batch
  layout (`a_batch[j*BQ+qi]`) eliminates L1 cache misses from strided access. Hot
  loops in `src/linear_batched_impl.inc` (parameterized template). Fused backward
  pass with L2-cache-aware checkpointing for large working sets.
- **Hybrid** (B=auto): Block build + FFT tree + bidirectional divide. Best for
  large k. Block size B selected by cost model (`select_best_B`) using calibration
  data — typically B=16 on M3 Max, B=32 on Zen 4. Players sorted by stack size.
  Paired cached correlate shares both FFT(g) across siblings and FFT(P) from build.
  Per-level FFT decisions use calibrated data from fft_config.h (no global crossover).
  On Apple Silicon, FFT execution dispatches to vDSP (`vDSP_DFT_Interleaved`) at
  supported sizes (10-18% faster, zero format conversion overhead).
  k padded to fastest smooth size via `best_k_pad()`.
- **Tree** (pure): FFT-accelerated subproduct tree. Slightly slower than hybrid
  in serial, but wins in parallel (simpler clone, same PATIENT-quality plans).

Dispatch rule: `select_engine(n, k)` compares estimated linear cost (O(nk) with
BQ=8 + L2 checkpoint overhead) against hybrid cost (block build + FFT tree from
`select_best_B` model). Returns optimal B if hybrid wins, 0 for linear. Replaces
the old fixed K_CROSS thresholds — adapts to each (n, k) pair based on calibrated
FFT costs and hardware parameters.

Shared helpers: `tree_build_levels()` and `tree_propagate_g()` are used by both the
tree and hybrid engines to avoid code duplication.

Note: `bench_grid` always uses the batched linear engine (BQ=8) for the linear
path in benchmarks and verification, matching `icm_equity()` production behavior.

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
only within small blocks on the COMPLETE (non-truncated) block product,
where amplification is bounded by |c|^B (safe for FP64 at B ≤ 64).

## Device-specific constants

All tuning constants live in `devices/<DEVICE>/fft_config.h` as `#define`s:

| Constant | M3 Max | Zen 4 | What | How to measure |
|---|---|---|---|---|
| `FMA_NS` | 0.25 | 0.13 | ns per scalar FMA | `./bench_grid profile` (schoolbook row) |
| `FFT_OVERHEAD_NS` | 40.0 | 50.0 | Per-call FFT overhead | `./bench_grid profile` (overhead table) |
| `PAIRED_CACHED_CORR_RATIO` | 1.03 | 1.08 | Paired correlate / full pipeline | `./bench_grid profile` (phase split) |
| `INDEP_PAIR_RATIO` | 1.25 | 1.35 | correlate_fft_pair / full pipeline | `./bench_grid profile` (phase split) |
| `calib_sizes[]` / `calib_times_ns[]` | 749 entries | 749 entries | Per-size FFT costs | `tools/calibrate` |
| `calib_lib[]` | n/a | 749 entries | FFTW(0) vs MKL(1) per size | `tools/calibrate_dual` |

Auto-tuned at runtime (no manual tuning needed):
| Feature | What |
|---|---|
| `select_engine()` | Linear vs hybrid dispatch, cost-based comparison per (n, k) |
| `select_best_B()` | Hybrid block size, derived from calibration data via cost model |
| `ckpt_interval_batched()` | Checkpoint interval sized to fit L2 cache |
| BQ=8 (all platforms) | Batched linear width, interleaved layout for cache efficiency |
| vDSP dispatch | FFT backend: vDSP at supported sizes on Apple Silicon, FFTW elsewhere |
| MKL dispatch | FFT backend: MKL via dlopen at calibrated sizes on Linux, FFTW elsewhere |
| `calib_times_ns[]` | M3 Max: includes vDSP dispatch times. Zen 4: min(FFTW, MKL) |

## Directory Structure

```
src/
  icm.h                — public API header
  icm.c                — library implementation (all engines, FFT infrastructure)
  linear_batched_impl.inc — BQ-parameterized batched linear engine template
  icm_gpu.h            — GPU public API header
  icm_gpu.cu           — GPU implementation (legacy monolithic, kept for reference)
  gpu/                  — GPU implementation (split into modules)
    gpu_internal.h     — shared GPU internal types and helpers
    gpu_kernels.cu     — CUDA kernels (schoolbook, FFT, block build, propagate)
    gpu_plan.cu        — GPU planner (cost model, memory strategy, tier selection)
    gpu_exec.cu        — GPU execution engine (graph capture, q-pipeline)
    gpu_api.cu         — GPU public API implementation (plan/execute/destroy)
bench/
  bench.c              — benchmark harness, verification, tuning tools
  bench_gpu.cu         — GPU benchmark + verification harness
tools/
  calibrate.c          — FFTW calibration (generates fft_config.h + wisdom)
  calibrate_dual.c     — FFTW vs MKL dual calibration (generates calib_lib[])
  accuracy_bench.c     — quadrature accuracy convergence benchmark
  contour_1s.c         — 1-second contour sweep tool
  calibrate_gpu.cu     — GPU FFT calibration tool
  heatmap_gpu.cu       — GPU performance heatmap generator
  push_limit_gpu.cu    — GPU frontier (max-n) benchmark
  validate_planner_gpu.cu — GPU planner validation
  bench_vkfft.cu       — VkFFT benchmark (optional dual-dispatch)
  gen_gpu_calib_lib.py — generate GPU calibration library selection arrays
  plot_contour.py      — matplotlib contour/heatmap plotting
  plot_heatmap.py      — matplotlib GPU heatmap plotting
  gpu_phase_profile.cu — per-phase GPU timing breakdown (build/prop/leaf)
  gpu_sample_plans.cu  — sample planner decisions across an (n,k) grid
  sample_plans.c       — CPU analogue of gpu_sample_plans
  test_cpu_cost_model.c — validate CPU cost-model dispatch against measured times
  test_gpu_cost_model.cu — validate GPU cost-model dispatch against measured times
  run_b200_campaign.sh — full B200 benchmark campaign (calibrate + heatmap + frontier)
  setup_b200.sh        — one-time B200 instance setup (cuFFTDx via nvidia-mathdx)
  tier_ablation.cu     — direct schoolbook/fused/cuFFT timing comparison per level
  fit_cost_model.py    — fit CPU cost-model constants from calibration data
  fit_gpu_cost_model.py — fit GPU cost-model constants from calibration data
  run_vkfft_calib.sh   — VkFFT dual-dispatch calibration helper
devices/
  m3_max/              — Apple M3 Max calibration
    fft_config.h       — calibrated FFT times + cost model constants
    fftw_wisdom.dat    — FFTW PATIENT plans
  zen4/                — AMD Ryzen 9 7950X calibration
    fft_config.h       — calibrated FFT times + cost model constants
    fftw_wisdom.dat    — FFTW PATIENT plans
  b200/                — NVIDIA B200 GPU calibration
    gpu_fft_config.h   — GPU FFT calibration data
  h200/                — NVIDIA H200 GPU (placeholder)
python/                — Python ctypes bindings (pip install)
Makefile               — build system
RESULTS.md             — performance grid and optimization history
OPTIMIZATION_GUIDE.md  — detailed optimization notes + porting guide
```

## Correctness invariant

After ANY change, run `./bench_grid quick` and confirm ALL TESTS PASSED.

## M3 Max validation and benchmarking

Run these steps on an Apple M3 Max to validate dispatch and collect benchmark data.

### Step-by-step

```bash
# 1. Build (serial)
make clean && make

# 2. Verify correctness (must see ALL TESTS PASSED)
./bench_grid verify

# 3. Validate dispatch decisions — every cell should show the correct winner
./bench_grid crossover
# If any cell shows dispatch disagreeing with the measured winner,
# recalibrate (step 4). Otherwise skip to step 5.

# 4. Recalibrate (only if dispatch is wrong)
make calibrate && ./calibrate --quick
# Produces fft_config.h in the current directory.
# Copy it to devices/m3_max/fft_config.h, then rebuild:
#   cp fft_config.h devices/m3_max/fft_config.h
#   make clean && make
#   ./bench_grid verify          # re-verify after recalibration
#   ./bench_grid crossover       # confirm dispatch is now correct

# 5. Profile: measure FMA_NS, FFT_OVERHEAD_NS, phase split ratios
./bench_grid profile
# Update devices/m3_max/fft_config.h with measured values if they differ.

# 6. Serial performance grid
./bench_grid > bench_grid_m3max_serial.txt

# 7. Parallel performance grid (16 threads = 12P + 4E on M3 Max)
make clean && make parallel
OMP_NUM_THREADS=16 ./bench_grid > bench_grid_m3max_parallel.txt

# 8. Serial contour sweep (Q=256, 1-second boundary)
make contour_1s
./contour_1s --contour > contour_m3max_serial_q256.csv

# 9. Parallel contour sweep
make contour_1s_par
OMP_NUM_THREADS=16 ./contour_1s_par --contour > contour_m3max_parallel_q256.csv

# 10. Accuracy convergence (Q sweep for paper Table 3)
gcc -O3 -march=native -Isrc -Idevices/m3_max -I/opt/homebrew/include \
    -o accuracy_bench tools/accuracy_bench.c \
    -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
./accuracy_bench > accuracy_m3max.csv

# 11. Generate plots (requires matplotlib)
python3 tools/plot_contour.py
```

### Notes

- Contour sweeps stall at k >= 200,000 (each binary search probe takes
  minutes). The sweep will produce partial data through ~k=100K and timeout.
- bench_grid always uses the batched linear engine (BQ=8) for the linear path.

### Paper data to collect

After running the above, fill in:
- **Table 1** (single-threaded performance): M3 Max column from `bench_grid_m3max_serial.txt`
- **Table 2** (parallel performance): M3 Max column from `bench_grid_m3max_parallel.txt`
- **Table 3** (bandwidth/hardware constants): FMA_NS,
  FFT_OVERHEAD_NS from `./bench_grid profile` output; L2_CACHE_SIZE from
  hardware spec (32MB for M3 Max)
- **Contour figures**: from the CSV files generated in steps 8-9

## Porting to a new device

1. Run `tools/calibrate` on the target machine → produces `fft_config.h` + `fftw_wisdom.dat`
2. Copy to `devices/<DEVICE>/`
3. `make DEVICE=<DEVICE>` and `./bench_grid profile` to measure platform constants
4. Update `#define`s in `fft_config.h` (FMA_NS, FFT_OVERHEAD_NS, etc.)
5. On Apple Silicon: vDSP dispatch is automatic. Recalibrate `calib_times_ns[]`
   to reflect actual dispatch cost.
6. On Linux with MKL: run `tools/calibrate_dual` → adds `calib_lib[]` + updates
   `calib_times_ns[]` with min(FFTW, MKL). MKL dispatch via dlopen is automatic.
7. `./bench_grid verify` then `./bench_grid` for final numbers
8. Engine dispatch is cost-based (`select_engine`), no K_CROSS tuning needed.
   `./bench_grid crossover` can verify the dispatch decisions are correct.
