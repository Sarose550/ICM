# ICM Calculator — Hardware-Optimized via Generating Function Quadrature

Computes the full n × n placement probability matrix for n-player poker
tournaments using the generating function integral approach with erfc_trap
quadrature.

## Files

```
icm.h              Shared types, inline helpers, function declarations
icm_common.c       Quadrature nodes, validation, test distributions
icm_avx2.c         AVX2 backend (4-player SIMD, works on any x86-64 from ~2013)
icm_avx512.c       AVX-512 backend (8-player SIMD, Zen 4 / Sapphire Rapids)
icm_cuda.cu        GPU backend (A100 / H100)
bench.c            CPU benchmark harness (5 benchmark suites, CSV output)
bench_cuda.cu      GPU benchmark harness (5 benchmark suites, CSV output)
plot_results.py    Generate plots from CSV data (overlays CPU + GPU)
Makefile           Build everything
```

## Quick Start

```bash
# AVX2 only (works anywhere)
make
taskset -c 0 ./bench_avx2 --quick

# AVX2 + AVX-512 (Zen 4 / SPR)
make avx512
taskset -c 0 ./bench_avx512 --quick

# GPU (A100)
make cuda CUDA_ARCH=sm_80
./bench_cuda --quick

# Generate plots
pip install matplotlib pandas numpy
python3 plot_results.py
```

## Build

```bash
make                  # AVX2 only → bench_avx2
make avx512           # + AVX-512 → bench_avx512
make cuda             # GPU → bench_cuda
make all              # Try everything, skip what fails
```

Override compiler or arch:
```bash
make CC=gcc-13
make avx512 CC=gcc-13 AVX512_FLAGS="-mavx512f -mavx512dq"
make cuda CUDA_ARCH=sm_90   # H100
```

## Run

```bash
# CPU (all options)
./bench_avx2                      # full suite, ~5 min
./bench_avx2 --quick              # reduced sweep, ~1 min
./bench_avx2 --reps 5             # more repetitions
./bench_avx2 --max-n 4096         # extend n sweep

# GPU
./bench_cuda                      # full suite
./bench_cuda --quick              # reduced sweep
```

Pin to a core for stable results:
```bash
taskset -c 0 ./bench_avx512
```

## Output

Both CPU and GPU benchmarks produce matching CSV files:

| Benchmark | CPU file | GPU file |
|-----------|----------|----------|
| Accuracy vs Q | `cpu_accuracy_vs_q.csv` | `accuracy_vs_q.csv` |
| Time vs Q | `cpu_time_vs_q.csv` | `time_vs_q.csv` |
| Time vs n | `cpu_time_vs_n.csv` | `time_vs_n.csv` |
| Max n under budget | `cpu_max_n_under_budget.csv` | `max_n_under_budget.csv` |
| Full scaling | `cpu_scaling.csv` | `scaling.csv` |

Copy CSVs from both machines into the same directory, then:
```bash
python3 plot_results.py
```

The plotting script auto-detects which files are present and overlays
CPU + GPU curves when both exist.

## Architecture

### AVX2 backend (`icm_avx2.c`)

Phase 1: Build Q=256 polynomials, store in ~4.2 MB (L3-resident).
Phase 2: Process 4 sorted players per block. 64 KB interleaved
accumulator stays in L1/L2. Fused SIMD divide+accumulate with
bidirectional stability. Stack sorting ensures >99.9% of iterations
hit the fast all-bottom-up or all-top-down SIMD path.

### AVX-512 backend (`icm_avx512.c`)

Same algorithm, 8 players per ZMM register. 128 KB accumulator fits
in Zen 4's 1 MB L2. Vectorized `exp()` via degree-11 minimax polynomial
replaces 16 scalar `exp()` calls with 2 `fast_exp_8` calls per block
per quad point.

### GPU backend (`icm_cuda.cu`)

Phase 1: Q blocks build polynomials in parallel using double-buffered
shared memory. Phase 2: One block per player, Q threads per block. Each
thread runs its own division recurrence. Tiled shared-memory reduction
(TILE_M=16 coefficients per tile) accumulates across Q threads without
any temp buffer.

## GPU Instance Selection

**CRITICAL: All computation is FP64.** Consumer GPUs have 1:32 FP64
throughput and will be slower than CPU. Use data-center GPUs only.

| Instance | GPU | FP64 TFLOPS | Cost |
|----------|-----|-------------|------|
| p4d.24xlarge | 8× A100 80GB | 8 × 19.5 | ~$32/hr |
| p5.48xlarge | 8× H100 | 8 × 67 | ~$98/hr |
| g5.xlarge | A10G (DO NOT USE) | 0.6 | waste of money |

## Projected Performance (n=2048, Q=256, uniform)

| Platform | Time | vs scalar |
|----------|------|-----------|
| Scalar baseline (2.8 GHz) | 2093 ms | 1.0× |
| AVX2 v5h (2.8 GHz) | 747 ms | 2.8× |
| AVX-512 (2.8 GHz Skylake) | 563 ms | 3.7× |
| AVX-512 (5.0 GHz Zen 4, projected) | ~250 ms | ~8× |
| 7950X (5.7 GHz, projected) | ~220 ms | ~10× |
| A100 GPU (projected) | ~2–5 ms | ~500× |
| GTO Wizard (claimed) | 740 ms | — |
