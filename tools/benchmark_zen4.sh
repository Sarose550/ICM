#!/bin/bash
# benchmark_zen4.sh — Full benchmark + tuning workflow for Ryzen 7950X
#
# Prerequisites:
#   - Run calibrate_zen4.sh first (generates fft_config.h + wisdom)
#   - Update FMA_NS, FFT_OVERHEAD_NS, etc. in devices/zen4/fft_config.h
#
# Usage (from repo root):
#   ./tools/benchmark_zen4.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== ICM Zen 4 Benchmark Suite ==="
echo "Device config: devices/zen4/fft_config.h"
echo ""

echo "--- Step 1: Rebuild with updated constants ---"
make clean
make DEVICE=zen4
echo ""

echo "--- Step 2: Verify correctness ---"
./bench_grid verify 2>&1 | tee zen4_verify.txt
echo ""
if grep -q "FAILED" zen4_verify.txt; then
    echo "ERROR: Verification failed! Fix before continuing."
    exit 1
fi
echo "Verification passed."
echo ""

echo "--- Step 3: Crossover sweep (linear vs hybrid) ---"
./bench_grid crossover 2>&1 | tee zen4_crossover.txt
echo ""
echo "  Review zen4_crossover.txt to determine k_cross values."
echo "  Look for the k where hybrid first beats linear at each n."
echo ""

echo "--- Step 4: Full serial benchmark grid ---"
nice -20 ./bench_grid 2>&1 | tee zen4_serial.txt
echo ""

echo "--- Step 5: Parallel benchmark (16 threads) ---"
make clean
make DEVICE=zen4 parallel
OMP_NUM_THREADS=16 nice -20 ./bench_grid 2>&1 | tee zen4_parallel_16t.txt
echo ""

echo "--- Step 6: Power-of-2 cliff test ---"
./bench_grid cliff 2>&1 | tee zen4_cliff.txt
echo ""

echo "--- Step 7: 1-second threshold ---"
./bench_grid threshold 2>&1 | tee zen4_threshold.txt
echo ""

echo "--- Step 8: Profile (serial) ---"
make clean
make DEVICE=zen4
./bench_grid profile 2>&1 | tee zen4_profile.txt
echo ""

echo "=== All Results Saved ==="
echo "  zen4_verify.txt        — correctness"
echo "  zen4_crossover.txt     — linear/hybrid crossover"
echo "  zen4_serial.txt        — serial performance grid"
echo "  zen4_parallel_16t.txt  — 16-thread parallel grid"
echo "  zen4_cliff.txt         — power-of-2 scaling"
echo "  zen4_threshold.txt     — 1-second boundary"
echo "  zen4_profile.txt       — FFT phase profiling"
echo ""
echo "Next: review crossover data and update k_cross in icm.c if needed."
echo "Then re-run: make DEVICE=zen4 && ./bench_grid verify && ./bench_grid"
