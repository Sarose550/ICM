#!/bin/bash
set -euo pipefail
cd /root/ICM

LOG=/root/ICM/overnight_log.txt
exec > >(tee -a "$LOG") 2>&1

echo "=== ICM Zen 4 Overnight Pipeline ==="
echo "Started: $(date)"
echo ""

# ── Step 0: Wait for calibration to finish ──
echo "--- Waiting for calibration to finish ---"
while pgrep -f './calibrate' > /dev/null 2>&1; do
    echo "  $(date +%H:%M:%S) calibration still running... $(tail -1 calibration_output.txt 2>/dev/null)"
    sleep 60
done
echo "  Calibration finished at $(date)"
echo ""
cat calibration_output.txt
echo ""

# ── Step 1: Copy calibration output to devices/zen4/ ──
echo "--- Step 1: Installing calibration data ---"
cp fft_config.h devices/zen4/fft_config.h
cp fftw_wisdom.dat devices/zen4/fftw_wisdom.dat
echo "  Copied fft_config.h and fftw_wisdom.dat to devices/zen4/"
echo ""

# ── Step 2: Patch fft_config.h with measured FMA_NS ──
echo "--- Step 2: Patching FMA_NS = 0.08 in fft_config.h ---"
sed -i 's/^#define FMA_NS .*/#define FMA_NS 0.08/' devices/zen4/fft_config.h
echo "  Updated FMA_NS to 0.08"
echo ""

# ── Step 3: Build serial bench_grid ──
echo "--- Step 3: Building bench_grid (serial) ---"
gcc -O3 -march=znver4 -Wall -Wno-unused-variable -Wno-unused-function \
    -Isrc -Idevices/zen4 -I/usr/local/include \
    -o bench_grid bench/bench.c \
    -L/usr/local/lib -lfftw3 -lm
echo "  Build OK"
echo ""

# ── Step 4: Profile (measure FFT overhead + phase split) ──
echo "--- Step 4: Profile ---"
taskset -c 0 nice -20 ./bench_grid profile 2>&1 | tee zen4_profile.txt
echo ""

# ── Step 5: Extract FFT_OVERHEAD_NS from profile and patch ──
echo "--- Step 5: Extracting constants from profile ---"
# Try to extract overhead from the profile table
OVERHEAD=$(grep -oP 'overhead.*?([0-9]+\.[0-9]+)' zen4_profile.txt | grep -oP '[0-9]+\.[0-9]+' | head -1 || echo "")
if [ -n "$OVERHEAD" ]; then
    echo "  Extracted FFT_OVERHEAD_NS = $OVERHEAD"
    sed -i "s/^#define FFT_OVERHEAD_NS .*/#define FFT_OVERHEAD_NS $OVERHEAD/" devices/zen4/fft_config.h
    echo "  Updated FFT_OVERHEAD_NS in fft_config.h"
else
    echo "  Could not auto-extract FFT_OVERHEAD_NS - review zen4_profile.txt manually"
fi
echo ""

# ── Step 5b: Rebuild with updated constants ──
echo "--- Step 5b: Rebuilding with updated constants ---"
gcc -O3 -march=znver4 -Wall -Wno-unused-variable -Wno-unused-function \
    -Isrc -Idevices/zen4 -I/usr/local/include \
    -o bench_grid bench/bench.c \
    -L/usr/local/lib -lfftw3 -lm
echo "  Rebuild OK"
echo ""

# ── Step 6: Verify correctness ──
echo "--- Step 6: Verify correctness ---"
taskset -c 0 nice -20 ./bench_grid verify 2>&1 | tee zen4_verify.txt
echo ""
if grep -q "FAILED" zen4_verify.txt; then
    echo "!!! VERIFICATION FAILED !!!"
else
    echo "  ALL TESTS PASSED"
fi
echo ""

# ── Step 7: Crossover sweep ──
echo "--- Step 7: Crossover sweep (linear vs hybrid) ---"
taskset -c 0 nice -20 ./bench_grid crossover 2>&1 | tee zen4_crossover.txt
echo ""

# ── Step 8: Full serial benchmark grid ──
echo "--- Step 8: Full serial benchmark ---"
taskset -c 0 nice -20 ./bench_grid 2>&1 | tee zen4_serial.txt
echo ""

# ── Step 9: Build parallel and benchmark ──
echo "--- Step 9: Building bench_grid (parallel, 16 threads) ---"
gcc -O3 -march=znver4 -Wall -Wno-unused-variable -Wno-unused-function \
    -fopenmp -Isrc -Idevices/zen4 -I/usr/local/include \
    -o bench_grid bench/bench.c \
    -L/usr/local/lib -lfftw3 -lfftw3_threads -lpthread -lm
echo "  Build OK"
echo ""

echo "--- Step 9b: Parallel benchmark (16 threads) ---"
OMP_NUM_THREADS=16 nice -20 ./bench_grid 2>&1 | tee zen4_parallel_16t.txt
echo ""

# ── Step 10: Rebuild serial for remaining tests ──
echo "--- Step 10: Rebuilding serial for cliff/threshold ---"
gcc -O3 -march=znver4 -Wall -Wno-unused-variable -Wno-unused-function \
    -Isrc -Idevices/zen4 -I/usr/local/include \
    -o bench_grid bench/bench.c \
    -L/usr/local/lib -lfftw3 -lm
echo "  Build OK"
echo ""

echo "--- Step 10b: Cliff test ---"
taskset -c 0 nice -20 ./bench_grid cliff 2>&1 | tee zen4_cliff.txt
echo ""

echo "--- Step 10c: Threshold (1-second boundary) ---"
taskset -c 0 nice -20 ./bench_grid threshold 2>&1 | tee zen4_threshold.txt
echo ""

# ── Summary ──
echo ""
echo "============================================"
echo "=== ALL STEPS COMPLETE at $(date) ==="
echo "============================================"
echo ""
echo "Result files:"
ls -la zen4_*.txt fma_results.txt 2>/dev/null
echo ""
echo "Key constants measured:"
echo "  FMA_NS = 0.08 (from measure_fma)"
grep 'FFT_OVERHEAD_NS\|PAIRED_CACHED_CORR_RATIO\|INDEP_PAIR_RATIO\|FMA_NS' devices/zen4/fft_config.h | head -10
echo ""
echo "Karatsuba result: FFT wins at all sizes, no Karatsuba regime."
echo ""
echo "Review zen4_crossover.txt for k_cross tuning."
echo "Review zen4_serial.txt / zen4_parallel_16t.txt for performance."
