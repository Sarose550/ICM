#!/bin/bash
# calibrate_zen4.sh — Full FFT calibration for Ryzen 7950X (Zen 4)
#
# Generates fft_config.h + fftw_wisdom.dat, then copies to devices/zen4/.
# Takes 10-30 minutes pinned to core 0 on a quiet machine.
#
# Usage (from repo root):
#   sudo nice -20 taskset -c 0 ./tools/calibrate_zen4.sh
#   (or without taskset if not available: sudo nice -20 ./tools/calibrate_zen4.sh)

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== ICM Zen 4 Calibration ==="
echo "Repo root: $(pwd)"
echo ""

echo "Step 1: Build calibration tool with -march=znver4..."
gcc -O3 -march=znver4 -o calibrate tools/calibrate.c -lfftw3 -lm
echo "  Built: ./calibrate"
echo ""

echo "Step 2: Run calibration (FFTW PATIENT wisdom + benchmark all 749 smooth sizes)..."
echo "  This may take 10-30 minutes. Be patient."
echo ""
./calibrate
echo ""

echo "Step 3: Copy outputs to devices/zen4/..."
mkdir -p devices/zen4
cp fft_config.h devices/zen4/fft_config.h
cp fftw_wisdom.dat devices/zen4/fftw_wisdom.dat
echo "  Copied fft_config.h -> devices/zen4/fft_config.h"
echo "  Copied fftw_wisdom.dat -> devices/zen4/fftw_wisdom.dat"
echo ""

echo "Step 4: Initial build with calibrated data..."
make clean
make DEVICE=zen4
echo ""

echo "Step 5: Run profile to measure platform constants..."
echo "  (FFT overhead, phase split, schoolbook FMA rate)"
echo ""
./bench_grid profile 2>&1 | tee zen4_profile_output.txt
echo ""
echo "  Profile output saved to zen4_profile_output.txt"

echo ""
echo "=== Calibration Complete ==="
echo ""
echo "NOW: Read zen4_profile_output.txt and update devices/zen4/fft_config.h:"
echo "  1. FMA_NS:                    from schoolbook row (expect ~0.06-0.08)"
echo "  2. FFT_OVERHEAD_NS:           from overhead column"
echo "  3. PAIRED_CACHED_CORR_RATIO:  from phase split (f_fwd + 2*(f_pw + f_ifft))"
echo "  4. INDEP_PAIR_RATIO:          from phase split (3*fwd + 2*pw + 2*ifft)/calib"
echo ""
echo "Then run: ./tools/benchmark_zen4.sh"
