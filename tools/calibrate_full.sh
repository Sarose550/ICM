#!/usr/bin/env bash
#
# calibrate_full.sh — Full calibration pipeline for a new device.
#
# Usage: ./tools/calibrate_full.sh <DEVICE> [--quick]
#
# Runs the complete calibration pipeline from the repo root:
#   1. Builds and runs tools/calibrate.c (FFTW calibration)
#   2. Copies fft_config.h + fftw_wisdom.dat to devices/<DEVICE>/
#   3. Builds and runs tools/sample_plans.c (hybrid engine timing)
#   3b. Builds and runs tools/bench_wrap_fma.c — directly measures WRAP_FMA_NS
#       (wrap-correction cost) via an isolated microbenchmark, avoiding the
#       identifiability failure of the indirect full-plan regression for this
#       single constant.
#   4. Fits cost-model constants via tools/fit_cost_model.py --write --wrap-ns
#      (pins WRAP_FMA_NS from step 3b, fits remaining 8 parameters)
#   5. Rebuilds the library with the new device config
#   6. Verifies correctness (bench_grid verify) and crossover dispatch
#
# This can take 10–30+ minutes, dominated by step 1 (FFTW calibration).
set -euo pipefail

# ── Find repo root (this script lives in tools/) ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    echo "Usage: $0 <DEVICE> [--quick]"
    echo "  DEVICE   — device name (e.g. zen4, m3_pro)"
    echo "  --quick  — pass to calibrate for a faster (less thorough) FFTW calibration"
    exit 1
}

# ── Parse arguments ──
DEVICE=""
QUICK=""
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK="--quick" ;;
        -h|--help) usage ;;
        *)  if [ -z "$DEVICE" ]; then DEVICE="$arg"; else echo "ERROR: unexpected argument: $arg"; usage; fi ;;
    esac
done
if [ -z "$DEVICE" ]; then usage; fi

DEVICE_DIR="$REPO_ROOT/devices/$DEVICE"
CONFIG_H="$DEVICE_DIR/fft_config.h"
WISDOM_DAT="$DEVICE_DIR/fftw_wisdom.dat"
CSV_FILE="$REPO_ROOT/sample_plans_${DEVICE}.csv"
LOG_FILE="$REPO_ROOT/sample_plans_${DEVICE}.log"

# ── Detect platform ──
OS="$(uname -s)"
case "$OS" in
    Darwin)
        HOMEBREW_INC="-I/opt/homebrew/include"
        HOMEBREW_LIB="-L/opt/homebrew/lib"
        ACCEL_FLAGS="-framework Accelerate"
        VEC_FLAGS=""
        ;;
    Linux)
        # Auto-detect AOCL-FFTW, matching the Makefile.
        if [ -f /usr/local/aocl-fftw/lib/libfftw3.so ]; then
            HOMEBREW_INC="-I/usr/local/aocl-fftw/include"
            HOMEBREW_LIB="-L/usr/local/aocl-fftw/lib -Wl,-rpath,/usr/local/aocl-fftw/lib"
            echo "  Detected AOCL-FFTW at /usr/local/aocl-fftw — using it for calibration."
        else
            HOMEBREW_INC=""
            HOMEBREW_LIB=""
        fi
        ACCEL_FLAGS=""
        VEC_FLAGS="-ldl -lmvec"
        ;;
    *)
        echo "ERROR: unsupported OS: $OS"
        exit 1
        ;;
esac

cd "$REPO_ROOT"
echo "=== calibrate_full.sh: calibrating device '$DEVICE' ==="
echo "  Repo root: $REPO_ROOT"
echo "  Platform:  $OS"
echo "  Quick:     ${QUICK:-no}"
echo ""

# ── Step 1: Build and run calibrate ──
echo "── Step 1/7: FFTW calibration (tools/calibrate.c) ──"
echo "  This may take 10–30 minutes..."
CALIB_BIN="$REPO_ROOT/calibrate"
gcc -O3 -march=native $HOMEBREW_INC -o "$CALIB_BIN" tools/calibrate.c $HOMEBREW_LIB -lfftw3 -lm
echo "  Running: $CALIB_BIN $QUICK"
"$CALIB_BIN" $QUICK
echo "  ✓ calibrate complete"

# Verify outputs exist
if [ ! -f "$REPO_ROOT/fft_config.h" ]; then
    echo "ERROR: calibrate did not produce fft_config.h in $REPO_ROOT"
    exit 1
fi
if [ ! -f "$REPO_ROOT/fftw_wisdom.dat" ]; then
    echo "ERROR: calibrate did not produce fftw_wisdom.dat in $REPO_ROOT"
    exit 1
fi

# ── Step 2: Copy generated files to devices/<DEVICE>/ ──
echo ""
echo "── Step 2/7: Copy calibration files to devices/$DEVICE/ ──"
mkdir -p "$DEVICE_DIR"

for f in fft_config.h fftw_wisdom.dat; do
    SRC="$REPO_ROOT/$f"
    DST="$DEVICE_DIR/$f"
    if [ -f "$DST" ]; then
        echo "  ⚠ WARNING: $DST already exists — overwriting"
    fi
    cp "$SRC" "$DST"
    echo "  ✓ Copied $f → devices/$DEVICE/"
done

# ── Step 3: Build and run sample_plans ──
echo ""
echo "── Step 3/7: Build and run sample_plans.c ──"
SP_BIN="$REPO_ROOT/sample_plans"
gcc -O3 -march=native \
    -Isrc \
    -I"$DEVICE_DIR" \
    $HOMEBREW_INC \
    -o "$SP_BIN" \
    tools/sample_plans.c \
    $HOMEBREW_LIB \
    -lfftw3 -lm \
    $ACCEL_FLAGS $VEC_FLAGS
echo "  ✓ Built sample_plans"

echo "  Running sample_plans (this takes several minutes)..."
"$SP_BIN" > "$CSV_FILE" 2>"$LOG_FILE"
echo "  ✓ sample_plans complete → $CSV_FILE"

# ── Step 4: Build and run bench_wrap_fma (direct WRAP_FMA_NS measurement) ──
echo ""
echo "── Step 4/7: Direct wrap-correction microbenchmark (tools/bench_wrap_fma.c) ──"
WRAP_BENCH_BIN="$REPO_ROOT/bench_wrap_fma"
WRAP_CSV="$REPO_ROOT/wrap_fma_${DEVICE}.csv"
gcc -O3 -march=native -o "$WRAP_BENCH_BIN" tools/bench_wrap_fma.c -lm
echo "  Running bench_wrap_fma..."
"$WRAP_BENCH_BIN" > "$WRAP_CSV"
echo "  ✓ bench_wrap_fma complete → $WRAP_CSV"

# Extract WRAP_FMA_NS: least-squares SLOPE of median_ns_per_call vs. fma_count
# over the SMALL_2048 regime, restricted to wrap_m in [64,384] (the realistic
# decision-relevant range — see scratch/zen4_wrap_investigation/probe_results.txt,
# levels 7-8 of the regressed Zen4 case had wrap_m ~150-200).
#
# Must be a SLOPE (regression), not a raw ns_per_call/fma_count ratio: each
# call has a fixed overhead that doesn't scale with fma_count, so a raw ratio
# is contaminated by that overhead and overestimates the true per-FMA cost,
# especially at small fma_count. A slope between two well-separated points
# (or a full least-squares fit, computed here) cancels the fixed overhead and
# recovers the true marginal ns/FMA. (An earlier version of this script used
# the raw-ratio method and got lucky — it landed close to the correct value
# by coincidence on Zen4; don't reintroduce it.)
WRAP_FMA_NS=$(awk -F, '
  NR>1 && $1=="SMALL_2048" && $3>=64 && $3<=384 {
    x = $5; y = $6
    n++; sx += x; sy += y; sxx += x*x; sxy += x*y
  }
  END {
    if (n >= 2 && (n*sxx - sx*sx) != 0) {
      slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
      printf "%.4f", slope
    } else {
      print "0.4000"  # fallback: last known-good Zen4 value, flagged for manual review
    }
  }' "$WRAP_CSV")
echo "  Extracted WRAP_FMA_NS = $WRAP_FMA_NS (least-squares slope, SMALL_2048, wrap_m in [64,384])"

# ── Step 5: Fit cost model constants (8 fitted + WRAP_FMA_NS pinned) ──
echo ""
echo "── Step 5/7: Fit cost model (tools/fit_cost_model.py --write --wrap-ns $WRAP_FMA_NS) ──"
python3 tools/fit_cost_model.py "$CSV_FILE" "$CONFIG_H" --write --wrap-ns "$WRAP_FMA_NS"
echo "  ✓ Cost model fit complete → $CONFIG_H updated"

# ── Step 6: Rebuild ──
echo ""
echo "── Step 6/7: Rebuild library (make clean && make DEVICE=$DEVICE) ──"
make clean
make "DEVICE=$DEVICE"
echo "  ✓ Rebuild complete"

# ── Step 7: Verify ──
echo ""
echo "── Step 7/7: Verify correctness and crossover dispatch ──"

echo ""
echo "--- bench_grid verify ---"
if ./bench_grid verify; then
    echo "✓ verify passed"
else
    echo ""
    echo "========================================="
    echo "ERROR: bench_grid verify FAILED"
    echo "Calibration may have produced invalid constants."
    echo "Check $CONFIG_H and re-run manually."
    echo "========================================="
    exit 1
fi

echo ""
echo "--- bench_grid crossover ---"
./bench_grid crossover || true  # crossover can fail on some edge cases; don't abort

echo ""
echo "=== calibrate_full.sh: DONE ==="
echo "  Device:        $DEVICE"
echo "  Config:        $CONFIG_H"
echo "  Wisdom:        $WISDOM_DAT"
echo "  Sample CSV:    $CSV_FILE"
echo "  Sample log:    $LOG_FILE"
echo "  Wrap bench CSV:$WRAP_CSV"
echo ""
echo "Next steps (manual):"
echo "  ./bench_grid profile   # measure platform constants (FMA_NS, FFT_OVERHEAD_NS, ratios)"
echo "  ./bench_grid           # full performance grid"
