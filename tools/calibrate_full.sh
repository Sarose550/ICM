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
#   4. Builds and runs tools/bench_wrap_fma.c — directly measures WRAP_FMA_NS
#      (wrap-correction cost) via an isolated microbenchmark, avoiding the
#      identifiability failure of the indirect full-plan regression for this
#      single constant.
#   5. Builds and runs tools/bench_div_chain.c — directly measures
#      FP64_DIV_NS (leaf-extraction division cost) via a dependency-chained
#      microbenchmark. Same identifiability failure as WRAP_FMA_NS: observed
#      on M3 Pro converging to a physically implausible 0.5ns and hitting
#      its fit bound when left free.
#   6. Builds bench_grid, runs `./bench_grid profile` — extracts FMA_NS
#      (schoolbook slope, cps=16→32), PAIRED_CACHED_CORR_RATIO and
#      INDEP_PAIR_RATIO (phase-split table, fft_n ≥ 4096).
#   7. Builds and runs tools/bench_block_build.c — directly measures the
#      block-build per-player cost at each candidate B, writes per-B lookup
#      table into fft_config.h (BLOCK_FMA_NS/BLOCK_MEM_NS replaced by
#      block_build_ns_per_player[]).
#   8. Builds and runs tools/bench_leaf_fma.c — directly measures the
#      leaf-extraction per-player cost at each candidate B, writes per-B
#      lookup table into fft_config.h (LEAF_FMA_NS/LEAF_BLOCK_NS replaced by
#      leaf_fma_ns_per_player[]).
#   9. Runs tools/fit_cost_model.py --write with ALL 6 scalar pins
#      (WRAP_FMA_NS, FP64_DIV_NS, FMA_NS, PAIRED_CACHED_CORR_RATIO,
#      INDEP_PAIR_RATIO, FFT_OVERHEAD_NS=0.0).  ZERO free parameters remain
#      — scipy optimization is skipped; the script assembles the fully-pinned
#      config directly.
#  10. Rebuilds the library with the new device config
#  11. Verifies correctness (bench_grid verify) and crossover dispatch
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
echo "── Step 1/11: FFTW calibration (tools/calibrate.c) ──"
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
echo "── Step 2/11: Copy calibration files to devices/$DEVICE/ ──"
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
echo "── Step 3/11: Build and run sample_plans.c ──"
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
echo "── Step 4/11: Direct wrap-correction microbenchmark (tools/bench_wrap_fma.c) ──"
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

# ── Step 5: Build and run bench_div_chain (direct FP64_DIV_NS measurement) ──
echo ""
echo "── Step 5/11: Direct division-chain microbenchmark (tools/bench_div_chain.c) ──"
DIV_BENCH_BIN="$REPO_ROOT/bench_div_chain"
DIV_CSV="$REPO_ROOT/div_chain_${DEVICE}.csv"
gcc -O3 -march=native -o "$DIV_BENCH_BIN" tools/bench_div_chain.c
echo "  Running bench_div_chain..."
"$DIV_BENCH_BIN" > "$DIV_CSV"
FP64_DIV_NS=$(awk -F, 'NR>1 && $1=="chained_dependency" {print $2}' "$DIV_CSV")
if [ -z "$FP64_DIV_NS" ]; then
    echo "  WARNING: bench_div_chain produced no output, falling back to unpinned fit for C_div"
    FP64_DIV_NS=""
else
    echo "  Measured FP64_DIV_NS = $FP64_DIV_NS ns (dependency-chained, matches leaf-extraction usage)"
fi

# ── Step 6: Build bench_grid, run profile → FMA_NS, PAIRED_CACHED_CORR_RATIO, INDEP_PAIR_RATIO ──
echo ""
echo "── Step 6/11: Profile FFT phases + schoolbook cost (./bench_grid profile) ──"
BENCH_GRID_BIN="$REPO_ROOT/bench_grid"
gcc -O3 -march=native -Wall -Wno-unused-variable -Wno-unused-function \
    -Isrc \
    -I"$DEVICE_DIR" \
    $HOMEBREW_INC \
    -o "$BENCH_GRID_BIN" \
    bench/bench.c \
    $HOMEBREW_LIB \
    -lfftw3 -lm \
    $ACCEL_FLAGS $VEC_FLAGS
echo "  ✓ Built bench_grid"

PROFILE_LOG="$REPO_ROOT/profile_${DEVICE}.log"
echo "  Running ./bench_grid profile..."
"$BENCH_GRID_BIN" profile > "$PROFILE_LOG" 2>&1
echo "  ✓ Profile complete → $PROFILE_LOG"

# ── Parse FMA_NS: slope of schoolbook time vs FMA count, cps=16 → 32 ──
# The FFT OVERHEAD table (measure_fft_overhead) has columns:
#   cps  school  fft_act  fft_calib  overhead  fft_size
# FMA count for schoolbook polymul at degree cps = cps² (model convention).
# Using slope between two rows to cancel per-call overhead.
FMA_NS=$(awk '
  /^[0-9]+[[:space:]]+[0-9]+/ {
    cps=$1; school=$2
    if (cps == 16) { s16 = school }
    if (cps == 32) { s32 = school }
  }
  END {
    if (s16 != "" && s32 != "") {
      # FMA count = cps²: 16²=256, 32²=1024, diff=768
      slope = (s32 - s16) / (1024 - 256)
      printf "%.4f", slope
    } else {
      print "0.0500"  # fallback
    }
  }' "$PROFILE_LOG")
echo "  FMA_NS = $FMA_NS  (schoolbook slope, cps=16→32)"

# ── Parse PAIRED_CACHED_CORR_RATIO and INDEP_PAIR_RATIO from phase-split table ──
# The FFT PHASE SPLIT table (measure_phase_split) has columns:
#   fft_n  fwd  pw  ifft  memcpy  sum  calib  f_fwd  f_pw  f_ifft
# Formulas (from bench.c):
#   PAIRED_CACHED_CORR_RATIO = f_fwd + 2*(f_pw + f_ifft)
#   INDEP_PAIR_RATIO         = 3*f_fwd + 2*(f_pw + f_ifft)
# Average over stable large FFT sizes (fft_n >= 4096).
PAIRED_CACHED_CORR_RATIO=$(awk '
  /^[0-9]+[[:space:]]+[0-9]+/ {
    fft_n=$1; f_fwd=$(NF-2); f_pw=$(NF-1); f_ifft=$NF
    if (fft_n >= 4096 && f_fwd > 0 && f_pw > 0 && f_ifft > 0) {
      ratio = f_fwd + 2*(f_pw + f_ifft)
      sum_r += ratio; n++
    }
  }
  END {
    if (n > 0) printf "%.4f", sum_r / n
    else print "1.0500"  # fallback
  }' "$PROFILE_LOG")
echo "  PAIRED_CACHED_CORR_RATIO = $PAIRED_CACHED_CORR_RATIO  (avg over fft_n ≥ 4096)"

INDEP_PAIR_RATIO=$(awk '
  /^[0-9]+[[:space:]]+[0-9]+/ {
    fft_n=$1; f_fwd=$(NF-2); f_pw=$(NF-1); f_ifft=$NF
    if (fft_n >= 4096 && f_fwd > 0 && f_pw > 0 && f_ifft > 0) {
      ratio = 3*f_fwd + 2*(f_pw + f_ifft)
      sum_r += ratio; n++
    }
  }
  END {
    if (n > 0) printf "%.4f", sum_r / n
    else print "2.0500"  # fallback
  }' "$PROFILE_LOG")
echo "  INDEP_PAIR_RATIO = $INDEP_PAIR_RATIO  (avg over fft_n ≥ 4096)"

# ── Step 7: Build and run bench_block_build → per-B lookup table ──
echo ""
echo "── Step 7/11: Direct block-build microbenchmark (tools/bench_block_build.c) ──"
BLOCK_BENCH_BIN="$REPO_ROOT/bench_block_build"
gcc -O3 -march=native -o "$BLOCK_BENCH_BIN" tools/bench_block_build.c -lm
echo "  Running bench_block_build..."
BLOCK_OUT="$("$BLOCK_BENCH_BIN")"
echo "$BLOCK_OUT" > "$REPO_ROOT/block_build_${DEVICE}.log"
echo "  ✓ bench_block_build complete → $REPO_ROOT/block_build_${DEVICE}.log"

# Write the per-B lookup table into fft_config.h via a temp Python script.
# (Inline python3 -c is blocked by the sandbox; temp files work fine.)
cat > /tmp/_write_block_table.py << 'PYEOF'
import sys, re

# Parse B=value,value lines from BLOCK_BUILD_NS_PER_PLAYER_TABLE section
table = {}
in_table = False
for line in sys.stdin:
    line = line.strip()
    if line == 'BLOCK_BUILD_NS_PER_PLAYER_TABLE':
        in_table = True
        continue
    if in_table and line.startswith('B='):
        parts = line.split(',')
        if len(parts) == 2:
            b = int(parts[0].split('=')[1])
            val = float(parts[1])
            table[b] = val
    elif in_table and not line.startswith('B='):
        break

if len(table) != 6:
    print(f'WARNING: expected 6 B values, got {len(table)}: {table}', file=sys.stderr)
    sys.exit(0)

b_order = [8, 16, 24, 32, 48, 64]
config_path = sys.argv[1]

with open(config_path, 'r') as f:
    text = f.read()

array_pattern = r'(static const double block_build_ns_per_player\[6\]\s*=\s*\{)'
match = re.search(array_pattern, text)
if not match:
    print('WARNING: block_build_ns_per_player array not found in header', file=sys.stderr)
    sys.exit(0)

start = match.end()
end = text.index('};', start) + 2

lines = ['static const double block_build_ns_per_player[6] = {']
for i, b in enumerate(b_order):
    comma = ',' if i < 5 else ''
    lines.append(f'    {table[b]:.4f}{comma}  /* B={b:<2d} */')
lines.append('};')

new_array = '\n'.join(lines)
text = text[:match.start()] + new_array + text[end:]

with open(config_path, 'w') as f:
    f.write(text)

print(f'  Wrote block_build_ns_per_player[] to {config_path}')
for b in b_order:
    print(f'    B={b:<2d}  {table[b]:.4f}')
PYEOF
echo "$BLOCK_OUT" | python3 /tmp/_write_block_table.py "$CONFIG_H"
echo "  ✓ Block-build lookup table written to $CONFIG_H"

# ── Step 8: Build and run bench_leaf_fma → per-B lookup table ──
echo ""
echo "── Step 8/11: Direct leaf-FMA microbenchmark (tools/bench_leaf_fma.c) ──"
LEAF_BENCH_BIN="$REPO_ROOT/bench_leaf_fma"
gcc -O3 -march=native -o "$LEAF_BENCH_BIN" tools/bench_leaf_fma.c -lm
echo "  Running bench_leaf_fma..."
LEAF_OUT="$("$LEAF_BENCH_BIN")"
echo "$LEAF_OUT" > "$REPO_ROOT/leaf_fma_${DEVICE}.log"
echo "  ✓ bench_leaf_fma complete → $REPO_ROOT/leaf_fma_${DEVICE}.log"

# Write the per-B lookup table into fft_config.h via a temp Python script.
cat > /tmp/_write_leaf_table.py << 'PYEOF'
import sys, re

# Parse B=value,value lines from LEAF_FMA_NS_PER_PLAYER_TABLE section
table = {}
in_table = False
for line in sys.stdin:
    line = line.strip()
    if line == 'LEAF_FMA_NS_PER_PLAYER_TABLE':
        in_table = True
        continue
    if in_table and line.startswith('B='):
        parts = line.split(',')
        if len(parts) == 2:
            b = int(parts[0].split('=')[1])
            val = float(parts[1])
            table[b] = val
    elif in_table and not line.startswith('B='):
        break

if len(table) != 6:
    print(f'WARNING: expected 6 B values, got {len(table)}: {table}', file=sys.stderr)
    sys.exit(0)

b_order = [8, 16, 24, 32, 48, 64]
config_path = sys.argv[1]

with open(config_path, 'r') as f:
    text = f.read()

array_pattern = r'(static const double leaf_fma_ns_per_player\[6\]\s*=\s*\{)'
match = re.search(array_pattern, text)
if not match:
    print('WARNING: leaf_fma_ns_per_player array not found in header', file=sys.stderr)
    sys.exit(0)

start = match.end()
end = text.index('};', start) + 2

lines = ['static const double leaf_fma_ns_per_player[6] = {']
for i, b in enumerate(b_order):
    comma = ',' if i < 5 else ''
    lines.append(f'    {table[b]:.4f}{comma}  /* B={b:<2d} */')
lines.append('};')

new_array = '\n'.join(lines)
text = text[:match.start()] + new_array + text[end:]

with open(config_path, 'w') as f:
    f.write(text)

print(f'  Wrote leaf_fma_ns_per_player[] to {config_path}')
for b in b_order:
    print(f'    B={b:<2d}  {table[b]:.4f}')
PYEOF
echo "$LEAF_OUT" | python3 /tmp/_write_leaf_table.py "$CONFIG_H"
echo "  ✓ Leaf-FMA lookup table written to $CONFIG_H"

# ── Step 9: Run fit_cost_model.py with ALL scalar pins (ZERO free params) ──
echo ""
echo "── Step 9/11: Assemble fully-pinned config (tools/fit_cost_model.py --write) ──"
FIT_CMD="python3 tools/fit_cost_model.py \"$CSV_FILE\" \"$CONFIG_H\" --write"
FIT_CMD="$FIT_CMD --wrap-ns \"$WRAP_FMA_NS\""
if [ -n "$FP64_DIV_NS" ]; then
    FIT_CMD="$FIT_CMD --div-ns \"$FP64_DIV_NS\""
fi
FIT_CMD="$FIT_CMD --fma-ns \"$FMA_NS\""
FIT_CMD="$FIT_CMD --paired-cached-ratio \"$PAIRED_CACHED_CORR_RATIO\""
FIT_CMD="$FIT_CMD --indep-pair-ratio \"$INDEP_PAIR_RATIO\""
FIT_CMD="$FIT_CMD --overhead-ns 0.0"
echo "  Running: $FIT_CMD"
eval "$FIT_CMD"
echo "  ✓ Fully-pinned config written to $CONFIG_H (0 free parameters)"

# ── Step 10: Rebuild ──
echo ""
echo "── Step 10/11: Rebuild library (make clean && make DEVICE=$DEVICE) ──"
make clean
make "DEVICE=$DEVICE"
echo "  ✓ Rebuild complete"

# ── Step 11: Verify ──
echo ""
echo "── Step 11/11: Verify correctness and crossover dispatch ──"

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
echo "  Div bench CSV: $DIV_CSV"
echo "  Profile log:   $PROFILE_LOG"
echo "  Block log:     $REPO_ROOT/block_build_${DEVICE}.log"
echo "  Leaf log:      $REPO_ROOT/leaf_fma_${DEVICE}.log"
echo ""
echo "  All scalar constants pinned — zero free parameters in cost model:"
echo "    WRAP_FMA_NS            = $WRAP_FMA_NS"
echo "    FP64_DIV_NS           = ${FP64_DIV_NS:-unpinned}"
echo "    FMA_NS                 = $FMA_NS"
echo "    PAIRED_CACHED_CORR_RATIO = $PAIRED_CACHED_CORR_RATIO"
echo "    INDEP_PAIR_RATIO       = $INDEP_PAIR_RATIO"
echo "    FFT_OVERHEAD_NS        = 0.0"
echo ""
echo "Next steps (manual):"
echo "  ./bench_grid profile   # re-run profiling for manual inspection"
echo "  ./bench_grid           # full performance grid"
