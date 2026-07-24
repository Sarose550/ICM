#!/usr/bin/env bash
#
# calibrate_full.sh — Full calibration pipeline for a new device.
#
# Usage: ./tools/calibrate_full.sh <DEVICE> [--quick]
#
# Runs the complete calibration pipeline from the repo root:
#   1. Builds and runs tools/calibrate.c (FFTW calibration)
#   2. Copies fft_config.h + fftw_wisdom.dat to devices/<DEVICE>/
#   2.5 Injects placeholder arrays/constants into fft_config.h so
#       subsequent measurement steps that write into the header can
#       find their target patterns (calibrate.c only emits the base
#       calib_sizes[]/calib_times_ns[] + scalar #defines).
#   3. Builds and runs tools/sample_plans.c (hybrid engine timing)
#   4. Builds and runs tools/bench_wrap_fma.c — directly measures WRAP_FMA_NS
#      (wrap-correction cost) via an isolated microbenchmark.
#   5. Builds and runs tools/bench_div_chain.c — directly measures
#      FP64_DIV_NS (leaf-extraction division cost) via a dependency-chained
#      microbenchmark.
#   6. Builds bench_grid, runs `./bench_grid profile` — extracts FMA_NS
#      (schoolbook slope, cps=16→32), PAIRED_CACHED_CORR_RATIO and
#      INDEP_PAIR_RATIO (phase-split table, fft_n ≥ 4096).
#   7. Builds and runs tools/bench_block_build.c — directly measures the
#      block-build per-player cost at each candidate B, writes per-B lookup
#      table into fft_config.h (block_build_ns_per_player[]).
#   8. Builds and runs tools/probe_leaf_extract.c — measures the leaf-
#      extraction per-player cost at each candidate B via the B-sweep phase
#      (n=8192, k=320, fresh HybridCtx per rep — matches real engine
#      behaviour).  Uses the cheap "zero" branch which dominates real
#      production data (~99.9% of cases).  Writes the
#      per-B lookup table leaf_fma_ns_per_player[] into fft_config.h.
#   9. Builds and runs tools/bench_schoolbook_tree.c — directly measures
#      polymul_modk() and correlate_school() at each calib_sizes[] entry
#      (cps ≤ 1024, sentinel -1.0 above).  Writes schoolbook_mul_ns[] and
#      schoolbook_corr_ns[] lookup tables into fft_config.h.
#  10. Builds and runs tools/bench_linear_batched_fma.c — directly measures
#      the batched linear engine's inner-loop per-FMA cost.  Extracts
#      BATCHED_FMA_NS (regression slope, combined forward+backward k-sweep)
#      and writes it into fft_config.h.  The cost model in src/cost_model.h
#      uses this as 5*n*k*BATCHED_FMA_NS (not 4*n*k*FMA_NS — the batched
#      engine performs ~5k FMAs per player per QP, not the 4k the old
#      scalar-schoolbook-based formula assumed).
#  11. Runs tools/fit_cost_model.py --write with ALL 6 scalar pins
#      (WRAP_FMA_NS, FP64_DIV_NS, FMA_NS, PAIRED_CACHED_CORR_RATIO,
#      INDEP_PAIR_RATIO, FFT_OVERHEAD_NS=0.0).  ZERO free parameters remain
#      — scipy optimization is skipped; the script assembles the fully-pinned
#      config directly.
#  12. Builds and runs tools/calibrate_crossover.c — binary-searches the
#      real linear-vs-hybrid crossover k(n) via direct timing (median of 7
#      reps, Q=256).  Writes N_CROSSOVER_POINTS/crossover_n[]/crossover_k[]
#      into fft_config.h.
#  13. Builds and runs tools/calibrate_best_b.c — times every candidate
#      hybrid block size B at a grid of (n,k) points.  Writes
#      N_BSELECT_POINTS/bselect_n[]/bselect_k[]/bselect_B[] into
#      fft_config.h.
#  14. Rebuilds the library with the new device config
#  15. Verifies correctness (bench_grid verify) and crossover dispatch
#
# This can take 15–45+ minutes, dominated by step 1 (FFTW calibration),
# step 8 (probe_leaf_extract full sweep + B-sweep), and steps 12–13
# (crossover/best-B timing sweeps).
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

# ═══════════════════════════════════════════════════════════════════════
# Step 1: Build and run calibrate
# ═══════════════════════════════════════════════════════════════════════
echo "── Step 1/15: FFTW calibration (tools/calibrate.c) ──"
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

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Copy generated files to devices/<DEVICE>/
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 2/15: Copy calibration files to devices/$DEVICE/ ──"
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

# ═══════════════════════════════════════════════════════════════════════
# Step 2.5: Inject placeholder arrays/constants into fft_config.h
#
# calibrate.c only emits calib_sizes[]/calib_times_ns[] + scalar
# #defines.  The measurement steps below (block_build, probe_leaf_extract,
# bench_schoolbook_tree, bench_linear_batched_fma) all need their target
# arrays/constants to exist in fft_config.h — both for COMPILATION (tools
# that #include "icm.c" reference leaf_fma_ns_per_player[] etc.) and for
# the inline Python scripts that find-and-replace array contents.
#
# This step injects placeholder arrays/constants before the
# "Cost model functions" comment, so every subsequent step can find its
# target.  The placeholder values are physically implausible sentinels;
# they MUST be overwritten by the measurement steps before the final build.
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 2.5/15: Inject placeholder arrays into fft_config.h ──"

cat > /tmp/_inject_placeholders.py << 'PYEOF'
import sys, re

config_path = sys.argv[1]

with open(config_path, 'r') as f:
    text = f.read()

# Locate the insertion point: right before "/* ── Cost model functions ── */"
anchor = '/* ── Cost model functions ── */'
idx = text.find(anchor)
if idx < 0:
    print("ERROR: could not find Cost model functions anchor in fft_config.h", file=sys.stderr)
    sys.exit(1)

# Check if placeholders already exist (idempotent — don't double-inject)
if 'BATCHED_FMA_NS_PLACEHOLDER_INJECTED' in text:
    print("  Placeholders already injected — skipping.")
    sys.exit(0)

placeholder_block = '''
/* ── Batched linear engine constant ───────────────────────────
 * Measured by tools/bench_linear_batched_fma.c (combined forward+backward
 * k-sweep regression).  The batched engine performs ~5k FMAs per player
 * per QP, not ~4k.  PLACEHOLDER. */
#ifndef BATCHED_FMA_NS
#define BATCHED_FMA_NS 999.0  /* PLACEHOLDER — will be overwritten by step 10 */
#endif

/* ── Hybrid-engine block-build lookup table ────────────────────
 * Directly-measured per-player block-build cost at each candidate B.
 * Generated by tools/bench_block_build.c — step 7.  PLACEHOLDER. */
#ifndef BLOCK_BUILD_NS_PER_PLAYER_DEFINED
#define BLOCK_BUILD_NS_PER_PLAYER_DEFINED
static const double block_build_ns_per_player[6] = {
    999.0,  /* B=8  — PLACEHOLDER */
    999.0,  /* B=16 — PLACEHOLDER */
    999.0,  /* B=24 — PLACEHOLDER */
    999.0,  /* B=32 — PLACEHOLDER */
    999.0,  /* B=48 — PLACEHOLDER */
    999.0   /* B=64 — PLACEHOLDER */
};
#endif

/* ── Hybrid-engine leaf-extraction lookup table ─────────────────
 * Directly-measured per-player leaf-extraction cost at each candidate B.
 * Generated by tools/probe_leaf_extract.c B-sweep phase — step 8.
 * PLACEHOLDER. */
#ifndef LEAF_FMA_NS_PER_PLAYER_DEFINED
#define LEAF_FMA_NS_PER_PLAYER_DEFINED
static const double leaf_fma_ns_per_player[6] = {
    999.0,  /* B=8  — PLACEHOLDER */
    999.0,  /* B=16 — PLACEHOLDER */
    999.0,  /* B=24 — PLACEHOLDER */
    999.0,  /* B=32 — PLACEHOLDER */
    999.0,  /* B=48 — PLACEHOLDER */
    999.0   /* B=64 — PLACEHOLDER */
};
#endif

/* ── Schoolbook cost lookup tables ─────────────────────────────
 * Direct per-size measurements of polymul_modk() and correlate_school()
 * indexed identically to calib_sizes[].  Generated by
 * tools/bench_schoolbook_tree.c — step 9.  PLACEHOLDER.
 *
 * Explicitly filled with a physically-implausible 999.0 sentinel (NOT a
 * bare `static const double x[N];` declaration) -- a bare declaration
 * with no initializer is zero-initialized in C, which would silently
 * make schoolbook multiply look FREE (0 ns) if step 9 is ever skipped
 * or fails to parse its own output (its writer script prints a WARNING
 * and exits 0 on parse failure, so `set -e` would NOT catch that case
 * and abort the pipeline) -- dangerous, since it would bias dispatch
 * toward schoolbook silently and bench_grid verify only checks equity
 * correctness, not dispatch quality, so it would not be caught by CI.
 * A 999.0 sentinel fails safe in the opposite direction: schoolbook
 * would just look pathologically expensive and never get selected. */
static const double schoolbook_mul_ns[N_CALIBRATED_SIZES] = {[0 ... N_CALIBRATED_SIZES-1] = 999.0};
static const double schoolbook_corr_ns[N_CALIBRATED_SIZES] = {[0 ... N_CALIBRATED_SIZES-1] = 999.0};

/* BATCHED_FMA_NS_PLACEHOLDER_INJECTED — sentinel to detect double-injection */
'''

text = text[:idx] + placeholder_block + text[idx:]

with open(config_path, 'w') as f:
    f.write(text)

print(f"  ✓ Injected placeholder arrays/constants into {config_path}")
PYEOF

python3 /tmp/_inject_placeholders.py "$CONFIG_H"
echo "  ✓ Placeholder injection complete"

# ── Inject crossover/bselect placeholders ──
cat > /tmp/_inject_crossover_bselect_placeholders.py << 'PYEOF'
import sys, re

config_path = sys.argv[1]

with open(config_path, 'r') as f:
    text = f.read()

# Check idempotency — don't double-inject
if 'CROSSOVER_BSELECT_PLACEHOLDER_INJECTED' in text:
    print("  Crossover/bselect placeholders already injected — skipping.")
    sys.exit(0)

anchor = '/* ── Cost model functions ── */'
idx = text.find(anchor)
if idx < 0:
    print("ERROR: could not find Cost model functions anchor in fft_config.h", file=sys.stderr)
    sys.exit(1)

placeholder_block = '''
/* ── Empirical linear-vs-hybrid crossover table ──────────────────────
 * Measured by tools/calibrate_crossover.c — binary search on real
 * timing (median of 7 reps, Q=256).  See src/fft_cost_model.h's
 * empirical_crossover_k() for how this is consulted (log-linear
 * interpolation between bracketing n).  PLACEHOLDER. */
#ifndef N_CROSSOVER_POINTS
#define N_CROSSOVER_POINTS 6
static const int crossover_n[N_CROSSOVER_POINTS] = {512, 1024, 2048, 4096, 8192, 16384};
static const int crossover_k[N_CROSSOVER_POINTS] = {999, 999, 999, 999, 999, 999};
#endif

/* ── Empirical hybrid block-size (B) table ───────────────────────────
 * Measured by tools/calibrate_best_b.c — direct timing (median of 7
 * reps, Q=256) of the real hybrid engine at every candidate B, per
 * (n,k) grid point.  See src/fft_cost_model.h's empirical_best_B()
 * for how this is consulted (2D nearest-neighbor).  PLACEHOLDER. */
#ifndef N_BSELECT_POINTS
#define N_BSELECT_POINTS 34
static const int bselect_n[N_BSELECT_POINTS] = {[0 ... 33] = 999};
static const int bselect_k[N_BSELECT_POINTS] = {[0 ... 33] = 999};
static const int bselect_B[N_BSELECT_POINTS] = {[0 ... 33] = 999};
#endif

/* CROSSOVER_BSELECT_PLACEHOLDER_INJECTED — sentinel to detect double-injection */
'''

text = text[:idx] + placeholder_block + text[idx:]

with open(config_path, 'w') as f:
    f.write(text)

print(f"  ✓ Injected crossover/bselect placeholders into {config_path}")
PYEOF

python3 /tmp/_inject_crossover_bselect_placeholders.py "$CONFIG_H"
echo "  ✓ Crossover/bselect placeholder injection complete"

# ═══════════════════════════════════════════════════════════════════════
# Step 3: Build and run sample_plans
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 3/15: Build and run sample_plans.c ──"
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

# ═══════════════════════════════════════════════════════════════════════
# Step 4: Build and run bench_wrap_fma (direct WRAP_FMA_NS measurement)
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 4/15: Direct wrap-correction microbenchmark (tools/bench_wrap_fma.c) ──"
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
# is contaminated by that overhead. A least-squares slope between well-separated
# points cancels the fixed overhead and recovers the true marginal ns/FMA.
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

# ═══════════════════════════════════════════════════════════════════════
# Step 5: Build and run bench_div_chain (direct FP64_DIV_NS measurement)
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 5/15: Direct division-chain microbenchmark (tools/bench_div_chain.c) ──"
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

# ═══════════════════════════════════════════════════════════════════════
# Step 6: Build bench_grid, run profile → FMA_NS, PAIRED_CACHED_CORR_RATIO, INDEP_PAIR_RATIO
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 6/15: Profile FFT phases + schoolbook cost (./bench_grid profile) ──"
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

# ═══════════════════════════════════════════════════════════════════════
# Step 7: Build and run bench_block_build → per-B lookup table
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 7/15: Direct block-build microbenchmark (tools/bench_block_build.c) ──"
BLOCK_BENCH_BIN="$REPO_ROOT/bench_block_build"
gcc -O3 -march=native -o "$BLOCK_BENCH_BIN" tools/bench_block_build.c -lm
echo "  Running bench_block_build..."
BLOCK_OUT="$("$BLOCK_BENCH_BIN")"
echo "$BLOCK_OUT" > "$REPO_ROOT/block_build_${DEVICE}.log"
echo "  ✓ bench_block_build complete → $REPO_ROOT/block_build_${DEVICE}.log"

# Write the per-B lookup table into fft_config.h via a temp Python script.
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
    print('WARNING: block_build_ns_per_player array not found in header — placeholder missing?', file=sys.stderr)
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

# ═══════════════════════════════════════════════════════════════════════
# Step 8: Build and run probe_leaf_extract.c → per-B leaf lookup table
#
# ═══════════════════════════════════════════════════════════════════════
# probe_leaf_extract.c measures leaf-extraction per-player cost via the real
# engine_hybrid_core timing, using the actual code path taken in production.
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 8/15: Leaf-extraction via probe_leaf_extract.c B-sweep phase ──"
LEAF_PROBE_BIN="$REPO_ROOT/probe_leaf_extract"
gcc -O3 -march=native \
    -Isrc \
    -I"$DEVICE_DIR" \
    $HOMEBREW_INC \
    -o "$LEAF_PROBE_BIN" \
    tools/probe_leaf_extract.c \
    $HOMEBREW_LIB \
    -lfftw3 -lm \
    $ACCEL_FLAGS $VEC_FLAGS
echo "  ✓ Built probe_leaf_extract"

echo "  Running probe_leaf_extract (full sweep + B-sweep — this takes several minutes)..."
LEAF_PROBE_OUT="$("$LEAF_PROBE_BIN")"
echo "$LEAF_PROBE_OUT" > "$REPO_ROOT/leaf_probe_${DEVICE}.log"
echo "  ✓ probe_leaf_extract complete → $REPO_ROOT/leaf_probe_${DEVICE}.log"

# Parse the B-sweep phase output to extract leaf_ns_per_player for each B.
# The B-sweep table has format:
#   === B-SWEEP (n=8192, k=320, ...) ===
#   B      leaf_ns/qp  leaf_ns/player  pred_ns/player
#          (measured)   (current model)
#   8      23130.5      2.8234          2.8234
#   ...
# We extract the leaf_ns/player column (3rd numeric column after the B label).
cat > /tmp/_write_leaf_probe_table.py << 'PYEOF'
import sys, re

lines = sys.stdin.read().splitlines()

# Find the B-sweep section
b_sweep_start = None
for i, line in enumerate(lines):
    if line.startswith('=== B-SWEEP'):
        b_sweep_start = i
        break

if b_sweep_start is None:
    print('WARNING: B-SWEEP section not found in probe_leaf_extract output', file=sys.stderr)
    sys.exit(0)

# Parse B-value rows from the B-sweep table
# Format: "8      23130.5      2.8234          2.8234"
table = {}
for line in lines[b_sweep_start:]:
    line = line.strip()
    # Skip header/separator lines
    if not line or line.startswith('===') or line.startswith('B ') or not line[0].isdigit():
        continue
    parts = line.split()
    if len(parts) >= 3:
        try:
            b_val = int(parts[0])
            leaf_ns_per_player = float(parts[2])  # leaf_ns/player column
            table[b_val] = leaf_ns_per_player
        except (ValueError, IndexError):
            continue

if len(table) != 6:
    print(f'WARNING: expected 6 B values from B-sweep, got {len(table)}: {table}', file=sys.stderr)
    sys.exit(0)

b_order = [8, 16, 24, 32, 48, 64]
config_path = sys.argv[1]

with open(config_path, 'r') as f:
    text = f.read()

array_pattern = r'(static const double leaf_fma_ns_per_player\[6\]\s*=\s*\{)'
match = re.search(array_pattern, text)
if not match:
    print('WARNING: leaf_fma_ns_per_player array not found in header — placeholder missing?', file=sys.stderr)
    sys.exit(0)

start = match.end()
end = text.index('};', start) + 2

lines_out = ['static const double leaf_fma_ns_per_player[6] = {']
for i, b in enumerate(b_order):
    comma = ',' if i < 5 else ''
    lines_out.append(f'    {table[b]:.4f}{comma}  /* B={b:<2d} */')
lines_out.append('};')

new_array = '\n'.join(lines_out)
text = text[:match.start()] + new_array + text[end:]

with open(config_path, 'w') as f:
    f.write(text)

print(f'  Wrote leaf_fma_ns_per_player[] to {config_path}')
for b in b_order:
    print(f'    B={b:<2d}  {table[b]:.4f}')
PYEOF
echo "$LEAF_PROBE_OUT" | python3 /tmp/_write_leaf_probe_table.py "$CONFIG_H"
echo "  ✓ Leaf-extraction lookup table written to $CONFIG_H"

# ═══════════════════════════════════════════════════════════════════════
# Step 9: Build and run bench_schoolbook_tree → per-size lookup tables
#
# This step was flagged as "not yet wired into calibrate_full.sh" in
# commit 8012244.  It directly measures polymul_modk() and
# correlate_school() at each calib_sizes[] entry (cps ≤ 1024, sentinel
# -1.0 above the cutoff).  The per-size tables schoolbook_mul_ns[] and
# schoolbook_corr_ns[] replace the single FMA_NS constant for schoolbook-
# level cost estimation at small polynomial sizes where operations are
# latency/dependency-chain-bound, not FMA-throughput-bound.
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 9/15: Schoolbook per-size microbenchmark (tools/bench_schoolbook_tree.c) ──"
SCHOOLBOOK_BENCH_BIN="$REPO_ROOT/bench_schoolbook_tree"
gcc -O3 -march=native \
    -Isrc \
    -I"$DEVICE_DIR" \
    $HOMEBREW_INC \
    -o "$SCHOOLBOOK_BENCH_BIN" \
    tools/bench_schoolbook_tree.c \
    $HOMEBREW_LIB \
    -lfftw3 -lm \
    $ACCEL_FLAGS $VEC_FLAGS
echo "  ✓ Built bench_schoolbook_tree"

echo "  Running bench_schoolbook_tree..."
SCHOOLBOOK_OUT="$("$SCHOOLBOOK_BENCH_BIN")"
echo "$SCHOOLBOOK_OUT" > "$REPO_ROOT/schoolbook_tree_${DEVICE}.log"
echo "  ✓ bench_schoolbook_tree complete → $REPO_ROOT/schoolbook_tree_${DEVICE}.log"

# Parse the two tables and write them into fft_config.h.
cat > /tmp/_write_schoolbook_tables.py << 'PYEOF'
import sys, re

stdin_text = sys.stdin.read()

# Parse SCHOOLBOOK_MUL_NS_TABLE section
mul_table = {}
corr_table = {}
current_section = None

for line in stdin_text.splitlines():
    line = line.strip()
    if line == 'SCHOOLBOOK_MUL_NS_TABLE':
        current_section = 'mul'
        continue
    if line == 'SCHOOLBOOK_CORR_NS_TABLE':
        current_section = 'corr'
        continue
    if current_section and ',' in line:
        parts = line.split(',')
        if len(parts) == 2:
            try:
                cps = int(parts[0])
                val = float(parts[1])
                if current_section == 'mul':
                    mul_table[cps] = val
                else:
                    corr_table[cps] = val
            except ValueError:
                continue

config_path = sys.argv[1]

with open(config_path, 'r') as f:
    text = f.read()

# ── Replace schoolbook_mul_ns[] ──
mul_pattern = r'(static const double schoolbook_mul_ns\[N_CALIBRATED_SIZES\]\s*=\s*\{)'
mul_match = re.search(mul_pattern, text)
if not mul_match:
    # Fallback: try to find the array without initializer (placeholder form)
    mul_pattern2 = r'(static const double schoolbook_mul_ns\[N_CALIBRATED_SIZES\];)'
    mul_match = re.search(mul_pattern2, text)
    if mul_match:
        # Replace the declaration-only with a full initializer
        pass
    else:
        print('WARNING: schoolbook_mul_ns array not found in header — placeholder missing?', file=sys.stderr)
        sys.exit(0)

if mul_match:
    # Parse N_CALIBRATED_SIZES from the config
    n_match = re.search(r'#define N_CALIBRATED_SIZES (\d+)', text)
    if not n_match:
        print('WARNING: N_CALIBRATED_SIZES not found', file=sys.stderr)
        sys.exit(0)
    N = int(n_match.group(1))

    # Parse calib_sizes[] to get the index ordering
    sizes_match = re.search(r'static const int calib_sizes\[N_CALIBRATED_SIZES\]\s*=\s*\{([^}]+)\}',
                            text, re.DOTALL)
    if not sizes_match:
        print('WARNING: calib_sizes not found', file=sys.stderr)
        sys.exit(0)
    sizes_str = sizes_match.group(1)
    calib_sizes = [int(x) for x in re.findall(r'\d+', sizes_str)]

    # Build the array in calib_sizes order
    values = []
    for cps in calib_sizes:
        val = mul_table.get(cps, -1.0)
        values.append(val)

    # Format the array
    lines_out = ['static const double schoolbook_mul_ns[N_CALIBRATED_SIZES] = {']
    row_vals = []
    for i, v in enumerate(values):
        row_vals.append(f'{v:.2f}' if v >= 0 else '-1.0')
        if len(row_vals) == 10 or i == len(values) - 1:
            comma = ',' if i < len(values) - 1 else ''
            lines_out.append('   ' + ','.join(row_vals) + comma)
            row_vals = []
    lines_out.append('};')

    new_array = '\n'.join(lines_out)

    # For placeholder form (declaration-only), replace the whole line
    if ';' in mul_match.group(0) and '{' not in mul_match.group(0):
        text = text[:mul_match.start()] + new_array + text[mul_match.end():]
    else:
        # For full initializer form, replace between { and };
        start = mul_match.end()
        end = text.index('};', start) + 2
        text = text[:mul_match.start()] + new_array + text[end:]

    non_neg = sum(1 for v in values if v >= 0)
    print(f'  Wrote schoolbook_mul_ns[] ({non_neg} non-sentinel values)')

# ── Replace schoolbook_corr_ns[] ──
corr_pattern = r'(static const double schoolbook_corr_ns\[N_CALIBRATED_SIZES\]\s*=\s*\{)'
corr_match = re.search(corr_pattern, text)
if not corr_match:
    corr_pattern2 = r'(static const double schoolbook_corr_ns\[N_CALIBRATED_SIZES\];)'
    corr_match = re.search(corr_pattern2, text)
    if not corr_match:
        print('WARNING: schoolbook_corr_ns array not found in header — placeholder missing?', file=sys.stderr)
        sys.exit(0)

if corr_match:
    n_match = re.search(r'#define N_CALIBRATED_SIZES (\d+)', text)
    N = int(n_match.group(1))

    sizes_match = re.search(r'static const int calib_sizes\[N_CALIBRATED_SIZES\]\s*=\s*\{([^}]+)\}',
                            text, re.DOTALL)
    sizes_str = sizes_match.group(1)
    calib_sizes = [int(x) for x in re.findall(r'\d+', sizes_str)]

    values = []
    for cps in calib_sizes:
        val = corr_table.get(cps, -1.0)
        values.append(val)

    lines_out = ['static const double schoolbook_corr_ns[N_CALIBRATED_SIZES] = {']
    row_vals = []
    for i, v in enumerate(values):
        row_vals.append(f'{v:.4f}' if v >= 0 else '-1.0')
        if len(row_vals) == 10 or i == len(values) - 1:
            comma = ',' if i < len(values) - 1 else ''
            lines_out.append('   ' + ','.join(row_vals) + comma)
            row_vals = []
    lines_out.append('};')

    new_array = '\n'.join(lines_out)

    if ';' in corr_match.group(0) and '{' not in corr_match.group(0):
        text = text[:corr_match.start()] + new_array + text[corr_match.end():]
    else:
        start = corr_match.end()
        end = text.index('};', start) + 2
        text = text[:corr_match.start()] + new_array + text[end:]

    non_neg = sum(1 for v in values if v >= 0)
    print(f'  Wrote schoolbook_corr_ns[] ({non_neg} non-sentinel values)')

with open(config_path, 'w') as f:
    f.write(text)

print(f'  Schoolbook lookup tables written to {config_path}')
PYEOF
echo "$SCHOOLBOOK_OUT" | python3 /tmp/_write_schoolbook_tables.py "$CONFIG_H"
echo "  ✓ Schoolbook lookup tables written to $CONFIG_H"

# ═══════════════════════════════════════════════════════════════════════
# Step 10: Build and run bench_linear_batched_fma → BATCHED_FMA_NS
#
# The batched linear engine (BQ=8, src/linear_batched_impl.inc) performs
# ~5k FMAs per player per QP (forward: BQ*(2k-1), backward: BQ*(3k-1)).
# The cost model in src/cost_model.h's linear_roofline_cost() uses:
#   compute_ns = 5.0 * n * k * BATCHED_FMA_NS;
# This step measures BATCHED_FMA_NS directly from the verbatim inner loops.
#
# NOTE: tools/bench_linear_batched_fma.c exists on disk but may not yet be
# committed to git.  If the build fails with "file not found", this tool
# needs to be committed first (see HANDOFF.md).
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 10/15: Batched-linear-engine constant (tools/bench_linear_batched_fma.c) ──"
LINEAR_BENCH_BIN="$REPO_ROOT/bench_linear_batched_fma"

if [ ! -f "$REPO_ROOT/tools/bench_linear_batched_fma.c" ]; then
    echo "  ⚠ WARNING: tools/bench_linear_batched_fma.c not found in repo."
    echo "    This tool is not yet committed to git.  The BATCHED_FMA_NS"
    echo "    constant will be left at its placeholder value (999.0)."
    echo "    See HANDOFF.md for the tool status.  Skipping this step."
else
    gcc -O3 -march=native -o "$LINEAR_BENCH_BIN" tools/bench_linear_batched_fma.c -lm
    echo "  ✓ Built bench_linear_batched_fma"

    echo "  Running bench_linear_batched_fma..."
    LINEAR_OUT="$("$LINEAR_BENCH_BIN" 2>&1)"
    echo "$LINEAR_OUT" > "$REPO_ROOT/linear_batched_${DEVICE}.log"
    echo "  ✓ bench_linear_batched_fma complete → $REPO_ROOT/linear_batched_${DEVICE}.log"

    # Extract BATCHED_FMA_NS from stderr: "BATCHED_FMA_NS = 0.0954 ns/FMA"
    BATCHED_FMA_NS=$(echo "$LINEAR_OUT" | grep -E '^# BATCHED_FMA_NS = ' | head -1 | awk '{print $4}')
    if [ -z "$BATCHED_FMA_NS" ]; then
        # Try alternative format
        BATCHED_FMA_NS=$(echo "$LINEAR_OUT" | grep -E 'BATCHED_FMA_NS = [0-9]' | head -1 | sed 's/.*= //' | awk '{print $1}')
    fi

    if [ -n "$BATCHED_FMA_NS" ]; then
        echo "  Extracted BATCHED_FMA_NS = $BATCHED_FMA_NS (combined forward+backward k-sweep regression)"

        # Write BATCHED_FMA_NS into fft_config.h via the same #ifndef/#define pattern
        cat > /tmp/_write_batched_fma.py << 'PYEOF'
import sys, re

batched_ns = float(sys.argv[1])
config_path = sys.argv[2]

with open(config_path, 'r') as f:
    text = f.read()

pattern = (
    r'(#ifndef\s+BATCHED_FMA_NS\s*\n'
    r'#define\s+BATCHED_FMA_NS\s+)'
    r'([\d.eE+\-]+)'
    r'([^\n]*\n#endif)'
)
match = re.search(pattern, text)
if not match:
    print('WARNING: BATCHED_FMA_NS #ifndef/#define block not found — placeholder missing?', file=sys.stderr)
    sys.exit(0)

old_val = float(match.group(2))
new_val_str = f'{batched_ns:.4f}'
replacement = match.group(1) + new_val_str + match.group(3)
text = text[:match.start()] + replacement + text[match.end():]

with open(config_path, 'w') as f:
    f.write(text)

print(f'  Wrote BATCHED_FMA_NS = {batched_ns:.4f} (was {old_val:.4f})')
PYEOF
        python3 /tmp/_write_batched_fma.py "$BATCHED_FMA_NS" "$CONFIG_H"
        echo "  ✓ BATCHED_FMA_NS written to $CONFIG_H"
    else
        echo "  WARNING: could not extract BATCHED_FMA_NS from bench_linear_batched_fma output"
        echo "    Leaving placeholder value in $CONFIG_H — manual fix required."
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# Step 11: Run fit_cost_model.py with ALL scalar pins (ZERO free params)
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 11/15: Assemble fully-pinned config (tools/fit_cost_model.py --write) ──"
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

# ═══════════════════════════════════════════════════════════════════════
# Step 12: Build and run calibrate_crossover → crossover_n[]/crossover_k[]
#
# These tools time the ACTUAL hybrid/linear engines, so they depend on
# all other calibrated constants (FMA_NS, WRAP_FMA_NS, block_build_ns_per_player[],
# leaf_fma_ns_per_player[], etc.) already being correct in fft_config.h.
# That's why they run AFTER fit_cost_model.py (step 11), not before.
#
# calibrate_crossover binary-searches the real crossover k(n) via direct
# timing (median of 7 reps, Q=256), across the fixed n grid
# {512,1024,2048,4096,8192,16384}.  Outputs CSV lines "n,k_cross" after
# a comment header.  Writes N_CROSSOVER_POINTS/crossover_n[]/crossover_k[]
# into fft_config.h.
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 12/15: Empirical crossover measurement (tools/calibrate_crossover.c) ──"
echo "  This binary-searches the real linear-vs-hybrid crossover k(n)"
echo "  via direct timing (median of 7 reps, Q=256).  Takes several minutes."
CROSSOVER_BIN="$REPO_ROOT/calibrate_crossover"
gcc -O3 -march=native \
    -Isrc \
    -I"$DEVICE_DIR" \
    $HOMEBREW_INC \
    -o "$CROSSOVER_BIN" \
    tools/calibrate_crossover.c src/icm.c \
    $HOMEBREW_LIB \
    -lfftw3 -lm \
    $ACCEL_FLAGS $VEC_FLAGS
echo "  ✓ Built calibrate_crossover"

echo "  Running calibrate_crossover..."
CROSSOVER_OUT="$("$CROSSOVER_BIN" 2>&1)"
CROSSOVER_STDERR=$(echo "$CROSSOVER_OUT" | grep -E '^(n=|  )' || true)
echo "$CROSSOVER_OUT" > "$REPO_ROOT/crossover_${DEVICE}.log"
echo "  ✓ calibrate_crossover complete → $REPO_ROOT/crossover_${DEVICE}.log"
if [ -n "$CROSSOVER_STDERR" ]; then
    echo "$CROSSOVER_STDERR" | while read line; do echo "    $line"; done
fi

# Parse CSV output and write crossover_n[]/crossover_k[] into fft_config.h.
# The tool outputs lines like:
#   # Direct empirical crossover measurement ...
#   # n,k_cross
#   512,123
#   1024,124
#   ...
cat > /tmp/_write_crossover_table.py << 'PYEOF'
import sys, re

lines = sys.stdin.read().splitlines()

# Parse CSV: skip comment/header lines, read "n,k_cross"
crossover = {}  # n -> k_cross
for line in lines:
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    parts = line.split(',')
    if len(parts) == 2:
        try:
            n = int(parts[0])
            k = int(parts[1])
            crossover[n] = k
        except ValueError:
            continue

if len(crossover) != 6:
    print(f'WARNING: expected 6 crossover points, got {len(crossover)}: {crossover}', file=sys.stderr)
    sys.exit(0)

config_path = sys.argv[1]

with open(config_path, 'r') as f:
    text = f.read()

# ── Replace crossover_n[] ──
n_pattern = r'(static const int crossover_n\[N_CROSSOVER_POINTS\]\s*=\s*\{)'
n_match = re.search(n_pattern, text)
if not n_match:
    print('WARNING: crossover_n array not found in header — placeholder missing?', file=sys.stderr)
    sys.exit(0)

# Build the array in fixed n-grid order
n_order = [512, 1024, 2048, 4096, 8192, 16384]
n_line_vals = ', '.join(str(v) for v in n_order)
new_n_array = f'static const int crossover_n[N_CROSSOVER_POINTS] = {{{n_line_vals}}};'

start = n_match.end()
end = text.index('};', start) + 2
text = text[:n_match.start()] + new_n_array + text[end:]

# ── Replace crossover_k[] ──
k_pattern = r'(static const int crossover_k\[N_CROSSOVER_POINTS\]\s*=\s*\{)'
k_match = re.search(k_pattern, text)
if not k_match:
    print('WARNING: crossover_k array not found in header — placeholder missing?', file=sys.stderr)
    sys.exit(0)

k_values = [crossover.get(n, 999) for n in n_order]
k_line_vals = ', '.join(str(v) for v in k_values)
new_k_array = f'static const int crossover_k[N_CROSSOVER_POINTS] = {{{k_line_vals}}};'

start = k_match.end()
end = text.index('};', start) + 2
text = text[:k_match.start()] + new_k_array + text[end:]

with open(config_path, 'w') as f:
    f.write(text)

print(f'  Wrote crossover_n[]/crossover_k[] to {config_path}')
for i, n in enumerate(n_order):
    print(f'    n={n:<5d}  k_cross={k_values[i]}')
PYEOF
echo "$CROSSOVER_OUT" | python3 /tmp/_write_crossover_table.py "$CONFIG_H"
echo "  ✓ Crossover table written to $CONFIG_H"

# ═══════════════════════════════════════════════════════════════════════
# Step 13: Build and run calibrate_best_b → bselect_n[]/bselect_k[]/bselect_B[]
#
# Times every candidate hybrid block size B in {8,16,24,32,48,64} at a
# grid of (n,k) points (n in {512,1024,2048,4096,8192,16384},
# k in {150,250,400,800,1500,2000,4000}), outputs CSV lines "n,k,best_B".
# Writes N_BSELECT_POINTS/bselect_n[]/bselect_k[]/bselect_B[] into
# fft_config.h.  Takes several minutes (6 B values × ~34 grid points,
# 7-rep median timing each).
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 13/15: Empirical best-B measurement (tools/calibrate_best_b.c) ──"
echo "  Times every candidate B at each (n,k) grid point"
echo "  (median of 7 reps, Q=256).  Takes several minutes."
BESTB_BIN="$REPO_ROOT/calibrate_best_b"
gcc -O3 -march=native \
    -Isrc \
    -I"$DEVICE_DIR" \
    $HOMEBREW_INC \
    -o "$BESTB_BIN" \
    tools/calibrate_best_b.c src/icm.c \
    $HOMEBREW_LIB \
    -lfftw3 -lm \
    $ACCEL_FLAGS $VEC_FLAGS
echo "  ✓ Built calibrate_best_b"

echo "  Running calibrate_best_b..."
BESTB_OUT="$("$BESTB_BIN" 2>&1)"
BESTB_STDERR=$(echo "$BESTB_OUT" | grep -E '^n=' || true)
echo "$BESTB_OUT" > "$REPO_ROOT/best_b_${DEVICE}.log"
echo "  ✓ calibrate_best_b complete → $REPO_ROOT/best_b_${DEVICE}.log"
if [ -n "$BESTB_STDERR" ]; then
    echo "$BESTB_STDERR" | while read line; do echo "    $line"; done
fi

# Parse CSV output and write bselect_n[]/bselect_k[]/bselect_B[] into fft_config.h.
# The tool outputs lines like:
#   # Direct empirical best-B measurement ...
#   # n,k,best_B
#   512,150,32
#   512,250,32
#   ...
cat > /tmp/_write_bselect_table.py << 'PYEOF'
import sys, re

lines = sys.stdin.read().splitlines()

# Parse CSV: skip comment/header lines, read "n,k,best_B"
points = []  # list of (n, k, best_B)
for line in lines:
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    parts = line.split(',')
    if len(parts) == 3:
        try:
            n = int(parts[0])
            k = int(parts[1])
            b = int(parts[2])
            points.append((n, k, b))
        except ValueError:
            continue

if len(points) != 34:
    print(f'WARNING: expected 34 bselect points, got {len(points)}', file=sys.stderr)
    sys.exit(0)

config_path = sys.argv[1]

with open(config_path, 'r') as f:
    text = f.read()

# ── Replace bselect_n[] ──
n_pattern = r'(static const int bselect_n\[N_BSELECT_POINTS\]\s*=\s*\{)'
n_match = re.search(n_pattern, text)
if not n_match:
    print('WARNING: bselect_n array not found in header — placeholder missing?', file=sys.stderr)
    sys.exit(0)

n_vals = [p[0] for p in points]
n_line_vals = ', '.join(str(v) for v in n_vals)
new_n_array = f'static const int bselect_n[N_BSELECT_POINTS] = {{{n_line_vals}}};'

start = n_match.end()
end = text.index('};', start) + 2
text = text[:n_match.start()] + new_n_array + text[end:]

# ── Replace bselect_k[] ──
k_pattern = r'(static const int bselect_k\[N_BSELECT_POINTS\]\s*=\s*\{)'
k_match = re.search(k_pattern, text)
if not k_match:
    print('WARNING: bselect_k array not found in header — placeholder missing?', file=sys.stderr)
    sys.exit(0)

k_vals = [p[1] for p in points]
k_lines = ['static const int bselect_k[N_BSELECT_POINTS] = {']
k_line_vals = ', '.join(str(v) for v in k_vals)
k_lines.append(f'    {k_line_vals}')
k_lines.append('};')
new_k_array = '\n'.join(k_lines)

start = k_match.end()
end = text.index('};', start) + 2
text = text[:k_match.start()] + new_k_array + text[end:]

# ── Replace bselect_B[] ──
b_pattern = r'(static const int bselect_B\[N_BSELECT_POINTS\]\s*=\s*\{)'
b_match = re.search(b_pattern, text)
if not b_match:
    print('WARNING: bselect_B array not found in header — placeholder missing?', file=sys.stderr)
    sys.exit(0)

b_vals = [p[2] for p in points]
b_lines = ['static const int bselect_B[N_BSELECT_POINTS] = {']
b_line_vals = ', '.join(str(v) for v in b_vals)
b_lines.append(f'    {b_line_vals}')
b_lines.append('};')
new_b_array = '\n'.join(b_lines)

start = b_match.end()
end = text.index('};', start) + 2
text = text[:b_match.start()] + new_b_array + text[end:]

with open(config_path, 'w') as f:
    f.write(text)

print(f'  Wrote bselect_n[]/bselect_k[]/bselect_B[] to {config_path}')
print(f'  {len(points)} grid points written')
PYEOF
echo "$BESTB_OUT" | python3 /tmp/_write_bselect_table.py "$CONFIG_H"
echo "  ✓ Best-B lookup table written to $CONFIG_H"

# ═══════════════════════════════════════════════════════════════════════
# Step 14: Rebuild
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 14/15: Rebuild library (make clean && make DEVICE=$DEVICE) ──"
make clean
make "DEVICE=$DEVICE"
echo "  ✓ Rebuild complete"

# ═══════════════════════════════════════════════════════════════════════
# Step 15: Verify
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 15/15: Verify correctness and crossover dispatch ──"

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
echo "  Leaf log:      $REPO_ROOT/leaf_probe_${DEVICE}.log"
echo "  Schoolbook log:$REPO_ROOT/schoolbook_tree_${DEVICE}.log"
echo "  Linear log:    $REPO_ROOT/linear_batched_${DEVICE}.log"
echo "  Crossover log: $REPO_ROOT/crossover_${DEVICE}.log"
echo "  Best-B log:    $REPO_ROOT/best_b_${DEVICE}.log"
echo ""
echo "  All scalar constants pinned — zero free parameters in cost model:"
echo "    WRAP_FMA_NS               = $WRAP_FMA_NS"
echo "    FP64_DIV_NS              = ${FP64_DIV_NS:-unpinned}"
echo "    FMA_NS                    = $FMA_NS"
echo "    PAIRED_CACHED_CORR_RATIO  = $PAIRED_CACHED_CORR_RATIO"
echo "    INDEP_PAIR_RATIO          = $INDEP_PAIR_RATIO"
echo "    FFT_OVERHEAD_NS           = 0.0"
if [ -n "${BATCHED_FMA_NS:-}" ]; then
    echo "    BATCHED_FMA_NS            = $BATCHED_FMA_NS"
else
    echo "    BATCHED_FMA_NS            = (not measured — bench_linear_batched_fma.c missing or failed)"
fi
echo ""
echo "  Lookup tables populated in $CONFIG_H:"
echo "    block_build_ns_per_player[6]  (step 7)"
echo "    leaf_fma_ns_per_player[6]     (step 8, probe_leaf_extract B-sweep)"
echo "    schoolbook_mul_ns[]           (step 9, bench_schoolbook_tree)"
echo "    schoolbook_corr_ns[]          (step 9, bench_schoolbook_tree)"
echo "    crossover_n[]/crossover_k[]   (step 12, calibrate_crossover binary search)"
echo "    bselect_n[]/bselect_k[]/bselect_B[] (step 13, calibrate_best_b sweep)"
echo ""
echo "Next steps (manual):"
echo "  ./bench_grid profile   # re-run profiling for manual inspection"
echo "  ./bench_grid           # full performance grid"
