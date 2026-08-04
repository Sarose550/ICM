#!/usr/bin/env bash
#
# sweep_best_b.sh: grid driver around tools/validate_best_b.c
#
# Sweeps the same (n,k) grid as bench/bench.c's performance section,
# calls the single-point oracle once per cell, and assembles the results
# into a dated CSV + markdown summary.  Resumable: skips cells already
# present in the CSV unless --force is passed.
#
# Usage:
#   DEVICE=m3_pro ./tools/sweep_best_b.sh
#   ./tools/sweep_best_b.sh --device m3_pro
#   ./tools/sweep_best_b.sh --device m3_pro --quick
#   ./tools/sweep_best_b.sh --device m3_pro --csv /tmp/b.csv --summary /tmp/b.md
#   ./tools/sweep_best_b.sh --device m3_pro --force
#
# Flags:
#   --device NAME    Device subdirectory under devices/ (required).
#   --quick          Use the bench_grid quick subset (n=256,1024,4096,8192).
#   --force          Re-measure every cell even if already in the CSV.
#   --csv PATH       Raw CSV output path (default: results/b_optimal_sweep_<device>_<date>.csv).
#   --summary PATH   Markdown summary path (default: results/b_optimal_sweep_<device>_<date>.md).
#   --help           Print this message.

set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────

die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARN: $*" >&2; }
info() { echo "[sweep] $*" >&2; }

usage() {
    sed -n '2,23p' "$0"
    exit 0
}

# ── arg parsing ──────────────────────────────────────────────────────

DEVICE="${DEVICE:-}"
QUICK=0
FORCE=0
CSV_PATH=""
SUMMARY_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --quick)  QUICK=1; shift ;;
        --force)  FORCE=1; shift ;;
        --csv)    CSV_PATH="$2"; shift 2 ;;
        --summary) SUMMARY_PATH="$2"; shift 2 ;;
        --help)   usage ;;
        -*)       die "Unknown flag: $1" ;;
        *)        die "Unexpected argument: $1" ;;
    esac
done

if [[ -z "$DEVICE" ]]; then
    die "DEVICE is required. Set DEVICE env var or pass --device. Examples: m3_pro, zen4, generic"
fi

DEVICE_DIR="devices/${DEVICE}"
if [[ ! -d "$DEVICE_DIR" ]]; then
    die "Device directory not found: $DEVICE_DIR"
fi
if [[ ! -f "${DEVICE_DIR}/fft_config.h" ]]; then
    die "Missing fft_config.h in $DEVICE_DIR"
fi

# ── platform detection ───────────────────────────────────────────────

OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
    IS_MACOS=1
    IS_LINUX=0
elif [[ "$OS" == "Linux" ]]; then
    IS_MACOS=0
    IS_LINUX=1
else
    die "Unsupported OS: $OS"
fi

# ── paths ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PROBE_SRC="$PROJECT_DIR/tools/validate_best_b.c"
ICM_SRC="$PROJECT_DIR/src/cpu/icm.c"
BUILD_DIR="$PROJECT_DIR/build"
PROBE_BIN="$BUILD_DIR/validate_best_b"

if [[ ! -f "$PROBE_SRC" ]]; then
    die "Probe source not found: $PROBE_SRC"
fi
if [[ ! -f "$ICM_SRC" ]]; then
    die "ICM source not found: $ICM_SRC"
fi

mkdir -p "$BUILD_DIR"

# ── build probe ──────────────────────────────────────────────────────

if [[ ! -x "$PROBE_BIN" ]]; then
    info "Building probe binary ..."
    INCLUDE_FLAGS="-Isrc/cpu -I${DEVICE_DIR}"

    # Detect Apple Silicon (even under Rosetta) and force arm64.
    CC_PREFIX=""
    if [[ "$IS_MACOS" -eq 1 ]]; then
        INCLUDE_FLAGS="$INCLUDE_FLAGS -I/opt/homebrew/include"
        LINK_FLAGS="-L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate"
        if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null)" == "1" ]]; then
            CC_PREFIX="arch -arm64"
        fi
    else
        INCLUDE_FLAGS="$INCLUDE_FLAGS -I/usr/local/aocl-fftw/include"
        LINK_FLAGS="-L/usr/local/aocl-fftw/lib -Wl,-rpath,/usr/local/aocl-fftw/lib -lfftw3 -lm -ldl -lmvec"
    fi

    info "Compile: ${CC_PREFIX}gcc -O3 -march=native ${INCLUDE_FLAGS} ..."
    ${CC_PREFIX} gcc -O3 -march=native $INCLUDE_FLAGS \
        -o "$PROBE_BIN" "$PROBE_SRC" "$ICM_SRC" \
        $LINK_FLAGS

    if [[ ! -x "$PROBE_BIN" ]]; then
        die "Probe build failed."
    fi
    info "Probe built: $PROBE_BIN"
else
    info "Probe binary exists: $PROBE_BIN"
fi

# ── grid definition (mirrors bench/bench.c) ──────────────────────────

FULL_NS=(64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
QUICK_NS=(256 1024 4096 8192)

if [[ "$QUICK" -eq 1 ]]; then
    NS=("${QUICK_NS[@]}")
else
    NS=("${FULL_NS[@]}")
fi

# k-specs: label, type (fixed/fraction), value
declare -a K_LABELS K_TYPES K_VALS
K_LABELS=("k=10" "k=50" "k=100" "k=n/4" "k=n/2" "k=n")
K_TYPES=("fixed" "fixed" "fixed" "frac" "frac" "frac")
K_VALS=(10 50 100 0.25 0.50 1.0)

# ── generate cell list (deduplicated) ────────────────────────────────

declare -a CELL_N CELL_K

for n in "${NS[@]}"; do
    for i in "${!K_LABELS[@]}"; do
        if [[ "${K_TYPES[$i]}" == "fixed" ]]; then
            k="${K_VALS[$i]}"
        else
            k=$(printf "%.0f" "$(echo "${K_VALS[$i]} * $n" | bc -l)")
        fi
        if (( k > n )); then k=$n; fi
        if (( k < 1 )); then k=1; fi

        # dedup
        dup=0
        for j in "${!CELL_N[@]}"; do
            if [[ "${CELL_N[$j]}" -eq "$n" && "${CELL_K[$j]}" -eq "$k" ]]; then
                dup=1
                break
            fi
        done
        if [[ "$dup" -eq 0 ]]; then
            CELL_N+=("$n")
            CELL_K+=("$k")
        fi
    done
done

TOTAL_CELLS=${#CELL_N[@]}

if [[ "$QUICK" -eq 1 ]]; then
    SKIPPED_NS=()
    for fn in "${FULL_NS[@]}"; do
        found=0
        for qn in "${QUICK_NS[@]}"; do
            if [[ "$fn" -eq "$qn" ]]; then found=1; break; fi
        done
        if [[ "$found" -eq 0 ]]; then SKIPPED_NS+=("$fn"); fi
    done
    info "--quick: using n=${QUICK_NS[*]} (skipped n=${SKIPPED_NS[*]})"
fi

info "Grid: ${TOTAL_CELLS} unique cells across ${#NS[@]} n-values"

# ── default output paths ─────────────────────────────────────────────

RUN_DATE="$(date +%Y-%m-%d)"
HOSTNAME="$(hostname -s 2>/dev/null || echo unknown)"
GIT_COMMIT="$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"

if [[ -z "$CSV_PATH" ]]; then
    CSV_PATH="$PROJECT_DIR/results/b_optimal_sweep_${DEVICE}_${RUN_DATE}.csv"
fi
if [[ -z "$SUMMARY_PATH" ]]; then
    SUMMARY_PATH="$PROJECT_DIR/results/b_optimal_sweep_${DEVICE}_${RUN_DATE}.md"
fi

# ── ensure output dirs ───────────────────────────────────────────────

mkdir -p "$(dirname "$CSV_PATH")" "$(dirname "$SUMMARY_PATH")"

# ── write CSV header if new file ─────────────────────────────────────

if [[ ! -f "$CSV_PATH" ]]; then
    echo "n,k,auto_B,auto_ns_qp,best_B,best_ns_qp,gap_pct" > "$CSV_PATH"
    info "Created CSV: $CSV_PATH"
fi

# ── helper: check if cell already in CSV ─────────────────────────────

cell_done() {
    local n="$1" k="$2"
    if [[ "$FORCE" -eq 1 ]]; then return 1; fi
    grep -q "^${n},${k}," "$CSV_PATH" 2>/dev/null
}

# ── sweep ────────────────────────────────────────────────────────────

DONE=0
SKIPPED=0
FAILED=0

info "Starting sweep: ${TOTAL_CELLS} cells"
info "CSV: $CSV_PATH"
info ""

for idx in "${!CELL_N[@]}"; do
    n="${CELL_N[$idx]}"
    k="${CELL_K[$idx]}"

    if cell_done "$n" "$k"; then
        info "[$((idx+1))/${TOTAL_CELLS}] skip n=$n k=$k (already in CSV)"
        ((SKIPPED++)) || true
        continue
    fi

    info "[$((idx+1))/${TOTAL_CELLS}] measuring n=$n k=$k ..."

    output="$("$PROBE_BIN" "$n" "$k" 2>/dev/null)" || {
        warn "Probe failed for n=$n k=$k (exit code $?)"
        ((FAILED++)) || true
        continue
    }

    if [[ -z "$output" ]]; then
        warn "Probe produced no output for n=$n k=$k"
        ((FAILED++)) || true
        continue
    fi

    echo "$output" >> "$CSV_PATH"
    ((DONE++)) || true
done

info ""
info "Sweep complete: ${DONE} measured, ${SKIPPED} skipped, ${FAILED} failed"

# ── markdown summary ──────────────────────────────────────────────────

info "Generating summary: $SUMMARY_PATH"

# Parse CSV for stats
TOTAL_CSV=0
EXACT=0
WITHIN_3PCT=0
declare -a OFFENDERS

while IFS=',' read -r cn ck cauto cautoms cbest cbestms cgap; do
    [[ "$cn" == "n" ]] && continue   # skip header
    [[ -z "$cn" ]] && continue
    ((TOTAL_CSV++)) || true
    if [[ "$cauto" == "$cbest" ]]; then
        ((EXACT++)) || true
        ((WITHIN_3PCT++)) || true
    else
        gt=$(echo "$cgap <= 3.0" | bc -l 2>/dev/null || echo 0)
        if [[ "$gt" == "1" ]]; then
            ((WITHIN_3PCT++)) || true
        fi
    fi
    OFFENDERS+=("$cgap|$cn|$ck|$cauto|$cbest")
done < "$CSV_PATH"

# Sort offenders descending by gap_pct, take top 10
SORTED_OFFENDERS=$(printf '%s\n' "${OFFENDERS[@]}" | sort -t'|' -k1 -rn | head -10)

OFFENDER_ROWS=""
while IFS='|' read -r gap cn ck cauto cbest; do
    [[ -z "$gap" ]] && continue
    OFFENDER_ROWS+="| $cn | $ck | $cauto | $cbest | ${gap}% |"$'\n'
done <<< "$SORTED_OFFENDERS"

if [[ -z "$OFFENDER_ROWS" ]]; then
    OFFENDER_ROWS="| (none) | | | | |"$'\n'
fi

NOW_ISO="$(date -u +"%Y-%m-%d %H:%M:%S UTC")"

cat > "$SUMMARY_PATH" <<EOF
# B-Optimality Sweep Report

**Device:** ${DEVICE}
**Date:** ${NOW_ISO}
**Hostname:** ${HOSTNAME}
**Git commit:** ${GIT_COMMIT}
**Mode:** $([[ "$QUICK" -eq 1 ]] && echo "quick (subset)" || echo "full grid")

## Summary

| Metric | Count |
|---|---|
| Total cells | ${TOTAL_CSV} |
| auto_B == best_B (exact) | ${EXACT} |
| Within 3% gap (effectively correct) | ${WITHIN_3PCT} |
| Cells measured this run | ${DONE} |
| Cells skipped (already in CSV) | ${SKIPPED} |
| Cells failed | ${FAILED} |

## Worst Offenders (by gap_pct)

| n | k | auto_B | best_B | gap_pct |
|---|---|---|---|---|
${OFFENDER_ROWS}

## Raw Data

\`${CSV_PATH}\`

## Methodology

Probe: \`tools/validate_best_b.c\` (single-point oracle).
Q=256, srand(42), payout[m]=n-m, S[i]=100+9900*rand()/RAND_MAX.
Median-of-7 for final head-to-head timing.
Discovery: 1 rep per candidate to rank, then 2 more reps on top-2 if within 3%, median of 3.

Grid: mirrors \`bench/bench.c\` performance grid.
EOF

info "Summary written: $SUMMARY_PATH"
info "Done."
