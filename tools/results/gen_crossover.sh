#!/usr/bin/env bash
# gen_crossover.sh: run bench_grid crossover sweep and save dated output.
#
# Usage:  ./tools/results/gen_crossover.sh <device>
# Example: ./tools/results/gen_crossover.sh m3_pro
#
# Output: results/crossover_<device>_YYYYMMDD_serial.txt
#         results/crossover_<device>_YYYYMMDD_parallel.txt
# Both modes are produced on purpose; see the comment above the run step.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <device>" >&2
    echo "  e.g. $0 m3_pro" >&2
    exit 1
fi

DEVICE="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

MAKE_DEVICE="$DEVICE"
BINARY="./bench_grid"

# Build if missing; if present, trust it (user can rebuild manually if needed).
if [ ! -x "$BINARY" ]; then
    echo "Building bench_grid for DEVICE=$MAKE_DEVICE ..."
    make DEVICE="$MAKE_DEVICE"
fi

mkdir -p results

DATE_SUFFIX="$(date +%Y%m%d)"
SERIAL_OUT="results/crossover_${DEVICE}_${DATE_SUFFIX}_serial.txt"
PARALLEL_OUT="results/crossover_${DEVICE}_${DATE_SUFFIX}_parallel.txt"

# Emit BOTH thread modes. They answer two different questions and the project
# wants both:
#
#  * serial   -- the dispatch tables this sweep checks (crossover_n[]/
#                crossover_k[], from tools/calibrate_crossover.c) are measured
#                single-threaded, so the serial sweep validates them in their
#                own calibration regime.
#  * parallel -- VERDICTS.md V15 records that those serial-calibrated tables are
#                deliberately reused for parallel execution, a known possible
#                suboptimality, and explicitly asks for measured *parallel*
#                dispatch accuracy as the artifact that tests whether the
#                heuristic holds.
#
# Neither mode alone answers both questions above, so pin both explicitly
# rather than inheriting whatever build and thread count (e.g. from
# refresh_all.sh exporting OMP_NUM_THREADS for the whole pipeline) happen
# to be in scope.
if command -v nproc &>/dev/null; then
    NCPU="$(nproc)"
else
    NCPU="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi
PAR_THREADS="${OMP_NUM_THREADS:-$NCPU}"
[ "$PAR_THREADS" = "1" ] && PAR_THREADS="$NCPU"

echo "Running bench_grid crossover for $DEVICE (single-threaded) ..."
OMP_NUM_THREADS=1 "$BINARY" crossover > "$SERIAL_OUT" 2>&1

echo "Running bench_grid crossover for $DEVICE (parallel, $PAR_THREADS threads) ..."
OMP_NUM_THREADS="$PAR_THREADS" "$BINARY" crossover > "$PARALLEL_OUT" 2>&1

# Each output's first line records the mode bench_grid actually ran in
# ("OpenMP disabled (serial mode)" / "OpenMP enabled: N threads"), so a
# serial-only build is self-documenting rather than silently mislabeled.
rc=0
for f in "$SERIAL_OUT" "$PARALLEL_OUT"; do
    if [ -s "$f" ]; then
        echo "Saved $f ($(wc -l < "$f") lines) -- $(head -1 "$f")"
    else
        echo "ERROR: empty output, crossover failed? ($f)" >&2
        rc=1
    fi
done
exit $rc
