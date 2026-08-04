#!/usr/bin/env bash
# gen_subset_speed.sh — run bench_grid subset-speed benchmark and save dated output.
#
# Usage:  ./tools/results/gen_subset_speed.sh <device>
# Example: ./tools/results/gen_subset_speed.sh m3_pro
#
# Output: results/subset_speed_<device>_YYYYMMDD.txt

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

# Build if missing; if present, trust it.
if [ ! -x "$BINARY" ]; then
    echo "Building bench_grid for DEVICE=$MAKE_DEVICE ..."
    make DEVICE="$MAKE_DEVICE"
fi

mkdir -p results

DATE_SUFFIX="$(date +%Y%m%d)"
OUTFILE="results/subset_speed_${DEVICE}_${DATE_SUFFIX}.txt"

# Subset-speed is reported single-threaded, and must be MEASURED that way.
#
# refresh_all.sh exports OMP_NUM_THREADS=$NCPU for the whole pipeline, and by
# the time this step runs, ./bench_grid on disk is the parallel build. Left
# alone, this step therefore produced a 12-thread measurement while the paper's
# subset table (and its caption) says "single-threaded" -- target-locality
# pruning shows a real 1.1-1.5x win serially but flattens to ~1.00x across the
# board in parallel, so the generated file silently stopped supporting the
# table it exists to back up. Pin one thread here rather than depending on
# whichever build happens to be on disk.
echo "Running bench_grid subset-speed for $DEVICE (single-threaded) ..."
OMP_NUM_THREADS=1 "$BINARY" subset-speed > "$OUTFILE" 2>&1

if [ -s "$OUTFILE" ]; then
    echo "Saved $OUTFILE ($(wc -l < "$OUTFILE") lines)"
else
    echo "ERROR: empty output — subset-speed failed?" >&2
    exit 1
fi
