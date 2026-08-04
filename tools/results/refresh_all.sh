#!/usr/bin/env bash
# refresh_all.sh: full results refresh for a given device.
#
# Usage:  ./tools/results/refresh_all.sh --device <device>
# Example: ./tools/results/refresh_all.sh --device m3_pro
#
# Rebuilds binaries, regenerates all raw data files (bench_grid, the real
# k=n 1-second threshold via binary search, contour CSVs, crossover,
# subset-speed), then regenerates all publication plots.  This is the
# one-command entrypoint that `make results-refresh` delegates to.
#
# The threshold step exists because it's easy to leave out otherwise: a
# "1-second threshold at k=n" number is easy to *approximate* by
# interpolating between grid points bench_grid already produces, but that
# interpolation has shipped as a real (wrong) headline number more than
# once. Always run the actual search instead of reaching for that shortcut.

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────

DEVICE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --device)
            DEVICE="$2"; shift 2 ;;
        *)
            echo "Usage: $0 --device <device>" >&2
            echo "  e.g. $0 --device m3_pro" >&2
            exit 1 ;;
    esac
done

if [ -z "$DEVICE" ]; then
    echo "ERROR: --device is required" >&2
    exit 1
fi

# ── Paths ──────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

# File tag: m3_pro → m3pro (matches DEVICE_CONFIGS convention)
if [ "$DEVICE" = "m3_pro" ]; then
    FILE_TAG="m3pro"
else
    FILE_TAG="$DEVICE"
fi

DATE_SUFFIX="$(date +%Y%m%d)"
RESULTS_DIR="$ROOT/results"
mkdir -p "$RESULTS_DIR"

# Number of threads for parallel runs.
#
# Use PHYSICAL cores, not logical. This workload is FPU/vector-port bound, so
# SMT siblings add no real throughput (CLAUDE.md states this explicitly for
# Zen4: "Always use OMP_NUM_THREADS=16 (one per physical core), not 32").
# `nproc` reports 32 on a 16-core 7950X, so the previous default silently
# produced 32-thread parallel numbers that are not comparable to every
# published 16-thread figure -- and nothing in the output said so.
NCPU=""
if command -v lscpu &>/dev/null; then
    # sockets * cores-per-socket, i.e. physical cores, SMT excluded
    NCPU="$(lscpu 2>/dev/null | awk -F: '
        /^Core\(s\) per socket/ {c=$2}
        /^Socket\(s\)/          {s=$2}
        END {gsub(/ /,"",c); gsub(/ /,"",s); if (c>0 && s>0) print c*s}')"
fi
if [ -z "$NCPU" ]; then
    NCPU="$(sysctl -n hw.physicalcpu 2>/dev/null || true)"
fi
if [ -z "$NCPU" ]; then
    NCPU="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NCPU}"
echo "=== parallel runs will use OMP_NUM_THREADS=$OMP_NUM_THREADS (physical cores) ==="

echo "=== refresh_all.sh: DEVICE=$DEVICE  FILE_TAG=$FILE_TAG  DATE=$DATE_SUFFIX ==="

# ── 1. Serial bench_grid ──────────────────────────────────────

echo "--- [1/12] Building serial bench_grid ---"
make DEVICE="$DEVICE"

echo "--- [2/12] Running serial bench_grid ---"
./bench_grid > "$RESULTS_DIR/bench_grid_${FILE_TAG}_serial.txt"

# Real k=n 1-second threshold (binary search), serial. This is NOT
# derivable from the grid above by interpolation -- past sessions
# repeatedly substituted an interpolated grid estimate here because this
# step didn't exist yet, which shipped a materially wrong number more
# than once. Slow (a handful of large-n probes, several minutes) but
# always run so a real threshold number is never missing.
echo "--- [2b/12] Running serial 1-second threshold search ---"
./bench_grid threshold > "$RESULTS_DIR/threshold_${FILE_TAG}_serial.txt"

# ── 2. Parallel bench_grid ────────────────────────────────────

echo "--- [3/12] Building parallel bench_grid ---"
make parallel DEVICE="$DEVICE"

echo "--- [4/12] Running parallel bench_grid ---"
./bench_grid > "$RESULTS_DIR/bench_grid_${FILE_TAG}_parallel.txt"

echo "--- [4b/12] Running parallel 1-second threshold search ---"
./bench_grid threshold > "$RESULTS_DIR/threshold_${FILE_TAG}_parallel.txt"

# ── 3. Serial contour ─────────────────────────────────────────

echo "--- [5/12] Building contour_1s ---"
make contour_1s DEVICE="$DEVICE"

echo "--- [6/12] Running serial contour sweep ---"
./contour_1s --contour > "$RESULTS_DIR/contour_${FILE_TAG}_serial_q256.csv"

# ── 4. Parallel contour ───────────────────────────────────────

echo "--- [7/12] Building contour_1s_par ---"
make contour_1s_par DEVICE="$DEVICE"

echo "--- [8/12] Running parallel contour sweep ---"
./contour_1s_par --contour > "$RESULTS_DIR/contour_${FILE_TAG}_parallel_q256.csv"

# ── 5. Crossover ──────────────────────────────────────────────

echo "--- [9/12] Running crossover sweep ---"
bash "$SCRIPT_DIR/gen_crossover.sh" "$DEVICE"

# ── 6. Subset-speed ───────────────────────────────────────────

echo "--- [10/12] Running subset-speed benchmark ---"
bash "$SCRIPT_DIR/gen_subset_speed.sh" "$DEVICE"

# ── 7. Plots ──────────────────────────────────────────────────

echo "=== Generating plots ==="

PLOT_DIR="$SCRIPT_DIR"

echo "  -> contour_1s"
python3 "$PLOT_DIR/plot_contour_fig.py" --device "$DEVICE"

echo "  -> parallel_speedup"
python3 "$PLOT_DIR/plot_speedup_fig.py" --device "$DEVICE"

echo "  -> engine_dispatch"
python3 "$PLOT_DIR/plot_dispatch_fig.py" --device "$DEVICE"

echo "  -> runtime_vs_n (CPU)"
python3 "$PLOT_DIR/plot_runtime_fig.py" --device "$DEVICE"

# GPU plots: only if GPU device (has gpu_fft_config.h) OR GPU heatmap data exists.
# Convention from calibrate_adaptive.py: GPU devices have devices/<dev>/gpu_fft_config.h.
if [ -f "devices/$DEVICE/gpu_fft_config.h" ] || [ -f "devices/b200/gpu_fft_config.h" ]; then
    echo "  -> GPU contour + GPU runtime_vs_n"
    python3 "$PLOT_DIR/plot_gpu_contour_fig.py" || echo "    (GPU plots skipped, no heatmap data)"
fi

echo "  -> accuracy_convergence"
python3 "$PLOT_DIR/plot_accuracy_fig.py"

echo "=== refresh_all.sh complete ==="
