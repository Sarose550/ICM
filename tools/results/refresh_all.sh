#!/usr/bin/env bash
# refresh_all.sh — full results refresh for a given device.
#
# Usage:  ./tools/results/refresh_all.sh --device <device>
# Example: ./tools/results/refresh_all.sh --device m3_pro
#
# Rebuilds binaries, regenerates all raw data files (bench_grid, contour CSVs,
# crossover, subset-speed), then regenerates all publication plots.  This is
# the one-command entrypoint that `make results-refresh` delegates to.

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

# Number of cores for parallel runs
if command -v nproc &>/dev/null; then
    NCPU="$(nproc)"
else
    NCPU="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NCPU}"

echo "=== refresh_all.sh: DEVICE=$DEVICE  FILE_TAG=$FILE_TAG  DATE=$DATE_SUFFIX ==="

# ── 1. Serial bench_grid ──────────────────────────────────────

echo "--- [1/10] Building serial bench_grid ---"
make DEVICE="$DEVICE"

echo "--- [2/10] Running serial bench_grid ---"
./bench_grid > "$RESULTS_DIR/bench_grid_${FILE_TAG}_serial_${DATE_SUFFIX}.txt"

# ── 2. Parallel bench_grid ────────────────────────────────────

echo "--- [3/10] Building parallel bench_grid ---"
make parallel DEVICE="$DEVICE"

echo "--- [4/10] Running parallel bench_grid ---"
./bench_grid > "$RESULTS_DIR/bench_grid_${FILE_TAG}_parallel_${DATE_SUFFIX}.txt"

# ── 3. Serial contour ─────────────────────────────────────────

echo "--- [5/10] Building contour_1s ---"
make contour_1s DEVICE="$DEVICE"

echo "--- [6/10] Running serial contour sweep ---"
./contour_1s --contour > "$RESULTS_DIR/contour_${FILE_TAG}_serial_q256_${DATE_SUFFIX}.csv"

# ── 4. Parallel contour ───────────────────────────────────────

echo "--- [7/10] Building contour_1s_par ---"
make contour_1s_par DEVICE="$DEVICE"

echo "--- [8/10] Running parallel contour sweep ---"
./contour_1s_par --contour > "$RESULTS_DIR/contour_${FILE_TAG}_parallel_q256_${DATE_SUFFIX}.csv"

# ── 5. Crossover ──────────────────────────────────────────────

echo "--- [9/10] Running crossover sweep ---"
bash "$SCRIPT_DIR/gen_crossover.sh" "$DEVICE"

# ── 6. Subset-speed ───────────────────────────────────────────

echo "--- [10/10] Running subset-speed benchmark ---"
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
    python3 "$PLOT_DIR/plot_gpu_contour_fig.py" || echo "    (GPU plots skipped — no heatmap data)"
fi

echo "  -> accuracy_convergence"
python3 "$PLOT_DIR/plot_accuracy_fig.py"

echo "=== refresh_all.sh complete ==="
