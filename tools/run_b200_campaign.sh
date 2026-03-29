#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${1:-results_b200}"
Q="${2:-256}"
FAST_FLAG="${3:-}"
DEVICE_NAME="${DEVICE_NAME:-zen4}"
CPU_CFLAGS="${CPU_CFLAGS:-}"
RUN_FINAL_HEATMAP="${RUN_FINAL_HEATMAP:-0}"
RUN_PLOTS="${RUN_PLOTS:-0}"
TIMEOUT_VERIFY_S="${TIMEOUT_VERIFY_S:-0}"
TIMEOUT_VERIFY_EXT_S="${TIMEOUT_VERIFY_EXT_S:-0}"
TIMEOUT_CALIB_S="${TIMEOUT_CALIB_S:-0}"
TIMEOUT_PLANNER_S="${TIMEOUT_PLANNER_S:-0}"
TIMEOUT_HEATMAP_S="${TIMEOUT_HEATMAP_S:-0}"
TIMEOUT_LIMIT_S="${TIMEOUT_LIMIT_S:-0}"
MATHDX_VERSION="${MATHDX_VERSION:-25.12.1}"
MATHDX_CUDA_FAMILY="${MATHDX_CUDA_FAMILY:-cuda13}"
MATHDX_BASE_DIR="${MATHDX_BASE_DIR:-/opt/nvidia/mathdx}"
MATHDX_SERIES="${MATHDX_VERSION%.*}"

MATHDX_ARCHIVE="nvidia-mathdx-${MATHDX_VERSION}-${MATHDX_CUDA_FAMILY}.tar.gz"
MATHDX_URL="https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/${MATHDX_CUDA_FAMILY}/${MATHDX_ARCHIVE}"
MATHDX_EXTRACT_DIR="${MATHDX_BASE_DIR}/nvidia-mathdx-${MATHDX_VERSION}-${MATHDX_CUDA_FAMILY}"
MATHDX_INCLUDE_DIR="${MATHDX_EXTRACT_DIR}/nvidia/mathdx/${MATHDX_SERIES}/include"

if [[ "${FAST_FLAG}" == "--fast" ]]; then
  [[ "${TIMEOUT_VERIFY_S}" == "0" ]] && TIMEOUT_VERIFY_S=180
  [[ "${TIMEOUT_VERIFY_EXT_S}" == "0" ]] && TIMEOUT_VERIFY_EXT_S=300
  [[ "${TIMEOUT_CALIB_S}" == "0" ]] && TIMEOUT_CALIB_S=420
  [[ "${TIMEOUT_PLANNER_S}" == "0" ]] && TIMEOUT_PLANNER_S=120
  [[ "${TIMEOUT_HEATMAP_S}" == "0" ]] && TIMEOUT_HEATMAP_S=180
  [[ "${TIMEOUT_LIMIT_S}" == "0" ]] && TIMEOUT_LIMIT_S=180
  : "${ICM_GPU_PLANNER_MAX_SECONDS:=90}"
  : "${ICM_GPU_PLANNER_MAX_CASES:=2}"
  : "${ICM_GPU_PLANNER_MAX_B_TRIALS:=2}"
  : "${ICM_GPU_PLANNER_CASE_BUDGET_MS:=5000}"
fi

run_with_timeout() {
  local sec="$1"
  shift
  if [[ "${sec}" -gt 0 ]]; then
    timeout "${sec}s" "$@"
  else
    "$@"
  fi
}

ensure_cufftdx() {
  mkdir -p "${MATHDX_BASE_DIR}"
  if [[ ! -f "${MATHDX_INCLUDE_DIR}/cufftdx.hpp" ]]; then
    echo "cuFFTDx headers not found. Installing MathDx ${MATHDX_VERSION} (${MATHDX_CUDA_FAMILY})..."
    (
      cd "${MATHDX_BASE_DIR}"
      if [[ ! -f "${MATHDX_ARCHIVE}" ]]; then
        curl -L -o "${MATHDX_ARCHIVE}" "${MATHDX_URL}"
      fi
      tar -xzf "${MATHDX_ARCHIVE}"
    )
  fi
  if [[ ! -f "${MATHDX_INCLUDE_DIR}/cufftdx.hpp" ]]; then
    echo "ERROR: cuFFTDx install failed: ${MATHDX_INCLUDE_DIR}/cufftdx.hpp not found"
    exit 1
  fi
  export CUFFTDX_INC="-I${MATHDX_INCLUDE_DIR}"
  echo "Using cuFFTDx includes at: ${MATHDX_INCLUDE_DIR}"
}

if [[ -z "${CPU_CFLAGS}" ]]; then
  if printf 'int main(){return 0;}\n' | gcc -x c - -march=znver4 -o /tmp/icm_znver4_probe >/dev/null 2>&1; then
    rm -f /tmp/icm_znver4_probe
  else
    CPU_CFLAGS="-O3 -march=native -Wall -Wno-unused-variable -Wno-unused-function"
    echo "GCC does not support -march=znver4 on this host. Falling back to -march=native."
  fi
fi

MAKE_ARGS=(DEVICE="${DEVICE_NAME}")
if [[ -n "${CPU_CFLAGS}" ]]; then
  MAKE_ARGS+=(CFLAGS="${CPU_CFLAGS}")
fi

mkdir -p "${OUT_DIR}"

ensure_cufftdx

echo "[1/8] Building GPU tools..."
make "${MAKE_ARGS[@]}" bench_gpu_fused calibrate_gpu heatmap_gpu push_limit_gpu validate_planner_gpu

echo "[2/8] Running GPU correctness verify..."
GPU_BENCH_BIN="./bench_gpu"
if [[ ! -x "${GPU_BENCH_BIN}" && -x "./bench_gpu_fused" ]]; then
  GPU_BENCH_BIN="./bench_gpu_fused"
fi
if ! run_with_timeout "${TIMEOUT_VERIFY_S}" "${GPU_BENCH_BIN}" verify | tee "${OUT_DIR}/verify.log"; then
  echo "WARNING: GPU verify failed; continuing campaign for profiling/calibration artifacts."
fi
if ! run_with_timeout "${TIMEOUT_VERIFY_EXT_S}" "${GPU_BENCH_BIN}" verify_ext | tee "${OUT_DIR}/verify_ext.log"; then
  echo "WARNING: GPU extended verify failed; continuing campaign for profiling/calibration artifacts."
fi

echo "[3/8] Calibrating GPU model..."
run_with_timeout "${TIMEOUT_CALIB_S}" ./calibrate_gpu "devices/b200/gpu_fft_config.h" 131072 ${FAST_FLAG} | tee "${OUT_DIR}/calibrate.log"

echo "[4/8] Rebuilding with updated calibration..."
make clean
make "${MAKE_ARGS[@]}" bench_gpu_fused heatmap_gpu push_limit_gpu validate_planner_gpu

echo "[5/8] Validating planner against anchor sweeps..."
run_with_timeout "${TIMEOUT_PLANNER_S}" env \
  ICM_GPU_PLANNER_MAX_SECONDS="${ICM_GPU_PLANNER_MAX_SECONDS:-0}" \
  ICM_GPU_PLANNER_MAX_CASES="${ICM_GPU_PLANNER_MAX_CASES:-0}" \
  ICM_GPU_PLANNER_MAX_B_TRIALS="${ICM_GPU_PLANNER_MAX_B_TRIALS:-0}" \
  ICM_GPU_PLANNER_CASE_BUDGET_MS="${ICM_GPU_PLANNER_CASE_BUDGET_MS:-0}" \
  ./validate_planner_gpu "${OUT_DIR}/planner_validation.csv" "${Q}" ${FAST_FLAG} | tee "${OUT_DIR}/planner_validation.log"

if [[ "${RUN_FINAL_HEATMAP}" == "1" ]]; then
  echo "[6/8] Generating full (n,k) heatmap CSV..."
  run_with_timeout "${TIMEOUT_HEATMAP_S}" ./heatmap_gpu "${OUT_DIR}/gpu_heatmap.csv" "${Q}" ${FAST_FLAG} | tee "${OUT_DIR}/heatmap.log"
else
  echo "[6/8] Skipping heatmap generation (RUN_FINAL_HEATMAP=${RUN_FINAL_HEATMAP})."
fi

echo "[7/8] Pushing k=n physical limit..."
run_with_timeout "${TIMEOUT_LIMIT_S}" ./push_limit_gpu "${OUT_DIR}/gpu_limit_frontier.csv" "${Q}" ${FAST_FLAG} | tee "${OUT_DIR}/limit.log"

if [[ "${RUN_FINAL_HEATMAP}" == "1" && "${RUN_PLOTS}" == "1" ]]; then
  echo "[8/8] Plotting heatmaps..."
  python3 tools/plot_heatmap.py "${OUT_DIR}/gpu_heatmap.csv" "${OUT_DIR}/gpu_heatmap"
else
  echo "[8/8] Skipping plotting (RUN_FINAL_HEATMAP=${RUN_FINAL_HEATMAP}, RUN_PLOTS=${RUN_PLOTS})."
fi

OUT_DIR_ENV="${OUT_DIR}" python3 - <<'PY'
import csv
import math
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR_ENV"])
csv_path = out_dir / "gpu_limit_frontier.csv"
best = None
if csv_path.exists():
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                n = int(row["n"])
                t = float(row["time_ms"])
                if not math.isfinite(t) or t > 1000:
                    continue
                if best is None or n > best["n"] or (n == best["n"] and t < best["time_ms"]):
                    best = {
                        "n": n,
                        "time_ms": t,
                        "B": int(row["B"]),
                        "M": int(row["M"]),
                        "T": int(row["T"]),
                        "peak_vram_mb": float(row["peak_vram_mb"]),
                    }
            except Exception:
                pass

report = out_dir / "headline.md"
with report.open("w") as f:
    f.write("# B200 Campaign Headline\n\n")
    if best is None:
        f.write("No k=n configuration at or below 1 second was found in this run.\n")
    else:
        f.write(
            f"Max n at k=n under 1s: n={best['n']} "
            f"(time={best['time_ms']:.2f} ms, B={best['B']}, M={best['M']}, "
            f"T={best['T']}, peak_vram={best['peak_vram_mb']:.1f} MB)\n"
        )
print(f"Wrote {report}")
PY

echo "Campaign completed. Outputs in ${OUT_DIR}/"
