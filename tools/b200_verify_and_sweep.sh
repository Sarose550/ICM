#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# b200_verify_and_sweep.sh — B200 verification & sweep orchestration
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs a gated, 5-stage verification-and-sweep pipeline on a remote B200
# instance.  Each stage gates on the previous one; first failure stops the
# whole script with a non-zero exit and a clear [GATE N] FAIL message.
#
# The script does NOT rent or destroy the cloud instance — that remains a
# manual, supervisor-only step.  It assumes an already-running, SSH-reachable
# box.
#
# ── Usage ────────────────────────────────────────────────────────────────────
#
#   B200_HOST=<host> B200_PORT=<port> ./tools/b200_verify_and_sweep.sh
#   ./tools/b200_verify_and_sweep.sh <host> <port>
#
#   Positional args take precedence over env vars.
#
# ── Optional env vars ────────────────────────────────────────────────────────
#
#   REPO_SOURCE    How to get the repo onto the remote box.
#                  - If it starts with "git@" or "https://", treated as a git
#                    clone URL (clones into /workspace/icm, checks out the
#                    current branch).
#                  - Otherwise treated as a local directory to rsync to the
#                    remote.  Default: the directory containing this script's
#                    parent (i.e. the repo root).
#   REPO_BRANCH    Branch to check out on the remote (default: results-gpu-section).
#   VERIFY_MIN_PASS  Minimum expected "PASS" lines from bench_gpu_fused verify
#                    (default: 36, the expected count from a known-good build).
#   CUFFTDX_GLOB   Glob pattern for detecting the cuFFTDx include path on the
#                  remote box (default: /usr/local/lib/python*/dist-packages/nvidia/mathdx/include).
#   REMOTE_WORKDIR Absolute path for the repo on the remote box (default: /workspace/icm).
#   MAKE_JOBS       -j flag for make (default: $(nproc)).
#   HEATMAP_CSV     Output filename for heatmap_gpu on the remote (default: gpu_heatmap.csv).
#   PUSH_LIMIT_CSV  Output filename for push_limit_gpu on the remote (default: gpu_limit_frontier.csv).
#   HEATMAP_ARGS    Extra args for heatmap_gpu (e.g. --fast).
#   PUSH_LIMIT_ARGS Extra args for push_limit_gpu (e.g. --fast).
#
# ── Output ───────────────────────────────────────────────────────────────────
#
#   All stage logs and pulled-back artifacts land in:
#     results/b200_verify_<ISO8601_timestamp>/
#
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────────────

cleanup() {
    # Best-effort cleanup of any local temp files.  Remote side is not touched
    # (the instance stays up for the supervisor to inspect or destroy).
    if [ -n "${LOCAL_TMPDIR:-}" ] && [ -d "$LOCAL_TMPDIR" ]; then
        rm -rf "$LOCAL_TMPDIR"
    fi
}
trap cleanup EXIT

die() {
    echo "FATAL: $*" >&2
    exit 1
}

gate_pass() {
    local gate="$1"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "[GATE $gate] PASS"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
}

gate_fail() {
    local gate="$1"
    shift
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "[GATE $gate] FAIL — $*"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    exit 1
}

remote() {
    # Run a command on the remote box, echoing it first.
    # The echo goes to stderr so it never contaminates $(remote ...) command
    # substitution -- only the remote command's real stdout is captured.
    echo "  [remote] $*" >&2
    ssh -p "$B200_PORT" "$B200_HOST" "$@"
}

# ── arg parsing ──────────────────────────────────────────────────────────────

B200_HOST="${1:-${B200_HOST:-}}"
B200_PORT="${2:-${B200_PORT:-22}}"

if [ -z "$B200_HOST" ]; then
    die "B200_HOST not set.  Provide it as a positional arg or B200_HOST env var."
fi

REPO_SOURCE="${REPO_SOURCE:-$PWD}"
REPO_BRANCH="${REPO_BRANCH:-results-gpu-section}"
VERIFY_MIN_PASS="${VERIFY_MIN_PASS:-36}"
CUFFTDX_GLOB="${CUFFTDX_GLOB:-/usr/local/lib/python*/dist-packages/nvidia/mathdx/include}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/workspace/icm}"
MAKE_JOBS="${MAKE_JOBS:-$(nproc 2>/dev/null || echo 4)}"
HEATMAP_CSV="${HEATMAP_CSV:-gpu_heatmap.csv}"
PUSH_LIMIT_CSV="${PUSH_LIMIT_CSV:-gpu_limit_frontier.csv}"
HEATMAP_ARGS="${HEATMAP_ARGS:-}"
PUSH_LIMIT_ARGS="${PUSH_LIMIT_ARGS:-}"

# Timestamp for the local results directory (captured once at start).
START_TIME="$(date -u +%Y%m%dT%H%M%SZ)"
START_EPOCH="$(date +%s)"
LOCAL_RESULTS="results/b200_verify_${START_TIME}"

echo "═══════════════════════════════════════════════════════════════"
echo "B200 VERIFY & SWEEP — $START_TIME"
echo "  host:     $B200_HOST"
echo "  port:     $B200_PORT"
echo "  remote:   $REMOTE_WORKDIR"
echo "  branch:   $REPO_BRANCH"
echo "  results:  $LOCAL_RESULTS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── local prep ───────────────────────────────────────────────────────────────

mkdir -p "$LOCAL_RESULTS"
LOCAL_TMPDIR="$(mktemp -d)"

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 0 — remote setup
# ──────────────────────────────────────────────────────────────────────────────

echo "=== STAGE 0: remote setup ==="

# 0a. Install system + pip packages
echo "--- 0a: install packages ---"
remote "sudo apt-get update -qq && sudo apt-get install -y -qq libfftw3-dev" || die "apt-get install failed"
remote "pip install nvidia-mathdx --break-system-packages 2>&1 || pip install nvidia-mathdx 2>&1" || die "pip install nvidia-mathdx failed"

# 0b. Get the repo onto the remote box
echo "--- 0b: deploy repo ---"
if echo "$REPO_SOURCE" | grep -Eq '^(git@|https://)'; then
    # Git clone from URL
    remote "if [ -d '$REMOTE_WORKDIR/.git' ]; then cd '$REMOTE_WORKDIR' && git fetch origin && git checkout '$REPO_BRANCH' && git pull origin '$REPO_BRANCH'; else git clone --branch '$REPO_BRANCH' '$REPO_SOURCE' '$REMOTE_WORKDIR'; fi" || die "git clone/pull failed"
else
    # Rsync from local path
    if [ ! -d "$REPO_SOURCE/.git" ]; then
        die "REPO_SOURCE '$REPO_SOURCE' is not a git repo (local directory not found or no .git)"
    fi
    remote "mkdir -p '$REMOTE_WORKDIR'" || die "mkdir remote workdir failed"
    echo "  rsync $REPO_SOURCE/ -> $B200_HOST:$REMOTE_WORKDIR/ (git-tracked + untracked-but-not-ignored files only)"
    # Transfer exactly what git knows about (tracked + untracked-but-unignored
    # files, e.g. not-yet-committed work) -- never a raw
    # directory rsync.  Local scratch dirs (venvs, old worktrees, prior
    # results) can be 100s of MB of junk that has no business on a build box.
    ( cd "$REPO_SOURCE" && git ls-files --cached --others --exclude-standard -z ) \
        | rsync -avz --from0 --files-from=- -e "ssh -p $B200_PORT" \
            "$REPO_SOURCE/" "$B200_HOST:$REMOTE_WORKDIR/" || die "rsync failed"
fi

# 0c. Detect cuFFTDx include path
echo "--- 0c: detect cuFFTDx include path ---"
CUFFTDX_PATH=$(remote "ls -d $CUFFTDX_GLOB 2>/dev/null | head -1" || echo "")
if [ -z "$CUFFTDX_PATH" ]; then
    die "Could not find cuFFTDx include path with glob: $CUFFTDX_GLOB"
fi
echo "  cuFFTDx include: $CUFFTDX_PATH"
CUFFTDX_INC="-I$CUFFTDX_PATH"

echo "[STAGE 0] complete"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 (Gate 1) — build bench_gpu_fused + verify
# ──────────────────────────────────────────────────────────────────────────────

echo "=== STAGE 1 (Gate 1): build bench_gpu_fused + verify ==="

# Build
echo "--- build bench_gpu_fused ---"
remote "cd '$REMOTE_WORKDIR' && make clean 2>&1" || true  # clean is best-effort
remote "cd '$REMOTE_WORKDIR' && make bench_gpu_fused CUDA_ARCH=sm_100 CUFFTDX_INC='$CUFFTDX_INC' -j$MAKE_JOBS 2>&1" || die "make bench_gpu_fused failed"

# Run verify and capture output
echo "--- run verify ---"
VERIFY_LOG="$LOCAL_RESULTS/stage1_verify.log"
remote "cd '$REMOTE_WORKDIR' && ./bench_gpu_fused verify 2>&1" | tee "$VERIFY_LOG" || true

# Gate check
FAIL_COUNT=$(grep -c "FAIL" "$VERIFY_LOG" 2>/dev/null || true)
PASS_COUNT=$(grep -c "PASS" "$VERIFY_LOG" 2>/dev/null || true)
echo "  FAIL lines: $FAIL_COUNT"
echo "  PASS lines: $PASS_COUNT (need >= $VERIFY_MIN_PASS with 0 FAIL)"

if [ "$FAIL_COUNT" -eq 0 ] && [ "$PASS_COUNT" -ge "$VERIFY_MIN_PASS" ]; then
    gate_pass "1"
else
    gate_fail "1" "FAIL=$FAIL_COUNT PASS=$PASS_COUNT (need 0 FAIL, >=$VERIFY_MIN_PASS PASS)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 (Gate 2) — build & run workspace-repro harness
# ──────────────────────────────────────────────────────────────────────────────

echo "=== STAGE 2 (Gate 2): build & run gpu_ws_repro ==="

# Compile the repro against the object files already built in Stage 1.
# Mirror the bench_gpu_fused build recipe from the Makefile, minus the CPU
# reference object (not needed here).
echo "--- build gpu_ws_repro ---"
CUDA_FLAGS="-O3 -std=c++17 -arch=sm_100"
GPU_INCLUDES="-Isrc/cpu -Idevices/b200"
CUFFTDX_FLAGS="$CUFFTDX_INC -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY"
REPRO_SRC="tools/gpu_ws_repro.cu"
REPRO_OBJ="build/gpu_ws_repro.o"
REPRO_BIN="gpu_ws_repro"

remote "cd '$REMOTE_WORKDIR' && \
    nvcc $CUDA_FLAGS $GPU_INCLUDES -Isrc/gpu $CUFFTDX_FLAGS -dc -o '$REPRO_OBJ' '$REPRO_SRC' 2>&1 && \
    nvcc $CUDA_FLAGS -o '$REPRO_BIN' '$REPRO_OBJ' build/gpu_gpu_kernels_fused.o build/gpu_gpu_plan_fused.o build/gpu_gpu_exec_fused.o build/gpu_gpu_api_fused.o build/gpu_dlink_fused.o -lcufft -lcudart 2>&1" || die "build gpu_ws_repro failed"

# Run
echo "--- run gpu_ws_repro ---"
REPRO_LOG="$LOCAL_RESULTS/stage2_repro.log"
set +e
remote "cd '$REMOTE_WORKDIR' && ./gpu_ws_repro 2>&1" | tee "$REPRO_LOG"
REPRO_RC=${PIPESTATUS[0]}
set -e

if [ "$REPRO_RC" -eq 0 ]; then
    gate_pass "2"
else
    gate_fail "2" "gpu_ws_repro exit code $REPRO_RC (expected 0)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3a — run calibrate_gpu, copy new header into place
# ──────────────────────────────────────────────────────────────────────────────

echo "=== STAGE 3a: calibrate_gpu (workspace calibration) ==="

# calibrate_gpu should already be built by make bench_gpu_fused (same object files).
# Verify it exists; if not, build it.
echo "--- ensure calibrate_gpu is built ---"
remote "cd '$REMOTE_WORKDIR' && test -x ./calibrate_gpu || make calibrate_gpu CUDA_ARCH=sm_100 CUFFTDX_INC='$CUFFTDX_INC' -j$MAKE_JOBS 2>&1" || die "calibrate_gpu not found and build failed"

echo "--- run calibrate_gpu ---"
CALIB_LOG="$LOCAL_RESULTS/stage3a_calibrate.log"
# ICM_GPU_CALIB_WS=1 is already the default per the code (confirmed in
# calibrate_gpu.cu: env_int_clamped("ICM_GPU_CALIB_WS", 1, 0, 1)), set it
# explicitly for clarity.
remote "cd '$REMOTE_WORKDIR' && ICM_GPU_CALIB_WS=1 ./calibrate_gpu devices/b200/gpu_fft_config.h 2>&1" | tee "$CALIB_LOG" || die "calibrate_gpu failed"

echo "--- calibrate_gpu wrote new header to devices/b200/gpu_fft_config.h ---"
# calibrate_gpu writes directly to the path given as its first argument
# (devices/b200/gpu_fft_config.h), which is also its default — no copy needed.

echo "[STAGE 3a] complete"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3b — rebuild bench_gpu_fused against new header, re-verify
# ──────────────────────────────────────────────────────────────────────────────

echo "=== STAGE 3b (Gate 3b): rebuild + re-verify against new fft_config.h ==="

echo "--- rebuild bench_gpu_fused ---"
remote "cd '$REMOTE_WORKDIR' && make clean 2>&1" || true
remote "cd '$REMOTE_WORKDIR' && make bench_gpu_fused CUDA_ARCH=sm_100 CUFFTDX_INC='$CUFFTDX_INC' -j$MAKE_JOBS 2>&1" || die "rebuild bench_gpu_fused failed"

echo "--- re-run verify ---"
VERIFY2_LOG="$LOCAL_RESULTS/stage3b_verify.log"
remote "cd '$REMOTE_WORKDIR' && ./bench_gpu_fused verify 2>&1" | tee "$VERIFY2_LOG" || true

FAIL_COUNT2=$(grep -c "FAIL" "$VERIFY2_LOG" 2>/dev/null || true)
PASS_COUNT2=$(grep -c "PASS" "$VERIFY2_LOG" 2>/dev/null || true)
echo "  FAIL lines: $FAIL_COUNT2"
echo "  PASS lines: $PASS_COUNT2 (need >= $VERIFY_MIN_PASS with 0 FAIL)"

if [ "$FAIL_COUNT2" -eq 0 ] && [ "$PASS_COUNT2" -ge "$VERIFY_MIN_PASS" ]; then
    gate_pass "3b"
else
    gate_fail "3b" "FAIL=$FAIL_COUNT2 PASS=$PASS_COUNT2 (need 0 FAIL, >=$VERIFY_MIN_PASS PASS)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3c — rebuild & re-run gpu_ws_repro against new header
# ──────────────────────────────────────────────────────────────────────────────

echo "=== STAGE 3c (Gate 3c): rebuild + re-run gpu_ws_repro ==="

echo "--- rebuild gpu_ws_repro ---"
remote "cd '$REMOTE_WORKDIR' && \
    nvcc $CUDA_FLAGS $GPU_INCLUDES -Isrc/gpu $CUFFTDX_FLAGS -dc -o '$REPRO_OBJ' '$REPRO_SRC' 2>&1 && \
    nvcc $CUDA_FLAGS -o '$REPRO_BIN' '$REPRO_OBJ' build/gpu_gpu_kernels_fused.o build/gpu_gpu_plan_fused.o build/gpu_gpu_exec_fused.o build/gpu_gpu_api_fused.o build/gpu_dlink_fused.o -lcufft -lcudart 2>&1" || die "rebuild gpu_ws_repro failed"

echo "--- re-run gpu_ws_repro ---"
REPRO2_LOG="$LOCAL_RESULTS/stage3c_repro.log"
set +e
remote "cd '$REMOTE_WORKDIR' && ./gpu_ws_repro 2>&1" | tee "$REPRO2_LOG"
REPRO2_RC=${PIPESTATUS[0]}
set -e

if [ "$REPRO2_RC" -eq 0 ]; then
    gate_pass "3c"
else
    gate_fail "3c" "gpu_ws_repro exit code $REPRO2_RC (expected 0)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4 — heatmap + push_limit sweep
# ──────────────────────────────────────────────────────────────────────────────

echo "=== STAGE 4: heatmap_gpu + push_limit_gpu ==="

echo "--- build heatmap_gpu + push_limit_gpu ---"
remote "cd '$REMOTE_WORKDIR' && make heatmap_gpu push_limit_gpu CUDA_ARCH=sm_100 CUFFTDX_INC='$CUFFTDX_INC' -j$MAKE_JOBS 2>&1" || die "make heatmap_gpu push_limit_gpu failed"

echo "--- run heatmap_gpu ---"
HEATMAP_LOG="$LOCAL_RESULTS/stage4_heatmap.log"
remote "cd '$REMOTE_WORKDIR' && ./heatmap_gpu $HEATMAP_ARGS '$HEATMAP_CSV' 2>&1" | tee "$HEATMAP_LOG" || die "heatmap_gpu failed"

echo "--- run push_limit_gpu ---"
PUSHLIMIT_LOG="$LOCAL_RESULTS/stage4_pushlimit.log"
remote "cd '$REMOTE_WORKDIR' && ./push_limit_gpu $PUSH_LIMIT_ARGS '$PUSH_LIMIT_CSV' 2>&1" | tee "$PUSHLIMIT_LOG" || die "push_limit_gpu failed"

echo "[STAGE 4] complete"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5 — pull results back to local
# ──────────────────────────────────────────────────────────────────────────────

echo "=== STAGE 5: rsync results back ==="

echo "--- pull gpu_fft_config.h ---"
scp -P "$B200_PORT" "$B200_HOST:$REMOTE_WORKDIR/devices/b200/gpu_fft_config.h" "$LOCAL_RESULTS/" || die "scp gpu_fft_config.h failed"

echo "--- pull heatmap CSV ---"
scp -P "$B200_PORT" "$B200_HOST:$REMOTE_WORKDIR/$HEATMAP_CSV" "$LOCAL_RESULTS/" || die "scp heatmap CSV failed"

echo "--- pull push_limit CSV ---"
scp -P "$B200_PORT" "$B200_HOST:$REMOTE_WORKDIR/$PUSH_LIMIT_CSV" "$LOCAL_RESULTS/" || die "scp push_limit CSV failed"

echo "[STAGE 5] complete"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

END_EPOCH="$(date +%s)"
WALL_SEC=$((END_EPOCH - START_EPOCH))
WALL_MIN=$((WALL_SEC / 60))
WALL_REM=$((WALL_SEC % 60))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "ALL GATES PASSED — SWEEP COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Gates: 1 2 3b 3c — all PASS"
echo "  Wall time: ${WALL_MIN}m ${WALL_REM}s"
echo "  Local results: $LOCAL_RESULTS/"
echo "  Contents:"
echo "    stage1_verify.log       — bench_gpu_fused verify (pre-calibration)"
echo "    stage2_repro.log         — gpu_ws_repro (pre-calibration)"
echo "    stage3a_calibrate.log    — calibrate_gpu output"
echo "    stage3b_verify.log       — bench_gpu_fused verify (post-calibration)"
echo "    stage3c_repro.log        — gpu_ws_repro (post-calibration)"
echo "    stage4_heatmap.log       — heatmap_gpu output"
echo "    stage4_pushlimit.log     — push_limit_gpu output"
echo "    gpu_fft_config.h         — regenerated calibration header"
echo "    $HEATMAP_CSV"
echo "    $PUSH_LIMIT_CSV"
echo ""
echo "  REMOTE INSTANCE IS STILL RUNNING — destroy it manually when done."
echo ""
