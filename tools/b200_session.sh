#!/bin/bash
# tools/b200_session.sh — Complete E2 GPU session for B200 (vast.ai, CUDA 13)
#
# Run: bash tools/b200_session.sh 2>&1 | tee results/session.log
# Designed for: fresh vast.ai B200 instance, repo rsynced to ~/ICM
# Total budget: <= 45 instance-minutes.  Every command under timeout.
#
# Prerequisites (run once before this script):
#   rsync -avz --delete /local/ICM/ user@instance:~/ICM/
#   ssh user@instance 'cd ~/ICM && bash tools/setup_b200.sh'
set -euo pipefail

REPO="${HOME}/ICM"
cd "$REPO"

mkdir -p results
LOG="results/session_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "============================================"
echo " B200 E2 Session — $(date)"
echo "============================================"

# ── Step 1: Environment check ──────────────────────────────────
echo ""
echo "=== Step 1: Environment ==="
timeout 30 nvidia-smi || { echo "FAIL: nvidia-smi"; exit 1; }
timeout 10 nvcc --version || { echo "FAIL: nvcc"; exit 1; }

# Check VRAM (expect ~192 GB on B200)
VRAM_BYTES=$(timeout 15 nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "VRAM reported: ${VRAM_BYTES} MiB"
if [ -z "$VRAM_BYTES" ] || [ "$VRAM_BYTES" -lt 100000 ]; then
    echo "WARNING: VRAM looks low — not a B200?"
fi

# ── Step 2: Build bench_gpu_fused ──────────────────────────────
echo ""
echo "=== Step 2: Build bench_gpu_fused (CUDA_ARCH=sm_100) ==="
timeout 300 make bench_gpu_fused CUDA_ARCH=sm_100 -j$(nproc) 2>&1 | tail -20 || { echo "FAIL: build"; exit 1; }
if [ ! -x ./bench_gpu_fused ]; then
    echo "FAIL: bench_gpu_fused not built"
    exit 1
fi
echo "BUILD OK."

# ── Step 3: Reproduce OOM (n=1,048,576, k=n) ──────────────────
echo ""
echo "=== Step 3: Repro OOM — n=1048576, k=n ==="
timeout 300 ./bench_gpu_fused bench 1048576 1048576 1 64 2>&1 || {
    RC=$?
    if [ $RC -eq 124 ]; then
        echo "REPRO OOM: timed out (>5 min) — consistent with OOM/hang"
    else
        echo "REPRO OOM: exit code $RC — check output for cudaMalloc/OOM error"
    fi
}
echo "OOM repro complete."

# ── Step 4: Reproduce cuFFT error (n=524,288, k=n) ────────────
echo ""
echo "=== Step 4: Repro cuFFT — n=524288, k=n ==="
timeout 300 ./bench_gpu_fused bench 524288 524288 1 64 2>&1 || {
    RC=$?
    if [ $RC -eq 124 ]; then
        echo "REPRO cuFFT: timed out"
    else
        echo "REPRO cuFFT: exit code $RC — check for CUFFT_INTERNAL_ERROR (code 5)"
    fi
}
echo "cuFFT repro complete."

# ── Step 5: Apply patches ──────────────────────────────────────
echo ""
echo "=== Step 5: Apply patches ==="
for patch in patches/oom_fix.patch patches/cufft_524k_fix.patch; do
    echo "  Checking $patch ..."
    timeout 30 git apply --check "$patch" || {
        echo "FAIL: git apply --check failed for $patch"
        exit 1
    }
done
for patch in patches/oom_fix.patch patches/cufft_524k_fix.patch; do
    echo "  Applying $patch ..."
    timeout 30 git apply "$patch" || {
        echo "FAIL: git apply failed for $patch"
        exit 1
    }
done
echo "Patches applied OK."

# ── Step 6: Rebuild ────────────────────────────────────────────
echo ""
echo "=== Step 6: Rebuild with patches ==="
timeout 300 make bench_gpu_fused CUDA_ARCH=sm_100 -j$(nproc) 2>&1 | tail -20 || { echo "FAIL: rebuild"; exit 1; }
if [ ! -x ./bench_gpu_fused ]; then
    echo "FAIL: rebuild failed"
    exit 1
fi
echo "REBUILD OK."

# ── Step 7: Verify fixes ───────────────────────────────────────
echo ""
echo "=== Step 7a: Verify OOM fix — n=1048576, k=n ==="
RC=0
timeout 300 ./bench_gpu_fused bench 1048576 1048576 1 64 2>&1 || RC=$?
if [ $RC -eq 0 ]; then
    echo "OOM FIX: passed (RC=0)"
else
    echo "OOM FIX: still failing (RC=$RC) — may need further qb reduction"
fi

echo ""
echo "=== Step 7b: Verify cuFFT fix — n=524288, k=n ==="
RC=0
timeout 300 ./bench_gpu_fused bench 524288 524288 1 64 2>&1 || RC=$?
if [ $RC -eq 0 ]; then
    echo "cuFFT FIX: passed (RC=0)"
else
    echo "cuFFT FIX: still failing (RC=$RC)"
fi

# ── Step 8: Correctness verification ───────────────────────────
echo ""
echo "=== Step 8: Verify correctness ==="
if timeout 600 ./bench_gpu_fused verify 2>&1 | tail -30; then
    echo "VERIFY: passed"
else
    echo "VERIFY: FAILED"
    exit 1
fi

# ── Step 9: Frontier confirmation (k=n at two sizes, 3 reps) ──
echo ""
echo "=== Step 9: Frontier confirmation ==="

frontier_test() {
    local n=$1
    local label=$2
    local results_file="results/frontier_${label}_$(date +%Y%m%d_%H%M%S).txt"
    echo "  Testing n=$n, k=n (3 reps, Q=256) -> $results_file"
    {
        echo "n=$n k=$n reps=3 Q=256"
        timeout 300 ./bench_gpu_fused bench "$n" "$n" 3 256 2>&1
    } > "$results_file" 2>&1 || {
        echo "  WARNING: n=$n did not complete all reps"
    }
    # Extract best time
    grep -E '(total|wall|ms|elapsed)' "$results_file" | tail -6 || true
}

frontier_test 1441792 "n1441792"
frontier_test 1572864 "n1572864"

echo "Frontier tests complete."

# ── Step 10: Write summary ─────────────────────────────────────
echo ""
echo "=== Step 10: Summary ==="
{
    echo "B200 E2 Session Summary"
    echo "Date: $(date)"
    echo "Host: $(hostname)"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
    echo "Patches applied:"
    echo "  - patches/oom_fix.patch"
    echo "  - patches/cufft_524k_fix.patch"
    echo ""
    echo "Frontier results:"
    echo "  n=1,441,792 k=n:"
    grep -E '(total|ms|elapsed)' results/frontier_n1441792_*.txt 2>/dev/null | tail -5 || echo "  (no data)"
    echo "  n=1,572,864 k=n:"
    grep -E '(total|ms|elapsed)' results/frontier_n1572864_*.txt 2>/dev/null | tail -5 || echo "  (no data)"
} > results/summary.txt

cat results/summary.txt

# ── Step 11: Rsync-back command ─────────────────────────────────
echo ""
echo "=== Step 11: Rsync-back ==="
echo ""
echo "  SESSION COMPLETE.  To pull results back:"
echo ""
echo "  rsync -avz user@$(hostname):~/ICM/results/ ./results_b200_session/"
echo ""

echo "============================================"
echo " B200 E2 Session DONE — $(date)"
echo "============================================"
