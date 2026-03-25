#!/bin/bash
# setup_zen4.sh — Provision a fresh Ubuntu 24.04 machine for ICM benchmarking
# Run as root (or with sudo) on a Cherry Servers Ryzen 7950X.
#
# Usage:
#   chmod +x tools/setup_zen4.sh
#   sudo ./tools/setup_zen4.sh

set -euo pipefail

echo "=== ICM Zen 4 Setup ==="
echo "Installing build dependencies on Ubuntu 24.04..."

apt-get update
apt-get install -y \
    build-essential \
    gcc \
    libfftw3-dev \
    libfftw3-double3 \
    linux-tools-common \
    linux-tools-generic \
    numactl \
    hwloc \
    git \
    htop

echo ""
echo "=== System Info ==="
gcc --version | head -1
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|MHz|Cache|NUMA"
echo ""
echo "FFTW3 installed:"
dpkg -l | grep fftw3 | awk '{print $2, $3}'

echo ""
echo "=== Disable CPU frequency scaling (for stable benchmarks) ==="
if command -v cpupower &>/dev/null; then
    cpupower frequency-set -g performance 2>/dev/null || echo "  (cpupower not available — check linux-tools-$(uname -r))"
else
    echo "  cpupower not found. Install linux-tools-$(uname -r) for frequency control."
    echo "  Trying manual approach..."
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$gov" 2>/dev/null || true
    done
    echo "  Set all CPUs to performance governor (if supported)."
fi

echo ""
echo "=== Disable turbo boost for consistent results (optional) ==="
echo "  To disable: echo 0 > /sys/devices/system/cpu/cpufreq/boost"
echo "  To re-enable: echo 1 > /sys/devices/system/cpu/cpufreq/boost"
echo "  (Current state: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 'unknown'))"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Clone your repo:  git clone <your-repo-url> && cd ICM"
echo "  2. Run calibration:  sudo nice -20 taskset -c 0 ./tools/calibrate_zen4.sh"
echo "  3. Build & benchmark: ./tools/benchmark_zen4.sh"
