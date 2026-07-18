#!/bin/bash
# Setup script for B200 vast.ai instance
# Run once after SSH in: bash tools/setup_b200.sh
set -e

echo "=== Installing dependencies ==="
apt-get update -qq && apt-get install -y -qq git make wget > /dev/null 2>&1
pip3 install nvidia-mathdx 2>/dev/null || pip install nvidia-mathdx 2>/dev/null

echo "=== Locating cuFFTDx ==="
MATHDX=$(python3 -c "import nvidia.mathdx; import os; print(os.path.dirname(nvidia.mathdx.__file__))" 2>/dev/null)
CUFFTDX_INC="$MATHDX/include"
if [ ! -f "$CUFFTDX_INC/cufftdx.hpp" ]; then
    echo "ERROR: cufftdx.hpp not found at $CUFFTDX_INC"
    find / -name "cufftdx.hpp" 2>/dev/null | head -5
    exit 1
fi
echo "  cuFFTDx found at: $CUFFTDX_INC"

echo "=== Building gpu_sample_plans ==="
cd /root/ICM
make gpu_sample_plans CUDA_ARCH=sm_100 CUFFTDX_INC="-I$CUFFTDX_INC"

echo "=== Running gpu_sample_plans ==="
./gpu_sample_plans > gpu_sample_plans_b200.csv 2> gpu_sample_plans_b200.log

echo "=== Done ==="
wc -l gpu_sample_plans_b200.csv
tail -5 gpu_sample_plans_b200.log
