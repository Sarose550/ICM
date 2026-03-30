#!/bin/bash
# Run bench_vkfft per-size with process isolation to handle VkFFT fatal errors
# Usage: ./tools/run_vkfft_calib.sh > vkfft_comparison_strided.csv 2>vkfft_calib.log

# Extract sizes from fft_config.h: grab array body between { and };
SIZES=$(sed -n '/calib_sizes\[/,/^};/p' devices/zen4/fft_config.h \
    | sed '1d;$d' \
    | tr ',' '\n' | tr -d ' ' | grep -E '^[0-9]+$' | awk '$1 >= 16')

count=$(echo "$SIZES" | wc -l)
echo >&2 "Will benchmark $count sizes"

echo "size,cufft_ns,vkfft_ns,winner,speedup"

total=0
ok=0
fail=0

for sz in $SIZES; do
    result=$(timeout 60 ./bench_vkfft $sz 2>/dev/null | head -1)
    if [ $? -eq 0 ] && [ -n "$result" ] && echo "$result" | grep -qP '^\d+,'; then
        echo "$result"
        ok=$((ok + 1))
    else
        echo >&2 "FAILED: size $sz (timeout or crash)"
        fail=$((fail + 1))
    fi
    total=$((total + 1))
    if [ $((total % 50)) -eq 0 ]; then
        echo >&2 "Progress: $total/$count sizes ($ok ok, $fail fail)"
    fi
done

echo >&2 ""
echo >&2 "=== Complete: $total sizes tested, $ok ok, $fail failed ==="
