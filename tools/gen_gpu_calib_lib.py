#!/usr/bin/env python3
"""Generate gpu_calib_lib[] for VkFFT dual-dispatch from vkfft_comparison.csv.

Reads vkfft_comparison.csv and gpu_fft_config.h to produce a C snippet:
  #define HAS_GPU_CALIB_LIB 1
  static const int gpu_calib_lib[GPU_N_CALIBRATED_SIZES] = { 0,0,1,... };

Rules:
  - Only mark VkFFT (1) for sizes where VkFFT was > 5% faster (speedup > 1.05)
  - Only mark VkFFT for sizes > 4096 (cuFFTDx handles <= 4096)
  - All other sizes default to cuFFT (0)
"""

import csv
import re
import sys

def parse_calib_sizes(header_path):
    """Extract gpu_calib_sizes[] from gpu_fft_config.h."""
    with open(header_path) as f:
        text = f.read()
    # Find the array
    m = re.search(r'gpu_calib_sizes\[.*?\]\s*=\s*\{([^}]+)\}', text, re.DOTALL)
    if not m:
        print("ERROR: cannot find gpu_calib_sizes in", header_path, file=sys.stderr)
        sys.exit(1)
    nums = [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
    return nums

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'vkfft_comparison.csv'
    header_path = sys.argv[2] if len(sys.argv) > 2 else 'devices/b200/gpu_fft_config.h'

    # Parse calibration sizes
    calib_sizes = parse_calib_sizes(header_path)
    n_sizes = len(calib_sizes)
    size_to_idx = {s: i for i, s in enumerate(calib_sizes)}

    # Parse VkFFT comparison
    vkfft_wins = set()
    total_compared = 0
    n_vkfft_selected = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sz = int(row['size'])
            total_compared += 1
            if sz <= 4096:
                continue  # cuFFTDx handles these
            winner = row['winner'].strip()
            speedup = float(row['speedup'])
            if winner == 'vkfft' and speedup > 1.05:
                if sz in size_to_idx:
                    vkfft_wins.add(sz)
                    n_vkfft_selected += 1

    # Generate array
    lib = [0] * n_sizes
    for sz in vkfft_wins:
        lib[size_to_idx[sz]] = 1

    # Print summary
    n_gt4096 = sum(1 for s in calib_sizes if s > 4096)
    print(f"/* VkFFT dual-dispatch: {n_vkfft_selected} of {n_gt4096} tier-3 sizes use VkFFT */", file=sys.stderr)
    print(f"/* (from {total_compared} compared sizes, {len(vkfft_wins)} selected with >5% speedup and >4096) */", file=sys.stderr)

    # Output C snippet
    print()
    print(f"#define HAS_GPU_CALIB_LIB 1")
    print(f"/* 0 = cuFFT, 1 = VkFFT. {n_vkfft_selected} VkFFT sizes out of {n_sizes} total. */")
    print(f"static const int gpu_calib_lib[GPU_N_CALIBRATED_SIZES] = {{")
    for i in range(0, n_sizes, 16):
        chunk = lib[i:i+16]
        line = ",".join(str(x) for x in chunk)
        if i + 16 < n_sizes:
            line += ","
        print(f"  {line}")
    print("};")

if __name__ == '__main__':
    main()
