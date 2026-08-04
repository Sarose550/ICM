#!/usr/bin/env python3
"""
Generate runtime_vs_n_cpu[_<device>].png: Runtime(n) at fixed k, log-log (CPU).

Usage:
  python3 tools/results/plot_runtime_fig.py --device m3_pro
  python3 tools/results/plot_runtime_fig.py --device zen4
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import (  # noqa: E402
    DEVICE_CONFIGS, RESULTS,
    find_latest, extract_runtime_data_from_bench_grid,
)


def plot_runtime_vs_n(data, title, out_path, k_values=None):
    """
    data: list of (n, k, time_ms, engine) tuples
    k_values: list of k values to plot. If None, auto-select.
    """
    # Group by k
    by_k = {}
    for n, k, t, eng in data:
        by_k.setdefault(k, []).append((n, t, eng))

    if k_values is None:
        k_values = sorted(by_k.keys())

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(k_values)))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, k in enumerate(k_values):
        if k not in by_k:
            continue
        pts = sorted(by_k[k])
        pts = [(n, t, e) for n, t, e in pts if t > 0]
        if not pts:
            continue
        ns = [p[0] for p in pts]
        ts = [p[1] for p in pts]
        label = f'k = {k}' if isinstance(k, int) else k
        ax.plot(ns, ts, 'o-', color=colors[i], markersize=5, linewidth=1.5, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # 1-second reference line
    ax.axhline(y=1000, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(ax.get_xlim()[0] * 1.3, 1150, '1 second', fontsize=9, color='gray', alpha=0.6)

    ax.set_xlabel('Players (n)', fontsize=13)
    ax.set_ylabel('Time (ms)', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate runtime-vs-n plot (CPU, log-log).')
    parser.add_argument('--device', choices=['zen4', 'm3_pro'], default='zen4',
                        help='Device to generate plot for (default: zen4)')
    parser.add_argument('--bench', default=None,
                        help='Path to bench_grid serial output (auto-detected if omitted)')
    parser.add_argument('--out', default=None,
                        help='Output PNG path (auto-generated if omitted)')
    args = parser.parse_args()

    cfg = DEVICE_CONFIGS[args.device]
    suffix = cfg['output_suffix']

    bench_full = args.bench or find_latest(cfg['bench_grid'])

    if not bench_full:
        print("Missing bench_grid output, cannot generate runtime-vs-n plot.")
        sys.exit(1)

    data = extract_runtime_data_from_bench_grid(bench_full)
    if not data:
        print("No runtime data found in bench_grid output.")
        sys.exit(1)

    all_k = sorted(set(k for _, k, _, _ in data))
    fixed_k = [k for k in [10, 50, 100] if k in all_k]

    # Build synthetic labels for ratio k values
    data_ext = list(data)
    ratio_labels = []
    for label, ratio in [('k=n/4', 4), ('k=n/2', 2), ('k=n', 1)]:
        pts = [(n, k, t, e) for n, k, t, e in data if k == n // ratio]
        if pts:
            for n, k, t, e in pts:
                data_ext.append((n, label, t, e))
            ratio_labels.append(label)

    out_path = args.out or os.path.join(RESULTS, f'runtime_vs_n_cpu{suffix}.png')
    plot_runtime_vs_n(data_ext,
                      f'Runtime vs n (Serial, {cfg["short"]}, Q = 256)',
                      out_path,
                      k_values=fixed_k + ratio_labels)
