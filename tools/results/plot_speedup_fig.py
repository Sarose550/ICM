#!/usr/bin/env python3
"""
Generate parallel_speedup[_<device>].png — speedup bar chart (CPU).

Usage:
  python3 tools/results/plot_speedup_fig.py --device m3_pro
  python3 tools/results/plot_speedup_fig.py --device zen4
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import (  # noqa: E402
    DEVICE_CONFIGS, RESULTS, GPU_COLOR, HYBRID_COLOR,
    find_latest, load_contour,
)


def plot_speedup(cfg, serial_path, parallel_path, out_path):
    ks, ns, es = load_contour(serial_path)
    kp, np_, ep = load_contour(parallel_path)

    common_k = sorted(set(ks) & set(kp))
    s_dict = dict(zip(ks, ns))
    p_dict = dict(zip(kp, np_))
    p_eng = {k: e for k, e in zip(kp, ep)}
    speedups = [p_dict[k] / s_dict[k] for k in common_k]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [GPU_COLOR if p_eng.get(k) == 'linear' else HYBRID_COLOR for k in common_k]
    ax.bar(range(len(common_k)), speedups, color=colors, alpha=0.85,
           edgecolor='white', linewidth=0.5)

    ax.set_xticks(range(len(common_k)))
    ax.set_xticklabels([f'{k:,}' if k < 1000 else f'{k // 1000}K' for k in common_k],
                       rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Payout terms (k)', fontsize=12)
    ax.set_ylabel('Parallel speedup (n_parallel / n_serial)', fontsize=12)
    ax.set_title(f'{cfg["n_cores"]}-Core Parallel Speedup at 1-Second Boundary ({cfg["short"]}, Q = 256)',
                 fontsize=12)
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_ylim(0, max(speedups) * 1.15)

    for i, v in enumerate(speedups):
        ax.text(i, v + 0.02 * max(speedups), f'{v:.1f}x',
                ha='center', fontsize=8, fontweight='bold')

    ax.legend(handles=[
        Patch(facecolor=GPU_COLOR, alpha=0.85, label='Linear engine'),
        Patch(facecolor=HYBRID_COLOR, alpha=0.85, label='Hybrid engine'),
    ], loc='upper left', fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate parallel-speedup bar chart.')
    parser.add_argument('--device', choices=['zen4', 'm3_pro'], default='zen4',
                        help='Device to generate plot for (default: zen4)')
    parser.add_argument('--serial', default=None,
                        help='Path to serial contour CSV (auto-detected if omitted)')
    parser.add_argument('--parallel', default=None,
                        help='Path to parallel contour CSV (auto-detected if omitted)')
    parser.add_argument('--out', default=None,
                        help='Output PNG path (auto-generated if omitted)')
    args = parser.parse_args()

    cfg = DEVICE_CONFIGS[args.device]
    suffix = cfg['output_suffix']

    serial_csv = args.serial or find_latest(cfg['serial_csv'])
    parallel_csv = args.parallel or find_latest(cfg['parallel_csv'])

    if not serial_csv or not parallel_csv:
        print("Missing contour CSV files — cannot generate speedup plot.")
        sys.exit(1)

    out_path = args.out or os.path.join(RESULTS, f'parallel_speedup{suffix}.png')
    plot_speedup(cfg, serial_csv, parallel_csv, out_path)
