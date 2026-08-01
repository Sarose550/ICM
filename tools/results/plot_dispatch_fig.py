#!/usr/bin/env python3
"""
Generate engine_dispatch[_<device>].png — engine coloring on contour (CPU).

Usage:
  python3 tools/results/plot_dispatch_fig.py --device m3_pro
  python3 tools/results/plot_dispatch_fig.py --device zen4
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import (  # noqa: E402
    DEVICE_CONFIGS, RESULTS, LINEAR_COLOR, HYBRID_COLOR,
    find_latest, load_contour, _trim_uptick,
)


def plot_dispatch(cfg, serial_path, out_path):
    ks, ns, es = load_contour(serial_path)

    # Trim uptick at tail for smooth line ending
    ks_line, ns_line = _trim_uptick(ks, ns)

    fig, ax = plt.subplots(figsize=(10, 6))
    for k_val, n_val, eng in zip(ks_line, ns_line, es[:len(ks_line)]):
        color = LINEAR_COLOR if eng == 'linear' else HYBRID_COLOR
        ax.scatter(k_val, n_val, c=color, s=80, zorder=5, edgecolors='white', linewidth=0.5)

    ax.plot(ks_line, ns_line, '-', color='gray', linewidth=1, alpha=0.5, zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Payout terms (k)', fontsize=13)
    ax.set_ylabel('Maximum players (n) in 1 second', fontsize=13)
    ax.set_title(f'Engine Dispatch at 1-Second Boundary (Serial, {cfg["short"]})', fontsize=14)

    ax.legend(handles=[
        Line2D([0], [0], color=LINEAR_COLOR, marker='o', markersize=8, linestyle='none',
               label='Linear engine'),
        Line2D([0], [0], color=HYBRID_COLOR, marker='o', markersize=8, linestyle='none',
               label='Hybrid engine'),
    ], loc='upper right', fontsize=10, framealpha=0.9)

    ax.set_xlim(1.5, max(ks) * 1.5)
    ax.set_ylim(min(ns) * 0.5, max(ns) * 2)
    ax.grid(True, which='both', alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate engine-dispatch contour plot.')
    parser.add_argument('--device', choices=['zen4', 'm3_pro'], default='zen4',
                        help='Device to generate plot for (default: zen4)')
    parser.add_argument('--serial', default=None,
                        help='Path to serial contour CSV (auto-detected if omitted)')
    parser.add_argument('--out', default=None,
                        help='Output PNG path (auto-generated if omitted)')
    args = parser.parse_args()

    cfg = DEVICE_CONFIGS[args.device]
    suffix = cfg['output_suffix']

    serial_csv = args.serial or find_latest(cfg['serial_csv'])

    if not serial_csv:
        print("Missing serial contour CSV — cannot generate dispatch plot.")
        sys.exit(1)

    out_path = args.out or os.path.join(RESULTS, f'engine_dispatch{suffix}.png')
    plot_dispatch(cfg, serial_csv, out_path)
