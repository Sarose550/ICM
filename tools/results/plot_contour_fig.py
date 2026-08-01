#!/usr/bin/env python3
"""
Generate contour_1s[_<device>].png — serial vs parallel 1-second boundary (CPU).

Usage:
  python3 tools/results/plot_contour_fig.py --device m3_pro
  python3 tools/results/plot_contour_fig.py --device zen4
  python3 tools/results/plot_contour_fig.py --device m3_pro --serial results/contour_m3pro_serial_q256.csv --parallel results/contour_m3pro_parallel_q256.csv --out results/contour_1s_m3pro.png
"""

import argparse
import os
import sys

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import (  # noqa: E402
    DEVICE_CONFIGS, RESULTS, SERIAL_COLOR, PARALLEL_COLOR,
    find_latest, load_contour, _trim_uptick,
)


def plot_contour(cfg, serial_path, parallel_path, out_path):
    ks, ns, es = load_contour(serial_path)
    kp, np_, ep = load_contour(parallel_path)

    # Trim any uptick at the tail so the connected line ends smoothly.
    ks_line, ns_line = _trim_uptick(ks, ns)
    kp_line, np_line = _trim_uptick(kp, np_)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.fill_between(ks, ns, alpha=0.06, color=SERIAL_COLOR)
    ax.fill_between(kp, np_, alpha=0.06, color=PARALLEL_COLOR)

    # Serial with engine markers
    s_lin = [(k, n) for k, n, e in zip(ks_line, ns_line, es[:len(ks_line)]) if e == 'linear']
    s_hyb = [(k, n) for k, n, e in zip(ks_line, ns_line, es[:len(ks_line)]) if e == 'hybrid']

    ax.plot(ks_line, ns_line, '-', color=SERIAL_COLOR, linewidth=2, alpha=0.6, zorder=4)
    if s_lin:
        ax.plot(*zip(*s_lin), 'o', color=SERIAL_COLOR, markersize=7, zorder=5)
    if s_hyb:
        ax.plot(*zip(*s_hyb), 's', color=SERIAL_COLOR, markersize=6, zorder=5)

    # Parallel with engine markers
    p_lin = [(k, n) for k, n, e in zip(kp_line, np_line, ep[:len(kp_line)]) if e == 'linear']
    p_hyb = [(k, n) for k, n, e in zip(kp_line, np_line, ep[:len(kp_line)]) if e == 'hybrid']

    ax.plot(kp_line, np_line, '-', color=PARALLEL_COLOR, linewidth=2, alpha=0.6, zorder=4)
    if p_lin:
        ax.plot(*zip(*p_lin), 'o', color=PARALLEL_COLOR, markersize=7, zorder=5)
    if p_hyb:
        ax.plot(*zip(*p_hyb), 's', color=PARALLEL_COLOR, markersize=6, zorder=5)

    # n = k reference line
    all_vals = np.concatenate([ks, ns, kp, np_])
    lo, hi = min(all_vals) * 0.5, max(all_vals) * 2
    k_diag = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.plot(k_diag, k_diag, '--', color='#666666', alpha=0.6, linewidth=1.5, zorder=2)
    mid = np.sqrt(lo * hi)
    ax.text(mid * 0.8, mid * 0.55, 'n = k', fontsize=10, color='#666666',
            alpha=0.7, rotation=38)

    # Mark intersection of n=k line with contours
    for label, k_arr, n_arr, color in [('serial', ks, ns, SERIAL_COLOR),
                                        ('parallel', kp, np_, PARALLEL_COLOR)]:
        for i in range(len(k_arr) - 1):
            if n_arr[i] >= k_arr[i] and n_arr[i + 1] <= k_arr[i + 1]:
                frac = (np.log(k_arr[i]) - np.log(n_arr[i])) / \
                       ((np.log(n_arr[i + 1]) - np.log(n_arr[i])) -
                        (np.log(k_arr[i + 1]) - np.log(k_arr[i])))
                k_cross = np.exp(np.log(k_arr[i]) + frac * (np.log(k_arr[i + 1]) - np.log(k_arr[i])))
                ax.plot(k_cross, k_cross, '*', color=color, markersize=12, zorder=10,
                        markeredgecolor='white', markeredgewidth=0.8)
                break

    # Engine crossover vertical line (serial)
    for i in range(len(es) - 1):
        if es[i] == 'linear' and es[i + 1] == 'hybrid':
            cross_k = (ks[i] + ks[i + 1]) / 2
            ax.axvline(x=cross_k, color='gray', linestyle=':', alpha=0.4, linewidth=1)
            break

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Payout terms (k)', fontsize=13)
    ax.set_ylabel('Players (n)', fontsize=13)
    ax.set_title(f'1-Second Contour: ICM Equity on {cfg["label"]} (Q = 256)', fontsize=14)

    legend_elements = [
        Line2D([0], [0], color=SERIAL_COLOR, marker='o', markersize=7, linewidth=2,
               label='Serial (1 core)'),
        Line2D([0], [0], color=PARALLEL_COLOR, marker='o', markersize=7, linewidth=2,
               label=f'Parallel ({cfg["n_cores"]} cores)'),
        Line2D([0], [0], color='gray', marker='o', markersize=6, linestyle='none',
               label='Linear engine'),
        Line2D([0], [0], color='gray', marker='s', markersize=6, linestyle='none',
               label='Hybrid engine'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9.5, framealpha=0.9)

    ax.set_xlim(1.5, max(max(ks), max(kp)) * 1.5)
    y_lo = min(min(ns), min(np_)) * 0.5
    y_hi = max(max(ns), max(np_)) * 2
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, which='both', alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate 1-second contour plot (CPU, serial vs parallel).')
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
        print("Missing contour CSV files — cannot generate contour plot.")
        sys.exit(1)

    out_path = args.out or os.path.join(RESULTS, f'contour_1s{suffix}.png')
    plot_contour(cfg, serial_csv, parallel_csv, out_path)
