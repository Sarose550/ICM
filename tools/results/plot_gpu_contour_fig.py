#!/usr/bin/env python3
"""
Generate gpu_contour.png: GPU 1-second contour from heatmap data,
and runtime_vs_n_gpu.png: Runtime(n) at fixed k, log-log (GPU).

Usage:
  python3 tools/results/plot_gpu_contour_fig.py
  python3 tools/results/plot_gpu_contour_fig.py --heatmap results/gpu_heatmap_b200.csv
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import (  # noqa: E402
    RESULTS, GPU_COLOR,
    find_latest, load_gpu_heatmap, extract_gpu_runtime_data,
)


def plot_gpu_contour(heatmap_path, out_path):
    """Extract 1-second boundary from GPU heatmap data."""
    rows = load_gpu_heatmap(heatmap_path)

    # For each k, find the largest n where time <= 1000ms
    by_k = {}
    for r in rows:
        by_k.setdefault(r['k'], []).append(r)

    contour_k, contour_n = [], []
    for k in sorted(by_k.keys()):
        pts = sorted(by_k[k], key=lambda r: r['n'])
        best_n = None
        for p in pts:
            if p['time_ms'] <= 1000:
                best_n = p['n']
        if best_n is not None:
            contour_k.append(k)
            contour_n.append(best_n)

    if not contour_k:
        print(f"No GPU contour data found in {heatmap_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 8.5))
    ax.plot(contour_k, contour_n, 'o-', color=GPU_COLOR, markersize=6, linewidth=2)
    ax.fill_between(contour_k, contour_n, alpha=0.08, color=GPU_COLOR)

    xlo, xhi = min(contour_k) * 0.7, max(contour_k) * 1.5
    ylo, yhi = min(contour_n) * 0.7, max(contour_n) * 1.5

    # n = k reference line
    lo = min(xlo, ylo)
    hi = max(xhi, yhi)
    k_diag = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.plot(k_diag, k_diag, '--', color='#666666', alpha=0.6, linewidth=1.5,
            zorder=2, clip_on=True)
    mid = np.sqrt(max(xlo, ylo) * min(xhi, yhi))
    ax.text(mid * 0.8, mid * 0.55, 'n = k', fontsize=10, color='#666666',
            alpha=0.7, rotation=38, clip_on=True)

    # Mark intersection (1-second k=n threshold)
    for i in range(len(contour_k) - 1):
        if contour_n[i] >= contour_k[i] and contour_n[i + 1] <= contour_k[i + 1]:
            frac = (np.log(contour_k[i]) - np.log(contour_n[i])) / \
                   ((np.log(contour_n[i + 1]) - np.log(contour_n[i])) -
                    (np.log(contour_k[i + 1]) - np.log(contour_k[i])))
            k_cross = np.exp(np.log(contour_k[i]) + frac * (np.log(contour_k[i + 1]) - np.log(contour_k[i])))
            ax.plot(k_cross, k_cross, '*', color=GPU_COLOR, markersize=14, zorder=10,
                    markeredgecolor='white', markeredgewidth=0.8)
            break

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Payout terms (k)', fontsize=13)
    ax.set_ylabel('Players (n)', fontsize=13)
    ax.set_title('1-Second Contour: ICM Equity on NVIDIA B200 (Q = 256)', fontsize=14)
    ax.grid(True, which='both', alpha=0.2)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_runtime_vs_n(data, title, out_path, k_values=None):
    """Generate runtime-vs-n log-log plot."""
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
        description='Generate GPU contour and GPU runtime-vs-n plots.')
    parser.add_argument('--heatmap', default=None,
                        help='Path to GPU heatmap CSV (auto-detected if omitted)')
    parser.add_argument('--out-contour', default=None,
                        help='Output PNG path for contour (default: results/gpu_contour.png)')
    parser.add_argument('--out-runtime', default=None,
                        help='Output PNG path for runtime (default: results/runtime_vs_n_gpu.png)')
    args = parser.parse_args()

    gpu_heatmap = args.heatmap or find_latest('gpu_heatmap_*.csv')

    if not gpu_heatmap:
        print("No GPU heatmap CSV found, cannot generate GPU plots.")
        sys.exit(1)

    # GPU contour
    out_contour = args.out_contour or os.path.join(RESULTS, 'gpu_contour.png')
    plot_gpu_contour(gpu_heatmap, out_contour)

    # GPU runtime vs n
    gpu_data = extract_gpu_runtime_data(gpu_heatmap)
    if gpu_data:
        all_k = sorted(set(k for _, k, _, _ in gpu_data))
        target_k = []
        for want in [2, 10, 50, 100]:
            closest = min(all_k, key=lambda x: abs(x - want))
            if closest not in target_k:
                target_k.append(closest)

        gpu_data_ext = list(gpu_data)
        for label, ratio in [('k=n/4', 4), ('k=n/2', 2), ('k=n', 1)]:
            pts = [(n, k, t, e) for n, k, t, e in gpu_data if k == n // ratio]
            if pts:
                for n, k, t, e in pts:
                    gpu_data_ext.append((n, label, t, e))
                target_k.append(label)

        out_runtime = args.out_runtime or os.path.join(RESULTS, 'runtime_vs_n_gpu.png')
        plot_runtime_vs_n(gpu_data_ext,
                          'Runtime vs n (NVIDIA B200, Q = 256)',
                          out_runtime,
                          k_values=target_k)
