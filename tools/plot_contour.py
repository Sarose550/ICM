#!/usr/bin/env python3
"""
Generate publication plots for ICM benchmark data.

Plots generated:
  1. contour_1s.png         — Serial vs parallel 1-second boundary (CPU)
  2. parallel_speedup.png   — Speedup bar chart
  3. engine_dispatch.png    — Engine coloring on contour
  4. runtime_vs_n_cpu.png   — Runtime(n) at fixed k values, log-log (CPU)
  5. runtime_vs_n_gpu.png   — Runtime(n) at fixed k values, log-log (GPU)
  6. gpu_contour.png        — GPU 1-second contour from heatmap data

Usage:
  python3 tools/plot_contour.py

Reads from project root:
  contour_zen4_serial_q256.csv, contour_zen4_parallel_q256.csv,
  bench_grid_full.txt, gpu_heatmap_new.csv
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import re
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
})

SERIAL_COLOR = '#2563eb'
PARALLEL_COLOR = '#dc2626'
LINEAR_COLOR = '#9333ea'
HYBRID_COLOR = '#059669'
GPU_COLOR = '#f97316'

# ─── Data loading ────────────────────────────────────────────

def load_contour(path, max_time_ms=2000):
    """Load contour CSV. Filter out points where time > max_time_ms."""
    k, n, engine = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ki, ni, ti = int(row['k']), int(row['n_max']), float(row['time_ms'])
            if ti > max_time_ms:
                continue
            k.append(ki)
            n.append(ni)
            engine.append(row['engine'])
    return np.array(k), np.array(n), engine


def load_bench_grid(path):
    """Parse bench_grid output into (n, k, time_ms, engine) tuples."""
    rows = []
    with open(path) as f:
        for line in f:
            # Match lines like: "  n=4096  k=100   L  28.3 ms"
            m = re.match(r'\s*n=(\d+)\s+k=(\d+)\s+([LHT])\S*\s+([\d.]+)\s*ms', line)
            if m:
                rows.append((int(m.group(1)), int(m.group(2)),
                             float(m.group(4)), m.group(3)))
    return rows


def load_gpu_heatmap(path):
    """Load GPU heatmap CSV."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'n': int(row['n']),
                'k': int(row['k']),
                'time_ms': float(row['time_ms']),
                'B': int(row['B']),
                'engine': row.get('engine', 'hybrid'),
            })
    return rows


# ─── Plot 1: 1-second contour (CPU) ─────────────────────────

def plot_contour(serial_path, parallel_path, out_path):
    ks, ns, es = load_contour(serial_path)
    kp, np_, ep = load_contour(parallel_path)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.fill_between(ks, ns, alpha=0.06, color=SERIAL_COLOR)
    ax.fill_between(kp, np_, alpha=0.06, color=PARALLEL_COLOR)

    # Serial with engine markers
    s_lin = [(k, n) for k, n, e in zip(ks, ns, es) if e == 'linear']
    s_hyb = [(k, n) for k, n, e in zip(ks, ns, es) if e == 'hybrid']

    ax.plot(ks, ns, '-', color=SERIAL_COLOR, linewidth=2, alpha=0.6, zorder=4)
    if s_lin:
        ax.plot(*zip(*s_lin), 'o', color=SERIAL_COLOR, markersize=7, zorder=5)
    if s_hyb:
        ax.plot(*zip(*s_hyb), 's', color=SERIAL_COLOR, markersize=6, zorder=5)

    # Parallel with engine markers
    p_lin = [(k, n) for k, n, e in zip(kp, np_, ep) if e == 'linear']
    p_hyb = [(k, n) for k, n, e in zip(kp, np_, ep) if e == 'hybrid']

    ax.plot(kp, np_, '-', color=PARALLEL_COLOR, linewidth=2, alpha=0.6, zorder=4)
    if p_lin:
        ax.plot(*zip(*p_lin), 'o', color=PARALLEL_COLOR, markersize=7, zorder=5)
    if p_hyb:
        ax.plot(*zip(*p_hyb), 's', color=PARALLEL_COLOR, markersize=6, zorder=5)

    # n = k reference line
    k_range = np.logspace(np.log10(min(ks)), np.log10(max(ks)), 100)
    ax.plot(k_range, k_range, '--', color='gray', alpha=0.4, linewidth=1, zorder=2)
    ax.text(max(ks) * 0.7, max(ks) * 0.5, 'n = k', fontsize=9, color='gray',
            alpha=0.6, rotation=35)

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
    ax.set_title('1-Second Contour: ICM Equity on Ryzen 9 7950X (Q = 256)', fontsize=14)

    legend_elements = [
        Line2D([0], [0], color=SERIAL_COLOR, marker='o', markersize=7, linewidth=2,
               label='Serial (1 core)'),
        Line2D([0], [0], color=PARALLEL_COLOR, marker='o', markersize=7, linewidth=2,
               label='Parallel (16 cores)'),
        Line2D([0], [0], color='gray', marker='o', markersize=6, linestyle='none',
               label='Linear engine'),
        Line2D([0], [0], color='gray', marker='s', markersize=6, linestyle='none',
               label='Hybrid engine'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9.5, framealpha=0.9)

    ax.set_xlim(1.5, max(ks) * 1.5)
    y_lo = min(min(ns), min(np_)) * 0.5
    y_hi = max(max(ns), max(np_)) * 2
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, which='both', alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ─── Plot 2: Parallel speedup ───────────────────────────────

def plot_speedup(serial_path, parallel_path, out_path):
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
    ax.set_title('16-Core Parallel Speedup at 1-Second Boundary (Zen 4, Q = 256)', fontsize=12)
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
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ─── Plot 3: Engine dispatch map ─────────────────────────────

def plot_dispatch(serial_path, out_path):
    ks, ns, es = load_contour(serial_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for k_val, n_val, eng in zip(ks, ns, es):
        color = LINEAR_COLOR if eng == 'linear' else HYBRID_COLOR
        ax.scatter(k_val, n_val, c=color, s=80, zorder=5, edgecolors='white', linewidth=0.5)

    ax.plot(ks, ns, '-', color='gray', linewidth=1, alpha=0.5, zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Payout terms (k)', fontsize=13)
    ax.set_ylabel('Maximum players (n) in 1 second', fontsize=13)
    ax.set_title('Engine Dispatch at 1-Second Boundary (Serial, Zen 4)', fontsize=14)

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
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ─── Plot 4/5: Runtime vs n at fixed k (log-log) ────────────

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

    # Filter to requested k values (exact or ratio)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(k_values)))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, k in enumerate(k_values):
        if k not in by_k:
            continue
        pts = sorted(by_k[k])
        ns = [p[0] for p in pts]
        ts = [p[1] for p in pts]
        label = f'k = {k}' if isinstance(k, int) else k
        ax.plot(ns, ts, 'o-', color=colors[i], markersize=5, linewidth=1.5, label=label)

    # 1-second reference line
    ax.axhline(y=1000, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(ax.get_xlim()[0] * 1.5, 1100, '1 second', fontsize=9, color='gray', alpha=0.6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Players (n)', fontsize=13)
    ax.set_ylabel('Time (ms)', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def extract_runtime_data_from_bench_grid(path):
    """Parse bench_grid full output for runtime-vs-n plots.

    Format: each data row starts with a number (n), then cells like
    'L:16' or 'H:295' or 'T:150' followed by detail in parens.
    Header line has 'k=10', 'k=50', ..., 'k=n/4', 'k=n/2', 'k=n'.
    """
    data = []
    k_headers = []

    with open(path) as f:
        lines = f.readlines()

    in_grid = False
    for line in lines:
        if 'PERFORMANCE GRID' in line:
            in_grid = True
            continue

        if in_grid and not k_headers and 'k=' in line:
            # Parse header: extract k=... tokens
            for m in re.finditer(r'k=(\S+)', line):
                k_headers.append(m.group(1))
            continue

        if in_grid and k_headers and line.strip() and line.strip()[0].isdigit():
            # Data row: "64      L:0   (T0 L0 H0 ) ..."
            n_match = re.match(r'\s*(\d+)', line)
            if not n_match:
                continue
            n = int(n_match.group(1))

            # Find all "X:NNN" patterns (best engine cells)
            cells = re.findall(r'([LHT]):(\d+)', line)
            for col, (eng, t_str) in enumerate(cells):
                if col >= len(k_headers):
                    break
                k_str = k_headers[col]
                if k_str == 'n':
                    k = n
                elif k_str.startswith('n/'):
                    k = n // int(k_str[2:])
                else:
                    k = int(k_str)
                data.append((n, k, float(t_str), eng))

    return data


def extract_gpu_runtime_data(heatmap_path):
    """Extract runtime data from GPU heatmap for runtime-vs-n plots."""
    rows = load_gpu_heatmap(heatmap_path)
    return [(r['n'], r['k'], r['time_ms'], 'H') for r in rows]


# ─── Plot 6: GPU contour from heatmap ───────────────────────

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

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(contour_k, contour_n, 'o-', color=GPU_COLOR, markersize=6, linewidth=2)
    ax.fill_between(contour_k, contour_n, alpha=0.08, color=GPU_COLOR)

    # n = k reference
    k_range = np.logspace(np.log10(min(contour_k)), np.log10(max(contour_k)), 100)
    ax.plot(k_range, k_range, '--', color='gray', alpha=0.4, linewidth=1, zorder=2)
    ax.text(max(contour_k) * 0.5, max(contour_k) * 0.35, 'n = k',
            fontsize=9, color='gray', alpha=0.6, rotation=35)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Payout terms (k)', fontsize=13)
    ax.set_ylabel('Players (n)', fontsize=13)
    ax.set_title('1-Second Contour: ICM Equity on NVIDIA B200 (Q = 256)', fontsize=14)
    ax.grid(True, which='both', alpha=0.2)

    ax.set_xlim(min(contour_k) * 0.7, max(contour_k) * 1.5)
    ax.set_ylim(min(contour_n) * 0.5, max(contour_n) * 2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ─── Main ────────────────────────────────────────────────────

if __name__ == '__main__':
    serial_csv = os.path.join(ROOT, 'contour_zen4_serial_q256.csv')
    parallel_csv = os.path.join(ROOT, 'contour_zen4_parallel_q256.csv')
    bench_full = os.path.join(ROOT, 'bench_grid_full.txt')
    gpu_heatmap = os.path.join(ROOT, 'gpu_heatmap_new.csv')

    # CPU contour plots
    if os.path.exists(serial_csv) and os.path.exists(parallel_csv):
        plot_contour(serial_csv, parallel_csv, os.path.join(ROOT, 'contour_1s.png'))
        plot_speedup(serial_csv, parallel_csv, os.path.join(ROOT, 'parallel_speedup.png'))
        plot_dispatch(serial_csv, os.path.join(ROOT, 'engine_dispatch.png'))
    else:
        print(f"Skipping CPU contour plots (missing CSV files)")

    # CPU runtime vs n — include fixed k and ratio k (n/4, n/2, n)
    if os.path.exists(bench_full):
        data = extract_runtime_data_from_bench_grid(bench_full)
        if data:
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

            plot_runtime_vs_n(data_ext,
                              'Runtime vs n (Serial, Zen 4, Q = 256)',
                              os.path.join(ROOT, 'runtime_vs_n_cpu.png'),
                              k_values=fixed_k + ratio_labels)

    # GPU plots
    if os.path.exists(gpu_heatmap):
        plot_gpu_contour(gpu_heatmap, os.path.join(ROOT, 'gpu_contour.png'))

        gpu_data = extract_gpu_runtime_data(gpu_heatmap)
        if gpu_data:
            all_k = sorted(set(k for _, k, _, _ in gpu_data))
            # Pick representative fixed k values
            target_k = []
            for want in [2, 10, 50, 100]:
                closest = min(all_k, key=lambda x: abs(x - want))
                if closest not in target_k:
                    target_k.append(closest)

            # Add k=n/4, k=n/2, k=n as synthetic labels
            gpu_data_ext = list(gpu_data)
            for label, ratio in [('k=n/4', 4), ('k=n/2', 2), ('k=n', 1)]:
                pts = [(n, k, t, e) for n, k, t, e in gpu_data if k == n // ratio]
                if pts:
                    for n, k, t, e in pts:
                        gpu_data_ext.append((n, label, t, e))
                    target_k.append(label)

            plot_runtime_vs_n(gpu_data_ext,
                              'Runtime vs n (NVIDIA B200, Q = 256)',
                              os.path.join(ROOT, 'runtime_vs_n_gpu.png'),
                              k_values=target_k)

    print("\nDone.")
