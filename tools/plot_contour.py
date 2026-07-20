#!/usr/bin/env python3
"""
Generate publication plots for ICM benchmark data.

Plots generated:
  1. contour_1s[_<device>].png      — Serial vs parallel 1-second boundary (CPU)
  2. parallel_speedup[_<device>].png — Speedup bar chart
  3. engine_dispatch[_<device>].png  — Engine coloring on contour
  4. runtime_vs_n_cpu[_<device>].png — Runtime(n) at fixed k values, log-log (CPU)
  5. runtime_vs_n_gpu.png           — Runtime(n) at fixed k values, log-log (GPU)
  6. gpu_contour.png                — GPU 1-second contour from heatmap data
  7. accuracy_convergence.png       — Quadrature accuracy vs Q (log-log)

Usage:
  python3 tools/plot_contour.py                        # defaults to zen4
  python3 tools/plot_contour.py --device zen4           # Zen4 (Ryzen 9 7950X)
  python3 tools/plot_contour.py --device m3_pro         # M3 Pro

Reads from project root:
  contour_<device>_serial_q256.csv, contour_<device>_parallel_q256.csv,
  bench_grid_<device>_serial.txt, gpu_heatmap_new.csv, accuracy_zen4.csv
"""

import argparse
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

# ─── Device configuration ───────────────────────────────────

# Each device has a key matching the --device flag value.
# To add a new device, add an entry here and place the corresponding
# contour_<key>_serial_q256.csv, contour_<key>_parallel_q256.csv, and
# bench_grid_<key>_serial.txt files at the repo root.

DEVICE_CONFIGS = {
    'zen4': {
        'key': 'zen4',
        'label': 'Ryzen 9 7950X',
        'short': 'Zen 4',
        'n_cores': 16,
        'output_suffix': '',   # no suffix — root-level names like contour_1s.png
        'serial_csv': 'contour_zen4_serial_q256.csv',
        'parallel_csv': 'contour_zen4_parallel_q256.csv',
        'bench_grid': 'bench_grid_zen4_serial.txt',
    },
    'm3_pro': {
        'key': 'm3_pro',
        'label': 'Apple M3 Pro',
        'short': 'M3 Pro',
        'n_cores': 12,
        'output_suffix': '_m3pro',
        'serial_csv': 'contour_m3pro_serial_q256.csv',
        'parallel_csv': 'contour_m3pro_parallel_q256.csv',
        'bench_grid': 'bench_grid_m3pro_serial.txt',
    },
}

# ─── Data loading ────────────────────────────────────────────

def load_contour(path, max_time_ms=2000):
    """Load contour CSV. Filter out points where time > max_time_ms.
    Keeps all 'ok' rows plus at most the first 'floor' row, then stops.
    This trims degenerate trailing floor rows from older data that predate
    the contour_1s.c fix (which now breaks after the first floor)."""
    k, n, engine = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ki, ni, ti = int(row['k']), int(row['n_max']), float(row['time_ms'])
            if ti > max_time_ms:
                continue
            status = row.get('status', 'ok')
            k.append(ki)
            n.append(ni)
            engine.append(row['engine'])
            if status == 'floor':
                break  # one floor row is enough — stop reading
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

def plot_contour(cfg, serial_path, parallel_path, out_path):
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

    # n = k reference line (prominent, full range)
    all_vals = np.concatenate([ks, ns, kp, np_])
    lo, hi = min(all_vals) * 0.5, max(all_vals) * 2
    k_diag = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.plot(k_diag, k_diag, '--', color='#666666', alpha=0.6, linewidth=1.5, zorder=2)
    # Label near the middle of visible range
    mid = np.sqrt(lo * hi)
    ax.text(mid * 0.8, mid * 0.55, 'n = k', fontsize=10, color='#666666',
            alpha=0.7, rotation=38)

    # Mark intersection of n=k line with contours (1-second k=n threshold)
    for label, k_arr, n_arr, color in [('serial', ks, ns, SERIAL_COLOR),
                                        ('parallel', kp, np_, PARALLEL_COLOR)]:
        # Find where contour crosses n=k (interpolate)
        for i in range(len(k_arr) - 1):
            # contour: n_arr[i] at k_arr[i]. n=k line: n=k.
            # crossing when n_arr goes from > k to < k
            if n_arr[i] >= k_arr[i] and n_arr[i+1] < k_arr[i+1]:
                # Linear interp in log space
                frac = (np.log(k_arr[i]) - np.log(n_arr[i])) / \
                       ((np.log(n_arr[i+1]) - np.log(n_arr[i])) - (np.log(k_arr[i+1]) - np.log(k_arr[i])))
                k_cross = np.exp(np.log(k_arr[i]) + frac * (np.log(k_arr[i+1]) - np.log(k_arr[i])))
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

    ax.set_xlim(1.5, max(ks) * 1.5)
    y_lo = min(min(ns), min(np_)) * 0.5
    y_hi = max(max(ns), max(np_)) * 2
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, which='both', alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ─── Plot 2: Parallel speedup ───────────────────────────────

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


# ─── Plot 3: Engine dispatch map ─────────────────────────────

def plot_dispatch(cfg, serial_path, out_path):
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
        # Filter out zero/negative times (break log scale)
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

    fig, ax = plt.subplots(figsize=(8, 8.5))
    ax.plot(contour_k, contour_n, 'o-', color=GPU_COLOR, markersize=6, linewidth=2)
    ax.fill_between(contour_k, contour_n, alpha=0.08, color=GPU_COLOR)

    xlo, xhi = min(contour_k) * 0.7, max(contour_k) * 1.5
    ylo, yhi = min(contour_n) * 0.7, max(contour_n) * 1.5

    # n = k reference line, clipped to the actual view so bbox_inches='tight'
    # can't be dragged into including off-screen geometry
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
        if contour_n[i] >= contour_k[i] and contour_n[i+1] < contour_k[i+1]:
            frac = (np.log(contour_k[i]) - np.log(contour_n[i])) / \
                   ((np.log(contour_n[i+1]) - np.log(contour_n[i])) - (np.log(contour_k[i+1]) - np.log(contour_k[i])))
            k_cross = np.exp(np.log(contour_k[i]) + frac * (np.log(contour_k[i+1]) - np.log(contour_k[i])))
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


# ─── Plot 7: Accuracy convergence ──────────────────────────

def load_accuracy_csv(path):
    """Load accuracy_zen4.csv into list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'scheme': row['scheme'],
                'n': int(row['n']),
                'k': int(row['k']),
                'Q': int(row['Q']),
                'max_abs_err': float(row['max_abs_err']),
                'max_rel_err': float(row['max_rel_err']),
                'payout_type': row['payout_type'],
                'distribution': row['distribution'],
            })
    return rows


def plot_accuracy_convergence(accuracy_path, out_path):
    """Plot max_rel_err vs Q for Gauss-Legendre at various n, plus tanh-sinh comparison."""
    rows = load_accuracy_csv(accuracy_path)

    # Gauss-Legendre, V1, uniform for selected n values
    gauss_n_values = [4, 8, 12, 16, 20]
    gauss_data = {}  # n -> [(Q, err), ...]
    for r in rows:
        if (r['scheme'] == 'gauss' and r['distribution'] == 'uniform'
                and r['payout_type'] == 'V1' and r['n'] in gauss_n_values):
            gauss_data.setdefault(r['n'], []).append((r['Q'], r['max_rel_err']))

    # tanh-sinh, n=10, V1, uniform
    tanh_data = []
    for r in rows:
        if (r['scheme'] == 'tanh_sinh' and r['distribution'] == 'uniform'
                and r['payout_type'] == 'V1' and r['n'] == 10):
            tanh_data.append((r['Q'], r['max_rel_err']))

    fig, ax = plt.subplots(figsize=(9, 6))

    # Color palette for Gauss lines
    gauss_colors = ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626']

    for i, n in enumerate(gauss_n_values):
        if n not in gauss_data:
            continue
        pts = sorted(gauss_data[n])
        qs = [p[0] for p in pts]
        errs = [p[1] for p in pts]
        ax.plot(qs, errs, 'o-', color=gauss_colors[i], markersize=5, linewidth=1.5,
                label=f'Gauss n={n}')

    # tanh-sinh comparison
    if tanh_data:
        pts = sorted(tanh_data)
        qs = [p[0] for p in pts]
        errs = [p[1] for p in pts]
        ax.plot(qs, errs, 's--', color='#6b7280', markersize=5, linewidth=1.5,
                label='tanh-sinh n=10')

    # Machine epsilon practical floor
    q_range = ax.get_xlim()
    ax.axhline(y=1e-12, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(5, 1.5e-12, 'practical floor (~1e-12)', fontsize=8, color='gray', alpha=0.7)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Quadrature points (Q)', fontsize=13)
    ax.set_ylabel('Max relative error', fontsize=13)
    ax.set_title('Quadrature Convergence: Gauss-Legendre vs tanh-sinh', fontsize=14)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ─── Main ────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ICM benchmark plots.')
    parser.add_argument('--device', choices=['zen4', 'm3_pro'], default='zen4',
                        help='Device to generate plots for (default: zen4)')
    args = parser.parse_args()

    cfg = DEVICE_CONFIGS[args.device]
    suffix = cfg['output_suffix']

    serial_csv = os.path.join(ROOT, cfg['serial_csv'])
    parallel_csv = os.path.join(ROOT, cfg['parallel_csv'])
    bench_full = os.path.join(ROOT, cfg['bench_grid'])
    gpu_heatmap = os.path.join(ROOT, 'gpu_heatmap_new.csv')

    # CPU contour plots
    if os.path.exists(serial_csv) and os.path.exists(parallel_csv):
        plot_contour(cfg, serial_csv, parallel_csv,
                     os.path.join(ROOT, f'contour_1s{suffix}.png'))
        plot_speedup(cfg, serial_csv, parallel_csv,
                     os.path.join(ROOT, f'parallel_speedup{suffix}.png'))
        plot_dispatch(cfg, serial_csv,
                      os.path.join(ROOT, f'engine_dispatch{suffix}.png'))
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
                              f'Runtime vs n (Serial, {cfg["short"]}, Q = 256)',
                              os.path.join(ROOT, f'runtime_vs_n_cpu{suffix}.png'),
                              k_values=fixed_k + ratio_labels)

    # GPU plots (device-independent — only run in default zen4 mode
    # to avoid overwriting the canonical GPU plots with duplicate runs)
    if args.device == 'zen4':
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

        # Accuracy convergence
        accuracy_csv = os.path.join(ROOT, 'accuracy_zen4.csv')
        if os.path.exists(accuracy_csv):
            plot_accuracy_convergence(accuracy_csv, os.path.join(ROOT, 'accuracy_convergence.png'))
        else:
            print("Skipping accuracy_convergence.png (missing accuracy_zen4.csv)")
    else:
        print("Skipping GPU/accuracy plots (non-zen4 device — only generated for zen4)")

    print("\nDone.")
