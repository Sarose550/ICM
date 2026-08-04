"""
Shared helpers for ICM benchmark plot scripts.

Extracted from tools/plot_contour.py: do not modify that file.
All per-chart scripts under tools/results/ import from here.
"""

import csv
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # repo root
RESULTS = os.path.join(ROOT, 'results')


def find_latest(pattern):
    """Return the most-recently-modified file in results/ matching pattern,
    or None if nothing matches. Lets callers point at a dated-file glob
    (e.g. 'bench_grid_zen4_serial_*.txt') instead of a single fixed name.

    glob.glob does NOT recurse into subdirectories, so files under
    results/evidence/ are naturally excluded.
    """
    matches = glob.glob(os.path.join(RESULTS, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


# ── Style ──────────────────────────────────────────────────────

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
# To add a new device, add an entry here. Filenames are glob patterns
# resolved against results/ via find_latest() -- the most recently
# modified match wins, so dated benchmark output files work without any
# renaming or copying.

DEVICE_CONFIGS = {
    'zen4': {
        'key': 'zen4',
        'label': 'Ryzen 9 7950X',
        'short': 'Zen 4',
        'n_cores': 16,
        'output_suffix': '',   # no suffix, root-level names like contour_1s.png
        'serial_csv': 'contour_zen4_serial_q256*.csv',
        'parallel_csv': 'contour_zen4_parallel_q256*.csv',
        'bench_grid': 'bench_grid_zen4_serial*.txt',
    },
    'm3_pro': {
        'key': 'm3_pro',
        'label': 'Apple M3 Pro',
        'short': 'M3 Pro',
        'n_cores': 12,
        'output_suffix': '_m3pro',
        'serial_csv': 'contour_m3pro_serial_q256*.csv',
        'parallel_csv': 'contour_m3pro_parallel_q256*.csv',
        'bench_grid': 'bench_grid_m3pro_serial*.txt',
    },
}


# ─── Data loading helpers ────────────────────────────────────

def _trim_uptick(k_arr, n_arr):
    """If the last point reverses the downward trend (n[-1] > n[-2]),
    return arrays without it so the connected line ends smoothly.
    Otherwise return the original arrays unchanged."""
    if len(n_arr) >= 2 and n_arr[-1] > n_arr[-2]:
        return k_arr[:-1], n_arr[:-1]
    return k_arr, n_arr


def load_contour(path, max_time_ms=2000):
    """Load contour CSV. Filter out 'ok' points where time > max_time_ms (these
    indicate anomalous bisection search behavior, not real 1s-budget data).
    Always includes the row where n_max <= k (the true n=k crossing) regardless
    of its measured time, then stops: floor-row probes intentionally use a
    generous timeout to confirm the crossing, so their time is expected to
    exceed max_time_ms and must not be filtered out."""
    k, n, engine = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ki, ni, ti = int(row['k']), int(row['n_max']), float(row['time_ms'])
            # Always keep the crossing row, regardless of time.
            if ni <= ki:
                k.append(ki)
                n.append(ni)
                engine.append(row['engine'])
                break
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
