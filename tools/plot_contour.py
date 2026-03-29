#!/usr/bin/env python3
"""
Plot 1-second contour and parallel speedup for ICM paper.

Generates:
  1. contour_1s.png      — Serial vs parallel (corrected) 1-second boundary
  2. parallel_speedup.png — Speedup bar chart
  3. engine_dispatch.png  — Engine coloring on serial contour
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# ─── Style ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
})

SERIAL_COLOR = '#2563eb'   # blue
PARALLEL_COLOR = '#dc2626' # red
LINEAR_COLOR = '#9333ea'   # purple
HYBRID_COLOR = '#059669'   # green
PAR_LINEAR_COLOR = '#f59e0b' # amber for parallel-linear points

# ─── Load CSV data ───────────────────────────────────────────

def load_contour(path, max_time_ms=2000):
    k, n, engine = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ki, ni, ti = int(row['k']), int(row['n_max']), float(row['time_ms'])
            if ti > max_time_ms or (ki >= 100000 and ti < 500):
                continue
            k.append(ki)
            n.append(ni)
            engine.append(row['engine'])
    return np.array(k), np.array(n), engine

ks, ns, es = load_contour(os.path.join(ROOT, 'contour_zen4_serial_q256.csv'))
kp, np_, ep = load_contour(os.path.join(ROOT, 'contour_zen4_parallel_q256.csv'))

# ─── Plot 1: 1-second contour ───────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6.5))

# Fill regions
ax.fill_between(ks, ns, alpha=0.06, color=SERIAL_COLOR)
ax.fill_between(kp, np_, alpha=0.06, color=PARALLEL_COLOR)

# Serial — single line with markers by engine
s_lin_k = [k for k, e in zip(ks, es) if e == 'linear']
s_lin_n = [n for n, e in zip(ns, es) if e == 'linear']
s_hyb_k = [k for k, e in zip(ks, es) if e == 'hybrid']
s_hyb_n = [n for n, e in zip(ns, es) if e == 'hybrid']

ax.plot(ks, ns, '-', color=SERIAL_COLOR, linewidth=2, alpha=0.6, zorder=4)
ax.plot(s_lin_k, s_lin_n, 'o', color=SERIAL_COLOR, markersize=7, zorder=5)
ax.plot(s_hyb_k, s_hyb_n, 's', color=SERIAL_COLOR, markersize=6, zorder=5)

# Parallel — split by engine
p_lin_k = [k for k, e in zip(kp, ep) if e == 'linear']
p_lin_n = [n for n, e in zip(np_, ep) if e == 'linear']
p_hyb_k = [k for k, e in zip(kp, ep) if e == 'hybrid']
p_hyb_n = [n for n, e in zip(np_, ep) if e == 'hybrid']

ax.plot(kp, np_, '-', color=PARALLEL_COLOR, linewidth=2, alpha=0.6, zorder=4)
ax.plot(p_lin_k, p_lin_n, 'o', color=PARALLEL_COLOR, markersize=7, zorder=5)
ax.plot(p_hyb_k, p_hyb_n, 's', color=PARALLEL_COLOR, markersize=6, zorder=5)

# Engine crossover annotation (serial)
for i in range(len(es)-1):
    if es[i] == 'linear' and es[i+1] == 'hybrid':
        cross_k = (ks[i] + ks[i+1]) / 2
        ax.axvline(x=cross_k, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        ax.text(cross_k * 1.1, 4.2e7, 'linear | hybrid', fontsize=8,
                color='gray', rotation=0, ha='left', va='top')
        break

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Payout terms (k)', fontsize=13)
ax.set_ylabel('Players (n)', fontsize=13)
ax.set_title('1-Second Contour: ICM Equity on Ryzen 9 7950X\n'
             'AOCL-FFTW 3.3.10 + AVX-512 PATIENT, Q = 256',
             fontsize=14, fontweight='bold')

# Reference lines
ax.axhline(y=1e6, color='gray', linestyle='--', alpha=0.25, linewidth=0.8)
ax.text(2.2, 1.1e6, '1M players', fontsize=8, color='gray', alpha=0.5)
ax.axhline(y=1e5, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
ax.text(2.2, 1.1e5, '100K players', fontsize=8, color='gray', alpha=0.4)

legend_elements = [
    Line2D([0], [0], color=SERIAL_COLOR, marker='o', markersize=7, linewidth=2,
           label='Serial (1 core)'),
    Line2D([0], [0], color=PARALLEL_COLOR, marker='o', markersize=7, linewidth=2,
           label='Parallel (16 cores)'),
    Line2D([0], [0], color='gray', marker='o', markersize=6, linestyle='none',
           label='Linear engine (serial)'),
    Line2D([0], [0], color='gray', marker='s', markersize=6, linestyle='none',
           label='Hybrid engine'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9.5, framealpha=0.9)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1.5, 30000)
ax.set_ylim(2e4, 5e7)
ax.grid(True, which='both', alpha=0.2)

fig.tight_layout()
fig.savefig(os.path.join(ROOT, 'contour_1s.png'), dpi=200, bbox_inches='tight')
print("Saved contour_1s.png")

# ─── Plot 2: Parallel speedup (corrected) ───────────────────

common_k = sorted(set(ks) & set(kp))
s_dict = dict(zip(ks, ns))
p_dict = dict(zip(kp, np_))
s_eng = {k: e for k, e in zip(ks, es)}
p_eng = {k: e for k, e in zip(kp, ep)}
speedups = [p_dict[k] / s_dict[k] for k in common_k]

fig2, ax2 = plt.subplots(figsize=(9, 5.5))

# Color by whether parallel used linear (bandwidth-limited) or hybrid (compute-parallel)
colors = []
for k in common_k:
    if p_eng.get(k) == 'linear':
        colors.append(PAR_LINEAR_COLOR)  # parallel linear = bandwidth-limited
    else:
        colors.append(HYBRID_COLOR)

bars = ax2.bar(range(len(common_k)), speedups, color=colors, alpha=0.85,
               edgecolor='white', linewidth=0.5)

ax2.set_xticks(range(len(common_k)))
ax2.set_xticklabels([f'{k:,}' if k < 1000 else f'{k//1000}K' for k in common_k],
                     rotation=45, ha='right', fontsize=9)
ax2.set_xlabel('Payout terms (k)', fontsize=12)
ax2.set_ylabel('Parallel speedup (n_parallel / n_serial)', fontsize=12)
ax2.set_title('16-Core Parallel Speedup at 1-Second Boundary\n'
              'Ryzen 9 7950X, AOCL-FFTW, Q = 256',
              fontsize=12, fontweight='bold')

ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax2.set_ylim(0, max(speedups) * 1.15)

ax2.legend(handles=[
    Patch(facecolor=PAR_LINEAR_COLOR, alpha=0.85,
          label='Linear engine (memory-bandwidth-limited)'),
    Patch(facecolor=HYBRID_COLOR, alpha=0.85,
          label='Hybrid engine (compute-parallel)'),
], loc='upper left', fontsize=9.5)

for i, (v, k) in enumerate(zip(speedups, common_k)):
    ax2.text(i, v + 0.02 * max(speedups), f'{v:.1f}x',
             ha='center', fontsize=8, fontweight='bold')

fig2.tight_layout()
fig2.savefig(os.path.join(ROOT, 'parallel_speedup.png'), dpi=200, bbox_inches='tight')
print("Saved parallel_speedup.png")

# ─── Plot 3: Engine dispatch map ─────────────────────────────

fig3, ax3 = plt.subplots(figsize=(10, 6))

for k_val, n_val, eng in zip(ks, ns, es):
    color = LINEAR_COLOR if eng == 'linear' else HYBRID_COLOR
    ax3.scatter(k_val, n_val, c=color, s=80, zorder=5, edgecolors='white', linewidth=0.5)

ax3.plot(ks, ns, '-', color='gray', linewidth=1, alpha=0.5, zorder=3)

lin_mask = np.array([e == 'linear' for e in es])
hyb_mask = ~lin_mask

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Payout terms (k)', fontsize=13)
ax3.set_ylabel('Maximum players (n) in 1 second', fontsize=13)
ax3.set_title('Engine Dispatch at 1-Second Boundary (Serial)\n'
              'O(nk) linear vs O(n log\u00b2k) hybrid',
              fontsize=14, fontweight='bold')

if any(lin_mask):
    ax3.annotate('Linear O(nk)\nn ~ 1/k', xy=(ks[lin_mask][-1], ns[lin_mask][-1]),
                xytext=(15, 8e6), fontsize=11, color=LINEAR_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=LINEAR_COLOR, lw=1.2))
if any(hyb_mask):
    mid = len(ks[hyb_mask]) // 2
    ax3.annotate('Hybrid O(n log\u00b2k)\nplateau ~28K\u2013172K',
                xy=(ks[hyb_mask][mid], ns[hyb_mask][mid]),
                xytext=(5000, 5e5), fontsize=11, color=HYBRID_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=HYBRID_COLOR, lw=1.2))

ax3.legend(handles=[
    Line2D([0], [0], color=LINEAR_COLOR, marker='o', markersize=8, linestyle='none',
           label='Linear engine'),
    Line2D([0], [0], color=HYBRID_COLOR, marker='o', markersize=8, linestyle='none',
           label='Hybrid engine (B=16\u201324)'),
], loc='upper right', fontsize=10, framealpha=0.9)

ax3.set_xlim(1.5, 30000)
ax3.set_ylim(2e4, 5e7)
ax3.grid(True, which='both', alpha=0.2)

fig3.tight_layout()
fig3.savefig(os.path.join(ROOT, 'engine_dispatch.png'), dpi=200, bbox_inches='tight')
print("Saved engine_dispatch.png")

print("\nDone! Generated 3 plots.")
