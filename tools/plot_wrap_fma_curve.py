#!/usr/bin/env python3
"""Generate results/wrap_fma_cost_curve.png from results/wrap_fma_bench_zen4.csv.

Run from the repo root: python3 tools/plot_wrap_fma_curve.py
"""
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(REPO_ROOT, "results", "wrap_fma_bench_zen4.csv")
out_path = os.path.join(REPO_ROOT, "results", "wrap_fma_cost_curve.png")

# Read data, keep only SMALL_2048 regime
data = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['regime'] == 'SMALL_2048':
            wrap_m = int(row['wrap_m'])
            fma_count = int(row['fma_count'])
            ns = float(row['median_ns_per_call'])
            data.append((wrap_m, fma_count, ns))

data.sort(key=lambda x: x[0])
wrap_ms = np.array([d[0] for d in data])
fma_counts = np.array([d[1] for d in data])
ns_vals = np.array([d[2] for d in data])

# Marginal cost: ns/FMA. But small wrap_m has overhead contamination.
# Use local slope between consecutive points for the marginal cost.
# Or just compute ns/fma_count as the apparent cost.
ns_per_fma = ns_vals / fma_counts

# Compute local slopes (finite differences) - marginal cost
marginal = []
mid_wrap = []
for i in range(1, len(data)):
    dw = fma_counts[i] - fma_counts[i-1]
    dn = ns_vals[i] - ns_vals[i-1]
    if dw > 0:
        marginal.append(dn / dw)
        mid_wrap.append((wrap_ms[i] + wrap_ms[i-1]) / 2)

marginal = np.array(marginal)
mid_wrap = np.array(mid_wrap)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: raw ns/FMA
ax1.semilogx(wrap_ms, ns_per_fma, 'o-', color='#2166ac', markersize=4)
ax1.axhline(y=0.40, color='#b2182b', linestyle='--', alpha=0.7, label=r'Production $\mathrm{WRAP\_FMA\_NS}=0.40$')
ax1.set_xlabel('wrap_m')
ax1.set_ylabel('Apparent ns/FMA')
ax1.set_title('Raw ns-per-FMA (includes overhead)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Right: marginal cost (local slope)
ax2.semilogx(mid_wrap, marginal, 's-', color='#2166ac', markersize=4)
ax2.axhline(y=0.40, color='#b2182b', linestyle='--', alpha=0.7, label=r'Production $\mathrm{WRAP\_FMA\_NS}=0.40$')
ax2.axvspan(64, 384, alpha=0.1, color='green', label='Operating window [64, 384]')
ax2.set_xlabel('wrap_m (midpoint)')
ax2.set_ylabel('Marginal ns/FMA (local slope)')
ax2.set_title('Marginal cost (slope, no overhead)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

fig.suptitle('Wrap-Correction FMA Cost vs. Working-Set Size (Zen 4)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
