#!/usr/bin/env python3
"""
Generate accuracy_convergence.png: Quadrature accuracy vs Q (log-log).

Usage:
  python3 tools/results/plot_accuracy_fig.py
  python3 tools/results/plot_accuracy_fig.py --csv results/accuracy_convergence.csv
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_common import (  # noqa: E402
    RESULTS, find_latest, load_accuracy_csv,
)


def plot_accuracy_convergence(accuracy_path, out_path):
    """Plot max_rel_err vs Q: left panel Gauss vs tanh-sinh (uniform, various n),
    right panel Gauss at n=20 across all 4 stack distributions."""
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

    # Gauss-Legendre, V1, n=20, across all 4 stack distributions
    dist_order = ['uniform', 'adversarial', 'geometric', 'adv_1e9']
    dist_labels = {'uniform': 'uniform', 'adversarial': 'adversarial (1e4)',
                   'geometric': 'geometric (5e5)', 'adv_1e9': 'extreme (1e9)'}
    dist_data = {}  # distribution -> [(Q, err), ...]
    for r in rows:
        if (r['scheme'] == 'gauss' and r['payout_type'] == 'V1'
                and r['n'] == 20 and r['distribution'] in dist_order):
            dist_data.setdefault(r['distribution'], []).append((r['Q'], r['max_rel_err']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Left panel: Gauss vs tanh-sinh, uniform stacks, various n ---
    gauss_colors = ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626']

    for i, n in enumerate(gauss_n_values):
        if n not in gauss_data:
            continue
        pts = sorted(gauss_data[n])
        qs = [p[0] for p in pts]
        errs = [p[1] for p in pts]
        ax1.plot(qs, errs, 'o-', color=gauss_colors[i], markersize=5, linewidth=1.5,
                 label=f'Gauss n={n}')

    if tanh_data:
        pts = sorted(tanh_data)
        qs = [p[0] for p in pts]
        errs = [p[1] for p in pts]
        ax1.plot(qs, errs, 's--', color='#6b7280', markersize=5, linewidth=1.5,
                 label='tanh-sinh n=10')

    ax1.axhline(y=1e-12, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax1.text(5, 1.5e-12, 'practical floor (~1e-12)', fontsize=8, color='gray', alpha=0.7)

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Quadrature points (Q)', fontsize=13)
    ax1.set_ylabel('Max relative error', fontsize=13)
    ax1.set_title('Gauss-Legendre vs tanh-sinh (uniform stacks)', fontsize=13)
    ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax1.grid(True, which='both', alpha=0.2)

    # --- Right panel: Gauss, n=20, across all 4 stack distributions ---
    dist_colors = {'uniform': '#2563eb', 'adversarial': '#059669',
                   'geometric': '#d97706', 'adv_1e9': '#dc2626'}

    for dist in dist_order:
        if dist not in dist_data:
            continue
        pts = sorted(dist_data[dist])
        qs = [p[0] for p in pts]
        errs = [p[1] for p in pts]
        ax2.plot(qs, errs, 'o-', color=dist_colors[dist], markersize=5, linewidth=1.5,
                 label=dist_labels[dist])

    ax2.axvline(x=128, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax2.axhline(y=1e-12, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Quadrature points (Q)', fontsize=13)
    ax2.set_ylabel('Max relative error', fontsize=13)
    ax2.set_title('Gauss-Legendre, n=20, across stack distributions', fontsize=13)
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax2.grid(True, which='both', alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate accuracy-convergence plot (Q vs error, log-log).')
    parser.add_argument('--csv', default=None,
                        help='Path to accuracy_convergence.csv (auto-detected if omitted)')
    parser.add_argument('--out', default=None,
                        help='Output PNG path (default: results/accuracy_convergence.png)')
    args = parser.parse_args()

    accuracy_csv = args.csv or os.path.join(RESULTS, 'accuracy_convergence.csv')
    if not os.path.exists(accuracy_csv):
        print(f"Missing {accuracy_csv}, cannot generate accuracy plot.")
        sys.exit(1)

    out_path = args.out or os.path.join(RESULTS, 'accuracy_convergence.png')
    plot_accuracy_convergence(accuracy_csv, out_path)
