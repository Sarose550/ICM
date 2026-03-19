#!/usr/bin/env python3
"""
plot_results.py — Generate publication-quality plots from ICM benchmark CSVs.

Reads:  accuracy_vs_q.csv, time_vs_q.csv, time_vs_n.csv,
        max_n_under_budget.csv, scaling.csv
Writes: accuracy_vs_q.png, time_vs_q.png, time_vs_n.png,
        max_n_under_budget.png, scaling_heatmap.png, summary.png

Requirements: pip install matplotlib pandas numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import sys

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox_inches': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'adversarial': '#e74c3c',
    'reverse_adv': '#3498db',
    'bimodal':     '#2ecc71',
    'geometric':   '#9b59b6',
    'uniform':     '#f39c12',
}
MARKERS = {
    'adversarial': 'o',
    'reverse_adv': 's',
    'bimodal':     'D',
    'geometric':   '^',
    'uniform':     'v',
}

def plot_accuracy_vs_q():
    """Plot 1: Error vs Q for each distribution (log-y). Overlays CPU + GPU."""
    has_gpu = os.path.exists('accuracy_vs_q.csv')
    has_cpu = os.path.exists('cpu_accuracy_vs_q.csv')
    if not has_gpu and not has_cpu:
        print("  Skipping: no accuracy_vs_q.csv found"); return

    fig, ax = plt.subplots(figsize=(9, 6))
    frames = []
    if has_cpu: frames.append(('CPU', pd.read_csv('cpu_accuracy_vs_q.csv'), '-'))
    if has_gpu: frames.append(('GPU', pd.read_csv('accuracy_vs_q.csv'), '--'))

    for label, df, ls in frames:
        for dist in df['distribution'].unique():
            sub = df[df['distribution'] == dist].sort_values('Q')
            lbl = f'{dist}' if len(frames) == 1 else f'{dist} ({label})'
            ax.semilogy(sub['Q'], sub['error'],
                        color=COLORS.get(dist, 'gray'),
                        marker=MARKERS.get(dist, 'o'),
                        markersize=5, linewidth=1.5, linestyle=ls, label=lbl)

    ax.axhline(1e-9, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(35, 1.5e-9, 'target: 1e-9', fontsize=8, color='gray')
    ax.axhline(5e-9, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    ax.text(35, 7e-9, 'logistic floor: 5e-9', fontsize=8, color='red', alpha=0.6)

    ax.set_xlabel('Quadrature nodes (Q)')
    ax.set_ylabel('Max relative V₁ error')
    ax.set_title('ICM Accuracy vs Quadrature Nodes (n=512, ratio=10⁹)')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8, ncol=2 if len(frames) > 1 else 1)
    ax.set_ylim(bottom=1e-16)
    fig.savefig('accuracy_vs_q.png')
    plt.close(fig)
    print("  -> accuracy_vs_q.png")


def plot_time_vs_q():
    """Plot 2: Kernel time vs Q for different n values. Overlays CPU + GPU."""
    has_gpu = os.path.exists('time_vs_q.csv')
    has_cpu = os.path.exists('cpu_time_vs_q.csv')
    if not has_gpu and not has_cpu:
        print("  Skipping: no time_vs_q.csv found"); return

    fig, ax = plt.subplots(figsize=(9, 6))
    n_colors = {512: '#3498db', 1024: '#2ecc71', 2048: '#e74c3c',
                4096: '#9b59b6', 8192: '#f39c12'}

    if has_cpu:
        df = pd.read_csv('cpu_time_vs_q.csv')
        time_col = 'avx512_ms' if 'avx512_ms' in df.columns else 'avx2_ms'
        for n_val in sorted(df['n'].unique()):
            sub = df[df['n'] == n_val].sort_values('Q')
            ax.plot(sub['Q'], sub[time_col],
                    color=n_colors.get(n_val, 'gray'),
                    marker='o', markersize=5, linewidth=1.5,
                    label=f'CPU n={n_val}')

    if has_gpu:
        df = pd.read_csv('time_vs_q.csv')
        for n_val in sorted(df['n'].unique()):
            sub = df[df['n'] == n_val].sort_values('Q')
            ax.plot(sub['Q'], sub['kernel_ms'],
                    color=n_colors.get(n_val, 'gray'),
                    marker='s', markersize=5, linewidth=1.5, linestyle='--',
                    label=f'GPU n={n_val}')

    ax.set_xlabel('Quadrature nodes (Q)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Compute Time vs Q (adversarial, ratio=10⁹)')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax.set_yscale('log')
    fig.savefig('time_vs_q.png')
    plt.close(fig)
    print("  -> time_vs_q.png")


def plot_time_vs_n():
    """Plot 3: Time + error vs n. Overlays CPU + GPU."""
    has_gpu = os.path.exists('time_vs_n.csv')
    has_cpu = os.path.exists('cpu_time_vs_n.csv')
    if not has_gpu and not has_cpu:
        print("  Skipping: no time_vs_n.csv found"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if has_cpu:
        df = pd.read_csv('cpu_time_vs_n.csv')
        time_col = 'avx512_ms' if 'avx512_ms' in df.columns else 'avx2_ms'
        impl_name = 'CPU avx512' if 'avx512_ms' in df.columns else 'CPU avx2'
        for dist in df['distribution'].unique():
            sub = df[df['distribution'] == dist].sort_values('n')
            ax1.loglog(sub['n'], sub[time_col],
                       color=COLORS.get(dist, 'gray'),
                       marker=MARKERS.get(dist, 'o'),
                       markersize=5, linewidth=1.5,
                       label=f'{dist} ({impl_name})')

    if has_gpu:
        df = pd.read_csv('time_vs_n.csv')
        for dist in df['distribution'].unique():
            sub = df[df['distribution'] == dist].sort_values('n')
            ax1.loglog(sub['n'], sub['kernel_ms'],
                       color=COLORS.get(dist, 'gray'),
                       marker=MARKERS.get(dist, 'o'),
                       markersize=5, linewidth=1.5, linestyle='--',
                       label=f'{dist} (GPU)')

    ax1.axhline(1.0, color='orange', linestyle=':', alpha=0.6, linewidth=1)
    ax1.text(40, 1.3, '1 ms', fontsize=8, color='orange')
    ax1.axhline(1000.0, color='red', linestyle=':', alpha=0.6, linewidth=1)
    ax1.text(40, 1300, '1 second', fontsize=8, color='red')
    ax1.set_xlabel('Number of players (n)')
    ax1.set_ylabel('Time (ms)')
    title = 'Compute Time vs n (Q=256, ratio=10⁹)'
    if has_cpu and has_gpu: title += '\nsolid=CPU, dashed=GPU'
    ax1.set_title(title)
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=7,
               ncol=2 if (has_cpu and has_gpu) else 1)

    src = pd.read_csv('cpu_time_vs_n.csv') if has_cpu else pd.read_csv('time_vs_n.csv')
    for dist in src['distribution'].unique():
        sub = src[src['distribution'] == dist].sort_values('n')
        ax2.semilogy(sub['n'], sub['error'],
                     color=COLORS.get(dist, 'gray'),
                     marker=MARKERS.get(dist, 'o'),
                     markersize=5, linewidth=1.5, label=dist)
    ax2.axhline(1e-9, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of players (n)')
    ax2.set_ylabel('Max relative V₁ error')
    ax2.set_title('Accuracy vs n (Q=256, ratio=10⁹)')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=8)

    fig.tight_layout()
    fig.savefig('time_vs_n.png')
    plt.close(fig)
    print("  -> time_vs_n.png")


def plot_max_n():
    """Plot 4: Maximum n under time budget. Overlays CPU + GPU."""
    has_gpu = os.path.exists('max_n_under_budget.csv')
    has_cpu = os.path.exists('cpu_max_n_under_budget.csv')
    if not has_gpu and not has_cpu:
        print("  Skipping: no max_n_under_budget.csv found"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: max n vs budget
    datasets = []
    if has_gpu: datasets.append(('GPU', pd.read_csv('max_n_under_budget.csv'), '#e74c3c', 'o'))
    if has_cpu: datasets.append(('CPU', pd.read_csv('cpu_max_n_under_budget.csv'), '#2c3e50', 's'))

    for label, df, color, marker in datasets:
        ax1.loglog(df['budget_ms'], df['max_n'], f'{marker}-',
                   color=color, markersize=7, linewidth=2, label=label)
        for _, row in df.iterrows():
            ax1.annotate(f"{int(row['max_n'])}",
                         xy=(row['budget_ms'], row['max_n']),
                         xytext=(5, 8), textcoords='offset points',
                         fontsize=6, color=color, alpha=0.7)

    ax1.axvline(1.0, color='orange', linestyle=':', alpha=0.6)
    ax1.axvline(1000.0, color='red', linestyle=':', alpha=0.6)
    ax1.set_xlabel('Time budget (ms)')
    ax1.set_ylabel('Maximum n (players)')
    ax1.set_title('Largest Tournament Computable Under Budget\n(Q=256, adversarial)')
    ax1.legend(framealpha=0.9)

    # Right: actual time vs max n
    for label, df, color, marker in datasets:
        ax2.loglog(df['max_n'], df['actual_ms'], f'{marker}-',
                   color=color, markersize=6, linewidth=1.5, label=f'{label} actual')
        ax2.loglog(df['max_n'], df['budget_ms'], f'{marker}--',
                   color=color, markersize=4, linewidth=1, alpha=0.4, label=f'{label} budget')
    ax2.set_xlabel('n (players)')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Actual Time at Maximum n')
    ax2.legend(framealpha=0.9, fontsize=8)

    fig.tight_layout()
    fig.savefig('max_n_under_budget.png')
    plt.close(fig)
    print("  -> max_n_under_budget.png")


def plot_summary():
    """Plot 5: Combined summary figure. Uses CPU or GPU data, whichever is available."""
    # Check for any data files (cpu_ or gpu prefix)
    acc_file = 'accuracy_vs_q.csv' if os.path.exists('accuracy_vs_q.csv') else \
               ('cpu_accuracy_vs_q.csv' if os.path.exists('cpu_accuracy_vs_q.csv') else None)
    time_n_file = 'time_vs_n.csv' if os.path.exists('time_vs_n.csv') else \
                  ('cpu_time_vs_n.csv' if os.path.exists('cpu_time_vs_n.csv') else None)
    max_n_file = 'max_n_under_budget.csv' if os.path.exists('max_n_under_budget.csv') else \
                 ('cpu_max_n_under_budget.csv' if os.path.exists('cpu_max_n_under_budget.csv') else None)
    scaling_file = 'scaling.csv' if os.path.exists('scaling.csv') else \
                   ('cpu_scaling.csv' if os.path.exists('cpu_scaling.csv') else None)

    if not any([acc_file, time_n_file, max_n_file, scaling_file]):
        print("  Skipping summary: no data files found"); return

    is_gpu = os.path.exists('time_vs_n.csv') or os.path.exists('scaling.csv')
    is_cpu = os.path.exists('cpu_time_vs_n.csv') or os.path.exists('cpu_scaling.csv')
    source = 'CPU + GPU' if (is_cpu and is_gpu) else ('GPU' if is_gpu else 'CPU')

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'ICM Generating Function Calculator — {source} Benchmark Summary',
                 fontsize=15, fontweight='bold', y=0.98)

    # Panel 1: Accuracy vs Q
    if acc_file:
        ax1 = fig.add_subplot(2, 2, 1)
        df = pd.read_csv(acc_file)
        for dist in df['distribution'].unique():
            sub = df[df['distribution'] == dist].sort_values('Q')
            ax1.semilogy(sub['Q'], sub['error'],
                         color=COLORS.get(dist, 'gray'),
                         marker=MARKERS.get(dist, 'o'),
                         markersize=4, linewidth=1.2, label=dist)
        ax1.axhline(1e-9, color='gray', linestyle='--', alpha=0.4, linewidth=0.7)
        ax1.set_xlabel('Q'); ax1.set_ylabel('Error')
        ax1.set_title('Accuracy vs Q (n=512)')
        ax1.legend(fontsize=7, framealpha=0.8)
        ax1.set_ylim(bottom=1e-16)

    # Panel 2: Time vs n — overlay CPU + GPU if both present
    if time_n_file or (os.path.exists('cpu_time_vs_n.csv') and os.path.exists('time_vs_n.csv')):
        ax2 = fig.add_subplot(2, 2, 2)
        if os.path.exists('cpu_time_vs_n.csv'):
            df = pd.read_csv('cpu_time_vs_n.csv')
            col = 'avx512_ms' if 'avx512_ms' in df.columns else 'avx2_ms'
            lbl = 'CPU'
            for dist in df['distribution'].unique():
                sub = df[df['distribution'] == dist].sort_values('n')
                ax2.loglog(sub['n'], sub[col], color=COLORS.get(dist, 'gray'),
                           marker=MARKERS.get(dist, 'o'), markersize=4, linewidth=1.2,
                           label=f'{dist} ({lbl})')
        if os.path.exists('time_vs_n.csv'):
            df = pd.read_csv('time_vs_n.csv')
            for dist in df['distribution'].unique():
                sub = df[df['distribution'] == dist].sort_values('n')
                ax2.loglog(sub['n'], sub['kernel_ms'], color=COLORS.get(dist, 'gray'),
                           marker=MARKERS.get(dist, 'o'), markersize=4, linewidth=1.2,
                           linestyle='--', label=f'{dist} (GPU)')
        ax2.axhline(1.0, color='orange', linestyle=':', alpha=0.5)
        ax2.axhline(1000, color='red', linestyle=':', alpha=0.5)
        ax2.set_xlabel('n'); ax2.set_ylabel('Time (ms)')
        ax2.set_title('Time vs n (Q=256)')
        ax2.legend(fontsize=6, framealpha=0.8, ncol=2)

    # Panel 3: Max n under budget — overlay CPU + GPU
    has_gpu_max = os.path.exists('max_n_under_budget.csv')
    has_cpu_max = os.path.exists('cpu_max_n_under_budget.csv')
    if has_gpu_max or has_cpu_max:
        ax3 = fig.add_subplot(2, 2, 3)
        if has_gpu_max:
            df = pd.read_csv('max_n_under_budget.csv')
            ax3.loglog(df['budget_ms'], df['max_n'], 'o-', color='#e74c3c',
                       markersize=5, linewidth=2, label='GPU')
        if has_cpu_max:
            df = pd.read_csv('cpu_max_n_under_budget.csv')
            ax3.loglog(df['budget_ms'], df['max_n'], 's-', color='#2c3e50',
                       markersize=5, linewidth=2, label='CPU')
        ax3.set_xlabel('Budget (ms)'); ax3.set_ylabel('Max n')
        ax3.set_title('Max Tournament Size Under Budget')
        ax3.legend(framealpha=0.9)

    # Panel 4: Scaling heatmap
    if scaling_file:
        ax4 = fig.add_subplot(2, 2, 4)
        df = pd.read_csv(scaling_file)
        time_col = 'kernel_ms' if 'kernel_ms' in df.columns else \
                   ('avx512_ms' if 'avx512_ms' in df.columns else 'avx2_ms')
        adv = df[df['distribution'] == 'adversarial']
        if len(adv) > 0:
            pivot = adv.pivot_table(index='n', columns='Q', values=time_col)
            pivot = pivot.sort_index(ascending=False)
            im = ax4.imshow(pivot.values, aspect='auto',
                           cmap='YlOrRd', interpolation='nearest')
            ax4.set_xticks(range(len(pivot.columns)))
            ax4.set_xticklabels([str(c) for c in pivot.columns])
            ax4.set_yticks(range(len(pivot.index)))
            ax4.set_yticklabels([str(i) for i in pivot.index])
            ax4.set_xlabel('Q'); ax4.set_ylabel('n')
            ax4.set_title(f'Time (ms) — adversarial ({source})')
            for yi, n_val in enumerate(pivot.index):
                for xi, q_val in enumerate(pivot.columns):
                    val = pivot.iloc[yi, xi]
                    if not np.isnan(val):
                        color = 'white' if val > pivot.values.max() * 0.6 else 'black'
                        ax4.text(xi, yi, f'{val:.1f}', ha='center', va='center',
                                fontsize=8, color=color, fontweight='bold')
            fig.colorbar(im, ax=ax4, shrink=0.8, label='ms')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig('summary.png')
    plt.close(fig)
    print("  -> summary.png")


def main():
    print("Generating plots from benchmark CSVs...\n")
    plot_accuracy_vs_q()
    plot_time_vs_q()
    plot_time_vs_n()
    plot_max_n()
    plot_summary()
    print("\nDone. Generated PNG files in current directory.")


if __name__ == '__main__':
    main()
