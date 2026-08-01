#!/usr/bin/env python3
"""
plot_zen4_1dpc_2dpc.py -- Zen4 1DPC vs 2DPC contour comparison.

The Zen4 reference box was redeployed 2026-07-27 in a 2-DIMMs-per-channel
(2DPC) configuration, an AMD AM5 electrical limit that caps RAM at 3600 MT/s
against a higher DIMM rating (VERDICTS.md V9). Prior 1DPC-era data (full
memory bandwidth) is kept, not deleted, because the effect is not a flat
percentage: linear/schoolbook timings are nearly unchanged while hybrid/FFT
timings are 40-65% slower at large k under 2DPC.

This script overlays the two eras' serial contour curves on one plot so the
difference is visible directly, instead of only living in prose.

Usage:
  python3 tools/results/plot_zen4_1dpc_2dpc.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_common import load_contour, RESULTS

ONE_DPC = os.path.join(RESULTS, 'contour_zen4_serial_q256.csv')
TWO_DPC = os.path.join(RESULTS, 'contour_zen4_serial_q256_20260727.csv')


def main():
    k1, n1, _ = load_contour(ONE_DPC)
    k2, n2, _ = load_contour(TWO_DPC)

    order1 = k1.argsort()
    order2 = k2.argsort()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(k1[order1], n1[order1], marker='o', label='1DPC (full bandwidth, pre-2026-07-27)')
    ax.loglog(k2[order2], n2[order2], marker='s', label='2DPC (3600 MT/s ceiling, standing reference)')
    ax.set_xlabel('$k$ (target players)')
    ax.set_ylabel('max $n$ within 1s')
    ax.set_title('Zen 4: 1DPC vs 2DPC one-second contour (serial, $Q=256$)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    out_path = os.path.join(RESULTS, 'zen4_1dpc_vs_2dpc.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
