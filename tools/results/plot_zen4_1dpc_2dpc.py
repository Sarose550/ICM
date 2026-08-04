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

# Both inputs are pinned to explicit, non-current filenames ON PURPOSE.
#
# 1. The 1DPC data used to live in the *undated* `contour_zen4_serial_q256.csv`,
#    which collides with this project's "undated file = current data" rule:
#    `tools/results/refresh_all.sh --device zen4` writes exactly that name, so a
#    routine refresh would have silently turned this figure into a 2DPC-vs-2DPC
#    plot with no error anywhere. It is now `..._1dpc.csv`.
# 2. The 2DPC side deliberately stays on the dated 2026-07-27 snapshot rather
#    than the current undated file, so the comparison isolates the ONE variable
#    it claims to isolate (memory configuration). Newer Zen4 data was taken
#    after the wrap-safety-margin fix and a calibration-ceiling extension, both
#    of which change dispatch decisions; pairing it against pre-fix 1DPC data
#    would conflate a hardware change with a code change.
# THREE machines, not two. M3 is a second box of the same nominal 2DPC
# configuration as M2, measured later; it tracks M1 (1DPC) rather than M2,
# which is why this plot can no longer be captioned as a bandwidth experiment.
M1_1DPC = os.path.join(RESULTS, 'contour_zen4_serial_q256_1dpc.csv')
M2_2DPC = os.path.join(RESULTS, 'contour_zen4_serial_q256_20260727.csv')
M3_2DPC = os.path.join(RESULTS, 'contour_zen4_serial_q256.csv')  # current box


def main():
    k1, n1, _ = load_contour(M1_1DPC)
    k2, n2, _ = load_contour(M2_2DPC)

    order1 = k1.argsort()
    order2 = k2.argsort()

    fig, ax = plt.subplots(figsize=(7.0, 5))
    ax.loglog(k1[order1], n1[order1], marker='o', label='M1: 1DPC (full rated bandwidth)')
    ax.loglog(k2[order2], n2[order2], marker='s', label='M2: 2DPC (3600 MT/s)')

    if os.path.exists(M3_2DPC):
        k3, n3, _ = load_contour(M3_2DPC)
        order3 = k3.argsort()
        ax.loglog(k3[order3], n3[order3], marker='^', linestyle='--',
                  label='M3: 2DPC (3600 MT/s), different machine')

    ax.set_xlabel('$k$ (target players)')
    ax.set_ylabel('max $n$ within 1s')
    ax.set_title('Zen 4 one-second contour, three machines (serial, $Q=256$)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    out_path = os.path.join(RESULTS, 'zen4_1dpc_vs_2dpc.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
