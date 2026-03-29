#!/usr/bin/env python3
import csv
import math
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                n = int(r["n"])
                k = int(r["k"])
                t = float(r["time_ms"])
                b = int(r["B"])
                eng = r["engine"]
                tier = r.get("dominant_tier", "")
            except Exception:
                continue
            if not math.isfinite(t):
                continue
            rows.append({"n": n, "k": k, "time_ms": t, "engine": eng, "B": b, "dominant_tier": tier})
    return rows


def build_grid(rows, key):
    ns = sorted({r["n"] for r in rows})
    ks = sorted({r["k"] for r in rows})
    n_idx = {v: i for i, v in enumerate(ns)}
    k_idx = {v: i for i, v in enumerate(ks)}
    grid = np.full((len(ns), len(ks)), np.nan, dtype=np.float64)
    for r in rows:
        i = n_idx[r["n"]]
        j = k_idx[r["k"]]
        grid[i, j] = r[key]
    return ns, ks, grid


def plot_time(rows, out_prefix):
    ns, ks, tgrid = build_grid(rows, "time_ms")
    z = np.log10(tgrid)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(z, origin="lower", aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log10(time_ms)")
    ax.set_title("ICM GPU Time Heatmap")
    ax.set_xlabel("k")
    ax.set_ylabel("n")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks], rotation=90)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(n) for n in ns])

    levels = [math.log10(v) for v in [1, 10, 100, 1000, 10000]]
    with np.errstate(invalid="ignore"):
        cs = ax.contour(z, levels=levels, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt=lambda x: f"{10**x:.0f}ms", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_time.png", dpi=180)
    plt.close(fig)


def plot_engine(rows, out_prefix):
    mapped = []
    for r in rows:
        v = 0 if r["engine"] == "linear" else 1
        mapped.append({"n": r["n"], "k": r["k"], "engine_id": v})
    ns = sorted({r["n"] for r in mapped})
    ks = sorted({r["k"] for r in mapped})
    n_idx = {v: i for i, v in enumerate(ns)}
    k_idx = {v: i for i, v in enumerate(ks)}
    grid = np.full((len(ns), len(ks)), np.nan)
    for r in mapped:
        grid[n_idx[r["n"]], k_idx[r["k"]]] = r["engine_id"]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(grid, origin="lower", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(["linear", "hybrid"])
    cbar.set_label("engine")
    ax.set_title("ICM GPU Engine Dispatch Map")
    ax.set_xlabel("k")
    ax.set_ylabel("n")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks], rotation=90)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(n) for n in ns])
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_engine.png", dpi=180)
    plt.close(fig)


def plot_tier(rows, out_prefix):
    tier_id = {"schoolbook": 0, "fused": 1, "cufft": 2}
    mapped = []
    for r in rows:
        if r.get("dominant_tier", "") not in tier_id:
            continue
        mapped.append({"n": r["n"], "k": r["k"], "tier_id": tier_id[r["dominant_tier"]]})
    if not mapped:
        return
    ns = sorted({r["n"] for r in mapped})
    ks = sorted({r["k"] for r in mapped})
    n_idx = {v: i for i, v in enumerate(ns)}
    k_idx = {v: i for i, v in enumerate(ks)}
    grid = np.full((len(ns), len(ks)), np.nan)
    for r in mapped:
        grid[n_idx[r["n"]], k_idx[r["k"]]] = r["tier_id"]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(grid, origin="lower", aspect="auto", interpolation="nearest", vmin=0, vmax=2)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(["schoolbook", "fused", "cufft"])
    cbar.set_label("dominant tier")
    ax.set_title("ICM GPU Dominant Tier Map")
    ax.set_xlabel("k")
    ax.set_ylabel("n")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks], rotation=90)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(n) for n in ns])
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_tier.png", dpi=180)
    plt.close(fig)


def plot_B(rows, out_prefix):
    ns, ks, bgrid = build_grid(rows, "B")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(bgrid, origin="lower", aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("B")
    ax.set_title("ICM GPU Block Size Map")
    ax.set_xlabel("k")
    ax.set_ylabel("n")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks], rotation=90)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(n) for n in ns])
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_B.png", dpi=180)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_heatmap.py <csv_path> [output_prefix]")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "gpu_heatmap"
    rows = read_rows(csv_path)
    if not rows:
        print("No valid rows found in CSV.")
        sys.exit(1)
    plot_time(rows, out_prefix)
    plot_engine(rows, out_prefix)
    plot_tier(rows, out_prefix)
    plot_B(rows, out_prefix)
    print(f"Wrote {out_prefix}_time.png, {out_prefix}_engine.png, {out_prefix}_tier.png, {out_prefix}_B.png")


if __name__ == "__main__":
    main()
