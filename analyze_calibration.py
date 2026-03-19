#!/usr/bin/env python3
"""
analyze_calibration.py — Derive B*(n) from calibrate_B.c output

Reads CSV output from calibrate_B and:
  1. Extracts α₁ (phase-1 per-FMA cost)
  2. Fits α₂_eff(n, B) = A(n) + c(n)/B from end-to-end data
  3. Shows how c(n) decays with n (L1-thrashing zone shrinks)
  4. Computes B*(n) = √(c(n)·n/α₁) and compares to measured optima
  5. Evaluates candidate clip() formulas

Usage:
  # Run all experiments, save output:
  ./calibrate_B > calibration.csv 2>&1

  # Or run just M1 + M4 (sufficient for the key analysis):
  ./calibrate_B 1 > m1.csv
  ./calibrate_B 4 > m4.csv
  cat m1.csv m4.csv > calibration.csv

  # Analyze:
  python3 analyze_calibration.py calibration.csv
"""

import sys
import numpy as np

def parse_sections(filename):
    """Parse calibrate_B CSV output into sections by # header."""
    sections = {}
    current = None
    rows = []
    meta = {}

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('# Hardware:'):
                for part in line.split(':')[1].split():
                    k, v = part.split('=')
                    meta[k] = int(v.rstrip('KB')) * 1024 if 'KB' in v else int(v)
                continue
            if line.startswith('# L1D_doubles='):
                for part in line[2:].split():
                    k, v = part.split('=')
                    meta[k] = int(v)
                continue
            if line.startswith('# M') and ':' in line:
                if current and rows:
                    sections[current] = np.array(rows)
                tag = line.split(':')[0].strip('# ')
                current = tag
                rows = []
                continue
            if line.startswith('#'):
                continue
            parts = [x.strip() for x in line.split(',')]
            try:
                vals = [float(x) for x in parts]
                rows.append(vals)
            except ValueError:
                continue

    if current and rows:
        sections[current] = np.array(rows)

    return sections, meta


def analyze_alpha1(data):
    """Extract asymptotic α₁ from M1 data."""
    # cols: B, alpha1_ns, total_ns, FMAs
    Bs = data[:, 0]
    alphas = data[:, 1]

    # Asymptote: average for B >= 192 where it stabilizes
    mask = Bs >= 192
    alpha1 = np.mean(alphas[mask]) if mask.sum() > 0 else np.mean(alphas[-3:])

    print(f"  α₁(B) measurements:")
    for i in range(len(Bs)):
        marker = " ←" if Bs[i] >= 192 else ""
        print(f"    B={int(Bs[i]):4d}: {alphas[i]:.4f} ns/FMA{marker}")
    print(f"  α₁ (asymptote, B≥192) = {alpha1:.4f} ns/FMA")
    return alpha1


def analyze_end_to_end(data, alpha1):
    """Extract B*(n), fit c(n), derive the formula."""
    # cols: n, B, C, build_us, seq_us, speedup
    ns = sorted(set(data[:, 0].astype(int)))

    # ── Measured optima ──
    print(f"\n  Measured optimal B* and 95% plateaus:")
    print(f"  {'n':>6s}  {'B*':>4s}  {'C*':>3s}  {'speedup':>8s}  {'95% range':>12s}")
    print(f"  {'─'*6}  {'─'*4}  {'─'*3}  {'─'*8}  {'─'*12}")

    optima = {}
    for n in ns:
        mask = data[:, 0] == n
        B = data[mask, 1].astype(int)
        spd = data[mask, 5]
        idx = np.argmax(spd)
        Bopt = B[idx]
        best_spd = spd[idx]
        optima[n] = (Bopt, best_spd)

        thresh = best_spd * 0.95
        in_range = B[spd >= thresh]
        lo, hi = in_range.min(), in_range.max()
        C = int(round(n / Bopt))
        print(f"  {n:6d}  {Bopt:4d}  {C:3d}  {best_spd:8.3f}  [{lo:4d}, {hi:4d}]")

    # ── Fit α₂_eff(B) = A + c/B at each n ──
    print(f"\n  Fitting α₂_eff(n, B) = A(n) + c(n)/B:")
    print(f"  T(n,B) = nBα₁/2 + (n²/2)·[A(n) + c(n)/B]")
    print(f"  B-dependent: nBα₁/2 + n²c(n)/(2B)  →  B* = n·√(c(n)/α₁)")
    print(f"\n  {'n':>6s}  {'A(n)':>8s}  {'c(n)':>8s}  {'B*_model':>9s}  {'B*_meas':>8s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}")

    c_measured = {}
    for n in ns:
        mask = data[:, 0] == n
        Bs = data[mask, 1]
        T_us = data[mask, 3]

        # α₂_eff = 2(T_ns - nBα₁/2) / n²
        alpha_eff = 2 * (T_us * 1000 - n * Bs * alpha1 / 2) / (n * n)

        # Fit A + c/B
        X = np.column_stack([np.ones_like(Bs), 1.0 / Bs])
        coeffs = np.linalg.lstsq(X, alpha_eff, rcond=None)[0]
        A, c = coeffs
        c_measured[n] = (A, c)

        B_model = np.sqrt(c * n / alpha1) if c > 0 else 999
        Bopt = optima[n][0]
        print(f"  {n:6d}  {A:8.4f}  {c:8.2f}  {B_model:9.0f}  {Bopt:8d}")

    # ── Characterize c(n) decay ──
    ns_arr = np.array([n for n in ns if n >= 512], dtype=float)
    cs_arr = np.array([c_measured[n][1] for n in ns_arr.astype(int)])

    # c(n) depends on fraction of work in L1-thrashing zone
    # Thrashing zone: d < L1D_doubles/3 (approximate)
    # Fraction of FMAs: (d_thresh/n)²
    print(f"\n  c(n) decay analysis:")
    print(f"  c(n) is large at small n (L1 thrashing amplifies B-dependence)")
    print(f"  c(n) → c_warm at large n (L2 streaming is nearly B-flat)")
    print(f"  c_warm ≈ {cs_arr[-1]:.2f} (from largest n)")
    print(f"  c_max  ≈ {cs_arr[0]:.2f} (from smallest n)")

    # ── Evaluate candidate formulas ──
    print(f"\n  Formula evaluation (regret = % slower than measured B*):")
    formulas = {
        'clip(64,384,n/4)': lambda n: max(64, min(384, ((n//4+3)//8)*8)),
        'clip(64,256,n/4)': lambda n: max(64, min(256, ((n//4+3)//8)*8)),
        'clip(64,512,n/4)': lambda n: max(64, min(512, ((n//4+3)//8)*8)),
        'fixed B=256':      lambda n: 256,
        'fixed B=384':      lambda n: 384,
        '4√n clamped':      lambda n: max(64, min(512, ((int(4*n**0.5)+3)//8)*8)),
    }

    header = f"  {'n':>6s}"
    for name in formulas:
        header += f"  {name:>16s}"
    print(header)
    print(f"  {'─'*6}" + f"  {'─'*16}" * len(formulas))

    max_regrets = {name: 0 for name in formulas}
    sum_regrets = {name: 0 for name in formulas}
    count = 0

    for n in ns:
        if n < 512:  # below BUILD_BLOCK_THRESH
            continue
        count += 1
        mask = data[:, 0] == n
        B_arr = data[mask, 1].astype(int)
        spd_arr = data[mask, 5]
        best_spd = optima[n][1]

        line = f"  {n:6d}"
        for name, func in formulas.items():
            B_f = func(n)
            # Find closest available B
            idx = np.argmin(np.abs(B_arr - B_f))
            spd_f = spd_arr[idx]
            regret = (1 - spd_f / best_spd) * 100
            max_regrets[name] = max(max_regrets[name], regret)
            sum_regrets[name] += regret
            line += f"  {regret:15.1f}%"
        print(line)

    line_max = f"  {'MAX':>6s}"
    line_avg = f"  {'AVG':>6s}"
    for name in formulas:
        line_max += f"  {max_regrets[name]:15.1f}%"
        line_avg += f"  {sum_regrets[name]/count:15.1f}%"
    print(f"  {'─'*6}" + f"  {'─'*16}" * len(formulas))
    print(line_max)
    print(line_avg)

    # ── Recommendation ──
    best_formula = min(max_regrets, key=max_regrets.get)
    print(f"\n  Recommendation: {best_formula} "
          f"(max regret {max_regrets[best_formula]:.1f}%)")

    return c_measured, optima


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_calibration.py <calibration.csv>")
        print("  where calibration.csv is the output of ./calibrate_B")
        sys.exit(1)

    filename = sys.argv[1]
    sections, meta = parse_sections(filename)

    print("=" * 64)
    print("  ICM Sequential-Combine B* Calibration Analysis")
    print("=" * 64)

    if meta:
        print(f"\n  Hardware: L1D={meta.get('L1D', '?')}KB "
              f"L2={meta.get('L2', '?')}KB L3={meta.get('L3', '?')}KB")
        if 'L1D_doubles' in meta:
            print(f"  L1D = {meta['L1D_doubles']} doubles, "
                  f"L2 = {meta.get('L2_doubles', '?')} doubles")

    # ── M1: α₁ ──
    alpha1 = 0.30  # fallback
    if 'M1' in sections:
        print(f"\n{'─'*64}")
        print(f"  M1: Phase-1 per-FMA cost α₁")
        print(f"{'─'*64}")
        alpha1 = analyze_alpha1(sections['M1'])
    else:
        print(f"\n  M1 not found; using default α₁ = {alpha1:.2f}")

    # ── M4: end-to-end (the main event) ──
    if 'M4' in sections:
        print(f"\n{'─'*64}")
        print(f"  M4: End-to-end T(n, B) analysis")
        print(f"{'─'*64}")
        c_measured, optima = analyze_end_to_end(sections['M4'], alpha1)
    else:
        print("\n  M4 not found — run ./calibrate_B 4 to get the key data")

    # ── M3: per-step overhead (supplementary) ──
    if 'M3' in sections:
        print(f"\n{'─'*64}")
        print(f"  M3: Per-step overhead summary")
        print(f"{'─'*64}")
        d3 = sections['M3']
        configs = set(zip(d3[:, 0].astype(int), d3[:, 1].astype(int)))
        print(f"  {'n':>6s}  {'B':>4s}  {'C':>4s}  {'F_total_μs':>10s}  "
              f"{'F_per_step_μs':>13s}")
        for n, B in sorted(configs):
            mask = (d3[:, 0] == n) & (d3[:, 1] == B)
            overheads = d3[mask, 6]
            C = int(d3[mask, 2].max()) + 1
            F_total = np.sum(overheads)
            F_per = F_total / len(overheads)
            print(f"  {n:6d}  {B:4d}  {C:4d}  {F_total/1e3:10.1f}  "
                  f"{F_per/1e3:13.3f}")

    print(f"\n{'='*64}")
    print(f"  Analysis complete.")
    print(f"{'='*64}")


if __name__ == '__main__':
    main()
