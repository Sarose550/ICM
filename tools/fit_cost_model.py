#!/usr/bin/env python3
"""
fit_cost_model.py — Fit physics-based cost model for hybrid tree engine.

Model (per Q-point):
  T = T_block + T_build + T_prop + T_leaf

  T_block = n * ((B+1)/2 * C_block_fma + C_block_mem)

  T_build = Σ_{FFT ℓ} nr[ℓ] * (calib[bfn] + bwm*(bwm+1)/2 * C_wrap)
          + Σ_{school ℓ} nr[ℓ] * d_eff² * C_school

  T_prop  = Σ_{FFT ℓ} nr[ℓ] * (R * calib[cfn] + cwm*(cwm+1) * C_wrap)
          + Σ_{school ℓ} nr[ℓ] * 2 * d_eff² * C_school

  T_leaf  = n * max(C_div, 2*B * C_leaf_fma) + (n/B) * C_leaf_block

9 fitted parameters, calib[fft_n] lookup from fft_config.h.
Objective: minimize Σ(log(pred/meas))².

Usage: python3 tools/fit_cost_model.py <sample_plans.csv> [devices/zen4/fft_config.h]
"""
import sys
import re
import numpy as np
from scipy.optimize import minimize

# ── Parameter indices ──
P_BLOCK_FMA   = 0  # ns per FMA in block build
P_BLOCK_MEM   = 1  # ns per player streaming a[j]
P_WRAP        = 2  # ns per FMA in wrap correction (build + correlate)
P_R           = 3  # propagate/build FFT ratio (PAIRED_CACHED_CORR_RATIO)
P_SCHOOL      = 4  # ns per FMA in schoolbook multiply
P_DIV         = 5  # ns per FP64 division in leaf extract
P_LEAF_FMA    = 6  # ns per FMA in leaf Horner chain
P_LEAF_BLOCK  = 7  # ns per block overhead in leaf extract
P_OVERHEAD    = 8  # fixed per-Q-point overhead (sort, dispatch, alloc)
N_PARAMS = 9

PARAM_NAMES = [
    'C_block_fma', 'C_block_mem', 'C_wrap', 'R',
    'C_school', 'C_div', 'C_leaf_fma', 'C_leaf_block', 'C_overhead',
]

# Physically plausible bounds (Zen4)
BOUNDS = [
    (0.05, 2.0),     # C_block_fma: ~0.13 ns (FMA throughput)
    (0.1,  20.0),    # C_block_mem: streaming + L1 latency
    (0.1,  10.0),    # C_wrap: memory-latency-bound FMA
    (0.5,  2.0),     # R: paired cached correlate ratio
    (0.05, 2.0),     # C_school: schoolbook per-FMA
    (0.5,  10.0),    # C_div: FP64 division throughput ~4-6 ns
    (0.01, 1.0),     # C_leaf_fma: Horner per-FMA
    (1.0,  500.0),   # C_leaf_block: per-block setup
    (0.0,  50000.0), # C_overhead: per-Q-point fixed cost
]

INITIAL = [0.15, 2.0, 4.0, 1.05, 0.15, 4.0, 0.05, 50.0, 1000.0]


def load_calib(path):
    """Parse fft_config.h, return dict mapping fft_size → pipeline_time_ns."""
    text = open(path).read()

    sizes_match = re.search(
        r'calib_sizes\[.*?\]\s*=\s*\{([^}]+)\}', text, re.DOTALL)
    times_match = re.search(
        r'calib_times_ns\[.*?\]\s*=\s*\{([^}]+)\}', text, re.DOTALL)
    if not sizes_match or not times_match:
        raise ValueError(f"Cannot parse calibration arrays from {path}")

    sizes = [int(x) for x in re.findall(r'\d+', sizes_match.group(1))]
    times = [float(x) for x in re.findall(r'[\d.]+', times_match.group(1))]
    assert len(sizes) == len(times), f"Size mismatch: {len(sizes)} vs {len(times)}"

    return dict(zip(sizes, times))


class Level:
    __slots__ = ('nr', 'cps', 'use_fft', 'bfn', 'bwm', 'cache',
                 'cfn', 'cwm', 'below', 'g_need')


class Plan:
    __slots__ = ('n', 'k', 'B', 'L', 'per_qp_ns', 'levels')


def parse_plans(path):
    """Parse sample_plans.csv output."""
    plans = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            p = Plan()
            p.n = int(parts[0])
            p.k = int(parts[1])
            p.B = int(parts[2])
            p.L = int(parts[3])
            # parts[4] = total_ms, parts[5] = per_qp_ns
            p.per_qp_ns = float(parts[5])
            p.levels = []
            for i in range(6, len(parts)):
                fields = parts[i].split(':')
                if len(fields) < 10:
                    continue
                lv = Level()
                lv.nr       = int(fields[0])
                lv.cps      = int(fields[1])
                lv.use_fft  = int(fields[2])
                lv.bfn      = int(fields[3])
                lv.bwm      = int(fields[4])
                lv.cache    = int(fields[5])
                lv.cfn      = int(fields[6])
                lv.cwm      = int(fields[7])
                lv.below    = int(fields[8])
                lv.g_need   = int(fields[9])
                p.levels.append(lv)
            plans.append(p)
    return plans


def predict_plan(params, plan, calib):
    """Predict per-Q-point time (ns) for a single plan."""
    n, B = plan.n, plan.B
    c_bfma  = params[P_BLOCK_FMA]
    c_bmem  = params[P_BLOCK_MEM]
    c_wrap  = params[P_WRAP]
    R       = params[P_R]
    c_sch   = params[P_SCHOOL]
    c_div   = params[P_DIV]
    c_lfma  = params[P_LEAF_FMA]
    c_lblk  = params[P_LEAF_BLOCK]
    c_ovhd  = params[P_OVERHEAD]

    T_block = n * ((B + 1) / 2.0 * c_bfma + c_bmem)

    T_build = 0.0
    T_prop = 0.0
    for lv in plan.levels:
        if lv.use_fft:
            build_calib = calib.get(lv.bfn, 0.0)
            corr_calib = calib.get(lv.cfn, 0.0) if lv.cfn > 0 else build_calib
            T_build += lv.nr * (build_calib + lv.bwm * (lv.bwm + 1) / 2.0 * c_wrap)
            T_prop += lv.nr * (R * corr_calib + lv.cwm * (lv.cwm + 1) * c_wrap)
        else:
            d_eff = lv.cps // 2 if lv.below else lv.cps - 1
            flops = (d_eff + 1) * (d_eff + 1)
            T_build += lv.nr * flops * c_sch
            T_prop += lv.nr * 2 * flops * c_sch

    T_leaf = n * max(c_div, 2 * B * c_lfma) + (n / B) * c_lblk

    return T_block + T_build + T_prop + T_leaf + c_ovhd


def objective(params, plans, calib):
    """Sum of (log(pred/meas))² — log-space least squares."""
    total = 0.0
    for p in plans:
        pred = predict_plan(params, p, calib)
        if pred <= 0:
            return 1e18
        ratio = np.log(pred / p.per_qp_ns)
        total += ratio * ratio
    return total


def fit(plans, calib):
    """Fit 9 parameters minimizing log-relative error."""
    best_result = None
    best_fun = float('inf')

    for trial in range(20):
        rng = np.random.RandomState(trial)
        if trial == 0:
            x0 = np.array(INITIAL)
        else:
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS])

        res = minimize(objective, x0, args=(plans, calib),
                       method='L-BFGS-B', bounds=BOUNDS,
                       options={'maxiter': 10000, 'ftol': 1e-15})
        if res.fun < best_fun:
            best_fun = res.fun
            best_result = res

    return best_result


def report(params, plans, calib):
    """Print fit results and per-plan diagnostics."""
    n_plans = len(plans)
    log_errs = []
    for p in plans:
        pred = predict_plan(params, p, calib)
        log_errs.append(np.log(pred / p.per_qp_ns))
    log_errs = np.array(log_errs)
    rms = np.sqrt(np.mean(log_errs**2)) * 100
    max_err = np.max(np.abs(log_errs)) * 100

    print(f"\n{'='*70}")
    print(f"FITTED COST MODEL — {n_plans} plans, {N_PARAMS} parameters")
    print(f"RMS log-relative error: {rms:.2f}%")
    print(f"Max log-relative error: {max_err:.2f}%")
    print(f"{'='*70}")

    print(f"\nParameters:")
    for i in range(N_PARAMS):
        print(f"  {PARAM_NAMES[i]:15s} = {params[i]:.4f}")

    print(f"\nPhysics checks:")
    print(f"  C_block_fma = {params[P_BLOCK_FMA]:.4f} ns"
          f"  (expect ~0.13 ns = 2 FMA/cycle @ 3.8 GHz on Zen4)")
    print(f"  C_school    = {params[P_SCHOOL]:.4f} ns"
          f"  (expect ~FMA_NS)")
    print(f"  C_div       = {params[P_DIV]:.3f} ns"
          f"  (expect ~4-6 ns for FP64 div on Zen4)")
    print(f"  R           = {params[P_R]:.4f}"
          f"  (expect ~1.03-1.08)")
    print(f"  C_wrap      = {params[P_WRAP]:.3f} ns"
          f"  (expect ~WRAP_FMA_NS, memory-latency-bound)")
    leaf_cross_B = params[P_DIV] / (2 * params[P_LEAF_FMA]) if params[P_LEAF_FMA] > 0 else 0
    print(f"  Leaf crossover B = {leaf_cross_B:.0f}"
          f"  (C_div / 2*C_leaf_fma)")

    # Per-plan breakdown
    print(f"\n{'n':>6s} {'k':>6s} {'B':>3s} {'L':>2s}"
          f" {'meas':>9s} {'pred':>9s} {'err%':>6s}"
          f" {'T_block':>8s} {'T_build':>8s} {'T_prop':>8s} {'T_leaf':>8s}")
    sorted_plans = sorted(plans, key=lambda p: (p.n, p.k, p.B))
    for p in sorted_plans:
        pred = predict_plan(params, p, calib)
        err = (pred / p.per_qp_ns - 1) * 100

        # Phase breakdown
        c = params
        T_block = p.n * ((p.B + 1) / 2.0 * c[P_BLOCK_FMA] + c[P_BLOCK_MEM])
        T_build = 0.0
        T_prop = 0.0
        for lv in p.levels:
            if lv.use_fft:
                bc = calib.get(lv.bfn, 0)
                cc = calib.get(lv.cfn, 0) if lv.cfn > 0 else bc
                T_build += lv.nr * (bc + lv.bwm * (lv.bwm + 1) / 2.0 * c[P_WRAP])
                T_prop += lv.nr * (c[P_R] * cc + lv.cwm * (lv.cwm + 1) * c[P_WRAP])
            else:
                d_eff = lv.cps // 2 if lv.below else lv.cps - 1
                flops = (d_eff + 1) * (d_eff + 1)
                T_build += lv.nr * flops * c[P_SCHOOL]
                T_prop += lv.nr * 2 * flops * c[P_SCHOOL]
        T_leaf = p.n * max(c[P_DIV], 2 * p.B * c[P_LEAF_FMA]) + (p.n / p.B) * c[P_LEAF_BLOCK]

        print(f"{p.n:6d} {p.k:6d} {p.B:3d} {p.L:2d}"
              f" {p.per_qp_ns:9.0f} {pred:9.0f} {err:+5.1f}%"
              f" {T_block:8.0f} {T_build:8.0f} {T_prop:8.0f} {T_leaf:8.0f}")

    # Worst outliers
    errors = [(abs(np.log(predict_plan(params, p, calib) / p.per_qp_ns)) * 100, p)
              for p in plans]
    errors.sort(reverse=True)
    print(f"\nWorst 10 outliers:")
    for err_pct, p in errors[:10]:
        pred = predict_plan(params, p, calib)
        print(f"  n={p.n:6d} k={p.k:6d} B={p.B:3d}"
              f"  meas={p.per_qp_ns:9.0f}  pred={pred:9.0f}"
              f"  err={err_pct:.1f}%")

    return rms


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/fit_cost_model.py <sample_plans.csv> [fft_config.h]")
        sys.exit(1)

    plans_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else 'devices/zen4/fft_config.h'

    print(f"Loading calibration from {config_path}")
    calib = load_calib(config_path)
    print(f"  {len(calib)} sizes, range [{min(calib)}..{max(calib)}]")

    print(f"Loading plans from {plans_path}")
    plans = parse_plans(plans_path)
    print(f"  {len(plans)} plans")

    # Filter out plans with suspiciously fast times (cold cache artifacts)
    valid = [p for p in plans if p.per_qp_ns > 100]
    if len(valid) < len(plans):
        print(f"  Filtered {len(plans) - len(valid)} plans with per_qp_ns < 100")
        plans = valid

    print(f"\nFitting {N_PARAMS} parameters to {len(plans)} plans...")
    result = fit(plans, calib)

    params = result.x
    rms = report(params, plans, calib)

    print(f"\n{'='*70}")
    print(f"SUMMARY — constants for fft_config.h:")
    print(f"{'='*70}")
    for i in range(N_PARAMS):
        print(f"  #define {PARAM_NAMES[i].upper():20s} {params[i]:.4f}")

    if rms > 1.0:
        print(f"\n⚠ RMS error {rms:.1f}% exceeds 1% target.")
        print(f"  Check phase breakdown for systematic bias.")
    else:
        print(f"\n✓ RMS error {rms:.2f}% meets <1% target.")


if __name__ == '__main__':
    main()
