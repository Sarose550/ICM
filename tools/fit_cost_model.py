#!/usr/bin/env python3
"""
fit_cost_model.py — Assemble cost-model constants for hybrid tree engine.

Model (per Q-point):
  T = T_block + T_build + T_prop + T_leaf

  T_block = n * ((B+1)/2 * C_block_fma + C_block_mem)

  T_build = Σ_{FFT ℓ} nr[ℓ] * (calib[bfn] + bwm*(bwm+1)/2 * C_wrap)
          + Σ_{school ℓ} nr[ℓ] * d_eff² * C_school

  T_prop  = Σ_{FFT ℓ} nr[ℓ] * (R * calib[cfn] + cwm*(cwm+1) * C_wrap)
          + Σ_{school ℓ} nr[ℓ] * 2 * d_eff² * C_school

  T_leaf  = n * max(C_div, 2*B * C_leaf_fma) + (n/B) * C_leaf_block

As of the SPRINT_MICROBENCH_MIGRATION, ALL scalar constants are pinned
to direct measurements — zero free parameters remain for regression:

  WRAP_FMA_NS         → --wrap-ns       (tools/bench_wrap_fma.c)
  FP64_DIV_NS         → --div-ns        (tools/bench_div_chain.c)
  FMA_NS              → --fma-ns        (./bench_grid profile, schoolbook slope)
  PAIRED_CACHED_CORR_RATIO → --paired-cached-ratio  (./bench_grid profile, phase split)
  INDEP_PAIR_RATIO    → --indep-pair-ratio  (./bench_grid profile, phase split)
  FFT_OVERHEAD_NS     → --overhead-ns    (always 0.0 — redundant with calib_times_ns)

BLOCK_FMA_NS / BLOCK_MEM_NS / LEAF_FMA_NS / LEAF_BLOCK_NS were
converted to per-B lookup tables by bench_block_build.c and
bench_leaf_fma.c — not this script's concern.

When all 6 scalar pins are provided, scipy optimization is skipped
entirely (a 0-parameter degenerate fit is meaningless).  The script
assembles the pinned values and writes them to fft_config.h directly.

Usage: python3 tools/fit_cost_model.py <sample_plans.csv> [config_h] [--write]
          [--wrap-ns N] [--div-ns N] [--fma-ns N]
          [--paired-cached-ratio N] [--indep-pair-ratio N] [--overhead-ns N]
"""
import sys
import re
import numpy as np
from scipy.optimize import minimize
import argparse

# ── Parameter indices ──
P_BLOCK_FMA   = 0  # ns per FMA in block build (→ lookup table, out of scope)
P_BLOCK_MEM   = 1  # ns per player streaming a[j] (→ lookup table, out of scope)
P_WRAP        = 2  # ns per FMA in wrap correction (build + correlate)
P_R           = 3  # propagate/build FFT ratio (PAIRED_CACHED_CORR_RATIO)
P_SCHOOL      = 4  # ns per FMA in schoolbook multiply (FMA_NS)
P_DIV         = 5  # ns per FP64 division in leaf extract
P_LEAF_FMA    = 6  # ns per FMA in leaf Horner chain (→ lookup table, out of scope)
P_LEAF_BLOCK  = 7  # ns per block overhead in leaf extract (→ lookup table, out of scope)
P_OVERHEAD    = 8  # fixed per-Q-point overhead (sort, dispatch, alloc)
N_PARAMS = 9

PARAM_NAMES = [
    'C_block_fma', 'C_block_mem', 'C_wrap', 'R',
    'C_school', 'C_div', 'C_leaf_fma', 'C_leaf_block', 'C_overhead',
]

# Parameters still handled by THIS script (not converted to lookup tables).
# When ALL of these are pinned, scipy optimization is skipped.
IN_SCOPE_PARAMS = {P_WRAP, P_R, P_SCHOOL, P_DIV, P_OVERHEAD}

# Physically plausible bounds (Zen4)
BOUNDS = [
    (0.05, 2.0),     # C_block_fma: ~0.13 ns (FMA throughput)
    (0.1,  20.0),    # C_block_mem: streaming + L1 latency
    (0.1,  10.0),    # C_wrap: memory-latency-bound FMA
    (0.5,  5.0),     # R: paired cached correlate ratio
    (0.05, 2.0),     # C_school: schoolbook per-FMA
    (0.5,  30.0),    # C_div: FP64 division throughput ~4-6 ns
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


def fit(plans, calib, pins=None):
    """Fit parameters minimizing log-relative error.

    pins: dict of {param_index: value} to hold fixed (e.g. {P_WRAP: 0.40}).
    Remaining parameters are fitted freely. Empty/None pins = legacy path,
    all 9 free (reproduces the known identifiability problem for whichever
    of C_wrap/C_div is left unpinned).
    """
    pins = pins or {}
    active_indices = [i for i in range(N_PARAMS) if i not in pins]
    bounds_active = [BOUNDS[i] for i in active_indices]
    initial_active = [INITIAL[i] for i in active_indices]

    best_result = None
    best_fun = float('inf')

    for trial in range(20):
        rng = np.random.RandomState(trial)
        if trial == 0:
            x0 = np.array(initial_active)
        else:
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds_active])

        # Build full parameter vector for the objective function
        def make_full(x):
            full = np.zeros(N_PARAMS)
            for pi, val in pins.items():
                full[pi] = val
            for j, ai in enumerate(active_indices):
                full[ai] = x[j]
            return full

        def obj(x):
            return objective(make_full(x), plans, calib)

        res = minimize(obj, x0,
                       method='L-BFGS-B', bounds=bounds_active,
                       options={'maxiter': 10000, 'ftol': 1e-15})
        if res.fun < best_fun:
            best_fun = res.fun
            best_result = res
            best_result.x = make_full(res.x)  # store full param vector

    return best_result


def report(params, plans, calib, pins=None):
    """Print fit results and per-plan diagnostics."""
    pins = pins or {}
    n_plans = len(plans)
    log_errs = []
    for p in plans:
        pred = predict_plan(params, p, calib)
        log_errs.append(np.log(pred / p.per_qp_ns))
    log_errs = np.array(log_errs)
    rms = np.sqrt(np.mean(log_errs**2)) * 100
    max_err = np.max(np.abs(log_errs)) * 100

    n_free = N_PARAMS - len(pins)
    pinned_note = ""
    if pins:
        pinned_note = " (" + ", ".join(
            f"{PARAM_NAMES[i]} pinned at {v:.4f}" for i, v in pins.items()) + ")"
    print(f"\n{'='*70}")
    print(f"FITTED COST MODEL — {n_plans} plans, {n_free} fitted + "
          f"{len(pins)} pinned parameters{pinned_note}")
    print(f"RMS log-relative error: {rms:.2f}%")
    print(f"Max log-relative error: {max_err:.2f}%")
    print(f"{'='*70}")

    print(f"\nParameters:")
    for i in range(N_PARAMS):
        tag = " (pinned)" if i in pins else ""
        in_scope = " [in-scope]" if i in IN_SCOPE_PARAMS else " [lookup table]"
        print(f"  {PARAM_NAMES[i]:15s} = {params[i]:.4f}{tag}{in_scope}")

    print(f"\nPhysics checks:")
    print(f"  C_block_fma = {params[P_BLOCK_FMA]:.4f} ns"
          f"  (now a per-B lookup table — this scalar is vestigial)")
    print(f"  C_school    = {params[P_SCHOOL]:.4f} ns"
          f"  (expect ~0.05-0.15 ns on modern cores)")
    div_source = "pinned (direct microbenchmark)" if P_DIV in pins else "fitted (identifiability risk — prefer --div-ns)"
    print(f"  C_div       = {params[P_DIV]:.3f} ns"
          f"  (expect ~3-6 ns for a dependency-chained FP64 div; {div_source})")
    print(f"  R           = {params[P_R]:.4f}"
          f"  (expect ~1.5 on Apple Silicon, ~1.05 on Zen4)")
    wrap_source = "pinned (direct microbenchmark)" if P_WRAP in pins else "fitted (identifiability warning — prefer --wrap-ns)"
    print(f"  C_wrap      = {params[P_WRAP]:.3f} ns"
          f"  ({wrap_source})")
    leaf_cross_B = params[P_DIV] / (2 * params[P_LEAF_FMA]) if params[P_LEAF_FMA] > 0 else 0
    print(f"  Leaf crossover B = {leaf_cross_B:.0f}"
          f"  (C_div / 2*C_leaf_fma — vestigial, now per-B table)")

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


# Mapping from fit parameter name -> fft_config.h macro name(s).
# BLOCK_FMA_NS/BLOCK_MEM_NS/LEAF_FMA_NS/LEAF_BLOCK_NS are now per-B
# lookup tables — removed from this mapping (handled by bench_block_build
# and bench_leaf_fma, not this script).
# PAIRED_CACHED_CORR_RATIO and INDEP_PAIR_RATIO are now separate values
# (INDEP_PAIR_RATIO is written via a separate path, not from params array).
FIT_TO_MACRO = [
    ('C_wrap',       ['WRAP_FMA_NS']),
    ('R',            ['PAIRED_CACHED_CORR_RATIO']),
    ('C_school',     ['FMA_NS']),
    ('C_div',        ['FP64_DIV_NS']),
    ('C_overhead',   ['FFT_OVERHEAD_NS']),
]


def _replace_ifndef_define_value(text, macro_name, new_val):
    """Replace the numeric value in an #ifndef/#define/#endif block.
    Returns (new_text, old_val, changed)."""
    pattern = (
        r'(#ifndef\s+' + re.escape(macro_name) + r'\s*\n'
        r'#define\s+' + re.escape(macro_name) + r'\s+)'
        r'([\d.eE+\-]+)'
        r'([^\n]*\n#endif)'
    )
    match = re.search(pattern, text)
    if not match:
        return text, None, False

    old_val_str = match.group(2)
    old_val = float(old_val_str)
    new_val_str = f"{new_val:.4f}"

    if abs(old_val - new_val) < 1e-8:
        return text, old_val, False

    replacement = match.group(1) + new_val_str + match.group(3)
    text = text[:match.start()] + replacement + text[match.end():]
    return text, old_val, True


def write_constants_to_header(params, config_path, indep_pair_ratio=None):
    """Rewrite fft_config.h in-place with pinned (or fitted) constants.

    Finds each #ifndef MACRO / #define MACRO value / #endif block and replaces
    just the numeric value, preserving all surrounding structure and comments.
    Prints a diff-style summary of old→new for each macro touched.

    indep_pair_ratio: if provided, writes INDEP_PAIR_RATIO separately from
    PAIRED_CACHED_CORR_RATIO (the params array only carries the latter as R).
    """
    with open(config_path, 'r') as f:
        text = f.read()

    # Map fit param index -> value
    fit_vals = {PARAM_NAMES[i]: params[i] for i in range(N_PARAMS)}

    changes = []

    for fit_name, macro_names in FIT_TO_MACRO:
        new_val = fit_vals[fit_name]
        for macro_name in macro_names:
            text, old_val, changed = _replace_ifndef_define_value(
                text, macro_name, new_val)
            if old_val is not None:
                changes.append((macro_name, old_val, new_val, changed))
            else:
                print(f"  ⚠ WARNING: #ifndef/#define/#endif block for {macro_name} not found — skipping")

    # Write INDEP_PAIR_RATIO separately (not in params array)
    if indep_pair_ratio is not None:
        text, old_val, changed = _replace_ifndef_define_value(
            text, 'INDEP_PAIR_RATIO', indep_pair_ratio)
        if old_val is not None:
            changes.append(('INDEP_PAIR_RATIO', old_val, indep_pair_ratio, changed))
        else:
            print(f"  ⚠ WARNING: #ifndef/#define/#endif block for INDEP_PAIR_RATIO not found — skipping")

    # Write back
    with open(config_path, 'w') as f:
        f.write(text)

    # Print diff summary
    print(f"\n{'='*70}")
    print(f"WROTE {config_path} — {len(changes)} macro(s) touched")
    print(f"{'='*70}")
    for macro_name, old_val, new_val, changed in changes:
        marker = '*' if changed else ' '
        print(f" {marker} {macro_name:30s} {old_val:12.4f} → {new_val:12.4f}"
              + (" (unchanged)" if not changed else ""))

    n_changed = sum(1 for _, _, _, c in changes if c)
    print(f"\n{n_changed} macro(s) updated, "
          f"{len(changes) - n_changed} macro(s) already matched.")


def main():
    parser = argparse.ArgumentParser(
        description='Assemble cost-model constants for hybrid tree engine.')
    parser.add_argument('plans_csv', help='Path to sample_plans CSV')
    parser.add_argument('config_h', nargs='?', default='devices/zen4/fft_config.h',
                        help='Path to fft_config.h (default: devices/zen4/fft_config.h)')
    parser.add_argument('--write', action='store_true',
                        help='Rewrite fft_config.h in-place with constants')
    # ── Scalar pins (all 6) ──
    parser.add_argument('--wrap-ns', type=float, default=None,
                        help='Pin WRAP_FMA_NS (C_wrap) — from bench_wrap_fma.c')
    parser.add_argument('--div-ns', type=float, default=None,
                        help='Pin FP64_DIV_NS (C_div) — from bench_div_chain.c')
    parser.add_argument('--fma-ns', type=float, default=None,
                        help='Pin FMA_NS (C_school) — from ./bench_grid profile, '
                             'schoolbook slope between cps=16 and cps=32')
    parser.add_argument('--paired-cached-ratio', type=float, default=None,
                        help='Pin PAIRED_CACHED_CORR_RATIO (R) — from ./bench_grid '
                             'profile, phase-split table (f_fwd+2*(f_pw+f_ifft))')
    parser.add_argument('--indep-pair-ratio', type=float, default=None,
                        help='Pin INDEP_PAIR_RATIO — from ./bench_grid profile, '
                             'phase-split table (3*f_fwd+2*(f_pw+f_ifft))')
    parser.add_argument('--overhead-ns', type=float, default=0.0,
                        help='Pin FFT_OVERHEAD_NS (C_overhead). Default 0.0 — '
                             'calib_times_ns already measures the full pipeline; '
                             'this is conceptually redundant.')
    args = parser.parse_args()

    plans_path = args.plans_csv
    config_path = args.config_h

    # ── Build pins dict ──
    pins = {}
    if args.wrap_ns is not None:
        pins[P_WRAP] = args.wrap_ns
    if args.div_ns is not None:
        pins[P_DIV] = args.div_ns
    if args.fma_ns is not None:
        pins[P_SCHOOL] = args.fma_ns
    if args.paired_cached_ratio is not None:
        pins[P_R] = args.paired_cached_ratio

    # FFT_OVERHEAD_NS is always pinned (default 0.0)
    pins[P_OVERHEAD] = args.overhead_ns

    indep_pair_ratio = args.indep_pair_ratio

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

    # ── Detect zero-free-parameter case ──
    all_in_scope_pinned = IN_SCOPE_PARAMS.issubset(pins.keys())

    if all_in_scope_pinned:
        print(f"\n{'='*70}")
        print(f"ALL IN-SCOPE PARAMETERS PINNED — zero free parameters.")
        print(f"Skipping scipy optimization (degenerate 0-parameter fit).")
        print(f"{'='*70}")

        # Assemble full parameter vector from pins + defaults for out-of-scope
        params = np.zeros(N_PARAMS)
        for i in range(N_PARAMS):
            if i in pins:
                params[i] = pins[i]
            else:
                params[i] = INITIAL[i]  # out-of-scope placeholder

        # Print the pinned values
        print(f"\nPinned constants:")
        print(f"  WRAP_FMA_NS               = {params[P_WRAP]:.4f}")
        print(f"  PAIRED_CACHED_CORR_RATIO  = {params[P_R]:.4f}")
        print(f"  FMA_NS                    = {params[P_SCHOOL]:.4f}")
        print(f"  FP64_DIV_NS              = {params[P_DIV]:.4f}")
        print(f"  FFT_OVERHEAD_NS           = {params[P_OVERHEAD]:.4f}")
        if indep_pair_ratio is not None:
            print(f"  INDEP_PAIR_RATIO          = {indep_pair_ratio:.4f}")
        else:
            print(f"  INDEP_PAIR_RATIO          = (not provided — will not be written)")

        # Optionally report against sample_plans for diagnostics
        # (uses scalar block/leaf model — approximate only)
        if plans:
            print(f"\n(Reporting against sample_plans for diagnostic purposes —")
            print(f" block/leaf constants are now per-B lookup tables; the")
            print(f" scalar model used here is only approximate.)")
            report(params, plans, calib, pins=pins)

        if args.write:
            write_constants_to_header(params, config_path,
                                      indep_pair_ratio=indep_pair_ratio)
            print(f"\n✓ Fully-pinned config written to {config_path}")
        else:
            print(f"\n(Dry run — use --write to update {config_path})")

        return

    # ── Partial-pin path: some in-scope params still free, run scipy ──
    n_free = N_PARAMS - len(pins)
    pinned_msg = ""
    if pins:
        pinned_msg = " (" + ", ".join(
            f"{PARAM_NAMES[i]} pinned at {v:.4f}" for i, v in pins.items() if i in IN_SCOPE_PARAMS) + ")"
    unpinned_in_scope = IN_SCOPE_PARAMS - pins.keys()
    if unpinned_in_scope:
        pinned_msg += " [" + ", ".join(PARAM_NAMES[i] for i in sorted(unpinned_in_scope)) + " still free]"

    print(f"\nFitting {n_free} parameters to {len(plans)} plans...{pinned_msg}")
    result = fit(plans, calib, pins=pins)

    params = result.x
    rms = report(params, plans, calib, pins=pins)

    print(f"\n{'='*70}")
    print(f"SUMMARY — constants for fft_config.h:")
    print(f"{'='*70}")
    for i in range(N_PARAMS):
        tag = " (pinned)" if i in pins else ""
        print(f"  #define {PARAM_NAMES[i].upper():20s} {params[i]:.4f}{tag}")
    if indep_pair_ratio is not None:
        print(f"  #define {'INDEP_PAIR_RATIO':20s} {indep_pair_ratio:.4f} (pinned)")

    if rms > 1.0:
        print(f"\n⚠ RMS error {rms:.1f}% exceeds 1% target.")
        print(f"  Check phase breakdown for systematic bias.")
    else:
        print(f"\n✓ RMS error {rms:.2f}% meets <1% target.")

    if args.write:
        write_constants_to_header(params, config_path,
                                  indep_pair_ratio=indep_pair_ratio)


if __name__ == '__main__':
    main()
