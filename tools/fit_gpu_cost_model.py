#!/usr/bin/env python3
"""
fit_gpu_cost_model.py — GPU cost model with empirical kernel lookups.

Tree costs: calibration tables + measured floors (cuFFT pipeline, fused)
Non-tree costs: empirical lookup tables from bench_kernels
Fitted: C_wrap, C_school, R, C_gap only (4 params)

All other costs are interpolated from measurements.
"""
import sys
import re
import math
import numpy as np
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

SM_COUNT = 148

# ── Measured floor tables ──

PIPE_FLOOR_SIZES = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536], dtype=np.float64)
PIPE_FLOOR_NS    = np.array([3.8, 6.0, 10.9, 20.2, 35.8, 73.6, 150.9, 371.3, 849.1, 2104.5, 3979.3])

FUSED_FLOOR_SIZES = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype=np.float64)
FUSED_FLOOR_NS    = np.array([2.36, 5.54, 9.48, 18.32, 48.84, 122.74, 246.43, 583.62])

def _interp_table(fft_n, sizes, ns):
    if fft_n <= sizes[0]: return ns[0] * fft_n / sizes[0]
    if fft_n >= sizes[-1]: return ns[-1] * fft_n / sizes[-1]
    for i in range(len(sizes) - 1):
        if fft_n <= sizes[i + 1]:
            t = (math.log(fft_n) - math.log(sizes[i])) / (math.log(sizes[i+1]) - math.log(sizes[i]))
            return ns[i] + t * (ns[i + 1] - ns[i])
    return ns[-1]

def interp_pipe_floor(fft_n): return _interp_table(fft_n, PIPE_FLOOR_SIZES, PIPE_FLOOR_NS)
def interp_fused_floor(fft_n): return _interp_table(fft_n, FUSED_FLOOR_SIZES, FUSED_FLOOR_NS)

def calib_batch_for_size(n):
    if n <= 2048: return 1024
    if n <= 8192: return 512
    if n <= 32768: return 128
    if n <= 65536: return 64
    if n <= 131072: return 16
    return 8

# ── Empirical kernel cost lookups (from bench_kernels) ──

def load_kernel_bench(path):
    """Parse bench_kernels CSV into lookup tables."""
    compute_a = []    # (n, ns_per_qp)
    block_build = []  # (n, B, ns_per_qp)
    leaf_extract = [] # (n, B, ns_per_qp)
    accumulate = []   # (n, ns_per_qp)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line: continue
            parts = line.split(',')
            kind = parts[0]
            if kind == 'compute_a':
                compute_a.append((int(parts[1]), float(parts[2])))
            elif kind == 'block_build':
                block_build.append((int(parts[1]), int(parts[2]), float(parts[5])))
            elif kind == 'leaf_extract':
                leaf_extract.append((int(parts[1]), int(parts[2]), float(parts[3])))
            elif kind == 'accumulate':
                accumulate.append((int(parts[1]), float(parts[2])))

    return {
        'compute_a': compute_a,
        'block_build': block_build,
        'leaf_extract': leaf_extract,
        'accumulate': accumulate,
    }


def build_kernel_interps(kb):
    """Build interpolators for each kernel from bench data."""
    interps = {}

    # compute_a: 1D interp on log(n) → ns/qp
    ca = kb['compute_a']
    log_n = np.log([x[0] for x in ca])
    ns = np.array([x[1] for x in ca])
    interps['compute_a'] = interp1d(log_n, ns, kind='linear', fill_value='extrapolate')

    # accumulate: 1D interp on log(n) → ns/qp
    ac = kb['accumulate']
    log_n = np.log([x[0] for x in ac])
    ns = np.array([x[1] for x in ac])
    interps['accumulate'] = interp1d(log_n, ns, kind='linear', fill_value='extrapolate')

    # block_build: 2D — group by n, interp on log(B) → ns/qp
    bb = {}
    for n, B, ns_qp in kb['block_build']:
        bb.setdefault(n, []).append((B, ns_qp))
    interps['block_build_by_n'] = {}
    for n, pts in bb.items():
        pts.sort()
        log_B = np.log([p[0] for p in pts])
        ns = np.array([p[1] for p in pts])
        interps['block_build_by_n'][n] = interp1d(log_B, ns, kind='linear', fill_value='extrapolate')
    interps['block_build_ns'] = sorted(bb.keys())

    # leaf_extract: same structure
    le = {}
    for n, B, ns_qp in kb['leaf_extract']:
        le.setdefault(n, []).append((B, ns_qp))
    interps['leaf_extract_by_n'] = {}
    for n, pts in le.items():
        pts.sort()
        log_B = np.log([p[0] for p in pts])
        ns = np.array([p[1] for p in pts])
        interps['leaf_extract_by_n'][n] = interp1d(log_B, ns, kind='linear', fill_value='extrapolate')
    interps['leaf_extract_ns'] = sorted(le.keys())

    return interps


def lookup_block_build(interps, n, B):
    """Interpolate block_build cost for (n, B) using bench data."""
    ns_list = interps['block_build_ns']
    by_n = interps['block_build_by_n']
    log_B = math.log(B)
    if n in by_n:
        return float(by_n[n](log_B))
    # Interpolate between measured n values
    if n <= ns_list[0]:
        return float(by_n[ns_list[0]](log_B)) * n / ns_list[0]
    if n >= ns_list[-1]:
        return float(by_n[ns_list[-1]](log_B)) * n / ns_list[-1]
    for i in range(len(ns_list) - 1):
        if n <= ns_list[i + 1]:
            t = (math.log(n) - math.log(ns_list[i])) / (math.log(ns_list[i+1]) - math.log(ns_list[i]))
            v0 = float(by_n[ns_list[i]](log_B))
            v1 = float(by_n[ns_list[i+1]](log_B))
            return v0 + t * (v1 - v0)
    return float(by_n[ns_list[-1]](log_B)) * n / ns_list[-1]


def lookup_leaf_extract(interps, n, B):
    ns_list = interps['leaf_extract_ns']
    by_n = interps['leaf_extract_by_n']
    log_B = math.log(max(B, 8))  # bench starts at B=8
    if n in by_n:
        return float(by_n[n](log_B))
    if n <= ns_list[0]:
        return float(by_n[ns_list[0]](log_B)) * n / ns_list[0]
    if n >= ns_list[-1]:
        return float(by_n[ns_list[-1]](log_B)) * n / ns_list[-1]
    for i in range(len(ns_list) - 1):
        if n <= ns_list[i + 1]:
            t = (math.log(n) - math.log(ns_list[i])) / (math.log(ns_list[i+1]) - math.log(ns_list[i]))
            v0 = float(by_n[ns_list[i]](log_B))
            v1 = float(by_n[ns_list[i+1]](log_B))
            return v0 + t * (v1 - v0)
    return float(by_n[ns_list[-1]](log_B)) * n / ns_list[-1]


# ── Calibration loader ──

def load_gpu_calib(path):
    text = open(path).read()
    def pa(name):
        m = re.search(rf'{name}\[.*?\]\s*=\s*\{{([^}}]+)\}}', text, re.DOTALL)
        return [float(x) for x in re.findall(r'[\d.eE+-]+', m.group(1))] if m else None
    sizes = [int(x) for x in re.findall(r'\d+', re.search(r'gpu_calib_sizes\[.*?\]\s*=\s*\{([^}]+)\}', text, re.DOTALL).group(1))]
    cufft = pa('gpu_calib_cufft_ns')
    fb = pa('gpu_calib_cufftdx_r2c_build_ns') or pa('gpu_calib_cufftdx_build_ns')
    fc = pa('gpu_calib_cufftdx_r2c_corr_ns') or pa('gpu_calib_cufftdx_corr_ns')
    oh_m = re.search(r'GPU_FFT_OVERHEAD_NS\s+([\d.]+)', text)
    oh = float(oh_m.group(1)) if oh_m else 0.0
    calib = {}
    for i, s in enumerate(sizes):
        calib[s] = {
            'cufft': cufft[i] + oh if cufft else 0.0,
            'fused_build': fb[i] if fb and fb[i] > 0 else None,
            'fused_corr': fc[i] if fc and fc[i] > 0 else None,
        }
    return calib


# ── Plan parsing ──

def parse_plans(path):
    plans = []
    with open(path) as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            class P: pass
            p = P()
            p.n, p.k, p.B = int(parts[0]), int(parts[1]), int(parts[2])
            p.qb, p.L = int(parts[3]), int(parts[4])
            p.per_qp_ns = float(parts[6])
            p.levels = []
            for i in range(7, len(parts)):
                fields = parts[i].split(':')
                if len(fields) < 11: continue
                class Lv: pass
                lv = Lv()
                lv.tier, lv.nr, lv.cps = int(fields[0]), int(fields[1]), int(fields[2])
                lv.use_fft, lv.fft_n, lv.bwm = int(fields[3]), int(fields[4]), int(fields[5])
                lv.cache, lv.cwm, lv.below = int(fields[6]), int(fields[7]), int(fields[8])
                lv.g_need, lv.out_needed = int(fields[9]), int(fields[10])
                p.levels.append(lv)
            plans.append(p)
    return [p for p in plans if p.per_qp_ns > 50]


# ── Model ──

# Fitted params: only things we can't measure directly
P_WRAP   = 0   # ns per FMA in wrap correction
P_SCHOOL = 1   # ns per FMA in schoolbook multiply
P_R      = 2   # cuFFT corr/build ratio
P_GAP    = 3   # ns per kernel transition in cuFFT pipeline
N_FIT = 4

FIT_NAMES = ['C_wrap', 'C_school', 'R', 'C_gap']
FIT_BOUNDS = [
    (0.001, 50.0),     # C_wrap
    (1e-5, 0.01),      # C_school
    (0.3, 1.5),        # R
    (100.0, 20000.0),  # C_gap
]


def predict_plan(params, plan, calib, interps):
    c_wrap = params[P_WRAP]
    c_sch  = params[P_SCHOOL]
    R      = params[P_R]
    c_gap  = params[P_GAP]
    n, B, qb = plan.n, plan.B, plan.qb

    # Non-tree: empirical lookups
    T_compute_a = float(interps['compute_a'](math.log(max(n, 256))))
    T_block = lookup_block_build(interps, n, B)
    T_leaf = lookup_leaf_extract(interps, n, B)
    T_accum = float(interps['accumulate'](math.log(max(n, 256))))

    # Tree: calibration + floors
    T_tree = 0.0
    n_gap_kernels = 0
    for lv in plan.levels:
        nn = lv.nr
        eb = qb * nn

        if lv.tier == 1:  # SCHOOLBOOK
            d_eff = lv.cps // 2 if lv.below else lv.cps - 1
            flops = (d_eff + 1)**2 + 2 * lv.cps * lv.out_needed
            T_tree += nn * flops * c_sch

        elif lv.tier == 2:  # FUSED
            c = calib.get(lv.fft_n, {})
            fb = c.get('fused_build')
            fc = c.get('fused_corr')
            if fb and fc:
                calib_pp = fb + fc
                fl = interp_fused_floor(lv.fft_n)
                fcb = calib_batch_for_size(lv.fft_n)
                pp = fl + max(0, calib_pp - fl) * fcb / max(1, eb)
            else:
                raw = c.get('cufft', 100.0)
                fl = interp_pipe_floor(lv.fft_n)
                cb = calib_batch_for_size(lv.fft_n)
                calib_total = raw * (1 + R)
                pp = fl + max(0, calib_total - fl) * cb / max(1, eb)
            # Wrap correction
            bw = lv.bwm * (lv.bwm + 1) / 2.0 * c_wrap
            cw = lv.cwm * (lv.cwm + 1) * c_wrap
            T_tree += nn * pp + nn * (bw + cw)

        else:  # cuFFT
            c = calib.get(lv.fft_n, {})
            raw = c.get('cufft', 100.0)
            fl = interp_pipe_floor(lv.fft_n)
            cb = calib_batch_for_size(lv.fft_n)
            calib_total = raw * (1 + R)
            pp = fl + max(0, calib_total - fl) * cb / max(1, eb)
            bw = lv.bwm * (lv.bwm + 1) / 2.0 * c_wrap
            cw = lv.cwm * (lv.cwm + 1) * c_wrap
            T_tree += nn * pp + nn * (bw + cw)
            n_gap_kernels += 6 if lv.cache else 8

    T_gap = n_gap_kernels * c_gap / qb
    return T_compute_a + T_block + T_tree + T_leaf + T_accum + T_gap


def precompute_non_tree(plans, interps, calib):
    """Precompute non-tree costs (empirical) for all plans."""
    N = len(plans)
    f = {}
    f['meas'] = np.array([p.per_qp_ns for p in plans])
    f['log_meas'] = np.log(f['meas'])
    f['T_compute_a'] = np.array([float(interps['compute_a'](math.log(max(p.n, 256)))) for p in plans])
    f['T_block'] = np.array([lookup_block_build(interps, p.n, p.B) for p in plans])
    f['T_leaf'] = np.array([lookup_leaf_extract(interps, p.n, p.B) for p in plans])
    f['T_accum'] = np.array([float(interps['accumulate'](math.log(max(p.n, 256)))) for p in plans])
    f['T_nontree'] = f['T_compute_a'] + f['T_block'] + f['T_leaf'] + f['T_accum']

    # Precompute tree level features
    # School: (plan_idx, nn * flops)
    sch_idx, sch_work = [], []
    # Fused: (plan_idx, nn * pp_at_floor, nn * overhead, fused_cb, eb)
    fus_idx, fus_floor, fus_ovhd, fus_cb, fus_eb = [], [], [], [], []
    # cuFFT: (plan_idx, nn, raw_build, pipe_floor, calib_batch, eb, n_gap_kernels)
    cuf_idx, cuf_nn, cuf_raw, cuf_floor, cuf_cb, cuf_eb, cuf_gap = [], [], [], [], [], [], []
    # Wrap: (plan_idx, nn * bwm*(bwm+1)/2 + nn * cwm*(cwm+1))
    wrap_idx, wrap_work = [], []

    for pi, p in enumerate(plans):
        for lv in p.levels:
            nn = float(lv.nr)
            eb = float(p.qb * lv.nr)

            if lv.tier == 1:
                d_eff = lv.cps // 2 if lv.below else lv.cps - 1
                flops = (d_eff + 1)**2 + 2 * lv.cps * lv.out_needed
                sch_idx.append(pi)
                sch_work.append(nn * flops)
            elif lv.tier == 2:
                c = calib.get(lv.fft_n, {})
                fb_val = c.get('fused_build')
                fc_val = c.get('fused_corr')
                if fb_val and fc_val:
                    calib_pp = fb_val + fc_val
                    fl = interp_fused_floor(lv.fft_n)
                    fcb = calib_batch_for_size(lv.fft_n)
                    fus_idx.append(pi)
                    fus_floor.append(nn * fl)
                    fus_ovhd.append(nn * max(0, calib_pp - fl))
                    fus_cb.append(fcb)
                    fus_eb.append(eb)
                else:
                    raw = c.get('cufft', 100.0)
                    fl = interp_pipe_floor(lv.fft_n)
                    cb = calib_batch_for_size(lv.fft_n)
                    cuf_idx.append(pi); cuf_nn.append(nn); cuf_raw.append(raw)
                    cuf_floor.append(fl); cuf_cb.append(cb); cuf_eb.append(eb)
                    cuf_gap.append(6.0 if lv.cache else 8.0)
            else:
                raw = c.get('cufft', 100.0) if (c := calib.get(lv.fft_n, {})) else 100.0
                fl = interp_pipe_floor(lv.fft_n)
                cb = calib_batch_for_size(lv.fft_n)
                cuf_idx.append(pi); cuf_nn.append(nn); cuf_raw.append(raw)
                cuf_floor.append(fl); cuf_cb.append(cb); cuf_eb.append(eb)
                cuf_gap.append(6.0 if lv.cache else 8.0)

            # Wrap for all FFT levels
            if lv.tier >= 2:
                bw = lv.bwm * (lv.bwm + 1) / 2.0
                cw = lv.cwm * (lv.cwm + 1)
                if bw > 0 or cw > 0:
                    wrap_idx.append(pi)
                    wrap_work.append(nn * (bw + cw))

    f['sch_idx'] = np.array(sch_idx, dtype=np.int32)
    f['sch_work'] = np.array(sch_work)
    f['fus_idx'] = np.array(fus_idx, dtype=np.int32)
    f['fus_floor'] = np.array(fus_floor)
    f['fus_ovhd'] = np.array(fus_ovhd)
    f['fus_cb'] = np.array(fus_cb, dtype=np.float64)
    f['fus_eb'] = np.array(fus_eb)
    f['cuf_idx'] = np.array(cuf_idx, dtype=np.int32)
    f['cuf_nn'] = np.array(cuf_nn)
    f['cuf_raw'] = np.array(cuf_raw)
    f['cuf_floor'] = np.array(cuf_floor)
    f['cuf_cb'] = np.array(cuf_cb, dtype=np.float64)
    f['cuf_eb'] = np.array(cuf_eb)
    f['cuf_gap'] = np.array(cuf_gap)
    f['wrap_idx'] = np.array(wrap_idx, dtype=np.int32)
    f['wrap_work'] = np.array(wrap_work)
    f['qb'] = np.array([float(p.qb) for p in plans])
    f['N'] = N
    return f


def objective(params, f):
    c_wrap = params[P_WRAP]
    c_sch  = params[P_SCHOOL]
    R      = params[P_R]
    c_gap  = params[P_GAP]
    N = f['N']

    T_tree = np.zeros(N)

    if len(f['sch_idx']) > 0:
        np.add.at(T_tree, f['sch_idx'], f['sch_work'] * c_sch)

    if len(f['fus_idx']) > 0:
        fus_pp = f['fus_floor'] + f['fus_ovhd'] * f['fus_cb'] / np.maximum(f['fus_eb'], 1.0)
        np.add.at(T_tree, f['fus_idx'], fus_pp)

    if len(f['cuf_idx']) > 0:
        calib_total = f['cuf_raw'] * (1.0 + R)
        ovhd = np.maximum(0.0, calib_total - f['cuf_floor'])
        cuf_pp = f['cuf_floor'] + ovhd * f['cuf_cb'] / np.maximum(f['cuf_eb'], 1.0)
        np.add.at(T_tree, f['cuf_idx'], f['cuf_nn'] * cuf_pp)
        gap_cost = np.zeros(N)
        np.add.at(gap_cost, f['cuf_idx'], f['cuf_gap'])
        T_tree += gap_cost * c_gap / f['qb']

    if len(f['wrap_idx']) > 0:
        np.add.at(T_tree, f['wrap_idx'], f['wrap_work'] * c_wrap)

    T_total = f['T_nontree'] + T_tree
    log_ratio = np.log(np.maximum(T_total, 1e-6)) - f['log_meas']
    return np.sum(log_ratio * log_ratio)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 tools/fit_gpu_cost_model.py <sample_plans.csv> <gpu_fft_config.h> [bench_kernels.csv]")
        sys.exit(1)

    plans_path = sys.argv[1]
    config_path = sys.argv[2]
    kernels_path = sys.argv[3] if len(sys.argv) > 3 else 'bench_kernels_b200.csv'

    calib = load_gpu_calib(config_path)
    print(f"Loaded {len(calib)} calibrated FFT sizes")

    plans = parse_plans(plans_path)
    print(f"Loaded {len(plans)} plans, n=[{min(p.n for p in plans)}..{max(p.n for p in plans)}]")

    kb = load_kernel_bench(kernels_path)
    interps = build_kernel_interps(kb)
    print(f"Loaded kernel bench: {len(kb['compute_a'])} compute_a, {len(kb['block_build'])} block_build, "
          f"{len(kb['leaf_extract'])} leaf_extract, {len(kb['accumulate'])} accumulate")

    f = precompute_non_tree(plans, interps, calib)

    # Diagnostic: non-tree only
    nontree_ratio = f['T_nontree'] / f['meas']
    print(f"\nNon-tree (empirical) / measured: mean={np.mean(nontree_ratio):.3f}")

    print(f"\nFitting {N_FIT} parameters (C_wrap, C_school, R, C_gap)...")
    res = differential_evolution(objective, FIT_BOUNDS, args=(f,),
                                 seed=42, maxiter=2000, tol=1e-14,
                                 popsize=30, mutation=(0.5, 1.5), recombination=0.9)
    params = res.x

    # Report
    log_errs = np.array([math.log(predict_plan(params, p, calib, interps) / p.per_qp_ns) for p in plans])
    rms = np.sqrt(np.mean(log_errs**2)) * 100
    max_err = np.max(np.abs(log_errs)) * 100

    print(f"\n{'='*70}")
    print(f"GPU COST MODEL — {len(plans)} plans, {N_FIT} fitted + empirical lookups")
    print(f"RMS log-relative error: {rms:.2f}%")
    print(f"Max log-relative error: {max_err:.2f}%")
    print(f"Measurement noise floor: ~0.4% CV")
    print(f"{'='*70}")
    print(f"\nFitted parameters:")
    for i in range(N_FIT):
        print(f"  {FIT_NAMES[i]:10s} = {params[i]:.6f}")

    # Error by n
    by_n = {}
    for p, le in zip(plans, log_errs):
        by_n.setdefault(p.n, []).append(le * 100)
    print(f"\nRMS error by n:")
    for n in sorted(by_n):
        errs = np.array(by_n[n])
        print(f"  n={n:7d}: {np.sqrt(np.mean(errs**2)):5.2f}%  (bias: {np.mean(errs):+5.2f}%)")

    # Worst 10
    errors = sorted(zip(np.abs(log_errs)*100, plans), reverse=True)
    print(f"\nWorst 10:")
    for err_pct, p in errors[:10]:
        pred = predict_plan(params, p, calib, interps)
        print(f"  n={p.n:7d} k={p.k:7d} B={p.B:4d}"
              f"  meas={p.per_qp_ns:9.0f} pred={pred:9.0f} err={err_pct:.2f}%")

    # Phase breakdown for a few plans
    print(f"\nPhase breakdown (selected plans):")
    print(f"  {'n':>6s} {'k':>6s} {'B':>4s} {'meas':>8s} {'pred':>8s} {'err':>6s} | {'cmp_a':>6s} {'block':>7s} {'tree':>7s} {'leaf':>7s} {'accum':>6s}")
    for p in sorted(plans, key=lambda p: (p.n, p.B)):
        if p.k != p.n: continue
        if p.n not in [4096, 16384, 65536]: continue
        pred = predict_plan(params, p, calib, interps)
        T_ca = float(interps['compute_a'](math.log(max(p.n, 256))))
        T_bl = lookup_block_build(interps, p.n, p.B)
        T_le = lookup_leaf_extract(interps, p.n, p.B)
        T_ac = float(interps['accumulate'](math.log(max(p.n, 256))))
        T_tree = pred - T_ca - T_bl - T_le - T_ac
        err = (pred / p.per_qp_ns - 1) * 100
        print(f"  {p.n:6d} {p.k:6d} {p.B:4d} {p.per_qp_ns:8.0f} {pred:8.0f} {err:+5.1f}% | {T_ca:6.0f} {T_bl:7.0f} {T_tree:7.0f} {T_le:7.0f} {T_ac:6.0f}")


if __name__ == '__main__':
    main()
