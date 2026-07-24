#!/usr/bin/env python3
"""
gen_calib_skeleton.py — Shared calibration skeleton + band-boundary generator.

Parameterized by device; produces the full calibration point list and band
boundaries per the adaptive calibration methodology.

Smooth-number logic is ported EXACTLY from the codebase:
  - CPU: build_fftw_size_table() in src/icm.c (lines ~524-536)
  - GPU: build_smooth_table()   in src/gpu/gpu_plan.cu (lines ~50-66)

Usage:
  python3 tools/gen_calib_skeleton.py --device m3_pro  > skeleton_m3_pro.csv
  python3 tools/gen_calib_skeleton.py --device zen4    > skeleton_zen4.csv
  python3 tools/gen_calib_skeleton.py --device b200    > skeleton_b200.csv

Also emits bands_<device>.csv (band_id, n_lo, n_hi, skeleton_n) for the
downstream adaptive orchestrator.
"""

import argparse
import math
import sys


# ────────────────────────────────────────────────────────────────────────────
# 7-smooth number generation — EXACT ports from the codebase
# ────────────────────────────────────────────────────────────────────────────

def generate_7smooth_cpu(max_n: int) -> list[int]:
    """
    Exact port of build_fftw_size_table() from src/icm.c lines 524-536.

    Hardcoded to 131072 in the original; parameterized here so we can
    reuse the identical loop structure at any cap.  No early-break
    optimisation, insertion sort, no dedup needed (each (h,i,j,k)
    tuple is unique).
    """
    smooth: list[int] = []
    a = 1
    while a <= max_n:
        b = a
        while b <= max_n:
            c = b
            while c <= max_n:
                d = c
                while d <= max_n:
                    smooth.append(d)
                    d *= 7
                c *= 5
            b *= 3
        a *= 2

    # Insertion sort — matches src/icm.c exactly (~500 elements, one-time)
    for i in range(1, len(smooth)):
        key = smooth[i]
        j = i - 1
        while j >= 0 and smooth[j] > key:
            smooth[j + 1] = smooth[j]
            j -= 1
        smooth[j + 1] = key

    return smooth


def generate_7smooth_gpu(max_n: int) -> list[int]:
    """
    Exact port of build_smooth_table() from src/gpu/gpu_plan.cu lines 50-66.

    Parameterised max_n, early-break guards, std::sort + unique.
    """
    smooth: list[int] = []
    a = 1
    while a <= max_n:
        b = a
        while b <= max_n:
            c = b
            while c <= max_n:
                d = c
                while d <= max_n:
                    smooth.append(d)
                    if d > max_n // 7:
                        break
                    d *= 7
                if c > max_n // 5:
                    break
                c *= 5
            if b > max_n // 3:
                break
            b *= 3
        if a > max_n // 2:
            break
        a *= 2

    smooth.sort()
    # Deduplicate (std::unique equivalent)
    uniq: list[int] = []
    for v in smooth:
        if not uniq or uniq[-1] != v:
            uniq.append(v)
    return uniq


def generate_7smooth(max_n: int, *, use_gpu_algo: bool = False) -> list[int]:
    """Unified entry point — both algorithms produce identical output."""
    if use_gpu_algo:
        return generate_7smooth_gpu(max_n)
    else:
        return generate_7smooth_cpu(max_n)


# ────────────────────────────────────────────────────────────────────────────
# Self-check
# ────────────────────────────────────────────────────────────────────────────

def is_7smooth(n: int) -> bool:
    """Return True iff all prime factors of n are in {2, 3, 5, 7}."""
    if n <= 0:
        return False
    x = n
    for p in (2, 3, 5, 7):
        while x % p == 0:
            x //= p
    return x == 1


# ────────────────────────────────────────────────────────────────────────────
# Skeleton n selection — log-spaced targets snapped to nearest 7-smooth
# ────────────────────────────────────────────────────────────────────────────

def pick_skeleton_n(lo: int, hi: int, ratio: float,
                    smooth: list[int]) -> list[int]:
    """
    Generate log-spaced targets from *lo* to *hi* stepping by *ratio*,
    snap each to the nearest 7-smooth number by log-distance, deduplicate,
    and return sorted.

    The *hi* cap is always included if it is 7-smooth (ensures coverage
    of the full calibrated FFT-size range).
    """
    if lo < 1:
        raise ValueError(f"lo={lo} must be >= 1")
    if hi < lo:
        raise ValueError(f"hi={hi} < lo={lo}")
    if ratio <= 1.0:
        raise ValueError(f"ratio={ratio} must be > 1.0")

    # Build a working set of smooth numbers — go beyond hi so every
    # target has candidates on both sides.
    work_smooth = [s for s in smooth if lo <= s <= hi * ratio]
    log_smooth = [math.log(s) for s in work_smooth]

    selected: list[int] = []
    i = 0
    while True:
        target = lo * (ratio ** i)
        if target > hi * ratio:
            break  # one full step past hi — enough for snapping
        i += 1

        # Find nearest by log-distance
        log_t = math.log(target)
        best_idx = 0
        best_dist = float('inf')
        for idx, ls in enumerate(log_smooth):
            d = abs(ls - log_t)
            if d < best_dist:
                best_dist = d
                best_idx = idx

        s = work_smooth[best_idx]
        if lo <= s <= hi and s not in selected:
            selected.append(s)

    selected.sort()

    # Ensure hi is included if it is 7-smooth (the snapping target for hi
    # may land on a neighbour instead).
    if is_7smooth(hi) and hi not in selected:
        selected.append(hi)
        selected.sort()

    # Also ensure lo is included if 7-smooth
    if is_7smooth(lo) and lo not in selected:
        selected.append(lo)
        selected.sort()

    return selected


# ────────────────────────────────────────────────────────────────────────────
# k-anchor expansion per skeleton n
# ────────────────────────────────────────────────────────────────────────────

def k_anchors_for_n(n: int, smooth_up_to_257: list[int]) -> list[int]:
    """
    Return the full k-anchor set for a given skeleton n, per the board's
    Context section item 3:

      1. {2 .. 16}                              (tiny, exhaustive)
      2. {s-1 : s 7-smooth, 16 < s-1 <= 256, s-1 <= n}
      3. {n/12, n/10, n/8, n/6, n/4, n/3, n/2, n}

    Deduplicate; drop k > n or k < 1.
    """
    kset: set[int] = set()

    # (1) Tiny exhaustive
    for k in range(2, 17):
        if k <= n:
            kset.add(k)

    # (2) Almost-7-smooth: k = s-1 where s is 7-smooth
    for s in smooth_up_to_257:
        k = s - 1
        if 16 < k <= 256 and k <= n:
            kset.add(k)

    # (3) Relative fractions
    for denom in (12, 10, 8, 6, 4, 3, 2, 1):
        k = n // denom
        if 1 <= k <= n:
            kset.add(k)

    # Drop k < 1 (already handled) and sort
    return sorted(kset)


# ────────────────────────────────────────────────────────────────────────────
# Band boundaries — midpoints in log-space between consecutive skeleton n
# ────────────────────────────────────────────────────────────────────────────

def build_bands(skeleton: list[int], lo: int, hi: int) -> list[dict]:
    """
    Partition the n-axis into bands, one per skeleton n.
    Band boundary = midpoint in log-space between consecutive skeleton n values.
    First band extends down to *lo*; last band extends up to *hi*.
    """
    bands: list[dict] = []
    m = len(skeleton)
    for i, sn in enumerate(skeleton):
        if i == 0:
            n_lo = lo
        else:
            n_lo = math.exp((math.log(skeleton[i - 1]) + math.log(sn)) / 2.0)
        if i == m - 1:
            n_hi = hi
        else:
            n_hi = math.exp((math.log(sn) + math.log(skeleton[i + 1])) / 2.0)
        bands.append({
            'band_id': i,
            'n_lo': n_lo,
            'n_hi': n_hi,
            'skeleton_n': sn,
        })
    return bands


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate calibration skeleton and band-boundary CSVs.'
    )
    parser.add_argument(
        '--device', required=True,
        choices=['m3_pro', 'zen4', 'b200'],
        help='Target device.  m3_pro/zen4 share CPU smooth-number logic; '
             'b200 uses GPU smooth-number logic.'
    )
    parser.add_argument(
        '--lo', type=int, default=None,
        help='Minimum n (inclusive).  CPU default 256; GPU default 1024.'
    )
    parser.add_argument(
        '--hi', type=int, default=None,
        help='Maximum n (inclusive).  CPU default 65536 (keeps the k=n root-level '
             'FFT length 2n-1 within the calibrated FFT table, which caps at '
             '131072); GPU default 4194304.'
    )
    parser.add_argument(
        '--ratio', type=float, default=None,
        help='Log-spacing ratio between consecutive skeleton n anchors.  '
             'CPU default 1.6 (~14 n-anchors); GPU default 1.8 (~15 n-anchors).'
    )
    parser.add_argument(
        '--bands-out', type=str, default=None,
        help='Path for bands CSV output.  Default: bands_<device>.csv'
    )
    parser.add_argument(
        '--skeleton-out', type=str, default=None,
        help='Path for skeleton CSV output.  Default: stdout (for > redirection).'
    )
    args = parser.parse_args()

    # ── Device-specific defaults ──────────────────────────────────────
    is_gpu = (args.device == 'b200')

    lo = args.lo
    hi = args.hi
    ratio = args.ratio

    if lo is None:
        lo = 1024 if is_gpu else 256
    if hi is None:
        hi = 4194304 if is_gpu else 65536
    if ratio is None:
        ratio = 1.8 if is_gpu else 1.6

    # ── Generate 7-smooth numbers ─────────────────────────────────────
    # GPU algorithm for b200 (parameterised, early-break), CPU algorithm
    # for m3_pro/zen4 (matches build_fftw_size_table exactly).
    # We generate up to max(hi * 2, 257) so we have candidates for snapping
    # AND for the almost-7-smooth k-anchor set.
    max_needed = max(hi * 2, 257)
    smooth = generate_7smooth(max_needed, use_gpu_algo=is_gpu)

    # ── Pick skeleton n values ────────────────────────────────────────
    skeleton = pick_skeleton_n(lo, hi, ratio, smooth)

    # ── Self-check: every skeleton n must be 7-smooth ─────────────────
    bad = [n for n in skeleton if not is_7smooth(n)]
    if bad:
        print(f"ERROR: skeleton n values are not 7-smooth: {bad}", file=sys.stderr)
        sys.exit(1)

    # Verify all skeleton n are in the smooth list
    smooth_set = set(smooth)
    missing = [n for n in skeleton if n not in smooth_set]
    if missing:
        print(f"ERROR: skeleton n values not in generated smooth list: {missing}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[{args.device}] lo={lo} hi={hi} ratio={ratio} -> "
          f"{len(skeleton)} skeleton n-anchors: {skeleton}",
          file=sys.stderr)

    # ── k-anchor expansion ────────────────────────────────────────────
    # Pre-compute 7-smooth numbers up to 257 for the almost-7-smooth category
    smooth_up_to_257 = [s for s in smooth if s <= 257]

    all_points: list[tuple[int, int]] = []
    for n in skeleton:
        ks = k_anchors_for_n(n, smooth_up_to_257)
        for k in ks:
            all_points.append((n, k))

    print(f"[{args.device}] {len(all_points)} total (n,k) calibration points",
          file=sys.stderr)

    # Quick estimate check — the three k-anchor categories produce
    # 15 (tiny) + ~63 (almost-7-smooth) + 8 (rel. fractions) ≈ 86 per n
    # with minimal overlap for n ≥ 256.
    points_per_n = len(all_points) / len(skeleton) if skeleton else 0
    print(f"[{args.device}] ~{points_per_n:.1f} k-anchors per skeleton n "
          f"(15 tiny + ~63 almost-7-smooth + 8 rel. fractions)",
          file=sys.stderr)

    # ── Emit skeleton CSV ─────────────────────────────────────────────
    skel_lines = ["n,k"]
    for n, k in all_points:
        skel_lines.append(f"{n},{k}")
    skel_out = "\n".join(skel_lines) + "\n"

    if args.skeleton_out:
        with open(args.skeleton_out, 'w') as f:
            f.write(skel_out)
        print(f"[{args.device}] Skeleton written to {args.skeleton_out}",
              file=sys.stderr)
    else:
        sys.stdout.write(skel_out)

    # ── Emit bands CSV ────────────────────────────────────────────────
    bands = build_bands(skeleton, lo, hi)
    bands_lines = ["band_id,n_lo,n_hi,skeleton_n"]
    for b in bands:
        bands_lines.append(
            f"{b['band_id']},{b['n_lo']:.6f},{b['n_hi']:.6f},{b['skeleton_n']}"
        )
    bands_out = "\n".join(bands_lines) + "\n"

    bands_path = args.bands_out
    if bands_path is None:
        bands_path = f"bands_{args.device}.csv"

    with open(bands_path, 'w') as f:
        f.write(bands_out)
    print(f"[{args.device}] Bands written to {bands_path} ({len(bands)} bands)",
          file=sys.stderr)


if __name__ == '__main__':
    main()
