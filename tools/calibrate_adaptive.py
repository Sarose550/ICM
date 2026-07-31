#!/usr/bin/env python3
"""
calibrate_adaptive.py — Adaptive mesh-refinement calibration orchestrator.

Replaces calibrate_block_size.py with a priority-queue-driven adaptive
refinement strategy directly analogous to adaptive quadrature: maintain a
candidate pool scored by expected value of information, always measure the
highest-scoring candidate next, stop when a fixed budget is exhausted OR the
queue's top score drops below a convergence threshold.

Key fixes over the old orchestrator:
  1. No second binary call — the validate probe already runs a full
     candidate-timing search internally; we take its best_B directly.
  2. No --narrow-around guess — there is no second call to attach a guess
     to, and the old approach assumed the true answer was close to the
     (known-wrong) dispatched answer, which real data disproved.

Usage:
  python3 tools/calibrate_adaptive.py --device m3_pro --budget 100
  python3 tools/calibrate_adaptive.py --device b200 --budget 30m
  python3 tools/calibrate_adaptive.py --device zen4 --budget 2h --dry-run
"""

import argparse
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from typing import Optional

from calib_common import (DEVICE_META, inject_table, read_existing_table,
                          run_validate_probe, _draw_log_uniform_nk)


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

POOL_REFILL_SIZE = 200
DISAGREEMENT_BONUS = 1.0
DEFAULT_CONVERGENCE_THRESHOLD = 0.15


# ────────────────────────────────────────────────────────────────────────────
# Priority-queue scoring
# ────────────────────────────────────────────────────────────────────────────

def _joint_log_distance(n1: int, k1: int, n2: int, k2: int) -> float:
    """
    The exact same metric the production lookup itself uses:
    hypot(log(n1)-log(n2), log(k1)-log(k2)).
    """
    return math.hypot(math.log(n1) - math.log(n2),
                      math.log(k1) - math.log(k2))


def _joint_log_distance_to_nearest(
        n: int, k: int,
        live_table: dict[tuple[int, int], int]) -> float:
    """
    Minimum joint-log-distance from (n,k) to any currently-calibrated point.
    This is the primary "value of information" driver: points far from
    any calibrated anchor have higher priority.
    """
    if not live_table:
        return float('inf')
    best = float('inf')
    for (ni, ki) in live_table:
        d = _joint_log_distance(n, k, ni, ki)
        if d < best:
            best = d
    return best


def _two_nearest_disagree(n: int, k: int,
                          live_table: dict[tuple[int, int], int]) -> bool:
    """
    Return True if the two nearest calibrated points (by joint-log-distance)
    disagree on B.
    """
    if len(live_table) < 2:
        return False
    # Collect all distances, take the two smallest
    dists: list[tuple[float, int]] = []
    for (ni, ki), bi in live_table.items():
        d = _joint_log_distance(n, k, ni, ki)
        dists.append((d, bi))
    dists.sort(key=lambda x: x[0])
    return dists[0][1] != dists[1][1]


def _score_candidate(n: int, k: int,
                     live_table: dict[tuple[int, int], int]) -> float:
    """
    Score a candidate (n,k) for priority:
      distance to nearest calibrated point + (1.0 if 2-nearest disagree else 0.0)
    """
    dist = _joint_log_distance_to_nearest(n, k, live_table)
    bonus = DISAGREEMENT_BONUS if _two_nearest_disagree(n, k, live_table) else 0.0
    return dist + bonus


# ────────────────────────────────────────────────────────────────────────────
# Skeleton generation wrapper
# ────────────────────────────────────────────────────────────────────────────

def _run_skeleton_for_landmarks(
        device: str, lo: Optional[int], hi: Optional[int],
        ratio: Optional[float]) -> list[int]:
    """
    Call gen_calib_skeleton.py to get a sparse set of landmark n-values.
    Uses a coarser ratio (~3x the device's normal ratio) to get far fewer
    n-anchors than the full skeleton.
    """
    script = os.path.join(os.path.dirname(__file__), "gen_calib_skeleton.py")

    # Build command: pass through user-specified --lo/--hi/--ratio if set;
    # if ratio is not set, multiply the device's default by 3 for sparseness.
    cmd = [sys.executable, script, "--device", device]
    if lo is not None:
        cmd += ["--lo", str(lo)]
    if hi is not None:
        cmd += ["--hi", str(hi)]

    # Determine effective ratio: user-specified, or compute ~3x default
    if ratio is not None:
        effective_ratio = ratio
    else:
        is_gpu = (device == "b200")
        default_ratio = 1.8 if is_gpu else 1.6
        effective_ratio = default_ratio * 3.0
    cmd += ["--ratio", str(effective_ratio)]

    # Temporary files for output
    skel_fd, skel_path = tempfile.mkstemp(suffix=".csv", prefix="skeleton_")
    bands_fd, bands_path = tempfile.mkstemp(suffix=".csv", prefix="bands_")
    os.close(skel_fd)
    os.close(bands_fd)

    cmd += ["--skeleton-out", skel_path, "--bands-out", bands_path]

    print(f"[skeleton] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[skeleton] stderr:\n{result.stderr}")
        raise RuntimeError(
            f"gen_calib_skeleton.py failed with code {result.returncode}")
    print(result.stderr.strip())

    # Parse skeleton CSV — extract just the unique n values
    landmark_ns: set[int] = set()
    with open(skel_path, "r") as f:
        header = f.readline()  # skip "n,k"
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_str = line.split(",")[0]
            landmark_ns.add(int(n_str))

    os.unlink(skel_path)
    os.unlink(bands_path)
    return sorted(landmark_ns)


# ────────────────────────────────────────────────────────────────────────────
# Budget parsing
# ────────────────────────────────────────────────────────────────────────────

def _parse_budget(budget_str: str) -> tuple[Optional[int], Optional[float]]:
    """
    Parse --budget argument.

    Returns (max_probes, max_seconds). Exactly one will be None.

    Bare integer → probe-count cap.
    Suffixed: 30s, 20m, 2h → time cap in seconds.
    """
    budget_str = budget_str.strip()
    if budget_str[-1].isdigit():
        # Bare integer
        return (int(budget_str), None)
    suffix = budget_str[-1].lower()
    number_str = budget_str[:-1]
    if suffix not in ('s', 'm', 'h'):
        raise argparse.ArgumentTypeError(
            f"Unrecognized budget suffix: '{suffix}'. "
            f"Use N, Ns, Nm, or Nh.")
    try:
        value = float(number_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid budget number: '{number_str}'")
    if suffix == 's':
        seconds = value
    elif suffix == 'm':
        seconds = value * 60
    else:  # 'h'
        seconds = value * 3600
    return (None, seconds)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive calibration orchestrator — priority-queue "
                    "refinement replacing calibrate_block_size.py."
    )
    parser.add_argument("--device", required=True,
                        choices=["m3_pro", "zen4", "b200"])
    parser.add_argument("--budget", required=True, type=str,
                        help="Probe budget: bare integer (probe count) or "
                             "time-suffixed (30s, 20m, 2h). Required, no "
                             "default — force an explicit choice every run.")
    parser.add_argument("--validate-bin", type=str, default=None,
                        help="Path to validate_best_b / validate_planner_gpu "
                             "binary.")
    parser.add_argument("--n-min", type=int, default=None,
                        help="Minimum n to calibrate (inclusive). Default: "
                             "the device's full real domain -- only narrow "
                             "this for a deliberately scoped/cheap run.")
    parser.add_argument("--n-max", type=int, default=None,
                        help="Maximum n to calibrate (inclusive). Default: "
                             "the device's full real domain -- only narrow "
                             "this for a deliberately scoped/cheap run.")
    parser.add_argument("--landmark-ratio", type=float, default=None,
                        help="Log-spacing ratio between landmark n-anchors "
                             "(if set, used directly; else ~3x the device's "
                             "own calibration ratio, for sparse seeding).")
    parser.add_argument("--config-header", type=str, default=None,
                        help="Override path to config header (for testing).")
    parser.add_argument("--convergence-threshold", type=float,
                        default=DEFAULT_CONVERGENCE_THRESHOLD,
                        help=f"Stop when top pool score drops below this "
                             f"(default {DEFAULT_CONVERGENCE_THRESHOLD}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run Step 1 skeleton generation only, print "
                             "a domain summary, then exit before any "
                             "measurement.")
    args = parser.parse_args()

    device = args.device
    meta = dict(DEVICE_META[device])  # shallow copy so we can override
    if args.config_header:
        meta["config_header"] = args.config_header
    is_gpu = meta["is_gpu"]

    # Budget
    max_probes, max_seconds = _parse_budget(args.budget)
    if max_probes is not None:
        budget_label = f"{max_probes} probes"
    else:
        budget_label = args.budget

    # Validate binary (the Makefile places these at the repo root, not
    # build/ -- confirmed 2026-07-31 after this default silently pointed
    # at a nonexistent path, see VERDICTS.md V6)
    validate_bin = args.validate_bin
    if validate_bin is None:
        if is_gpu:
            validate_bin = "./validate_planner_gpu"
        else:
            validate_bin = "./validate_best_b"

    if not os.path.isfile(validate_bin):
        print(f"WARNING: validate binary not found at '{validate_bin}'. "
              f"Will attempt to run anyway (may fail if not on PATH).",
              file=sys.stderr)

    # ── Step 1: Skeleton generation for landmarks ──────────────────────
    print("── Step 1: Generate sparse landmark n-anchors ──")
    landmark_ns = _run_skeleton_for_landmarks(
        device, args.n_min, args.n_max, args.landmark_ratio)

    # Determine effective domain lo/hi for reporting
    # (gen_calib_skeleton.py applied its defaults; use the min/max
    # from the landmark set itself as a reasonable proxy, and also
    # re-derive from gen_calib_skeleton's own defaults for reporting.)
    if args.n_min is not None:
        domain_lo = args.n_min
    else:
        domain_lo = 1024 if is_gpu else 256
    if args.n_max is not None:
        domain_hi = args.n_max
    else:
        domain_hi = 33554432 if is_gpu else 65536

    # Build the landmark (n,k) set
    landmark_points: list[tuple[int, int]] = []
    for n in landmark_ns:
        ks: set[int] = set()
        # k = 2, k = 16
        ks.add(2)
        ks.add(16)
        # n/8, n/4, n/2, n (clamped to [2, n])
        for denom in (8, 4, 2, 1):
            k = max(2, min(n, n // denom))
            ks.add(k)
        for k in sorted(ks):
            landmark_points.append((n, k))

    print(f"  Domain: lo={domain_lo}, hi={domain_hi}")
    print(f"  Landmark n-anchors: {len(landmark_ns)}")
    print(f"  Landmark (n,k) points: {len(landmark_points)} "
          f"(~{len(landmark_points)//max(1,len(landmark_ns))} per n)")
    print(f"  Budget: {budget_label}")
    print()

    if args.dry_run:
        print("  --dry-run: stopping before any measurement.")
        print(f"  Would run Step 0 (load existing) + Step 1 landmark "
              f"probing ({len(landmark_points)} candidate probes) "
              f"+ Step 2 adaptive refinement up to budget {budget_label}.")
        return

    # ── Step 0: Load existing table ────────────────────────────────────
    existing_table = read_existing_table(meta["config_header"], meta)
    live_table: dict[tuple[int, int], int] = {
        (n, k): b for n, k, b in existing_table}
    if live_table:
        print(f"── Step 0: Loaded {len(live_table)} pre-existing calibrated "
              f"points from {meta['config_header']} (will be preserved) ──")
        print()

    # ── Step 1 (continued): Probe landmark points ──────────────────────
    # Skip any landmark (n,k) already in the existing table
    landmarks_to_probe = [(n, k) for n, k in landmark_points
                          if (n, k) not in live_table]

    if landmarks_to_probe:
        print(f"── Step 1 (probe): Measuring {len(landmarks_to_probe)} "
              f"landmark points ──")
    else:
        print(f"── Step 1 (probe): All {len(landmark_points)} landmark "
              f"points already in table, nothing to measure ──")

    probe_count = 0
    run_start = time.monotonic()
    probed_this_run: set[tuple[int, int]] = set()
    total_points_added = 0

    for n, k in landmarks_to_probe:
        # Budget check before each probe
        if max_probes is not None and probe_count >= max_probes:
            print(f"\n  Budget exhausted ({probe_count} probes).")
            break
        if max_seconds is not None:
            elapsed = time.monotonic() - run_start
            if elapsed >= max_seconds:
                print(f"\n  Budget exhausted ({elapsed:.1f}s elapsed).")
                break

        probe_start = time.monotonic()
        probe = run_validate_probe(validate_bin, n, k, is_gpu)
        probe_elapsed = time.monotonic() - probe_start
        probe_count += 1
        probed_this_run.add((n, k))
        total_points_added += 1

        live_table[(n, k)] = probe["best_B"]

        elapsed_total = time.monotonic() - run_start
        avg_probe_s = elapsed_total / probe_count

        # Progress line
        est_remaining = ""
        if max_seconds is not None and avg_probe_s > 0:
            remaining_budget_s = max_seconds - elapsed_total
            est_probes_left = int(remaining_budget_s / avg_probe_s)
            est_remaining = f" ~{est_probes_left} est. remaining"
        elif max_probes is not None:
            est_remaining = f" {max_probes - probe_count} remaining"

        print(f"  [{probe_count}] n={n} k={k} "
              f"auto_B={probe['auto_B']} best_B={probe['best_B']} "
              f"gap={probe['gap_pct']:.2f}% "
              f"({probe_elapsed:.1f}s this, "
              f"{elapsed_total/60:.1f}m total, "
              f"{avg_probe_s:.1f}s/probe avg"
              f"{est_remaining})")

    # Batch-inject after landmark pass
    if landmarks_to_probe:
        inject_table(meta["config_header"], meta,
                     [(n, k, live_table[(n, k)]) for n, k in live_table])
        print(f"  → Injected {len(live_table)} total points after "
              f"landmark pass\n")

    # ── Step 2: Priority-queue adaptive refinement ─────────────────────
    print("── Step 2: Priority-queue adaptive refinement ──")
    print()

    # Determine n_lo/n_hi for log-uniform draws over the FULL domain
    n_lo = float(domain_lo)
    n_hi = float(domain_hi)

    pool: list[tuple[int, int]] = []
    pool_refilled_this_iteration = False
    converged = False
    budget_capped = False

    while True:
        # 1. Refill pool if empty
        if not pool:
            fresh = []
            attempts = 0
            max_attempts = POOL_REFILL_SIZE * 10  # safety valve
            exclude = set(live_table.keys()) | probed_this_run
            while len(fresh) < POOL_REFILL_SIZE and attempts < max_attempts:
                n, k = _draw_log_uniform_nk(n_lo, n_hi, exclude)
                if (n, k) not in exclude:
                    fresh.append((n, k))
                    exclude.add((n, k))
                attempts += 1
            pool = fresh
            pool_refilled_this_iteration = True
            if not pool:
                # Couldn't find any more candidates — domain exhausted
                print("  Pool exhausted (domain fully covered). Converged.")
                converged = True
                break

        # 2. Score all pool candidates against current live_table
        best_score = -1.0
        best_candidate: Optional[tuple[int, int]] = None
        best_idx: int = -1
        for i, (n, k) in enumerate(pool):
            s = _score_candidate(n, k, live_table)
            if s > best_score:
                best_score = s
                best_candidate = (n, k)
                best_idx = i

        # 3. Convergence check
        if pool_refilled_this_iteration:
            if best_score < args.convergence_threshold:
                print(f"  Top pool score {best_score:.4f} < "
                      f"{args.convergence_threshold} "
                      f"(convergence threshold) on fresh refill. Converged.")
                converged = True
                break
            # Not converged yet — clear the flag for subsequent iterations
            # (only the very first pop after a fresh refill is the
            # convergence checkpoint)

        # Pop the candidate
        n, k = best_candidate  # type: ignore[misc]
        del pool[best_idx]
        pool_refilled_this_iteration = False

        # 4. Measure
        # Budget check BEFORE probe
        if max_probes is not None and probe_count >= max_probes:
            print(f"\n  Budget exhausted ({probe_count} probes reached "
                  f"cap of {max_probes}).")
            budget_capped = True
            break
        if max_seconds is not None:
            elapsed = time.monotonic() - run_start
            if elapsed >= max_seconds:
                print(f"\n  Budget exhausted ({elapsed:.1f}s elapsed >= "
                      f"{max_seconds}s).")
                budget_capped = True
                break

        probe_start = time.monotonic()
        probe = run_validate_probe(validate_bin, n, k, is_gpu)
        probe_elapsed = time.monotonic() - probe_start
        probe_count += 1
        probed_this_run.add((n, k))
        total_points_added += 1

        live_table[(n, k)] = probe["best_B"]

        # Immediate re-injection (crash-safety)
        inject_table(meta["config_header"], meta,
                     [(pn, pk, live_table[(pn, pk)])
                      for pn, pk in live_table])

        elapsed_total = time.monotonic() - run_start
        avg_probe_s = elapsed_total / probe_count

        # Progress line
        est_remaining = ""
        if max_seconds is not None and avg_probe_s > 0:
            remaining_budget_s = max_seconds - elapsed_total
            est_probes_left = int(remaining_budget_s / avg_probe_s)
            est_remaining = f" ~{est_probes_left} est. remaining"
        elif max_probes is not None:
            est_remaining = f" {max_probes - probe_count} remaining"

        print(f"  [{probe_count}] n={n} k={k} "
              f"auto_B={probe['auto_B']} best_B={probe['best_B']} "
              f"gap={probe['gap_pct']:.2f}% "
              f"score={best_score:.4f} "
              f"({probe_elapsed:.1f}s this, "
              f"{elapsed_total/60:.1f}m total, "
              f"{avg_probe_s:.1f}s/probe avg"
              f"{est_remaining})")

        # 5. Loop to 1.

    # ── Step 3: Final report ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"  Device:               {device}")
    print(f"  Domain:               lo={domain_lo}, hi={domain_hi}")
    print(f"  Budget:               {budget_label}")
    print(f"  Total probes:         {probe_count}")
    print(f"  Points added:         {total_points_added}")
    print(f"  Live table size:      {len(live_table)}")

    if converged:
        print(f"  Status:               CONVERGED "
              f"(threshold={args.convergence_threshold})")
    elif budget_capped:
        print(f"  Status:               BUDGET-CAPPED (not necessarily "
              f"converged)")
        # Compute current top-of-pool score for caller signal
        if pool:
            top_score = max(_score_candidate(n, k, live_table)
                            for n, k in pool)
            print(f"  Top remaining score:  {top_score:.4f}")
            if top_score > args.convergence_threshold * 3:
                print(f"    → Score well above threshold — more budget "
                      f"would likely help.")
            elif top_score > args.convergence_threshold:
                print(f"    → Score moderately above threshold — "
                      f"run was close to done.")
            else:
                print(f"    → Score near/below threshold — run was "
                      f"nearly converged anyway.")
        else:
            print(f"  Top remaining score:  N/A (pool empty)")
    else:
        print(f"  Status:               STOPPED (unexpected exit)")

    print()


if __name__ == "__main__":
    main()
