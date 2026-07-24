#!/usr/bin/env python3
"""
calibrate_block_size.py — Adaptive calibration orchestrator.

One command per device.  Drives skeleton generation, base-table sweep,
and per-band adaptive refinement with immediate re-injection.  Does NOT
rebuild the C library or run verify — that is the calling node's job.

Usage:
  python3 tools/calibrate_block_size.py --device m3_pro
      --calibrate-bin ./build/calibrate_best_b
      --validate-bin ./build/validate_best_b

  python3 tools/calibrate_block_size.py --device b200
      --calibrate-bin ./build/calibrate_gpu_best_b
      --validate-bin ./build/validate_planner_gpu

Full CLI:
  --device {m3_pro,zen4,b200}
  [--clean-streak-target 25] [--max-probes-per-band 150] [--gap-threshold 0.02]
  [--calibrate-bin PATH] [--validate-bin PATH]
  [--skeleton-lo N] [--skeleton-hi N] [--skeleton-ratio F]
"""

import argparse
import csv
import io
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# Device metadata
# ────────────────────────────────────────────────────────────────────────────

DEVICE_META = {
    "m3_pro": {
        "is_gpu": False,
        "config_header": "devices/m3_pro/fft_config.h",
        "array_prefix": "bselect",
        "n_macro": "N_BSELECT_POINTS",
        "fallback_B": 32,
    },
    "zen4": {
        "is_gpu": False,
        "config_header": "devices/zen4/fft_config.h",
        "array_prefix": "bselect",
        "n_macro": "N_BSELECT_POINTS",
        "fallback_B": 32,
    },
    "b200": {
        "is_gpu": True,
        "config_header": "devices/b200/gpu_fft_config.h",
        "array_prefix": "gbselect",
        "n_macro": "GPU_N_BSELECT_POINTS",
        "fallback_B": 64,
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Skeleton generation (subprocess call to gen_calib_skeleton.py)
# ────────────────────────────────────────────────────────────────────────────

def run_skeleton_generator(device: str, lo: Optional[int], hi: Optional[int],
                           ratio: Optional[float]) -> tuple[list[tuple[int, int]],
                                                            list[dict]]:
    """Call tools/gen_calib_skeleton.py, return (points, bands)."""
    script = os.path.join(os.path.dirname(__file__), "gen_calib_skeleton.py")
    cmd = [sys.executable, script, "--device", device]
    if lo is not None:
        cmd += ["--lo", str(lo)]
    if hi is not None:
        cmd += ["--hi", str(hi)]
    if ratio is not None:
        cmd += ["--ratio", str(ratio)]

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
        raise RuntimeError(f"gen_calib_skeleton.py failed with code {result.returncode}")
    print(result.stderr.strip())

    # Parse skeleton CSV
    points: list[tuple[int, int]] = []
    with open(skel_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((int(row["n"]), int(row["k"])))

    # Parse bands CSV
    bands: list[dict] = []
    with open(bands_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bands.append({
                "band_id": int(row["band_id"]),
                "n_lo": float(row["n_lo"]),
                "n_hi": float(row["n_hi"]),
                "skeleton_n": int(row["skeleton_n"]),
            })

    os.unlink(skel_path)
    os.unlink(bands_path)
    return points, bands


# ────────────────────────────────────────────────────────────────────────────
# Calibrate binary interface
# ────────────────────────────────────────────────────────────────────────────

def run_calibrate_sweep(calibrate_bin: str, input_csv_path: str,
                        output_csv_path: str, is_gpu: bool,
                        narrow_around: Optional[list[int]] = None) -> list[tuple[int, int, int]]:
    """Run calibrate binary on input CSV, return list of (n, k, best_B)."""
    if is_gpu:
        cmd = [calibrate_bin, input_csv_path, output_csv_path]
        if narrow_around:
            cmd += ["--narrow-around", ",".join(str(b) for b in narrow_around)]
    else:
        cmd = [calibrate_bin, input_csv_path, "-o", output_csv_path]
        if narrow_around:
            cmd += ["--narrow-around", ",".join(str(b) for b in narrow_around)]

    print(f"[calibrate] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stderr.strip())
    if result.returncode != 0:
        # Print stdout too on failure for debugging
        if result.stdout:
            print(f"[calibrate] stdout:\n{result.stdout}")
        raise RuntimeError(f"Calibrate binary failed with code {result.returncode}")

    # Parse output CSV
    rows: list[tuple[int, int, int]] = []
    with open(output_csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("n,k,") or line.startswith("n,k,best_B"):
                continue
            parts = line.split(",")
            if len(parts) == 3:
                try:
                    n = int(parts[0])
                    k = int(parts[1])
                    b = int(parts[2])
                    rows.append((n, k, b))
                except ValueError:
                    continue
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Validate binary interface (single-point probe oracle)
# ────────────────────────────────────────────────────────────────────────────

def run_validate_probe(validate_bin: str, n: int, k: int,
                       is_gpu: bool) -> dict:
    """
    Call validate binary in single-point-probe mode.
    Returns dict with keys: auto_B, auto_ms, best_B, best_ms, gap_pct.

    CPU contract (validate_best_b):
      Output: "n,k,auto_B,auto_ms,best_B,best_ms,gap_pct"
      auto_ms/best_ms are in NANOSECONDS PER QP (despite "_ms" name).
      gap_pct is floored at 0.0.

    GPU contract (validate_planner_gpu):
      Output: "auto_B,auto_ms,best_B,best_ms,gap_pct" (NO leading n,k).
      auto_ms/best_ms are in MILLISECONDS.
      gap_pct is NOT floored.
    """
    if is_gpu:
        cmd = [validate_bin, str(n), str(k)]
    else:
        cmd = [validate_bin, str(n), str(k)]

    print(f"[validate] n={n} k={k}  ->  ", end="", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        print(f"  stderr: {result.stderr.strip()}")
        raise RuntimeError(f"Validate binary failed for n={n} k={k}")

    line = result.stdout.strip().split("\n")[-1]  # last non-empty line
    parts = line.split(",")

    if is_gpu:
        # GPU: "auto_B,auto_ms,best_B,best_ms,gap_pct" (5 columns)
        if len(parts) != 5:
            raise RuntimeError(f"Unexpected GPU validate output: {line}")
        auto_B = int(parts[0])
        auto_ms = float(parts[1])
        best_B = int(parts[2])
        best_ms = float(parts[3])
        gap_pct = float(parts[4])
    else:
        # CPU: "n,k,auto_B,auto_ms,best_B,best_ms,gap_pct" (7 columns)
        if len(parts) != 7:
            raise RuntimeError(f"Unexpected CPU validate output: {line}")
        # n, k from output (columns 0,1) — we already know them
        auto_B = int(parts[2])
        auto_ms = float(parts[3])
        best_B = int(parts[4])
        best_ms = float(parts[5])
        gap_pct = float(parts[6])

    print(f"auto_B={auto_B} best_B={best_B} gap={gap_pct:.2f}%")
    return {
        "auto_B": auto_B,
        "auto_ms": auto_ms,
        "best_B": best_B,
        "best_ms": best_ms,
        "gap_pct": gap_pct,
    }


# ────────────────────────────────────────────────────────────────────────────
# Config header injection
# ────────────────────────────────────────────────────────────────────────────

def inject_table(config_path: str, device_meta: dict,
                 table: list[tuple[int, int, int]]) -> None:
    """
    Replace the bselect_n[]/bselect_k[]/bselect_B[] (CPU) or
    gbselect_n[]/gbselect_k[]/gbselect_B[] (GPU) arrays in-place.
    table is a list of (n, k, best_B) tuples.
    """
    prefix = device_meta["array_prefix"]
    n_macro = device_meta["n_macro"]

    with open(config_path, "r") as f:
        text = f.read()

    n_points = len(table)
    n_vals = [p[0] for p in table]
    k_vals = [p[1] for p in table]
    b_vals = [p[2] for p in table]

    # ── Update the N macro ──
    text = re.sub(
        rf'#define\s+{n_macro}\s+\d+',
        f'#define {n_macro} {n_points}',
        text,
    )

    # ── Replace n array ──
    n_pattern = rf'(static const int {prefix}_n\[{n_macro}\]\s*=\s*\{{)'
    n_match = re.search(n_pattern, text)
    if not n_match:
        raise RuntimeError(f"{prefix}_n[] array not found in {config_path}")
    # Find the matching closing brace — we need to find from the opening brace
    # of the initializer (which starts at n_match.end() - 1)
    brace_start = n_match.end() - 1  # position of '{'
    brace_end = _find_matching_brace(text, brace_start)
    new_n_array = _format_int_array(f"static const int {prefix}_n[{n_macro}]", n_vals)
    text = text[:n_match.start()] + new_n_array + text[brace_end + 1:]

    # ── Replace k array ──
    k_pattern = rf'(static const int {prefix}_k\[{n_macro}\]\s*=\s*\{{)'
    k_match = re.search(k_pattern, text)
    if not k_match:
        raise RuntimeError(f"{prefix}_k[] array not found in {config_path}")
    brace_start = k_match.end() - 1
    brace_end = _find_matching_brace(text, brace_start)
    new_k_array = _format_int_array(f"static const int {prefix}_k[{n_macro}]", k_vals)
    text = text[:k_match.start()] + new_k_array + text[brace_end + 1:]

    # ── Replace B array ──
    b_pattern = rf'(static const int {prefix}_B\[{n_macro}\]\s*=\s*\{{)'
    b_match = re.search(b_pattern, text)
    if not b_match:
        raise RuntimeError(f"{prefix}_B[] array not found in {config_path}")
    brace_start = b_match.end() - 1
    brace_end = _find_matching_brace(text, brace_start)
    new_b_array = _format_int_array(f"static const int {prefix}_B[{n_macro}]", b_vals)
    text = text[:b_match.start()] + new_b_array + text[brace_end + 1:]

    with open(config_path, "w") as f:
        f.write(text)

    print(f"[inject] Wrote {n_points} points to {config_path} "
          f"({prefix}_n/{prefix}_k/{prefix}_B)")


def _find_matching_brace(text: str, open_pos: int) -> int:
    """Given position of '{', return position of matching '}'."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unmatched brace")


def _format_int_array(decl: str, values: list[int]) -> str:
    """Format an int array with reasonable line wrapping."""
    # For small arrays, single line; for larger, multi-line with indentation
    if len(values) <= 12:
        inner = ", ".join(str(v) for v in values)
        return f"{decl} = {{{inner}}};\n"
    else:
        lines = [f"{decl} = {{"]
        # Chunk into lines of ~12 values
        chunk_size = 12
        for i in range(0, len(values), chunk_size):
            chunk = values[i:i + chunk_size]
            inner = ", ".join(str(v) for v in chunk)
            if i + chunk_size < len(values):
                inner += ","
            lines.append(f"    {inner}")
        lines.append("};")
        return "\n".join(lines) + "\n"


# ────────────────────────────────────────────────────────────────────────────
# Log-uniform sampling within a band
# ────────────────────────────────────────────────────────────────────────────

def _draw_log_uniform_nk(n_lo: float, n_hi: float,
                         exclude: set[tuple[int, int]]) -> tuple[int, int]:
    """
    Draw a random (n, k) point within [n_lo, n_hi) for n,
    and k log-uniform in [2, n].  Exclude points already in the
    calibration set or already probed.
    Returns integer (n, k).
    """
    max_attempts = 200
    for _ in range(max_attempts):
        log_n = random.uniform(math.log(n_lo), math.log(n_hi))
        n = int(round(math.exp(log_n)))
        n = max(2, n)  # n >= 2

        log_k = random.uniform(math.log(2), math.log(n))
        k = int(round(math.exp(log_k)))
        k = max(2, min(k, n))

        if (n, k) not in exclude:
            return n, k

    # Fallback: just pick the band's skeleton n with a random k
    n = int(round(math.exp((math.log(n_lo) + math.log(n_hi)) / 2)))
    n = max(2, n)
    k = max(2, n // 2)
    while (n, k) in exclude and k < n:
        k += 1
    return n, k


# ────────────────────────────────────────────────────────────────────────────
# Main adaptive loop
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive calibration orchestrator — one command per device."
    )
    parser.add_argument("--device", required=True,
                        choices=["m3_pro", "zen4", "b200"])
    parser.add_argument("--clean-streak-target", type=int, default=25)
    parser.add_argument("--max-probes-per-band", type=int, default=150)
    parser.add_argument("--gap-threshold", type=float, default=0.02,
                        help="Fraction (e.g. 0.02 = 2%%); compared against "
                             "gap_pct (percent) as gap_pct > threshold*100.")
    parser.add_argument("--calibrate-bin", type=str, default=None,
                        help="Path to calibrate_best_b / calibrate_gpu_best_b binary.")
    parser.add_argument("--validate-bin", type=str, default=None,
                        help="Path to validate_best_b / validate_planner_gpu binary.")
    parser.add_argument("--skeleton-lo", type=int, default=None)
    parser.add_argument("--skeleton-hi", type=int, default=None)
    parser.add_argument("--skeleton-ratio", type=float, default=None)
    parser.add_argument("--config-header", type=str, default=None,
                        help="Override path to config header (for testing).")
    args = parser.parse_args()

    device = args.device
    meta = dict(DEVICE_META[device])  # shallow copy so we can override
    if args.config_header:
        meta["config_header"] = args.config_header
    is_gpu = meta["is_gpu"]

    # Default binary paths if not specified
    calibrate_bin = args.calibrate_bin
    validate_bin = args.validate_bin
    if calibrate_bin is None:
        if is_gpu:
            calibrate_bin = "./build/calibrate_gpu_best_b"
        else:
            calibrate_bin = "./build/calibrate_best_b"
    if validate_bin is None:
        if is_gpu:
            validate_bin = "./build/validate_planner_gpu"
        else:
            validate_bin = "./build/validate_best_b"

    # Verify binaries exist
    for binpath, name in [(calibrate_bin, "calibrate"), (validate_bin, "validate")]:
        if not os.path.isfile(binpath):
            print(f"WARNING: {name} binary not found at '{binpath}'. "
                  f"Will attempt to run anyway (may fail if not on PATH).",
                  file=sys.stderr)

    clean_streak_target = args.clean_streak_target
    max_probes_per_band = args.max_probes_per_band
    gap_threshold = args.gap_threshold  # fraction, e.g. 0.02

    print(f"=== Adaptive Calibration Orchestrator ===")
    print(f"  Device:             {device}")
    print(f"  GPU:                {is_gpu}")
    print(f"  Config header:      {meta['config_header']}")
    print(f"  Calibrate binary:   {calibrate_bin}")
    print(f"  Validate binary:    {validate_bin}")
    print(f"  Clean-streak target:{clean_streak_target}")
    print(f"  Max probes/band:    {max_probes_per_band}")
    print(f"  Gap threshold:      {gap_threshold} ({gap_threshold*100:.1f}%)")
    print()

    # ── Step 1: Generate skeleton + bands ──────────────────────────────
    print("── Step 1: Generate skeleton ──")
    skeleton_points, bands = run_skeleton_generator(
        device, args.skeleton_lo, args.skeleton_hi, args.skeleton_ratio)
    print(f"  Skeleton: {len(skeleton_points)} points across {len(bands)} bands")
    print()

    # ── Step 2: Base-table sweep ───────────────────────────────────────
    print("── Step 2: Base-table sweep (calibrate binary on full skeleton) ──")
    skel_fd, skel_csv_path = tempfile.mkstemp(suffix=".csv", prefix="skel_")
    os.close(skel_fd)
    with open(skel_csv_path, "w") as f:
        f.write("n,k\n")
        for n, k in skeleton_points:
            f.write(f"{n},{k}\n")

    base_fd, base_csv_path = tempfile.mkstemp(suffix=".csv", prefix="base_")
    os.close(base_fd)

    base_table = run_calibrate_sweep(calibrate_bin, skel_csv_path,
                                     base_csv_path, is_gpu)
    print(f"  Base table: {len(base_table)} points")
    os.unlink(skel_csv_path)
    os.unlink(base_csv_path)

    # Build the live table as a dict (n,k) -> best_B for fast lookup
    live_table: dict[tuple[int, int], int] = {}
    for n, k, b in base_table:
        live_table[(n, k)] = b

    # ── Step 3: Inject base table ──────────────────────────────────────
    print("── Step 3: Inject base table into config header ──")
    inject_table(meta["config_header"], meta,
                 [(n, k, live_table[(n, k)]) for n, k, _ in base_table])
    print()

    # ── Step 4: Per-band adaptive loop ─────────────────────────────────
    print("── Step 4: Per-band adaptive refinement ──")
    print()

    # Track all probed points this run (excluding skeleton base points)
    probed_this_run: set[tuple[int, int]] = set()
    # Calibration set = all points in live_table
    calib_set: set[tuple[int, int]] = set(live_table.keys())

    per_band_results: list[dict] = []

    for band in bands:
        band_id = band["band_id"]
        n_lo = band["n_lo"]
        n_hi = band["n_hi"]
        skeleton_n = band["skeleton_n"]

        clean_streak = 0
        probes_in_band = 0
        points_added = 0
        hit_safety_cap = False

        print(f"  Band {band_id}: n ∈ [{n_lo:.1f}, {n_hi:.1f})  "
              f"skeleton_n={skeleton_n}")

        while clean_streak < clean_streak_target and probes_in_band < max_probes_per_band:
            # Draw a random (n,k) within this band, excluding already-known points
            exclude = calib_set | probed_this_run
            n, k = _draw_log_uniform_nk(n_lo, n_hi, exclude)
            probed_this_run.add((n, k))
            probes_in_band += 1

            # Probe with validate binary
            probe = run_validate_probe(validate_bin, n, k, is_gpu)
            gap_pct = probe["gap_pct"]

            # Compare gap_pct (percent) against gap_threshold*100
            if gap_pct > gap_threshold * 100.0:
                # Gap exceeds threshold — refine this point
                auto_B = probe["auto_B"]
                print(f"    [{probes_in_band}] n={n} k={k} gap={gap_pct:.2f}% "
                      f"> {gap_threshold*100:.1f}% → refining near auto_B={auto_B}")

                # Call calibrate binary with --narrow-around for this point
                narrow_fd, narrow_in_path = tempfile.mkstemp(
                    suffix=".csv", prefix="narrow_")
                os.close(narrow_fd)
                with open(narrow_in_path, "w") as f:
                    f.write("n,k\n")
                    f.write(f"{n},{k}\n")

                narrow_out_fd, narrow_out_path = tempfile.mkstemp(
                    suffix=".csv", prefix="narrow_out_")
                os.close(narrow_out_fd)

                new_rows = run_calibrate_sweep(
                    calibrate_bin, narrow_in_path, narrow_out_path,
                    is_gpu, narrow_around=[auto_B])

                os.unlink(narrow_in_path)
                os.unlink(narrow_out_path)

                if new_rows:
                    new_b = new_rows[0][2]
                    live_table[(n, k)] = new_b
                    calib_set.add((n, k))
                    points_added += 1

                    # IMMEDIATE re-injection
                    inject_table(meta["config_header"], meta,
                                 [(pn, pk, live_table[(pn, pk)])
                                  for pn, pk in calib_set])
                    print(f"      → best_B={new_b}, injected immediately")
                else:
                    print(f"      → calibrate returned no result, skipping")

                clean_streak = 0
            else:
                # Gap within threshold
                clean_streak += 1
                if probes_in_band <= 5 or probes_in_band % 20 == 0:
                    print(f"    [{probes_in_band}] n={n} k={k} "
                          f"gap={gap_pct:.2f}% ✓ (clean_streak={clean_streak})")

        # Band finished
        if probes_in_band >= max_probes_per_band and clean_streak < clean_streak_target:
            hit_safety_cap = True
            status = f"HIT SAFETY CAP ({probes_in_band} probes, "
            status += f"clean_streak={clean_streak}/{clean_streak_target})"
        else:
            status = f"converged ({probes_in_band} probes, "
            status += f"{points_added} points added)"

        print(f"  Band {band_id} {status}")
        if hit_safety_cap:
            print(f"    ⚠  This band did not converge — region may need human attention.")
        print()

        per_band_results.append({
            "band_id": band_id,
            "skeleton_n": skeleton_n,
            "n_lo": n_lo,
            "n_hi": n_hi,
            "probes": probes_in_band,
            "points_added": points_added,
            "hit_safety_cap": hit_safety_cap,
            "final_clean_streak": clean_streak,
        })

    # ── Step 5: Final re-injection ─────────────────────────────────────
    print("── Step 5: Final re-injection ──")
    inject_table(meta["config_header"], meta,
                 [(n, k, live_table[(n, k)]) for n, k in calib_set])
    print()

    # ── Step 6: Summary ────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Device:             {device}")
    print(f"  Base skeleton:      {len(base_table)} points")
    total_probes = sum(r["probes"] for r in per_band_results)
    total_added = sum(r["points_added"] for r in per_band_results)
    print(f"  Total probes:       {total_probes}")
    print(f"  Total points added: {total_added}")
    print(f"  Final table size:   {len(calib_set)}")
    print()

    capped_bands = [r for r in per_band_results if r["hit_safety_cap"]]
    if capped_bands:
        print(f"  ⚠  {len(capped_bands)} band(s) hit the safety cap without converging:")
        for r in capped_bands:
            print(f"      Band {r['band_id']} (n≈{r['skeleton_n']}, "
                  f"n∈[{r['n_lo']:.0f},{r['n_hi']:.0f})): "
                  f"{r['probes']} probes, clean_streak={r['final_clean_streak']}")
    else:
        print("  All bands converged within their clean-streak targets.")

    print()
    print("Per-band details:")
    for r in per_band_results:
        cap = " ⚠ SAFETY CAP" if r["hit_safety_cap"] else ""
        print(f"  Band {r['band_id']:3d}  n≈{r['skeleton_n']:7d}  "
              f"[{r['n_lo']:9.1f}, {r['n_hi']:9.1f})  "
              f"probes={r['probes']:3d}  added={r['points_added']:3d}"
              f"{cap}")

    print()
    print("Done.  Config header updated.  Rebuild and run verify to confirm.")


if __name__ == "__main__":
    main()
