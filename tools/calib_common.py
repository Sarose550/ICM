#!/usr/bin/env python3
"""
calib_common.py, Shared utilities for calibration tooling.

Imported by calibrate_adaptive.py and splice_calib_points.py so there is
exactly one implementation of the core table-I/O and probe functions in
the repo.
"""

import math
import os
import random
import re
import subprocess
import sys
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
    # Find the matching closing brace, we need to find from the opening brace
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


def read_existing_table(config_path: str, device_meta: dict) -> list[tuple[int, int, int]]:
    """
    Parse the current {prefix}_n[]/{prefix}_k[]/{prefix}_B[] arrays out of
    an already-calibrated config header. Returns [] if the header doesn't
    exist yet or the arrays aren't found (fresh device, nothing to merge).

    This exists so a re-run of this orchestrator MERGES with whatever is
    already calibrated instead of silently discarding it: the landmark
    seeding and adaptive refinement only cover a sparse/subset of the
    domain, which does NOT include hand-added gap-fill anchors at
    non-7-smooth n values (e.g. points added to close a specific
    B-selection cliff) — those points would vanish on the next full
    orchestrator run if this function's result weren't seeded into
    live_table before any injection.
    """
    prefix = device_meta["array_prefix"]
    n_macro = device_meta["n_macro"]

    if not os.path.isfile(config_path):
        return []

    with open(config_path, "r") as f:
        text = f.read()

    def _extract(array_name: str) -> list[int]:
        pattern = rf'static const int {array_name}\[{n_macro}\]\s*=\s*\{{'
        m = re.search(pattern, text)
        if not m:
            return []
        brace_start = m.end() - 1
        brace_end = _find_matching_brace(text, brace_start)
        body = text[brace_start + 1:brace_end]
        return [int(tok) for tok in re.findall(r'-?\d+', body)]

    n_vals = _extract(f"{prefix}_n")
    k_vals = _extract(f"{prefix}_k")
    b_vals = _extract(f"{prefix}_B")

    if not (len(n_vals) == len(k_vals) == len(b_vals)) or not n_vals:
        return []

    return list(zip(n_vals, k_vals, b_vals))


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
        # n, k from output (columns 0,1), we already know them
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
# Log-uniform sampling
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

    # Fallback: just pick the midpoint n with a random k
    n = int(round(math.exp((math.log(n_lo) + math.log(n_hi)) / 2)))
    n = max(2, n)
    k = max(2, n // 2)
    while (n, k) in exclude and k < n:
        k += 1
    return n, k


# ────────────────────────────────────────────────────────────────────────────
# CSV point reader (shared between splice_calib_points and calibrate_adaptive)
# ────────────────────────────────────────────────────────────────────────────

def read_points_csv(path: str) -> list[tuple[int, int, int]]:
    """Read (n, k, best_B) triples from a CSV file."""
    points: list[tuple[int, int, int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("n,k"):
                continue
            parts = line.split(",")
            if len(parts) != 3:
                continue
            try:
                n, k, b = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            points.append((n, k, b))
    return points
