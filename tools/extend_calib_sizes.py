#!/usr/bin/env python3
"""
extend_calib_sizes.py, Safely extend a device's calibrated FFT-size ceiling.

Problem this replaces: CLAUDE.md documents re-running
`./calibrate --max-size N` on an already-calibrated device and copying the
resulting fft_config.h straight into devices/<device>/. That is wrong for
any device that has been calibrated beyond the base calibrate.c pass --
which is every shipped device -- because calibrate.c's own output only
contains calib_sizes[]/calib_times_ns[]/CALIBRATED_MAX_CONV_LEN. A raw `cp`
silently discards the crossover table, the B-selection table, WRAP_FMA_NS,
and the schoolbook cost tables that were layered in afterward by separate
tools (calibrate_crossover.c, calibrate_best_b.c, bench_wrap_fma.c,
bench_schoolbook_tree.c). Found 2026-08-03 doing exactly this by hand for
the M3 Pro 262144 extension: a raw copy left schoolbook_mul_ns[]/
schoolbook_corr_ns[] re-declared at the new (larger) N_CALIBRATED_SIZES but
still only initialized with the old number of values. C zero-fills the
remaining slots, and 0.0 is not the ">=0 means measured" sentinel this
table already uses for "never measured, always prefer FFT" (that sentinel
is -1.0, per tools/bench_schoolbook_tree.c and the check in icm.c) -- so
every new large block size would read a fabricated schoolbook cost of
0.0ns and the dispatcher would force schoolbook (O(len^2)) there instead
of FFT. bench_grid verify would not have caught it: verify only exercises
n up to 65536, well short of the newly-extended range.

What this script actually does: take the raw calibrate.c output (just the
four things it produces) and merge it into the device's *existing* header,
preserving every other table byte-for-byte, and safely extending the two
other tables that happen to share the same N_CALIBRATED_SIZES index space
(schoolbook_mul_ns[]/schoolbook_corr_ns[]) by padding new entries with
their own already-established -1.0 "unmeasured" sentinel -- not by
guessing, and not silently: an array sized [N_CALIBRATED_SIZES] that isn't
on the known-safe extension allow-list below aborts the run.

Usage:
  ./calibrate --max-size 262144        # produces ./fft_config.h, ./fftw_wisdom.dat
  python3 tools/extend_calib_sizes.py --device m3_pro fft_config.h
  cp fftw_wisdom.dat devices/m3_pro/fftw_wisdom.dat   # wisdom has no merge issue, replace wholesale
  make clean && make DEVICE=m3_pro && ./bench_grid verify && ./bench_grid crossover
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_common import (DEVICE_META, find_array_block, find_matching_brace,  # noqa: E402
                          format_number_array, read_number_array)

# Arrays sized exactly [N_CALIBRATED_SIZES] that are safe to extend by
# padding, and the sentinel to pad with. Anything sized [N_CALIBRATED_SIZES]
# found in a device header that is NOT in this dict aborts the run --
# guessing a sentinel for an unknown table is exactly the class of mistake
# this script exists to prevent.
KNOWN_EXTENDABLE_TABLES = {
    "schoolbook_mul_ns": -1.0,
    "schoolbook_corr_ns": -1.0,
}


def read_header_calib_block(path: str):
    with open(path) as f:
        text = f.read()
    m = re.search(r'#define\s+N_CALIBRATED_SIZES\s+(\d+)', text)
    if not m:
        raise RuntimeError(f"N_CALIBRATED_SIZES not found in {path}")
    n = int(m.group(1))
    sizes = read_number_array(
        text, r'static const int calib_sizes\[N_CALIBRATED_SIZES\]\s*=\s*\{', int)
    times = read_number_array(
        text, r'static const double calib_times_ns\[N_CALIBRATED_SIZES\]\s*=\s*\{', float)
    if len(sizes) != n or len(times) != n:
        raise RuntimeError(
            f"{path}: N_CALIBRATED_SIZES={n} but calib_sizes has {len(sizes)} "
            f"and calib_times_ns has {len(times)} entries")
    return text, n, sizes, times


def find_all_sized_arrays(text: str) -> list[str]:
    """Every `static const TYPE name[N_CALIBRATED_SIZES]` array name."""
    return re.findall(
        r'static const \w+ (\w+)\[N_CALIBRATED_SIZES\]\s*=\s*\{', text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, choices=list(DEVICE_META))
    ap.add_argument("new_config",
                    help="Freshly generated fft_config.h from "
                         "`./calibrate --max-size N` (run from repo root).")
    ap.add_argument("--out", default=None,
                    help="Output path (default: overwrite the device's own "
                         "config header in place).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    existing_path = DEVICE_META[args.device]["config_header"]
    out_path = args.out or existing_path

    new_text, new_n, new_sizes, new_times = read_header_calib_block(args.new_config)
    old_text, old_n, old_sizes, old_times = read_header_calib_block(existing_path)

    if new_n < old_n:
        raise RuntimeError(
            f"New calibration ({new_n} sizes) is SMALLER than the existing "
            f"one ({old_n}). This script only extends, it does not shrink; "
            f"pass a --max-size at least as large as the existing ceiling.")

    if new_sizes[:old_n] != old_sizes:
        raise RuntimeError(
            "New calib_sizes[] is not a superset of the existing table "
            "(the first old_n entries don't match exactly). The 7-smooth "
            "enumeration is expected to produce the same prefix at any "
            "--max-size >= the old ceiling; a mismatch means something "
            "about size generation changed and the padding assumption "
            "below (existing per-size tables map 1:1 onto the new table's "
            "prefix) no longer holds. Refusing to guess.")

    m = re.search(r'#define\s+CALIBRATED_MAX_CONV_LEN\s+(-?\d+)', new_text)
    if not m:
        raise RuntimeError(f"CALIBRATED_MAX_CONV_LEN not found in {args.new_config}")
    new_ceiling = int(m.group(1))

    print(f"[extend] {args.device}: {old_n} -> {new_n} calibrated sizes "
          f"(ceiling {new_ceiling})")

    # ── Replace CALIBRATED_MAX_CONV_LEN, N_CALIBRATED_SIZES, calib_sizes[], calib_times_ns[] wholesale ──
    text = old_text
    text = re.sub(r'#define\s+CALIBRATED_MAX_CONV_LEN\s+-?\d+',
                  f'#define CALIBRATED_MAX_CONV_LEN {new_ceiling}', text)
    text = re.sub(r'#define\s+N_CALIBRATED_SIZES\s+\d+',
                  f'#define N_CALIBRATED_SIZES {new_n}', text)
    # The hand-written prose comment above CALIBRATED_MAX_CONV_LEN cites the
    # old max(calib_sizes)/ceiling numbers directly; the #define substitution
    # above doesn't touch prose. Pull the equivalent comment straight out of
    # calibrate.c's own fresh output instead of re-deriving the wording here.
    new_comment_match = re.search(
        r'/\* Largest convolution length.*?non-sentinel cost for L.*?\*/', new_text, re.S)
    old_comment_match = re.search(
        r'/\* Largest convolution length.*?non-sentinel cost for L.*?\*/', text, re.S)
    if new_comment_match and old_comment_match:
        text = text[:old_comment_match.start()] + new_comment_match.group(0) + \
            text[old_comment_match.end():]

    ds, bs, be = find_array_block(
        text, r'static const int calib_sizes\[N_CALIBRATED_SIZES\]\s*=\s*\{')
    text = text[:ds] + format_number_array(
        "static const int calib_sizes[N_CALIBRATED_SIZES]", new_sizes)[:-1] + text[be + 1:]

    ds, bs, be = find_array_block(
        text, r'static const double calib_times_ns\[N_CALIBRATED_SIZES\]\s*=\s*\{')
    text = text[:ds] + format_number_array(
        "static const double calib_times_ns[N_CALIBRATED_SIZES]", new_times)[:-1] + text[be + 1:]

    # ── Extend every other [N_CALIBRATED_SIZES]-sized array by padding ──
    other_arrays = [a for a in find_all_sized_arrays(old_text)
                    if a not in ("calib_sizes", "calib_times_ns")]
    for name in other_arrays:
        if name not in KNOWN_EXTENDABLE_TABLES:
            raise RuntimeError(
                f"{existing_path} has an array `{name}[N_CALIBRATED_SIZES]` "
                f"this script doesn't know how to extend safely. Add it to "
                f"KNOWN_EXTENDABLE_TABLES with the correct sentinel value "
                f"(check the array's own provenance tool / usage in icm.c "
                f"first -- do not guess), or extend it manually and re-run "
                f"with a version of this file that already reflects it.")
        sentinel = KNOWN_EXTENDABLE_TABLES[name]
        m2 = re.search(rf'static const double {name}\[N_CALIBRATED_SIZES\]\s*=\s*\{{', text)
        if not m2:
            raise RuntimeError(f"{name}[] not found during extension pass")
        old_vals = read_number_array(
            text, rf'static const double {name}\[N_CALIBRATED_SIZES\]\s*=\s*\{{', float)
        if len(old_vals) != old_n:
            raise RuntimeError(
                f"{name}[] has {len(old_vals)} entries, expected {old_n} "
                f"(matching the pre-extension N_CALIBRATED_SIZES)")
        new_vals = old_vals + [sentinel] * (new_n - old_n)
        ds, bs, be = find_array_block(
            text, rf'static const double {name}\[N_CALIBRATED_SIZES\]\s*=\s*\{{')
        text = text[:ds] + format_number_array(
            f"static const double {name}[N_CALIBRATED_SIZES]", new_vals)[:-1] + text[be + 1:]
        print(f"[extend] {name}[]: {old_n} real values kept, "
              f"{new_n - old_n} padded with sentinel {sentinel}")

    # ── Self-verify before writing anything ──
    for arr in ["calib_sizes", "calib_times_ns"] + other_arrays:
        cast = int if arr == "calib_sizes" else float
        pattern = (rf'static const int {arr}\[N_CALIBRATED_SIZES\]\s*=\s*\{{'
                  if arr == "calib_sizes" else
                  rf'static const double {arr}\[N_CALIBRATED_SIZES\]\s*=\s*\{{')
        vals = read_number_array(text, pattern, cast)
        assert len(vals) == new_n, (
            f"self-check failed: {arr}[] has {len(vals)} entries after "
            f"merge, expected {new_n}")

    # Everything outside the touched arrays/macros must be untouched. Cheap
    # proxy check: line count of the "rest of file" section (after the last
    # touched array) must match exactly, since format_number_array's line
    # wrapping only affects the arrays it's given.
    unrelated_pattern = r'/\* ── Empirical linear-vs-hybrid crossover'
    old_rest = old_text[old_text.index(unrelated_pattern.replace(r'\* ── ', '* ── ').replace('\\', '')):] \
        if unrelated_pattern.replace(r'\* ── ', '').replace('\\', '') in old_text else None
    if old_rest is not None:
        new_rest_idx = text.find(old_rest[:80])
        if new_rest_idx == -1 or text[new_rest_idx:] != old_rest:
            raise RuntimeError(
                "self-check failed: content after the size-indexed tables "
                "changed. Refusing to write -- this should be byte-for-byte "
                "identical to the input device header.")

    print(f"[extend] self-checks passed")

    if args.dry_run:
        print("[extend] --dry-run: not writing")
        return

    with open(out_path, "w") as f:
        f.write(text)
    print(f"[extend] Wrote {out_path}")
    print("[extend] Next: rebuild and run `./bench_grid verify` + "
          "`./bench_grid crossover`, and build libicm.a/libicm.{dylib,so} "
          "(bench_grid alone is not sufficient, see CLAUDE.md).")


if __name__ == "__main__":
    main()
