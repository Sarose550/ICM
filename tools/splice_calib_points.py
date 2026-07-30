#!/usr/bin/env python3
"""
splice_calib_points.py, Merge a targeted calibrate_best_b/calibrate_gpu_best_b
run into a device's live config header.

This is the canonical, reproducible second half of V6's targeted-anchor
methodology (VERDICTS.md): measure specific (n,k) points directly with
--narrow-around (cheap, avoids the full 48-candidate sweep that VERDICTS.md
V6 measured at ~$28/4.5h for the B200's large-n region), then land the
result. Reuses calibrate_block_size.py's inject_table()/read_existing_table()
so there is exactly one implementation of "safely merge a new (n,k,B) point
into the live table" in the repo, not one written by hand at each call site.
Hand-editing devices/<device>/*fft_config.h directly is what this replaces;
it is easy to miscount array lengths or drop the N_POINTS macro update by
hand, and read_existing_table()'s merge (not overwrite) semantics matter,
see its own docstring.

Usage:
  # 1. Measure targeted points directly.
  ./build/calibrate_gpu_best_b skeleton.csv new_points.csv \
      --narrow-around 96,112,128,144
  # (CPU: ./build/calibrate_best_b skeleton.csv -o new_points.csv --narrow-around ...)

  # 2. Merge the result into the live config header.
  python3 tools/splice_calib_points.py --device b200 new_points.csv

Points already in the header are preserved; an exact (n,k) match in the new
CSV overwrites its old B, everything else (including prior gap-fill anchors
outside any skeleton) is additive. Rebuild and run verify + the device's
bselect regression test after splicing, this script does not do that for
you.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_block_size import DEVICE_META, inject_table, read_existing_table  # noqa: E402


def read_points_csv(path: str) -> list[tuple[int, int, int]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge calibrate_best_b/calibrate_gpu_best_b output into "
                     "a device's live config header."
    )
    parser.add_argument("--device", required=True, choices=list(DEVICE_META))
    parser.add_argument("--config-header", default=None,
                         help="Override path to config header (default: the "
                              "device's own, for testing).")
    parser.add_argument("points_csv",
                         help="Output CSV from calibrate_best_b / "
                              "calibrate_gpu_best_b (n,k,best_B rows, "
                              "'#'-prefixed comments and a header row are "
                              "both fine).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the merge summary without writing the "
                              "header.")
    args = parser.parse_args()

    meta = dict(DEVICE_META[args.device])
    if args.config_header:
        meta["config_header"] = args.config_header

    new_points = read_points_csv(args.points_csv)
    if not new_points:
        print(f"No points read from {args.points_csv}", file=sys.stderr)
        sys.exit(1)

    existing = read_existing_table(meta["config_header"], meta)
    live: dict[tuple[int, int], int] = {(n, k): b for n, k, b in existing}
    overwritten = sum(1 for n, k, _ in new_points if (n, k) in live)
    added = len(new_points) - overwritten

    for n, k, b in new_points:
        live[(n, k)] = b

    print(f"[splice] {len(existing)} existing points in {meta['config_header']}")
    print(f"[splice] {len(new_points)} points in {args.points_csv} "
          f"({overwritten} overwrite an existing (n,k), {added} new)")
    print(f"[splice] {len(live)} total points after merge")

    if args.dry_run:
        print("[splice] --dry-run: not writing")
        return

    inject_table(meta["config_header"], meta,
                 [(n, k, b) for (n, k), b in live.items()])
    print("[splice] Done. Rebuild and run verify + the device's bselect "
          "regression test to confirm.")


if __name__ == "__main__":
    main()
