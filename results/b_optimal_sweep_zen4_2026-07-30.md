# B-Optimality Sweep Report

**Device:** zen4
**Date:** 2026-07-30 19:34:50 UTC
**Hostname:** calm-jaybird
**Git commit:** unknown
**Mode:** full grid

## Summary

| Metric | Count |
|---|---|
| Total cells | 65 |
| auto_B == best_B (exact) | 23 |
| Within 3% gap (effectively correct) | 28 |
| Cells measured this run | 65 |
| Cells skipped (already in CSV) | 0 |
| Cells failed | 0 |

## Worst Offenders (by gap_pct)

| n | k | auto_B | best_B | gap_pct |
|---|---|---|---|---|
| 65536 | 10 | 8 | 32 | 44.3880% |
| 64 | 10 | 8 | 32 | 44.1754% |
| 256 | 10 | 8 | 32 | 41.5846% |
| 8192 | 10 | 8 | 32 | 38.5924% |
| 128 | 10 | 8 | 32 | 38.5103% |
| 512 | 10 | 8 | 32 | 38.3965% |
| 16384 | 10 | 8 | 32 | 38.2858% |
| 4096 | 10 | 8 | 32 | 37.9448% |
| 2048 | 10 | 8 | 32 | 37.5123% |
| 1024 | 10 | 8 | 32 | 37.0482% |


## Raw Data

`/root/ICM/results/b_optimal_sweep_zen4_2026-07-30.csv`

## Methodology

Probe: `tools/validate_best_b.c` (single-point oracle).
Q=256, srand(42), payout[m]=n-m, S[i]=100+9900*rand()/RAND_MAX.
Median-of-7 for final head-to-head timing.
Discovery: 1 rep per candidate to rank, then 2 more reps on top-2 if within 3%, median of 3.

Grid: mirrors `bench/bench.c` performance grid.
