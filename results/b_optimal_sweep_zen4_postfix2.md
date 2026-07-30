# B-Optimality Sweep Report

**Device:** zen4
**Date:** 2026-07-30 20:53:08 UTC
**Hostname:** calm-jaybird
**Git commit:** unknown
**Mode:** full grid

## Summary

| Metric | Count |
|---|---|
| Total cells | 65 |
| auto_B == best_B (exact) | 31 |
| Within 3% gap (effectively correct) | 38 |
| Cells measured this run | 65 |
| Cells skipped (already in CSV) | 0 |
| Cells failed | 0 |

## Worst Offenders (by gap_pct)

| n | k | auto_B | best_B | gap_pct |
|---|---|---|---|---|
| 8192 | 8192 | 24 | 32 | 21.7930% |
| 16384 | 16384 | 24 | 32 | 18.1682% |
| 16384 | 8192 | 24 | 32 | 17.6560% |
| 1024 | 50 | 24 | 64 | 16.8624% |
| 2048 | 2048 | 24 | 32 | 14.6938% |
| 1024 | 1024 | 24 | 32 | 14.0592% |
| 4096 | 4096 | 24 | 32 | 13.8922% |
| 64 | 64 | 16 | 32 | 13.6939% |
| 32768 | 32768 | 24 | 32 | 13.3246% |
| 8192 | 10 | 16 | 32 | 11.7923% |


## Raw Data

`results/b_optimal_sweep_zen4_postfix2.csv`

## Methodology

Probe: `tools/validate_best_b.c` (single-point oracle).
Q=256, srand(42), payout[m]=n-m, S[i]=100+9900*rand()/RAND_MAX.
Median-of-7 for final head-to-head timing.
Discovery: 1 rep per candidate to rank, then 2 more reps on top-2 if within 3%, median of 3.

Grid: mirrors `bench/bench.c` performance grid.
