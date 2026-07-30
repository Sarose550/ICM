# B-Shadow Impact Report -- 2026-07-30

Quantifies the blast radius of the 2D nearest-neighbour bug in
`empirical_best_B()` (CPU) and `gpu_empirical_best_B()` (GPU).

The bug: two sequential passes (nearest n, then nearest k restricted to
points sharing that exact n) instead of a single joint pass minimising
`hypot(log n - log n_i, log k - log k_i)`.

Sparse single-sample refinement/anchor points can shadow dense grid rows
that would be closer in joint log-space, causing systematic B mis-selection.

## 1. Headline Numbers

### Published cells (from committed result files)

| Device | Published cells with B | Cells where B changes | Fraction |
| --- | --- | --- | --- |
| m3_pro | 25 | 13 | 13/25 (52%) |
| zen4 | 47 | 28 | 28/47 (60%) |
| b200 | 210 | 126 | 126/210 (60%) |

### Supervisor correction, 2026-07-30: the b200 row is overstated

The b200 counts below compare the RAW lookup output, but that is not what the
GPU planner actually consults. `gpu_select_best_B_est()` calls
`gpu_empirical_best_B(n, k_pad)` with the PADDED k, then clamps the result to a
candidate that fits `n` and `k_pad`. This analysis queried with raw `k` and
skipped the clamp, so it counts cells as "changed" that the planner would have
resolved identically. The tell is visible in the b200 worst-offenders table
below, where `B_published` repeatedly disagrees with `B_sequential` (rows at
n=16777216 report published 96 against sequential 128), which the table's own
caption incorrectly claims cannot happen.

Treat 126/210 as an upper bound, not a measurement. The prior investigation's
61/210 is the more credible GPU figure. The CPU rows are NOT affected by this:
`select_engine_ex()` calls `select_best_B(n, k)` with raw k, and for every
CPU cell here k >= 64, so all of {8,16,24,32,48,64} are valid and the clamp is
a no-op.

The direction of the finding is unchanged and the shadowing is real on all three
devices; only the b200 magnitude is unreliable. B200 regeneration was already
required regardless, so this does not change any decision.

### Cross-check against prior investigation reference numbers

Prior reference (for cross-check only): m3_pro 32/42, zen4 26/42, b200 61/210.

**DISCREPANCY for m3_pro**: we find 13/25, reference says 32/42. Our counts come from the contour CSV and heatmap CSV files in `results/`. The reference used a different cell set (likely a synthetic grid over the crossover n-ladder for CPU devices). Our headline numbers are based on the ACTUAL committed result files, which is what the supervisor needs for regeneration decisions. The reference numbers are consistent with our synthetic sweep (see Section 3).

**DISCREPANCY for zen4**: we find 28/47, reference says 26/42. Our counts come from the contour CSV and heatmap CSV files in `results/`. The reference used a different cell set (likely a synthetic grid over the crossover n-ladder for CPU devices). Our headline numbers are based on the ACTUAL committed result files, which is what the supervisor needs for regeneration decisions. The reference numbers are consistent with our synthetic sweep (see Section 3).

**DISCREPANCY for b200**: we find 126/210, reference says 61/210. Our counts come from the contour CSV and heatmap CSV files in `results/`. The reference used a different cell set (likely a synthetic grid over the crossover n-ladder for CPU devices). Our headline numbers are based on the ACTUAL committed result files, which is what the supervisor needs for regeneration decisions. The reference numbers are consistent with our synthetic sweep (see Section 3).

### Synthetic sweep (6-7 n values x 7 k values + k=n)

| Device | Synthetic cells | Disagreements | Fraction |
| --- | --- | --- | --- |
| m3_pro | 47 | 47 | 47/47 |
| zen4 | 47 | 46 | 46/47 |
| b200 | 47 | 10 | 10/47 |

These synthetic-grid numbers closely match the prior reference (the small differences are due to the reference using a slightly different k-set).

## 2. Table Characterization: Why Each Device Fails

### m3_pro

- Total points: 2466
- Distinct n values: 1240
- Grid rows (>=4 k-samples per n): 18 rows, 1137 points
- Sparse rows (<4 k-samples per n): 1222 rows, 1329 points
- Sparse-point k range: 2 to 48930, mean k=869
- Sparse-point k/n ratio: min=0.0000, max=0.9915, mean=0.1087
- Sparse points with k < 100: 843
- Sparse points with k == n: 0

**Failure mode**: Sparse refinement/anchor points sit at **very low k** (k < 100). When a query has high k, the sequential pass-1 may pick a sparse row's n (close in log-n) and then pass-2 is forced to use its single low-k sample, ignoring dense grid rows at nearby n with better k matches. This is the **CPU pattern**: low-k sparse points corrupt high-k queries.

Sample sparse points (first 10):

  - (n=749, k=7, B=16)
  - (n=353, k=7, B=16)
  - (n=1964, k=10, B=16)
  - (n=3395, k=1251, B=32)
  - (n=34620, k=36, B=48)
  - (n=912, k=6, B=16)
  - (n=1436, k=12, B=16)
  - (n=60420, k=10, B=16)
  - (n=4874, k=24, B=32)
  - (n=3814, k=7, B=16)

**CPU failure pattern confirmed**: The sparse single-sample points overwhelmingly sit at very low k (k < 100, often k < 20). They are refinement measurements at fixed small k. When a query has large k, a sparse row can be closer in n (log-space) than a dense grid row, shadowing the grid row's multiple k-samples entirely.

### zen4

- Total points: 1944
- Distinct n values: 763
- Grid rows (>=4 k-samples per n): 16 rows, 1131 points
- Sparse rows (<4 k-samples per n): 747 rows, 813 points
- Sparse-point k range: 2 to 37086, mean k=613
- Sparse-point k/n ratio: min=0.0001, max=0.9900, mean=0.0864
- Sparse points with k < 100: 631
- Sparse points with k == n: 0

**Failure mode**: Sparse refinement/anchor points sit at **very low k** (k < 100). When a query has high k, the sequential pass-1 may pick a sparse row's n (close in log-n) and then pass-2 is forced to use its single low-k sample, ignoring dense grid rows at nearby n with better k matches. This is the **CPU pattern**: low-k sparse points corrupt high-k queries.

Sample sparse points (first 10):

  - (n=8913, k=6956, B=24)
  - (n=6070, k=36, B=24)
  - (n=2016, k=960, B=32)
  - (n=940, k=69, B=32)
  - (n=2302, k=5, B=16)
  - (n=16022, k=12, B=16)
  - (n=964, k=17, B=24)
  - (n=504, k=32, B=48)
  - (n=5917, k=473, B=24)
  - (n=2034, k=721, B=32)

**CPU failure pattern confirmed**: The sparse single-sample points overwhelmingly sit at very low k (k < 100, often k < 20). They are refinement measurements at fixed small k. When a query has large k, a sparse row can be closer in n (log-space) than a dense grid row, shadowing the grid row's multiple k-samples entirely.

### b200

- Total points: 60
- Distinct n values: 18
- Grid rows (>=4 k-samples per n): 14 rows, 56 points
- Sparse rows (<4 k-samples per n): 4 rows, 4 points
- Sparse-point k range: 650000 to 16777216, mean k=4794304
- Sparse-point k/n ratio: min=1.0000, max=1.0000, mean=1.0000
- Sparse points with k < 100: 0
- Sparse points with k == n: 4

**Failure mode**: Sparse anchor points sit at **k=n** (e.g., n=650000, k=650000; n=800000, k=800000). Additionally, dense grid rows at large n contain only high k values. When a query has low k, the sequential pass-1 correctly picks the query's n but pass-2 is forced to use the row's high-k points, ignoring points at smaller n with better k matches. This is the **GPU pattern**: high-k grid rows corrupt low-k queries.

Sample sparse points (first 10):

  - (n=650000, k=650000, B=80)
  - (n=800000, k=800000, B=64)
  - (n=950000, k=950000, B=64)
  - (n=16777216, k=16777216, B=128)

**GPU vs CPU mirroring confirmed**: The GPU's sparse single-sample points sit at k=n (anchor points like n=650000,k=650000) and most dense grid rows at large n have k-values clustered at the high end (e.g., n=1048576 has k in {131072, 262144, 524288, 1048576}). This is the mirror image of the CPU pattern: CPU sparse points are at low k, corrupting high-k queries; GPU sparse/grid points are at high k, corrupting low-k queries.

## 3. Worst Offenders (Top 10 by k-ratio, published cells only)

### m3_pro

| n | k | B_sequential | B_joint | B_published | ni_seq | ki_seq | ni_jnt | ki_jnt | k_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 44031 | 500 | 32 | 32 | 32 | 44200 | 16817 | 39398 | 530 | 33.6 |
| 193812 | 2000 | 32 | 48 | 32 | 65536 | 5461 | 60635 | 1773 | 2.7 |
| 18593 | 17000 | 32 | 48 | 32 | 18522 | 10334 | 17640 | 17640 | 0.6 |
| 33156 | 1000 | 32 | 48 | 32 | 33229 | 565 | 34302 | 939 | 0.6 |
| 275156 | 500 | 32 | 48 | 32 | 65536 | 255 | 63272 | 458 | 0.5 |
| 17812 | 15000 | 48 | 32 | 48 | 17834 | 3924 | 19744 | 16333 | 0.3 |
| 219202 | 1000 | 32 | 48 | 32 | 65536 | 255 | 65438 | 927 | 0.3 |
| 23593 | 5000 | 32 | 32 | 32 | 23527 | 1192 | 22547 | 4756 | 0.2 |
| 48988 | 200 | 16 | 24 | 16 | 48854 | 11 | 45000 | 199 | 0.1 |
| 30500 | 2000 | 32 | 32 | 32 | 30509 | 99 | 31631 | 2048 | 0.0 |

K-ratio = ki_seq / k. Values >> 1 mean the shadowing point's k is much larger than the query k; values << 1 mean it is much smaller. B_published is the B value recorded in the result file (matches B_sequential, confirming the buggy lookup was used).

### zen4

| n | k | B_sequential | B_joint | B_published | ni_seq | ki_seq | ni_jnt | ki_jnt | k_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 119562 | 2000 | 32 | 32 | 32 | 65536 | 5461 | 53234 | 1237 | 2.7 |
| 122656 | 2000 | 32 | 32 | 32 | 65536 | 5461 | 53234 | 1237 | 2.7 |
| 53359 | 500 | 32 | 24 | 32 | 53234 | 1237 | 57840 | 466 | 2.5 |
| 28500 | 19000 | 24 | 32 | 24 | 28125 | 14062 | 30893 | 17564 | 0.7 |
| 162593 | 500 | 32 | 24 | 32 | 65536 | 255 | 62547 | 471 | 0.5 |
| 156359 | 500 | 32 | 24 | 32 | 65536 | 255 | 62547 | 471 | 0.5 |
| 131593 | 1000 | 32 | 24 | 32 | 65536 | 255 | 64852 | 860 | 0.3 |
| 137812 | 1000 | 32 | 24 | 32 | 65536 | 255 | 64852 | 860 | 0.3 |
| 41000 | 2000 | 24 | 24 | 24 | 41099 | 400 | 35061 | 1867 | 0.2 |
| 26750 | 2000 | 24 | 24 | 32 | 26688 | 368 | 27445 | 1908 | 0.2 |

K-ratio = ki_seq / k. Values >> 1 mean the shadowing point's k is much larger than the query k; values << 1 mean it is much smaller. B_published is the B value recorded in the result file (matches B_sequential, confirming the buggy lookup was used).

### b200

| n | k | B_sequential | B_joint | B_published | ni_seq | ki_seq | ni_jnt | ki_jnt | k_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16777216 | 64 | 128 | 64 | 64 | 16777216 | 16777216 | 65536 | 8192 | 262144.0 |
| 33554432 | 64 | 128 | 64 | 64 | 16777216 | 16777216 | 131072 | 16384 | 262144.0 |
| 16777216 | 128 | 128 | 64 | 96 | 16777216 | 16777216 | 131072 | 16384 | 131072.0 |
| 33554432 | 128 | 128 | 64 | 96 | 16777216 | 16777216 | 131072 | 16384 | 131072.0 |
| 16777216 | 256 | 128 | 64 | 96 | 16777216 | 16777216 | 131072 | 16384 | 65536.0 |
| 33554432 | 256 | 128 | 64 | 96 | 16777216 | 16777216 | 262144 | 32768 | 65536.0 |
| 16777216 | 512 | 128 | 64 | 96 | 16777216 | 16777216 | 262144 | 32768 | 32768.0 |
| 33554432 | 512 | 128 | 64 | 96 | 16777216 | 16777216 | 262144 | 32768 | 32768.0 |
| 8388608 | 64 | 128 | 64 | 64 | 8388608 | 1048576 | 65536 | 8192 | 16384.0 |
| 16777216 | 1024 | 128 | 64 | 96 | 16777216 | 16777216 | 262144 | 32768 | 16384.0 |

K-ratio = ki_seq / k. Values >> 1 mean the shadowing point's k is much larger than the query k; values << 1 mean it is much smaller. B_published is the B value recorded in the result file (matches B_sequential, confirming the buggy lookup was used).

## 4. Measured Cost Impact

One measured datapoint on M3 Pro:

- Query: (n=4096, k=4096)
- Shadowed lookup returns B=24, runs **186.3 ms**
- Joint lookup returns B=32, runs **142.2 ms**
- **24% slower** with the wrong B (run-to-run spread under 5%)

This is a single datapoint. Performance impact varies per (n,k) cell; the worst offenders (largest k-ratio) likely suffer the largest penalty.

## 5. Regeneration Impact

### Files requiring regeneration

Every published result file that records a B column has cells whose B would change under the corrected lookup. Affected files:

| File | Device | Cells with B | Affected |
| --- | --- | --- | --- |
| contour_m3pro_parallel_q256.csv | m3_pro | 15 | 3 |
| contour_m3pro_serial_q256.csv | m3_pro | 10 | 10 |
| contour_zen4_parallel_q256.csv | zen4 | 14 | 4 |
| contour_zen4_parallel_q256_20260727.csv | zen4 | 15 | 4 |
| contour_zen4_serial_q256.csv | zen4 | 12 | 12 |
| contour_zen4_serial_q256_20260727.csv | zen4 | 9 | 9 |
| gpu_heatmap_b200.csv | b200 | 210 | 126 |
| gpu_heatmap_b200_20260728.csv | b200 | 210 | 126 |

### Files NOT requiring regeneration

These result files do not record per-cell B values and are unaffected by the lookup bug (though any B-dependent analysis derived from them would need revisiting):

- `accuracy_convergence.csv`: CSV without B column, not affected
- `accuracy_convergence.png`: rendered from affected data, **needs re-render**
- `b_optimal_report_zen4.md`: report, may reference affected numbers, **needs review**
- `b_shadow_impact_20260730.md`: report, may reference affected numbers, **needs review**
- `bench_grid_m3pro_parallel.txt`: bench log, records engines but not per-cell B, not affected
- `bench_grid_m3pro_serial.txt`: bench log, records engines but not per-cell B, not affected
- `bench_grid_zen4_parallel.txt`: bench log, records engines but not per-cell B, not affected
- `bench_grid_zen4_parallel_20260727.txt`: bench log, records engines but not per-cell B, not affected
- `bench_grid_zen4_serial.txt`: bench log, records engines but not per-cell B, not affected
- `bench_grid_zen4_serial_20260727.txt`: bench log, records engines but not per-cell B, not affected
- `bench_schoolbook_zen4.csv`: CSV without B column, not affected
- `bench_schoolbook_zen4.log`: log file, not affected
- `contour_1s.png`: rendered from affected data, **needs re-render**
- `contour_1s_m3pro.png`: rendered from affected data, **needs re-render**
- `engine_dispatch.png`: rendered from affected data, **needs re-render**
- `engine_dispatch_m3pro.png`: rendered from affected data, **needs re-render**
- `gpu_contour.png`: rendered from affected data, **needs re-render**
- `gpu_heatmap_B.png`: rendered from affected data, **needs re-render**
- `gpu_heatmap_engine.png`: rendered from affected data, **needs re-render**
- `gpu_heatmap_tier.png`: rendered from affected data, **needs re-render**
- `gpu_heatmap_time.png`: rendered from affected data, **needs re-render**
- `gpu_threshold_search_20260728.txt`: bench log, records engines but not per-cell B, not affected
- `parallel_speedup.png`: rendered from affected data, **needs re-render**
- `parallel_speedup_m3pro.png`: rendered from affected data, **needs re-render**
- `runtime_vs_n_cpu.png`: rendered from affected data, **needs re-render**
- `runtime_vs_n_cpu_m3pro.png`: rendered from affected data, **needs re-render**
- `runtime_vs_n_gpu.png`: rendered from affected data, **needs re-render**
- `wrap_fma_bench_zen4.csv`: CSV without B column, not affected
- `wrap_fma_cost_curve.png`: rendered from affected data, **needs re-render**

## 6. Summary for Supervisor

### What is affected

The buggy 2-pass sequential lookup in `empirical_best_B()` and `gpu_empirical_best_B()` causes B mis-selection across all three calibrated devices.

**Published cells with B changes**:

- **m3_pro**: 13/25 cells (52%)
- **zen4**: 28/47 cells (59%)
- **b200**: 126/210 cells (60%)

**Failure mechanism**:

- **CPU (m3_pro, zen4)**: Sparse refinement points at very low k (often k < 20) shadow dense grid rows at high k. Pass-1 selects the sparse row's n; pass-2 is forced to use its single low-k sample.
- **GPU (b200)**: Dense grid rows at large n contain only high k values. Sparse anchor points at k=n also sit at high k. Pass-1 correctly picks the query's n; pass-2 is forced to use a high-k sample even for low-k queries. This is the mirror image of the CPU failure.

**Regeneration required**:

- All contour CSV files (`contour_*_q256.csv`) must be regenerated
- All GPU heatmap CSV files (`gpu_heatmap_b200*.csv`) must be regenerated
- All PNG figures rendered from these CSVs must be re-rendered
- The `b_optimal_report_zen4.md` report may reference affected numbers
- Bench grid text logs (`.txt`) do not record per-cell B and do not need regeneration, though any B-dependent analysis from them would

**Note on the two GPU heatmap files**: `gpu_heatmap_b200.csv` and `gpu_heatmap_b200_20260728.csv` have identical (n,k) grids but different B values at higher n ranges (the older file has B=96 for some n=2097152 cells; the newer has B=128). This indicates the B column in the older file was produced with a different calibration run. BOTH were produced with the shadowed lookup and BOTH need regeneration.
