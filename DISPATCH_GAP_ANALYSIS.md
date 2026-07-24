# Dispatch Gap Quantification, Post-Schoolbook-Fix Analysis

**Date:** $(date)
**Machine:** M3 Pro (Apple Silicon)
**Worktree:** ea5323

## 1. Schoolbook Fix Impact

The schoolbook fix (commit `8012244`: per-size lookup replacing extrapolated FMA_NS
constant for `polymul_modk` and `correlate_school`) dramatically improved the
cost-model accuracy:

| Metric | Pre-Fix | Post-Fix |
|--------|---------|----------|
| geo_mean(measured/predicted) | 1.740 | **1.126** |
| log-stddev | 0.236 | **0.143** |
| ratio range | [0.89, 3.67] | [1.00, 2.32] |
| mean abs % error | n/a | 14.0% |

The model bias dropped from +74% to +12.6%. The schoolbook fix was real and
necessary. However, the dispatch crossover point barely moved, still at
k≈260-320 instead of the ground-truth k≈120-160.

## 2. Leaf-Extraction Phase Profiling (Embedded)

Using `tools/probe_leaf_extract.c` (new tool, mirrors `probe_tree_levels.c`
methodology: Q=256, median over 21 independent hybrid-engine runs, phase-by-phase
timing within `engine_hybrid_core`):

| Phase | geo_mean(meas/pred) | log-stddev | Interpretation |
|-------|---------------------|------------|----------------|
| **Leaf divide** | **0.484** | 0.023 | Model OVER-predicts by 2.07× |
| Block build | 1.089 | 0.023 | Model underpredicts by 8.9% |
| **Tree** | **1.158** | 0.056 | Model underpredicts by 15.8% |

**Key finding: leaf extraction on M3 Pro is the OPPOSITE of what Zen4 found.**
On Zen4, leaf was *underpredicted* by 1.86-2.44×. On M3 Pro, the model
OVER-predicts leaf cost by 2.07x: the real per-player leaf cost is ~2.94 ns
vs. the model's 5.875 ns (and even vs. FP64_DIV_NS=3.80 ns).

The isolated microbenchmark (`tools/bench_leaf_fma.c`) that produced the
`leaf_fma_ns_per_player[]` table does not translate to the embedded hybrid-engine
context on M3 Pro. In the real engine, the division/FMA chains in the leaf
divide overlap across blocks, achieving throughput below either isolated
bottleneck.

## 3. Remaining Gap Decomposition

The total hybrid-cost model bias (+12.6%) decomposes as:

- **Tree underprediction (+15.8%)**: The dominant error source. Tree is ~65-91%
  of total hybrid cost at crossover-relevant (n,k). The FFT-path cost model
  (`calib_times_ns` + `FMA_NS` × wrap terms + `PAIRED_CACHED_CORR_RATIO`) is
  systematically low by ~16%.

- **Block-build underprediction (+8.9%)**: Minor. Block build is only ~6-20% of
  total cost.

- **Leaf OVER-prediction (−51.6%)**: Partially masks the tree error. Leaf is
  only ~3-8% of total cost, so this masking effect is small (~3-5 percentage
  points).

**The tree underprediction and leaf overprediction partially cancel**, which is
why the net model bias is only +12.6% despite the tree being off by +15.8%.

## 4. Crossover Impact

If we fix ONLY leaf (reduce `leaf_fma_ns_per_player[0]` from 5.875 to match
reality at ~2.84, making `FP64_DIV_NS=3.80` the binding constraint):
- At n=512: hybrid prediction drops by ~1,063 ns/qp
- At n=8192: hybrid prediction drops by ~17,012 ns/qp
- Dispatch would switch to hybrid at slightly lower k, but the tree
  underprediction (~70,000 ns/qp at n=8192,k=320) still dominates

**Leaf fix alone would NOT close the dispatch gap to k≈120-160.** It would move
the crossover maybe 10-30 k-units lower but would not reach the target.

## 5. Verdict

| Source | % of remaining gap | Action needed |
|--------|-------------------|---------------|
| Tree cost model | ~75% | Investigate FFT-path cost formula (FMA_NS in wrap correction? PAIRED_CACHED_CORR_RATIO? calib_times_ns accuracy in multi-level trees?) |
| Leaf over-prediction | ~15% (masking) | Re-measure `leaf_fma_ns_per_player[]` embedded (not isolated), or lower to FP64_DIV_NS floor |
| Block build | ~10% | Minor; re-measure if tree+leaf fixes aren't enough |

**The tree cost model is the next target.** The leaf fix is a distraction on
M3 Pro: it would make the model predict hybrid as *cheaper*, partially
compensating for the tree underprediction, but not solving the root cause.

## Tools Committed

- `tools/quantify_dispatch_gap.c`: combined model-vs-measured comparison +
  crossover projection (Part 1 + synthesis from task description)
- `tools/probe_leaf_extract.c`: phase-by-phase timing of block_build, tree,
  and leaf_divide embedded in real hybrid engine runs (Part 2 from task)
