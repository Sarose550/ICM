# HANDOFF.md

## What this project is

ICM (Independent Chip Model) equity computation for poker tournaments —
a high-performance C library computing tournament placement equities via
generating-function quadrature. Three CPU engines (linear/hybrid/tree)
with cost-model-driven automatic dispatch, plus a CUDA GPU implementation.
Repo: GitHub `Sarose550/ICM`, working branch `results-gpu-section`, PR #7
open (https://github.com/Sarose550/ICM/pull/7) — **do not merge without
the user's explicit go-ahead**, keep pushing commits onto it. Sibling
repo `~/Documents/ICM_paper` (local-only, no remote) holds the
accompanying academic paper (`icm_paper.tex`); the compiled PDF is
copied into `paper/icm_paper.pdf` in the main repo and committed there.

## Goal

Ship this repo publicly in genuinely portfolio-ready shape: professional
code, an accurate paper, a friction-free device-porting story, and
nothing stale, hand-waved, or silently broken. This has been true across
multiple sessions.

## START HERE if you are a fresh supervisor session (2026-07-23)

The `SPRINT_CALIBRATION_AND_READINESS_DAG.md` board that drove this
session's work is now CLOSED and deleted (ephemeral, per the
`supervisor-dag` skill — do not go looking for it). Everything it
produced is summarized below and in the "Adaptive B-selection
calibration" section. Check `.claude/.dag-active-lock.json` doesn't
exist (it shouldn't — released at close) before starting any new work.

### This session's outcome, in one paragraph

Widened the CPU/GPU B-selection calibration from a naive rectangular
grid to an adaptively-refined one (7-smooth skeleton + per-band
convergence-based refinement loop), landed the new
`tools/calibrate_block_size.py` orchestrator, fixed two real bugs found
while running it on real hardware (a SIGFPE crash for small-`k` points,
and a quadratic-cost cliff at large `n` with `k=n`), and regenerated
both M3 Pro's and Zen4's B-selection tables with it. Also closed out
every standing readiness item carried over from the prior session:
Zen4 memory-wall documented, paper synced, codebase comments cleaned
up, subset-dispatch path investigated (found genuinely broken, not
fixed — new scoped follow-up below), B200's large-n B-selection
irregularity explained (real VRAM/batch-count effect, table already
correct). B200's own adaptive calibration run (B4) was intentionally
skipped — see below. PR #7 merge decision is still open, unchanged.

**What the board covers, in one paragraph:** both CPU (`select_best_B()`)
and GPU (`gpu_select_best_B_est()`) dispatch now use empirical B-selection
lookup tables (fixed this session — see "GPU B-selection" section below),
but both tables were built from a naive rectangular grid. The board widens
the calibration methodology (7-smooth-biased skeleton `n` values reusing
the codebase's own smooth-number tables; `k`-anchors covering tiny values
2-16, "almost-7-smooth-minus-1" up to 256, AND relative fractions spanning
the FULL range from n/12 up through k=n itself — not just a narrow
"typical payout %" band, since `n` here is players/payouts remaining at
call time, and late-tournament calls can have `k` be most or all of a
much-smaller `n`; a fully-specified per-band adaptive refinement loop
with a convergence-based stopping rule — N consecutive clean off-grid
probes, not a fixed point budget, tracked independently per `n`-band so
an easy region can't mask an under-covered one; and reduced-rep timing)
and applies it to all three platforms (M3 Pro, Zen4, B200) — while ALSO
carrying forward every standing item from this file's "Next Steps" (paper
sync, codebase cleanup, subset-dispatch check, the Zen4 memory-wall
documentation decision, and the still-unresolved PR #7 merge decision).
Nothing standing was dropped when the board was written; it's additive,
not a restart. Full rationale and the exact algorithm are in the board's
own "Context" section — don't re-derive it, read it there. **The actual
deliverable is one simple orchestrator script per device**
(`tools/calibrate_block_size.py`, board node A4) that a user runs as a single
command — the C/CUDA timing tools are measurement primitives it calls,
not something a user chains by hand.

**Key constraint carried into the board:** calibration POINTS are chosen
adaptively (offline, ahead of time); nothing is adaptive or probed live at
runtime — dispatch stays O(1) nearest-neighbor lookup, unchanged.

**Model-routing correction, learned mid-session:** DeepSeek workers CAN
use SSH/network (`deck spawn --allow-network`) — an earlier board draft
wrongly assumed they're always network-blocked. Zen4 (rented, staying up
regardless) is fine to delegate with that flag. B200 is never delegated
to DeepSeek (cost + destructive-action risk, per explicit user
instruction). M3 Pro is more subtle: a DOCUMENTED prior-session constraint
(`feedback_deepseek_deck_long_processes.md`) says DeepSeek workers cannot
reliably keep a local background process alive past their own session —
even `nohup`/`setsid` got killed when the sandbox tore down, observed
directly during last week's M3 Pro calibration run — so the board scopes
M3 Pro's actual long calibration RUN to supervisor (launched directly,
monitored via `Monitor`/`ScheduleWakeup`), with DeepSeek only doing the
bounded setup work. General rule threaded through the whole board: any
DeepSeek node whose work is a long-running execution does setup + kicks
it off + verifies the launch succeeded, then hands off to supervisor for
monitoring rather than waiting itself — don't trust a DeepSeek worker to
babysit something that risks burning significant wall-clock time with no
one competent watching if it stalls.

**Standing account/credential notes**: Zen4 box `185.8.107.239` (see
`reference_zen4_new_password.md` memory) was still up and reachable as of
2026-07-23 — never terminate it without explicit user go-ahead (destroying
is fine to avoid — it's the *terminate* that's forbidden). B200 work uses
vast.ai; user has explicitly greenlit renting instances for this specific
calibration work, budget-conscious (~$6-9 per session), always destroy the
B200 instance immediately after each session (already-established
practice, not new).

## Status as of 2026-07-22 — the CPU dispatch-crossover investigation is CLOSED on both platforms

The multi-session investigation into why `icm_select_engine()`'s real
dispatch decision didn't match the ground-truth measured crossover is
**resolved on both M3 Pro and Zen4**, via a methodology change, not
another constant fix. Both platforms now have `icm_select_engine()`
matching `bench_grid crossover`'s empirical L→H transition essentially
exactly.

### The resolution: replaced the summed-constants formula with an empirically-measured crossover table

Every individual constant feeding the old analytical cost formula
(`calib_times_ns[]`, `WRAP_FMA_NS`, `PAIRED_CACHED_CORR_RATIO`,
`leaf_fma_ns_per_player[]`, `BATCHED_FMA_NS`, `block_build_ns_per_player[]`)
was, over the course of this session, directly validated against real
embedded execution and found individually accurate or fixed to be so —
see "Root causes fixed along the way" below. Yet the AGGREGATE go/no-go
dispatch decision still didn't match reality, on both platforms, by a
wide margin, even after four parallel DeepSeek analyses each targeting a
different remaining hypothesis (schoolbook dead-code, FFT-cached ratio,
wrap-correction contribution, missing overhead terms) came back small or
inconclusive.

Research (this session) confirmed this matches a well-established result
in the HPC autotuning literature: closed-form analytical cost models are
fragile in aggregate even when every individual term is correct, because
summing terms can't capture real microarchitectural effects (cache
associativity, prefetch behavior, TLB pressure). FFTW's own `ESTIMATE`
mode is an admitted-inaccurate heuristic for exactly this reason — its
`MEASURE`/`PATIENT` modes abandon modeling and time real candidates
directly. ATLAS's AEOS paradigm does the same at install time. The
precise structural analog is **LAPACK's `ILAENV` ISPEC=3 (`NX`)
parameter**: a problem-size crossover between two algorithms (blocked vs
unblocked), determined by direct empirical benchmarking per machine, no
live racing in production — just a cheap runtime threshold read.

Implemented (commit `27cc356`):
- `tools/calibrate_crossover.c` — binary-searches the real crossover
  `k_cross(n)` via median-of-7-reps timing at a sparse grid of n
  (512..16384), matching `bench_grid crossover`'s exact methodology
  (same payout convention, Q=256, `icm_select_best_B()` for the hybrid
  side) but far less noisy. **Important finding along the way**:
  `bench_grid crossover` itself takes only a single, un-averaged sample
  per cell — a real, non-trivial noise source that was silently
  distorting the "ground truth" this whole investigation had been
  chasing (visible as a cold-start bias on the linear engine's first
  call in single-shot timing). This is a one-time, offline calibration
  step — it never runs in production, same as FFTW wisdom generation.
- `src/fft_cost_model.h`'s `empirical_crossover_k()` — log-linear
  interpolation between the two bracketing calibrated `n` values,
  clamping (not extrapolating) outside the measured range.
- Per-device `crossover_n[]`/`crossover_k[]` tables in each
  `fft_config.h`, following the existing calibrated-constant convention.
- `select_engine_ex()` (`src/icm.c`) now uses the table for full-equity
  queries (`n_targets <= 0`). Subset queries still use the analytical
  formula — the table was only calibrated for full-equity dispatch.
  **`select_best_B()` (the block-size choice within hybrid) is
  untouched** — explicit user instruction: "if B selection is still an
  issue we'll address it after." Revisit separately if needed.

### Verified results

- **M3 Pro**: `crossover_n`/`crossover_k` = `{512,1024,2048,4096,8192,16384}`
  / `{123,124,122,122,122,122}` — remarkably tight and consistent.
  Dispatch now matches `bench_grid crossover`'s empirical transition
  (k=120 still L, k=160 H, for every n) essentially exactly.
- **Zen4** (box `84.32.71.47`, see `reference_zen4_new_password.md`):
  `crossover_k` = `{194,231,242,242,242,242}` for the same n grid.
  Dispatch now matches `bench_grid crossover`'s empirical transition
  (k=240 still L, k=260 H, for n≥1024) essentially exactly.
- `bench_grid verify`: ALL TESTS PASSED on both platforms, both before
  and after this change (correctness untouched, only dispatch moved).

### Root causes fixed along the way (in case this pattern recurs elsewhere)

These were all real, individually-validated fixes — necessary, and each
moved dispatch closer to reality, but none alone closed the gap; the
crossover-table methodology change is what actually closed it.

1. **Leaf-extraction cost overpredicted ~2x on both platforms.** The old
   isolated microbenchmark (`tools/bench_leaf_fma.c`, retired) generated
   synthetic `a[j]` values forcing 100% of measured players through the
   expensive forward-divide branch; real production data is ~99.9% the
   cheap "zero" branch (`aj` underflows, plain FMA accumulate, no
   division). Fixed via `tools/probe_leaf_extract.c`'s B-sweep phase
   (real embedded execution, fresh `HybridCtx` per rep).
2. **Dispatch-formula/execution-path constant mismatch.**
   `select_engine_ex`/`select_best_B` used `FMA_NS` for the wrap-
   correction term where the actually-executed code used `WRAP_FMA_NS`
   (7.6x discrepancy on M3 Pro) — fixed as a correctness issue
   independent of dispatch effects.
3. **Linear-engine cost model undercounted FMA operations.**
   `cost_model.h`'s `linear_roofline_cost()` assumed `4*n*k` FMAs/QP;
   the real BQ=8 batched inner loop (`src/linear_batched_impl.inc`) does
   `~5*n*k`. Fixed with a new `BATCHED_FMA_NS` constant, fit directly
   against real `icm_run_linear_batched()` measurements.
4. **A genuine, universal correctness bug** (not a dispatch/calibration
   issue): `correlate_fft()`/`correlate_fft_pair()` (the non-cached
   correlate variants, used only at the tree's root) had **no
   wrap-correction logic at all**, unlike their "cached" siblings.
   Silently zeroed output positions whenever the root's FFT size was
   smaller than the full convolution length. Found via a Zen4
   `bench_grid verify` failure (`xchk FAIL`, up to 65% diff), confirmed
   universal by reproducing the identical wrong answer on M3 Pro via a
   forced code path. Fixed (commit `dce6b74`) — first attempt copied the
   wrong indexing scheme from the cached variant and made things worse,
   caught by testing the isolated primitive against a schoolbook
   reference before re-testing the full tree.
5. **AOCL-FFTW wasn't actually being used on the fresh Zen4 box.** Built
   with only `--enable-avx`, but the committed wisdom file contains
   AVX/AVX2/AVX-512 codelets — 100% wisdom-lookup miss, silently falling
   back to slow `FFTW_ESTIMATE` for every single FFT plan. Fixed by
   rebuilding AOCL-FFTW (`github.com/amd/amd-fftw`) with the full
   `--enable-sse2 --enable-avx --enable-avx2 --enable-avx512
   --enable-amd-opt` flag set. Also note: `fft_cache_create_sizes()`
   calls `wisdom_save()` unconditionally on every run, which can silently
   clobber a good wisdom file with an incomplete one if run against a
   broken FFTW build — re-copy `devices/<DEVICE>/fftw_wisdom.dat` to the
   repo root before any run if in doubt.

### Follow-up completed: `select_best_B()` had the same problem, now fixed too

Checked for methodological consistency (same class of formula, same
"summed constants fragile in aggregate" risk) before trusting it for
result-data regeneration. `tools/validate_best_b.c` confirmed
`select_best_B()` was measurably wrong — 7-11% slower on M3 Pro (12/19
test points off, systematic bias toward B=64 when B=32 real-wins), 2-9%
slower on Zen4 (bias toward B=48 when B=24 real-wins). Same root cause,
same direction, as the crossover bug.

Fixed the same way (commit `c70ca4e`): `tools/calibrate_best_b.c`
directly times every candidate B at a 34-point (n,k) grid per device
(median of 7 reps, Q=256); `src/fft_cost_model.h`'s new
`empirical_best_B()` does 2D nearest-neighbor lookup (not
interpolation — B is discrete, no meaningful value between B=32 and
B=64). `select_best_B()` itself is drastically simplified — the entire
summed-constants tree-cost loop removed, replaced by the table lookup
with a safety fallback (largest valid candidate ≤ n and ≤ k) for values
outside the calibrated range.

Real optimum clusters tightly around B=32 on M3 Pro (one clean, isolated
exception at k=400/n≥2048 → B=48) and around B=24/32 on Zen4 (noisier —
these two are close enough that the winner flips between adjacent grid
points; used the raw measured data as-is rather than smoothing, since
even a "wrong" pick between them costs little per the validation data).
`bench_grid verify`: ALL TESTS PASSED on both platforms.
`icm_select_best_B()` now matches the calibration data exactly across
the test grid; `bench_grid crossover`'s L→H transition unchanged on both
platforms, confirming no regression to the crossover-table fix.

### Explicitly deferred (still open)

- **Subset-query dispatch** (`n_targets > 0`) still uses the old
  analytical formula for both the linear-vs-hybrid decision and B
  selection. Never measured/calibrated directly this session — revisit
  if subset dispatch is shown to need the same treatment.

### Done since the above was written

- **Result data regenerated on both platforms** (commits `e9bde27` Zen4,
  `8daba99` M3 Pro) after fixing two bugs found along the way: FFTW
  wisdom silently degrading (`wisdom_save()`'s unconditional overwrite —
  re-copy `devices/<DEVICE>/fftw_wisdom.dat` before trusting any run,
  verify with a direct `FFTW_WISDOM_ONLY` hit-rate test, not just file
  size) and a `make results-refresh` bug where `all`/`parallel` both
  build the same `$(OUT)` binary, so listing both as prerequisites let
  the OpenMP-enabled binary silently serve the "serial" run. Fixed by
  rebuilding explicitly inside the recipe body immediately before each
  binary's use.
- **Docs-closeout wave** (2026-07-22/23): a process audit caught that
  none of this session's boards had wired the two new calibration tools
  (`tools/calibrate_crossover.c`, `tools/calibrate_best_b.c`) into
  `tools/calibrate_full.sh`, and that `CLAUDE.md`/`OPTIMIZATION_GUIDE.md`/
  `README.md`/`RESULTS.md` still described the old summed-analytical
  dispatch mechanism or carried stale "predates this session's fixes"
  warnings. Fixed all four. **Caught and fixed a real bug in the
  DeepSeek-authored `calibrate_full.sh` update**: its new build commands
  for `calibrate_crossover.c`/`calibrate_best_b.c` omitted `src/icm.c` as
  a source file (these two tools `#include "icm.h"` only, unlike
  `sample_plans.c`'s single-TU `#include "icm.c"` convention) — silent
  link failure, would have broken a fresh device port at exactly the new
  steps. Verified the fix by compiling and running both tools standalone
  against the real M3 Pro config; results matched the already-committed
  calibration tables exactly. Also caught and fixed a stale "B=16 on M3
  Pro" claim in the DeepSeek-authored `OPTIMIZATION_GUIDE.md` diff (real
  data says B=32 dominant on both platforms now).
- **Paper sync**: still open, see Next Steps.
- **Decide with the user whether to merge PR #7**: still open.

### New finding: Zen4 parallel scaling collapses at n≥16384 (unexplained, under investigation)

While reviewing the regenerated Zen4 results, the raw parallel-mode
benchmark data (`results/bench_grid_zen4_parallel.txt`, not a docs
transcription error — verified directly against the file) shows parallel
speedup falling off a cliff right at the n=8192→16384 boundary: n=8192,
k=n is 12.0x; n=16384, k=n is 3.3x; n=65536, k=n is 3.3x. Below n=16384,
speedup is a healthy 10-14x as expected for 16 physical cores.

**RESOLVED (root cause confirmed on real hardware, box `185.8.107.239`,
2026-07-23):** it is a genuine memory-bandwidth/cache-capacity wall, NOT
a thread-affinity/NUMA/CCD-migration issue.

Initial hypothesis (refuted): `src/icm.c:2513`'s sole `#pragma omp
parallel for` has no explicit thread-affinity pinning, and the 7950X is
2 CCDs × 8 cores (confirmed via `lscpu -e`: CORE 0-7 = L3 domain 0, CORE
8-15 = L3 domain 1; logical CPUs 16-31 are SMT siblings of 0-15) — cross-
CCD Infinity Fabric traffic seemed a plausible culprit. Tested directly:
`OMP_PROC_BIND=close`, `OMP_PROC_BIND=spread`, and explicit `taskset
-c 0-15` + `GOMP_CPU_AFFINITY=0-15` pinning to all 16 distinct physical
cores — **none recovered speedup**; all landed at the same ~167-170ms
plateau as the unpinned baseline (150-153ms) at n=16384,k=16384, some
even slightly worse. Also tested raising glibc's `MALLOC_MMAP_THRESHOLD_`
in case large per-thread FFT buffer allocations were hitting mmap/munmap
lock contention — no improvement either.

**Actual cause, confirmed via `perf stat`:** comparing n=8192 (healthy,
~12x parallel speedup) vs n=16384 (collapsed, ~3.5x) under
`OMP_NUM_THREADS=16`:
- n=8192: IPC=1.53, cache-miss rate 4.4% of references
- n=16384: IPC=0.57 (2.7x worse), cache-miss rate 10.5% (2.4x worse) —
  cycles grew 6.3x while instructions only grew 2.35x, i.e. almost all
  the extra time is memory stalls, not more work.

This is consistent with the aggregate working set across 16 concurrently-
running hybrid-engine FFT trees crossing the combined 64MB L3 capacity
(2 CCDs × 32MB) somewhere between n=8192 and n=16384, forcing heavy DRAM
traffic that 16 threads then contend over — a real cache-capacity/
bandwidth ceiling, not a scheduling or placement bug, and not something
`OMP_PROC_BIND` can fix since the problem is total live data volume, not
locality of a fixed volume.

**Not attempted this session** (would need its own scoped pass): reducing
the hybrid engine's per-thread memory footprint at large n (e.g. tighter
buffer reuse across tree levels) to push the wall higher, or simply
documenting this as a known, real scaling limit in the paper/RESULTS.md
rather than treating it as a bug to fix. Recommend the latter unless a
memory-footprint reduction is independently worthwhile.

### New finding: GPU B-selection has the same systematic bias CPU had, confirmed on real B200 hardware

Spun up a B200 instance (vast.ai, ~$6 budget, cheapest reliable offer,
CUDA 12.8 devel image — the first image tried, `pytorch/pytorch:2.5.1-
cuda12.4-cudnn9-devel`, predates Blackwell/sm_100 support and was
destroyed within a minute of creation before real cost accrued). Built
and ran the existing `tools/validate_planner_gpu.cu` (already in the
repo, never previously run this session) — it forces every candidate B
in `{16,24,32,48,64,96,...,896}` at each of 12 (n,k) points for
n∈{65536,131072,262144,524288}, k∈{n/4,n/2,n}, and compares against the
planner's automatic choice.

**Result: 12/12 mismatches (100%).** `gpu_select_best_B_est()`
(`src/gpu/gpu_plan.cu`) picks B=128 every time; real measured optimum is
always B=64, consistently 2-4% faster (e.g. n=524288,k=n: auto 219.06ms
vs best 214.75ms). Same failure mode, same direction, as the CPU
`select_best_B()` bug fixed earlier this session (overestimating the
benefit of a larger block size) — strong evidence the same
"summed-analytical-constants fragile in aggregate" architectural problem
applies to the GPU cost model too, exactly as flagged when `gpu_plan.cu`
was read (not modified) earlier in the session.

Also ran `tools/gpu_sample_plans.cu` (250 real measured (n,k,B) plans,
`results_b200_validation/gpu_sample_plans_b200.csv`) — usable as seed
data for a future empirical B-selection table, same methodology as
`tools/calibrate_best_b.c` on CPU. `results_b200_validation/
planner_validation.csv` has the raw 12-row mismatch table. Instance
destroyed immediately after downloading both files — total B200 wall
time was well under the budget.

### FIXED: empirical GPU B-selection table (2026-07-23, second B200 session)

Same methodology as the CPU fix: added `tools/calibrate_gpu_best_b.cu`
(median of 3 reps — GPU timing is far less noisy than CPU wall-clock, so
7 wasn't needed) sweeping a representative subset of `kBCandidates`
across a 32-point `(n,k)` grid spanning n=4096 through the 1,572,864
frontier (`k` = n/8, n/4, n/2, n). Added `gpu_empirical_best_B(n,k)` in
`src/gpu/gpu_plan.cu` (2D nearest-neighbor lookup over
`gbselect_n[]`/`gbselect_k[]`/`gbselect_B[]` in
`devices/b200/gpu_fft_config.h`, mirroring CPU's `empirical_best_B()`
exactly) and rewired `gpu_select_best_B_est()` to consult it first,
falling back to the old analytical estimate only if no candidate fits
`n`/`k_pad` at all (mirrors CPU's `select_best_B()` fallback structure).

Ran the full 32-point calibration sweep on real B200 hardware (second
short paid session, same $6.89/hr instance class, ~35 min total —
largest points near the 1.5M frontier take ~1s per candidate). Real
data: B=64 dominant for n up to 524288 (matches the `validate_planner_gpu`
finding exactly), B=32 at n=1048576, and B=96/192/112 at n=1572864
depending on k — notably more varied at the largest scale than CPU's
tables ever were, plausibly because VRAM/occupancy effects become
relevant near the frontier.

**Verified fixed**: re-ran `tools/validate_planner_gpu.cu` after
rebuilding — **12/12 matches (100%)**, up from 0/12, with auto/best
timings now identical to measurement noise (e.g. n=524288,k=n: auto
214.78ms vs best 214.75ms). Also ran `bench_gpu_fused verify` (installed
plain FFTW via apt for the CPU cross-check reference — unrelated to the
CPU AOCL-FFTW-wisdom rule, just a build dependency) — all cases PASS, no
correctness regression. Instance destroyed immediately after downloading
results. Raw data: `results_b200_validation2/` (calibration CSV/log,
post-fix `validate_planner_gpu` output).

## Adaptive B-selection calibration methodology (2026-07-23 session)

The flat 34/32-point rectangular grids behind `select_best_B()` and
`gpu_select_best_B_est()` (see above) were themselves built cheaply and
never adaptively refined. This session replaced the grid-generation
step with an adaptive skeleton + per-band convergence loop, applied to
all three platforms.

### Methodology

- **Skeleton**: `tools/gen_calib_skeleton.py` picks `n`-anchors
  log-spaced then snapped to the nearest 7-smooth number (reusing the
  exact smooth-number logic already in `src/icm.c` /
  `src/gpu/gpu_plan.cu`, so skeleton points always match real
  calibrated FFT sizes). For each `n`, the `k`-anchor set spans three
  categories: `{2..16}` (tiny, exhaustive), `{s-1 : s` 7-smooth,
  `16<s-1<=256}` (forces a small nonzero wrap-correction), and relative
  fractions `{n/12,...,n/2,n}` — deliberately including `k=n` itself
  (the min-cash/bubble state, where `k` is the number of payout places
  *remaining at call time*, not the original field size, so it can be
  most or all of a much-smaller `n` late in a tournament).
- **Orchestrator**: `tools/calibrate_block_size.py` (one command per
  device) runs the base skeleton sweep, injects it into
  `fft_config.h`/`gpu_fft_config.h`, then a per-band adaptive
  refinement loop: draw a random point in the band, probe it via the
  single-point `validate_best_b`/`validate_planner_gpu` oracle, and if
  the table's current choice is more than 2% off the real optimum,
  measure that point properly and inject it into the table
  *immediately* (so later probes benefit right away). Each band stops
  independently on 25 consecutive clean probes, or a 150-probe safety
  cap (a signal the region needs attention, not silently absorbed).
  Calibration POINTS are chosen adaptively offline; runtime dispatch is
  still pure O(1) nearest-neighbor lookup, unchanged.
- **Primitives upgraded**: `tools/calibrate_best_b.c`/`validate_best_b.c`
  and their GPU counterparts now take a point-list CSV instead of a
  hardcoded grid, support a single-point-probe mode (the oracle above),
  a `--narrow-around` flag for single-point refinement, resumability,
  and a 1-rep-rank + confirm-if-close-top-2 timing strategy (replacing
  median-of-N-on-every-candidate).

### Two real bugs found running this on real hardware

1. **SIGFPE crash on small `k`.** The B-candidate validity filter in
   both CPU and GPU tools excluded `B > k`. For `k < 8` (smaller than
   every candidate), that leaves zero valid candidates; the CPU tool
   then read out-of-bounds into `B_candidates[-1]` and crashed with a
   floating-point exception on Zen4 (twice) — undefined behavior that
   happened not to crash on M3 Pro, purely by stack-layout luck.
   Production's `select_best_B()`/`gpu_select_best_B_est()` don't treat
   `B>k` as invalid (they fall back to a sane default), so all four
   tools were fixed to match: only exclude `B>n`.
2. **Quadratic wrap-correction cost cliff near the FFT calibration
   ceiling.** The CPU skeleton's original `--hi` default (131072)
   matched the FFT calibration table's own cap — but a `k=n` query
   needs a root-level FFT size of `~2n-1`, so points with `n` near that
   cap needed FFT sizes far outside the calibrated range. The
   wrap-correction cost model is quadratic in that shortfall, so those
   points took 20+ minutes each instead of seconds. Fixed by capping
   the CPU skeleton's `--hi` at 65536, so `k=n` never needs an FFT size
   beyond what's calibrated.

### Results

- **M3 Pro**: base skeleton 1117 points → 1950 adaptive probes → 1349
  points added → final table 2466 points.
- **Zen4**: same skeleton → 1950 probes → 827 points added → final
  table 1944 points. Run executed on a redeployed box (`84.32.71.35`)
  after the original instance ran out of provider credits mid-run;
  AOCL-FFTW wisdom was ported directly from the committed
  `devices/zen4/fftw_wisdom.dat`, never regenerated (~3.5-hour asset
  from a much earlier session, preserved).
- **All 13 bands hit their 150-probe safety cap without reaching the
  25-clean-streak target, on BOTH platforms independently.** This is a
  real finding, not noise: the 2%-gap/25-streak stopping criteria was
  tuned tighter than actual measurement noise allows on this hardware,
  so no band ever produced an "official" convergence signal — but the
  adaptive loop still measurably improved both tables (827-1349 real
  refinement points added beyond the base skeleton). If revisited,
  either widen the gap threshold or lower the clean-streak target to
  match real noise floors.
- Both platforms: wisdom files verified byte-identical before/after
  (no silent regeneration), `bench_grid verify` ALL TESTS PASSED.
- **B200 (B4) intentionally skipped** — user decision, mid-session:
  B200 already has a validated 12/12-match table from the earlier flat
  32-point sweep (see B1's finding above), and given how long the
  adaptive treatment took on CPU (multiple hours per platform), the
  cost of the same run on a paid-by-the-hour B200 instance wasn't
  justified when the existing table already works correctly.

## What Worked

- **Always re-running the actual `icm_select_engine()` dispatch decision
  after every fix** — never an aggregate ratio, never `bench_grid
  crossover`'s own (as it turned out, noisy single-shot) empirical
  winner alone. This caught multiple fixes that moved dispatch the wrong
  way, each time revealing there was more to find rather than declaring
  victory early.
- **Stepping back from "fix the next constant" to "is the whole approach
  right"** when four parallel, well-scoped analyses all came back
  inconclusive. Researching how mature HPC libraries (FFTW, ATLAS,
  LAPACK) solve this exact class of problem — direct empirical
  measurement of the real crossover, not summed analytical terms — is
  what actually closed the investigation, not another round of
  constant-hunting.
- **Testing hypotheses directly instead of accepting a plausible-sounding
  explanation.** Refuted the QoS-pinning hypothesis for M3 Pro's leaf
  anomaly by direct A/B test. Refuted "isolated benchmark under-amortizes
  fixed overhead" as the M3-Pro-vs-my-own-crossover-tool discrepancy by
  directly testing Q=32 vs Q=256 — then found the REAL explanation
  (hardcoded B=8 vs `select_best_B()`) by checking one more concrete
  difference instead of stopping at the first plausible theory.
- **Reviewing every DeepSeek worker's diff/output before merging, even on
  "success."** Caught a dangerous zero-initialized placeholder-array bug
  in an onboarding-script worker's diff this session (would have made
  schoolbook multiply look free if a calibration step ever silently
  failed) — `bench_grid verify` would not have caught it.
- **Verifying claimed "precedent" facts before asserting them.** When
  naming LAPACK's `ILAENV` as the analog to the crossover-table design,
  verified via research rather than asserting from memory — and it
  turned out to be the more precise fit over an initially-plausible
  MAGMA comparison.

## What Didn't Work / Failure modes to expect again

- **DeepSeek workers erroring mid-task while still producing useful
  partial output.** Several nodes this session hit an `error` status
  with an empty compact result, but the full log transcript revealed
  real, load-bearing findings already derived before erroring (the
  `FMA_NS`/`WRAP_FMA_NS` mismatch came from exactly this pattern).
  Always check the full log before writing off an errored run.
- **A separate recalibration tool duplicating an existing one's job can
  introduce its own bug instead of just being redundant.**
  `tools/calibrate_leaf_realistic.c` reused one `HybridCtx` across all
  reps (unrealistically cache-hot); the existing `probe_leaf_extract.c`
  (fresh context per rep) was already correct. Extending existing,
  validated tools beat writing new ones under time pressure.
- **A "roughly accurate" finding from one tool can be invalidated by a
  bug in a different tool sharing the same buggy formula.**
  `probe_tree_levels.c`'s "tree roughly accurate near crossover" finding
  was computed against its own stale copy of the schoolbook formula (pre-
  dating commit `8012244`) AND the same `FMA_NS`/`WRAP_FMA_NS` bug found
  in `src/icm.c` — invalid from the start until both were fixed.
- **Hand-rolling test harnesses against the exported C API is
  error-prone under time pressure.** Hit two real bugs writing ad-hoc
  direct-comparison test programs this session: wrong `EngineKind` enum
  values passed to `icm_ctx_destroy` (segfault), and a hardcoded `B=8`
  instead of calling `icm_select_best_B()` (silently wrong comparison,
  not a crash — much more dangerous). When comparing against an existing
  tool's ground truth, match its methodology exactly rather than
  approximating it.

## Guiding principles reinforced this session

- **When several individually-targeted fixes/analyses all come back
  small or inconclusive, question the overall approach, not just the
  next constant.** Don't joint-fit the existing formula's constants
  together (explicitly rejected by the user, and rightly — that's an
  empirical workaround, not a principled fix) — but do look at whether
  the *architecture* of the approach (summed analytical terms) is itself
  the mismatch, informed by how established prior art solves the same
  class of problem.
- **Always re-run the actual production entry point
  (`icm_select_engine()`) as the acceptance test** — never an aggregate
  ratio, never a single-shot "empirical" measurement without checking
  its own noise floor.
- **Review every worker diff before merging, even on "success."**
- **Never terminate a compute instance without explicit go-ahead. No
  `Co-Authored-By` trailers on any commit, ever.**
- **Conservative with GPU/remote-compute credits** — write/design code
  locally first, only spin up or touch a paid/remote instance when ready,
  and ask before starting.

## Next Steps

1. ~~Zen4 parallel-scaling cliff at n≥16384~~ **DONE** — documented as a
   known, real scaling limit in `RESULTS.md`'s Zen4 section (root cause:
   memory-bandwidth/cache-capacity wall, confirmed via `perf stat`); no
   fix attempted, per the default recommendation (unclear payoff vs.
   effort).
2. ~~Build a real empirical B-selection table for the GPU cost model~~
   **DONE** — `gpu_empirical_best_B()` + `tools/calibrate_gpu_best_b.cu`,
   verified 12/12 match via `validate_planner_gpu`.
3. ~~Paper sync~~ **DONE** — Table 1/2 reworked with post-fix numbers,
   dispatch-accuracy figure recomputed, em-dashes stripped, cost-model
   sections rewritten for the empirical-table mechanism, ATLAS citation
   corrected, PDF recompiled and copied into `paper/icm_paper.pdf`.
4. ~~Codebase-wide pass to remove stray/AI-tell comments~~ **DONE** —
   repo-wide pass over `src/**`/`tools/**`, verified with a clean
   rebuild + `bench_grid quick` before and after.
5. **NEW, scoped this session — subset-query dispatch is measurably
   wrong.** `icm_select_engine_ex()`'s analytical formula for
   `n_targets > 0` picks the wrong engine by a wide margin: confirmed
   37.1% and 45.1% slower than the correct choice at two representative
   points (`n=4096,k=200,n_targets=1024` and
   `n=8192,k=200,n_targets=2048`), same failure mode as the (already
   fixed) full-equity crossover/B-selection bugs. Root cause: the
   formula models linear subset cost as scaling with `target_frac`, but
   the linear engine's forward pass and g-propagation are always
   full-cost regardless of `n_targets` — only the final inner-product is
   skipped, so real linear subset cost is ~95%+ of full cost, not the
   ~62.5% the formula assumes at `target_frac=0.25`. **Not fixed** — the
   real fix needs its own calibration (`(n, target_frac) → crossover_k`,
   same empirical-table methodology as the full-equity fix already
   shipped) and should be its own scoped board/session, not bolted onto
   this one. Secondary, lower-priority finding: `select_best_B()`'s B
   choice may also be ~10% suboptimal for subset queries.
6. **Widened B-selection calibration (this session)** — see "Adaptive
   B-selection calibration methodology" above for the full writeup.
   Landed on M3 Pro and Zen4; B200's adaptive run intentionally skipped
   (user decision — existing flat-grid B200 table already validated,
   not worth the paid-instance cost). If ever revisited: consider
   widening the 2% gap threshold or lowering the 25-clean-streak target,
   since every band on both CPU platforms hit the 150-probe safety cap
   without an "official" convergence signal despite real, measurable
   table improvement.
7. **Decide with the user whether to merge PR #7.** Still open, still
   never auto-decided.

## Process note for future DAG boards

When a new calibration tool is created, its onboarding-script wiring
(`tools/calibrate_full.sh`) and any doc references to the mechanism it
replaces MUST be a node in the SAME wave, not a follow-up caught later by
audit. This session needed a dedicated `SPRINT_DOCS_CLOSEOUT_DAG.md` wave
after the fact to catch exactly this gap for
`tools/calibrate_crossover.c`/`tools/calibrate_best_b.c` — avoidable next
time by including it upfront.
