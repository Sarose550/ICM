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

### Explicitly deferred (by user instruction)

- **`select_best_B()`** (block-size choice within the hybrid engine) is
  untouched. If dispatch still looks off in a way traceable to a bad B
  choice, address it as its own scoped follow-up — do not fold it into
  the crossover-table fix.
- **Subset-query dispatch** (`n_targets > 0`) still uses the old
  analytical formula. Never measured/calibrated directly this session.

### Not yet done

- **Regenerate result data** (performance grids, contour sweeps) on both
  platforms now that dispatch is trustworthy — per `CLAUDE.md`'s
  M3 Pro/Zen4 validation steps. The numbers in `RESULTS.md` predate all
  of this session's fixes.
- **Paper sync**: Table 1/2 shared-k-column rework in
  `~/Documents/ICM_paper`, using post-fix numbers.
- **GPU kernel microbenchmark (B200)** — independent of all CPU work
  above, needs explicit user go-ahead to spin up a paid instance. GPU
  cost model uses a completely different mechanism (kernel lookup
  tables + SM-occupancy penalties) — checked this session, the bug
  classes found on the CPU side do not ripple into `src/gpu/gpu_plan.cu`.
- **Decide with the user whether to merge PR #7.**

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

1. **Regenerate result data** (performance grids, contour sweeps) on
   both M3 Pro and Zen4 now that dispatch is trustworthy on both.
2. **Paper sync**: Table 1/2 shared-k-column rework in
   `~/Documents/ICM_paper`, using post-fix numbers.
3. **If `select_best_B()` still looks wrong** in the regenerated data,
   address it as its own scoped follow-up (same crossover-table
   methodology could apply, calibrated separately — but confirm it's
   actually broken first, don't assume).
4. **GPU kernel microbenchmark (B200)** — needs explicit user go-ahead
   to spin up a paid instance.
5. **Decide with the user whether to merge PR #7.**
