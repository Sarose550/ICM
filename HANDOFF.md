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

## Status as of this session (2026-07-22) — the M3 Pro dispatch-crossover investigation is CLOSED

The multi-session investigation into why `icm_select_engine()`'s real
dispatch decision didn't match the ground-truth measured crossover (the
thing the last several sessions kept circling back to) **is resolved for
M3 Pro**. Real dispatch crossover moved from k~260-320 (session start) to
k~100-120, matching `bench_grid crossover`'s own empirical linear→hybrid
transition (k=120 still linear, k=160 hybrid, across n in
{512,1024,2048,4096,8192}) closely. The sprint board for this exact
investigation was `SPRINT_HYBRID_COST_MODEL_VALIDATION_DAG.md`, now
deleted per the supervisor-dag skill's R_CLOSE step (ephemeral, not
durable law) — the durable summary is this file.

### The actual root causes (in case this pattern recurs elsewhere)

1. **Leaf-extraction cost model overpredicted 2x on M3 Pro.** The old
   isolated microbenchmark (`tools/bench_leaf_fma.c`, retired) generated
   synthetic `a[j]` values uniformly in `[0.5, 0.99]`, forcing 100% of
   measured players through the expensive forward-divide branch. Real
   production data (stack sizes 100-10000, realistic quadrature sweep) is
   **~99.9% the cheap "zero" branch** (`aj` underflows below 1e-15 → plain
   FMA accumulate, no division, no dependency chain — `src/icm.c` ~line
   2068). Fixed by extending `tools/probe_leaf_extract.c` (which embeds
   real `engine_hybrid_core` execution, fresh `HybridCtx` per rep — cold
   allocation, matching one real `icm_equity()` call) with a B-sweep
   phase, cross-validated within 2% against its existing multi-(n,k)
   sweep. A separate attempt at a dedicated recalibration tool
   (`tools/calibrate_leaf_realistic.c`) turned out to have its own bug —
   it reused one `HybridCtx`/buffer set across all reps, making everything
   unrealistically cache-hot — and was abandoned in favor of extending the
   already-correct `probe_leaf_extract.c` instead. **This fix alone only
   moved the crossover from k~260-320 to k~220-255 — real but partial.**
2. **Dispatch-formula/execution-path constant mismatch (`src/icm.c`).**
   `select_engine_ex` and `select_best_B`'s tree-cost prediction used
   `FMA_NS` (0.0677ns on M3 Pro) for the wrap-correction term, but the
   ACTUALLY-EXECUTED code (`correlate_fft_cached_pair_wrap`) and
   `src/fft_cost_model.h` both correctly use `WRAP_FMA_NS` (0.5160ns) for
   the same physical quantity — a 7.6x discrepancy between what dispatch
   predicts and what really runs. Fixed (6 occurrences, `FMA_NS`→
   `WRAP_FMA_NS`) as a correctness fix independent of its effect on the
   crossover. **This moved the crossover the WRONG way (to k~240-285)** —
   informative, not a regression: it revealed that `tools/probe_tree_levels.c`'s
   earlier "tree roughly accurate near crossover" finding had been computed
   against its own copy of the same buggy formula, an invalid comparison.
3. **The real dominant bug: linear-engine cost model, untouched all prior
   sessions.** `src/cost_model.h`'s `linear_roofline_cost()` assumed
   `4*n*k` FMAs per quadrature point using `FMA_NS` — measured from an
   unrelated scalar schoolbook microbenchmark (`polymul_modk`), not the
   batched linear engine's real BQ=8 interleaved inner loop
   (`src/linear_batched_impl.inc`). Direct measurement via the exported
   API (`icm_run_linear_batched`) showed a consistent **~1.73-1.80x
   underprediction across every (n,k) tested** — a flat multiplicative
   bias, meaning the model's FORM was basically right but the constant was
   wrong. Root cause: the real inner loop does **~5*n*k** FMAs/QP, not
   4*n*k (forward pass: `BQ*(2k-1)`/player; fused backward pass:
   `BQ*(3k-1)`/player) — a genuine FMA-count bug, not just a wrong
   constant. Fixed by introducing `BATCHED_FMA_NS`, fit directly against
   real `icm_run_linear_batched()` measurements using the corrected `5*n*k`
   form (CV 1.27% across 10 test points — the corrected form fits real
   data far better than the old one ever did). **This is what actually
   closed the gap.**

### Commits (chronological, all on `results-gpu-section`)
- `bc9af1e` — leaf-extraction fix (root cause 1)
- `bddf6b2` — wrap-correction constant fix (root cause 2)
- `c481336` — linear-engine cost model fix (root cause 3, closes the gate)
- `40e54e8` — onboarding script wiring (schoolbook/leaf/linear steps added
  to `tools/calibrate_full.sh`) + `OPTIMIZATION_GUIDE.md`/`README.md`/
  `RESULTS.md` corrections

### What's explicitly NOT done yet

- **Zen4 is unverified.** Only the shared-code wrap-correction fix
  (root cause 2) automatically applies to Zen4 (it's in `src/icm.c`,
  not device-specific). The leaf-extraction fix was M3-Pro-only this
  session (Zen4's leaf bias runs the OPPOSITE direction per the old
  `DISPATCH_GAP_ANALYSIS.md` — underpredicts, not overpredicts — so the
  same mechanism may not even apply the same way there). Zen4's
  `BATCHED_FMA_NS=0.0973` in `devices/zen4/fft_config.h` is an explicitly
  flagged PLACEHOLDER (scaled from Zen4's own `FMA_NS` by the M3 Pro
  ratio), NOT a real measurement. **Do not trust Zen4 dispatch decisions
  until `tools/bench_linear_batched_fma.c` and the leaf B-sweep are run
  for real on Zen4 hardware** (`185.8.107.239` — an ad-hoc, freshly
  `git init`'d local repo lives there, unrelated to the real GitHub
  history; use the established pattern of scp-ing files into a local
  worktree and committing there rather than trying to `git fetch`/`pull`
  directly from the box).
- **RESULTS.md's performance tables predate these fixes** — flagged with
  a stale-data warning (not regenerated — needs a fresh `./bench_grid`
  run) by this session's docs-audit pass.
- **G1A/G1B** (regenerate all result data invalidated by the
  broken-cost-model window) and any remaining paper Table 1/2 rework are
  now unblocked (dispatch is trustworthy on M3 Pro) but not yet done.
- **F: GPU kernel microbenchmark (B200)** — independent of all CPU work
  above, still queued, needs explicit user go-ahead to spin up a paid
  instance. GPU cost model uses a completely different mechanism (kernel
  lookup tables + SM-occupancy penalties, no `FMA_NS`/`WRAP_FMA_NS`) —
  checked this session, the FMA-count/wrong-constant bug class found on
  the CPU side does NOT ripple into `src/gpu/gpu_plan.cu`.
- **11 DeepSeek workers from this exact investigation are gone/replaced**
  — the deck folder for this sprint was `41994c`
  ("hybrid-cost-model-validation"). Check `deck ps`/`deck folder ls` if
  continuing related work; several nodes errored mid-task but left
  useful partial transcripts before erroring (this is a real, repeatable
  failure mode — see "What Didn't Work" below).

## What Worked

- **Always re-running the actual `icm_select_engine()` dispatch decision
  after every fix**, never trusting an aggregate ratio improving or
  `bench_grid crossover`'s own empirical winner as a proxy. This is what
  caught that the wrap-correction fix (root cause 2) moved the crossover
  the WRONG way, forcing the investigation to look elsewhere (linear
  engine) instead of declaring victory on a plausible-sounding but
  incomplete fix.
- **Testing hypotheses directly instead of accepting a plausible-sounding
  explanation.** The QoS-pinning (P/E-core scheduling) hypothesis for M3
  Pro's leaf anomaly was directly tested (pin vs no-pin, vary rep
  count/duration) and REFUTED — pinned and unpinned measurements were
  statistically identical. The real cause (unrealistic synthetic branch
  distribution in the isolated benchmark) was found only by directly
  instrumenting the actual branch taken under realistic data, not by
  further isolated-benchmark tweaking.
- **Reviewing every DeepSeek worker's diff/output line-by-line before
  merging, even after "success."** Caught a real, dangerous bug this
  session: an onboarding-script worker's injected placeholder arrays
  (`schoolbook_mul_ns[]`/`schoolbook_corr_ns[]`) were bare
  `static const double x[N];` declarations — legal C, silently
  zero-initialized, which would make schoolbook multiply look FREE if
  the real measurement step ever silently failed (its own writer prints
  a warning and exits 0 on parse failure, which `set -e` would NOT catch)
  — `bench_grid verify` only checks equity correctness, not dispatch
  quality, so this would not have been caught by CI. Fixed with an
  explicit `999.0` fail-safe sentinel matching the pattern already used
  elsewhere in the same script.
- **Checking for ripple effects before declaring a bug class closed.**
  After finding the `FMA_NS`/`WRAP_FMA_NS` mismatch and the FMA-count bug,
  explicitly grepped for other uses of the same constants and checked
  whether the GPU cost model shared the same mechanism (it doesn't) before
  moving on, rather than assuming the fix was fully contained.
- **Delegating aggressively to DeepSeek via the `deck` CLI /
  `supervisor-dag` skill**, keeping the supervisor's own reasoning for
  judgment calls (which fix to trust, what to commit, the final dispatch
  gate check) rather than re-deriving worker findings by hand.

## What Didn't Work / Failure modes to expect again

- **DeepSeek workers erroring mid-task while still producing useful
  partial output.** Two nodes this session (`T2_PROPOSE_TREE_FIX`,
  `L1b_RECONCILE_LEAF`) hit an `error` status with an empty compact
  result (`deck result` showed nothing useful), but `deck log` on the
  full transcript revealed real, load-bearing findings the worker had
  already derived before erroring (the `FMA_NS`/`WRAP_FMA_NS` mismatch
  came from exactly this pattern). **Lesson: when a worker errors, always
  check the full log before writing off its run as wasted — the failure
  is often in the final "wrap up and report" step, not the investigation
  itself.**
- **A separate recalibration tool duplicating an existing one's job can
  introduce its OWN bug instead of just being redundant.**
  `tools/calibrate_leaf_realistic.c` was written to recalibrate the leaf
  table, but reused one `HybridCtx`/buffer set across all reps —
  unrealistically cache-hot, ~4x too optimistic. The existing
  `tools/probe_leaf_extract.c` (fresh `HybridCtx` per rep) was already
  correct; extending it was the right move, not trusting the new tool's
  plausible-looking, internally-consistent-but-wrong numbers.
- **A "roughly accurate" finding from one tool can be invalidated by a
  bug discovered later in a DIFFERENT tool that happens to share the
  same buggy formula.** `probe_tree_levels.c`'s finding that the tree
  FFT-cached formula was "roughly accurate near the crossover" was itself
  computed against the same `FMA_NS`-instead-of-`WRAP_FMA_NS` bug later
  found in `src/icm.c` — the comparison was invalid from the start, not
  actually informative until the shared bug was found and fixed.

## Guiding principles reinforced this session

- **Delegate implementation/verification work to DeepSeek; the supervisor
  spends its own reasoning on judgment calls and final verification**,
  not on doing the delegable work itself.
- **Always re-run the actual production entry point
  (`icm_select_engine()`) as the acceptance test** — never an aggregate
  ratio, never `bench_grid crossover`'s own empirical winner alone.
- **Review every worker diff before merging, even on "success."**
- **Check for ripple effects of a bug class (other call sites, other
  subsystems like the GPU cost model) before considering it closed.**
- **Never terminate a compute instance without explicit go-ahead. No
  `Co-Authored-By` trailers on any commit, ever.**
- **Conservative with GPU/remote-compute credits** — write/design code
  locally first, only spin up or touch a paid/remote instance when ready,
  and ask before starting.

## Next Steps

1. **Zen4 verification** (needs real Zen4 hardware — box exists at
   `185.8.107.239`, ask before touching it): run `tools/bench_linear_batched_fma.c`
   and `tools/probe_leaf_extract.c`'s B-sweep for real on Zen4, replace
   the flagged placeholder `BATCHED_FMA_NS`, re-verify Zen4's leaf table
   (different bias direction than M3 Pro, don't assume the same fix
   applies), then re-check `icm_select_engine()` dispatch against Zen4's
   ground-truth crossover (k~260-270 per earlier sessions — re-verify
   this is still the right target after any Zen4-side fixes).
2. **Regenerate result data** (G1A/G1B) invalidated by the broken-cost-model
   window, now that M3 Pro dispatch is trustworthy — full performance
   grids, contour sweeps, per `CLAUDE.md`'s M3 Pro validation steps.
3. **Paper sync**: Table 1/2 shared-k-column rework in
   `~/Documents/ICM_paper`, using post-fix numbers.
4. **GPU kernel microbenchmark (B200)** — independent of all the above,
   needs explicit user go-ahead to spin up a paid instance.
5. **Decide with the user whether to merge PR #7.**
