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
multiple sessions; the current thread specifically is a full migration
of the CPU cost model away from aggregate regression (after that
approach was caught breaking production dispatch on real hardware), and
— as of this session — a deep, still-open investigation into why the
resulting microbenchmark-based model *still* doesn't correctly predict
the real hybrid-vs-linear dispatch crossover, even after every individual
constant was directly, correctly measured.

## Current Progress

### Prior sprints (SPRINT_ZEN4_AOCL_DAG.md, SPRINT_MICROBENCH_MIGRATION_DAG.md — see those files)

Both DAG files describe earlier, now largely-superseded work: AOCL-FFTW
confirmed correct on Zen4, the aggregate-regression cost-model approach
retired in favor of directly-measured constants (C1 in the second DAG),
block-build/leaf-extraction converted to per-B lookup tables (B1), and
D1/D2 (fresh recalibration runs on Zen4 and M3 Pro with the new pinned
constants). **All of that is done and merged.** What follows is the
NEW work from this session, which the DAG files do not yet reflect —
**a new DAG file should probably be created** (e.g.
`SPRINT_HYBRID_COST_MODEL_VALIDATION_DAG.md`) to track it properly,
since it turned out to be a much bigger sub-investigation than a single
"E: verify dispatch sanity" checklist item.

### This session's actual arc

1. **Merged C1, D1, D2** (the tail end of the prior DAG) — Zen4 and M3
   Pro both got fresh, fully-pinned (zero-free-parameter) recalibrations.
   `bench_grid verify` passed on both. Commits `72c188c`, `e0d13ed`
   (M3 Pro), `e01e87c`+`059c92b` (Zen4, merged).

2. **Did node E properly — and it failed.** The DAG's own E node says to
   check `icm_select_engine()`'s *actual dispatch decision* against the
   real measured crossover (`bench_grid crossover`), not just eyeball
   `bench_grid crossover`'s own output (crossover only tests
   `select_best_B()`'s hybrid-side timing, not whether `select_engine()`
   picks the right engine at all). Built a standalone harness
   (`icm_select_engine()` is exported, no rebuild needed) and found: on
   M3 Pro, dispatch persisted with linear until k≈240-300 instead of the
   ground-truth k≈120-160; on Zen4, dispatch switched to hybrid too
   early, around k≈160-200 instead of the ground-truth k≈260-270.
   **Opposite-direction errors on the two platforms** — ruled out a
   simple one-constant sign error.

3. **Root-caused via real-execution instrumentation, not more isolated
   microbenchmarks.** Restricting the measured-vs-predicted comparison to
   B=8 (the actual B selected in production) still showed the hybrid
   cost model's prediction swinging 1.4x-3.7x off from real measured
   `sample_plans_*.csv` data — ruling out "just fit one correction
   constant." Wrote `tools/probe_tree_levels.c` (committed, merged) to
   instrument the REAL tree-build/propagate execution with per-level
   timers and compare against the model's own per-level formula. Found,
   **cross-validated on both Zen4 and M3 Pro independently**: the tree's
   *schoolbook*-level cost formula (`(d_eff+1)^2 * FMA_NS`) underpredicts
   real cost by 1.6x-3.7x, worst at small polynomial sizes — because
   `FMA_NS` was measured as an asymptotic throughput slope, but small
   schoolbook multiplies are dependency-chain/latency-bound, not
   throughput-bound (same mechanism as the earlier B1 fix, just not yet
   applied to this term). FFT-based tree levels were confirmed accurate
   (~0.85-0.95x) on both platforms — not touched.

4. **Fixed the schoolbook cost model** (`tools/bench_schoolbook_tree.c`,
   new direct per-size lookup table reusing `calib_sizes[]`'s existing
   indexing, wired into all three call sites: `select_engine_ex`,
   `select_best_B`, `tree_ctx_create_ex2`'s `use_fft` decision). Commits:
   `8012244` (M3 Pro fix + data), `06c1eb1`→`77069e8` (Zen4 data, see
   "What Didn't Work" for a real bug caught here). **Found and fixed a
   second, independent bug during verification**: cps values beyond the
   microbenchmark's measured cutoff read a `-1.0` "unmeasured" sentinel,
   which a naive numeric comparison treated as an artificially cheap
   cost — wrongly forcing the O(cps²) schoolbook path at very large
   sizes instead of FFT. Fixed by forcing `use_fft=true` whenever the
   sentinel is hit. `bench_grid verify`: ALL TESTS PASSED after both
   fixes, on both platforms.

5. **Re-ran the actual acceptance test (icm_select_engine() dispatch) —
   it STILL didn't close the gap.** Dispatch still transitioned at
   k≈260-320 on M3 Pro, barely moved from before the schoolbook fix.
   Did NOT declare victory on a "looks better" signal (repeat of the
   same discipline as step 2) — dug into why with
   `tools/eval_model_vs_plans.c` (new, committed, merged).

6. **Took stock via a dedicated DeepSeek worker** (per explicit user
   instruction to consult DeepSeek before continuing, given "too many
   open ends") rather than immediately attempting another blind fix.
   Findings (committed in worktree `.deck-worktrees/ea5323`, commit
   `e6deee3` — **NOT yet merged into `results-gpu-section`**, see Next
   Steps):
   - Aggregate measured/predicted ratio improved from 1.740 → **1.126**
     after the schoolbook fix (bias cut from +74% to +12.6%) — real,
     substantial progress, just not sufficient to move the crossover.
   - Remaining error breakdown: **Tree (FFT-path) underpredicts by
     15.8%** (now the dominant term, ~75% of remaining gap — NOTE this
     contradicts `probe_tree_levels.c`'s earlier finding that FFT-cached
     levels were accurate; likely because that earlier probe's sample
     never actually hit an FFT-*uncached* level — "0 rows" in that
     bucket on both platforms — and the crossover-relevant n/k range may
     be where FFT-uncached levels start appearing). Block-build
     underpredicts by 8.9% (minor). **Leaf-extraction OVER-predicts by
     2.07x on M3 Pro** — the OPPOSITE direction from what an earlier
     probe found on Zen4 (leaf underpredicted 1.86x-2.44x there).
   - Explicit verdict: **do NOT fix leaf next** — it's currently masking
     part of the tree error on M3 Pro, and a leaf-only fix would not
     reach k≈120-160. The tree FFT-path formula is the next real target.
   - New tools from this worker, uncommitted-to-main:
     `tools/probe_leaf_extract.c`, `tools/quantify_dispatch_gap.c`,
     `DISPATCH_GAP_ANALYSIS.md` (full writeup) — all in worktree `ea5323`.

7. **User pushed back that this is too tangled to keep guessing at, and
   asked for hardware/architecture research** to actually understand the
   platform-specific inconsistency (M3 Pro's bias flips direction
   between tables; Zen4's doesn't) rather than another blind fix.
   Did that research (see next section) — **this is where the session
   ended**, mid-investigation. The research produced a concrete,
   testable hypothesis but no code changes yet.

### The hardware research finding (not yet acted on)

- **M3 Pro is heterogeneous**: 6 Performance cores @ 4.06GHz + 6
  Efficiency cores @ 2.8GHz, where each E-core has roughly HALF a P-core's
  execution width (so P-vs-E throughput for FP-heavy code can differ by
  2-3x, not just the ~1.45x clock ratio). **Zen4 is homogeneous** — no
  such split. This asymmetry only exists on the platform showing
  direction-inconsistent bias.
- **None of this project's benchmark tools set a QoS class.** On macOS
  there is no `taskset`-equivalent hard core-pinning — the only lever is
  Quality-of-Service (`pthread_set_qos_class_self_np`), and which core
  type a thread lands on depends on that QoS *and* how many P-cores are
  already occupied at that instant. A default command-line process
  generally prefers P-cores when they're free, but under any P-core
  contention it can silently fall back to E-cores with zero indication
  in the tool's own output. Every M3 Pro measurement this whole session
  (isolated microbenchmarks AND embedded probes) was subject to this,
  uncontrolled.
- **Two distinct candidate mechanisms, not mutually exclusive:**
  1. A universal, platform-independent effect (applies to both Zen4 and
     M3 Pro): isolated microbenchmarks loop the SAME operation
     thousands of times, letting the branch predictor/I-cache lock onto
     one code path; the real embedded hybrid engine interleaves block
     build + tree-FFT + tree-schoolbook + leaf-divide within the SAME
     hot per-quadrature-point loop, causing genuine contention no
     isolated measurement can capture. This is Heiser's "confusing
     calibration with evaluation" systems-benchmarking crime, textbook,
     and plausibly explains the shared "isolated underestimates real
     cost" direction seen in most terms on both platforms.
  2. An M3-Pro-specific effect (P/E core scheduling): could explain why
     M3 Pro's bias *flips direction* between lookup tables (schoolbook
     underpredicted, leaf *over*predicted) while Zen4's bias stays
     one-directional — Zen4 physically cannot have this problem
     (homogeneous cores), and it doesn't.
- Full sourcing (Apple core specs, Dougall Johnson's Apple
  microarchitecture research, Eclectic Light Co.'s QoS/core-type
  writeups) is in the chat transcript of this session, not yet copied
  into a repo doc — worth doing if this hypothesis is confirmed.

## What Worked

- **Doing the actual acceptance test (real `icm_select_engine()` dispatch
  decision), not the adjacent one (`bench_grid crossover`'s empirical
  winner).** Caught this exact substitution TWICE this session — once
  when first checking node E, and again right after the schoolbook fix,
  where "the crossover sweep still looks clean" would have been a false
  green light. This is the single most important lesson from this
  session: **a fix that changes `select_best_B`'s behavior (confirmed
  via a model-internals diagnostic) is not the same as a fix that closes
  the dispatch gap** — always re-run the actual production entry point.
- **Comparing the model against a disjoint, real, held-out measurement
  set** (`sample_plans_*.csv` — real `(n,k,B,total_ms)` from actually
  running the hybrid engine, never used to fit anything in this new
  pinned-constants regime) instead of just re-checking each isolated
  microbenchmark's own internal consistency. This is what surfaced that
  the composed formula didn't match reality even though every individual
  constant was correctly, directly measured — Heiser's "confusing
  calibration with evaluation" crime, caught by having actual evaluation
  data on hand.
- **Real-execution per-level/per-phase instrumentation**
  (`tools/probe_tree_levels.c`, `tools/probe_leaf_extract.c`) as a
  distinct diagnostic step from isolated microbenchmarks — this is what
  let the schoolbook bug be pinpointed to a specific bucket (schoolbook
  levels, worst at small cps) rather than a vague "something in hybrid
  is off."
- **Reviewing a DeepSeek worker's diff line-by-line before merging,
  even when the worker reports success and tests pass.** Caught two real
  bugs this way that `bench_grid verify` alone did NOT catch:
  1. A `-1.0` sentinel-comparison bug (forced schoolbook at huge sizes,
     causing a 60+-second `bench_grid verify` that looked like a hang).
  2. **Two independently-dispatched workers (M3 Pro and Zen4) writing
     the same lookup-table format with DIFFERENT unit conventions** —
     M3 Pro's `schoolbook_corr_ns[]` was a per-unit rate (raw time
     divided by cps²), Zen4's was a raw total call time. Since
     `src/icm.c`'s formula is shared across devices and assumes ONE
     convention, merging Zen4's data as-is would have silently inflated
     its correlate cost by ~cps². Caught by manually inspecting both
     tools' source before merging, converted Zen4's data from the raw
     CSV to match M3 Pro's rate convention before committing.
  **Lesson for any future parallel-worker dispatch of "build the same
  kind of tool" tasks: independently-written tools that produce
  same-shaped data can still disagree on units/semantics — check this
  explicitly, don't assume shape-compatibility means semantic
  compatibility.**
- **Cutting scope aggressively when pushed to.** The user explicitly
  called out a drift toward speculative, low-value parallel tasks (a
  proposed 6-way investigation fan-out) and asked for "the correct
  direction we won't have to second-guess." Re-deriving the decisive
  test from data already in hand (restricting the hybrid ratio
  comparison to B=8 specifically) cut that down to 2 tasks and pointed
  straight at the tree cost formula — this is the process the user
  wants repeated: before fanning out, check whether a cheap analysis of
  existing data already tells you where to look.
- **Consulting real architecture/hardware sources (WebSearch) when
  hitting an unexplained, platform-specific inconsistency**, per this
  session's explicit final instruction — rather than continuing to
  theorize from first principles alone. Produced a concrete, falsifiable
  hypothesis (QoS/core-heterogeneity confound) instead of another guess.

## What Didn't Work

- **DeepSeek workers repeatedly hitting `max_turns` on multi-part
  tasks.** This happened again this session: the M3 Pro schoolbook task
  hit 60 turns, the Zen4 schoolbook task hit 50, the stock-take task hit
  45. In every case the worker had ALREADY done the real, valuable work
  and just hadn't finished committing/relaying it back — resuming with a
  tight "just commit and relay, skip remaining polish" message, or (for
  the M3 Pro case) the supervisor finishing the last few mechanical steps
  directly, recovered the work without redoing it. **Lesson: for tasks
  this size, either budget MORE turns up front (60-70), or split the
  "do the work" step from the "commit + relay + verify" step into two
  dispatches.**
- **Trusting a worker's own "ALL TESTS PASSED" + qualitative
  crossover-sweep read as sufficient evidence of a fix.** Happened twice
  this session (see "What Worked" above) — a real, working, correctly-
  wired fix (schoolbook) still left the actual acceptance criterion
  (dispatch crossover point) unmet. Neither the M3 Pro nor the Zen4
  worker was asked to check `icm_select_engine()`'s literal output
  against the ground truth — they were asked to, but a supervisor-level
  re-verification after merge is what actually caught it. **Always
  re-run the exact acceptance test yourself after merging, don't take a
  worker's own report of it at face value even when it did try.**
- **A 30-second impatience threshold nearly caused a false "this is
  hung" diagnosis.** After the schoolbook fix, `bench_grid verify` was
  killed at 30s looking stuck, which led to ~20 minutes of unnecessary
  isolated-repro debugging (n=1024 hybrid/tree engine checks, all fast)
  before realizing the real run just took 61 seconds total and stdout
  was fully-buffered (not line-buffered) when redirected to a file,
  making the visible tail look frozen. **Lesson: when redirecting a
  verify/benchmark run's stdout to a file for live-tailing, remember
  full buffering can make it look stalled long after real progress has
  been made — sample the actual process (macOS `sample <pid>`) before
  assuming a hang, rather than killing and re-diagnosing from scratch.**
- **Assuming a real bug was "the same everywhere."** Leaf-extraction's
  bias direction is OPPOSITE between Zen4 (underpredicted) and M3 Pro
  (overpredicted) — a naive "leaf needs the same kind of fix as
  schoolbook, roll it out to both platforms the same way" would have
  been wrong. The stock-take worker's explicit verdict — don't fix leaf
  next, it's currently a partial compensating error on M3 Pro — is the
  kind of finding that requires actually decomposing per-platform,
  per-term contributions before touching code, not just extrapolating
  from one platform's result.

## Guiding principles the user has given explicitly this/prior sessions

- **Delegate aggressively to DeepSeek via the `deck` CLI /
  `supervisor-dag` skill; minimize the supervisor's own token spend.**
  Read `deck result` (compact), not `deck log` (full transcript), except
  when debugging a failure. This session additionally used `Monitor` to
  watch for worker state transitions instead of manual polling loops —
  keep doing that.
- **Actually review diffs before merging — don't trust a worker's
  self-report.** Reinforced repeatedly, including two NEW concrete
  catches this session (sentinel bug, units-mismatch bug) that
  `bench_grid verify` alone did not surface.
- **When something is found broken or non-obvious in one spot, think
  about wider ripples across the whole project before proposing a local
  fix** — check whether the project already has an established pattern
  for this class of problem elsewhere. This is how the schoolbook fix's
  design (reuse `calib_sizes[]`'s existing indexing rather than invent a
  new discretization) was derived. Durable memory:
  `feedback_full_picture_ripples.md`.
- **Prefer direct, isolated measurement over indirect aggregate
  regression** — but this session's big lesson REFINES this principle:
  isolated measurement alone is not sufficient either. **The real
  standard is: directly measure each constant AND validate the composed
  model against real, disjoint, held-out end-to-end data** before
  trusting it. Calibration and evaluation must be different datasets
  (Heiser). This is now the central methodology, not "isolated
  measurement" alone.
- **Don't accept "empirically found, no explanation" as a final
  answer** — dig for the actual mechanism. This directly drove the final
  pivot to hardware/architecture research this session, per explicit
  user request, when the platform-inconsistent bias defied a clean
  algorithmic explanation.
- **Investigate anomalies directly; don't re-run hoping they go away.**
- **Cut scope to the decisive experiment; don't fan out into speculative
  parallel investigation "just to be thorough."** Explicit user
  correction this session after a proposed 6-way task fan-out — check
  what existing data already tells you before spawning more workers.
- **When told to "take stock" or consult DeepSeek/research before
  proceeding, actually pause and do that — don't fold it into "one more
  quick fix attempt."** This session, a dedicated stock-take worker
  produced a genuinely new, load-bearing finding (leaf's platform-
  inconsistent direction) that a "just try fixing leaf too" approach
  would have missed or gotten backwards.
- **Professional open-source packaging is a real, first-class goal** —
  public-facing docs must accurately describe the current architecture.
  `tools/calibrate_full.sh`/`tools/fit_cost_model.py` still do NOT
  include the schoolbook calibration step (flagged, not yet done) —
  this is a real onboarding-story gap until fixed.
- **Never terminate a compute instance without explicit go-ahead.** No
  `Co-Authored-By` trailers on any commit, ever.
- **Conservative with GPU credits** — write/design code locally first,
  only spin up a paid instance when ready to be productive with rented
  time, and ask before starting one.

## Next Steps

This DAG's orchestration should continue in this order:

1. ~~Test the QoS-pinning hypothesis~~ **DONE this session — REFUTED.**
   Added `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`
   to `tools/bench_leaf_fma.c` and `tools/probe_leaf_extract.c` (merged
   from worktree `ea5323` first, then edited directly on
   `results-gpu-section`). Results on M3 Pro:
   - Isolated `bench_leaf_fma`: pinned vs unpinned LEAF_FMA_NS slope
     ~0.248-0.250 ns either way (3 runs each) — **no difference**.
   - Embedded `probe_leaf_extract`: leaf geo_mean(meas/pred) = 0.472
     (pinned) vs 0.482 (unpinned) — statistically the same, both match
     the original un-instrumented 0.484 finding closely.
   - Also spot-checked the thermal-throttling fallback hypothesis
     (varying `n_blocks` 500→12000 and `n_reps` 3→7): LEAF_FMA_NS
     converges to ~0.25 with more blocks/reps, no monotonic drift
     upward — **no thermal signature either.**
   - **Conclusion: the P/E-core scheduling confound (mechanism 2) is
     RULED OUT for the leaf anomaly.** Neither QoS pinning nor duration
     changes the isolated-vs-embedded gap. This leaves mechanism 1
     (isolated microbenchmarks looping the same dependency chain
     thousands of times cannot reproduce the latency-hiding available
     when the real engine interleaves leaf-divide with adjacent
     block-build/tree work via out-of-order execution) as the leading
     explanation — a calibration-vs-evaluation gap, not a measurement
     bug, and not something a "measure harder" fix resolves. This is
     consistent with `DISPATCH_GAP_ANALYSIS.md`'s existing verdict: do
     not chase leaf further, it isn't gating the crossover anyway.
   - QoS pinning is still worth keeping in both tools going forward
     (harmless, removes one confound class for any FUTURE M3 Pro
     microbenchmark work) — kept in the diff, not reverted.
2. **Now the clear next priority: investigate the tree's FFT-path formula
   underprediction (15.8%, ~75% of the remaining gap).** This needs a
   NEW per-level probe sweep specifically covering the n/k range near
   the real crossover where FFT-*uncached* levels actually appear —
   `tools/probe_tree_levels.c`'s existing sweep found ZERO FFT-uncached
   rows on both platforms, so this bucket has never actually been
   measured. Widen the (n,k,B) sweep in a new tool or an extended
   version of that one.
3. **Do NOT declare this fixed until `icm_select_engine()`'s own
   dispatch decision** (not `bench_grid crossover`'s empirical winner,
   not an aggregate ratio improving) **matches k≈120-160 on M3 Pro and
   k≈260-270 on Zen4.** This has been the actual, hard-won acceptance
   criterion twice now — don't relax it a third time.
4. ~~Merge worktree `ea5323`'s stock-take commit~~ **DONE this session.**
   Reviewed (`probe_leaf_extract.c` compiled clean, no gitignore/units
   issues found) and merged into `results-gpu-section` via
   `git merge deck/ea5323`.
5. **Wire the schoolbook calibration step into
   `tools/calibrate_full.sh`/`tools/fit_cost_model.py`** — flagged as
   not-done in the `8012244` commit message, needed before the
   one-command device-porting onboarding story is true again.
6. **The Zen4 box (`185.8.107.239`) still has an ad-hoc, freshly-`git
   init`'d local repo at `/root/icm`, unrelated to the real GitHub
   history.** Keep using the established pattern (scp files back into a
   local worktree, commit there, supervisor reviews and merges) rather
   than trying to `git fetch`/`git pull` directly from the box.
7. Several DeepSeek workers from this exact DAG folder (`1f7d2b`) are
   still alive in `awaiting_input`/`error` state and could in principle
   be resumed with `deck send <id> "..."` instead of respawning fresh,
   if continuing their exact prior context is useful: `ef71ca` (C1,
   done), `a016dd` (D1, done), `aa4225` (Zen4 probe, done), `3a522f`
   (Zen4 schoolbook data, done). One worker, `6649ee` ("A2-leaf-fix"),
   exists in this folder with no context captured in this handoff or
   in memory — check `deck result 6649ee` before assuming it's stale or
   relevant.
8. Once dispatch is verified correct on both platforms: **G1A/G1B**
   (regenerate all result data invalidated by the broken-cost-model
   window — see the older `SPRINT_MICROBENCH_MIGRATION_DAG.md` for the
   full file list) and **G3** (`OPTIMIZATION_GUIDE.md` audit — still not
   done, was queued to run in parallel with D1/D2 but got superseded by
   this deeper investigation).
9. After all of the above: resume the **still-open earlier-sprint
   items** — paper Table 1/2 shared-k-column rework, README/RESULTS.md
   sync with final numbers (blocked on trustworthy dispatch, since those
   numbers depend on it).
10. **F** (GPU kernel microbenchmark, B200) — independent of all the
    above, still queued, needs explicit user go-ahead to spin up a paid
    instance.
11. Eventually: decide with the user whether to merge PR #7.
