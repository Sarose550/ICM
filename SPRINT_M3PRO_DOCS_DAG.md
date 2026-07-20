# SPRINT_M3PRO_DOCS_DAG

<!--
Live strike board managed by the supervisor-dag skill.
Ephemeral — DELETE at R_CLOSE. Not design or decisions law.
Rewritten in full 2026-07-20 after the crossover-validation session
uncovered a real dispatch bug (not just stale calibration) and the scope
was reset around a concrete "definition of done."
-->

## Definition of done (user-confirmed 2026-07-20)

1. Zen4: FFT calib data (calib_sizes/calib_times_ns/wisdom) stays as-is —
   already good. Cost-model constants (FMA_NS, BLOCK_*/LEAF_*/correlate
   ratios) refit and validated. **DONE.**
2. M3 Pro: cost-model constants refit and validated the same way.
   FFT calib data itself (calib_sizes/calib_times_ns/wisdom) is still
   reused M3 Max data — a real FFTW PATIENT `./calibrate` run is deferred
   to overnight, user's call. Performance-table numbers in the paper draft
   stay as placeholders until that overnight run lands regardless.
3. **New hard requirement:** anyone cloning the repo needs a single,
   documented command to generate their device's full calibrated
   parameter set (FFT wisdom + cost-model constants), not a manual
   multi-step process. This didn't exist before this session — the fit
   tooling (`fit_cost_model.py`) existed but wasn't wired to anything and
   had a real bug (see W7).
4. README/CLAUDE.md/RESULTS.md/OPTIMIZATION_GUIDE.md performance tables:
   Zen4 numbers can be real now; M3 Pro numbers stay stubbed until the
   overnight PATIENT run.
5. Deslopify pass — still pending, own wave, after everything above lands.
6. Nothing commits until the user explicitly asks.

## Resolved this session

### [x] W6_DISPATCH_BUG_ROOT_CAUSE — real cost-model bug, not calibration staleness

`select_engine()` was dispatching to hybrid across the *entire* measured
crossover grid (n=512..8192, k=40..150) on **both** M3 Pro and Zen4, when
linear actually won almost every cell. Root cause: `FMA_NS` (and five other
tree-side constants: `BLOCK_FMA_NS`, `BLOCK_MEM_NS`, `WRAP_FMA_NS`,
`PAIRED_CACHED_CORR_RATIO`/`INDEP_PAIR_RATIO`, `FP64_DIV_NS`,
`LEAF_FMA_NS`, `LEAF_BLOCK_NS`) were device-specific and had never been
re-derived for either the M3 Pro or this new Zen4 instance — they were
inherited from a prior M3 Max box and old Zen4 calibration, silently wrong
by 2-2.7x in places, enough to flip the linear-vs-hybrid decision across the
whole small-(n,k) regime.

### [x] W7_SAMPLE_PLANS_BUG — calibration tool itself was broken

`tools/sample_plans.c` (used to generate the training data for
`fit_cost_model.py`) forced a specific B via `ICM_FORCE_B` and then called
`icm_equity()` — but `icm_equity()` still runs the (broken) `select_engine()`
dispatch internally, so on an affected device it could silently run *linear*
instead of the intended hybrid B, corrupting the calibration sample. Fixed:
`sample_plans.c` now calls `hybrid_ctx_create()`/`run_engine_ctx()` directly,
bypassing dispatch entirely — a calibration tool must be able to force the
engine it's measuring. Deployed to both machines.

Also fixed: `tools/fit_cost_model.py`'s parameter bounds for `R` and `C_div`
were too tight (0.5-2.0 and 0.5-10.0) — Zen4's true fit values pinned both
at the upper bound, meaning the "fit" was clamped, not a real optimum.
Widened to (0.5-5.0) and (0.5-30.0); Zen4 refit dropped from clamped/
unreported error to 6.0% RMS with no more clamped parameters.

### [x] W8_M3PRO_REFIT — constants fit and validated on real M3 Pro hardware

Ran `sample_plans` (200 sampled (n,k,B) plans) + `fit_cost_model.py` on this
M3 Pro. 7.7% RMS log-relative error. Applied to `devices/m3_pro/fft_config.h`
(new directory — FFT calib table/wisdom still copied from `devices/m3_max`
as placeholder, only the 9 cost-model constants are M3-Pro-specific).
Re-ran `bench_grid verify` (ALL TESTS PASSED) and `bench_grid crossover`:
dispatch now agrees with measured winners across the entire k=40-150 sweep
at all n, except a small conservative-linear bias right at the true
crossover boundary (n=4096/8192, k>=120-130, where hybrid wins by a small
margin but the model still narrowly picks linear) — a minor residual, not
a wholesale flip.

### [x] W9_ZEN4_REFIT — constants fit and validated on real Zen4 hardware

Same procedure on the Zen4 box (ssh, supervisor-only — Deck workers are
network-sandboxed). 6.0% RMS log-relative error after the bounds fix.
Applied to `devices/zen4/fft_config.h` (FFT calib table/wisdom untouched
per user instruction — only the 9 cost-model constants replaced).
`bench_grid verify` → ALL TESTS PASSED. `bench_grid crossover`: dispatch
now agrees with linear across the entire measured sweep (no hybrid cells
in this grid range on either the measured or predicted side — consistent).

## Pending (not started, or blocked)

1. **Build the one-command calibration pipeline (the new hard requirement).**
   Wire `./calibrate` (FFT wisdom) → `sample_plans` → `fit_cost_model.py`
   (cost-model constants) into a single script/make target so a fresh
   clone produces a complete `devices/<name>/fft_config.h` in one step.
   Currently these are three separate manual tools a person has to know to
   run in sequence with matching build flags. Document in
   `CLAUDE.md`'s "Porting to a new device" section.
2. **Run the Zen4 serial + parallel 3-sig-fig performance grid** with the
   now-correct dispatch — this was the original point of going to Zen4.
   Unblocked now (constants fixed, verify passes).
3. **M3 Pro: decide when to run the overnight PATIENT `./calibrate`** for
   real FFT wisdom/calib data (currently still M3 Max's). Cost-model
   constants are already M3-Pro-real; only the FFT timing table itself is
   still borrowed. User said this is a later/overnight step, not blocking
   the rest of the session.
4. **README.md performance tables:** fill in real Zen4 numbers from #2.
   M3 Pro stays "recalibrating" stub until #3 lands — same for the paper
   draft's performance sections (explicit user instruction: placeholders
   for any M3 Pro results in the paper for now).
5. **CLAUDE.md:** VkFFT removal (mechanical, deferred from original
   session scope) + M3 Max→M3 Pro device table update (partially
   unblocked — cost-model constants known, FFT calib still pending) +
   OMP_NUM_THREADS=16 correction for Zen4 (already correct in-conversation,
   needs to land in the doc) + document the new one-command calibration
   pipeline from #1.
6. **RESULTS.md:** Zen4 section refresh (blocked on #2). M3 Max→M3 Pro
   section: constants can be updated now, full numbers blocked on #3.
7. **OPTIMIZATION_GUIDE.md:** M3 Max→M3 Pro numbers in "Final Performance"
   (blocked on #3 for FFT/perf numbers; cost-model constants section can
   be updated now).
8. **Commit `devices/*/fftw_wisdom.dat` to git** — user approved earlier.
   Zen4's wisdom is real and stable, could commit now. M3 Pro's is still
   the M3 Max placeholder — hold until #3 replaces it, don't want to commit
   wisdom about to be thrown away.
9. **Research paper LaTeX first draft** — new deliverable mentioned by the
   user this session. M3 Pro numbers are explicit placeholders per user
   instruction; Zen4 numbers can be real once #2 lands.
10. **"Deslopify" pass** — user request, not started, own wave after
    everything above lands cleanly.
11. **Nothing has been git-committed yet.** `devices/m3_pro/` (new),
    `devices/zen4/fft_config.h`, `tools/sample_plans.c`,
    `tools/fit_cost_model.py`, plus the pre-existing README.md/bench.c/
    icm.c changes from the prior session are all uncommitted. Do NOT
    commit until the user explicitly asks.
12. **Zen4 box housekeeping:** minor — scratch/debug binaries and sources
    from the earlier subset-bug investigation, plus now
    `sample_plans`/`sample_plans_zen4.csv`/`.log`, sitting in `/root/icm/`
    on the Zen4 box. Harmless, worth a cleanup pass once validation work
    there is fully done.

## R_CLOSE (not yet — do not delete this board until all of the above is
resolved or explicitly deprioritized by the user)


## Wave W1 — active (deepseek, dispatched 2026-07-20)

- **Deck folders:** `030c6f` (icm-m3pro-docs-w1, workspace `/Users/samrosenstrauch/Documents/ICM`), `6f2d48` (icm-paper-draft-w1, workspace `/Users/samrosenstrauch/Documents/ICM_paper`)
- **Deny lock:** written at `/Users/samrosenstrauch/Documents/ICM/.claude/.dag-active-lock.json`, covers `tools/fit_cost_model.py`, `tools/calibrate_full.sh`, `Makefile`, `CLAUDE.md`, and the two paper files.

### [x] CALIB_PIPELINE

- **Model:** `deepseek`
- **Worker id:** `4851ba`
- **Depends:** W8_M3PRO_REFIT, W9_ZEN4_REFIT (both done — real fitted constants exist to protect)
- **Allowed files:** `tools/fit_cost_model.py`, `tools/calibrate_full.sh` (new), `Makefile`, `CLAUDE.md` (Porting section only)
- **Exit criteria:** one-command calibration pipeline exists and is documented; `devices/zen4/fft_config.h` and `devices/m3_pro/fft_config.h` untouched (git diff empty)
- **Kill deadline:** 45 min

### [x] PAPER_DRAFT

- **Model:** `deepseek`
- **Worker id:** `29c0a8`
- **Depends:** none (independent of pipeline)
- **Allowed files:** `~/Documents/ICM_paper/icm_paper.tex`, `~/Documents/ICM_paper/references.bib`
- **Exit criteria:** coherent first-draft paper, M3 Pro (and unfinalized Zen4) numbers as explicit placeholders only
- **Kill deadline:** 45 min



## Wave W1 result

- **CALIB_PIPELINE**: clean finish (28 turns). `tools/calibrate_full.sh` created
  (one command: calibrate -> sample_plans -> fit_cost_model.py --write ->
  rebuild -> verify+crossover). `fit_cost_model.py --write` added, validated
  against a scratch copy only. `devices/zen4/fft_config.h` diff confirmed to
  be exactly this session's own earlier manual edit (untouched by the worker);
  `devices/m3_pro/fft_config.h` untouched. CLAUDE.md Porting section updated.
- **PAPER_DRAFT**: hit the 50-turn cap once (matches the pattern noted in
  HANDOFF.md from the prior session), resumed via `deck send`, finished
  cleanly on the second pass (57 turns total). M3 Pro numbers in all 3 tables
  now explicit `[PLACEHOLDER]` (was `[UPDATE]`), with a new explanatory
  paragraph. Zen4 numbers left as-is (existing real `bench_grid` data) — 
  **flagged below as needing a recheck** once the corrected-dispatch Zen4
  grid (running now) lands, since the existing paper numbers may predate
  this session's dispatch bug fix. GPU/B200 numbers untouched. Two bogus
  `references.bib` entries removed, one placeholder citation replaced with
  a real one. `pdflatex` wasn't available in the worker's sandbox to verify
  compilation — brace/environment balance was checked programmatically only.

**New pending item:** verify whether the paper's existing Zen4 performance
numbers predate this session's dispatch-bug fix (if so, they reflect the
broken always-hybrid dispatch and need regenerating from the in-progress
corrected-dispatch grid run, not just left alone).


## Wave W4 — SUBSET_SPEED_BENCH (user question: "did we ever verify subset improves speed")

Answer at time of question: **no** — only correctness (bit-exact) had ever
been verified; speed was never measured anywhere in the codebase.

Dispatched a deepseek node to add `./bench_grid subset-speed` (n x
n_targets/n sweep, subset vs full `icm_equity`, median-of-5). It built
correctly and ran, BUT used the Makefile's default `DEVICE=m3_max` (stale,
pre-fix constants) instead of `DEVICE=m3_pro` — a real methodology bug in
its validation step (the task string said "check what DEVICE is expected"
but didn't force it, and the worker didn't catch the ambiguity). Its
reported numbers (up to 1.85x speedup at n=1024/1% targets) were run under
the WRONG device's dispatch table.

**Supervisor reran locally with `DEVICE=m3_pro`** (correct, session-validated
constants). Corrected numbers are meaningfully different:

| n | speedup @1% | speedup @5-10% | speedup @25%+ |
|---|---|---|---|
| 1024 | 0.92x (net loss) | 0.91-0.92x (net loss) | 0.91x (net loss) |
| 4096 | 1.39x | 1.09-1.12x | 1.05-1.07x |
| 16384 | 1.27x | 1.06-1.08x | 1.03-1.05x |
| 65536 | 1.16x | 1.04-1.05x | 1.02-1.03x |

**Real conclusion**: pruning helps only for n>=4096 and only at low target
fractions (<=5-10%), topping out ~1.2-1.4x, fading to breakeven by 25-50%
targets. At n=1024 it's a net loss at every tested ratio — pruning
bookkeeping overhead exceeds savings on a small tree. This contradicts the
(wrong-device) worker's more optimistic numbers and is meaningfully weaker
than the feature's implicit premise. `bench/bench.c`'s new `subset-speed`
subcommand itself is fine and correctly built — only the worker's own
validation run used the wrong device.

**New pending item:** decide whether this weaker-than-assumed speedup profile
(no benefit below n=4096, net loss at n=1024) changes how `icm_equity_subset`
should be documented/recommended, and whether it's worth investigating why
n=1024 regresses (pruning overhead) rather than just noting it.


## Documentation policy (user-confirmed 2026-07-20)

README.md = algorithm fundamentals only (what it computes, why correct, the
core method at a conceptual level). Implementation-optimization rigor
(tradeoff/overhead analysis, benchmarked speedup profiles for specific
engineering choices like subset-pruning's hot/cold bitmask) goes in the
LaTeX paper draft, NOT the README. Saved as durable memory
(feedback_docs_split_readme_vs_paper). Applies going forward to all doc waves.

**Action item for a future PAPER wave:** add the subset-speed finding
(W4/SUBSET_SPEED_BENCH above) as a talking point in the paper — real numbers,
the two overhead sources (O(n) mask/hotmask setup + branchy pruned-path
execution), not the README.


## Wave W5 result

- **PAPER_SUBSET_TALKING_POINT**: clean finish (21 turns). Added
  `\subsection{Target-Locality Pruning: Subset Equity Performance}` to
  icm_paper.tex Section 6, with the real (n=1024 net-loss included) subset
  speedup table and the two-overhead-source mechanism explanation. Nothing
  else touched.
- **DESLOPIFY**: hit the 50-turn cap once (same pattern as before), resumed,
  finished at 58 turns. Conservative pass — only 5 total changes: 2 stale
  comment fixes in src/icm.c, removal of one confirmed-dead 52-line function
  (`polymul_fft_modk`, verified zero references repo-wide, independently
  re-confirmed by supervisor), 2 device-specific-comment genericizations in
  cost_model.h. Explicitly left device fft_config.h files and most
  build-instruction comments alone (correctly judged as useful docs, not
  slop). Supervisor independently rebuilt (zero warnings) and re-ran
  `bench_grid verify` (ALL TESTS PASSED) after the final handoff — confirmed
  before trusting the worker's own report.

## Remaining pending (updated)

- M3 Pro overnight PATIENT `./calibrate` — user's call, deferred, not blocking.
- Commit `devices/*/fftw_wisdom.dat` and everything else to git — still
  requires explicit user ask per standing rule; nothing committed this
  session.
- Paper: M3 Pro / not-yet-finalized-Zen4 placeholders remain (correct,
  waiting on the overnight calibration); everything else in the paper is
  now at a genuine first-draft state including the new subset-speed section.
- All originally-scoped doc work (README/CLAUDE.md/RESULTS.md/
  OPTIMIZATION_GUIDE.md), the one-command calibration pipeline, the Zen4
  perf grid, Zen4 box housekeeping, and the deslopify pass are DONE.
