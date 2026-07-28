# Sprint: Repo Polish + Calibration-Boundary Engineering — 2026-07-28

Ephemeral — DELETE at R_CLOSE. Not design/decisions law.
Branch: `results-gpu-section` (same PR).
Binding inputs: `scratch/repo_polish_plan_20260728.md`,
`scratch/tier0_and_size_ceiling_design_20260728.md` (both as amended in
conversation — see per-node briefs for the final, locked version of each
decision).

## Goal

Ship this repo publicly in genuinely portfolio-ready shape: clean,
symmetric directory structure, no stale/diverged duplicate code, no dead
files, no compiler warnings, honest and current docs, a graceful
calibration-boundary story (no silent catastrophic dispatch), paper
numbers in sync.

## Lanes

| Lane | Nodes | Risk |
|---|---|---|
| A: Structure | S1 | Foundational, must land first, build-system risk |
| B: Mechanical cleanup | S2, S3, S4(audit) | Low, DeepSeek-delegable |
| C: Calibration engineering | L1 | Core dispatch logic, high review bar |
| D: Docs/paper | D1, D2, D3 | Low-medium, some judgment |
| E: Final | F1 | Verification + close-out |

## Conflicts

- `RESULTS.md`: S2 (merge zen4_qa_report) and the IP redaction — redaction
  done directly by supervisor before Wave 1 dispatch to avoid a same-file
  collision.
- `CLAUDE.md`: no node touches it except D3 (final synthesis) — everyone
  else's structural changes get reflected there once, at the end, not
  piecemeal.
- Tip commit: supervisor only.
- GPU hardware (B200 rental): none of this DAG's nodes require it. The
  optional `gpu_plan.cu` split (post-audit) is explicitly NOT auto-executed
  — flagged for a separate go-ahead if the audit finds it worthwhile,
  since verifying a GPU-code change needs a rental.

---

### [x] PRE — RESULTS.md IP redaction

**Model:** supervisor. Trivial one-line fix, done before Wave 1 to avoid
a same-file conflict with S2. `84.32.71.35` → "Zen4 reference machine".

---

### [x] S1 — src/cpu/ + src/gpu/ symmetric restructure

**DONE 2026-07-28, commit `acd2205`.** `make clean && make` (serial,
m3_pro) succeeds; `./bench_grid verify` → ALL TESTS PASSED; `DEVICE=zen4`
path-checked (fails only on `-march=znver4` under Apple clang, which is
expected on this host — the `-Isrc/cpu -Idevices/zen4` paths are correct).
Dead codegen block removed from `tools/calibrate.c`; it recompiles clean
under `-Wall`.

> **Incident found and recovered during S1**: the working tree contained a
> silent partial revert of the three most recent commits (`e315d9a`,
> `7cf4c5f`, `b8c4b1e`) across 10 files — `README.md`,
> `OPTIMIZATION_GUIDE.md`, `src/gpu/gpu_plan.cu`, `Makefile`,
> `scripts/DIAGNOSTIC_REPORT.md`, and 5 `tools/` files were each
> byte-identical to their pre-commit versions. Verified by comparing each
> file against every recent commit's blob, then restored with
> `git checkout HEAD --` and the S1 path edits re-applied on top. Cause
> unknown (a bare `git checkout --`/`git stash` from an earlier worker is
> the likely candidate). **Nothing was lost.** Re-check `git status` for
> unexpected reverts before trusting the working tree again.

**Original brief:**

**Model:** supervisor (build-system correctness risk, foundational —
everything else's paths depend on this landing cleanly first).
**Depends:** none. **Allowed files:** `src/**`, `Makefile`, anything
`#include`-ing moved headers.
**Work:**
- Create `src/cpu/`: move `icm.c`, `icm.h`, `cost_model.h`,
  `fft_cost_model.h`, `linear_batched_impl.inc`.
- `src/gpu/`: move `icm_gpu.h` in alongside the existing split files.
- Delete `src/icm_gpu.cu` (stale monolith, confirmed superseded, none of
  this session's fixes touched it).
- Delete `tools/calibrate.c`'s dead codegen block that duplicates
  `best_fft_config()`/`best_fft_config_joint()` (confirmed: current
  `devices/*/fft_config.h` no longer contain these, centralized in
  `src/cpu/fft_cost_model.h`).
- Fix every `#include` reference and Makefile `-I` path.
**Exit criteria:** `make clean && make` (serial, m3_pro) succeeds,
`./bench_grid verify` passes ALL TESTS. `make clean && make DEVICE=zen4`
path-checked (full hardware verify only if Zen4 is reachable this
session — not required to land).

---

### [x] S2 — scripts/ elimination

**DONE 2026-07-28.** `scripts/` no longer exists. `gpu_dispatch_validate.cu`,
`threshold_search_gpu.cu`, `b200_verify_and_sweep.sh` → `tools/`;
`DIAGNOSTIC_REPORT.md`, `frontier_probe.cu` → `scratch/`. `gpu_ws_repro.cu`
→ `tools/` with an explanatory header rather than folded into
`bench/bench_gpu.cu`: the repro uses the plan API
(`icm_gpu_plan_create`) while `bench_gpu.cu` is entirely equity-API
(`icm_gpu_equity`) + CPU cross-check, so integration was not the trivial
case the brief allowed for, and there is no local CUDA toolchain to verify
it. Correct call. One durable finding merged into `RESULTS.md` from
`zen4_qa_report_20260727.md` (parallel B-selection divergence: current
reference box picks B=32 where the prior box picked B=64 at k=10000-26000;
the 1944-point `bselect_*` table was carried over unchanged and has not been
re-verified against this box's bandwidth profile — flagged as open). Report
file deleted.

**Original brief:**

**Model:** deepseek. **Depends:** S1 (stable paths first).
**Allowed files:** `scripts/**`, `tools/**` (new files only), `RESULTS.md`
(append zen4_qa_report findings only, no other edits), `bench/*.cu` (to
wire `gpu_ws_repro.cu` in as a named regression check, if a clean
integration point exists — otherwise leave it as a standalone tool in
`tools/` and say so, don't force it).
**Forbidden:** `git apply` (bare), `git checkout --`, `git stash`,
`git reset`, `git restore` on anything outside this node's own edits.
**Work:** move `gpu_dispatch_validate.cu`, `threshold_search_gpu.cu`,
`b200_verify_and_sweep.sh` → `tools/`. Fold `gpu_ws_repro.cu` into the
verify suite or leave in `tools/` with a clear comment on its regression-
check purpose. Move `DIAGNOSTIC_REPORT.md`, `gpu_below_sat_fix_plan.md`,
`gpu_ragged_tree_fix_plan.md`, `gpu_ragged_tree_fix_premortem.md`,
`ragged_tree_cufftdx_research.md`, `analyze_fftsize_bug_blast_radius.py`,
`monotonicity_final_verdict.py`, `frontier_probe.cu`,
`remeasure_nonmono.cu`, `l_skeleton_large_n.csv`, `b200_session_20260726/`
→ `scratch/` (gitignored). Merge any real findings from
`zen4_qa_report_20260727.md` into `RESULTS.md`, then delete both the
report and the (now-empty) `scripts/` directory.
**Exit criteria:** `scripts/` no longer exists. `git status` clean of
anything unexpected. Report back which files went where.
**Kill deadline:** 25 turns.

---

### [x] S3 — Doc consolidation

**DONE 2026-07-28** (two passes; the first hit its turn cap mid-report, the
work itself was complete). `DISPATCH_GAP_ANALYSIS.md` deleted.
`COST_MODEL_EXPLAINED.md`: the wrap-correction fix is now stated as landed
and hardware-verified with the real measured numbers; the "3-4 orders of
magnitude" caveat and its two downstream dependents are gone.

Supervisor caught one thing on review: the first pass left resolved items in
the "Open methodological gaps" list as `~~strikethrough~~ DONE` — the exact
changelog/diary pattern this project treats as a trap. Sent back; items 1/6/7
deleted outright and the list renumbered 1-4. Verified independently: zero
`~~` remaining, section renumbered contiguously.

Also verified independently rather than taking the worker's word: the claim
"all five inversions resolved" checks out against
`results/gpu_heatmap_b200_20260728.csv` — every k=128 vs k=256 pair from
n=65,536 to n=8,388,608 is now correctly ordered, and a full k-monotonicity
scan over all 211 cells finds exactly one remaining 3% dip
(n=4,194,304, k=2048→4096).

**Original brief:**

**Model:** deepseek. **Depends:** none (independent of S1/S2).
**Allowed files:** `COST_MODEL_EXPLAINED.md`, `DISPATCH_GAP_ANALYSIS.md`.
**Work:** Fix `COST_MODEL_EXPLAINED.md`'s staleness — the wrap-correction
fix is hardware-verified (2026-07-26), not "awaiting verification"; the
"3-4 orders of magnitude" caveat was refuted the same session and must be
removed, not repeated. Delete `DISPATCH_GAP_ANALYSIS.md` entirely (stale,
unexpanded `$(date)`, references a nonexistent tool, superseded by the
empirical crossover table).
**Exit criteria:** `COST_MODEL_EXPLAINED.md` contains no claims that
contradict RESULTS.md's current state. `DISPATCH_GAP_ANALYSIS.md` gone.
**Kill deadline:** 15 turns.

---

### [x] S4 — gpu_plan.cu reorganization audit (propose only)

**DONE 2026-07-28.** Deliverable: `scratch/gpu_plan_split_audit_20260728.md`
(537 lines) — 49-function inventory with line ranges, coupling graph naming
the actual `static` symbols, three ranked split options plus a don't-split
option, and a verification plan.

**Verdict: do not split.** Best split option (A: extract the ~800-line /
23-function cost model into `gpu_cost_model.cu`; all symbols already
declared in `gpu_internal.h`, only 4 statics need exporting) is mechanically
clean but verifiable only via a B200 rental, and the failure mode is silent:
a differently-resolved `#ifdef` or duplicated constant changes planner
decisions without changing correctness, so `bench_gpu_fused verify` would
still pass 36/0 and you would only catch it by diffing
`ICM_GPU_DEBUG_PLAN=1` output against a pre-split baseline. Cost/benefit
does not justify it for a single-GPU-architecture project. **No further
action; no go-ahead needed.** Confirmed the node touched zero tracked files.

**Original brief:**

**Model:** deepseek. **Depends:** S1 (path stability).
**Allowed files:** read-only over `src/gpu/**`. Write-allowed: ONLY a new
file, `scratch/gpu_plan_split_audit_20260728.md`.
**Context:** `gpu_plan.cu` is 1,947 lines / 43 functions, the largest and
densest file in the GPU split, and the one that absorbed this session's
register-pressure fix, B-selection tables, cost-model comparisons, memory
strategy, and device sorting without a holistic reorganization pass.
**Exit criteria:** a concrete, prioritized proposal for splitting it by
concern (e.g. cost-model/dispatch logic vs. memory/plan-construction vs.
device sorting) — or a reasoned case that it shouldn't be split. Do NOT
execute any split. This is a proposal for supervisor+user review; actual
execution (if warranted) needs a GPU rental to verify and is a separate,
explicitly-gated decision, not part of this node.
**Kill deadline:** 30 turns.

---

### [x] L1 — Calibration-boundary implementation

**DONE 2026-07-28, commit `f2ad24e`.** All 7 spec points landed. m3_pro:
`./bench_grid verify` ALL TESTS PASSED, `./bench_grid crossover` dispatch
pattern unchanged (L→H at k=120-160 at every n, matching the calibrated
table). `libicm.a`, `libicm.dylib`, `bench_grid` all build with **zero**
warnings. New `tools/test_uncalibrated_fallback.c`: 6/6 pass.

The DeepSeek node hit its turn cap twice and never produced a final report,
so everything below was found by supervisor review rather than disclosed.
**Four real defects, three of them shipping-blockers:**

1. **Crossover fallback inverted (correctness).** The empty-table guard
   returned `1e9`, with a comment reasoning that a large crossover forces
   hybrid. `select_engine_ex()` reads it as `k < k_cross ? linear : hybrid`,
   so `1e9` makes an uncalibrated device **always linear** — O(nk), the exact
   catastrophic-slowness failure mode L1 exists to prevent, just relocated.
   Fixed to `0.0`.
2. **`make libicm.a` did not build at all.** `next_smooth_ge()` calls
   `next_pow2()`, which was defined only under `#if defined(ICM_BENCH_INCLUDE)`.
   `bench_grid` defines that macro, so `./bench_grid verify` passed while the
   shipped static and shared libraries — what the Python bindings link
   against — failed to compile. Un-gated `next_pow2`.
3. **The uncalibrated path returned WRONG ANSWERS.** It picked the smallest
   smooth size ≥ `L/2+1`, i.e. the maximum-wrap-correction case. At
   n=4096,k=4096 the generic build gave `sum-1 = -0.496` at every Q while the
   calibrated build converged to `-4.8e-13` on identical input — not
   quadrature error, genuinely wrong, and silently size-dependent (n=65536
   looked fine). The calibrated search may pick a small size *because it
   prices the wrap correction*; with no cost model there is nothing to weigh.
   Changed to the smallest smooth size ≥ the **full** convolution length, so
   the cyclic convolution equals the linear one exactly and `wrap_m = 0`.
   Generic now matches calibrated digit-for-digit — and n=65536,k=65536,Q=64
   dropped from **49.3s to 5.85s**, since the discarded wrap correction was
   O(m²) with m ≈ L/2.
4. **The delivered test did not compile**, written against three functions
   that do not exist (`ICMContext`, `icm_ctx_create`, `icm_compute`) plus a
   wrong `run_engine_ctx` arity, and its assertions encoded defect 1 as
   expected behaviour. Rewritten by the supervisor against a closed form
   (identical stacks, payouts summing to 1 → every equity exactly 1/n), with
   symmetry and quadrature accuracy checked separately since only the latter
   depends on Q. Attempting to build it is what exposed defect 2.

Also removed a dead `CALIBRATED_MAX_CONV_LEN` the node added to
`devices/b200/gpu_fft_config.h`: nothing reads it, and a constant named
"calibrated max" implies a GPU-side ceiling guard that does not exist.
**Open follow-up: the GPU planner has no equivalent calibration-boundary
guard.** Same class of bug, not in this sprint's scope, needs a B200 to verify.

Verified rather than assumed: the lazily-extended smooth-number table is not
a data race, because contexts are created once and *cloned* per thread
before the `#pragma omp parallel for`, so it is only ever mutated
single-threaded.

**Original brief:**

**Model:** deepseek (implementation), with supervisor doing a full
line-by-line review + local hardware verification before it's trusted —
same treatment as this session's GPU fix candidates.
**Depends:** S1 (files must be at their final `src/cpu/` home first).
**Allowed files:** `src/cpu/fft_cost_model.h`, `src/cpu/icm.c`,
`tools/calibrate.c`, `devices/generic/` (new).
**Binding spec** (locked in conversation, do not deviate without flagging
back): see `scratch/tier0_and_size_ceiling_design_20260728.md` for full
context, but the FINAL locked design (supersedes that doc's own open
questions) is:
1. New explicit `CALIBRATED_MAX_CONV_LEN` constant in `fft_config.h`,
   checked per tree level against that level's actual FFT convolution
   length `L` (not top-level `n`). `-1` means "never calibrated."
2. When `L` exceeds the ceiling (or ceiling is `-1`): skip the
   schoolbook-vs-FFT cost comparison entirely (do not consult
   `calib_times_ns[]`/`schoolbook_mul_ns[]` at all in this branch — they
   may be nonsensical/absent). Always choose FFT. Pick the next viable
   FFT size at or above what's needed (same structural search
   `fastest_fft_ge` already does, just not constrained to sizes with
   calibration data). Plan it with `FFTW_ESTIMATE`. No timing, no
   measurement, no execution at plan-creation time — matches the
   already-existing, already-correct pattern at `src/cpu/icm.c:339-345`
   (`FFTW_MEASURE | FFTW_WISDOM_ONLY` → null-check → `FFTW_ESTIMATE`),
   do not diverge from that established idiom.
3. Crossover table (`empirical_crossover_k()`), uncalibrated (table
   empty/absent): skip the lookup entirely, always dispatch hybrid. Fix
   the real bug found this session — today it reads
   `crossover_n[0]`/`crossover_k[hi]` out of bounds on an empty table
   rather than degrading gracefully; add an explicit empty-table guard.
4. B-selection, uncalibrated: fixed `B=32`. This already exists as
   `empirical_best_B()`'s "sane fallback" — confirm it actually triggers
   correctly when the table is truly empty (not just when a query misses
   a specific point), don't assume it already works end-to-end.
5. `tools/calibrate.c`: `MAX_SIZE` (currently a hardcoded `#define
   131072`) becomes a CLI flag (e.g. `--max-size N`), documented in
   README/CLAUDE.md with the honest cost tradeoff (higher ceiling = longer
   offline calibration, larger fully-optimal range before fallback
   engages).
6. `devices/generic/`: a minimal stub (not a "conservative tuned
   defaults" file — per point 2/3/4 above, the scalar constants are never
   read once both lookups are skipped, so this can be genuinely minimal:
   `CALIBRATED_MAX_CONV_LEN = -1`, empty/absent crossover and B-select
   tables). Used automatically when `DEVICE` doesn't match a real
   `devices/<name>/` directory — this also means removing the current
   silent `DEVICE ?= m3_pro` default in `Makefile` in favor of this
   explicit generic fallback.
7. A clear, impossible-to-miss build-time message when the generic
   fallback is active: what it is, that results are correct but
   unoptimized, and the exact command to get real calibration.
**Explicitly NOT in scope** (locked decision from conversation, do not
reintroduce): no live timing/measurement of any kind in any fallback
path. No live probing of the linear-vs-hybrid crossover. No persisting
measured data back to calibration files (dropped once the design moved to
pure `FFTW_ESTIMATE`, nothing real gets measured to persist).
**Exit criteria:**
- `./bench_grid verify` and `./bench_grid crossover` pass **unchanged**
  on m3_pro (and zen4 if reachable) — this must not alter calibrated-
  device behavior at all.
- A new explicit test/repro exercising the uncalibrated path: build
  against `devices/generic/`, confirm (a) it always dispatches hybrid
  with B=32, (b) a query at a size far beyond any real calibration still
  produces correct results via FFT+`FFTW_ESTIMATE` rather than
  catastrophic schoolbook, (c) no crash on the previously-out-of-bounds
  crossover-table read.
- Zero new compiler warnings.
**Kill deadline:** 40 turns for the DeepSeek implementation pass; the
supervisor review + verification afterward is unbounded (correctness
review, not a turn-budget task).

---

### [ ] D1 — README additions

**Model:** deepseek. **Depends:** L1 (Automatic Dispatch section should
describe the real, landed calibration-boundary behavior, not a stale
description).
**Allowed files:** `README.md`.
**Work:** Automatic Dispatch subsection (3-4 sentences, link to
OPTIMIZATION_GUIDE.md), Platform support matrix (M3 Pro/Zen4/B200 ×
arch/FFT-backend/status), Getting Help / Reporting Issues section,
GitHub-only citation placeholder (no arXiv ID yet — that's a real,
separate follow-up once the paper is submitted). Add `scripts/`... wait,
`scripts/` no longer exists post-S2 — add `results/`, `paper/` to the
Project Structure tree, reflect the new `src/cpu/`/`src/gpu/` split.
**Exit criteria:** README's directory tree matches the real, post-S1/S2
repo exactly.
**Kill deadline:** 15 turns.

---

### [x] D2 — Paper sync

**DONE 2026-07-28.** Paper repo commit `e7a2493`; PDF rebuilt (26pp, 0
undefined refs) and committed into the main repo as `paper/icm_paper.pdf`.

GPU: Table 1 resynced to the regenerated 211-point heatmap (+ the
n=33,554,432 row). Table 2 replaced entirely — the old five
`push_limit_gpu` frontier probes predate the wrap-correction and
B-selection fixes, so they were not merely stale but measuring different
code. It now reports the real binary-search one-second threshold (k=n at
n=1,490,944 / 918.8ms; k=100 at n=7,975,936 / 998.9ms) with the bracketing
candidate above each.

Zen4 — larger than expected, and worth flagging: the paper was still
reporting the *previous* Zen4 box. RESULTS.md's standing reference is the
2026-07-27 redeployment, whose RAM runs at 3600 MT/s vs its 5600 MT/s
rating (AM5 2DPC electrical limit; per RESULTS.md this was an explicit user
decision to accept as permanent reference hardware). Both tables' Zen4
columns resynced. The effect is not a flat factor — linear is unaffected,
hybrid is bandwidth-bound and materially slower at large k — so the setup
section now discloses it, and the linear-to-hybrid cliff moved from the
quoted "k≈300-500" to between k=100 and k=200, with the worked numbers and
the figure caption corrected. One-second single-threaded threshold
17,216 → 17,984 (the old figure was an interpolation between grid points;
17,984 is the first real `bench_grid threshold` binary search on Zen4).

Slop pass: the paper was already clean — 0 em-dashes, no AI-tell phrasing,
3 LaTeX comments all structural. Nothing to strip. Argument structure,
prose voice and section ordering untouched, per the brief.

**Two incidental fixes made along the way:**
- The 2026-07-27 Zen4 contour CSVs were *damaged at capture time*: the
  serial file had lost its header row, the parallel file had 42 stderr
  progress lines interleaved into it (captured with `2>&1`).
  `tools/contour_1s.c` is itself correct — it already routes progress to
  stderr — so this is capture damage, not a tool bug. Repaired losslessly;
  no measurement altered. Without this, `plot_contour.py --device zen4`
  crashes with `KeyError: 'k'`, which is why the Zen4 plots were still
  being built from 07-24 data.
- Regenerated every plot from current data, and fixed three doc references
  to the now-removed `scripts/` directory.

**Original brief:**

**Model:** supervisor (judgment-heavy — "facts + slop only," your prose
stays yours; touches the sibling `~/Documents/ICM_paper` repo).
**Depends:** none (RESULTS.md's GPU numbers already final from the
heatmap regen).
**Work:** sync GPU numbers/claims against `results/gpu_heatmap_b200_20260728.csv`
and `results/gpu_threshold_search_20260728.txt`. Remove AI-tell phrasing,
m-dash tics, LaTeX comment cruft referencing past drafts. Do not touch
argument structure, prose voice, or section ordering. Recompile and
recommit `paper/icm_paper.pdf`.
**Exit criteria:** no numeric or factual claim in the paper contradicts
RESULTS.md. PDF recompiled and committed in the main repo.

---

### [ ] D3 — CLAUDE.md final synthesis

**Model:** supervisor (single synthesis point, depends on everything else
landing first — avoid piecemeal edits from multiple nodes).
**Depends:** S1, S2, S3, S4, L1, D1.
**Work:** update Directory Structure section (new `src/cpu/`/`src/gpu/`
split, `scripts/` removed, `tools/`'s 3 new arrivals), remove the 2
already-confirmed stale tool references, document the new calibration-
boundary behavior and `devices/generic/` fallback, document `MAX_SIZE`'s
new CLI-flag form.

---

### [ ] F1 — Final verification + close-out

**Model:** supervisor.
**Depends:** everything above.
**Work:** `make clean && make` (serial + parallel, m3_pro at minimum) —
zero warnings. `./bench_grid verify` + `crossover` pass. Confirm no
leftover dead code / stray references. Delete both scratch design docs
(`repo_polish_plan_20260728.md`, `tier0_and_size_ceiling_design_20260728.md`)
and this board file. Final commit.
