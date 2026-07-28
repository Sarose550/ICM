# Codebase Cleanliness Audit — 2026-07-27

**Scope:** git-tracked files only (`git ls-files`). Prior cleanup pass
(`SPRINT_CLEANUP_AUDIT_DAG`, closed 2026-07-25/26) already fixed: stale paper
sync, `memory_strategy` doc-comment, `CLAUDE.md` missing `scripts/` dir,
diagnostic-file keep/delete decisions in `scripts/`, and a `RESULTS.md`
read-through. None of those are re-flagged below.

---

## 1. Directory Structure Sanity

### What's good

The repo follows a conventional HPC/scientific-computing layout that a
newcomer would recognize immediately:

```
src/       — implementation (CPU + GPU split)
tools/     — calibration, benchmarking, plotting, porting
devices/   — per-machine calibration data
bench/     — harnesses
results/   — published data + plots
paper/     — compiled paper (icm_paper.pdf)
python/    — ctypes bindings
scripts/   — investigation/orchestration tools (documented as "not part of regular build")
```

This is a clean, navigable split. `src/` and `tools/` are exactly where
a domain scientist would look first. `devices/` is a well-chosen name —
self-documenting that calibration is per-machine, not committed to the
source tree.

### Issues

1. **`scripts/` vs `tools/` boundary is blurred.** `scripts/` is
   documented as "one-off GPU investigation/orchestration tools, not part
   of the regular build." But files like `scripts/gpu_dispatch_validate.cu`
   (the GPU analogue of `./bench_grid crossover`) and
   `scripts/threshold_search_gpu.cu` (the GPU threshold binary search) are
   NOT one-off investigations — they are the standing, production GPU QA
   and benchmarking tools. A newcomer reading CLAUDE.md's directory
   structure will see `scripts/` described as investigative cruft and
   might not look there for the tool they actually need. Meanwhile, the
   real one-off scripts (e.g., `analyze_fftsize_bug_blast_radius.py`,
   `monotonicity_final_verdict.py`) sit alongside these production tools
   with no separation.

   **Recommendation:** Either move the production-quality GPU tools
   (`gpu_dispatch_validate.cu`, `threshold_search_gpu.cu`,
   `verify_below_sat_fix.cu`) into `tools/` where they belong alongside
   their CPU analogues (`validate_best_b.c`, `contour_1s.c`,
   `calibrate_crossover.c`), OR split `scripts/` into `scripts/` (ad-hoc
   investigation artifacts) and a new `gpu_tools/` or equivalent
   directory.

2. **`scripts/b200_session_20260726/` is a provenance dump, not a
   discoverable tool.** It contains raw CSVs (`gap_best_b_b200.csv`,
   `gap2_best_b_b200.csv`) and partial output logs
   (`gpu_dispatch_validate_run1_vacuous.txt`,
   `gpu_dispatch_validate_partial_run2.txt`) from a specific rental
   session. These are useful for provenance but have zero discoverability
   for a newcomer. HANDOFF.md already refers to them; they don't need to
   be at the top of `scripts/`.

   **Recommendation:** Archive under `results/b200_session_20260726/` or a
   similar `results/` subdirectory, or leave as-is if provenance trumps
   discoverability (supervisor call).

3. **`results/` contains both committed reference data and regeneratable
   plots — but the `.gitignore` is correct.** The negations
   (`!results/bench_grid_*.txt`, etc.) properly protect the committed
   reference files while ignoring regeneratable artifacts. No change
   needed.

---

## 2. Top-Level Doc Sprawl

Eight (8) markdown files are git-tracked at the repo root:

| File | Lines | Nature |
|------|-------|--------|
| `README.md` | 299 | Public-facing: what, quickstart, API, perf, build |
| `CLAUDE.md` | 480 | **Gitignored** (not shipped). AI-agent instructions |
| `HANDOFF.md` | 939 | Internal AI-agent session log |
| `RESULTS.md` | 521 | Public-facing: complete performance data |
| `OPTIMIZATION_GUIDE.md` | 633 | Public: optimization history, porting guide, GPU cost model |
| `COST_MODEL_EXPLAINED.md` | 421 | Public(?): "plain English" cost-model explainer |
| `DISPATCH_GAP_ANALYSIS.md` | 95 | Point-in-time investigation report (stale) |
| `SPRINT_GPU_MONOTONICITY_DAG.md` | 1462 | Internal AI-agent sprint board |

### CLAUDE.md

**Status:** Gitignored (`.gitignore` line 138: `# AI agent project instructions -- not for public repo view`). Not tracked by git (`git ls-files` confirms it's absent). This is correct — it's an AI-agent instruction file, not user-facing documentation. No action needed beyond the stale-tool-references flagged in §4 below.

### HANDOFF.md — AI-Agent Session Log, Contains Sensitive Data

**Issue: Should this ship in a public repo?**

Arguments for REMOVING/REDACTING before public release:

1. **Contains vast.ai rental contract IDs.** Nine (9) distinct contract
   numbers (e.g., `45932804`, `45964239`, `45965783`, `45969931`,
   `45953979`, `45957726`) appear in the file. These are third-party
   commercial transaction identifiers. While not secret keys, they tie
   the repo to specific paid cloud rental sessions — irrelevant to the
   library's purpose and arguably none of the public's business.

2. **Refers to a real server IP address by cross-reference.** The IP
   `84.32.71.35` appears 6 times in `SPRINT_GPU_MONOTONICITY_DAG.md` and
   once each in `scripts/zen4_qa_report_20260727.md` and
   `devices/zen4/fft_config.h`. HANDOFF.md doesn't contain the IP
   directly, but its Zen4 section ("Credentials are not in this repo or
   this document — ask the user directly") correctly keeps credentials
   out. However, the DAG it cross-references DOES contain the IP in
   plaintext.

3. **It IS an internal AI-agent session log,** exactly as described. It
   contains: exact turn-by-turn descriptions of what DeepSeek workers
   did, cost accounting ($6.07 → $0.22, ~$1.54 remaining), real mistakes
   made and their fixes, and operational details (exact `rsync` commands,
   `ssh root@...` patterns). None of this is useful to a library user. It
   would be confusing and intimidating — a 939-line narrative of bug
   hunts is not a README.

4. **It documents bugs that were already fixed.** A reader encountering
   HANDOFF.md before the actual documentation might conclude the library
   is unstable. The GPU OOM section, the three separate non-monotonicity
   mechanisms, the ragged-tree regression hunt — all real, all fixed or
   tracked — read as a bug tracker, not a library.

Arguments for KEEPING (trimmed):

- It's the most complete record of what was done and why. The project has
  no other design doc or decision log at this level of detail.
- Some sections (Architecture: what's actually load-bearing, Critical
  operational notes: B200 build recipe, Zen4 porting procedure) contain
  genuinely load-bearing institutional knowledge not duplicated elsewhere.

**Recommendation:** Do NOT ship HANDOFF.md as-is. Options (supervisor+user
decision):
- **Option A (preferred):** Split into two files. (a) A new `docs/` or
  `devices/` file with the load-bearing operational knowledge (B200 build
  recipe, Zen4 porting procedure, architecture dispatch-layers section)
  — keep and ship. (b) Delete the session-log narrative from the repo
  (keep it locally as `HANDOFF.md` in `.gitignore` if useful for AI
  agents, which it already would be since CLAUDE.md is gitignored).
- **Option B:** Redact contract IDs and any IP references, move the file
  to `docs/development-log.md` with a clear header: "Historical
  development log, 2026-07. Kept for provenance; not a user-facing
  document."
- **Option C:** Delete entirely from public repo. The load-bearing
  operational knowledge is already partially duplicated in CLAUDE.md
  (Zen4/procedures) and README.md (B200 build recipe). Not all of it is,
  though — the architecture dispatch-layers section and the
  `calibrate_block_size.py` bug documentation would be lost.

### SPRINT_GPU_MONOTONICITY_DAG.md — Live Sprint Board, Contains IP Address

**Issue: Should this ship at all?**

1. **Contains the project's Zen4 server's real public IP address
   (`84.32.71.35`) in plaintext, 6 times.** This is in the Conflicts
   table ("Zen4 instance (84.32.71.35, real but not paid-per-use)"), in
   the Z-lane node specs, in rsync examples, and in hardware-finding
   notes. This is a live server IP tied to a specific person's rented
   infrastructure. Publishing it in a public GitHub repo is a security
   concern — it invites scanning, brute-force attempts, and ties the
   repo to a specific physical machine.

2. **Contains 9 vast.ai contract IDs** (same ones as HANDOFF.md). Same
   concern as above.

3. **It IS a live sprint board** — the file's own header says "Ephemeral
   — DELETE at R_CLOSE. Not design or decisions law." It is meant to be
   temporary. It tracks in-progress work, unresolved bugs (node K's
   ragged-tree regression), and internal task assignments ("Model:
   deepseek", "Model: supervisor"). It is the definition of "not for
   public consumption."

**Recommendation:** This file should NOT ship publicly. **DELETE before
release** (per its own header instruction). The IP `84.32.71.35` alone is
reason enough. If any content needs to survive (e.g., the confirmed root
causes of the three non-monotonicity mechanisms as design notes), extract
those into a clean design document — do not ship the board.

**Also flagging:** The IP `84.32.71.35` also appears in:
- `scripts/zen4_qa_report_20260727.md` (1 occurrence)
- `devices/zen4/fft_config.h` (1 occurrence, in a comment: "RECALIBRATED
  2026-07-27 (box 84.32.71.35, a fresh redeployment")")

Both should be redacted before public release.

### COST_MODEL_EXPLAINED.md — Stale in Critical Ways, Overlaps with OPTIMIZATION_GUIDE.md

**Issue 1: Contains factually wrong, now-refuted claims.**

Section 3 ("The specific bug found and fixed"):
- Says the wrap-correction fix is "awaiting real-hardware verification."
  Per HANDOFF.md, this fix was **hardware-verified on 2026-07-26** on a
  real B200 — it produced a real, large, correct improvement (890ms →
  511.7ms at the target cell). This section is simply out of date.
- Contains the "3-4 orders of magnitude" caveat claiming the cost model
  predicts only a ~0.6ns–38µs improvement while the real gap is ~280ms.
  This caveat was **independently investigated and refuted** (same
  session): the worker who made this claim was wrong — the calibration
  data is correctly normalized and the mismatch was an approximation
  error in the offline Python simulation scripts, not in the production
  C++ cost model. HANDOFF.md documents this refutation explicitly. Yet
  COST_MODEL_EXPLAINED.md still presents the caveat as fact.

Section 5 ("Open methodological gaps"):
- Gap #1 (verify wrap-correction fix): **DONE.** Hardware-verified.
- Gap #2 (GPU FFT floor measurements): Still open, legitimately tracked.
- Gap #3 (GPU wrap-correction microbenchmark): Still open.
- Gap #4 (GPU engine-selection formula → empirical table): Still open.
- Gap #5 (clean up `fit_gpu_cost_model.py`): Still open.
- Gap #6 (re-measure bad heatmap data point): Partially addressed by the
  fix but the specific re-measurement with reps≥5 may still be needed.
- Gap #7 (run GPU threshold search): Still blocked on the ragged-tree
  regression.

A reader cannot tell which gaps are still open vs resolved without
cross-referencing HANDOFF.md (which itself shouldn't ship as-is).

**Issue 2: Massive overlap with OPTIMIZATION_GUIDE.md.**

Both documents explain:
- The three decision layers (engine, block size, per-level FFT)
- The lookup-table-vs-formula design pattern
- The same external precedents (FFTW ESTIMATE/MEASURE/PATIENT, ATLAS AEOS,
  BeBOP/Sparsity, LAPACK ILAENV)
- The wrap-correction tier-lock-in bug and its mechanism
- The calibration pipeline

COST_MODEL_EXPLAINED.md is written in a "plain English, no code" narrative
style (~421 lines). OPTIMIZATION_GUIDE.md is more technical, with formulas
and architecture diagrams (~633 lines). The overlap is substantial — a
reader who reads both will feel they're reading the same material twice.
A reader who reads only one might miss information that's only in the
other (e.g., OPTIMIZATION_GUIDE.md has the full optimization history and
porting guide; COST_MODEL_EXPLAINED.md has the "why this design pattern"
precedent discussion).

**Recommendation:** Merge the non-overlapping content from
COST_MODEL_EXPLAINED.md into OPTIMIZATION_GUIDE.md (or vice versa),
UPDATE all stale claims to reflect the hardware-verified state, and delete
the duplicate. The "plain English" approach is valuable for a public repo
— a single document that's both accessible AND technically complete is
better than two that disagree with each other. If the plain-English style
is preferred for the whole document, rewrite OPTIMIZATION_GUIDE.md in that
voice and retire COST_MODEL_EXPLAINED.md.

### DISPATCH_GAP_ANALYSIS.md — Stale Point-in-Time Report

**Issue:** This is a 95-line investigation report from a specific session
(worktree `ea5323`, schoolbook-fix commit `8012244`). Its date field is
literally the unexpanded shell variable `$(date)`. It references tools
that no longer exist in the repo (`tools/quantify_dispatch_gap.c` — not
tracked, not present). It analyzes a problem (dispatch gap on M3 Pro) that
was subsequently resolved by the empirical crossover table approach
documented throughout HANDOFF.md.

This file is a snapshot of a concluded investigation. It has no ongoing
relevance to a library user. Its findings were either acted upon (the
schoolbook fix) or superseded (the empirical crossover table).

**Recommendation:** DELETE from public repo, or archive to
`docs/investigation-archive/` if provenance is valued.

### README.md vs CLAUDE.md Build/Test Duplication

**Status:** The split is sensible and should be preserved.

README.md's Build section targets users: "here's how to build on your OS,
here are the device shortcuts, here's GPU." CLAUDE.md's Build section
targets AI agents: "here are ALL the build variants, including
`calibrate_dual`, `bench_grid profile`, GPU calibration, and the exact
compiler flags for every scenario." There IS overlap (both document `make`,
`make parallel`, `make DEVICE=...`, `make bench_gpu_fused CUDA_ARCH=...`),
but it's not harmful — CLAUDE.md is gitignored and doesn't ship, so the
duplication doesn't confuse users. For the AI-agent use case, having the
full build matrix in one place (CLAUDE.md) is correct even if some lines
also appear in README.md.

Same answer for Test commands: README.md shows `./bench_grid verify` as a
quickstart check; CLAUDE.md shows the full test matrix (`quick`, `verify`,
`crossover`, `cliff`, `threshold`, `profile`, `bench`). Appropriate split.

### What a Newcomer Would Read First

Currently, a newcomer who clones the repo sees:
```
README.md       ← obvious starting point
RESULTS.md      ← "performance results"
OPTIMIZATION_GUIDE.md  ← "how we optimized this"
COST_MODEL_EXPLAINED.md  ← "the cost model, in plain English"
DISPATCH_GAP_ANALYSIS.md  ← (what is this?)
HANDOFF.md      ← (939-line AI session log?)
SPRINT_GPU_MONOTONICITY_DAG.md  ← (1462-line sprint board?)
```

README.md does link to OPTIMIZATION_GUIDE.md and RESULTS.md under a
"Documentation" section at the bottom, which is good. But it doesn't link
to COST_MODEL_EXPLAINED.md or DISPATCH_GAP_ANALYSIS.md — so a newcomer
would find those by browsing the file list and might reasonably try to
read them. COST_MODEL_EXPLAINED.md would give them stale information.
DISPATCH_GAP_ANALYSIS.md would confuse them (it has no context for what
problem it's solving or whether it's current). HANDOFF.md and the DAG
would be overwhelming and possibly alarming.

**Recommendation:** After the cleanup actions above, README.md's
"Documentation" section should point to the (now-correct) surviving docs
and nothing else. Every top-level .md file should be either (a) linked
from README.md with a clear one-line description of what it's for, or
(b) not at the repo root.

---

## 3. Onboarding / Quickstart Quality

### What README.md does well

- **CI badge, license badge, paper link, status line** — professional
  first impression, matches conventions (numpy, pytorch, FFTW all do
  this).
- **"What is ICM?" section** — one paragraph, accessible to a non-poker
  audience.
- **Quick Start is genuinely quick:** `make && ./bench_grid verify &&
  ./bench_grid`. Three commands, 30 seconds on a Mac. This is the right
  bar.
- **API section with copy-pasteable C code** — exactly what a developer
  needs.
- **Performance table in the README itself** — crucial for a performance
  library. Shows CPU and GPU numbers side by side. This is what FFTW's
  homepage does well and what many HPC libraries omit (hiding perf behind
  a separate page).

### What README.md is missing compared to well-known HPC libraries

**Compared to FFTW (fftw.org, its README and manual):**

1. **No "Which engine/method gets used when?" section.** FFTW's manual
   prominently explains that it picks plans automatically and gives the
   user enough intuition to know when to trust the automatic choice. ICM
   has the same story (three engines, automatic dispatch) but README.md
   never explains it. The "How It Works" section is a mathematical
   derivation, not a user-facing dispatch explanation. A user worried
   about performance has to read OPTIMIZATION_GUIDE.md (633 lines) to
   understand what their `icm_equity()` call actually dispatches.

   **Fix:** Add a 3-4 sentence "Automatic Dispatch" subsection to
   README.md: "The library picks between a batched linear engine
   (fastest for small k), a hybrid block-FFT engine (fastest for large
   k), and a pure tree engine based on your (n,k) — you never configure
   this manually." Link to OPTIMIZATION_GUIDE.md for details.

2. **No "Platform support" matrix.** FFTW's README has a clear table of
   supported platforms, compilers, and architectures. ICM currently
   supports macOS ARM64 (M3 Pro), Linux x86-64 (Zen 4), and NVIDIA B200
   GPU — but you have to piece this together from scattered mentions in
   the Build section.

   **Fix:** Add a small table in README.md: Platform | Architecture |
   FFT Backend | Status. M3 Pro | ARM64 (Apple Silicon) | vDSP/FFTW |
   Shipped. Zen 4 | x86-64 (AMD) | AOCL-FFTW | Shipped. B200 | NVIDIA
   sm_100 | cuFFT+cuFFTDx | Shipped.

3. **No "Getting help / Reporting issues" section.** Standard in every
   major HPC library (OpenBLAS, FFTW, Eigen all have it). Simple but
   signals that the project is maintained.

**Compared to OpenBLAS (GitHub README):**

4. **No one-line "What problem does this solve?" for non-domain
   readers.** OpenBLAS: "OpenBLAS is an optimized BLAS library based on
   GotoBLAS2." ICM: the poker-domain framing is excellent for the target
   audience, but a one-line "this is a high-performance combinatorics
   library" alternative framing would broaden the appeal. The paper
   abstract does this well — stealing one sentence from it for the README
   would help.

**Compared to LAPACK (netlib, its README):**

5. **No "Citing this work" section.** The paper exists but there's no
   BibTeX or citation instruction. Standard for academic-adjacent
   software.

### CLAUDE.md Build/Test Duplication

Addressed in §2 above — the split is fine. CLAUDE.md is gitignored and
doesn't ship. No change needed beyond the stale-tool-reference fix in §4.

---

## 4. Stale / Leftover Tracked Files

### scripts/ — Investigation Artifacts vs Production Tools

**Git-tracked scripts (16 files):**

| File | Assessment |
|------|-----------|
| `DIAGNOSTIC_REPORT.md` | KEEP. Permanent record of the long-sweep GPU OOM root cause. HANDOFF.md confirms status. |
| `analyze_fftsize_bug_blast_radius.py` | **STALE.** One-off Python simulation used during the wrap-correction bug investigation. Its results were validated against real hardware output and the fix is now landed. The script has no ongoing use — it simulates decision logic now superseded by the actual fix in `gpu_plan.cu`. HANDOFF.md's own correction note says this script had "approximation errors." Archive or delete. |
| `b200_session_20260726/` | **STALE as active tooling, KEEP for provenance.** Raw CSVs and partial output logs from a specific rental session. Two `.cu` probe tools (`boundary_probe.cu`, `bselect_gap_probe.cu`) that were one-off diagnostic instruments for that session. Not general-purpose. HANDOFF.md references these files. Move to `results/` or an archive. |
| `b200_verify_and_sweep.sh` | KEEP. Gated multi-stage verification orchestration for B200 instances. Still useful for anyone renting a B200. |
| `frontier_probe.cu` | **LIKELY STALE.** HANDOFF.md says this was a "substitute" that "only measures timing at the 5 OLD threshold points — it does not search for a new threshold." It was replaced by `threshold_search_gpu.cu`. If `threshold_search_gpu.cu` is the standing tool, `frontier_probe.cu` is dead code. **Confirm with supervisor** — if `threshold_search_gpu.cu` supercedes it, delete. |
| `gpu_below_sat_fix_plan.md` | KEEP. Design doc for the below_sat fix (node J1). The fix is landed and hardware-verified. This is the permanent design record. |
| `gpu_dispatch_validate.cu` | KEEP — but MISPLACED. THIS IS A PRODUCTION TOOL. It's the GPU analogue of `./bench_grid crossover`. Per HANDOFF.md, it's the standing validation harness that found the B-selection gap above n=1,572,864. Should be in `tools/`, not `scripts/`. |
| `gpu_ragged_tree_fix_plan.md` | KEEP. Design doc for the ragged-tree fix (node J). Still current — the fix is blocked but the design doc is the reference. |
| `gpu_ragged_tree_fix_premortem.md` | KEEP. Independent review checklist (node N). Used to grade M's diff. Still valid per board. |
| `gpu_ws_repro.cu` | KEEP. Standalone repro for the original GPU OOM at n=2,097,152. HANDOFF.md says "useful as a regression check." |
| `l_bselect_extension_prep.md` | KEEP. Prep work for node L (extend B-selection table above n=1,572,864). Not yet executed — still an open board item. |
| `monotonicity_final_verdict.py` | **STALE.** One-off Python analysis script from the non-monotonicity re-verification pass. Its conclusions are already recorded in RESULTS.md and HANDOFF.md. The script itself has no ongoing use. Archive or delete. |
| `remeasure_nonmono.cu` | **LIKELY STALE.** One-off probe tool used to re-measure specific non-monotonicity cells during a specific rental session. The measurements it took are now recorded in `scripts/b200_fix_verified_20260726.txt` (gitignored) and HANDOFF.md. `gpu_dispatch_validate.cu` is the general-purpose replacement. |
| `threshold_search_gpu.cu` | KEEP — but MISPLACED. THIS IS A PRODUCTION TOOL. It's the standing GPU 1-second-threshold binary search. Never been run (blocked on the ragged-tree regression). Should be in `tools/`. |
| `verify_below_sat_fix.cu` | KEEP. The standing "correctness at small scale, timing at large scale" pattern for GPU verification. Explicitly documented as the pattern to follow for future work. Should probably be in `tools/`. |
| `ragged_tree_cufftdx_research.md` | KEEP (new, this session). External research findings (node R0). |
| `zen4_qa_report_20260727.md` | KEEP but REDACT. Contains the Zen4 IP address. Findings should be merged into RESULTS.md. |
| `l_skeleton_large_n.csv` | **STALE if L is executed.** Skeleton data for node L. If L runs and generates real calibration data, this CSV is superseded. Gitignored (`*.csv` pattern). |

**Gitignored files in `scripts/` (visible in `ls` but not tracked):**
- `b200_fix_verified_20260726.txt` — raw debug output, gitignored. OK.
- `b200_nonmono_debug_20260726.txt` — raw debug output, gitignored. OK.

### tools/ — Generally Clean

All 34 tracked files in `tools/` appear to be genuine, active tools. No
obvious dead code. One concern:

- **`tools/fit_gpu_cost_model.py`** — per COST_MODEL_EXPLAINED.md's own
  Gap #4, this tool "fits four parameters that are never used by any
  code." The tool produces output that nothing reads. This is either dead
  code (delete) or unfinished integration work (document as such).
  HANDOFF.md doesn't mention it; COST_MODEL_EXPLAINED.md flags it as an
  open gap. **Recommendation:** Either delete or add a prominent comment
  at the top: "NOTE: The fitted parameters (C_wrap, C_school, R, C_gap)
  are not currently consumed by any GPU source code. This tool is kept
  for future integration. See COST_MODEL_EXPLAINED.md §5 Gap #4."

### CLAUDE.md Stale Tool References

CLAUDE.md's Directory Structure section lists:
- **`tools/tier_ablation.cu`** — does NOT exist. Not tracked, not present.
- **`tools/quantify_dispatch_gap.c`** — does NOT exist. Not tracked, not
  present. Referenced in DISPATCH_GAP_ANALYSIS.md as well (which itself
  should be deleted).

Since CLAUDE.md is gitignored (doesn't ship), these stale references only
harm AI-agent productivity. Still worth fixing — remove both from
CLAUDE.md's directory listing.

### DISPATCH_GAP_ANALYSIS.md

Already addressed in §2. **DELETE** or archive. It references
`tools/quantify_dispatch_gap.c` and `tools/probe_leaf_extract.c`; the
former doesn't exist, the latter does but the report's conclusions were
acted upon and superseded.

### COST_MODEL_EXPLAINED.md Caveat — Already Addressed

Addressed in §2. The "3-4 orders of magnitude" caveat about the cost
model's prediction accuracy is **false** — it was refuted the same
session. The file is serving stale, wrong information to anyone who reads
it.

---

## 5. Prioritized Punch List

### 🔴 CRITICAL — Must Address Before Public Release

1. **DELETE `SPRINT_GPU_MONOTONICITY_DAG.md` from the public repo.**
   Contains a real public IP address (`84.32.71.35`, 6 occurrences) and 9
   vast.ai contract IDs. The file's own header says it's ephemeral.
   Extract any design notes worth keeping (the three non-monotonicity
   root causes) into a clean document if needed.

2. **REDACT IP `84.32.71.35` from `devices/zen4/fft_config.h` line 229**
   and from `scripts/zen4_qa_report_20260727.md` line 3. Replace with
   "Zen4 reference machine" or similar anonymized description.

3. **Resolve HANDOFF.md before shipping.** Options in §2 above. At minimum:
   redact contract IDs if the file ships at all. The preferred option is
   to split it — extract the load-bearing operational knowledge (B200
   build recipe, Zen4 porting procedure, architecture dispatch-layers
   section) into permanent docs, delete the session-log narrative from
   the public repo.

### 🟠 HIGH — Confusing or Wrong for Newcomers

4. **MERGE and UPDATE `COST_MODEL_EXPLAINED.md` into
   `OPTIMIZATION_GUIDE.md`.** COST_MODEL_EXPLAINED.md is stale in
   critical ways (says a fix is "awaiting verification" when it was
   verified; repeats a refuted caveat about cost-model accuracy). Delete
   the standalone file after merging its non-overlapping content
   (primarily the "why this design pattern" precedent discussion in §4).

5. **DELETE `DISPATCH_GAP_ANALYSIS.md`** from the repo root. Point-in-time
   investigation report with an unexpanded `$(date)` field, referencing
   tools that no longer exist. Superseded by the empirical crossover
   table approach.

6. **Add to README.md:**
   - "Automatic Dispatch" subsection (3-4 sentences explaining the three
     engines, with link to OPTIMIZATION_GUIDE.md)
   - Platform support matrix (M3 Pro / Zen 4 / B200, architectures, FFT
     backends, status)
   - "Getting help / Reporting issues" section
   - "Citing this work" section with BibTeX

7. **Fix stale references in CLAUDE.md's Directory Structure:**
   - Remove `tools/tier_ablation.cu` (doesn't exist)
   - Remove `tools/quantify_dispatch_gap.c` (doesn't exist)

### 🟡 MEDIUM — Cleanliness and Discoverability

8. **Move production GPU tools from `scripts/` to `tools/`:**
   - `gpu_dispatch_validate.cu` → `tools/`
   - `threshold_search_gpu.cu` → `tools/`
   - `verify_below_sat_fix.cu` → `tools/`
   Update CLAUDE.md and README.md references accordingly.

9. **Archive or delete stale one-off scripts:**
   - `scripts/analyze_fftsize_bug_blast_radius.py` — served its purpose,
     now superseded by the landed fix in `gpu_plan.cu`
   - `scripts/monotonicity_final_verdict.py` — conclusions recorded in
     RESULTS.md
   - `scripts/frontier_probe.cu` — superseded by
     `threshold_search_gpu.cu` (confirm with supervisor first)

10. **Move `scripts/b200_session_20260726/` to
    `results/b200_session_20260726/`** (or equivalent archive location).
    These are provenance artifacts, not tools.

### 🟢 LOW — Nice to Have

11. **Tag or document `tools/fit_gpu_cost_model.py`'s status.** Either
    delete it (dead code) or add a comment explaining that its fitted
    parameters are not currently consumed — this is a known open gap,
    not an oversight.

12. **Review `scripts/remeasure_nonmono.cu`.** It may be superseded by
    `gpu_dispatch_validate.cu`. If so, delete. If it tests something the
    validation harness doesn't cover, document what.

13. **Document `scripts/` in README.md's Project Structure section.**
    Currently, README.md's "Project Structure" tree view shows `src/`,
    `tools/`, `devices/`, `bench/`, `python/` but NOT `scripts/` or
    `results/` or `paper/`. A complete tree helps newcomers navigate.

---

## Summary

The repo is in solid structural shape — the `src/`/`tools/`/`devices/`
layout is clean, the build system works, and the README.md quickstart is
genuinely quick. The issues are concentrated in three areas:

1. **Two files contain live infrastructure identifiers that must not ship
   publicly** (SPRINT_GPU_MONOTONICITY_DAG.md: IP + contract IDs;
   devices/zen4/fft_config.h + scripts/zen4_qa_report_20260727.md: IP).

2. **Doc sprawl at the repo root creates confusion and staleness** — 8
   markdown files, several overlapping, one provably stale
   (COST_MODEL_EXPLAINED.md), one point-in-time with no ongoing value
   (DISPATCH_GAP_ANALYSIS.md), two that are internal AI-agent artifacts
   never meant for public view (HANDOFF.md, SPRINT_GPU_MONOTONICITY_DAG.md).

3. **`scripts/` has accumulated both production tools and one-off
   investigation artifacts** with no separation between them, contrary to
   its documented purpose as "not part of the regular build."

After the cleanup above, the repo should have 3-4 top-level .md files
(README.md, RESULTS.md, OPTIMIZATION_GUIDE.md, possibly a merged
cost-model doc), all linked from README.md's Documentation section, with
clear roles that don't overlap. SPRINT_GPU_MONOTONICITY_DAG.md and
HANDOFF.md should not be in the public repo in their current form.
