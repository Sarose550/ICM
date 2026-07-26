# HANDOFF.md

## What this project is

ICM (Independent Chip Model) equity computation for poker tournaments: a
high-performance C library computing tournament placement equities via
generating-function quadrature. Three CPU engines (linear/hybrid/tree)
with cost-model-driven automatic dispatch, plus a CUDA GPU implementation.
Repo: GitHub `Sarose550/ICM`, working branch `results-gpu-section`, PR #7
open (https://github.com/Sarose550/ICM/pull/7). Do not merge without the
user's explicit go-ahead; pushing commits onto it is fine and expected.
Sibling repo `~/Documents/ICM_paper` (local-only, no remote) holds the
accompanying academic paper (`icm_paper.tex`); the compiled PDF is copied
into `paper/icm_paper.pdf` in the main repo and committed there. Anything
that goes in `RESULTS.md` should also be reflected in the paper; the two
must never diverge.

## Goal

Ship this repo publicly in genuinely portfolio-ready shape: professional
code, an accurate paper, a friction-free device-porting story, and
nothing stale, hand-waved, or silently broken.

## START HERE

Continue using the `supervisor-dag` skill/pattern for this work: a live
strike board, DeepSeek Deck workers for the mechanical/investigative
lifting, supervisor review of every diff before it lands, real hardware
verification before anything is called done. That pattern is how every
finding below got found, fixed, and verified across several rounds
without losing track of state -- keep doing it that way, don't drop to
ad-hoc work. (One real lapse this session: three single-node DeepSeek
dispatches were done via direct `deck spawn` without a board file, purely
out of habit, before the user caught it and asked for it to stop --
they were retroactively logged onto a board afterward, but don't repeat
the lapse. Every `Model: deepseek` node, however small, goes on a board.)

**Current live board: `SPRINT_GPU_MONOTONICITY_DAG.md`.** Read it first --
it has the up-to-date node graph. As of this writing: nodes A, B, G are
done; **E (hardware-verify the B-selection calibration gap) is the next
action**, followed by F (recalibrate, only if E confirms a real
inversion), then H (finally run the threshold search). All three remaining
nodes need a real B200 rental -- **do not rent without an explicit
user go-ahead**, this was emphasized repeatedly this session after money
was spent on a rental that then surfaced a new problem. See "Next action,
on the next B200 rental" near the end of the "Non-monotonicity" section
below for the exact ordered checklist, and the vast.ai balance snapshot
there (~$3.93 as of this writing -- re-verify against the real balance,
don't trust a stale number).

**Historical, already fully resolved (do not re-derive, kept only for
provenance)**: an earlier cleanup/audit pass (`SPRINT_CLEANUP_AUDIT_DAG.md`,
closed and deleted 2026-07-25/26) found and fixed several instances of
repo drift: the paper (`~/Documents/ICM_paper`) was stale relative to
`RESULTS.md` (still true today, see Next Steps -- deliberately deferred
until the GPU numbers below are finalized, not an oversight); a stale doc
comment on `memory_strategy` in `src/icm_gpu.h` (fixed); `CLAUDE.md`'s
Directory Structure section missing `scripts/` (fixed); a keep/delete
decision on diagnostic-only files in `scripts/` (applied); and a full
`RESULTS.md` read-through that found nothing else stale. M3 Pro CPU data
(`results/*m3pro*`) was separately confirmed NOT stale relative to any
subsequent fix and does not need re-verification unless CPU dispatch code
changes again.

## GPU OOM: history, both findings fixed and hardware-verified (2026-07-25 to 2026-07-26)

### Original OOM: n=2,097,152 k=256/512

Four stacked commits, each reviewed line-by-line before landing (`8ce8a07`,
`2f8fd22`, `66a2bc8`, `abd9fa4`):
1. Reactive retry-on-OOM for the shared cuFFT workspace allocation.
2. Proactive: query real cuFFT workspace size before committing to a
   `q_batch`, instead of an incomplete estimate.
3. Replace live per-plan cuFFT workspace probing with an offline
   calibration table (mirrors the existing FFT-timing calibration
   pattern), live probing kept as a fallback for uncalibrated sizes.
4. **Root cause**: `allocate_level_buffers()` was unconditionally
   creating a real cuFFT plan (with real, batch-scaled VRAM workspace)
   for every FFT-tier level, including FUSED-tier (cuFFTDx) levels that
   never actually execute against that plan and, per NVIDIA's own
   cuFFTDx docs, need zero workspace at every size this codebase uses
   (powers of 2, 64-8192, all under the documented 32768 no-workspace
   threshold). Now skipped for confirmed cuFFTDx-supported FUSED levels;
   a lazy on-demand fallback exists for the rare case cuFFTDx dispatch
   itself fails at runtime.

Verified on a rented B200 (2026-07-25): `bench_gpu_fused verify` passes
36/0; the exact failing point (n=2097152, k=256/512) passes in an
isolated repro, debug output confirming `cufft_ws=0.0 MB` (every level in
its tree is FUSED-tier, so zero cuFFT plans are even created for it).

### Second OOM: long-sweep-only, found while verifying the first fix

The full 211-point `heatmap_gpu` sweep (2026-07-25) failed on 21/211
cells, all at large n (2,097,152 - 33,554,432), consistently in a
middle-k band. Every one of those exact points passed cleanly in
isolation, meaning the failure depended on long-running process state
(~190 prior sequential plan create/destroy cycles), not the (n,k) point
itself.

**Root cause**: a DeepSeek static-analysis pass (`scripts/DIAGNOSTIC_REPORT.md`)
traced every allocation/free pairing and found no genuine leak. Leading
hypothesis: CUDA default-allocator fragmentation. Two real bugs found in
this codebase's existing (but non-functional for the allocations that
matter) memory-pool support: the main arena allocation used a raw
`cudaMalloc` completely independent of `plan->use_async_pool` at every
setting, and a stray `plan->use_async_pool = false` disabled pooling for
the one allocation that could previously use it.

**Fix** (commit `386c856`): route the arena and shared cuFFT workspace
allocations through a CUDA stream-ordered memory pool
(`cudaMallocAsync`/`cudaFreeAsync`), remove the stray override, make
pooling the default (matching PyTorch/RAPIDS RMM practice), add
`icm_gpu_release_pooled_memory()` as an explicit escape valve. Release
threshold set to `UINT64_MAX` (never auto-release, not a bounded
fraction -- individual arenas run up to ~160GB, so a modest fraction-of-
VRAM cap would provide little real protection; the escape valve makes
unconditional retention safe).

**Verified on real B200 hardware (2026-07-26)**: `bench_gpu_fused verify`
passes 36/0. The full 211-point `heatmap_gpu` sweep now passes with
**zero errors**, confirmed directly against all 6 representative
previously-failing points. `RESULTS.md` and `results/gpu_heatmap_*.png`
regenerated from the complete, gap-free data. A DeepSeek sanity pass over
the new CSV flagged several apparent non-monotonicities; triaged and
found none indicate a real problem (mix of expected tool behavior --
`reps=1` above 100ms is intentional, `cv=0.000` for single-rep cells is
`cv_ms()`'s defined return, `B=32` at n=1,048,576 is that exact point's
real calibrated B-selection value -- and two plausible single-shot
measurement-noise points, not re-verified further).

### Still open: the 1-second threshold / frontier probes

**This was explicitly asked for and is NOT done.** `tools/push_limit_gpu.cu`'s
real methodology (a binary search for the 1000ms crossing) is what
originally produced the threshold numbers in `RESULTS.md`. That tool also
does an exhaustive B/M/T hyperparameter grid search alongside the
threshold search, which was measured as impractically slow (>3.5 min on
just its first, smallest n value, no early-exit) and was not run to
completion. A substitute (`scripts/frontier_probe.cu`) was used instead,
but it only measures timing at the 5 OLD threshold points -- it does not
search for a new threshold, so the actual number the user asked for was
never produced. `RESULTS.md`'s "1-second threshold" section currently
says "needs re-measurement" rather than a number, correctly, not a
placeholder guess.

**Tool written, not yet run (2026-07-26).** `scripts/threshold_search_gpu.cu`:
a real binary search using the lightweight single-measurement approach
`frontier_probe.cu` already established (one `icm_gpu_equity_with_plan`
call per candidate n, median of 5 reps, via `icm_gpu_plan_create` +
`icm_gpu_equity_with_plan`, not the full B/M/T sweep). Searches both
curves (k=n and k=100) for the 1000ms crossing. Written and reviewed
line-by-line against `src/icm_gpu.h`'s real API signatures and the
Makefile's actual `push_limit_gpu` build recipe (both check out exactly
-- no fabricated calls, no wrong `-arch` flag this time). Cannot be
built/run in this dev environment (no local CUDA hardware); needs a B200
rental to actually execute. Expected runtime ~2-4 minutes.

**Non-monotonicity data-quality gap, now closed (2026-07-26).** The
"two plausible single-shot measurement noise" points flagged in commit
`f568d40`'s message were explicitly not re-verified at the time. Re-checked
from scratch this session (independent DeepSeek re-derivation, all 211
CSV rows, all directions): one point (`n=512, k=64`, 0.11ms) is confirmed
genuine timer jitter, harmless to any conclusion. The other
(`n=4,194,304, k=128`, `893.3ms`, `reps=1`) is a **real bad data point**
-- ~50-80% too high versus its neighbors (expected ~550-600ms based on
the trend from `n=2,097,152`/`n=8,388,608`), the largest non-monotonicity
in the dataset (31% dip to the next k). **Must be re-measured with
reps>=5 on the next B200 rental before this cell is treated as final for
`RESULTS.md`/the paper.** Two more cells (`n=8,388,608, k=128` and
`n=1,048,576, k=128`) are lower-priority "should re-measure if time
permits" -- both single-rep, both plausibly noisy but not confirmed bad.
Everything else previously triaged as benign (`reps=1` above 100ms,
`cv=0.000` for single-rep cells, `B=32` at `n=1,048,576`) was
independently re-confirmed, still holds. Full data + reasoning in this
session's `SPRINT_GPU_THRESHOLD_DAG.md` node `B_NONMONOTONICITY_REVERIFY`
(board deleted at close per convention; this paragraph is the permanent
record).

**Non-monotonicity: root cause CONFIRMED via real B200 rental
(2026-07-26), NOT measurement noise.** A prior planning pass (DeepSeek
root-cause hypotheses, supervisor-reviewed) concluded single-rep
measurement noise was the most plausible explanation. That hypothesis
was tested directly on rented B200 hardware and **refuted**: re-measuring
`n=4,194,304, k=128` at reps=5 returned **890.063ms with cv=0.0000**
(fully deterministic across all 5 reps), matching the original 893.3ms
almost exactly. The neighbor cell `n=4,194,304, k=256` (612ms) is
genuinely faster despite having roughly double the convolution length
(`build_conv=511` vs `255`) at every tree level -- a real, reproducible,
counter-intuitive result, not noise.

**Confirmed mechanism** (via `ICM_GPU_DEBUG_PLAN=1`, raw output kept at
`scripts/b200_nonmono_debug_20260726.txt`; code read directly, not taken
on faith):
1. `best_fft_config_joint_gpu()` (`src/gpu/gpu_plan.cu:300`) picks
   `fft_n`/wrap-correction size using ONLY cuFFT-batched cost estimates
   (`estimate_cufft_pipeline_ns_batched`). For this cell it picked
   `fft_n=128` with `wrap_m=127` (a large wrap correction) because that
   looked cheapest under the cuFFT cost model, for 9 of 16 tree levels.
2. `pick_tier_for_fft_len()` (`gpu_plan.cu:437`) then compares
   fused-vs-cuFFT cost AT THAT ALREADY-CHOSEN fft_n (128) and picks
   `GPU_TIER_FUSED` (fused nearly always wins at small sizes).
3. The one check that would catch this -- "is a clean power-of-2 fused
   size actually cheaper?" (`gpu_plan.cu:693`, and its twin at the
   propagate-level path, `gpu_plan.cu:940`) -- is gated on
   `tier != GPU_TIER_FUSED`. Since tier is already FUSED from step 2,
   this comparison never runs. The model never checks "fused at 128 with
   heavy wrap correction" against "fused at 256 with zero wrap
   correction" -- exactly what the `k=256` neighbor cleanly uses instead,
   and measures faster on real hardware.

**This is a genuine, fixable cost-model bug** (not an inherent hardware
property) at two symmetric call sites. Fix direction (not yet
implemented -- needs care plus runtime re-verification, not a quick
patch): the gate at both `gpu_plan.cu:693` and `:940` should also fire
when tier is already FUSED but paying a nonzero wrap penalty
(`bwrap > 0 || cwrap > 0`), not just when tier isn't FUSED at all. The
"current cost" baseline computed in the `else` branch at those sites
(currently always `estimate_cufft_pipeline_ns_batched(...)`) would also
need to switch to `estimate_fused_build_ns`/`estimate_fused_corr_ns` when
entering with tier already FUSED, or the comparison baseline would be
wrong.

**Session tooling notes**: both `scripts/threshold_search_gpu.cu` and
the new `scripts/remeasure_nonmono.cu` originally had a doc-comment bug
where `python*/dist-packages` (a literal `*/` from the glob pattern)
closed the C block comment early, breaking compilation -- fixed in both
(now uses `find ... -path '*dist-packages/...'` instead, no `*/`
adjacency). Also: the Makefile's `GPU_OBJS_FUSED` pattern produces
double-prefixed object names (`build/gpu_gpu_kernels_fused.o`, not
`build/gpu_kernels_fused.o` as both tools' header comments assumed) --
harmless for the Makefile itself but any hand-written link line (as in
both scripts' build-recipe comments) must use the double-prefixed names.

**Fix implemented (2026-07-26), not yet hardware-verified.**
DeepSeek-drafted, supervisor-reviewed line-by-line against the actual
diff (not accepted on the worker's own "done" framing). Applied at both
call sites in `src/gpu/gpu_plan.cu` (~line 698, build-level path in
`estimate_candidate_cost()`; ~line 959, propagate-level path in
`build_plan_metadata()`): the gate now fires when
`tier != GPU_TIER_FUSED || bwrap > 0 || cwrap > 0` (site-2 uses
`build_wrap_m`/`corr_wrap_m`), instead of only `tier != GPU_TIER_FUSED`.
When entering with tier already FUSED, the "current cost" baseline now
uses `estimate_fused_build_ns`/`estimate_fused_corr_ns` plus the existing
wrap-penalty formula shape (verified: build penalty uses `/2.0`, corr
penalty doesn't -- an asymmetry that already existed in the original code
and was preserved exactly, not "cleaned up"). Neither
`best_fft_config_joint_gpu()`, `best_fft_config_gpu()`, nor
`pick_tier_for_fft_len()` were touched, per constraint. Braces/syntax
verified by direct read (no local CUDA toolchain to compile against);
cannot be built/tested until the next B200 rental.

**Blast radius (2026-07-26), offline simulation, real-hardware
validated.** A Python reimplementation of the decision logic
(`scripts/analyze_fftsize_bug_blast_radius.py`) exactly reproduced every
real value from `scripts/b200_nonmono_debug_20260726.txt` before being
trusted for a sweep (validation checked directly, not assumed).
Sweeping 189 `(n,k)` grid points: **12 points affected (6.3%)**, every one
sharing the identical signature `conv_build=255` (i.e. `k_pad=128`) at
whichever tree levels hit that build length -- BEFORE: `fft_n=128,
bwm=127/cwm=127`; AFTER: `fft_n=256, bwm=0/cwm=0`. **This is the same bug
signature as the k=128->256 dips at n=65,536-524,288 (B=64) already
cataloged from the earlier non-monotonicity re-verification pass** --
confirmed directly by running the simulation at those exact points, not
just by pattern-matching. That means this one fix, if it holds up on
hardware, likely resolves 5 of the 11 previously-cataloged
non-monotonicities (the mandatory `n=4,194,304,k=128` cell plus the four
`n=65,536`-`524,288` k=128->256 dips), not just the one cell it was
found from.

**Fix hardware-verified on a second B200 rental (2026-07-26): real, large,
correct.** Rebuilt with the fix, `bench_gpu_fused verify` passed 36/0 (no
regression). Re-ran `scripts/remeasure_nonmono.cu` (extended with a
generalization check at `n=524,288`): every previously-bad cell improved
substantially and stayed fully deterministic (`cv≈0`) --
`n=4,194,304,k=128` moved from ~890ms to **511.7ms** (right in the
predicted ~500-650ms range); `n=8,388,608,k=128` from 1349.8ms to
1025.8ms; `n=1,048,576,k=128` from 122.6ms to 93.7ms;
`n=524,288,k=128` from 54.17ms to 42.2ms, now correctly below its
`k=256` neighbor (51.2ms), resolving the inversion. Both previously-
flagged non-monotonic pairs are now correctly ordered. Raw output kept
at `scripts/b200_fix_verified_20260726.txt`.

**Correction to the earlier caveat below (RESOLVED, was based on a wrong
claim -- corrected same session):** an earlier note here said the cost
model's own predicted benefit was 3-4 orders of magnitude smaller than
the real ~280ms hardware gap, and speculated the GPU wrap-correction cost
was under-modeled at the C-code/calibration level. **That speculation was
investigated and refuted.** A dedicated trace (verified directly against
source, not taken on a worker's framing) confirmed
`icm_gpu_measure_fused_pair_ns()` (`src/gpu/gpu_api.cu:747-748`) DOES
correctly divide the batched measurement by the batch count
(`bsamp[...]/  (double)nparents`) before storing it -- the calibration
values in `gpu_fft_config.h` are genuine, correctly-normalized per-FFT
nanoseconds, not batch totals and not mislabeled microseconds (a
different worker's claim to the contrary this session was independently
checked and was wrong). The CPU calibration path (`tools/calibrate.c`)
never batches at all, so the same class of risk does not exist there
either -- **this does NOT ripple into the CPU cost model.** The real
source of the earlier magnitude mismatch was an approximation error in
the offline Python blast-radius/monotonicity simulation scripts
(`scripts/analyze_fftsize_bug_blast_radius.py` and its extension), not in
the production C++ cost model or the calibration data -- consistent with
the fix's hardware-verified real benefit being large and correct as
measured above.

**Second, DIFFERENT non-monotonicity mechanism found (2026-07-26), not
yet hardware-verified, blocks the threshold search.** While live-watching
a `scripts/threshold_search_gpu.cu` run (killed early, not saved), the
user observed a real-time cost drop near n≈2^20. Investigation traced
this to the GPU B-selection calibration table (`gbselect_*` in
`devices/b200/gpu_fft_config.h`): only 8 calibrated n-values exist, and
the gap between `n=1,048,576` (`B=32`) and `n=1,572,864` (`B=96-192`) is
a ~50% jump with nothing calibrated in between. Nearest-neighbor lookup
therefore applies `B=32` as a hard cliff to every real n up to the
geometric midpoint (~1,285,000) and jumps straight to `B=96+` above it --
independently confirmed by directly reading the calibration table (not
dependent on the simulation script above). **The correct fix is adding
real calibration points in that gap, not "enforcing monotonicity"** --
deliberately picking a worse B just to smooth the timing curve would defeat
the entire purpose of the cost model and was explicitly rejected by the
user. Per a dedicated investigation: this is achievable with EXISTING
tooling (`tools/calibrate_gpu_best_b.cu`'s narrow-around resume mode +
`tools/gen_calib_skeleton.py` for new anchor points + `calibrate_block_size.py`
for injection), no new tools needed, estimated ~3-16 minutes of B200 time
for ~8 new `(n,k)` points spanning the gap. **Not yet independently
re-verified on real hardware** that this cliff actually causes a real
wall-clock inversion (only confirmed via table-reading + offline
simulation so far) -- do that before or alongside the recalibration.

**GPU dispatch-validation harness written and reviewed (2026-07-26), not
yet run.** `scripts/gpu_dispatch_validate.cu` -- the standing GPU
analogue of `./bench_grid crossover` (Next Steps item 3 above). For a
grid of `(n,k)` points spanning both the calibrated B-selection anchors
and the identified sparse gap (n=1,200,000/1,300,000/1,400,000 included
by design), it measures the real dispatched configuration and forces
nearby alternative B values via the real `ICM_GPU_FORCE_B` mechanism
(`gpu_plan.cu:850-854`), flagging any point where the dispatched choice
isn't the fastest measured, plus a separate n-monotonicity sweep at fixed
k. Supervisor-reviewed against actual source: API signatures, the
`kBCandidates` table (verified an exact 48-element match against
`gpu_plan.cu:11-16`), and the `ICM_GPU_FORCE_B` env-var mechanism all
check out. **One real bug caught and fixed**: the build recipe comment
had the dlink object misnamed `gpu_gpu_dlink_fused.o` (double-prefixed,
copying the pattern of the other four objects) when the real file
(confirmed against this session's actual build log) is single-prefixed,
`gpu_dlink_fused.o` -- the `-dlink` step produces its own name, not the
`GPU_OBJS_FUSED` pattern substitution the other four go through.
**FIXED (was initially mis-assessed as a tolerable limitation, corrected
same session):** `ICM_GPU_FORCE_B`'s override is silently ignored by the
real dispatch code if the requested B exceeds `plan->k_pad`
(`gpu_plan.cu:853`'s own constraint), which would otherwise make the tool
compare the dispatched config against a second measurement of itself and
report a false "optimal" verdict. Traced concretely (not hypothetically):
this collides directly with the tool's own highest-priority test points
(n=1,200,000/1,300,000/1,400,000 at k=100, right in the B-selection gap)
-- if the dispatched B there lands past the calibration cliff (96+), the
tool's "double" alternative would very plausibly exceed that k's
rounded-up size and get silently rejected, masking a real answer exactly
where it matters most. Fixed: `check_b_optimality()` now compares the
plan's actual applied B against the B it intended to test, skips (does
not count as a valid comparison) any mismatch with an explicit printed
reason, and returns "skipped" for a point if every alternative was
rejected this way, instead of silently reporting "optimal."

**Next action, on the next B200 rental** (this session's second rental is
destroyed; exact vast.ai invoice total this session: $1.376+$0.013 (first
rental, GPU+storage) + $1.262+$0.011 (second rental) = **$2.662 spent**
against the original ~$6.59 available (~$1.59 pre-existing + $5.00 added
this session) -- **~$3.93 actually remains**, not the ~$4.19 estimated
earlier in this document; verify against the real vast.ai balance before
committing to a rental scope regardless, this number is a snapshot):
(1) verify the B-selection
cliff causes a real hardware timing inversion near n≈1.05M-1.57M on the
`k=n` and/or `k=100` curves (a few targeted `remeasure`-style points,
not a full sweep), (2) if confirmed, generate new skeleton anchor points
in that gap and re-run the existing B-selection calibration tool (narrow-
around mode) to close it, (3) spot-check the newly-identified affected
cells from the fix already applied (`n=65,536`-`524,288, k=128`) if not
already covered, (4) run `scripts/gpu_dispatch_validate.cu` (written,
reviewed, ready -- see above) before trusting monotonicity broadly, (5)
only then run `scripts/threshold_search_gpu.cu` for the actual threshold
numbers. Live tracking for this whole arc is now on
`SPRINT_GPU_MONOTONICITY_DAG.md` (nodes E/F/H still open, all
hardware-gated -- do not rent without explicit user go-ahead).

## Architecture: what's actually load-bearing (read before touching cost-model code)

Dispatch happens in three separate layers. Only two of them were ever the
"fragile aggregate formula" problem; the third was never broken and was
never replaced.

1. **Which engine?** (linear / hybrid / tree), `select_engine_ex()` in
   `src/icm.c`. Uses the empirical crossover table
   (`crossover_n[]`/`crossover_k[]` in `fft_config.h`, log-linear
   interpolation via `empirical_crossover_k()` in `src/fft_cost_model.h`)
   for BOTH full-equity and subset queries as of commit `44bc959` (the
   subset-specific analytical formula was removed; subset dispatch now
   reuses the same empirical table, which closed a confirmed 37-45%
   dispatch-accuracy gap).
2. **Which block size B?** (only matters inside the hybrid engine),
   `select_best_B()` in `src/icm.c`. Uses the empirical `bselect` table
   (`bselect_n[]`/`bselect_k[]`/`bselect_B[]`, 2D nearest-neighbor via
   `empirical_best_B()`), built by `tools/calibrate_best_b.c` +
   `tools/calibrate_block_size.py`. GPU equivalent:
   `gpu_select_best_B_est()` in `src/gpu/gpu_plan.cu`, same mechanism,
   `gbselect_*` tables from `tools/calibrate_gpu_best_b.cu`.
3. **Inside the tree/hybrid engine, at every single tree level, every
   single call**: schoolbook or FFT? If FFT, which calibrated size?
   `best_fft_config()` / `best_fft_config_joint()` in
   `src/fft_cost_model.h`, called directly from `tree_ctx_create_ex2()`.
   Reads `calib_times_ns[]` / `calib_sizes[]` (the per-FFT-size table
   from `tools/calibrate.c`, real FFTW PATIENT measurements) directly,
   picking whichever calibrated size (or schoolbook) minimizes real
   measured cost for that exact convolution length.

**Layer 3 was never the problem.** It already compares real measured
per-size timings directly, not summed abstract constants. Layers 1 and 2
summed many individually-correct constants into one aggregate go/no-go
decision, and that's what turned out fragile in aggregate and got
replaced with empirical lookup tables. `tools/calibrate.c`'s output
(consumed by layer 3) is genuinely foundational, not dead code, and must
never be skipped when porting to a new device.

**GPU memory allocation** (new as of commit `386c856`): the arena and
shared cuFFT workspace now go through a CUDA stream-ordered memory pool
by default (`memory_strategy=0`, the default used everywhere in this
codebase). `memory_strategy=1` opts out for callers needing strict
non-pooled allocation. `icm_gpu_release_pooled_memory()` lets a caller
explicitly return pooled memory to the driver. See the GPU OOM section
above for why this exists.

## Critical operational notes

**DeepSeek Deck network access.** The `deck` binary on `$PATH` resolves
to a stale plugin-cache copy that does not support `--allow-network`.
The real, working dev copy is at `~/Documents/deepseek-deck/bin/deck`
(same daemon serves both paths; the dev repo's CLI binary is what
actually matters). Always invoke that path directly when a node needs
`--allow-network`.

**Zen4 rental.** As of this writing, still completely out of stock on
vast.ai (checked repeatedly across multiple sessions), no clean
hourly-billed alternative found (Cherry Servers: right billing model,
0 in stock everywhere; Hetzner: has the hardware, monthly billing only).
Check vast.ai stock again before assuming it's still unavailable.

**When a Zen4 box is available again**: do NOT re-run the adaptive
B-selection calibration (already correct, already committed, 1944
points) and do NOT rebuild AOCL-FFTW wisdom from scratch. Copy
`devices/zen4/fftw_wisdom.dat` directly onto the new box (byte-identical
port, verified working this exact way in a prior session), build
AOCL-FFTW from source with the full flag set (`--enable-sse2 --enable-avx
--enable-avx2 --enable-avx512 --enable-amd-opt`, needs `texinfo` package
installed first or the docs sub-build fails), then rebuild the ICM
binaries against the already-correct config. The ONLY thing that needs
re-running there is the benchmark sweep itself. **Zen4 needs
`OMP_NUM_THREADS=16` explicitly**, never the default `nproc` (32,
SMT-inclusive) -- SMT siblings add no real throughput for this
FPU/vector-port-bound workload; oversubscription silently corrupts
parallel timing at small n (up to 20x+ slower, looks exactly like a real
regression until you check thread counts).

**`make results-refresh`'s parallel-binary gotcha.** The recipe ends by
leaving the OpenMP-enabled `bench_grid` binary sitting in the working
directory. If you manually run `./bench_grid ...` afterward expecting
serial behavior, you'll silently get parallel timing instead. Always
check for "OpenMP enabled: N threads" vs "OpenMP disabled (serial mode)"
in the output, or `make clean && make` before any manual serial probe.

**B200 build recipe, proven across 3 separate rented instances**: pip
install `nvidia-mathdx` (may need `python3-pip` installed first, not
always present on fresh images), apt install `libfftw3-dev`, glob
`/usr/local/lib/python*/dist-packages/nvidia/mathdx/include` for
`CUFFTDX_INC` (don't hardcode the Python minor version), `make
bench_gpu_fused CUDA_ARCH=sm_100 CUFFTDX_INC=...`. **Deploy the repo via
`git ls-files --cached --others --exclude-standard -z | rsync --from0
--files-from=-`, never a raw directory rsync** -- a raw rsync pulled in a
143MB local scratch/venv directory with zero relation to the build on one
occasion. `devices/b200/gpu_fft_config.h` contains TWO independently
calibrated sections that must both survive any regeneration: the FFT
timing/workspace table (from `tools/calibrate_gpu.cu`, pass a real max
size like `67108864`, its default of `131072` silently loses coverage
above ~4M) and the `gbselect_*` B-selection table (from a separate tool,
`tools/calibrate_gpu_best_b.cu` + `tools/calibrate_block_size.py`) --
`calibrate_gpu`'s `write_header()` overwrites the WHOLE file, so
re-running it clobbers the B-selection section unless you splice it back
in from the pre-existing header afterward (do not re-run the B-selection
calibration itself, it's expensive and already correct).

## What Worked

- **Doing real research before dispatching a fix, not guessing.** Fetched
  official NVIDIA cuFFT/cuFFTDx documentation and read the actual kernel
  dispatch code directly before writing the G4 task brief -- found the
  real root cause (wasted cuFFT plan creation for FUSED-tier levels) this
  way, not by assumption.
- **Manually reviewing every DeepSeek diff line-by-line before committing,
  every time, no exceptions.** Caught and fixed real bugs in three
  separate deliverables this way: an infinite loop in G2 (qb==1 +
  persistent failure never terminated), a silent-zero sentinel gap in G3
  (probe failures defaulting to "confidently calibrated, needs 0 bytes"),
  and a build command targeting the wrong GPU architecture in the
  diagnostic report. None of these would have been caught by trusting a
  worker's own "done and verified" framing.
- **Isolated repro testing to distinguish "fixed" from "looks fixed."**
  Directly reproducing the exact failing (n,k) points standalone, both to
  confirm the original OOM fix worked and to prove the long-sweep OOM was
  a genuinely different bug (same points passed in isolation, failed only
  in the long sweep) -- this distinction was the key that unlocked the
  correct root cause instead of chasing the wrong fix.
- **Checking real industry practice (PyTorch, RAPIDS RMM) before deciding
  a memory-pool design**, rather than reasoning from first principles
  alone -- this directly informed making pooling the default rather than
  a narrow opt-in, and the user's own pushback on the release threshold
  (correctly identified 25% as too conservative for this workload's
  actual allocation sizes) came from checking the real numbers, not
  accepting a plausible-sounding default.
- **Setting a hard, explicit time/credit budget before an unattended
  overnight GPU run**, and killing a sweep mid-run when a mistake was
  found (undersized calibration range) rather than letting bad data
  finish generating. Preserved real money and avoided committing numbers
  that would have needed redoing anyway.

## What Didn't Work / mistakes to avoid repeating

- **Using `--fast` flags on benchmark tools without checking what they
  actually cut.** `heatmap_gpu --fast` doesn't just reduce repetitions,
  it truncates the n-range itself (stopped at 524K instead of 33M) --
  producing data that looked complete but was missing most of the range
  needed for the paper. Always read what a `--fast`/quick-mode flag
  actually skips before using it for anything that will be reported.
- **Substituting a smaller/faster tool for an established one without
  flagging the methodology change loudly enough up front.** Using
  `frontier_probe.cu` instead of `push_limit_gpu.cu` for frontier numbers
  was reasonable given the real time constraint, but the substitution
  meant the actual thing asked for (a real binary-search threshold) never
  got produced, and that gap wasn't surfaced clearly until the user asked
  directly. State explicitly what a substitution does NOT give you, not
  just what it does.
- **Running an orchestration script's remote command with `sudo` and
  debug-echoing to stdout without checking for `$(...)` capture
  contamination.** A `remote()` helper's own `echo "[remote] cmd"` line
  going to stdout silently corrupted a `$(remote ...)` command
  substitution used to detect an include path, causing a real build
  failure that looked unrelated to its actual cause. Route debug/logging
  output to stderr in any helper whose stdout might be captured.
- **`git filter-repo --message-callback` without scoping the commit
  range rewrites the ENTIRE reachable history**, not just the commits you
  intend to touch (from an earlier session) -- breaks shared ancestry
  with `origin/main`. Fix: cherry-pick + `git rebase <base> --exec` on
  just the commits needing changes, verify with `git merge-tree` before
  pushing.
- **Ad-hoc single-rep manual probes outside established sweep tools
  produce noise, not signal**, and burn paid-instance time for no
  committed benefit (from an earlier session). Use the canonical tools
  (`bench_grid`, `tools/contour_1s.c --contour`, `bench_grid threshold`
  for the CPU 1-second boundary specifically) or a properly-designed
  binary search, not manual interpolation.
- **Asserting "nothing changed so old data is still valid" without
  checking commit dates against data file dates** (from an earlier
  session) -- always check dates, don't reason from session boundaries.
  This exact class of mistake is why the "M3 Pro data status" check
  above was done explicitly rather than assumed, this time.

## Next Steps

Ordered by priority; start at the top.

1. **Cleanup/audit pass -- DONE except the paper resync (item 1 above),
   which stays deliberately deferred until step 2 lands** (see START HERE
   above): `memory_strategy` doc comment fixed, `scripts/` documented in
   `CLAUDE.md`, diagnostic-file keep/delete decision made and applied,
   full `RESULTS.md` read-through found nothing stale. Don't re-derive
   any of this.
2. **DO NOT run the 1-second-threshold binary search yet.** The
   wrap-correction/tier-lock-in fix (`gpu_plan.cu:693`/`:940`) is
   hardware-verified and real (see "Non-monotonicity" section above --
   every re-measured cell improved substantially, two previously-inverted
   pairs now correctly ordered). A second, DIFFERENT non-monotonicity
   mechanism was found immediately after (the B-selection calibration
   gap, see "Non-monotonicity" section) -- confirmed real by directly
   reading the calibration table, but not yet independently re-verified
   as an actual hardware timing inversion. **The correct fix is adding
   more calibration points in that gap, NOT "enforcing monotonicity"**
   -- deliberately picking a worse-performing B just to smooth the timing
   curve would defeat the entire purpose of the cost model and was
   explicitly rejected by the user. A separate worry (a possible unit/
   batch-normalization bug in the GPU calibration tool) was raised,
   investigated, and **refuted** -- the calibration values are correct,
   this does not need further work. This is now fully tracked on
   `SPRINT_GPU_MONOTONICITY_DAG.md` (nodes E/F/H) -- **the threshold
   search (node H) must not run until E confirms/resolves the B-selection
   gap on real hardware**, since the binary search's correctness depends
   on genuine monotonicity.
3. **GPU-side end-to-end dispatch-validation harness -- DONE, reviewed,
   not yet run.** `scripts/gpu_dispatch_validate.cu`: the GPU analogue of
   `./bench_grid crossover` that this session identified as a long-
   standing gap (a major reason both bugs above went unnoticed for as
   long as they did -- there was no systematic real-hardware check that
   would have caught either). For a grid of `(n,k)` points spanning the
   calibrated B-selection anchors and the identified sparse gap, it
   measures the real dispatched configuration against nearby alternatives
   on real hardware and checks n-monotonicity at fixed k. Two real bugs
   were found in the tool itself during review and fixed before it's
   trusted: a misnamed build object in its own build-recipe comment, and
   a silent-override-rejection issue that could have masked a real
   problem exactly at the border-region points this tool exists to check
   (see "Non-monotonicity" section for both). Ready to run on the next
   B200 rental (board node G, done).
4. **Zen4**: still blocked on stock. Check vast.ai again; if available,
   follow the exact procedure in "Critical operational notes" above (port
   wisdom, don't recalibrate, `OMP_NUM_THREADS=16` explicit).
5. **Regenerate `RESULTS.md` and re-sync the paper** once steps 2-4 land
   (or once it's clear Zen4 will stay blocked for a while and the user
   wants to proceed without it): every number, table, and plot in
   `RESULTS.md` should also be in the paper, in agreement. Recompile the
   PDF, copy into `paper/icm_paper.pdf`, commit.
6. **Standing, still open**: decide with the user whether to merge PR #7.
   Never auto-decide this.

## Process note

When a benchmark/calibration run touches a rented instance, log exactly
which script was invoked and when, cross-check the resulting data file's
mtime against the commit date of anything it depends on, and set an
explicit credit/time budget before letting anything run unattended.
Verify DeepSeek worker output by reading the actual diff, never by
trusting its own summary of what it did -- this session caught real bugs
(an infinite loop, a silent-zero sentinel gap, a wrong build target) that
the worker's own "complete and verified" framing did not disclose.
