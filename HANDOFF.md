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
verification before anything is called done. That pattern is how the
GPU OOM chain below got found, fixed, and verified across several rounds
without losing track of state -- keep doing it that way, don't drop to
ad-hoc work.

**First priority: a cleanup/audit pass.** The user flagged (2026-07-26)
that repo state may have drifted from what was explicitly asked for
across the last few sessions. Concrete, CONFIRMED drift found so far
(don't re-derive, these are checked, not suspected):

1. **The paper (`~/Documents/ICM_paper`) is stale.** Last commit
   `98b7f79`, 2026-07-24 01:41 -- before the subset-dispatch fix, the dead
   code deletion, both GPU OOM fixes, and the regenerated heatmap data.
   `RESULTS.md` has changed substantially since; the paper has not.
   Violates this file's own standing rule ("must never diverge"). Needs a
   full re-sync pass once the remaining GPU numbers below are finalized.
2. **`src/icm_gpu.h`'s `memory_strategy` doc comment is stale.** Still
   reads `/* 0=auto, 1=full, 2=pool, 3=selective recompute */` -- as of
   commit `386c856`, `0=auto` now ALSO enables the memory pool by
   default (matching what used to be `2`'s exclusive behavior), and `1`
   is now specifically the pool opt-out. The comment no longer describes
   what the values actually do. Needs a rewrite.
3. **`CLAUDE.md`'s Directory Structure section doesn't mention
   `scripts/`.** That directory (created 2026-07-25/26) holds
   `gpu_ws_repro.cu`, `frontier_probe.cu`, `b200_verify_and_sweep.sh`,
   `heatmap_gpu_reset_every_cell.cu`, `DIAGNOSTIC_REPORT.md`. Decide
   which of these are still earning their place (see item 4) and
   document the survivors.
4. **Judgment call needed, not yet made**: now that the long-sweep OOM is
   fixed, is `scripts/DIAGNOSTIC_REPORT.md` (a diagnosis of a now-closed
   issue) and `scripts/heatmap_gpu_reset_every_cell.cu` (a diagnostic
   tool for a hypothesis that's now confirmed) still worth keeping, or
   are they dead weight per this project's own established convention of
   deleting superseded investigation tools (see the precedent: 6 files
   already deleted this way, commit `318a3ed`)? Lean toward keeping
   `DIAGNOSTIC_REPORT.md` as a permanent record (it's genuinely
   informative, unlike the old one-off dead tools) but this is worth a
   real decision, not a default.
5. **Verify nothing else in `RESULTS.md`/paper references numbers that
   predate the 2026-07-26 fixes.** Specifically check the CPU-side
   sections too, not just GPU -- the M3 Pro data (see below) checked out
   fine, but do a full read-through, not a spot check.

**M3 Pro data status, already checked, do not re-derive**: `results/*m3pro*`
files (bench grids, contour sweeps, plots) were last regenerated in
commit `a05652e`, ~11 minutes before the subset-dispatch fix (commit
`44bc959`) landed the same session. Checked whether this matters: the
subset-dispatch fix only touches `n_targets > 0` paths, and the only
"subset" content in the M3 Pro files is a single pre-existing correctness
check line (`subset n=1024 10/1024 PASS`), not a performance number --
`bench_grid`'s `subset-speed` performance benchmark was never run/included
in these files at all. **Conclusion: M3 Pro data is NOT stale relative to
the subset fix.** Both files show `ALL TESTS PASSED`. Nothing funky found.
This does not need re-verification unless something else changes CPU
dispatch code again.

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

**Next action**: write a real binary search using the lightweight
single-measurement approach `frontier_probe.cu` already established (one
`icm_gpu_equity` call per candidate n via `icm_gpu_plan_create` +
`icm_gpu_equity_with_plan`, not the full B/M/T sweep) -- narrow in on
where k=n crosses 1000ms and separately where k=100 crosses 1000ms, same
two curves as the original `RESULTS.md` table. Should take a few minutes
on a B200, not the 3.5+ minutes-per-n the exhaustive tool needs. Not yet
written as of this handoff.

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

1. **Cleanup/audit pass** (see START HERE above for the specific,
   already-confirmed items -- don't re-derive, just fix): resync the
   paper, fix the `memory_strategy` doc comment, document `scripts/` in
   CLAUDE.md, decide on keeping vs. archiving the diagnostic-only files,
   do a full read-through of `RESULTS.md` for any other stale numbers.
   Good candidate for a DAG wave with DeepSeek doing the mechanical
   doc-sync work and supervisor reviewing before commit, same pattern as
   the rest of this session.
2. **Write and run the real 1-second-threshold binary search** (see "GPU
   OOM... Still open" above for the exact approach). This needs a fresh
   B200 rental. Verify against the existing `frontier_probe.cu` points as
   a sanity check, then produce the actual threshold numbers for both
   k=n and k=100 curves.
3. **Zen4**: still blocked on stock. Check vast.ai again; if available,
   follow the exact procedure in "Critical operational notes" above (port
   wisdom, don't recalibrate, `OMP_NUM_THREADS=16` explicit).
4. **Regenerate `RESULTS.md` and re-sync the paper** once steps 2-3 land
   (or once it's clear Zen4 will stay blocked for a while and the user
   wants to proceed without it): every number, table, and plot in
   `RESULTS.md` should also be in the paper, in agreement. Recompile the
   PDF, copy into `paper/icm_paper.pdf`, commit.
5. **Standing, still open**: decide with the user whether to merge PR #7.
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
