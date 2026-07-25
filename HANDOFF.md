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

**Read "GPU OOM: what's fixed and what's still open" below before touching
anything GPU-related.** Two distinct OOM findings from the 2026-07-25 to
2026-07-26 sessions, both now fixed and hardware-verified (six stacked
commits total).

The repo is mid-cleanup after a long session that widened the B-selection
calibration methodology, then hit several real problems while trying to
verify and regenerate results from it. `bench_grid verify` and
`bench_gpu_fused verify` both pass cleanly as of the last commit (CPU and
GPU, confirmed on real M3 Pro and B200 hardware respectively). There is a
clear, agreed punch list to reach the final portfolio-ready state, see
**Next Steps** below, plus one newly-found open GPU issue that needs a
dedicated investigation before the GPU results can be called final. Read
the **Architecture: what's actually load-bearing** section before touching
any calibration/cost-model code, it's easy to misjudge what's safe to
change or delete without it.

## GPU OOM: history, both findings now fixed (2026-07-25 to 2026-07-26)

### Fixed, hardware-verified: the original n=2,097,152 k=256/512 OOM

Four stacked commits, each reviewed line-by-line before landing (see
`git log` for `8ce8a07`, `2f8fd22`, `66a2bc8`, `abd9fa4`):
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

Verified directly on a rented B200 (2026-07-25): `bench_gpu_fused verify`
passes 36/0 with the fix; the exact failing point (n=2097152, k=256 and
k=512) passes in an isolated repro, with debug output confirming
`cufft_ws=0.0 MB` for that specific case (every level in its tree is
FUSED-tier, so the fix means zero cuFFT plans are even created for it).

### FIXED 2026-07-26: long-sweep-only OOM

The full 211-point `heatmap_gpu` sweep (run 2026-07-25, after all 4 fixes
above and a correctly-ranged 67.1M-max calibration) failed on 21/211
cells, all at large n (2,097,152 - 33,554,432), consistently in a
middle-k band (e.g. at n=2097152: k=64 and k=128 pass, k=256/512/1024/2048
all fail, k=4096+ passes again -- same pattern repeats at n=4194304,
8388608, 16777216).

**Root cause**: a static-analysis pass (`scripts/DIAGNOSTIC_REPORT.md`)
traced every allocation/free pairing across the plan lifecycle and found
no genuine leak. Leading hypothesis: CUDA default-allocator fragmentation
after ~190 varying-size `cudaMalloc`/`cudaFree` cycles in one process --
consistent with every observed symptom (fails only after many cycles,
only at large n, passes in complete isolation). Two real bugs found in
this codebase's existing (but non-functional for the allocations that
matter) memory-pool support: the main arena allocation used a raw
`cudaMalloc` completely independent of `plan->use_async_pool` at every
setting, and a stray `plan->use_async_pool = false` disabled pooling for
the one allocation that could previously use it, regardless of
`memory_strategy`.

**Fix** (commit `386c856`): route the arena and shared cuFFT workspace
allocations through a CUDA stream-ordered memory pool
(`cudaMallocAsync`/`cudaFreeAsync`), remove the stray override, make
pooling the default (`memory_strategy=0`, matching PyTorch's CUDA caching
allocator and RAPIDS RMM's pool allocator both defaulting to pooling
rather than gating it to specific workload types), and add
`icm_gpu_release_pooled_memory()` as an explicit escape valve. Release
threshold set to `UINT64_MAX` (never auto-release) rather than a bounded
fraction of VRAM -- a 25% draft was tried and rejected during review
(178GB * 0.25 = ~48GB, smaller than several of the ~78-162GB arenas it
needed to protect), and the new escape valve makes unconditional
retention safe since a caller can explicitly reclaim memory when needed.

**Verified on real B200 hardware (2026-07-26)**: `bench_gpu_fused verify`
passes 36/0 with the fix (no correctness regression from the async
alloc/free changes). The full 211-point `heatmap_gpu` sweep now passes
with **zero errors**, including all 21 previously-failing cells --
confirmed directly against the exact failing points (n=2097152 k=256/512,
n=4194304 k=1024, n=8388608 k=64/262144, n=16777216 k=128), all now
returning real timing/memory data. See `results/gpu_heatmap_time.png` for
the visual (no gaps) and `results/gpu_heatmap_b200.csv` for the full data.

**Diagnosis trail** (kept for reference, not action items -- issue is
closed): confirmed NOT the same bug as the original OOM (every failing
point passed in isolation via `scripts/frontier_probe.cu`, meaning the
failure depended on long-running process state, not the (n,k) point
itself); ruled out a short 3-call sequential repro and an options/config
difference; deliberately did not patch reactively without a diagnosed
cause first, per explicit user instruction. Full detail in
`scripts/DIAGNOSTIC_REPORT.md` and git history for commit `386c856`.

The "1-second threshold" and `push_limit_gpu` frontier numbers in
`RESULTS.md` still need re-measurement (unrelated follow-up, not blocked
by this fix -- `tools/push_limit_gpu.cu`'s full exhaustive search was
separately found impractically slow, see item 4 in Next Steps below).

## Architecture: what's actually load-bearing (read before touching cost-model code)

Dispatch happens in three separate layers. Only two of them were ever the
"fragile aggregate formula" problem; the third was never broken and was
never replaced.

1. **Which engine?** (linear / hybrid / tree), `select_engine_ex()` in
   `src/icm.c`. For full-equity queries (`n_targets <= 0`), uses the
   empirical crossover table (`crossover_n[]`/`crossover_k[]` in
   `fft_config.h`, log-linear interpolation via
   `empirical_crossover_k()` in `src/fft_cost_model.h`). For **subset**
   queries (`n_targets > 0`), still uses the old summed-analytical
   formula, which is confirmed measurably wrong (37-45% slower than
   optimal at representative points), see Next Steps.
2. **Which block size B?** (only matters inside the hybrid engine) ,
   `select_best_B()` in `src/icm.c`. Uses the empirical `bselect` table
   (`bselect_n[]`/`bselect_k[]`/`bselect_B[]`, 2D nearest-neighbor via
   `empirical_best_B()`), built by `tools/calibrate_best_b.c` +
   `tools/calibrate_block_size.py`. GPU equivalent:
   `gpu_select_best_B_est()` in `src/gpu/gpu_plan.cu`, same mechanism,
   `gbselect_*` tables from `tools/calibrate_gpu_best_b.cu`.
3. **Inside the tree/hybrid engine, at every single tree level, every
   single call**: schoolbook or FFT? If FFT, which calibrated size?
   `best_fft_config()` / `best_fft_config_joint()` in
   `src/fft_cost_model.h`, called directly from `tree_ctx_create_ex2()`
   (`src/icm.c:1192,1239,1243,1271`). Reads `calib_times_ns[]` /
   `calib_sizes[]` (the 749-entry per-FFT-size table from
   `tools/calibrate.c`, real FFTW PATIENT measurements) and
   `schoolbook_mul_ns[]` directly, picking whichever calibrated size (or
   schoolbook) minimizes real measured cost for that exact convolution
   length.

**Layer 3 was never the problem and was never replaced with a lookup
table.** It doesn't need to be: it already compares real measured
per-size timings directly, not summed abstract constants. Only layers 1
and 2 summed many individually-correct constants into one aggregate
go/no-go decision, and that's what turned out to be fragile in aggregate
and got replaced this session (layer 2) and the session before (layer 1,
full-equity only). `tools/calibrate.c`'s output (`calib_times_ns[]`,
consumed by layer 3) is genuinely foundational and still load-bearing on
every single computation, it is not dead code, and must never be
skipped when porting to a new device.

## Critical operational notes

**DeepSeek Deck network access.** The `deck` binary on `$PATH` resolves
to a stale plugin-cache copy
(`~/.claude/plugins/cache/deepseek-deck/deepseek-deck/1.0.0/bin/deck`)
that does **not** support `--allow-network`, its argparse simply doesn't
have the flag, even though the underlying worker sandbox (`deepseek-mcp`
tools.py) supports network toggling. The **real, working** dev copy with
`--allow-network` wired all the way through is at
`~/Documents/deepseek-deck/bin/deck` (confirmed: same daemon serves both
paths, since the CLI is just a thin HTTP client to a local daemon on port
8787, the daemon that's actually running is built from the dev repo, so
using the dev repo's CLI binary is what actually matters). **Always
invoke `/Users/samrosenstrauch/Documents/deepseek-deck/bin/deck` directly
(or alias it) when a node needs `--allow-network`**, not whatever
resolves from `$PATH`.

**Zen4 rental.** As of this writing, the AMD Ryzen 9 7950X-class instance
is completely out of stock on vast.ai, and there is no clean hourly-billed
alternative, checked Cherry Servers (right billing model, hourly, but
currently 0 in stock at all 6 locations), Hetzner (has 7950X/7950X3D but
monthly billing only, poor fit for a few hours of verification work).
Check vast.ai stock again before assuming it's still unavailable; if
using a different provider, note the billing model may force a different
workflow (can't just spin up/destroy in an hour).

**When a Zen4 box is available again**: do NOT re-run the adaptive
B-selection calibration (already correct, already committed, 1944
points) and do NOT rebuild AOCL-FFTW wisdom from scratch. Copy
`devices/zen4/fftw_wisdom.dat` directly onto the new box (byte-identical
port, verified working this exact way earlier this session), build
AOCL-FFTW from source with the full flag set (`--enable-sse2 --enable-avx
--enable-avx2 --enable-avx512 --enable-amd-opt`, needs `texinfo` package
installed first or the docs sub-build fails), then rebuild the ICM
binaries against the already-correct config. The ONLY thing that needs
re-running there is the benchmark sweep itself.

**`make results-refresh`'s parallel-binary gotcha.** The recipe ends by
leaving the OpenMP-enabled `bench_grid` binary sitting in the working
directory (it runs `make all` -> serial grid -> `make parallel` -> parallel
grid, and both targets build the same output file name). If you
manually run `./bench_grid ...` afterward expecting serial behavior,
you'll silently get parallel timing instead. Always check for "OpenMP
enabled: N threads" vs "OpenMP disabled (serial mode)" in the output, or
just `make clean && make` before any manual serial probe. This caused a
real, hours-long false alarm this session (a "3x anomaly" that was
actually just this binary mix-up).

**Zen4 needs `OMP_NUM_THREADS=16` explicitly**, never the default
`nproc` (32, SMT-inclusive), SMT siblings add no real throughput for
this FPU/vector-port-bound workload. `make results-refresh` does NOT set
this for you; you must prefix it (`OMP_NUM_THREADS=16 make
results-refresh DEVICE=zen4`) or the parallel grid silently corrupts at
small `n` (oversubscription contention, up to 20x+ slower than serial in
some cells, looks exactly like a real regression until you check thread
counts).

**B200.** As of this writing, no instance is rented (destroyed after the
last session, standard practice). `results/gpu_heatmap_b200.csv` (dated
2026-07-21) **predates** the GPU B-selection empirical-table fix
(`src/gpu/gpu_plan.cu`, commit `b581dab`, dated 2026-07-23), meaning the
current committed B200 numbers in `RESULTS.md` and the paper reflect the
OLD buggy dispatch (which picked B=128 when B=64 was optimal, 2-4%
slower). This must be regenerated. Do NOT re-time the individual FFTs or
re-run the GPU FFT calibration (`calibrate_gpu.cu`) or the B-selection
adaptive calibration, all already correct and committed. The only thing
that needs re-running is the benchmark sweep: `tools/heatmap_gpu.cu`
(systematic grid, feeds `tab:gpu`/`fig:gpu-contour`) and
`tools/push_limit_gpu.cu` (frontier probes, feeds `tab:gpu-frontier`).

## Dead code, confirmed by tracing actual usage (not guessed)

These are leftover from the analytical-cost-model-fitting era and the
now-closed crossover investigation. None are invoked by
`tools/calibrate_full.sh`, any Makefile target, `results-refresh`, or
`calibrate_block_size.py`. Confirmed by reading each file's own header
comment, which in every case names its own successor:

- `tools/bench_leaf_fma.c`, superseded by `calibrate_leaf_realistic.c`
  per its own successor's docstring.
- `tools/calibrate_leaf_realistic.c`, itself superseded (found buggy:
  reused one `HybridCtx` across reps, unrealistically cache-hot) by
  `tools/probe_leaf_extract.c`, which IS in active use.
- `tools/b_optimal_sweep.c`, "validates `select_best_B()` against
  measured optimum" by sweeping every B; fully superseded by
  `calibrate_best_b.c` + `validate_best_b.c`.
- `tools/eval_model_vs_plans.c`, evaluates the old summed-analytical
  formula, which layers 1/2 above no longer use.
- `tools/quantify_dispatch_gap.c`, one-off diagnostic, own header says
  "quantify remaining dispatch-point gap after schoolbook fix"; that
  investigation is closed.
- `tools/probe_tree_levels.c`, compares against the old formula from
  `select_engine_ex()`; also flagged in this file's own prior history as
  having produced an invalid finding due to a stale-formula bug.

Their stale output files (`results/b_optimal_sweep_zen4.csv`,
`results/B_probe_zen4.txt`) should go with them.

Also: `CLAUDE.md`'s directory listing references `tools/gen_gpu_calib_lib.py`,
which does not exist anywhere in `tools/`. Fix or remove that line.

**Not dead, despite superficial similarity**: `gpu_phase_profile.cu`
(feeds `fit_gpu_cost_model.py`'s constants for GPU engine/tier selection,
a separate mechanism from B-selection that was never replaced),
`test_cpu_cost_model.c` / `test_gpu_cost_model.cu` (test structural
invariants of code still in active use, not the abandoned formula),
`profile_harness.c` (generic ad-hoc profiling utility).

## What Worked

- **Tracing exact code usage instead of guessing** when asked "is this
  dead code" or "is this still load-bearing", reading each file's own
  header comment and grep-ing actual call sites resolved every question
  definitively, no hand-waving needed.
- **Cross-referencing commit dates against data file mtimes** to catch
  the B200 staleness bug (`git log -1 --format=%ci <commit> -- <path>`
  vs `ls -la` on the data file), this is how the true "B200 predates its
  own dispatch fix" finding was actually confirmed, not asserted.
- **Direct reproduction of anomalies before believing them.** The
  "3x variance" scare traced to a real, findable cause (serial/parallel
  binary mix-up) once actually investigated with `ps`/build-flag checks,
  rather than accepted as unexplained noise.
- **Using `git-filter-repo` for a clean commit-message rewrite**, once
  scoped correctly (see below), it does exactly what's needed with
  verifiable output (message content changed, tree content unchanged,
  confirmed via `git diff <old> <new> --stat` being empty).

## What Didn't Work / mistakes to avoid repeating

- **`git filter-repo --message-callback` without scoping the commit
  range rewrites the ENTIRE reachable history, not just the commits you
  intend to touch.** This gave every commit back to the beginning of the
  branch a new SHA (even ones with no message change needed), breaking
  the shared ancestry with `origin/main` and turning a clean PR into one
  GitHub reported as `CONFLICTING`. The fix: build the corrected branch
  by checking out the known-good base commit, cherry-picking just the
  commits that need editing, and amending each one individually (`git
  rebase <base> --exec <script>` works non-interactively and only
  touches commits after `<base>`). Verify with `git merge-tree
  $(git merge-base origin/main HEAD) origin/main HEAD` before pushing ,
  zero `<<<<<<< .our` lines means genuinely no conflict, not just no
  visible diff.
- **Ad-hoc single-rep manual probes (`./bench_grid bench <n> <n> 1`)
  outside the established sweep tools produce noise, not signal**, and
  burn paid-instance time for no committed benefit. The project has
  exactly two canonical tools for producing `RESULTS.md`/paper numbers:
  `bench_grid` (full grid, no subcommand) and `tools/contour_1s.c`
  (`--contour` mode). There is also `bench_grid threshold`, a real
  binary-search tool for the precise `k=n` 1-second boundary specifically
 , use that instead of eyeballing an interpolation or manually probing,
  next time this number is needed.
- **A crude 2-point linear interpolation across a wide n-range
  (e.g. n=16,384 to n=32,768) systematically overshoots a superlinear
  timing curve.** The "n≈29,000" Zen4 1-second-threshold figure currently
  in `RESULTS.md` was computed this way and is measurably wrong (real
  direct probes near n=26,000-27,000 show the true crossing is close to
  there, not 29,000). Use `bench_grid threshold` for this number, not
  interpolation.
- **Asserting "nothing changed so the old data is still valid" without
  checking commit dates against data file dates.** Told the user the
  B200 numbers were fine because "nothing GPU-related changed this
  session", wrong, the B-selection fix landed earlier the same day,
  before this session's board even started, and the heatmap was never
  re-swept after it. Always check dates, don't reason from session
  boundaries.
- **Committing without checking for standing rules first.** Added
  `Co-Authored-By`/`Claude-Session` trailers to every commit this
  session, violating an explicit standing rule already in this file's
  own history ("No Co-Authored-By trailers on any commit, ever"). Check
  this file's own accumulated rules before the first commit of a
  session, not after.

## Next Steps

Ordered roughly as agreed with the user; items are independent except
where noted.

1. **Point subset-query dispatch at the existing empirical `bselect`
   table** instead of the known-broken analytical formula, as an
   interim improvement (not the fully-correct fix, which would need its
   own `(n, target_frac) -> crossover_k` calibration, that's out of
   scope here, flagged as a possible future board). This is a small,
   low-risk patch to `select_engine_ex()`'s subset-query branch in
   `src/icm.c`. Verify with the same real-measurement methodology C4
   used (median-of-7, real `icm_equity_subset()` calls) at the same
   representative points already on record (37%/45% gaps) to confirm
   the gap shrinks.
2. **Delete the 6 confirmed-dead files** listed above, their stale
   output CSVs, and fix the `gen_gpu_calib_lib.py` doc reference in
   `CLAUDE.md`. Rebuild + `bench_grid verify` afterward to confirm
   nothing was actually depended on.
3. **Zen4**: get a fresh box (stock permitting), port wisdom directly
   (no regeneration), build AOCL-FFTW with correct flags, rebuild
   against the already-correct committed B-selection table, run ONLY
   `make results-refresh DEVICE=zen4` (with `OMP_NUM_THREADS=16`
   explicit) to refresh `results/`. Also run `bench_grid threshold` for
   a precise, non-interpolated `k=n` 1-second boundary.
4. **B200**: DONE, both the original and a second OOM found while
   verifying it are now fixed and hardware-confirmed (see "GPU OOM:
   what's fixed and what's still open" above -- 6 commits total,
   `8ce8a07`..`386c856`). The full 211-point heatmap now passes with
   zero errors. Remaining, smaller follow-up (unrelated to either OOM
   fix): `tools/push_limit_gpu.cu`'s full exhaustive B/M/T search was
   measured as impractically slow (>3.5 min on just its first, smallest
   n value, no early-exit) and was not run to completion; a targeted
   auto-dispatch probe (`scripts/frontier_probe.cu`) was used instead
   for the 5 `RESULTS.md` frontier points, but this is not equivalent to
   the original tool's methodology. Follow-up needed: (a) either speed
   up `push_limit_gpu` or budget real time for it, (b) redo the
   1-second threshold binary search properly.
5. **De-slopify all MD files and the two files C3's cleanup pass never
   touched** (`tools/gen_calib_skeleton.py`, `tools/calibrate_block_size.py`).
   227 em-dashes were counted across `*.md` files this session (paper
   itself is already clean, 0 em-dashes), strip them to standard
   punctuation, and check for any remaining session-narrative/AI-tell
   comments in code.
6. **Regenerate `RESULTS.md` and re-sync the paper** once steps 3-4 land:
   every number, table, and plot in `RESULTS.md` should also be in the
   paper, in agreement, nothing stale in either. Recompile the PDF, copy
   into `paper/icm_paper.pdf`, commit.
7. **Standing, still open**: decide with the user whether to merge PR #7.
   Never auto-decide this.

## Process note

When a benchmark/calibration run touches a rented instance, log exactly
which script was invoked and when, and cross-check the resulting data
file's mtime against the commit date of anything it depends on (dispatch
tables, cost-model code) before trusting it's current. This session lost
significant time and trust to two variants of the same root problem:
assuming data was current without checking dates against what actually
changed.
