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
anything GPU-related.** It has two distinct findings from the 2026-07-25
session: one fully fixed and hardware-verified (four stacked commits), one
genuinely NOT fixed and not yet understood. Do not conflate them.

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

## GPU OOM: what's fixed and what's still open (2026-07-25)

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

### NOT fixed, newly found, root cause unknown: long-sweep-only OOM

The full 211-point `heatmap_gpu` sweep (run 2026-07-25, after all 4 fixes
above and a correctly-ranged 67.1M-max calibration) failed on 21/211
cells, all at large n (2,097,152 - 33,554,432), consistently in a
middle-k band (e.g. at n=2097152: k=64 and k=128 pass, k=256/512/1024/2048
all fail, k=4096+ passes again -- same pattern repeats at n=4194304,
8388608, 16777216). See `results/b200_2026-07-25/heatmap_full.log` for
the full log and `results/gpu_heatmap_time.png` for the visual (white
gaps in a clean column band).

**Confirmed NOT the same bug as above**: every one of the 21 failing
(n,k) points, when re-run standalone in an isolated process
(`scripts/frontier_probe.cu`), passes cleanly -- including the exact
points that failed in the sweep (n=2097152 k=256, n=4194304 k=512,
n=8388608 k=128, n=16777216 k=128, all confirmed individually). This
means the failure depends on the LONG-RUNNING PROCESS STATE from many
prior sequential plan create/destroy cycles within `heatmap_gpu`'s single
process, not anything about the (n,k) point itself in isolation.

**Ruled out this session** (don't re-check these, move on to new
hypotheses): a short 3-call sequential repro (k=64 -> 128 -> 256 at
n=2097152, mimicking the exact lead-up to the first failure) does NOT
reproduce it -- so it's not "any prior allocation at all," it needs
something closer to the full ~190-cell sequence before it manifests.
`IcmGpuOptions` are identical between the isolated repro and the sweep
tool (checked directly, both use `memory_strategy=0`,
`force_uncached_*_levels=-1`), so it's not an options/config difference.

**Not attempted this session, deliberately**: no live patch was applied.
The user explicitly said not to reach for a reactive/hacky fix here (per
the same principle that made the retry-only version of the OOM fix above
unacceptable as a *primary* mechanism). This needs real diagnosis before
a fix is written -- candidate directions for a future session:
- CUDA's default allocator (not the async pool -- the main arena uses
  plain `cudaMalloc`) fragmenting after many large alloc/free cycles of
  varying sizes in one process. Test by inserting `cudaDeviceReset()` +
  `icm_gpu_init(0)` between every cell (not just failures) in a
  diagnostic copy of `heatmap_gpu.cu` and seeing if the failures
  disappear -- if so, that's strong confirmation, and the real fix would
  be understanding WHY the allocator can't reclaim/coalesce rather than
  just resetting every cell as a workaround.
- Check for an actual leak (not just fragmentation) in
  `icm_gpu_plan_destroy()` -- does every allocation made during
  `allocate_plan_device_memory()` get freed, including anything the G4
  lazy-fallback path (`ensure_cufft_plans_for_level`) might allocate on
  the rare cells where it fires?
- Instrument with `ICM_GPU_DEBUG_PLAN=1` across a long real sequence
  (not just the failing cell in isolation) to see if `current_vram_bytes`
  / actual free VRAM (`cudaMemGetInfo`) drifts from what the plan
  accounting believes it is over many cells.

This is why the "1-second threshold" and `push_limit_gpu` frontier
numbers in `RESULTS.md` are marked as needing re-measurement rather than
finalized -- until this is understood, any long-running sweep's numbers
near the affected n/k range should be treated with appropriate caution.

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
4. **B200**: DONE 2026-07-25, but with an open follow-up. Rented an
   instance, found and fixed a real GPU OOM bug (4 stacked commits, see
   "GPU OOM: what's fixed and what's still open" above), regenerated
   the FFT+workspace calibration (67.1M max size, matching the
   previously-validated range) and the full 211-point heatmap. 190/211
   cells succeeded cleanly; 21 failed with a newly-found, NOT-YET-FIXED
   issue that only manifests in the long-running sweep, not in
   isolation (full details + candidate diagnosis directions in the
   section above). `tools/push_limit_gpu.cu`'s full exhaustive B/M/T
   search was measured as impractically slow (>3.5 min on just its
   first, smallest n value, no early-exit) and was not run to
   completion; a targeted auto-dispatch probe
   (`scripts/frontier_probe.cu`) was used instead for the 5
   `RESULTS.md` frontier points, but this is not equivalent to the
   original tool's methodology. Follow-up needed: (a) diagnose and fix
   the long-sweep OOM, (b) either speed up `push_limit_gpu` or budget
   real time for it, (c) re-run the full heatmap once (a) is fixed to
   fill the 21 missing cells, (d) redo the 1-second threshold binary
   search properly.
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
