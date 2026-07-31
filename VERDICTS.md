# VERDICTS.md

Audit log of decisions this project has made, and why. **Do not relitigate an
entry here without new evidence.** If you overturn one, edit the entry in place,
record the date and the evidence that overturned it, and keep the old reasoning
visible.

Every entry cites where the decision is substantiated. An entry with no
citation is explicitly marked as such and must be treated as unverified.

---

## V1. Empirical calibration tables replace analytical cost models (CPU dispatch)

**Decided:** 2026-07-22. **Evidence:** commits `27cc356` (crossover), `c70ca4e` (B).

Engine dispatch (`select_engine_ex`) and block-size selection (`select_best_B`)
both dropped summed-analytical-cost formulas in favour of empirically measured
lookup tables.

**Why:** every individual constant in the old formulas was validated against
real embedded execution, and the aggregate comparison *still* did not match the
measured crossover on either M3 Pro or Zen 4. Measured penalty before the fix:
7-11% on M3 Pro (12/19 points wrong), 2-9% on Zen 4. This is a known result in
the autotuning literature; see V2.

## V2. The precedent this design follows

**Evidence:** `COST_MODEL_EXPLAINED.md` section 4; paper sections 3.5 and 5.x.

FFTW (`ESTIMATE` vs `MEASURE`/`PATIENT`), ATLAS (AEOS), BeBOP/Sparsity
register-blocking search (Demmel/Dongarra/Whaley 2004 section 4.2), and LAPACK
`ILAENV` `ISPEC=3` (`NX`). The last is the structural precedent: a problem-size
crossover measured empirically per machine, consulted as a cheap runtime
threshold, with no live racing in production.

Note that none of these extrapolate outside their tuned range. `ILAENV` returns
a fixed default. That is the basis for V8.

## V3. Layer 3 was never the problem and was never replaced

**Evidence:** `HANDOFF.md` architecture section; `src/cpu/fft_cost_model.h`.

Per-level schoolbook-vs-FFT and FFT-size selection (`best_fft_config`,
`best_fft_config_joint`) always compared real measured per-size timings from
`calib_times_ns[]`. Only the two aggregate-formula layers were fragile. Do not
"modernize" layer 3; it is already the thing the other layers were changed into.

## V4. B is discrete: nearest-neighbour lookup, never interpolation

**Evidence:** `c70ca4e`; `COST_MODEL_EXPLAINED.md`.

B is chosen from a fixed candidate set. Interpolating would produce values like
"B=48.7" which do not exist. The crossover k, by contrast, *is* continuous and
*is* log-linearly interpolated. The two decisions are deliberately different.

## V5. GPU B-selection uses an empirical table, same as CPU

**Decided:** 2026-07-23. **Evidence:** commit `b581dab`.

`validate_planner_gpu.cu` on real B200 showed 12/12 mismatch before the change
(the analytical model always chose B=128 where B=64 measured faster) and 12/12
match after.

## V6. GPU calibration scope is deliberately targeted, not a full sweep

**Decided:** 2026-07-27. **Evidence:** commit `b06379e`.

The GPU B-selection table is **not** a full adaptive sweep like the CPU tables.
Above the k=n frontier (n=1,572,864) it uses **targeted anchors**: 13 points at
n=2M/4M/8.4M (k = n/8, n/4, n/2, n) plus n=16.7M (k=n only), measured with
`calibrate_gpu_best_b.cu --narrow-around 96,112,128,144` rather than sweeping
all 48 candidates. **All 13 converged uniformly on B=128.**

**Why:** a full sweep is prohibitively expensive at these sizes. Measured from
the 2026-07-30 heatmap, a full 48-candidate sweep to n=33.5M would cost roughly
4.5 hours and $28 before adaptive refinement, and n=33,554,432 alone is 54% of
total heatmap compute time. The targeted anchor approach cost a fraction of that
and answered the question.

**Important:** the full calibration tooling still exists and works for third
parties (`calibrate_gpu_best_b.cu` + `calibrate_block_size.py` support
point-list CSVs, single-point probes, `--narrow-around`, and resumability).
Only *this project's own B200 data* is a deliberate targeted subset. Say so in
any writeup; do not imply the GPU received the CPU's full adaptive treatment.

**Correction, 2026-07-30 (same day as V7's two-round anchor fill).**
`calibrate_block_size.py` had apparently never actually been run for this
project's own B200 data (its adaptive Step 4 loop, if run, would very
plausibly have caught V7's regressions itself via random per-band sampling,
rather than needing a human-driven audit to find them). Reviewing it while
answering "what's the reproducible way to land these points" found a real
bug: Step 4's refinement re-called `calibrate_gpu_best_b --narrow-around
auto_B` on a gap, which (a) re-paid for a full candidate search that
`validate_planner_gpu`'s probe had *already done* to compute the same
point's `best_B` (used to derive `gap_pct` in the first place), and (b)
narrowed around the dispatch's own already-wrong answer, which can miss a
true optimum more than one candidate step away -- exactly what V7 observed.
Fixed to use the probe's own `best_B` directly: no second binary call, no
narrowing, no guessing. `tools/splice_calib_points.py` (new) is now the
reproducible way to land a `--narrow-around` measurement into a config
header, replacing hand-edited C arrays; `--skip-base-sweep` and `--dry-run`
were added to `calibrate_block_size.py` so a caller can see cost (skeleton
size, worst-case probe count) before spending any GPU time, and choose
statistical-sampling coverage over a full skeleton sweep as an explicit,
transparent tradeoff rather than an implicit one. See the tooling commit
for the full detail. Not yet run for real on B200 (would need funding);
this is a tooling fix, not a claim that V7's remaining cell is now closed.

**Superseded, same day.** The user asked for a fully non-discretionary,
one-command production tool -- patching `calibrate_block_size.py`'s
Step 2/Step 4 split still left real hand-tuned knobs (`--skeleton-lo/hi`,
`--clean-streak-target`, `--max-probes-per-band`) a fresh device port
would have to guess at. Replaced entirely with
`tools/calibrate_adaptive.py` (+ shared `tools/calib_common.py`):
priority-queue refinement scored by the same joint-log-distance metric
the production lookup itself uses (plus a disagreement-with-neighbors
bonus), a single required `--budget` knob (probe count or wall-clock
time), and a convergence stopping rule (top-of-queue score below
threshold on a fresh sample) instead of per-band random walks.
`calibrate_block_size.py` is deleted. `gen_calib_skeleton.py`'s GPU
domain default was also fixed while doing this (was `4194304`, real
GPU usage per `heatmap_gpu.cu` goes to `33554432`) -- an 8x undershoot
that had been silently baked into every prior calibration default.
`--n-min`/`--n-max` (named `--skeleton-lo`/`--skeleton-hi` until a
same-day rename, see below) remain as an explicit opt-in narrowing
(used for this project's own targeted fixes); the unscoped default is
the full domain, so a fresh device port gets real coverage with zero
required judgment calls. See the tooling commit for the full design
rationale.

**First real run, 2026-07-31: found a serious design flaw, reverted,
nothing shipped.** Ran `calibrate_adaptive.py --device b200
--skeleton-lo 1024 --skeleton-hi 524288 --budget 20m` for real (contract
`46358305`) -- exact command as typed at the time; the flags were
renamed to `--n-min`/`--n-max` immediately afterward, see the
flag-rename entry below. It worked as designed and found far more real disagreement
than expected: 520 probes in 20 minutes, most landing well off the
standard heatmap grid (random log-uniform points like n=1980, n=90283,
n=411295 -- points nobody had ever measured before), many with large
gaps (39-87%). This itself is a genuine, valuable discovery: the old
table was only ever validated against a fixed grid of "nice" n values;
between those points its true accuracy had never been measured and
turns out to be much worse than assumed.

**But applying all 520 fixes together made the table measurably worse
on the standard grid it's actually judged against.** A `heatmap_gpu
--fast` diff (n=64-524288, the grid `bench_gpu`/`heatmap_gpu` actually
test) against the pre-run baseline showed 36 regressions out of 84
cells, several catastrophic (n=1024,k=64 B64->384 +96%; n=16384,k=64
B64->1024 +260%; n=2048,k=512 B64->1280 +281%), and the grid's total
time got **worse**, not better (+4.57% vs the pre-run baseline).
Independently re-confirmed two of the worst with direct, freshly-built
`validate_planner_gpu` probes to rule out a stale-binary artifact (there
was actually a real one mid-session -- see below -- but it does not
explain this): `n=2048,k=512` now dispatched B=1280 when direct
measurement said B=64 is right (gap 292.75%); `n=16384,k=64` dispatched
B=1024 against a true B=48 (gap 265.32%). Both are real, not measurement
noise, and both are much worse than what the *unmodified* baseline
already had at those same cells (63.04% and 7.31% gaps respectively --
the baseline was already imperfect there, but nowhere near this badly).

**Root cause: pure joint-log-distance nearest-neighbor is not a safe
table-construction strategy for this function.** Two compounding
factors, both real: (1) `B*(n,k)` is far noisier / more locally variable
than the "smooth regions with occasional cliffs" model implicit in
nearest-neighbor lookup assumes -- confirmed by how much real
disagreement 520 essentially-random samples turned up. (2)
`validate_planner_gpu`/`validate_best_b` measure with a single rep, no
repeats -- confirmed contaminating results directly: one probe during
the run reported `auto_B=64, best_B=64, gap=21.44%` for the *same* B,
which can only be timing noise between two separate single-shot runs.
Combined, a single new point -- possibly itself noisy -- can become the
nearest neighbor for many nearby *unmeasured* queries and silently pull
them to a worse answer, exactly the V7 failure mode, but now happening
at the scale of hundreds of essentially-random points instead of a
handful of deliberately-chosen ones.

**Reverted in full.** `devices/b200/gpu_fft_config.h` was restored to
the exact committed state (93 points, `md5 d23ad491...`) before
rebuilding and destroying the instance. Nothing from this run shipped,
including the one manually re-derived point (`65536,2048,B=80`) that
*was* correctly measured -- it's real and could be re-added on its own,
but landing it alone, separately from the broken bulk run, was judged
not worth a special-cased partial commit tonight. `results/gpu_heatmap_
b200_20260731_adaptive.csv` and `..._final.csv` are kept as evidence of
this finding, not as usable calibration data -- do not treat either as
a baseline.

**Also found and fixed mid-session (operational, not a data problem):**
`validate_planner_gpu`'s default binary path
(`./build/validate_planner_gpu`) doesn't match where the Makefile
actually places it (repo root, `./validate_planner_gpu`); worked around
with an explicit `--validate-bin` flag, needs a real default-path fix
(see the flag-rename item below, batch it in). Separately, a real
stale-binary trap: rebuilding one target (e.g. `make validate_planner_
gpu`) does NOT relink other already-built executables (`heatmap_gpu`)
even when their shared `.o` dependencies changed underneath them --
always rebuild every binary you're about to run after any config
change, not just the one you think you need.

**Do not run `calibrate_adaptive.py` again as a table-construction tool
until this is fixed.** It remains an excellent, honest *discovery* tool
(the disagreement rate it found is itself useful signal about the old
table's real coverage gaps). As a tool that *writes* the live table, it
needs at least one of: multi-rep/median measurement in
`validate_best_b`/`validate_planner_gpu` to kill single-shot noise, and
a smarter merge strategy than raw nearest-neighbor (e.g. require a
confirmation probe on nearby *existing* points before accepting a new
one, or a distance-weighted local fit instead of pure NN) so one new
point can't silently override answers for queries it was never actually
tested against.

**Renamed, same session:** `--skeleton-lo`/`--skeleton-hi`/
`--skeleton-ratio` -> `--n-min`/`--n-max`/`--landmark-ratio`. The old
names leaked the tool's own internal implementation detail (they only
made sense once you knew Step 1 calls `gen_calib_skeleton.py`
internally); the new names describe what the flags actually mean to a
caller. Also fixed while in there: the default `--validate-bin` path
pointed at `./build/validate_planner_gpu`/`./build/validate_best_b`,
which doesn't exist -- the Makefile places these at the repo root.
Found this the hard way mid-run today (see above); worked around with
an explicit `--validate-bin` flag at the time, now fixed at the source.

## V7. The GPU anchors were co-designed with the sequential lookup (and V11 broke that) (OPEN: down to 1 known regression from 16, chasing to zero has diminishing returns)

**Recorded:** 2026-07-30. **Evidence:** `b06379e` read against the 2026-07-30
heatmap comparison.

The V6 anchors were placed so that a **nearest-n-first** lookup would land on
them and return B=128 above the frontier. When the lookup changed to joint
(n,k) nearest neighbour (V11), n-locality was no longer respected, so large-n
queries began matching smaller-n points carrying smaller B. Result: 12 cells
regressed, every one of them `B 128->64` or `128->48`, all above n=1,572,864.

**Partially closed, 2026-07-30.** `calibrate_gpu_best_b --narrow-around
96,112,128,144` measured all 12 regressed cells directly on a rented B200
(contract 46316595) and the results were spliced into
`devices/b200/gpu_fft_config.h` (`GPU_N_BSELECT_POINTS` 60 -> 72). Mostly
confirms B=128 as the correct anchor, but not uniformly: n=4194304,k=1024
measured B=112 and n=8388608,k=128/1024 measured B=80/96, so this is real
per-point data, not an assumption that 128 is always right above the
frontier.

**Confirmed clean with the new anchors, same box, same session:**
`bench_gpu_fused verify` (0 FAIL) and `test_gpu_cost_model` (483 passed, 5
failed, the same 5 pre-existing failures as before the fix, all below the
frontier and already explained in HANDOFF.md -- no new failures from the
new anchors).

**Second attempt, 2026-07-30 (later the same day, after a vast.ai top-up).**
Rented a second B200 (contract `46322663`, different host than the first)
and reran the full 210-cell heatmap regen to completion:
`results/gpu_heatmap_b200_20260730_postfix2.csv`. This run also carries the
V16 `select_best_B`/`gpu_select_best_B_est` fix (same commit), so it
validates both fixes together.

It sailed cleanly through n=33,554,432 at the exact k values where the
first attempt's instance died (peak VRAM there: 155.5 GB of 183 GB, same
ballpark as the crash run). **That settles the open question from the
first attempt: the earlier `exited` container was host-specific
infrastructure flakiness, not a real VRAM ceiling.**

**V12 monotonicity gate:** 3 violations against the fixed CSV, against a
1-violation pre-fix baseline (`results/gpu_heatmap_b200_20260728.csv`) and
an 8-violation broken-anchor run (`..._20260730.csv`). Read literally
that's over the "<=1" bar, but inspecting each one: one is the exact same
pre-existing violation from the original baseline (n=4194304, k=2048->4096,
967.9->939.0ms then vs. 971.1->942.1ms now, B=128 unchanged both times).
The other two (n=256 k=128->256; n=512 k=64->128) have **identical B in
both the pre-fix baseline and this run** (B=64, unchanged) and sit at
0.08-0.11ms, a regime where GPU launch overhead dominates and cv jumped
from ~1-3% to ~5-10% between runs at the same cells -- machine noise, not
a B-selection effect. **Zero of the 3 monotonicity violations are
attributable to either fix.** Confirmed via `bench_gpu_fused verify` (0
FAIL) and `test_gpu_cost_model` (483 passed, 5 failed, identical
pre-existing failure set both before and after).

**The monotonicity gate passing is not the same as "no regressions,"
and it missed one.** A full cell-by-cell diff against the original
`results/gpu_heatmap_b200_20260728.csv` baseline (not just the 12
originally-targeted cells, not just the monotonicity check) turned up
**16 newly-regressed cells the 12-point narrow-around fill never touched
directly**: n in {131072, 262144, 524288, 1048576} (below the frontier)
plus n in {2097152, 4194304} at k=64 (above it), all previously B=64 or
B=32, now B=128, up to +17.3% slower (n=131072/262144/524288/2097152/
4194304, all k=64). Verified these are genuinely new, not inherited from
V11: replayed the joint-NN lookup against `results/gpu_heatmap_b200_
20260730.csv` (V11 applied, anchors not yet filled) and every one of these
16 cells was still correctly B=64/B=32 there, matching the pre-V11
baseline almost exactly. **Only the anchor addition changed them.**

**Root cause: the 12-point `--narrow-around` fill was too narrow.** It
added low-k anchors only at n >= 2,097,152, on the assumption that the
table's low-k sparsity was specifically an "above the frontier" problem
(per the original diagnosis). It is not -- the table was already sparse
at low k across a much wider n range (at least 131,072 through
4,194,304). Adding anchors far away in n but close in k gave the
joint-NN log-space distance a new "nearest neighbour" for several
medium-n queries that previously had no nearby competitor, and for those
cells the new large-n anchor's B is measurably wrong.

**Net effect is still a small aggregate win** (total time across all 210
common cells: 451.7s -> 444.3s, -1.6%; the originally-targeted 12 cells:
-1.5% net, back to roughly their pre-V11 baseline; the 16 newly-regressed
cells: +11.3% within that subset). But "aggregate is still positive"
does not make 16 individual regressions acceptable, especially since
several are worse than 15%.

**NOT YET FIXED.** The correct fix, following the same methodology as
the original fill: measure real B directly (`calibrate_gpu_best_b
--narrow-around`) at the 16 regressed cells (or more broadly, sweep low-k
anchors across n in {131072, 262144, 524288, 1048576} the way the
original fix did for n >= 2,097,152), splice in, and rerun the full
heatmap + cell-by-cell diff again -- including this same check against
ALL 210 cells, not just the newly-touched ones, since this exact mistake
(narrow, targeted fix; broad, untargeted side effect) could repeat.
**Blocked on funding**: this session's B200 credit ran out (~$0.75 left)
before this was caught.

**Second pass, same day, after another top-up.** Built a 21-point skeleton
covering the actual gap: low-k anchors at n in {131072, 262144, 524288,
1048576} (all four cells that regressed in the first pass), plus k=64
buffer points at n=2097152/4194304 (contract `46331871`,
`calibrate_gpu_best_b --narrow-around 32,48,64,80,96`). Directly measured,
mostly B=48, not the B=64 the old table implied -- the assumption behind
the first pass's fix (that everything in this range was correctly B=64)
was itself never verified; it wasn't quite right either.
`GPU_N_BSELECT_POINTS` 72 -> 93.

**This time, ran the full comprehensive diff against ALL 210 cells before
declaring anything, not just the ones touched.** Result: 61 cells changed
B versus the original baseline, 54 improved (>2%), 6 within noise, **1
regression**: n=65536,k=2048, B 64->48, +16.5% (12.13ms -> 14.13ms).
Traced the same way as before: replayed the joint-NN lookup against the
new 93-point table and confirmed the new anchor at (131072,1024,B=48) is
now the nearest neighbour for this query (distance 0.980) and pulls it
away from its correct answer -- **the identical failure mode as the first
pass, on a smaller scale**: a targeted fix for one cell became a false
match for a nearby, untested one.

Total grid time across all 210 common cells: 451.7s -> 441.1s (-2.34%,
better than the first pass's -1.6%). Monotonicity gate (V12): 4
violations, of which one is the known n=65536,k=2048 regression, one
(n=2097152, k=8192->16384) is not a regression at all but a bigger
improvement at k=16384 outpacing a smaller one at k=8192 (both B changed,
both got faster, just by different amounts), one is the same
pre-existing baseline violation from V7's very first entry
(n=4194304,k=2048->4096, still ~967ms->~939ms), and one is sub-0.1ms
launch-overhead noise (n=128).

**Not chasing this further right now.** Each additional targeted anchor
addition has, twice now, fixed its target and created exactly one class
of new problem elsewhere; each verification cycle costs a full B200
rental to confirm (or refute) the next one. Diminishing returns: 16
regressions to 1 for the first follow-up pass; continuing this
whack-a-mole pattern to chase the last single cell is not obviously worth
another full rental cycle when the aggregate signal (-2.34% total time,
54 genuine improvements) is unambiguous and the remaining defect is one
cell, isolated, understood, and not a repeat of an unknown mechanism.
**If revisited:** measure n=65536,k=2048 directly, splice it in as one
more anchor, and run the full 210-cell diff one more time -- given the
pattern, budget for the possibility that this fixes n=65536,k=2048 but
creates a new false match at some other untested nearby cell, and decide
then whether to keep iterating or accept whatever single cell remains
wrong at that point.

## V8. Past the calibration ceiling, use fixed fallbacks, never extrapolation

**Evidence:** `CLAUDE.md` calibration-boundary section; commits `a597557`,
`f2ad24e`.

On CPU, past `CALIBRATED_MAX_CONV_LEN`: always FFT, wrap-free FFT sizing, always
hybrid, fixed B=32, `FFTW_ESTIMATE`. Choosing maximum wrap unconditionally was
measured to be both wrong (`sum-1 = -0.496`) and ~8x slower. `tools/test_uncalibrated_fallback.c`
pins this. Do not optimize it back without a cost model to justify it.

## V9. Zen 4: the 2DPC box is the standing reference

**Decided:** 2026-07-27. **Evidence:** commit `5b3ba3e`; `RESULTS.md`.

The reference box runs RAM at 3600 MT/s against a higher DIMM rating, an AMD AM5
2-DIMMs-per-channel electrical limit, confirmed not fixable at OS/BIOS level and
re-confirmed by `dmidecode` on the 2026-07-30 redeploy. No 1DPC replacement was
available.

Prior higher-bandwidth (1DPC-era) data is **kept and reframed**, not deleted,
because the ceiling's effect is not a flat percentage: linear/schoolbook timings
are nearly unchanged (~0.97-1.0x) while hybrid/FFT timings are 40-65% slower at
large k. A single correction factor would misrepresent it.

**Trap:** `results/` contains BOTH `bench_grid_zen4_serial.txt` (Jul 24, 1DPC
era) and `bench_grid_zen4_serial_20260727.txt` (Jul 27, 2DPC reference). Compare
new data against the **dated 20260727 files**. Comparing against the undated
files shows a spurious ~1.8x slowdown that is purely the 1DPC/2DPC difference.
This mistake was made on 2026-07-30 and corrected.

## V10. Zen 4 parallel scaling cliff is a documented limit, not a bug to fix

**Decided:** 2026-07-23. **Evidence:** commit `a4ce099`.

Parallel speedup falls from 10-14x to ~3.3x between n=8,192 and n=16,384.
Root-caused via `perf stat`: IPC 1.53->0.57, cache-miss rate 4.4%->10.5%, cycles
grow 6.3x while instructions grow 2.35x. `OMP_PROC_BIND` and explicit core/CCD
pinning were tested and did not recover it, ruling out affinity/NUMA. Cause is
16 concurrent hybrid FFT trees exceeding combined L3. Documented as a real
limit; a per-thread footprint reduction was judged a nontrivial separate
undertaking with unclear payoff.

**Also standing:** Zen 4 must use `OMP_NUM_THREADS=16` (physical cores), never
the default 32. SMT siblings add nothing for this FPU/vector-port-bound workload
and silently corrupt small-n parallel timings.

## V11. B-selection lookup is single-pass joint (n,k) nearest neighbour

**Decided:** 2026-07-30. **Evidence:** commit `bf2886b`.

Both `empirical_best_B()` and `gpu_empirical_best_B()` previously resolved
sequentially (nearest n, then nearest k within that one n) while their docstrings
and CLAUDE.md claimed 2D nearest neighbour. Sparse points shadowed dense grid
rows: on M3 Pro a query at (n=4096, k=4096) was answered by a k=21 sample,
returning B=24 at 186.3 ms against B=32 at 142.2 ms, 24% slower.

Now one pass minimising `hypot(log n - log n_i, log k - log k_i)`, first-wins
tie-breaking. `tools/test_bselect_lookup.c` pins it, device-agnostically, in CI
across all three device targets.

**CPU outcome:** unambiguous win. **GPU outcome:** see V7 for the open
consequence.

## V12. Acceptance criterion is monotone TIME, not monotone B

**Decided:** 2026-07-30. **Evidence:** user decision this session.

B may be non-monotonic in k; that reflects kernel/memory-layout quirks and is
expected. What must hold is that **measured time increases with n and with k**.

This is an effective regression detector: the 2026-07-30 GPU heatmap had 8
time-monotonicity violations against the pre-change baseline's 1, and all 8 were
exactly the V7 regression cells. **Acceptance gate for the V7 fix: violations
back to <= 1.**

## V13. FP64 is required

**Evidence:** memory `project_fp64_requirement`.

Lower precision is not viable for this algorithm. Related: cuFFT LTO callbacks
were rejected because their per-element rounding compounds across tree levels to
~2% error, despite being within cuFFT's accuracy spec.

## V14. Never regenerate Zen 4 FFTW wisdom on redeploy

**Evidence:** standing user instruction; memory `feedback_zen4_wisdom_port_no_regen`.

Copy `devices/zen4/fftw_wisdom.dat` onto the new box byte-identical (verify by
md5) and rebuild AOCL-FFTW from source with the full flag set. Do not re-run a
PATIENT calibration from scratch. Verified working again 2026-07-30.

---

## V15. Dispatch tables are calibrated serial and reused for parallel

**Decided:** before 2026-07-30, clarified by the user 2026-07-30.
**Evidence:** the code itself.

Both dispatch decisions are **thread-count-blind**: `select_best_B(int n, int k)`
and `empirical_crossover_k(int n)` take no thread count, and
`tools/calibrate_best_b.c` / `tools/calibrate_crossover.c` measure
single-threaded. The only thread-awareness in `src/cpu/icm.c` is parallelising
over Q (quadrature points), not dispatch. So the same tables serve both modes.

**This is deliberate.** Re-calibrating a separate parallel B table was judged not
worth the effort; the serial optimum is accepted as a good-enough heuristic for
the parallel case. It is a **known possible suboptimality**, not an oversight.

**This does NOT mean parallel sweeps are skipped.** Parallel grids and parallel
contour sweeps are run and are required for the paper's figures and tables. Only
the *calibration* of the dispatch tables is serial-only.

**Supporting evidence to generate:** measured parallel dispatch accuracy is the
test of whether the heuristic holds. If serial-calibrated tables dispatch
correctly on nearly all parallel cells, the heuristic is validated. That artifact
does not exist yet and is queued; it also backs the paper's dispatch-accuracy
claims. **Disclose this scoping decision in the paper** rather than leaving a
reviewer to discover it.

## V16. select_best_B() discarded the calibration table's answer whenever k was small (FIXED)

**Recorded:** 2026-07-30. **Evidence:** `tools/sweep_best_b.sh --device zen4`,
first run ever on real hardware, `results/b_optimal_sweep_zen4_2026-07-30.csv`.

Every one of the 11 tested `n` values (64 through 65536) at `k=10` dispatches
`B=8` while the directly-measured best is `B=32`, a 37-44% gap at every single
point (worst: n=65536, 44.4%; best: n=32768, 34.0%). No other `k` value shows
this pattern this consistently; the next-worst offenders are scattered
one-offs at 15-27%.

This was not the V11/V7 joint-NN shadowing bug: `test_bselect_lookup.c` passed
113/113 on this exact box, so the table lookup itself was fine. Confirmed by
reproducing the raw joint-NN lookup in Python directly against
`devices/zen4/fft_config.h`: at k=10, the table's real answer is B=16 or B=32
for essentially every n tested. The table was never wrong.

**Root cause: `select_best_B()` (`src/cpu/icm.c`) discarded that answer before
ever checking it.** Its candidate-validity filter was `if (B > k || B > n)
continue;` over candidates `{8,16,24,32,48,64}`. Since 16 is the second-smallest
candidate, any `k < 16` eliminates every candidate except 8, so the function
returned B=8 unconditionally for k<16, regardless of what the table said. The
`B > k` clause was a straight carryover from the pre-empirical analytical
version of this function (`d128bca2`, March 2026) that nobody revisited when
`c70ca4e` (July 22) swapped in the calibration-table lookup. It was never a
correctness requirement: `tree_ctx_create_ex2()` already caps each tree
level's working polynomial size at `min(B * 2^level, k)`, so B > k is
structurally fine.

**Fixed:** removed the `B > k` clause (kept `B > n`, which guards a different,
still-real case: nearest-neighbor answers that don't fit tiny n, e.g. B=32 for
n=20). The identical bug existed in the GPU path
(`gpu_select_best_B_est()`, `src/gpu/gpu_cost_model.cu`) and got the same fix.

**Re-verified on Zen4** (`tools/sweep_best_b.sh --force`, full grid): exact
`auto_B == best_B` matches rose from 23/65 to 31/65, within-3% from 28/65 to
38/65, and the k=10 column is gone from the worst-offenders list entirely
(remaining worst offenders are unrelated large-k cells at 14-22%, pre-existing
ordinary nearest-neighbor imprecision, not this bug).

**Practically low-stakes for full-equity CPU dispatch**, worth noting: the
crossover table (`crossover_n[]`/`crossover_k[]`) never returns a threshold
below k~249 anywhere (clamped at both ends), and doesn't depend on
`select_best_B()` at all. So for `icm_equity()`'s normal full-equity path,
linear always wins below k~250 regardless of this bug -- B gets computed and
silently discarded in that branch. The bug mattered for: subset-query dispatch
(a different code path, `n_targets > 0`, not gated by the same crossover
table), any tool or caller that forces/queries the hybrid engine directly at
low k (which is exactly how `sweep_best_b.sh`'s `validate_best_b.c` found it),
and the GPU path where the equivalent crossover gate does not exist. Not yet
checked whether M3 Pro's table shows the same k<16 pattern -- expected to,
since the bug is in shared `src/cpu/icm.c`, not per-device data, but unverified
(M3 Pro out of scope this session).

## Unverified recollections

None outstanding. (The Zen 4 parallel-sweep item previously recorded here was
clarified by the user on 2026-07-30 and promoted to V15; the original wording
misread a calibration-scope decision as a data-collection-scope decision.)

## Known data-hygiene issues, not yet resolved

- `RESULTS.md`'s Zen 4 **parallel** column mixes two machines: cells at n=128
  and n=256 match an older undated file while n >= 512 match the cited
  `20260727` file. Serial is clean (all 66 cells match).
- The Zen 4 1-second threshold (n=17,984) has no saved artifact, unlike the GPU
  one.
- `tools/sweep_best_b.sh` has now been run on Zen4 (2026-07-30, full grid,
  `results/b_optimal_sweep_zen4_2026-07-30.csv`), closing this gap on one
  platform; it surfaced V16 (open). Not yet run on M3 Pro (blocked, see
  HANDOFF.md) or as a GPU analogue.
