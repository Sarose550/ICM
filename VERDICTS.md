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

**Evidence:** paper sections 3.5 and 5.x; `CLAUDE.md`'s architecture notes.

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

**Evidence:** `c70ca4e`; `CLAUDE.md`'s architecture notes.

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
true optimum more than one candidate step away, exactly what V7 observed.
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

**Superseded, same day.** A project requirement called for a fully non-discretionary,
one-command production tool; patching `calibrate_block_size.py`'s
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
GPU usage per `heatmap_gpu.cu` goes to `33554432`), an 8x undershoot
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
`46358305`); exact command as typed at the time, the flags were
renamed to `--n-min`/`--n-max` immediately afterward, see the
flag-rename entry below. It worked as designed and found far more real disagreement
than expected: 520 probes in 20 minutes, most landing well off the
standard heatmap grid (random log-uniform points like n=1980, n=90283,
n=411295, points nobody had ever measured before), many with large
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
was actually a real one mid-session, see below, but it does not
explain this): `n=2048,k=512` now dispatched B=1280 when direct
measurement said B=64 is right (gap 292.75%); `n=16384,k=64` dispatched
B=1024 against a true B=48 (gap 265.32%). Both are real, not measurement
noise, and both are much worse than what the *unmodified* baseline
already had at those same cells (63.04% and 7.31% gaps respectively;
the baseline was already imperfect there, but nowhere near this badly).

**Root cause: pure joint-log-distance nearest-neighbor is not a safe
table-construction strategy for this function.** Two compounding
factors, both real: (1) `B*(n,k)` is far noisier / more locally variable
than the "smooth regions with occasional cliffs" model implicit in
nearest-neighbor lookup assumes: confirmed by how much real
disagreement 520 essentially-random samples turned up. (2)
`validate_planner_gpu`/`validate_best_b` measure with a single rep, no
repeats; confirmed contaminating results directly: one probe during
the run reported `auto_B=64, best_B=64, gap=21.44%` for the *same* B,
which can only be timing noise between two separate single-shot runs.
Combined, a single new point (possibly itself noisy) can become the
nearest neighbor for many nearby *unmeasured* queries and silently pull
them to a worse answer, exactly the V7 failure mode, but now happening
at the scale of hundreds of essentially-random points instead of a
handful of deliberately-chosen ones.

**Reverted in full.** `devices/b200/gpu_fft_config.h` was restored to
the exact committed state (93 points, `md5 d23ad491...`) before
rebuilding and destroying the instance. Nothing from this run shipped,
including the one manually re-derived point (`65536,2048,B=80`) that
*was* correctly measured; it's real and could be re-added on its own,
but landing it alone, separately from the broken bulk run, was judged
not worth a special-cased partial commit tonight. `results/gpu_heatmap_
b200_20260731_adaptive.csv` and `..._final.csv` are kept as evidence of
this finding, not as usable calibration data; do not treat either as
a baseline.

**Also found and fixed mid-session (operational, not a data problem):**
`validate_planner_gpu`'s default binary path
(`./build/validate_planner_gpu`) doesn't match where the Makefile
actually places it (repo root, `./validate_planner_gpu`); worked around
with an explicit `--validate-bin` flag, needs a real default-path fix
(see the flag-rename item below, batch it in). Separately, a real
stale-binary trap: rebuilding one target (e.g. `make validate_planner_
gpu`) does NOT relink other already-built executables (`heatmap_gpu`)
even when their shared `.o` dependencies changed underneath them;
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
which doesn't exist; the Makefile places these at the repo root.
Found this the hard way mid-run today (see above); worked around with
an explicit `--validate-bin` flag at the time, now fixed at the source.

**Root cause found and fixed, same day, before another B200 rental.**
Review pushback on "nearest-neighbor is unsafe" as too vague for a
281% regression called for the actual mechanism. Traced it precisely:

- Every worst regression (n=2048,k=512 -> B=1280 vs true B=64, gap
  292.75%; n=16384,k=64 -> B=1024 vs true B=48, gap 265.32%) landed at
  a problem size finishing in **under ~2ms total**, GPU kernel-launch-
  overhead territory.
- `validate_planner_gpu.cu`'s candidate sweep (the oracle
  `calibrate_adaptive.py` trusts directly) timed every candidate with a
  **single rep, no median, no noise check**: `if (t_ms < best_ms)
  best_B = B`, full stop. `calibrate_gpu_best_b.cu` (used for
  `--narrow-around`) and the CPU `calibrate_best_b.c`/
  `validate_best_b.c` had the same shape of bug: 1-rep-rank, with a
  runoff/confirmation step that only ever re-measures the top-2; if
  noise mis-ranks the true winner *outside* the top-2, nothing catches
  it.
- The GPU candidate set spans B=16 to B=1536, a 96x range (CPU's is
  8x, {8,16,24,32,48,64}), so a noise-driven pick at small n can land
  on a structurally very different B, amplifying a small timing fluke
  into a catastrophic real regression once dispatched for real.
- Rounds 1 and 2 operated at n>=65,536 (tens of ms per candidate,
  noise a tiny fraction of signal), masking the bug. Round 3 pushed
  into n=1,024-524,288, squarely in the noisy regime, and exposed it.
  This also explains "why only now": the bug was always there, the
  earlier rounds just never queried a region where it mattered.
- The production `heatmap_gpu.cu` tool already solved exactly this:
  adaptive reps (up to 10 for sub-10ms cases, extending to 15 if
  cv>3%) converging on a median. It was never applied to the
  calibration-writing oracles.

**Fix:** ported `heatmap_gpu.cu`'s adaptive median/cv-convergence loop
into all four measurement tools:
`validate_planner_gpu.cu`/`calibrate_gpu_best_b.cu` (GPU) and
`validate_best_b.c`/`calibrate_best_b.c` (CPU), for parity. Every
candidate now gets its own converged measurement (3 reps minimum,
extended to 15 until cv<=3%); the old top-2-only runoff/confirmation
logic is removed as redundant (there's no longer a single noisy rep
left to mis-rank). Cost impact is close to zero in aggregate: fast/
noisy cases get more reps but each rep is proportionally cheap; slow
cases (>100ms) still get 1 rep, unchanged from before, mirroring
`heatmap_gpu.cu`'s own cost profile which was already proven not to
blow up runtime.

**Verification plan, not yet executed (needs a B200 rental):** re-probe
the three worst regressed cells (n=2048,k=512; n=16384,k=64;
n=65536,k=2048) with the fixed `validate_planner_gpu` and confirm they
now report a low, stable gap_pct against the known-good B; only after
that spot check passes is a real re-run over the broken domain
(n=1,024-524,288) worth funding.

**Spot check executed, confirmed the fix, same night.** Rented a fresh
B200 (contract `46367689`), rebuilt with the fix, re-probed the three
cells directly: n=2048,k=512 gap collapsed 292.75% -> -0.10%;
n=16384,k=64 collapsed 265.32% -> 0.34%; n=65536,k=2048 (the one
already-known-real regression, unrelated to the noise bug) stayed at
17.76%, essentially unchanged from its documented 16.5%. Exactly the
right signature: fake noise-driven signal collapsed to zero, real
signal stayed real. The timing-noise root cause is confirmed correct.

**Same night, M3 Pro (real hardware, CPU side): clean, real result,
shipped.** `calibrate_adaptive.py --device m3_pro --budget 25m` with
the fixed `validate_best_b`: 92 probes, 2466 -> 2558 points,
budget-capped but nearly converged (top score 0.1453 vs 0.15
threshold). Gaps modest throughout (0-8%), no wild swings; CPU's
narrower 8x B-candidate range (vs GPU's 96x) is much less exposed to
the mechanism above, and the fix confirms that empirically as well as
analytically. `bench_grid verify` PASSED, `bench_grid crossover` sane,
`libicm.a`/`libicm.dylib` both build clean. Committed `af219d0`.

**Same night, real full re-run over the broken GPU domain: found a
SECOND, separate failure mode, not shipped.** With the noise bug fixed
and individually confirmed correct, ran
`calibrate_adaptive.py --device b200 --n-min 1024 --n-max 524288
--budget 20m` for real (contract `46367689` again); the actual
domain round 3 broke. 252 probes, 93 -> 345 points, budget-capped
(top score 1.31, well above threshold: lots of genuine signal in
this newly-explored region). `bench_gpu_fused verify` (0 FAIL) and
`test_gpu_cost_model` (483/5, identical pre-existing failures) both
clean.

**But the mandatory full diff against the original `..._20260728.csv`
baseline showed 40 regressions out of 84 `--fast`-range cells, some
catastrophic** (n=1024,k=128: B64->1024, +402.7%; n=65536,k=128:
B64->1536, +329.3%; n=32768,k=256: B64->1536, +275.1%), **total time
+5.23% worse than the original baseline**, materially worse than
round 3's own 36/84 disaster, despite every individual measurement now
being genuinely correct (spot-checked above). This is NOT the noise
bug reappearing. It is a second, separate, previously-undiagnosed
mechanism:

**Root cause: correct individual measurements do not make a sparse
nearest-neighbor table safe.** B*(n,k) has real local structure in
this small-n region (unlike the large-n region rounds 1-2 touched,
which already had reasonably dense prior coverage). Populating a
previously-empty region from scratch via priority-queue sampling
means every injected point becomes the nearest neighbor for a
*neighborhood* of never-directly-probed queries, and nothing checks
whether the neighborhood actually agrees with the one point that got
measured. `calibrate_adaptive.py`'s original design (this session,
`b8a68f3`) deliberately removed the old orchestrator's dense
base-sweep step as "redundant, adaptive refinement finds gaps on its
own"; that judgment was wrong for genuinely unexplored regions. The
base sweep wasn't just expensive waste; it also guaranteed the
baseline local density nearest-neighbor lookup implicitly assumes.
Removing it without replacing that guarantee is what let this ship
undetected tonight, exactly as removing per-candidate noise robustness
let round 3 ship undetected.

**Not shipped.** `devices/b200/gpu_fft_config.h` was never touched
locally (`live_table` only lived on the remote instance, an important
accident of this session's architecture); nothing to revert. The
broken 345-point table is preserved as
`results/gpu_fft_config_20260731_run2_BROKEN_evidence.h`, and its
heatmap as `results/evidence/gpu_heatmap_b200_20260731_run2_final_NOT_SHIPPED.csv`
(moved into `results/evidence/` in a later hygiene pass), both
evidence, neither usable calibration data. Instance destroyed
(`46367689`).

**Fix, built and validated the same night: a canary safety net in
`calibrate_adaptive.py` itself (commit follows this entry), not a
heuristic wrapper.** Before any run, snapshot the config header byte-
for-byte and measure a small, fixed, held-out canary set (3 k-fractions
per landmark n-anchor, deliberately disjoint from the landmark and
adaptive-refinement k-fractions) using the exact same converged
oracle (`run_validate_probe`) everything else in this session already
trusts. After the run, rebuild `validate_bin` against the new table
(the caller MUST supply `--rebuild-cmd`; there is no safe default:
GPU needs an environment-specific `CUFFTDX_INC`, and `validate_best_b`
has no Makefile target at all, so a guessed default risked silently
verifying against a stale binary, exactly the trap that caused a
`test_gpu_cost_model`/`heatmap_gpu` staleness bug earlier this same
session) and re-measure the same canaries. Auto-revert to the exact
pre-run snapshot if the aggregate regresses >2% or any single canary
regresses >5%; otherwise keep. Validated three ways before trusting
it: (1) a real, live M3 Pro run through the full pass path, confirmed
by direct md5sum that nothing changed when canaries agreed; (2) a
real, live run with a deliberately broken `--rebuild-cmd`, confirmed
by direct md5sum that the file was restored byte-for-byte; (3) direct
unit tests of the comparison logic against synthetic data, including a
literal replay of tonight's own worst cell (n=1024,k=128,
+402.7%) to confirm it fails exactly the way it should. This closes
the fail-safe gap that let tonight's run and round 3 both ship
undetected; it does not fix the underlying density assumption (see
above), which remains a real, disclosed limitation: `--budget` alone
still can't guarantee a new region gets dense enough coverage for
nearest-neighbor to be locally valid, only that whatever the run does
produce is net-positive on the canaries actually checked, before it's
trusted.

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
frontier and already explained in HANDOFF.md; no new failures from the
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
from ~1-3% to ~5-10% between runs at the same cells: machine noise, not
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
(per the original diagnosis). It is not; the table was already sparse
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
heatmap + cell-by-cell diff again, including this same check against
ALL 210 cells, not just the newly-touched ones, since this exact mistake
(narrow, targeted fix; broad, untargeted side effect) could repeat.
**Blocked on funding**: this session's B200 credit ran out (~$0.75 left)
before this was caught.

**Second pass, same day, after another top-up.** Built a 21-point skeleton
covering the actual gap: low-k anchors at n in {131072, 262144, 524288,
1048576} (all four cells that regressed in the first pass), plus k=64
buffer points at n=2097152/4194304 (contract `46331871`,
`calibrate_gpu_best_b --narrow-around 32,48,64,80,96`). Directly measured,
mostly B=48, not the B=64 the old table implied; the assumption behind
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
away from its correct answer: **the identical failure mode as the first
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
more anchor, and run the full 210-cell diff one more time; given the
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

### V9a. Correction: the "1DPC" label was a theory, and it was wrong (2026-08-04)

The "1DPC" characterization of the earlier, faster box above was an
inference from its higher measured bandwidth, never a verified hardware
fact, and the 2026-08-03 replacement box disproved the framing: it
measures 32.7 GB/s streaming, ordinary 3600 MT/s 2DPC, with no elevated
bandwidth anywhere, and the original box was lost before its discrepancy
was ever explained. Per an explicit project decision, the multi-machine
comparison is dropped entirely: the current box's data is the single Zen4
reference, presented without configuration qualifiers, and the unexplained
old-box discrepancy is not chased further.

Final disposition: `contour_zen4_{serial,parallel}_q256_1dpc.csv`,
`results/zen4_1dpc_vs_2dpc.png`, and `tools/results/plot_zen4_1dpc_2dpc.py`
are deleted (git history retains the data; the filenames asserted an
unverified configuration as fact). The undated
`bench_grid_zen4_*.txt`/`contour_zen4_*_q256.csv` files now hold
current-box data (2026-08-03 refresh), so this entry's "Trap" paragraph
and its "kept and reframed" decision are historical record only. "1DPC"
mentions in the hygiene entry further down have the same status.

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

**Decided:** 2026-07-30. **Evidence:** engineering decision this session.

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

**Evidence:** standing project instruction; memory `feedback_zen4_wisdom_port_no_regen`.

Copy `devices/zen4/fftw_wisdom.dat` onto the new box byte-identical (verify by
md5) and rebuild AOCL-FFTW from source with the full flag set. Do not re-run a
PATIENT calibration from scratch. Verified working again 2026-07-30.

---

## V15. Dispatch tables are calibrated serial and reused for parallel

**Decided:** before 2026-07-30, clarified 2026-07-30.
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
linear always wins below k~250 regardless of this bug; B gets computed and
silently discarded in that branch. The bug mattered for: subset-query dispatch
(a different code path, `n_targets > 0`, not gated by the same crossover
table), any tool or caller that forces/queries the hybrid engine directly at
low k (which is exactly how `sweep_best_b.sh`'s `validate_best_b.c` found it),
and the GPU path where the equivalent crossover gate does not exist. Not yet
checked whether M3 Pro's table shows the same k<16 pattern; expected to,
since the bug is in shared `src/cpu/icm.c`, not per-device data, but unverified
(M3 Pro out of scope this session).

## V17. Final closeout pass: device portability, dead-code removal, a real dispatch validator, and repo hygiene

**Decided:** 2026-08-01. **Evidence:** commits on `results-gpu-section`
(PR #7), this session.

**Device portability.** `calibrate_adaptive.py --device` is now genuinely
free-text instead of a hardcoded 3-entry allowlist; `--gpu` is the explicit
CPU/GPU signal `config_header`/`array_prefix`/`n_macro` derive from.
`gen_calib_skeleton.py` had the identical hardcoded pattern (called as a
subprocess) and got the same fix.

**Dead-code removal.** Deleted `src/cpu/cost_model.h` (zero live callers)
and the 10 already-flagged dead constants from all three device config
headers. Fallout: several standalone calibration tools
(`probe_leaf_extract.c`, `test_cpu_cost_model.c`, `calibrate.c`'s
fresh-device emission, `fit_cost_model.py`, `calibrate_full.sh`,
`bench_linear_batched_fma.c`) still referenced these macros and needed
follow-up fixes, caught only because a live rebuild of each affected
tool was actually attempted, not assumed clean from the deletion alone.

**M3 Pro dispatch-accuracy claim had no real tool behind it.** The paper's
"100% serial / 83/84 parallel" dispatch-accuracy claim traced to an
ephemeral, never-committed process from an earlier session (icm_paper
commit `759833e`, 2026-07-23); confirmed via git history that no
committed tool, including two deleted ones with plausible-sounding names
(`quantify_dispatch_gap.c`, `eval_model_vs_plans.c`), ever did this
comparison; both evaluate the old pre-empirical-table analytical formula,
unrelated to the current `select_engine()` lookup. New
`tools/validate_dispatch.c` closes this gap: compares `select_engine()`'s
choice against the true measured-faster engine over the same 42-cell
grid as the paper's Table 1/2. 40 of 42 cells verified correct on M3 Pro
serial; the two largest (n=65536, k=n/2 and k=n) hit a background-task
duration limit in this environment three times in a row, not a dispatch
failure; every surrounding cell at that same n passed.

**M3 Pro data regenerated twice.** An earlier regeneration pass was
silently contaminated by a stuck DeepSeek Deck daemon process (166% CPU
for 43+ hours, unrelated to this session, killed once found) inflating
timings across the board, including calibration-independent linear-engine
cells that have no mechanism to be affected by anything this session
touched. Caught via a sanity check on exactly such a cell (n=1024,k=10:
contaminated run read 2.08ms vs. a clean 1.72ms against an old baseline
of 1.79ms), then fully redone on a verified-quiet system.

**Real values were stale in the paper by about a week.** `WRAP_FMA_NS`
was re-measured 2026-07-22 (`e0d13ed` M3 Pro, `e01e87c` Zen4) but the
paper still cited the pre-recalibration values (0.4942/0.40 instead of
the live 0.5160/0.4360); CLAUDE.md's own "Live constants" table had the
same staleness. The M3 Pro B-selection table's mode also shifted from
B=16 to B=32 after tonight's real recalibration (2,466 -> 2,563 points);
"typical B=16 on M3 Pro" was already inconsistent with the paper's own
`sec:dispatch` text (which said B=32 for both platforms) even before
tonight; this recalibration made the stale claim wrong on the numbers
too, not just internally inconsistent.

**Repo-hygiene findings from a full 144-file PR-diff audit** (delegated
to a Sonnet subagent, independently spot-verified): `results/gpu_heatmap_
b200.csv` (the most canonical-looking filename in the directory) was
actually the *stale* GPU calibration heatmap (differed from the correct,
current 93-point-table-matching data on all 211 rows), while the file
that was actually authoritative carried a workflow-internal `_gapfill`
suffix no reader would know to trust more. Fixed by swapping names so
the canonical filename points at canonical data. Also deleted one
genuine leftover (`results/b_shadow_impact_20260730.md`, a
subagent-to-supervisor working note that was never real documentation)
and deduped M3 Pro's undated/`_20260731`-dated bench_grid and contour
snapshot pairs (unlike the Zen4 1DPC/2DPC pair, which encodes a real
hardware distinction worth keeping both sides of, the M3 Pro pair had no
documented reason to keep both). The `results/evidence/` directory and
other `BROKEN`/`NOT_SHIPPED`-labeled files were reviewed and deliberately
left alone; they have real provenance in V6/V7 above and read as
documented engineering rigor, not clutter.

**Paper's past-methodology-flaw narrative was too long.** Explicit review
feedback: "nobody gives a fuck what past bugs looked like... at most one
paragraph in the entire paper about it." Both calibration-methodology
subsections (CPU and GPU) previously carried multi-paragraph blow-by-blow
bug narratives (specific regression percentages, `(n,k)` cell
coordinates, dates) inherited from earlier sessions' drafting. Condensed
to roughly one combined paragraph across the whole paper while preserving
the actual methodology, all precedent citations, and the two things that
are current design/infrastructure rather than past-bug narrative (the
GPU's deliberately-targeted above-frontier anchor scope, and the canary
safety net's real mechanism). 124->82 and 128->60 lines in the two
subsections; paper 30->28 pages.

## V18. M3 Pro tree-beats-hybrid at large n/k was a B-selection coverage gap, not a regression (FIXED)

**Recorded:** 2026-08-02. **Evidence:** `tools/calibrate_best_b.c --narrow-around`
run directly on this M3 Pro; `git log` diff of `results/bench_grid_m3pro_serial.txt`
across every regeneration since the first real PATIENT calibration (2026-07-20).

A paper read-through flagged the pure tree engine beating hybrid at n=8192
(k=n/2, k=n) and n=16384 (k=n/4, k=n) in both serial and parallel tables, and
it was recalled this hadn't shown up in earlier, already-vetted data. Git
history confirmed the recollection was correct: hybrid won comfortably at these
exact cells from 2026-07-20 through 2026-07-23, then narrowed and flipped
starting with the 2026-07-24 "widened B-selection tables" regen and again
after 2026-08-01's refresh.

**Root cause:** none of the 4 affected `(n,k)` cells were themselves real
calibration anchors in `devices/m3_pro/fft_config.h`'s `bselect_*` tables;
all four were served by nearest-neighbor lookup from points 0.01-0.02
log-distance away that happened to agree on `B=48`. Direct adaptive-median
measurement (`calibrate_best_b --narrow-around 32,48`, same converged
methodology as the live table) found the true best at all 4 points is
`B=32`, not `B=48`, an 8-16% gap, enough to flip the tree/hybrid ordering
at cells where the margin was already thin.

Not a production correctness issue: `select_engine()` never dispatches to
tree at all (confirmed via `dispatch_validation_m3pro_serial_20260731.csv`,
which only ever contains `L`/`H`); tree is a `bench_grid`/paper comparison
baseline only. The bug was specifically hybrid's *within-engine* B choice
at those 4 points.

**Fix:** measured the 4 points directly, spliced the corrected `B=32`
anchors into `devices/m3_pro/fft_config.h` (2563 -> 2567 points) via
`tools/splice_calib_points.py`, rebuilt (`bench_grid`, `libicm.a`,
`libicm.dylib` all clean), reconfirmed `bench_grid verify` (ALL TESTS
PASSED) and `crossover` (unaffected). Regenerated the full M3 Pro
serial/parallel grids, contours, and plots; hybrid now wins at all 4
cells (and everywhere else in the grid) in both modes. `RESULTS.md` and
the paper's `tab:serial`/`tab:parallel` updated from the fresh data.

**Also found and fixed in passing:** tonight's earlier `refresh_all.sh`
rewrite (part of V17's device-portability work) had silently reintroduced
date-suffixed filenames for the serial/parallel grid and contour steps
(`bench_grid_m3pro_serial_20260802.txt` instead of the stable
`bench_grid_m3pro_serial.txt`), undoing a 2026-07-21 fix that specifically
existed to keep those names stable (plots already tolerated either, via
`find_latest()`'s mtime-based glob, which is why this went unnoticed).
Fixed at the source in `tools/results/refresh_all.sh`; `gen_crossover.sh`/
`gen_subset_speed.sh` are dated by design and untouched.

## V19. The results pipeline quietly changed what it was measuring (FIXED)

**Recorded:** 2026-08-03. **Evidence:** the mode banner `bench_grid` prints
on line 1 of its own output, compared across `results/` files from three
dates, plus the scripts themselves.

Three defects in `tools/results/`, all the same shape: a generated artifact
stopped supporting the claim it exists to back, and nothing errored.

**1. Thread-mode drift (crossover, subset-speed).** `refresh_all.sh` exports
`OMP_NUM_THREADS=$NCPU` for the whole run and rebuilds `bench_grid` as the
parallel binary at step 3/12. `gen_crossover.sh` and `gen_subset_speed.sh`
then "build if missing; if present, trust it", so from 2026-08-02 both ran
12-thread, having been serial through 2026-07-31 (and on Zen4). This is not
cosmetic: the paper's `tab:subset` is captioned *single-threaded*, and the
effect is genuinely mode-dependent. Target-locality pruning is worth
1.11--1.33x serially on M3 Pro at 1% targets, but flattens to ~1.00x at
every $n$ and every target fraction at 12 threads. The regenerated file
therefore silently stopped supporting the table it exists for.
**Fixed:** `gen_subset_speed.sh` pins `OMP_NUM_THREADS=1` (matching the
paper's caption); M3 Pro subset data regenerated serially.

**2. Crossover now emits BOTH modes, and that closed V15's open item.**
Forcing crossover serial would have destroyed something the project
explicitly wants: V15 records that the dispatch tables are calibrated
single-threaded and deliberately reused for parallel, and asks for measured
*parallel* dispatch accuracy as the artifact that tests whether that
heuristic holds ("that artifact does not exist yet and is queued").
`gen_crossover.sh` now writes `crossover_<dev>_<date>_serial.txt` and
`..._parallel.txt`, pinning both thread counts explicitly instead of
inheriting whichever build is on disk.

**Result: the heuristic holds on M3 Pro and does NOT hold on Zen4.** Over
n in {512..8192} against 20 values of k:

- *M3 Pro*: the linear-to-hybrid transition lands at the same k in both
  modes at every n (k=80 at n=512, k=120 for n>=1024). Free.
- *Zen4*: the parallel transition sits **below** the serial one. Hybrid
  already wins from k≈200 at n=4096/8192 where serially it does not win
  until k≈240-260, and from k≈40-80 at n=512-1024. The serially-calibrated
  threshold therefore selects linear across a band of k where hybrid is
  actually faster at 16 threads.

So V15's "serial optimum is a good-enough heuristic for the parallel case"
is platform-dependent, and had only ever been spot-checked on the platform
where it happens to be true. Both results are now disclosed in the paper
rather than reporting only the favourable one. A thread-count-aware
crossover table is the obvious fix; not implemented.

**3. The 1DPC contour data was living under the "current data" filename.**
`results/contour_zen4_{serial,parallel}_q256.csv` (undated) held 1DPC-era
data, but in this repo undated means *current*, and
`refresh_all.sh --device zen4` writes exactly those two names.
`tools/results/plot_zen4_1dpc_2dpc.py` hardcoded the undated path as its
**1DPC** input, so the first Zen4 refresh would have silently redrawn
`fig:1dpc-2dpc` as 2DPC-vs-2DPC, with both curves labelled as different
configurations. Renamed to `..._1dpc.csv` (via `git mv`, history kept), and
both of that script's inputs are now explicit and commented, including why
the 2DPC side deliberately stays pinned to the dated 2026-07-27 snapshot
(pairing pre-fix 1DPC data against post-fix 2DPC data would conflate a
hardware change with a code change).

**Standing lesson:** `bench_grid` prints its thread mode on line 1 of every
output file. That banner is the cheapest available check that a results file
means what a document says it means, and it caught all of this. Read it
before citing a number.

## V20. The cost model could pick correlate FFT sizes smaller than the g operand, which the correlates silently truncate (FIXED)

**Recorded:** 2026-08-03 (initial, wrong diagnosis). **Corrected and fixed:**
2026-08-03, same night. **Evidence:** Zen4 recalibration to a 150,000 ceiling
(776 sizes), controlled rebuilds on the box, live per-level tracing of the
failing case, and algebraic analysis of the wrap corrections confirmed by
direct experiment.

Extending Zen4's calibration and re-measuring `calib_times_ns[]` on the
current reference box made `./bench_grid verify` fail 68 checks. All 68 are
one root cause, isolated to the **tree** engine:

| engine | result |
|---|---|
| `tree` | 37 FAIL, 0 PASS |
| `V2` | 25 FAIL (its reference test runs `engine_tree_ctx`, same bug) |
| `xchk` | 6 FAIL (cross-checks that include tree) |
| `hyb8` | 37 PASS, 0 FAIL |
| `linear` | 31 PASS, 0 FAIL |
| `naive` | 25 PASS, 0 FAIL |
| `subset` | PASS |

**Ruled out, each by direct experiment rather than argument:**

- *New FFTW wisdom*: old config + new wisdom: ALL TESTS PASSED.
- *The wrap-safety-margin fix*: still fails with
  `-DWRAP_SAFE_MARGIN=1000000000` (guard effectively disabled).
- *The `CALIBRATED_MAX_CONV_LEN` change*: still fails with the ceiling
  manually reverted to 262143.
- *The 27 new sizes*: the old 749 sizes with only the **new times** spliced
  in still fails.
- *`polymul_fft_cyclic()` wrap correction invalid at `wrap_m >= 5`*: this
  entry's **original diagnosis, disproven**. That function is bench-only
  (`ICM_BENCH_INCLUDE`, one call site in bench.c's profiling loop) and is not
  on the tree engine's compute path at all. Directly testing it at exactly
  the `(fft_n, wrap_m)` pairs `best_fft_config()` chooses for L=2..64 with
  the real 776-size data: every plan correct against a naive reference,
  including `wrap_m` well past 5.
- *Wrap-correction math in the live correlates*: also correct within its
  domain. Tracing the failing `tree n=16 uniform` (err 1.63e-01) showed
  `build_wrap_m = 0` at every level, and the one nonzero-wrap correlate that
  runs through the non-cached `correlate_fft_pair()` path checked clean
  against a direct reference. The instrumented branch, however, was not the
  branch being taken: at the failing level `build_fft_n == corr_fft_n`, so
  `fft_cache_ok = 1` and propagation goes through
  `correlate_fft_cached_pair_wrap()`, which was uninstrumented. "No
  mismatch printed" meant "branch never ran", a trap worth remembering.

So the trigger is purely the re-measured per-size FFT times. They differ from
the previous box's by only 0.82--1.27x (median 0.996), but that is enough to
flip `best_fft_config()`'s choice at 46 of the first 299 convolution lengths,
generally toward a *smaller* transform with a *nonzero* wrap.

**Real root cause: a missing feasibility bound, not broken wrap math.** Every
correlate implementation (`correlate_fft`, `correlate_fft_pair`,
`correlate_fft_cached_wrap`, `correlate_fft_cached_pair_wrap`) requires
`fft_n >= len_g`: the g operand must fit the transform whole. The cached
variants do `copy_g = min(len_g, fft_n)` (past the bound they silently
TRUNCATE g) while their wrap corrections model the cyclic aliasing of a g
that fully fit. The non-cached variants would heap-overflow instead
(`memcpy(rbuf, g, len_g)` with no clamp). But `best_fft_config()` and
`best_fft_config_joint()` used `len_P`/`p_eff` only to *price* the wrap
correction, never as a constraint, so nothing stopped them from choosing
`fft_n < len_g` (equivalently `wrap_m > len_P - 1`) once timing data made
such a candidate the cost winner.

Concretely, at the failing `n=16` level 3: `corr_conv=13`, `p_eff=5`, so
`len_g=9`; the joint search picked `fft_n=8, wrap_m=5`. `g[8]` is dropped,
`out[4]` loses its true `P[4]*g[8]` term, and the output-side correction
wrongly subtracts `P[0]*g[8]` from `out[0]` (it models an output wrap that
never happened, because g was truncated rather than wrapped). Two O(1)
errors: the observed 16.3%. This also explains the earlier bisection
finding "`wrap_m <= 4` passes, `>= 5` fails": clamping every wrap to 4
restored `wrap_m <= p_eff - 1` feasibility at every failing level; the knob
was different but the boundary was the same.

**Fix (src/cpu/fft_cost_model.h):** a hard feasibility constraint in both
searches: `best_fft_config()` skips candidates with `m >= len_P` in
correlate mode (`len_P > 0`), `best_fft_config_joint()` skips candidates
with `mc >= p_eff`. Pure convolution (`len_P == 0`, the build path) is
unconstrained: `polymul_fft_wrap()` reads its wrap terms from the original
input arrays, so any wrap within `WRAP_SAFE_MARGIN` is feasible there, and
the below-saturation build truncates only structural zeros. The failing
level now selects `fft_n=12, wrap_m=1` (feasible) instead of `8/5`.

**Regression test:** `tools/test_wrap_feasibility.c` pins the bound with a
SYNTHETIC calibration table crafted so the infeasible candidate wins every
cost race it is allowed to enter, device-independent, exactly because the
original bug needed a data-dependent cost race to surface. 4 checks: the
exact n=16 shape through both choosers, a convolution-mode guard proving the
build path was not over-constrained, and a full (len_P, out) sweep. Fails
against the pre-fix chooser, passes post-fix.

**Verification:** Zen4 776-size calibration: 68 failures -> 0
(`ALL TESTS PASSED`), `libicm.a`/`libicm.so` build clean. M3 Pro: ALL TESTS
PASSED, `libicm.a`/`libicm.dylib` clean (M3 Pro's own data never triggered
the bug, but was exposed to the identical mechanism). One nuance: `xchk
n=4096 adversarial` measured 1.14e-13 against a 1e-13 cross-check tolerance
both BEFORE and AFTER the fix (identical to three digits): that cell's
k=10 tree never chose an infeasible config, and the excess is legitimate
rounding drift from different-but-feasible size choices under the new timing
data (the 749-size data measured 7.56e-14 at the same cell, already 76% of
budget; both engines individually pass their 5e-12 V1-reference tolerance
with 10x margin). bench.c now gives n=4096 an intermediate 5e-13 cross-check
bucket, documented in place. The 776-size Zen4 calibration ships.

### V20a. GPU port: the same bug was live on B200, via a THIRD mechanism the CPU does not have (FIXED)

**Recorded:** 2026-08-04. The GPU exposure recorded above as "still open" is
now closed. It was not a clean port: the CPU fix, applied as-is, would have
left the shipped B200 failure completely unfixed.

**Method (no GPU rental).** The GPU decision path (`gpu_cost_model.cu` and
`gpu_plan.cu`) is pure host C++; only `gpu_kernels.cu` / `gpu_exec.cu` need
a CUDA toolchain. Compiling those two translation units *verbatim* against
thin CUDA shims (`cudaStream_t`, `cufftDoubleComplex`, a `cub` stub) makes
the real, shipped planner runnable on the dev Mac against the real
`devices/b200/gpu_fft_config.h`. Every claim below comes from running the
shipped code on the shipped data, not from reading it.

**The bug is live on B200, inside the reported range.** Auditing every FFT
level of 1467 planned (n,k) cells: **142 levels select `fft_n < g_eff`**, so
the correlate silently drops part of g. These are not exotic corners --
n=131072/k=1024, n=262144/k=2048 and n=1048576/k=32768 are clean powers of
two well inside the published heatmap range. The 210/210 heatmap
verification did not catch it because it predates the finding and never
checked this invariant.

**Root cause is a third mechanism, absent on CPU.** At the failing levels
*both* searches return a FEASIBLE size. For n=39224,k=709,ell=3
(`p_eff=193`, `g_eff=526`, `corr_conv=718`) `best_fft_config_gpu()` and
`best_fft_config_joint_gpu()` both return `fft_n=729, wrap=0`. The
infeasible plan comes from the **fused power-of-2 override**
(`gpu_plan.cu`, mirrored in `estimate_candidate_cost()`), which replaces
that with `p2 = next_pow2_int(conv_build) = 512`, wrap 206. `p2` is derived
from the BUILD convolution only, but the same fused kernel also loads the
full g operand for the correlate, and 512 < 526. Constraining only the two
searches, as on CPU, fixes nothing here.

**Numerical consequence, measured.** Transcribing the fused correlate
(cyclic correlation at `fft_n` with `cufftdx_load_real`'s truncation) and
`k_wrap_corr_pair()`'s two correction loops into host doubles, at the real
failing shape: **max relative error 1.39e-01** (13.9%), the same order as
the CPU's 16.3%. At the post-fix size (1024) the same code is exact to
2.8e-16. Feasible-but-nonzero-wrap shapes from neighbouring levels
(`fft_n=256, wrap=79`; `fft_n=128, wrap=16`) are exact to 2e-16, so, as on
CPU, the wrap-correction math is correct *within its domain* and only the
missing feasibility bound is at fault.

**Build/correlate asymmetry: resolved, and it does transfer.** The concern
that GPU build mode might need the same protection (build and correlate
share `cufftdx_load_real`, unlike CPU where they are separate code paths)
was **checked and refuted with evidence, not assumed**. Across all 10809
audited FFT levels there are **zero** build-side violations: not even a
truncated structural-zero tail. The reason is structural: `min_size =
conv_len/2 + 1` inside both searches, taken over `max_conv >= build_conv`,
already forces `fft_n >= cps` (non-below-saturation) or `>= p_eff`
(below-saturation), and the fused `p2 >= conv_build` by construction. So
the fix stays correlate-only on GPU exactly as on CPU, but for a different
reason. `tools/test_gpu_wrap_feasibility.cu`'s Test 2 pins the asymmetry: it
FAILS if build mode ever becomes over-constrained.

**Fix.** One shared predicate, `corr_wrap_feasible(wrap_m, len_P)` in
`gpu_internal.h` (`fft_n >= len_g <=> wrap_m < len_P`), applied at four
sites: the candidate loops of `best_fft_config_gpu()` and
`best_fft_config_joint_gpu()`, plus `fused_p2_feasible()` replacing the bare
`next_pow2_int(conv_build)` in the fused override in BOTH `gpu_plan.cu` and
`estimate_candidate_cost()` (the estimate must model the size the planner
will really choose). The fused bump is self-limiting: if the larger power of
two has no fused calibration the existing `isfinite()` test rejects the
candidate and the cuFFT config stands, which is also what keeps it within
the sizes cuFFTDx can instantiate.

**Verification.** 142 infeasible levels -> **0**, over the same sweep. The
n=39224,k=709,ell=3 level moves from `512/wrap 206` (infeasible) to
`1024/wrap 0`. Neighbouring feasible nonzero-wrap levels keep their choices,
so the bound is not over-tightening. `tools/test_gpu_wrap_feasibility.cu`
(new): 1598 failures against the pre-fix sources, `ALL TESTS PASSED` (32879
checks) against the fixed ones. `tools/test_gpu_cost_model.cu` holds at its
pre-existing 483 passed / 5 failed both before and after: the 5 are the
known B-selection optimality mismatches (V7), not this. Its own plan dump
had been printing an infeasible config all along
(`ell=3 ... fft_n=1024 cwm=414` with `p_eff=385`), now `fft_n=1296 cwm=142`.

**NOT verified: end-to-end numerics on real GPU hardware.** No rental
happened; see the budget note below. What is verified is the planner
decision (shipped code, shipped data) and the arithmetic consequence
(transcribed kernel semantics). What is not is a live `bench_gpu_fused`
run, nor that the fix compiles under nvcc: the changes are plain host C++
in files nvcc compiles, but no CUDA toolchain exists on the dev machine.
**Run `make test_gpu_wrap_feasibility` and `./bench_gpu_fused verify` on the
next B200 session before treating this as fully closed.** *(Done 2026-08-04:
both pass on real hardware; see V20c below.)*

**Why no rental (budget $0.66).** B200 is $5.84/hr on vast.ai: 6.8 minutes,
less than provisioning. Cheap consumer cards (RTX 4090/5090, ~$0.28/hr) are
not drop-in for this codebase, contrary to the assumption that `CUDA_ARCH`
parameterizes the build: `gpu_kernels.cu` hardcodes `cufftdx::SM<1000>()` in
all four cuFFTDx template aliases (Blackwell datacenter only), and
`GPU_VRAM_BYTES = 191 GB` drives the q-batch budget, so a 24 GB card needs
at least three test-only source patches before it can run at all. That is
an unbounded debugging loop against a hard cap, so the money was not spent.

**Follow-up worth considering:** make the cuFFTDx SM trait track
`CUDA_ARCH` rather than hardcoding 1000. It is the single thing that would
make this class of bug verifiable on a $0.28/hr card instead of a $5.84/hr
one. *(Done 2026-08-04; see V20c.)*

### V20b. The published B200 numbers ARE affected (RESULTS.md not final)

**Recorded:** 2026-08-04. Replaying the **pre-fix** planner (the code that
actually produced the published data) over every published B200 cell:

| population | cells | affected |
|---|---|---|
| RESULTS.md systematic (n,k) grid | 32 | **5** |
| `results/gpu_heatmap_b200.csv` | 210 | **21** |
| RESULTS.md 1-second threshold brackets | 4 | **2** |

Affected systematic-grid cells (these are published runtimes in the table):

- n=65,536 k=1,024 · n=262,144 k=1,024 · n=4,194,304 k=1,024
  (all at `ell=3`, `fft_n=512` vs `g_eff=526`, or `ell=2` `512` vs `559`)
- n=1,048,576 k=524,288 (k=n/2) and k=1,048,576 (k=n)
  (`ell=2`, `fft_n=128` vs `g_eff=159`)

Affected heatmap cells (21): (65536, 1024/2048), (131072, 1024/2048),
(262144, 1024/2048), (524288, 1024/2048), (1048576, 2048/32768/65536/
131072/262144/524288/1048576), (4194304, 1024), (8388608, 256/2048),
(16777216, 256/2048), (33554432, 256).

**Both k=n threshold endpoints are affected** (n=1,490,944 and n=1,506,304),
so the published "k = n, 1-second frontier = 1,490,944" bracket rests on
timings from an infeasible plan. The k=100 endpoints are clean.

**Post-fix, all 246 published cells are clean: 0 infeasible levels.**

**What this does and does not mean.** These are *timing* tables, so nothing
in them is a wrong equity value. What is wrong is that the affected cells
were timed on a plan the fix now rejects: a transform smaller than the
operand, which was chosen precisely because it looked cheaper. Post-fix
those cells move to a larger FFT with zero wrap, so their runtimes will
change and must be re-measured before RESULTS.md or the paper's GPU numbers
are treated as final. The likely direction is slower (bigger transform),
but that is not certain: the discarded plan also paid an O(wrap_m^2)
correction (wrap_m=206 and 461 at the worst levels), so the net could go
either way at some cells. Do not guess it; re-run the grid.

**Why the correctness gate never caught this.** `bench_gpu_fused verify` is
the only GPU test that compares against a CPU reference, and auditing its
exact grid (n up to 65,536 basic / 131,072 extended, k in {16, 100, 512,
n/2, n}) gives **zero** infeasible levels pre-fix. The infeasible band lives
around k ~ 1,024-2,048 at n >= 65,536, which the verify grid steps straight
over. Meanwhile `tools/heatmap_gpu.cu` (the source of the "all 210 cells
pass with zero errors" claim) performs **no numerical comparison at all**
(grep: zero references to `icm_equity` or any reference/error computation);
its "error" column means OOM or a CUDA execution failure. Since the
truncation is memory-safe by construction it can never raise one. So
"210/210 pass" always meant "210/210 ran", never "210/210 correct". That
sentence in RESULTS.md should be reworded regardless of this bug.

**Action before RESULTS.md/paper are final:** re-measure at minimum the 5
systematic-grid cells, the 21 heatmap cells, and the k=n threshold search;
and extend the verify grid to cover k ~ 1,024-2,048 at n >= 65,536 so this
band is under the CPU-referenced gate in future.

### V20c. Hardware re-verification and re-measurement: CLOSED (2026-08-04)

Everything V20a/V20b left open was closed on a fresh B200 rental
(vast.ai, ~50 min at $5.00/hr) the same day:

- **The fix compiles under real nvcc and passes on real hardware.**
  `tools/test_gpu_wrap_feasibility`: ALL TESTS PASSED (32,879 checks) on
  the B200 itself, not just the host-shim replay.
- **The verify grid now covers the bug band, and passes.**
  `bench_gpu_fused verify` extended with k=1,024 and k=2,048 at
  n >= 65,536 (the band the historical grid stepped over, which is why 28
  affected cells could be published in the first place): 42/42 PASS
  against the CPU reference, worst error ~1.2e-13 in the new band.
- **All 21 affected heatmap cells re-measured** post-fix via a new
  `heatmap_gpu --cells` option (identical measurement path as the full
  run), spliced into `results/gpu_heatmap_b200.csv` with the pre-splice
  file kept as a dated backup. Direction was mixed, exactly as V20b
  predicted: the k=1,024-2,048 fused/cufft cells got slower on their
  now-feasible plans (worst: n=1,048,576 k=2,048, 175.5 -> 283.0ms;
  n=4,194,304 k=1,024, 671.1 -> 753.7ms), while the k=256 giant-n cells
  got FASTER (n=33,554,432 k=256, 3,740 -> 3,496ms) and the n=1,048,576
  large-k cells moved <1.5%.
- **The k=n threshold bracket survived re-measurement unchanged**: the
  full binary search re-run (`threshold_search_gpu kn`, median of 5,
  trace `results/gpu_threshold_search_kn_20260804.txt`) lands on the SAME
  bracket, 1,490,944 (943.9ms) / 1,506,304 (1,150.5ms). The paper's
  "~1.5 million players in about a second" headline stands, now on a
  feasible plan (the old 918.8ms endpoint was an infeasible-plan timing).
- **V20a's follow-up (cuFFTDx SM trait hardcoded to SM<1000>) is done**:
  `gpu_kernels.cu` now derives it from `CUDA_ARCH` via `ICM_CUFFTDX_ARCH`,
  with per-arch unsupported sizes gated by `cufftdx::is_supported` (the
  planner's `is_cufftdx_supported_fft_n()` stays the runtime source of
  truth and the cuFFT fallback cascade absorbs absent sizes). Verified:
  sm_100 builds and runs all of the above; sm_90 and sm_89 compile clean
  (compile-check only, no non-Blackwell hardware rented).

## Unverified recollections

None outstanding. (The Zen 4 parallel-sweep item previously recorded here was
clarified 2026-07-30 and promoted to V15; the original wording
misread a calibration-scope decision as a data-collection-scope decision.)

## Known data-hygiene issues

- RESOLVED 2026-08-03: the Zen 4 parallel-column two-machine mix and the
  missing Zen 4 threshold artifact were both superseded by the full
  re-measurement on the current reference box; `results/threshold_zen4_*.txt`
  now ship alongside the M3 Pro equivalents.
- Snapshot pruning (2026-08-04, per an explicit project decision to ship
  only the newest data generation): 14 superseded dated snapshots were
  removed from tracking (the Zen4 bench-grid/contour 20260727 and 20260730
  sets, crossover/subset 20260730-0802 sets, the uncited
  `gpu_heatmap_b200_20260730.csv`, and `b_optimal_sweep_zen4_postfix2.csv`).
  All remain in git history. What ships: current undated files, the newest
  dated generation (20260803), dated files with live citations
  (`gpu_heatmap_b200_20260728.csv`, `_20260730_postfix2.csv`,
  `b_optimal_sweep_zen4_2026-07-30.csv`, the threshold traces), and the
  V6/V7-documented `results/evidence/` set.
- Still open: `tools/sweep_best_b.sh` has been run on Zen4 (2026-07-30, full
  grid, `results/b_optimal_sweep_zen4_2026-07-30.csv`), closing this gap on
  one platform; it surfaced V16 (open). Not yet run on M3 Pro or as a GPU
  analogue.
