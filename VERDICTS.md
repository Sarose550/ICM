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

## V7. The GPU anchors were co-designed with the sequential lookup (and V11 broke that)

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

**Not yet confirmed: the acceptance gate itself (V12).** The full 210-cell
heatmap regen was ~191/211 through (partway into n=33,554,432) when the
rented instance's container exited unexpectedly and the box became
unrecoverable; the vast.ai account had no remaining credit to rent a
replacement. The partial CSV was not preserved (a clean rerun on fresh
hardware is the right move anyway, per this project's own
machine-drift-control practice of not mixing partial runs across
instances). Peak VRAM was climbing steeply through that region (154.8 GB
of 183 GB total at n=33554432,k=8192, still rising) right before the
crash; possibly a real VRAM ceiling at extreme sizes rather than
infrastructure noise, but there is not enough evidence to call it either
way -- no OOM was captured in the container logs, just an unexplained
`exited` status.

**Remaining work, blocked on funding:** rerun the full 210-cell heatmap
regen on a fresh B200 instance and check the V12 monotonicity gate
(currently unconfirmed post-fix; pre-fix baseline was 1 violation, the
broken-anchor run had 8). Do not consider V7 fully resolved until that
gate is checked. Do not "fix" the underlying lookup by reverting it
without reading V11.

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

## V16. Zen4 CPU bselect table is systematically wrong at k=10 (OPEN, unfixed)

**Recorded:** 2026-07-30. **Evidence:** `tools/sweep_best_b.sh --device zen4`,
first run ever on real hardware, `results/b_optimal_sweep_zen4_2026-07-30.csv`.

Every one of the 11 tested `n` values (64 through 65536) at `k=10` dispatches
`B=8` while the directly-measured best is `B=32`, a 37-44% gap at every single
point (worst: n=65536, 44.4%; best: n=32768, 34.0%). No other `k` value shows
this pattern this consistently; the next-worst offenders are scattered
one-offs at 15-27%.

This is not the V11/V7 joint-NN shadowing bug: `test_bselect_lookup.c` passes
113/113 on this exact box at this exact commit, so the lookup algorithm is
correctly finding its nearest calibrated anchor. The defect, if there is one,
is upstream: either the calibration table (`bselect_*` in
`devices/zen4/fft_config.h`, built by `tools/calibrate_best_b.c`) has bad data
at low k, or `tools/calibrate_best_b.c` and `tools/validate_best_b.c` (the
sweep's single-point oracle) measure this regime differently and one of them
is wrong. Both are plausible; neither is diagnosed yet.

**Do not patch this by hand-editing the table.** Same rule as everywhere else
in this file: find the mechanism first. Likely next step: rerun
`calibrate_best_b.c` at k=10 specifically with elevated rep count and compare
against `validate_best_b.c`'s reps/methodology to see where they disagree.

**Not yet checked on M3 Pro or GPU.** This may be Zen4-specific, or may be a
property of k=10 as a boundary case (smallest k in the grid) that reproduces
everywhere.

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
