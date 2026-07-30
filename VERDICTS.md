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

**This is the open item.** The data must be extended to support the new lookup;
see the handoff. Do not "fix" it by reverting the lookup without reading V11.

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

## Unverified recollections

Recorded so the next agent knows they are **not** substantiated, rather than
rediscovering the ambiguity.

- **"We deliberately skipped the full parallel sweep on Zen 4 and stopped at
  serial."** Recalled by the user 2026-07-30. **Not corroborated by the repo.**
  Searched commit messages, `RESULTS.md`, and deleted `HANDOFF.md` revisions.
  Contrary evidence: full parallel grid *and* parallel contour data exist for the
  current reference box (`bench_grid_zen4_parallel_20260727.txt`,
  `contour_zen4_parallel_q256_20260727.csv`) and both are published in
  `RESULTS.md`. The nearest real scope decisions are V10 and CLAUDE.md's note
  that contour sweeps stall at k >= 200,000 and yield partial data through
  ~k=100K. If the user reaffirms this verdict, record it here properly.

## Known data-hygiene issues, not yet resolved

- `RESULTS.md`'s Zen 4 **parallel** column mixes two machines: cells at n=128
  and n=256 match an older undated file while n >= 512 match the cited
  `20260727` file. Serial is clean (all 66 cells match).
- The Zen 4 1-second threshold (n=17,984) has no saved artifact, unlike the GPU
  one.
- `tools/sweep_best_b.sh` has never been run on hardware, so the B-optimality
  artifact whose absence hid V11's bug still does not exist.
