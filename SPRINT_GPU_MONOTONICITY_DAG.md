# SPRINT_GPU_MONOTONICITY_DAG

<!--
Live strike board managed by the supervisor-dag skill.
Ephemeral — DELETE at R_CLOSE. Not design or decisions law.
-->

Scope: the GPU non-monotonicity investigation arc, post-fix. Two nodes
below (A, B) were already dispatched and completed via direct `deck spawn`
calls before this board existed -- logged here retroactively for a clean
audit trail, per user correction that this should have been tracked on a
board from the start. Everything from here forward goes through the
board properly.

## Graph

```mermaid
graph TD
  A[A_MONOTONICITY_AUDIT] --> E[E_HW_VERIFY_BSELECT_GAP]
  B[B_CALIB_UNIT_TRACE] --> E
  E --> F[F_RECALIBRATE_BSELECT_GAP]
  F --> I[I_DIAGNOSE_THIRD_MECHANISM]
  I --> J0[J0_MEASUREMENT_AB_KPAD_CONFOUND]
  J0 --> J1[J1_DESIGN_BELOW_SAT_FIX]
  J1 --> K1[K1_IMPLEMENT_BELOW_SAT_FIX]
  J0 -.deferred.-> J[J_DESIGN_RAGGED_TREE_GPU_FIX]
  J -.deferred.-> K[K_IMPLEMENT_RAGGED_TREE_GPU_FIX]
  F --> G[G_BUILD_GPU_VALIDATION_HARNESS]
  K1 --> H[H_RUN_THRESHOLD_SEARCH]
  G --> H
```

## Lanes

| Lane        | Nodes                              | Default owner |
|-------------|---------------------------------------|----------------|
| audit       | A_MONOTONICITY_AUDIT (done, logged)    | deepseek       |
| audit       | B_CALIB_UNIT_TRACE (done, logged)      | deepseek       |
| hw-verify   | E_HW_VERIFY_BSELECT_GAP                | supervisor     |
| calibration | F_RECALIBRATE_BSELECT_GAP              | supervisor     |
| tooling     | G_BUILD_GPU_VALIDATION_HARNESS         | deepseek       |
| land        | H_RUN_THRESHOLD_SEARCH                 | supervisor     |

## Conflicts

| Resource                                       | Rule                                          | Nodes touching |
|--------------------------------------------------|--------------------------------------------------|-----------------|
| B200 instance (real money, one at a time)        | Serialized, supervisor-only rental decisions     | E, F, H         |
| `devices/b200/gpu_fft_config.h`                  | Supervisor only, after real hardware confirmation | F               |
| New GPU validation harness tool (path TBD)       | G only                                            | G               |

## Nodes

### [x] A_MONOTONICITY_AUDIT

- **Model:** `deepseek`
- **Depends:** none
- **Allowed files:** (as dispatched) `scripts/analyze_fftsize_bug_blast_radius.py`,
  read-only over `src/gpu/gpu_plan.cu`, `devices/b200/gpu_fft_config.h`,
  `scripts/b200_nonmono_debug_20260726.txt`, `scripts/b200_fix_verified_20260726.txt`,
  `scripts/threshold_search_gpu.cu`, `results/gpu_heatmap_b200.csv`.
- **Status:** DONE (dispatched without a board -- retroactively logged).
  Found the B-selection sparse-grid cliff at n≈1,285,000 (B jumps
  32→96-192). Did NOT reproduce the user's specific "n≈999k→n≈1.05M,
  1000ms→500ms" live observation exactly, but found the underlying
  mechanism (B-selection cliff) independently and correctly via direct
  table inspection. Also claimed a GPU calibration "microseconds
  mislabeled as nanoseconds" unit bug -- **this claim was WRONG**, refuted
  by node B below. Supervisor reviewed the surviving finding (B-selection
  gap) against the raw `gbselect_*` table directly -- confirmed real.

### [x] B_CALIB_UNIT_TRACE

- **Model:** `deepseek`
- **Depends:** none (ran after A, in response to A's unit claim)
- **Allowed files:** read-only over `tools/calibrate_gpu.cu`,
  `tools/calibrate_gpu_best_b.cu`, `tools/calibrate_block_size.py`,
  `tools/calibrate.c`, `src/gpu/gpu_api.cu`, `src/gpu/gpu_plan.cu`,
  `devices/b200/gpu_fft_config.h`, `devices/m3_pro/fft_config.h`.
- **Status:** DONE. Traced the full measurement path and found
  `icm_gpu_measure_fused_pair_ns()` (`gpu_api.cu:747-748`) correctly
  divides the batched measurement by batch count before storing --
  supervisor independently verified this exact line against the live
  file. **Refutes node A's unit-bug claim.** Confirmed CPU calibration
  never batches, so no analogous risk exists there. Attributed the
  earlier "3-4 orders of magnitude" prediction-vs-hardware gap to an
  approximation error in the Python simulation scripts, not the
  production C++ cost model. Also scoped the B-selection densification
  fix as achievable with existing tooling (`calibrate_gpu_best_b`'s
  narrow-around resume mode), ~3-16 min of B200 time for ~8 new points.

### [x] E_HW_VERIFY_BSELECT_GAP

- **Model:** `supervisor`
- **Depends:** A_MONOTONICITY_AUDIT, B_CALIB_UNIT_TRACE
- **Allowed files:** none (measurement only, no source edits this node)
- **Exit criteria:** on a real B200 rental, confirm whether the
  B-selection cliff near n≈1,048,576-1,572,864 causes an ACTUAL wall-
  clock inversion (not just a simulated one) on the `k=n` and/or `k=100`
  curves. A handful of targeted points (e.g. n=1,100,000; n=1,285,000;
  n=1,450,000 at both curve's respective k) via a small extension of
  `remeasure_nonmono.cu`-style tooling, reps>=5 each. Record real numbers
  either way -- confirming "no real inversion" is also a valid, useful
  outcome, not a failure.
- **Kill deadline:** budget alongside F below in one rental if confirmed.
- **Binding law:** HANDOFF.md's B-selection gap section.
- **Status:** DONE (2026-07-26, third B200 rental, contract `45932804`,
  user-approved). Real inversion confirmed on BOTH curves (not just
  simulated): k=n curve n=1,100,000 (1183.3ms) slower than n=1,285,000
  (1141.4ms); k=100 curve n=1,100,000 (181.3ms) slower than n=1,285,000
  (121.3ms). Root cause confirmed compound: B-selection nearest-neighbor
  cliff + a tree-depth power-of-2 boundary crossing (L=16->17) at the
  same point, both confirmed via `ICM_GPU_DEBUG_PLAN=1`. See HANDOFF.md
  "Third B200 rental" section for full detail.

### [x] F_RECALIBRATE_BSELECT_GAP

- **Model:** `supervisor`
- **Depends:** E_HW_VERIFY_BSELECT_GAP (only if E confirms a real
  inversion -- skip if E finds no real hardware effect)
- **Allowed files:** `devices/b200/gpu_fft_config.h` (gbselect_* section
  only -- do not touch the FFT timing tables while re-injecting)
- **Exit criteria:** new calibration anchor points added in the
  n=1,048,576-1,572,864 gap via `tools/gen_calib_skeleton.py` +
  `tools/calibrate_gpu_best_b.cu` (narrow-around mode) +
  `tools/calibrate_block_size.py` injection. Re-run `bench_gpu_fused
  verify` + spot-check E's points afterward to confirm the cliff is
  closed.
- **Kill deadline:** ~20 min B200 time (per B_CALIB_UNIT_TRACE's
  estimate).
- **Binding law:** do NOT hand-edit B values to force monotonicity --
  only real added calibration measurements are acceptable.
- **Status:** DONE for the two gaps tested, but NOT a full guarantee of
  monotonicity everywhere (2026-07-26). n=1,048,576-1,572,864 gap: 3 new
  anchors (n=1,150,000/1,285,000/1,420,000) added via real
  `calibrate_gpu_best_b --narrow-around` measurement (best_B=48-80, not
  the naive 32-or-96+ nearest-neighbor cliff), 32->44 points, rebuilt,
  `bench_gpu_fused verify` 36/0, re-measured: both curves now fully
  monotonic across this gap. A SECOND gap at n=524,288-1,048,576 (same
  mechanism, one B step down) was found while spot-checking and also
  fixed (3 more anchors, 44->47 points, verify still 36/0) -- but a
  re-measurement afterward found ONE inversion still remaining there
  (n=1,000,000 vs n=1,048,576, same B/L/nblocks yet ~24% timing gap) that
  is NOT the B-selection cliff and is NOT yet understood -- see HANDOFF.md
  "Third B200 rental" section, item 2. **Also found and fixed A REAL BUG
  in `tools/calibrate_block_size.py`** (the orchestrator meant to make
  this reproducible for future devices/users): Step 2/3 unconditionally
  discarded the ENTIRE existing calibration table on every run, which
  would have silently reverted both fixes above the next time anyone ran
  it. Fixed with a merge instead of overwrite (`read_existing_table()` +
  seed-then-merge), verified locally via a round-trip test against the
  real committed header (no GPU needed for this part). CSVs and ad-hoc
  probe tools from this session kept at `scripts/b200_session_20260726/`
  for provenance.

### [x] G_BUILD_GPU_VALIDATION_HARNESS

- **Model:** `deepseek`
- **Depends:** none (can run in parallel with E/F, no B200 needed to draft)
- **Allowed files:** create a new tool under `scripts/` or `tools/` (path
  TBD by the node) implementing HANDOFF.md Next Steps item 3: a GPU
  analogue of `./bench_grid crossover` that, across a grid of (n,k)
  points, runs the actual dispatched configuration AND brute-forces
  nearby alternatives on real hardware, flagging any point where the
  dispatched choice isn't actually fastest.
- **Exit criteria:** tool written and reviewed (supervisor line-by-line,
  same standing practice), ready to run on the next rental. Does not need
  to run this sprint.
- **Kill deadline:** 30 min.
- **Binding law:** HANDOFF.md Next Steps item 3.

### [x] I_DIAGNOSE_THIRD_MECHANISM

- **Model:** `supervisor`
- **Depends:** F_RECALIBRATE_BSELECT_GAP
- **Allowed files:** `src/gpu/gpu_plan.cu` only if a real fix is found and
  confirmed on hardware (read-only investigation otherwise)
- **Exit criteria:** explain why n=1,000,000 and n=1,048,576 (identical
  B=32, L=16, nblocks rounds to the same 32768) differ in wall-clock time
  by ~24% (632ms vs 512ms) on the k=n curve. Start with
  `ICM_GPU_DEBUG_PLAN=1` diffing the two plans' per-level `fft_n`/`bwm`/
  `cwm`/tier choices line by line -- the same technique that found and
  fixed the wrap-penalty tier-lock-in bug and the B-selection cliffs
  earlier this session. If a genuine cost-model bug, fix and
  hardware-verify (`bench_gpu_fused verify` 36/0 + re-measure the
  specific pair) before marking done.
- **Kill deadline:** ~20-30 min B200 time for diagnosis; more if a fix is
  found and needs verification.
- **Binding law:** do NOT force monotonicity by hand-picking a worse
  config -- same rule as F.
- **Also do while budget allows:** a broader systematic sweep (not just
  spot checks) across the whole `gbselect_n` range via
  `scripts/gpu_dispatch_validate.cu` (node G, already written/reviewed)
  before trusting monotonicity generally -- this session found 3 distinct
  mechanisms by testing only 2 gaps, so more are plausible elsewhere.
- **Status:** DONE (2026-07-26), diagnosed WITHOUT a B200 rental -- pure
  source read, zero cost. **Root cause found and confirmed, and it is
  NOT a cost-model cliff like the first two bugs.** CPU's
  `tree_build_levels()`/`tree_propagate_g()` (`src/icm.c`) implement a
  genuine ragged tree: `n_real[]` tracks the true non-padding node count
  at each level (`n_real[0]=n_leaves`, `n_real[ell]=ceil(n_real[ell-1]/2)`
  going up), and the merge loop explicitly checks
  `if (2*j+1 >= nr_child)` -- a lone unpaired real node is just
  `memcpy`'d forward to the next level, no wasted multiply/FFT against a
  phantom sibling. GPU's planner (`src/gpu/gpu_plan.cu`) computes the
  identical `n_real[]` array (same formula, confirmed line-for-line) but
  it is used ONLY for cost estimation -- grepped every execution kernel
  launch in `src/gpu/gpu_exec.cu` (5 call sites) and all 5 size their
  work off `plan->nn[ell]` (the padded, power-of-two-derived width), NOT
  `plan->n_real[ell]`. There is no equivalent anywhere in
  `src/gpu/gpu_kernels.cu` of the CPU's "is this child real or padding,
  skip if padding" check. Concretely: n=1,000,000 at B=32 has 31,250 real
  blocks, not a power of two, so the tree pads to 32,768 slots and every
  GPU kernel (build, FFT, wrap-correction) runs full computation on all
  32,768 slots including the 1,518 that don't correspond to real player
  data, at every one of the 16 tree levels (the real:padded ratio stays
  roughly constant level to level since both series roughly halve each
  step). n=1,048,576 needs zero padding (1,048,576/32 = 32,768 exactly),
  so it does zero wasted work anywhere. This is a real architectural gap
  in the GPU implementation relative to CPU, not a mis-tuned decision --
  confirmed by direct code inspection, not a hypothesis.

### [x] J0_MEASUREMENT_AB_KPAD_CONFOUND

- **Model:** `supervisor`
- **Depends:** I_DIAGNOSE_THIRD_MECHANISM
- **Status:** DONE (2026-07-26, fourth B200 rental, contract `45937997`,
  autonomous continuation, user-approved before leaving). Measurement A
  (fixed k=256 for both n=1,000,000 and n=1,048,576): n=1,000,000 measured
  114.4ms, n=1,048,576 measured 116.1ms -- correctly monotonic, correct
  direction, ~1.5% gap, consistent with the small ragged-tree padding-
  waste estimate from node J's design doc. **This refutes the ragged-tree
  padding-waste bug as the dominant cause of the originally-measured
  23.5% k=n-curve inversion** -- it's real (confirmed by direct code
  read: `plan->nn[ell]` used instead of `plan->n_real[ell]` at 5 execution
  call sites in `gpu_exec.cu`) but small in practice (~1-2%), not the main
  story.

  Follow-up: captured full `ICM_GPU_DEBUG_PLAN=1` output for the ORIGINAL
  k=n comparison (n=1,000,000 vs n=1,048,576, each with k=n) and diffed
  line-by-line. **Found the real, dominant mechanism**: at tree level
  ell=14, both n values have identical child poly size `cps=524288`.
  `below_sat[ell]` (`src/gpu/gpu_plan.cu` line 529:
  `if (psz[ell] == 2 * cps && cps >= 2) below_sat[ell] = 1;`) is a STRICT
  EQUALITY check. For n=1,048,576, `psz[14] = 1,048,576 = 2*cps` exactly
  (k=n is itself a power of two, so the natural doubling sequence lands
  exactly on the cap) -- `below_sat` fires, halving the effective
  polynomial size (`p_eff = cps/2+1 = 262,145` instead of `cps=524,288`).
  For n=1,000,000, `psz[14] = 1,000,000 < 2*cps = 1,048,576` (k=n=1,000,000
  is 7-smooth so `k_pad=k` exactly, same as the other case, but the
  doubling sequence doesn't land exactly on this k because k itself isn't
  a power of two) -- misses the equality by 48,576, `below_sat` does NOT
  fire even though the real degree needed is still meaningfully below the
  full doubled size, forcing the full, un-halved `p_eff=cps=524,288` path.
  Confirmed in the raw debug output: at ell=14, `fft_n=2,097,152` for
  n=1,000,000 vs `fft_n=1,048,576` for n=1,048,576 (exactly 2x), with
  `school_build` cost estimates of `46.023ms` vs `11.506ms` (~4x) at that
  one level alone -- large enough to plausibly explain most of the
  originally-measured 23.5% wall-clock gap on its own. **CPU has the
  identical strict-equality check** (`src/icm.c` line 1151, same
  `psz[ell] == 2 * cps && cps >= 2` form, same consumers at lines 1183-
  1230 and 1504-1511) -- not chased further this session (CPU dispatch
  is separately validated via `bench_grid crossover`), but flagged as a
  latent inefficiency there too, worth a future look.

  **This supersedes the ragged-tree-padding fix as the primary target.**
  Node J's design doc (`scripts/gpu_ragged_tree_fix_plan.md`) is still
  real and worth doing eventually (small, safe, ~1-2% win), but the
  `below_sat` exact-equality bug is the dominant mechanism and needs to
  be fixed first. New node J1 below replaces J as the active work item;
  J's Option B is deferred, not abandoned.

### [ ] J1_DESIGN_BELOW_SAT_FIX

- **Model:** `deepseek`
- **Depends:** J0_MEASUREMENT_AB_KPAD_CONFOUND
- **Allowed files:** read-only over `src/gpu/gpu_plan.cu`,
  `src/gpu/gpu_exec.cu`, `src/gpu/gpu_kernels.cu`, `src/icm.c` (read the
  `below_sat`/`is_below` consumers at lines ~1183-1230 build and
  ~1504-1511 propagate, and the CPU `tree_build_levels()`/
  `tree_propagate_g()` functions that actually use `is_below` to decide
  `build_conv_len`/`p_eff`/`g_eff_max` -- this is CORRECTNESS-CRITICAL
  numerical code, not just a cost estimate), `HANDOFF.md`. Write-allowed:
  a single new file, `scripts/gpu_below_sat_fix_plan.md` (design doc
  only -- no source edits from this node).
- **Exit criteria**: a written design doc explaining (1) the actual
  mathematical/numerical MEANING of `below_sat` -- why does
  `psz[ell] == 2*cps` exactly let the code safely halve `p_eff` to
  `cps/2+1` and shrink `build_conv_len`/`g_eff_max`, in terms of what
  polynomial coefficients are actually needed vs discarded (read the
  consumers, don't guess); (2) whether the exact-equality check is
  mathematically REQUIRED for correctness, or whether it's overly strict
  and a numerically-safe generalization exists (e.g. `psz[ell] >= 2*cps`
  might be UNSAFE if it changes which coefficients get kept vs truncated
  incorrectly when they're not exactly equal -- reason through what
  happens to numerical correctness if `psz[ell] < 2*cps`, which is
  exactly the n=1,000,000 case, using the SAME `p_eff = cps/2+1`
  truncation as the exact-equality case); (3) if a safe generalization
  exists, specify EXACTLY what changes (condition, and whether `p_eff`/
  `build_conv_len`/`g_eff_max` formulas need to change too, not just the
  trigger condition) in both `src/gpu/gpu_plan.cu` (GPU) and, separately
  flagged as an optional follow-up NOT for this sprint, `src/icm.c` (CPU,
  same bug, out of scope, do not touch); (4) if NO safe generalization
  exists (i.e. the exact-equality requirement is real and this is just
  the honest cost of a non-power-of-two k on this specific code path,
  not a bug), say so clearly and explain why, and instead evaluate
  whether a DIFFERENT k_pad choice for non-power-of-two k (biasing
  `best_k_pad_gpu()` towards values that land exactly on a `psz[ell]=2*cps`
  boundary at the levels that matter most) could sidestep the problem
  without touching the correctness-critical `below_sat` logic at all --
  this is likely the safer fix if the equality truly can't be relaxed.
  (5) explicit recommendation for what to measure/verify on the next
  B200 rental before implementing anything.
- **Kill deadline:** 45 min.
- **Binding law:** this touches numerical correctness of a poker-equity
  library -- an incorrect generalization could silently produce wrong
  equity values, not just wrong timings. If in doubt about safety, say so
  explicitly rather than proposing a confident-sounding but unverified
  change. Do NOT propose anything that isn't independently justified by
  reading the actual consumer code, not by pattern-matching to the two
  previously-fixed bugs (which were pure cost-model/dispatch bugs, not
  correctness-adjacent truncation logic like this one).

### [ ] J_DESIGN_RAGGED_TREE_GPU_FIX (deferred, not abandoned)

- **Model:** `deepseek`
- **Depends:** I_DIAGNOSE_THIRD_MECHANISM
- **Allowed files:** read-only over `src/gpu/gpu_kernels.cu`,
  `src/gpu/gpu_exec.cu`, `src/gpu/gpu_plan.cu`, `src/gpu/gpu_internal.h`,
  `src/icm.c` (CPU reference for the ragged-tree pattern already proven
  correct there), `HANDOFF.md`. Write-allowed: a single new file,
  `scripts/gpu_ragged_tree_fix_plan.md` (design doc only -- no source
  edits from this node).
- **Exit criteria:** a written fix-design doc covering: (1) exactly which
  kernel(s)/call sites in `gpu_exec.cu`/`gpu_kernels.cu` would need to
  skip or shrink work for padding slots beyond `n_real[ell]` at each of
  the 5 launch sites identified in node I; (2) whether a GPU-appropriate
  equivalent of CPU's per-node branch (`if (2*j+1 >= nr_child)`) is even
  the right shape for a CUDA kernel, given warp-level thread divergence
  and batch-uniformity concerns that don't exist in CPU's scalar loop --
  concretely reason about whether skipping causes harmful divergence
  within a warp/block or whether padding slots can cheaply be excluded at
  the batch/launch-configuration level instead (e.g. launching with
  `n_real[ell]`-sized batches directly rather than branching inside the
  kernel); (3) at least two concrete implementation options ranked by
  expected benefit vs. implementation/verification risk, given this
  bug's real but narrow footprint (only matters when nblocks isn't a
  power of two, which is most real-world n); (4) how the estimated
  wasted-work fraction was checked against the observed ~24% measured
  slowdown (n=1,000,000 pads 31,250 real blocks up to 32,768, roughly a
  constant ~4.6% relative padding fraction per level per the geometry
  math in node I -- reason about whether raw wasted-compute alone
  plausibly explains the full 24%, or whether the padding is ALSO
  perturbing per-level FFT-size choices the way the first two bugs did,
  compounding the effect); (5) explicit recommendation for what to
  measure on the next B200 rental to confirm whichever fix is chosen,
  before it's implemented for real. Do NOT write or modify any GPU
  source file -- this node produces a plan for the supervisor to
  implement and hardware-verify, not a patch.
- **Kill deadline:** 45 min.
- **Binding law:** ground every claim in the actual current source (this
  session already found and corrected two prior instances of a worker
  stating something wrong with confidence -- verify against the live
  files, don't extrapolate from the summary above alone). Do NOT propose
  "just pad harder" or any change that fixes this by making the timing
  curve look smoother without reducing real wasted work -- the fix must
  address the actual mechanism (real GPU compute wasted on non-existent
  data), not the appearance of it.

### [ ] K_IMPLEMENT_RAGGED_TREE_GPU_FIX (deferred, not abandoned)

- **Model:** `supervisor`
- **Depends:** J_DESIGN_RAGGED_TREE_GPU_FIX
- **Allowed files:** `src/gpu/gpu_kernels.cu`, `src/gpu/gpu_exec.cu`,
  `src/gpu/gpu_plan.cu` (whichever J's chosen option actually touches)
- **Exit criteria:** implement J's recommended option, hardware-verify on
  the next approved B200 rental: `bench_gpu_fused verify` 36/0 (no
  regression), then re-measure n=1,000,000 vs n=1,048,576 (and ideally a
  couple more ragged/non-ragged pairs) to confirm the gap closes for a
  real, attributable reason, not by accident.
- **Kill deadline:** budget-dependent, plan with the user before renting.
- **Binding law:** do NOT rent without explicit user go-ahead (standing
  rule this whole sprint).
- **Status:** deferred behind K1 (see J0) -- real ~1-2% win, small
  relative to the below_sat bug, worth doing once K1 lands and there's
  budget left.

### [x] K1_IMPLEMENT_BELOW_SAT_FIX

- **Model:** `supervisor`
- **Depends:** J1_DESIGN_BELOW_SAT_FIX
- **Allowed files:** `src/gpu/gpu_plan.cu` only (per J1's scope -- CPU
  `src/icm.c` is explicitly out of scope this sprint even though it has
  the same bug)
- **Exit criteria:** implement J1's recommended fix (or the k_pad-bias
  sidestep if J1 concludes the equality check can't be safely relaxed),
  hardware-verify on the next approved B200 rental: `bench_gpu_fused
  verify` 36/0 (no regression -- this is correctness-critical, verify
  extra carefully), then re-measure n=1,000,000 vs n=1,048,576 on the
  actual k=n curve (both with k=n, not the fixed-k=256 control) to
  confirm the gap closes for the diagnosed reason. Also spot-check a
  couple more non-power-of-two-k=n points for the same signature.
- **Kill deadline:** budget-dependent.
- **Binding law:** do NOT rent without explicit user go-ahead in a normal
  session; this autonomous session has standing approval from the user
  before they stepped away, but still verify extra carefully given this
  touches correctness-critical truncation logic, not just a cost
  estimate -- if verify doesn't pass cleanly or the fix is ambiguous,
  stop and document rather than force it through.
- **Status:** DONE, hardware-verified (2026-07-26). Fix implemented in
  `src/gpu/gpu_plan.cu` (three changes: generalized `below_sat` trigger
  in `build_tree_geometry()`, `g_eff_max` clamp in both
  `estimate_candidate_cost()` and `build_plan_metadata()`), matching J1's
  design doc exactly -- supervisor independently re-derived the
  mathematical justification from the actual `psz[]` doubling-then-
  capping structure before applying (not just trusting J1's write-up),
  confirmed the generalized trigger fires for exactly one additional
  level (the transition boundary) versus the original, and confirmed the
  clamp is genuinely necessary (without it, the boundary level can have
  `psz[ell]` as low as `cps+1`, below the unclamped `cps+cps/2`, a real
  out-of-bounds read).

  **First verification attempt (fifth B200 rental, contract `45953979`)
  wasted most of the session's budget** ($6.07 -> $0.22): a correctness
  check was launched at the original bug-discovery scale
  (n=k=1,000,000) comparing GPU output against the CPU reference, and
  ran for 45+ minutes without completing before being killed. Root
  cause, confirmed by reading the actual calibration tables afterward
  (not guessed): `bench_gpu_fused`'s CPU reference is compiled against
  `devices/m3_pro` calibration constants regardless of what CPU actually
  runs it, and those tables' calibrated range tops out well below
  1,000,000 (`crossover_n[]` to 16,384, `bselect_n[]` to 65,536,
  `calib_sizes[]` to 131,072) -- past that range the CPU reference's own
  per-level FFT-vs-schoolbook decision can fall back to brute-force
  O(len^2) multiplication at a huge convolution length, plausibly
  explaining the stall. This was a real, avoidable mistake: the test's
  complexity should have been reasoned through by reading the dispatch
  code before running it on paid hardware, not discovered by waiting.

  **Second attempt, done right (sixth B200 rental, contract `45957726`)**:
  rewrote `scripts/verify_below_sat_fix.cu` (now committed) to split
  correctness checking from timing measurement -- small-scale points
  (n up to 16,384, safely within every M3-Pro calibration ceiling) get a
  full GPU-vs-CPU-reference correctness comparison; large-scale points
  (the original n=1,000,000/1,048,576 discovery scale) get GPU-only
  timing, never a CPU reference call. `bench_gpu_fused verify`: **36/0,
  no regression**. Small-scale correctness: **7/7 PASS**, max relative
  error ~1e-14 to 1e-15 (floating-point noise), including the exact
  `g_eff_max` clamp boundary case (k=513=2^9+1) and a mid-scale trigger
  case (n=k=10,000) beyond the original single-level example. Large-scale
  timing at the actual discovery scale: **n=1,000,000 moved from 632ms
  to 514.3ms; n=1,048,576 moved from 511.9ms to 511.7ms** -- the
  inversion is resolved, the gap collapsed from 120ms/23% to 2.6ms/0.5%
  (noise level), and the boundary-clamp case (k=524,289) ran cleanly
  with no crash. **This fix is now genuinely hardware-verified, both for
  correctness and for resolving the originally-measured inversion.**
  Session cost for this rental: ~$0.43 (vs ~$5.85 wasted on the first
  attempt) -- the difference between reasoning about a test's cost
  before running it and finding out the hard way.

### [ ] H_RUN_THRESHOLD_SEARCH

- **Model:** `supervisor`
- **Depends:** E_HW_VERIFY_BSELECT_GAP, F_RECALIBRATE_BSELECT_GAP,
  K1_IMPLEMENT_BELOW_SAT_FIX, G_BUILD_GPU_VALIDATION_HARNESS
- **Allowed files:** none (execution only)
- **Exit criteria:** `scripts/threshold_search_gpu.cu` run for real,
  producing the actual 1-second-threshold numbers for both `k=n` and
  `k=100` curves, only once monotonicity is hardware-confirmed (not just
  simulated) along both curves.
- **Kill deadline:** ~10 min B200 time.
- **Binding law:** do not run before E/F/G land -- this was the whole
  point of this investigation arc.
