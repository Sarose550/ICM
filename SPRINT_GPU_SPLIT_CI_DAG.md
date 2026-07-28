# Sprint: GPU refactor + CI leanness — 2026-07-28

Ephemeral — DELETE at R_CLOSE. Not design/decisions law.
Branch: `results-gpu-section` (PR #7).

## Goal

`gpu_plan.cu` restructured into the conventional HPC layout (cost model /
memory / library bindings / orchestration), with every step verifiable as
pure code motion so one B200 rental closes the whole thing. CI trimmed to
1-2 minutes with per-device coverage.

## Lanes

| Lane | Nodes | Model | Risk |
|---|---|---|---|
| A: CI leanness | C1 | deepseek (supervisor audits) | Low, fully parallel |
| B: GPU dead code | G0 | supervisor | Low, unblocks G1-G3 |
| C: GPU split | G1, G2, G3 | supervisor | Pure code motion |
| D: Hardware gate | H1 | supervisor + B200 | One rental, closes C |
| E: Close | F1 | supervisor | |

## Conflicts

- `src/gpu/**` and `Makefile`: owned exclusively by Lane B/C. C1 must not
  touch either.
- `.github/workflows/`, `bench/bench.c`: owned exclusively by C1.
- Lanes A and B/C are fully independent and run concurrently.
- Tip commit: supervisor only.

## The verification contract for Lane C (this is what makes it one rental)

Every node in Lane C is **pure code motion**. After each step, mechanically
assert — not by eye:

1. Every moved function body is **byte-identical** to its original.
2. The set of defined symbols across all GPU TUs is **unchanged** — nothing
   lost, nothing duplicated.
3. The only other deltas are from an **enumerated allowlist**: `static`
   removals, added declarations in `gpu_internal.h`, added `#include`s,
   Makefile object entries.

Anything outside that list is a bug by construction. `tools/verify_code_motion.py`
(built in G1) implements this and is re-run at G2 and G3.

---

### [x] PRE — CI red fix

**Model:** supervisor. **DONE**, commit `a597557`, pushed.
Uncalibrated path now uses schoolbook below conv_len 128 instead of FFT.
`tree n=65536 adversarial` 1.15e-11 -> 1.33e-12; m3_pro unchanged. Verified
with no `fftw_wisdom.dat` present (a stray one changes FFTW plan selection
and made an earlier local run non-reproducible).

---

### [ ] C1 — CI: per-device matrix, 1-2 minute target

**Model:** deepseek. **Depends:** none. Runs concurrently with all of Lane B/C.
**Allowed files:** `.github/workflows/ci.yml`, `bench/bench.c` (only if a new
verify subset mode is needed).
**Forbidden:** `src/gpu/**`, `Makefile`, `src/cpu/icm.c` — Lane B/C owns those.

**Problem:** CI takes 8m10s and is a single bare `make`, which since the
`DEVICE` default change means it only ever tests `devices/generic`.

**Work:**
- Matrix over `m3_pro`, `zen4`, `generic`.
- **`zen4` cannot test zen4 codegen on a shared x86 runner** — `-march=znver4`
  emits AVX-512 the runner may not execute. Override the arch flag so CI
  tests zen4's *calibration data and dispatch decisions*, not its codegen.
  Same for `m3_pro` (tests data/logic, not NEON). Say so in a comment so
  nobody later mistakes this for real per-arch coverage.
- Drop n=65536 from the CI verify set. n=4096-16384 exercises every engine
  and every tier decision; 65536 mostly buys runtime.
- `generic` runs `tools/test_uncalibrated_fallback.c` instead of the full
  grid.
- Keep `make libicm.a` + `libicm.so` in the matrix — a change can pass
  `bench_grid verify` while the shipped library fails to compile, which has
  already happened once.
- **Do NOT loosen any accuracy tolerance.** Speed comes from dropping large
  cases, not from weakening the signal.

**Exit criteria:** total wall-clock under ~2 min; all three devices covered;
tolerances untouched; CI green.
**Kill deadline:** 25 turns.

---

### [ ] G0 — Delete VkFFT dead code

**Model:** supervisor. **Depends:** none. **Blocks:** G1, G2, G3.
**Allowed files:** `src/gpu/**`, `Makefile`.

`ICM_HAVE_VKFFT` is 1 only under `USE_VKFFT`, which requires `VKFFT_INC` on
the make line. **Nothing anywhere passes it** — not the Makefile, not CI, not
any tool script or doc. Verified 2026-07-28. So the dual-dispatch path is
dead: **492 lines across 4 files**, 168 of them inside `gpu_plan.cu`.

| File | Dead lines |
|---|---|
| `gpu_exec.cu` | 302 |
| `gpu_plan.cu` | 168 |
| `gpu_internal.h` | 19 |
| `gpu_api.cu` | 3 |

Remove the guarded blocks, the `VKFFT_FLAGS`/`VKFFT_LIBS` Makefile plumbing,
and the `ICM_HAVE_VKFFT` definition itself.

**Why first:** it deletes Option B's single biggest complication (VkFFT
ifdefs tangled through `allocate_level_buffers`) and shrinks G1 to a clean
~150-line move.

**Exit criteria:** zero `VKFFT`/`vkFFT` references outside git history;
`gpu_plan.cu` down to ~1,777 lines.

---

### [ ] G1 — Extract `gpu_fft_plans.cu` (~150 ln post-G0)

**Model:** supervisor. **Depends:** G0.
**Moves:** `cufft_batch_would_overflow_32bit`, `create_cufft_plan`,
`estimate_cufft_workspace_bytes`.
**Statics to export:** `cufft_batch_would_overflow_32bit`.
**Why first of the three:** `create_cufft_plan` is *already* called cross-TU
from `gpu_exec.cu`, so this boundary is proven in production.
**Also:** build `tools/verify_code_motion.py` here and run it.
**Exit criteria:** code-motion contract passes.

---

### [ ] G2 — Extract `gpu_cost_model.cu` (~800 ln, 23 fns)

**Model:** supervisor. **Depends:** G1.
**Moves:** `estimate_cufft_pipeline_ns` through `best_k_pad_gpu`, plus
`estimate_candidate_cost` and its dependency chain.
**Statics to export:** `cufft_floor_ns`, `calib_batch_for_size`,
`wrap_m_cap_gpu`, `model_ns_per_fma_override`.
**Key finding from the audit:** all 11 symbols needing cross-TU visibility are
**already declared in `gpu_internal.h`**. Zero new declarations. The cost
model is already API-clean; it just happens to share a file with its callers.
Carries the calibration-boundary `static_assert` from `fc7b1b6` with it.
**Exit criteria:** code-motion contract passes.

---

### [ ] G3 — Extract `gpu_memory.cu` (~500 ln)

**Model:** supervisor. **Depends:** G1, G2.
**Moves:** `maybe_init_mem_pool`, `allocate_level_buffers`,
`init_pad_identity`, `allocate_plan_device_memory`, `alloc_device`,
`free_device`, `update_vram_alloc`.
**Statics to export:** `maybe_init_mem_pool`, `allocate_level_buffers`.

**Highest-risk node in the sprint.** This is where the OOM bugs lived, where
the arena retry loop with `q_batch` halving lives, where the CUDA memory pool
is initialised — code already debugged at length on real hardware. Its
failure mode is a **runtime crash, not a compile error**. G0 removes the
VkFFT tangle, and the code-motion contract catches anything that is not pure
relocation, but this is the node to be slowest and most careful on.

**Exit criteria:** code-motion contract passes. `gpu_plan.cu` ends at
~500-600 lines of plan construction.

---

### [ ] H1 — B200 hardware gate (ONE rental)

**Model:** supervisor. **Depends:** G0, G1, G2, G3.

**Capture the baseline BEFORE applying the split**, on the same instance:
```
git stash / checkout pre-split commit
./bench_gpu_fused verify                      > baseline_verify.txt
ICM_GPU_DEBUG_PLAN=1 ./bench_gpu_fused ...    > baseline_plan.txt
./gpu_sample_plans                            > baseline_plans.csv
```
Then apply the split and diff. **`baseline_plan.txt` and `baseline_plans.csv`
must be byte-identical.** That diff is the silent-planner-drift failure mode
the audit flagged as unverifiable without hardware — it is the entire reason
"don't split" was the verdict, and a rental converts it into a checkable fact.

**Setup notes (learned expensively 2026-07-28):** pass the pubkey via
`--onstart-cmd`; vast may not inject the account key. Build with
`DEVICE=zen4`, not bare `make`, or the CPU cross-check is uncalibrated and
~5x slower. Destroy immediately after.

**Exit criteria:** builds under `sm_100`; `bench_gpu_fused verify` 36/0;
plan + sample_plans byte-identical to baseline.

---

### [ ] F1 — Close-out

**Model:** supervisor. **Depends:** all.
Update `CLAUDE.md` directory structure for the new GPU TUs, note the VkFFT
removal, delete this board, final commit + push.
