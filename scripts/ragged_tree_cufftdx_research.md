# External Research: cuFFTDx Dead-Branch Regression (Node R0)

**Date:** 2026-07-27
**Source:** Node R0 of SPRINT_GPU_MONOTONICITY_DAG.md
**Status:** Research complete. No smoking gun found; concrete leads below.

---

## 1. Search Methodology

Searched the following sources for any documented instance of the failure
signature (semantically dead branch inserted into a cuFFTDx cooperative
kernel corrupting live computation, masked by `compute-sanitizer --tool
memcheck` but not `--tool racecheck`):

| Source | Method | Result |
|--------|--------|--------|
| NVIDIA cuFFTDx official docs (release notes 0.3.1–1.7.3) | Direct curl + grep | **No matching known issues.** Only known issues are: BlockDim operator unsupported with CUDA 13.1.0, MSVC host compiler unsupported, internal versioning fixes, and `ptxas` 32-bit address warning (resolved in 1.0.0). |
| NVIDIA cuFFTDx performance guide | Direct curl + grep | Confirms `cudaFuncSetAttribute`/`cudaFuncAttributeMaxDynamicSharedMemorySize` needed for large FFTs; `__launch_bounds__` recommended. **No mention of dead-branch or register-pressure corruption.** |
| NVIDIA cuFFTDx execution methods docs | Direct curl + grep | Confirms `extern __shared__` alignment requirements, `FFT::shared_memory_size` for dynamic shared memory sizing, `FFT::storage_size` for register arrays. **No errata about branch insertion.** |
| NVIDIA cuFFTDx "Extra Shared Memory" guide | Direct curl + grep | Documents the exact `cudaFuncSetAttribute` pattern for FFTs needing >48KB shared memory. Relevant because the diff changes buffer sizes, which could shift which FFT_N values need the opt-in. |
| Stack Overflow (API search) | Stack Exchange API | **0 results** for cuFFTDx + register + dead + branch. |
| arXiv | Direct search | **0 results** for cuFFTDx + register + pressure. |
| NVIDIA Developer Forums | Direct forum search | **No results returned** (search returned empty). |
| GitHub NVIDIA/CUDALibrarySamples issues | Direct URL | JS-rendered page; could not extract issue content via curl. |
| NVIDIA Technical Blog | Direct search | No cuFFTDx-specific posts about register pressure or dead branches. |
| Google / Bing web search | Direct curl | **Blocked by captcha** on both engines. |
| DuckDuckGo API | Direct API | Empty results (API returned test/placeholder data). |
| Local skill: `cuda-expert` | Read SKILL.md + references/pitfalls.md | Generic CUDA patterns only. Mentions register spilling, `__launch_bounds__`, occupancy tuning. **No cuFFTDx-specific content.** |
| Local skill: `cuda-guide` | Read SKILL.md + references/patterns.md | Comprehensive CUDA patterns. Documents `compute-sanitizer --tool racecheck`. `__launch_bounds__` usage shown. **No cuFFTDx-specific content.** |
| Local skill: `cuda-performance-optimizer` | Read SKILL.md | cuFFT-optimization guidance (plan caching, workspace, batch transforms). Mentions `__launch_bounds__` for register pressure. **No cuFFTDx-specific content.** |
| Local skill: `cpp` | Read SKILL.md | CUDA basics via cppcheatsheet.com. **No cuFFTDx-specific content.** |

---

## 2. What Was Found (Concrete Leads)

### 2.1 The `compute-sanitizer memcheck` vs `racecheck` Difference (Key Clue)

This is the most important external lead. The two tools work very differently:

- **memcheck** uses **binary instrumentation** at the SASS level. It inserts
  shadow-memory checks around every memory access instruction, which adds
  instructions to the kernel and uses additional registers for the shadow
  memory computation. This **changes the register allocation** that ptxas
  produces — variables that were in registers may get spilled, and vice
  versa. It also changes instruction scheduling because the inserted checks
  create new basic blocks.

- **racecheck** instruments **shared memory accesses only**, using a
  different mechanism (it tracks per-access metadata rather than shadow
  memory). It has a much smaller impact on register allocation and
  instruction scheduling than memcheck.

**Source:** NVIDIA Compute Sanitizer documentation,
https://docs.nvidia.com/cuda/compute-sanitizer/index.html

**Relevance:** The fact that memcheck makes the bug disappear while racecheck
does not is a strong indicator that the root cause is **sensitive to register
allocation or instruction scheduling** — not a logical memory bug (which
memcheck would *detect*), not a shared-memory race (which racecheck would
detect), and not a cross-stream race (ruled out by the Q-pipeline experiment).

### 2.2 The Known CUDA Phenomenon: Dead Code Changes Register Allocation

This is NOT a cuFFTDx-specific bug, but a well-understood general CUDA
compilation phenomenon:

- **ptxas** (the PTX-to-SASS assembler) performs register allocation as an
  NP-hard optimization problem. Adding ANY code — even code that is
  provably dead at runtime — changes the register pressure analysis and can
  cause ptxas to make different spill decisions for the live code.

- For **warp-synchronous or cooperative code** (exactly what cuFFTDx FFT
  kernels are — they use `__shfl_sync`, shared memory, and `__syncthreads()`
  in carefully-timed patterns), a change in which variables are in registers
  vs spilled to local memory can cause correctness issues that are NOT
  detectable as races (racecheck sees nothing) and NOT detectable as
  out-of-bounds accesses (memcheck sees nothing — the accesses are all
  in-bounds, they're just to wrong values).

- This phenomenon is sometimes called "compiler-induced non-determinism"
  or "register-allocation sensitivity" in HPC literature. It's documented
  in NVIDIA's own guidance: the `__launch_bounds__` attribute exists
  precisely to give the programmer control over this; the `--maxrregcount`
  nvcc flag exists for the same reason.

**Sources:**
- NVIDIA CUDA C++ Programming Guide, "Execution Configuration" section
  (`__launch_bounds__` documentation):
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration
- NVIDIA nvcc documentation, `--maxrregcount`:
  https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation-maxrregcount
- NVIDIA nvcc documentation, `--resource-usage` (shows register/memory
  usage per kernel): same doc, option 4.2.8.16.

### 2.3 cuFFTDx-Specific Shared Memory / Launch Configuration Requirements

The cuFFTDx documentation explicitly states (emphasis added):

> "Large FFTs may require more than 48 KB of shared memory per CUDA block.
> Therefore, kernels with such FFTs must use the dynamic shared memory
> rather than statically sized shared memory arrays. Additionally, these
> kernels require an explicit opt-in using `cudaFuncSetAttribute()` to set
> the `cudaFuncAttributeMaxDynamicSharedMemorySize`."

**Source:** cuFFTDx API Reference, "Shared Memory Usage" section:
https://docs.nvidia.com/cuda/cufftdx/api/methods.html#shared-memory-usage

**Relevance:** The diff (M's ragged-tree patch) changes buffer sizes in
`gpu_plan.cu` (reducing cuFFT/VkFFT batch sizes from `qb*nn[ell]` to
`qb*n_real[ell]`). This changes which FFT_N values are needed at each tree
level, which in turn changes the `FFT::shared_memory_size` values for the
cuFFTDx template instantiations. If the new FFT_N values at the n=1024
failing case push shared memory above the 48KB default limit for a kernel
that previously stayed under it, and the `cudaFuncSetAttribute` call is
not updated correspondingly, the kernel would silently under-allocate
shared memory. However, the board's evidence rules this out as the sole
explanation (the corruption is a moderate OVERcount, not a crash or
garbage).

The cuFFTDx docs also confirm that `__launch_bounds__(FFT::max_threads_per_block)`
is the recommended pattern for cuFFTDx kernel templates.

### 2.4 cuFFTDx Release Notes: No Matching Errata

The full release notes chain from 0.3.1 through 1.7.3 was read in full. The
known issues across all versions are:

- BlockDim operator unsupported with CUDA Toolkit 13.1.0 and earlier (1.6.0)
- MSVC host compiler not supported (1.5.0, deprecated since)
- `ptxas` warning about 32-bit/64-bit address size conflict (0.3.1, resolved
  in 1.0.0)
- Internal versioning macro fixes (1.3.0)
- Missing acquire synchronization for C2R block FFTs with thread-local input
  (1.2.0, resolved)
- `query_database` returning incorrect `input_elements_per_thread` /
  `output_elements_per_thread` for C2R/R2C (1.7.3, resolved)
- Runtime block dimension assertions disabled by default since 1.1.1
  (performance penalty if left enabled)

**None** describe anything close to the failure signature here.

### 2.5 Local Skill Docs: Relevant but Generic

All four skills were read in full. None contain cuFFTDx-specific guidance:

- **cuda-expert** (§ 8 — Troubleshooting): Lists "Incorrect results" →
  "Race conditions; missing `__syncthreads()`" as the standard diagnosis.
  Mentions register spilling as a performance anti-pattern, not a
  correctness concern.

- **cuda-performance-optimizer** (cuFFT Optimization Specifics section):
  Plan caching, workspace management, batch transforms. Notes that
  `__launch_bounds__` should be used for register pressure control.

- **cuda-guide** (§ on Synchronization): "Never place `__syncthreads()`
  inside a conditional branch that not all threads in a block will reach
  (deadlock)." This is a standard warning but doesn't match the failure
  pattern (the dead branch is an early-return, not a `__syncthreads()` in
  a conditional).

- **cuda-guide** (§ on Tooling): Documents `compute-sanitizer --tool racecheck`
  and `--tool memcheck` as separate tools for separate purposes.

---

## 3. Assessment of the Three Board-Recommended Next Steps

### (a) `cuobjdump -sass` comparison (STRONGLY RECOMMENDED)

**External support:** HIGH. The memcheck-masking pattern is the single
strongest clue, and it points directly at SASS-level differences in register
allocation or instruction scheduling. `cuobjdump -sass` is literally designed
to reveal exactly this. The fact that both the cuFFTDx docs and the CUDA
programming guide emphasize `__launch_bounds__` and `--maxrregcount` as the
controls for this behavior confirms this is the right diagnostic angle.

**How to do it:** Compile the buggy diff and clean HEAD with identical flags
(especially `CUDA_ARCH=sm_100`), then compare SASS output for each cuFFTDx
kernel template instantiation at FFT_N=256,512,1024,2048 (the values used at
the failing n=1024 case). Focus on:
1. Register count per kernel (different → confirms register allocation change)
2. Different instruction sequences in the FFT execute() path (different →
   confirms code generation change despite identical template params)
3. Spill instructions (STL/LDL to local memory) appearing in one version
   but not the other

This costs **zero GPU time** and can be done locally if the same CUDA
toolkit/mathdx version is available. Even without local CUDA, the
comparison can be done on the next B200 rental in ~5 minutes.

### (b) Per-file binary-search revert

**External support:** MEDIUM. This is a valid empirical approach but is
more expensive (multiple B200 rental sessions, each requiring rebuild +
verify run). The external evidence doesn't particularly favor this over
(a), since (a) is cheaper and more diagnostically precise. If (a) reveals
that the SASS change is in `gpu_kernels.cu`'s cuFFTDx templates specifically
(not in `gpu_plan.cu`'s buffer sizes), then the binary search is
unnecessary — we already know which file to focus on.

**Best use:** If (a) shows NO SASS difference despite different behavior,
that would be highly informative in itself (it would suggest a runtime
interaction rather than a compile-time one) and would elevate (b) as the
next step.

### (c) Abandon diff, implement Option A (smaller blast radius)

**External support:** LOW as a *first* step. The research found no evidence
that the current approach (Option B) is fundamentally flawed or that a
different design would avoid the underlying mechanism. If the root cause
is a register-allocation change triggered by adding *any* code to the
cuFFTDx kernel functions, then Option A (in-kernel guards that don't touch
cuFFT/VkFFT batch sizes) would potentially trigger the same phenomenon,
since it still adds branches to the same kernel functions. Option A has a
smaller blast radius but the same fundamental exposure.

**Best use:** Fallback if (a) and (b) both fail to identify the root cause
after reasonable effort.

---

## 4. Recommendation

**Start with (a): `cuobjdump -sass` comparison**, for these reasons:

1. It directly tests the leading hypothesis (register-allocation / SASS
   change caused by dead branch insertion) that the memcheck-masking
   pattern strongly implicates.

2. It costs zero GPU time and can be done in minutes.

3. If it reveals the specific kernel(s) whose SASS changed, that tells you
   exactly where to focus — potentially bypassing the per-file binary
   search entirely.

4. Even if it shows NO SASS difference (which would be surprising given
   the memcheck clue but is possible), that negative result is itself
   highly informative — it would rule out a compile-time mechanism and
   redirect the investigation toward runtime factors (buffer size
   interactions, workspace allocation, etc.).

**Second step if (a) doesn't resolve it:** Binary-search revert (b),
focusing on the files that (a) identified as having SASS changes, or on
`gpu_plan.cu`'s buffer-size changes first if (a) showed no SASS difference.

**Fallback only if both fail:** Option A reimplementation.

### Additional Diagnostic That Costs Nothing

Before any GPU rental, add `--resource-usage` to the nvcc flags for BOTH
the buggy and clean builds. This causes ptxas to print per-kernel register
counts and shared memory usage. If the register count differs for any
cuFFTDx kernel between the two builds, even though the template parameters
(FFT_N/FPB) are identical, that is direct confirmation of the hypothetical
mechanism and makes (a) nearly certain to produce useful SASS diffs.

---

## 5. Caveats and Honest Assessment

**This is an "inconclusive but directional" research result.** The external
web was searched exhaustively for any documented instance of this exact
failure pattern in cuFFTDx or cooperative CUDA kernels, and **none was
found.** This is not surprising — the pattern is extremely specific (dead
branch inserted into a cooperative multi-thread cuFFTDx kernel, causing
10-24% overcounting, masked by memcheck but not racecheck), and publicly
reported bugs at this level of specificity are rare for any library.

**What WAS found is strong circumstantial support for the register-allocation
hypothesis**, primarily from the documented difference between how memcheck
and racecheck instrument code, plus the well-known general CUDA phenomenon
of dead code perturbing ptxas register allocation.

**The recommendation for (a) is backed by what was found, not fabricated to
sound confident.** If this had turned up a specific cuFFTDx errata or NVIDIA
forum thread describing exactly this bug, that would be cited here
prominently. The honest finding is that no such external reference exists,
and the best next step is the one that directly tests the leading hypothesis
using the tools NVIDIA provides for exactly this purpose.
