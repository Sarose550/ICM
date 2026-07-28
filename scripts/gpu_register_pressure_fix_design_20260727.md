# GPU Register Pressure Fix Design — Ragged-Tree cuFFTDx Boundary Guard

**Date:** 2026-07-27
**Node:** GPU_REGISTER_PRESSURE_FIX_DESIGN (SPRINT_GPU_MONOTONICITY_DAG.md)
**Status:** Design doc only — no implementation, no hardware testing authorized here.

---

## 0. Context (Condensed From Prior Investigation)

### 0.1 The Bug

M's ragged-tree patch (~520 lines across 4 files, currently UNCOMMITTED in
the working tree) added a boundary guard to `k_cufftdx_build_parent_r2c` and
`k_cufftdx_corr_pair_parent_r2c` (`src/gpu/gpu_kernels.cu`, lines ~352 and
~399).  The guard handles the case where the last real parent at a tree level
has only one real child (`2*p+1 >= n_real_child`) — it copies the left child
through as identity and returns early, skipping the cuFFTDx FFT pipeline.

The guard contains a debug counter (`g_debug_boundary_hits` atomicAdd) placed
BEFORE the per-thread register arrays that cuFFTDx's cooperative `execute()`
depends on:

```cpp
// Current (buggy) shape — both kernels follow this pattern:
if (p >= nparents) return;
// ... compute L, out pointers ...
if ((size_t)(2 * p + 1) >= (size_t)n_real_child) {
    if (threadIdx.x == 0) atomicAdd(&g_debug_boundary_hits, 1ULL);  // ← LINE ~352
    for (int i = threadIdx.x; i < pps; i += blockDim.x)             // copy loop
        out[i] = (i < cps) ? L[i] : 0.0;
    return;                                                          // early return
}
// ⬆ GUARD (with atomicAdd) is ABOVE the register arrays ⬆
complex_t a[R2C::storage_size];    // ← register arrays for cuFFTDx
complex_t b[R2C::storage_size];
extern __shared__ ... shared_mem[];
// ... cuFFTDx R2C execute, multiply, C2R execute ...
```

### 0.2 Hardware-Confirmed Register Shift

`cuobjdump -res-usage` comparison (real B200, contract `46026441`,
2026-07-27) between clean git HEAD and the buggy working tree, at the exact
template instantiation relevant to the failing `n=1024` test case:

| Kernel | Clean HEAD | Buggy Diff | Delta |
|--------|-----------|------------|-------|
| `k_cufftdx_build_parent_r2c<1024,1>` | 116 regs | 124 regs | **+8** |
| `k_cufftdx_corr_pair_parent_r2c<1024,1>` | 126 regs | 124 regs | **−2** |

The guard is confirmed dead at runtime for this test case (0 hits via the
debug counter), and the test case has exact power-of-two block counts at
every tree level (the guard condition `2*p+1 >= n_real_child` is provably
unsatisfiable).  The register shift is therefore a pure compiler artifact
from the dead branch's presence — not from any runtime behavior.

### 0.3 Mechanism (External Research Corroboration)

`scripts/ragged_tree_cufftdx_research.md` (node R0) found that:

- `compute-sanitizer --tool memcheck` uses binary instrumentation that
  changes register allocation — this explains why memcheck masked the bug
  while `--tool racecheck` did not.

- Dead-code insertion perturbing ptxas register allocation is a
  well-understood general CUDA phenomenon, especially dangerous for
  warp-synchronous/cooperative code (exactly what cuFFTDx FFT kernels are).

- No public source documents this EXACT failure signature (dead branch in
  cuFFTDx kernel → 10-24% overcounting → masked by memcheck), but the
  underlying mechanism (register-allocation sensitivity) is well-known.

- NVIDIA's `__launch_bounds__` and `--maxrregcount` exist precisely because
  this class of problem is real.

### 0.4 Why Register Count Matters for cuFFTDx

cuFFTDx's cooperative FFT (`R2C().execute(a, shared_mem)`) uses a carefully
choreographed sequence of `__shfl_sync`, shared memory, and `__syncthreads()`
across all threads in the block.  The template computes `R2C::storage_size`
and `R2C::elements_per_thread` at compile time based on `FFT_N` and the
assumed register budget.  If the actual register allocation differs from what
the cuFFTDx library authors assumed (because the compiler had to accommodate
an unrelated branch's register demands), the internal register/shared-memory
choreography can silently break — producing wrong numerical output without
any CUDA error, race condition, or out-of-bounds access.

The measured shifts (+8/−2 registers) are small but sufficient: cuFFTDx's
internal register allocation is tightly packed, and an extra 8 registers for
one kernel changes which variables the compiler puts in registers vs spills
to local memory.  For a cooperative shuffle-based algorithm, a spilled
variable that should be in a register breaks the shuffle pattern.

---

## 1. Candidate Restructurings

### 1.1 Candidate A: Remove `g_debug_boundary_hits` AtomicAdd Only

**What changes** (pseudocode diff):

```
-  __device__ unsigned long long g_debug_boundary_hits = 0;
-  extern "C" void icm_gpu_debug_reset_boundary_hits() { ... }
-  extern "C" unsigned long long icm_gpu_debug_get_boundary_hits() { ... }

   // In k_cufftdx_build_parent_r2c:
   if ((size_t)(2 * p + 1) >= (size_t)n_real_child) {
-      if (threadIdx.x == 0) atomicAdd(&g_debug_boundary_hits, 1ULL);
       for (int i = threadIdx.x; i < pps; i += blockDim.x)
           out[i] = (i < cps) ? L[i] : 0.0;
       return;
   }

   // In k_cufftdx_corr_pair_parent_r2c:
   if ((size_t)(2 * p + 1) >= (size_t)n_real_child) {
-      if (threadIdx.x == 0) atomicAdd(&g_debug_boundary_hits, 1ULL);
       // ... copy loop unchanged ...
       return;
   }
```

The boundary-guard branch, the copy loop, and the early return all stay.
The ONLY thing removed is the atomicAdd line and the three associated
symbol-definition/accessor functions (~20 lines total).

**Specific mechanism by which this should help:**

The `atomicAdd(&g_debug_boundary_hits, 1ULL)` instruction requires:

1. A 64-bit global memory address register holding `&g_debug_boundary_hits`
   (a `__device__` symbol, not a kernel parameter — the compiler must
   materialize this address in a register).
2. The `ATOMG.E.ADD.64` SASS instruction itself, which may have specific
   register constraints (e.g., requiring the address in an even register
   pair, or the value in specific registers).

The SASS diff already confirmed that every cuFFTDx kernel instantiation in
the buggy build gained a new `ATOMG.E.ADD.64` that is absent from clean HEAD.
This instruction didn't exist before — its register demand (for the address
and the atomic operands) is purely additive to the kernel's register budget.

The copy loop in the guard (`for (int i = threadIdx.x; i < pps; ...`) uses
registers that are mostly already live in the normal path (`pps`, `cps`, `L`,
`out`, `threadIdx.x` — all needed for the FFT path too).  The atomicAdd is
the one thing in the guard path that introduces a genuinely NEW register
demand not shared with the FFT path.  Removing it may reduce the union of
both paths' register sets enough to restore the original allocation.

**Why this might NOT be sufficient:**

The branch structure itself — even without the atomicAdd — creates a second
control-flow path before the register array declarations.  The compiler still
sees two paths (guard-taken vs guard-not-taken) and must allocate registers
for the union.  The copy loop's registers (`i`, the loop counter) might still
shift things slightly, though less dramatically than the atomicAdd's address
register.  The +8/−2 shift might shrink to +2/0 or similar — still a shift,
potentially still enough to perturb cuFFTDx.

**Confidence:** MEDIUM.  The atomicAdd is the most "alien" thing in the guard
path — it's the only global memory operation, the only atomic, and the only
thing requiring a new address register not otherwise live.  Removing it is
the obvious first experiment.  But it may not be the whole story.

**Risk:** ZERO.  The counter already served its purpose (confirmed 0 hits on
the failing test case).  Removing it loses no capability — if we ever want
boundary-hit instrumentation again, we can add a different mechanism
(e.g., a kernel parameter rather than a device symbol).

**Blast radius:** Minimal — ~20 lines removed, no control-flow change, no
data-flow change, no launch-configuration change.

---

### 1.2 Candidate B: Move Register-Array Declarations Before the Guard

**What changes** (pseudocode diff):

```diff
  // k_cufftdx_build_parent_r2c:
  int p = (int)blockIdx.x * FPB + (int)threadIdx.y;
  if (p >= nparents) return;

  const double *L = child + (size_t)(2 * p) * (size_t)child_stride;
  double *out = parent + (size_t)p * (size_t)parent_stride;

+ // Declare register arrays BEFORE the guard
+ complex_t a[R2C::storage_size];
+ complex_t b[R2C::storage_size];
+ extern __shared__ __align__(alignof(double2)) complex_t shared_mem[];
+
  if ((size_t)(2 * p + 1) >= (size_t)n_real_child) {
      if (threadIdx.x == 0) atomicAdd(&g_debug_boundary_hits, 1ULL);
      for (int i = threadIdx.x; i < pps; i += blockDim.x)
          out[i] = (i < cps) ? L[i] : 0.0;
      return;
  }

- complex_t a[R2C::storage_size];
- complex_t b[R2C::storage_size];
- extern __shared__ __align__(alignof(double2)) complex_t shared_mem[];
-
  const double *R_ptr = child + (size_t)(2 * p + 1) * (size_t)child_stride;
  // ... rest unchanged ...
```

Same for `k_cufftdx_corr_pair_parent_r2c`: move `gbuf`, `pbuf`, `gspec_saved`,
and `shared_mem` declarations above the boundary guard.

**Specific mechanism — and why it's likely a NO-OP:**

The conventional wisdom (and NVIDIA's own guidance) is that CUDA register
allocation operates on SSA-based live ranges computed by ptxas, not on C++
textual declaration order.  The `complex_t a[R2C::storage_size]` array:

- Has no constructor (it's a `double2` alias, a POD type).  The "declaration"
  generates zero code — it just informs the compiler about storage.
- Is only USED after the guard — its live range starts at the first
  `cufftdx_load_real_r2c<R2C>(L, cps, a)` call.
- Is not live in the guard-taken path (the guard returns before reaching
  any use of `a` or `b`).

Therefore, SSA-based liveness analysis will assign `a` and `b` the SAME live
ranges regardless of whether their C++ declaration appears before or after
the guard.  The compiler already knows `a` and `b` are dead in the guard path
and live in the FFT path — moving the declaration doesn't give the compiler
new information.

**Why it MIGHT not be a complete no-op (devil's advocate):**

1. `extern __shared__` declarations have special semantics in CUDA — they
   declare shared memory that is dynamically sized at launch.  Moving this
   declaration earlier technically makes the shared memory "in scope" for the
   guard path, but since the guard never references `shared_mem`, SSA should
   still optimize it away.

2. ptxas is not a perfect SSA optimizer.  It's possible that declaration
   order affects heuristics even if it shouldn't affect the theoretical
   optimum.  However, this would be a ptxas implementation artifact, not
   something we can reason about portably across toolkit versions.

3. If the compiler applies a simplistic "all locals in the function get
   registers reserved at function entry" strategy (very unlikely for ptxas
   at `-O2` or higher, which is the default for GPU code), then moving the
   arrays earlier would make no difference — they were already reserved for
   the whole function.

**Confidence:** VERY LOW that this helps.  The mechanism (textual declaration
order → register allocation) is not how ptxas works.  This is included for
completeness and as an argument for why NOT to try it first.

**Risk:** ZERO.  Pure code motion, no semantic change.  If the compiler truly
is SSA-based, it produces identical SASS.

**Blast radius:** Minimal — ~6 lines moved per kernel, no logic change.

---

### 1.3 Candidate C: Pre-Fill Phantom Child With Identity Polynomial, Remove Boundary Guard

This is the most structurally different approach.  Instead of handling the
boundary parent inside the kernel via a branch, we pre-fill the phantom right
child's polynomial data with `[1.0, 0.0, 0.0, …]` BEFORE any kernel that
reads it.  The kernel then processes ALL parents uniformly — no boundary
guard, no branch, no early return.

**What changes in the kernel** (pseudocode diff):

```diff
  // k_cufftdx_build_parent_r2c:
- // Remove the boundary guard entirely
- // Remove the n_real_child parameter (no longer needed)
  int p = (int)blockIdx.x * FPB + (int)threadIdx.y;
  if (p >= nparents) return;

  const double *L = child + (size_t)(2 * p) * (size_t)child_stride;
  double *out = parent + (size_t)p * (size_t)parent_stride;

- // NO boundary guard — every parent has two children (phantom is pre-filled)

  complex_t a[R2C::storage_size];
  complex_t b[R2C::storage_size];
  extern __shared__ __align__(alignof(double2)) complex_t shared_mem[];

  const double *R_ptr = child + (size_t)(2 * p + 1) * (size_t)child_stride;

  cufftdx_load_real_r2c<R2C>(L, cps, a);
  R2C().execute(a, shared_mem);
  cufftdx_load_real_r2c<R2C>(R_ptr, cps, b);  // ← reads pre-filled identity
  R2C().execute(b, shared_mem);
  // ... multiply: specL * all_ones = specL → left child passed through ...
  C2R().execute(a, shared_mem);
  cufftdx_store_real_c2r<C2R>(a, out, pps, inv_fft_n);
```

Same for `k_cufftdx_corr_pair_parent_r2c`: remove the guard, remove
`n_real_child` parameter, let the kernel read the pre-filled phantom child.

**What changes at the launch site** (in `gpu_exec.cu`):

Before launching the cuFFTDx fused kernel (or the cuFFT/VkFFT forward FFT),
when `n_real[ell-1]` is odd, pre-fill the phantom child's polynomial at
index `n_real[ell-1]` in `d_poly_levels[ell-1]` with `[1.0, 0.0, 0.0, …]`:

```
// Pseudocode for the pre-fill step, called once per Q-point per level
// before the cuFFTDx fused kernel (or before gather+cuFFT forward FFT):
if (n_real[ell-1] % 2 == 1) {
    // Phantom child is at child index n_real[ell-1]
    // Its polynomial is at offset (n_real[ell-1] * child_stride) in the child array
    // Pre-fill with [1.0, 0.0, 0.0, ..., 0.0] (length = cps)
    double *phantom = d_poly_levels[ell-1] + n_real[ell-1] * child_stride;
    cudaMemsetAsync(phantom, 0, cps * sizeof(double), stream);  // zero all
    double one = 1.0;
    cudaMemcpyAsync(phantom, &one, sizeof(double), cudaMemcpyHostToDevice, stream);  // set [0]=1.0
    // Or better: launch a tiny 1-block kernel that writes {1.0, 0.0, ...}
}
```

**Specific mechanism by which this avoids register perturbation:**

The kernel body becomes control-flow-identical to the clean-HEAD version
(before M's diff).  There is ONE code path from `p >= nparents` check to
function end.  The compiler sees:

```
check p >= nparents → return
declare arrays
do FFT
```

This is structurally identical to the pre-diff kernel that produced correct
results.  The register allocator encounters the same live ranges, the same
spill decisions, and (we expect) the same register count.  The boundary
handling moves from CONTROL flow (a branch inside the kernel) to DATA (a
pre-filled input value), which is invisible to ptxas.

**Why the identity polynomial works (mathematical justification):**

For R2C FFT of length `fft_n`, the input `[1.0, 0.0, 0.0, …, 0.0]` (one at
DC, zeros elsewhere) produces complex output `[1+0i, 1+0i, …, 1+0i]` at
all `cn = fft_n/2+1` frequency bins.  This is the FFT of a delta at position
0 — the identity element for convolution.

- **Build direction**: `specL × identity × (1/fft_n)` = `specL / fft_n`.
  C2R gives back the left child's polynomial.  ✓
- **Correlate direction**: `g_hat × conj(identity)` = `g_hat × 1` = `g_hat`.
  The left child's gradient = parent's gradient, which is correct when the
  right child is phantom (no contribution to the parent).  ✓

This holds for ALL backends (cuFFTDx, cuFFT, VkFFT, schoolbook) because:

- cuFFT/VkFFT: same R2C FFT → same identity spectrum.
- cuFFTDx fused: same R2C/C2R pipeline (unnormalized, scale applied on store).
- Schoolbook: `sum(L[j] * R[pos-j])` with `R = [1, 0, 0, …]` = `L[pos]`.
  The result is the left child's polynomial.  ✓

**Cost of the pre-fill:**

For each tree level where `n_real[ell-1]` is odd, we do ONE extra FFT on the
phantom child (to R2C-transform the pre-filled identity polynomial into the
all-ones spectrum).  This costs ~1 FFT per ~n_real[ell-1] real children.
At the leaf level (n=1,000,000, B=32, n_real[0]=31,250), this is 1/31,250 ≈
0.003% overhead — completely negligible.  At higher levels, `n_real` shrinks
but the overhead fraction stays similar (~1/n_real[ell-1]).

The pre-fill itself (cudaMemset + one-element cudaMemcpy, or a tiny kernel)
is sub-microsecond — negligible compared to the FFT work.

**Edge cases and Q-batched path:**

For the Q-batched path, there are `qb` Q-points, each with a separate set of
child polynomials.  The phantom child must be pre-filled for EACH Q-point
where `n_real[ell-1]` is odd.  This is a loop over Q-points of the same
pre-fill operation — still negligible.

For the schoolbook path: the pre-fill must happen in the child polynomial
array BEFORE the schoolbook kernel reads it.  Same pre-fill, same data.

**What about the non-cuFFTDx paths (cuFFT, VkFFT, schoolbook)?**

The SAME pre-fill applies to all paths, because they all read the child
polynomial data from `d_poly_levels[ell-1]`.  The pre-fill is backend-agnostic
— it just puts the right data in the right place before any kernel reads it.

**Removal of debug instrumentation:**

Under this candidate, `g_debug_boundary_hits`, `icm_gpu_debug_reset_boundary_hits()`,
and `icm_gpu_debug_get_boundary_hits()` are all removed (no branch to
instrument).  Their diagnostic purpose is fully served.

**Signature changes:**

Both `k_cufftdx_build_parent_r2c` and `k_cufftdx_corr_pair_parent_r2c` lose
their `n_real_child` parameter.  The dispatch functions
(`launch_cufftdx_build_r2c_t`, `launch_cufftdx_corr_r2c_t`) lose it too.
All call sites in `gpu_exec.cu` drop the argument.  This simplifies the
interface.

**Confidence:** HIGH.  The mechanism is direct: remove the branch → remove
the register allocation perturbation.  The pre-fill approach is
mathematically sound and has been verified against cuFFT's R2C convention
by both the premortem (item 1.1) and the plan doc.  The only risk is
implementation errors in the pre-fill (wrong index, wrong length, wrong
Q-point stride), which are conventional host-side bugs — not subtle
compiler-perturbation issues.

**Risk:** LOW-MEDIUM.  The pre-fill must happen at the right time (after the
child data is written, before any kernel reads it) and for the right Q-points
and levels.  Missing a pre-fill means the phantom child's data is whatever
was there before (zeros from initialization, or garbage) → wrong but
deterministic output.  The correctness check (`bench_gpu_fused verify` 36/0)
would catch this.

**Blast radius:** Medium.  Changes:
- Kernel: ~15 lines removed (guard + atomicAdd) per kernel, parameter removed
- Dispatch: `n_real_child` parameter removed from 2 dispatch functions + 2
  FPB2 paths each (4 sites)
- Launch sites (`gpu_exec.cu`): new pre-fill logic at all `run_build_level_*`
  and `run_prop_level_*` functions (~10 sites, ~3-5 lines each)
- Premortem items rendered moot: 0 (pre-existing bug claim refuted), 1.3
  (fused kernel guard), 1.5a-c (cache slot), 1.6a-c (off-by-one), 1.7 (both
  directions), 1.8 (schoolbook guard) — all become "not applicable" because
  the boundary is handled in data, not control flow.

---

### 1.4 Candidate D: Separate Kernel Launch for Boundary Parent

Instead of pre-filling data, handle the boundary parent as a SEPARATE,
minimal kernel launch.  The main cuFFTDx kernel only processes parents
`0..n_real[ell]-2` (all of which have two real children).  The boundary
parent `p = n_real[ell]-1` is handled by a trivial copy kernel.

**What changes in the kernel** (pseudocode):

The main cuFFTDx kernel reverts to its pre-M-diff shape — no boundary guard,
no `n_real_child` parameter, no atomicAdd.  At the launch site:

```c
// In run_build_level_fused():
int n_real_parents = plan->n_real[ell];
int n_real_children = plan->n_real[ell-1];
bool has_boundary = (n_real_children % 2 == 1);

int nparents_main = has_boundary ? n_real_parents - 1 : n_real_parents;

// Launch main cuFFTDx kernel for non-boundary parents
if (nparents_main > 0) {
    launch_cufftdx_build_r2c_dispatch(..., nparents_main, ...);
}

// Handle boundary parent separately
if (has_boundary) {
    // Boundary parent p = n_real_parents - 1
    // Left child at index 2*p = 2*(n_real_parents-1) = n_real_children-1 (last real child)
    // Just copy left child polynomial to parent polynomial
    k_copy_parent_identity<<<1, 256, 0, stream>>>(
        child_poly + (n_real_children-1) * child_stride, cps,
        parent_poly + (n_real_parents-1) * parent_stride, pps);
}
```

Where `k_copy_parent_identity` is a trivial kernel:
```cpp
__global__ void k_copy_parent_identity(const double *src, int src_len,
                                        double *dst, int dst_len) {
    for (int i = threadIdx.x; i < dst_len; i += blockDim.x)
        dst[i] = (i < src_len) ? src[i] : 0.0;
}
```

Similarly for the correlate direction: copy `g_parent` to `g_child[2*p]`,
zero `g_child[2*p+1]`.

**Specific mechanism:**

The hot cuFFTDx kernel has ZERO branches beyond the standard `p >= nparents`
guard.  Its register allocation is identical to the clean-HEAD version
(assuming `nparents` is passed as `n_real_parents` or `n_real_parents-1`,
both of which are runtime values that don't affect compile-time register
allocation).  The boundary case is handled by a completely separate kernel
with its own (trivial) register allocation.

**Why this is cleaner than the in-kernel guard:**

1. The hot kernel's control flow is identical to pre-diff — one code path.
2. The boundary kernel is so simple it can't plausibly perturb anything.
3. Zero wasted FFT work on the phantom child (unlike Candidate C, which still
   does one FFT per odd-level).

**Why Candidate C might still be preferable:**

1. Candidate D requires a new kernel (`k_copy_parent_identity`) and a new
   launch at every odd-level — more host-side complexity.
2. The extra kernel launch has overhead (~5-10 µs per launch on B200).  For
   ~8 odd-levels in a 16-level tree, this is ~40-80 µs extra per call to
   `icm_gpu_fused_pair` — negligible for large n but measurable for small n.
3. Candidate C has zero extra kernel launches — the pre-fill can be folded
   into existing operations (e.g., done right after `cudaMemset` that already
   zeros the child array).
4. Candidate C works uniformly across all backends (cuFFTDx, cuFFT, VkFFT,
   schoolbook) because it changes DATA, not launch logic.  Candidate D would
   need a separate copy kernel for each backend's launch site — more
   duplication.

**Confidence:** HIGH that the mechanism works (same as Candidate C — remove
the branch, remove the perturbation).  MEDIUM that the implementation is
worth the complexity vs Candidate C.

**Risk:** LOW-MEDIUM.  Same correctness concerns as Candidate C (is the
boundary index right? is the copy length right?), plus the risk of getting
the launch count wrong (`nparents_main = n_real_parents - 1` when odd, else
`n_real_parents`).

**Blast radius:** Medium.  New kernel + changes at every launch site.

---

## 2. Ranking and Recommendation

| Rank | Candidate | Confidence | Risk | Blast Radius | Try First? |
|------|-----------|-----------|------|-------------|------------|
| **1** | **A: Remove atomicAdd only** | MEDIUM | ZERO | Minimal | **YES** |
| 2 | C: Pre-fill identity, remove guard | HIGH | LOW-MED | Medium | If A fails |
| 3 | D: Separate boundary kernel | HIGH | LOW-MED | Medium | If C is messy |
| 4 | B: Move declarations before guard | VERY LOW | ZERO | Minimal | Don't bother |

### Why Candidate A first:

1. **It costs nothing to try.**  The `g_debug_boundary_hits` counter has
   already served its purpose (confirmed 0 hits on the failing case).  It
   doesn't need to stay in the shipped kernel.  Removing it is just good
   hygiene regardless of whether it fixes the register pressure.

2. **It directly targets the most suspicious element.**  The SASS diff
   confirmed that every cuFFTDx kernel instantiation gained an `ATOMG.E.ADD.64`
   instruction that wasn't there before.  The atomicAdd's address register is
   the most "alien" demand in the guard path — everything else in the copy
   loop uses registers already needed by the FFT path.

3. **It can be verified with zero GPU time.**  Build with the atomicAdd
   removed, run `cuobjdump -res-usage`, and check if register counts return
   to clean-HEAD values (116 and 126).  If they do — done, move to correctness
   verification.  If they don't — we learned something (the branch structure
   itself, not just the atomic, is the perturbation), and Candidate C is the
   next step.

4. **Minimal blast radius.**  If Candidate A doesn't fix the register count,
   we revert the 3-line removal and move on.  No complex pre-fill logic to
   unwind.

### Why Candidate C second:

If removing the atomicAdd doesn't restore register counts (or restores them
only partially), the branch structure itself is the problem.  Candidate C
eliminates the branch entirely by moving the boundary handling from control
flow to data.  It's a more invasive change (pre-fill logic at ~10 launch
sites) but has high confidence because the kernel body becomes structurally
identical to the known-good clean-HEAD version.

### Why Candidate D is third, not second:

Candidate C and D both eliminate the branch.  Candidate C does it with a
data pre-fill (no new kernel, no launch-count changes); Candidate D does it
with a separate kernel launch.  Candidate C is simpler to implement correctly
because it changes fewer call sites and doesn't require a new kernel.  The
one-extra-FFT-per-odd-level cost of Candidate C is truly negligible (0.003%
at the leaf level, less above).  Candidate D is the fallback if the pre-fill
approach turns out to have unexpected interactions with buffer sizing or
cache invalidation.

### Why Candidate B isn't worth trying:

As reasoned in §1.2, ptxas register allocation is SSA-based and insensitive
to C++ declaration order for POD types.  Moving the declarations is unlikely
to change the SASS output at all.  It's a distraction — skip it.

---

## 3. Validation Plan

### Phase 0: Register Count Check (zero GPU time, build-only)

For whichever candidate we try first (Candidate A, or A+C combined):

1. Build with the fix applied:
   ```
   make bench_gpu_fused CUDA_ARCH=sm_100
   ```

2. Check register usage:
   ```
   cuobjdump -res-usage build/gpu_gpu_kernels_fused.o | grep -A1 "k_cufftdx_build_parent_r2c<1024,1>"
   cuobjdump -res-usage build/gpu_gpu_kernels_fused.o | grep -A1 "k_cufftdx_corr_pair_parent_r2c<1024,1>"
   ```

3. **Pass criterion**: Register counts match clean-HEAD values:
   - `k_cufftdx_build_parent_r2c<1024,1>`: **116** registers (currently 124)
   - `k_cufftdx_corr_pair_parent_r2c<1024,1>`: **126** registers (currently 124)

4. **If pass**: Proceed to Phase 1.  **If fail** (counts still shifted):
   - If trying Candidate A alone: A is insufficient; proceed to Candidate C
     (or A+C combined — remove atomicAdd AND add pre-fill), rebuild, re-check.
   - If A+C still shows shifted counts: something else is going on (e.g., the
     parameter list change — `n_real_child` is an extra parameter even if
     unused).  Consider removing `n_real_child` from the kernel signature
     entirely (it's not needed for C or D).  Rebuild, re-check.
   - If removing `n_real_child` AND the atomicAdd AND the branch STILL doesn't
     restore counts: escalate — this would mean the register shift is caused
     by something else entirely (unlikely but possible; the SASS comparison
     would need to be re-done to find what else changed).

### Phase 1: Small-Scale Correctness (B200 rental, ~5-10 min)

Only proceed if Phase 0 passes (register counts restored to clean-HEAD values).

1. Build for B200:
   ```
   make bench_gpu_fused CUDA_ARCH=sm_100
   ```

2. Run the standard verify suite:
   ```
   ./bench_gpu_fused verify
   ```
   **Pass criterion**: 36/0 (same as clean HEAD).

3. Run small-scale ragged-tree correctness checks (GPU vs CPU reference):
   - Same cases that K node used to refute the pre-existing-bug alarm:
     - n=480, B=32 (nblocks=15, odd at multiple levels)
     - n=480, B=30 (nblocks=16, power-of-two control)
     - n=1000, B=16 (nblocks=63)
     - n=1000, B=8 (nblocks=125)
   - Plus the boundary-clamp case from K1's verification: k=524,289
   - All with `ICM_GPU_DEBUG_PLAN=1` to confirm FUSED/cuFFTDx tier is exercised
   - **Pass criterion**: All PASS at ~1e-15 relative error (same as clean HEAD)

4. **If any failure in Phase 1**: STOP.  Do not proceed to Phase 2.  Capture
   full debug output.  The register-count restoration was necessary but not
   sufficient — something else in M's diff is interacting.

### Phase 2: Large-Scale Timing Only (B200 rental, same session, ~5 min)

Only proceed if Phase 1 passes cleanly.

1. Measure the original failing pair:
   ```
   ICM_GPU_DEBUG_PLAN=1 ./bench_gpu n=1000000 k=256
   ICM_GPU_DEBUG_PLAN=1 ./bench_gpu n=1048576 k=256
   ```
   (Fixed k=256 to control for k-pad confound, per the plan doc's §5.)

2. **Pass criterion**: The gap between the two timing measurements is ~1-2%
   (the raw padding-waste fraction), NOT the pre-fix 23.5%.  The ragged n
   should be very slightly slower (more levels with odd n_real → one extra
   FFT per odd-level for Candidate C, or one extra kernel launch for
   Candidate D), but nowhere near the original cliff.

3. Measure k=n curve at the original discovery scale (n=1,000,000 and
   n=1,048,576) — GPU-only timing, **no CPU reference** (per this session's
   documented near-miss: CPU calibration tables top out well below 1,000,000,
   and a CPU reference at this scale would stall for 45+ minutes).

4. **Pass criterion**: k=n curve is monotonic across this boundary (n=1,000,000
   ≤ n=1,048,576 in wall-clock time).  The below_sat fix (K1, already
   hardware-verified) should have already resolved this, but verify it still
   holds with the register-pressure fix in place.

### What NOT to do:

- Do NOT run a full `gpu_dispatch_validate` sweep (that's node Q's job,
  after L and K both land).
- Do NOT run the threshold search (node H) — blocked on Q.
- Do NOT compare against CPU reference at n > 16,384 (calibration ceiling).
- Do NOT rent a B200 until Phase 0 passes locally (or on the cheapest
  possible instance just for the build — the `cuobjdump -res-usage` check
  needs a GPU toolkit but not a GPU).

---

## 4. Implementation Notes (For Supervisor, Not for This Node)

This section is guidance for whoever implements the chosen candidate — NOT
part of this node's deliverable, but included per the task's request for
actionable next steps.

### If Candidate A (remove atomicAdd) passes Phase 0:

The diff is ~20 lines:
- Remove `g_debug_boundary_hits` declaration + 2 accessor functions (~15 lines)
- Remove the `if (threadIdx.x == 0) atomicAdd(...)` line from both kernels (2 lines)
- The boundary guard and `n_real_child` parameter stay

### If Candidate A fails, proceed to Candidate C (pre-fill):

The diff is larger but still localized:
1. **`gpu_kernels.cu`**: Remove boundary guard + atomicAdd + debug counter +
   `n_real_child` parameter from both kernels.  Remove `n_real_child` from
   both dispatch functions' signatures and all internal calls.
2. **`gpu_exec.cu`**: Add pre-fill logic before each kernel launch site that
   reads child polynomials.  The pre-fill must happen:
   - After the child data for this level is fully written (after build phase
     for build direction; after the forward FFT cache is populated for
     correlate direction — but the correlate reads `d_poly_levels[ell-1]`,
     so the pre-fill should happen before the build's forward FFT writes the
     cache, i.e., right after the child polynomials are computed).
   - For each Q-point (loop over `qb`).
   - Only when `n_real[ell-1]` is odd (guard the pre-fill with `if
     (plan->n_real[ell-1] & 1)`).
3. **Signature cleanup**: Remove `n_real_child` from all kernel signatures,
   dispatch functions, and call sites (this parameter is no longer needed
   by any kernel — the pre-fill makes every child "real" from the kernel's
   perspective).

### If Candidate C turns out to be messy (e.g., pre-fill timing interacts
badly with cache invalidation in the correlate path):

Fall back to Candidate D — separate kernel launch for boundary parent.
Simpler data flow (no pre-fill), but more launch sites to change.

---

## 5. Summary

The root cause is established: a semantically-dead branch placed before
cuFFTDx's register arrays perturbs ptxas register allocation, shifting
register counts by +8/−2 at the exact template instantiation used by the
failing test case.  This is a known class of CUDA compiler behavior
(corroborated by external research), even though this exact failure
signature is not publicly documented.

Four candidates are designed and ranked:

1. **Candidate A** (remove `g_debug_boundary_hits`): Try first — trivial,
   zero-risk, directly targets the `ATOMG.E.ADD.64` instruction that appeared
   in every buggy-build SASS dump.
2. **Candidate C** (pre-fill phantom child with identity polynomial): Try
   second if A doesn't restore register counts.  Eliminates the branch
   entirely by moving boundary handling from control flow to data.
3. **Candidate D** (separate boundary kernel): Fallback if C is too messy.
4. **Candidate B** (move declarations): Skip — likely a no-op given SSA-based
   register allocation.

Validation starts with a zero-GPU-time `cuobjdump -res-usage` check before
any B200 rental.  Only if register counts return to clean-HEAD values do we
proceed to hardware correctness + timing verification.
