# GPU `below_sat` Fix Design Document

**Status:** Design-phase investigation — no code changes yet  
**Scope:** `src/gpu/gpu_plan.cu` only (CPU `src/icm.c` explicitly out of scope)  
**Date:** 2025-07-17  
**Trigger:** Reproduced wall-clock inversion on B200: n=1,000,000 took 632ms,
n=1,048,576 took 511.9ms, despite n=1,048,576 being strictly harder.
Root cause traced to `below_sat[14]` firing for the power-of-two k but
not for the other, because of an exact-equality check at line 529 of
`src/gpu/gpu_plan.cu`.

---

## 1. The actual meaning of `below_sat`

### 1.1 Where it is set

**GPU** (`src/gpu/gpu_plan.cu`, `build_tree_geometry()`, line 529):
```c
if (psz[ell] == 2 * cps && cps >= 2) below_sat[ell] = 1;
```

**CPU** (`src/icm.c`, `tree_ctx_create_ex2()`, line 1149–1151):
```c
/* Below saturation: psz doubles each level, so psz[ell] = 2*psz[ell-1].
 * This means the actual degree of children is cps/2, not cps-1. */
if (tc->psz[ell] == 2 * cps && cps >= 2)
    tc->below_sat[ell] = 1;
```

Both are textually identical. The CPU comment gives the key insight:
**the actual (non-zero) degree of the child polynomials at level ell-1
is cps/2, not the full cps-1, when the doubling is still in progress.**

### 1.2 Why `cps/2`?

The tree polynomial sizes follow the recurrence
`psz[ell] = min(leaf_degree * 2^(ell+1), k_pad)`. At levels below the
k_pad cap, `psz[ell] = 2 * psz[ell-1]` — the doubling is still active.

Consider level ell-1 whose children (at level ell-2) have size cps/2.  
The polynomial multiplication that built level ell-1 from level ell-2
convolved two polys of effective degree cps/2, producing a product of
degree `cps/2 + cps/2 = cps`. This product is stored in cps slots
(=`psz[ell-1]`). Only the first `cps/2 + 1` coefficients are non-zero;
the upper half (indices cps/2+1 through cps-1) is zero.

Thus the child polynomials at level ell-1, though stored in arrays of
length cps, carry only cps/2+1 non-zero coefficients. This is the
"below saturation" property.

### 1.3 What `below_sat` controls (CPU reference: `tree_propagate_g()`, lines 1504–1512)

When `is_below = 1` (i.e. `below_sat[ell] == 1`), the CPU computes:

```c
int p_eff   = is_below ? cps/2 + 1 : cps;           // line 1505
int g_eff_needed = out_needed + p_eff - 1;           // line 1507
int g_eff_max    = is_below ? (cgsz + cps/2) : pgsz; // line 1508
int g_eff = min(g_eff_needed, g_eff_max);             // line 1509
```

These values are used in the correlate (top-down g-propagation):

- **`p_eff`**: The number of non-zero polynomial coefficients to
  correlate against. Halved from cps to cps/2+1 because the upper
  half of the child polynomial is zero — those terms would contribute
  zero to the dot product `sum_j P[j] * g[m+j]`. Discarding them is
  lossless; they were always zero.

- **`g_eff`**: The number of parent g-vector coefficients actually
  needed for the correlate. Since P has only p_eff non-zero terms,
  the correlate at child output position m needs g[m] through
  g[m + p_eff - 1]. The maximum index accessed is
  `out_needed + p_eff - 2`, so g_eff = out_needed + p_eff - 1
  (clamped to g_eff_max). When `is_below`, the parent g is shorter
  because fewer parent coefficients are needed to produce the full
  child output.

- **`g_eff_max`**: When `is_below`, the parent g array itself only
  needs `cgsz + cps/2` coefficients (the child needs cgsz output
  coefficients, each computed from at most cps/2+1 parent g entries;
  the maximum parent index touched is `cgsz-1 + cps/2`). The rest of
  the parent array is never read.

In the **build** (bottom-up polymul) direction (`tree_build_levels()`,
line 1190; also `tree_ctx_create_ex2()` line 1190):

```c
int build_conv_len = is_below ? (2 * (cps / 2)) : (2 * cps - 1);
```

When `is_below`, the convolution length drops from `2*cps - 1` to
`cps` (= `2 * (cps/2)`). This is safe because both child operands
have only cps/2 non-zero coefficients each, so their product has at
most `cps/2 + cps/2 = cps` non-zero terms. The upper `cps - 1`
coefficients of the full `2*cps - 1` convolution would all be zero.

**GPU** (`src/gpu/gpu_plan.cu`, `build_plan_metadata()`, lines 888–896
and `estimate_candidate_cost()`, lines 626–634) computes the identical
formulas:

```c
int p_eff      = is_below ? (cps / 2 + 1) : cps;          // line 889
int g_eff_max  = is_below ? (cps + cps / 2) : pgsz;       // line 892
int conv_build = is_below ? (2 * (cps / 2) + 1) : (2 * cps - 1);  // line 894
```

These are stored in `GpuLevelPlan` fields (`p_eff`, `g_eff`, `build_conv`)
and consumed by the execution kernels at runtime (`src/gpu/gpu_exec.cu`).

### 1.4 Why the halving is safe — the mathematical guarantee

The child polynomial P at level ell-1 has effective degree cps/2 **iff**
it was formed by multiplying two grandchildren of effective degree
cps/4. By induction from the leaves (where the leaf degree is B, and
psz[0] = 2B, so effective degree = B = psz[0]/2), every level before
the k_pad cap has effective degree exactly half its allocated size.
This holds regardless of the parent's size — it's a property of the
**child's own construction**, not the parent's.

The correlate `correlate_school(g, len_g, P, len_P, out, len_out)`
(`src/icm.c` line 481) computes `out[m] = sum_{j=0}^{len_P-1} P[j] * g[m+j]`
for m = 0..len_out-1. If P[j] = 0 for j > cps/2, then setting
len_P = cps/2+1 (instead of cps) computes the identical result with
fewer operations. No information is lost — the discarded coefficients
were identically zero.

---

## 2. Is the exact equality mathematically required, or overly strict?

### 2.1 What the condition `psz[ell] == 2*cps` actually checks

It checks whether the **parent** at level ell doubles exactly from the
child. This is a **proxy** for "the children at ell-1 are below
saturation" — but an overly strict proxy. The children's effective
degree depends on **their own** construction (from level ell-2), not
on the parent's size.

### 2.2 The n=1,000,000 counterexample

| Level | n=1,048,576 (k=2^20) | n=1,000,000 (k=10^6) |
|-------|----------------------|----------------------|
| ell=13 (cps) | 524,288 | 524,288 |
| ell=14 (psz) | 1,048,576 = 2*cps | 1,000,000 < 2*cps |
| psz[14]==2*cps? | YES → below_sat fires | NO → missed |

At ell=14 for n=1,000,000:
- Children at ell=13 have size cps=524,288.
- psz[12] = 262,144, and psz[13] = 524,288 = 2*262,144.  
  So the children **were** built in the doubling regime, and their
  effective degree is cps/2 = 262,144.
- The parent at ell=14 has psz[14] = 1,000,000 (capped by k_pad).
  This is less than 2*cps = 1,048,576, so the exact-equality check
  fails — but the **children still have effective degree cps/2**.

The optimization should fire. It doesn't.

### 2.3 When below_sat should *genuinely* not fire

The children lose the cps/2 property only when **they themselves** were
built at the cap. That happens when `psz[ell-1] != 2*psz[ell-2]`, i.e.
the child level had already saturated. At that point the children have
effective degree cps-1 (full), not cps/2, and the optimization would
incorrectly discard non-zero coefficients.

Example: n=1,000,000 at ell=15 (if the tree were deep enough):
- cps = psz[14] = 1,000,000, psz[13] = 524,288.
- psz[14] = 1,000,000 ≠ 2*524,288 = 1,048,576.
- Children at ell=14 were NOT built in doubling regime → effective
  degree is 999,999 (full). below_sat should NOT fire.

### 2.4 The correct condition

The children at ell-1 have effective degree cps/2 iff they were built
in the doubling regime, i.e. **`psz[ell-1] == 2 * psz[ell-2]`** (for
ell ≥ 2). For ell = 1, the condition is **`psz[0] == 2 * leaf_degree`**
(which is true whenever `psz[0] < k_pad`, since psz[0] = min(2*B, k_pad)).

Equivalently: the child level ell-1 is below saturation iff
`psz[ell-2] < k_pad` (the grandchild hasn't hit the cap yet). Both
formulations are equivalent because psz doubles at every level until
hitting k_pad.

### 2.5 Is the generalization safe for *all* derived values?

When below_sat fires under the generalized condition, we set:
- `p_eff = cps/2 + 1` (safe — justified above)
- `build_conv_len = cps` (safe — product of two cps/2-effective-degree
  polys has degree cps, fits in parent's psz[ell] slots as long as
  psz[ell] ≥ cps+1; proven in §2.6)
- `g_eff_max = cps + cps/2` (**needs clamping** — see §2.7)

### 2.6 Build output fits in parent

With the generalized condition, psz[ell] could be as low as cps+1
(when k_pad barely exceeds cps). The build convolution produces
cps+1 coefficients (degrees 0 through cps). These need cps+1 slots.
Since psz[ell] ≥ cps+1 (if psz[ell] = cps, then k_pad = cps, which
would mean saturation already happened at ell-1, contradicting the
generalized trigger), the output always fits. No overflow.

### 2.7 **Critical safety issue: g_eff_max must be clamped**

This is the one place where the generalization is **not** a drop-in
replacement. When `is_below = 1`, the original code computes:

```c
g_eff_max = cps + cps/2;   // line 892, gpu_plan.cu
```

In the original (strict-equality) case, `psz[ell] = 2*cps ≥ cps + cps/2`,
so `g_eff_max ≤ pgsz` always. With the generalization, psz[ell] could
be as low as cps+1, and `cps + cps/2` could exceed psz[ell].

If `g_eff = min(g_eff_needed, g_eff_max)` produces a value > psz[ell],
the execution kernels (`k_schoolbook_corr_pair`, `k_schoolbook_corr_pair_smem_parent`,
etc., in `src/gpu/gpu_kernels.cu`) will read `g_parent[m + j]` for
indices up to `len_g - 1`, which would read beyond the allocated g array
— an out-of-bounds memory access, undefined behavior, and potentially
silently wrong equity values.

**The fix must include clamping `g_eff_max` to `pgsz`**:
```c
g_eff_max = is_below ? min(cps + cps/2, pgsz) : pgsz;
```

This is safe because clamping g_eff to pgsz means we use all available
g coefficients; the correlate simply can't access beyond what was
computed. The result remains correct — we just can't shorten the g
vector as much as we'd like at this boundary level. The performance
benefit of the optimization is slightly reduced at this one level, but
correctness is preserved.

---

## 3. Safe generalization: exact proposed changes

### 3.1 File: `src/gpu/gpu_plan.cu`

#### Change 1: `build_tree_geometry()` (line 527–529)

**Replace:**
```c
    for (int ell = 1; ell < L; ++ell) {
        int cps = psz[ell - 1];
        if (psz[ell] == 2 * cps && cps >= 2) below_sat[ell] = 1;
    }
```

**With:**
```c
    for (int ell = 1; ell < L; ++ell) {
        int cps = psz[ell - 1];
        /* below_sat fires when children at ell-1 are genuinely below
         * saturation: their effective degree is cps/2, not cps-1.
         * This is true iff the child level itself doubled from its
         * own child (i.e. the doubling was still in effect when the
         * children were built).  For ell==1, the leaf level is below
         * saturation iff psz[0] == 2*leaf_degree (i.e. the leaf cap
         * hasn't been hit yet, which is psz[0] < k_pad in practice). */
        int child_doubled;
        if (ell >= 2) {
            child_doubled = (psz[ell-1] == 2 * psz[ell-2]);
        } else {
            child_doubled = (psz[0] == 2 * leaf_degree);
        }
        if (child_doubled && cps >= 2) below_sat[ell] = 1;
    }
```

#### Change 2: `estimate_candidate_cost()` (line 631)

**Replace:**
```c
        int g_eff_max = is_below ? (cps + cps / 2) : pgsz;
```

**With:**
```c
        int g_eff_max = is_below ? std::min(cps + cps / 2, pgsz) : pgsz;
```

(Requires `<algorithm>` already included via `gpu_internal.h`.)

#### Change 3: `build_plan_metadata()` (line 892)

**Replace:**
```c
        int g_eff_max = is_below ? (cps + cps / 2) : pgsz;
```

**With:**
```c
        int g_eff_max = is_below ? std::min(cps + cps / 2, pgsz) : pgsz;
```

### 3.2 CPU latent bug — out of scope, flagged for future

`src/icm.c` line 1151 has the identical `psz[ell] == 2 * cps` check.
It suffers from the same overly-strict condition but is explicitly OUT
OF SCOPE for this task. It should be addressed in a separate pass after
GPU verification. The CPU's g_eff_max at line 1508 also lacks the
`min(..., pgsz)` clamp, though on CPU the OOB risk is mitigated by the
g_buf allocation being max_g-sized (line 1488–1489) — still worth
auditing separately.

---

## 4. Alternative: k_pad bias sidestep (if the generalization were unsafe)

This section is included for completeness. The generalization **is**
safe (with the g_eff_max clamp), so this is the fallback only if future
analysis finds an issue we missed.

### 4.1 Mechanism

`best_k_pad_gpu()` (`src/gpu/gpu_plan.cu` lines 465–484) currently selects
a 7-smooth k_pad ≥ k to minimize saturated-level FFT cost. It does not
consider the below_sat optimization at all. One could add a bias term
that favors k_pad values that land on a `psz[ell] == 2*cps` boundary
at the tree level where the convolution is largest (which dominates
total cost).

### 4.2 Practical trade-off

- **Pro**: Zero change to correctness-critical below_sat logic.
  Purely a cost-model preference in k_pad selection.

- **Con**: k_pad would be rounded up to the next value that makes
  `k_pad = leaf_degree * 2^(ell+1)` for some ell, which could mean
  a k_pad up to ~2× larger than k (in the worst case where k is just
  above a power of two). This inflates the polynomial sizes at all
  saturated levels, adding FMA work that may outweigh the benefit of
  firing below_sat at one extra level.

- **Con**: Does not fix the root cause. The optimization would still
  be missed for any non-power-of-two k whose best k_pad (per the FFT
  cost model) doesn't happen to align.

- **Verdict**: Not recommended as the primary fix. The generalization
  (§3) is mathematically sound and strictly dominates this approach.

---

## 5. What to measure/verify on the next B200 rental

### 5.1 Minimum correctness gate

```
# Must pass with zero regression vs baseline
./build/bench_gpu_fused 36 0
```

This runs the full equity computation at the calibrated 36/0 benchmark
point. Any change in output equity values (beyond floating-point roundoff,
which should be zero since we're only changing which coefficients are
computed, not how) is a HARD BLOCK.

### 5.2 Specific correctness spot-checks

These are critical because the fix touches truncation logic — a bug
here produces silently wrong numbers, not crashes:

| Test | n | k | Why |
|------|---|---|-----|
| A | 1,000,000 | 1,000,000 | The exact case that triggered this investigation. Must produce identical equity values before/after (timing should improve). |
| B | 1,048,576 | 1,048,576 | Power-of-two baseline. Must be numerically identical before/after (below_sat already fired; fix should be a no-op here). |
| C | 500,000 | 500,000 | k=500,000 is 7-smooth (2^4×5^6? needs checking) but not a power of two. Tests a different "miss" pattern. |
| D | 524,289 | 524,289 | k = 2^19 + 1. Tests the boundary case where psz[ell] barely exceeds cps (g_eff_max clamp is exercised). |
| E | 750,000 | 750,000 | Arbitrary non-power-of-two. Should see timing improvement vs baseline at levels that previously missed below_sat. |

For **each** of A–E:
1. Run `bench_gpu_fused` at that (n,k) with `ICM_GPU_DEBUG_PLAN=1` to
   dump per-level plan decisions.
2. Verify that `below_sat` fires at exactly the levels where
   `psz[ell-1] == 2*psz[ell-2]` (and not elsewhere).
3. Compare full-precision equity output values (all player equities)
   between the fixed build and the baseline build (compile both, run
   both, diff the output). Use a tolerance of 0.0 — any non-zero
   difference is a regression.
4. Record wall-clock timing for each (n,k) to quantify the speedup.

### 5.3 Regression sweep

Run the full monotonicity sweep from the earlier session (varying k
for fixed n, and varying n for fixed k) to confirm the inversion at
the n=1,000,000 / n=1,048,576 pair is resolved and no new inversions
appear.

### 5.4 Memory safety

Run under `cuda-memcheck` (or `compute-sanitizer`) at test point D
(k = 524,289) to confirm no out-of-bounds accesses from the g_eff
computation.

---

## Summary

The `below_sat` optimization correctly halves the effective polynomial
size when children are below the k_pad saturation cap. The trigger
condition `psz[ell] == 2*cps` is overly strict: it checks the parent's
doubling status when it should check the child's. The fix changes the
condition to `psz[ell-1] == 2*psz[ell-2]` (child doubled from grandchild),
which fires for all cases where the optimization is mathematically valid.
One additional safety clamp (`g_eff_max = min(cps + cps/2, pgsz)`) is
required to prevent out-of-bounds memory access at boundary k_pad values.
The CPU code (`src/icm.c`) has the identical latent bug and should be
fixed separately after GPU verification.
