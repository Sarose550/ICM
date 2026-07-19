# Verify Regression Root-Cause Analysis

**Date:** 2026-07-18  
**Bug:** `./bench_gpu_fused verify` fails for all k=n (full top-k) cases at n ≥ 4096 with B > 0  
**Candidate fix:** `patches/verify_fix.patch`  
**Confidence:** High (~85%)

---

## Summary

The verify failure is caused by a **pre-existing off-by-one bug in the hybrid
engine's "below-saturation" convolution-length computation**, not by either of
the E1_GPU_PREP patches (oom_fix, cufft_524k_fix).  The patches are innocent —
the bug was always present in committed `src/gpu/gpu_plan.cu` and was never
caught because prior audits (E1_GPU_PREP, W0R_REVIEW) were local/CPU-only;
this E2 B200 session was the first time the verify suite ever ran on real GPU
hardware.

---

## Failure Pattern (from `results/e2_verify_failure_20260718.log`)

| n      | k     | B   | result | error       |
|--------|-------|-----|--------|-------------|
| 256    | 256   | 0   | PASS   | ~1e-15      |
| 1024   | 1024  | 0   | PASS   | ~2e-15      |
| 4096   | 100   | 64  | PASS   | ~2e-15      |
| 4096   | 4096  | 64  | **FAIL** | **1.9e+01** |
| 16384  | 100   | 64  | PASS   | ~5e-15      |
| 16384  | 16384 | 128 | **FAIL** | **2.1e+01** |
| 65536  | 100   | 64  | PASS   | ~2e-14      |
| 65536  | 65536 | 128 | **FAIL** | **1.8e+01** |

- Every k=100 (partial top-k) case passes with normal FP64 error.
- Every k=n at n=256,1024 passes (these use B=0 → single-kernel engine,
  not hybrid).
- **Every k=n at n ≥ 4096 fails catastrophically** (errors of 1–20, not FP
  noise).  These all show B=64 or B=128 — the hybrid block engine.
- k=100 cases at the SAME n, SAME B (64) pass → the hybrid engine itself is
  not broken; the bug is specific to k=n (full top-k).

---

## Root Cause: Off-by-One in `conv_build` for Below-Saturation Levels

### What is "below saturation"?

In the hybrid tree, a level `ell` is marked `below_sat = 1` when
`psz[ell] == 2 * psz[ell-1]` and `cps = psz[ell-1] >= 2`.  This means the
children's polynomials are only "half-full" (their actual degree is ≤ cps/2
rather than the full cps−1), so the convolution length is shorter than the
worst-case `2*cps − 1`.  The code computes an effective convolution length:

```c
int conv_build = is_below ? (2 * (cps / 2)) : (2 * cps - 1);
```

### The off-by-one

For a below-saturation level, each child polynomial has `p_eff = cps/2 + 1`
non-zero coefficients (degree = cps/2).  The convolution of two such
polynomials yields **`2*(cps/2) + 1 = cps + 1`** non-zero coefficients
(degree = cps), **not** `2*(cps/2) = cps`.

- **Current code:** `conv_build = cps`  (off by −1)
- **Correct:**      `conv_build = cps + 1`

The off-by-one causes `best_fft_config_gpu` to choose an FFT size that is
1 too small for the actual convolution, and `wrap_m` to be 1 too small (often
0 when it should be 1).  The highest-degree coefficient of the convolution
is silently dropped (neither computed by the FFT nor recovered by wrap
correction).  This error propagates and compounds up the tree.

### Why k=100 is unaffected

When k=100 (k_pad ≈ 100), the polynomial sizes `psz[ell]` saturate at k_pad
early.  At a level where `psz[ell] == psz[ell-1]` (both capped at k_pad),
`psz[ell] ≠ 2*psz[ell-1]`, so `below_sat = 0` and the buggy code path is
never entered.  The full `2*cps − 1` formula is used, which is correct.

When k=n (k_pad = n, a power of 2), `below_sat = 1` at most intermediate
levels because psz doubles cleanly.  The buggy path is entered at every such
level, and the error accumulates.

### Why n=256,1024 are unaffected

For n ≤ 1024, `single_kernel_max_n(k)` returns 1024, and `icm_gpu_equity`
takes the single-kernel (pure-linear) path, which does not use the hybrid
engine or below_sat at all.

---

## Evidence

1. **Geometric derivation:** Two polynomials with `p_eff = cps/2+1` non-zero
   coefficients convolve to `(cps/2+1)+(cps/2+1)−1 = cps+1` coefficients.
   `2*(cps/2) = cps` is off by 1.

2. **k=100 passes with same B:** At n=4096, both k=100 and k=4096 use B=64
   and the batched hybrid path (`qb=256`).  The only difference is k_pad
   (100 vs 4096), which determines whether `below_sat` activates.

3. **Error magnitude:** Errors are O(1)–O(20), not O(1e-15).  This is
   consistent with dropping one coefficient per level for ~5–7 levels
   (the error compounds through polynomial multiplication).

4. **Patches are innocent:** Neither `oom_fix.patch` nor
   `cufft_524k_fix.patch` modifies `conv_build` or any below_sat logic.
   The OOM fix changes `per_q_bytes` (affects q_batch selection) and adds
   retry-on-OOM; the cuFFT fix adds a 64-bit API path for large batches.
   For n=4096 on B200, q_batch stays at 256 with or without the OOM fix,
   and the cuFFT 64-bit path is not triggered (batch×n is well under 2^31).

---

## Fix: `patches/verify_fix.patch`

Changes both occurrences of the `conv_build` formula in `src/gpu/gpu_plan.cu`
(lines 598 and 808):

```diff
-        int conv_build = is_below ? (2 * (cps / 2)) : (2 * cps - 1);
+        int conv_build = is_below ? (2 * (cps / 2) + 1) : (2 * cps - 1);
```

This corrects `conv_build` from `cps` to `cps+1` in the below-saturation case,
which in turn causes `best_fft_config_gpu` to either:

- Choose a 1-larger FFT size (no wrap, no lost coefficient), or
- Increase `wrap_m` by 1 (the wrap correction kernel `k_wrap_build` then
  recovers the previously-dropped highest coefficient).

Both outcomes are correct; the cost model selects whichever is cheaper.

---

## What Would Confirm/Refute on Next GPU Access

1. **Apply the fix** (`git apply patches/verify_fix.patch`), rebuild,
   run `./bench_gpu_fused verify`.  All k=n cases should pass.

2. **If they still fail:** the off-by-one is real but the wrap correction
   kernel `k_wrap_build` may have a secondary bug when `wrap_m > 0` at
   below_sat levels (the kernel was never exercised there before because
   `wrap_m` was always 0).  Inspect `k_wrap_build` in `gpu_kernels.cu` for
   indexing errors in the `pos = fft_n + i` path when the parent stride
   is `2*cps`.

3. **Refutation path:** If the fix doesn't resolve the failure, the bug may
   be in the below-saturation g-value propagation (`p_eff` at below_sat
   levels in `run_prop_level_fft`) rather than the build phase.  The
   correlation `conv_corr = g_eff + p_eff - 1` appears correct, but the
   effective `len_P` and `len_g` passed to the FFT kernels should be
   double-checked for below_sat levels.

4. **Quick sanity check:** Add `ICM_GPU_DEBUG_PLAN=1` to the verify run and
   compare the `conv_build` and `fft_n` values logged with and without the
   fix.  At any below-saturation level, `conv_build` should increase by 1
   and either `fft_n` or `wrap_m` should adjust accordingly.

---

## Appendix: Derivation of the Correct `conv_build`

At level `ell` with `below_sat[ell] = 1`:

- `cps = psz[ell-1]` (child polynomial stride)
- Child has `cps/2 + 1` non-zero entries (degree `cps/2`)
- Convolution of two children: L has entries 0..cps/2, R has entries 0..cps/2
- Result has entries 0..cps (degree cps, cps+1 coefficients)
- Therefore `conv_build = cps + 1 = 2*(cps/2) + 1`

The non-below-sat case: child has entries 0..cps-1, convolution has entries
0..2*cps-2, so `conv_build = 2*cps - 1`.  This is already correct.
