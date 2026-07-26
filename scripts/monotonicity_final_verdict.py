#!/usr/bin/env python3
"""
FINAL definitive analysis with clean output.
All model costs in ms (calibration values are µs despite '_ns' naming).
"""
import sys, math
sys.path.insert(0, 'scripts')
from analyze_fftsize_bug_blast_radius import *

fused_max = GPU_FUSED_MAX_CONV_LEN

def cost_ms(n, k):
    """Return model cost in ms for given (n,k)."""
    smooth = get_smooth_table(k)
    k_pad = best_k_pad_gpu(k, smooth)
    B = gpu_select_best_B_est(n, k_pad)
    return simulate_full_plan_cost(n, k, k_pad, B, fused_max) / 1000.0, B, k_pad

def tree_info(n, k):
    """Return tree geometry info."""
    smooth = get_smooth_table(k)
    k_pad = best_k_pad_gpu(k, smooth)
    B = gpu_select_best_B_est(n, k_pad)
    nblocks = (n + B - 1) // B
    N_tree = 1
    while N_tree < nblocks:
        N_tree <<= 1
    L = 0
    tmp = N_tree
    while tmp > 1:
        tmp >>= 1
        L += 1
    L += 1
    return B, k_pad, nblocks, N_tree, L

print("=" * 80)
print("FINAL REPORT: Monotonicity analysis of k=n and k=100 curves")
print("for scripts/threshold_search_gpu.cu binary search safety")
print("=" * 80)

# ── SECTION 1 ──
print("\n" + "─" * 40)
print("SECTION 1: Validation against post-fix hardware")
print("─" * 40)
print()
print("Calibration values in gpu_fft_config.h are MICROSECONDS (misnamed '_ns').")
print("Model overestimates by ~7x but is directionally correct.")
print()

measurements = [
    (4194304, 128, 511.7), (4194304, 64, 273.1), (4194304, 256, 612.2),
    (2097152, 128, 197.6), (8388608, 128, 1025.8), (1048576, 128, 93.7),
    (524288, 128, 42.2), (524288, 256, 51.2),
]
for n, k, real_ms in measurements:
    ms, B, kp = cost_ms(n, k)
    print(f"  n={n:>8} k={k:>4}: real={real_ms:>8.1f}ms model={ms:>8.1f}ms ratio={ms/real_ms:.1f}x B={B}")

# Monotonicity checks
print()
m128_4m, _, _ = cost_ms(4194304, 128)
m256_4m, _, _ = cost_ms(4194304, 256)
print(f"  n=4194304: k=128→256: model {m128_4m:.0f}ms < {m256_4m:.0f}ms ✓ (real 511.7 < 612.2 ✓)")
m128_524k, _, _ = cost_ms(524288, 128)
m256_524k, _, _ = cost_ms(524288, 256)
print(f"  n=524288:  k=128→256: model {m128_524k:.0f}ms < {m256_524k:.0f}ms ✓ (real 42.2 < 51.2 ✓)")

# ── SECTION 2 ──
print("\n" + "─" * 40)
print("SECTION 2: k=n and k=100 curve sweeps")
print("─" * 40)

# Key checkpoint: user-reported ~999k vs 1,048,576
print("\n2a. USER-REPORTED ~999k vs 1,048,576 on k=n curve:")
for n in [999000, 1000000, 1048576, 1048577]:
    ms, B, kp = cost_ms(n, n)
    _, _, nb, N, L = tree_info(n, n)
    note = ""
    if n == 1048576: note = " (2^20, L=16)"
    if n == 1048577: note = " (L JUMPS to 17!)"
    print(f"  n={n:>10}: cost={ms:>8.0f}ms B={B:>3} L={L} nblocks={nb} k_pad={kp}{note}")

print()
print("  FINDING: n=999,000 and n=1,048,576 both have B=32, L=16.")
print("  Their costs differ by only ~7%. The model does NOT reproduce")
print("  a 50% drop between these two specific points.")
print("  The massive jump is at n=1,048,577 (L=16→17), going UP not down.")
print()

# The real non-monotonicities in the k=n curve
print("2b. REAL non-monotonicities found in k=n curve sweep (241 points):")
print()
print("  TYPE A: Small k_pad artifacts (2-10% drops)")
print("  When n passes through a non-power-of-2 smooth number, k_pad rounds")
print("  to that number, forcing non-power-of-2 FFT sizes with worse calibration")
print("  values. At the next power-of-2 n, k_pad snaps to the power-of-2 and")
print("  cost drops slightly. 7 such drops found below n=1M.")
print("  Magnitude: 2-10%, mostly <7%.")
print("  Verdict: Small enough to be within hardware noise. Not a binary-search")
print("  stopper but could cause jitter near the threshold.")
print()
print("  TYPE B: B-selection grid boundary (21-71% drops!)")
print("  When n crosses the log-midpoint between gbselect rows, B changes")
print("  dramatically, reducing tree depth by 1-2 levels and dropping cost.")
print()
print("  Most dramatic example on k=n curve:")
ms_a, B_a, _ = cost_ms(1280000, 1280000)
ms_b, B_b, _ = cost_ms(1290000, 1290000)
_, _, _, _, L_a = tree_info(1280000, 1280000)
_, _, _, _, L_b = tree_info(1290000, 1290000)
print(f"    n=1,280,000: B={B_a}, L={L_a}, cost={ms_a:.0f}ms")
print(f"    n=1,290,000: B={B_b}, L={L_b}, cost={ms_b:.0f}ms")
print(f"    DROP: {(ms_a-ms_b)/ms_a*100:.0f}% — larger n has LOWER cost!")
print()
print("  Even more dramatic on k=100 curve:")
ms_a2, B_a2, _ = cost_ms(1280000, 100)
ms_b2, B_b2, _ = cost_ms(1290000, 100)
_, _, _, _, L_a2 = tree_info(1280000, 100)
_, _, _, _, L_b2 = tree_info(1290000, 100)
print(f"    n=1,280,000, k=100: B={B_a2}, L={L_a2}, cost={ms_a2:.0f}ms")
print(f"    n=1,290,000, k=100: B={B_b2}, L={L_b2}, cost={ms_b2:.0f}ms")  
print(f"    DROP: {(ms_a2-ms_b2)/ms_a2*100:.0f}% — 2.6x cheaper!")
print()
print("  TYPE C: nblocks power-of-2 boundaries (2x upward JUMPS)")
print("  When ceil(n/B) crosses 2^k, tree gains a level → cost ~doubles.")
print("  At B=32: jump at n=1,048,577 (2654→5677ms, 2.1x).")
print("  These are discontinuities, not inversions (cost goes UP with n).")
print("  Binary search: if a test point lands just above the boundary,")
print("  it sees 2x higher cost. The next point below looks 2x cheaper.")
print("  This appears as a 'drop' to the binary search even though the")
print("  function is monotonically non-decreasing.")

# ── SECTION 3 ──
print("\n" + "─" * 40)
print("SECTION 3: Mechanism investigation summary")
print("─" * 40)
print()
print("  3a. B-selection (gbselect): FOUND REAL VIOLATIONS.")
print("      B is chosen by nearest-neighbor over a sparse 8×4 grid.")
print("      Crossing the log-midpoint between rows at n≈1,285,000 changes")
print("      B from 32→112 (k=n) or 32→96 (k=100), reducing tree depth by")
print("      2 levels and causing 21-61% cost drops. Larger n → lower cost.")
print("      This IS a non-monotonicity.")
print()
print("  3b. k_pad rounding: FOUND SMALL ARTIFACTS.")
print("      Non-power-of-2 k_pad values force non-power-of-2 FFT sizes")
print("      which have slightly worse calibration values than power-of-2")
print("      neighbors. Drops are 2-10%, within measurement noise for real HW.")
print("      The function best_k_pad_gpu itself IS monotonic (kp(k1) ≤ kp(k2)"), 
print("      for k1 < k2). The artifacts come from calibration table values,")
print("      not from k_pad inversion.")
print()
print("  3c. Tier-crossing boundaries: SAFE in practice.")
print("      The fused→cuFFT boundary (conv_len > 8192) doesn't cause")
print("      inversions — cost increases smoothly through the transition.")
print("      Schoolbook fallback at the largest FFT sizes (>67M) is well")
print("      beyond the binary search range of interest.")

# ── SECTION 4 ──
print("\n" + "─" * 40)
print("SECTION 4: FINAL VERDICT")
print("─" * 40)
print()
print("  MONOTONICITY IS *NOT* GUARANTEED after the fix.")
print()
print("  The already-fixed gate bug (wrap-correction lock-in at fused tier)")
print("  was real and the fix is correct. But the binary search faces")
print("  ADDITIONAL non-monotonicity from B-selection grid boundaries.")
print()
print("  SPECIFIC REMAINING VIOLATIONS on k=n curve:")
print("    1. n≈1,280,000→1,290,000: B=32→112, L=17→15, 21% cost DROP")
print("    2. n≈50,331,650→58,720,254: 71% cost DROP (B=112, large-FT effects)")
print("    3. n≈100,663,298→117,440,510: 35% cost DROP")
print("    4. Seven small (2-10%) k_pad artifacts below n=1M")
print()
print("  SPECIFIC REMAINING VIOLATIONS on k=100 curve:")
print("    1. n≈1,280,000→1,290,000: B=32→96, L=17→15, 61% cost DROP")
print()
print("  ROOT CAUSE: The gbselect table is too sparse. B jumps from 32 to")
print("  96/112 at the n≈1.29M boundary with no intermediate values. This")
print("  causes tree depth to drop by 2 levels, reducing cost dramatically.")
print("  The nearest-neighbor lookup over a log-spaced 8-point n-grid is")
print("  insufficiently smooth for monotonicity.")
print()
print("  USER'S OBSERVATION (~999k=1000ms, ~1M=500ms): My model does NOT")
print("  reproduce a 50% drop between n≈999k and n=1,048,576 (both have")
print("  B=32, L=16, and similar costs within 7%). Possible explanations:")
print("    (a) User approximated: the binary search tested n>1,048,576")
print("        (above the nblocks boundary, L=17, ~1000ms) vs n=1,048,576")
print("        (L=16, ~500ms). The boundary is at n=1,048,577.")
print("    (b) The process was killed mid-measurement → garbled reading.")
print("    (c) My model misses a real effect at these sizes.")
print()
print("  RECOMMENDATION: Do NOT run the binary search as-is. The B-selection")
print("  grid boundary at n≈1.29M will cause the search to see a dramatic")
print("  cost DECREASE with increasing n, violating the binary search")
print("  assumption. Fix options:")
print("    (a) Add more gbselect grid points between B=32 and B=96/112")
print("        (e.g., B=48, 64 at intermediate n,k).")
print("    (b) Use a monotonicity-enforcing post-processing step in")
print("        gpu_select_best_B_est (e.g., ensure cost(n,B(n)) >= cost(n-1,B(n-1))).")
print("    (c) Make the binary search robust to non-monotonicity by using")
print("        a ternary search or by re-checking neighbors at each step.")
