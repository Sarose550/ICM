/* test_wrap_feasibility.c: pins the correlate wrap feasibility bound in
 * best_fft_config() / best_fft_config_joint() (VERDICTS.md V20).
 *
 * The correlate implementations (correlate_fft, correlate_fft_pair,
 * correlate_fft_cached_wrap, correlate_fft_cached_pair_wrap in
 * src/cpu/icm.c) require fft_n >= len_g: the g operand must fit the
 * transform whole. In chooser terms, for a correlate of convolution
 * length L = len_g + len_P - 1, that is wrap_m <= len_P - 1.
 *
 * Past that bound the cached correlates silently TRUNCATE g
 * (copy_g = min(len_g, fft_n)) while their wrap corrections model the
 * cyclic aliasing of a g that fully fit, producing wrong output (the
 * mechanism behind the 2026-08-03 Zen4 776-size tree-engine verify
 * failures: n=16 chose fft_n=8 for len_g=9, dropping g[8] and
 * corrupting out[0] and out[4] by O(1) terms). The non-cached
 * correlates would heap-overflow instead.
 *
 * This test uses a SYNTHETIC calibration table crafted so that the
 * infeasible candidate wins the cost race outright; only the hard
 * feasibility constraint can reject it. That makes the test independent
 * of any real device's calibration data, which is exactly how the
 * original bug hid: it needed a data-dependent cost race to surface.
 *
 * Build & run (any platform, no FFTW needed):
 *   gcc -O3 -Wall -Wno-unused-function -Isrc/cpu \
 *       -o test_wrap_feasibility tools/test_wrap_feasibility.c -lm
 *   ./test_wrap_feasibility     # expect 4/4 passed
 */

#include <stdio.h>
#include <math.h>

/* ── Synthetic device data (stands in for devices/<D>/fft_config.h) ──
 * Size 8 is priced absurdly cheap so it wins every cost comparison it
 * is allowed to enter; everything larger is expensive. */
#define N_CALIBRATED_SIZES 13
static const int calib_sizes[N_CALIBRATED_SIZES] =
    { 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32 };
static const double calib_times_ns[N_CALIBRATED_SIZES] =
    { 1.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0,
      500.0, 500.0, 500.0, 500.0, 500.0, 500.0 };
#define WRAP_FMA_NS 0.5
#define PAIRED_CACHED_CORR_RATIO 1.55
#define N_CROSSOVER_POINTS 0
static const int crossover_n[1] = {0};
static const int crossover_k[1] = {0};
#define N_BSELECT_POINTS 0
static const int bselect_n[1] = {0};
static const int bselect_k[1] = {0};
static const int bselect_B[1] = {0};

/* Minimal 7-smooth successor (only exercised by the wrap-free fallback) */
static int next_smooth_ge(int n) {
    for (int s = n; ; s++) {
        int r = s;
        while (r % 2 == 0) r /= 2;
        while (r % 3 == 0) r /= 3;
        while (r % 5 == 0) r /= 5;
        while (r % 7 == 0) r /= 7;
        if (r == 1) return s;
    }
}

#include "fft_cost_model.h"

int main(void) {
    int passed = 0, failed = 0;

    /* Case 1: best_fft_config, correlate mode (len_P > 0).
     * L=13, len_P=5 -> len_g=9. Size 8 (wrap 5) is the cost winner but
     * infeasible (8 < len_g); the chooser must return size >= 9. */
    {
        int fn, wm;
        best_fft_config(13, &fn, &wm, 5);
        int len_g = 13 - 5 + 1;
        if (fn >= len_g && wm <= 4) {
            printf("PASS best_fft_config correlate: size=%d wrap=%d (len_g=%d)\n",
                   fn, wm, len_g);
            passed++;
        } else {
            printf("FAIL best_fft_config correlate: size=%d wrap=%d violates "
                   "fft_n >= len_g=%d\n", fn, wm, len_g);
            failed++;
        }
    }

    /* Case 2: best_fft_config_joint. build_conv=8, corr_conv=13, p_eff=5
     * (the exact n=16 tree level-3 shape). Joint size 8 wins on cost but
     * needs corr wrap 5 >= p_eff; must be rejected. */
    {
        int S, mb, mc;
        best_fft_config_joint(8, 13, 5, &S, &mb, &mc);
        int len_g = 13 - 5 + 1;
        if (S >= len_g && mc <= 4) {
            printf("PASS best_fft_config_joint: size=%d build_m=%d corr_m=%d\n",
                   S, mb, mc);
            passed++;
        } else {
            printf("FAIL best_fft_config_joint: size=%d corr_m=%d violates "
                   "fft_n >= len_g=%d\n", S, mc, len_g);
            failed++;
        }
    }

    /* Case 3: convolution mode (len_P == 0) must NOT be over-constrained:
     * polymul reads its wrap terms from the original input arrays, so the
     * cheap size-8 candidate with wrap 5 is legitimate there. */
    {
        int fn, wm;
        best_fft_config(13, &fn, &wm, 0);
        if (fn == 8 && wm == 5) {
            printf("PASS convolution mode still allows wrap: size=%d wrap=%d\n",
                   fn, wm);
            passed++;
        } else {
            printf("FAIL convolution mode over-constrained: size=%d wrap=%d "
                   "(expected 8/5)\n", fn, wm);
            failed++;
        }
    }

    /* Case 4: sweep every (L, len_P) shape a tree/hybrid level can
     * produce and assert the invariant holds throughout. */
    {
        int bad = 0;
        for (int len_P = 1; len_P <= 33; len_P++) {
            for (int out = 1; out <= 64; out++) {
                int len_g = out + len_P - 1;
                int L = len_g + len_P - 1;
                int fn, wm;
                best_fft_config(L, &fn, &wm, len_P);
                if (fn < len_g) {
                    printf("FAIL sweep: L=%d len_P=%d -> size=%d < len_g=%d\n",
                           L, len_P, fn, len_g);
                    bad = 1;
                }
                int S, mb, mc;
                best_fft_config_joint(2 * len_P - 1, L, len_P, &S, &mb, &mc);
                if (S < len_g) {
                    printf("FAIL sweep joint: L=%d p_eff=%d -> size=%d < len_g=%d\n",
                           L, len_P, S, len_g);
                    bad = 1;
                }
            }
        }
        if (!bad) { printf("PASS feasibility sweep (len_P 1..33, out 1..64)\n"); passed++; }
        else failed++;
    }

    printf("%d/%d passed\n", passed, passed + failed);
    return failed ? 1 : 0;
}
