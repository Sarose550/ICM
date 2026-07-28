/*
 * test_uncalibrated_fallback.c: Verify graceful behaviour when built
 * against devices/generic/ (no calibration data at all).
 *
 * The failure this guards against: past the calibrated ceiling, the
 * schoolbook-vs-FFT cost comparison consults tables that do not cover the
 * requested size. Before the calibration-boundary work, both sides of that
 * comparison were meaningless there and schoolbook always won, so a query
 * past the ceiling silently degraded to O(len^2) multiplication: correct
 * results, arbitrarily slowly, with no warning. These tests pin the
 * replacement behaviour.
 *
 * Tests:
 *   (a) CALIBRATED_MAX_CONV_LEN is -1 (never calibrated) for this device.
 *   (b) The empty crossover / B-selection tables are guarded: no
 *       out-of-bounds read, crossover 0 (always hybrid), B = 32.
 *   (c) Dispatch is hybrid at every size, never linear.
 *   (d) The smooth-number table extends past its initial 131072 ceiling,
 *       so an uncalibrated device can still pick a viable FFT size at
 *       any practical scale.
 *   (e) A real query at a size past any calibration matches the closed
 *       form, and a large one is obviously not O(n^2).
 *
 * Build (macOS):
 *   gcc -O3 -march=native -Wall -Isrc/cpu -Idevices/generic \
 *       -I/opt/homebrew/include \
 *       -o test_uncalibrated_fallback tools/test_uncalibrated_fallback.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 *
 * Build (Linux):
 *   gcc -O3 -march=native -Wall -Isrc/cpu -Idevices/generic \
 *       -o test_uncalibrated_fallback tools/test_uncalibrated_fallback.c \
 *       -lfftw3 -lm
 *
 * Run: ./test_uncalibrated_fallback   (exit 0 = all passed)
 *
 * Machine-independent: the correctness reference is a closed form, not a
 * comparison against another engine or a recorded timing.
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

/* Include the implementation directly, as bench.c does, to reach the
 * internal dispatch helpers. Pulls in devices/generic/fft_config.h. */
#include "icm.c"

static int tests_run = 0, tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  %-46s ", name); fflush(stdout); } while (0)
#define PASS()     do { printf("PASS\n"); tests_passed++; } while (0)
#define FAIL(msg)  do { printf("FAIL: %s\n", msg); } while (0)

/* ── (a) the ceiling constant ───────────────────────────────────── */

static void test_ceiling_constant(void) {
    TEST("CALIBRATED_MAX_CONV_LEN == -1");
    if (CALIBRATED_MAX_CONV_LEN == -1) PASS();
    else { printf("(got %d) ", CALIBRATED_MAX_CONV_LEN); FAIL("expected -1"); }
}

/* ── (b) empty-table guards ─────────────────────────────────────── */

static void test_empty_table_guards(void) {
    TEST("empty crossover/B tables: guarded, no OOB read");

    if (N_CROSSOVER_POINTS != 0) { FAIL("expected N_CROSSOVER_POINTS == 0"); return; }
    if (N_BSELECT_POINTS   != 0) { FAIL("expected N_BSELECT_POINTS == 0");   return; }

    /* Before the guards these read crossover_n[0] / bselect_n[0] on a
     * zero-length table. Exercise a spread of n so a stale index would
     * be likely to show up. */
    int bad = 0;
    for (int n = 16; n <= (1 << 20); n *= 2) {
        double kc = empirical_crossover_k(n);
        /* select_engine_ex() reads this as `k < k_cross ? linear : hybrid`,
         * so "always hybrid" is a crossover of ZERO, not a large one. */
        if (kc != 0.0) { printf("(n=%d crossover=%.0f, expected 0) ", n, kc); bad++; break; }
        if (empirical_best_B(n, 200) != 32) { printf("(n=%d B!=32) ", n); bad++; break; }
    }
    if (!bad) PASS(); else FAIL("wrong uncalibrated fallback value");
}

/* ── (c) dispatch is always hybrid ──────────────────────────────── */

static void test_always_hybrid(void) {
    TEST("dispatch is hybrid at every size, never linear");

    static const int cases[][2] = {
        {   64,   16}, {  128,   32}, {  256,   64}, {  512,  128},
        { 1024,  200}, { 2048,  400}, { 4096,  800}, { 8192, 1000},
        {16384, 2000}, {32768, 4000}, {65536, 8000},
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        int n = cases[i][0], k = cases[i][1];
        int B = select_engine(n, k);
        if (B == 0) { printf("(n=%d,k=%d chose linear) ", n, k); bad++; break; }
        /* B is clamped to <= min(n,k) internally, so only assert the
         * uncalibrated default where that clamp cannot bind. */
        if (n >= 64 && k >= 64 && B != 32) {
            printf("(n=%d,k=%d B=%d, expected 32) ", n, k, B); bad++; break;
        }
    }
    if (!bad) PASS(); else FAIL("unexpected dispatch");
}

/* ── (d) the smooth table extends past its initial ceiling ──────── */

static void test_smooth_table_extends(void) {
    TEST("smooth-number table extends past 131072");

    /* The table is built to 131072 initially. Without on-demand extension
     * an uncalibrated device could not pick an FFT size above that, which
     * is exactly the regime this fallback exists to serve. */
    int bad = 0;
    const int probes[] = {131073, 200000, 1000000, 5000000};
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        int want = probes[i];
        int got = next_smooth_ge(want);
        if (got < want) { printf("(next_smooth_ge(%d)=%d < input) ", want, got); bad++; break; }
        /* Must be genuinely 7-smooth, not a next_pow2 bailout. */
        int r = got;
        while (r % 2 == 0) r /= 2;
        while (r % 3 == 0) r /= 3;
        while (r % 5 == 0) r /= 5;
        while (r % 7 == 0) r /= 7;
        if (r != 1) { printf("(next_smooth_ge(%d)=%d not 7-smooth) ", want, got); bad++; break; }
    }
    if (!bad) PASS(); else FAIL("smooth table did not extend correctly");
}

/* ── (e) correctness and complexity past the ceiling ────────────── */

/* Identical stacks with a payout vector summing to 1: by symmetry every
 * player finishes in each position with equal probability, so every equity
 * is exactly 1/n and they sum to 1. A closed form, independent of any
 * engine, so this cannot pass by agreeing with a second wrong answer.
 *
 * Two properties are checked separately because they have different
 * sensitivities. The SPREAD between equities is exact symmetry and holds to
 * rounding at any Q. The SUM carries the quadrature error, so its tolerance
 * must match the Q actually used (Q=256 reaches ~1e-13; Q=64 only ~1e-5).
 * Conflating the two is how a correct result gets read as a failure. */
static int run_uniform_case(int n, int k, int Q, double sum_tol,
                            double *out_secs, double *out_sum, double *out_spread) {
    double *S      = (double *)malloc((size_t)n * sizeof(double));
    double *payout = (double *)malloc((size_t)k * sizeof(double));
    double *equity = (double *)malloc((size_t)n * sizeof(double));
    if (!S || !payout || !equity) { free(S); free(payout); free(equity); return -1; }

    for (int i = 0; i < n; i++) S[i]      = 1000.0;
    for (int j = 0; j < k; j++) payout[j] = 1.0 / (double)k;

    double t0 = now_ns();
    icm_equity(n, S, Q, payout, k, equity);
    *out_secs = (now_ns() - t0) * 1e-9;

    double expect = 1.0 / (double)n, lo = equity[0], hi = equity[0], sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (equity[i] < lo) lo = equity[i];
        if (equity[i] > hi) hi = equity[i];
        sum += equity[i];
    }
    *out_sum    = sum;
    *out_spread = (hi - lo) / expect;

    free(S); free(payout); free(equity);
    return (*out_spread < 1e-12) && (fabs(sum - 1.0) < sum_tol);
}

static void test_correct_past_ceiling(void) {
    TEST("n=8192 k=8192 Q=256: matches closed form");
    double secs, sum, spread;
    int ok = run_uniform_case(8192, 8192, 256, 1e-11, &secs, &sum, &spread);
    printf("[%.2fs, sum-1=%+.2e, spread=%.1e] ", secs, sum - 1.0, spread);
    if (ok == 1) PASS(); else if (ok < 0) FAIL("allocation failed"); else FAIL("wrong equities");
}

static void test_not_quadratic(void) {
    const int n = 65536, k = 65536, Q = 64;
    TEST("n=65536 k=65536: not O(n^2)");

    /* O(n^2) here is ~4.3e9 FMAs per quadrature point, ~2.7e14 in total. No
     * machine finishes that in two minutes, so this bound discriminates
     * algorithmic collapse rather than asserting a performance level.
     * Q=64 keeps the run short; the sum tolerance is widened to match. */
    double secs, sum, spread;
    int ok = run_uniform_case(n, k, Q, 1e-4, &secs, &sum, &spread);
    printf("[%.2fs, sum-1=%+.2e, spread=%.1e] ", secs, sum - 1.0, spread);

    if (ok < 0)           FAIL("allocation failed");
    else if (!ok)         FAIL("wrong equities");
    else if (secs >= 120) FAIL("too slow - likely quadratic fallback");
    else                  PASS();
}

int main(void) {
    printf("=== Uncalibrated fallback (devices/generic/) ===\n\n");

    test_ceiling_constant();
    test_empty_table_guards();
    test_always_hybrid();
    test_smooth_table_extends();
    test_correct_past_ceiling();
    test_not_quadratic();

    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
