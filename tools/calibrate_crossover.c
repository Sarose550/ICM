/* calibrate_crossover.c — Direct empirical measurement of the real
 * linear-vs-hybrid crossover point k_cross(n), replacing the summed-
 * constants analytical formula's go/no-go decision with a small,
 * per-device lookup table (LAPACK ILAENV's NX crossover parameter is the
 * precedent: two algorithms, an empirically-measured problem-size
 * crossover, baked into a cheap runtime comparison -- no live racing in
 * production).
 *
 * For each n in a fixed sparse grid, binary-searches k to find where the
 * hybrid engine starts winning against the batched linear engine, using
 * real timing (median of several reps) at each candidate k -- exactly
 * what tools/bench.c's "crossover" mode already does on a fixed grid,
 * just refined to the exact boundary via bisection instead of reading it
 * off coarse steps.
 *
 * This is a ONE-TIME, OFFLINE calibration step (like tools/calibrate.c's
 * PATIENT wisdom sweep) -- it never runs in production. Output is a
 * small table of (n, k_cross) pairs to paste into devices/<DEVICE>/fft_config.h.
 *
 * Build (macOS M3 Pro):
 *   gcc -O3 -march=native -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o build/calibrate_crossover tools/calibrate_crossover.c src/icm.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 * Build (Linux/Zen4, AOCL-FFTW):
 *   gcc -O3 -march=znver4 -Isrc -Idevices/zen4 -I/usr/local/aocl-fftw/include \
 *       -o build/calibrate_crossover tools/calibrate_crossover.c src/icm.c \
 *       -L/usr/local/aocl-fftw/lib -Wl,-rpath,/usr/local/aocl-fftw/lib \
 *       -lfftw3 -lm -ldl -lmvec
 */
#include "icm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_REPS 7

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) ? 1 : (da < db) ? -1 : 0;
}

/* Q=256 matches bench_grid crossover's own sweep exactly -- using fewer
 * quadrature points would under-amortize any fixed per-call overhead
 * (FFTW plan lookup, etc.) differently between engines, systematically
 * biasing the measured crossover away from what bench_grid crossover
 * reports as ground truth. */
#define Q_PROBE 256

/* Median real per-QP time for the linear engine at (n,k). */
static double time_linear(int n, int k, const double *S, const double *payout,
                           double *equity) {
    double samples[N_REPS];
    void *lc = icm_linear_ctx_create(n, k);
    for (int r = 0; r < N_REPS; r++)
        samples[r] = icm_run_linear_batched(n, S, Q_PROBE, payout, k, equity, lc) / (double)Q_PROBE;
    qsort(samples, N_REPS, sizeof(double), cmp_double);
    return samples[N_REPS / 2];
}

/* Median real per-QP time for the hybrid engine at (n,k), fresh context
 * each rep. */
static double time_hybrid(int n, int k, const double *S, const double *payout,
                           double *equity) {
    double samples[N_REPS];
    int B = icm_select_best_B(n, k); /* matches bench_grid crossover's own
                                       * methodology -- comparing against a
                                       * fixed/hardcoded B would handicap
                                       * hybrid and shift the measured
                                       * crossover later than the true one. */
    for (int r = 0; r < N_REPS; r++) {
        void *hc = icm_hybrid_ctx_create(n, S, k, B);
        samples[r] = icm_run_engine(n, S, Q_PROBE, payout, k, equity,
                                     icm_engine_hybrid(), hc) / (double)Q_PROBE;
    }
    qsort(samples, N_REPS, sizeof(double), cmp_double);
    return samples[N_REPS / 2];
}

/* Returns 1 if hybrid wins (is faster) at (n,k), 0 if linear wins. */
static int hybrid_wins(int n, int k, const double *S, double *payout,
                        double *equity) {
    /* Matches bench/bench.c's crossover-sweep payout exactly (payout[m] =
     * n-m), so this tool measures the SAME crossover bench_grid crossover
     * reports as ground truth -- not a differently-shaped one. */
    for (int q = 0; q < k; q++) payout[q] = (double)(n - q);
    double lin_ns = time_linear(n, k, S, payout, equity);
    double hyb_ns = time_hybrid(n, k, S, payout, equity);
    return hyb_ns < lin_ns;
}

int main(void) {
    icm_init(NULL);

    int n_vals[] = {512, 1024, 2048, 4096, 8192, 16384};
    int n_n = 6;

    printf("# Direct empirical crossover measurement (median of %d reps per point)\n", N_REPS);
    printf("# n,k_cross\n");

    for (int ni = 0; ni < n_n; ni++) {
        int n = n_vals[ni];
        double *S = (double *)malloc(n * sizeof(double));
        double *payout = (double *)malloc(2001 * sizeof(double));
        double *equity = (double *)malloc(n * sizeof(double));
        srand(42);
        for (int i = 0; i < n; i++)
            S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);

        /* Binary search k in [lo, hi] for the crossover: largest k where
         * linear still wins, +1. Bracket first: confirm linear wins at
         * lo=10 and hybrid wins at hi=n (or a large cap), matching this
         * project's own bench_grid crossover sweep's practical range. */
        int lo = 10, hi = (n < 2000) ? n : 2000;

        int lo_hybrid = hybrid_wins(n, lo, S, payout, equity);
        int hi_hybrid = hybrid_wins(n, hi, S, payout, equity);
        fprintf(stderr, "n=%d: bracket lo=%d(%s) hi=%d(%s)\n",
                n, lo, lo_hybrid ? "H" : "L", hi, hi_hybrid ? "H" : "L");

        if (lo_hybrid) {
            /* Hybrid already wins at the smallest k tested -- crossover is
             * at or below lo. Report lo itself. */
            printf("%d,%d\n", n, lo);
            fflush(stdout);
            free(S); free(payout); free(equity);
            continue;
        }
        if (!hi_hybrid) {
            /* Linear still wins even at hi -- crossover is above the
             * tested range. Report hi as a lower-bound sentinel. */
            fprintf(stderr, "  WARNING: linear still wins at hi=%d, crossover above range\n", hi);
            printf("%d,%d\n", n, hi);
            fflush(stdout);
            free(S); free(payout); free(equity);
            continue;
        }

        while (hi - lo > 4) {
            int mid = (lo + hi) / 2;
            int w = hybrid_wins(n, mid, S, payout, equity);
            fprintf(stderr, "  probe k=%d -> %s\n", mid, w ? "H" : "L");
            if (w) hi = mid; else lo = mid;
        }

        printf("%d,%d\n", n, hi);
        fflush(stdout);

        free(S); free(payout); free(equity);
    }

    return 0;
}
