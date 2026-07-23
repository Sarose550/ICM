/* calibrate_best_b.c — Direct empirical measurement of the real fastest
 * hybrid-engine block size B(n,k), replacing select_best_B()'s summed-
 * analytical-constants prediction with a small, per-device empirical
 * lookup table -- same methodology and rationale as
 * tools/calibrate_crossover.c (LAPACK ILAENV precedent: measure the real
 * decision directly rather than summing calibrated constants, which was
 * found fragile in aggregate even with every individual constant
 * validated, both for the linear-vs-hybrid crossover AND for B
 * selection -- see tools/validate_best_b.c's confirmation that
 * select_best_B() is measurably wrong by 7-11% on M3 Pro and 2-9% on
 * Zen4, same systematic direction: overestimating the benefit of larger
 * B).
 *
 * Unlike the crossover table (a continuous threshold, log-linearly
 * interpolated), B is a discrete/categorical choice among {8,16,24,32,
 * 48,64} -- there is no meaningful "interpolation" between B=32 and
 * B=64. Lookup is nearest-neighbor over a 2D (n,k) grid instead.
 *
 * For each (n,k) grid point, times the REAL hybrid engine at every
 * candidate B (median of 7 reps, Q=256, matching bench_grid crossover's
 * exact conventions), and records the empirically-fastest B.
 *
 * Build (macOS M3 Pro):
 *   gcc -O3 -march=native -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o build/calibrate_best_b tools/calibrate_best_b.c src/icm.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 * Build (Linux/Zen4, AOCL-FFTW):
 *   gcc -O3 -march=znver4 -Isrc -Idevices/zen4 -I/usr/local/aocl-fftw/include \
 *       -o build/calibrate_best_b tools/calibrate_best_b.c src/icm.c \
 *       -L/usr/local/aocl-fftw/lib -Wl,-rpath,/usr/local/aocl-fftw/lib \
 *       -lfftw3 -lm -ldl -lmvec
 */
#include "icm.h"
#include <stdio.h>
#include <stdlib.h>

#define N_REPS 7

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) ? 1 : (da < db) ? -1 : 0;
}

static double time_hybrid_B(int n, int k, int B, const double *S,
                            const double *payout, double *equity) {
    double samples[N_REPS];
    for (int r = 0; r < N_REPS; r++) {
        void *hc = icm_hybrid_ctx_create(n, S, k, B);
        samples[r] = icm_run_engine(n, S, 256, payout, k, equity,
                                     icm_engine_hybrid(), hc) / 256.0;
        /* Deliberately no icm_ctx_destroy(): this is a short-lived
         * calibration program, and getting the EngineKind enum wrong
         * (EK_TREE=0, EK_NAIVE=1, EK_LINEAR=2, EK_HYBRID=3) caused a
         * real segfault earlier this session. Leaking memory in a
         * one-shot offline tool is safe; guessing the enum wrong isn't. */
    }
    qsort(samples, N_REPS, sizeof(double), cmp_double);
    return samples[N_REPS / 2];
}

int main(void) {
    icm_init(NULL);

    int n_vals[] = {512, 1024, 2048, 4096, 8192, 16384};
    int n_n = 6;
    int k_vals[] = {150, 250, 400, 800, 1500, 2000, 4000};
    int n_k = 7;
    int B_vals[] = {8, 16, 24, 32, 48, 64};
    int n_B = 6;

    printf("# Direct empirical best-B measurement (median of %d reps per point)\n", N_REPS);
    printf("# n,k,best_B\n");

    for (int ni = 0; ni < n_n; ni++) {
        int n = n_vals[ni];
        double *S = (double *)malloc(n * sizeof(double));
        srand(42);
        for (int i = 0; i < n; i++)
            S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);

        for (int ki = 0; ki < n_k; ki++) {
            int k = k_vals[ki];
            if (k > n) continue;
            double *payout = (double *)malloc(k * sizeof(double));
            double *equity = (double *)malloc(n * sizeof(double));
            for (int q = 0; q < k; q++) payout[q] = (double)(n - q);

            double best_time = 1e18;
            int best_B = B_vals[0];
            for (int bi = 0; bi < n_B; bi++) {
                int B = B_vals[bi];
                if (B > n || B > k) continue;
                double t = time_hybrid_B(n, k, B, S, payout, equity);
                if (t < best_time) { best_time = t; best_B = B; }
            }
            printf("%d,%d,%d\n", n, k, best_B);
            fflush(stdout);
            fprintf(stderr, "n=%d k=%d -> best_B=%d (%.0f ns/qp)\n", n, k, best_B, best_time);

            free(payout); free(equity);
        }
        free(S);
    }

    return 0;
}
