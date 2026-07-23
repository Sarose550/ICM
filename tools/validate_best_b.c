/* validate_best_b.c — Direct empirical validation of icm_select_best_B(n,k).
 *
 * For each (n,k) in a sparse grid spanning the just-fixed linear/hybrid
 * crossover region (where B-selection actually matters), this tool times
 * the REAL hybrid engine at EVERY candidate B ∈ {8,16,24,32,48,64} using
 * median-of-7-reps timing — the project's established discipline — and
 * reports whether the cost-model-driven icm_select_best_B chooses the
 * empirically-fastest B.
 *
 * Methodology:
 *   Q=256 (matches bench_grid/crossover conventions)
 *   payout[m] = (double)(n - m)  (matches calibrate_crossover)
 *   S[i] = 100.0 + 9900.0 * rand()/RAND_MAX  with srand(42)
 *   median of 7 reps per candidate
 *
 * Build:
 *   gcc -O3 -march=native -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o build/validate_best_b tools/validate_best_b.c src/icm.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 *
 * This is a throwaway diagnostic tool — context objects are deliberately
 * leaked rather than calling icm_ctx_destroy with potentially-wrong
 * EngineKind enum values.
 */

#include "icm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_REPS   7
#define Q_PROBE  256

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) ? 1 : (da < db) ? -1 : 0;
}

int main(void) {
    icm_init(NULL);

    int n_vals[]      = {1024, 4096, 8192, 16384};
    int n_n            = 4;
    int k_vals[]       = {150, 250, 400, 800, 2000};
    int n_k            = 5;
    int B_candidates[] = {8, 16, 24, 32, 48, 64};
    int n_B            = 6;

    /* Table header */
    printf("    n     k  model_B  real_fastest_B  "
           "model_B_time_ns  real_fastest_B_time_ns  %%slower\n");
    printf("------ ----- -------- ---------------  "
           "---------------  ----------------------  --------\n");

    int verdict_ok = 1;

    for (int ni = 0; ni < n_n; ni++) {
        int n = n_vals[ni];

        for (int ki = 0; ki < n_k; ki++) {
            int k = k_vals[ki];
            if (k > n) continue;

            /* Allocate working arrays */
            double *S      = (double *)malloc(n * sizeof(double));
            double *payout = (double *)malloc(k * sizeof(double));
            double *equity = (double *)malloc(n * sizeof(double));

            /* Generate stacks: same convention as every other tool in this project */
            srand(42);
            for (int i = 0; i < n; i++)
                S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);

            /* payout[m] = (double)(n-m) — matches bench_grid crossover */
            for (int m = 0; m < k; m++)
                payout[m] = (double)(n - m);

            /* Model's choice */
            int model_B = icm_select_best_B(n, k);

            /* Time every candidate B, find the empirically fastest */
            int    best_B     = -1;
            double best_time  = 1e18;

            for (int bi = 0; bi < n_B; bi++) {
                int B = B_candidates[bi];
                if (B > k || B > n) continue;

                double samples[N_REPS];
                for (int r = 0; r < N_REPS; r++) {
                    void *hc = icm_hybrid_ctx_create(n, S, k, B);
                    samples[r] = icm_run_engine(n, S, Q_PROBE, payout, k,
                                                 equity, icm_engine_hybrid(), hc);
                    /* Deliberately leak hc — see file header */
                }
                qsort(samples, N_REPS, sizeof(double), cmp_double);
                double med = samples[N_REPS / 2];

                if (med < best_time) {
                    best_time = med;
                    best_B    = B;
                }
            }

            /* Find model_B's median time: re-run a fresh timing specifically
             * for model_B so the number is directly comparable (same noise
             * environment) rather than cherry-picked from the scan above. */
            double model_time = 0.0;
            {
                double samples[N_REPS];
                for (int r = 0; r < N_REPS; r++) {
                    void *hc = icm_hybrid_ctx_create(n, S, k, model_B);
                    samples[r] = icm_run_engine(n, S, Q_PROBE, payout, k,
                                                 equity, icm_engine_hybrid(), hc);
                }
                qsort(samples, N_REPS, sizeof(double), cmp_double);
                model_time = samples[N_REPS / 2];
            }

            /* Also re-time the real fastest B for a fair head-to-head.
             * The scan above gave us a rank-order, but we want a fresh
             * median from the same "noise epoch" as model_time. */
            double real_time = 0.0;
            {
                double samples[N_REPS];
                for (int r = 0; r < N_REPS; r++) {
                    void *hc = icm_hybrid_ctx_create(n, S, k, best_B);
                    samples[r] = icm_run_engine(n, S, Q_PROBE, payout, k,
                                                 equity, icm_engine_hybrid(), hc);
                }
                qsort(samples, N_REPS, sizeof(double), cmp_double);
                real_time = samples[N_REPS / 2];
            }

            /* Calculate % slower: (model_time - real_time) / real_time * 100 */
            double pct_slower = 100.0 * (model_time - real_time) / real_time;
            if (pct_slower < 0.0) pct_slower = 0.0;  /* model was faster — good */

            printf("%6d %5d %8d %15d  %15.1f %22.1f  %7.2f%%\n",
                   n, k, model_B, best_B,
                   model_time, real_time, pct_slower);

            if (best_B != model_B && pct_slower > 5.0) {
                verdict_ok = 0;
            }

            free(S);
            free(payout);
            free(equity);
        }
    }

    printf("\n── VERDICT ──\n");
    if (verdict_ok) {
        printf("PASS: icm_select_best_B selects the empirically-optimal B "
               "(or within ~5%%) across the tested grid.\n");
    } else {
        printf("ACTIONABLE GAP: icm_select_best_B does NOT reliably select "
               "the fastest B — the summed-constants cost model is inaccurate "
               "for B-selection, just as it was for the linear/hybrid "
               "crossover decision.\n");
    }
    printf("\n");

    return verdict_ok ? 0 : 1;
}
