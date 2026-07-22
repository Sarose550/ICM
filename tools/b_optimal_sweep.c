/* b_optimal_sweep.c — Measure hybrid engine time for every B at each (n,k).
 *
 * For each (n,k) cell, directly times the hybrid engine at EVERY B in
 * {8,16,24,32,48,64,96,128,192,256}. Q=256, 3 reps per cell (median).
 * Also records what icm_select_best_B(n,k) would have picked.
 *
 * Output CSV: n,k,B,time_ms,cost_model_pick
 *
 * Compile (Zen4 Linux):
 *   gcc -O3 -march=native -Isrc -Idevices/zen4 \
 *       -o b_optimal_sweep tools/b_optimal_sweep.c \
 *       -lfftw3 -lm -ldl -lmvec -fopenmp
 * Compile (macOS):
 *   gcc -O3 -march=native -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o b_optimal_sweep tools/b_optimal_sweep.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 */

#define ICM_BENCH_INCLUDE
#include "icm.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int dbl_cmp(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Generate k values for a given n.
 * Places values in ks[] (caller-allocated, max 32 slots).
 * Returns number of k values. */
static int gen_k_grid(int n, int *ks) {
    int nk = 0;

    /* Crossover region (k ≈ 275 on Zen4) — always include dense bracket */
    int cross_ks[] = {200, 240, 260, 275, 290, 310, 350, 400};
    int n_cross = sizeof(cross_ks) / sizeof(cross_ks[0]);

    for (int i = 0; i < n_cross; i++) {
        if (cross_ks[i] <= n && cross_ks[i] >= 16) {
            ks[nk++] = cross_ks[i];
        }
    }

    /* Log-spaced grid above crossover, up to n.
     * Use roughly 6-8 additional points (depending on n range). */
    int n_extra = (n <= 4096) ? 8 : (n <= 16384) ? 6 : 4;
    double log_min = log(500.0);  /* start above crossover bracket */
    double log_max = log((double)n);

    for (int i = 0; i < n_extra; i++) {
        double t = (double)(i + 1) / (double)(n_extra + 1);
        int k = (int)round(exp(log_min + t * (log_max - log_min)));
        /* Round to nice numbers */
        if (k < 500) k = 500;
        if (k > n) k = n;
        /* Avoid duplicates near crossover */
        int dup = 0;
        for (int j = 0; j < nk; j++) {
            if (abs(ks[j] - k) < 20) { dup = 1; break; }
        }
        if (!dup && k >= 16 && k <= n) ks[nk++] = k;
    }

    /* Always include k=n if not already present */
    int has_n = 0;
    for (int i = 0; i < nk; i++) if (ks[i] == n) { has_n = 1; break; }
    if (!has_n && n >= 16) ks[nk++] = n;

    /* Sort */
    for (int i = 0; i < nk; i++)
        for (int j = i + 1; j < nk; j++)
            if (ks[j] < ks[i]) { int t = ks[i]; ks[i] = ks[j]; ks[j] = t; }

    return nk;
}

int main(void) {
    build_fftw_size_table();
    icm_init(NULL);

    int n_values[] = {512, 1024, 2048, 4096, 8192, 16384, 32768};
    int n_n = 7;

    int B_values[] = {8, 16, 24, 32, 48, 64, 96, 128, 192, 256};
    int n_B = 10;

    int Q = 256;
    int reps = 3;

    /* CSV header */
    printf("n,k,B,time_ms,cost_model_pick\n");
    fflush(stdout);

    for (int ni = 0; ni < n_n; ni++) {
        int n = n_values[ni];

        /* Generate k grid for this n */
        int ks[32];
        int nk = gen_k_grid(n, ks);

        /* Allocate once per n */
        double *S = (double *)malloc(n * sizeof(double));
        double *equity = (double *)malloc(n * sizeof(double));
        if (!S || !equity) { fprintf(stderr, "OOM n=%d\n", n); return 1; }

        /* Deterministic stacks */
        srand(42);
        for (int i = 0; i < n; i++)
            S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);

        for (int ki_idx = 0; ki_idx < nk; ki_idx++) {
            int k = ks[ki_idx];

            /* Payout: linear decay */
            double *payout = (double *)malloc(k * sizeof(double));
            if (!payout) { fprintf(stderr, "OOM payout k=%d\n", k); continue; }
            for (int m = 0; m < k; m++)
                payout[m] = (double)(n - m);

            /* What does the cost model pick? */
            int cost_model_B = icm_select_best_B(n, k);

            /* Test each B */
            for (int bi = 0; bi < n_B; bi++) {
                int B = B_values[bi];
                if (B > k || B > n) continue;

                double times[3];
                int valid = 0;

                for (int rep = 0; rep < reps; rep++) {
                    HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
                    if (!hc) {
                        fprintf(stderr, "hybrid_ctx_create failed n=%d k=%d B=%d\n", n, k, B);
                        continue;
                    }
                    memset(equity, 0, n * sizeof(double));
                    double t0 = now_ns();
                    run_engine_ctx(n, S, Q, payout, k, equity,
                                   engine_hybrid_ctx, hc);
                    double elapsed = now_ns() - t0;
                    hybrid_ctx_destroy(hc);

                    times[rep] = elapsed;
                    valid++;
                }

                if (valid == reps) {
                    /* Median */
                    qsort(times, reps, sizeof(double), dbl_cmp);
                    double median_ns = times[reps / 2];
                    double median_ms = median_ns / 1e6;

                    printf("%d,%d,%d,%.3f,%d\n",
                           n, k, B, median_ms, cost_model_B);
                    fflush(stdout);

                    fprintf(stderr, "  n=%d k=%d B=%d -> %.1f ms  (cost_model=%d)\n",
                            n, k, B, median_ms, cost_model_B);
                } else {
                    fprintf(stderr, "  n=%d k=%d B=%d FAILED\n", n, k, B);
                }
            }

            free(payout);
        }

        free(S);
        free(equity);
    }

    fprintf(stderr, "\nDone.\n");
    return 0;
}
