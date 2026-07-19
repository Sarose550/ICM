/* force_hybrid_test.c — Directly measure hybrid engine at k=1638 bypassing dispatch.
 *
 * Build: gcc -O3 -march=native -Isrc -Idevices/m3_max -I/opt/homebrew/include
 *        -o tools/force_hybrid_test tools/force_hybrid_test.c
 *        -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 */
#include "icm.c"
#include <stdio.h>
#include <stdlib.h>

static double measure(const char *label, int n, int k, int B, int Q, int nrep) {
    double *S = (double *)malloc(n * sizeof(double));
    double *equity = (double *)malloc(n * sizeof(double));
    double *payout = (double *)malloc(k * sizeof(double));
    if (!S || !equity || !payout) { fprintf(stderr, "OOM\n"); exit(1); }

    srand(42);
    for (int i = 0; i < n; i++) S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
    for (int q = 0; q < k; q++) payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

    double *times = (double *)malloc(nrep * sizeof(double));

    if (B > 0) {
        /* Directly create hybrid context and run */
        void *hc = icm_hybrid_ctx_create(n, S, k, B);
        if (!hc) { fprintf(stderr, "hybrid_ctx_create failed!\n"); exit(1); }
        /* Warmup */
        icm_run_engine(n, S, Q, payout, k, equity, icm_engine_hybrid(), hc);
        for (int rep = 0; rep < nrep; rep++) {
            double t0 = now_ns();
            icm_run_engine(n, S, Q, payout, k, equity, icm_engine_hybrid(), hc);
            times[rep] = now_ns() - t0;
        }
        icm_ctx_destroy(hc, ICM_ENGINE_HYBRID);
    } else {
        /* Linear */
        void *lc = icm_linear_ctx_create(n, k);
        icm_run_linear_batched(n, S, Q, payout, k, equity, lc);
        for (int rep = 0; rep < nrep; rep++) {
            double t0 = now_ns();
            icm_run_linear_batched(n, S, Q, payout, k, equity, lc);
            times[rep] = now_ns() - t0;
        }
        icm_ctx_destroy(lc, ICM_ENGINE_LINEAR);
    }

    /* Median */
    for (int i = 0; i < nrep; i++)
        for (int j = i+1; j < nrep; j++)
            if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

    double median = times[nrep/2];
    double min_ns = times[0], max_ns = times[nrep-1];
    printf("%s  median=%.3f ms  per_qp_ns=%.0f  range=%.1f%%\n",
           label, median/1e6, median/Q,
           (max_ns-min_ns)/median*100);

    free(times);
    free(S); free(equity); free(payout);
    return median;
}

int main(void) {
    build_fftw_size_table();
    icm_init(NULL);

    int Q = 64, nrep = 5;

    printf("=== n=16384 k=1638: LINEAR vs HYBRID (direct, bypassing dispatch) ===\n");
    printf("select_engine(16384,1638) = %d\n", icm_select_engine(16384, 1638));
    printf("select_best_B(16384,1638) = %d\n", icm_select_best_B(16384, 1638));

    measure("LINEAR                ", 16384, 1638, 0, Q, nrep);
    measure("HYBRID B=32           ", 16384, 1638, 32, Q, nrep);
    measure("HYBRID B=64           ", 16384, 1638, 64, Q, nrep);
    measure("HYBRID B=128          ", 16384, 1638, 128, Q, nrep);
    measure("HYBRID B=256          ", 16384, 1638, 256, Q, nrep);

    printf("\n=== n=16384 k=4096 ===\n");
    printf("select_engine(16384,4096) = %d\n", icm_select_engine(16384, 4096));
    measure("LINEAR                ", 16384, 4096, 0, Q, nrep);
    measure("HYBRID B=64 (dispatch)", 16384, 4096, 64, Q, nrep);

    printf("\n=== n=16384 k=2048 ===\n");
    printf("select_engine(16384,2048) = %d\n", icm_select_engine(16384, 2048));
    measure("LINEAR                ", 16384, 2048, 0, Q, nrep);
    measure("HYBRID B=64           ", 16384, 2048, 64, Q, nrep);

    return 0;
}
