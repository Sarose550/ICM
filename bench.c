/*
 * bench.c — Full n×n ICM benchmark (CPU)
 *
 * Benchmarks the selected backend (AVX2 or AVX-512) across all test
 * distributions and n values.  Reports timing and accuracy (V1 relative
 * error).
 *
 * Compile (AVX2 only):
 *   gcc -O3 -march=native -mavx2 -mfma -o bench_avx2 \
 *       bench.c icm_common.c icm_detect.c icm_avx2.c -lm
 *
 * Compile (with AVX-512):
 *   gcc -O3 -march=native -mavx2 -mfma -mavx512f -mavx512dq -o bench_avx512 \
 *       bench.c icm_common.c icm_detect.c icm_avx2.c icm_avx512.c -lm
 *
 * Run:
 *   ./bench_avx2                # default: n=2048 Q=256
 *   ./bench_avx2 512 256        # n Q
 *   ./bench_avx2 --quick        # quick correctness test
 */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "icm.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double bench_fn(ICMFunc fn, int n, const double *S, int Q,
                       const QP *pts, double *prob, int reps) {
    fn(n, S, Q, pts, prob); /* warmup */
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        fn(n, S, Q, pts, prob);
        double t = (now_s() - t0) * 1000.0;
        if (t < best) best = t;
    }
    return best;
}

int main(int argc, char **argv) {
    int quick = 0;
    int n = 2048, Q = 256, reps = 5;
    double ratio = 1e9;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick = 1; n = 512; reps = 2;
        } else if (i == 1) n = atoi(argv[1]);
        else if (i == 2) Q = atoi(argv[2]);
        else if (i == 3) reps = atoi(argv[3]);
    }

    printf("========================================================\n");
    printf("       ICM FULL n×n BENCHMARK (CPU)\n");
    printf("========================================================\n");
    icm_print_cpu_info(stdout);

    ICMFunc fn = icm_best_backend();
    printf("n=%d, Q=%d, ratio=%.0e, reps=%d\n\n", n, Q, ratio, reps);

    const char *dnames[] = {"adversarial", "reverse_adv", "bimodal", "geometric", "uniform"};

    double *S    = (double *)malloc((size_t)n * sizeof(double));
    double *eV1  = (double *)malloc((size_t)n * sizeof(double));
    double *prob = (double *)malloc((size_t)n * n * sizeof(double));
    QP     *pts  = (QP *)malloc((size_t)Q * sizeof(QP));

    /* ── Per-distribution benchmark ───────────────────────────── */
    printf("%14s %10s %12s\n", "distribution", "time (ms)", "V1 rel err");
    printf("-------------------------------------------\n");

    for (int di = 0; di < 5; di++) {
        icm_make_stacks(n, ratio, di, S);
        double Smax = icm_smax(n, S);
        icm_make_nodes(Q, Smax, pts);
        icm_exact_V1(n, S, eV1);

        double ms = bench_fn(fn, n, S, Q, pts, prob, reps);
        double err = icm_max_relV1(n, prob, eV1);

        printf("%14s %9.1f ms %12.2e\n", dnames[di], ms, err);
    }

    if (quick) {
        printf("\nQuick test passed.\n");
        free(S); free(eV1); free(prob); free(pts);
        return 0;
    }

    /* ── Scaling sweep ────────────────────────────────────────── */
    printf("\n========================================================\n");
    printf("  SCALING (adversarial, Q=%d)\n", Q);
    printf("========================================================\n");
    printf("%6s %10s %12s\n", "n", "time (ms)", "V1 rel err");
    printf("----------------------------------\n");

    int sweep[] = {128, 256, 512, 1024, 2048, 4096};
    int nsweep = 6;

    for (int si = 0; si < nsweep; si++) {
        int sn = sweep[si];
        if (sn > n * 2) break;  /* don't go way beyond requested n */

        double *sS = (double *)malloc((size_t)sn * sizeof(double));
        double *sV = (double *)malloc((size_t)sn * sizeof(double));
        double *sp = (double *)malloc((size_t)sn * sn * sizeof(double));
        QP     *sq = (QP *)malloc((size_t)Q * sizeof(QP));

        icm_make_stacks(sn, ratio, 0, sS);
        icm_make_nodes(Q, icm_smax(sn, sS), sq);
        icm_exact_V1(sn, sS, sV);

        double ms = bench_fn(fn, sn, sS, Q, sq, sp, reps);
        double err = icm_max_relV1(sn, sp, sV);

        printf("%6d %9.1f ms %12.2e\n", sn, ms, err);

        free(sS); free(sV); free(sp); free(sq);
    }

    free(S); free(eV1); free(prob); free(pts);
    printf("\nDone.\n");
    return 0;
}
