/* bench_topk.c — Benchmark for simplified top-k API
 * gcc -O3 -march=native -mavx2 -mfma -o bench_topk bench_topk.c icm_common.c icm_avx2.c icm_topk.c -lm
 */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "icm.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double bench(void (*fn)(void *), void *ctx, int R) {
    fn(ctx); /* warmup */
    double best = 1e30;
    for (int r = 0; r < R; r++) {
        double t0 = now_s(); fn(ctx); double t = (now_s() - t0) * 1000;
        if (t < best) best = t;
    }
    return best;
}

/* Contexts for benchmarking */
typedef struct { int n; const double *S; int Q; const QP *pts; double *prob; } FullCtx;
typedef struct { int n; const double *S; int Q; const QP *pts; int k; double *prob; } TkCtx;
typedef struct { int n; const double *S; int Q; const QP *pts; const double *pay; int k; double *eq; } EqCtx;
typedef struct { int n; const double *S; int Q; const QP *pts; const int *pl; int np; const double *pay; int k; double *eq; } SubCtx;

static void run_full(void *c) { FullCtx *x = c; icm_avx2(x->n, x->S, x->Q, x->pts, x->prob); }
static void run_topk(void *c) { TkCtx *x = c; icm_topk(x->n, x->S, x->Q, x->pts, x->k, x->prob); }
static void run_eq(void *c)   { EqCtx *x = c; icm_equity(x->n, x->S, x->Q, x->pts, x->pay, x->k, x->eq); }
static void run_sub(void *c)  { SubCtx *x = c; icm_equity_sub(x->n, x->S, x->Q, x->pts, x->pl, x->np, x->pay, x->k, x->eq); }

static double equity_err(int n, int k, const double *pay, const double *eq, const double *full) {
    double mx = 0;
    for (int i = 0; i < n; i++) {
        double ref = 0;
        for (int m = 0; m < k; m++) ref += pay[m] * full[(size_t)i * n + m];
        double re = (fabs(ref) > 1e-300) ? fabs(eq[i] - ref) / fabs(ref) : fabs(eq[i]);
        if (re > mx) mx = re;
    }
    return mx;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 2048;
    int Q = (argc > 2) ? atoi(argv[2]) : 256;
    double ratio = 1e9;
    int R = 3;

    printf("========================================================\n");
    printf("       TOP-K BENCHMARK (simplified API)\n");
    printf("========================================================\n");
    icm_print_cpu_info(stdout);
    printf("n=%d, Q=%d, ratio=%.0e\n\n", n, Q, ratio);

    int ks[] = {5, 10, 20, 50};
    int nk = 4; while (nk > 0 && ks[nk-1] >= n) nk--;

    const char *dn[] = {"adversarial", "geometric", "uniform"};
    int dl[] = {0, 3, 4};

    double *S    = malloc(n * sizeof(double));
    double *full = malloc((size_t)n * n * sizeof(double));
    QP     *pts  = malloc(Q * sizeof(QP));

    for (int d = 0; d < 3; d++) {
        icm_make_stacks(n, ratio, dl[d], S);
        icm_make_nodes(Q, icm_smax(n, S), pts);
        printf("--- %s ---\n\n", dn[d]);

        FullCtx fc = {n, S, Q, pts, full};
        double mf = bench(run_full, &fc, R);
        printf("  Full n*n:     %8.1f ms\n\n", mf);

        printf("  %5s | %9s %9s %9s | %8s %8s | %10s\n",
               "k", "topk", "equity", "sub(10)", "spd_eq", "spd_sub", "eq_error");
        printf("  ------|-------------------------------|------------------|----------\n");

        for (int ki = 0; ki < nk; ki++) {
            int k = ks[ki];
            double pay[50]; for (int m = 0; m < k; m++) pay[m] = (double)(k - m);

            double *tk = malloc((size_t)n * k * sizeof(double));
            double *eq = malloc(n * sizeof(double));
            double eq_sub[10];
            int players[10]; for (int i = 0; i < 10; i++) players[i] = i;

            TkCtx tc = {n, S, Q, pts, k, tk};
            EqCtx ec = {n, S, Q, pts, pay, k, eq};
            SubCtx sc = {n, S, Q, pts, players, 10, pay, k, eq_sub};

            double mt = bench(run_topk, &tc, R);
            double me = bench(run_eq, &ec, R);
            double ms = bench(run_sub, &sc, R);
            double err = equity_err(n, k, pay, eq, full);

            printf("  %5d | %8.1fms %8.1fms %8.1fms | %7.1fx %7.1fx | %10.2e\n",
                   k, mt, me, ms, mf / me, mf / ms, err);

            free(tk); free(eq);
        }
        printf("\n");
    }

    /* Scaling: top-10 equity across n */
    printf("========================================================\n");
    printf("  SCALING: equity top-10 across n (adversarial)\n");
    printf("========================================================\n");
    int sns[] = {256, 512, 1024, 2048, 4096, 8192};
    double pay10[10]; for (int m = 0; m < 10; m++) pay10[m] = 10 - m;

    printf("  %6s | %10s %10s %10s | %8s %8s | %10s\n",
           "n", "full", "equity", "sub(10)", "spd_eq", "spd_sub", "error");
    printf("  -------|----------------------------------|------------------|-----------\n");

    for (int si = 0; si < 6; si++) {
        int sn = sns[si];
        double *sS = malloc(sn * sizeof(double));
        double *sf = malloc((size_t)sn * sn * sizeof(double));
        double *se = malloc(sn * sizeof(double));
        double se_sub[10];
        int pl10[10]; for (int i = 0; i < 10; i++) pl10[i] = i;
        QP *sp = malloc(Q * sizeof(QP));

        icm_make_stacks(sn, ratio, 0, sS);
        icm_make_nodes(Q, ratio, sp);

        FullCtx fc = {sn, sS, Q, sp, sf};
        EqCtx ec = {sn, sS, Q, sp, pay10, 10, se};
        SubCtx sc = {sn, sS, Q, sp, pl10, 10, pay10, 10, se_sub};

        double mf = bench(run_full, &fc, R);
        double me = bench(run_eq, &ec, R);
        double ms = bench(run_sub, &sc, R);
        double err = equity_err(sn, 10, pay10, se, sf);

        printf("  %6d | %9.1fms %9.1fms %9.1fms | %7.1fx %7.1fx | %10.2e\n",
               sn, mf, me, ms, mf / me, mf / ms, err);

        free(sS); free(sf); free(se); free(sp);
    }

    free(S); free(full); free(pts);
    printf("\nDone.\n");
    return 0;
}
