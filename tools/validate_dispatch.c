/* validate_dispatch.c: Validate icm_select_engine()'s linear-vs-hybrid
 * choice against the real measured fastest engine, over the same 6-column
 * x 7-row (n,k) grid used in the paper's Table 1 (tab:serial) and Table 2
 * (tab:parallel): n in {1024,2048,4096,8192,16384,32768,65536}, k in
 * {10,50,100,n/4,n/2,n} -- 42 cells.
 *
 * This is the reproducible source for the paper's dispatch-accuracy claim
 * ("100% correct... on all serial bench_grid cells, 83/84 parallel").
 *
 * Usage:
 *   validate_dispatch [--parallel]
 *
 * Output: one line per cell to stdout (CSV, header prefixed with #),
 * plus a summary line to stderr.
 *   n,k,auto_engine,auto_ms,true_engine,true_ms,correct
 *
 * auto_engine/true_engine: L or H. correct: 1 if they match.
 *
 * Timing: adaptive median-of-N (3-15 reps, cv<=3%), same convergence loop
 * as validate_best_b.c/calibrate_best_b.c (VERDICTS.md V6 -- a single-rep
 * timing is not trustworthy at small problem sizes).
 *
 * Conventions match bench_grid/validate_best_b exactly: Q=256, srand(42),
 * payout[m]=n-m, S[i]=100+9900*rand()/RAND_MAX.
 *
 * Build (macOS M3 Pro, serial):
 *   gcc -O3 -march=native -Isrc/cpu -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o validate_dispatch tools/validate_dispatch.c build/libicm.a \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 * Build (macOS M3 Pro, parallel -- pass --parallel at runtime, link OMP lib):
 *   gcc -O3 -march=native -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
 *       -Isrc/cpu -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o validate_dispatch_par tools/validate_dispatch.c build/libicm_omp.a \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate \
 *       -L/opt/homebrew/opt/libomp/lib -lomp -lfftw3_threads
 */
/* #include "icm.c" (not icm.h): icm_run_linear_batched()'s public wrapper
 * still requires the INTERNAL LinearCtx* from linear_ctx_create(), a
 * different, faster (BQ=8 batched) code path than the generic
 * icm_engine_linear()/icm_linear_ctx_create() public dispatch interface --
 * confirmed by reading icm_equity()'s own production dispatch, which
 * calls run_linear_batched_bq8, never the generic linear engine. Using
 * the generic path here silently measured the wrong (slower) linear
 * implementation and produced a false 292.75%-scale swap: an early
 * version of this tool falsely reported hybrid beating linear at
 * n=1024,k=10, contradicting bench_grid's already-verified L=1.72ms vs
 * H=4.48ms at that exact cell. */
#include "icm.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Q_PROBE     256
#define N_REPS_MIN  3
#define N_REPS_MAX  15
#define CONVERGE_CV 0.03

static const int NS[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536};
#define N_ROWS (int)(sizeof(NS) / sizeof(NS[0]))

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) ? 1 : (da < db) ? -1 : 0;
}

static double cv_of(const double *x, int count) {
    if (count < 2) return 0.0;
    double mean = 0.0;
    for (int i = 0; i < count; i++) mean += x[i];
    mean /= (double)count;
    if (mean <= 0.0) return 0.0;
    double var = 0.0;
    for (int i = 0; i < count; i++) { double d = x[i] - mean; var += d * d; }
    var /= (double)(count - 1);
    return sqrt(var) / mean;
}

static double time_linear_once(int n, int k, const double *S,
                               const double *payout, double *equity) {
    LinearCtx *lc = linear_ctx_create(n, k);
    double t = run_linear_batched(n, S, Q_PROBE, payout, k, equity, lc)
             / (double)Q_PROBE;
    return t; /* deliberate leak, short-lived offline tool */
}

static double time_hybrid_once(int n, int k, int B, const double *S,
                               const double *payout, double *equity) {
    HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
    double t = run_engine_ctx(n, S, Q_PROBE, payout, k, equity,
                              engine_hybrid_ctx, hc) / (double)Q_PROBE;
    return t;
}

static double adaptive(double (*probe)(int, int, int, const double *,
                                       const double *, double *),
                       int n, int k, int B, const double *S,
                       const double *payout, double *equity) {
    double samples[N_REPS_MAX];
    int count = 0;
    for (; count < N_REPS_MIN; count++)
        samples[count] = probe(n, k, B, S, payout, equity);
    while (count < N_REPS_MAX && cv_of(samples, count) > CONVERGE_CV)
        samples[count++] = probe(n, k, B, S, payout, equity);
    qsort(samples, count, sizeof(double), cmp_double);
    return samples[count / 2];
}

static double linear_probe(int n, int k, int B, const double *S,
                           const double *payout, double *equity) {
    (void)B;
    return time_linear_once(n, k, S, payout, equity);
}

int main(int argc, char **argv) {
    int parallel_mode = 0;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--parallel")) parallel_mode = 1;
    (void)parallel_mode; /* label only; actual thread count is a build/env choice */

    icm_init(NULL);

    int n_correct = 0, n_total = 0;
    printf("#n,k,auto_engine,auto_ms,true_engine,true_ms,correct\n");

    for (int ri = 0; ri < N_ROWS; ri++) {
        int n = NS[ri];
        int ks[6] = {10, 50, 100, n / 4, n / 2, n};

        double *S = (double *)malloc((size_t)n * sizeof(double));
        srand(42);
        for (int i = 0; i < n; i++)
            S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
        double *equity = (double *)malloc((size_t)n * sizeof(double));

        for (int ki = 0; ki < 6; ki++) {
            int k = ks[ki];
            if (k < 1) k = 1;
            if (k > n) k = n;

            double *payout = (double *)malloc((size_t)k * sizeof(double));
            for (int m = 0; m < k; m++) payout[m] = (double)(n - m);

            int auto_B = select_engine(n, k);   /* 0 = linear, >0 = hybrid B */
            int auto_is_hybrid = (auto_B > 0);

            double lin_ms = adaptive(linear_probe, n, k, 0, S, payout, equity);
            int best_B = select_best_B(n, k);
            double hyb_ms = adaptive(time_hybrid_once, n, k, best_B, S, payout, equity);

            int true_is_hybrid = (hyb_ms < lin_ms);
            double auto_ms = auto_is_hybrid ? hyb_ms : lin_ms;
            double true_ms = true_is_hybrid ? hyb_ms : lin_ms;
            int correct = (auto_is_hybrid == true_is_hybrid);

            n_total++;
            if (correct) n_correct++;

            printf("%d,%d,%s,%.6f,%s,%.6f,%d\n",
                   n, k, auto_is_hybrid ? "H" : "L", auto_ms,
                   true_is_hybrid ? "H" : "L", true_ms, correct);
            fflush(stdout);

            free(payout);
        }
        free(S);
        free(equity);
    }

    fprintf(stderr, "\nDispatch accuracy: %d/%d correct (%.1f%%)\n",
            n_correct, n_total, 100.0 * n_correct / n_total);

    return 0;
}
