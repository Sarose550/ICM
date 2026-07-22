/* bench_schoolbook_tree.c — Direct per-size microbenchmark for schoolbook
 * tree-level primitives: polymul_modk (build/multiply) and correlate_school
 * (propagate/correlate).
 *
 * Produces lookup tables schoolbook_mul_ns[] and schoolbook_corr_ns[]
 * indexed identically to calib_sizes[] (N_CALIBRATED_SIZES entries).
 * Above the cutoff (1024), sentinel values of -1.0.
 *
 * Build: gcc -O3 -march=native -Isrc -Idevices/m3_pro \
 *        -I/opt/homebrew/include -o bench_schoolbook_tree \
 *        tools/bench_schoolbook_tree.c -L/opt/homebrew/lib -lfftw3 -lm \
 *        -framework Accelerate
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "icm.c"

static double median_of(int n, double *times) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (times[j] < times[i]) {
                double t = times[i]; times[i] = times[j]; times[j] = t;
            }
    return times[n / 2];
}

#define MAX_CPS 512
#define NTIMES 7

int main(void) {
    /* Only need smooth_nums for index lookups; skip FFTW wisdom (not used here) */
    build_fftw_size_table();

    int N = N_CALIBRATED_SIZES;
    const int *sizes = calib_sizes;

    int cutoff_idx = N;
    for (int i = 0; i < N; i++) {
        if (sizes[i] > MAX_CPS) { cutoff_idx = i; break; }
    }
    int n_bench = cutoff_idx;

    fprintf(stderr, "# bench_schoolbook_tree: benchmarking %d sizes (cps <= %d)\n",
            n_bench, MAX_CPS);

    double *mul_ns  = (double *)calloc((size_t)N, sizeof(double));
    double *corr_ns = (double *)calloc((size_t)N, sizeof(double));
    for (int i = 0; i < N; i++) {
        mul_ns[i]  = -1.0;
        corr_ns[i] = -1.0;
    }

    int max_cps = sizes[n_bench - 1];
    double *a  = (double *)malloc((size_t)max_cps * sizeof(double));
    double *b  = (double *)malloc((size_t)max_cps * sizeof(double));
    double *c  = (double *)malloc((size_t)(2 * max_cps) * sizeof(double));
    double *g  = (double *)malloc((size_t)(2 * max_cps) * sizeof(double));
    double *P  = (double *)malloc((size_t)(max_cps + 1) * sizeof(double));
    double *out = (double *)malloc((size_t)max_cps * sizeof(double));

    srand(42);
    for (int i = 0; i < max_cps; i++) {
        a[i] = 0.1 + 0.8 * ((double)rand() / RAND_MAX);
        b[i] = 0.1 + 0.8 * ((double)rand() / RAND_MAX);
    }
    for (int i = 0; i < 2 * max_cps; i++)
        g[i] = 0.1 + 0.8 * ((double)rand() / RAND_MAX);
    for (int i = 0; i <= max_cps; i++)
        P[i] = 0.1 + 0.8 * ((double)rand() / RAND_MAX);

    double sink = 0.0;

    for (int si = 0; si < n_bench; si++) {
        int cps = sizes[si];
        if (cps < 2) continue;

        int pps = 2 * cps;
        long long est_fmas = (long long)cps * cps;
        /* Target ~30ms per timing run for stable median.
         * Rough estimate: ~0.1 ns/FMA at small sizes (latency-bound),
         * ~0.07 ns/FMA at larger sizes. Use 0.1 for safety. */
        int reps = (int)(30e6 / (est_fmas * 0.12));
        if (reps < 50) reps = 50;
        if (reps > 1000000) reps = 1000000;

        /* ── polymul_modk ── */
        double mul_times[NTIMES];
        for (int rep = 0; rep < NTIMES; rep++) {
            double t0 = now_ns();
            for (int r = 0; r < reps; r++)
                polymul_modk(a, cps, b, cps, c, pps);
            double t1 = now_ns();
            mul_times[rep] = (t1 - t0) / (double)reps;
            sink += c[0] + c[pps-1];
        }
        mul_ns[si] = median_of(NTIMES, mul_times);

        /* ── correlate_school ── */
        int len_g = 2 * cps - 1;
        int len_P = cps;
        int len_out = cps;
        double corr_times[NTIMES];
        for (int rep = 0; rep < NTIMES; rep++) {
            double t0 = now_ns();
            for (int r = 0; r < reps; r++)
                correlate_school(g, len_g, P, len_P, out, len_out);
            double t1 = now_ns();
            corr_times[rep] = (t1 - t0) / (double)reps;
            sink += out[0] + out[len_out-1];
        }
        double corr_med = median_of(NTIMES, corr_times);
        /* Store per-(cps*g_needed) unit cost */
        corr_ns[si] = corr_med / ((double)cps * (double)cps);

        if (si < 15 || si % 30 == 0)
            fprintf(stderr, "# cps=%4d  mul=%.1f ns  corr_per_unit=%.4f ns  reps=%d\n",
                    cps, mul_ns[si], corr_ns[si], reps);
    }

    fprintf(stderr, "# sink=%.6f\n", sink);

    /* ── Emit tables parseable by calibrate_full.sh ── */
    printf("SCHOOLBOOK_MUL_NS_TABLE\n");
    for (int i = 0; i < N; i++)
        printf("%d,%.6f\n", sizes[i], mul_ns[i]);

    printf("SCHOOLBOOK_CORR_NS_TABLE\n");
    for (int i = 0; i < N; i++)
        printf("%d,%.6f\n", sizes[i], corr_ns[i]);

    free(mul_ns); free(corr_ns);
    free(a); free(b); free(c); free(g); free(P); free(out);
    return 0;
}
