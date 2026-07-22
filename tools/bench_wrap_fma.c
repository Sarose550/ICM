/* bench_wrap_fma.c — isolated microbenchmark for the wrap-correction loop.
 *
 * Directly measures WRAP_FMA_NS by timing ONLY the wrap-correction inner
 * loop (verbatim copy of src/icm.c's polymul_fft_wrap correction), sweeping
 * wrap_m over a wide range so it dominates the measured time by construction.
 *
 * The indirect regression in tools/fit_cost_model.py is structurally unable
 * to identify WRAP_FMA_NS from full-plan timing data alone (wrap-correction
 * cost never exceeds ~1.5% of any sampled plan's total predicted time — the
 * fit has no real signal for it).  This microbenchmark provides a direct,
 * isolated measurement that should be pinned in the fit rather than floated.
 *
 * Two array-size regimes (SMALL/MED/LARGE) to check whether the constant is
 * stable across memory-locality conditions, matching the range of polynomial
 * sizes actually seen at wrap-correcting tree levels.  The extraction script
 * (or calibrate_full.sh) should take the local slope in the realistic wrap_m
 * operating range for the SMALL regime where the inner loop is L1-resident.
 *
 * Build: gcc -O3 -march=native -o bench_wrap_fma bench_wrap_fma.c
 *
 * Output: CSV on stdout (regime,na,wrap_m,fft_n,fma_count,median_ns_per_call).
 * The WRAP_FMA_NS estimate = median_ns_per_call / fma_count, averaged over
 * the small-wrap_m region (wrap_m ≤ 32) of the SMALL regime.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Verbatim copy of the correction loop body from polymul_fft_wrap
 * (src/icm.c).  Returns exact FMA count performed. */
static long long wrap_correct(const double *a, int na, const double *b, int nb,
                               double *c, int k, int fft_n, int wrap_m,
                               double *out_sink) {
    int da = na - 1, db = nb - 1;
    long long fma_count = 0;
    double sink = 0.0;
    int fft_out = (fft_n < k) ? fft_n : k;
    for (int i = 0; i <= wrap_m; i++) {
        int pos = fft_n + i;
        double high = 0;
        int j_lo = pos - db; if (j_lo < 0) j_lo = 0;
        int j_hi = da; if (j_hi > pos) j_hi = pos;
        for (int j = j_lo; j <= j_hi; j++) {
            high += a[j] * b[pos - j];
            fma_count++;
        }
        if (i < fft_out) c[i] -= high;
        if (pos < k) c[pos] = high;
        sink += high;
    }
    *out_sink = sink;
    return fma_count;
}

static void run_regime(const char *label, int na, int nb) {
    int k = na + nb;
    double *a = (double *)malloc((size_t)na * sizeof(double));
    double *b = (double *)malloc((size_t)nb * sizeof(double));
    double *c = (double *)malloc((size_t)k * sizeof(double));
    srand(42);
    for (int i = 0; i < na; i++) a[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < nb; i++) b[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < k; i++) c[i] = 0.0;

    int wrap_ms[] = {4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};
    int n_wm = (int)(sizeof(wrap_ms) / sizeof(wrap_ms[0]));
    int REPS = 4000;
    double sink_guard = 0.0;

    for (int wi = 0; wi < n_wm; wi++) {
        int wrap_m = wrap_ms[wi];
        int fft_n = na + nb - 1 - wrap_m - 1;
        if (fft_n < 1 || fft_n >= na + nb) continue;

        double times[9];
        long long fma_count = 0;
        for (int rep = 0; rep < 9; rep++) {
            double t0 = now_ns();
            for (int r = 0; r < REPS; r++) {
                double sink;
                fma_count = wrap_correct(a, na, b, nb, c, k, fft_n, wrap_m, &sink);
                sink_guard += sink;
            }
            double t1 = now_ns();
            times[rep] = (t1 - t0) / REPS;
        }
        for (int i = 0; i < 9; i++)
            for (int j = i + 1; j < 9; j++)
                if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
        printf("%s,%d,%d,%d,%lld,%.5f\n", label, na, wrap_m, fft_n, fma_count, times[4]);
        fflush(stdout);
    }
    fprintf(stderr, "sink_guard(ignore)=%.6f\n", sink_guard);
    free(a); free(b); free(c);
}

int main(void) {
    printf("regime,na,wrap_m,fft_n,fma_count,median_ns_per_call\n");
    run_regime("SMALL_2048", 2048, 2048);
    run_regime("MED_8192", 8192, 8192);
    run_regime("LARGE_32768", 32768, 32768);
    return 0;
}
