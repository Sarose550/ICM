/* bench_div_chain.c — isolated microbenchmark for FP64_DIV_NS.
 *
 * Directly measures the cost of an FP64 division embedded in a genuine
 * sequential dependency chain (each division depends on the previous
 * result) — this matches how FP64_DIV_NS is actually used in src/icm.c's
 * leaf extraction (the bidirectional synthetic-division recurrence,
 * Q_m = (P_m - c*Q_{m-1})/a_j, see icm_paper.tex's Lemma lem:division-
 * stability). An independent/vectorizable division loop measures a very
 * different (much smaller) number — division throughput, not latency —
 * and is NOT representative of this usage. Same rationale as
 * bench_wrap_fma.c: this constant showed the same identifiability
 * failure signature as WRAP_FMA_NS when left to an indirect aggregate
 * fit (observed on M3 Pro: fit converged to an implausible 0.5ns and hit
 * its lower bound; this direct measurement is what should be pinned
 * instead via fit_cost_model.py --div-ns).
 *
 * Build: gcc -O3 -march=native -o bench_div_chain bench_div_chain.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main(void) {
    int N = 5000000;
    double a = 0.7, c = 0.3;  /* |c/a| < 1: numerically stable chain */
    double *P = (double *)malloc((size_t)N * sizeof(double));
    srand(42);
    for (int i = 0; i < N; i++) P[i] = (double)rand() / RAND_MAX;

    double times[7];
    for (int rep = 0; rep < 7; rep++) {
        double Q = 0.5;
        double t0 = now_ns();
        for (int i = 0; i < N; i++) {
            Q = (P[i] - c * Q) / a;   /* genuine dependency chain */
        }
        double t1 = now_ns();
        times[rep] = (t1 - t0) / N;
        if (Q != Q) { fprintf(stderr, "NaN encountered, aborting\n"); return 1; }
    }
    for (int i = 0; i < 7; i++)
        for (int j = i + 1; j < 7; j++)
            if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

    printf("regime,ns_per_division\n");
    printf("chained_dependency,%.4f\n", times[3]);  /* median of 7 */
    return 0;
}
