/* bench_schoolbook_tree.c — isolated microbenchmark for schoolbook tree ops.
 *
 * Directly measures polymul_modk() (schoolbook multiply/build) and
 * correlate_school() (schoolbook correlate) for every calib_sizes[] entry
 * up to a generous cutoff (4096), replicating the exact operand sizes
 * seen at real tree-level call sites.
 *
 * polymul_modk() call site (tree_build_levels, src/icm.c ~line 1395):
 *   polymul_modk(Lc, cps, Rc, cps, out, pps)
 * where pps = min(2*cps, k_padded), i.e., 2*cps in unsaturated case.
 * Both child polynomials have degree cps-1 stored in cps slots.
 *
 * correlate_school() call site (tree_propagate_g, src/icm.c ~line 1474):
 *   correlate_school(gp, g_eff, PR, p_eff, gL, out_needed)
 * where p_eff = cps (saturated), out_needed ≈ cps,
 *       g_eff = out_needed + p_eff - 1 ≈ 2*cps - 1.
 *
 * Methodology (same rigor as bench_wrap_fma.c / bench_block_build.c):
 *   - Median-of-9 timing for noise suppression
 *   - Rep count adaptive to target ~50ms per timing run
 *   - Least-squares slope of ns vs fma_count extracts per-FMA coefficient,
 *     eliminating fixed per-call overhead from the raw ratio
 *   - Sink variable to prevent compiler from optimizing away dead stores
 *
 * Output:
 *   1. CSV on stdout with per-size measurements
 *   2. Final block with C array initializers for schoolbook_mul_ns[] and
 *      schoolbook_corr_ns[], indexed identically to calib_sizes[].
 *   3. Least-squares fit summary (SCHOOLBOOK_MUL_FMA_NS, SCHOOLBOOK_CORR_FMA_NS)
 *
 * Build (Zen4):
 *   gcc -O3 -march=znver4 -o bench_schoolbook_tree \
 *       tools/bench_schoolbook_tree.c \
 *       -Isrc -Idevices/zen4 -lfftw3 -lm -ldl -lmvec
 */
#include "icm.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define SCHOOLBOOK_CUTOFF 4096

/* now_ns() is provided by icm.c (included above) */

/* ── Benchmark polymul_modk ──────────────────────────────────────
 *
 * Replicates: polymul_modk(Lc, cps, Rc, cps, out, 2*cps)
 * Full-range unsaturated case — worst-case that dominates cost model.
 */
static double bench_polymul_modk(int cps, int reps, double *sink) {
    int out_sz = 2 * cps;
    double *A = (double *)calloc((size_t)cps, sizeof(double));
    double *B = (double *)calloc((size_t)cps, sizeof(double));
    double *C = (double *)calloc((size_t)out_sz, sizeof(double));

    srand((unsigned)(42 + cps));
    for (int i = 0; i < cps; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    double local_sink = 0.0;
    double t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        polymul_modk(A, cps, B, cps, C, out_sz);
        local_sink += C[0] + C[out_sz - 1];
    }
    double t1 = now_ns();
    *sink += local_sink;

    free(A); free(B); free(C);
    return (t1 - t0) / (double)reps;
}

/* Exact FMA count for polymul_modk(A, na, B, nb, C, k) */
static long long polymul_modk_fma_count(int na, int nb, int k) {
    /* The inner loop: for i in [0, min(na,k)-1], j in [0, min(nb, k-i)-1]
     * Each iteration is one FMA. */
    long long count = 0;
    int imax = (na < k) ? na : k;
    for (int i = 0; i < imax; i++) {
        int jmax = nb;
        if (i + jmax > k) jmax = k - i;
        count += jmax;
    }
    return count;
}

/* ── Benchmark correlate_school ──────────────────────────────────
 *
 * Replicates: correlate_school(gp, g_eff, PR, p_eff, gL, out_needed)
 * with p_eff = cps, out_needed = cps, g_eff = 2*cps - 1 (saturated case).
 */
static double bench_correlate_school(int cps, int reps, double *sink) {
    int g_eff    = 2 * cps - 1;
    int p_eff    = cps;
    int out_need = cps;

    double *G = (double *)calloc((size_t)g_eff, sizeof(double));
    double *P = (double *)calloc((size_t)p_eff, sizeof(double));
    double *Out = (double *)calloc((size_t)out_need, sizeof(double));

    srand((unsigned)(42 + cps));
    for (int i = 0; i < g_eff; i++) G[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < p_eff; i++) P[i] = (double)rand() / RAND_MAX;

    double local_sink = 0.0;
    double t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        correlate_school(G, g_eff, P, p_eff, Out, out_need);
        local_sink += Out[0] + Out[out_need - 1];
    }
    double t1 = now_ns();
    *sink += local_sink;

    free(G); free(P); free(Out);
    return (t1 - t0) / (double)reps;
}

/* Exact FMA count for correlate_school(g, len_g, P, len_P, out, len_out) */
static long long correlate_school_fma_count(int len_g, int len_P, int len_out) {
    /* For each m in [0, len_out-1]: jmax = min(len_P, len_g - m) FMAs */
    long long count = 0;
    for (int m = 0; m < len_out; m++) {
        int jmax = len_P;
        if (m + jmax > len_g) jmax = len_g - m;
        if (jmax > 0) count += jmax;
    }
    return count;
}

/* ── Simple 2-parameter least-squares ──────────────────────────── */
static void fit_linear(int ndata, const double *x, const double *y,
                       double *slope, double *intercept, double *r2) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < ndata; i++) {
        sx  += x[i];
        sy  += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
    }
    double denom = (double)ndata * sxx - sx * sx;
    if (fabs(denom) < 1e-30) { *slope = *intercept = *r2 = 0.0; return; }
    *slope     = ((double)ndata * sxy - sx * sy) / denom;
    *intercept = (sy - (*slope) * sx) / (double)ndata;
    double ymean = sy / (double)ndata;
    double ss_res = 0, ss_tot = 0;
    for (int i = 0; i < ndata; i++) {
        double ypred = (*slope) * x[i] + (*intercept);
        ss_res += (y[i] - ypred) * (y[i] - ypred);
        ss_tot += (y[i] - ymean) * (y[i] - ymean);
    }
    *r2 = (ss_tot > 1e-30) ? 1.0 - ss_res / ss_tot : 0.0;
}

/* ── Main ──────────────────────────────────────────────────────── */

int main(void) {
    /* Disable stdout buffering — this runs as a background job and
     * we don't want to lose output if killed. */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    build_fftw_size_table();
    icm_init(NULL);

    int NTIMES = 9;

    /* Find cutoff index in calib_sizes */
    int cutoff_idx = 0;
    while (cutoff_idx < N_CALIBRATED_SIZES && calib_sizes[cutoff_idx] <= SCHOOLBOOK_CUTOFF)
        cutoff_idx++;

    fprintf(stderr, "Benchmarking %d calib_sizes entries up to cutoff %d (N_CALIBRATED_SIZES=%d)\n",
            cutoff_idx, SCHOOLBOOK_CUTOFF, N_CALIBRATED_SIZES);

    /* Allocate result arrays */
    double *mul_ns  = (double *)calloc((size_t)N_CALIBRATED_SIZES, sizeof(double));
    double *corr_ns = (double *)calloc((size_t)N_CALIBRATED_SIZES, sizeof(double));
    /* Init all to -1 (sentinel for above-cutoff) */
    for (int i = 0; i < N_CALIBRATED_SIZES; i++) {
        mul_ns[i] = -1.0;
        corr_ns[i] = -1.0;
    }

    /* For least-squares fit: exclude tiny sizes where overhead dominates */
    #define MAX_FIT_PTS 1024
    double fit_x_mul[MAX_FIT_PTS], fit_y_mul[MAX_FIT_PTS];
    double fit_x_corr[MAX_FIT_PTS], fit_y_corr[MAX_FIT_PTS];
    int nfit_mul = 0, nfit_corr = 0;
    int fit_start_cps = 16;  /* exclude cps < 16 where overhead dominates */

    double global_sink = 0.0;

    printf("cps_idx,cps,mul_ns,corr_ns,mul_fma_count,corr_fma_count\n");

    for (int i = 0; i < cutoff_idx; i++) {
        int cps = calib_sizes[i];

        /* FMA counts for the exact call patterns we benchmark */
        long long mul_fma  = polymul_modk_fma_count(cps, cps, 2 * cps);
        long long corr_fma = correlate_school_fma_count(2 * cps - 1, cps, cps);

        /* Separate reps for mul and corr: target ~50ms per timing run.
         * corr is 3-7x more expensive per call (cache/memory bound). */
        double est_mul_ns  = (double)cps * (double)cps * 0.07;
        int reps_mul = (int)(50e6 / est_mul_ns);
        if (reps_mul < 20) reps_mul = 20;
        if (reps_mul > 5000000) reps_mul = 5000000;

        double est_corr_ns = (double)cps * (double)cps * 0.30;  /* corr is ~4x costlier */
        int reps_corr = (int)(50e6 / est_corr_ns);
        if (reps_corr < 10) reps_corr = 10;
        if (reps_corr > 2000000) reps_corr = 2000000;

        /* Median-of-N for polymul_modk */
        {
            double times[9];
            for (int rep = 0; rep < NTIMES; rep++)
                times[rep] = bench_polymul_modk(cps, reps_mul, &global_sink);
            for (int a = 0; a < NTIMES; a++)
                for (int b = a + 1; b < NTIMES; b++)
                    if (times[b] < times[a]) { double t = times[a]; times[a] = times[b]; times[b] = t; }
            mul_ns[i] = times[NTIMES / 2];
        }

        /* Median-of-N for correlate_school */
        {
            double times[9];
            for (int rep = 0; rep < NTIMES; rep++)
                times[rep] = bench_correlate_school(cps, reps_corr, &global_sink);
            for (int a = 0; a < NTIMES; a++)
                for (int b = a + 1; b < NTIMES; b++)
                    if (times[b] < times[a]) { double t = times[a]; times[a] = times[b]; times[b] = t; }
            corr_ns[i] = times[NTIMES / 2];
        }

        printf("%d,%d,%.4f,%.4f,%lld,%lld\n",
               i, cps, mul_ns[i], corr_ns[i], mul_fma, corr_fma);
        fflush(stdout);

        fprintf(stderr, "  [%3d/%3d] cps=%4d  mul=%9.1f ns  corr=%9.1f ns  (reps_mul=%d reps_corr=%d)\n",
                i + 1, cutoff_idx, cps, mul_ns[i], corr_ns[i], reps_mul, reps_corr);

        /* Accumulate for least-squares fit (exclude tiny sizes) */
        if (cps >= fit_start_cps && nfit_mul < MAX_FIT_PTS) {
            fit_x_mul[nfit_mul] = (double)mul_fma;
            fit_y_mul[nfit_mul] = mul_ns[i];
            nfit_mul++;
        }
        if (cps >= fit_start_cps && nfit_corr < MAX_FIT_PTS) {
            fit_x_corr[nfit_corr] = (double)corr_fma;
            fit_y_corr[nfit_corr] = corr_ns[i];
            nfit_corr++;
        }
    }

    fprintf(stderr, "sink_guard(ignore)=%.6f\n", global_sink);

    /* ── Least-squares fit ─────────────────────────────────────── */
    double mul_slope, mul_intercept, mul_r2;
    double corr_slope, corr_intercept, corr_r2;

    fit_linear(nfit_mul, fit_x_mul, fit_y_mul, &mul_slope, &mul_intercept, &mul_r2);
    fit_linear(nfit_corr, fit_x_corr, fit_y_corr, &corr_slope, &corr_intercept, &corr_r2);

    printf("\n");
    printf("# Least-squares fit (ns vs FMA count, cps >= %d):\n", fit_start_cps);
    printf("# SCHOOLBOOK_MUL_FMA_NS=%.6f  (intercept=%.2f ns, R²=%.6f, n=%d)\n",
           mul_slope, mul_intercept, mul_r2, nfit_mul);
    printf("# SCHOOLBOOK_CORR_FMA_NS=%.6f  (intercept=%.2f ns, R²=%.6f, n=%d)\n",
           corr_slope, corr_intercept, corr_r2, nfit_corr);

    /* ── Emit C arrays ─────────────────────────────────────────── */
    printf("\n");
    printf("/* ── Schoolbook per-size lookup tables ── */\n");
    printf("/* Generated by tools/bench_schoolbook_tree.c on Zen4 */\n");
    printf("/* Indexed identically to calib_sizes[].  -1.0 = above cutoff (not measured). */\n");
    printf("\n");
    printf("#ifndef SCHOOLBOOK_MUL_NS_DEFINED\n");
    printf("#define SCHOOLBOOK_MUL_NS_DEFINED\n");
    printf("static const double schoolbook_mul_ns[N_CALIBRATED_SIZES] = {\n");
    for (int i = 0; i < N_CALIBRATED_SIZES; i++) {
        if (i % 10 == 0) printf("    ");
        if (mul_ns[i] < 0)
            printf("-1.0");
        else
            printf("%.4f", mul_ns[i]);
        if (i < N_CALIBRATED_SIZES - 1) printf(",");
        if ((i + 1) % 10 == 0) printf("\n");
        if (i < N_CALIBRATED_SIZES - 1 && (i + 1) % 10 != 0) printf(" ");
    }
    printf("\n};\n");
    printf("#endif /* SCHOOLBOOK_MUL_NS_DEFINED */\n");
    printf("\n");
    printf("#ifndef SCHOOLBOOK_CORR_NS_DEFINED\n");
    printf("#define SCHOOLBOOK_CORR_NS_DEFINED\n");
    printf("static const double schoolbook_corr_ns[N_CALIBRATED_SIZES] = {\n");
    for (int i = 0; i < N_CALIBRATED_SIZES; i++) {
        if (i % 10 == 0) printf("    ");
        if (corr_ns[i] < 0)
            printf("-1.0");
        else
            printf("%.4f", corr_ns[i]);
        if (i < N_CALIBRATED_SIZES - 1) printf(",");
        if ((i + 1) % 10 == 0) printf("\n");
        if (i < N_CALIBRATED_SIZES - 1 && (i + 1) % 10 != 0) printf(" ");
    }
    printf("\n};\n");
    printf("#endif /* SCHOOLBOOK_CORR_NS_DEFINED */\n");
    printf("\n");
    printf("/* Fitted per-FMA coefficients (ns per scalar FMA in schoolbook inner loops) */\n");
    printf("#ifndef SCHOOLBOOK_MUL_FMA_NS\n");
    printf("#define SCHOOLBOOK_MUL_FMA_NS %.6f\n", mul_slope);
    printf("#endif\n");
    printf("#ifndef SCHOOLBOOK_CORR_FMA_NS\n");
    printf("#define SCHOOLBOOK_CORR_FMA_NS %.6f\n", corr_slope);
    printf("#endif\n");

    free(mul_ns); free(corr_ns);
    return 0;
}
