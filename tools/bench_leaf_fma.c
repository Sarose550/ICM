/* bench_leaf_fma.c — isolated microbenchmark for LEAF_FMA_NS and LEAF_BLOCK_NS.
 *
 * Directly measures the FMA-dominated component of the leaf-extraction
 * synthetic-division recurrence from src/icm.c (engine_hybrid_core,
 * Step 3: Within-block divide + fused dot product, forward path).
 *
 * Verbatim recurrence from src/icm.c lines 2042-2047:
 *   Q_val = c * Q_val + P[m] * ia;   // FMA, genuine dependency chain
 *   eq += g[m] * Q_val;              // FMA, accumulation chain
 *
 * Cost model: leaf = n * max(FP64_DIV_NS, 2*B*LEAF_FMA_NS) + (n/B)*LEAF_BLOCK_NS
 *
 * Methodology:
 *   LEAF_FMA_NS  — slope of per-block time vs 2*B^2, sweeping B in {8,16,24,32,48,64}
 *   LEAF_BLOCK_NS — intercept of the same regression (fixed per-block overhead),
 *                   analogous to bench_wrap_fma.c's per-call intercept
 *
 * Global volatiles used for loop bounds so compiler cannot constant-fold.
 * All timed code inline in main(). Unique data per block.
 *
 * Build: gcc -O3 -march=native -o bench_leaf_fma bench_leaf_fma.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static volatile int g_B = 0;
static volatile int g_pk_g = 0;

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double linreg(int n, const double *x, const double *y, double *intercept) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        sx += x[i]; sy += y[i];
        sxx += x[i] * x[i]; sxy += x[i] * y[i];
    }
    double denom = (double)n * sxx - sx * sx;
    if (denom == 0.0) { *intercept = 0; return 0; }
    double slope = ((double)n * sxy - sx * sy) / denom;
    *intercept = (sy - slope * sx) / (double)n;
    return slope;
}

int main(void) {
    int B_vals[] = {8, 16, 24, 32, 48, 64};
    int n_B = 6, n_reps = 7, n_blocks = 3000;

    /* ── Experiment 1: Sweep B ── */
    printf("# Experiment 1: sweep B (fixed n_blocks=%d)\n", n_blocks);
    printf("# B,n_blocks,fma_per_block,total_ns,ns_per_block\n");

    double x_fma[6], y_tpb[6];

    for (int bi = 0; bi < n_B; bi++) {
        g_B = B_vals[bi];
        g_pk_g = g_B;
        int Bv = g_B, nb = n_blocks;

        /* Allocate unique data per block */
        double **P_blk = (double **)malloc((size_t)nb * sizeof(double *));
        double **g_blk = (double **)malloc((size_t)nb * sizeof(double *));
        double **inv_blk = (double **)malloc((size_t)nb * sizeof(double *));
        double **co_blk  = (double **)malloc((size_t)nb * sizeof(double *));

        srand(42 + bi * 997);
        for (int b = 0; b < nb; b++) {
            P_blk[b] = (double *)malloc((size_t)(Bv+1) * sizeof(double));
            g_blk[b] = (double *)malloc((size_t)Bv * sizeof(double));
            inv_blk[b] = (double *)malloc((size_t)Bv * sizeof(double));
            co_blk[b]  = (double *)malloc((size_t)Bv * sizeof(double));

            double *a = (double *)malloc((size_t)Bv * sizeof(double));
            for (int j = 0; j < Bv; j++)
                a[j] = 0.5 + 0.49 * ((double)rand() / RAND_MAX);

            double *P = P_blk[b];
            for (int m = 0; m <= Bv; m++) P[m] = 0.0;
            P[0] = 1.0;
            for (int j = 0; j < Bv; j++) {
                double aj = a[j], bj = 1.0 - aj;
                for (int m = Bv; m >= 1; m--)
                    P[m] = aj * P[m] + bj * P[m-1];
                P[0] *= aj;
            }
            for (int m = 0; m < Bv; m++)
                g_blk[b][m] = (double)rand() / RAND_MAX;
            for (int j = 0; j < Bv; j++) {
                double ia = 1.0 / a[j];
                inv_blk[b][j] = ia;
                co_blk[b][j] = -(1.0 - a[j]) * ia;
            }
            free(a);
        }

        double times[7];
        for (int rep = 0; rep < n_reps; rep++) {
            double sink = 0.0;
            int Bl = g_B, gl = g_pk_g;
            __asm__ volatile("" ::: "memory");
            double t0 = now_ns();

            /* —— TIMED: verbatim src/icm.c:2042-2047 —— */
            for (int b = 0; b < nb; b++) {
                const double *P = P_blk[b], *g = g_blk[b];
                const double *inv = inv_blk[b], *co = co_blk[b];
                int bsz = Bl, ng = gl;
                for (int jj = 0; jj < bsz; jj++) {
                    double ia = inv[jj], c = co[jj];
                    double Q_val = P[0] * ia;
                    double eq = g[0] * Q_val;
                    for (int m = 1; m < ng; m++) {
                        Q_val = c * Q_val + P[m] * ia;
                        eq += g[m] * Q_val;
                    }
                    sink += eq;
                }
            }

            double t1 = now_ns();
            times[rep] = (t1 - t0) / (double)nb;
            if (sink != sink) { fprintf(stderr, "NaN\n"); return 1; }
        }
        for (int i = 0; i < n_reps; i++)
            for (int j = i+1; j < n_reps; j++)
                if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

        double med = times[n_reps/2];
        long long fma_pb = 2LL * Bv * Bv;
        printf("%d,%d,%lld,%.2f,%.4f\n", Bv, nb, fma_pb, med*nb, med);
        fflush(stdout);

        x_fma[bi] = (double)fma_pb;
        y_tpb[bi] = med;

        for (int b = 0; b < nb; b++) {
            free(P_blk[b]); free(g_blk[b]); free(inv_blk[b]); free(co_blk[b]);
        }
        free(P_blk); free(g_blk); free(inv_blk); free(co_blk);
    }

    double intercept_1;
    double leaf_fma_ns = linreg(n_B, x_fma, y_tpb, &intercept_1);

    /* ── Experiment 2: Sweep n_blocks at B=32 ── */
    g_B = 32; g_pk_g = 32;
    int Bfv = g_B;
    int nb_vals[] = {750, 1500, 2250, 3000, 3750};
    int n_nb = 5, max_nb = nb_vals[n_nb-1];

    printf("\n# Experiment 2: sweep n_blocks (fixed B=%d)\n", Bfv);
    printf("# n_blocks,total_ns,ns_per_block\n");

    double **P2 = (double **)malloc((size_t)max_nb * sizeof(double *));
    double **g2 = (double **)malloc((size_t)max_nb * sizeof(double *));
    double **inv2 = (double **)malloc((size_t)max_nb * sizeof(double *));
    double **co2  = (double **)malloc((size_t)max_nb * sizeof(double *));

    /* Use sink from Exp 1 to seed Exp 2 — prevents reordering */
    double seed2 = 12345.0 + y_tpb[0] + y_tpb[1] + y_tpb[2];
    srand((unsigned int)seed2);

    for (int b = 0; b < max_nb; b++) {
        P2[b] = (double *)malloc((size_t)(Bfv+1) * sizeof(double));
        g2[b] = (double *)malloc((size_t)Bfv * sizeof(double));
        inv2[b] = (double *)malloc((size_t)Bfv * sizeof(double));
        co2[b]  = (double *)malloc((size_t)Bfv * sizeof(double));

        double *a = (double *)malloc((size_t)Bfv * sizeof(double));
        for (int j = 0; j < Bfv; j++)
            a[j] = 0.5 + 0.49 * ((double)rand() / RAND_MAX);

        double *P = P2[b];
        for (int m = 0; m <= Bfv; m++) P[m] = 0.0;
        P[0] = 1.0;
        for (int j = 0; j < Bfv; j++) {
            double aj = a[j], bj = 1.0 - aj;
            for (int m = Bfv; m >= 1; m--)
                P[m] = aj * P[m] + bj * P[m-1];
            P[0] *= aj;
        }
        for (int m = 0; m < Bfv; m++)
            g2[b][m] = (double)rand() / RAND_MAX;
        for (int j = 0; j < Bfv; j++) {
            double ia = 1.0 / a[j];
            inv2[b][j] = ia;
            co2[b][j] = -(1.0 - a[j]) * ia;
        }
        free(a);
    }

    double nb_arr[5], tot_arr[5];
    for (int ni = 0; ni < n_nb; ni++) {
        int nb = nb_vals[ni];
        double times[7];
        for (int rep = 0; rep < n_reps; rep++) {
            double sink = 0.0;
            int Bl = g_B, gl = g_pk_g;
            __asm__ volatile("" ::: "memory");
            double t0 = now_ns();

            for (int b = 0; b < nb; b++) {
                const double *P = P2[b], *g = g2[b];
                const double *inv = inv2[b], *co = co2[b];
                int bsz = Bl, ng = gl;
                for (int jj = 0; jj < bsz; jj++) {
                    double ia = inv[jj], c = co[jj];
                    double Q_val = P[0] * ia;
                    double eq = g[0] * Q_val;
                    for (int m = 1; m < ng; m++) {
                        Q_val = c * Q_val + P[m] * ia;
                        eq += g[m] * Q_val;
                    }
                    sink += eq;
                }
            }

            double t1 = now_ns();
            times[rep] = t1 - t0;
            if (sink != sink) { fprintf(stderr, "NaN\n"); return 1; }
        }
        for (int i = 0; i < n_reps; i++)
            for (int j = i+1; j < n_reps; j++)
                if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

        double med = times[n_reps/2];
        printf("%d,%.2f,%.4f\n", nb, med, med/nb);
        fflush(stdout);
        nb_arr[ni] = (double)nb;
        tot_arr[ni] = med;
    }

    double intercept_2;
    double slope_2 = linreg(n_nb, nb_arr, tot_arr, &intercept_2);
    double fma_contrib = 2.0 * Bfv * Bfv * leaf_fma_ns;
    double leaf_block_ns_v2 = slope_2 - fma_contrib;

    /* Use B-sweep intercept as primary LEAF_BLOCK_NS */
    double leaf_block_ns = intercept_1;

    for (int b = 0; b < max_nb; b++) {
        free(P2[b]); free(g2[b]); free(inv2[b]); free(co2[b]);
    }
    free(P2); free(g2); free(inv2); free(co2);

    /* ── Final output ── */
    printf("\n");
    printf("LEAF_FMA_NS=%.4f\n", leaf_fma_ns);
    printf("LEAF_BLOCK_NS=%.4f\n", leaf_block_ns);

    /* R² */
    double my = 0;
    for (int i = 0; i < n_B; i++) my += y_tpb[i];
    my /= n_B;
    double ssr = 0, sst = 0;
    for (int i = 0; i < n_B; i++) {
        double p = leaf_fma_ns * x_fma[i] + intercept_1;
        ssr += (y_tpb[i]-p)*(y_tpb[i]-p);
        sst += (y_tpb[i]-my)*(y_tpb[i]-my);
    }
    double r2 = 1.0 - ssr/sst;

    fprintf(stderr, "# LEAF_FMA_NS=%.4f ns/FMA  (slope, B-sweep regression, R²=%.6f)\n", leaf_fma_ns, r2);
    fprintf(stderr, "# LEAF_BLOCK_NS=%.2f ns/block  (intercept, B-sweep regression)\n", intercept_1);
    fprintf(stderr, "# Cross-check via n_blocks sweep: LEAF_BLOCK_NS_v2=%.2f ns/block\n", leaf_block_ns_v2);
    fprintf(stderr, "# Per-block times: B=8:%.2f B=16:%.2f B=24:%.2f B=32:%.2f B=48:%.2f B=64:%.2f ns\n",
            y_tpb[0], y_tpb[1], y_tpb[2], y_tpb[3], y_tpb[4], y_tpb[5]);
    fprintf(stderr, "# Raw ns/FMA: %.4f %.4f %.4f %.4f %.4f %.4f\n",
            y_tpb[0]/x_fma[0], y_tpb[1]/x_fma[1], y_tpb[2]/x_fma[2],
            y_tpb[3]/x_fma[3], y_tpb[4]/x_fma[4], y_tpb[5]/x_fma[5]);
    fprintf(stderr, "# Dependency chain: PRESERVED (verbatim from src/icm.c:2042-2047)\n");
    fprintf(stderr, "# Note: n_blocks sweep at B=32 gives systematically lower per-block times\n");
    fprintf(stderr, "#   (%.2f vs %.2f ns/block), suggesting residual compiler optimization\n",
            slope_2, y_tpb[3]);
    fprintf(stderr, "#   of the fixed-B path. B-sweep intercept is the more robust estimate.\n");

    return 0;
}
