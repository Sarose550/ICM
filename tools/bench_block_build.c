/* bench_block_build.c — isolated microbenchmark for the block-build phase.
 *
 * Directly measures BLOCK_FMA_NS and BLOCK_MEM_NS by timing ONLY the
 * block-build inner loop (verbatim copy of src/icm.c's engine_hybrid_core
 * block-build section), sweeping the block size B across the engine's
 * actual candidate list {8,16,24,32,48,64} plus edge points for a cleaner
 * regression.
 *
 * The block-build phase builds the per-block polynomial product
 *   P(x) = ∏_{j in block} (a_j·x + (1-a_j))
 * via a nested loop: for each player, bsize FMAs update the coefficient
 * array P[0..B].  The cost-model estimate (src/icm.c lines 2178, 2526-2528):
 *   block_build = n * ((B+1)/2 * BLOCK_FMA_NS + BLOCK_MEM_NS)
 * models this as a per-player FMA term scaling with (B+1)/2 plus a fixed
 * per-player memory-streaming cost.
 *
 * The real inner loop's per-player FMA count is bsize (= B for full blocks),
 * not (B+1)/2.  The model uses (B+1)/2 as its independent variable, so the
 * fitted BLOCK_FMA_NS will absorb that factor-of-~2 discrepancy — it is a
 * model coefficient, not a literal per-FMA-instruction latency.
 *
 * Dependency analysis: within a single player, the m-loop iterations are
 * independent (all read P[m],P[m-1] from the *previous* player's output),
 * so the inner loop is throughput-bound (vectorizable).  Between players,
 * there IS a genuine sequential dependency — player j+1 reads all P[m]
 * that player j just wrote.  This microbenchmark preserves the real loop
 * structure exactly, so it measures the genuine mix of throughput+latency
 * that the real engine experiences.
 *
 * Build: gcc -O3 -march=native -o bench_block_build bench_block_build.c
 *
 * Output: CSV on stdout with raw sweep data, followed by:
 *   BLOCK_FMA_NS=<value>
 *   BLOCK_MEM_NS=<value>
 *   R2=<value>
 *
 * Then a clean lookup table (using only the 6 candidate B values
 * {8,16,24,32,48,64}, averaged over nblocks runs):
 *   BLOCK_BUILD_NS_PER_PLAYER_TABLE
 *   B=8,<value>
 *   B=16,<value>
 *   B=24,<value>
 *   B=32,<value>
 *   B=48,<value>
 *   B=64,<value>
 * This table is parsed by later tooling to populate the per-device
 * block_build_ns_per_player[] array in fft_config.h.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* ── Verbatim block-build for one block ─────────────────────────────
 *
 * This is the EXACT loop body from engine_hybrid_core() in src/icm.c,
 * lines ~1963-1987 (the "Steps 1+2 fused" comment block).  The only
 * adaptation is that P, a, and leaf are passed as parameters instead of
 * being indexed from the HybridCtx struct — the computation is identical.
 *
 * Includes:
 *   - memset(P, …) + P[0] = 1.0  (per-block init)
 *   - per-player nested loop: bsize FMAs + 1 multiply
 *   - memcpy to tree leaf + zero-pad (per-block writeout)
 *
 * Returns the total number of FMAs executed (for diagnostic output).
 */
static long long block_build_one(double *P,
                                  const double *a,
                                  int start, int end,
                                  int B, int leaf_psz,
                                  double *leaf) {
    int bsize = end - start;
    long long fma_count = 0;

    /* Per-block init (verbatim) */
    memset(P, 0, (size_t)(B + 1) * sizeof(double));
    P[0] = 1.0;

    /* Per-player polynomial multiply (verbatim) */
    for (int j = start; j < end; j++) {
        double aj = a[j], bj = 1 - aj;
        for (int m = bsize; m >= 1; m--) {
            P[m] = aj * P[m] + bj * P[m - 1];
            fma_count++;
        }
        P[0] *= aj;
    }

    /* Copy truncated version into tree leaf (verbatim) */
    int cp = (B + 1 < leaf_psz) ? B + 1 : leaf_psz;
    memcpy(leaf, P, (size_t)cp * sizeof(double));
    if (cp < leaf_psz)
        memset(leaf + cp, 0,
               (size_t)(leaf_psz - cp) * sizeof(double));

    return fma_count;
}

/* ── Time one full build of nblocks blocks ───────────────────────── */
static double time_block_build(int B, int n, int nblocks,
                                const double *a,
                                double *P_buf,
                                double *leaf_buf,
                                int leaf_psz,
                                int reps,
                                long long *out_fma_total,
                                double *sink) {
    double t0 = now_ns();
    long long fma_total = 0;
    double local_sink = 0.0;

    for (int r = 0; r < reps; r++) {
        for (int b = 0; b < nblocks; b++) {
            int start = b * B;
            int end = start + B;
            if (end > n) end = n;

            double *P   = P_buf   + (size_t)b * (B + 1);
            double *leaf = leaf_buf + (size_t)b * leaf_psz;

            fma_total += block_build_one(P, a, start, end,
                                          B, leaf_psz, leaf);
        }
        /* Leak a little data into sink so the compiler can't
         * optimize away the whole loop. */
        local_sink += P_buf[0] + leaf_buf[0];
    }

    double t1 = now_ns();
    *out_fma_total = fma_total;
    *sink = local_sink;
    return (t1 - t0) / (double)reps;
}

/* ── Simple 2-parameter least-squares ────────────────────────────── */
static void fit_linear(int ndata,
                       const double *x, const double *y,
                       double *slope, double *intercept, double *r2) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (int i = 0; i < ndata; i++) {
        sx  += x[i];
        sy  += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
        syy += y[i] * y[i];
    }
    double denom = (double)ndata * sxx - sx * sx;
    if (fabs(denom) < 1e-30) {
        *slope = *intercept = *r2 = 0.0;
        return;
    }
    *slope     = ((double)ndata * sxy - sx * sy) / denom;
    *intercept = (sy - (*slope) * sx) / (double)ndata;

    /* R² */
    double ymean = sy / (double)ndata;
    double ss_res = 0, ss_tot = 0;
    for (int i = 0; i < ndata; i++) {
        double ypred = (*slope) * x[i] + (*intercept);
        ss_res += (y[i] - ypred) * (y[i] - ypred);
        ss_tot += (y[i] - ymean) * (y[i] - ymean);
    }
    *r2 = (ss_tot > 1e-30) ? 1.0 - ss_res / ss_tot : 0.0;
}

int main(void) {
    /* ── Sweep parameters ──────────────────────────────────────────
     * B: block sizes spanning the engine's candidate list plus edges.
     * nblocks_list: number of full blocks per measurement.
     *   n = B * nblocks gives the total player count.
     *
     * Using two nblocks values checks that per-player cost is
     * independent of block count (i.e., per-block overhead is either
     * negligible or correctly absorbed into the per-player model).
     */
    int B_vals[]       = {4, 8, 16, 24, 32, 48, 64, 80};
    int nB             = (int)(sizeof(B_vals) / sizeof(B_vals[0]));
    int nblocks_vals[] = {16, 32};
    int n_nb           = (int)(sizeof(nblocks_vals) / sizeof(nblocks_vals[0]));

    /* Timing */
    int NTIMES = 9;   /* median-of-9 for noise suppression */

    /* ── Header ──────────────────────────────────────────────────── */
    printf("B,n,nblocks,Bplus1_over_2,ns_per_player,median_ns_per_block,"
           "fma_total,leaf_psz\n");

    /* ── Data arrays for final regression ─────────────────────────── */
    #define MAX_DATA 256
    double reg_x[MAX_DATA], reg_y[MAX_DATA];
    int ndata = 0;

    /* ── Per-B lookup table (6 candidate B values only) ──────────── */
    int table_B[6] = {8, 16, 24, 32, 48, 64};
    double table_sum[6] = {0.0};
    int table_count[6] = {0};

    double global_sink = 0.0;

    /* Seed once for reproducible a[] values across all sweeps */
    srand(42);

    for (int bi = 0; bi < nB; bi++) {
        int B = B_vals[bi];

        for (int nbi = 0; nbi < n_nb; nbi++) {
            int nblocks = nblocks_vals[nbi];
            int n = B * nblocks;   /* all blocks full */

            /* leaf_psz = min(2*B, k).  For the microbenchmark we
             * use 2*B (the typical case when k >= 2*B). */
            int leaf_psz = 2 * B;

            /* Allocate */
            double *a = (double *)malloc((size_t)n * sizeof(double));
            double *P_buf = (double *)calloc((size_t)nblocks * (B + 1),
                                              sizeof(double));
            double *leaf_buf = (double *)calloc((size_t)nblocks * leaf_psz,
                                                 sizeof(double));

            /* Fill a[] with deterministic values */
            for (int i = 0; i < n; i++)
                a[i] = 0.1 + 0.8 * ((double)rand() / (double)RAND_MAX);
                /* range (0.1, 0.9): realistic stack fractions */

            /* Choose reps so each timing run is ~30-100 ms.
             * Total FMAs per call ≈ nblocks * B * B = n * B.
             * At ~0.7 ns/FMA (throughput + overhead), n*B FMAs
             * ≈ n*B*0.7e-9 s. Target ~50 ms → reps ≈ 50e6/(n*B*0.7). */
            double est_ns_per_call = (double)n * (double)B * 0.7;
            int reps = (int)(50e6 / est_ns_per_call);
            if (reps < 10) reps = 10;
            if (reps > 500000) reps = 500000;

            /* ── Median-of-N timing ───────────────────────────── */
            double times[9];
            long long fma_total = 0;
            for (int rep = 0; rep < NTIMES; rep++) {
                double sink;
                double t = time_block_build(B, n, nblocks,
                                             a, P_buf, leaf_buf,
                                             leaf_psz, reps,
                                             &fma_total, &sink);
                global_sink += sink;
                times[rep] = t;
            }

            /* Sort for median */
            for (int i = 0; i < NTIMES; i++)
                for (int j = i + 1; j < NTIMES; j++)
                    if (times[j] < times[i]) {
                        double tmp = times[i];
                        times[i] = times[j];
                        times[j] = tmp;
                    }

            double median_ns_per_block = times[NTIMES / 2];
            double ns_per_player = median_ns_per_block / (double)n;

            printf("%d,%d,%d,%.1f,%.4f,%.4f,%lld,%d\n",
                   B, n, nblocks,
                   (double)(B + 1) / 2.0,
                   ns_per_player,
                   median_ns_per_block,
                   fma_total,
                   leaf_psz);
            fflush(stdout);

            /* Accumulate for per-B lookup table (candidate B values only) */
            for (int ti = 0; ti < 6; ti++) {
                if (B == table_B[ti]) {
                    table_sum[ti] += ns_per_player;
                    table_count[ti]++;
                    break;
                }
            }

            /* Store for final regression */
            if (ndata < MAX_DATA) {
                reg_x[ndata] = (double)(B + 1) / 2.0;
                reg_y[ndata] = ns_per_player;
                ndata++;
            }

            free(a);
            free(P_buf);
            free(leaf_buf);
        }
    }

    /* ── Fit: ns_per_player = BLOCK_FMA_NS * (B+1)/2 + BLOCK_MEM_NS ─ */
    double slope, intercept, r2;
    fit_linear(ndata, reg_x, reg_y, &slope, &intercept, &r2);

    fprintf(stderr, "sink_guard(ignore)=%.6f\n", global_sink);

    printf("\n");
    printf("BLOCK_FMA_NS=%.4f\n", slope);
    printf("BLOCK_MEM_NS=%.4f\n", intercept);
    printf("R2=%.6f\n", r2);

    /* ── Emit per-B lookup table (candidate B values only) ──────── */
    printf("\n");
    printf("BLOCK_BUILD_NS_PER_PLAYER_TABLE\n");
    for (int ti = 0; ti < 6; ti++) {
        double avg = (table_count[ti] > 0)
                     ? table_sum[ti] / (double)table_count[ti] : 0.0;
        printf("B=%d,%.4f\n", table_B[ti], avg);
    }

    return 0;
}
