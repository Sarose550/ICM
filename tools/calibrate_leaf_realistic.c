/* calibrate_leaf_realistic.c — Recalibrates leaf_fma_ns_per_player[] from
 * REAL embedded hybrid-engine execution on realistic data, replacing
 * bench_leaf_fma.c's isolated-microbenchmark table.
 *
 * Root cause this fixes: bench_leaf_fma.c's synthetic a[j] values were drawn
 * uniformly from [0.5, 0.99], forcing EVERY player through the expensive
 * forward-divide branch (real division + serial recurrence chain). Under
 * realistic conditions (S in [100,10000], quadrature logv sweep matching
 * production usage), measured with tools/count_paths-equivalent instrumentation:
 * ~99.9% of players actually take the cheap "zero" branch in
 * engine_hybrid_core's leaf-divide step (aj underflows below 1e-15, so the
 * code executes a plain FMA accumulate with NO division and NO dependency
 * chain — see src/icm.c line ~2068). The isolated benchmark never exercised
 * this branch at all, which is why it overpredicted real leaf cost by ~2x on
 * M3 Pro (see HANDOFF.md, DISPATCH_GAP_ANALYSIS.md).
 *
 * This tool mirrors probe_leaf_extract.c's methodology (real
 * engine_hybrid_core-equivalent execution, median of N_REPS) but sweeps
 * across all 6 candidate B values at a fixed large n, using the SAME
 * S-generation as every other calibration/benchmark tool in this project
 * (S ~ 100 + 9900*U[0,1]), so the branch mix matches production.
 *
 * Build (macOS M3 Pro):
 *   gcc -O3 -march=native -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o build/calibrate_leaf_realistic tools/calibrate_leaf_realistic.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 */
#ifdef __APPLE__
#include <pthread.h>
#endif
#include "icm.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) ? 1 : (da < db) ? -1 : 0;
}

#define N_REPS 21
#define Q_PROBE 256
#define N_CAL 8192   /* large n: block-overhead per player is well amortized */

/* Times ONLY the leaf-divide phase (block build + tree already primed). */
static double leaf_ns_for_B(int B) {
    int n = N_CAL, k = n; /* k=n: g_need spans full range, representative */
    double *S = (double *)malloc(n * sizeof(double));
    double *payout = (double *)malloc(k * sizeof(double));
    srand(42);
    for (int i = 0; i < n; i++)
        S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
    for (int q = 0; q < k; q++)
        payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

    HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
    if (!hc) { fprintf(stderr, "hc failed for B=%d\n", B); exit(1); }

    int nblocks = hc->nblocks;
    TreeCtx *tc = hc->tc;
    int leaf_psz = tc->psz[0];
    double *plev_data = tc->ws;
    double *a_qp = (double *)malloc(n * sizeof(double));
    double *a_sorted = (double *)malloc(n * sizeof(double));
    double *inner_sorted = (double *)malloc(n * sizeof(double));
    const int *perm = hc->sort_perm;

    double leaf_samples[N_REPS];

    for (int rep = 0; rep < N_REPS; rep++) {
        double total_leaf_ns = 0;

        for (int qp = 0; qp < Q_PROBE; qp++) {
            double logv = -(((double)qp + 0.5) / (double)Q_PROBE) * 10.0;
            for (int i = 0; i < n; i++) {
                double arg = S[i] * logv;
                a_qp[i] = (arg < -700) ? 0.0 : exp(arg);
            }
            for (int i = 0; i < n; i++) a_sorted[i] = a_qp[perm[i]];

            memset(plev_data, 0, tc->ws_size * sizeof(double));

            /* Block build (untimed — we only want leaf cost) */
            int leaf_psz0 = tc->psz[0];
            for (int b = 0; b < nblocks; b++) {
                int start = b * B, end = start + B;
                if (end > n) end = n;
                int bsize = end - start;
                double *P = hc->block_prods + (size_t)b * (B + 1);
                memset(P, 0, (B + 1) * sizeof(double));
                P[0] = 1.0;
                for (int j = start; j < end; j++) {
                    double aj = a_sorted[j], bj = 1 - aj;
                    for (int m = bsize; m >= 1; m--)
                        P[m] = aj * P[m] + bj * P[m - 1];
                    P[0] *= aj;
                }
                double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz0;
                int cp = (B + 1 < leaf_psz0) ? B + 1 : leaf_psz0;
                memcpy(leaf, P, cp * sizeof(double));
                if (cp < leaf_psz0) memset(leaf + cp, 0, (leaf_psz0 - cp) * sizeof(double));
            }
            for (int b = nblocks; b < tc->N; b++) {
                double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz0;
                memset(leaf, 0, leaf_psz0 * sizeof(double));
                leaf[0] = 1.0;
            }

            tree_build_levels(tc);
            double *g_leaf = tree_propagate_g(tc, k, payout, hc->hot_mask);
            int g_need = tc->g_needed[0];

            /* ── Leaf divide: TIMED, verbatim engine_hybrid_core Step 3 ── */
            double t0 = now_ns();
            for (int b = 0; b < nblocks; b++) {
                int start = b * B, end = start + B;
                if (end > n) end = n;
                int bsize = end - start;

                double *P_b = hc->block_prods + (size_t)b * (B + 1);
                double *g_b = g_leaf + (size_t)b * leaf_psz;
                int pk_g = g_need < bsize ? g_need : bsize;
                if (pk_g > k) pk_g = k;

                double inv_arr[bsize], coeff_arr[bsize];
                int fwd_arr[bsize];
                for (int j = 0; j < bsize; j++) {
                    double aj = a_sorted[start + j], bj_val = 1 - aj;
                    if (aj > 0.5) {
                        double ia = 1.0 / aj;
                        inv_arr[j] = ia;
                        coeff_arr[j] = -bj_val * ia;
                        fwd_arr[j] = 1;
                    } else if (aj > 1e-15) {
                        double ib = 1.0 / bj_val;
                        inv_arr[j] = ib;
                        coeff_arr[j] = -aj * ib;
                        fwd_arr[j] = 0;
                    } else {
                        inv_arr[j] = 0;
                        coeff_arr[j] = 0;
                        fwd_arr[j] = -1;
                    }
                }

                for (int jj = 0; jj < bsize; jj++) {
                    double eq = 0;
                    if (fwd_arr[jj] == 1) {
                        double ia = inv_arr[jj], c = coeff_arr[jj];
                        double Q_val = P_b[0] * ia;
                        eq = g_b[0] * Q_val;
                        for (int m = 1; m < pk_g; m++) {
                            Q_val = c * Q_val + P_b[m] * ia;
                            eq += g_b[m] * Q_val;
                        }
                    } else if (fwd_arr[jj] == 0) {
                        double ib = inv_arr[jj], c = coeff_arr[jj];
                        double Q_prev = P_b[bsize] * ib;
                        double Q_arr[bsize];
                        Q_arr[bsize - 1] = Q_prev;
                        for (int m = bsize - 2; m >= 0; m--) {
                            Q_prev = c * Q_prev + P_b[m + 1] * ib;
                            Q_arr[m] = Q_prev;
                        }
                        for (int m = 0; m < pk_g; m++)
                            eq += g_b[m] * Q_arr[m];
                    } else {
                        for (int m = 0; m < pk_g; m++)
                            eq += g_b[m] * P_b[m + 1];
                    }
                    inner_sorted[start + jj] = eq;
                }
            }
            double t1 = now_ns();
            total_leaf_ns += (t1 - t0);
        }
        leaf_samples[rep] = total_leaf_ns / Q_PROBE;
    }

    qsort(leaf_samples, N_REPS, sizeof(double), cmp_double);
    double med_leaf_ns = leaf_samples[N_REPS / 2];

    hybrid_ctx_destroy(hc);
    free(S); free(payout); free(a_qp); free(a_sorted); free(inner_sorted);

    return med_leaf_ns / (double)n; /* ns per player */
}

int main(void) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    build_fftw_size_table();
    icm_init(NULL);

    int B_vals[] = {8, 16, 24, 32, 48, 64};
    printf("# Realistic-data leaf calibration (n=%d, Q=%d, %d reps median)\n",
           N_CAL, Q_PROBE, N_REPS);
    printf("# Uses real hybrid engine with backward pass, not synthetic forward-only data\n");
    printf("# B,ns_per_player\n");

    printf("\nLEAF_FMA_NS_PER_PLAYER_TABLE\n");
    for (int i = 0; i < 6; i++) {
        double ns = leaf_ns_for_B(B_vals[i]);
        printf("B=%d,%.4f\n", B_vals[i], ns);
        fflush(stdout);
    }
    return 0;
}
