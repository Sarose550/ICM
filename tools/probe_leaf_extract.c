/* probe_leaf_extract.c — Measure leaf-extraction cost embedded in real hybrid
 * engine runs, and compare against the cost-model prediction
 * (leaf_fma_ns_per_player[] + FP64_DIV_NS floor).
 *
 * Methodology: replicates engine_hybrid_core with timing splits at each phase
 * boundary — block_build, tree_build+propagate, leaf_divide. Runs Q=256 points,
 * median over N_REPS independent runs. This is the same rigor as
 * probe_tree_levels.c.
 *
 * Build (macOS M3 Pro):
 *   gcc -O3 -march=native -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o build/probe_leaf_extract tools/probe_leaf_extract.c \
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

/* Replicates engine_hybrid_core exactly, but times each phase independently.
 * Returns per-phase times (total ns for all Q points). */
static void engine_hybrid_core_timed(int n, const double *a,
                                      const double *payout, int k,
                                      double *inner, HybridCtx *hc,
                                      double *out_block_ns,
                                      double *out_tree_ns,
                                      double *out_leaf_ns) {
    int B = hc->B;
    int nblocks = hc->nblocks;
    TreeCtx *tc = hc->tc;
    int N = tc->N;
    int *psz = tc->psz;
    double *plev_data = tc->ws;

    /* ── Block build ── */
    double t0 = now_ns();

    int leaf_psz = psz[0];
    for (int b = 0; b < nblocks; b++) {
        int start = b * B, end = start + B;
        if (end > n) end = n;
        int bsize = end - start;
        double *P = hc->block_prods + (size_t)b * (B + 1);
        memset(P, 0, (B + 1) * sizeof(double));
        P[0] = 1.0;
        for (int j = start; j < end; j++) {
            double aj = a[j], bj = 1 - aj;
            for (int m = bsize; m >= 1; m--)
                P[m] = aj * P[m] + bj * P[m - 1];
            P[0] *= aj;
        }
        double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz;
        int cp = (B + 1 < leaf_psz) ? B + 1 : leaf_psz;
        memcpy(leaf, P, cp * sizeof(double));
        if (cp < leaf_psz) memset(leaf + cp, 0, (leaf_psz - cp) * sizeof(double));
    }
    for (int b = nblocks; b < N; b++) {
        double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz;
        memset(leaf, 0, leaf_psz * sizeof(double));
        leaf[0] = 1.0;
    }

    double t1 = now_ns();
    *out_block_ns = t1 - t0;

    /* ── Tree build + propagate ── */
    tree_build_levels(tc);
    double *g_leaf = tree_propagate_g(tc, k, payout, hc->hot_mask);

    double t2 = now_ns();
    *out_tree_ns = t2 - t1;

    int g_need = tc->g_needed[0];

    /* ── Leaf divide ── */
    const uint8_t *active = hc->active;
    for (int b = 0; b < nblocks; b++) {
        int start = b * B, end = start + B;
        if (end > n) end = n;
        int bsize = end - start;

        if (active) {
            int any = 0;
            for (int j = start; j < end; j++) if (active[j]) { any = 1; break; }
            if (!any) continue;
        }

        double *P_b = hc->block_prods + (size_t)b * (B + 1);
        double *g_b = g_leaf + (size_t)b * leaf_psz;
        int pk_g = g_need < bsize ? g_need : bsize;
        if (pk_g > k) pk_g = k;

        double inv_arr[bsize], coeff_arr[bsize];
        int fwd_arr[bsize];
        for (int j = 0; j < bsize; j++) {
            double aj = a[start + j], bj_val = 1 - aj;
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
            if (active && !active[start + jj]) { inner[start + jj] = 0; continue; }
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
            inner[start + jj] = eq;
        }
    }

    double t3 = now_ns();
    *out_leaf_ns = t3 - t2;
}

/* Wrapper matching EquityEngine signature, used with engine_hybrid_ctx's
 * sorted/unsorted logic via run_engine_ctx. We do the sorted approach
 * manually for the timed version. */
static void probe_phases(int n, const double *S,
                          const double *payout, int k,
                          HybridCtx *hc,
                          double *block_ns,
                          double *tree_ns,
                          double *leaf_ns) {
    double *a_qp = (double *)malloc(n * sizeof(double));
    double *inner = (double *)malloc(n * sizeof(double));
    double *equity = (double *)malloc(n * sizeof(double));
    if (!a_qp || !inner || !equity) { fprintf(stderr, "OOM\n"); exit(1); }

    double total_block = 0, total_tree = 0, total_leaf = 0;

    for (int qp = 0; qp < Q_PROBE; qp++) {
        double logv = ((double)qp + 0.5) / (double)Q_PROBE;
        logv = -logv * 10.0;

        for (int i = 0; i < n; i++) {
            double arg = S[i] * logv;
            a_qp[i] = (arg < -700) ? 0.0 : exp(arg);
        }

        /* Sort a_qp to stack-descending order matching engine_hybrid_ctx */
        double *a_sorted = (double *)malloc(n * sizeof(double));
        double *inner_sorted = (double *)malloc(n * sizeof(double));
        if (!a_sorted || !inner_sorted) { fprintf(stderr, "OOM\n"); exit(1); }

        const int *perm = hc->sort_perm;
        for (int i = 0; i < n; i++) {
            a_sorted[i] = a_qp[perm[i]];
        }

        /* Must reset workspace between QPs (tree_build_levels and propagate
         * mutate tc->ws). We just clear it fully. */
        memset(hc->tc->ws, 0, hc->tc->ws_size * sizeof(double));

        double b_ns, t_ns, l_ns;
        engine_hybrid_core_timed(n, a_sorted, payout, k, inner_sorted, hc,
                                  &b_ns, &t_ns, &l_ns);
        total_block += b_ns;
        total_tree += t_ns;
        total_leaf += l_ns;

        /* Unpermute (reverse of engine_hybrid_ctx's forward permute, same perm[]) */
        for (int i = 0; i < n; i++)
            inner[perm[i]] = inner_sorted[i];

        free(a_sorted);
        free(inner_sorted);
    }

    /* Accumulate equity (not strictly needed for timing, but we do it for
     * correctness) */
    for (int i = 0; i < n; i++) equity[i] = 0;
    /* The quadrature integration is done by run_engine_ctx; here we just
     * measure phases. The timing is per-QP sum, not integrated. */

    *block_ns = total_block;
    *tree_ns = total_tree;
    *leaf_ns = total_leaf;

    free(a_qp); free(inner); free(equity);
}

int main(void) {
#ifdef __APPLE__
    /* Pin to P-cores: without this, the scheduler can silently place this
     * thread on an E-core (half a P-core's FP throughput) under any
     * contention, corrupting the measurement with no indication in the
     * tool's own output. See HANDOFF.md's QoS-pinning hypothesis. */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    build_fftw_size_table();
    icm_init(NULL);

    int B = 8;

    int n_vals[] = {512, 1024, 2048, 4096, 8192};
    int n_n = 5;
    int k_vals[] = {40, 80, 120, 160, 200, 260, 320, 400};
    int n_k = 8;

    printf("=== PHASE-BY-PHASE TIMING (B=%d, Q=%d, %d reps median) ===\n\n",
           B, Q_PROBE, N_REPS);
    printf("%-6s %-6s %12s %12s %12s %14s %14s %14s %8s %8s %8s\n",
           "n", "k", "block/qp", "tree/qp", "leaf/qp",
           "leaf_pred/qp", "block_pred/qp", "total_pred/qp",
           "l_ratio", "b_ratio", "t_ratio");

    double leaf_sum_log = 0, leaf_sum_log2 = 0;
    double block_sum_log = 0, block_sum_log2 = 0;
    int n_rows = 0;
    double leaf_best = 1e9, leaf_worst = 0;
    double block_best = 1e9, block_worst = 0;

    for (int ni = 0; ni < n_n; ni++) {
        int n = n_vals[ni];
        for (int ki = 0; ki < n_k; ki++) {
            int k = k_vals[ki];
            if (k > n) continue;
            if (B > k || B > n) continue;

            fprintf(stderr, "Probing n=%d k=%d B=%d...\n", n, k, B);

            double *S = (double *)malloc(n * sizeof(double));
            double *payout = (double *)malloc(k * sizeof(double));
            if (!S || !payout) { fprintf(stderr, "OOM\n"); return 1; }
            srand(42);
            for (int i = 0; i < n; i++)
                S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
            for (int q = 0; q < k; q++)
                payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

            double leaf_samples[N_REPS], block_samples[N_REPS], tree_samples[N_REPS];

            for (int rep = 0; rep < N_REPS; rep++) {
                HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
                if (!hc) { fprintf(stderr, "hc failed\n"); return 1; }

                double block_total, tree_total, leaf_total;
                probe_phases(n, S, payout, k, hc,
                             &block_total, &tree_total, &leaf_total);

                block_samples[rep] = block_total / Q_PROBE;
                tree_samples[rep] = tree_total / Q_PROBE;
                leaf_samples[rep] = leaf_total / Q_PROBE;

                hybrid_ctx_destroy(hc);
            }

            qsort(block_samples, N_REPS, sizeof(double), cmp_double);
            qsort(tree_samples, N_REPS, sizeof(double), cmp_double);
            qsort(leaf_samples, N_REPS, sizeof(double), cmp_double);

            double med_block = block_samples[N_REPS / 2];
            double med_tree = tree_samples[N_REPS / 2];
            double med_leaf = leaf_samples[N_REPS / 2];

            /* Model predictions */
            int bidx = B_to_table_index(B);
            double pred_block = (double)n * block_build_ns_per_player[bidx];
            double le_cost = (FP64_DIV_NS > leaf_fma_ns_per_player[bidx])
                             ? FP64_DIV_NS : leaf_fma_ns_per_player[bidx];
            double pred_leaf = (double)n * le_cost;

            /* Tree prediction from model */
            int nblocks = (n + B - 1) / B;
            TreeCtx *tc = tree_ctx_create_ex2(nblocks, B, k, B);
            double pred_tree = 0;
            for (int ell = 1; ell < tc->L - 1; ell++) {
                int cps = tc->psz[ell-1], nr = tc->n_real[ell];
                if (tc->use_fft[ell]) {
                    int bfn = tc->build_fft_n[ell];
                    int bwm = tc->build_wrap_m[ell];
                    int idx = 0;
                    { int lo=0,hi=N_CALIBRATED_SIZES-1;
                      while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<bfn)lo=m+1;else hi=m;}
                      idx=lo; }
                    double build_fft = calib_times_ns[idx] + FFT_OVERHEAD_NS
                                     + (double)bwm*(bwm+1)/2.0*FMA_NS;
                    double corr;
                    if (tc->fft_cache_ok[ell]) {
                        corr = calib_times_ns[idx] * PAIRED_CACHED_CORR_RATIO
                             + (double)tc->corr_wrap_m[ell]*(tc->corr_wrap_m[ell]+1)*FMA_NS;
                    } else {
                        int cfn = tc->corr_fft_n[ell];
                        int cwm = tc->corr_wrap_m[ell];
                        int cidx=0;
                        {int lo=0,hi=N_CALIBRATED_SIZES-1;
                         while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<cfn)lo=m+1;else hi=m;}
                         cidx=lo;}
                        corr = INDEP_PAIR_RATIO * calib_times_ns[cidx]
                             + (double)cwm*(cwm+1)*FMA_NS;
                    }
                    pred_tree += nr * (build_fft + corr);
                } else {
                    int idx;
                    { int lo=0,hi=N_CALIBRATED_SIZES-1;
                      while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<cps)lo=m+1;else hi=m;}
                      idx=lo; }
                    double s = schoolbook_mul_ns[idx];
                    double c = (double)cps * tc->g_needed[ell-1] * schoolbook_corr_ns[idx];
                    pred_tree += nr * (s + c);
                }
            }
            tree_ctx_destroy(tc);

            double pred_total = pred_block + pred_tree + pred_leaf;
            double l_ratio = (pred_leaf > 0) ? med_leaf / pred_leaf : 0;
            double b_ratio = (pred_block > 0) ? med_block / pred_block : 0;
            double t_ratio = (pred_total > 0) ? (med_block+med_tree+med_leaf) / pred_total : 0;

            printf("%-6d %-6d %12.1f %12.1f %12.1f %14.1f %14.1f %14.1f %8.3f %8.3f %8.3f\n",
                   n, k, med_block, med_tree, med_leaf,
                   pred_leaf, pred_block, pred_total,
                   l_ratio, b_ratio, t_ratio);

            if (l_ratio > 0.1) {
                double lr = log(l_ratio);
                leaf_sum_log += lr;
                leaf_sum_log2 += lr * lr;
                if (l_ratio < leaf_best) leaf_best = l_ratio;
                if (l_ratio > leaf_worst) leaf_worst = l_ratio;
            }
            if (b_ratio > 0.1) {
                double lr = log(b_ratio);
                block_sum_log += lr;
                block_sum_log2 += lr * lr;
                if (b_ratio < block_best) block_best = b_ratio;
                if (b_ratio > block_worst) block_worst = b_ratio;
            }
            n_rows++;

            free(S); free(payout);
        }
    }

    double leaf_geo = exp(leaf_sum_log / n_rows);
    double leaf_sd = sqrt(leaf_sum_log2/n_rows - (leaf_sum_log/n_rows)*(leaf_sum_log/n_rows));
    double block_geo = exp(block_sum_log / n_rows);
    double block_sd = sqrt(block_sum_log2/n_rows - (block_sum_log/n_rows)*(block_sum_log/n_rows));

    printf("\n=== LEAF EXTRACTION ===\n");
    printf("geo_mean(meas/pred) = %.3f\n", leaf_geo);
    printf("log-stddev          = %.3f\n", leaf_sd);
    printf("ratio range         = [%.3f, %.3f]\n", leaf_best, leaf_worst);
    printf("prediction: FP64_DIV_NS=%.4f, leaf_fma_ns_per_player[%d]=%.4f\n",
           FP64_DIV_NS, B_to_table_index(B), leaf_fma_ns_per_player[B_to_table_index(B)]);

    printf("\n=== BLOCK BUILD ===\n");
    printf("geo_mean(meas/pred) = %.3f\n", block_geo);
    printf("log-stddev          = %.3f\n", block_sd);
    printf("ratio range         = [%.3f, %.3f]\n", block_best, block_worst);
    printf("prediction: block_build_ns_per_player[%d]=%.4f\n",
           B_to_table_index(B), block_build_ns_per_player[B_to_table_index(B)]);

    return 0;
}
