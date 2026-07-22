/* quantify_dispatch_gap.c — Quantify remaining dispatch-point gap after schoolbook fix.
 *
 * Three-in-one diagnostic:
 *   PART 1: Generate sample_plans CSV for the crossover region (n,k,B=8)
 *   PART 2: Compute model-vs-measured ratios (same logic as eval_model_vs_plans.c)
 *   PART 3: Instrument leaf-extraction phase embedded in real hybrid runs,
 *           compare against leaf_fma_ns_per_player[] predictions
 *
 * Build (macOS M3 Pro):
 *   gcc -O3 -march=native -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o build/quantify_dispatch_gap tools/quantify_dispatch_gap.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 */
#include "icm.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── PART 1+2: Model-vs-measured comparison ─────────────────────── */

static double predict_hybrid_per_qp_ns(int n, int k, int B) {
    int nblocks = (n + B - 1) / B;
    double block_build = (double)n * block_build_ns_per_player[B_to_table_index(B)];
    TreeCtx *tc = tree_ctx_create_ex2(nblocks, B, k, B);
    double tree = 0;
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
            tree += nr * (build_fft + corr);
        } else {
            int idx;
            { int lo=0,hi=N_CALIBRATED_SIZES-1;
              while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<cps)lo=m+1;else hi=m;}
              idx=lo; }
            double s = schoolbook_mul_ns[idx];
            double c = (double)cps * tc->g_needed[ell-1] * schoolbook_corr_ns[idx];
            tree += nr * (s + c);
        }
    }
    tree_ctx_destroy(tc);

    int bidx = B_to_table_index(B);
    double le_cost_per_player = (FP64_DIV_NS > leaf_fma_ns_per_player[bidx])
                                ? FP64_DIV_NS : leaf_fma_ns_per_player[bidx];
    double leaf_extract = (double)n * le_cost_per_player;
    return block_build + tree + leaf_extract;
}

/* ── PART 3: Leaf-extraction embedded profiling ────────────────── */

/* Mirrors the leaf-divide step in engine_hybrid_core exactly.
 * Returns elapsed ns for the leaf-extraction phase.
 * Also records per-block timing breakdown if requested. */
static double leaf_divide_timed(int n, int B, int nblocks, int k,
                                 const double *a,
                                 HybridCtx *hc, TreeCtx *tc,
                                 const double *leaf_g) {
    double t0 = now_ns();

    int B_val = B;
    int leaf_psz = tc->psz[0];
    int g_need = tc->g_needed[0];
    double *inner = (double *)malloc(n * sizeof(double));
    if (!inner) return 0;

    for (int b = 0; b < nblocks; b++) {
        int start = b * B_val, end = start + B_val;
        if (end > n) end = n;
        int bsize = end - start;
        double *P_b = hc->block_prods + (size_t)b * (B_val + 1);
        double *g_b = (double *)leaf_g + (size_t)b * leaf_psz;
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

    double dt = now_ns() - t0;
    free(inner);
    return dt;
}

/* tree_propagate_g() already exists in icm.c and returns the leaf-level g buffer.
 * We call it directly with hot_mask=NULL (full-equity path). */

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) ? 1 : (da < db) ? -1 : 0;
}

#define LEAF_REPS 50

/* Profile leaf extraction for one (n,k,B) combo.
 * Runs the full hybrid engine (block_build + tree_build + tree_propagate +
 * leaf_divide) LEAF_REPS times, timing the leaf_divide phase each rep.
 * Returns median leaf_divide time (ns), and also returns the model-predicted
 * leaf cost for comparison. */
static double profile_leaf(int n, int k, int B,
                           double *pred_leaf_ns_out,
                           double *pred_per_player_out) {
    double *S = (double *)malloc(n * sizeof(double));
    double *payout = (double *)malloc(k * sizeof(double));
    double *a = (double *)malloc(n * sizeof(double));
    if (!S || !payout || !a) { fprintf(stderr, "OOM\n"); exit(1); }

    srand(42);
    for (int i = 0; i < n; i++)
        S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
    for (int q = 0; q < k; q++)
        payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

    double logv = -1.0; /* representative quadrature point */
    for (int i = 0; i < n; i++) {
        double arg = S[i] * logv;
        a[i] = (arg < -700) ? 0.0 : exp(arg);
    }

    double leaf_samples[LEAF_REPS];
    int n_reps = 0;
    int nblocks = (n + B - 1) / B;

    for (int rep = 0; rep < LEAF_REPS; rep++) {
        HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
        if (!hc) { fprintf(stderr, "hybrid_ctx_create failed\n"); exit(1); }
        TreeCtx *tc = hc->tc;

        /* Clear workspace */
        memset(tc->ws, 0, tc->ws_size * sizeof(double));

        /* Block build */
        int B_val = hc->B;
        int leaf_psz = tc->psz[0];
        double *plev_data = tc->ws;

        for (int b = 0; b < nblocks; b++) {
            int start = b * B_val, end = start + B_val;
            if (end > n) end = n;
            int bsize = end - start;
            double *P = hc->block_prods + (size_t)b * (B_val + 1);
            memset(P, 0, (B_val + 1) * sizeof(double));
            P[0] = 1.0;
            for (int j = start; j < end; j++) {
                double aj = a[j], bj = 1 - aj;
                for (int m = bsize; m >= 1; m--)
                    P[m] = aj * P[m] + bj * P[m - 1];
                P[0] *= aj;
            }
            double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz;
            int cp = (B_val + 1 < leaf_psz) ? B_val + 1 : leaf_psz;
            memcpy(leaf, P, cp * sizeof(double));
            if (cp < leaf_psz) memset(leaf + cp, 0, (leaf_psz - cp) * sizeof(double));
        }
        for (int b = nblocks; b < tc->N; b++) {
            double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz;
            memset(leaf, 0, leaf_psz * sizeof(double));
            leaf[0] = 1.0;
        }

        /* Tree build */
        tree_build_levels(tc);

        /* Tree propagate — get leaf_g pointer (tree_propagate_g returns it) */
        double *leaf_g = tree_propagate_g(tc, k, payout, NULL);

        /* Time leaf divide */
        double t_leaf = leaf_divide_timed(n, B, nblocks, k, a, hc, tc, leaf_g);

        leaf_samples[rep] = t_leaf;
        n_reps++;

        hybrid_ctx_destroy(hc);
    }

    /* Median */
    qsort(leaf_samples, n_reps, sizeof(double), cmp_double);
    double med_leaf = leaf_samples[n_reps / 2];

    /* Model prediction */
    int bidx = B_to_table_index(B);
    double le_cost_per_player = (FP64_DIV_NS > leaf_fma_ns_per_player[bidx])
                                ? FP64_DIV_NS : leaf_fma_ns_per_player[bidx];
    double pred_leaf_total = (double)n * le_cost_per_player;

    *pred_leaf_ns_out = pred_leaf_total;
    *pred_per_player_out = le_cost_per_player;

    free(S); free(payout); free(a);
    return med_leaf;
}

/* ── Driver ──────────────────────────────────────────────────── */

int main(void) {
    build_fftw_size_table();
    icm_init(NULL);

    int B = 8; /* focus on B=8 (the dispatch B in the crossover region) */

    /* Crossover-region grid: n and k values */
    int n_vals[] = {512, 1024, 2048, 4096, 8192};
    int n_n = 5;
    int k_vals[] = {40, 60, 80, 100, 120, 140, 160, 200, 260, 320, 400};
    int n_k = 11;

    printf("=== PART 1: Generating sample plans + model comparison ===\n\n");
    printf("%-6s %-6s %-3s %14s %14s %8s\n",
           "n", "k", "B", "measured_ns", "predicted_ns", "meas/pred");

    double sum_log_ratio = 0, sum_log_ratio2 = 0;
    double sum_abs_pct = 0;
    double worst_ratio = 1.0, best_ratio = 1.0;
    int n_rows = 0;

    for (int ni = 0; ni < n_n; ni++) {
        int n = n_vals[ni];
        for (int ki = 0; ki < n_k; ki++) {
            int k = k_vals[ki];
            if (k > n) continue;
            if (B > k || B > n) continue;

            /* Measure real hybrid engine time */
            double *S = (double *)malloc(n * sizeof(double));
            double *equity = (double *)malloc(n * sizeof(double));
            double *payout = (double *)malloc(k * sizeof(double));
            if (!S || !equity || !payout) { fprintf(stderr, "OOM\n"); return 1; }

            srand(42);
            for (int i = 0; i < n; i++)
                S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
            for (int q = 0; q < k; q++)
                payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

            int Q = 64; /* fewer Q points for speed; still enough for stable timing */
            double times[3];
            for (int rep = 0; rep < 3; rep++) {
                HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
                double t0 = now_ns();
                run_engine_ctx(n, S, Q, payout, k, equity, engine_hybrid_ctx, hc);
                times[rep] = now_ns() - t0;
                hybrid_ctx_destroy(hc);
            }
            /* Median */
            for (int i = 0; i < 3; i++)
                for (int j = i+1; j < 3; j++)
                    if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
            double meas_per_qp = times[1] / Q;

            double pred = predict_hybrid_per_qp_ns(n, k, B);
            double ratio = meas_per_qp / pred;

            printf("%-6d %-6d %-3d %14.1f %14.1f %8.3f\n",
                   n, k, B, meas_per_qp, pred, ratio);

            double log_ratio = log(ratio);
            sum_log_ratio += log_ratio;
            sum_log_ratio2 += log_ratio * log_ratio;
            sum_abs_pct += fabs(ratio - 1.0) * 100.0;
            if (ratio > worst_ratio) worst_ratio = ratio;
            if (ratio < best_ratio) best_ratio = ratio;
            n_rows++;

            free(S); free(equity); free(payout);
        }
    }

    double mean_log = sum_log_ratio / n_rows;
    double var_log = sum_log_ratio2 / n_rows - mean_log * mean_log;
    double geo_mean_ratio = exp(mean_log);

    printf("\n=== PART 1 SUMMARY (%d rows) ===\n", n_rows);
    printf("geometric mean(measured/predicted) = %.3f\n", geo_mean_ratio);
    printf("log-ratio stddev = %.3f\n", sqrt(var_log));
    printf("mean abs %% error = %.1f%%\n", sum_abs_pct / n_rows);
    printf("ratio range = [%.3f, %.3f]\n", best_ratio, worst_ratio);
    printf("\nBASELINE (pre-schoolbook-fix): geo_mean=1.740, log-stddev=0.236, range=[0.89,3.67]\n");

    /* ── PART 3: Leaf-extraction profiling ── */
    printf("\n=== PART 3: Leaf-extraction embedded profiling (B=%d) ===\n\n", B);
    printf("%-6s %-6s %14s %14s %8s %18s\n",
           "n", "k", "measured_leaf_ns", "pred_leaf_ns", "ratio", "pred_per_player_ns");

    double leaf_sum_log = 0, leaf_sum_log2 = 0;
    int leaf_n = 0;
    double leaf_best = 1e9, leaf_worst = 0;

    for (int ni = 0; ni < n_n; ni++) {
        int n = n_vals[ni];
        for (int ki = 0; ki < n_k; ki++) {
            int k = k_vals[ki];
            if (k > n) continue;
            if (B > k || B > n) continue;

            double pred_leaf_ns, pred_per_player;
            double meas_leaf = profile_leaf(n, k, B, &pred_leaf_ns, &pred_per_player);
            double ratio = meas_leaf / pred_leaf_ns;

            printf("%-6d %-6d %14.1f %14.1f %8.3f %18.3f\n",
                   n, k, meas_leaf, pred_leaf_ns, ratio, pred_per_player);

            double lr = log(ratio);
            leaf_sum_log += lr;
            leaf_sum_log2 += lr * lr;
            leaf_n++;
            if (ratio < leaf_best) leaf_best = ratio;
            if (ratio > leaf_worst) leaf_worst = ratio;
        }
    }

    double leaf_mean_log = leaf_sum_log / leaf_n;
    double leaf_var_log = leaf_sum_log2 / leaf_n - leaf_mean_log * leaf_mean_log;
    double leaf_geo = exp(leaf_mean_log);

    printf("\n=== LEAF EXTRACTION SUMMARY (%d rows) ===\n", leaf_n);
    printf("geometric mean(measured/predicted) = %.3f\n", leaf_geo);
    printf("log-ratio stddev = %.3f\n", sqrt(leaf_var_log));
    printf("ratio range = [%.3f, %.3f]\n", leaf_best, leaf_worst);
    printf("prediction uses: FP64_DIV_NS=%.4f, leaf_fma_ns_per_player[%d]=%.4f\n",
           FP64_DIV_NS, B_to_table_index(B), leaf_fma_ns_per_player[B_to_table_index(B)]);

    /* ── SYNTHESIS: Project crossover with adjusted leaf cost ── */
    printf("\n=== SYNTHESIS: Crossover projection ===\n");
    printf("If leaf_fma_ns_per_player[8] were multiplied by %.3f (measured ratio):\n", leaf_geo);

    double orig_leaf = leaf_fma_ns_per_player[B_to_table_index(B)];
    double adj_leaf = orig_leaf * leaf_geo;

    printf("  Original leaf per-player: %.3f ns\n", orig_leaf);
    printf("  Adjusted leaf per-player:  %.3f ns\n", adj_leaf);
    printf("\n");

    /* Compute select_engine_ex decisions with original and adjusted leaf costs
     * across the crossover grid to see where dispatch changes. */
    printf("%-6s %-6s %12s %12s %12s %12s\n",
           "n", "k", "linear_ns", "hyb_orig_ns", "hyb_adj_leaf", "dispatch");

    /* Temporarily override leaf_fma_ns_per_player for the "adjusted" column.
     * We do this by computing both predictions manually. */
    for (int ni = 0; ni < n_n; ni++) {
        int n = n_vals[ni];
        for (int ki = 0; ki < n_k; ki++) {
            int k = k_vals[ki];
            if (k > n) continue;
            if (B > k || B > n) continue;

            /* Linear cost */
            double linear_full = linear_roofline_cost(n, k, LINEAR_BQ);
            double linear_per_qp = linear_full * 1.0; /* full equity */

            /* Hybrid cost with original leaf */
            double hyb_orig = predict_hybrid_per_qp_ns(n, k, B);

            /* Hybrid cost with adjusted leaf */
            double block_build = (double)n * block_build_ns_per_player[B_to_table_index(B)];
            int nblocks = (n + B - 1) / B;
            TreeCtx *tc = tree_ctx_create_ex2(nblocks, B, k, B);
            double tree = 0;
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
                    tree += nr * (build_fft + corr);
                } else {
                    int idx;
                    { int lo=0,hi=N_CALIBRATED_SIZES-1;
                      while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<cps)lo=m+1;else hi=m;}
                      idx=lo; }
                    double s = schoolbook_mul_ns[idx];
                    double c = (double)cps * tc->g_needed[ell-1] * schoolbook_corr_ns[idx];
                    tree += nr * (s + c);
                }
            }
            tree_ctx_destroy(tc);
            double leaf_adj = (double)n * ((FP64_DIV_NS > adj_leaf) ? FP64_DIV_NS : adj_leaf);
            double hyb_adj = block_build + tree + leaf_adj;

            const char *orig_choice = (hyb_orig < linear_per_qp) ? "HYBRID" : "linear";
            const char *adj_choice = (hyb_adj < linear_per_qp) ? "HYBRID" : "linear";

            printf("%-6d %-6d %12.1f %12.1f %12.1f %s",
                   n, k, linear_per_qp, hyb_orig, hyb_adj, orig_choice);
            if (strcmp(orig_choice, adj_choice) != 0)
                printf(" -> %s", adj_choice);
            printf("\n");
        }
    }

    return 0;
}
