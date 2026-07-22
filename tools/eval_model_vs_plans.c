/* eval_model_vs_plans.c — Compare the CURRENT hybrid cost-model formula
 * (block_build_ns_per_player + tree + leaf, exactly as in select_engine_ex /
 * select_best_B in src/icm.c) against REAL measured sample_plans data, at
 * the SAME B recorded in each row (removing the B-selection confound
 * entirely). This is the disjoint "evaluation" set the aggregate-regression
 * calibration was never checked against after the D1/C1 pin migration.
 *
 * Usage: eval_model_vs_plans sample_plans_<device>.csv
 * Reads columns: n,k,B,L,total_ms,per_qp_ns,... (levels_json ignored)
 */
#include "icm.c"
#include <stdio.h>
#include <string.h>

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
            int d_eff = tc->below_sat[ell] ? cps/2 : cps-1;
            double s = (double)(d_eff+1)*(d_eff+1)*FMA_NS;
            double c = (double)cps * tc->g_needed[ell-1] * FMA_NS * 2;
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

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s sample_plans.csv\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }

    char line[8192];
    fgets(line, sizeof(line), f); /* header */

    int n_rows = 0;
    double sum_log_ratio = 0, sum_log_ratio2 = 0;
    double sum_abs_pct = 0;
    double worst_ratio = 1.0, best_ratio = 1.0;

    printf("%-6s %-6s %-3s %-3s %14s %14s %8s\n",
           "n", "k", "B", "L", "measured_ns", "predicted_ns", "meas/pred");

    while (fgets(line, sizeof(line), f)) {
        int n, k, B, L;
        double total_ms, per_qp_ns;
        int nf = sscanf(line, "%d,%d,%d,%d,%lf,%lf", &n, &k, &B, &L, &total_ms, &per_qp_ns);
        if (nf < 6) continue;
        if (B > k || B > n) continue; /* invalid row, shouldn't happen */

        double pred = predict_hybrid_per_qp_ns(n, k, B);
        double ratio = per_qp_ns / pred;

        printf("%-6d %-6d %-3d %-3d %14.1f %14.1f %8.3f\n",
               n, k, B, L, per_qp_ns, pred, ratio);

        double log_ratio = log(ratio);
        sum_log_ratio += log_ratio;
        sum_log_ratio2 += log_ratio * log_ratio;
        sum_abs_pct += fabs(ratio - 1.0) * 100.0;
        if (ratio > worst_ratio) worst_ratio = ratio;
        if (ratio < best_ratio) best_ratio = ratio;
        n_rows++;
    }
    fclose(f);

    double mean_log = sum_log_ratio / n_rows;
    double var_log = sum_log_ratio2 / n_rows - mean_log * mean_log;
    double geo_mean_ratio = exp(mean_log);

    printf("\n=== SUMMARY (%d rows) ===\n", n_rows);
    printf("geometric mean(measured/predicted) = %.3f  (1.0 = perfect; >1 means model UNDER-predicts hybrid cost; <1 means model OVER-predicts)\n", geo_mean_ratio);
    printf("log-ratio stddev = %.3f (multiplicative spread, e.g. 0.3 =~ x1.35 typical swing)\n", sqrt(var_log));
    printf("mean abs %% error = %.1f%%\n", sum_abs_pct / n_rows);
    printf("ratio range = [%.3f, %.3f]\n", best_ratio, worst_ratio);
    return 0;
}
