/* dump_cost_model.c — Print cost model estimates for comparison with reality.
 *
 * Build: gcc -O3 -march=native -Isrc -Idevices/m3_pro -I/opt/homebrew/include
 *        -o tools/dump_cost_model tools/dump_cost_model.c
 *        -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 */
#include "icm.c"
#include <stdio.h>
#include <stdlib.h>

static void dump_costs(const char *label, int n, int k) {
    int B = select_best_B(n, k);
    printf("\n=== %s: n=%d k=%d ===\n", label, n, k);

    /* Linear roofline cost */
    double linear_full = linear_roofline_cost(n, k, LINEAR_BQ);
    double linear_per_qp = linear_full * 0.5;
    printf("linear_roofline_cost(full) = %.0f ns/qp\n", linear_full);
    printf("linear_per_qp (0.5x)      = %.0f ns/qp\n", linear_per_qp);

    /* Hybrid cost */
    printf("select_best_B = %d\n", B);
    int nblocks = (n + B - 1) / B;
    double block_build = (double)n * ((double)(B+1) / 2.0 * BLOCK_FMA_NS + BLOCK_MEM_NS);
    printf("block_build = %.0f ns\n", block_build);

    TreeCtx *tc = tree_ctx_create_ex2(nblocks, B, k, B);
    printf("Tree: L=%d\n", tc->L);
    printf("  psz: ");
    for (int i = 0; i < tc->L; i++) printf("%d ", tc->psz[i]);
    printf("\n");
    printf("  n_real: ");
    for (int i = 0; i < tc->L; i++) printf("%d ", tc->n_real[i]);
    printf("\n");
    printf("  use_fft: ");
    for (int i = 0; i < tc->L; i++) printf("%d ", tc->use_fft[i]);
    printf("\n");
    printf("  below_sat: ");
    for (int i = 0; i < tc->L; i++) printf("%d ", tc->below_sat[i]);
    printf("\n");

    double tree = 0;
    for (int ell = 1; ell < tc->L - 1; ell++) {
        int cps = tc->psz[ell-1], nr = tc->n_real[ell];
        double level_cost;
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
            level_cost = nr * (build_fft + corr);
            printf("  L%d: FFT  bfn=%d cfn=%d cache_ok=%d bwm=%d cwm=%d  build=%.0f corr=%.0f level=%.0f\n",
                   ell, bfn, tc->corr_fft_n[ell], tc->fft_cache_ok[ell],
                   bwm, tc->corr_wrap_m[ell], build_fft, corr, level_cost);
        } else {
            int d_eff = tc->below_sat[ell] ? cps/2 : cps-1;
            double s = (double)(d_eff+1)*(d_eff+1)*FMA_NS;
            double c = (double)cps * tc->g_needed[ell-1] * FMA_NS * 2;
            level_cost = nr * (s + c);
            printf("  L%d: DIR  cps=%d d_eff=%d g_need=%d  s=%.0f c=%.0f level=%.0f\n",
                   ell, cps, d_eff, tc->g_needed[ell-1], s, c, level_cost);
        }
        tree += level_cost;
    }

    double le_div = FP64_DIV_NS, le_fma = 2.0 * B * LEAF_FMA_NS;
    double leaf_extract = (double)n * (le_div > le_fma ? le_div : le_fma)
                        + (double)n / B * LEAF_BLOCK_NS;
    double hybrid_total = block_build + tree + leaf_extract;

    printf("tree           = %.0f ns\n", tree);
    printf("leaf_extract   = %.0f ns\n", leaf_extract);
    printf("hybrid_total   = %.0f ns/qp\n", hybrid_total);

    printf("RESULT: hybrid_total (%.0f) %s linear_per_qp (%.0f) => select=%d\n",
           hybrid_total,
           hybrid_total < linear_per_qp ? "<" : ">=",
           linear_per_qp,
           hybrid_total < linear_per_qp ? B : 0);

    tree_ctx_destroy(tc);
}

int main(void) {
    build_fftw_size_table();
    icm_init(NULL);

    dump_costs("ANOMALOUS", 16384, 1638);
    dump_costs("FAST", 16384, 4096);
    dump_costs("BOUNDARY", 16384, 2048);

    /* Scan k to find the threshold */
    printf("\n=== Scanning k at n=16384 ===\n");
    printf("k,linear_per_qp,hybrid_total,select_B\n");
    for (int k = 1024; k <= 5000; k += 128) {
        int B = select_best_B(16384, k);
        double linear_full = linear_roofline_cost(16384, k, LINEAR_BQ);
        double linear_per_qp = linear_full * 0.5;

        int nblocks = (16384 + B - 1) / B;
        double block_build = (double)16384 * ((double)(B+1) / 2.0 * BLOCK_FMA_NS + BLOCK_MEM_NS);
        TreeCtx *tc2 = tree_ctx_create_ex2(nblocks, B, k, B);
        double tree2 = 0;
        for (int ell = 1; ell < tc2->L - 1; ell++) {
            int cps = tc2->psz[ell-1], nr = tc2->n_real[ell];
            if (tc2->use_fft[ell]) {
                int bfn = tc2->build_fft_n[ell];
                int bwm = tc2->build_wrap_m[ell];
                int idx = 0;
                { int lo=0,hi=N_CALIBRATED_SIZES-1;
                  while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<bfn)lo=m+1;else hi=m;}
                  idx=lo; }
                double build_fft = calib_times_ns[idx] + FFT_OVERHEAD_NS
                                 + (double)bwm*(bwm+1)/2.0*FMA_NS;
                double corr;
                if (tc2->fft_cache_ok[ell]) {
                    corr = calib_times_ns[idx] * PAIRED_CACHED_CORR_RATIO
                         + (double)tc2->corr_wrap_m[ell]*(tc2->corr_wrap_m[ell]+1)*FMA_NS;
                } else {
                    int cfn = tc2->corr_fft_n[ell];
                    int cwm = tc2->corr_wrap_m[ell];
                    int cidx=0;
                    {int lo=0,hi=N_CALIBRATED_SIZES-1;
                     while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<cfn)lo=m+1;else hi=m;}
                     cidx=lo;}
                    corr = INDEP_PAIR_RATIO * calib_times_ns[cidx]
                         + (double)cwm*(cwm+1)*FMA_NS;
                }
                tree2 += nr * (build_fft + corr);
            } else {
                int d_eff = tc2->below_sat[ell] ? cps/2 : cps-1;
                double s = (double)(d_eff+1)*(d_eff+1)*FMA_NS;
                double c = (double)cps * tc2->g_needed[ell-1] * FMA_NS * 2;
                tree2 += nr * (s + c);
            }
        }
        double le_div = FP64_DIV_NS, le_fma = 2.0 * B * LEAF_FMA_NS;
        double leaf_extract = (double)16384 * (le_div > le_fma ? le_div : le_fma)
                            + (double)16384 / B * LEAF_BLOCK_NS;
        double hybrid_total = block_build + tree2 + leaf_extract;
        tree_ctx_destroy(tc2);

        printf("%d,%.0f,%.0f,%d\n", k, linear_per_qp, hybrid_total,
               hybrid_total < linear_per_qp ? B : 0);
    }

    return 0;
}
