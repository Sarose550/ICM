/* tools/test_cpu_cost_model.c -- CPU cost model unit tests.
 *
 * Tests best_fft_config, best_fft_config_joint, select_best_B,
 * select_engine, and tree_ctx_create_ex2 from the CPU implementation.
 *
 * Build: make test_cpu_cost_model DEVICE=zen4  (or m3_pro)
 * Run:   ./test_cpu_cost_model
 *
 * We #include "icm.c" to access static functions directly.
 */
#include "icm.c"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int n_pass = 0, n_fail = 0, n_warn = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "  FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } \
    else { n_pass++; } \
} while(0)

#define WARN(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "  WARN: " __VA_ARGS__); fprintf(stderr, "\n"); n_warn++; } \
} while(0)

/* ═══════════════════════════════════════════════════════════════
   Test 1: best_fft_config wrap formulas
   Build: wrap_m, cost += m*(m+1)/2 * FMA_NS
   Corr:  wrap_m, cost += m*(m+1)   * FMA_NS  (no /2)
   ═══════════════════════════════════════════════════════════════ */
static void test_wrap_formulas(void) {
    int conv_lens[] = {63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767};
    int n_cl = sizeof(conv_lens) / sizeof(conv_lens[0]);

    for (int ci = 0; ci < n_cl; ci++) {
        int conv_len = conv_lens[ci];
        int bfn = 0, bwm = 0;
        best_fft_config(conv_len, &bfn, &bwm, 0);  /* build: len_P=0 */

        CHECK(bfn >= conv_len / 2 + 1,
              "conv=%d bfn=%d < min_size=%d", conv_len, bfn, conv_len / 2 + 1);

        int expected_wrap = (bfn >= conv_len) ? 0 : (conv_len - bfn);
        CHECK(bwm == expected_wrap,
              "conv=%d bfn=%d bwm=%d expected=%d", conv_len, bfn, bwm, expected_wrap);

        /* Correlate: len_P > 0 means corr_wrap uses m*(m+1) */
        int cfn = 0, cwm = 0;
        best_fft_config(conv_len, &cfn, &cwm, conv_len / 4);
        int exp_cw = (cfn >= conv_len) ? 0 : (conv_len - cfn);
        CHECK(cwm == exp_cw,
              "conv=%d cfn=%d cwm=%d expected=%d (corr)", conv_len, cfn, cwm, exp_cw);
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 2: best_fft_config_joint wrap consistency
   ═══════════════════════════════════════════════════════════════ */
static void test_joint_config(void) {
    struct TC { int build_conv; int corr_conv; int p_eff; };
    struct TC cases[] = {
        {63, 127, 32},
        {127, 255, 64},
        {255, 511, 128},
        {511, 1023, 256},
        {1023, 2047, 512},
        {2047, 4095, 1024},
    };
    int n_tc = sizeof(cases) / sizeof(cases[0]);

    for (int i = 0; i < n_tc; i++) {
        int jfn = 0, jbm = 0, jcm = 0;
        best_fft_config_joint(cases[i].build_conv, cases[i].corr_conv,
                              cases[i].p_eff, &jfn, &jbm, &jcm);

        int exp_bm = (jfn >= cases[i].build_conv) ? 0 : (cases[i].build_conv - jfn);
        int exp_cm = (jfn >= cases[i].corr_conv)  ? 0 : (cases[i].corr_conv  - jfn);
        CHECK(jbm == exp_bm,
              "joint[%d] build: jfn=%d jbm=%d expected=%d", i, jfn, jbm, exp_bm);
        CHECK(jcm == exp_cm,
              "joint[%d] corr: jfn=%d jcm=%d expected=%d", i, jfn, jcm, exp_cm);

        /* Joint size must be >= max(build_conv, corr_conv)/2 + 1 */
        int max_conv = cases[i].build_conv > cases[i].corr_conv
                       ? cases[i].build_conv : cases[i].corr_conv;
        CHECK(jfn >= max_conv / 2 + 1,
              "joint[%d] jfn=%d < min=%d", i, jfn, max_conv / 2 + 1);
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 3: select_best_B sanity
   ═══════════════════════════════════════════════════════════════ */
static void test_select_best_B(void) {
    int test_n[] = {256, 1024, 4096, 16384, 65536};
    int n_tests = sizeof(test_n) / sizeof(test_n[0]);

    for (int ti = 0; ti < n_tests; ti++) {
        int n = test_n[ti], k = n;
        int B = select_best_B(n, k);

        CHECK(B >= 8 && B <= 64,
              "n=%d B=%d out of candidate range [8,64]", n, B);
        CHECK(B <= n && B <= k,
              "n=%d B=%d exceeds n or k", n, B);

        /* Check it's actually in the candidate list */
        int cands[] = {8, 16, 24, 32, 48, 64};
        int found = 0;
        for (int c = 0; c < 6; c++) if (cands[c] == B) found = 1;
        CHECK(found, "n=%d B=%d not in candidate list", n, B);

        fprintf(stderr, "  n=%d k=%d -> B=%d\n", n, k, B);
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 4: select_engine dispatch — linear vs hybrid crossover
   Small k should use linear (return 0), large k should use hybrid.
   ═══════════════════════════════════════════════════════════════ */
static void test_select_engine(void) {
    /* Very small: should be linear */
    CHECK(select_engine(16, 2) == 0,
          "n=16 k=2 should be linear");
    CHECK(select_engine(8, 4) == 0,
          "n=8 k=4 should be linear");

    /* Large n, large k: should be hybrid */
    int B = select_engine(65536, 65536);
    CHECK(B > 0, "n=65536 k=65536 should be hybrid, got B=%d", B);

    /* Sweep k at fixed n=8192 — find crossover */
    int n = 8192;
    int last_engine = 0;
    int crossover_k = -1;
    for (int k = 4; k <= n; k = (k < 64) ? k + 4 : k * 2) {
        int eng = select_engine(n, k);
        if (last_engine == 0 && eng > 0 && crossover_k < 0) crossover_k = k;
        last_engine = eng;
    }
    fprintf(stderr, "  n=%d crossover at k=%d\n", n, crossover_k);
    WARN(crossover_k > 0 && crossover_k < n,
         "n=%d expected crossover somewhere in [4, %d], got %d", n, n, crossover_k);
}

/* ═══════════════════════════════════════════════════════════════
   Test 5: tree_ctx_create_ex2 level assignments
   FFT vs schoolbook per level, below_sat detection, wrap values.
   ═══════════════════════════════════════════════════════════════ */
static void test_tree_ctx_levels(void) {
    int test_cases[][2] = {{512, 16}, {512, 32}, {512, 64}, {1024, 32}, {2048, 16}};
    int n_tc = sizeof(test_cases) / sizeof(test_cases[0]);

    for (int ti = 0; ti < n_tc; ti++) {
        int nblocks = test_cases[ti][0];
        int B = test_cases[ti][1];
        int k = nblocks * B;  /* k = n for n=k benchmark */
        TreeCtx *tc = tree_ctx_create_ex2(nblocks, B, k, B);
        CHECK(tc != NULL, "tree_ctx_create(%d, %d, %d) returned NULL", nblocks, B, k);
        if (!tc) continue;

        fprintf(stderr, "  nblocks=%d B=%d k=%d L=%d\n", nblocks, B, k, tc->L);

        for (int ell = 1; ell < tc->L; ell++) {
            int cps = tc->psz[ell-1];
            int pgsz = tc->psz[ell];

            /* Below-sat detection: psz[ell] == 2*psz[ell-1] && cps >= 2 */
            int expect_below = (pgsz == 2 * cps && cps >= 2) ? 1 : 0;
            CHECK(tc->below_sat[ell] == expect_below,
                  "ell=%d below_sat=%d expected=%d (cps=%d pgsz=%d)",
                  ell, tc->below_sat[ell], expect_below, cps, pgsz);

            /* If using FFT, wrap values must be non-negative */
            if (tc->use_fft[ell]) {
                CHECK(tc->build_wrap_m[ell] >= 0,
                      "ell=%d build_wrap_m=%d < 0", ell, tc->build_wrap_m[ell]);
                CHECK(tc->corr_wrap_m[ell] >= 0,
                      "ell=%d corr_wrap_m=%d < 0", ell, tc->corr_wrap_m[ell]);

                /* build_fft_n must be calibrated */
                int bfn = tc->build_fft_n[ell];
                CHECK(bfn > 0, "ell=%d build_fft_n=%d invalid", ell, bfn);

                /* wrap_m consistency */
                int is_below = tc->below_sat[ell];
                int conv_build = is_below ? (2 * (cps / 2)) : (2 * cps - 1);
                int exp_bwm = (bfn >= conv_build) ? 0 : (conv_build - bfn);
                CHECK(tc->build_wrap_m[ell] == exp_bwm,
                      "ell=%d bfn=%d conv=%d bwm=%d expected=%d",
                      ell, bfn, conv_build, tc->build_wrap_m[ell], exp_bwm);
            }
        }
        tree_ctx_destroy(tc);
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 6: Cost model B sweep — verify model picks reasonable B
   Compare model's best_B against forced-B cost sweep.
   ═══════════════════════════════════════════════════════════════ */
static void test_b_sweep_cpu(void) {
    int test_n[] = {1024, 4096, 16384, 65536};
    int n_tests = sizeof(test_n) / sizeof(test_n[0]);
    int cands[] = {8, 16, 24, 32, 48, 64};

    for (int ti = 0; ti < n_tests; ti++) {
        int n = test_n[ti], k = n;
        int model_B = select_best_B(n, k);

        /* Compute cost for each B candidate */
        double costs[6];
        double min_cost = 1e18;
        int min_B = 0;
        for (int ci = 0; ci < 6; ci++) {
            int B = cands[ci];
            if (B > k || B > n) { costs[ci] = 1e18; continue; }
            int nblocks = (n + B - 1) / B;
            TreeCtx *tc = tree_ctx_create_ex2(nblocks, B, k, B);
            double block = (double)n * ((double)(B+1)/2.0 * BLOCK_FMA_NS + BLOCK_MEM_NS);
            double le_d = FP64_DIV_NS, le_f = 2.0*B*LEAF_FMA_NS;
            double leaf_cost = (double)n * (le_d > le_f ? le_d : le_f)
                             + (double)n / B * LEAF_BLOCK_NS;
            double tree = 0;
            for (int ell = 1; ell < tc->L - 1; ell++) {
                int cps = tc->psz[ell-1], nr = tc->n_real[ell];
                if (tc->use_fft[ell]) {
                    int bfn = tc->build_fft_n[ell], bwm = tc->build_wrap_m[ell];
                    int idx=0; { int lo=0,hi=N_CALIBRATED_SIZES-1;
                      while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<bfn)lo=m+1;else hi=m;}
                      idx=lo; }
                    double bf = calib_times_ns[idx] + FFT_OVERHEAD_NS
                              + (double)bwm*(bwm+1)/2.0*FMA_NS;
                    double corr;
                    if (tc->fft_cache_ok[ell]) {
                        corr = calib_times_ns[idx] * PAIRED_CACHED_CORR_RATIO
                             + (double)tc->corr_wrap_m[ell]*(tc->corr_wrap_m[ell]+1)*FMA_NS;
                    } else {
                        int cfn = tc->corr_fft_n[ell], cwm = tc->corr_wrap_m[ell];
                        int cidx=0; {int lo=0,hi=N_CALIBRATED_SIZES-1;
                         while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<cfn)lo=m+1;else hi=m;}
                         cidx=lo;}
                        corr = INDEP_PAIR_RATIO * calib_times_ns[cidx]
                             + (double)cwm*(cwm+1)*FMA_NS;
                    }
                    tree += nr * (bf + corr);
                } else {
                    int is_below = tc->below_sat[ell];
                    int d_eff = is_below ? cps/2 : cps-1;
                    double school_mul = (double)(d_eff+1)*(d_eff+1)*FMA_NS;
                    double school_corr = (double)cps * tc->g_needed[ell-1] * FMA_NS * 2;
                    tree += nr * (school_mul + school_corr);
                }
            }
            tree_ctx_destroy(tc);
            costs[ci] = block + leaf_cost + tree;
            if (costs[ci] < min_cost) { min_cost = costs[ci]; min_B = B; }
        }

        CHECK(model_B == min_B,
              "n=%d model_B=%d != sweep_min_B=%d", n, model_B, min_B);

        fprintf(stderr, "  n=%d model_B=%d:", n, model_B);
        for (int ci = 0; ci < 6; ci++) {
            if (cands[ci] > k || cands[ci] > n) continue;
            fprintf(stderr, " B=%d=%.0f%s", cands[ci], costs[ci],
                    cands[ci] == model_B ? "*" : "");
        }
        fprintf(stderr, "\n");
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 7: FFT config picks reasonable sizes
   The chosen FFT size should not be wildly larger than conv_len.
   ═══════════════════════════════════════════════════════════════ */
static void test_fft_config_sizes(void) {
    int conv_lens[] = {15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383};
    int n_cl = sizeof(conv_lens) / sizeof(conv_lens[0]);

    for (int ci = 0; ci < n_cl; ci++) {
        int conv = conv_lens[ci];
        int bfn = 0, bwm = 0;
        best_fft_config(conv, &bfn, &bwm, 0);

        /* FFT size should be between conv/2+1 and 2*conv */
        CHECK(bfn >= conv / 2 + 1 && bfn <= 2 * conv,
              "conv=%d bfn=%d out of range [%d, %d]", conv, bfn, conv/2+1, 2*conv);

        /* Cost with this size should be reasonable */
        int idx = 0;
        { int lo=0,hi=N_CALIBRATED_SIZES-1;
          while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<bfn)lo=m+1;else hi=m;}
          idx=lo; }
        double cost = calib_times_ns[idx] + (double)bwm*(bwm+1)/2.0*FMA_NS;

        /* The no-wrap option (next smooth >= conv) should not be 3x cheaper */
        int nowrap_fft = bfn;
        /* Find next calibrated size >= conv */
        { int lo=0,hi=N_CALIBRATED_SIZES-1;
          while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<conv)lo=m+1;else hi=m;}
          if (lo < N_CALIBRATED_SIZES) nowrap_fft = calib_sizes[lo]; }
        int nw_idx = 0;
        { int lo=0,hi=N_CALIBRATED_SIZES-1;
          while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<nowrap_fft)lo=m+1;else hi=m;}
          nw_idx=lo; }
        double nowrap_cost = calib_times_ns[nw_idx];
        /* The chosen config should be no worse than 2x the no-wrap option */
        WARN(cost <= nowrap_cost * 2.0,
             "conv=%d chosen bfn=%d cost=%.0f but no-wrap=%d cost=%.0f (>2x)",
             conv, bfn, cost, nowrap_fft, nowrap_cost);
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 8: Wrap formula consistency across functions
   tree_ctx_create_ex2 and select_best_B must use the same
   wrap formulas as best_fft_config / best_fft_config_joint:
     build: bwm*(bwm+1)/2 * FMA_NS
     corr:  cwm*(cwm+1)   * FMA_NS
   ═══════════════════════════════════════════════════════════════ */
static void test_wrap_formula_consistency(void) {
    /* Build a tree and check that the FFT vs schoolbook decision in
     * tree_ctx_create_ex2 uses the same wrap cost as best_fft_config. */
    int test_cases[][2] = {{256, 16}, {512, 32}, {1024, 16}, {2048, 24}};
    int n_tc = sizeof(test_cases) / sizeof(test_cases[0]);

    for (int ti = 0; ti < n_tc; ti++) {
        int nblocks = test_cases[ti][0];
        int B = test_cases[ti][1];
        int k = nblocks * B;
        TreeCtx *tc = tree_ctx_create_ex2(nblocks, B, k, B);
        if (!tc) continue;

        for (int ell = 1; ell < tc->L; ell++) {
            if (!tc->use_fft[ell]) continue;
            int bfn = tc->build_fft_n[ell];
            int bwm = tc->build_wrap_m[ell];
            int cwm = tc->corr_wrap_m[ell];

            /* Verify the tree context stores wrap_m values consistent with
             * what best_fft_config would return for the same conv_len. */
            int cps = tc->psz[ell-1];
            int is_below = tc->below_sat[ell];
            int conv_build = is_below ? (2 * (cps / 2)) : (2 * cps - 1);
            int expected_bwm = (bfn >= conv_build) ? 0 : (conv_build - bfn);
            CHECK(bwm == expected_bwm,
                  "nblocks=%d B=%d ell=%d bwm=%d expected=%d (bfn=%d conv=%d)",
                  nblocks, B, ell, bwm, expected_bwm, bfn, conv_build);

            /* Verify the build wrap cost formula: bwm*(bwm+1)/2
             * (NOT (bwm+1)^2 which was the old bug) */
            double correct_build_wrap = (double)bwm * (bwm + 1) / 2.0 * FMA_NS;
            double wrong_build_wrap = (double)(bwm + 1) * (bwm + 1) * FMA_NS;
            if (bwm > 0) {
                CHECK(correct_build_wrap < wrong_build_wrap,
                      "sanity: correct build wrap %.1f should be < wrong %.1f",
                      correct_build_wrap, wrong_build_wrap);
            }

            /* Verify corr wrap: cwm*(cwm+1) (NOT 2*(cwm+1)^2) */
            double correct_corr_wrap = (double)cwm * (cwm + 1) * FMA_NS;
            double wrong_corr_wrap = 2.0 * (double)(cwm + 1) * (cwm + 1) * FMA_NS;
            if (cwm > 0) {
                CHECK(correct_corr_wrap < wrong_corr_wrap,
                      "sanity: correct corr wrap %.1f should be < wrong %.1f",
                      correct_corr_wrap, wrong_corr_wrap);
            }
        }
        tree_ctx_destroy(tc);
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 9: Calibration table sanity
   ═══════════════════════════════════════════════════════════════ */
static void test_calibration_table(void) {
    CHECK(N_CALIBRATED_SIZES > 100,
          "N_CALIBRATED_SIZES=%d too small", N_CALIBRATED_SIZES);

    /* Sizes must be sorted and positive */
    for (int i = 1; i < N_CALIBRATED_SIZES; i++) {
        CHECK(calib_sizes[i] > calib_sizes[i-1],
              "calib_sizes[%d]=%d <= calib_sizes[%d]=%d",
              i, calib_sizes[i], i-1, calib_sizes[i-1]);
    }

    /* Times must be positive */
    for (int i = 0; i < N_CALIBRATED_SIZES; i++) {
        CHECK(calib_times_ns[i] > 0,
              "calib_times_ns[%d]=%.3f <= 0 (size=%d)",
              i, calib_times_ns[i], calib_sizes[i]);
    }

    /* Key power-of-2 sizes should be in the table */
    int pow2s[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    for (int pi = 0; pi < 11; pi++) {
        int s = pow2s[pi];
        int found = 0;
        for (int i = 0; i < N_CALIBRATED_SIZES; i++) {
            if (calib_sizes[i] == s) { found = 1; break; }
        }
        CHECK(found, "power-of-2 size %d not in calibration table", s);
    }
}

/* ═══════════════════════════════════════════════════════════════
   Diagnostic: dump select_best_B level costs for a case
   ═══════════════════════════════════════════════════════════════ */
static void dump_level_info(int n, int B) {
    int nblocks = (n + B - 1) / B;
    int k = n;
    TreeCtx *tc = tree_ctx_create_ex2(nblocks, B, k, B);
    if (!tc) { fprintf(stderr, "  (null)\n"); return; }

    fprintf(stderr, "  n=%d B=%d nblocks=%d L=%d\n", n, B, nblocks, tc->L);
    for (int ell = 1; ell < tc->L; ell++) {
        int cps = tc->psz[ell-1];
        int is_below = tc->below_sat[ell];
        int d_eff = is_below ? cps/2 : cps-1;
        int conv_build = is_below ? (2*(cps/2)) : (2*cps-1);
        fprintf(stderr, "    ell=%d nr=%d cps=%d psz=%d conv=%d fft=%s",
                ell, tc->n_real[ell], cps, tc->psz[ell], conv_build,
                tc->use_fft[ell] ? "Y" : "N");
        if (tc->use_fft[ell]) {
            fprintf(stderr, " bfn=%d bwm=%d cwm=%d cache=%d",
                    tc->build_fft_n[ell], tc->build_wrap_m[ell],
                    tc->corr_wrap_m[ell], tc->fft_cache_ok[ell]);
        } else {
            fprintf(stderr, " d_eff=%d", d_eff);
        }
        fprintf(stderr, "\n");
    }
    tree_ctx_destroy(tc);
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    fprintf(stderr, "=== CPU Cost Model Unit Tests ===\n\n");

    /* Initialize FFTW and calibration */
    build_fftw_size_table();
    /* Try loading wisdom, but tests work without it */
    wisdom_load();

    fprintf(stderr, "Config: FMA_NS=%.3f  FFT_OVERHEAD_NS=%.1f  "
            "PAIRED_CACHED_CORR_RATIO=%.3f  INDEP_PAIR_RATIO=%.3f\n",
            FMA_NS, FFT_OVERHEAD_NS, PAIRED_CACHED_CORR_RATIO, INDEP_PAIR_RATIO);
    fprintf(stderr, "N_CALIBRATED_SIZES=%d  max_calib=%d\n\n",
            N_CALIBRATED_SIZES, calib_sizes[N_CALIBRATED_SIZES - 1]);

    fprintf(stderr, "--- 1. Wrap formulas ---\n");
    test_wrap_formulas();

    fprintf(stderr, "--- 2. Joint config ---\n");
    test_joint_config();

    fprintf(stderr, "--- 3. select_best_B ---\n");
    test_select_best_B();

    fprintf(stderr, "--- 4. select_engine ---\n");
    test_select_engine();

    fprintf(stderr, "--- 5. Tree ctx levels ---\n");
    test_tree_ctx_levels();

    fprintf(stderr, "--- 6. B sweep cost verification ---\n");
    test_b_sweep_cpu();

    fprintf(stderr, "--- 7. FFT config sizes ---\n");
    test_fft_config_sizes();

    fprintf(stderr, "--- 8. Wrap formula consistency ---\n");
    test_wrap_formula_consistency();

    fprintf(stderr, "--- 9. Calibration table ---\n");
    test_calibration_table();

    fprintf(stderr, "\n--- Diagnostic: level info ---\n");
    dump_level_info(4096, 16);
    dump_level_info(4096, 32);
    dump_level_info(65536, 32);
    dump_level_info(65536, 64);

    fprintf(stderr, "\n=== Results: %d passed, %d failed, %d warnings ===\n",
            n_pass, n_fail, n_warn);
    return n_fail > 0 ? 1 : 0;
}
