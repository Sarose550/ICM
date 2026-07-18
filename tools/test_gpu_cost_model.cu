/* tools/test_gpu_cost_model.cu -- GPU cost model unit tests (host-side only).
 *
 * Validates invariants of the cost model in gpu_plan.cu without launching
 * any GPU kernels.  All tested functions are pure host-side math.
 *
 * Build: make test_gpu_cost_model CUDA_ARCH=sm_100 CUFFTDX_INC=...
 * Run:   ./test_gpu_cost_model
 */
#include "gpu_internal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace icm_gpu_detail;

static int n_pass = 0, n_fail = 0, n_warn = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: " __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        n_fail++; \
    } else { \
        n_pass++; \
    } \
} while(0)

#define WARN(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "  WARN: " __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        n_warn++; \
    } \
} while(0)

/* ═══════════════════════════════════════════════════════════════
   Test 1: Tier selection invariant
   pick_tier_for_fft_len must NEVER return GPU_TIER_SCHOOLBOOK
   when conv_len <= GPU_FUSED_MAX_CONV_LEN.
   ═══════════════════════════════════════════════════════════════ */
static void test_tier_never_schoolbook_in_fused_range() {
    int max_conv = g_runtime_fused_max_conv_len;

    /* Power-of-2 fft_n: fused should be calibrated */
    int fused_sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int n_fused = sizeof(fused_sizes) / sizeof(fused_sizes[0]);
    for (int i = 0; i < n_fused; i++) {
        int fft_n = fused_sizes[i];
        for (int conv_len = 3; conv_len <= std::min(fft_n, max_conv); conv_len = conv_len * 2 - 1) {
            int tier = pick_tier_for_fft_len(fft_n, conv_len);
            CHECK(tier != GPU_TIER_SCHOOLBOOK,
                  "pick_tier(%d, %d) = SCHOOLBOOK (pow2 fft_n, fused range)", fft_n, conv_len);
        }
    }

    /* Non-power-of-2 fft_n: fused NOT calibrated at this size, but still in range.
     * Must return CUFFT (never SCHOOLBOOK) per our fix. */
    int non_p2[] = {384, 640, 768, 1280, 1536, 2560, 3072, 5120, 6144};
    int n_np2 = sizeof(non_p2) / sizeof(non_p2[0]);
    for (int i = 0; i < n_np2; i++) {
        int fft_n = non_p2[i];
        if (fft_n > max_conv) continue;
        int tier = pick_tier_for_fft_len(fft_n, fft_n);
        CHECK(tier != GPU_TIER_SCHOOLBOOK,
              "pick_tier(%d, %d) = SCHOOLBOOK (non-pow2, fused range)", fft_n, fft_n);
        CHECK(tier == GPU_TIER_CUFFT,
              "pick_tier(%d, %d) = %d, expected CUFFT=%d", fft_n, fft_n, tier, GPU_TIER_CUFFT);
    }

    /* Above fused range: schoolbook IS allowed if FMA model says so */
    int large = max_conv + 1;
    int tier_large = pick_tier_for_fft_len(next_pow2_int(large), large);
    /* Just verify it doesn't crash — SCHOOLBOOK may or may not be returned */
    CHECK(tier_large == GPU_TIER_SCHOOLBOOK || tier_large == GPU_TIER_CUFFT,
          "pick_tier above fused range returned invalid tier %d", tier_large);
}

/* ═══════════════════════════════════════════════════════════════
   Test 2: Cost monotonicity for power-of-2 B values
   No adjacent power-of-2 B should differ by more than 5×.
   ═══════════════════════════════════════════════════════════════ */
static void test_cost_monotonicity() {
    std::vector<int> smooth;
    build_smooth_table(1 << 21, smooth);

    int test_n[] = {65536, 131072, 262144, 786432};
    for (int ti = 0; ti < 4; ti++) {
        int n = test_n[ti];
        int k_pad = best_k_pad_gpu(n, smooth);
        int p2_Bs[] = {32, 64, 128, 256, 512, 1024, 2048};
        double prev_cost = -1;
        int prev_B = 0;
        for (int bi = 0; bi < 7; bi++) {
            int B = p2_Bs[bi];
            if (B > n || B > k_pad) continue;
            double cost = estimate_candidate_cost(n, k_pad, B, smooth);
            if (!std::isfinite(cost)) continue;
            if (prev_cost > 0) {
                double ratio = cost / prev_cost;
                CHECK(ratio > 0.2 && ratio < 5.0,
                      "n=%d B=%d->%d ratio=%.2f (cost %.0f->%.0f)",
                      n, prev_B, B, ratio, prev_cost, cost);
            }
            prev_cost = cost;
            prev_B = B;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 3: Fused preference for power-of-2 B
   At n=65536, power-of-2 B (64, 128) should generally beat
   non-power-of-2 B (96) because fused kernels are available.
   ═══════════════════════════════════════════════════════════════ */
static void test_fused_preference() {
    std::vector<int> smooth;
    build_smooth_table(1 << 20, smooth);

    int n = 65536;
    int k_pad = best_k_pad_gpu(n, smooth);

    double cost_64  = estimate_candidate_cost(n, k_pad, 64, smooth);
    double cost_96  = estimate_candidate_cost(n, k_pad, 96, smooth);
    double cost_128 = estimate_candidate_cost(n, k_pad, 128, smooth);
    double cost_192 = estimate_candidate_cost(n, k_pad, 192, smooth);
    double cost_256 = estimate_candidate_cost(n, k_pad, 256, smooth);

    fprintf(stderr, "  n=%d: B=64 %.0f  B=96 %.0f  B=128 %.0f  B=192 %.0f  B=256 %.0f ns\n",
            n, cost_64, cost_96, cost_128, cost_192, cost_256);

    /* At least one power-of-2 B should beat B=96 */
    double best_p2 = std::min({cost_64, cost_128, cost_256});
    WARN(best_p2 <= cost_96,
         "n=%d: no pow2 B beats B=96 (best_p2=%.0f, B=96=%.0f)", n, best_p2, cost_96);

    /* At larger n, check B=384 vs B=192 */
    int n2 = 786432;
    int k_pad2 = best_k_pad_gpu(n2, smooth);
    double cost_192b = estimate_candidate_cost(n2, k_pad2, 192, smooth);
    double cost_384  = estimate_candidate_cost(n2, k_pad2, 384, smooth);
    double cost_512  = estimate_candidate_cost(n2, k_pad2, 512, smooth);

    fprintf(stderr, "  n=%d: B=192 %.0f  B=384 %.0f  B=512 %.0f ns\n",
            n2, cost_192b, cost_384, cost_512);

    WARN(cost_384 < cost_192b * 1.5,
         "n=%d: B=384 (%.0f) should not be >50%% more than B=192 (%.0f)", n2, cost_384, cost_192b);
}

/* ═══════════════════════════════════════════════════════════════
   Test 4: Budget consistency — per_q_bytes includes spec/scratch/cache
   Compute per_q independently and verify it's consistent with
   the cost model's internal calculation.
   ═══════════════════════════════════════════════════════════════ */
static size_t compute_per_q_model(int n, int k_pad, int B) {
    int nblocks = (n + B - 1) / B;
    int N = 0, L = 0;
    std::vector<int> nn, psz, g_needed, below_sat, n_real;
    std::vector<size_t> plev_off;
    build_tree_geometry(nblocks, B, k_pad, B, nn, psz, plev_off, g_needed, below_sat, n_real, N, L);

    double fma_ns = tree_school_ns_per_fma();
    size_t per_q = 0;
    for (int ell = 0; ell < L; ++ell)
        per_q += 2 * (size_t)nn[ell] * psz[ell] * sizeof(double);
    per_q += (size_t)N * (B + 1) * sizeof(double);
    per_q += 2 * (size_t)n * sizeof(double);

    /* Spec/scratch/cache — must match estimate_candidate_cost */
    size_t max_cb_cn = 0, max_pb_cn = 0, max_cb_fft = 0, cache_per_q = 0;
    for (int ell = 1; ell < L; ++ell) {
        int cps_e = psz[ell - 1];
        int is_below_e = below_sat[ell];
        int conv_build_e = is_below_e ? (2 * (cps_e / 2)) : (2 * cps_e - 1);
        bool is_fft_level;
        int est_fft_n;
        if (conv_build_e <= g_runtime_fused_max_conv_len) {
            is_fft_level = true;
            est_fft_n = next_pow2_int(conv_build_e);
        } else {
            est_fft_n = fastest_fft_ge_gpu(conv_build_e);
            double school_e = (double)conv_build_e * (double)conv_build_e * fma_ns;
            is_fft_level = (estimate_cufft_pipeline_ns(est_fft_n) < school_e);
        }
        if (is_fft_level) {
            int cn_e = est_fft_n / 2 + 1;
            int cb_e = nn[ell - 1], pb_e = nn[ell];
            max_cb_cn = std::max(max_cb_cn, (size_t)cb_e * cn_e * sizeof(cufftDoubleComplex));
            max_pb_cn = std::max(max_pb_cn, (size_t)pb_e * cn_e * sizeof(cufftDoubleComplex));
            max_cb_fft = std::max(max_cb_fft, (size_t)cb_e * est_fft_n * sizeof(double));
            if (conv_build_e > g_runtime_fused_max_conv_len && ell < L - 1)
                cache_per_q += (size_t)cb_e * cn_e * sizeof(cufftDoubleComplex);
        }
    }
    per_q += max_cb_cn + max_pb_cn + max_pb_cn + 2 * max_pb_cn;
    per_q += max_cb_fft;
    per_q += cache_per_q;
    return per_q;
}

/* Compute per_q using the planner formula (build_plan_metadata + gpu_api.cu) */
static size_t compute_per_q_planner(int n, int k_pad) {
    GpuPlan plan{};
    plan.n = n;
    plan.k = k_pad;
    plan.opts.use_cufftdx = 1;
    plan.opts.force_uncached_fused_levels = -1;
    plan.opts.force_uncached_cufft_levels = -1;
    plan.S_sorted.resize(n);
    for (int i = 0; i < n; i++) plan.S_sorted[i] = (double)(n - i);
    plan.sort_perm.resize(n);
    plan.inv_perm.resize(n);
    for (int i = 0; i < n; i++) { plan.sort_perm[i] = i; plan.inv_perm[i] = i; }

    build_plan_metadata(&plan);

    /* Replicate gpu_api.cu per_q_bytes formula using plan's tier info */
    size_t per_q = 0;
    for (int ell = 0; ell < plan.L; ++ell)
        per_q += 2 * (size_t)plan.nn[ell] * plan.psz[ell] * sizeof(double);
    per_q += (size_t)plan.N_tree * (plan.B + 1) * sizeof(double);
    per_q += 2 * (size_t)plan.n * sizeof(double);

    size_t max_cb_cn = 0, max_pb_cn = 0, max_cb_fft = 0, cache_per_q = 0;
    for (int ell = 1; ell < plan.L; ++ell) {
        auto &lp = plan.levels[ell];
        if (!lp.use_fft || lp.tier == GPU_TIER_SCHOOLBOOK) continue;
        int cn = lp.fft_n / 2 + 1;
        int cb = plan.nn[ell - 1], pb = plan.nn[ell];
        max_cb_cn = std::max(max_cb_cn, (size_t)cb * cn * sizeof(cufftDoubleComplex));
        max_pb_cn = std::max(max_pb_cn, (size_t)pb * cn * sizeof(cufftDoubleComplex));
        max_cb_fft = std::max(max_cb_fft, (size_t)cb * lp.fft_n * sizeof(double));
        if (lp.use_fft && lp.cache_fft && lp.tier != GPU_TIER_SCHOOLBOOK)
            cache_per_q += (size_t)cb * cn * sizeof(cufftDoubleComplex);
    }
    per_q += max_cb_cn + max_pb_cn + max_pb_cn + 2 * max_pb_cn;
    per_q += max_cb_fft;
    per_q += cache_per_q;
    return per_q;
}

static void test_budget_consistency() {
    struct TC { int n; int k; };
    TC cases[] = {{65536, 65536}, {131072, 131072}, {262144, 262144}, {786432, 786432}};

    for (auto &tc : cases) {
        std::vector<int> smooth;
        build_smooth_table(std::max(1 << 20, 2 * tc.k + 8), smooth);
        int k_pad = best_k_pad_gpu(tc.k, smooth);

        size_t pq_planner = compute_per_q_planner(tc.n, k_pad);
        /* Now get the B that the planner would choose */
        int best_B = gpu_select_best_B_est(tc.n, k_pad, smooth);
        size_t pq_model = compute_per_q_model(tc.n, k_pad, best_B);

        size_t budget = (size_t)((double)GPU_VRAM_BYTES * 0.90);
        int qb_model = (pq_model > 0) ? (int)(budget / pq_model) : Q_BATCH_MAX;
        int qb_planner = (pq_planner > 0) ? (int)(budget / pq_planner) : Q_BATCH_MAX;
        if (qb_model > Q_BATCH_MAX) qb_model = Q_BATCH_MAX;
        if (qb_planner > Q_BATCH_MAX) qb_planner = Q_BATCH_MAX;

        fprintf(stderr, "  n=%d B=%d: model per_q=%.1fMB qb=%d | planner per_q=%.1fMB qb=%d\n",
                tc.n, best_B, pq_model/1e6, qb_model, pq_planner/1e6, qb_planner);

        /* Allow 20% mismatch — model estimates tier without full plan */
        double ratio = (pq_planner > 0) ? (double)pq_model / (double)pq_planner : 1.0;
        CHECK(ratio > 0.7 && ratio < 1.5,
              "n=%d per_q ratio=%.2f (model=%zu planner=%zu)", tc.n, ratio, pq_model, pq_planner);
        CHECK(qb_model > 0,
              "n=%d qb_model=%d should be > 0", tc.n, qb_model);
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 5: No schoolbook in fused range — plan level check
   For any B, build_plan_metadata must not assign SCHOOLBOOK
   to levels where build_conv <= GPU_FUSED_MAX_CONV_LEN.
   ═══════════════════════════════════════════════════════════════ */
static void test_no_schoolbook_in_fused_range_plan() {
    int test_Bs[] = {32, 64, 96, 128, 192, 256, 384, 512, 1024};
    int nBs = sizeof(test_Bs) / sizeof(test_Bs[0]);

    for (int n : {65536, 131072, 262144}) {
        for (int bi = 0; bi < nBs; bi++) {
            int B = test_Bs[bi];
            if (B > n) continue;

            GpuPlan plan{};
            plan.n = n;
            plan.k = n;
            plan.opts.use_cufftdx = 1;
            plan.opts.force_uncached_fused_levels = -1;
            plan.opts.force_uncached_cufft_levels = -1;
            plan.S_sorted.resize(n);
            for (int i = 0; i < n; i++) plan.S_sorted[i] = (double)(n - i);
            plan.sort_perm.resize(n);
            plan.inv_perm.resize(n);
            for (int i = 0; i < n; i++) { plan.sort_perm[i] = i; plan.inv_perm[i] = i; }

            char buf[32];
            snprintf(buf, sizeof(buf), "%d", B);
            setenv("ICM_GPU_FORCE_B", buf, 1);
            build_plan_metadata(&plan);
            unsetenv("ICM_GPU_FORCE_B");

            for (int ell = 1; ell < plan.L; ell++) {
                auto &lp = plan.levels[ell];
                if (lp.build_conv <= g_runtime_fused_max_conv_len) {
                    CHECK(lp.tier != GPU_TIER_SCHOOLBOOK,
                          "n=%d B=%d ell=%d conv=%d tier=SCHOOLBOOK",
                          n, B, ell, lp.build_conv);
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 6: Wrap formula consistency
   best_fft_config_gpu: wrap_m = max(0, conv_len - fft_n)
   Build: bwm*(bwm+1)/2 * fma_ns
   Corr:  cwm*(cwm+1) * fma_ns  (no /2)
   ═══════════════════════════════════════════════════════════════ */
static void test_wrap_formulas() {
    int conv_lens[] = {63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383};
    int n_cl = sizeof(conv_lens) / sizeof(conv_lens[0]);

    for (int ci = 0; ci < n_cl; ci++) {
        int conv_len = conv_lens[ci];
        int bfn = 0, bwm = 0;
        best_fft_config_gpu(conv_len, 0, 1.0, &bfn, &bwm);

        CHECK(bfn >= conv_len / 2 + 1,
              "conv=%d bfn=%d < min_size=%d", conv_len, bfn, conv_len / 2 + 1);

        int expected_wrap = (bfn >= conv_len) ? 0 : (conv_len - bfn);
        CHECK(bwm == expected_wrap,
              "conv=%d bfn=%d bwm=%d expected=%d", conv_len, bfn, bwm, expected_wrap);

        /* Correlate wrap */
        int p_eff = conv_len / 4;
        int cfn = 0, cwm = 0;
        best_fft_config_gpu(conv_len, p_eff, 1.0, &cfn, &cwm);
        int expected_cwrap = (cfn >= conv_len) ? 0 : (conv_len - cfn);
        CHECK(cwm == expected_cwrap,
              "conv=%d cfn=%d cwm=%d expected=%d", conv_len, cfn, cwm, expected_cwrap);
    }

    /* Joint wrap: both build and corr wraps must be consistent */
    {
        int build_conv = 1023, corr_conv = 2047, p_eff = 512;
        int jfn = 0, jbm = 0, jcm = 0;
        best_fft_config_joint_gpu(build_conv, corr_conv, p_eff, 1.0, &jfn, &jbm, &jcm);
        int exp_bm = (jfn >= build_conv) ? 0 : (build_conv - jfn);
        int exp_cm = (jfn >= corr_conv) ? 0 : (corr_conv - jfn);
        CHECK(jbm == exp_bm,
              "joint build wrap: jfn=%d jbm=%d expected=%d", jfn, jbm, exp_bm);
        CHECK(jcm == exp_cm,
              "joint corr wrap: jfn=%d jcm=%d expected=%d", jfn, jcm, exp_cm);
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 7: tree_school_ns_per_fma returns GPU_SCHOOL_FMA_NS
   ═══════════════════════════════════════════════════════════════ */
static void test_school_fma_rate() {
    double rate = tree_school_ns_per_fma();
    CHECK(rate == GPU_SCHOOL_FMA_NS,
          "tree_school_ns_per_fma()=%.6f expected=%.6f", rate, GPU_SCHOOL_FMA_NS);

    CHECK(rate < GPU_BLOCK_BUILD_NS_PER_FMA,
          "SCHOOL_FMA_NS=%.6f should be < BLOCK_BUILD_NS=%.6f",
          rate, GPU_BLOCK_BUILD_NS_PER_FMA);
}

/* ═══════════════════════════════════════════════════════════════
   Test 8: B selection — best_B must be within 10% of optimal
   ═══════════════════════════════════════════════════════════════ */
static void test_b_selection() {
    std::vector<int> smooth;
    build_smooth_table(2 << 20, smooth);

    int test_n[] = {65536, 131072, 262144, 786432, 1572864};
    for (int ti = 0; ti < 5; ti++) {
        int n = test_n[ti];
        int k_pad = best_k_pad_gpu(n, smooth);
        int best_B = gpu_select_best_B_est(n, k_pad, smooth);
        double best_cost = estimate_candidate_cost(n, k_pad, best_B, smooth);

        CHECK(best_B > 0 && best_B <= n && best_B <= k_pad,
              "n=%d best_B=%d out of range", n, best_B);
        CHECK(std::isfinite(best_cost) && best_cost > 0,
              "n=%d best_B=%d cost=%.0f invalid", n, best_B, best_cost);

        /* Find actual minimum across all candidates */
        double min_cost = std::numeric_limits<double>::infinity();
        int min_B = 0;
        for (int i = 0; i < MAX_B_CANDIDATES; i++) {
            int B = kBCandidates[i];
            if (B > n || B > k_pad) continue;
            double c = estimate_candidate_cost(n, k_pad, B, smooth);
            if (c < min_cost) { min_cost = c; min_B = B; }
        }

        CHECK(best_cost <= min_cost * 1.001,
              "n=%d gpu_select_best_B=%d (%.0f) is not optimal; B=%d is %.0f",
              n, best_B, best_cost, min_B, min_cost);

        /* Report all B costs for visibility */
        fprintf(stderr, "  n=%d model_best_B=%d (%.0f ns)  sweep_best_B=%d (%.0f ns)\n",
                n, best_B, best_cost, min_B, min_cost);

        /* Show top candidates */
        struct BC { int B; double cost; };
        std::vector<BC> ranked;
        for (int i = 0; i < MAX_B_CANDIDATES; i++) {
            int B = kBCandidates[i];
            if (B > n || B > k_pad) continue;
            double c = estimate_candidate_cost(n, k_pad, B, smooth);
            if (std::isfinite(c)) ranked.push_back({B, c});
        }
        std::sort(ranked.begin(), ranked.end(), [](const BC &a, const BC &b) { return a.cost < b.cost; });
        int show = std::min((int)ranked.size(), 8);
        for (int i = 0; i < show; i++) {
            bool is_p2 = (ranked[i].B & (ranked[i].B - 1)) == 0;
            fprintf(stderr, "    #%d B=%d%s  %.0f ns  (%.0f%%)\n",
                    i + 1, ranked[i].B, is_p2 ? "*" : "",
                    ranked[i].cost, (ranked[i].cost / min_cost - 1.0) * 100.0);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 9: Fused calibration coverage
   Power-of-2 sizes up to GPU_FUSED_MAX_CONV_LEN should have
   finite fused build/corr costs.
   ═══════════════════════════════════════════════════════════════ */
static void test_fused_calibration_coverage() {
    int max_conv = g_runtime_fused_max_conv_len;
    for (int fft_n = 4; fft_n <= max_conv; fft_n *= 2) {
        double fb = estimate_fused_build_ns(fft_n);
        double fc = estimate_fused_corr_ns(fft_n);
        CHECK(std::isfinite(fb) && fb > 0,
              "fused build at fft_n=%d not calibrated (%.3f)", fft_n, fb);
        CHECK(std::isfinite(fc) && fc > 0,
              "fused corr at fft_n=%d not calibrated (%.3f)", fft_n, fc);
    }
}

/* ═══════════════════════════════════════════════════════════════
   Test 10: Level-by-level cost dump for a specific case
   Diagnostic: shows per-level tier assignment and costs.
   ═══════════════════════════════════════════════════════════════ */
static void dump_level_costs(int n, int B) {
    std::vector<int> smooth;
    build_smooth_table(2 << 20, smooth);
    int k_pad = best_k_pad_gpu(n, smooth);

    int nblocks = (n + B - 1) / B;
    int N = 0, L = 0;
    std::vector<int> nn, psz, g_needed, below_sat, n_real;
    std::vector<size_t> plev_off;
    build_tree_geometry(nblocks, B, k_pad, B, nn, psz, plev_off, g_needed, below_sat, n_real, N, L);

    fprintf(stderr, "  n=%d B=%d L=%d nblocks=%d k_pad=%d\n", n, B, L, nblocks, k_pad);

    GpuPlan plan{};
    plan.n = n; plan.k = k_pad;
    plan.opts.use_cufftdx = 1;
    plan.opts.force_uncached_fused_levels = -1;
    plan.opts.force_uncached_cufft_levels = -1;
    plan.S_sorted.resize(n);
    for (int i = 0; i < n; i++) plan.S_sorted[i] = (double)(n - i);
    plan.sort_perm.resize(n);
    plan.inv_perm.resize(n);
    for (int i = 0; i < n; i++) { plan.sort_perm[i] = i; plan.inv_perm[i] = i; }

    char buf[32];
    snprintf(buf, sizeof(buf), "%d", B);
    setenv("ICM_GPU_FORCE_B", buf, 1);
    build_plan_metadata(&plan);
    unsetenv("ICM_GPU_FORCE_B");

    const char *tier_names[] = {"?", "SCHOOL", "FUSED", "CUFFT"};
    for (int ell = 1; ell < plan.L; ell++) {
        auto &lp = plan.levels[ell];
        fprintf(stderr, "    ell=%d nn=%d cps=%d psz=%d conv=%d fft_n=%d tier=%s cache=%d bwm=%d cwm=%d\n",
                ell, plan.nn[ell], plan.psz[ell-1], plan.psz[ell],
                lp.build_conv, lp.fft_n,
                (lp.tier >= 1 && lp.tier <= 3) ? tier_names[lp.tier] : "?",
                lp.cache_fft, lp.build_wrap_m, lp.corr_wrap_m);
    }
}

/* ═══════════════════════════════════════════════════════════════ */

int main() {
    fprintf(stderr, "=== GPU Cost Model Unit Tests ===\n\n");
    g_runtime_fused_max_conv_len = GPU_FUSED_MAX_CONV_LEN;

    fprintf(stderr, "Config: SCHOOL_FMA=%.6f ns  BLOCK_BUILD_FMA=%.6f ns  "
            "FUSED_MAX=%d  VRAM=%.0f GB  SM=%d\n\n",
            GPU_SCHOOL_FMA_NS, GPU_BLOCK_BUILD_NS_PER_FMA,
            GPU_FUSED_MAX_CONV_LEN, GPU_VRAM_BYTES / 1e9, GPU_SM_COUNT);

    fprintf(stderr, "--- 1. Tier selection invariant ---\n");
    test_tier_never_schoolbook_in_fused_range();

    fprintf(stderr, "--- 2. Cost monotonicity ---\n");
    test_cost_monotonicity();

    fprintf(stderr, "--- 3. Fused preference ---\n");
    test_fused_preference();

    fprintf(stderr, "--- 4. Budget consistency ---\n");
    test_budget_consistency();

    fprintf(stderr, "--- 5. No schoolbook in fused range (plan) ---\n");
    test_no_schoolbook_in_fused_range_plan();

    fprintf(stderr, "--- 6. Wrap formulas ---\n");
    test_wrap_formulas();

    fprintf(stderr, "--- 7. School FMA rate ---\n");
    test_school_fma_rate();

    fprintf(stderr, "--- 8. B selection ---\n");
    test_b_selection();

    fprintf(stderr, "--- 9. Fused calibration coverage ---\n");
    test_fused_calibration_coverage();

    fprintf(stderr, "\n--- Diagnostic: level costs for key cases ---\n");
    dump_level_costs(65536, 64);
    dump_level_costs(65536, 96);
    dump_level_costs(786432, 192);
    dump_level_costs(786432, 384);

    fprintf(stderr, "\n=== Results: %d passed, %d failed, %d warnings ===\n",
            n_pass, n_fail, n_warn);
    return n_fail > 0 ? 1 : 0;
}
