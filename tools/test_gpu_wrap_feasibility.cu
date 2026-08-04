/* tools/test_gpu_wrap_feasibility.cu -- GPU correlate FFT-size feasibility.
 *
 * The GPU half of VERDICTS.md V20; the CPU half is
 * tools/test_wrap_feasibility.c.
 *
 * INVARIANT UNDER TEST
 *
 *   Every correlate path loads the whole g operand into an fft_n-sized
 *   transform. cufftdx_load_real() / cufftdx_load_real_r2c()
 *   (src/gpu/gpu_kernels.cu) copy src[idx] only for idx < fft_n and
 *   zero-fill the rest, so an fft_n below len_g does not fault, overrun, or
 *   fail -- it silently DROPS g[fft_n .. len_g-1] while k_wrap_corr_pair()
 *   goes on correcting for the cyclic aliasing of a g that fully fit.
 *   Wrong answers, no diagnostic. Hence:
 *
 *       fft_n >= len_g   <=>   corr_wrap_m < len_P
 *
 *   must hold at every FFT level of every plan. It is a hard feasibility
 *   constraint, never a cost term.
 *
 * WHY A DEDICATED TEST
 *
 *   The bug only surfaces when real calibration timings make an infeasible
 *   candidate the cost winner, so it cannot be caught by inspection of the
 *   cost model, and a passing verify on one device's data says nothing about
 *   another's. Test 3 therefore re-plans a grid of (n,k) against whatever
 *   calibration data the build is using and audits every level, which turns
 *   any future recalibration into a build-time-checkable event.
 *
 * NO GPU REQUIRED. Every function exercised here is pure host-side math:
 * no kernel launches, no device allocation, no CUDA context. It needs nvcc
 * to compile (it includes gpu_internal.h) but runs anywhere, including a
 * machine with no NVIDIA device present.
 *
 * Build: make test_gpu_wrap_feasibility CUDA_ARCH=sm_100 CUFFTDX_INC=-I...
 * Run:   ./test_gpu_wrap_feasibility
 */
#include "gpu_internal.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace icm_gpu_detail;

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: " __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        n_fail++; \
    } else { \
        n_pass++; \
    } \
} while (0)

/* ═══════════════════════════════════════════════════════════════
   Test 1: best_fft_config_gpu() never returns an infeasible correlate
   config.

   Swept over shapes wide enough that, for some of them, a smaller
   transform plus a large wrap correction really is the cheaper choice
   under the calibration data -- which is exactly the situation that
   produced the bug.
   ═══════════════════════════════════════════════════════════════ */
static void test_single_chooser(void) {
    printf("Test 1: best_fft_config_gpu() correlate feasibility\n");
    int checked = 0;
    for (int len_P = 2; len_P <= 4096; len_P = (len_P * 3) / 2 + 1) {
        for (int len_g = len_P; len_g <= 65536; len_g = (len_g * 3) / 2 + 1) {
            int conv = len_g + len_P - 1;
            for (double scale = 1.0; scale <= 256.0; scale *= 16.0) {
                int fft_n = 0, wrap_m = 0;
                best_fft_config_gpu(conv, len_P, scale, &fft_n, &wrap_m);
                ++checked;
                CHECK(fft_n >= len_g,
                      "len_g=%d len_P=%d scale=%.0f -> fft_n=%d < len_g (g truncated)",
                      len_g, len_P, scale, fft_n);
                CHECK(corr_wrap_feasible(wrap_m, len_P),
                      "len_g=%d len_P=%d scale=%.0f -> wrap_m=%d >= len_P",
                      len_g, len_P, scale, wrap_m);
            }
        }
    }
    printf("  %d shapes checked\n", checked);
}

/* ═══════════════════════════════════════════════════════════════
   Test 2: build mode (len_P == 0) is NOT constrained.

   The build path is a pure convolution whose wrap terms are read from the
   original operands, so any wrap is feasible there. Over-constraining it
   would be a silent performance regression, so pin the asymmetry: at least
   one build-mode shape must still choose a nonzero wrap.
   ═══════════════════════════════════════════════════════════════ */
static void test_build_mode_unconstrained(void) {
    printf("Test 2: build mode (len_P=0) stays unconstrained\n");
    int nonzero_wrap_seen = 0, checked = 0;
    for (int conv = 8; conv <= 200000; conv = (conv * 3) / 2 + 1) {
        int fft_n = 0, wrap_m = 0;
        best_fft_config_gpu(conv, 0, 1.0, &fft_n, &wrap_m);
        ++checked;
        CHECK(fft_n > 0, "conv=%d -> fft_n=%d", conv, fft_n);
        CHECK(wrap_m == 0 || fft_n + wrap_m >= conv,
              "conv=%d -> fft_n=%d wrap_m=%d inconsistent", conv, fft_n, wrap_m);
        if (wrap_m > 0) ++nonzero_wrap_seen;
    }
    CHECK(nonzero_wrap_seen > 0,
          "no build-mode shape chose a nonzero wrap: the build path looks "
          "over-constrained (it must NOT inherit the correlate bound)");
    printf("  %d shapes checked, %d chose a nonzero wrap\n", checked, nonzero_wrap_seen);
}

/* ═══════════════════════════════════════════════════════════════
   Test 3: joint chooser + fused power-of-2 override.

   fused_p2_feasible() is what makes the fused candidate safe. On the
   shipped B200 data the fused override was the ONLY producer of infeasible
   plans: both searches returned feasible sizes and the override then
   replaced them with a smaller power of two whose wrap ran past len_g.
   ═══════════════════════════════════════════════════════════════ */
static void test_joint_and_fused(void) {
    printf("Test 3: joint chooser and fused power-of-2 override\n");
    int checked = 0;
    for (int p_eff = 2; p_eff <= 4096; p_eff = (p_eff * 3) / 2 + 1) {
        for (int len_g = p_eff; len_g <= 65536; len_g = (len_g * 3) / 2 + 1) {
            int corr_conv = len_g + p_eff - 1;
            for (int build_conv = p_eff; build_conv <= 2 * len_g;
                 build_conv = (build_conv * 5) / 2 + 1) {
                int fft_n = 0, bwm = 0, cwm = 0;
                best_fft_config_joint_gpu(build_conv, corr_conv, p_eff, 1.0,
                                          &fft_n, &bwm, &cwm);
                ++checked;
                CHECK(fft_n >= len_g,
                      "joint: build_conv=%d len_g=%d p_eff=%d -> fft_n=%d < len_g",
                      build_conv, len_g, p_eff, fft_n);
                CHECK(corr_wrap_feasible(cwm, p_eff),
                      "joint: build_conv=%d len_g=%d p_eff=%d -> cwm=%d >= p_eff",
                      build_conv, len_g, p_eff, cwm);

                int p2 = fused_p2_feasible(build_conv, corr_conv, p_eff);
                CHECK(p2 >= len_g,
                      "fused p2: build_conv=%d len_g=%d p_eff=%d -> p2=%d < len_g",
                      build_conv, len_g, p_eff, p2);
                CHECK(p2 >= build_conv,
                      "fused p2: build_conv=%d -> p2=%d < build_conv", build_conv, p2);
                CHECK((p2 & (p2 - 1)) == 0, "fused p2=%d is not a power of two", p2);
            }
        }
    }
    printf("  %d shapes checked\n", checked);
}

/* ═══════════════════════════════════════════════════════════════
   Test 4: whole-plan audit against this build's real calibration data.

   The regression cells are the ones the shipped B200 table actually got
   wrong before the fix. n=131072/k=1024, n=262144/k=2048 and
   n=1048576/k=32768 are clean powers of two inside the reported heatmap
   range -- this was never an exotic-corner bug.
   ═══════════════════════════════════════════════════════════════ */
static void audit_plan(int n, int k, int *levels_checked) {
    GpuPlan plan;
    plan.n = n;
    plan.k = k;
    plan.q_batch = 64;
    plan.opts.use_cufftdx = 1;
    plan.opts.force_uncached_fused_levels = -1;
    plan.opts.force_uncached_cufft_levels = -1;
    plan.S_sorted.assign(n, 1.0);
    if (!build_plan_metadata(&plan)) {
        fprintf(stderr, "  FAIL: build_plan_metadata failed for n=%d k=%d\n", n, k);
        n_fail++;
        return;
    }
    for (int ell = 1; ell < plan.L; ++ell) {
        const GpuLevelPlan &lp = plan.levels[ell];
        if (!lp.use_fft) continue;
        ++(*levels_checked);
        CHECK(lp.fft_n >= lp.g_eff,
              "n=%d k=%d B=%d ell=%d: fft_n=%d < g_eff=%d (corr_wrap_m=%d, p_eff=%d) "
              "-- g truncated",
              n, k, plan.B, ell, lp.fft_n, lp.g_eff, lp.corr_wrap_m, lp.p_eff);
        CHECK(corr_wrap_feasible(lp.corr_wrap_m, lp.p_eff),
              "n=%d k=%d ell=%d: corr_wrap_m=%d >= p_eff=%d",
              n, k, ell, lp.corr_wrap_m, lp.p_eff);
        CHECK(lp.fft_n >= lp.p_eff,
              "n=%d k=%d ell=%d: fft_n=%d < p_eff=%d (P truncated)",
              n, k, ell, lp.fft_n, lp.p_eff);
        /* Build operand: cufftdx_load_real(L, cps, a) copies cps reals. */
        int cps = plan.psz[ell - 1];
        int build_eff = plan.below_sat[ell] ? (cps / 2 + 1) : cps;
        CHECK(lp.fft_n >= build_eff,
              "n=%d k=%d ell=%d: fft_n=%d < build operand %d",
              n, k, ell, lp.fft_n, build_eff);
    }
}

static void test_plan_sweep(void) {
    printf("Test 4: whole-plan audit over an (n,k) grid\n");
    int levels = 0, plans = 0;

    /* Cells that failed before the fix, kept as explicit regressions. */
    static const int regress[][2] = {
        {39224, 709}, {131072, 1024}, {132385, 1064}, {198578, 1597},
        {262144, 2048}, {297868, 2396}, {1048576, 32768}, {1005308, 65536},
    };
    for (size_t i = 0; i < sizeof(regress) / sizeof(regress[0]); ++i) {
        audit_plan(regress[i][0], regress[i][1], &levels);
        ++plans;
    }

    for (int n = 16; n <= 4194304; n = (int)(n * 1.5) + 1) {
        for (int k = 2; k <= 65536; k = (int)(k * 1.5) + 1) {
            if (k > n) continue;
            audit_plan(n, k, &levels);
            ++plans;
        }
    }
    for (int pn = 4; pn <= 22; ++pn) {
        for (int pk = 1; pk <= 16; ++pk) {
            int n = 1 << pn, k = 1 << pk;
            if (k > n) continue;
            audit_plan(n, k, &levels);
            ++plans;
        }
    }
    printf("  %d plans, %d FFT levels audited\n", plans, levels);
}

int main(void) {
    printf("GPU correlate wrap-feasibility tests\n");
    printf("calibration: %d FFT sizes, fused ceiling %d\n\n",
           GPU_N_CALIBRATED_SIZES, GPU_FUSED_MAX_CONV_LEN);

    g_runtime_fused_max_conv_len = fused_max_conv_len_runtime();

    test_single_chooser();
    test_build_mode_unconstrained();
    test_joint_and_fused();
    test_plan_sweep();

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    if (n_fail == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    }
    return 1;
}
