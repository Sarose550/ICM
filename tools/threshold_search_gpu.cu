/* threshold_search_gpu.cu — Binary search for the exact n where GPU
 * runtime crosses 1000ms, using the lightweight single-measurement
 * approach from frontier_probe.cu (one icm_gpu_equity_with_plan call per
 * candidate n), NOT the exhaustive B/M/T hyperparameter grid search from
 * push_limit_gpu.cu.
 *
 * Two independent binary searches:
 *   Curve 1: k = n  (full-equity-like, k scales with n)
 *   Curve 2: k = 100 (fixed small k, the "1-second threshold" curve from
 *                      RESULTS.md)
 *
 * Each candidate n is measured via median of N_REPS=5 repetitions.
 * Output: per-candidate log lines as the search narrows, then the final
 * threshold n for each curve.
 *
 * Build (B200, cuFFTDx-enabled — matches Makefile's push_limit_gpu target):
 *
 *   # Pre-flight: the fused GPU objects + device-link object must already
 *   # be built (e.g. via 'make bench_gpu_fused' first).  Then:
 *
 *   CUFFTDX_INC=$(find /usr/local/lib -maxdepth 4 -type d -path '*dist-packages/nvidia/mathdx/include' | head -1)
 *
 *   nvcc -O3 -std=c++17 -arch=sm_100 -Isrc/cpu -Idevices/b200 -Isrc/gpu \
 *        -I"$CUFFTDX_INC" -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX \
 *        -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY \
 *        -dc -o build/threshold_search_gpu.o tools/threshold_search_gpu.cu
 *
 *   nvcc -O3 -std=c++17 -arch=sm_100 \
 *        -o threshold_search_gpu build/threshold_search_gpu.o \
 *        build/gpu_kernels_fused.o build/gpu_plan_fused.o \
 *        build/gpu_exec_fused.o build/gpu_api_fused.o \
 *        build/gpu_dlink_fused.o -lcufft -lcudart
 *
 *   # If VKFFT is enabled, append: -lnvrtc -lcuda
 *
 * Architecture note: -arch=sm_100 targets B200 (Blackwell).  Verify
 * against the Makefile's CUDA_ARCH default for your target device.
 *
 * Runtime expectation: ~2–4 minutes on a B200 (roughly 20–30 candidate
 * points × 5 reps × ~2s worst-case wall time per rep).
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "icm_gpu.h"

/* ── helpers ─────────────────────────────────────────────────────── */

static void make_stacks(int n, std::vector<double> &S) {
    S.resize(n);
    srand(123 + n);
    for (int i = 0; i < n; ++i)
        S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}

static void make_payout(int n, int k, std::vector<double> &payout) {
    payout.resize(k);
    for (int m = 0; m < k; ++m)
        payout[m] = (double)(n - m);
}

static double median_ms(std::vector<double> &x) {
    if (x.empty()) return NAN;
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

/* ── single-point measurement ───────────────────────────────────── */

#define N_REPS  5
#define Q_POINTS 256

/* Measure median runtime in milliseconds for one (n,k) pair.
 * Returns the median ms on success, or a negative value on failure
 * (plan creation failure, execution error, etc.).  The caller should
 * treat negative returns as "untestable at this n" (e.g. OOM). */
static double measure_ms(int n, int k) {
    std::vector<double> S, payout, eq;
    make_stacks(n, S);
    make_payout(n, k, payout);
    eq.assign(n, 0.0);

    IcmGpuOptions opts{};
    opts.device_id                 = 0;
    opts.use_cufftdx               = 1;
    opts.enable_graphs             = 0;
    opts.enable_q_pipeline         = 1;
    opts.memory_strategy           = 0;
    opts.force_uncached_fused_levels  = -1;
    opts.force_uncached_cufft_levels  = -1;

    IcmGpuPlan *plan = icm_gpu_plan_create(n, S.data(), k, &opts);
    if (!plan) {
        /* OOM or invalid config — treat as "over threshold" so the binary
         * search backs off to a smaller n. */
        fprintf(stderr, "  [measure_ms] n=%d k=%d plan_create FAIL: %s\n",
                n, k, icm_gpu_last_error());
        /* Attempt recovery: release pooled memory and re-init, in case
         * fragmentation from prior measurements caused the failure. */
        icm_gpu_release_pooled_memory();
        return -1.0;
    }

    /* Warmup — one call to amortise any lazy allocation / JIT overhead */
    IcmGpuRunStats warm_stats{};
    int warm_ok = icm_gpu_equity_with_plan(plan, Q_POINTS,
                                            payout.data(), eq.data(),
                                            &warm_stats);
    if (warm_ok != 0) {
        fprintf(stderr, "  [measure_ms] n=%d k=%d warmup FAIL: %s\n",
                n, k, icm_gpu_last_error());
        icm_gpu_plan_destroy(plan);
        icm_gpu_release_pooled_memory();
        return -1.0;
    }

    /* Timed reps */
    std::vector<double> samples;
    samples.reserve(N_REPS);
    for (int r = 0; r < N_REPS; ++r) {
        IcmGpuRunStats stats{};
        int status = icm_gpu_equity_with_plan(plan, Q_POINTS,
                                               payout.data(), eq.data(),
                                               &stats);
        if (status != 0) {
            fprintf(stderr, "  [measure_ms] n=%d k=%d rep %d FAIL: %s\n",
                    n, k, r, icm_gpu_last_error());
            samples.clear();
            break;
        }
        samples.push_back(stats.total_ns / 1e6);
    }

    icm_gpu_plan_destroy(plan);
    icm_gpu_release_pooled_memory();

    if (samples.empty()) return -1.0;
    return median_ms(samples);
}

/* ── binary search over n ───────────────────────────────────────── */

#define MAX_ITER 20
#define MIN_STEP 16384

/* Run one binary search for a given k-strategy.
 *   k_mode: 0 → k = n      1 → k = 100
 * Prints progress to stdout; returns the threshold n (largest n with
 * median runtime ≤ 1000 ms).  Returns -1 if the search couldn't even
 * establish a bracket. */
static int binary_search_threshold(int k_mode) {
    const char *label = (k_mode == 0) ? "k=n" : "k=100";

    printf("\n=== Binary search: %s ===\n", label);
    printf("%-12s %-8s %-12s %s\n", "n", "k", "median_ms", "verdict");
    printf("----------------------------------------\n");

    /* ── establish bracket ── */

    int lo, hi;
    if (k_mode == 0) {
        lo = 262144;   /* 256K — should be well under 1s */
        hi = 4194304;  /* 4M   — should be over 1s at this n */
    } else {
        lo = 1048576;   /* 1M */
        hi = 16777216;  /* 16M */
    }

    /* Verify lo is under 1000ms */
    int k_lo = (k_mode == 0) ? lo : 100;
    double t_lo = measure_ms(lo, k_lo);
    if (t_lo < 0.0) {
        fprintf(stderr, "FATAL: cannot measure lo=%d for %s\n", lo, label);
        return -1;
    }
    printf("%-12d %-8d %-12.3f %s\n", lo, k_lo, t_lo,
           (t_lo <= 1000.0) ? "BELOW" : "ABOVE");
    fflush(stdout);

    if (t_lo > 1000.0) {
        /* Even the smallest n is over 1s — halve until we go under */
        while (lo > 8192 && t_lo > 1000.0) {
            lo /= 2;
            k_lo = (k_mode == 0) ? lo : 100;
            t_lo = measure_ms(lo, k_lo);
            if (t_lo < 0.0) { lo *= 2; break; }  /* back off */
            printf("%-12d %-8d %-12.3f %s\n", lo, k_lo, t_lo,
                   (t_lo <= 1000.0) ? "BELOW" : "ABOVE");
            fflush(stdout);
        }
        if (t_lo > 1000.0) {
            fprintf(stderr, "FATAL: even n=%d is over 1000ms for %s\n",
                    lo, label);
            return -1;
        }
    }

    /* Verify hi is over 1000ms */
    int k_hi = (k_mode == 0) ? hi : 100;
    double t_hi = measure_ms(hi, k_hi);
    if (t_hi < 0.0) {
        /* hi OOMed — halve until it works */
        while (hi > lo && t_hi < 0.0) {
            hi /= 2;
            k_hi = (k_mode == 0) ? hi : 100;
            t_hi = measure_ms(hi, k_hi);
        }
        if (t_hi < 0.0) {
            fprintf(stderr, "FATAL: cannot find measurable hi for %s\n", label);
            return -1;
        }
    }
    printf("%-12d %-8d %-12.3f %s\n", hi, k_hi, t_hi,
           (t_hi <= 1000.0) ? "BELOW" : "ABOVE");
    fflush(stdout);

    if (t_hi <= 1000.0) {
        /* Even the largest testable n is under 1s — double until over */
        while (hi < 134217728 && t_hi <= 1000.0) {  /* cap at 128M */
            hi *= 2;
            k_hi = (k_mode == 0) ? hi : 100;
            t_hi = measure_ms(hi, k_hi);
            if (t_hi < 0.0) { hi /= 2; break; }  /* OOM — back off */
            printf("%-12d %-8d %-12.3f %s\n", hi, k_hi, t_hi,
                   (t_hi <= 1000.0) ? "BELOW" : "ABOVE (expanding)");
            fflush(stdout);
        }
        if (t_hi <= 1000.0) {
            printf("\n*** %s: entire tested range under 1000ms (hi=%d, %.3f ms) ***\n",
                   label, hi, t_hi);
            printf("    Reporting hi as lower-bound threshold.\n");
            return hi;
        }
    }

    /* ── binary search ── */
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        if (hi - lo <= MIN_STEP) break;

        int mid = lo + (hi - lo) / 2;
        int k_mid = (k_mode == 0) ? mid : 100;

        double t_mid = measure_ms(mid, k_mid);
        if (t_mid < 0.0) {
            /* Measurement failed (likely OOM) — treat as "above
             * threshold" so we back off to a smaller n. */
            fprintf(stderr, "  [search] n=%d FAIL, treating as ABOVE\n", mid);
            hi = mid;
            continue;
        }

        const char *verdict;
        if (t_mid <= 1000.0) {
            verdict = "BELOW";
            lo = mid;
        } else {
            verdict = "ABOVE";
            hi = mid;
        }

        printf("%-12d %-8d %-12.3f %-6s [lo=%d hi=%d]\n",
               mid, k_mid, t_mid, verdict, lo, hi);
        fflush(stdout);
    }

    /* ── final answer ── */
    /* lo is the largest n known to be ≤ 1000ms */
    printf("\n--- %s threshold ---\n", label);
    printf("Largest n ≤ 1000ms: %d\n", lo);
    printf("Smallest n > 1000ms: %d\n", hi);
    printf("Search bracket width: %d (%.1f%% of lo)\n",
           hi - lo, 100.0 * (hi - lo) / (double)lo);

    return lo;
}

/* ── main ───────────────────────────────────────────────────────── */

int main() {
    if (!icm_gpu_init(0)) {
        fprintf(stderr, "icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    printf("threshold_search_gpu — binary search for 1000ms crossing\n");
    printf("method: median of %d reps per candidate n, Q=%d\n",
           N_REPS, Q_POINTS);
    printf("plan-based API (icm_gpu_plan_create + icm_gpu_equity_with_plan)\n");

    int threshold_kn = binary_search_threshold(0);   /* k=n */
    int threshold_k100 = binary_search_threshold(1); /* k=100 */

    printf("\n========================================\n");
    printf("FINAL RESULTS:\n");
    if (threshold_kn >= 0)
        printf("  k=n   threshold n = %d\n", threshold_kn);
    else
        printf("  k=n   threshold: SEARCH FAILED\n");
    if (threshold_k100 >= 0)
        printf("  k=100 threshold n = %d\n", threshold_k100);
    else
        printf("  k=100 threshold: SEARCH FAILED\n");
    printf("========================================\n");

    icm_gpu_shutdown();
    return 0;
}
