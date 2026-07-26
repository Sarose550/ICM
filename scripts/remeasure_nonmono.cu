/* remeasure_nonmono.cu — targeted re-measurement of the confirmed-bad
 * heatmap cell (n=4194304, k=128, 893.3ms/reps=1 in results/gpu_heatmap_b200.csv)
 * plus two lower-priority suspects and their clean neighbors, at reps=5
 * each, to distinguish single-rep measurement noise from a real
 * structural issue.  See HANDOFF.md "Non-monotonicity data-quality gap"
 * for the full root-cause analysis this tool is meant to settle.
 *
 * Build (B200, cuFFTDx-enabled — same recipe as scripts/threshold_search_gpu.cu):
 *
 *   CUFFTDX_INC=$(find /usr/local/lib -maxdepth 4 -type d -path '*dist-packages/nvidia/mathdx/include' | head -1)
 *
 *   nvcc -O3 -std=c++17 -arch=sm_100 -Isrc -Idevices/b200 -Isrc/gpu \
 *        -I"$CUFFTDX_INC" -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX \
 *        -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY \
 *        -dc -o build/remeasure_nonmono.o scripts/remeasure_nonmono.cu
 *
 *   nvcc -O3 -std=c++17 -arch=sm_100 \
 *        -o remeasure_nonmono build/remeasure_nonmono.o \
 *        build/gpu_kernels_fused.o build/gpu_plan_fused.o \
 *        build/gpu_exec_fused.o build/gpu_api_fused.o \
 *        build/gpu_dlink_fused.o -lcufft -lcudart
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "icm_gpu.h"

#define N_REPS   5
#define Q_POINTS 256

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

static double median_of(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

static double mean_of(const std::vector<double> &x) {
    double s = 0.0;
    for (double v : x) s += v;
    return s / (double)x.size();
}

static double cv_of(const std::vector<double> &x) {
    if (x.size() < 2) return 0.0;
    double m = mean_of(x);
    if (m <= 0.0) return 0.0;
    double var = 0.0;
    for (double v : x) var += (v - m) * (v - m);
    var /= (double)(x.size() - 1);
    return std::sqrt(var) / m;
}

struct Point { int n, k; const char *tag; };

/* The three flagged cells (mandatory + two lower-priority) plus clean
 * neighbors, per HANDOFF.md's investigation plan. Plus one generalization
 * check (n=524288,k=128/256): the blast-radius simulation identified this
 * as the same bug signature, and it was already a multi-rep (reps=3),
 * low-cv measurement in results/gpu_heatmap_b200.csv (54.17ms vs 51.80ms
 * for k=256 -- a real, reproducible inversion, not single-rep noise),
 * so a fix here should show a real, attributable improvement. */
static const Point kPoints[] = {
    {4194304, 128, "BAD-mandatory (was 893.3ms, expected ~500-650ms)"},
    {4194304, 64,  "neighbor (was 273ms)"},
    {4194304, 256, "neighbor (was 613ms)"},
    {2097152, 128, "neighbor (was 256ms)"},
    {8388608, 128, "BAD-suspect (was 1349.8ms)"},
    {1048576, 128, "BAD-suspect (was 122.6ms)"},
    {524288,  128, "GENERALIZATION-check (was 54.17ms, k=256 neighbor 51.80ms)"},
    {524288,  256, "GENERALIZATION-check neighbor (was 51.80ms)"},
};
static const int kNumPoints = sizeof(kPoints) / sizeof(kPoints[0]);

int main() {
    if (!icm_gpu_init(0)) {
        fprintf(stderr, "icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    printf("remeasure_nonmono — reps=%d per cell, Q=%d\n\n", N_REPS, Q_POINTS);
    printf("%-10s %-8s %-30s %-10s %-8s %s\n",
           "n", "k", "tag", "median_ms", "cv", "samples_ms");

    for (int p = 0; p < kNumPoints; ++p) {
        int n = kPoints[p].n, k = kPoints[p].k;

        std::vector<double> S, payout, eq;
        make_stacks(n, S);
        make_payout(n, k, payout);
        eq.assign(n, 0.0);

        IcmGpuOptions opts{};
        opts.device_id           = 0;
        opts.use_cufftdx         = 1;
        opts.enable_graphs       = 0;
        opts.enable_q_pipeline   = 1;
        opts.memory_strategy     = 0;
        opts.force_uncached_fused_levels = -1;
        opts.force_uncached_cufft_levels = -1;

        IcmGpuPlan *plan = icm_gpu_plan_create(n, S.data(), k, &opts);
        if (!plan) {
            printf("%-10d %-8d %-30s PLAN_CREATE_FAIL: %s\n",
                   n, k, kPoints[p].tag, icm_gpu_last_error());
            icm_gpu_release_pooled_memory();
            continue;
        }

        /* Warmup */
        IcmGpuRunStats warm{};
        int warm_ok = icm_gpu_equity_with_plan(plan, Q_POINTS, payout.data(),
                                                eq.data(), &warm);
        if (warm_ok != 0) {
            printf("%-10d %-8d %-30s WARMUP_FAIL: %s\n",
                   n, k, kPoints[p].tag, icm_gpu_last_error());
            icm_gpu_plan_destroy(plan);
            icm_gpu_release_pooled_memory();
            continue;
        }

        std::vector<double> samples;
        for (int r = 0; r < N_REPS; ++r) {
            IcmGpuRunStats stats{};
            int status = icm_gpu_equity_with_plan(plan, Q_POINTS,
                                                   payout.data(), eq.data(),
                                                   &stats);
            if (status != 0) {
                fprintf(stderr, "  n=%d k=%d rep %d FAIL: %s\n",
                        n, k, r, icm_gpu_last_error());
                break;
            }
            samples.push_back(stats.total_ns / 1e6);
        }

        icm_gpu_plan_destroy(plan);
        icm_gpu_release_pooled_memory();

        if (samples.size() < (size_t)N_REPS) {
            printf("%-10d %-8d %-30s INCOMPLETE (%zu/%d reps)\n",
                   n, k, kPoints[p].tag, samples.size(), N_REPS);
            continue;
        }

        double med = median_of(samples);
        double cv  = cv_of(samples);

        printf("%-10d %-8d %-30s %-10.3f %-8.4f [", n, k, kPoints[p].tag, med, cv);
        for (size_t i = 0; i < samples.size(); ++i)
            printf("%s%.3f", i ? "," : "", samples[i]);
        printf("]\n");
        fflush(stdout);
    }

    icm_gpu_shutdown();
    return 0;
}
