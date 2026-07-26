/* Ad-hoc, NOT part of the repo -- node E_HW_VERIFY_BSELECT_GAP measurement
 * only. Probes the B-selection calibration gap (n=1,048,576 B=32 to
 * n=1,572,864 B=96-192) on both the k=n and k=100 curves, reps=5 each,
 * to check for a real wall-clock inversion (higher n running FASTER than
 * a lower n due to the B nearest-neighbor cliff). */
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

static const Point kPoints[] = {
    /* k = n curve, spanning the calibration gap */
    {1048576, 1048576, "k=n anchor (B=32, calibrated)"},
    {1100000, 1100000, "k=n GAP (B nearest-neighbor -> 32?)"},
    {1285000, 1285000, "k=n GAP midpoint (B nearest-neighbor -> ?)"},
    {1450000, 1450000, "k=n GAP (B nearest-neighbor -> 96+?)"},
    {1572864, 1572864, "k=n anchor (B=96-192, calibrated)"},
    /* k = 100 curve, same n span */
    {1048576, 100, "k=100 anchor (B=32, calibrated)"},
    {1100000, 100, "k=100 GAP"},
    {1285000, 100, "k=100 GAP midpoint"},
    {1450000, 100, "k=100 GAP"},
    {1572864, 100, "k=100 anchor (B=96-192, calibrated)"},
};
static const int kNumPoints = sizeof(kPoints) / sizeof(kPoints[0]);

int main() {
    if (!icm_gpu_init(0)) {
        fprintf(stderr, "icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    printf("bselect_gap_probe -- reps=%d per cell, Q=%d\n\n", N_REPS, Q_POINTS);
    printf("%-10s %-10s %-32s %-10s %-8s %s\n",
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
            printf("%-10d %-10d %-32s PLAN_CREATE_FAIL: %s\n",
                   n, k, kPoints[p].tag, icm_gpu_last_error());
            icm_gpu_release_pooled_memory();
            continue;
        }

        IcmGpuRunStats warm{};
        int warm_ok = icm_gpu_equity_with_plan(plan, Q_POINTS, payout.data(),
                                                eq.data(), &warm);
        if (warm_ok != 0) {
            printf("%-10d %-10d %-32s WARMUP_FAIL: %s\n",
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
            printf("%-10d %-10d %-32s INCOMPLETE (%zu/%d reps)\n",
                   n, k, kPoints[p].tag, samples.size(), N_REPS);
            continue;
        }

        double med = median_of(samples);
        double cv  = cv_of(samples);

        printf("%-10d %-10d %-32s %-10.3f %-8.4f [", n, k, kPoints[p].tag, med, cv);
        for (size_t i = 0; i < samples.size(); ++i)
            printf("%s%.3f", i ? "," : "", samples[i]);
        printf("]\n");
        fflush(stdout);
    }

    icm_gpu_shutdown();
    return 0;
}
