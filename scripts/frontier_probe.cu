#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "icm_gpu.h"

static void make_stacks(int n, std::vector<double> &S) {
    S.resize(n);
    srand(123 + n);
    for (int i = 0; i < n; ++i) S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}
static void make_payout(int n, int k, std::vector<double> &payout) {
    payout.resize(k);
    for (int m = 0; m < k; ++m) payout[m] = (double)(n - m);
}
static double median_ms(std::vector<double> &x) {
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

static void run_case(int n, int k, int reps) {
    std::vector<double> S, payout, eq;
    make_stacks(n, S);
    make_payout(n, k, payout);
    eq.assign(n, 0.0);
    IcmGpuOptions opts{};
    opts.device_id = 0; opts.use_cufftdx = 1; opts.enable_graphs = 0;
    opts.enable_q_pipeline = 1; opts.memory_strategy = 0;
    opts.force_uncached_fused_levels = -1; opts.force_uncached_cufft_levels = -1;

    IcmGpuPlan *plan = icm_gpu_plan_create(n, S.data(), k, &opts);
    if (!plan) {
        printf("n=%d k=%d FAIL: %s\n", n, k, icm_gpu_last_error());
        cudaDeviceReset();
        icm_gpu_init(0);
        return;
    }
    IcmGpuRunStats stats{};
    /* warmup */
    icm_gpu_equity_with_plan(plan, 256, payout.data(), eq.data(), &stats);
    std::vector<double> samples;
    for (int r = 0; r < reps; ++r) {
        int status = icm_gpu_equity_with_plan(plan, 256, payout.data(), eq.data(), &stats);
        if (status != 0) { samples.clear(); break; }
        samples.push_back(stats.total_ns / 1e6);
    }
    if (samples.empty()) {
        printf("n=%d k=%d FAIL(exec): %s\n", n, k, icm_gpu_last_error());
    } else {
        printf("n=%d k=%d time_ms=%.3f peak_vram_mb=%.1f B=%d reps=%d\n",
               n, k, median_ms(samples), stats.peak_vram_bytes / 1e6, stats.B, reps);
    }
    icm_gpu_plan_destroy(plan);
}

int main() {
    if (!icm_gpu_init(0)) { printf("init failed: %s\n", icm_gpu_last_error()); return 1; }

    printf("=== RESULTS.md frontier table points (k=n and k=100) ===\n");
    int reps = 5;
    run_case(1441792, 1441792, reps);
    run_case(1572864, 1572864, reps);
    run_case(6291456, 100, reps);
    run_case(8388608, 100, reps);
    run_case(16777216, 10, reps);

    printf("\n=== confirming the new OOM-band finding at representative points ===\n");
    run_case(2097152, 256, reps);
    run_case(4194304, 512, reps);
    run_case(8388608, 128, reps);
    run_case(16777216, 128, reps);

    return 0;
}
