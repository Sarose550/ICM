// gpu_ws_repro.cu — standalone regression check for GPU workspace-sizing OOM.
//
// Reproduces the original bug where the GPU planner over-allocated workspace
// for large-n plans, causing out-of-memory failures at n=2,097,152 with
// k=256 and k=512. Uses the plan-based API (icm_gpu_plan_create) — tests
// that plan creation succeeds, not that equity results are correct.
//
// Kept as a standalone tool (not integrated into bench/bench_gpu.cu) because
// bench_gpu.cu uses the equity API (icm_gpu_equity) exclusively, and
// introducing a plan-API regression check there would require untestable
// restructuring without a CUDA toolchain on hand. Run directly:
//   nvcc -I src/gpu -o gpu_ws_repro tools/gpu_ws_repro.cu && ./gpu_ws_repro
//
// Regression check for: original GPU workspace-sizing OOM bug
// (n=2097152, k=256/512).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "icm_gpu.h"

static void make_stacks_uniform(int n, std::vector<double> &S) {
    S.resize(n);
    srand(42 + n);
    for (int i = 0; i < n; ++i) S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}

static void make_payout(int n, int k, std::vector<double> &payout) {
    payout.resize(k);
    for (int m = 0; m < k; ++m) payout[m] = (double)(n - m);
}

int main() {
    if (!icm_gpu_init(0)) {
        printf("FAIL icm_gpu_init: %s\n", icm_gpu_last_error());
        return 1;
    }

    struct TestCase {
        int n, k;
    } cases[] = {
        {2097152, 256},
        {2097152, 512},
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));
    int any_fail = 0;

    for (int i = 0; i < n_cases; ++i) {
        int n = cases[i].n;
        int k = cases[i].k;

        std::vector<double> S, payout;
        make_stacks_uniform(n, S);
        make_payout(n, k, payout);

        IcmGpuOptions opts{};
        opts.device_id = 0;
        opts.use_cufftdx = 1;
        opts.enable_graphs = 0;
        opts.enable_q_pipeline = 1;
        opts.memory_strategy = 0;
        opts.force_uncached_fused_levels = -1;
        opts.force_uncached_cufft_levels = -1;

        IcmGpuPlan *plan = icm_gpu_plan_create(n, S.data(), k, &opts);
        if (!plan) {
            printf("FAIL n=%d k=%d: %s\n", n, k, icm_gpu_last_error());
            cudaDeviceReset();
            icm_gpu_init(0);
            any_fail = 1;
        } else {
            printf("PASS n=%d k=%d\n", n, k);
            icm_gpu_plan_destroy(plan);
        }
    }

    icm_gpu_shutdown();
    return any_fail ? 1 : 0;
}
