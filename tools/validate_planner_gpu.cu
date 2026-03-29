#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "icm_gpu.h"

static void make_stacks_uniform(int n, std::vector<double> &S) {
    S.resize(n);
    srand(123 + n);
    for (int i = 0; i < n; ++i) S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}

static void make_payout(int n, int k, std::vector<double> &payout) {
    payout.resize(k);
    for (int m = 0; m < k; ++m) payout[m] = (double)(n - m);
}

static double run_case(int n, int k, int Q, int force_B, int *out_B) {
    std::vector<double> S, payout, eq;
    make_stacks_uniform(n, S);
    make_payout(n, k, payout);
    eq.assign(n, 0.0);

    if (force_B > 0) {
        char bbuf[32];
        snprintf(bbuf, sizeof(bbuf), "%d", force_B);
        setenv("ICM_GPU_FORCE_B", bbuf, 1);
    } else {
        unsetenv("ICM_GPU_FORCE_B");
    }

    IcmGpuOptions opts{};
    opts.device_id = 0;
    opts.use_cufftdx = 1;
    opts.enable_graphs = 0;
    opts.enable_q_pipeline = 0;
    opts.memory_strategy = 0;
    opts.force_uncached_fused_levels = -1;
    opts.force_uncached_cufft_levels = -1;

    IcmGpuRunStats stats{};
    double t_ns = icm_gpu_equity(n, S.data(), Q, payout.data(), k, eq.data(), &opts, &stats);
    if (t_ns < 0) return -1.0;
    if (out_B) *out_B = stats.B;
    return t_ns / 1e6;
}

int main(int argc, char **argv) {
    const char *out_csv = "planner_validation.csv";
    int Q = 256;
    if (argc > 1) out_csv = argv[1];
    if (argc > 2) Q = atoi(argv[2]);

    if (!icm_gpu_init(0)) {
        printf("icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    std::vector<int> n_grid = {65536, 131072, 262144, 524288};
    std::vector<int> b_grid = {
        16, 24, 32, 48, 64, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240,
        256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 896
    };

    FILE *f = fopen(out_csv, "w");
    if (!f) {
        printf("Cannot open %s\n", out_csv);
        return 1;
    }
    fprintf(f, "n,k,auto_B,auto_ms,best_B,best_ms,match\n");

    for (int n : n_grid) {
        std::vector<int> k_grid = {n / 4, n / 2, n};
        for (int k : k_grid) {
            if (k < 1) continue;

            int auto_B = 0;
            double auto_ms = run_case(n, k, Q, 0, &auto_B);
            if (auto_ms < 0.0) {
                printf("n=%d k=%d auto run failed: %s\n", n, k, icm_gpu_last_error());
                fclose(f);
                return 1;
            }

            double best_ms = 1e30;
            int best_B = 0;
            for (int B : b_grid) {
                if (B > n || B > k) continue;
                double t_ms = run_case(n, k, Q, B, nullptr);
                if (t_ms < 0.0) {
                    printf("n=%d k=%d B=%d run failed: %s\n", n, k, B, icm_gpu_last_error());
                    fclose(f);
                    return 1;
                }
                if (t_ms < best_ms) {
                    best_ms = t_ms;
                    best_B = B;
                }
            }

            int match = (auto_B == best_B) ? 1 : 0;
            printf("n=%d k=%d auto(B=%d %.2fms) best(B=%d %.2fms) match=%d\n",
                   n, k, auto_B, auto_ms, best_B, best_ms, match);
            fprintf(f, "%d,%d,%d,%.6f,%d,%.6f,%d\n",
                    n, k, auto_B, auto_ms, best_B, best_ms, match);
            fflush(f);
        }
    }

    unsetenv("ICM_GPU_FORCE_B");
    fclose(f);
    icm_gpu_shutdown();
    printf("Wrote %s\n", out_csv);
    return 0;
}
