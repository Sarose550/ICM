#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

static std::string engine_name(int engine) {
    return engine == 1 ? "hybrid" : "linear";
}

static std::string dominant_tier(const IcmGpuPlanSummary &s) {
    if (s.n_tier2 >= s.n_tier1 && s.n_tier2 >= s.n_tier3) return "fused";
    if (s.n_tier3 >= s.n_tier1 && s.n_tier3 >= s.n_tier2) return "cufft";
    return "schoolbook";
}

static double median_ms(std::vector<double> &x) {
    if (x.empty()) return NAN;
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

static double cv_ms(const std::vector<double> &x) {
    if (x.size() < 2) return 0.0;
    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= (double)x.size();
    if (mean <= 0.0) return 0.0;
    double var = 0.0;
    for (double v : x) {
        double d = v - mean;
        var += d * d;
    }
    var /= (double)(x.size() - 1);
    return sqrt(var) / mean;
}

int main(int argc, char **argv) {
    const char *out_csv = "gpu_heatmap.csv";
    int fast = 0;
    int Q = 256;
    if (argc > 1) out_csv = argv[1];
    if (argc > 2) Q = atoi(argv[2]);
    if (argc > 3 && strcmp(argv[3], "--fast") == 0) fast = 1;

    if (!icm_gpu_init(0)) {
        printf("icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    std::vector<int> grid = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
        16777216, 33554432
    };
    /* Capped at 33M to avoid OOM crashes on B200 (192GB VRAM).
     * Sizes above 33M at k=n exceed VRAM and crash the container. */
    if (fast && grid.size() > 14) grid.resize(14);

    FILE *f = fopen(out_csv, "w");
    if (!f) {
        printf("Cannot open %s\n", out_csv);
        return 1;
    }
    fprintf(f, "n,k,time_ms,peak_vram_mb,engine,B,reps,cv,tier1_levels,tier2_levels,tier3_levels,dominant_tier\n");

    for (size_t ni = 0; ni < grid.size(); ++ni) {
        int n = grid[ni];
        std::vector<int> ks = grid;
        if (std::find(ks.begin(), ks.end(), n) == ks.end()) ks.push_back(n);
        for (int k : ks) {
            if (k > n) continue;
            if (fast && (k != n) && (k > std::min(n, 4096))) continue;
            printf("n=%d k=%d ... ", n, k);
            fflush(stdout);

            std::vector<double> S;
            std::vector<double> payout;
            std::vector<double> equity;
            make_stacks_uniform(n, S);
            make_payout(n, k, payout);
            equity.assign(n, 0.0);

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
                fprintf(f, "%d,%d,nan,nan,error,0,0,nan,0,0,0,error\n", n, k);
                printf("ERR(%s)\n", icm_gpu_last_error());
                /* Reset device after OOM to clear error state */
                cudaDeviceReset();
                icm_gpu_init(0);
                continue;
            }
            IcmGpuPlanSummary ps{};
            icm_gpu_plan_summary(plan, &ps);

            IcmGpuRunStats warm{};
            double warm_ns = icm_gpu_equity_with_plan(plan, Q, payout.data(), equity.data(), &warm);
            if (warm_ns < 0) {
                icm_gpu_plan_destroy(plan);
                fprintf(f, "%d,%d,nan,nan,error,0,0,nan,0,0,0,error\n", n, k);
                printf("ERR(%s)\n", icm_gpu_last_error());
                cudaDeviceReset();
                icm_gpu_init(0);
                continue;
            }

            int reps = 3;
            double warm_ms = warm_ns / 1e6;
            if (warm_ms < 10.0) reps = 10;
            else if (warm_ms > 100.0) reps = 1;
            if (fast) reps = std::min(reps, 3);
            int max_reps = fast ? 5 : 15;

            std::vector<double> samples;
            IcmGpuRunStats stats{};
            for (int r = 0; r < reps; ++r) {
                double t = icm_gpu_equity_with_plan(plan, Q, payout.data(), equity.data(), &stats);
                if (t < 0) {
                    samples.clear();
                    break;
                }
                samples.push_back(t / 1e6);
            }
            while (!samples.empty() && (int)samples.size() < max_reps) {
                double cv = cv_ms(samples);
                if (cv <= 0.03) break;
                double t = icm_gpu_equity_with_plan(plan, Q, payout.data(), equity.data(), &stats);
                if (t < 0) break;
                samples.push_back(t / 1e6);
            }
            if (samples.empty()) {
                icm_gpu_plan_destroy(plan);
                fprintf(f, "%d,%d,nan,nan,error,0,0,nan,0,0,0,error\n", n, k);
                printf("ERR(run)\n");
                continue;
            }
            double time_ms = median_ms(samples);
            double cv = cv_ms(samples);
            double vram_mb = (double)stats.peak_vram_bytes / (1024.0 * 1024.0);
            std::string dom = dominant_tier(ps);
            fprintf(f, "%d,%d,%.6f,%.3f,%s,%d,%d,%.6f,%d,%d,%d,%s\n",
                    n, k, time_ms, vram_mb, engine_name(stats.engine).c_str(), stats.B,
                    (int)samples.size(), cv,
                    ps.n_tier1, ps.n_tier2, ps.n_tier3, dom.c_str());
            printf("%.2f ms  B=%d  peak=%.1f MB  reps=%d  cv=%.3f\n",
                   time_ms, stats.B, vram_mb, (int)samples.size(), cv);
            icm_gpu_plan_destroy(plan);
        }
        fflush(f);
    }

    fclose(f);
    icm_gpu_shutdown();
    printf("Wrote %s\n", out_csv);
    return 0;
}
