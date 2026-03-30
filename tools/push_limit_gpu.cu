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
    srand(123 + n);
    for (int i = 0; i < n; ++i) S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}

static void make_payout(int n, std::vector<double> &payout) {
    payout.resize(n);
    for (int m = 0; m < n; ++m) payout[m] = (double)(n - m);
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

struct Row {
    int n = 0;
    int B = 0;
    int M = 0;
    int T = 0;
    int reps = 0;
    double cv = 0.0;
    double time_ms = 0.0;
    double peak_vram_mb = 0.0;
    double block_ms = 0.0;
    double tree_build_ms = 0.0;
    double prop_cached_ms = 0.0;
    double prop_recomp_ms = 0.0;
    double leaf_ms = 0.0;
    double q_ovh_ms = 0.0;
};

int main(int argc, char **argv) {
    const char *out_csv = "gpu_limit_frontier.csv";
    int Q = 256;
    int fast = 0;
    if (argc > 1) out_csv = argv[1];
    if (argc > 2) Q = atoi(argv[2]);
    if (argc > 3 && strcmp(argv[3], "--fast") == 0) fast = 1;

    if (!icm_gpu_init(0)) {
        printf("icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    std::vector<int> n_grid = {
        8192, 16384, 32768, 65536, 131072, 262144, 524288, 786432, 917504,
        1048576, 1310720, 1572864, 1835008, 2097152, 2621440, 3145728,
        4194304, 5242880, 6291456, 8388608
    };
    if (fast && n_grid.size() > 9) n_grid.resize(9);

    std::vector<int> b_grid = {
        16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 288, 320,
        384, 448, 512, 640, 768, 896, 1024
    };
    if (fast && b_grid.size() > 10) b_grid.resize(10);

    FILE *f = fopen(out_csv, "w");
    if (!f) {
        printf("Cannot open %s\n", out_csv);
        return 1;
    }
    fprintf(f, "n,k,B,M,T,time_ms,peak_vram_mb,block_ms,tree_build_ms,prop_cached_ms,prop_recomp_ms,leaf_ms,q_ovh_ms,reps,cv\n");

    std::vector<Row> feasible_rows;
    int consecutive_misses = 0;

    for (int n : n_grid) {
        int k = n;
        std::vector<double> S, payout, eq;
        make_stacks_uniform(n, S);
        make_payout(n, payout);
        eq.assign(n, 0.0);

        printf("=== n=%d k=n ===\n", n);
        Row best{};
        double best_time = 1e30;
        bool found = false;

        int maxM = fast ? 2 : 10;
        int maxT = fast ? 1 : 4;
        for (int B : b_grid) {
            if (B > n) continue;
            char bbuf[32];
            snprintf(bbuf, sizeof(bbuf), "%d", B);
            setenv("ICM_GPU_FORCE_B", bbuf, 1);

            for (int M = 0; M <= maxM; ++M) {
                for (int T = 0; T <= maxT; ++T) {
                    IcmGpuOptions opts{};
                    opts.device_id = 0;
                    opts.use_cufftdx = 1;
                    opts.enable_graphs = 0;
                    opts.enable_q_pipeline = 1;
                    opts.memory_strategy = 3;
                    opts.force_uncached_fused_levels = M;
                    opts.force_uncached_cufft_levels = T;

                    IcmGpuRunStats warm{};
                    double warm_ns = icm_gpu_equity(n, S.data(), Q, payout.data(), k, eq.data(), &opts, &warm);
                    if (warm_ns < 0) {
                        printf("  B=%d M=%d T=%d -> ERR(%s)\n", B, M, T, icm_gpu_last_error());
                        continue;
                    }

                    double warm_ms = warm_ns / 1e6;
                    int reps = 3;
                    if (warm_ms < 10.0) reps = 10;
                    else if (warm_ms > 100.0) reps = 1;
                    if (fast) reps = std::min(reps, 3);
                    int max_reps = fast ? 5 : 12;

                    std::vector<double> samples;
                    samples.reserve(max_reps);
                    IcmGpuRunStats stats{};
                    for (int r = 0; r < reps; ++r) {
                        double t_ns = icm_gpu_equity(n, S.data(), Q, payout.data(), k, eq.data(), &opts, &stats);
                        if (t_ns < 0) {
                            samples.clear();
                            break;
                        }
                        samples.push_back(t_ns / 1e6);
                    }
                    while (!samples.empty() && (int)samples.size() < max_reps) {
                        double cv = cv_ms(samples);
                        if (cv <= 0.03) break;
                        double t_ns = icm_gpu_equity(n, S.data(), Q, payout.data(), k, eq.data(), &opts, &stats);
                        if (t_ns < 0) break;
                        samples.push_back(t_ns / 1e6);
                    }
                    if (samples.empty()) continue;

                    Row row{};
                    row.n = n;
                    row.B = stats.B;
                    row.M = M;
                    row.T = T;
                    row.reps = (int)samples.size();
                    row.cv = cv_ms(samples);
                    row.time_ms = median_ms(samples);
                    row.peak_vram_mb = (double)stats.peak_vram_bytes / (1024.0 * 1024.0);
                    row.block_ms = stats.block_build_ns / 1e6;
                    row.tree_build_ms = stats.tree_build_ns / 1e6;
                    row.prop_cached_ms = stats.tree_propagate_cached_ns / 1e6;
                    row.prop_recomp_ms = stats.tree_propagate_recomputed_ns / 1e6;
                    row.leaf_ms = stats.leaf_extract_ns / 1e6;
                    row.q_ovh_ms = stats.quadrature_overhead_ns / 1e6;

                    fprintf(f, "%d,%d,%d,%d,%d,%.6f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f\n",
                            n, k, row.B, row.M, row.T, row.time_ms, row.peak_vram_mb,
                            row.block_ms, row.tree_build_ms, row.prop_cached_ms,
                            row.prop_recomp_ms, row.leaf_ms, row.q_ovh_ms,
                            row.reps, row.cv);
                    printf("  B=%d M=%d T=%d -> %.2f ms peak=%.1f MB reps=%d cv=%.3f\n",
                           row.B, row.M, row.T, row.time_ms, row.peak_vram_mb, row.reps, row.cv);

                    if (row.time_ms <= 1000.0 && row.time_ms < best_time) {
                        best_time = row.time_ms;
                        best = row;
                        found = true;
                    }
                }
                fflush(f);
            }
        }
        unsetenv("ICM_GPU_FORCE_B");

        if (found) {
            consecutive_misses = 0;
            feasible_rows.push_back(best);
            printf("  best <=1s: B=%d M=%d T=%d time=%.2f ms peak=%.1f MB reps=%d cv=%.3f\n",
                   best.B, best.M, best.T, best.time_ms, best.peak_vram_mb, best.reps, best.cv);
        } else {
            ++consecutive_misses;
            printf("  no <=1s configuration found for n=%d\n", n);
            if (!fast && consecutive_misses >= 2) {
                printf("  stopping search after %d consecutive misses\n", consecutive_misses);
                break;
            }
        }
    }

    fclose(f);

    if (!feasible_rows.empty()) {
        auto headline = *std::max_element(feasible_rows.begin(), feasible_rows.end(),
                                          [](const Row &a, const Row &b) {
                                              if (a.n != b.n) return a.n < b.n;
                                              return a.time_ms > b.time_ms;
                                          });
        printf("\nHeadline candidate: n=%d (k=n), B=%d M=%d T=%d time=%.2f ms peak=%.1f MB\n",
               headline.n, headline.B, headline.M, headline.T,
               headline.time_ms, headline.peak_vram_mb);
    } else {
        printf("\nNo <=1s configurations found in scanned range.\n");
    }

    icm_gpu_shutdown();
    printf("Wrote %s\n", out_csv);
    return 0;
}
