#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "icm.h"
#include "icm_gpu.h"

static inline double now_ns() {
    using namespace std::chrono;
    return duration<double, std::nano>(steady_clock::now().time_since_epoch()).count();
}

static inline double rel_err(double a, double b) {
    double d = fabs(a - b);
    double s = fabs(b);
    if (s < 1e-14) s = 1.0;
    return d / s;
}

static void make_stacks(int n, int dist, std::vector<double> &S) {
    S.resize(n);
    srand(42);
    switch (dist) {
    case 0:
        for (int i = 0; i < n; ++i) S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
        break;
    case 1:
        S[0] = 10000.0;
        for (int i = 1; i < n; ++i) S[i] = 1.0;
        break;
    case 2:
        for (int i = 0; i < n; ++i) S[i] = pow(2.0, (double)i * 10.0 / n);
        break;
    case 3:
        for (int i = 0; i < n; ++i) S[i] = 100.0;
        break;
    default:
        for (int i = 0; i < n; ++i) S[i] = 1.0 + ((double)rand() / RAND_MAX);
        break;
    }
}

static void make_payout(int n, int k, std::vector<double> &payout) {
    payout.resize(k);
    for (int m = 0; m < k; ++m) payout[m] = (double)(n - m);
}

static void apply_env_overrides(IcmGpuOptions &opts) {
    const char *v = nullptr;
    v = getenv("ICM_GPU_BENCH_ENABLE_GRAPHS");
    if (v && v[0]) opts.enable_graphs = atoi(v) ? 1 : 0;
    v = getenv("ICM_GPU_BENCH_ENABLE_Q_PIPELINE");
    if (v && v[0]) opts.enable_q_pipeline = atoi(v) ? 1 : 0;
    v = getenv("ICM_GPU_BENCH_USE_CUFFTDX");
    if (v && v[0]) opts.use_cufftdx = atoi(v) ? 1 : 0;
    v = getenv("ICM_GPU_BENCH_MEMORY_STRATEGY");
    if (v && v[0]) opts.memory_strategy = atoi(v);
}

static int run_verify(int extended) {
    const int Q = 256;
    const int n_cases_basic[] = {64, 256, 1024, 4096, 16384, 65536};
    const int n_cases_ext[] = {64, 128, 256, 1024, 4096, 16384, 65536, 131072};
    std::vector<int> n_cases_vec;
    if (extended) n_cases_vec.assign(n_cases_ext, n_cases_ext + (int)(sizeof(n_cases_ext) / sizeof(n_cases_ext[0])));
    else n_cases_vec.assign(n_cases_basic, n_cases_basic + (int)(sizeof(n_cases_basic) / sizeof(n_cases_basic[0])));
    if (extended) {
        const char *max_n_env = getenv("ICM_GPU_VERIFY_EXT_MAX_N");
        if (max_n_env && max_n_env[0]) {
            int max_n = atoi(max_n_env);
            if (max_n > 0) {
                std::vector<int> filtered;
                for (int n : n_cases_vec) if (n <= max_n) filtered.push_back(n);
                if (!filtered.empty()) n_cases_vec.swap(filtered);
            }
        }
    }
    const int *n_cases = n_cases_vec.data();
    const int n_n = (int)n_cases_vec.size();
    const int dists_basic[] = {0, 1, 2};
    const int dists_ext[] = {0, 1, 2, 3};
    const int *dists = extended ? dists_ext : dists_basic;
    const int n_d = extended ? 4 : 3;

    int all_pass = 1;
    double worst_err = 0.0;
    int worst_n = 0;
    int worst_k = 0;
    int worst_dist = 0;
    int worst_idx = -1;
    IcmGpuRunStats worst_stats{};
    for (int ni = 0; ni < n_n; ++ni) {
        int n = n_cases[ni];
        for (int di = 0; di < n_d; ++di) {
            std::vector<double> S;
            make_stacks(n, dists[di], S);

            int ks_basic[] = {std::min(100, n), n};
            int ks_ext[] = {
                std::min(16, n),
                std::min(100, n),
                std::min(512, n),
                std::max(1, n / 2),
                n
            };
            if (extended) {
                const char *lite_env = getenv("ICM_GPU_VERIFY_EXT_LITE");
                if (lite_env && lite_env[0] && atoi(lite_env) != 0) {
                    ks_ext[0] = std::min(100, n);
                    ks_ext[1] = std::max(1, n / 2);
                    ks_ext[2] = n;
                }
            }
            const int *ks = extended ? ks_ext : ks_basic;
            int n_k = extended ? ((getenv("ICM_GPU_VERIFY_EXT_LITE") && atoi(getenv("ICM_GPU_VERIFY_EXT_LITE")) != 0) ? 3 : 5) : 2;
            for (int ki = 0; ki < n_k; ++ki) {
                int k = ks[ki];
                std::vector<double> payout;
                make_payout(n, k, payout);

                std::vector<double> cpu_eq(n, 0.0), gpu_eq(n, 0.0);
                double t_cpu0 = now_ns();
                icm_equity(n, S.data(), Q, payout.data(), k, cpu_eq.data());
                double t_cpu_ns = now_ns() - t_cpu0;

                IcmGpuOptions opts{};
                opts.device_id = 0;
                opts.use_cufftdx = 1;
                opts.enable_graphs = 0;
                opts.enable_q_pipeline = 1;
                opts.memory_strategy = 0;
                opts.force_uncached_fused_levels = -1;
                opts.force_uncached_cufft_levels = -1;
                apply_env_overrides(opts);
                IcmGpuRunStats stats{};
                int status = icm_gpu_equity(n, S.data(), Q, payout.data(), k,
                                            gpu_eq.data(), &opts, &stats);
                if (status != 0) {
                    printf("FAIL n=%d k=%d dist=%d gpu-error=%s\n",
                           n, k, dists[di], icm_gpu_last_error());
                    all_pass = 0;
                    continue;
                }

                double max_rel = 0.0;
                int max_i = -1;
                for (int i = 0; i < n; ++i) {
                    double r = rel_err(gpu_eq[i], cpu_eq[i]);
                    if (r > max_rel) {
                        max_rel = r;
                        max_i = i;
                    }
                }
                int pass = (max_rel < 1e-8);
                if (!pass) all_pass = 0;
                if (max_rel > worst_err) {
                    worst_err = max_rel;
                    worst_n = n;
                    worst_k = k;
                    worst_dist = dists[di];
                    worst_idx = max_i;
                    worst_stats = stats;
                }
                printf("%s n=%-7d k=%-7d dist=%d  err=%.3e  cpu=%.1f ms gpu=%.1f ms  B=%d tiers(cache/recomp)=%.1f/%.1f ms\n",
                       pass ? "PASS" : "FAIL", n, k, dists[di], max_rel,
                       t_cpu_ns / 1e6, stats.total_ns / 1e6, stats.B,
                       stats.tree_propagate_cached_ns / 1e6,
                       stats.tree_propagate_recomputed_ns / 1e6);
                fflush(stdout);
                if (extended && !pass) {
                    printf("  diag: idx=%d cpu=%.17g gpu=%.17g abs=%.3e\n",
                           max_i, cpu_eq[max_i], gpu_eq[max_i], fabs(cpu_eq[max_i] - gpu_eq[max_i]));
                    fflush(stdout);
                }
            }
        }
    }
    if (extended) {
        printf("verify_ext_worst: err=%.3e n=%d k=%d dist=%d idx=%d B=%d peak=%.1fMB block=%.2f build=%.2f prop_cached=%.2f prop_recomp=%.2f leaf=%.2f\n",
               worst_err, worst_n, worst_k, worst_dist, worst_idx, worst_stats.B,
               (double)worst_stats.peak_vram_bytes / (1024.0 * 1024.0),
               worst_stats.block_build_ns / 1e6,
               worst_stats.tree_build_ns / 1e6,
               worst_stats.tree_propagate_cached_ns / 1e6,
               worst_stats.tree_propagate_recomputed_ns / 1e6,
               worst_stats.leaf_extract_ns / 1e6);
    }
    return all_pass ? 0 : 1;
}

static int run_single_bench(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: bench_gpu bench <n> <k> [reps] [Q]\n");
        return 1;
    }
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int reps = (argc > 4) ? atoi(argv[4]) : 3;
    int Q = (argc > 5) ? atoi(argv[5]) : 256;
    if (k > n) k = n;
    if (k < 1) k = 1;

    std::vector<double> S;
    make_stacks(n, 0, S);
    std::vector<double> payout;
    make_payout(n, k, payout);
    std::vector<double> equity(n, 0.0);

    IcmGpuOptions opts{};
    opts.device_id = 0;
    opts.use_cufftdx = 1;
    opts.enable_graphs = 0;
    opts.enable_q_pipeline = 1;
    opts.memory_strategy = 0;
    opts.force_uncached_fused_levels = -1;
    opts.force_uncached_cufft_levels = -1;
    apply_env_overrides(opts);

    double best_ms = 1e30;
    for (int r = 0; r < reps; ++r) {
        IcmGpuRunStats stats{};
        int status = icm_gpu_equity(n, S.data(), Q, payout.data(), k,
                                    equity.data(), &opts, &stats);
        if (status != 0) {
            printf("ERROR: %s\n", icm_gpu_last_error());
            return 1;
        }
        double ms = stats.total_ns / 1e6;
        if (ms < best_ms) best_ms = ms;
        printf("run=%d  total=%.2f ms  B=%d  engine=%d  peak_vram=%.1f MB  "
               "block=%.2f tree_build=%.2f prop_cached=%.2f prop_recomp=%.2f leaf=%.2f\n",
               r + 1, ms, stats.B, stats.engine,
               (double)stats.peak_vram_bytes / (1024.0 * 1024.0),
               stats.block_build_ns / 1e6, stats.tree_build_ns / 1e6,
               stats.tree_propagate_cached_ns / 1e6,
               stats.tree_propagate_recomputed_ns / 1e6,
               stats.leaf_extract_ns / 1e6);
    }
    printf("best=%.2f ms  (n=%d k=%d Q=%d)\n", best_ms, n, k, Q);
    return 0;
}

static int run_quick_grid() {
    const int Q = 64;
    int ns[] = {256, 1024, 4096, 8192, 16384};
    int n_n = (int)(sizeof(ns) / sizeof(ns[0]));
    printf("=== GPU QUICK GRID (Q=%d) ===\n", Q);
    for (int ni = 0; ni < n_n; ++ni) {
        int n = ns[ni];
        int ks[] = {std::min(64, n), std::min(256, n), n};
        std::vector<double> S;
        make_stacks(n, 0, S);
        for (int ki = 0; ki < 3; ++ki) {
            int k = ks[ki];
            std::vector<double> payout;
            make_payout(n, k, payout);
            std::vector<double> eq(n, 0.0);
            IcmGpuOptions opts{};
            opts.device_id = 0;
            opts.use_cufftdx = 1;
            opts.enable_graphs = 0;
            opts.enable_q_pipeline = 1;
            opts.memory_strategy = 0;
            opts.force_uncached_fused_levels = -1;
            opts.force_uncached_cufft_levels = -1;
            apply_env_overrides(opts);
            IcmGpuRunStats stats{};
            int status = icm_gpu_equity(n, S.data(), Q, payout.data(), k, eq.data(), &opts, &stats);
            if (status != 0) {
                printf("n=%d k=%d ERROR: %s\n", n, k, icm_gpu_last_error());
                return 1;
            }
            printf("n=%-7d k=%-7d time=%8.2f ms  B=%-4d peak=%8.1f MB\n",
                   n, k, stats.total_ns / 1e6, stats.B, (double)stats.peak_vram_bytes / (1024.0 * 1024.0));
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (!icm_gpu_init(0)) {
        printf("ERROR: %s\n", icm_gpu_last_error());
        return 1;
    }
    icm_init(nullptr);

    if (argc > 1 && strcmp(argv[1], "verify") == 0) {
        int rc = run_verify(0);
        icm_gpu_shutdown();
        return rc;
    }
    if (argc > 1 && strcmp(argv[1], "verify_ext") == 0) {
        int rc = run_verify(1);
        icm_gpu_shutdown();
        return rc;
    }
    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        int rc = run_single_bench(argc, argv);
        icm_gpu_shutdown();
        return rc;
    }
    if (argc > 1 && strcmp(argv[1], "quick") == 0) {
        int rc = run_quick_grid();
        icm_gpu_shutdown();
        return rc;
    }

    printf("Usage:\n");
    printf("  bench_gpu verify\n");
    printf("  bench_gpu verify_ext\n");
    printf("  bench_gpu quick\n");
    printf("  bench_gpu bench <n> <k> [reps] [Q]\n");
    icm_gpu_shutdown();
    return 0;
}
