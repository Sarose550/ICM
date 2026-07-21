#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "icm_gpu.h"

/* ══════════════════════════════════════════════════════════════
   SHARED HELPERS
   ══════════════════════════════════════════════════════════════ */

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

static IcmGpuOptions default_opts() {
    IcmGpuOptions opts{};
    opts.device_id = 0;
    opts.use_cufftdx = 1;
    opts.enable_graphs = 0;
    opts.enable_q_pipeline = 1;
    opts.memory_strategy = 0;
    opts.force_uncached_fused_levels = -1;
    opts.force_uncached_cufft_levels = -1;
    return opts;
}

/* Time a single (n, k) point on GPU.
 * Returns median time in ms, or NAN on failure (OOM, execution error).
 * Fills stats with the last run's stats. Resets GPU on OOM. */
static double gpu_time_ms(int n, int k, int Q, int fast,
                          IcmGpuRunStats *stats, double timeout_ms) {
    std::vector<double> S, payout, equity;
    make_stacks_uniform(n, S);
    make_payout(n, k, payout);
    equity.assign(n, 0.0);

    IcmGpuOptions opts = default_opts();
    IcmGpuPlan *plan = icm_gpu_plan_create(n, S.data(), k, &opts);
    if (!plan) {
        cudaDeviceReset();
        icm_gpu_init(0);
        return NAN;
    }

    IcmGpuRunStats warm{};
    int warm_status = icm_gpu_equity_with_plan(plan, Q, payout.data(), equity.data(), &warm);
    if (warm_status != 0) {
        icm_gpu_plan_destroy(plan);
        cudaDeviceReset();
        icm_gpu_init(0);
        return NAN;
    }
    double warm_ms = warm.total_ns / 1e6;

    /* Bail early if warmup already exceeds timeout */
    if (timeout_ms > 0 && warm_ms > timeout_ms) {
        icm_gpu_plan_destroy(plan);
        return warm_ms;
    }

    int reps = 3;
    if (warm_ms < 10.0) reps = 10;
    else if (warm_ms > 100.0) reps = 1;
    if (fast) reps = std::min(reps, 3);
    int max_reps = fast ? 5 : 15;

    std::vector<double> samples;
    *stats = {};
    for (int r = 0; r < reps; ++r) {
        int status = icm_gpu_equity_with_plan(plan, Q, payout.data(), equity.data(), stats);
        if (status != 0) { samples.clear(); break; }
        samples.push_back(stats->total_ns / 1e6);
    }
    while (!samples.empty() && (int)samples.size() < max_reps) {
        double cv = cv_ms(samples);
        if (cv <= 0.03) break;
        int status = icm_gpu_equity_with_plan(plan, Q, payout.data(), equity.data(), stats);
        if (status != 0) break;
        samples.push_back(stats->total_ns / 1e6);
    }

    icm_gpu_plan_destroy(plan);
    if (samples.empty()) return NAN;
    return median_ms(samples);
}

/* ══════════════════════════════════════════════════════════════
   MODE 1: 2D HEATMAP SWEEP
   ══════════════════════════════════════════════════════════════ */

static void run_heatmap(const char *out_csv, int Q, int fast) {
    std::vector<int> grid = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
        16777216, 33554432
    };
    if (fast && grid.size() > 14) grid.resize(14);

    FILE *f = fopen(out_csv, "w");
    if (!f) { printf("Cannot open %s\n", out_csv); return; }
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

            /* Need plan summary for tier breakdown */
            std::vector<double> S, payout, equity;
            make_stacks_uniform(n, S);
            make_payout(n, k, payout);
            equity.assign(n, 0.0);

            IcmGpuOptions opts = default_opts();
            IcmGpuPlan *plan = icm_gpu_plan_create(n, S.data(), k, &opts);
            if (!plan) {
                fprintf(f, "%d,%d,nan,nan,error,0,0,nan,0,0,0,error\n", n, k);
                printf("ERR(%s)\n", icm_gpu_last_error());
                cudaDeviceReset();
                icm_gpu_init(0);
                continue;
            }
            IcmGpuPlanSummary ps{};
            icm_gpu_plan_summary(plan, &ps);

            IcmGpuRunStats warm{};
            int warm_status = icm_gpu_equity_with_plan(plan, Q, payout.data(), equity.data(), &warm);
            if (warm_status != 0) {
                icm_gpu_plan_destroy(plan);
                fprintf(f, "%d,%d,nan,nan,error,0,0,nan,0,0,0,error\n", n, k);
                printf("ERR(%s)\n", icm_gpu_last_error());
                cudaDeviceReset();
                icm_gpu_init(0);
                continue;
            }

            int reps = 3;
            double warm_ms = warm.total_ns / 1e6;
            if (warm_ms < 10.0) reps = 10;
            else if (warm_ms > 100.0) reps = 1;
            if (fast) reps = std::min(reps, 3);
            int max_reps = fast ? 5 : 15;

            std::vector<double> samples;
            IcmGpuRunStats stats{};
            for (int r = 0; r < reps; ++r) {
                int status = icm_gpu_equity_with_plan(plan, Q, payout.data(), equity.data(), &stats);
                if (status != 0) { samples.clear(); break; }
                samples.push_back(stats.total_ns / 1e6);
            }
            while (!samples.empty() && (int)samples.size() < max_reps) {
                double cv = cv_ms(samples);
                if (cv <= 0.03) break;
                int status = icm_gpu_equity_with_plan(plan, Q, payout.data(), equity.data(), &stats);
                if (status != 0) break;
                samples.push_back(stats.total_ns / 1e6);
            }

            icm_gpu_plan_destroy(plan);
            if (samples.empty()) {
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
        }
        fflush(f);
    }

    fclose(f);
    printf("Wrote %s\n", out_csv);
}

/* ══════════════════════════════════════════════════════════════
   MODE 2: n=k THRESHOLD (binary search along the diagonal)
   ══════════════════════════════════════════════════════════════ */

static void run_nk_threshold(int Q, int fast) {
    double target_ms = 1000.0;
    double timeout_ms = 2000.0;  /* kill probes beyond 2s — clearly above frontier */

    /* Exponential expansion to bracket the frontier */
    int n_lo = 1024;
    int n_hi = 1024;
    int max_n = 33554432;

    IcmGpuRunStats stats{};
    double t_hi = gpu_time_ms(n_hi, n_hi, Q, fast, &stats, timeout_ms);
    printf("n=%d -> %.2f ms\n", n_hi, t_hi);

    while (!std::isnan(t_hi) && t_hi < target_ms && n_hi < max_n) {
        n_lo = n_hi;
        n_hi = std::min(n_hi * 2, max_n);
        t_hi = gpu_time_ms(n_hi, n_hi, Q, fast, &stats, timeout_ms);
        printf("n=%d -> %.2f ms\n", n_hi, std::isnan(t_hi) ? -1.0 : t_hi);
    }

    /* OOM or timeout → upper bound found */
    if (std::isnan(t_hi) || t_hi >= target_ms) {
        /* Bisect to 5% precision */
        while ((double)(n_hi - n_lo) > 0.05 * (double)n_lo && n_hi - n_lo > 100) {
            int n_mid = n_lo + (n_hi - n_lo) / 2;
            double t_mid = gpu_time_ms(n_mid, n_mid, Q, fast, &stats, timeout_ms);
            printf("n=%d -> %.2f ms\n", n_mid, std::isnan(t_mid) ? -1.0 : t_mid);
            if (!std::isnan(t_mid) && t_mid <= target_ms) {
                n_lo = n_mid;
            } else {
                n_hi = n_mid;
            }
        }
    } else {
        /* Even max_n is under 1s */
        n_lo = n_hi;
    }

    /* Final measurement at n_lo with full reps */
    double final_ms = gpu_time_ms(n_lo, n_lo, Q, 0, &stats, 0);
    printf("\nFRONTIER: n=k=%d  time=%.2f ms  B=%d  peak=%.1f MB\n",
           n_lo, final_ms, stats.B, (double)stats.peak_vram_bytes / (1024.0 * 1024.0));
    printf("n_eq_k,%d,%.2f,%s,%d\n",
           n_lo, final_ms, engine_name(stats.engine).c_str(), stats.B);
}

/* ══════════════════════════════════════════════════════════════
   MAIN
   ══════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    const char *out_csv = "gpu_heatmap.csv";
    int Q = 256;
    int fast = 0;
    enum { MODE_HEATMAP, MODE_NK } mode = MODE_HEATMAP;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--nk") == 0) mode = MODE_NK;
        else if (strcmp(argv[i], "--fast") == 0) fast = 1;
        else if (strcmp(argv[i], "--Q") == 0 && i + 1 < argc) Q = atoi(argv[++i]);
        else if (argv[i][0] != '-') out_csv = argv[i];
    }

    if (!icm_gpu_init(0)) {
        printf("icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    switch (mode) {
    case MODE_HEATMAP: run_heatmap(out_csv, Q, fast); break;
    case MODE_NK:      run_nk_threshold(Q, fast); break;
    }

    icm_gpu_shutdown();
    return 0;
}
