/* heatmap_gpu_reset_every_cell.cu — diagnostic variant of tools/heatmap_gpu.cu
 *
 * PURPOSE: Test the "CUDA allocator fragmentation" hypothesis for the
 * long-sweep-only OOM (HANDOFF.md, GPU OOM section, NOT FIXED).
 *
 * HYPOTHESIS: The full 211-point heatmap_gpu sweep fails 21 cells at large n
 * because the CUDA default allocator (cudaMalloc) fragments after ~190 prior
 * sequential plan create/destroy cycles with varying arena sizes — not because
 * of a genuine leak in icm_gpu_plan_create/destroy.  Every one of the 21
 * failing points passes in isolation in a fresh process.
 *
 * MECHANISM: Each plan allocates one large cudaMalloc for its arena (tens of
 * GB at large n) plus one smaller cudaMalloc for shared cuFFT workspace.
 * These sizes vary significantly across the heatmap's (n,k) grid.  After many
 * alloc/free cycles with different sizes, the allocator's free list fragments
 * and a large allocation request fails even though total free VRAM exceeds the
 * request size — no single contiguous block is large enough.
 *
 * WHAT THIS TOOL DOES: Identical to tools/heatmap_gpu.cu's heatmap sweep,
 * except it calls cudaDeviceReset() + icm_gpu_init(0) between EVERY cell
 * (not just failures), which destroys and recreates the CUDA primary context,
 * fully resetting the allocator state.  If the 21 cells now pass, fragmentation
 * is confirmed as the root cause.
 *
 * WHAT RESULT CONFIRMS THE HYPOTHESIS: All 211 cells pass cleanly (or at
 * minimum, the 21 previously-failing cells now pass).
 *
 * WHAT RESULT REFUTES THE HYPOTHESIS: The same 21 cells still fail.  If this
 * happens, there is a genuine leak or state accumulation in the CUDA context
 * that survives cudaDeviceReset (which would be extraordinary — it would point
 * to a driver bug, a cuFFT-internal leak, or something outside the ICM
 * codebase entirely).
 *
 * HOW TO RUN (on the B200) -- corrected by supervisor review, the
 * originally-generated command used -arch=sm_90 (H100, not B200) and
 * link flags that don't match this repo's actual build recipe:
 *   CUFFTDX_INC=$(ls -d /usr/local/lib/python*/dist-packages/nvidia/mathdx/include | head -1)
 *   nvcc -O3 -std=c++17 -arch=sm_100 -Isrc -Idevices/b200 -Isrc/gpu \
 *        -I"$CUFFTDX_INC" -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX \
 *        -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY \
 *        -dc -o build/heatmap_reset_diag.o scripts/heatmap_gpu_reset_every_cell.cu
 *   nvcc -O3 -std=c++17 -arch=sm_100 -o heatmap_reset_diag \
 *        build/heatmap_reset_diag.o build/gpu_gpu_kernels_fused.o \
 *        build/gpu_gpu_plan_fused.o build/gpu_gpu_exec_fused.o \
 *        build/gpu_gpu_api_fused.o build/gpu_dlink_fused.o -lcufft -lcudart
 *   ./heatmap_reset_diag gpu_heatmap_reset_diag.csv
 * (Requires `make bench_gpu_fused` to have already populated build/ with
 * the four gpu_gpu_*_fused.o objects and gpu_dlink_fused.o -- same
 * pattern used for scripts/gpu_ws_repro.cu and scripts/frontier_probe.cu.)
 *
 * Compare output CSV row count (wc -l) against the original sweep's 190
 * success / 21 failure split.  If this tool produces 211 data rows (not
 * "nan"/"error" rows), the fragmentation hypothesis is confirmed.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "icm_gpu.h"

/* ── Helpers (identical to heatmap_gpu.cu) ─────────────────────── */

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

/* ── Main ──────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *out_csv = "gpu_heatmap_reset_diag.csv";
    int Q = 256;
    int fast = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--fast") == 0) fast = 1;
        else if (strcmp(argv[i], "--Q") == 0 && i + 1 < argc) Q = atoi(argv[++i]);
        else if (argv[i][0] != '-') out_csv = argv[i];
    }

    std::vector<int> grid = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
        16777216, 33554432
    };
    if (fast && grid.size() > 14) grid.resize(14);

    FILE *f = fopen(out_csv, "w");
    if (!f) { printf("Cannot open %s\n", out_csv); return 1; }
    fprintf(f, "n,k,time_ms,peak_vram_mb,engine,B,reps,cv,tier1_levels,tier2_levels,tier3_levels,dominant_tier\n");

    int total_cells = 0;
    int pass_cells = 0;
    int fail_cells = 0;

    /* Initialise once so the first cudaDeviceReset() has a context to reset. */
    if (!icm_gpu_init(0)) {
        printf("FATAL: initial icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    /* ── Unlike heatmap_gpu.cu, we do cudaDeviceReset()+icm_gpu_init(0)
     *    between EVERY cell, not just failures.  This isolates each cell
     *    to a fresh allocator state, testing the fragmentation hypothesis. */

    for (size_t ni = 0; ni < grid.size(); ++ni) {
        int n = grid[ni];
        std::vector<int> ks = grid;
        if (std::find(ks.begin(), ks.end(), n) == ks.end()) ks.push_back(n);

        for (int k : ks) {
            if (k > n) continue;
            if (fast && (k != n) && (k > std::min(n, 4096))) continue;

            /* ── Fresh CUDA context per cell ── */
            cudaDeviceReset();
            if (!icm_gpu_init(0)) {
                printf("n=%d k=%d FATAL: icm_gpu_init failed after reset\n", n, k);
                fclose(f);
                return 1;
            }

            total_cells++;
            printf("[%d/%d] n=%d k=%d ... ", total_cells, (int)(grid.size() * grid.size()), n, k);
            fflush(stdout);

            std::vector<double> S, payout, equity;
            make_stacks_uniform(n, S);
            make_payout(n, k, payout);
            equity.assign(n, 0.0);

            IcmGpuOptions opts = default_opts();
            IcmGpuPlan *plan = icm_gpu_plan_create(n, S.data(), k, &opts);
            if (!plan) {
                fprintf(f, "%d,%d,nan,nan,error,0,0,nan,0,0,0,error\n", n, k);
                printf("FAIL(plan_create): %s\n", icm_gpu_last_error());
                fail_cells++;
                continue;
            }

            IcmGpuPlanSummary ps{};
            icm_gpu_plan_summary(plan, &ps);

            IcmGpuRunStats warm{};
            int warm_status = icm_gpu_equity_with_plan(plan, Q, payout.data(), equity.data(), &warm);
            if (warm_status != 0) {
                icm_gpu_plan_destroy(plan);
                fprintf(f, "%d,%d,nan,nan,error,0,0,nan,0,0,0,error\n", n, k);
                printf("FAIL(warmup): %s\n", icm_gpu_last_error());
                fail_cells++;
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
                printf("FAIL(exec): %s\n", icm_gpu_last_error());
                fail_cells++;
            } else {
                double time_ms = median_ms(samples);
                double cv = cv_ms(samples);
                double vram_mb = (double)stats.peak_vram_bytes / (1024.0 * 1024.0);
                std::string dom = dominant_tier(ps);
                fprintf(f, "%d,%d,%.6f,%.3f,%s,%d,%d,%.6f,%d,%d,%d,%s\n",
                        n, k, time_ms, vram_mb, engine_name(stats.engine).c_str(), stats.B,
                        (int)samples.size(), cv,
                        ps.n_tier1, ps.n_tier2, ps.n_tier3, dom.c_str());
                printf("OK %.2f ms  B=%d  peak=%.1f MB  reps=%d\n",
                       time_ms, stats.B, vram_mb, (int)samples.size());
                pass_cells++;
            }
        }
        fflush(f);
    }

    fclose(f);

    printf("\n=== DIAGNOSTIC SUMMARY ===\n");
    printf("Total cells: %d\n", total_cells);
    printf("Passed: %d\n", pass_cells);
    printf("Failed: %d\n", fail_cells);
    printf("Wrote %s\n", out_csv);

    if (fail_cells == 0) {
        printf("\nINTERPRETATION: All cells passed with cudaDeviceReset between cells.\n");
        printf("This CONFIRMS the CUDA allocator fragmentation hypothesis.\n");
        printf("The original sweep's failures are caused by allocator state accumulating\n");
        printf("across ~190 sequential plan create/destroy cycles, NOT by a genuine leak.\n");
    } else {
        printf("\nINTERPRETATION: %d cells still fail even with fresh allocator state.\n", fail_cells);
        printf("This REFUTES the fragmentation hypothesis.  The root cause is something\n");
        printf("that survives cudaDeviceReset — a driver bug, cuFFT-internal leak, or\n");
        printf("hardware issue.  Investigate further.\n");
    }

    icm_gpu_shutdown();
    return (fail_cells == 0) ? 0 : 1;
}
