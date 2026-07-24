/* validate_planner_gpu.cu — Single-point probe oracle for the GPU planner.
 *
 * Given ONE (n,k) via CLI args, runs the real GPU hybrid engine at the
 * planner's chosen B (auto) and at every candidate B in the full set,
 * then reports auto_B, auto_ms, best_B, best_ms, gap_pct on one
 * machine-readable line to stdout.
 *
 * The orchestration of which points to probe is handled by
 * tools/calibrate_block_size.py (A4), which calls this binary
 * one point at a time as its oracle.
 *
 * Usage:
 *   ./validate_planner_gpu <n> <k> [Q]
 *
 * Output (one line to stdout):
 *   auto_B,auto_ms,best_B,best_ms,gap_pct
 *
 *   auto_B    — planner's chosen B (from gpu_empirical_best_B / gpu_select_best_B_est)
 *   auto_ms   — measured wall-clock time for auto_B (milliseconds)
 *   best_B    — empirically-fastest B from full candidate sweep
 *   best_ms   — measured wall-clock time for best_B (milliseconds)
 *   gap_pct   — (auto_ms - best_ms) / best_ms * 100  (positive = auto slower)
 *
 * Build:
 *   make validate_planner_gpu CUDA_ARCH=sm_100 CUFFTDX_INC=-I<path>
 */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "icm_gpu.h"

/* Full candidate B set — must match calibrate_gpu_best_b.cu and
 * kBCandidates in src/gpu/gpu_plan.cu. */
static const std::vector<int> kBCandidates = {
    16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256,
    320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536
};

/* ── Utilities ────────────────────────────────────────────────────────── */

static void make_stacks_uniform(int n, std::vector<double> &S) {
    S.resize(n);
    srand(123 + n);
    for (int i = 0; i < n; ++i) S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}

static void make_payout(int n, int k, std::vector<double> &payout) {
    payout.resize(k);
    for (int m = 0; m < k; ++m) payout[m] = (double)(n - m);
}

/* Run a single (n,k,Q,force_B) case.  Returns time in ms, or -1.0 on
 * failure.  If out_B is non-null, stores the actual B used (planner's
 * choice when force_B=0; force_B otherwise). */
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
    int status = icm_gpu_equity(n, S.data(), Q, payout.data(), k,
                                 eq.data(), &opts, &stats);
    if (status != 0) return -1.0;
    if (out_B) *out_B = stats.B;
    return stats.total_ns / 1e6;
}

/* ── Main ─────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <n> <k> [Q] [--dry-run]\n"
                "  Single-point probe oracle for the GPU planner.\n"
                "  Output: auto_B,auto_ms,best_B,best_ms,gap_pct\n",
                argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    int Q = 256;
    bool dry_run = false;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--dry-run") == 0) {
            dry_run = true;
        } else {
            Q = atoi(argv[i]);
        }
    }

    if (n < 1 || k < 1 || k > n || Q < 1) {
        fprintf(stderr, "Invalid args: n=%d k=%d Q=%d\n", n, k, Q);
        return 1;
    }

    if (dry_run) {
        printf("=== DRY RUN (no CUDA calls made) ===\n");
        printf("n             = %d\n", n);
        printf("k             = %d\n", k);
        printf("Q             = %d\n", Q);
        printf("candidates    = [");
        for (size_t i = 0; i < kBCandidates.size(); i++) {
            if (i > 0) printf(", ");
            printf("%d", kBCandidates[i]);
        }
        printf("]\n");
        printf("Would output: auto_B,auto_ms,best_B,best_ms,gap_pct\n");
        return 0;
    }

    if (!icm_gpu_init(0)) {
        fprintf(stderr, "icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    /* ── Auto (planner's choice) ── */
    int auto_B = 0;
    double auto_ms = run_case(n, k, Q, 0, &auto_B);
    if (auto_ms < 0.0) {
        fprintf(stderr, "auto run failed: %s\n", icm_gpu_last_error());
        icm_gpu_shutdown();
        return 1;
    }

    /* ── Full sweep to find empirically-best B ── */
    double best_ms = 1e30;
    int best_B = 0;
    for (int B : kBCandidates) {
        if (B > n) continue;
        double t_ms = run_case(n, k, Q, B, nullptr);
        if (t_ms < 0.0) continue;
        if (t_ms < best_ms) {
            best_ms = t_ms;
            best_B  = B;
        }
    }

    if (best_B == 0) {
        fprintf(stderr, "all candidate sweeps failed: %s\n", icm_gpu_last_error());
        icm_gpu_shutdown();
        return 1;
    }

    /* ── Report ── */
    double gap_pct = (best_ms > 0.0)
                     ? 100.0 * (auto_ms - best_ms) / best_ms
                     : 0.0;

    /* Machine-readable single line to stdout — parsed by
     * tools/calibrate_block_size.py (A4).
     * Columns: auto_B,auto_ms,best_B,best_ms,gap_pct */
    printf("%d,%.6f,%d,%.6f,%.2f\n",
           auto_B, auto_ms, best_B, best_ms, gap_pct);

    unsetenv("ICM_GPU_FORCE_B");
    icm_gpu_shutdown();
    return 0;
}
