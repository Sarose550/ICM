/* Correctness + timing check for the below_sat generalization fix
 * (K1_IMPLEMENT_BELOW_SAT_FIX in SPRINT_GPU_MONOTONICITY_DAG.md).
 *
 * IMPORTANT split, learned the hard way this session: correctness
 * (GPU vs CPU reference) is ONLY checked at small sizes, safely within
 * every M3 Pro calibration table's covered range (crossover_n[] tops
 * out at 16384, bselect_n[] at 65536, calib_sizes[] at 131072 --
 * verified against devices/m3_pro/fft_config.h directly). bench_gpu_fused
 * links its CPU reference against those M3-Pro-calibrated tables
 * regardless of what CPU actually runs it; past their calibrated range,
 * the CPU reference's own per-level FFT-vs-schoolbook decision can fall
 * back to brute-force multiplication at a huge convolution length --
 * O(len^2) cost, which is why a prior run of the CPU reference at
 * n=1,000,000 ran for 45+ minutes and burned most of a B200 rental's
 * budget before being caught. That is a benchmark-tool characteristic,
 * unrelated to the GPU fix itself.
 *
 * For large sizes (where the original bug was found), ONLY a GPU run is
 * allowed -- no CPU reference call at all. --large below runs
 * icm_gpu_equity() directly and reports timing plus (via
 * ICM_GPU_DEBUG_PLAN=1, set by the caller) per-level plan decisions, so
 * the below_sat fix's effect can be confirmed structurally (does it fire
 * where expected?) and via timing (does the gap close?), without ever
 * invoking the slow CPU reference at that scale.
 *
 * Build (same recipe as the other scripts/ tools this session; uses
 * find with -path instead of a python-version wildcard glob, since a
 * literal "*" followed immediately by "/" closes a C block comment early -- see
 * HANDOFF.md's "Session tooling notes" for the same bug found twice
 * already in other scripts/ tools this session):
 *   CUFFTDX_INC=$(find /usr/local/lib -maxdepth 4 -type d -path '*dist-packages/nvidia/mathdx/include')
 *   nvcc -O3 -std=c++17 -arch=sm_100 -Isrc -Idevices/b200 -Isrc/gpu \
 *        -I"$CUFFTDX_INC" -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX \
 *        -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY \
 *        -dc -o build/verify_below_sat_fix.o scripts/verify_below_sat_fix.cu
 *   nvcc -O3 -std=c++17 -arch=sm_100 -o verify_below_sat_fix \
 *        build/verify_below_sat_fix.o build/gpu_gpu_kernels_fused.o \
 *        build/gpu_gpu_plan_fused.o build/gpu_gpu_exec_fused.o \
 *        build/gpu_gpu_api_fused.o build/gpu_dlink_fused.o \
 *        build/icm_cpu_ref.o -lfftw3 -lm -ldl -lmvec -lcufft -lcudart
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "icm.h"
#include "icm_gpu.h"

static void make_stacks(int n, std::vector<double> &S) {
    S.resize(n);
    srand(123 + n);
    for (int i = 0; i < n; ++i) S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}
static void make_payout(int n, int k, std::vector<double> &p) {
    p.resize(k);
    for (int m = 0; m < k; ++m) p[m] = (double)(n - m);
}

struct Point { int n, k; const char *tag; };

/* Small-scale points: same structural pattern as the original bug (k=n,
 * k itself 7-smooth but not a power of two, so k_pad=k exactly and the
 * below_sat transition boundary is exercised), all with n <= 16384 --
 * safely within EVERY M3 Pro calibration table's range (crossover up to
 * 16384, bselect up to 65536, FFT sizes up to 131072), so the CPU
 * reference stays fast and its own dispatch decisions stay within their
 * calibrated, trustworthy range. */
static const Point kPointsSmall[] = {
    {1000, 1000, "A: exact trigger case, scaled down 1000x"},
    {1024, 1024, "B: power-of-two baseline (no-op check)"},
    {500,  500,  "C: different miss pattern"},
    {513,  513,  "D: k=2^9+1, g_eff_max clamp boundary"},
    {750,  750,  "E: arbitrary non-power-of-two"},
    {10000, 10000, "F: mid-scale trigger case"},
    {16384, 16384, "G: mid-scale power-of-two baseline (at crossover ceiling)"},
};
static const int kNumSmall = sizeof(kPointsSmall) / sizeof(kPointsSmall[0]);

/* Large-scale points: the original scale where the bug was found.
 * GPU TIMING ONLY -- never call icm_equity() (the CPU reference) at
 * this scale, it is well beyond every M3 Pro calibration table's range
 * and can fall back to a catastrophically slow O(len^2) path. */
static const Point kPointsLarge[] = {
    {1000000, 1000000, "A-large: exact trigger case"},
    {1048576, 1048576, "B-large: power-of-two baseline (no-op check)"},
    {500000,  500000,  "C-large: different miss pattern"},
    {524289,  524289,  "D-large: k=2^19+1, g_eff_max clamp boundary"},
    {750000,  750000,  "E-large: arbitrary non-power-of-two"},
};
static const int kNumLarge = sizeof(kPointsLarge) / sizeof(kPointsLarge[0]);

/* Small-scale: full correctness check, GPU output vs CPU reference. */
static int run_small_correctness(const Point *pts, int n_pts, int Q) {
    int all_pass = 1;
    printf("%-10s %-10s %-38s %-10s %s\n", "n", "k", "tag", "max_rel_err", "verdict");
    for (int p = 0; p < n_pts; ++p) {
        int n = pts[p].n, k = pts[p].k;
        std::vector<double> S, payout, cpu_eq(n, 0.0), gpu_eq(n, 0.0);
        make_stacks(n, S);
        make_payout(n, k, payout);

        icm_equity(n, S.data(), Q, payout.data(), k, cpu_eq.data());

        IcmGpuOptions opts{};
        opts.device_id = 0;
        opts.use_cufftdx = 1;
        opts.enable_graphs = 0;
        opts.enable_q_pipeline = 1;
        opts.memory_strategy = 0;
        opts.force_uncached_fused_levels = -1;
        opts.force_uncached_cufft_levels = -1;
        IcmGpuRunStats stats{};
        int status = icm_gpu_equity(n, S.data(), Q, payout.data(), k,
                                     gpu_eq.data(), &opts, &stats);
        if (status != 0) {
            printf("%-10d %-10d %-38s GPU_FAIL: %s\n", n, k, pts[p].tag,
                   icm_gpu_last_error());
            all_pass = 0;
            continue;
        }

        double max_rel = 0.0;
        int max_i = -1;
        for (int i = 0; i < n; ++i) {
            double c = cpu_eq[i], g = gpu_eq[i];
            double denom = std::max(std::fabs(c), 1e-12);
            double r = std::fabs(g - c) / denom;
            if (r > max_rel) { max_rel = r; max_i = i; }
        }
        int pass = (max_rel < 1e-8);
        if (!pass) all_pass = 0;
        printf("%-10d %-10d %-38s %-10.3e %s (gpu_ms=%.1f, worst_i=%d cpu=%.17g gpu=%.17g)\n",
               n, k, pts[p].tag, max_rel, pass ? "PASS" : "FAIL",
               stats.total_ns / 1e6, max_i, cpu_eq[max_i], gpu_eq[max_i]);
        fflush(stdout);
    }
    return all_pass;
}

/* Large-scale: GPU-only. No CPU reference call. Reports timing; run with
 * ICM_GPU_DEBUG_PLAN=1 set by the caller to also see per-level plan
 * decisions (confirms below_sat fires where expected). */
static void run_large_gpu_only(const Point *pts, int n_pts, int Q) {
    printf("%-10s %-10s %-38s %-10s\n", "n", "k", "tag", "gpu_ms");
    for (int p = 0; p < n_pts; ++p) {
        int n = pts[p].n, k = pts[p].k;
        std::vector<double> S, payout, gpu_eq(n, 0.0);
        make_stacks(n, S);
        make_payout(n, k, payout);

        IcmGpuOptions opts{};
        opts.device_id = 0;
        opts.use_cufftdx = 1;
        opts.enable_graphs = 0;
        opts.enable_q_pipeline = 1;
        opts.memory_strategy = 0;
        opts.force_uncached_fused_levels = -1;
        opts.force_uncached_cufft_levels = -1;
        IcmGpuRunStats stats{};
        int status = icm_gpu_equity(n, S.data(), Q, payout.data(), k,
                                     gpu_eq.data(), &opts, &stats);
        if (status != 0) {
            printf("%-10d %-10d %-38s GPU_FAIL: %s\n", n, k, pts[p].tag,
                   icm_gpu_last_error());
            continue;
        }
        printf("%-10d %-10d %-38s %-10.1f\n", n, k, pts[p].tag, stats.total_ns / 1e6);
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    bool run_large = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--large") == 0) run_large = true;
    }

    icm_init(nullptr);
    if (!icm_gpu_init(0)) {
        fprintf(stderr, "icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    const int Q = 256;
    printf("=== Small-scale correctness check (GPU vs CPU reference) ===\n");
    int pass_small = run_small_correctness(kPointsSmall, kNumSmall, Q);

    if (run_large) {
        printf("\n=== Large-scale GPU-ONLY timing check (no CPU reference) ===\n");
        run_large_gpu_only(kPointsLarge, kNumLarge, Q);
    } else {
        printf("\n(skipping large-scale points; pass --large to include them --\n");
        printf(" GPU timing only, never a CPU reference call at that scale)\n");
    }

    icm_gpu_shutdown();
    printf("\n%s\n", pass_small ? "SMALL-SCALE CORRECTNESS CHECKS PASSED" : "SMALL-SCALE CHECKS FAILED");
    return pass_small ? 0 : 1;
}
