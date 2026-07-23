/* calibrate_gpu_best_b.cu — Direct empirical measurement of the real
 * fastest hybrid-engine block size B(n,k) on the GPU, replacing
 * gpu_select_best_B_est()'s summed-analytical-constants prediction with
 * a small, per-device empirical lookup table — same methodology and
 * rationale as tools/calibrate_best_b.c on CPU (LAPACK ILAENV precedent:
 * measure the real decision directly rather than summing calibrated
 * constants). Confirmed via tools/validate_planner_gpu.cu that
 * gpu_select_best_B_est() is measurably wrong: 12/12 mismatches at
 * n>=65536, always picking B=128 when B=64 real-wins by 2-4%.
 *
 * For each (n,k) grid point, times the REAL hybrid engine at every
 * candidate B in a representative subset of kBCandidates (median of 3
 * reps — GPU runs are far less noisy than CPU wall-clock timing, so 3
 * reps suffices where CPU needed 7), and records the empirically-fastest
 * B. This is a ONE-TIME, OFFLINE calibration step — it never runs in
 * production.
 *
 * Build:
 *   make calibrate_gpu_best_b CUDA_ARCH=sm_100 CUFFTDX_INC=-I<path>
 * Run:
 *   ./calibrate_gpu_best_b > gpu_best_b_b200.csv
 */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "icm_gpu.h"

#define N_REPS 3

static void make_stacks_uniform(int n, std::vector<double> &S) {
    S.resize(n);
    srand(123 + n);
    for (int i = 0; i < n; ++i) S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}

static void make_payout(int n, int k, std::vector<double> &payout) {
    payout.resize(k);
    for (int m = 0; m < k; ++m) payout[m] = (double)(n - m);
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) ? 1 : (da < db) ? -1 : 0;
}

/* Median-of-N_REPS timing for one (n,k,B) case. Returns -1.0 on failure. */
static double time_case(int n, int k, int Q, int force_B) {
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

    double times[N_REPS];
    int n_ok = 0;
    for (int r = 0; r < N_REPS; r++) {
        IcmGpuRunStats stats{};
        int status = icm_gpu_equity(n, S.data(), Q, payout.data(), k, eq.data(), &opts, &stats);
        if (status != 0) continue;
        times[n_ok++] = stats.total_ns / 1e6;
    }
    if (n_ok == 0) return -1.0;
    qsort(times, n_ok, sizeof(double), cmp_double);
    return times[n_ok / 2];
}

int main(int argc, char **argv) {
    int Q = (argc > 1) ? atoi(argv[1]) : 256;

    if (!icm_gpu_init(0)) {
        fprintf(stderr, "icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    /* n grid: spans from where the CPU/GPU crossover matters up through
     * the ~1.5M frontier. k grid: n/8, n/4, n/2, n (skip if < 1). */
    std::vector<int> n_grid = {4096, 16384, 65536, 131072, 262144, 524288, 1048576, 1572864};

    /* Representative subset of kBCandidates (src/gpu/gpu_plan.cu) — full
     * 48-candidate sweep would be excessive for a calibration pass;
     * this subset brackets the B=64/B=128 region confirmed wrong plus
     * enough coverage elsewhere to catch a similar bias at other n. */
    std::vector<int> b_grid = {
        16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256,
        320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536
    };

    printf("# Direct empirical GPU best-B measurement (median of %d reps, Q=%d)\n", N_REPS, Q);
    printf("# n,k,best_B\n");

    for (int n : n_grid) {
        std::vector<int> k_grid = {n / 8, n / 4, n / 2, n};
        for (int k : k_grid) {
            if (k < 1) continue;

            double best_ms = 1e30;
            int best_B = 0;
            for (int B : b_grid) {
                if (B > n || B > k) continue;
                double t_ms = time_case(n, k, Q, B);
                if (t_ms < 0.0) continue; /* plan creation failure (VRAM), skip */
                if (t_ms < best_ms) {
                    best_ms = t_ms;
                    best_B = B;
                }
            }
            if (best_B > 0) {
                printf("%d,%d,%d\n", n, k, best_B);
                fflush(stdout);
                fprintf(stderr, "n=%d k=%d -> best_B=%d (%.3f ms)\n", n, k, best_B, best_ms);
            }
        }
    }

    return 0;
}
