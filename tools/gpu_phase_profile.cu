/* gpu_phase_profile.cu — Per-phase GPU timing for cost model fitting.
 *
 * Runs icm_gpu_equity_with_plan in verbose (instrumented) mode to get
 * per-phase breakdown: compute_a, block_build, tree_build, tree_prop_cached,
 * tree_prop_recomp, leaf_extract, accumulate.
 *
 * Uses single-Q mode (qb=1) to isolate per-Q-point phase costs without
 * q-batch amortization effects.
 *
 * Output CSV: n,k,B,qb,L,total_ns,compute_a_ns,block_ns,tree_build_ns,
 *             tree_prop_cached_ns,tree_prop_recomp_ns,leaf_ns,accum_ns,overhead_ns
 *             + per-level plan details (same format as gpu_sample_plans)
 *
 * Build: make gpu_phase_profile CUDA_ARCH=sm_100
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "icm.h"
#include "icm_gpu.h"
#include "gpu/gpu_internal.h"

using namespace icm_gpu_detail;

static inline double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static void profile_plan(int n, int k, int forced_B) {
    std::vector<double> S(n), payout(k), equity(n);
    srand(42);
    for (int i = 0; i < n; i++) S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
    for (int q = 0; q < k; q++) payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

    IcmGpuOptions opts{};
    opts.use_cufftdx = 1;
    opts.memory_strategy = 2;
    opts.verbose = 1;  // enables instrumented path with per-phase timing
    opts.enable_graphs = 0;
    opts.enable_q_pipeline = 0;
    setenv("ICM_GPU_Q_BATCH", "1", 1);  // force single-Q mode to get per-phase breakdown

    char env_buf[32];
    snprintf(env_buf, sizeof(env_buf), "%d", forced_B);
    setenv("ICM_GPU_FORCE_B", env_buf, 1);

    IcmGpuPlan *plan_opaque = icm_gpu_plan_create(n, S.data(), k, &opts);
    if (!plan_opaque) {
        fprintf(stderr, "  SKIP n=%d k=%d B=%d: %s\n", n, k, forced_B, icm_gpu_last_error());
        return;
    }
    auto *plan = reinterpret_cast<GpuPlan *>(plan_opaque);
    int Q = 64;  // fewer Q-points since instrumented path is slower (per-kernel sync)

    // Warmup
    icm_gpu_equity_with_plan(plan_opaque, Q, payout.data(), equity.data(), nullptr);

    // Timed run with stats
    IcmGpuRunStats stats{};
    std::fill(equity.begin(), equity.end(), 0.0);
    double t0 = now_ns();
    icm_gpu_equity_with_plan(plan_opaque, Q, payout.data(), equity.data(), &stats);
    cudaDeviceSynchronize();
    double total_ns = now_ns() - t0;

    double per_qp = total_ns / Q;
    double block_per_qp = stats.block_build_ns / Q;
    double build_per_qp = stats.tree_build_ns / Q;
    double prop_cached_per_qp = stats.tree_propagate_cached_ns / Q;
    double prop_recomp_per_qp = stats.tree_propagate_recomputed_ns / Q;
    double leaf_per_qp = stats.leaf_extract_ns / Q;

    double known_phases = block_per_qp + build_per_qp + prop_cached_per_qp
                        + prop_recomp_per_qp + leaf_per_qp;
    double overhead_per_qp = per_qp - known_phases;
    // overhead includes: compute_a, set_root_g, accumulate_equity, memcpy, k_zero

    int actual_B = plan->B;
    int L = plan->L;
    int qb = plan->q_batch;

    // CSV: plan info + phase breakdown
    printf("%d,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f",
           n, k, actual_B, qb, L,
           per_qp, overhead_per_qp, block_per_qp, build_per_qp,
           prop_cached_per_qp, prop_recomp_per_qp, leaf_per_qp);

    // Per-level details (same format as gpu_sample_plans)
    for (int ell = 1; ell < L; ell++) {
        auto &lp = plan->levels[ell];
        int nr = plan->n_real[ell];
        int cps = plan->psz[ell - 1];
        int below = plan->below_sat[ell];
        int g_need = plan->g_needed[ell - 1];
        printf(",%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
               lp.tier, nr, cps, lp.use_fft, lp.fft_n,
               lp.build_wrap_m, lp.cache_fft,
               lp.corr_wrap_m, below, g_need, lp.out_needed);
    }
    printf("\n");
    fflush(stdout);

    fprintf(stderr, "  n=%d k=%d B=%d: total=%.0f block=%.0f build=%.0f prop_c=%.0f prop_r=%.0f leaf=%.0f overhead=%.0f ns/qp\n",
            n, k, actual_B,
            per_qp, block_per_qp, build_per_qp, prop_cached_per_qp,
            prop_recomp_per_qp, leaf_per_qp, overhead_per_qp);

    icm_gpu_plan_destroy(plan_opaque);
    unsetenv("ICM_GPU_FORCE_B");
}

int main(void) {
    icm_gpu_init(0);

    // Representative (n, k, B) triples spanning the space
    // Fewer than sample_plans — these are slower due to per-kernel sync
    struct Triple { int n, k, B; };
    std::vector<Triple> triples;

    int n_vals[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int B_vals[] = {8, 32, 64, 128, 256, 512, 1024};

    for (int ni = 0; ni < 8; ni++) {
        int n = n_vals[ni];
        for (int bi = 0; bi < 7; bi++) {
            int B = B_vals[bi];
            if (B > n) continue;
            int k = n;  // k=n is the hardest case
            if (B > k) continue;
            if ((double)n * k > 5e11) continue;
            triples.push_back({n, k, B});
        }
    }

    printf("n,k,B,qb,L,per_qp_ns,overhead_ns,block_ns,tree_build_ns,"
           "tree_prop_cached_ns,tree_prop_recomp_ns,leaf_ns,levels...\n");

    for (auto &t : triples) {
        profile_plan(t.n, t.k, t.B);
    }

    fprintf(stderr, "\nProfiled %zu plans\n", triples.size());
    return 0;
}
