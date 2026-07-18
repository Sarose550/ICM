/* gpu_sample_plans.cu — Sample GPU plans and measure per-Q-point runtime.
 *
 * For each (n, k, B) triple, creates a GPU plan, runs icm_gpu_equity,
 * and reports plan structure + measured runtime for cost model fitting.
 *
 * Build:
 *   make gpu_sample_plans CUDA_ARCH=sm_100
 *
 * Output CSV: n,k,B,qb,L,total_ms,per_qp_ns,levels...
 * where each level field is: tier:nr:cps:use_fft:fft_n:bwm:cache:cwm:below:g_need:out_needed
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "icm.h"
#include "icm_gpu.h"

/* Access internal plan structure for level details */
#include "gpu/gpu_internal.h"
using namespace icm_gpu_detail;

static void make_stacks(int n, std::vector<double> &S) {
    S.resize(n);
    srand(42);
    for (int i = 0; i < n; i++)
        S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
}

static void make_payout(int k, std::vector<double> &payout) {
    payout.resize(k);
    for (int q = 0; q < k; q++)
        payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);
}

static inline double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static void emit_plan(int n, int k, int forced_B) {
    std::vector<double> S, payout, equity(n);
    make_stacks(n, S);
    make_payout(k, payout);

    int Q = 256;

    IcmGpuOptions opts{};
    opts.use_cufftdx = 1;
    opts.memory_strategy = 2;
    opts.verbose = 0;
    opts.enable_graphs = 0;
    opts.enable_q_pipeline = 0;

    /* Force B via env var */
    char env_buf[32];
    snprintf(env_buf, sizeof(env_buf), "%d", forced_B);
    setenv("ICM_GPU_FORCE_B", env_buf, 1);

    IcmGpuPlan *plan_opaque = icm_gpu_plan_create(n, S.data(), k, &opts);
    if (!plan_opaque) {
        fprintf(stderr, "  SKIP n=%d k=%d B=%d: plan creation failed: %s\n",
                n, k, forced_B, icm_gpu_last_error());
        return;
    }
    auto *plan = reinterpret_cast<GpuPlan *>(plan_opaque);

    /* Warmup (1 rep) */
    icm_gpu_equity_with_plan(plan_opaque, Q, payout.data(), equity.data(), nullptr);

    /* Timed reps (3, take median) */
    double times[3];
    for (int rep = 0; rep < 3; rep++) {
        std::fill(equity.begin(), equity.end(), 0.0);
        double t0 = now_ns();
        icm_gpu_equity_with_plan(plan_opaque, Q, payout.data(), equity.data(), nullptr);
        cudaDeviceSynchronize();
        times[rep] = now_ns() - t0;
    }

    /* Median */
    std::sort(times, times + 3);
    double total_ns = times[1];
    double per_qp_ns = total_ns / Q;

    int actual_B = plan->B;
    int L = plan->L;
    int qb = plan->q_batch;

    /* CSV row */
    printf("%d,%d,%d,%d,%d,%.3f,%.1f",
           n, k, actual_B, qb, L, total_ns / 1e6, per_qp_ns);

    /* Per-level details */
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

    fprintf(stderr, "  n=%d k=%d B=%d qb=%d L=%d -> %.1f ms (%.0f ns/qp)\n",
            n, k, actual_B, qb, L, total_ns / 1e6, per_qp_ns);

    icm_gpu_plan_destroy(plan_opaque);
    unsetenv("ICM_GPU_FORCE_B");
}

int main(void) {
    icm_gpu_init(0);

    /* Sample space: diverse (n, k, B) covering small to frontier */
    int n_values[] = {
        512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576
    };
    int n_n = 12;

    int B_values[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int n_B = 8;

    double k_fracs[] = {0.1, 0.25, 0.5, 1.0};
    int n_kf = 4;

    printf("n,k,B,qb,L,total_ms,per_qp_ns,levels...\n");

    int count = 0;
    for (int ni = 0; ni < n_n; ni++) {
        int n = n_values[ni];
        for (int bi = 0; bi < n_B; bi++) {
            int B = B_values[bi];
            if (B > n) continue;
            for (int ki = 0; ki < n_kf; ki++) {
                int k = (int)(n * k_fracs[ki]);
                if (k < 4) k = 4;
                if (k > n) k = n;
                if (B > k) continue;

                /* Skip if estimated > 30 seconds */
                if ((double)n * (double)k > 5e11) continue;

                emit_plan(n, k, B);
                count++;

                if (count >= 250) goto done;
            }
        }
    }
done:
    fprintf(stderr, "\nSampled %d plans\n", count);
    return 0;
}
