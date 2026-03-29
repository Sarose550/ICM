#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "icm.h"
#include "icm_gpu.h"

static void make_stacks(int n, std::vector<double> &S) {
    S.resize(n);
    srand(42);
    for (int i = 0; i < n; ++i) S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}

static void make_payout(int n, int k, std::vector<double> &p) {
    p.resize(k);
    for (int i = 0; i < k; ++i) p[i] = (double)(n - i);
}

int main() {
    const int n = 64;
    const int k = 64;
    const int Q = 256;

    std::vector<double> S, payout, cpu(n), gpu(n);
    make_stacks(n, S);
    make_payout(n, k, payout);

    icm_init(nullptr);
    double t_cpu = icm_equity(n, S.data(), Q, payout.data(), k, cpu.data());

    if (!icm_gpu_init(0)) {
        printf("gpu init fail: %s\n", icm_gpu_last_error());
        return 1;
    }

    IcmGpuOptions opts{};
    opts.device_id = 0;
    opts.use_cufftdx = 1;
    opts.enable_graphs = 0;
    opts.enable_q_pipeline = 0;
    opts.memory_strategy = 0;
    opts.force_uncached_fused_levels = -1;
    opts.force_uncached_cufft_levels = -1;

    IcmGpuRunStats st{};
    double t_gpu = icm_gpu_equity(n, S.data(), Q, payout.data(), k, gpu.data(), &opts, &st);
    if (t_gpu < 0) {
        printf("gpu run fail: %s\n", icm_gpu_last_error());
        return 2;
    }

    double sum_cpu = 0.0;
    double sum_gpu = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    int argmax = -1;
    for (int i = 0; i < n; ++i) {
        sum_cpu += cpu[i];
        sum_gpu += gpu[i];
        double d = fabs(cpu[i] - gpu[i]);
        double s = fabs(cpu[i]);
        if (s < 1e-14) s = 1.0;
        double r = d / s;
        if (r > max_rel) {
            max_rel = r;
            max_abs = d;
            argmax = i;
        }
    }

    printf("cpu_ms=%.3f gpu_ms=%.3f B=%d\n", t_cpu / 1e6, t_gpu / 1e6, st.B);
    printf("sum_cpu=%.17g sum_gpu=%.17g diff=%.17g\n", sum_cpu, sum_gpu, sum_gpu - sum_cpu);
    printf("max_rel=%.6e max_abs=%.6e at i=%d cpu=%.17g gpu=%.17g\n",
           max_rel, max_abs, argmax, cpu[argmax], gpu[argmax]);
    for (int i = 0; i < 10; ++i) {
        printf("i=%d cpu=%.17g gpu=%.17g diff=%.3e\n", i, cpu[i], gpu[i], gpu[i] - cpu[i]);
    }
    return 0;
}
