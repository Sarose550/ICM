/*
 * icm_cuda_bench.cu — Comprehensive ICM benchmark (GPU + optional CPU comparison)
 *
 * Produces CSV files for plotting:
 *   accuracy_vs_q.csv   — error by Q for each distribution
 *   time_vs_q.csv       — kernel time by Q for each distribution  
 *   time_vs_n.csv       — kernel time by n (adversarial)
 *   max_n_under_1s.csv  — largest n computable under various time budgets
 *   scaling.csv          — full (n, Q, dist, time, error) sweep
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -o icm_cuda_bench icm_cuda_bench.cu -lm
 *
 * Run:
 *   ./icm_cuda_bench              # full suite
 *   ./icm_cuda_bench --quick      # reduced sweep for testing
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <unistd.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TILE_M 16
#define BUILD_THREADS 256

static int next_pow2(int v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/* ================================================================
   Host-side: quadrature (identical to icm_cuda.cu)
   ================================================================ */

typedef struct { double x, w; } QP;

static inline double log_sigma_h(double x) {
    return (x >= 0) ? -log1p(exp(-x)) : x - log1p(exp(x));
}
static double log_Phi_h(double y) {
    if (y >= 0) { double ec = erfc(y / sqrt(2.0)); return log1p(-ec / 2.0); }
    else        { double ec = erfc(-y / sqrt(2.0)); return log(ec / 2.0); }
}
static void erfc_domain_h(double Smax, double *ylo, double *yhi) {
    double lo = -20, hi = 0;
    for (int i = 0; i < 100; i++) { double m = (lo+hi)/2; if (log_Phi_h(m) < -25) lo = m; else hi = m; }
    *ylo = lo - 1;
    lo = 0; hi = 20; double tgt = 1e-10 / Smax;
    for (int i = 0; i < 100; i++) { double m = (lo+hi)/2; if (-log_Phi_h(m) > tgt) lo = m; else hi = m; }
    *yhi = hi + 1;
}
static double y_to_xlog_h(double y) {
    double lv = log_Phi_h(y), l1mv;
    if (y >= 0) { double ec = erfc(y / sqrt(2.0)); l1mv = log(ec / 2.0); }
    else        { double ec = erfc(-y / sqrt(2.0)); l1mv = log1p(-ec / 2.0); }
    return lv - l1mv;
}
static void make_nodes_h(int Q, double Smax, QP *pts) {
    double yl, yh; erfc_domain_h(Smax, &yl, &yh);
    double h = (yh - yl) / (Q - 1);
    for (int q = 0; q < Q; q++) {
        double y = yl + q * h, phi = exp(-y*y/2) / sqrt(2*M_PI);
        pts[q].x = y_to_xlog_h(y); pts[q].w = h * phi;
        if (q == 0 || q == Q-1) pts[q].w *= 0.5;
    }
}
static void exact_V1_h(int n, const double *S, double *V1) {
    for (int i = 0; i < n; i++) {
        double v = 1;
        for (int j = 0; j < n; j++) if (j != i) v += S[i] / (S[i] + S[j]);
        V1[i] = v;
    }
}
static double max_relV1_h(int n, const double *prob, const double *eV1) {
    double mx = 0;
    for (int i = 0; i < n; i++) {
        const double *r = prob + (size_t)i * n;
        double nv = 0;
        for (int m = 0; m < n; m++) nv += (double)(n - m) * r[m];
        double re = (eV1[i] != 0) ? fabs(nv - eV1[i]) / fabs(eV1[i]) : fabs(nv);
        if (re > mx) mx = re;
    }
    return mx;
}

/* ================================================================
   Device code (kernels)
   ================================================================ */

__device__ __forceinline__ double log_sigma_d(double x) {
    return (x >= 0.0) ? -log1p(exp(-x)) : x - log1p(exp(x));
}

__global__ void kernel_build(int n, int Q, const double *S,
                             const double * __restrict__ logv_store,
                             double * __restrict__ P_store) {
    int q = blockIdx.x;
    if (q >= Q) return;
    extern __shared__ double smem[];
    double *P0 = smem;
    double *P1 = smem + (n + 1);
    double logv = logv_store[q];
    double *Pcur = P0, *Pnew = P1;
    if (threadIdx.x == 0) Pcur[0] = 1.0;
    for (int j = threadIdx.x + 1; j <= n; j += BUILD_THREADS) Pcur[j] = 0.0;
    __syncthreads();
    int deg = 0;
    for (int j = 0; j < n; j++) {
        double arg = S[j] * logv;
        double aj = (arg < -700.0) ? 0.0 : exp(arg);
        double bj = 1.0 - aj;
        int nd = (deg + 1 < n) ? deg + 1 : n;
        for (int m = threadIdx.x; m <= nd; m += BUILD_THREADS) {
            if (m == 0) Pnew[0] = aj * Pcur[0];
            else        Pnew[m] = aj * Pcur[m] + bj * Pcur[m - 1];
        }
        for (int m = nd + 1 + threadIdx.x; m <= n; m += BUILD_THREADS) Pnew[m] = 0.0;
        __syncthreads();
        double *tmp = Pcur; Pcur = Pnew; Pnew = tmp;
        deg = nd;
    }
    size_t base = (size_t)q * (n + 1);
    for (int m = threadIdx.x; m <= n; m += BUILD_THREADS) P_store[base + m] = Pcur[m];
}

__global__ void kernel_divide(int n, int Q, int Qpad,
                              const double * __restrict__ S,
                              const double * __restrict__ P_store,
                              const double * __restrict__ logv_store,
                              const double * __restrict__ wq_store,
                              double * __restrict__ prob) {
    int i = blockIdx.x;
    if (i >= n) return;
    int q = threadIdx.x;
    extern __shared__ double red[];
    double Si = S[i], Si_m1 = Si - 1.0;
    double logv = 0, wq = 0, ai = 0, bi = 0, pw = 0;
    int is_bu = 1;
    if (q < Q) {
        logv = logv_store[q]; wq = wq_store[q];
        double arg = Si * logv;
        ai = (arg < -700.0) ? 0.0 : exp(arg);
        bi = 1.0 - ai;
        double lw = Si_m1 * logv;
        double vp = (lw < -700.0) ? 0.0 : exp(lw);
        pw = wq * Si * vp;
        if (pw != pw || wq == 0) pw = 0;
        is_bu = (ai > 0.5) ? 1 : 0;
    }
    double inv_d = 0;
    if (pw != 0) inv_d = is_bu ? (1.0 / ai) : (1.0 / bi);
    const double *Pq = P_store + (size_t)q * (n + 1);
    double qm_bu = 0, qm_td = 0;
    if (pw != 0) {
        if (is_bu) qm_bu = Pq[0] * inv_d;
        else       qm_td = Pq[n] * inv_d;
    }
    int n_tiles = (n + TILE_M - 1) / TILE_M;
    int bu_m = 0, td_m = n - 1;
    double *prob_row = prob + (size_t)i * n;
    for (int tile = 0; tile < n_tiles; tile++) {
        int tile_size = TILE_M;
        if (tile * TILE_M + tile_size > n) tile_size = n - tile * TILE_M;
        for (int dm = 0; dm < tile_size; dm++) {
            double bu_val = 0, td_val = 0;
            if (pw != 0 && is_bu) {
                if (bu_m == 0) bu_val = pw * qm_bu;
                else { qm_bu = (Pq[bu_m] - bi * qm_bu) * inv_d; bu_val = pw * qm_bu; }
                bu_m++;
            }
            if (pw != 0 && !is_bu) {
                if (td_m == n - 1) td_val = pw * qm_td;
                else { qm_td = (Pq[td_m + 1] - ai * qm_td) * inv_d; td_val = pw * qm_td; }
                td_m--;
            }
            red[threadIdx.x * 2 * TILE_M + dm] = bu_val;
            red[threadIdx.x * 2 * TILE_M + TILE_M + dm] = td_val;
        }
        for (int dm = tile_size; dm < TILE_M; dm++) {
            red[threadIdx.x * 2 * TILE_M + dm] = 0;
            red[threadIdx.x * 2 * TILE_M + TILE_M + dm] = 0;
        }
        __syncthreads();
        for (int stride = Qpad / 2; stride > 0; stride >>= 1) {
            if ((int)threadIdx.x < stride) {
                int bm = threadIdx.x * 2 * TILE_M, bo = (threadIdx.x + stride) * 2 * TILE_M;
                for (int dm = 0; dm < 2 * TILE_M; dm++) red[bm + dm] += red[bo + dm];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            int bu_base = tile * TILE_M;
            int td_base = (n - 1) - tile * TILE_M;
            for (int dm = 0; dm < tile_size; dm++) {
                prob_row[bu_base + dm] += red[dm];
                int td_idx = td_base - dm;
                if (td_idx >= 0 && td_idx < n) prob_row[td_idx] += red[TILE_M + dm];
            }
        }
        __syncthreads();
    }
}

/* ================================================================
   Host driver
   ================================================================ */

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* Run kernels and return kernel time in ms. prob_host gets the result. */
static float run_icm_gpu(int n, const double *S_host, int Q, const QP *pts_host,
                         double *prob_host, int reps) {
    double *logv_h = (double *)malloc(Q * sizeof(double));
    double *wq_h   = (double *)malloc(Q * sizeof(double));
    for (int q = 0; q < Q; q++) {
        logv_h[q] = log_sigma_h(pts_host[q].x);
        wq_h[q]   = pts_host[q].w;
    }

    double *d_S, *d_logv, *d_wq, *d_P, *d_prob;
    size_t P_sz = (size_t)Q * (n + 1) * sizeof(double);
    size_t prob_sz = (size_t)n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_S,    n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_logv, Q * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_wq,   Q * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P,    P_sz));
    CUDA_CHECK(cudaMalloc(&d_prob, prob_sz));
    CUDA_CHECK(cudaMemcpy(d_S,    S_host, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_logv, logv_h, Q * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wq,   wq_h,   Q * sizeof(double), cudaMemcpyHostToDevice));

    int Qpad = next_pow2(Q);
    size_t build_smem = (size_t)2 * (n + 1) * sizeof(double);
    size_t div_smem   = (size_t)Qpad * 2 * TILE_M * sizeof(double);
    if (build_smem > 48 * 1024)
        cudaFuncSetAttribute(kernel_build, cudaFuncAttributeMaxDynamicSharedMemorySize, build_smem);
    if (div_smem > 48 * 1024)
        cudaFuncSetAttribute(kernel_divide, cudaFuncAttributeMaxDynamicSharedMemorySize, div_smem);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Warmup */
    CUDA_CHECK(cudaMemset(d_prob, 0, prob_sz));
    kernel_build<<<Q, BUILD_THREADS, build_smem>>>(n, Q, d_S, d_logv, d_P);
    kernel_divide<<<n, Qpad, div_smem>>>(n, Q, Qpad, d_S, d_P, d_logv, d_wq, d_prob);
    CUDA_CHECK(cudaDeviceSynchronize());

    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        CUDA_CHECK(cudaMemset(d_prob, 0, prob_sz));
        CUDA_CHECK(cudaEventRecord(start));
        kernel_build<<<Q, BUILD_THREADS, build_smem>>>(n, Q, d_S, d_logv, d_P);
        kernel_divide<<<n, Qpad, div_smem>>>(n, Q, Qpad, d_S, d_P, d_logv, d_wq, d_prob);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }

    CUDA_CHECK(cudaMemcpy(prob_host, d_prob, prob_sz, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_S); cudaFree(d_logv); cudaFree(d_wq); cudaFree(d_P); cudaFree(d_prob);
    free(logv_h); free(wq_h);
    return best;
}

/* ================================================================
   Stack distributions
   ================================================================ */

static void make_stacks(int n, double ratio, int dist, double *S) {
    switch (dist) {
    case 0: for (int i=0;i<n;i++) S[i]=1; S[0]=ratio; break;
    case 1: for (int i=0;i<n;i++) S[i]=ratio; S[0]=1; break;
    case 2: for (int i=0;i<n;i++) S[i]=(i<n/2)?1:ratio; break;
    case 3: for (int i=0;i<n;i++) S[i]=pow(ratio,(double)i/(n-1)); break;
    case 4: { srand(42); double mn=1e30;
        for (int i=0;i<n;i++){S[i]=1+(ratio-1)*((double)rand()/RAND_MAX);if(S[i]<mn)mn=S[i];}
        for (int i=0;i<n;i++) S[i]/=mn; break; }
    }
}

/* ================================================================
   Main
   ================================================================ */

int main(int argc, char **argv) {
    int quick = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--quick") == 0) quick = 1;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "GPU: %s (SM %d.%d, %d SMs, %.0f MHz, %.1f GB, L2=%.0f KB)\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount,
            prop.clockRate / 1000.0, prop.totalGlobalMem / 1e9,
            prop.l2CacheSize / 1024.0);
    fprintf(stderr, "FP64 throughput: ~%.1f TFLOPS (estimated)\n\n",
            (double)prop.multiProcessorCount * prop.clockRate * 1e-6 * 64 / 1000.0);

    double ratio = 1e9;
    int reps = quick ? 3 : 5;
    const char *dnames[] = {"adversarial", "reverse_adv", "bimodal", "geometric", "uniform"};
    int ndist = 5;

    /* ============================================================
       Benchmark 1: Accuracy vs Q
       ============================================================ */
    fprintf(stderr, "=== Benchmark 1: Accuracy vs Q (n=512, ratio=1e9) ===\n");
    {
        FILE *f = fopen("accuracy_vs_q.csv", "w");
        fprintf(f, "Q,distribution,error\n");

        int n = 512;
        int Qs[] = {32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384};
        int nQ = quick ? 7 : 11;

        double *S    = (double *)malloc(n * sizeof(double));
        double *eV1  = (double *)malloc(n * sizeof(double));
        double *prob = (double *)malloc((size_t)n * n * sizeof(double));
        QP     *pts  = (QP *)malloc(384 * sizeof(QP));

        for (int di = 0; di < ndist; di++) {
            make_stacks(n, ratio, di, S);
            exact_V1_h(n, S, eV1);
            double Smax = 0;
            for (int i = 0; i < n; i++) if (S[i] > Smax) Smax = S[i];

            for (int qi = 0; qi < nQ; qi++) {
                int Q = Qs[qi];
                make_nodes_h(Q, Smax, pts);
                run_icm_gpu(n, S, Q, pts, prob, 1);
                double err = max_relV1_h(n, prob, eV1);
                fprintf(f, "%d,%s,%.6e\n", Q, dnames[di], err);
                fprintf(stderr, "  %s Q=%d: %.2e\n", dnames[di], Q, err);
            }
        }
        free(S); free(eV1); free(prob); free(pts);
        fclose(f);
        fprintf(stderr, "  -> accuracy_vs_q.csv\n\n");
    }

    /* ============================================================
       Benchmark 2: Time vs Q
       ============================================================ */
    fprintf(stderr, "=== Benchmark 2: Time vs Q (n=2048, adversarial) ===\n");
    {
        FILE *f = fopen("time_vs_q.csv", "w");
        fprintf(f, "Q,n,kernel_ms\n");

        int ns[] = {512, 1024, 2048};
        int n_count = quick ? 2 : 3;
        int Qs[] = {64, 96, 128, 160, 192, 224, 256, 320, 384};
        int nQ = quick ? 5 : 9;

        for (int ni = 0; ni < n_count; ni++) {
            int n = ns[ni];
            double *S    = (double *)malloc(n * sizeof(double));
            double *prob = (double *)malloc((size_t)n * n * sizeof(double));
            QP     *pts  = (QP *)malloc(384 * sizeof(QP));
            make_stacks(n, ratio, 0, S); /* adversarial */
            double Smax = ratio;

            for (int qi = 0; qi < nQ; qi++) {
                int Q = Qs[qi];
                make_nodes_h(Q, Smax, pts);
                float ms = run_icm_gpu(n, S, Q, pts, prob, reps);
                fprintf(f, "%d,%d,%.4f\n", Q, n, ms);
                fprintf(stderr, "  n=%d Q=%d: %.2f ms\n", n, Q, ms);
            }
            free(S); free(prob); free(pts);
        }
        fclose(f);
        fprintf(stderr, "  -> time_vs_q.csv\n\n");
    }

    /* ============================================================
       Benchmark 3: Time vs n (fixed Q=256)
       ============================================================ */
    fprintf(stderr, "=== Benchmark 3: Time vs n (Q=256, all distributions) ===\n");
    {
        FILE *f = fopen("time_vs_n.csv", "w");
        fprintf(f, "n,distribution,kernel_ms,error\n");

        int ns[] = {32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096};
        int n_count = quick ? 7 : 11;
        int Q = 256;

        for (int ni = 0; ni < n_count; ni++) {
            int n = ns[ni];
            /* Check memory: prob matrix = n*n*8 bytes */
            size_t prob_bytes = (size_t)n * n * sizeof(double);
            size_t P_bytes    = (size_t)Q * (n + 1) * sizeof(double);
            size_t total = prob_bytes + P_bytes + n * 8 + Q * 16;
            if (total > (size_t)(prop.totalGlobalMem * 0.8)) {
                fprintf(stderr, "  n=%d: skipping (%.1f GB > 80%% of GPU mem)\n",
                        n, total / 1e9);
                continue;
            }

            double *S    = (double *)malloc(n * sizeof(double));
            double *eV1  = (double *)malloc(n * sizeof(double));
            double *prob = (double *)malloc(prob_bytes);
            QP     *pts  = (QP *)malloc(Q * sizeof(QP));

            for (int di = 0; di < ndist; di++) {
                make_stacks(n, ratio, di, S);
                exact_V1_h(n, S, eV1);
                double Smax = 0;
                for (int i = 0; i < n; i++) if (S[i] > Smax) Smax = S[i];
                make_nodes_h(Q, Smax, pts);

                float ms = run_icm_gpu(n, S, Q, pts, prob, reps);
                double err = max_relV1_h(n, prob, eV1);
                fprintf(f, "%d,%s,%.4f,%.6e\n", n, dnames[di], ms, err);
                fprintf(stderr, "  n=%d %s: %.2f ms (err=%.2e)\n", n, dnames[di], ms, err);
            }
            free(S); free(eV1); free(prob); free(pts);
        }
        fclose(f);
        fprintf(stderr, "  -> time_vs_n.csv\n\n");
    }

    /* ============================================================
       Benchmark 4: Maximum n under time budget
       ============================================================ */
    fprintf(stderr, "=== Benchmark 4: Max n under time budget (Q=256, adversarial) ===\n");
    {
        FILE *f = fopen("max_n_under_budget.csv", "w");
        fprintf(f, "budget_ms,max_n,actual_ms,error\n");

        double budgets[] = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0};
        int n_budgets = quick ? 6 : 10;
        int Q = 256;

        /* Binary search for max n under each budget */
        for (int bi = 0; bi < n_budgets; bi++) {
            double budget = budgets[bi];
            int lo = 4, hi = 10240;

            /* Upper bound: check if hi is feasible memory-wise */
            while (hi > lo) {
                size_t prob_bytes = (size_t)hi * hi * sizeof(double);
                if (prob_bytes > (size_t)(prop.totalGlobalMem * 0.7)) { hi /= 2; continue; }
                break;
            }

            int best_n = lo;
            float best_ms = 0;
            double best_err = 0;

            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                /* Round to multiple of 4 */
                mid = (mid / 4) * 4;
                if (mid < 4) mid = 4;

                size_t prob_bytes = (size_t)mid * mid * sizeof(double);
                double *S    = (double *)malloc(mid * sizeof(double));
                double *eV1  = (double *)malloc(mid * sizeof(double));
                double *prob = (double *)malloc(prob_bytes);
                QP     *pts  = (QP *)malloc(Q * sizeof(QP));

                make_stacks(mid, ratio, 0, S);
                exact_V1_h(mid, S, eV1);
                make_nodes_h(Q, ratio, pts);

                float ms = run_icm_gpu(mid, S, Q, pts, prob, 3);
                double err = max_relV1_h(mid, prob, eV1);

                if (ms <= budget) {
                    best_n = mid;
                    best_ms = ms;
                    best_err = err;
                    lo = mid + 4;
                } else {
                    hi = mid - 4;
                }

                free(S); free(eV1); free(prob); free(pts);
            }

            fprintf(f, "%.1f,%d,%.4f,%.6e\n", budget, best_n, best_ms, best_err);
            fprintf(stderr, "  budget=%.0fms: max n=%d (%.2f ms, err=%.2e)\n",
                    budget, best_n, best_ms, best_err);
        }
        fclose(f);
        fprintf(stderr, "  -> max_n_under_budget.csv\n\n");
    }

    /* ============================================================
       Benchmark 5: Full scaling sweep (for comprehensive plot)
       ============================================================ */
    fprintf(stderr, "=== Benchmark 5: Full scaling sweep ===\n");
    {
        FILE *f = fopen("scaling.csv", "w");
        fprintf(f, "n,Q,distribution,kernel_ms,error,prob_MB\n");

        int ns[] = {64, 256, 512, 1024, 2048};
        int n_count = quick ? 3 : 5;
        int Qs[] = {128, 192, 256};
        int nQs = 3;

        for (int ni = 0; ni < n_count; ni++) {
            int n = ns[ni];
            for (int qi = 0; qi < nQs; qi++) {
                int Q = Qs[qi];
                double *S    = (double *)malloc(n * sizeof(double));
                double *eV1  = (double *)malloc(n * sizeof(double));
                double *prob = (double *)malloc((size_t)n * n * sizeof(double));
                QP     *pts  = (QP *)malloc(Q * sizeof(QP));

                for (int di = 0; di < ndist; di++) {
                    make_stacks(n, ratio, di, S);
                    exact_V1_h(n, S, eV1);
                    double Smax = 0;
                    for (int i = 0; i < n; i++) if (S[i] > Smax) Smax = S[i];
                    make_nodes_h(Q, Smax, pts);

                    float ms = run_icm_gpu(n, S, Q, pts, prob, reps);
                    double err = max_relV1_h(n, prob, eV1);
                    double prob_mb = (double)n * n * 8.0 / 1e6;
                    fprintf(f, "%d,%d,%s,%.4f,%.6e,%.1f\n",
                            n, Q, dnames[di], ms, err, prob_mb);
                }
                free(S); free(eV1); free(prob); free(pts);
            }
        }
        fclose(f);
        fprintf(stderr, "  -> scaling.csv\n\n");
    }

    fprintf(stderr, "All benchmarks complete. Run: python3 plot_results.py\n\n");

    /* Print summary table to stdout */
    printf("========================================\n");
    printf("       ICM GPU BENCHMARK SUMMARY\n");
    printf("========================================\n");
    printf("GPU: %s (%d SMs, %.0f MHz)\n\n", prop.name,
           prop.multiProcessorCount, prop.clockRate / 1000.0);

    if (access("max_n_under_budget.csv", 0) == 0) {
        printf("Max n under time budget (Q=256, adversarial):\n");
        printf("  %10s  %8s  %10s\n", "Budget", "Max n", "Actual");
        printf("  ----------------------------------\n");
        FILE *f = fopen("max_n_under_budget.csv", "r");
        char line[256];
        if (fgets(line, sizeof(line), f)) {}
        while (fgets(line, sizeof(line), f)) {
            double budget, actual;
            int max_n;
            double err;
            if (sscanf(line, "%lf,%d,%lf,%lf", &budget, &max_n, &actual, &err) == 4) {
                if (budget < 1.0)
                    printf("  %8.1f ms  %8d  %8.2f ms\n", budget, max_n, actual);
                else if (budget < 1000.0)
                    printf("  %8.0f ms  %8d  %8.1f ms\n", budget, max_n, actual);
                else
                    printf("  %7.0f ms  %8d  %7.0f ms\n", budget, max_n, actual);
            }
        }
        fclose(f);
    }

    printf("\nCSV files: accuracy_vs_q.csv, time_vs_q.csv, time_vs_n.csv,\n");
    printf("           max_n_under_budget.csv, scaling.csv\n");
    printf("Plot:      python3 plot_results.py\n");

    return 0;
}
