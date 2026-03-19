/*
 * icm_cuda.cu — A100-optimized ICM probability matrix computation
 *
 * Architecture overview:
 *
 *   Phase 1 (build): Q=256 independent polynomial builds.
 *     - One block per quad point, 256 threads cooperate on coefficient sweep.
 *     - Double-buffered shared memory to avoid WAR hazards.
 *     - Polynomial stored to global (L2-resident: 4.2 MB total).
 *
 *   Phase 2 (divide + accumulate): one block per player, Q=256 threads.
 *     - Each thread handles one quad point, runs the sequential division.
 *     - Bidirectional: bottom-up (a_i > 0.5) or top-down (a_i <= 0.5).
 *     - Tiled Q-reduction via shared memory, TILE_M coefficients at a time.
 *     - Two reductions per tile: one for BU contributions (forward m),
 *       one for TD contributions (reverse m). Avoids the direction mismatch.
 *     - Output: prob[i][m] written once per tile.
 *
 *   Memory budget at n=2048, Q=256:
 *     P_store:  4.2 MB  (L2)
 *     prob:    32   MB  (HBM, written once)
 *     Shared per block: ~80 KB (fits in 164 KB configurable)
 *     No temp buffer needed (reduction is on-chip).
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -o icm_cuda icm_cuda.cu -lm
 *
 * For A100 (sm_80). For H100: -arch=sm_90. For consumer: -arch=sm_86 (RTX 3090).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Max polynomial degree. Adjust if needed. */
#define MAX_N 2048
#define MAX_Q 256

/* Tile size for the coefficient-dimension reduction.
   Shared memory per block = Q * 2 * TILE_M * 8.
   With Q=256, TILE_M=16: 256 * 2 * 16 * 8 = 64 KB. */
#define TILE_M 16

static int next_pow2(int v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/* ================================================================
   Host-side: quadrature nodes (same as CPU version)
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

/* Validation */
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
   Device helpers
   ================================================================ */

__device__ __forceinline__ double log_sigma_d(double x) {
    return (x >= 0.0) ? -log1p(exp(-x)) : x - log1p(exp(x));
}

/* ================================================================
   Kernel 1: Build all Q polynomials
   
   Grid:  Q blocks
   Block: BUILD_THREADS threads (256)
   Shared: 2 × (n+1) doubles (double buffer) + n doubles (a-values)
   
   The polynomial build multiplies n linear factors. Each factor is an
   O(deg) parallel sweep across coefficients. Between factors: __syncthreads.
   Double buffering avoids the WAR hazard in the parallel sweep.
   ================================================================ */

#define BUILD_THREADS 256

__global__ void kernel_build(int n, int Q, const double *S,
                             const double * __restrict__ logv_store,
                             double * __restrict__ P_store) {
    int q = blockIdx.x;
    if (q >= Q) return;

    /* Dynamic shared memory: P_buf[2][n+1] — no a[] array needed */
    extern __shared__ double smem[];
    double *P0 = smem;
    double *P1 = smem + (n + 1);

    double logv = logv_store[q];

    /* Initialize P0 */
    double *Pcur = P0, *Pnew = P1;
    if (threadIdx.x == 0) Pcur[0] = 1.0;
    for (int j = threadIdx.x + 1; j <= n; j += BUILD_THREADS)
        Pcur[j] = 0.0;
    __syncthreads();

    int deg = 0;
    for (int j = 0; j < n; j++) {
        /* Compute a[j] on the fly — all threads read same S[j] (L2 cached) */
        double arg = S[j] * logv;
        double aj = (arg < -700.0) ? 0.0 : exp(arg);
        double bj = 1.0 - aj;
        int nd = (deg + 1 < n) ? deg + 1 : n;

        /* Parallel sweep: Pnew[m] = aj * Pcur[m] + bj * Pcur[m-1] */
        for (int m = threadIdx.x; m <= nd; m += BUILD_THREADS) {
            if (m == 0)
                Pnew[0] = aj * Pcur[0];
            else
                Pnew[m] = aj * Pcur[m] + bj * Pcur[m - 1];
        }
        /* Zero out higher coefficients (only needed on first few iterations) */
        for (int m = nd + 1 + threadIdx.x; m <= n; m += BUILD_THREADS)
            Pnew[m] = 0.0;
        __syncthreads();

        /* Swap buffers */
        double *tmp = Pcur; Pcur = Pnew; Pnew = tmp;
        deg = nd;
    }

    /* Write Pcur to global memory */
    size_t base = (size_t)q * (n + 1);
    for (int m = threadIdx.x; m <= n; m += BUILD_THREADS)
        P_store[base + m] = Pcur[m];
}

/* ================================================================
   Kernel 2: Divide + Accumulate with tiled Q-reduction
   
   Grid:  n blocks (one per player)
   Block: Q threads (one per quad point, Q <= 256)
   
   Each thread runs the division recurrence for its (player, quad point).
   Bottom-up threads produce coefficients m = 0, 1, ..., n-1 (forward).
   Top-down threads produce coefficients m = n-1, n-2, ..., 0 (reverse).
   
   At each tile of TILE_M coefficients, both BU and TD threads write
   their contributions to shared memory, then we do two tree reductions:
     - BU reduction  → prob[i][m_base + dm]       (forward coefficients)
     - TD reduction  → prob[i][(n-1-m_base) - dm]  (reverse coefficients)
   
   Shared memory: Q * 2 * TILE_M * 8 bytes.
     Q=256, TILE_M=16: 64 KB.
   ================================================================ */

__global__ void kernel_divide(int n, int Q, int Qpad,
                              const double * __restrict__ S,
                              const double * __restrict__ P_store,
                              const double * __restrict__ logv_store,
                              const double * __restrict__ wq_store,
                              double * __restrict__ prob) {
    int i = blockIdx.x;
    if (i >= n) return;
    int q = threadIdx.x;

    /* Shared memory for tiled reduction: [Qpad][2][TILE_M] */
    extern __shared__ double red[];

    double Si = S[i];
    double Si_m1 = Si - 1.0;

    double logv = 0, wq = 0, ai = 0, bi = 0, pw = 0;
    int is_bu = 1;

    if (q < Q) {
        logv = logv_store[q];
        wq = wq_store[q];
        double arg = Si * logv;
        ai = (arg < -700.0) ? 0.0 : exp(arg);
        bi = 1.0 - ai;
        double lw = Si_m1 * logv;
        double vp = (lw < -700.0) ? 0.0 : exp(lw);
        pw = wq * Si * vp;
        if (pw != pw || wq == 0) pw = 0;  /* NaN check + zero wq */
        is_bu = (ai > 0.5) ? 1 : 0;
    }

    /* Precompute inverse divisor */
    double inv_d = 0;
    if (pw != 0) {
        inv_d = is_bu ? (1.0 / ai) : (1.0 / bi);
    }

    /* Pointer to this quad point's polynomial */
    const double *Pq = P_store + (size_t)q * (n + 1);

    /* Running state for the recurrence */
    double qm_bu = 0, qm_td = 0;
    if (pw != 0) {
        if (is_bu) qm_bu = Pq[0] * inv_d;
        else       qm_td = Pq[n] * inv_d;
    }

    /* BU step counter goes 0, 1, 2, ..., n-1 (forward) */
    /* TD step counter goes n-1, n-2, ..., 0 (reverse) */
    int n_tiles = (n + TILE_M - 1) / TILE_M;

    int bu_m = 0;       /* next BU coefficient to produce: starts at 0 */
    int td_m = n - 1;   /* next TD coefficient to produce: starts at n-1 */

    double *prob_row = prob + (size_t)i * n;

    for (int tile = 0; tile < n_tiles; tile++) {
        int tile_size = TILE_M;
        if (tile * TILE_M + tile_size > n) tile_size = n - tile * TILE_M;

        /* Each thread produces TILE_M values in its direction */
        for (int dm = 0; dm < tile_size; dm++) {
            double bu_val = 0, td_val = 0;

            if (pw != 0 && is_bu) {
                if (bu_m == 0) {
                    bu_val = pw * qm_bu;
                } else {
                    qm_bu = (Pq[bu_m] - bi * qm_bu) * inv_d;
                    bu_val = pw * qm_bu;
                }
                bu_m++;
            }

            if (pw != 0 && !is_bu) {
                if (td_m == n - 1) {
                    td_val = pw * qm_td;
                } else {
                    qm_td = (Pq[td_m + 1] - ai * qm_td) * inv_d;
                    td_val = pw * qm_td;
                }
                td_m--;
            }

            red[threadIdx.x * 2 * TILE_M + dm] = bu_val;
            red[threadIdx.x * 2 * TILE_M + TILE_M + dm] = td_val;
        }
        /* Zero remaining slots if tile_size < TILE_M */
        for (int dm = tile_size; dm < TILE_M; dm++) {
            red[threadIdx.x * 2 * TILE_M + dm] = 0;
            red[threadIdx.x * 2 * TILE_M + TILE_M + dm] = 0;
        }
        __syncthreads();

        /* Tree reduction across Qpad threads (Qpad is always a power of 2) */
        for (int stride = Qpad / 2; stride > 0; stride >>= 1) {
            if ((int)threadIdx.x < stride) {
                int base_me    = threadIdx.x * 2 * TILE_M;
                int base_other = (threadIdx.x + stride) * 2 * TILE_M;
                for (int dm = 0; dm < 2 * TILE_M; dm++)
                    red[base_me + dm] += red[base_other + dm];
            }
            __syncthreads();
        }

        /* Thread 0 writes results to global memory */
        if (threadIdx.x == 0) {
            int bu_base = tile * TILE_M;           /* forward */
            int td_base = (n - 1) - tile * TILE_M; /* reverse */
            for (int dm = 0; dm < tile_size; dm++) {
                prob_row[bu_base + dm] += red[dm];
                int td_idx = td_base - dm;
                if (td_idx >= 0 && td_idx < n)
                    prob_row[td_idx] += red[TILE_M + dm];
            }
        }
        __syncthreads();
    }
}

/* ================================================================
   Host: orchestrate
   ================================================================ */

/* CUDA error check macro */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

void icm_cuda(int n, const double *S_host, int Q, const QP *pts_host,
              double *prob_host) {
    /* Precompute logv, wq on host */
    double *logv_host = (double *)malloc(Q * sizeof(double));
    double *wq_host   = (double *)malloc(Q * sizeof(double));
    for (int q = 0; q < Q; q++) {
        logv_host[q] = log_sigma_h(pts_host[q].x);
        wq_host[q]   = pts_host[q].w;
    }

    /* Allocate device memory */
    double *d_S, *d_logv, *d_wq, *d_P_store, *d_prob;
    size_t P_size = (size_t)Q * (n + 1) * sizeof(double);
    size_t prob_size = (size_t)n * n * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_S,       n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_logv,    Q * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_wq,      Q * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P_store, P_size));
    CUDA_CHECK(cudaMalloc(&d_prob,    prob_size));

    CUDA_CHECK(cudaMemcpy(d_S,    S_host,    n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_logv, logv_host, Q * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wq,   wq_host,   Q * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_prob, 0, prob_size));

    /* Phase 1: Build polynomials */
    /* Shared: 2*(n+1) doubles for double-buffered polynomial */
    size_t build_smem = (size_t)2 * (n + 1) * sizeof(double);
    if (build_smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel_build,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            build_smem));
    }
    kernel_build<<<Q, BUILD_THREADS, build_smem>>>(
        n, Q, d_S, d_logv, d_P_store);
    CUDA_CHECK(cudaGetLastError());

    /* Phase 2: Divide + accumulate */
    int Qpad = next_pow2(Q);
    size_t divide_smem = (size_t)Qpad * 2 * TILE_M * sizeof(double);
    if (divide_smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel_divide,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            divide_smem));
    }
    kernel_divide<<<n, Qpad, divide_smem>>>(
        n, Q, Qpad, d_S, d_P_store, d_logv, d_wq, d_prob);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back */
    CUDA_CHECK(cudaMemcpy(prob_host, d_prob, prob_size, cudaMemcpyDeviceToHost));

    /* Cleanup */
    cudaFree(d_S); cudaFree(d_logv); cudaFree(d_wq);
    cudaFree(d_P_store); cudaFree(d_prob);
    free(logv_host); free(wq_host);
}

/* ================================================================
   Benchmark
   ================================================================ */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int n    = (argc > 1) ? atoi(argv[1]) : 2048;
    int Q    = (argc > 2) ? atoi(argv[2]) : 256;
    int reps = (argc > 3) ? atoi(argv[3]) : 5;
    double ratio = 1e9;

    /* Print GPU info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "GPU: %s (SM %d.%d, %d SMs, %.0f MHz, %.1f GB)\n",
            prop.name, prop.major, prop.minor,
            prop.multiProcessorCount, prop.clockRate / 1000.0,
            prop.totalGlobalMem / 1e9);
    fprintf(stderr, "L2 cache: %.0f KB, max shared/block: %.0f KB\n",
            prop.l2CacheSize / 1024.0,
            prop.sharedMemPerBlockOptin / 1024.0);
    fprintf(stderr, "n=%d, Q=%d, ratio=%.0e, reps=%d\n\n", n, Q, ratio, reps);

    const char *dnames[] = {"adversarial", "reverse_adv", "bimodal", "geometric", "uniform"};

    double *S    = (double *)malloc(n * sizeof(double));
    double *eV1  = (double *)malloc(n * sizeof(double));
    double *prob = (double *)malloc((size_t)n * n * sizeof(double));
    QP     *pts  = (QP *)malloc(Q * sizeof(QP));

    fprintf(stderr, "%14s %10s %10s %12s\n", "distribution", "gpu (ms)", "gpu+xfer", "error");
    fprintf(stderr, "------------------------------------------------------\n");

    for (int di = 0; di < 5; di++) {
        switch (di) {
            case 0: for (int i=0;i<n;i++) S[i]=1; S[0]=ratio; break;
            case 1: for (int i=0;i<n;i++) S[i]=ratio; S[0]=1; break;
            case 2: for (int i=0;i<n;i++) S[i]=(i<n/2)?1:ratio; break;
            case 3: for (int i=0;i<n;i++) S[i]=pow(ratio,(double)i/(n-1)); break;
            case 4: { srand(42); double mn=1e30;
                for (int i=0;i<n;i++){S[i]=1+(ratio-1)*((double)rand()/RAND_MAX);if(S[i]<mn)mn=S[i];}
                for (int i=0;i<n;i++) S[i]/=mn; break; }
        }
        exact_V1_h(n, S, eV1);
        double Smax = 0;
        for (int i = 0; i < n; i++) if (S[i] > Smax) Smax = S[i];
        make_nodes_h(Q, Smax, pts);

        /* Warmup */
        icm_cuda(n, S, Q, pts, prob);

        /* Timed: kernel only (use events) */
        /* For simplicity, time the full icm_cuda call (includes H2D/D2H) */
        double best_total = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = now_sec();
            icm_cuda(n, S, Q, pts, prob);
            double t = (now_sec() - t0) * 1000;
            if (t < best_total) best_total = t;
        }

        /* Also time just the kernels using CUDA events */
        /* Precompute logv/wq and allocate outside timing loop */
        double *logv_h = (double *)malloc(Q * sizeof(double));
        double *wq_h   = (double *)malloc(Q * sizeof(double));
        for (int q = 0; q < Q; q++) {
            logv_h[q] = log_sigma_h(pts[q].x);
            wq_h[q]   = pts[q].w;
        }
        double *d_S2, *d_logv2, *d_wq2, *d_P2, *d_prob2;
        CUDA_CHECK(cudaMalloc(&d_S2,    n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_logv2, Q * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_wq2,   Q * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_P2,    (size_t)Q * (n+1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_prob2, (size_t)n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_S2,    S,      n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_logv2, logv_h, Q * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_wq2,   wq_h,   Q * sizeof(double), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        float best_kernel = 1e30f;
        for (int r = 0; r < reps; r++) {
            CUDA_CHECK(cudaMemset(d_prob2, 0, (size_t)n * n * sizeof(double)));

            size_t build_smem = (size_t)(3 * n + 2) * sizeof(double);
            int Qpad2 = next_pow2(Q);
            size_t div_smem = (size_t)Qpad2 * 2 * TILE_M * sizeof(double);

            CUDA_CHECK(cudaEventRecord(start));

            kernel_build<<<Q, BUILD_THREADS, build_smem>>>(
                n, Q, d_S2, d_logv2, d_P2);

            if (div_smem > 48 * 1024)
                cudaFuncSetAttribute(kernel_divide,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, div_smem);
            kernel_divide<<<n, Qpad2, div_smem>>>(
                n, Q, Qpad2, d_S2, d_P2, d_logv2, d_wq2, d_prob2);

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (ms < best_kernel) best_kernel = ms;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        double err = max_relV1_h(n, prob, eV1);

        fprintf(stderr, "%14s %9.2fms %9.2fms %12.2e\n",
                dnames[di], best_kernel, best_total, err);

        cudaFree(d_S2); cudaFree(d_logv2); cudaFree(d_wq2);
        cudaFree(d_P2); cudaFree(d_prob2);
        free(logv_h); free(wq_h);
    }

    /* Scaling sweep */
    fprintf(stderr, "\n--- Scaling (adversarial, Q=%d) ---\n", Q);
    int sweep_ns[] = {64, 128, 256, 512, 1024, 2048};
    int nsweep = 6;
    fprintf(stderr, "%6s %10s %10s %12s\n", "n", "kernel", "total", "error");

    for (int si = 0; si < nsweep; si++) {
        int sn = sweep_ns[si];
        if (sn > n) break;
        double *sS = (double *)malloc(sn * sizeof(double));
        double *seV1 = (double *)malloc(sn * sizeof(double));
        double *sprob = (double *)malloc((size_t)sn * sn * sizeof(double));
        QP *spts = (QP *)malloc(Q * sizeof(QP));
        for (int i = 0; i < sn; i++) sS[i] = 1; sS[0] = ratio;
        exact_V1_h(sn, sS, seV1);
        make_nodes_h(Q, ratio, spts);

        /* Warmup */
        icm_cuda(sn, sS, Q, spts, sprob);

        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = now_sec();
            icm_cuda(sn, sS, Q, spts, sprob);
            double t = (now_sec() - t0) * 1000;
            if (t < best) best = t;
        }
        double err = max_relV1_h(sn, sprob, seV1);
        fprintf(stderr, "%6d %9.2fms %9.2fms %12.2e\n", sn, 0.0, best, err);

        free(sS); free(seV1); free(sprob); free(spts);
    }

    free(S); free(eV1); free(prob); free(pts);
    fprintf(stderr, "\nDone.\n");
    return 0;
}
