#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "icm_gpu.h"

static bool cuda_ok(cudaError_t st, const char *what) {
    if (st == cudaSuccess) return true;
    printf("CUDA error at %s: %s\n", what, cudaGetErrorString(st));
    return false;
}

static bool cufft_ok(cufftResult st, const char *what) {
    if (st == CUFFT_SUCCESS) return true;
    printf("cuFFT error at %s: %d\n", what, (int)st);
    return false;
}

static void build_smooth_table(int max_n, std::vector<int> &smooth) {
    smooth.clear();
    for (int a = 1; a <= max_n; a *= 2) {
        for (int b = a; b <= max_n; b *= 3) {
            for (int c = b; c <= max_n; c *= 5) {
                for (int d = c; d <= max_n; d *= 7) {
                    smooth.push_back(d);
                    if (d > max_n / 7) break;
                }
                if (c > max_n / 5) break;
            }
            if (b > max_n / 3) break;
        }
        if (a > max_n / 2) break;
    }
    std::sort(smooth.begin(), smooth.end());
    smooth.erase(std::unique(smooth.begin(), smooth.end()), smooth.end());
}

static bool parse_mults_csv(const char *csv, std::vector<int> &mults) {
    mults.clear();
    if (!csv || !csv[0]) return false;
    const char *p = csv;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',') ++p;
        if (!*p) break;
        char *end = nullptr;
        long v = strtol(p, &end, 10);
        if (end == p) {
            while (*p && *p != ',') ++p;
            continue;
        }
        if (v > 0 && v <= std::numeric_limits<int>::max()) mults.push_back((int)v);
        p = end;
    }
    std::sort(mults.begin(), mults.end());
    mults.erase(std::unique(mults.begin(), mults.end()), mults.end());
    return !mults.empty();
}

static void build_pow2_mult_table(int max_n, const std::vector<int> &mults, std::vector<int> &sizes) {
    sizes.clear();
    if (max_n < 1 || mults.empty()) return;
    for (long long p2 = 1; p2 <= (long long)max_n; p2 <<= 1) {
        for (int m : mults) {
            long long v = (long long)m * p2;
            if (v >= 1 && v <= (long long)max_n) sizes.push_back((int)v);
        }
        if (p2 > (long long)max_n / 2) break;
    }
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
}

static double median(std::vector<double> &x) {
    if (x.empty()) return NAN;
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

static int env_int_clamped(const char *name, int fallback, int lo, int hi) {
    const char *v = getenv(name);
    if (!v || !v[0]) return fallback;
    int x = atoi(v);
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    return x;
}

static double env_double_positive(const char *name, double fallback) {
    const char *v = getenv(name);
    if (!v || !v[0]) return fallback;
    double x = atof(v);
    if (!std::isfinite(x) || x <= 0.0) return fallback;
    return x;
}

static int pick_warmup(int quick) {
    int base = quick ? 2 : 4;
    return env_int_clamped("ICM_GPU_CALIB_WARMUP", base, 0, 32);
}

static int pick_batch_for_fft_n(int n) {
    int batch = 1;
    if (n <= 2048) batch = 1024;
    else if (n <= 8192) batch = 512;
    else if (n <= 32768) batch = 128;
    else if (n <= 65536) batch = 64;
    else if (n <= 131072) batch = 16;
    else if (n <= 262144) batch = 8;
    else if (n <= 524288) batch = 4;
    else if (n <= 1048576) batch = 2;
    else batch = 1;

    int min_batch = env_int_clamped("ICM_GPU_CALIB_MIN_BATCH", 1, 1, 65536);
    int max_batch = env_int_clamped("ICM_GPU_CALIB_MAX_BATCH", batch, min_batch, 65536);
    if (batch < min_batch) batch = min_batch;
    if (batch > max_batch) batch = max_batch;

    /* Memory guard for very large sizes. The indep-pair probe is the heaviest:
     * roughly O(80 * batch * n) bytes plus cuFFT work buffers.
     * Keep a safety margin by reserving only a fraction of currently free VRAM.
     */
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes > 0) {
        double frac = env_double_positive("ICM_GPU_CALIB_MEM_FRACTION", 0.40);
        if (frac > 0.95) frac = 0.95;
        double budget = (double)free_bytes * frac;
        double bytes_per_batch_n = env_double_positive("ICM_GPU_CALIB_BYTES_PER_BATCH_N", 96.0);
        double need_per_batch = bytes_per_batch_n * (double)n;
        int safe_batch = (int)(budget / std::max(1.0, need_per_batch));
        if (safe_batch < 1) safe_batch = 1;
        if (batch > safe_batch) batch = safe_batch;
    }
    return batch;
}

static int pick_reps_for_work(long long work, int quick) {
    double target_default = quick ? 5.0e7 : 2.0e8;
    int min_reps_default = quick ? 2 : 3;
    int max_reps_default = quick ? 16 : 36;
    double target = env_double_positive("ICM_GPU_CALIB_TARGET_WORK", target_default);
    int min_reps = env_int_clamped("ICM_GPU_CALIB_MIN_REPS", min_reps_default, 1, 256);
    int max_reps = env_int_clamped("ICM_GPU_CALIB_MAX_REPS", max_reps_default, min_reps, 512);
    double scale = env_double_positive("ICM_GPU_CALIB_REPS_SCALE", 1.0);

    int reps = (int)(target / (double)(work + 1));
    reps = (int)llround((double)reps * scale);
    if (reps < min_reps) reps = min_reps;
    if (reps > max_reps) reps = max_reps;
    return reps;
}

static double time_ns(const std::function<void()> &fn, int warmup, int reps) {
    cudaEvent_t e0 = nullptr;
    cudaEvent_t e1 = nullptr;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();

    std::vector<double> samples;
    samples.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(e0);
        fn();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        samples.push_back((double)ms * 1e6);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return median(samples);
}

__global__ static void k_pointwise_mul(cufftDoubleComplex *a, const cufftDoubleComplex *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cufftDoubleComplex x = a[i];
    cufftDoubleComplex y = b[i];
    cufftDoubleComplex o;
    o.x = x.x * y.x - x.y * y.y;
    o.y = x.x * y.y + x.y * y.x;
    a[i] = o;
}

__global__ static void k_paired_corr_freq(const cufftDoubleComplex *g_hat,
                                          const cufftDoubleComplex *cached_child_spec,
                                          int cn, int nparents,
                                          cufftDoubleComplex *child_out_spec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nparents * cn;
    if (idx >= total) return;
    int p = idx / cn;
    int f = idx % cn;
    cufftDoubleComplex g = g_hat[idx];
    cufftDoubleComplex specL = cached_child_spec[(2 * p) * cn + f];
    cufftDoubleComplex specR = cached_child_spec[(2 * p + 1) * cn + f];
    cufftDoubleComplex outL;
    outL.x = g.x * specR.x + g.y * specR.y;
    outL.y = g.y * specR.x - g.x * specR.y;
    cufftDoubleComplex outR;
    outR.x = g.x * specL.x + g.y * specL.y;
    outR.y = g.y * specL.x - g.x * specL.y;
    child_out_spec[(2 * p) * cn + f] = outL;
    child_out_spec[(2 * p + 1) * cn + f] = outR;
}

__global__ static void k_triplet_corr_freq(const cufftDoubleComplex *g_hat,
                                           const cufftDoubleComplex *pl_hat,
                                           const cufftDoubleComplex *pr_hat,
                                           int cn, int nparents,
                                           cufftDoubleComplex *child_out_spec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nparents * cn;
    if (idx >= total) return;
    cufftDoubleComplex g = g_hat[idx];
    cufftDoubleComplex l = pl_hat[idx];
    cufftDoubleComplex r = pr_hat[idx];
    cufftDoubleComplex outL;
    outL.x = g.x * r.x + g.y * r.y;
    outL.y = g.y * r.x - g.x * r.y;
    cufftDoubleComplex outR;
    outR.x = g.x * l.x + g.y * l.y;
    outR.y = g.y * l.x - g.x * l.y;
    child_out_spec[2 * idx] = outL;
    child_out_spec[2 * idx + 1] = outR;
}

__global__ static void k_fma_stream(const double *a, const double *b, double *c, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = c[i];
    for (int t = 0; t < iters; ++t) x = a[i] * b[i] + x;
    c[i] = x;
}

__global__ static void k_block_build_prod(const double *a_sorted, int n, int B,
                                          int nblocks, int N_tree,
                                          int leaf_psz, double *leaves, double *block_prods) {
    int b = blockIdx.x;
    if (b >= N_tree) return;
    int t = threadIdx.x;
    double *leaf = leaves + (size_t)b * (size_t)leaf_psz;
    double *P = block_prods + (size_t)b * (size_t)(B + 1);
    extern __shared__ double sh[];
    double *curr = sh;
    double *next = sh + (B + 1);

    if (b >= nblocks) {
        for (int m = t; m < B + 1; m += blockDim.x) P[m] = 0.0;
        for (int m = t; m < leaf_psz; m += blockDim.x) leaf[m] = 0.0;
        if (t == 0) {
            P[0] = 1.0;
            leaf[0] = 1.0;
        }
        return;
    }

    int start = b * B;
    int end = start + B;
    if (end > n) end = n;
    int bsize = end - start;

    for (int m = t; m < B + 1; m += blockDim.x) {
        curr[m] = 0.0;
        next[m] = 0.0;
    }
    __syncthreads();
    if (t == 0) curr[0] = 1.0;
    __syncthreads();

    for (int r = 0; r < bsize; ++r) {
        double aj = a_sorted[start + r];
        double bj = 1.0 - aj;
        int active_m = r + 1;
        for (int m = t; m < B + 1; m += blockDim.x) {
            double v = 0.0;
            if (m == 0) v = aj * curr[0];
            else if (m <= active_m) v = aj * curr[m] + bj * curr[m - 1];
            next[m] = v;
        }
        __syncthreads();
        double *tmp = curr;
        curr = next;
        next = tmp;
        __syncthreads();
    }

    int cp = (B + 1 < leaf_psz) ? (B + 1) : leaf_psz;
    for (int m = t; m < B + 1; m += blockDim.x) P[m] = curr[m];
    for (int m = t; m < leaf_psz; m += blockDim.x) leaf[m] = (m < cp) ? curr[m] : 0.0;
}

__global__ static void k_leaf_extract_prod(const double *a_sorted, int n, int B, int nblocks,
                                           const double *block_prods, const double *g_leaf,
                                           int leaf_psz, int g_need, int k,
                                           double *inner_sorted) {
    int b = blockIdx.x;
    if (b >= nblocks) return;
    int start = b * B;
    int end = start + B;
    if (end > n) end = n;
    int bsize = end - start;
    const double *P_b = block_prods + (size_t)b * (size_t)(B + 1);
    const double *g_b = g_leaf + (size_t)b * (size_t)leaf_psz;
    int pk_g = g_need < bsize ? g_need : bsize;
    if (pk_g > k) pk_g = k;

    for (int t = threadIdx.x; t < bsize; t += blockDim.x) {
        int j = start + t;
        double aj = a_sorted[j];
        double bj = 1.0 - aj;
        double eq = 0.0;
        if (aj > 0.5) {
            double ia = 1.0 / aj;
            double c = -bj / aj;
            double q = P_b[0] * ia;
            eq = g_b[0] * q;
            for (int m = 1; m < pk_g; ++m) {
                q = c * q + P_b[m] * ia;
                eq += g_b[m] * q;
            }
        } else if (aj > 1e-15) {
            double ib = 1.0 / bj;
            double c = -aj / bj;
            double q = P_b[bsize] * ib;
            if (bsize - 1 < pk_g) eq += g_b[bsize - 1] * q;
            for (int m = bsize - 2; m >= 0; --m) {
                q = c * q + P_b[m + 1] * ib;
                if (m < pk_g) eq += g_b[m] * q;
            }
        } else {
            for (int m = 0; m < pk_g; ++m) eq += g_b[m] * P_b[m + 1];
        }
        inner_sorted[j] = eq;
    }
}

static bool make_plan(cufftHandle *plan, int n, int batch, bool r2c) {
    if (!cufft_ok(cufftCreate(plan), "cufftCreate")) return false;
    int rank = 1;
    int dims[1] = {n};
    int inembed[1] = {n};
    int onembed[1] = {n / 2 + 1};
    size_t ws = 0;
    if (r2c) {
        return cufft_ok(cufftMakePlanMany(*plan, rank, dims, inembed, 1, n,
                                          onembed, 1, n / 2 + 1, CUFFT_D2Z, batch, &ws),
                        "cufftMakePlanMany r2c");
    }
    return cufft_ok(cufftMakePlanMany(*plan, rank, dims, onembed, 1, n / 2 + 1,
                                      inembed, 1, n, CUFFT_Z2D, batch, &ws),
                    "cufftMakePlanMany c2r");
}

static double measure_cufft_build_ns(int fft_n, int batch, int quick) {
    int cn = fft_n / 2 + 1;
    size_t bytes_r = (size_t)batch * (size_t)fft_n * sizeof(double);
    size_t bytes_c = (size_t)batch * (size_t)cn * sizeof(cufftDoubleComplex);
    double *d_r0 = nullptr;
    double *d_r1 = nullptr;
    cufftDoubleComplex *d_c0 = nullptr;
    cufftDoubleComplex *d_c1 = nullptr;
    cudaMalloc(&d_r0, bytes_r);
    cudaMalloc(&d_r1, bytes_r);
    cudaMalloc(&d_c0, bytes_c);
    cudaMalloc(&d_c1, bytes_c);
    cudaMemset(d_r0, 1, bytes_r);
    cudaMemset(d_r1, 2, bytes_r);
    cufftHandle fwd = 0;
    cufftHandle inv = 0;
    make_plan(&fwd, fft_n, batch, true);
    make_plan(&inv, fft_n, batch, false);

    int threads = 256;
    int blocks = (batch * cn + threads - 1) / threads;
    int reps = pick_reps_for_work((long long)fft_n * batch, quick);
    int warmup = pick_warmup(quick);
    auto fn = [&]() {
        cufftExecD2Z(fwd, d_r0, d_c0);
        cufftExecD2Z(fwd, d_r1, d_c1);
        k_pointwise_mul<<<blocks, threads>>>(d_c0, d_c1, batch * cn);
        cufftExecZ2D(inv, d_c0, d_r0);
    };
    double ns = time_ns(fn, warmup, reps) / (double)batch;
    cufftDestroy(fwd);
    cufftDestroy(inv);
    cudaFree(d_r0);
    cudaFree(d_r1);
    cudaFree(d_c0);
    cudaFree(d_c1);
    return ns;
}

static double measure_cufft_corr_ns(int fft_n, int batch, int quick) {
    int cn = fft_n / 2 + 1;
    size_t bytes_g = (size_t)batch * (size_t)fft_n * sizeof(double);
    size_t bytes_spec = (size_t)batch * (size_t)cn * sizeof(cufftDoubleComplex);
    size_t bytes_child_spec = (size_t)(2 * batch) * (size_t)cn * sizeof(cufftDoubleComplex);
    size_t bytes_out = (size_t)(2 * batch) * (size_t)fft_n * sizeof(double);
    double *d_g = nullptr;
    cufftDoubleComplex *d_g_hat = nullptr;
    cufftDoubleComplex *d_child = nullptr;
    cufftDoubleComplex *d_out_spec = nullptr;
    double *d_out = nullptr;
    cudaMalloc(&d_g, bytes_g);
    cudaMalloc(&d_g_hat, bytes_spec);
    cudaMalloc(&d_child, bytes_child_spec);
    cudaMalloc(&d_out_spec, bytes_child_spec);
    cudaMalloc(&d_out, bytes_out);
    cudaMemset(d_g, 1, bytes_g);
    cudaMemset(d_child, 1, bytes_child_spec);
    cufftHandle fwd = 0;
    cufftHandle inv = 0;
    make_plan(&fwd, fft_n, batch, true);
    make_plan(&inv, fft_n, 2 * batch, false);
    int threads = 256;
    int blocks = (batch * cn + threads - 1) / threads;
    int reps = pick_reps_for_work((long long)fft_n * batch, quick);
    int warmup = pick_warmup(quick);
    auto fn = [&]() {
        cufftExecD2Z(fwd, d_g, d_g_hat);
        k_paired_corr_freq<<<blocks, threads>>>(d_g_hat, d_child, cn, batch, d_out_spec);
        cufftExecZ2D(inv, d_out_spec, d_out);
    };
    double ns = time_ns(fn, warmup, reps) / (double)batch;
    cufftDestroy(fwd);
    cufftDestroy(inv);
    cudaFree(d_g);
    cudaFree(d_g_hat);
    cudaFree(d_child);
    cudaFree(d_out_spec);
    cudaFree(d_out);
    return ns;
}

static double measure_cufft_indep_pair_ns(int fft_n, int batch, int quick) {
    int cn = fft_n / 2 + 1;
    size_t bytes_r = (size_t)batch * (size_t)fft_n * sizeof(double);
    size_t bytes_c = (size_t)batch * (size_t)cn * sizeof(cufftDoubleComplex);
    size_t bytes_out_spec = (size_t)(2 * batch) * (size_t)cn * sizeof(cufftDoubleComplex);
    size_t bytes_out = (size_t)(2 * batch) * (size_t)fft_n * sizeof(double);
    double *d_g = nullptr;
    double *d_l = nullptr;
    double *d_r = nullptr;
    cufftDoubleComplex *d_g_hat = nullptr;
    cufftDoubleComplex *d_l_hat = nullptr;
    cufftDoubleComplex *d_r_hat = nullptr;
    cufftDoubleComplex *d_out_spec = nullptr;
    double *d_out = nullptr;
    cudaMalloc(&d_g, bytes_r);
    cudaMalloc(&d_l, bytes_r);
    cudaMalloc(&d_r, bytes_r);
    cudaMalloc(&d_g_hat, bytes_c);
    cudaMalloc(&d_l_hat, bytes_c);
    cudaMalloc(&d_r_hat, bytes_c);
    cudaMalloc(&d_out_spec, bytes_out_spec);
    cudaMalloc(&d_out, bytes_out);
    cudaMemset(d_g, 1, bytes_r);
    cudaMemset(d_l, 1, bytes_r);
    cudaMemset(d_r, 2, bytes_r);
    cufftHandle fwd = 0;
    cufftHandle inv = 0;
    make_plan(&fwd, fft_n, batch, true);
    make_plan(&inv, fft_n, 2 * batch, false);
    int threads = 256;
    int blocks = (batch * cn + threads - 1) / threads;
    int reps = pick_reps_for_work((long long)fft_n * batch, quick);
    int warmup = pick_warmup(quick);
    auto fn = [&]() {
        cufftExecD2Z(fwd, d_g, d_g_hat);
        cufftExecD2Z(fwd, d_l, d_l_hat);
        cufftExecD2Z(fwd, d_r, d_r_hat);
        k_triplet_corr_freq<<<blocks, threads>>>(d_g_hat, d_l_hat, d_r_hat, cn, batch, d_out_spec);
        cufftExecZ2D(inv, d_out_spec, d_out);
    };
    double ns = time_ns(fn, warmup, reps) / (double)batch;
    cufftDestroy(fwd);
    cufftDestroy(inv);
    cudaFree(d_g);
    cudaFree(d_l);
    cudaFree(d_r);
    cudaFree(d_g_hat);
    cudaFree(d_l_hat);
    cudaFree(d_r_hat);
    cudaFree(d_out_spec);
    cudaFree(d_out);
    return ns;
}

static double measure_school_fma_ns(int quick) {
    int n = 1 << 24;
    int iters = quick ? 8 : 32;
    double *a = nullptr;
    double *b = nullptr;
    double *c = nullptr;
    cudaMalloc(&a, (size_t)n * sizeof(double));
    cudaMalloc(&b, (size_t)n * sizeof(double));
    cudaMalloc(&c, (size_t)n * sizeof(double));
    cudaMemset(a, 1, (size_t)n * sizeof(double));
    cudaMemset(b, 2, (size_t)n * sizeof(double));
    cudaMemset(c, 0, (size_t)n * sizeof(double));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int reps = pick_reps_for_work((long long)n * (long long)iters, quick);
    int warmup = pick_warmup(quick);
    auto fn = [&]() { k_fma_stream<<<blocks, threads>>>(a, b, c, n, iters); };
    double ns = time_ns(fn, warmup, reps);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    double fmas = (double)n * (double)iters;
    return ns / fmas;
}

static double measure_block_build_ns_per_fma(int quick) {
    const int B = 256;
    const int blocks = quick ? 1024 : 4096;
    const int n = blocks * B;
    double *a = nullptr;
    double *leaves = nullptr;
    double *prods = nullptr;
    cudaMalloc(&a, (size_t)n * sizeof(double));
    cudaMalloc(&leaves, (size_t)blocks * (size_t)(B + 1) * sizeof(double));
    cudaMalloc(&prods, (size_t)blocks * (size_t)(B + 1) * sizeof(double));
    cudaMemset(a, 1, (size_t)n * sizeof(double));
    long long work = (long long)blocks * (long long)B * (long long)B;
    int reps = pick_reps_for_work(work, quick);
    int warmup = pick_warmup(quick);
    size_t shmem = (size_t)(2 * (B + 1)) * sizeof(double);
    auto fn = [&]() {
        k_block_build_prod<<<blocks, 256, shmem>>>(a, n, B, blocks, blocks, B + 1, leaves, prods);
    };
    double ns = time_ns(fn, warmup, reps);
    cudaFree(a);
    cudaFree(leaves);
    cudaFree(prods);
    double fmas = (double)blocks * ((double)B * (double)(B + 1) / 2.0);
    return ns / fmas;
}

static double measure_leaf_extract_ns_per_fma(int quick) {
    const int B = 256;
    const int blocks = quick ? 1024 : 4096;
    const int n = blocks * B;
    double *a = nullptr;
    double *P = nullptr;
    double *g = nullptr;
    double *out = nullptr;
    cudaMalloc(&a, (size_t)n * sizeof(double));
    cudaMalloc(&P, (size_t)blocks * (size_t)(B + 1) * sizeof(double));
    cudaMalloc(&g, (size_t)blocks * (size_t)(B + 1) * sizeof(double));
    cudaMalloc(&out, (size_t)n * sizeof(double));
    cudaMemset(a, 1, (size_t)n * sizeof(double));
    cudaMemset(P, 1, (size_t)blocks * (size_t)(B + 1) * sizeof(double));
    cudaMemset(g, 1, (size_t)blocks * (size_t)(B + 1) * sizeof(double));
    long long work = (long long)blocks * (long long)B * (long long)B;
    int reps = pick_reps_for_work(work, quick);
    int warmup = pick_warmup(quick);
    auto fn = [&]() {
        k_leaf_extract_prod<<<blocks, 256>>>(a, n, B, blocks, P, g, B + 1, B, B, out);
    };
    double ns = time_ns(fn, warmup, reps);
    cudaFree(a);
    cudaFree(P);
    cudaFree(g);
    cudaFree(out);
    double fmas = (double)blocks * (double)B * (double)B;
    return ns / fmas;
}

static double estimate_fft_overhead_ns(const std::vector<int> &sizes, const std::vector<double> &times) {
    if (sizes.size() < 3 || times.size() < 3) return 0.0;
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    int m = 0;
    for (size_t i = 0; i < sizes.size() && m < 8; ++i) {
        if (sizes[i] > 512) break;
        double x = (double)sizes[i];
        double y = times[i];
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        ++m;
    }
    if (m < 2) return 0.0;
    double denom = (double)m * sxx - sx * sx;
    if (fabs(denom) < 1e-9) return 0.0;
    double slope = ((double)m * sxy - sx * sy) / denom;
    double intercept = (sy - slope * sx) / (double)m;
    if (!std::isfinite(intercept) || intercept < 0.0) intercept = 0.0;
    return intercept;
}

static int write_header(const char *path,
                        const std::vector<int> &sizes,
                        const std::vector<double> &cufft_build_ns,
                        const std::vector<double> &dx_build_ns,
                        const std::vector<double> &dx_corr_ns,
                        double school_ns,
                        double fft_overhead_ns,
                        double hbm_gbps,
                        int fused_max_conv_len,
                        double paired_ratio,
                        double indep_ratio,
                        double block_build_ns_per_fma,
                        double leaf_extract_ns_per_fma,
                        unsigned long long vram_bytes,
                        int sm_count) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "/* Auto-generated by tools/calibrate_gpu.cu */\n");
    fprintf(f, "#ifndef ICM_GPU_FFT_CONFIG_H\n#define ICM_GPU_FFT_CONFIG_H\n\n");
    fprintf(f, "#define GPU_N_CALIBRATED_SIZES %d\n", (int)sizes.size());
    fprintf(f, "static const int gpu_calib_sizes[GPU_N_CALIBRATED_SIZES] = {\n  ");
    for (size_t i = 0; i < sizes.size(); ++i) {
        fprintf(f, "%d%s", sizes[i], (i + 1 == sizes.size()) ? "" : ",");
        if ((i + 1) % 16 == 0 && i + 1 != sizes.size()) fprintf(f, "\n  ");
    }
    fprintf(f, "\n};\n\n");

    auto write_arr = [&](const char *name, const std::vector<double> &v) {
        fprintf(f, "static const double %s[GPU_N_CALIBRATED_SIZES] = {\n  ", name);
        for (size_t i = 0; i < v.size(); ++i) {
            fprintf(f, "%.1f%s", v[i], (i + 1 == v.size()) ? "" : ",");
            if ((i + 1) % 12 == 0 && i + 1 != v.size()) fprintf(f, "\n  ");
        }
        fprintf(f, "\n};\n\n");
    };

    write_arr("gpu_calib_cufft_ns", cufft_build_ns);
    write_arr("gpu_calib_cufftdx_build_ns", dx_build_ns);
    write_arr("gpu_calib_cufftdx_corr_ns", dx_corr_ns);

    fprintf(f, "#define GPU_SCHOOL_FMA_NS %.8f\n", school_ns);
    fprintf(f, "#define GPU_FFT_OVERHEAD_NS %.8f\n", fft_overhead_ns);
    fprintf(f, "#define GPU_HBM_BANDWIDTH %.8f\n", hbm_gbps);
    fprintf(f, "#define GPU_FUSED_MAX_CONV_LEN %d\n", fused_max_conv_len);
    fprintf(f, "#define GPU_PAIRED_CACHED_CORR_RATIO %.8f\n", paired_ratio);
    fprintf(f, "#define GPU_INDEP_PAIR_RATIO %.8f\n", indep_ratio);
    fprintf(f, "#define GPU_BLOCK_BUILD_NS_PER_FMA %.8f\n", block_build_ns_per_fma);
    fprintf(f, "#define GPU_LEAF_EXTRACT_NS_PER_FMA %.8f\n", leaf_extract_ns_per_fma);
    fprintf(f, "#define GPU_VRAM_BYTES (%lluULL)\n", vram_bytes);
    fprintf(f, "#define GPU_SM_COUNT %d\n", sm_count);
    fprintf(f, "\n#endif\n");
    fclose(f);
    return 1;
}

int main(int argc, char **argv) {
    const char *out_path = "devices/b200/gpu_fft_config.h";
    int max_size = 131072;
    int quick = 0;
    if (argc > 1) out_path = argv[1];
    if (argc > 2) max_size = atoi(argv[2]);
    if (argc > 3 && strcmp(argv[3], "--quick") == 0) quick = 1;

    if (!icm_gpu_init(0)) {
        printf("icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    std::vector<int> smooth;
    std::vector<int> mults;
    const char *mults_env = getenv("ICM_GPU_CALIB_MULTS");
    bool use_pow2_family = parse_mults_csv(mults_env, mults);
    if (use_pow2_family) build_pow2_mult_table(max_size, mults, smooth);
    else build_smooth_table(max_size, smooth);

    int measure_ratios = env_int_clamped("ICM_GPU_CALIB_MEASURE_RATIOS", 1, 0, 1);
    int ratio_max_n = env_int_clamped("ICM_GPU_CALIB_RATIO_MAX_N", max_size, 1, max_size);

    if (use_pow2_family) {
        printf("Calibrating %zu FFT sizes up to %d (quick=%d, mults=%s)\n",
               smooth.size(), max_size, quick, mults_env ? mults_env : "");
    } else {
        printf("Calibrating %zu FFT sizes up to %d (quick=%d, smooth=2^a*3^b*5^c*7^d)\n",
               smooth.size(), max_size, quick);
    }
    printf("Ratio probes: %s (max_n=%d)\n", measure_ratios ? "enabled" : "disabled", ratio_max_n);

    std::vector<double> cufft_ns(smooth.size(), 0.0);
    std::vector<double> dx_build_ns(smooth.size(), 0.0);
    std::vector<double> dx_corr_ns(smooth.size(), 0.0);
    int fused_max = 0;
    std::vector<double> paired_ratios;
    std::vector<double> indep_ratios;

    for (size_t i = 0; i < smooth.size(); ++i) {
        int n = smooth[i];
        int batch = pick_batch_for_fft_n(n);
        double t_build = measure_cufft_build_ns(n, batch, quick);
        bool do_ratio = (measure_ratios != 0) && (n <= ratio_max_n);
        double t_corr = NAN;
        double t_indep = NAN;
        if (do_ratio) {
            t_corr = measure_cufft_corr_ns(n, batch, quick);
            t_indep = measure_cufft_indep_pair_ns(n, batch, quick);
        }
        cufft_ns[i] = t_build;
        double t_dx_build = 0.0;
        double t_dx_corr = 0.0;
        if (icm_gpu_measure_fused_pair_ns(n, batch, quick, &t_dx_build, &t_dx_corr)) {
            dx_build_ns[i] = t_dx_build;
            dx_corr_ns[i] = t_dx_corr;
            if (n > fused_max) fused_max = n;
        }
        if (std::isfinite(t_corr) && std::isfinite(t_build) && t_build > 0.0) paired_ratios.push_back(t_corr / t_build);
        if (std::isfinite(t_indep) && std::isfinite(t_build) && t_build > 0.0) indep_ratios.push_back(t_indep / t_build);
        if (do_ratio) {
            printf("[%4zu/%4zu] n=%-9d batch=%-4d build=%.1fns corr=%.1fns indep=%.1fns dx_build=%.1fns dx_corr=%.1fns\n",
                   i + 1, smooth.size(), n, batch, t_build, t_corr, t_indep, t_dx_build, t_dx_corr);
        } else {
            printf("[%4zu/%4zu] n=%-9d batch=%-4d build=%.1fns corr=SKIP indep=SKIP dx_build=%.1fns dx_corr=%.1fns\n",
                   i + 1, smooth.size(), n, batch, t_build, t_dx_build, t_dx_corr);
        }
        fflush(stdout);
    }

    double hbm = 0.0;
    if (!icm_gpu_measure_hbm_bandwidth_gbps(&hbm)) hbm = 0.0;
    double school_ns = measure_school_fma_ns(quick);
    double block_ns = measure_block_build_ns_per_fma(quick);
    double leaf_ns = measure_leaf_extract_ns_per_fma(quick);
    double overhead_ns = estimate_fft_overhead_ns(smooth, cufft_ns);
    double paired_ratio = paired_ratios.empty() ? 1.0 : median(paired_ratios);
    double indep_ratio = indep_ratios.empty() ? 1.33 : median(indep_ratios);

    printf("GPU_SCHOOL_FMA_NS=%.8f\n", school_ns);
    printf("GPU_BLOCK_BUILD_NS_PER_FMA=%.8f\n", block_ns);
    printf("GPU_LEAF_EXTRACT_NS_PER_FMA=%.8f\n", leaf_ns);
    printf("GPU_FFT_OVERHEAD_NS=%.8f\n", overhead_ns);
    printf("GPU_PAIRED_CACHED_CORR_RATIO=%.8f\n", paired_ratio);
    printf("GPU_INDEP_PAIR_RATIO=%.8f\n", indep_ratio);
    printf("GPU_HBM_BANDWIDTH=%.4f GB/s\n", hbm);
    printf("GPU_FUSED_MAX_CONV_LEN=%d\n", fused_max);
    printf("GPU_VRAM_BYTES=%llu  GPU_SM_COUNT=%d\n",
           (unsigned long long)prop.totalGlobalMem, prop.multiProcessorCount);

    if (!write_header(out_path, smooth, cufft_ns, dx_build_ns, dx_corr_ns,
                      school_ns, overhead_ns, hbm, fused_max, paired_ratio, indep_ratio,
                      block_ns, leaf_ns,
                      (unsigned long long)prop.totalGlobalMem, prop.multiProcessorCount)) {
        printf("Failed writing %s\n", out_path);
        return 1;
    }

    printf("Wrote %s\n", out_path);
    icm_gpu_shutdown();
    return 0;
}
