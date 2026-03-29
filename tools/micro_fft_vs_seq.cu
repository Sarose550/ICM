#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <vector>

static inline double now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

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

static int next_7smooth_ge(const std::vector<int> &smooth, int target) {
    auto it = std::lower_bound(smooth.begin(), smooth.end(), target);
    if (it == smooth.end()) return smooth.back();
    return *it;
}

static double median(std::vector<double> &x) {
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
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

/* Fully sequential schoolbook convolution per instance (thread 0 only).
 * This represents the "sequential" baseline the user requested. */
__global__ static void k_seq_conv_batch(const double *a, const double *b, double *c,
                                        int len, int out_len, int batch) {
    int bid = blockIdx.x;
    if (bid >= batch || threadIdx.x != 0) return;
    const double *A = a + (size_t)bid * (size_t)len;
    const double *B = b + (size_t)bid * (size_t)len;
    double *C = c + (size_t)bid * (size_t)out_len;
    for (int m = 0; m < out_len; ++m) {
        int j_lo = m - (len - 1);
        if (j_lo < 0) j_lo = 0;
        int j_hi = m;
        if (j_hi > len - 1) j_hi = len - 1;
        double sum = 0.0;
        for (int j = j_lo; j <= j_hi; ++j) sum += A[j] * B[m - j];
        C[m] = sum;
    }
}

static int pick_batch_for_len(int len) {
    long long ops = (long long)len * (long long)len;
    if (ops <= 64LL * 64LL) return 16384;
    if (ops <= 256LL * 256LL) return 4096;
    if (ops <= 1024LL * 1024LL) return 1024;
    if (ops <= 2048LL * 2048LL) return 256;
    if (ops <= 4096LL * 4096LL) return 64;
    return 16;
}

static double measure_seq_ns_per_conv(int len, int batch) {
    int out_len = 2 * len - 1;
    size_t bytes_ab = (size_t)batch * (size_t)len * sizeof(double);
    size_t bytes_c = (size_t)batch * (size_t)out_len * sizeof(double);
    double *d_a = nullptr;
    double *d_b = nullptr;
    double *d_c = nullptr;
    if (!cuda_ok(cudaMalloc(&d_a, bytes_ab), "seq cudaMalloc a")) return NAN;
    if (!cuda_ok(cudaMalloc(&d_b, bytes_ab), "seq cudaMalloc b")) return NAN;
    if (!cuda_ok(cudaMalloc(&d_c, bytes_c), "seq cudaMalloc c")) return NAN;
    if (!cuda_ok(cudaMemset(d_a, 1, bytes_ab), "seq memset a")) return NAN;
    if (!cuda_ok(cudaMemset(d_b, 2, bytes_ab), "seq memset b")) return NAN;
    if (!cuda_ok(cudaMemset(d_c, 0, bytes_c), "seq memset c")) return NAN;

    int warmup = 3;
    for (int i = 0; i < warmup; ++i) k_seq_conv_batch<<<batch, 1>>>(d_a, d_b, d_c, len, out_len, batch);
    if (!cuda_ok(cudaDeviceSynchronize(), "seq warmup sync")) return NAN;

    int reps = 7;
    std::vector<double> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        double t0 = now_ns();
        k_seq_conv_batch<<<batch, 1>>>(d_a, d_b, d_c, len, out_len, batch);
        if (!cuda_ok(cudaDeviceSynchronize(), "seq rep sync")) return NAN;
        samples.push_back(now_ns() - t0);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return median(samples) / (double)batch;
}

static double measure_fft_ns_per_conv(int len, int fft_n, int batch) {
    int cn = fft_n / 2 + 1;
    size_t bytes_r = (size_t)batch * (size_t)fft_n * sizeof(double);
    size_t bytes_c = (size_t)batch * (size_t)cn * sizeof(cufftDoubleComplex);
    double *d_r0 = nullptr;
    double *d_r1 = nullptr;
    cufftDoubleComplex *d_c0 = nullptr;
    cufftDoubleComplex *d_c1 = nullptr;
    if (!cuda_ok(cudaMalloc(&d_r0, bytes_r), "fft cudaMalloc r0")) return NAN;
    if (!cuda_ok(cudaMalloc(&d_r1, bytes_r), "fft cudaMalloc r1")) return NAN;
    if (!cuda_ok(cudaMalloc(&d_c0, bytes_c), "fft cudaMalloc c0")) return NAN;
    if (!cuda_ok(cudaMalloc(&d_c1, bytes_c), "fft cudaMalloc c1")) return NAN;
    if (!cuda_ok(cudaMemset(d_r0, 1, bytes_r), "fft memset r0")) return NAN;
    if (!cuda_ok(cudaMemset(d_r1, 2, bytes_r), "fft memset r1")) return NAN;

    cufftHandle fwd = 0;
    cufftHandle inv = 0;
    if (!cufft_ok(cufftCreate(&fwd), "cufftCreate fwd")) return NAN;
    if (!cufft_ok(cufftCreate(&inv), "cufftCreate inv")) return NAN;
    int rank = 1;
    int dims[1] = {fft_n};
    int inembed[1] = {fft_n};
    int onembed[1] = {cn};
    size_t ws = 0;
    if (!cufft_ok(cufftMakePlanMany(fwd, rank, dims,
                                    inembed, 1, fft_n,
                                    onembed, 1, cn,
                                    CUFFT_D2Z, batch, &ws), "cufftMakePlanMany fwd")) return NAN;
    if (!cufft_ok(cufftMakePlanMany(inv, rank, dims,
                                    onembed, 1, cn,
                                    inembed, 1, fft_n,
                                    CUFFT_Z2D, batch, &ws), "cufftMakePlanMany inv")) return NAN;

    int threads = 256;
    int blocks = (batch * cn + threads - 1) / threads;
    int warmup = 3;
    for (int i = 0; i < warmup; ++i) {
        cufftExecD2Z(fwd, d_r0, d_c0);
        cufftExecD2Z(fwd, d_r1, d_c1);
        k_pointwise_mul<<<blocks, threads>>>(d_c0, d_c1, batch * cn);
        cufftExecZ2D(inv, d_c0, d_r0);
    }
    if (!cuda_ok(cudaDeviceSynchronize(), "fft warmup sync")) return NAN;

    int reps = 7;
    std::vector<double> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        double t0 = now_ns();
        cufftExecD2Z(fwd, d_r0, d_c0);
        cufftExecD2Z(fwd, d_r1, d_c1);
        k_pointwise_mul<<<blocks, threads>>>(d_c0, d_c1, batch * cn);
        cufftExecZ2D(inv, d_c0, d_r0);
        if (!cuda_ok(cudaDeviceSynchronize(), "fft rep sync")) return NAN;
        samples.push_back(now_ns() - t0);
    }

    cufftDestroy(fwd);
    cufftDestroy(inv);
    cudaFree(d_r0);
    cudaFree(d_r1);
    cudaFree(d_c0);
    cudaFree(d_c1);
    return median(samples) / (double)batch;
}

int main(int argc, char **argv) {
    int max_len = 4096;
    if (argc > 1) max_len = atoi(argv[1]);
    if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice")) return 1;

    std::vector<int> smooth;
    build_smooth_table(1 << 20, smooth);

    std::vector<int> lens = {
        8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
        768, 1024, 1536, 2048, 3072, 4096
    };

    printf("len,fft_n,batch,seq_ns_per_conv,fft_ns_per_conv,seq_over_fft\n");
    int crossover = -1;
    for (int len : lens) {
        if (len > max_len) continue;
        int batch = pick_batch_for_len(len);
        int fft_n = next_7smooth_ge(smooth, 2 * len - 1);
        double seq_ns = measure_seq_ns_per_conv(len, batch);
        double fft_ns = measure_fft_ns_per_conv(len, fft_n, batch);
        double ratio = seq_ns / fft_ns;
        printf("%d,%d,%d,%.2f,%.2f,%.3f\n", len, fft_n, batch, seq_ns, fft_ns, ratio);
        fflush(stdout);
        if (crossover < 0 && std::isfinite(seq_ns) && std::isfinite(fft_ns) && fft_ns < seq_ns) {
            crossover = len;
        }
    }

    if (crossover > 0) {
        printf("CROSSOVER_LEN=%d\n", crossover);
    } else {
        printf("CROSSOVER_LEN=not-found-up-to-%d\n", max_len);
    }
    return 0;
}
