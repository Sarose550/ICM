/*
 * bench_vkfft.cu — Compare VkFFT vs cuFFT for R2C/C2R FP64 pipeline
 *
 * For each 7-smooth size from fft_config.h:
 *   - R2C forward + pointwise complex multiply + C2R inverse
 *   - batch=64, FP64, 3 reps, min time
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_100 -I/root/VkFFT/vkFFT -I/root/ICM/src \
 *        -I/root/ICM/devices/zen4 -DVKFFT_BACKEND=1 \
 *        -o bench_vkfft tools/bench_vkfft.cu -lcufft -lcudart
 *
 * Output: CSV to stdout (size,cufft_ns,vkfft_ns,winner,speedup)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <cufft.h>

#define VKFFT_BACKEND 1
#include "vkFFT.h"

/* Pull in calib_sizes from fft_config.h */
#include "fft_config.h"

#define BATCH 64
#define REPS 3
#define WARMUP 1

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* Simple pointwise multiply kernel: out[i] *= filter[i] for complex doubles */
__global__ void pointwise_multiply(cufftDoubleComplex *data,
                                    const cufftDoubleComplex *filter,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double a = data[idx].x, b = data[idx].y;
        double c = filter[idx].x, d = filter[idx].y;
        data[idx].x = a*c - b*d;
        data[idx].y = a*d + b*c;
    }
}

/* Benchmark cuFFT R2C -> pointwise -> C2R pipeline */
static double bench_cufft(int fft_size, double *d_real, cufftDoubleComplex *d_complex,
                          cufftDoubleComplex *d_filter, cudaStream_t stream) {
    int complex_size = fft_size / 2 + 1;
    int total_complex = complex_size * BATCH;

    /* Create plans */
    cufftHandle plan_fwd, plan_inv;
    int n_arr[1] = {fft_size};
    size_t workSize;

    int ret;
    ret = cufftPlanMany(&plan_fwd, 1, n_arr,
                        n_arr, 1, fft_size,
                        n_arr, 1, complex_size,
                        CUFFT_D2Z, BATCH);
    if (ret != CUFFT_SUCCESS) return -1;

    ret = cufftPlanMany(&plan_inv, 1, n_arr,
                        n_arr, 1, complex_size,
                        n_arr, 1, fft_size,
                        CUFFT_Z2D, BATCH);
    if (ret != CUFFT_SUCCESS) { cufftDestroy(plan_fwd); return -1; }

    cufftSetStream(plan_fwd, stream);
    cufftSetStream(plan_inv, stream);

    int threads = 256;
    int blocks = (total_complex + threads - 1) / threads;

    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        cufftExecD2Z(plan_fwd, d_real, d_complex);
        pointwise_multiply<<<blocks, threads, 0, stream>>>(d_complex, d_filter, total_complex);
        cufftExecZ2D(plan_inv, d_complex, d_real);
    }
    cudaStreamSynchronize(stream);

    /* Timed reps */
    double best_ns = 1e18;
    for (int r = 0; r < REPS; r++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        cufftExecD2Z(plan_fwd, d_real, d_complex);
        pointwise_multiply<<<blocks, threads, 0, stream>>>(d_complex, d_filter, total_complex);
        cufftExecZ2D(plan_inv, d_complex, d_real);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        double ns = (double)ms * 1e6;
        if (ns < best_ns) best_ns = ns;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    return best_ns;
}

/* Benchmark VkFFT R2C -> pointwise -> C2R pipeline */
static double bench_vkfft(int fft_size, double *d_real, cufftDoubleComplex *d_complex,
                          cufftDoubleComplex *d_filter, cudaStream_t stream) {
    int complex_size = fft_size / 2 + 1;
    int total_complex = complex_size * BATCH;

    /* Buffer size — must be large enough for in-place R2C */
    uint64_t buf_size = (uint64_t)complex_size * BATCH * sizeof(cufftDoubleComplex);

    /* VkFFT configuration */
    VkFFTConfiguration config = {};
    config.FFTdim = 1;
    config.size[0] = fft_size;
    config.numberBatches = BATCH;
    config.doublePrecision = 1;
    config.performR2C = 1;
    config.bufferSize = &buf_size;
    config.bufferNum = 1;

    /* CUDA device/stream */
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);
    config.device = &cuDevice;
    config.stream = &stream;
    config.num_streams = 1;

    VkFFTApplication app = {};
    VkFFTResult res = initializeVkFFT(&app, config);
    if (res != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFT init failed for size %d: error %d\n", fft_size, (int)res);
        return -1;
    }

    int threads = 256;
    int blocks = (total_complex + threads - 1) / threads;

    /* Launch params — for CUDA backend, only buffer pointer needed */
    VkFFTLaunchParams launchParams = {};
    launchParams.buffer = (void**)&d_real;

    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        VkFFTAppend(&app, -1, &launchParams);  /* forward R2C */
        pointwise_multiply<<<blocks, threads, 0, stream>>>(
            (cufftDoubleComplex*)d_real, d_filter, total_complex);
        VkFFTAppend(&app, 1, &launchParams);   /* inverse C2R */
    }
    cudaStreamSynchronize(stream);

    /* Timed reps */
    double best_ns = 1e18;
    for (int r = 0; r < REPS; r++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        VkFFTAppend(&app, -1, &launchParams);  /* forward R2C */
        pointwise_multiply<<<blocks, threads, 0, stream>>>(
            (cufftDoubleComplex*)d_real, d_filter, total_complex);
        VkFFTAppend(&app, 1, &launchParams);   /* inverse C2R */

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        double ns = (double)ms * 1e6;
        if (ns < best_ns) best_ns = ns;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    deleteVkFFT(&app);
    return best_ns;
}

int main() {
    /* Initialize CUDA */
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* Initialize CUDA driver API for VkFFT */
    cuInit(0);

    /* Print header */
    fprintf(stdout, "size,cufft_ns,vkfft_ns,winner,speedup\n");
    fflush(stdout);

    int cufft_wins = 0, vkfft_wins = 0, ties = 0, errors = 0;

    /* For each calibrated size */
    for (int i = 0; i < N_CALIBRATED_SIZES; i++) {
        int fft_size = calib_sizes[i];

        /* Skip very small sizes (< 16) — not meaningful for GPU */
        if (fft_size < 16) {
            continue;
        }

        int complex_size = fft_size / 2 + 1;
        size_t real_bytes = (size_t)fft_size * BATCH * sizeof(double);
        /* For VkFFT R2C in-place, the buffer must be large enough for complex output */
        size_t inplace_bytes = (size_t)complex_size * BATCH * sizeof(cufftDoubleComplex);
        size_t alloc_bytes = (real_bytes > inplace_bytes) ? real_bytes : inplace_bytes;
        size_t complex_bytes = (size_t)complex_size * BATCH * sizeof(cufftDoubleComplex);

        double *d_real = nullptr;
        cufftDoubleComplex *d_complex = nullptr;
        cufftDoubleComplex *d_filter = nullptr;

        cudaError_t err;
        err = cudaMalloc(&d_real, alloc_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "OOM at size %d (real alloc %zu bytes)\n", fft_size, alloc_bytes);
            break;
        }
        err = cudaMalloc(&d_complex, complex_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_real);
            fprintf(stderr, "OOM at size %d (complex alloc %zu bytes)\n", fft_size, complex_bytes);
            break;
        }
        err = cudaMalloc(&d_filter, complex_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_real); cudaFree(d_complex);
            fprintf(stderr, "OOM at size %d (filter alloc %zu bytes)\n", fft_size, complex_bytes);
            break;
        }

        /* Initialize with random-ish data */
        cudaMemset(d_real, 0, alloc_bytes);
        cudaMemset(d_complex, 0, complex_bytes);
        cudaMemset(d_filter, 1, complex_bytes); /* non-zero filter */

        /* Benchmark cuFFT */
        double cufft_ns = bench_cufft(fft_size, d_real, d_complex, d_filter, stream);

        /* Benchmark VkFFT — use d_real as in-place buffer (R2C in-place) */
        cudaMemset(d_real, 0, alloc_bytes);
        double vkfft_ns = bench_vkfft(fft_size, d_real, d_complex, d_filter, stream);

        /* Report */
        if (cufft_ns > 0 && vkfft_ns > 0) {
            const char *winner;
            double speedup;
            if (vkfft_ns < cufft_ns * 0.99) {
                winner = "vkfft";
                speedup = cufft_ns / vkfft_ns;
                vkfft_wins++;
            } else if (cufft_ns < vkfft_ns * 0.99) {
                winner = "cufft";
                speedup = vkfft_ns / cufft_ns;
                cufft_wins++;
            } else {
                winner = "tie";
                speedup = 1.0;
                ties++;
            }
            fprintf(stdout, "%d,%.1f,%.1f,%s,%.3f\n",
                    fft_size, cufft_ns, vkfft_ns, winner, speedup);
        } else {
            const char *note = "error";
            if (cufft_ns < 0 && vkfft_ns < 0) note = "both_fail";
            else if (cufft_ns < 0) note = "cufft_fail";
            else note = "vkfft_fail";
            fprintf(stdout, "%d,%.1f,%.1f,%s,0.000\n",
                    fft_size, cufft_ns, vkfft_ns, note);
            errors++;
        }
        fflush(stdout);

        cudaFree(d_real);
        cudaFree(d_complex);
        cudaFree(d_filter);
    }

    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "cuFFT wins: %d\n", cufft_wins);
    fprintf(stderr, "VkFFT wins: %d\n", vkfft_wins);
    fprintf(stderr, "Ties (<1%% diff): %d\n", ties);
    fprintf(stderr, "Errors: %d\n", errors);
    fprintf(stderr, "Total tested: %d\n", cufft_wins + vkfft_wins + ties + errors);

    cudaStreamDestroy(stream);
    return 0;
}
