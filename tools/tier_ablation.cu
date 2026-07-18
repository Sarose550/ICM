/* tools/tier_ablation.cu — Tier Ablation Runner (E3 Node)
 *
 * Per B200_RESUME_CHECKPOINT.md step 4: for a set of representative tree
 * levels / convolution sizes, directly time schoolbook kernel vs cuFFTDx
 * fused kernel vs batched cuFFT for the same convolution (build + correlate
 * pair), printing a CSV of measured crossovers so they can be hard-wired
 * into the planner.
 *
 * Output CSV columns:
 *   size, batch, t_schoolbook_ms, t_fused_ms, t_cufft_ms, winner
 *
 * "size" is the convolution size (build conv = size, corr conv ≈ size).
 * "batch" is the number of parent nodes at this tree level.
 *
 * Build: make tier_ablation CUDA_ARCH=sm_100
 *
 * STATUS: UNTESTED — drafted locally, cannot compile/run without a GPU.
 * Compile-plausible: mirrors includes from tools/gpu_phase_profile.cu.
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

/* ── Helpers ───────────────────────────────────────────────────── */

static inline double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static double median(std::vector<double> &v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* ── Schoolbook timing ─────────────────────────────────────────── */

static double time_schoolbook_pair(int conv_size, int batch, int warmup, int reps,
                                   cudaStream_t stream) {
    /* conv_size is the "build" convolution length.
     * Schoolbook build: correlate two child polys of degree cps → parent of degree pps.
     * For a fair comparison, use cps = conv_size (below_sat), pps = cps.
     *
     * Build: each parent = child[p] * child[p+1] for p in [0, batch).
     *        Effective: cps × cps convolution → 2*cps-1 output.
     * Corr:  parent_g (len g_eff ≈ cps) × child_poly (len p_eff ≈ cps)
     *        → child_g (len out_needed ≈ cps).
     *
     * We use cps = conv_size / 2, pps = cps, which is the below_sat regime
     * (the common case for power-of-two tree levels). */

    int cps = conv_size / 2;          /* child poly size */
    if (cps < 2) cps = 2;
    int pps = cps;                     /* parent poly size (below_sat: psz unchanged) */
    int stride = cps;                  /* compact stride = poly size */
    int nparents = batch;

    /* Allocate device memory */
    size_t child_bytes = (size_t)(2 * nparents) * stride * sizeof(double);
    size_t parent_bytes = (size_t)nparents * stride * sizeof(double);
    double *d_child = nullptr, *d_parent = nullptr, *d_g_parent = nullptr, *d_g_child = nullptr;
    size_t g_bytes = (size_t)nparents * stride * sizeof(double);

    if (cudaMalloc(&d_child, child_bytes) != cudaSuccess) return -1.0;
    if (cudaMalloc(&d_parent, parent_bytes) != cudaSuccess) { cudaFree(d_child); return -1.0; }
    if (cudaMalloc(&d_g_parent, g_bytes) != cudaSuccess) { cudaFree(d_child); cudaFree(d_parent); return -1.0; }
    if (cudaMalloc(&d_g_child, 2 * g_bytes) != cudaSuccess) {
        cudaFree(d_child); cudaFree(d_parent); cudaFree(d_g_parent); return -1.0;
    }

    cudaMemsetAsync(d_child, 0, child_bytes, stream);
    cudaMemsetAsync(d_parent, 0, parent_bytes, stream);
    cudaMemsetAsync(d_g_parent, 0, g_bytes, stream);
    cudaMemsetAsync(d_g_child, 0, 2 * g_bytes, stream);
    cudaStreamSynchronize(stream);

    int threads = GPU_THREADS_PER_BLOCK;
    int blocks_build = nparents;  /* one block per parent for smem variant */
    size_t shmem_build = (size_t)(2 * cps) * sizeof(double);

    int len_g = cps;
    int len_P = cps;
    int len_out = cps;
    int blocks_corr = nparents;
    size_t shmem_corr = (size_t)(len_g + 2 * len_P) * sizeof(double);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    /* Warmup */
    for (int i = 0; i < warmup; ++i) {
        k_schoolbook_build_smem_parent<<<blocks_build, threads, shmem_build, stream>>>(
            d_child, cps, d_parent, pps, nparents, stride, stride);
        k_schoolbook_corr_pair_smem_parent<<<blocks_corr, threads, shmem_corr, stream>>>(
            d_g_parent, stride, len_g,
            d_child, stride, len_P,
            d_g_child, stride, len_out, nparents, stride, stride, stride);
    }
    cudaStreamSynchronize(stream);

    /* Timing */
    std::vector<double> samples;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(e0, stream);
        k_schoolbook_build_smem_parent<<<blocks_build, threads, shmem_build, stream>>>(
            d_child, cps, d_parent, pps, nparents, stride, stride);
        k_schoolbook_corr_pair_smem_parent<<<blocks_corr, threads, shmem_corr, stream>>>(
            d_g_parent, stride, len_g,
            d_child, stride, len_P,
            d_g_child, stride, len_out, nparents, stride, stride, stride);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        samples.push_back((double)ms);
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_child);
    cudaFree(d_parent);
    cudaFree(d_g_parent);
    cudaFree(d_g_child);

    return median(samples);
}

/* ── cuFFTDx fused timing ──────────────────────────────────────── */

static double time_fused_pair(int fft_n, int conv_size, int batch, int warmup, int reps,
                              cudaStream_t stream) {
    (void)conv_size;
    if (!is_cufftdx_supported_fft_n(fft_n)) return -1.0;

    int nparents = batch;
    int cps = std::min(fft_n, 2 * conv_size);   /* child poly size (not exceeding fft_n) */
    int pps = cps;                                /* parent poly size */
    int stride = cps;

    size_t child_bytes = (size_t)(2 * nparents) * stride * sizeof(double);
    size_t parent_bytes = (size_t)nparents * stride * sizeof(double);
    size_t g_bytes = (size_t)nparents * stride * sizeof(double);
    double *d_child = nullptr, *d_parent = nullptr, *d_g_parent = nullptr, *d_g_child = nullptr;

    if (cudaMalloc(&d_child, child_bytes) != cudaSuccess) return -1.0;
    if (cudaMalloc(&d_parent, parent_bytes) != cudaSuccess) { cudaFree(d_child); return -1.0; }
    if (cudaMalloc(&d_g_parent, g_bytes) != cudaSuccess) { cudaFree(d_child); cudaFree(d_parent); return -1.0; }
    if (cudaMalloc(&d_g_child, 2 * g_bytes) != cudaSuccess) {
        cudaFree(d_child); cudaFree(d_parent); cudaFree(d_g_parent); return -1.0;
    }

    cudaMemsetAsync(d_child, 0, child_bytes, stream);
    cudaMemsetAsync(d_parent, 0, parent_bytes, stream);
    cudaMemsetAsync(d_g_parent, 0, g_bytes, stream);
    cudaMemsetAsync(d_g_child, 0, 2 * g_bytes, stream);
    cudaStreamSynchronize(stream);

    double inv_fft_n = 1.0 / (double)fft_n;
    int len_g = cps, len_P = cps, len_out = cps;

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    /* Warmup */
    for (int i = 0; i < warmup; ++i) {
        launch_cufftdx_build_r2c_dispatch(fft_n, d_child, cps, d_parent, pps,
                                          nparents, inv_fft_n, stream, stride, stride);
        launch_cufftdx_corr_r2c_dispatch(fft_n, d_g_parent, stride, len_g,
                                         d_child, stride, len_P,
                                         d_g_child, stride, len_out, nparents,
                                         inv_fft_n, stream, stride, stride, stride);
    }
    cudaStreamSynchronize(stream);

    std::vector<double> samples;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(e0, stream);
        launch_cufftdx_build_r2c_dispatch(fft_n, d_child, cps, d_parent, pps,
                                          nparents, inv_fft_n, stream, stride, stride);
        launch_cufftdx_corr_r2c_dispatch(fft_n, d_g_parent, stride, len_g,
                                         d_child, stride, len_P,
                                         d_g_child, stride, len_out, nparents,
                                         inv_fft_n, stream, stride, stride, stride);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        samples.push_back((double)ms);
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_child);
    cudaFree(d_parent);
    cudaFree(d_g_parent);
    cudaFree(d_g_child);

    return median(samples);
}

/* ── Batched cuFFT timing ──────────────────────────────────────── */

static double time_cufft_pair(int fft_n, int conv_size, int batch, int warmup, int reps,
                              cudaStream_t stream) {
    (void)conv_size;
    int nparents = batch;
    int cn = fft_n / 2 + 1;
    int cps = std::min(fft_n, 2 * conv_size);
    int stride = cps;

    /* cuFFT plans: build R2C + C2R, corr R2C + C2R */
    cufftHandle plan_bfwd = 0, plan_binv = 0, plan_cfwd = 0, plan_cinv = 0;
    int child_batch = 2 * nparents;
    int parent_batch = nparents;
    int corr_child_batch = 2 * nparents;

    if (!create_cufft_plan(&plan_bfwd, fft_n, child_batch, true, fft_n)) return -1.0;
    if (!create_cufft_plan(&plan_binv, fft_n, parent_batch, false, fft_n)) return -1.0;
    if (!create_cufft_plan(&plan_cfwd, fft_n, parent_batch, true, fft_n)) return -1.0;
    if (!create_cufft_plan(&plan_cinv, fft_n, corr_child_batch, false, fft_n)) return -1.0;
    cufftSetStream(plan_bfwd, stream);
    cufftSetStream(plan_binv, stream);
    cufftSetStream(plan_cfwd, stream);
    cufftSetStream(plan_cinv, stream);

    /* Workspaces */
    size_t ws = 0, w = 0;
    cufftGetSize(plan_bfwd, &w); ws = std::max(ws, w);
    cufftGetSize(plan_binv, &w); ws = std::max(ws, w);
    cufftGetSize(plan_cfwd, &w); ws = std::max(ws, w);
    cufftGetSize(plan_cinv, &w); ws = std::max(ws, w);
    void *d_workspace = nullptr;
    if (ws > 0) {
        if (cudaMalloc(&d_workspace, ws) != cudaSuccess) {
            cufftDestroy(plan_bfwd); cufftDestroy(plan_binv);
            cufftDestroy(plan_cfwd); cufftDestroy(plan_cinv);
            return -1.0;
        }
        cufftSetWorkArea(plan_bfwd, d_workspace);
        cufftSetWorkArea(plan_binv, d_workspace);
        cufftSetWorkArea(plan_cfwd, d_workspace);
        cufftSetWorkArea(plan_cinv, d_workspace);
    }

    /* Device arrays */
    size_t child_bytes = (size_t)child_batch * stride * sizeof(double);
    size_t parent_bytes = (size_t)parent_batch * stride * sizeof(double);
    size_t g_bytes = (size_t)parent_batch * stride * sizeof(double);
    size_t spec_build_bytes = (size_t)child_batch * cn * sizeof(cufftDoubleComplex);
    size_t spec_mid_bytes = (size_t)parent_batch * cn * sizeof(cufftDoubleComplex);
    size_t spec_corr_in_bytes = (size_t)parent_batch * cn * sizeof(cufftDoubleComplex);
    size_t spec_corr_out_bytes = (size_t)corr_child_batch * cn * sizeof(cufftDoubleComplex);

    double *d_child = nullptr, *d_parent = nullptr, *d_g_parent = nullptr, *d_g_child = nullptr;
    cufftDoubleComplex *d_spec_build = nullptr, *d_spec_mid = nullptr;
    cufftDoubleComplex *d_spec_corr_in = nullptr, *d_spec_corr_out = nullptr;
    double *d_scratch = nullptr;
    size_t scratch_bytes = (size_t)child_batch * fft_n * sizeof(double);

    bool ok = true;
    ok = ok && (cudaMalloc(&d_child, child_bytes) == cudaSuccess);
    ok = ok && (cudaMalloc(&d_parent, parent_bytes) == cudaSuccess);
    ok = ok && (cudaMalloc(&d_g_parent, g_bytes) == cudaSuccess);
    ok = ok && (cudaMalloc(&d_g_child, 2 * g_bytes) == cudaSuccess);
    ok = ok && (cudaMalloc(&d_spec_build, spec_build_bytes) == cudaSuccess);
    ok = ok && (cudaMalloc(&d_spec_mid, spec_mid_bytes) == cudaSuccess);
    ok = ok && (cudaMalloc(&d_spec_corr_in, spec_corr_in_bytes) == cudaSuccess);
    ok = ok && (cudaMalloc(&d_spec_corr_out, spec_corr_out_bytes) == cudaSuccess);
    ok = ok && (cudaMalloc(&d_scratch, scratch_bytes) == cudaSuccess);

    if (!ok) {
        /* cleanup and return */
        if (d_child) cudaFree(d_child);
        if (d_parent) cudaFree(d_parent);
        if (d_g_parent) cudaFree(d_g_parent);
        if (d_g_child) cudaFree(d_g_child);
        if (d_spec_build) cudaFree(d_spec_build);
        if (d_spec_mid) cudaFree(d_spec_mid);
        if (d_spec_corr_in) cudaFree(d_spec_corr_in);
        if (d_spec_corr_out) cudaFree(d_spec_corr_out);
        if (d_scratch) cudaFree(d_scratch);
        if (d_workspace) cudaFree(d_workspace);
        cufftDestroy(plan_bfwd); cufftDestroy(plan_binv);
        cufftDestroy(plan_cfwd); cufftDestroy(plan_cinv);
        return -1.0;
    }

    cudaMemsetAsync(d_child, 0, child_bytes, stream);
    cudaMemsetAsync(d_parent, 0, parent_bytes, stream);
    cudaMemsetAsync(d_g_parent, 0, g_bytes, stream);
    cudaMemsetAsync(d_g_child, 0, 2 * g_bytes, stream);
    cudaStreamSynchronize(stream);

    int threads = GPU_THREADS_PER_BLOCK;
    double inv_fft_n = 1.0 / (double)fft_n;

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    /* Warmup */
    for (int i = 0; i < warmup; ++i) {
        /* Build: gather → R2C → pairwise mul → C2R → scatter */
        k_gather_to_fft<<<(child_batch * fft_n + threads - 1) / threads, threads, 0, stream>>>(
            d_child, stride, d_scratch, fft_n, child_batch);
        cufftExecD2Z(plan_bfwd, d_scratch, d_spec_build);
        k_pairwise_mul<<<(parent_batch * cn + threads - 1) / threads, threads, 0, stream>>>(
            d_spec_build, cn, d_spec_mid, parent_batch, inv_fft_n);
        cufftExecZ2D(plan_binv, d_spec_mid, d_scratch);
        k_scatter_from_fft<<<(parent_batch * std::min(cps, fft_n) + threads - 1) / threads, threads, 0, stream>>>(
            d_scratch, fft_n, d_parent, stride, std::min(cps, fft_n), parent_batch);

        /* Corr: gather g → R2C; rebuild child spec; paired corr; C2R → scatter */
        k_gather_to_fft<<<(parent_batch * fft_n + threads - 1) / threads, threads, 0, stream>>>(
            d_g_parent, stride, d_scratch, fft_n, parent_batch);
        cufftExecD2Z(plan_cfwd, d_scratch, d_spec_corr_in);
        /* Recompute child spec (no cache in ablation) */
        k_gather_to_fft<<<(child_batch * fft_n + threads - 1) / threads, threads, 0, stream>>>(
            d_child, stride, d_scratch, fft_n, child_batch);
        cufftExecD2Z(plan_bfwd, d_scratch, d_spec_build);
        k_paired_corr_freq<<<(parent_batch * cn + threads - 1) / threads, threads, 0, stream>>>(
            d_spec_corr_in, d_spec_build, cn, parent_batch, d_spec_corr_out, inv_fft_n);
        cufftExecZ2D(plan_cinv, d_spec_corr_out, d_scratch);
        k_scatter_from_fft<<<(corr_child_batch * std::min(cps, fft_n) + threads - 1) / threads, threads, 0, stream>>>(
            d_scratch, fft_n, d_g_child, stride, std::min(cps, fft_n), corr_child_batch);
    }
    cudaStreamSynchronize(stream);

    std::vector<double> samples;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(e0, stream);
        /* Build */
        k_gather_to_fft<<<(child_batch * fft_n + threads - 1) / threads, threads, 0, stream>>>(
            d_child, stride, d_scratch, fft_n, child_batch);
        cufftExecD2Z(plan_bfwd, d_scratch, d_spec_build);
        k_pairwise_mul<<<(parent_batch * cn + threads - 1) / threads, threads, 0, stream>>>(
            d_spec_build, cn, d_spec_mid, parent_batch, inv_fft_n);
        cufftExecZ2D(plan_binv, d_spec_mid, d_scratch);
        k_scatter_from_fft<<<(parent_batch * std::min(cps, fft_n) + threads - 1) / threads, threads, 0, stream>>>(
            d_scratch, fft_n, d_parent, stride, std::min(cps, fft_n), parent_batch);
        /* Corr */
        k_gather_to_fft<<<(parent_batch * fft_n + threads - 1) / threads, threads, 0, stream>>>(
            d_g_parent, stride, d_scratch, fft_n, parent_batch);
        cufftExecD2Z(plan_cfwd, d_scratch, d_spec_corr_in);
        k_gather_to_fft<<<(child_batch * fft_n + threads - 1) / threads, threads, 0, stream>>>(
            d_child, stride, d_scratch, fft_n, child_batch);
        cufftExecD2Z(plan_bfwd, d_scratch, d_spec_build);
        k_paired_corr_freq<<<(parent_batch * cn + threads - 1) / threads, threads, 0, stream>>>(
            d_spec_corr_in, d_spec_build, cn, parent_batch, d_spec_corr_out, inv_fft_n);
        cufftExecZ2D(plan_cinv, d_spec_corr_out, d_scratch);
        k_scatter_from_fft<<<(corr_child_batch * std::min(cps, fft_n) + threads - 1) / threads, threads, 0, stream>>>(
            d_scratch, fft_n, d_g_child, stride, std::min(cps, fft_n), corr_child_batch);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        samples.push_back((double)ms);
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    /* Cleanup */
    cudaFree(d_child); cudaFree(d_parent); cudaFree(d_g_parent); cudaFree(d_g_child);
    cudaFree(d_spec_build); cudaFree(d_spec_mid);
    cudaFree(d_spec_corr_in); cudaFree(d_spec_corr_out);
    cudaFree(d_scratch);
    if (d_workspace) cudaFree(d_workspace);
    cufftDestroy(plan_bfwd); cufftDestroy(plan_binv);
    cufftDestroy(plan_cfwd); cufftDestroy(plan_cinv);

    return median(samples);
}

/* ── Main ──────────────────────────────────────────────────────── */

int main(void) {
    if (!icm_gpu_init(0)) {
        fprintf(stderr, "ERROR: icm_gpu_init failed\n");
        return 1;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    /* Representative convolution sizes (build conv = size).
     * These span the full range: within fused (≤8192), crossover
     * (16384–65536), and large (cuFFT-only, 131072+). */
    int sizes[] = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072, 262144, 524288
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    /* Representative batch sizes (parent count at a tree level).
     * Small batches (1, 4) represent upper tree levels;
     * medium (16, 64) represent mid-tree;
     * large (256) represents lower tree levels with high occupancy. */
    int batches[] = {1, 4, 16, 64, 256};
    int n_batches = sizeof(batches) / sizeof(batches[0]);

    int warmup = 2;
    int reps = 8;
    const char *rep_env = getenv("TIER_ABLATION_REPS");
    if (rep_env && rep_env[0]) { int r = atoi(rep_env); if (r > 0) reps = r; }

    printf("size,batch,t_schoolbook_ms,t_fused_ms,t_cufft_ms,winner\n");
    fflush(stdout);

    for (int si = 0; si < n_sizes; ++si) {
        int size = sizes[si];
        /* FFT size for this convolution: next power of 2 (or calibrated size).
         * For ablation we use the next power of 2 as the neutral baseline. */
        int fft_n = 1;
        while (fft_n < size) fft_n <<= 1;
        /* Ensure fft_n is at least 2× the conv for safe R2C (needs fft_n ≥ conv+1) */
        if (fft_n <= size) fft_n <<= 1;

        for (int bi = 0; bi < n_batches; ++bi) {
            int batch = batches[bi];

            /* Skip combinations that would exceed VRAM on a test GPU.
             * The cuFFT path allocates ~batch × fft_n × several buffers;
             * cap at roughly 4 GB of temporary allocations. */
            size_t cufft_est = (size_t)batch * (size_t)fft_n * 6 * sizeof(cufftDoubleComplex);
            if (cufft_est > 4ULL * 1024 * 1024 * 1024) continue;

            double t_sb = time_schoolbook_pair(size, batch, warmup, reps, stream);
            double t_fused = time_fused_pair(fft_n, size, batch, warmup, reps, stream);
            double t_cufft = time_cufft_pair(fft_n, size, batch, warmup, reps, stream);

            const char *winner = "none";
            double best = -1.0;
            if (t_sb >= 0 && (best < 0 || t_sb < best)) { best = t_sb; winner = "schoolbook"; }
            if (t_fused >= 0 && (best < 0 || t_fused < best)) { best = t_fused; winner = "fused"; }
            if (t_cufft >= 0 && (best < 0 || t_cufft < best)) { best = t_cufft; winner = "cufft"; }

            printf("%d,%d,%.4f,%.4f,%.4f,%s\n",
                   size, batch,
                   t_sb >= 0 ? t_sb : -1.0,
                   t_fused >= 0 ? t_fused : -1.0,
                   t_cufft >= 0 ? t_cufft : -1.0,
                   winner);
            fflush(stdout);

            fprintf(stderr, "  size=%d batch=%d: sb=%.3f fus=%.3f cufft=%.3f -> %s\n",
                    size, batch,
                    t_sb >= 0 ? t_sb : -1.0,
                    t_fused >= 0 ? t_fused : -1.0,
                    t_cufft >= 0 ? t_cufft : -1.0,
                    winner);
        }
    }

    cudaStreamDestroy(stream);
    fprintf(stderr, "\nTier ablation complete.\n");
    return 0;
}
