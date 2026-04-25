/* gpu_exec.cu -- Tree level execution, hybrid runners, graph capture, destroy. */
#include "gpu_internal.h"

namespace icm_gpu_detail {

/* ── make_nodes (quadrature) ───────────────────────────────────── */

static double log_Phi(double y) {
    if (y >= 0) return log1p(-erfc(y / sqrt(2.0)) / 2.0);
    return log(erfc(-y / sqrt(2.0)) / 2.0);
}

void make_nodes(int Q, double Smax, std::vector<QP> &pts) {
    pts.resize(Q);
    double y_lo = -7.7;
    double y_hi = sqrt(2.0) * sqrt(log(Smax) + 25.0);
    if (y_hi < 6.5) y_hi = 6.5;
    double h = (Q > 1) ? (y_hi - y_lo) / (Q - 1) : 0.0;
    for (int q = 0; q < Q; ++q) {
        double y = y_lo + q * h;
        pts[q].logv = log_Phi(y);
        pts[q].w = h * exp(-0.5 * y * y) / sqrt(2.0 * M_PI);
    }
}

int single_kernel_max_n(int k) {
    (void)k;
    const char *env = getenv("ICM_GPU_SINGLE_KERNEL_MAX_N");
    if (env && env[0]) {
        int v = atoi(env);
        if (v > 0) return v;
    }
    return 1024;
}

/* ── destroy_fft_buffers / destroy_plan ────────────────────────── */

bool destroy_fft_buffers(GpuPlan *plan, GpuFftBuffers &b, cudaStream_t stream) {
    (void)plan; (void)stream;
#if ICM_HAVE_VKFFT
    if (b.vkfft_fwd_initialized) { destroy_vkfft_app(&b.vkfft_app_fwd); b.vkfft_fwd_initialized = 0; }
    if (b.vkfft_inv_initialized) { destroy_vkfft_app(&b.vkfft_app_inv); b.vkfft_inv_initialized = 0; }
#endif
    if (b.plan_fwd) { if (!CUFFT_OK(cufftDestroy(b.plan_fwd))) return false; b.plan_fwd = 0; }
    if (b.plan_inv) { if (!CUFFT_OK(cufftDestroy(b.plan_inv))) return false; b.plan_inv = 0; }
    b = GpuFftBuffers{};
    return true;
}

void destroy_plan(GpuPlan *plan) {
    if (!plan) return;
    cudaStream_t stream = plan->stream_compute;
    if (stream) cudaStreamSynchronize(stream);

    for (int ell = 1; ell < plan->L; ++ell) {
        destroy_fft_buffers(plan, plan->build_fft[ell], stream);
        destroy_fft_buffers(plan, plan->corr_fft[ell], stream);
    }

    if (plan->shared_cufft_workspace) {
        cudaFree(plan->shared_cufft_workspace);
        plan->shared_cufft_workspace = nullptr;
    }

    if (plan->arena_base) {
        cudaFree(plan->arena_base);
        plan->arena_base = nullptr;
    }

    for (int q = 0; q < 2; ++q) {
        if (plan->graph_exec[q]) cudaGraphExecDestroy(plan->graph_exec[q]);
        if (plan->graph[q]) cudaGraphDestroy(plan->graph[q]);
    }
    if (plan->evt_a_ready[0]) cudaEventDestroy(plan->evt_a_ready[0]);
    if (plan->evt_a_ready[1]) cudaEventDestroy(plan->evt_a_ready[1]);
    if (plan->evt_prop_done) cudaEventDestroy(plan->evt_prop_done);
    if (plan->stream_aux) cudaStreamDestroy(plan->stream_aux);
    if (plan->stream_compute) cudaStreamDestroy(plan->stream_compute);
    delete plan;
}

/* ── Single-Q build level runners ──────────────────────────────── */

bool run_build_level_schoolbook(GpuPlan *plan, int ell) {
    int cps = plan->psz[ell - 1];
    int pps = plan->psz[ell];
    int child_stride = plan->fft_stride[ell - 1];
    int parent_stride = plan->fft_stride[ell];
    int nparents = plan->nn[ell];
    if (nparents <= 0 || cps <= 0 || pps <= 0) return true;

    int conv = plan->levels[ell].build_conv;
    bool use_warp_regime = (conv <= GPU_SCHOOL_WARP_MAX_CONV && nparents > 1);
    if (use_warp_regime) {
        int threads = GPU_SCHOOL_WARPS_PER_BLOCK * 32;
        int blocks = (nparents + GPU_SCHOOL_WARPS_PER_BLOCK - 1) / GPU_SCHOOL_WARPS_PER_BLOCK;
        size_t shmem = (size_t)GPU_SCHOOL_WARPS_PER_BLOCK * (size_t)(2 * cps) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_build_warp_batch<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents,
                child_stride, parent_stride);
        } else {
            int fb_threads = GPU_THREADS_PER_BLOCK;
            size_t total = (size_t)nparents * (size_t)pps;
            int fb_blocks = (int)((total + fb_threads - 1) / fb_threads);
            k_schoolbook_build<<<fb_blocks, fb_threads, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents,
                child_stride, parent_stride);
        }
    } else {
        int threads = GPU_THREADS_PER_BLOCK;
        int blocks = nparents;
        size_t shmem = (size_t)(2 * cps) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_build_smem_parent<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents,
                child_stride, parent_stride);
        } else {
            size_t total = (size_t)nparents * (size_t)pps;
            int fb_blocks = (int)((total + threads - 1) / threads);
            k_schoolbook_build<<<fb_blocks, threads, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents,
                child_stride, parent_stride);
        }
    }
    if (!CUDA_OK(cudaGetLastError())) return false;
    return true;
}

bool run_build_level_fft(GpuPlan *plan, int ell) {
    int pps = plan->psz[ell];
    int child_batch = plan->nn[ell - 1];
    int parent_batch = plan->nn[ell];
    int parent_stride = plan->fft_stride[ell];
    int child_stride = plan->fft_stride[ell - 1];
    auto &lp = plan->levels[ell];
    auto &b = plan->build_fft[ell];
    int threads = GPU_THREADS_PER_BLOCK;
    int qb = plan->q_batch;
    int total_child = qb * child_batch;
    int total_parent = qb * parent_batch;

#if ICM_HAVE_VKFFT
    if (b.use_vkfft) {
        /* VkFFT out-of-place R2C path — no gather/scatter.
         * Forward: reads strided real from poly_levels (inputBuffer),
         *          writes contiguous complex to spec_in (buffer).
         * Inverse: reads contiguous complex from buffer,
         *          writes strided real to poly_levels (outputBuffer). */
        int fft_n = b.fft_n;
        int cn = b.cn;

        /* 1. VkFFT R2C forward: poly_levels[ell-1] → spec_in */
        {
            double *input_ptr = plan->d_poly_levels[ell - 1];
            VkFFTLaunchParams lp_fwd = {};
            lp_fwd.buffer = (void **)&b.spec_in;
            lp_fwd.inputBuffer = (void **)&input_ptr;
            VkFFTResult res = VkFFTAppend(&b.vkfft_app_fwd, -1, &lp_fwd);
            if (res != VKFFT_SUCCESS) return false;
        }

        /* spec_in now holds contiguous complex spectra (same as cuFFT output) */
        cufftDoubleComplex *fwd_out = (lp.cache_fft && plan->d_fft_cache[ell]) ? plan->d_fft_cache[ell] : b.spec_in;
        if (fwd_out != b.spec_in) {
            size_t copy_bytes = (size_t)total_child * cn * sizeof(cufftDoubleComplex);
            if (!CUDA_OK(cudaMemcpyAsync(fwd_out, b.spec_in, copy_bytes,
                                          cudaMemcpyDeviceToDevice, plan->stream_compute))) return false;
        }
        if (lp.cache_fft && plan->d_fft_cache[ell]) {
            if (ell < (int)plan->fft_cache_valid.size()) plan->fft_cache_valid[ell] = true;
        }

        /* 2. Pairwise multiply (contiguous complex — identical to cuFFT path) */
        cufftDoubleComplex *mul_out = (fwd_out != b.spec_in) ? b.spec_in : b.spec_mid;
        size_t mul_total = (size_t)total_parent * (size_t)cn;
        int blocks_mul = (int)((mul_total + threads - 1) / threads);
        double inv_fft_n = 1.0 / (double)fft_n;
        k_pairwise_mul<<<blocks_mul, threads, 0, plan->stream_compute>>>(
            fwd_out, cn, mul_out, total_parent, inv_fft_n);
        if (!CUDA_OK(cudaGetLastError())) return false;

        /* 3. VkFFT C2R inverse: mul_out → poly_levels[ell] */
        {
            double *output_ptr = plan->d_poly_levels[ell];
            VkFFTLaunchParams lp_inv = {};
            lp_inv.buffer = (void **)&mul_out;
            lp_inv.outputBuffer = (void **)&output_ptr;
            VkFFTResult res = VkFFTAppend(&b.vkfft_app_inv, 1, &lp_inv);
            if (res != VKFFT_SUCCESS) return false;
        }

        if (lp.build_wrap_m > 0) {
            int cps = plan->psz[ell - 1];
            k_wrap_build<<<total_parent, 64, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell], pps, total_parent,
                plan->d_poly_levels[ell - 1], cps, lp.build_conv,
                fft_n, lp.build_wrap_m,
                parent_stride, child_stride);
            if (!CUDA_OK(cudaGetLastError())) return false;
        }
        return true;
    }
#endif /* ICM_HAVE_VKFFT */

    /* cuFFT path (original) */
    cufftDoubleComplex *fwd_out = (lp.cache_fft && plan->d_fft_cache[ell]) ? plan->d_fft_cache[ell] : b.spec_in;
    if (!CUFFT_OK(cufftExecD2Z(b.plan_fwd, plan->d_poly_levels[ell - 1], fwd_out))) return false;

    if (lp.cache_fft && plan->d_fft_cache[ell]) {
        if (ell < (int)plan->fft_cache_valid.size()) plan->fft_cache_valid[ell] = true;
    }

    cufftDoubleComplex *inv_in;
    if (b.lto_build_active) {
        /* LTO callback fuses multiply into C2R — pass fwd_out directly */
        inv_in = fwd_out;
    } else {
        cufftDoubleComplex *mul_out = (fwd_out != b.spec_in) ? b.spec_in : b.spec_mid;
        size_t mul_total = (size_t)parent_batch * (size_t)b.cn;
        int blocks_mul = (int)((mul_total + threads - 1) / threads);
        double inv_fft_n = 1.0 / (double)b.fft_n;
        k_pairwise_mul<<<blocks_mul, threads, 0, plan->stream_compute>>>(
            fwd_out, b.cn, mul_out, parent_batch, inv_fft_n);
        if (!CUDA_OK(cudaGetLastError())) return false;
        inv_in = mul_out;
    }

    if (!CUFFT_OK(cufftExecZ2D(b.plan_inv, inv_in, plan->d_poly_levels[ell]))) return false;

    if (parent_stride > pps) {
        size_t szp_total = (size_t)parent_batch * (size_t)parent_stride;
        int blocks_szp = (int)((szp_total + threads - 1) / threads);
        k_zero_pad<<<blocks_szp, threads, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], parent_stride, pps, parent_batch);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    if (lp.build_wrap_m > 0) {
        int cps = plan->psz[ell - 1];
        k_wrap_build<<<parent_batch, 64, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], pps, parent_batch,
            plan->d_poly_levels[ell - 1], cps, lp.build_conv,
            b.fft_n, lp.build_wrap_m,
            parent_stride, child_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

bool run_build_level_fused(GpuPlan *plan, int ell) {
    auto &lp = plan->levels[ell];
    if (!plan->opts.use_cufftdx || g_runtime_fused_max_conv_len <= 0 ||
        lp.build_conv > g_runtime_fused_max_conv_len || !is_cufftdx_supported_fft_n(lp.fft_n)) {
        return run_build_level_fft(plan, ell);
    }
    int cps = plan->psz[ell - 1];
    int pps = plan->psz[ell];
    int child_stride = plan->fft_stride[ell - 1];
    int parent_stride = plan->fft_stride[ell];
    int nparents = plan->nn[ell];
    if (nparents <= 0 || cps <= 0 || pps <= 0) return true;
    bool ok = launch_cufftdx_build_r2c_dispatch(lp.fft_n,
                                                plan->d_poly_levels[ell - 1], child_stride,
                                                plan->d_poly_levels[ell], parent_stride, nparents,
                                                1.0 / (double)lp.fft_n,
                                                plan->stream_compute);
    if (!ok) {
        ok = launch_cufftdx_build_dispatch(lp.fft_n,
                                           plan->d_poly_levels[ell - 1], child_stride,
                                           plan->d_poly_levels[ell], parent_stride, nparents,
                                           1.0 / (double)lp.fft_n,
                                           plan->stream_compute);
    }
    if (!ok) return run_build_level_fft(plan, ell);
    if (lp.build_wrap_m > 0) {
        k_wrap_build<<<nparents, 64, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], pps, nparents,
            plan->d_poly_levels[ell - 1], cps, lp.build_conv,
            lp.fft_n, lp.build_wrap_m,
            parent_stride, child_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

/* ── Single-Q prop level runners ───────────────────────────────── */

bool run_prop_level_schoolbook(GpuPlan *plan, int ell) {
    int parent_gsz = plan->psz[ell];
    int child_gsz = plan->psz[ell - 1];
    int cps = plan->psz[ell - 1];
    int parent_stride = plan->fft_stride[ell];
    int child_stride = plan->fft_stride[ell - 1];
    int nparents = plan->nn[ell];
    auto &lp = plan->levels[ell];
    int len_g = lp.g_eff;
    int len_P = lp.p_eff;
    int len_out = lp.out_needed;
    if (nparents <= 0 || len_out <= 0 || len_g <= 0 || len_P <= 0) return true;

    int conv = lp.corr_conv;
    bool use_warp_regime = (conv <= GPU_SCHOOL_WARP_MAX_CONV && nparents > 1);
    if (use_warp_regime) {
        int threads = GPU_SCHOOL_WARPS_PER_BLOCK * 32;
        int blocks = (nparents + GPU_SCHOOL_WARPS_PER_BLOCK - 1) / GPU_SCHOOL_WARPS_PER_BLOCK;
        size_t shmem = (size_t)GPU_SCHOOL_WARPS_PER_BLOCK
            * (size_t)(len_g + 2 * len_P) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_corr_pair_warp_batch<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents,
                parent_stride, child_stride, child_stride);
        } else {
            int fb_threads = GPU_THREADS_PER_BLOCK;
            size_t total = (size_t)nparents * (size_t)len_out;
            int fb_blocks = (int)((total + fb_threads - 1) / fb_threads);
            k_schoolbook_corr_pair<<<fb_blocks, fb_threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents,
                parent_stride, child_stride, child_stride);
        }
    } else {
        int threads = GPU_THREADS_PER_BLOCK;
        int blocks = nparents;
        size_t shmem = (size_t)(len_g + 2 * len_P) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_corr_pair_smem_parent<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents,
                parent_stride, child_stride, child_stride);
        } else {
            size_t total = (size_t)nparents * (size_t)len_out;
            int fb_blocks = (int)((total + threads - 1) / threads);
            k_schoolbook_corr_pair<<<fb_blocks, threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents,
                parent_stride, child_stride, child_stride);
        }
    }
    if (!CUDA_OK(cudaGetLastError())) return false;
    return true;
}

bool run_prop_level_fft(GpuPlan *plan, int ell) {
    int child_gsz = plan->psz[ell - 1];
    int nparents = plan->nn[ell];
    int parent_stride = plan->fft_stride[ell];
    int child_stride = plan->fft_stride[ell - 1];
    auto &lp = plan->levels[ell];
    auto &c = plan->corr_fft[ell];
    auto &b_fft = plan->build_fft[ell];
    int len_g = lp.g_eff;
    int len_P = lp.p_eff;
    int len_out = lp.out_needed;
    int threads = GPU_THREADS_PER_BLOCK;
#if ICM_HAVE_VKFFT
    if (c.use_vkfft) {
        /* VkFFT out-of-place propagation — no gather/scatter. */
        int fft_n = c.fft_n;
        int cn = c.cn;

        /* 1. VkFFT R2C forward: g_levels[ell] → spec_in */
        {
            double *input_ptr = plan->d_g_levels[ell];
            VkFFTLaunchParams lp_fwd = {};
            lp_fwd.buffer = (void **)&c.spec_in;
            lp_fwd.inputBuffer = (void **)&input_ptr;
            VkFFTResult res = VkFFTAppend(&c.vkfft_app_fwd, -1, &lp_fwd);
            if (res != VKFFT_SUCCESS) return false;
        }

        /* 2. Get child spectra (from cache or recompute) */
        const cufftDoubleComplex *child_spec = nullptr;
        if (plan->d_fft_cache[ell] && ell < (int)plan->fft_cache_valid.size() && plan->fft_cache_valid[ell]) {
            child_spec = plan->d_fft_cache[ell];
        }
        if (!child_spec) {
            int child_batch = plan->nn[ell - 1];
            if (b_fft.use_vkfft) {
                double *child_input = plan->d_poly_levels[ell - 1];
                VkFFTLaunchParams lp_bfwd = {};
                lp_bfwd.buffer = (void **)&b_fft.spec_in;
                lp_bfwd.inputBuffer = (void **)&child_input;
                VkFFTResult res = VkFFTAppend(&b_fft.vkfft_app_fwd, -1, &lp_bfwd);
                if (res != VKFFT_SUCCESS) return false;
            } else {
                if (!CUFFT_OK(cufftExecD2Z(b_fft.plan_fwd, plan->d_poly_levels[ell - 1], b_fft.spec_in))) return false;
            }
            child_spec = b_fft.spec_in;
        }

        /* 3. Paired correlate in frequency domain */
        size_t corr_total = (size_t)nparents * (size_t)cn;
        int blocks_corr = (int)((corr_total + threads - 1) / threads);
        double inv_fft_n_corr = 1.0 / (double)fft_n;
        k_paired_corr_freq<<<blocks_corr, threads, 0, plan->stream_compute>>>(
            c.spec_in, child_spec, cn, nparents, c.spec_mid, inv_fft_n_corr);
        if (!CUDA_OK(cudaGetLastError())) return false;

        /* 4. VkFFT C2R inverse: spec_mid → g_levels[ell-1] */
        {
            double *output_ptr = plan->d_g_levels[ell - 1];
            VkFFTLaunchParams lp_inv = {};
            lp_inv.buffer = (void **)&c.spec_mid;
            lp_inv.outputBuffer = (void **)&output_ptr;
            VkFFTResult res = VkFFTAppend(&c.vkfft_app_inv, 1, &lp_inv);
            if (res != VKFFT_SUCCESS) return false;
        }

        if (lp.corr_wrap_m > 0) {
            k_wrap_corr_pair<<<nparents, 64, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell - 1], child_gsz, nparents,
                plan->d_g_levels[ell], plan->psz[ell], len_g,
                plan->d_poly_levels[ell - 1], plan->psz[ell - 1], len_P,
                len_out,
                fft_n, lp.corr_wrap_m,
                child_stride, parent_stride, child_stride);
            if (!CUDA_OK(cudaGetLastError())) return false;
        }

        if (plan->opts.memory_strategy >= 2 && ell < (int)plan->fft_cache_valid.size()) {
            plan->fft_cache_valid[ell] = false;
        }
        return true;
    }
#endif /* ICM_HAVE_VKFFT */

    /* cuFFT path (original) */
    if (!CUFFT_OK(cufftExecD2Z(c.plan_fwd, plan->d_g_levels[ell], c.spec_in))) return false;

    const cufftDoubleComplex *child_spec = (plan->d_fft_cache[ell] && ell < (int)plan->fft_cache_valid.size() && plan->fft_cache_valid[ell]) ? plan->d_fft_cache[ell] : nullptr;
    if (!child_spec) {
        if (!CUFFT_OK(cufftExecD2Z(b_fft.plan_fwd, plan->d_poly_levels[ell - 1], b_fft.spec_in))) return false;
        child_spec = b_fft.spec_in;
    }

    size_t corr_total = (size_t)nparents * (size_t)c.cn;
    int blocks_corr = (int)((corr_total + threads - 1) / threads);
    double inv_fft_n_corr = 1.0 / (double)c.fft_n;
    k_paired_corr_freq<<<blocks_corr, threads, 0, plan->stream_compute>>>(
        c.spec_in, child_spec, c.cn, nparents, c.spec_mid, inv_fft_n_corr);
    if (!CUDA_OK(cudaGetLastError())) return false;

    if (!CUFFT_OK(cufftExecZ2D(c.plan_inv, c.spec_mid, plan->d_g_levels[ell - 1]))) return false;

    int n_children = 2 * nparents;
    if (child_stride > child_gsz) {
        size_t szp_total = (size_t)n_children * (size_t)child_stride;
        int blocks_szp = (int)((szp_total + threads - 1) / threads);
        k_zero_pad<<<blocks_szp, threads, 0, plan->stream_compute>>>(
            plan->d_g_levels[ell - 1], child_stride, child_gsz, n_children);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    if (lp.corr_wrap_m > 0) {
        k_wrap_corr_pair<<<nparents, 64, 0, plan->stream_compute>>>(
            plan->d_g_levels[ell - 1], child_gsz, nparents,
            plan->d_g_levels[ell], plan->psz[ell], len_g,
            plan->d_poly_levels[ell - 1], plan->psz[ell - 1], len_P,
            len_out,
            c.fft_n, lp.corr_wrap_m,
            child_stride, parent_stride, child_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    if (plan->opts.memory_strategy >= 2 && ell < (int)plan->fft_cache_valid.size()) {
        plan->fft_cache_valid[ell] = false;
    }
    return true;
}

bool run_prop_level_fused(GpuPlan *plan, int ell) {
    auto &lp = plan->levels[ell];
    if (!plan->opts.use_cufftdx || g_runtime_fused_max_conv_len <= 0 ||
        lp.corr_conv > g_runtime_fused_max_conv_len || !is_cufftdx_supported_fft_n(lp.fft_n)) {
        return run_prop_level_fft(plan, ell);
    }
    int parent_gsz = plan->psz[ell];
    int child_gsz = plan->psz[ell - 1];
    int cps = plan->psz[ell - 1];
    int parent_stride = plan->fft_stride[ell];
    int child_stride = plan->fft_stride[ell - 1];
    int nparents = plan->nn[ell];
    int len_g = lp.g_eff;
    int len_P = lp.p_eff;
    int len_out = lp.out_needed;
    if (nparents <= 0 || len_out <= 0 || len_g <= 0 || len_P <= 0) return true;
    bool ok = launch_cufftdx_corr_r2c_dispatch(lp.fft_n,
                                               plan->d_g_levels[ell], parent_stride, len_g,
                                               plan->d_poly_levels[ell - 1], child_stride, len_P,
                                               plan->d_g_levels[ell - 1], child_stride, len_out, nparents,
                                               1.0 / (double)lp.fft_n,
                                               plan->stream_compute);
    if (!ok) {
        ok = launch_cufftdx_corr_dispatch(lp.fft_n,
                                          plan->d_g_levels[ell], parent_stride, len_g,
                                          plan->d_poly_levels[ell - 1], child_stride, len_P,
                                          plan->d_g_levels[ell - 1], child_stride, len_out, nparents,
                                          1.0 / (double)lp.fft_n,
                                          plan->stream_compute);
    }
    if (!ok) return run_prop_level_fft(plan, ell);
    if (lp.corr_wrap_m > 0) {
        k_wrap_corr_pair<<<nparents, 64, 0, plan->stream_compute>>>(
            plan->d_g_levels[ell - 1], child_gsz, nparents,
            plan->d_g_levels[ell], parent_gsz, len_g,
            plan->d_poly_levels[ell - 1], cps, len_P,
            len_out,
            lp.fft_n, lp.corr_wrap_m,
            child_stride, parent_stride, child_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

/* ── Q-batched build/prop level runners ────────────────────────── */

bool run_build_level_schoolbook_qb(GpuPlan *plan, int ell, int qb) {
    int cps = plan->psz[ell - 1]; int pps = plan->psz[ell];
    int cs = plan->fft_stride[ell - 1]; int ps = plan->fft_stride[ell];
    int nparents_total = qb * plan->nn[ell];
    if (nparents_total <= 0 || cps <= 0 || pps <= 0) return true;
    int conv = plan->levels[ell].build_conv;
    bool use_warp_regime = (conv <= GPU_SCHOOL_WARP_MAX_CONV && nparents_total > 1);
    if (use_warp_regime) {
        int threads = GPU_SCHOOL_WARPS_PER_BLOCK * 32;
        int blocks = (nparents_total + GPU_SCHOOL_WARPS_PER_BLOCK - 1) / GPU_SCHOOL_WARPS_PER_BLOCK;
        size_t shmem = (size_t)GPU_SCHOOL_WARPS_PER_BLOCK * (size_t)(2 * cps) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_build_warp_batch<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps, plan->d_poly_levels[ell], pps, nparents_total, cs, ps);
        } else {
            int fb_threads = GPU_THREADS_PER_BLOCK;
            size_t total = (size_t)nparents_total * (size_t)pps;
            int fb_blocks = (int)((total + fb_threads - 1) / fb_threads);
            k_schoolbook_build<<<fb_blocks, fb_threads, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps, plan->d_poly_levels[ell], pps, nparents_total, cs, ps);
        }
    } else {
        int threads = GPU_THREADS_PER_BLOCK; int blocks = nparents_total;
        size_t shmem = (size_t)(2 * cps) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_build_smem_parent<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps, plan->d_poly_levels[ell], pps, nparents_total, cs, ps);
        } else {
            size_t total = (size_t)nparents_total * (size_t)pps;
            int fb_blocks = (int)((total + threads - 1) / threads);
            k_schoolbook_build<<<fb_blocks, threads, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps, plan->d_poly_levels[ell], pps, nparents_total, cs, ps);
        }
    }
    if (!CUDA_OK(cudaGetLastError())) return false;
    return true;
}

bool run_build_level_fft_qb(GpuPlan *plan, int ell, int qb) {
    int cps = plan->psz[ell - 1]; int pps = plan->psz[ell];
    int child_batch = qb * plan->nn[ell - 1];
    int parent_batch = qb * plan->nn[ell];
    int parent_stride = plan->fft_stride[ell];
    int child_stride = plan->fft_stride[ell - 1];
    auto &lp = plan->levels[ell]; auto &b = plan->build_fft[ell];
    int threads = GPU_THREADS_PER_BLOCK;

#if ICM_HAVE_VKFFT
    if (b.use_vkfft) {
        int fft_n = b.fft_n; int cn = b.cn;
        /* VkFFT out-of-place Q-batch build — no gather/scatter */
        { double *input_ptr = plan->d_poly_levels[ell - 1];
          VkFFTLaunchParams lp_fwd = {}; lp_fwd.buffer = (void **)&b.spec_in;
          lp_fwd.inputBuffer = (void **)&input_ptr;
          if (VkFFTAppend(&b.vkfft_app_fwd, -1, &lp_fwd) != VKFFT_SUCCESS) return false; }

        cufftDoubleComplex *fwd_out = (lp.cache_fft && plan->d_fft_cache[ell]) ? plan->d_fft_cache[ell] : b.spec_in;
        if (fwd_out != b.spec_in) {
            size_t copy_bytes = (size_t)child_batch * cn * sizeof(cufftDoubleComplex);
            if (!CUDA_OK(cudaMemcpyAsync(fwd_out, b.spec_in, copy_bytes,
                                          cudaMemcpyDeviceToDevice, plan->stream_compute))) return false;
        }
        if (lp.cache_fft && plan->d_fft_cache[ell]) {
            if (ell < (int)plan->fft_cache_valid.size()) plan->fft_cache_valid[ell] = true;
        }
        cufftDoubleComplex *mul_out = (fwd_out != b.spec_in) ? b.spec_in : b.spec_mid;
        size_t mul_total = (size_t)parent_batch * (size_t)cn;
        int blocks_mul = (int)((mul_total + threads - 1) / threads);
        double inv_fft_n_qb = 1.0 / (double)fft_n;
        k_pairwise_mul<<<blocks_mul, threads, 0, plan->stream_compute>>>(fwd_out, cn, mul_out, parent_batch, inv_fft_n_qb);
        if (!CUDA_OK(cudaGetLastError())) return false;

        { double *output_ptr = plan->d_poly_levels[ell];
          VkFFTLaunchParams lp_inv = {}; lp_inv.buffer = (void **)&mul_out;
          lp_inv.outputBuffer = (void **)&output_ptr;
          if (VkFFTAppend(&b.vkfft_app_inv, 1, &lp_inv) != VKFFT_SUCCESS) return false; }
        if (lp.build_wrap_m > 0) {
            k_wrap_build<<<parent_batch, 64, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell], pps, parent_batch,
                plan->d_poly_levels[ell - 1], cps, lp.build_conv, fft_n, lp.build_wrap_m, parent_stride, child_stride);
            if (!CUDA_OK(cudaGetLastError())) return false;
        }
        return true;
    }
#endif

    cufftDoubleComplex *fwd_out = (lp.cache_fft && plan->d_fft_cache[ell]) ? plan->d_fft_cache[ell] : b.spec_in;
    if (!CUFFT_OK(cufftExecD2Z(b.plan_fwd, plan->d_poly_levels[ell - 1], fwd_out))) return false;
    if (lp.cache_fft && plan->d_fft_cache[ell]) {
        if (ell < (int)plan->fft_cache_valid.size()) plan->fft_cache_valid[ell] = true;
    }
    cufftDoubleComplex *inv_in;
    if (b.lto_build_active) {
        inv_in = fwd_out;
    } else {
        cufftDoubleComplex *mul_out = (fwd_out != b.spec_in) ? b.spec_in : b.spec_mid;
        size_t mul_total = (size_t)parent_batch * (size_t)b.cn;
        int blocks_mul = (int)((mul_total + threads - 1) / threads);
        double inv_fft_n_qb = 1.0 / (double)b.fft_n;
        k_pairwise_mul<<<blocks_mul, threads, 0, plan->stream_compute>>>(fwd_out, b.cn, mul_out, parent_batch, inv_fft_n_qb);
        if (!CUDA_OK(cudaGetLastError())) return false;
        inv_in = mul_out;
    }
    if (!CUFFT_OK(cufftExecZ2D(b.plan_inv, inv_in, plan->d_poly_levels[ell]))) return false;
    if (parent_stride > pps) {
        size_t szp_total = (size_t)parent_batch * (size_t)parent_stride;
        int blocks_szp = (int)((szp_total + threads - 1) / threads);
        k_zero_pad<<<blocks_szp, threads, 0, plan->stream_compute>>>(plan->d_poly_levels[ell], parent_stride, pps, parent_batch);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    if (lp.build_wrap_m > 0) {
        k_wrap_build<<<parent_batch, 64, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], pps, parent_batch,
            plan->d_poly_levels[ell - 1], cps, lp.build_conv, b.fft_n, lp.build_wrap_m, parent_stride, child_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

bool run_build_level_fused_qb(GpuPlan *plan, int ell, int qb) {
    auto &lp = plan->levels[ell];
    int nparents_total = qb * plan->nn[ell];
    if (!plan->opts.use_cufftdx || g_runtime_fused_max_conv_len <= 0 ||
        lp.build_conv > g_runtime_fused_max_conv_len || !is_cufftdx_supported_fft_n(lp.fft_n)) {
        return run_build_level_fft_qb(plan, ell, qb);
    }
    int cps = plan->psz[ell - 1]; int pps = plan->psz[ell];
    int cs = plan->fft_stride[ell - 1]; int ps = plan->fft_stride[ell];
    if (nparents_total <= 0 || cps <= 0 || pps <= 0) return true;
    bool ok = launch_cufftdx_build_r2c_dispatch(lp.fft_n, plan->d_poly_levels[ell - 1], cs,
                                                plan->d_poly_levels[ell], ps, nparents_total,
                                                1.0 / (double)lp.fft_n, plan->stream_compute);
    if (!ok) ok = launch_cufftdx_build_dispatch(lp.fft_n, plan->d_poly_levels[ell - 1], cs,
                                                 plan->d_poly_levels[ell], ps, nparents_total,
                                                 1.0 / (double)lp.fft_n, plan->stream_compute);
    if (!ok) return run_build_level_fft_qb(plan, ell, qb);
    if (lp.build_wrap_m > 0) {
        k_wrap_build<<<nparents_total, 64, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], pps, nparents_total,
            plan->d_poly_levels[ell - 1], cps, lp.build_conv, lp.fft_n, lp.build_wrap_m, ps, cs);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

bool run_prop_level_schoolbook_qb(GpuPlan *plan, int ell, int qb) {
    int parent_gsz = plan->psz[ell]; int child_gsz = plan->psz[ell - 1];
    int cps = plan->psz[ell - 1];
    int ps = plan->fft_stride[ell]; int cs = plan->fft_stride[ell - 1];
    int nparents_total = qb * plan->nn[ell];
    auto &lp = plan->levels[ell];
    int len_g = lp.g_eff; int len_P = lp.p_eff; int len_out = lp.out_needed;
    if (nparents_total <= 0 || len_out <= 0 || len_g <= 0 || len_P <= 0) return true;
    int conv = lp.corr_conv;
    bool use_warp_regime = (conv <= GPU_SCHOOL_WARP_MAX_CONV && nparents_total > 1);
    if (use_warp_regime) {
        int threads = GPU_SCHOOL_WARPS_PER_BLOCK * 32;
        int blocks = (nparents_total + GPU_SCHOOL_WARPS_PER_BLOCK - 1) / GPU_SCHOOL_WARPS_PER_BLOCK;
        size_t shmem = (size_t)GPU_SCHOOL_WARPS_PER_BLOCK * (size_t)(len_g + 2 * len_P) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_corr_pair_warp_batch<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents_total, ps, cs, cs);
        } else {
            int fb_threads = GPU_THREADS_PER_BLOCK;
            size_t total = (size_t)nparents_total * (size_t)len_out;
            int fb_blocks = (int)((total + fb_threads - 1) / fb_threads);
            k_schoolbook_corr_pair<<<fb_blocks, fb_threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents_total, ps, cs, cs);
        }
    } else {
        int threads = GPU_THREADS_PER_BLOCK; int blocks = nparents_total;
        size_t shmem = (size_t)(len_g + 2 * len_P) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_corr_pair_smem_parent<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents_total, ps, cs, cs);
        } else {
            size_t total = (size_t)nparents_total * (size_t)len_out;
            int fb_blocks = (int)((total + threads - 1) / threads);
            k_schoolbook_corr_pair<<<fb_blocks, threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents_total, ps, cs, cs);
        }
    }
    if (!CUDA_OK(cudaGetLastError())) return false;
    return true;
}

bool run_prop_level_fft_qb(GpuPlan *plan, int ell, int qb) {
    int child_gsz = plan->psz[ell - 1];
    int nparents_total = qb * plan->nn[ell];
    int ps = plan->fft_stride[ell]; int cs = plan->fft_stride[ell - 1];
    auto &lp = plan->levels[ell]; auto &c = plan->corr_fft[ell];
    auto &b_fft = plan->build_fft[ell];
    int len_g = lp.g_eff; int len_P = lp.p_eff; int len_out = lp.out_needed;
    int threads = GPU_THREADS_PER_BLOCK;

#if ICM_HAVE_VKFFT
    if (c.use_vkfft) {
        int fft_n = c.fft_n; int cn = c.cn;
        int n_children = 2 * nparents_total;
        /* VkFFT out-of-place Q-batch propagation — no gather/scatter */
        { double *input_ptr = plan->d_g_levels[ell];
          VkFFTLaunchParams lp_fwd = {}; lp_fwd.buffer = (void **)&c.spec_in;
          lp_fwd.inputBuffer = (void **)&input_ptr;
          if (VkFFTAppend(&c.vkfft_app_fwd, -1, &lp_fwd) != VKFFT_SUCCESS) return false; }

        const cufftDoubleComplex *child_spec = (plan->d_fft_cache[ell] && ell < (int)plan->fft_cache_valid.size() && plan->fft_cache_valid[ell]) ? plan->d_fft_cache[ell] : nullptr;
        if (!child_spec) {
            int child_batch = qb * plan->nn[ell - 1];
            if (b_fft.use_vkfft) {
                double *child_input = plan->d_poly_levels[ell - 1];
                VkFFTLaunchParams lp_bfwd = {}; lp_bfwd.buffer = (void **)&b_fft.spec_in;
                lp_bfwd.inputBuffer = (void **)&child_input;
                if (VkFFTAppend(&b_fft.vkfft_app_fwd, -1, &lp_bfwd) != VKFFT_SUCCESS) return false;
            } else {
                if (!CUFFT_OK(cufftExecD2Z(b_fft.plan_fwd, plan->d_poly_levels[ell - 1], b_fft.spec_in))) return false;
            }
            child_spec = b_fft.spec_in;
        }
        size_t corr_total = (size_t)nparents_total * (size_t)cn;
        int blocks_corr = (int)((corr_total + threads - 1) / threads);
        double inv_fft_n_cqb = 1.0 / (double)fft_n;
        k_paired_corr_freq<<<blocks_corr, threads, 0, plan->stream_compute>>>(c.spec_in, child_spec, cn, nparents_total, c.spec_mid, inv_fft_n_cqb);
        if (!CUDA_OK(cudaGetLastError())) return false;

        { double *output_ptr = plan->d_g_levels[ell - 1];
          VkFFTLaunchParams lp_inv = {}; lp_inv.buffer = (void **)&c.spec_mid;
          lp_inv.outputBuffer = (void **)&output_ptr;
          if (VkFFTAppend(&c.vkfft_app_inv, 1, &lp_inv) != VKFFT_SUCCESS) return false; }
        if (lp.corr_wrap_m > 0) {
            k_wrap_corr_pair<<<nparents_total, 64, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell - 1], child_gsz, nparents_total,
                plan->d_g_levels[ell], plan->psz[ell], len_g,
                plan->d_poly_levels[ell - 1], plan->psz[ell - 1], len_P, len_out,
                fft_n, lp.corr_wrap_m, cs, ps, cs);
            if (!CUDA_OK(cudaGetLastError())) return false;
        }
        if (plan->opts.memory_strategy >= 2 && ell < (int)plan->fft_cache_valid.size()) {
            plan->fft_cache_valid[ell] = false;
        }
        return true;
    }
#endif

    if (!CUFFT_OK(cufftExecD2Z(c.plan_fwd, plan->d_g_levels[ell], c.spec_in))) return false;
    const cufftDoubleComplex *child_spec = (plan->d_fft_cache[ell] && ell < (int)plan->fft_cache_valid.size() && plan->fft_cache_valid[ell]) ? plan->d_fft_cache[ell] : nullptr;
    if (!child_spec) {
        if (!CUFFT_OK(cufftExecD2Z(b_fft.plan_fwd, plan->d_poly_levels[ell - 1], b_fft.spec_in))) return false;
        child_spec = b_fft.spec_in;
    }
    size_t corr_total = (size_t)nparents_total * (size_t)c.cn;
    int blocks_corr = (int)((corr_total + threads - 1) / threads);
    double inv_fft_n_cqb = 1.0 / (double)c.fft_n;
    k_paired_corr_freq<<<blocks_corr, threads, 0, plan->stream_compute>>>(c.spec_in, child_spec, c.cn, nparents_total, c.spec_mid, inv_fft_n_cqb);
    if (!CUDA_OK(cudaGetLastError())) return false;
    if (!CUFFT_OK(cufftExecZ2D(c.plan_inv, c.spec_mid, plan->d_g_levels[ell - 1]))) return false;
    int n_children = 2 * nparents_total;
    if (cs > child_gsz) {
        size_t szp_total = (size_t)n_children * (size_t)cs;
        int blocks_szp = (int)((szp_total + threads - 1) / threads);
        k_zero_pad<<<blocks_szp, threads, 0, plan->stream_compute>>>(plan->d_g_levels[ell - 1], cs, child_gsz, n_children);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    if (lp.corr_wrap_m > 0) {
        k_wrap_corr_pair<<<nparents_total, 64, 0, plan->stream_compute>>>(
            plan->d_g_levels[ell - 1], child_gsz, nparents_total,
            plan->d_g_levels[ell], plan->psz[ell], len_g,
            plan->d_poly_levels[ell - 1], plan->psz[ell - 1], len_P, len_out,
            c.fft_n, lp.corr_wrap_m, cs, ps, cs);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    if (plan->opts.memory_strategy >= 2 && ell < (int)plan->fft_cache_valid.size()) {
        plan->fft_cache_valid[ell] = false;
    }
    return true;
}

bool run_prop_level_fused_qb(GpuPlan *plan, int ell, int qb) {
    auto &lp = plan->levels[ell];
    int nparents_total = qb * plan->nn[ell];
    if (!plan->opts.use_cufftdx || g_runtime_fused_max_conv_len <= 0 ||
        lp.corr_conv > g_runtime_fused_max_conv_len || !is_cufftdx_supported_fft_n(lp.fft_n)) {
        return run_prop_level_fft_qb(plan, ell, qb);
    }
    int parent_gsz = plan->psz[ell]; int child_gsz = plan->psz[ell - 1];
    int cps = plan->psz[ell - 1];
    int ps = plan->fft_stride[ell]; int cs = plan->fft_stride[ell - 1];
    int len_g = lp.g_eff; int len_P = lp.p_eff; int len_out = lp.out_needed;
    if (nparents_total <= 0 || len_out <= 0 || len_g <= 0 || len_P <= 0) return true;
    bool ok = launch_cufftdx_corr_r2c_dispatch(lp.fft_n, plan->d_g_levels[ell], ps, len_g,
                                               plan->d_poly_levels[ell - 1], cs, len_P,
                                               plan->d_g_levels[ell - 1], cs, len_out, nparents_total,
                                               1.0 / (double)lp.fft_n, plan->stream_compute);
    if (!ok) ok = launch_cufftdx_corr_dispatch(lp.fft_n, plan->d_g_levels[ell], ps, len_g,
                                                plan->d_poly_levels[ell - 1], cs, len_P,
                                                plan->d_g_levels[ell - 1], cs, len_out, nparents_total,
                                                1.0 / (double)lp.fft_n, plan->stream_compute);
    if (!ok) return run_prop_level_fft_qb(plan, ell, qb);
    if (lp.corr_wrap_m > 0) {
        k_wrap_corr_pair<<<nparents_total, 64, 0, plan->stream_compute>>>(
            plan->d_g_levels[ell - 1], child_gsz, nparents_total,
            plan->d_g_levels[ell], parent_gsz, len_g,
            plan->d_poly_levels[ell - 1], cps, len_P, len_out,
            lp.fft_n, lp.corr_wrap_m, cs, ps, cs);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

/* ── run_hybrid_batched_q / run_hybrid_single_q ────────────────── */
/* These are large functions copied verbatim from the original.
 * They orchestrate kernel launches using the level runners above. */

bool run_hybrid_batched_q(GpuPlan *plan, const QP *pts, int qb) {
    int threads = GPU_THREADS_PER_BLOCK;
    int blocks_n = (plan->n + threads - 1) / threads;
    (void)blocks_n;

    /* Pin S_sorted in L2 persistent cache — read every Q-batch by k_compute_a */
    {
        cudaAccessPolicyWindow window = {};
        window.base_ptr = (void *)plan->d_S_sorted;
        window.num_bytes = (size_t)plan->n * sizeof(double);
        window.hitRatio = 1.0f;
        window.hitProp = cudaAccessPropertyPersisting;
        window.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAccessPolicyWindow(plan->stream_compute, window);
    }

    double *h_a_ptrs[Q_BATCH_MAX];
    double h_logv[Q_BATCH_MAX];
    for (int qi = 0; qi < qb; ++qi) {
        h_a_ptrs[qi] = plan->d_a_qbatch[qi];
        h_logv[qi] = pts[qi].logv;
    }
    double **d_a_ptrs = plan->d_qb_a_ptrs;
    if (!CUDA_OK(cudaMemcpyAsync(d_a_ptrs, h_a_ptrs, (size_t)qb * sizeof(double *),
                                 cudaMemcpyHostToDevice, plan->stream_compute))) return false;
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_qb_inv_vs, h_logv, (size_t)qb * sizeof(double),
                                 cudaMemcpyHostToDevice, plan->stream_compute))) return false;
    {
        int total_threads = qb * plan->n;
        int blocks_a = (total_threads + threads - 1) / threads;
        k_compute_a_qbatch<<<blocks_a, threads, 0, plan->stream_compute>>>(
            plan->d_S_sorted, d_a_ptrs, plan->d_qb_inv_vs, plan->n, qb);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    size_t leaf_stride = (size_t)plan->nn[0] * (size_t)plan->fft_stride[0];
    if (plan->B <= 1) {
        int total_leaves = qb * plan->N_tree;
        int blocks_leaves = (total_leaves + GPU_THREADS_PER_BLOCK - 1) / GPU_THREADS_PER_BLOCK;
        k_set_leaves_b1_qbatch<<<blocks_leaves, GPU_THREADS_PER_BLOCK, 0, plan->stream_compute>>>(
            (double * const *)d_a_ptrs, plan->n, plan->N_tree,
            plan->fft_stride[0], qb, plan->d_poly_levels[0], leaf_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    } else {
        size_t bp_stride = (size_t)plan->N_tree * (size_t)(plan->B + 1);
        int threads_block = ((plan->B + 32) / 32) * 32;
        if (threads_block > GPU_THREADS_PER_BLOCK) threads_block = GPU_THREADS_PER_BLOCK;
        size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);
        k_block_build_qbatch<<<qb * plan->N_tree, threads_block, shmem_block, plan->stream_compute>>>(
            (const double * const *)d_a_ptrs, plan->n, plan->B,
            plan->nblocks, plan->N_tree, qb,
            plan->fft_stride[0], plan->d_poly_levels[0], leaf_stride,
            plan->d_block_prods_qbatch, bp_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    for (int ell = 1; ell < plan->L - 1; ++ell) {
        auto &lp = plan->levels[ell];
        if (!lp.use_fft) { if (!run_build_level_schoolbook_qb(plan, ell, qb)) return false; }
        else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_build_level_fused_qb(plan, ell, qb)) return false; }
        else { if (!run_build_level_fft_qb(plan, ell, qb)) return false; }
    }

    int top = plan->L - 1;
    int root_gsz = plan->fft_stride[top];
    size_t g_root_stride = (size_t)plan->nn[top] * (size_t)root_gsz;
    int root_total = qb * root_gsz;
    int blocks_root = (root_total + threads - 1) / threads;
    k_set_root_g_qbatch<<<blocks_root, threads, 0, plan->stream_compute>>>(
        plan->d_g_levels[top], root_gsz, plan->d_payout, plan->k, qb, g_root_stride);
    if (!CUDA_OK(cudaGetLastError())) return false;

    for (int ell = top; ell >= 1; --ell) {
        auto &lp = plan->levels[ell];
        if (!lp.use_fft) { if (!run_prop_level_schoolbook_qb(plan, ell, qb)) return false; }
        else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_prop_level_fused_qb(plan, ell, qb)) return false; }
        else { if (!run_prop_level_fft_qb(plan, ell, qb)) return false; }
    }

    size_t leaf_g_stride = (size_t)plan->nn[0] * (size_t)plan->fft_stride[0];
    size_t inner_stride = (size_t)plan->n;
    if (plan->B <= 1) {
        int total_extract = qb * plan->n;
        int blocks_extract = (total_extract + GPU_THREADS_PER_BLOCK - 1) / GPU_THREADS_PER_BLOCK;
        k_leaf_extract_b1_qbatch<<<blocks_extract, GPU_THREADS_PER_BLOCK, 0, plan->stream_compute>>>(
            plan->n, plan->d_g_levels[0], plan->fft_stride[0], qb,
            leaf_g_stride, plan->d_inner_qbatch, inner_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    } else if (plan->d_active_mask) {
        size_t bp_stride_local = (size_t)plan->N_tree * (size_t)(plan->B + 1);
        int threads_leaf = plan->B;
        if (threads_leaf > 1024) threads_leaf = 1024;
        k_leaf_extract_qbatch_masked<<<qb * plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
            (const double * const *)d_a_ptrs, plan->n, plan->B, plan->nblocks,
            plan->d_block_prods_qbatch, bp_stride_local,
            plan->d_g_levels[0], plan->fft_stride[0], leaf_g_stride,
            plan->g_needed[0], plan->k, qb,
            plan->d_inner_qbatch, inner_stride, plan->d_active_mask);
        if (!CUDA_OK(cudaGetLastError())) return false;
    } else {
        size_t bp_stride_local = (size_t)plan->N_tree * (size_t)(plan->B + 1);
        int threads_leaf = plan->B;
        if (threads_leaf > 1024) threads_leaf = 1024;
        k_leaf_extract_qbatch<<<qb * plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
            (const double * const *)d_a_ptrs, plan->n, plan->B, plan->nblocks,
            plan->d_block_prods_qbatch, bp_stride_local,
            plan->d_g_levels[0], plan->fft_stride[0], leaf_g_stride,
            plan->g_needed[0], plan->k, qb,
            plan->d_inner_qbatch, inner_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    double h_weights[Q_BATCH_MAX];
    double h_inv_vs[Q_BATCH_MAX];
    for (int qi = 0; qi < qb; ++qi) {
        h_weights[qi] = pts[qi].w;
        h_inv_vs[qi] = exp(-pts[qi].logv);
    }
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_qb_weights, h_weights, (size_t)qb * sizeof(double),
                                 cudaMemcpyHostToDevice, plan->stream_compute))) return false;
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_qb_inv_vs, h_inv_vs, (size_t)qb * sizeof(double),
                                 cudaMemcpyHostToDevice, plan->stream_compute))) return false;

    int accum_total = qb * plan->n;
    int blocks_accum = (accum_total + threads - 1) / threads;
    if (plan->d_active_mask) {
        k_accumulate_equity_qbatch_masked<<<blocks_accum, threads, 0, plan->stream_compute>>>(
            plan->d_inner_qbatch, inner_stride, (const double * const *)d_a_ptrs,
            plan->d_S_sorted, plan->d_sort_perm, plan->n, plan->d_qb_weights, plan->d_qb_inv_vs,
            qb, plan->d_equity, plan->d_active_mask);
    } else {
        k_accumulate_equity_qbatch<<<blocks_accum, threads, 0, plan->stream_compute>>>(
            plan->d_inner_qbatch, inner_stride, (const double * const *)d_a_ptrs,
            plan->d_S_sorted, plan->d_sort_perm, plan->n, plan->d_qb_weights, plan->d_qb_inv_vs,
            qb, plan->d_equity);
    }
    if (!CUDA_OK(cudaGetLastError())) return false;
    return true;
}

bool run_hybrid_single_q(GpuPlan *plan, int a_buf_idx,
                         double logv, double w,
                         bool skip_compute_a, bool fast_mode,
                         double *block_ns, double *tree_build_ns,
                         double *tree_prop_cached_ns,
                         double *tree_prop_recomp_ns,
                         double *leaf_ns, double *accum_ns) {
    int threads = GPU_THREADS_PER_BLOCK;
    int blocks_n = (plan->n + threads - 1) / threads;
    int curr = a_buf_idx;

    if (plan->opts.enable_graphs && plan->graph_ready[curr]) {
        double scale = w * exp(-logv);
        if (!CUDA_OK(cudaMemcpyAsync(plan->d_graph_logv[curr], &logv, sizeof(double),
                                     cudaMemcpyHostToDevice, plan->stream_compute))) return false;
        if (!CUDA_OK(cudaMemcpyAsync(plan->d_graph_scale[curr], &scale, sizeof(double),
                                     cudaMemcpyHostToDevice, plan->stream_compute))) return false;
        if (!CUDA_OK(cudaGraphLaunch(plan->graph_exec[curr], plan->stream_compute))) return false;
        return true;
    }

    if (fast_mode) {
        if (!skip_compute_a) {
            k_compute_a<<<blocks_n, threads, 0, plan->stream_compute>>>(
                plan->d_S_sorted, plan->d_a_sorted[curr], plan->n, logv);
        }
        if (plan->B <= 1) {
            int bl = (plan->N_tree + GPU_THREADS_PER_BLOCK - 1) / GPU_THREADS_PER_BLOCK;
            k_set_leaves_b1<<<bl, GPU_THREADS_PER_BLOCK, 0, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->N_tree, plan->fft_stride[0], plan->d_poly_levels[0]);
        } else {
            int threads_block = ((plan->B + 32) / 32) * 32;
            if (threads_block > GPU_THREADS_PER_BLOCK) threads_block = GPU_THREADS_PER_BLOCK;
            size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);
            k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->B,
                plan->nblocks, plan->N_tree, plan->fft_stride[0], plan->d_poly_levels[0], plan->d_block_prods);
        }
        for (int ell = 1; ell < plan->L - 1; ++ell) {
            auto &lp = plan->levels[ell];
            if (!lp.use_fft) { if (!run_build_level_schoolbook(plan, ell)) return false; }
            else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_build_level_fused(plan, ell)) return false; }
            else { if (!run_build_level_fft(plan, ell)) return false; }
        }
        int top = plan->L - 1;
        int root_gsz = plan->fft_stride[top];
        int blocks_root = (root_gsz + threads - 1) / threads;
        k_set_root_g<<<blocks_root, threads, 0, plan->stream_compute>>>(
            plan->d_g_levels[top], root_gsz, plan->d_payout, plan->k);
        for (int ell = top; ell >= 1; --ell) {
            auto &lp = plan->levels[ell];
            if (!lp.use_fft) { if (!run_prop_level_schoolbook(plan, ell)) return false; }
            else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_prop_level_fused(plan, ell)) return false; }
            else { if (!run_prop_level_fft(plan, ell)) return false; }
        }
        if (plan->B <= 1) {
            int bl = (plan->n + GPU_THREADS_PER_BLOCK - 1) / GPU_THREADS_PER_BLOCK;
            k_leaf_extract_b1<<<bl, GPU_THREADS_PER_BLOCK, 0, plan->stream_compute>>>(
                plan->n, plan->d_g_levels[0], plan->fft_stride[0], plan->d_inner_sorted);
        } else {
            int threads_leaf = plan->B;
            if (threads_leaf > 1024) threads_leaf = 1024;
            k_leaf_extract<<<plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->B, plan->nblocks,
                plan->d_block_prods, plan->d_g_levels[0], plan->fft_stride[0],
                plan->g_needed[0], plan->k, plan->d_inner_sorted);
        }
        double inv_v = exp(-logv);
        k_accumulate_equity<<<blocks_n, threads, 0, plan->stream_compute>>>(
            plan->d_inner_sorted, plan->d_a_sorted[curr], plan->d_S_sorted,
            plan->d_sort_perm, plan->n, w, inv_v, plan->d_equity);
        return true;
    }

    /* Instrumented path */
    if (!skip_compute_a) {
        k_compute_a<<<blocks_n, threads, 0, plan->stream_compute>>>(
            plan->d_S_sorted, plan->d_a_sorted[curr], plan->n, logv);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    double t0 = now_ns_host();
    if (plan->B <= 1) {
        int bl = (plan->N_tree + GPU_THREADS_PER_BLOCK - 1) / GPU_THREADS_PER_BLOCK;
        k_set_leaves_b1<<<bl, GPU_THREADS_PER_BLOCK, 0, plan->stream_compute>>>(
            plan->d_a_sorted[curr], plan->n, plan->N_tree, plan->fft_stride[0], plan->d_poly_levels[0]);
        if (!CUDA_OK(cudaGetLastError())) return false;
    } else {
        int threads_block = ((plan->B + 32) / 32) * 32;
        if (threads_block > GPU_THREADS_PER_BLOCK) threads_block = GPU_THREADS_PER_BLOCK;
        size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);
        k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_compute>>>(
            plan->d_a_sorted[curr], plan->n, plan->B,
            plan->nblocks, plan->N_tree, plan->fft_stride[0], plan->d_poly_levels[0], plan->d_block_prods);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
    *block_ns += (now_ns_host() - t0);

    for (int ell = 1; ell < plan->L - 1; ++ell) {
        auto &lp = plan->levels[ell];
        t0 = now_ns_host();
        bool ok = false;
        if (!lp.use_fft) ok = run_build_level_schoolbook(plan, ell);
        else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) ok = run_build_level_fused(plan, ell);
        else ok = run_build_level_fft(plan, ell);
        if (!ok) return false;
        if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
        *tree_build_ns += (now_ns_host() - t0);
    }

    int top = plan->L - 1;
    int root_gsz = plan->fft_stride[top];
    int blocks_root = (root_gsz + threads - 1) / threads;
    k_set_root_g<<<blocks_root, threads, 0, plan->stream_compute>>>(
        plan->d_g_levels[top], root_gsz, plan->d_payout, plan->k);
    if (!CUDA_OK(cudaGetLastError())) return false;

    for (int ell = top; ell >= 1; --ell) {
        auto &lp = plan->levels[ell];
        t0 = now_ns_host();
        bool ok = false;
        if (!lp.use_fft) ok = run_prop_level_schoolbook(plan, ell);
        else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) ok = run_prop_level_fused(plan, ell);
        else ok = run_prop_level_fft(plan, ell);
        if (!ok) return false;
        if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
        double dt = now_ns_host() - t0;
        if (lp.use_fft && lp.cache_fft) *tree_prop_cached_ns += dt;
        else if (lp.use_fft) *tree_prop_recomp_ns += dt;
        else *tree_prop_cached_ns += dt;
    }

    t0 = now_ns_host();
    if (plan->B <= 1) {
        int bl = (plan->n + GPU_THREADS_PER_BLOCK - 1) / GPU_THREADS_PER_BLOCK;
        k_leaf_extract_b1<<<bl, GPU_THREADS_PER_BLOCK, 0, plan->stream_compute>>>(
            plan->n, plan->d_g_levels[0], plan->fft_stride[0], plan->d_inner_sorted);
        if (!CUDA_OK(cudaGetLastError())) return false;
    } else {
        int threads_leaf = plan->B;
        if (threads_leaf > 1024) threads_leaf = 1024;
        k_leaf_extract<<<plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
            plan->d_a_sorted[curr], plan->n, plan->B, plan->nblocks,
            plan->d_block_prods, plan->d_g_levels[0], plan->fft_stride[0],
            plan->g_needed[0], plan->k, plan->d_inner_sorted);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
    *leaf_ns += (now_ns_host() - t0);

    t0 = now_ns_host();
    double inv_v = exp(-logv);
    k_accumulate_equity<<<blocks_n, threads, 0, plan->stream_compute>>>(
        plan->d_inner_sorted, plan->d_a_sorted[curr], plan->d_S_sorted,
        plan->d_sort_perm, plan->n, w, inv_v, plan->d_equity);
    if (!CUDA_OK(cudaGetLastError())) return false;
    if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
    *accum_ns += (now_ns_host() - t0);
    return true;
}

/* ── create_graph_stub ─────────────────────────────────────────── */

bool create_graph_stub(GpuPlan *plan) {
    if (!plan->opts.enable_graphs) return true;
    int threads = GPU_THREADS_PER_BLOCK;
    int blocks_n = (plan->n + threads - 1) / threads;
    int top = plan->L - 1;
    int root_gsz = plan->fft_stride[top];
    int blocks_root = (root_gsz + threads - 1) / threads;
    int threads_leaf = plan->B;
    if (threads_leaf > 1024) threads_leaf = 1024;
    int threads_block = ((plan->B + 32) / 32) * 32;
    if (threads_block > GPU_THREADS_PER_BLOCK) threads_block = GPU_THREADS_PER_BLOCK;
    size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);

    for (int curr = 0; curr < 2; ++curr) {
        if (plan->graph_ready[curr]) continue;
        if (!CUDA_OK(cudaMemsetAsync(plan->d_graph_logv[curr], 0, sizeof(double), plan->stream_compute))) return false;
        if (!CUDA_OK(cudaMemsetAsync(plan->d_graph_scale[curr], 0, sizeof(double), plan->stream_compute))) return false;
        if (!CUDA_OK(cudaStreamBeginCapture(plan->stream_compute, cudaStreamCaptureModeGlobal))) return false;

        k_compute_a_from_ptr<<<blocks_n, threads, 0, plan->stream_compute>>>(
            plan->d_S_sorted, plan->d_a_sorted[curr], plan->n, plan->d_graph_logv[curr]);
        if (!CUDA_OK(cudaGetLastError())) return false;

        if (plan->B <= 1) {
            int bl = (plan->N_tree + GPU_THREADS_PER_BLOCK - 1) / GPU_THREADS_PER_BLOCK;
            k_set_leaves_b1<<<bl, GPU_THREADS_PER_BLOCK, 0, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->N_tree, plan->fft_stride[0], plan->d_poly_levels[0]);
        } else {
            k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->B,
                plan->nblocks, plan->N_tree, plan->fft_stride[0], plan->d_poly_levels[0], plan->d_block_prods);
        }
        if (!CUDA_OK(cudaGetLastError())) return false;

        for (int ell = 1; ell < plan->L - 1; ++ell) {
            auto &lp = plan->levels[ell];
            bool ok = false;
            if (!lp.use_fft) ok = run_build_level_schoolbook(plan, ell);
            else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) ok = run_build_level_fused(plan, ell);
            else ok = run_build_level_fft(plan, ell);
            if (!ok) return false;
        }

        k_set_root_g<<<blocks_root, threads, 0, plan->stream_compute>>>(
            plan->d_g_levels[top], root_gsz, plan->d_payout, plan->k);
        if (!CUDA_OK(cudaGetLastError())) return false;

        for (int ell = top; ell >= 1; --ell) {
            auto &lp = plan->levels[ell];
            bool ok = false;
            if (!lp.use_fft) ok = run_prop_level_schoolbook(plan, ell);
            else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) ok = run_prop_level_fused(plan, ell);
            else ok = run_prop_level_fft(plan, ell);
            if (!ok) return false;
        }

        if (plan->B <= 1) {
            int bl = (plan->n + GPU_THREADS_PER_BLOCK - 1) / GPU_THREADS_PER_BLOCK;
            k_leaf_extract_b1<<<bl, GPU_THREADS_PER_BLOCK, 0, plan->stream_compute>>>(
                plan->n, plan->d_g_levels[0], plan->fft_stride[0], plan->d_inner_sorted);
        } else {
            k_leaf_extract<<<plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->B, plan->nblocks,
                plan->d_block_prods, plan->d_g_levels[0], plan->fft_stride[0],
                plan->g_needed[0], plan->k, plan->d_inner_sorted);
        }
        if (!CUDA_OK(cudaGetLastError())) return false;

        k_accumulate_equity_scaled<<<blocks_n, threads, 0, plan->stream_compute>>>(
            plan->d_inner_sorted, plan->d_a_sorted[curr], plan->d_S_sorted,
            plan->d_sort_perm, plan->n, plan->d_graph_scale[curr], plan->d_equity);
        if (!CUDA_OK(cudaGetLastError())) return false;

        if (!CUDA_OK(cudaStreamEndCapture(plan->stream_compute, &plan->graph[curr]))) return false;
        if (!CUDA_OK(cudaGraphInstantiate(&plan->graph_exec[curr], plan->graph[curr], nullptr, nullptr, 0))) return false;
        plan->graph_ready[curr] = true;
    }
    return true;
}

}  // namespace icm_gpu_detail
