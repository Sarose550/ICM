/*
 * gpu_memory.cu — device memory: the arena, the CUDA memory pool, and the
 * per-level buffer layout.
 *
 * Everything that ALLOCATES, as opposed to everything that decides
 * (gpu_cost_model.cu) or executes (gpu_exec.cu). Covers VRAM accounting, the
 * stream-ordered memory pool, per-level FFT buffer allocation, the padded
 * identity table, and the single arena allocation with its q_batch retry loop.
 *
 * Split out of gpu_plan.cu unchanged. Treat this file with more care than the
 * other two extractions: this is where the OOM bugs lived and where the arena
 * retry and pool initialisation were debugged on real B200 hardware. Its
 * failure mode is a runtime crash, not a compile error.
 */

#include "gpu_internal.h"

namespace icm_gpu_detail {

void update_vram_alloc(GpuPlan *plan, size_t bytes) {
    plan->current_vram_bytes += bytes;
    if (plan->current_vram_bytes > plan->peak_vram_bytes) {
        plan->peak_vram_bytes = plan->current_vram_bytes;
    }
}

bool alloc_device(GpuPlan *plan, void **ptr, size_t bytes, cudaStream_t stream) {
    if (bytes == 0) {
        *ptr = nullptr;
        return true;
    }
    if (plan->use_async_pool) {
        if (!CUDA_OK(cudaMallocAsync(ptr, bytes, stream))) return false;
    } else {
        if (!CUDA_OK(cudaMalloc(ptr, bytes))) return false;
    }
    update_vram_alloc(plan, bytes);
    return true;
}

void free_device(GpuPlan *plan, void *ptr, cudaStream_t stream) {
    if (!ptr) return;
    if (plan->use_async_pool) {
        cudaFreeAsync(ptr, stream);
    } else {
        cudaFree(ptr);
    }
}

/* ── Smooth table ──────────────────────────────────────────────── */

/* ── allocate_plan_device_memory ───────────────────────────────── */

static bool maybe_init_mem_pool(GpuPlan *plan) {
    /* memory_strategy=1 ("full") opts out of pooling entirely.
     * All other strategies (0=auto, 2=pool, 3=selective recompute)
     * use the CUDA stream-ordered memory pool by default, matching
     * PyTorch/RAPIDS RMM practice for long-lived processes with many
     * varying-size allocations. */
    if (plan->opts.memory_strategy == 1) {
        plan->use_async_pool = false;
        return true;
    }
    cudaMemPool_t pool = nullptr;
    if (!CUDA_OK(cudaDeviceGetDefaultMemPool(&pool, g_cuda_device))) return false;
    /* Never auto-release: individual arenas in this workload run up to
     * ~160GB, far bigger than any modest fraction of VRAM would retain,
     * so a bounded threshold provides little protection against the
     * fragmentation this pool exists to prevent. Unlike an unconditional
     * "never release" with no way out, icm_gpu_release_pooled_memory()
     * gives callers an explicit way to reclaim pooled memory when they
     * actually need it, so defaulting to max retention here is safe. */
    uint64_t threshold = UINT64_MAX;
    if (!CUDA_OK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold))) return false;
    plan->mem_pool = pool;
    plan->use_async_pool = true;
    return true;
}

static bool allocate_level_buffers(GpuPlan *plan, int ell, const std::vector<int> &fft_sizes) {
    (void)fft_sizes;
    if (ell <= 0 || ell >= plan->L) return true;
    auto &lp = plan->levels[ell];
    if (!lp.use_fft) return true;

    int fft_n = lp.fft_n;
    int cn = lp.cn;
    int child_batch = plan->nn[ell - 1];
    int parent_batch = plan->nn[ell];
    int qb = plan->q_batch;

    auto &b = plan->build_fft[ell];
    b.fft_n = fft_n;
    b.cn = cn;
    b.batch_fwd = qb * child_batch;
    b.batch_inv = qb * parent_batch;
    b.real_in = nullptr;
    b.spec_in = plan->shared_build_work.spec_in;
    b.spec_mid = plan->shared_build_work.spec_mid;
    b.real_out = nullptr;
    /* FUSED-tier levels with cuFFTDx support need no cuFFT plan workspace
     * (NVIDIA docs: powers of 2 up to 32768 require zero workspace).
     * Skip plan creation here; lazily created on the rare fallback path. */
    bool skip_cufft = (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx && is_cufftdx_supported_fft_n(fft_n));
    if (!skip_cufft) {
        if (!create_cufft_plan(&b.plan_fwd, fft_n, qb * child_batch, true)) return false;
        if (!create_cufft_plan(&b.plan_inv, fft_n, qb * parent_batch, false)) return false;
        if (!CUFFT_OK(cufftSetStream(b.plan_fwd, plan->stream_compute))) return false;
        if (!CUFFT_OK(cufftSetStream(b.plan_inv, plan->stream_compute))) return false;
    }


    auto &c = plan->corr_fft[ell];
    c.fft_n = fft_n;
    c.cn = cn;
    c.batch_fwd = qb * parent_batch;
    c.batch_inv = qb * 2 * parent_batch;
    c.real_in = nullptr;
    c.spec_in = plan->shared_corr_work.spec_in;
    c.spec_mid = plan->shared_corr_work.spec_mid;
    c.real_out = nullptr;
    if (!skip_cufft) {
        if (!create_cufft_plan(&c.plan_fwd, fft_n, qb * parent_batch, true)) return false;
        if (!create_cufft_plan(&c.plan_inv, fft_n, qb * 2 * parent_batch, false)) return false;
        if (!CUFFT_OK(cufftSetStream(c.plan_fwd, plan->stream_compute))) return false;
        if (!CUFFT_OK(cufftSetStream(c.plan_inv, plan->stream_compute))) return false;
    }


    if (lp.cache_fft && !plan->d_fft_cache[ell]) {
        size_t bytes_cache = (size_t)qb * (size_t)child_batch * (size_t)cn * sizeof(cufftDoubleComplex);
        if (!alloc_device(plan, (void **)&plan->d_fft_cache[ell], bytes_cache, plan->stream_compute)) return false;
    }
    return true;
}

/* Establish the phantom-node invariant for the whole polynomial tree.
 *
 * Every padded slot of every level (and of the pipelined alternate leaf
 * buffer) gets the identity polynomial [1, 0, 0, ...], in every Q-plane.
 * Run once, right after the arena is allocated and zeroed.
 *
 * Nothing ever overwrites those slots with anything else: a level computed
 * at full width recomputes conv(identity, identity) = identity, and a level
 * computed at reduced width does not touch them at all.  That is what lets
 * the level runners stop at n_work[ell] parents while the parent that owns
 * the ragged boundary still reads a well-defined phantom sibling. */
bool init_pad_identity(GpuPlan *plan) {
    int threads = GPU_THREADS_PER_BLOCK;
    int qb = plan->q_batch;
    for (int ell = 0; ell < plan->L; ++ell) {
        int nn_slots = plan->nn[ell];
        int first_pad = plan->n_real[ell];
        int stride = plan->fft_stride[ell];
        if (first_pad >= nn_slots || stride <= 0) continue;
        size_t total = (size_t)qb * (size_t)(nn_slots - first_pad) * (size_t)stride;
        int blocks = (int)((total + threads - 1) / threads);
        k_pad_identity<<<blocks, threads, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], first_pad, nn_slots, stride, qb);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    /* The q-pipeline swaps d_poly_levels[0] with this alternate leaf buffer
     * (single Q-point only), so it needs the same phantom leaves. */
    if (plan->d_poly_leaves_alt && plan->n_real[0] < plan->nn[0]) {
        int stride = plan->fft_stride[0];
        size_t total = (size_t)(plan->nn[0] - plan->n_real[0]) * (size_t)stride;
        int blocks = (int)((total + threads - 1) / threads);
        k_pad_identity<<<blocks, threads, 0, plan->stream_compute>>>(
            plan->d_poly_leaves_alt, plan->n_real[0], plan->nn[0], stride, 1);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

bool allocate_plan_device_memory(GpuPlan *plan) {
    int arena_retries = 0;
retry_arena:
    if (!CUDA_OK(cudaStreamCreate(&plan->stream_compute))) return false;
    if (!CUDA_OK(cudaStreamCreate(&plan->stream_aux))) return false;
    if (!CUDA_OK(cudaEventCreateWithFlags(&plan->evt_a_ready[0], cudaEventDisableTiming))) return false;
    if (!CUDA_OK(cudaEventCreateWithFlags(&plan->evt_a_ready[1], cudaEventDisableTiming))) return false;

    if (!maybe_init_mem_pool(plan)) return false;

    int qb = plan->q_batch;
    size_t block_prod_bytes = (plan->B > 1) ? (size_t)plan->N_tree * (plan->B + 1) * sizeof(double) : 0;

    plan->d_poly_levels.assign(plan->L, nullptr);
    plan->d_g_levels.assign(plan->L, nullptr);
    plan->d_fft_cache.assign(plan->L, nullptr);
    plan->fft_cache_valid.assign(plan->L, false);
    plan->build_fft.assign(plan->L, GpuFftBuffers{});
    plan->corr_fft.assign(plan->L, GpuFftBuffers{});

    size_t mb_si=0, mb_sm=0, mc_si=0, mc_sm=0;
    for (int ell = 1; ell < plan->L; ++ell) {
        auto &lp = plan->levels[ell];
        if (!lp.use_fft || lp.tier == GPU_TIER_SCHOOLBOOK) continue;
        int cn = lp.cn, cb = plan->nn[ell-1], pb = plan->nn[ell];
        mb_si = std::max(mb_si, (size_t)qb*cb*cn*sizeof(cufftDoubleComplex));
        mb_sm = std::max(mb_sm, (size_t)qb*pb*cn*sizeof(cufftDoubleComplex));
        mc_si = std::max(mc_si, (size_t)qb*pb*cn*sizeof(cufftDoubleComplex));
        mc_sm = std::max(mc_sm, (size_t)qb*2*pb*cn*sizeof(cufftDoubleComplex));
    }

    /* Scratch buffer for cuFFT gather/scatter (compact storage).
     * Sized to the largest batch across all FFT levels. */
    size_t fft_scratch_bytes = 0;
    for (int ell = 1; ell < plan->L; ++ell) {
        auto &lp = plan->levels[ell];
        if (!lp.use_fft || lp.tier == GPU_TIER_SCHOOLBOOK) continue;
        int fft_n = lp.fft_n;
        int cb = plan->nn[ell - 1];  /* child_batch = max(child, parent, 2*parent) */
        size_t need = (size_t)qb * cb * fft_n * sizeof(double);
        fft_scratch_bytes = std::max(fft_scratch_bytes, need);
    }

    /* Arena allocation */
    size_t arena_sz = 0;
    #define A(sz) do { arena_sz = (arena_sz + 255) & ~(size_t)255; arena_sz += (sz); } while(0)
    A((size_t)plan->n * sizeof(double));
    A((size_t)plan->n * sizeof(int));
    A((size_t)plan->n * sizeof(int));
    A((size_t)plan->n * sizeof(double));
    A((size_t)plan->n * sizeof(double));
    A(sizeof(double)); A(sizeof(double));
    A(sizeof(double)); A(sizeof(double));
    A((size_t)plan->n * sizeof(double));
    A((size_t)plan->n * sizeof(double));
    A((size_t)plan->k * sizeof(double));
    A(block_prod_bytes);
    if (plan->opts.enable_q_pipeline) {
        A((size_t)plan->nn[0] * plan->fft_stride[0] * sizeof(double));
        A(block_prod_bytes);
    }
    if (qb > 1) {
        for (int qi = 0; qi < qb; ++qi) A((size_t)plan->n * sizeof(double));
        A((size_t)qb * plan->n * sizeof(double));
        A((size_t)qb * block_prod_bytes);
        A((size_t)qb * sizeof(double *));
        A((size_t)qb * sizeof(double));
        A((size_t)qb * sizeof(double));
    }
    for (int ell = 0; ell < plan->L; ++ell) {
        size_t pb = (size_t)qb * plan->nn[ell] * plan->fft_stride[ell] * sizeof(double);
        A(pb); A(pb);
    }
    for (int ell = 1; ell < plan->L; ++ell) {
        auto &lp = plan->levels[ell];
        if (lp.use_fft && lp.cache_fft && lp.tier != GPU_TIER_SCHOOLBOOK)
            A((size_t)qb * plan->nn[ell-1] * lp.cn * sizeof(cufftDoubleComplex));
    }
    A(mb_si); A(mb_sm);
    A(mc_si); A(mc_sm);
    A(fft_scratch_bytes);
    #undef A

    char *arena = nullptr;
    if (arena_sz == 0) { set_last_errorf("Arena size is 0"); return false; }
    if (getenv("ICM_GPU_DEBUG_PLAN"))
        fprintf(stderr, "  arena_sz=%.1f MB  fft_scratch=%.1f MB  spec=(%.1f,%.1f,%.1f,%.1f) MB\n",
            arena_sz/1e6, fft_scratch_bytes/1e6, mb_si/1e6, mb_sm/1e6, mc_si/1e6, mc_sm/1e6);
    {
        cudaError_t arena_err;
        if (plan->use_async_pool) {
            arena_err = cudaMallocAsync((void **)&arena, arena_sz, plan->stream_compute);
        } else {
            arena_err = cudaMalloc(&arena, arena_sz);
        }
        if (arena_err != cudaSuccess) {
            if (plan->q_batch > 1 && arena_retries < 4) {
                plan->q_batch = std::max(1, plan->q_batch / 2);
                arena_retries++;
                cudaStreamDestroy(plan->stream_compute); plan->stream_compute = nullptr;
                cudaStreamDestroy(plan->stream_aux);     plan->stream_aux = nullptr;
                cudaEventDestroy(plan->evt_a_ready[0]);  plan->evt_a_ready[0] = nullptr;
                cudaEventDestroy(plan->evt_a_ready[1]);  plan->evt_a_ready[1] = nullptr;
                goto retry_arena;
            }
            set_last_errorf("CUDA arena alloc(%zu) failed after %d retries: %s",
                            arena_sz, arena_retries, cudaGetErrorString(arena_err));
            return false;
        }
    }
    if (!CUDA_OK(cudaMemset(arena, 0, arena_sz))) return false;
    plan->arena_base = arena;
    plan->arena_total_bytes = arena_sz;
    plan->peak_vram_bytes = arena_sz;
    plan->current_vram_bytes = arena_sz;

    /* Assign pointers from arena */
    size_t off = 0;
    #define P(ptr, type, sz) do { off = (off + 255) & ~(size_t)255; (ptr) = (type)(arena + off); off += (sz); } while(0)
    P(plan->d_S_sorted, double*, (size_t)plan->n * sizeof(double));
    P(plan->d_sort_perm, int*, (size_t)plan->n * sizeof(int));
    P(plan->d_inv_perm, int*, (size_t)plan->n * sizeof(int));
    P(plan->d_a_sorted[0], double*, (size_t)plan->n * sizeof(double));
    P(plan->d_a_sorted[1], double*, (size_t)plan->n * sizeof(double));
    P(plan->d_graph_logv[0], double*, sizeof(double));
    P(plan->d_graph_logv[1], double*, sizeof(double));
    P(plan->d_graph_scale[0], double*, sizeof(double));
    P(plan->d_graph_scale[1], double*, sizeof(double));
    P(plan->d_inner_sorted, double*, (size_t)plan->n * sizeof(double));
    P(plan->d_equity, double*, (size_t)plan->n * sizeof(double));
    P(plan->d_payout, double*, (size_t)plan->k * sizeof(double));
    if (block_prod_bytes > 0) P(plan->d_block_prods, double*, block_prod_bytes);
    if (plan->opts.enable_q_pipeline) {
        P(plan->d_poly_leaves_alt, double*, (size_t)plan->nn[0] * plan->fft_stride[0] * sizeof(double));
        if (block_prod_bytes > 0) P(plan->d_block_prods_alt, double*, block_prod_bytes);
        if (!CUDA_OK(cudaEventCreateWithFlags(&plan->evt_prop_done, cudaEventDisableTiming))) return false;
    }
    if (qb > 1) {
        for (int qi = 0; qi < qb; ++qi) P(plan->d_a_qbatch[qi], double*, (size_t)plan->n * sizeof(double));
        P(plan->d_inner_qbatch, double*, (size_t)qb * plan->n * sizeof(double));
        P(plan->d_block_prods_qbatch, double*, (size_t)qb * block_prod_bytes);
        P(plan->d_qb_a_ptrs, double**, (size_t)qb * sizeof(double*));
        P(plan->d_qb_weights, double*, (size_t)qb * sizeof(double));
        P(plan->d_qb_inv_vs, double*, (size_t)qb * sizeof(double));
    }
    for (int ell = 0; ell < plan->L; ++ell) {
        size_t pb = (size_t)qb * plan->nn[ell] * plan->fft_stride[ell] * sizeof(double);
        P(plan->d_poly_levels[ell], double*, pb);
        P(plan->d_g_levels[ell], double*, pb);
    }
    for (int ell = 1; ell < plan->L; ++ell) {
        auto &lp = plan->levels[ell];
        if (lp.use_fft && lp.cache_fft && lp.tier != GPU_TIER_SCHOOLBOOK)
            P(plan->d_fft_cache[ell], cufftDoubleComplex*, (size_t)qb * plan->nn[ell-1] * lp.cn * sizeof(cufftDoubleComplex));
    }
    auto &sb = plan->shared_build_work;
    auto &sc = plan->shared_corr_work;
    sb.real_in_bytes=0; sb.spec_in_bytes=mb_si; sb.spec_mid_bytes=mb_sm; sb.real_out_bytes=0;
    sc.real_in_bytes=0; sc.spec_in_bytes=mc_si; sc.spec_mid_bytes=mc_sm; sc.real_out_bytes=0;
    sb.real_in=nullptr; sb.real_out=nullptr;
    sc.real_in=nullptr; sc.real_out=nullptr;
    P(sb.spec_in, cufftDoubleComplex*, mb_si);
    P(sb.spec_mid, cufftDoubleComplex*, mb_sm);
    P(sc.spec_in, cufftDoubleComplex*, mc_si);
    P(sc.spec_mid, cufftDoubleComplex*, mc_sm);
    if (fft_scratch_bytes > 0) P(plan->d_fft_scratch, double*, fft_scratch_bytes);
    #undef P
    for (int ell = 1; ell < plan->L; ++ell) {
        if (!allocate_level_buffers(plan, ell, {})) return false;
    }


    /* Share cuFFT workspace */
    {
        size_t max_ws = 0;
        for (int ell = 1; ell < plan->L; ++ell) {
            auto &lp = plan->levels[ell];
            if (!lp.use_fft || lp.tier == GPU_TIER_SCHOOLBOOK) continue;
            size_t ws = 0;
            if (plan->build_fft[ell].plan_fwd) { cufftGetSize(plan->build_fft[ell].plan_fwd, &ws); max_ws = std::max(max_ws, ws); }
            if (plan->build_fft[ell].plan_inv) { cufftGetSize(plan->build_fft[ell].plan_inv, &ws); max_ws = std::max(max_ws, ws); }
            if (plan->corr_fft[ell].plan_fwd) { cufftGetSize(plan->corr_fft[ell].plan_fwd, &ws); max_ws = std::max(max_ws, ws); }
            if (plan->corr_fft[ell].plan_inv) { cufftGetSize(plan->corr_fft[ell].plan_inv, &ws); max_ws = std::max(max_ws, ws); }
        }
        if (max_ws > 0) {
            if (!alloc_device(plan, &plan->shared_cufft_workspace, max_ws, plan->stream_compute)) {
                if (plan->q_batch > 1 && arena_retries < 4) {
                    for (int e = 1; e < plan->L; ++e) {
                        auto &bf = plan->build_fft[e];
                        auto &cf = plan->corr_fft[e];
                        if (bf.plan_fwd) { cufftDestroy(bf.plan_fwd); bf.plan_fwd = 0; }
                        if (bf.plan_inv) { cufftDestroy(bf.plan_inv); bf.plan_inv = 0; }
                        if (cf.plan_fwd) { cufftDestroy(cf.plan_fwd); cf.plan_fwd = 0; }
                        if (cf.plan_inv) { cufftDestroy(cf.plan_inv); cf.plan_inv = 0; }
                    }
                    free_device(plan, plan->arena_base, plan->stream_compute); plan->arena_base = nullptr;
                    plan->arena_total_bytes = 0;
                    plan->peak_vram_bytes = 0;
                    plan->current_vram_bytes = 0;
                    cudaStreamDestroy(plan->stream_compute); plan->stream_compute = nullptr;
                    cudaStreamDestroy(plan->stream_aux);     plan->stream_aux = nullptr;
                    cudaEventDestroy(plan->evt_a_ready[0]);  plan->evt_a_ready[0] = nullptr;
                    cudaEventDestroy(plan->evt_a_ready[1]);  plan->evt_a_ready[1] = nullptr;
                    if (plan->evt_prop_done) { cudaEventDestroy(plan->evt_prop_done); plan->evt_prop_done = nullptr; }
                    plan->q_batch = std::max(1, plan->q_batch / 2);
                    arena_retries++;
                    goto retry_arena;
                }
                return false;
            }
            plan->shared_cufft_workspace_bytes = max_ws;
            for (int ell = 1; ell < plan->L; ++ell) {
                auto &lp = plan->levels[ell];
                if (!lp.use_fft || lp.tier == GPU_TIER_SCHOOLBOOK) continue;
                auto &b = plan->build_fft[ell]; auto &c = plan->corr_fft[ell];
                if (b.plan_fwd) { CUFFT_OK(cufftSetAutoAllocation(b.plan_fwd, 0)); CUFFT_OK(cufftSetWorkArea(b.plan_fwd, plan->shared_cufft_workspace)); }
                if (b.plan_inv) { CUFFT_OK(cufftSetAutoAllocation(b.plan_inv, 0)); CUFFT_OK(cufftSetWorkArea(b.plan_inv, plan->shared_cufft_workspace)); }
                if (c.plan_fwd) { CUFFT_OK(cufftSetAutoAllocation(c.plan_fwd, 0)); CUFFT_OK(cufftSetWorkArea(c.plan_fwd, plan->shared_cufft_workspace)); }
                if (c.plan_inv) { CUFFT_OK(cufftSetAutoAllocation(c.plan_inv, 0)); CUFFT_OK(cufftSetWorkArea(c.plan_inv, plan->shared_cufft_workspace)); }
            }
        }
    }

    if (!init_pad_identity(plan)) return false;

    if (!CUDA_OK(cudaMemcpyAsync(plan->d_S_sorted, plan->S_sorted.data(),
                                 (size_t)plan->n * sizeof(double), cudaMemcpyHostToDevice,
                                 plan->stream_compute))) return false;
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_sort_perm, plan->sort_perm.data(),
                                 (size_t)plan->n * sizeof(int), cudaMemcpyHostToDevice,
                                 plan->stream_compute))) return false;
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_inv_perm, plan->inv_perm.data(),
                                 (size_t)plan->n * sizeof(int), cudaMemcpyHostToDevice,
                                 plan->stream_compute))) return false;
    if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
    return true;
}

}  // namespace icm_gpu_detail
