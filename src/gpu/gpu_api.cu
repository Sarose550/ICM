/* gpu_api.cu -- Public API, global state, calibration helpers.
 *
 * All extern "C" functions live here, outside the namespace.
 * The icm_gpu_detail namespace is used for internal access.
 */
#include "gpu_internal.h"

/* ── Global state definitions ──────────────────────────────────── */
namespace icm_gpu_detail {
    std::string g_last_error;
    int g_cuda_device = -1;
    int g_runtime_fused_max_conv_len = GPU_FUSED_MAX_CONV_LEN;

    void set_last_errorf(const char *fmt, ...) {
        char buf[1024];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        g_last_error = buf;
    }

    bool cuda_ok(cudaError_t err, const char *expr, const char *file, int line) {
        if (err == cudaSuccess) return true;
        set_last_errorf("CUDA error at %s:%d for %s: %s", file, line, expr, cudaGetErrorString(err));
        return false;
    }

    bool cufft_ok(cufftResult err, const char *expr, const char *file, int line) {
        if (err == CUFFT_SUCCESS) return true;
        set_last_errorf("cuFFT error at %s:%d for %s: code=%d", file, line, expr, (int)err);
        return false;
    }
}  // namespace icm_gpu_detail

using namespace icm_gpu_detail;

/* ── GPU/CPU crossover ─────────────────────────────────────────── */

#ifndef GPU_CPU_CROSSOVER
#define GPU_CPU_CROSSOVER 0
#endif

static int gpu_cpu_crossover() {
    static int val = -1;
    if (val < 0) {
        val = GPU_CPU_CROSSOVER;
        const char *env = getenv("ICM_GPU_CPU_CROSSOVER");
        if (env && env[0]) {
            int v = atoi(env);
            if (v >= 0) val = v;
        }
    }
    return val;
}

/* ── Public C API ──────────────────────────────────────────────── */

extern "C" {

int icm_gpu_init(int device_id) {
    int count = 0;
    if (!CUDA_OK(cudaGetDeviceCount(&count))) return 0;
    if (count <= 0) {
        set_last_errorf("No CUDA devices found");
        return 0;
    }
    if (device_id < 0) device_id = 0;
    if (device_id >= count) {
        set_last_errorf("Invalid CUDA device id %d (count=%d)", device_id, count);
        return 0;
    }
    if (!CUDA_OK(cudaSetDevice(device_id))) return 0;
    int max_optin = 0;
    if (CUDA_OK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id)) &&
        max_optin > 0) {
        cudaFuncSetAttribute(k_block_build, cudaFuncAttributeMaxDynamicSharedMemorySize, max_optin);
    }
    g_cuda_device = device_id;
    return 1;
}

void icm_gpu_shutdown(void) {
    if (g_cuda_device >= 0) {
        cudaSetDevice(g_cuda_device);
        cudaDeviceSynchronize();
    }
}

const char *icm_gpu_last_error(void) {
    return g_last_error.c_str();
}

IcmGpuPlan *icm_gpu_plan_create(int n, const double *S, int k, const IcmGpuOptions *opts) {
    if (g_cuda_device < 0) {
        set_last_errorf("icm_gpu_init must be called before icm_gpu_plan_create");
        return nullptr;
    }
    if (n <= 0 || !S || k <= 0 || k > n) {
        set_last_errorf("Invalid plan args n=%d k=%d", n, k);
        return nullptr;
    }

    auto *plan = new GpuPlan();
    plan->n = n;
    plan->k = k;
    plan->opts = IcmGpuOptions{};
    plan->opts.force_uncached_fused_levels = -1;
    plan->opts.force_uncached_cufft_levels = -1;
    if (opts) plan->opts = *opts;
    plan->opts.device_id = g_cuda_device;
    plan->opts.enable_q_pipeline = plan->opts.enable_q_pipeline ? 1 : 0;
    plan->S_sorted.assign(S, S + n);

    double t_sort0 = now_ns_host();
    if (!device_sort_players(plan)) {
        destroy_plan(plan);
        return nullptr;
    }
    (void)t_sort0;

    if (!build_plan_metadata(plan)) {
        destroy_plan(plan);
        return nullptr;
    }

    /* Determine q_batch */
    {
        const char *qb_env = getenv("ICM_GPU_Q_BATCH");
        int qb_override = 0;
        if (qb_env && qb_env[0]) {
            int v = atoi(qb_env);
            if (v >= 1 && v <= Q_BATCH_MAX) qb_override = v;
        }

        size_t per_q_bytes = 0;
        for (int ell = 0; ell < plan->L; ++ell)
            per_q_bytes += 2 * (size_t)plan->nn[ell] * plan->psz[ell] * sizeof(double);
        per_q_bytes += (size_t)plan->N_tree * (plan->B + 1) * sizeof(double);
        per_q_bytes += 2 * (size_t)plan->n * sizeof(double);
        size_t budget = (size_t)((double)GPU_VRAM_BYTES * 0.60);

        int best_qb = 1;
        if (!plan->opts.enable_graphs && !qb_override) {
            best_qb = (per_q_bytes > 0) ? (int)(budget / per_q_bytes) : Q_BATCH_MAX;
            if (best_qb > Q_BATCH_MAX) best_qb = Q_BATCH_MAX;
            if (best_qb < 1) best_qb = 1;
        }
        int qb = qb_override ? qb_override : best_qb;
        if (plan->opts.enable_graphs) qb = 1;
        plan->q_batch = qb;
    }

    if (!allocate_plan_device_memory(plan)) {
        destroy_plan(plan);
        return nullptr;
    }

    if (!create_graph_stub(plan)) {
        destroy_plan(plan);
        return nullptr;
    }

    plan->planned_peak_vram_bytes = plan->peak_vram_bytes;
    return reinterpret_cast<IcmGpuPlan *>(plan);
}

void icm_gpu_plan_destroy(IcmGpuPlan *plan_opaque) {
    auto *plan = reinterpret_cast<GpuPlan *>(plan_opaque);
    destroy_plan(plan);
}

int icm_gpu_plan_summary(const IcmGpuPlan *plan_opaque, IcmGpuPlanSummary *summary) {
    const auto *plan = reinterpret_cast<const GpuPlan *>(plan_opaque);
    if (!plan || !summary) return 0;
    memset(summary, 0, sizeof(*summary));
    summary->n = plan->n;
    summary->k = plan->k;
    summary->B = plan->B;
    summary->engine = plan->engine;
    summary->n_levels = plan->L;
    for (int ell = 1; ell < plan->L; ++ell) {
        if (!plan->levels[ell].use_fft) summary->n_tier1++;
        else if (plan->levels[ell].tier == GPU_TIER_FUSED) summary->n_tier2++;
        else {
            summary->n_tier3++;
#if ICM_HAVE_VKFFT
            if (plan->build_fft[ell].use_vkfft) summary->n_vkfft++;
#endif
        }
    }
    summary->q_batch = plan->q_batch;
    summary->planned_peak_vram_bytes = plan->planned_peak_vram_bytes;
    return 1;
}

double icm_gpu_equity_with_plan(IcmGpuPlan *plan_opaque, int Q,
                                const double *payout, double *equity,
                                IcmGpuRunStats *stats) {
    auto *plan = reinterpret_cast<GpuPlan *>(plan_opaque);
    if (!plan || !payout || !equity || Q <= 0) {
        set_last_errorf("icm_gpu_equity_with_plan invalid arguments");
        return -1.0;
    }
    if (!CUDA_OK(cudaSetDevice(g_cuda_device))) return -1.0;
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_payout, payout, (size_t)plan->k * sizeof(double),
                                 cudaMemcpyHostToDevice, plan->stream_compute))) return -1.0;

    int threads = 256;
    int blocks = (plan->n + threads - 1) / threads;
    k_zero<<<blocks, threads, 0, plan->stream_compute>>>(plan->d_equity, (size_t)plan->n);
    if (!CUDA_OK(cudaGetLastError())) return -1.0;

    double Smax = 0.0;
    for (int i = 0; i < plan->n; ++i) if (plan->S_sorted[i] > Smax) Smax = plan->S_sorted[i];
    std::vector<QP> pts;
    make_nodes(Q, Smax, pts);

    double sort_ns = 0.0;
    double quad_ovh_ns = 0.0;
    double block_ns = 0.0;
    double tree_build_ns = 0.0;
    double tree_prop_cached_ns = 0.0;
    double tree_prop_recomp_ns = 0.0;
    double leaf_ns = 0.0;
    double accum_ns = 0.0;

    bool fast = !plan->opts.verbose;

    int qb = plan->q_batch;
    if (qb > Q) qb = Q;

    double t0 = now_ns_host();
    if (plan->opts.enable_graphs && (plan->graph_ready[0] || plan->graph_ready[1])) {
        for (int q = 0; q < Q; ++q) {
            if (pts[q].w == 0.0) continue;
            int curr = q & 1;
            if (!run_hybrid_single_q(plan, curr, pts[q].logv, pts[q].w, false, fast,
                                     &block_ns, &tree_build_ns, &tree_prop_cached_ns,
                                     &tree_prop_recomp_ns, &leaf_ns, &accum_ns)) {
                return -1.0;
            }
        }
    } else if (qb > 1) {
        std::vector<QP> active_pts;
        active_pts.reserve(Q);
        for (int q = 0; q < Q; ++q) {
            if (pts[q].w != 0.0) active_pts.push_back(pts[q]);
        }
        int n_active = (int)active_pts.size();
        for (int q = 0; q < n_active; q += qb) {
            int batch_sz = std::min(qb, n_active - q);
            if (batch_sz == qb) {
                if (!run_hybrid_batched_q(plan, &active_pts[q], qb)) return -1.0;
            } else {
                QP padded[Q_BATCH_MAX];
                for (int r = 0; r < batch_sz; ++r) padded[r] = active_pts[q + r];
                for (int r = batch_sz; r < qb; ++r) {
                    padded[r].logv = active_pts[q].logv;
                    padded[r].w = 0.0;
                }
                if (!run_hybrid_batched_q(plan, padded, qb)) return -1.0;
            }
        }
    } else if (plan->opts.enable_q_pipeline && plan->d_poly_leaves_alt) {
        double *leaf_bufs[2] = { plan->d_poly_levels[0], plan->d_poly_leaves_alt };
        double *bp_bufs[2]   = { plan->d_block_prods,    plan->d_block_prods_alt };
        double *orig_poly_levels_0 = plan->d_poly_levels[0];
        double *orig_block_prods   = plan->d_block_prods;
        int buf_idx = 0;
        bool pipeline_ok = true;

        int q_start = 0;
        while (q_start < Q && pts[q_start].w == 0.0) ++q_start;

        if (q_start < Q) {
            int curr = q_start & 1;
            k_compute_a<<<blocks, threads, 0, plan->stream_aux>>>(
                plan->d_S_sorted, plan->d_a_sorted[curr], plan->n, pts[q_start].logv);
            if (!CUDA_OK(cudaGetLastError())) { pipeline_ok = false; goto pipeline_cleanup; }
            if (plan->B <= 1) {
                int bl = (plan->N_tree + 255) / 256;
                k_set_leaves_b1<<<bl, 256, 0, plan->stream_aux>>>(
                    plan->d_a_sorted[curr], plan->n, plan->N_tree, plan->fft_stride[0], leaf_bufs[buf_idx]);
            } else {
                int threads_block = 256;
                size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);
                k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_aux>>>(
                    plan->d_a_sorted[curr], plan->n, plan->B,
                    plan->nblocks, plan->N_tree, plan->fft_stride[0],
                    leaf_bufs[buf_idx], bp_bufs[buf_idx]);
            }
            if (!CUDA_OK(cudaGetLastError())) { pipeline_ok = false; goto pipeline_cleanup; }
            if (!CUDA_OK(cudaEventRecord(plan->evt_a_ready[curr], plan->stream_aux))) { pipeline_ok = false; goto pipeline_cleanup; }
        }

        for (int q = q_start; q < Q; ++q) {
            if (pts[q].w == 0.0) continue;
            int curr = q & 1;
            if (!CUDA_OK(cudaStreamWaitEvent(plan->stream_compute, plan->evt_a_ready[curr], 0))) { pipeline_ok = false; break; }
            plan->d_poly_levels[0] = leaf_bufs[buf_idx];
            if (plan->B > 1) plan->d_block_prods = bp_bufs[buf_idx];

            int qn = q + 1;
            while (qn < Q && pts[qn].w == 0.0) ++qn;
            int next_buf = 1 - buf_idx;

            if (qn < Q) {
                if (!CUDA_OK(cudaStreamWaitEvent(plan->stream_aux, plan->evt_prop_done, 0))) { pipeline_ok = false; break; }
                int next = qn & 1;
                k_compute_a<<<blocks, threads, 0, plan->stream_aux>>>(
                    plan->d_S_sorted, plan->d_a_sorted[next], plan->n, pts[qn].logv);
                if (!CUDA_OK(cudaGetLastError())) { pipeline_ok = false; break; }
                if (plan->B <= 1) {
                    int bl = (plan->N_tree + 255) / 256;
                    k_set_leaves_b1<<<bl, 256, 0, plan->stream_aux>>>(
                        plan->d_a_sorted[next], plan->n, plan->N_tree, plan->fft_stride[0], leaf_bufs[next_buf]);
                } else {
                    int threads_block = 256;
                    size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);
                    k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_aux>>>(
                        plan->d_a_sorted[next], plan->n, plan->B,
                        plan->nblocks, plan->N_tree, plan->fft_stride[0],
                        leaf_bufs[next_buf], bp_bufs[next_buf]);
                }
                if (!CUDA_OK(cudaGetLastError())) { pipeline_ok = false; break; }
                if (!CUDA_OK(cudaEventRecord(plan->evt_a_ready[next], plan->stream_aux))) { pipeline_ok = false; break; }
            }

            for (int ell = 1; ell < plan->L - 1; ++ell) {
                auto &lp = plan->levels[ell];
                if (!lp.use_fft) { if (!run_build_level_schoolbook(plan, ell)) { pipeline_ok = false; break; } }
                else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_build_level_fused(plan, ell)) { pipeline_ok = false; break; } }
                else { if (!run_build_level_fft(plan, ell)) { pipeline_ok = false; break; } }
            }
            if (!pipeline_ok) break;

            int top = plan->L - 1;
            int root_gsz = plan->fft_stride[top];
            int blocks_root = (root_gsz + threads - 1) / threads;
            k_set_root_g<<<blocks_root, threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[top], root_gsz, plan->d_payout, plan->k);

            for (int ell = top; ell >= 1; --ell) {
                auto &lp = plan->levels[ell];
                if (!lp.use_fft) { if (!run_prop_level_schoolbook(plan, ell)) { pipeline_ok = false; break; } }
                else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_prop_level_fused(plan, ell)) { pipeline_ok = false; break; } }
                else { if (!run_prop_level_fft(plan, ell)) { pipeline_ok = false; break; } }
            }
            if (!pipeline_ok) break;

            if (!CUDA_OK(cudaEventRecord(plan->evt_prop_done, plan->stream_compute))) { pipeline_ok = false; break; }

            if (plan->B <= 1) {
                int bl = (plan->n + 255) / 256;
                k_leaf_extract_b1<<<bl, 256, 0, plan->stream_compute>>>(
                    plan->n, plan->d_g_levels[0], plan->fft_stride[0], plan->d_inner_sorted);
            } else {
                int threads_leaf = plan->B;
                if (threads_leaf > 1024) threads_leaf = 1024;
                k_leaf_extract<<<plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
                    plan->d_a_sorted[curr], plan->n, plan->B, plan->nblocks,
                    plan->d_block_prods, plan->d_g_levels[0], plan->fft_stride[0],
                    plan->g_needed[0], plan->k, plan->d_inner_sorted);
            }

            double inv_v = exp(-pts[q].logv);
            k_accumulate_equity<<<blocks, threads, 0, plan->stream_compute>>>(
                plan->d_inner_sorted, plan->d_a_sorted[curr], plan->d_S_sorted,
                plan->d_sort_perm, plan->n, pts[q].w, inv_v, plan->d_equity);
            buf_idx = next_buf;
        }

pipeline_cleanup:
        plan->d_poly_levels[0] = orig_poly_levels_0;
        if (plan->B > 1) plan->d_block_prods = orig_block_prods;
        if (!pipeline_ok) return -1.0;
    } else {
        for (int q = 0; q < Q; ++q) {
            if (pts[q].w == 0.0) continue;
            int curr = q & 1;
            if (!run_hybrid_single_q(plan, curr, pts[q].logv, pts[q].w, false, fast,
                                     &block_ns, &tree_build_ns, &tree_prop_cached_ns,
                                     &tree_prop_recomp_ns, &leaf_ns, &accum_ns)) {
                return -1.0;
            }
        }
    }
    if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return -1.0;
    double total_ns = now_ns_host() - t0;

    if (!CUDA_OK(cudaMemcpy(equity, plan->d_equity, (size_t)plan->n * sizeof(double),
                            cudaMemcpyDeviceToHost))) return -1.0;

    if (stats) {
        memset(stats, 0, sizeof(*stats));
        stats->total_ns = total_ns;
        stats->sort_ns = sort_ns;
        stats->quadrature_overhead_ns = std::max(0.0, quad_ovh_ns);
        stats->block_build_ns = block_ns;
        stats->tree_build_ns = tree_build_ns;
        stats->tree_propagate_cached_ns = tree_prop_cached_ns;
        stats->tree_propagate_recomputed_ns = tree_prop_recomp_ns;
        stats->leaf_extract_ns = leaf_ns;
        stats->peak_vram_bytes = std::max(plan->peak_vram_bytes, plan->planned_peak_vram_bytes);
        stats->engine = plan->engine;
        stats->B = plan->B;
        stats->uncached_fused_levels = plan->uncached_fused_levels;
        stats->uncached_cufft_levels = plan->uncached_cufft_levels;
    }
    return total_ns;
}

double icm_gpu_equity(int n, const double *S, int Q,
                      const double *payout, int k,
                      double *equity, const IcmGpuOptions *opts,
                      IcmGpuRunStats *stats) {
    if (g_cuda_device < 0) {
        if (!icm_gpu_init(0)) return -1.0;
    }

    int sk_max = single_kernel_max_n(k);
    if (n <= sk_max && n > 0 && k > 0 && k <= n) {
        double t0_sk = now_ns_host();

        std::vector<std::pair<double, int>> pairs(n);
        for (int i = 0; i < n; i++) pairs[i] = {S[i], i};
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
            return a.first > b.first ? true : (a.first < b.first ? false : a.second < b.second);
        });
        std::vector<double> S_sorted(n);
        std::vector<int> sort_perm(n);
        for (int i = 0; i < n; i++) {
            S_sorted[i] = pairs[i].first;
            sort_perm[i] = pairs[i].second;
        }

        int N = 1;
        while (N < n) N <<= 1;
        int L = 0;
        { int tmp = N; while (tmp > 1) { tmp >>= 1; L++; } L++; }
        std::vector<int> nn(L), psz(L), g_needed(L);
        std::vector<size_t> plev_off(L);
        nn[0] = N;
        for (int ell = 1; ell < L; ell++) nn[ell] = N >> ell;
        size_t off = 0;
        for (int ell = 0; ell < L; ell++) {
            long d = 1L << (ell + 1);
            psz[ell] = (d > k) ? k : (int)d;
            plev_off[ell] = off;
            off += (size_t)nn[ell] * psz[ell];
        }
        int total_poly = (int)off;
        g_needed[0] = 1;
        for (int ell = 1; ell < L; ell++) {
            int need = g_needed[ell - 1] + psz[ell - 1] - 1;
            g_needed[ell] = (need < psz[ell]) ? need : psz[ell];
        }
        int max_g = 0;
        for (int ell = 0; ell < L; ell++) {
            int sz = nn[ell] * psz[ell];
            if (sz > max_g) max_g = sz;
        }

        size_t shmem_bytes = ((size_t)total_poly + 2 * (size_t)max_g) * sizeof(double);
        if (shmem_bytes > 200 * 1024) {
            goto standard_gpu_path;
        }

        double Smax = S_sorted[0];
        std::vector<QP> pts;
        make_nodes(Q, Smax, pts);
        int active_Q = 0;
        for (int q = 0; q < Q; q++) if (pts[q].w != 0.0) active_Q++;
        (void)active_Q;

        double *d_S = nullptr, *d_payout_buf = nullptr, *d_equity_buf = nullptr;
        double *d_logv = nullptr, *d_weights = nullptr;
        int *d_perm = nullptr, *d_nn = nullptr, *d_psz = nullptr, *d_g_needed_d = nullptr;
        size_t *d_plev_off_d = nullptr;

        size_t arena_sz = 0;
        arena_sz += 256; arena_sz += n * sizeof(double);
        arena_sz += 256; arena_sz += n * sizeof(int);
        arena_sz += 256; arena_sz += k * sizeof(double);
        arena_sz += 256; arena_sz += n * sizeof(double);
        arena_sz += 256; arena_sz += Q * sizeof(double);
        arena_sz += 256; arena_sz += Q * sizeof(double);
        arena_sz += 256; arena_sz += L * sizeof(int);
        arena_sz += 256; arena_sz += L * sizeof(int);
        arena_sz += 256; arena_sz += L * sizeof(int);
        arena_sz += 256; arena_sz += L * sizeof(size_t);
        char *sk_arena = nullptr;
        if (!CUDA_OK(cudaMalloc(&sk_arena, arena_sz))) return -1.0;
        if (!CUDA_OK(cudaMemset(sk_arena, 0, arena_sz))) { cudaFree(sk_arena); return -1.0; }

        size_t sk_off = 0;
        #define SKP(ptr, type, sz) do { sk_off = (sk_off + 255) & ~(size_t)255; (ptr) = (type)(sk_arena + sk_off); sk_off += (sz); } while(0)
        SKP(d_S, double*, n * sizeof(double));
        SKP(d_perm, int*, n * sizeof(int));
        SKP(d_payout_buf, double*, k * sizeof(double));
        SKP(d_equity_buf, double*, n * sizeof(double));
        SKP(d_logv, double*, Q * sizeof(double));
        SKP(d_weights, double*, Q * sizeof(double));
        SKP(d_nn, int*, L * sizeof(int));
        SKP(d_psz, int*, L * sizeof(int));
        SKP(d_g_needed_d, int*, L * sizeof(int));
        SKP(d_plev_off_d, size_t*, L * sizeof(size_t));
        #undef SKP

        std::vector<double> h_logv(Q), h_weights(Q);
        for (int q = 0; q < Q; q++) { h_logv[q] = pts[q].logv; h_weights[q] = pts[q].w; }
        cudaMemcpy(d_S, S_sorted.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_perm, sort_perm.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_payout_buf, payout, k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_logv, h_logv.data(), Q * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, h_weights.data(), Q * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nn, nn.data(), L * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_psz, psz.data(), L * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g_needed_d, g_needed.data(), L * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_plev_off_d, plev_off.data(), L * sizeof(size_t), cudaMemcpyHostToDevice);

        cudaFuncSetAttribute(k_icm_single_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_bytes);

        int nthreads = 256;
        if (n < 256) nthreads = ((n + 31) / 32) * 32;
        k_icm_single_kernel<<<Q, nthreads, shmem_bytes>>>(
            d_S, d_perm, n, Q, d_logv, d_weights, d_payout_buf, k, d_equity_buf,
            N, L, d_nn, d_psz, d_g_needed_d, d_plev_off_d, total_poly, max_g);
        cudaDeviceSynchronize();

        memset(equity, 0, n * sizeof(double));
        cudaMemcpy(equity, d_equity_buf, n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(sk_arena);

        double total_ns = now_ns_host() - t0_sk;
        if (stats) {
            memset(stats, 0, sizeof(*stats));
            stats->total_ns = total_ns;
            stats->engine = 2;
        }
        return total_ns;
    }

standard_gpu_path:
    IcmGpuPlan *plan = icm_gpu_plan_create(n, S, k, opts);
    if (!plan) return -1.0;
    double t = icm_gpu_equity_with_plan(plan, Q, payout, equity, stats);
    icm_gpu_plan_destroy(plan);
    return t;
}

double icm_gpu_equity_subset(int n, const double *S, int Q,
                             const double *payout, int k,
                             double *equity,
                             const int *targets, int n_targets,
                             const IcmGpuOptions *opts,
                             IcmGpuRunStats *stats) {
    if (n <= 0 || k <= 0 || n_targets <= 0 || !targets) return -1.0;
    if (g_cuda_device < 0) {
        if (!icm_gpu_init(0)) return -1.0;
    }

    IcmGpuPlan *plan = icm_gpu_plan_create(n, S, k, opts);
    if (!plan) return -1.0;

    auto *gplan = reinterpret_cast<GpuPlan *>(plan);
    std::vector<uint8_t> h_mask(n, 0);
    for (int i = 0; i < n_targets; ++i) {
        int orig = targets[i];
        if (orig >= 0 && orig < n) {
            h_mask[gplan->inv_perm[orig]] = 1;
        }
    }

    uint8_t *d_mask = nullptr;
    if (!CUDA_OK(cudaMalloc(&d_mask, (size_t)n * sizeof(uint8_t)))) {
        icm_gpu_plan_destroy(plan);
        return -1.0;
    }
    if (!CUDA_OK(cudaMemcpy(d_mask, h_mask.data(), (size_t)n * sizeof(uint8_t),
                            cudaMemcpyHostToDevice))) {
        cudaFree(d_mask);
        icm_gpu_plan_destroy(plan);
        return -1.0;
    }
    gplan->d_active_mask = d_mask;

    double t = icm_gpu_equity_with_plan(plan, Q, payout, equity, stats);

    gplan->d_active_mask = nullptr;
    cudaFree(d_mask);
    icm_gpu_plan_destroy(plan);

    if (t >= 0) {
        std::vector<bool> is_target(n, false);
        for (int i = 0; i < n_targets; ++i) {
            if (targets[i] >= 0 && targets[i] < n) is_target[targets[i]] = true;
        }
        for (int i = 0; i < n; ++i) {
            if (!is_target[i]) equity[i] = 0.0;
        }
    }
    return t;
}

/* ── Calibration helpers ───────────────────────────────────────── */

int icm_gpu_measure_fused_pair_ns(int fft_n, int batch, int quick,
                                  double *build_ns_out, double *corr_ns_out) {
    if (!build_ns_out || !corr_ns_out || fft_n <= 0 || batch <= 0) return 0;
    *build_ns_out = 0.0;
    *corr_ns_out = 0.0;
    if (g_cuda_device < 0) { if (!icm_gpu_init(0)) return 0; }
    if (!CUDA_OK(cudaSetDevice(g_cuda_device))) return 0;
#if !ICM_HAVE_CUFFTDX
    (void)quick;
    return 0;
#else
    if (!is_cufftdx_supported_fft_n(fft_n)) return 0;
    int nparents = batch;
    int cps = fft_n, pps = fft_n, len_g = fft_n, len_P = fft_n, len_out = fft_n;
    double *d_child=0, *d_parent=0, *d_g_parent=0, *d_child_poly=0, *d_g_child=0;
    cudaStream_t stream=0; cudaEvent_t e0=0, e1=0;
    int warmup = quick ? 1 : 3, reps = quick ? 4 : 12;
    const char *warm_env = getenv("ICM_GPU_CALIB_WARMUP");
    if (warm_env && warm_env[0]) { int w = atoi(warm_env); if (w >= 0 && w <= 32) warmup = w; }
    const char *rep_env = getenv("ICM_GPU_CALIB_MIN_REPS");
    if (rep_env && rep_env[0]) { int r = atoi(rep_env); if (r > reps) reps = r; }
    std::vector<double> bsamp, csamp;
    bsamp.reserve((size_t)reps); csamp.reserve((size_t)reps);
    size_t cb = (size_t)(2*nparents)*cps*sizeof(double);
    size_t pb = (size_t)nparents*pps*sizeof(double);
    size_t gb = (size_t)nparents*fft_n*sizeof(double);
    size_t gcb = (size_t)(2*nparents)*fft_n*sizeof(double);
    if (!CUDA_OK(cudaStreamCreate(&stream))) goto fail;
    if (!CUDA_OK(cudaMalloc(&d_child, cb))) goto fail;
    if (!CUDA_OK(cudaMalloc(&d_parent, pb))) goto fail;
    if (!CUDA_OK(cudaMalloc(&d_g_parent, gb))) goto fail;
    if (!CUDA_OK(cudaMalloc(&d_child_poly, cb))) goto fail;
    if (!CUDA_OK(cudaMalloc(&d_g_child, gcb))) goto fail;
    cudaMemsetAsync(d_child, 1, cb, stream);
    cudaMemsetAsync(d_parent, 0, pb, stream);
    cudaMemsetAsync(d_g_parent, 1, gb, stream);
    cudaMemsetAsync(d_child_poly, 2, cb, stream);
    cudaMemsetAsync(d_g_child, 0, gcb, stream);
    cudaStreamSynchronize(stream);
    if (!CUDA_OK(cudaEventCreate(&e0))) goto fail;
    if (!CUDA_OK(cudaEventCreate(&e1))) goto fail;
    for (int i = 0; i < warmup; ++i)
        if (!launch_cufftdx_build_dispatch(fft_n, d_child, cps, d_parent, pps, nparents, 1.0/(double)fft_n, stream)) goto fail;
    cudaStreamSynchronize(stream);
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(e0, stream);
        if (!launch_cufftdx_build_dispatch(fft_n, d_child, cps, d_parent, pps, nparents, 1.0/(double)fft_n, stream)) goto fail;
        cudaEventRecord(e1, stream); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        bsamp.push_back((double)ms * 1e6);
    }
    for (int i = 0; i < warmup; ++i)
        if (!launch_cufftdx_corr_dispatch(fft_n, d_g_parent, fft_n, len_g, d_child_poly, cps, len_P,
                                          d_g_child, fft_n, len_out, nparents, 1.0/(double)fft_n, stream)) goto fail;
    cudaStreamSynchronize(stream);
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(e0, stream);
        if (!launch_cufftdx_corr_dispatch(fft_n, d_g_parent, fft_n, len_g, d_child_poly, cps, len_P,
                                          d_g_child, fft_n, len_out, nparents, 1.0/(double)fft_n, stream)) goto fail;
        cudaEventRecord(e1, stream); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        csamp.push_back((double)ms * 1e6);
    }
    std::sort(bsamp.begin(), bsamp.end());
    std::sort(csamp.begin(), csamp.end());
    if (bsamp.empty() || csamp.empty()) goto fail;
    *build_ns_out = bsamp[bsamp.size()/2] / (double)nparents;
    *corr_ns_out = csamp[csamp.size()/2] / (double)nparents;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_g_child); cudaFree(d_child_poly); cudaFree(d_g_parent);
    cudaFree(d_parent); cudaFree(d_child); cudaStreamDestroy(stream);
    return 1;
fail:
    if (e0) cudaEventDestroy(e0); if (e1) cudaEventDestroy(e1);
    if (d_g_child) cudaFree(d_g_child); if (d_child_poly) cudaFree(d_child_poly);
    if (d_g_parent) cudaFree(d_g_parent); if (d_parent) cudaFree(d_parent);
    if (d_child) cudaFree(d_child); if (stream) cudaStreamDestroy(stream);
    *build_ns_out = 0.0; *corr_ns_out = 0.0;
    return 0;
#endif
}

int icm_gpu_measure_fused_r2c_pair_ns(int fft_n, int batch, int quick,
                                      double *build_ns_out, double *corr_ns_out) {
    if (!build_ns_out || !corr_ns_out || fft_n <= 0 || batch <= 0) return 0;
    *build_ns_out = 0.0; *corr_ns_out = 0.0;
    if (g_cuda_device < 0) { if (!icm_gpu_init(0)) return 0; }
    if (!CUDA_OK(cudaSetDevice(g_cuda_device))) return 0;
#if !ICM_HAVE_CUFFTDX || !ICM_HAVE_CUFFTDX_R2C
    (void)fft_n; (void)batch; (void)quick;
    return 0;
#else
    if (!is_cufftdx_supported_fft_n(fft_n)) return 0;
    int nparents = batch;
    int cps = fft_n, pps = fft_n, len_g = fft_n, len_P = fft_n, len_out = fft_n;
    double *d_child=0, *d_parent=0, *d_g_parent=0, *d_child_poly=0, *d_g_child=0;
    cudaStream_t stream=0; cudaEvent_t e0=0, e1=0;
    int warmup = quick ? 1 : 3, reps = quick ? 4 : 12;
    std::vector<double> bsamp, csamp;
    size_t cb = (size_t)(2*nparents)*cps*sizeof(double);
    size_t pb = (size_t)nparents*pps*sizeof(double);
    size_t gb = (size_t)nparents*fft_n*sizeof(double);
    size_t gcb = (size_t)(2*nparents)*fft_n*sizeof(double);
    if (!CUDA_OK(cudaStreamCreate(&stream))) goto r2c_fail;
    if (!CUDA_OK(cudaMalloc(&d_child, cb))) goto r2c_fail;
    if (!CUDA_OK(cudaMalloc(&d_parent, pb))) goto r2c_fail;
    if (!CUDA_OK(cudaMalloc(&d_g_parent, gb))) goto r2c_fail;
    if (!CUDA_OK(cudaMalloc(&d_child_poly, cb))) goto r2c_fail;
    if (!CUDA_OK(cudaMalloc(&d_g_child, gcb))) goto r2c_fail;
    cudaMemsetAsync(d_child, 1, cb, stream);
    cudaMemsetAsync(d_parent, 0, pb, stream);
    cudaMemsetAsync(d_g_parent, 1, gb, stream);
    cudaMemsetAsync(d_child_poly, 2, cb, stream);
    cudaMemsetAsync(d_g_child, 0, gcb, stream);
    cudaStreamSynchronize(stream);
    if (!CUDA_OK(cudaEventCreate(&e0))) goto r2c_fail;
    if (!CUDA_OK(cudaEventCreate(&e1))) goto r2c_fail;
    for (int i = 0; i < warmup; ++i)
        if (!launch_cufftdx_build_r2c_dispatch(fft_n, d_child, cps, d_parent, pps, nparents, 1.0/(double)fft_n, stream)) goto r2c_fail;
    cudaStreamSynchronize(stream);
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(e0, stream);
        if (!launch_cufftdx_build_r2c_dispatch(fft_n, d_child, cps, d_parent, pps, nparents, 1.0/(double)fft_n, stream)) goto r2c_fail;
        cudaEventRecord(e1, stream); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        bsamp.push_back((double)ms * 1e6);
    }
    for (int i = 0; i < warmup; ++i)
        if (!launch_cufftdx_corr_r2c_dispatch(fft_n, d_g_parent, fft_n, len_g, d_child_poly, cps, len_P,
                                               d_g_child, fft_n, len_out, nparents, 1.0/(double)fft_n, stream)) goto r2c_fail;
    cudaStreamSynchronize(stream);
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(e0, stream);
        if (!launch_cufftdx_corr_r2c_dispatch(fft_n, d_g_parent, fft_n, len_g, d_child_poly, cps, len_P,
                                               d_g_child, fft_n, len_out, nparents, 1.0/(double)fft_n, stream)) goto r2c_fail;
        cudaEventRecord(e1, stream); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        csamp.push_back((double)ms * 1e6);
    }
    std::sort(bsamp.begin(), bsamp.end()); std::sort(csamp.begin(), csamp.end());
    if (bsamp.empty() || csamp.empty()) goto r2c_fail;
    *build_ns_out = bsamp[bsamp.size()/2] / (double)nparents;
    *corr_ns_out = csamp[csamp.size()/2] / (double)nparents;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_g_child); cudaFree(d_child_poly); cudaFree(d_g_parent);
    cudaFree(d_parent); cudaFree(d_child); cudaStreamDestroy(stream);
    return 1;
r2c_fail:
    if (e0) cudaEventDestroy(e0); if (e1) cudaEventDestroy(e1);
    if (d_g_child) cudaFree(d_g_child); if (d_child_poly) cudaFree(d_child_poly);
    if (d_g_parent) cudaFree(d_g_parent); if (d_parent) cudaFree(d_parent);
    if (d_child) cudaFree(d_child); if (stream) cudaStreamDestroy(stream);
    *build_ns_out = 0.0; *corr_ns_out = 0.0;
    return 0;
#endif
}

int icm_gpu_measure_hbm_bandwidth_gbps(double *gbps_out) {
    if (!gbps_out) return 0;
    const size_t bytes = (size_t)1 << 30;
    void *d_a = nullptr, *d_b = nullptr;
    if (!CUDA_OK(cudaMalloc(&d_a, bytes))) return 0;
    if (!CUDA_OK(cudaMalloc(&d_b, bytes))) { cudaFree(d_a); return 0; }
    cudaMemset(d_a, 1, bytes); cudaMemset(d_b, 2, bytes);
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (!CUDA_OK(cudaEventCreate(&e0)) || !CUDA_OK(cudaEventCreate(&e1))) {
        if (e0) cudaEventDestroy(e0); if (e1) cudaEventDestroy(e1);
        cudaFree(d_a); cudaFree(d_b); return 0;
    }
    cudaEventRecord(e0);
    for (int i = 0; i < 16; ++i) cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_a); cudaFree(d_b);
    double sec = ms / 1000.0;
    double total_bytes = (double)bytes * 16.0;
    *gbps_out = (total_bytes / sec) / 1e9;
    return 1;
}

int icm_gpu_write_config_header(const char *output_path) {
    if (!output_path) return 0;
    FILE *f = fopen(output_path, "w");
    if (!f) { set_last_errorf("Cannot open %s for write", output_path); return 0; }
    double gbps = 0.0;
    icm_gpu_measure_hbm_bandwidth_gbps(&gbps);
    fprintf(f, "/* Auto-generated bootstrap GPU config. Replace with calibrate_gpu.cu output. */\n");
    fprintf(f, "#ifndef ICM_GPU_FFT_CONFIG_H\n#define ICM_GPU_FFT_CONFIG_H\n\n");
    fprintf(f, "#define GPU_N_CALIBRATED_SIZES %d\n", GPU_N_CALIBRATED_SIZES);
    fprintf(f, "static const int gpu_calib_sizes[GPU_N_CALIBRATED_SIZES] = {");
    for (int i = 0; i < GPU_N_CALIBRATED_SIZES; ++i) fprintf(f, "%s%d", (i ? "," : ""), gpu_calib_sizes[i]);
    fprintf(f, "};\n");
    fprintf(f, "static const double gpu_calib_cufft_ns[GPU_N_CALIBRATED_SIZES] = {");
    for (int i = 0; i < GPU_N_CALIBRATED_SIZES; ++i) fprintf(f, "%s%.1f", (i ? "," : ""), gpu_calib_cufft_ns[i]);
    fprintf(f, "};\n");
    fprintf(f, "static const double gpu_calib_cufftdx_build_ns[GPU_N_CALIBRATED_SIZES] = {");
    for (int i = 0; i < GPU_N_CALIBRATED_SIZES; ++i) fprintf(f, "%s%.1f", (i ? "," : ""), gpu_calib_cufftdx_build_ns[i]);
    fprintf(f, "};\n");
    fprintf(f, "static const double gpu_calib_cufftdx_corr_ns[GPU_N_CALIBRATED_SIZES] = {");
    for (int i = 0; i < GPU_N_CALIBRATED_SIZES; ++i) fprintf(f, "%s%.1f", (i ? "," : ""), gpu_calib_cufftdx_corr_ns[i]);
    fprintf(f, "};\n\n");
    fprintf(f, "#define GPU_SCHOOL_FMA_NS %.6f\n", GPU_SCHOOL_FMA_NS);
    fprintf(f, "#define GPU_FFT_OVERHEAD_NS %.6f\n", GPU_FFT_OVERHEAD_NS);
    fprintf(f, "#define GPU_HBM_BANDWIDTH %.3f\n", (gbps > 0.0 ? gbps : GPU_HBM_BANDWIDTH));
    fprintf(f, "#define GPU_FUSED_MAX_CONV_LEN %d\n", GPU_FUSED_MAX_CONV_LEN);
    fprintf(f, "#define GPU_PAIRED_CACHED_CORR_RATIO %.6f\n", GPU_PAIRED_CACHED_CORR_RATIO);
    fprintf(f, "#define GPU_INDEP_PAIR_RATIO %.6f\n", GPU_INDEP_PAIR_RATIO);
    fprintf(f, "#define GPU_BLOCK_BUILD_NS_PER_FMA %.6f\n", GPU_BLOCK_BUILD_NS_PER_FMA);
    fprintf(f, "#define GPU_LEAF_EXTRACT_NS_PER_FMA %.6f\n", GPU_LEAF_EXTRACT_NS_PER_FMA);
    fprintf(f, "#define GPU_VRAM_BYTES %lluULL\n", (unsigned long long)GPU_VRAM_BYTES);
    fprintf(f, "#define GPU_SM_COUNT %d\n", GPU_SM_COUNT);
    fprintf(f, "\n#endif\n");
    fclose(f);
    return 1;
}

}  // extern "C"
