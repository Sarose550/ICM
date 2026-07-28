/* gpu_plan.cu -- Cost model, B selection, plan creation, memory allocation. */
#include "gpu_internal.h"

/* LTO callbacks disabled: cuFFT's JIT-compiled callback introduces ~1e-9
 * rounding error per level at batch >= 64 (cuFFT multi-upload codepath).
 * This compounds across tree levels to unacceptable error for FP64 ICM. */

namespace icm_gpu_detail {

/* ── Constants definition ──────────────────────────────────────── */
const int kBCandidates[MAX_B_CANDIDATES] = {
    1, 2, 4,
    8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192,
    208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512,
    576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1536, 1792,
    2048, 2560, 3072, 3584, 4096
};

/* ── Utility functions ─────────────────────────────────────────── */

int next_pow2_int(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

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

void build_smooth_table(int max_n, std::vector<int> &smooth) {
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

/* ── Plan metadata ─────────────────────────────────────────────── */

bool build_plan_metadata(GpuPlan *plan) {
    if (plan->opts.use_cufftdx) {
        g_runtime_fused_max_conv_len = fused_max_conv_len_runtime();
    } else {
        g_runtime_fused_max_conv_len = 0;
    }

    std::vector<int> smooth;
    int max_smooth = std::max(1 << 20, 2 * plan->k + 8);
    build_smooth_table(max_smooth, smooth);
    plan->k_pad = best_k_pad_gpu(plan->k, smooth);
    plan->B = gpu_select_best_B_est(plan->n, plan->k_pad, smooth);
    const char *force_b_env = getenv("ICM_GPU_FORCE_B");
    if (force_b_env && force_b_env[0]) {
        int fb = atoi(force_b_env);
        if (fb > 0 && fb <= plan->n && fb <= plan->k_pad) {
            plan->B = fb;
        }
    }
    plan->engine = gpu_select_engine_est(plan->n, plan->k_pad, plan->B, smooth);
    if (plan->engine == GPU_ENGINE_LINEAR) {
        plan->engine = GPU_ENGINE_HYBRID;
    }

    plan->nblocks = (plan->n + plan->B - 1) / plan->B;
    build_tree_geometry(plan->nblocks, plan->B, plan->k_pad, plan->B,
                        plan->nn, plan->psz, plan->plev_off,
                        plan->g_needed, plan->below_sat, plan->n_real,
                        plan->N_tree, plan->L);

    const char *force_tier_env = getenv("ICM_GPU_FORCE_TIER");
    int force_tier_mode = 0;
    if (force_tier_env && force_tier_env[0]) {
        if (strcmp(force_tier_env, "fft") == 0) force_tier_mode = 1;
        else if (strcmp(force_tier_env, "schoolbook") == 0) force_tier_mode = 2;
        else if (strcmp(force_tier_env, "fused") == 0) force_tier_mode = 3;
    }

    plan->levels.assign(plan->L, GpuLevelPlan{});
    int debug_plan = 0;
    const char *dbg_env = getenv("ICM_GPU_DEBUG_PLAN");
    if (dbg_env && dbg_env[0] && atoi(dbg_env) != 0) debug_plan = 1;
    if (debug_plan) {
        fprintf(stderr, "gpu_plan n=%d k=%d k_pad=%d B=%d L=%d nblocks=%d q_batch=%d\n",
                plan->n, plan->k, plan->k_pad, plan->B, plan->L, plan->nblocks, plan->q_batch);
    }
    for (int ell = 1; ell < plan->L; ++ell) {
        int cps = plan->psz[ell - 1];
        int pgsz = plan->psz[ell];
        int nparents = plan->nn[ell];
        int is_below = plan->below_sat[ell];
        int p_eff = is_below ? (cps / 2 + 1) : cps;
        int out_needed = plan->g_needed[ell - 1];
        int g_eff_needed = out_needed + p_eff - 1;
        /* Clamp: with the generalized below_sat trigger above, the
         * boundary level (where k_pad's cap first kicks in) can have
         * psz[ell] as low as cps+1, below the unclamped cps+cps/2 --
         * without this clamp g_eff could exceed the allocated parent
         * g-array size (pgsz), an out-of-bounds read in the correlate
         * kernels. Under the OLD strict-equality trigger this never
         * mattered (psz[ell] was always exactly 2*cps >= cps+cps/2), so
         * this clamp is a no-op there and only bites at the newly-fired
         * boundary level. */
        int g_eff_max = is_below ? std::min(cps + cps / 2, pgsz) : pgsz;
        int g_eff = std::min(g_eff_needed, g_eff_max);
        int conv_build = is_below ? (2 * (cps / 2) + 1) : (2 * cps - 1);
        int conv_corr = g_eff + p_eff - 1;
        int d_eff = is_below ? (cps / 2) : (cps - 1);
        double wrap_scale = wrap_serial_penalty_gpu(nparents);

        int bfn = 0, bwm = 0;
        best_fft_config_gpu(conv_build, 0, wrap_scale, &bfn, &bwm);
        double fft_build = estimate_cufft_pipeline_ns(bfn)
            + (double)bwm * (double)(bwm + 1) / 2.0 * tree_school_ns_per_fma() * wrap_scale;
        double school_build = (double)(d_eff + 1) * (double)(d_eff + 1) * tree_school_ns_per_fma();
        if (debug_plan) {
            fprintf(stderr,
                    "    build_choice ell=%d nparents=%d wrap_scale=%.2f bfn=%d bwm=%d fft_build=%.3fms school_build=%.3fms\n",
                    ell, nparents, wrap_scale, bfn, bwm, fft_build / 1e6, school_build / 1e6);
        }

        /* When fused is available, always use FFT path — the schoolbook
         * FMA model underestimates actual cost at small conv_len where
         * per-parent overhead dominates over raw FMAs. */
        double fused_build_check = std::numeric_limits<double>::infinity();
        if (conv_build <= g_runtime_fused_max_conv_len) {
            int fused_fft_n = next_pow2_int(conv_build);
            fused_build_check = estimate_fused_build_ns(fused_fft_n);
        }
        bool fused_avail = (conv_build <= g_runtime_fused_max_conv_len
                            && std::isfinite(fused_build_check));
        int use_fft = fused_avail || (fft_build < school_build);
        int fft_n = bfn;
        int build_wrap_m = bwm;
        int corr_wrap_m = 0;
        int cache_fft = 0;
        int tier = GPU_TIER_SCHOOLBOOK;

        if (use_fft) {
            int cfn = 0, cwm = 0;
            best_fft_config_gpu(conv_corr, p_eff, wrap_scale, &cfn, &cwm);
            fft_n = cfn;
            build_wrap_m = (cfn >= conv_build) ? 0 : (conv_build - cfn);
            corr_wrap_m = cwm;
            cache_fft = 0;

            if (ell < plan->L - 1) {
                int jfn = 0, jbm = 0, jcm = 0;
                double joint_cost = best_fft_config_joint_gpu(conv_build, conv_corr, p_eff, wrap_scale, &jfn, &jbm, &jcm);
                double indep_cost = fft_build
                    + estimate_cufft_pipeline_ns(cfn) * GPU_INDEP_PAIR_RATIO
                    + (double)cwm * (double)(cwm + 1) * tree_school_ns_per_fma() * wrap_scale;
                if (joint_cost < indep_cost) {
                    fft_n = jfn;
                    build_wrap_m = jbm;
                    corr_wrap_m = jcm;
                    cache_fft = 1;
                }
            }

            tier = pick_tier_for_fft_len(fft_n, conv_build);

            /* If the cuFFT config chose a non-fused size, check whether a
             * power-of-2 fused size would be cheaper.  Override fft_n so
             * the execution engine uses the fused kernel.
             * Also fire when tier is already GPU_TIER_FUSED but paying a
             * nonzero wrap penalty: best_fft_config_joint_gpu() may have
             * chosen a small fft_n with heavy wrap correction that looked
             * cheaper under the cuFFT cost model, but a clean power-of-2
             * fused size with zero wrap may be faster on real hardware. */
            if (conv_build <= g_runtime_fused_max_conv_len
                && (tier != GPU_TIER_FUSED || build_wrap_m > 0 || corr_wrap_m > 0)) {
                int p2 = next_pow2_int(conv_build);
                double fb = estimate_fused_build_ns(p2);
                double fc = estimate_fused_corr_ns(p2);
                if (std::isfinite(fb) && std::isfinite(fc)) {
                    int p2_bwm = (p2 >= conv_build) ? 0 : (conv_build - p2);
                    int p2_cwm = (p2 >= conv_corr) ? 0 : (conv_corr - p2);
                    double fused_total = fb + fc
                        + (double)p2_bwm * (double)(p2_bwm + 1) / 2.0 * tree_school_ns_per_fma() * wrap_scale
                        + (double)p2_cwm * (double)(p2_cwm + 1) * tree_school_ns_per_fma() * wrap_scale;
                    double current_total;
                    if (tier == GPU_TIER_FUSED) {
                        /* Already-fused: baseline is the current fused cost
                         * including its wrap penalties — compare apples-to-apples
                         * against a clean power-of-2 fused alternative. */
                        current_total = estimate_fused_build_ns(fft_n)
                            + estimate_fused_corr_ns(fft_n)
                            + (double)build_wrap_m * (double)(build_wrap_m + 1) / 2.0 * tree_school_ns_per_fma() * wrap_scale
                            + (double)corr_wrap_m * (double)(corr_wrap_m + 1) * tree_school_ns_per_fma() * wrap_scale;
                    } else {
                        current_total = estimate_cufft_pipeline_ns(fft_n)
                            + (double)build_wrap_m * (double)(build_wrap_m + 1) / 2.0 * tree_school_ns_per_fma() * wrap_scale
                            + estimate_cufft_pipeline_ns(fft_n) * GPU_PAIRED_CACHED_CORR_RATIO
                            + (double)corr_wrap_m * (double)(corr_wrap_m + 1) * tree_school_ns_per_fma() * wrap_scale;
                    }
                    if (fused_total < current_total) {
                        fft_n = p2;
                        build_wrap_m = p2_bwm;
                        corr_wrap_m = p2_cwm;
                        cache_fft = 0;
                        tier = GPU_TIER_FUSED;
                    }
                }
            }
        }

        if (force_tier_mode == 1) {
            use_fft = 1;
            tier = GPU_TIER_CUFFT;
        } else if (force_tier_mode == 2) {
            use_fft = 0;
            tier = GPU_TIER_SCHOOLBOOK;
            cache_fft = 0;
        } else if (force_tier_mode == 3) {
            use_fft = 1;
            tier = GPU_TIER_FUSED;
            cache_fft = 0;
            int ffn = 0, fbm = 0, fcm = 0;
            best_fft_config_joint_gpu(conv_build, conv_corr, p_eff, wrap_scale, &ffn, &fbm, &fcm);
            if (ffn > 0) {
                fft_n = ffn;
                build_wrap_m = fbm;
                corr_wrap_m = fcm;
            }
        }

        auto &lp = plan->levels[ell];
        lp.ell = ell;
        lp.tier = tier;
        lp.use_fft = use_fft;
        lp.cache_fft = (use_fft && cache_fft && ell < plan->L - 1 && tier != GPU_TIER_FUSED);
        lp.fft_n = fft_n;
        lp.cn = fft_n / 2 + 1;
        lp.p_eff = p_eff;
        lp.out_needed = out_needed;
        lp.g_eff = g_eff;
        lp.build_conv = conv_build;
        lp.corr_conv = conv_corr;
        lp.build_wrap_m = build_wrap_m;
        lp.corr_wrap_m = corr_wrap_m;
        if (debug_plan) {
            fprintf(stderr,
                    "  ell=%d cps=%d pgsz=%d p_eff=%d out=%d g_eff=%d build_conv=%d corr_conv=%d fft_n=%d bwm=%d cwm=%d tier=%d cache=%d use_fft=%d\n",
                    ell, cps, pgsz, p_eff, out_needed, g_eff, conv_build, conv_corr,
                    fft_n, build_wrap_m, corr_wrap_m, tier, lp.cache_fft, use_fft);
        }
    }
    choose_uncached_levels(plan);

    /* Compact fft_stride: store polynomials at psz spacing.
     * cuFFT uses a scratch buffer with gather/scatter for fft_n-strided access. */
    plan->fft_stride.assign(plan->L, 0);
    for (int ell = 0; ell < plan->L; ++ell) {
        plan->fft_stride[ell] = plan->psz[ell];
    }
    /* Ragged-tree work counts.
     *
     * nn[ell] slots exist per Q-point but only n_real[ell] of them carry
     * real data.  n_work[ell] is how many are worth computing:
     *
     *   - at least n_real[ell], so every real node is produced;
     *   - rounded up to an even count, so that a cuFFTDx launch with
     *     FFTs-per-block=2 never ends in a half-empty block (a partially
     *     retired block would leave the surviving half calling cuFFTDx's
     *     internal __syncthreads() alone).  The one extra node is a phantom
     *     built from two phantom children, i.e. identity*identity=identity,
     *     which is exactly the value the prefill already put there;
     *   - never above nn[ell] (the allocated slot count).
     *
     * n_work[0] is the real leaf-block count; phantom leaf blocks come from
     * init_pad_identity() instead of being rebuilt on every call. */
    plan->ragged_skip = 1;
    {
        const char *rag_env = getenv("ICM_GPU_RAGGED");
        if (rag_env && rag_env[0] && atoi(rag_env) == 0) plan->ragged_skip = 0;
    }
    plan->n_work.assign(plan->L, 0);
    plan->n_work[0] = std::min(plan->nblocks, plan->nn[0]);
    for (int ell = 1; ell < plan->L; ++ell) {
        int w = plan->n_real[ell] + (plan->n_real[ell] & 1);
        plan->n_work[ell] = std::min(w, plan->nn[ell]);
    }
    if (debug_plan) {
        for (int ell = 0; ell < plan->L; ++ell) {
            fprintf(stderr, "  level %d: nn=%d n_real=%d n_work=%d%s\n",
                    ell, plan->nn[ell], plan->n_real[ell], plan->n_work[ell],
                    (plan->n_work[ell] < plan->nn[ell]) ? "  (ragged: phantom tail skipped)" : "");
        }
        if (!plan->ragged_skip)
            fprintf(stderr, "  ICM_GPU_RAGGED=0: phantom-node skipping disabled\n");
    }

    if (debug_plan) {
        for (int ell = 0; ell < plan->L; ++ell) {
            fprintf(stderr, "  fft_stride[%d] = %d  (psz=%d)\n",
                    ell, plan->fft_stride[ell], plan->psz[ell]);
        }
    }

    return true;
}

/* ── choose_uncached_levels ────────────────────────────────────── */

bool choose_uncached_levels(GpuPlan *plan) {
    plan->uncached_level.assign(plan->L, 0);
    if (plan->opts.memory_strategy != 3) return true;

    auto uncache_lowest = [&](int tier, int count, int *counter) {
        if (count <= 0) return;
        for (int ell = 1; ell < plan->L - 1 && count > 0; ++ell) {
            if (!plan->levels[ell].cache_fft) continue;
            if (plan->levels[ell].tier != tier) continue;
            plan->uncached_level[ell] = 1;
            plan->levels[ell].cache_fft = 0;
            (*counter)++;
            --count;
        }
    };

    if (plan->opts.force_uncached_fused_levels >= 0 || plan->opts.force_uncached_cufft_levels >= 0) {
        uncache_lowest(GPU_TIER_FUSED, std::max(0, plan->opts.force_uncached_fused_levels),
                       &plan->uncached_fused_levels);
        uncache_lowest(GPU_TIER_CUFFT, std::max(0, plan->opts.force_uncached_cufft_levels),
                       &plan->uncached_cufft_levels);
        return true;
    }

    size_t total = 0;
    for (int ell = 1; ell < plan->L - 1; ++ell) {
        if (!plan->levels[ell].cache_fft) continue;
        int nchild = plan->nn[ell - 1];
        int cn = plan->levels[ell].cn;
        total += (size_t)nchild * (size_t)cn * sizeof(cufftDoubleComplex);
    }
    double cache_budget_frac = 0.35;
    const char *budget_env = getenv("ICM_GPU_CACHE_BUDGET_FRAC");
    if (budget_env && budget_env[0]) {
        double x = atof(budget_env);
        if (x > 0.05 && x < 0.95) cache_budget_frac = x;
    }
    size_t cache_budget = (size_t)((double)GPU_VRAM_BYTES * cache_budget_frac);
    if (total <= cache_budget) return true;

    for (int ell = 1; ell < plan->L - 1; ++ell) {
        if (!plan->levels[ell].cache_fft) continue;
        if (plan->levels[ell].tier == GPU_TIER_FUSED) {
            plan->uncached_level[ell] = 1;
            plan->levels[ell].cache_fft = 0;
            plan->uncached_fused_levels++;
            int nchild = plan->nn[ell - 1];
            int cn = plan->levels[ell].cn;
            total -= (size_t)nchild * (size_t)cn * sizeof(cufftDoubleComplex);
            if (total <= cache_budget) return true;
        }
    }
    for (int ell = 1; ell < plan->L - 1; ++ell) {
        if (!plan->levels[ell].cache_fft) continue;
        if (plan->levels[ell].tier == GPU_TIER_CUFFT) {
            plan->uncached_level[ell] = 1;
            plan->levels[ell].cache_fft = 0;
            plan->uncached_cufft_levels++;
            int nchild = plan->nn[ell - 1];
            int cn = plan->levels[ell].cn;
            total -= (size_t)nchild * (size_t)cn * sizeof(cufftDoubleComplex);
            if (total <= cache_budget) return true;
        }
    }
    return true;
}

/* ── device_sort_players ───────────────────────────────────────── */

bool device_sort_players(GpuPlan *plan) {
    std::vector<std::pair<double, int>> pairs(plan->n);
    for (int i = 0; i < plan->n; ++i) pairs[i] = {plan->S_sorted[i], i};
    std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
        if (a.first > b.first) return true;
        if (a.first < b.first) return false;
        return a.second < b.second;
    });
    std::vector<double> sorted(plan->n);
    std::vector<int> perm(plan->n), inv(plan->n);
    for (int i = 0; i < plan->n; ++i) {
        sorted[i] = pairs[i].first;
        perm[i] = pairs[i].second;
        inv[pairs[i].second] = i;
    }
    plan->S_sorted.swap(sorted);
    plan->sort_perm.swap(perm);
    plan->inv_perm.swap(inv);

    double *d_keys_in = nullptr;
    double *d_keys_out = nullptr;
    int *d_vals_in = nullptr;
    int *d_vals_out = nullptr;
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    bool ok = true;
    if (!CUDA_OK(cudaMalloc(&d_keys_in, plan->n * sizeof(double)))) ok = false;
    if (!CUDA_OK(cudaMalloc(&d_keys_out, plan->n * sizeof(double)))) ok = false;
    if (!CUDA_OK(cudaMalloc(&d_vals_in, plan->n * sizeof(int)))) ok = false;
    if (!CUDA_OK(cudaMalloc(&d_vals_out, plan->n * sizeof(int)))) ok = false;
    if (ok) {
        std::vector<int> seq(plan->n);
        std::iota(seq.begin(), seq.end(), 0);
        if (!CUDA_OK(cudaMemcpy(d_keys_in, plan->S_sorted.data(), plan->n * sizeof(double), cudaMemcpyHostToDevice))) ok = false;
        if (!CUDA_OK(cudaMemcpy(d_vals_in, seq.data(), plan->n * sizeof(int), cudaMemcpyHostToDevice))) ok = false;
    }
    if (ok) {
        if (!CUDA_OK(cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_bytes,
                                                               d_keys_in, d_keys_out,
                                                               d_vals_in, d_vals_out,
                                                               plan->n))) ok = false;
    }
    if (ok) {
        if (!CUDA_OK(cudaMalloc(&d_temp, temp_bytes))) ok = false;
    }
    if (ok) {
        if (!CUDA_OK(cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_bytes,
                                                               d_keys_in, d_keys_out,
                                                               d_vals_in, d_vals_out,
                                                               plan->n))) ok = false;
    }
    if (d_temp) cudaFree(d_temp);
    if (d_keys_in) cudaFree(d_keys_in);
    if (d_keys_out) cudaFree(d_keys_out);
    if (d_vals_in) cudaFree(d_vals_in);
    if (d_vals_out) cudaFree(d_vals_out);
    return ok;
}

/* ── cuFFT plan creation ───────────────────────────────────────── */




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
