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

        /* When fused is available, always use FFT path; the schoolbook
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
                         * including its wrap penalties; compare apples-to-apples
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




}  // namespace icm_gpu_detail
