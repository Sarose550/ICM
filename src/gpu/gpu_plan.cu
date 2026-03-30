/* gpu_plan.cu -- Cost model, B selection, plan creation, memory allocation. */
#include "gpu_internal.h"

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

/* ── Calibration lookups ───────────────────────────────────────── */

int first_calib_ge(int n) {
    int lo = 0;
    int hi = GPU_N_CALIBRATED_SIZES - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (gpu_calib_sizes[mid] < n) lo = mid + 1;
        else hi = mid;
    }
    if (lo < GPU_N_CALIBRATED_SIZES && gpu_calib_sizes[lo] >= n) return lo;
    return -1;
}

int find_calib_index(int fft_n) {
    int lo = 0;
    int hi = GPU_N_CALIBRATED_SIZES - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (gpu_calib_sizes[mid] < fft_n) lo = mid + 1;
        else hi = mid;
    }
    if (lo < GPU_N_CALIBRATED_SIZES && gpu_calib_sizes[lo] == fft_n) return lo;
    return -1;
}

double estimate_cufft_pipeline_ns(int fft_n) {
    int idx = find_calib_index(fft_n);
    if (idx >= 0) return gpu_calib_cufft_ns[idx] + GPU_FFT_OVERHEAD_NS;
    return (double)fft_n * 0.9 + GPU_FFT_OVERHEAD_NS;
}

int fastest_fft_ge_gpu(int n) {
    if (n <= 1) return 2;
    int p2 = next_pow2_int(n);
    int i0 = first_calib_ge(n);
    if (i0 < 0) return p2;

    int best = p2;
    double best_cost = estimate_cufft_pipeline_ns(p2);
    for (int i = i0; i < GPU_N_CALIBRATED_SIZES && gpu_calib_sizes[i] <= p2; ++i) {
        int s = gpu_calib_sizes[i];
        double cost = gpu_calib_cufft_ns[i] + GPU_FFT_OVERHEAD_NS;
        if (cost < best_cost) {
            best_cost = cost;
            best = s;
        }
    }
    return best;
}

/* ── Wrap penalty ──────────────────────────────────────────────── */

double wrap_serial_penalty_gpu(int nparents) {
    int np = std::max(1, nparents);
    double p = (double)GPU_SM_COUNT / (double)np;
    if (p < 1.0) p = 1.0;
    const char *env = getenv("ICM_GPU_WRAP_SERIAL_SCALE");
    if (env && env[0]) {
        double x = atof(env);
        if (x > 0.0 && std::isfinite(x)) p *= x;
    }
    return p;
}

static int wrap_m_cap_gpu() {
    int cap = std::numeric_limits<int>::max();
    const char *env = getenv("ICM_GPU_WRAP_M_MAX");
    if (env && env[0]) {
        int v = atoi(env);
        if (v >= 0) cap = v;
    }
    return cap;
}

/* ── FFT config selection ──────────────────────────────────────── */

void best_fft_config_gpu(int conv_len, int len_P, double correction_scale,
                         int *out_fft_n, int *out_wrap_m) {
    int lo = first_calib_ge((conv_len > 1 ? conv_len / 2 + 1 : 1));
    if (lo < 0) {
        *out_fft_n = fastest_fft_ge_gpu(conv_len);
        *out_wrap_m = 0;
        return;
    }
    double fma_ns = tree_school_ns_per_fma();
    int wrap_cap = wrap_m_cap_gpu();
    double best_cost = std::numeric_limits<double>::infinity();
    int best_n = 0;
    int best_m = 0;
    int min_size = conv_len / 2 + 1;
    for (int i = lo; i < GPU_N_CALIBRATED_SIZES; ++i) {
        int s = gpu_calib_sizes[i];
        if (s > 2 * conv_len) break;
        if (s < min_size) continue;
        int m = (s >= conv_len) ? 0 : (conv_len - s);
        if (m > wrap_cap) continue;
        double correction = ((double)(m + 1) * (double)(m + 1) + (double)m * (double)len_P)
            * fma_ns * correction_scale;
        double cost = estimate_cufft_pipeline_ns(s) + correction;
        if (cost < best_cost) {
            best_cost = cost;
            best_n = s;
            best_m = m;
        }
    }
    if (best_n <= 0) {
        best_n = fastest_fft_ge_gpu(conv_len);
        best_m = 0;
    }
    *out_fft_n = best_n;
    *out_wrap_m = best_m;
}

double best_fft_config_joint_gpu(int build_conv, int corr_conv, int p_eff,
                                 double correction_scale,
                                 int *out_fft_n, int *out_build_wrap_m, int *out_corr_wrap_m) {
    int max_conv = std::max(build_conv, corr_conv);
    int lo = first_calib_ge((max_conv > 1 ? max_conv / 2 + 1 : 1));
    if (lo < 0) {
        *out_fft_n = fastest_fft_ge_gpu(max_conv);
        *out_build_wrap_m = 0;
        *out_corr_wrap_m = 0;
        return std::numeric_limits<double>::infinity();
    }

    double fma_ns = tree_school_ns_per_fma();
    int wrap_cap = wrap_m_cap_gpu();
    double best_cost = std::numeric_limits<double>::infinity();
    int best_n = 0;
    int best_bm = 0;
    int best_cm = 0;
    int min_size = max_conv / 2 + 1;
    for (int i = lo; i < GPU_N_CALIBRATED_SIZES; ++i) {
        int s = gpu_calib_sizes[i];
        if (s > 2 * max_conv) break;
        if (s < min_size) continue;
        int bm = (s >= build_conv) ? 0 : (build_conv - s);
        int cm = (s >= corr_conv) ? 0 : (corr_conv - s);
        if (bm > wrap_cap || cm > wrap_cap) continue;
        double corr_input_wrap = (double)cm * (double)p_eff;
        double cost = estimate_cufft_pipeline_ns(s)
            + (double)(bm + 1) * (double)(bm + 1) * fma_ns * correction_scale
            + estimate_cufft_pipeline_ns(s) * GPU_PAIRED_CACHED_CORR_RATIO
            + 2.0 * ((double)(cm + 1) * (double)(cm + 1) + (double)corr_input_wrap)
                * fma_ns * correction_scale;
        if (cost < best_cost) {
            best_cost = cost;
            best_n = s;
            best_bm = bm;
            best_cm = cm;
        }
    }
    if (best_n <= 0) {
        best_n = fastest_fft_ge_gpu(max_conv);
        best_bm = 0;
        best_cm = 0;
    }
    *out_fft_n = best_n;
    *out_build_wrap_m = best_bm;
    *out_corr_wrap_m = best_cm;
    return best_cost;
}

/* ── Fused cost estimates ──────────────────────────────────────── */

double estimate_fused_build_ns(int fft_n) {
    int idx = find_calib_index(fft_n);
    if (idx < 0) return std::numeric_limits<double>::infinity();
#ifdef GPU_HAS_R2C_CALIB
    if (gpu_calib_cufftdx_r2c_build_ns[idx] > 0.0)
        return gpu_calib_cufftdx_r2c_build_ns[idx];
#endif
    if (gpu_calib_cufftdx_build_ns[idx] <= 0.0) return std::numeric_limits<double>::infinity();
    return gpu_calib_cufftdx_build_ns[idx];
}

double estimate_fused_corr_ns(int fft_n) {
    int idx = find_calib_index(fft_n);
    if (idx < 0) return std::numeric_limits<double>::infinity();
#ifdef GPU_HAS_R2C_CALIB
    if (gpu_calib_cufftdx_r2c_corr_ns[idx] > 0.0)
        return gpu_calib_cufftdx_r2c_corr_ns[idx];
#endif
    if (gpu_calib_cufftdx_corr_ns[idx] <= 0.0) return std::numeric_limits<double>::infinity();
    return gpu_calib_cufftdx_corr_ns[idx];
}

int fused_max_conv_len_runtime() {
    int v = GPU_FUSED_MAX_CONV_LEN;
    const char *env = getenv("ICM_GPU_FUSED_MAX_CONV_LEN");
    if (env && env[0]) {
        int x = atoi(env);
        if (x >= 0) v = x;
    }
    return v;
}

/* ── FMA cost models ───────────────────────────────────────────── */

double tree_school_ns_per_fma() {
#ifdef GPU_TREE_SCHOOL_NS_PER_FMA
    return GPU_TREE_SCHOOL_NS_PER_FMA;
#else
    return std::max(GPU_SCHOOL_FMA_NS, GPU_BLOCK_BUILD_NS_PER_FMA);
#endif
}

static double model_ns_per_fma_override(const char *env_name, double fallback) {
    const char *v = getenv(env_name);
    if (!v || !v[0]) return fallback;
    double x = atof(v);
    if (!(x > 0.0) || !std::isfinite(x)) return fallback;
    return x;
}

double block_build_ns_per_fma_model() {
    double base = GPU_BLOCK_BUILD_NS_PER_FMA;
    if (!(base > 0.0) || !std::isfinite(base)) base = std::max(GPU_SCHOOL_FMA_NS, 1e-3);
    return model_ns_per_fma_override("ICM_GPU_BLOCK_FMA_NS", base);
}

double leaf_extract_ns_per_fma_model() {
    double base = GPU_LEAF_EXTRACT_NS_PER_FMA;
    if (!(base > 0.0) || !std::isfinite(base)) base = std::max(GPU_SCHOOL_FMA_NS, 1e-3);
    return model_ns_per_fma_override("ICM_GPU_LEAF_FMA_NS", base);
}

/* ── Tier selection ────────────────────────────────────────────── */

int pick_tier_for_fft_len(int fft_n, int conv_len) {
    double school = (double)conv_len * (double)conv_len * tree_school_ns_per_fma();
    double cufft = estimate_cufft_pipeline_ns(fft_n);
    double fused = estimate_fused_build_ns(fft_n);
    if (conv_len <= g_runtime_fused_max_conv_len && fused < school && fused < cufft) return GPU_TIER_FUSED;
    if (school < cufft) return GPU_TIER_SCHOOLBOOK;
    return GPU_TIER_CUFFT;
}

/* ── k-pad ─────────────────────────────────────────────────────── */

int best_k_pad_gpu(int k, const std::vector<int> &smooth) {
    if (k <= 2) return k;
    if ((k & (k - 1)) == 0) return k;
    int ceil_k = k + k / 8;
    if (ceil_k < k + 4) ceil_k = k + 4;
    auto it = std::lower_bound(smooth.begin(), smooth.end(), k);
    int best = k;
    double best_cost = std::numeric_limits<double>::infinity();
    for (; it != smooth.end() && *it <= ceil_k; ++it) {
        int kp = *it;
        int conv_len = 2 * kp - 1;
        int fft_n = fastest_fft_ge_gpu(conv_len);
        double cost = estimate_cufft_pipeline_ns(fft_n) + 0.5 * (double)(kp - k);
        if (cost < best_cost) {
            best_cost = cost;
            best = kp;
        }
    }
    return best;
}

/* ── Tree geometry ─────────────────────────────────────────────── */

void build_tree_geometry(int n_leaves, int leaf_degree, int k_pad,
                         int leaf_extract, std::vector<int> &nn,
                         std::vector<int> &psz, std::vector<size_t> &plev_off,
                         std::vector<int> &g_needed, std::vector<int> &below_sat,
                         std::vector<int> &n_real, int &N, int &L) {
    N = 1;
    while (N < n_leaves) N <<= 1;
    L = 0;
    int tmp = N;
    while (tmp > 1) {
        tmp >>= 1;
        ++L;
    }
    ++L;
    nn.assign(L, 0);
    psz.assign(L, 0);
    plev_off.assign(L, 0);
    g_needed.assign(L, 0);
    below_sat.assign(L, 0);
    n_real.assign(L, 0);

    n_real[0] = n_leaves;
    for (int ell = 1; ell < L; ++ell) n_real[ell] = (n_real[ell - 1] + 1) / 2;

    size_t off = 0;
    for (int ell = 0; ell < L; ++ell) {
        nn[ell] = N >> ell;
        long d = (long)leaf_degree * (1L << (ell + 1));
        psz[ell] = (d > k_pad) ? k_pad : (int)d;
        plev_off[ell] = off;
        off += (size_t)nn[ell] * (size_t)psz[ell];
    }

    g_needed[0] = std::min(leaf_extract, psz[0]);
    for (int ell = 1; ell < L; ++ell) {
        int need = g_needed[ell - 1] + psz[ell - 1] - 1;
        g_needed[ell] = std::min(need, psz[ell]);
    }
    for (int ell = 1; ell < L; ++ell) {
        int cps = psz[ell - 1];
        if (psz[ell] == 2 * cps && cps >= 2) below_sat[ell] = 1;
    }
}

/* ── Candidate cost estimation ─────────────────────────────────── */

double estimate_candidate_cost(int n, int k_pad, int B, const std::vector<int> &smooth) {
    (void)smooth;
    if (B > n || B > k_pad) return std::numeric_limits<double>::infinity();

    int nblocks = (n + B - 1) / B;
    int N = 0;
    int L = 0;
    std::vector<int> nn, psz, g_needed, below_sat, n_real;
    std::vector<size_t> plev_off;
    build_tree_geometry(nblocks, B, k_pad, B, nn, psz, plev_off, g_needed, below_sat, n_real, N, L);

    double nblocks_real = (double)n / (double)B;
    double occ_penalty = 1.0;
    if (nblocks_real < (double)GPU_SM_COUNT) occ_penalty = (double)GPU_SM_COUNT / std::max(1.0, nblocks_real);

    double block_fmas = ((double)n / B) * ((double)B * (B + 1) / 2.0);
    double block_ns = block_fmas * block_build_ns_per_fma_model() * occ_penalty;

    double fma_ns = tree_school_ns_per_fma();
    double tree_ns = 0.0;
    for (int ell = 1; ell < L - 1; ++ell) {
        int cps = psz[ell - 1];
        int pgsz = psz[ell];
        int is_below = below_sat[ell];
        int d_eff = is_below ? (cps / 2) : (cps - 1);
        int p_eff = is_below ? (cps / 2 + 1) : cps;
        int out_needed = g_needed[ell - 1];
        int g_eff_needed = out_needed + p_eff - 1;
        int g_eff_max = is_below ? (cps + cps / 2) : pgsz;
        int g_eff = std::min(g_eff_needed, g_eff_max);

        int conv_build = is_below ? (2 * (cps / 2)) : (2 * cps - 1);
        int conv_corr = g_eff + p_eff - 1;
        double wrap_scale = wrap_serial_penalty_gpu(nn[ell]);
        double school_build = (double)(d_eff + 1) * (double)(d_eff + 1) * fma_ns;
        double school_corr = 2.0 * (double)p_eff * (double)out_needed * fma_ns;

        int bfn = 0, bwm = 0;
        best_fft_config_gpu(conv_build, 0, wrap_scale, &bfn, &bwm);
        double fft_build = estimate_cufft_pipeline_ns(bfn)
            + (double)(bwm + 1) * (double)(bwm + 1) * fma_ns * wrap_scale;
        if (fft_build >= school_build) {
            tree_ns += (double)nn[ell] * (school_build + school_corr);
            continue;
        }

        int jfn = 0, jbm = 0, jcm = 0;
        double joint_cost = best_fft_config_joint_gpu(conv_build, conv_corr, p_eff, wrap_scale, &jfn, &jbm, &jcm);

        int cfn = 0, cwm = 0;
        best_fft_config_gpu(conv_corr, p_eff, wrap_scale, &cfn, &cwm);
        double indep_cost = fft_build
            + estimate_cufft_pipeline_ns(cfn) * GPU_INDEP_PAIR_RATIO
            + 2.0 * ((double)(cwm + 1) * (double)(cwm + 1) + (double)cwm * (double)p_eff)
                * fma_ns * wrap_scale;

        int fft_n = 0;
        int bwrap = 0;
        int cwrap = 0;
        if (joint_cost < indep_cost) {
            fft_n = jfn;
            bwrap = jbm;
            cwrap = jcm;
        } else {
            fft_n = cfn;
            bwrap = (cfn >= conv_build) ? 0 : (conv_build - cfn);
            cwrap = cwm;
        }

        int tier = pick_tier_for_fft_len(fft_n, conv_build);
        double build_ns = 0.0;
        double corr_ns = 0.0;
        if (tier == GPU_TIER_SCHOOLBOOK) {
            build_ns = school_build;
            corr_ns = school_corr;
        } else if (tier == GPU_TIER_FUSED) {
            build_ns = estimate_fused_build_ns(fft_n);
            corr_ns = estimate_fused_corr_ns(fft_n);
            if (!std::isfinite(build_ns) || !std::isfinite(corr_ns)) {
                build_ns = estimate_cufft_pipeline_ns(fft_n);
                corr_ns = build_ns * GPU_PAIRED_CACHED_CORR_RATIO;
            }
            build_ns += (double)(bwrap + 1) * (double)(bwrap + 1) * fma_ns * wrap_scale;
            corr_ns += 2.0 * ((double)(cwrap + 1) * (double)(cwrap + 1) + (double)cwrap * (double)p_eff)
                * fma_ns * wrap_scale;
        } else {
            build_ns = estimate_cufft_pipeline_ns(fft_n)
                + (double)(bwrap + 1) * (double)(bwrap + 1) * fma_ns * wrap_scale;
            corr_ns = estimate_cufft_pipeline_ns(fft_n) * GPU_PAIRED_CACHED_CORR_RATIO
                + 2.0 * ((double)(cwrap + 1) * (double)(cwrap + 1) + (double)cwrap * (double)p_eff)
                    * fma_ns * wrap_scale;
        }
        tree_ns += (double)nn[ell] * (build_ns + corr_ns);
    }

    double leaf_fmas = (double)n * (double)B;
    double leaf_ns = leaf_fmas * leaf_extract_ns_per_fma_model() * occ_penalty;
    return block_ns + tree_ns + leaf_ns;
}

/* ── B selection / engine dispatch ─────────────────────────────── */

int gpu_select_best_B_est(int n, int k_pad, const std::vector<int> &smooth) {
    CandidateCost best{};
    best.B = 16;
    for (int i = 0; i < MAX_B_CANDIDATES; ++i) {
        int B = kBCandidates[i];
        if (B > n || B > k_pad) continue;
        double c = estimate_candidate_cost(n, k_pad, B, smooth);
        if (c < best.total_ns) {
            best.total_ns = c;
            best.B = B;
        }
    }
    return best.B;
}

int gpu_select_engine_est(int n, int k_pad, int B, const std::vector<int> &smooth) {
    if (n < 16 || k_pad < 4) return GPU_ENGINE_LINEAR;
    double linear_fma_ns = std::max(GPU_SCHOOL_FMA_NS, block_build_ns_per_fma_model());
    double linear_ns = (double)n * (double)k_pad * 2.0 * linear_fma_ns;
    double hybrid_ns = estimate_candidate_cost(n, k_pad, B, smooth);
    return (hybrid_ns < linear_ns) ? GPU_ENGINE_HYBRID : GPU_ENGINE_LINEAR;
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
        int g_eff_max = is_below ? (cps + cps / 2) : pgsz;
        int g_eff = std::min(g_eff_needed, g_eff_max);
        int conv_build = is_below ? (2 * (cps / 2)) : (2 * cps - 1);
        int conv_corr = g_eff + p_eff - 1;
        int d_eff = is_below ? (cps / 2) : (cps - 1);
        double wrap_scale = wrap_serial_penalty_gpu(nparents);

        int bfn = 0, bwm = 0;
        best_fft_config_gpu(conv_build, 0, wrap_scale, &bfn, &bwm);
        double fft_build = estimate_cufft_pipeline_ns(bfn)
            + (double)(bwm + 1) * (double)(bwm + 1) * tree_school_ns_per_fma() * wrap_scale;
        double school_build = (double)(d_eff + 1) * (double)(d_eff + 1) * tree_school_ns_per_fma();
        if (debug_plan) {
            fprintf(stderr,
                    "    build_choice ell=%d nparents=%d wrap_scale=%.2f bfn=%d bwm=%d fft_build=%.3fms school_build=%.3fms\n",
                    ell, nparents, wrap_scale, bfn, bwm, fft_build / 1e6, school_build / 1e6);
        }

        int use_fft = (fft_build < school_build);
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
                    + 2.0 * ((double)(cwm + 1) * (double)(cwm + 1) + (double)cwm * (double)p_eff)
                        * tree_school_ns_per_fma() * wrap_scale;
                if (joint_cost < indep_cost) {
                    fft_n = jfn;
                    build_wrap_m = jbm;
                    corr_wrap_m = jcm;
                    cache_fft = 1;
                }
            }

            tier = pick_tier_for_fft_len(fft_n, conv_build);
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

    /* Compute fft_stride */
    plan->fft_stride.assign(plan->L, 0);
    for (int ell = 0; ell < plan->L; ++ell) {
        int s = plan->psz[ell];
        if (ell >= 1 && plan->levels[ell].use_fft &&
            plan->levels[ell].tier != GPU_TIER_SCHOOLBOOK) {
            s = std::max(s, plan->levels[ell].fft_n);
        }
        if (ell + 1 < plan->L && plan->levels[ell + 1].use_fft &&
            plan->levels[ell + 1].tier != GPU_TIER_SCHOOLBOOK) {
            s = std::max(s, plan->levels[ell + 1].fft_n);
        }
        plan->fft_stride[ell] = s;
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

bool create_cufft_plan(cufftHandle *plan, int n, int batch, bool r2c, int real_dist) {
    if (real_dist <= 0) real_dist = n;
    if (!CUFFT_OK(cufftCreate(plan))) return false;
    int rank = 1;
    int n_arr[1] = {n};
    int cn = n / 2 + 1;
    size_t work_size = 0;
    if (r2c) {
        int ie[1] = {real_dist};
        int oe[1] = {cn};
        if (!CUFFT_OK(cufftMakePlanMany(*plan, rank, n_arr,
                                        ie, 1, real_dist,
                                        oe, 1, cn,
                                        CUFFT_D2Z, batch, &work_size))) return false;
    } else {
        int ie[1] = {cn};
        int oe[1] = {real_dist};
        if (!CUFFT_OK(cufftMakePlanMany(*plan, rank, n_arr,
                                        ie, 1, cn,
                                        oe, 1, real_dist,
                                        CUFFT_Z2D, batch, &work_size))) return false;
    }
    return true;
}

#if ICM_HAVE_VKFFT
/* ── VkFFT plan helpers ───────────────────────────────────────── */

bool should_use_vkfft(int fft_n) {
    /* Only use VkFFT for tier-3 sizes (> GPU_FUSED_MAX_CONV_LEN) where calibration says VkFFT wins */
    if (fft_n <= 4096) return false;
    int idx = find_calib_index(fft_n);
    if (idx < 0) return false;
    return gpu_calib_lib[idx] == 1;
}

/* Note: stream_ptr must point to a persistent cudaStream_t (e.g., &plan->stream_compute).
 * VkFFT stores the pointer, not a copy, so it must remain valid for the app's lifetime. */
static int s_vkfft_cuda_init_done = 0;

bool create_vkfft_r2c_plan(VkFFTApplication *app, int n, int batch, int stride, cudaStream_t *stream_ptr) {
    if (!s_vkfft_cuda_init_done) {
        cuInit(0);
        s_vkfft_cuda_init_done = 1;
    }
    /* Buffer covers batch * stride doubles (real side). VkFFT reads/writes
     * at stride spacing, matching our fft_stride polynomial layout — no
     * gather/scatter needed. */
    int real_stride = (stride > 0) ? stride : n;
    uint64_t buf_size = (uint64_t)real_stride * batch * sizeof(double);

    VkFFTConfiguration config = {};
    config.FFTdim = 1;
    config.size[0] = (uint64_t)n;
    config.numberBatches = (uint64_t)batch;
    config.doublePrecision = 1;
    config.performR2C = 1;
    config.bufferSize = &buf_size;
    config.bufferNum = 1;
    /* Set stride to match our fft_stride layout so VkFFT operates
     * directly on poly_levels/g_levels without gather/scatter. */
    config.bufferStride[0] = (uint64_t)real_stride;

    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);
    config.device = &cuDevice;
    config.stream = stream_ptr;
    config.num_streams = 1;

    VkFFTResult res = initializeVkFFT(app, config);
    if (res != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFT init failed for n=%d batch=%d stride=%d: error %d\n",
                n, batch, real_stride, (int)res);
        return false;
    }
    return true;
}

void destroy_vkfft_app(VkFFTApplication *app) {
    deleteVkFFT(app);
}

/* Gather strided real data into contiguous buffer for VkFFT in-place R2C.
 * src[batch_idx * src_stride + j] -> dst[batch_idx * (cn*2) + j] for j in [0, fft_n)
 * where cn = fft_n/2+1, padded stride for in-place R2C = cn*2.
 * Zeroes padding element at j = fft_n..cn*2-1. */
__global__ void k_gather_strided(const double *src, int src_stride, double *dst,
                                 int fft_n, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cn = fft_n / 2 + 1;
    int dst_stride = cn * 2;  /* in-place R2C stride in doubles */
    int total = batch * dst_stride;
    if (idx >= total) return;
    int b = idx / dst_stride;
    int j = idx % dst_stride;
    if (j < fft_n) {
        dst[idx] = src[b * src_stride + j];
    } else {
        dst[idx] = 0.0;  /* zero padding for in-place R2C */
    }
}

/* Scatter contiguous VkFFT C2R output back to strided layout.
 * src[batch_idx * (cn*2) + j] -> dst[batch_idx * dst_stride + j] for j in [0, valid_len)
 * Also zeroes dst[j] for j in [valid_len, dst_stride) if needed. */
__global__ void k_scatter_strided(const double *src, int fft_n, double *dst,
                                  int dst_stride, int valid_len, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dst_stride;
    if (idx >= total) return;
    int b = idx / dst_stride;
    int j = idx % dst_stride;
    int cn = fft_n / 2 + 1;
    int src_stride = cn * 2;
    if (j < valid_len && j < fft_n) {
        dst[idx] = src[b * src_stride + j];
    } else {
        dst[idx] = 0.0;
    }
}
#endif /* ICM_HAVE_VKFFT */

/* ── allocate_plan_device_memory ───────────────────────────────── */

static bool maybe_init_mem_pool(GpuPlan *plan) {
    if (!plan->opts.enable_graphs && plan->opts.memory_strategy != 2 && plan->opts.memory_strategy != 3) {
        plan->use_async_pool = false;
        return true;
    }
    cudaMemPool_t pool = nullptr;
    if (!CUDA_OK(cudaDeviceGetDefaultMemPool(&pool, g_cuda_device))) return false;
    uint64_t threshold = GPU_VRAM_BYTES;
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

    int child_stride = plan->fft_stride[ell - 1];
    int parent_stride = plan->fft_stride[ell];

    auto &b = plan->build_fft[ell];
    b.fft_n = fft_n;
    b.cn = cn;
    b.batch_fwd = qb * child_batch;
    b.batch_inv = qb * parent_batch;
    b.real_in = nullptr;
    b.spec_in = plan->shared_build_work.spec_in;
    b.spec_mid = plan->shared_build_work.spec_mid;
    b.real_out = nullptr;
    if (!create_cufft_plan(&b.plan_fwd, fft_n, qb * child_batch, true, child_stride)) return false;
    if (!create_cufft_plan(&b.plan_inv, fft_n, qb * parent_batch, false, parent_stride)) return false;
    if (!CUFFT_OK(cufftSetStream(b.plan_fwd, plan->stream_compute))) return false;
    if (!CUFFT_OK(cufftSetStream(b.plan_inv, plan->stream_compute))) return false;

#if ICM_HAVE_VKFFT
    /* Create VkFFT R2C plans if calibration says VkFFT wins for this size.
     * VkFFT reads/writes at bufferStride spacing, matching our fft_stride
     * layout — no gather/scatter needed. */
    if (lp.tier == GPU_TIER_CUFFT && should_use_vkfft(fft_n)) {
        /* Build: fwd reads children at child_stride, inv writes parents at parent_stride */
        if (create_vkfft_r2c_plan(&b.vkfft_app_fwd, fft_n, qb * child_batch, child_stride, &plan->stream_compute)) {
            b.vkfft_fwd_initialized = 1;
            if (create_vkfft_r2c_plan(&b.vkfft_app_inv, fft_n, qb * parent_batch, parent_stride, &plan->stream_compute)) {
                b.vkfft_inv_initialized = 1;
                b.use_vkfft = 1;
            } else {
                /* Inv failed, clean up fwd and fall back to cuFFT */
                destroy_vkfft_app(&b.vkfft_app_fwd);
                b.vkfft_fwd_initialized = 0;
            }
        }
        /* If VkFFT init failed, cuFFT plans are already created as fallback */
    }
#endif

    auto &c = plan->corr_fft[ell];
    c.fft_n = fft_n;
    c.cn = cn;
    c.batch_fwd = qb * parent_batch;
    c.batch_inv = qb * 2 * parent_batch;
    c.real_in = nullptr;
    c.spec_in = plan->shared_corr_work.spec_in;
    c.spec_mid = plan->shared_corr_work.spec_mid;
    c.real_out = nullptr;
    if (!create_cufft_plan(&c.plan_fwd, fft_n, qb * parent_batch, true, parent_stride)) return false;
    if (!create_cufft_plan(&c.plan_inv, fft_n, qb * 2 * parent_batch, false, child_stride)) return false;
    if (!CUFFT_OK(cufftSetStream(c.plan_fwd, plan->stream_compute))) return false;
    if (!CUFFT_OK(cufftSetStream(c.plan_inv, plan->stream_compute))) return false;

#if ICM_HAVE_VKFFT
    /* Create VkFFT plans for correlate if build also uses VkFFT */
    if (b.use_vkfft) {
        /* Corr: fwd = parent_batch signals, inv = 2*parent_batch signals */
        /* Corr: fwd reads g at parent_stride, inv writes children at child_stride */
        if (create_vkfft_r2c_plan(&c.vkfft_app_fwd, fft_n, qb * parent_batch, parent_stride, &plan->stream_compute)) {
            c.vkfft_fwd_initialized = 1;
            if (create_vkfft_r2c_plan(&c.vkfft_app_inv, fft_n, qb * 2 * parent_batch, child_stride, &plan->stream_compute)) {
                c.vkfft_inv_initialized = 1;
                c.use_vkfft = 1;
            } else {
                destroy_vkfft_app(&c.vkfft_app_fwd);
                c.vkfft_fwd_initialized = 0;
            }
        }
    }
#endif

    if (lp.cache_fft && !plan->d_fft_cache[ell]) {
        size_t bytes_cache = (size_t)qb * (size_t)child_batch * (size_t)cn * sizeof(cufftDoubleComplex);
        if (!alloc_device(plan, (void **)&plan->d_fft_cache[ell], bytes_cache, plan->stream_compute)) return false;
    }
    return true;
}

bool allocate_plan_device_memory(GpuPlan *plan) {
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
    #undef A

    char *arena = nullptr;
    if (arena_sz == 0) { set_last_errorf("Arena size is 0"); return false; }
    if (!CUDA_OK(cudaMalloc(&arena, arena_sz))) return false;
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
    #undef P
    plan->use_async_pool = false;
    for (int ell = 1; ell < plan->L; ++ell) {
        if (!allocate_level_buffers(plan, ell, {})) return false;
    }

#if ICM_HAVE_VKFFT
    {
        int n_vkfft = 0, n_cufft_only = 0;
        for (int ell = 1; ell < plan->L; ++ell) {
            if (!plan->levels[ell].use_fft || plan->levels[ell].tier == GPU_TIER_SCHOOLBOOK) continue;
            if (plan->levels[ell].tier != GPU_TIER_CUFFT) continue;
            if (plan->build_fft[ell].use_vkfft) n_vkfft++;
            else n_cufft_only++;
        }
        if (n_vkfft > 0) {
            fprintf(stderr, "VkFFT dual-dispatch: %d tier-3 levels use VkFFT, %d use cuFFT\n",
                    n_vkfft, n_cufft_only);
        }
    }
#endif

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
            if (!alloc_device(plan, &plan->shared_cufft_workspace, max_ws, plan->stream_compute)) return false;
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
