/*
 * gpu_cost_model.cu — the GPU analytical cost model and plan-parameter choice.
 *
 * Everything that DECIDES what a plan should look like, with no side effects
 * beyond reading the calibration tables in gpu_fft_config.h: per-size FFT and
 * fused-kernel cost estimates, the FMA-rate accessors, tier dispatch, FFT-size
 * and wrap-correction selection, k-padding, tree geometry, the full candidate
 * cost estimate, and the B / engine choices built on it.
 *
 * Split out of gpu_plan.cu unchanged. Every symbol this file exports was
 * already declared in gpu_internal.h before the move -- the cost model was
 * already API-clean across translation units, it merely shared a file with its
 * callers. Nothing here allocates device memory or launches a kernel; that is
 * gpu_memory.cu and gpu_kernels.cu respectively.
 *
 * The calibration-boundary static_assert lives here, above
 * pick_tier_for_fft_len(). Read its comment before touching
 * GPU_SCHOOL_FMA_NS or GPU_FUSED_MAX_CONV_LEN.
 */

#include "gpu_internal.h"

namespace icm_gpu_detail {

/* Fallback for device configs predating GPU_UNCALIB_NS_PER_POINT. A device's
 * gpu_fft_config.h should define it; this keeps an older config building, and
 * the static_assert above pick_tier_for_fft_len() still validates whatever
 * value ends up in effect. */
#ifndef GPU_UNCALIB_NS_PER_POINT
#define GPU_UNCALIB_NS_PER_POINT 0.9
#endif

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

/* Look up offline-calibrated cuFFT workspace per-batch-unit (bytes).
 * Returns SIZE_MAX if the size is not in the table, the table is absent
 * (no GPU_HAS_CUFFT_WS_CALIB), or the size was probed but never
 * successfully calibrated (calibrate_gpu.cu stores its own SIZE_MAX-style
 * sentinel for those -- never treat that as a real 0-ish byte count).
 * Mirrors find_calib_index exactly. */
#ifdef GPU_HAS_CUFFT_WS_CALIB
static size_t lookup_cufft_ws_bytes(int fft_n, bool r2c) {
    int idx = find_calib_index(fft_n);
    if (idx < 0) return SIZE_MAX;
    unsigned long long v = r2c ? gpu_calib_r2c_ws_bytes[idx]
                                : gpu_calib_c2r_ws_bytes[idx];
    if (v == ~0ULL) return SIZE_MAX;
    return (size_t)v;
}
#else
static size_t lookup_cufft_ws_bytes(int, bool) {
    return SIZE_MAX;
}
#endif

double estimate_cufft_pipeline_ns(int fft_n) {
    int idx = find_calib_index(fft_n);
    if (idx >= 0) return gpu_calib_cufft_ns[idx] + GPU_FFT_OVERHEAD_NS;

    /* Past the calibrated ceiling (max(gpu_calib_sizes), 67108864 on B200).
     * Fall back to a LINEAR extrapolation -- deliberately linear, not the
     * O(n log n) an FFT actually costs. Two reasons, both about staying on
     * the safe side of a decision rather than being accurate:
     *
     *   1. It UNDER-estimates real FFT cost at these sizes, which biases
     *      pick_tier_for_fft_len() toward FFT over schoolbook. Choosing FFT
     *      when schoolbook was marginally better costs a constant factor;
     *      choosing schoolbook is O(conv_len^2) and unbounded. See the
     *      invariant asserted above pick_tier_for_fft_len().
     *   2. There is no honest per-size number to report here, so a
     *      cheap monotone estimate is preferable to a precise-looking one
     *      that is equally unvalidated.
     *
     * Warn once, not per call: fastest_fft_ge_gpu() calls this in a loop and
     * plan creation runs per level, so a per-call warning floods stderr
     * during a long sweep. */
    static bool warned = false;
    if (!warned) {
        warned = true;
        fprintf(stderr,
                "WARNING: FFT size %d exceeds the calibrated range (max %d). "
                "Cost estimates past the ceiling are extrapolated, so tier and "
                "size choices there are correct but not optimal. Extend the "
                "table with calibrate_gpu to remove this. "
                "(further occurrences suppressed)\n",
                fft_n, gpu_calib_sizes[GPU_N_CALIBRATED_SIZES - 1]);
    }
    return (double)fft_n * GPU_UNCALIB_NS_PER_POINT + GPU_FFT_OVERHEAD_NS;
}

/* Batch-aware cuFFT per-pipeline cost.
 *
 * The calibrated per-pipeline cost (from gpu_calib_cufft_ns) is measured at
 * a moderate batch that varies by FFT size — typically 1024 for N ≤ 2048,
 * 512 for N ≤ 8192, down to 16 for the largest sizes.  This calibration
 * batch is large enough for reasonable GPU utilisation, but in the q-batch
 * execution pipeline the effective batch per cuFFT call is q_batch × nn[ell],
 * which can reach 10 000–100 000.  At these large batches, kernel-launch
 * overhead fully amortises and the per-pipeline cost drops to a floor
 * determined by HBM bandwidth and compute throughput.
 *
 * The floor values are measured directly:
 *   (1) bench_batch.cu sweeps batch=1..65536 for R2C+C2R to find the
 *       fwd+inv floor at each FFT size.
 *   (2) level_timer.cu measures the full tree pipeline (fwd + pw + inv +
 *       scale) per level at qb=64 to get the full-pipeline floor.
 * These are stored in gpu_floor_ns[] in gpu_fft_config.h.
 *
 * The model interpolates between the calibrated cost and the floor:
 *
 *   cost(N, eb) = floor(N) + [calib(N) - floor(N)] × (calib_batch / eb)
 *
 * This is exact at eb = calib_batch (returns calib(N)) and approaches
 * floor(N) as eb → ∞.  The interpolation follows from the linear
 * decomposition: calib(N) = floor(N) + overhead(N) / calib_batch,
 * cost(N, eb) = floor(N) + overhead(N) / eb. */
#ifdef GPU_HAS_CUFFT_FLOOR
static double cufft_floor_ns(int fft_n) {
    /* Log-linear interpolation in the gpu_floor_ns table. */
    if (fft_n <= gpu_floor_fft_sizes[0])
        return gpu_floor_ns[0] * (double)fft_n / (double)gpu_floor_fft_sizes[0];
    if (fft_n >= gpu_floor_fft_sizes[GPU_N_FLOOR_SIZES - 1])
        return gpu_floor_ns[GPU_N_FLOOR_SIZES - 1]
             * (double)fft_n / (double)gpu_floor_fft_sizes[GPU_N_FLOOR_SIZES - 1];
    for (int i = 0; i < GPU_N_FLOOR_SIZES - 1; ++i) {
        if (fft_n <= gpu_floor_fft_sizes[i + 1]) {
            double log_lo = std::log((double)gpu_floor_fft_sizes[i]);
            double log_hi = std::log((double)gpu_floor_fft_sizes[i + 1]);
            double t = (std::log((double)fft_n) - log_lo) / (log_hi - log_lo);
            return gpu_floor_ns[i] + t * (gpu_floor_ns[i + 1] - gpu_floor_ns[i]);
        }
    }
    return gpu_floor_ns[GPU_N_FLOOR_SIZES - 1];
}

/* Return the calibration batch that was used when measuring gpu_calib_cufft_ns.
 * Must match tools/calibrate_gpu.cu pick_batch_for_fft_n(). */
static int calib_batch_for_size(int n) {
    if (n <= 2048) return 1024;
    if (n <= 8192) return 512;
    if (n <= 32768) return 128;
    if (n <= 65536) return 64;
    if (n <= 131072) return 16;
    return 8;
}

double estimate_cufft_pipeline_ns_batched(int fft_n, double effective_batch) {
    double calib = estimate_cufft_pipeline_ns(fft_n);
    double floor_val = cufft_floor_ns(fft_n);
    if (floor_val >= calib) return calib;  /* floor should never exceed calib */
    int cb = calib_batch_for_size(fft_n);
    double overhead_at_calib = calib - floor_val;  /* = overhead(N) / cb */
    /* overhead(N) / eb = overhead_at_calib × cb / eb */
    return floor_val + overhead_at_calib * (double)cb / std::max(1.0, effective_batch);
}
#else
double estimate_cufft_pipeline_ns_batched(int fft_n, double effective_batch) {
    (void)effective_batch;
    return estimate_cufft_pipeline_ns(fft_n);
}
#endif

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
        double correction = (double)m * (double)(m + 1) / 2.0 * fma_ns * correction_scale;
        double cost = estimate_cufft_pipeline_ns(s) + correction;
        if (cost < best_cost) {
            best_cost = cost;
            best_n = s;
            best_m = m;
        }
    }
    /* Also consider the smallest smooth FFT >= conv_len (wrap_m = 0).
     * This matters when conv_len exceeds the largest calibrated size — the
     * extrapolated cost without any wrap correction can beat a calibrated
     * size that requires large wrap. */
    if (best_m > 0) {
        int nowrap = fastest_fft_ge_gpu(conv_len);
        double nowrap_cost = estimate_cufft_pipeline_ns(nowrap);
        if (nowrap_cost < best_cost) {
            best_cost = nowrap_cost;
            best_n = nowrap;
            best_m = 0;
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
        double cost = estimate_cufft_pipeline_ns(s)
            + (double)bm * (double)(bm + 1) / 2.0 * fma_ns * correction_scale
            + estimate_cufft_pipeline_ns(s) * GPU_PAIRED_CACHED_CORR_RATIO
            + (double)cm * (double)(cm + 1) * fma_ns * correction_scale;
        if (cost < best_cost) {
            best_cost = cost;
            best_n = s;
            best_bm = bm;
            best_cm = cm;
        }
    }
    /* Also consider the no-wrap option when conv exceeds calibration range */
    if (best_bm > 0 || best_cm > 0) {
        int nowrap = fastest_fft_ge_gpu(max_conv);
        double nowrap_cost = estimate_cufft_pipeline_ns(nowrap)
            + estimate_cufft_pipeline_ns(nowrap) * GPU_PAIRED_CACHED_CORR_RATIO;
        if (nowrap_cost < best_cost) {
            best_cost = nowrap_cost;
            best_n = nowrap;
            best_bm = 0;
            best_cm = 0;
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
    /* The static_assert above pick_tier_for_fft_len() pins the compile-time
     * value; this env override can defeat it at runtime. Lowering the fused
     * ceiling below the schoolbook/cuFFT crossover would let schoolbook win
     * past the calibrated FFT ceiling, where its cost is O(conv_len^2) and
     * unbounded. Clamp rather than obey -- this knob exists for tier
     * experiments, not for changing the safety floor. */
    /* Exact positive root of c^2*F - c*U - OV = 0, the conv_len above which
     * cuFFT beats schoolbook under the past-ceiling extrapolation. */
    const double F = (double)GPU_SCHOOL_FMA_NS;
    const double U = (double)GPU_UNCALIB_NS_PER_POINT;
    const double OV = (double)GPU_FFT_OVERHEAD_NS;
    int floor_v = (int)((U + std::sqrt(U * U + 4.0 * F * OV)) / (2.0 * F)) + 1;
    if (v < floor_v) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr,
                    "WARNING: ICM_GPU_FUSED_MAX_CONV_LEN=%d is below the "
                    "calibration-boundary safety floor %d; clamping. See "
                    "pick_tier_for_fft_len() in gpu_plan.cu.\n", v, floor_v);
        }
        v = floor_v;
    }
    return v;
}

/* ── FMA cost models ───────────────────────────────────────────── */

double tree_school_ns_per_fma() {
#ifdef GPU_TREE_SCHOOL_NS_PER_FMA
    return GPU_TREE_SCHOOL_NS_PER_FMA;
#else
    /* Use the calibrated schoolbook FMA throughput — the physical rate at which
     * the GPU executes FMA instructions when the SMs are saturated.  This is
     * correct for wrap correction costs (large working set, GPU fully busy)
     * and for schoolbook tier comparisons at sizes where fused is not available
     * (conv_len > GPU_FUSED_MAX_CONV_LEN, so conv_len^2 is large enough to
     * saturate the GPU). Do not blend in GPU_BLOCK_BUILD_NS_PER_FMA here — that
     * rate reflects block-build overhead, not raw FMA throughput, and inflating
     * this value causes wrong B selection at large n where wrap-correction
     * cost dominates. */
    return GPU_SCHOOL_FMA_NS;
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

/* ── Calibration-boundary safety invariant ──────────────────────────
 *
 * The CPU side had a real bug here (fixed in commit f2ad24e): past the
 * calibrated ceiling both sides of its schoolbook-vs-FFT comparison were
 * meaningless, and schoolbook -- O(len^2) -- always won, silently. The GPU
 * does NOT share that bug, but only because of an arithmetic coincidence
 * that nothing was enforcing. This assert enforces it.
 *
 * Below, schoolbook is only ever *considered* when
 * conv_len > fused_max_conv_len_runtime(). Past the calibrated ceiling the
 * cuFFT side is the linear extrapolation from estimate_cufft_pipeline_ns(),
 * so the comparison reduces to:
 *
 *     conv_len^2 * GPU_SCHOOL_FMA_NS
 *         vs   conv_len * GPU_UNCALIB_NS_PER_POINT + GPU_FFT_OVERHEAD_NS
 *
 * The left side is quadratic and the right side linear, so if cuFFT wins at
 * the SMALLEST conv_len where schoolbook is even considered, it wins at every
 * larger one. That smallest value is GPU_FUSED_MAX_CONV_LEN, so asserting the
 * inequality there is exact and needs no square root.
 *
 * On B200 the crossover sits at conv_len ~= 5396, comfortably below
 * GPU_FUSED_MAX_CONV_LEN (8192) -- so schoolbook can never win past the
 * ceiling, and FFT is always chosen there. That is the same end state the
 * CPU fix had to be engineered to produce.
 *
 * (Do not simplify this to GPU_UNCALIB_NS_PER_POINT / GPU_SCHOOL_FMA_NS.
 * Dropping GPU_FFT_OVERHEAD_NS moves the crossover from 5396 to 5375 and
 * admits a band of conv_len where schoolbook really does win -- caught by
 * the standalone check when this was first written.)
 *
 * The danger is that this holds by numeric accident. Recalibrating
 * GPU_SCHOOL_FMA_NS downward, or lowering GPU_FUSED_MAX_CONV_LEN, would
 * silently reintroduce the CPU's failure mode on the GPU. Fail the build
 * instead. */
static_assert((double)GPU_FUSED_MAX_CONV_LEN * (double)GPU_FUSED_MAX_CONV_LEN
                  * (double)GPU_SCHOOL_FMA_NS
                  > (double)GPU_FUSED_MAX_CONV_LEN * (double)GPU_UNCALIB_NS_PER_POINT
                      + (double)GPU_FFT_OVERHEAD_NS,
              "Calibration-boundary safety broken: schoolbook could now win past "
              "the calibrated FFT ceiling, where its O(conv_len^2) cost is "
              "unbounded. Either GPU_SCHOOL_FMA_NS was recalibrated too low or "
              "GPU_FUSED_MAX_CONV_LEN was lowered. See the comment above "
              "pick_tier_for_fft_len().");

int pick_tier_for_fft_len(int fft_n, int conv_len) {
    double cufft = estimate_cufft_pipeline_ns(fft_n);
    double fused = estimate_fused_build_ns(fft_n);

    /* When fused (cuFFTDx) is calibrated at this fft_n, compare directly.
     * Both costs are calibrated per-pipeline measurements that capture the
     * full cost: data load + FFT + pointwise multiply + IFFT + store. */
    if (conv_len <= g_runtime_fused_max_conv_len && std::isfinite(fused)) {
        return (fused < cufft) ? GPU_TIER_FUSED : GPU_TIER_CUFFT;
    }

    /* Fused not calibrated at this fft_n but conv_len is within the fused
     * range — the schoolbook FMA model is unreliable here (per-parent
     * overhead dominates at small conv_len, underestimating actual cost
     * by 10-20×).  Return cuFFT; the caller's fused fallback will check
     * the power-of-2 fused option separately. */
    if (conv_len <= g_runtime_fused_max_conv_len) {
        return GPU_TIER_CUFFT;
    }

    /* Large conv_len (above fused range): FMA throughput model is accurate
     * because the GPU is fully saturated with FMAs. */
    double school = (double)conv_len * (double)conv_len * tree_school_ns_per_fma();
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
        /* below_sat[ell] means the CHILD level (ell-1) has effective
         * polynomial degree cps/2, not cps-1 -- i.e. the child was itself
         * built while psz was still doubling (not yet capped by k_pad).
         * That is a property of the child's own construction (does
         * psz[ell-1] double from its own child, psz[ell-2]?), not of
         * whether the parent also happens to double from the child.
         * Checking psz[ell] == 2*cps (the parent's doubling status) is
         * an overly strict proxy: it misses exactly the level where
         * k_pad's cap first kicks in (psz[ell-1] still doubled cleanly,
         * but psz[ell] gets capped below 2*cps), even though the child's
         * effective-degree-cps/2 property still holds there. */
        int child_doubled = (ell >= 2) ? (psz[ell - 1] == 2 * psz[ell - 2])
                                        : (psz[0] == 2 * leaf_degree);
        if (child_doubled && cps >= 2) below_sat[ell] = 1;
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
#ifdef GPU_HAS_BLOCK_BUILD_SPLIT
    /* Two-component block build cost: per-block overhead + per-FMA compute.
     * Captures the CUDA block setup cost (shared mem init, syncs) that
     * dominates at small B where each block does little FMA work. */
    double block_ns = (double)nblocks * GPU_BLOCK_BUILD_OVERHEAD_NS
                    + block_fmas * GPU_BLOCK_BUILD_BASE_FMA_NS;
    block_ns *= occ_penalty;
#else
    double block_ns = block_fmas * block_build_ns_per_fma_model() * occ_penalty;
#endif

    double fma_ns = tree_school_ns_per_fma();

    /* Estimate q_batch from VRAM budget, matching gpu_api.cu per_q_bytes.
     * Includes poly+g arrays, block prods, a_sorted, PLUS spec buffers,
     * FFT scratch, and FFT cache — all of which scale with q_batch. */
    double assumed_qb = 64.0;
    {
        size_t per_q = 0;
        for (int ell = 0; ell < L; ++ell)
            per_q += 2 * (size_t)nn[ell] * psz[ell] * sizeof(double);
        per_q += (size_t)N * (B + 1) * sizeof(double);
        per_q += 2 * (size_t)n * sizeof(double);

        /* Spec buffers, FFT scratch, and paired cache (all qb-proportional).
         * Pre-scan levels to estimate FFT sizes and tiers. */
        size_t max_cb_cn = 0, max_pb_cn = 0, max_cb_fft = 0, cache_per_q = 0;
        for (int ell = 1; ell < L; ++ell) {
            int cps_e = psz[ell - 1];
            int is_below_e = below_sat[ell];
            int conv_build_e = is_below_e ? (2 * (cps_e / 2)) : (2 * cps_e - 1);
            /* Within fused range: always FFT (fused or cuFFT, never schoolbook).
             * Above fused range: FMA model is accurate, check schoolbook vs cuFFT. */
            bool is_fft_level;
            int est_fft_n;
            if (conv_build_e <= g_runtime_fused_max_conv_len) {
                is_fft_level = true;
                est_fft_n = next_pow2_int(conv_build_e);
            } else {
                est_fft_n = fastest_fft_ge_gpu(conv_build_e);
                double school_e = (double)conv_build_e * (double)conv_build_e * fma_ns;
                is_fft_level = (estimate_cufft_pipeline_ns(est_fft_n) < school_e);
            }
            if (is_fft_level) {
                int cn_e = est_fft_n / 2 + 1;
                int cb_e = nn[ell - 1], pb_e = nn[ell];
                max_cb_cn = std::max(max_cb_cn, (size_t)cb_e * cn_e * sizeof(cufftDoubleComplex));
                max_pb_cn = std::max(max_pb_cn, (size_t)pb_e * cn_e * sizeof(cufftDoubleComplex));
                max_cb_fft = std::max(max_cb_fft, (size_t)cb_e * est_fft_n * sizeof(double));
                /* cuFFT-tier levels (above fused range, not at top) typically cache */
                if (conv_build_e > g_runtime_fused_max_conv_len && ell < L - 1)
                    cache_per_q += (size_t)cb_e * cn_e * sizeof(cufftDoubleComplex);
            }
        }
        /* 4 shared spec buffers: build(si,sm) + corr(si, sm=2×parent) */
        per_q += max_cb_cn + max_pb_cn + max_pb_cn + 2 * max_pb_cn;
        per_q += max_cb_fft;   /* FFT scratch buffer */
        per_q += cache_per_q;  /* paired FFT cache */
        /* qb>1 extras: a_qbatch[qi], inner_qbatch, block_prods_qbatch.
         * Kept in sync with gpu_api.cu per_q_bytes so the timing heuristic
         * (assumed_qb) does not drift from the real budget check. */
        per_q += (size_t)n * sizeof(double);          /* a_qbatch slot */
        per_q += (size_t)n * sizeof(double);          /* inner_qbatch share */
        per_q += (size_t)N * (B + 1) * sizeof(double); /* block_prods_qbatch */

        size_t budget = (size_t)((double)GPU_VRAM_BYTES * 0.90);
        int est_qb = (per_q > 0) ? (int)(budget / per_q) : Q_BATCH_MAX;
        if (est_qb > Q_BATCH_MAX) est_qb = Q_BATCH_MAX;
        if (est_qb < 1) est_qb = 1;
        assumed_qb = (double)est_qb;
    }
    double tree_ns = 0.0;
    for (int ell = 1; ell < L - 1; ++ell) {
        int cps = psz[ell - 1];
        int pgsz = psz[ell];
        int is_below = below_sat[ell];
        int d_eff = is_below ? (cps / 2) : (cps - 1);
        int p_eff = is_below ? (cps / 2 + 1) : cps;
        int out_needed = g_needed[ell - 1];
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
        double wrap_scale = wrap_serial_penalty_gpu(nn[ell]);
        double school_build = (double)(d_eff + 1) * (double)(d_eff + 1) * fma_ns;
        double school_corr = 2.0 * (double)p_eff * (double)out_needed * fma_ns;

        double eff_batch = assumed_qb * (double)nn[ell];

        int bfn = 0, bwm = 0;
        best_fft_config_gpu(conv_build, 0, wrap_scale, &bfn, &bwm);
        double fft_build = estimate_cufft_pipeline_ns_batched(bfn, eff_batch)
            + (double)bwm * (double)(bwm + 1) / 2.0 * fma_ns * wrap_scale;
        /* Check whether fused (cuFFTDx) is available for this level. */
        double fused_build_cost = std::numeric_limits<double>::infinity();
        if (conv_build <= g_runtime_fused_max_conv_len) {
            int fused_fft_n = next_pow2_int(conv_build);
            fused_build_cost = estimate_fused_build_ns(fused_fft_n);
        }
        /* Short-circuit to schoolbook only when fused is NOT available.
         * When fused IS available the schoolbook FMA model underestimates
         * actual cost (per-parent overhead dominates over raw FMAs at
         * small conv_len).  Proceed to the full tier comparison instead. */
        bool fused_available = (conv_build <= g_runtime_fused_max_conv_len
                                && std::isfinite(fused_build_cost));
        if (!fused_available && fft_build >= school_build) {
            tree_ns += (double)nn[ell] * (school_build + school_corr);
            continue;
        }

        int jfn = 0, jbm = 0, jcm = 0;
        double joint_cost = best_fft_config_joint_gpu(conv_build, conv_corr, p_eff, wrap_scale, &jfn, &jbm, &jcm);

        int cfn = 0, cwm = 0;
        best_fft_config_gpu(conv_corr, p_eff, wrap_scale, &cfn, &cwm);
        double indep_cost = fft_build
            + estimate_cufft_pipeline_ns_batched(cfn, eff_batch) * GPU_INDEP_PAIR_RATIO
            + (double)cwm * (double)(cwm + 1) * fma_ns * wrap_scale;

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

        /* If the cuFFT config chose a non-fused size, check whether a
         * power-of-2 fused size would be cheaper.  The fused kernel operates
         * at conv_len <= GPU_FUSED_MAX_CONV_LEN with zero-padding up to the
         * next power-of-2 — this is often faster than cuFFT at a smaller
         * non-power-of-2 size because the single-kernel fused approach
         * eliminates 5+ kernel launches.
         * Also fire when tier is already GPU_TIER_FUSED but paying a nonzero
         * wrap penalty (bwrap>0 || cwrap>0): best_fft_config_joint_gpu() may
         * have chosen a small fft_n with heavy wrap correction that looked
         * cheaper under the cuFFT cost model, but a clean power-of-2 fused
         * size with zero wrap may be faster on real hardware. */
        if (conv_build <= g_runtime_fused_max_conv_len
            && (tier != GPU_TIER_FUSED || bwrap > 0 || cwrap > 0)) {
            int p2 = next_pow2_int(conv_build);
            double fb = estimate_fused_build_ns(p2);
            double fc = estimate_fused_corr_ns(p2);
            if (std::isfinite(fb) && std::isfinite(fc)) {
                /* Wrap from padding: fft_n=p2, conv_build may exceed p2 for
                 * correlate.  Build wrap: p2 >= conv_build always (p2 is
                 * next power-of-2 of conv_build). Corr wrap: may need wrap
                 * if conv_corr > p2. */
                int p2_bwrap = (p2 >= conv_build) ? 0 : (conv_build - p2);
                int p2_cwrap = (p2 >= conv_corr) ? 0 : (conv_corr - p2);
                double fused_total = fb + fc
                    + (double)p2_bwrap * (double)(p2_bwrap + 1) / 2.0 * fma_ns * wrap_scale
                    + (double)p2_cwrap * (double)(p2_cwrap + 1) * fma_ns * wrap_scale;
                double current_total;
                if (tier == GPU_TIER_SCHOOLBOOK) {
                    current_total = school_build + school_corr;
                } else if (tier == GPU_TIER_FUSED) {
                    /* Already-fused: baseline is the current fused cost
                     * including its wrap penalties — compare apples-to-apples
                     * against a clean power-of-2 fused alternative. */
                    current_total = estimate_fused_build_ns(fft_n)
                        + estimate_fused_corr_ns(fft_n)
                        + (double)bwrap * (double)(bwrap + 1) / 2.0 * fma_ns * wrap_scale
                        + (double)cwrap * (double)(cwrap + 1) * fma_ns * wrap_scale;
                } else {
                    current_total = estimate_cufft_pipeline_ns_batched(fft_n, eff_batch)
                        + (double)bwrap * (double)(bwrap + 1) / 2.0 * fma_ns * wrap_scale
                        + estimate_cufft_pipeline_ns_batched(fft_n, eff_batch) * GPU_PAIRED_CACHED_CORR_RATIO
                        + (double)cwrap * (double)(cwrap + 1) * fma_ns * wrap_scale;
                }
                if (fused_total < current_total) {
                    fft_n = p2;
                    bwrap = p2_bwrap;
                    cwrap = p2_cwrap;
                    tier = GPU_TIER_FUSED;
                }
            }
        }

        double build_ns = 0.0;
        double corr_ns = 0.0;
        if (tier == GPU_TIER_SCHOOLBOOK) {
            build_ns = school_build;
            corr_ns = school_corr;
        } else if (tier == GPU_TIER_FUSED) {
            build_ns = estimate_fused_build_ns(fft_n);
            corr_ns = estimate_fused_corr_ns(fft_n);
            if (!std::isfinite(build_ns) || !std::isfinite(corr_ns)) {
                build_ns = estimate_cufft_pipeline_ns_batched(fft_n, eff_batch);
                corr_ns = build_ns * GPU_PAIRED_CACHED_CORR_RATIO;
            }
            build_ns += (double)bwrap * (double)(bwrap + 1) / 2.0 * fma_ns * wrap_scale;
            corr_ns += (double)cwrap * (double)(cwrap + 1) * fma_ns * wrap_scale;
        } else {
            build_ns = estimate_cufft_pipeline_ns_batched(fft_n, eff_batch)
                + (double)bwrap * (double)(bwrap + 1) / 2.0 * fma_ns * wrap_scale;
            corr_ns = estimate_cufft_pipeline_ns_batched(fft_n, eff_batch) * GPU_PAIRED_CACHED_CORR_RATIO
                + (double)cwrap * (double)(cwrap + 1) * fma_ns * wrap_scale;
        }
        tree_ns += (double)nn[ell] * (build_ns + corr_ns);
    }

    /* Per-level fixed overhead: each tree level requires at minimum 2 kernel
     * launches (build + correlate).  At ~5µs per launch on this GPU, this
     * adds ~10µs per level per q-batch.  Per Q-point: 10µs / assumed_qb.
     * This cost is small but breaks ties between B values that produce
     * trees with different depths. */
    int build_levels = std::max(0, L - 2);  /* levels 1..L-2 */
    double level_launch_ns = (double)build_levels * 10000.0 / std::max(1.0, assumed_qb);

    double leaf_fmas = (double)n * (double)B;
    double leaf_ns = leaf_fmas * leaf_extract_ns_per_fma_model() * occ_penalty;
    return block_ns + tree_ns + leaf_ns + level_launch_ns;
}

/* ── B selection / engine dispatch ─────────────────────────────── */

/* 2D nearest-neighbor lookup over the calibrated (n,k,B) grid
 * in devices/<DEVICE>/gpu_fft_config.h (gbselect_n[]/gbselect_k[]/
 * gbselect_B[], produced by tools/calibrate_gpu_best_b.c).
 * B is a discrete choice among kBCandidates, so nearest-neighbor
 * (not interpolation) is the right lookup. */
int gpu_empirical_best_B(int n, int k) {
    double log_n = log((double)n);
    int best_n = gbselect_n[0];
    double best_n_dist = fabs(log_n - log((double)gbselect_n[0]));
    for (int i = 1; i < GPU_N_BSELECT_POINTS; i++) {
        double d = fabs(log_n - log((double)gbselect_n[i]));
        if (d < best_n_dist) { best_n_dist = d; best_n = gbselect_n[i]; }
    }
    double log_k = log((double)k);
    int best_B = 64; /* sane fallback; overwritten below as long as the table is non-empty */
    double best_k_dist = 1e18;
    for (int i = 0; i < GPU_N_BSELECT_POINTS; i++) {
        if (gbselect_n[i] != best_n) continue;
        double d = fabs(log_k - log((double)gbselect_k[i]));
        if (d < best_k_dist) { best_k_dist = d; best_B = gbselect_B[i]; }
    }
    return best_B;
}

int gpu_select_best_B_est(int n, int k_pad, const std::vector<int> &smooth) {
    int emp_B = gpu_empirical_best_B(n, k_pad);
    int largest_valid = -1;
    for (int i = 0; i < MAX_B_CANDIDATES; ++i) {
        int B = kBCandidates[i];
        if (B > n || B > k_pad) continue;
        if (B == emp_B) return B;
        if (B > largest_valid) largest_valid = B;
    }
    if (largest_valid > 0) return largest_valid;

    /* No candidate fits n/k_pad at all (shouldn't happen in practice) —
     * fall back to the analytical estimate. */
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

}  // namespace icm_gpu_detail
