/*
 * gpu_fft_plans.cu: cuFFT plan creation and workspace estimation.
 *
 * The binding layer between this codebase and the cuFFT library: plan
 * creation (including the 64-bit API path for batch x size products that
 * overflow cuFFT's internal 32-bit indexing) and offline-calibrated
 * workspace estimation.
 *
 * Split out of gpu_plan.cu unchanged. create_cufft_plan was already called
 * across translation units from gpu_exec.cu's lazy fallback path, so this
 * boundary was proven in production before the move.
 */

#include "gpu_internal.h"

namespace icm_gpu_detail {

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

/* Check whether batch × n would overflow signed 32-bit internally in cuFFT.
 * cufftMakePlanMany takes int parameters; very large batch×size products
 * can overflow cuFFT's internal 32-bit indexing, producing
 * CUFFT_INTERNAL_ERROR (code 5).  Threshold: batch × max(n, dist) ≥ 2^31.
 *
 * At n=524,288 with qb=256 and child_batch=16: 256×16×524,288 = 2^31. */
static bool cufft_batch_would_overflow_32bit(int n, int batch, int real_dist) {
    long long span = (real_dist > n) ? real_dist : n;
    return ((long long)batch * span) > 2147483648LL;
}

/* Detection of the 64-bit cuFFT API (cufftMakePlanMany64).
 * Available since CUDA 11.0. */
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
#define ICM_HAS_CUFFT_64BIT 1
#else
#define ICM_HAS_CUFFT_64BIT 0
#endif

/* Forward declaration for 64-bit cuFFT API (may not be in all headers). */
#if ICM_HAS_CUFFT_64BIT
extern "C" cufftResult cufftMakePlanMany64(cufftHandle, int, long long*,
    long long*, long long, long long, long long*, long long, long long,
    cufftType, long long, size_t*);
#endif

bool create_cufft_plan(cufftHandle *plan, int n, int batch, bool r2c, int real_dist) {
    if (real_dist <= 0) real_dist = n;
    if (!CUFFT_OK(cufftCreate(plan))) return false;
    /* Disable auto workspace allocation; we use a shared workspace set later.
     * Without this, each plan allocates its own workspace on creation,
     * which exhausts VRAM when many large plans coexist. */
    if (!CUFFT_OK(cufftSetAutoAllocation(*plan, 0))) return false;
    int rank = 1;
    int n_arr[1] = {n};
    int cn = n / 2 + 1;
    size_t work_size = 0;
    bool use_64bit = ICM_HAS_CUFFT_64BIT && cufft_batch_would_overflow_32bit(n, batch, real_dist);

    if (r2c) {
        if (use_64bit) {
            long long n64[1] = {(long long)n};
            long long ie64[1] = {(long long)real_dist};
            long long oe64[1] = {(long long)cn};
            if (!CUFFT_OK(cufftMakePlanMany64(*plan, rank, n64,
                                              ie64, 1LL, (long long)real_dist,
                                              oe64, 1LL, (long long)cn,
                                              CUFFT_D2Z, (long long)batch,
                                              &work_size))) return false;
        } else {
            int ie[1] = {real_dist};
            int oe[1] = {cn};
            if (!CUFFT_OK(cufftMakePlanMany(*plan, rank, n_arr,
                                            ie, 1, real_dist,
                                            oe, 1, cn,
                                            CUFFT_D2Z, batch, &work_size))) return false;
        }
    } else {
        if (use_64bit) {
            long long n64[1] = {(long long)n};
            long long ie64[1] = {(long long)cn};
            long long oe64[1] = {(long long)real_dist};
            if (!CUFFT_OK(cufftMakePlanMany64(*plan, rank, n64,
                                              ie64, 1LL, (long long)cn,
                                              oe64, 1LL, (long long)real_dist,
                                              CUFFT_Z2D, (long long)batch,
                                              &work_size))) return false;
        } else {
            int ie[1] = {cn};
            int oe[1] = {real_dist};
            if (!CUFFT_OK(cufftMakePlanMany(*plan, rank, n_arr,
                                            ie, 1, cn,
                                            oe, 1, real_dist,
                                            CUFFT_Z2D, batch, &work_size))) return false;
        }
    }
    return true;
}

/* ── cuFFT workspace estimation ────────────────────────────────── */

/* Create trial cuFFT plans at candidate qb, query their work_size
 * (zero VRAM committed: auto-allocation is disabled), take the max
 * across all FFT levels, then destroy all trial plans.
 * Returns SIZE_MAX if any trial plan creation fails (infeasible qb). */
size_t estimate_cufft_workspace_bytes(GpuPlan *plan, int qb) {
    /* Primary path: offline calibration table lookup.
     * For each FFT level, look up the per-batch-unit workspace size
     * (calibrated at batch=1), multiply by this level's actual batch
     * count, and take the max across all 4 plan types (build fwd/inv,
     * corr fwd/inv).  The per-batch-unit value was verified linear
     * during calibration (measured at batch=1 and at the timing batch).
     * If any level misses the table, fall through to live probing. */
    size_t table_max_ws = 0;
    bool table_ok = true;
    for (int ell = 1; ell < plan->L; ++ell) {
        auto &lp = plan->levels[ell];
        if (!lp.use_fft || lp.tier == GPU_TIER_SCHOOLBOOK) continue;
        /* FUSED-tier cuFFTDx-supported levels execute via cuFFTDx (zero workspace);
         * skip; no cuFFT workspace contribution. */
        if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx && is_cufftdx_supported_fft_n(lp.fft_n)) continue;
        int fft_n = lp.fft_n;
        int child_batch = plan->nn[ell - 1];
        int parent_batch = plan->nn[ell];

        size_t pb_r2c = lookup_cufft_ws_bytes(fft_n, true);
        size_t pb_c2r = lookup_cufft_ws_bytes(fft_n, false);
        if (pb_r2c == SIZE_MAX || pb_c2r == SIZE_MAX) {
            table_ok = false;
            break;
        }
        size_t w0 = pb_r2c * (size_t)(qb * child_batch);       /* build fwd */
        size_t w1 = pb_c2r * (size_t)(qb * parent_batch);      /* build inv */
        size_t w2 = pb_r2c * (size_t)(qb * parent_batch);      /* corr fwd */
        size_t w3 = pb_c2r * (size_t)(qb * 2 * parent_batch);  /* corr inv */
        size_t level_max = w0;
        if (w1 > level_max) level_max = w1;
        if (w2 > level_max) level_max = w2;
        if (w3 > level_max) level_max = w3;
        if (level_max > table_max_ws) table_max_ws = level_max;
    }
    if (table_ok) return table_max_ws;

    /* Fallback: live trial-plan creation (G2, commit 2f8fd22).
     * Creates 4 throwaway cuFFT plans per FFT level at candidate qb,
     * queries work_size, takes the max, destroys all.  Kept as a
     * safety net for any fft_n not in the calibration table. */
    size_t max_ws = 0;
    for (int ell = 1; ell < plan->L; ++ell) {
        auto &lp = plan->levels[ell];
        if (!lp.use_fft || lp.tier == GPU_TIER_SCHOOLBOOK) continue;
        if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx && is_cufftdx_supported_fft_n(lp.fft_n)) continue;
        int fft_n = lp.fft_n;
        int child_batch = plan->nn[ell - 1];
        int parent_batch = plan->nn[ell];

        cufftHandle plans[4] = {0, 0, 0, 0};
        int n_created = 0;
        if (!create_cufft_plan(&plans[0], fft_n, qb * child_batch, true))   goto infeasible;
        n_created = 1;
        if (!create_cufft_plan(&plans[1], fft_n, qb * parent_batch, false)) goto infeasible;
        n_created = 2;
        if (!create_cufft_plan(&plans[2], fft_n, qb * parent_batch, true))  goto infeasible;
        n_created = 3;
        if (!create_cufft_plan(&plans[3], fft_n, qb * 2 * parent_batch, false)) goto infeasible;
        n_created = 4;

        for (int i = 0; i < 4; ++i) {
            size_t ws = 0;
            cufftGetSize(plans[i], &ws);
            if (ws > max_ws) max_ws = ws;
        }
        for (int i = 0; i < 4; ++i) cufftDestroy(plans[i]);
        continue;

    infeasible:
        for (int i = 0; i < n_created; ++i) cufftDestroy(plans[i]);
        return SIZE_MAX;
    }
    return max_ws;
}

}  // namespace icm_gpu_detail
