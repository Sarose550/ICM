/*
 * fft_cost_model.h — Shared FFT cost-model decision logic.
 *
 * best_fft_config() and best_fft_config_joint() are defined ONCE here
 * and shared across all CPU device targets.  They consume the per-device
 * DATA (#define constants, calib_sizes[], calib_times_ns[]) from the
 * device's fft_config.h, which must be #include'd before this header.
 *
 * This follows the FFTW/ATLAS precedent: planning logic is shared across
 * all target machines; only the calibration/wisdom DATA is machine-specific.
 *
 * Requires (from the including translation unit's fft_config.h):
 *   N_CALIBRATED_SIZES
 *   calib_sizes[]
 *   calib_times_ns[]
 *   WRAP_FMA_NS
 *   PAIRED_CACHED_CORR_RATIO
 */

#ifndef FFT_COST_MODEL_H
#define FFT_COST_MODEL_H

#include <stddef.h>  /* NULL */

/* Joint optimization of build + paired cached correlate at one shared FFT size.
 * p_eff = build_conv/2 + 1 (polynomial size at this level) for input-wrap cost. */
static double best_fft_config_joint(int build_conv, int corr_conv, int p_eff,
                                     int *out_size, int *out_build_m, int *out_corr_m) {
    int max_conv = (build_conv > corr_conv) ? build_conv : corr_conv;
    int min_size = max_conv / 2 + 1;

    int lo = 0, hi = N_CALIBRATED_SIZES - 1;
    int half = min_size;
    while (lo < hi) { int mid = (lo+hi)>>1; if (calib_sizes[mid] < half) lo = mid+1; else hi = mid; }

    double best_cost = 1e18;
    *out_size = 0; *out_build_m = 0; *out_corr_m = 0;

    for (int i = lo; i < N_CALIBRATED_SIZES; i++) {
        int S = calib_sizes[i];
        if (S > 2 * max_conv) break;
        if (S < min_size) continue;
        int mb = (S >= build_conv) ? 0 : build_conv - S;
        int mc = (S >= corr_conv) ? 0 : corr_conv - S;
        double cost = calib_times_ns[i]
                    + (double)mb*(mb+1)/2.0 * WRAP_FMA_NS
                    + calib_times_ns[i] * PAIRED_CACHED_CORR_RATIO
                    + (double)mc*(mc+1) * WRAP_FMA_NS;
        if (cost < best_cost) {
            best_cost = cost;
            *out_size = S;
            *out_build_m = mb;
            *out_corr_m = mc;
        }
    }
    return best_cost;
}

/* For a needed convolution length L, find the fastest FFT size.
 * len_P: polynomial size for input-wrap cost (pass 0 for pure convolution). */
static void best_fft_config(int L, int *out_size, int *out_wrap_m, int len_P) {
    int lo = 0, hi = N_CALIBRATED_SIZES - 1;
    int half_L = L > 1 ? L / 2 : 1;
    while (lo < hi) { int mid = (lo+hi)>>1; if (calib_sizes[mid] < half_L) lo = mid+1; else hi = mid; }

    double best_cost = 1e18;
    *out_size = 0; *out_wrap_m = 0;

    int min_size = L / 2 + 1;
    for (int i = lo; i < N_CALIBRATED_SIZES; i++) {
        int S = calib_sizes[i];
        if (S > 2 * L) break;
        if (S < min_size) continue;
        int m = (S >= L) ? 0 : L - S;
        double correction = (len_P > 0) ? (double)m * (m + 1) * WRAP_FMA_NS
                                        : (double)m * (m + 1) / 2.0 * WRAP_FMA_NS;
        double cost = calib_times_ns[i] + correction;
        if (cost < best_cost) {
            best_cost = cost;
            *out_size = S;
            *out_wrap_m = m;
        }
    }
}

#endif /* FFT_COST_MODEL_H */
