/*
 * fft_cost_model.h: Shared FFT cost-model decision logic.
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
#include <math.h>    /* log, exp */

/* Top of WRAP_FMA_NS's validated operating range (bench_wrap_fma.c fits it
 * over wrap_m in [64,384], see CLAUDE.md). best_fft_config()/
 * best_fft_config_joint() must never choose a candidate whose wrap
 * correction falls outside this, regardless of gaps in calib_sizes[] --
 * a gap anywhere in the table (not just past the largest calibrated size)
 * can otherwise let an under-priced huge wrap_m win a cost comparison it
 * has no business winning (the exact mechanism behind the n=89,856/433.7s
 * regression, HANDOFF.md 2026-08-03). */
#ifndef WRAP_SAFE_MARGIN
#define WRAP_SAFE_MARGIN 384
#endif

/* ── Empirical linear-vs-hybrid crossover lookup ──────────────────────
 *
 * Precedent: LAPACK's ILAENV ISPEC=3 (NX) parameter -- a problem-size
 * crossover between two algorithms (blocked vs unblocked), determined by
 * direct empirical benchmarking on the target machine rather than a
 * closed-form cost model, consulted at runtime as a cheap threshold
 * comparison. No live racing of both candidates in production.
 *
 * Closed-form cost models miss microarchitectural effects that are hard
 * to represent as summed terms, even when every constant is individually
 * correct (a known result in the autotuning literature: FFTW's
 * PATIENT/MEASURE vs its ESTIMATE heuristic, ATLAS's AEOS install-time
 * search). The fix is to measure the real crossover directly
 * (tools/calibrate_crossover.c), baking the result into a small per-
 * device table.
 *
 * Requires (from the including translation unit's fft_config.h):
 *   N_CROSSOVER_POINTS
 *   crossover_n[]  (ascending problem sizes)
 *   crossover_k[]  (measured crossover k at each corresponding n)
 *
 * Scope: this covers FULL-equity dispatch only (n_targets == 0). Subset
 * queries (n_targets > 0) still use the analytical formula in
 * select_engine_ex(), since the empirical table was calibrated only for
 * the full-equity case. */
static double empirical_crossover_k(int n) {
    /* Empty-table guard: when no crossover data exists (uncalibrated device),
     * always dispatch hybrid. select_engine_ex() reads this as
     * `k < k_cross ? linear : hybrid`, so "always hybrid" is a crossover of
     * zero, not a large one. This also fixes the out-of-bounds read on
     * crossover_n[0] that would otherwise occur when N_CROSSOVER_POINTS == 0. */
    if (N_CROSSOVER_POINTS == 0) return 0.0;

    int lo = 0, hi = N_CROSSOVER_POINTS - 1;
    if (n <= crossover_n[0]) return (double)crossover_k[0];
    if (n >= crossover_n[hi]) return (double)crossover_k[hi];
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (crossover_n[mid] <= n) lo = mid; else hi = mid;
    }
    if (crossover_n[lo] == n) return (double)crossover_k[lo];
    double log_n  = log((double)n);
    double log_n0 = log((double)crossover_n[lo]);
    double log_n1 = log((double)crossover_n[hi]);
    double log_k0 = log((double)crossover_k[lo]);
    double log_k1 = log((double)crossover_k[hi]);
    double t = (log_n - log_n0) / (log_n1 - log_n0);
    return exp(log_k0 + t * (log_k1 - log_k0));
}

/* ── Empirical hybrid-engine block-size (B) lookup ────────────────────
 *
 * select_best_B() (src/cpu/icm.c) chooses which block size B in
 * {8,16,24,32,48,64} the hybrid engine uses, via 2D nearest-neighbor
 * lookup over a calibrated (n,k,B) grid (see tools/calibrate_best_b.c).
 *
 * Unlike the crossover table (a continuous threshold, log-linearly
 * interpolated), B is a discrete/categorical choice -- there is no
 * meaningful interpolation between B=32 and B=64. Lookup is nearest-
 * neighbor over a 2D (n,k) grid instead, joint (n,k) in log space,
 * single pass:
 *   d_i = hypot(log n - log bselect_n[i], log k - log bselect_k[i])
 * B is taken from the calibration point minimizing this joint distance.
 *
 * Requires (from fft_config.h): N_BSELECT_POINTS, bselect_n[],
 * bselect_k[], bselect_B[] (flat parallel arrays from
 * tools/calibrate_best_b.c; not necessarily sorted -- linear scan over
 * ~30-40 points per call is negligible). */
static int empirical_best_B(int n, int k) {
    /* Empty-table guard: when no B-selection calibration data exists
     * (uncalibrated device), return the fixed uncalibrated default B=32.
     * This also fixes the out-of-bounds read on bselect_n[0] when
     * N_BSELECT_POINTS == 0. */
    if (N_BSELECT_POINTS == 0) return 32;

    double log_n = log((double)n);
    double log_k = log((double)k);
    int best_i = 0;
    double best_dist = hypot(log_n - log((double)bselect_n[0]),
                             log_k - log((double)bselect_k[0]));
    for (int i = 1; i < N_BSELECT_POINTS; i++) {
        double d = hypot(log_n - log((double)bselect_n[i]),
                         log_k - log((double)bselect_k[i]));
        if (d < best_dist) { best_dist = d; best_i = i; }
    }
    return bselect_B[best_i];
}

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
        /* Skip candidates needing an untrustworthy wrap on EITHER side --
         * a candidate that's cheap on build but needs a huge correlate
         * wrap (or vice versa) is exactly as unsafe as one that's bad on
         * both. */
        if (mb > WRAP_SAFE_MARGIN || mc > WRAP_SAFE_MARGIN) continue;
        /* FEASIBILITY (not cost): the correlate implementations require
         * fft_n >= len_g, i.e. the g operand must fit the transform whole.
         * len_g = corr_conv - p_eff + 1, so the constraint is mc <= p_eff-1.
         * Past it, correlate_fft_cached_*_wrap silently TRUNCATES g
         * (copy_g = min(len_g, fft_n)) and its wrap corrections -- which
         * model cyclic aliasing of a g that fully fit -- produce wrong
         * output; the non-cached correlate_fft/_pair would heap-overflow.
         * This is a domain-of-validity bound on the implementations, not a
         * tunable: do not relax it without teaching the correlates to
         * restore the truncated tail terms (VERDICTS.md V20). */
        if (mc >= p_eff) continue;
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
    if (*out_size == 0) {
        /* No calibrated candidate in [min_size, 2*max_conv] has a
         * trustworthy wrap on both sides. Fall back exactly like the
         * fully-uncalibrated path: smallest smooth size >= max_conv,
         * wrap-free on both build and correlate. */
        *out_size = next_smooth_ge(max_conv);
        *out_build_m = 0;
        *out_corr_m = 0;
        int fidx;
        { int lo2=0,hi2=N_CALIBRATED_SIZES-1;
          while(lo2<hi2){int m2=(lo2+hi2)>>1;if(calib_sizes[m2]<*out_size)lo2=m2+1;else hi2=m2;}
          fidx = lo2; }
        /* If this fallback size happens to be a real calibrated entry
         * (possible when it's just past the window this loop searched),
         * use its real timing; otherwise an honest "uncalibrated, don't
         * trust this number for a joint-vs-independent choice" sentinel --
         * the caller's cost comparison will then correctly prefer whichever
         * side (joint vs independent) has real data instead of silently
         * comparing against a wrong neighboring size's timing. */
        best_cost = (fidx < N_CALIBRATED_SIZES && calib_sizes[fidx] == *out_size)
                   ? calib_times_ns[fidx] * (1.0 + PAIRED_CACHED_CORR_RATIO) : 1e18;
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
        /* Skip candidates needing an untrustworthy wrap -- see
         * WRAP_SAFE_MARGIN's comment. A wrap-free (or small-wrap)
         * candidate anywhere in the window will naturally win the cost
         * comparison anyway; this only matters when NO such candidate
         * exists nearby (an internal gap in calib_sizes[], not just past
         * the largest calibrated size). */
        if (m > WRAP_SAFE_MARGIN) continue;
        /* FEASIBILITY for correlate mode (len_P > 0): fft_n >= len_g, i.e.
         * m <= len_P - 1. Same bound as best_fft_config_joint()'s mc
         * constraint -- see the comment there (VERDICTS.md V20). Pure
         * convolution (len_P == 0, the build/polymul path) reads its wrap
         * terms directly from the original input arrays, so any m within
         * WRAP_SAFE_MARGIN is feasible there. */
        if (len_P > 0 && m >= len_P) continue;
        double correction = (len_P > 0) ? (double)m * (m + 1) * WRAP_FMA_NS
                                        : (double)m * (m + 1) / 2.0 * WRAP_FMA_NS;
        double cost = calib_times_ns[i] + correction;
        if (cost < best_cost) {
            best_cost = cost;
            *out_size = S;
            *out_wrap_m = m;
        }
    }
    if (*out_size == 0) {
        /* No calibrated candidate in [L/2+1, 2L] has a trustworthy wrap.
         * Fall back exactly like the fully-uncalibrated path: smallest
         * smooth size >= L, wrap-free. */
        *out_size = next_smooth_ge(L);
        *out_wrap_m = 0;
    }
}

#endif /* FFT_COST_MODEL_H */
