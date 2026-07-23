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
#include <math.h>    /* log, exp */

/* ── Empirical linear-vs-hybrid crossover lookup ──────────────────────
 *
 * Precedent: LAPACK's ILAENV ISPEC=3 (NX) parameter -- a problem-size
 * crossover between two algorithms (blocked vs unblocked), determined by
 * direct empirical benchmarking on the target machine rather than a
 * closed-form cost model, consulted at runtime as a cheap threshold
 * comparison. No live racing of both candidates in production.
 *
 * Rationale: every individual constant feeding the summed analytical
 * cost formula (calib_times_ns[], WRAP_FMA_NS, the ratio constants,
 * leaf/block/linear per-element costs) has been directly validated
 * against real embedded execution, yet the AGGREGATE go/no-go decision
 * still didn't match the true measured crossover on real hardware (this
 * was chased at length on both M3 Pro and Zen4). This matches a known
 * result in the autotuning literature (FFTW's PATIENT/MEASURE modes vs
 * its own ESTIMATE heuristic; ATLAS's AEOS install-time search): closed-
 * form cost models miss microarchitectural effects that are hard to
 * represent as summed terms, even when every constant is individually
 * correct. The fix is to stop summing terms for the FINAL decision and
 * measure the real crossover directly instead (tools/calibrate_crossover.c),
 * baking the result into a small per-device table.
 *
 * Requires (from the including translation unit's fft_config.h):
 *   N_CROSSOVER_POINTS
 *   crossover_n[]  (ascending problem sizes)
 *   crossover_k[]  (measured crossover k at each corresponding n)
 *
 * Scope: this covers FULL-equity dispatch only (n_targets == 0). Subset
 * queries (n_targets > 0) still use the analytical formula in
 * select_engine_ex(), since the empirical table was calibrated only for
 * the full-equity case and subset behavior was never measured directly --
 * revisit if subset dispatch is shown to need the same fix. */
static double empirical_crossover_k(int n) {
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
 * select_best_B() (src/icm.c) chooses which block size B in
 * {8,16,24,32,48,64} the hybrid engine uses, via the same summed-
 * analytical-constants approach as the (now-fixed) linear-vs-hybrid
 * crossover. Direct validation (tools/validate_best_b.c) confirmed the
 * SAME class of error: measurably wrong by 7-11% on M3 Pro (systematic
 * bias toward B=64 when B=32 real-wins) and 2-9% on Zen4 (bias toward
 * B=48 when B=24 real-wins) -- same root cause, same direction
 * (overestimating the benefit of larger B), as the crossover decision.
 *
 * Unlike the crossover table (a continuous threshold, log-linearly
 * interpolated), B is a discrete/categorical choice -- there is no
 * meaningful interpolation between B=32 and B=64. Lookup is nearest-
 * neighbor over a 2D (n,k) grid instead (log-distance on each axis):
 * find the calibrated n closest to the query, then among that n's
 * entries, the calibrated k closest to the query.
 *
 * Requires (from fft_config.h): N_BSELECT_POINTS, bselect_n[],
 * bselect_k[], bselect_B[] (flat parallel arrays from
 * tools/calibrate_best_b.c; not necessarily sorted -- linear scan over
 * ~30-40 points per call is negligible). */
static int empirical_best_B(int n, int k) {
    double log_n = log((double)n);
    int best_n = bselect_n[0];
    double best_n_dist = fabs(log_n - log((double)bselect_n[0]));
    for (int i = 1; i < N_BSELECT_POINTS; i++) {
        double d = fabs(log_n - log((double)bselect_n[i]));
        if (d < best_n_dist) { best_n_dist = d; best_n = bselect_n[i]; }
    }
    double log_k = log((double)k);
    int best_B = 32; /* sane fallback; overwritten below as long as the table is non-empty */
    double best_k_dist = 1e18;
    for (int i = 0; i < N_BSELECT_POINTS; i++) {
        if (bselect_n[i] != best_n) continue;
        double d = fabs(log_k - log((double)bselect_k[i]));
        if (d < best_k_dist) { best_k_dist = d; best_B = bselect_B[i]; }
    }
    return best_B;
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
