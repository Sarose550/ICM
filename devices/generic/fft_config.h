/*
 * devices/generic/fft_config.h: Uncalibrated fallback device stub.
 *
 * This provides the absolute minimum for the code to compile when no
 * device-specific calibration exists. Every constant marked "compile-only"
 * is guarded by CALIBRATED_MAX_CONV_LEN == -1: the calibration-boundary
 * checks in icm.c and fft_cost_model.h prevent any execution path from
 * reading them at runtime on this path.
 *
 * To get real performance, run:
 *   ./tools/calibrate_full.sh <DEVICE>
 * and then build with DEVICE=<DEVICE>.
 */

#ifndef ICM_GENERIC_FFT_CONFIG_H
#define ICM_GENERIC_FFT_CONFIG_H

/* ── Calibration ceiling ─────────────────────────────────────────
 * -1 = never calibrated.  Every tree level skips the schoolbook-vs-FFT
 * cost comparison and always uses FFT with FFTW_ESTIMATE plans.  Results
 * are CORRECT but UNOPTIMIZED. */
#define CALIBRATED_MAX_CONV_LEN (-1)

/* ── FFT calibration data ─────────────────────────────────────
 * Compile-only: the ceiling guard in tree_ctx_create_ex2() prevents
 * best_fft_config() / best_fft_config_joint() from being called when
 * CALIBRATED_MAX_CONV_LEN == -1, so calib_sizes[] and calib_times_ns[]
 * are never consulted.  The single-element arrays exist only to satisfy
 * the compiler (zero-length static arrays are non-standard). */
#define N_CALIBRATED_SIZES 0
static const int    calib_sizes[1]    = {0};
static const double calib_times_ns[1] = {0.0};

/* ── Crossover table ──────────────────────────────────────────
 * Compile-only: empirical_crossover_k() in fft_cost_model.h guards
 * N_CROSSOVER_POINTS == 0 and returns immediately (always dispatch
 * hybrid), so these arrays are never accessed at runtime. */
#define N_CROSSOVER_POINTS 0
static const int crossover_n[1] = {0};
static const int crossover_k[1] = {0};

/* ── B-selection table ────────────────────────────────────────
 * Compile-only: empirical_best_B() in fft_cost_model.h guards
 * N_BSELECT_POINTS == 0 and returns 32 immediately. */
#define N_BSELECT_POINTS 0
static const int bselect_n[1] = {0};
static const int bselect_k[1] = {0};
static const int bselect_B[1] = {0};

/* ── Schoolbook lookup tables ─────────────────────────────────
 * Compile-only: the ceiling guard prevents the schoolbook comparison
 * path entirely, so schoolbook_mul_ns[] is never read.  The single
 * -1.0 entry (the "unmeasured" sentinel) is harmless either way. */
static const double schoolbook_mul_ns[1]  = {-1.0};
static const double schoolbook_corr_ns[1] = {-1.0};

/* ── Cost-model scalar constants ──────────────────────────────
 * Compile-only on the uncalibrated path (the cost comparison is
 * skipped).  Plausible defaults that let the code compile; never
 * used for any dispatch decision when CALIBRATED_MAX_CONV_LEN == -1. */
#define FMA_NS                     0.25
#define FFT_OVERHEAD_NS            0.0
#define WRAP_FMA_NS                4.0
#define PAIRED_CACHED_CORR_RATIO   1.03
#define INDEP_PAIR_RATIO           1.25
#define FP64_DIV_NS                10.0
#define LEAF_FMA_NS                0.25
#define LEAF_BLOCK_NS              100.0
#define BLOCK_FMA_NS               0.05
#define BLOCK_MEM_NS               0.1

/* ── Batched linear-engine constant ───────────────────────────
 * Compile-only on the uncalibrated path.  Fit against real
 * icm_run_linear_batched() measurements; NOT the same as FMA_NS. */
#define BATCHED_FMA_NS             0.0954

/* ── Cache hierarchy ──────────────────────────────────────────
 * Compile-only on the uncalibrated path.  Values are plausible
 * defaults for a modern desktop CPU; never used for dispatch
 * decisions when CALIBRATED_MAX_CONV_LEN == -1. */
#define L2_CACHE_SIZE   1048576
#define L3_CACHE_SIZE   33554432
#define L2_BW_GBS       100.0
#define L3_BW_GBS       50.0
#define DRAM_BW_GBS     20.0

/* ── MKL dual-dispatch table (Linux only) ─────────────────────
 * Compile-only: the HAS_CALIB_LIB guard prevents access on the
 * uncalibrated path (this file does not define HAS_CALIB_LIB). */

#endif /* ICM_GENERIC_FFT_CONFIG_H */
