/*
 * icm.h — ICM (Independent Chip Model) equity computation library
 *
 * Computes tournament placement equities using generating-function quadrature.
 * Three engines with automatic dispatch: linear (small k), hybrid (large k),
 * tree (pure FFT). Per-level FFT decisions use offline-calibrated data.
 */

#ifndef ICM_H
#define ICM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Public API ──────────────────────────────────────────────── */

/* Compute equities for all n players.
 *   S[n]:       chip stacks (positive doubles)
 *   Q:          number of quadrature points (typically 256)
 *   payout[k]:  payout functional coefficients
 *   k:          number of payout terms (1 ≤ k ≤ n)
 *   equity[n]:  output array (caller-allocated)
 * Returns elapsed wall time in nanoseconds. */
double icm_equity(int n, const double *S, int Q,
                  const double *payout, int k,
                  double *equity);

/* Compute equities for a subset of players.
 *   targets[n_targets]: indices of players to compute (0-based)
 *   Only equity[targets[i]] values are set in the output.
 * Returns elapsed wall time in nanoseconds. */
double icm_equity_subset(int n, const double *S, int Q,
                         const double *payout, int k,
                         double *equity,
                         const int *targets, int n_targets);

/* Initialize the library (call once before any computation).
 * Loads FFTW wisdom and builds smooth number tables.
 * wisdom_path: path to fftw_wisdom.dat (NULL for default "fftw_wisdom.dat"). */
void icm_init(const char *wisdom_path);

/* ── Engine types (for advanced use / benchmarking) ────────── */

typedef void (*IcmEngine)(int n, const double *a,
                          const double *payout, int k,
                          double *inner, void *ctx);

/* Context creation / destruction */
void *icm_tree_ctx_create(int n, int k);
void *icm_hybrid_ctx_create(int n, const double *S, int k, int B);
void *icm_linear_ctx_create(int n, int k);
void  icm_ctx_destroy(void *ctx, int engine_kind);

/* Engine function pointers (for run_engine) */
IcmEngine icm_engine_tree(void);
IcmEngine icm_engine_hybrid(void);
IcmEngine icm_engine_linear(void);

/* Run a specific engine with pre-created context.
 * Returns elapsed wall time in nanoseconds. */
double icm_run_engine(int n, const double *S, int Q,
                      const double *payout, int k,
                      double *equity, IcmEngine engine, void *ctx);

/* Batched linear engine (separate entry point, manages its own quadrature). */
double icm_run_linear_batched(int n, const double *S, int Q,
                              const double *payout, int k,
                              double *equity, void *ctx);

/* Engine kind constants */
#define ICM_ENGINE_TREE   0
#define ICM_ENGINE_LINEAR 1
#define ICM_ENGINE_HYBRID 2

/* ── Dispatch ────────────────────────────────────────────────── */

/* Engine dispatch: returns optimal block size B if hybrid wins, 0 if linear wins.
 * Callers can use this to display which engine will be selected for a given (n,k). */
int icm_select_engine(int n, int k);

/* Select optimal hybrid block size for (n, k). */
int icm_select_best_B(int n, int k);

/* ── Diagnostic / profiling ──────────────────────────────────── */

/* Measure per-call FFT overhead (prints to stdout). */
void icm_measure_fft_overhead(void);

/* Closed-form references for verification */
void icm_v1_exact(int n, const double *S, double *V1);
void icm_v2_exact(int n, const double *S, double *V2);

#ifdef __cplusplus
}
#endif

#endif /* ICM_H */
