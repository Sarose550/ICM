#ifndef ICM_H
#define ICM_H
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { double logv, w; } QP;

static inline double icm_log_Phi(double y) {
    if (y >= 0) { double ec = erfc(y / sqrt(2.0)); return log1p(-ec / 2.0); }
    else        { double ec = erfc(-y / sqrt(2.0)); return log(ec / 2.0); }
}

/* ── Common (icm_common.c) ────────────────────────────────────── */
void icm_make_nodes(int Q, double Smax, QP *pts);
void icm_exact_V1(int n, const double *S, double *V1);
double icm_max_relV1(int n, const double *prob, const double *eV1);
void icm_make_stacks(int n, double ratio, int dist, double *S);
double icm_smax(int n, const double *S);

/* ── Full n×n backends ────────────────────────────────────────── */
void icm_avx2(int n, const double *S, int Q, const QP *pts, double *prob);
void icm_avx512(int n, const double *S, int Q, const QP *pts, double *prob)
#ifdef __cplusplus
    ;
#else
    __attribute__((weak));
#endif

/* ── Runtime detection (icm_detect.c) ─────────────────────────── */
typedef void (*ICMFunc)(int, const double *, int, const QP *, double *);

/* Returns best backend for current CPU (CPUID + XGETBV + link check) */
ICMFunc icm_best_backend(void);
const char *icm_backend_name(void);  /* "avx2" or "avx512" */
const char *icm_cpu_model(void);     /* e.g. "AMD EPYC 9654 96-Core Processor" */
void icm_print_cpu_info(FILE *f);    /* Print CPU model, features, selected backend */

/* ── Top-k (icm_topk.c) ──────────────────────────────────────── */

/* Top-k probabilities: prob[i*k + m] for all n players, m=0..k-1 */
void icm_topk(int n, const double *S, int Q, const QP *pts,
              int k, double *prob);

/* Top-k probabilities for a subset of players */
void icm_topk_sub(int n, const double *S, int Q, const QP *pts,
                  const int *players, int n_sub, int k, double *prob);

/* Equity under payout structure (fused divide + dot product) */
void icm_equity(int n, const double *S, int Q, const QP *pts,
                const double *payouts, int k, double *equity);

/* Equity for a subset of players */
void icm_equity_sub(int n, const double *S, int Q, const QP *pts,
                    const int *players, int n_sub,
                    const double *payouts, int k, double *equity);

#endif
