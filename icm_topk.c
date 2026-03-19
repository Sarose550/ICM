/*
 * icm_topk.c — Top-k ICM with client-facing API
 *
 * Simplified path: compensated build to degree k, pure bottom-up division.
 * No guard band, no top-down, no setup-time branching.
 * Valid for k <= 50 at n <= 100K with epsilon = 1e-9.
 * Falls back to full icm_avx2 when k > K_MAX_TOPK.
 *
 * Three API levels:
 *   icm_equity()  — payouts vector → n equity values (most common use)
 *   icm_topk()    — n × k probability matrix
 *   icm_topk_sub()— subset of players, n_sub × k probability matrix
 */
#include "icm.h"
#include <stdio.h>

#define K_MAX_TOPK 50  /* pure BU safe up to this k */

/* ── Compensated arithmetic ───────────────────────────────────── */

static inline void two_product(double a, double b, double *s, double *e) {
    *s = a * b; *e = fma(a, b, -(*s));
}
static inline void two_sum(double a, double b, double *s, double *e) {
    *s = a + b; double v = *s - a; *e = (a - (*s - v)) + (b - v);
}

/* ── Compensated build to degree k ────────────────────────────── */

static void build_comp_k(int n, int k, const double *S, int Q,
                         const QP *pts, double *Ps, double *lv, double *wq) {
    size_t ps = (size_t)(k + 1);
    double *Ph = (double *)malloc((k + 2) * sizeof(double));
    double *Pl = (double *)malloc((k + 2) * sizeof(double));

    for (int q = 0; q < Q; q++) {
        wq[q] = pts[q].w;
        if (pts[q].w == 0) {
            lv[q] = 0;
            memset(Ps + (size_t)q * ps, 0, ps * sizeof(double));
            continue;
        }
        double logv = pts[q].logv;
        lv[q] = logv;

        Ph[0] = 1; Pl[0] = 0;
        for (int j = 1; j <= k; j++) { Ph[j] = 0; Pl[j] = 0; }
        int deg = 0;

        for (int j = 0; j < n; j++) {
            double arg = S[j] * logv;
            double aj = (arg < -700) ? 0 : exp(arg);
            double bj = 1 - aj;
            int nd = (deg + 1 < k) ? deg + 1 : k;

            for (int m = nd; m >= 1; m--) {
                double s1, e1, s2, e2, s3, e3;
                two_product(aj, Ph[m], &s1, &e1);
                two_product(bj, Ph[m - 1], &s2, &e2);
                two_sum(s1, s2, &s3, &e3);
                Pl[m] = (e1 + e2 + e3) + aj * Pl[m] + bj * Pl[m - 1];
                Ph[m] = s3;
            }
            double e0;
            two_product(aj, Ph[0], &Ph[0], &e0);
            Pl[0] = e0 + aj * Pl[0];
            deg = nd;
        }

        double *out = Ps + (size_t)q * ps;
        for (int m = 0; m <= k; m++) out[m] = Ph[m] + Pl[m];
    }
    free(Ph); free(Pl);
}

/* ── Bottom-up division: all players ──────────────────────────── */

static void divide_bu_all(int n, int k, int Q,
                          const double *S, const double *Ps,
                          const double *lv, const double *wq,
                          double *prob) {
    size_t ps = (size_t)(k + 1);
    for (int i = 0; i < n; i++) {
        double *row = prob + (size_t)i * k;
        memset(row, 0, k * sizeof(double));
        double Si = S[i], Sm = Si - 1;

        for (int q = 0; q < Q; q++) {
            if (wq[q] == 0) continue;
            double logv = lv[q];
            const double *Pq = Ps + (size_t)q * ps;
            double arg = Si * logv;
            double ai = (arg < -700) ? 0 : exp(arg);
            double bi = 1 - ai;
            double lw = Sm * logv;
            double vp = (lw < -700) ? 0 : exp(lw);
            double pw = wq[q] * Si * vp;
            if (pw == 0 || !isfinite(pw)) continue;

            double ia = 1.0 / ai;
            double qm = Pq[0] * ia;
            row[0] += pw * qm;
            for (int m = 1; m < k; m++) {
                qm = (Pq[m] - bi * qm) * ia;
                row[m] += pw * qm;
            }
        }
    }
}

/* ── Bottom-up division: subset of players ────────────────────── */

static void divide_bu_sub(int n, const int *players, int n_sub, int k, int Q,
                          const double *S, const double *Ps,
                          const double *lv, const double *wq,
                          double *prob) {
    (void)n;  /* subset uses S[players[si]], not all n players */
    size_t ps = (size_t)(k + 1);
    for (int si = 0; si < n_sub; si++) {
        int i = players[si];
        double *row = prob + (size_t)si * k;
        memset(row, 0, k * sizeof(double));
        double Si = S[i], Sm = Si - 1;

        for (int q = 0; q < Q; q++) {
            if (wq[q] == 0) continue;
            double logv = lv[q];
            const double *Pq = Ps + (size_t)q * ps;
            double arg = Si * logv;
            double ai = (arg < -700) ? 0 : exp(arg);
            double bi = 1 - ai;
            double lw = Sm * logv;
            double vp = (lw < -700) ? 0 : exp(lw);
            double pw = wq[q] * Si * vp;
            if (pw == 0 || !isfinite(pw)) continue;

            double ia = 1.0 / ai;
            double qm = Pq[0] * ia;
            row[0] += pw * qm;
            for (int m = 1; m < k; m++) {
                qm = (Pq[m] - bi * qm) * ia;
                row[m] += pw * qm;
            }
        }
    }
}

/* ── Bottom-up equity: fused division + payout dot product ────── */

static void equity_bu_all(int n, int k, int Q, const double *payouts,
                          const double *S, const double *Ps,
                          const double *lv, const double *wq,
                          double *equity) {
    size_t ps = (size_t)(k + 1);
    memset(equity, 0, n * sizeof(double));

    for (int i = 0; i < n; i++) {
        double Si = S[i], Sm = Si - 1;
        double eq = 0;

        for (int q = 0; q < Q; q++) {
            if (wq[q] == 0) continue;
            double logv = lv[q];
            const double *Pq = Ps + (size_t)q * ps;
            double arg = Si * logv;
            double ai = (arg < -700) ? 0 : exp(arg);
            double bi = 1 - ai;
            double lw = Sm * logv;
            double vp = (lw < -700) ? 0 : exp(lw);
            double pw = wq[q] * Si * vp;
            if (pw == 0 || !isfinite(pw)) continue;

            double ia = 1.0 / ai;
            double qm = Pq[0] * ia;
            eq += pw * qm * payouts[0];
            for (int m = 1; m < k; m++) {
                qm = (Pq[m] - bi * qm) * ia;
                eq += pw * qm * payouts[m];
            }
        }
        equity[i] = eq;
    }
}

/* ── Shared build helper ──────────────────────────────────────── */

typedef struct {
    double *Ps, *lv, *wq;
    int k_eff;  /* actual degree built */
} BuildResult;

static BuildResult do_build(int n, int k, int Q, const QP *pts, const double *S) {
    BuildResult b;
    b.k_eff = (k <= K_MAX_TOPK) ? k : 0;  /* 0 = use full */
    if (b.k_eff == 0) { b.Ps = NULL; b.lv = NULL; b.wq = NULL; return b; }
    size_t ps = (size_t)(k + 1);
    b.Ps = (double *)malloc((size_t)Q * ps * sizeof(double));
    b.lv = (double *)malloc(Q * sizeof(double));
    b.wq = (double *)malloc(Q * sizeof(double));
    build_comp_k(n, k, S, Q, pts, b.Ps, b.lv, b.wq);
    return b;
}

static void free_build(BuildResult *b) {
    free(b->Ps); free(b->lv); free(b->wq);
}

/* ================================================================
   Public API
   ================================================================ */

/*
 * icm_topk: compute top-k placement probabilities for all players.
 * prob must be n * k doubles. prob[i*k + m] = Pr(player i finishes m+1).
 */
void icm_topk(int n, const double *S, int Q, const QP *pts,
              int k, double *prob) {
    if (k >= n || k > K_MAX_TOPK) {
        /* Fall back to full */
        double *full = (double *)malloc((size_t)n * n * sizeof(double));
        icm_avx2(n, S, Q, pts, full);
        int kk = (k < n) ? k : n;
        for (int i = 0; i < n; i++)
            memcpy(prob + (size_t)i * kk, full + (size_t)i * n, kk * sizeof(double));
        free(full);
        return;
    }
    BuildResult b = do_build(n, k, Q, pts, S);
    divide_bu_all(n, k, Q, S, b.Ps, b.lv, b.wq, prob);
    free_build(&b);
}

/*
 * icm_topk_sub: compute top-k probabilities for a subset of players.
 * players[0..n_sub-1] are indices into S[0..n-1].
 * prob must be n_sub * k doubles.
 */
void icm_topk_sub(int n, const double *S, int Q, const QP *pts,
                  const int *players, int n_sub, int k, double *prob) {
    if (k >= n || k > K_MAX_TOPK) {
        double *full = (double *)malloc((size_t)n * n * sizeof(double));
        icm_avx2(n, S, Q, pts, full);
        int kk = (k < n) ? k : n;
        for (int si = 0; si < n_sub; si++)
            memcpy(prob + (size_t)si * kk,
                   full + (size_t)players[si] * n, kk * sizeof(double));
        free(full);
        return;
    }
    BuildResult b = do_build(n, k, Q, pts, S);
    divide_bu_sub(n, players, n_sub, k, Q, S, b.Ps, b.lv, b.wq, prob);
    free_build(&b);
}

/*
 * icm_equity: compute equity for all players under a payout structure.
 * payouts[0..k-1]: payout for finishing 1st, 2nd, ..., kth.
 * equity[0..n-1]: output expected value for each player.
 */
void icm_equity(int n, const double *S, int Q, const QP *pts,
                const double *payouts, int k, double *equity) {
    if (k >= n || k > K_MAX_TOPK) {
        double *full = (double *)malloc((size_t)n * n * sizeof(double));
        icm_avx2(n, S, Q, pts, full);
        int kk = (k < n) ? k : n;
        memset(equity, 0, n * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int m = 0; m < kk; m++)
                equity[i] += payouts[m] * full[(size_t)i * n + m];
        free(full);
        return;
    }
    BuildResult b = do_build(n, k, Q, pts, S);
    equity_bu_all(n, k, Q, payouts, S, b.Ps, b.lv, b.wq, equity);
    free_build(&b);
}

/*
 * icm_equity_sub: equity for a subset of players.
 */
void icm_equity_sub(int n, const double *S, int Q, const QP *pts,
                    const int *players, int n_sub,
                    const double *payouts, int k, double *equity) {
    if (k >= n || k > K_MAX_TOPK) {
        double *full = (double *)malloc((size_t)n * n * sizeof(double));
        icm_avx2(n, S, Q, pts, full);
        int kk = (k < n) ? k : n;
        memset(equity, 0, n_sub * sizeof(double));
        for (int si = 0; si < n_sub; si++) {
            int i = players[si];
            for (int m = 0; m < kk; m++)
                equity[si] += payouts[m] * full[(size_t)i * n + m];
        }
        free(full);
        return;
    }
    /* Build once, then fused divide+payout for subset */
    BuildResult b = do_build(n, k, Q, pts, S);
    size_t ps = (size_t)(k + 1);
    memset(equity, 0, n_sub * sizeof(double));

    for (int si = 0; si < n_sub; si++) {
        int i = players[si];
        double Si = S[i], Sm = Si - 1, eq = 0;
        for (int q = 0; q < Q; q++) {
            if (b.wq[q] == 0) continue;
            double logv = b.lv[q];
            const double *Pq = b.Ps + (size_t)q * ps;
            double arg = Si * logv;
            double ai = (arg < -700) ? 0 : exp(arg), bi = 1 - ai;
            double lw = Sm * logv;
            double vp = (lw < -700) ? 0 : exp(lw);
            double pw = b.wq[q] * Si * vp;
            if (pw == 0 || !isfinite(pw)) continue;
            double ia = 1.0 / ai, qm = Pq[0] * ia;
            eq += pw * qm * payouts[0];
            for (int m = 1; m < k; m++) {
                qm = (Pq[m] - bi * qm) * ia;
                eq += pw * qm * payouts[m];
            }
        }
        equity[si] = eq;
    }
    free_build(&b);
}
