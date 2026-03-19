/*
 * icm_common.c — Quadrature nodes, validation, test distributions
 */
#include "icm.h"
#include <stdio.h>

/* ── Domain bounds in normal space ────────────────────────────── */

static void erfc_domain(double Smax, double *ylo, double *yhi) {
    /* Lower bound: bisect for log_Phi(y) < -25 */
    double lo = -20, hi = 0;
    for (int i = 0; i < 100; i++) {
        double m = (lo + hi) / 2;
        if (icm_log_Phi(m) < -25) lo = m; else hi = m;
    }
    *ylo = lo - 1.0;

    /* Upper bound: bisect for -log_Phi(y) > 1e-10/Smax */
    lo = 0; hi = 20;
    double tgt = 1e-10 / Smax;
    for (int i = 0; i < 100; i++) {
        double m = (lo + hi) / 2;
        if (-icm_log_Phi(m) > tgt) lo = m; else hi = m;
    }
    *yhi = hi + 1.0;
}

/* ── Quadrature node generation ───────────────────────────────── */

void icm_make_nodes(int Q, double Smax, QP *pts) {
    double yl, yh;
    erfc_domain(Smax, &yl, &yh);
    double h = (yh - yl) / (Q - 1);
    for (int q = 0; q < Q; q++) {
        double y = yl + q * h;
        double phi = exp(-y * y / 2) / sqrt(2 * M_PI);
        pts[q].logv = icm_log_Phi(y);
        pts[q].w = h * phi;
        if (q == 0 || q == Q - 1) pts[q].w *= 0.5;
    }
}

/* ── Validation ───────────────────────────────────────────────── */

void icm_exact_V1(int n, const double *S, double *V1) {
    for (int i = 0; i < n; i++) {
        double v = 1;
        for (int j = 0; j < n; j++)
            if (j != i) v += S[i] / (S[i] + S[j]);
        V1[i] = v;
    }
}

double icm_max_relV1(int n, const double *prob, const double *eV1) {
    double mx = 0;
    for (int i = 0; i < n; i++) {
        const double *r = prob + (size_t)i * n;
        double nv = 0;
        for (int m = 0; m < n; m++) nv += (double)(n - m) * r[m];
        double re = (eV1[i] != 0) ? fabs(nv - eV1[i]) / fabs(eV1[i]) : fabs(nv);
        if (re > mx) mx = re;
    }
    return mx;
}

/* ── Test distributions ───────────────────────────────────────── */

void icm_make_stacks(int n, double ratio, int dist, double *S) {
    switch (dist) {
    case 0: /* adversarial: 1 big, rest = 1 */
        for (int i = 0; i < n; i++) S[i] = 1;
        S[0] = ratio;
        break;
    case 1: /* reverse_adv: 1 small, rest = big */
        for (int i = 0; i < n; i++) S[i] = ratio;
        S[0] = 1;
        break;
    case 2: /* bimodal: half small, half big */
        for (int i = 0; i < n; i++) S[i] = (i < n / 2) ? 1 : ratio;
        break;
    case 3: /* geometric: log-spaced */
        for (int i = 0; i < n; i++) S[i] = pow(ratio, (double)i / (n - 1));
        break;
    case 4: /* uniform random */
        { srand(42); double mn = 1e30;
          for (int i = 0; i < n; i++) {
              S[i] = 1 + (ratio - 1) * ((double)rand() / RAND_MAX);
              if (S[i] < mn) mn = S[i];
          }
          for (int i = 0; i < n; i++) S[i] /= mn;
        }
        break;
    }
}

double icm_smax(int n, const double *S) {
    double mx = 0;
    for (int i = 0; i < n; i++) if (S[i] > mx) mx = S[i];
    return mx;
}
