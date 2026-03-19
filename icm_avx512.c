/*
 * icm_avx512.c — AVX-512 compute backend (Zen 4 / Sapphire Rapids)
 *
 * Same algorithm as icm_avx2.c but with 8-player SIMD batches,
 * vectorized exp(), and 512-bit polynomial build.
 *
 * Compile: gcc -O3 -march=znver4 -mavx512f -mavx512dq -mfma -c icm_avx512.c
 */
#include "icm.h"
#include <immintrin.h>

/* ── Vectorized exp() for 8 doubles ───────────────────────────── */

static inline __m512d fast_exp_8(__m512d x) {
    const __m512d LOG2E  = _mm512_set1_pd(1.4426950408889634);
    const __m512d LN2_HI = _mm512_set1_pd(6.93145751953125e-1);
    const __m512d LN2_LO = _mm512_set1_pd(1.42860682030941723212e-6);
    const __m512d C1  = _mm512_set1_pd(1.0);
    const __m512d C2  = _mm512_set1_pd(0.5);
    const __m512d C3  = _mm512_set1_pd(1.6666666666666666e-1);
    const __m512d C4  = _mm512_set1_pd(4.1666666666666664e-2);
    const __m512d C5  = _mm512_set1_pd(8.3333333333333332e-3);
    const __m512d C6  = _mm512_set1_pd(1.3888888888888889e-3);
    const __m512d C7  = _mm512_set1_pd(1.9841269841269841e-4);
    const __m512d C8  = _mm512_set1_pd(2.4801587301587302e-5);
    const __m512d C9  = _mm512_set1_pd(2.7557319223985893e-6);
    const __m512d C10 = _mm512_set1_pd(2.7557319223985888e-7);
    const __m512d C11 = _mm512_set1_pd(2.5052108385441720e-8);

    __m512d k = _mm512_roundscale_pd(
        _mm512_mul_pd(x, LOG2E), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512d r = _mm512_fnmadd_pd(k, LN2_HI, x);
    r = _mm512_fnmadd_pd(k, LN2_LO, r);

    __m512d p = _mm512_fmadd_pd(C11, r, C10);
    p = _mm512_fmadd_pd(p, r, C9);  p = _mm512_fmadd_pd(p, r, C8);
    p = _mm512_fmadd_pd(p, r, C7);  p = _mm512_fmadd_pd(p, r, C6);
    p = _mm512_fmadd_pd(p, r, C5);  p = _mm512_fmadd_pd(p, r, C4);
    p = _mm512_fmadd_pd(p, r, C3);  p = _mm512_fmadd_pd(p, r, C2);
    p = _mm512_fmadd_pd(p, r, C1);  p = _mm512_fmadd_pd(p, r, C1);

    __m512i ki = _mm512_cvtpd_epi64(k);
    ki = _mm512_add_epi64(ki, _mm512_set1_epi64(1023));
    ki = _mm512_slli_epi64(ki, 52);
    __m512d result = _mm512_mul_pd(p, _mm512_castsi512_pd(ki));

    __mmask8 under = _mm512_cmp_pd_mask(x, _mm512_set1_pd(-708.0), _CMP_LT_OQ);
    return _mm512_mask_blend_pd(under, result, _mm512_setzero_pd());
}

/* ── Polynomial build with AVX-512 + vectorized exp + seqcombine ── */

#define BUILD_BLOCK_THRESH_512 512

static inline int build_block_size_512(int n) {
    int B = n / 4;
    if (B < 64)  B = 64;
    if (B > 384) B = 384;
    B = (B + 7) & ~7;
    return B;
}

/* Sequential build of one polynomial (AVX-512 coefficient sweep) */
static void build_poly_sequential_512(int n, const double *a, double *P) {
    P[0] = 1;
    for (int j = 1; j <= n; j++) P[j] = 0;
    int deg = 0;
    for (int j = 0; j < n; j++) {
        double aj = a[j], bj = 1 - aj;
        int nd = (deg + 1 < n) ? deg + 1 : n;
        __m512d vaj = _mm512_set1_pd(aj), vbj = _mm512_set1_pd(bj);
        int m = nd;
        for (; m >= 8; m -= 8) {
            __m512d cur  = _mm512_loadu_pd(&P[m - 7]);
            __m512d prev = _mm512_loadu_pd(&P[m - 8]);
            _mm512_storeu_pd(&P[m - 7],
                _mm512_fmadd_pd(vaj, cur, _mm512_mul_pd(vbj, prev)));
        }
        __m256d vaj4 = _mm256_broadcast_sd(&aj), vbj4 = _mm256_broadcast_sd(&bj);
        for (; m >= 4; m -= 4) {
            __m256d cur  = _mm256_loadu_pd(&P[m - 3]);
            __m256d prev = _mm256_loadu_pd(&P[m - 4]);
            _mm256_storeu_pd(&P[m - 3],
                _mm256_fmadd_pd(vaj4, cur, _mm256_mul_pd(vbj4, prev)));
        }
        for (; m >= 1; m--) P[m] = aj * P[m] + bj * P[m - 1];
        P[0] *= aj;
        deg = nd;
    }
}

/* Sequential-combine build: blocks of B → schoolbook combine */
static void build_poly_seqcombine_512(int n, const double *a, int B,
                                      double *buf0, double *buf1,
                                      double *chunks) {
    int C = (n + B - 1) / B;
    int ps = B + 1;

    /* Phase 1: build each chunk */
    for (int c = 0; c < C; c++) {
        int start = c * B;
        int count = (start + B <= n) ? B : (n - start);
        double *ch = chunks + (size_t)c * ps;
        ch[0] = 1;
        for (int j = 1; j <= B; j++) ch[j] = 0;
        for (int j = 0; j < count; j++) {
            double aj = a[start + j], bj = 1 - aj;
            int nd = j + 1;
            for (int m = nd; m >= 1; m--)
                ch[m] = aj * ch[m] + bj * ch[m - 1];
            ch[0] *= aj;
        }
    }

    /* Phase 2: sequential combine with ping-pong */
    int first_count = (B <= n) ? B : n;
    memcpy(buf0, chunks, (first_count + 1) * sizeof(double));
    int deg = first_count;
    double *src = buf0, *dst = buf1;

    for (int c = 1; c < C; c++) {
        int start = c * B;
        int count = (start + B <= n) ? B : (n - start);
        const double *ch = chunks + (size_t)c * ps;
        int ch_deg = count;
        int new_deg = deg + ch_deg;
        if (new_deg > n) new_deg = n;

        memset(dst, 0, (new_deg + 1) * sizeof(double));

        /* Input-major: dst[i+j] += src[i] * ch[j] */
        for (int i = 0; i <= deg; i++) {
            double si = src[i];
            if (si == 0) continue;
            int jmax = ch_deg;
            if (i + jmax > new_deg) jmax = new_deg - i;
            double *d = dst + i;
            for (int j = 0; j <= jmax; j++)
                d[j] += si * ch[j];
        }

        deg = new_deg;
        double *tmp = src; src = dst; dst = tmp;
    }

    if (src != buf0)
        memcpy(buf0, src, (n + 1) * sizeof(double));
}

static void build_all_polys_512(int n, const double *S, int Q, const QP *pts,
                                double *P_store, double *logv_store, double *wq_store) {
    size_t ps = (size_t)(n + 1);
    double *a = (double *)malloc((size_t)n * sizeof(double));

    int use_seqcombine = (n >= BUILD_BLOCK_THRESH_512);
    int B = build_block_size_512(n);

    double *buf0 = NULL, *buf1 = NULL, *chunks = NULL;
    if (use_seqcombine) {
        buf0 = (double *)malloc((n + 1) * sizeof(double));
        buf1 = (double *)malloc((n + 1) * sizeof(double));
        int C = (n + B - 1) / B;
        chunks = (double *)malloc((size_t)C * (B + 1) * sizeof(double));
    } else {
        buf0 = (double *)malloc((n + 2) * sizeof(double));
    }

    for (int q = 0; q < Q; q++) {
        wq_store[q] = pts[q].w;
        if (pts[q].w == 0) {
            logv_store[q] = 0;
            memset(P_store + (size_t)q * ps, 0, ps * sizeof(double));
            continue;
        }
        double logv = pts[q].logv;
        logv_store[q] = logv;

        /* Vectorized a[j] = exp(S[j] * logv) */
        __m512d vlogv = _mm512_set1_pd(logv);
        int j = 0;
        for (; j + 7 < n; j += 8) {
            __m512d vs = _mm512_loadu_pd(&S[j]);
            _mm512_storeu_pd(&a[j], fast_exp_8(_mm512_mul_pd(vs, vlogv)));
        }
        for (; j < n; j++) {
            double arg = S[j] * logv;
            a[j] = (arg < -700) ? 0 : exp(arg);
        }

        if (use_seqcombine) {
            build_poly_seqcombine_512(n, a, B, buf0, buf1, chunks);
            memcpy(P_store + (size_t)q * ps, buf0, ps * sizeof(double));
        } else {
            build_poly_sequential_512(n, a, buf0);
            memcpy(P_store + (size_t)q * ps, buf0, ps * sizeof(double));
        }
    }

    free(a); free(buf0);
    if (use_seqcombine) { free(buf1); free(chunks); }
}

/* ── 8-player fused divide + accumulate ───────────────────────── */

static void divide_accum_8(
    int n, int batch, int Q, int qstart,
    const double *Si, const double *Si_m1,
    const double *P_store, const double *logv_store, const double *wq_store,
    double *acc)
{
    size_t ps = (size_t)(n + 1);
    for (int q = qstart; q < Q; q++) {
        double wq = wq_store[q];
        if (wq == 0) continue;
        double logv = logv_store[q];
        const double *Pq = P_store + (size_t)q * ps;

        __m512d vlogv = _mm512_set1_pd(logv), vwq = _mm512_set1_pd(wq);
        __m512d vSi   = _mm512_load_pd(Si), vSi_m1 = _mm512_load_pd(Si_m1);
        __m512d vai   = fast_exp_8(_mm512_mul_pd(vSi, vlogv));
        __m512d vvp   = fast_exp_8(_mm512_mul_pd(vSi_m1, vlogv));
        __m512d vbi   = _mm512_sub_pd(_mm512_set1_pd(1.0), vai);
        __m512d vpw   = _mm512_mul_pd(_mm512_mul_pd(vwq, vSi), vvp);

        if (batch < 8) {
            __mmask8 active = (__mmask8)((1 << batch) - 1);
            vai = _mm512_maskz_mov_pd(active, vai);
            vbi = _mm512_mask_blend_pd(active, _mm512_setzero_pd(), vbi);
            vpw = _mm512_maskz_mov_pd(active, vpw);
        }

        /* Kill NaN/inf/zero pw lanes */
        __mmask8 bad = _mm512_cmp_pd_mask(vpw, _mm512_setzero_pd(), _CMP_EQ_OQ) |
                       _mm512_cmp_pd_mask(_mm512_abs_pd(vpw), _mm512_set1_pd(DBL_MAX), _CMP_GT_OQ);
        vpw = _mm512_maskz_mov_pd(~bad, vpw);
        if (_mm512_reduce_add_pd(vpw) == 0.0) continue;

        __mmask8 active_mask = (batch < 8) ? (__mmask8)((1 << batch) - 1) : 0xFF;
        __mmask8 bu_mask = _mm512_cmp_pd_mask(vai, _mm512_set1_pd(0.5), _CMP_GT_OQ) & active_mask;
        int all_bu = (bu_mask == active_mask);
        int all_td = (bu_mask == 0);

        if (all_bu) {
            __m512d via = _mm512_div_pd(_mm512_set1_pd(1.0), vai);
            __m512d vc = _mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(), vbi), via);
            __m512d qp = _mm512_mul_pd(_mm512_set1_pd(Pq[0]), via);
            _mm512_store_pd(&acc[0], _mm512_fmadd_pd(vpw, qp, _mm512_load_pd(&acc[0])));
            for (int m = 0; m < n-1; m++) {
                __m512d dm = _mm512_mul_pd(_mm512_set1_pd(Pq[m+1]), via);
                qp = _mm512_fmadd_pd(vc, qp, dm);
                _mm512_store_pd(&acc[(m+1)*8],
                    _mm512_fmadd_pd(vpw, qp, _mm512_load_pd(&acc[(m+1)*8])));
            }
        } else if (all_td) {
            __m512d vib = _mm512_div_pd(_mm512_set1_pd(1.0), vbi);
            __m512d vc = _mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(), vai), vib);
            __m512d qp = _mm512_mul_pd(_mm512_set1_pd(Pq[n]), vib);
            _mm512_store_pd(&acc[(n-1)*8],
                _mm512_fmadd_pd(vpw, qp, _mm512_load_pd(&acc[(n-1)*8])));
            for (int m = n-1; m >= 1; m--) {
                __m512d dm = _mm512_mul_pd(_mm512_set1_pd(Pq[m]), vib);
                qp = _mm512_fmadd_pd(vc, qp, dm);
                _mm512_store_pd(&acc[(m-1)*8],
                    _mm512_fmadd_pd(vpw, qp, _mm512_load_pd(&acc[(m-1)*8])));
            }
        } else {
            double ai_arr[8] __attribute__((aligned(64)));
            double bi_arr[8] __attribute__((aligned(64)));
            double pw_arr[8] __attribute__((aligned(64)));
            _mm512_store_pd(ai_arr, vai);
            _mm512_store_pd(bi_arr, vbi);
            _mm512_store_pd(pw_arr, vpw);
            for (int b = 0; b < batch; b++) {
                if (pw_arr[b] == 0) continue;
                double ai = ai_arr[b], bi = bi_arr[b], pw = pw_arr[b];
                if (ai > 0.5) {
                    double ia = 1.0/ai, c_s = -bi*ia;
                    double qm = Pq[0]*ia;
                    acc[b] += pw * qm;
                    for (int m = 0; m < n-1; m++) {
                        qm = c_s*qm + Pq[m+1]*ia;
                        acc[(m+1)*8+b] += pw * qm;
                    }
                } else {
                    double ib = 1.0/bi, c_s = -ai*ib;
                    double qm = Pq[n]*ib;
                    acc[(n-1)*8+b] += pw * qm;
                    for (int m = n-1; m >= 1; m--) {
                        qm = c_s*qm + Pq[m]*ib;
                        acc[(m-1)*8+b] += pw * qm;
                    }
                }
            }
        }
    }
}

/* ── Public entry point ───────────────────────────────────────── */

void icm_avx512(int n, const double *S, int Q, const QP *pts, double *prob) {
    memset(prob, 0, (size_t)n * n * sizeof(double));

    size_t ps = (size_t)(n + 1);
    double *Ps  = (double *)malloc((size_t)Q * ps * sizeof(double));
    double *lv  = (double *)malloc(Q * sizeof(double));
    double *wqs = (double *)malloc(Q * sizeof(double));
    build_all_polys_512(n, S, Q, pts, Ps, lv, wqs);

    /* Sort players by stack */
    int *ord = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) ord[i] = i;
    for (int gap = n/2; gap > 0; gap /= 2)
        for (int i = gap; i < n; i++) {
            int t = ord[i]; double sv = S[t]; int j = i;
            while (j >= gap && S[ord[j-gap]] > sv) { ord[j] = ord[j-gap]; j -= gap; }
            ord[j] = t;
        }

    int *ql = (int *)malloc(n * sizeof(int));
    for (int k = 0; k < n; k++) {
        double th = -700.0 / S[ord[k]];
        int lo = 0, hi = Q;
        while (lo < hi) { int mid = (lo+hi)/2; if (lv[mid] < th) lo = mid+1; else hi = mid; }
        ql[k] = lo;
    }

    /* 8-player blocks. acc = n*8*8 bytes. At n=2048: 128 KB (fits Zen 4 L2). */
    double *acc = (double *)_mm_malloc((size_t)n * 8 * sizeof(double), 64);
    double Si_buf[8] __attribute__((aligned(64)));
    double Sm_buf[8] __attribute__((aligned(64)));

    for (int k0 = 0; k0 < n; k0 += 8) {
        int batch = ((k0+8) <= n) ? 8 : (n-k0);
        memset(acc, 0, (size_t)n * 8 * sizeof(double));
        int qs = Q;
        int pidx[8] = {0};
        for (int b = 0; b < batch; b++) {
            pidx[b] = ord[k0+b]; Si_buf[b] = S[pidx[b]]; Sm_buf[b] = Si_buf[b]-1;
            if (ql[k0+b] < qs) qs = ql[k0+b];
        }
        for (int b = batch; b < 8; b++) { Si_buf[b] = 1; Sm_buf[b] = 0; }

        divide_accum_8(n, batch, Q, qs, Si_buf, Sm_buf, Ps, lv, wqs, acc);

        for (int b = 0; b < batch; b++) {
            double *row = prob + (size_t)pidx[b] * n;
            for (int m = 0; m < n; m++) row[m] = acc[m*8+b];
        }
    }

    _mm_free(acc); free(ord); free(ql); free(Ps); free(lv); free(wqs);
}
