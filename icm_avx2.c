/*
 * icm_avx2.c — AVX2 compute backend
 *
 * Algorithm: loop-inverted (i-outer, q-inner), 4-player interleaved
 * accumulator, stack-sorted batching, fused SIMD divide+accumulate.
 *
 * Compile: gcc -O3 -march=native -mavx2 -mfma -c icm_avx2.c
 */
#include "icm.h"
#include <immintrin.h>

/* ── Polynomial build (shared between build and divide) ───────── */

/* ── Block size for sequential-combine build ──────────────────── */
#define BUILD_BLOCK_THRESH 512   /* use seqcombine for n >= this */

/* Adaptive B = clip(64, 384, n/4), rounded to a multiple of 8.
 *
 * Two regimes determine the optimum:
 *
 *   Small n (n < ~1536): B* = n/4  (i.e. C ≈ 4 chunks).
 *     With few combine steps, minimizing their count dominates.
 *     C = 4 is the sweet spot: enough steps for the schoolbook
 *     inner loop to be non-trivial, few enough that per-step
 *     overhead (memset, cache warm-up) doesn't eat the savings.
 *
 *   Large n (n >= ~1536): B* ≈ 384  (constant, independent of n).
 *     The B-dependent cost is T_opt(B) = (nB/2)·α₁ + (n/B)·f̄
 *     where α₁ ≈ 0.31 ns/FMA is the phase-1 per-FMA cost and
 *     f̄ ≈ 23 μs is the average per-step overhead.  By AM-GM
 *     B* = √(2f̄/α₁) which is independent of n.  The measured
 *     optimum sits at B ≈ 384 across n = 1536 … 16384, with a
 *     95% plateau typically spanning [200, 500+].
 *
 * The lower clamp at 64 ensures the schoolbook inner loop (B+1
 * FMAs) is long enough to auto-vectorize efficiently (≥16 AVX2
 * vector ops).  It never activates for n >= BUILD_BLOCK_THRESH.
 *
 * The upper clamp at 384 keeps the degree-B chunk in L1 (384
 * doubles = 3 KB) and prevents phase-1 cost from dominating.
 * Raising to 512 hurts at moderate n (6144) by ~5%. */
static inline int build_block_size(int n) {
    int B = n / 4;
    if (B < 64)  B = 64;
    if (B > 384) B = 384;
    B = (B + 7) & ~7;  /* round up to multiple of 8 for alignment */
    return B;
}

/* Sequential build of a single polynomial from n linear factors.
   Used directly for small n, and within each block for seqcombine. */
static void build_poly_sequential(int n, const double *a, double *P) {
    P[0] = 1;
    for (int j = 1; j <= n; j++) P[j] = 0;
    int deg = 0;
    for (int j = 0; j < n; j++) {
        double aj = a[j], bj = 1 - aj;
        int nd = (deg + 1 < n) ? deg + 1 : n;
        __m256d vaj = _mm256_broadcast_sd(&aj);
        __m256d vbj = _mm256_broadcast_sd(&bj);
        int m = nd;
        for (; m >= 4; m -= 4) {
            __m256d cur  = _mm256_loadu_pd(&P[m - 3]);
            __m256d prev = _mm256_loadu_pd(&P[m - 4]);
            _mm256_storeu_pd(&P[m - 3],
                _mm256_fmadd_pd(vaj, cur, _mm256_mul_pd(vbj, prev)));
        }
        for (; m >= 1; m--) P[m] = aj * P[m] + bj * P[m - 1];
        P[0] *= aj;
        deg = nd;
    }
}

/* Sequential-combine build: group n factors into blocks of B,
   build each block sequentially, then combine by schoolbook-
   multiplying the running product by each degree-B chunk.
   
   The inner combine loop is input-major: for each src[i], sweep
   dst[i+j] += src[i] * ch[j] over j=0..B. This is a fixed-length
   FMA sweep that auto-vectorizes perfectly. Ping-pong buffers
   avoid the per-step memcpy. */
static void build_poly_seqcombine(int n, const double *a, int B,
                                  double *buf0, double *buf1,
                                  double *chunks) {
    int C = (n + B - 1) / B;
    int ps = B + 1;

    /* Phase 1: build each chunk via sequential multiply */
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

        /* Zero only the output region */
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

    /* Result is in src; if caller passed buf0, it might be in buf1.
       Copy to buf0 so caller always finds it there. */
    if (src != buf0)
        memcpy(buf0, src, (n + 1) * sizeof(double));
}

static void build_all_polys(int n, const double *S, int Q, const QP *pts,
                            double *P_store, double *logv_store, double *wq_store) {
    size_t ps = (size_t)(n + 1);
    double *a = (double *)malloc((size_t)n * sizeof(double));

    int use_seqcombine = (n >= BUILD_BLOCK_THRESH);
    int B = build_block_size(n);

    /* Persistent buffers for seqcombine (allocated once, reused per q) */
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

        for (int j = 0; j < n; j++) {
            double arg = S[j] * logv;
            a[j] = (arg < -700) ? 0 : exp(arg);
        }

        if (use_seqcombine) {
            build_poly_seqcombine(n, a, B, buf0, buf1, chunks);
            memcpy(P_store + (size_t)q * ps, buf0, ps * sizeof(double));
        } else {
            build_poly_sequential(n, a, buf0);
            memcpy(P_store + (size_t)q * ps, buf0, ps * sizeof(double));
        }
    }

    free(a); free(buf0);
    if (use_seqcombine) { free(buf1); free(chunks); }
}

/* ── AVX2 fast exp (4-wide, Cody-Waite + degree-11 minimax) ───── */

static inline __m256d fast_exp_4(__m256d x) {
    const __m256d LOG2E  = _mm256_set1_pd(1.4426950408889634);
    const __m256d LN2_HI = _mm256_set1_pd(6.93145751953125e-1);
    const __m256d LN2_LO = _mm256_set1_pd(1.42860682030941723212e-6);
    const __m256d C1  = _mm256_set1_pd(1.0);
    const __m256d C2  = _mm256_set1_pd(0.5);
    const __m256d C3  = _mm256_set1_pd(1.6666666666666666e-1);
    const __m256d C4  = _mm256_set1_pd(4.1666666666666664e-2);
    const __m256d C5  = _mm256_set1_pd(8.3333333333333332e-3);
    const __m256d C6  = _mm256_set1_pd(1.3888888888888889e-3);
    const __m256d C7  = _mm256_set1_pd(1.9841269841269841e-4);
    const __m256d C8  = _mm256_set1_pd(2.4801587301587302e-5);
    const __m256d C9  = _mm256_set1_pd(2.7557319223985893e-6);
    const __m256d C10 = _mm256_set1_pd(2.7557319223985888e-7);
    const __m256d C11 = _mm256_set1_pd(2.5052108385441720e-8);

    __m256d k = _mm256_round_pd(_mm256_mul_pd(x, LOG2E),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = _mm256_fnmadd_pd(k, LN2_HI, x);
    r = _mm256_fnmadd_pd(k, LN2_LO, r);

    __m256d p = _mm256_fmadd_pd(C11, r, C10);
    p = _mm256_fmadd_pd(p, r, C9);  p = _mm256_fmadd_pd(p, r, C8);
    p = _mm256_fmadd_pd(p, r, C7);  p = _mm256_fmadd_pd(p, r, C6);
    p = _mm256_fmadd_pd(p, r, C5);  p = _mm256_fmadd_pd(p, r, C4);
    p = _mm256_fmadd_pd(p, r, C3);  p = _mm256_fmadd_pd(p, r, C2);
    p = _mm256_fmadd_pd(p, r, C1);  p = _mm256_fmadd_pd(p, r, C1);

    /* 2^k via integer exponent. AVX2 doesn't have cvtpd_epi64, so we
       use the well-known "magic number" trick for doubles in [0, 2^52). */
    __m256d kshift = _mm256_add_pd(k, _mm256_set1_pd(1023.0 + 4503599627370496.0));
    /* Extract low 64 bits as integer, shift left 52. We abuse the fact that
       after adding 2^52, the mantissa bits ARE the integer value. */
    __m256i ki = _mm256_castpd_si256(kshift);
    ki = _mm256_slli_epi64(ki, 52);
    __m256d scale = _mm256_castsi256_pd(ki);
    __m256d result = _mm256_mul_pd(p, scale);

    /* Clamp underflow: x < -708 → 0 */
    __m256d mask = _mm256_cmp_pd(x, _mm256_set1_pd(-708.0), _CMP_LT_OQ);
    return _mm256_andnot_pd(mask, result);
}

/* ── Fused SIMD divide + accumulate for 4 players ─────────────── 
   
   Two key optimizations over the baseline:
   
   1. inv_v precomputation: vp = a_i * inv_v instead of exp((S-1)*logv).
      Saves 4 exp() calls per (block, quad point) → ~13 ms at n=2048.
   
   2. fast_exp_4: vectorized exp for the 4 a_i values.
      Replaces 4 sequential scalar exp() with 1 AVX2 operation → ~11 ms.
   
   3. 2-way q-unrolling: process 2 quad points per iteration.
      The two division recurrences are independent, giving the OOO engine
      twice the ILP to fill execution ports. The acc updates are combined:
      acc[m] += pw0*qp0 + pw1*qp1 in one load-fmadd-fmadd-store sequence.
   ──────────────────────────────────────────────────────────────── */

static void divide_accum_4(
    int n, int batch, int Q, int qstart,
    const double *Si, const double *Si_m1,
    const double *P_store, const double *logv_store, const double *wq_store,
    const double *inv_v_store,  /* NEW: precomputed exp(-logv) per q */
    double *acc)
{
    (void)Si_m1;  /* inv_v precomputation replaced exp((S-1)*logv) */
    size_t ps = (size_t)(n + 1);
    __m256d vSi   = _mm256_loadu_pd(Si);

    int q = qstart;

    /* ── 2-way unrolled main loop ─────────────────────────────── */
    for (; q + 1 < Q; q += 2) {
        double wq0 = wq_store[q], wq1 = wq_store[q+1];
        if (wq0 == 0 && wq1 == 0) continue;
        double logv0 = logv_store[q], logv1 = logv_store[q+1];
        const double *Pq0 = P_store + (size_t)q * ps;
        const double *Pq1 = P_store + (size_t)(q+1) * ps;

        /* Vectorized a_i = exp(Si * logv) for both quad points */
        __m256d arg0 = _mm256_mul_pd(vSi, _mm256_set1_pd(logv0));
        __m256d arg1 = _mm256_mul_pd(vSi, _mm256_set1_pd(logv1));
        __m256d ai0 = fast_exp_4(arg0);
        __m256d ai1 = fast_exp_4(arg1);
        __m256d bi0 = _mm256_sub_pd(_mm256_set1_pd(1.0), ai0);
        __m256d bi1 = _mm256_sub_pd(_mm256_set1_pd(1.0), ai1);

        /* vp = a_i * inv_v (replaces exp((S-1)*logv)) */
        __m256d vp0 = _mm256_mul_pd(ai0, _mm256_set1_pd(inv_v_store[q]));
        __m256d vp1 = _mm256_mul_pd(ai1, _mm256_set1_pd(inv_v_store[q+1]));
        __m256d pw0 = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(wq0), vSi), vp0);
        __m256d pw1 = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(wq1), vSi), vp1);

        /* Direction: check if all 4 players agree for each q */
        /* With sorted stacks, they almost always do, and since q0/q1 are
           adjacent, they usually agree with each other too. */
        __m256d half = _mm256_set1_pd(0.5);
        int bu0 = _mm256_movemask_pd(_mm256_cmp_pd(ai0, half, _CMP_GT_OQ));
        int bu1 = _mm256_movemask_pd(_mm256_cmp_pd(ai1, half, _CMP_GT_OQ));
        int active_mask = (1 << batch) - 1;
        int all_bu_both = ((bu0 & active_mask) == active_mask) &&
                          ((bu1 & active_mask) == active_mask);
        int all_td_both = ((bu0 & active_mask) == 0) &&
                          ((bu1 & active_mask) == 0);

        if (all_bu_both) {
            __m256d via0 = _mm256_div_pd(_mm256_set1_pd(1.0), ai0);
            __m256d via1 = _mm256_div_pd(_mm256_set1_pd(1.0), ai1);
            __m256d vc0 = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), bi0), via0);
            __m256d vc1 = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), bi1), via1);

            __m256d qp0 = _mm256_mul_pd(_mm256_broadcast_sd(&Pq0[0]), via0);
            __m256d qp1 = _mm256_mul_pd(_mm256_broadcast_sd(&Pq1[0]), via1);

            /* m=0: acc[0] += pw0*qp0 + pw1*qp1 */
            __m256d av = _mm256_load_pd(&acc[0]);
            av = _mm256_fmadd_pd(pw0, qp0, av);
            av = _mm256_fmadd_pd(pw1, qp1, av);
            _mm256_store_pd(&acc[0], av);

            for (int m = 0; m < n - 1; m++) {
                /* Two independent recurrences — ILP across FP ports */
                __m256d sp0 = _mm256_mul_pd(_mm256_broadcast_sd(&Pq0[m+1]), via0);
                __m256d sp1 = _mm256_mul_pd(_mm256_broadcast_sd(&Pq1[m+1]), via1);
                qp0 = _mm256_fmadd_pd(vc0, qp0, sp0);  /* port 0 */
                qp1 = _mm256_fmadd_pd(vc1, qp1, sp1);  /* port 1 */

                av = _mm256_load_pd(&acc[(m+1)*4]);
                av = _mm256_fmadd_pd(pw0, qp0, av);
                av = _mm256_fmadd_pd(pw1, qp1, av);
                _mm256_store_pd(&acc[(m+1)*4], av);
            }
        } else if (all_td_both) {
            __m256d vib0 = _mm256_div_pd(_mm256_set1_pd(1.0), bi0);
            __m256d vib1 = _mm256_div_pd(_mm256_set1_pd(1.0), bi1);
            __m256d vc0 = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), ai0), vib0);
            __m256d vc1 = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), ai1), vib1);

            __m256d qp0 = _mm256_mul_pd(_mm256_broadcast_sd(&Pq0[n]), vib0);
            __m256d qp1 = _mm256_mul_pd(_mm256_broadcast_sd(&Pq1[n]), vib1);

            __m256d av = _mm256_load_pd(&acc[(n-1)*4]);
            av = _mm256_fmadd_pd(pw0, qp0, av);
            av = _mm256_fmadd_pd(pw1, qp1, av);
            _mm256_store_pd(&acc[(n-1)*4], av);

            for (int m = n - 1; m >= 1; m--) {
                __m256d sp0 = _mm256_mul_pd(_mm256_broadcast_sd(&Pq0[m]), vib0);
                __m256d sp1 = _mm256_mul_pd(_mm256_broadcast_sd(&Pq1[m]), vib1);
                qp0 = _mm256_fmadd_pd(vc0, qp0, sp0);
                qp1 = _mm256_fmadd_pd(vc1, qp1, sp1);

                av = _mm256_load_pd(&acc[(m-1)*4]);
                av = _mm256_fmadd_pd(pw0, qp0, av);
                av = _mm256_fmadd_pd(pw1, qp1, av);
                _mm256_store_pd(&acc[(m-1)*4], av);
            }
        } else {
            /* Rare mixed case: fall back to 1-at-a-time for this pair */
            for (int qx = q; qx <= q+1; qx++) {
                double wqx = wq_store[qx]; if (wqx == 0) continue;
                double logvx = logv_store[qx];
                const double *Pqx = P_store + (size_t)qx * ps;
                double ai_arr[4], bi_arr[4], pw_arr[4];
                int any = 0;
                for (int b = 0; b < batch; b++) {
                    double arg = Si[b] * logvx;
                    double ai = (arg < -700) ? 0.0 : exp(arg);
                    ai_arr[b] = ai; bi_arr[b] = 1.0 - ai;
                    double vp = ai * inv_v_store[qx];
                    double pw = wqx * Si[b] * vp;
                    if (pw == 0 || !isfinite(pw)) pw = 0;
                    pw_arr[b] = pw; if (pw != 0) any = 1;
                }
                if (!any) continue;
                int bux = _mm256_movemask_pd(
                    _mm256_cmp_pd(_mm256_loadu_pd(ai_arr), half, _CMP_GT_OQ));
                if ((bux & active_mask) == active_mask) {
                    __m256d vai=_mm256_loadu_pd(ai_arr), vbi_=_mm256_loadu_pd(bi_arr);
                    __m256d via=_mm256_div_pd(_mm256_set1_pd(1.0),vai);
                    __m256d vpw=_mm256_loadu_pd(pw_arr);
                    __m256d vc=_mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(),vbi_),via);
                    __m256d qp=_mm256_mul_pd(_mm256_broadcast_sd(&Pqx[0]),via);
                    _mm256_store_pd(&acc[0],_mm256_fmadd_pd(vpw,qp,_mm256_load_pd(&acc[0])));
                    for (int m=0;m<n-1;m++) {
                        __m256d dm=_mm256_mul_pd(_mm256_broadcast_sd(&Pqx[m+1]),via);
                        qp=_mm256_fmadd_pd(vc,qp,dm);
                        _mm256_store_pd(&acc[(m+1)*4],_mm256_fmadd_pd(vpw,qp,_mm256_load_pd(&acc[(m+1)*4])));
                    }
                } else if ((bux & active_mask) == 0) {
                    __m256d vai=_mm256_loadu_pd(ai_arr), vbi_=_mm256_loadu_pd(bi_arr);
                    __m256d vib=_mm256_div_pd(_mm256_set1_pd(1.0),vbi_);
                    __m256d vpw=_mm256_loadu_pd(pw_arr);
                    __m256d vc=_mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(),vai),vib);
                    __m256d qp=_mm256_mul_pd(_mm256_broadcast_sd(&Pqx[n]),vib);
                    _mm256_store_pd(&acc[(n-1)*4],_mm256_fmadd_pd(vpw,qp,_mm256_load_pd(&acc[(n-1)*4])));
                    for (int m=n-1;m>=1;m--) {
                        __m256d dm=_mm256_mul_pd(_mm256_broadcast_sd(&Pqx[m]),vib);
                        qp=_mm256_fmadd_pd(vc,qp,dm);
                        _mm256_store_pd(&acc[(m-1)*4],_mm256_fmadd_pd(vpw,qp,_mm256_load_pd(&acc[(m-1)*4])));
                    }
                } else {
                    for (int b=0;b<batch;b++) {
                        if (pw_arr[b]==0) continue;
                        double aib=ai_arr[b],bib=bi_arr[b],pwb=pw_arr[b];
                        if (aib>0.5) {
                            double ia=1.0/aib,c_s=-bib*ia,qm=Pqx[0]*ia;
                            acc[b]+=pwb*qm;
                            for (int m=0;m<n-1;m++){qm=c_s*qm+Pqx[m+1]*ia;acc[(m+1)*4+b]+=pwb*qm;}
                        } else {
                            double ib=1.0/bib,c_s=-aib*ib,qm=Pqx[n]*ib;
                            acc[(n-1)*4+b]+=pwb*qm;
                            for (int m=n-1;m>=1;m--){qm=c_s*qm+Pqx[m]*ib;acc[(m-1)*4+b]+=pwb*qm;}
                        }
                    }
                }
            }
        }
    }

    /* ── Handle odd leftover quad point ───────────────────────── */
    for (; q < Q; q++) {
        double wq = wq_store[q]; if (wq == 0) continue;
        double logv = logv_store[q];
        const double *Pq = P_store + (size_t)q * ps;

        __m256d arg = _mm256_mul_pd(vSi, _mm256_set1_pd(logv));
        __m256d ai_v = fast_exp_4(arg);
        __m256d bi_v = _mm256_sub_pd(_mm256_set1_pd(1.0), ai_v);
        __m256d vp_v = _mm256_mul_pd(ai_v, _mm256_set1_pd(inv_v_store[q]));
        __m256d pw_v = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(wq), vSi), vp_v);

        __m256d half = _mm256_set1_pd(0.5);
        int bm = _mm256_movemask_pd(_mm256_cmp_pd(ai_v, half, _CMP_GT_OQ));
        int active_mask = (1 << batch) - 1;

        if ((bm & active_mask) == active_mask) {
            __m256d via = _mm256_div_pd(_mm256_set1_pd(1.0), ai_v);
            __m256d vc  = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), bi_v), via);
            __m256d qp  = _mm256_mul_pd(_mm256_broadcast_sd(&Pq[0]), via);
            _mm256_store_pd(&acc[0], _mm256_fmadd_pd(pw_v, qp, _mm256_load_pd(&acc[0])));
            for (int m = 0; m < n-1; m++) {
                __m256d dm = _mm256_mul_pd(_mm256_broadcast_sd(&Pq[m+1]), via);
                qp = _mm256_fmadd_pd(vc, qp, dm);
                _mm256_store_pd(&acc[(m+1)*4],
                    _mm256_fmadd_pd(pw_v, qp, _mm256_load_pd(&acc[(m+1)*4])));
            }
        } else if ((bm & active_mask) == 0) {
            __m256d vib = _mm256_div_pd(_mm256_set1_pd(1.0), bi_v);
            __m256d vc  = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), ai_v), vib);
            __m256d qp  = _mm256_mul_pd(_mm256_broadcast_sd(&Pq[n]), vib);
            _mm256_store_pd(&acc[(n-1)*4], _mm256_fmadd_pd(pw_v, qp, _mm256_load_pd(&acc[(n-1)*4])));
            for (int m = n-1; m >= 1; m--) {
                __m256d dm = _mm256_mul_pd(_mm256_broadcast_sd(&Pq[m]), vib);
                qp = _mm256_fmadd_pd(vc, qp, dm);
                _mm256_store_pd(&acc[(m-1)*4],
                    _mm256_fmadd_pd(pw_v, qp, _mm256_load_pd(&acc[(m-1)*4])));
            }
        } else {
            double ai_a[4], bi_a[4], pw_a[4];
            _mm256_storeu_pd(ai_a, ai_v); _mm256_storeu_pd(bi_a, bi_v); _mm256_storeu_pd(pw_a, pw_v);
            for (int b = 0; b < batch; b++) {
                if (pw_a[b] == 0) continue;
                double ai=ai_a[b],bi=bi_a[b],pw=pw_a[b];
                if (ai > 0.5) {
                    double ia=1.0/ai,c_s=-bi*ia,qm=Pq[0]*ia;
                    acc[b]+=pw*qm;
                    for (int m=0;m<n-1;m++){qm=c_s*qm+Pq[m+1]*ia;acc[(m+1)*4+b]+=pw*qm;}
                } else {
                    double ib=1.0/bi,c_s=-ai*ib,qm=Pq[n]*ib;
                    acc[(n-1)*4+b]+=pw*qm;
                    for (int m=n-1;m>=1;m--){qm=c_s*qm+Pq[m]*ib;acc[(m-1)*4+b]+=pw*qm;}
                }
            }
        }
    }
}

/* ── Public entry point ───────────────────────────────────────── */

void icm_avx2(int n, const double *S, int Q, const QP *pts, double *prob) {
    memset(prob, 0, (size_t)n * n * sizeof(double));

    size_t ps = (size_t)(n + 1);
    double *Ps  = (double *)malloc((size_t)Q * ps * sizeof(double));
    double *lv  = (double *)malloc(Q * sizeof(double));
    double *wqs = (double *)malloc(Q * sizeof(double));
    build_all_polys(n, S, Q, pts, Ps, lv, wqs);

    /* Precompute inv_v[q] = exp(-logv[q]) = 1/v.
       Then vp_i = exp((S_i-1)*logv) = exp(S_i*logv) * exp(-logv) = a_i * inv_v.
       This replaces 4 exp() calls per block per q with 4 multiplies. */
    double *inv_v = (double *)malloc(Q * sizeof(double));
    for (int q = 0; q < Q; q++)
        inv_v[q] = exp(-lv[q]);

    /* Sort players by stack */
    int *ord = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) ord[i] = i;
    for (int gap = n/2; gap > 0; gap /= 2)
        for (int i = gap; i < n; i++) {
            int t = ord[i]; double sv = S[t]; int j = i;
            while (j >= gap && S[ord[j-gap]] > sv) { ord[j] = ord[j-gap]; j -= gap; }
            ord[j] = t;
        }

    /* q_lo per sorted player */
    int *ql = (int *)malloc(n * sizeof(int));
    for (int k = 0; k < n; k++) {
        double th = -700.0 / S[ord[k]];
        int lo = 0, hi = Q;
        while (lo < hi) { int mid = (lo+hi)/2; if (lv[mid] < th) lo = mid+1; else hi = mid; }
        ql[k] = lo;
    }

    /* Divide + accumulate, player-major */
    double *acc = (double *)_mm_malloc((size_t)n * 4 * sizeof(double), 32);

    for (int k0 = 0; k0 < n; k0 += 4) {
        int batch = ((k0+4) <= n) ? 4 : (n-k0);
        memset(acc, 0, (size_t)n * 4 * sizeof(double));
        double Si[4], Sm[4]; int qs = Q, pidx[4];
        for (int b = 0; b < batch; b++) {
            pidx[b] = ord[k0+b]; Si[b] = S[pidx[b]]; Sm[b] = Si[b]-1;
            if (ql[k0+b] < qs) qs = ql[k0+b];
        }
        for (int b = batch; b < 4; b++) { Si[b] = 1; Sm[b] = 0; }

        divide_accum_4(n, batch, Q, qs, Si, Sm, Ps, lv, wqs, inv_v, acc);

        for (int b = 0; b < batch; b++) {
            double *row = prob + (size_t)pidx[b] * n;
            for (int m = 0; m < n; m++) row[m] = acc[m*4+b];
        }
    }

    _mm_free(acc); free(ord); free(ql); free(Ps); free(lv); free(wqs); free(inv_v);
}
