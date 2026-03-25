/*
 * icm.c — ICM equity computation library implementation
 * See icm.h for the public API.
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include "icm.h"
#include "fft_config.h"  /* device-specific calibrated FFT times */
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

/* Number of OpenMP threads for quadrature parallelism */
#ifndef OMP_NUM_THREADS_DEFAULT
#define OMP_NUM_THREADS_DEFAULT 16
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ══════════════════════════════════════════════════════════════
   Quadrature (erfc_trap)
   ══════════════════════════════════════════════════════════════ */

typedef struct { double logv, w; } QP;

static double log_Phi(double y) {
    if (y >= 0) return log1p(-erfc(y/sqrt(2.0))/2.0);
    else        return log(erfc(-y/sqrt(2.0))/2.0);
}

static void make_nodes(int Q, double Smax, QP *pts) {
    double y_lo = -7.7;
    double y_hi = sqrt(2.0) * sqrt(log(Smax) + 25.0);
    if (y_hi < 6.5) y_hi = 6.5;
    double h = (y_hi - y_lo) / (Q - 1);
    for (int q = 0; q < Q; q++) {
        double y = y_lo + q * h;
        pts[q].logv = log_Phi(y);
        pts[q].w = h * exp(-0.5*y*y) / sqrt(2*M_PI);
    }
}

/* ══════════════════════════════════════════════════════════════
   V1 closed form
   ══════════════════════════════════════════════════════════════ */

static void v1_exact(int n, const double *S, double *V1) {
    for (int i = 0; i < n; i++) {
        double v = 1.0;
        for (int j = 0; j < n; j++)
            if (j != i) v += S[i] / (S[i] + S[j]);
        V1[i] = v;
    }
}

/* V2 closed form: V2(i) = Σ_{j<k, j≠i, k≠i} S_i / (S_i + S_j + S_k)
 * Uses quadratic payout p[m] = C(n-1-m, 2).
 * Each term = Pr(i eliminated first in triple {i,j,k}) under MH model.
 * O(n^3) — only feasible for small n. */
static void v2_exact(int n, const double *S, double *V2) {
    for (int i = 0; i < n; i++) {
        double v = 0;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            for (int k = j + 1; k < n; k++) {
                if (k == i) continue;
                v += S[i] / (S[i] + S[j] + S[k]);
            }
        }
        V2[i] = v;
    }
}

/* ══════════════════════════════════════════════════════════════
   ENGINE INTERFACE
   ══════════════════════════════════════════════════════════════ */

typedef void (*EquityEngine)(int n, const double *a,
                             const double *payout, int k,
                             double *inner, void *ctx);

/* ══════════════════════════════════════════════════════════════
   FFTW WISDOM — persist MEASURE results across runs
   ══════════════════════════════════════════════════════════════ */

#define WISDOM_FILE "fftw_wisdom.dat"

static void wisdom_load(void) {
    fftw_import_wisdom_from_filename(WISDOM_FILE);
}

static void wisdom_save(void) {
    fftw_export_wisdom_to_filename(WISDOM_FILE);
}

/* Forward declarations for helpers defined below */
static int next_pow2(int n);

/* ══════════════════════════════════════════════════════════════
   FFTW PLAN CACHE — create once, reuse across all quad points
   ══════════════════════════════════════════════════════════════ */

typedef struct {
    int fft_n;            /* padded size (FFTW-friendly composite) */
    fftw_plan fwd_plan;   /* r2c forward */
    fftw_plan inv_plan;   /* c2r inverse */
    double *rbuf;         /* real buffer [fft_n] */
    fftw_complex *cbuf;   /* complex buffer [fft_n/2+1] */
} FFTPlan;

typedef struct {
    FFTPlan *plans;       /* dynamically allocated array of FFTW-friendly sizes */
    int n_plans;
    int plans_cap;        /* allocated capacity */
    /* Scratch buffers for polymul/correlate (second operand) */
    double *rbuf2;
    fftw_complex *cbuf2;
    fftw_complex *cbuf3;  /* extra buffer for correlate_fft_pair (saves FFT(g)) */
    int max_fft_n;
} FFTCache;

/* Create FFT plan cache for a specific set of FFT sizes.
 * sizes[]: array of FFTW-friendly sizes to create plans for (must be sorted ascending).
 * n_sizes: number of sizes. */
static FFTCache *fft_cache_create_sizes(const int *sizes, int n_sizes) {
    FFTCache *fc = (FFTCache *)calloc(1, sizeof(FFTCache));
    int max_fft_n = (n_sizes > 0) ? sizes[n_sizes - 1] : 4;
    fc->max_fft_n = max_fft_n;

    fc->plans_cap = n_sizes;
    fc->plans = (FFTPlan *)calloc(n_sizes, sizeof(FFTPlan));

    /* Create FFTW_MEASURE plans for the requested sizes. With wisdom loaded
     * from a previous run, plan creation is <1ms each. */
    fc->n_plans = 0;
    for (int i = 0; i < n_sizes; i++) {
        int sz = sizes[i];
        FFTPlan *p = &fc->plans[fc->n_plans];
        p->fft_n = sz;
        p->rbuf = fftw_malloc(sz * sizeof(double));
        p->cbuf = fftw_malloc((sz/2 + 1) * sizeof(fftw_complex));
        memset(p->rbuf, 0, sz * sizeof(double));
        /* Try MEASURE with wisdom; fall back to ESTIMATE if no wisdom exists
         * (MEASURE without wisdom benchmarks from scratch — minutes at large sizes) */
        p->fwd_plan = fftw_plan_dft_r2c_1d(sz, p->rbuf, p->cbuf, FFTW_MEASURE | FFTW_WISDOM_ONLY);
        p->inv_plan = fftw_plan_dft_c2r_1d(sz, p->cbuf, p->rbuf, FFTW_MEASURE | FFTW_WISDOM_ONLY);
        if (!p->fwd_plan || !p->inv_plan) {
            if (p->fwd_plan) fftw_destroy_plan(p->fwd_plan);
            if (p->inv_plan) fftw_destroy_plan(p->inv_plan);
            p->fwd_plan = fftw_plan_dft_r2c_1d(sz, p->rbuf, p->cbuf, FFTW_ESTIMATE);
            p->inv_plan = fftw_plan_dft_c2r_1d(sz, p->cbuf, p->rbuf, FFTW_ESTIMATE);
        }
        fc->n_plans++;
    }

    /* Scratch buffers (for the largest size) */
    fc->rbuf2 = fftw_malloc(max_fft_n * sizeof(double));
    fc->cbuf2 = fftw_malloc((max_fft_n/2 + 1) * sizeof(fftw_complex));
    fc->cbuf3 = fftw_malloc((max_fft_n/2 + 1) * sizeof(fftw_complex));

    /* Save wisdom so next run skips MEASURE benchmarking */
    wisdom_save();
    return fc;
}


static void fft_cache_destroy(FFTCache *fc) {
    for (int i = 0; i < fc->n_plans; i++) {
        fftw_destroy_plan(fc->plans[i].fwd_plan);
        fftw_destroy_plan(fc->plans[i].inv_plan);
        fftw_free(fc->plans[i].rbuf);
        fftw_free(fc->plans[i].cbuf);
    }
    free(fc->plans);
    fftw_free(fc->rbuf2);
    fftw_free(fc->cbuf2);
    fftw_free(fc->cbuf3);
    free(fc);
}

/* Direct-index lookup: fft_n → plan pointer. Avoids linear scan. */
static FFTPlan *fft_cache_get(FFTCache *fc, int needed_n) {
    /* Plans are sorted ascending. The caller always passes next_pow2(L)
     * which is an exact match for some plan. Use binary search. */
    int lo = 0, hi = fc->n_plans - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (fc->plans[mid].fft_n < needed_n)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < fc->n_plans && fc->plans[lo].fft_n >= needed_n)
        return &fc->plans[lo];
    return NULL;
}

/* ══════════════════════════════════════════════════════════════
   SCHOOLBOOK KERNELS — with inlined small-size fast paths
   ══════════════════════════════════════════════════════════════ */

/* Inline multiply for size 2 (tree leaves: degree 1 × degree 1) */
static inline void polymul_2x2(const double *a, const double *b,
                                double *c, int k) {
    c[0] = a[0] * b[0];
    if (k > 1) c[1] = a[0]*b[1] + a[1]*b[0];
    if (k > 2) c[2] = a[1] * b[1];
    if (k > 3) c[3] = 0;
}

static void polymul_modk(const double *restrict a, int na,
                         const double *restrict b, int nb,
                         double *restrict c, int k) {
    if (na == 2 && nb == 2 && k <= 4) { polymul_2x2(a, b, c, k); return; }
    memset(c, 0, k * sizeof(double));
    for (int i = 0; i < na && i < k; i++) {
        double ai = a[i];
        if (ai == 0.0) continue;
        int jmax = nb;
        if (i + jmax > k) jmax = k - i;
        double *restrict ci = c + i;
        for (int j = 0; j < jmax; j++)
            ci[j] += ai * b[j];
    }
}

/* Inline correlate for size 2 (leaf level: out[m] = P[0]*g[m] + P[1]*g[m+1]) */
static inline void correlate_2(const double *restrict g,
                                const double *restrict P,
                                double *restrict out, int len_out) {
    for (int m = 0; m < len_out; m++)
        out[m] = P[0] * g[m] + P[1] * g[m + 1];
}

static void correlate_school(const double *restrict g, int len_g,
                             const double *restrict P, int len_P,
                             double *restrict out, int len_out) {
    /* Fast path for size 2 (dominates at leaf level) */
    if (len_P == 2 && len_out + 1 <= len_g) {
        correlate_2(g, P, out, len_out);
        return;
    }
    for (int m = 0; m < len_out; m++) {
        double sum = 0;
        int jmax = len_P;
        if (m + jmax > len_g) jmax = len_g - m;
        if (jmax <= 0) { out[m] = 0; continue; }
        const double *restrict gm = g + m;
        for (int j = 0; j < jmax; j++)
            sum += P[j] * gm[j];
        out[m] = sum;
    }
}

/* ══════════════════════════════════════════════════════════════
   FFT KERNELS — polynomial multiply and correlate via FFTW
   ══════════════════════════════════════════════════════════════ */

/* FFT vs schoolbook is now decided per-level in tree_ctx_create_ex2()
 * using calibrated FFT times from fft_config.h. No global crossover. */

/* Helper: compute next power of 2 >= n */
static inline int next_pow2(int n) {
    int p = 1; while (p < n) p <<= 1; return p;
}

/* Helper: compute next FFTW-friendly size >= n (products of 2,3,5,7).
 * FFTW is highly optimized for these composite sizes, often 30-50% faster
 * Composite FFTW-friendly sizes help at saturated levels when k is non-power-of-2.
 * k is rounded to next_fftw_size(k) (not next_pow2) to preserve the benefit. */

/* Sorted array of all 7-smooth numbers up to 131072. ~500 entries = 2KB.
 * Binary search gives O(log 500) ≈ 9 comparisons without polluting L1
 * (unlike a 512KB direct-index table). */
static int smooth_nums[800];  /* 749 smooth numbers up to 131072 */
static int n_smooth = 0;

static void build_fftw_size_table(void) {
    if (n_smooth > 0) return;
    for (int a = 1; a <= 131072; a *= 2)
        for (int b = a; b <= 131072; b *= 3)
            for (int c = b; c <= 131072; c *= 5)
                for (int d = c; d <= 131072; d *= 7)
                    smooth_nums[n_smooth++] = d;
    /* Insertion sort (only ~500 elements, one-time) */
    for (int i = 1; i < n_smooth; i++) {
        int key = smooth_nums[i], j = i - 1;
        while (j >= 0 && smooth_nums[j] > key) { smooth_nums[j+1] = smooth_nums[j]; j--; }
        smooth_nums[j+1] = key;
    }
}

/* next_fftw_size: binary search for smallest smooth number >= n.
 * Only called when next_pow2 wastes significant padding. */
static int next_fftw_size(int n) {
    /* Binary search in sorted smooth numbers (~2KB, stays in L1) */
    int lo = 0, hi = n_smooth - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (smooth_nums[mid] < n) lo = mid + 1; else hi = mid;
    }
    return (lo < n_smooth) ? smooth_nums[lo] : next_pow2(n);
}

/* best_k_pad: find the k' >= k that minimizes the saturated-level FFT cost.
 * At saturated levels, build multiply conv_len = 2*k' - 1. The FFT must be
 * at least this large (no wrapping for standard multiply). We search smooth
 * numbers from k up to k*1.15, computing the calibrated FFT cost for the
 * resulting conv_len, and pick the k' with the cheapest total. */
static int best_k_pad(int k) {
    if (k <= 2) return k;
    /* Power-of-2 fast path: already optimal */
    if ((k & (k - 1)) == 0) return k;

    /* Binary search for first smooth >= k */
    int lo = 0, hi = n_smooth - 1;
    while (lo < hi) { int mid = (lo+hi)>>1; if (smooth_nums[mid] < k) lo = mid+1; else hi = mid; }

    int ceil_k = k + k / 8;  /* search up to k * 1.125 */
    if (ceil_k < k + 4) ceil_k = k + 4;
    double best_cost = 1e18;
    int best_k = k;

    for (int i = lo; i < n_smooth && smooth_nums[i] <= ceil_k; i++) {
        int kp = smooth_nums[i];
        /* Find calibrated FFT cost for conv_len = 2*kp - 1 (saturated-level build) */
        int conv_len = 2 * kp - 1;
        /* Smallest smooth >= conv_len */
        int lo2 = 0, hi2 = N_CALIBRATED_SIZES - 1;
        while (lo2 < hi2) { int m = (lo2+hi2)>>1; if (calib_sizes[m] < conv_len) lo2 = m+1; else hi2 = m; }
        if (lo2 >= N_CALIBRATED_SIZES) continue;
        double fft_cost = calib_times_ns[lo2];
        /* Slight penalty for padding: each extra coefficient adds work at saturated levels.
         * Approximate: (kp - k) extra coefficients per saturated-level multiply.
         * But this is tiny compared to FFT cost, so we mainly optimize FFT cost. */
        double pad_penalty = (kp - k) * 0.5;  /* ~0.5ns per extra coeff (negligible) */
        double cost = fft_cost + pad_penalty;
        if (cost < best_cost) {
            best_cost = cost;
            best_k = kp;
        }
    }
    return best_k;
}

/* fastest_fft_ge: find the fastest calibrated FFT size >= n.
 * Unlike next_fftw_size (smallest smooth >= n), this picks the smooth with
 * the lowest calibrated time. Crucial because e.g. 245 (5×7²) is 1.55x slower
 * than 256 (2⁸) despite being smaller. Searches up to n*1.3 or next_pow2. */
static int fastest_fft_ge(int n) {
    if (n <= 1) return 2;
    /* Power-of-2 fast path */
    int p2 = next_pow2(n);
    if (p2 == n) return n;

    /* Binary search for first calib_size >= n */
    int lo = 0, hi = N_CALIBRATED_SIZES - 1;
    while (lo < hi) { int mid = (lo+hi)>>1; if (calib_sizes[mid] < n) lo = mid+1; else hi = mid; }

    /* Search up to next_pow2 (which is always a valid upper bound) */
    double best_cost = 1e18;
    int best = p2;
    for (int i = lo; i < N_CALIBRATED_SIZES && calib_sizes[i] <= p2; i++) {
        if (calib_times_ns[i] < best_cost) {
            best_cost = calib_times_ns[i];
            best = calib_sizes[i];
        }
    }
    return best;
}

/* Cyclic FFT multiply for below-saturation tree levels.
 * Polys have actual degree d = na/2, stored in na = 2d slots.
 * Product has degree 2d. Uses calibrated best FFT size, potentially wrapping
 * m > 0 extra coefficients to reach a faster FFT size.
 *
 * Cyclic convolution of size N = fft_n wraps terms C[N..2d] to C[0..2d-N].
 * The correction is a schoolbook product of the top (m+1) terms of each input:
 * cost = (m+1)² FMAs, where m = 2d - N. Typically m = 0..3. */
static void polymul_fft_cyclic(const double *a, int na,
                                const double *b, int nb,
                                double *c, int k,
                                FFTCache *fc,
                                int fft_n, int wrap_m) {
    int d = na / 2;
    int conv_len = 2 * d;
    FFTPlan *plan = fft_cache_get(fc, fft_n);
    fft_n = plan->fft_n;
    int cn = fft_n / 2 + 1;
    double inv = 1.0 / fft_n;

    /* FFT(a): copy d+1 terms, zero-pad rest */
    memcpy(plan->rbuf, a, (d + 1) * sizeof(double));
    if (d + 1 < fft_n) memset(plan->rbuf + d + 1, 0, (fft_n - d - 1) * sizeof(double));
    fftw_execute(plan->fwd_plan);

    /* FFT(b) */
    memcpy(fc->rbuf2, b, (d + 1) * sizeof(double));
    if (d + 1 < fft_n) memset(fc->rbuf2 + d + 1, 0, (fft_n - d - 1) * sizeof(double));
    fftw_execute_dft_r2c(plan->fwd_plan, fc->rbuf2, fc->cbuf2);

    /* Pointwise multiply */
    for (int i = 0; i < cn; i++) {
        double re = plan->cbuf[i][0] * fc->cbuf2[i][0]
                  - plan->cbuf[i][1] * fc->cbuf2[i][1];
        double im = plan->cbuf[i][0] * fc->cbuf2[i][1]
                  + plan->cbuf[i][1] * fc->cbuf2[i][0];
        plan->cbuf[i][0] = re;
        plan->cbuf[i][1] = im;
    }
    fftw_execute(plan->inv_plan);

    /* Extract: positions 0..fft_n-1 come from the IFFT (cyclic values).
     * Positions fft_n..2d come entirely from the correction below. */
    int fft_out = (fft_n < k) ? fft_n : k;
    for (int i = 0; i < fft_out; i++) c[i] = plan->rbuf[i] * inv;
    /* Zero everything above fft_n (will be filled by correction if needed) */
    if (fft_out < k) memset(c + fft_out, 0, (k - fft_out) * sizeof(double));

    /* Correction: the cyclic convolution of size fft_n wraps positions fft_n..2d
     * back to positions 0..wrap_m. Compute the true values at those high positions
     * via schoolbook of the relevant input terms, then:
     *   - subtract from the wrapped low positions (to undo the aliasing)
     *   - place at the correct high positions */
    for (int i = 0; i <= wrap_m; i++) {
        int pos = fft_n + i;  /* true position of this wrapped term */
        double high = 0;
        int j_lo = pos - d; if (j_lo < 0) j_lo = 0;
        int j_hi = d; if (j_hi > pos) j_hi = pos;
        for (int j = j_lo; j <= j_hi; j++)
            high += a[j] * b[pos - j];
        if (i < fft_out) c[i] -= high;   /* undo aliasing at wrapped position */
        if (pos < k) c[pos] = high;       /* place at correct high position */
    }
}

/* General cyclic FFT multiply with m-wrap correction.
 * c[0..k-1] = (a[0..na-1] * b[0..nb-1]) mod x^k, using cyclic FFT of size
 * fft_n < na+nb-1 with wrap_m = (na+nb-1) - fft_n correction terms.
 * Works for any input sizes (below-sat, saturated, or mixed).
 * If fft_a_out/fft_b_out are non-NULL, caches FFT(a)/FFT(b) for reuse. */
static void polymul_fft_wrap(const double *a, int na,
                              const double *b, int nb,
                              double *c, int k,
                              FFTCache *fc,
                              fftw_complex *fft_a_out,
                              fftw_complex *fft_b_out,
                              int fft_n, int wrap_m) {
    FFTPlan *plan = fft_cache_get(fc, fft_n);
    fft_n = plan->fft_n;
    int cn = fft_n / 2 + 1;
    double inv = 1.0 / fft_n;
    int conv_len = na + nb - 1;

    /* FFT(a) */
    int copy_a = (na < fft_n) ? na : fft_n;
    memcpy(plan->rbuf, a, copy_a * sizeof(double));
    if (copy_a < fft_n) memset(plan->rbuf + copy_a, 0, (fft_n - copy_a) * sizeof(double));
    fftw_execute(plan->fwd_plan);
    if (fft_a_out) memcpy(fft_a_out, plan->cbuf, cn * sizeof(fftw_complex));

    /* FFT(b) */
    int copy_b = (nb < fft_n) ? nb : fft_n;
    memcpy(fc->rbuf2, b, copy_b * sizeof(double));
    if (copy_b < fft_n) memset(fc->rbuf2 + copy_b, 0, (fft_n - copy_b) * sizeof(double));
    fftw_execute_dft_r2c(plan->fwd_plan, fc->rbuf2, fc->cbuf2);
    if (fft_b_out) memcpy(fft_b_out, fc->cbuf2, cn * sizeof(fftw_complex));

    /* Pointwise multiply */
    for (int i = 0; i < cn; i++) {
        double re = plan->cbuf[i][0] * fc->cbuf2[i][0]
                  - plan->cbuf[i][1] * fc->cbuf2[i][1];
        double im = plan->cbuf[i][0] * fc->cbuf2[i][1]
                  + plan->cbuf[i][1] * fc->cbuf2[i][0];
        plan->cbuf[i][0] = re;
        plan->cbuf[i][1] = im;
    }
    fftw_execute(plan->inv_plan);

    /* Extract cyclic result */
    int fft_out = (fft_n < k) ? fft_n : k;
    for (int i = 0; i < fft_out; i++) c[i] = plan->rbuf[i] * inv;
    if (fft_out < k) memset(c + fft_out, 0, (k - fft_out) * sizeof(double));

    /* Correction: undo cyclic aliasing at positions 0..wrap_m */
    int da = na - 1, db = nb - 1;  /* max degrees */
    for (int i = 0; i <= wrap_m; i++) {
        int pos = fft_n + i;
        double high = 0;
        int j_lo = pos - db; if (j_lo < 0) j_lo = 0;
        int j_hi = da; if (j_hi > pos) j_hi = pos;
        for (int j = j_lo; j <= j_hi; j++)
            high += a[j] * b[pos - j];
        if (i < fft_out) c[i] -= high;
        if (pos < k) c[pos] = high;
    }
}

/* Polynomial multiply mod x^k using FFT.
 * c[0..k-1] = (a[0..na-1] * b[0..nb-1]) mod x^k
 * Optionally caches FFT(a) and FFT(b) into fft_a_out/fft_b_out for reuse.
 * target_fft_n: if > 0, use this FFT size instead of the minimum. */
static void polymul_fft_modk(const double *a, int na,
                             const double *b, int nb,
                             double *c, int k,
                             FFTCache *fc,
                             fftw_complex *fft_a_out,
                             fftw_complex *fft_b_out,
                             int fft_n) {
    FFTPlan *plan = fft_cache_get(fc, fft_n);
    fft_n = plan->fft_n;
    int cn = fft_n / 2 + 1;
    double inv = 1.0 / fft_n;

    /* FFT(a): copy data then zero-pad remainder only */
    memcpy(plan->rbuf, a, na * sizeof(double));
    if (na < fft_n) memset(plan->rbuf + na, 0, (fft_n - na) * sizeof(double));
    fftw_execute(plan->fwd_plan);
    if (fft_a_out)
        memcpy(fft_a_out, plan->cbuf, cn * sizeof(fftw_complex));

    /* FFT(b): copy then zero-pad */
    memcpy(fc->rbuf2, b, nb * sizeof(double));
    if (nb < fft_n) memset(fc->rbuf2 + nb, 0, (fft_n - nb) * sizeof(double));
    fftw_execute_dft_r2c(plan->fwd_plan, fc->rbuf2, fc->cbuf2);
    /* Cache FFT(b) if requested */
    if (fft_b_out)
        memcpy(fft_b_out, fc->cbuf2, cn * sizeof(fftw_complex));

    /* Pointwise complex multiply: cbuf = cbuf * cbuf2 */
    for (int i = 0; i < cn; i++) {
        double re = plan->cbuf[i][0] * fc->cbuf2[i][0]
                  - plan->cbuf[i][1] * fc->cbuf2[i][1];
        double im = plan->cbuf[i][0] * fc->cbuf2[i][1]
                  + plan->cbuf[i][1] * fc->cbuf2[i][0];
        plan->cbuf[i][0] = re;
        plan->cbuf[i][1] = im;
    }

    fftw_execute(plan->inv_plan);

    int conv_len = na + nb - 1;
    int out_len = (conv_len < k) ? conv_len : k;
    for (int i = 0; i < out_len; i++)
        c[i] = plan->rbuf[i] * inv;
    if (k > conv_len)
        memset(c + conv_len, 0, (k - conv_len) * sizeof(double));
}

/* Correlate via FFT (from scratch — FFTs both g and P):
 * out[m] = sum_j P[j] * g[m+j] */
static void correlate_fft(const double *g, int len_g,
                          const double *P, int len_P,
                          double *out, int len_out,
                          FFTCache *fc, int fft_n) {
    FFTPlan *plan = fft_cache_get(fc, fft_n);
    fft_n = plan->fft_n;
    int cn = fft_n / 2 + 1;
    double inv = 1.0 / fft_n;

    memcpy(plan->rbuf, g, len_g * sizeof(double));
    if (len_g < fft_n) memset(plan->rbuf + len_g, 0, (fft_n - len_g) * sizeof(double));
    fftw_execute(plan->fwd_plan);

    for (int j = 0; j < len_P; j++)
        fc->rbuf2[j] = P[len_P - 1 - j];
    if (len_P < fft_n) memset(fc->rbuf2 + len_P, 0, (fft_n - len_P) * sizeof(double));
    fftw_execute_dft_r2c(plan->fwd_plan, fc->rbuf2, fc->cbuf2);

    for (int i = 0; i < cn; i++) {
        double re = plan->cbuf[i][0] * fc->cbuf2[i][0]
                  - plan->cbuf[i][1] * fc->cbuf2[i][1];
        double im = plan->cbuf[i][0] * fc->cbuf2[i][1]
                  + plan->cbuf[i][1] * fc->cbuf2[i][0];
        plan->cbuf[i][0] = re;
        plan->cbuf[i][1] = im;
    }

    fftw_execute(plan->inv_plan);

    int offset = len_P - 1;
    for (int m = 0; m < len_out; m++) {
        int idx = m + offset;
        out[m] = (idx < fft_n) ? plan->rbuf[idx] * inv : 0;
    }
}

/* Correlate g with TWO polynomials, sharing the forward FFT of g.
 * Saves 1 forward FFT vs calling correlate_fft twice.
 * Used at non-cached levels where both correlates need the same g. */
static void correlate_fft_pair(const double *g, int len_g,
                                const double *PL, const double *PR, int len_P,
                                double *outL, double *outR, int len_out,
                                FFTCache *fc, int fft_n) {
    FFTPlan *plan = fft_cache_get(fc, fft_n);
    fft_n = plan->fft_n;
    int cn = fft_n / 2 + 1;
    double inv = 1.0 / fft_n;

    /* FFT(g) — done ONCE */
    memcpy(plan->rbuf, g, len_g * sizeof(double));
    if (len_g < fft_n) memset(plan->rbuf + len_g, 0, (fft_n - len_g) * sizeof(double));
    fftw_execute(plan->fwd_plan);

    /* Save FFT(g) into cbuf3 — cbuf2 is used for FFT(P) below */
    fftw_complex *g_hat = fc->cbuf3;
    memcpy(g_hat, plan->cbuf, cn * sizeof(fftw_complex));

    /* First correlate: g × PR → outL */
    for (int j = 0; j < len_P; j++) fc->rbuf2[j] = PR[len_P - 1 - j];
    if (len_P < fft_n) memset(fc->rbuf2 + len_P, 0, (fft_n - len_P) * sizeof(double));
    fftw_execute_dft_r2c(plan->fwd_plan, fc->rbuf2, fc->cbuf2);
    for (int i = 0; i < cn; i++) {
        plan->cbuf[i][0] = g_hat[i][0]*fc->cbuf2[i][0] - g_hat[i][1]*fc->cbuf2[i][1];
        plan->cbuf[i][1] = g_hat[i][0]*fc->cbuf2[i][1] + g_hat[i][1]*fc->cbuf2[i][0];
    }
    fftw_execute(plan->inv_plan);
    int offset = len_P - 1;
    for (int m = 0; m < len_out; m++)
        outL[m] = (m + offset < fft_n) ? plan->rbuf[m + offset] * inv : 0;

    /* Second correlate: g × PL → outR (reuse g_hat) */
    for (int j = 0; j < len_P; j++) fc->rbuf2[j] = PL[len_P - 1 - j];
    if (len_P < fft_n) memset(fc->rbuf2 + len_P, 0, (fft_n - len_P) * sizeof(double));
    fftw_execute_dft_r2c(plan->fwd_plan, fc->rbuf2, fc->cbuf2);
    for (int i = 0; i < cn; i++) {
        plan->cbuf[i][0] = g_hat[i][0]*fc->cbuf2[i][0] - g_hat[i][1]*fc->cbuf2[i][1];
        plan->cbuf[i][1] = g_hat[i][0]*fc->cbuf2[i][1] + g_hat[i][1]*fc->cbuf2[i][0];
    }
    fftw_execute(plan->inv_plan);
    for (int m = 0; m < len_out; m++)
        outR[m] = (m + offset < fft_n) ? plan->rbuf[m + offset] * inv : 0;
}

/* Correlate via FFT with CACHED FFT(P) — saves one forward FFT.
 * Cross-correlation: out[m] = sum_j P[j] * g[m+j]
 *                  = IFFT(FFT(g) * conj(FFT(P)))[m]
 * For real P, conj(FFT(P)) is trivially obtained from cached FFT(P).
 * cached_fft_P: pre-computed FFT(P zero-padded to cached_fft_n).
 * cached_fft_n: must equal next_pow2(len_g + len_P - 1). */
static void correlate_fft_cached(const double *g, int len_g,
                                 int len_P,
                                 double *out, int len_out,
                                 FFTCache *fc,
                                 const fftw_complex *cached_fft_P,
                                 int cached_fft_n) {
    if (cached_fft_n <= 0) return;

    FFTPlan *plan = fft_cache_get(fc, cached_fft_n);
    int fft_n = plan->fft_n;
    int cn = fft_n / 2 + 1;
    double inv = 1.0 / fft_n;

    /* FFT(g) */
    memcpy(plan->rbuf, g, len_g * sizeof(double));
    if (len_g < fft_n) memset(plan->rbuf + len_g, 0, (fft_n - len_g) * sizeof(double));
    fftw_execute(plan->fwd_plan);

    /* Pointwise: FFT(g) * conj(FFT(P)) — cross-correlation in freq domain */
    for (int i = 0; i < cn; i++) {
        double pr = cached_fft_P[i][0], pi = cached_fft_P[i][1];
        double gr = plan->cbuf[i][0], gi = plan->cbuf[i][1];
        plan->cbuf[i][0] = gr * pr + gi * pi;
        plan->cbuf[i][1] = gi * pr - gr * pi;
    }

    fftw_execute(plan->inv_plan);

    /* Cross-correlation result is directly at index m (no offset) */
    for (int m = 0; m < len_out; m++)
        out[m] = (m < fft_n) ? plan->rbuf[m] * inv : 0;
}

/* Cached correlate PAIR with m-wrap correction.
 * Shares FFT(g) across two correlations (one per sibling), AND uses
 * cached FFT(P) from the build phase. Saves one forward FFT per parent node
 * vs calling correlate_fft_cached_wrap twice. */
static void correlate_fft_cached_pair_wrap(
        const double *g, int len_g,
        const double *PL, const double *PR, int len_P,
        double *outL, double *outR, int len_out,
        FFTCache *fc,
        const fftw_complex *cached_fft_PL,
        const fftw_complex *cached_fft_PR,
        int cached_fft_n, int wrap_m) {
    FFTPlan *plan = fft_cache_get(fc, cached_fft_n);
    int fft_n = plan->fft_n;
    int cn = fft_n / 2 + 1;
    double inv = 1.0 / fft_n;
    int conv_len = len_g + len_P - 1;

    /* FFT(g) — done ONCE, saved to cbuf3 */
    int copy_g = (len_g < fft_n) ? len_g : fft_n;
    memcpy(plan->rbuf, g, copy_g * sizeof(double));
    if (copy_g < fft_n) memset(plan->rbuf + copy_g, 0, (fft_n - copy_g) * sizeof(double));
    fftw_execute(plan->fwd_plan);
    fftw_complex *g_hat = fc->cbuf3;
    memcpy(g_hat, plan->cbuf, cn * sizeof(fftw_complex));

    /* First correlate: g × PR → outL (using cached FFT(PR)) */
    for (int i = 0; i < cn; i++) {
        double pr = cached_fft_PR[i][0], pi = cached_fft_PR[i][1];
        double gr = g_hat[i][0], gi = g_hat[i][1];
        plan->cbuf[i][0] = gr * pr + gi * pi;
        plan->cbuf[i][1] = gi * pr - gr * pi;
    }
    fftw_execute(plan->inv_plan);
    for (int m = 0; m < len_out; m++)
        outL[m] = (m < fft_n) ? plan->rbuf[m] * inv : 0;
    if (wrap_m > 0) {
        for (int i = 0; i <= wrap_m; i++) {
            int pos = fft_n + i;
            if (pos >= conv_len) break;
            double high = 0;
            int j_max = len_g - pos;
            if (j_max > len_P) j_max = len_P;
            for (int j = 0; j < j_max; j++)
                high += PR[j] * g[pos + j];
            if (i < len_out) outL[i] -= high;
            if (pos < len_out) outL[pos] = high;
        }
    }

    /* Second correlate: g × PL → outR (reuse g_hat, using cached FFT(PL)) */
    for (int i = 0; i < cn; i++) {
        double pr = cached_fft_PL[i][0], pi = cached_fft_PL[i][1];
        double gr = g_hat[i][0], gi = g_hat[i][1];
        plan->cbuf[i][0] = gr * pr + gi * pi;
        plan->cbuf[i][1] = gi * pr - gr * pi;
    }
    fftw_execute(plan->inv_plan);
    for (int m = 0; m < len_out; m++)
        outR[m] = (m < fft_n) ? plan->rbuf[m] * inv : 0;
    if (wrap_m > 0) {
        for (int i = 0; i <= wrap_m; i++) {
            int pos = fft_n + i;
            if (pos >= conv_len) break;
            double high = 0;
            int j_max = len_g - pos;
            if (j_max > len_P) j_max = len_P;
            for (int j = 0; j < j_max; j++)
                high += PL[j] * g[pos + j];
            if (i < len_out) outR[i] -= high;
            if (pos < len_out) outR[pos] = high;
        }
    }
}

/* Cached correlate with m-wrap correction.
 * Same as correlate_fft_cached, but the FFT size may be smaller than
 * len_g + len_P - 1, with wrap_m correction terms.
 * The correction undoes cyclic aliasing by computing the true cross-correlation
 * of the tail terms via schoolbook. */
static void correlate_fft_cached_wrap(const double *g, int len_g,
                                       const double *P, int len_P,
                                       double *out, int len_out,
                                       FFTCache *fc,
                                       const fftw_complex *cached_fft_P,
                                       int cached_fft_n, int wrap_m) {
    if (wrap_m == 0) {
        correlate_fft_cached(g, len_g, len_P, out, len_out, fc, cached_fft_P, cached_fft_n);
        return;
    }

    FFTPlan *plan = fft_cache_get(fc, cached_fft_n);
    int fft_n = plan->fft_n;
    int cn = fft_n / 2 + 1;
    double inv = 1.0 / fft_n;
    int conv_len = len_g + len_P - 1;

    /* FFT(g) */
    int copy_g = (len_g < fft_n) ? len_g : fft_n;
    memcpy(plan->rbuf, g, copy_g * sizeof(double));
    if (copy_g < fft_n) memset(plan->rbuf + copy_g, 0, (fft_n - copy_g) * sizeof(double));
    fftw_execute(plan->fwd_plan);

    /* Pointwise: FFT(g) * conj(FFT(P)) */
    for (int i = 0; i < cn; i++) {
        double pr = cached_fft_P[i][0], pi = cached_fft_P[i][1];
        double gr = plan->cbuf[i][0], gi = plan->cbuf[i][1];
        plan->cbuf[i][0] = gr * pr + gi * pi;
        plan->cbuf[i][1] = gi * pr - gr * pi;
    }
    fftw_execute(plan->inv_plan);

    /* Extract cyclic cross-correlation result */
    for (int m = 0; m < len_out; m++)
        out[m] = (m < fft_n) ? plan->rbuf[m] * inv : 0;

    /* Correction: the cyclic cross-correlation aliases position i with i+fft_n.
     * Cross-correlation: (g ⋆ P)[m] = Σ_j P[j] * g[m+j].
     * True value at position pos = fft_n + i:
     *   corr_true[pos] = Σ_{j: 0≤j<len_P, 0≤pos+j<len_g} P[j] * g[pos + j] */
    for (int i = 0; i <= wrap_m; i++) {
        int pos = fft_n + i;
        if (pos >= conv_len) break;
        double high = 0;
        int j_max = len_g - pos;
        if (j_max > len_P) j_max = len_P;
        for (int j = 0; j < j_max; j++)
            high += P[j] * g[pos + j];
        if (i < len_out) out[i] -= high;      /* undo alias at wrapped position */
        if (pos < len_out) out[pos] = high;    /* place at true position */
    }
}

/* ══════════════════════════════════════════════════════════════
   TREE ENGINE (schoolbook + FFT hybrid, FFTW cached plans)
   ══════════════════════════════════════════════════════════════ */

#define MAX_TREE_LEVELS 20

typedef struct {
    /* Per-level metadata */
    int L, N;
    int psz[MAX_TREE_LEVELS], nn[MAX_TREE_LEVELS];
    int n_real[MAX_TREE_LEVELS];              /* number of REAL (non-padding) nodes at each level */
    size_t plev_off[MAX_TREE_LEVELS];
    size_t plev_total, max_g;
    /* Workspace: flat buffer for Plev data + 2 g-level buffers */
    double *ws;
    size_t ws_size;
    /* FFT cache */
    FFTCache *fft;
    /* Per-level FFT coefficient cache (for build→propagate reuse) */
    fftw_complex *fft_coeff[MAX_TREE_LEVELS]; /* flat array of cached FFTs */
    int fft_coeff_n[MAX_TREE_LEVELS];         /* FFT size (0 = not cached) */
    int fft_coeff_cn[MAX_TREE_LEVELS];        /* complex array length per poly */
    int fft_cache_ok[MAX_TREE_LEVELS];        /* 1 if build/correlate FFT sizes match */
    int below_sat[MAX_TREE_LEVELS];           /* 1 if polys at level ell-1 have
                                                 actual degree = cps/2 (use folded multiply) */
    int g_needed[MAX_TREE_LEVELS];            /* how many g coefficients level ell-1 actually
                                                 needs from propagation (truncated propagate) */
    /* Precomputed FFT configuration per level (from calibrated data).
     * Avoids 33ns lookup overhead per FFT call in the hot path. */
    int build_fft_n[MAX_TREE_LEVELS];        /* FFT size for build multiply */
    int build_wrap_m[MAX_TREE_LEVELS];       /* wrap coefficient count for cyclic build */
    int corr_fft_n[MAX_TREE_LEVELS];         /* FFT size for correlate */
    int corr_wrap_m[MAX_TREE_LEVELS];        /* wrap coefficient count for cyclic correlate */
    int use_fft[MAX_TREE_LEVELS];            /* 1 = use FFT, 0 = schoolbook (per-level decision) */
} TreeCtx;

/* Create tree context. leaf_degree=1 for standard tree, B for hybrid.
 * leaf_extract: how many g coefficients the leaf consumer needs
 *               (1 for pure tree leaf extraction, B for hybrid divide). */
static TreeCtx *tree_ctx_create_ex2(int n_leaves, int leaf_degree, int k,
                                     int leaf_extract) {
    /* Round k to the fastest smooth size >= k, using calibrated FFT data.
     * Power-of-2 k stays as-is; others search for the k' that minimizes
     * the saturated-level FFT cost (which dominates at large k). */
    k = best_k_pad(k);

    TreeCtx *tc = (TreeCtx *)calloc(1, sizeof(TreeCtx));
    tc->N = 1;
    while (tc->N < n_leaves) tc->N <<= 1;
    tc->L = 0;
    { int tmp = tc->N; while (tmp > 1) { tmp >>= 1; tc->L++; } tc->L++; }

    /* Compute n_real: number of real (non-padding) nodes at each level */
    tc->n_real[0] = n_leaves;
    for (int ell = 1; ell < tc->L; ell++)
        tc->n_real[ell] = (tc->n_real[ell-1] + 1) / 2;

    size_t off = 0;
    tc->max_g = 0;
    int max_psz = 0;
    for (int ell = 0; ell < tc->L; ell++) {
        tc->nn[ell] = tc->N >> ell;
        /* psz[ell] = min(leaf_degree * 2^(ell+1), k) */
        long d = (long)leaf_degree * (1L << (ell + 1));
        tc->psz[ell] = (d > k) ? k : (int)d;
        tc->plev_off[ell] = off;
        size_t level_sz = (size_t)tc->nn[ell] * tc->psz[ell];
        off += level_sz;
        if (level_sz > tc->max_g) tc->max_g = level_sz;
        if (tc->psz[ell] > max_psz) max_psz = tc->psz[ell];
    }
    tc->plev_total = off;
    tc->ws_size = tc->plev_total + 2 * tc->max_g;
    tc->ws = (double *)malloc(tc->ws_size * sizeof(double));

    /* Detect below-saturation levels where polys have actual degree = cps/2.
     * At these levels, the folded multiply halves the FFT size. */
    for (int ell = 0; ell < tc->L; ell++) {
        tc->below_sat[ell] = 0;
        if (ell == 0) continue;
        int cps = tc->psz[ell-1];
        /* Below saturation: psz doubles each level, so psz[ell] = 2*psz[ell-1].
         * This means the actual degree of children is cps/2, not cps-1. */
        if (tc->psz[ell] == 2 * cps && cps >= 2)
            tc->below_sat[ell] = 1;
    }

    /* Compute g_needed: how many g coefficients each level actually needs.
     * Propagates upward from the leaf extraction size. */
    tc->g_needed[0] = (leaf_extract < tc->psz[0]) ? leaf_extract : tc->psz[0];
    for (int ell = 1; ell < tc->L; ell++) {
        /* Level ell's correlate produces g for level ell-1.
         * Level ell-1 needs g_needed[ell-1] terms.
         * The correlate accesses g_parent[m+j] for m=0..g_needed[ell-1]-1
         * and j=0..psz[ell-1]-1. So g_parent needs g_needed[ell-1]+psz[ell-1]-1 terms. */
        int need = tc->g_needed[ell-1] + tc->psz[ell-1] - 1;
        tc->g_needed[ell] = (need < tc->psz[ell]) ? need : tc->psz[ell];
    }

    /* Precompute per-level FFT configuration using calibrated data.
     * Determines: use_fft (vs schoolbook), build FFT size + wrap_m,
     * correlate FFT size. All lookups happen here (one-time), so the
     * hot path reads from arrays with zero lookup overhead. */
    {
        int needed[4 * MAX_TREE_LEVELS];
        int n_needed = 0;

        for (int ell = 0; ell < tc->L; ell++) {
            tc->use_fft[ell] = 0;
            tc->build_fft_n[ell] = 0;
            tc->build_wrap_m[ell] = 0;
            tc->corr_fft_n[ell] = 0;
            if (ell == 0) continue;

            int cps = tc->psz[ell-1];
            int pgsz = tc->psz[ell];
            int is_below = tc->below_sat[ell];
            int is_root = (ell == tc->L - 1);

            /* Per-level FFT decision: compare schoolbook cost vs calibrated FFT cost.
             * Below-sat polys have actual degree cps/2 (upper half is zero), so
             * schoolbook cost is (d+1)² not cps². Using cps² overestimates by 4x. */
            int d_eff = is_below ? cps / 2 : cps - 1;
            long long school_cost_flops = (long long)(d_eff + 1) * (d_eff + 1);
            int build_conv_len = is_below ? (2 * (cps / 2)) : (2 * cps - 1);
            int bfn, bwm;
            best_fft_config(build_conv_len, &bfn, &bwm);
            /* Use FFT if calibrated FFT time + per-call overhead < schoolbook time.
             * Overhead (plan lookup, buffer copy, extraction) measured at ~40ns for
             * m=0 cases via `./bench_grid profile` microbenchmark. */
            int fft_idx;
            { int lo=0,hi=N_CALIBRATED_SIZES-1;
              while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<bfn)lo=m+1;else hi=m;}
              fft_idx = lo; }
            double fft_time = (fft_idx < N_CALIBRATED_SIZES && calib_sizes[fft_idx] == bfn)
                              ? calib_times_ns[fft_idx] : 1e18;
            double fft_overhead = FFT_OVERHEAD_NS;
            double correction_ns = (double)(bwm + 1) * (bwm + 1) * FMA_NS;
            double school_time = school_cost_flops * FMA_NS;
            tc->use_fft[ell] = (fft_time + fft_overhead + correction_ns < school_time);

            if (tc->use_fft[ell]) {
                tc->build_fft_n[ell] = bfn;
                tc->build_wrap_m[ell] = bwm;
                needed[n_needed++] = bfn;

                /* Correlate FFT size */
                int p_eff = is_below ? cps/2 + 1 : cps;
                int out_needed = tc->g_needed[ell-1];
                int g_eff_needed = out_needed + p_eff - 1;
                int g_eff_max = is_below ? (cps + cps/2) : pgsz;
                int g_eff = (g_eff_needed < g_eff_max) ? g_eff_needed : g_eff_max;
                int corr_conv = g_eff + p_eff - 1;

                if (!is_root) {
                    /* Compare joint (cached) vs independent (uncached).
                     * Joint: one FFT size, cache FFT(P) from build, reuse in correlates.
                     * Independent: build and correlate each pick optimal sizes, no caching. */
                    int jfn, jbm, jcm;
                    double jcost = best_fft_config_joint(build_conv_len, corr_conv,
                                                          &jfn, &jbm, &jcm);

                    int cfn, cwm;
                    best_fft_config(corr_conv, &cfn, &cwm);
                    /* Independent: build(bfn) + correlate_fft_pair(cfn).
                     * Pair shares g FFT: 1 fwd(g) + 2 fwd(P_rev) + 2 pw + 2 ifft
                     * ≈ 1.25× full pipeline. */
                    int cfn_idx = 0;
                    { int lo2=0, hi2=N_CALIBRATED_SIZES-1;
                      while(lo2<hi2){int m2=(lo2+hi2)>>1;if(calib_sizes[m2]<cfn)lo2=m2+1;else hi2=m2;}
                      cfn_idx = lo2; }
                    double icost = fft_time + correction_ns
                        + INDEP_PAIR_RATIO * calib_times_ns[cfn_idx]
                        + 2.0 * (double)(cwm+1)*(cwm+1)*FMA_NS;

                    if (jcost < icost) {
                        tc->build_fft_n[ell] = jfn;
                        tc->build_wrap_m[ell] = jbm;
                        tc->corr_fft_n[ell] = jfn;
                        tc->corr_wrap_m[ell] = jcm;
                        needed[n_needed - 1] = jfn;
                    } else {
                        tc->corr_fft_n[ell] = cfn;
                        tc->corr_wrap_m[ell] = cwm;
                        needed[n_needed++] = cfn;
                    }
                } else {
                    /* Root: no caching (build skipped), optimize correlate independently */
                    int cfn, cwm;
                    best_fft_config(corr_conv, &cfn, &cwm);
                    tc->corr_fft_n[ell] = cfn;
                    tc->corr_wrap_m[ell] = cwm;
                    needed[n_needed++] = cfn;
                }
            }
        }

        /* Create FFT plan cache from the needed sizes */
        if (n_needed > 0) {
            /* De-duplicate and sort */
            for (int i = 0; i < n_needed; i++)
                for (int j = i+1; j < n_needed; j++)
                    if (needed[j] < needed[i]) { int t=needed[i]; needed[i]=needed[j]; needed[j]=t; }
            int uniq[4 * MAX_TREE_LEVELS];
            int n_uniq = 0;
            for (int i = 0; i < n_needed; i++)
                if (n_uniq == 0 || uniq[n_uniq-1] != needed[i])
                    uniq[n_uniq++] = needed[i];
            tc->fft = fft_cache_create_sizes(uniq, n_uniq);
        } else {
            tc->fft = NULL;
        }
    }

    /* Allocate per-level FFT coefficient caches for build→propagate reuse.
     * Cache only at levels where joint optimization won (build_fft_n == corr_fft_n).
     * Skip ell=0 (leaves) and ell=L-1 (root, build skipped). */
    for (int ell = 0; ell < tc->L; ell++) {
        tc->fft_coeff[ell] = NULL;
        tc->fft_coeff_n[ell] = 0;
        tc->fft_cache_ok[ell] = 0;
        if (ell == 0 || ell == tc->L - 1 || !tc->use_fft[ell]) continue;
        if (tc->build_fft_n[ell] != tc->corr_fft_n[ell]) continue;  /* independent won */
        int cps = tc->psz[ell-1];
        int pgsz = tc->psz[ell];
        int corr_fft_n = tc->corr_fft_n[ell];
        int cn = corr_fft_n / 2 + 1;
        int n_children = tc->nn[ell-1];
        tc->fft_coeff[ell] = (fftw_complex *)fftw_malloc(
            (size_t)n_children * cn * sizeof(fftw_complex));
        tc->fft_coeff_n[ell] = corr_fft_n;
        tc->fft_coeff_cn[ell] = cn;
        tc->fft_cache_ok[ell] = 1;
    }

    return tc;
}

static TreeCtx *tree_ctx_create_ex(int n_leaves, int leaf_degree, int k) {
    /* Default leaf_extract: 1 for standard tree (only g[0] at leaves) */
    return tree_ctx_create_ex2(n_leaves, leaf_degree, k, 1);
}

static TreeCtx *tree_ctx_create(int n, int k) {
    return tree_ctx_create_ex(n, 1, k);
}

static void tree_ctx_destroy(TreeCtx *tc) {
    free(tc->ws);
    if (tc->fft) fft_cache_destroy(tc->fft);
    for (int ell = 0; ell < tc->L; ell++)
        if (tc->fft_coeff[ell]) fftw_free(tc->fft_coeff[ell]);
    free(tc);
}

/* Clone a TreeCtx: copies metadata, allocates fresh workspace + FFT cache.
 * The FFT plans inside the cloned FFTCache share the same FFTW plan objects
 * but have independent buffers, which is safe because fftw_execute_* with
 * explicit buffers is thread-safe. We create independent FFTCache + buffers. */
static FFTCache *fft_cache_clone(const FFTCache *src) {
    if (!src) return NULL;
    FFTCache *fc = (FFTCache *)calloc(1, sizeof(FFTCache));
    fc->max_fft_n = src->max_fft_n;
    fc->n_plans = src->n_plans;
    fc->plans_cap = src->n_plans;
    fc->plans = (FFTPlan *)calloc(src->n_plans, sizeof(FFTPlan));
    for (int i = 0; i < src->n_plans; i++) {
        FFTPlan *dp = &fc->plans[i];
        const FFTPlan *sp = &src->plans[i];
        dp->fft_n = sp->fft_n;
        dp->rbuf = fftw_malloc(dp->fft_n * sizeof(double));
        dp->cbuf = fftw_malloc((dp->fft_n/2 + 1) * sizeof(fftw_complex));
        memset(dp->rbuf, 0, dp->fft_n * sizeof(double));
        /* Create plans from wisdom (PATIENT plans satisfy MEASURE requests).
         * Fall back to ESTIMATE only if wisdom is missing for this size. */
        dp->fwd_plan = fftw_plan_dft_r2c_1d(dp->fft_n, dp->rbuf, dp->cbuf, FFTW_MEASURE | FFTW_WISDOM_ONLY);
        dp->inv_plan = fftw_plan_dft_c2r_1d(dp->fft_n, dp->cbuf, dp->rbuf, FFTW_MEASURE | FFTW_WISDOM_ONLY);
        if (!dp->fwd_plan || !dp->inv_plan) {
            if (dp->fwd_plan) fftw_destroy_plan(dp->fwd_plan);
            if (dp->inv_plan) fftw_destroy_plan(dp->inv_plan);
            dp->fwd_plan = fftw_plan_dft_r2c_1d(dp->fft_n, dp->rbuf, dp->cbuf, FFTW_ESTIMATE);
            dp->inv_plan = fftw_plan_dft_c2r_1d(dp->fft_n, dp->cbuf, dp->rbuf, FFTW_ESTIMATE);
        }
    }
    fc->rbuf2 = fftw_malloc(fc->max_fft_n * sizeof(double));
    fc->cbuf2 = fftw_malloc((fc->max_fft_n/2 + 1) * sizeof(fftw_complex));
    fc->cbuf3 = fftw_malloc((fc->max_fft_n/2 + 1) * sizeof(fftw_complex));
    return fc;
}

static TreeCtx *tree_ctx_clone(const TreeCtx *src) {
    TreeCtx *tc = (TreeCtx *)calloc(1, sizeof(TreeCtx));
    memcpy(tc, src, sizeof(TreeCtx));
    /* Allocate fresh workspace */
    tc->ws = (double *)malloc(tc->ws_size * sizeof(double));
    /* Clone FFT cache (independent buffers + plans) */
    tc->fft = fft_cache_clone(src->fft);
    /* Clone FFT coefficient caches */
    for (int ell = 0; ell < tc->L; ell++) {
        if (src->fft_coeff[ell]) {
            size_t sz = (size_t)src->nn[ell-1] * src->fft_coeff_cn[ell] * sizeof(fftw_complex);
            tc->fft_coeff[ell] = (fftw_complex *)fftw_malloc(sz);
            /* Contents will be filled during tree build, no need to copy */
        } else {
            tc->fft_coeff[ell] = NULL;
        }
    }
    return tc;
}

/* ── Shared tree build + propagate (used by both tree and hybrid engines) ── */

/* Build tree levels 1..L-2 (skip root — never read). Operates on tc->ws. */
static void tree_build_levels(TreeCtx *tc) {
    int L = tc->L;
    int *psz = tc->psz;
    size_t *plev_off = tc->plev_off;
    double *plev_data = tc->ws;

    for (int ell = 1; ell < L - 1; ell++) {
        int cps = psz[ell-1], pps = psz[ell];
        double *child_base = plev_data + plev_off[ell-1];
        double *parent_base = plev_data + plev_off[ell];
        int use_fft = tc->use_fft[ell];
        int cache_fft = tc->fft_cache_ok[ell];

        int cn = cache_fft ? tc->fft_coeff_cn[ell] : 0;
        int nr_parent = tc->n_real[ell];
        int nr_child = tc->n_real[ell-1];
        for (int j = 0; j < nr_parent; j++) {
            double *Lc = child_base + (size_t)(2*j) * cps;
            double *out = parent_base + (size_t)j * pps;
            if (2*j+1 >= nr_child) {
                int cp = (cps < pps) ? cps : pps;
                memcpy(out, Lc, cp * sizeof(double));
                if (cp < pps) memset(out + cp, 0, (pps - cp) * sizeof(double));
                if (cache_fft) {
                    fftw_complex *fft_L = tc->fft_coeff[ell] + (size_t)(2*j) * cn;
                    int tfn = tc->fft_coeff_n[ell];
                    FFTPlan *plan = fft_cache_get(tc->fft, tfn);
                    int fft_n = plan->fft_n;
                    memcpy(plan->rbuf, Lc, cps * sizeof(double));
                    if (cps < fft_n) memset(plan->rbuf + cps, 0, (fft_n - cps) * sizeof(double));
                    fftw_execute(plan->fwd_plan);
                    memcpy(fft_L, plan->cbuf, (fft_n/2+1) * sizeof(fftw_complex));
                }
            } else {
                double *Rc = child_base + (size_t)(2*j+1) * cps;
                if (use_fft) {
                    fftw_complex *fft_L = cache_fft ?
                        tc->fft_coeff[ell] + (size_t)(2*j) * cn : NULL;
                    fftw_complex *fft_R = cache_fft ?
                        tc->fft_coeff[ell] + (size_t)(2*j+1) * cn : NULL;
                    polymul_fft_wrap(Lc, cps, Rc, cps, out, pps,
                                     tc->fft, fft_L, fft_R,
                                     tc->build_fft_n[ell], tc->build_wrap_m[ell]);
                } else {
                    polymul_modk(Lc, cps, Rc, cps, out, pps);
                }
            }
        }
    }
}

/* Propagate g-vectors top-down through the tree.
 * k: number of payout terms (original, not padded).
 * Returns pointer to leaf-level g vectors. */
static double *tree_propagate_g(TreeCtx *tc, int k, const double *payout) {
    int L = tc->L;
    int *psz = tc->psz, *nn = tc->nn;
    size_t *plev_off = tc->plev_off;
    double *plev_data = tc->ws;
    double *g_buf0 = tc->ws + tc->plev_total;
    double *g_buf1 = tc->ws + tc->plev_total + tc->max_g;

    int top = L - 1;
    int root_gsz = psz[top];
    double *g_parent = g_buf0;
    memset(g_parent, 0, (size_t)nn[top] * root_gsz * sizeof(double));
    int copy = (k < root_gsz) ? k : root_gsz;
    memcpy(g_parent, payout, copy * sizeof(double));

    for (int ell = top; ell >= 1; ell--) {
        int pgsz = psz[ell], cgsz = psz[ell-1], cps = psz[ell-1];
        double *g_child = (g_parent == g_buf0) ? g_buf1 : g_buf0;
        double *child_base = plev_data + plev_off[ell-1];
        int use_fft = tc->use_fft[ell];
        int cache_fft = tc->fft_cache_ok[ell];
        int is_below = tc->below_sat[ell];
        int p_eff = is_below ? cps/2 + 1 : cps;
        int cn = cache_fft ? tc->fft_coeff_cn[ell] : 0;
        int cached_fft_n = cache_fft ? tc->fft_coeff_n[ell] : 0;

        int out_needed = tc->g_needed[ell-1];
        int g_eff_needed = out_needed + p_eff - 1;
        int g_eff_max = is_below ? (cgsz + cps/2) : pgsz;
        int g_eff = (g_eff_needed < g_eff_max) ? g_eff_needed : g_eff_max;

        int nr_parent = tc->n_real[ell];
        int nr_child = tc->n_real[ell-1];
        for (int j = 0; j < nr_parent; j++) {
            double *gp = g_parent + (size_t)j * pgsz;
            double *gL = g_child + (size_t)(2*j) * cgsz;
            if (2*j+1 >= nr_child) {
                int cp = (out_needed < pgsz) ? out_needed : pgsz;
                memcpy(gL, gp, cp * sizeof(double));
                if (cp < out_needed) memset(gL + cp, 0, (out_needed - cp) * sizeof(double));
            } else {
                double *PL = child_base + (size_t)(2*j) * cps;
                double *PR = child_base + (size_t)(2*j+1) * cps;
                double *gR = g_child + (size_t)(2*j+1) * cgsz;
                if (use_fft && cache_fft) {
                    const fftw_complex *fft_R = tc->fft_coeff[ell] + (size_t)(2*j+1)*cn;
                    const fftw_complex *fft_L = tc->fft_coeff[ell] + (size_t)(2*j)*cn;
                    int cwm = tc->corr_wrap_m[ell];
                    correlate_fft_cached_pair_wrap(gp, g_eff, PL, PR, p_eff,
                                                   gL, gR, out_needed,
                                                   tc->fft, fft_L, fft_R,
                                                   cached_fft_n, cwm);
                } else if (use_fft) {
                    correlate_fft_pair(gp, g_eff, PL, PR, p_eff,
                                       gL, gR, out_needed, tc->fft,
                                       tc->corr_fft_n[ell]);
                } else {
                    correlate_school(gp, g_eff, PR, p_eff, gL, out_needed);
                    correlate_school(gp, g_eff, PL, p_eff, gR, out_needed);
                }
            }
        }
        g_parent = g_child;
    }
    return g_parent;
}

static void engine_tree_ctx(int n, const double *a,
                            const double *payout, int k,
                            double *inner, void *ctx) {
    TreeCtx *tc = (TreeCtx *)ctx;
    int N = tc->N;
    int *psz = tc->psz;
    double *plev_data = tc->ws;

    memset(plev_data + tc->plev_off[0], 0, (size_t)N * psz[0] * sizeof(double));

    for (int j = 0; j < N; j++) {
        double *P = plev_data + tc->plev_off[0] + (size_t)j * psz[0];
        if (j < n) {
            P[0] = a[j];
            if (psz[0] > 1) P[1] = 1.0 - a[j];
        } else {
            P[0] = 1.0;
        }
    }

    tree_build_levels(tc);
    double *g_leaf = tree_propagate_g(tc, k, payout);

    int lgsz = psz[0];
    for (int j = 0; j < n; j++)
        inner[j] = g_leaf[j * lgsz];
}

/* ══════════════════════════════════════════════════════════════
   NAIVE (full build + bidir divide)
   ══════════════════════════════════════════════════════════════ */

typedef struct { double *ws; } NaiveCtx;

static NaiveCtx *naive_ctx_create(int n, int k) {
    (void)k;
    NaiveCtx *nc = (NaiveCtx *)calloc(1, sizeof(NaiveCtx));
    nc->ws = (double *)malloc(((size_t)(n + 1) + n) * sizeof(double));
    return nc;
}
static void naive_ctx_destroy(NaiveCtx *nc) { free(nc->ws); free(nc); }

static NaiveCtx *naive_ctx_clone(const NaiveCtx *src, int n, int k) {
    (void)k;
    NaiveCtx *nc = (NaiveCtx *)calloc(1, sizeof(NaiveCtx));
    nc->ws = (double *)malloc(((size_t)(n + 1) + n) * sizeof(double));
    return nc;
}

static void engine_naive_ctx(int n, const double *a,
                             const double *payout, int k,
                             double *inner, void *ctx) {
    NaiveCtx *nc = (NaiveCtx *)ctx;
    int pk = (k < n) ? k : n;
    double *P = nc->ws;
    double *Q = nc->ws + (n + 1);

    memset(P, 0, (n + 1) * sizeof(double));
    P[0] = 1;
    for (int j = 0; j < n; j++) {
        double aj = a[j], bj = 1 - aj;
        for (int m = (j+1 < n ? j+1 : n); m >= 1; m--)
            P[m] = aj * P[m] + bj * P[m - 1];
        P[0] *= aj;
    }

    for (int i = 0; i < n; i++) {
        double ai = a[i], bi = 1 - ai;
        if (ai > 0.5) {
            double ia=1/ai, c=-bi*ia;
            Q[0]=P[0]*ia;
            for (int m=1;m<n;m++) Q[m]=c*Q[m-1]+P[m]*ia;
        } else if (ai > 1e-15) {
            double ib=1/bi, c=-ai*ib;
            Q[n-1]=P[n]*ib;
            for (int m=n-2;m>=0;m--) Q[m]=c*Q[m+1]+P[m+1]*ib;
        } else {
            for (int m=0;m<n;m++) Q[m]=P[m+1];
        }
        double eq = 0;
        for (int m = 0; m < pk; m++) eq += payout[m] * Q[m];
        inner[i] = eq;
    }
}

/* ══════════════════════════════════════════════════════════════
   LINEAR FORWARD-BACKWARD
   ══════════════════════════════════════════════════════════════ */

typedef struct {
    double *ws; size_t ws_size;
    uint8_t *active;  /* per-player mask: active[i]=1 → compute inner[i]. NULL = all. */
} LinearCtx;

/* Threshold: if g_store > 32MB, use checkpointed linear */
#define CKPT_THRESHOLD 4194304  /* 32MB / 8 bytes */

static int ckpt_interval(int n) {
    int C = (int)sqrt((double)n);
    if (C < 2) C = 2;
    return C;
}

static LinearCtx *linear_ctx_create(int n, int k) {
    LinearCtx *lc = (LinearCtx *)calloc(1, sizeof(LinearCtx));
    if ((size_t)n * k > CKPT_THRESHOLD) {
        int C = ckpt_interval(n);
        int n_ckpt = n / C;
        lc->ws_size = (size_t)n_ckpt * k + (size_t)C * k + 3 * (size_t)k;
    } else {
        lc->ws_size = (size_t)n * k + (size_t)k;
    }
    lc->ws = (double *)malloc(lc->ws_size * sizeof(double));
    return lc;
}
static void linear_ctx_destroy(LinearCtx *lc) { free(lc->ws); free(lc); }

static LinearCtx *linear_ctx_clone(const LinearCtx *src) {
    LinearCtx *lc = (LinearCtx *)calloc(1, sizeof(LinearCtx));
    lc->ws_size = src->ws_size;
    lc->ws = (double *)malloc(lc->ws_size * sizeof(double));
    return lc;
}

static inline void apply_factor(const double *restrict g_in,
                                double *restrict g_out,
                                double a_val, double b_val, int k) {
    for (int m = 0; m < k - 1; m++)
        g_out[m] = a_val * g_in[m] + b_val * g_in[m + 1];
    g_out[k - 1] = a_val * g_in[k - 1];
}

static void engine_linear_ctx(int n, const double *a,
                               const double *payout, int k,
                               double *inner, void *ctx) {
    LinearCtx *lc = (LinearCtx *)ctx;
    double *ws = lc->ws;

    if ((size_t)n * k > CKPT_THRESHOLD) {
        /* Checkpointed variant */
        int C = ckpt_interval(n);
        int n_ckpt = n / C;
        double *ckpt    = ws;
        double *local_g = ckpt + (size_t)n_ckpt * k;
        double *R       = local_g + (size_t)C * k;
        double *buf0    = R + k;
        double *buf1    = buf0 + k;

        const double *g_prev = payout;
        for (int j = 0; j < n; j++) {
            double *g_cur = (j & 1) ? buf1 : buf0;
            apply_factor(g_prev, g_cur, a[j], 1.0 - a[j], k);
            if ((j + 1) % C == 0)
                memcpy(ckpt + (size_t)((j+1)/C - 1) * k, g_cur, k * sizeof(double));
            g_prev = g_cur;
        }

        memset(R, 0, k * sizeof(double));
        R[0] = 1.0;
        int n_seg = (n + C - 1) / C;
        for (int s = n_seg - 1; s >= 0; s--) {
            int ss = s * C, se = ss + C;
            if (se > n) se = n;
            const double *prefix = (s == 0) ? payout : (ckpt + (size_t)(s-1)*k);
            const double *lp = prefix;
            for (int j = 0; j < se - ss; j++) {
                double *g_cur = local_g + (size_t)j * k;
                apply_factor(lp, g_cur, a[ss+j], 1.0 - a[ss+j], k);
                lp = g_cur;
            }
            const uint8_t *active = lc->active;
            for (int j = se - ss - 1; j >= 0; j--) {
                double aj = a[ss+j], bj = 1 - aj;
                if (!active || active[ss+j]) {
                    const double *gb = (j == 0) ? prefix : (local_g + (size_t)(j-1)*k);
                    double eq = gb[0] * R[0];
                    for (int m = k-1; m >= 1; m--) {
                        eq += gb[m] * R[m];
                        R[m] = aj * R[m] + bj * R[m-1];
                    }
                    R[0] = aj * R[0];
                    inner[ss+j] = eq;
                } else {
                    for (int m = k-1; m >= 1; m--)
                        R[m] = aj * R[m] + bj * R[m-1];
                    R[0] = aj * R[0];
                }
            }
        }
    } else {
        /* Flat variant */
        double *g_store = ws;
        double *R = ws + (size_t)n * k;
        const double *g_prev = payout;

        for (int j = 0; j < n; j++) {
            double aj = a[j], bj = 1 - aj;
            double *g_cur = g_store + (size_t)j * k;
            for (int m = 0; m < k - 1; m++)
                g_cur[m] = aj * g_prev[m] + bj * g_prev[m + 1];
            g_cur[k-1] = aj * g_prev[k-1];
            g_prev = g_cur;
        }

        memset(R, 0, k * sizeof(double));
        R[0] = 1.0;
        const uint8_t *active = lc->active;
        for (int j = n - 1; j >= 0; j--) {
            double aj = a[j], bj = 1 - aj;
            if (!active || active[j]) {
                /* Fused dot product + suffix update */
                const double *gb = (j > 0) ? (g_store + (size_t)(j-1)*k) : payout;
                double eq = gb[0] * R[0];
                for (int m = k - 1; m >= 1; m--) {
                    eq += gb[m] * R[m];
                    R[m] = aj * R[m] + bj * R[m-1];
                }
                R[0] = aj * R[0];
                inner[j] = eq;
            } else {
                /* Suffix update only — skip dot product */
                for (int m = k - 1; m >= 1; m--)
                    R[m] = aj * R[m] + bj * R[m-1];
                R[0] = aj * R[0];
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════
   BATCHED LINEAR ENGINE — process 2 quad points per pass
   Vectorizes across quad points (NEON width 2) instead of across k.
   ══════════════════════════════════════════════════════════════ */

/* Interleaved layout: g[j * k * 2 + m * 2 + q] for quad point q in {0,1} */
#define BQ 2

static double run_linear_batched(int n, const double *S, int Q,
                                  const double *payout, int k,
                                  double *equity, void *ctx) {
    LinearCtx *lc_ctx = (LinearCtx *)ctx;
    const uint8_t *active = lc_ctx ? lc_ctx->active : NULL;
    double Smax = 0;
    for (int i = 0; i < n; i++) if (S[i] > Smax) Smax = S[i];
    QP *pts = (QP *)malloc(Q * sizeof(QP));
    make_nodes(Q, Smax, pts);
    memset(equity, 0, n * sizeof(double));

    /* Interleaved payout: {payout[0], payout[0], payout[1], payout[1], ...} */
    double *payout_il = (double *)malloc(k * BQ * sizeof(double));
    for (int m = 0; m < k; m++) {
        payout_il[m * 2] = payout[m];
        payout_il[m * 2 + 1] = payout[m];
    }

    int n_pairs = Q / 2;
    int has_leftover = (Q % 2 != 0);

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads > n_pairs) nthreads = (n_pairs > 0) ? n_pairs : 1;

    /* Per-thread workspace */
    size_t gstride = (size_t)k * BQ;
    double **t_gstore = (double **)malloc(nthreads * sizeof(double *));
    double **t_R      = (double **)malloc(nthreads * sizeof(double *));
    double **t_a0     = (double **)malloc(nthreads * sizeof(double *));
    double **t_a1     = (double **)malloc(nthreads * sizeof(double *));
    double **t_inner0 = (double **)malloc(nthreads * sizeof(double *));
    double **t_inner1 = (double **)malloc(nthreads * sizeof(double *));
    double **t_equity = (double **)malloc(nthreads * sizeof(double *));
    for (int t = 0; t < nthreads; t++) {
        t_gstore[t] = (double *)malloc((size_t)n * gstride * sizeof(double));
        t_R[t]      = (double *)malloc(k * BQ * sizeof(double));
        t_a0[t]     = (double *)malloc(n * sizeof(double));
        t_a1[t]     = (double *)malloc(n * sizeof(double));
        t_inner0[t] = (double *)malloc(n * sizeof(double));
        t_inner1[t] = (double *)malloc(n * sizeof(double));
        t_equity[t] = (double *)calloc(n, sizeof(double));
    }

    double t0 = now_ns();
    #pragma omp parallel for schedule(dynamic, 4) num_threads(nthreads)
    for (int p = 0; p < n_pairs; p++) {
        int q = p * 2;
        if (pts[q].w == 0 && pts[q+1].w == 0) continue;
        int tid = omp_get_thread_num();
        double *g_store = t_gstore[tid];
        double *R = t_R[tid];
        double *a0 = t_a0[tid];
        double *a1 = t_a1[tid];
        double *inner0 = t_inner0[tid];
        double *inner1 = t_inner1[tid];
        double *eq_local = t_equity[tid];

        double logv0 = pts[q].logv, logv1 = pts[q+1].logv;
#ifdef __APPLE__
        for (int j = 0; j < n; j++) { a0[j] = S[j]*logv0; a1[j] = S[j]*logv1; }
        int vn = n;
        vvexp(a0, a0, &vn);
        vvexp(a1, a1, &vn);
        for (int j = 0; j < n; j++) {
            if (S[j]*logv0 < -700) a0[j] = 0;
            if (S[j]*logv1 < -700) a1[j] = 0;
        }
#else
        for (int j = 0; j < n; j++) {
            double arg0 = S[j]*logv0, arg1 = S[j]*logv1;
            a0[j] = (arg0 < -700) ? 0 : exp(arg0);
            a1[j] = (arg1 < -700) ? 0 : exp(arg1);
        }
#endif

        const double *g_prev = payout_il;
        for (int j = 0; j < n; j++) {
            double aj0 = a0[j], bj0 = 1 - aj0;
            double aj1 = a1[j], bj1 = 1 - aj1;
            double *g_cur = g_store + (size_t)j * gstride;
            for (int m = 0; m < k - 1; m++) {
                int idx = m * 2;
                g_cur[idx]   = aj0 * g_prev[idx]   + bj0 * g_prev[idx+2];
                g_cur[idx+1] = aj1 * g_prev[idx+1] + bj1 * g_prev[idx+3];
            }
            int last = (k-1) * 2;
            g_cur[last]   = aj0 * g_prev[last];
            g_cur[last+1] = aj1 * g_prev[last+1];
            g_prev = g_cur;
        }

        memset(R, 0, k * BQ * sizeof(double));
        R[0] = 1.0; R[1] = 1.0;
        for (int j = n - 1; j >= 0; j--) {
            const double *gb = (j > 0) ?
                (g_store + (size_t)(j-1) * gstride) : payout_il;
            /* Fused dot product + suffix update */
            double aj0 = a0[j], bj0 = 1 - aj0;
            double aj1 = a1[j], bj1 = 1 - aj1;
            double eq0 = gb[0] * R[0];
            double eq1 = gb[1] * R[1];
            for (int m = k - 1; m >= 1; m--) {
                eq0 += gb[m*2]   * R[m*2];
                eq1 += gb[m*2+1] * R[m*2+1];
                R[m*2]   = aj0 * R[m*2]   + bj0 * R[(m-1)*2];
                R[m*2+1] = aj1 * R[m*2+1] + bj1 * R[(m-1)*2+1];
            }
            R[0] = aj0 * R[0];
            R[1] = aj1 * R[1];
            inner0[j] = eq0;
            inner1[j] = eq1;
        }

        double wq0 = pts[q].w, wq1 = pts[q+1].w;
        double iv0 = exp(-logv0), iv1 = exp(-logv1);
        for (int i = 0; i < n; i++) {
            if (active && !active[i]) continue;
            double pw0 = wq0 * S[i] * a0[i] * iv0;
            double pw1 = wq1 * S[i] * a1[i] * iv1;
            if (!isfinite(pw0)) pw0 = 0;
            if (!isfinite(pw1)) pw1 = 0;
            eq_local[i] += pw0 * inner0[i] + pw1 * inner1[i];
        }
    }

    /* Merge thread-local equity arrays */
    for (int t = 0; t < nthreads; t++) {
        for (int i = 0; i < n; i++)
            equity[i] += t_equity[t][i];
    }

    /* Handle leftover odd quad point (serial) */
    if (has_leftover) {
        int q = Q - 1;
        if (pts[q].w != 0) {
            double logv = pts[q].logv, wq = pts[q].w;
            double *a_tmp = t_a0[0];
            double *inner_tmp = t_inner0[0];
            for (int j = 0; j < n; j++) {
                double arg = S[j]*logv;
                a_tmp[j] = (arg < -700) ? 0 : exp(arg);
            }
            /* Create a temporary linear ctx for the leftover point */
            LinearCtx *lc_tmp = linear_ctx_create(n, k);
            lc_tmp->active = (uint8_t *)active;
            engine_linear_ctx(n, a_tmp, payout, k, inner_tmp, lc_tmp);
            lc_tmp->active = NULL;
            linear_ctx_destroy(lc_tmp);
            double iv = exp(-logv);
            for (int i = 0; i < n; i++) {
                if (active && !active[i]) continue;
                double pw = wq * S[i] * a_tmp[i] * iv;
                if (!isfinite(pw)) pw = 0;
                equity[i] += pw * inner_tmp[i];
            }
        }
    }

    double elapsed = now_ns() - t0;
    for (int t = 0; t < nthreads; t++) {
        free(t_gstore[t]); free(t_R[t]);
        free(t_a0[t]); free(t_a1[t]);
        free(t_inner0[t]); free(t_inner1[t]);
        free(t_equity[t]);
    }
    free(t_gstore); free(t_R); free(t_a0); free(t_a1);
    free(t_inner0); free(t_inner1); free(t_equity);

#else  /* No OpenMP — serial fallback */
    size_t gstride = (size_t)k * BQ;
    double *g_store = (double *)malloc((size_t)n * gstride * sizeof(double));
    double *R = (double *)malloc(k * BQ * sizeof(double));
    double *a0 = (double *)malloc(n * sizeof(double));
    double *a1 = (double *)malloc(n * sizeof(double));
    double *inner0 = (double *)malloc(n * sizeof(double));
    double *inner1 = (double *)malloc(n * sizeof(double));

    double t0 = now_ns();
    int q = 0;
    for (; q + 1 < Q; q += 2) {
        if (pts[q].w == 0 && pts[q+1].w == 0) continue;
        double logv0 = pts[q].logv, logv1 = pts[q+1].logv;
#ifdef __APPLE__
        for (int j = 0; j < n; j++) { a0[j] = S[j]*logv0; a1[j] = S[j]*logv1; }
        int vn = n;
        vvexp(a0, a0, &vn);
        vvexp(a1, a1, &vn);
        for (int j = 0; j < n; j++) {
            if (S[j]*logv0 < -700) a0[j] = 0;
            if (S[j]*logv1 < -700) a1[j] = 0;
        }
#else
        for (int j = 0; j < n; j++) {
            double arg0 = S[j]*logv0, arg1 = S[j]*logv1;
            a0[j] = (arg0 < -700) ? 0 : exp(arg0);
            a1[j] = (arg1 < -700) ? 0 : exp(arg1);
        }
#endif
        const double *g_prev = payout_il;
        for (int j = 0; j < n; j++) {
            double aj0 = a0[j], bj0 = 1 - aj0;
            double aj1 = a1[j], bj1 = 1 - aj1;
            double *g_cur = g_store + (size_t)j * gstride;
            for (int m = 0; m < k - 1; m++) {
                int idx = m * 2;
                g_cur[idx]   = aj0 * g_prev[idx]   + bj0 * g_prev[idx+2];
                g_cur[idx+1] = aj1 * g_prev[idx+1] + bj1 * g_prev[idx+3];
            }
            int last = (k-1) * 2;
            g_cur[last]   = aj0 * g_prev[last];
            g_cur[last+1] = aj1 * g_prev[last+1];
            g_prev = g_cur;
        }
        memset(R, 0, k * BQ * sizeof(double));
        R[0] = 1.0; R[1] = 1.0;
        for (int j = n - 1; j >= 0; j--) {
            const double *gb = (j > 0) ?
                (g_store + (size_t)(j-1) * gstride) : payout_il;
            /* Fused dot product + suffix update */
            double aj0 = a0[j], bj0 = 1 - aj0;
            double aj1 = a1[j], bj1 = 1 - aj1;
            double eq0 = gb[0] * R[0];
            double eq1 = gb[1] * R[1];
            for (int m = k - 1; m >= 1; m--) {
                eq0 += gb[m*2]   * R[m*2];
                eq1 += gb[m*2+1] * R[m*2+1];
                R[m*2]   = aj0 * R[m*2]   + bj0 * R[(m-1)*2];
                R[m*2+1] = aj1 * R[m*2+1] + bj1 * R[(m-1)*2+1];
            }
            R[0] = aj0 * R[0];
            R[1] = aj1 * R[1];
            inner0[j] = eq0;
            inner1[j] = eq1;
        }
        double wq0 = pts[q].w, wq1 = pts[q+1].w;
        double iv0 = exp(-logv0), iv1 = exp(-logv1);
        for (int i = 0; i < n; i++) {
            if (active && !active[i]) continue;
            double pw0 = wq0 * S[i] * a0[i] * iv0;
            double pw1 = wq1 * S[i] * a1[i] * iv1;
            if (!isfinite(pw0)) pw0 = 0;
            if (!isfinite(pw1)) pw1 = 0;
            equity[i] += pw0 * inner0[i] + pw1 * inner1[i];
        }
    }
    if (q < Q && pts[q].w != 0) {
        double logv = pts[q].logv, wq = pts[q].w;
        for (int j = 0; j < n; j++) {
            double arg = S[j]*logv;
            a0[j] = (arg < -700) ? 0 : exp(arg);
        }
        LinearCtx *lc_tmp = linear_ctx_create(n, k);
        lc_tmp->active = (uint8_t *)active;
        engine_linear_ctx(n, a0, payout, k, inner0, lc_tmp);
        lc_tmp->active = NULL;
        linear_ctx_destroy(lc_tmp);
        double iv = exp(-logv);
        for (int i = 0; i < n; i++) {
            if (active && !active[i]) continue;
            double pw = wq * S[i] * a0[i] * iv;
            if (!isfinite(pw)) pw = 0;
            equity[i] += pw * inner0[i];
        }
    }
    double elapsed = now_ns() - t0;
    free(g_store); free(R);
    free(a0); free(a1); free(inner0); free(inner1);
#endif /* _OPENMP */

    free(pts); free(payout_il);
    return elapsed;
}

/* ══════════════════════════════════════════════════════════════
   HYBRID ENGINE (block build + tree + bidirectional divide)
   ══════════════════════════════════════════════════════════════ */

typedef struct {
    int B, nblocks;
    double *block_prods;  /* nblocks_padded * (B+1) doubles */
    TreeCtx *tc;          /* inter-block tree */
    int *sort_perm;       /* sort_perm[i] = original index of sorted player i */
    double *S_sorted;     /* stack sizes in sorted order */
    uint8_t *active;      /* per-player mask (sorted order). NULL = all. */
} HybridCtx;

/* Comparator for qsort: sort player indices by stack size descending. */
static const double *qsort_S_ptr;
static int cmp_stack_desc(const void *a, const void *b) {
    double sa = qsort_S_ptr[*(const int *)a];
    double sb = qsort_S_ptr[*(const int *)b];
    return (sa > sb) ? -1 : (sa < sb) ? 1 : 0;
}

static HybridCtx *hybrid_ctx_create(int n, const double *S, int k, int B) {
    HybridCtx *hc = (HybridCtx *)calloc(1, sizeof(HybridCtx));
    hc->B = B;
    hc->nblocks = (n + B - 1) / B;
    int N_tree = 1;
    while (N_tree < hc->nblocks) N_tree <<= 1;

    /* Sort players by stack size (descending).
     * This groups backward-divide players (large S → small a) together
     * and forward-divide players (small S → large a) together.
     * Within each block, all players tend to use the same divide direction. */
    hc->sort_perm = (int *)malloc(n * sizeof(int));
    hc->S_sorted = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) hc->sort_perm[i] = i;
    qsort_S_ptr = S;
    qsort(hc->sort_perm, n, sizeof(int), cmp_stack_desc);
    for (int i = 0; i < n; i++) hc->S_sorted[i] = S[hc->sort_perm[i]];

    hc->block_prods = (double *)calloc((size_t)N_tree * (B + 1), sizeof(double));
    hc->tc = tree_ctx_create_ex2(hc->nblocks, B, k, B);

    return hc;
}

static void hybrid_ctx_destroy(HybridCtx *hc) {
    free(hc->block_prods);
    free(hc->sort_perm);
    free(hc->S_sorted);
    tree_ctx_destroy(hc->tc);
    free(hc);
}

static HybridCtx *hybrid_ctx_clone(const HybridCtx *src, int n) {
    HybridCtx *hc = (HybridCtx *)calloc(1, sizeof(HybridCtx));
    hc->B = src->B;
    hc->nblocks = src->nblocks;
    int N_tree = src->tc->N;
    hc->block_prods = (double *)calloc((size_t)N_tree * (src->B + 1), sizeof(double));
    /* Clone the inner tree context */
    hc->tc = tree_ctx_clone(src->tc);
    /* Share sort_perm and S_sorted (read-only during engine execution) */
    hc->sort_perm = (int *)malloc(n * sizeof(int));
    memcpy(hc->sort_perm, src->sort_perm, n * sizeof(int));
    hc->S_sorted = (double *)malloc(n * sizeof(double));
    memcpy(hc->S_sorted, src->S_sorted, n * sizeof(double));
    return hc;
}

/* Hybrid engine: block build + tree + bidirectional divide */
static void engine_hybrid_core(int n, const double *a,
                                const double *payout, int k,
                                double *inner, HybridCtx *hc) {
    int B = hc->B;
    int nblocks = hc->nblocks;
    TreeCtx *tc = hc->tc;
    int N = tc->N;
    int *psz = tc->psz;
    double *plev_data = tc->ws;

    /* ── Steps 1+2 fused: build block products directly into tree leaves,
     *    AND into block_prods (needed for the divide step). ── */
    int leaf_psz = psz[0];
    for (int b = 0; b < nblocks; b++) {
        int start = b * B, end = start + B;
        if (end > n) end = n;
        int bsize = end - start;
        /* Build into block_prods (full degree B, used by divide) */
        double *P = hc->block_prods + (size_t)b * (B + 1);
        memset(P, 0, (B + 1) * sizeof(double));
        P[0] = 1.0;
        for (int j = start; j < end; j++) {
            double aj = a[j], bj = 1 - aj;
            for (int m = bsize; m >= 1; m--)
                P[m] = aj * P[m] + bj * P[m - 1];
            P[0] *= aj;
        }
        /* Copy truncated version into tree leaf (may be shorter if k < B+1) */
        double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz;
        int cp = (B + 1 < leaf_psz) ? B + 1 : leaf_psz;
        memcpy(leaf, P, cp * sizeof(double));
        if (cp < leaf_psz) memset(leaf + cp, 0, (leaf_psz - cp) * sizeof(double));
    }
    /* Padding leaves = 1 */
    for (int b = nblocks; b < N; b++) {
        double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz;
        memset(leaf, 0, leaf_psz * sizeof(double));
        leaf[0] = 1.0;
    }

    tree_build_levels(tc);
    double *g_leaf = tree_propagate_g(tc, k, payout);
    int g_need = tc->g_needed[0];

    /* ── Step 3: Within-block divide + fused dot product ── */
    const uint8_t *active = hc->active;
    for (int b = 0; b < nblocks; b++) {
        int start = b * B, end = start + B;
        if (end > n) end = n;
        int bsize = end - start;

        /* Skip entire block if no active players in it */
        if (active) {
            int any = 0;
            for (int j = start; j < end; j++) if (active[j]) { any = 1; break; }
            if (!any) continue;
        }

        double *P_b = hc->block_prods + (size_t)b * (B + 1);
        double *g_b = g_leaf + (size_t)b * leaf_psz;
        int pk_g = g_need < bsize ? g_need : bsize;
        if (pk_g > k) pk_g = k;

        /* Precompute reciprocals and recurrence coefficients for the block */
        double inv_arr[bsize], coeff_arr[bsize];
        int fwd_arr[bsize]; /* 1=forward, 0=backward, -1=zero */
        for (int j = 0; j < bsize; j++) {
            double aj = a[start + j], bj_val = 1 - aj;
            if (aj > 0.5) {
                inv_arr[j] = 1.0 / aj;
                coeff_arr[j] = -bj_val / aj;
                fwd_arr[j] = 1;
            } else if (aj > 1e-15) {
                inv_arr[j] = 1.0 / bj_val;
                coeff_arr[j] = -aj / bj_val;
                fwd_arr[j] = 0;
            } else {
                inv_arr[j] = 0;
                coeff_arr[j] = 0;
                fwd_arr[j] = -1;
            }
        }

        for (int jj = 0; jj < bsize; jj++) {
            if (active && !active[start + jj]) { inner[start + jj] = 0; continue; }
            double eq = 0;
            if (fwd_arr[jj] == 1) {
                /* Forward divide, fused with dot product */
                double ia = inv_arr[jj], c = coeff_arr[jj];
                double Q_val = P_b[0] * ia;
                eq = g_b[0] * Q_val;
                for (int m = 1; m < pk_g; m++) {
                    Q_val = c * Q_val + P_b[m] * ia;
                    eq += g_b[m] * Q_val;
                }
            } else if (fwd_arr[jj] == 0) {
                /* Backward divide, fused with dot product */
                double ib = inv_arr[jj], c = coeff_arr[jj];
                double Q_prev = P_b[bsize] * ib;
                double Q_arr[bsize];
                Q_arr[bsize - 1] = Q_prev;
                for (int m = bsize - 2; m >= 0; m--) {
                    Q_prev = c * Q_prev + P_b[m + 1] * ib;
                    Q_arr[m] = Q_prev;
                }
                for (int m = 0; m < pk_g; m++)
                    eq += g_b[m] * Q_arr[m];
            } else {
                /* a ≈ 0: Q[m] = P[m+1] */
                for (int m = 0; m < pk_g; m++)
                    eq += g_b[m] * P_b[m + 1];
            }
            inner[start + jj] = eq;
        }
    }
}

/* engine_hybrid_ctx: EquityEngine-compatible wrapper.
 * Reorders a[] to sorted order, runs engine, unpermutes output. */
static void engine_hybrid_ctx(int n, const double *a,
                               const double *payout, int k,
                               double *inner, void *ctx) {
    HybridCtx *hc = (HybridCtx *)ctx;
    double *a_sorted = (double *)malloc(n * sizeof(double));
    double *inner_sorted = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) a_sorted[i] = a[hc->sort_perm[i]];

    engine_hybrid_core(n, a_sorted, payout, k, inner_sorted, hc);

    for (int i = 0; i < n; i++) inner[hc->sort_perm[i]] = inner_sorted[i];
    free(a_sorted);
    free(inner_sorted);
}

/* ══════════════════════════════════════════════════════════════
   ENGINE KIND — for context cloning in parallel dispatch
   ══════════════════════════════════════════════════════════════ */

typedef enum { EK_TREE, EK_NAIVE, EK_LINEAR, EK_HYBRID } EngineKind;

/* Clone a context by engine kind. The clone has independent workspace. */
static void *ctx_clone(const void *ctx, EngineKind ek, int n, int k) {
    switch (ek) {
    case EK_TREE:   return tree_ctx_clone((const TreeCtx *)ctx);
    case EK_NAIVE:  return naive_ctx_clone((const NaiveCtx *)ctx, n, k);
    case EK_LINEAR: return linear_ctx_clone((const LinearCtx *)ctx);
    case EK_HYBRID: return hybrid_ctx_clone((const HybridCtx *)ctx, n);
    }
    return NULL;
}

static void ctx_destroy(void *ctx, EngineKind ek) {
    switch (ek) {
    case EK_TREE:   tree_ctx_destroy((TreeCtx *)ctx); break;
    case EK_NAIVE:  naive_ctx_destroy((NaiveCtx *)ctx); break;
    case EK_LINEAR: linear_ctx_destroy((LinearCtx *)ctx); break;
    case EK_HYBRID: hybrid_ctx_destroy((HybridCtx *)ctx); break;
    }
}

/* ══════════════════════════════════════════════════════════════
   INTEGRATION WRAPPER (OpenMP parallel over quadrature points)
   ══════════════════════════════════════════════════════════════ */

/* Forward declaration */
static double run_engine_ctx_ex(int n, const double *S, int Q,
                                const double *payout, int k,
                                double *equity, EquityEngine engine,
                                void *ctx, EngineKind ek);

/* Auto-detect engine kind from function pointer */
static EngineKind detect_engine_kind(EquityEngine engine) {
    if (engine == engine_tree_ctx)   return EK_TREE;
    if (engine == engine_naive_ctx)  return EK_NAIVE;
    if (engine == engine_linear_ctx) return EK_LINEAR;
    if (engine == engine_hybrid_ctx) return EK_HYBRID;
    return EK_TREE; /* fallback */
}

/* Main integration wrapper — auto-detects engine kind for cloning */
static double run_engine_ctx(int n, const double *S, int Q,
                             const double *payout, int k,
                             double *equity, EquityEngine engine,
                             void *ctx) {
    return run_engine_ctx_ex(n, S, Q, payout, k, equity, engine, ctx,
                             detect_engine_kind(engine));
}

/* Compute equities for a subset of players.
 * targets: array of player indices (0-based), length n_targets.
 * equity: output array of length n (only equity[targets[i]] are set).
 * Dispatches to the best engine with active-player masks for skipping work. */
static double compute_equity_subset(int n, const double *S, int Q,
                                     const double *payout, int k,
                                     double *equity,
                                     const int *targets, int n_targets) {
    /* Build active mask */
    uint8_t *active = (uint8_t *)calloc(n, sizeof(uint8_t));
    for (int i = 0; i < n_targets; i++) active[targets[i]] = 1;

    /* Dispatch: same rule as full computation */
    int k_cross = (n >= 2048) ? 95 : 70;
    if (k >= k_cross && n >= 256) {
        HybridCtx *hc = hybrid_ctx_create(n, S, k, 8);
        /* Build sorted-order active mask */
        uint8_t *sorted_active = (uint8_t *)calloc(n, sizeof(uint8_t));
        for (int i = 0; i < n; i++) sorted_active[i] = active[hc->sort_perm[i]];
        hc->active = sorted_active;
        double t = run_engine_ctx(n, S, Q, payout, k, equity, engine_hybrid_ctx, hc);
        free(sorted_active);
        hc->active = NULL;
        hybrid_ctx_destroy(hc);
        free(active);
        return t;
    } else {
        LinearCtx *lc = linear_ctx_create(n, k);
        lc->active = active;
        double t;
        if (n >= 2048)
            t = run_linear_batched(n, S, Q, payout, k, equity, lc);
        else
            t = run_engine_ctx(n, S, Q, payout, k, equity, engine_linear_ctx, lc);
        lc->active = NULL;
        linear_ctx_destroy(lc);
        free(active);
        return t;
    }
}

static double run_engine_ctx_ex(int n, const double *S, int Q,
                                const double *payout, int k,
                                double *equity, EquityEngine engine,
                                void *ctx, EngineKind ek) {
    double Smax = 0;
    for (int i = 0; i < n; i++) if (S[i] > Smax) Smax = S[i];
    QP *pts = (QP *)malloc(Q * sizeof(QP));
    make_nodes(Q, Smax, pts);
    memset(equity, 0, n * sizeof(double));

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads > Q) nthreads = Q;
    if (nthreads < 1) nthreads = 1;

    /* Per-thread: context clone, a[], inner[], exp_args[], equity_local[] */
    void **t_ctx = (void **)malloc(nthreads * sizeof(void *));
    double **t_a = (double **)malloc(nthreads * sizeof(double *));
    double **t_inner = (double **)malloc(nthreads * sizeof(double *));
    double **t_equity = (double **)malloc(nthreads * sizeof(double *));
#ifdef __APPLE__
    double **t_exp = (double **)malloc(nthreads * sizeof(double *));
#endif
    for (int t = 0; t < nthreads; t++) {
        t_ctx[t] = (t == 0) ? ctx : ctx_clone(ctx, ek, n, k);
        t_a[t] = (double *)malloc(n * sizeof(double));
        t_inner[t] = (double *)malloc(n * sizeof(double));
        t_equity[t] = (double *)calloc(n, sizeof(double));
#ifdef __APPLE__
        t_exp[t] = (double *)malloc(n * sizeof(double));
#endif
    }

    double t0 = now_ns();
    #pragma omp parallel for schedule(dynamic, 4) num_threads(nthreads)
    for (int q = 0; q < Q; q++) {
        if (pts[q].w == 0) continue;
        int tid = omp_get_thread_num();
        double *a = t_a[tid];
        double *inner = t_inner[tid];
        double *eq_local = t_equity[tid];
        void *my_ctx = t_ctx[tid];

        double logv = pts[q].logv, wq = pts[q].w;
#ifdef __APPLE__
        double *exp_args = t_exp[tid];
        for (int j = 0; j < n; j++) exp_args[j] = S[j] * logv;
        int vn = n;
        vvexp(a, exp_args, &vn);
        for (int j = 0; j < n; j++)
            if (exp_args[j] < -700) a[j] = 0;
#else
        for (int j = 0; j < n; j++) {
            double arg = S[j] * logv;
            a[j] = (arg < -700) ? 0 : exp(arg);
        }
#endif
        engine(n, a, payout, k, inner, my_ctx);
        double inv_v = exp(-logv);
        for (int i = 0; i < n; i++) {
            double pw = wq * S[i] * a[i] * inv_v;
            if (!isfinite(pw)) pw = 0;
            eq_local[i] += pw * inner[i];
        }
    }
    double elapsed = now_ns() - t0;

    /* Merge thread-local equity arrays */
    for (int t = 0; t < nthreads; t++) {
        for (int i = 0; i < n; i++)
            equity[i] += t_equity[t][i];
    }

    /* Cleanup */
    for (int t = 0; t < nthreads; t++) {
        if (t != 0) ctx_destroy(t_ctx[t], ek);
        free(t_a[t]); free(t_inner[t]); free(t_equity[t]);
#ifdef __APPLE__
        free(t_exp[t]);
#endif
    }
    free(t_ctx); free(t_a); free(t_inner); free(t_equity);
#ifdef __APPLE__
    free(t_exp);
#endif

#else  /* No OpenMP — serial fallback */
    double *a = (double *)malloc(n * sizeof(double));
    double *inner = (double *)malloc(n * sizeof(double));
#ifdef __APPLE__
    double *exp_args = (double *)malloc(n * sizeof(double));
#endif

    double t0 = now_ns();
    for (int q = 0; q < Q; q++) {
        if (pts[q].w == 0) continue;
        double logv = pts[q].logv, wq = pts[q].w;
#ifdef __APPLE__
        for (int j = 0; j < n; j++) exp_args[j] = S[j] * logv;
        int vn = n;
        vvexp(a, exp_args, &vn);
        for (int j = 0; j < n; j++)
            if (exp_args[j] < -700) a[j] = 0;
#else
        for (int j = 0; j < n; j++) {
            double arg = S[j] * logv;
            a[j] = (arg < -700) ? 0 : exp(arg);
        }
#endif
        engine(n, a, payout, k, inner, ctx);
        double inv_v = exp(-logv);
        for (int i = 0; i < n; i++) {
            double pw = wq * S[i] * a[i] * inv_v;
            if (!isfinite(pw)) pw = 0;
            equity[i] += pw * inner[i];
        }
    }
    double elapsed = now_ns() - t0;
    free(a); free(inner);
#ifdef __APPLE__
    free(exp_args);
#endif
#endif /* _OPENMP */

    free(pts);
    return elapsed;
}

/* ══════════════════════════════════════════════════════════════
   PUBLIC API WRAPPERS
   ══════════════════════════════════════════════════════════════ */

void icm_init(const char *wisdom_path) {
    build_fftw_size_table();
    if (wisdom_path) {
        fftw_import_wisdom_from_filename(wisdom_path);
    } else {
        wisdom_load();
    }
#ifdef _OPENMP
    fftw_make_planner_thread_safe();
#endif
}

double icm_equity(int n, const double *S, int Q,
                  const double *payout, int k,
                  double *equity) {
    int k_cross = (n >= 2048) ? 95 : 70;
    if (k >= k_cross && n >= 256) {
        HybridCtx *hc = hybrid_ctx_create(n, S, k, 8);
        double t = run_engine_ctx(n, S, Q, payout, k, equity,
                                  engine_hybrid_ctx, hc);
        hybrid_ctx_destroy(hc);
        return t;
    } else if (n >= 2048) {
        LinearCtx *lc = linear_ctx_create(n, k);
        double t = run_linear_batched(n, S, Q, payout, k, equity, lc);
        linear_ctx_destroy(lc);
        return t;
    } else {
        LinearCtx *lc = linear_ctx_create(n, k);
        double t = run_engine_ctx(n, S, Q, payout, k, equity,
                                  engine_linear_ctx, lc);
        linear_ctx_destroy(lc);
        return t;
    }
}

double icm_equity_subset(int n, const double *S, int Q,
                         const double *payout, int k,
                         double *equity,
                         const int *targets, int n_targets) {
    return compute_equity_subset(n, S, Q, payout, k, equity, targets, n_targets);
}

void *icm_tree_ctx_create(int n, int k) { return tree_ctx_create(n, k); }
void *icm_hybrid_ctx_create(int n, const double *S, int k, int B) { return hybrid_ctx_create(n, S, k, B); }
void *icm_linear_ctx_create(int n, int k) { return linear_ctx_create(n, k); }

void icm_ctx_destroy(void *ctx, int engine_kind) {
    switch (engine_kind) {
    case ICM_ENGINE_TREE:   tree_ctx_destroy((TreeCtx *)ctx); break;
    case ICM_ENGINE_LINEAR: linear_ctx_destroy((LinearCtx *)ctx); break;
    case ICM_ENGINE_HYBRID: hybrid_ctx_destroy((HybridCtx *)ctx); break;
    }
}

IcmEngine icm_engine_tree(void) { return engine_tree_ctx; }
IcmEngine icm_engine_hybrid(void) { return engine_hybrid_ctx; }
IcmEngine icm_engine_linear(void) { return engine_linear_ctx; }

double icm_run_engine(int n, const double *S, int Q,
                      const double *payout, int k,
                      double *equity, IcmEngine engine, void *ctx) {
    return run_engine_ctx(n, S, Q, payout, k, equity, engine, ctx);
}

double icm_run_linear_batched(int n, const double *S, int Q,
                              const double *payout, int k,
                              double *equity, void *ctx) {
    return run_linear_batched(n, S, Q, payout, k, equity, ctx);
}

void icm_v1_exact(int n, const double *S, double *V1) { v1_exact(n, S, V1); }
void icm_v2_exact(int n, const double *S, double *V2) { v2_exact(n, S, V2); }
