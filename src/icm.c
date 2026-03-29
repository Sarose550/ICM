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
#ifdef __linux__
#include <dlfcn.h>
#endif
#include "amx.h"

/* ══════════════════════════════════════════════════════════════
   MKL DUAL DISPATCH — runtime dlopen on Linux
   MKL's FFTW wrapper has identical output format to FFTW (no repacking).
   Plans created by MKL must be executed by MKL's fftw_execute.
   ══════════════════════════════════════════════════════════════ */

#ifdef __linux__
typedef fftw_plan (*mkl_plan_r2c_fn)(int, double*, fftw_complex*, unsigned);
typedef fftw_plan (*mkl_plan_c2r_fn)(int, fftw_complex*, double*, unsigned);
typedef void (*mkl_execute_fn)(const fftw_plan);
typedef void (*mkl_execute_r2c_fn)(const fftw_plan, double*, fftw_complex*);
typedef void (*mkl_destroy_fn)(fftw_plan);

static struct {
    void *handle;
    mkl_plan_r2c_fn plan_r2c;
    mkl_plan_c2r_fn plan_c2r;
    mkl_execute_fn execute;
    mkl_execute_r2c_fn execute_r2c;
    mkl_destroy_fn destroy_plan;
} mkl = {0};

static int mkl_available = 0;

static void mkl_init(void) {
    if (mkl.handle) return;
    setenv("MKL_THREADING_LAYER", "SEQUENTIAL", 0);
    mkl.handle = dlopen("libmkl_rt.so", RTLD_NOW | RTLD_LOCAL);
    if (!mkl.handle)
        mkl.handle = dlopen("/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so",
                            RTLD_NOW | RTLD_LOCAL);
    if (!mkl.handle) return;
    mkl.plan_r2c    = (mkl_plan_r2c_fn)dlsym(mkl.handle, "fftw_plan_dft_r2c_1d");
    mkl.plan_c2r    = (mkl_plan_c2r_fn)dlsym(mkl.handle, "fftw_plan_dft_c2r_1d");
    mkl.execute     = (mkl_execute_fn)dlsym(mkl.handle, "fftw_execute");
    mkl.execute_r2c = (mkl_execute_r2c_fn)dlsym(mkl.handle, "fftw_execute_dft_r2c");
    mkl.destroy_plan = (mkl_destroy_fn)dlsym(mkl.handle, "fftw_destroy_plan");
    if (mkl.plan_r2c && mkl.plan_c2r && mkl.execute && mkl.execute_r2c && mkl.destroy_plan)
        mkl_available = 1;
}
#endif /* __linux__ */

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

/* ══════════════════════════════════════════════════════════════
   FFT PLAN CACHE — FFTW + vDSP interleaved dual-dispatch.

   On Apple Silicon, vDSP's interleaved-complex DFT API is 5-23% faster
   than FFTW at supported sizes (f × 2^g where f ∈ {1,3,5,15}, g ≥ 4).
   It uses the SAME memory format as FFTW (interleaved double pairs),
   so dispatch is nearly zero-overhead: just call a different execute
   function and fix up DC/Nyquist packing in bin 0.

   vDSP r2c packing: out[0] = DC + j*Nyquist (both purely real, packed).
   FFTW r2c packing: cbuf[0] = DC+0i, cbuf[N/2] = Nyquist+0i (separate).
   Fixup: 2 scalar copies per forward, 2 per inverse. Negligible.
   ══════════════════════════════════════════════════════════════ */

typedef struct {
    int fft_n;            /* padded size */
    fftw_plan fwd_plan;   /* r2c forward (FFTW) */
    fftw_plan inv_plan;   /* c2r inverse (FFTW) */
    double *rbuf;         /* real buffer [fft_n] */
    fftw_complex *cbuf;   /* complex buffer [fft_n/2+1] — used by BOTH backends */
#ifdef __APPLE__
    int use_vdsp;                          /* 1 if vDSP handles this size */
    vDSP_DFT_Interleaved_SetupD vdsp_fwd; /* interleaved r2c forward */
    vDSP_DFT_Interleaved_SetupD vdsp_inv; /* interleaved c2r inverse */
#endif
#ifdef __linux__
    int use_mkl;              /* 1 if MKL handles this size (from calib_lib[]) */
    fftw_plan mkl_fwd_plan;   /* MKL r2c plan (opaque, incompatible with FFTW) */
    fftw_plan mkl_inv_plan;   /* MKL c2r plan */
#endif
} FFTPlan;

typedef struct {
    FFTPlan *plans;
    int n_plans, plans_cap;
    double *rbuf2;
    fftw_complex *cbuf2;
    fftw_complex *cbuf3;  /* saved FFT(g) for correlate_fft_pair */
    int max_fft_n;
} FFTCache;

/* ── vDSP interleaved dispatch ──────────────────────────────
 * vDSP r2c with DFT length N/2 operates on DSPDoubleComplex arrays,
 * which are layout-compatible with fftw_complex (both are double[2]).
 * The only difference: bin 0 packing. vDSP: {DC, Nyquist}. FFTW: {DC, 0}.
 * After forward: unpack bin 0. Before inverse: repack bin 0.
 * Scaling: vDSP forward output = DFT × 2. Apply ×0.5 to match FFTW.
 * vDSP inverse: apply ×2 to input, inverse gives IDFT (unnormalized = ×N).
 * Measured: output is 2× FFTW, so apply final ×0.5. */

/* vDSP scaling strategy (eliminates 2 of 3 vDSP_vsmulD per round-trip):
 *
 *   Forward:  output = DFT × 2 (no scaling — leave the ×2 factor in cbuf)
 *   Forward2: same (cbuf2 also has ×2)
 *   Pointwise: A × B = (2×DFT_a) × (2×DFT_b) = 4 × DFT_a × DFT_b
 *   Inverse:  vDSP inverse of (4×product) = IDFT(4×product)/2 = 2 × N × conv
 *             Scale output by 0.5 → N × conv = FFTW convention. ✓
 *
 * Only ONE vDSP_vsmulD per round-trip (on the real output, after inverse).
 * The ×2 factor in cbuf is transparent to pointwise multiply and caching
 * because both operands carry the same factor at any given tree level. */

static inline void fft_exec_fwd(FFTPlan *p) {
#ifdef __APPLE__
    if (p->use_vdsp) {
        int hn = p->fft_n / 2;
        vDSP_DFT_Interleaved_ExecuteD(p->vdsp_fwd,
            (const DSPDoubleComplex *)p->rbuf, (DSPDoubleComplex *)p->cbuf);
        /* Unpack bin 0 only: vDSP {DC, Nyquist} → FFTW {DC, 0} + {Nyq, 0} */
        double nyq = p->cbuf[0][1];
        p->cbuf[0][1] = 0.0;
        p->cbuf[hn][0] = nyq;
        p->cbuf[hn][1] = 0.0;
        return;
    }
#endif
#ifdef __linux__
    if (p->use_mkl) { mkl.execute(p->mkl_fwd_plan); return; }
#endif
    fftw_execute(p->fwd_plan);
}

static inline void fft_exec_fwd2(FFTPlan *p, FFTCache *fc) {
#ifdef __APPLE__
    if (p->use_vdsp) {
        int hn = p->fft_n / 2;
        vDSP_DFT_Interleaved_ExecuteD(p->vdsp_fwd,
            (const DSPDoubleComplex *)fc->rbuf2, (DSPDoubleComplex *)fc->cbuf2);
        double nyq = fc->cbuf2[0][1];
        fc->cbuf2[0][1] = 0.0;
        fc->cbuf2[hn][0] = nyq;
        fc->cbuf2[hn][1] = 0.0;
        return;
    }
#endif
#ifdef __linux__
    if (p->use_mkl) { mkl.execute_r2c(p->mkl_fwd_plan, fc->rbuf2, fc->cbuf2); return; }
#endif
    fftw_execute_dft_r2c(p->fwd_plan, fc->rbuf2, fc->cbuf2);
}

static inline void fft_exec_inv(FFTPlan *p) {
#ifdef __APPLE__
    if (p->use_vdsp) {
        int n = p->fft_n, hn = n / 2;
        /* Repack bin 0: cbuf[hn] → cbuf[0].imag for vDSP convention */
        p->cbuf[0][1] = p->cbuf[hn][0];
        /* Execute inverse (cbuf carries ×2 or ×4 from forward+pointwise —
         * vDSP inverse divides by 2, giving the correct IDFT × {1 or 2}) */
        vDSP_DFT_Interleaved_ExecuteD(p->vdsp_inv,
            (const DSPDoubleComplex *)p->cbuf, (DSPDoubleComplex *)p->rbuf);
        /* Scale output: vDSP_inv(X) = X×N. The full pipeline has:
         * fwd(a)=2A, fwd2(b)=2B, pw=4AB, inv(4AB)=4NAB.
         * FFTW gives NAB. So scale by 0.25.
         * For paths with only one forward (e.g., cached correlate where
         * one operand is pre-transformed): pw=2A×cached(2B)=4AB → same factor.
         * For paths with no pointwise (bare fwd+inv round-trip): NOT used
         * in the codebase (all inv calls follow a pointwise multiply). */
        double quarter = 0.25;
        vDSP_vsmulD(p->rbuf, 1, &quarter, p->rbuf, 1, n);
        /* Restore cbuf bin 0 (in case cbuf is reused for caching) */
        p->cbuf[0][1] = 0.0;
        return;
    }
#endif
#ifdef __linux__
    if (p->use_mkl) { mkl.execute(p->mkl_inv_plan); return; }
#endif
    fftw_execute(p->inv_plan);
}

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
#ifdef __APPLE__
        /* vDSP interleaved DFT: 5-51% faster than FFTW at supported sizes.
         * Overhead per call: ~2ns (bin-0 unpack, 2 scalar copies) — negligible.
         * The ×2 scaling from vDSP forward is absorbed into pointwise multiply
         * and corrected once in the inverse (single vDSP_vsmulD on real output).
         * Supported: f × 2^g where f ∈ {1,3,5,15}, g ≥ 4 (min DFT length 16). */
        p->vdsp_fwd = NULL;
        if (sz >= 32 && (sz % 2 == 0)) {
            p->vdsp_fwd = vDSP_DFT_Interleaved_CreateSetupD(NULL, sz / 2,
                vDSP_DFT_FORWARD, vDSP_DFT_Interleaved_RealtoComplex);
        }
        if (p->vdsp_fwd) {
            p->vdsp_inv = vDSP_DFT_Interleaved_CreateSetupD(p->vdsp_fwd, sz / 2,
                vDSP_DFT_INVERSE, vDSP_DFT_Interleaved_RealtoComplex);
            p->use_vdsp = (p->vdsp_inv != NULL);
            if (!p->use_vdsp) {
                vDSP_DFT_Interleaved_DestroySetupD(p->vdsp_fwd);
                p->vdsp_fwd = NULL;
            }
        } else {
            p->use_vdsp = 0;
            p->vdsp_inv = NULL;
        }
#endif
#ifdef __linux__
        p->use_mkl = 0;
#ifdef HAS_CALIB_LIB
        if (mkl_available) {
            int lo = 0, hi = N_CALIBRATED_SIZES - 1;
            while (lo < hi) { int mid = (lo+hi)>>1; if (calib_sizes[mid] < sz) lo = mid+1; else hi = mid; }
            if (lo < N_CALIBRATED_SIZES && calib_sizes[lo] == sz && calib_lib[lo] == 1) {
                p->mkl_fwd_plan = mkl.plan_r2c(sz, p->rbuf, p->cbuf, FFTW_ESTIMATE);
                p->mkl_inv_plan = mkl.plan_c2r(sz, p->cbuf, p->rbuf, FFTW_ESTIMATE);
                if (p->mkl_fwd_plan && p->mkl_inv_plan) {
                    p->use_mkl = 1;
                } else {
                    if (p->mkl_fwd_plan) mkl.destroy_plan(p->mkl_fwd_plan);
                    if (p->mkl_inv_plan) mkl.destroy_plan(p->mkl_inv_plan);
                    p->mkl_fwd_plan = NULL;
                    p->mkl_inv_plan = NULL;
                }
            }
        }
#endif /* HAS_CALIB_LIB */
#endif /* __linux__ */
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
#ifdef __APPLE__
        if (fc->plans[i].vdsp_fwd) vDSP_DFT_Interleaved_DestroySetupD(fc->plans[i].vdsp_fwd);
        if (fc->plans[i].vdsp_inv) vDSP_DFT_Interleaved_DestroySetupD(fc->plans[i].vdsp_inv);
#endif
#ifdef __linux__
        if (fc->plans[i].use_mkl) {
            mkl.destroy_plan(fc->plans[i].mkl_fwd_plan);
            mkl.destroy_plan(fc->plans[i].mkl_inv_plan);
        }
#endif
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
   AMX SCHOOLBOOK KERNELS — FP64 outer-product tiled multiplication
   Uses lazy block-column accumulation (IACR CiC 2024, Section 4.3).
   ══════════════════════════════════════════════════════════════ */

/* Set USE_AMX=0 to disable AMX codepaths for A/B testing */
#ifndef USE_AMX
#define USE_AMX 1
#endif

#if HAS_AMX && USE_AMX && USE_AMX

/* Minimum degree for AMX schoolbook to beat scalar.
 * Below this, scalar polymul_modk / correlate_school is used.
 * Tuned empirically: AMX setup + load overhead dominates at small sizes. */
/* AMX polymul crossover: measured at d≈170 (AMX wins above this).
 * Below this, scalar polymul_modk with NEON auto-vectorization is faster.
 * With mmap page isolation this could drop to ~90. */
#ifndef AMX_SCHOOL_MIN_DEG
#define AMX_SCHOOL_MIN_DEG 160
#endif

/* Aligned workspace for AMX kernels (thread-local via engine context).
 * amx_ws must be 128-byte aligned and hold at least:
 *   - pad_a: ceil(na/8)*8 doubles (padded input a)
 *   - pad_b: ceil(nb/8)*8 doubles (padded input b)
 *   - c_lo:  8 doubles  (VECFP low anti-diag output)
 *   - c_hi:  8 doubles  (VECFP high anti-diag output)
 *   - ones:  8 doubles  (all 1.0 for VECFP multiply-by-1)
 *   - zeros: 8 doubles  (zero buffer for LDZ/LDY)
 */

static void polymul_modk_amx(const double *restrict a, int na,
                              const double *restrict b, int nb,
                              double *restrict c, int k,
                              double *amx_ws) {
    memset(c, 0, k * sizeof(double));
    int na8 = (na + 7) & ~7, nb8 = (nb + 7) & ~7;
    int nxa = na8 >> 3, nyb = nb8 >> 3;

    /* Partition workspace: pad_a | pad_b | c_lo | c_hi | ones | zeros */
    double *pad_a = amx_ws;
    double *pad_b = pad_a + na8;
    double *c_lo  = pad_b + nb8;
    double *c_hi  = c_lo + 8;
    double *ones  = c_hi + 8;
    double *zeros = ones + 8;

    memset(pad_a, 0, na8 * sizeof(double));
    memset(pad_b, 0, nb8 * sizeof(double));
    memcpy(pad_a, a, na * sizeof(double));
    memcpy(pad_b, b, nb * sizeof(double));
    for (int i = 0; i < 8; i++) ones[i] = 1.0;
    memset(zeros, 0, 64);

    int max_col = nxa + nyb - 2;

    for (int col = 0; col <= max_col; col++) {
        int base = col << 3;
        if (base >= k) break;

        /* Accumulate all outer products a_p ⊗ b_q where p+q = col */
        int first = 1;
        for (int p = (col < nxa ? col : nxa - 1); p >= 0; p--) {
            int q = col - p;
            if (q >= nyb) continue;
            AMX_LDX(amx_ldx_op(pad_a + (p << 3), 0));
            AMX_LDY(amx_ldy_op(pad_b + (q << 3), 0));
            AMX_FMA64(amx_fma64_outer(0, 0, 0, first));
            first = 0;
        }
        if (first) continue;

        /* VECFP in-register extraction: anti-diagonal sums → c_lo, c_hi */
        amx_extract_setup(ones, zeros);
        amx_extract_antidiag_vecfp(c_lo, c_hi, 0, zeros);

        /* Accumulate into output (with bounds checking) */
        int remain = k - base;
        int nlo = remain < 8 ? remain : 8;
        for (int i = 0; i < nlo; i++) c[base + i] += c_lo[i];
        if (remain > 8) {
            int nhi = remain - 8;
            if (nhi > 7) nhi = 7;
            for (int i = 0; i < nhi; i++) c[base + 8 + i] += c_hi[i];
        }
    }
}

static void correlate_school_amx(const double *restrict g, int len_g,
                                  const double *restrict P, int len_P,
                                  double *restrict out, int len_out,
                                  double *amx_ws) {
    /*
     * out[m] = Σ_j P[j] * g[m+j]  for m = 0..len_out-1
     *
     * Implemented as conv(P_rev, g) with anti-diagonal extraction:
     *   P_rev[i] = P[len_P - 1 - i]
     *   out[m] = conv(P_rev, g)[m + len_P - 1]
     *
     * This maps directly to the polymul anti-diagonal tiling because
     * i+j anti-diagonals of the outer product P_rev ⊗ g naturally
     * accumulate across block columns (unlike i-j diagonals which don't).
     */
    int conv_len = len_P + len_g - 1;
    int nP8 = (len_P + 7) & ~7, ng8 = (len_g + 7) & ~7;
    int nxa = nP8 >> 3, nyb = ng8 >> 3;

    /* Partition workspace: pad_Prev | pad_g | c_lo | c_hi | ones | zeros */
    double *pad_Prev = amx_ws;
    double *pad_g    = pad_Prev + nP8;
    double *c_lo     = pad_g + ng8;
    double *c_hi     = c_lo + 8;
    double *ones     = c_hi + 8;
    double *zeros    = ones + 8;

    /* Reverse P into padded buffer */
    memset(pad_Prev, 0, nP8 * sizeof(double));
    for (int i = 0; i < len_P; i++) pad_Prev[i] = P[len_P - 1 - i];

    memset(pad_g, 0, ng8 * sizeof(double));
    memcpy(pad_g, g, len_g * sizeof(double));
    for (int i = 0; i < 8; i++) ones[i] = 1.0;
    memset(zeros, 0, 64);

    /* Compute conv(P_rev, g) using polymul block-column tiling.
     * Only extract the range [len_P-1 .. len_P-1+len_out-1] into out[]. */
    memset(out, 0, len_out * sizeof(double));
    int max_col = nxa + nyb - 2;
    int out_start = len_P - 1;  /* conv index where out[0] lives */

    for (int col = 0; col <= max_col; col++) {
        int base = col << 3;
        if (base + 14 < out_start) continue;         /* before needed range */
        if (base > out_start + len_out - 1) break;   /* past needed range */

        /* Accumulate outer products P_rev_p ⊗ g_q where p+q = col */
        int first = 1;
        for (int p = (col < nxa ? col : nxa - 1); p >= 0; p--) {
            int q = col - p;
            if (q >= nyb) continue;
            AMX_LDX(amx_ldx_op(pad_Prev + (p << 3), 0));
            AMX_LDY(amx_ldy_op(pad_g + (q << 3), 0));
            AMX_FMA64(amx_fma64_outer(0, 0, 0, first));
            first = 0;
        }
        if (first) continue;

        /* VECFP anti-diagonal extraction → c_lo[0..7], c_hi[0..6] */
        amx_extract_setup(ones, zeros);
        amx_extract_antidiag_vecfp(c_lo, c_hi, 0, zeros);

        /* Map conv indices to out indices: out[m] = conv[m + out_start] */
        for (int d = 0; d < 8; d++) {
            int conv_idx = base + d;
            int m = conv_idx - out_start;
            if (m >= 0 && m < len_out) out[m] += c_lo[d];
        }
        for (int d = 0; d < 7; d++) {
            int conv_idx = base + 8 + d;
            int m = conv_idx - out_start;
            if (m >= 0 && m < len_out) out[m] += c_hi[d];
        }
    }
}

#endif /* HAS_AMX */

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
    fft_exec_fwd(plan);

    /* FFT(b) */
    memcpy(fc->rbuf2, b, (d + 1) * sizeof(double));
    if (d + 1 < fft_n) memset(fc->rbuf2 + d + 1, 0, (fft_n - d - 1) * sizeof(double));
    fft_exec_fwd2(plan, fc);

    /* Pointwise multiply */
    for (int i = 0; i < cn; i++) {
        double re = plan->cbuf[i][0] * fc->cbuf2[i][0]
                  - plan->cbuf[i][1] * fc->cbuf2[i][1];
        double im = plan->cbuf[i][0] * fc->cbuf2[i][1]
                  + plan->cbuf[i][1] * fc->cbuf2[i][0];
        plan->cbuf[i][0] = re;
        plan->cbuf[i][1] = im;
    }
    fft_exec_inv(plan);

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
    fft_exec_fwd(plan);
    if (fft_a_out) memcpy(fft_a_out, plan->cbuf, cn * sizeof(fftw_complex));

    /* FFT(b) */
    int copy_b = (nb < fft_n) ? nb : fft_n;
    memcpy(fc->rbuf2, b, copy_b * sizeof(double));
    if (copy_b < fft_n) memset(fc->rbuf2 + copy_b, 0, (fft_n - copy_b) * sizeof(double));
    fft_exec_fwd2(plan, fc);
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
    fft_exec_inv(plan);

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
    fft_exec_fwd(plan);
    if (fft_a_out)
        memcpy(fft_a_out, plan->cbuf, cn * sizeof(fftw_complex));

    /* FFT(b): copy then zero-pad */
    memcpy(fc->rbuf2, b, nb * sizeof(double));
    if (nb < fft_n) memset(fc->rbuf2 + nb, 0, (fft_n - nb) * sizeof(double));
    fft_exec_fwd2(plan, fc);
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

    fft_exec_inv(plan);

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
    fft_exec_fwd(plan);

    for (int j = 0; j < len_P; j++)
        fc->rbuf2[j] = P[len_P - 1 - j];
    if (len_P < fft_n) memset(fc->rbuf2 + len_P, 0, (fft_n - len_P) * sizeof(double));
    fft_exec_fwd2(plan, fc);

    for (int i = 0; i < cn; i++) {
        double re = plan->cbuf[i][0] * fc->cbuf2[i][0]
                  - plan->cbuf[i][1] * fc->cbuf2[i][1];
        double im = plan->cbuf[i][0] * fc->cbuf2[i][1]
                  + plan->cbuf[i][1] * fc->cbuf2[i][0];
        plan->cbuf[i][0] = re;
        plan->cbuf[i][1] = im;
    }

    fft_exec_inv(plan);

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
    fft_exec_fwd(plan);

    /* Save FFT(g) into cbuf3 — cbuf2 is used for FFT(P) below */
    fftw_complex *g_hat = fc->cbuf3;
    memcpy(g_hat, plan->cbuf, cn * sizeof(fftw_complex));

    /* First correlate: g × PR → outL */
    for (int j = 0; j < len_P; j++) fc->rbuf2[j] = PR[len_P - 1 - j];
    if (len_P < fft_n) memset(fc->rbuf2 + len_P, 0, (fft_n - len_P) * sizeof(double));
    fft_exec_fwd2(plan, fc);
    for (int i = 0; i < cn; i++) {
        plan->cbuf[i][0] = g_hat[i][0]*fc->cbuf2[i][0] - g_hat[i][1]*fc->cbuf2[i][1];
        plan->cbuf[i][1] = g_hat[i][0]*fc->cbuf2[i][1] + g_hat[i][1]*fc->cbuf2[i][0];
    }
    fft_exec_inv(plan);
    int offset = len_P - 1;
    for (int m = 0; m < len_out; m++)
        outL[m] = (m + offset < fft_n) ? plan->rbuf[m + offset] * inv : 0;

    /* Second correlate: g × PL → outR (reuse g_hat) */
    for (int j = 0; j < len_P; j++) fc->rbuf2[j] = PL[len_P - 1 - j];
    if (len_P < fft_n) memset(fc->rbuf2 + len_P, 0, (fft_n - len_P) * sizeof(double));
    fft_exec_fwd2(plan, fc);
    for (int i = 0; i < cn; i++) {
        plan->cbuf[i][0] = g_hat[i][0]*fc->cbuf2[i][0] - g_hat[i][1]*fc->cbuf2[i][1];
        plan->cbuf[i][1] = g_hat[i][0]*fc->cbuf2[i][1] + g_hat[i][1]*fc->cbuf2[i][0];
    }
    fft_exec_inv(plan);
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
    fft_exec_fwd(plan);

    /* Pointwise: FFT(g) * conj(FFT(P)) — cross-correlation in freq domain */
    for (int i = 0; i < cn; i++) {
        double pr = cached_fft_P[i][0], pi = cached_fft_P[i][1];
        double gr = plan->cbuf[i][0], gi = plan->cbuf[i][1];
        plan->cbuf[i][0] = gr * pr + gi * pi;
        plan->cbuf[i][1] = gi * pr - gr * pi;
    }

    fft_exec_inv(plan);

    /* Cross-correlation result is directly at index m (no offset) */
    for (int m = 0; m < len_out; m++)
        out[m] = (m < fft_n) ? plan->rbuf[m] * inv : 0;
}

/* Cached correlate PAIR with m-wrap correction.
 * Shares FFT(g) across two correlations (one per sibling), AND uses
 * cached FFT(P) from the build phase. Saves one forward FFT per parent node
 * vs calling correlate_fft_cached_wrap twice. */
/* Correct input-side cyclic aliasing in cross-correlation.
 * When FFT size N < len_g + len_P - 1, positions m near the end of the
 * output pick up spurious contributions from g wrapping around mod N.
 * Specifically: at position m, for j where m+j >= N, the cyclic FFT
 * reads g[(m+j)-N] instead of 0. This adds P[j]*g[(m+j)-N] to corr[m].
 * We compute and subtract these spurious terms via schoolbook. */
static inline void correlate_wrap_input_correction(double *out, int len_out,
                                             const double *P, int len_P,
                                             const double *g, int len_g,
                                             int fft_n) {
    int m_start = fft_n - len_P + 1;
    if (m_start < 0) m_start = 0;
    for (int m = m_start; m < len_out && m < fft_n; m++) {
        double alias = 0;
        int j_lo = fft_n - m;
        for (int j = j_lo; j < len_P; j++) {
            int g_idx = (m + j) - fft_n;
            if (g_idx >= 0 && g_idx < len_g)
                alias += P[j] * g[g_idx];
        }
        out[m] -= alias;
    }
}

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
    fft_exec_fwd(plan);
    fftw_complex *g_hat = fc->cbuf3;
    memcpy(g_hat, plan->cbuf, cn * sizeof(fftw_complex));

    /* First correlate: g × PR → outL (using cached FFT(PR)) */
    for (int i = 0; i < cn; i++) {
        double pr = cached_fft_PR[i][0], pi = cached_fft_PR[i][1];
        double gr = g_hat[i][0], gi = g_hat[i][1];
        plan->cbuf[i][0] = gr * pr + gi * pi;
        plan->cbuf[i][1] = gi * pr - gr * pi;
    }
    fft_exec_inv(plan);
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
        correlate_wrap_input_correction(outL, len_out, PR, len_P, g, len_g, fft_n);
    }

    /* Second correlate: g × PL → outR (reuse g_hat, using cached FFT(PL)) */
    for (int i = 0; i < cn; i++) {
        double pr = cached_fft_PL[i][0], pi = cached_fft_PL[i][1];
        double gr = g_hat[i][0], gi = g_hat[i][1];
        plan->cbuf[i][0] = gr * pr + gi * pi;
        plan->cbuf[i][1] = gi * pr - gr * pi;
    }
    fft_exec_inv(plan);
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
        correlate_wrap_input_correction(outR, len_out, PL, len_P, g, len_g, fft_n);
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
    fft_exec_fwd(plan);

    /* Pointwise: FFT(g) * conj(FFT(P)) */
    for (int i = 0; i < cn; i++) {
        double pr = cached_fft_P[i][0], pi = cached_fft_P[i][1];
        double gr = plan->cbuf[i][0], gi = plan->cbuf[i][1];
        plan->cbuf[i][0] = gr * pr + gi * pi;
        plan->cbuf[i][1] = gi * pr - gr * pi;
    }
    fft_exec_inv(plan);

    /* Extract cyclic cross-correlation result */
    for (int m = 0; m < len_out; m++)
        out[m] = (m < fft_n) ? plan->rbuf[m] * inv : 0;

    /* Correction 1: OUTPUT aliasing — high positions (fft_n..fft_n+wrap_m) wrap to 0..wrap_m.
     * Compute true cross-correlation at those high positions, subtract from wrapped low
     * positions, and place at the correct high positions. */
    for (int i = 0; i <= wrap_m; i++) {
        int pos = fft_n + i;
        if (pos >= conv_len) break;
        double high = 0;
        int j_max = len_g - pos;
        if (j_max > len_P) j_max = len_P;
        for (int j = 0; j < j_max; j++)
            high += P[j] * g[pos + j];
        if (i < len_out) out[i] -= high;
        if (pos < len_out) out[pos] = high;
    }

    /* Correction 2: INPUT aliasing (g wraps within cyclic FFT period) */
    if (wrap_m > 0)
        correlate_wrap_input_correction(out, len_out, P, len_P, g, len_g, fft_n);
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
#if HAS_AMX && USE_AMX
    double *amx_ws;                          /* 128-byte aligned AMX scratch (pad_a + pad_b + tile + zeros) */
    size_t amx_ws_size;
    int any_amx_school;                      /* 1 if any level dispatches to AMX schoolbook */
#endif
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

#if HAS_AMX && USE_AMX
    /* AMX workspace: enough for largest schoolbook at any level.
     * Layout: pad_a[max_psz_8] + pad_b[max_psz_8] + c_lo[8] + c_hi[8] + ones[8] + zeros[8]
     * where max_psz_8 = ceil(max_psz / 8) * 8. */
    {
        int max_psz_8 = (max_psz + 7) & ~7;
        tc->amx_ws_size = 2 * max_psz_8 + 32;
        posix_memalign((void **)&tc->amx_ws, 128, tc->amx_ws_size * sizeof(double));
    }
#endif

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

            /* Per-level FFT vs schoolbook decision.
             * Compare TOTAL cost (build + correlate) for both paths, since
             * choosing schoolbook for build also means schoolbook for correlate.
             * Below-sat polys have actual degree cps/2 (upper half is zero). */
            int d_eff = is_below ? cps / 2 : cps - 1;
            long long school_cost_flops = (long long)(d_eff + 1) * (d_eff + 1);
            int build_conv_len = is_below ? (2 * (cps / 2)) : (2 * cps - 1);
            int bfn, bwm;
            best_fft_config(build_conv_len, &bfn, &bwm, 0);  /* polymul: no input-wrap */
            int fft_idx;
            { int lo=0,hi=N_CALIBRATED_SIZES-1;
              while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<bfn)lo=m+1;else hi=m;}
              fft_idx = lo; }
            double fft_build = (fft_idx < N_CALIBRATED_SIZES && calib_sizes[fft_idx] == bfn)
                               ? calib_times_ns[fft_idx] : 1e18;
            double fft_overhead = FFT_OVERHEAD_NS;
            double build_correction = (double)(bwm + 1) * (bwm + 1) * FMA_NS;

            /* Build cost comparison: schoolbook vs FFT.
             * Uses build-phase cost only (not correlate), as this was empirically
             * validated to give correct B and FFT crossover decisions.
             * The correlate cost is implicitly handled: when schoolbook wins the
             * build comparison, correlate also uses schoolbook, and the cost model
             * in select_best_B accounts for both via the tree cost sum. */
            double school_build = school_cost_flops * FMA_NS;
            int p_eff = is_below ? cps/2 + 1 : cps;
            int out_needed = tc->g_needed[ell-1];
            tc->use_fft[ell] = (fft_build + fft_overhead + build_correction < school_build);

            if (tc->use_fft[ell]) {
                tc->build_fft_n[ell] = bfn;
                tc->build_wrap_m[ell] = bwm;
                needed[n_needed++] = bfn;

                /* Correlate FFT size (p_eff, out_needed already computed above) */
                int g_eff_needed = out_needed + p_eff - 1;
                int g_eff_max = is_below ? (cps + cps/2) : pgsz;
                int g_eff = (g_eff_needed < g_eff_max) ? g_eff_needed : g_eff_max;
                int corr_conv = g_eff + p_eff - 1;

                if (!is_root) {
                    /* Compare joint (cached) vs independent (uncached).
                     * Joint: one FFT size, cache FFT(P) from build, reuse in correlates.
                     * Independent: build and correlate each pick optimal sizes, no caching. */
                    int jfn, jbm, jcm;
                    double jcost = best_fft_config_joint(build_conv_len, corr_conv, p_eff,
                                                          &jfn, &jbm, &jcm);

                    int cfn, cwm;
                    best_fft_config(corr_conv, &cfn, &cwm, p_eff);  /* correlate: input-wrap cost */
                    /* Independent: build(bfn) + correlate_fft_pair(cfn).
                     * Pair shares g FFT: 1 fwd(g) + 2 fwd(P_rev) + 2 pw + 2 ifft
                     * ≈ 1.25× full pipeline. */
                    int cfn_idx = 0;
                    { int lo2=0, hi2=N_CALIBRATED_SIZES-1;
                      while(lo2<hi2){int m2=(lo2+hi2)>>1;if(calib_sizes[m2]<cfn)lo2=m2+1;else hi2=m2;}
                      cfn_idx = lo2; }
                    /* Independent correlate cost: output-wrap + input-wrap (for correlate) */
                    double corr_correction = 2.0 * ((double)(cwm+1)*(cwm+1) + (double)cwm * p_eff) * FMA_NS;
                    double icost = fft_build + build_correction
                        + INDEP_PAIR_RATIO * calib_times_ns[cfn_idx]
                        + corr_correction;

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
                    best_fft_config(corr_conv, &cfn, &cwm, p_eff);  /* correlate: input-wrap cost */
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

#if HAS_AMX && USE_AMX
    /* Determine if any level will dispatch to AMX schoolbook.
     * If not, skip AMX_SET/CLR to avoid overhead. */
    tc->any_amx_school = 0;
    for (int ell = 1; ell < tc->L; ell++) {
        if (!tc->use_fft[ell]) {
            int cps = tc->psz[ell-1];
            int d_eff_b = tc->below_sat[ell] ? cps/2 : cps-1;
            if (d_eff_b + 1 >= AMX_SCHOOL_MIN_DEG) {
                tc->any_amx_school = 1;
                break;
            }
        }
    }
#endif

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
#if HAS_AMX && USE_AMX
    free(tc->amx_ws);
#endif
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
#ifdef __APPLE__
        dp->use_vdsp = sp->use_vdsp;
        if (sp->use_vdsp) {
            dp->vdsp_fwd = vDSP_DFT_Interleaved_CreateSetupD(NULL, dp->fft_n / 2,
                vDSP_DFT_FORWARD, vDSP_DFT_Interleaved_RealtoComplex);
            dp->vdsp_inv = vDSP_DFT_Interleaved_CreateSetupD(dp->vdsp_fwd, dp->fft_n / 2,
                vDSP_DFT_INVERSE, vDSP_DFT_Interleaved_RealtoComplex);
        } else {
            dp->vdsp_fwd = NULL;
            dp->vdsp_inv = NULL;
        }
#endif
#ifdef __linux__
        dp->use_mkl = sp->use_mkl;
        if (sp->use_mkl && mkl_available) {
            dp->mkl_fwd_plan = mkl.plan_r2c(dp->fft_n, dp->rbuf, dp->cbuf, FFTW_ESTIMATE);
            dp->mkl_inv_plan = mkl.plan_c2r(dp->fft_n, dp->cbuf, dp->rbuf, FFTW_ESTIMATE);
        } else {
            dp->mkl_fwd_plan = NULL;
            dp->mkl_inv_plan = NULL;
            dp->use_mkl = 0;
        }
#endif
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
#if HAS_AMX && USE_AMX
    posix_memalign((void **)&tc->amx_ws, 128, tc->amx_ws_size * sizeof(double));
#endif
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
                    fft_exec_fwd(plan);
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
#if HAS_AMX && USE_AMX
                    /* Dispatch on effective degree, not cps: below-sat polys
                     * have actual degree cps/2 (upper half zero), where scalar
                     * polymul_modk skips zeros but AMX doesn't. */
                    int d_eff_b = tc->below_sat[ell] ? cps/2 : cps-1;
                    if (d_eff_b + 1 >= AMX_SCHOOL_MIN_DEG)
                        polymul_modk_amx(Lc, cps, Rc, cps, out, pps, tc->amx_ws);
                    else
#endif
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
#if HAS_AMX && USE_AMX
                    if (p_eff >= AMX_SCHOOL_MIN_DEG) {
                        correlate_school_amx(gp, g_eff, PR, p_eff, gL, out_needed, tc->amx_ws);
                        correlate_school_amx(gp, g_eff, PL, p_eff, gR, out_needed, tc->amx_ws);
                    } else {
#endif
                    correlate_school(gp, g_eff, PR, p_eff, gL, out_needed);
                    correlate_school(gp, g_eff, PL, p_eff, gR, out_needed);
#if HAS_AMX && USE_AMX
                    }
#endif
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

#if HAS_AMX && USE_AMX
    if (tc->any_amx_school) AMX_SET();
#endif
    tree_build_levels(tc);
    double *g_leaf = tree_propagate_g(tc, k, payout);
#if HAS_AMX && USE_AMX
    if (tc->any_amx_school) AMX_CLR();
#endif

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
#ifndef CKPT_THRESHOLD
#define CKPT_THRESHOLD 4194304
#endif  /* 32MB / 8 bytes */

/* L2 cache size per core (bytes). Used for batched linear checkpointing. */
#ifndef L2_CACHE_SIZE
#define L2_CACHE_SIZE 1048576  /* 1MB for Zen 4; 32MB for M3 Max */
#endif

/* Bandwidth constants for roofline cost model (measured by calibrate).
 * Fallback defaults for uncalibrated devices. */
#ifndef L2_BW_GBS
#define L2_BW_GBS 200.0
#endif
#ifndef L3_BW_GBS
#define L3_BW_GBS 80.0
#endif
#ifndef DRAM_BW_GBS
#define DRAM_BW_GBS 45.0
#endif
#ifndef L3_CACHE_SIZE
#define L3_CACHE_SIZE 33554432  /* 32MB */
#endif

/* ══════════════════════════════════════════════════════════════
   ADAPTIVE BQ — compile-time parameterized batched linear engine.
   Two versions (BQ=4 and BQ=8) compiled via template inclusion.
   Runtime dispatch based on L2 working set: n*k*BQ*8 ≤ L2_CACHE_SIZE.
   ══════════════════════════════════════════════════════════════ */

/* BQ for non-batched paths (engine_linear_ctx, etc.) — always 2 */
#ifndef BQ
#define BQ 2
#endif


static int ckpt_interval(int n) {
    int C = (int)sqrt((double)n);
    if (C < 2) C = 2;
    return C;
}

/* ckpt_interval_batched is now in linear_batched_impl.inc (parameterized by BQ) */

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
   BATCHED LINEAR ENGINE — adaptive BQ via template instantiation.
   Two versions compiled: BQ=8 (3x throughput) and BQ=4 (2x throughput).
   Runtime dispatch: BQ=8 when g_store fits in L2, BQ=4 otherwise.
   Inner q-loops auto-vectorize to full-width FMAs with -O3.
   ══════════════════════════════════════════════════════════════ */

/* Instantiate BQ=8 batched linear engine.
 * BQ=8 wins on all platforms: AVX-512 (native 512-bit), Apple Silicon
 * (4 NEON FMA ports × 128-bit = 512-bit aggregate). The interleaved
 * a_batch layout (a_batch[j*BQ+qi]) ensures cache-friendly access
 * regardless of n, so no BQ=4 fallback is needed. */
#define BQ_IMPL 8
#define SUFFIX _bq8
#include "linear_batched_impl.inc"
#undef SUFFIX
#undef BQ_IMPL

#define run_linear_batched run_linear_batched_bq8

/* ══════════════════════════════════════════════════════════════
   HYBRID ENGINE (block build + tree + bidirectional divide)
   ══════════════════════════════════════════════════════════════ */

typedef struct {
    int B, nblocks, n;
    double *block_prods;  /* nblocks_padded * (B+1) doubles */
    TreeCtx *tc;          /* inter-block tree */
    int *sort_perm;       /* sort_perm[i] = original index of sorted player i */
    double *S_sorted;     /* stack sizes in sorted order */
    double *a_sorted;     /* pre-allocated permutation buffer (n doubles) */
    double *inner_sorted; /* pre-allocated permutation buffer (n doubles) */
    uint8_t *active;      /* per-player mask (sorted order). NULL = all. */
    int owns_sorted;      /* 1 if this ctx owns sort_perm/S_sorted (0 for clones) */
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
    hc->n = n;
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
    hc->a_sorted = (double *)malloc(n * sizeof(double));
    hc->inner_sorted = (double *)malloc(n * sizeof(double));
    hc->owns_sorted = 1;

    return hc;
}

static void hybrid_ctx_destroy(HybridCtx *hc) {
    free(hc->block_prods);
    if (hc->owns_sorted) {
        free(hc->sort_perm);
        free(hc->S_sorted);
    }
    free(hc->a_sorted);
    free(hc->inner_sorted);
    tree_ctx_destroy(hc->tc);
    free(hc);
}

static HybridCtx *hybrid_ctx_clone(const HybridCtx *src, int n) {
    HybridCtx *hc = (HybridCtx *)calloc(1, sizeof(HybridCtx));
    hc->B = src->B;
    hc->n = n;
    hc->nblocks = src->nblocks;
    int N_tree = src->tc->N;
    hc->block_prods = (double *)calloc((size_t)N_tree * (src->B + 1), sizeof(double));
    hc->tc = tree_ctx_clone(src->tc);
    /* Share sort_perm and S_sorted (read-only during engine execution) */
    hc->sort_perm = src->sort_perm;
    hc->S_sorted = src->S_sorted;
    hc->a_sorted = (double *)malloc(n * sizeof(double));
    hc->inner_sorted = (double *)malloc(n * sizeof(double));
    hc->owns_sorted = 0;
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

#if HAS_AMX && USE_AMX
    if (tc->any_amx_school) AMX_SET();
#endif
    tree_build_levels(tc);
    double *g_leaf = tree_propagate_g(tc, k, payout);
#if HAS_AMX && USE_AMX
    if (tc->any_amx_school) AMX_CLR();
#endif
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
    double *a_sorted = hc->a_sorted;
    double *inner_sorted = hc->inner_sorted;
    for (int i = 0; i < n; i++) a_sorted[i] = a[hc->sort_perm[i]];

    engine_hybrid_core(n, a_sorted, payout, k, inner_sorted, hc);

    for (int i = 0; i < n; i++) inner[hc->sort_perm[i]] = inner_sorted[i];
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
static int select_best_B(int n, int k);

/* Batch width of the linear engine used by icm_equity (run_linear_batched_bq8). */
#define LINEAR_BQ 8

/* Effective streaming bandwidth for a working set of `bytes` total.
 * When data fits in a cache level, use that level's bandwidth.
 * When data spills across a boundary, blend using harmonic mean:
 *   eff_bw = 1 / (hit_frac / hit_bw + miss_frac / miss_bw)
 * This follows from: total_time = hit_bytes/hit_bw + miss_bytes/miss_bw. */
static double blended_bw(double bytes) {
    if (bytes <= (double)L2_CACHE_SIZE)
        return L2_BW_GBS;
    if (bytes <= (double)L3_CACHE_SIZE) {
        double l2_frac = (double)L2_CACHE_SIZE / bytes;
        return 1.0 / (l2_frac / L2_BW_GBS + (1.0 - l2_frac) / L3_BW_GBS);
    }
    double l3_frac = (double)L3_CACHE_SIZE / bytes;
    return 1.0 / (l3_frac / L3_BW_GBS + (1.0 - l3_frac) / DRAM_BW_GBS);
}

/* Engine dispatch: compare estimated linear vs hybrid cost for given (n, k).
 * Returns the optimal B if hybrid wins, or 0 if linear wins.
 * Linear cost: roofline model — bytes streamed / bandwidth at the relevant cache level.
 * Hybrid cost: block build + FFT tree from the same model as select_best_B. */
static int select_engine(int n, int k) {
    if (n < 16 || k < 4) return 0;

    /* ── Roofline cost model for the batched linear engine ──
     *
     * The batched engine (BQ=8) is memory-bandwidth-limited: its arithmetic
     * intensity (~0.15 FLOP/byte) is far below the machine balance point.
     * Instead of a single POLYMUL_FMA_NS constant, we estimate the bytes
     * streamed through each cache level and divide by measured bandwidth.
     *
     * Checkpoint interval C = L2_CACHE_SIZE / (k * BQ * sizeof(double)).
     * All costs are per quadrature point (batch cost / BQ). */

    int C = (int)((size_t)L2_CACHE_SIZE / ((size_t)k * LINEAR_BQ * sizeof(double)));
    if (C < 1) C = 1;

    double linear_per_qp;

    if (C >= n) {
        /* No checkpointing — g_store fits in L2.
         * Forward + backward: 2 passes of n*k doubles (read+write).
         * Per QP (amortized over BQ): 2 * n * k * 8 bytes. */
        double bytes_per_qp = 2.0 * n * k * 8.0;
        double bytes_per_batch = bytes_per_qp * LINEAR_BQ;
        double bw = blended_bw(bytes_per_batch);
        linear_per_qp = bytes_per_qp / bw;
    } else {
        /* Checkpointed: local_g fits in L2 (by design of C).
         *
         * Inner work (L2-resident): recompute forward + backward within each
         * segment.  2 passes of n*k doubles, all hitting L2.
         * Per QP: 2 * n * k * 8 / L2_BW. */
        double inner_bytes = 2.0 * n * k * 8.0;
        double inner_time = inner_bytes / L2_BW_GBS;

        /* Outer I/O (L3 and/or DRAM):
         *   Checkpoints: (n/C) × k×BQ×8 bytes, read+write → 2×(n/C)×k×BQ×8
         *   a_batch: n×BQ×8 bytes, read 3 times → 3×n×BQ×8
         * Per QP (÷BQ): 2*(n/C)*k*8 + 3*n*8 */
        double ckpt_bytes = 2.0 * ((double)n / C) * k * 8.0;
        double abatch_bytes = 3.0 * n * 8.0;
        double outer_bytes = ckpt_bytes + abatch_bytes;
        double outer_total = outer_bytes * LINEAR_BQ;
        double outer_bw = blended_bw(outer_total);
        double outer_time = outer_bytes / outer_bw;

        linear_per_qp = inner_time + outer_time;
    }

    int B = select_best_B(n, k);
    int nblocks = (n + B - 1) / B;
    double block = ((double)n / B * ((double)B * (B+1) / 2.0) + (double)n * 3.0 * B) * FMA_NS;
    TreeCtx *tc = tree_ctx_create_ex2(nblocks, B, k, B);
    double tree = 0;
    for (int ell = 1; ell < tc->L - 1; ell++) {
        int cps = tc->psz[ell-1], nr = tc->n_real[ell];
        if (tc->use_fft[ell]) {
            int bfn = tc->build_fft_n[ell];
            int idx = 0;
            { int lo=0,hi=N_CALIBRATED_SIZES-1;
              while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<bfn)lo=m+1;else hi=m;}
              idx=lo; }
            double fft_cost = calib_times_ns[idx] + FFT_OVERHEAD_NS;
            double corr_cost = fft_cost * PAIRED_CACHED_CORR_RATIO;
            tree += nr * (fft_cost + corr_cost);
        } else {
            int d_eff = tc->below_sat[ell] ? cps/2 : cps-1;
            double s = (double)(d_eff+1)*(d_eff+1)*FMA_NS;
            double c = (double)cps * tc->g_needed[ell-1] * FMA_NS * 2;
            tree += nr * (s + c);
        }
    }
    tree_ctx_destroy(tc);
    return (block + tree < linear_per_qp) ? B : 0;
}

static double compute_equity_subset(int n, const double *S, int Q,
                                     const double *payout, int k,
                                     double *equity,
                                     const int *targets, int n_targets) {
    /* Build active mask */
    uint8_t *active = (uint8_t *)calloc(n, sizeof(uint8_t));
    for (int i = 0; i < n_targets; i++) active[targets[i]] = 1;

    int B = select_engine(n, k);
    if (B > 0) {
        HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
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
        t = run_linear_batched(n, S, Q, payout, k, equity, lc);
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
    double *a_buf = (double *)malloc((size_t)BQ * n * sizeof(double));
    double *inner_buf = (double *)malloc((size_t)BQ * n * sizeof(double));

    int n_batches_e = Q / BQ;
    int leftover_e = Q - n_batches_e * BQ;

    double t0 = now_ns();
    for (int b = 0; b < n_batches_e; b++) {
        int qb = b * BQ;
        int all_zero = 1;
        for (int qi = 0; qi < BQ; qi++) if (pts[qb+qi].w != 0) { all_zero = 0; break; }
        if (all_zero) continue;

        for (int qi = 0; qi < BQ; qi++) {
            double logv = pts[qb + qi].logv;
            double *aq = a_buf + (size_t)qi * n;
            for (int j = 0; j < n; j++) {
                double arg = S[j] * logv;
                aq[j] = (arg < -700) ? 0 : exp(arg);
            }
        }

        for (int qi = 0; qi < BQ; qi++) {
            if (pts[qb + qi].w == 0) continue;
            engine(n, a_buf + (size_t)qi * n, payout, k,
                   inner_buf + (size_t)qi * n, ctx);
        }

        double wq_b[BQ], iv_b[BQ];
        for (int qi = 0; qi < BQ; qi++) {
            wq_b[qi] = pts[qb + qi].w;
            iv_b[qi] = exp(-pts[qb + qi].logv);
        }
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int qi = 0; qi < BQ; qi++) {
                double pw = wq_b[qi] * S[i] * a_buf[(size_t)qi * n + i] * iv_b[qi];
                if (!isfinite(pw)) pw = 0;
                sum += pw * inner_buf[(size_t)qi * n + i];
            }
            equity[i] += sum;
        }
    }

    /* Handle leftover quad points */
    for (int q = n_batches_e * BQ; q < Q; q++) {
        if (pts[q].w == 0) continue;
        double logv = pts[q].logv, wq = pts[q].w;
        for (int j = 0; j < n; j++) {
            double arg = S[j] * logv;
            a_buf[j] = (arg < -700) ? 0 : exp(arg);
        }
        engine(n, a_buf, payout, k, inner_buf, ctx);
        double inv_v = exp(-logv);
        for (int i = 0; i < n; i++) {
            double pw = wq * S[i] * a_buf[i] * inv_v;
            if (!isfinite(pw)) pw = 0;
            equity[i] += pw * inner_buf[i];
        }
    }

    double elapsed = now_ns() - t0;
    free(a_buf); free(inner_buf);
#endif /* _OPENMP */

    free(pts);
    return elapsed;
}


/* ---- Optimal B selector (cost-model driven) ---- */


/* Select optimal hybrid block size B from calibration data.
 * Creates lightweight TreeCtx for each candidate to use the real per-level
 * planning decisions (schoolbook vs FFT, joint vs independent, FFT sizes).
 * O(log n) per candidate, negligible vs engine work. */
static int select_best_B(int n, int k) {
    int candidates[] = {8, 16, 24, 32, 48, 64};
    int n_cand = 6;
    double best_cost = 1e18;
    int best_B = 16;
    for (int ci = 0; ci < n_cand; ci++) {
        int B = candidates[ci];
        if (B > k || B > n) continue;
        int n_leaves = (n + B - 1) / B;
        TreeCtx *tc = tree_ctx_create_ex2(n_leaves, B, k, B);
        int L = tc->L;
        double block = ((double)n / B * ((double)B * (B+1) / 2.0) + (double)n * 3.0 * B) * FMA_NS;
        double tree = 0;
        for (int ell = 1; ell < L - 1; ell++) {
            int cps = tc->psz[ell-1];
            int nr = tc->n_real[ell];
            if (tc->use_fft[ell]) {
                int bfn = tc->build_fft_n[ell];
                int bwm = tc->build_wrap_m[ell];
                int idx = 0;
                { int lo=0,hi=N_CALIBRATED_SIZES-1;
                  while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<bfn)lo=m+1;else hi=m;}
                  idx=lo; }
                double build_fft = calib_times_ns[idx] + FFT_OVERHEAD_NS
                                 + (double)(bwm+1)*(bwm+1)*FMA_NS;
                double corr;
                if (tc->fft_cache_ok[ell]) {
                    corr = calib_times_ns[idx] * PAIRED_CACHED_CORR_RATIO
                         + 2.0*(double)(tc->corr_wrap_m[ell]+1)*(tc->corr_wrap_m[ell]+1)*FMA_NS;
                } else {
                    int cfn = tc->corr_fft_n[ell];
                    int cwm = tc->corr_wrap_m[ell];
                    int cidx=0;
                    {int lo=0,hi=N_CALIBRATED_SIZES-1;
                     while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<cfn)lo=m+1;else hi=m;}
                     cidx=lo;}
                    corr = INDEP_PAIR_RATIO * calib_times_ns[cidx]
                         + 2.0*(double)(cwm+1)*(cwm+1)*FMA_NS;
                }
                tree += nr * (build_fft + corr);
            } else {
                int is_below = tc->below_sat[ell];
                int d_eff = is_below ? cps/2 : cps-1;
                double school_mul, school_corr;
#if HAS_AMX && USE_AMX
                if (d_eff + 1 >= AMX_SCHOOL_MIN_DEG) {
                    int nb = ((d_eff+1)+7) >> 3;
                    school_mul = nb*nb*AMX_TILE_NS + (2*nb-1)*AMX_PERCOL_NS + AMX_CALL_NS;
                } else
#endif
                    school_mul = (double)(d_eff+1)*(d_eff+1)*FMA_NS;
                /* Correlate uses scalar (inner loop is memory-bound at FMA_NS) */
                school_corr = (double)cps * tc->g_needed[ell-1] * FMA_NS * 2;
                tree += nr * (school_mul + school_corr);
            }
        }
        tree_ctx_destroy(tc);
        double total = block + tree;
        if (total < best_cost) { best_cost = total; best_B = B; }
    }
    return best_B;
}

/* ==============================================================
   PUBLIC API WRAPPERS
   ============================================================== */

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
#ifdef __linux__
    mkl_init();
#endif
}

double icm_equity(int n, const double *S, int Q,
                  const double *payout, int k,
                  double *equity) {
    int B = select_engine(n, k);
    if (B > 0) {
        HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
        double t = run_engine_ctx(n, S, Q, payout, k, equity,
                                  engine_hybrid_ctx, hc);
        hybrid_ctx_destroy(hc);
        return t;
    } else {
        LinearCtx *lc = linear_ctx_create(n, k);
        double t = run_linear_batched(n, S, Q, payout, k, equity, lc);
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

