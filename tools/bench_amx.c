/*
 * bench_amx.c — AMX FP64 outer-product polynomial multiply benchmark
 *
 * Validates AMX FMA64 encoding, measures throughput, and compares
 * AMX-accelerated schoolbook polynomial multiplication against
 * NEON-vectorized scalar schoolbook.
 *
 * Build:  clang -O3 -o bench_amx tools/bench_amx.c -lm
 * Run:    ./bench_amx
 *
 * Apple Silicon only (M1/M2/M3/M4). Uses undocumented AMX instructions
 * via inline assembly (corsix/amx encoding).
 *
 * References:
 *   - https://github.com/corsix/amx
 *   - "Fast polynomial multiplication using matrix multiplication
 *     accelerators" (IACR CiC 2024), Gazzoni Filho et al.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "../src/amx.h"

/* Backward-compat aliases for the rest of this file */
#if HAS_AMX
#define ld_xy_op     amx_ldx_op
#define ld_st_z_op   amx_ldz_op
#define fma64_outer  amx_fma64_outer
static int g_z_stride = 0;
#endif

/* ══════════════════════════════════════════════════════════════
   Timing
   ══════════════════════════════════════════════════════════════ */

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ══════════════════════════════════════════════════════════════
   STEP 1: Probe FMA64 Z row layout
   ══════════════════════════════════════════════════════════════ */

#if HAS_AMX
static int probe_fma64(void) {
    /*
     * Load X0 = [1, 0, 0, 0, 0, 0, 0, 0]
     *      Y0 = [1, 2, 3, 4, 5, 6, 7, 8]
     *
     * Outer product Z[j][i] = X[i]*Y[j]:
     *   row for Y[0]=1: [1, 0, 0, 0, 0, 0, 0, 0]
     *   row for Y[1]=2: [2, 0, 0, 0, 0, 0, 0, 0]
     *   ...
     *   row for Y[7]=8: [8, 0, 0, 0, 0, 0, 0, 0]
     *
     * We scan all 64 Z rows to find which contain data → reveals the stride.
     */
    double *x, *y, *z;
    posix_memalign((void **)&x, 128, 64);
    posix_memalign((void **)&y, 128, 64);
    posix_memalign((void **)&z, 128, 64);

    memset(x, 0, 64);
    ((double *)x)[0] = 1.0;
    for (int i = 0; i < 8; i++) ((double *)y)[i] = (double)(i + 1);

    AMX_SET();

    /* Load inputs */
    AMX_LDX(ld_xy_op(x, 0));
    AMX_LDY(ld_xy_op(y, 0));

    /* Zero ALL 64 Z rows */
    memset(z, 0, 64);
    for (int r = 0; r < 64; r++)
        AMX_LDZ(ld_st_z_op(z, r));

    /* Execute outer product: Z = X0 outer Y0, no accumulate (overwrite) */
    AMX_FMA64(fma64_outer(0, 0, 0, 1));

    /* Scan Z rows for results */
    printf("FMA64 Z row probe (z_row_base=0):\n");
    int found[8], nfound = 0;
    for (int r = 0; r < 64 && nfound < 8; r++) {
        AMX_STZ(ld_st_z_op(z, r));
        double *zd = (double *)z;
        if (zd[0] != 0.0) {
            printf("  Z[%2d] = [%.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f]\n",
                   r, zd[0], zd[1], zd[2], zd[3], zd[4], zd[5], zd[6], zd[7]);
            found[nfound++] = r;
        }
    }

    AMX_CLR();

    /* Determine stride */
    int stride = 0;
    if (nfound >= 2) {
        stride = found[1] - found[0];
        int ok = 1;
        for (int i = 2; i < nfound; i++)
            if (found[i] - found[i - 1] != stride) { ok = 0; break; }
        printf("  → stride = %d, %d independent accumulators",
               stride, stride);
        if (!ok) printf(" (WARNING: non-uniform stride!)");
        printf("\n");
    } else if (nfound == 1) {
        printf("  → only 1 row found (Z[%d]). Trying alternate encoding...\n",
               found[0]);
        stride = 0;
    } else {
        printf("  → no rows contain data! FMA64 encoding may be wrong.\n");
        printf("    Trying no_accum=0 with pre-zeroed Z...\n");

        /* Retry: zero Z, then accumulate (no_accum=0) */
        AMX_SET();
        AMX_LDX(ld_xy_op(x, 0));
        AMX_LDY(ld_xy_op(y, 0));
        memset(z, 0, 64);
        for (int r = 0; r < 64; r++)
            AMX_LDZ(ld_st_z_op(z, r));
        AMX_FMA64(fma64_outer(0, 0, 0, 0));  /* accumulate mode */

        for (int r = 0; r < 64 && nfound < 8; r++) {
            AMX_STZ(ld_st_z_op(z, r));
            double *zd = (double *)z;
            if (zd[0] != 0.0) {
                printf("  Z[%2d] = [%.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f]\n",
                       r, zd[0], zd[1], zd[2], zd[3], zd[4], zd[5], zd[6], zd[7]);
                found[nfound++] = r;
            }
        }
        AMX_CLR();

        if (nfound >= 2)
            stride = found[1] - found[0];
    }

    /* Validate: each found row j should contain [y[j], 0, 0, ..., 0] */
    if (nfound == 8 && stride > 0) {
        printf("  Validation: ");
        int pass = 1;
        AMX_SET();
        AMX_LDX(ld_xy_op(x, 0));
        AMX_LDY(ld_xy_op(y, 0));
        memset(z, 0, 64);
        for (int r = 0; r < 64; r++)
            AMX_LDZ(ld_st_z_op(z, r));
        AMX_FMA64(fma64_outer(0, 0, 0, 1));

        for (int j = 0; j < 8; j++) {
            AMX_STZ(ld_st_z_op(z, found[j]));
            double *zd = (double *)z;
            double expected = (double)(j + 1);
            if (fabs(zd[0] - expected) > 1e-15 || zd[1] != 0.0) {
                printf("FAIL at j=%d (got %.1f, expected %.1f)\n",
                       j, zd[0], expected);
                pass = 0;
                break;
            }
        }
        if (pass) printf("PASS — Z[j][i] = X[i]*Y[j] confirmed\n");
        AMX_CLR();
    }

    free(x); free(y); free(z);
    return stride;
}
#endif

/* ══════════════════════════════════════════════════════════════
   STEP 2: FMA64 throughput benchmark
   ══════════════════════════════════════════════════════════════ */

#if HAS_AMX
static void bench_fma64_throughput(void) {
    double *x, *y;
    posix_memalign((void **)&x, 128, 64);
    posix_memalign((void **)&y, 128, 64);
    for (int i = 0; i < 8; i++) {
        ((double *)x)[i] = 1.0 + 0.01 * i;
        ((double *)y)[i] = 1.0 - 0.01 * i;
    }

    int stride = g_z_stride;
    /* Use min(stride, 4) independent accumulators to pipeline the 4-cycle latency */
    int n_accum = stride < 4 ? stride : 4;
    if (n_accum < 1) n_accum = 1;

    int iters = 10000000;
    AMX_SET();
    AMX_LDX(ld_xy_op(x, 0));
    AMX_LDY(ld_xy_op(y, 0));

    /* Zero the Z rows we'll use */
    double *zz;
    posix_memalign((void **)&zz, 128, 64);
    memset(zz, 0, 64);
    for (int a = 0; a < n_accum; a++)
        for (int j = 0; j < 8; j++)
            AMX_LDZ(ld_st_z_op(zz, a + j * stride));

    /* Test with varying unroll depths to find peak throughput.
     * 8 is the max number of independent Z accumulators for FP64 (stride=8, 64 rows).
     * Beyond 8, we cycle back to accumulator 0 (safe since the 4-cycle latency
     * has elapsed by then). Deeper unrolling tests instruction issue bandwidth. */
    printf("FMA64 throughput sweep:\n");

    /* 1 accumulator (pure latency-bound) */
    {
        double t0 = now_ns();
        for (int i = 0; i < iters; i++)
            AMX_FMA64(fma64_outer(0, 0, 0, 0));
        double el = now_ns() - t0;
        double ti = (double)iters;
        printf("  %2d instr/iter (%d accum): %.2f ns/instr, %.1f GFLOPS, %.2f cyc\n",
               1, 1, el/ti, 2.0*ti*64/el, el/ti*4.064);
    }
    /* 2 accumulators */
    {
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            AMX_FMA64(fma64_outer(0, 0, 0, 0));
            AMX_FMA64(fma64_outer(0, 0, 1, 0));
        }
        double el = now_ns() - t0;
        double ti = (double)iters * 2;
        printf("  %2d instr/iter (%d accum): %.2f ns/instr, %.1f GFLOPS, %.2f cyc\n",
               2, 2, el/ti, 2.0*ti*64/el, el/ti*4.064);
    }
    /* 4 accumulators */
    {
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            AMX_FMA64(fma64_outer(0, 0, 0, 0));
            AMX_FMA64(fma64_outer(0, 0, 1, 0));
            AMX_FMA64(fma64_outer(0, 0, 2, 0));
            AMX_FMA64(fma64_outer(0, 0, 3, 0));
        }
        double el = now_ns() - t0;
        double ti = (double)iters * 4;
        printf("  %2d instr/iter (%d accum): %.2f ns/instr, %.1f GFLOPS, %.2f cyc\n",
               4, 4, el/ti, 2.0*ti*64/el, el/ti*4.064);
    }
    /* 8 accumulators (max independent) */
    {
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            AMX_FMA64(fma64_outer(0, 0, 0, 0));
            AMX_FMA64(fma64_outer(0, 0, 1, 0));
            AMX_FMA64(fma64_outer(0, 0, 2, 0));
            AMX_FMA64(fma64_outer(0, 0, 3, 0));
            AMX_FMA64(fma64_outer(0, 0, 4, 0));
            AMX_FMA64(fma64_outer(0, 0, 5, 0));
            AMX_FMA64(fma64_outer(0, 0, 6, 0));
            AMX_FMA64(fma64_outer(0, 0, 7, 0));
        }
        double el = now_ns() - t0;
        double ti = (double)iters * 8;
        printf("  %2d instr/iter (%d accum): %.2f ns/instr, %.1f GFLOPS, %.2f cyc\n",
               8, 8, el/ti, 2.0*ti*64/el, el/ti*4.064);
    }
    /* 16 instructions/iter: 2 rounds of 8 accumulators (tests issue bandwidth) */
    {
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            AMX_FMA64(fma64_outer(0, 0, 0, 0));
            AMX_FMA64(fma64_outer(0, 0, 1, 0));
            AMX_FMA64(fma64_outer(0, 0, 2, 0));
            AMX_FMA64(fma64_outer(0, 0, 3, 0));
            AMX_FMA64(fma64_outer(0, 0, 4, 0));
            AMX_FMA64(fma64_outer(0, 0, 5, 0));
            AMX_FMA64(fma64_outer(0, 0, 6, 0));
            AMX_FMA64(fma64_outer(0, 0, 7, 0));
            AMX_FMA64(fma64_outer(0, 0, 0, 0));
            AMX_FMA64(fma64_outer(0, 0, 1, 0));
            AMX_FMA64(fma64_outer(0, 0, 2, 0));
            AMX_FMA64(fma64_outer(0, 0, 3, 0));
            AMX_FMA64(fma64_outer(0, 0, 4, 0));
            AMX_FMA64(fma64_outer(0, 0, 5, 0));
            AMX_FMA64(fma64_outer(0, 0, 6, 0));
            AMX_FMA64(fma64_outer(0, 0, 7, 0));
        }
        double el = now_ns() - t0;
        double ti = (double)iters * 16;
        printf("  %2d instr/iter (%d accum): %.2f ns/instr, %.1f GFLOPS, %.2f cyc\n",
               16, 8, el/ti, 2.0*ti*64/el, el/ti*4.064);
    }
    /* 32 instructions/iter: 4 rounds of 8 accumulators */
    {
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            AMX_FMA64(fma64_outer(0, 0, 0, 0)); AMX_FMA64(fma64_outer(0, 0, 1, 0));
            AMX_FMA64(fma64_outer(0, 0, 2, 0)); AMX_FMA64(fma64_outer(0, 0, 3, 0));
            AMX_FMA64(fma64_outer(0, 0, 4, 0)); AMX_FMA64(fma64_outer(0, 0, 5, 0));
            AMX_FMA64(fma64_outer(0, 0, 6, 0)); AMX_FMA64(fma64_outer(0, 0, 7, 0));
            AMX_FMA64(fma64_outer(0, 0, 0, 0)); AMX_FMA64(fma64_outer(0, 0, 1, 0));
            AMX_FMA64(fma64_outer(0, 0, 2, 0)); AMX_FMA64(fma64_outer(0, 0, 3, 0));
            AMX_FMA64(fma64_outer(0, 0, 4, 0)); AMX_FMA64(fma64_outer(0, 0, 5, 0));
            AMX_FMA64(fma64_outer(0, 0, 6, 0)); AMX_FMA64(fma64_outer(0, 0, 7, 0));
            AMX_FMA64(fma64_outer(0, 0, 0, 0)); AMX_FMA64(fma64_outer(0, 0, 1, 0));
            AMX_FMA64(fma64_outer(0, 0, 2, 0)); AMX_FMA64(fma64_outer(0, 0, 3, 0));
            AMX_FMA64(fma64_outer(0, 0, 4, 0)); AMX_FMA64(fma64_outer(0, 0, 5, 0));
            AMX_FMA64(fma64_outer(0, 0, 6, 0)); AMX_FMA64(fma64_outer(0, 0, 7, 0));
            AMX_FMA64(fma64_outer(0, 0, 0, 0)); AMX_FMA64(fma64_outer(0, 0, 1, 0));
            AMX_FMA64(fma64_outer(0, 0, 2, 0)); AMX_FMA64(fma64_outer(0, 0, 3, 0));
            AMX_FMA64(fma64_outer(0, 0, 4, 0)); AMX_FMA64(fma64_outer(0, 0, 5, 0));
            AMX_FMA64(fma64_outer(0, 0, 6, 0)); AMX_FMA64(fma64_outer(0, 0, 7, 0));
        }
        double el = now_ns() - t0;
        double ti = (double)iters * 32;
        printf("  %2d instr/iter (%d accum): %.2f ns/instr, %.1f GFLOPS, %.2f cyc\n",
               32, 8, el/ti, 2.0*ti*64/el, el/ti*4.064);
    }
    /* 8 instr/iter with different X/Y sources (tests load port contention) */
    {
        AMX_LDX(ld_xy_op(x, 1));  /* also load to X1, Y1 */
        AMX_LDY(ld_xy_op(y, 1));
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            AMX_FMA64(fma64_outer(0, 0, 0, 0));
            AMX_FMA64(fma64_outer(1, 1, 1, 0));
            AMX_FMA64(fma64_outer(0, 0, 2, 0));
            AMX_FMA64(fma64_outer(1, 1, 3, 0));
            AMX_FMA64(fma64_outer(0, 0, 4, 0));
            AMX_FMA64(fma64_outer(1, 1, 5, 0));
            AMX_FMA64(fma64_outer(0, 0, 6, 0));
            AMX_FMA64(fma64_outer(1, 1, 7, 0));
        }
        double el = now_ns() - t0;
        double ti = (double)iters * 8;
        printf("  %2d instr/iter (%d accum, 2 XY): %.2f ns/instr, %.1f GFLOPS, %.2f cyc\n",
               8, 8, el/ti, 2.0*ti*64/el, el/ti*4.064);
    }
    /* 8 accum with interleaved LDX/LDY (realistic poly mul pattern) */
    {
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            AMX_LDX(ld_xy_op(x, 0));
            AMX_LDY(ld_xy_op(y, 0));
            AMX_FMA64(fma64_outer(0, 0, 0, 0));
            AMX_FMA64(fma64_outer(0, 0, 1, 0));
            AMX_FMA64(fma64_outer(0, 0, 2, 0));
            AMX_FMA64(fma64_outer(0, 0, 3, 0));
            AMX_LDX(ld_xy_op(x, 1));
            AMX_LDY(ld_xy_op(y, 1));
            AMX_FMA64(fma64_outer(1, 1, 4, 0));
            AMX_FMA64(fma64_outer(1, 1, 5, 0));
            AMX_FMA64(fma64_outer(1, 1, 6, 0));
            AMX_FMA64(fma64_outer(1, 1, 7, 0));
        }
        double el = now_ns() - t0;
        double ti = (double)iters * 8;  /* 8 FMA64 per iter */
        printf("  %2d FMA+4 LD/iter (realistic): %.2f ns/instr, %.1f GFLOPS, %.2f cyc\n",
               8, el/ti, 2.0*ti*64/el, el/ti*4.064);
    }
    AMX_CLR();

    free(x); free(y); free(zz);
}
#endif

/* ══════════════════════════════════════════════════════════════
   Scalar schoolbook polynomial multiply (baseline)
   ══════════════════════════════════════════════════════════════ */

static void polymul_scalar(const double *restrict a, int na,
                           const double *restrict b, int nb,
                           double *restrict c, int nc) {
    memset(c, 0, nc * sizeof(double));
    for (int i = 0; i < na && i < nc; i++) {
        double ai = a[i];
        if (ai == 0.0) continue;
        int jmax = nb;
        if (i + jmax > nc) jmax = nc - i;
        double *restrict ci = c + i;
        for (int j = 0; j < jmax; j++)
            ci[j] += ai * b[j];
    }
}

/* ══════════════════════════════════════════════════════════════
   AMX schoolbook polynomial multiply
   (lazy block-column accumulation, store + scalar extraction)

   Algorithm: IACR paper Section 4.3 adapted for FP64 (8x8 tiles).
   For each block-column col:
     1. Accumulate all outer products a_p * b_q^T with p+q = col
     2. Store the 8x8 accumulated Z tile to memory
     3. Extract anti-diagonal sums → output coefficients

   This is a FIRST VERSION using scalar extraction. A fully-optimized
   version would use AMX VECFP + EXTRH for in-register extraction
   (see the IACR paper's Algorithm 4.1 for the int16 equivalent).
   ══════════════════════════════════════════════════════════════ */

#if HAS_AMX

/* Store 8 Z rows of an FP64 tile to a flat buffer[64] */
static inline void store_tile(double *buf, int z_base, int stride) {
    for (int j = 0; j < 8; j++)
        AMX_STZ(ld_st_z_op(buf + j * 8, z_base + j * stride));
}

/* Extract anti-diagonal sums from 8x8 FP64 tile and accumulate to c[].
 * tile[j*8 + i] = a[base_a + i] * b[base_b + j].
 * Anti-diagonal d (d=0..14): c[base + d] += sum_{i+j=d} tile[j*8+i]. */
static inline void extract_antidiag(const double *tile, double *c,
                                     int base, int nc) {
    for (int d = 0; d < 15; d++) {
        int idx = base + d;
        if (idx >= nc) break;
        double sum = 0;
        int jlo = d > 7 ? d - 7 : 0;
        int jhi = d < 8 ? d : 7;
        for (int j = jlo; j <= jhi; j++)
            sum += tile[j * 8 + (d - j)];
        c[idx] += sum;
    }
}

/*
 * polymul_amx: c[0..nc-1] = (a * b)[0..nc-1]
 *
 * Pre-conditions:
 *   - AMX_SET() already called
 *   - ap: a[] zero-padded to multiple of 8, 128-byte aligned
 *   - bp: b[] zero-padded to multiple of 8, 128-byte aligned
 *   - tile: 64-double scratch buffer, 128-byte aligned
 */
static void polymul_amx_inner(const double *ap, int nxa,
                               const double *bp, int nyb,
                               double *restrict c, int nc,
                               double *tile) {
    int stride = g_z_stride;
    int max_col = nxa + nyb - 2;

    for (int col = 0; col <= max_col; col++) {
        int base = 8 * col;
        if (base >= nc) break;

        int first = 1;
        for (int p = (col < nxa ? col : nxa - 1); p >= 0; p--) {
            int q = col - p;
            if (q < 0 || q >= nyb) continue;

            AMX_LDX(ld_xy_op(ap + p * 8, 0));
            AMX_LDY(ld_xy_op(bp + q * 8, 0));
            AMX_FMA64(fma64_outer(0, 0, 0, first));  /* first=1: overwrite; 0: accumulate */
            first = 0;
        }

        if (first) continue;

        store_tile(tile, 0, stride);
        extract_antidiag(tile, c, base, nc);
    }
}
#endif

/* ══════════════════════════════════════════════════════════════
   Benchmark driver
   ══════════════════════════════════════════════════════════════ */

static int check_poly(const double *ref, const double *test, int n, const char *name) {
    double maxerr = 0;
    int worst = -1;
    for (int i = 0; i < n; i++) {
        double err = fabs(ref[i] - test[i]);
        if (err > maxerr) { maxerr = err; worst = i; }
    }
    if (maxerr > 1e-8) {
        printf("  %-10s FAIL  max_abs=%.2e at c[%d] (ref=%.6e got=%.6e)\n",
               name, maxerr, worst, ref[worst], test[worst]);
        return 0;
    }
    printf("  %-10s PASS  max_err=%.2e\n", name, maxerr);
    return 1;
}

static void bench_degree(int deg) {
    int na = deg, nb = deg;
    int nc = 2 * deg - 1;

    double *a, *b, *c_ref;
    posix_memalign((void **)&a, 128, na * sizeof(double));
    posix_memalign((void **)&b, 128, nb * sizeof(double));
    posix_memalign((void **)&c_ref, 128, nc * sizeof(double));

    /* Initialize with realistic ICM-like values (probabilities in [0.3, 0.7]) */
    srand(42 + deg);
    for (int i = 0; i < na; i++) a[i] = 0.3 + 0.4 * ((double)rand() / RAND_MAX);
    for (int i = 0; i < nb; i++) b[i] = 0.3 + 0.4 * ((double)rand() / RAND_MAX);

    /* Scalar reference */
    polymul_scalar(a, na, b, nb, c_ref, nc);

    /* Determine iteration count based on degree */
    int iters = 1000000;
    if (deg >= 64) iters = 100000;
    if (deg >= 256) iters = 10000;
    int warmup = iters / 10;
    if (warmup < 100) warmup = 100;

    /* --- Scalar timing --- */
    for (int i = 0; i < warmup; i++)
        polymul_scalar(a, na, b, nb, c_ref, nc);

    double t0 = now_ns();
    for (int i = 0; i < iters; i++)
        polymul_scalar(a, na, b, nb, c_ref, nc);
    double scalar_ns = (now_ns() - t0) / iters;

    printf("degree=%d (%d output coefficients):\n", deg, nc);
    printf("  Scalar:  %8.1f ns  (%d FMAs, %.1f FMAs/ns)\n",
           scalar_ns, na * nb, (double)(na * nb) / scalar_ns);

#if HAS_AMX
    /* --- AMX setup --- */
    int na8 = (na + 7) & ~7, nb8 = (nb + 7) & ~7;
    int nxa = na8 / 8, nyb = nb8 / 8;

    double *ap, *bp, *tile, *c_amx;
    posix_memalign((void **)&ap, 128, na8 * sizeof(double));
    posix_memalign((void **)&bp, 128, nb8 * sizeof(double));
    posix_memalign((void **)&tile, 128, 64 * sizeof(double));
    posix_memalign((void **)&c_amx, 128, nc * sizeof(double));

    memset(ap, 0, na8 * sizeof(double));
    memset(bp, 0, nb8 * sizeof(double));
    memcpy(ap, a, na * sizeof(double));
    memcpy(bp, b, nb * sizeof(double));

    /* Correctness check */
    AMX_SET();
    memset(c_amx, 0, nc * sizeof(double));
    polymul_amx_inner(ap, nxa, bp, nyb, c_amx, nc, tile);
    AMX_CLR();
    check_poly(c_ref, c_amx, nc, "AMX");

    /* AMX timing (AMX_SET/CLR outside timing loop) */
    AMX_SET();

    for (int i = 0; i < warmup; i++) {
        memset(c_amx, 0, nc * sizeof(double));
        polymul_amx_inner(ap, nxa, bp, nyb, c_amx, nc, tile);
    }

    t0 = now_ns();
    for (int i = 0; i < iters; i++) {
        memset(c_amx, 0, nc * sizeof(double));
        polymul_amx_inner(ap, nxa, bp, nyb, c_amx, nc, tile);
    }
    double amx_ns = (now_ns() - t0) / iters;

    AMX_CLR();

    int n_tiles = nxa * nyb;
    int n_cols = nxa + nyb - 1;
    printf("  AMX:     %8.1f ns  (%d tiles, %d extractions, %.2fx vs scalar)\n",
           amx_ns, n_tiles, n_cols, scalar_ns / amx_ns);

    free(ap); free(bp); free(tile); free(c_amx);
#endif

    free(a); free(b); free(c_ref);
}

int main(void) {
    printf("══════════════════════════════════════════════════\n");
    printf("  AMX FP64 Polynomial Multiply Microbenchmark\n");
    printf("══════════════════════════════════════════════════\n\n");

#if !HAS_AMX
    printf("AMX not available (requires Apple Silicon aarch64).\n");
    printf("Running scalar benchmarks only.\n\n");
    int degrees[] = {8, 16, 32, 64, 128, 256};
    for (int i = 0; i < 6; i++) {
        bench_degree(degrees[i]);
        printf("\n");
    }
    return 0;
#else
    /* Step 1: Probe FMA64 Z layout */
    printf("── Step 1: Probe FMA64 Z Row Layout ──\n\n");
    g_z_stride = probe_fma64();
    if (g_z_stride == 0) {
        printf("\nFATAL: Could not determine FMA64 Z row layout.\n");
        printf("The FMA64 operand encoding may need adjustment.\n");
        printf("Try adjusting fma64_outer() bit field positions.\n");
        return 1;
    }
    printf("\n");

    /* Step 2: Raw FMA64 throughput */
    printf("── Step 2: FMA64 Raw Throughput ──\n\n");
    bench_fma64_throughput();
    printf("\n");

    /* Step 3: Polynomial multiply benchmarks */
    printf("── Step 3: Polynomial Multiply Benchmarks ──\n");
    printf("   (AMX = lazy block-column with scalar extraction)\n\n");

    int degrees[] = {8, 16, 32, 64, 128, 256};
    for (int i = 0; i < 6; i++) {
        bench_degree(degrees[i]);
        printf("\n");
    }

    printf("── Notes ──\n");
    printf("  - AMX version uses store + scalar anti-diagonal extraction.\n");
    printf("  - A fully-optimized version using AMX VECFP + EXTRH for\n");
    printf("    in-register extraction would be significantly faster,\n");
    printf("    especially at larger degrees (see IACR paper Sec. 4.2).\n");
    printf("  - The 'tiles' count is the number of FMA64 outer products.\n");
    printf("  - The 'extractions' count is the number of store+flatten ops.\n");

    return 0;
#endif
}
