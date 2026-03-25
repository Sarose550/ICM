/*
 * bench_karatsuba.c — Compare schoolbook vs Karatsuba vs FFT polynomial multiply.
 *
 * Tests sizes 16, 32, 64, 128, 256, 512, 1024 to find if there's a regime
 * between schoolbook and FFT where Karatsuba wins on Zen 4 (AVX-512).
 *
 * Run on a dedicated core: taskset -c 2 ./bench_karatsuba
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ── Schoolbook O(n^2) ── */
static void polymul_school(const double *a, int na, const double *b, int nb, double *c) {
    int nc = na + nb - 1;
    memset(c, 0, nc * sizeof(double));
    for (int i = 0; i < na; i++)
        for (int j = 0; j < nb; j++)
            c[i+j] += a[i] * b[j];
}

/* ── Karatsuba recursive ── */
#define KARATSUBA_BASE 32

static void polymul_karatsuba(const double *a, const double *b, int n, double *c, double *tmp) {
    if (n <= KARATSUBA_BASE) {
        memset(c, 0, (2*n - 1) * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                c[i+j] += a[i] * b[j];
        return;
    }

    int h = n / 2;
    int full = n;
    const double *a0 = a, *a1 = a + h;
    const double *b0 = b, *b1 = b + h;
    int h1 = full - h;

    double *z0 = c;
    double *z2 = tmp;
    double *z1_tmp = tmp + (2*h1 - 1);
    double *a_sum = z1_tmp + (2*h1 - 1);
    double *b_sum = a_sum + h1;
    double *sub_tmp = b_sum + h1;

    /* z0 = a0 * b0 */
    polymul_karatsuba(a0, b0, h, z0, sub_tmp);
    int z0_len = 2*h - 1;

    /* z2 = a1 * b1 */
    polymul_karatsuba(a1, b1, h1, z2, sub_tmp);
    int z2_len = 2*h1 - 1;

    /* a_sum = a0 + a1, b_sum = b0 + b1 (zero-padded to h1) */
    for (int i = 0; i < h; i++) { a_sum[i] = a0[i] + a1[i]; b_sum[i] = b0[i] + b1[i]; }
    for (int i = h; i < h1; i++) { a_sum[i] = a1[i]; b_sum[i] = b1[i]; }

    /* z1_tmp = (a0+a1)*(b0+b1) */
    polymul_karatsuba(a_sum, b_sum, h1, z1_tmp, sub_tmp);
    int z1_len = 2*h1 - 1;

    /* z1 = z1_tmp - z0 - z2 */
    for (int i = 0; i < z0_len && i < z1_len; i++) z1_tmp[i] -= z0[i];
    for (int i = 0; i < z2_len && i < z1_len; i++) z1_tmp[i] -= z2[i];

    /* c = z0 + z1*x^h + z2*x^(2h), but z0 is already in c[0..z0_len-1] */
    int c_len = 2*full - 1;
    /* Extend c with zeros past z0 */
    for (int i = z0_len; i < c_len; i++) c[i] = 0;
    /* Add z1 shifted by h */
    for (int i = 0; i < z1_len; i++) c[h + i] += z1_tmp[i];
    /* Add z2 shifted by 2h */
    for (int i = 0; i < z2_len; i++) c[2*h + i] += z2[i];
}

/* ── FFT-based multiply using FFTW ── */
static void polymul_fft(const double *a, int na, const double *b, int nb, double *c,
                        int fft_n, fftw_plan fwd, fftw_plan inv,
                        double *rbuf_a, fftw_complex *cbuf_a,
                        double *rbuf_b, fftw_complex *cbuf_b) {
    int nc = na + nb - 1;
    int cn = fft_n / 2 + 1;

    memset(rbuf_a, 0, fft_n * sizeof(double));
    memset(rbuf_b, 0, fft_n * sizeof(double));
    memcpy(rbuf_a, a, na * sizeof(double));
    memcpy(rbuf_b, b, nb * sizeof(double));

    fftw_execute_dft_r2c(fwd, rbuf_a, cbuf_a);
    fftw_execute_dft_r2c(fwd, rbuf_b, cbuf_b);

    double scale = 1.0 / fft_n;
    for (int i = 0; i < cn; i++) {
        double re = cbuf_a[i][0]*cbuf_b[i][0] - cbuf_a[i][1]*cbuf_b[i][1];
        double im = cbuf_a[i][0]*cbuf_b[i][1] + cbuf_a[i][1]*cbuf_b[i][0];
        cbuf_a[i][0] = re * scale;
        cbuf_a[i][1] = im * scale;
    }

    fftw_execute_dft_c2r(inv, cbuf_a, rbuf_a);
    memcpy(c, rbuf_a, nc * sizeof(double));
}

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

int main(void) {
    printf("=== KARATSUBA vs SCHOOLBOOK vs FFT BENCHMARK (Zen 4 AVX-512) ===\n");
    printf("KARATSUBA_BASE = %d\n\n", KARATSUBA_BASE);

    /* Try to load wisdom if available */
    fftw_import_wisdom_from_filename("fftw_wisdom.dat");

    int sizes[] = {16, 32, 64, 128, 256, 512, 1024};
    int n_sizes = 7;

    printf("%-6s  %12s  %12s  %12s  %8s  %8s  %s\n",
           "n", "schoolbook", "karatsuba", "fft", "K/S", "F/S", "winner");
    printf("%-6s  %12s  %12s  %12s  %8s  %8s  %s\n",
           "", "(ns)", "(ns)", "(ns)", "ratio", "ratio", "");

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        double *a = malloc(n * sizeof(double));
        double *b = malloc(n * sizeof(double));
        double *c_s = malloc((2*n) * sizeof(double));
        double *c_k = malloc((2*n) * sizeof(double));
        double *c_f = malloc((2*n) * sizeof(double));
        /* Karatsuba needs ~4n scratch space */
        double *k_tmp = malloc(16 * n * sizeof(double));

        for (int i = 0; i < n; i++) {
            a[i] = 1.0 + 0.001 * i;
            b[i] = 1.0 - 0.001 * i;
        }

        /* FFT setup */
        int fft_n = next_pow2(2 * n);
        double *rbuf_a = fftw_malloc(fft_n * sizeof(double));
        fftw_complex *cbuf_a = fftw_malloc((fft_n/2+1) * sizeof(fftw_complex));
        double *rbuf_b = fftw_malloc(fft_n * sizeof(double));
        fftw_complex *cbuf_b = fftw_malloc((fft_n/2+1) * sizeof(fftw_complex));
        memset(rbuf_a, 0, fft_n * sizeof(double));

        fftw_plan fwd = fftw_plan_dft_r2c_1d(fft_n, rbuf_a, cbuf_a,
                            FFTW_MEASURE | FFTW_WISDOM_ONLY);
        fftw_plan inv = fftw_plan_dft_c2r_1d(fft_n, cbuf_a, rbuf_a,
                            FFTW_MEASURE | FFTW_WISDOM_ONLY);
        if (!fwd || !inv) {
            if (fwd) fftw_destroy_plan(fwd);
            if (inv) fftw_destroy_plan(inv);
            fwd = fftw_plan_dft_r2c_1d(fft_n, rbuf_a, cbuf_a, FFTW_MEASURE);
            inv = fftw_plan_dft_c2r_1d(fft_n, cbuf_a, rbuf_a, FFTW_MEASURE);
        }

        /* Determine reps */
        int reps = (int)(5e8 / ((double)n * n));
        if (reps < 20) reps = 20;
        if (reps > 200000) reps = 200000;
        int reps_fft = reps;
        if (n >= 512) reps_fft = reps / 2;

        /* Warmup */
        for (int r = 0; r < 5; r++) {
            polymul_school(a, n, b, n, c_s);
            polymul_karatsuba(a, b, n, c_k, k_tmp);
            polymul_fft(a, n, b, n, c_f, fft_n, fwd, inv, rbuf_a, cbuf_a, rbuf_b, cbuf_b);
        }

        /* Correctness check */
        double max_err_k = 0, max_err_f = 0;
        for (int i = 0; i < 2*n - 1; i++) {
            double ek = fabs(c_k[i] - c_s[i]);
            double ef = fabs(c_f[i] - c_s[i]);
            if (ek > max_err_k) max_err_k = ek;
            if (ef > max_err_f) max_err_f = ef;
        }

        /* Benchmark schoolbook */
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            polymul_school(a, n, b, n, c_s);
        double school_ns = (now_ns() - t0) / reps;

        /* Benchmark Karatsuba */
        t0 = now_ns();
        for (int r = 0; r < reps; r++)
            polymul_karatsuba(a, b, n, c_k, k_tmp);
        double kara_ns = (now_ns() - t0) / reps;

        /* Benchmark FFT */
        t0 = now_ns();
        for (int r = 0; r < reps_fft; r++)
            polymul_fft(a, n, b, n, c_f, fft_n, fwd, inv, rbuf_a, cbuf_a, rbuf_b, cbuf_b);
        double fft_ns = (now_ns() - t0) / reps_fft;

        const char *winner = "school";
        double best = school_ns;
        if (kara_ns < best) { best = kara_ns; winner = "KARA"; }
        if (fft_ns < best) { best = fft_ns; winner = "FFT"; }

        printf("%-6d  %10.0f ns  %10.0f ns  %10.0f ns  %7.2fx  %7.2fx  %-8s  (err: K=%.1e F=%.1e)\n",
               n, school_ns, kara_ns, fft_ns,
               kara_ns / school_ns, fft_ns / school_ns,
               winner, max_err_k, max_err_f);

        fftw_destroy_plan(fwd);
        fftw_destroy_plan(inv);
        fftw_free(rbuf_a); fftw_free(cbuf_a);
        fftw_free(rbuf_b); fftw_free(cbuf_b);
        free(a); free(b); free(c_s); free(c_k); free(c_f); free(k_tmp);
    }

    printf("\nNote: K/S < 1.0 means Karatsuba wins; F/S < 1.0 means FFT wins.\n");
    printf("Prior from M3 Max: Karatsuba was 1.5x slower at all sizes.\n");
    printf("On Zen 4, wider SIMD inflates schoolbook regime further.\n");
    return 0;
}
