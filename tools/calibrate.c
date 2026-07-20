/*
 * calibrate.c — Generate fft_config.h for the current machine.
 *
 * Produces:
 *   1. fftw_wisdom.dat  — FFTW PATIENT plans for all 7-smooth sizes up to MAX_SIZE
 *   2. fft_config.h     — C header with calib_sizes[], calib_times_ns[], and
 *                          best_fft_config() / best_fft_config_joint() functions
 *
 * The generated fft_config.h should be placed in devices/<DEVICE>/fft_config.h.
 *
 * Usage:
 *   gcc -O3 -march=native -o calibrate tools/calibrate.c -lfftw3 -lm
 *   ./calibrate                   # full calibration (may take 10-30 minutes)
 *   ./calibrate --wisdom-only     # only generate wisdom (skip timing)
 *   ./calibrate --quick           # fewer reps (faster, less accurate)
 *
 * On Linux, pin to one core for stable results:
 *   taskset -c 0 nice -20 ./calibrate
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#define MAX_SIZE 131072
#define WISDOM_FILE "fftw_wisdom.dat"

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ── Generate all 7-smooth numbers up to MAX_SIZE ── */

static int smooth_nums[800];
static int n_smooth = 0;

static void build_smooth_table(void) {
    for (int a = 1; a <= MAX_SIZE; a *= 2)
        for (int b = a; b <= MAX_SIZE; b *= 3)
            for (int c = b; c <= MAX_SIZE; c *= 5)
                for (int d = c; d <= MAX_SIZE; d *= 7)
                    smooth_nums[n_smooth++] = d;
    /* Sort */
    for (int i = 1; i < n_smooth; i++) {
        int key = smooth_nums[i], j = i - 1;
        while (j >= 0 && smooth_nums[j] > key) { smooth_nums[j+1] = smooth_nums[j]; j--; }
        smooth_nums[j+1] = key;
    }
}

/* ── Phase 1: Generate FFTW PATIENT wisdom ── */

static void generate_wisdom(void) {
    printf("Phase 1: Generating FFTW PATIENT wisdom for %d smooth sizes...\n", n_smooth);
    fftw_import_wisdom_from_filename(WISDOM_FILE);

    for (int i = 0; i < n_smooth; i++) {
        int sz = smooth_nums[i];
        if (sz < 2) continue;

        double *rbuf = fftw_malloc(sz * sizeof(double));
        fftw_complex *cbuf = fftw_malloc((sz/2 + 1) * sizeof(fftw_complex));
        memset(rbuf, 0, sz * sizeof(double));

        fftw_plan fwd = fftw_plan_dft_r2c_1d(sz, rbuf, cbuf, FFTW_PATIENT);
        fftw_plan inv = fftw_plan_dft_c2r_1d(sz, cbuf, rbuf, FFTW_PATIENT);

        if (fwd) fftw_destroy_plan(fwd);
        if (inv) fftw_destroy_plan(inv);
        fftw_free(rbuf);
        fftw_free(cbuf);

        if ((i + 1) % 50 == 0 || i == n_smooth - 1) {
            printf("  %d/%d (size=%d)\n", i + 1, n_smooth, sz);
            fftw_export_wisdom_to_filename(WISDOM_FILE);
        }
    }
    fftw_export_wisdom_to_filename(WISDOM_FILE);
    printf("  Wisdom saved to %s\n\n", WISDOM_FILE);
}

/* ── Phase 2: Benchmark each size ── */

static double calib_times[800];

/* ── Phase 2.5: Measure streaming bandwidth at each cache level ── */

static double bw_l2_gbs, bw_l3_gbs, bw_dram_gbs;

static double measure_bw(size_t bytes) {
    size_t n = bytes / sizeof(double);
    double *a = (double *)malloc(bytes);
    double *b = (double *)malloc(bytes);
    double *c = (double *)malloc(bytes);
    if (!a || !b || !c) { fprintf(stderr, "malloc failed in measure_bw\n"); exit(1); }

    for (size_t i = 0; i < n; i++) { a[i] = 0; b[i] = 1.0 + 0.001*i; c[i] = 2.0 - 0.001*i; }
    double s = 0.42;

    for (int r = 0; r < 3; r++)
        for (size_t i = 0; i < n; i++) a[i] = b[i] * s + c[i];

    int reps = (int)(2e8 / (double)(n + 1));
    if (reps < 10) reps = 10;
    if (reps > 100000) reps = 100000;

    double t0 = now_ns();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < n; i++) a[i] = b[i] * s + c[i];
    double elapsed_ns = now_ns() - t0;

    volatile double sink = a[n/2];
    (void)sink;
    free(a); free(b); free(c);

    /* 24 bytes per element: read b (8) + read c (8) + write a (8) */
    double total_bytes = 24.0 * (double)n * reps;
    return total_bytes / elapsed_ns;  /* GB/s (bytes/ns = GB/s) */
}

static void benchmark_bandwidth(void) {
    printf("Phase 2.5: Measuring streaming bandwidth at each cache level...\n");
    bw_l2_gbs = measure_bw(512 * 1024);
    printf("  L2 (512KB):   %.1f GB/s\n", bw_l2_gbs);
    bw_l3_gbs = measure_bw(16 * 1024 * 1024);
    printf("  L3 (16MB):    %.1f GB/s\n", bw_l3_gbs);
    bw_dram_gbs = measure_bw(256 * 1024 * 1024);
    printf("  DRAM (256MB):  %.1f GB/s\n\n", bw_dram_gbs);
}

static void benchmark_sizes(int quick) {
    printf("Phase 2: Benchmarking FFT pipeline at each size...\n");
    fftw_import_wisdom_from_filename(WISDOM_FILE);

    for (int i = 0; i < n_smooth; i++) {
        int sz = smooth_nums[i];
        if (sz < 1) { calib_times[i] = 0; continue; }

        double *rbuf = fftw_malloc(sz * sizeof(double));
        fftw_complex *cbuf = fftw_malloc((sz/2 + 1) * sizeof(fftw_complex));
        double *rbuf2 = fftw_malloc(sz * sizeof(double));
        fftw_complex *cbuf2 = fftw_malloc((sz/2 + 1) * sizeof(fftw_complex));
        memset(rbuf, 0, sz * sizeof(double));
        memset(rbuf2, 0, sz * sizeof(double));

        /* Create MEASURE plans (from PATIENT wisdom — instant) */
        fftw_plan fwd = fftw_plan_dft_r2c_1d(sz, rbuf, cbuf, FFTW_MEASURE | FFTW_WISDOM_ONLY);
        fftw_plan inv = fftw_plan_dft_c2r_1d(sz, cbuf, rbuf, FFTW_MEASURE | FFTW_WISDOM_ONLY);
        if (!fwd || !inv) {
            if (fwd) fftw_destroy_plan(fwd);
            if (inv) fftw_destroy_plan(inv);
            fwd = fftw_plan_dft_r2c_1d(sz, rbuf, cbuf, FFTW_ESTIMATE);
            inv = fftw_plan_dft_c2r_1d(sz, cbuf, rbuf, FFTW_ESTIMATE);
        }

        /* Fill with test data (two input polynomials of degree sz-1) */
        for (int j = 0; j < sz; j++) rbuf[j] = 1.0 + 0.001 * j;
        for (int j = 0; j < sz; j++) rbuf2[j] = 1.0 + 0.002 * j;

        /* Determine rep count: target ~100ms per size */
        int reps = (int)(1e8 / (double)(sz + 1));
        if (quick) reps /= 10;
        if (reps < 100) reps = 100;
        if (reps > 2000000) reps = 2000000;

        int cn = sz / 2 + 1;
        double inv_n = 1.0 / sz;

        /* Full pipeline = memcpy_in + fwd(a) + fwd(b) + pointwise + ifft + scale.
         * This matches what polymul_fft_wrap actually does per parent in the tree.
         * Measured warm (second pass) — matches 255/256 Q-points. */

        /* Warm up (plan + µop cache) */
        for (int r = 0; r < 5; r++) {
            memcpy(rbuf, rbuf2, sz * sizeof(double));
            fftw_execute(fwd);
            fftw_execute_dft_r2c(fwd, rbuf2, cbuf2);
            for (int j = 0; j < cn; j++) {
                double re = cbuf[j][0]*cbuf2[j][0] - cbuf[j][1]*cbuf2[j][1];
                double im = cbuf[j][0]*cbuf2[j][1] + cbuf[j][1]*cbuf2[j][0];
                cbuf[j][0] = re; cbuf[j][1] = im;
            }
            fftw_execute(inv);
            for (int j = 0; j < sz; j++) rbuf[j] *= inv_n;
        }

        /* Time: full polymul pipeline (memcpy + 2×fwd + pointwise + inv + scale) */
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memset(rbuf, 0, sz * sizeof(double));
            rbuf[0] = 1.0 + 0.001 * r;
            fftw_execute(fwd);
            memcpy(rbuf2, rbuf, sz * sizeof(double));
            rbuf2[0] = 1.0 + 0.002 * r;
            fftw_execute_dft_r2c(fwd, rbuf2, cbuf2);
            for (int j = 0; j < cn; j++) {
                double re = cbuf[j][0]*cbuf2[j][0] - cbuf[j][1]*cbuf2[j][1];
                double im = cbuf[j][0]*cbuf2[j][1] + cbuf[j][1]*cbuf2[j][0];
                cbuf[j][0] = re; cbuf[j][1] = im;
            }
            fftw_execute(inv);
            for (int j = 0; j < sz; j++) rbuf[j] *= inv_n;
        }
        calib_times[i] = (now_ns() - t0) / reps;

        fftw_destroy_plan(fwd);
        fftw_destroy_plan(inv);
        fftw_free(rbuf); fftw_free(cbuf);
        fftw_free(rbuf2); fftw_free(cbuf2);

        if ((i + 1) % 50 == 0 || i == n_smooth - 1)
            printf("  %d/%d (size=%d, %.0f ns)\n", i + 1, n_smooth, sz, calib_times[i]);
    }
    printf("\n");
}

/* ── Phase 3: Write fft_config.h ── */

static void write_config(const char *filename) {
    printf("Phase 3: Writing %s...\n", filename);
    FILE *f = fopen(filename, "w");
    if (!f) { perror(filename); exit(1); }

    fprintf(f, "/* Auto-generated FFT configuration from calibrate */\n");
    fprintf(f, "/* Generated on this machine — do not use on different hardware */\n\n");

    /* calib_sizes[] */
    fprintf(f, "#define N_CALIBRATED_SIZES %d\n", n_smooth);
    fprintf(f, "static const int calib_sizes[N_CALIBRATED_SIZES] = {\n   ");
    for (int i = 0; i < n_smooth; i++) {
        fprintf(f, "%d", smooth_nums[i]);
        if (i < n_smooth - 1) fprintf(f, ",");
        if ((i + 1) % 20 == 0 && i < n_smooth - 1) fprintf(f, "\n   ");
    }
    fprintf(f, "\n};\n\n");

    /* calib_times_ns[] */
    fprintf(f, "static const double calib_times_ns[N_CALIBRATED_SIZES] = {\n   ");
    for (int i = 0; i < n_smooth; i++) {
        fprintf(f, "%.1f", calib_times[i]);
        if (i < n_smooth - 1) fprintf(f, ",");
        if ((i + 1) % 10 == 0 && i < n_smooth - 1) fprintf(f, "\n   ");
    }
    fprintf(f, "\n};\n\n");

    /* ── Device constants (all #ifndef guarded for manual override) ── */
    fprintf(f,
"/* ── Device constants ── */\n"
"/* calib_times_ns now measures the full polymul_fft_wrap pipeline\n"
" * (memcpy + 2×FFT + pointwise + scale), so FFT_OVERHEAD_NS = 0.\n"
" * Wrap correction is modeled separately with WRAP_FMA_NS. */\n"
"#ifndef FMA_NS\n"
"#define FMA_NS 0.25  /* ns per scalar FMA — re-measure via ./bench_grid profile */\n"
"#endif\n"
"#ifndef FFT_OVERHEAD_NS\n"
"#define FFT_OVERHEAD_NS 0.0  /* baked into calib_times_ns (full pipeline) */\n"
"#endif\n"
"#ifndef WRAP_FMA_NS\n"
"#define WRAP_FMA_NS 4.0  /* ns per FMA in wrap correction (memory-latency-bound) */\n"
"#endif\n"
"#ifndef PAIRED_CACHED_CORR_RATIO\n"
"#define PAIRED_CACHED_CORR_RATIO 1.03  /* paired cached correlate / full pipeline */\n"
"#endif\n"
"#ifndef INDEP_PAIR_RATIO\n"
"#define INDEP_PAIR_RATIO 1.25  /* correlate_fft_pair / full pipeline */\n"
"#endif\n"
"/* Hybrid-engine block/leaf constants — placeholders until\n"
" * tools/fit_cost_model.py --write overwrites them with a real fit. */\n"
"#ifndef FP64_DIV_NS\n"
"#define FP64_DIV_NS 10.0  /* ns per FP64 division — re-fit via fit_cost_model.py */\n"
"#endif\n"
"#ifndef LEAF_FMA_NS\n"
"#define LEAF_FMA_NS 0.25  /* ns per FMA in leaf blocks — re-fit via fit_cost_model.py */\n"
"#endif\n"
"#ifndef LEAF_BLOCK_NS\n"
"#define LEAF_BLOCK_NS 100.0  /* ns per leaf block overhead — re-fit via fit_cost_model.py */\n"
"#endif\n"
"#ifndef BLOCK_FMA_NS\n"
"#define BLOCK_FMA_NS 0.05  /* ns per FMA in block build — re-fit via fit_cost_model.py */\n"
"#endif\n"
"#ifndef BLOCK_MEM_NS\n"
"#define BLOCK_MEM_NS 0.1  /* ns per block-build memory op — re-fit via fit_cost_model.py */\n"
"#endif\n\n");

    /* Cache and bandwidth constants */
    fprintf(f,
"/* ── Cache hierarchy ── */\n"
"#ifndef L2_CACHE_SIZE\n"
"#define L2_CACHE_SIZE 1048576  /* per-core L2 in bytes — update for this hardware */\n"
"#endif\n"
"#ifndef L3_CACHE_SIZE\n"
"#define L3_CACHE_SIZE 33554432  /* shared L3 in bytes — update for this hardware */\n"
"#endif\n\n");

    /* Bandwidth constants from measurement */
    fprintf(f,
"/* ── Streaming bandwidth (measured by calibrate) ── */\n"
"#ifndef L2_BW_GBS\n"
"#define L2_BW_GBS %.1f\n"
"#endif\n"
"#ifndef L3_BW_GBS\n"
"#define L3_BW_GBS %.1f\n"
"#endif\n"
"#ifndef DRAM_BW_GBS\n"
"#define DRAM_BW_GBS %.1f\n"
"#endif\n\n",
        bw_l2_gbs, bw_l3_gbs, bw_dram_gbs);

    /* best_fft_config_joint() — 6-arg version with p_eff for input-wrap cost */
    fprintf(f,
"/* ── Cost model functions ── */\n\n"
"/* Joint optimization of build + paired cached correlate at one shared FFT size.\n"
" * p_eff = build_conv/2 + 1 (polynomial size at this level) for input-wrap cost. */\n"
"static double best_fft_config_joint(int build_conv, int corr_conv, int p_eff,\n"
"                                     int *out_size, int *out_build_m, int *out_corr_m) {\n"
"    int max_conv = (build_conv > corr_conv) ? build_conv : corr_conv;\n"
"    int min_size = max_conv / 2 + 1;\n"
"\n"
"    int lo = 0, hi = N_CALIBRATED_SIZES - 1;\n"
"    int half = min_size;\n"
"    while (lo < hi) { int mid = (lo+hi)>>1; if (calib_sizes[mid] < half) lo = mid+1; else hi = mid; }\n"
"\n"
"    double best_cost = 1e18;\n"
"    *out_size = 0; *out_build_m = 0; *out_corr_m = 0;\n"
"\n"
"    for (int i = lo; i < N_CALIBRATED_SIZES; i++) {\n"
"        int S = calib_sizes[i];\n"
"        if (S > 2 * max_conv) break;\n"
"        if (S < min_size) continue;\n"
"        int mb = (S >= build_conv) ? 0 : build_conv - S;\n"
"        int mc = (S >= corr_conv) ? 0 : corr_conv - S;\n"
"        double cost = calib_times_ns[i]\n"
"                    + (double)mb*(mb+1)/2.0 * FMA_NS\n"
"                    + calib_times_ns[i] * PAIRED_CACHED_CORR_RATIO\n"
"                    + (double)mc*(mc+1) * FMA_NS;\n"
"        if (cost < best_cost) {\n"
"            best_cost = cost;\n"
"            *out_size = S;\n"
"            *out_build_m = mb;\n"
"            *out_corr_m = mc;\n"
"        }\n"
"    }\n"
"    return best_cost;\n"
"}\n\n");

    /* best_fft_config() — 4-arg version with len_P for input-wrap cost */
    fprintf(f,
"/* For a needed convolution length L, find the fastest FFT size.\n"
" * len_P: polynomial size for input-wrap cost (pass 0 for pure convolution). */\n"
"static void best_fft_config(int L, int *out_size, int *out_wrap_m, int len_P) {\n"
"    int lo = 0, hi = N_CALIBRATED_SIZES - 1;\n"
"    int half_L = L > 1 ? L / 2 : 1;\n"
"    while (lo < hi) { int mid = (lo+hi)>>1; if (calib_sizes[mid] < half_L) lo = mid+1; else hi = mid; }\n"
"\n"
"    double best_cost = 1e18;\n"
"    *out_size = 0; *out_wrap_m = 0;\n"
"\n"
"    int min_size = L / 2 + 1;\n"
"    for (int i = lo; i < N_CALIBRATED_SIZES; i++) {\n"
"        int S = calib_sizes[i];\n"
"        if (S > 2 * L) break;\n"
"        if (S < min_size) continue;\n"
"        int m = (S >= L) ? 0 : L - S;\n"
"        double correction = (len_P > 0) ? (double)m * (m + 1) * FMA_NS\n"
"                                        : (double)m * (m + 1) / 2.0 * FMA_NS;\n"
"        double cost = calib_times_ns[i] + correction;\n"
"        if (cost < best_cost) {\n"
"            best_cost = cost;\n"
"            *out_size = S;\n"
"            *out_wrap_m = m;\n"
"        }\n"
"    }\n"
"}\n");

    fclose(f);
    printf("  Written %s (%d sizes)\n\n", filename, n_smooth);
}

int main(int argc, char **argv) {
    int wisdom_only = 0, quick = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--wisdom-only") == 0) wisdom_only = 1;
        if (strcmp(argv[i], "--quick") == 0) quick = 1;
    }

    build_smooth_table();
    printf("Found %d 7-smooth numbers up to %d\n\n", n_smooth, MAX_SIZE);

    generate_wisdom();
    if (wisdom_only) { printf("Done (wisdom only).\n"); return 0; }

    benchmark_sizes(quick);
    benchmark_bandwidth();
    write_config("fft_config.h");

    printf("Done. Next steps:\n");
    printf("  1. cp fft_config.h devices/<DEVICE>/fft_config.h\n");
    printf("  2. cp fftw_wisdom.dat devices/<DEVICE>/fftw_wisdom.dat\n");
    printf("  3. make DEVICE=<DEVICE> && ./bench_grid verify\n");
    printf("  4. ./bench_grid profile   # measure device constants\n");
    printf("  5. Update #defines in fft_config.h with measured values:\n");
    printf("     FMA_NS, FFT_OVERHEAD_NS, PAIRED_CACHED_CORR_RATIO,\n");
    printf("     INDEP_PAIR_RATIO, L2_CACHE_SIZE, L3_CACHE_SIZE\n");
    printf("  6. ./bench_grid verify && ./bench_grid\n");
    return 0;
}
