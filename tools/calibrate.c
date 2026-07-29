/*
 * calibrate.c: generate fft_config.h for the current machine.
 *
 * Produces:
 *   1. fftw_wisdom.dat:  FFTW PATIENT plans for all 7-smooth sizes up to max_size
 *   2. fft_config.h:     C header with calib_sizes[], calib_times_ns[],
 *                          CALIBRATED_MAX_CONV_LEN, and per-device constants
 *
 * The cost-model *functions* (best_fft_config, best_fft_config_joint) are not
 * generated here; they live once, for all devices, in src/cpu/fft_cost_model.h
 * and consume the data this tool emits.
 *
 * The generated fft_config.h should be placed in devices/<DEVICE>/fft_config.h.
 *
 * Usage:
 *   # macOS (Homebrew FFTW)
 *   gcc -O3 -march=native -I/opt/homebrew/include -o calibrate tools/calibrate.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm
 *   # Linux
 *   gcc -O3 -march=native -o calibrate tools/calibrate.c -lfftw3 -lm
 *
 *   ./calibrate                   # full calibration (may take 10-30 minutes)
 *   ./calibrate --wisdom-only     # only generate wisdom (skip timing)
 *   ./calibrate --quick           # fewer reps (faster, less accurate)
 *   ./calibrate --max-size N      # calibrate up to N (default 131072)
 *
 * --max-size tradeoff: a higher ceiling means a longer offline calibration
 * run (more FFT sizes to benchmark) but gives a larger fully-optimal range
 * before the uncalibrated FFTW_ESTIMATE fallback engages at runtime.
 * The default 131072 keeps calibration under ~30 minutes on modern hardware.
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

static int max_size = 131072;  /* overridable via --max-size N */
#define WISDOM_FILE "fftw_wisdom.dat"

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ── Generate all 7-smooth numbers up to max_size ── */

static int *smooth_nums = NULL;
static int n_smooth = 0;

/* Count 7-smooth numbers ≤ limit without storing them. */
static int count_smooth(int limit) {
    int cnt = 0;
    for (int a = 1; a <= limit; a *= 2)
        for (int b = a; b <= limit; b *= 3)
            for (int c = b; c <= limit; c *= 5)
                for (int d = c; d <= limit; d *= 7)
                    cnt++;
    return cnt;
}

static void build_smooth_table(void) {
    n_smooth = count_smooth(max_size);
    smooth_nums = (int *)malloc((size_t)n_smooth * sizeof(int));
    if (!smooth_nums) { fprintf(stderr, "malloc smooth_nums failed\n"); exit(1); }
    int idx = 0;
    for (int a = 1; a <= max_size; a *= 2)
        for (int b = a; b <= max_size; b *= 3)
            for (int c = b; c <= max_size; c *= 5)
                for (int d = c; d <= max_size; d *= 7)
                    smooth_nums[idx++] = d;
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

static double *calib_times = NULL;  /* allocated after n_smooth is known */

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

    /* The inner loop body doesn't depend on r, so a[i]=b[i]*s+c[i] computes
     * the identical value on every repetition. Without a barrier, -O3 can
     * (and on Zen4/GCC, does) prove the repeated stores are redundant and
     * collapse the whole outer loop to ~1 real pass while `reps` full
     * passes are still charged in total_bytes below -- inflating the
     * reported bandwidth by ~reps x. The asm volatile forces the compiler
     * to treat memory as externally observed after each pass, so it can't
     * eliminate the "redundant" work. (Confirmed post hoc: dividing the
     * pre-fix Zen4 numbers by their respective reps gives 112/34/32 GB/s
     * for L2/L3/DRAM -- physically plausible -- matching this exactly.) */
    double t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        for (size_t i = 0; i < n; i++) a[i] = b[i] * s + c[i];
        __asm__ __volatile__("" : : "r"(a) : "memory");
    }
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

        /* Create MEASURE plans (from PATIENT wisdom, instant) */
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
         * Measured warm (second pass), matching 255/256 Q-points. */

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
    fprintf(f, "/* Generated on this machine; do not use on different hardware */\n\n");

    /* CALIBRATED_MAX_CONV_LEN: largest convolution length the FFT timing
     * table can honestly cover.  Derived from max(calib_sizes):
     * best_fft_config(L) returns a non-sentinel cost for
     * L ≤ 2 * max(calib_sizes) - 1. */
    int max_calib = smooth_nums[n_smooth - 1];
    int calib_max_conv_len = 2 * max_calib - 1;
    fprintf(f, "/* Largest convolution length the FFT timing table can honestly cover.\n");
    fprintf(f, " * Derived from max(calib_sizes) = %d: best_fft_config(L) returns a\n", max_calib);
    fprintf(f, " * non-sentinel cost for L ≤ 2*%d-1 = %d. */\n", max_calib, calib_max_conv_len);
    fprintf(f, "#ifndef CALIBRATED_MAX_CONV_LEN\n");
    fprintf(f, "#define CALIBRATED_MAX_CONV_LEN %d\n", calib_max_conv_len);
    fprintf(f, "#endif\n\n");

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
"#define FMA_NS 0.25  /* ns per scalar FMA, re-measure via ./bench_grid profile */\n"
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
"/* Hybrid-engine block/leaf constants, placeholders until\n"
" * tools/fit_cost_model.py --write overwrites them with a real fit. */\n"
"#ifndef FP64_DIV_NS\n"
"#define FP64_DIV_NS 10.0  /* ns per FP64 division, re-fit via fit_cost_model.py */\n"
"#endif\n"
"#ifndef LEAF_FMA_NS\n"
"#define LEAF_FMA_NS 0.25  /* ns per FMA in leaf blocks, re-fit via fit_cost_model.py */\n"
"#endif\n"
"#ifndef LEAF_BLOCK_NS\n"
"#define LEAF_BLOCK_NS 100.0  /* ns per leaf block overhead, re-fit via fit_cost_model.py */\n"
"#endif\n"
"#ifndef BLOCK_FMA_NS\n"
"#define BLOCK_FMA_NS 0.05  /* ns per FMA in block build, re-fit via fit_cost_model.py */\n"
"#endif\n"
"#ifndef BLOCK_MEM_NS\n"
"#define BLOCK_MEM_NS 0.1  /* ns per block-build memory op, re-fit via fit_cost_model.py */\n"
"#endif\n\n");

    /* Cache and bandwidth constants */
    fprintf(f,
"/* ── Cache hierarchy ── */\n"
"#ifndef L2_CACHE_SIZE\n"
"#define L2_CACHE_SIZE 1048576  /* per-core L2 in bytes, update for this hardware */\n"
"#endif\n"
"#ifndef L3_CACHE_SIZE\n"
"#define L3_CACHE_SIZE 33554432  /* shared L3 in bytes, update for this hardware */\n"
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

    fclose(f);
    printf("  Written %s (%d sizes)\n\n", filename, n_smooth);
}

int main(int argc, char **argv) {
    int wisdom_only = 0, quick = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--wisdom-only") == 0) wisdom_only = 1;
        if (strcmp(argv[i], "--quick") == 0) quick = 1;
        if (strcmp(argv[i], "--max-size") == 0 && i + 1 < argc) {
            max_size = atoi(argv[++i]);
            if (max_size < 2) { fprintf(stderr, "--max-size must be >= 2\n"); return 1; }
        }
    }

    build_smooth_table();
    calib_times = (double *)calloc((size_t)n_smooth, sizeof(double));
    if (!calib_times) { fprintf(stderr, "calloc calib_times failed\n"); return 1; }
    printf("Found %d 7-smooth numbers up to %d\n\n", n_smooth, max_size);

    generate_wisdom();
    if (wisdom_only) { printf("Done (wisdom only).\n"); free(smooth_nums); free(calib_times); return 0; }

    benchmark_sizes(quick);
    benchmark_bandwidth();
    write_config("fft_config.h");

    int max_calib = smooth_nums[n_smooth - 1];
    int calib_max_conv_len = 2 * max_calib - 1;
    printf("Done. Next steps:\n");
    printf("  1. cp fft_config.h devices/<DEVICE>/fft_config.h\n");
    printf("  2. cp fftw_wisdom.dat devices/<DEVICE>/fftw_wisdom.dat\n");
    printf("  3. make DEVICE=<DEVICE> && ./bench_grid verify\n");
    printf("  4. ./bench_grid profile   # measure device constants\n");
    printf("  5. Update #defines in fft_config.h with measured values:\n");
    printf("     FMA_NS, FFT_OVERHEAD_NS, PAIRED_CACHED_CORR_RATIO,\n");
    printf("     INDEP_PAIR_RATIO, L2_CACHE_SIZE, L3_CACHE_SIZE\n");
    printf("  6. ./bench_grid verify && ./bench_grid\n");
    printf("\n  Calibration ceiling: conv lengths up to %d are fully optimal;\n", calib_max_conv_len);
    printf("  beyond that the uncalibrated FFTW_ESTIMATE fallback engages.\n");
    printf("  To raise the ceiling, re-run with --max-size N (higher N = longer calibration).\n");
    free(smooth_nums);
    free(calib_times);
    return 0;
}
