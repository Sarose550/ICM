/*
 * calibrate_dual.c — Benchmark FFTW vs MKL at each smooth size via dlopen.
 *
 * Both libraries export the same fftw_* symbols, so we dlopen each separately
 * and resolve function pointers. For each 7-smooth size up to 131072, we time
 * the full r2c + pointwise + c2r pipeline with both libraries and record which
 * is faster.
 *
 * Produces: fft_config_dual.h with calib_times_ns[] (best of both) and
 *           calib_lib[] (0=FFTW, 1=MKL) per size.
 *
 * Usage:
 *   gcc -O3 -march=znver4 -o calibrate_dual tools/calibrate_dual.c -ldl -lm
 *   taskset -c 0 nice -20 ./calibrate_dual
 *
 * Requires: libfftw3.so and libmkl_rt.so both loadable via dlopen.
 *   FFTW: system install (apt install libfftw3-dev)
 *   MKL:  Intel oneAPI MKL (libmkl_rt.so)
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>

#define MAX_SIZE 131072
#define WISDOM_FILE "fftw_wisdom.dat"

typedef double fftw_complex_pair[2];

/* Function pointer types matching FFTW3 API */
typedef void *fftw_plan_t;
typedef fftw_plan_t (*fn_plan_r2c)(int, double*, fftw_complex_pair*, unsigned);
typedef fftw_plan_t (*fn_plan_c2r)(int, fftw_complex_pair*, double*, unsigned);
typedef void (*fn_execute)(fftw_plan_t);
typedef void (*fn_execute_r2c)(fftw_plan_t, double*, fftw_complex_pair*);
typedef void (*fn_destroy_plan)(fftw_plan_t);
typedef void *(*fn_malloc)(size_t);
typedef void (*fn_free)(void*);
typedef int (*fn_import_wisdom)(const char*);
typedef int (*fn_export_wisdom)(const char*);

typedef struct {
    const char *name;
    void *handle;
    fn_plan_r2c plan_r2c;
    fn_plan_c2r plan_c2r;
    fn_execute execute;
    fn_execute_r2c execute_r2c;
    fn_destroy_plan destroy_plan;
    fn_malloc fmalloc;
    fn_free ffree;
    fn_import_wisdom import_wisdom;
    fn_export_wisdom export_wisdom;
} FFTLib;

static int load_lib(FFTLib *lib, const char *name, const char *path) {
    lib->name = name;
    lib->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib->handle) {
        fprintf(stderr, "Warning: cannot load %s (%s): %s\n", name, path, dlerror());
        return 0;
    }
    lib->plan_r2c     = (fn_plan_r2c)dlsym(lib->handle, "fftw_plan_dft_r2c_1d");
    lib->plan_c2r     = (fn_plan_c2r)dlsym(lib->handle, "fftw_plan_dft_c2r_1d");
    lib->execute       = (fn_execute)dlsym(lib->handle, "fftw_execute");
    lib->execute_r2c   = (fn_execute_r2c)dlsym(lib->handle, "fftw_execute_dft_r2c");
    lib->destroy_plan  = (fn_destroy_plan)dlsym(lib->handle, "fftw_destroy_plan");
    lib->fmalloc       = (fn_malloc)dlsym(lib->handle, "fftw_malloc");
    lib->ffree         = (fn_free)dlsym(lib->handle, "fftw_free");
    lib->import_wisdom = (fn_import_wisdom)dlsym(lib->handle, "fftw_import_wisdom_from_filename");
    lib->export_wisdom = (fn_export_wisdom)dlsym(lib->handle, "fftw_export_wisdom_to_filename");

    if (!lib->plan_r2c || !lib->plan_c2r || !lib->execute || !lib->destroy_plan ||
        !lib->fmalloc || !lib->ffree) {
        fprintf(stderr, "Warning: %s missing required symbols\n", name);
        dlclose(lib->handle);
        lib->handle = NULL;
        return 0;
    }
    return 1;
}

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* 7-smooth numbers up to MAX_SIZE */
static int smooth_nums[800];
static int n_smooth = 0;

static void build_smooth_table(void) {
    for (int a = 1; a <= MAX_SIZE; a *= 2)
        for (int b = a; b <= MAX_SIZE; b *= 3)
            for (int c = b; c <= MAX_SIZE; c *= 5)
                for (int d = c; d <= MAX_SIZE; d *= 7)
                    smooth_nums[n_smooth++] = d;
    for (int i = 1; i < n_smooth; i++) {
        int key = smooth_nums[i], j = i - 1;
        while (j >= 0 && smooth_nums[j] > key) { smooth_nums[j+1] = smooth_nums[j]; j--; }
        smooth_nums[j+1] = key;
    }
}

#define FFTW_PATIENT (1U << 5)
#define FFTW_MEASURE 0U
#define FFTW_WISDOM_ONLY (1U << 21)
#define FFTW_ESTIMATE (1U << 6)

/* Benchmark one library at one size. Returns ns per pipeline call. */
static double bench_lib(FFTLib *lib, int sz, int quick) {
    if (!lib->handle) return 1e18;
    if (sz < 2) return 1e18;

    double *rbuf  = lib->fmalloc(sz * sizeof(double));
    double *rbuf2 = lib->fmalloc(sz * sizeof(double));
    fftw_complex_pair *cbuf  = lib->fmalloc((sz/2+1) * sizeof(fftw_complex_pair));
    fftw_complex_pair *cbuf2 = lib->fmalloc((sz/2+1) * sizeof(fftw_complex_pair));
    memset(rbuf, 0, sz * sizeof(double));
    memset(rbuf2, 0, sz * sizeof(double));

    /* Create plans — try MEASURE+WISDOM_ONLY first (uses PATIENT wisdom if available) */
    fftw_plan_t fwd = lib->plan_r2c(sz, rbuf, cbuf, FFTW_MEASURE | FFTW_WISDOM_ONLY);
    fftw_plan_t inv = lib->plan_c2r(sz, cbuf, rbuf, FFTW_MEASURE | FFTW_WISDOM_ONLY);
    if (!fwd || !inv) {
        if (fwd) lib->destroy_plan(fwd);
        if (inv) lib->destroy_plan(inv);
        fwd = lib->plan_r2c(sz, rbuf, cbuf, FFTW_ESTIMATE);
        inv = lib->plan_c2r(sz, cbuf, rbuf, FFTW_ESTIMATE);
    }
    if (!fwd || !inv) {
        if (fwd) lib->destroy_plan(fwd);
        if (inv) lib->destroy_plan(inv);
        lib->ffree(rbuf); lib->ffree(rbuf2);
        lib->ffree(cbuf); lib->ffree(cbuf2);
        return 1e18;
    }

    for (int j = 0; j < sz; j++) rbuf[j] = 1.0 + 0.001 * j;

    /* Adaptive: run for ~5ms per measurement (enough for stable medians) */
    int reps = (int)(5e6 / (double)(sz + 1));
    if (quick) reps /= 5;
    if (reps < 20) reps = 20;
    if (reps > 500000) reps = 500000;
    int cn = sz / 2 + 1;

    /* Warm up */
    for (int r = 0; r < 5; r++) {
        lib->execute(fwd);
        if (lib->execute_r2c)
            lib->execute_r2c(fwd, rbuf2, cbuf2);
        for (int j = 0; j < cn; j++) {
            double re = cbuf[j][0]*cbuf2[j][0] - cbuf[j][1]*cbuf2[j][1];
            double im = cbuf[j][0]*cbuf2[j][1] + cbuf[j][1]*cbuf2[j][0];
            cbuf[j][0] = re; cbuf[j][1] = im;
        }
        lib->execute(inv);
    }

    double t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        lib->execute(fwd);
        lib->execute_r2c(fwd, rbuf2, cbuf2);
        for (int j = 0; j < cn; j++) {
            double re = cbuf[j][0]*cbuf2[j][0] - cbuf[j][1]*cbuf2[j][1];
            double im = cbuf[j][0]*cbuf2[j][1] + cbuf[j][1]*cbuf2[j][0];
            cbuf[j][0] = re; cbuf[j][1] = im;
        }
        lib->execute(inv);
    }
    double ns = (now_ns() - t0) / reps;

    lib->destroy_plan(fwd);
    lib->destroy_plan(inv);
    lib->ffree(rbuf); lib->ffree(rbuf2);
    lib->ffree(cbuf); lib->ffree(cbuf2);
    return ns;
}

static double fftw_times[800], mkl_times[800];
static int best_lib[800]; /* 0=FFTW, 1=MKL */

int main(int argc, char **argv) {
    int quick = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--quick") == 0) quick = 1;

    build_smooth_table();
    printf("Found %d 7-smooth numbers up to %d\n\n", n_smooth, MAX_SIZE);

    /* Load both libraries */
    FFTLib fftw_lib = {0}, mkl_lib = {0};
    int have_fftw = load_lib(&fftw_lib, "FFTW", "libfftw3.so");
    int have_mkl  = load_lib(&mkl_lib,  "MKL",  "libmkl_rt.so");
    if (!have_mkl)
        have_mkl = load_lib(&mkl_lib, "MKL", "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so");

    if (!have_fftw && !have_mkl) {
        fprintf(stderr, "Error: neither FFTW nor MKL could be loaded\n");
        return 1;
    }
    printf("Libraries: FFTW=%s MKL=%s\n\n",
           have_fftw ? "loaded" : "MISSING",
           have_mkl  ? "loaded" : "MISSING");

    /* Import FFTW PATIENT wisdom — try multiple paths */
    const char *wisdom_paths[] = {
        WISDOM_FILE,
        "devices/zen4/fftw_wisdom.dat",
        "devices/m3_max/fftw_wisdom.dat",
        NULL
    };
    if (have_fftw && fftw_lib.import_wisdom) {
        int loaded = 0;
        for (int wp = 0; wisdom_paths[wp] && !loaded; wp++)
            loaded = fftw_lib.import_wisdom(wisdom_paths[wp]);
        printf("FFTW wisdom: %s\n", loaded ? "loaded" : "NOT FOUND (using ESTIMATE — results will be inaccurate!)");
    }
    if (have_mkl && mkl_lib.import_wisdom)
        mkl_lib.import_wisdom(WISDOM_FILE);

    /* Benchmark each size */
    printf("Benchmarking %d sizes...\n", n_smooth);
    printf("%-8s %-12s %-12s %-8s %-8s\n", "size", "FFTW(ns)", "MKL(ns)", "winner", "speedup");

    int fftw_wins = 0, mkl_wins = 0;
    for (int i = 0; i < n_smooth; i++) {
        int sz = smooth_nums[i];
        fftw_times[i] = bench_lib(&fftw_lib, sz, quick);
        mkl_times[i]  = bench_lib(&mkl_lib, sz, quick);

        if (fftw_times[i] <= mkl_times[i]) {
            best_lib[i] = 0;
            fftw_wins++;
        } else {
            best_lib[i] = 1;
            mkl_wins++;
        }

        if ((i + 1) % 50 == 0 || i == n_smooth - 1 || sz <= 64 ||
            (sz <= 1024 && sz % 128 == 0) || sz % 1024 == 0) {
            double winner_ns = (best_lib[i] == 0) ? fftw_times[i] : mkl_times[i];
            double loser_ns  = (best_lib[i] == 0) ? mkl_times[i] : fftw_times[i];
            printf("%-8d %-12.1f %-12.1f %-8s %.2fx\n",
                   sz, fftw_times[i], mkl_times[i],
                   best_lib[i] == 0 ? "FFTW" : "MKL",
                   loser_ns / winner_ns);
        }
    }
    printf("\nSummary: FFTW wins %d sizes, MKL wins %d sizes\n\n", fftw_wins, mkl_wins);

    /* Write output header */
    const char *outfile = "fft_config_dual.h";
    printf("Writing %s...\n", outfile);
    FILE *f = fopen(outfile, "w");
    if (!f) { perror(outfile); return 1; }

    fprintf(f, "/* Auto-generated dual-library FFT configuration */\n");
    fprintf(f, "/* FFTW vs MKL per-size benchmark on this machine */\n\n");

    fprintf(f, "#define N_CALIBRATED_SIZES %d\n", n_smooth);

    /* calib_sizes[] */
    fprintf(f, "static const int calib_sizes[N_CALIBRATED_SIZES] = {\n   ");
    for (int i = 0; i < n_smooth; i++) {
        fprintf(f, "%d", smooth_nums[i]);
        if (i < n_smooth - 1) fprintf(f, ",");
        if ((i + 1) % 20 == 0 && i < n_smooth - 1) fprintf(f, "\n   ");
    }
    fprintf(f, "\n};\n\n");

    /* calib_times_ns[] — best of both libraries */
    fprintf(f, "static const double calib_times_ns[N_CALIBRATED_SIZES] = {\n   ");
    for (int i = 0; i < n_smooth; i++) {
        double best = (best_lib[i] == 0) ? fftw_times[i] : mkl_times[i];
        fprintf(f, "%.1f", best);
        if (i < n_smooth - 1) fprintf(f, ",");
        if ((i + 1) % 10 == 0 && i < n_smooth - 1) fprintf(f, "\n   ");
    }
    fprintf(f, "\n};\n\n");

    /* calib_lib[] — which library won at each size */
    fprintf(f, "/* 0 = FFTW, 1 = MKL */\n");
    fprintf(f, "static const int calib_lib[N_CALIBRATED_SIZES] = {\n   ");
    for (int i = 0; i < n_smooth; i++) {
        fprintf(f, "%d", best_lib[i]);
        if (i < n_smooth - 1) fprintf(f, ",");
        if ((i + 1) % 40 == 0 && i < n_smooth - 1) fprintf(f, "\n   ");
    }
    fprintf(f, "\n};\n");

    fclose(f);
    printf("Done. %s written with %d sizes.\n", outfile, n_smooth);
    printf("\nNext: merge into devices/zen4/fft_config.h and add dual-dispatch to icm.c\n");

    if (have_fftw) dlclose(fftw_lib.handle);
    if (have_mkl)  dlclose(mkl_lib.handle);
    return 0;
}
