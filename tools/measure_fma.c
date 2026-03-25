/*
 * measure_fma.c — Measure scalar FMA throughput and memory bandwidth.
 * Run on a dedicated core: taskset -c 1 ./measure_fma
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* Measure FMA throughput: schoolbook polynomial multiply */
static double measure_fma_ns(void) {
    int sizes[] = {32, 64, 128, 256, 512, 1024};
    int n_sizes = 6;
    double best_ns_per_fma = 1e9;

    for (int si = 0; si < n_sizes; si++) {
        int n = sizes[si];
        double *a = calloc(n, sizeof(double));
        double *b = calloc(n, sizeof(double));
        double *c = calloc(2*n, sizeof(double));
        for (int i = 0; i < n; i++) { a[i] = 1.0 + 0.001*i; b[i] = 1.0 - 0.001*i; }

        int reps = (int)(2e9 / ((double)n * n));
        if (reps < 10) reps = 10;
        if (reps > 100000) reps = 100000;

        /* Warmup */
        for (int r = 0; r < 5; r++) {
            memset(c, 0, 2*n*sizeof(double));
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    c[i+j] += a[i] * b[j];
        }

        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memset(c, 0, 2*n*sizeof(double));
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    c[i+j] += a[i] * b[j];
        }
        double elapsed = now_ns() - t0;
        long long fmas = (long long)n * n * reps;
        double ns_per = elapsed / fmas;

        printf("  schoolbook n=%-5d  %8d reps  %.3f ns/FMA  (%.1f GFLOPS)\n",
               n, reps, ns_per, 2.0 / ns_per);

        if (ns_per < best_ns_per_fma) best_ns_per_fma = ns_per;
        free(a); free(b); free(c);
    }
    return best_ns_per_fma;
}

/* Measure memory bandwidth: sequential read */
static void measure_bandwidth(void) {
    printf("\n=== MEMORY BANDWIDTH ===\n");
    int sizes_mb[] = {1, 4, 16, 64, 256};
    int n_bw = 5;

    for (int si = 0; si < n_bw; si++) {
        size_t bytes = (size_t)sizes_mb[si] * 1024 * 1024;
        size_t n = bytes / sizeof(double);
        double *buf = malloc(bytes);
        for (size_t i = 0; i < n; i++) buf[i] = (double)i;

        int reps = (int)(2e10 / bytes);
        if (reps < 5) reps = 5;

        /* Warmup */
        volatile double sink = 0;
        for (int r = 0; r < 3; r++) {
            double s = 0;
            for (size_t i = 0; i < n; i++) s += buf[i];
            sink += s;
        }

        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            double s = 0;
            for (size_t i = 0; i < n; i++) s += buf[i];
            sink += s;
        }
        double elapsed = now_ns() - t0;
        double gb_per_s = (double)bytes * reps / elapsed;

        printf("  %4d MB  %4d reps  %.1f GB/s\n", sizes_mb[si], reps, gb_per_s);
        free(buf);
    }
}

/* Measure L1/L2/L3 latency via pointer chasing */
static void measure_cache_latency(void) {
    printf("\n=== CACHE LATENCY ===\n");
    int sizes_kb[] = {16, 32, 64, 256, 1024, 4096, 32768};
    int n_lat = 7;

    for (int si = 0; si < n_lat; si++) {
        size_t bytes = (size_t)sizes_kb[si] * 1024;
        size_t n = bytes / sizeof(void*);
        void **buf = malloc(bytes);

        /* Build random cycle through all elements */
        int *order = malloc(n * sizeof(int));
        for (size_t i = 0; i < n; i++) order[i] = i;
        srand(42);
        for (size_t i = n - 1; i > 0; i--) {
            size_t j = rand() % (i + 1);
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }
        for (size_t i = 0; i < n - 1; i++)
            buf[order[i]] = &buf[order[i+1]];
        buf[order[n-1]] = &buf[order[0]];
        free(order);

        int reps = (int)(1e9 / n);
        if (reps < 100) reps = 100;

        volatile void *p = buf;
        for (int r = 0; r < reps; r++)
            for (size_t i = 0; i < n; i++)
                p = *(void**)p;

        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            for (size_t i = 0; i < n; i++)
                p = *(void**)p;
        double elapsed = now_ns() - t0;
        double ns_per = elapsed / ((double)reps * n);

        printf("  %6d KB  %.1f ns/access\n", sizes_kb[si], ns_per);
        free(buf);
    }
}

int main(void) {
    printf("=== FMA THROUGHPUT (schoolbook polynomial multiply) ===\n");
    double fma_ns = measure_fma_ns();
    printf("\n  >> Best FMA_NS = %.4f  (use this for fft_config.h)\n", fma_ns);

    measure_bandwidth();
    measure_cache_latency();

    printf("\n=== SUMMARY ===\n");
    printf("  FMA_NS = %.4f ns\n", fma_ns);
    printf("  (For fft_config.h: #define FMA_NS %.2f)\n", fma_ns);
    return 0;
}
