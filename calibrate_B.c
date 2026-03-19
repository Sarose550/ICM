/*
 * calibrate_B.c — Measure the CPU-specific parameters that determine B*
 *
 * Profiles the sequential-combine polynomial build to understand how the
 * optimal block size B depends on cache hierarchy, FMA throughput, and
 * prefetcher behavior.  Run on a new CPU to reproduce the B* analysis.
 *
 * Five experiments (run individually or all at once):
 *   M1: α₁(B)    — phase-1 per-FMA cost (block build, all in L1)
 *   M2: α₂(d,B)  — phase-2 per-FMA cost (schoolbook, isolated, warm cache)
 *   M3: overhead  — per-step cache-pollution penalty during real builds
 *                   (step_time − FMA_count × α₂_warm, self-calibrated)
 *   M4: T(n,B)   — end-to-end build time, fine B sweep across n values
 *   M5: memset    — warm and cold memset rates vs buffer size
 *
 * Feed the CSV output to analyze_calibration.py for cost-model fitting
 * and B*(n) derivation.
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -mfma -Wall -o calibrate_B calibrate_B.c -lm
 *
 * Usage:
 *   ./calibrate_B          # run all experiments (may take 5-10 minutes)
 *   ./calibrate_B 1        # run only M1
 *   ./calibrate_B 4        # run only M4 (end-to-end, most important)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

static void *aligned_malloc(size_t alignment, size_t size) {
    void *p = NULL;
    posix_memalign(&p, alignment, size);
    return p;
}

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ── Standard kernels (matching production code) ─────────── */

static void build_block(int count, const double *a, double *P, int B) {
    P[0] = 1;
    for (int j = 1; j <= B; j++) P[j] = 0;
    for (int j = 0; j < count; j++) {
        double aj = a[j], bj = 1 - aj;
        int nd = j + 1;
        for (int m = nd; m >= 1; m--)
            P[m] = aj * P[m] + bj * P[m - 1];
        P[0] *= aj;
    }
}

static void schoolbook(const double *src, int src_deg,
                       const double *ch, int ch_deg,
                       double *dst, int max_deg) {
    int new_deg = src_deg + ch_deg;
    if (new_deg > max_deg) new_deg = max_deg;
    memset(dst, 0, (new_deg + 1) * sizeof(double));
    for (int i = 0; i <= src_deg; i++) {
        double si = src[i];
        if (si == 0) continue;
        int jmax = ch_deg;
        if (i + jmax > new_deg) jmax = new_deg - i;
        double *d = dst + i;
        for (int j = 0; j <= jmax; j++)
            d[j] += si * ch[j];
    }
}

static void fill_random(double *P, int n, unsigned seed) {
    srand(seed);
    double sum = 0;
    for (int i = 0; i <= n; i++) {
        P[i] = (double)(rand() % 1000 + 1) / 1000.0;
        sum += P[i];
    }
    for (int i = 0; i <= n; i++) P[i] /= sum;
}

/* ══════════════════════════════════════════════════════════════
   M1: Phase-1 per-FMA cost α₁(B)
   ══════════════════════════════════════════════════════════════ */

static void measure_alpha1(void) {
    printf("# M1: alpha1(B) — phase-1 per-FMA cost\n");
    printf("# B, alpha1_ns, total_ns, FMAs\n");

    double *a = malloc(512 * sizeof(double));
    double *P = aligned_malloc(64, 513 * sizeof(double));
    srand(42);
    for (int i = 0; i < 512; i++)
        a[i] = 0.3 + 0.4 * ((double)rand() / RAND_MAX);

    int Bs[] = {32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512};
    for (int bi = 0; bi < 15; bi++) {
        int B = Bs[bi];
        long long fmas = (long long)B * (B + 1) / 2;
        int reps = (int)(2e9 / (double)fmas);
        if (reps < 5) reps = 5;
        if (reps > 3000) reps = 3000;

        build_block(B, a, P, B); /* warmup */

        double best = 1e30;
        for (int trial = 0; trial < 5; trial++) {
            double t0 = now_ns();
            for (int r = 0; r < reps; r++)
                build_block(B, a, P, B);
            double el = (now_ns() - t0) / reps;
            if (el < best) best = el;
        }
        printf("%4d, %8.4f, %10.1f, %lld\n", B, best / fmas, best, fmas);
    }
    free(a); free(P);
}

/* ══════════════════════════════════════════════════════════════
   M2: Phase-2 per-FMA cost α₂(d, B) — isolated, warm cache
   ══════════════════════════════════════════════════════════════ */

static void measure_alpha2_warm(void) {
    printf("\n# M2: alpha2_warm(d, B) — schoolbook per-FMA, warm cache\n");
    printf("# d, B, alpha2_ns, ws_KB\n");

    int max_d = 32768, max_B = 512;
    double *src = aligned_malloc(64, (max_d + 1) * sizeof(double));
    double *dst = aligned_malloc(64, (max_d + max_B + 2) * sizeof(double));
    double *ch  = aligned_malloc(64, (max_B + 1) * sizeof(double));

    int Bs[] = {128, 256, 384, 512};
    int degrees[] = {128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096,
                     6144, 8192, 12288, 16384, 24576, 32768};

    for (int bi = 0; bi < 4; bi++) {
        int B = Bs[bi];
        fill_random(ch, B, 99);
        for (int di = 0; di < 15; di++) {
            int d = degrees[di];
            fill_random(src, d, 42);
            long long fmas = (long long)(d + 1) * (B + 1);
            int reps = (int)(5e8 / (double)fmas);
            if (reps < 3) reps = 3;
            if (reps > 200) reps = 200;

            schoolbook(src, d, ch, B, dst, d + B); /* warmup */
            double best = 1e30;
            for (int trial = 0; trial < 3; trial++) {
                double t0 = now_ns();
                for (int r = 0; r < reps; r++)
                    schoolbook(src, d, ch, B, dst, d + B);
                double el = (now_ns() - t0) / reps;
                if (el < best) best = el;
            }
            double ws_kb = ((2.0 * d + B + 3) * 8) / 1024.0;
            printf("%6d, %4d, %8.4f, %8.1f\n", d, B, best / fmas, ws_kb);
        }
    }
    free(src); free(dst); free(ch);
}

/* ══════════════════════════════════════════════════════════════
   M3: Per-step overhead during real builds
   For each step: f_step = step_time - FMAs × α₂_warm(d, B)
   This captures the cache pollution penalty that the isolated
   benchmark misses.
   ══════════════════════════════════════════════════════════════ */

static void measure_step_overhead(void) {
    printf("\n# M3: per-step overhead during real build\n");
    printf("# n, B, step, d, step_ns, fma_ns_pred, overhead_ns, overhead_frac\n");

    int ns[] = {2048, 4096, 8192, 16384};
    int Bs[] = {128, 256, 384};

    /* Self-calibrate: quick warm-cache α₂ measurement for each B */
    double alpha2_by_B[3];
    {
        int cal_d = 4096;
        double *s = aligned_malloc(64, (cal_d + 1) * sizeof(double));
        double *d = aligned_malloc(64, (cal_d + 513) * sizeof(double));
        double *c = aligned_malloc(64, 513 * sizeof(double));
        fill_random(s, cal_d, 42);
        for (int bi = 0; bi < 3; bi++) {
            int B = Bs[bi];
            fill_random(c, B, 99);
            long long fmas = (long long)(cal_d + 1) * (B + 1);
            int reps = (int)(3e8 / (double)fmas);
            if (reps < 3) reps = 3;
            schoolbook(s, cal_d, c, B, d, cal_d + B); /* warmup */
            double best = 1e30;
            for (int t = 0; t < 3; t++) {
                double t0 = now_ns();
                for (int r = 0; r < reps; r++)
                    schoolbook(s, cal_d, c, B, d, cal_d + B);
                double el = (now_ns() - t0) / reps;
                if (el < best) best = el;
            }
            alpha2_by_B[bi] = best / fmas;
            fprintf(stderr, "# M3 calibration: B=%d alpha2_warm=%.4f ns/FMA\n",
                    B, alpha2_by_B[bi]);
        }
        free(s); free(d); free(c);
    }

    for (int ni = 0; ni < 4; ni++) {
        int n = ns[ni];
        double *a = malloc(n * sizeof(double));
        srand(42);
        for (int i = 0; i < n; i++)
            a[i] = 0.1 + 0.8 * ((double)rand() / RAND_MAX);

        for (int bi = 0; bi < 3; bi++) {
            int B = Bs[bi];
            int C = (n + B - 1) / B;
            int ps = B + 1;
            double *buf0 = aligned_malloc(64, (n + 2) * sizeof(double));
            double *buf1 = aligned_malloc(64, (n + 2) * sizeof(double));
            double *chunks = aligned_malloc(64, (size_t)C * ps * sizeof(double));

            /* Phase 1 */
            for (int c = 0; c < C; c++) {
                int start = c * B;
                int count = (start + B <= n) ? B : (n - start);
                double *ch = chunks + (size_t)c * ps;
                build_block(count, a + start, ch, B);
            }

            /* Phase 2 with per-step decomposition */
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

                long long fmas = (long long)(deg + 1) * (ch_deg + 1);

                /* Time the full step */
                double t0 = now_ns();
                memset(dst, 0, (new_deg + 1) * sizeof(double));
                for (int i = 0; i <= deg; i++) {
                    double si = src[i];
                    if (si == 0) continue;
                    int jmax = ch_deg;
                    if (i + jmax > new_deg) jmax = new_deg - i;
                    double *d = dst + i;
                    for (int j = 0; j <= jmax; j++)
                        d[j] += si * ch[j];
                }
                double step_ns = now_ns() - t0;

                /* Predict FMA time from warm-cache α₂ (self-calibrated) */
                double alpha2_w = alpha2_by_B[bi];

                double fma_pred = fmas * alpha2_w;
                double overhead = step_ns - fma_pred;
                double overhead_frac = overhead / step_ns;

                printf("%6d, %4d, %4d, %6d, %12.1f, %12.1f, %12.1f, %8.4f\n",
                       n, B, c, deg, step_ns, fma_pred, overhead, overhead_frac);

                deg = new_deg;
                double *tmp = src; src = dst; dst = tmp;
            }
            free(buf0); free(buf1); free(chunks);
        }
        free(a);
    }
}

/* ══════════════════════════════════════════════════════════════
   M4: End-to-end validation — T(n, B) measured vs predicted
   ══════════════════════════════════════════════════════════════ */

static void build_seq(int n, const double *a, double *P) {
    P[0] = 1;
    for (int j = 1; j <= n; j++) P[j] = 0;
    int deg = 0;
    for (int j = 0; j < n; j++) {
        double aj = a[j], bj = 1 - aj;
        int nd = (deg + 1 < n) ? deg + 1 : n;
        for (int m = nd; m >= 1; m--)
            P[m] = aj * P[m] + bj * P[m - 1];
        P[0] *= aj;
        deg = nd;
    }
}

static void build_sc(int n, const double *a, int B,
                     double *b0, double *b1, double *ch) {
    int C = (n + B - 1) / B, ps = B + 1;
    for (int c = 0; c < C; c++) {
        int s = c * B, cnt = (s + B <= n) ? B : (n - s);
        build_block(cnt, a + s, ch + (size_t)c * ps, B);
    }
    int fc = (B <= n) ? B : n;
    memcpy(b0, ch, (fc + 1) * 8);
    int deg = fc;
    double *src = b0, *dst = b1;
    for (int c = 1; c < C; c++) {
        int s = c * B, cnt = (s + B <= n) ? B : (n - s);
        const double *p = ch + (size_t)c * ps;
        int cd = cnt, nd = deg + cd;
        if (nd > n) nd = n;
        memset(dst, 0, (nd + 1) * 8);
        for (int i = 0; i <= deg; i++) {
            double si = src[i]; if (si == 0) continue;
            int jm = cd; if (i + jm > nd) jm = nd - i;
            double *d = dst + i;
            for (int j = 0; j <= jm; j++) d[j] += si * p[j];
        }
        deg = nd;
        double *t = src; src = dst; dst = t;
    }
    if (src != b0) memcpy(b0, src, (n + 1) * 8);
}

static void measure_end_to_end(void) {
    printf("\n# M4: end-to-end T(n, B)\n");
    printf("# n, B, C, build_us, seq_us, speedup\n");

    int ns[] = {256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384};
    int Bs[] = {32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288,
                320, 352, 384, 416, 448, 480, 512, 576, 640, 768};
    int nB = 22;

    for (int ni = 0; ni < 10; ni++) {
        int n = ns[ni];
        double *a = malloc(n * sizeof(double));
        srand(42);
        for (int i = 0; i < n; i++)
            a[i] = 0.1 + 0.8 * ((double)rand() / RAND_MAX);

        double *P = aligned_malloc(64, (n + 2) * sizeof(double));
        int reps = (int)(5e8 / ((double)n * n));
        if (reps < 2) reps = 2;
        if (reps > 200) reps = 200;

        build_seq(n, a, P);
        double best_seq = 1e30;
        for (int t = 0; t < 3; t++) {
            double t0 = now_ns();
            for (int r = 0; r < reps; r++) build_seq(n, a, P);
            double el = (now_ns() - t0) / reps;
            if (el < best_seq) best_seq = el;
        }
        free(P);

        for (int bi = 0; bi < nB; bi++) {
            int B = Bs[bi];
            if (B >= n) continue;
            int C = (n + B - 1) / B;
            double *b0 = aligned_malloc(64, (n + 2) * 8);
            double *b1 = aligned_malloc(64, (n + 2) * 8);
            double *ch = aligned_malloc(64, (size_t)C * (B + 1) * 8);

            build_sc(n, a, B, b0, b1, ch);
            double best = 1e30;
            for (int t = 0; t < 3; t++) {
                double t0 = now_ns();
                for (int r = 0; r < reps; r++) build_sc(n, a, B, b0, b1, ch);
                double el = (now_ns() - t0) / reps;
                if (el < best) best = el;
            }
            printf("%6d, %4d, %4d, %10.1f, %10.1f, %6.3f\n",
                   n, B, C, best / 1e3, best_seq / 1e3, best_seq / best);
            free(b0); free(b1); free(ch);
        }
        free(a);
    }
}

/* ══════════════════════════════════════════════════════════════
   M5: memset rate vs size (warm and cold)
   ══════════════════════════════════════════════════════════════ */

static void measure_memset(void) {
    printf("\n# M5: memset rate\n");
    printf("# size_doubles, size_KB, warm_ns_per_dbl, cold_ns_per_dbl\n");

    int max_sz = 65536;
    double *buf = aligned_malloc(64, (max_sz + 1) * sizeof(double));
    /* "polluter" to evict buf from cache */
    double *polluter = aligned_malloc(64, 2 * 1024 * 1024);

    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192,
                   16384, 32768, 65536};

    for (int si = 0; si < 11; si++) {
        int sz = sizes[si];
        int reps = (int)(1e9 / (double)sz);
        if (reps < 10) reps = 10;
        if (reps > 50000) reps = 50000;

        /* Warm: repeated memset */
        memset(buf, 0, sz * 8);
        double best_warm = 1e30;
        for (int trial = 0; trial < 5; trial++) {
            double t0 = now_ns();
            for (int r = 0; r < reps; r++) {
                memset(buf, 0, sz * 8);
                asm volatile("" ::: "memory");
            }
            double el = (now_ns() - t0) / reps / sz;
            if (el < best_warm) best_warm = el;
        }

        /* Cold: memset after cache pollution */
        int cold_reps = reps / 10;
        if (cold_reps < 5) cold_reps = 5;
        double sum_cold = 0;
        for (int r = 0; r < cold_reps; r++) {
            /* Pollute: touch 2MB of other data */
            for (int i = 0; i < 2*1024*1024/8; i += 8)
                polluter[i] = (double)i;
            asm volatile("" ::: "memory");
            double t0 = now_ns();
            memset(buf, 0, sz * 8);
            asm volatile("" ::: "memory");
            sum_cold += (now_ns() - t0) / sz;
        }
        double avg_cold = sum_cold / cold_reps;

        printf("%6d, %8.1f, %8.4f, %8.4f\n",
               sz, sz * 8.0 / 1024, best_warm, avg_cold);
    }
    free(buf); free(polluter);
}

/* ── Main ────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    long l1d = 32768, l2 = 1048576, l3 = 33554432;
#ifdef __APPLE__
    size_t sz = sizeof(long);
    sysctlbyname("hw.l1dcachesize", &l1d, &sz, NULL, 0);
    sz = sizeof(long);
    sysctlbyname("hw.l2cachesize", &l2, &sz, NULL, 0);
    sz = sizeof(long);
    sysctlbyname("hw.l3cachesize", &l3, &sz, NULL, 0);
#else
    FILE *f;
    f = popen("getconf LEVEL1_DCACHE_SIZE 2>/dev/null", "r");
    if (f) { if(fscanf(f, "%ld", &l1d)){} pclose(f); }
    f = popen("getconf LEVEL2_CACHE_SIZE 2>/dev/null", "r");
    if (f) { if(fscanf(f, "%ld", &l2)){} pclose(f); }
    f = popen("getconf LEVEL3_CACHE_SIZE 2>/dev/null", "r");
    if (f) { if(fscanf(f, "%ld", &l3)){} pclose(f); }
#endif

    printf("# Hardware: L1D=%ldKB L2=%ldKB L3=%ldKB\n",
           l1d/1024, l2/1024, l3/1024);
    printf("# L1D_doubles=%ld L2_doubles=%ld\n\n",
           l1d/8, l2/8);

    int run_all = (argc < 2);
    int m = (argc >= 2) ? atoi(argv[1]) : 0;

    if (run_all || m == 1) measure_alpha1();
    if (run_all || m == 2) measure_alpha2_warm();
    if (run_all || m == 3) measure_step_overhead();
    if (run_all || m == 4) measure_end_to_end();
    if (run_all || m == 5) measure_memset();

    return 0;
}
