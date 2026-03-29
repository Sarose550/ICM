/*
 * contour_1s.c — Performance heatmap and contour sweep for (n, k) space.
 *
 * Modes:
 *   ./contour_1s              # 2D heatmap: sweep grid, output time/engine/memory CSV
 *   ./contour_1s --contour    # 1D contour: binary-search for 1s boundary per k
 *
 * Outputs CSV to stdout. Progress to stderr.
 *
 * Build (serial):
 *   gcc -O3 -march=znver4 -Isrc -Idevices/zen4 \
 *       -I/usr/local/aocl-fftw/include -L/usr/local/aocl-fftw/lib \
 *       -Wl,-rpath,/usr/local/aocl-fftw/lib \
 *       -o contour_1s tools/contour_1s.c -lfftw3 -lm -ldl
 *
 * Build (parallel):
 *   gcc -O3 -march=znver4 -fopenmp -Isrc -Idevices/zen4 \
 *       -I/usr/local/aocl-fftw/include -L/usr/local/aocl-fftw/lib \
 *       -Wl,-rpath,/usr/local/aocl-fftw/lib \
 *       -o contour_1s_par tools/contour_1s.c -lfftw3 -lfftw3_threads -lm -ldl
 *   OMP_NUM_THREADS=16 ./contour_1s_par
 */

#include "icm.c"

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

/* ── Timing ────────────────────────────────────────────────── */

static double wall_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Memory measurement (Linux) ────────────────────────────── */

/* Read VmHWM (high-water RSS) from /proc/self/status in KB */
static long read_vmhwm_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    long hwm = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, " %ld", &hwm);
            break;
        }
    }
    fclose(f);
    return hwm;
}

/* Measure memory for a single (n, k) point by forking a child.
 * The child initializes, runs icm_equity, reads VmHWM, writes it to a pipe.
 * Returns VmHWM in KB, or 0 on failure. */
static long measure_memory_kb(int n, int k, const double *S, double *payout) {
    int pipefd[2];
    if (pipe(pipefd) < 0) return 0;

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return 0; }

    if (pid == 0) {
        /* Child: inherited FFTW wisdom from parent via COW */
        close(pipefd[0]);

        int Q = 256;
        double *eq = (double *)malloc((size_t)n * sizeof(double));
        if (!eq) { _exit(1); }

        icm_equity(n, S, Q, payout, k, eq);

        long hwm = read_vmhwm_kb();
        free(eq);

        write(pipefd[1], &hwm, sizeof(hwm));
        close(pipefd[1]);
        _exit(0);
    }

    /* Parent */
    close(pipefd[1]);
    long hwm = 0;
    read(pipefd[0], &hwm, sizeof(hwm));
    close(pipefd[0]);
    waitpid(pid, NULL, 0);
    return hwm;
}

/* ── Stack generation ──────────────────────────────────────── */

static void make_random_stacks(int n, double *S, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
}

/* ── Time measurement ──────────────────────────────────────── */

static double measure_time(int n, const double *S, int k, double *payout,
                           double *equity, int reps) {
    int Q = 256;

    /* Warm-up */
    icm_equity(n, S, Q, payout, k, equity);

    double times[5];
    if (reps > 5) reps = 5;

    for (int r = 0; r < reps; r++) {
        double t0 = wall_time_sec();
        icm_equity(n, S, Q, payout, k, equity);
        times[r] = wall_time_sec() - t0;
    }

    /* Sort for median */
    for (int i = 0; i < reps; i++)
        for (int j = i + 1; j < reps; j++)
            if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

    return times[reps / 2];
}

/* ══════════════════════════════════════════════════════════════
   MODE 1: 2D HEATMAP SWEEP
   ══════════════════════════════════════════════════════════════ */

static void run_heatmap(int measure_mem) {
    /* Log-spaced n values */
    int n_values[] = {
        100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
        100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000
    };
    int n_n = sizeof(n_values) / sizeof(n_values[0]);

    /* Log-spaced k values */
    int k_values[] = {
        2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
        5000, 10000, 20000, 50000, 100000
    };
    int n_k = sizeof(k_values) / sizeof(k_values[0]);

    /* Time cutoff: skip (n, k) points that would take too long */
    double cutoff_sec = 10.0;

    /* Allocate max buffers */
    int max_n = n_values[n_n - 1];
    int max_k = k_values[n_k - 1];
    double *S = (double *)malloc((size_t)max_n * sizeof(double));
    double *equity = (double *)malloc((size_t)max_n * sizeof(double));
    double *payout = (double *)malloc((size_t)max_k * sizeof(double));
    if (!S || !equity || !payout) {
        fprintf(stderr, "Failed to allocate buffers for n=%d\n", max_n);
        exit(1);
    }

    for (int q = 0; q < max_k; q++) payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

    /* CSV header */
    if (measure_mem)
        printf("n,k,time_ms,engine,block_size,rss_kb\n");
    else
        printf("n,k,time_ms,engine,block_size\n");
    fflush(stdout);

    int total = 0, done = 0;
    for (int ki = 0; ki < n_k; ki++)
        for (int ni = 0; ni < n_n; ni++)
            if (k_values[ki] <= n_values[ni]) total++;

    for (int ki = 0; ki < n_k; ki++) {
        int k = k_values[ki];
        int skipping = 0;

        for (int ni = 0; ni < n_n; ni++) {
            int n = n_values[ni];
            if (k > n) continue;

            done++;

            if (skipping) {
                fprintf(stderr, "  [%d/%d] n=%d k=%d — skipped (exceeded cutoff)\n",
                        done, total, n, k);
                continue;
            }

            /* Generate stacks */
            make_random_stacks(n, S, 42);

            /* Determine engine via select_engine */
            int B = select_engine(n, k);
            const char *engine = (B > 0) ? "hybrid" : "linear";

            /* Adaptive reps: more reps for fast points */
            int reps = 3;

            fprintf(stderr, "  [%d/%d] n=%d k=%d (%s B=%d)...",
                    done, total, n, k, engine, B);

            double t = measure_time(n, S, k, payout, equity, reps);

            /* Memory measurement via fork (clean per-point) */
            long rss_kb = 0;
            if (measure_mem && t < 5.0) {
                rss_kb = measure_memory_kb(n, k, S, payout);
            }

            if (measure_mem)
                printf("%d,%d,%.1f,%s,%d,%ld\n", n, k, t * 1000.0, engine, B, rss_kb);
            else
                printf("%d,%d,%.1f,%s,%d\n", n, k, t * 1000.0, engine, B);
            fflush(stdout);

            fprintf(stderr, " %.1f ms\n", t * 1000.0);

            if (t > cutoff_sec) {
                skipping = 1;
                fprintf(stderr, "  (k=%d exceeded %.0fs cutoff at n=%d, skipping larger n)\n",
                        k, cutoff_sec, n);
            }
        }
    }

    free(S); free(equity); free(payout);
}

/* ══════════════════════════════════════════════════════════════
   MODE 2: CONTOUR SEARCH (binary search for 1s boundary per k)
   ══════════════════════════════════════════════════════════════ */

static int find_n_max(int k, double target_sec, double *out_time,
                      double *S_buf, double *payout, double *equity_buf,
                      int max_n) {
    int n_lo = k;
    int n_hi = (int)((double)50000000 / (k > 1 ? k : 1));
    if (n_hi > max_n) n_hi = max_n;
    if (n_hi < n_lo) n_hi = n_lo;

    make_random_stacks(n_hi, S_buf, 42);
    double t_hi = measure_time(n_hi, S_buf, k, payout, equity_buf, 1);

    while (t_hi < target_sec && n_hi < max_n) {
        n_hi = (int)((double)n_hi * 2.0);
        if (n_hi > max_n) n_hi = max_n;
        make_random_stacks(n_hi, S_buf, 42);
        t_hi = measure_time(n_hi, S_buf, k, payout, equity_buf, 1);
    }

    if (t_hi < target_sec) { *out_time = t_hi; return n_hi; }

    make_random_stacks(n_lo, S_buf, 42);
    double t_lo = measure_time(n_lo, S_buf, k, payout, equity_buf, 1);
    if (t_lo > target_sec) { *out_time = t_lo; return n_lo; }

    while ((double)(n_hi - n_lo) > 0.05 * (double)n_lo && n_hi - n_lo > 10) {
        int n_mid = n_lo + (n_hi - n_lo) / 2;
        make_random_stacks(n_mid, S_buf, 42);
        int reps = (t_lo + t_hi) / 2.0 < 0.1 ? 3 : 1;
        double t_mid = measure_time(n_mid, S_buf, k, payout, equity_buf, reps);
        if (t_mid <= target_sec) { n_lo = n_mid; t_lo = t_mid; }
        else { n_hi = n_mid; t_hi = t_mid; }
    }

    *out_time = t_lo;
    return n_lo;
}

static void run_contour(void) {
    int k_values[] = {
        2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
        5000, 10000, 20000, 50000, 100000, 200000, 500000,
        1000000, 2000000, 5000000
    };
    int n_k = sizeof(k_values) / sizeof(k_values[0]);
    double target_sec = 1.0;
    int max_n = 100000000;

    int max_k = k_values[n_k - 1];
    int alloc_n = 20000000;
    double *S = (double *)malloc((size_t)alloc_n * sizeof(double));
    double *equity = (double *)malloc((size_t)alloc_n * sizeof(double));
    double *payout = (double *)malloc((size_t)max_k * sizeof(double));
    if (!S || !equity || !payout) { fprintf(stderr, "alloc failed\n"); exit(1); }
    for (int q = 0; q < max_k; q++) payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

    printf("k,n_max,time_ms,engine,block_size\n");
    fflush(stdout);

    for (int ki = 0; ki < n_k; ki++) {
        int k = k_values[ki];
        if (k > max_n) break;

        int need_n = (int)((double)50000000 / (k > 1 ? k : 1));
        if (need_n > max_n) need_n = max_n;
        if (need_n < k) need_n = k;
        need_n *= 4;
        if (need_n > max_n) need_n = max_n;

        if (need_n > alloc_n) {
            alloc_n = need_n;
            free(S); free(equity);
            S = (double *)malloc((size_t)alloc_n * sizeof(double));
            equity = (double *)malloc((size_t)alloc_n * sizeof(double));
            if (!S || !equity) { fprintf(stderr, "realloc failed n=%d\n", alloc_n); exit(1); }
        }

        fprintf(stderr, "k=%d: searching for 1s boundary...\n", k);

        double time_sec;
        int eff_max = max_n < alloc_n ? max_n : alloc_n;
        int n_max = find_n_max(k, target_sec, &time_sec, S, payout, equity, eff_max);

        int B = select_engine(n_max, k);
        const char *engine = (B > 0) ? "hybrid" : "linear";

        printf("%d,%d,%.0f,%s,%d\n", k, n_max, time_sec * 1000.0, engine, B);
        fflush(stdout);
        fprintf(stderr, "  k=%d -> n_max=%d (%.0f ms) [%s B=%d]\n",
                k, n_max, time_sec * 1000.0, engine, B);
    }

    free(S); free(equity); free(payout);
}

/* ── Main ──────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int contour_mode = 0;
    int measure_mem = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--contour") == 0) contour_mode = 1;
        if (strcmp(argv[i], "--memory") == 0) measure_mem = 1;
    }

    icm_init("devices/zen4/fftw_wisdom.dat");

    if (contour_mode) {
        run_contour();
    } else {
        run_heatmap(measure_mem);
    }

    return 0;
}
