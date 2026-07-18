/*
 * contour_1s.c — Performance heatmap, contour, and frontier for (n, k) space.
 *
 * Modes:
 *   ./contour_1s              # 2D heatmap: sweep grid, output time/engine CSV
 *   ./contour_1s --contour    # 1D contour: binary-search for 1s boundary per k
 *   ./contour_1s --nk         # n=k threshold: binary-search along the diagonal
 *
 * Options:
 *   --timeout <sec>           # hard kill timeout per probe (default 10s)
 *   --wisdom <path>           # FFTW wisdom file (default: fftw_wisdom.dat in CWD)
 *   --memory                  # measure RSS (heatmap mode only, Linux)
 *
 * Outputs CSV to stdout. Progress to stderr.
 *
 * Build: make contour_1s (serial) or make contour_1s_par (parallel)
 * Links against libicm.a — does not #include icm.c.
 */

#include "icm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

/* ── Timing ────────────────────────────────────────────────── */

static double wall_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Timeout infrastructure (fork-based) ──────────────────── */

/* Run icm_equity in a forked child with a hard wall-clock timeout.
 * Returns elapsed seconds, or -1.0 if killed by timeout.
 * The fork isolates FFTW state — no corruption on kill. */
static double fork_timed_equity(int n, const double *S, int Q, double *payout,
                                int k, double *equity, double timeout_sec) {
    int pipefd[2];
    if (pipe(pipefd) < 0) return -1.0;

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return -1.0; }

    if (pid == 0) {
        /* Child: inherited FFTW wisdom + plans via COW */
        close(pipefd[0]);
        double t0 = wall_time_sec();
        icm_equity(n, S, Q, payout, k, equity);
        double elapsed = wall_time_sec() - t0;
        write(pipefd[1], &elapsed, sizeof(elapsed));
        close(pipefd[1]);
        _exit(0);
    }

    /* Parent: wait with timeout */
    close(pipefd[1]);

    double t0 = wall_time_sec();
    double elapsed = -1.0;
    while (1) {
        int status;
        pid_t w = waitpid(pid, &status, WNOHANG);
        if (w > 0) {
            /* Child finished */
            read(pipefd[0], &elapsed, sizeof(elapsed));
            break;
        }
        if (wall_time_sec() - t0 > timeout_sec) {
            /* Timeout — kill child */
            kill(pid, SIGKILL);
            waitpid(pid, NULL, 0);
            elapsed = -1.0;
            break;
        }
        usleep(1000);  /* 1ms poll */
    }
    close(pipefd[0]);
    return elapsed;
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

/* Hard timeout for a single icm_equity call (seconds). */
static double g_call_timeout = 10.0;

/* If timeout > 0, use fork-based isolation to enforce it.
 * For the contour's binary search, only the initial probe at large n needs this;
 * once we know the rough scale, subsequent probes are fast and run in-process. */
static int g_use_fork_timeout = 0;

static double measure_time(int n, const double *S, int k, double *payout,
                           double *equity, int reps) {
    int Q = 256;

    /* Warm-up — use fork timeout if enabled */
    double warmup;
    if (g_use_fork_timeout) {
        warmup = fork_timed_equity(n, S, Q, payout, k, equity, g_call_timeout);
        if (warmup < 0) return g_call_timeout;  /* killed by timeout */
    } else {
        double t0 = wall_time_sec();
        icm_equity(n, S, Q, payout, k, equity);
        warmup = wall_time_sec() - t0;
    }
    if (warmup > 2.0) return warmup;

    double times[5];
    if (reps > 5) reps = 5;

    for (int r = 0; r < reps; r++) {
        double t;
        if (g_use_fork_timeout) {
            t = fork_timed_equity(n, S, Q, payout, k, equity, g_call_timeout);
            if (t < 0) { times[r] = g_call_timeout; reps = r + 1; break; }
        } else {
            double t0 = wall_time_sec();
            icm_equity(n, S, Q, payout, k, equity);
            t = wall_time_sec() - t0;
        }
        times[r] = t;
        if (r == 0 && times[0] > 1.5) { reps = 1; break; }
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
    /* Power-of-2 grid matching GPU heatmap for direct comparison */
    int n_values[] = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576
    };
    int n_n = sizeof(n_values) / sizeof(n_values[0]);

    int k_values[] = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576
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
            int B = icm_select_engine(n, k);
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
   BINARY SEARCH HELPERS
   ══════════════════════════════════════════════════════════════ */

/* Probe function type: given n, return wall-clock time in seconds.
 * Implementations set up stacks/payout internally based on the mode. */
typedef double (*probe_fn)(int n, void *ctx);

/* Generic binary search: find largest n in [n_lo, max_n] where probe(n) <= target_sec.
 * Starts at n_start, doubles until exceeding target, then bisects.
 * Returns the threshold n and writes its time to *out_time.
 * If even n_lo exceeds target, returns n_lo with its time. */
static int bisect_threshold(probe_fn probe, void *ctx,
                            int n_lo, int n_start, int max_n,
                            double target_sec, double *out_time) {
    int n_hi = n_start;
    if (n_hi > max_n) n_hi = max_n;
    if (n_hi < n_lo) n_hi = n_lo;

    double t_hi = probe(n_hi, ctx);

    /* Expand until we exceed the target or hit the ceiling */
    while (t_hi < target_sec && n_hi < max_n) {
        n_hi = (int)((double)n_hi * 2.0);
        if (n_hi > max_n) n_hi = max_n;
        t_hi = probe(n_hi, ctx);
    }

    if (t_hi < target_sec) { *out_time = t_hi; return n_hi; }
    if (n_lo == n_hi) { *out_time = t_hi; return n_lo; }

    double t_lo = probe(n_lo, ctx);
    if (t_lo > target_sec) { *out_time = t_lo; return n_lo; }

    /* Bisect to 5% precision */
    while ((double)(n_hi - n_lo) > 0.05 * (double)n_lo && n_hi - n_lo > 10) {
        int n_mid = n_lo + (n_hi - n_lo) / 2;
        int reps_hint = (t_lo + t_hi) / 2.0 < 0.1 ? 3 : 1;
        (void)reps_hint; /* used by callers that set rep count in ctx */
        double t_mid = probe(n_mid, ctx);
        if (t_mid <= target_sec) { n_lo = n_mid; t_lo = t_mid; }
        else { n_hi = n_mid; t_hi = t_mid; }
    }

    *out_time = t_lo;
    return n_lo;
}

/* ── Probe: fixed k, varying n ─────────────────────────────── */

typedef struct {
    int k;
    double *S_buf;
    double *payout;
    double *equity_buf;
} contour_probe_ctx;

static double contour_probe(int n, void *raw) {
    contour_probe_ctx *ctx = (contour_probe_ctx *)raw;
    make_random_stacks(n, ctx->S_buf, 42);
    return measure_time(n, ctx->S_buf, ctx->k, ctx->payout, ctx->equity_buf, 1);
}

/* ── Probe: n=k (diagonal) ─────────────────────────────────── */

typedef struct {
    double *S_buf;
    double *payout;
    double *equity_buf;
    int alloc_n;
    int alloc_k;
} nk_probe_ctx;

static double nk_probe(int n, void *raw) {
    nk_probe_ctx *ctx = (nk_probe_ctx *)raw;
    /* Grow payout if needed (k=n may exceed previous allocation) */
    if (n > ctx->alloc_k) {
        free(ctx->payout);
        ctx->alloc_k = n;
        ctx->payout = (double *)malloc((size_t)n * sizeof(double));
        for (int q = 0; q < n; q++) ctx->payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);
    }
    /* Grow stacks/equity if needed */
    if (n > ctx->alloc_n) {
        free(ctx->S_buf); free(ctx->equity_buf);
        ctx->alloc_n = n;
        ctx->S_buf = (double *)malloc((size_t)n * sizeof(double));
        ctx->equity_buf = (double *)malloc((size_t)n * sizeof(double));
    }
    make_random_stacks(n, ctx->S_buf, 42);
    return measure_time(n, ctx->S_buf, n, ctx->payout, ctx->equity_buf, 1);
}

/* ══════════════════════════════════════════════════════════════
   MODE 2: CONTOUR SEARCH (binary search for 1s boundary per k)
   ══════════════════════════════════════════════════════════════ */

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

    contour_probe_ctx ctx = { .k = 0, .S_buf = S, .payout = payout, .equity_buf = equity };

    printf("k,n_max,time_ms,engine,block_size,status\n");
    fflush(stdout);

    int consecutive_timeouts = 0;
    for (int ki = 0; ki < n_k; ki++) {
        int k = k_values[ki];
        if (k > max_n) break;
        if (consecutive_timeouts >= 3) {
            fprintf(stderr, "k=%d: skipped (contour converged to n=k diagonal)\n", k);
            continue;
        }

        /* Ensure buffers are large enough */
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
            ctx.S_buf = S;
            ctx.equity_buf = equity;
        }

        ctx.k = k;
        int eff_max = max_n < alloc_n ? max_n : alloc_n;
        int n_start = (int)((double)50000000 / (k > 1 ? k : 1));

        fprintf(stderr, "k=%d: searching for 1s boundary...\n", k);

        double time_sec;
        int n_max = bisect_threshold(contour_probe, &ctx, k, n_start, eff_max,
                                     target_sec, &time_sec);

        int B = icm_select_engine(n_max, k);
        const char *engine = (B > 0) ? "hybrid" : "linear";

        int valid = (time_sec <= target_sec * 1.05);  /* 5% tolerance */
        if (!valid)
            consecutive_timeouts++;
        else
            consecutive_timeouts = 0;

        printf("%d,%d,%.0f,%s,%d,%s\n", k, n_max, time_sec * 1000.0, engine, B,
               valid ? "ok" : "floor");
        fflush(stdout);
        fprintf(stderr, "  k=%d -> n_max=%d (%.0f ms) [%s B=%d]%s\n",
                k, n_max, time_sec * 1000.0, engine, B,
                valid ? "" : " FLOOR");
    }

    free(S); free(equity); free(payout);
}

/* ══════════════════════════════════════════════════════════════
   MODE 3: n=k THRESHOLD (binary search along the diagonal)
   ══════════════════════════════════════════════════════════════ */

static void run_nk_threshold(void) {
    double target_sec = 1.0;
    int max_n = 100000000;
    int init_alloc = 100000;

    nk_probe_ctx ctx = {
        .S_buf = (double *)malloc((size_t)init_alloc * sizeof(double)),
        .payout = (double *)malloc((size_t)init_alloc * sizeof(double)),
        .equity_buf = (double *)malloc((size_t)init_alloc * sizeof(double)),
        .alloc_n = init_alloc,
        .alloc_k = init_alloc,
    };
    if (!ctx.S_buf || !ctx.payout || !ctx.equity_buf) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    for (int q = 0; q < init_alloc; q++)
        ctx.payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

    fprintf(stderr, "Searching for max n=k under %.0fs...\n", target_sec);

    double time_sec;
    int n_threshold = bisect_threshold(nk_probe, &ctx, 64, 1024, max_n,
                                       target_sec, &time_sec);

    int B = icm_select_engine(n_threshold, n_threshold);
    const char *engine = (B > 0) ? "hybrid" : "linear";

    printf("n_eq_k,%d,%.0f,%s,%d\n", n_threshold, time_sec * 1000.0, engine, B);
    fprintf(stderr, "FRONTIER: n=k=%d  time=%.0f ms  [%s B=%d]\n",
            n_threshold, time_sec * 1000.0, engine, B);

    free(ctx.S_buf); free(ctx.payout); free(ctx.equity_buf);
}

/* ── Main ──────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    enum { MODE_HEATMAP, MODE_CONTOUR, MODE_NK } mode = MODE_HEATMAP;
    int measure_mem = 0;

    const char *wisdom_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--contour") == 0) mode = MODE_CONTOUR;
        else if (strcmp(argv[i], "--nk") == 0) mode = MODE_NK;
        else if (strcmp(argv[i], "--memory") == 0) measure_mem = 1;
        else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc)
            g_call_timeout = atof(argv[++i]);
        else if (strcmp(argv[i], "--wisdom") == 0 && i + 1 < argc)
            wisdom_path = argv[++i];
    }

    icm_init(wisdom_path);
    g_use_fork_timeout = 1;

    switch (mode) {
    case MODE_HEATMAP: run_heatmap(measure_mem); break;
    case MODE_CONTOUR: run_contour(); break;
    case MODE_NK:      run_nk_threshold(); break;
    }

    return 0;
}
