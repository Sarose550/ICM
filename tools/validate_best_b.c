/* validate_best_b.c — Single-point probe: for a given (n,k), report the
 * cost-model choice (auto_B) vs. the empirically-fastest B (best_B) with
 * timing and gap.
 *
 * This is the "oracle" a later adaptive loop calls per probe — one point
 * at a time, fast, with machine-parseable output.
 *
 * Usage:
 *   validate_best_b <n> <k> [--config /path/to/fft_config.h]
 *
 * Output (one line to stdout, CSV with header prefix #):
 *   n,k,auto_B,auto_ms,best_B,best_ms,gap_pct
 *
 * Columns:
 *   n,k       — input parameters (int)
 *   auto_B    — B chosen by icm_select_best_B(n,k) (int)
 *   auto_ms   — median-of-7 timing of hybrid engine at auto_B, in ms (double)
 *   best_B    — empirically-fastest B in {8,16,24,32,48,64} (int)
 *   best_ms   — median-of-7 timing at best_B, in ms (double)
 *   gap_pct   — (auto_ms - best_ms) / best_ms * 100; 0.0 if auto_B == best_B
 *               or auto is faster (double)
 *
 * All measurements: Q=256, srand(42), payout[m]=n-m, S[i]=100+9900*rand()/RAND_MAX
 * — matching bench_grid crossover and calibrate_best_b conventions exactly.
 *
 * Discovery strategy for best_B:
 *   1 rep per candidate to rank, then 2 more reps on top-2 if within 3%,
 *   median of those 3 determines the winner. (Same as calibrate_best_b.)
 *   Then a fresh median-of-7 for both auto_B and best_B for the final
 *   reported ms values — ensures fair, low-noise head-to-head.
 *
 * Build (macOS M3 Pro):
 *   gcc -O3 -march=native -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o build/validate_best_b tools/validate_best_b.c src/icm.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 * Build (Linux/Zen4, AOCL-FFTW):
 *   gcc -O3 -march=znver4 -Isrc -Idevices/zen4 -I/usr/local/aocl-fftw/include \
 *       -o build/validate_best_b tools/validate_best_b.c src/icm.c \
 *       -L/usr/local/aocl-fftw/lib -Wl,-rpath,/usr/local/aocl-fftw/lib \
 *       -lfftw3 -lm -ldl -lmvec
 */

#include "icm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Q_PROBE        256
#define N_REPS_FINAL   7
#define N_CANDIDATES   6
#define RUNOFF_PCT     3.0

static const int B_candidates[N_CANDIDATES] = {8, 16, 24, 32, 48, 64};

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) ? 1 : (da < db) ? -1 : 0;
}

/* ── Single-rep timing ─────────────────────────────────────────── */

static double time_one(int n, int k, int B, const double *S,
                       const double *payout, double *equity) {
    void *hc = icm_hybrid_ctx_create(n, S, k, B);
    double t = icm_run_engine(n, S, Q_PROBE, payout, k, equity,
                               icm_engine_hybrid(), hc) / (double)Q_PROBE;
    /* Deliberately leak hc — this is a short-lived offline calibration
     * tool where getting EngineKind enum wrong for icm_ctx_destroy
     * would segfault. Leaking is safe; guessing the enum wrong isn't. */
    return t;
}

/* ── Median-of-7 timing ────────────────────────────────────────── */

static double time_median7(int n, int k, int B, const double *S,
                           const double *payout, double *equity) {
    double samples[N_REPS_FINAL];
    for (int r = 0; r < N_REPS_FINAL; r++) {
        samples[r] = time_one(n, k, B, S, payout, equity);
    }
    qsort(samples, N_REPS_FINAL, sizeof(double), cmp_double);
    return samples[N_REPS_FINAL / 2];
}

/* ── Main ───────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int n = 0, k = 0;
    const char *config_path = NULL;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--config")) {
            if (i + 1 < argc) config_path = argv[++i];
            else { fprintf(stderr, "--config requires a path\n"); return 1; }
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown flag: %s\n", argv[i]);
            return 1;
        } else if (n == 0) {
            n = atoi(argv[i]);
        } else if (k == 0) {
            k = atoi(argv[i]);
        } else {
            fprintf(stderr, "Extra argument: %s\n", argv[i]);
            return 1;
        }
    }

    if (n <= 0 || k <= 0 || k > n) {
        fprintf(stderr,
                "Usage: validate_best_b <n> <k> [--config /path/to/fft_config.h]\n"
                "  n > 0, 0 < k <= n\n");
        return 1;
    }

    /* ── Init ICM ─────────────────────────────────────────────── */
    /* --config is accepted for future use (in-progress candidate table);
     * currently icm_init(NULL) reads the compiled-in fft_config.h. */
    (void)config_path;
    icm_init(NULL);

    /* ── Allocate ─────────────────────────────────────────────── */
    double *S      = (double *)malloc(n * sizeof(double));
    double *payout = (double *)malloc(k * sizeof(double));
    double *equity = (double *)malloc(n * sizeof(double));
    if (!S || !payout || !equity) { fprintf(stderr, "OOM\n"); return 1; }

    /* Generate stacks: same convention as every other tool */
    srand(42);
    for (int i = 0; i < n; i++)
        S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);

    /* payout[m] = (n-m) */
    for (int m = 0; m < k; m++)
        payout[m] = (double)(n - m);

    /* ── auto_B from cost model ──────────────────────────────── */
    int auto_B = icm_select_best_B(n, k);

    /* ── Phase 1: 1 rep per candidate to find best_B ────────── */
    double t1[N_CANDIDATES];
    int    valid[N_CANDIDATES];
    int    n_valid = 0;

    for (int bi = 0; bi < N_CANDIDATES; bi++) {
        int B = B_candidates[bi];
        if (B > n) { valid[bi] = 0; continue; }
        valid[bi] = 1;
        n_valid++;
        t1[bi] = time_one(n, k, B, S, payout, equity);
    }

    if (n_valid == 0) {
        fprintf(stderr, "n=%d k=%d: no valid B candidates (n < smallest candidate B=%d)\n",
                n, k, B_candidates[0]);
        free(S); free(payout); free(equity);
        return 1;
    }

    /* Find top-2 */
    int    best_idx = -1, second_idx = -1;
    double best_t = 1e18, second_t = 1e18;

    for (int bi = 0; bi < N_CANDIDATES; bi++) {
        if (!valid[bi]) continue;
        if (t1[bi] < best_t) {
            second_t   = best_t;
            second_idx = best_idx;
            best_t     = t1[bi];
            best_idx   = bi;
        } else if (t1[bi] < second_t) {
            second_t   = t1[bi];
            second_idx = bi;
        }
    }

    int best_B;

    if (n_valid == 1) {
        best_B = B_candidates[best_idx];
    } else {
        /* Check if runoff needed */
        int do_runoff = 0;
        if (best_idx >= 0 && second_idx >= 0 && best_t > 0.0) {
            double pct = 100.0 * (second_t - best_t) / best_t;
            if (pct <= RUNOFF_PCT) do_runoff = 1;
        }

        if (do_runoff) {
            double a_s[3], b_s[3];
            a_s[0] = t1[best_idx];
            b_s[0] = t1[second_idx];
            for (int r = 1; r < 3; r++) {
                a_s[r] = time_one(n, k, B_candidates[best_idx], S, payout, equity);
                b_s[r] = time_one(n, k, B_candidates[second_idx], S, payout, equity);
            }
            qsort(a_s, 3, sizeof(double), cmp_double);
            qsort(b_s, 3, sizeof(double), cmp_double);
            best_B = (a_s[1] <= b_s[1]) ? B_candidates[best_idx]
                                        : B_candidates[second_idx];
        } else {
            best_B = B_candidates[best_idx];
        }
    }

    /* ── Phase 2: fresh median-of-7 for final ms values ──────── */
    double auto_ms = time_median7(n, k, auto_B, S, payout, equity);
    double best_ms = time_median7(n, k, best_B, S, payout, equity);

    /* gap: positive means auto_B is slower than best_B */
    double gap_pct = 0.0;
    if (best_ms > 0.0) {
        gap_pct = 100.0 * (auto_ms - best_ms) / best_ms;
        if (gap_pct < 0.0) gap_pct = 0.0;  /* auto was faster — no gap */
    }

    /* ── Machine-readable output to stdout ────────────────────
     * Format: n,k,auto_B,auto_ms,best_B,best_ms,gap_pct
     * auto_ms and best_ms are in NANOSECONDS per QP (ns/qp),
     * matching the convention used throughout the codebase.
     * To convert to milliseconds: divide by 1e6.
     * gap_pct is dimensionless (percentage).
     */
    printf("%d,%d,%d,%.6f,%d,%.6f,%.4f\n",
           n, k, auto_B, auto_ms, best_B, best_ms, gap_pct);
    fflush(stdout);

    /* Debug to stderr */
    fprintf(stderr, "[%d,%d] auto_B=%d (%.1f ns/qp) best_B=%d (%.1f ns/qp) gap=%.2f%%\n",
            n, k, auto_B, auto_ms, best_B, best_ms, gap_pct);

    free(S); free(payout); free(equity);
    return 0;
}
