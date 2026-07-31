/* validate_best_b.c: Single-point probe: for a given (n,k), report the
 * cost-model choice (auto_B) vs. the empirically-fastest B (best_B) with
 * timing and gap.
 *
 * This is the "oracle" a later adaptive loop calls per probe; one point
 * at a time, fast, with machine-parseable output.
 *
 * Usage:
 *   validate_best_b <n> <k> [--config /path/to/fft_config.h]
 *
 * Output (one line to stdout, CSV with header prefix #):
 *   n,k,auto_B,auto_ms,best_B,best_ms,gap_pct
 *
 * Columns:
 *   n,k       : input parameters  (int)
 *   auto_B    : B chosen by icm_select_best_B(n,k)  (int)
 *   auto_ms   : adaptive median-of-N (3-15, cv<=3%) timing at auto_B, in ms  (double)
 *   best_B    : empirically-fastest B in {8,16,24,32,48,64}  (int)
 *   best_ms   : adaptive median-of-N (3-15, cv<=3%) timing at best_B, in ms  (double)
 *   gap_pct   : (auto_ms - best_ms) / best_ms * 100; 0.0 if auto_B == best_B
 *               or auto is faster  (double)
 *
 * All measurements: Q=256, srand(42), payout[m]=n-m, S[i]=100+9900*rand()/RAND_MAX,
 * matching bench_grid crossover and calibrate_best_b conventions exactly.
 *
 * Discovery strategy for best_B:
 *   Every candidate (and, in Phase 2, auto_B/best_B) is measured with an
 *   adaptive median-of-N: start at 3 reps, extend up to 15 until
 *   cv<=3%. Replaces an earlier "1-rep-rank, runoff only on the top-2"
 *   scheme, which had a real gap: a noise fluke that mis-ranked the true
 *   winner outside the top-2 was never corrected, since the runoff only
 *   ever compared the (already wrong) top-2 against each other. Found
 *   and fixed on the GPU side first (VERDICTS.md V6, a 281% regression
 *   on B200 traced to exactly this at sub-few-ms problem sizes); ported
 *   here for CPU/GPU parity even though CPU's narrower 6-candidate,
 *   8x-range B set makes it much less exposed in practice.
 *
 * Build (macOS M3 Pro):
 *   gcc -O3 -march=native -Isrc/cpu -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o build/validate_best_b tools/validate_best_b.c src/cpu/icm.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 * Build (Linux/Zen4, AOCL-FFTW):
 *   gcc -O3 -march=znver4 -Isrc/cpu -Idevices/zen4 -I/usr/local/aocl-fftw/include \
 *       -o build/validate_best_b tools/validate_best_b.c src/cpu/icm.c \
 *       -L/usr/local/aocl-fftw/lib -Wl,-rpath,/usr/local/aocl-fftw/lib \
 *       -lfftw3 -lm -ldl -lmvec
 */

#include "icm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define Q_PROBE        256
#define N_CANDIDATES   6
#define N_REPS_MIN     3
#define N_REPS_MAX     15
#define CONVERGE_CV    0.03

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
    /* Deliberately leak hc; this is a short-lived offline calibration
     * tool where getting EngineKind enum wrong for icm_ctx_destroy
     * would segfault. Leaking is safe; guessing the enum wrong isn't. */
    return t;
}

static double cv_of(const double *x, int count) {
    if (count < 2) return 0.0;
    double mean = 0.0;
    for (int i = 0; i < count; i++) mean += x[i];
    mean /= (double)count;
    if (mean <= 0.0) return 0.0;
    double var = 0.0;
    for (int i = 0; i < count; i++) {
        double d = x[i] - mean;
        var += d * d;
    }
    var /= (double)(count - 1);
    return sqrt(var) / mean;
}

/* ── Adaptive median-of-N timing ──────────────────────────────────
 * Start at N_REPS_MIN reps, extend up to N_REPS_MAX until cv<=3%.
 * Replaces the old fixed-count time_one()/time_median7() split: every
 * candidate (Phase 1) and the final auto/best comparison (Phase 2) now
 * go through this one converging measurement, closing the mis-ranking
 * gap described above the includes. */
static double time_adaptive(int n, int k, int B, const double *S,
                            const double *payout, double *equity) {
    double samples[N_REPS_MAX];
    int count = 0;
    for (; count < N_REPS_MIN; count++)
        samples[count] = time_one(n, k, B, S, payout, equity);
    while (count < N_REPS_MAX && cv_of(samples, count) > CONVERGE_CV)
        samples[count++] = time_one(n, k, B, S, payout, equity);
    qsort(samples, count, sizeof(double), cmp_double);
    return samples[count / 2];
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

    /* ── Phase 1: adaptive median-of-N per candidate to find best_B ──
     * Every candidate gets its own converged measurement (see
     * time_adaptive() above); no separate runoff needed since there's
     * no single noisy rep left to mis-rank. */
    double t1[N_CANDIDATES];
    int    valid[N_CANDIDATES];
    int    n_valid = 0;

    for (int bi = 0; bi < N_CANDIDATES; bi++) {
        int B = B_candidates[bi];
        if (B > n) { valid[bi] = 0; continue; }
        valid[bi] = 1;
        n_valid++;
        t1[bi] = time_adaptive(n, k, B, S, payout, equity);
    }

    if (n_valid == 0) {
        fprintf(stderr, "n=%d k=%d: no valid B candidates (n < smallest candidate B=%d)\n",
                n, k, B_candidates[0]);
        free(S); free(payout); free(equity);
        return 1;
    }

    int    best_idx = -1;
    double best_t = 1e18;
    for (int bi = 0; bi < N_CANDIDATES; bi++) {
        if (!valid[bi]) continue;
        if (t1[bi] < best_t) {
            best_t   = t1[bi];
            best_idx = bi;
        }
    }
    int best_B = B_candidates[best_idx];

    /* ── Phase 2: fresh adaptive measurement for final ms values ──── */
    double auto_ms = time_adaptive(n, k, auto_B, S, payout, equity);
    double best_ms = time_adaptive(n, k, best_B, S, payout, equity);

    /* gap: positive means auto_B is slower than best_B */
    double gap_pct = 0.0;
    if (best_ms > 0.0) {
        gap_pct = 100.0 * (auto_ms - best_ms) / best_ms;
        if (gap_pct < 0.0) gap_pct = 0.0;  /* auto was faster; no gap */
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
