/*
 * bench.c — ICM benchmark harness, correctness verification, and tuning tools
 *
 * This file includes icm.c directly (single compilation unit) so it can
 * access internal types (TreeCtx, HybridCtx, FFTCache, etc.) for per-engine
 * benchmarking and FFT profiling. This is the only file that does this —
 * all other tools link against libicm.a and use only the icm.h public API.
 *
 * Compile (serial, macOS / Apple Silicon):
 *   gcc -O3 -march=native -Wall -Wno-unused-variable -Wno-unused-function \
 *       -o bench bench/bench.c \
 *       -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 *
 * Compile (parallel, macOS with libomp):
 *   gcc -O3 -march=native -Wall -Wno-unused-variable -Wno-unused-function \
 *       -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
 *       -o bench bench/bench.c \
 *       -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -L/opt/homebrew/lib -L/opt/homebrew/opt/libomp/lib \
 *       -lfftw3 -lfftw3_threads -lm -framework Accelerate -lomp
 *
 * Run:
 *   ./bench              # full grid
 *   ./bench verify       # correctness only
 *   ./bench quick        # subset grid (for iteration)
 *   ./bench crossover    # linear vs hybrid crossover sweep
 *   ./bench cliff        # power-of-2 cliff test
 *   ./bench threshold    # binary search for 1-second boundary
 *   ./bench profile      # FFT overhead measurement + phase profiling
 */

/* Include the library source directly for access to internal types.
 * ICM_BENCH_INCLUDE gates functions in icm.c that are only needed by the
 * benchmark harness (not by libicm.a). */
#define ICM_BENCH_INCLUDE
#include "icm.c"

/* ══════════════════════════════════════════════════════════════
   FORMATTING
   ══════════════════════════════════════════════════════════════ */

/* Format a millisecond timing to 3 significant figures in fixed-point
 * notation (never scientific) — e.g. 3 -> "3.00", 16 -> "16.0", 4392 -> "4390". */
static void fmt_ms_3sf(double v, char *buf, size_t bufsz) {
    if (v <= 0) { snprintf(buf, bufsz, "0.00"); return; }
    int exp = (int)floor(log10(v));
    int decimals = 2 - exp;
    if (decimals >= 0) {
        snprintf(buf, bufsz, "%.*f", decimals, v);
    } else {
        double scale = pow(10, -decimals);
        snprintf(buf, bufsz, "%.0f", round(v / scale) * scale);
    }
}

/* Benchmark repetition convention */
#define BENCH_REPS 5
#define MEDIAN5(arr) do { qsort(arr, BENCH_REPS, sizeof(double), dbl_cmp); } while(0)

/* ══════════════════════════════════════════════════════════════
   STACKS GENERATION
   ══════════════════════════════════════════════════════════════ */

static void make_stacks(int n, int dist, double *S) {
    srand(42);
    switch (dist) {
    case 0:
        for (int i = 0; i < n; i++)
            S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
        break;
    case 1:
        S[0] = 10000;
        for (int i = 1; i < n; i++) S[i] = 1.0;
        break;
    case 2:
        for (int i = 0; i < n; i++)
            S[i] = pow(2.0, (double)i * 10.0 / n);
        break;
    case 3:
        for (int i = 0; i < n; i++) S[i] = 100.0;
        break;
    case 4:
        S[0] = 1e9;
        for (int i = 1; i < n; i++) S[i] = 1.0;
        break;
    }
}

/* ══════════════════════════════════════════════════════════════
   PROFILING MODE — time individual phases
   ══════════════════════════════════════════════════════════════ */

/* Measure per-call FFT overhead: time actual polymul calls vs calibrated FFT cost.
 * The difference (constant across sizes) is the overhead from plan lookup,
 * buffer copies, and result extraction not captured in calibration. */
static void measure_fft_overhead(void) {
    printf("=== FFT OVERHEAD MEASUREMENT ===\n");
    printf("Timing polymul_fft_cyclic vs polymul_modk (schoolbook) at small sizes.\n");
    printf("Overhead = measured_fft - calibrated_fft (constant per call).\n\n");

    int sizes[] = {16, 32, 64, 128, 256};
    int n_sizes = 5;
    int reps = 200000;

    /* Need an FFT cache with plans for these sizes */
    int plan_sizes[10];
    int n_plans = 0;
    for (int i = 0; i < n_sizes; i++) {
        int bfn, bwm;
        best_fft_config(sizes[i], &bfn, &bwm, 0);
        plan_sizes[n_plans++] = bfn;
    }
    FFTCache *fc = fft_cache_create_sizes(plan_sizes, n_plans);

    printf("%-6s %-8s %-8s %-10s %-10s %-10s\n",
           "cps", "school", "fft_act", "fft_calib", "overhead", "fft_size");

    for (int si = 0; si < n_sizes; si++) {
        int cps = sizes[si];
        int d = cps / 2;
        double *a = (double *)calloc(cps, sizeof(double));
        double *b = (double *)calloc(cps, sizeof(double));
        double *c = (double *)calloc(cps, sizeof(double));
        for (int i = 0; i <= d; i++) { a[i] = 1.0 + 0.01*i; b[i] = 1.0 - 0.01*i; }

        int bfn, bwm;
        best_fft_config(2 * d, &bfn, &bwm, 0);

        /* Time schoolbook */
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            polymul_modk(a, cps, b, cps, c, cps);
        }
        double school_ns = (now_ns() - t0) / reps;

        /* Time FFT cyclic */
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            polymul_fft_cyclic(a, cps, b, cps, c, cps, fc, bfn, bwm);
        }
        double fft_ns = (now_ns() - t0) / reps;

        /* Calibrated FFT time */
        int lo=0, hi=N_CALIBRATED_SIZES-1;
        while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<bfn)lo=m+1;else hi=m;}
        double calib_ns = calib_times_ns[lo];

        printf("%-6d %-8.0f %-8.0f %-10.0f %-10.0f %-10d\n",
               cps, school_ns, fft_ns, calib_ns, fft_ns - calib_ns, bfn);

        free(a); free(b); free(c);
    }
    fft_cache_destroy(fc);
    printf("\n");
}

/* Measure the cost split between FFT phases: fwd, pointwise, ifft, memcpy.
 * Reports each phase as a fraction of the full calibrated pipeline time. */
static void measure_phase_split(void) {
    printf("=== FFT PHASE SPLIT ===\n");
    printf("Measuring fwd(r2c), pointwise, ifft(c2r), and memcpy fractions.\n\n");

    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int n_sizes = 9;

    /* Collect all unique FFT sizes needed */
    int plan_sizes[16];
    int n_plans = 0;
    for (int i = 0; i < n_sizes; i++) {
        int s = next_fftw_size(sizes[i]);
        int dup = 0;
        for (int j = 0; j < n_plans; j++) if (plan_sizes[j] == s) { dup = 1; break; }
        if (!dup) plan_sizes[n_plans++] = s;
    }
    /* Sort */
    for (int i = 0; i < n_plans; i++)
        for (int j = i+1; j < n_plans; j++)
            if (plan_sizes[j] < plan_sizes[i]) { int t=plan_sizes[i]; plan_sizes[i]=plan_sizes[j]; plan_sizes[j]=t; }
    FFTCache *fc = fft_cache_create_sizes(plan_sizes, n_plans);

    printf("%-8s %-8s %-8s %-8s %-8s %-8s %-8s %-6s %-6s %-6s\n",
           "fft_n", "fwd", "pw", "ifft", "memcpy", "sum", "calib",
           "f_fwd", "f_pw", "f_ifft");

    for (int si = 0; si < n_sizes; si++) {
        int fft_n = next_fftw_size(sizes[si]);
        FFTPlan *plan = fft_cache_get(fc, fft_n);
        if (!plan) continue;
        int actual_n = plan->fft_n;
        int cn = actual_n / 2 + 1;

        /* Allocate test data */
        double *rbuf_a = fftw_malloc(actual_n * sizeof(double));
        fftw_complex *cbuf_a = fftw_malloc(cn * sizeof(fftw_complex));
        double *rbuf_b = fftw_malloc(actual_n * sizeof(double));
        fftw_complex *cbuf_b = fftw_malloc(cn * sizeof(fftw_complex));
        for (int i = 0; i < actual_n; i++) { rbuf_a[i] = 1.0 + 0.001*i; rbuf_b[i] = 1.0 - 0.001*i; }

        /* Scale reps to keep each measurement ~100ms */
        int reps = (int)(1e8 / (double)actual_n);
        if (reps < 100) reps = 100;
        if (reps > 500000) reps = 500000;

        /* Warm up */
        for (int r = 0; r < 10; r++) {
            memcpy(plan->rbuf, rbuf_a, actual_n * sizeof(double));
            fftw_execute(plan->fwd_plan);
            fftw_execute(plan->inv_plan);
        }

        /* Time forward FFT (r2c) */
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(plan->rbuf, rbuf_a, actual_n * sizeof(double));
            fftw_execute(plan->fwd_plan);
        }
        double fwd_plus_cpy = (now_ns() - t0) / reps;

        /* Time inverse FFT (c2r) */
        /* First do a forward to fill cbuf with valid data */
        memcpy(plan->rbuf, rbuf_a, actual_n * sizeof(double));
        fftw_execute(plan->fwd_plan);
        memcpy(cbuf_a, plan->cbuf, cn * sizeof(fftw_complex));
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(plan->cbuf, cbuf_a, cn * sizeof(fftw_complex));
            fftw_execute(plan->inv_plan);
        }
        double ifft_plus_cpy = (now_ns() - t0) / reps;

        /* Time pointwise complex multiply */
        memcpy(plan->rbuf, rbuf_b, actual_n * sizeof(double));
        fftw_execute_dft_r2c(plan->fwd_plan, plan->rbuf, cbuf_b);
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (int i = 0; i < cn; i++) {
                double re = cbuf_a[i][0] * cbuf_b[i][0] - cbuf_a[i][1] * cbuf_b[i][1];
                double im = cbuf_a[i][0] * cbuf_b[i][1] + cbuf_a[i][1] * cbuf_b[i][0];
                plan->cbuf[i][0] = re;
                plan->cbuf[i][1] = im;
            }
        }
        double pw_ns = (now_ns() - t0) / reps;

        /* Time memcpy alone (to subtract from fwd/ifft measurements) */
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(plan->rbuf, rbuf_a, actual_n * sizeof(double));
        }
        double cpy_r_ns = (now_ns() - t0) / reps;

        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(plan->cbuf, cbuf_a, cn * sizeof(fftw_complex));
        }
        double cpy_c_ns = (now_ns() - t0) / reps;

        double fwd_ns = fwd_plus_cpy - cpy_r_ns;
        double ifft_ns = ifft_plus_cpy - cpy_c_ns;
        double sum_ns = fwd_ns + pw_ns + ifft_ns;

        /* Calibrated time */
        int lo=0, hi=N_CALIBRATED_SIZES-1;
        while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<actual_n)lo=m+1;else hi=m;}
        double calib_ns = calib_times_ns[lo];

        printf("%-8d %-8.0f %-8.0f %-8.0f %-8.0f %-8.0f %-8.0f %-6.2f %-6.2f %-6.2f\n",
               actual_n, fwd_ns, pw_ns, ifft_ns, cpy_r_ns, sum_ns, calib_ns,
               fwd_ns / sum_ns, pw_ns / sum_ns, ifft_ns / sum_ns);

        fftw_free(rbuf_a); fftw_free(cbuf_a);
        fftw_free(rbuf_b); fftw_free(cbuf_b);
    }
    fft_cache_destroy(fc);
    printf("\nf_fwd/f_pw/f_ifft = fraction of (fwd+pw+ifft) sum, excluding memcpy.\n");
    printf("Paired cached correlate cost = fwd + 2×(pw+ifft) = (f_fwd + 2×(f_pw+f_ifft)) × sum.\n\n");
}

static void run_profile(void) {
    printf("=== PHASE PROFILING ===\n\n");
    int Q = 256;

    typedef struct { int n; int k; } NK;
    NK cases[] = {
        {256,  100}, {256,  256},
        {1024, 100}, {1024, 256}, {1024, 1024},
        {4096, 100}, {4096, 1024}, {4096, 4096},
        /* Ragged tree comparison: power-of-2 vs non-power-of-2 */
        {8192, 8192}, {10000, 10000},
        {8192, 100},  {10000, 100},
    };
    int n_cases = sizeof(cases) / sizeof(cases[0]);

    for (int ci = 0; ci < n_cases; ci++) {
        int n = cases[ci].n, k = cases[ci].k;
        printf("--- n=%d, k=%d ---\n", n, k);

        double *S = (double *)malloc(n * sizeof(double));
        make_stacks(n, 0, S);
        double *payout = (double *)malloc(k * sizeof(double));
        for (int m = 0; m < k; m++) payout[m] = (double)(n - m);
        double *eq = (double *)calloc(n, sizeof(double));

        /* Tree */
        {
            TreeCtx *tc = tree_ctx_create(n, k);
            double t = run_engine_ctx(n, S, Q, payout, k, eq,
                                      engine_tree_ctx, tc) / 1e6;
            char tbuf[32]; fmt_ms_3sf(t, tbuf, sizeof(tbuf));
            printf("  tree:   %7s ms\n", tbuf);
            tree_ctx_destroy(tc);
        }

        /* Hybrid B=8 */
        {
            HybridCtx *hctx = hybrid_ctx_create(n, S, k, select_best_B(n, k));
            memset(eq, 0, n * sizeof(double));
            double t = run_engine_ctx(n, S, Q, payout, k, eq,
                                      engine_hybrid_ctx, hctx) / 1e6;
            char tbuf[32]; fmt_ms_3sf(t, tbuf, sizeof(tbuf));
            printf("  hyb8:  %7s ms\n", tbuf);
            hybrid_ctx_destroy(hctx);
        }

        /* Linear (always batched BQ=8 — matches icm_equity() behavior) */
        if ((double)n * k < 1e8) {
            LinearCtx *lc = linear_ctx_create(n, k);
            memset(eq, 0, n * sizeof(double));
            double t = run_linear_batched(n, S, Q, payout, k, eq, lc) / 1e6;
            char tbuf[32]; fmt_ms_3sf(t, tbuf, sizeof(tbuf));
            printf("  linear: %7s ms\n", tbuf);
            linear_ctx_destroy(lc);
        }

        /* Naive (small n only) */
        if ((double)n * n < 5e7) {
            NaiveCtx *nc = naive_ctx_create(n, k);
            memset(eq, 0, n * sizeof(double));
            double t = run_engine_ctx(n, S, Q, payout, k, eq,
                                      engine_naive_ctx, nc) / 1e6;
            char tbuf[32]; fmt_ms_3sf(t, tbuf, sizeof(tbuf));
            printf("  naive:  %7s ms\n", tbuf);
            naive_ctx_destroy(nc);
        }

        printf("\n");
        free(S); free(payout); free(eq);
    }
}

/* ══════════════════════════════════════════════════════════════
   MAIN
   ══════════════════════════════════════════════════════════════ */

static int dbl_cmp(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    int verify_only  = (argc > 1 && strcmp(argv[1], "verify") == 0);
    int quick        = (argc > 1 && strcmp(argv[1], "quick") == 0);
    int profile      = (argc > 1 && strcmp(argv[1], "profile") == 0);
    int crossover    = (argc > 1 && strcmp(argv[1], "crossover") == 0);
    int threshold    = (argc > 1 && strcmp(argv[1], "threshold") == 0);
    int single       = (argc > 1 && strcmp(argv[1], "bench") == 0);
    int subset_speed = (argc > 1 && strcmp(argv[1], "subset-speed") == 0);

#ifdef _OPENMP
    /* Make FFTW planner thread-safe before any plan creation */
    fftw_make_planner_thread_safe();
    /* Set default thread count */
    if (!getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(OMP_NUM_THREADS_DEFAULT);
    printf("OpenMP enabled: %d threads\n", omp_get_max_threads());
#else
    printf("OpenMP disabled (serial mode)\n");
#endif

    build_fftw_size_table();
    wisdom_load();

    if (profile) { measure_fft_overhead(); measure_phase_split(); run_profile(); return 0; }

    /* ── Single (n, k) benchmark: ./bench_grid bench <n> <k> [reps] ── */
    if (single) {
        if (argc < 4) { printf("Usage: bench_grid bench <n> <k> [reps]\n"); return 1; }
        int n = atoi(argv[2]), k = atoi(argv[3]);
        int reps = (argc > 4) ? atoi(argv[4]) : 5;
        int Q = 256;
        double *S = (double *)malloc(n * sizeof(double));
        srand(42);
        for (int i = 0; i < n; i++) S[i] = 1.0 + (double)rand() / (double)RAND_MAX;
        double *payout = (double *)malloc(k * sizeof(double));
        for (int m = 0; m < k; m++) {
            double v = 1.0;
            for (int j = 0; j < m; j++) v *= (double)(n - 1 - j) / (n - j);
            payout[m] = v;
        }
        double *equity = (double *)calloc(n, sizeof(double));

        /* Warmup */
        icm_equity(n, S, Q, payout, k, equity);

        double best = 1e18;
        for (int r = 0; r < reps; r++) {
            double t0 = now_ns();
            icm_equity(n, S, Q, payout, k, equity);
            double t = now_ns() - t0;
            double ms = t / 1e6;
            if (ms < best) best = ms;
            printf("  run %d: %.1f ms\n", r + 1, ms);
        }
        printf("best: %.1f ms (n=%d k=%d Q=%d)\n", best, n, k, Q);
        free(S); free(payout); free(equity);
        return 0;
    }

    /* ── Crossover sweep: linear vs hybrid at fine k granularity ── */
    if (crossover) {
        int Q = 256;
        int sweep_ns[] = {512, 1024, 2048, 4096, 8192};
        int n_sweep = 5;
        /* Coarse coverage across full range + fine bracket around L→H
         * transition.  Probe on Zen4 (2025-07-22) showed transition at
         * k≈275 for all n∈{512,1024,2048,4096,8192}.  This single shared
         * k-list brackets the transition tightly for all n rows:
         *   - Coarse below: 40, 80, 120, 160, 200
         *   - Fine bracket:  240…340 (every 10-20)
         *   - Coarse above: 400, 500, 750, 1000, 1500, 2000
         */
        int sweep_ks[] = {40, 80, 120, 160, 200, 240, 260, 270, 280, 290,
                          300, 310, 320, 340, 400, 500, 750, 1000, 1500, 2000};
        int n_ks = 20;

        printf("=== LINEAR vs HYBRID CROSSOVER SWEEP (ms, Q=%d) ===\n\n", Q);
        printf("%-6s", "n\\k");
        for (int ki = 0; ki < n_ks; ki++) printf("  k=%-5d", sweep_ks[ki]);
        printf("\n");
        for (int i = 0; i < 6 + n_ks * 8; i++) printf("-");
        printf("\n");

        for (int ni = 0; ni < n_sweep; ni++) {
            int n = sweep_ns[ni];
            double *S = (double *)malloc(n * sizeof(double));
            make_stacks(n, 0, S);
            printf("%-6d", n);
            fflush(stdout);

            for (int ki = 0; ki < n_ks; ki++) {
                int k = sweep_ks[ki];
                if (k > n) k = n;
                double *payout = (double *)malloc(k * sizeof(double));
                for (int m = 0; m < k; m++) payout[m] = (double)(n - m);
                double *eq = (double *)calloc(n, sizeof(double));

                /* Linear (always batched) */
                LinearCtx *lc = linear_ctx_create(n, k);
                double t_lin = run_linear_batched(n, S, Q, payout, k, eq, lc) / 1e6;
                linear_ctx_destroy(lc);

                /* Hybrid */
                HybridCtx *hc = hybrid_ctx_create(n, S, k, select_best_B(n, k));
                memset(eq, 0, n * sizeof(double));
                double t_hyb = run_engine_ctx(n, S, Q, payout, k, eq,
                                              engine_hybrid_ctx, hc) / 1e6;
                hybrid_ctx_destroy(hc);

                char cell[16];
                if (t_lin <= t_hyb)
                    snprintf(cell, sizeof(cell), "L%-3.0f", t_lin);
                else
                    snprintf(cell, sizeof(cell), "H%-3.0f", t_hyb);
                printf("  %-7s", cell);
                fflush(stdout);

                free(payout); free(eq);
            }
            printf("\n");
            free(S);
        }
        printf("\nL=linear wins, H=hybrid wins. Crossover: where L→H transition occurs.\n");
        return 0;
    }

    /* ── Threshold search: binary search for largest n where k=n < 1 second ── */
    if (threshold) {
        int Q = 256;
        double target_ms = 1000.0;
        printf("=== BINARY SEARCH: largest n where k=n < %.0fms (Q=%d, single-threaded) ===\n\n", target_ms, Q);

        int lo = 8192, hi = 32768;
        /* First bracket: find upper bound */
        while (1) {
            double *S = (double *)malloc(hi * sizeof(double));
            make_stacks(hi, 0, S);
            double *payout = (double *)malloc(hi * sizeof(double));
            for (int m = 0; m < hi; m++) payout[m] = (double)(hi - m);
            double *eq = (double *)calloc(hi, sizeof(double));
            HybridCtx *hc = hybrid_ctx_create(hi, S, hi, select_best_B(hi, hi));
            double t = run_engine_ctx(hi, S, Q, payout, hi, eq, engine_hybrid_ctx, hc) / 1e6;
            hybrid_ctx_destroy(hc);
            free(S); free(payout); free(eq);
            printf("  n=%-6d k=n  → %.0f ms\n", hi, t);
            if (t >= target_ms) break;
            lo = hi;
            hi *= 2;
        }

        /* Binary search */
        while (hi - lo > 256) {
            int mid = ((lo + hi) / 2) & ~0xF;  /* round to 16 */
            double *S = (double *)malloc(mid * sizeof(double));
            make_stacks(mid, 0, S);
            double *payout = (double *)malloc(mid * sizeof(double));
            for (int m = 0; m < mid; m++) payout[m] = (double)(mid - m);
            double *eq = (double *)calloc(mid, sizeof(double));
            HybridCtx *hc = hybrid_ctx_create(mid, S, mid, select_best_B(mid, mid));
            double t = run_engine_ctx(mid, S, Q, payout, mid, eq, engine_hybrid_ctx, hc) / 1e6;
            hybrid_ctx_destroy(hc);
            free(S); free(payout); free(eq);
            printf("  n=%-6d k=n  → %.0f ms\n", mid, t);
            if (t < target_ms) lo = mid; else hi = mid;
        }
        printf("\nResult: largest n with k=n < 1s is approximately n=%d\n", lo);
        return 0;
    }

    /* ── Cliff test: power-of-2 vs non-power-of-2 scaling ── */
    if (argc > 1 && strcmp(argv[1], "cliff") == 0) {
        int Q = 256;
        int test_ns[] = {8192, 16384, 32768, 65536, 131072};
        int n_tests = 5;
        printf("=== POWER-OF-2 CLIFF TEST (ms, Q=%d, k=n) ===\n\n", Q);
        printf("%-8s %-8s %-10s\n", "n", "ms", "ratio_to_8192");
        double base = 0;
        for (int i = 0; i < n_tests; i++) {
            int n = test_ns[i];
            double *S = (double *)malloc(n * sizeof(double));
            make_stacks(n, 0, S);
            double *payout = (double *)malloc(n * sizeof(double));
            for (int m = 0; m < n; m++) payout[m] = (double)(n - m);
            double *eq = (double *)calloc(n, sizeof(double));
            HybridCtx *hc = hybrid_ctx_create(n, S, n, select_best_B(n, n));
            double t = run_engine_ctx(n, S, Q, payout, n, eq, engine_hybrid_ctx, hc) / 1e6;
            hybrid_ctx_destroy(hc);
            free(S); free(payout); free(eq);
            if (i == 0) base = t;
            printf("%-8d %-8.0f %.2fx\n", n, t, t / base);
            fflush(stdout);
        }
        return 0;
    }

    /* ── Subset speed benchmark: icm_equity_subset vs full icm_equity ── */
    if (subset_speed) {
        int Q = 256;
        int ns[] = {1024, 4096, 16384, 65536};
        int n_ns = 4;
        double ratios[] = {0.01, 0.05, 0.10, 0.25, 0.50, 1.00};
        int n_ratios = 6;
        const char *ratio_labels[] = {"1%", "5%", "10%", "25%", "50%", "100%"};

        printf("=== SUBSET SPEED BENCHMARK (Q=%d, k=n/4, median of %d runs) ===\n\n",
               Q, BENCH_REPS);
        printf("%-8s %-8s %-8s %-12s %-12s %-10s\n",
               "n", "n_tgts", "ratio", "subset_ms", "full_ms", "speedup");
        for (int i = 0; i < 60; i++) printf("-");
        printf("\n");

        for (int ni = 0; ni < n_ns; ni++) {
            int n = ns[ni];
            int k = n / 4;
            if (k < 1) k = 1;

            double *S = (double *)malloc(n * sizeof(double));
            make_stacks(n, 0, S);
            double *payout = (double *)malloc(k * sizeof(double));
            for (int m = 0; m < k; m++) payout[m] = (double)(n - m);
            double *equity = (double *)calloc(n, sizeof(double));

            /* Time full icm_equity (same n,k for all ratios — done once per n) */
            /* warmup */
            icm_equity(n, S, Q, payout, k, equity);
            double full_samples[BENCH_REPS];
            for (int r = 0; r < BENCH_REPS; r++) {
                double t0 = now_ns();
                icm_equity(n, S, Q, payout, k, equity);
                full_samples[r] = (now_ns() - t0) / 1e6;
            }
            MEDIAN5(full_samples);
            double full_ms = full_samples[BENCH_REPS / 2];

            for (int ri = 0; ri < n_ratios; ri++) {
                int n_targets = (int)(n * ratios[ri]);
                if (n_targets < 1) n_targets = 1;

                /* Evenly-spaced target indices for deterministic coverage */
                int *targets = (int *)malloc(n_targets * sizeof(int));
                for (int t = 0; t < n_targets; t++) {
                    targets[t] = (int)((double)t * n / n_targets);
                }

                /* Time subset */
                /* warmup */
                icm_equity_subset(n, S, Q, payout, k, equity, targets, n_targets);
                double sub_samples[BENCH_REPS];
                for (int r = 0; r < BENCH_REPS; r++) {
                    double t0 = now_ns();
                    icm_equity_subset(n, S, Q, payout, k, equity,
                                      targets, n_targets);
                    sub_samples[r] = (now_ns() - t0) / 1e6;
                }
                MEDIAN5(sub_samples);
                double sub_ms = sub_samples[BENCH_REPS / 2];

                double speedup = full_ms / sub_ms;

                char sbuf[32], fbuf[32];
                fmt_ms_3sf(sub_ms, sbuf, sizeof(sbuf));
                fmt_ms_3sf(full_ms, fbuf, sizeof(fbuf));

                printf("%-8d %-8d %-8s %-12s %-12s %-10.2fx\n",
                       n, n_targets, ratio_labels[ri], sbuf, fbuf, speedup);
                fflush(stdout);

                free(targets);
            }
            printf("\n");
            free(S); free(payout); free(equity);
        }
        return 0;
    }

    int Q = 256;
    const char *dist_names[] = {"uniform", "adversarial", "geometric", "equal", "adv_1e9"};

    /* ── Correctness verification ─────────────────────────── */
    printf("=== CORRECTNESS VERIFICATION ===\n\n");

    int verify_ns_quick[] = {16, 32, 64, 128, 256, 1024, 4096};
    int verify_ns_full[]  = {16, 32, 64, 128, 256, 1024, 4096, 16384, 65536};
    int *verify_ns = verify_only ? verify_ns_full : verify_ns_quick;
    int n_vn = verify_only ? 9 : 7;

    int all_pass = 1;

    for (int ni = 0; ni < n_vn; ni++) {
        int n = verify_ns[ni];
        int n_dists = (n <= 256) ? 5 : 3;
        /* At large n, test: uniform, adversarial(1e4), adv_1e9 */
        int large_n_dists[] = {0, 1, 4};
        for (int dii = 0; dii < n_dists; dii++) {
            int di = (n <= 256) ? dii : large_n_dists[dii];
            double *S = (double *)malloc(n * sizeof(double));
            make_stacks(n, di, S);

            double *payout = (double *)malloc(n * sizeof(double));
            for (int m = 0; m < n; m++) payout[m] = (double)(n - m);

            double *v1 = (double *)malloc(n * sizeof(double));
            v1_exact(n, S, v1);

            /* Test each engine against V1 at k=n.
             * Skip naive for n > 256 (O(n^2) — V1 closed form is the reference).
             * Skip linear for n > 4096 (O(nk) at k=n is very slow). */
            typedef struct { const char *name; EquityEngine fn; void *ctx; } VE;
            TreeCtx *tc = tree_ctx_create(n, n);
            NaiveCtx *nc = (n <= 256) ? naive_ctx_create(n, n) : NULL;
            LinearCtx *lc = (n <= 4096) ? linear_ctx_create(n, n) : NULL;
            HybridCtx *hc = hybrid_ctx_create(n, S, n, select_best_B(n, n));
            VE ves[4];
            int n_eng = 0;
            ves[n_eng++] = (VE){"tree",   engine_tree_ctx,   tc};
            if (nc) ves[n_eng++] = (VE){"naive",  engine_naive_ctx,  nc};
            if (lc) ves[n_eng++] = (VE){"linear", engine_linear_ctx, lc};
            ves[n_eng++] = (VE){"hyb8",   engine_hybrid_ctx, hc};

            for (int ei = 0; ei < n_eng; ei++) {
                double *eq = (double *)calloc(n, sizeof(double));
                run_engine_ctx(n, S, Q, payout, n, eq, ves[ei].fn, ves[ei].ctx);
                double max_rel = 0;
                for (int i = 0; i < n; i++) {
                    if (fabs(v1[i]) < 1e-10) continue;
                    double r = fabs(eq[i] - v1[i]) / fabs(v1[i]);
                    if (r > max_rel) max_rel = r;
                }
                double tol = (di == 4) ? 1e-9 : 5e-12;
                int pass = (max_rel < tol);
                if (!pass) all_pass = 0;
                printf("%-6s n=%-4d %-12s %-8s err=%.2e  %s\n",
                       ves[ei].name, n, dist_names[di],
                       pass ? "PASS" : "FAIL", max_rel,
                       pass ? "" : "!!!");
                free(eq);
            }

            /* Cross-check tree vs linear for k < n */
            int k = (n > 20) ? 10 : n;
            double *pay_k = (double *)malloc(k * sizeof(double));
            for (int m = 0; m < k; m++) pay_k[m] = (double)(n - m);

            TreeCtx *tc2 = tree_ctx_create(n, k);
            LinearCtx *lc2 = linear_ctx_create(n, k);
            double *eq_tree = (double *)calloc(n, sizeof(double));
            double *eq_lin  = (double *)calloc(n, sizeof(double));
            run_engine_ctx(n, S, Q, pay_k, k, eq_tree, engine_tree_ctx, tc2);
            run_engine_ctx(n, S, Q, pay_k, k, eq_lin, engine_linear_ctx, lc2);

            double max_diff = 0;
            for (int i = 0; i < n; i++) {
                double sc = fabs(eq_tree[i]) > 1e-10 ? fabs(eq_tree[i]) : 1;
                double d = fabs(eq_tree[i] - eq_lin[i]) / sc;
                if (d > max_diff) max_diff = d;
            }
            /* Cross-check threshold: 1e-13 for small n, looser for large n
             * (different evaluation orders accumulate different FP rounding) */
            double xchk_tol = (n <= 4096) ? 1e-13 : 1e-11;
            int kpass = (max_diff < xchk_tol);
            if (!kpass) all_pass = 0;
            printf("%-6s n=%-4d %-12s %-8s diff=%.2e  %s\n",
                   "xchk", n, dist_names[di],
                   kpass ? "PASS" : "FAIL", max_diff,
                   kpass ? "" : "!!!");

            /* V2 test: quadratic payout p[m] = C(n-1-m, 2), O(n^3) reference */
            if (n <= 256) {
                double *v2 = (double *)malloc(n * sizeof(double));
                v2_exact(n, S, v2);
                double *pay_v2 = (double *)malloc(n * sizeof(double));
                for (int m = 0; m < n; m++) {
                    int r = n - 1 - m; /* players remaining after position m */
                    pay_v2[m] = (r >= 2) ? (double)(r * (r - 1)) / 2.0 : 0;
                }
                /* Test tree and hybrid against V2 */
                TreeCtx *tc_v2 = tree_ctx_create(n, n);
                double *eq_v2 = (double *)calloc(n, sizeof(double));
                run_engine_ctx(n, S, Q, pay_v2, n, eq_v2,
                               engine_tree_ctx, tc_v2);
                double max_rel_v2 = 0;
                for (int i = 0; i < n; i++) {
                    if (fabs(v2[i]) < 1e-10) continue;
                    double r = fabs(eq_v2[i] - v2[i]) / fabs(v2[i]);
                    if (r > max_rel_v2) max_rel_v2 = r;
                }
                double v2tol = (di == 4) ? 1e-9 : 5e-12;
                int v2pass = (max_rel_v2 < v2tol);
                if (!v2pass) all_pass = 0;
                printf("%-6s n=%-4d %-12s %-8s err=%.2e  %s\n",
                       "V2", n, dist_names[di],
                       v2pass ? "PASS" : "FAIL", max_rel_v2,
                       v2pass ? "" : "!!!");
                tree_ctx_destroy(tc_v2);
                free(v2); free(pay_v2); free(eq_v2);
            }

            tree_ctx_destroy(tc);
            if (nc) naive_ctx_destroy(nc);
            if (lc) linear_ctx_destroy(lc);
            hybrid_ctx_destroy(hc);
            tree_ctx_destroy(tc2); linear_ctx_destroy(lc2);
            free(S); free(payout); free(v1);
            free(pay_k); free(eq_tree); free(eq_lin);
        }
    }

    /* Subset API test: compute_equity_subset should match full computation */
    {
        int n = 1024, k_sub = 100;
        double *S = (double *)malloc(n * sizeof(double));
        make_stacks(n, 0, S);
        double *payout = (double *)malloc(k_sub * sizeof(double));
        for (int m = 0; m < k_sub; m++) payout[m] = (double)(n - m);

        /* Full computation via public API (uses same dispatch as subset) */
        double *eq_full = (double *)calloc(n, sizeof(double));
        icm_equity(n, S, Q, payout, k_sub, eq_full);

        /* Subset: 10 random players */
        int targets[] = {0, 7, 42, 100, 255, 500, 777, 900, 999, 1023};
        int n_targets = 10;
        double *eq_sub = (double *)calloc(n, sizeof(double));
        compute_equity_subset(n, S, Q, payout, k_sub, eq_sub, targets, n_targets);

        double max_diff = 0;
        for (int i = 0; i < n_targets; i++) {
            int p = targets[i];
            double sc = fabs(eq_full[p]) > 1e-10 ? fabs(eq_full[p]) : 1;
            double d = fabs(eq_full[p] - eq_sub[p]) / sc;
            if (d > max_diff) max_diff = d;
        }
        int spass = (max_diff < 1e-13);
        if (!spass) all_pass = 0;
        printf("%-6s n=%-4d %-12s %-8s diff=%.2e  %s\n",
               "subset", n, "10/1024", spass ? "PASS" : "FAIL", max_diff,
               spass ? "" : "!!!");
        free(S); free(payout); free(eq_full); free(eq_sub);
    }

    printf("\n%s\n\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED !!!");

    if (verify_only) return all_pass ? 0 : 1;

    /* ── Performance grid ─────────────────────────────────── */
    printf("=== PERFORMANCE GRID (ms, Q=%d, uniform stacks) ===\n\n", Q);

    int bench_ns_quick[] = {256, 1024, 4096, 8192};
    int bench_ns_full[]  = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int *bench_ns = quick ? bench_ns_quick : bench_ns_full;
    int n_bn = quick ? 4 : 11;

    typedef struct { const char *label; double frac; int fixed; } KSpec;
    KSpec kspecs[] = {
        {"k=10",  0, 10},
        {"k=50",  0, 50},
        {"k=100", 0, 100},
        {"k=n/4", 0.25, 0},
        {"k=n/2", 0.5, 0},
        {"k=n",   1.0, 0},
    };
    int n_ks = 6;

    printf("%-6s", "n");
    for (int ki = 0; ki < n_ks; ki++) printf("  %-34s", kspecs[ki].label);
    printf("\n");
    for (int i = 0; i < 6 + n_ks * 36; i++) printf("-");
    printf("\n");
    printf("(median of %d runs per engine per cell)\n\n", BENCH_REPS);

    for (int ni = 0; ni < n_bn; ni++) {
        int n = bench_ns[ni];
        double *S = (double *)malloc(n * sizeof(double));
        make_stacks(n, 0, S);

        printf("%-6d", n);
        fflush(stdout);

        for (int ki = 0; ki < n_ks; ki++) {
            int k = kspecs[ki].fixed ? kspecs[ki].fixed : (int)(kspecs[ki].frac * n);
            if (k > n) k = n;
            if (k < 1) k = 1;

            double *payout = (double *)malloc(k * sizeof(double));
            for (int m = 0; m < k; m++) payout[m] = (double)(n - m);
            double *eq = (double *)calloc(n, sizeof(double));

            double times[4] = {-1, -1, -1, -1};
            const char *names[4] = {"T", "N", "L", "H"};

            /* Tree (pure, for comparison) — median of BENCH_REPS */
            if ((double)n * k < 5e9) {
                TreeCtx *tc = tree_ctx_create(n, k);
                double samples[BENCH_REPS];
                /* warmup */
                memset(eq, 0, n * sizeof(double));
                run_engine_ctx(n, S, Q, payout, k, eq, engine_tree_ctx, tc);
                for (int r = 0; r < BENCH_REPS; r++) {
                    memset(eq, 0, n * sizeof(double));
                    samples[r] = run_engine_ctx(n, S, Q, payout, k, eq,
                                                engine_tree_ctx, tc) / 1e6;
                }
                MEDIAN5(samples);
                times[0] = samples[BENCH_REPS / 2];
                tree_ctx_destroy(tc);
            }

            /* Hybrid — median of BENCH_REPS */
            if ((double)n * k < 5e9 && n >= 16) {
                HybridCtx *hctx = hybrid_ctx_create(n, S, k, select_best_B(n, k));
                double samples[BENCH_REPS];
                /* warmup */
                memset(eq, 0, n * sizeof(double));
                run_engine_ctx(n, S, Q, payout, k, eq, engine_hybrid_ctx, hctx);
                for (int r = 0; r < BENCH_REPS; r++) {
                    memset(eq, 0, n * sizeof(double));
                    samples[r] = run_engine_ctx(n, S, Q, payout, k, eq,
                                                engine_hybrid_ctx, hctx) / 1e6;
                }
                MEDIAN5(samples);
                times[3] = samples[BENCH_REPS / 2];
                hybrid_ctx_destroy(hctx);
            }

            /* Linear: only where it could plausibly beat hybrid (k ≤ 200 or n ≤ 512) */
            if (k <= 200 || n <= 512) {
                LinearCtx *lc = linear_ctx_create(n, k);
                double samples[BENCH_REPS];
                /* warmup (always batched) */
                memset(eq, 0, n * sizeof(double));
                run_linear_batched(n, S, Q, payout, k, eq, lc);
                for (int r = 0; r < BENCH_REPS; r++) {
                    memset(eq, 0, n * sizeof(double));
                    samples[r] = run_linear_batched(n, S, Q, payout, k, eq, lc) / 1e6;
                }
                MEDIAN5(samples);
                times[2] = samples[BENCH_REPS / 2];
                linear_ctx_destroy(lc);
            }

            int best = -1;
            double best_t = 1e30;
            for (int e = 0; e < 4; e++)
                if (times[e] > 0 && times[e] < best_t) { best_t = times[e]; best = e; }

            char cell[48];
            if (best >= 0) {
                char detail[96] = "";
                int dlen = 0;
                for (int e = 0; e < 4; e++)
                    if (times[e] > 0) {
                        char ebuf[32]; fmt_ms_3sf(times[e], ebuf, sizeof(ebuf));
                        dlen += snprintf(detail+dlen, sizeof(detail)-dlen,
                                         "%s%s ", names[e], ebuf);
                    }
                char bbuf[32]; fmt_ms_3sf(best_t, bbuf, sizeof(bbuf));
                snprintf(cell, sizeof(cell), "%s:%-6s(%s)",
                         names[best], bbuf, detail);
            } else {
                snprintf(cell, sizeof(cell), "---");
            }
            printf("  %-34s", cell);
            fflush(stdout);

            free(payout); free(eq);
        }
        printf("\n");
        free(S);
    }

    printf("\nLegend: T=tree(+FFT), L=linear, H=hybrid(B=8). Best engine shown.\n");
    printf("Done.\n");
    return all_pass ? 0 : 1;
}
