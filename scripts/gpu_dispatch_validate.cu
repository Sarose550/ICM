/* gpu_dispatch_validate.cu — Standing GPU dispatch-validation harness.
 *
 * Analogous to the CPU-side `bench_grid crossover`: for a grid of (n,k)
 * points, measures the ACTUAL wall-clock time of the dispatched
 * configuration (B, tier, fft_n chosen by icm_gpu_plan_create with
 * default options) and compares it against nearby alternative B values
 * forced via ICM_GPU_FORCE_B.  Flags any point where the dispatched
 * configuration is not the fastest measured, and checks monotonicity of
 * dispatched timing across n at fixed k.
 *
 * Two checks:
 *   1. B-optimality: at each (n,k), is the dispatched B the fastest among
 *      itself and a few nearby alternatives, measured on real hardware?
 *   2. n-monotonicity: across n at fixed k, does dispatched time increase
 *      monotonically?  (Finds inversions like the two bugs found this
 *      session — the tier-lock-in wrap-correction bug and the B-selection
 *      sparse-grid non-monotonicity.)
 *
 * Default grid includes n values both ON and BETWEEN the calibrated
 * B-selection anchor points in devices/b200/gpu_fft_config.h, including
 * points in the n=1,048,576–1,572,864 gap that this session identified.
 *
 * Build (B200, cuFFTDx-enabled):
 *
 *   CUFFTDX_INC=$(find /usr/local/lib -maxdepth 4 -type d -path '*dist-packages/nvidia/mathdx/include' | head -1)
 *
 *   nvcc -O3 -std=c++17 -arch=sm_100 -Isrc -Idevices/b200 -Isrc/gpu \
 *        -I"$CUFFTDX_INC" -DUSE_CUFFTDX -DICM_REQUIRE_CUFFTDX \
 *        -DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY \
 *        -dc -o build/gpu_dispatch_validate.o scripts/gpu_dispatch_validate.cu
 *
 *   nvcc -O3 -std=c++17 -arch=sm_100 \
 *        -o gpu_dispatch_validate build/gpu_dispatch_validate.o \
 *        build/gpu_gpu_kernels_fused.o build/gpu_gpu_plan_fused.o \
 *        build/gpu_gpu_exec_fused.o build/gpu_gpu_api_fused.o \
 *        build/gpu_dlink_fused.o -lcufft -lcudart
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "icm_gpu.h"

/* ── helpers ─────────────────────────────────────────────────────── */

#define N_REPS   5
#define Q_POINTS 256

static void make_stacks(int n, std::vector<double> &S) {
    S.resize(n);
    srand(123 + n);
    for (int i = 0; i < n; ++i)
        S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}

static void make_payout(int n, int k, std::vector<double> &payout) {
    payout.resize(k);
    for (int m = 0; m < k; ++m)
        payout[m] = (double)(n - m);
}

static double median_of(std::vector<double> x) {
    if (x.empty()) return NAN;
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

static double mean_of(const std::vector<double> &x) {
    double s = 0.0;
    for (double v : x) s += v;
    return s / (double)x.size();
}

static double cv_of(const std::vector<double> &x) {
    if (x.size() < 2) return 0.0;
    double m = mean_of(x);
    if (m <= 0.0) return 0.0;
    double var = 0.0;
    for (double v : x) var += (v - m) * (v - m);
    var /= (double)(x.size() - 1);
    return std::sqrt(var) / m;
}

/* ── B candidate table (mirrors kBCandidates in src/gpu/gpu_plan.cu) ── */

static const int kBCandidates[] = {
    1, 2, 4,
    8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192,
    208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512,
    576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1536, 1792,
    2048, 2560, 3072, 3584, 4096
};
static const int kNBCandidates = sizeof(kBCandidates) / sizeof(kBCandidates[0]);

/* Return index of B in kBCandidates, or -1 if not found. */
static int find_b_index(int B) {
    for (int i = 0; i < kNBCandidates; ++i)
        if (kBCandidates[i] == B) return i;
    return -1;
}

/* Return a small set of alternative B values near `dispatched_B`:
 * half, double, and the adjacent candidates (±1), deduplicated,
 * clamped to valid range (<= n and <= k).  Max 5 alternatives. */
static void nearby_alternatives(int dispatched_B, int n, int k,
                                std::vector<int> &alts) {
    alts.clear();
    int idx = find_b_index(dispatched_B);
    if (idx < 0) {
        /* Dispatched B not in the candidate table (shouldn't happen);
         * just test half and double. */
        int half = dispatched_B / 2;
        int dbl  = dispatched_B * 2;
        if (half > 0 && half != dispatched_B && half <= n && half <= k)
            alts.push_back(half);
        if (dbl != dispatched_B && dbl <= n && dbl <= k)
            alts.push_back(dbl);
        return;
    }

    /* Collect candidates: half, double, ±1, ±2 index neighbors */
    int candidates[6];
    int nc = 0;

    /* Half: find candidate <= dispatched_B/2 */
    int half_target = dispatched_B / 2;
    for (int i = idx - 1; i >= 0; --i) {
        if (kBCandidates[i] <= half_target) { candidates[nc++] = kBCandidates[i]; break; }
    }
    /* Double: find candidate >= dispatched_B*2 */
    int dbl_target = dispatched_B * 2;
    for (int i = idx + 1; i < kNBCandidates; ++i) {
        if (kBCandidates[i] >= dbl_target) { candidates[nc++] = kBCandidates[i]; break; }
    }
    /* Adjacent ±1 */
    if (idx - 1 >= 0) candidates[nc++] = kBCandidates[idx - 1];
    if (idx + 1 < kNBCandidates) candidates[nc++] = kBCandidates[idx + 1];
    /* ±2 (wider neighborhood for sparse tables) */
    if (idx - 2 >= 0) candidates[nc++] = kBCandidates[idx - 2];
    if (idx + 2 < kNBCandidates) candidates[nc++] = kBCandidates[idx + 2];

    /* Deduplicate and filter */
    for (int i = 0; i < nc; ++i) {
        int B = candidates[i];
        if (B == dispatched_B) continue;
        if (B > n || B > k) continue;
        bool dup = false;
        for (int j = 0; j < (int)alts.size(); ++j)
            if (alts[j] == B) { dup = true; break; }
        if (!dup) alts.push_back(B);
    }

    /* Limit to 5 alternatives */
    if ((int)alts.size() > 5) alts.resize(5);
}

/* ── measurement ─────────────────────────────────────────────────── */

/* Measure median wall-clock time in milliseconds for one (n,k) pair
 * at a specific B value (forced via ICM_GPU_FORCE_B if B >= 0,
 * otherwise use default dispatch).  Returns NAN on failure. */
static double measure_at_b(int n, int k, int B, int *out_actual_B) {
    std::vector<double> S, payout, eq;
    make_stacks(n, S);
    make_payout(n, k, payout);
    eq.assign(n, 0.0);

    IcmGpuOptions opts{};
    opts.device_id                 = 0;
    opts.use_cufftdx               = 1;
    opts.enable_graphs             = 0;
    opts.enable_q_pipeline         = 1;
    opts.memory_strategy           = 0;
    opts.force_uncached_fused_levels  = -1;
    opts.force_uncached_cufft_levels  = -1;

    /* Force B via env var if requested */
    if (B > 0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", B);
        setenv("ICM_GPU_FORCE_B", buf, 1);
    } else {
        unsetenv("ICM_GPU_FORCE_B");
    }

    IcmGpuPlan *plan = icm_gpu_plan_create(n, S.data(), k, &opts);
    if (!plan) {
        fprintf(stderr, "  [measure_at_b] n=%d k=%d B=%d plan_create FAIL: %s\n",
                n, k, B, icm_gpu_last_error());
        icm_gpu_release_pooled_memory();
        return NAN;
    }

    /* Extract actual B used. icm_gpu_plan_summary() returns 1 on success,
     * 0 on failure (see gpu_api.cu) -- the original "== 0" check here was
     * inverted, so *out_actual_B was only ever written on the failure
     * path (never taken in practice), leaving dispatched_B stuck at its
     * caller-side initial value of 0 for every measurement. That silently
     * broke nearby_alternatives() (find_b_index(0) never matches, so
     * half/double both collapsed to 0 and got filtered as duplicates),
     * making every B-optimality check vacuously "optimal" with zero real
     * alternatives ever measured -- confirmed by an actual B200 run
     * (2026-07-26) showing "dispatched B=0" on all 41 grid points and
     * "no valid alternatives to test" on every single one. */
    IcmGpuPlanSummary summary{};
    if (icm_gpu_plan_summary(plan, &summary) != 0 && out_actual_B)
        *out_actual_B = summary.B;

    /* Warmup */
    IcmGpuRunStats warm{};
    int warm_ok = icm_gpu_equity_with_plan(plan, Q_POINTS, payout.data(),
                                            eq.data(), &warm);
    if (warm_ok != 0) {
        fprintf(stderr, "  [measure_at_b] n=%d k=%d B=%d warmup FAIL: %s\n",
                n, k, B, icm_gpu_last_error());
        icm_gpu_plan_destroy(plan);
        icm_gpu_release_pooled_memory();
        return NAN;
    }

    /* Timed reps */
    std::vector<double> samples;
    for (int r = 0; r < N_REPS; ++r) {
        IcmGpuRunStats stats{};
        int status = icm_gpu_equity_with_plan(plan, Q_POINTS,
                                               payout.data(), eq.data(),
                                               &stats);
        if (status != 0) {
            fprintf(stderr, "  [measure_at_b] n=%d k=%d B=%d rep %d FAIL: %s\n",
                    n, k, B, r, icm_gpu_last_error());
            break;
        }
        samples.push_back(stats.total_ns / 1e6);
    }

    icm_gpu_plan_destroy(plan);
    icm_gpu_release_pooled_memory();

    if ((int)samples.size() < N_REPS) return NAN;
    return median_of(samples);
}

/* ── B-optimality check for one (n,k) point ─────────────────────── */

struct AltResult {
    int B;
    double median_ms;
};

/* Returns: 0 = optimal (dispatched within 5% of best),
 *          1 = non-optimal (a faster alternative found),
 *         -1 = skipped (measurement failure). */
static int check_b_optimality(int n, int k) {
    /* Measure dispatched B first */
    int dispatched_B = 0;
    double disp_ms = measure_at_b(n, k, -1, &dispatched_B);
    if (std::isnan(disp_ms)) {
        printf("  [%d,%d] SKIP: dispatched measurement failed\n", n, k);
        return -1;
    }
    printf("  [%d,%d] dispatched B=%-4d  %.3f ms\n", n, k, dispatched_B, disp_ms);

    /* Gather nearby alternatives */
    std::vector<int> alts;
    nearby_alternatives(dispatched_B, n, k, alts);
    if (alts.empty()) {
        printf("  [%d,%d] no valid alternatives to test (B=%d)\n", n, k, dispatched_B);
        return 0;  /* vacuously optimal — nothing to compare against */
    }

    /* Measure each alternative */
    std::vector<AltResult> results;
    results.push_back({dispatched_B, disp_ms});
    for (int alt_B : alts) {
        int actual_B = 0;
        double alt_ms = measure_at_b(n, k, alt_B, &actual_B);
        if (std::isnan(alt_ms)) {
            printf("  [%d,%d] alt B=%-4d FAIL\n", n, k, alt_B);
            continue;
        }
        /* ICM_GPU_FORCE_B is silently ignored by the real dispatch code
         * whenever the requested B exceeds plan->k_pad (gpu_plan.cu:853).
         * If that happened, actual_B == dispatched_B and this "alternative"
         * is really just a second measurement of the exact same
         * configuration -- comparing against it would produce a false
         * "optimal" verdict instead of a real comparison.  Detect and
         * skip explicitly rather than silently trusting it. */
        if (actual_B != alt_B) {
            printf("  [%d,%d] alt B=%-4d SKIPPED: override rejected "
                   "(actual B=%d, likely exceeds k_pad) -- not a real "
                   "alternative, not counted\n", n, k, alt_B, actual_B);
            continue;
        }
        printf("  [%d,%d] alt B=%-4d  %.3f ms  (actual=%d)\n",
               n, k, alt_B, alt_ms, actual_B);
        results.push_back({actual_B, alt_ms});
    }
    if (results.size() == 1) {
        printf("  [%d,%d] WARNING: every alternative was rejected by the "
               "k_pad constraint -- this point was NOT actually validated, "
               "only the dispatched config was measured\n", n, k);
        return -1;  /* skipped, not a real "optimal" verdict */
    }

    /* Find the fastest */
    double best_ms = disp_ms;
    int best_B = dispatched_B;
    for (const auto &r : results) {
        if (r.median_ms < best_ms) {
            best_ms = r.median_ms;
            best_B = r.B;
        }
    }

    /* Is dispatched within 5% of best? */
    double ratio = disp_ms / best_ms;
    bool optimal = (ratio <= 1.05);
    if (!optimal) {
        double pct = (disp_ms - best_ms) / best_ms * 100.0;
        printf("  *** NON-OPTIMAL: dispatched B=%d (%.3f ms) vs best B=%d (%.3f ms), "
               "missed %.1f%% improvement\n",
               dispatched_B, disp_ms, best_B, best_ms, pct);
        return 1;
    }
    return 0;
}

/* ── n-monotonicity check for fixed k ────────────────────────────── */

static void check_n_monotonicity(const char *label,
                                  const std::vector<int> &n_vals, int k,
                                  int *inv_count) {
    printf("\n── n-monotonicity: %s (k=%d) ──\n", label, k);
    printf("%-12s %-6s %-10s %s\n", "n", "B", "ms", "delta");
    printf("----------------------------------------\n");

    double prev_ms = NAN;
    for (int n : n_vals) {
        int dispatched_B = 0;
        double ms = measure_at_b(n, k, -1, &dispatched_B);
        if (std::isnan(ms)) {
            printf("%-12d %-6s %-10s FAIL\n", n, "-", "-");
            prev_ms = NAN;
            continue;
        }

        const char *flag = "";
        if (!std::isnan(prev_ms)) {
            double delta = ms - prev_ms;
            if (delta < -0.005) {  /* more than 5µs drop — real inversion */
                flag = "  *** NON-MONOTONIC (drop)";
                (*inv_count)++;
            }
            printf("%-12d %-6d %-10.3f %+.3f%s\n", n, dispatched_B, ms, delta, flag);
        } else {
            printf("%-12d %-6d %-10.3f\n", n, dispatched_B, ms);
        }
        prev_ms = ms;
        fflush(stdout);
    }
}

/* ── default grid ────────────────────────────────────────────────── */

struct GridPoint { int n, k; };

/* Build default grid: cover B-selection anchor n-values from
 * devices/b200/gpu_fft_config.h, include points BETWEEN anchors
 * (especially the n=1,048,576–1,572,864 gap), and span small-to-large n.
 *
 * Anchor n-values: 4096, 16384, 65536, 131072, 262144, 524288,
 *                   1048576, 1572864
 *
 * k values: for each n, include (a) k=n, (b) one or more k values
 * that actually appear near that n in the gbselect_k[] table,
 * (c) a fixed small k (k=100) for the monotonicity-direction check. */
static const GridPoint kDefaultGrid[] = {
    /* Small n */
    {4096,   4096},
    {4096,   2048},
    {16384,  16384},
    {16384,  8192},

    /* Medium n — on anchor points */
    {65536,  65536},
    {65536,  32768},
    {131072, 131072},
    {131072, 65536},
    {262144, 262144},
    {262144, 131072},
    {524288, 524288},
    {524288, 262144},
    {524288, 128},      /* small k at the n where k=128/k=256 inversion was found */

    /* The gap region: B=32 cliff at n=1,048,576, B=96-192 at n=1,572,864 */
    {1048576, 1048576},
    {1048576, 524288},
    {1048576, 128},      /* k=128: previously-bad cell, now fixed */
    {1200000, 100},       /* BETWEEN anchors, k=100 monotonicity probe */
    {1300000, 100},       /* BETWEEN anchors, ~geometric midpoint */
    {1400000, 100},       /* BETWEEN anchors */
    {1572864, 1572864},
    {1572864, 786432},

    /* Large n */
    {2097152, 2097152},
    {2097152, 128},      /* k=128: neighbor of the original bad cell */
    {2097152, 256},
    {4194304, 4194304},
    {4194304, 128},      /* k=128: the original confirmed-bad cell */
    {4194304, 256},
    {8388608, 8388608},
    {8388608, 128},      /* k=128: previously suspect cell */
    {16777216, 100},      /* large n, small k — OOM boundary probe */

    /* k=100 monotonicity sweep (same n as anchors, for cross-n check) */
    {4096,   100},
    {16384,  100},
    {65536,  100},
    {131072, 100},
    {262144, 100},
    {524288, 100},
    {1048576,100},
    {1200000,100},       /* already above; part of the monotonicity sweep */
    {1572864,100},
    {2097152,100},
    {4194304,100},
    {8388608,100},
};
static const int kNumDefaultPoints = sizeof(kDefaultGrid) / sizeof(kDefaultGrid[0]);

/* Deduplicate grid points (some appear in both B-optimality and
 * monotonicity sweeps above).  Returns deduplicated vector. */
static void dedup_grid(const GridPoint *grid, int npts,
                       std::vector<GridPoint> &out) {
    out.clear();
    for (int i = 0; i < npts; ++i) {
        bool dup = false;
        for (const auto &g : out) {
            if (g.n == grid[i].n && g.k == grid[i].k) { dup = true; break; }
        }
        if (!dup) out.push_back(grid[i]);
    }
}

/* ── main ─────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    /* Allow overriding the grid via command line: n k pairs */
    std::vector<GridPoint> points;
    if (argc >= 3 && (argc - 1) % 2 == 0) {
        for (int i = 1; i < argc; i += 2) {
            int n = atoi(argv[i]);
            int k = atoi(argv[i + 1]);
            if (n > 0 && k > 0 && k <= n)
                points.push_back({n, k});
        }
        if (points.empty()) {
            fprintf(stderr, "Usage: %s [n1 k1 n2 k2 ...]\n", argv[0]);
            return 1;
        }
    } else {
        dedup_grid(kDefaultGrid, kNumDefaultPoints, points);
    }

    if (!icm_gpu_init(0)) {
        fprintf(stderr, "icm_gpu_init failed: %s\n", icm_gpu_last_error());
        return 1;
    }

    printf("gpu_dispatch_validate — reps=%d per measurement, Q=%d\n",
           N_REPS, Q_POINTS);
    printf("plan-based API (icm_gpu_plan_create + icm_gpu_equity_with_plan)\n");
    printf("memory_strategy=0, icm_gpu_release_pooled_memory() between points\n");
    printf("B alternatives forced via ICM_GPU_FORCE_B env var\n");
    printf("\nGrid: %zu (n,k) points\n\n", points.size());

    /* ── Phase 1: B-optimality ── */
    printf("══════════════════════════════════════════════\n");
    printf("PHASE 1: B-optimality (dispatched vs alternatives)\n");
    printf("══════════════════════════════════════════════\n");

    int n_optimal = 0, n_nonopt = 0, n_skipped = 0;
    for (const auto &pt : points) {
        printf("\n--- (%d, %d) ---\n", pt.n, pt.k);
        int result = check_b_optimality(pt.n, pt.k);
        if (result == 0) n_optimal++;
        else if (result == 1) n_nonopt++;
        else n_skipped++;
    }

    /* ── Phase 2: n-monotonicity ── */
    printf("\n\n══════════════════════════════════════════════\n");
    printf("PHASE 2: n-monotonicity (across n at fixed k)\n");
    printf("══════════════════════════════════════════════\n");

    /* Sweep across n at k=n (full-equity-like).
     * We iterate with explicit (n, k=n) pairs since we need the actual k. */
    int inv_k_n = 0;
    {
        printf("\n── n-monotonicity: k=n ──\n");
        printf("%-12s %-6s %-10s %s\n", "n", "B", "ms", "delta");
        printf("----------------------------------------\n");
        int n_vals[] = {
            4096, 16384, 65536, 131072, 262144, 524288,
            1048576, 1200000, 1300000, 1400000, 1572864,
            2097152, 4194304, 8388608, 16777216
        };
        int nn = sizeof(n_vals) / sizeof(n_vals[0]);
        double prev_ms = NAN;
        for (int i = 0; i < nn; ++i) {
            int n = n_vals[i];
            int dispatched_B = 0;
            double ms = measure_at_b(n, n, -1, &dispatched_B);
            if (std::isnan(ms)) {
                printf("%-12d %-6s %-10s FAIL\n", n, "-", "-");
                prev_ms = NAN;
                continue;
            }
            const char *flag = "";
            if (!std::isnan(prev_ms)) {
                double delta = ms - prev_ms;
                if (delta < -0.005) { flag = "  *** NON-MONOTONIC"; inv_k_n++; }
                printf("%-12d %-6d %-10.3f %+.3f%s\n", n, dispatched_B, ms, delta, flag);
            } else {
                printf("%-12d %-6d %-10.3f\n", n, dispatched_B, ms);
            }
            prev_ms = ms;
            fflush(stdout);
        }
        printf("  inversions: %d\n", inv_k_n);
    }

    /* Sweep across n at k=100 (fixed small k, threshold-style) */
    int inv_k100 = 0;
    {
        std::vector<int> n_sweep = {
            4096, 16384, 65536, 131072, 262144, 524288,
            1048576, 1200000, 1300000, 1400000, 1572864,
            2097152, 4194304, 8388608
        };
        check_n_monotonicity("k=100", n_sweep, 100, &inv_k100);
    }

    /* Also do k=128 sweep (the bug signature k) */
    int inv_k128 = 0;
    {
        std::vector<int> n_sweep = {
            65536, 131072, 262144, 524288,
            1048576, 1572864, 2097152, 4194304, 8388608
        };
        check_n_monotonicity("k=128 (bug-signature k)", n_sweep, 128, &inv_k128);
    }

    /* ── Summary ── */
    printf("\n\n══════════════════════════════════════════════\n");
    printf("SUMMARY\n");
    printf("══════════════════════════════════════════════\n");
    printf("B-optimality: %d optimal, %d non-optimal, %d skipped\n",
           n_optimal, n_nonopt, n_skipped);
    printf("Monotonicity inversions:\n");
    printf("  k=n:   %d\n", inv_k_n);
    printf("  k=100: %d\n", inv_k100);
    printf("  k=128: %d\n", inv_k128);
    printf("(Inversion = dispatched time decreases as n increases by more than 5µs)\n");
    printf("Interpretation:\n");
    printf("  Non-optimal B: dispatch picked a B that is measurably slower than\n");
    printf("    a nearby alternative on real hardware (\"wrong B\" pattern).\n");
    printf("  Monotonicity inversion: dispatch time drops as n grows — indicates\n");
    printf("    a cost-model bug at that (n,k) region (e.g. wrong tier, wrong\n");
    printf("    FFT size, or sparse B-selection grid).\n");

    icm_gpu_shutdown();
    return (n_nonopt > 0) ? 1 : 0;
}
