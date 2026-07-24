/* calibrate_gpu_best_b.cu — Direct empirical measurement of the real
 * fastest hybrid-engine block size B(n,k) on the GPU, replacing
 * gpu_select_best_B_est()'s summed-analytical-constants prediction with
 * a small, per-device empirical lookup table — same methodology and
 * rationale as tools/calibrate_best_b.c on CPU (LAPACK ILAENV precedent:
 * measure the real decision directly rather than summing calibrated
 * constants).
 *
 * UPGRADED (A3): reads skeleton CSV from gen_calib_skeleton.py instead of
 * hardcoded n_grid/k_grid; uses 1-rep-rank + confirm-if-close-top-2
 * timing instead of median-of-3-on-every-candidate; supports
 * --narrow-around for single-point refinement; supports resumability
 * (skips already-computed rows in the output CSV on restart).
 *
 * This is a ONE-TIME, OFFLINE calibration step — it never runs in
 * production.
 *
 * Build:
 *   make calibrate_gpu_best_b CUDA_ARCH=sm_100 CUFFTDX_INC=-I<path>
 * Run:
 *   ./calibrate_gpu_best_b skeleton_b200.csv gpu_best_b_b200.csv
 *   ./calibrate_gpu_best_b skeleton_b200.csv gpu_best_b_b200.csv --narrow-around 64,128
 */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "icm_gpu.h"

/* ── Full candidate B set — matches kBCandidates in src/gpu/gpu_plan.cu ── */
static const std::vector<int> kBCandidates = {
    16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256,
    320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536
};

/* ── Utilities ────────────────────────────────────────────────────────── */

static void make_stacks_uniform(int n, std::vector<double> &S) {
    S.resize(n);
    srand(123 + n);
    for (int i = 0; i < n; ++i) S[i] = 1.0 + 99.0 * ((double)rand() / RAND_MAX);
}

static void make_payout(int n, int k, std::vector<double> &payout) {
    payout.resize(k);
    for (int m = 0; m < k; ++m) payout[m] = (double)(n - m);
}

/* Single-rep timing for one (n,k,B) case. Returns -1.0 on failure. */
static double time_one_rep(int n, int k, int Q, int force_B) {
    std::vector<double> S, payout, eq;
    make_stacks_uniform(n, S);
    make_payout(n, k, payout);
    eq.assign(n, 0.0);

    if (force_B > 0) {
        char bbuf[32];
        snprintf(bbuf, sizeof(bbuf), "%d", force_B);
        setenv("ICM_GPU_FORCE_B", bbuf, 1);
    } else {
        unsetenv("ICM_GPU_FORCE_B");
    }

    IcmGpuOptions opts{};
    opts.device_id = 0;
    opts.use_cufftdx = 1;
    opts.enable_graphs = 0;
    opts.enable_q_pipeline = 0;
    opts.memory_strategy = 0;
    opts.force_uncached_fused_levels = -1;
    opts.force_uncached_cufft_levels = -1;

    IcmGpuRunStats stats{};
    int status = icm_gpu_equity(n, S.data(), Q, payout.data(), k, eq.data(), &opts, &stats);
    if (status != 0) return -1.0;
    return stats.total_ns / 1e6;
}

/* Median of three values. */
static double median3(double a, double b, double c) {
    if (a > b) { double t = a; a = b; b = t; }
    if (b > c) { double t = b; b = c; c = t; }
    if (a > b) { double t = a; a = b; b = t; }
    return b;
}

/* ── 1-rep-rank + confirm-if-close-top-2 search ──────────────────────── */

struct CandidateTiming {
    int B;
    double t_ms;
};

/* Find the empirically-fastest B among candidates using:
 *   1. 1 rep per candidate → rank by time
 *   2. If top 2 are within 3% of each other: run 2 more reps on each,
 *      take median of all 3 for each, pick winner by median.
 *   3. Otherwise: the 1-rep winner is the answer.
 *
 * Returns best_B, or -1 if every candidate failed. */
static int find_best_B(int n, int k, int Q,
                       const std::vector<int> &candidates) {
    std::vector<CandidateTiming> results;

    /* Phase 1: 1 rep per candidate */
    for (int B : candidates) {
        if (B > n) continue;
        double t = time_one_rep(n, k, Q, B);
        if (t < 0.0) continue;
        results.push_back({B, t});
    }
    if (results.empty()) return -1;

    /* Sort ascending by time */
    std::sort(results.begin(), results.end(),
              [](const CandidateTiming &a, const CandidateTiming &b) {
                  return a.t_ms < b.t_ms;
              });

    int best_B;
    if (results.size() >= 2) {
        double gap = (results[1].t_ms - results[0].t_ms) / results[0].t_ms;
        if (gap < 0.03) {
            /* Phase 2: confirm top-2 with 2 more reps each (total 3). */
            double s0[3] = {results[0].t_ms, -1.0, -1.0};
            double s1[3] = {results[1].t_ms, -1.0, -1.0};
            int n0 = 1, n1 = 1;
            for (int r = 0; r < 2; r++) {
                double t0 = time_one_rep(n, k, Q, results[0].B);
                double t1 = time_one_rep(n, k, Q, results[1].B);
                if (t0 >= 0.0) s0[n0++] = t0;
                if (t1 >= 0.0) s1[n1++] = t1;
            }
            /* Median of collected reps (2 or 3). */
            double med0 = (n0 >= 3) ? median3(s0[0], s0[1], s0[2])
                        : (n0 == 2) ? ((s0[0] + s0[1]) * 0.5) : s0[0];
            double med1 = (n1 >= 3) ? median3(s1[0], s1[1], s1[2])
                        : (n1 == 2) ? ((s1[0] + s1[1]) * 0.5) : s1[0];
            best_B = (med0 <= med1) ? results[0].B : results[1].B;
        } else {
            best_B = results[0].B;
        }
    } else {
        best_B = results[0].B;
    }
    return best_B;
}

/* ── CSV parsing ──────────────────────────────────────────────────────── */

/* Read skeleton CSV (format from gen_calib_skeleton.py):
 *   header "n,k" then data rows "n,k".
 * Lines starting with '#' are comments and skipped. */
static std::vector<std::pair<int,int>> read_skeleton_csv(const char *path) {
    std::vector<std::pair<int,int>> points;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open skeleton CSV: %s\n", path);
        return points;
    }
    std::string line;
    bool first_data = true;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (first_data) { first_data = false; continue; } /* skip "n,k" header */
        std::stringstream ss(line);
        std::string ns, ks;
        if (!std::getline(ss, ns, ',') || !std::getline(ss, ks, ','))
            continue;
        int n = atoi(ns.c_str());
        int k = atoi(ks.c_str());
        if (n > 0 && k > 0) points.push_back({n, k});
    }
    return points;
}

/* Read existing output CSV to find already-computed (n,k) pairs.
 * Lines starting with '#' are skipped; data format is "n,k,best_B". */
static std::unordered_set<std::string> read_existing_output(const char *path) {
    std::unordered_set<std::string> existing;
    std::ifstream f(path);
    if (!f.is_open()) return existing;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        /* Extract "n,k" prefix (up to second comma). */
        size_t c1 = line.find(',');
        if (c1 == std::string::npos) continue;
        size_t c2 = line.find(',', c1 + 1);
        existing.insert((c2 != std::string::npos) ? line.substr(0, c2)
                                                  : line);
    }
    return existing;
}

/* ── Narrow-around helper ─────────────────────────────────────────────── */

/* Given target B values, return the set of those values plus their
 * immediate neighbors in kBCandidates. */
static std::vector<int> narrow_candidates(const std::vector<int> &targets) {
    std::unordered_set<int> cand_set;
    for (int t : targets) {
        cand_set.insert(t);
        for (size_t i = 0; i < kBCandidates.size(); i++) {
            if (kBCandidates[i] == t) {
                if (i > 0) cand_set.insert(kBCandidates[i - 1]);
                if (i + 1 < kBCandidates.size())
                    cand_set.insert(kBCandidates[i + 1]);
            }
        }
    }
    std::vector<int> result(cand_set.begin(), cand_set.end());
    std::sort(result.begin(), result.end());
    return result;
}

/* ── Main ─────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *skeleton_csv = nullptr;
    const char *output_csv   = nullptr;
    const char *narrow_str   = nullptr;
    int Q = 256;
    bool dry_run = false;

    /* ── Parse CLI ── */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--narrow-around") == 0 && i + 1 < argc) {
            narrow_str = argv[++i];
        } else if (strcmp(argv[i], "--Q") == 0 && i + 1 < argc) {
            Q = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            dry_run = true;
        } else if (!skeleton_csv) {
            skeleton_csv = argv[i];
        } else if (!output_csv) {
            output_csv = argv[i];
        }
    }

    if (!skeleton_csv || !output_csv) {
        fprintf(stderr,
                "Usage: %s <skeleton_csv> <output_csv> "
                "[--narrow-around B1,B2,...] [--Q Q] [--dry-run]\n",
                argv[0]);
        fprintf(stderr,
                "  skeleton_csv : path to skeleton CSV from gen_calib_skeleton.py\n"
                "  output_csv   : path for output (appended; resumable)\n"
                "  --narrow-around B1,B2,... : only test listed B + immediate neighbors\n"
                "  --Q Q        : quadrature points (default 256)\n"
                "  --dry-run    : print parsed args and exit (no CUDA calls)\n");
        return 1;
    }

    /* ── Read skeleton ── */
    auto points = read_skeleton_csv(skeleton_csv);
    if (points.empty()) {
        fprintf(stderr, "No points read from skeleton CSV: %s\n", skeleton_csv);
        return 1;
    }
    fprintf(stderr, "Read %zu calibration points from skeleton\n", points.size());

    /* ── Build candidate set ── */
    std::vector<int> candidates;
    if (narrow_str) {
        std::vector<int> targets;
        std::stringstream ss(narrow_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            int B = atoi(token.c_str());
            if (B > 0) targets.push_back(B);
        }
        if (targets.empty()) {
            fprintf(stderr, "--narrow-around requires at least one B value\n");
            return 1;
        }
        candidates = narrow_candidates(targets);
        fprintf(stderr, "Narrow-around mode: %zu candidates (targets + neighbors)\n",
                candidates.size());
    } else {
        candidates = kBCandidates;
    }

    /* ── Resumability ── */
    auto existing = read_existing_output(output_csv);
    if (!existing.empty()) {
        fprintf(stderr, "Resuming: %zu (n,k) pairs already in output, skipping\n",
                existing.size());
    }

    /* ── Open output (append mode) ── */
    bool file_exists = static_cast<bool>(std::ifstream(output_csv));
    FILE *fout = fopen(output_csv, "a");
    if (!fout) {
        fprintf(stderr, "Cannot open output CSV: %s\n", output_csv);
        return 1;
    }
    if (!file_exists) {
        fprintf(fout,
                "# Direct empirical GPU best-B measurement "
                "(1-rep-rank + confirm-if-close-top-2, Q=%d)\n", Q);
        fprintf(fout, "# n,k,best_B\n");
    }

    /* ── Dry-run: print parsed args and exit (no CUDA calls) ── */
    if (dry_run) {
        printf("=== DRY RUN (no CUDA calls made) ===\n");
        printf("skeleton_csv  = %s\n", skeleton_csv);
        printf("output_csv    = %s\n", output_csv);
        printf("Q             = %d\n", Q);
        printf("narrow_around = %s\n", narrow_str ? narrow_str : "(full sweep)");
        printf("candidates    = [");
        for (size_t i = 0; i < candidates.size(); i++) {
            if (i > 0) printf(", ");
            printf("%d", candidates[i]);
        }
        printf("]\n");
        printf("total points  = %zu\n", points.size());
        printf("already done  = %zu\n", existing.size());
        printf("to measure    = %zu\n",
               points.size() - existing.size());
        return 0;
    }

    /* ── GPU init ── */
    if (!icm_gpu_init(0)) {
        fprintf(stderr, "icm_gpu_init failed: %s\n", icm_gpu_last_error());
        fclose(fout);
        return 1;
    }

    /* ── Main loop ── */
    int n_skipped  = 0;
    int n_measured = 0;
    int n_failed   = 0;

    for (auto &pt : points) {
        int n = pt.first, k = pt.second;

        char nk_buf[64];
        snprintf(nk_buf, sizeof(nk_buf), "%d,%d", n, k);
        if (existing.count(nk_buf)) {
            n_skipped++;
            continue;
        }

        int best_B = find_best_B(n, k, Q, candidates);
        if (best_B < 0) {
            fprintf(stderr, "n=%d k=%d FAILED (all candidates failed)\n", n, k);
            n_failed++;
            continue;
        }

        fprintf(fout, "%d,%d,%d\n", n, k, best_B);
        fflush(fout);
        fprintf(stderr, "n=%d k=%d -> best_B=%d\n", n, k, best_B);
        n_measured++;
    }

    fclose(fout);
    icm_gpu_shutdown();

    fprintf(stderr,
            "Done: %d measured, %d skipped, %d failed (total %zu points)\n",
            n_measured, n_skipped, n_failed, points.size());
    return (n_failed > 0) ? 1 : 0;
}
