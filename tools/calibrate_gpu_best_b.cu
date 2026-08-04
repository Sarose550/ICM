/* calibrate_gpu_best_b.cu; Direct empirical measurement of the real
 * fastest hybrid-engine block size B(n,k) on the GPU, replacing
 * gpu_select_best_B_est()'s summed-analytical-constants prediction with
 * a small, per-device empirical lookup table; same methodology and
 * rationale as tools/calibrate_best_b.c on CPU (LAPACK ILAENV precedent:
 * measure the real decision directly rather than summing calibrated
 * constants).
 *
 * The (n,k) grid comes from a skeleton CSV produced by
 * gen_calib_skeleton.py.  --narrow-around restricts the candidate set for
 * single-point refinement, and runs are resumable: rows already present in
 * the output CSV are skipped on restart.
 *
 * Timing methodology: every candidate B gets its own adaptive
 * median-of-N measurement (3 reps minimum, up to 15 until cv<=3%), then
 * argmin.  Do not replace this with a cheap "1-rep-rank + confirm the
 * top-2" scheme: a noise fluke at sub-few-ms problem sizes can rank the
 * true winner outside the top-2, where the confirmation step never sees
 * it, and that cost a 281% regression on B200.  See VERDICTS.md V6.
 *
 * This is a ONE-TIME, OFFLINE calibration step; it never runs in
 * production.
 *
 * Build:
 *   make calibrate_gpu_best_b CUDA_ARCH=sm_100 CUFFTDX_INC=-I<path>
 * Run:
 *   ./calibrate_gpu_best_b skeleton_b200.csv gpu_best_b_b200.csv
 *   ./calibrate_gpu_best_b skeleton_b200.csv gpu_best_b_b200.csv --narrow-around 64,128
 */
#include <algorithm>
#include <cmath>
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

/* ── Candidate B set swept by this tool: the subset of kBCandidates
 *    (src/gpu/gpu_plan.cu) worth measuring on the GPU ── */
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

static double median_of(std::vector<double> &x) {
    if (x.empty()) return -1.0;
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

static double cv_of(const std::vector<double> &x) {
    if (x.size() < 2) return 0.0;
    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= (double)x.size();
    if (mean <= 0.0) return 0.0;
    double var = 0.0;
    for (double v : x) {
        double d = v - mean;
        var += d * d;
    }
    var /= (double)(x.size() - 1);
    return sqrt(var) / mean;
}

/* Adaptive median-of-N timing for one (n,k,Q,B) case: 3 reps minimum,
 * extended up to 15 until cv<=3%. Same loop as validate_planner_gpu.cu's
 * run_case_median() and heatmap_gpu.cu's gpu_time_ms(); see VERDICTS.md
 * V6 and the timing-methodology note at the top of this file. */
static double time_adaptive_ms(int n, int k, int Q, int B) {
    double first = time_one_rep(n, k, Q, B);
    if (first < 0.0) return -1.0;
    int reps = 3;
    if (first < 10.0) reps = 10;
    else if (first > 100.0) reps = 1;
    int max_reps = 15;
    std::vector<double> samples;
    samples.push_back(first);
    for (int r = 1; r < reps; ++r) {
        double t = time_one_rep(n, k, Q, B);
        if (t < 0.0) break;
        samples.push_back(t);
    }
    while ((int)samples.size() < max_reps) {
        if (cv_of(samples) <= 0.03) break;
        double t = time_one_rep(n, k, Q, B);
        if (t < 0.0) break;
        samples.push_back(t);
    }
    return median_of(samples);
}

struct CandidateTiming {
    int B;
    double t_ms;
};

/* Find the empirically-fastest B among candidates: every candidate gets
 * its own adaptive median-of-N measurement (time_adaptive_ms above),
 * then argmin. Returns best_B, or -1 if every candidate failed. */
static int find_best_B(int n, int k, int Q,
                       const std::vector<int> &candidates) {
    std::vector<CandidateTiming> results;
    for (int B : candidates) {
        if (B > n) continue;
        double t = time_adaptive_ms(n, k, Q, B);
        if (t < 0.0) continue;
        results.push_back({B, t});
    }
    if (results.empty()) return -1;

    int best_B = results[0].B;
    double best_t = results[0].t_ms;
    for (const auto &r : results) {
        if (r.t_ms < best_t) { best_t = r.t_ms; best_B = r.B; }
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
                "(adaptive median-of-N per candidate, cv<=3%%, Q=%d)\n", Q);
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
