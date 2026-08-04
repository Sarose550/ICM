/* probe_leaf_extract.c: Measure leaf-extraction cost embedded in real hybrid
 * engine runs.
 *
 * Methodology: replicates engine_hybrid_core with timing splits at each phase
 * boundary; block_build, tree_build+propagate, leaf_divide. Runs Q=256 points,
 * median over N_REPS independent runs.
 *
 * The main sweep reports raw per-phase timings (block, tree, leaf) at a grid
 * of (n,k) values.  The B-sweep phase (n=8192, k=320, all 6 candidate B
 * values) reports measured leaf-extraction ns/player for each B, producing
 * the leaf_fma_ns_per_player[] lookup table consumed by the hybrid engine.
 *
 * Build (macOS M3 Pro):
 *   gcc -O3 -march=native -Isrc/cpu -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o build/probe_leaf_extract tools/probe_leaf_extract.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 */
#ifdef __APPLE__
#include <pthread.h>
#endif
#include "icm.c"

/* Map candidate block size B to the leaf/block lookup-table index 0-5.
 * Candidate set is {8, 16, 24, 32, 48, 64}. This lives here rather than in
 * icm.c because the library itself no longer uses it -- only this probe does. */
static int B_to_table_index(int B) {
    switch (B) {
        case 8:  return 0;
        case 16: return 1;
        case 24: return 2;
        case 32: return 3;
        case 48: return 4;
        case 64: return 5;
        default: return 0;
    }
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) ? 1 : (da < db) ? -1 : 0;
}

#define N_REPS 21
#define Q_PROBE 256

/* Replicates engine_hybrid_core exactly, but times each phase independently.
 * Returns per-phase times (total ns for all Q points). */
static void engine_hybrid_core_timed(int n, const double *a,
                                      const double *payout, int k,
                                      double *inner, HybridCtx *hc,
                                      double *out_block_ns,
                                      double *out_tree_ns,
                                      double *out_leaf_ns) {
    int B = hc->B;
    int nblocks = hc->nblocks;
    TreeCtx *tc = hc->tc;
    int N = tc->N;
    int *psz = tc->psz;
    double *plev_data = tc->ws;

    /* ── Block build ── */
    double t0 = now_ns();

    int leaf_psz = psz[0];
    for (int b = 0; b < nblocks; b++) {
        int start = b * B, end = start + B;
        if (end > n) end = n;
        int bsize = end - start;
        double *P = hc->block_prods + (size_t)b * (B + 1);
        memset(P, 0, (B + 1) * sizeof(double));
        P[0] = 1.0;
        for (int j = start; j < end; j++) {
            double aj = a[j], bj = 1 - aj;
            for (int m = bsize; m >= 1; m--)
                P[m] = aj * P[m] + bj * P[m - 1];
            P[0] *= aj;
        }
        double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz;
        int cp = (B + 1 < leaf_psz) ? B + 1 : leaf_psz;
        memcpy(leaf, P, cp * sizeof(double));
        if (cp < leaf_psz) memset(leaf + cp, 0, (leaf_psz - cp) * sizeof(double));
    }
    for (int b = nblocks; b < N; b++) {
        double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz;
        memset(leaf, 0, leaf_psz * sizeof(double));
        leaf[0] = 1.0;
    }

    double t1 = now_ns();
    *out_block_ns = t1 - t0;

    /* ── Tree build + propagate ── */
    tree_build_levels(tc);
    double *g_leaf = tree_propagate_g(tc, k, payout, hc->hot_mask);

    double t2 = now_ns();
    *out_tree_ns = t2 - t1;

    int g_need = tc->g_needed[0];

    /* ── Leaf divide ── */
    const uint8_t *active = hc->active;
    for (int b = 0; b < nblocks; b++) {
        int start = b * B, end = start + B;
        if (end > n) end = n;
        int bsize = end - start;

        if (active) {
            int any = 0;
            for (int j = start; j < end; j++) if (active[j]) { any = 1; break; }
            if (!any) continue;
        }

        double *P_b = hc->block_prods + (size_t)b * (B + 1);
        double *g_b = g_leaf + (size_t)b * leaf_psz;
        int pk_g = g_need < bsize ? g_need : bsize;
        if (pk_g > k) pk_g = k;

        double inv_arr[bsize], coeff_arr[bsize];
        int fwd_arr[bsize];
        for (int j = 0; j < bsize; j++) {
            double aj = a[start + j], bj_val = 1 - aj;
            if (aj > 0.5) {
                double ia = 1.0 / aj;
                inv_arr[j] = ia;
                coeff_arr[j] = -bj_val * ia;
                fwd_arr[j] = 1;
            } else if (aj > 1e-15) {
                double ib = 1.0 / bj_val;
                inv_arr[j] = ib;
                coeff_arr[j] = -aj * ib;
                fwd_arr[j] = 0;
            } else {
                inv_arr[j] = 0;
                coeff_arr[j] = 0;
                fwd_arr[j] = -1;
            }
        }

        for (int jj = 0; jj < bsize; jj++) {
            if (active && !active[start + jj]) { inner[start + jj] = 0; continue; }
            double eq = 0;
            if (fwd_arr[jj] == 1) {
                double ia = inv_arr[jj], c = coeff_arr[jj];
                double Q_val = P_b[0] * ia;
                eq = g_b[0] * Q_val;
                for (int m = 1; m < pk_g; m++) {
                    Q_val = c * Q_val + P_b[m] * ia;
                    eq += g_b[m] * Q_val;
                }
            } else if (fwd_arr[jj] == 0) {
                double ib = inv_arr[jj], c = coeff_arr[jj];
                double Q_prev = P_b[bsize] * ib;
                double Q_arr[bsize];
                Q_arr[bsize - 1] = Q_prev;
                for (int m = bsize - 2; m >= 0; m--) {
                    Q_prev = c * Q_prev + P_b[m + 1] * ib;
                    Q_arr[m] = Q_prev;
                }
                for (int m = 0; m < pk_g; m++)
                    eq += g_b[m] * Q_arr[m];
            } else {
                for (int m = 0; m < pk_g; m++)
                    eq += g_b[m] * P_b[m + 1];
            }
            inner[start + jj] = eq;
        }
    }

    double t3 = now_ns();
    *out_leaf_ns = t3 - t2;
}

/* Wrapper matching EquityEngine signature, used with engine_hybrid_ctx's
 * sorted/unsorted logic via run_engine_ctx. We do the sorted approach
 * manually for the timed version. */
static void probe_phases(int n, const double *S,
                          const double *payout, int k,
                          HybridCtx *hc,
                          double *block_ns,
                          double *tree_ns,
                          double *leaf_ns) {
    double *a_qp = (double *)malloc(n * sizeof(double));
    double *inner = (double *)malloc(n * sizeof(double));
    double *equity = (double *)malloc(n * sizeof(double));
    if (!a_qp || !inner || !equity) { fprintf(stderr, "OOM\n"); exit(1); }

    double total_block = 0, total_tree = 0, total_leaf = 0;

    for (int qp = 0; qp < Q_PROBE; qp++) {
        double logv = ((double)qp + 0.5) / (double)Q_PROBE;
        logv = -logv * 10.0;

        for (int i = 0; i < n; i++) {
            double arg = S[i] * logv;
            a_qp[i] = (arg < -700) ? 0.0 : exp(arg);
        }

        /* Sort a_qp to stack-descending order matching engine_hybrid_ctx */
        double *a_sorted = (double *)malloc(n * sizeof(double));
        double *inner_sorted = (double *)malloc(n * sizeof(double));
        if (!a_sorted || !inner_sorted) { fprintf(stderr, "OOM\n"); exit(1); }

        const int *perm = hc->sort_perm;
        for (int i = 0; i < n; i++) {
            a_sorted[i] = a_qp[perm[i]];
        }

        /* Must reset workspace between QPs (tree_build_levels and propagate
         * mutate tc->ws). */
        memset(hc->tc->ws, 0, hc->tc->ws_size * sizeof(double));

        double b_ns, t_ns, l_ns;
        engine_hybrid_core_timed(n, a_sorted, payout, k, inner_sorted, hc,
                                  &b_ns, &t_ns, &l_ns);
        total_block += b_ns;
        total_tree += t_ns;
        total_leaf += l_ns;

        /* Unpermute (reverse of engine_hybrid_ctx's forward permute, same perm[]) */
        for (int i = 0; i < n; i++)
            inner[perm[i]] = inner_sorted[i];

        free(a_sorted);
        free(inner_sorted);
    }

    /* Accumulate equity (not strictly needed for timing, but we do it for
     * correctness) */
    for (int i = 0; i < n; i++) equity[i] = 0;
    /* Quadrature integration is done by run_engine_ctx; here we only
     * measure phases. The timing is per-QP sum, not integrated. */

    *block_ns = total_block;
    *tree_ns = total_tree;
    *leaf_ns = total_leaf;

    free(a_qp); free(inner); free(equity);
}

int main(void) {
#ifdef __APPLE__
    /* Pin to P-cores: without this, the scheduler can silently place this
     * thread on an E-core (half a P-core's FP throughput) under any
     * contention, corrupting the measurement with no indication in the
     * tool's own output. */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    build_fftw_size_table();
    icm_init(NULL);

    int B = 8;

    int n_vals[] = {512, 1024, 2048, 4096, 8192};
    int n_n = 5;
    int k_vals[] = {40, 80, 120, 160, 200, 260, 320, 400};
    int n_k = 8;

    printf("=== PHASE-BY-PHASE TIMING (B=%d, Q=%d, %d reps median) ===\n\n",
           B, Q_PROBE, N_REPS);
    printf("%-6s %-6s %12s %12s %12s %12s %12s %12s\n",
           "n", "k", "block/qp", "tree/qp", "leaf/qp",
           "block/player", "tree/player", "leaf/player");

    for (int ni = 0; ni < n_n; ni++) {
        int n = n_vals[ni];
        for (int ki = 0; ki < n_k; ki++) {
            int k = k_vals[ki];
            if (k > n) continue;
            if (B > k || B > n) continue;

            fprintf(stderr, "Probing n=%d k=%d B=%d...\n", n, k, B);

            double *S = (double *)malloc(n * sizeof(double));
            double *payout = (double *)malloc(k * sizeof(double));
            if (!S || !payout) { fprintf(stderr, "OOM\n"); return 1; }
            srand(42);
            for (int i = 0; i < n; i++)
                S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
            for (int q = 0; q < k; q++)
                payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

            double leaf_samples[N_REPS], block_samples[N_REPS], tree_samples[N_REPS];

            for (int rep = 0; rep < N_REPS; rep++) {
                HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
                if (!hc) { fprintf(stderr, "hc failed\n"); return 1; }

                double block_total, tree_total, leaf_total;
                probe_phases(n, S, payout, k, hc,
                             &block_total, &tree_total, &leaf_total);

                block_samples[rep] = block_total / Q_PROBE;
                tree_samples[rep] = tree_total / Q_PROBE;
                leaf_samples[rep] = leaf_total / Q_PROBE;

                hybrid_ctx_destroy(hc);
            }

            qsort(block_samples, N_REPS, sizeof(double), cmp_double);
            qsort(tree_samples, N_REPS, sizeof(double), cmp_double);
            qsort(leaf_samples, N_REPS, sizeof(double), cmp_double);

            double med_block = block_samples[N_REPS / 2];
            double med_tree = tree_samples[N_REPS / 2];
            double med_leaf = leaf_samples[N_REPS / 2];

            printf("%-6d %-6d %12.1f %12.1f %12.1f %12.4f %12.4f %12.4f\n",
                   n, k, med_block, med_tree, med_leaf,
                   med_block / n, med_tree / n, med_leaf / n);

            free(S); free(payout);
        }
    }

    /* ── B-SWEEP PHASE: n=8192, k=320, sweep B ∈ {8,16,24,32,48,64} ──
     * Same fresh-HybridCtx-per-rep discipline as the main sweep above.
     * n=8192 is large enough that per-block overhead is well amortised;
     * k=320 is near the real crossover region (not an extreme value).
     * Each B runs N_REPS independent HybridCtx alloc→probe→destroy cycles. */
    printf("\n\n=== B-SWEEP (n=8192, k=320, Q=%d, %d reps median) ===\n\n",
           Q_PROBE, N_REPS);
    printf("%-6s %12s %12s\n",
           "B", "leaf_ns/qp", "leaf_ns/player");

    {
        int bs_n = 8192, bs_k = 320;
        int B_vals[] = {8, 16, 24, 32, 48, 64};
        int n_Bvals = 6;
        double leaf_table[6];  /* store measured ns/player for final table */

        double *S_bs = (double *)malloc(bs_n * sizeof(double));
        double *payout_bs = (double *)malloc(bs_k * sizeof(double));
        if (!S_bs || !payout_bs) { fprintf(stderr, "OOM\n"); return 1; }
        srand(42);
        for (int i = 0; i < bs_n; i++)
            S_bs[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
        for (int q = 0; q < bs_k; q++)
            payout_bs[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

        for (int bi = 0; bi < n_Bvals; bi++) {
            int Bv = B_vals[bi];
            fprintf(stderr, "B-sweep: B=%d n=%d k=%d...\n", Bv, bs_n, bs_k);

            double leaf_samples[N_REPS];
            double block_samples[N_REPS];

            for (int rep = 0; rep < N_REPS; rep++) {
                HybridCtx *hc = hybrid_ctx_create(bs_n, S_bs, bs_k, Bv);
                if (!hc) { fprintf(stderr, "hc failed at B=%d\n", Bv); return 1; }

                double block_total, tree_total, leaf_total;
                probe_phases(bs_n, S_bs, payout_bs, bs_k, hc,
                             &block_total, &tree_total, &leaf_total);

                block_samples[rep] = block_total / Q_PROBE;
                leaf_samples[rep] = leaf_total / Q_PROBE;

                hybrid_ctx_destroy(hc);
            }

            qsort(block_samples, N_REPS, sizeof(double), cmp_double);
            qsort(leaf_samples, N_REPS, sizeof(double), cmp_double);

            double med_block = block_samples[N_REPS / 2];
            double med_leaf = leaf_samples[N_REPS / 2];
            double leaf_ns_per_player = med_leaf / (double)bs_n;
            double block_ns_per_player = med_block / (double)bs_n;
            leaf_table[bi] = leaf_ns_per_player;

            printf("%-6d %12.1f %12.4f\n",
                   Bv, med_leaf, leaf_ns_per_player);
        }

        printf("\n=== FINAL leaf_fma_ns_per_player[] TABLE ===\n");
        printf("/* Paste into devices/m3_pro/fft_config.h ; \n");
        printf("   replace the LEAF_FMA_NS_PER_PLAYER_DEFINED block. */\n");
        printf("static const double leaf_fma_ns_per_player[6] = {\n");
        for (int bi = 0; bi < n_Bvals; bi++) {
            const char *comment = (B_vals[bi] == 8)  ? "  /* B=8  */" :
                                  (B_vals[bi] == 16) ? "  /* B=16 */" :
                                  (B_vals[bi] == 24) ? "  /* B=24 */" :
                                  (B_vals[bi] == 32) ? "  /* B=32 */" :
                                  (B_vals[bi] == 48) ? "  /* B=48 */" :
                                                        "  /* B=64 */";
            printf("    %.4f,%s\n", leaf_table[bi], comment);
        }
        printf("};\n");

        free(S_bs); free(payout_bs);
    }

    return 0;
}
