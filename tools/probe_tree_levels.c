/* probe_tree_levels.c — Per-level real-vs-predicted tree cost decomposition.
 *
 * Instruments the real hybrid engine's per-QP tree phases (build + propagate)
 * with per-level timing, and compares against the cost model's per-level
 * prediction formula from select_engine_ex().
 *
 * Output CSV (aggregated per (n,k,B) cell):
 *   n,k,B,num_schoolbook,geo_schoolbook,
 *   num_fft_cached,geo_fft_cached,
 *   num_fft_uncached,geo_fft_uncached,min_fft_uncached,max_fft_uncached,
 *   block_build_ns,leaf_divide_ns
 *
 * Where geo_* is the geometric mean of (measured_ns / predicted_ns) across
 * all levels in that bucket, and num_* is the count of levels in that bucket.
 *
 * Build (macOS M3 Pro):
 *   gcc -O3 -march=native -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o build/probe_tree_levels tools/probe_tree_levels.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 */
#include "icm.c"
#include <stdio.h>

/* ── Per-level cost model prediction (exact copy of select_engine_ex's formula) ── */

static double predict_level_ns(TreeCtx *tc, int ell) {
    int cps = tc->psz[ell-1], nr = tc->n_real[ell];
    if (tc->use_fft[ell]) {
        int bfn = tc->build_fft_n[ell];
        int bwm = tc->build_wrap_m[ell];
        int idx = 0;
        { int lo=0,hi=N_CALIBRATED_SIZES-1;
          while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<bfn)lo=m+1;else hi=m;}
          idx=lo; }
        double build_fft = calib_times_ns[idx] + FFT_OVERHEAD_NS
                         + (double)bwm*(bwm+1)/2.0*WRAP_FMA_NS;
        double corr;
        if (tc->fft_cache_ok[ell]) {
            corr = calib_times_ns[idx] * PAIRED_CACHED_CORR_RATIO
                 + (double)tc->corr_wrap_m[ell]*(tc->corr_wrap_m[ell]+1)*WRAP_FMA_NS;
        } else {
            int cfn = tc->corr_fft_n[ell];
            int cwm = tc->corr_wrap_m[ell];
            int cidx=0;
            {int lo=0,hi=N_CALIBRATED_SIZES-1;
             while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<cfn)lo=m+1;else hi=m;}
             cidx=lo;}
            corr = INDEP_PAIR_RATIO * calib_times_ns[cidx]
                 + (double)cwm*(cwm+1)*WRAP_FMA_NS;
        }
        return nr * (build_fft + corr);
    } else {
        /* Schoolbook cost via direct per-size lookup table -- exact copy of
         * select_engine_ex's current formula (src/icm.c ~line 2317-2327).
         * This branch was stale until this fix: it previously used the OLD
         * (d_eff+1)^2*FMA_NS formula from before commit 8012244 replaced it
         * with per-size lookup tables, making every schoolbook-bucket
         * comparison from this tool invalid since that commit. */
        int idx;
        { int lo=0,hi=N_CALIBRATED_SIZES-1;
          while(lo<hi){int m=(lo+hi)>>1;if(calib_sizes[m]<cps)lo=m+1;else hi=m;}
          idx=lo; }
        double s = schoolbook_mul_ns[idx];
        double c = (double)cps * tc->g_needed[ell-1] * schoolbook_corr_ns[idx];
        return nr * (s + c);
    }
}

/* ── Timed wrappers for tree_build_levels and tree_propagate_g ── */

/* Replicates tree_build_levels() body exactly, but times each ell iteration.
 * level_build_ns[ell] receives the elapsed ns for that level.
 * Returns total build time. */
static double tree_build_levels_timed(TreeCtx *tc, double level_build_ns[]) {
    int L = tc->L;
    int *psz = tc->psz;
    size_t *plev_off = tc->plev_off;
    double *plev_data = tc->ws;
    double total = 0;

    for (int ell = 1; ell < L - 1; ell++) {
        double t0 = now_ns();

        int cps = psz[ell-1], pps = psz[ell];
        double *child_base = plev_data + plev_off[ell-1];
        double *parent_base = plev_data + plev_off[ell];
        int use_fft = tc->use_fft[ell];
        int cache_fft = tc->fft_cache_ok[ell];

        int cn = cache_fft ? tc->fft_coeff_cn[ell] : 0;
        int nr_parent = tc->n_real[ell];
        int nr_child = tc->n_real[ell-1];
        for (int j = 0; j < nr_parent; j++) {
            double *Lc = child_base + (size_t)(2*j) * cps;
            double *out = parent_base + (size_t)j * pps;
            if (2*j+1 >= nr_child) {
                int cp = (cps < pps) ? cps : pps;
                memcpy(out, Lc, cp * sizeof(double));
                if (cp < pps) memset(out + cp, 0, (pps - cp) * sizeof(double));
                if (cache_fft) {
                    fftw_complex *fft_L = tc->fft_coeff[ell] + (size_t)(2*j) * cn;
                    int tfn = tc->fft_coeff_n[ell];
                    FFTPlan *plan = fft_cache_get(tc->fft, tfn);
                    int fft_n = plan->fft_n;
                    memcpy(plan->rbuf, Lc, cps * sizeof(double));
                    if (cps < fft_n) memset(plan->rbuf + cps, 0, (fft_n - cps) * sizeof(double));
                    fft_exec_fwd(plan);
                    memcpy(fft_L, plan->cbuf, (fft_n/2+1) * sizeof(fftw_complex));
                }
            } else {
                double *Rc = child_base + (size_t)(2*j+1) * cps;
                if (use_fft) {
                    fftw_complex *fft_L = cache_fft ?
                        tc->fft_coeff[ell] + (size_t)(2*j) * cn : NULL;
                    fftw_complex *fft_R = cache_fft ?
                        tc->fft_coeff[ell] + (size_t)(2*j+1) * cn : NULL;
                    polymul_fft_wrap(Lc, cps, Rc, cps, out, pps,
                                     tc->fft, fft_L, fft_R,
                                     tc->build_fft_n[ell], tc->build_wrap_m[ell]);
                } else {
                    polymul_modk(Lc, cps, Rc, cps, out, pps);
                }
            }
        }

        double dt = now_ns() - t0;
        level_build_ns[ell] = dt;
        total += dt;
    }
    return total;
}

/* Replicates tree_propagate_g() body exactly (full-equity fast path only),
 * but times each ell iteration. level_prop_ns[ell] receives the elapsed ns
 * for propagation from level ell down to ell-1.
 * Returns total propagate time. */
static double tree_propagate_g_timed(TreeCtx *tc, int k, const double *payout,
                                      double level_prop_ns[]) {
    int L = tc->L;
    int *psz = tc->psz, *nn = tc->nn;
    size_t *plev_off = tc->plev_off;
    double *plev_data = tc->ws;
    double *g_buf0 = tc->ws + tc->plev_total;
    double *g_buf1 = tc->ws + tc->plev_total + tc->max_g;

    int top = L - 1;
    int root_gsz = psz[top];
    double *g_parent = g_buf0;
    memset(g_parent, 0, (size_t)nn[top] * root_gsz * sizeof(double));
    int copy = (k < root_gsz) ? k : root_gsz;
    memcpy(g_parent, payout, copy * sizeof(double));

    double total = 0;

    for (int ell = top; ell >= 1; ell--) {
        double t0 = now_ns();

        int pgsz = psz[ell], cgsz = psz[ell-1], cps = psz[ell-1];
        double *g_child = (g_parent == g_buf0) ? g_buf1 : g_buf0;
        double *child_base = plev_data + plev_off[ell-1];
        int use_fft = tc->use_fft[ell];
        int cache_fft = tc->fft_cache_ok[ell];
        int is_below = tc->below_sat[ell];
        int p_eff = is_below ? cps/2 + 1 : cps;
        int cn = cache_fft ? tc->fft_coeff_cn[ell] : 0;
        int cached_fft_n = cache_fft ? tc->fft_coeff_n[ell] : 0;

        int out_needed = tc->g_needed[ell-1];
        int g_eff_needed = out_needed + p_eff - 1;
        int g_eff_max = is_below ? (cgsz + cps/2) : pgsz;
        int g_eff = (g_eff_needed < g_eff_max) ? g_eff_needed : g_eff_max;

        int nr_parent = tc->n_real[ell];
        int nr_child = tc->n_real[ell-1];

        /* Full-equity fast path (hot_mask == NULL for our diagnostic) */
        for (int j = 0; j < nr_parent; j++) {
            double *gp = g_parent + (size_t)j * pgsz;
            double *gL = g_child + (size_t)(2*j) * cgsz;
            if (2*j+1 >= nr_child) {
                int cp = (out_needed < pgsz) ? out_needed : pgsz;
                memcpy(gL, gp, cp * sizeof(double));
                if (cp < out_needed) memset(gL + cp, 0, (out_needed - cp) * sizeof(double));
            } else {
                double *PL = child_base + (size_t)(2*j) * cps;
                double *PR = child_base + (size_t)(2*j+1) * cps;
                double *gR = g_child + (size_t)(2*j+1) * cgsz;
                if (use_fft && cache_fft) {
                    const fftw_complex *fft_R = tc->fft_coeff[ell] + (size_t)(2*j+1)*cn;
                    const fftw_complex *fft_L = tc->fft_coeff[ell] + (size_t)(2*j)*cn;
                    int cwm = tc->corr_wrap_m[ell];
                    correlate_fft_cached_pair_wrap(gp, g_eff, PL, PR, p_eff,
                                                   gL, gR, out_needed,
                                                   tc->fft, fft_L, fft_R,
                                                   cached_fft_n, cwm);
                } else if (use_fft) {
                    correlate_fft_pair(gp, g_eff, PL, PR, p_eff,
                                       gL, gR, out_needed, tc->fft,
                                       tc->corr_fft_n[ell], tc->corr_wrap_m[ell]);
                } else {
                    correlate_school(gp, g_eff, PR, p_eff, gL, out_needed);
                    correlate_school(gp, g_eff, PL, p_eff, gR, out_needed);
                }
            }
        }

        double dt = now_ns() - t0;
        level_prop_ns[ell] = dt;
        total += dt;

        /* Swap buffers for next iteration */
        g_parent = g_child;
    }

    return total;
}

/* ── Comparison helper ── */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) ? 1 : (da < db) ? -1 : 0;
}

/* ── Bucket helper ── */
typedef struct {
    double *ratios;
    int count;
    int cap;
} Bucket;

static void bucket_init(Bucket *b) {
    b->ratios = NULL;
    b->count = 0;
    b->cap = 0;
}

static void bucket_add(Bucket *b, double ratio) {
    if (b->count >= b->cap) {
        int new_cap = b->cap ? b->cap * 2 : 16;
        b->ratios = (double *)realloc(b->ratios, new_cap * sizeof(double));
        b->cap = new_cap;
    }
    b->ratios[b->count++] = ratio;
}

static double bucket_geo_mean(const Bucket *b) {
    if (b->count == 0) return 0.0;
    double sum_log = 0.0;
    int valid = 0;
    for (int i = 0; i < b->count; i++) {
        if (b->ratios[i] > 0.0) {
            sum_log += log(b->ratios[i]);
            valid++;
        }
    }
    if (valid == 0) return 0.0;
    return exp(sum_log / valid);
}

static double bucket_min(const Bucket *b) {
    if (b->count == 0) return 0.0;
    double v = b->ratios[0];
    for (int i = 1; i < b->count; i++)
        if (b->ratios[i] < v) v = b->ratios[i];
    return v;
}

static double bucket_max(const Bucket *b) {
    if (b->count == 0) return 0.0;
    double v = b->ratios[0];
    for (int i = 1; i < b->count; i++)
        if (b->ratios[i] > v) v = b->ratios[i];
    return v;
}

static void bucket_free(Bucket *b) {
    free(b->ratios);
    b->ratios = NULL;
    b->count = b->cap = 0;
}

/* ── Driver ── */

#define MAX_REPS 80
#define MAX_L 20

static void probe_one(int n, int k, int B,
                       Bucket *sb, Bucket *fft_c, Bucket *fft_u,
                       double *out_block_ns, double *out_leaf_ns) {
    double *S = (double *)malloc(n * sizeof(double));
    double *payout = (double *)malloc(k * sizeof(double));
    double *a = (double *)malloc(n * sizeof(double));
    double *inner = (double *)malloc(n * sizeof(double));
    if (!S || !payout || !a || !inner) {
        fprintf(stderr, "OOM n=%d\n", n);
        goto cleanup;
    }

    srand(42);
    for (int i = 0; i < n; i++)
        S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
    for (int q = 0; q < k; q++)
        payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

    /* Generate a[] for one representative quadrature point (logv = -1.0). */
    double logv = -1.0;
    for (int i = 0; i < n; i++) {
        double arg = S[i] * logv;
        a[i] = (arg < -700) ? 0.0 : exp(arg);
    }

    /* Create the hybrid context once (static structures only, not timed). */
    HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
    if (!hc) { fprintf(stderr, "hybrid_ctx_create failed n=%d k=%d B=%d\n", n, k, B); goto cleanup; }
    TreeCtx *tc = hc->tc;
    int L = tc->L;
    if (L > MAX_L) { fprintf(stderr, "tree depth %d exceeds MAX_L=%d\n", L, MAX_L); goto cleanup; }

    /* Per-level accumulation arrays for median computation */
    double build_samples[MAX_L][MAX_REPS];
    double prop_samples[MAX_L][MAX_REPS];
    double block_samples[MAX_REPS];
    double leaf_samples[MAX_REPS];
    int n_reps = 0;

    for (int rep = 0; rep < MAX_REPS; rep++) {
        memset(tc->ws, 0, tc->ws_size * sizeof(double));

        double t_block0 = now_ns();

        /* ── Block build (exactly as in engine_hybrid_core) ── */
        int B_val = hc->B;
        int nblocks = hc->nblocks;
        int N_tree = tc->N;
        int leaf_psz = tc->psz[0];
        double *plev_data = tc->ws;

        for (int b = 0; b < nblocks; b++) {
            int start = b * B_val, end = start + B_val;
            if (end > n) end = n;
            int bsize = end - start;
            double *P = hc->block_prods + (size_t)b * (B_val + 1);
            memset(P, 0, (B_val + 1) * sizeof(double));
            P[0] = 1.0;
            for (int j = start; j < end; j++) {
                double aj = a[j], bj = 1 - aj;
                for (int m = bsize; m >= 1; m--)
                    P[m] = aj * P[m] + bj * P[m - 1];
                P[0] *= aj;
            }
            double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz;
            int cp = (B_val + 1 < leaf_psz) ? B_val + 1 : leaf_psz;
            memcpy(leaf, P, cp * sizeof(double));
            if (cp < leaf_psz) memset(leaf + cp, 0, (leaf_psz - cp) * sizeof(double));
        }
        for (int b = nblocks; b < N_tree; b++) {
            double *leaf = plev_data + tc->plev_off[0] + (size_t)b * leaf_psz;
            memset(leaf, 0, leaf_psz * sizeof(double));
            leaf[0] = 1.0;
        }

        double t_block1 = now_ns();
        block_samples[rep] = t_block1 - t_block0;

        /* ── Timed tree build ── */
        double level_build_ns[MAX_L] = {0};
        tree_build_levels_timed(tc, level_build_ns);

        for (int ell = 1; ell < L - 1; ell++)
            build_samples[ell][rep] = level_build_ns[ell];

        /* ── Timed tree propagate ── */
        double level_prop_ns[MAX_L] = {0};
        tree_propagate_g_timed(tc, k, payout, level_prop_ns);

        for (int ell = 1; ell < L; ell++)
            prop_samples[ell][rep] = level_prop_ns[ell];

        /* ── Leaf divide (timed for context) ── */
        double t_leaf0 = now_ns();

        double *leaf_g = (L % 2 == 0)
            ? (tc->ws + tc->plev_total)
            : (tc->ws + tc->plev_total + tc->max_g);

        int g_need = tc->g_needed[0];
        for (int b = 0; b < nblocks; b++) {
            int start = b * B_val, end = start + B_val;
            if (end > n) end = n;
            int bsize = end - start;
            double *P_b = hc->block_prods + (size_t)b * (B_val + 1);
            double *g_b = leaf_g + (size_t)b * leaf_psz;
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

        double t_leaf1 = now_ns();
        leaf_samples[rep] = t_leaf1 - t_leaf0;

        n_reps++;
    }

    /* ── Compute medians and bucket per-level ratios ── */
    qsort(block_samples, n_reps, sizeof(double), cmp_double);
    *out_block_ns = block_samples[n_reps / 2];

    qsort(leaf_samples, n_reps, sizeof(double), cmp_double);
    *out_leaf_ns = leaf_samples[n_reps / 2];

    for (int ell = 1; ell < L - 1; ell++) {
        double combined[MAX_REPS];
        for (int r = 0; r < n_reps; r++)
            combined[r] = build_samples[ell][r] + prop_samples[ell][r];
        qsort(combined, n_reps, sizeof(double), cmp_double);
        double med_measured = combined[n_reps / 2];

        double predicted = predict_level_ns(tc, ell);
        double ratio = (predicted > 0) ? med_measured / predicted : 0;

        int use_fft = tc->use_fft[ell];
        int cache_ok = tc->fft_cache_ok[ell];

        if (!use_fft) {
            bucket_add(sb, ratio);
        } else if (cache_ok) {
            bucket_add(fft_c, ratio);
        } else {
            bucket_add(fft_u, ratio);
        }
    }

    hybrid_ctx_destroy(hc);

cleanup:
    free(S); free(payout); free(a); free(inner);
}

/* ── Plan dumper (for diagnostic verification) ── */
static void dump_plan(int n, int k, int B) {
    double *S = (double *)malloc(n * sizeof(double));
    if (!S) return;
    srand(42);
    for (int i = 0; i < n; i++)
        S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
    HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
    if (!hc) { free(S); return; }
    TreeCtx *tc = hc->tc;
    int L = tc->L;
    fprintf(stderr, "  PLAN n=%d k=%d B=%d L=%d:\n", n, k, B, L);
    for (int ell = 1; ell < L - 1; ell++) {
        fprintf(stderr, "    ell=%d cps=%d nr=%d use_fft=%d cache_ok=%d "
                "bfn=%d bwm=%d cfn=%d cwm=%d g_need=%d below=%d\n",
                ell, tc->psz[ell-1], tc->n_real[ell],
                tc->use_fft[ell], tc->fft_cache_ok[ell],
                tc->build_fft_n[ell], tc->build_wrap_m[ell],
                tc->corr_fft_n[ell], tc->corr_wrap_m[ell],
                tc->g_needed[ell-1], tc->below_sat[ell]);
    }
    hybrid_ctx_destroy(hc);
    free(S);
}

int main(void) {
    build_fftw_size_table();
    icm_init(NULL);

    /* Sweep grid designed to hit the FFT-uncached crossover region on M3 Pro.
     * n: powers of two from 512 to 16384
     * k: medium-to-large payout counts that span the hybrid/linear crossover
     * B: 8 (the block size selected by production dispatch for these sizes) */
    int n_vals[] = {512, 1024, 2048, 4096, 8192, 16384};
    int n_n = 6;
    int k_vals[] = {80, 120, 160, 200, 240, 280, 320, 400};
    int n_k = 8;
    int B = 8;

    /* Header */
    printf("n,k,B,num_schoolbook,geo_schoolbook,"
           "num_fft_cached,geo_fft_cached,"
           "num_fft_uncached,geo_fft_uncached,min_fft_uncached,max_fft_uncached,"
           "block_build_ns,leaf_divide_ns\n");

    /* ── Phase 1: dump plans for a few representative cells to stderr ── */
    fprintf(stderr, "=== Plan diagnostics (B=8) ===\n");
    dump_plan(512, 400, 8);
    dump_plan(2048, 400, 8);
    dump_plan(8192, 400, 8);
    dump_plan(16384, 400, 8);

    /* Also try B=32 for comparison (known to hit uncached on Zen4) */
    fprintf(stderr, "=== Plan diagnostics (B=32) ===\n");
    dump_plan(4096, 400, 32);
    dump_plan(16384, 400, 32);
    dump_plan(16384, 2000, 32);

    /* ── Phase 2: main sweep (B=8) ── */
    for (int ni = 0; ni < n_n; ni++) {
        int n = n_vals[ni];
        for (int ki = 0; ki < n_k; ki++) {
            int k = k_vals[ki];
            if (k > n) continue;  /* skip invalid cells */

            int bestB = select_best_B(n, k);
            fprintf(stderr, "Probing n=%d k=%d B=%d (best_B=%d)...\n", n, k, B, bestB);

            Bucket sb, fft_c, fft_u;
            bucket_init(&sb);
            bucket_init(&fft_c);
            bucket_init(&fft_u);

            double block_ns = 0, leaf_ns = 0;
            probe_one(n, k, B, &sb, &fft_c, &fft_u, &block_ns, &leaf_ns);

            double geo_sb = bucket_geo_mean(&sb);
            double geo_fft_c = bucket_geo_mean(&fft_c);
            double geo_fft_u = bucket_geo_mean(&fft_u);
            double min_fft_u = bucket_min(&fft_u);
            double max_fft_u = bucket_max(&fft_u);

            printf("%d,%d,%d,%d,%.4f,%d,%.4f,%d,%.4f,%.4f,%.4f,%.3f,%.3f\n",
                   n, k, B,
                   sb.count, geo_sb,
                   fft_c.count, geo_fft_c,
                   fft_u.count, geo_fft_u, min_fft_u, max_fft_u,
                   block_ns, leaf_ns);
            fflush(stdout);

            bucket_free(&sb);
            bucket_free(&fft_c);
            bucket_free(&fft_u);
        }
    }

    /* ── Phase 3: spot-check with B=32 to see if uncached appears ── */
    fprintf(stderr, "=== Spot check B=32 (known uncached on Zen4) ===\n");
    {
        int b32_n[] = {4096, 8192, 16384};
        int b32_k[] = {400, 800, 2000};
        for (int ni = 0; ni < 3; ni++) {
            int n = b32_n[ni];
            for (int ki = 0; ki < 3; ki++) {
                int k = b32_k[ki];
                if (k > n) continue;
                fprintf(stderr, "Probing n=%d k=%d B=32...\n", n, k);
                Bucket sb, fft_c, fft_u;
                bucket_init(&sb); bucket_init(&fft_c); bucket_init(&fft_u);
                double block_ns = 0, leaf_ns = 0;
                probe_one(n, k, 32, &sb, &fft_c, &fft_u, &block_ns, &leaf_ns);
                double geo_sb = bucket_geo_mean(&sb);
                double geo_fft_c = bucket_geo_mean(&fft_c);
                double geo_fft_u = bucket_geo_mean(&fft_u);
                double min_fft_u = bucket_min(&fft_u);
                double max_fft_u = bucket_max(&fft_u);
                printf("%d,%d,%d,%d,%.4f,%d,%.4f,%d,%.4f,%.4f,%.4f,%.3f,%.3f\n",
                       n, k, 32,
                       sb.count, geo_sb,
                       fft_c.count, geo_fft_c,
                       fft_u.count, geo_fft_u, min_fft_u, max_fft_u,
                       block_ns, leaf_ns);
                fflush(stdout);
                bucket_free(&sb); bucket_free(&fft_c); bucket_free(&fft_u);
            }
        }
    }

    fprintf(stderr, "Done.\n");
    return 0;
}
