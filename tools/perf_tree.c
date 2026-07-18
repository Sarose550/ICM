/* perf_tree.c — Profile individual tree levels with perf stat.
 * Builds the tree normally, then re-runs each level individually
 * under perf stat to get hardware counters per level.
 *
 * Usage: compile, then run via:
 *   perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,\
 *   dTLB-loads,dTLB-load-misses,cache-references,cache-misses \
 *   ./perf_tree <ell>
 *
 * Or run without perf for timing only:
 *   ./perf_tree          (all levels)
 *   ./perf_tree 5        (just level 5)
 */
#include "icm.c"
#include <stdio.h>

static void rebuild_below(TreeCtx *tc, int target_ell) {
    for (int e = 1; e < target_ell; e++) {
        int c = tc->psz[e-1], p = tc->psz[e];
        double *cb = tc->ws + tc->plev_off[e-1];
        double *pb = tc->ws + tc->plev_off[e];
        int nrp = tc->n_real[e], nrc = tc->n_real[e-1];
        for (int j = 0; j < nrp; j++) {
            double *Lc = cb + (size_t)(2*j) * c;
            double *out = pb + (size_t)j * p;
            if (2*j+1 >= nrc) {
                memcpy(out, Lc, ((c < p) ? c : p) * sizeof(double));
            } else {
                double *Rc = cb + (size_t)(2*j+1) * c;
                if (tc->use_fft[e])
                    polymul_fft_wrap(Lc, c, Rc, c, out, p, tc->fft, NULL, NULL,
                                     tc->build_fft_n[e], tc->build_wrap_m[e]);
                else
                    polymul_modk(Lc, c, Rc, c, out, p);
            }
        }
    }
}

static void run_level(TreeCtx *tc, int ell) {
    int cps = tc->psz[ell-1], pps = tc->psz[ell];
    double *child_base = tc->ws + tc->plev_off[ell-1];
    double *parent_base = tc->ws + tc->plev_off[ell];
    int nr = tc->n_real[ell], nc = tc->n_real[ell-1];
    int fft_n = tc->build_fft_n[ell];
    int wrap_m = tc->build_wrap_m[ell];

    for (int j = 0; j < nr; j++) {
        double *Lc = child_base + (size_t)(2*j) * cps;
        double *out = parent_base + (size_t)j * pps;
        if (2*j+1 >= nc) {
            memcpy(out, Lc, ((cps < pps) ? cps : pps) * sizeof(double));
        } else {
            double *Rc = child_base + (size_t)(2*j+1) * cps;
            if (tc->use_fft[ell])
                polymul_fft_wrap(Lc, cps, Rc, cps, out, pps,
                                 tc->fft, NULL, NULL, fft_n, wrap_m);
            else
                polymul_modk(Lc, cps, Rc, cps, out, pps);
        }
    }
}

int main(int argc, char **argv) {
    build_fftw_size_table();
    wisdom_load();

    int n = 65536, k = 65536, B = 16;
    int target_ell = (argc > 1) ? atoi(argv[1]) : -1;

    double *S = (double *)malloc(n * sizeof(double));
    double *payout = (double *)malloc(k * sizeof(double));
    srand(42);
    for (int i = 0; i < n; i++) S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
    for (int q = 0; q < k; q++) payout[q] = 1.0 / (q+1) - 1.0 / (q+2);

    HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
    TreeCtx *tc = hc->tc;

    /* Compute a values */
    double *a = (double *)malloc(n * sizeof(double));
    double logv = log(1.0 / hc->S_sorted[0]);
    for (int i = 0; i < n; i++) a[i] = exp(logv * hc->S_sorted[i]);

    /* Build leaves */
    int leaf_psz = tc->psz[0];
    for (int b = 0; b < hc->nblocks; b++) {
        int start = b*B, end = start+B;
        if (end > n) end = n;
        double *P = hc->block_prods + (size_t)b * (B+1);
        memset(P, 0, (B+1) * sizeof(double));
        P[0] = 1.0;
        for (int j = start; j < end; j++) {
            double aj = a[j], bj = 1 - aj;
            for (int m = (end-start); m >= 1; m--)
                P[m] = aj * P[m] + bj * P[m-1];
            P[0] *= aj;
        }
        double *leaf = tc->ws + tc->plev_off[0] + (size_t)b * leaf_psz;
        int cp = (B+1 < leaf_psz) ? B+1 : leaf_psz;
        memcpy(leaf, P, cp * sizeof(double));
        if (cp < leaf_psz) memset(leaf + cp, 0, (leaf_psz - cp) * sizeof(double));
    }
    for (int b = hc->nblocks; b < tc->N; b++) {
        double *leaf = tc->ws + tc->plev_off[0] + (size_t)b * leaf_psz;
        memset(leaf, 0, leaf_psz * sizeof(double));
        leaf[0] = 1.0;
    }

    /* Initial full build */
    tree_build_levels(tc);

    if (target_ell > 0) {
        /* Single level: rebuild below, then run target (good for perf stat) */
        fprintf(stderr, "Profiling level %d: fft_n=%d nr=%d cps=%d wrap_m=%d\n",
                target_ell, tc->build_fft_n[target_ell], tc->n_real[target_ell],
                tc->psz[target_ell-1], tc->build_wrap_m[target_ell]);
        rebuild_below(tc, target_ell);
        /* Run target level 10x for statistical stability */
        for (int rep = 0; rep < 10; rep++) {
            rebuild_below(tc, target_ell);
            run_level(tc, target_ell);
        }
    } else {
        /* All levels: time each individually */
        for (int ell = 2; ell < tc->L - 1; ell++) {
            if (!tc->use_fft[ell]) continue;
            rebuild_below(tc, ell);
            double t0 = now_ns();
            run_level(tc, ell);
            double elapsed = now_ns() - t0;
            int nr = tc->n_real[ell];
            printf("ell=%d fft_n=%d nr=%d cps=%d wrap=%d  %.0f ns/parent  %.0f us total\n",
                   ell, tc->build_fft_n[ell], nr, tc->psz[ell-1],
                   tc->build_wrap_m[ell], elapsed / nr, elapsed / 1000);
        }
    }

    free(a); free(S); free(payout);
    hybrid_ctx_destroy(hc);
    return 0;
}
