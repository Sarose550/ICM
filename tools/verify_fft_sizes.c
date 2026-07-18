/* verify_fft_sizes.c — For each tree level, measure polymul_fft_wrap at
 * several candidate FFT sizes and verify best_fft_config chose optimally.
 *
 * Build: gcc -O3 -march=native -Isrc -Idevices/zen4 -o verify_fft_sizes tools/verify_fft_sizes.c -lfftw3 -lm -ldl
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

int main(void) {
    build_fftw_size_table();
    wisdom_load();

    int n = 65536, k = 65536, B = 16;
    double *S = (double *)malloc(n * sizeof(double));
    double *payout = (double *)malloc(k * sizeof(double));
    srand(42);
    for (int i = 0; i < n; i++) S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
    for (int q = 0; q < k; q++) payout[q] = 1.0 / (q+1) - 1.0 / (q+2);

    HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
    TreeCtx *tc = hc->tc;
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
    tree_build_levels(tc);

    /* All candidate FFT sizes */
    int all_sizes[] = {56, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512,
                       768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
                       16384, 17920, 32768, 33600};
    int n_all = sizeof(all_sizes) / sizeof(all_sizes[0]);
    FFTCache *fc_all = fft_cache_create_sizes(all_sizes, n_all);

    for (int ell = 2; ell < tc->L - 1 && ell <= 9; ell++) {
        if (!tc->use_fft[ell]) continue;
        int cps = tc->psz[ell-1], pps = tc->psz[ell];
        int conv = 2 * cps - 1;
        int nr = tc->n_real[ell], nc = tc->n_real[ell-1];
        int chosen_fn = tc->build_fft_n[ell];
        int chosen_wm = tc->build_wrap_m[ell];
        double *child_base = tc->ws + tc->plev_off[ell-1];
        double *parent_base = tc->ws + tc->plev_off[ell];

        fprintf(stderr, "\nell=%d cps=%d conv=%d nr=%d chosen_fft_n=%d chosen_wrap_m=%d\n",
                ell, cps, conv, nr, chosen_fn, chosen_wm);

        double best_pp = 1e18;
        int best_fn = 0;

        for (int si = 0; si < n_all; si++) {
            int fn = all_sizes[si];
            if (fn < conv / 2 + 1 || fn > 2 * conv) continue;
            int wm = (fn >= conv) ? 0 : (conv - fn);

            rebuild_below(tc, ell);
            double t0 = now_ns();
            for (int j = 0; j < nr; j++) {
                double *Lc = child_base + (size_t)(2*j) * cps;
                double *out = parent_base + (size_t)j * pps;
                if (2*j+1 >= nc) {
                    memcpy(out, Lc, ((cps < pps) ? cps : pps) * sizeof(double));
                } else {
                    double *Rc = child_base + (size_t)(2*j+1) * cps;
                    polymul_fft_wrap(Lc, cps, Rc, cps, out, pps,
                                     fc_all, NULL, NULL, fn, wm);
                }
            }
            double pp = (now_ns() - t0) / nr;

            int is_chosen = (fn == chosen_fn);
            char mark = is_chosen ? '*' : ' ';
            if (pp < best_pp) { best_pp = pp; best_fn = fn; }
            fprintf(stderr, "  %c fft_n=%5d wrap_m=%3d  %.0f ns/parent  level=%.0f us\n",
                    mark, fn, wm, pp, pp * nr / 1000);
        }

        int optimal = (chosen_fn == best_fn);
        fprintf(stderr, "  -> best=%d (%.0f ns)  chosen=%d  %s\n",
                best_fn, best_pp, chosen_fn,
                optimal ? "OPTIMAL" : "SUBOPTIMAL");
        printf("ell=%d conv=%d nr=%d chosen=%d best_measured=%d %s\n",
               ell, conv, nr, chosen_fn, best_fn,
               optimal ? "OK" : "WRONG");
        fflush(stdout);
    }

    free(a); free(S); free(payout);
    hybrid_ctx_destroy(hc);
    return 0;
}
