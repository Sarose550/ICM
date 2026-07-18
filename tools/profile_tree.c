/* profile_tree.c — Collect per-level tree build timings at multiple B values.
 *
 * Outputs CSV with one row per (B, level) pair:
 *   B,ell,fft_n,nr,cps,use_fft,calib_ns,level_ns,per_parent_ns,wrap_m
 *
 * Build: gcc -O3 -march=native -Isrc -Idevices/zen4 -o profile_tree tools/profile_tree.c -lfftw3 -lm -ldl
 */
#include "icm.c"
#include <stdio.h>

static void profile_one(int n, int k, int B) {
    double *S = (double *)malloc(n * sizeof(double));
    double *payout = (double *)malloc(k * sizeof(double));
    srand(42);
    for (int i = 0; i < n; i++) S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
    for (int q = 0; q < k; q++) payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

    HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
    TreeCtx *tc = hc->tc;

    /* Compute a values */
    double *a = (double *)malloc(n * sizeof(double));
    double logv = log(1.0 / hc->S_sorted[0]);
    for (int i = 0; i < n; i++) a[i] = exp(logv * hc->S_sorted[i]);

    /* Block build (populates tree leaves) */
    int leaf_psz = tc->psz[0];
    for (int b = 0; b < hc->nblocks; b++) {
        int start = b * B, end = start + B;
        if (end > n) end = n;
        double *P = hc->block_prods + (size_t)b * (B + 1);
        memset(P, 0, (B + 1) * sizeof(double));
        P[0] = 1.0;
        for (int j = start; j < end; j++) {
            double aj = a[j], bj = 1 - aj;
            for (int m = (end - start); m >= 1; m--)
                P[m] = aj * P[m] + bj * P[m - 1];
            P[0] *= aj;
        }
        double *leaf = tc->ws + tc->plev_off[0] + (size_t)b * leaf_psz;
        int cp = (B + 1 < leaf_psz) ? B + 1 : leaf_psz;
        memcpy(leaf, P, cp * sizeof(double));
        if (cp < leaf_psz) memset(leaf + cp, 0, (leaf_psz - cp) * sizeof(double));
    }
    for (int b = hc->nblocks; b < tc->N; b++) {
        double *leaf = tc->ws + tc->plev_off[0] + (size_t)b * leaf_psz;
        memset(leaf, 0, leaf_psz * sizeof(double));
        leaf[0] = 1.0;
    }

    /* Per-level tree build timing (3 reps, median) */
    for (int rep_pass = 0; rep_pass < 2; rep_pass++) {
        /* rep_pass 0 = warmup, 1 = measure */
        for (int ell = 1; ell < tc->L - 1; ell++) {
            int cps = tc->psz[ell - 1], pps = tc->psz[ell];
            double *child_base = tc->ws + tc->plev_off[ell - 1];
            double *parent_base = tc->ws + tc->plev_off[ell];
            int nr_parent = tc->n_real[ell];
            int nr_child = tc->n_real[ell - 1];

            double times[5];
            int n_reps = (rep_pass == 0) ? 1 : 5;

            for (int rep = 0; rep < n_reps; rep++) {
                /* Rebuild previous levels to get cold data */
                if (rep > 0) {
                    for (int e2 = 1; e2 < ell; e2++) {
                        int c2 = tc->psz[e2 - 1], p2 = tc->psz[e2];
                        double *cb2 = tc->ws + tc->plev_off[e2 - 1];
                        double *pb2 = tc->ws + tc->plev_off[e2];
                        int nr2 = tc->n_real[e2], nc2 = tc->n_real[e2 - 1];
                        for (int j = 0; j < nr2; j++) {
                            double *Lc = cb2 + (size_t)(2*j) * c2;
                            double *out = pb2 + (size_t)j * p2;
                            if (2*j+1 >= nc2) {
                                int cp = (c2 < p2) ? c2 : p2;
                                memcpy(out, Lc, cp * sizeof(double));
                            } else {
                                double *Rc = cb2 + (size_t)(2*j+1) * c2;
                                if (tc->use_fft[e2])
                                    polymul_fft_wrap(Lc, c2, Rc, c2, out, p2,
                                                     tc->fft, NULL, NULL,
                                                     tc->build_fft_n[e2], tc->build_wrap_m[e2]);
                                else
                                    polymul_modk(Lc, c2, Rc, c2, out, p2);
                            }
                        }
                    }
                }

                double t0 = now_ns();
                for (int j = 0; j < nr_parent; j++) {
                    double *Lc = child_base + (size_t)(2*j) * cps;
                    double *out = parent_base + (size_t)j * pps;
                    if (2*j+1 >= nr_child) {
                        int cp = (cps < pps) ? cps : pps;
                        memcpy(out, Lc, cp * sizeof(double));
                        if (cp < pps) memset(out + cp, 0, (pps - cp) * sizeof(double));
                    } else {
                        double *Rc = child_base + (size_t)(2*j+1) * cps;
                        if (tc->use_fft[ell])
                            polymul_fft_wrap(Lc, cps, Rc, cps, out, pps,
                                             tc->fft, NULL, NULL,
                                             tc->build_fft_n[ell], tc->build_wrap_m[ell]);
                        else
                            polymul_modk(Lc, cps, Rc, cps, out, pps);
                    }
                }
                times[rep] = now_ns() - t0;
            }

            if (rep_pass == 0) continue;

            /* Median of 5 */
            for (int i = 0; i < 5; i++)
                for (int j = i + 1; j < 5; j++)
                    if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
            double level_ns = times[2];

            /* Look up warm calibration */
            int bfn = tc->use_fft[ell] ? tc->build_fft_n[ell] : 0;
            double calib_val = 0;
            if (bfn > 0) {
                for (int i = 0; i < N_CALIBRATED_SIZES; i++) {
                    if (calib_sizes[i] == bfn) { calib_val = calib_times_ns[i]; break; }
                }
            }

            printf("%d,%d,%d,%d,%d,%d,%.1f,%.1f,%.1f,%d\n",
                   B, ell, bfn, nr_parent, cps,
                   tc->use_fft[ell] ? 1 : 0,
                   calib_val, level_ns, level_ns / nr_parent,
                   tc->use_fft[ell] ? tc->build_wrap_m[ell] : 0);
        }
    }

    free(a); free(S); free(payout);
    hybrid_ctx_destroy(hc);
}

int main(void) {
    build_fftw_size_table();
    wisdom_load();

    printf("B,ell,fft_n,nr,cps,use_fft,calib_ns,level_ns,per_parent_ns,wrap_m\n");

    int test_n[] = {16384, 32768, 65536};
    int test_B[] = {8, 16, 32, 64};

    for (int ni = 0; ni < 3; ni++) {
        int n = test_n[ni];
        for (int bi = 0; bi < 4; bi++) {
            int B = test_B[bi];
            fprintf(stderr, "profiling n=%d B=%d...\n", n, B);
            profile_one(n, n, B);
            fflush(stdout);
        }
    }

    return 0;
}
