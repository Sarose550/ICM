/* sample_plans.c — Sample 200 plans and measure avg-over-256 Q-point runtime.
 *
 * For each (n, k, B) triple, creates a plan (tree geometry), runs icm_equity
 * with Q=256, and reports the total time plus the plan details needed for
 * the cost model fit.
 *
 * Output CSV: n,k,B,Q,total_ms,per_qp_ns,L,levels_json
 * where levels_json encodes per-level: ell,nr,fft_n,wrap_m,cps,use_fft,cache_ok,corr_fft_n,corr_wrap_m
 *
 * Build (DEVICE = target device dir under devices/, e.g. zen4 or m3_pro):
 *   # macOS
 *   gcc -O3 -march=native -Isrc -Idevices/<DEVICE> -I/opt/homebrew/include \
 *       -o sample_plans tools/sample_plans.c -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 *   # Linux
 *   gcc -O3 -march=native -Isrc -Idevices/<DEVICE> \
 *       -o sample_plans tools/sample_plans.c -lfftw3 -lm -ldl -lmvec
 */
#include "icm.c"
#include <stdio.h>

static void emit_plan(int n, int k, int B) {
    double *S = (double *)malloc(n * sizeof(double));
    double *equity = (double *)malloc(n * sizeof(double));
    double *payout = (double *)malloc(k * sizeof(double));
    if (!S || !equity || !payout) { fprintf(stderr, "OOM n=%d\n", n); goto cleanup; }

    srand(42);
    for (int i = 0; i < n; i++) S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
    for (int q = 0; q < k; q++) payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

    /* Get the plan details by creating a tree context */
    int nblocks = (n + B - 1) / B;
    TreeCtx *tc = tree_ctx_create_ex2(nblocks, B, k, B);
    if (!tc) { fprintf(stderr, "tree_ctx failed n=%d B=%d\n", n, B); goto cleanup; }

    /* Run the hybrid engine directly at the requested B (3 reps, median).
     * NOT via icm_equity()/select_engine() — dispatch is cost-model-driven
     * and may pick linear instead, which would silently corrupt the sample
     * (this is a calibration tool: it must measure hybrid at *this* B). */
    int Q = 256;
    double times[3];
    for (int rep = 0; rep < 3; rep++) {
        HybridCtx *hc = hybrid_ctx_create(n, S, k, B);
        double t0 = now_ns();
        run_engine_ctx(n, S, Q, payout, k, equity, engine_hybrid_ctx, hc);
        times[rep] = now_ns() - t0;
        hybrid_ctx_destroy(hc);
    }

    /* Median */
    for (int i = 0; i < 3; i++)
        for (int j = i+1; j < 3; j++)
            if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
    double total_ns = times[1];
    double per_qp_ns = total_ns / Q;

    /* Emit CSV row: plan + measurement */
    /* Header fields: n,k,B,L,total_ms,per_qp_ns */
    printf("%d,%d,%d,%d,%.3f,%.1f",
           n, k, B, tc->L, total_ns / 1e6, per_qp_ns);

    /* Per-level details as semicolon-separated fields */
    for (int ell = 1; ell < tc->L; ell++) {
        int cps = tc->psz[ell-1];
        int nr = tc->n_real[ell];
        int use_fft = tc->use_fft[ell];
        int bfn = use_fft ? tc->build_fft_n[ell] : 0;
        int bwm = use_fft ? tc->build_wrap_m[ell] : 0;
        int cache = tc->fft_cache_ok[ell];
        int cfn = use_fft ? tc->corr_fft_n[ell] : 0;
        int cwm = use_fft ? tc->corr_wrap_m[ell] : 0;
        int below = tc->below_sat[ell];
        int g_need = tc->g_needed[ell-1];
        printf(",%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
               nr, cps, use_fft, bfn, bwm, cache, cfn, cwm, below, g_need);
    }
    printf("\n");
    fflush(stdout);

    fprintf(stderr, "  n=%d k=%d B=%d L=%d -> %.1f ms (%.0f ns/qp)\n",
            n, k, B, tc->L, total_ns / 1e6, per_qp_ns);

    tree_ctx_destroy(tc);
cleanup:
    free(S); free(equity); free(payout);
}

int main(void) {
    build_fftw_size_table();
    icm_init(NULL);

    int n_values[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int n_n = 9;
    int B_values[] = {8, 16, 24, 32, 48, 64};
    int n_B = 6;

    /* k fractions of n */
    double k_fracs[] = {0.1, 0.25, 0.5, 1.0};
    int n_kf = 4;

    /* Header */
    printf("n,k,B,L,total_ms,per_qp_ns");
    /* Variable number of level columns — reader must parse by L */
    printf(",levels...\n");

    int count = 0;
    for (int ni = 0; ni < n_n; ni++) {
        int n = n_values[ni];
        for (int bi = 0; bi < n_B; bi++) {
            int B = B_values[bi];
            if (B > n) continue;
            for (int ki = 0; ki < n_kf; ki++) {
                int k = (int)(n * k_fracs[ki]);
                if (k < 4) k = 4;
                if (k > n) k = n;
                if (B > k) continue;

                /* Skip if this would take too long (> ~30 sec) */
                if ((double)n * k > 2e9) continue;

                emit_plan(n, k, B);
                count++;

                if (count >= 200) goto done;
            }
        }
    }
done:
    fprintf(stderr, "\nSampled %d plans\n", count);
    return 0;
}
