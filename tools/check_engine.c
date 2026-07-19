/* check_engine.c — Verify which engine is being used for the anomalous case.
 *
 * Build: gcc -O3 -march=native -Isrc -Idevices/m3_max -I/opt/homebrew/include
 *        -o tools/check_engine tools/check_engine.c
 *        -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 */
#include "icm.c"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    build_fftw_size_table();
    icm_init(NULL);

    /* Test what select_engine returns (0=linear, B>0=hybrid with block size B) */
    printf("select_engine results:\n");
    printf("  n=16384, k=1638: B=%d\n", icm_select_engine(16384, 1638));
    printf("  n=16384, k=4096: B=%d\n", icm_select_engine(16384, 4096));
    printf("  n=16384, k=2048: B=%d\n", icm_select_engine(16384, 2048));
    printf("  n=8192,  k=819:  B=%d\n", icm_select_engine(8192, 819));
    printf("  n=8192,  k=2048: B=%d\n", icm_select_engine(8192, 2048));
    printf("  n=4096,  k=409:  B=%d\n", icm_select_engine(4096, 409));
    printf("  n=4096,  k=1024: B=%d\n", icm_select_engine(4096, 1024));
    printf("  n=4096,  k=2048: B=%d\n", icm_select_engine(4096, 2048));

    /* select_best_B */
    printf("\nselect_best_B results:\n");
    printf("  n=16384, k=1638: B=%d\n", icm_select_best_B(16384, 1638));
    printf("  n=16384, k=4096: B=%d\n", icm_select_best_B(16384, 4096));

    /* Check if hybrid_ctx_create works for anomalous case */
    printf("\nTesting hybrid_ctx_create for n=16384, k=1638, B=32:\n");
    double *S = (double *)malloc(16384 * sizeof(double));
    for (int i = 0; i < 16384; i++) S[i] = 100.0;
    void *hc = icm_hybrid_ctx_create(16384, S, 1638, 32);
    printf("  ctx=%p\n", hc);
    if (hc) {
        /* Check the tree L and psz values */
        HybridCtx *h = (HybridCtx *)hc;
        printf("  B=%d nblocks=%d\n", h->B, h->nblocks);
        TreeCtx *tc = h->tc;
        printf("  Tree: L=%d N=%d\n", tc->L, tc->N);
        printf("  psz: ");
        for (int i = 0; i < tc->L; i++) printf("%d ", tc->psz[i]);
        printf("\n");
        printf("  n_real: ");
        for (int i = 0; i < tc->L; i++) printf("%d ", tc->n_real[i]);
        printf("\n");
        printf("  use_fft: ");
        for (int i = 0; i < tc->L; i++) printf("%d ", tc->use_fft[i]);
        printf("\n");
        printf("  below_sat: ");
        for (int i = 0; i < tc->L; i++) printf("%d ", tc->below_sat[i]);
        printf("\n");
        printf("  g_needed: ");
        for (int i = 0; i < tc->L; i++) printf("%d ", tc->g_needed[i]);
        printf("\n");
        printf("  build_fft_n: ");
        for (int i = 0; i < tc->L; i++) printf("%d ", tc->build_fft_n[i]);
        printf("\n");
        printf("  corr_fft_n: ");
        for (int i = 0; i < tc->L; i++) printf("%d ", tc->corr_fft_n[i]);
        printf("\n");
        printf("  fft_cache_ok: ");
        for (int i = 0; i < tc->L; i++) printf("%d ", tc->fft_cache_ok[i]);
        printf("\n");
        printf("  build_wrap_m: ");
        for (int i = 0; i < tc->L; i++) printf("%d ", tc->build_wrap_m[i]);
        printf("\n");
        printf("  corr_wrap_m: ");
        for (int i = 0; i < tc->L; i++) printf("%d ", tc->corr_wrap_m[i]);
        printf("\n");
        icm_ctx_destroy(hc, ICM_ENGINE_HYBRID);
    }
    free(S);

    return 0;
}
