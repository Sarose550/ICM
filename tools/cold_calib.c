/* cold_calib.c — Measure actual per-parent polymul cost with cold data.
 *
 * For each FFT size, runs polymul_fft_wrap on many independent input pairs
 * so that each parent's data is NOT in cache. This matches the tree's
 * actual behavior where each parent processes different polynomials.
 *
 * Compare output against calib_times_ns[] to see the warm-vs-cold gap.
 *
 * Build: gcc -O3 -march=native -Isrc -Idevices/zen4 -o cold_calib tools/cold_calib.c -lfftw3 -lm -ldl
 */
#include "icm.c"
#include <stdio.h>

int main(void) {
    build_fftw_size_table();
    wisdom_load();

    int test_sizes[] = {56, 64, 80, 128, 192, 256, 384, 512, 768, 1024,
                        2048, 4096, 8192, 16384, 17920, 32768, 33600, 65536};
    int n_test = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("fft_n,warm_calib_ns,cold_per_parent_ns,ratio,cold_ns_per_element\n");

    for (int ti = 0; ti < n_test; ti++) {
        int fft_n = test_sizes[ti];

        /* Find stored warm-cache calibration value */
        int idx = -1;
        for (int i = 0; i < N_CALIBRATED_SIZES; i++)
            if (calib_sizes[i] == fft_n) { idx = i; break; }
        double warm_calib = (idx >= 0) ? calib_times_ns[idx] : -1;

        /* Number of parents: enough so total data >> L2 (1 MB on Zen4).
         * Each parent touches ~4*fft_n*8 bytes. */
        int bytes_per_parent = 4 * fft_n * 8;
        int n_parents = (4 * 1048576) / (bytes_per_parent > 0 ? bytes_per_parent : 1);
        if (n_parents < 8) n_parents = 8;
        if (n_parents > 2048) n_parents = 2048;

        /* Polynomial size: cps = fft_n/2 (typical tree level) */
        int cps = fft_n / 2;
        if (cps < 2) cps = 2;
        int conv_len = 2 * cps - 1;
        int pps = conv_len;

        /* Allocate independent arrays for all parents */
        double **a_arr = (double **)malloc(n_parents * sizeof(double *));
        double **b_arr = (double **)malloc(n_parents * sizeof(double *));
        double **c_arr = (double **)malloc(n_parents * sizeof(double *));
        for (int p = 0; p < n_parents; p++) {
            a_arr[p] = (double *)malloc(cps * sizeof(double));
            b_arr[p] = (double *)malloc(cps * sizeof(double));
            c_arr[p] = (double *)calloc(pps, sizeof(double));
            for (int j = 0; j < cps; j++) {
                a_arr[p][j] = 1.0 / (j + 1 + p);
                b_arr[p][j] = 1.0 / (j + 2 + p);
            }
        }

        /* Create FFT cache with a plan for this size */
        FFTCache *fc = fft_cache_create_sizes(&fft_n, 1);
        if (!fc) {
            printf("%d,%.1f,-1,-1,-1\n", fft_n, warm_calib);
            goto cleanup;
        }
        FFTPlan *plan = fft_cache_get(fc, fft_n);
        if (!plan) {
            printf("%d,%.1f,-1,-1,-1\n", fft_n, warm_calib);
            goto cleanup;
        }
        int wrap_m = (plan->fft_n >= conv_len) ? 0 : (conv_len - plan->fft_n);

        /* Warmup pass */
        for (int p = 0; p < n_parents; p++) {
            polymul_fft_wrap(a_arr[p], cps, b_arr[p], cps,
                             c_arr[p], pps, fc, NULL, NULL,
                             fft_n, wrap_m);
        }

        /* Timed: 3 passes, take median */
        double pass_times[3];
        for (int pass = 0; pass < 3; pass++) {
            double t0 = now_ns();
            for (int p = 0; p < n_parents; p++) {
                polymul_fft_wrap(a_arr[p], cps, b_arr[p], cps,
                                 c_arr[p], pps, fc, NULL, NULL,
                                 fft_n, wrap_m);
            }
            pass_times[pass] = (now_ns() - t0) / n_parents;
        }
        /* Sort for median */
        for (int i = 0; i < 3; i++)
            for (int j = i + 1; j < 3; j++)
                if (pass_times[j] < pass_times[i]) {
                    double t = pass_times[i];
                    pass_times[i] = pass_times[j];
                    pass_times[j] = t;
                }
        double cold = pass_times[1];

        double ratio = (warm_calib > 0) ? cold / warm_calib : -1;
        printf("%d,%.1f,%.1f,%.2f,%.2f\n",
               fft_n, warm_calib, cold, ratio, cold / fft_n);
        fflush(stdout);

cleanup:
        for (int p = 0; p < n_parents; p++) {
            free(a_arr[p]); free(b_arr[p]); free(c_arr[p]);
        }
        free(a_arr); free(b_arr); free(c_arr);
    }
    return 0;
}
