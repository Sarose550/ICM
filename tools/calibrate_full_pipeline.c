/* calibrate_full_pipeline.c — Measure full per-parent polymul_fft_wrap cost
 * under two conditions: clean (same plan reused) and polluted (different-size
 * plan interleaved, simulating the tree's level-by-level traversal).
 *
 * Outputs CSV: fft_n,old_calib_ns,clean_ns,polluted_ns,pollution_ratio
 *
 * Build: gcc -O3 -march=native -Isrc -Idevices/zen4 -o calibrate_full_pipeline tools/calibrate_full_pipeline.c -lfftw3 -lm -ldl
 */
#include "icm.c"
#include <stdio.h>

static double measure_clean(FFTCache *fc, int fft_n, int nr,
                            double *child, double *out) {
    int cps = fft_n, pps = 2 * cps - 1;
    /* Warmup */
    for (int p = 0; p < nr; p++)
        polymul_fft_wrap(child + (size_t)(2*p)*cps, cps,
                         child + (size_t)(2*p+1)*cps, cps,
                         out + (size_t)p*pps, pps, fc, NULL, NULL, fft_n, 0);
    /* Measure (3 passes, median) */
    double times[3];
    for (int pass = 0; pass < 3; pass++) {
        double t0 = now_ns();
        for (int p = 0; p < nr; p++)
            polymul_fft_wrap(child + (size_t)(2*p)*cps, cps,
                             child + (size_t)(2*p+1)*cps, cps,
                             out + (size_t)p*pps, pps, fc, NULL, NULL, fft_n, 0);
        times[pass] = (now_ns() - t0) / nr;
    }
    for (int i = 0; i < 3; i++)
        for (int j = i+1; j < 3; j++)
            if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
    return times[1];
}

static double measure_polluted(FFTCache *fc, int fft_n, int nr,
                               double *child, double *out,
                               int polluter_fft_n,
                               double *poll_child, double *poll_out, int poll_nr) {
    int cps = fft_n, pps = 2 * cps - 1;
    int pcps = polluter_fft_n, ppps = 2 * pcps - 1;
    /* Warmup */
    for (int p = 0; p < poll_nr; p++)
        polymul_fft_wrap(poll_child + (size_t)(2*p)*pcps, pcps,
                         poll_child + (size_t)(2*p+1)*pcps, pcps,
                         poll_out + (size_t)p*ppps, ppps, fc, NULL, NULL, polluter_fft_n, 0);
    for (int p = 0; p < nr; p++)
        polymul_fft_wrap(child + (size_t)(2*p)*cps, cps,
                         child + (size_t)(2*p+1)*cps, cps,
                         out + (size_t)p*pps, pps, fc, NULL, NULL, fft_n, 0);
    /* Measure: run polluter, then measure target */
    double times[3];
    for (int pass = 0; pass < 3; pass++) {
        /* Pollute: run a full pass at the polluter size */
        for (int p = 0; p < poll_nr; p++)
            polymul_fft_wrap(poll_child + (size_t)(2*p)*pcps, pcps,
                             poll_child + (size_t)(2*p+1)*pcps, pcps,
                             poll_out + (size_t)p*ppps, ppps, fc, NULL, NULL, polluter_fft_n, 0);
        /* Now measure target (plan is cold) */
        double t0 = now_ns();
        for (int p = 0; p < nr; p++)
            polymul_fft_wrap(child + (size_t)(2*p)*cps, cps,
                             child + (size_t)(2*p+1)*cps, cps,
                             out + (size_t)p*pps, pps, fc, NULL, NULL, fft_n, 0);
        times[pass] = (now_ns() - t0) / nr;
    }
    for (int i = 0; i < 3; i++)
        for (int j = i+1; j < 3; j++)
            if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
    return times[1];
}

int main(void) {
    build_fftw_size_table();
    wisdom_load();

    /* Test sizes: the actual FFT sizes that appear in the tree */
    int test_sizes[] = {
        56, 64, 80, 128, 192, 256, 384, 512, 768, 1024,
        2048, 4096, 8192, 16384, 17920, 32768, 33600
    };
    int n_test = sizeof(test_sizes) / sizeof(test_sizes[0]);

    /* Create cache with all sizes (so plan lookups work for polluters) */
    FFTCache *fc = fft_cache_create_sizes(test_sizes, n_test);

    printf("fft_n,old_calib_ns,clean_ns,polluted_ns,pollution_ratio\n");

    for (int si = 0; si < n_test; si++) {
        int fft_n = test_sizes[si];
        int cps = fft_n, pps = 2 * cps - 1;

        /* Look up old calibration */
        double old_calib = -1;
        for (int i = 0; i < N_CALIBRATED_SIZES; i++)
            if (calib_sizes[i] == fft_n) { old_calib = calib_times_ns[i]; break; }

        /* Number of parents: enough to exceed L2 */
        int nr = (4 * 1048576) / (2 * cps * 8);
        if (nr < 16) nr = 16;
        if (nr > 2048) nr = 2048;

        /* Allocate packed data */
        double *child = (double *)calloc((size_t)2 * nr * cps, sizeof(double));
        double *out = (double *)calloc((size_t)nr * pps, sizeof(double));
        for (int p = 0; p < nr; p++)
            for (int j = 0; j < 2 * cps; j++)
                child[(size_t)(2*p)*cps + j] = 1.0 / (j + 1 + p * 0.001);

        /* Pick a polluter: use the next-smaller size (simulates adjacent level) */
        int poll_si = (si > 0) ? si - 1 : si + 1;
        int poll_fft_n = test_sizes[poll_si];
        int poll_cps = poll_fft_n, poll_pps = 2 * poll_cps - 1;
        int poll_nr = (2 * 1048576) / (2 * poll_cps * 8);
        if (poll_nr < 16) poll_nr = 16;
        if (poll_nr > 1024) poll_nr = 1024;
        double *poll_child = (double *)calloc((size_t)2 * poll_nr * poll_cps, sizeof(double));
        double *poll_out = (double *)calloc((size_t)poll_nr * poll_pps, sizeof(double));
        for (int p = 0; p < poll_nr; p++)
            for (int j = 0; j < 2 * poll_cps; j++)
                poll_child[(size_t)(2*p)*poll_cps + j] = 1.0 / (j + 1 + p * 0.002);

        double clean = measure_clean(fc, fft_n, nr, child, out);
        double polluted = measure_polluted(fc, fft_n, nr, child, out,
                                           poll_fft_n, poll_child, poll_out, poll_nr);

        printf("%d,%.1f,%.1f,%.1f,%.2f\n",
               fft_n, old_calib, clean, polluted, polluted / clean);
        fflush(stdout);
        fprintf(stderr, "  fft_n=%5d: old=%8.0f clean=%8.0f polluted=%8.0f ratio=%.2f\n",
                fft_n, old_calib, clean, polluted, polluted / clean);

        free(child); free(out); free(poll_child); free(poll_out);
    }
    return 0;
}
