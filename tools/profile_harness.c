/* profile_harness.c — Long-running harness for sampling profiler.
 * Runs icm_equity in a tight loop for target wall-clock seconds.
 *
 * Usage: ./tools/profile_harness <n> <k> <B> <Q> <target_seconds>
 *   B=0 means use default dispatch (don't set ICM_FORCE_B).
 *   Prints PID then runs. Attach: /usr/bin/sample <pid> <duration> -f /tmp/profile.txt
 *
 * Build:
 *   gcc -O3 -march=native -Isrc -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o tools/profile_harness tools/profile_harness.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 */
#include "icm.c"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <n> <k> <B> <Q> <target_seconds>\n", argv[0]);
        fprintf(stderr, "  B=0: use default dispatch (no ICM_FORCE_B)\n");
        return 1;
    }

    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    int B = atoi(argv[3]);
    int Q = atoi(argv[4]);
    double target_sec = atof(argv[5]);

    build_fftw_size_table();
    icm_init(NULL);

    double *S = (double *)malloc(n * sizeof(double));
    double *equity = (double *)malloc(n * sizeof(double));
    double *payout = (double *)malloc(k * sizeof(double));
    if (!S || !equity || !payout) { fprintf(stderr, "OOM\n"); return 1; }

    srand(42);
    for (int i = 0; i < n; i++) S[i] = 100.0 + 9900.0 * ((double)rand() / RAND_MAX);
    for (int q = 0; q < k; q++) payout[q] = 1.0 / (q + 1) - 1.0 / (q + 2);

    if (B > 0) {
        char env_buf[32];
        snprintf(env_buf, sizeof(env_buf), "%d", B);
        setenv("ICM_FORCE_B", env_buf, 1);
    } else {
        unsetenv("ICM_FORCE_B");
    }

    /* Warmup: one call to init FFTW plans, caches */
    printf("Warmup...\n");
    fflush(stdout);
    icm_equity(n, S, Q, payout, k, equity);

    /* Time one call to estimate how many we need */
    double t0 = now_ns();
    icm_equity(n, S, Q, payout, k, equity);
    double one_call_ns = now_ns() - t0;
    double one_call_sec = one_call_ns / 1e9;
    long long niters = (long long)(target_sec / one_call_sec);
    if (niters < 3) niters = 3; /* at least a few */
    if (niters > 1000000) niters = 1000000;

    printf("PID=%d  n=%d k=%d B=%d Q=%d  one_call=%.3fms  niters=%lld  est_wall=%.1fs\n",
           getpid(), n, k, B, Q, one_call_ns/1e6, niters, niters * one_call_sec);
    fflush(stdout);

    /* Write PID to a file for reliable profiler attachment */
    {
        FILE *pidf = fopen("/tmp/profile_harness_pid.txt", "w");
        if (pidf) {
            fprintf(pidf, "%d\n", getpid());
            fclose(pidf);
        }
    }
    printf("PID file written. Sleeping 3s for profiler attachment...\n");
    fflush(stdout);
    sleep(3);

    double start_ns = now_ns();
    for (long long rep = 0; rep < niters; rep++) {
        icm_equity(n, S, Q, payout, k, equity);
    }
    double elapsed_ns = now_ns() - start_ns;
    double per_qp_ns = elapsed_ns / (niters * Q);

    printf("Done. niters=%lld  total=%.3fs  per_qp_ns=%.0f\n",
           niters, elapsed_ns/1e9, per_qp_ns);

    if (B > 0) unsetenv("ICM_FORCE_B");
    free(S); free(equity); free(payout);
    return 0;
}
