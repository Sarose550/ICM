/* perf_level.c — Read hardware counters for individual tree levels using
 * perf_event_open. Counters are started/stopped precisely around the
 * target level's work, excluding rebuild overhead.
 *
 * Build: gcc -O3 -march=native -g -Isrc -Idevices/zen4 -o perf_level tools/perf_level.c -lfftw3 -lm -ldl
 */
#include "icm.c"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>

struct perf_fd {
    int fd;
    const char *name;
};

static int perf_open(uint32_t type, uint64_t config) {
    struct perf_event_attr pe = {0};
    pe.type = type;
    pe.size = sizeof(pe);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    return (int)syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
}

#define N_COUNTERS 6
static struct perf_fd counters[N_COUNTERS];

static void perf_setup(void) {
    counters[0] = (struct perf_fd){perf_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES), "cycles"};
    counters[1] = (struct perf_fd){perf_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS), "instructions"};
    counters[2] = (struct perf_fd){perf_open(PERF_TYPE_HW_CACHE,
        PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
        "L1d_misses"};
    counters[3] = (struct perf_fd){perf_open(PERF_TYPE_HW_CACHE,
        PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
        "L1d_loads"};
    counters[4] = (struct perf_fd){perf_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES), "LLC_refs"};
    counters[5] = (struct perf_fd){perf_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES), "LLC_misses"};
    for (int i = 0; i < N_COUNTERS; i++)
        if (counters[i].fd < 0)
            fprintf(stderr, "WARNING: could not open counter %s\n", counters[i].name);
}

static void perf_start(void) {
    for (int i = 0; i < N_COUNTERS; i++)
        if (counters[i].fd >= 0) {
            ioctl(counters[i].fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(counters[i].fd, PERF_EVENT_IOC_ENABLE, 0);
        }
}

static void perf_stop(void) {
    for (int i = 0; i < N_COUNTERS; i++)
        if (counters[i].fd >= 0)
            ioctl(counters[i].fd, PERF_EVENT_IOC_DISABLE, 0);
}

static long long perf_read(int idx) {
    long long val = 0;
    if (counters[idx].fd >= 0)
        read(counters[idx].fd, &val, sizeof(val));
    return val;
}

static void rebuild_below(TreeCtx *tc, int target_ell) {
    for (int e = 1; e < target_ell; e++) {
        int c = tc->psz[e-1], p = tc->psz[e];
        double *cb = tc->ws + tc->plev_off[e-1];
        double *pb = tc->ws + tc->plev_off[e];
        int nrp = tc->n_real[e], nrc = tc->n_real[e-1];
        for (int j = 0; j < nrp; j++) {
            double *Lc = cb + (size_t)(2*j) * c;
            double *out = pb + (size_t)j * p;
            if (2*j+1 >= nrc)
                memcpy(out, Lc, ((c < p) ? c : p) * sizeof(double));
            else {
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
    perf_setup();

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

    printf("ell,fft_n,nr,cps,ns_per_parent,cycles,instructions,IPC,L1d_loads,L1d_misses,L1d_miss_pct,LLC_refs,LLC_misses,LLC_miss_pct\n");

    for (int ell = 2; ell < tc->L - 1; ell++) {
        if (!tc->use_fft[ell]) continue;
        int cps = tc->psz[ell-1], pps = tc->psz[ell];
        int nr = tc->n_real[ell], nc = tc->n_real[ell-1];
        int fft_n = tc->build_fft_n[ell];
        int wrap_m = tc->build_wrap_m[ell];
        double *child_base = tc->ws + tc->plev_off[ell-1];
        double *parent_base = tc->ws + tc->plev_off[ell];

        /* Rebuild to get realistic cache state */
        rebuild_below(tc, ell);

        /* START counters + timer — ONLY around the target level */
        perf_start();
        double t0 = now_ns();

        for (int j = 0; j < nr; j++) {
            double *Lc = child_base + (size_t)(2*j) * cps;
            double *out = parent_base + (size_t)j * pps;
            if (2*j+1 >= nc) {
                memcpy(out, Lc, ((cps < pps) ? cps : pps) * sizeof(double));
            } else {
                double *Rc = child_base + (size_t)(2*j+1) * cps;
                polymul_fft_wrap(Lc, cps, Rc, cps, out, pps,
                                 tc->fft, NULL, NULL, fft_n, wrap_m);
            }
        }

        double elapsed = now_ns() - t0;
        perf_stop();

        long long cycles = perf_read(0);
        long long instrs = perf_read(1);
        long long l1_miss = perf_read(2);
        long long l1_load = perf_read(3);
        long long llc_ref = perf_read(4);
        long long llc_miss = perf_read(5);
        double ipc = (cycles > 0) ? (double)instrs / cycles : 0;
        double l1_pct = (l1_load > 0) ? (double)l1_miss / l1_load * 100 : 0;
        double llc_pct = (llc_ref > 0) ? (double)llc_miss / llc_ref * 100 : 0;

        printf("%d,%d,%d,%d,%.1f,%lld,%lld,%.2f,%lld,%lld,%.2f,%lld,%lld,%.2f\n",
               ell, fft_n, nr, cps, elapsed / nr,
               cycles, instrs, ipc, l1_load, l1_miss, l1_pct,
               llc_ref, llc_miss, llc_pct);
        fflush(stdout);
    }

    free(a); free(S); free(payout);
    hybrid_ctx_destroy(hc);
    return 0;
}
