/*
 * Generates reference vectors for WASM verification: deterministic inputs,
 * equities computed by the native library (calibrated device build), written
 * as JSON to stdout. web/verify/verify_node.mjs replays the same inputs
 * through the WASM module and compares.
 *
 * Build (from repo root, after `make DEVICE=m3_pro`):
 *   gcc -O3 -Isrc/cpu -Idevices/m3_pro -I/opt/homebrew/include \
 *       -o gen_reference web/verify/gen_reference.c build/libicm.a \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 *   ./gen_reference > web/verify/reference.json
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "icm.h"

static uint64_t rng_state;

static uint64_t splitmix64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* uniform double in [lo, hi) */
static double uni(double lo, double hi) {
    double u = (double)(splitmix64() >> 11) * (1.0 / 9007199254740992.0);
    return lo + u * (hi - lo);
}

static void print_array(const char *name, const double *v, int n) {
    printf("    \"%s\": [", name);
    for (int i = 0; i < n; i++)
        printf("%s%.17g", i ? "," : "", v[i]);
    printf("]");
}

/* payout[i] = prize for finishing position i+1, geometric-ish decay
 * normalized to sum to `pool` */
static void make_payouts(double *payout, int k, double pool) {
    double sum = 0.0;
    for (int i = 0; i < k; i++) {
        payout[i] = 1.0 / (1.0 + 0.5 * i);
        sum += payout[i];
    }
    for (int i = 0; i < k; i++)
        payout[i] *= pool / sum;
}

typedef enum { ST_UNIFORM, ST_LOGUNIFORM, ST_EQUAL, ST_RATIO_1E9 } StackKind;

static void make_stacks(double *S, int n, StackKind kind) {
    switch (kind) {
    case ST_UNIFORM:
        for (int i = 0; i < n; i++) S[i] = uni(500.0, 200000.0);
        break;
    case ST_LOGUNIFORM:
        for (int i = 0; i < n; i++) {
            double e = uni(0.0, 6.0);
            double m = 1.0;
            for (int j = 0; j < (int)e; j++) m *= 10.0;
            S[i] = uni(1.0, 10.0) * m;
        }
        break;
    case ST_EQUAL:
        for (int i = 0; i < n; i++) S[i] = 10000.0;
        break;
    case ST_RATIO_1E9:
        /* exact boundary case: max/min = 1e9 */
        S[0] = 1.0;
        S[1] = 1e9;
        for (int i = 2; i < n; i++) S[i] = uni(1.0, 1e9);
        break;
    }
}

static int first_case = 1;

/* Q is chosen per case: quadrature error grows with n times stack spread,
 * so the wide log-uniform cases need Q above 256 to converge (the widget's
 * Q-ladder handles this at runtime; here the converged value is baked in).
 * Verification compares wasm and native at the SAME Q, so per-case Q does
 * not weaken the cross-check. */
static void emit_case(const char *name, int n, int k, int Q, StackKind kind,
                      double pool) {
    double *S = malloc((size_t)n * sizeof(double));
    double *payout = malloc((size_t)k * sizeof(double));
    double *equity = malloc((size_t)n * sizeof(double));

    make_stacks(S, n, kind);
    make_payouts(payout, k, pool);
    icm_equity(n, S, Q, payout, k, equity);

    double esum = 0.0, psum = 0.0;
    for (int i = 0; i < n; i++) esum += equity[i];
    for (int i = 0; i < k; i++) psum += payout[i];
    fprintf(stderr, "%-12s n=%-6d k=%-6d sum(eq)-sum(pay) = %.3e\n",
            name, n, k, esum - psum);

    printf("%s  {\n    \"name\": \"%s\",\n    \"n\": %d,\n    \"k\": %d,\n    \"Q\": %d,\n",
           first_case ? "" : ",\n", name, n, k, Q);
    print_array("stacks", S, n);
    printf(",\n");
    print_array("payouts", payout, k);
    printf(",\n");
    print_array("equity", equity, n);
    printf("\n  }");
    first_case = 0;

    free(S); free(payout); free(equity);
}

int main(void) {
    icm_init("devices/m3_pro/fftw_wisdom.dat");
    rng_state = 0x1CEB00DA;

    printf("[\n");
    emit_case("headsup",    2,    1,  256, ST_UNIFORM,    1.0);
    emit_case("ft9",        9,    3,  256, ST_UNIFORM,    1.0);
    emit_case("sng10",     10,   10,  256, ST_UNIFORM,    1.0);
    emit_case("mtt45",     45,    7,  256, ST_LOGUNIFORM, 45000.0);
    emit_case("mtt200",   200,   27,  512, ST_LOGUNIFORM, 200000.0);
    emit_case("mtt1000", 1000,  150, 1024, ST_LOGUNIFORM, 1e6);
    emit_case("mtt3000", 3000,  450, 1024, ST_LOGUNIFORM, 3e6);
    emit_case("mtt10000", 10000, 1500, 2048, ST_LOGUNIFORM, 1e7);
    emit_case("allpaid2000", 2000, 2000, 256, ST_UNIFORM, 1.0);
    emit_case("ratio1e9",   9,    3,  256, ST_RATIO_1E9,  1.0);
    emit_case("ratio1e9_100", 100, 15, 512, ST_RATIO_1E9, 1.0);
    emit_case("equal100", 100,   15,  256, ST_EQUAL,      1.0);
    emit_case("cash6",      6,    2,  256, ST_UNIFORM,    2000.0);
    emit_case("k1_500",   500,    1,  256, ST_LOGUNIFORM, 1.0);
    emit_case("allpaid300", 300, 300, 256, ST_LOGUNIFORM, 1.0);
    emit_case("linear27",  27,    5,  256, ST_UNIFORM,    1.0);
    printf("\n]\n");

    /* n=2 closed form sanity: winner-take-all equity is the stack fraction */
    {
        double S[2] = { 7500.0, 2500.0 };
        double payout[1] = { 1.0 };
        double eq[2];
        icm_equity(2, S, 256, payout, 1, eq);
        fprintf(stderr, "closedform2 |eq0 - 0.75| = %.3e\n", eq[0] - 0.75);
        if (eq[0] - 0.75 > 1e-12 || eq[0] - 0.75 < -1e-12) {
            fprintf(stderr, "FAIL: n=2 closed form\n");
            return 1;
        }
    }
    return 0;
}
