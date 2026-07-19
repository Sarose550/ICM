/*
 * accuracy_bench.c — Quadrature accuracy benchmark for ICM
 *
 * Compares Gauss-Legendre (erfc_trap) vs tanh-sinh quadrature at varying Q.
 * Reports convergence of V1 and V2 equities against exact closed-form values.
 *
 * Compile (macOS / Apple Silicon):
 *   gcc -O3 -march=native -Isrc -Idevices/m3_max -I/opt/homebrew/include \
 *       -o accuracy_bench tools/accuracy_bench.c \
 *       -L/opt/homebrew/lib -lfftw3 -lm -framework Accelerate
 */

/* Include the library source directly for access to internal types */
#include "icm.c"

/* ══════════════════════════════════════════════════════════════
   TANH-SINH QUADRATURE NODE GENERATION
   ══════════════════════════════════════════════════════════════ */

/*
 * Tanh-sinh quadrature for the integral ∫₀¹ f(v) dv.
 *
 * The substitution is: v = (1 + tanh(π/2 · sinh(x))) / 2
 * which maps x ∈ (-∞, ∞) to v ∈ (0, 1).
 *
 * The weight is: dv/dx = π/2 · cosh(x) / (2 · cosh²(π/2 · sinh(x)))
 *
 * We truncate to x ∈ [-x_max, x_max] with step h, yielding 2*N+1 points.
 *
 * For ICM, the engine expects QP nodes with logv and w fields, where:
 *   - logv = log(v) for the quadrature point v ∈ (0,1)
 *   - w    = the quadrature weight (in the log-space integral)
 *
 * The erfc_trap scheme integrates ∫_{-∞}^{∞} f(Φ(y)) · φ(y) dy
 * where Φ is the normal CDF, φ the density. It stores logv = log(Φ(y)) and
 * w = h · φ(y). The engine then computes a_i = exp(S_i · logv) = v^{S_i},
 * runs the engine, and accumulates equity[i] += w · S_i · a_i / v · inner[i].
 *
 * For tanh-sinh, we need the same form. Our substitution gives v directly,
 * so logv = log(v) and the weight w must account for the Jacobian dv.
 * The integral is:
 *   ∫₀¹ g(v) dv = Σ_j w_j · g(v_j)
 * where g(v) = Σ_i S_i · v^{S_i - 1} · P_i(v)
 *
 * But the engine accumulates:
 *   equity[i] += w · S_i · v^{S_i} · (1/v) · inner[i]
 *             = w · S_i · v^{S_i - 1} · inner[i]
 *
 * So w is just the tanh-sinh quadrature weight for ∫₀¹ f(v) dv:
 *   w_j = h · (π/2 · cosh(x_j)) / (2 · cosh²(π/2 · sinh(x_j)))
 */
static void make_nodes_tanh_sinh(int Q, double Smax, QP *pts) {
    /* Determine truncation: we need cosh²(π/2 · sinh(x_max)) to not overflow.
     * π/2 · sinh(x_max) ≈ 350 keeps cosh² in range. sinh(x_max)≈223 → x_max≈6.1 */
    double x_max = 5.5;

    /* Step size: Q points spanning [-x_max, x_max] */
    double h = 2.0 * x_max / (Q - 1);
    int q = 0;
    for (int j = 0; j < Q; j++) {
        double x = -x_max + j * h;
        double sx = sinh(x);
        double cx = cosh(x);
        double pi2_sx = (M_PI / 2.0) * sx;

        /* v = (1 + tanh(π/2 · sinh(x))) / 2 */
        double th = tanh(pi2_sx);
        double v = 0.5 * (1.0 + th);

        /* Skip degenerate points */
        if (v <= 0.0 || v >= 1.0) {
            pts[j].logv = -700.0;  /* effectively zero */
            pts[j].w = 0.0;
            continue;
        }

        pts[j].logv = log(v);

        /* w = h · (π/2 · cosh(x)) / (2 · cosh²(π/2 · sinh(x))) */
        double cosh_pi2_sx = cosh(pi2_sx);
        if (cosh_pi2_sx > 1e150) {
            pts[j].w = 0.0;
        } else {
            pts[j].w = h * (M_PI / 2.0) * cx / (2.0 * cosh_pi2_sx * cosh_pi2_sx);
        }
    }
}

/* ══════════════════════════════════════════════════════════════
   STACK DISTRIBUTIONS
   ══════════════════════════════════════════════════════════════ */

static void make_stacks_uniform(int n, double *S) {
    for (int i = 0; i < n; i++)
        S[i] = 100.0;
}

static void make_stacks_adversarial(int n, double *S) {
    S[0] = 10000.0;
    for (int i = 1; i < n; i++)
        S[i] = 1.0;
}

/* Extreme adversarial: ratio bounded by 1e9 — the practical worst case
 * that motivated the choice of Gaussian quadrature over tanh-sinh. */
static void make_stacks_adv_1e9(int n, double *S) {
    S[0] = 1e9;
    for (int i = 1; i < n; i++)
        S[i] = 1.0;
}

static void make_stacks_geometric(int n, double *S) {
    for (int i = 0; i < n; i++)
        S[i] = pow(2.0, (double)i);
}

/* ══════════════════════════════════════════════════════════════
   COMPUTE EQUITIES WITH CUSTOM NODES
   ══════════════════════════════════════════════════════════════ */

/*
 * Run the integration loop with arbitrary QP nodes.
 * This replicates the serial path of run_engine_ctx_ex() but uses
 * caller-provided nodes instead of make_nodes().
 */
static void compute_equity_with_nodes(int n, const double *S, int Q,
                                      const double *payout, int k,
                                      double *equity, QP *pts) {
    memset(equity, 0, n * sizeof(double));

    double *a_buf = (double *)malloc(n * sizeof(double));
    double *inner_buf = (double *)malloc(n * sizeof(double));

    /* Use linear engine for these small cases — simple, correct */
    LinearCtx *lc = linear_ctx_create(n, k);

    for (int q = 0; q < Q; q++) {
        if (pts[q].w == 0.0) continue;
        double logv = pts[q].logv;
        double wq = pts[q].w;

        for (int j = 0; j < n; j++) {
            double arg = S[j] * logv;
            a_buf[j] = (arg < -700) ? 0 : exp(arg);
        }

        engine_linear_ctx(n, a_buf, payout, k, inner_buf, lc);

        double inv_v = exp(-logv);
        for (int i = 0; i < n; i++) {
            double pw = wq * S[i] * a_buf[i] * inv_v;
            if (!isfinite(pw)) pw = 0;
            equity[i] += pw * inner_buf[i];
        }
    }

    linear_ctx_destroy(lc);
    free(a_buf);
    free(inner_buf);
}

/* ══════════════════════════════════════════════════════════════
   ACCURACY MEASUREMENT
   ══════════════════════════════════════════════════════════════ */

typedef struct {
    double max_abs;
    double max_rel;
} ErrorMetrics;

static ErrorMetrics compute_errors(int n, const double *computed,
                                   const double *exact) {
    ErrorMetrics em = {0.0, 0.0};
    for (int i = 0; i < n; i++) {
        double ae = fabs(computed[i] - exact[i]);
        if (ae > em.max_abs) em.max_abs = ae;
        if (fabs(exact[i]) > 1e-15) {
            double re = ae / fabs(exact[i]);
            if (re > em.max_rel) em.max_rel = re;
        }
    }
    return em;
}

/* ══════════════════════════════════════════════════════════════
   MAIN
   ══════════════════════════════════════════════════════════════ */

int main(void) {
    icm_init(NULL);

    int test_ns[] = {4, 6, 8, 10, 12, 14, 16, 18, 20};
    int n_tests = sizeof(test_ns) / sizeof(test_ns[0]);

    int Q_values[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int n_Q = sizeof(Q_values) / sizeof(Q_values[0]);

    const char *dist_names[] = {"uniform", "adversarial", "geometric",
                                "adv_1e9"};
    typedef void (*StackFn)(int, double*);
    StackFn dist_fns[] = {make_stacks_uniform, make_stacks_adversarial,
                          make_stacks_geometric, make_stacks_adv_1e9};
    int n_dists = 4;

    /* CSV header */
    printf("scheme,n,k,Q,max_abs_err,max_rel_err,payout_type,distribution\n");

    for (int di = 0; di < n_dists; di++) {
        for (int ni = 0; ni < n_tests; ni++) {
            int n = test_ns[ni];
            int k = n;  /* k = n for full placement */

            double *S = (double *)malloc(n * sizeof(double));
            dist_fns[di](n, S);

            /* Exact references */
            double *v1_ref = (double *)malloc(n * sizeof(double));
            double *v2_ref = (double *)malloc(n * sizeof(double));
            v1_exact(n, S, v1_ref);
            v2_exact(n, S, v2_ref);

            /* V1 payout: p[m] = n - m (linear) */
            double *pay_v1 = (double *)malloc(n * sizeof(double));
            for (int m = 0; m < n; m++) pay_v1[m] = (double)(n - m);

            /* V2 payout: p[m] = C(n-1-m, 2) (quadratic) */
            double *pay_v2 = (double *)malloc(n * sizeof(double));
            for (int m = 0; m < n; m++) {
                int r = n - 1 - m;
                pay_v2[m] = (r >= 2) ? (double)(r * (r - 1)) / 2.0 : 0.0;
            }

            double *equity = (double *)malloc(n * sizeof(double));

            double Smax = 0;
            for (int i = 0; i < n; i++) if (S[i] > Smax) Smax = S[i];

            for (int qi = 0; qi < n_Q; qi++) {
                int Q = Q_values[qi];
                QP *pts = (QP *)malloc(Q * sizeof(QP));

                /* ---- Gauss-Legendre (erfc_trap) scheme ---- */
                make_nodes(Q, Smax, pts);

                /* V1 */
                compute_equity_with_nodes(n, S, Q, pay_v1, k, equity, pts);
                ErrorMetrics em = compute_errors(n, equity, v1_ref);
                printf("gauss,%d,%d,%d,%.6e,%.6e,V1,%s\n",
                       n, k, Q, em.max_abs, em.max_rel, dist_names[di]);

                /* V2 */
                compute_equity_with_nodes(n, S, Q, pay_v2, k, equity, pts);
                em = compute_errors(n, equity, v2_ref);
                printf("gauss,%d,%d,%d,%.6e,%.6e,V2,%s\n",
                       n, k, Q, em.max_abs, em.max_rel, dist_names[di]);

                /* ---- Tanh-sinh scheme ---- */
                make_nodes_tanh_sinh(Q, Smax, pts);

                /* V1 */
                compute_equity_with_nodes(n, S, Q, pay_v1, k, equity, pts);
                em = compute_errors(n, equity, v1_ref);
                printf("tanh_sinh,%d,%d,%d,%.6e,%.6e,V1,%s\n",
                       n, k, Q, em.max_abs, em.max_rel, dist_names[di]);

                /* V2 */
                compute_equity_with_nodes(n, S, Q, pay_v2, k, equity, pts);
                em = compute_errors(n, equity, v2_ref);
                printf("tanh_sinh,%d,%d,%d,%.6e,%.6e,V2,%s\n",
                       n, k, Q, em.max_abs, em.max_rel, dist_names[di]);

                free(pts);
            }

            free(S);
            free(v1_ref);
            free(v2_ref);
            free(pay_v1);
            free(pay_v2);
            free(equity);
        }
    }

    return 0;
}
