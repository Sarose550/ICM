/* linear_batched_impl.inc — Template for BQ-parameterized batched linear engine.
 * Included twice from icm.c with different BQ_IMPL values (4 and 8).
 * Do NOT compile this file standalone.
 *
 * Required defines before #include:
 *   BQ_IMPL  — the batch width (4 or 8)
 *   SUFFIX   — function name suffix (_bq4 or _bq8)
 */

#ifndef BQ_IMPL
#error "BQ_IMPL must be defined before including linear_batched_impl.inc"
#endif

#define PASTE2(a,b) a##b
#define PASTE(a,b) PASTE2(a,b)

static int PASTE(ckpt_interval_batched, SUFFIX)(int n, int k) {
    int C = L2_CACHE_SIZE / (k * BQ_IMPL * (int)sizeof(double));
    if (C < 4) C = 4;
    if (C > n) C = n;
    return C;
}

static inline void PASTE(apply_factor_bq, SUFFIX)(const double *restrict g_in,
                                    double *restrict g_out,
                                    const double *ab, const double *bb, int k) {
    for (int m = 0; m < k - 1; m++)
        for (int qi = 0; qi < BQ_IMPL; qi++)
            g_out[m * BQ_IMPL + qi] = ab[qi] * g_in[m * BQ_IMPL + qi] + bb[qi] * g_in[(m+1) * BQ_IMPL + qi];
    for (int qi = 0; qi < BQ_IMPL; qi++)
        g_out[(k-1) * BQ_IMPL + qi] = ab[qi] * g_in[(k-1) * BQ_IMPL + qi];
}

static double PASTE(run_linear_batched, SUFFIX)(int n, const double *S, int Q,
                                  const double *payout, int k,
                                  double *equity, void *ctx) {
    LinearCtx *lc_ctx = (LinearCtx *)ctx;
    const uint8_t *active = lc_ctx ? lc_ctx->active : NULL;
    double Smax = 0;
    for (int i = 0; i < n; i++) if (S[i] > Smax) Smax = S[i];
    QP *pts = (QP *)malloc(Q * sizeof(QP));
    make_nodes(Q, Smax, pts);
    memset(equity, 0, n * sizeof(double));

    double *payout_il = (double *)malloc(k * BQ_IMPL * sizeof(double));
    for (int m = 0; m < k; m++)
        for (int qi = 0; qi < BQ_IMPL; qi++)
            payout_il[m * BQ_IMPL + qi] = payout[m];

    int n_batches = Q / BQ_IMPL;
    int leftover_start = n_batches * BQ_IMPL;

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads > n_batches) nthreads = (n_batches > 0) ? n_batches : 1;

    size_t gstride = (size_t)k * BQ_IMPL;
    double **t_gstore = (double **)malloc(nthreads * sizeof(double *));
    double **t_R      = (double **)malloc(nthreads * sizeof(double *));
    double **t_abatch = (double **)malloc(nthreads * sizeof(double *));
    double **t_ibatch = (double **)malloc(nthreads * sizeof(double *));
    double **t_equity = (double **)malloc(nthreads * sizeof(double *));
    for (int t = 0; t < nthreads; t++) {
        t_gstore[t] = (double *)malloc((size_t)n * gstride * sizeof(double));
        t_R[t]      = (double *)malloc(k * BQ_IMPL * sizeof(double));
        t_abatch[t] = (double *)malloc((size_t)BQ_IMPL * n * sizeof(double));
        t_ibatch[t] = (double *)malloc((size_t)BQ_IMPL * n * sizeof(double));
        t_equity[t] = (double *)calloc(n, sizeof(double));
    }

    double t0 = now_ns();
    #pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int b = 0; b < n_batches; b++) {
        int qb = b * BQ_IMPL;
        int all_zero = 1;
        for (int qi = 0; qi < BQ_IMPL; qi++) if (pts[qb+qi].w != 0) { all_zero = 0; break; }
        if (all_zero) continue;

        int tid = omp_get_thread_num();
        double *g_store = t_gstore[tid];
        double *R = t_R[tid];
        double *a_batch = t_abatch[tid];
        double *inner_batch = t_ibatch[tid];
        double *eq_local = t_equity[tid];

        /* Interleaved layout: a_batch[j * BQ_IMPL + qi] for cache-friendly
         * strided reads in the forward/backward pass inner loop. */
        {
            double logv_bq[BQ_IMPL];
            for (int qi = 0; qi < BQ_IMPL; qi++) logv_bq[qi] = pts[qb + qi].logv;
            for (int j = 0; j < n; j++)
                for (int qi = 0; qi < BQ_IMPL; qi++) {
                    double arg = S[j] * logv_bq[qi];
                    a_batch[(size_t)j * BQ_IMPL + qi] = (arg < -700) ? 0 : exp(arg);
                }
        }

        const double *g_prev = payout_il;
        for (int j = 0; j < n; j++) {
            double ab[BQ_IMPL], bb[BQ_IMPL];
            for (int qi = 0; qi < BQ_IMPL; qi++) {
                ab[qi] = a_batch[(size_t)j * BQ_IMPL + qi];
                bb[qi] = 1.0 - ab[qi];
            }
            double *g_cur = g_store + (size_t)j * gstride;
            for (int m = 0; m < k - 1; m++)
                for (int qi = 0; qi < BQ_IMPL; qi++)
                    g_cur[m * BQ_IMPL + qi] = ab[qi] * g_prev[m * BQ_IMPL + qi] + bb[qi] * g_prev[(m+1) * BQ_IMPL + qi];
            for (int qi = 0; qi < BQ_IMPL; qi++)
                g_cur[(k-1) * BQ_IMPL + qi] = ab[qi] * g_prev[(k-1) * BQ_IMPL + qi];
            g_prev = g_cur;
        }

        memset(R, 0, k * BQ_IMPL * sizeof(double));
        for (int qi = 0; qi < BQ_IMPL; qi++) R[qi] = 1.0;
        for (int j = n - 1; j >= 0; j--) {
            const double *gb = (j > 0) ? (g_store + (size_t)(j-1) * gstride) : payout_il;
            double ab[BQ_IMPL], bb[BQ_IMPL], eq[BQ_IMPL];
            for (int qi = 0; qi < BQ_IMPL; qi++) {
                ab[qi] = a_batch[(size_t)j * BQ_IMPL + qi];
                bb[qi] = 1.0 - ab[qi];
                eq[qi] = gb[qi] * R[qi];
            }
            for (int m = k - 1; m >= 1; m--)
                for (int qi = 0; qi < BQ_IMPL; qi++) {
                    eq[qi] += gb[m * BQ_IMPL + qi] * R[m * BQ_IMPL + qi];
                    R[m * BQ_IMPL + qi] = ab[qi] * R[m * BQ_IMPL + qi] + bb[qi] * R[(m-1) * BQ_IMPL + qi];
                }
            for (int qi = 0; qi < BQ_IMPL; qi++) R[qi] = ab[qi] * R[qi];
            for (int qi = 0; qi < BQ_IMPL; qi++) inner_batch[(size_t)j * BQ_IMPL + qi] = eq[qi];
        }

        double wq_b[BQ_IMPL], iv_b[BQ_IMPL];
        for (int qi = 0; qi < BQ_IMPL; qi++) {
            wq_b[qi] = pts[qb + qi].w;
            iv_b[qi] = exp(-pts[qb + qi].logv);
        }
        for (int i = 0; i < n; i++) {
            if (active && !active[i]) continue;
            double sum = 0;
            for (int qi = 0; qi < BQ_IMPL; qi++) {
                double pw = wq_b[qi] * S[i] * a_batch[(size_t)i * BQ_IMPL + qi] * iv_b[qi];
                if (!isfinite(pw)) pw = 0;
                sum += pw * inner_batch[(size_t)i * BQ_IMPL + qi];
            }
            eq_local[i] += sum;
        }
    }

    for (int t = 0; t < nthreads; t++)
        for (int i = 0; i < n; i++)
            equity[i] += t_equity[t][i];

    /* Handle leftover quad points that don't fill a batch */
    for (int q = leftover_start; q < Q; q++) {
        if (pts[q].w == 0) continue;
        double logv = pts[q].logv, wq = pts[q].w;
        double *a_tmp = t_abatch[0];
        double *inner_tmp = t_ibatch[0];
        for (int j = 0; j < n; j++) {
            double arg = S[j] * logv;
            a_tmp[j] = (arg < -700) ? 0 : exp(arg);
        }
        LinearCtx *lc_tmp = linear_ctx_create(n, k);
        lc_tmp->active = (uint8_t *)active;
        engine_linear_ctx(n, a_tmp, payout, k, inner_tmp, lc_tmp);
        lc_tmp->active = NULL;
        linear_ctx_destroy(lc_tmp);
        double iv = exp(-logv);
        for (int i = 0; i < n; i++) {
            if (active && !active[i]) continue;
            double pw = wq * S[i] * a_tmp[i] * iv;
            if (!isfinite(pw)) pw = 0;
            equity[i] += pw * inner_tmp[i];
        }
    }

    double elapsed = now_ns() - t0;
    for (int t = 0; t < nthreads; t++) {
        free(t_gstore[t]); free(t_R[t]);
        free(t_abatch[t]); free(t_ibatch[t]);
        free(t_equity[t]);
    }
    free(t_gstore); free(t_R); free(t_abatch); free(t_ibatch); free(t_equity);

#else  /* No OpenMP — serial fallback */
    size_t gstride = (size_t)k * BQ_IMPL;
    int use_ckpt = ((size_t)n * k * BQ_IMPL * sizeof(double) > L2_CACHE_SIZE);
    int C = use_ckpt ? PASTE(ckpt_interval_batched, SUFFIX)(n, k) : 0;

    double *g_store = NULL, *ckpt_buf = NULL, *local_g = NULL, *fbuf0 = NULL, *fbuf1 = NULL;
    if (use_ckpt) {
        int n_ckpt = n / C;
        ckpt_buf = (double *)malloc((size_t)n_ckpt * gstride * sizeof(double));
        local_g  = (double *)malloc((size_t)C * gstride * sizeof(double));
        fbuf0    = (double *)malloc(gstride * sizeof(double));
        fbuf1    = (double *)malloc(gstride * sizeof(double));
    } else {
        g_store = (double *)malloc((size_t)n * gstride * sizeof(double));
    }
    double *R = (double *)malloc(k * BQ_IMPL * sizeof(double));
    double *a_batch = (double *)malloc((size_t)BQ_IMPL * n * sizeof(double));
    double *inner_batch = (double *)malloc((size_t)BQ_IMPL * n * sizeof(double));

    double t0 = now_ns();
    for (int b = 0; b < n_batches; b++) {
        int qb = b * BQ_IMPL;
        int all_zero = 1;
        for (int qi = 0; qi < BQ_IMPL; qi++) if (pts[qb+qi].w != 0) { all_zero = 0; break; }
        if (all_zero) continue;

        /* Interleaved layout: a_batch[j * BQ_IMPL + qi] for cache-friendly
         * strided reads in the forward/backward pass inner loop. */
        {
            double logv_bq[BQ_IMPL];
            for (int qi = 0; qi < BQ_IMPL; qi++) logv_bq[qi] = pts[qb + qi].logv;
            for (int j = 0; j < n; j++)
                for (int qi = 0; qi < BQ_IMPL; qi++) {
                    double arg = S[j] * logv_bq[qi];
                    a_batch[(size_t)j * BQ_IMPL + qi] = (arg < -700) ? 0 : exp(arg);
                }
        }

        if (use_ckpt) {
            /* Checkpointed forward: ping-pong buffers, save every C players */
            const double *g_prev = payout_il;
            for (int j = 0; j < n; j++) {
                double ab[BQ_IMPL], bb[BQ_IMPL];
                for (int qi = 0; qi < BQ_IMPL; qi++) {
                    ab[qi] = a_batch[(size_t)j * BQ_IMPL + qi];
                    bb[qi] = 1.0 - ab[qi];
                }
                double *g_cur = (j & 1) ? fbuf1 : fbuf0;
                PASTE(apply_factor_bq, SUFFIX)(g_prev, g_cur, ab, bb, k);
                if ((j + 1) % C == 0)
                    memcpy(ckpt_buf + (size_t)((j+1)/C - 1) * gstride, g_cur, gstride * sizeof(double));
                g_prev = g_cur;
            }

            /* Checkpointed backward: recompute within each segment */
            memset(R, 0, k * BQ_IMPL * sizeof(double));
            for (int qi = 0; qi < BQ_IMPL; qi++) R[qi] = 1.0;
            int n_seg = (n + C - 1) / C;
            for (int s = n_seg - 1; s >= 0; s--) {
                int ss = s * C, se = ss + C;
                if (se > n) se = n;
                const double *prefix = (s == 0) ? payout_il : (ckpt_buf + (size_t)(s-1) * gstride);

                /* Recompute local forward chain within segment */
                const double *lp = prefix;
                for (int j = 0; j < se - ss; j++) {
                    double ab[BQ_IMPL], bb[BQ_IMPL];
                    for (int qi = 0; qi < BQ_IMPL; qi++) {
                        ab[qi] = a_batch[(size_t)(ss+j) * BQ_IMPL + qi];
                        bb[qi] = 1.0 - ab[qi];
                    }
                    double *g_cur = local_g + (size_t)j * gstride;
                    PASTE(apply_factor_bq, SUFFIX)(lp, g_cur, ab, bb, k);
                    lp = g_cur;
                }

                /* Backward fused within segment */
                for (int j = se - ss - 1; j >= 0; j--) {
                    double ab[BQ_IMPL], bb[BQ_IMPL];
                    for (int qi = 0; qi < BQ_IMPL; qi++) {
                        ab[qi] = a_batch[(size_t)(ss+j) * BQ_IMPL + qi];
                        bb[qi] = 1.0 - ab[qi];
                    }
                    if (!active || active[ss+j]) {
                        const double *gb = (j == 0) ? prefix : (local_g + (size_t)(j-1) * gstride);
                        double eq[BQ_IMPL];
                        for (int qi = 0; qi < BQ_IMPL; qi++) eq[qi] = gb[qi] * R[qi];
                        for (int m = k - 1; m >= 1; m--)
                            for (int qi = 0; qi < BQ_IMPL; qi++) {
                                eq[qi] += gb[m * BQ_IMPL + qi] * R[m * BQ_IMPL + qi];
                                R[m * BQ_IMPL + qi] = ab[qi] * R[m * BQ_IMPL + qi] + bb[qi] * R[(m-1) * BQ_IMPL + qi];
                            }
                        for (int qi = 0; qi < BQ_IMPL; qi++) R[qi] = ab[qi] * R[qi];
                        for (int qi = 0; qi < BQ_IMPL; qi++) inner_batch[(size_t)(ss+j) * BQ_IMPL + qi] = eq[qi];
                    } else {
                        for (int m = k - 1; m >= 1; m--)
                            for (int qi = 0; qi < BQ_IMPL; qi++)
                                R[m * BQ_IMPL + qi] = ab[qi] * R[m * BQ_IMPL + qi] + bb[qi] * R[(m-1) * BQ_IMPL + qi];
                        for (int qi = 0; qi < BQ_IMPL; qi++) R[qi] = ab[qi] * R[qi];
                    }
                }
            }
        } else {
            /* Flat forward: store all n rows */
            const double *g_prev = payout_il;
            for (int j = 0; j < n; j++) {
                double ab[BQ_IMPL], bb[BQ_IMPL];
                for (int qi = 0; qi < BQ_IMPL; qi++) {
                    ab[qi] = a_batch[(size_t)j * BQ_IMPL + qi];
                    bb[qi] = 1.0 - ab[qi];
                }
                double *g_cur = g_store + (size_t)j * gstride;
                PASTE(apply_factor_bq, SUFFIX)(g_prev, g_cur, ab, bb, k);
                g_prev = g_cur;
            }

            /* Flat backward: fused dot + suffix update */
            memset(R, 0, k * BQ_IMPL * sizeof(double));
            for (int qi = 0; qi < BQ_IMPL; qi++) R[qi] = 1.0;
            for (int j = n - 1; j >= 0; j--) {
                const double *gb = (j > 0) ? (g_store + (size_t)(j-1) * gstride) : payout_il;
                double ab[BQ_IMPL], bb[BQ_IMPL], eq[BQ_IMPL];
                for (int qi = 0; qi < BQ_IMPL; qi++) {
                    ab[qi] = a_batch[(size_t)j * BQ_IMPL + qi];
                    bb[qi] = 1.0 - ab[qi];
                    eq[qi] = gb[qi] * R[qi];
                }
                for (int m = k - 1; m >= 1; m--)
                    for (int qi = 0; qi < BQ_IMPL; qi++) {
                        eq[qi] += gb[m * BQ_IMPL + qi] * R[m * BQ_IMPL + qi];
                        R[m * BQ_IMPL + qi] = ab[qi] * R[m * BQ_IMPL + qi] + bb[qi] * R[(m-1) * BQ_IMPL + qi];
                    }
                for (int qi = 0; qi < BQ_IMPL; qi++) R[qi] = ab[qi] * R[qi];
                for (int qi = 0; qi < BQ_IMPL; qi++) inner_batch[(size_t)j * BQ_IMPL + qi] = eq[qi];
            }
        }

        double wq_b[BQ_IMPL], iv_b[BQ_IMPL];
        for (int qi = 0; qi < BQ_IMPL; qi++) {
            wq_b[qi] = pts[qb + qi].w;
            iv_b[qi] = exp(-pts[qb + qi].logv);
        }
        for (int i = 0; i < n; i++) {
            if (active && !active[i]) continue;
            double sum = 0;
            for (int qi = 0; qi < BQ_IMPL; qi++) {
                double pw = wq_b[qi] * S[i] * a_batch[(size_t)i * BQ_IMPL + qi] * iv_b[qi];
                if (!isfinite(pw)) pw = 0;
                sum += pw * inner_batch[(size_t)i * BQ_IMPL + qi];
            }
            equity[i] += sum;
        }
    }

    /* Handle leftover quad points that don't fill a batch */
    for (int q = leftover_start; q < Q; q++) {
        if (pts[q].w == 0) continue;
        double logv = pts[q].logv, wq = pts[q].w;
        for (int j = 0; j < n; j++) {
            double arg = S[j] * logv;
            a_batch[j] = (arg < -700) ? 0 : exp(arg);
        }
        LinearCtx *lc_tmp = linear_ctx_create(n, k);
        lc_tmp->active = (uint8_t *)active;
        engine_linear_ctx(n, a_batch, payout, k, inner_batch, lc_tmp);
        lc_tmp->active = NULL;
        linear_ctx_destroy(lc_tmp);
        double iv = exp(-logv);
        for (int i = 0; i < n; i++) {
            if (active && !active[i]) continue;
            double pw = wq * S[i] * a_batch[i] * iv;
            if (!isfinite(pw)) pw = 0;
            equity[i] += pw * inner_batch[i];
        }
    }

    double elapsed = now_ns() - t0;
    free(R); free(a_batch); free(inner_batch);
    if (use_ckpt) { free(ckpt_buf); free(local_g); free(fbuf0); free(fbuf1); }
    else { free(g_store); }
#endif /* _OPENMP */

    free(pts); free(payout_il);
    return elapsed;
}

#undef PASTE
#undef PASTE2
