/*
 * cost_model.h — Shared roofline cost model utilities for ICM engine dispatch.
 *
 * Used by both CPU (icm.c) and GPU (icm_gpu.cu) dispatch functions.
 * The blended_bw() function computes effective streaming bandwidth for a
 * working set that may span multiple cache levels, using harmonic-mean
 * interpolation to model partial hits.
 *
 * Each platform defines its own bandwidth constants (L2_BW_GBS, etc.)
 * in its fft_config.h / gpu_fft_config.h. This header provides only the
 * shared logic.
 */

#ifndef COST_MODEL_H
#define COST_MODEL_H

#include <stddef.h>  /* size_t */

/*
 * Effective streaming bandwidth for a working set of `bytes` total.
 *
 * When data fits in a cache level, use that level's bandwidth.
 * When data spills across a boundary, blend using harmonic mean:
 *   eff_bw = 1 / (hit_frac / hit_bw + miss_frac / miss_bw)
 *
 * This follows from: total_time = hit_bytes/hit_bw + miss_bytes/miss_bw.
 *
 * Requires: L2_CACHE_SIZE, L3_CACHE_SIZE (bytes),
 *           L2_BW_GBS, L3_BW_GBS, DRAM_BW_GBS (GB/s) — defined by caller.
 */
static inline double blended_bw(double bytes) {
    if (bytes <= (double)L2_CACHE_SIZE)
        return L2_BW_GBS;
    if (bytes <= (double)L3_CACHE_SIZE) {
        double l2_frac = (double)L2_CACHE_SIZE / bytes;
        return 1.0 / (l2_frac / L2_BW_GBS + (1.0 - l2_frac) / L3_BW_GBS);
    }
    double l3_frac = (double)L3_CACHE_SIZE / bytes;
    return 1.0 / (l3_frac / L3_BW_GBS + (1.0 - l3_frac) / DRAM_BW_GBS);
}

/*
 * Roofline cost for the batched linear engine (per quadrature point).
 *
 * The batched engine (width batch_width) is memory-bandwidth-limited: its arithmetic
 * intensity (~0.15 FLOP/byte) is far below the machine balance point.
 * This function estimates bytes streamed through each cache level and
 * divides by measured bandwidth.
 *
 * C = checkpoint interval = L2_CACHE_SIZE / (k * batch_width * sizeof(double)).
 *
 * Returns estimated nanoseconds per quadrature point.
 */
static inline double linear_roofline_cost(int n, int k, int batch_width) {
    int C = (int)((size_t)L2_CACHE_SIZE / ((size_t)k * batch_width * sizeof(double)));
    if (C < 1) C = 1;

    if (C >= n) {
        /* No checkpointing — g_store fits in L2.
         * Forward + backward: 2 passes of n*k doubles (read+write).
         * Per QP (amortized over BQ): 2 * n * k * 8 bytes. */
        double bytes_per_qp = 2.0 * n * k * 8.0;
        double bytes_per_batch = bytes_per_qp * batch_width;
        double bw = blended_bw(bytes_per_batch);
        return bytes_per_qp / bw;
    }

    /* Checkpointed: local_g fits in L2 (by design of C).
     *
     * Inner work (L2-resident): recompute forward + backward within each
     * segment.  2 passes of n*k doubles, all hitting L2.
     * Per QP: 2 * n * k * 8 / L2_BW. */
    double inner_bytes = 2.0 * n * k * 8.0;
    double inner_time = inner_bytes / L2_BW_GBS;

    /* Outer I/O (L3 and/or DRAM):
     *   Checkpoints: (n/C) × k×BQ×8 bytes, read+write → 2×(n/C)×k×BQ×8
     *   a_batch: n×BQ×8 bytes, read 3 times → 3×n×BQ×8
     * Per QP (÷BQ): 2*(n/C)*k*8 + 3*n*8 */
    double ckpt_bytes = 2.0 * ((double)n / C) * k * 8.0;
    double abatch_bytes = 3.0 * n * 8.0;
    double outer_bytes = ckpt_bytes + abatch_bytes;
    double outer_total = outer_bytes * batch_width;
    double outer_bw = blended_bw(outer_total);
    double outer_time = outer_bytes / outer_bw;

    return inner_time + outer_time;
}

#endif /* COST_MODEL_H */
