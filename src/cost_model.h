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
 * Cost estimate for the batched linear engine (per quadrature point).
 *
 * The batched linear engine is compute-bound for all realistic (n,k):
 * it performs ~4*n*k fused multiply-add operations per quadrature point
 * (forward: ~1*n*k, backward: ~3*n*k).  The arithmetic intensity of the
 * inner loop (~0.5 FMA/byte) exceeds the machine balance point on modern
 * hardware (e.g. ~0.06 FMA/byte at ~350 GB/s, ~4 TFLOPS), so
 * memory bandwidth is not the bottleneck.
 *
 * When checkpointing is required (working set > L2), there is an additional
 * outer I/O cost for checkpoint writes/reads and a_batch reloads.
 *
 * C = checkpoint interval = L2_CACHE_SIZE / (k * batch_width * sizeof(double)).
 *
 * Returns estimated nanoseconds per quadrature point.
 *
 * Requires: FMA_NS, L2_CACHE_SIZE, L2_BW_GBS, L3_BW_GBS, DRAM_BW_GBS
 * (from fft_config.h).
 */
static inline double linear_roofline_cost(int n, int k, int batch_width) {
    /* Core compute: ~4*n*k FMAs per Q-point (empirically verified within 6%). */
    double compute_ns = 4.0 * n * k * FMA_NS;

    int C = (int)((size_t)L2_CACHE_SIZE / ((size_t)k * batch_width * sizeof(double)));
    if (C < 1) C = 1;

    if (C >= n) {
        /* No checkpointing — everything fits in L2, pure compute bound. */
        return compute_ns;
    }

    /* Checkpointed: inner loop is compute-bound (local_g fits in L2 by
     * design of C).  Outer I/O for checkpoint writes/reads and a_batch
     * reloads adds a small bandwidth-bound term.
     *
     * Checkpoints: (n/C) × k×BQ×8 bytes, read+write → 2×(n/C)×k×BQ×8
     * a_batch: n×BQ×8 bytes, read 3 times → 3×n×BQ×8
     * Per QP (÷BQ): 2*(n/C)*k*8 + 3*n*8 */
    double ckpt_bytes = 2.0 * ((double)n / C) * k * 8.0;
    double abatch_bytes = 3.0 * n * 8.0;
    double outer_bytes = ckpt_bytes + abatch_bytes;
    double outer_total = outer_bytes * batch_width;
    double outer_bw = blended_bw(outer_total);
    double outer_time = outer_bytes / outer_bw;

    return compute_ns + outer_time;
}

#endif /* COST_MODEL_H */
