#ifndef ICM_GPU_H
#define ICM_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IcmGpuPlan IcmGpuPlan;

typedef struct {
    int device_id;              /* CUDA device index (default: 0) */
    int use_cufftdx;            /* Enable Tier-2 fused kernels when available */
    int enable_graphs;          /* Enable CUDA Graph execution */
    int enable_q_pipeline;      /* Enable q+1 build overlap with q propagate */
    int memory_strategy;        /* 0=auto (default; pooled via cudaMallocAsync),
                                  * 1=full (opts OUT of the memory pool, raw
                                  * cudaMalloc instead), 2=pool (explicit,
                                  * same pooled behavior as 0), 3=selective
                                  * recompute (also pooled; additionally
                                  * skips caching FFT results at some
                                  * levels, recomputing them instead) */
    int force_uncached_fused_levels; /* -1 auto, else exact M for fused levels */
    int force_uncached_cufft_levels; /* -1 auto, else exact T for cuFFT levels */
    int fast_mode;              /* Fast mode may reduce Q in tools */
    int verbose;                /* Print plan/runtime diagnostics */
} IcmGpuOptions;

typedef struct {
    double total_ns;
    double sort_ns;
    double quadrature_overhead_ns;
    double block_build_ns;
    double tree_build_ns;
    double tree_propagate_cached_ns;
    double tree_propagate_recomputed_ns;
    double leaf_extract_ns;
    size_t peak_vram_bytes;
    int engine;                 /* 0=linear-schoolbook, 1=hybrid-tree */
    int B;
    int uncached_fused_levels;  /* M */
    int uncached_cufft_levels;  /* T */
} IcmGpuRunStats;

typedef struct {
    int n;
    int k;
    int B;
    int engine;                 /* 0=linear-schoolbook, 1=hybrid-tree */
    int n_levels;
    int n_tier1;
    int n_tier2;
    int n_tier3;
    int q_batch;                /* Q-points processed per tree traversal */
    size_t planned_peak_vram_bytes;
} IcmGpuPlanSummary;

/* Library lifecycle */
int icm_gpu_init(int device_id);
void icm_gpu_shutdown(void);
const char *icm_gpu_last_error(void);

/* Release all memory held by the CUDA stream-ordered memory pool back to
 * the driver.  Safe to call at any time; subsequent allocations will
 * re-populate the pool as needed.  Returns 0 on success, non-zero if no
 * pool exists or the trim operation fails. */
int icm_gpu_release_pooled_memory(void);

/* Plan lifecycle */
IcmGpuPlan *icm_gpu_plan_create(int n, const double *S, int k, const IcmGpuOptions *opts);
void icm_gpu_plan_destroy(IcmGpuPlan *plan);
int icm_gpu_plan_summary(const IcmGpuPlan *plan, IcmGpuPlanSummary *summary);

/* Execution API
 *
 * All three functions return 0 on success, -1 on failure (OOM, CUDA error,
 * invalid arguments, etc.).  Check icm_gpu_last_error() for a diagnostic
 * message on failure.
 *
 * Timing is available through the optional IcmGpuRunStats *stats out-
 * parameter: if non-NULL, stats->total_ns carries the wall-clock time in
 * nanoseconds.  (Pass NULL to suppress timing; you still get 0/-1 return
 * status.)
 */
int icm_gpu_equity_with_plan(IcmGpuPlan *plan, int Q,
                             const double *payout, double *equity,
                             IcmGpuRunStats *stats);

int icm_gpu_equity(int n, const double *S, int Q,
                   const double *payout, int k,
                   double *equity, const IcmGpuOptions *opts,
                   IcmGpuRunStats *stats);

/* Compute equities for a subset of players.
 * Only equity[targets[i]] values are set in the output.
 * Currently computes all equities internally and extracts the subset. */
int icm_gpu_equity_subset(int n, const double *S, int Q,
                          const double *payout, int k,
                          double *equity,
                          const int *targets, int n_targets,
                          const IcmGpuOptions *opts,
                          IcmGpuRunStats *stats);

/* Calibration + diagnostics helpers */
int icm_gpu_write_config_header(const char *output_path);
int icm_gpu_measure_hbm_bandwidth_gbps(double *gbps_out);
int icm_gpu_measure_fused_pair_ns(int fft_n, int batch, int quick,
                                  double *build_ns_out, double *corr_ns_out);
int icm_gpu_measure_fused_r2c_pair_ns(int fft_n, int batch, int quick,
                                       double *build_ns_out, double *corr_ns_out);

#ifdef __cplusplus
}
#endif

#endif /* ICM_GPU_H */
