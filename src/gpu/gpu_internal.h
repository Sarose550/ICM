/* gpu_internal.h -- shared declarations for GPU compilation units.
 *
 * All .cu files in src/gpu/ include this header.  It replaces the anonymous
 * namespace that wrapped the original monolithic icm_gpu.cu with a named
 * namespace (icm_gpu_detail) so that symbols are visible across TUs while
 * remaining hidden from public linkage.
 */
#ifndef ICM_GPU_INTERNAL_H
#define ICM_GPU_INTERNAL_H

/* ── External headers ─────────────────────────────────────────── */
#include "../icm_gpu.h"
#include "../icm.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <time.h>
#include <utility>
#include <vector>

#include "gpu_fft_config.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── cuFFTDx detection ─────────────────────────────────────────── */
#if defined(USE_CUFFTDX) && defined(__has_include)
#if __has_include(<cufftdx.hpp>)
#include <cufftdx.hpp>
#define ICM_HAVE_CUFFTDX 1
/* Check for R2C/C2R support (available in cuFFTDx 1.2+) */
#if __has_include(<cufftdx/traits/detail/fft_traits.hpp>)
#define ICM_HAVE_CUFFTDX_R2C 1
#else
#define ICM_HAVE_CUFFTDX_R2C 1  /* Assume available in modern cuFFTDx */
#endif
#else
#define ICM_HAVE_CUFFTDX 0
#define ICM_HAVE_CUFFTDX_R2C 0
#endif

#if defined(USE_CUFFTDX) && defined(ICM_REQUIRE_CUFFTDX) && !ICM_HAVE_CUFFTDX
#error "USE_CUFFTDX requested, but <cufftdx.hpp> was not found. Set CUFFTDX_INC to MathDx include path."
#endif
#else
#define ICM_HAVE_CUFFTDX 0
#define ICM_HAVE_CUFFTDX_R2C 0
#endif

/* ── VkFFT detection (dual-dispatch: cuFFT vs VkFFT per size) ──── */
#if defined(HAS_GPU_CALIB_LIB) && defined(USE_VKFFT)
#ifndef VKFFT_BACKEND
#define VKFFT_BACKEND 1  /* CUDA backend */
#endif
#include "vkFFT.h"
#include <cuda.h>        /* CUdevice for VkFFT config */
#define ICM_HAVE_VKFFT 1
#else
#define ICM_HAVE_VKFFT 0
#endif

/* ── Named namespace for internal symbols ──────────────────────── */
namespace icm_gpu_detail {

/* ── Constants ──────────────────────────────────────────────────── */
constexpr int MAX_B_CANDIDATES = 48;
extern const int kBCandidates[MAX_B_CANDIDATES];

constexpr int Q_BATCH_MAX = 256;

constexpr int GPU_THREADS_PER_BLOCK = 256;

constexpr int GPU_SCHOOL_WARP_MAX_CONV = 128;
constexpr int GPU_SCHOOL_WARPS_PER_BLOCK = 4;
constexpr size_t GPU_SCHOOL_SMEM_SAFE_BYTES = 48u * 1024u;

/* ── Enums ──────────────────────────────────────────────────────── */
enum {
    GPU_ENGINE_LINEAR = 0,
    GPU_ENGINE_HYBRID = 1
};

enum {
    GPU_TIER_SCHOOLBOOK = 1,
    GPU_TIER_FUSED = 2,
    GPU_TIER_CUFFT = 3
};

/* ── Structs ────────────────────────────────────────────────────── */
struct GpuLevelPlan {
    int ell = 0;
    int tier = GPU_TIER_CUFFT;
    int use_fft = 1;
    int cache_fft = 0;
    int fft_n = 0;
    int cn = 0;
    int p_eff = 0;
    int out_needed = 0;
    int g_eff = 0;
    int build_conv = 0;
    int corr_conv = 0;
    int build_wrap_m = 0;
    int corr_wrap_m = 0;
};

struct GpuFftBuffers {
    double *real_in = nullptr;
    cufftDoubleComplex *spec_in = nullptr;
    cufftDoubleComplex *spec_mid = nullptr;
    double *real_out = nullptr;
    cufftHandle plan_fwd = 0;
    cufftHandle plan_inv = 0;
    int batch_fwd = 0;
    int batch_inv = 0;
    int fft_n = 0;
    int cn = 0;
#if ICM_HAVE_VKFFT
    VkFFTApplication vkfft_app_fwd = {};
    VkFFTApplication vkfft_app_inv = {};
    int use_vkfft = 0;             /* 1 if this buffer set uses VkFFT */
    int vkfft_fwd_initialized = 0; /* track init state for cleanup */
    int vkfft_inv_initialized = 0;
#endif
};

struct SharedFftWork {
    double *real_in = nullptr;
    cufftDoubleComplex *spec_in = nullptr;
    cufftDoubleComplex *spec_mid = nullptr;
    double *real_out = nullptr;
    size_t real_in_bytes = 0;
    size_t spec_in_bytes = 0;
    size_t spec_mid_bytes = 0;
    size_t real_out_bytes = 0;
};

struct GpuPlan {
    int n = 0;
    int k = 0;
    int k_pad = 0;
    int B = 0;
    int engine = GPU_ENGINE_HYBRID;
    int nblocks = 0;
    int N_tree = 0;
    int L = 0;
    int q_batch = 1;

    IcmGpuOptions opts{};

    std::vector<int> sort_perm;
    std::vector<int> inv_perm;
    std::vector<double> S_sorted;
    std::vector<double> payout_host;

    std::vector<int> nn;
    std::vector<int> psz;
    std::vector<int> fft_stride;
    std::vector<size_t> plev_off;
    std::vector<int> g_needed;
    std::vector<int> below_sat;
    std::vector<int> n_real;
    std::vector<GpuLevelPlan> levels;

    std::vector<int> uncached_level;
    int uncached_fused_levels = 0;
    int uncached_cufft_levels = 0;

    size_t planned_peak_vram_bytes = 0;

    double *d_S_sorted = nullptr;
    int *d_sort_perm = nullptr;
    int *d_inv_perm = nullptr;
    double *d_a_sorted[2] = {nullptr, nullptr};
    double *d_graph_logv[2] = {nullptr, nullptr};
    double *d_graph_scale[2] = {nullptr, nullptr};
    double *d_inner_sorted = nullptr;
    double *d_equity = nullptr;
    double *d_payout = nullptr;
    double *d_block_prods = nullptr;

    double *d_a_qbatch[Q_BATCH_MAX] = {};
    double *d_inner_qbatch = nullptr;
    double *d_block_prods_qbatch = nullptr;
    double **d_qb_a_ptrs = nullptr;
    double *d_qb_weights = nullptr;
    double *d_qb_inv_vs = nullptr;

    uint8_t *d_active_mask = nullptr;

    std::vector<double *> d_poly_levels;
    std::vector<double *> d_g_levels;
    std::vector<cufftDoubleComplex *> d_fft_cache;
    std::vector<GpuFftBuffers> build_fft;
    std::vector<GpuFftBuffers> corr_fft;

    cudaStream_t stream_compute = nullptr;
    cudaStream_t stream_aux = nullptr;
    cudaEvent_t evt_a_ready[2] = {nullptr, nullptr};

    double *d_poly_leaves_alt = nullptr;
    double *d_block_prods_alt = nullptr;
    cudaEvent_t evt_prop_done = nullptr;

    cudaGraph_t graph[2] = {nullptr, nullptr};
    cudaGraphExec_t graph_exec[2] = {nullptr, nullptr};
    bool graph_ready[2] = {false, false};

    bool use_async_pool = false;
    cudaMemPool_t mem_pool = nullptr;

    std::vector<bool> fft_cache_valid;

    SharedFftWork shared_build_work;
    SharedFftWork shared_corr_work;
    void *shared_cufft_workspace = nullptr;
    size_t shared_cufft_workspace_bytes = 0;

    double *d_fft_scratch = nullptr;

    void *arena_base = nullptr;
    size_t arena_total_bytes = 0;

    size_t peak_vram_bytes = 0;
    size_t current_vram_bytes = 0;
};

struct CandidateCost {
    int B = 0;
    double total_ns = std::numeric_limits<double>::infinity();
};

struct QP {
    double logv;
    double w;
};

/* ── Global state (defined in gpu_api.cu) ──────────────────────── */
extern std::string g_last_error;
extern int g_cuda_device;
extern int g_runtime_fused_max_conv_len;

/* ── Error handling ─────────────────────────────────────────────── */
void set_last_errorf(const char *fmt, ...);

bool cuda_ok(cudaError_t err, const char *expr, const char *file, int line);
bool cufft_ok(cufftResult err, const char *expr, const char *file, int line);

#define CUDA_OK(expr) icm_gpu_detail::cuda_ok((expr), #expr, __FILE__, __LINE__)
#define CUFFT_OK(expr) icm_gpu_detail::cufft_ok((expr), #expr, __FILE__, __LINE__)

/* ── Utility functions ──────────────────────────────────────────── */
inline double now_ns_host() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int next_pow2_int(int n);
void update_vram_alloc(GpuPlan *plan, size_t bytes);
bool alloc_device(GpuPlan *plan, void **ptr, size_t bytes, cudaStream_t stream);

/* ── Plan helpers (gpu_plan.cu) ─────────────────────────────────── */
void build_smooth_table(int max_n, std::vector<int> &smooth);
int first_calib_ge(int n);
int find_calib_index(int fft_n);
double estimate_cufft_pipeline_ns(int fft_n);
double estimate_cufft_pipeline_ns_batched(int fft_n, double effective_batch);
int fastest_fft_ge_gpu(int n);
double wrap_serial_penalty_gpu(int nparents);
void best_fft_config_gpu(int conv_len, int len_P, double correction_scale,
                         int *out_fft_n, int *out_wrap_m);
double best_fft_config_joint_gpu(int build_conv, int corr_conv, int p_eff,
                                 double correction_scale,
                                 int *out_fft_n, int *out_build_wrap_m, int *out_corr_wrap_m);
double estimate_fused_build_ns(int fft_n);
double estimate_fused_corr_ns(int fft_n);
int fused_max_conv_len_runtime();
int best_k_pad_gpu(int k, const std::vector<int> &smooth);
int pick_tier_for_fft_len(int fft_n, int conv_len);
double tree_school_ns_per_fma();
double block_build_ns_per_fma_model();
double leaf_extract_ns_per_fma_model();
void build_tree_geometry(int n_leaves, int leaf_degree, int k_pad,
                         int leaf_extract, std::vector<int> &nn,
                         std::vector<int> &psz, std::vector<size_t> &plev_off,
                         std::vector<int> &g_needed, std::vector<int> &below_sat,
                         std::vector<int> &n_real, int &N, int &L);
double estimate_candidate_cost(int n, int k_pad, int B, const std::vector<int> &smooth);
int gpu_select_best_B_est(int n, int k_pad, const std::vector<int> &smooth);
int gpu_select_engine_est(int n, int k_pad, int B, const std::vector<int> &smooth);
bool build_plan_metadata(GpuPlan *plan);
bool device_sort_players(GpuPlan *plan);
bool allocate_plan_device_memory(GpuPlan *plan);
bool choose_uncached_levels(GpuPlan *plan);
bool create_cufft_plan(cufftHandle *plan, int n, int batch, bool r2c, int real_dist = 0);

#if ICM_HAVE_VKFFT
/* VkFFT plan creation and dispatch helpers (gpu_plan.cu / gpu_exec.cu) */
bool create_vkfft_r2c_plan(VkFFTApplication *app, int n, int batch, int stride, cudaStream_t *stream_ptr);
void destroy_vkfft_app(VkFFTApplication *app);
bool should_use_vkfft(int fft_n);

/* Gather/scatter kernels for strided <-> contiguous conversion */
__global__ void k_gather_strided(const double *src, int src_stride, double *dst,
                                 int fft_n, int batch);
__global__ void k_scatter_strided(const double *src, int fft_n, double *dst,
                                  int dst_stride, int valid_len, int batch);
#endif

/* ── Kernel dispatch (gpu_kernels.cu) ───────────────────────────── */
bool is_cufftdx_supported_fft_n(int fft_n);
bool launch_cufftdx_build_dispatch(int fft_n,
                                   const double *child, int cps,
                                   double *parent, int pps, int nparents,
                                   double inv_fft_n, cudaStream_t stream,
                                   int child_stride, int parent_stride);
bool launch_cufftdx_corr_dispatch(int fft_n,
                                  const double *g_parent, int parent_gsz, int len_g,
                                  const double *child_poly, int cps, int len_P,
                                  double *g_child, int child_gsz, int len_out, int nparents,
                                  double inv_fft_n, cudaStream_t stream,
                                  int g_parent_stride, int poly_child_stride,
                                  int g_child_stride);
bool launch_cufftdx_build_r2c_dispatch(int fft_n,
                                       const double *child, int cps,
                                       double *parent, int pps, int nparents,
                                       double inv_fft_n, cudaStream_t stream,
                                       int child_stride, int parent_stride);
bool launch_cufftdx_corr_r2c_dispatch(int fft_n,
                                      const double *g_parent, int parent_gsz, int len_g,
                                      const double *child_poly, int cps, int len_P,
                                      double *g_child, int child_gsz, int len_out, int nparents,
                                      double inv_fft_n, cudaStream_t stream,
                                      int g_parent_stride, int poly_child_stride,
                                      int g_child_stride);

/* Kernel declarations needed by gpu_exec.cu and gpu_api.cu */
__global__ void k_compute_a(const double *S_sorted, double *a_sorted, int n, double logv);
__global__ void k_compute_a_from_ptr(const double *S_sorted, double *a_sorted,
                                     int n, const double *logv_ptr);
__global__ void k_compute_a_qbatch(const double *S_sorted,
                                   double * const *a_ptrs,
                                   const double *logv_array,
                                   int n, int qb);
__global__ void k_set_leaves_b1(const double *a_sorted, int n, int N_tree,
                                int leaf_psz, double *leaves);
__global__ void k_set_leaves_b1_qbatch(double * const *a_ptrs,
                                       int n, int N_tree,
                                       int leaf_psz, int qb,
                                       double *leaves, size_t leaf_stride);
__global__ void k_zero(double *x, size_t n);
__global__ void k_set_root_g(double *g_root, int root_gsz, const double *payout, int k);
__global__ void k_block_build(const double *a_sorted, int n, int B,
                              int nblocks, int N_tree,
                              int leaf_psz, double *leaves, double *block_prods);
__global__ void k_schoolbook_build(const double *child, int cps,
                                   double *parent, int pps, int nparents,
                                   int child_stride, int parent_stride);
__global__ void k_schoolbook_build_smem_parent(const double *child, int cps,
                                               double *parent, int pps, int nparents,
                                               int child_stride, int parent_stride);
__global__ void k_schoolbook_build_warp_batch(const double *child, int cps,
                                              double *parent, int pps, int nparents,
                                              int child_stride, int parent_stride);
__global__ void k_pairwise_mul(const cufftDoubleComplex *child_spec, int cn,
                               cufftDoubleComplex *parent_spec, int nparents,
                               double scale);
__global__ void k_gather_to_fft(const double *src, int src_stride,
                                double *dst, int fft_n, int batch);
__global__ void k_scatter_from_fft(const double *src, int fft_n,
                                   double *dst, int dst_stride,
                                   int valid_len, int batch);
__global__ void k_wrap_build(double *parent, int pps, int nparents,
                             const double *child, int cps, int conv_len,
                             int fft_n, int wrap_m,
                             int parent_stride, int child_stride);
__global__ void k_paired_corr_freq(const cufftDoubleComplex *g_hat,
                                   const cufftDoubleComplex *cached_child_spec,
                                   int cn, int nparents,
                                   cufftDoubleComplex *child_out_spec,
                                   double scale);
__global__ void k_wrap_corr_pair(double *g_child, int child_gsz, int nparents,
                                 const double *g_parent, int parent_gsz, int len_g,
                                 const double *child_poly, int cps, int len_P,
                                 int len_out,
                                 int fft_n, int wrap_m,
                                 int child_g_stride, int parent_g_stride,
                                 int child_poly_stride);
__global__ void k_schoolbook_corr_pair(const double *g_parent, int parent_gsz,
                                       int len_g,
                                       const double *child_poly, int cps, int len_P,
                                       double *g_child, int child_gsz, int len_out, int nparents,
                                       int parent_g_stride, int child_poly_stride, int child_g_stride);
__global__ void k_schoolbook_corr_pair_smem_parent(const double *g_parent, int parent_gsz,
                                                   int len_g,
                                                   const double *child_poly, int cps, int len_P,
                                                   double *g_child, int child_gsz,
                                                   int len_out, int nparents,
                                                   int parent_g_stride, int child_poly_stride,
                                                   int child_g_stride);
__global__ void k_schoolbook_corr_pair_warp_batch(const double *g_parent, int parent_gsz,
                                                  int len_g,
                                                  const double *child_poly, int cps, int len_P,
                                                  double *g_child, int child_gsz,
                                                  int len_out, int nparents,
                                                  int parent_g_stride, int child_poly_stride,
                                                  int child_g_stride);
__global__ void k_leaf_extract(const double *a_sorted, int n, int B, int nblocks,
                               const double *block_prods, const double *g_leaf,
                               int leaf_psz, int g_need, int k,
                               double *inner_sorted);
__global__ void k_leaf_extract_b1(int n, const double *g_leaf,
                                  int leaf_psz, double *inner_sorted);
__global__ void k_leaf_extract_b1_qbatch(int n, const double *g_leaf,
                                         int leaf_psz, int qb,
                                         size_t g_stride, double *inner,
                                         size_t inner_stride);
__global__ void k_accumulate_equity(const double *inner_sorted,
                                    const double *a_sorted,
                                    const double *S_sorted,
                                    const int *sort_perm,
                                    int n, double weight, double inv_v,
                                    double *equity);
__global__ void k_accumulate_equity_scaled(const double *inner_sorted,
                                           const double *a_sorted,
                                           const double *S_sorted,
                                           const int *sort_perm,
                                           int n, const double *scale_ptr,
                                           double *equity);
__global__ void k_icm_single_kernel(
        const double *S_sorted, const int *sort_perm, int n,
        int Q, const double *d_logv, const double *d_weights,
        const double *payout, int k,
        double *equity,
        int N, int L, const int *d_nn, const int *d_psz, const int *d_g_needed,
        const size_t *d_plev_off, int total_poly, int max_g);
__global__ void k_block_build_qbatch(
        const double * const *a_ptrs, int n, int B,
        int nblocks, int N_tree, int q_batch,
        int leaf_psz, double *leaves, size_t leaf_stride,
        double *block_prods, size_t bp_stride);
__global__ void k_set_root_g_qbatch(double *g_root, int root_gsz,
                                    const double *payout, int k,
                                    int q_batch, size_t g_stride);
__global__ void k_leaf_extract_qbatch(
        const double * const *a_ptrs, int n, int B, int nblocks,
        const double *block_prods, size_t bp_stride,
        const double *g_leaf, int leaf_psz, size_t leaf_g_stride,
        int g_need, int k, int q_batch,
        double *inner_sorted, size_t inner_stride);
__global__ void k_accumulate_equity_qbatch(
        const double *inner_sorted, size_t inner_stride,
        const double * const *a_ptrs,
        const double *S_sorted,
        const int *sort_perm,
        int n, const double *weights, const double *inv_vs,
        int q_batch, double *equity);
__global__ void k_leaf_extract_qbatch_masked(
        const double * const *a_ptrs, int n, int B, int nblocks,
        const double *block_prods, size_t bp_stride,
        const double *g_leaf, int leaf_psz, size_t leaf_g_stride,
        int g_need, int k, int q_batch,
        double *inner_sorted, size_t inner_stride,
        const uint8_t *active_mask);
__global__ void k_accumulate_equity_qbatch_masked(
        const double *inner_sorted, size_t inner_stride,
        const double * const *a_ptrs,
        const double *S_sorted,
        const int *sort_perm,
        int n, const double *weights, const double *inv_vs,
        int q_batch, double *equity,
        const uint8_t *active_mask);

/* ── Execution (gpu_exec.cu) ─────────────────────────────────────── */
bool run_build_level_schoolbook(GpuPlan *plan, int ell);
bool run_build_level_fft(GpuPlan *plan, int ell);
bool run_build_level_fused(GpuPlan *plan, int ell);
bool run_prop_level_schoolbook(GpuPlan *plan, int ell);
bool run_prop_level_fft(GpuPlan *plan, int ell);
bool run_prop_level_fused(GpuPlan *plan, int ell);

bool run_build_level_schoolbook_qb(GpuPlan *plan, int ell, int qb);
bool run_build_level_fft_qb(GpuPlan *plan, int ell, int qb);
bool run_build_level_fused_qb(GpuPlan *plan, int ell, int qb);
bool run_prop_level_schoolbook_qb(GpuPlan *plan, int ell, int qb);
bool run_prop_level_fft_qb(GpuPlan *plan, int ell, int qb);
bool run_prop_level_fused_qb(GpuPlan *plan, int ell, int qb);

bool run_hybrid_batched_q(GpuPlan *plan, const QP *pts, int qb);
bool run_hybrid_single_q(GpuPlan *plan, int a_buf_idx,
                         double logv, double w,
                         bool skip_compute_a, bool fast_mode,
                         double *block_ns, double *tree_build_ns,
                         double *tree_prop_cached_ns,
                         double *tree_prop_recomp_ns,
                         double *leaf_ns, double *accum_ns);
bool create_graph_stub(GpuPlan *plan);
void destroy_plan(GpuPlan *plan);
bool destroy_fft_buffers(GpuPlan *plan, GpuFftBuffers &b, cudaStream_t stream);
void make_nodes(int Q, double Smax, std::vector<QP> &pts);
int single_kernel_max_n(int k);

}  // namespace icm_gpu_detail

#endif /* ICM_GPU_INTERNAL_H */
