#include "icm_gpu.h"
#include "icm.h"

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

#include "../devices/b200/gpu_fft_config.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

namespace {

constexpr int MAX_B_CANDIDATES = 48;
constexpr int kBCandidates[MAX_B_CANDIDATES] = {
    1, 2, 4,
    8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192,
    208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512,
    576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1536, 1792,
    2048, 2560, 3072, 3584, 4096
};
static int g_runtime_fused_max_conv_len = GPU_FUSED_MAX_CONV_LEN;
constexpr int GPU_SCHOOL_WARP_MAX_CONV = 128;
constexpr int GPU_SCHOOL_WARPS_PER_BLOCK = 4;
constexpr size_t GPU_SCHOOL_SMEM_SAFE_BYTES = 48u * 1024u;

constexpr int Q_BATCH_MAX = 8;   /* Maximum supported Q-batch width */
constexpr int Q_BATCH_DEFAULT = 4; /* Default Q-batch width */

enum {
    GPU_ENGINE_LINEAR = 0,
    GPU_ENGINE_HYBRID = 1
};

enum {
    GPU_TIER_SCHOOLBOOK = 1,
    GPU_TIER_FUSED = 2,
    GPU_TIER_CUFFT = 3
};

static double tree_school_ns_per_fma();

static std::string g_last_error;
static int g_cuda_device = -1;

static inline double now_ns_host() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static void set_last_errorf(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_last_error = buf;
}

static bool cuda_ok(cudaError_t err, const char *expr, const char *file, int line) {
    if (err == cudaSuccess) return true;
    set_last_errorf("CUDA error at %s:%d for %s: %s", file, line, expr, cudaGetErrorString(err));
    return false;
}

static bool cufft_ok(cufftResult err, const char *expr, const char *file, int line) {
    if (err == CUFFT_SUCCESS) return true;
    set_last_errorf("cuFFT error at %s:%d for %s: code=%d", file, line, expr, (int)err);
    return false;
}

#define CUDA_OK(expr) cuda_ok((expr), #expr, __FILE__, __LINE__)
#define CUFFT_OK(expr) cufft_ok((expr), #expr, __FILE__, __LINE__)

static int next_pow2_int(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void build_smooth_table(int max_n, std::vector<int> &smooth) {
    smooth.clear();
    for (int a = 1; a <= max_n; a *= 2) {
        for (int b = a; b <= max_n; b *= 3) {
            for (int c = b; c <= max_n; c *= 5) {
                for (int d = c; d <= max_n; d *= 7) {
                    smooth.push_back(d);
                    if (d > max_n / 7) break;
                }
                if (c > max_n / 5) break;
            }
            if (b > max_n / 3) break;
        }
        if (a > max_n / 2) break;
    }
    std::sort(smooth.begin(), smooth.end());
    smooth.erase(std::unique(smooth.begin(), smooth.end()), smooth.end());
}

static int next_7smooth_ge(const std::vector<int> &smooth, int n) {
    auto it = std::lower_bound(smooth.begin(), smooth.end(), n);
    if (it != smooth.end()) return *it;
    return next_pow2_int(n);
}

static int first_calib_ge(int n) {
    int lo = 0;
    int hi = GPU_N_CALIBRATED_SIZES - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (gpu_calib_sizes[mid] < n) lo = mid + 1;
        else hi = mid;
    }
    if (lo < GPU_N_CALIBRATED_SIZES && gpu_calib_sizes[lo] >= n) return lo;
    return -1;
}

static int find_calib_index(int fft_n) {
    int lo = 0;
    int hi = GPU_N_CALIBRATED_SIZES - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (gpu_calib_sizes[mid] < fft_n) lo = mid + 1;
        else hi = mid;
    }
    if (lo < GPU_N_CALIBRATED_SIZES && gpu_calib_sizes[lo] == fft_n) return lo;
    return -1;
}

static double estimate_cufft_pipeline_ns(int fft_n) {
    int idx = find_calib_index(fft_n);
    if (idx >= 0) return gpu_calib_cufft_ns[idx] + GPU_FFT_OVERHEAD_NS;
    return (double)fft_n * 0.9 + GPU_FFT_OVERHEAD_NS;
}

/* Follow CPU logic: choose the fastest calibrated smooth size >= n,
 * searching through next_pow2(n) rather than blindly picking the next smooth. */
static int fastest_fft_ge_gpu(int n) {
    if (n <= 1) return 2;
    int p2 = next_pow2_int(n);
    int i0 = first_calib_ge(n);
    if (i0 < 0) return p2;

    int best = p2;
    double best_cost = estimate_cufft_pipeline_ns(p2);
    for (int i = i0; i < GPU_N_CALIBRATED_SIZES && gpu_calib_sizes[i] <= p2; ++i) {
        int s = gpu_calib_sizes[i];
        double cost = gpu_calib_cufft_ns[i] + GPU_FFT_OVERHEAD_NS;
        if (cost < best_cost) {
            best_cost = cost;
            best = s;
        }
    }
    return best;
}

static double wrap_serial_penalty_gpu(int nparents) {
    /* Wrap kernels launch one thread per parent node. When parent count is below
     * SM count, correction work is effectively serialized and should be penalized
     * in planning (unlike fully parallel FMAs used elsewhere in the model). */
    int np = std::max(1, nparents);
    double p = (double)GPU_SM_COUNT / (double)np;
    if (p < 1.0) p = 1.0;
    const char *env = getenv("ICM_GPU_WRAP_SERIAL_SCALE");
    if (env && env[0]) {
        double x = atof(env);
        if (x > 0.0 && std::isfinite(x)) p *= x;
    }
    return p;
}

static int wrap_m_cap_gpu() {
    /* Optional hard cap for debugging; disabled by default. */
    int cap = std::numeric_limits<int>::max();
    const char *env = getenv("ICM_GPU_WRAP_M_MAX");
    if (env && env[0]) {
        int v = atoi(env);
        if (v >= 0) cap = v;
    }
    return cap;
}

static void best_fft_config_gpu(int conv_len, int len_P, double correction_scale,
                                int *out_fft_n, int *out_wrap_m) {
    int lo = first_calib_ge((conv_len > 1 ? conv_len / 2 + 1 : 1));
    if (lo < 0) {
        *out_fft_n = fastest_fft_ge_gpu(conv_len);
        *out_wrap_m = 0;
        return;
    }
    double fma_ns = tree_school_ns_per_fma();
    int wrap_cap = wrap_m_cap_gpu();
    double best_cost = std::numeric_limits<double>::infinity();
    int best_n = 0;
    int best_m = 0;
    int min_size = conv_len / 2 + 1;
    for (int i = lo; i < GPU_N_CALIBRATED_SIZES; ++i) {
        int s = gpu_calib_sizes[i];
        if (s > 2 * conv_len) break;
        if (s < min_size) continue;
        int m = (s >= conv_len) ? 0 : (conv_len - s);
        if (m > wrap_cap) continue;
        double correction = ((double)(m + 1) * (double)(m + 1) + (double)m * (double)len_P)
            * fma_ns * correction_scale;
        double cost = estimate_cufft_pipeline_ns(s) + correction;
        if (cost < best_cost) {
            best_cost = cost;
            best_n = s;
            best_m = m;
        }
    }
    if (best_n <= 0) {
        best_n = fastest_fft_ge_gpu(conv_len);
        best_m = 0;
    }
    *out_fft_n = best_n;
    *out_wrap_m = best_m;
}

static double best_fft_config_joint_gpu(int build_conv, int corr_conv, int p_eff,
                                        double correction_scale,
                                        int *out_fft_n, int *out_build_wrap_m, int *out_corr_wrap_m) {
    int max_conv = std::max(build_conv, corr_conv);
    int lo = first_calib_ge((max_conv > 1 ? max_conv / 2 + 1 : 1));
    if (lo < 0) {
        *out_fft_n = fastest_fft_ge_gpu(max_conv);
        *out_build_wrap_m = 0;
        *out_corr_wrap_m = 0;
        return std::numeric_limits<double>::infinity();
    }

    double fma_ns = tree_school_ns_per_fma();
    int wrap_cap = wrap_m_cap_gpu();
    double best_cost = std::numeric_limits<double>::infinity();
    int best_n = 0;
    int best_bm = 0;
    int best_cm = 0;
    int min_size = max_conv / 2 + 1;
    for (int i = lo; i < GPU_N_CALIBRATED_SIZES; ++i) {
        int s = gpu_calib_sizes[i];
        if (s > 2 * max_conv) break;
        if (s < min_size) continue;
        int bm = (s >= build_conv) ? 0 : (build_conv - s);
        int cm = (s >= corr_conv) ? 0 : (corr_conv - s);
        if (bm > wrap_cap || cm > wrap_cap) continue;
        double corr_input_wrap = (double)cm * (double)p_eff;
        double cost = estimate_cufft_pipeline_ns(s)
            + (double)(bm + 1) * (double)(bm + 1) * fma_ns * correction_scale
            + estimate_cufft_pipeline_ns(s) * GPU_PAIRED_CACHED_CORR_RATIO
            + 2.0 * ((double)(cm + 1) * (double)(cm + 1) + (double)corr_input_wrap)
                * fma_ns * correction_scale;
        if (cost < best_cost) {
            best_cost = cost;
            best_n = s;
            best_bm = bm;
            best_cm = cm;
        }
    }
    if (best_n <= 0) {
        best_n = fastest_fft_ge_gpu(max_conv);
        best_bm = 0;
        best_cm = 0;
    }
    *out_fft_n = best_n;
    *out_build_wrap_m = best_bm;
    *out_corr_wrap_m = best_cm;
    return best_cost;
}

static double estimate_fused_build_ns(int fft_n) {
    int idx = find_calib_index(fft_n);
    if (idx < 0) return std::numeric_limits<double>::infinity();
    if (gpu_calib_cufftdx_build_ns[idx] <= 0.0) return std::numeric_limits<double>::infinity();
    return gpu_calib_cufftdx_build_ns[idx];
}

static double estimate_fused_corr_ns(int fft_n) {
    int idx = find_calib_index(fft_n);
    if (idx < 0) return std::numeric_limits<double>::infinity();
    if (gpu_calib_cufftdx_corr_ns[idx] <= 0.0) return std::numeric_limits<double>::infinity();
    return gpu_calib_cufftdx_corr_ns[idx];
}

static int fused_max_conv_len_runtime() {
    int v = GPU_FUSED_MAX_CONV_LEN;
    const char *env = getenv("ICM_GPU_FUSED_MAX_CONV_LEN");
    if (env && env[0]) {
        int x = atoi(env);
        if (x >= 0) v = x;
    }
    return v;
}

#if ICM_HAVE_CUFFTDX
template<int FFT_N>
using cufftdx_fft_fwd_t = decltype(cufftdx::Block() + cufftdx::Size<FFT_N>() +
                                   cufftdx::Type<cufftdx::fft_type::c2c>() +
                                   cufftdx::Direction<cufftdx::fft_direction::forward>() +
                                   cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<1>() +
                                   cufftdx::SM<1000>());

template<int FFT_N>
using cufftdx_fft_inv_t = decltype(cufftdx::Block() + cufftdx::Size<FFT_N>() +
                                   cufftdx::Type<cufftdx::fft_type::c2c>() +
                                   cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                                   cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<1>() +
                                   cufftdx::SM<1000>());

#if ICM_HAVE_CUFFTDX_R2C
template<int FFT_N>
using cufftdx_r2c_t = decltype(cufftdx::Block() + cufftdx::Size<FFT_N>() +
                                cufftdx::Type<cufftdx::fft_type::r2c>() +
                                cufftdx::Direction<cufftdx::fft_direction::forward>() +
                                cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<1>() +
                                cufftdx::SM<1000>());

template<int FFT_N>
using cufftdx_c2r_t = decltype(cufftdx::Block() + cufftdx::Size<FFT_N>() +
                                cufftdx::Type<cufftdx::fft_type::c2r>() +
                                cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                                cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<1>() +
                                cufftdx::SM<1000>());
#endif /* ICM_HAVE_CUFFTDX_R2C */

template<class FFT>
__device__ inline void cufftdx_load_real(const double *src, int copy_len,
                                         typename FFT::value_type *thread_data) {
    using value_t = typename FFT::value_type;
    constexpr unsigned N = cufftdx::size_of<FFT>::value;
    const unsigned stride = FFT::stride;
    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        unsigned idx = i * stride + threadIdx.x;
        if (idx < N) {
            value_t v;
            if (idx < (unsigned)copy_len) {
                v.x = src[idx];
                v.y = 0.0;
            } else {
                v.x = 0.0;
                v.y = 0.0;
            }
            thread_data[i] = v;
        }
    }
}

template<class FFT>
__device__ inline void cufftdx_store_real(const typename FFT::value_type *thread_data,
                                          double *dst, int out_len, double scale) {
    constexpr unsigned N = cufftdx::size_of<FFT>::value;
    const unsigned stride = FFT::stride;
    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        unsigned idx = i * stride + threadIdx.x;
        if (idx < N && idx < (unsigned)out_len) {
            dst[idx] = thread_data[i].x * scale;
        }
    }
}

template<class FFT>
__device__ inline void cufftdx_mul_freq_inplace(typename FFT::value_type *lhs,
                                                const typename FFT::value_type *rhs) {
    using value_t = typename FFT::value_type;
    constexpr unsigned N = cufftdx::size_of<FFT>::value;
    const unsigned stride = FFT::stride;
    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        unsigned idx = i * stride + threadIdx.x;
        if (idx < N) {
            value_t a = lhs[i];
            value_t b = rhs[i];
            value_t o;
            o.x = a.x * b.x - a.y * b.y;
            o.y = a.x * b.y + a.y * b.x;
            lhs[i] = o;
        }
    }
}

template<class FFT>
__device__ inline void cufftdx_mul_freq_conj_inplace(typename FFT::value_type *lhs,
                                                     const typename FFT::value_type *rhs) {
    using value_t = typename FFT::value_type;
    constexpr unsigned N = cufftdx::size_of<FFT>::value;
    const unsigned stride = FFT::stride;
    for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
        unsigned idx = i * stride + threadIdx.x;
        if (idx < N) {
            value_t a = lhs[i];
            value_t b = rhs[i];
            value_t o;
            o.x = a.x * b.x + a.y * b.y;
            o.y = a.y * b.x - a.x * b.y;
            lhs[i] = o;
        }
    }
}

#if ICM_HAVE_CUFFTDX_R2C
/* ── R2C/C2R load/store helpers ──
 *
 * For R2C, the input is real: we load into the scalar (real) components of thread_data.
 * For C2R, the output is real: we extract from the scalar (real) components of thread_data.
 * In cuFFTDx R2C/C2R, the thread_data is still complex_t[], but the real-domain side
 * is accessed by reinterpreting as scalar_t*.
 */
template<class R2C_FFT>
__device__ inline void cufftdx_load_real_r2c(const double *src, int copy_len,
                                              typename R2C_FFT::value_type *thread_data) {
    using scalar_t = typename R2C_FFT::value_type::value_type;
    const unsigned stride = R2C_FFT::stride;
    for (unsigned i = 0; i < R2C_FFT::elements_per_thread; ++i) {
        unsigned idx = i * stride + threadIdx.x;
        reinterpret_cast<scalar_t*>(thread_data)[i] =
            (idx < (unsigned)copy_len) ? src[idx] : 0.0;
    }
}

template<class C2R_FFT>
__device__ inline void cufftdx_store_real_c2r(const typename C2R_FFT::value_type *thread_data,
                                               double *dst, int out_len, double scale) {
    using scalar_t = typename C2R_FFT::value_type::value_type;
    const unsigned stride = C2R_FFT::stride;
    for (unsigned i = 0; i < C2R_FFT::elements_per_thread; ++i) {
        unsigned idx = i * stride + threadIdx.x;
        if (idx < (unsigned)out_len) {
            dst[idx] = reinterpret_cast<const scalar_t*>(thread_data)[i] * scale;
        }
    }
}
#endif /* ICM_HAVE_CUFFTDX_R2C */

template<int FFT_N>
__launch_bounds__(cufftdx_fft_fwd_t<FFT_N>::max_threads_per_block)
__global__ static void k_cufftdx_build_parent(const double *child, int cps,
                                              double *parent, int pps,
                                              int nparents, double inv_fft_n) {
    if (blockIdx.x >= (unsigned)nparents) return;
    using FFTFwd = cufftdx_fft_fwd_t<FFT_N>;
    using FFTInv = cufftdx_fft_inv_t<FFT_N>;
    using complex_t = typename FFTFwd::value_type;
    complex_t a[FFTFwd::storage_size];
    complex_t b[FFTFwd::storage_size];
    extern __shared__ __align__(alignof(double2)) complex_t shared_mem[];

    int p = (int)blockIdx.x;
    const double *L = child + (size_t)(2 * p) * (size_t)cps;
    const double *R = child + (size_t)(2 * p + 1) * (size_t)cps;
    double *out = parent + (size_t)p * (size_t)pps;

    cufftdx_load_real<FFTFwd>(L, cps, a);
    FFTFwd().execute(a, shared_mem);
    cufftdx_load_real<FFTFwd>(R, cps, b);
    FFTFwd().execute(b, shared_mem);
    cufftdx_mul_freq_inplace<FFTFwd>(a, b);
    FFTInv().execute(a, shared_mem);
    cufftdx_store_real<FFTInv>(a, out, pps, inv_fft_n);
}

template<int FFT_N>
__launch_bounds__(cufftdx_fft_fwd_t<FFT_N>::max_threads_per_block)
__global__ static void k_cufftdx_corr_pair_parent(const double *g_parent, int parent_gsz, int len_g,
                                                  const double *child_poly, int cps, int len_P,
                                                  double *g_child, int child_gsz, int len_out,
                                                  int nparents, double inv_fft_n) {
    if (blockIdx.x >= (unsigned)nparents) return;
    using FFTFwd = cufftdx_fft_fwd_t<FFT_N>;
    using FFTInv = cufftdx_fft_inv_t<FFT_N>;
    using complex_t = typename FFTFwd::value_type;
    complex_t gbuf[FFTFwd::storage_size];
    complex_t pbuf[FFTFwd::storage_size];
    complex_t gspec_saved[FFTFwd::elements_per_thread];
    extern __shared__ __align__(alignof(double2)) complex_t shared_mem[];

    int p = (int)blockIdx.x;
    const double *gp = g_parent + (size_t)p * (size_t)parent_gsz;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)cps;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)cps;
    double *outL = g_child + (size_t)(2 * p) * (size_t)child_gsz;
    double *outR = g_child + (size_t)(2 * p + 1) * (size_t)child_gsz;

    cufftdx_load_real<FFTFwd>(gp, len_g, gbuf);
    FFTFwd().execute(gbuf, shared_mem);
    for (unsigned i = 0; i < FFTFwd::elements_per_thread; ++i) {
        gspec_saved[i] = gbuf[i];
    }

    cufftdx_load_real<FFTFwd>(PR, len_P, pbuf);
    FFTFwd().execute(pbuf, shared_mem);
    cufftdx_mul_freq_conj_inplace<FFTFwd>(gbuf, pbuf);
    FFTInv().execute(gbuf, shared_mem);
    cufftdx_store_real<FFTInv>(gbuf, outL, len_out, inv_fft_n);

    cufftdx_load_real<FFTFwd>(PL, len_P, pbuf);
    FFTFwd().execute(pbuf, shared_mem);
    for (unsigned i = 0; i < FFTFwd::elements_per_thread; ++i) {
        gbuf[i] = gspec_saved[i];
    }
    cufftdx_mul_freq_conj_inplace<FFTFwd>(gbuf, pbuf);
    FFTInv().execute(gbuf, shared_mem);
    cufftdx_store_real<FFTInv>(gbuf, outR, len_out, inv_fft_n);
}

template<int FFT_N>
static bool launch_cufftdx_build_t(const double *child, int cps,
                                   double *parent, int pps, int nparents,
                                   double inv_fft_n, cudaStream_t stream) {
    using FFTFwd = cufftdx_fft_fwd_t<FFT_N>;
    using FFTInv = cufftdx_fft_inv_t<FFT_N>;
    static_assert(FFTFwd::ffts_per_block == 1, "Expected one FFT per block");
    static_assert(FFTFwd::block_dim.y == 1, "Unexpected cuFFTDx block_dim.y");
    size_t shmem = std::max((size_t)FFTFwd::shared_memory_size, (size_t)FFTInv::shared_memory_size);
    if (!CUDA_OK(cudaFuncSetAttribute(k_cufftdx_build_parent<FFT_N>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)shmem))) return false;
    k_cufftdx_build_parent<FFT_N><<<nparents, FFTFwd::block_dim, shmem, stream>>>(
        child, cps, parent, pps, nparents, inv_fft_n);
    return CUDA_OK(cudaGetLastError());
}

template<int FFT_N>
static bool launch_cufftdx_corr_t(const double *g_parent, int parent_gsz, int len_g,
                                  const double *child_poly, int cps, int len_P,
                                  double *g_child, int child_gsz, int len_out, int nparents,
                                  double inv_fft_n, cudaStream_t stream) {
    using FFTFwd = cufftdx_fft_fwd_t<FFT_N>;
    using FFTInv = cufftdx_fft_inv_t<FFT_N>;
    static_assert(FFTFwd::ffts_per_block == 1, "Expected one FFT per block");
    static_assert(FFTFwd::block_dim.y == 1, "Unexpected cuFFTDx block_dim.y");
    size_t shmem = std::max((size_t)FFTFwd::shared_memory_size, (size_t)FFTInv::shared_memory_size);
    if (!CUDA_OK(cudaFuncSetAttribute(k_cufftdx_corr_pair_parent<FFT_N>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)shmem))) return false;
    k_cufftdx_corr_pair_parent<FFT_N><<<nparents, FFTFwd::block_dim, shmem, stream>>>(
        g_parent, parent_gsz, len_g,
        child_poly, cps, len_P,
        g_child, child_gsz, len_out, nparents, inv_fft_n);
    return CUDA_OK(cudaGetLastError());
}
#endif

/* ── R2C/C2R fused kernels ──
 *
 * These do the same work as the C2C fused kernels above, but use R2C for the forward
 * transform and C2R for the inverse. Since the input polynomials are real, R2C does
 * roughly half the FFT work (only computing N/2+1 complex outputs for Hermitian symmetry).
 * The pointwise multiply operates on the complex frequency-domain data, and C2R converts
 * back to real output.
 */
#if ICM_HAVE_CUFFTDX_R2C

template<int FFT_N>
__launch_bounds__(cufftdx_r2c_t<FFT_N>::max_threads_per_block)
__global__ static void k_cufftdx_build_parent_r2c(const double *child, int cps,
                                                    double *parent, int pps,
                                                    int nparents, double inv_fft_n) {
    if (blockIdx.x >= (unsigned)nparents) return;
    using R2C = cufftdx_r2c_t<FFT_N>;
    using C2R = cufftdx_c2r_t<FFT_N>;
    using complex_t = typename R2C::value_type;
    complex_t a[R2C::storage_size];
    complex_t b[R2C::storage_size];
    extern __shared__ __align__(alignof(double2)) complex_t shared_mem[];

    int p = (int)blockIdx.x;
    const double *L = child + (size_t)(2 * p) * (size_t)cps;
    const double *R_ptr = child + (size_t)(2 * p + 1) * (size_t)cps;
    double *out = parent + (size_t)p * (size_t)pps;

    cufftdx_load_real_r2c<R2C>(L, cps, a);
    R2C().execute(a, shared_mem);
    cufftdx_load_real_r2c<R2C>(R_ptr, cps, b);
    R2C().execute(b, shared_mem);

    /* Pointwise complex multiply in frequency domain */
    for (unsigned i = 0; i < R2C::elements_per_thread; ++i) {
        complex_t va = a[i], vb = b[i];
        complex_t vo;
        vo.x = va.x * vb.x - va.y * vb.y;
        vo.y = va.x * vb.y + va.y * vb.x;
        a[i] = vo;
    }

    C2R().execute(a, shared_mem);
    cufftdx_store_real_c2r<C2R>(a, out, pps, inv_fft_n);
}

template<int FFT_N>
__launch_bounds__(cufftdx_r2c_t<FFT_N>::max_threads_per_block)
__global__ static void k_cufftdx_corr_pair_parent_r2c(
        const double *g_parent, int parent_gsz, int len_g,
        const double *child_poly, int cps, int len_P,
        double *g_child, int child_gsz, int len_out,
        int nparents, double inv_fft_n) {
    if (blockIdx.x >= (unsigned)nparents) return;
    using R2C = cufftdx_r2c_t<FFT_N>;
    using C2R = cufftdx_c2r_t<FFT_N>;
    using complex_t = typename R2C::value_type;
    complex_t gbuf[R2C::storage_size];
    complex_t pbuf[R2C::storage_size];
    complex_t gspec_saved[R2C::elements_per_thread];
    extern __shared__ __align__(alignof(double2)) complex_t shared_mem[];

    int p = (int)blockIdx.x;
    const double *gp = g_parent + (size_t)p * (size_t)parent_gsz;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)cps;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)cps;
    double *outL = g_child + (size_t)(2 * p) * (size_t)child_gsz;
    double *outR = g_child + (size_t)(2 * p + 1) * (size_t)child_gsz;

    /* Forward R2C of g_parent */
    cufftdx_load_real_r2c<R2C>(gp, len_g, gbuf);
    R2C().execute(gbuf, shared_mem);

    /* Save g spectrum for reuse */
    for (unsigned i = 0; i < R2C::elements_per_thread; ++i)
        gspec_saved[i] = gbuf[i];

    /* Left child: g * conj(PR) */
    cufftdx_load_real_r2c<R2C>(PR, len_P, pbuf);
    R2C().execute(pbuf, shared_mem);
    for (unsigned i = 0; i < R2C::elements_per_thread; ++i) {
        complex_t g = gbuf[i], p_val = pbuf[i];
        gbuf[i].x = g.x * p_val.x + g.y * p_val.y;
        gbuf[i].y = g.y * p_val.x - g.x * p_val.y;
    }
    C2R().execute(gbuf, shared_mem);
    cufftdx_store_real_c2r<C2R>(gbuf, outL, len_out, inv_fft_n);

    /* Right child: g * conj(PL), restore g spectrum */
    cufftdx_load_real_r2c<R2C>(PL, len_P, pbuf);
    R2C().execute(pbuf, shared_mem);
    for (unsigned i = 0; i < R2C::elements_per_thread; ++i)
        gbuf[i] = gspec_saved[i];
    for (unsigned i = 0; i < R2C::elements_per_thread; ++i) {
        complex_t g = gbuf[i], p_val = pbuf[i];
        gbuf[i].x = g.x * p_val.x + g.y * p_val.y;
        gbuf[i].y = g.y * p_val.x - g.x * p_val.y;
    }
    C2R().execute(gbuf, shared_mem);
    cufftdx_store_real_c2r<C2R>(gbuf, outR, len_out, inv_fft_n);
}

template<int FFT_N>
static bool launch_cufftdx_build_r2c_t(const double *child, int cps,
                                        double *parent, int pps, int nparents,
                                        double inv_fft_n, cudaStream_t stream) {
    using R2C = cufftdx_r2c_t<FFT_N>;
    using C2R = cufftdx_c2r_t<FFT_N>;
    static_assert(R2C::ffts_per_block == 1, "Expected one FFT per block");
    static_assert(R2C::block_dim.y == 1, "Unexpected cuFFTDx block_dim.y");
    size_t shmem = std::max((size_t)R2C::shared_memory_size, (size_t)C2R::shared_memory_size);
    if (!CUDA_OK(cudaFuncSetAttribute(k_cufftdx_build_parent_r2c<FFT_N>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)shmem))) return false;
    k_cufftdx_build_parent_r2c<FFT_N><<<nparents, R2C::block_dim, shmem, stream>>>(
        child, cps, parent, pps, nparents, inv_fft_n);
    return CUDA_OK(cudaGetLastError());
}

template<int FFT_N>
static bool launch_cufftdx_corr_r2c_t(const double *g_parent, int parent_gsz, int len_g,
                                       const double *child_poly, int cps, int len_P,
                                       double *g_child, int child_gsz, int len_out, int nparents,
                                       double inv_fft_n, cudaStream_t stream) {
    using R2C = cufftdx_r2c_t<FFT_N>;
    using C2R = cufftdx_c2r_t<FFT_N>;
    static_assert(R2C::ffts_per_block == 1, "Expected one FFT per block");
    static_assert(R2C::block_dim.y == 1, "Unexpected cuFFTDx block_dim.y");
    size_t shmem = std::max((size_t)R2C::shared_memory_size, (size_t)C2R::shared_memory_size);
    if (!CUDA_OK(cudaFuncSetAttribute(k_cufftdx_corr_pair_parent_r2c<FFT_N>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)shmem))) return false;
    k_cufftdx_corr_pair_parent_r2c<FFT_N><<<nparents, R2C::block_dim, shmem, stream>>>(
        g_parent, parent_gsz, len_g,
        child_poly, cps, len_P,
        g_child, child_gsz, len_out, nparents, inv_fft_n);
    return CUDA_OK(cudaGetLastError());
}

#endif /* ICM_HAVE_CUFFTDX_R2C */

static bool is_cufftdx_supported_fft_n(int fft_n) {
    switch (fft_n) {
        case 64:
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
            return true;
        default:
            return false;
    }
}

static bool launch_cufftdx_build_dispatch(int fft_n,
                                          const double *child, int cps,
                                          double *parent, int pps, int nparents,
                                          double inv_fft_n, cudaStream_t stream) {
#if ICM_HAVE_CUFFTDX
    switch (fft_n) {
        case 64: return launch_cufftdx_build_t<64>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 128: return launch_cufftdx_build_t<128>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 256: return launch_cufftdx_build_t<256>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 512: return launch_cufftdx_build_t<512>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 1024: return launch_cufftdx_build_t<1024>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 2048: return launch_cufftdx_build_t<2048>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 4096: return launch_cufftdx_build_t<4096>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        default: return false;
    }
#else
    (void)fft_n; (void)child; (void)cps; (void)parent; (void)pps; (void)nparents; (void)inv_fft_n; (void)stream;
    return false;
#endif
}

static bool launch_cufftdx_corr_dispatch(int fft_n,
                                         const double *g_parent, int parent_gsz, int len_g,
                                         const double *child_poly, int cps, int len_P,
                                         double *g_child, int child_gsz, int len_out, int nparents,
                                         double inv_fft_n, cudaStream_t stream) {
#if ICM_HAVE_CUFFTDX
    switch (fft_n) {
        case 64:
            return launch_cufftdx_corr_t<64>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                             g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 128:
            return launch_cufftdx_corr_t<128>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                              g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 256:
            return launch_cufftdx_corr_t<256>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                              g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 512:
            return launch_cufftdx_corr_t<512>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                              g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 1024:
            return launch_cufftdx_corr_t<1024>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                               g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 2048:
            return launch_cufftdx_corr_t<2048>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                               g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 4096:
            return launch_cufftdx_corr_t<4096>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                               g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        default:
            return false;
    }
#else
    (void)fft_n; (void)g_parent; (void)parent_gsz; (void)len_g; (void)child_poly; (void)cps; (void)len_P;
    (void)g_child; (void)child_gsz; (void)len_out; (void)nparents; (void)inv_fft_n; (void)stream;
    return false;
#endif
}

static bool launch_cufftdx_build_r2c_dispatch(int fft_n,
                                               const double *child, int cps,
                                               double *parent, int pps, int nparents,
                                               double inv_fft_n, cudaStream_t stream) {
#if ICM_HAVE_CUFFTDX_R2C
    switch (fft_n) {
        case 64: return launch_cufftdx_build_r2c_t<64>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 128: return launch_cufftdx_build_r2c_t<128>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 256: return launch_cufftdx_build_r2c_t<256>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 512: return launch_cufftdx_build_r2c_t<512>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 1024: return launch_cufftdx_build_r2c_t<1024>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 2048: return launch_cufftdx_build_r2c_t<2048>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        case 4096: return launch_cufftdx_build_r2c_t<4096>(child, cps, parent, pps, nparents, inv_fft_n, stream);
        default: return false;
    }
#else
    (void)fft_n; (void)child; (void)cps; (void)parent; (void)pps; (void)nparents; (void)inv_fft_n; (void)stream;
    return false;
#endif
}

static bool launch_cufftdx_corr_r2c_dispatch(int fft_n,
                                              const double *g_parent, int parent_gsz, int len_g,
                                              const double *child_poly, int cps, int len_P,
                                              double *g_child, int child_gsz, int len_out, int nparents,
                                              double inv_fft_n, cudaStream_t stream) {
#if ICM_HAVE_CUFFTDX_R2C
    switch (fft_n) {
        case 64:
            return launch_cufftdx_corr_r2c_t<64>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                   g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 128:
            return launch_cufftdx_corr_r2c_t<128>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                    g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 256:
            return launch_cufftdx_corr_r2c_t<256>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                    g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 512:
            return launch_cufftdx_corr_r2c_t<512>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                    g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 1024:
            return launch_cufftdx_corr_r2c_t<1024>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                     g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 2048:
            return launch_cufftdx_corr_r2c_t<2048>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                     g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        case 4096:
            return launch_cufftdx_corr_r2c_t<4096>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                     g_child, child_gsz, len_out, nparents, inv_fft_n, stream);
        default:
            return false;
    }
#else
    (void)fft_n; (void)g_parent; (void)parent_gsz; (void)len_g; (void)child_poly; (void)cps; (void)len_P;
    (void)g_child; (void)child_gsz; (void)len_out; (void)nparents; (void)inv_fft_n; (void)stream;
    return false;
#endif
}

static int best_k_pad_gpu(int k, const std::vector<int> &smooth) {
    if (k <= 2) return k;
    if ((k & (k - 1)) == 0) return k;
    int ceil_k = k + k / 8;
    if (ceil_k < k + 4) ceil_k = k + 4;
    auto it = std::lower_bound(smooth.begin(), smooth.end(), k);
    int best = k;
    double best_cost = std::numeric_limits<double>::infinity();
    for (; it != smooth.end() && *it <= ceil_k; ++it) {
        int kp = *it;
        int conv_len = 2 * kp - 1;
        int fft_n = fastest_fft_ge_gpu(conv_len);
        double cost = estimate_cufft_pipeline_ns(fft_n) + 0.5 * (double)(kp - k);
        if (cost < best_cost) {
            best_cost = cost;
            best = kp;
        }
    }
    return best;
}

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
    int q_batch = 1;  /* Number of Q-points processed per batched iteration */

    IcmGpuOptions opts{};

    std::vector<int> sort_perm;
    std::vector<int> inv_perm;
    std::vector<double> S_sorted;
    std::vector<double> payout_host;

    std::vector<int> nn;
    std::vector<int> psz;
    std::vector<int> fft_stride;  /* Per-level memory stride: >= psz, accommodates cuFFT idist/odist */
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

    /* Q-batch buffers: one a_sorted buffer per Q-point in the batch */
    double *d_a_qbatch[Q_BATCH_MAX] = {};
    double *d_inner_qbatch = nullptr;       /* q_batch * n doubles */
    double *d_block_prods_qbatch = nullptr;  /* q_batch * N_tree * (B+1) doubles */
    /* Small device buffers for Q-batch kernel parameters */
    double **d_qb_a_ptrs = nullptr;         /* q_batch device pointers */
    double *d_qb_weights = nullptr;         /* q_batch doubles */
    double *d_qb_inv_vs = nullptr;          /* q_batch doubles */

    std::vector<double *> d_poly_levels;
    std::vector<double *> d_g_levels;
    std::vector<cufftDoubleComplex *> d_fft_cache;
    std::vector<GpuFftBuffers> build_fft;
    std::vector<GpuFftBuffers> corr_fft;

    cudaStream_t stream_compute = nullptr;
    cudaStream_t stream_aux = nullptr;
    cudaEvent_t evt_a_ready[2] = {nullptr, nullptr};

    /* Multi-stream Q-pipeline: alternate leaf/block_prods for double-buffering */
    double *d_poly_leaves_alt = nullptr;   /* alternate leaf level for pipelining */
    double *d_block_prods_alt = nullptr;   /* alternate block_prods for pipelining */
    cudaEvent_t evt_prop_done = nullptr;   /* signals propagation is done reading poly_levels */

    cudaGraph_t graph[2] = {nullptr, nullptr};
    cudaGraphExec_t graph_exec[2] = {nullptr, nullptr};
    bool graph_ready[2] = {false, false};

    bool use_async_pool = false;
    cudaMemPool_t mem_pool = nullptr;

    /* Phase A2: FFT cache validity tracking (instead of freeing cache buffers) */
    std::vector<bool> fft_cache_valid;

    /* Phase B: Shared FFT work buffers across all cuFFT levels */
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
    SharedFftWork shared_build_work;
    SharedFftWork shared_corr_work;
    void *shared_cufft_workspace = nullptr;
    size_t shared_cufft_workspace_bytes = 0;

    /* Arena: single device allocation for all plan buffers */
    void *arena_base = nullptr;
    size_t arena_total_bytes = 0;

    size_t peak_vram_bytes = 0;
    size_t current_vram_bytes = 0;
};

static void update_vram_alloc(GpuPlan *plan, size_t bytes) {
    plan->current_vram_bytes += bytes;
    if (plan->current_vram_bytes > plan->peak_vram_bytes) {
        plan->peak_vram_bytes = plan->current_vram_bytes;
    }
}

static void update_vram_free(GpuPlan *plan, size_t bytes) {
    if (bytes > plan->current_vram_bytes) {
        plan->current_vram_bytes = 0;
    } else {
        plan->current_vram_bytes -= bytes;
    }
}

static bool alloc_device(GpuPlan *plan, void **ptr, size_t bytes, cudaStream_t stream) {
    if (bytes == 0) {
        *ptr = nullptr;
        return true;
    }
    if (plan->use_async_pool) {
        if (!CUDA_OK(cudaMallocAsync(ptr, bytes, stream))) return false;
    } else {
        if (!CUDA_OK(cudaMalloc(ptr, bytes))) return false;
    }
    update_vram_alloc(plan, bytes);
    return true;
}

static bool free_device(GpuPlan *plan, void *ptr, size_t bytes, cudaStream_t stream) {
    if (!ptr) return true;
    if (plan->use_async_pool) {
        if (!CUDA_OK(cudaFreeAsync(ptr, stream))) return false;
    } else {
        if (!CUDA_OK(cudaFree(ptr))) return false;
    }
    update_vram_free(plan, bytes);
    return true;
}

static int pick_fft_size_for_conv(const std::vector<int> &smooth, int conv_len) {
    (void)smooth;
    return fastest_fft_ge_gpu(conv_len);
}

static double tree_school_ns_per_fma() {
    /* GPU_SCHOOL_FMA_NS is measured from a pure FMA stream and materially
     * under-estimates full polynomial schoolbook kernels (register pressure,
     * indexing, memory traffic, __syncwarp). Clamp by the measured block-build
     * FMA cost so tier assignment does not over-predict schoolbook.
     * Benchmarking confirmed B=1 (all-schoolbook bottom levels) is 20-56%
     * SLOWER than larger B, validating this clamp. */
#ifdef GPU_TREE_SCHOOL_NS_PER_FMA
    return GPU_TREE_SCHOOL_NS_PER_FMA;
#else
    return std::max(GPU_SCHOOL_FMA_NS, GPU_BLOCK_BUILD_NS_PER_FMA);
#endif
}

static double model_ns_per_fma_override(const char *env_name, double fallback) {
    const char *v = getenv(env_name);
    if (!v || !v[0]) return fallback;
    double x = atof(v);
    if (!(x > 0.0) || !std::isfinite(x)) return fallback;
    return x;
}

static double block_build_ns_per_fma_model() {
    /* Use measured calibration value directly; clamp only invalid inputs. */
    double base = GPU_BLOCK_BUILD_NS_PER_FMA;
    if (!(base > 0.0) || !std::isfinite(base)) base = std::max(GPU_SCHOOL_FMA_NS, 1e-3);
    return model_ns_per_fma_override("ICM_GPU_BLOCK_FMA_NS", base);
}

static double leaf_extract_ns_per_fma_model() {
    /* Use measured calibration value directly; clamp only invalid inputs. */
    double base = GPU_LEAF_EXTRACT_NS_PER_FMA;
    if (!(base > 0.0) || !std::isfinite(base)) base = std::max(GPU_SCHOOL_FMA_NS, 1e-3);
    return model_ns_per_fma_override("ICM_GPU_LEAF_FMA_NS", base);
}

static int pick_tier_for_fft_len(int fft_n, int conv_len) {
    double school = (double)conv_len * (double)conv_len * tree_school_ns_per_fma();
    double cufft = estimate_cufft_pipeline_ns(fft_n);
    double fused = estimate_fused_build_ns(fft_n);
    if (conv_len <= g_runtime_fused_max_conv_len && fused < school && fused < cufft) return GPU_TIER_FUSED;
    if (school < cufft) return GPU_TIER_SCHOOLBOOK;
    return GPU_TIER_CUFFT;
}

struct CandidateCost {
    int B = 0;
    double total_ns = std::numeric_limits<double>::infinity();
};

static void build_tree_geometry(int n_leaves, int leaf_degree, int k_pad,
                                int leaf_extract, std::vector<int> &nn,
                                std::vector<int> &psz, std::vector<size_t> &plev_off,
                                std::vector<int> &g_needed, std::vector<int> &below_sat,
                                std::vector<int> &n_real, int &N, int &L) {
    N = 1;
    while (N < n_leaves) N <<= 1;
    L = 0;
    int tmp = N;
    while (tmp > 1) {
        tmp >>= 1;
        ++L;
    }
    ++L;
    nn.assign(L, 0);
    psz.assign(L, 0);
    plev_off.assign(L, 0);
    g_needed.assign(L, 0);
    below_sat.assign(L, 0);
    n_real.assign(L, 0);

    n_real[0] = n_leaves;
    for (int ell = 1; ell < L; ++ell) n_real[ell] = (n_real[ell - 1] + 1) / 2;

    size_t off = 0;
    for (int ell = 0; ell < L; ++ell) {
        nn[ell] = N >> ell;
        long d = (long)leaf_degree * (1L << (ell + 1));
        psz[ell] = (d > k_pad) ? k_pad : (int)d;
        plev_off[ell] = off;
        off += (size_t)nn[ell] * (size_t)psz[ell];
    }

    g_needed[0] = std::min(leaf_extract, psz[0]);
    for (int ell = 1; ell < L; ++ell) {
        int need = g_needed[ell - 1] + psz[ell - 1] - 1;
        g_needed[ell] = std::min(need, psz[ell]);
    }
    for (int ell = 1; ell < L; ++ell) {
        int cps = psz[ell - 1];
        if (psz[ell] == 2 * cps && cps >= 2) below_sat[ell] = 1;
    }
}

static double estimate_candidate_cost(int n, int k_pad, int B, const std::vector<int> &smooth) {
    (void)smooth;
    if (B > n || B > k_pad) return std::numeric_limits<double>::infinity();

    int nblocks = (n + B - 1) / B;
    int N = 0;
    int L = 0;
    std::vector<int> nn, psz, g_needed, below_sat, n_real;
    std::vector<size_t> plev_off;
    build_tree_geometry(nblocks, B, k_pad, B, nn, psz, plev_off, g_needed, below_sat, n_real, N, L);

    double nblocks_real = (double)n / (double)B;
    double occ_penalty = 1.0;
    if (nblocks_real < (double)GPU_SM_COUNT) occ_penalty = (double)GPU_SM_COUNT / std::max(1.0, nblocks_real);

    double block_fmas = ((double)n / B) * ((double)B * (B + 1) / 2.0);
    double block_ns = block_fmas * block_build_ns_per_fma_model() * occ_penalty;

    double fma_ns = tree_school_ns_per_fma();
    double tree_ns = 0.0;
    for (int ell = 1; ell < L - 1; ++ell) {
        int cps = psz[ell - 1];
        int pgsz = psz[ell];
        int is_below = below_sat[ell];
        int d_eff = is_below ? (cps / 2) : (cps - 1);
        int p_eff = is_below ? (cps / 2 + 1) : cps;
        int out_needed = g_needed[ell - 1];
        int g_eff_needed = out_needed + p_eff - 1;
        int g_eff_max = is_below ? (cps + cps / 2) : pgsz;
        int g_eff = std::min(g_eff_needed, g_eff_max);

        int conv_build = is_below ? (2 * (cps / 2)) : (2 * cps - 1);
        int conv_corr = g_eff + p_eff - 1;
        double wrap_scale = wrap_serial_penalty_gpu(nn[ell]);
        double school_build = (double)(d_eff + 1) * (double)(d_eff + 1) * fma_ns;
        double school_corr = 2.0 * (double)p_eff * (double)out_needed * fma_ns;

        int bfn = 0, bwm = 0;
        best_fft_config_gpu(conv_build, 0, wrap_scale, &bfn, &bwm);
        double fft_build = estimate_cufft_pipeline_ns(bfn)
            + (double)(bwm + 1) * (double)(bwm + 1) * fma_ns * wrap_scale;
        if (fft_build >= school_build) {
            /* Runtime executes padded level width nn[ell] (power-of-two tree),
             * not only the number of real nodes. */
            tree_ns += (double)nn[ell] * (school_build + school_corr);
            continue;
        }

        int jfn = 0, jbm = 0, jcm = 0;
        double joint_cost = best_fft_config_joint_gpu(conv_build, conv_corr, p_eff, wrap_scale, &jfn, &jbm, &jcm);

        int cfn = 0, cwm = 0;
        best_fft_config_gpu(conv_corr, p_eff, wrap_scale, &cfn, &cwm);
        double indep_cost = fft_build
            + estimate_cufft_pipeline_ns(cfn) * GPU_INDEP_PAIR_RATIO
            + 2.0 * ((double)(cwm + 1) * (double)(cwm + 1) + (double)cwm * (double)p_eff)
                * fma_ns * wrap_scale;

        int fft_n = 0;
        int bwrap = 0;
        int cwrap = 0;
        if (joint_cost < indep_cost) {
            fft_n = jfn;
            bwrap = jbm;
            cwrap = jcm;
        } else {
            fft_n = cfn;
            bwrap = (cfn >= conv_build) ? 0 : (conv_build - cfn);
            cwrap = cwm;
        }

        int tier = pick_tier_for_fft_len(fft_n, conv_build);
        double build_ns = 0.0;
        double corr_ns = 0.0;
        if (tier == GPU_TIER_SCHOOLBOOK) {
            build_ns = school_build;
            corr_ns = school_corr;
        } else if (tier == GPU_TIER_FUSED) {
            build_ns = estimate_fused_build_ns(fft_n);
            corr_ns = estimate_fused_corr_ns(fft_n);
            if (!std::isfinite(build_ns) || !std::isfinite(corr_ns)) {
                build_ns = estimate_cufft_pipeline_ns(fft_n);
                corr_ns = build_ns * GPU_PAIRED_CACHED_CORR_RATIO;
            }
            build_ns += (double)(bwrap + 1) * (double)(bwrap + 1) * fma_ns * wrap_scale;
            corr_ns += 2.0 * ((double)(cwrap + 1) * (double)(cwrap + 1) + (double)cwrap * (double)p_eff)
                * fma_ns * wrap_scale;
        } else {
            build_ns = estimate_cufft_pipeline_ns(fft_n)
                + (double)(bwrap + 1) * (double)(bwrap + 1) * fma_ns * wrap_scale;
            corr_ns = estimate_cufft_pipeline_ns(fft_n) * GPU_PAIRED_CACHED_CORR_RATIO
                + 2.0 * ((double)(cwrap + 1) * (double)(cwrap + 1) + (double)cwrap * (double)p_eff)
                    * fma_ns * wrap_scale;
        }
        tree_ns += (double)nn[ell] * (build_ns + corr_ns);
    }

    double leaf_fmas = (double)n * (double)B;
    double leaf_ns = leaf_fmas * leaf_extract_ns_per_fma_model() * occ_penalty;
    return block_ns + tree_ns + leaf_ns;
}

static int gpu_select_best_B_est(int n, int k_pad, const std::vector<int> &smooth) {
    CandidateCost best{};
    best.B = 16;
    for (int i = 0; i < MAX_B_CANDIDATES; ++i) {
        int B = kBCandidates[i];
        if (B > n || B > k_pad) continue;
        double c = estimate_candidate_cost(n, k_pad, B, smooth);
        if (c < best.total_ns) {
            best.total_ns = c;
            best.B = B;
        }
    }
    return best.B;
}

static int gpu_select_engine_est(int n, int k_pad, int B, const std::vector<int> &smooth) {
    if (n < 16 || k_pad < 4) return GPU_ENGINE_LINEAR;
    double linear_fma_ns = std::max(GPU_SCHOOL_FMA_NS, block_build_ns_per_fma_model());
    double linear_ns = (double)n * (double)k_pad * 2.0 * linear_fma_ns;
    double hybrid_ns = estimate_candidate_cost(n, k_pad, B, smooth);
    return (hybrid_ns < linear_ns) ? GPU_ENGINE_HYBRID : GPU_ENGINE_LINEAR;
}

__global__ static void k_compute_a(const double *S_sorted, double *a_sorted, int n, double logv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double arg = S_sorted[i] * logv;
    a_sorted[i] = (arg < -700.0) ? 0.0 : exp(arg);
}

__global__ static void k_compute_a_from_ptr(const double *S_sorted, double *a_sorted,
                                            int n, const double *logv_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double logv = logv_ptr[0];
    double arg = S_sorted[i] * logv;
    a_sorted[i] = (arg < -700.0) ? 0.0 : exp(arg);
}

__global__ static void k_zero(double *x, size_t n) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i < n) x[i] = 0.0;
}

__global__ static void k_set_root_g(double *g_root, int root_gsz, const double *payout, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= root_gsz) return;
    g_root[i] = (i < k) ? payout[i] : 0.0;
}

/* B=1 leaf setup: each player is a leaf polynomial [a[i], 1-a[i]] */
__global__ static void k_set_leaves_b1(const double *a_sorted, int n, int N_tree,
                                       int leaf_psz, double *leaves) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_tree) return;
    double *leaf = leaves + (size_t)i * (size_t)leaf_psz;
    if (i < n) {
        double ai = a_sorted[i];
        leaf[0] = ai;
        if (leaf_psz > 1) leaf[1] = 1.0 - ai;
        for (int m = 2; m < leaf_psz; ++m) leaf[m] = 0.0;
    } else {
        /* Padding leaf = 1 (identity for multiplication) */
        leaf[0] = 1.0;
        for (int m = 1; m < leaf_psz; ++m) leaf[m] = 0.0;
    }
}

/* B=1 leaf extract: inner[i] = g_leaf[i][0] (no synthetic division needed) */
__global__ static void k_leaf_extract_b1(int n, const double *g_leaf,
                                         int leaf_psz, double *inner_sorted) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    inner_sorted[i] = g_leaf[(size_t)i * (size_t)leaf_psz];
}

__global__ static void k_block_build(const double *a_sorted, int n, int B,
                                     int nblocks, int N_tree,
                                     int leaf_psz, double *leaves, double *block_prods) {
    int b = blockIdx.x;
    if (b >= N_tree) return;
    int t = threadIdx.x;

    double *leaf = leaves + (size_t)b * (size_t)leaf_psz;
    double *P = block_prods + (size_t)b * (size_t)(B + 1);
    extern __shared__ double sh[];
    double *curr = sh;
    double *next = sh + (B + 1);

    if (b >= nblocks) {
        for (int m = t; m < B + 1; m += blockDim.x) P[m] = 0.0;
        for (int m = t; m < leaf_psz; m += blockDim.x) leaf[m] = 0.0;
        if (t == 0) {
            P[0] = 1.0;
            leaf[0] = 1.0;
        }
        return;
    }

    int start = b * B;
    int end = start + B;
    if (end > n) end = n;
    int bsize = end - start;

    for (int m = t; m < B + 1; m += blockDim.x) {
        curr[m] = 0.0;
        next[m] = 0.0;
    }
    __syncthreads();
    if (t == 0) curr[0] = 1.0;
    __syncthreads();

    for (int r = 0; r < bsize; ++r) {
        double aj = a_sorted[start + r];
        double bj = 1.0 - aj;
        int active_m = r + 1;
        for (int m = t; m < B + 1; m += blockDim.x) {
            double v = 0.0;
            if (m == 0) {
                v = aj * curr[0];
            } else if (m <= active_m) {
                v = aj * curr[m] + bj * curr[m - 1];
            }
            next[m] = v;
        }
        __syncthreads();
        double *tmp = curr;
        curr = next;
        next = tmp;
        __syncthreads();
    }

    int cp = (B + 1 < leaf_psz) ? (B + 1) : leaf_psz;
    for (int m = t; m < B + 1; m += blockDim.x) P[m] = curr[m];
    for (int m = t; m < leaf_psz; m += blockDim.x) {
        leaf[m] = (m < cp) ? curr[m] : 0.0;
    }
}

__global__ static void k_schoolbook_build(const double *child, int cps,
                                          double *parent, int pps, int nparents,
                                          int child_stride, int parent_stride) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)nparents * (size_t)pps;
    if (idx >= total) return;
    int p = (int)(idx / (size_t)pps);
    int m = (int)(idx % (size_t)pps);
    const double *L = child + (size_t)(2 * p) * (size_t)child_stride;
    const double *R = child + (size_t)(2 * p + 1) * (size_t)child_stride;
    int j_lo = m - (cps - 1);
    if (j_lo < 0) j_lo = 0;
    int j_hi = m;
    if (j_hi > cps - 1) j_hi = cps - 1;
    double sum = 0.0;
    for (int j = j_lo; j <= j_hi; ++j) sum += L[j] * R[m - j];
    parent[(size_t)p * (size_t)parent_stride + (size_t)m] = sum;
}

/* Shared-memory block kernel: one parent per block. */
__global__ static void k_schoolbook_build_smem_parent(const double *child, int cps,
                                                      double *parent, int pps, int nparents,
                                                      int child_stride, int parent_stride) {
    int p = blockIdx.x;
    if (p >= nparents) return;
    extern __shared__ double sh[];
    double *Lsh = sh;
    double *Rsh = sh + cps;

    const double *L = child + (size_t)(2 * p) * (size_t)child_stride;
    const double *R = child + (size_t)(2 * p + 1) * (size_t)child_stride;
    for (int i = threadIdx.x; i < cps; i += blockDim.x) {
        Lsh[i] = L[i];
        Rsh[i] = R[i];
    }
    __syncthreads();

    double *out = parent + (size_t)p * (size_t)parent_stride;
    for (int m = threadIdx.x; m < pps; m += blockDim.x) {
        int j_lo = m - (cps - 1);
        if (j_lo < 0) j_lo = 0;
        int j_hi = m;
        if (j_hi > cps - 1) j_hi = cps - 1;
        double sum = 0.0;
        for (int j = j_lo; j <= j_hi; ++j) sum += Lsh[j] * Rsh[m - j];
        out[m] = sum;
    }
}

/* Warp-batched kernel: one warp computes one parent pair. */
__global__ static void k_schoolbook_build_warp_batch(const double *child, int cps,
                                                     double *parent, int pps, int nparents,
                                                     int child_stride, int parent_stride) {
    constexpr int WARP = 32;
    int lane = threadIdx.x & (WARP - 1);
    int warp = threadIdx.x / WARP;
    int warps_per_block = blockDim.x / WARP;
    int p = blockIdx.x * warps_per_block + warp;
    if (p >= nparents) return;

    extern __shared__ double sh[];
    double *warp_sh = sh + (size_t)warp * (size_t)(2 * cps);
    double *Lsh = warp_sh;
    double *Rsh = warp_sh + cps;

    const double *L = child + (size_t)(2 * p) * (size_t)child_stride;
    const double *R = child + (size_t)(2 * p + 1) * (size_t)child_stride;
    for (int i = lane; i < cps; i += WARP) {
        Lsh[i] = L[i];
        Rsh[i] = R[i];
    }
    __syncwarp();

    double *out = parent + (size_t)p * (size_t)parent_stride;
    for (int m = lane; m < pps; m += WARP) {
        int j_lo = m - (cps - 1);
        if (j_lo < 0) j_lo = 0;
        int j_hi = m;
        if (j_hi > cps - 1) j_hi = cps - 1;
        double sum = 0.0;
        for (int j = j_lo; j <= j_hi; ++j) sum += Lsh[j] * Rsh[m - j];
        out[m] = sum;
    }
}

__global__ static void k_pack_level_to_fft(const double *src, int src_stride, int batch,
                                           double *dst, int fft_n, int copy_len) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)batch * (size_t)fft_n;
    if (idx >= total) return;
    int b = (int)(idx / (size_t)fft_n);
    int m = (int)(idx % (size_t)fft_n);
    const double *s = src + (size_t)b * (size_t)src_stride;
    dst[idx] = (m < copy_len) ? s[m] : 0.0;
}

__global__ static void k_pairwise_mul(const cufftDoubleComplex *child_spec, int cn,
                                      cufftDoubleComplex *parent_spec, int nparents) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)nparents * (size_t)cn;
    if (idx >= total) return;
    int p = (int)(idx / (size_t)cn);
    int f = (int)(idx % (size_t)cn);
    cufftDoubleComplex a = child_spec[(size_t)(2 * p) * (size_t)cn + (size_t)f];
    cufftDoubleComplex b = child_spec[(size_t)(2 * p + 1) * (size_t)cn + (size_t)f];
    cufftDoubleComplex o;
    o.x = a.x * b.x - a.y * b.y;
    o.y = a.x * b.y + a.y * b.x;
    parent_spec[idx] = o;
}

__global__ static void k_unpack_fft_to_level(const double *src, int fft_n, double inv_fft_n,
                                             int pps, int batch, double *dst) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)batch * (size_t)pps;
    if (idx >= total) return;
    int b = (int)(idx / (size_t)pps);
    int m = (int)(idx % (size_t)pps);
    const double *in = src + (size_t)b * (size_t)fft_n;
    dst[idx] = (m < fft_n) ? in[m] * inv_fft_n : 0.0;
}

/* In-place scale + zero-pad kernel for FFT-stride layout.
 * After cuFFT Z2D writes fft_n elements per batch into a buffer with stride
 * fft_stride (>= fft_n), this kernel:
 *   - scales elements [0, valid_len) by inv_fft_n
 *   - zeros elements [valid_len, fft_stride)
 * Each thread handles one element within the fft_stride-wide slot.
 * Total threads = batch * fft_stride. */
__global__ static void k_scale_zero_pad(double *data, int fft_stride, int valid_len,
                                        double inv_fft_n, int batch) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)batch * (size_t)fft_stride;
    if (idx >= total) return;
    int b = (int)(idx / (size_t)fft_stride);
    int m = (int)(idx % (size_t)fft_stride);
    (void)b;
    if (m < valid_len) {
        data[idx] *= inv_fft_n;
    } else {
        data[idx] = 0.0;
    }
}

__global__ static void k_wrap_build(double *parent, int pps, int nparents,
                                    const double *child, int cps, int conv_len,
                                    int fft_n, int wrap_m,
                                    int parent_stride, int child_stride) {
    int p = blockIdx.x;
    if (p >= nparents || threadIdx.x != 0) return;
    double *out = parent + (size_t)p * (size_t)parent_stride;
    const double *L = child + (size_t)(2 * p) * (size_t)child_stride;
    const double *R = child + (size_t)(2 * p + 1) * (size_t)child_stride;
    int da = cps - 1;
    int db = cps - 1;
    for (int i = 0; i <= wrap_m; ++i) {
        int pos = fft_n + i;
        if (pos >= conv_len) break;
        double high = 0.0;
        int j_lo = pos - db;
        if (j_lo < 0) j_lo = 0;
        int j_hi = da;
        if (j_hi > pos) j_hi = pos;
        for (int j = j_lo; j <= j_hi; ++j) {
            high += L[j] * R[pos - j];
        }
        if (i < pps) out[i] -= high;
        if (pos < pps) out[pos] = high;
    }
}

__global__ static void k_paired_corr_freq(const cufftDoubleComplex *g_hat,
                                          const cufftDoubleComplex *cached_child_spec,
                                          int cn, int nparents,
                                          cufftDoubleComplex *child_out_spec) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)nparents * (size_t)cn;
    if (idx >= total) return;
    int p = (int)(idx / (size_t)cn);
    int f = (int)(idx % (size_t)cn);

    cufftDoubleComplex g = g_hat[idx];
    cufftDoubleComplex specL = cached_child_spec[(size_t)(2 * p) * (size_t)cn + (size_t)f];
    cufftDoubleComplex specR = cached_child_spec[(size_t)(2 * p + 1) * (size_t)cn + (size_t)f];

    cufftDoubleComplex out_left;
    out_left.x = g.x * specR.x + g.y * specR.y;
    out_left.y = g.y * specR.x - g.x * specR.y;

    cufftDoubleComplex out_right;
    out_right.x = g.x * specL.x + g.y * specL.y;
    out_right.y = g.y * specL.x - g.x * specL.y;

    child_out_spec[(size_t)(2 * p) * (size_t)cn + (size_t)f] = out_left;
    child_out_spec[(size_t)(2 * p + 1) * (size_t)cn + (size_t)f] = out_right;
}

__global__ static void k_unpack_corr_children(const double *ifft_out, int fft_n, double inv_fft_n,
                                              int child_gsz, int len_out, int nparents, double *g_child) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)(2 * nparents) * (size_t)len_out;
    if (idx >= total) return;
    int c = (int)(idx / (size_t)len_out);
    int m = (int)(idx % (size_t)len_out);
    const double *in = ifft_out + (size_t)c * (size_t)fft_n;
    g_child[(size_t)c * (size_t)child_gsz + (size_t)m] = (m < fft_n) ? in[m] * inv_fft_n : 0.0;
}

__global__ static void k_wrap_corr_pair(double *g_child, int child_gsz, int nparents,
                                        const double *g_parent, int parent_gsz, int len_g,
                                        const double *child_poly, int cps, int len_P,
                                        int len_out,
                                        int fft_n, int wrap_m,
                                        int child_g_stride, int parent_g_stride,
                                        int child_poly_stride) {
    int p = blockIdx.x;
    if (p >= nparents || threadIdx.x != 0) return;

    double *outL = g_child + (size_t)(2 * p) * (size_t)child_g_stride;
    double *outR = g_child + (size_t)(2 * p + 1) * (size_t)child_g_stride;
    const double *gp = g_parent + (size_t)p * (size_t)parent_g_stride;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)child_poly_stride;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)child_poly_stride;

    int conv_len = len_g + len_P - 1;

    for (int i = 0; i <= wrap_m; ++i) {
        int pos = fft_n + i;
        if (pos >= conv_len) break;
        double highL = 0.0;
        double highR = 0.0;
        int j_max = len_g - pos;
        if (j_max > len_P) j_max = len_P;
        for (int j = 0; j < j_max; ++j) {
            highL += PR[j] * gp[pos + j];
            highR += PL[j] * gp[pos + j];
        }
        if (i < len_out) outL[i] -= highL;
        if (i < len_out) outR[i] -= highR;
        if (pos < len_out) outL[pos] = highL;
        if (pos < len_out) outR[pos] = highR;
    }

    int m_start = fft_n - len_P + 1;
    if (m_start < 0) m_start = 0;
    for (int m = m_start; m < len_out && m < fft_n; ++m) {
        double aliasL = 0.0;
        double aliasR = 0.0;
        int j_lo = fft_n - m;
        for (int j = j_lo; j < len_P; ++j) {
            int g_idx = (m + j) - fft_n;
            if (g_idx >= 0 && g_idx < len_g) {
                aliasL += PR[j] * gp[g_idx];
                aliasR += PL[j] * gp[g_idx];
            }
        }
        outL[m] -= aliasL;
        outR[m] -= aliasR;
    }
}

__global__ static void k_schoolbook_corr_pair(const double *g_parent, int parent_gsz,
                                              int len_g,
                                              const double *child_poly, int cps, int len_P,
                                              double *g_child, int child_gsz, int len_out, int nparents,
                                              int parent_g_stride, int child_poly_stride, int child_g_stride) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)nparents * (size_t)len_out;
    if (idx >= total) return;
    int p = (int)(idx / (size_t)len_out);
    int m = (int)(idx % (size_t)len_out);
    const double *gp = g_parent + (size_t)p * (size_t)parent_g_stride;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)child_poly_stride;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)child_poly_stride;
    double sumL = 0.0;
    double sumR = 0.0;
    int j_max = len_g - m;
    if (j_max > len_P) j_max = len_P;
    for (int j = 0; j < j_max; ++j) {
        sumL += PR[j] * gp[m + j];
        sumR += PL[j] * gp[m + j];
    }
    g_child[(size_t)(2 * p) * (size_t)child_g_stride + (size_t)m] = sumL;
    g_child[(size_t)(2 * p + 1) * (size_t)child_g_stride + (size_t)m] = sumR;
}

/* Shared-memory block kernel: one parent per block for paired correlation. */
__global__ static void k_schoolbook_corr_pair_smem_parent(const double *g_parent, int parent_gsz,
                                                           int len_g,
                                                           const double *child_poly, int cps, int len_P,
                                                           double *g_child, int child_gsz,
                                                           int len_out, int nparents,
                                                           int parent_g_stride, int child_poly_stride,
                                                           int child_g_stride) {
    int p = blockIdx.x;
    if (p >= nparents) return;

    extern __shared__ double sh[];
    double *gp_sh = sh;
    double *pl_sh = gp_sh + len_g;
    double *pr_sh = pl_sh + len_P;

    const double *gp = g_parent + (size_t)p * (size_t)parent_g_stride;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)child_poly_stride;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)child_poly_stride;
    for (int i = threadIdx.x; i < len_g; i += blockDim.x) gp_sh[i] = gp[i];
    for (int i = threadIdx.x; i < len_P; i += blockDim.x) {
        pl_sh[i] = PL[i];
        pr_sh[i] = PR[i];
    }
    __syncthreads();

    double *outL = g_child + (size_t)(2 * p) * (size_t)child_g_stride;
    double *outR = g_child + (size_t)(2 * p + 1) * (size_t)child_g_stride;
    for (int m = threadIdx.x; m < len_out; m += blockDim.x) {
        int j_max = len_g - m;
        if (j_max > len_P) j_max = len_P;
        double sumL = 0.0;
        double sumR = 0.0;
        for (int j = 0; j < j_max; ++j) {
            double gv = gp_sh[m + j];
            sumL += pr_sh[j] * gv;
            sumR += pl_sh[j] * gv;
        }
        outL[m] = sumL;
        outR[m] = sumR;
    }
}

/* Warp-batched kernel: one warp computes one parent pair. */
__global__ static void k_schoolbook_corr_pair_warp_batch(const double *g_parent, int parent_gsz,
                                                          int len_g,
                                                          const double *child_poly, int cps, int len_P,
                                                          double *g_child, int child_gsz,
                                                          int len_out, int nparents,
                                                          int parent_g_stride, int child_poly_stride,
                                                          int child_g_stride) {
    constexpr int WARP = 32;
    int lane = threadIdx.x & (WARP - 1);
    int warp = threadIdx.x / WARP;
    int warps_per_block = blockDim.x / WARP;
    int p = blockIdx.x * warps_per_block + warp;
    if (p >= nparents) return;

    int per_warp = len_g + 2 * len_P;
    extern __shared__ double sh[];
    double *warp_sh = sh + (size_t)warp * (size_t)per_warp;
    double *gp_sh = warp_sh;
    double *pl_sh = gp_sh + len_g;
    double *pr_sh = pl_sh + len_P;

    const double *gp = g_parent + (size_t)p * (size_t)parent_g_stride;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)child_poly_stride;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)child_poly_stride;
    for (int i = lane; i < len_g; i += WARP) gp_sh[i] = gp[i];
    for (int i = lane; i < len_P; i += WARP) {
        pl_sh[i] = PL[i];
        pr_sh[i] = PR[i];
    }
    __syncwarp();

    double *outL = g_child + (size_t)(2 * p) * (size_t)child_g_stride;
    double *outR = g_child + (size_t)(2 * p + 1) * (size_t)child_g_stride;
    for (int m = lane; m < len_out; m += WARP) {
        int j_max = len_g - m;
        if (j_max > len_P) j_max = len_P;
        double sumL = 0.0;
        double sumR = 0.0;
        for (int j = 0; j < j_max; ++j) {
            double gv = gp_sh[m + j];
            sumL += pr_sh[j] * gv;
            sumR += pl_sh[j] * gv;
        }
        outL[m] = sumL;
        outR[m] = sumR;
    }
}

__global__ static void k_leaf_extract(const double *a_sorted, int n, int B, int nblocks,
                                      const double *block_prods, const double *g_leaf,
                                      int leaf_psz, int g_need, int k,
                                      double *inner_sorted) {
    int b = blockIdx.x;
    if (b >= nblocks) return;
    int start = b * B;
    int end = start + B;
    if (end > n) end = n;
    int bsize = end - start;
    const double *P_b = block_prods + (size_t)b * (size_t)(B + 1);
    const double *g_b = g_leaf + (size_t)b * (size_t)leaf_psz;
    int pk_g = g_need < bsize ? g_need : bsize;
    if (pk_g > k) pk_g = k;

    for (int t = threadIdx.x; t < bsize; t += blockDim.x) {
        int j = start + t;
        double aj = a_sorted[j];
        double bj = 1.0 - aj;
        double eq = 0.0;

        if (aj > 0.5) {
            double ia = 1.0 / aj;
            double c = -bj / aj;
            double q = P_b[0] * ia;
            eq = g_b[0] * q;
            for (int m = 1; m < pk_g; ++m) {
                q = c * q + P_b[m] * ia;
                eq += g_b[m] * q;
            }
        } else if (aj > 1e-15) {
            double ib = 1.0 / bj;
            double c = -aj / bj;
            double q = P_b[bsize] * ib;
            if (bsize - 1 < pk_g) eq += g_b[bsize - 1] * q;
            for (int m = bsize - 2; m >= 0; --m) {
                q = c * q + P_b[m + 1] * ib;
                if (m < pk_g) eq += g_b[m] * q;
            }
        } else {
            for (int m = 0; m < pk_g; ++m) eq += g_b[m] * P_b[m + 1];
        }
        inner_sorted[j] = eq;
    }
}

__global__ static void k_accumulate_equity(const double *inner_sorted,
                                           const double *a_sorted,
                                           const double *S_sorted,
                                           const int *sort_perm,
                                           int n, double weight, double inv_v,
                                           double *equity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int orig = sort_perm[i];
    double pw = weight * S_sorted[i] * a_sorted[i] * inv_v;
    if (!isfinite(pw)) pw = 0.0;
    atomicAdd(&equity[orig], pw * inner_sorted[i]);
}

__global__ static void k_accumulate_equity_scaled(const double *inner_sorted,
                                                  const double *a_sorted,
                                                  const double *S_sorted,
                                                  const int *sort_perm,
                                                  int n, const double *scale_ptr,
                                                  double *equity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int orig = sort_perm[i];
    double scale = scale_ptr[0];
    double pw = scale * S_sorted[i] * a_sorted[i];
    if (!isfinite(pw)) pw = 0.0;
    atomicAdd(&equity[orig], pw * inner_sorted[i]);
}

/* ── Single-kernel shared-memory engine for small n ──
 *
 * One block per Q-point. Entire tree lives in shared memory.
 * B=1: each player is a leaf, schoolbook at all levels.
 * No cuFFT, no HBM intermediates, no plan creation overhead.
 * Works for n up to ~2048 (shared memory limit).
 *
 * Shared memory layout:
 *   poly[0..total_poly-1] : tree polynomial data (all levels interleaved)
 *   g[0..total_g-1]       : g-vector data (two alternating buffers)
 */
__global__ static void k_icm_single_kernel(
        const double *S_sorted, const int *sort_perm, int n,
        int Q, const double *d_logv, const double *d_weights,
        const double *payout, int k,
        double *equity,
        /* Tree geometry (compile-time-like params) */
        int N, int L, const int *d_nn, const int *d_psz, const int *d_g_needed,
        const size_t *d_plev_off, int total_poly, int max_g) {

    int q = blockIdx.x;
    if (q >= Q) return;

    double logv = d_logv[q];
    double w = d_weights[q];
    if (w == 0.0) return;

    extern __shared__ double smem[];
    double *poly = smem;                        /* tree polynomial levels */
    double *g0   = smem + total_poly;           /* g buffer 0 */
    double *g1   = g0 + max_g;                  /* g buffer 1 */

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    /* ── 1. Set leaves: P[i] = [a[i], 1-a[i]] ── */
    int leaf_psz = d_psz[0];
    for (int i = tid; i < N; i += nthreads) {
        double *leaf = poly + d_plev_off[0] + (size_t)i * leaf_psz;
        if (i < n) {
            double arg = S_sorted[i] * logv;
            double ai = (arg < -700.0) ? 0.0 : exp(arg);
            leaf[0] = ai;
            if (leaf_psz > 1) leaf[1] = 1.0 - ai;
            for (int m = 2; m < leaf_psz; m++) leaf[m] = 0.0;
        } else {
            leaf[0] = 1.0;
            for (int m = 1; m < leaf_psz; m++) leaf[m] = 0.0;
        }
    }
    __syncthreads();

    /* ── 2. Tree build: schoolbook multiply bottom-up ── */
    for (int ell = 1; ell < L - 1; ell++) {
        int cps = d_psz[ell - 1];
        int pps = d_psz[ell];
        int nn_parent = d_nn[ell];
        double *child_base = poly + d_plev_off[ell - 1];
        double *parent_base = poly + d_plev_off[ell];

        for (int j = tid; j < nn_parent; j += nthreads) {
            double *Lc = child_base + (size_t)(2 * j) * cps;
            double *Rc = child_base + (size_t)(2 * j + 1) * cps;
            double *out = parent_base + (size_t)j * pps;
            /* Schoolbook: out[m] = sum_{i+j=m} Lc[i]*Rc[j], truncated to pps */
            for (int m = 0; m < pps; m++) {
                int j_lo = m - (cps - 1);
                if (j_lo < 0) j_lo = 0;
                int j_hi = m;
                if (j_hi > cps - 1) j_hi = cps - 1;
                double sum = 0.0;
                for (int jj = j_lo; jj <= j_hi; jj++)
                    sum += Lc[jj] * Rc[m - jj];
                out[m] = sum;
            }
        }
        __syncthreads();
    }

    /* ── 3. Set root g = payout ── */
    int top = L - 1;
    int root_gsz = d_psz[top];
    double *g_parent = g0;
    for (int m = tid; m < root_gsz; m += nthreads)
        g_parent[m] = (m < k) ? payout[m] : 0.0;
    __syncthreads();

    /* ── 4. Tree propagate: schoolbook correlate top-down ── */
    for (int ell = top; ell >= 1; ell--) {
        int cps = d_psz[ell - 1];
        int pgsz = d_psz[ell];
        int nn_parent = d_nn[ell];
        int out_needed = d_g_needed[ell - 1];
        int p_eff = cps;  /* for B=1 all-schoolbook, p_eff = cps */
        double *child_base = poly + d_plev_off[ell - 1];
        double *g_child = (g_parent == g0) ? g1 : g0;

        for (int j = tid; j < nn_parent; j += nthreads) {
            double *gp = g_parent + (size_t)j * pgsz;
            double *PL = child_base + (size_t)(2 * j) * cps;
            double *PR = child_base + (size_t)(2 * j + 1) * cps;
            double *gL = g_child + (size_t)(2 * j) * cps;
            double *gR = g_child + (size_t)(2 * j + 1) * cps;

            int len_g = pgsz;  /* use full g at this level */
            for (int m = 0; m < out_needed; m++) {
                double sumL = 0.0, sumR = 0.0;
                int j_max = len_g - m;
                if (j_max > p_eff) j_max = p_eff;
                for (int jj = 0; jj < j_max; jj++) {
                    double gv = gp[m + jj];
                    sumL += PR[jj] * gv;
                    sumR += PL[jj] * gv;
                }
                gL[m] = sumL;
                gR[m] = sumR;
            }
        }
        __syncthreads();
        g_parent = g_child;
    }

    /* ── 5. Extract inner[i] = g_leaf[i][0] and accumulate equity ── */
    double inv_v = exp(-logv);
    int leaf_gsz = d_psz[0];
    for (int i = tid; i < n; i += nthreads) {
        double inner = g_parent[(size_t)i * leaf_gsz];
        double arg = S_sorted[i] * logv;
        double ai = (arg < -700.0) ? 0.0 : exp(arg);
        double pw = w * S_sorted[i] * ai * inv_v;
        if (!isfinite(pw)) pw = 0.0;
        int orig = sort_perm[i];
        atomicAdd(&equity[orig], pw * inner);
    }
}

/* Maximum n for single-kernel engine (shared memory limit) */
static int single_kernel_max_n(int k) {
    /* Compute shared memory needed for n players with B=1 */
    /* poly: sum(nn[ell] * psz[ell]) for all levels */
    /* g: 2 * max_level_g_size */
    /* Conservative: each level stores ~2n doubles for poly, ~2n for g */
    /* With 228KB = 28672 doubles, and ~4n per level, max levels ~7 → n ~1024 */
    /* For k=n: total ≈ 2 * sum(nn[ell]*psz[ell]) ≈ 2 * n * L * 2 */
    /* Be conservative: limit to n=1024 */
    (void)k;
    const char *env = getenv("ICM_GPU_SINGLE_KERNEL_MAX_N");
    if (env && env[0]) {
        int v = atoi(env);
        if (v > 0) return v;
    }
    return 1024;
}

/* ── Q-batch kernels ── */

/* Block build for Q-batched execution.
 * Grid: q_batch * N_tree blocks. Each block handles one (qi, b) pair.
 * Layout: Q0's N_tree leaves, then Q1's N_tree leaves, etc.
 * a_ptrs[qi] points to the a_sorted for the qi-th Q-point.
 * leaf_stride = N_tree * leaf_psz (stride between Q-points in leaves buffer)
 * bp_stride = N_tree * (B+1)     (stride between Q-points in block_prods buffer)
 */
__global__ static void k_block_build_qbatch(
        const double * const *a_ptrs, int n, int B,
        int nblocks, int N_tree, int q_batch,
        int leaf_psz, double *leaves, size_t leaf_stride,
        double *block_prods, size_t bp_stride) {
    int global_b = blockIdx.x;
    int total_blocks = q_batch * N_tree;
    if (global_b >= total_blocks) return;
    int qi = global_b / N_tree;
    int b = global_b % N_tree;
    int t = threadIdx.x;

    double *leaf = leaves + (size_t)qi * leaf_stride + (size_t)b * (size_t)leaf_psz;
    double *P = block_prods + (size_t)qi * bp_stride + (size_t)b * (size_t)(B + 1);
    const double *a_sorted = a_ptrs[qi];
    extern __shared__ double sh[];
    double *curr = sh;
    double *next = sh + (B + 1);

    if (b >= nblocks) {
        for (int m = t; m < B + 1; m += blockDim.x) P[m] = 0.0;
        for (int m = t; m < leaf_psz; m += blockDim.x) leaf[m] = 0.0;
        if (t == 0) {
            P[0] = 1.0;
            leaf[0] = 1.0;
        }
        return;
    }

    int start = b * B;
    int end = start + B;
    if (end > n) end = n;
    int bsize = end - start;

    for (int m = t; m < B + 1; m += blockDim.x) {
        curr[m] = 0.0;
        next[m] = 0.0;
    }
    __syncthreads();
    if (t == 0) curr[0] = 1.0;
    __syncthreads();

    for (int r = 0; r < bsize; ++r) {
        double aj = a_sorted[start + r];
        double bj = 1.0 - aj;
        int active_m = r + 1;
        for (int m = t; m < B + 1; m += blockDim.x) {
            double v = 0.0;
            if (m == 0) {
                v = aj * curr[0];
            } else if (m <= active_m) {
                v = aj * curr[m] + bj * curr[m - 1];
            }
            next[m] = v;
        }
        __syncthreads();
        double *tmp = curr;
        curr = next;
        next = tmp;
        __syncthreads();
    }

    int cp = (B + 1 < leaf_psz) ? (B + 1) : leaf_psz;
    for (int m = t; m < B + 1; m += blockDim.x) P[m] = curr[m];
    for (int m = t; m < leaf_psz; m += blockDim.x) {
        leaf[m] = (m < cp) ? curr[m] : 0.0;
    }
}

/* Set root g for Q-batch: write payout into each Q-point's root g slot.
 * Grid: ceil(q_batch * root_gsz / 256) blocks. */
__global__ static void k_set_root_g_qbatch(double *g_root, int root_gsz,
                                            const double *payout, int k,
                                            int q_batch, size_t g_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = q_batch * root_gsz;
    if (idx >= total) return;
    int qi = idx / root_gsz;
    int i = idx % root_gsz;
    g_root[(size_t)qi * g_stride + (size_t)i] = (i < k) ? payout[i] : 0.0;
}

/* Leaf extract for Q-batch.
 * Grid: q_batch * nblocks blocks. Each block handles one (qi, b) pair. */
__global__ static void k_leaf_extract_qbatch(
        const double * const *a_ptrs, int n, int B, int nblocks,
        const double *block_prods, size_t bp_stride,
        const double *g_leaf, int leaf_psz, size_t leaf_g_stride,
        int g_need, int k, int q_batch,
        double *inner_sorted, size_t inner_stride) {
    int global_b = blockIdx.x;
    int total_blocks = q_batch * nblocks;
    if (global_b >= total_blocks) return;
    int qi = global_b / nblocks;
    int b = global_b % nblocks;

    int start = b * B;
    int end = start + B;
    if (end > n) end = n;
    int bsize = end - start;
    const double *a_sorted = a_ptrs[qi];
    const double *P_b = block_prods + (size_t)qi * bp_stride + (size_t)b * (size_t)(B + 1);
    const double *g_b = g_leaf + (size_t)qi * leaf_g_stride + (size_t)b * (size_t)leaf_psz;
    double *inner_out = inner_sorted + (size_t)qi * inner_stride;
    int pk_g = g_need < bsize ? g_need : bsize;
    if (pk_g > k) pk_g = k;

    for (int t = threadIdx.x; t < bsize; t += blockDim.x) {
        int j = start + t;
        double aj = a_sorted[j];
        double bj = 1.0 - aj;
        double eq = 0.0;

        if (aj > 0.5) {
            double ia = 1.0 / aj;
            double c = -bj / aj;
            double q = P_b[0] * ia;
            eq = g_b[0] * q;
            for (int m = 1; m < pk_g; ++m) {
                q = c * q + P_b[m] * ia;
                eq += g_b[m] * q;
            }
        } else if (aj > 1e-15) {
            double ib = 1.0 / bj;
            double c = -aj / bj;
            double q = P_b[bsize] * ib;
            if (bsize - 1 < pk_g) eq += g_b[bsize - 1] * q;
            for (int m = bsize - 2; m >= 0; --m) {
                q = c * q + P_b[m + 1] * ib;
                if (m < pk_g) eq += g_b[m] * q;
            }
        } else {
            for (int m = 0; m < pk_g; ++m) eq += g_b[m] * P_b[m + 1];
        }
        inner_out[j] = eq;
    }
}

/* Accumulate equity for Q-batch: processes one (qi, i) pair per thread. */
__global__ static void k_accumulate_equity_qbatch(
        const double *inner_sorted, size_t inner_stride,
        const double * const *a_ptrs,
        const double *S_sorted,
        const int *sort_perm,
        int n, const double *weights, const double *inv_vs,
        int q_batch, double *equity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = q_batch * n;
    if (idx >= total) return;
    int qi = idx / n;
    int i = idx % n;
    int orig = sort_perm[i];
    double w = weights[qi];
    double inv_v = inv_vs[qi];
    double pw = w * S_sorted[i] * a_ptrs[qi][i] * inv_v;
    if (!isfinite(pw)) pw = 0.0;
    atomicAdd(&equity[orig], pw * inner_sorted[(size_t)qi * inner_stride + (size_t)i]);
}

/* ── End Q-batch kernels ── */

struct QP {
    double logv;
    double w;
};

static double log_Phi(double y) {
    if (y >= 0) return log1p(-erfc(y / sqrt(2.0)) / 2.0);
    return log(erfc(-y / sqrt(2.0)) / 2.0);
}

static void make_nodes(int Q, double Smax, std::vector<QP> &pts) {
    pts.resize(Q);
    double y_lo = -7.7;
    double y_hi = sqrt(2.0) * sqrt(log(Smax) + 25.0);
    if (y_hi < 6.5) y_hi = 6.5;
    double h = (Q > 1) ? (y_hi - y_lo) / (Q - 1) : 0.0;
    for (int q = 0; q < Q; ++q) {
        double y = y_lo + q * h;
        pts[q].logv = log_Phi(y);
        pts[q].w = h * exp(-0.5 * y * y) / sqrt(2.0 * M_PI);
    }
}

/* Create cuFFT plan with custom real-side stride (real_dist).
 * For D2Z: reads from real data at stride real_dist, writes complex at stride cn.
 * For Z2D: reads complex at stride cn, writes real data at stride real_dist.
 * When real_dist == n, this is equivalent to the old contiguous layout. */
static bool create_cufft_plan(cufftHandle *plan, int n, int batch, bool r2c, int real_dist = 0) {
    if (real_dist <= 0) real_dist = n;
    if (!CUFFT_OK(cufftCreate(plan))) return false;
    int rank = 1;
    int n_arr[1] = {n};
    int cn = n / 2 + 1;
    size_t work_size = 0;
    if (r2c) {
        int ie[1] = {real_dist};
        int oe[1] = {cn};
        if (!CUFFT_OK(cufftMakePlanMany(*plan, rank, n_arr,
                                        ie, 1, real_dist,
                                        oe, 1, cn,
                                        CUFFT_D2Z, batch, &work_size))) return false;
    } else {
        int ie[1] = {cn};
        int oe[1] = {real_dist};
        if (!CUFFT_OK(cufftMakePlanMany(*plan, rank, n_arr,
                                        ie, 1, cn,
                                        oe, 1, real_dist,
                                        CUFFT_Z2D, batch, &work_size))) return false;
    }
    return true;
}

static bool maybe_init_mem_pool(GpuPlan *plan) {
    if (!plan->opts.enable_graphs && plan->opts.memory_strategy != 2 && plan->opts.memory_strategy != 3) {
        plan->use_async_pool = false;
        return true;
    }
    cudaMemPool_t pool = nullptr;
    if (!CUDA_OK(cudaDeviceGetDefaultMemPool(&pool, g_cuda_device))) return false;
    uint64_t threshold = GPU_VRAM_BYTES;
    if (!CUDA_OK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold))) return false;
    plan->mem_pool = pool;
    plan->use_async_pool = true;
    return true;
}

static bool allocate_level_buffers(GpuPlan *plan, int ell, const std::vector<int> &fft_sizes) {
    (void)fft_sizes;
    if (ell <= 0 || ell >= plan->L) return true;
    auto &lp = plan->levels[ell];
    if (!lp.use_fft) return true;
    /* Fused levels still need cuFFT fallback infrastructure (when cuFFTDx unavailable) */

    int fft_n = lp.fft_n;
    int cn = lp.cn;
    int child_batch = plan->nn[ell - 1];
    int parent_batch = plan->nn[ell];
    int qb = plan->q_batch;

    /* Build FFT: cuFFT reads directly from poly_levels[ell-1] at fft_stride[ell-1]
     * and writes directly to poly_levels[ell] at fft_stride[ell].
     * The real_dist parameter sets the batch pitch on the real side.
     * Spectral buffers (spec_in, spec_mid) remain contiguous at cn stride. */
    int child_stride = plan->fft_stride[ell - 1];
    int parent_stride = plan->fft_stride[ell];

    auto &b = plan->build_fft[ell];
    b.fft_n = fft_n;
    b.cn = cn;
    b.batch_fwd = qb * child_batch;
    b.batch_inv = qb * parent_batch;
    b.real_in = nullptr;   /* cuFFT reads poly_levels directly */
    b.spec_in = plan->shared_build_work.spec_in;
    b.spec_mid = plan->shared_build_work.spec_mid;
    b.real_out = nullptr;  /* cuFFT writes poly_levels directly */
    if (!create_cufft_plan(&b.plan_fwd, fft_n, qb * child_batch, true, child_stride)) return false;
    if (!create_cufft_plan(&b.plan_inv, fft_n, qb * parent_batch, false, parent_stride)) return false;
    if (!CUFFT_OK(cufftSetStream(b.plan_fwd, plan->stream_compute))) return false;
    if (!CUFFT_OK(cufftSetStream(b.plan_inv, plan->stream_compute))) return false;

    /* Corr FFT: forward reads g_levels[ell] at parent_stride,
     * inverse writes g_levels[ell-1] at child_stride. */
    auto &c = plan->corr_fft[ell];
    c.fft_n = fft_n;
    c.cn = cn;
    c.batch_fwd = qb * parent_batch;
    c.batch_inv = qb * 2 * parent_batch;
    c.real_in = nullptr;   /* cuFFT reads g_levels directly */
    c.spec_in = plan->shared_corr_work.spec_in;
    c.spec_mid = plan->shared_corr_work.spec_mid;
    c.real_out = nullptr;  /* cuFFT writes g_levels directly */
    if (!create_cufft_plan(&c.plan_fwd, fft_n, qb * parent_batch, true, parent_stride)) return false;
    if (!create_cufft_plan(&c.plan_inv, fft_n, qb * 2 * parent_batch, false, child_stride)) return false;
    if (!CUFFT_OK(cufftSetStream(c.plan_fwd, plan->stream_compute))) return false;
    if (!CUFFT_OK(cufftSetStream(c.plan_inv, plan->stream_compute))) return false;

    /* FFT cache: allocated by arena or by alloc_device if arena not used.
     * Skip if already set (arena mode). */
    if (lp.cache_fft && !plan->d_fft_cache[ell]) {
        size_t bytes_cache = (size_t)qb * (size_t)child_batch * (size_t)cn * sizeof(cufftDoubleComplex);
        if (!alloc_device(plan, (void **)&plan->d_fft_cache[ell], bytes_cache, plan->stream_compute)) return false;
    }
    return true;
}

static bool choose_uncached_levels(GpuPlan *plan) {
    plan->uncached_level.assign(plan->L, 0);
    if (plan->opts.memory_strategy != 3) return true;

    auto uncache_lowest = [&](int tier, int count, int *counter) {
        if (count <= 0) return;
        for (int ell = 1; ell < plan->L - 1 && count > 0; ++ell) {
            if (!plan->levels[ell].cache_fft) continue;
            if (plan->levels[ell].tier != tier) continue;
            plan->uncached_level[ell] = 1;
            plan->levels[ell].cache_fft = 0;
            (*counter)++;
            --count;
        }
    };

    if (plan->opts.force_uncached_fused_levels >= 0 || plan->opts.force_uncached_cufft_levels >= 0) {
        uncache_lowest(GPU_TIER_FUSED, std::max(0, plan->opts.force_uncached_fused_levels),
                       &plan->uncached_fused_levels);
        uncache_lowest(GPU_TIER_CUFFT, std::max(0, plan->opts.force_uncached_cufft_levels),
                       &plan->uncached_cufft_levels);
        return true;
    }

    size_t total = 0;
    for (int ell = 1; ell < plan->L - 1; ++ell) {
        if (!plan->levels[ell].cache_fft) continue;
        int nchild = plan->nn[ell - 1];
        int cn = plan->levels[ell].cn;
        total += (size_t)nchild * (size_t)cn * sizeof(cufftDoubleComplex);
    }
    double cache_budget_frac = 0.35;
    const char *budget_env = getenv("ICM_GPU_CACHE_BUDGET_FRAC");
    if (budget_env && budget_env[0]) {
        double x = atof(budget_env);
        if (x > 0.05 && x < 0.95) cache_budget_frac = x;
    }
    size_t cache_budget = (size_t)((double)GPU_VRAM_BYTES * cache_budget_frac);
    if (total <= cache_budget) return true;

    for (int ell = 1; ell < plan->L - 1; ++ell) {
        if (!plan->levels[ell].cache_fft) continue;
        if (plan->levels[ell].tier == GPU_TIER_FUSED) {
            plan->uncached_level[ell] = 1;
            plan->levels[ell].cache_fft = 0;
            plan->uncached_fused_levels++;
            int nchild = plan->nn[ell - 1];
            int cn = plan->levels[ell].cn;
            total -= (size_t)nchild * (size_t)cn * sizeof(cufftDoubleComplex);
            if (total <= cache_budget) return true;
        }
    }
    for (int ell = 1; ell < plan->L - 1; ++ell) {
        if (!plan->levels[ell].cache_fft) continue;
        if (plan->levels[ell].tier == GPU_TIER_CUFFT) {
            plan->uncached_level[ell] = 1;
            plan->levels[ell].cache_fft = 0;
            plan->uncached_cufft_levels++;
            int nchild = plan->nn[ell - 1];
            int cn = plan->levels[ell].cn;
            total -= (size_t)nchild * (size_t)cn * sizeof(cufftDoubleComplex);
            if (total <= cache_budget) return true;
        }
    }
    return true;
}

static bool build_plan_metadata(GpuPlan *plan) {
    if (plan->opts.use_cufftdx) {
        g_runtime_fused_max_conv_len = fused_max_conv_len_runtime();
    } else {
        g_runtime_fused_max_conv_len = 0;
    }

    std::vector<int> smooth;
    int max_smooth = std::max(1 << 20, 2 * plan->k + 8);
    build_smooth_table(max_smooth, smooth);
    plan->k_pad = best_k_pad_gpu(plan->k, smooth);
    plan->B = gpu_select_best_B_est(plan->n, plan->k_pad, smooth);
    const char *force_b_env = getenv("ICM_GPU_FORCE_B");
    if (force_b_env && force_b_env[0]) {
        int fb = atoi(force_b_env);
        if (fb > 0 && fb <= plan->n && fb <= plan->k_pad) {
            plan->B = fb;
        }
    }
    plan->engine = gpu_select_engine_est(plan->n, plan->k_pad, plan->B, smooth);
    if (plan->engine == GPU_ENGINE_LINEAR) {
        /* Keep hybrid path as implementation baseline, but preserve dispatch metadata. */
        plan->engine = GPU_ENGINE_HYBRID;
    }

    plan->nblocks = (plan->n + plan->B - 1) / plan->B;
    build_tree_geometry(plan->nblocks, plan->B, plan->k_pad, plan->B,
                        plan->nn, plan->psz, plan->plev_off,
                        plan->g_needed, plan->below_sat, plan->n_real,
                        plan->N_tree, plan->L);

    const char *force_tier_env = getenv("ICM_GPU_FORCE_TIER");
    int force_tier_mode = 0; /* 0=auto, 1=fft, 2=schoolbook, 3=fused */
    if (force_tier_env && force_tier_env[0]) {
        if (strcmp(force_tier_env, "fft") == 0) force_tier_mode = 1;
        else if (strcmp(force_tier_env, "schoolbook") == 0) force_tier_mode = 2;
        else if (strcmp(force_tier_env, "fused") == 0) force_tier_mode = 3;
    }

    plan->levels.assign(plan->L, GpuLevelPlan{});
    int debug_plan = 0;
    const char *dbg_env = getenv("ICM_GPU_DEBUG_PLAN");
    if (dbg_env && dbg_env[0] && atoi(dbg_env) != 0) debug_plan = 1;
    if (debug_plan) {
        fprintf(stderr, "gpu_plan n=%d k=%d k_pad=%d B=%d L=%d nblocks=%d q_batch=%d\n",
                plan->n, plan->k, plan->k_pad, plan->B, plan->L, plan->nblocks, plan->q_batch);
    }
    for (int ell = 1; ell < plan->L; ++ell) {
        int cps = plan->psz[ell - 1];
        int pgsz = plan->psz[ell];
        int nparents = plan->nn[ell];
        int is_below = plan->below_sat[ell];
        int p_eff = is_below ? (cps / 2 + 1) : cps;
        int out_needed = plan->g_needed[ell - 1];
        int g_eff_needed = out_needed + p_eff - 1;
        int g_eff_max = is_below ? (cps + cps / 2) : pgsz;
        int g_eff = std::min(g_eff_needed, g_eff_max);
        int conv_build = is_below ? (2 * (cps / 2)) : (2 * cps - 1);
        int conv_corr = g_eff + p_eff - 1;
        int d_eff = is_below ? (cps / 2) : (cps - 1);
        double wrap_scale = wrap_serial_penalty_gpu(nparents);

        int bfn = 0, bwm = 0;
        best_fft_config_gpu(conv_build, 0, wrap_scale, &bfn, &bwm);
        double fft_build = estimate_cufft_pipeline_ns(bfn)
            + (double)(bwm + 1) * (double)(bwm + 1) * tree_school_ns_per_fma() * wrap_scale;
        double school_build = (double)(d_eff + 1) * (double)(d_eff + 1) * tree_school_ns_per_fma();
        if (debug_plan) {
            fprintf(stderr,
                    "    build_choice ell=%d nparents=%d wrap_scale=%.2f bfn=%d bwm=%d fft_build=%.3fms school_build=%.3fms\n",
                    ell, nparents, wrap_scale, bfn, bwm, fft_build / 1e6, school_build / 1e6);
        }

        int use_fft = (fft_build < school_build);
        int fft_n = bfn;
        int build_wrap_m = bwm;
        int corr_wrap_m = 0;
        int cache_fft = 0;
        int tier = GPU_TIER_SCHOOLBOOK;

        if (use_fft) {
            int cfn = 0, cwm = 0;
            best_fft_config_gpu(conv_corr, p_eff, wrap_scale, &cfn, &cwm);
            fft_n = cfn;
            build_wrap_m = (cfn >= conv_build) ? 0 : (conv_build - cfn);
            corr_wrap_m = cwm;
            cache_fft = 0;

            if (ell < plan->L - 1) {
                int jfn = 0, jbm = 0, jcm = 0;
                double joint_cost = best_fft_config_joint_gpu(conv_build, conv_corr, p_eff, wrap_scale, &jfn, &jbm, &jcm);
                double indep_cost = fft_build
                    + estimate_cufft_pipeline_ns(cfn) * GPU_INDEP_PAIR_RATIO
                    + 2.0 * ((double)(cwm + 1) * (double)(cwm + 1) + (double)cwm * (double)p_eff)
                        * tree_school_ns_per_fma() * wrap_scale;
                if (joint_cost < indep_cost) {
                    fft_n = jfn;
                    build_wrap_m = jbm;
                    corr_wrap_m = jcm;
                    cache_fft = 1;
                }
            }

            tier = pick_tier_for_fft_len(fft_n, conv_build);
        }

        if (force_tier_mode == 1) {
            use_fft = 1;
            tier = GPU_TIER_CUFFT;
        } else if (force_tier_mode == 2) {
            use_fft = 0;
            tier = GPU_TIER_SCHOOLBOOK;
            cache_fft = 0;
        } else if (force_tier_mode == 3) {
            use_fft = 1;
            tier = GPU_TIER_FUSED;
            cache_fft = 0;
            int ffn = 0, fbm = 0, fcm = 0;
            best_fft_config_joint_gpu(conv_build, conv_corr, p_eff, wrap_scale, &ffn, &fbm, &fcm);
            if (ffn > 0) {
                fft_n = ffn;
                build_wrap_m = fbm;
                corr_wrap_m = fcm;
            }
        }

        auto &lp = plan->levels[ell];
        lp.ell = ell;
        lp.tier = tier;
        lp.use_fft = use_fft;
        lp.cache_fft = (use_fft && cache_fft && ell < plan->L - 1 && tier != GPU_TIER_FUSED);
        lp.fft_n = fft_n;
        lp.cn = fft_n / 2 + 1;
        lp.p_eff = p_eff;
        lp.out_needed = out_needed;
        lp.g_eff = g_eff;
        lp.build_conv = conv_build;
        lp.corr_conv = conv_corr;
        lp.build_wrap_m = build_wrap_m;
        lp.corr_wrap_m = corr_wrap_m;
        if (debug_plan) {
            fprintf(stderr,
                    "  ell=%d cps=%d pgsz=%d p_eff=%d out=%d g_eff=%d build_conv=%d corr_conv=%d fft_n=%d bwm=%d cwm=%d tier=%d cache=%d use_fft=%d\n",
                    ell, cps, pgsz, p_eff, out_needed, g_eff, conv_build, conv_corr,
                    fft_n, build_wrap_m, corr_wrap_m, tier, lp.cache_fft, use_fft);
        }
    }
    choose_uncached_levels(plan);

    /* ── Compute fft_stride: the per-level memory stride for poly/g levels ──
     * For each level ell, fft_stride[ell] must be large enough so that cuFFT
     * at any referencing level can use it as idist/odist directly.
     *
     * poly_levels[ell] is read by:
     *   - build forward D2Z at level ell+1  (needs stride >= fft_n[ell+1])
     *   - corr forward D2Z at level ell+1   (same fft_n due to joint optimization)
     * poly_levels[ell] is written by:
     *   - build inverse Z2D at level ell     (needs stride >= fft_n[ell])
     *
     * g_levels[ell] is read by:
     *   - corr forward D2Z at level ell      (needs stride >= fft_n[ell])
     * g_levels[ell] is written by:
     *   - corr inverse Z2D at level ell+1    (needs stride >= fft_n[ell+1])
     *
     * So: fft_stride[ell] = max(psz[ell], fft_n[ell] if use_fft, fft_n[ell+1] if use_fft)
     * For schoolbook-only levels with no adjacent cuFFT levels: fft_stride = psz.
     */
    plan->fft_stride.assign(plan->L, 0);
    for (int ell = 0; ell < plan->L; ++ell) {
        int s = plan->psz[ell];
        /* Level ell's own cuFFT (build/corr at level ell) */
        if (ell >= 1 && plan->levels[ell].use_fft &&
            plan->levels[ell].tier != GPU_TIER_SCHOOLBOOK) {
            s = std::max(s, plan->levels[ell].fft_n);
        }
        /* Level ell+1's cuFFT reads poly_levels[ell] as children / writes g_levels[ell] */
        if (ell + 1 < plan->L && plan->levels[ell + 1].use_fft &&
            plan->levels[ell + 1].tier != GPU_TIER_SCHOOLBOOK) {
            s = std::max(s, plan->levels[ell + 1].fft_n);
        }
        plan->fft_stride[ell] = s;
    }
    if (debug_plan) {
        for (int ell = 0; ell < plan->L; ++ell) {
            fprintf(stderr, "  fft_stride[%d] = %d  (psz=%d)\n",
                    ell, plan->fft_stride[ell], plan->psz[ell]);
        }
    }

    return true;
}

static bool device_sort_players(GpuPlan *plan) {
    std::vector<std::pair<double, int>> pairs(plan->n);
    for (int i = 0; i < plan->n; ++i) pairs[i] = {plan->S_sorted[i], i};
    std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
        if (a.first > b.first) return true;
        if (a.first < b.first) return false;
        return a.second < b.second;
    });
    std::vector<double> sorted(plan->n);
    std::vector<int> perm(plan->n), inv(plan->n);
    for (int i = 0; i < plan->n; ++i) {
        sorted[i] = pairs[i].first;
        perm[i] = pairs[i].second;
        inv[pairs[i].second] = i;
    }
    plan->S_sorted.swap(sorted);
    plan->sort_perm.swap(perm);
    plan->inv_perm.swap(inv);

    /* Touch CUB path once so the production path can switch to pure device sort
     * without API changes; this keeps the standard pipeline dependency in place. */
    double *d_keys_in = nullptr;
    double *d_keys_out = nullptr;
    int *d_vals_in = nullptr;
    int *d_vals_out = nullptr;
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    bool ok = true;
    if (!CUDA_OK(cudaMalloc(&d_keys_in, plan->n * sizeof(double)))) ok = false;
    if (!CUDA_OK(cudaMalloc(&d_keys_out, plan->n * sizeof(double)))) ok = false;
    if (!CUDA_OK(cudaMalloc(&d_vals_in, plan->n * sizeof(int)))) ok = false;
    if (!CUDA_OK(cudaMalloc(&d_vals_out, plan->n * sizeof(int)))) ok = false;
    if (ok) {
        std::vector<int> seq(plan->n);
        std::iota(seq.begin(), seq.end(), 0);
        if (!CUDA_OK(cudaMemcpy(d_keys_in, plan->S_sorted.data(), plan->n * sizeof(double), cudaMemcpyHostToDevice))) ok = false;
        if (!CUDA_OK(cudaMemcpy(d_vals_in, seq.data(), plan->n * sizeof(int), cudaMemcpyHostToDevice))) ok = false;
    }
    if (ok) {
        if (!CUDA_OK(cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_bytes,
                                                               d_keys_in, d_keys_out,
                                                               d_vals_in, d_vals_out,
                                                               plan->n))) ok = false;
    }
    if (ok) {
        if (!CUDA_OK(cudaMalloc(&d_temp, temp_bytes))) ok = false;
    }
    if (ok) {
        if (!CUDA_OK(cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_bytes,
                                                               d_keys_in, d_keys_out,
                                                               d_vals_in, d_vals_out,
                                                               plan->n))) ok = false;
    }
    if (d_temp) cudaFree(d_temp);
    if (d_keys_in) cudaFree(d_keys_in);
    if (d_keys_out) cudaFree(d_keys_out);
    if (d_vals_in) cudaFree(d_vals_in);
    if (d_vals_out) cudaFree(d_vals_out);
    return ok;
}

static bool allocate_shared_fft_buffers(GpuPlan *plan) {
    size_t mb_si = 0, mb_sm = 0;
    size_t mc_si = 0, mc_sm = 0;
    int qb = plan->q_batch;
    for (int ell = 1; ell < plan->L; ++ell) {
        auto &lp = plan->levels[ell];
        /* Include ALL FFT-using levels (cuFFT AND fused, since fused falls back to cuFFT) */
        if (!lp.use_fft || lp.tier == GPU_TIER_SCHOOLBOOK) continue;
        int cn = lp.cn;
        int cb = plan->nn[ell - 1], pb = plan->nn[ell];
        /* With FFT-stride layout, real_in/real_out are no longer needed.
         * Only spectral buffers remain shared. */
        mb_si = std::max(mb_si, (size_t)qb * (size_t)cb * cn * sizeof(cufftDoubleComplex));
        mb_sm = std::max(mb_sm, (size_t)qb * (size_t)pb * cn * sizeof(cufftDoubleComplex));
        mc_si = std::max(mc_si, (size_t)qb * (size_t)pb * cn * sizeof(cufftDoubleComplex));
        mc_sm = std::max(mc_sm, (size_t)qb * (size_t)(2 * pb) * cn * sizeof(cufftDoubleComplex));
    }
    auto &sb = plan->shared_build_work;
    sb.real_in_bytes = 0; sb.spec_in_bytes = mb_si;
    sb.spec_mid_bytes = mb_sm; sb.real_out_bytes = 0;
    sb.real_in = nullptr; sb.real_out = nullptr;
    auto &sc = plan->shared_corr_work;
    sc.real_in_bytes = 0; sc.spec_in_bytes = mc_si;
    sc.spec_mid_bytes = mc_sm; sc.real_out_bytes = 0;
    sc.real_in = nullptr; sc.real_out = nullptr;
    if (!alloc_device(plan, (void **)&sb.spec_in, mb_si, plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&sb.spec_mid, mb_sm, plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&sc.spec_in, mc_si, plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&sc.spec_mid, mc_sm, plan->stream_compute)) return false;
    return true;
}

static bool allocate_plan_device_memory(GpuPlan *plan) {
    if (!CUDA_OK(cudaStreamCreate(&plan->stream_compute))) return false;
    if (!CUDA_OK(cudaStreamCreate(&plan->stream_aux))) return false;
    if (!CUDA_OK(cudaEventCreateWithFlags(&plan->evt_a_ready[0], cudaEventDisableTiming))) return false;
    if (!CUDA_OK(cudaEventCreateWithFlags(&plan->evt_a_ready[1], cudaEventDisableTiming))) return false;

    if (!maybe_init_mem_pool(plan)) return false;

    int qb = plan->q_batch;
    size_t block_prod_bytes = (plan->B > 1) ? (size_t)plan->N_tree * (plan->B + 1) * sizeof(double) : 0;

    plan->d_poly_levels.assign(plan->L, nullptr);
    plan->d_g_levels.assign(plan->L, nullptr);
    plan->d_fft_cache.assign(plan->L, nullptr);
    plan->fft_cache_valid.assign(plan->L, false);
    plan->build_fft.assign(plan->L, GpuFftBuffers{});
    plan->corr_fft.assign(plan->L, GpuFftBuffers{});

    /* Compute shared FFT buffer max sizes.
     * With FFT-stride layout, cuFFT reads/writes directly from poly/g levels,
     * so real_in and real_out are no longer needed.  Only the spectral buffers
     * (spec_in for forward output / cache source, spec_mid for pairwise multiply
     * output / inverse input) are still required. */
    size_t mb_ri=0, mb_si=0, mb_sm=0, mb_ro=0, mc_ri=0, mc_si=0, mc_sm=0, mc_ro=0;
    for (int ell = 1; ell < plan->L; ++ell) {
        auto &lp = plan->levels[ell];
        if (!lp.use_fft || lp.tier == GPU_TIER_SCHOOLBOOK) continue;
        int cn = lp.cn, cb = plan->nn[ell-1], pb = plan->nn[ell];
        /* real_in / real_out no longer needed (cuFFT uses poly/g directly) */
        mb_si = std::max(mb_si, (size_t)qb*cb*cn*sizeof(cufftDoubleComplex));
        mb_sm = std::max(mb_sm, (size_t)qb*pb*cn*sizeof(cufftDoubleComplex));
        mc_si = std::max(mc_si, (size_t)qb*pb*cn*sizeof(cufftDoubleComplex));
        mc_sm = std::max(mc_sm, (size_t)qb*2*pb*cn*sizeof(cufftDoubleComplex));
    }

    /* ── Arena: single cudaMalloc for all plan buffers ── */
    size_t arena_sz = 0;
    #define A(sz) do { arena_sz = (arena_sz + 255) & ~(size_t)255; arena_sz += (sz); } while(0)
    A((size_t)plan->n * sizeof(double));       /* d_S_sorted */
    A((size_t)plan->n * sizeof(int));          /* d_sort_perm */
    A((size_t)plan->n * sizeof(int));          /* d_inv_perm */
    A((size_t)plan->n * sizeof(double));       /* d_a_sorted[0] */
    A((size_t)plan->n * sizeof(double));       /* d_a_sorted[1] */
    A(sizeof(double)); A(sizeof(double));      /* d_graph_logv[0,1] */
    A(sizeof(double)); A(sizeof(double));      /* d_graph_scale[0,1] */
    A((size_t)plan->n * sizeof(double));       /* d_inner_sorted */
    A((size_t)plan->n * sizeof(double));       /* d_equity */
    A((size_t)plan->k * sizeof(double));       /* d_payout */
    A(block_prod_bytes);                        /* d_block_prods */
    if (plan->opts.enable_q_pipeline) {
        A((size_t)plan->nn[0] * plan->fft_stride[0] * sizeof(double)); /* d_poly_leaves_alt */
        A(block_prod_bytes);                    /* d_block_prods_alt */
    }
    if (qb > 1) {
        for (int qi = 0; qi < qb; ++qi) A((size_t)plan->n * sizeof(double)); /* d_a_qbatch */
        A((size_t)qb * plan->n * sizeof(double)); /* d_inner_qbatch */
        A((size_t)qb * block_prod_bytes);          /* d_block_prods_qbatch */
        A((size_t)qb * sizeof(double *));          /* d_qb_a_ptrs */
        A((size_t)qb * sizeof(double));            /* d_qb_weights */
        A((size_t)qb * sizeof(double));            /* d_qb_inv_vs */
    }
    for (int ell = 0; ell < plan->L; ++ell) {
        size_t pb = (size_t)qb * plan->nn[ell] * plan->fft_stride[ell] * sizeof(double);
        A(pb); A(pb); /* poly + g (at fft_stride spacing for cuFFT direct access) */
    }
    for (int ell = 1; ell < plan->L; ++ell) {
        auto &lp = plan->levels[ell];
        if (lp.use_fft && lp.cache_fft && lp.tier != GPU_TIER_SCHOOLBOOK)
            A((size_t)qb * plan->nn[ell-1] * lp.cn * sizeof(cufftDoubleComplex));
    }
    /* real_in / real_out no longer needed; only spectral buffers */
    A(mb_si); A(mb_sm);
    A(mc_si); A(mc_sm);
    /* callback info slots removed (callbacks disabled) */
    #undef A

    char *arena = nullptr;
    fprintf(stderr, "arena: allocating %zu bytes (%.1f MB)\n", arena_sz, (double)arena_sz / (1024.0 * 1024.0));
    if (arena_sz == 0) { set_last_errorf("Arena size is 0"); return false; }
    if (!CUDA_OK(cudaMalloc(&arena, arena_sz))) return false;
    if (!CUDA_OK(cudaMemset(arena, 0, arena_sz))) return false;
    plan->arena_base = arena;
    plan->arena_total_bytes = arena_sz;
    plan->peak_vram_bytes = arena_sz;
    plan->current_vram_bytes = arena_sz;

    /* ── Assign pointers from arena ── */
    size_t off = 0;
    #define P(ptr, type, sz) do { off = (off + 255) & ~(size_t)255; (ptr) = (type)(arena + off); off += (sz); } while(0)
    P(plan->d_S_sorted, double*, (size_t)plan->n * sizeof(double));
    P(plan->d_sort_perm, int*, (size_t)plan->n * sizeof(int));
    P(plan->d_inv_perm, int*, (size_t)plan->n * sizeof(int));
    P(plan->d_a_sorted[0], double*, (size_t)plan->n * sizeof(double));
    P(plan->d_a_sorted[1], double*, (size_t)plan->n * sizeof(double));
    P(plan->d_graph_logv[0], double*, sizeof(double));
    P(plan->d_graph_logv[1], double*, sizeof(double));
    P(plan->d_graph_scale[0], double*, sizeof(double));
    P(plan->d_graph_scale[1], double*, sizeof(double));
    P(plan->d_inner_sorted, double*, (size_t)plan->n * sizeof(double));
    P(plan->d_equity, double*, (size_t)plan->n * sizeof(double));
    P(plan->d_payout, double*, (size_t)plan->k * sizeof(double));
    if (block_prod_bytes > 0) P(plan->d_block_prods, double*, block_prod_bytes);
    if (plan->opts.enable_q_pipeline) {
        P(plan->d_poly_leaves_alt, double*, (size_t)plan->nn[0] * plan->fft_stride[0] * sizeof(double));
        if (block_prod_bytes > 0) P(plan->d_block_prods_alt, double*, block_prod_bytes);
        if (!CUDA_OK(cudaEventCreateWithFlags(&plan->evt_prop_done, cudaEventDisableTiming))) return false;
    }
    if (qb > 1) {
        for (int qi = 0; qi < qb; ++qi) P(plan->d_a_qbatch[qi], double*, (size_t)plan->n * sizeof(double));
        P(plan->d_inner_qbatch, double*, (size_t)qb * plan->n * sizeof(double));
        P(plan->d_block_prods_qbatch, double*, (size_t)qb * block_prod_bytes);
        P(plan->d_qb_a_ptrs, double**, (size_t)qb * sizeof(double*));
        P(plan->d_qb_weights, double*, (size_t)qb * sizeof(double));
        P(plan->d_qb_inv_vs, double*, (size_t)qb * sizeof(double));
    }
    for (int ell = 0; ell < plan->L; ++ell) {
        size_t pb = (size_t)qb * plan->nn[ell] * plan->fft_stride[ell] * sizeof(double);
        P(plan->d_poly_levels[ell], double*, pb);
        P(plan->d_g_levels[ell], double*, pb);
    }
    for (int ell = 1; ell < plan->L; ++ell) {
        auto &lp = plan->levels[ell];
        if (lp.use_fft && lp.cache_fft && lp.tier != GPU_TIER_SCHOOLBOOK)
            P(plan->d_fft_cache[ell], cufftDoubleComplex*, (size_t)qb * plan->nn[ell-1] * lp.cn * sizeof(cufftDoubleComplex));
    }
    auto &sb = plan->shared_build_work;
    auto &sc = plan->shared_corr_work;
    /* real_in / real_out no longer needed with FFT-stride layout */
    sb.real_in_bytes=0; sb.spec_in_bytes=mb_si; sb.spec_mid_bytes=mb_sm; sb.real_out_bytes=0;
    sc.real_in_bytes=0; sc.spec_in_bytes=mc_si; sc.spec_mid_bytes=mc_sm; sc.real_out_bytes=0;
    sb.real_in=nullptr; sb.real_out=nullptr;
    sc.real_in=nullptr; sc.real_out=nullptr;
    P(sb.spec_in, cufftDoubleComplex*, mb_si);
    P(sb.spec_mid, cufftDoubleComplex*, mb_sm);
    P(sc.spec_in, cufftDoubleComplex*, mc_si);
    P(sc.spec_mid, cufftDoubleComplex*, mc_sm);
    /* callback info slots removed (callbacks disabled) */
    #undef P
    plan->use_async_pool = false;
    fprintf(stderr, "arena: pointers assigned, off=%zu/%zu\n", off, arena_sz);

    for (int ell = 1; ell < plan->L; ++ell) {
        fprintf(stderr, "arena: creating cuFFT plans for level %d\n", ell);
        if (!allocate_level_buffers(plan, ell, {})) return false;
    }
    fprintf(stderr, "arena: all cuFFT plans created\n");

    /* Share cuFFT workspace across all plans */
    {
        size_t max_ws = 0;
        for (int ell = 1; ell < plan->L; ++ell) {
            auto &lp = plan->levels[ell];
            if (!lp.use_fft || lp.tier == GPU_TIER_SCHOOLBOOK) continue;
            size_t ws = 0;
            if (plan->build_fft[ell].plan_fwd) { cufftGetSize(plan->build_fft[ell].plan_fwd, &ws); max_ws = std::max(max_ws, ws); }
            if (plan->build_fft[ell].plan_inv) { cufftGetSize(plan->build_fft[ell].plan_inv, &ws); max_ws = std::max(max_ws, ws); }
            if (plan->corr_fft[ell].plan_fwd) { cufftGetSize(plan->corr_fft[ell].plan_fwd, &ws); max_ws = std::max(max_ws, ws); }
            if (plan->corr_fft[ell].plan_inv) { cufftGetSize(plan->corr_fft[ell].plan_inv, &ws); max_ws = std::max(max_ws, ws); }
        }
        if (max_ws > 0) {
            if (!alloc_device(plan, &plan->shared_cufft_workspace, max_ws, plan->stream_compute)) return false;
            plan->shared_cufft_workspace_bytes = max_ws;
            for (int ell = 1; ell < plan->L; ++ell) {
                auto &lp = plan->levels[ell];
                if (!lp.use_fft || lp.tier == GPU_TIER_SCHOOLBOOK) continue;
                auto &b = plan->build_fft[ell]; auto &c = plan->corr_fft[ell];
                if (b.plan_fwd) { CUFFT_OK(cufftSetAutoAllocation(b.plan_fwd, 0)); CUFFT_OK(cufftSetWorkArea(b.plan_fwd, plan->shared_cufft_workspace)); }
                if (b.plan_inv) { CUFFT_OK(cufftSetAutoAllocation(b.plan_inv, 0)); CUFFT_OK(cufftSetWorkArea(b.plan_inv, plan->shared_cufft_workspace)); }
                if (c.plan_fwd) { CUFFT_OK(cufftSetAutoAllocation(c.plan_fwd, 0)); CUFFT_OK(cufftSetWorkArea(c.plan_fwd, plan->shared_cufft_workspace)); }
                if (c.plan_inv) { CUFFT_OK(cufftSetAutoAllocation(c.plan_inv, 0)); CUFFT_OK(cufftSetWorkArea(c.plan_inv, plan->shared_cufft_workspace)); }
            }
        }
    }

    fprintf(stderr, "arena: workspace done, starting H->D copies\n");
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_S_sorted, plan->S_sorted.data(),
                                 (size_t)plan->n * sizeof(double), cudaMemcpyHostToDevice,
                                 plan->stream_compute))) return false;
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_sort_perm, plan->sort_perm.data(),
                                 (size_t)plan->n * sizeof(int), cudaMemcpyHostToDevice,
                                 plan->stream_compute))) return false;
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_inv_perm, plan->inv_perm.data(),
                                 (size_t)plan->n * sizeof(int), cudaMemcpyHostToDevice,
                                 plan->stream_compute))) return false;
    if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
    fprintf(stderr, "arena: allocate_plan_device_memory done\n");
    return true;
}

static bool destroy_fft_buffers(GpuPlan *plan, GpuFftBuffers &b, cudaStream_t stream) {
    (void)plan; (void)stream;
    /* Only destroy cuFFT plan handles; device memory is shared and freed centrally */
    if (b.plan_fwd) { if (!CUFFT_OK(cufftDestroy(b.plan_fwd))) return false; b.plan_fwd = 0; }
    if (b.plan_inv) { if (!CUFFT_OK(cufftDestroy(b.plan_inv))) return false; b.plan_inv = 0; }
    b = GpuFftBuffers{};
    return true;
}

static void destroy_plan(GpuPlan *plan) {
    if (!plan) return;
    cudaStream_t stream = plan->stream_compute;
    if (stream) cudaStreamSynchronize(stream);

    /* Destroy cuFFT plan handles (lightweight, no device memory) */
    for (int ell = 1; ell < plan->L; ++ell) {
        destroy_fft_buffers(plan, plan->build_fft[ell], stream);
        destroy_fft_buffers(plan, plan->corr_fft[ell], stream);
    }

    /* cuFFT workspace is a separate allocation (not in arena) */
    if (plan->shared_cufft_workspace) {
        cudaFree(plan->shared_cufft_workspace);
        plan->shared_cufft_workspace = nullptr;
    }

    /* Single free for the entire arena */
    if (plan->arena_base) {
        cudaFree(plan->arena_base);
        plan->arena_base = nullptr;
    }

    /* Pipeline event */
    for (int q = 0; q < 2; ++q) {
        if (plan->graph_exec[q]) cudaGraphExecDestroy(plan->graph_exec[q]);
        if (plan->graph[q]) cudaGraphDestroy(plan->graph[q]);
    }
    if (plan->evt_a_ready[0]) cudaEventDestroy(plan->evt_a_ready[0]);
    if (plan->evt_a_ready[1]) cudaEventDestroy(plan->evt_a_ready[1]);
    if (plan->evt_prop_done) cudaEventDestroy(plan->evt_prop_done);
    if (plan->stream_aux) cudaStreamDestroy(plan->stream_aux);
    if (plan->stream_compute) cudaStreamDestroy(plan->stream_compute);
    delete plan;
}

static bool run_build_level_schoolbook(GpuPlan *plan, int ell);
static bool run_prop_level_schoolbook(GpuPlan *plan, int ell);

static bool run_build_level_fft(GpuPlan *plan, int ell) {
    int pps = plan->psz[ell];
    int child_batch = plan->nn[ell - 1];
    int parent_batch = plan->nn[ell];
    int parent_stride = plan->fft_stride[ell];
    int child_stride = plan->fft_stride[ell - 1];
    auto &lp = plan->levels[ell];
    auto &b = plan->build_fft[ell];
    int threads = 256;

    /* D2Z: read directly from poly_levels[ell-1] at fft_stride[ell-1] */
    /* When caching, write FFT output directly to cache to avoid D2D copy */
    cufftDoubleComplex *fwd_out = (lp.cache_fft && plan->d_fft_cache[ell]) ? plan->d_fft_cache[ell] : b.spec_in;
    if (!CUFFT_OK(cufftExecD2Z(b.plan_fwd, plan->d_poly_levels[ell - 1], fwd_out))) return false;

    if (lp.cache_fft && plan->d_fft_cache[ell]) {
        if (ell < (int)plan->fft_cache_valid.size()) plan->fft_cache_valid[ell] = true;
    }

    /* Pairwise multiply: when fwd_out != spec_in, write product to spec_in
     * (avoids needing spec_mid for the build path) */
    cufftDoubleComplex *mul_out = (fwd_out != b.spec_in) ? b.spec_in : b.spec_mid;
    size_t mul_total = (size_t)parent_batch * (size_t)b.cn;
    int blocks_mul = (int)((mul_total + threads - 1) / threads);
    k_pairwise_mul<<<blocks_mul, threads, 0, plan->stream_compute>>>(
        fwd_out, b.cn, mul_out, parent_batch);
    if (!CUDA_OK(cudaGetLastError())) return false;

    /* Z2D: write directly to poly_levels[ell] at fft_stride[ell] */
    if (!CUFFT_OK(cufftExecZ2D(b.plan_inv, mul_out, plan->d_poly_levels[ell]))) return false;

    /* Scale valid coefficients by 1/fft_n and zero the padding region */
    size_t szp_total = (size_t)parent_batch * (size_t)parent_stride;
    int blocks_szp = (int)((szp_total + threads - 1) / threads);
    k_scale_zero_pad<<<blocks_szp, threads, 0, plan->stream_compute>>>(
        plan->d_poly_levels[ell], parent_stride, pps,
        1.0 / (double)b.fft_n, parent_batch);
    if (!CUDA_OK(cudaGetLastError())) return false;

    if (lp.build_wrap_m > 0) {
        int cps = plan->psz[ell - 1];
        k_wrap_build<<<parent_batch, 1, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], pps, parent_batch,
            plan->d_poly_levels[ell - 1], cps, lp.build_conv,
            b.fft_n, lp.build_wrap_m,
            parent_stride, child_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

static bool run_build_level_fused(GpuPlan *plan, int ell) {
    auto &lp = plan->levels[ell];
    if (!plan->opts.use_cufftdx || g_runtime_fused_max_conv_len <= 0 ||
        lp.build_conv > g_runtime_fused_max_conv_len || !is_cufftdx_supported_fft_n(lp.fft_n)) {
        return run_build_level_fft(plan, ell);
    }
    int cps = plan->psz[ell - 1];
    int pps = plan->psz[ell];
    int child_stride = plan->fft_stride[ell - 1];
    int parent_stride = plan->fft_stride[ell];
    int nparents = plan->nn[ell];
    if (nparents <= 0 || cps <= 0 || pps <= 0) return true;
    /* Try R2C first (half FFT work for real polynomials), fall back to C2C, then cuFFT */
    bool ok = launch_cufftdx_build_r2c_dispatch(lp.fft_n,
                                                plan->d_poly_levels[ell - 1], child_stride,
                                                plan->d_poly_levels[ell], parent_stride, nparents,
                                                1.0 / (double)lp.fft_n,
                                                plan->stream_compute);
    if (!ok) {
        ok = launch_cufftdx_build_dispatch(lp.fft_n,
                                           plan->d_poly_levels[ell - 1], child_stride,
                                           plan->d_poly_levels[ell], parent_stride, nparents,
                                           1.0 / (double)lp.fft_n,
                                           plan->stream_compute);
    }
    if (!ok) return run_build_level_fft(plan, ell);
    if (lp.build_wrap_m > 0) {
        k_wrap_build<<<nparents, 1, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], pps, nparents,
            plan->d_poly_levels[ell - 1], cps, lp.build_conv,
            lp.fft_n, lp.build_wrap_m,
            parent_stride, child_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

static bool run_build_level_schoolbook(GpuPlan *plan, int ell) {
    int cps = plan->psz[ell - 1];
    int pps = plan->psz[ell];
    int child_stride = plan->fft_stride[ell - 1];
    int parent_stride = plan->fft_stride[ell];
    int nparents = plan->nn[ell];
    if (nparents <= 0 || cps <= 0 || pps <= 0) return true;

    int conv = plan->levels[ell].build_conv;
    bool use_warp_regime = (conv <= GPU_SCHOOL_WARP_MAX_CONV && nparents > 1);
    if (use_warp_regime) {
        int threads = GPU_SCHOOL_WARPS_PER_BLOCK * 32;
        int blocks = (nparents + GPU_SCHOOL_WARPS_PER_BLOCK - 1) / GPU_SCHOOL_WARPS_PER_BLOCK;
        size_t shmem = (size_t)GPU_SCHOOL_WARPS_PER_BLOCK * (size_t)(2 * cps) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_build_warp_batch<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents,
                child_stride, parent_stride);
        } else {
            int fb_threads = 256;
            size_t total = (size_t)nparents * (size_t)pps;
            int fb_blocks = (int)((total + fb_threads - 1) / fb_threads);
            k_schoolbook_build<<<fb_blocks, fb_threads, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents,
                child_stride, parent_stride);
        }
    } else {
        int threads = 256;
        int blocks = nparents;
        size_t shmem = (size_t)(2 * cps) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_build_smem_parent<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents,
                child_stride, parent_stride);
        } else {
            size_t total = (size_t)nparents * (size_t)pps;
            int fb_blocks = (int)((total + threads - 1) / threads);
            k_schoolbook_build<<<fb_blocks, threads, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents,
                child_stride, parent_stride);
        }
    }
    if (!CUDA_OK(cudaGetLastError())) return false;
    return true;
}

static bool run_prop_level_fft(GpuPlan *plan, int ell) {
    int child_gsz = plan->psz[ell - 1];
    int nparents = plan->nn[ell];
    int parent_stride = plan->fft_stride[ell];
    int child_stride = plan->fft_stride[ell - 1];
    auto &lp = plan->levels[ell];
    auto &c = plan->corr_fft[ell];
    int len_g = lp.g_eff;
    int len_P = lp.p_eff;
    int len_out = lp.out_needed;
    int threads = 256;

    /* D2Z: read directly from g_levels[ell] at fft_stride[ell] */
    if (!CUFFT_OK(cufftExecD2Z(c.plan_fwd, plan->d_g_levels[ell], c.spec_in))) return false;

    const cufftDoubleComplex *child_spec = (plan->d_fft_cache[ell] && ell < (int)plan->fft_cache_valid.size() && plan->fft_cache_valid[ell]) ? plan->d_fft_cache[ell] : nullptr;
    if (!child_spec) {
        /* Re-compute child spectra: D2Z directly from poly_levels[ell-1] */
        auto &b = plan->build_fft[ell];
        if (!CUFFT_OK(cufftExecD2Z(b.plan_fwd, plan->d_poly_levels[ell - 1], b.spec_in))) return false;
        child_spec = b.spec_in;
    }

    size_t corr_total = (size_t)nparents * (size_t)c.cn;
    int blocks_corr = (int)((corr_total + threads - 1) / threads);
    k_paired_corr_freq<<<blocks_corr, threads, 0, plan->stream_compute>>>(
        c.spec_in, child_spec, c.cn, nparents, c.spec_mid);
    if (!CUDA_OK(cudaGetLastError())) return false;

    /* Z2D: write directly to g_levels[ell-1] at fft_stride[ell-1] */
    if (!CUFFT_OK(cufftExecZ2D(c.plan_inv, c.spec_mid, plan->d_g_levels[ell - 1]))) return false;

    /* Scale valid coefficients by 1/fft_n and zero the padding region */
    int n_children = 2 * nparents;
    size_t szp_total = (size_t)n_children * (size_t)child_stride;
    int blocks_szp = (int)((szp_total + threads - 1) / threads);
    k_scale_zero_pad<<<blocks_szp, threads, 0, plan->stream_compute>>>(
        plan->d_g_levels[ell - 1], child_stride, child_gsz,
        1.0 / (double)c.fft_n, n_children);
    if (!CUDA_OK(cudaGetLastError())) return false;

    if (lp.corr_wrap_m > 0) {
        k_wrap_corr_pair<<<nparents, 1, 0, plan->stream_compute>>>(
            plan->d_g_levels[ell - 1], child_gsz, nparents,
            plan->d_g_levels[ell], plan->psz[ell], len_g,
            plan->d_poly_levels[ell - 1], plan->psz[ell - 1], len_P,
            len_out,
            c.fft_n, lp.corr_wrap_m,
            child_stride, parent_stride, child_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    /* Mark cache as consumed (don't free -- needed for graph address stability) */
    if (plan->opts.memory_strategy >= 2 && ell < (int)plan->fft_cache_valid.size()) {
        plan->fft_cache_valid[ell] = false;
    }
    return true;
}

static bool run_prop_level_fused(GpuPlan *plan, int ell) {
    auto &lp = plan->levels[ell];
    if (!plan->opts.use_cufftdx || g_runtime_fused_max_conv_len <= 0 ||
        lp.corr_conv > g_runtime_fused_max_conv_len || !is_cufftdx_supported_fft_n(lp.fft_n)) {
        return run_prop_level_fft(plan, ell);
    }
    int parent_gsz = plan->psz[ell];
    int child_gsz = plan->psz[ell - 1];
    int cps = plan->psz[ell - 1];
    int parent_stride = plan->fft_stride[ell];
    int child_stride = plan->fft_stride[ell - 1];
    int nparents = plan->nn[ell];
    int len_g = lp.g_eff;
    int len_P = lp.p_eff;
    int len_out = lp.out_needed;
    if (nparents <= 0 || len_out <= 0 || len_g <= 0 || len_P <= 0) return true;
    /* Try R2C first (half FFT work for real polynomials), fall back to C2C, then cuFFT */
    bool ok = launch_cufftdx_corr_r2c_dispatch(lp.fft_n,
                                               plan->d_g_levels[ell], parent_stride, len_g,
                                               plan->d_poly_levels[ell - 1], child_stride, len_P,
                                               plan->d_g_levels[ell - 1], child_stride, len_out, nparents,
                                               1.0 / (double)lp.fft_n,
                                               plan->stream_compute);
    if (!ok) {
        ok = launch_cufftdx_corr_dispatch(lp.fft_n,
                                          plan->d_g_levels[ell], parent_stride, len_g,
                                          plan->d_poly_levels[ell - 1], child_stride, len_P,
                                          plan->d_g_levels[ell - 1], child_stride, len_out, nparents,
                                          1.0 / (double)lp.fft_n,
                                          plan->stream_compute);
    }
    if (!ok) return run_prop_level_fft(plan, ell);
    if (lp.corr_wrap_m > 0) {
        k_wrap_corr_pair<<<nparents, 1, 0, plan->stream_compute>>>(
            plan->d_g_levels[ell - 1], child_gsz, nparents,
            plan->d_g_levels[ell], parent_gsz, len_g,
            plan->d_poly_levels[ell - 1], cps, len_P,
            len_out,
            lp.fft_n, lp.corr_wrap_m,
            child_stride, parent_stride, child_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

static bool run_prop_level_schoolbook(GpuPlan *plan, int ell) {
    int parent_gsz = plan->psz[ell];
    int child_gsz = plan->psz[ell - 1];
    int cps = plan->psz[ell - 1];
    int parent_stride = plan->fft_stride[ell];
    int child_stride = plan->fft_stride[ell - 1];
    int nparents = plan->nn[ell];
    auto &lp = plan->levels[ell];
    int len_g = lp.g_eff;
    int len_P = lp.p_eff;
    int len_out = lp.out_needed;
    if (nparents <= 0 || len_out <= 0 || len_g <= 0 || len_P <= 0) return true;

    int conv = lp.corr_conv;
    bool use_warp_regime = (conv <= GPU_SCHOOL_WARP_MAX_CONV && nparents > 1);
    if (use_warp_regime) {
        int threads = GPU_SCHOOL_WARPS_PER_BLOCK * 32;
        int blocks = (nparents + GPU_SCHOOL_WARPS_PER_BLOCK - 1) / GPU_SCHOOL_WARPS_PER_BLOCK;
        size_t shmem = (size_t)GPU_SCHOOL_WARPS_PER_BLOCK
            * (size_t)(len_g + 2 * len_P) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_corr_pair_warp_batch<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents,
                parent_stride, child_stride, child_stride);
        } else {
            int fb_threads = 256;
            size_t total = (size_t)nparents * (size_t)len_out;
            int fb_blocks = (int)((total + fb_threads - 1) / fb_threads);
            k_schoolbook_corr_pair<<<fb_blocks, fb_threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents,
                parent_stride, child_stride, child_stride);
        }
    } else {
        int threads = 256;
        int blocks = nparents;
        size_t shmem = (size_t)(len_g + 2 * len_P) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_corr_pair_smem_parent<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents,
                parent_stride, child_stride, child_stride);
        } else {
            size_t total = (size_t)nparents * (size_t)len_out;
            int fb_blocks = (int)((total + threads - 1) / threads);
            k_schoolbook_corr_pair<<<fb_blocks, threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents,
                parent_stride, child_stride, child_stride);
        }
    }
    if (!CUDA_OK(cudaGetLastError())) return false;
    return true;
}

/* ── Q-batched tree build/propagate ──
 *
 * These functions run the tree build and propagation for q_batch Q-points
 * simultaneously. The poly/g level buffers have layout:
 *   [Q0_node0..Q0_nodeN, Q1_node0..Q1_nodeN, ... Qq_node0..Qq_nodeN]
 * where each Q-point's nodes occupy nn[ell] * fft_stride[ell] doubles.
 *
 * Because nn[ell-1] = 2 * nn[ell], the parent-child indexing is preserved:
 * parent at (global) index nn[ell]*qi + p reads children at
 * nn[ell-1]*qi + 2p and nn[ell-1]*qi + 2p + 1, which is correct because
 * the Q-point blocks are contiguous and parent p maps to children 2p, 2p+1
 * within each Q-point's block. The existing kernels (mul,
 * schoolbook, wrap, scale_zero_pad) all operate on flat arrays indexed by nparents, so we
 * just multiply nparents (and all batch counts) by q_batch.
 */

static bool run_build_level_schoolbook_qb(GpuPlan *plan, int ell, int qb) {
    int cps = plan->psz[ell - 1];
    int pps = plan->psz[ell];
    int cs = plan->fft_stride[ell - 1];
    int ps = plan->fft_stride[ell];
    int nparents_total = qb * plan->nn[ell];
    if (nparents_total <= 0 || cps <= 0 || pps <= 0) return true;

    int conv = plan->levels[ell].build_conv;
    bool use_warp_regime = (conv <= GPU_SCHOOL_WARP_MAX_CONV && nparents_total > 1);
    if (use_warp_regime) {
        int threads = GPU_SCHOOL_WARPS_PER_BLOCK * 32;
        int blocks = (nparents_total + GPU_SCHOOL_WARPS_PER_BLOCK - 1) / GPU_SCHOOL_WARPS_PER_BLOCK;
        size_t shmem = (size_t)GPU_SCHOOL_WARPS_PER_BLOCK * (size_t)(2 * cps) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_build_warp_batch<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents_total, cs, ps);
        } else {
            int fb_threads = 256;
            size_t total = (size_t)nparents_total * (size_t)pps;
            int fb_blocks = (int)((total + fb_threads - 1) / fb_threads);
            k_schoolbook_build<<<fb_blocks, fb_threads, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents_total, cs, ps);
        }
    } else {
        int threads = 256;
        int blocks = nparents_total;
        size_t shmem = (size_t)(2 * cps) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_build_smem_parent<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents_total, cs, ps);
        } else {
            size_t total = (size_t)nparents_total * (size_t)pps;
            int fb_blocks = (int)((total + threads - 1) / threads);
            k_schoolbook_build<<<fb_blocks, threads, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents_total, cs, ps);
        }
    }
    if (!CUDA_OK(cudaGetLastError())) return false;
    return true;
}

static bool run_build_level_fft_qb(GpuPlan *plan, int ell, int qb) {
    int cps = plan->psz[ell - 1];
    int pps = plan->psz[ell];
    int child_batch = qb * plan->nn[ell - 1];
    int parent_batch = qb * plan->nn[ell];
    int parent_stride = plan->fft_stride[ell];
    int child_stride = plan->fft_stride[ell - 1];
    auto &lp = plan->levels[ell];
    auto &b = plan->build_fft[ell];
    int threads = 256;

    /* D2Z: read directly from poly_levels[ell-1] at fft_stride */
    cufftDoubleComplex *fwd_out = (lp.cache_fft && plan->d_fft_cache[ell]) ? plan->d_fft_cache[ell] : b.spec_in;
    if (!CUFFT_OK(cufftExecD2Z(b.plan_fwd, plan->d_poly_levels[ell - 1], fwd_out))) return false;

    if (lp.cache_fft && plan->d_fft_cache[ell]) {
        if (ell < (int)plan->fft_cache_valid.size()) plan->fft_cache_valid[ell] = true;
    }

    cufftDoubleComplex *mul_out = (fwd_out != b.spec_in) ? b.spec_in : b.spec_mid;
    size_t mul_total = (size_t)parent_batch * (size_t)b.cn;
    int blocks_mul = (int)((mul_total + threads - 1) / threads);
    k_pairwise_mul<<<blocks_mul, threads, 0, plan->stream_compute>>>(
        fwd_out, b.cn, mul_out, parent_batch);
    if (!CUDA_OK(cudaGetLastError())) return false;

    /* Z2D: write directly to poly_levels[ell] at fft_stride */
    if (!CUFFT_OK(cufftExecZ2D(b.plan_inv, mul_out, plan->d_poly_levels[ell]))) return false;

    /* Scale valid coefficients and zero padding */
    size_t szp_total = (size_t)parent_batch * (size_t)parent_stride;
    int blocks_szp = (int)((szp_total + threads - 1) / threads);
    k_scale_zero_pad<<<blocks_szp, threads, 0, plan->stream_compute>>>(
        plan->d_poly_levels[ell], parent_stride, pps,
        1.0 / (double)b.fft_n, parent_batch);
    if (!CUDA_OK(cudaGetLastError())) return false;
    if (lp.build_wrap_m > 0) {
        k_wrap_build<<<parent_batch, 1, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], pps, parent_batch,
            plan->d_poly_levels[ell - 1], cps, lp.build_conv,
            b.fft_n, lp.build_wrap_m,
            parent_stride, child_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

static bool run_build_level_fused_qb(GpuPlan *plan, int ell, int qb) {
    auto &lp = plan->levels[ell];
    int nparents_total = qb * plan->nn[ell];
    if (!plan->opts.use_cufftdx || g_runtime_fused_max_conv_len <= 0 ||
        lp.build_conv > g_runtime_fused_max_conv_len || !is_cufftdx_supported_fft_n(lp.fft_n)) {
        return run_build_level_fft_qb(plan, ell, qb);
    }
    int cps = plan->psz[ell - 1];
    int pps = plan->psz[ell];
    int cs = plan->fft_stride[ell - 1];
    int ps = plan->fft_stride[ell];
    if (nparents_total <= 0 || cps <= 0 || pps <= 0) return true;
    /* Try R2C first (half FFT work for real polynomials), fall back to C2C, then cuFFT */
    bool ok = launch_cufftdx_build_r2c_dispatch(lp.fft_n,
                                                plan->d_poly_levels[ell - 1], cs,
                                                plan->d_poly_levels[ell], ps, nparents_total,
                                                1.0 / (double)lp.fft_n,
                                                plan->stream_compute);
    if (!ok) {
        ok = launch_cufftdx_build_dispatch(lp.fft_n,
                                            plan->d_poly_levels[ell - 1], cs,
                                            plan->d_poly_levels[ell], ps, nparents_total,
                                            1.0 / (double)lp.fft_n,
                                            plan->stream_compute);
    }
    if (!ok) return run_build_level_fft_qb(plan, ell, qb);
    if (lp.build_wrap_m > 0) {
        k_wrap_build<<<nparents_total, 1, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], pps, nparents_total,
            plan->d_poly_levels[ell - 1], cps, lp.build_conv,
            lp.fft_n, lp.build_wrap_m,
            ps, cs);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

static bool run_prop_level_schoolbook_qb(GpuPlan *plan, int ell, int qb) {
    int parent_gsz = plan->psz[ell];
    int child_gsz = plan->psz[ell - 1];
    int cps = plan->psz[ell - 1];
    int ps = plan->fft_stride[ell];
    int cs = plan->fft_stride[ell - 1];
    int nparents_total = qb * plan->nn[ell];
    auto &lp = plan->levels[ell];
    int len_g = lp.g_eff;
    int len_P = lp.p_eff;
    int len_out = lp.out_needed;
    if (nparents_total <= 0 || len_out <= 0 || len_g <= 0 || len_P <= 0) return true;

    int conv = lp.corr_conv;
    bool use_warp_regime = (conv <= GPU_SCHOOL_WARP_MAX_CONV && nparents_total > 1);
    if (use_warp_regime) {
        int threads = GPU_SCHOOL_WARPS_PER_BLOCK * 32;
        int blocks = (nparents_total + GPU_SCHOOL_WARPS_PER_BLOCK - 1) / GPU_SCHOOL_WARPS_PER_BLOCK;
        size_t shmem = (size_t)GPU_SCHOOL_WARPS_PER_BLOCK
            * (size_t)(len_g + 2 * len_P) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_corr_pair_warp_batch<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents_total,
                ps, cs, cs);
        } else {
            int fb_threads = 256;
            size_t total = (size_t)nparents_total * (size_t)len_out;
            int fb_blocks = (int)((total + fb_threads - 1) / fb_threads);
            k_schoolbook_corr_pair<<<fb_blocks, fb_threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents_total,
                ps, cs, cs);
        }
    } else {
        int threads = 256;
        int blocks = nparents_total;
        size_t shmem = (size_t)(len_g + 2 * len_P) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_corr_pair_smem_parent<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents_total,
                ps, cs, cs);
        } else {
            size_t total = (size_t)nparents_total * (size_t)len_out;
            int fb_blocks = (int)((total + threads - 1) / threads);
            k_schoolbook_corr_pair<<<fb_blocks, threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents_total,
                ps, cs, cs);
        }
    }
    if (!CUDA_OK(cudaGetLastError())) return false;
    return true;
}

static bool run_prop_level_fft_qb(GpuPlan *plan, int ell, int qb) {
    int child_gsz = plan->psz[ell - 1];
    int nparents_total = qb * plan->nn[ell];
    int ps = plan->fft_stride[ell];
    int cs = plan->fft_stride[ell - 1];
    auto &lp = plan->levels[ell];
    auto &c = plan->corr_fft[ell];
    int len_g = lp.g_eff;
    int len_P = lp.p_eff;
    int len_out = lp.out_needed;
    int threads = 256;

    /* D2Z: read directly from g_levels[ell] at fft_stride */
    if (!CUFFT_OK(cufftExecD2Z(c.plan_fwd, plan->d_g_levels[ell], c.spec_in))) return false;

    const cufftDoubleComplex *child_spec = (plan->d_fft_cache[ell] && ell < (int)plan->fft_cache_valid.size() && plan->fft_cache_valid[ell]) ? plan->d_fft_cache[ell] : nullptr;
    if (!child_spec) {
        auto &b = plan->build_fft[ell];
        if (!CUFFT_OK(cufftExecD2Z(b.plan_fwd, plan->d_poly_levels[ell - 1], b.spec_in))) return false;
        child_spec = b.spec_in;
    }

    size_t corr_total = (size_t)nparents_total * (size_t)c.cn;
    int blocks_corr = (int)((corr_total + threads - 1) / threads);
    k_paired_corr_freq<<<blocks_corr, threads, 0, plan->stream_compute>>>(
        c.spec_in, child_spec, c.cn, nparents_total, c.spec_mid);
    if (!CUDA_OK(cudaGetLastError())) return false;

    /* Z2D: write directly to g_levels[ell-1] at fft_stride */
    if (!CUFFT_OK(cufftExecZ2D(c.plan_inv, c.spec_mid, plan->d_g_levels[ell - 1]))) return false;

    /* Scale valid coefficients and zero padding */
    int n_children = 2 * nparents_total;
    size_t szp_total = (size_t)n_children * (size_t)cs;
    int blocks_szp = (int)((szp_total + threads - 1) / threads);
    k_scale_zero_pad<<<blocks_szp, threads, 0, plan->stream_compute>>>(
        plan->d_g_levels[ell - 1], cs, child_gsz,
        1.0 / (double)c.fft_n, n_children);
    if (!CUDA_OK(cudaGetLastError())) return false;

    if (lp.corr_wrap_m > 0) {
        k_wrap_corr_pair<<<nparents_total, 1, 0, plan->stream_compute>>>(
            plan->d_g_levels[ell - 1], child_gsz, nparents_total,
            plan->d_g_levels[ell], plan->psz[ell], len_g,
            plan->d_poly_levels[ell - 1], plan->psz[ell - 1], len_P,
            len_out,
            c.fft_n, lp.corr_wrap_m,
            cs, ps, cs);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    /* Mark cache as consumed */
    if (plan->opts.memory_strategy >= 2 && ell < (int)plan->fft_cache_valid.size()) {
        plan->fft_cache_valid[ell] = false;
    }
    return true;
}

static bool run_prop_level_fused_qb(GpuPlan *plan, int ell, int qb) {
    auto &lp = plan->levels[ell];
    int nparents_total = qb * plan->nn[ell];
    if (!plan->opts.use_cufftdx || g_runtime_fused_max_conv_len <= 0 ||
        lp.corr_conv > g_runtime_fused_max_conv_len || !is_cufftdx_supported_fft_n(lp.fft_n)) {
        return run_prop_level_fft_qb(plan, ell, qb);
    }
    int parent_gsz = plan->psz[ell];
    int child_gsz = plan->psz[ell - 1];
    int cps = plan->psz[ell - 1];
    int ps = plan->fft_stride[ell];
    int cs = plan->fft_stride[ell - 1];
    int len_g = lp.g_eff;
    int len_P = lp.p_eff;
    int len_out = lp.out_needed;
    if (nparents_total <= 0 || len_out <= 0 || len_g <= 0 || len_P <= 0) return true;
    /* Try R2C first (half FFT work for real polynomials), fall back to C2C, then cuFFT */
    bool ok = launch_cufftdx_corr_r2c_dispatch(lp.fft_n,
                                               plan->d_g_levels[ell], ps, len_g,
                                               plan->d_poly_levels[ell - 1], cs, len_P,
                                               plan->d_g_levels[ell - 1], cs, len_out, nparents_total,
                                               1.0 / (double)lp.fft_n,
                                               plan->stream_compute);
    if (!ok) {
        ok = launch_cufftdx_corr_dispatch(lp.fft_n,
                                           plan->d_g_levels[ell], ps, len_g,
                                           plan->d_poly_levels[ell - 1], cs, len_P,
                                           plan->d_g_levels[ell - 1], cs, len_out, nparents_total,
                                           1.0 / (double)lp.fft_n,
                                           plan->stream_compute);
    }
    if (!ok) return run_prop_level_fft_qb(plan, ell, qb);
    if (lp.corr_wrap_m > 0) {
        k_wrap_corr_pair<<<nparents_total, 1, 0, plan->stream_compute>>>(
            plan->d_g_levels[ell - 1], child_gsz, nparents_total,
            plan->d_g_levels[ell], parent_gsz, len_g,
            plan->d_poly_levels[ell - 1], cps, len_P,
            len_out,
            lp.fft_n, lp.corr_wrap_m,
            cs, ps, cs);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

/* ── Main Q-batch hybrid execution function ──
 *
 * Processes qb Q-points in a single pass through the tree.
 * pts[0..qb-1] are the Q-points to process (logv, w).
 */
static bool run_hybrid_batched_q(GpuPlan *plan, const QP *pts, int qb) {
    int threads = 256;
    int blocks_n = (plan->n + threads - 1) / threads;

    /* 1. Compute a_sorted for each Q-point */
    for (int qi = 0; qi < qb; ++qi) {
        k_compute_a<<<blocks_n, threads, 0, plan->stream_compute>>>(
            plan->d_S_sorted, plan->d_a_qbatch[qi], plan->n, pts[qi].logv);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    /* 2. Block build for all Q-points.
     * Upload a_ptrs to pre-allocated device buffer. */
    double *h_a_ptrs[Q_BATCH_MAX];
    for (int qi = 0; qi < qb; ++qi) h_a_ptrs[qi] = plan->d_a_qbatch[qi];

    double **d_a_ptrs = plan->d_qb_a_ptrs;
    if (!CUDA_OK(cudaMemcpyAsync(d_a_ptrs, h_a_ptrs, (size_t)qb * sizeof(double *),
                                 cudaMemcpyHostToDevice, plan->stream_compute))) return false;

    size_t leaf_stride = (size_t)plan->nn[0] * (size_t)plan->fft_stride[0];
    if (plan->B <= 1) {
        /* B=1: set leaves directly for each Q-point in the batch */
        for (int qi = 0; qi < qb; ++qi) {
            int bl = (plan->N_tree + 255) / 256;
            double *leaves_qi = plan->d_poly_levels[0] + qi * leaf_stride;
            k_set_leaves_b1<<<bl, 256, 0, plan->stream_compute>>>(
                plan->d_a_qbatch[qi], plan->n, plan->N_tree,
                plan->fft_stride[0], leaves_qi);
        }
        if (!CUDA_OK(cudaGetLastError())) return false;
    } else {
        size_t bp_stride = (size_t)plan->N_tree * (size_t)(plan->B + 1);
        int threads_block = 256;
        size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);
        k_block_build_qbatch<<<qb * plan->N_tree, threads_block, shmem_block, plan->stream_compute>>>(
            (const double * const *)d_a_ptrs, plan->n, plan->B,
            plan->nblocks, plan->N_tree, qb,
            plan->fft_stride[0], plan->d_poly_levels[0], leaf_stride,
            plan->d_block_prods_qbatch, bp_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    /* 3. Tree build for all Q-points (q_batch-scaled batch sizes) */
    for (int ell = 1; ell < plan->L - 1; ++ell) {
        auto &lp = plan->levels[ell];
        if (!lp.use_fft) { if (!run_build_level_schoolbook_qb(plan, ell, qb)) return false; }
        else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_build_level_fused_qb(plan, ell, qb)) return false; }
        else { if (!run_build_level_fft_qb(plan, ell, qb)) return false; }
    }

    /* 4. Set root g for all Q-points */
    int top = plan->L - 1;
    int root_gsz = plan->fft_stride[top];
    size_t g_root_stride = (size_t)plan->nn[top] * (size_t)root_gsz;
    int root_total = qb * root_gsz;
    int blocks_root = (root_total + threads - 1) / threads;
    k_set_root_g_qbatch<<<blocks_root, threads, 0, plan->stream_compute>>>(
        plan->d_g_levels[top], root_gsz, plan->d_payout, plan->k,
        qb, g_root_stride);
    if (!CUDA_OK(cudaGetLastError())) return false;

    /* 5. Tree propagation for all Q-points */
    for (int ell = top; ell >= 1; --ell) {
        auto &lp = plan->levels[ell];
        if (!lp.use_fft) { if (!run_prop_level_schoolbook_qb(plan, ell, qb)) return false; }
        else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_prop_level_fused_qb(plan, ell, qb)) return false; }
        else { if (!run_prop_level_fft_qb(plan, ell, qb)) return false; }
    }

    /* 6. Leaf extract for all Q-points */
    size_t leaf_g_stride = (size_t)plan->nn[0] * (size_t)plan->fft_stride[0];
    size_t inner_stride = (size_t)plan->n;
    if (plan->B <= 1) {
        /* B=1: direct read of g_leaf[i][0] for each Q-point */
        for (int qi = 0; qi < qb; ++qi) {
            int bl = (plan->n + 255) / 256;
            double *g_leaf_qi = plan->d_g_levels[0] + qi * leaf_g_stride;
            double *inner_qi = plan->d_inner_qbatch + qi * inner_stride;
            k_leaf_extract_b1<<<bl, 256, 0, plan->stream_compute>>>(
                plan->n, g_leaf_qi, plan->fft_stride[0], inner_qi);
        }
        if (!CUDA_OK(cudaGetLastError())) return false;
    } else {
        size_t bp_stride_local = (size_t)plan->N_tree * (size_t)(plan->B + 1);
        int threads_leaf = plan->B;
        if (threads_leaf > 1024) threads_leaf = 1024;
        k_leaf_extract_qbatch<<<qb * plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
            (const double * const *)d_a_ptrs, plan->n, plan->B, plan->nblocks,
            plan->d_block_prods_qbatch, bp_stride_local,
            plan->d_g_levels[0], plan->fft_stride[0], leaf_g_stride,
            plan->g_needed[0], plan->k, qb,
            plan->d_inner_qbatch, inner_stride);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    /* 7. Accumulate equity for all Q-points.
     * Upload weights and inv_vs to pre-allocated device buffers. */
    double h_weights[Q_BATCH_MAX];
    double h_inv_vs[Q_BATCH_MAX];
    for (int qi = 0; qi < qb; ++qi) {
        h_weights[qi] = pts[qi].w;
        h_inv_vs[qi] = exp(-pts[qi].logv);
    }
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_qb_weights, h_weights, (size_t)qb * sizeof(double),
                                 cudaMemcpyHostToDevice, plan->stream_compute))) return false;
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_qb_inv_vs, h_inv_vs, (size_t)qb * sizeof(double),
                                 cudaMemcpyHostToDevice, plan->stream_compute))) return false;

    int accum_total = qb * plan->n;
    int blocks_accum = (accum_total + threads - 1) / threads;
    k_accumulate_equity_qbatch<<<blocks_accum, threads, 0, plan->stream_compute>>>(
        plan->d_inner_qbatch, inner_stride,
        (const double * const *)d_a_ptrs,
        plan->d_S_sorted, plan->d_sort_perm,
        plan->n, plan->d_qb_weights, plan->d_qb_inv_vs,
        qb, plan->d_equity);
    if (!CUDA_OK(cudaGetLastError())) return false;

    return true;
}

static bool run_hybrid_single_q(GpuPlan *plan, int a_buf_idx,
                                double logv, double w,
                                bool skip_compute_a, bool fast_mode,
                                double *block_ns, double *tree_build_ns,
                                double *tree_prop_cached_ns,
                                double *tree_prop_recomp_ns,
                                double *leaf_ns, double *accum_ns) {
    int threads = 256;
    int blocks_n = (plan->n + threads - 1) / threads;

    int curr = a_buf_idx;
    if (plan->opts.enable_graphs && plan->graph_ready[curr]) {
        double scale = w * exp(-logv);
        if (!CUDA_OK(cudaMemcpyAsync(plan->d_graph_logv[curr], &logv, sizeof(double),
                                     cudaMemcpyHostToDevice, plan->stream_compute))) return false;
        if (!CUDA_OK(cudaMemcpyAsync(plan->d_graph_scale[curr], &scale, sizeof(double),
                                     cudaMemcpyHostToDevice, plan->stream_compute))) return false;
        if (!CUDA_OK(cudaGraphLaunch(plan->graph_exec[curr], plan->stream_compute))) return false;
        /* No sync here -- caller syncs once at the end */
        return true;
    }

    /* ── Fast path: launch all kernels without intermediate syncs ── */
    if (fast_mode) {
        if (!skip_compute_a) {
            k_compute_a<<<blocks_n, threads, 0, plan->stream_compute>>>(
                plan->d_S_sorted, plan->d_a_sorted[curr], plan->n, logv);
        }

        /* Block build (or B=1 leaf setup) */
        if (plan->B <= 1) {
            int bl = (plan->N_tree + 255) / 256;
            k_set_leaves_b1<<<bl, 256, 0, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->N_tree,
                plan->fft_stride[0], plan->d_poly_levels[0]);
        } else {
            int threads_block = 256;
            size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);
            k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->B,
                plan->nblocks, plan->N_tree, plan->fft_stride[0],
                plan->d_poly_levels[0], plan->d_block_prods);
        }

        for (int ell = 1; ell < plan->L - 1; ++ell) {
            auto &lp = plan->levels[ell];
            if (!lp.use_fft) { if (!run_build_level_schoolbook(plan, ell)) return false; }
            else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_build_level_fused(plan, ell)) return false; }
            else { if (!run_build_level_fft(plan, ell)) return false; }
        }

        int top = plan->L - 1;
        int root_gsz = plan->fft_stride[top];
        int blocks_root = (root_gsz + threads - 1) / threads;
        k_set_root_g<<<blocks_root, threads, 0, plan->stream_compute>>>(
            plan->d_g_levels[top], root_gsz, plan->d_payout, plan->k);

        for (int ell = top; ell >= 1; --ell) {
            auto &lp = plan->levels[ell];
            if (!lp.use_fft) { if (!run_prop_level_schoolbook(plan, ell)) return false; }
            else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_prop_level_fused(plan, ell)) return false; }
            else { if (!run_prop_level_fft(plan, ell)) return false; }
        }

        /* Leaf extract (or B=1 direct read) */
        if (plan->B <= 1) {
            int bl = (plan->n + 255) / 256;
            k_leaf_extract_b1<<<bl, 256, 0, plan->stream_compute>>>(
                plan->n, plan->d_g_levels[0], plan->fft_stride[0], plan->d_inner_sorted);
        } else {
            int threads_leaf = plan->B;
            if (threads_leaf > 1024) threads_leaf = 1024;
            k_leaf_extract<<<plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->B, plan->nblocks,
                plan->d_block_prods, plan->d_g_levels[0], plan->fft_stride[0],
                plan->g_needed[0], plan->k, plan->d_inner_sorted);
        }

        double inv_v = exp(-logv);
        k_accumulate_equity<<<blocks_n, threads, 0, plan->stream_compute>>>(
            plan->d_inner_sorted, plan->d_a_sorted[curr], plan->d_S_sorted,
            plan->d_sort_perm, plan->n, w, inv_v, plan->d_equity);
        /* No sync -- caller syncs once at the end */
        return true;
    }

    /* ── Instrumented path: per-stage timing with syncs ── */
    if (!skip_compute_a) {
        k_compute_a<<<blocks_n, threads, 0, plan->stream_compute>>>(
            plan->d_S_sorted, plan->d_a_sorted[curr], plan->n, logv);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    /* Block build (or B=1 leaf setup) */
    double t0 = now_ns_host();
    if (plan->B <= 1) {
        int bl = (plan->N_tree + 255) / 256;
        k_set_leaves_b1<<<bl, 256, 0, plan->stream_compute>>>(
            plan->d_a_sorted[curr], plan->n, plan->N_tree,
            plan->fft_stride[0], plan->d_poly_levels[0]);
        if (!CUDA_OK(cudaGetLastError())) return false;
    } else {
        int threads_block = 256;
        size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);
        k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_compute>>>(
            plan->d_a_sorted[curr], plan->n, plan->B,
            plan->nblocks, plan->N_tree, plan->fft_stride[0],
            plan->d_poly_levels[0], plan->d_block_prods);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
    *block_ns += (now_ns_host() - t0);

    for (int ell = 1; ell < plan->L - 1; ++ell) {
        auto &lp = plan->levels[ell];
        t0 = now_ns_host();
        bool ok = false;
        if (!lp.use_fft) ok = run_build_level_schoolbook(plan, ell);
        else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) ok = run_build_level_fused(plan, ell);
        else ok = run_build_level_fft(plan, ell);
        if (!ok) return false;
        if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
        *tree_build_ns += (now_ns_host() - t0);
    }

    int top = plan->L - 1;
    int root_gsz = plan->fft_stride[top];
    int blocks_root = (root_gsz + threads - 1) / threads;
    k_set_root_g<<<blocks_root, threads, 0, plan->stream_compute>>>(
        plan->d_g_levels[top], root_gsz, plan->d_payout, plan->k);
    if (!CUDA_OK(cudaGetLastError())) return false;

    for (int ell = top; ell >= 1; --ell) {
        auto &lp = plan->levels[ell];
        t0 = now_ns_host();
        bool ok = false;
        if (!lp.use_fft) ok = run_prop_level_schoolbook(plan, ell);
        else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) ok = run_prop_level_fused(plan, ell);
        else ok = run_prop_level_fft(plan, ell);
        if (!ok) return false;
        if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
        double dt = now_ns_host() - t0;
        if (lp.use_fft && lp.cache_fft) *tree_prop_cached_ns += dt;
        else if (lp.use_fft) *tree_prop_recomp_ns += dt;
        else *tree_prop_cached_ns += dt;
    }

    /* Leaf extract (or B=1 direct read) */
    t0 = now_ns_host();
    if (plan->B <= 1) {
        int bl = (plan->n + 255) / 256;
        k_leaf_extract_b1<<<bl, 256, 0, plan->stream_compute>>>(
            plan->n, plan->d_g_levels[0], plan->fft_stride[0], plan->d_inner_sorted);
        if (!CUDA_OK(cudaGetLastError())) return false;
    } else {
        int threads_leaf = plan->B;
        if (threads_leaf > 1024) threads_leaf = 1024;
        k_leaf_extract<<<plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
            plan->d_a_sorted[curr], plan->n, plan->B, plan->nblocks,
            plan->d_block_prods, plan->d_g_levels[0], plan->fft_stride[0],
            plan->g_needed[0], plan->k, plan->d_inner_sorted);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
    *leaf_ns += (now_ns_host() - t0);

    t0 = now_ns_host();
    double inv_v = exp(-logv);
    k_accumulate_equity<<<blocks_n, threads, 0, plan->stream_compute>>>(
        plan->d_inner_sorted, plan->d_a_sorted[curr], plan->d_S_sorted,
        plan->d_sort_perm, plan->n, w, inv_v, plan->d_equity);
    if (!CUDA_OK(cudaGetLastError())) return false;
    if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
    *accum_ns += (now_ns_host() - t0);

    return true;
}

static bool create_graph_stub(GpuPlan *plan) {
    if (!plan->opts.enable_graphs) return true;
    /* Phase A2: graphs now work with all memory strategies because cache buffers
     * are always allocated (never freed during execution). Validity is tracked
     * via fft_cache_valid instead. */

    int threads = 256;
    int blocks_n = (plan->n + threads - 1) / threads;
    int top = plan->L - 1;
    int root_gsz = plan->fft_stride[top];
    int blocks_root = (root_gsz + threads - 1) / threads;
    int threads_leaf = plan->B;
    if (threads_leaf > 1024) threads_leaf = 1024;
    int threads_block = 256;
    size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);

    for (int curr = 0; curr < 2; ++curr) {
        if (plan->graph_ready[curr]) continue;
        if (!CUDA_OK(cudaMemsetAsync(plan->d_graph_logv[curr], 0, sizeof(double), plan->stream_compute))) return false;
        if (!CUDA_OK(cudaMemsetAsync(plan->d_graph_scale[curr], 0, sizeof(double), plan->stream_compute))) return false;
        if (!CUDA_OK(cudaStreamBeginCapture(plan->stream_compute, cudaStreamCaptureModeGlobal))) return false;

        k_compute_a_from_ptr<<<blocks_n, threads, 0, plan->stream_compute>>>(
            plan->d_S_sorted, plan->d_a_sorted[curr], plan->n, plan->d_graph_logv[curr]);
        if (!CUDA_OK(cudaGetLastError())) return false;

        if (plan->B <= 1) {
            int bl = (plan->N_tree + 255) / 256;
            k_set_leaves_b1<<<bl, 256, 0, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->N_tree,
                plan->fft_stride[0], plan->d_poly_levels[0]);
        } else {
            k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->B,
                plan->nblocks, plan->N_tree, plan->fft_stride[0],
                plan->d_poly_levels[0], plan->d_block_prods);
        }
        if (!CUDA_OK(cudaGetLastError())) return false;

        for (int ell = 1; ell < plan->L - 1; ++ell) {
            auto &lp = plan->levels[ell];
            bool ok = false;
            if (!lp.use_fft) ok = run_build_level_schoolbook(plan, ell);
            else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) ok = run_build_level_fused(plan, ell);
            else ok = run_build_level_fft(plan, ell);
            if (!ok) return false;
        }

        k_set_root_g<<<blocks_root, threads, 0, plan->stream_compute>>>(
            plan->d_g_levels[top], root_gsz, plan->d_payout, plan->k);
        if (!CUDA_OK(cudaGetLastError())) return false;

        for (int ell = top; ell >= 1; --ell) {
            auto &lp = plan->levels[ell];
            bool ok = false;
            if (!lp.use_fft) ok = run_prop_level_schoolbook(plan, ell);
            else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) ok = run_prop_level_fused(plan, ell);
            else ok = run_prop_level_fft(plan, ell);
            if (!ok) return false;
        }

        if (plan->B <= 1) {
            int bl = (plan->n + 255) / 256;
            k_leaf_extract_b1<<<bl, 256, 0, plan->stream_compute>>>(
                plan->n, plan->d_g_levels[0], plan->fft_stride[0], plan->d_inner_sorted);
        } else {
            k_leaf_extract<<<plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
                plan->d_a_sorted[curr], plan->n, plan->B, plan->nblocks,
                plan->d_block_prods, plan->d_g_levels[0], plan->fft_stride[0],
                plan->g_needed[0], plan->k, plan->d_inner_sorted);
        }
        if (!CUDA_OK(cudaGetLastError())) return false;

        k_accumulate_equity_scaled<<<blocks_n, threads, 0, plan->stream_compute>>>(
            plan->d_inner_sorted, plan->d_a_sorted[curr], plan->d_S_sorted,
            plan->d_sort_perm, plan->n, plan->d_graph_scale[curr], plan->d_equity);
        if (!CUDA_OK(cudaGetLastError())) return false;

        if (!CUDA_OK(cudaStreamEndCapture(plan->stream_compute, &plan->graph[curr]))) return false;
        if (!CUDA_OK(cudaGraphInstantiate(&plan->graph_exec[curr], plan->graph[curr], nullptr, nullptr, 0))) return false;
        plan->graph_ready[curr] = true;
    }
    return true;
}

}  // namespace

/* For small n, CPU is faster due to GPU plan creation overhead (~5-7ms).
 * Default crossover at n=1024; overridable via gpu_fft_config.h or env var. */
#ifndef GPU_CPU_CROSSOVER
#define GPU_CPU_CROSSOVER 0  /* GPU competitive at all n with arena allocator */
#endif

static int gpu_cpu_crossover() {
    static int val = -1;
    if (val < 0) {
        val = GPU_CPU_CROSSOVER;
        const char *env = getenv("ICM_GPU_CPU_CROSSOVER");
        if (env && env[0]) {
            int v = atoi(env);
            if (v >= 0) val = v;
        }
    }
    return val;
}

extern "C" {

int icm_gpu_init(int device_id) {
    int count = 0;
    if (!CUDA_OK(cudaGetDeviceCount(&count))) return 0;
    if (count <= 0) {
        set_last_errorf("No CUDA devices found");
        return 0;
    }
    if (device_id < 0) device_id = 0;
    if (device_id >= count) {
        set_last_errorf("Invalid CUDA device id %d (count=%d)", device_id, count);
        return 0;
    }
    if (!CUDA_OK(cudaSetDevice(device_id))) return 0;
    int max_optin = 0;
    if (CUDA_OK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id)) &&
        max_optin > 0) {
        cudaFuncSetAttribute(k_block_build, cudaFuncAttributeMaxDynamicSharedMemorySize, max_optin);
    }
    g_cuda_device = device_id;
    return 1;
}

void icm_gpu_shutdown(void) {
    if (g_cuda_device >= 0) {
        cudaSetDevice(g_cuda_device);
        cudaDeviceSynchronize();
    }
}

const char *icm_gpu_last_error(void) {
    return g_last_error.c_str();
}

IcmGpuPlan *icm_gpu_plan_create(int n, const double *S, int k, const IcmGpuOptions *opts) {
    if (g_cuda_device < 0) {
        set_last_errorf("icm_gpu_init must be called before icm_gpu_plan_create");
        return nullptr;
    }
    if (n <= 0 || !S || k <= 0 || k > n) {
        set_last_errorf("Invalid plan args n=%d k=%d", n, k);
        return nullptr;
    }

    auto *plan = new GpuPlan();
    plan->n = n;
    plan->k = k;
    plan->opts = IcmGpuOptions{};
    plan->opts.force_uncached_fused_levels = -1;
    plan->opts.force_uncached_cufft_levels = -1;
    if (opts) plan->opts = *opts;
    plan->opts.device_id = g_cuda_device;
    plan->opts.enable_q_pipeline = plan->opts.enable_q_pipeline ? 1 : 0;
    plan->S_sorted.assign(S, S + n);

    double t_sort0 = now_ns_host();
    if (!device_sort_players(plan)) {
        destroy_plan(plan);
        return nullptr;
    }
    (void)t_sort0;

    if (!build_plan_metadata(plan)) {
        destroy_plan(plan);
        return nullptr;
    }

    /* Determine q_batch via cost model: pick QB that minimizes total time
     * subject to VRAM constraint. QB helps at upper tree levels where
     * nn[ell] is small (wider cuFFT batches improve GPU utilization).
     * At large n, nn[ell] is already large, so QB>1 adds VRAM cost
     * without proportional benefit. */
    {
        const char *qb_env = getenv("ICM_GPU_Q_BATCH");
        int qb_override = 0;
        if (qb_env && qb_env[0]) {
            int v = atoi(qb_env);
            if (v >= 1 && v <= Q_BATCH_MAX) qb_override = v;
        }

        /* Compute per-Q-point VRAM overhead */
        size_t per_q_bytes = 0;
        for (int ell = 0; ell < plan->L; ++ell)
            per_q_bytes += 2 * (size_t)plan->nn[ell] * plan->psz[ell] * sizeof(double);
        per_q_bytes += (size_t)plan->N_tree * (plan->B + 1) * sizeof(double);
        per_q_bytes += 2 * (size_t)plan->n * sizeof(double); /* a + inner */
        size_t budget = (size_t)((double)GPU_VRAM_BYTES * 0.60);

        int best_qb = 1;
        if (!plan->opts.enable_graphs && !qb_override) {
            /* Evaluate QB candidates: the benefit is proportional to how many
             * tree levels have nn[ell] < GPU_SM_COUNT (underutilized). */
            int underutil_levels = 0;
            for (int ell = 1; ell < plan->L - 1; ++ell) {
                if (plan->levels[ell].use_fft && plan->nn[ell] < GPU_SM_COUNT)
                    underutil_levels++;
            }
            /* If most levels already have enough parallelism, QB=1 is optimal */
            if (underutil_levels >= 2) {
                for (int qb_try = Q_BATCH_MAX; qb_try >= 2; qb_try /= 2) {
                    if ((size_t)qb_try * per_q_bytes <= budget) {
                        best_qb = qb_try;
                        break;
                    }
                }
            }
        }
        int qb = qb_override ? qb_override : best_qb;
        if (plan->opts.enable_graphs) qb = 1;
        plan->q_batch = qb;
    }

    if (!allocate_plan_device_memory(plan)) {
        destroy_plan(plan);
        return nullptr;
    }

    if (!create_graph_stub(plan)) {
        destroy_plan(plan);
        return nullptr;
    }

    plan->planned_peak_vram_bytes = plan->peak_vram_bytes;
    return reinterpret_cast<IcmGpuPlan *>(plan);
}

void icm_gpu_plan_destroy(IcmGpuPlan *plan_opaque) {
    auto *plan = reinterpret_cast<GpuPlan *>(plan_opaque);
    destroy_plan(plan);
}

int icm_gpu_plan_summary(const IcmGpuPlan *plan_opaque, IcmGpuPlanSummary *summary) {
    const auto *plan = reinterpret_cast<const GpuPlan *>(plan_opaque);
    if (!plan || !summary) return 0;
    memset(summary, 0, sizeof(*summary));
    summary->n = plan->n;
    summary->k = plan->k;
    summary->B = plan->B;
    summary->engine = plan->engine;
    summary->n_levels = plan->L;
    for (int ell = 1; ell < plan->L; ++ell) {
        if (!plan->levels[ell].use_fft) summary->n_tier1++;
        else if (plan->levels[ell].tier == GPU_TIER_FUSED) summary->n_tier2++;
        else summary->n_tier3++;
    }
    summary->q_batch = plan->q_batch;
    summary->planned_peak_vram_bytes = plan->planned_peak_vram_bytes;
    return 1;
}

double icm_gpu_equity_with_plan(IcmGpuPlan *plan_opaque, int Q,
                                const double *payout, double *equity,
                                IcmGpuRunStats *stats) {
    auto *plan = reinterpret_cast<GpuPlan *>(plan_opaque);
    if (!plan || !payout || !equity || Q <= 0) {
        set_last_errorf("icm_gpu_equity_with_plan invalid arguments");
        return -1.0;
    }
    if (!CUDA_OK(cudaSetDevice(g_cuda_device))) return -1.0;
    if (!CUDA_OK(cudaMemcpyAsync(plan->d_payout, payout, (size_t)plan->k * sizeof(double),
                                 cudaMemcpyHostToDevice, plan->stream_compute))) return -1.0;

    int threads = 256;
    int blocks = (plan->n + threads - 1) / threads;
    k_zero<<<blocks, threads, 0, plan->stream_compute>>>(plan->d_equity, (size_t)plan->n);
    if (!CUDA_OK(cudaGetLastError())) return -1.0;

    double Smax = 0.0;
    for (int i = 0; i < plan->n; ++i) if (plan->S_sorted[i] > Smax) Smax = plan->S_sorted[i];
    std::vector<QP> pts;
    make_nodes(Q, Smax, pts);

    double sort_ns = 0.0;
    double quad_ovh_ns = 0.0;
    double block_ns = 0.0;
    double tree_build_ns = 0.0;
    double tree_prop_cached_ns = 0.0;
    double tree_prop_recomp_ns = 0.0;
    double leaf_ns = 0.0;
    double accum_ns = 0.0;

    bool fast = !plan->opts.verbose;

    int qb = plan->q_batch;

    double t0 = now_ns_host();
    if (plan->opts.enable_graphs && (plan->graph_ready[0] || plan->graph_ready[1])) {
        /* Graph path: single-Q iteration (graphs are pre-captured) */
        for (int q = 0; q < Q; ++q) {
            if (pts[q].w == 0.0) continue;
            int curr = q & 1;
            if (!run_hybrid_single_q(plan, curr, pts[q].logv, pts[q].w, false, fast,
                                     &block_ns, &tree_build_ns, &tree_prop_cached_ns,
                                     &tree_prop_recomp_ns, &leaf_ns, &accum_ns)) {
                return -1.0;
            }
        }
    } else if (qb > 1) {
        /* ── Q-batched path: process qb Q-points per tree traversal ── */
        /* First, collect non-zero-weight Q-points */
        std::vector<QP> active_pts;
        active_pts.reserve(Q);
        for (int q = 0; q < Q; ++q) {
            if (pts[q].w != 0.0) active_pts.push_back(pts[q]);
        }

        int n_active = (int)active_pts.size();
        /* Process in batches of qb */
        for (int q = 0; q < n_active; q += qb) {
            int batch_sz = std::min(qb, n_active - q);
            if (batch_sz == qb) {
                /* Full batch */
                if (!run_hybrid_batched_q(plan, &active_pts[q], qb)) {
                    return -1.0;
                }
            } else {
                /* Remainder: pad with zero-weight Q-points to make a full batch.
                 * The zero-weight points will contribute nothing to equity
                 * (weight=0 in accumulate), but the tree kernels still execute
                 * correctly since cuFFT plans are sized for full q_batch. */
                QP padded[Q_BATCH_MAX];
                for (int r = 0; r < batch_sz; ++r) {
                    padded[r] = active_pts[q + r];
                }
                /* Pad remaining slots: use same logv as first point (arbitrary),
                 * but weight=0 so they don't contribute to equity. */
                for (int r = batch_sz; r < qb; ++r) {
                    padded[r].logv = active_pts[q].logv;
                    padded[r].w = 0.0;
                }
                if (!run_hybrid_batched_q(plan, padded, qb)) {
                    return -1.0;
                }
            }
        }
    } else if (plan->opts.enable_q_pipeline && plan->d_poly_leaves_alt) {
        /* Multi-stream pipeline: overlap compute_a + block_build of q+1
         * with tree_build + propagation + leaf_extract + accumulate of q.
         * Double-buffered leaf level (d_poly_levels[0]) and d_block_prods
         * prevent write-after-read hazards between streams. */
        double *leaf_bufs[2] = { plan->d_poly_levels[0], plan->d_poly_leaves_alt };
        double *bp_bufs[2]   = { plan->d_block_prods,    plan->d_block_prods_alt };
        double *orig_poly_levels_0 = plan->d_poly_levels[0];
        double *orig_block_prods   = plan->d_block_prods;
        int buf_idx = 0;
        bool pipeline_ok = true;

        int q_start = 0;
        while (q_start < Q && pts[q_start].w == 0.0) ++q_start;

        if (q_start < Q) {
            /* Pre-launch compute_a + block_build for q_start on stream_aux */
            int curr = q_start & 1;
            k_compute_a<<<blocks, threads, 0, plan->stream_aux>>>(
                plan->d_S_sorted, plan->d_a_sorted[curr], plan->n, pts[q_start].logv);
            if (!CUDA_OK(cudaGetLastError())) { pipeline_ok = false; goto pipeline_cleanup; }

            if (plan->B <= 1) {
                int bl = (plan->N_tree + 255) / 256;
                k_set_leaves_b1<<<bl, 256, 0, plan->stream_aux>>>(
                    plan->d_a_sorted[curr], plan->n, plan->N_tree,
                    plan->fft_stride[0], leaf_bufs[buf_idx]);
            } else {
                int threads_block = 256;
                size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);
                k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_aux>>>(
                    plan->d_a_sorted[curr], plan->n, plan->B,
                    plan->nblocks, plan->N_tree, plan->fft_stride[0],
                    leaf_bufs[buf_idx], bp_bufs[buf_idx]);
            }
            if (!CUDA_OK(cudaGetLastError())) { pipeline_ok = false; goto pipeline_cleanup; }
            if (!CUDA_OK(cudaEventRecord(plan->evt_a_ready[curr], plan->stream_aux))) { pipeline_ok = false; goto pipeline_cleanup; }
        }

        for (int q = q_start; q < Q; ++q) {
            if (pts[q].w == 0.0) continue;
            int curr = q & 1;

            /* Wait for this Q-point's compute_a + block_build to finish */
            if (!CUDA_OK(cudaStreamWaitEvent(plan->stream_compute, plan->evt_a_ready[curr], 0))) { pipeline_ok = false; break; }

            /* Point d_poly_levels[0] and d_block_prods at this Q's buffer */
            plan->d_poly_levels[0] = leaf_bufs[buf_idx];
            if (plan->B > 1) plan->d_block_prods = bp_bufs[buf_idx];

            /* Find next non-zero Q-point */
            int qn = q + 1;
            while (qn < Q && pts[qn].w == 0.0) ++qn;
            int next_buf = 1 - buf_idx;

            /* Launch next Q's compute_a + block_build on stream_aux (overlapped) */
            if (qn < Q) {
                /* Wait for current Q's propagation to finish reading poly_levels[0] */
                if (!CUDA_OK(cudaStreamWaitEvent(plan->stream_aux, plan->evt_prop_done, 0))) { pipeline_ok = false; break; }
                int next = qn & 1;
                k_compute_a<<<blocks, threads, 0, plan->stream_aux>>>(
                    plan->d_S_sorted, plan->d_a_sorted[next], plan->n, pts[qn].logv);
                if (!CUDA_OK(cudaGetLastError())) { pipeline_ok = false; break; }

                if (plan->B <= 1) {
                    int bl = (plan->N_tree + 255) / 256;
                    k_set_leaves_b1<<<bl, 256, 0, plan->stream_aux>>>(
                        plan->d_a_sorted[next], plan->n, plan->N_tree,
                        plan->fft_stride[0], leaf_bufs[next_buf]);
                } else {
                    int threads_block = 256;
                    size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);
                    k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_aux>>>(
                        plan->d_a_sorted[next], plan->n, plan->B,
                        plan->nblocks, plan->N_tree, plan->fft_stride[0],
                        leaf_bufs[next_buf], bp_bufs[next_buf]);
                }
                if (!CUDA_OK(cudaGetLastError())) { pipeline_ok = false; break; }
                if (!CUDA_OK(cudaEventRecord(plan->evt_a_ready[next], plan->stream_aux))) { pipeline_ok = false; break; }
            }

            /* Run tree_build + propagate + leaf_extract + accumulate on stream_compute.
             * block_build already done (skip_compute_a=true, and we also skip block_build
             * by calling the tree/prop/leaf/accum stages directly). */

            /* Tree build: levels 1..L-2 (reads poly_levels[0] at level 1) */
            for (int ell = 1; ell < plan->L - 1; ++ell) {
                auto &lp = plan->levels[ell];
                if (!lp.use_fft) { if (!run_build_level_schoolbook(plan, ell)) { pipeline_ok = false; break; } }
                else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_build_level_fused(plan, ell)) { pipeline_ok = false; break; } }
                else { if (!run_build_level_fft(plan, ell)) { pipeline_ok = false; break; } }
            }
            if (!pipeline_ok) break;

            /* Set root g */
            int top = plan->L - 1;
            int root_gsz = plan->fft_stride[top];
            int blocks_root = (root_gsz + threads - 1) / threads;
            k_set_root_g<<<blocks_root, threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[top], root_gsz, plan->d_payout, plan->k);

            /* Propagate: levels top..1 (reads poly_levels at each level) */
            for (int ell = top; ell >= 1; --ell) {
                auto &lp = plan->levels[ell];
                if (!lp.use_fft) { if (!run_prop_level_schoolbook(plan, ell)) { pipeline_ok = false; break; } }
                else if (lp.tier == GPU_TIER_FUSED && plan->opts.use_cufftdx) { if (!run_prop_level_fused(plan, ell)) { pipeline_ok = false; break; } }
                else { if (!run_prop_level_fft(plan, ell)) { pipeline_ok = false; break; } }
            }
            if (!pipeline_ok) break;

            /* Signal that propagation is done reading poly_levels[0].
             * stream_aux can now safely overwrite the alternate buffer. */
            if (!CUDA_OK(cudaEventRecord(plan->evt_prop_done, plan->stream_compute))) { pipeline_ok = false; break; }

            /* Leaf extract + accumulate (still on stream_compute) */
            if (plan->B <= 1) {
                int bl = (plan->n + 255) / 256;
                k_leaf_extract_b1<<<bl, 256, 0, plan->stream_compute>>>(
                    plan->n, plan->d_g_levels[0], plan->fft_stride[0], plan->d_inner_sorted);
            } else {
                int threads_leaf = plan->B;
                if (threads_leaf > 1024) threads_leaf = 1024;
                k_leaf_extract<<<plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
                    plan->d_a_sorted[curr], plan->n, plan->B, plan->nblocks,
                    plan->d_block_prods, plan->d_g_levels[0], plan->fft_stride[0],
                    plan->g_needed[0], plan->k, plan->d_inner_sorted);
            }

            double inv_v = exp(-pts[q].logv);
            k_accumulate_equity<<<blocks, threads, 0, plan->stream_compute>>>(
                plan->d_inner_sorted, plan->d_a_sorted[curr], plan->d_S_sorted,
                plan->d_sort_perm, plan->n, pts[q].w, inv_v, plan->d_equity);

            buf_idx = next_buf;
        }

pipeline_cleanup:
        /* Restore original pointers so destroy_plan frees the right buffers */
        plan->d_poly_levels[0] = orig_poly_levels_0;
        if (plan->B > 1) plan->d_block_prods = orig_block_prods;
        if (!pipeline_ok) return -1.0;
    } else {
        for (int q = 0; q < Q; ++q) {
            if (pts[q].w == 0.0) continue;
            int curr = q & 1;
            if (!run_hybrid_single_q(plan, curr, pts[q].logv, pts[q].w, false, fast,
                                     &block_ns, &tree_build_ns, &tree_prop_cached_ns,
                                     &tree_prop_recomp_ns, &leaf_ns, &accum_ns)) {
                return -1.0;
            }
        }
    }
    if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return -1.0;
    double total_ns = now_ns_host() - t0;

    if (!CUDA_OK(cudaMemcpy(equity, plan->d_equity, (size_t)plan->n * sizeof(double),
                            cudaMemcpyDeviceToHost))) return -1.0;

    if (stats) {
        memset(stats, 0, sizeof(*stats));
        stats->total_ns = total_ns;
        stats->sort_ns = sort_ns;
        stats->quadrature_overhead_ns = std::max(0.0, quad_ovh_ns);
        stats->block_build_ns = block_ns;
        stats->tree_build_ns = tree_build_ns;
        stats->tree_propagate_cached_ns = tree_prop_cached_ns;
        stats->tree_propagate_recomputed_ns = tree_prop_recomp_ns;
        stats->leaf_extract_ns = leaf_ns;
        stats->peak_vram_bytes = std::max(plan->peak_vram_bytes, plan->planned_peak_vram_bytes);
        stats->engine = plan->engine;
        stats->B = plan->B;
        stats->uncached_fused_levels = plan->uncached_fused_levels;
        stats->uncached_cufft_levels = plan->uncached_cufft_levels;
    }
    return total_ns;
}

double icm_gpu_equity(int n, const double *S, int Q,
                      const double *payout, int k,
                      double *equity, const IcmGpuOptions *opts,
                      IcmGpuRunStats *stats) {
    if (g_cuda_device < 0) {
        if (!icm_gpu_init(0)) return -1.0;
    }

    int sk_max = single_kernel_max_n(k);
    if (n <= sk_max && n > 0 && k > 0 && k <= n) {
        /* Single-kernel shared-memory engine: no plan, no cuFFT, no HBM intermediates.
         * Entire tree computation in one kernel launch. */
        double t0_sk = now_ns_host();

        /* Sort players on host (same as plan creation) */
        std::vector<std::pair<double, int>> pairs(n);
        for (int i = 0; i < n; i++) pairs[i] = {S[i], i};
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
            return a.first > b.first ? true : (a.first < b.first ? false : a.second < b.second);
        });
        std::vector<double> S_sorted(n);
        std::vector<int> sort_perm(n);
        for (int i = 0; i < n; i++) {
            S_sorted[i] = pairs[i].first;
            sort_perm[i] = pairs[i].second;
        }

        /* Compute tree geometry (B=1) */
        int N = 1;
        while (N < n) N <<= 1;
        int L = 0;
        { int tmp = N; while (tmp > 1) { tmp >>= 1; L++; } L++; }
        std::vector<int> nn(L), psz(L), g_needed(L);
        std::vector<size_t> plev_off(L);
        nn[0] = N;
        for (int ell = 1; ell < L; ell++) nn[ell] = N >> ell;
        size_t off = 0;
        for (int ell = 0; ell < L; ell++) {
            long d = 1L << (ell + 1);  /* B=1: leaf_degree=1 */
            psz[ell] = (d > k) ? k : (int)d;
            plev_off[ell] = off;
            off += (size_t)nn[ell] * psz[ell];
        }
        int total_poly = (int)off;
        g_needed[0] = 1;  /* B=1: only need g_leaf[0] */
        for (int ell = 1; ell < L; ell++) {
            int need = g_needed[ell - 1] + psz[ell - 1] - 1;
            g_needed[ell] = (need < psz[ell]) ? need : psz[ell];
        }
        int max_g = 0;
        for (int ell = 0; ell < L; ell++) {
            int sz = nn[ell] * psz[ell];
            if (sz > max_g) max_g = sz;
        }

        size_t shmem_bytes = ((size_t)total_poly + 2 * (size_t)max_g) * sizeof(double);
        if (shmem_bytes > 200 * 1024) {
            /* Too large for shared memory, fall back to standard GPU path */
            goto standard_gpu_path;
        }

        /* Compute quadrature points */
        double Smax = S_sorted[0];
        std::vector<QP> pts;
        make_nodes(Q, Smax, pts);
        int active_Q = 0;
        for (int q = 0; q < Q; q++) if (pts[q].w != 0.0) active_Q++;

        /* Upload to device: small buffers, few mallocs */
        double *d_S = nullptr, *d_payout = nullptr, *d_equity_buf = nullptr;
        double *d_logv = nullptr, *d_weights = nullptr;
        int *d_perm = nullptr, *d_nn = nullptr, *d_psz = nullptr, *d_g_needed = nullptr;
        size_t *d_plev_off_d = nullptr;

        size_t arena_sz = 0;
        arena_sz += 256; arena_sz += n * sizeof(double);         /* S */
        arena_sz += 256; arena_sz += n * sizeof(int);            /* perm */
        arena_sz += 256; arena_sz += k * sizeof(double);         /* payout */
        arena_sz += 256; arena_sz += n * sizeof(double);         /* equity */
        arena_sz += 256; arena_sz += Q * sizeof(double);         /* logv */
        arena_sz += 256; arena_sz += Q * sizeof(double);         /* weights */
        arena_sz += 256; arena_sz += L * sizeof(int);            /* nn */
        arena_sz += 256; arena_sz += L * sizeof(int);            /* psz */
        arena_sz += 256; arena_sz += L * sizeof(int);            /* g_needed */
        arena_sz += 256; arena_sz += L * sizeof(size_t);         /* plev_off */
        char *sk_arena = nullptr;
        if (!CUDA_OK(cudaMalloc(&sk_arena, arena_sz))) return -1.0;
        if (!CUDA_OK(cudaMemset(sk_arena, 0, arena_sz))) { cudaFree(sk_arena); return -1.0; }

        size_t sk_off = 0;
        #define SKP(ptr, type, sz) do { sk_off = (sk_off + 255) & ~(size_t)255; (ptr) = (type)(sk_arena + sk_off); sk_off += (sz); } while(0)
        SKP(d_S, double*, n * sizeof(double));
        SKP(d_perm, int*, n * sizeof(int));
        SKP(d_payout, double*, k * sizeof(double));
        SKP(d_equity_buf, double*, n * sizeof(double));
        SKP(d_logv, double*, Q * sizeof(double));
        SKP(d_weights, double*, Q * sizeof(double));
        SKP(d_nn, int*, L * sizeof(int));
        SKP(d_psz, int*, L * sizeof(int));
        SKP(d_g_needed, int*, L * sizeof(int));
        SKP(d_plev_off_d, size_t*, L * sizeof(size_t));
        #undef SKP

        /* Upload data */
        std::vector<double> h_logv(Q), h_weights(Q);
        for (int q = 0; q < Q; q++) { h_logv[q] = pts[q].logv; h_weights[q] = pts[q].w; }
        cudaMemcpy(d_S, S_sorted.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_perm, sort_perm.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_payout, payout, k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_logv, h_logv.data(), Q * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, h_weights.data(), Q * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nn, nn.data(), L * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_psz, psz.data(), L * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g_needed, g_needed.data(), L * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_plev_off_d, plev_off.data(), L * sizeof(size_t), cudaMemcpyHostToDevice);

        /* Set shared memory limit for the kernel */
        cudaFuncSetAttribute(k_icm_single_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_bytes);

        /* Launch: one block per Q-point, 256 threads per block */
        int nthreads = 256;
        if (n < 256) nthreads = ((n + 31) / 32) * 32;  /* at least 1 warp per player */
        k_icm_single_kernel<<<Q, nthreads, shmem_bytes>>>(
            d_S, d_perm, n, Q, d_logv, d_weights, d_payout, k, d_equity_buf,
            N, L, d_nn, d_psz, d_g_needed, d_plev_off_d, total_poly, max_g);
        cudaDeviceSynchronize();

        /* Download result */
        memset(equity, 0, n * sizeof(double));
        cudaMemcpy(equity, d_equity_buf, n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(sk_arena);

        double total_ns = now_ns_host() - t0_sk;
        if (stats) {
            memset(stats, 0, sizeof(*stats));
            stats->total_ns = total_ns;
            stats->engine = 2; /* single-kernel engine */
        }
        return total_ns;
    }

standard_gpu_path:
    IcmGpuPlan *plan = icm_gpu_plan_create(n, S, k, opts);
    if (!plan) return -1.0;
    double t = icm_gpu_equity_with_plan(plan, Q, payout, equity, stats);
    icm_gpu_plan_destroy(plan);
    return t;
}

int icm_gpu_measure_fused_pair_ns(int fft_n, int batch, int quick,
                                  double *build_ns_out, double *corr_ns_out) {
    if (!build_ns_out || !corr_ns_out || fft_n <= 0 || batch <= 0) return 0;
    *build_ns_out = 0.0;
    *corr_ns_out = 0.0;
    if (g_cuda_device < 0) {
        if (!icm_gpu_init(0)) return 0;
    }
    if (!CUDA_OK(cudaSetDevice(g_cuda_device))) return 0;
#if !ICM_HAVE_CUFFTDX
    (void)quick;
    return 0;
#else
    if (!is_cufftdx_supported_fft_n(fft_n)) return 0;

    int nparents = batch;
    int cps = fft_n;
    int pps = fft_n;
    int len_g = fft_n;
    int len_P = fft_n;
    int len_out = fft_n;

    double *d_child = nullptr;
    double *d_parent = nullptr;
    double *d_g_parent = nullptr;
    double *d_child_poly = nullptr;
    double *d_g_child = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t e0 = nullptr;
    cudaEvent_t e1 = nullptr;

    int warmup = quick ? 1 : 3;
    int reps = quick ? 4 : 12;
    const char *warm_env = getenv("ICM_GPU_CALIB_WARMUP");
    if (warm_env && warm_env[0]) {
        int w = atoi(warm_env);
        if (w >= 0 && w <= 32) warmup = w;
    }
    const char *rep_env = getenv("ICM_GPU_CALIB_MIN_REPS");
    if (rep_env && rep_env[0]) {
        int r = atoi(rep_env);
        if (r > reps) reps = r;
    }
    std::vector<double> build_samples;
    std::vector<double> corr_samples;
    build_samples.reserve((size_t)reps);
    corr_samples.reserve((size_t)reps);

    size_t child_bytes = (size_t)(2 * nparents) * (size_t)cps * sizeof(double);
    size_t parent_bytes = (size_t)nparents * (size_t)pps * sizeof(double);
    size_t g_parent_bytes = (size_t)nparents * (size_t)fft_n * sizeof(double);
    size_t g_child_bytes = (size_t)(2 * nparents) * (size_t)fft_n * sizeof(double);

    if (!CUDA_OK(cudaStreamCreate(&stream))) goto fail;
    if (!CUDA_OK(cudaMalloc(&d_child, child_bytes))) goto fail;
    if (!CUDA_OK(cudaMalloc(&d_parent, parent_bytes))) goto fail;
    if (!CUDA_OK(cudaMalloc(&d_g_parent, g_parent_bytes))) goto fail;
    if (!CUDA_OK(cudaMalloc(&d_child_poly, child_bytes))) goto fail;
    if (!CUDA_OK(cudaMalloc(&d_g_child, g_child_bytes))) goto fail;
    if (!CUDA_OK(cudaMemsetAsync(d_child, 1, child_bytes, stream))) goto fail;
    if (!CUDA_OK(cudaMemsetAsync(d_parent, 0, parent_bytes, stream))) goto fail;
    if (!CUDA_OK(cudaMemsetAsync(d_g_parent, 1, g_parent_bytes, stream))) goto fail;
    if (!CUDA_OK(cudaMemsetAsync(d_child_poly, 2, child_bytes, stream))) goto fail;
    if (!CUDA_OK(cudaMemsetAsync(d_g_child, 0, g_child_bytes, stream))) goto fail;
    if (!CUDA_OK(cudaStreamSynchronize(stream))) goto fail;
    if (!CUDA_OK(cudaEventCreate(&e0))) goto fail;
    if (!CUDA_OK(cudaEventCreate(&e1))) goto fail;

    for (int i = 0; i < warmup; ++i) {
        if (!launch_cufftdx_build_dispatch(fft_n, d_child, cps, d_parent, pps, nparents,
                                           1.0 / (double)fft_n, stream)) goto fail;
    }
    if (!CUDA_OK(cudaStreamSynchronize(stream))) goto fail;
    for (int i = 0; i < reps; ++i) {
        if (!CUDA_OK(cudaEventRecord(e0, stream))) goto fail;
        if (!launch_cufftdx_build_dispatch(fft_n, d_child, cps, d_parent, pps, nparents,
                                           1.0 / (double)fft_n, stream)) goto fail;
        if (!CUDA_OK(cudaEventRecord(e1, stream))) goto fail;
        if (!CUDA_OK(cudaEventSynchronize(e1))) goto fail;
        float ms = 0.0f;
        if (!CUDA_OK(cudaEventElapsedTime(&ms, e0, e1))) goto fail;
        build_samples.push_back((double)ms * 1e6);
    }

    for (int i = 0; i < warmup; ++i) {
        if (!launch_cufftdx_corr_dispatch(fft_n,
                                          d_g_parent, fft_n, len_g,
                                          d_child_poly, cps, len_P,
                                          d_g_child, fft_n, len_out, nparents,
                                          1.0 / (double)fft_n, stream)) goto fail;
    }
    if (!CUDA_OK(cudaStreamSynchronize(stream))) goto fail;
    for (int i = 0; i < reps; ++i) {
        if (!CUDA_OK(cudaEventRecord(e0, stream))) goto fail;
        if (!launch_cufftdx_corr_dispatch(fft_n,
                                          d_g_parent, fft_n, len_g,
                                          d_child_poly, cps, len_P,
                                          d_g_child, fft_n, len_out, nparents,
                                          1.0 / (double)fft_n, stream)) goto fail;
        if (!CUDA_OK(cudaEventRecord(e1, stream))) goto fail;
        if (!CUDA_OK(cudaEventSynchronize(e1))) goto fail;
        float ms = 0.0f;
        if (!CUDA_OK(cudaEventElapsedTime(&ms, e0, e1))) goto fail;
        corr_samples.push_back((double)ms * 1e6);
    }

    std::sort(build_samples.begin(), build_samples.end());
    std::sort(corr_samples.begin(), corr_samples.end());
    if (build_samples.empty() || corr_samples.empty()) goto fail;
    *build_ns_out = build_samples[build_samples.size() / 2] / (double)nparents;
    *corr_ns_out = corr_samples[corr_samples.size() / 2] / (double)nparents;

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_g_child);
    cudaFree(d_child_poly);
    cudaFree(d_g_parent);
    cudaFree(d_parent);
    cudaFree(d_child);
    cudaStreamDestroy(stream);
    return 1;

fail:
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (d_g_child) cudaFree(d_g_child);
    if (d_child_poly) cudaFree(d_child_poly);
    if (d_g_parent) cudaFree(d_g_parent);
    if (d_parent) cudaFree(d_parent);
    if (d_child) cudaFree(d_child);
    if (stream) cudaStreamDestroy(stream);
    *build_ns_out = 0.0;
    *corr_ns_out = 0.0;
    return 0;
#endif
}

int icm_gpu_measure_hbm_bandwidth_gbps(double *gbps_out) {
    if (!gbps_out) return 0;
    const size_t bytes = (size_t)1 << 30;
    void *d_a = nullptr;
    void *d_b = nullptr;
    if (!CUDA_OK(cudaMalloc(&d_a, bytes))) return 0;
    if (!CUDA_OK(cudaMalloc(&d_b, bytes))) {
        cudaFree(d_a);
        return 0;
    }
    if (!CUDA_OK(cudaMemset(d_a, 1, bytes))) {
        cudaFree(d_a); cudaFree(d_b);
        return 0;
    }
    if (!CUDA_OK(cudaMemset(d_b, 2, bytes))) {
        cudaFree(d_a); cudaFree(d_b);
        return 0;
    }
    cudaEvent_t e0 = nullptr;
    cudaEvent_t e1 = nullptr;
    if (!CUDA_OK(cudaEventCreate(&e0)) || !CUDA_OK(cudaEventCreate(&e1))) {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        cudaFree(d_a); cudaFree(d_b);
        return 0;
    }
    if (!CUDA_OK(cudaEventRecord(e0))) {
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(d_a); cudaFree(d_b);
        return 0;
    }
    for (int i = 0; i < 16; ++i) {
        if (!CUDA_OK(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice))) {
            cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(d_a); cudaFree(d_b);
            return 0;
        }
    }
    if (!CUDA_OK(cudaEventRecord(e1)) || !CUDA_OK(cudaEventSynchronize(e1))) {
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(d_a); cudaFree(d_b);
        return 0;
    }
    float ms = 0.0f;
    if (!CUDA_OK(cudaEventElapsedTime(&ms, e0, e1))) {
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(d_a); cudaFree(d_b);
        return 0;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_a);
    cudaFree(d_b);

    double sec = ms / 1000.0;
    double total_bytes = (double)bytes * 16.0;
    *gbps_out = (total_bytes / sec) / 1e9;
    return 1;
}

int icm_gpu_write_config_header(const char *output_path) {
    if (!output_path) return 0;
    FILE *f = fopen(output_path, "w");
    if (!f) {
        set_last_errorf("Cannot open %s for write", output_path);
        return 0;
    }
    double gbps = 0.0;
    icm_gpu_measure_hbm_bandwidth_gbps(&gbps);
    fprintf(f, "/* Auto-generated bootstrap GPU config. Replace with calibrate_gpu.cu output. */\n");
    fprintf(f, "#ifndef ICM_GPU_FFT_CONFIG_H\n#define ICM_GPU_FFT_CONFIG_H\n\n");
    fprintf(f, "#define GPU_N_CALIBRATED_SIZES %d\n", GPU_N_CALIBRATED_SIZES);
    fprintf(f, "static const int gpu_calib_sizes[GPU_N_CALIBRATED_SIZES] = {");
    for (int i = 0; i < GPU_N_CALIBRATED_SIZES; ++i) {
        fprintf(f, "%s%d", (i ? "," : ""), gpu_calib_sizes[i]);
    }
    fprintf(f, "};\n");
    fprintf(f, "static const double gpu_calib_cufft_ns[GPU_N_CALIBRATED_SIZES] = {");
    for (int i = 0; i < GPU_N_CALIBRATED_SIZES; ++i) {
        fprintf(f, "%s%.1f", (i ? "," : ""), gpu_calib_cufft_ns[i]);
    }
    fprintf(f, "};\n");
    fprintf(f, "static const double gpu_calib_cufftdx_build_ns[GPU_N_CALIBRATED_SIZES] = {");
    for (int i = 0; i < GPU_N_CALIBRATED_SIZES; ++i) {
        fprintf(f, "%s%.1f", (i ? "," : ""), gpu_calib_cufftdx_build_ns[i]);
    }
    fprintf(f, "};\n");
    fprintf(f, "static const double gpu_calib_cufftdx_corr_ns[GPU_N_CALIBRATED_SIZES] = {");
    for (int i = 0; i < GPU_N_CALIBRATED_SIZES; ++i) {
        fprintf(f, "%s%.1f", (i ? "," : ""), gpu_calib_cufftdx_corr_ns[i]);
    }
    fprintf(f, "};\n\n");
    fprintf(f, "#define GPU_SCHOOL_FMA_NS %.6f\n", GPU_SCHOOL_FMA_NS);
    fprintf(f, "#define GPU_FFT_OVERHEAD_NS %.6f\n", GPU_FFT_OVERHEAD_NS);
    fprintf(f, "#define GPU_HBM_BANDWIDTH %.3f\n", (gbps > 0.0 ? gbps : GPU_HBM_BANDWIDTH));
    fprintf(f, "#define GPU_FUSED_MAX_CONV_LEN %d\n", GPU_FUSED_MAX_CONV_LEN);
    fprintf(f, "#define GPU_PAIRED_CACHED_CORR_RATIO %.6f\n", GPU_PAIRED_CACHED_CORR_RATIO);
    fprintf(f, "#define GPU_INDEP_PAIR_RATIO %.6f\n", GPU_INDEP_PAIR_RATIO);
    fprintf(f, "#define GPU_BLOCK_BUILD_NS_PER_FMA %.6f\n", GPU_BLOCK_BUILD_NS_PER_FMA);
    fprintf(f, "#define GPU_LEAF_EXTRACT_NS_PER_FMA %.6f\n", GPU_LEAF_EXTRACT_NS_PER_FMA);
    fprintf(f, "#define GPU_VRAM_BYTES %lluULL\n", (unsigned long long)GPU_VRAM_BYTES);
    fprintf(f, "#define GPU_SM_COUNT %d\n", GPU_SM_COUNT);
    fprintf(f, "\n#endif\n");
    fclose(f);
    return 1;
}

}  // extern "C"
