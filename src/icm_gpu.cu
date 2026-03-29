#include "icm_gpu.h"

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
#else
#define ICM_HAVE_CUFFTDX 0
#endif

#if defined(USE_CUFFTDX) && defined(ICM_REQUIRE_CUFFTDX) && !ICM_HAVE_CUFFTDX
#error "USE_CUFFTDX requested, but <cufftdx.hpp> was not found. Set CUFFTDX_INC to MathDx include path."
#endif
#else
#define ICM_HAVE_CUFFTDX 0
#endif

namespace {

constexpr int MAX_B_CANDIDATES = 45;
constexpr int kBCandidates[MAX_B_CANDIDATES] = {
    8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192,
    208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512,
    576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1536, 1792,
    2048, 2560, 3072, 3584, 4096
};
static int g_runtime_fused_max_conv_len = GPU_FUSED_MAX_CONV_LEN;
constexpr int GPU_SCHOOL_WARP_MAX_CONV = 128;
constexpr int GPU_SCHOOL_WARPS_PER_BLOCK = 4;
constexpr size_t GPU_SCHOOL_SMEM_SAFE_BYTES = 48u * 1024u;

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

template<class FFT>
__device__ inline void cufftdx_load_real(const double *src, int copy_len,
                                         typename FFT::value_type *thread_data) {
    using value_t = typename FFT::value_type;
    constexpr unsigned N = cufftdx::size_of<FFT>::value;
    const unsigned stride = FFT::stride;
    if (FFT::working_group::is_thread_active()) {
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
}

template<class FFT>
__device__ inline void cufftdx_store_real(const typename FFT::value_type *thread_data,
                                          double *dst, int out_len, double scale) {
    constexpr unsigned N = cufftdx::size_of<FFT>::value;
    const unsigned stride = FFT::stride;
    if (FFT::working_group::is_thread_active()) {
        for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
            unsigned idx = i * stride + threadIdx.x;
            if (idx < N && idx < (unsigned)out_len) {
                dst[idx] = thread_data[i].x * scale;
            }
        }
    }
}

template<class FFT>
__device__ inline void cufftdx_mul_freq_inplace(typename FFT::value_type *lhs,
                                                const typename FFT::value_type *rhs) {
    using value_t = typename FFT::value_type;
    constexpr unsigned N = cufftdx::size_of<FFT>::value;
    const unsigned stride = FFT::stride;
    if (FFT::working_group::is_thread_active()) {
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
}

template<class FFT>
__device__ inline void cufftdx_mul_freq_conj_inplace(typename FFT::value_type *lhs,
                                                     const typename FFT::value_type *rhs) {
    using value_t = typename FFT::value_type;
    constexpr unsigned N = cufftdx::size_of<FFT>::value;
    const unsigned stride = FFT::stride;
    if (FFT::working_group::is_thread_active()) {
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
}

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
    if (FFTFwd::working_group::is_thread_active()) {
        const unsigned stride = FFTFwd::stride;
        constexpr unsigned N = cufftdx::size_of<FFTFwd>::value;
        for (unsigned i = 0; i < FFTFwd::elements_per_thread; ++i) {
            unsigned idx = i * stride + threadIdx.x;
            if (idx < N) {
                gspec_saved[i] = gbuf[i];
            }
        }
    }

    cufftdx_load_real<FFTFwd>(PR, len_P, pbuf);
    FFTFwd().execute(pbuf, shared_mem);
    cufftdx_mul_freq_conj_inplace<FFTFwd>(gbuf, pbuf);
    FFTInv().execute(gbuf, shared_mem);
    cufftdx_store_real<FFTInv>(gbuf, outL, len_out, inv_fft_n);

    cufftdx_load_real<FFTFwd>(PL, len_P, pbuf);
    FFTFwd().execute(pbuf, shared_mem);
    if (FFTFwd::working_group::is_thread_active()) {
        const unsigned stride = FFTFwd::stride;
        constexpr unsigned N = cufftdx::size_of<FFTFwd>::value;
        for (unsigned i = 0; i < FFTFwd::elements_per_thread; ++i) {
            unsigned idx = i * stride + threadIdx.x;
            if (idx < N) {
                gbuf[i] = gspec_saved[i];
            }
        }
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

static bool is_cufftdx_supported_fft_n(int fft_n) {
    switch (fft_n) {
        case 64:
        case 128:
        case 256:
        case 512:
        case 1024:
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

    IcmGpuOptions opts{};

    std::vector<int> sort_perm;
    std::vector<int> inv_perm;
    std::vector<double> S_sorted;
    std::vector<double> payout_host;

    std::vector<int> nn;
    std::vector<int> psz;
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

    std::vector<double *> d_poly_levels;
    std::vector<double *> d_g_levels;
    std::vector<cufftDoubleComplex *> d_fft_cache;
    std::vector<GpuFftBuffers> build_fft;
    std::vector<GpuFftBuffers> corr_fft;

    cudaStream_t stream_compute = nullptr;
    cudaStream_t stream_aux = nullptr;
    cudaEvent_t evt_a_ready[2] = {nullptr, nullptr};

    cudaGraph_t graph[2] = {nullptr, nullptr};
    cudaGraphExec_t graph_exec[2] = {nullptr, nullptr};
    bool graph_ready[2] = {false, false};

    bool use_async_pool = false;
    cudaMemPool_t mem_pool = nullptr;

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
    /* GPU_SCHOOL_FMA_NS is measured from a pure FMA stream and can materially
     * under-estimate full polynomial schoolbook kernels (register pressure,
     * indexing, and memory traffic). Clamp by the measured block-build FMA cost
     * so tier assignment does not over-predict schoolbook. */
    return std::max(GPU_SCHOOL_FMA_NS, GPU_BLOCK_BUILD_NS_PER_FMA);
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
                                          double *parent, int pps, int nparents) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)nparents * (size_t)pps;
    if (idx >= total) return;
    int p = (int)(idx / (size_t)pps);
    int m = (int)(idx % (size_t)pps);
    const double *L = child + (size_t)(2 * p) * (size_t)cps;
    const double *R = child + (size_t)(2 * p + 1) * (size_t)cps;
    int j_lo = m - (cps - 1);
    if (j_lo < 0) j_lo = 0;
    int j_hi = m;
    if (j_hi > cps - 1) j_hi = cps - 1;
    double sum = 0.0;
    for (int j = j_lo; j <= j_hi; ++j) sum += L[j] * R[m - j];
    parent[idx] = sum;
}

/* Shared-memory block kernel: one parent per block. */
__global__ static void k_schoolbook_build_smem_parent(const double *child, int cps,
                                                      double *parent, int pps, int nparents) {
    int p = blockIdx.x;
    if (p >= nparents) return;
    extern __shared__ double sh[];
    double *Lsh = sh;
    double *Rsh = sh + cps;

    const double *L = child + (size_t)(2 * p) * (size_t)cps;
    const double *R = child + (size_t)(2 * p + 1) * (size_t)cps;
    for (int i = threadIdx.x; i < cps; i += blockDim.x) {
        Lsh[i] = L[i];
        Rsh[i] = R[i];
    }
    __syncthreads();

    double *out = parent + (size_t)p * (size_t)pps;
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
                                                     double *parent, int pps, int nparents) {
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

    const double *L = child + (size_t)(2 * p) * (size_t)cps;
    const double *R = child + (size_t)(2 * p + 1) * (size_t)cps;
    for (int i = lane; i < cps; i += WARP) {
        Lsh[i] = L[i];
        Rsh[i] = R[i];
    }
    __syncwarp();

    double *out = parent + (size_t)p * (size_t)pps;
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

__global__ static void k_wrap_build(double *parent, int pps, int nparents,
                                    const double *child, int cps, int conv_len,
                                    int fft_n, int wrap_m) {
    int p = blockIdx.x;
    if (p >= nparents || threadIdx.x != 0) return;
    double *out = parent + (size_t)p * (size_t)pps;
    const double *L = child + (size_t)(2 * p) * (size_t)cps;
    const double *R = child + (size_t)(2 * p + 1) * (size_t)cps;
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
                                        int fft_n, int wrap_m) {
    int p = blockIdx.x;
    if (p >= nparents || threadIdx.x != 0) return;

    double *outL = g_child + (size_t)(2 * p) * (size_t)child_gsz;
    double *outR = g_child + (size_t)(2 * p + 1) * (size_t)child_gsz;
    const double *gp = g_parent + (size_t)p * (size_t)parent_gsz;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)cps;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)cps;

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
                                              double *g_child, int child_gsz, int len_out, int nparents) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)nparents * (size_t)len_out;
    if (idx >= total) return;
    int p = (int)(idx / (size_t)len_out);
    int m = (int)(idx % (size_t)len_out);
    const double *gp = g_parent + (size_t)p * (size_t)parent_gsz;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)cps;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)cps;
    double sumL = 0.0;
    double sumR = 0.0;
    int j_max = len_g - m;
    if (j_max > len_P) j_max = len_P;
    for (int j = 0; j < j_max; ++j) {
        sumL += PR[j] * gp[m + j];
        sumR += PL[j] * gp[m + j];
    }
    g_child[(size_t)(2 * p) * (size_t)child_gsz + (size_t)m] = sumL;
    g_child[(size_t)(2 * p + 1) * (size_t)child_gsz + (size_t)m] = sumR;
}

/* Shared-memory block kernel: one parent per block for paired correlation. */
__global__ static void k_schoolbook_corr_pair_smem_parent(const double *g_parent, int parent_gsz,
                                                           int len_g,
                                                           const double *child_poly, int cps, int len_P,
                                                           double *g_child, int child_gsz,
                                                           int len_out, int nparents) {
    int p = blockIdx.x;
    if (p >= nparents) return;

    extern __shared__ double sh[];
    double *gp_sh = sh;
    double *pl_sh = gp_sh + len_g;
    double *pr_sh = pl_sh + len_P;

    const double *gp = g_parent + (size_t)p * (size_t)parent_gsz;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)cps;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)cps;
    for (int i = threadIdx.x; i < len_g; i += blockDim.x) gp_sh[i] = gp[i];
    for (int i = threadIdx.x; i < len_P; i += blockDim.x) {
        pl_sh[i] = PL[i];
        pr_sh[i] = PR[i];
    }
    __syncthreads();

    double *outL = g_child + (size_t)(2 * p) * (size_t)child_gsz;
    double *outR = g_child + (size_t)(2 * p + 1) * (size_t)child_gsz;
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
                                                          int len_out, int nparents) {
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

    const double *gp = g_parent + (size_t)p * (size_t)parent_gsz;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)cps;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)cps;
    for (int i = lane; i < len_g; i += WARP) gp_sh[i] = gp[i];
    for (int i = lane; i < len_P; i += WARP) {
        pl_sh[i] = PL[i];
        pr_sh[i] = PR[i];
    }
    __syncwarp();

    double *outL = g_child + (size_t)(2 * p) * (size_t)child_gsz;
    double *outR = g_child + (size_t)(2 * p + 1) * (size_t)child_gsz;
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

static bool create_cufft_plan(cufftHandle *plan, int n, int batch, bool r2c) {
    if (!CUFFT_OK(cufftCreate(plan))) return false;
    int rank = 1;
    int n_arr[1] = {n};
    int inembed[1] = {n};
    int onembed[1] = {n / 2 + 1};
    size_t work_size = 0;
    if (r2c) {
        if (!CUFFT_OK(cufftMakePlanMany(*plan, rank, n_arr,
                                        inembed, 1, n,
                                        onembed, 1, n / 2 + 1,
                                        CUFFT_D2Z, batch, &work_size))) return false;
    } else {
        if (!CUFFT_OK(cufftMakePlanMany(*plan, rank, n_arr,
                                        onembed, 1, n / 2 + 1,
                                        inembed, 1, n,
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
    if (ell <= 0 || ell >= plan->L) return true;
    auto &lp = plan->levels[ell];
    if (!lp.use_fft) return true;

    int fft_n = lp.fft_n;
    int cn = lp.cn;
    int child_batch = plan->nn[ell - 1];
    int parent_batch = plan->nn[ell];

    auto &b = plan->build_fft[ell];
    b.fft_n = fft_n;
    b.cn = cn;
    b.batch_fwd = child_batch;
    b.batch_inv = parent_batch;

    size_t bytes_real_in = (size_t)child_batch * (size_t)fft_n * sizeof(double);
    size_t bytes_spec_in = (size_t)child_batch * (size_t)cn * sizeof(cufftDoubleComplex);
    size_t bytes_spec_mid = (size_t)parent_batch * (size_t)cn * sizeof(cufftDoubleComplex);
    size_t bytes_real_out = (size_t)parent_batch * (size_t)fft_n * sizeof(double);

    if (!alloc_device(plan, (void **)&b.real_in, bytes_real_in, plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&b.spec_in, bytes_spec_in, plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&b.spec_mid, bytes_spec_mid, plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&b.real_out, bytes_real_out, plan->stream_compute)) return false;
    if (!create_cufft_plan(&b.plan_fwd, fft_n, child_batch, true)) return false;
    if (!create_cufft_plan(&b.plan_inv, fft_n, parent_batch, false)) return false;
    if (!CUFFT_OK(cufftSetStream(b.plan_fwd, plan->stream_compute))) return false;
    if (!CUFFT_OK(cufftSetStream(b.plan_inv, plan->stream_compute))) return false;

    auto &c = plan->corr_fft[ell];
    c.fft_n = fft_n;
    c.cn = cn;
    c.batch_fwd = parent_batch;
    c.batch_inv = 2 * parent_batch;
    size_t bytes_corr_real_in = (size_t)parent_batch * (size_t)fft_n * sizeof(double);
    size_t bytes_corr_spec_in = (size_t)parent_batch * (size_t)cn * sizeof(cufftDoubleComplex);
    size_t bytes_corr_spec_mid = (size_t)(2 * parent_batch) * (size_t)cn * sizeof(cufftDoubleComplex);
    size_t bytes_corr_real_out = (size_t)(2 * parent_batch) * (size_t)fft_n * sizeof(double);

    if (!alloc_device(plan, (void **)&c.real_in, bytes_corr_real_in, plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&c.spec_in, bytes_corr_spec_in, plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&c.spec_mid, bytes_corr_spec_mid, plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&c.real_out, bytes_corr_real_out, plan->stream_compute)) return false;
    if (!create_cufft_plan(&c.plan_fwd, fft_n, parent_batch, true)) return false;
    if (!create_cufft_plan(&c.plan_inv, fft_n, 2 * parent_batch, false)) return false;
    if (!CUFFT_OK(cufftSetStream(c.plan_fwd, plan->stream_compute))) return false;
    if (!CUFFT_OK(cufftSetStream(c.plan_inv, plan->stream_compute))) return false;

    if (lp.cache_fft && plan->opts.memory_strategy < 2) {
        size_t bytes_cache = (size_t)child_batch * (size_t)cn * sizeof(cufftDoubleComplex);
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
        fprintf(stderr, "gpu_plan n=%d k=%d k_pad=%d B=%d L=%d nblocks=%d\n",
                plan->n, plan->k, plan->k_pad, plan->B, plan->L, plan->nblocks);
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

static bool allocate_plan_device_memory(GpuPlan *plan) {
    if (!CUDA_OK(cudaStreamCreate(&plan->stream_compute))) return false;
    if (!CUDA_OK(cudaStreamCreate(&plan->stream_aux))) return false;
    if (!CUDA_OK(cudaEventCreateWithFlags(&plan->evt_a_ready[0], cudaEventDisableTiming))) return false;
    if (!CUDA_OK(cudaEventCreateWithFlags(&plan->evt_a_ready[1], cudaEventDisableTiming))) return false;

    if (!maybe_init_mem_pool(plan)) return false;

    if (!alloc_device(plan, (void **)&plan->d_S_sorted, (size_t)plan->n * sizeof(double), plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&plan->d_sort_perm, (size_t)plan->n * sizeof(int), plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&plan->d_inv_perm, (size_t)plan->n * sizeof(int), plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&plan->d_a_sorted[0], (size_t)plan->n * sizeof(double), plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&plan->d_a_sorted[1], (size_t)plan->n * sizeof(double), plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&plan->d_graph_logv[0], sizeof(double), plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&plan->d_graph_logv[1], sizeof(double), plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&plan->d_graph_scale[0], sizeof(double), plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&plan->d_graph_scale[1], sizeof(double), plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&plan->d_inner_sorted, (size_t)plan->n * sizeof(double), plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&plan->d_equity, (size_t)plan->n * sizeof(double), plan->stream_compute)) return false;
    if (!alloc_device(plan, (void **)&plan->d_payout, (size_t)plan->k * sizeof(double), plan->stream_compute)) return false;

    size_t block_prod_bytes = (size_t)plan->N_tree * (size_t)(plan->B + 1) * sizeof(double);
    if (!alloc_device(plan, (void **)&plan->d_block_prods, block_prod_bytes, plan->stream_compute)) return false;

    plan->d_poly_levels.assign(plan->L, nullptr);
    plan->d_g_levels.assign(plan->L, nullptr);
    plan->d_fft_cache.assign(plan->L, nullptr);
    plan->build_fft.assign(plan->L, GpuFftBuffers{});
    plan->corr_fft.assign(plan->L, GpuFftBuffers{});

    for (int ell = 0; ell < plan->L; ++ell) {
        size_t poly_bytes = (size_t)plan->nn[ell] * (size_t)plan->psz[ell] * sizeof(double);
        if (!alloc_device(plan, (void **)&plan->d_poly_levels[ell], poly_bytes, plan->stream_compute)) return false;
        if (!alloc_device(plan, (void **)&plan->d_g_levels[ell], poly_bytes, plan->stream_compute)) return false;
    }

    for (int ell = 1; ell < plan->L; ++ell) {
        if (!allocate_level_buffers(plan, ell, {})) return false;
    }

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
    return true;
}

static bool destroy_fft_buffers(GpuPlan *plan, GpuFftBuffers &b, cudaStream_t stream) {
    size_t real_in_bytes = (size_t)b.batch_fwd * (size_t)b.fft_n * sizeof(double);
    size_t spec_in_bytes = (size_t)b.batch_fwd * (size_t)b.cn * sizeof(cufftDoubleComplex);
    size_t spec_mid_bytes = (size_t)b.batch_inv * (size_t)b.cn * sizeof(cufftDoubleComplex);
    size_t real_out_bytes = (size_t)b.batch_inv * (size_t)b.fft_n * sizeof(double);
    if (b.plan_fwd) { if (!CUFFT_OK(cufftDestroy(b.plan_fwd))) return false; b.plan_fwd = 0; }
    if (b.plan_inv) { if (!CUFFT_OK(cufftDestroy(b.plan_inv))) return false; b.plan_inv = 0; }
    if (!free_device(plan, b.real_in, real_in_bytes, stream)) return false;
    if (!free_device(plan, b.spec_in, spec_in_bytes, stream)) return false;
    if (!free_device(plan, b.spec_mid, spec_mid_bytes, stream)) return false;
    if (!free_device(plan, b.real_out, real_out_bytes, stream)) return false;
    b = GpuFftBuffers{};
    return true;
}

static void destroy_plan(GpuPlan *plan) {
    if (!plan) return;
    cudaStream_t stream = plan->stream_compute;
    if (stream) cudaStreamSynchronize(stream);

    for (int ell = 1; ell < plan->L; ++ell) {
        destroy_fft_buffers(plan, plan->build_fft[ell], stream);
        destroy_fft_buffers(plan, plan->corr_fft[ell], stream);
        if (plan->d_fft_cache[ell]) {
            size_t bytes = (size_t)plan->nn[ell - 1] * (size_t)plan->levels[ell].cn * sizeof(cufftDoubleComplex);
            free_device(plan, plan->d_fft_cache[ell], bytes, stream);
            plan->d_fft_cache[ell] = nullptr;
        }
    }
    for (int ell = 0; ell < plan->L; ++ell) {
        if (plan->d_poly_levels.size() > (size_t)ell && plan->d_poly_levels[ell]) {
            size_t bytes = (size_t)plan->nn[ell] * (size_t)plan->psz[ell] * sizeof(double);
            free_device(plan, plan->d_poly_levels[ell], bytes, stream);
        }
        if (plan->d_g_levels.size() > (size_t)ell && plan->d_g_levels[ell]) {
            size_t bytes = (size_t)plan->nn[ell] * (size_t)plan->psz[ell] * sizeof(double);
            free_device(plan, plan->d_g_levels[ell], bytes, stream);
        }
    }

    if (plan->d_block_prods) {
        size_t bytes = (size_t)plan->N_tree * (size_t)(plan->B + 1) * sizeof(double);
        free_device(plan, plan->d_block_prods, bytes, stream);
    }
    if (plan->d_payout) free_device(plan, plan->d_payout, (size_t)plan->k * sizeof(double), stream);
    if (plan->d_equity) free_device(plan, plan->d_equity, (size_t)plan->n * sizeof(double), stream);
    if (plan->d_inner_sorted) free_device(plan, plan->d_inner_sorted, (size_t)plan->n * sizeof(double), stream);
    if (plan->d_graph_scale[1]) free_device(plan, plan->d_graph_scale[1], sizeof(double), stream);
    if (plan->d_graph_scale[0]) free_device(plan, plan->d_graph_scale[0], sizeof(double), stream);
    if (plan->d_graph_logv[1]) free_device(plan, plan->d_graph_logv[1], sizeof(double), stream);
    if (plan->d_graph_logv[0]) free_device(plan, plan->d_graph_logv[0], sizeof(double), stream);
    if (plan->d_a_sorted[0]) free_device(plan, plan->d_a_sorted[0], (size_t)plan->n * sizeof(double), stream);
    if (plan->d_a_sorted[1]) free_device(plan, plan->d_a_sorted[1], (size_t)plan->n * sizeof(double), stream);
    if (plan->d_inv_perm) free_device(plan, plan->d_inv_perm, (size_t)plan->n * sizeof(int), stream);
    if (plan->d_sort_perm) free_device(plan, plan->d_sort_perm, (size_t)plan->n * sizeof(int), stream);
    if (plan->d_S_sorted) free_device(plan, plan->d_S_sorted, (size_t)plan->n * sizeof(double), stream);

    for (int q = 0; q < 2; ++q) {
        if (plan->graph_exec[q]) cudaGraphExecDestroy(plan->graph_exec[q]);
        if (plan->graph[q]) cudaGraphDestroy(plan->graph[q]);
    }
    if (plan->evt_a_ready[0]) cudaEventDestroy(plan->evt_a_ready[0]);
    if (plan->evt_a_ready[1]) cudaEventDestroy(plan->evt_a_ready[1]);
    if (plan->stream_aux) cudaStreamDestroy(plan->stream_aux);
    if (plan->stream_compute) cudaStreamDestroy(plan->stream_compute);
    delete plan;
}

static bool run_build_level_schoolbook(GpuPlan *plan, int ell);
static bool run_prop_level_schoolbook(GpuPlan *plan, int ell);

static bool run_build_level_fft(GpuPlan *plan, int ell) {
    int cps = plan->psz[ell - 1];
    int pps = plan->psz[ell];
    int child_batch = plan->nn[ell - 1];
    int parent_batch = plan->nn[ell];
    auto &lp = plan->levels[ell];
    auto &b = plan->build_fft[ell];
    int threads = 256;
    size_t pack_total = (size_t)child_batch * (size_t)b.fft_n;
    int blocks_pack = (int)((pack_total + threads - 1) / threads);
    k_pack_level_to_fft<<<blocks_pack, threads, 0, plan->stream_compute>>>(
        plan->d_poly_levels[ell - 1], cps, child_batch,
        b.real_in, b.fft_n, cps);
    if (!CUDA_OK(cudaGetLastError())) return false;

    if (!CUFFT_OK(cufftExecD2Z(b.plan_fwd, b.real_in, b.spec_in))) return false;

    if (lp.cache_fft) {
        size_t bytes = (size_t)child_batch * (size_t)b.cn * sizeof(cufftDoubleComplex);
        if (!plan->d_fft_cache[ell]) {
            if (!alloc_device(plan, (void **)&plan->d_fft_cache[ell], bytes, plan->stream_compute)) return false;
        }
        if (!CUDA_OK(cudaMemcpyAsync(plan->d_fft_cache[ell], b.spec_in, bytes,
                                     cudaMemcpyDeviceToDevice, plan->stream_compute))) return false;
    }

    size_t mul_total = (size_t)parent_batch * (size_t)b.cn;
    int blocks_mul = (int)((mul_total + threads - 1) / threads);
    k_pairwise_mul<<<blocks_mul, threads, 0, plan->stream_compute>>>(
        b.spec_in, b.cn, b.spec_mid, parent_batch);
    if (!CUDA_OK(cudaGetLastError())) return false;

    if (!CUFFT_OK(cufftExecZ2D(b.plan_inv, b.spec_mid, b.real_out))) return false;

    size_t unpack_total = (size_t)parent_batch * (size_t)pps;
    int blocks_unpack = (int)((unpack_total + threads - 1) / threads);
    k_unpack_fft_to_level<<<blocks_unpack, threads, 0, plan->stream_compute>>>(
        b.real_out, b.fft_n, 1.0 / (double)b.fft_n,
        pps, parent_batch, plan->d_poly_levels[ell]);
    if (!CUDA_OK(cudaGetLastError())) return false;
    if (lp.build_wrap_m > 0) {
        k_wrap_build<<<parent_batch, 1, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], pps, parent_batch,
            plan->d_poly_levels[ell - 1], cps, lp.build_conv,
            b.fft_n, lp.build_wrap_m);
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
    int nparents = plan->nn[ell];
    if (nparents <= 0 || cps <= 0 || pps <= 0) return true;
    bool ok = launch_cufftdx_build_dispatch(lp.fft_n,
                                            plan->d_poly_levels[ell - 1], cps,
                                            plan->d_poly_levels[ell], pps, nparents,
                                            1.0 / (double)lp.fft_n,
                                            plan->stream_compute);
    if (!ok) return run_build_level_fft(plan, ell);
    if (lp.build_wrap_m > 0) {
        k_wrap_build<<<nparents, 1, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell], pps, nparents,
            plan->d_poly_levels[ell - 1], cps, lp.build_conv,
            lp.fft_n, lp.build_wrap_m);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

static bool run_build_level_schoolbook(GpuPlan *plan, int ell) {
    int cps = plan->psz[ell - 1];
    int pps = plan->psz[ell];
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
                plan->d_poly_levels[ell], pps, nparents);
        } else {
            int fb_threads = 256;
            size_t total = (size_t)nparents * (size_t)pps;
            int fb_blocks = (int)((total + fb_threads - 1) / fb_threads);
            k_schoolbook_build<<<fb_blocks, fb_threads, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents);
        }
    } else {
        int threads = 256;
        int blocks = nparents;
        size_t shmem = (size_t)(2 * cps) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_build_smem_parent<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents);
        } else {
            size_t total = (size_t)nparents * (size_t)pps;
            int fb_blocks = (int)((total + threads - 1) / threads);
            k_schoolbook_build<<<fb_blocks, threads, 0, plan->stream_compute>>>(
                plan->d_poly_levels[ell - 1], cps,
                plan->d_poly_levels[ell], pps, nparents);
        }
    }
    if (!CUDA_OK(cudaGetLastError())) return false;
    return true;
}

static bool run_prop_level_fft(GpuPlan *plan, int ell) {
    int parent_gsz = plan->psz[ell];
    int child_gsz = plan->psz[ell - 1];
    int cps = plan->psz[ell - 1];
    int nparents = plan->nn[ell];
    auto &lp = plan->levels[ell];
    auto &c = plan->corr_fft[ell];
    int len_g = lp.g_eff;
    int len_P = lp.p_eff;
    int len_out = lp.out_needed;
    int threads = 256;

    size_t pack_total = (size_t)nparents * (size_t)c.fft_n;
    int blocks_pack = (int)((pack_total + threads - 1) / threads);
    k_pack_level_to_fft<<<blocks_pack, threads, 0, plan->stream_compute>>>(
        plan->d_g_levels[ell], parent_gsz, nparents,
        c.real_in, c.fft_n, len_g);
    if (!CUDA_OK(cudaGetLastError())) return false;

    if (!CUFFT_OK(cufftExecD2Z(c.plan_fwd, c.real_in, c.spec_in))) return false;

    const cufftDoubleComplex *child_spec = plan->d_fft_cache[ell];
    if (!child_spec) {
        auto &b = plan->build_fft[ell];
        int child_batch = plan->nn[ell - 1];
        size_t child_pack_total = (size_t)child_batch * (size_t)b.fft_n;
        int child_pack_blocks = (int)((child_pack_total + threads - 1) / threads);
        k_pack_level_to_fft<<<child_pack_blocks, threads, 0, plan->stream_compute>>>(
            plan->d_poly_levels[ell - 1], cps, child_batch,
            b.real_in, b.fft_n, len_P);
        if (!CUDA_OK(cudaGetLastError())) return false;
        if (!CUFFT_OK(cufftExecD2Z(b.plan_fwd, b.real_in, b.spec_in))) return false;
        child_spec = b.spec_in;
    }

    size_t corr_total = (size_t)nparents * (size_t)c.cn;
    int blocks_corr = (int)((corr_total + threads - 1) / threads);
    k_paired_corr_freq<<<blocks_corr, threads, 0, plan->stream_compute>>>(
        c.spec_in, child_spec, c.cn, nparents, c.spec_mid);
    if (!CUDA_OK(cudaGetLastError())) return false;

    if (!CUFFT_OK(cufftExecZ2D(c.plan_inv, c.spec_mid, c.real_out))) return false;

    size_t unpack_total = (size_t)(2 * nparents) * (size_t)len_out;
    int blocks_unpack = (int)((unpack_total + threads - 1) / threads);
    k_unpack_corr_children<<<blocks_unpack, threads, 0, plan->stream_compute>>>(
        c.real_out, c.fft_n, 1.0 / (double)c.fft_n, child_gsz, len_out,
        nparents, plan->d_g_levels[ell - 1]);
    if (!CUDA_OK(cudaGetLastError())) return false;

    if (lp.corr_wrap_m > 0) {
        k_wrap_corr_pair<<<nparents, 1, 0, plan->stream_compute>>>(
            plan->d_g_levels[ell - 1], child_gsz, nparents,
            plan->d_g_levels[ell], parent_gsz, len_g,
            plan->d_poly_levels[ell - 1], cps, len_P,
            len_out,
            c.fft_n, lp.corr_wrap_m);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    if (plan->opts.memory_strategy >= 2 && lp.cache_fft && plan->d_fft_cache[ell]) {
        size_t bytes = (size_t)plan->nn[ell - 1] * (size_t)c.cn * sizeof(cufftDoubleComplex);
        if (!free_device(plan, plan->d_fft_cache[ell], bytes, plan->stream_compute)) return false;
        plan->d_fft_cache[ell] = nullptr;
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
    int nparents = plan->nn[ell];
    int len_g = lp.g_eff;
    int len_P = lp.p_eff;
    int len_out = lp.out_needed;
    if (nparents <= 0 || len_out <= 0 || len_g <= 0 || len_P <= 0) return true;
    bool ok = launch_cufftdx_corr_dispatch(lp.fft_n,
                                           plan->d_g_levels[ell], parent_gsz, len_g,
                                           plan->d_poly_levels[ell - 1], cps, len_P,
                                           plan->d_g_levels[ell - 1], child_gsz, len_out, nparents,
                                           1.0 / (double)lp.fft_n,
                                           plan->stream_compute);
    if (!ok) return run_prop_level_fft(plan, ell);
    if (lp.corr_wrap_m > 0) {
        k_wrap_corr_pair<<<nparents, 1, 0, plan->stream_compute>>>(
            plan->d_g_levels[ell - 1], child_gsz, nparents,
            plan->d_g_levels[ell], parent_gsz, len_g,
            plan->d_poly_levels[ell - 1], cps, len_P,
            len_out,
            lp.fft_n, lp.corr_wrap_m);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }
    return true;
}

static bool run_prop_level_schoolbook(GpuPlan *plan, int ell) {
    int parent_gsz = plan->psz[ell];
    int child_gsz = plan->psz[ell - 1];
    int cps = plan->psz[ell - 1];
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
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents);
        } else {
            int fb_threads = 256;
            size_t total = (size_t)nparents * (size_t)len_out;
            int fb_blocks = (int)((total + fb_threads - 1) / fb_threads);
            k_schoolbook_corr_pair<<<fb_blocks, fb_threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents);
        }
    } else {
        int threads = 256;
        int blocks = nparents;
        size_t shmem = (size_t)(len_g + 2 * len_P) * sizeof(double);
        if (shmem <= GPU_SCHOOL_SMEM_SAFE_BYTES) {
            k_schoolbook_corr_pair_smem_parent<<<blocks, threads, shmem, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents);
        } else {
            size_t total = (size_t)nparents * (size_t)len_out;
            int fb_blocks = (int)((total + threads - 1) / threads);
            k_schoolbook_corr_pair<<<fb_blocks, threads, 0, plan->stream_compute>>>(
                plan->d_g_levels[ell], parent_gsz, len_g,
                plan->d_poly_levels[ell - 1], cps, len_P,
                plan->d_g_levels[ell - 1], child_gsz, len_out, nparents);
        }
    }
    if (!CUDA_OK(cudaGetLastError())) return false;
    return true;
}

static bool run_hybrid_single_q(GpuPlan *plan, int a_buf_idx,
                                double logv, double w,
                                bool skip_compute_a,
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
        double t0 = now_ns_host();
        if (!CUDA_OK(cudaGraphLaunch(plan->graph_exec[curr], plan->stream_compute))) return false;
        if (!CUDA_OK(cudaStreamSynchronize(plan->stream_compute))) return false;
        /* Graph currently captures compute_a + build + propagate + leaf + accumulate. */
        *tree_build_ns += (now_ns_host() - t0);
        return true;
    }
    if (!skip_compute_a) {
        k_compute_a<<<blocks_n, threads, 0, plan->stream_compute>>>(
            plan->d_S_sorted, plan->d_a_sorted[curr], plan->n, logv);
        if (!CUDA_OK(cudaGetLastError())) return false;
    }

    double t0 = now_ns_host();
    int threads_block = 256;
    size_t shmem_block = (size_t)(2 * (plan->B + 1)) * sizeof(double);
    k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_compute>>>(
        plan->d_a_sorted[curr], plan->n, plan->B,
        plan->nblocks, plan->N_tree, plan->psz[0],
        plan->d_poly_levels[0], plan->d_block_prods);
    if (!CUDA_OK(cudaGetLastError())) return false;
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
    int root_gsz = plan->psz[top];
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

    t0 = now_ns_host();
    int threads_leaf = plan->B;
    if (threads_leaf > 1024) threads_leaf = 1024;
    k_leaf_extract<<<plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
        plan->d_a_sorted[curr], plan->n, plan->B, plan->nblocks,
        plan->d_block_prods, plan->d_g_levels[0], plan->psz[0],
        plan->g_needed[0], plan->k, plan->d_inner_sorted);
    if (!CUDA_OK(cudaGetLastError())) return false;
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
    /* Keep graph path deterministic while memory-recycling frees are active. */
    if (plan->opts.memory_strategy >= 2) return true;

    int threads = 256;
    int blocks_n = (plan->n + threads - 1) / threads;
    int top = plan->L - 1;
    int root_gsz = plan->psz[top];
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

        k_block_build<<<plan->N_tree, threads_block, shmem_block, plan->stream_compute>>>(
            plan->d_a_sorted[curr], plan->n, plan->B,
            plan->nblocks, plan->N_tree, plan->psz[0],
            plan->d_poly_levels[0], plan->d_block_prods);
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

        k_leaf_extract<<<plan->nblocks, threads_leaf, 0, plan->stream_compute>>>(
            plan->d_a_sorted[curr], plan->n, plan->B, plan->nblocks,
            plan->d_block_prods, plan->d_g_levels[0], plan->psz[0],
            plan->g_needed[0], plan->k, plan->d_inner_sorted);
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

    double t0 = now_ns_host();
    if (plan->opts.enable_graphs && (plan->graph_ready[0] || plan->graph_ready[1])) {
        for (int q = 0; q < Q; ++q) {
            if (pts[q].w == 0.0) continue;
            int curr = q & 1;
            double stage_before = block_ns + tree_build_ns + tree_prop_cached_ns +
                                  tree_prop_recomp_ns + leaf_ns + accum_ns;
            double q0 = now_ns_host();
            if (!run_hybrid_single_q(plan, curr, pts[q].logv, pts[q].w, false,
                                     &block_ns, &tree_build_ns, &tree_prop_cached_ns,
                                     &tree_prop_recomp_ns, &leaf_ns, &accum_ns)) {
                return -1.0;
            }
            double stage_after = block_ns + tree_build_ns + tree_prop_cached_ns +
                                 tree_prop_recomp_ns + leaf_ns + accum_ns;
            quad_ovh_ns += (now_ns_host() - q0) - (stage_after - stage_before);
        }
    } else if (plan->opts.enable_q_pipeline) {
        int q_start = 0;
        while (q_start < Q && pts[q_start].w == 0.0) ++q_start;
        if (q_start < Q) {
            int curr = q_start & 1;
            k_compute_a<<<blocks, threads, 0, plan->stream_aux>>>(
                plan->d_S_sorted, plan->d_a_sorted[curr], plan->n, pts[q_start].logv);
            if (!CUDA_OK(cudaGetLastError())) return -1.0;
            if (!CUDA_OK(cudaEventRecord(plan->evt_a_ready[curr], plan->stream_aux))) return -1.0;
        }

        for (int q = q_start; q < Q; ++q) {
            if (pts[q].w == 0.0) continue;
            int curr = q & 1;
            if (!CUDA_OK(cudaStreamWaitEvent(plan->stream_compute, plan->evt_a_ready[curr], 0))) return -1.0;

            int qn = q + 1;
            while (qn < Q && pts[qn].w == 0.0) ++qn;
            if (qn < Q) {
                int next = qn & 1;
                k_compute_a<<<blocks, threads, 0, plan->stream_aux>>>(
                    plan->d_S_sorted, plan->d_a_sorted[next], plan->n, pts[qn].logv);
                if (!CUDA_OK(cudaGetLastError())) return -1.0;
                if (!CUDA_OK(cudaEventRecord(plan->evt_a_ready[next], plan->stream_aux))) return -1.0;
            }

            double stage_before = block_ns + tree_build_ns + tree_prop_cached_ns +
                                  tree_prop_recomp_ns + leaf_ns + accum_ns;
            double q0 = now_ns_host();
            if (!run_hybrid_single_q(plan, curr, pts[q].logv, pts[q].w, true,
                                     &block_ns, &tree_build_ns, &tree_prop_cached_ns,
                                     &tree_prop_recomp_ns, &leaf_ns, &accum_ns)) {
                return -1.0;
            }
            double stage_after = block_ns + tree_build_ns + tree_prop_cached_ns +
                                 tree_prop_recomp_ns + leaf_ns + accum_ns;
            quad_ovh_ns += (now_ns_host() - q0) - (stage_after - stage_before);
        }
    } else {
        for (int q = 0; q < Q; ++q) {
            if (pts[q].w == 0.0) continue;
            int curr = q & 1;
            double stage_before = block_ns + tree_build_ns + tree_prop_cached_ns +
                                  tree_prop_recomp_ns + leaf_ns + accum_ns;
            double q0 = now_ns_host();
            if (!run_hybrid_single_q(plan, curr, pts[q].logv, pts[q].w, false,
                                     &block_ns, &tree_build_ns, &tree_prop_cached_ns,
                                     &tree_prop_recomp_ns, &leaf_ns, &accum_ns)) {
                return -1.0;
            }
            double stage_after = block_ns + tree_build_ns + tree_prop_cached_ns +
                                 tree_prop_recomp_ns + leaf_ns + accum_ns;
            quad_ovh_ns += (now_ns_host() - q0) - (stage_after - stage_before);
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
