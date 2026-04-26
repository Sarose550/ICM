/* gpu_kernels.cu -- All __global__ and __device__ kernel functions.
 *
 * cuFFTDx template instantiations and dispatch functions live here
 * because they must be in the same TU where they are used.
 */
#include "gpu_internal.h"

namespace icm_gpu_detail {

/* ── cuFFTDx C2C templates and helpers ─────────────────────────── */

#if ICM_HAVE_CUFFTDX
template<int FFT_N, int FPB = 1>
using cufftdx_fft_fwd_t = decltype(cufftdx::Block() + cufftdx::Size<FFT_N>() +
                                   cufftdx::Type<cufftdx::fft_type::c2c>() +
                                   cufftdx::Direction<cufftdx::fft_direction::forward>() +
                                   cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<FPB>() +
                                   cufftdx::SM<1000>());

template<int FFT_N, int FPB = 1>
using cufftdx_fft_inv_t = decltype(cufftdx::Block() + cufftdx::Size<FFT_N>() +
                                   cufftdx::Type<cufftdx::fft_type::c2c>() +
                                   cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                                   cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<FPB>() +
                                   cufftdx::SM<1000>());

#if ICM_HAVE_CUFFTDX_R2C
template<int FFT_N, int FPB = 1>
using cufftdx_r2c_t = decltype(cufftdx::Block() + cufftdx::Size<FFT_N>() +
                                cufftdx::Type<cufftdx::fft_type::r2c>() +
                                cufftdx::Direction<cufftdx::fft_direction::forward>() +
                                cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<FPB>() +
                                cufftdx::SM<1000>());

template<int FFT_N, int FPB = 1>
using cufftdx_c2r_t = decltype(cufftdx::Block() + cufftdx::Size<FFT_N>() +
                                cufftdx::Type<cufftdx::fft_type::c2r>() +
                                cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                                cufftdx::Precision<double>() + cufftdx::FFTsPerBlock<FPB>() +
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

/* ── C2C fused kernels ─────────────────────────────────────────── */

template<int FFT_N, int FPB = 1>
__launch_bounds__(cufftdx_fft_fwd_t<FFT_N, FPB>::max_threads_per_block)
__global__ static void k_cufftdx_build_parent(const double *child, int cps,
                                              double *parent, int pps,
                                              int nparents, double inv_fft_n,
                                              int child_stride, int parent_stride) {
    using FFTFwd = cufftdx_fft_fwd_t<FFT_N, FPB>;
    using FFTInv = cufftdx_fft_inv_t<FFT_N, FPB>;
    using complex_t = typename FFTFwd::value_type;
    int p = (int)blockIdx.x * FPB + (int)threadIdx.y;
    if (p >= nparents) return;

    complex_t a[FFTFwd::storage_size];
    complex_t b[FFTFwd::storage_size];
    extern __shared__ __align__(alignof(double2)) complex_t shared_mem[];

    const double *L = child + (size_t)(2 * p) * (size_t)child_stride;
    const double *R = child + (size_t)(2 * p + 1) * (size_t)child_stride;
    double *out = parent + (size_t)p * (size_t)parent_stride;

    cufftdx_load_real<FFTFwd>(L, cps, a);
    FFTFwd().execute(a, shared_mem);
    cufftdx_load_real<FFTFwd>(R, cps, b);
    FFTFwd().execute(b, shared_mem);
    cufftdx_mul_freq_inplace<FFTFwd>(a, b);
    FFTInv().execute(a, shared_mem);
    cufftdx_store_real<FFTInv>(a, out, pps, inv_fft_n);
}

template<int FFT_N, int FPB = 1>
__launch_bounds__(cufftdx_fft_fwd_t<FFT_N, FPB>::max_threads_per_block)
__global__ static void k_cufftdx_corr_pair_parent(const double *g_parent, int parent_gsz, int len_g,
                                                  const double *child_poly, int cps, int len_P,
                                                  double *g_child, int child_gsz, int len_out,
                                                  int nparents, double inv_fft_n,
                                                  int g_parent_stride, int poly_child_stride,
                                                  int g_child_stride) {
    using FFTFwd = cufftdx_fft_fwd_t<FFT_N, FPB>;
    using FFTInv = cufftdx_fft_inv_t<FFT_N, FPB>;
    using complex_t = typename FFTFwd::value_type;
    int p = (int)blockIdx.x * FPB + (int)threadIdx.y;
    if (p >= nparents) return;

    complex_t gbuf[FFTFwd::storage_size];
    complex_t pbuf[FFTFwd::storage_size];
    complex_t gspec_saved[FFTFwd::elements_per_thread];
    extern __shared__ __align__(alignof(double2)) complex_t shared_mem[];

    const double *gp = g_parent + (size_t)p * (size_t)g_parent_stride;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)poly_child_stride;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)poly_child_stride;
    double *outL = g_child + (size_t)(2 * p) * (size_t)g_child_stride;
    double *outR = g_child + (size_t)(2 * p + 1) * (size_t)g_child_stride;

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
                                   double inv_fft_n, cudaStream_t stream,
                                   int child_stride, int parent_stride) {
    constexpr int FPB2 = 2;
    using FFTFwd2 = cufftdx_fft_fwd_t<FFT_N, FPB2>;
    using FFTInv2 = cufftdx_fft_inv_t<FFT_N, FPB2>;
    size_t shmem2 = std::max((size_t)FFTFwd2::shared_memory_size, (size_t)FFTInv2::shared_memory_size);
    if (nparents >= 4 && shmem2 <= 200 * 1024) {
        if (CUDA_OK(cudaFuncSetAttribute(k_cufftdx_build_parent<FFT_N, FPB2>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          (int)shmem2))) {
            int grid = (nparents + FPB2 - 1) / FPB2;
            k_cufftdx_build_parent<FFT_N, FPB2><<<grid, FFTFwd2::block_dim, shmem2, stream>>>(
                child, cps, parent, pps, nparents, inv_fft_n, child_stride, parent_stride);
            if (CUDA_OK(cudaGetLastError())) return true;
        }
    }
    using FFTFwd = cufftdx_fft_fwd_t<FFT_N>;
    using FFTInv = cufftdx_fft_inv_t<FFT_N>;
    size_t shmem = std::max((size_t)FFTFwd::shared_memory_size, (size_t)FFTInv::shared_memory_size);
    if (!CUDA_OK(cudaFuncSetAttribute(k_cufftdx_build_parent<FFT_N>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)shmem))) return false;
    k_cufftdx_build_parent<FFT_N><<<nparents, FFTFwd::block_dim, shmem, stream>>>(
        child, cps, parent, pps, nparents, inv_fft_n, child_stride, parent_stride);
    return CUDA_OK(cudaGetLastError());
}

template<int FFT_N>
static bool launch_cufftdx_corr_t(const double *g_parent, int parent_gsz, int len_g,
                                  const double *child_poly, int cps, int len_P,
                                  double *g_child, int child_gsz, int len_out, int nparents,
                                  double inv_fft_n, cudaStream_t stream,
                                  int g_parent_stride, int poly_child_stride,
                                  int g_child_stride) {
    constexpr int FPB2 = 2;
    using FFTFwd2 = cufftdx_fft_fwd_t<FFT_N, FPB2>;
    using FFTInv2 = cufftdx_fft_inv_t<FFT_N, FPB2>;
    size_t shmem2 = std::max((size_t)FFTFwd2::shared_memory_size, (size_t)FFTInv2::shared_memory_size);
    if (nparents >= 4 && shmem2 <= 200 * 1024) {
        if (CUDA_OK(cudaFuncSetAttribute(k_cufftdx_corr_pair_parent<FFT_N, FPB2>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          (int)shmem2))) {
            int grid = (nparents + FPB2 - 1) / FPB2;
            k_cufftdx_corr_pair_parent<FFT_N, FPB2><<<grid, FFTFwd2::block_dim, shmem2, stream>>>(
                g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                g_child, child_gsz, len_out, nparents, inv_fft_n,
                g_parent_stride, poly_child_stride, g_child_stride);
            if (CUDA_OK(cudaGetLastError())) return true;
        }
    }
    using FFTFwd = cufftdx_fft_fwd_t<FFT_N>;
    using FFTInv = cufftdx_fft_inv_t<FFT_N>;
    size_t shmem = std::max((size_t)FFTFwd::shared_memory_size, (size_t)FFTInv::shared_memory_size);
    if (!CUDA_OK(cudaFuncSetAttribute(k_cufftdx_corr_pair_parent<FFT_N>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)shmem))) return false;
    k_cufftdx_corr_pair_parent<FFT_N><<<nparents, FFTFwd::block_dim, shmem, stream>>>(
        g_parent, parent_gsz, len_g,
        child_poly, cps, len_P,
        g_child, child_gsz, len_out, nparents, inv_fft_n,
        g_parent_stride, poly_child_stride, g_child_stride);
    return CUDA_OK(cudaGetLastError());
}
#endif /* ICM_HAVE_CUFFTDX */

/* ── R2C/C2R fused kernels ─────────────────────────────────────── */
#if ICM_HAVE_CUFFTDX_R2C

template<int FFT_N, int FPB = 1>
__launch_bounds__(cufftdx_r2c_t<FFT_N, FPB>::max_threads_per_block)
__global__ static void k_cufftdx_build_parent_r2c(const double *child, int cps,
                                                    double *parent, int pps,
                                                    int nparents, double inv_fft_n,
                                                    int child_stride, int parent_stride) {
    using R2C = cufftdx_r2c_t<FFT_N, FPB>;
    using C2R = cufftdx_c2r_t<FFT_N, FPB>;
    using complex_t = typename R2C::value_type;
    int p = (int)blockIdx.x * FPB + (int)threadIdx.y;
    if (p >= nparents) return;
    complex_t a[R2C::storage_size];
    complex_t b[R2C::storage_size];
    extern __shared__ __align__(alignof(double2)) complex_t shared_mem[];

    const double *L = child + (size_t)(2 * p) * (size_t)child_stride;
    const double *R_ptr = child + (size_t)(2 * p + 1) * (size_t)child_stride;
    double *out = parent + (size_t)p * (size_t)parent_stride;

    cufftdx_load_real_r2c<R2C>(L, cps, a);
    R2C().execute(a, shared_mem);
    cufftdx_load_real_r2c<R2C>(R_ptr, cps, b);
    R2C().execute(b, shared_mem);

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

template<int FFT_N, int FPB = 1>
__launch_bounds__(cufftdx_r2c_t<FFT_N, FPB>::max_threads_per_block)
__global__ static void k_cufftdx_corr_pair_parent_r2c(
        const double *g_parent, int parent_gsz, int len_g,
        const double *child_poly, int cps, int len_P,
        double *g_child, int child_gsz, int len_out,
        int nparents, double inv_fft_n,
        int g_parent_stride, int poly_child_stride,
        int g_child_stride) {
    using R2C = cufftdx_r2c_t<FFT_N, FPB>;
    using C2R = cufftdx_c2r_t<FFT_N, FPB>;
    using complex_t = typename R2C::value_type;
    int p = (int)blockIdx.x * FPB + (int)threadIdx.y;
    if (p >= nparents) return;
    complex_t gbuf[R2C::storage_size];
    complex_t pbuf[R2C::storage_size];
    complex_t gspec_saved[R2C::elements_per_thread];
    extern __shared__ __align__(alignof(double2)) complex_t shared_mem[];

    const double *gp = g_parent + (size_t)p * (size_t)g_parent_stride;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)poly_child_stride;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)poly_child_stride;
    double *outL = g_child + (size_t)(2 * p) * (size_t)g_child_stride;
    double *outR = g_child + (size_t)(2 * p + 1) * (size_t)g_child_stride;

    cufftdx_load_real_r2c<R2C>(gp, len_g, gbuf);
    R2C().execute(gbuf, shared_mem);
    for (unsigned i = 0; i < R2C::elements_per_thread; ++i)
        gspec_saved[i] = gbuf[i];

    cufftdx_load_real_r2c<R2C>(PR, len_P, pbuf);
    R2C().execute(pbuf, shared_mem);
    for (unsigned i = 0; i < R2C::elements_per_thread; ++i) {
        complex_t g = gbuf[i], p_val = pbuf[i];
        gbuf[i].x = g.x * p_val.x + g.y * p_val.y;
        gbuf[i].y = g.y * p_val.x - g.x * p_val.y;
    }
    C2R().execute(gbuf, shared_mem);
    cufftdx_store_real_c2r<C2R>(gbuf, outL, len_out, inv_fft_n);

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
                                        double inv_fft_n, cudaStream_t stream,
                                        int child_stride, int parent_stride) {
    constexpr int FPB2 = 2;
    using R2C2 = cufftdx_r2c_t<FFT_N, FPB2>;
    using C2R2 = cufftdx_c2r_t<FFT_N, FPB2>;
    size_t shmem2 = std::max((size_t)R2C2::shared_memory_size, (size_t)C2R2::shared_memory_size);
    if (nparents >= 4 && shmem2 <= 200 * 1024) {
        if (CUDA_OK(cudaFuncSetAttribute(k_cufftdx_build_parent_r2c<FFT_N, FPB2>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          (int)shmem2))) {
            int grid = (nparents + FPB2 - 1) / FPB2;
            k_cufftdx_build_parent_r2c<FFT_N, FPB2><<<grid, R2C2::block_dim, shmem2, stream>>>(
                child, cps, parent, pps, nparents, inv_fft_n, child_stride, parent_stride);
            if (CUDA_OK(cudaGetLastError())) return true;
        }
    }
    using R2C = cufftdx_r2c_t<FFT_N>;
    using C2R = cufftdx_c2r_t<FFT_N>;
    size_t shmem = std::max((size_t)R2C::shared_memory_size, (size_t)C2R::shared_memory_size);
    if (!CUDA_OK(cudaFuncSetAttribute(k_cufftdx_build_parent_r2c<FFT_N>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)shmem))) return false;
    k_cufftdx_build_parent_r2c<FFT_N><<<nparents, R2C::block_dim, shmem, stream>>>(
        child, cps, parent, pps, nparents, inv_fft_n, child_stride, parent_stride);
    return CUDA_OK(cudaGetLastError());
}

template<int FFT_N>
static bool launch_cufftdx_corr_r2c_t(const double *g_parent, int parent_gsz, int len_g,
                                       const double *child_poly, int cps, int len_P,
                                       double *g_child, int child_gsz, int len_out, int nparents,
                                       double inv_fft_n, cudaStream_t stream,
                                       int g_parent_stride, int poly_child_stride,
                                       int g_child_stride) {
    constexpr int FPB2 = 2;
    using R2C2 = cufftdx_r2c_t<FFT_N, FPB2>;
    using C2R2 = cufftdx_c2r_t<FFT_N, FPB2>;
    size_t shmem2 = std::max((size_t)R2C2::shared_memory_size, (size_t)C2R2::shared_memory_size);
    if (nparents >= 4 && shmem2 <= 200 * 1024) {
        if (CUDA_OK(cudaFuncSetAttribute(k_cufftdx_corr_pair_parent_r2c<FFT_N, FPB2>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          (int)shmem2))) {
            int grid = (nparents + FPB2 - 1) / FPB2;
            k_cufftdx_corr_pair_parent_r2c<FFT_N, FPB2><<<grid, R2C2::block_dim, shmem2, stream>>>(
                g_parent, parent_gsz, len_g,
                child_poly, cps, len_P,
                g_child, child_gsz, len_out, nparents, inv_fft_n,
                g_parent_stride, poly_child_stride, g_child_stride);
            if (CUDA_OK(cudaGetLastError())) return true;
        }
    }
    using R2C = cufftdx_r2c_t<FFT_N>;
    using C2R = cufftdx_c2r_t<FFT_N>;
    size_t shmem = std::max((size_t)R2C::shared_memory_size, (size_t)C2R::shared_memory_size);
    if (!CUDA_OK(cudaFuncSetAttribute(k_cufftdx_corr_pair_parent_r2c<FFT_N>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)shmem))) return false;
    k_cufftdx_corr_pair_parent_r2c<FFT_N><<<nparents, R2C::block_dim, shmem, stream>>>(
        g_parent, parent_gsz, len_g,
        child_poly, cps, len_P,
        g_child, child_gsz, len_out, nparents, inv_fft_n,
        g_parent_stride, poly_child_stride, g_child_stride);
    return CUDA_OK(cudaGetLastError());
}

#endif /* ICM_HAVE_CUFFTDX_R2C */

/* ── Dispatch functions (visible to other TUs via header) ──────── */

bool is_cufftdx_supported_fft_n(int fft_n) {
    switch (fft_n) {
        case 64:
        case 128:
        case 256:
        case 512:
        case 1024:
        case 2048:
        case 4096:
#if GPU_FUSED_MAX_CONV_LEN >= 8192
        case 8192:
#endif
            return true;
        default:
            return false;
    }
}

bool launch_cufftdx_build_dispatch(int fft_n,
                                   const double *child, int cps,
                                   double *parent, int pps, int nparents,
                                   double inv_fft_n, cudaStream_t stream,
                                   int child_stride, int parent_stride) {
#if ICM_HAVE_CUFFTDX
    switch (fft_n) {
        case 64: return launch_cufftdx_build_t<64>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 128: return launch_cufftdx_build_t<128>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 256: return launch_cufftdx_build_t<256>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 512: return launch_cufftdx_build_t<512>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 1024: return launch_cufftdx_build_t<1024>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 2048: return launch_cufftdx_build_t<2048>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 4096: return launch_cufftdx_build_t<4096>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
#if GPU_FUSED_MAX_CONV_LEN >= 8192
        case 8192: return launch_cufftdx_build_t<8192>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
#endif
        default: return false;
    }
#else
    (void)fft_n; (void)child; (void)cps; (void)parent; (void)pps; (void)nparents; (void)inv_fft_n; (void)stream;
    (void)child_stride; (void)parent_stride;
    return false;
#endif
}

bool launch_cufftdx_corr_dispatch(int fft_n,
                                  const double *g_parent, int parent_gsz, int len_g,
                                  const double *child_poly, int cps, int len_P,
                                  double *g_child, int child_gsz, int len_out, int nparents,
                                  double inv_fft_n, cudaStream_t stream,
                                  int g_parent_stride, int poly_child_stride,
                                  int g_child_stride) {
#if ICM_HAVE_CUFFTDX
    switch (fft_n) {
        case 64:
            return launch_cufftdx_corr_t<64>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                             g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                             g_parent_stride, poly_child_stride, g_child_stride);
        case 128:
            return launch_cufftdx_corr_t<128>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                              g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                              g_parent_stride, poly_child_stride, g_child_stride);
        case 256:
            return launch_cufftdx_corr_t<256>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                              g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                              g_parent_stride, poly_child_stride, g_child_stride);
        case 512:
            return launch_cufftdx_corr_t<512>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                              g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                              g_parent_stride, poly_child_stride, g_child_stride);
        case 1024:
            return launch_cufftdx_corr_t<1024>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                               g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                               g_parent_stride, poly_child_stride, g_child_stride);
        case 2048:
            return launch_cufftdx_corr_t<2048>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                               g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                               g_parent_stride, poly_child_stride, g_child_stride);
        case 4096:
            return launch_cufftdx_corr_t<4096>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                               g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                               g_parent_stride, poly_child_stride, g_child_stride);
#if GPU_FUSED_MAX_CONV_LEN >= 8192
        case 8192:
            return launch_cufftdx_corr_t<8192>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                               g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                               g_parent_stride, poly_child_stride, g_child_stride);
#endif
        default:
            return false;
    }
#else
    (void)fft_n; (void)g_parent; (void)parent_gsz; (void)len_g; (void)child_poly; (void)cps; (void)len_P;
    (void)g_child; (void)child_gsz; (void)len_out; (void)nparents; (void)inv_fft_n; (void)stream;
    (void)g_parent_stride; (void)poly_child_stride; (void)g_child_stride;
    return false;
#endif
}

bool launch_cufftdx_build_r2c_dispatch(int fft_n,
                                       const double *child, int cps,
                                       double *parent, int pps, int nparents,
                                       double inv_fft_n, cudaStream_t stream,
                                       int child_stride, int parent_stride) {
#if ICM_HAVE_CUFFTDX_R2C
    switch (fft_n) {
        case 64: return launch_cufftdx_build_r2c_t<64>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 128: return launch_cufftdx_build_r2c_t<128>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 256: return launch_cufftdx_build_r2c_t<256>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 512: return launch_cufftdx_build_r2c_t<512>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 1024: return launch_cufftdx_build_r2c_t<1024>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 2048: return launch_cufftdx_build_r2c_t<2048>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
        case 4096: return launch_cufftdx_build_r2c_t<4096>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
#if GPU_FUSED_MAX_CONV_LEN >= 8192
        case 8192: return launch_cufftdx_build_r2c_t<8192>(child, cps, parent, pps, nparents, inv_fft_n, stream, child_stride, parent_stride);
#endif
        default: return false;
    }
#else
    (void)fft_n; (void)child; (void)cps; (void)parent; (void)pps; (void)nparents; (void)inv_fft_n; (void)stream;
    (void)child_stride; (void)parent_stride;
    return false;
#endif
}

bool launch_cufftdx_corr_r2c_dispatch(int fft_n,
                                      const double *g_parent, int parent_gsz, int len_g,
                                      const double *child_poly, int cps, int len_P,
                                      double *g_child, int child_gsz, int len_out, int nparents,
                                      double inv_fft_n, cudaStream_t stream,
                                      int g_parent_stride, int poly_child_stride,
                                      int g_child_stride) {
#if ICM_HAVE_CUFFTDX_R2C
    switch (fft_n) {
        case 64:
            return launch_cufftdx_corr_r2c_t<64>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                   g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                                   g_parent_stride, poly_child_stride, g_child_stride);
        case 128:
            return launch_cufftdx_corr_r2c_t<128>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                    g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                                    g_parent_stride, poly_child_stride, g_child_stride);
        case 256:
            return launch_cufftdx_corr_r2c_t<256>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                    g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                                    g_parent_stride, poly_child_stride, g_child_stride);
        case 512:
            return launch_cufftdx_corr_r2c_t<512>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                    g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                                    g_parent_stride, poly_child_stride, g_child_stride);
        case 1024:
            return launch_cufftdx_corr_r2c_t<1024>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                     g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                                     g_parent_stride, poly_child_stride, g_child_stride);
        case 2048:
            return launch_cufftdx_corr_r2c_t<2048>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                     g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                                     g_parent_stride, poly_child_stride, g_child_stride);
        case 4096:
            return launch_cufftdx_corr_r2c_t<4096>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                     g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                                     g_parent_stride, poly_child_stride, g_child_stride);
#if GPU_FUSED_MAX_CONV_LEN >= 8192
        case 8192:
            return launch_cufftdx_corr_r2c_t<8192>(g_parent, parent_gsz, len_g, child_poly, cps, len_P,
                                                     g_child, child_gsz, len_out, nparents, inv_fft_n, stream,
                                                     g_parent_stride, poly_child_stride, g_child_stride);
#endif
        default:
            return false;
    }
#else
    (void)fft_n; (void)g_parent; (void)parent_gsz; (void)len_g; (void)child_poly; (void)cps; (void)len_P;
    (void)g_child; (void)child_gsz; (void)len_out; (void)nparents; (void)inv_fft_n; (void)stream;
    (void)g_parent_stride; (void)poly_child_stride; (void)g_child_stride;
    return false;
#endif
}

/* ── Standard __global__ kernels ───────────────────────────────── */

__global__ void k_compute_a(const double *S_sorted, double *a_sorted, int n, double logv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double arg = S_sorted[i] * logv;
    a_sorted[i] = (arg < -700.0) ? 0.0 : exp(arg);
}

__global__ void k_compute_a_from_ptr(const double *S_sorted, double *a_sorted,
                                     int n, const double *logv_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double logv = logv_ptr[0];
    double arg = S_sorted[i] * logv;
    a_sorted[i] = (arg < -700.0) ? 0.0 : exp(arg);
}

__global__ void k_compute_a_qbatch(const double *S_sorted,
                                   double * const *a_ptrs,
                                   const double *logv_array,
                                   int n, int qb) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = qb * n;
    if (global_idx >= total) return;
    int qi = global_idx / n;
    int i = global_idx % n;
    double arg = S_sorted[i] * logv_array[qi];
    a_ptrs[qi][i] = (arg < -700.0) ? 0.0 : exp(arg);
}

__global__ void k_set_leaves_b1_qbatch(double * const *a_ptrs,
                                       int n, int N_tree,
                                       int leaf_psz, int qb,
                                       double *leaves, size_t leaf_stride) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = qb * N_tree;
    if (global_idx >= total) return;
    int qi = global_idx / N_tree;
    int i = global_idx % N_tree;
    double *leaf = leaves + qi * leaf_stride + (size_t)i * (size_t)leaf_psz;
    if (i < n) {
        double ai = a_ptrs[qi][i];
        leaf[0] = ai;
        if (leaf_psz > 1) leaf[1] = 1.0 - ai;
        for (int m = 2; m < leaf_psz; ++m) leaf[m] = 0.0;
    } else {
        leaf[0] = 1.0;
        for (int m = 1; m < leaf_psz; ++m) leaf[m] = 0.0;
    }
}

__global__ void k_zero(double *x, size_t n) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i < n) x[i] = 0.0;
}

__global__ void k_set_root_g(double *g_root, int root_gsz, const double *payout, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= root_gsz) return;
    g_root[i] = (i < k) ? payout[i] : 0.0;
}

__global__ void k_set_leaves_b1(const double *a_sorted, int n, int N_tree,
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
        leaf[0] = 1.0;
        for (int m = 1; m < leaf_psz; ++m) leaf[m] = 0.0;
    }
}

__global__ void k_leaf_extract_b1(int n, const double *g_leaf,
                                  int leaf_psz, double *inner_sorted) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    inner_sorted[i] = g_leaf[(size_t)i * (size_t)leaf_psz];
}

__global__ void k_leaf_extract_b1_qbatch(int n, const double *g_leaf,
                                         int leaf_psz, int qb,
                                         size_t g_stride, double *inner,
                                         size_t inner_stride) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = qb * n;
    if (global_idx >= total) return;
    int qi = global_idx / n;
    int i = global_idx % n;
    inner[qi * inner_stride + i] = g_leaf[qi * g_stride + (size_t)i * (size_t)leaf_psz];
}

__global__ void k_block_build(const double *a_sorted, int n, int B,
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

__global__ void k_schoolbook_build(const double *child, int cps,
                                   double *parent, int pps, int nparents,
                                   int child_stride, int parent_stride) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)nparents * (size_t)parent_stride;
    if (idx >= total) return;
    int p = (int)(idx / (size_t)parent_stride);
    int m = (int)(idx % (size_t)parent_stride);
    if (m >= pps) {
        parent[(size_t)p * (size_t)parent_stride + (size_t)m] = 0.0;
        return;
    }
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

__global__ void k_schoolbook_build_smem_parent(const double *child, int cps,
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
    for (int m = threadIdx.x; m < parent_stride; m += blockDim.x) {
        if (m >= pps) { out[m] = 0.0; continue; }
        int j_lo = m - (cps - 1);
        if (j_lo < 0) j_lo = 0;
        int j_hi = m;
        if (j_hi > cps - 1) j_hi = cps - 1;
        double sum = 0.0;
        for (int j = j_lo; j <= j_hi; ++j) sum += Lsh[j] * Rsh[m - j];
        out[m] = sum;
    }
}

__global__ void k_schoolbook_build_warp_batch(const double *child, int cps,
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
    for (int m = lane; m < parent_stride; m += WARP) {
        if (m >= pps) { out[m] = 0.0; continue; }
        int j_lo = m - (cps - 1);
        if (j_lo < 0) j_lo = 0;
        int j_hi = m;
        if (j_hi > cps - 1) j_hi = cps - 1;
        double sum = 0.0;
        for (int j = j_lo; j <= j_hi; ++j) sum += Lsh[j] * Rsh[m - j];
        out[m] = sum;
    }
}

__global__ void k_pairwise_mul(const cufftDoubleComplex * __restrict__ child_spec, int cn,
                    cufftDoubleComplex * __restrict__ parent_spec, int nparents,
                    double scale) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)nparents * (size_t)cn;
    if (idx >= total) return;
    int p = (int)(idx / (size_t)cn);
    int f = (int)(idx % (size_t)cn);
    cufftDoubleComplex a = child_spec[(size_t)(2 * p) * (size_t)cn + (size_t)f];
    cufftDoubleComplex b = child_spec[(size_t)(2 * p + 1) * (size_t)cn + (size_t)f];
    cufftDoubleComplex o;
    o.x = (a.x * b.x - a.y * b.y) * scale;
    o.y = (a.x * b.y + a.y * b.x) * scale;
    parent_spec[idx] = o;
}

__global__ void k_scale_zero_pad(double *data, int fft_stride, int valid_len,
                                 double inv_fft_n, int batch) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)batch * (size_t)fft_stride;
    if (idx >= total) return;
    int m = (int)(idx % (size_t)fft_stride);
    if (m < valid_len) {
        data[idx] *= inv_fft_n;
    } else {
        data[idx] = 0.0;
    }
}

__global__ void k_zero_pad(double *data, int fft_stride, int valid_len, int batch) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)batch * (size_t)fft_stride;
    if (idx >= total) return;
    int m = (int)(idx % (size_t)fft_stride);
    if (m >= valid_len) data[idx] = 0.0;
}

__global__ void k_wrap_build(double *parent, int pps, int nparents,
                             const double *child, int cps, int conv_len,
                             int fft_n, int wrap_m,
                             int parent_stride, int child_stride) {
    int p = blockIdx.x;
    if (p >= nparents) return;
    double *out = parent + (size_t)p * (size_t)parent_stride;
    const double *L = child + (size_t)(2 * p) * (size_t)child_stride;
    const double *R = child + (size_t)(2 * p + 1) * (size_t)child_stride;
    int da = cps - 1;
    int db = cps - 1;
    for (int i = threadIdx.x; i <= wrap_m; i += blockDim.x) {
        int pos = fft_n + i;
        if (pos >= conv_len) continue;
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

__global__ void k_paired_corr_freq(const cufftDoubleComplex * __restrict__ g_hat,
                        const cufftDoubleComplex * __restrict__ cached_child_spec,
                        int cn, int nparents,
                        cufftDoubleComplex * __restrict__ child_out_spec,
                        double scale) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = (size_t)nparents * (size_t)cn;
    if (idx >= total) return;
    int p = (int)(idx / (size_t)cn);
    int f = (int)(idx % (size_t)cn);

    cufftDoubleComplex g = g_hat[idx];
    cufftDoubleComplex specL = cached_child_spec[(size_t)(2 * p) * (size_t)cn + (size_t)f];
    cufftDoubleComplex specR = cached_child_spec[(size_t)(2 * p + 1) * (size_t)cn + (size_t)f];

    cufftDoubleComplex out_left;
    out_left.x = (g.x * specR.x + g.y * specR.y) * scale;
    out_left.y = (g.y * specR.x - g.x * specR.y) * scale;

    cufftDoubleComplex out_right;
    out_right.x = (g.x * specL.x + g.y * specL.y) * scale;
    out_right.y = (g.y * specL.x - g.x * specL.y) * scale;

    child_out_spec[(size_t)(2 * p) * (size_t)cn + (size_t)f] = out_left;
    child_out_spec[(size_t)(2 * p + 1) * (size_t)cn + (size_t)f] = out_right;
}

__global__ void k_wrap_corr_pair(double *g_child, int child_gsz, int nparents,
                                 const double *g_parent, int parent_gsz, int len_g,
                                 const double *child_poly, int cps, int len_P,
                                 int len_out,
                                 int fft_n, int wrap_m,
                                 int child_g_stride, int parent_g_stride,
                                 int child_poly_stride) {
    int p = blockIdx.x;
    if (p >= nparents) return;

    double *outL = g_child + (size_t)(2 * p) * (size_t)child_g_stride;
    double *outR = g_child + (size_t)(2 * p + 1) * (size_t)child_g_stride;
    const double *gp = g_parent + (size_t)p * (size_t)parent_g_stride;
    const double *PL = child_poly + (size_t)(2 * p) * (size_t)child_poly_stride;
    const double *PR = child_poly + (size_t)(2 * p + 1) * (size_t)child_poly_stride;

    int conv_len = len_g + len_P - 1;

    for (int i = threadIdx.x; i <= wrap_m; i += blockDim.x) {
        int pos = fft_n + i;
        if (pos >= conv_len) continue;
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
    int m_end = len_out < fft_n ? len_out : fft_n;
    for (int m = m_start + threadIdx.x; m < m_end; m += blockDim.x) {
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

__global__ void k_schoolbook_corr_pair(const double *g_parent, int parent_gsz,
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

__global__ void k_schoolbook_corr_pair_smem_parent(const double *g_parent, int parent_gsz,
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

__global__ void k_schoolbook_corr_pair_warp_batch(const double *g_parent, int parent_gsz,
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

__global__ void k_leaf_extract(const double *a_sorted, int n, int B, int nblocks,
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
            double c = -bj * ia;
            double q = P_b[0] * ia;
            eq = g_b[0] * q;
            for (int m = 1; m < pk_g; ++m) {
                q = c * q + P_b[m] * ia;
                eq += g_b[m] * q;
            }
        } else if (aj > 1e-15) {
            double ib = 1.0 / bj;
            double c = -aj * ib;
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

__global__ void k_accumulate_equity(const double *inner_sorted,
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

__global__ void k_accumulate_equity_scaled(const double *inner_sorted,
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

__global__ void k_icm_single_kernel(
        const double *S_sorted, const int *sort_perm, int n,
        int Q, const double *d_logv, const double *d_weights,
        const double *payout, int k,
        double *equity,
        int N, int L, const int *d_nn, const int *d_psz, const int *d_g_needed,
        const size_t *d_plev_off, int total_poly, int max_g) {

    int q = blockIdx.x;
    if (q >= Q) return;

    double logv = d_logv[q];
    double w = d_weights[q];
    if (w == 0.0) return;

    extern __shared__ double smem[];
    double *poly = smem;
    double *g0   = smem + total_poly;
    double *g1   = g0 + max_g;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    /* 1. Set leaves */
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

    /* 2. Tree build */
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

    /* 3. Set root g */
    int top = L - 1;
    int root_gsz = d_psz[top];
    double *g_parent = g0;
    for (int m = tid; m < root_gsz; m += nthreads)
        g_parent[m] = (m < k) ? payout[m] : 0.0;
    __syncthreads();

    /* 4. Tree propagate */
    for (int ell = top; ell >= 1; ell--) {
        int cps = d_psz[ell - 1];
        int pgsz = d_psz[ell];
        int nn_parent = d_nn[ell];
        int out_needed = d_g_needed[ell - 1];
        int p_eff = cps;
        double *child_base = poly + d_plev_off[ell - 1];
        double *g_child = (g_parent == g0) ? g1 : g0;

        for (int j = tid; j < nn_parent; j += nthreads) {
            double *gp = g_parent + (size_t)j * pgsz;
            double *PL = child_base + (size_t)(2 * j) * cps;
            double *PR = child_base + (size_t)(2 * j + 1) * cps;
            double *gL = g_child + (size_t)(2 * j) * cps;
            double *gR = g_child + (size_t)(2 * j + 1) * cps;

            int len_g = pgsz;
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

    /* 5. Extract and accumulate */
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

/* ── Q-batch kernels ──────────────────────────────────────────── */

__global__ void k_block_build_qbatch(
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

__global__ void k_set_root_g_qbatch(double *g_root, int root_gsz,
                                    const double *payout, int k,
                                    int q_batch, size_t g_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = q_batch * root_gsz;
    if (idx >= total) return;
    int qi = idx / root_gsz;
    int i = idx % root_gsz;
    g_root[(size_t)qi * g_stride + (size_t)i] = (i < k) ? payout[i] : 0.0;
}

__global__ void k_leaf_extract_qbatch(
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
            double c = -bj * ia;
            double q = P_b[0] * ia;
            eq = g_b[0] * q;
            for (int m = 1; m < pk_g; ++m) {
                q = c * q + P_b[m] * ia;
                eq += g_b[m] * q;
            }
        } else if (aj > 1e-15) {
            double ib = 1.0 / bj;
            double c = -aj * ib;
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

__global__ void k_accumulate_equity_qbatch(
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

__global__ void k_leaf_extract_qbatch_masked(
        const double * const *a_ptrs, int n, int B, int nblocks,
        const double *block_prods, size_t bp_stride,
        const double *g_leaf, int leaf_psz, size_t leaf_g_stride,
        int g_need, int k, int q_batch,
        double *inner_sorted, size_t inner_stride,
        const uint8_t *active_mask) {
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
        if (!active_mask[j]) { inner_out[j] = 0.0; continue; }

        double aj = a_sorted[j];
        double bj = 1.0 - aj;
        double eq = 0.0;

        if (aj > 0.5) {
            double ia = 1.0 / aj;
            double c = -bj * ia;
            double q = P_b[0] * ia;
            eq = g_b[0] * q;
            for (int m = 1; m < pk_g; ++m) {
                q = c * q + P_b[m] * ia;
                eq += g_b[m] * q;
            }
        } else if (aj > 1e-15) {
            double ib = 1.0 / bj;
            double c = -aj * ib;
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

__global__ void k_accumulate_equity_qbatch_masked(
        const double *inner_sorted, size_t inner_stride,
        const double * const *a_ptrs,
        const double *S_sorted,
        const int *sort_perm,
        int n, const double *weights, const double *inv_vs,
        int q_batch, double *equity,
        const uint8_t *active_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = q_batch * n;
    if (idx >= total) return;
    int qi = idx / n;
    int i = idx % n;
    if (!active_mask[i]) return;
    int orig = sort_perm[i];
    double w = weights[qi];
    double inv_v = inv_vs[qi];
    double pw = w * S_sorted[i] * a_ptrs[qi][i] * inv_v;
    if (!isfinite(pw)) pw = 0.0;
    atomicAdd(&equity[orig], pw * inner_sorted[(size_t)qi * inner_stride + (size_t)i]);
}

}  // namespace icm_gpu_detail
