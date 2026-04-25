/* gpu_lto_callback.cu -- cuFFT LTO callback device code for fused multiply.
 *
 * Compiled separately to LTO-IR fatbin, then linked into cuFFT plans at
 * plan creation time via cufftXtSetJITCallback. This fuses the pointwise
 * complex multiply into the C2R inverse FFT, eliminating k_pairwise_mul
 * and one global memory round-trip per tree level.
 *
 * Build:
 *   nvcc --std=c++17 --generate-code arch=compute_100,code=lto_100 \
 *       -dc -fatbin -o gpu_lto_callback.fatbin gpu_lto_callback.cu
 *   bin2c --name gpu_lto_callback_fatbin --type char \
 *       gpu_lto_callback.fatbin > gpu_lto_callback_fatbin.h
 */

#include <cufft.h>

/* ── Build multiply callback ─────────────────────────────────────
 * Fuses sibling-pair complex multiply into the C2R inverse FFT load.
 *
 * The R2C forward FFT writes 2*nparents batches of cn complex elements
 * to spec_in (children interleaved as [L0, R0, L1, R1, ...]).
 * The C2R inverse FFT reads nparents batches of cn complex elements.
 * This callback intercepts each C2R load and computes L[f]*R[f]*scale
 * on the fly, so the separate k_pairwise_mul kernel is eliminated.
 *
 * NOTE: cuFFT C2R load callbacks may fire >1x per element. Our callback
 * is idempotent (pure read-only), so this is safe.
 */
struct BuildMulCBData {
    int cn;            /* complex elements per batch (fft_n/2 + 1) */
    double inv_fft_n;  /* 1.0 / fft_n scaling factor */
};

extern "C" __device__ cufftDoubleComplex icm_build_mul_load_cb(
    void *dataIn, unsigned long long offset,
    void *callerInfo, void * /*sharedPtr*/)
{
    const cufftDoubleComplex *spec = static_cast<const cufftDoubleComplex *>(dataIn);
    const BuildMulCBData *p = static_cast<const BuildMulCBData *>(callerInfo);
    int parent = static_cast<int>(offset / static_cast<unsigned long long>(p->cn));
    int freq   = static_cast<int>(offset % static_cast<unsigned long long>(p->cn));
    cufftDoubleComplex a = spec[static_cast<size_t>(2 * parent) * p->cn + freq];
    cufftDoubleComplex b = spec[static_cast<size_t>(2 * parent + 1) * p->cn + freq];
    cufftDoubleComplex r;
    r.x = (a.x * b.x - a.y * b.y) * p->inv_fft_n;
    r.y = (a.x * b.y + a.y * b.x) * p->inv_fft_n;
    return r;
}

/* ── Correlation callback ────────────────────────────────────────
 * Fuses paired correlation multiply into the C2R inverse FFT load.
 *
 * The propagation path computes, for each parent p:
 *   g_left[f]  = conj(g_hat[f]) * child_spec_R[f] * scale
 *   g_right[f] = conj(g_hat[f]) * child_spec_L[f] * scale
 *
 * The C2R inverse has batch = 2*nparents (interleaved [L0, R0, L1, R1, ...]).
 * Child index = offset / cn, parent = child / 2, is_right = child & 1.
 * For the left child (even): multiply g with right sibling's spec.
 * For the right child (odd): multiply g with left sibling's spec.
 */
struct CorrCBData {
    const cufftDoubleComplex *g_hat;        /* parent FFT spectra */
    const cufftDoubleComplex *child_spec;   /* cached child spectra from build */
    int cn;                                 /* complex elements per batch */
    int g_cn;                               /* complex elements per g batch (may differ) */
    double inv_fft_n;
};

extern "C" __device__ cufftDoubleComplex icm_corr_load_cb(
    void *dataIn, unsigned long long offset,
    void *callerInfo, void * /*sharedPtr*/)
{
    (void)dataIn;
    const CorrCBData *p = static_cast<const CorrCBData *>(callerInfo);
    int child  = static_cast<int>(offset / static_cast<unsigned long long>(p->cn));
    int freq   = static_cast<int>(offset % static_cast<unsigned long long>(p->cn));
    int parent = child >> 1;
    int is_right = child & 1;

    cufftDoubleComplex g = p->g_hat[static_cast<size_t>(parent) * p->g_cn + freq];
    /* For left child: correlate with RIGHT sibling's spec
     * For right child: correlate with LEFT sibling's spec */
    int sibling = is_right ? (2 * parent) : (2 * parent + 1);
    cufftDoubleComplex spec = p->child_spec[static_cast<size_t>(sibling) * p->cn + freq];

    /* Conjugate correlation: conj(g) * spec = (g.x*spec.x + g.y*spec.y, g.y*spec.x - g.x*spec.y) */
    cufftDoubleComplex r;
    r.x = (g.x * spec.x + g.y * spec.y) * p->inv_fft_n;
    r.y = (g.y * spec.x - g.x * spec.y) * p->inv_fft_n;
    return r;
}
