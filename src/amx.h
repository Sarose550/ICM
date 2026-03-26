/*
 * amx.h — Apple AMX FP64 outer-product primitives for ICM
 *
 * Undocumented Apple Silicon coprocessor instructions via inline assembly.
 * Based on reverse engineering by Dougall Johnson, Maynard Handley, and
 * Peter Cawley (corsix/amx).  Apple does not document or support these
 * instructions — they may change in future hardware.
 *
 * Guard: compiles to nothing on non-Apple or non-aarch64.
 */

#ifndef ICM_AMX_H
#define ICM_AMX_H

#if defined(__APPLE__) && defined(__aarch64__)
#define HAS_AMX 1

#include <stdint.h>
#include <string.h>

/* ── Raw instruction emission ─────────────────────────────── */

#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm volatile("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" \
                   : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr) \
    __asm volatile(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" \
                   : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

/* ── Instruction mnemonics ────────────────────────────────── */

#define AMX_SET()       AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR()       AMX_NOP_OP_IMM5(17, 1)
#define AMX_LDX(gpr)    AMX_OP_GPR(0, gpr)
#define AMX_LDY(gpr)    AMX_OP_GPR(1, gpr)
#define AMX_STX(gpr)    AMX_OP_GPR(2, gpr)
#define AMX_STY(gpr)    AMX_OP_GPR(3, gpr)
#define AMX_LDZ(gpr)    AMX_OP_GPR(4, gpr)
#define AMX_STZ(gpr)    AMX_OP_GPR(5, gpr)
#define AMX_LDZI(gpr)   AMX_OP_GPR(6, gpr)
#define AMX_STZI(gpr)   AMX_OP_GPR(7, gpr)
#define AMX_EXTRX(gpr)  AMX_OP_GPR(8, gpr)    /* extrh / extrx */
#define AMX_EXTRY(gpr)  AMX_OP_GPR(9, gpr)    /* extrv / extry */
#define AMX_FMA64(gpr)  AMX_OP_GPR(10, gpr)
#define AMX_FMS64(gpr)  AMX_OP_GPR(11, gpr)
#define AMX_FMA32(gpr)  AMX_OP_GPR(12, gpr)
#define AMX_FMS32(gpr)  AMX_OP_GPR(13, gpr)
#define AMX_MAC16(gpr)  AMX_OP_GPR(14, gpr)
#define AMX_FMA16(gpr)  AMX_OP_GPR(15, gpr)
#define AMX_FMS16(gpr)  AMX_OP_GPR(16, gpr)
#define AMX_VECINT(gpr) AMX_OP_GPR(18, gpr)
#define AMX_VECFP(gpr)  AMX_OP_GPR(19, gpr)
#define AMX_MATINT(gpr) AMX_OP_GPR(20, gpr)
#define AMX_MATFP(gpr)  AMX_OP_GPR(21, gpr)
#define AMX_GENLUT(gpr) AMX_OP_GPR(22, gpr)

/* ── FP64 constants ───────────────────────────────────────── */

/* Z row stride for FP64 outer products.
 * 4x4 subgrid → 8 result rows spaced 8 apart: Z[base, base+8, ..., base+56].
 * Gives 8 independent accumulators (base = 0..7). */
#define AMX_FP64_Z_STRIDE  8

/* ── Operand encoding: address mask ───────────────────────── */

#define AMX_ADDR(p) ((uint64_t)(p) & 0x00FFFFFFFFFFFFFFULL)

/* ── LDX / LDY / STX / STY ───────────────────────────────
 * bits 55:0  = memory address (64-byte aligned)
 * bits 58:56 = register index (0-7)
 * bit  62    = pair mode (load 128 bytes into 2 consecutive regs)
 * ─────────────────────────────────────────────────────────── */

static inline uint64_t amx_ldx_op(const void *ptr, int reg) {
    return AMX_ADDR(ptr) | ((uint64_t)(reg & 7) << 56);
}
#define amx_ldy_op amx_ldx_op
#define amx_stx_op amx_ldx_op
#define amx_sty_op amx_ldx_op

/* ── LDZ / STZ / LDZI / STZI ─────────────────────────────
 * bits 55:0  = memory address (64-byte aligned)
 * bits 61:56 = Z row index (0-63)
 * bit  62    = pair mode (load/store 2 consecutive rows)
 * ─────────────────────────────────────────────────────────── */

static inline uint64_t amx_ldz_op(const void *ptr, int row) {
    return AMX_ADDR(ptr) | ((uint64_t)(row & 63) << 56);
}
#define amx_stz_op amx_ldz_op

/* ── FMA64 (matrix mode = 8×8 FP64 outer product) ────────
 *   Z[j][i] += X[i] * Y[j]   for i,j ∈ [0,8)
 *
 * bits  0-8:  Y byte offset (reg * 64, into 512-byte Y file)
 * bits 10-18: X byte offset (reg * 64, into 512-byte X file)
 * bits 20-25: Z row base (result row j → Z[base + j*8])
 * bit  27:    1 = overwrite (Z = X⊗Y), 0 = accumulate (Z += X⊗Y)
 * ─────────────────────────────────────────────────────────── */

static inline uint64_t amx_fma64_outer(int x_reg, int y_reg,
                                        int z_row_base, int no_accum) {
    return ((uint64_t)(y_reg & 7) * 64)
         | (((uint64_t)(x_reg & 7) * 64) << 10)
         | ((uint64_t)(z_row_base & 63) << 20)
         | ((uint64_t)(no_accum & 1) << 27);
}

/* ── EXTRH: copy Z row → X register (op 8, bit 26 = 0) ───
 * bits 16-18: X register index (0-7)
 * bits 20-25: Z row to extract
 * (bit 26 = 0 for copy-only mode)
 * ─────────────────────────────────────────────────────────── */

static inline uint64_t amx_extrh_x_op(int z_row, int x_reg) {
    return ((uint64_t)(x_reg & 7) << 16)
         | ((uint64_t)(z_row & 63) << 20);
}

/* ── EXTRH: copy Z row → Y register (op 8, bit 26 = 1, bit 10 = 1) ─
 * bits  0-8:  Y byte offset for destination (reg * 64)
 * bits 20-25: Z row to extract
 * ─────────────────────────────────────────────────────────── */

static inline uint64_t amx_extrh_y_op(int z_row, int y_byte_off) {
    return ((uint64_t)(y_byte_off & 0x1FF))
         | ((uint64_t)(z_row & 63) << 20)
         | (1ULL << 26) | (1ULL << 10);
}

/* ── VECFP: z[row][i] ±= f(x[i], y[i])  ─────────────────
 * bits  0-8:  Y byte offset
 * bits 10-18: X byte offset
 * bits 20-25: Z row
 * bits 42-45: lane width mode (7 = f64)
 * bits 47-42: ALU mode (0 = z+x*y, 1 = z-x*y)
 *
 * For z += x (addition only): use ALU mode 0 with Y pre-loaded to 1.0
 *   → z[row][i] += x[i] * 1.0 = z[row][i] += x[i]
 * ─────────────────────────────────────────────────────────── */

static inline uint64_t amx_vecfp_fma_f64(int x_byte_off, int y_byte_off,
                                           int z_row) {
    return ((uint64_t)(y_byte_off & 0x1FF))
         | ((uint64_t)(x_byte_off & 0x1FF) << 10)
         | ((uint64_t)(z_row & 63) << 20)
         | (7ULL << 42);  /* lane width mode 7 = f64 */
}

/* ══════════════════════════════════════════════════════════════
   HIGH-LEVEL HELPERS
   ══════════════════════════════════════════════════════════════ */

/* Zero a single Z row */
static inline void amx_zero_z_row(int row, const void *zero_buf) {
    AMX_LDZ(amx_ldz_op(zero_buf, row));
}

/* Zero all 8 Z rows used by an FP64 accumulator */
static inline void amx_zero_fp64_accum(int z_base, const void *zero_buf) {
    for (int j = 0; j < 8; j++)
        AMX_LDZ(amx_ldz_op(zero_buf, z_base + j * AMX_FP64_Z_STRIDE));
}

/* Store 8×8 FP64 tile from Z accumulator to flat buffer[64].
 * buf[j*8 + i] = Z[z_base + j*8][i]. */
static inline void amx_store_fp64_tile(double *buf, int z_base) {
    for (int j = 0; j < 8; j++)
        AMX_STZ(amx_stz_op(buf + j * 8, z_base + j * AMX_FP64_Z_STRIDE));
}

/* ══════════════════════════════════════════════════════════════
   ANTI-DIAGONAL EXTRACTION (VECFP in-register, 4.7x faster than STZ+scalar)

   After outer products are accumulated in Z[0, 8, 16, ..., 56]:
   1. Z[0] has j=0 contributions (no shift needed)
   2. For j=1..7: EXTRH Z[j*8]→Y1, then VECFP to accumulate
      shifted Y into Z[0] (low anti-diags) and Z[1] (high anti-diags)
   3. Store Z[0] (c[0..7]) and Z[1] (c[8..14]) via STZ

   Uses Y-path extraction: X0=ones, Y0,Y2=zeros, data in Y1.
   VECFP: Z[row] += X[0:ones] * Y[shifted] = Z[row] += Y[shifted].
   ══════════════════════════════════════════════════════════════ */

/* Pre-load registers for VECFP extraction. Call once before extraction loop.
 * Loads: X0=ones, Y0=zeros, Y2=zeros. */
static inline void amx_extract_setup(const double *ones_buf, const double *zero_buf) {
    AMX_LDX(amx_ldx_op(ones_buf, 0));
    AMX_LDY(amx_ldy_op(zero_buf, 0));
    AMX_LDY(amx_ldy_op(zero_buf, 2));
}

/* VECFP in-register anti-diagonal extraction from Z accumulator z_base.
 * Results: c_lo[0..7] = anti-diags 0-7, c_hi[0..6] = anti-diags 8-14.
 * Requires: X0=ones, Y0=zeros, Y2=zeros (call amx_extract_setup first).
 * Clobbers: Y1, Z[1]. */
static inline void amx_extract_antidiag_vecfp(double *c_lo, double *c_hi,
                                               int z_base,
                                               const double *zero_buf) {
    /* Zero the high accumulator */
    AMX_LDZ(amx_ldz_op(zero_buf, 1));

    /* Z[z_base] already has j=0 contributions in the right positions.
     * For j=1..7: extract Z[z_base + j*8] → Y1, then shift+add. */
    for (int j = 1; j < 8; j++) {
        AMX_EXTRX(amx_extrh_y_op(z_base + j * AMX_FP64_Z_STRIDE, 64));
        /* Low part: right-shifted by j. Y offset = 64 - j*8 */
        AMX_VECFP(amx_vecfp_fma_f64(0, 64 - j * 8, 0));
        /* High part: overflow. Y offset = 128 - j*8 */
        AMX_VECFP(amx_vecfp_fma_f64(0, 128 - j * 8, 1));
    }

    AMX_STZ(amx_stz_op(c_lo, 0));
    AMX_STZ(amx_stz_op(c_hi, 1));
}

/* ══════════════════════════════════════════════════════════════
   ANTI-DIAGONAL EXTRACTION (STZ + scalar, fallback)

   Given an 8×8 outer product tile in buf[j*8+i] = a[p+i]*b[q+j],
   extract anti-diagonal sums: c[base+d] += Σ_{i+j=d} buf[j*8+i]
   for d = 0..14.  Caller must ensure base+14 < nc OR pass nc for
   bounds checking.
   ══════════════════════════════════════════════════════════════ */

static inline void amx_extract_antidiag(const double *tile, double *c,
                                         int base, int nc) {
    for (int d = 0; d < 15; d++) {
        int idx = base + d;
        if (idx >= nc) break;
        double sum = 0;
        int jlo = d > 7 ? d - 7 : 0;
        int jhi = d < 8 ? d : 7;
        for (int j = jlo; j <= jhi; j++)
            sum += tile[j * 8 + (d - j)];
        c[idx] += sum;
    }
}

/* Column-sum extraction for correlate:
 * tile[j*8+i] = g[m_base+i] * P[j_base+j]
 * out[m_base+i] += Σ_j tile[j*8+i]
 *
 * This is simpler than anti-diagonal — just sum along rows for each column. */
static inline void amx_extract_colsum(const double *tile, double *out,
                                       int m_base, int len_out,
                                       int jmax) {
    for (int i = 0; i < 8 && m_base + i < len_out; i++) {
        double sum = 0;
        for (int j = 0; j < jmax; j++)
            sum += tile[j * 8 + i];
        out[m_base + i] += sum;
    }
}

#else  /* not Apple aarch64 */
#define HAS_AMX 0
#endif

#endif /* ICM_AMX_H */
