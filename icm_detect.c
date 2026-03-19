/*
 * icm_detect.c — Runtime CPU feature detection and backend dispatch
 *
 * Uses CPUID + XGETBV to detect AVX2, FMA, AVX-512F/DQ at runtime.
 * Returns the best available ICM function pointer for the current machine.
 *
 * Also reports CPU model name and detected features for benchmark headers.
 */
#include "icm.h"
#include <stdio.h>
#include <string.h>

#ifdef _MSC_VER
#include <intrin.h>
static void cpuid(int leaf, int subleaf, unsigned int *eax, unsigned int *ebx,
                  unsigned int *ecx, unsigned int *edx) {
    int regs[4];
    __cpuidex(regs, leaf, subleaf);
    *eax = regs[0]; *ebx = regs[1]; *ecx = regs[2]; *edx = regs[3];
}
static unsigned long long xgetbv(unsigned int xcr) {
    return _xgetbv(xcr);
}
#else
#include <cpuid.h>
static void cpuid(int leaf, int subleaf, unsigned int *eax, unsigned int *ebx,
                  unsigned int *ecx, unsigned int *edx) {
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
}
static unsigned long long xgetbv(unsigned int xcr) {
    unsigned int lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(xcr));
    return ((unsigned long long)hi << 32) | lo;
}
#endif

/* ── Feature detection ────────────────────────────────────────── */

typedef struct {
    int has_avx2;
    int has_fma;
    int has_avx512f;
    int has_avx512dq;
    int os_avx512;      /* OS enabled ZMM state save */
    char model[64];     /* CPU model string */
} CpuInfo;

static CpuInfo detect_cpu(void) {
    CpuInfo info;
    memset(&info, 0, sizeof(info));

    unsigned int eax, ebx, ecx, edx;

    /* Get max CPUID leaf */
    cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    unsigned int max_leaf = eax;

    /* Leaf 1: AVX, FMA, OSXSAVE */
    if (max_leaf >= 1) {
        cpuid(1, 0, &eax, &ebx, &ecx, &edx);
        int has_avx    = (ecx >> 28) & 1;
        int has_osxsave = (ecx >> 27) & 1;
        info.has_fma   = (ecx >> 12) & 1;

        /* Check OS enabled YMM state save (XCR0 bits 1-2) */
        int os_avx = 0;
        if (has_avx && has_osxsave) {
            unsigned long long xcr0 = xgetbv(0);
            os_avx = ((xcr0 & 0x6) == 0x6);

            /* Check OS enabled ZMM state save (XCR0 bits 5-7) */
            info.os_avx512 = ((xcr0 & 0xE0) == 0xE0);
        }

        if (!os_avx) {
            info.has_fma = 0; /* Can't use FMA without AVX OS support */
        }
    }

    /* Leaf 7: AVX2, AVX-512F, AVX-512DQ */
    if (max_leaf >= 7) {
        cpuid(7, 0, &eax, &ebx, &ecx, &edx);
        info.has_avx2    = (ebx >> 5) & 1;
        info.has_avx512f = (ebx >> 16) & 1;
        info.has_avx512dq = (ebx >> 17) & 1;

        /* AVX-512 requires OS support */
        if (!info.os_avx512) {
            info.has_avx512f = 0;
            info.has_avx512dq = 0;
        }
    }

    /* CPU model string: leaves 0x80000002-0x80000004 */
    cpuid(0x80000000, 0, &eax, &ebx, &ecx, &edx);
    if (eax >= 0x80000004) {
        unsigned int *p = (unsigned int *)info.model;
        cpuid(0x80000002, 0, p+0, p+1, p+2, p+3);
        cpuid(0x80000003, 0, p+4, p+5, p+6, p+7);
        cpuid(0x80000004, 0, p+8, p+9, p+10, p+11);
        info.model[48] = '\0';
        /* Trim leading spaces */
        char *s = info.model;
        while (*s == ' ') s++;
        if (s != info.model) memmove(info.model, s, strlen(s) + 1);
        /* Trim trailing spaces */
        int len = strlen(info.model);
        while (len > 0 && info.model[len-1] == ' ') info.model[--len] = '\0';
    } else {
        strcpy(info.model, "Unknown CPU");
    }

    return info;
}

/* ── Singleton: detect once ───────────────────────────────────── */

static int g_detected = 0;
static CpuInfo g_info;

static void ensure_detected(void) {
    if (!g_detected) {
        g_info = detect_cpu();
        g_detected = 1;
    }
}

/* ── Public API ───────────────────────────────────────────────── */

typedef void (*ICMFunc)(int, const double *, int, const QP *, double *);

ICMFunc icm_best_backend(void) {
    ensure_detected();
    /* Check if AVX-512 backend is linked (weak symbol) AND cpu supports it */
    if (g_info.has_avx512f && g_info.has_avx512dq && icm_avx512 != NULL)
        return icm_avx512;
    return icm_avx2;
}

const char *icm_backend_name(void) {
    ensure_detected();
    if (g_info.has_avx512f && g_info.has_avx512dq && icm_avx512 != NULL)
        return "avx512";
    return "avx2";
}

const char *icm_cpu_model(void) {
    ensure_detected();
    return g_info.model;
}

void icm_print_cpu_info(FILE *f) {
    ensure_detected();
    fprintf(f, "CPU: %s\n", g_info.model);
    fprintf(f, "Features: AVX2=%s FMA=%s AVX-512F=%s AVX-512DQ=%s (OS ZMM=%s)\n",
            g_info.has_avx2 ? "yes" : "no",
            g_info.has_fma ? "yes" : "no",
            g_info.has_avx512f ? "yes" : "no",
            g_info.has_avx512dq ? "yes" : "no",
            g_info.os_avx512 ? "yes" : "no");
    fprintf(f, "Backend: %s", icm_backend_name());
    if (g_info.has_avx512f && g_info.has_avx512dq && icm_avx512 == NULL)
        fprintf(f, " (AVX-512 detected but backend not linked)");
    fprintf(f, "\n");
}
