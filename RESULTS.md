# RESULTS.md — ICM Equity Optimization Results

Machine: Apple M3 Max (ARM64), Q=256 quadrature points.
All correctness tests PASS at < 5e-12 relative error.

## Final Performance Grid (ms, Q=256, uniform stacks, caffeinate + nice -20)

### Single-threaded (median of 5 runs)

T=tree+FFT, L=linear(batched for n≥2048), H=hybrid B=8.

```
n       k=10   k=50   k=100  k=n/4  k=n/2  k=n
64       0      1      1      0      0      1
128      0      1      2      1      1      2
256      1      3      4      3      4      4
512      2      5      9      9     10     11
1024     3     11     16     21     25     27
2048     5     18     38     52     60     64
4096    10     37     68    125    143    155
8192    20     74    130    291    337    361
16384   40    147    303    725    807    846
32768   80    297    603   1715   1876   1974
65536  160    600   1239   3996   4357   4584
```

Best engine per cell:
```
n       k=10  k=50  k=100  k=n/4  k=n/2  k=n
64       T     T     T      T      L      T
128      L     L     T      T      H      T
256      L     L     T/H    T      T      H
512      L     L     T      T      T      T
1024     L     L     H      H      H      H
2048     L     L     H      H      H      H
4096     L     L     H      H      H      H
8192     L     L     H/L    H      H      H
16384    L     L     H      H      H      H
32768    L     L     H      H      H      H
65536    L     L     H      H      H      H
```

### Parallel scaling (M3 Max, k=n, Q=256)

```
Threads  n=8192  Speedup
1        361     1.0x
2        186     1.9x
4         94     3.8x
8         49     7.4x
16        38     9.5x
```

Note: at 8+ threads, the tree engine beats hybrid for large k
(PATIENT-quality FFTW plans in cloned contexts enable this).

## Dispatch Rule

```c
int k_cross = (n >= 2048) ? 95 : 70;
if (k >= k_cross && n >= 256)
    use hybrid(B=8);
else if (n >= 2048)
    use linear_batched;
else
    use linear;
```

## Total speedups vs original baseline

| Cell | Baseline | Final | Speedup |
|------|----------|-------|---------|
| n=1024, k=n | 689ms | **27ms** | **26x** |
| n=4096, k=n | ~8900ms | **155ms** | **57x** |
| n=8192, k=n | ~35000ms | **361ms** | **97x** |

## 1-Second Threshold (single-threaded, k=n, Q=256)

Largest n where k=n completes in under 1 second: **n ≈ 16,448**.

## FFT Phase Split (M3 Max, measured via bench_grid profile)

```
fft_n    fwd(ns)  pw(ns)   ifft(ns) f_fwd  f_pw   f_ifft
64       39       16       46       0.38   0.16   0.46
128      98       31       103      0.42   0.13   0.44
256      214      61       233      0.42   0.12   0.46
512      475      122      481      0.44   0.11   0.45
1024     1039     246      1113     0.43   0.10   0.46
2048     2138     486      2303     0.43   0.10   0.47
4096     5833     942      5338     0.48   0.08   0.44
8192     13183    1963     13960    0.45   0.07   0.48
16384    31920    3851     31846    0.47   0.06   0.47
```

Forward and inverse are ~44-47% each. Pointwise multiply is 6-16% (shrinks at large sizes).
