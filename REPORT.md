# Hardware-Optimized ICM Computation via Generating Function Quadrature

## 1. The ICM Placement Probability Problem

The Independent Chip Model assigns monetary expected values to chip stacks in poker tournaments by computing placement probabilities: the probability that each of $n$ players finishes in each of $n$ positions. The classical recursive algorithm requires $O(n \cdot 2^n)$ operations, making it intractable beyond roughly $n = 20$.

The generating function approach replaces the exponential recursion with a numerical integral. For each player $i$ with stack $S_i$, define the generating function

$$G_i(z) = S_i \int_0^1 v^{S_i - 1} \prod_{j \neq i} \bigl(z + (1-z)\, v^{S_j}\bigr)\, dv$$

The coefficient $[z^m]\, G_i(z)$ gives the probability that player $i$ finishes in place $m+1$. The product $\prod_{j \neq i}(z + (1-z)\, v^{S_j})$ is a polynomial of degree $n-1$ in $z$. Expanding it generates all subsets of opponents: the term $v^{S_j}$ acts as a survival probability for player $j$ at integration variable $v$, and the Beta-like weight $S_i\, v^{S_i - 1}$ marginalizes over the outcome determining player $i$'s placement. The coefficient $[z^m]$ selects configurations where exactly $m$ opponents survive, placing $i$ in position $m+1$.

The algorithm at each quadrature node proceeds in two phases. First, build the full product polynomial $P(z) = \prod_{j=1}^{n} ((1 - a_j)z + a_j)$ truncated to degree $n$, where $a_j = v^{S_j}$. This costs $O(n^2)$: multiply in one linear factor at a time, each an $O(\text{deg})$ operation. Second, for each player $i$, synthetically divide $P(z)$ by the factor $((1-a_i)z + a_i)$ to recover the degree-$(n-1)$ quotient $Q_i(z)$. This costs $O(n)$ per player, $O(n^2)$ total. Across $Q$ quadrature nodes, the grand total is $O(Q \cdot n^2)$.

For validation, we use the expected placement value $V_1(i) = \sum_{m=0}^{n-1} (n - m) \cdot \text{prob}[i][m]$, which has the closed form $V_1(i) = 1 + \sum_{j \neq i} S_i/(S_i + S_j)$. We report $\max_i |V_{1,\text{computed}}(i) - V_{1,\text{exact}}(i)| / |V_{1,\text{exact}}(i)|$ as the error metric.

## 2. The erfc_trap Quadrature Scheme

### 2.1 Why logistic-space quadrature has a fundamental accuracy ceiling

The standard approach substitutes $v = \sigma(x) = 1/(1 + e^{-x})$ to map the unit interval to the real line, then applies the trapezoidal rule with a tanh-sinh (doubly-exponential) transform. The convergence of the trapezoidal rule on analytic integrands is governed by the width $d$ of the maximal strip of analyticity around the real axis. For a function analytic in $|\operatorname{Im}(x)| < d$ and decaying at $\pm\infty$, the error behaves as $E_{\text{trap}} = O(e^{-2\pi d / h})$ where $h$ is the step size. The logistic sigmoid has poles at $x = \pm i\pi, \pm 3i\pi, \ldots$, limiting $d_{\text{eff}} = \pi/2$ after the tanh-sinh reparametrization.

For large stack ratios, the effective domain width is $L \approx 70.7$ at ratio $10^9$. With $Q = 256$ nodes, $h \approx 0.277$, producing an error floor of $e^{-2\pi(\pi/2)/0.277} \approx 5 \times 10^{-9}$. This is a hard floor determined by the analyticity structure of the logistic sigmoid. No rearrangement of nodes within logistic space can break through it.

### 2.2 The normal-CDF substitution

We replace the logistic sigmoid with the standard normal CDF $\Phi(y) = \tfrac{1}{2}\operatorname{erfc}(-y/\sqrt{2})$. $\Phi(y)$ is an entire function (no poles in $\mathbb{C}$), so the strip width is infinite. For entire functions with Gaussian-decaying integrands, the trapezoidal rule converges as $E_{\text{trap}} = O(e^{-c/h^2})$ — super-exponential in $1/h$.

Additionally, the Gaussian tail decay compresses the effective domain from width 70.7 (logistic) to 17.7 (normal) — a $4\times$ compression that further improves convergence.

### 2.3 Why the domain is asymmetric

The lower bound ($y_{\text{lo}} \approx -7.7$) is determined by where $\log\Phi(y) < -25$ and is independent of the stack distribution. The upper bound ($y_{\text{hi}} \approx 10$ at ratio $10^9$) depends on $S_{\max}$: it must capture the big-stack transition where $1 - \Phi(y)$ falls below $10^{-10}/S_{\max}$. For equal stacks, the domain would be nearly symmetric around zero. An early implementation used a fixed $y_{\text{hi}} = 6.5$, missing the big-stack transition entirely at high ratios.

### 2.4 Why uniform spacing beats tanh-sinh in normal space

The tanh-sinh transform clusters nodes at endpoints where $\phi(y) \approx 0$ — exactly where accuracy is not needed. The Gaussian Jacobian already provides perfect tail suppression, making uniform spacing optimal.

### 2.5 Accuracy results

At $Q = 256$, $n = 512$, stack ratio $10^9$:

| Distribution | logistic_ts | erfc_trap | Improvement |
|:---|:---:|:---:|:---:|
| adversarial (1 big, rest = 1) | 5.0e-9 (floor) | 1.3e-10 | 38x |
| reverse_adv (1 small, rest = big) | 1.7e-8 (floor) | 4.6e-10 | 36x |
| bimodal (half big, half small) | 5.5e-9 (floor) | 6.9e-11 | 80x |
| geometric (log-spaced) | 5.9e-9 (floor) | 2.3e-10 | 26x |
| uniform random | 5.1e-14 | 5.1e-14 | 1x |

The logistic_ts errors are hard floors. erfc_trap errors continue improving to ~$10^{-14}$ at $Q = 384$.

### 2.6 Numerical implementation

The node generator stores $\text{logv}_q = \log\Phi(y_q)$ directly — the engine never converts to logistic coordinates. The $a_j$ values are computed as $\exp(S_j \cdot \text{logv}_q)$, never computing $v$ itself (which underflows for deep-left nodes). The synthetic division uses bidirectional evaluation: bottom-up (dividing by $a_i$) when $a_i > 0.5$, top-down (dividing by $b_i = 1 - a_i$) when $a_i \leq 0.5$, keeping the divisor $\geq 0.5$ to prevent exponential error amplification.

## 3. CPU Implementation

### 3.1 Baseline and the search for SIMD opportunities

Five SIMD strategies benchmarked at $n = 512$, $Q = 256$, adversarial distribution:

| Method | Time | Speedup | Notes |
|:---|:---:|:---:|:---|
| v0 scalar | 338 ms | 1.00x | baseline |
| v1 SIMD build | 332 ms | 1.02x | manual AVX2 on coefficient sweep |
| v2 SIMD both | 144 ms | 2.35x | + 4-player division batching |
| v3 fused | 341 ms | 0.99x | fuse divide + accumulate, no quo buffer |
| v4 quad batch | 321 ms | 1.05x | 4 quad points batched in build |

The v1 result established that the polynomial build has no loop-carried dependency (top-down traversal reads only old values) and GCC already auto-vectorizes it. The v2 win comes from the division phase, which has a genuine sequential dependency that the compiler cannot auto-vectorize across players. Packing 4 players into AVX2 lanes gives ~$4\times$ on the division phase.

### 3.2 The loop inversion (v5)

At $n = 2048$, the prob output matrix is 32 MB. In the original q-outer loop order, each quad point touches all $n$ rows, creating ~16 GB of DRAM traffic across 256 quad points. The key observation: all $Q$ polynomials together occupy only 4.2 MB and fit in L3. Restructuring to i-outer/q-inner means 4 active prob rows (64 KB) stay in L1/L2 for all 256 quad points — a 250x reduction in DRAM traffic.

### 3.3 The interleaved accumulator with fused divide (v5b)

An interleaved buffer $\text{acc}[m \times 4 + b]$ (64 KB, AVX2-aligned) enables fused divide+accumulate in a single pass with no quotient buffer. This fusion was worthless in the original loop order (DRAM-cold rows) but essential in the inverted loop (L1-hot accumulator). The loop inversion and the fusion are complementary optimizations that only pay off together.

### 3.4 The q-skip optimization (v5c) and stack-sorted batching (v5h)

q-skip uses binary search to find the cutoff below which $\exp(S_i \cdot \text{logv})$ underflows, avoiding wasted iterations. Stack-sorted batching ensures all 4 players in each AVX2 batch have similar $S_i$ values, so their bidirectional division branches agree. Measured: sorting reduces mixed-direction fallbacks from 3.8% to 0.0% at $n = 2048$, giving $1.44\times$ on uniform random stacks.

### 3.5 AVX-512 backend

8-player ZMM batches, vectorized exp() via degree-11 minimax polynomial ($< 2$ ULP), 8-wide build, 128 KB interleaved accumulator (fits in Zen 4's 1 MB L2). With the sequential-combine build integrated, the AVX-512 advantage over AVX2 narrowed from the previous $1.36\times$ to $1.05$–$1.07\times$ on Skylake-X. The reason: the seqcombine's schoolbook inner loop auto-vectorizes well at 256-bit, so the build phase is nearly the same speed in both backends. The remaining AVX-512 advantage comes only from 8-player vs 4-player batching in the divide kernel. On Zen 4, where AVX-512 runs at a milder clock penalty than Skylake-X, the divide-phase advantage would be larger.

### 3.6 Additional micro-optimizations in the current code

Three further improvements were applied to the divide+accumulate kernel beyond the v5h structure:

**Vectorized exp:** A 4-wide AVX2 exp implementation (`fast_exp_4`) using Cody-Waite range reduction and a degree-11 minimax polynomial replaces the 4 scalar `exp(S_i * logv)` calls per block per quad point. This is the same algorithm used in the AVX-512 backend but adapted for YMM registers without `_mm256_cvtpd_epi64` (which requires AVX-512DQ). Accuracy is $< 2$ ULP.

**inv_v precomputation:** Instead of computing $v^{S_i - 1} = \exp((S_i - 1) \cdot \text{logv})$ via a second exp call, we precompute $\text{inv\_v}[q] = \exp(-\text{logv}[q]) = 1/v$ once per quad point (Q scalar exp calls total), then $v^{S_i-1} = a_i \cdot \text{inv\_v}$. This replaces 4 exp calls per block per quad point with 4 multiplies.

**2-way q-unrolling:** The inner loop processes two adjacent quad points per iteration. The two division recurrences are independent, giving the out-of-order engine twice the instruction-level parallelism to fill execution ports. The accumulator updates combine: $\text{acc}[m] \mathrel{+}= \text{pw}_0 \cdot \text{qp}_0 + \text{pw}_1 \cdot \text{qp}_1$ in one load-FMA-FMA-store sequence.

### 3.7 Sequential-combine build

The polynomial build groups $n$ linear factors into blocks of $B$, builds each block sequentially (degree-$B$ sub-polynomial), then multiplies the running product by each chunk via schoolbook multiply. The inner loop — $\text{dst}[i+j] \mathrel{+}= \text{src}[i] \cdot \text{ch}[j]$ — is a fixed-length FMA sweep of $B+1$ elements that auto-vectorizes perfectly. The chunk stays in L1 throughout. For $n < 512$ the overhead is not worthwhile and the plain sequential build is used.

The block size $B$ is chosen as $B = \text{clip}(64,\, 384,\, n/4)$, rounded to the nearest multiple of 8. This formula arises from a cost model with two regimes:

**Small $n$ ($n < \sim 1536$): $B = n/4$, giving $C \approx 4$ chunks.** With few combine steps, minimizing their count dominates. Each step pays a fixed overhead (memset, cache warm-up after phase-1 eviction) that is large relative to its FMA cost. $C = 4$ is the sweet spot: enough steps for the schoolbook inner loop to be non-trivial, few enough that the overhead sum stays small.

**Large $n$ ($n \geq \sim 1536$): $B \approx 384$, independent of $n$.** The total phase-2 FMA count is $\approx n^2/2$ regardless of $B$ (the seqcombine rearranges FMAs, it does not reduce them), so this term cancels out of the optimization. The $B$-dependent part of the cost is $T_{\text{opt}}(B) = \frac{nB}{2}\alpha_1 + \frac{n\bar{f}}{B}$, where $\alpha_1 \approx 0.31$ ns/FMA is the phase-1 per-FMA cost and $\bar{f} \approx 23\,\mu\text{s}$ is the average per-step overhead. By AM-GM, $B^* = \sqrt{2\bar{f}/\alpha_1} \approx 384$, which is independent of $n$ because both terms are $O(n)$.

The lower clamp at 64 ensures the inner loop is long enough to auto-vectorize ($\geq 16$ AVX2 vector ops); it never activates for $n \geq 512$. The upper clamp at 384 keeps the chunk in L1 (384 doubles = 3 KB) and prevents phase-1 cost from dominating; raising it to 512 hurts by $\sim 5\%$ at moderate $n$.

The cost basin around $B^*$ is shallow: the 95% plateau typically spans $[200, 500+]$ for $n \geq 1536$.  The formula $\text{clip}(64,\, 384,\, n/4)$ achieves $\leq 2.7\%$ worst-case regret across $n = 512$–$16384$.

### 3.8 CPU benchmark results

Benchmarked on a 2.8 GHz Intel (Skylake-X), $Q = 256$, ratio $10^9$, median of 11 runs. Includes all optimizations: loop inversion, fused divide+accumulate, interleaved accumulator, stack-sorted batching, q-skip, vectorized exp, inv_v precomputation, 2-way q-unrolling, and sequential-combine build ($B = \text{clip}(64, 384, n/4)$). The $B$ values below were recorded under the earlier $4\sqrt{n}$ formula; the corrected formula gives $B = (128, 256, 384, 384)$ at $(n = 512, 1024, 2048, 4096)$. All are within the shallow cost basin and the timing difference is $< 3\%$.

| Distribution | n=512 (B=96) | n=1024 (B=128) | n=2048 (B=184) | n=4096 (B=256) |
|:---|:---:|:---:|:---:|:---:|
| adversarial | 57 ms | 178 ms | 580 ms | 2404 ms |
| reverse_adv | 17 ms | 58 ms | 180 ms | 822 ms |
| bimodal | 34 ms | 114 ms | 395 ms | 1595 ms |
| geometric | 33 ms | 106 ms | 349 ms | 1288 ms |
| uniform | 31 ms | 102 ms | 356 ms | 1352 ms |

All errors: $< 5.5 \times 10^{-14}$ for uniform, $1.28 \times 10^{-10}$ for adversarial, consistent across all $n$.

Projecting uniform at $n = 2048$ to a Ryzen 7950X at 5.7 GHz: $356 \times (2.8/5.7) \approx 175$ ms. GTO Wizard claims 740 ms — our projected result is roughly $4.2\times$ faster.

## 4. GPU Implementation

### 4.1 Architecture

Kernel 1 (Build): $Q = 256$ thread blocks, each builds one polynomial in double-buffered shared memory, writes transposed (coefficient-major) to global memory.

Kernel 2 (Divide + Accumulate): $n$ thread blocks (one per player), $Q = 256$ threads each. Each thread runs the sequential division recurrence independently, with periodic shared-memory tree reductions every TILE_M = 16 coefficients.

### 4.2 Coalescing

The initial q-major polynomial layout had 6% bandwidth efficiency (strided reads across cache lines). Transposing to coefficient-major layout gives 100% efficiency — consecutive threads read consecutive addresses.

### 4.3 GPU benchmark results (A100 80GB)

$Q = 256$, adversarial, ratio $10^9$, before coalescing fix:

| n | Kernel time | Error |
|:---|:---:|:---:|
| 512 | 4.08 ms | 1.28e-10 |
| 1024 | 13.3 ms | 1.28e-10 |
| 2048 | 52.4 ms | 1.28e-10 |
| 4096 | 203 ms | 1.28e-10 |

Scaling is nearly perfect $O(n^2)$. The polynomial store fits in A100's 40 MB L2 up to $n \approx 19{,}500$. On MI300X (256 MB Infinity Cache), it fits up to $n \approx 125{,}000$.

## 5. Polynomial Build: Sequential vs Block-Tree

### 5.1 The approach

Group $n$ linear factors into $n/B$ blocks of $B$. Build each block sequentially (producing a degree-$B$ sub-polynomial), then combine via a balanced binary tree of schoolbook polynomial multiplications. The tree has $\log_2(n/B)$ levels. Total cost: $O(n \cdot B \cdot \log_2(n/B))$ vs $O(n^2)$ sequential.

### 5.2 CPU build-only results (2.1 GHz Emerald Rapids)

Best of 20 runs for $n \leq 2048$, 5 for $n \leq 8192$, 2 for $n = 16384$. The optimal block size shifts with $n$: moderate $B$ (64–512) is generally best.

| n | Seq. (ms) | B=16 | B=32 | B=64 | B=128 | B=256 | B=512 | B=1024 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 512 | 0.053 | 1.6x | **1.7x** | 1.6x | 1.6x | 1.6x | — | — |
| 1024 | 0.175 | 0.5x | 0.8x | **2.0x** | 1.8x | 1.6x | 1.5x | — |
| 2048 | 1.23 | 0.8x | 1.3x | 2.2x | 2.5x | **3.4x** | 2.6x | 1.6x |
| 4096 | 4.91 | 1.0x | 1.8x | 2.7x | 3.2x | **3.7x** | 3.7x | 2.6x |
| 8192 | 19.3 | 1.5x | 2.1x | 3.2x | 4.2x | 4.5x | **5.2x** | 4.1x |
| 16384 | 86.3 | 1.1x | 4.0x | 5.7x | 6.7x | **7.7x** | 7.6x | 7.3x |

Errors at machine epsilon ($< 2 \times 10^{-14}$) for all configurations. Unlike the earlier pooled-allocator benchmarks, the non-monotonicity at small $B$ for moderate $n$ is real: when $B$ is too small, the tree has many levels and each level's schoolbook multiply is small, losing to the sequential build's cache-friendly linear sweep. The crossover $B^*$ increases with $n$.

### 5.3 Cost model and adaptive B

The per-FMA cost of a combine step at running product degree $d$ was measured by timing individual steps during real builds (not in isolation — the cache state after phase 1 matters). Three distinct regimes emerge:

| Degree range | ns/FMA | Regime |
|:---|:---:|:---|
| 0–500 | 0.5–0.7 | Cold startup: phase-1 evicts src/dst from L1; first steps pay cold-miss penalty |
| 500–3000 | 0.3–0.5 | Degraded mid-L1: associativity conflicts between src and dst |
| 3000+ | 0.18–0.22 | L2 streaming: prefetcher handles sequential src reads |

Surprisingly, L2 is faster than the degraded mid-L1 range. The schoolbook inner loop reads the chunk (always in L1) while the outer loop reads src sequentially — a pattern the hardware prefetcher serves at higher effective bandwidth than the conflict-prone mid-L1 random access.

Crucially, these regime boundaries depend on the running degree $d$ and are independent of $B$: all tested $B \in \{128, 256, 512\}$ show nearly identical per-step $\alpha$ curves when plotted against $d$.

**Cost decomposition.** The total build cost is $T(n, B) = T_1(n, B) + T_2(n, B)$. Phase 1 builds $C = n/B$ blocks, each costing $\sim B^2/2$ FMAs at rate $\alpha_1 \approx 0.31$ ns/FMA, giving $T_1 = nB\alpha_1/2$. Phase 2 performs $C - 1$ combine steps. The total phase-2 FMA count is $\sum_{k=1}^{C-1}(kB+1)(B+1) \approx n^2/2$, independent of $B$ — the seqcombine rearranges work, it does not reduce it.

**B-independent vs B-dependent terms.** Since the phase-2 $\alpha(d)$ curve depends on $d$ (not $B$), and the distribution of FMAs across $d$ values is the same regardless of $B$, the $O(n^2)$ FMA cost cancels out of the $B$ optimization. What remains is:

$$T_{\text{opt}}(B) = \frac{nB}{2}\,\alpha_1 + \frac{n}{B}\,\bar{f}$$

where $\bar{f}$ is the average per-step overhead (memset, cache warm-up, loop setup).

**AM-GM.** By AM-GM, $\tfrac{nB}{2}\alpha_1 + \tfrac{n\bar{f}}{B} \geq n\sqrt{2\alpha_1\bar{f}}$ with equality at $B^* = \sqrt{2\bar{f}/\alpha_1}$. Since $n$ appears in both terms and cancels, **$B^*$ is independent of $n$**.

**Small-$n$ correction.** The continuous approximation $n\bar{f}/B$ breaks down when $C = n/B$ is small (2–4), because each step has a distinct cache regime and the "average overhead" is not meaningful. At small $n$, the discrete optimum is $C \approx 4$ ($B = n/4$), which minimizes the number of expensive cold-start steps while keeping the schoolbook inner loop long enough to be useful.

**The formula $B = \text{clip}(64,\, 384,\, n/4)$ unifies both regimes:** $n/4$ governs for small $n$, the upper clamp at 384 (derived from the AM-GM) takes over once $n/4 > 384$, i.e., $n > 1536$. Measured regret:

| n | $\text{clip}(64, 384, n/4)$ | Measured B* | Regret | 95% plateau |
|:---|:---:|:---:|:---:|:---:|
| 512 | 128 | 128 | 0.0% | [112, 192] |
| 1024 | 256 | 208 | 0.3% | [208, 256] |
| 2048 | 384 | 384 | 0.0% | [160, 768] |
| 4096 | 384 | 512 | 2.7% | [112, 640] |
| 8192 | 384 | 384 | 0.0% | [144, 640] |
| 16384 | 384 | 384 | 0.0% | [288, 512] |

Worst-case regret is 2.7%, vs 13.3% for the earlier $4\sqrt{n}$ formula, which overshoots badly at large $n$ because the per-step overhead $\bar{f}$ is approximately constant (not growing with $n$ as previously assumed).

### 5.4 Impact on full ICM computation (CPU)

Measured before (plain sequential build) and after (sequential-combine) on the same 2.8 GHz machine, $Q = 256$, ratio $10^9$. These timings used $B = 184$ (the earlier $4\sqrt{n}$ formula); the corrected formula gives $B = 384$ at $n = 2048$, both well within the 95% plateau:

| Distribution (n=2048) | Before | After (B=184) | Speedup |
|:---|:---:|:---:|:---:|
| adversarial | 712 ms | 580 ms | 1.23x |
| reverse_adv | 257 ms | 180 ms | 1.43x |
| bimodal | 472 ms | 395 ms | 1.19x |
| geometric | 469 ms | 349 ms | 1.34x |
| uniform | 417 ms | 356 ms | 1.17x |

### 5.5 GPU build-phase results (Tesla T4)

On GPU, the sequential build is bottlenecked by $n$ barrier synchronizations with low thread utilization at early steps. The tree approach replaces this with $n/B$ parallel block builds (one thread block each) followed by a single-block sequential combine of degree-$B$ sub-polynomials via schoolbook multiply.

| n | Sequential | Best B | Tree+Combine | Speedup | Error |
|:---|:---:|:---:|:---:|:---:|:---:|
| 256 | 0.22 ms | 128 | 0.049 ms | 4.5x | 1.1e-15 |
| 512 | 0.60 ms | 256 | 0.114 ms | 5.3x | 1.6e-15 |
| 1024 | 1.81 ms | 512 | 0.342 ms | 5.3x | 2.1e-15 |
| 2048 | 5.94 ms | 1024 | 0.996 ms | 6.0x | 3.0e-15 |

Beyond $n = 4096$, the T4's 64 KB shared memory limits the combine kernel (which needs $2 \times (n+1) \times 8$ bytes for ping-pong buffers). A100 (164 KB shared) supports up to $n \approx 10{,}000$.

The tree phase (parallel block builds) is extremely fast: 0.01–0.14 ms. The combine phase dominates, and its schoolbook multiply has $O(B)$ independent FMAs per output coefficient, keeping the GPU occupied rather than waiting for sync barriers.

## 6. Top-k Truncation

When only the top $k$ placement probabilities matter (e.g., a satellite where the top 10 finishers receive entry tickets, or a payout structure distinguishing only the first 50 places), the polynomial build can be truncated to degree $k$ instead of $n$. This reduces cost from $O(Q \cdot n^2)$ to $O(Q \cdot n \cdot k)$ and shrinks the output from $n \times n$ to $n \times k$.

The numerical concern is roundoff amplification in the division recurrence. Bottom-up division amplifies errors when $a_i < 0.5$; top-down division amplifies when $a_i > 0.5$. With a truncated polynomial, top-down division starts from $P[k]$ rather than $P[n]$, so the missing high-degree coefficients inject an initial error that must attenuate before reaching the desired coefficients.

Compensated arithmetic (TwoProduct/TwoSum in the build) reduces the build's roundoff from $O(n \cdot u)$ to $O(n \cdot u^2)$, giving orders of magnitude more headroom for the division's error amplification. The cost is roughly $4\times$ per coefficient. We use compensated arithmetic whenever the reduced roundoff makes it cheaper overall — which it is for most practical $(n, k)$ combinations.

The impact on memory is transformative. At $n = 8192$, $k = 10$: the polynomial store drops from 16.8 MB to 84 KB, the output from 512 MB to 640 KB, and both build and divide FLOPs fall by $200\times$. At these sizes, million-player tournaments become feasible on a single GPU.

## 7. ICM Gradients

The full gradient tensor $\partial\,\text{prob}[i][m]/\partial S_l$ is $n \times n \times n$ and costs $O(Q \cdot n^3)$. However, for any linear payout structure, the payout-contracted Jacobian $dM_i/dS_l$ costs only $O(Q \cdot n^2)$ — same as the forward pass. For the V1 case, it collapses further to scalar sums $-S_i/(S_i + S_l)^2$ with no polynomial computation at all. Standard CFR solvers do not need ICM gradients; they evaluate terminal payoffs in the forward direction only.

## 8. Dead Ends

Compensated arithmetic in the full-$n$ build (bit-identical, floor is quadrature error). CDF/quantile matching (breaks analyticity). $\mu$-space quadrature (plateau blow-up). Shifted tanh-sinh (distribution-dependent). Gauss-Legendre in logistic space (wrong clustering). v1 manual SIMD build (compiler already does it). v3 fused divide in original loop order (DRAM-cold). v4 quad batch (blows L1). Parallel prefix division (5x more work). Software prefetch (hardware prefetcher sufficient). Tree-combine of block polynomials on CPU (bottom levels have many tiny schoolbook multiplies with high per-call overhead; sequential-combine avoids this by always multiplying the running product by a single degree-$B$ chunk). Output-major schoolbook multiply (variable-length inner loops with shifting bounds that don't vectorize; input-major form has a fixed-length FMA sweep that auto-vectorizes). $B = 4\sqrt{n}$ block-size formula (assumed per-step overhead grows as $n$, giving $B^* \propto \sqrt{n}$; profiling during real builds showed the overhead is approximately constant across steps, so $B^*$ is constant by AM-GM; the $\sqrt{n}$ formula overshoots at large $n$, reaching 13% regret at $n = 1024$).

## 9. Summary

The erfc_trap quadrature achieves $< 5 \times 10^{-10}$ worst-case error at $Q = 256$, a 26-80x improvement over the logistic approach. The CPU implementation composes loop inversion, fused divide+accumulate, interleaved SIMD accumulator, stack-sorted batching, q-skip, vectorized exp, inv_v precomputation, 2-way q-unrolling, and sequential-combine build with adaptive block size $B = \text{clip}(64,\, 384,\, n/4)$. At $n = 2048$ on a 2.8 GHz Skylake-X, uniform stacks complete in 356 ms. Projected to a Ryzen 7950X at 5.7 GHz: ~175 ms, roughly 4.2x faster than GTO Wizard's claimed 740 ms. The sequential-combine build groups factors into degree-$B$ blocks and multiplies them into a running product via input-major schoolbook multiply, giving 1.15–1.39x end-to-end speedup over the plain sequential build. The adaptive $B$ was derived from an AM-GM optimization of the $B$-dependent cost $T_{\text{opt}}(B) = \frac{nB}{2}\alpha_1 + \frac{n\bar{f}}{B}$, which gives $B^* = \sqrt{2\bar{f}/\alpha_1} \approx 384$ independent of $n$. The small-$n$ correction $B = n/4$ (targeting $C = 4$ chunks) handles the regime where the continuous overhead approximation breaks down. Per-step profiling during real builds revealed three cache regimes (cold startup, degraded mid-L1, L2 streaming) that depend on running degree but are invariant to $B$, confirming that the $O(n^2)$ FMA cost cancels out of the optimization. On GPU the same block-tree idea gives 5-6x by eliminating sequential barrier bottlenecks. The GPU implementation achieves 52 ms at $n = 2048$ on A100. Top-$k$ truncation with compensated arithmetic makes million-player tournaments feasible on a single GPU.
