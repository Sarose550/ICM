[![CI](https://github.com/Sarose550/ICM/actions/workflows/ci.yml/badge.svg)](https://github.com/Sarose550/ICM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# ICM -- Independent Chip Model Equity Computation

High-performance C library for computing tournament placement equities using generating-function quadrature. Computes exact ICM equities for poker tournaments with up to ~17,216 players / payouts in 1 second*.

## Quick Start

```bash
# Build (requires FFTW3)
make

# Verify correctness
./bench_grid verify

# Full benchmark grid
./bench_grid
```

## API

```c
#include "icm.h"

// Initialize (call once -- loads FFTW wisdom, builds lookup tables)
icm_init("fftw_wisdom.dat");

// Compute equities for all n players
//   S[n]       -- chip stacks
//   Q          -- quadrature points (typically 256)
//   payout[k]  -- payout coefficients
//   equity[n]  -- output (caller-allocated)
double ns = icm_equity(n, S, Q, payout, k, equity);

// Compute equities for a subset of players
double ns = icm_equity_subset(n, S, Q, payout, k, equity, targets, n_targets);
```

Returns wall-clock time in nanoseconds. All correctness tests pass at < 5e-12 relative error.

**Python bindings.** `python/` provides a ctypes wrapper (`icm.equity(stacks, payouts)`)
that calls straight into the same compiled shared library the C API uses.
See [python/README.md](python/README.md) for setup (`make libicm`, then
`import icm`).

## How It Works

**1. The problem.** A tournament has `n` players with chip stacks
`S_1, ..., S_n` and a payout structure `(π_1, ..., π_k)` where `k ≤ n`
positions receive nonzero prizes. ICM computes each player's expected
payout - the sum over finishing positions of the prize for that position
times the probability the player finishes there. The naive answer enumerates
all `n!` elimination orderings, weights each by its probability under the
Malmuth–Harville model (Harville, 1973; Malmuth, 2001), and sums the
resulting payouts.

**2. Why naive enumeration is dead almost immediately.** `n!` grows
catastrophically fast: 15! ≈ 1.3 × 10^12, 20! ≈ 2.4 × 10^18. As the
HoldemResources.net blog post ["High Accuracy ICM Calculations for Large
Fields"](https://www.holdemresources.net/blog/high-accuracy-mtt-icm/)
notes, "Naive implementations of ICM can handle about 15 players, and even
optimized versions can't calculate exact Malmuth-Harville values beyond
25-30 players." The naive enumeration wall is around `n ≈ 15` - any attempt
to enumerate all orderings for 16+ players runs into years of compute time.

**3. The industry-standard exact method: bitmask dynamic programming.**
Rather than enumerating orderings, one can track the set of players who
have busted so far. Let `dp[mask]` be the probability that exactly the
players in the bitmask `mask` have been eliminated. From each state, for
each surviving player `j`, the transition adds `dp[mask] · (S_j / total
remaining stack)` to `dp[mask | (1<<j)]`. There are `2^n` states and up to
`n` candidate transitions per state, giving `O(n · 2^n)` total work - the
per-state cost comes from looping over surviving players, not from the
payout structure. This is the method used by real poker tools; see GTO
Wizard's ["Theoretical Breakthroughs in
ICM"](https://blog.gtowizard.com/theoretical-breakthroughs-in-icm/) post
for a practical discussion and Helmuth Melcher's 2015 TU Wien diploma
thesis ["Evaluation of Equity Models for Tournament
Poker"](https://repositum.tuwien.at/handle/20.500.12708/79991) for the
academic writeup. The practical wall is roughly 25–30 players - at
`n = 30`, `2^30` states already pushes into gigabytes of memory.

**4. How real tools scale past 30 players: Monte Carlo via the "exponential
clock" framing.** The Malmuth–Harville elimination rule ("at each step, the
probability any surviving player busts next is proportional to their
stack") has an equivalent continuous-time formulation. Assign each player
`j` an independent exponential random variable `T_j` with rate equal to
their chip stack `S_j` (an "elimination clock"), and eliminate players in
order of increasing `T_j`. The memoryless property of the exponential
distribution guarantees this recovers exactly the same stack-proportional
elimination rule at every step. This gives a simple, unbiased way to
*sample* a full elimination order in one shot - draw `n` exponentials, sort
- instead of simulating step-by-step. Tysen Streib introduced this
technique in a TwoPlusTwo forum thread, ["New Algorithm: Calculate ICM
Large
Tournaments"](https://forumserver.twoplustwo.com/15/poker-theory-amp-gto/new-algorithm-calculate-icm-large-tournaments-1098489/).
Error shrinks as `O(1/√N)` in the number of sampled tournaments `N`, so
high precision gets expensive. (A refinement uses [Quasi-Monte Carlo
sampling](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method) -
deterministic low-discrepancy point sequences instead of independent random
draws, giving closer to `O(1/N)` convergence for smooth integrands - but
this repo does not build on that approach.)

**5. This repo's approach: make the Monte Carlo estimate exact.** The
exponential-clock model above yields an integral representation of "player
`i` finishes in position `r`." Substituting `v = e^(-t)` turns the
combinatorial sum inside that integral into the coefficient of a generating
function. For player `i`, define:

```
a_j(v) = v^(S_j),    b_j(v) = 1 - v^(S_j)
Q_i(x; v) = Π_{j ≠ i} (a_j(v) + b_j(v) · x)
```

The coefficient of `x^m` in `Q_i(x; v)` captures exactly the combinatorial
term that Monte Carlo would otherwise have to sample - the sum over all
subsets of `m` other players of the product of their elimination
probabilities times the remaining players' survival probabilities. So
instead of drawing `N` random samples of the exponential race and
averaging, this repo evaluates the exact 1-D integral over `v` via
quadrature (after a change of variables `v = Φ(y)` using the standard
normal CDF to make the integrand decay rapidly). With `Q = 256`
Gauss-Legendre nodes, this yields deterministic double-precision accuracy
(relative error < 5 × 10^(-12); see Accuracy section below).

The remaining computational challenge is evaluating, for *every* player `i`
simultaneously, the coefficients of `Q_i(x; v)` - the product of everyone
else's per-player factor. Computed naively, one player at a time, that's `n`
separate degree-`(n-1)` products: `O(n²)` factors to multiply. The
subproduct tree computes all `n` of them together in `O(n log n)`
multiplications (each accelerated further by FFT convolution, which
multiplies two degree-`d` polynomials in `O(d log d)` time instead of the
schoolbook `O(d²)`; see Wikipedia's article on the [Fast Fourier
Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) for why
that's faster). Here's how.

**6. Computing all `n` leave-one-out products at once: the subproduct tree.**
Restated in linear-algebra terms, this is a dot-product problem. Represent a
truncated polynomial by its coefficient vector, and define the pairing
`⟨f, g⟩ = Σ_m f[m]·g[m]` (an ordinary dot product). What every player `i`
actually needs is `⟨payout, Q_i(x; v)⟩` - the payout vector dotted against
the product of everyone else's factor. Built one player at a time, that
requires `n` separate `O(n)`-degree products before the dot product is even
possible: `O(n²)` total.

Here is the shortcut. "Multiply by a fixed polynomial `P`", `T_P(f) = P·f`
(truncated), is a *linear* operator on coefficient vectors. Under the
dot-product pairing above, its adjoint - the operator `T_P*` satisfying
`⟨T_P(f), g⟩ = ⟨f, T_P*(g)⟩` for every `f, g` - is exactly a
*cross-correlation* with `P`: `T_P*(g)[m] = Σ_j P[j]·g[m+j]`. This falls
straight out of writing `T_P` as a matrix: it's a convolution (Toeplitz-style)
matrix, and the transpose of a convolution matrix is a correlation matrix.

Arrange the `n` players' factors `P_j(x) = a_j(v) + b_j(v)·x` as the leaves
of a balanced binary tree. Building the tree bottom-up is just composing a
chain of these `T_P` operators, one per level. `Q_i` - "everyone except leaf
`i`" - is what that chain computes if you skip every `T_P` on leaf `i`'s own
root-to-leaf path. Adjoints reverse the order of composition
(`(A∘B)* = B*∘A*`), so applying the *adjoints* of that same chain, starting
from `payout` at the root and walking downward, computes `⟨payout, Q_i⟩`
directly - one adjoint per level, shared across every leaf, branching only
where paths diverge. Concretely, at any node the "own subtree" the walk is
about to enter is one child; the part it must still account for is the
*other* child, i.e. the sibling. So the adjoint applied at each step down is
`T_{P_sibling}*` - correlation with the sibling's polynomial - which is
exactly the mechanism below:

- *Build (bottom-up).* Each internal node's polynomial is the product of its
  two children's polynomials, truncated to whatever degree bound is actually
  needed downstream (never more than `k`, since the payout vector has only
  `k` nonzero terms and nothing past that degree can ever be read out).
  After this pass, every node holds the product of all the leaf factors in
  its subtree - this is the `T_P` chain being composed.

- *Propagate (top-down).* Seed the root with the payout vector itself,
  `(π_1, ..., π_k)`, treated as the coefficients of a polynomial `g_root(x)`.
  Then walk back down the tree, applying `T_{P_sibling}*` at each level:
  each child's new coefficients are
  `g_child[m] = Σ_j P_sibling[j] · g_parent[m+j]` - the cross-correlation
  derived above, computable via FFT the same way convolution is. Descend
  all the way to the leaves.

`g_leaf_i[0]` is then `⟨payout, Q_i(x; v)⟩`, truncated to its constant term:
exactly the coefficient the generating-function argument in step 5 needs,
for every `i`, without ever having built `Q_i(x; v)` on its own. One build
pass, one propagate pass, `O(n log n)` total work (times `O(log n)` per
FFT-accelerated multiply/correlate), not `n` separate `O(n)` products.

The hybrid engine (below) runs this same two-pass algorithm over *blocks*
of `B` players rather than individual players: a block's leaf polynomial
is the product of its `B` players' factors, multiplied directly
(schoolbook, not FFT - `B` is small). That collapses `n` tree leaves down
to `n/B`, shrinking the tree's depth and per-level FFT count. The cost is
one extra step at the very end: a block's leave-one-out `g`-vector describes
"everything outside this block," not any individual player inside it, so
recovering a single player's coefficient means dividing the block's
*complete* (non-truncated) polynomial product by that player's own factor,
polynomial division, done only on this small, complete, `B`-degree product,
where the resulting numerical amplification is bounded by `|c|^B` and safe
in double precision for `B` up to 64. (Division elsewhere in this codebase
is deliberately avoided, see the correctness constraint in `CLAUDE.md`,
because doing the same thing on the full, *truncated* `n`-player product is
numerically unstable.)

**Three CPU engines with cost-based dispatch.** The library picks the
fastest engine per `(n, k)` pair via `select_engine()`, which compares a
roofline linear-cost estimate against a calibrated hybrid-cost model - no
hand-tuned crossover thresholds:

1. **Linear (batched):** `O(nk)` forward-backward pass. Interleaves BQ=8
   quadrature points for SIMD (NEON on Apple Silicon, AVX-512 on Zen 4).
   Best for small `k`.
2. **Hybrid (block + tree):** Partitions `n` players into cost-model-selected
   blocks, builds block products sequentially, then runs an FFT-accelerated
   binary subproduct tree over the blocks. Best for large `k`.
3. **Tree (pure FFT):** FFT-accelerated subproduct tree without blocking.
   Slightly slower than hybrid in serial but wins in parallel.

**GPU path (cuFFTDx fused-kernel).** On NVIDIA B200/H200, `src/gpu/`
implements a planner, execution engine, and API. The planner assigns each
tree level to one of three tiers - schoolbook (small degrees), cuFFTDx
fused kernels (medium), or batched cuFFT (large) - and executes via CUDA
graph capture for near-zero launch overhead. See the Performance tables
below for current throughput numbers.

## Accuracy

The library is validated against exact closed-form reference values for two
special payout structures, not against a slow general-purpose reference
(which would cap validation at ~20–30 players). These closed forms are exact
for *any* `n` because they follow from linearity of expectation over pairs
and triples of players, not from enumerating elimination orderings:

Both derivations reuse the exponential-clock construction from step 4 above:
`T_1, ..., T_n` independent, `T_j ~ Exponential(rate = S_j)`, and finishing
order = increasing `T_j` (smallest `T` finishes 1st). One elementary fact
about competing independent exponentials does all the work: for any subset
of players, the probability that a particular one of them has the smallest
`T` - i.e. finishes best - *within that subset* is that player's stack
divided by the subset's total stack. (Proof: for player `i` against a
group, `T_i` and the minimum of everyone else's `T`s are independent, the
latter is itself exponential with rate equal to the sum of their stacks -
the minimum of independent exponentials is exponential with the summed
rate - and `P(T_i < T_other)` for two independent exponentials with rates
`a, b` is `a / (a + b)`.) Applied to a pair `{i, j}`: `P(i beats j) =
S_i / (S_i + S_j)`. Applied to a triple `{i, j, k}`: `P(i beats both) =
S_i / (S_i + S_j + S_k)`.

Now write player `i`'s actual finishing position as `m` other players
finishing ahead of them (`m = 0` is 1st place). For any `t`, the number of
ways to choose `t` of the players who finish *behind* `i` is `C(n-1-m, t)`
- an exact combinatorial identity on the realized outcome, no probability
involved yet: it's just choosing `t` players from the `n-1-m` who rank
below `i`. Equivalently, it's a sum of indicators over every `t`-subset `T`
of the other `n-1` players, counting the ones `i` beats entirely:

```
C(n-1-m, t) = Σ_{T ⊆ others, |T| = t}  1[i finishes better than every player in T]
```

- **V1 (linear payout, `payout[m] = n - m`):** Since `n - m = C(n-1-m,0) +
  C(n-1-m,1)`, apply the identity at `t = 0` (always 1, trivially - the
  empty subset) and `t = 1` (one term per opponent `j`):

  ```
  E[payout_i] = E[1] + E[ Σ_{j≠i} 1[i beats j] ]
              = 1 + Σ_{j≠i} P(i beats j)          <- linearity of expectation
              = 1 + Σ_{j≠i} S_i / (S_i + S_j)
  ```

  This is exactly `v1_exact()`'s formula, `O(n²)` to compute directly.

- **V2 (quadratic payout, `payout[m] = C(n-1-m, 2)`):** Apply the identity
  at `t = 2` - one term per opponent *pair* `{j, k}`:

  ```
  E[payout_i] = E[ Σ_{j<k, j,k≠i} 1[i beats j and k] ]
              = Σ_{j<k, j,k≠i} P(i beats both j and k)   <- linearity of expectation
              = Σ_{j<k, j,k≠i} S_i / (S_i + S_j + S_k)
  ```

  since "`i` beats both `j` and `k`" is exactly "`i` has the smallest `T`
  among the trio," which is the competing-exponentials fact above applied
  to `{i, j, k}`. This is exactly `v2_exact()`'s formula, `O(n³)` to
  compute directly.

In both cases the move from a *combinatorial identity on one realized
outcome* to an *exact formula for the expectation* is linearity of
expectation, applied term-by-term to a sum of indicator variables: it
costs nothing to push the expectation through a sum, no matter how the
individual indicator events are correlated with each other. Higher payout
schedules follow the same pattern for larger `t`; V1 and V2 are the `t ≤ 2`
cases used here as exact, closed-form, arbitrary-`n` ground truth.

These are implemented as `v1_exact()` and `v2_exact()` in `src/icm.c`
(publicly exposed as `icm_v1_exact()` / `icm_v2_exact()` in `icm.h`).
The tool `tools/accuracy_bench.c` sweeps the quadrature node count `Q`
and reports convergence against both closed forms across four stack
distributions: uniform (all stacks equal), adversarial (100:1 ratio),
geometric, and an extreme 1e9:1 adversarial case.

**Headline result:** Gauss-Legendre quadrature converges to ~5 × 10^(-13)
relative error by `Q = 1024` against both V1 and V2 closed forms across all
tested distributions. The convergence is rapid - here are representative
rows from `results/accuracy_m3max_20260718.csv` for the `gauss` scheme on
uniform stacks (V1 payout):

| Q | max_rel_err (n=4, uniform, V1) |
|---|-------------------------------|
| 4 | 4.10 × 10^0 |
| 8 | 4.36 × 10^(-1) |
| 16 | 1.32 × 10^(-1) |
| 64 | 3.08 × 10^(-8) |
| 128 | 6.79 × 10^(-13) |
| 256 | 8.87 × 10^(-13) |
| 1024 | 1.07 × 10^(-12) |

At `Q = 1024`, the maximum relative error across *all* tested configurations
(`n` up to 20, all four stack distributions, both V1 and V2) stays below
~2 × 10^(-12) for uniform stacks and below ~6 × 10^(-13) for the adversarial
and 1e9:1 cases. The production default is `Q = 256`, which already delivers
sub-2 × 10^(-12) relative error - sufficient for any practical poker
application.

![Accuracy convergence](accuracy_convergence.png)

## Performance

Three engines with cost-based automatic dispatch:

| Engine | Strategy | Best for |
|--------|----------|----------|
| **Linear** (batched) | O(nk), BQ=8 quad-point batching, interleaved layout, L2-aware checkpointing | Small k |
| **Hybrid** (B=auto) | Block build + FFT tree + bidirectional divide, calibrated block size | Large k |
| **Tree** (pure FFT) | FFT-accelerated subproduct tree | Parallel workloads |

`select_engine(n, k)` chooses the optimal engine for each (n, k) pair based on calibrated FFT costs and hardware parameters. No manual tuning required.

### Single-threaded (ms, Q=256)

| n | k=10 | k=100 | k=n/2 | k=n | | k=10 | k=100 | k=n/2 | k=n |
|---|------|-------|-------|-----|-|------|-------|-------|-----|
| | **M3 Pro (recalibrating)** |||| | **Zen 4 7950X** ||||
| 1024 | - | - | - | - | | 1.28 | 7.61 | 29.1 | 34.0 |
| 4096 | - | - | - | - | | 7.32 | 28.3 | 161 | 168 |
| 8192 | - | - | - | - | | 14.5 | 53.4 | 376 | 382 |
| 16384 | - | - | - | - | | 29.6 | 112 | 866 | 835 |
| 65536 | - | - | - | - | | 117 | 419 | 4170 | 4490 |

*M3 Pro numbers are being recalibrated after a hardware migration - this table will be refreshed once that run completes.*

### 16-thread parallel (ms, Q=256)

| n | k=10 | k=100 | k=n/2 | k=n | | k=10 | k=100 | k=n/2 | k=n |
|---|------|-------|-------|-----|-|------|-------|-------|-----|
| | **M3 Pro (recalibrating)** |||| | **Zen 4 7950X** ||||
| 1024 | - | - | - | - | | 0.125 | 0.562 | 2.23 | 2.48 |
| 4096 | - | - | - | - | | 0.604 | 2.35 | 11.4 | 11.9 |
| 8192 | - | - | - | - | | 1.18 | 4.86 | 26.6 | 26.9 |
| 16384 | - | - | - | - | | 2.39 | 10.4 | 67.5 | 81.2 |
| 65536 | - | - | - | - | | 19.5 | 45.2 | 530 | 631 |

*M3 Pro numbers are being recalibrated after a hardware migration - this table will be refreshed once that run completes.*

See [RESULTS.md](RESULTS.md) for complete performance tables across platforms.

## Building

### macOS (Apple Silicon)

```bash
# Serial
make

# Parallel (requires: brew install libomp)
make parallel
```

Uses Accelerate framework (vDSP) for FFT dispatch at supported sizes.

### Linux

```bash
# Install FFTW3
sudo apt-get install libfftw3-dev    # Debian/Ubuntu
sudo dnf install fftw-devel          # Fedora/RHEL

# Serial
make

# Parallel
make parallel
```

Automatically detects MKL (via dlopen) for dual-dispatch FFT when available.

### Linux with AOCL-FFTW (AMD)

```bash
# Install AOCL-FFTW to /usr/local/aocl-fftw
make DEVICE=zen4
make DEVICE=zen4 parallel
```

Auto-detected if installed at `/usr/local/aocl-fftw`.

### GPU (NVIDIA)

```bash
make bench_gpu_fused CUDA_ARCH=sm_100    # B200/B100
make bench_gpu_fused CUDA_ARCH=sm_90     # H100/H200
```

Requires CUDA toolkit and cuFFTDx.

**B200 performance** (Q=256, fused cuFFTDx kernels):

| n | k=n | Time |
|---|-----|------|
| 65,536 | 65,536 | 24.75 ms |
| 262,144 | 262,144 | 117.90 ms |
| 1,441,792 | 1,441,792 | 866 ms |
| 1,572,864 | 1,572,864 | 1,148 ms |

See `devices/b200/gpu_fft_config.h` for calibration data.

## Calibrating for a New Device

If your hardware matches an already-calibrated device (`devices/m3_pro`, `devices/zen4`), you don't need to run `./calibrate` at all - build straight against the shipped wisdom and config:

```bash
make DEVICE=m3_pro   # or zen4 - whichever matches your machine
./bench_grid verify
./bench_grid crossover   # confirm dispatch decisions match measured winners on YOUR unit
```

`fftw_wisdom.dat` and the `calib_times_ns[]` table are measured on one specific physical machine. FFTW will happily load wisdom from a different unit of the same CPU model - it just isn't guaranteed to have picked the fastest codelet for *your* silicon, and the nanosecond timings the cost model reads for FFT-vs-schoolbook and engine-dispatch decisions won't necessarily match your machine's actual behavior (different DIMM speed, microcode revision, thermal/boost profile, or memory bandwidth can all shift these numbers). `./bench_grid crossover` is the check that catches this: if every cell's dispatch decision agrees with the measured winner, the shipped calibration is good enough and you're done. Only recalibrate from scratch (below) if it disagrees - and definitely recalibrate if you're on hardware unlike anything already in `devices/`.

One command runs the whole pipeline (FFTW calibration, hybrid-engine timing,
and cost-model constant fitting) and finishes with a `verify` + `crossover`
check:

```bash
./tools/calibrate_full.sh mydevice   # add --quick for a faster, less precise FFTW pass
```

If you want to see (or run) each step by hand
instead:

```bash
# Generate calibration data
gcc -O3 -march=native -o calibrate tools/calibrate.c -lfftw3 -lm
./calibrate

# Copy to device directory
mkdir -p devices/mydevice
cp fft_config.h fftw_wisdom.dat devices/mydevice/

# Build and verify
make DEVICE=mydevice
./bench_grid verify
./bench_grid profile    # measure FMA_NS, FFT_OVERHEAD_NS, etc.
```

Update the `#define` constants in `fft_config.h` with measured values from `./bench_grid profile`. See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) for details on each constant.

## Python Bindings

Python bindings are in `python/`. Build the shared library first:

```bash
make libicm.a
```

## Project Structure

```
src/icm.h                    -- public CPU API
src/icm.c                    -- all CPU engines + FFT infrastructure
src/linear_batched_impl.inc  -- batched linear engine template
src/icm_gpu.h                -- GPU API header
src/gpu/                     -- GPU implementation (split modules)
  gpu_internal.h             -- shared GPU types and helpers
  gpu_kernels.cu             -- CUDA kernels
  gpu_plan.cu                -- GPU planner and cost model
  gpu_exec.cu                -- GPU execution engine
  gpu_api.cu                 -- GPU public API
bench/bench.c                -- CPU benchmark + verification harness
bench/bench_gpu.cu           -- GPU benchmark + verification harness
tools/calibrate.c            -- FFTW calibration tool
tools/calibrate_gpu.cu       -- GPU FFT calibration tool
devices/                     -- per-device calibration data
python/                      -- Python ctypes bindings
```

## Documentation

- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) -- detailed optimization notes, porting guide, and algorithm descriptions
- [RESULTS.md](RESULTS.md) -- complete performance tables, head-to-head comparisons, and phase-split analysis

## License

MIT. See [LICENSE](LICENSE).

---
\* Single-threaded, AMD Ryzen 9 7950X.
