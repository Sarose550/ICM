# The ICM Cost Model, Explained in Plain English

## 1. What problem is this solving?

This library computes tournament poker equities: numbers like "what is my
expected share of the prize pool, given everyone's chip stacks?" There are
several different ways to compute the same answer. Think of them as different
"engines" or "methods." Some engines are faster for small problems, some are
faster for large problems, and it is not obvious from simple rules which one
will be fastest for a given problem size.

The program has to automatically pick, for every single query the user makes,
which engine to use and how to configure it. The user never specifies this
themselves. This whole system of automatic picking is what we call **the cost
model.**

"Always use the theoretically best engine" fails because on real hardware, crude
math does not predict real speed. The CLAUDE.md project file documents a concrete
example: every individual number in an older formula was verified correct against
real execution, but when those individually-correct numbers were combined into one
big go/no-go prediction, the prediction was wrong on both M3 Pro and Zen4
hardware. The problem is that a formula has to correctly account for every
microarchitectural effect (cache sizes, memory bandwidth, instruction pipeline
behavior) and getting all of that right in one equation is extraordinarily hard.

The solution this project adopted: wherever possible, **replace formula-based
predictions with direct real measurements.** Time the actual candidate choices
on the real hardware, store those measurements in tables, and consult those
tables at runtime instead of computing a guess from an equation.

---

## 2. The three decisions, explained one at a time, in order

Every time the program runs a computation, it makes three separate decisions.
Here they are, in the order they happen.

### Decision (a): Which engine to use: linear, hybrid, or tree

**What these engines do, in one sentence each:**

- The **linear engine** does the math directly, one multiplication at a time.
  It uses no special tricks. It is simple, but its cost grows in direct
  proportion to both the number of players (n) and the number of top-finishing
  positions to compute (k), written as "O(n × k)." For small k, this is the
  fastest option because there is no setup overhead to pay.

- The **hybrid engine** breaks the problem into chunks called blocks, processes
  each block separately, and then combines results using a mathematical shortcut:
  the Fast Fourier Transform, or FFT. An FFT is a clever way to multiply many
  numbers together at once, much faster than doing it pair by pair, but it has
  a fixed startup cost. Using an FFT is like renting a bulldozer: it takes time
  to start the engine and drive it to the site, but once it is there it moves
  dirt much faster than a shovel. The FFT only pays off above a certain problem
  size. The hybrid engine tries to get the best of both worlds: use the simple
  method for small sub-problems, use the FFT for large ones.

- The **tree engine** is a simpler variant that uses FFTs everywhere without
  the block-chunking of the hybrid engine. It is sometimes slower than hybrid
  on a single processor core, but easier to run in parallel on many cores
  simultaneously.

**How this decision is made (CPU side):**

This decision uses a **lookup table combined with interpolation.** A dedicated
tool (`tools/calibrate_crossover.c`) runs both engines at many different problem
sizes, directly times them on the real hardware, and records the exact problem
size where hybrid becomes faster than linear: the "crossover point." These
crossover points are stored in a small table in each device's configuration file
(`fft_config.h`). At runtime, when the program needs to decide for a problem
size that falls between two measured crossover points, it draws a smooth curve
through the measured points and reads off the in-between answer. This technique
of connecting two known points with a curve on log-scaled axes is called
**log-linear interpolation.**

**Why this approach (and not a formula?):**

The decision is between two discrete options (linear or hybrid), but the
threshold where the answer flips from one to the other is a continuous quantity:
it could be any value of k. A lookup table with interpolation makes sense
here: measure a modest number of crossover points across the range, interpolate
between them, and you get accurate predictions at every problem size without
having to measure every single one.

**How this decision is made (GPU side):**

⚠ **The GPU still uses the older formula-based approach**, not the empirical
crossover table that the CPU side was upgraded to. `gpu_select_engine_est()`
in `src/gpu/gpu_plan.cu` (line 828) computes a linear-engine cost estimate
from raw FMA-operation counts and compares it to the hybrid-engine estimate.
In practice, the GPU code almost always picks hybrid anyway (the code overrides
"linear" to "hybrid" at line 858), so this gap has limited practical impact at
the moment. But it means the GPU engine-selection logic was never given the
same measurement-based fix that the CPU received.

---

### Decision (b): Once hybrid is chosen, how big a "block" to use

**What is a block?**

A block is a chunk of the problem that gets processed together as a unit before
combining its results with other chunks. The hybrid engine splits the full list
of players into groups (blocks), multiplies the numbers within each block using
the simple direct method, and then uses FFTs to combine results across blocks.

The block size can only be one of a small set of specific values: 8, 16, 24,
32, 48, or 64. There is no such thing as "block size 40." This is a **discrete
choice**: you pick one of the available options, not a number on a continuous
slider.

**How this decision is made:**

This decision uses a **pure lookup table, with no interpolation.** A dedicated
tool (`tools/calibrate_best_b.c`) runs the hybrid engine at every candidate
block size across a grid of (n, k) problem sizes, times each one directly, and
records which block size was fastest. The resulting table maps a pair of numbers
(n, k) to a block size B. At runtime, when the program needs to decide for a
particular (n, k), it finds the closest (n, k) that was actually measured and
uses the B from that entry, a technique called **nearest-neighbor lookup.**

**Why this approach (and not interpolation?):**

Interpolation between B=32 and B=64 would produce imaginary block sizes like
"B=48.7." There is no such thing. B is a discrete choice among a small set of
options. The right method for a discrete choice is to look up the answer from
the closest point where someone actually tested all the options and found which
one won. Interpolation only makes sense when the answer is a continuous number
that can be any value along a range (like the crossover threshold in decision
(a)).

---

### Decision (c): Within each step of the hybrid engine, which computation method to use

At every level of the computation tree inside the hybrid engine, the program
faces two choices for each multiplication:

1. Use the simple direct method (plain multiplication, one pair at a time), or
2. Use an FFT (the fast-multiplication trick).

If it uses an FFT, it must also pick exactly what FFT size to use from a large
menu of calibrated sizes. The FFT size must be at least as large as the
computation requires, but it could be larger. Using a larger FFT is more
expensive per FFT, but a smaller FFT that is "too small" forces the program to
do extra cleanup work afterward; this cleanup is called **wrap correction**
(explained below).

**What is wrap correction?**

When the program uses an FFT that is slightly smaller than the full computation
needs, the FFT produces a result with a predictable error: some of the answer
"wraps around" and lands in the wrong place, like a car's odometer rolling over
from 99,999 back to 0. The program can fix this error afterward by computing the
wrapped-around portion separately and adding it back. This extra fixing step is
called **wrap correction.** It costs real computation time. The cost model has
to weigh whether it is cheaper to use a small FFT plus wrap correction, or a
larger FFT with no wrap correction at all.

**How this decision is made:**

This decision uses **direct comparison of calibrated real measurements.** A
calibration tool (`tools/calibrate.c` on CPU, `tools/calibrate_gpu.cu` on GPU)
runs every FFT size on the real hardware, records the exact wall-clock time
each one takes, and stores those timings in a table (749 entries on CPU, over
3,000 on GPU). At runtime, for each computation step, the program scans the
calibrated sizes that could work, adds the estimated wrap-correction cost to
each candidate's measured FFT time, and picks the combination with the lowest
total cost. This is **not interpolation and not a formula**: it is searching
through real measurements for the cheapest valid option.

The wrap-correction cost itself is estimated by a formula (number of cleanup
multiplications × measured cost per multiplication), but the FFT time at each
size comes directly from a real measurement.

**Why this approach?**

This decision involves comparing costs across many concrete, measurable options
(dozens of calibrated FFT sizes, plus the schoolbook option). Every option's
cost can be measured directly and stored in a table. There is no interpolation
between FFT sizes because every size that is actually available has its own
real measurement. This is the one layer of the cost model that was already
measurement-based from the start; it was never the problem.

---

## 3. Where this session found things going wrong

### The specific bug found and fixed

At one exact problem configuration (n=4,194,304 players, k=128 top positions),
the GPU cost model made a bad choice. It selected an FFT size of 128 with a
wrap correction of 127 (meaning: the FFT was 127 units too small, requiring
nearly as much cleanup work as the original computation). Meanwhile, the very
next column in the results table (same n, but k=256) ran substantially faster
(612ms vs 890ms) even though it was computing roughly twice as much work.

**Why the bad choice happened:**

The cost model picks an FFT size by comparing all calibrated sizes and choosing
the one with the lowest estimated total cost (FFT time + wrap-correction time).
At this particular problem size, the cuFFT cost model estimated that fft_n=128
with heavy wrap correction was cheaper than fft_n=256 with zero wrap correction.

Later in the decision chain, a separate check exists specifically to catch this
kind of mistake: "Is a clean power-of-2 fused-kernel size actually cheaper than
what we chose?" This check is at line 698 and line 959 of
`src/gpu/gpu_plan.cu`. But this check had a gate on it: it only ran when the
selected execution method was NOT already the fused kernel
(`tier != GPU_TIER_FUSED`).

The problem: at small FFT sizes, the fused kernel is almost always chosen
(because it is reliably faster than the alternative cuFFT library at small
sizes). So by the time the program reached the "is a clean size cheaper?"
check, the tier was already FUSED, and the check was skipped. The program
never asked the question "is fused at 128 with heavy wrap correction actually
cheaper than fused at 256 with zero wrap correction?", exactly the comparison
that would have caught the mistake.

**The fix** (implemented and hardware-verified on a rented B200, 2026-07-26):
the gate now also fires when the tier is already FUSED but the chosen
configuration has a nonzero wrap penalty. When it fires from an already-FUSED
state, it correctly compares against fused-kernel costs instead of cuFFT costs,
so the comparison is apples-to-apples. `bench_gpu_fused verify` passed 36/0
after the fix. The previously-bad cell at n=4,194,304,k=128 dropped from
~890ms to 511.7ms; n=8,388,608,k=128 from 1349.8ms to 1025.8ms;
n=1,048,576,k=128 from 122.6ms to 93.7ms; n=524,288,k=128 from 54.17ms to
42.2ms (now correctly below its k=256 neighbour, resolving the inversion).

### The scale of impact

A Python simulation (kept out of tree, in `scratch/`) replicated
the exact decision logic and swept 189 grid points of (n, k) values. It found
that **12 of 189 points (6.3%)** were affected, every one sharing the identical
signature: a computation that fit in a 128-point FFT with heavy wrap correction
but would have been better off with a 256-point FFT with zero wrap correction.

The same analysis confirmed the fix resolves 5 of the 11
previously-cataloged non-monotonicities in the GPU heatmap data (cells where
a larger k ran faster than a smaller k at the same n, a counter-intuitive
result that should not happen with a correct cost model). Hardware
verification confirmed all five inversions resolved.

### Two deeper gaps found during the investigation

**Gap 1: Wrap-correction cost has never been directly measured on the GPU.**

On the CPU side, there is a dedicated tool (`tools/bench_wrap_fma.c`) that
directly measures how long one wrap-correction multiplication takes, by
running the wrap-correction loop in isolation across a wide range of sizes
and fitting a line to the results. This direct measurement feeds into the
CPU cost model as the `WRAP_FMA_NS` constant.

On the GPU side, no equivalent tool exists. The GPU cost model estimates
wrap-correction cost using a generic FMA-operation rate (`GPU_SCHOOL_FMA_NS`,
the speed of one multiply-and-add operation when the GPU is fully saturated
with work). But wrap correction is not a fully-saturating workload: it involves
a separate kernel launch, operates on irregularly-sized data, and does not
benefit from the same throughput optimizations as a large dense matrix
multiplication. The formula-based estimate has not been validated against
direct measurement.

**Gap 2: The batch-adjustment system for FFT costs is inactive on the current GPU.**

Running many small jobs together at once is often cheaper per-job than running
them one at a time; this is called **batching.** The GPU cost model has a
well-designed system for estimating how the per-call cost of an FFT drops as
more jobs are batched together. It uses a concept of a "floor cost": the
asymptotic minimum cost per FFT call when the GPU is fully loaded with work,
measured directly for each FFT size.

This system is gated on a compile-time flag called `GPU_HAS_CUFFT_FLOOR`. The
flag is not defined in the current B200 device configuration
(`devices/b200/gpu_fft_config.h`). When the flag is absent, the batch-aware
function `estimate_cufft_pipeline_ns_batched()` falls back to ignoring the
batch size entirely and returning the same cost regardless of batching. The
real measurements this system needs (the floor costs at each FFT size) were
never collected for the B200 GPU.

**Gap 3: The GPU's engine-choice decision still uses the old formula approach.**

As noted in section 2(a): the CPU side replaced its formula-based engine
selection with an empirical crossover table in commit `44bc959`, closing a
confirmed 37-45% dispatch-accuracy gap for subset queries. The GPU side was
never given the same treatment. Its `gpu_select_engine_est()` function still
computes a cost comparison from raw FMA-operation counts. In current practice
the GPU always picks hybrid anyway (the code overrides "linear" to "hybrid"
at line 858 of `gpu_plan.cu`), so this gap does not cause wrong answers today,
but it means the mechanism was never brought up to parity with the CPU side.

**Gap 4: Fitted numbers from `tools/fit_gpu_cost_model.py` are never used.**

The tool `tools/fit_gpu_cost_model.py` takes a set of real GPU timing
measurements and fits four numerical parameters (named `C_wrap`, `C_school`,
`R`, and `C_gap` in the Python source). It prints these four numbers. But a
direct check this session confirmed: these four numbers do not appear anywhere
in the actual GPU source code under `src/gpu/`. They are not in the device
configuration header. Nothing reads them back in. The tool appears to be
leftover work from an earlier design that was replaced by the current
lookup-table approach, but the documentation was never updated to say so.

---

## 4. Why this design pattern (lookup tables replacing formulas) at all?

This design borrows from a well-established pattern in the numerical software
world. Several major software libraries faced the exact same problem: picking
the fastest algorithm or configuration for a given problem size on a given
machine, without the user having to figure it out manually.

The core insight is that **directly timing real candidate choices and storing
the results in a table produces more reliable automatic decisions than
computing a decision from a mathematical formula.** A formula requires
correctly modeling every relevant hardware effect (cache sizes at every level,
memory bandwidth under different access patterns, instruction pipeline
behavior, branch prediction, and so on). A real measurement automatically
captures all of that, whether or not anyone understood it in advance.

**Specific precedents cited in the project's CLAUDE.md:**

- **FFTW** (the "Fastest Fourier Transform in the West," a widely-used library
  for computing FFTs): FFTW has two planning modes. Its "ESTIMATE" mode uses a
  formula to guess which FFT plan will be fastest. Its "MEASURE" and "PATIENT"
  modes actually run the candidate plans on the real hardware and time them.
  The FFTW documentation itself admits that ESTIMATE is frequently wrong
  compared to the measurement-based modes. This project's calibration tool
  (`tools/calibrate.c`) uses FFTW's PATIENT mode for exactly this reason.

- **ATLAS** (the "Automatically Tuned Linear Algebra Software," a system that
  automatically generates optimized linear algebra code for whatever machine
  it is installed on): ATLAS searches through candidate implementations at
  install time, timing each one on the real hardware, rather than assuming a
  formula can predict which version will be fastest. The paradigm is called
  "AEOS": Automated Empirical Optimization of Software.

- **BeBOP/Sparsity** (a research project on optimizing sparse matrix
  operations, from Demmel, Dongarra, Whaley, and others, 2004): This project
  searched through register-blocking parameters (choices about how to group
  operations to fit into the CPU's fastest memory) by timing real candidates,
  directly analogous to this project's block-size selection.

- **LAPACK's ILAENV** (a standard linear algebra library used in scientific
  computing): LAPACK includes a function called `ILAENV` that answers
  configuration questions like "at what problem size should I switch from the
  simple algorithm to the blocked algorithm?" The answer (parameter NX,
  ISPEC=3) is measured empirically on each machine and consulted as a cheap
  threshold comparison at runtime, with no live racing of both candidates in
  production, exactly the pattern this project uses for its own
  linear-vs-hybrid crossover.

---

## 5. Open methodological gaps, prioritized

1. **Collect the GPU FFT floor measurements and enable the batch-adjustment
   system.** The floor measurements (`gpu_floor_ns[]`, `gpu_floor_fft_sizes[]`)
   need to be collected from real B200 hardware (the tools `bench_batch.cu` and
   `level_timer.cu` are referenced in the source comments as the measurement
   tools). Once collected, define `GPU_HAS_CUFFT_FLOOR` in the B200 device
   config header to activate the batch-aware cost estimates. Right now, batch
   size is silently ignored.

2. **Build a GPU-specific wrap-correction microbenchmark.** Mirror what
   `tools/bench_wrap_fma.c` does for the CPU: measure wrap-correction cost
   directly on the GPU by running the wrap-correction kernel in isolation
   across a wide range of sizes. Feed the resulting measurement into the cost
   model instead of the generic `GPU_SCHOOL_FMA_NS` rate.

3. **Replace the GPU engine-selection formula with an empirical crossover
   table.** The CPU side already did this (commit `44bc959`) and closed a
   37-45% dispatch-accuracy gap for subset queries. The GPU side still uses a
   formula-based comparison in `gpu_select_engine_est()`. In current practice
   the GPU always picks hybrid anyway, so this has no practical effect today,
   but the mechanism should be brought to parity with the CPU side to prevent
   future problems if the linear engine ever becomes a real option on GPU.

4. **Clean up the fit_gpu_cost_model.py situation.** The tool fits four
   parameters that are never used by any code. Either: delete the tool if it
   is genuinely obsolete, or wire its outputs into the GPU cost model if the
   fit approach is still intended to be used. Document the decision in
   CLAUDE.md so future readers are not confused.

---

*Document compiled from: CLAUDE.md, HANDOFF.md (2026-07-26 session record),
src/fft_cost_model.h, src/cost_model.h, src/gpu/gpu_plan.cu,
tools/fit_gpu_cost_model.py, and the B200 device configuration header.*
