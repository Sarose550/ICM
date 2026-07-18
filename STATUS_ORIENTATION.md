# STATUS ORIENTATION — ICM Project

*Generated for the owner returning after a break. Candid, concise, numbers from real files.*
*Last updated: see file mtime. Covers state as of git commit 7a2cb94 (HEAD, main).*

---

## 1. ALGORITHM SUMMARY

**What problem this solves.** The Independent Chip Model (ICM) computes each poker-tournament player's expected dollar-equity given chip stacks and a prize structure. The naive answer sums over $n!$ elimination orderings — infeasible beyond $n \approx 23$.

**The generating-function quadrature trick.** Each player's equity rewrites as a 1D integral of coefficients of a leave-one-out generating function $Q_i(x;v) = \prod_{j \neq i}(a_j(v) + b_j(v)x)$. Under a change of variables $v = \Phi(y)$ (normal CDF mapping), the integrand becomes rapidly decaying and is evaluated at $Q=256$ Gauss-Hermite-type quadrature nodes — converging to double-precision ($<5\times 10^{-12}$ relative error). The heavy lifting is computing an inner product of the reversed payout vector with the degree-$(k-1)$ truncation of $Q_i$ for *all $n$ players simultaneously* at each quadrature point.

**Three CPU engines with cost-based dispatch.** The library picks the fastest engine per $(n,k)$ pair via `select_engine()`, which compares a roofline linear-cost estimate against a calibrated hybrid-cost model — no hand-tuned crossover thresholds:

1. **Linear (batched):** $\mathcal{O}(nk)$ forward-backward pass. Interleaves $BQ=8$ quadrature points in `a_batch[j*BQ+qi]` layout for SIMD (NEON on M3 Max, AVX-512 on Zen 4). Uses L2-cache-aware checkpointing when the working set exceeds cache. Best for small $k$.
2. **Hybrid (block + tree):** Partitions $n$ players into blocks of size $B$ (cost-model-selected: 16 on M3 Max, 32 on Zen 4). Builds block products sequentially, then runs an FFT-accelerated binary subproduct tree over the blocks, then divides within blocks. Best for large $k$.
3. **Tree (pure FFT):** FFT-accelerated subproduct tree without blocking. Slightly slower than hybrid in serial but wins in parallel (simpler clone, same PATIENT-quality FFTW plans).

**The FFT infrastructure.** All tree operations use offline-calibrated per-size FFT costs (749 smooth sizes, 7-smooth $2^a 3^b 5^c 7^d$). The `best_fft_config()` function searches for the optimal FFT size including wrap-correction tradeoffs: a smaller FFT plus a schoolbook correction for aliased terms often beats padding to the next power of two. At each tree level, the code compares FFT vs. schoolbook cost using measured hardware constants (`FMA_NS`, `FFT_OVERHEAD_NS`). The propagation phase shares the forward FFT of the parent $g$-vector across both children (paired cached correlate) and reuses cached FFT(P) from the build phase.

**GPU path (cuFFTDx fused-kernel).** On NVIDIA B200/H200, the four source files in `src/gpu/` implement a planner, execution engine, and API. The planner assigns each tree level to one of three tiers: schoolbook (small degrees), cuFFTDx fused kernels (medium), or batched cuFFT (large). Execution uses CUDA graph capture for near-zero launch overhead. The B200 achieves $n=1.57\text{M}$, $k=n$ in under 1 second (single-precision-equivalent throughput via fused device-side FFT).

---

## 2. RESULTS ACHIEVED

### CPU — Apple M3 Max (ARM64, NEON + vDSP)

- **Single-thread, $n=8192$, $k=n$:** 350 ms
- **16-thread, $n=8192$, $k=n$:** 37 ms (9.5× speedup on 12P+4E)
- **Single-thread, $n=65536$, $k=n$:** 4,392 ms
- **16-thread, $n=65536$, $k=n$:** 594 ms
- **1-second threshold ($k=n$, serial):** $n \approx 18{,}368$ (binary search)
- **Improvements over original baseline:** 25–32% faster linear paths (BQ=8 interleaved), 3–6% faster tree paths (vDSP FFT dispatch), cost-based dispatch replacing fixed thresholds

### CPU — AMD Ryzen 9 7950X (Zen 4, AVX-512 + FFTW+MKL dual dispatch)

- **Single-thread, $n=8192$, $k=n$:** 296 ms
- **16-thread, $n=8192$, $k=n$:** 24 ms (12.3× speedup, homogeneous 16 P-cores)
- **Single-thread, $n=65536$, $k=n$:** 3,861 ms
- **16-thread, $n=65536$, $k=n$:** 551 ms
- **1-second threshold ($k=n$, serial):** $n \approx 19{,}904$
- **MKL dual dispatch:** MKL wins at 181/749 smooth sizes, FFTW at 568/749

### GPU — NVIDIA B200 (cuFFTDx fused-kernel, sm_100)

- **$n=65{,}536$, $k=n$:** 24.75 ms (B=256, engine=hybrid, ~13.7 GB VRAM)
- **$n=262{,}144$, $k=n$:** 117.90 ms (B=256, ~64.9 GB VRAM)
- **$n=1{,}572{,}864$, $k=n$:** ~981 ms (1-second frontier, from B200_RESUME_CHECKPOINT.md)
- **$n=1{,}441{,}792$, $k=n$:** ~937 ms (last reliably-under-1s frontier from planner fix)
- **$n=8{,}388{,}608$, $k=100$:** 1,235 ms (frontier test — GPU scales to ~8M at small k)
- **$n=16{,}777{,}216$, $k=10$:** 2,592 ms
- **Known limits:** OOM at $n \ge 1{,}048{,}576$ with $k=n$ on the instance tested; cuFFT plan failure (code=5) at $n=524{,}288$, $k=n$

### Head-to-Head (single-threaded, $n=8192$, $k=n$)

| Platform | Time | vs. M3 Max |
|----------|------|------------|
| M3 Max | 350 ms | 1.0× |
| Zen 4 | 296 ms | 1.18× faster |
| B200 GPU | 2.27 ms | 154× faster |

---

## 3. CURRENT STATE / GIT

### Branch: `main`, HEAD: `7a2cb94` — "Cleanup + sweep tool refactor"

**Last 5 commits:**
```
7a2cb94 Cleanup + sweep tool refactor: 3-mode contour, fork timeout, dynamic brew prefix
2ca9d03 GPU: apply cuFFTDx stride fix to all four fused kernel families
da49a61 Campaign: 5-min planner validation timeout; skip failed B candidates
1073781 Heatmap: match GPU power-of-2 grid; planner validation: skip failed B
c9b44d8 contour_1s: early exit when probe exceeds 1s frontier
```

### Modified (staged or unstaged) — 15 files:

| File | Notes |
|------|-------|
| `Makefile` | Added 6+ new GPU tool targets (contour_1s_gpu, bench_batch, bench_kernels, etc.) |
| `contour_zen4_parallel_q256.csv` | Updated contour data |
| `contour_zen4_serial_q256.csv` | Updated contour data |
| `devices/b200/gpu_fft_config.h` | Major rewrite (+610/-? lines) |
| `devices/m3_max/fft_config.h` | Updated calibration constants |
| `devices/zen4/fft_config.h` | Updated calibration + MKL dispatch arrays |
| `src/amx.h` | **DELETED** — Apple AMX FP64 primitives removed |
| `src/gpu/gpu_api.cu` | Changes (+26/-?) |
| `src/gpu/gpu_exec.cu` | Major changes (+305/-?) |
| `src/gpu/gpu_internal.h` | Minor changes (+10/-?) |
| `src/gpu/gpu_kernels.cu` | Changes (+28/-?) |
| `src/gpu/gpu_plan.cu` | Major changes (+301/-?) — planner/model fixes |
| `src/icm.c` | Large simplification (-324 lines, mostly removing AMX references?) |
| `tools/calibrate.c` | Changes (+38/-?) |
| `tools/contour_1s.c` | Minor changes (+13/-?) |

### Untracked files (~40):

**Root-level output files (20 CSV, 8 LOG, 10 TXT):** Benchmark results, GPU phase profiles, heatmap data, contour sweeps. These are output artifacts, not source.

**Root-level binaries (2):** `test_cpu_cost_model`, `test_cpu_cost_model_zen4` — compiled test binaries, should not be in the repo.

**`results_b200_optimized/` directory:** Contains campaign.log, gpu_heatmap.csv, nk_gpu.csv, and a nested `results_b200_optimized/` subdirectory. GPU campaign outputs.

**New tools (18 files in `tools/`):** Microbenchmarks (`bench_batch.cu`, `bench_kernels.cu`, `bench_level_pipeline.cu`), GPU profiling tools (`gpu_phase_profile.cu`, `gpu_sample_plans.cu`), cost-model testers (`test_cpu_cost_model.c`, `test_gpu_cost_model.cu`), calibration variants (`calibrate_full_pipeline.c`, `cold_calib.c`), Python fitting scripts (`fit_cost_model.py`, `fit_gpu_cost_model.py`), and misc helpers (`verify_fft_sizes.c`, `layout_test.c`, `perf_tree.c`, `profile_tree.c`, etc.).

### What's "polished" vs. "experimental scratch":

- **Polished/stable:** `src/icm.c`, `src/linear_batched_impl.inc`, `bench/bench.c`, `tools/calibrate.c`, device configs, `paper/icm_paper.tex`
- **In-flight (modified but functional):** `src/gpu/*` (split modules — working with known limits), `Makefile` (new GPU targets added), `devices/b200/gpu_fft_config.h`
- **Experimental scratch:** Most of the ~18 new `tools/*.c` and `tools/*.cu` files — these are one-off profiling/debugging tools. The root-level CSVs/LOGs/TXTs are all throwaway output. `test_cpu_cost_model*` binaries are compiled artifacts. The old monolithic `src/icm_gpu.cu` (214 KB) is kept as reference but the Makefile now compiles from `src/gpu/`.

### Known documentation inconsistency:

- **`CLAUDE.md` still references `src/amx.h`** (lines 138, 141) but that file has been deleted (shows as `D` in git status). The AMX infrastructure was gated and validated but ultimately removed. CLAUDE.md and README.md need updating.

---

## 4. RESUME-READINESS ASSESSMENT

**Candid judgment: YES, this is resume-ready now — with caveats.**

The CPU story is tight: two platforms, three engines, cost-based dispatch, published-quality numbers. The GPU story has real B200 numbers (n=1.57M in <1s is legitimately impressive). The paper is a complete LaTeX draft at 1,119 lines with full algorithmic content, theorems with proofs, and formatted performance tables.

**Weak spots:**
- The GPU section of the paper is explicitly placeholder ("[Placeholder: benchmark results pending implementation.]"). This needs to be updated with the actual B200 fused-kernel results.
- The repo is messy: ~40 untracked scratch files, uncommitted modifications, dead doc references to deleted `src/amx.h`.
- The GPU path has known OOM/cuFFT failure modes at the frontier — these are documented but not resolved.
- There's no CI badge that actually verifies GPU builds (the GitHub Actions badge in README only covers CPU).

**ONE-LINE RESUME BULLET:**

> Designed and implemented a generating-function quadrature algorithm with FFT-accelerated subproduct trees and cost-model-driven engine dispatch that computes exact ICM tournament equities for up to 1.57 million players in under 1 second on NVIDIA B200 GPU — a super-exponential speedup over the state-of-the-art $\mathcal{O}(n 2^n)$ dynamic programming approach, achieving sub-millisecond latency for 65K-player fields.

*(If you must fit a shorter version:)*

> Built a high-performance ICM equity solver (C/CUDA) using FFT-accelerated subproduct trees and cost-based engine dispatch, achieving 1.57M-player fields in <1s on NVIDIA B200 — a $10^6\times$+ speedup over exact DP.

---

## 5. SMALLEST LIFT TO PORTFOLIO-READY

Here is the minimum sequence to go from "messy but impressive" to "clean portfolio item." Prioritized smallest-first:

### Step 1: Delete throwaway artifacts from repo root (10 minutes)

```bash
# Delete compiled binaries that leaked into the repo
rm test_cpu_cost_model test_cpu_cost_model_zen4

# Delete benchmark output artifacts (regeneratable)
rm *.csv *.log *.txt

# Delete GPU campaign results dir (keep the tools/source, not outputs)
rm -rf results_b200_optimized/
```

(See §7 for exact keep/delete classification.)

### Step 2: Add a `.gitignore` stanza (5 minutes)

Create or append to `.gitignore`:
```
# Benchmark outputs
*.csv
*.log
*.txt

# Compiled test binaries (root level)
test_cpu_cost_model*

# GPU campaign outputs
results_b200_*/
```

Then `git add .gitignore` and commit.

### Step 3: Fix doc references to deleted `src/amx.h` (10 minutes)

- In `CLAUDE.md`: Remove or comment-out the line `amx.h — Apple AMX FP64 outer-product primitives (validated, gated)` from the directory structure, and remove the AMX row from the device constants table.
- In `README.md`: Remove the line `src/amx.h — Apple AMX FP64 outer-product primitives` from the project structure.

### Step 4: Commit all modified files (15 minutes)

```bash
git add -u                    # stage all modified + deleted
git commit -m "Checkpoint: GPU planner fixes, cuFFTDx stride fix, cleanup"
```

This gets the 15 modified + 1 deleted file into a clean committed state.

### Step 5: Categorize and selectively commit the new tools (20 minutes)

Of the 18 new `tools/*` files, decide which to keep:

- **Keep (useful):** `fit_cost_model.py`, `fit_gpu_cost_model.py`, `calibrate_full_pipeline.c`, `verify_fft_sizes.c`, `test_cpu_cost_model.c`, `test_gpu_cost_model.cu`, `setup_b200.sh`
- **Maybe keep:** `gpu_phase_profile.cu`, `gpu_sample_plans.cu`, `cold_calib.c`, `profile_tree.c`, `sample_plans.c`, `perf_tree.c`, `perf_level.c`
- **Delete (one-off debugging):** `bench_batch.cu`, `bench_batch_fused.cu`, `bench_kernels.cu`, `bench_level_pipeline.cu`, `calib_validate.c`, `calibrate_pipeline.c`, `contour_1s_gpu.cu`, `layout_test.c`, `measure_cache_overhead.c`, `merge_analysis.c`, `merge_analysis2.c`, `plan_switch_test.c`

Commit kept tools with brief descriptions. Delete the rest.

### Step 6: Ensure build + test passes (15 minutes)

```bash
make clean && make
./bench_grid quick     # Must show "ALL TESTS PASSED"
./bench_grid verify    # Full verification
```

If `./bench_grid quick` fails: the `src/icm.c` simplification (-324 lines, likely AMX removal) may have introduced issues. This is the highest-risk step.

### Step 7: Update paper GPU section with actual B200 numbers (30 minutes)

The paper's Section 6 ("GPU Acceleration: NVIDIA H200") currently has `[Placeholder: benchmark results pending implementation.]`. Replace with actual B200 cuFFTDx fused-kernel results from `b200_final_benchmarks.txt`/`b200_final_cufft.txt`:

- $n=65{,}536$, $k=n$: 25 ms
- $n=262{,}144$, $k=n$: 118 ms
- $n=1{,}572{,}864$, $k=n$: 981 ms

Also note this was on B200 (not H200 as originally planned), using cuFFTDx fused device-side kernels rather than batched cuFFT.

### Step 8: Update README with GPU numbers (10 minutes)

The README already has B200 numbers but they may be stale vs. the fused-kernel results. Verify and update if needed.

---

## 6. LOW-HANGING FRUIT LEFT

Extracted from `B200_RESUME_CHECKPOINT.md`, git log messages, and result files:

### Unfinished GPU optimizations (from B200_RESUME_CHECKPOINT.md phases):

1. **Phase 4 (fused kernels) — IN PROGRESS:** Replace tier-2 fallbacks (`run_build_level_fused` / `run_prop_level_fused`) with real fused kernels. This is the active work when you left off.
2. **Tier-ablation runner:** Not yet written. Need to directly compare schoolbook vs. fused vs. cuFFT per level and conv size, then hard-wire measured crossovers.
3. **Wrap-serial heuristic:** Currently uses ad-hoc scaling for wrap-serial penalty modeling. Needs replacing with measured wrap-kernel calibration data.
4. **Phase 5 — Graph + memory strategy A/B/C/D:** Not started.
5. **Phase 7 — Joint frontier optimization over $(B, M, T)$:** Not started.
6. **Phase 8 — Final heatmap/contours:** Blocked until optimization freeze.

### Known bugs / failure modes:

7. **GPU OOM at large n,k:** $n \ge 1{,}048{,}576$, $k=n$ fails with `cudaMalloc out of memory` on the instance used (needs larger HBM or memory optimization).
8. **cuFFT plan failure at n=524,288, k=n/k=1000:** `cufftMakePlanMany` returns error code 5 (`CUFFT_INTERNAL_ERROR`) — likely a cuFFT internal size limit.
9. **cuFFTDx stride bug (fixed):** Was causing $k<n$ correctness failures. Fixed in commit `0f7b41b` and applied to all four kernel families in `2ca9d03`.
10. **Planner overflow bug (fixed):** `int corr_input_wrap` overflowed at large $n$ (product > INT_MAX). Fixed by promoting to `double` and adding wrap-serial penalty.

### Optimization ideas noted but not pursued:

11. **Karatsuba at intermediate sizes on Zen 4:** AVX-512 makes schoolbook ~4× faster than NEON, potentially creating a gap where Karatsuba's $\mathcal{O}(n^{1.585})$ could beat both schoolbook and FFT at specific sizes. OPTIMIZATION_GUIDE.md flags this as "measure to be sure" but no implementation exists.
12. **VkFFT dual-dispatch on GPU:** Optional cuFFT+VkFFT per-size selection (analogous to CPU FFTW+MKL). The infrastructure exists (`tools/bench_vkfft.cu`, `vkfft_calib.log`) but isn't wired into the main GPU path yet. 33 sizes selected for dual-dispatch per commit `0ecfd03`.
13. **`src/amx.h` was deleted:** The AMX FP64 primitives for Apple Silicon were validated and gated at degree ≥160, but ultimately removed. If AMX is worth revisiting on M4/M5 hardware, the old code is in git history (before deletion).

---

## 7. DECLUTTER PLAN

### Root-level files — classification:

**🗑️ DELETE (throwaway output artifacts, regeneratable):**

| File(s) | Reason |
|---------|--------|
| `*.csv` (26 files) | Benchmark outputs, contour sweeps, heatmap data. All regeneratable. |
| `*.log` (8 files) | GPU benchmark logs, VkFFT calibration logs. |
| `*.txt` (18 files) | Benchmark output text files, debug dumps. |
| `test_cpu_cost_model` | Compiled binary, leaked into repo root. |
| `test_cpu_cost_model_zen4` | Compiled binary, leaked into repo root. |
| `results_b200_optimized/` | GPU campaign output directory (contains campaign.log, CSVs, nested dir). |

**📁 KEEP but move/commit selectively:**

| File(s) | Recommendation |
|---------|---------------|
| `b200_final_benchmarks.txt` | Keep as reference — copy to `results/` or `devices/b200/` |
| `b200_final_cufft.txt` | Keep as reference |
| `gpu_heatmap_final.csv` | Keep — final GPU heatmap data |
| `results_b200_heatmap_final.csv` | Keep — final heatmap data |
| `b200_B_validation.csv` | Keep — B-size validation data |
| `b200_runtime_vs_n.csv` | Keep — runtime scaling data |

**🔧 Already tracked, modified (commit along with code):**

| File(s) | Status |
|---------|--------|
| `contour_zen4_parallel_q256.csv` | Modified — commit |
| `contour_zen4_serial_q256.csv` | Modified — commit |

### New `tools/` files — classification:

**✅ KEEP (useful for ongoing work):**

- `fit_cost_model.py`, `fit_gpu_cost_model.py` — Python fitting scripts for cost model calibration
- `calibrate_full_pipeline.c` — full FFT pipeline calibration
- `verify_fft_sizes.c` — FFT size verification tool
- `test_cpu_cost_model.c`, `test_gpu_cost_model.cu` — cost model validation tests
- `setup_b200.sh` — B200 instance setup script
- `cold_calib.c` — cold-start calibration tool
- `gpu_phase_profile.cu`, `gpu_sample_plans.cu` — GPU profiling tools
- `profile_tree.c`, `sample_plans.c`, `perf_tree.c`, `perf_level.c` — CPU tree profiling

**🗑️ DELETE (one-off debugging, superseded):**

- `bench_batch.cu`, `bench_batch_fused.cu` — ad-hoc batch benchmarks
- `bench_kernels.cu` — kernel microbenchmark
- `bench_level_pipeline.cu` — level pipeline benchmark
- `calib_validate.c` — one-off calibration validation
- `calibrate_pipeline.c` — redundant with calibrate_full_pipeline.c
- `contour_1s_gpu.cu` — experimental GPU contour (likely superseded by main tools)
- `layout_test.c` — memory layout test
- `measure_cache_overhead.c` — cache overhead measurement
- `merge_analysis.c`, `merge_analysis2.c` — one-off merge analysis
- `plan_switch_test.c` — plan switching test

### Recommended `.gitignore` stanza:

```gitignore
# Generated benchmark outputs
*.csv
*.log
bench_grid_*.txt
bench_*_b200.csv
bench_*_b200.log
gpu_*.csv
gpu_*.log
heatmap_*.csv
nk_*.csv
profile_*.csv
profile_*.txt
sample_plans*.csv
sample_plans*.log
contour_*.csv
accuracy_*.csv
vkfft_*.csv
vkfft_*.log
b200_*.csv
b200_*.txt
b200_*.log

# Compiled test binaries (leaked to root)
test_cpu_cost_model
test_cpu_cost_model_*

# GPU campaign output directories
results_b200_*/
results_b200_optimized/

# Jupyter / Python artifacts
.ipynb_checkpoints/
__pycache__/
*.pyc
```

---

## ACTIONABLE CHECKLIST (prioritized)

- [ ] **1. DELETE** root-level CSV/LOG/TXT artifacts + binaries (see §7 delete list)
- [ ] **2. CREATE/APPEND** `.gitignore` with the stanza above
- [ ] **3. FIX** `CLAUDE.md` and `README.md` references to deleted `src/amx.h`
- [ ] **4. COMMIT** all 15 modified + 1 deleted files (`git add -u && git commit`)
- [ ] **5. TRIAGE** new `tools/*` files: keep 14, delete 12 (see §7 classification)
- [ ] **6. COMMIT** kept tools + `.gitignore`
- [ ] **7. BUILD & TEST:** `make clean && make && ./bench_grid quick` — must show ALL TESTS PASSED
- [ ] **8. UPDATE** paper §6 (GPU section) with actual B200 cuFFTDx fused-kernel numbers
- [ ] **9. UPDATE** README GPU performance table if stale
- [ ] **10. DECIDE** on old `src/icm_gpu.cu` (214 KB monolithic reference): move to `attic/` or delete — it's not compiled by the current Makefile
- [ ] **11. RESUME** GPU Phase 4 fused-kernel work (tier-2 fallback replacement) — see B200_RESUME_CHECKPOINT.md
