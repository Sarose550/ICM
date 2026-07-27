# Pre-Mortem: Option B Ragged-Tree Fix (GPU Monotonicity)

This document identifies the most likely correctness bugs in a naive
implementation of "Option B" from `scripts/gpu_ragged_tree_fix_plan.md`.
Every item is grounded in the current source code with explicit file/line
references.  The supervisor should check the other worker's diff against
this checklist item-by-item.

---

## 0. Pre-Existing Correctness Bug Discovered During Analysis

**This is the most important finding.**  Before the fix, the GPU code does not
merely *waste compute* on padding slots — it produces **wrong equity values**
for every tree level where `n_real[ell-1]` is odd and `nn[ell-1] > n_real[ell-1]`
(i.e., non-power-of-two block counts).  The mechanism:

- Parent `p = n_real[ell]-1` (the last *real* parent) has children at indices
  `2p` and `2p+1`.  When `n_real[ell-1]` is odd, `2p+1 = n_real[ell-1]` —
  i.e., the right child is a padding slot whose polynomial is all zeros
  (never computed).

- **Build direction**: `specR = FFT(zero_poly) = [0,0,…,0]`.  The pairwise
  multiply `specL × specR × (1/N)` produces all zeros → IFFT → zero parent
  polynomial.  The correct result is `poly_parent = poly_leftchild` (identity
  convolution).  See `k_pairwise_mul` at `gpu_kernels.cu:881-894`.

- **Correlate direction**: `k_paired_corr_freq` (`gpu_kernels.cu:947-975`)
  computes `out_left = g_hat × conj(specR)` for the left child's gradient.
  With `specR = [0,…,0]`, the left child (which *is* a real child) gets zero
  gradient.  The correct result is `g_child_L = g_parent` (correlation with
  identity polynomial).

- This affects **every backend path** (cuFFT, VkFFT, cuFFTDx/fused) and both
  single-Q and Q-batched variants, because all of them read the phantom right
  child's data.  It also affects the **schoolbook** path (`k_schoolbook_build`,
  `k_schoolbook_corr_pair`, and their smem/warp variants), which compute
  `L[j] * R[pos-j]` where `R[j]` is all zeros.

- For `n=1,000,000, B=32`: `n_real[0]=31250` (even — no boundary), but
  `n_real[1]=15625` (odd — boundary at ell=2), `n_real[2]=7813` (odd —
  boundary at ell=3), `n_real[3]=3907` (odd — boundary at ell=4).

**Checklist item 0:** Does the diff fix the boundary-parent correctness bug
(not just the wasted-compute issue)?  If the diff only shrinks launch configs
and cuFFT batch sizes *without* adding the identity-spectrum boundary logic,
the equity results will remain wrong for ragged trees — possibly even more
wrong because the cuFFT plan no longer processes the padding right child
(which was all zeros, producing wrong-but-deterministic output), and instead
the multiply kernel reads whatever garbage happens to be in the un-initialized
tail of the child spectrum buffer.

---

## 1. The Identity-Spectrum Boundary Trick

### 1.1 R2C FFT of [1,0,0,…] — convention verification

**Reference**: `create_cufft_plan()` at `gpu_plan.cu:1244-1295`, cuFFT call
sites at `gpu_exec.cu:260` (`cufftExecD2Z`) and `gpu_exec.cu:276`
(`cufftExecZ2D`).

- cuFFT is used in **default mode** (no `cufftSetCompatibilityMode` call).
  Default cuFFT: both forward and inverse are unnormalized; round-trip
  `D2Z → Z2D` scales by `N = fft_n`.
- `D2Z([1, 0, 0, …, 0])` (length `fft_n`, real) → `[1+0i, 1+0i, …, 1+0i]`
  (length `cn = fft_n/2+1`, complex).  Confirmed: the DC bin = sum of inputs = 1,
  all other bins = exp(0) = 1.
- The multiply kernel injects `scale = 1.0 / fft_n` (see `gpu_exec.cu:271`,
  `gpu_exec.cu:830`).  So `specL × identity × (1/N) = specL / N` → IFFT gives
  back the original left-child polynomial.  ✓

**Checklist item 1.1:** Verify the identity spectrum is written as
`{1.0, 0.0}` for real and imaginary parts at every frequency bin — *not*
`{fft_n, 0.0}` or `{1.0/fft_n, 0.0}`.  A single wrong value produces wrong
equity.

### 1.2 VkFFT normalization convention

**Reference**: `create_vkfft_r2c_plan()` at `gpu_plan.cu:1395-1452`,
VkFFT call sites at `gpu_exec.cu:190-196` (forward) and `gpu_exec.cu:210-216`
(inverse).

- VkFFT uses the same unnormalized convention as cuFFT (no normalization flag
  is set).  The same `k_pairwise_mul` kernel with `scale = 1/fft_n` is used
  for the VkFFT path (`gpu_exec.cu:207`).  ✓ Identity spectrum convention is
  identical.

**Checklist item 1.2:** If the implementation uses a different identity-spectrum
value for the VkFFT path (e.g., because VkFFT's `performR2C` output convention
differs), it would produce a path-dependent equity discrepancy.  Confirm the
same identity value is used for cuFFT and VkFFT.

### 1.3 cuFFTDx fused path convention

**Reference**: `k_cufftdx_build_parent_r2c` at `gpu_kernels.cu:256-278`.

- The fused kernel does R2C → complex multiply (no intermediate scale) →
  C2R → apply `inv_fft_n` on store.  This is equivalent to the cuFFT pipeline.
  Identity spectrum convention is the same.

**Checklist item 1.3:** The fused kernel accesses the right child at
`child + (2*p+1) * child_stride` (line 271 of `gpu_kernels.cu`).  If the
kernel's `nparents` parameter is shrunk to `n_real[ell]`, the boundary parent
`p = n_real[ell]-1` WILL be processed (it's a real parent).  The fused kernel
must include its own boundary guard: when `2*p+1 >= n_real_child`, skip the
R2C of the right child and use the identity spectrum directly in the multiply.
A naive diff that only shrinks `nparents` in the *launch* site
(`gpu_exec.cu:377` → `nparents = plan->n_real[ell]`) without modifying the
fused kernel body will read out-of-bounds or garbage from the right-child
slot.  Check whether the fused dispatch functions (`launch_cufftdx_build_*`,
`launch_cufftdx_corr_*`) receive a new `n_real_child` parameter for this check.

### 1.4 Could fft_n differ between build and correlate paths?

**Reference**: `allocate_level_buffers()` at `gpu_plan.cu:1532-1535, 1573-1574`.

```c
int fft_n = lp.fft_n;
b.fft_n = fft_n;
…
c.fft_n = fft_n;
```

Both `build_fft[ell].fft_n` and `corr_fft[ell].fft_n` are set to the same
`lp.fft_n`.  The child spectrum cache (`d_fft_cache[ell]`) is written during
the build phase with `b.fft_n` and read during the correlate phase with
`c.fft_n` — same value, same `cn`.  ✓ No mismatch possible in the current
plan structure.

**Checklist item 1.4:** Even though they're the same today, if the diff
changes `lp.fft_n` based on `n_real`-derived wrap_scale (see §4 below), it
must ensure `b.fft_n == c.fft_n` still holds.  A mismatch would mean the
correlate path reads the cache with a different `cn` than it was written
with — silent data corruption.

### 1.5 Cache slot for the phantom right child

**Reference**: `allocate_level_buffers()` at `gpu_plan.cu:1610-1613`,
arena cache allocation at `gpu_plan.cu:1726-1728`.

The child FFT cache is allocated with `child_batch` elements.  Currently
`child_batch = nn[ell-1]`.  If this is shrunk to `n_real[ell-1]`, the cache
has exactly `n_real[ell-1]` slots (indices 0 through `n_real[ell-1]-1`).

When `n_real[ell-1]` is odd, the phantom right child for the boundary parent
is at index `n_real[ell-1]` — **one past the end of a n_real-sized cache**.
If the identity spectrum is pre-filled at this index, it's an out-of-bounds
write.

**Checklist item 1.5a:** If the cache is shrunk to `n_real[ell-1]`, does the
diff add +1 to the cache size at levels where `n_real[ell-1]` is odd?  Or does
it handle the identity in the multiply kernel inline (no pre-fill needed)?

**Checklist item 1.5b:** If the cache is NOT shrunk (stays at `nn[ell-1]`),
the phantom right child's spectrum slot at index `n_real[ell-1]` already exists
in the cache but contains whatever the forward FFT wrote there — for padding
children this is all zeros (since the padding child polynomial is zero).  Does
the diff explicitly overwrite this slot with the identity spectrum between the
forward FFT and the multiply kernel?

**Checklist item 1.5c:** If the cache is kept at `nn[ell-1]` size but the
cuFFT forward plan's batch is shrunk to `n_real[ell-1]`, the forward FFT
never writes to the phantom child's spectrum slot (index `n_real[ell-1]`).
That slot contains whatever was left from a previous Q-point or garbage
(uninitialized).  The diff must explicitly fill this slot with the identity
spectrum via a separate kernel or `cudaMemset` before the multiply step.

### 1.6 The "which index is the boundary child" off-by-one

**Reference**: `build_tree_geometry()` at `gpu_plan.cu:510`:
`n_real[ell] = (n_real[ell-1] + 1) / 2`.

The boundary condition is: parent `p` has a phantom right child when
`2*p + 1 >= n_real[ell-1]`.  The last real parent is `p = n_real[ell] - 1`.
So `2*(n_real[ell]-1) + 1 = 2*n_real[ell] - 1`.

- When `n_real[ell-1]` is odd: `n_real[ell] = (n_real[ell-1]+1)/2`,
  so `2*n_real[ell] - 1 = n_real[ell-1]`.  The phantom right child is at
  index `n_real[ell-1]`.  Check: `2*p+1 >= n_real[ell-1]` → true.  ✓
- When `n_real[ell-1]` is even: `n_real[ell] = n_real[ell-1]/2`,
  so `2*n_real[ell] - 1 = n_real[ell-1] - 1`.  The right child is at
  index `n_real[ell-1] - 1`, which is valid.  No boundary case.  ✓

**Checklist item 1.6a:** The boundary check must be `if (2*p + 1 >= n_real_child)`,
**not** `if (2*p + 1 > n_real_child)`.  Using `>` would miss the boundary
case when `n_real[ell-1]` is odd (where `2*p+1 == n_real[ell-1]` exactly).

**Checklist item 1.6b:** The child index for the phantom right child is
`2*p + 1 = n_real[ell-1]` (when odd).  If the code uses `n_real[ell-1] - 1`
as the phantom index, it will overwrite a real child's spectrum instead.
Verify the identity spectrum is written at index `n_real[ell-1]`, not
`n_real[ell-1] - 1`.

**Checklist item 1.6c:** The boundary check needs `n_real_child` (= `n_real[ell-1]`)
at runtime.  The current kernels (`k_pairwise_mul`, `k_paired_corr_freq`)
don't receive this parameter — they only receive `nparents` and `cn`.  Verify
the diff adds the necessary parameter to every affected kernel, or (for the
pre-fill approach) correctly computes the phantom index at the host side.

### 1.7 Both build AND correlate directions need the boundary fix

**Reference**: Build multiply at `gpu_kernels.cu:881-894`, correlate multiply
at `gpu_kernels.cu:947-975`.

In the build direction, the phantom right child → substitute identity spectrum
so `specL × identity = specL` (parent = left child).

In the correlate direction, the phantom right child → substitute identity
spectrum so `g_hat × conj(identity) = g_hat` for the left child's output.
(Conversely, if the *left* child were phantom — which never happens because
`2*p` is always < `2*p+1` — the right child's output would need identity.)

**Checklist item 1.7:** Verify the diff handles the identity substitution in
**both** `k_pairwise_mul` and `k_paired_corr_freq` (and their calling
conventions in both single-Q and Q-batched paths, across cuFFT, VkFFT, and
cuFFTDx backends).  A build-only fix leaves the correlate direction wrong;
a correlate-only fix leaves the build direction wrong.

### 1.8 Schoolbook path boundary case

**Reference**: Schoolbook kernels at `gpu_kernels.cu:792-872`
(`k_schoolbook_build`, `k_schoolbook_build_smem_parent`,
`k_schoolbook_build_warp_batch`) and `gpu_kernels.cu:1016-1082`
(`k_schoolbook_corr_pair`, `k_schoolbook_corr_pair_smem_parent`,
`k_schoolbook_corr_pair_warp_batch`).

If a level uses the schoolbook tier (common for small conv lengths or when
`ICM_GPU_FORCE_TIER=schoolbook`), the boundary parent also needs the identity
fix: when `2*p+1 >= n_real_child`, the right child polynomial is all zeros
and the convolution produces all zeros.  The correct result is to copy the
left child's polynomial.

**Checklist item 1.8:** If the schoolbook kernels' launch configs are shrunk
to `n_real[ell]`, the boundary parent `p = n_real[ell]-1` IS launched but its
right child is phantom.  Verify the schoolbook kernels have a guard that
detects `2*p+1 >= n_real_child` and copies the left child polynomial (build)
or handles the correlation-with-identity (prop).  If the diff doesn't add this
guard, schoolbook-tier levels produce wrong results at the boundary.

---

## 2. cuFFT / VkFFT Plan Batch-Size Changes

### 2.1 Consistency between plan creation and workspace estimation

**Reference**: `allocate_level_buffers()` at `gpu_plan.cu:1527-1615` (plan
creation), `estimate_cufft_workspace_bytes()` at `gpu_plan.cu:1304-1381`
(workspace estimation), `allocate_plan_device_memory()` at
`gpu_plan.cu:1807-1853` (workspace allocation).

The critical chain is:
1. `estimate_cufft_workspace_bytes(qb)` creates trial plans with batch =
   `qb * nn[ell]` → queries `cufftGetSize` → takes max → returns `max_ws`.
2. `allocate_plan_device_memory()` allocates `shared_cufft_workspace` of
   `max_ws` bytes.
3. `allocate_level_buffers()` creates real plans, also with batch =
   `qb * nn[ell]`.
4. All plans are bound to `shared_cufft_workspace` via `cufftSetWorkArea`.

**If the plan-creation batch is shrunk to `n_real` but the workspace
estimation still uses `nn`, the workspace will be over-sized (safe but wastes
VRAM).**

**If the workspace estimation is shrunk to `n_real` but plan creation still
uses `nn`, the workspace will be under-sized.**  cuFFT will write past the
allocated workspace buffer.  This is a **silent out-of-bounds write** — no
CUDA error, no segfault, just corrupted memory that may manifest as wrong
equity values or a crash much later.

**Checklist item 2.1a:** Verify that `estimate_cufft_workspace_bytes()` and
`allocate_level_buffers()` both use the same batch values.  There are TWO
code paths in `estimate_cufft_workspace_bytes`: the table-lookup path
(lines 1316-1345, using `plan->nn[ell]`) and the fallback trial-plan path
(lines 1348-1381, also using `plan->nn[ell]`).  Both must change.

**Checklist item 2.1b:** The table-lookup path multiplies per-batch-unit
workspace by `qb * child_batch` (where `child_batch = nn[ell-1]`).  If the
table was calibrated at batch=1, the linear scaling `pb_r2c * batch` is
correct.  Verify the diff uses `n_real[ell-1]` here (not `n_real[ell]`).

### 2.2 Lazy plan creation path (FUSED-tier fallback)

**Reference**: `ensure_cufft_plans_for_level()` at `gpu_exec.cu:303-363`.

When a FUSED-tier level's cuFFTDx dispatch fails at runtime, this function
creates cuFFT plans lazily using the stored `b.batch_fwd`, `b.batch_inv`,
etc.  These fields are set in `allocate_level_buffers()` from the *local
variables* `child_batch` and `parent_batch`.

**Checklist item 2.2:** If `allocate_level_buffers()` changes `child_batch`
and `parent_batch` to `n_real`, then `b.batch_fwd` etc. are automatically
updated.  Verify that `ensure_cufft_plans_for_level()` is NOT independently
hardcoding `nn`-based batch values (it reads from the stored fields, so
it should be fine — but verify).

### 2.3 VkFFT plan batch sizes

**Reference**: `gpu_plan.cu:1562-1571` (VkFFT build plans),
`gpu_plan.cu:1593-1602` (VkFFT corr plans).

VkFFT plans are created with `batch` derived from `qb * child_batch` and
`qb * parent_batch`.  Same consistency requirement as cuFFT.

**Checklist item 2.3:** Verify VkFFT plan creation uses `n_real`-based batch
sizes.  Note that VkFFT doesn't have a separate workspace allocation (it
uses the `bufferSize` computed from `cn * batch * sizeof(complex)`), so there's
no separate workspace-estimation step to check.  But the `buf_size` and
`io_buf_size` values at lines 1415-1417 must match the actual data size
passed at runtime.

### 2.4 Spec buffer and FFT scratch buffer sizing

**Reference**: `allocate_plan_device_memory()` at `gpu_plan.cu:1645-1652`
(spec buffers) and `gpu_plan.cu:1658-1665` (FFT scratch).

- Spec buffers (`mb_si`, `mb_sm`, `mc_si`, `mc_sm`): computed as MAX across
  levels of `qb * cb * cn * sizeof(complex)` where `cb = nn[ell-1]` and
  `pb = nn[ell]`.  These are shared across all levels — each level reuses the
  same buffer.  The size must be ≥ the largest level's requirement.

- FFT scratch: `qb * cb * fft_n * sizeof(double)` where `cb = nn[ell-1]`.

**Checklist item 2.4a:** If these sizes stay at `nn`-based values, they're
larger than needed but safe.  If they're shrunk to `n_real`-based values,
verify that EVERY call site that writes to these buffers uses a batch ≤ the
new size.  Specifically: `k_gather_to_fft` at `gpu_exec.cu:185` uses
`batch = total_child`, and if `total_child` was shrunk to `qb * n_real[ell-1]`
but the scratch buffer was computed with `nn[ell-1]`, it's safe (buffer larger
than needed).  The dangerous case is if scratch is shrunk but `batch` isn't —
then `k_gather_to_fft` writes OOB.

**Checklist item 2.4b:** The arena allocation for `d_fft_cache[ell]` at
`gpu_plan.cu:1726-1728` uses `nn[ell-1]`.  If the actual cache usage shrinks
to `n_real[ell-1]`, the arena slot is oversized — safe.  If the arena
allocation is shrunk but `allocate_level_buffers` (or the separate
`alloc_device` path for cache) still allocates at `nn` size, there's a
mismatch between the arena slot and the separately-allocated buffer — but
`allocate_level_buffers` only allocates if `d_fft_cache[ell]` is null, which
it won't be after the arena assignment.  So no conflict.

### 2.5 FFT cache validity and recompute path

**Reference**: `run_prop_level_fft()` at `gpu_exec.cu:507-531` (single-Q,
VkFFT child recompute) and `gpu_exec.cu:593-613` (single-Q, cuFFT child
recompute).

When the cache is invalidated (`fft_cache_valid[ell] = false`), the correlate
path recomputes the child spectra by running a forward FFT on
`d_poly_levels[ell-1]`.  The recompute uses `child_batch = plan->nn[ell-1]`
(lines 511, 600) to size the gather and FFT.

**Checklist item 2.5:** If the child FFT plan's batch was shrunk to
`n_real[ell-1]`, the recompute path must use the same `n_real[ell-1]` batch —
otherwise `cufftExecD2Z(b_fft.plan_fwd, …)` will process fewer or more
signals than the plan expects, which is undefined behavior.

---

## 3. Under-Allocation / Under-Launch From Partial Fix

### 3.1 The "some sites changed, some didn't" hazard

There are **18+ distinct locations** where `plan->nn[ell]` or
`plan->nn[ell-1]` controls GPU work size in `gpu_exec.cu`:

| Function | Variable | Line(s) |
|---|---|---|
| `run_build_level_schoolbook` | `nparents = nn[ell]` | 115 |
| `run_build_level_fft` | `child_batch = nn[ell-1]`, `parent_batch = nn[ell]` | 167-168 |
| `run_build_level_fused` | `nparents = nn[ell]` | 377 |
| `run_prop_level_schoolbook` | `nparents = nn[ell]` | 416 |
| `run_prop_level_fft` | `nparents = nn[ell]` | 472 |
| `run_prop_level_fused` | `nparents = nn[ell]` | 666 |
| `run_build_level_schoolbook_qb` | `nparents_total = qb * nn[ell]` | 708 |
| `run_build_level_fft_qb` | `child_batch = qb * nn[ell-1]`, `parent_batch = qb * nn[ell]` | 745-746 |
| `run_build_level_fused_qb` | `nparents_total = qb * nn[ell]` | 854 |
| `run_prop_level_schoolbook_qb` | `nparents_total = qb * nn[ell]` | 886 |
| `run_prop_level_fft_qb` | `nparents_total = qb * nn[ell]` | 934 |
| `run_prop_level_fused_qb` | `nparents_total = qb * nn[ell]` | 1081 |
| (implicit) scatter in correlate paths | `n_children = 2 * nparents` | 479, 935 |

Plus the `build_plan_metadata()` usage at line 887 and `choose_uncached_levels()`
at lines 1113, 1137, 1149.

**Checklist item 3.1:** Print a diffstat of every line that changes.  If any
of the above sites still uses `nn` while others use `n_real`, flag it.
Specifically:

- **If `run_build_level_fft` changes `child_batch`/`parent_batch` but
  `run_build_level_fft_qb` doesn't**: single-Q works, Q-batched is wrong.
- **If plan creation (`allocate_level_buffers`) changes batch but execute
  sites don't**: the cuFFT plan expects `n_real` signals but is given `nn`
  signals' worth of data → cuFFT reads beyond what was gathered.
- **If execute sites change but plan creation doesn't**: the cuFFT plan
  processes `nn` signals but only `n_real` were gathered → cuFFT reads
  uninitialized data for the tail.

### 3.2 n_children computation

**Reference**: `gpu_exec.cu:479`: `int n_children = 2 * nparents;`

This is used for the scatter kernel after inverse FFT in the correlate path.
If `nparents` shrinks to `n_real[ell]`, then `n_children = 2 * n_real[ell]`.
When `n_real[ell-1]` is odd, `2 * n_real[ell] = n_real[ell-1] + 1`.  The
scatter writes child index `n_real[ell-1]` (one past the last real child).
This is fine as long as the destination `d_g_levels[ell-1]` array is still
sized to `nn[ell-1]` (it is — see §3.4 of the design doc).

**Checklist item 3.2:** Confirm the scatter doesn't truncate the last
output.  The scatter kernel writes `n_children * scatter_len` elements.
With `n_children = 2 * n_real[ell]`, this includes the phantom child's
output.  The phantom child's output data is harmless (never read by the
next level, which only processes `n_real[ell-1]` children), but the scatter
must not be truncated to `n_real[ell-1]` children — it would drop the
rightmost real child's gradient when `n_real[ell-1]` is even, or produce
misaligned output.

### 3.3 d_fft_scratch sizing vs batch

**Reference**: `gpu_plan.cu:1658-1665`:
```c
int cb = plan->nn[ell - 1];
size_t need = (size_t)qb * cb * fft_n * sizeof(double);
fft_scratch_bytes = std::max(fft_scratch_bytes, need);
```

**Checklist item 3.3:** If `cb` is changed to `n_real[ell-1]`, the scratch
buffer shrinks.  Every gather kernel that writes to scratch must use a batch
≤ `n_real[ell-1] * qb`.  Currently, the execute sites gather with
`batch = child_batch` or `batch = parent_batch` or `batch = nparents`.  If
the execute site shrinks to `n_real`, the gather batch shrinks too → safe.
But if ANY execute site still gathers with `nn` batch into a scratch buffer
sized for `n_real`, it's an OOB write.

### 3.4 Single-Q vs Q-batched symmetry

**Checklist item 3.4:** Every change made to a `run_*_level_*` function
(without `_qb` suffix) must have a corresponding change in the `_qb` variant.
The functions are structurally identical (differing only in `qb` factor and
use of `plan->d_poly_levels` vs `plan->d_poly_levels` with Q-batch stride).
Verify line-by-line symmetry.

---

## 4. Interaction With the `below_sat` / `g_eff_max` Fix

### 4.1 Direct interaction: none

**Reference**: `build_plan_metadata()` at `gpu_plan.cu:923-924`:
```c
int g_eff_max = is_below ? std::min(cps + cps / 2, pgsz) : pgsz;
int g_eff = std::min(g_eff_needed, g_eff_max);
```

`g_eff_max` depends on `cps`, `pgsz`, and `is_below` (= `below_sat[ell]`).
None of these involve `nn` or `n_real`.  The ragged-tree fix changes the
*number* of parents processed, not the polynomial sizes *per parent*.
`g_eff` is a per-parent quantity.  ✓ No direct interaction.

### 4.2 Indirect interaction through wrap_scale → FFT size selection

**Reference**: `build_plan_metadata()` at `gpu_plan.cu:930`:
```c
double wrap_scale = wrap_serial_penalty_gpu(nparents);
```
where `nparents = plan->nn[ell]` (line 912).

`wrap_scale` feeds into `best_fft_config_gpu()` and
`best_fft_config_joint_gpu()` as the `correction_scale` parameter.  This
affects the cost of wrap correction, which influences FFT size selection and
therefore `fft_n`, `cn`, and `lp.cn`.

If `nparents` changes from `nn[ell]` to `n_real[ell]`, `wrap_scale` changes
(slightly larger penalty because fewer parents → less parallelism → more
serial wrap correction).  This could change the selected `fft_n` at some
levels, which changes `cn = fft_n/2 + 1`.  `cn` is used by:
- `lp.cn` → cache sizing, spec buffer sizing, kernel launch sizing
- `lp.corr_conv` computation depends on `g_eff` and `p_eff` (neither depends
  on `cn`), so `corr_conv` is unaffected
- `lp.build_conv` is unaffected

**Checklist item 4.1:** A change in `fft_n` at any level would cascade to
changes in `cn`, which affects cache sizes, spec buffer sizes, and the
identity spectrum length.  Verify that: (a) the diff changes `nparents` to
`n_real[ell]` in `build_plan_metadata()` line 912, and (b) if `fft_n` changes
as a result, the FFT cache and spec buffer sizing (which are computed later
in `allocate_plan_device_memory`) use the updated `lp.cn` values (they do,
since `lp.cn` is read from the per-level plan, which was already computed).

### 4.3 choose_uncached_levels cache budget

**Reference**: `choose_uncached_levels()` at `gpu_plan.cu:1110-1153`.

The cache budget computation uses `nchild = plan->nn[ell-1]` (lines 1113,
1137, 1149) to estimate total cache bytes.  If the actual cache size shrinks
to `n_real[ell-1]` but the budget still uses `nn[ell-1]`, the budget check
thinks the cache is larger than it really is — it may unnecessarily uncache
levels (thinking the budget is exceeded when it isn't).

**Checklist item 4.2:** This is a performance-only issue (no correctness
impact), but if the diff changes `nn[ell-1]` to `n_real[ell-1]` in
`choose_uncached_levels`, verify the arena cache allocation at
`gpu_plan.cu:1726-1728` also uses `n_real[ell-1]` — otherwise the arena
reserves more space than `choose_uncached_levels` accounts for.

### 4.4 estimate_candidate_cost uses nn

**Reference**: `estimate_candidate_cost()` at `gpu_plan.cu:656-658`:
```c
double eff_batch = assumed_qb * (double)nn[ell];
…
tree_ns += (double)nn[ell] * (build_ns + corr_ns);
```

This function is used for B-selection and engine-selection during plan
creation.  It estimates total tree cost by multiplying per-parent cost by
`nn[ell]` (the padded count).  If execution switches to `n_real[ell]` but
the cost model still uses `nn[ell]`, the model overestimates cost.  This
could cause:
- Wrong B selection (model thinks large B is more expensive than it is)
- Wrong engine selection (model prefers linear engine when hybrid is actually
  cheaper)

**Checklist item 4.3:** If the diff changes the cost model to use
`n_real[ell]`, the B-selection and engine-selection heuristics become more
accurate.  If it doesn't, the model is slightly pessimistic for ragged trees
but still correct for power-of-two trees (where `nn == n_real`).  This is
a non-critical but desirable change.

---

## Summary Checklist (For Quick Diff Review)

- [ ] **0.** Fixes the pre-existing correctness bug (zero output for boundary parents), not just wasted compute.
- [ ] **1.1.** Identity spectrum = `{1.0, 0.0}` at every frequency bin (not `fft_n` or `1/fft_n`).
- [ ] **1.2.** Same identity value used for cuFFT and VkFFT paths.
- [ ] **1.3.** Fused (cuFFTDx) kernels have boundary guard (can't just shrink launch config).
- [ ] **1.4.** `b.fft_n` still equals `c.fft_n` if `fft_n` changes due to wrap_scale change.
- [ ] **1.5a-c.** Phantom child's spectrum slot is either (a) in a +1-sized cache, (b) explicitly overwritten if cache stays at nn size, or (c) explicitly filled if forward FFT batch was shrunk.
- [ ] **1.6a.** Boundary check uses `>=`, not `>`.
- [ ] **1.6b.** Phantom child index = `n_real[ell-1]`, not `n_real[ell-1]-1`.
- [ ] **1.6c.** `n_real_child` parameter added to multiply/correlate kernels (or pre-fill avoids need).
- [ ] **1.7.** Identity substitution in BOTH `k_pairwise_mul` and `k_paired_corr_freq`.
- [ ] **1.8.** Schoolbook kernels have boundary guard if their launch configs are shrunk.
- [ ] **2.1a.** `estimate_cufft_workspace_bytes` (both table and fallback paths) uses same batch as plan creation.
- [ ] **2.1b.** Table-lookup path multiplies by `n_real[ell-1]`, not `nn[ell-1]`.
- [ ] **2.2.** `ensure_cufft_plans_for_level` reads stored `batch_fwd`/`batch_inv` (auto-updated if `allocate_level_buffers` changed).
- [ ] **2.3.** VkFFT plan creation uses `n_real`-based batch.
- [ ] **2.4a-b.** Spec buffer, scratch buffer, and cache arena sizes consistent with actual usage.
- [ ] **2.5.** Child recompute path in correlate uses `n_real[ell-1]` batch.
- [ ] **3.1.** All 18+ call sites changed consistently (single-Q, Q-batched, all tiers).
- [ ] **3.2.** `n_children = 2 * n_real[ell]` scatter not truncated; phantom child output is harmless.
- [ ] **3.3.** Scratch buffer sized ≥ largest gather batch.
- [ ] **3.4.** Single-Q and Q-batched variants changed symmetrically.
- [ ] **4.1.** `wrap_scale` uses `n_real[ell]` in `build_plan_metadata`; cascade to `fft_n`/`cn` tracked.
- [ ] **4.2.** `choose_uncached_levels` budget uses same `n_real` as actual cache size.
- [ ] **4.3.** `estimate_candidate_cost` uses `n_real[ell]` (desirable but not critical).
