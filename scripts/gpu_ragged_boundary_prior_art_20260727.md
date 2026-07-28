# Prior Art Research: Ragged-Boundary Handling in Register-Heavy GPU Kernels

**Date:** 2026-07-27
**Scope:** External research only — no code changes, no builds, no hardware
runs. This is a distinct research pass from `scripts/ragged_tree_cufftdx_research.md`
(node R0), which investigated the register-perturbation *mechanism* itself.
This document investigates whether the two already-designed fixes
(Candidate C: pre-fill identity element; Candidate D: separate boundary
kernel) have real prior art, and whether cuFFTDx/cuFFT expose a
built-in mechanism that would make either unnecessary.

**Bottom line up front:** No cuFFTDx-specific or cuFFT-callback-based
shortcut exists. Both Candidate C and Candidate D turn out to be
specific instances of two well-established, independently-documented
general GPU patterns for ragged boundaries — they were derived from
first-principles reasoning, but that reasoning reinvented real prior
art rather than departing from it. No Candidate E emerged.

---

## 1. cuFFTDx-specific: no callback mechanism, no ragged-batch handling

**Searched:** cuFFTDx official docs (Examples index, Execution Methods,
Requirements and Functionality, Introduction), the full examples list at
`docs.nvidia.com/cuda/cufftdx/examples.html`, NVIDIA developer forum
threads mentioning cuFFTDx + batch/padding, and the
`NVIDIA/CUDALibrarySamples` GitHub repo (`MathDx/cuFFTDx`).

**Findings:**

- The complete cuFFTDx examples list (fetched directly) is:
  `introduction_example`; `simple_fft_thread[_fp16]`,
  `simple_fft_block[_half2|_fp16|_c2r|_r2c|_shared|_std_complex|_cub_io|_block_dim]`;
  `nvrtc_fft_thread`, `nvrtc_fft_block`, `nvrtc_query_database_fft_block`;
  `block_fft_performance[_many]`; `convolution`, `convolution_padded`,
  `convolution_r2c_c2r`, `convolution_performance`; `convolution_3d[_c2r|_r2c|_padded|_padded_r2c]`;
  `fft_2d[_single_kernel|_r2c_c2r|_single_kernel_block_dim]`, `fft_3d`,
  `fft_3d_box_single_block`, `fft_3d_cube_single_block`;
  `mixed_precision_fft_1d`, `mixed_precision_fft_2d`.
  **None** of these address variable/irregular batch counts, a ragged
  "last block has fewer real elements" pattern, or load/store callbacks.
  `convolution_padded` is the closest-sounding name, but it pads FFT
  *length* to a fast/smooth size (the same kind of padding this
  codebase already does via `best_k_pad()`), not batch *count*.

- cuFFTDx has **no callback API analogous to cuFFT's `cufftXtSetCallback`.**
  It is a header-only, compile-time-templated, register-resident library:
  the "load"/"store" functions documented in its Execution Methods page
  (e.g. `cufftdx_load_real_r2c` in this codebase) are ordinary user-written
  device functions that populate the per-thread register array before
  `FFT().execute(thread_data, shared_memory)` — they are inlined into the
  *same* kernel body and compiled as part of the *same* function. This is
  architecturally different from cuFFT's callback mechanism, which
  registers a separate device function pointer that cuFFT's own runtime
  invokes at a fixed point in its (also register-resident, but
  library-owned) generated kernel. Using cuFFTDx's load function to
  "inject an identity value at load time" is not a distinct capability
  from Candidate C — it *is* Candidate C's mechanism (supply identity data
  before/at the point of load), just executed inside the same kernel
  instead of via a pre-fill kernel/memset. It does not let you avoid
  either the extra FFT-on-identity-data cost (still runs `R2C().execute()`
  on the phantom child) or a branch (if you wanted to skip the FFT
  entirely for the phantom, you're back to a branch).

- No forum thread, release-note entry, or GitHub issue in
  `NVIDIA/CUDALibrarySamples` discusses ragged/irregular batch handling for
  cuFFTDx. One forum thread ("Decide FFT size during runtime with
  cufftDx") asks the adjacent question (runtime-variable FFT *size*, not
  batch count) and has no substantive answer in the thread. This
  reinforces the R0 finding: this exact problem class is not publicly
  discussed for cuFFTDx.

**Conclusion for §1:** cuFFTDx offers nothing that would let you inject
the phantom-child identity value cheaper than Candidate C already
proposes, and nothing that would let you skip the FFT without a branch.
The "callback at load time, same kernel invocation, avoiding both C's
wasted FFT and D's extra launch" possibility raised in the task does not
exist for cuFFTDx.

---

## 2. cuFFT's callback API: real, but doesn't transfer

**Searched:** cuFFT documentation (`cufftXtSetCallback`, `CUFFT_CB_LD_REAL`
etc.), the NVIDIA "CUDA Pro Tip: Use cuFFT Callbacks for Custom Data
Processing" blog post, cuFFT LTO-EA docs (`cufftXtSetJITCallback`), and
forum/GitHub threads discussing callback edge cases.

**Findings:**

- `cufftXtSetCallback` is real, well-documented, and old (present since
  early cuFFT versions, with a newer LTO-IR-based `cufftXtSetJITCallback`
  variant). It registers a device function that cuFFT's own generated
  kernel calls at the load point (receiving the input address + element
  offset, returning the value cuFFT should use) or the store point
  (receiving the computed value, transforming it before write). This is
  exactly the shape of mechanism the task asked about — a way to inject
  a computed/substitute value at load time, inside the same kernel,
  without an extra launch.
- The official NVIDIA blog post introducing this feature frames it purely
  as a bandwidth optimization for DSP pre/post-processing (avoiding a
  separate elementwise kernel before/after the FFT). **It does not discuss
  padding, ragged/irregular batch counts, or supplying a default/identity
  value for a missing/boundary element.** No source found (blog, docs, or
  forum) documents using a cuFFT load callback specifically for a
  "phantom element" or ragged-boundary pattern.
- Critically, this is cuFFT's **classic host-orchestrated plan/execute
  API**, not cuFFTDx. The two libraries are unrelated at the
  implementation level (cuFFTDx does not sit "on top of" cuFFT's callback
  machinery); a cuFFT callback pattern does not translate into cuFFTDx
  because cuFFTDx has no equivalent registration point to hook (§1). This
  codebase deliberately uses cuFFTDx for the fused hot path specifically
  *because* it avoids cuFFT's plan/execute host-orchestration overhead —
  adopting a cuFFT-callback-style pattern here would mean reverting to
  the classic cuFFT API for this kernel, which is a different, larger
  architectural change than either Candidate C or D and was not what was
  asked.

**Conclusion for §2:** The mechanism exists in a sibling library, but
there's no documented use of it for exactly this pattern, and it doesn't
transfer to cuFFTDx because cuFFTDx has no callback hook at all.

---

## 3. General GPU HPC prior art for ragged/boundary cases without in-kernel branches

This is where real, concrete prior art was found — converging on the same
two mechanisms already designed as Candidates C and D.

### 3.1 ModernGPU segmented reduction (Sean Baxter) — matches Candidate D's philosophy

`moderngpu.github.io/segreduce.html` (a well-known, widely-cited
open-source GPU primitives library) documents a segmented-reduction
design that explicitly avoids branching on segment/tile boundaries inside
the hot per-element kernel:

- A **separate partitioning kernel** maps CSR segment descriptors to
  tiles via binary search, decoupling tile assignment from segment
  geometry entirely — "tiles always process NV uniform elements" whether
  or not that tile's elements span a ragged segment boundary.
- Within a tile, segment ends are recorded as **flag bits in a bitfield**
  (`endFlags`) rather than checked with a per-element conditional branch —
  a data-parallel/predicated representation, not control flow.
- Segments that **span a tile boundary** (the actual "ragged" case,
  directly analogous to this codebase's tree-level boundary parent) are
  NOT resolved inside the hot kernel at all. They are handled by "a
  cooperative segmented-scan on the carry-out values to produce carry-in
  values," i.e. a **separate fix-up kernel phase** run after the main
  reduction.

This is structurally the same shape as **Candidate D**: keep the hot,
register/occupancy-sensitive kernel's control flow uniform and free of
data-dependent branches for the boundary case; handle the boundary
condition in a distinct kernel launch instead.

### 3.2 CORA tensor compiler (ragged tensors, arXiv:2110.10221) — matches Candidate C's philosophy

CORA ("Cora: A Tensor Compiler for Compiling Ragged Tensors with Minimal
Padding") targets exactly the problem class named in the task: ragged
(non-rectangular) tensor operations on GPUs. Its stated approach is
padding ragged data to rectangular form and having uniform kernel code
operate over the padded shape, using neutral/identity elements at the
padded positions (e.g., zero for a sum-reducing op, one for a
product-reducing op) so the padded region doesn't perturb the result —
the same **pad-with-identity-so-every-lane-does-the-same-work** idea as
Candidate C.

**Caveat on this citation:** the automated fetch/summarization tool
used to read this PDF returned a summary that tracked the wording of my
own question suspiciously closely (it echoed "register allocation
disruption," "occupancy," etc., in language that reads more like it was
answering my prompt than quoting the paper). I could not independently
re-verify a verbatim quote from the PDF this session. I'm reporting the
paper's *existence and title* (a real, citable arXiv paper on minimal-
padding compilation for ragged tensors) and the *general, well-known
padding-for-uniformity idea in the ragged-tensor compiler literature* as
solid; I am **not** confident in the specific "register allocation
disruption" phrasing attributed to it above, and you should not cite that
exact language as coming from CORA without reading the PDF directly.

### 3.3 Ragged Paged Attention for TPU (arXiv:2604.15464) — same philosophy, different hardware

This is a modern (2026), actively-discussed kernel (also integrated into
vLLM's TPU backend) for exactly the "ragged sequence length" problem in
LLM inference attention kernels — a close conceptual cousin of a ragged
subproduct-tree boundary. Its documented approach: keep ragged/dynamic
dimensions **off the tiling dimensions** used by the compute kernel, and
resolve the raggedness via layout/indexing decisions (dynamic slicing)
made by the surrounding orchestration (XLA layout assignment / Pallas
grid), not via per-element branches inside the hot compute loop. This is
TPU/Pallas-Mosaic, not CUDA, so it isn't directly portable code, but it's
independent, contemporary confirmation of the same design principle:
push raggedness out of the register/occupancy-sensitive innermost kernel
into either data (padding) or surrounding launch/indexing logic — never
a data-dependent branch in the hot path.

### 3.4 Segmented scan (Blelloch-style) literature — same head-flag pattern

General segmented-scan literature (Sengupta et al., "Scan Primitives for
GPU Computing"; the Accelerate-language "Irregular Segmented Look-Back
Scans" thesis) confirms head-flags/bitfields are the standard mechanism
for marking ragged segment boundaries within a GPU scan, consistent with
§3.1 — flags-as-data, not branches-as-control-flow, and boundary-spanning
segments get a dedicated carry-propagation step. This is the same pattern
independently re-confirmed across multiple sources/decades of GPU
primitives work, not a one-off.

### 3.5 PyTorch NestedTensor / RaggedTensor — weak signal, not a strong counter-example

PyTorch NestedTensor is explicitly a "prototype" feature with narrow
operator coverage (index, dropout, softmax, transpose, reshape, linear,
bmm). Documentation confirms that converting a NestedTensor to a dense
tensor for use with most ops still means **padding to uniform shape**.
This doesn't add a new pattern beyond §3.2/§3.4 (pad-to-uniform is still
the operative idea), and coverage is too narrow to draw strong
conclusions about kernel-level implementation choices from it.

**Conclusion for §3:** Two real, independently-documented, well-known
patterns exist for exactly this class of problem: (a) pad the ragged
boundary with a neutral/identity element so every lane/tile does uniform
work (CORA; segmented-scan head-flags/carry-in this-generation
descendant) — this is Candidate C; (b) push the boundary case out of the
hot kernel into a separate launch/phase (ModernGPU's carry-in fix-up
kernel; TPU ragged-attention's layout-level resolution) — this is
Candidate D. No third pattern was found.

---

## 4. The register-perturbation-from-branches angle: partially documented, not to the exact specificity needed

**Searched:** CUDA C++ Best Practices Guide (table of contents fetched;
full "Branch Predication" §12.2 body could not be retrieved this session
— the document is too long for the fetch tool and was truncated before
reaching that section), "Control Flow Management in Modern GPUs"
(arXiv:2407.02944), NVIDIA's "How to Improve CUDA Kernel Performance with
Shared Memory Register Spilling" blog, "Register Cache: Caching for
Warp-Centric CUDA Programs" blog, and forum/blog searches specifically for
"branch before register-heavy code perturbs allocation."

**Findings:**

- The CUDA C++ Best Practices Guide does have a dedicated "Branch
  Predication" section (§12.2, confirmed present in the table of
  contents), but I could not retrieve its literal text this session (tool
  truncation on a very long document). From general, well-established
  CUDA knowledge (not verified via a fresh quote this session, flagged
  accordingly): the guide's documented predication behavior is about the
  compiler converting *short* branch bodies into predicated instructions
  below some instruction-count threshold — this is a different mechanism
  from "a branch's mere textual/structural presence shifts register
  allocation for the whole function," and the guide (as best I can
  recall/reconstruct, unverified this session) does not discuss the
  latter.
- "Control Flow Management in Modern GPUs" (arXiv:2407.02944) — fetched
  and read directly — covers SIMT control-flow/reconvergence mechanisms
  (stack-based execution, `BMOV`/`BSSY`/`BSYNC`/`BREAK` compiler-inserted
  instructions for branch synchronization) but **does not discuss**
  branch presence affecting register allocation for surrounding code, nor
  best practices for structuring branches to preserve allocator decisions
  in cooperative/warp-synchronous kernels. This is a negative result:
  a directly-relevant-sounding 2026 paper on GPU control flow turned out
  not to cover this specific angle.
- No blog post, forum thread, or paper was found that documents the
  specific idiom "structure your branch to be register-neutral" or "place
  branches after register-heavy declarations to avoid perturbing
  allocation" as a named NVIDIA best practice. The closest real artifacts
  are the existence of `__launch_bounds__` and `--maxrregcount`
  (confirmed, real, standard tools for constraining/tuning register
  allocation) — but these control the *ceiling*, not *which* variables
  get register-allocated when a dead branch is textually present; they
  don't constitute a documented fix for this specific failure mode, only
  general-purpose register-pressure knobs one could try empirically.
- Search for "avoid conditional before shared memory declaration register
  spill" returned only generic CUDA warp-programming and register-spilling
  material (Cooperative Groups, `__shfl_sync`/`__syncwarp`, CUDA 13's new
  shared-memory register-spilling feature) — nothing that names this
  specific branch-placement idiom.

**Conclusion for §4:** This mirrors R0's finding almost exactly — the
general phenomenon (dead code changes ptxas's register allocation, which
is a known, real, NP-hard-optimization-driven compiler behavior) is
real and stems from well-documented adjacent tools (`__launch_bounds__`,
`--maxrregcount` existing for this class of problem), but no source found
in either research pass documents the exact idiom "place/avoid branches
this way relative to register-heavy code to prevent allocation
perturbation" as a named best practice. There is no compiler-level
Candidate E (a pragma or idiom) to recommend from this angle — the
prior-art-backed answer is still "don't have the branch at all" (C or D),
not "structure the branch more carefully."

---

## 5. Verdict

**Does established practice support Candidate C, Candidate D, both, or
neither?**

**Both — independently, not as guesses.** Candidate C (pad the ragged
boundary with an identity/neutral element, uniform control flow) matches
CORA's documented approach to ragged-tensor compilation and the
head-flag/neutral-element idiom used across the segmented-scan/reduction
literature. Candidate D (handle the boundary case in a separate kernel
launch, keep the hot kernel branch-free) matches ModernGPU's
segmented-reduction carry-in fix-up kernel design, and the same
"resolve raggedness outside the hot compute kernel" philosophy shows up
independently in a 2026 TPU ragged-attention kernel. These are two
established, real, independently-sourced patterns for exactly this
problem class — the fact that this session's earlier first-principles
design work (`gpu_register_pressure_fix_design_20260727.md`) arrived at
both of them independently is a point in their favor, not a coincidence
to be suspicious of.

**Is there a Candidate E worth proposing?** No. The one plausible source
of a genuinely new option — a cuFFTDx or cuFFT load/store callback that
injects the identity value at load time within the *same* kernel
invocation, avoiding both C's extra FFT and D's extra launch — does not
exist for cuFFTDx (no callback mechanism at all, confirmed via docs +
examples + absence of any forum/GitHub discussion) and exists but doesn't
transfer for cuFFT's classic API (real feature, never documented for this
use case, and architecturally foreign to cuFFTDx's register-resident
design that this codebase specifically chose cuFFTDx to get). No other
candidate mechanism turned up in the general GPU-HPC literature search
beyond the pad-with-identity / separate-kernel-phase duality already
captured by C and D.

**Recommendation given this research:** Proceed with the existing
ranking in `gpu_register_pressure_fix_design_20260727.md` (Candidate A
first as a cheap, zero-risk check; Candidate C second, now with
independent literature support from CORA/segmented-scan neutral-element
padding; Candidate D third, now with independent literature support from
ModernGPU's fix-up-kernel pattern). Prior art gives no reason to reorder
C ahead of D or vice versa — both are legitimate, real-world patterns for
this exact problem class, and the choice between them should continue to
rest on the practical tradeoffs already identified in the design doc (one
extra FFT per odd level vs. one extra kernel launch per odd level), not
on one having stronger prior-art backing than the other.

**Honest gaps:** (1) I could not retrieve the literal CUDA C++ Best
Practices Guide §12.2 text this session (tool truncation) — the
predication-threshold claim in §4 is from general knowledge, not a fresh
quote, and is flagged as such. (2) The CORA-paper summary in §3.2 came
from an automated PDF-summarization tool whose output tracked my prompt
suspiciously closely; I'm confident the paper is real and about
minimal-padding ragged-tensor compilation, but not confident in the exact
"register allocation disruption" language attributed to it. (3) No exact
public documentation of the specific register-perturbation mechanism
(§4) was found in this pass, consistent with R0's earlier finding — this
remains corroborated-by-adjacent-evidence, not directly confirmed by a
named source.
