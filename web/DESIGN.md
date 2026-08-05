# Web widget design decisions

Browser ICM calculator: the real C library compiled to WebAssembly, running
client-side on a static page. This file records every decision made during the
build and why. Owner-settled choices from the project charter are marked
(charter); choices made during implementation are marked (build) with their
evidence.

## Architecture

- **Client-side WASM, static hosting, no server** (charter). `src/cpu/icm.c`
  is compiled unmodified with Emscripten against `devices/generic/`, the
  uncalibrated fallback profile: correct results, FFTW_ESTIMATE plans,
  always-hybrid dispatch at B=32. The browser is treated as just another
  uncalibrated device.
- **No C shim** (build). `icm_init` and `icm_equity` are exported directly
  (`-sEXPORTED_FUNCTIONS`). Input validation, zero-stack filtering, and the
  Q ladder live in the JavaScript worker layer, so `src/cpu/` carries zero
  browser-specific code.
- **FFTW 3.3.10 compiled to WASM** with emconfigure/emmake, version pinned and
  sha256-verified in `web/build_fftw.sh` (charter: FFTW stays; swapping FFT
  backends to dodge GPL was explicitly rejected for v1).
- **Compute runs in a Web Worker** (charter) so the UI thread never blocks.
- **Vanilla JS, no framework, hand-rolled SVG chart** (owner, 2026-08-05):
  zero dependencies beyond FFTW keeps the page auditable and the licensing
  story minimal.
- **Prebuilt `web/dist/` is committed** (build): GitHub Pages deploys the
  repo content as-is with no Emscripten toolchain in CI. The build scripts
  that reproduce the bundle are committed alongside (GPL compliance item).

## Numerical policy

- **Q ladder instead of fixed Q=256** (build; deviates from the charter's
  "Q fixed at 256", flagged to the owner). Measurement on the native library
  showed Q=256 is not sufficient across the widget's input envelope: required
  Q grows with n and with stack spread. Evidence (relative residual of
  sum(equity) against sum(payouts), native m3_pro build):
  - n=1000, k=150, spread 1e6: Q=256 gives 8e-4, Q=1024 gives 4e-12
  - n=10000, k=1500, spread 1e3: Q=256 gives 5e-4, Q=2048 gives 7e-11
  - n=20000, k=20000, spread 1e2: Q=256 already at the 2e-10 FP64 floor
  The worker computes at Q=256 and doubles Q until the residual converges or
  stops improving (ladder 256, 512, 1024, 2048, 4096). "Stops improving"
  requires two consecutive rungs without a 10x improvement: on the
  n=10000/k=1500 reference shape the first doubling improves the residual by
  only 1.8x before later rungs drop it five orders of magnitude, so a
  single-rung stall test quits too early. The residual is a real
  a-posteriori error indicator, not a heuristic: equities must sum to the
  prize pool exactly, so the residual measures quadrature aliasing directly.
  The UI reports the achieved residual and the Q used. Total cost of the
  ladder is at most about 2x the final rung.
- **Zero stacks are filtered in the worker** (owner): a zero stack has zero
  equity by definition; the library contract requires positive stacks. Rows
  with zero stacks show equity 0 and are excluded from the ratio check.
- **Stack ratio guard** (owner): max/min over nonzero stacks must be at most
  1e9, enforced in the worker with an error message, never silently clamped.
- **Automatic stack generator is log-normal** (owner feedback, 2026-08-05):
  the original exponential-in-rank generator is a log-uniform distribution,
  which renders as the same straight line on an auto-scaled log axis at
  every spread setting. The empirical literature supports an inflected
  curve instead: Sire, "Universal statistical properties of poker
  tournaments" (arXiv:physics/0703122), validated against real tournament
  chip counts, finds a stationary stack distribution with near-exponential
  decay (and a Gumbel-distributed chip leader), and any such distribution
  has an inflection in log-stack vs percentile. The log-normal family was
  chosen over exponential because it spans equal stacks to heavy skew with
  a single parameter: the spread slider maps to sigma = spread x 2.5,
  clamped per n so the max/min ratio stays inside the 1e9 guard, sampled
  deterministically via the quantile function (Acklam probit). Log-normal
  was kept over a gamma family by owner decision (simplicity; both sit in
  the empirically supported inflected class, Sire's validated stationary
  bulk being near-exponential).
- **Charts use linear axes with a hover crosshair** (owner, 2026-08-05):
  the stack preview and the equity chart plot values linearly (the earlier
  log axis was inherited from the reference design and hid shape changes),
  with a top max-value label and a zero baseline. Hovering either chart
  shows a dashed vertical crosshair, a point marker, and a tooltip reading
  "Player i: value" for the nearest sampled player.
- **Payout presets carry real provenance where possible** (owner feedback,
  2026-08-05, researched): WSOP Main Event 2026 final table (fixed pay
  table normalized to the money at the table), classic PokerStars 180-man
  (labeled classic; current lobbies may differ), double-or-nothing (exact
  by definition), satellite equal-seats generator, plus evergreen 50/30/20
  and winner-take-all. A previously drafted 45-man structure was dropped
  because no source verified its percentages, and a Sunday Million table
  was rejected as a deal-affected single outcome rather than a structure.
- **Q is not shown in the UI** (owner feedback, 2026-08-05): the results
  summary reports elapsed time and the achieved accuracy residual only, and
  ladder progress displays as "refining accuracy" rather than a Q value.
  The ladder mechanics are unchanged.
- **k is clamped to the number of nonzero stacks** (build): with fewer live
  players than paid places, the trailing payouts are unreachable; the UI
  surfaces a notice when this happens rather than erroring.

## Verification

- `web/verify/gen_reference.c` produces `reference.json`: deterministic
  inputs (splitmix64, fixed seed) with equities from the native calibrated
  library, spanning n=2 to n=10000, all engines, edge cases (equal stacks,
  ratio exactly 1e9, k=1, k=n), plus an n=2 closed-form check.
- `web/verify/verify_node.mjs` replays every case through the WASM module at
  the same Q; the native and WASM builds share no FFT plan sizes or dispatch
  decisions (calibrated m3_pro vs generic fallback), so agreement is a strong
  end-to-end check of the numerics, not a build-artifact identity.
- The gate additionally replays the reference vectors inside real browsers
  (Chromium and WebKit) before shipping.

## Performance and the input cap

Numbers and the chosen default cap are recorded in the Performance section
below after measurement; the cap is a JS constant overridable per-visit with
the `?maxn=` URL parameter (owner). Inputs above a few hundred rows render
behind a scrolling container.

## Performance (measured)

WASM module in node 26 (V8, same engine as Chromium) on an Apple M3 Pro,
single `icm_equity` call at Q=256, best of up to 3 reps
(`web/verify/perf_node.mjs`, 2026-08-05):

| shape | n | time |
|---|---|---|
| k=n (all paid) | 9,600 | 822 ms |
| k=n (all paid) | 12,800 | 1.17 s |
| k=n (all paid) | 25,600 | 2.67 s |
| k=n/6 (typical MTT) | 9,600 | 617 ms |
| k=n/6 (typical MTT) | 19,200 | 1.42 s |
| k=15 | 64,000 | 507 ms |
| k=15 | 128,000 | 1.01 s |

Chosen limits:
- `DEFAULT_MAX_N = 10000`: at Q=256 even the worst shape (all paid) stays
  under about a second, and any-k typical fields are comfortably inside it.
  Wide stack spreads that force the Q ladder above 256 take proportionally
  longer; the UI reports each rung as progress, so the envelope is honest
  rather than silently exceeded.
- `HARD_MAX_N = 100000`: ceiling for the `?maxn=` override. Small-k queries
  (final-table equities from a huge field) stay near a second there; large-k
  queries can take minutes and the UI says so before computing.
- Time scales close to linearly in Q; the full ladder costs at most about 2x
  its final rung.

End-to-end UI measurements (Chromium via Playwright, M3 Pro, 2026-08-05):
n=10000 with k=1500 completes in about 17 s of compute (ladder-escalated for
a wide spread, residual 1.2e-12) plus 2 s of chunked table rendering, and the
page stays interactive. At the ?maxn= hard ceiling (n=100000, k=15) compute
takes 2.1 s but progressively appending 100000 table rows takes about 70 s;
the page remains scrollable and responsive during and after (sub-400 ms
event latency). No crashes or hangs at either size.
