# Zen4 CPU-side QA report — 2026-07-27

Box: user's own dedicated Zen4 instance (fresh redeployment). Fresh
from-scratch deploy this session: AOCL-FFTW built from source with the
documented AM5 flag set, `devices/zen4/fftw_wisdom.dat` ported byte-identical
(not regenerated), `bench_grid verify` passes 36/0.

## RAM bandwidth ceiling (root cause, already diagnosed — not re-derived here)

This box's 4 DIMMs (2 per channel, 2DPC) run at a **Configured Memory Speed
of 3600 MT/s** vs their 5600 MT/s rating — AMD's documented AM5 electrical
limit for 2DPC configurations, confirmed not fixable at the OS/BIOS level
(bare metal, `performance` governor already active). A 1DPC replacement
wasn't available. **Per explicit user decision, this box is now the
standing Zen4 reference**, not a temporary anomaly — the numbers below are
real and permanent for this hardware, not a regression to chase.

`devices/zen4/fft_config.h`'s crossover table was recalibrated for this box
this session (committed `eb40e2d`/`2aff562`) and is unaffected by anything
below.

## Old (pre-existing, dated 2026-07-22/24) vs New (this session) — headline numbers

### 1-second threshold, serial (`bench_grid threshold` / contour k=n)

| Metric | Old (committed) | New (this box) |
|---|---|---|
| Serial 1s threshold (k=n) | n≈29,000 (interpolated from 2 grid points: n=16384@491ms, n=32768@1140ms) | **n=17,984** (real binary search, `bench_grid threshold`) |
| n=16,384, k=16384 | 491ms (interpolated) | 809ms (measured directly) |
| n=32,768, k=32768 | 1140ms (interpolated) | 1955ms (measured directly) |

Note the old number was itself only an interpolation between two grid
points, never a real binary search — the new number is the first real
`bench_grid threshold` binary search this project has run for Zen4.

### Serial contour (Q=256, 1s boundary per k)

| k | Old n_max (ms) | New n_max (ms) | New/Old n_max ratio |
|---|---|---|---|
| 2 | 402,833 (976) | 402,833 (959) | 1.00 |
| 10 | 351,571 (993) | 341,805 (990) | 0.97 |
| 100 | 128,980 (973) | 117,264 (972) | 0.91 |
| 200 | 64,601 (924) | 45,085 (933) | 0.70 |
| 1000 | 48,468 (990) | 31,625 (971) | 0.65 |
| 10000 | 30,625 (925) | 17,500 (983) | 0.57 |
| 13000 | 28,843 (931) | 16,250 (884) | 0.56 |
| 17000 | 28,687 (966) | 17,000 (1064, **floor**) | 0.59 |
| 30000 | 30,000 (1141, floor) | *(sweep floored out already by k=17,000)* | — |

**Important nuance — the RAM ceiling's effect is NOT a flat percentage.**
At k=2-10 (pure linear/schoolbook engine, compute-bound not
memory-bandwidth-bound) the new box is essentially IDENTICAL to the old
one (ratio ~0.97-1.00). The gap only opens up once k crosses into
hybrid/FFT territory (k>=200), reaching ~0.56-0.65x by k=1000-17000. This
matches the full-grid data below: linear-engine timings are nearly
unaffected by the RAM ceiling, hybrid/FFT timings are cut by 40-45% at
large k. **A single flat "divide everything by ~1.6" extrapolation would
be wrong** — it overcorrects the linear-dominated region and roughly
matches the hybrid-dominated region. If a 1DPC-equivalent number is
needed for commentary, scale by regime: ~1.0x for k<100 (linear-bound),
~1.6-1.8x for k>=1000 (hybrid/FFT-bound) — ROUGH ESTIMATE, not a
measurement, and only defensible because the old data shows the same
qualitative split.

### Parallel contour (Q=256, OMP_NUM_THREADS=16, 1s boundary per k)

| k | Old n_max (ms) | New n_max (ms) | Note |
|---|---|---|---|
| 2 | 1,513,672 (981) | 1,513,672 (975) | identical |
| 100 | 1,000,050 (981) | 906,259 (951) | 9% lower |
| 1000 | 131,593 (958) | 137,812 (997) | New slightly HIGHER |
| 10000 | 99,062 (959, **B=64**) | 94,375 (1000, **B=32**) | different B selected |
| 13000 | 95,468 (989, B=64) | 86,937 (978, B=32) | different B selected |
| 19000 | 89,656 (972, B=64) | 83,421 (989, B=32) | different B selected |
| 22000 | 83,875 (970, B=64) | 81,812 (980, B=32) | different B selected |
| 26000 | 84,500 (995, B=64) | 77,187 (988, B=32) | different B selected |
| 70000 | 70,000 (1683, floor) | 70,000 (1718, floor) | floor matches almost exactly |

**Parallel numbers are surprisingly close to the old box overall** —
much closer than the serial numbers. Plausible explanation (not verified
further): with 16 threads all contending for the same memory bus,
aggregate achievable bandwidth may already have been the bottleneck on
the OLD box too, making the marginal cost of this box's lower per-channel
speed less pronounced once 16 threads are already saturating the bus.
Not chased further this session.

**New finding, flagged not fixed**: the parallel dispatch selects a
DIFFERENT block size B at several k values (B=32 here vs B=64 on the old
box, at k=10000/13000/19000/22000/26000). This is the same class of
question the crossover table already needed this session —
`devices/zen4/fft_config.h`'s `bselect_*` table (1944 points) was carried
over UNCHANGED from the prior box and has not been re-verified against
this box's real bandwidth profile. Whether B=32 is actually optimal here
or whether the table itself needs the same kind of re-verification the
crossover table got is an open question — **not investigated this
session**, flagging for a future pass rather than triggering another
full recalibration cycle right now.

## Full performance grid (`bench_grid`, no args)

Same regime split as the contour data: at n=65,536 (largest grid point),
linear engine is ~118ms new vs ~125ms old (statistically indistinguishable
given single-measurement noise), while hybrid is 4110-5110ms new vs
2620-3300ms old — roughly 55-65% slower, consistent with the
memory-bandwidth-bound hybrid/FFT path being what actually pays for the
3600 vs 5600 MT/s gap. Full files: `results/bench_grid_zen4_serial_20260727.txt`,
`results/bench_grid_zen4_parallel_20260727.txt`.

## Stale documentation finding (independent of the RAM ceiling)

`RESULTS.md`'s "Zen 4 bandwidth constants, root cause diagnosed and fixed,
pending re-verification" section (around line 351) is **already stale**.
It says the fix (sane `L2_BW_GBS`/`L3_BW_GBS`/`DRAM_BW_GBS` values) was
"not yet re-verified with a fresh calibration run on Zen4 hardware." In
fact commit `18bf1c3` ("Zen4: fresh AOCL-FFTW PATIENT recalibration on new
Zen4 box", 2026-07-22) already did this re-verification and committed sane
values (`L2_BW_GBS=131.5`, etc., currently in `devices/zen4/fft_config.h`)
— the paragraph in `RESULTS.md` was simply never updated afterward. This
is a documentation-only fix (delete/update that paragraph), not a code or
data issue.

## What this report does NOT do

Per the board's binding law for this node: this report does not edit
`RESULTS.md`, does not commit anything, and does not decide what the
paper's final numbers should be. The supervisor reviews this report and
the fresh data files (`results/bench_grid_zen4_{serial,parallel}_20260727.txt`,
`results/contour_zen4_{serial,parallel}_q256_20260727.csv`) before any of
it is folded into `RESULTS.md`/the paper.
