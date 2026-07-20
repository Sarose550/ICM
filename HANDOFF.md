# Handoff — ICM repo polish + Zen4 refresh + subset-pruning bug

Written so a fresh session can pick this up with zero prior context.
Live tracking board: `SPRINT_M3PRO_DOCS_DAG.md` (same directory) — has full
detail on every item below plus exact next steps. Read that too.

## What this session was doing

Started as a GitHub-facing README audit/polish (badges, LaTeX rendering,
VkFFT cleanup, better ICM explanation for newcomers), expanded into a
hardware refresh (new Zen4 rental to replace stale benchmark data, M3 Pro
replacing an M3 Max that's no longer available), and then hit a real
correctness bug in `icm_equity_subset()` that consumed most of the session.

## Current repo state (uncommitted!)

```
git status --short
 M README.md
 M bench/bench.c
 M src/icm.c
?? SPRINT_M3PRO_DOCS_DAG.md   (ephemeral board, delete when everything's done)
?? HANDOFF.md                 (this file)
```

**Nothing has been committed.** Do not commit without asking the user first
— that's a standing rule, not specific to this session.

## The three real code changes

1. **`bench/bench.c`** — added `fmt_ms_3sf()`, a 3-significant-figure
   millisecond formatter (fixed-point, never scientific notation), applied
   to the performance grid and `profile` subcommand output. Previously
   everything was `%.0f` (integer ms, 1-4 sig figs depending on magnitude).
   Done, tested, no issues.

2. **`src/icm.c`** — fixed a real correctness bug in `icm_equity_subset()`'s
   target-locality tree pruning. Full story: the pruning feature reordered
   players (moved targets to the front) so target-containing tree "blocks"
   would cluster contiguously and cold blocks could be skipped as one
   chunk. That reordering broke the hybrid engine's global descending-
   stack-size invariant, which its per-block division step needs for
   numerical stability. The bug was invisible on Apple Silicon (ARM/vDSP)
   but real and reproducible on Zen4 (x86/AVX-512) — a platform-dependent
   FP-rounding edge case, present on both platforms mathematically but only
   visible on one. **Fixed** by redesigning pruning to use a per-block
   hot/cold bitmask (propagated bottom-up through the tree) instead of
   reordering — `sort_perm` is now never touched by the subset path, it's
   the exact same array `icm_equity()` always uses. **Verified fixed on
   the actual Zen4 hardware**: `bench_grid verify` passes, a 120-trial
   randomized stress test passes bit-exact (0.0 diff). This was NOT a
   quick fix — three wrong hypotheses were tried and disproven first
   (index bug in clustering — real bug, fixed, but not the cause; FFTW
   buffer corruption — false; per-block-only monotonicity — real artifact,
   fixed, still not the cause). Full debugging trail is in the
   conversation transcript if the mechanism ever needs re-explaining.

3. **`README.md`** — full content polish. CI badge fixed (was pointing at
   `samrosenstrauch/icm`, should be `Sarose550/ICM` — this is the actual
   GitHub remote). All VkFFT mentions removed (built, tested, measured
   1-2% slower than cuFFT-only on B200, no benefit, effectively abandoned
   per git history). "How It Works" rewritten without LaTeX (GitHub renders
   `$...$` inconsistently next to punctuation) as a narrative: naive n!
   enumeration → industry-standard bitmask DP (**O(n·2ⁿ)**, not O(k·2ⁿ) —
   this was a real error in an earlier draft, now corrected and cited) →
   Monte Carlo exponential-clock sampling (cites Tysen Streib's original
   TwoPlusTwo thread) → this repo's exact generating-function quadrature
   (shown as literally making the Monte Carlo method exact). New Accuracy
   section explaining the V1/V2 closed-form ground-truth test cases
   (`src/icm.c`'s `v1_exact()`/`v2_exact()`) with real convergence numbers
   pulled from `results/accuracy_m3max_20260718.csv`. Performance tables
   are currently **stubbed** — "M3 Pro (recalibrating)" / "Zen4 pending
   refresh" — because neither platform's numbers are finalized yet (see
   Pending below).

## Pending work — see the board for full detail, summary here

1. **Zen4 calibration was never actually validated.** We've been running
   on `devices/zen4/`'s *old* calibration data (from a dead prior box) on
   this brand-new Zen4 instance the whole session, without ever running
   `./bench_grid crossover` to check it's still giving correct dispatch
   decisions. Do that first (cheap, few minutes); only recalibrate from
   scratch (`./calibrate`, 10-30 min) if crossover disagrees.
2. **Run the actual Zen4 serial + parallel 3-sig-fig performance grid** —
   this was the original point of getting on Zen4, before the subset bug
   ate the session. Blocked on #1.
3. **M3 Pro validation never finished either.** A fresh M3 Pro calibration
   attempt was killed at the 30-minute mark by unrelated CPU load on the
   machine and never retried. We've been running on reused `devices/m3_max`
   data, also never crossover-validated. Cheap next step: run
   `./bench_grid crossover` locally (a few minutes, no contention risk).
   Only do a fresh `./calibrate` if that disagrees. The user explicitly
   said not to throw away the Mac support effort given how little is
   actually left — this crossover check is that "little."
4. **Once #1-#3 settle:** fill in the real numbers in README.md's stubbed
   performance tables, and propagate to `CLAUDE.md` (also still has VkFFT
   mentions to remove — that cleanup was scoped to README.md only this
   session, not CLAUDE.md), `RESULTS.md`, and `OPTIMIZATION_GUIDE.md` (all
   three were fully deferred this session, untouched).
5. **Commit `devices/*/fftw_wisdom.dat` to git** — user approved this
   earlier in the session (currently gitignored, not in the public repo).
   Wait until #1/#3 settle so we don't commit wisdom files about to be
   replaced by a fresh calibration.
6. **"Deslopify" pass — user request, not started.** A DeepSeek-managed
   pass over the whole codebase: strip stale/deprecated/overly-long
   comments, comments referencing things no longer relevant, general
   comment cleanup. Review code organization for actual smells — refactor
   only where it doesn't cost efficiency. Zero compiler warnings
   requirement. Reason given: repo is going open source and is a
   reflection on the user professionally. Do this as its own wave, after
   the numeric/doc work above lands, so it isn't fighting in-flight
   changes on the same files.
7. **Minor:** the Zen4 box (`/root/icm/`) has scratch debug files sitting
   in it (`isolate_bug.c`, `stress_subset.c`, `debug_levels.c`,
   `shuffle_test.c`, and their compiled binaries, plus `dbg.log`/
   `shadow.log`). Harmless — dedicated box, not in the git repo — but
   worth clearing once validation work there is fully done.

## Zen4 machine access

Credentials are in memory `zen4-server-credentials` (auto-memory system,
not in this repo). Fresh Cherry Servers instance, IP `84.32.220.210`,
AMD Ryzen 9 7950X — **16 physical cores, 32 logical via SMT** (use
`OMP_NUM_THREADS=16` for parallel runs, not 32 — this workload is FPU/
vector-port bound, SMT siblings don't add real throughput, and 16 matches
prior Zen4 benchmark methodology in RESULTS.md). Repo is rsync'd to
`/root/icm` there (not a git clone — matches this local working tree as of
the last deploy, keep it in sync with `rsync`/`scp` if you make more
changes locally that need testing there).

## Notes on how this session worked

Followed the `supervisor-dag` skill: DeepSeek Deck workers did the file
edits and static-analysis-heavy investigation (folder `e308ee`), supervisor
did anything requiring the live Zen4 hardware (ssh is network-sandboxed for
Deck workers, so that leg can never be delegated) plus final judgment
calls. Several DeepSeek dispatches hit the 50-turn cap without finishing
and had to be resumed or re-dispatched fresh — if you see worker ids in the
transcript (`057d12`, `06864b`, etc.) those are DeepSeek Deck workers, not
native subagents; use `deck result <id>` to pull their compact summaries if
you need the history, not `deck log` (full transcript, expensive).
