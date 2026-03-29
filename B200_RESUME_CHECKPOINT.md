# B200 Throughput Resume Checkpoint

## Current Status

- Phases completed:
  - phase1-parity-matrix
  - phase1-verify-extensions
  - phase2-calibrate-rewrite
  - phase2-config-regen
  - phase3-planner-refit
- Current in-progress phase:
  - phase4-fused-kernels

## Recently Resolved

- Root-cause for `B=224` timeout at `k=n` was identified and fixed:
  - Overflow in planner joint-cost path:
    - `best_fft_config_joint_gpu`: `int corr_input_wrap = cm * p_eff`
    - For the failing case: `cm=37829`, `p_eff=57344`, product `2,169,266,176` > `INT_MAX`
    - Overflow made correction cost incorrect and allowed huge wrap (`cwm~37829`), which caused pathological wrap-correction work.
  - Fix applied:
    - `corr_input_wrap` promoted to `double` multiplication.
    - Added wrap-serial penalty modeling (`wrap_serial_penalty_gpu`) so top-level low-parent-count wrap work is costed more realistically.
  - Post-fix validation:
    - `ICM_GPU_FORCE_B=224 timeout 20s ./bench_gpu_fused bench 65536 65536 1 16` now completes (`RC=0`).
    - Per-level debug shows ell=8 now uses `fft_n=131072`, `bwm=0`, `cwm=0` (no giant wrap).
    - `./bench_gpu_fused verify` passes after fix.
  - Planner validation anchors completed:
    - `Q=64`: all fast-anchor cases matched planner-selected `B`.
    - `Q=128`: anchor sweep initially showed one mismatch (`65536,k=65536`), but 5-rep targeted check confirmed `B=128` remains best over `B=160`.
  - Artifacts copied locally to:
    - `results_b200_instance_33698132/checkpoint_2026-03-28/`

- Planner/model mismatch for large `n` was identified and fixed:
  - Root cause:
    - `estimate_candidate_cost()` priced tree work using `n_real[ell]`, but runtime executes padded power-of-two tree width `nn[ell]`.
    - This under-priced candidate `B` values with non-power-of-two block counts and biased selection toward bad `B` (often `128`).
  - Fix applied:
    - Changed tree-cost weighting from `n_real[ell]` to `nn[ell]` in both schoolbook and FFT branches.
    - Expanded B candidate grid to finer spacing:
      - now includes `... 96,112,128,144,160,176,192,208,...` through `4096`.
  - Validation:
    - `./bench_gpu_fused verify` still passes.
    - Auto-selected `B` now adapts (not stuck at `128`):
      - `n=k=1179648` -> `B=144`, `~761 ms`
      - `n=k=1310720` -> `B=160`, `~867 ms`
      - `n=k=1441792` -> `B=176`, `~937 ms`
      - `n=k=1572864` -> `~1096 ms`
    - Current measured under-1s frontier is now around `n=1,441,792` (Q=256), up from `~1,048,576`.

## Immediate Next Steps (in order)

1. Bring up a B200 instance and sync latest local code.
2. Rebuild `bench_gpu_fused` and `validate_planner_gpu`.
3. Finish phase4 fused-kernel work:
   - replace tier-2 fallbacks (`run_build_level_fused` / `run_prop_level_fused`) with real fused kernels.
4. Add/finish tier-ablation runner for direct schoolbook vs fused vs cuFFT comparison per level and conv size, then hard-wire measured crossovers.
5. Replace wrap-serial heuristic with measured wrap-kernel calibration data (remove ad-hoc scaling).
6. Keep using bounded fast-loop runs for iteration; only remove caps for checkpoint sweeps.

## Optimization Coverage Check

Yes, the current plan explicitly includes full optimization exploration, including batched FFT decisions:

- Tier assignment by calibrated costs (`gpu_assign_tiers`): schoolbook vs fused vs batched cuFFT.
- Tier ablation and crossover enforcement (phase4-tier-ablation).
- Graph + memory strategy completion A/B/C/D (phase5).
- Joint frontier optimization over `(B, M, T)` (phase7).
- Final full heatmap/contours only after freeze (phase8).

## Iteration Guardrails (to avoid stalls)

- Keep fast-loop runs hard-bounded with `timeout` during debugging.
- Ensure only one heavy validator/benchmark process runs at a time.
- Do not run final contour/heatmap until optimization freeze.
