#!/bin/bash
set -euo pipefail

# =============================================================================
# ICM Repository Cleanup Script
# Implements STATUS_ORIENTATION.md §7 declutter plan
# DRAFT ONLY — designed to be reviewed (--dry-run) before a human executes it.
# NEVER run this script unattended. NEVER git add/commit/rm inside this script.
# =============================================================================

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# ── Sanity: must be at repo root ────────────────────────────────
if [[ ! -f "Makefile" ]] || [[ ! -f "src/icm.c" ]]; then
  echo "ERROR: Not at repo root (missing Makefile or src/icm.c)" >&2
  exit 1
fi

echo "=== ICM Cleanup Script ==="
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY RUN MODE] — no files will be deleted or moved"
fi
echo ""

# ── Helper: is a path tracked by git? ───────────────────────────
# Returns 0 (success) if the path is UNTRACKED or does not exist (safe to rm/mv).
# Returns 1 if the path IS tracked by git (SKIP).
is_untracked() {
  local f="$1"
  if [[ ! -e "$f" ]] && [[ ! -L "$f" ]]; then
    return 0  # doesn't exist — safe (rm -f is a no-op, mv would fail anyway)
  fi
  if git ls-files --error-unmatch "$f" &>/dev/null; then
    return 1  # TRACKED — DO NOT TOUCH
  fi
  return 0  # exists on disk but not in git index — untracked, safe
}

# ── Safe delete (only untracked files) ───────────────────────────
safe_delete() {
  local f="$1"
  if [[ ! -e "$f" ]] && [[ ! -L "$f" ]]; then
    return 0  # already gone — idempotent
  fi
  if ! is_untracked "$f"; then
    echo "WARNING: Skipping delete of '$f' — TRACKED in git"
    return 0
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY] Would delete: $f"
  else
    echo "DELETE: $f"
    rm -f "$f"
  fi
}

# ── Safe recursive delete (only untracked directory) ─────────────
safe_delete_dir() {
  local d="$1"
  if [[ ! -d "$d" ]]; then
    return 0  # already gone
  fi
  if ! is_untracked "$d"; then
    echo "WARNING: Skipping delete of '$d' — TRACKED in git"
    return 0
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY] Would delete dir: $d"
  else
    echo "DELETE DIR: $d"
    rm -rf "$d"
  fi
}

# ── Safe move (only untracked files; tracked => warn & skip) ─────
safe_move() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$src" ]] && [[ ! -L "$src" ]]; then
    echo "NOTE: Skipping move of '$src' — file does not exist"
    return 0
  fi
  if ! is_untracked "$src"; then
    echo "WARNING: Skipping move of '$src' — TRACKED in git (use 'git mv' manually if needed)"
    return 0
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY] Would move: $src -> $dst"
  else
    echo "MOVE: $src -> $dst"
    mv "$src" "$dst"
  fi
}

# =============================================================================
# STEP 1 — Create results/ directory
# =============================================================================
echo "── Step 1: Ensure results/ directory exists ──"
if [[ -d results ]]; then
  echo "results/ already exists"
else
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY] Would create: results/"
  else
    mkdir -p results
    echo "CREATED: results/"
  fi
fi
echo ""

# =============================================================================
# STEP 2 — Move KEEP reference artifacts to results/
# Per STATUS_ORIENTATION.md §7, these 6 files are reference artifacts to keep.
# All 6 are currently git-tracked at repo root, so the safety guard will skip
# the mv.  A human should run `git mv` for these after review.
# =============================================================================
echo "── Step 2: Move §7 KEEP artifacts to results/ ──"
KEEP_FILES=(
  "b200_final_benchmarks.txt"
  "b200_final_cufft.txt"
  "gpu_heatmap_final.csv"
  "results_b200_heatmap_final.csv"
  "b200_B_validation.csv"
  "b200_runtime_vs_n.csv"
)

MOVED_COUNT=0
SKIPPED_COUNT=0
MISSING_COUNT=0
for f in "${KEEP_FILES[@]}"; do
  if [[ ! -e "$f" ]]; then
    echo "NOTE: KEEP file '$f' does not exist — skipping"
    ((MISSING_COUNT++)) || true
    continue
  fi
  if ! is_untracked "$f"; then
    echo "WARNING: '$f' is git-tracked — skipping mv (use 'git mv $f results/' manually)"
    ((SKIPPED_COUNT++)) || true
  else
    safe_move "$f" "results/$f"
    ((MOVED_COUNT++)) || true
  fi
done
echo "  → $MOVED_COUNT moved, $SKIPPED_COUNT skipped (git-tracked), $MISSING_COUNT missing"
echo ""

# =============================================================================
# STEP 3 — Delete leaked compiled binaries at repo root
# =============================================================================
echo "── Step 3: Delete leaked compiled binaries ──"
LEAKED_BINARIES=(
  "test_cpu_cost_model"
  "test_cpu_cost_model_zen4"
)
for f in "${LEAKED_BINARIES[@]}"; do
  safe_delete "$f"
done
echo ""

# =============================================================================
# STEP 4 — Delete results_b200_optimized/ directory
# =============================================================================
echo "── Step 4: Delete results_b200_optimized/ directory ──"
safe_delete_dir "results_b200_optimized"
echo ""

# =============================================================================
# STEP 5 — Delete untracked root-level CSV, LOG, TXT output artifacts
# Per §7: these are throwaway benchmark outputs, all regeneratable.
# git-tracked root CSV/LOG/TXT files are PROTECTED by the safety guard.
# The KEEP files from Step 2 are also git-tracked, so the guard would
# protect them here too — but we've already attempted to move them above.
# =============================================================================
echo "── Step 5: Delete untracked root-level output artifacts ──"

# Build lists explicitly so we don't accidentally expand into results/
ROOT_CSV_FILES=( *.csv )
ROOT_LOG_FILES=( *.log )
ROOT_TXT_FILES=( *.txt )

# These are the §7 KEEP artifacts — double-protect them (they're also
# git-tracked so the guard would catch them, but be explicit).
KEEP_SET=(
  "b200_final_benchmarks.txt"
  "b200_final_cufft.txt"
  "gpu_heatmap_final.csv"
  "results_b200_heatmap_final.csv"
  "b200_B_validation.csv"
  "b200_runtime_vs_n.csv"
)

is_keeper() {
  local needle="$1"
  for k in "${KEEP_SET[@]}"; do
    if [[ "$needle" == "$k" ]]; then
      return 0
    fi
  done
  return 1
}

DEL_COUNT=0
SKIP_COUNT=0

for f in "${ROOT_CSV_FILES[@]}" "${ROOT_LOG_FILES[@]}" "${ROOT_TXT_FILES[@]}"; do
  # Skip if the glob didn't match anything (literal "*.csv" string)
  if [[ "$f" == "*.csv" || "$f" == "*.log" || "$f" == "*.txt" ]]; then
    continue
  fi
  # Skip if it's in the KEEP set
  if is_keeper "$f"; then
    continue
  fi
  # Skip if it doesn't actually exist (belt and suspenders)
  if [[ ! -e "$f" ]]; then
    continue
  fi

  if ! is_untracked "$f"; then
    echo "PROTECTED: '$f' is git-tracked — skipping"
    ((SKIP_COUNT++)) || true
  else
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[DRY] Would delete: $f"
    else
      echo "DELETE: $f"
      rm -f "$f"
    fi
    ((DEL_COUNT++)) || true
  fi
done
echo "  → $DEL_COUNT untracked files deleted, $SKIP_COUNT git-tracked files protected"
echo ""

# =============================================================================
# STEP 6 — Delete 12 one-off tools/ files per §7 DELETE list
# =============================================================================
echo "── Step 6: Delete 12 one-off experimental tools/ files ──"
TOOLS_DELETE=(
  "tools/bench_batch.cu"
  "tools/bench_batch_fused.cu"
  "tools/bench_kernels.cu"
  "tools/bench_level_pipeline.cu"
  "tools/calib_validate.c"
  "tools/calibrate_pipeline.c"
  "tools/contour_1s_gpu.cu"
  "tools/layout_test.c"
  "tools/measure_cache_overhead.c"
  "tools/merge_analysis.c"
  "tools/merge_analysis2.c"
  "tools/plan_switch_test.c"
)

TOOLS_DEL_COUNT=0
TOOLS_MISSING=0
for f in "${TOOLS_DELETE[@]}"; do
  if [[ ! -e "$f" ]]; then
    echo "NOTE: '$f' does not exist — skipping"
    ((TOOLS_MISSING++)) || true
    continue
  fi
  safe_delete "$f"
  if [[ $? -eq 0 ]]; then
    ((TOOLS_DEL_COUNT++)) || true
  fi
done
echo "  → $TOOLS_DEL_COUNT deleted, $TOOLS_MISSING already missing"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "═══════════════════════════════════════════"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "DRY RUN COMPLETE.  Review the [DRY] lines above."
  echo "To execute for real, run:"
  echo "  bash tools/cleanup_repo.sh"
else
  echo "CLEANUP COMPLETE."
fi
echo ""
echo "MANUAL POST-CLEANUP (do NOT run in this script):"
echo "  1. Move git-tracked KEEP files:"
for f in "${KEEP_FILES[@]}"; do
  if [[ -e "$f" ]]; then
    echo "     git mv '$f' results/"
  fi
done
echo "  2. Stage everything:  git add -A"
echo "  3. Review diff:       git diff --staged"
echo "  4. Commit:            git commit -m 'Declutter: remove throwaway artifacts'"
echo "═══════════════════════════════════════════"
