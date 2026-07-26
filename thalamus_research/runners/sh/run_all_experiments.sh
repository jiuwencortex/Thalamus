#!/usr/bin/env bash
# Full THALAMUS research pipeline — R1 through R5.
#
# Stages that can run immediately (no logged turns required):
#   R1  Baseline evaluation
#   R2  Ablation study
#   R3a Cross-path co-inclusion analysis
#   R3b ε* estimation (bandit exploration rate)
#
# Stages that require logged agent turns (run after collecting data):
#   R3b Convergence analysis  (run with: MODE=convergence bash run_04_bandit.sh)
#   R4  Set-level quality model training
#   R5  Meta-learning KB extraction
#
# Usage:
#   bash run_all_experiments.sh
#   QUERIES_FILE=/my/queries.json ORACLE_DIR=/my/oracle bash run_all_experiments.sh
#
# To skip a stage:
#   SKIP_R2=1 bash run_all_experiments.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_config.sh"

SKIP_R1="${SKIP_R1:-0}"
SKIP_R2="${SKIP_R2:-0}"
SKIP_R3A="${SKIP_R3A:-0}"
SKIP_R3B="${SKIP_R3B:-0}"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   THALAMUS Research Pipeline — R1 through R3b        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Oracle:      $ORACLE_DIR"
echo "Results dir: $RESULTS_DIR"
echo ""
echo "Phases requiring turn logs (run separately after data collection):"
echo "  R3b convergence → MODE=convergence bash run_04_bandit.sh"
echo "  R4 set quality  → bash run_05_set_quality.sh"
echo "  R5 meta-learning → bash run_06_meta_learning.sh"
echo ""

# ── R1: Baseline evaluation ───────────────────────────────────────────────────
if [[ "$SKIP_R1" != "1" ]]; then
    bash "$SCRIPT_DIR/run_01_baselines.sh"
else
    echo "[SKIP] R1 baselines"
fi

echo ""

# ── R2: Ablation study ────────────────────────────────────────────────────────
if [[ "$SKIP_R2" != "1" ]]; then
    bash "$SCRIPT_DIR/run_02_ablation.sh"
else
    echo "[SKIP] R2 ablation"
fi

echo ""

# ── R3a: Cross-path analysis ──────────────────────────────────────────────────
if [[ "$SKIP_R3A" != "1" ]]; then
    bash "$SCRIPT_DIR/run_03_cross_path.sh"
else
    echo "[SKIP] R3a cross-path"
fi

echo ""

# ── R3b: Bandit exploration rate ──────────────────────────────────────────────
if [[ "$SKIP_R3B" != "1" ]]; then
    MODE=estimate bash "$SCRIPT_DIR/run_04_bandit.sh"
else
    echo "[SKIP] R3b bandit"
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Pre-data phases complete.                           ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Results in: $RESULTS_DIR"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Next steps (after collecting agent turn logs):       ║"
echo "║    1. Set exploration rate from R3b epsilon report    ║"
echo "║    2. MODE=convergence bash run_04_bandit.sh          ║"
echo "║    3. bash run_05_set_quality.sh  (needs ~20+ turns)  ║"
echo "║    4. bash run_06_meta_learning.sh  (multi-deploy)    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
