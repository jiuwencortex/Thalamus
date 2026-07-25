#!/usr/bin/env bash
# Phase R4 — Set-level quality model.
#
# Trains a GradientBoostingRegressor on (component_set, outcome_quality) pairs
# from logged agent turns.  The model captures non-linear pairwise interactions
# between components that the GA's linear sum-of-scores fitness misses.
#
# Requires:
#   - ORACLE_DIR with context_configs.pkl (for cluster/component metadata)
#   - TURN_LOG_DIR with turns_*.jsonl files whose 'quality' fields are filled
#     (null entries are skipped automatically)
#   - ~20+ labelled turns per cluster for meaningful signal
#
# Modes (set via MODE env var):
#   train     Fit and save the model (default)
#   evaluate  Load saved model, report RMSE and R² on available turns
#   both      Train then evaluate in-sample (quick sanity check)
#
# Usage:
#   bash run_05_set_quality.sh
#   MODE=evaluate bash run_05_set_quality.sh
#   MODE=both TURN_LOG_DIR=/my/logs bash run_05_set_quality.sh
#
# Output:
#   $ORACLE_DIR/set_quality_model/model.pkl   (train)
#   $ORACLE_DIR/set_quality_model/meta.json   (train)
#   $RESULTS_DIR/r4_set_quality_train.json    (train report)
#   $RESULTS_DIR/r4_set_quality_eval.json     (evaluate report)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_config.sh"

MODE="${MODE:-train}"
MODEL_DIR="${MODEL_DIR:-$ORACLE_DIR/set_quality_model}"

_section "R4 — Set-Level Quality Model"
echo "Oracle:     $ORACLE_DIR"
echo "Turn logs:  $TURN_LOG_DIR"
echo "Model dir:  $MODEL_DIR"
echo "Mode:       $MODE"
echo ""

if [[ ! -d "$TURN_LOG_DIR" ]]; then
    echo "ERROR: TURN_LOG_DIR not found ($TURN_LOG_DIR)."
    echo "R4 requires logged agent turns with filled 'quality' fields."
    echo "Collect turns first, then re-run."
    exit 1
fi

if [[ "$MODE" == "train" || "$MODE" == "both" ]]; then
    echo "── Training ────────────────────────────────────────────"
    thalamus-research set-quality \
        --oracle-dir "$ORACLE_DIR" \
        --turn-log-dir "$TURN_LOG_DIR" \
        --model-dir "$MODEL_DIR" \
        --subcommand train \
        --out "$RESULTS_DIR/r4_set_quality_train.json"
    echo ""
    echo "Model saved to: $MODEL_DIR"
    echo "Report:         $RESULTS_DIR/r4_set_quality_train.json"
fi

if [[ "$MODE" == "evaluate" || "$MODE" == "both" ]]; then
    echo ""
    echo "── Evaluation ──────────────────────────────────────────"
    thalamus-research set-quality \
        --oracle-dir "$ORACLE_DIR" \
        --turn-log-dir "$TURN_LOG_DIR" \
        --model-dir "$MODEL_DIR" \
        --subcommand evaluate \
        --out "$RESULTS_DIR/r4_set_quality_eval.json"
    echo ""
    echo "Report: $RESULTS_DIR/r4_set_quality_eval.json"
fi

echo ""
echo "── Production integration ──────────────────────────────"
echo "To use the XGB model as the GA fitness function, re-run the oracle build:"
echo ""
echo "  thalamus-oracle evolve \\"
echo "    --oracle-dir \"$ORACLE_DIR\" \\"
echo "    --fitness-model xgb \\"
echo "    --fitness-model-dir \"$MODEL_DIR\""
echo ""
echo "This rebuilds context_configs.json with set-level quality fitness (R4)."
echo "Compare the new configs against the marginal baseline in:"
echo "  $RESULTS_DIR/r4_set_quality_eval.json"
