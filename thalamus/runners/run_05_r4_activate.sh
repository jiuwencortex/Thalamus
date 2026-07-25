#!/usr/bin/env bash
# R4 — Activate set-level quality fitness in the production oracle.
#
# Run AFTER collecting ~500–1000 agent turns with filled 'quality' fields.
#
# What this script does:
#   1. Trains the GradientBoostingRegressor on (component_set, quality) pairs
#      from logged turns (via thalamus-research set-quality).
#   2. Rebuilds context_configs.json using the XGB model as the GA fitness
#      function instead of the marginal sum-of-scores heuristic.
#
# Prerequisites:
#   - thalamus-research installed: uv pip install -e ./thalamus_research
#   - ORACLE_DIR with context_configs.pkl  (run run_02_oracle.sh first)
#   - TURN_LOG_DIR with turns_*.jsonl files whose 'quality' fields are filled
#     (see TurnLogger.update_outcome — task_completed, follow_up_correction, etc.)
#
# Required env vars:
#   ORACLE_DIR   — oracle directory (default: ~/.jiuwenswarm/agent/workspace/oracle)
#   TURN_LOG_DIR — directory with JSONL turn logs (default: ORACLE_DIR/online_logs)
#
# Optional env vars:
#   MODEL_DIR    — where to write model.pkl (default: ORACLE_DIR/set_quality_model)
#   MIN_TURNS    — minimum labelled turns needed to train (default: 50)
#   N_CLUSTERS   — cluster count for oracle rebuild (default: from context_configs.pkl)
#
# Usage:
#   ORACLE_DIR=/my/oracle TURN_LOG_DIR=/my/logs bash run_05_r4_activate.sh

set -euo pipefail

ORACLE_DIR="${ORACLE_DIR:-~/.jiuwenswarm/agent/workspace/oracle}"
TURN_LOG_DIR="${TURN_LOG_DIR:-$ORACLE_DIR/online_logs}"
MODEL_DIR="${MODEL_DIR:-$ORACLE_DIR/set_quality_model}"
MIN_TURNS="${MIN_TURNS:-50}"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   R4 — Set-Level Quality Fitness Activation          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Oracle:     $ORACLE_DIR"
echo "Turn logs:  $TURN_LOG_DIR"
echo "Model dir:  $MODEL_DIR"
echo ""

# ── Guard: require oracle dir ───────────────────────────────────────────────
if [[ ! -d "$ORACLE_DIR" ]]; then
    echo "ERROR: ORACLE_DIR not found: $ORACLE_DIR" >&2
    echo "Run run_02_oracle.sh first to build the oracle." >&2
    exit 1
fi

# ── Guard: require turn logs ─────────────────────────────────────────────────
if [[ ! -d "$TURN_LOG_DIR" ]]; then
    echo "ERROR: TURN_LOG_DIR not found: $TURN_LOG_DIR" >&2
    echo "Collect agent turn logs with filled 'quality' fields first." >&2
    echo "See: TurnLogger.update_outcome(turn_id, task_completed=..., ...)" >&2
    exit 1
fi

n_turns=$(ls "$TURN_LOG_DIR"/turns_*.jsonl 2>/dev/null | xargs grep -l '"quality"' 2>/dev/null | wc -l || echo 0)
echo "Turn log files with quality data found: $n_turns"
echo ""

# ── Step 1: Train XGB set-quality model ──────────────────────────────────────
echo "══ Step 1: Training set-quality model (XGB) ══════════════"
thalamus-research set-quality \
    --oracle-dir   "$ORACLE_DIR" \
    --turn-log-dir "$TURN_LOG_DIR" \
    --model-dir    "$MODEL_DIR" \
    --subcommand   train \
    --min-turns    "$MIN_TURNS" \
    --out          "$ORACLE_DIR/research_results/r4_set_quality_train.json"

echo ""
echo "Model saved: $MODEL_DIR/model.pkl"
echo ""

# ── Step 2: Evaluate model on held-out turns ──────────────────────────────────
echo "══ Step 2: Evaluating set-quality model ══════════════════"
thalamus-research set-quality \
    --oracle-dir   "$ORACLE_DIR" \
    --turn-log-dir "$TURN_LOG_DIR" \
    --model-dir    "$MODEL_DIR" \
    --subcommand   evaluate \
    --out          "$ORACLE_DIR/research_results/r4_set_quality_eval.json"

echo ""
echo "Eval report: $ORACLE_DIR/research_results/r4_set_quality_eval.json"
echo ""

# ── Step 3: Rebuild oracle with XGB fitness ───────────────────────────────────
echo "══ Step 3: Rebuilding oracle with XGB fitness ════════════"
thalamus-oracle evolve \
    --oracle-dir       "$ORACLE_DIR" \
    --fitness-model    xgb \
    --fitness-model-dir "$MODEL_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   R4 activation complete.                            ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  context_configs.json rebuilt with XGB fitness.      ║"
echo "║  Eval report: $ORACLE_DIR/research_results/r4_set_quality_eval.json"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  To revert to marginal fitness:                      ║"
echo "║    thalamus-oracle evolve --oracle-dir $ORACLE_DIR   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
