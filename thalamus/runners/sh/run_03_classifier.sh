#!/usr/bin/env bash
# Phase 4: Train the component inclusion classifier from logged agent turns.
#
# What this does:
#   - Reads JSONL turn logs from ORACLE_DIR/online_logs/
#   - Trains independent logistic regression models (one per component)
#   - Evaluates precision/recall/F1/AUC per component on a held-out val split
#   - Writes classifier.pkl + classifier_registry.json to ORACLE_DIR
#
# Requires at least MIN_TURNS logged turns.
# Run after the agent has accumulated real interaction data (post-oracle deploy).

set -euo pipefail

# ── Configure ────────────────────────────────────────────────────────────────
ORACLE_DIR="${ORACLE_DIR:-~/.jiuwenswarm/agent/workspace/oracle}"
LOG_DIR="${LOG_DIR:-$ORACLE_DIR/online_logs}"
MIN_TURNS="${MIN_TURNS:-10}"
MAX_WEEKS="${MAX_WEEKS:-8}"
C_REGULARIZATION="${C_REGULARIZATION:-1.0}"   # inverse L2 strength

echo "=== Phase 4: Classifier Training ==="
echo "  Oracle dir : $ORACLE_DIR"
echo "  Log dir    : $LOG_DIR"
echo "  Min turns  : $MIN_TURNS"
echo ""

python -m thalamus.oracle train-classifier \
  --oracle-dir "$ORACLE_DIR" \
  --log-dir    "$LOG_DIR" \
  --min-turns  "$MIN_TURNS" \
  --max-weeks  "$MAX_WEEKS" \
  --C          "$C_REGULARIZATION"

echo ""
echo "Done. Classifier written to: $ORACLE_DIR/classifier.pkl"
echo "      Registry updated  at : $ORACLE_DIR/classifier_registry.json"
