#!/usr/bin/env bash
# Phase 3: Build the evolutionary oracle (context_configs.json).
#
# What this does:
#   - Reads all scoring_matrix_*.json files from ORACLE_DIR
#   - Clusters the query space with K-means (TF-IDF or sentence-transformer)
#   - Runs a genetic algorithm (no LLM calls) to find optimal component combos
#     per cluster × budget combination
#   - Writes context_configs.json + context_configs.pkl to ORACLE_DIR
#
# Run after run_01_score.sh.

set -euo pipefail

# ── Configure ────────────────────────────────────────────────────────────────
ORACLE_DIR="${ORACLE_DIR:-~/.jiuwenswarm/agent/workspace/oracle}"

# Clustering
EMBEDDER="${EMBEDDER:-tfidf}"         # tfidf | sentence
N_CLUSTERS="${N_CLUSTERS:-20}"        # ignored when --auto-k is set
AUTO_K="${AUTO_K:-true}"              # auto-select optimal K via elbow+silhouette

# Evolutionary search
POPULATION="${POPULATION:-100}"
GENERATIONS="${GENERATIONS:-200}"
MUTATION_RATE="${MUTATION_RATE:-0.05}"
LAMBDA="${LAMBDA:-0.1}"               # token-penalty weight in fitness

# Token budgets (per budget tier)
BUDGET_SMALL="${BUDGET_SMALL:-2000}"
BUDGET_MEDIUM="${BUDGET_MEDIUM:-4000}"
BUDGET_LARGE="${BUDGET_LARGE:-8000}"

echo "=== Phase 3: Oracle Building ==="
echo "  Oracle dir : $ORACLE_DIR"
echo "  Embedder   : $EMBEDDER"
echo "  Auto-K     : $AUTO_K"
echo "  Population : $POPULATION  Generations: $GENERATIONS"
echo ""

AUTO_K_FLAG=""
if [[ "$AUTO_K" == "true" ]]; then
  AUTO_K_FLAG="--auto-k"
fi

python -m thalamus.oracle evolve \
  --oracle-dir    "$ORACLE_DIR" \
  --embedder      "$EMBEDDER" \
  --n-clusters    "$N_CLUSTERS" \
  --population    "$POPULATION" \
  --generations   "$GENERATIONS" \
  --mutation-rate "$MUTATION_RATE" \
  --lambda        "$LAMBDA" \
  --budget-small  "$BUDGET_SMALL" \
  --budget-medium "$BUDGET_MEDIUM" \
  --budget-large  "$BUDGET_LARGE" \
  $AUTO_K_FLAG

echo ""
echo "Done. Oracle written to: $ORACLE_DIR/context_configs.json"
