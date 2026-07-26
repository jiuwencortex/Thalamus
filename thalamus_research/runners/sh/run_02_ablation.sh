#!/usr/bin/env bash
# Phase R2 — Ablation study.
#
# Runs four ablation selectors that each isolate one THALAMUS design choice,
# then compares them against the full system on the same query set.
#
# Ablation variants:
#   TopK          (C1) removes GA evolutionary search → greedy TF-IDF × mean_score
#   PathBOnly     (C3) removes Path A fallback → classifier-only, cold-start = None
#   NoBookend     (C5) removes bookend ordering → relevance-descending order
#   SingleBudget  (C6) removes adaptive budget → fixed "medium" tier only
#
# Usage:
#   bash run_02_ablation.sh
#   QUERIES_FILE=/my/queries.json bash run_02_ablation.sh
#
# Output:
#   $RESULTS_DIR/r2_ablation.json   — per-selector latency + overlap statistics

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_config.sh"

_section "R2 — Ablation Study"
echo "Oracle:   $ORACLE_DIR"
echo "Results:  $RESULTS_DIR/r2_ablation.json"
echo ""

QUERY_ARGS=()
[[ -n "$QUERIES_FILE" ]] && QUERY_ARGS+=(--query-file "$QUERIES_FILE") \
                         || QUERY_ARGS+=(--query "Write a unit test for the payment module"
                                                  "Set up a CI/CD pipeline"
                                                  "Debug a memory leak in production"
                                                  "Refactor the authentication service"
                                                  "Generate API documentation")

thalamus-research ablation \
    --oracle-dir "$ORACLE_DIR" \
    "${QUERY_ARGS[@]}" \
    --budget auto \
    --ordering bookend \
    --n-repeats 3 \
    --out "$RESULTS_DIR/r2_ablation.json"

echo ""
echo "Done. Result: $RESULTS_DIR/r2_ablation.json"
echo "Key metrics to compare: overlap with thalamus (higher = ablation safe to remove),"
echo "latency_ms (lower = design choice adds cost), and component count per budget."
