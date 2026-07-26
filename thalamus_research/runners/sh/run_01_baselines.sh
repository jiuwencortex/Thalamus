#!/usr/bin/env bash
# Phase R1 — Baseline evaluation.
#
# Benchmarks all retrieval baselines (TF-IDF, BM25, dense, random, all)
# against the full THALAMUS selector over a query set.  Writes a JSON result
# file that a jiuwenswarm quality-measurement pass can fill with outcome scores.
#
# Usage:
#   bash run_01_baselines.sh
#   QUERIES_FILE=/my/queries.json bash run_01_baselines.sh
#
# Output:
#   $RESULTS_DIR/r1_baselines.json   — latency + overlap statistics per selector
#   (stdout)                         — ASCII comparison table

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_config.sh"

_section "R1 — Baseline Evaluation"
echo "Oracle:   $ORACLE_DIR"
echo "Results:  $RESULTS_DIR/r1_baselines.json"
[[ -n "$QUERIES_FILE" ]] && echo "Queries:  $QUERIES_FILE" || echo "Queries:  (inline default)"
echo ""

QUERY_ARGS=()
[[ -n "$QUERIES_FILE" ]] && QUERY_ARGS+=(--queries-file "$QUERIES_FILE") \
                         || QUERY_ARGS+=(--query "Write a unit test for the payment module"
                                                  "Set up a CI/CD pipeline"
                                                  "Debug a memory leak in production"
                                                  "Refactor the authentication service"
                                                  "Generate API documentation")

thalamus-research eval \
    --oracle-dir "$ORACLE_DIR" \
    --selectors thalamus tfidf bm25 random all \
    "${QUERY_ARGS[@]}" \
    --budget medium \
    --ordering bookend \
    --n-repeats 5 \
    --reference thalamus \
    --print-report \
    --output "$RESULTS_DIR/r1_baselines.json"

echo ""
echo "Done. Result: $RESULTS_DIR/r1_baselines.json"
echo "Next: fill 'quality' fields by running agent tasks, then proceed to R2."
