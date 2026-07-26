#!/usr/bin/env bash
# Runtime: Select context for a query (cluster-based lookup).
#
# What this does:
#   - Vectorizes QUERY against the trained TF-IDF / sentence-transformer model
#   - Predicts the nearest K-means cluster
#   - Returns the precomputed optimal {skills, memory, tools} config for that
#     cluster + budget combination — no LLM calls, no network access
#
# Requires context_configs.json from run_02_oracle.sh.
# Pass QUERY as an env var or edit the default below.

set -euo pipefail

# ── Configure ────────────────────────────────────────────────────────────────
ORACLE_DIR="${ORACLE_DIR:-~/.jiuwenswarm/agent/workspace/oracle}"
QUERY="${QUERY:-Set up a CI pipeline for my new microservice}"
BUDGET="${BUDGET:-auto}"            # small | medium | large | auto
ORDERING="${ORDERING:-bookend}"     # relevance | bookend | none

echo "=== Runtime: Context Selection ==="
echo "  Oracle dir : $ORACLE_DIR"
echo "  Query      : $QUERY"
echo "  Budget     : $BUDGET"
echo "  Ordering   : $ORDERING"
echo ""

python -m thalamus.selection lookup \
  --oracle-dir "$ORACLE_DIR" \
  --query      "$QUERY" \
  --budget     "$BUDGET" \
  --ordering   "$ORDERING"
