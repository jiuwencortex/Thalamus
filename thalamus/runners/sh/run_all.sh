#!/usr/bin/env bash
# Full Thalamus pipeline: score → oracle → select.
#
# Steps:
#   1. Score all components via LLM (Phase 1-2)
#   2. Build evolutionary oracle — no LLM calls (Phase 3)
#   3. Validate runtime lookup with a test query (Phase 3 output check)
#
# Phase 4 (classifier training) is omitted here because it requires
# accumulated agent turn logs that don't exist on first run.
# Run run_03_classifier.sh separately once logs are available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Shared config (override via env vars before running) ─────────────────────
export SKILLS_DIR="${SKILLS_DIR:-~/.jiuwenswarm/agent/workspace/skills}"
export PROJECT_DIR="${PROJECT_DIR:-~/.jiuwenswarm/agent/workspace}"
export TOOLS_DIR="${TOOLS_DIR:-~/.jiuwenswarm/agent/workspace/tools}"
export ORACLE_DIR="${ORACLE_DIR:-~/.jiuwenswarm/agent/workspace/oracle}"
export MODEL="${MODEL:-gpt-4o-mini}"
# OPENAI_API_KEY must be set in your environment

echo "======================================================"
echo " THALAMUS — Full Pipeline"
echo "======================================================"
echo ""

bash "$SCRIPT_DIR/run_01_score.sh"
echo ""

bash "$SCRIPT_DIR/run_02_oracle.sh"
echo ""

# Quick sanity-check: resolve a test query against the freshly built oracle
export QUERY="${QUERY:-Write a unit test for the payment module}"
export BUDGET="auto"
export ORDERING="bookend"
bash "$SCRIPT_DIR/run_04_select.sh"

echo ""
echo "======================================================"
echo " Pipeline complete."
echo " Next step (after agent accumulates logs):"
echo "   bash runners/run_03_classifier.sh"
echo "======================================================"
