#!/usr/bin/env bash
# Full Thalamus pipeline: ingest → score → oracle → select.
#
# Steps:
#   0. Ingest jiuwenswarm sessions → online_logs/  (Phase 0)
#   1. Score all components via LLM                (Phase 1-2)
#   2. Build evolutionary oracle — no LLM calls    (Phase 3)
#   3. Validate runtime lookup with a test query   (Phase 3 output check)
#
# Phase 4 (classifier training) is omitted here because it requires
# enough logged turns (>= 10). After step 0 has populated online_logs/,
# run run_03_classifier.sh separately.

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

bash "$SCRIPT_DIR/run_00_ingest_sessions.sh"
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
echo " To enable the classifier (needs >= 10 logged turns):"
echo "   bash runners/sh/run_00_ingest_sessions.sh  # ingest sessions"
echo "   bash runners/sh/run_03_classifier.sh       # train classifier"
echo "======================================================"
