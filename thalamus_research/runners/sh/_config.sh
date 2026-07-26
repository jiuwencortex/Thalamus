#!/usr/bin/env bash
# Shared configuration for thalamus-research runners.
# Source this file at the top of each runner script:
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/_config.sh"
#
# Override any variable by exporting it before calling the runner:
#   ORACLE_DIR=/my/oracle bash run_01_baselines.sh

# ── Required ──────────────────────────────────────────────────────────────────
# Directory produced by `thalamus-oracle evolve` — contains context_configs.json,
# context_configs.pkl, and (after training) classifier_current.pkl.
ORACLE_DIR="${ORACLE_DIR:-~/.jiuwenswarm/agent/workspace/oracle}"

# ── Results output ────────────────────────────────────────────────────────────
# All JSON result files are written here.
RESULTS_DIR="${RESULTS_DIR:-$ORACLE_DIR/research_results}"

# ── Queries ───────────────────────────────────────────────────────────────────
# JSON file with a list of {"id": str, "query": str} objects, or a plain-text
# file with one query per line.  Used by eval and ablation runners.
QUERIES_FILE="${QUERIES_FILE:-}"   # empty → each runner uses --query default

# ── Turn logs ─────────────────────────────────────────────────────────────────
# Directory containing turns_YYYY-WNN.jsonl files.  Required for R3b, R4, R5.
TURN_LOG_DIR="${TURN_LOG_DIR:-$ORACLE_DIR/online_logs}"

# ── R3a cross-path ────────────────────────────────────────────────────────────
CO_INCLUSION_LAM="${CO_INCLUSION_LAM:-0.2}"   # λ weight for fitness augmentation
TOP_PAIRS="${TOP_PAIRS:-20}"                  # pairs printed in co-inclusion report

# ── R3b bandit ────────────────────────────────────────────────────────────────
BANDIT_N_MIN="${BANDIT_N_MIN:-10}"      # min samples per component for ε* calc
BANDIT_T_TARGET="${BANDIT_T_TARGET:-500}"  # target total turns for ε* calc
BANDIT_WINDOW="${BANDIT_WINDOW:-50}"    # rolling window for convergence analysis
BANDIT_BUDGET="${BANDIT_BUDGET:-medium}"

# ── R5 meta-learning ──────────────────────────────────────────────────────────
# Shared knowledge base file (written by extract, read by transfer).
KB_PATH="${KB_PATH:-~/.jiuwenswarm/thalamus_knowledge_base.json}"
# A second oracle to warm-start from the KB (transfer subcommand only).
NEW_ORACLE_DIR="${NEW_ORACLE_DIR:-}"

# ── Helpers ───────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

_require_var() {
    local var="$1" label="$2"
    if [[ -z "${!var}" ]]; then
        echo "ERROR: $label is not set. Export $var before running." >&2
        exit 1
    fi
}

_section() {
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  $*"
    echo "══════════════════════════════════════════════════════"
    echo ""
}
