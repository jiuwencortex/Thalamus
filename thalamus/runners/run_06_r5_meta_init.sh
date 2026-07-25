#!/usr/bin/env bash
# R5 — Cross-deployment meta-learning: extract knowledge then warm-start a new oracle.
#
# Run AFTER operating 2+ jiuwenswarm deployments for at least a few weeks.
#
# What this script does:
#   1. Extracts component quality statistics from an existing (mature) oracle and
#      adds them to the shared cross-deployment knowledge base (knowledge_base.json).
#   2. Warm-starts a new oracle directory using matching component fingerprints
#      from the KB, writing transfer_priors.json so the first oracle build uses
#      real quality evidence instead of flat cold-start priors.
#
# Prerequisites:
#   - thalamus-research installed: uv pip install -e ./thalamus_research
#   - SOURCE_ORACLE_DIR: a mature oracle with context_configs.json + turn logs
#     (at least ~100 turns with outcome data for meaningful KB entries)
#   - NEW_ORACLE_DIR: a fresh oracle dir for the new deployment
#     (must already have scoring_matrix_*.json files from run_01_score.sh)
#
# Required env vars:
#   SOURCE_ORACLE_DIR  — completed oracle to extract knowledge from
#   NEW_ORACLE_DIR     — new oracle dir to warm-start
#   KB_PATH            — path to shared knowledge_base.json
#                        (created on first extract; grows across deployments)
#
# Optional env vars:
#   EXTRACT_ONLY=1     — skip the transfer step (extract only)
#   TRANSFER_ONLY=1    — skip the extract step (requires KB_PATH to already exist)
#
# Usage:
#   SOURCE_ORACLE_DIR=/deploy1/oracle \
#   NEW_ORACLE_DIR=/deploy2/oracle \
#   KB_PATH=~/.jiuwenswarm/knowledge_base.json \
#   bash run_06_r5_meta_init.sh

set -euo pipefail

SOURCE_ORACLE_DIR="${SOURCE_ORACLE_DIR:-}"
NEW_ORACLE_DIR="${NEW_ORACLE_DIR:-}"
KB_PATH="${KB_PATH:-$HOME/.jiuwenswarm/thalamus_knowledge_base.json}"
EXTRACT_ONLY="${EXTRACT_ONLY:-0}"
TRANSFER_ONLY="${TRANSFER_ONLY:-0}"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   R5 — Cross-Deployment Meta-Learning                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "KB path:       $KB_PATH"
[[ -n "$SOURCE_ORACLE_DIR" ]] && echo "Source oracle: $SOURCE_ORACLE_DIR"
[[ -n "$NEW_ORACLE_DIR" ]]    && echo "New oracle:    $NEW_ORACLE_DIR"
echo ""

# ── Step 1: Extract knowledge from source oracle ──────────────────────────────
if [[ "$TRANSFER_ONLY" != "1" ]]; then
    if [[ -z "$SOURCE_ORACLE_DIR" ]]; then
        echo "ERROR: SOURCE_ORACLE_DIR is not set." >&2
        echo "  SOURCE_ORACLE_DIR=/path/to/mature/oracle bash $0" >&2
        exit 1
    fi
    if [[ ! -d "$SOURCE_ORACLE_DIR" ]]; then
        echo "ERROR: SOURCE_ORACLE_DIR not found: $SOURCE_ORACLE_DIR" >&2
        exit 1
    fi
    if [[ ! -f "$SOURCE_ORACLE_DIR/context_configs.json" ]]; then
        echo "ERROR: $SOURCE_ORACLE_DIR/context_configs.json not found." >&2
        echo "Run run_02_oracle.sh on the source deployment first." >&2
        exit 1
    fi

    mkdir -p "$(dirname "$KB_PATH")"

    echo "══ Step 1: Extracting knowledge into KB ══════════════════"
    echo "  Source: $SOURCE_ORACLE_DIR"
    echo "  KB:     $KB_PATH"
    echo ""
    thalamus-research meta-learning \
        --oracle-dir "$SOURCE_ORACLE_DIR" \
        --kb-path    "$KB_PATH" \
        --subcommand extract \
        --out        "$SOURCE_ORACLE_DIR/research_results/r5_extract.json"

    echo ""
    echo "KB updated: $KB_PATH"
    echo "Report:     $SOURCE_ORACLE_DIR/research_results/r5_extract.json"
    echo ""
fi

# ── Step 2: Warm-start new oracle from KB ─────────────────────────────────────
if [[ "$EXTRACT_ONLY" != "1" ]]; then
    if [[ -z "$NEW_ORACLE_DIR" ]]; then
        echo "ERROR: NEW_ORACLE_DIR is not set." >&2
        echo "  NEW_ORACLE_DIR=/path/to/new/oracle bash $0" >&2
        exit 1
    fi
    if [[ ! -d "$NEW_ORACLE_DIR" ]]; then
        echo "ERROR: NEW_ORACLE_DIR not found: $NEW_ORACLE_DIR" >&2
        echo "Create the oracle dir and run run_01_score.sh first." >&2
        exit 1
    fi
    if [[ ! -f "$KB_PATH" ]]; then
        echo "ERROR: KB not found at $KB_PATH." >&2
        echo "Run Step 1 (extract) first to build the knowledge base." >&2
        exit 1
    fi

    echo "══ Step 2: Warm-starting new oracle from KB ══════════════"
    echo "  New oracle: $NEW_ORACLE_DIR"
    echo "  KB:         $KB_PATH"
    echo ""
    thalamus-oracle meta-init \
        --oracle-dir "$NEW_ORACLE_DIR" \
        --kb-path    "$KB_PATH"

    echo ""
    echo "transfer_priors.json written to: $NEW_ORACLE_DIR"
    echo ""
    echo "══ Step 3: Build oracle with warm-start priors ═══════════"
    echo "Run the oracle build — it will read transfer_priors.json automatically:"
    echo ""
    echo "  thalamus-oracle evolve --oracle-dir \"$NEW_ORACLE_DIR\""
    echo ""
fi

echo "╔══════════════════════════════════════════════════════╗"
echo "║   R5 meta-init complete.                             ║"
echo "╠══════════════════════════════════════════════════════╣"
if [[ "$EXTRACT_ONLY" != "1" ]]; then
echo "║  transfer_priors.json → $NEW_ORACLE_DIR"
fi
echo "╠══════════════════════════════════════════════════════╣"
echo "║  To add more deployments to the KB, re-run with:    ║"
echo "║    SOURCE_ORACLE_DIR=/next/oracle EXTRACT_ONLY=1    ║"
echo "║    bash run_06_r5_meta_init.sh                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
