#!/usr/bin/env bash
# Phase R5 — Cross-deployment meta-learning.
#
# Transfers component quality knowledge between jiuwenswarm deployments using
# SHA-256 content fingerprints.  Two operations:
#
#   extract   Read component statistics from a completed oracle and add them
#             to the shared knowledge base (KB).  Run once per deployment.
#
#   transfer  Look up the new deployment's components in the KB and write
#             transfer_priors.json to its oracle dir.  Run before the first
#             thalamus-oracle evolve on a new deployment.
#
# Requires:
#   - ORACLE_DIR: a completed oracle (has context_configs.json)
#   - KB_PATH: path to the shared knowledge_base.json (created on first extract)
#   - NEW_ORACLE_DIR: oracle dir for the deployment to warm-start (transfer only)
#
# Usage:
#   bash run_06_meta_learning.sh                # extract only
#   MODE=transfer NEW_ORACLE_DIR=/new bash ...  # transfer only
#   MODE=both NEW_ORACLE_DIR=/new bash ...      # extract then transfer
#
# Output:
#   $KB_PATH                                   (extract — updated KB)
#   $RESULTS_DIR/r5_extract.json               (extract report)
#   $NEW_ORACLE_DIR/transfer_priors.json       (transfer — warm-start priors)
#   $RESULTS_DIR/r5_transfer.json              (transfer report)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_config.sh"

MODE="${MODE:-extract}"

_section "R5 — Cross-Deployment Meta-Learning"
echo "Oracle:         $ORACLE_DIR"
echo "KB path:        $KB_PATH"
echo "Mode:           $MODE"
[[ -n "$NEW_ORACLE_DIR" ]] && echo "New oracle:     $NEW_ORACLE_DIR"
echo ""

if [[ "$MODE" == "extract" || "$MODE" == "both" ]]; then
    echo "── Extract: ingesting $ORACLE_DIR into KB ──────────────"
    thalamus-research meta-learning \
        --oracle-dir "$ORACLE_DIR" \
        --kb-path "$KB_PATH" \
        --subcommand extract \
        --out "$RESULTS_DIR/r5_extract.json"
    echo ""
    echo "KB updated: $KB_PATH"
    echo "Report:     $RESULTS_DIR/r5_extract.json"
fi

if [[ "$MODE" == "transfer" || "$MODE" == "both" ]]; then
    echo ""
    if [[ -z "$NEW_ORACLE_DIR" ]]; then
        echo "ERROR: NEW_ORACLE_DIR must be set for transfer mode."
        echo "  NEW_ORACLE_DIR=/path/to/new/oracle bash run_06_meta_learning.sh"
        exit 1
    fi
    if [[ ! -f "$KB_PATH" ]]; then
        echo "ERROR: KB not found at $KB_PATH."
        echo "Run extract mode first to build the knowledge base."
        exit 1
    fi
    echo "── Transfer: warm-starting $NEW_ORACLE_DIR ─────────────"
    thalamus-research meta-learning \
        --oracle-dir "$NEW_ORACLE_DIR" \
        --kb-path "$KB_PATH" \
        --subcommand transfer \
        --out "$RESULTS_DIR/r5_transfer.json"
    echo ""
    echo "Priors written: $NEW_ORACLE_DIR/transfer_priors.json"
    echo "Report:         $RESULTS_DIR/r5_transfer.json"
    echo ""
    echo "Next: run thalamus-oracle evolve on the new deployment."
    echo "The GA will read transfer_priors.json to warm-start fitness scores."
    echo ""
    echo "Production shorthand (equivalent to the transfer step above):"
    echo "  thalamus-oracle meta-init \\"
    echo "    --oracle-dir \"$NEW_ORACLE_DIR\" \\"
    echo "    --kb-path \"$KB_PATH\""
fi
