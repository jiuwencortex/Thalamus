#!/usr/bin/env bash
# Phase R3a — Cross-path knowledge transfer.
#
# Extracts co-inclusion signal from the Path B classifier's weight matrix W:
#   co_inclusion(c_i, c_j) = cosine(W[i], W[j])
#
# High co-inclusion → the classifier learned these components are jointly useful.
# This signal augments the GA fitness function:
#   fitness_aug = base_fitness + λ × mean_pairwise_co_inclusion(S)
#
# Two modes:
#   1. Report mode  — print top joint and redundant component pairs (always runs)
#   2. Augment mode — write augmented context_configs.json (opt-in via AUGMENT=1)
#
# Requires: classifier_current.pkl in ORACLE_DIR (run thalamus-oracle train-classifier first)
#
# Usage:
#   bash run_03_cross_path.sh               # report only
#   AUGMENT=1 bash run_03_cross_path.sh     # report + write augmented configs
#   LAM=0.3 AUGMENT=1 bash run_03_cross_path.sh
#
# Output:
#   $RESULTS_DIR/r3a_cross_path.json
#   $RESULTS_DIR/context_configs_augmented.json   (if AUGMENT=1)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_config.sh"

AUGMENT="${AUGMENT:-0}"
LAM="${LAM:-$CO_INCLUSION_LAM}"

_section "R3a — Cross-Path Co-Inclusion Analysis"
echo "Oracle:     $ORACLE_DIR"
echo "λ (lambda): $LAM"
echo "Top pairs:  $TOP_PAIRS"
echo "Augment:    $AUGMENT"
echo ""

# Always run the co-inclusion report
thalamus-research cross-path \
    --oracle-dir "$ORACLE_DIR" \
    --top-pairs "$TOP_PAIRS" \
    --lam "$LAM" \
    --out "$RESULTS_DIR/r3a_cross_path.json"

echo ""
echo "Done. Report: $RESULTS_DIR/r3a_cross_path.json"

# Optionally produce augmented context_configs.json
if [[ "$AUGMENT" == "1" ]]; then
    echo ""
    echo "── Augmenting context_configs.json (λ=$LAM) ──"
    thalamus-research cross-path \
        --oracle-dir "$ORACLE_DIR" \
        --augment-configs \
        --lam "$LAM" \
        --out "$RESULTS_DIR/context_configs_augmented.json"
    echo "Augmented configs: $RESULTS_DIR/context_configs_augmented.json"
    echo ""
    echo "To use augmented fitness in the GA:"
    echo "  cp $RESULTS_DIR/context_configs_augmented.json $ORACLE_DIR/context_configs.json"
    echo "  (then re-run thalamus-oracle evolve to regenerate the oracle with augmented fitness)"
fi
