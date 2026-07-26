#!/usr/bin/env bash
# Phase R3b — Contextual bandit analysis.
#
# Two sub-analyses:
#
#   estimate-rate   Derives ε* — the minimum exploration rate needed so that
#                   every component receives at least N_MIN logged turns within
#                   T_TARGET total agent interactions.  Run this BEFORE setting
#                   the exploration rate in the turn logger.
#
#   convergence     Measures how similar Path B's (classifier) action distribution
#                   has become to Path A's (GA oracle) over the logged turn history.
#                   Jaccard similarity ≥ 0.85 over the last WINDOW turns signals
#                   that Path B has converged.  Run this periodically after logging.
#
# Usage:
#   bash run_04_bandit.sh               # both sub-analyses
#   MODE=estimate bash run_04_bandit.sh # estimate-rate only
#   MODE=convergence bash run_04_bandit.sh
#
# Output:
#   $RESULTS_DIR/r3b_bandit_epsilon.json      (estimate-rate)
#   $RESULTS_DIR/r3b_bandit_convergence.json  (convergence)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_config.sh"

MODE="${MODE:-both}"   # both | estimate | convergence

_section "R3b — Bandit Analysis"
echo "Oracle:        $ORACLE_DIR"
echo "Turn log dir:  $TURN_LOG_DIR"
echo "Mode:          $MODE"
echo ""

if [[ "$MODE" == "both" || "$MODE" == "estimate" ]]; then
    echo "── ε* Estimation ──────────────────────────────────────"
    echo "n_min=$BANDIT_N_MIN  T_target=$BANDIT_T_TARGET"
    echo ""
    thalamus-research bandit \
        --oracle-dir "$ORACLE_DIR" \
        --subcommand estimate-rate \
        --n-min "$BANDIT_N_MIN" \
        --T-target "$BANDIT_T_TARGET" \
        --out "$RESULTS_DIR/r3b_bandit_epsilon.json"
    echo ""
    echo "Result: $RESULTS_DIR/r3b_bandit_epsilon.json"
    echo "→ Set TurnLogger(exploration_rate=<epsilon_star>) in jiuwenswarm config."
fi

if [[ "$MODE" == "both" || "$MODE" == "convergence" ]]; then
    echo ""
    echo "── Convergence Analysis ──────────────────────────────"
    echo "Window: $BANDIT_WINDOW turns  Budget: $BANDIT_BUDGET"
    echo ""

    if [[ ! -d "$TURN_LOG_DIR" ]]; then
        echo "WARNING: TURN_LOG_DIR not found ($TURN_LOG_DIR)."
        echo "Convergence analysis requires logged agent turns."
        echo "Skipping convergence — run again after turns are available."
    else
        thalamus-research bandit \
            --oracle-dir "$ORACLE_DIR" \
            --turn-log-dir "$TURN_LOG_DIR" \
            --subcommand convergence \
            --window-size "$BANDIT_WINDOW" \
            --budget "$BANDIT_BUDGET" \
            --out "$RESULTS_DIR/r3b_bandit_convergence.json"
        echo ""
        echo "Result: $RESULTS_DIR/r3b_bandit_convergence.json"
        echo "→ When converged_to_path_a=true, Path B is ready for production solo use."
    fi
fi
