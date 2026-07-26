#!/usr/bin/env python3
"""
R4 — Activate set-level quality fitness in the production oracle.

Python equivalent of run_05_r4_activate.sh — import-and-call, no subprocess boundary.

Run AFTER collecting ~500–1000 agent turns with filled 'quality' fields.

Steps:
  1. Train GradientBoostingRegressor on (component_set, quality) pairs.
  2. Evaluate the model on a held-out split (RMSE, R²).
  3. Rebuild context_configs.json using the XGB model as GA fitness.

Config via environment variables:
  ORACLE_DIR    Oracle directory (must have context_configs.pkl)    (required)
  TURN_LOG_DIR  Directory with turns_*.jsonl                        (default: ORACLE_DIR/online_logs)
  MODEL_DIR     Where to write model.pkl                            (default: ORACLE_DIR/set_quality_model)
  MIN_TURNS     Minimum labelled turns to train                     (default: 50)

Prerequisites:
  - thalamus_research installed (uv pip install -e ./thalamus_research)
  - Oracle built (run_02_oracle.py)
  - Turn logs with quality fields filled (TurnLogger.update_outcome)
"""
import os
import sys
from pathlib import Path


def main() -> None:
    oracle_dir   = os.path.expanduser(os.environ.get("ORACLE_DIR",    "~/.jiuwenswarm/agent/workspace/oracle"))
    turn_log_dir = os.environ.get("TURN_LOG_DIR",  os.path.join(oracle_dir, "online_logs"))
    model_dir    = os.environ.get("MODEL_DIR",     os.path.join(oracle_dir, "set_quality_model"))
    min_turns    = os.environ.get("MIN_TURNS",     "50")

    results_dir = os.path.join(oracle_dir, "research_results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   R4 — Set-Level Quality Fitness Activation          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"Oracle:     {oracle_dir}")
    print(f"Turn logs:  {turn_log_dir}")
    print(f"Model dir:  {model_dir}")
    print()

    if not os.path.isdir(oracle_dir):
        print(f"ERROR: ORACLE_DIR not found: {oracle_dir}", file=sys.stderr)
        print("Run run_02_oracle.py first to build the oracle.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(turn_log_dir):
        print(f"ERROR: TURN_LOG_DIR not found: {turn_log_dir}", file=sys.stderr)
        print("Collect agent turn logs with filled 'quality' fields first.", file=sys.stderr)
        sys.exit(1)

    from thalamus_research.cli import main as research_main  # noqa: PLC0415
    from thalamus.oracle.cli import main as oracle_main      # noqa: PLC0415

    # Step 1 — Train XGB set-quality model
    print("══ Step 1: Training set-quality model (XGB) ══════════════")
    sys.argv = [
        "thalamus-research", "set-quality",
        "--oracle-dir",   oracle_dir,
        "--turn-log-dir", turn_log_dir,
        "--model-dir",    model_dir,
        "--subcommand",   "train",
        "--min-turns",    min_turns,
        "--out",          os.path.join(results_dir, "r4_set_quality_train.json"),
    ]
    research_main()
    print()
    print(f"Model saved: {model_dir}/model.pkl")
    print()

    # Step 2 — Evaluate model on held-out turns
    print("══ Step 2: Evaluating set-quality model ══════════════════")
    sys.argv = [
        "thalamus-research", "set-quality",
        "--oracle-dir",   oracle_dir,
        "--turn-log-dir", turn_log_dir,
        "--model-dir",    model_dir,
        "--subcommand",   "evaluate",
        "--out",          os.path.join(results_dir, "r4_set_quality_eval.json"),
    ]
    research_main()
    print()
    print(f"Eval report: {results_dir}/r4_set_quality_eval.json")
    print()

    # Step 3 — Rebuild oracle with XGB fitness
    print("══ Step 3: Rebuilding oracle with XGB fitness ════════════")
    sys.argv = [
        "thalamus-oracle", "evolve",
        "--oracle-dir",        oracle_dir,
        "--fitness-model",     "xgb",
        "--fitness-model-dir", model_dir,
    ]
    oracle_main()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   R4 activation complete.                            ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  context_configs.json rebuilt with XGB fitness.      ║")
    print(f"║  Eval report: {results_dir}/r4_set_quality_eval.json")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  To revert to marginal fitness:                      ║")
    print(f"║    thalamus-oracle evolve --oracle-dir {oracle_dir}  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
