#!/usr/bin/env python3
"""
Phase 4: Train the component inclusion classifier from logged agent turns.

Python equivalent of run_03_classifier.sh — import-and-call, no subprocess boundary.

Config via environment variables:
  ORACLE_DIR          Oracle directory                         (required)
  LOG_DIR             JSONL turn logs directory                (default: ORACLE_DIR/online_logs)
  MIN_TURNS           Minimum logged turns to train            (default: 10)
  MAX_WEEKS           Weekly log files to scan                 (default: 8)
  C_REGULARIZATION    Inverse L2 strength for LogisticReg      (default: 1.0)

Requires at least MIN_TURNS logged turns (written by TurnLogger.log_turn +
update_outcome). Run after the agent has accumulated real interaction data.
"""
import os
import sys


def main() -> None:
    oracle_dir = os.environ.get("ORACLE_DIR",        "~/.jiuwenswarm/agent/workspace/oracle")
    log_dir    = os.environ.get("LOG_DIR",           os.path.join(oracle_dir, "online_logs"))
    min_turns  = os.environ.get("MIN_TURNS",         "10")
    max_weeks  = os.environ.get("MAX_WEEKS",         "8")
    c_reg      = os.environ.get("C_REGULARIZATION",  "1.0")

    print("=== Phase 4: Classifier Training ===")
    print(f"  Oracle dir : {oracle_dir}")
    print(f"  Log dir    : {log_dir}")
    print(f"  Min turns  : {min_turns}")
    print()

    from thalamus.oracle.cli import main as oracle_main  # noqa: PLC0415

    sys.argv = [
        "thalamus-oracle", "train-classifier",
        "--oracle-dir", oracle_dir,
        "--log-dir",    log_dir,
        "--min-turns",  min_turns,
        "--max-weeks",  max_weeks,
        "--C",          c_reg,
    ]
    oracle_main()

    print()
    print(f"Done. Classifier written to:  {oracle_dir}/classifier.pkl")
    print(f"      Registry updated at:    {oracle_dir}/classifier_registry.json")


if __name__ == "__main__":
    main()
