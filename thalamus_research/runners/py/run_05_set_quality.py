#!/usr/bin/env python3
"""
R4 — Set-level quality model.

Python equivalent of run_05_set_quality.sh — import-and-call, no subprocess boundary.

Trains a GradientBoostingRegressor on (component_set, outcome_quality) pairs
from logged agent turns.  Captures non-linear pairwise interactions between
components that the GA's linear sum-of-scores fitness misses.

Modes (controlled by MODE env var):
  train      Fit and save the model (default)
  evaluate   Load saved model, report RMSE and R² on available turns
  both       Train then evaluate in-sample (quick sanity check)

Config via environment variables (see _config.py):
  ORACLE_DIR     Oracle directory (needs context_configs.pkl)    (required)
  TURN_LOG_DIR   Directory with turns_*.jsonl                    (default: ORACLE_DIR/online_logs)
  RESULTS_DIR    Output directory                                (default: ORACLE_DIR/research_results)
  MODEL_DIR      Where to write model.pkl                        (default: ORACLE_DIR/set_quality_model)
  MODE           train | evaluate | both                         (default: train)

Requires:
  - ~20+ labelled turns per cluster for meaningful signal
  - turn logs whose 'quality' fields are filled (null entries are skipped)

Output:
  $MODEL_DIR/model.pkl                     (train)
  $MODEL_DIR/meta.json                     (train)
  $RESULTS_DIR/r4_set_quality_train.json   (train report)
  $RESULTS_DIR/r4_set_quality_eval.json    (evaluate report)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _config import cfg, section  # noqa: E402


def main() -> None:
    mode      = os.environ.get("MODE",      "train")
    model_dir = os.environ.get("MODEL_DIR", os.path.join(cfg.oracle_dir, "set_quality_model"))

    section("R4 — Set-Level Quality Model")
    print(f"Oracle:     {cfg.oracle_dir}")
    print(f"Turn logs:  {cfg.turn_log_dir}")
    print(f"Model dir:  {model_dir}")
    print(f"Mode:       {mode}")
    print()

    if not os.path.isdir(cfg.turn_log_dir):
        print(f"ERROR: TURN_LOG_DIR not found ({cfg.turn_log_dir}).", file=sys.stderr)
        print("R4 requires logged agent turns with filled 'quality' fields.", file=sys.stderr)
        print("Collect turns first, then re-run.", file=sys.stderr)
        sys.exit(1)

    from thalamus_research.cli import main as research_main  # noqa: PLC0415

    if mode in ("train", "both"):
        print("── Training ────────────────────────────────────────────")
        sys.argv = [
            "thalamus-research", "set-quality",
            "--oracle-dir",   cfg.oracle_dir,
            "--turn-log-dir", cfg.turn_log_dir,
            "--model-dir",    model_dir,
            "--subcommand",   "train",
            "--out",          os.path.join(cfg.results_dir, "r4_set_quality_train.json"),
        ]
        research_main()
        print()
        print(f"Model saved to: {model_dir}")
        print(f"Report:         {cfg.results_dir}/r4_set_quality_train.json")

    if mode in ("evaluate", "both"):
        print()
        print("── Evaluation ──────────────────────────────────────────")
        sys.argv = [
            "thalamus-research", "set-quality",
            "--oracle-dir",   cfg.oracle_dir,
            "--turn-log-dir", cfg.turn_log_dir,
            "--model-dir",    model_dir,
            "--subcommand",   "evaluate",
            "--out",          os.path.join(cfg.results_dir, "r4_set_quality_eval.json"),
        ]
        research_main()
        print()
        print(f"Report: {cfg.results_dir}/r4_set_quality_eval.json")

    print()
    print("── Production integration ──────────────────────────────")
    print("To use the XGB model as the GA fitness function, re-run the oracle build:")
    print()
    print(f"  ORACLE_DIR={cfg.oracle_dir} python run_05_r4_activate.py")
    print()
    print("Or directly:")
    print(f"  thalamus-oracle evolve \\")
    print(f"    --oracle-dir \"{cfg.oracle_dir}\" \\")
    print(f"    --fitness-model xgb \\")
    print(f"    --fitness-model-dir \"{model_dir}\"")


if __name__ == "__main__":
    main()
