#!/usr/bin/env python3
"""
R3b — Contextual bandit analysis.

Python equivalent of run_04_bandit.sh — import-and-call, no subprocess boundary.

Two sub-analyses (controlled by MODE env var):

  estimate   Derives ε* — the minimum exploration rate needed so that every
             component receives at least N_MIN logged turns within T_TARGET
             total agent interactions.  Run BEFORE setting the exploration rate.

  convergence  Measures how similar Path B's action distribution has become
               to Path A's over the logged turn history.  Jaccard ≥ 0.85 over
               the last WINDOW turns signals convergence.  Run periodically.

  both (default)  Runs estimate first, then convergence if turn logs exist.

Config via environment variables (see _config.py):
  ORACLE_DIR         Oracle directory                            (required)
  TURN_LOG_DIR       Directory with turns_*.jsonl               (default: ORACLE_DIR/online_logs)
  RESULTS_DIR        Output directory                            (default: ORACLE_DIR/research_results)
  MODE               estimate | convergence | both               (default: both)
  BANDIT_N_MIN       Min samples per component for ε* calc      (default: 10)
  BANDIT_T_TARGET    Target total turns for ε* calc              (default: 500)
  BANDIT_WINDOW      Rolling window for convergence analysis     (default: 50)
  BANDIT_BUDGET      Budget tier for convergence analysis        (default: medium)

Output:
  $RESULTS_DIR/r3b_bandit_epsilon.json      (estimate mode)
  $RESULTS_DIR/r3b_bandit_convergence.json  (convergence mode)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _config import cfg, section  # noqa: E402


def main() -> None:
    mode = os.environ.get("MODE", "both")

    section("R3b — Bandit Analysis")
    print(f"Oracle:        {cfg.oracle_dir}")
    print(f"Turn log dir:  {cfg.turn_log_dir}")
    print(f"Mode:          {mode}")
    print()

    from thalamus_research.cli import main as research_main  # noqa: PLC0415

    if mode in ("both", "estimate"):
        print("── ε* Estimation ──────────────────────────────────────")
        print(f"n_min={cfg.bandit_n_min}  T_target={cfg.bandit_t_target}")
        print()
        sys.argv = [
            "thalamus-research", "bandit",
            "--oracle-dir",  cfg.oracle_dir,
            "--subcommand",  "estimate-rate",
            "--n-min",       cfg.bandit_n_min,
            "--T-target",    cfg.bandit_t_target,
            "--out",         os.path.join(cfg.results_dir, "r3b_bandit_epsilon.json"),
        ]
        research_main()
        print()
        print(f"Result: {cfg.results_dir}/r3b_bandit_epsilon.json")
        print("→ Set TurnLogger(exploration_rate=<epsilon_star>) in jiuwenswarm config.")

    if mode in ("both", "convergence"):
        print()
        print("── Convergence Analysis ──────────────────────────────")
        print(f"Window: {cfg.bandit_window} turns  Budget: {cfg.bandit_budget}")
        print()

        if not os.path.isdir(cfg.turn_log_dir):
            print(f"WARNING: TURN_LOG_DIR not found ({cfg.turn_log_dir}).")
            print("Convergence analysis requires logged agent turns.")
            print("Skipping convergence — run again after turns are available.")
        else:
            sys.argv = [
                "thalamus-research", "bandit",
                "--oracle-dir",   cfg.oracle_dir,
                "--turn-log-dir", cfg.turn_log_dir,
                "--subcommand",   "convergence",
                "--window-size",  cfg.bandit_window,
                "--budget",       cfg.bandit_budget,
                "--out",          os.path.join(cfg.results_dir, "r3b_bandit_convergence.json"),
            ]
            research_main()
            print()
            print(f"Result: {cfg.results_dir}/r3b_bandit_convergence.json")
            print("→ When converged_to_path_a=true, Path B is ready for production solo use.")


if __name__ == "__main__":
    main()
