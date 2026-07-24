"""CLI handler: thalamus-research bandit — Phase R3b exploration analysis.

Two subcommands:

  estimate-rate  Derive ε* from oracle action distribution (no turn logs needed)
  convergence    Measure Path B → Path A policy similarity over logged turns

Usage::

    # Derive minimum exploration rate (pure oracle analysis)
    thalamus-research bandit --oracle-dir /oracle --subcommand estimate-rate
    thalamus-research bandit --oracle-dir /oracle --subcommand estimate-rate \\
        --n-min 10 --T-target 500 --out exploration_rate.json

    # Measure Path B convergence to Path A over turn history
    thalamus-research bandit --oracle-dir /oracle --subcommand convergence
    thalamus-research bandit --oracle-dir /oracle --subcommand convergence \\
        --turn-log-dir /logs --window-size 50 --out convergence.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run(args) -> None:  # noqa: ANN001
    """Entry point for ``thalamus-research bandit``."""
    oracle_dir = Path(args.oracle_dir)
    if not oracle_dir.exists():
        logger.error("oracle-dir not found: %s", oracle_dir)
        sys.exit(1)

    subcommand: str = getattr(args, "subcommand", "estimate-rate")
    out_path: Path | None = Path(args.out) if getattr(args, "out", None) else None

    if subcommand == "estimate-rate":
        _run_estimate_rate(oracle_dir, args, out_path)
    elif subcommand == "convergence":
        _run_convergence(oracle_dir, args, out_path)
    else:
        logger.error("Unknown bandit subcommand: %s", subcommand)
        sys.exit(1)


def _run_estimate_rate(oracle_dir: Path, args, out_path: Path | None) -> None:
    n_min: int = int(getattr(args, "n_min", 10))
    T_target: int = int(getattr(args, "T_target", 500))

    try:
        from thalamus.research.bandit.exploration_rate import ExplorationRateEstimator
        estimator = ExplorationRateEstimator.load(oracle_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    result = estimator.estimate(n_min=n_min, T_target=T_target)

    output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Exploration rate analysis written to: {out_path}", file=sys.stderr)
    else:
        # Print human-readable summary
        print(f"\nContextual Bandit — Exploration Rate Derivation (Phase R3b)")
        print(f"Oracle: {oracle_dir}")
        print(f"Parameters: n_min={n_min}, T_target={T_target}")
        print()
        print(f"ε* (minimum exploration rate) = {result.epsilon_star:.4f}")
        print(f"Critical component: '{result.critical_component}' "
              f"(Path A coverage = {result.critical_p_path_a * 100:.1f}%)")
        print()
        print(result.interpretation)
        print()

        # Per-component table (those that need exploration)
        need_exploration = [c for c in result.component_coverage if c.requires_exploration]
        if need_exploration:
            print(f"Components requiring exploration ({len(need_exploration)} of "
                  f"{result.n_components}):")
            print(f"{'Component':<35} {'p_PathA':>8} {'p_total@ε*':>12} {'n_true@T':>10}")
            print("-" * 67)
            for c in need_exploration[:20]:
                print(
                    f"{c.name:<35} {c.p_path_a:>8.3f} "
                    f"{c.p_total_at_epsilon:>12.3f} {c.n_true_at_T:>10.1f}"
                )
            if len(need_exploration) > 20:
                print(f"  ... and {len(need_exploration) - 20} more (use --out to get full list)")
        else:
            print("No components require exploration: Path A provides sufficient coverage.")


def _run_convergence(oracle_dir: Path, args, out_path: Path | None) -> None:
    turn_log_dir: Path | None = (
        Path(args.turn_log_dir) if getattr(args, "turn_log_dir", None) else None
    )
    window_size: int = int(getattr(args, "window_size", 50))
    budget: str = getattr(args, "budget", "medium")

    try:
        from thalamus.research.bandit.convergence import ConvergenceAnalyzer
        analyzer = ConvergenceAnalyzer.load(oracle_dir, turn_log_dir=turn_log_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    try:
        result = analyzer.analyze(window_size=window_size, budget=budget)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Convergence analysis written to: {out_path}", file=sys.stderr)
        result.print_report()
    else:
        result.print_report()
