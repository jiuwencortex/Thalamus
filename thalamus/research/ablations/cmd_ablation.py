"""CLI handler: thalamus-research ablation — R2 ablation study runner.

Runs the full ablation suite (TopK, NoBookend, SingleBudget, PathBOnly,
plus the full ContextSelector as reference) over a query file and writes
a structured EvalRun JSON.

Usage::

    thalamus-research ablation \\
        --oracle-dir /oracle \\
        --query-file tasks.jsonl \\
        --budget auto \\
        --out results/ablation_run.json \\
        [--no-report]
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Default ablation names to run; all implement SelectorProtocol.
_DEFAULT_ABLATIONS = ["topk", "no_bookend", "single_budget_med", "path_b_only"]


def _load_selectors(oracle_dir: Path) -> dict:
    """Load the full ContextSelector plus all query-time ablation selectors."""
    from thalamus.context_selectors import ContextSelector
    from thalamus.research.ablations import (
        TopKSelector,
        NoBookendSelector,
        SingleBudgetSelector,
        PathBOnlySelector,
    )
    return {
        "thalamus": ContextSelector.load(oracle_dir),
        "topk": TopKSelector.load(oracle_dir),
        "no_bookend": NoBookendSelector.load(oracle_dir),
        "single_budget_med": SingleBudgetSelector.load(oracle_dir, fixed_budget="medium"),
        "single_budget_lg": SingleBudgetSelector.load(oracle_dir, fixed_budget="large"),
        "path_b_only": PathBOnlySelector.load(oracle_dir),
    }


def _load_queries(query_file: Path | None, extra_queries: list[str]) -> list[str]:
    """Combine queries from a JSONL file with any --query CLI args."""
    queries: list[str] = list(extra_queries)
    if query_file is not None:
        if not query_file.exists():
            logger.error("Query file not found: %s", query_file)
            sys.exit(1)
        with query_file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    queries.append(obj.get("query") or obj.get("text") or str(obj))
                except json.JSONDecodeError:
                    queries.append(line)
    if not queries:
        logger.error("No queries supplied. Use --query-file or --query.")
        sys.exit(1)
    return queries


def run(args) -> None:  # noqa: ANN001
    """Entry point for ``thalamus-research ablation``."""
    oracle_dir = Path(args.oracle_dir)
    if not oracle_dir.exists():
        logger.error("oracle-dir not found: %s", oracle_dir)
        sys.exit(1)

    queries = _load_queries(
        Path(args.query_file) if getattr(args, "query_file", None) else None,
        getattr(args, "query", []) or [],
    )

    budget: str | None = None if getattr(args, "budget", "auto") == "auto" else args.budget
    ordering: str = getattr(args, "ordering", "bookend")
    n_repeats: int = getattr(args, "n_repeats", 3)
    out_path: Path | None = Path(args.out) if getattr(args, "out", None) else None
    show_report: bool = not getattr(args, "no_report", False)

    logger.info(
        "Loading ablation selectors from %s (budget=%s, ordering=%s, n_repeats=%d)",
        oracle_dir, budget or "auto", ordering, n_repeats,
    )

    try:
        selectors = _load_selectors(oracle_dir)
    except FileNotFoundError as exc:
        logger.error("Failed to load selectors: %s", exc)
        sys.exit(1)

    from thalamus.research.evaluation import BenchmarkRunner, print_report

    runner = BenchmarkRunner(selectors, reference_selector="thalamus")
    run_result = runner.run(
        queries=queries,
        oracle_dir=str(oracle_dir),
        budget=budget,
        ordering=ordering,
        n_repeats=n_repeats,
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(run_result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Ablation run written to: {out_path}", file=sys.stderr)

    if show_report:
        print_report(run_result)
