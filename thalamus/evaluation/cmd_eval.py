"""CLI command: thalamus-select eval

Run all configured selectors over a query set, measure latency and component
statistics, and write a JSON result file.  Quality scores are null until a
separate jiuwenswarm quality-measurement pass fills them in.

Usage::

    # Compare thalamus vs tfidf vs bm25 on inline queries
    thalamus-select eval \\
        --oracle-dir /oracle \\
        --selectors thalamus tfidf bm25 random \\
        --query "Set up CI/CD" "Fix authentication bug" "Add user registration" \\
        --budget medium --ordering bookend --n-repeats 5 \\
        --output results.json

    # Load queries from a JSON file  (list of {"id": str, "query": str})
    thalamus-select eval \\
        --oracle-dir /oracle \\
        --selectors thalamus tfidf bm25 \\
        --queries-file task_suite.json \\
        --output results.json

The output ``results.json`` contains per-query latency + component sets with
``quality: null`` placeholders.  Use ``--print-report`` to also print a summary
table to stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_SELECTOR_CHOICES = ["thalamus", "all", "random", "tfidf", "bm25", "dense"]


def cmd_eval(args: argparse.Namespace) -> None:
    oracle_dir: Path = args.oracle_dir
    budget: str | None = args.budget if args.budget != "auto" else None
    ordering: str = args.ordering
    n_repeats: int = args.n_repeats
    selector_names: list[str] = args.selectors
    output_path: Path | None = args.output
    print_report: bool = args.print_report

    # ── Load queries ──────────────────────────────────────────────────────────
    queries: list[dict] = []
    if args.queries_file:
        try:
            raw = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
            if isinstance(raw, list):
                queries = raw
            else:
                # dict with a "queries" key
                queries = raw.get("queries", raw)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"ERROR: could not read queries file: {exc}", file=sys.stderr)
            sys.exit(1)
    if args.query:
        for i, q in enumerate(args.query):
            queries.append({"id": f"inline_{i:03d}", "query": q})

    if not queries:
        print("ERROR: no queries provided. Use --query or --queries-file.", file=sys.stderr)
        sys.exit(1)

    logger.info("Loaded %d queries", len(queries))

    # ── Load selectors ────────────────────────────────────────────────────────
    selectors = {}
    for name in selector_names:
        try:
            selectors[name] = _load_selector(name, oracle_dir, seed=args.seed)
            logger.info("Loaded selector: %s (active_path=%s)", name,
                        getattr(selectors[name], "active_path", "?"))
        except Exception as exc:
            print(f"WARNING: could not load selector '{name}': {exc}", file=sys.stderr)

    if not selectors:
        print("ERROR: no selectors could be loaded.", file=sys.stderr)
        sys.exit(1)

    # ── Determine reference selector ──────────────────────────────────────────
    reference = args.reference or (
        "thalamus" if "thalamus" in selectors else next(iter(selectors))
    )
    if reference not in selectors:
        reference = next(iter(selectors))

    # ── Run benchmark ─────────────────────────────────────────────────────────
    from .benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(selectors, reference_selector=reference)
    run = runner.run(queries, oracle_dir=oracle_dir, budget=budget,
                     ordering=ordering, n_repeats=n_repeats)

    # ── Write JSON output ─────────────────────────────────────────────────────
    result_dict = run.to_dict()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Results written to: {output_path}", file=sys.stderr)
    else:
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))

    # ── Print human-readable report ───────────────────────────────────────────
    if print_report or output_path:
        from .report import print_report as _print_report
        _print_report(run)


def _load_selector(name: str, oracle_dir: Path, seed: int | None = None):
    if name == "thalamus":
        from ..context_selectors import ContextSelector
        return ContextSelector.load(oracle_dir)
    elif name == "all":
        from ..baselines import AllSelector
        return AllSelector.load(oracle_dir)
    elif name == "random":
        from ..baselines import RandomSelector
        return RandomSelector.load(oracle_dir, seed=seed)
    elif name == "tfidf":
        from ..baselines import TFIDFSelector
        return TFIDFSelector.load(oracle_dir)
    elif name == "bm25":
        from ..baselines import BM25Selector
        return BM25Selector.load(oracle_dir)
    elif name == "dense":
        from ..baselines.dense_selector import DenseSelector
        return DenseSelector.load(oracle_dir)
    else:
        raise ValueError(f"Unknown selector: {name!r}")
