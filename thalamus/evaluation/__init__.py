"""Evaluation harness for research Phase R1.

Runs multiple selectors over a query set, measures latency and component
statistics, and writes a JSON result file that a jiuwenswarm quality-measurement
pass can update with actual task success scores.

Usage::

    from thalamus.evaluation import BenchmarkRunner, EvalRun, print_report
    from thalamus.context_selectors import ContextSelector
    from thalamus.baselines import TFIDFSelector, BM25Selector

    selectors = {
        "thalamus": ContextSelector.load(oracle_dir),
        "tfidf":    TFIDFSelector.load(oracle_dir),
        "bm25":     BM25Selector.load(oracle_dir),
    }
    runner = BenchmarkRunner(selectors, reference_selector="thalamus")
    run = runner.run(queries, oracle_dir=oracle_dir, budget="medium", n_repeats=5)
    print_report(run)

CLI::

    thalamus-select eval \\
        --oracle-dir /oracle \\
        --selectors thalamus tfidf bm25 random \\
        --queries-file task_suite.json \\
        --output results.json
"""

from .result_schema import (
    EvalRun,
    SelectorResult,
    QueryResult,
    AggregateStats,
    OverlapStats,
)
from .benchmark_runner import BenchmarkRunner
from .report import print_report

__all__ = [
    "EvalRun",
    "SelectorResult",
    "QueryResult",
    "AggregateStats",
    "OverlapStats",
    "BenchmarkRunner",
    "print_report",
]
