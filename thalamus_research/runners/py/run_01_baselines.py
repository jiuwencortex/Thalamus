#!/usr/bin/env python3
"""
R1 — Baseline evaluation.

Python equivalent of run_01_baselines.sh — import-and-call, no subprocess boundary.

Benchmarks TF-IDF, BM25, dense, random, and all-components selectors against
the full THALAMUS selector over a query set. Writes a JSON result file.

Config via environment variables (see _config.py):
  ORACLE_DIR     Oracle directory with context_configs.json    (required)
  RESULTS_DIR    Output directory                              (default: ORACLE_DIR/research_results)
  QUERIES_FILE   Path to queries JSON or plain-text file       (default: uses inline queries)

Output:
  $RESULTS_DIR/r1_baselines.json   — latency + overlap statistics per selector
  stdout                           — ASCII comparison table
"""
import os
import sys
from pathlib import Path

# Make sibling runners importable when run directly
sys.path.insert(0, str(Path(__file__).parent))
from _config import cfg, section  # noqa: E402


_DEFAULT_QUERIES = [
    "Write a unit test for the payment module",
    "Set up a CI/CD pipeline",
    "Debug a memory leak in production",
    "Refactor the authentication service",
    "Generate API documentation",
]


def main() -> None:
    section("R1 — Baseline Evaluation")
    print(f"Oracle:   {cfg.oracle_dir}")
    print(f"Results:  {cfg.results_dir}/r1_baselines.json")
    if cfg.queries_file:
        print(f"Queries:  {cfg.queries_file}")
    else:
        print("Queries:  (inline defaults)")
    print()

    from thalamus_research.cli import main as research_main  # noqa: PLC0415

    query_args: list[str] = []
    if cfg.queries_file:
        query_args = ["--queries-file", cfg.queries_file]
    else:
        for q in _DEFAULT_QUERIES:
            query_args += ["--query", q]

    sys.argv = [
        "thalamus-research", "eval",
        "--oracle-dir", cfg.oracle_dir,
        "--selectors",  "thalamus", "tfidf", "bm25", "random", "all",
        *query_args,
        "--budget",       "medium",
        "--ordering",     "bookend",
        "--n-repeats",    "5",
        "--reference",    "thalamus",
        "--print-report",
        "--output",       os.path.join(cfg.results_dir, "r1_baselines.json"),
    ]
    research_main()

    print()
    print(f"Done. Result: {cfg.results_dir}/r1_baselines.json")
    print("Next: fill 'quality' fields by running agent tasks, then proceed to R2.")


if __name__ == "__main__":
    main()
