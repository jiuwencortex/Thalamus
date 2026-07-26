#!/usr/bin/env python3
"""
R2 — Ablation study.

Python equivalent of run_02_ablation.sh — import-and-call, no subprocess boundary.

Runs four ablation selectors that each isolate one THALAMUS design choice,
then compares them against the full system on the same query set.

Ablation variants:
  TopK         (C1) removes GA evolutionary search → greedy TF-IDF × mean_score
  PathBOnly    (C3) removes Path A fallback → classifier-only, cold-start = None
  NoBookend    (C5) removes bookend ordering → relevance-descending order
  SingleBudget (C6) removes adaptive budget → fixed "medium" tier only

Config via environment variables (see _config.py):
  ORACLE_DIR     Oracle directory                              (required)
  RESULTS_DIR    Output directory                              (default: ORACLE_DIR/research_results)
  QUERIES_FILE   Path to queries file                          (default: uses inline queries)

Output:
  $RESULTS_DIR/r2_ablation.json   — per-selector latency + overlap statistics
"""
import os
import sys
from pathlib import Path

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
    section("R2 — Ablation Study")
    print(f"Oracle:   {cfg.oracle_dir}")
    print(f"Results:  {cfg.results_dir}/r2_ablation.json")
    print()

    from thalamus_research.cli import main as research_main  # noqa: PLC0415

    query_args: list[str] = []
    if cfg.queries_file:
        query_args = ["--query-file", cfg.queries_file]
    else:
        for q in _DEFAULT_QUERIES:
            query_args += ["--query", q]

    sys.argv = [
        "thalamus-research", "ablation",
        "--oracle-dir", cfg.oracle_dir,
        *query_args,
        "--budget",    "auto",
        "--ordering",  "bookend",
        "--n-repeats", "3",
        "--out",       os.path.join(cfg.results_dir, "r2_ablation.json"),
    ]
    research_main()

    print()
    print(f"Done. Result: {cfg.results_dir}/r2_ablation.json")
    print("Key metrics: overlap with thalamus (higher = ablation safe to remove),")
    print("latency_ms (lower = design choice adds cost), component count per budget.")


if __name__ == "__main__":
    main()
