#!/usr/bin/env python3
"""
Runtime: Resolve a query to its optimal context configuration (cluster lookup).

Python equivalent of run_04_select.sh — import-and-call, no subprocess boundary.

Config via environment variables:
  ORACLE_DIR   Directory with context_configs.json   (required)
  QUERY        Query text                             (default: sample query)
  BUDGET       small | medium | large | auto          (default: auto)
  ORDERING     relevance | bookend | none             (default: bookend)

No LLM calls. No network access. Under 10 ms latency.
Requires context_configs.json from run_02_oracle.py.
"""
import os
import sys


def main() -> None:
    oracle_dir = os.path.expanduser(os.environ.get("ORACLE_DIR", "~/.jiuwenswarm/agent/workspace/oracle"))
    query      = os.environ.get("QUERY",      "Set up a CI pipeline for my new microservice")
    budget     = os.environ.get("BUDGET",     "auto")
    ordering   = os.environ.get("ORDERING",   "bookend")

    print("=== Runtime: Context Selection ===")
    print(f"  Oracle dir : {oracle_dir}")
    print(f"  Query      : {query}")
    print(f"  Budget     : {budget}")
    print(f"  Ordering   : {ordering}")
    print()

    from thalamus.selection.cli import main as select_main  # noqa: PLC0415

    sys.argv = [
        "thalamus-select", "lookup",
        "--oracle-dir", oracle_dir,
        "--query",      query,
        "--budget",     budget,
        "--ordering",   ordering,
    ]
    select_main()


if __name__ == "__main__":
    main()
