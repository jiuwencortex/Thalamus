#!/usr/bin/env python3
"""
Full Thalamus pipeline: score → oracle → validate.

Python equivalent of run_all.sh — calls each sub-runner's main() directly,
so the entire pipeline is debuggable in one process.

Steps:
  1. Score all components via LLM             (run_01_score.py)
  2. Build evolutionary oracle — no LLM calls (run_02_oracle.py)
  3. Validate runtime lookup with a test query (run_04_select.py)

Phase 4 (classifier training) is omitted: it requires accumulated agent turn
logs that don't exist on first run. Run run_03_classifier.py separately once
logs are available.

Config via the same environment variables as the individual runners.
OPENAI_API_KEY must be set in the environment.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_01_score   # noqa: E402
import run_02_oracle  # noqa: E402
import run_04_select  # noqa: E402


def main() -> None:
    print("======================================================")
    print(" THALAMUS — Full Pipeline")
    print("======================================================")
    print()

    run_01_score.main()
    print()

    run_02_oracle.main()
    print()

    # Quick sanity-check: resolve a test query against the freshly built oracle
    os.environ.setdefault("QUERY",    "Write a unit test for the payment module")
    os.environ.setdefault("BUDGET",   "auto")
    os.environ.setdefault("ORDERING", "bookend")
    run_04_select.main()

    print()
    print("======================================================")
    print(" Pipeline complete.")
    print(" Next step (after agent accumulates logs):")
    print("   python runners/py/run_03_classifier.py")
    print("======================================================")


if __name__ == "__main__":
    main()
