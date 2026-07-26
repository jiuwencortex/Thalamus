#!/usr/bin/env python3
"""
Full Thalamus pipeline: ingest → score → oracle → validate.

Python equivalent of run_all.sh — calls each sub-runner's main() directly,
so the entire pipeline is debuggable in one process.

Steps:
  0. Ingest jiuwenswarm sessions → online_logs/ (run_00_ingest_sessions.py)
  1. Score all components via LLM              (run_01_score.py)
  2. Build evolutionary oracle — no LLM calls  (run_02_oracle.py)
  3. Validate runtime lookup with a test query  (run_04_select.py)

Phase 4 (classifier training) is omitted: it requires enough logged turns
(MIN_TURNS, default 10). Run run_00_ingest_sessions.py + run_03_classifier.py
after step 0 has populated online_logs/ sufficiently.

Config via the same environment variables as the individual runners.
OPENAI_API_KEY must be set in the environment.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_00_ingest_sessions  # noqa: E402
import run_01_score            # noqa: E402
import run_02_oracle           # noqa: E402
import run_04_select           # noqa: E402


def main() -> None:
    print("======================================================")
    print(" THALAMUS — Full Pipeline")
    print("======================================================")
    print()

    run_00_ingest_sessions.main()
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
    print(" To enable the classifier (needs >= 10 logged turns):")
    print("   python runners/py/run_00_ingest_sessions.py  # ingest sessions")
    print("   python runners/py/run_03_classifier.py       # train classifier")
    print("======================================================")


if __name__ == "__main__":
    main()
