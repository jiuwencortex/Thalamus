#!/usr/bin/env python3
"""
R5 — Cross-deployment meta-learning: extract knowledge then warm-start a new oracle.

Python equivalent of run_06_r5_meta_init.sh — import-and-call, no subprocess boundary.

Run AFTER operating 2+ deployments for at least a few weeks.

Steps:
  1. Extract component quality stats from a mature oracle into the shared KB.
  2. Warm-start a new oracle using fingerprint-matched KB entries.

Config via environment variables:
  SOURCE_ORACLE_DIR   Mature oracle to extract knowledge from  (required unless TRANSFER_ONLY=1)
  NEW_ORACLE_DIR      New oracle to warm-start                 (required unless EXTRACT_ONLY=1)
  KB_PATH             Shared knowledge_base.json               (default: ~/.jiuwenswarm/thalamus_knowledge_base.json)
  EXTRACT_ONLY        1 — skip transfer step
  TRANSFER_ONLY       1 — skip extract step (KB_PATH must already exist)

Prerequisites:
  - thalamus_research installed (uv pip install -e ./thalamus_research)
  - SOURCE_ORACLE_DIR has context_configs.json + turn logs with outcome data
  - NEW_ORACLE_DIR has scoring_matrix_*.json from run_01_score.py
"""
import os
import sys
from pathlib import Path


def main() -> None:
    source_oracle_dir = os.environ.get("SOURCE_ORACLE_DIR", "")
    new_oracle_dir    = os.environ.get("NEW_ORACLE_DIR",    "")
    kb_path           = os.environ.get("KB_PATH",           os.path.expanduser("~/.jiuwenswarm/thalamus_knowledge_base.json"))
    extract_only      = os.environ.get("EXTRACT_ONLY",  "0") == "1"
    transfer_only     = os.environ.get("TRANSFER_ONLY", "0") == "1"

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   R5 — Cross-Deployment Meta-Learning                ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"KB path:       {kb_path}")
    if source_oracle_dir:
        print(f"Source oracle: {source_oracle_dir}")
    if new_oracle_dir:
        print(f"New oracle:    {new_oracle_dir}")
    print()

    from thalamus_research.cli import main as research_main  # noqa: PLC0415
    from thalamus.oracle.cli import main as oracle_main      # noqa: PLC0415

    # Step 1 — Extract knowledge from source oracle
    if not transfer_only:
        if not source_oracle_dir:
            print("ERROR: SOURCE_ORACLE_DIR is not set.", file=sys.stderr)
            print("  SOURCE_ORACLE_DIR=/path/to/mature/oracle python run_06_r5_meta_init.py", file=sys.stderr)
            sys.exit(1)
        if not os.path.isdir(source_oracle_dir):
            print(f"ERROR: SOURCE_ORACLE_DIR not found: {source_oracle_dir}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(os.path.join(source_oracle_dir, "context_configs.json")):
            print(f"ERROR: {source_oracle_dir}/context_configs.json not found.", file=sys.stderr)
            print("Run run_02_oracle.py on the source deployment first.", file=sys.stderr)
            sys.exit(1)

        Path(kb_path).parent.mkdir(parents=True, exist_ok=True)
        results_dir = os.path.join(source_oracle_dir, "research_results")
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        print("══ Step 1: Extracting knowledge into KB ══════════════════")
        print(f"  Source: {source_oracle_dir}")
        print(f"  KB:     {kb_path}")
        print()
        sys.argv = [
            "thalamus-research", "meta-learning",
            "--oracle-dir",  source_oracle_dir,
            "--kb-path",     kb_path,
            "--subcommand",  "extract",
            "--out",         os.path.join(results_dir, "r5_extract.json"),
        ]
        research_main()
        print()
        print(f"KB updated: {kb_path}")
        print(f"Report:     {results_dir}/r5_extract.json")
        print()

    # Step 2 — Warm-start new oracle from KB
    if not extract_only:
        if not new_oracle_dir:
            print("ERROR: NEW_ORACLE_DIR is not set.", file=sys.stderr)
            print("  NEW_ORACLE_DIR=/path/to/new/oracle python run_06_r5_meta_init.py", file=sys.stderr)
            sys.exit(1)
        if not os.path.isdir(new_oracle_dir):
            print(f"ERROR: NEW_ORACLE_DIR not found: {new_oracle_dir}", file=sys.stderr)
            print("Create the oracle dir and run run_01_score.py first.", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(kb_path):
            print(f"ERROR: KB not found at {kb_path}.", file=sys.stderr)
            print("Run Step 1 (extract) first to build the knowledge base.", file=sys.stderr)
            sys.exit(1)

        print("══ Step 2: Warm-starting new oracle from KB ══════════════")
        print(f"  New oracle: {new_oracle_dir}")
        print(f"  KB:         {kb_path}")
        print()
        sys.argv = [
            "thalamus-oracle", "meta-init",
            "--oracle-dir", new_oracle_dir,
            "--kb-path",    kb_path,
        ]
        oracle_main()
        print()
        print(f"transfer_priors.json written to: {new_oracle_dir}")
        print()
        print("══ Step 3: Build oracle with warm-start priors ═══════════")
        print("Run the oracle build — it will read transfer_priors.json automatically:")
        print()
        print(f'  python run_02_oracle.py   # with ORACLE_DIR="{new_oracle_dir}"')
        print()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   R5 meta-init complete.                             ║")
    print("╠══════════════════════════════════════════════════════╣")
    if not extract_only and new_oracle_dir:
        print(f"║  transfer_priors.json → {new_oracle_dir}")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  To add more deployments to the KB, re-run with:    ║")
    print("║    EXTRACT_ONLY=1 SOURCE_ORACLE_DIR=/next/oracle    ║")
    print("║    python run_06_r5_meta_init.py                    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
