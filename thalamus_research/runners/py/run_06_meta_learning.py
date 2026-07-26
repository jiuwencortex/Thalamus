#!/usr/bin/env python3
"""
R5 — Cross-deployment meta-learning.

Python equivalent of run_06_meta_learning.sh — import-and-call, no subprocess boundary.

Transfers component quality knowledge between deployments using SHA-256
content fingerprints.  Two operations:

  extract   Read component statistics from a completed oracle and add them to
            the shared knowledge base (KB).  Run once per mature deployment.

  transfer  Look up the new deployment's components in the KB and write
            transfer_priors.json to its oracle dir.  Run before the first
            thalamus-oracle evolve on a new deployment.

  both      extract then transfer (requires both SOURCE and NEW oracle dirs)

Config via environment variables (see _config.py):
  ORACLE_DIR        Source oracle to extract from (extract / both)   (required)
  NEW_ORACLE_DIR    New oracle to warm-start (transfer / both)        (required for transfer)
  KB_PATH           Shared knowledge_base.json                        (default: ~/.jiuwenswarm/thalamus_knowledge_base.json)
  RESULTS_DIR       Output directory                                  (default: ORACLE_DIR/research_results)
  MODE              extract | transfer | both                         (default: extract)

Output:
  $KB_PATH                                  (extract — updated KB)
  $RESULTS_DIR/r5_extract.json              (extract report)
  $NEW_ORACLE_DIR/transfer_priors.json      (transfer — warm-start priors)
  $RESULTS_DIR/r5_transfer.json             (transfer report)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _config import cfg, section  # noqa: E402


def main() -> None:
    mode = os.environ.get("MODE", "extract")

    section("R5 — Cross-Deployment Meta-Learning")
    print(f"Oracle:         {cfg.oracle_dir}")
    print(f"KB path:        {cfg.kb_path}")
    print(f"Mode:           {mode}")
    if cfg.new_oracle_dir:
        print(f"New oracle:     {cfg.new_oracle_dir}")
    print()

    from thalamus_research.cli import main as research_main  # noqa: PLC0415

    if mode in ("extract", "both"):
        print(f"── Extract: ingesting {cfg.oracle_dir} into KB ──────────────")
        Path(cfg.kb_path).parent.mkdir(parents=True, exist_ok=True)
        sys.argv = [
            "thalamus-research", "meta-learning",
            "--oracle-dir",  cfg.oracle_dir,
            "--kb-path",     cfg.kb_path,
            "--subcommand",  "extract",
            "--out",         os.path.join(cfg.results_dir, "r5_extract.json"),
        ]
        research_main()
        print()
        print(f"KB updated: {cfg.kb_path}")
        print(f"Report:     {cfg.results_dir}/r5_extract.json")

    if mode in ("transfer", "both"):
        print()
        if not cfg.new_oracle_dir:
            print("ERROR: NEW_ORACLE_DIR must be set for transfer mode.", file=sys.stderr)
            print("  NEW_ORACLE_DIR=/path/to/new/oracle python run_06_meta_learning.py", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(cfg.kb_path):
            print(f"ERROR: KB not found at {cfg.kb_path}.", file=sys.stderr)
            print("Run extract mode first to build the knowledge base.", file=sys.stderr)
            sys.exit(1)

        results_dir_transfer = os.path.join(cfg.new_oracle_dir, "research_results")
        Path(results_dir_transfer).mkdir(parents=True, exist_ok=True)

        print(f"── Transfer: warm-starting {cfg.new_oracle_dir} ─────────────")
        sys.argv = [
            "thalamus-research", "meta-learning",
            "--oracle-dir",  cfg.new_oracle_dir,
            "--kb-path",     cfg.kb_path,
            "--subcommand",  "transfer",
            "--out",         os.path.join(results_dir_transfer, "r5_transfer.json"),
        ]
        research_main()
        print()
        print(f"Priors written: {cfg.new_oracle_dir}/transfer_priors.json")
        print(f"Report:         {results_dir_transfer}/r5_transfer.json")
        print()
        print("Next: run the oracle build on the new deployment.")
        print("The GA will read transfer_priors.json to warm-start fitness scores.")
        print()
        print("Production shorthand (equivalent to the transfer step above):")
        print(f'  thalamus-oracle meta-init \\')
        print(f'    --oracle-dir "{cfg.new_oracle_dir}" \\')
        print(f'    --kb-path "{cfg.kb_path}"')


if __name__ == "__main__":
    main()
