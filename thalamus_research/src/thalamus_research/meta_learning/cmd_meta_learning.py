"""CLI handler: thalamus-research meta-learning — Phase R5 cross-deployment transfer.

Two subcommands:

  extract     Extract component statistics from an oracle and update the KB
  transfer    Warm-start a new deployment from the KB (write transfer_priors.json)

Usage::

    # Extract statistics from an existing deployment into the shared KB
    thalamus-research meta-learning \\
        --oracle-dir /oracle/deployment_1 \\
        --kb-path /shared/knowledge_base.json \\
        --subcommand extract

    # Warm-start a new deployment
    thalamus-research meta-learning \\
        --oracle-dir /oracle/new_deployment \\
        --kb-path /shared/knowledge_base.json \\
        --subcommand transfer \\
        --out transfer_report.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run(args) -> None:  # noqa: ANN001
    """Entry point for ``thalamus-research meta-learning``."""
    oracle_dir = Path(args.oracle_dir)
    if not oracle_dir.exists():
        logger.error("oracle-dir not found: %s", oracle_dir)
        sys.exit(1)

    kb_path_str: str | None = getattr(args, "kb_path", None)
    if not kb_path_str:
        logger.error("--kb-path is required for meta-learning commands")
        sys.exit(1)
    kb_path = Path(kb_path_str)

    subcommand: str = getattr(args, "subcommand", "extract")
    out_path: Path | None = Path(args.out) if getattr(args, "out", None) else None

    if subcommand == "extract":
        _run_extract(oracle_dir, kb_path, out_path)
    elif subcommand == "transfer":
        _run_transfer(oracle_dir, kb_path, out_path)
    else:
        logger.error("Unknown meta-learning subcommand: %s", subcommand)
        sys.exit(1)


# ── subcommand handlers ───────────────────────────────────────────────────────


def _run_extract(oracle_dir: Path, kb_path: Path, out_path: Path | None) -> None:
    try:
        from .knowledge_base import KnowledgeBase
    except ImportError as exc:
        logger.error("Import error: %s", exc)
        sys.exit(1)

    kb = KnowledgeBase(kb_path)
    print(f"\nPhase R5 — Knowledge Base Extract")
    print(f"Oracle:  {oracle_dir}")
    print(f"KB path: {kb_path}")
    print(f"KB entries before: {len(kb)}")

    try:
        n_updated = kb.update_from_oracle(oracle_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    kb.save()
    print(f"KB entries after:  {len(kb)}")
    print(f"Updated/created:   {n_updated}")

    if out_path is not None:
        report = {
            "oracle_dir": str(oracle_dir),
            "kb_path": str(kb_path),
            "n_entries": len(kb),
            "n_updated": n_updated,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Extract report written to: {out_path}", file=sys.stderr)


def _run_transfer(oracle_dir: Path, kb_path: Path, out_path: Path | None) -> None:
    if not kb_path.exists():
        logger.error("KB file not found: %s  (run --subcommand extract first)", kb_path)
        sys.exit(1)

    try:
        from .transfer_initializer import TransferInitializer
    except ImportError as exc:
        logger.error("Import error: %s", exc)
        sys.exit(1)

    try:
        ti = TransferInitializer(kb_path)
        result = ti.transfer(oracle_dir, write_priors=True)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    result.print_report()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nTransfer report written to: {out_path}", file=sys.stderr)
