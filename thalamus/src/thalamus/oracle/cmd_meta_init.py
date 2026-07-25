"""R5 — meta-init subcommand: warm-start a new oracle from the cross-deployment KB."""
from __future__ import annotations

import sys
import argparse


def cmd_meta_init(args: argparse.Namespace) -> None:
    """Warm-start *oracle-dir* from the cross-deployment knowledge base.

    Reads ``knowledge_base.json`` at *kb-path*, fingerprints every component in
    *oracle-dir*, and writes ``transfer_priors.json`` so the next
    ``thalamus-oracle evolve`` uses KB-derived prior scores instead of flat priors.
    """
    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.kb_path.exists():
        print(
            f"ERROR: --kb-path does not exist: {args.kb_path}\n"
            "Run the extract step first:\n"
            "  bash thalamus_research/runners/run_06_meta_learning.sh",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from thalamus_research.meta_learning.transfer_initializer import TransferInitializer
    except ImportError:
        print(
            "ERROR: meta-init requires thalamus-research.\n"
            "Install with: uv pip install -e ./thalamus_research",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"R5 — Transfer Initializer")
    print(f"  KB:         {args.kb_path}")
    print(f"  Oracle:     {args.oracle_dir}")
    print()

    ti = TransferInitializer(kb_path=args.kb_path)
    result = ti.transfer(
        new_oracle_dir=args.oracle_dir,
        write_priors=not args.dry_run,
    )
    result.print_report()

    if args.dry_run:
        print("\n[dry-run] transfer_priors.json was NOT written.")
    else:
        priors_path = args.oracle_dir / "transfer_priors.json"
        print(f"\nWritten: {priors_path}")
        print(
            "\nNext: run thalamus-oracle evolve on this oracle.\n"
            "The GA will read transfer_priors.json to warm-start component fitness scores."
        )

    if result.match_rate < 0.5 and result.n_total > 0:
        print(
            f"\nWARNING: Only {result.match_rate * 100:.1f}% of components matched the KB "
            f"({result.n_matched}/{result.n_total}).\n"
            "Consider running the extract step on more existing deployments to grow the KB.",
            file=sys.stderr,
        )
