# context_selectors/cli_args_parser.py
from __future__ import annotations

import argparse
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m context_selectors",
        description=(
            "Runtime context selection tools.\n\n"
            "  lookup   — cluster-based lookup using context_configs.json\n"
            "  classify — classifier-based selection using classifier.pkl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # Subcommand: cluster-based lookup
    lookup = sub.add_parser(
        "lookup",
        help="Look up the precomputed context config for a query (ClusterSelector)",
    )
    lookup.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing context_configs.json",
    )
    lookup.add_argument(
        "--query", default=None,
        help="Query text: show which cluster and config would be selected",
    )
    lookup.add_argument(
        "--budget", default="medium", choices=["small", "medium", "large", "auto"],
        help="Budget level to look up, or 'auto' to estimate from query text (default: medium)",
    )
    lookup.add_argument(
        "--ordering", default="relevance", choices=["relevance", "bookend", "none"],
        help=(
            "Component ordering strategy: 'relevance' (most-relevant first, default), "
            "'bookend' (most-relevant at edges to combat lost-in-the-middle), "
            "'none' (original insertion order)"
        ),
    )

    # Subcommand: classifier-based selection
    classify = sub.add_parser(
        "classify",
        help="Predict component inclusion from an embedding (ClassifierSelector)",
    )
    classify.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing classifier.pkl",
    )
    classify.add_argument(
        "--embedding", default=None, type=Path,
        help="Path to a .npy file containing the query embedding vector",
    )
    classify.add_argument(
        "--threshold", type=float, default=0.5,
        help="Inclusion threshold for classifier output (default: 0.5)",
    )
    classify.add_argument(
        "--verbose", action="store_true",
        help="Show per-component probability scores",
    )

    return p
