# context_selectors/cli_args_parser.py
from __future__ import annotations

import argparse
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m thalamus.context_selectors",
        description=(
            "Runtime context selection tools.\n\n"
            "  lookup          — cluster-based lookup using context_configs.json\n"
            "  classify        — classifier-based selection using classifier.pkl\n"
            "  baseline-lookup — retrieval baselines for research evaluation (R1)\n"
            "  eval            — benchmark all selectors on a query set, write results JSON"
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

    # Subcommand: baseline retrieval lookup (Phase R1)
    baseline = sub.add_parser(
        "baseline-lookup",
        help="Run a query through a retrieval baseline (research evaluation Phase R1)",
    )
    baseline.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing scoring matrices (and optionally context_configs.json)",
    )
    baseline.add_argument(
        "--query", default=None,
        help="Query text to evaluate",
    )
    baseline.add_argument(
        "--method", nargs="+",
        choices=["all", "random", "tfidf", "bm25", "dense"],
        default=["tfidf"],
        help=(
            "Baseline method(s) to run.  Multiple values produce a comparison table.\n"
            "  all    — return every component (quality upper bound)\n"
            "  random — random k components (null hypothesis)\n"
            "  tfidf  — TF-IDF cosine similarity top-k\n"
            "  bm25   — BM25 top-k (no extra deps)\n"
            "  dense  — sentence-transformer cosine top-k (requires thalamus[sentence])"
        ),
    )
    baseline.add_argument(
        "--budget", default="medium", choices=["small", "medium", "large", "auto"],
        help="Budget tier (default: medium).  'auto' passes None to let the selector decide.",
    )
    baseline.add_argument(
        "--ordering", default="bookend", choices=["relevance", "bookend", "none"],
        help="Component ordering strategy (default: bookend)",
    )
    baseline.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the 'random' baseline (default: None = system entropy)",
    )

    # Subcommand: full evaluation benchmark (Phase R1)
    ev = sub.add_parser(
        "eval",
        help="Benchmark selectors on a query set and write a results JSON (Phase R1)",
    )
    ev.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing scoring matrices, context_configs.json, and classifier",
    )
    ev.add_argument(
        "--selectors", nargs="+",
        choices=["thalamus", "all", "random", "tfidf", "bm25", "dense"],
        default=["thalamus", "tfidf", "bm25"],
        metavar="SELECTOR",
        help=(
            "Selectors to benchmark.  One or more of: "
            "thalamus all random tfidf bm25 dense  (default: thalamus tfidf bm25)"
        ),
    )
    ev.add_argument(
        "--query", nargs="*", default=None,
        metavar="QUERY",
        help="Inline query strings.  May be combined with --queries-file.",
    )
    ev.add_argument(
        "--queries-file", default=None, type=Path,
        help=(
            "JSON file containing queries: list of {\"id\": str, \"query\": str} dicts, "
            "or a dict with a 'queries' key."
        ),
    )
    ev.add_argument(
        "--budget", default="medium", choices=["small", "medium", "large", "auto"],
        help="Budget tier passed to every selector (default: medium)",
    )
    ev.add_argument(
        "--ordering", default="bookend", choices=["relevance", "bookend", "none"],
        help="Component ordering strategy passed to every selector (default: bookend)",
    )
    ev.add_argument(
        "--n-repeats", type=int, default=5,
        help="Latency measurement repetitions per (query, selector) pair (default: 5)",
    )
    ev.add_argument(
        "--reference", default=None,
        help=(
            "Name of the reference selector for overlap statistics.  "
            "Defaults to 'thalamus' if present, otherwise the first listed selector."
        ),
    )
    ev.add_argument(
        "--output", default=None, type=Path,
        help="Write results to this JSON file (prints JSON to stdout when omitted)",
    )
    ev.add_argument(
        "--print-report", action="store_true",
        help="Print human-readable comparison table to stdout after writing JSON",
    )
    ev.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the 'random' selector baseline",
    )

    return p
