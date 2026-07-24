"""Argument parser for the thalamus-research CLI.

All research subcommands live here, separate from the production
``thalamus-select`` CLI which handles only ``lookup`` and ``classify``.

Research subcommands
--------------------
  baseline-lookup   Run a single query through one or more baseline selectors (R1)
  eval              Benchmark all selectors over a query set, write results JSON (R1)
  ablation          Run THALAMUS ablation study (TopK / NoBookend / SingleBudget / PathBOnly) (R2)
  cross-path        Analyze classifier co-inclusion signal for GA fitness transfer (R3a)
  bandit            Estimate optimal exploration rate ε* and measure convergence (R3b)
"""
from __future__ import annotations

import argparse
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m thalamus.research",
        description=(
            "THALAMUS research tools.\n\n"
            "  baseline-lookup — run a query through retrieval baselines (Phase R1)\n"
            "  eval            — benchmark selectors on a query set, write results JSON (Phase R1)\n"
            "  ablation        — run ablation study: TopK/NoBookend/SingleBudget/PathBOnly (Phase R2)\n"
            "  cross-path      — analyze classifier co-inclusion for GA transfer (Phase R3a)\n"
            "  bandit          — estimate optimal exploration rate ε* (Phase R3b)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # ── baseline-lookup ───────────────────────────────────────────────────────
    baseline = sub.add_parser(
        "baseline-lookup",
        help="Run a query through a retrieval baseline (Phase R1)",
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

    # ── eval ──────────────────────────────────────────────────────────────────
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

    # ── ablation (Phase R2) ───────────────────────────────────────────────────
    abl = sub.add_parser(
        "ablation",
        help=(
            "Run R2 ablation study: compare TopK, NoBookend, SingleBudget, PathBOnly "
            "against full THALAMUS"
        ),
    )
    abl.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing scoring matrices, context_configs.json, and optionally classifier",
    )
    abl.add_argument(
        "--query", nargs="*", default=None, metavar="QUERY",
        help="Inline query strings (may be combined with --query-file)",
    )
    abl.add_argument(
        "--query-file", default=None, type=Path,
        help="JSONL file with one query per line: {\"query\": str} or plain text",
    )
    abl.add_argument(
        "--budget", default="auto", choices=["small", "medium", "large", "auto"],
        help="Budget tier passed to every selector (default: auto)",
    )
    abl.add_argument(
        "--ordering", default="bookend", choices=["relevance", "bookend", "none"],
        help="Ordering for GA-based selectors (default: bookend)",
    )
    abl.add_argument(
        "--n-repeats", type=int, default=3,
        help="Latency measurement repetitions per (query, selector) pair (default: 3)",
    )
    abl.add_argument(
        "--out", default=None,
        help="Write EvalRun JSON to this path (default: print to stdout)",
    )
    abl.add_argument(
        "--no-report", action="store_true",
        help="Skip printing the ASCII comparison table",
    )

    # ── cross-path (Phase R3a) ────────────────────────────────────────────────
    cp = sub.add_parser(
        "cross-path",
        help="Analyze classifier co-inclusion signal for GA fitness transfer (Phase R3a)",
    )
    cp.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing context_configs.json and classifier_current.pkl",
    )
    cp.add_argument(
        "--top-pairs", type=int, default=20,
        help="Number of top co-inclusion component pairs to report (default: 20)",
    )
    cp.add_argument(
        "--augment-configs", action="store_true",
        help=(
            "Produce an augmented context_configs.json with fitness_augmented annotations.  "
            "Requires both context_configs.json and classifier_current.pkl in --oracle-dir."
        ),
    )
    cp.add_argument(
        "--lam", type=float, default=0.2,
        help=(
            "Co-inclusion interaction weight λ for fitness augmentation.  "
            "fitness_aug = base_fitness + λ × co_inclusion(S).  Default: 0.2"
        ),
    )
    cp.add_argument(
        "--out", default=None,
        help="Write output JSON to this path (default: print to stdout)",
    )

    # ── bandit (Phase R3b) ────────────────────────────────────────────────────
    bandit = sub.add_parser(
        "bandit",
        help="Estimate optimal exploration rate ε* and measure convergence (Phase R3b)",
    )
    bandit.add_argument(
        "--oracle-dir", required=True, type=Path,
        help="Directory containing context_configs.json (for Path A action distribution)",
    )
    bandit.add_argument(
        "--turn-log-dir", default=None, type=Path,
        help="Directory containing turns_YYYY-WNN.jsonl files (default: oracle-dir)",
    )
    bandit.add_argument(
        "--subcommand", choices=["estimate-rate", "convergence"],
        default="estimate-rate",
        help=(
            "estimate-rate: derive ε* from action distribution divergence (default). "
            "convergence: measure Path B vs Path A policy similarity over turn history."
        ),
    )
    bandit.add_argument(
        "--n-min", type=int, default=10, dest="n_min",
        help="Min samples per class for ε* derivation (estimate-rate only; default: 10)",
    )
    bandit.add_argument(
        "--T-target", type=int, default=500, dest="T_target",
        help="Target total turns for ε* derivation (estimate-rate only; default: 500)",
    )
    bandit.add_argument(
        "--window-size", type=int, default=50, dest="window_size",
        help="Rolling window size for convergence analysis (convergence only; default: 50)",
    )
    bandit.add_argument(
        "--budget", default="medium", choices=["small", "medium", "large"],
        help="Budget tier for Path A reconstruction in convergence analysis (default: medium)",
    )
    bandit.add_argument(
        "--out", default=None,
        help="Write analysis JSON to this path (default: print to stdout)",
    )

    return p
