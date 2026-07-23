"""Human-readable comparison table from an EvalRun result.

Prints to stdout so it can be piped into other tools or captured in CI logs.

Example output::

    THALAMUS EVALUATION — run_id=a3f2c1b7 — 2025-10-15T09:12:00Z
    oracle_dir: /path/to/oracle   budget: medium   ordering: bookend
    reference:  thalamus          n_repeats: 5     n_queries: 30

    ┌─────────────┬────────┬──────────┬─────────┬─────────┬─────────┬────────────┬──────────┬────────┐
    │ selector    │ path   │ mean_ms  │ p95_ms  │ mean_n  │ jaccard │ precision  │ recall   │ quality│
    ├─────────────┼────────┼──────────┼─────────┼─────────┼─────────┼────────────┼──────────┼────────┤
    │ thalamus    │cluster │    6.12  │   9.44  │    8.3  │    —    │     —      │    —     │  null  │
    │ tfidf       │tfidf   │    2.31  │   3.17  │    8.3  │  0.482  │   0.614    │  0.391   │  null  │
    │ bm25        │bm25    │    1.87  │   2.54  │    8.3  │  0.451  │   0.582    │  0.363   │  null  │
    │ random      │random  │    0.18  │   0.24  │    8.3  │  0.213  │   0.214    │  0.215   │  null  │
    │ all         │all     │    0.05  │   0.07  │   42.0  │  1.000  │   1.000    │  1.000   │  null  │
    └─────────────┴────────┴──────────┴─────────┴─────────┴─────────┴────────────┴──────────┴────────┘

    Note: quality=null — run jiuwenswarm quality pass to fill in task success scores.
"""
from __future__ import annotations

from .result_schema import EvalRun


def print_report(run: EvalRun, file=None) -> None:
    """Print a formatted comparison table for an :class:`EvalRun`."""
    import sys
    out = file or sys.stdout

    def p(line: str = "") -> None:
        print(line, file=out)

    # Header
    p()
    p(f"THALAMUS EVALUATION — run_id={run.run_id} — {run.timestamp}")
    p(f"oracle_dir: {run.oracle_dir}   budget: {run.budget or 'auto'}   ordering: {run.ordering}")
    p(f"reference:  {run.reference_selector:<16}   n_repeats: {run.n_repeats}")
    p()

    # Column widths
    w_sel = max(12, max(len(n) for n in run.selector_names) + 1)
    w_path = 10

    # Header row
    hdr = (
        f"{'selector':<{w_sel}} {'path':<{w_path}} "
        f"{'mean_ms':>9} {'p95_ms':>8} {'mean_n':>7} "
        f"{'jaccard':>8} {'precision':>10} {'recall':>8} {'quality':>8}"
    )
    sep = "-" * len(hdr)
    p(hdr)
    p(sep)

    for name in run.selector_names:
        res = run.results.get(name)
        if res is None:
            p(f"{name:<{w_sel}} — not evaluated")
            continue

        agg = res.aggregate
        ovl = res.overlap_vs_reference

        mean_ms = f"{agg.mean_latency_ms:.2f}" if agg else "—"
        p95_ms = f"{agg.p95_latency_ms:.2f}" if agg else "—"
        mean_n = f"{agg.mean_n_total:.1f}" if agg else "—"
        quality = (
            f"{agg.mean_quality:.4f}" if agg and agg.mean_quality is not None else "null"
        )

        if name == run.reference_selector:
            jaccard = precision = recall = "—"
        elif ovl:
            jaccard = f"{ovl.mean_jaccard:.3f}"
            precision = f"{ovl.mean_precision:.3f}"
            recall = f"{ovl.mean_recall:.3f}"
        else:
            jaccard = precision = recall = "n/a"

        p(
            f"{name:<{w_sel}} {res.active_path:<{w_path}} "
            f"{mean_ms:>9} {p95_ms:>8} {mean_n:>7} "
            f"{jaccard:>8} {precision:>10} {recall:>8} {quality:>8}"
        )

    p(sep)

    # Quality note
    any_quality = any(
        r.aggregate and r.aggregate.mean_quality is not None
        for r in run.results.values()
    )
    if not any_quality:
        p()
        p(
            "Note: quality=null — run a jiuwenswarm quality measurement pass to fill in "
            "task success scores."
        )
        p(
            "      The result JSON has per-query 'quality' placeholders ready to be updated."
        )
    p()
