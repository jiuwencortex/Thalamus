"""Benchmark runner: measures latency and component statistics for all selectors.

For each (query, selector) pair:
  - Calls ``selector.select()`` ``n_repeats`` times
  - Records the median latency (removes JIT / cache warm-up noise)
  - Records the returned component set

Then computes overlap statistics for each non-reference selector vs the reference.

Quality scores (task success, LLM judge) are NOT measured here.  The result file
has ``quality: null`` placeholders.  A separate jiuwenswarm quality-measurement
pass fills those in after running actual agent tasks.

This separation means latency benchmarks and quality measurements can run
independently, on different machines, at different times.
"""
from __future__ import annotations

import logging
import statistics
import time
from pathlib import Path

from ..baselines.protocol import SelectorProtocol
from .overlap_stats import compute_overlap
from .result_schema import AggregateStats, EvalRun, QueryResult, SelectorResult

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run a set of selectors over a query list and collect performance data.

    Usage::

        from thalamus.context_selectors import ContextSelector
        from thalamus.baselines import TFIDFSelector, BM25Selector

        selectors = {
            "thalamus": ContextSelector.load(oracle_dir),
            "tfidf":    TFIDFSelector.load(oracle_dir),
            "bm25":     BM25Selector.load(oracle_dir),
        }
        runner = BenchmarkRunner(selectors, reference_selector="thalamus")
        run = runner.run(queries, budget="medium", ordering="bookend", n_repeats=5)
        print(run.to_dict())
    """

    def __init__(
        self,
        selectors: dict[str, SelectorProtocol],
        reference_selector: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        selectors:
            Ordered mapping of selector name → selector instance.
        reference_selector:
            Name of the selector whose component sets are used as the ground truth
            for overlap statistics.  Defaults to the first selector in the dict.
            Typically ``"thalamus"``.
        """
        self._selectors = selectors
        self._reference = reference_selector or next(iter(selectors))

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        queries: list[dict],
        oracle_dir: str | Path,
        budget: str | None = None,
        ordering: str = "bookend",
        n_repeats: int = 5,
    ) -> EvalRun:
        """Run all selectors on all queries and return an :class:`EvalRun`.

        Parameters
        ----------
        queries:
            List of ``{"id": str, "query": str}`` dicts.  If ``id`` is absent,
            a sequential ``"q{i:03d}"`` is assigned.
        oracle_dir:
            Path to oracle dir (recorded in the result for traceability).
        budget:
            Budget tier passed to every selector.  ``None`` = selector's default.
        ordering:
            Ordering strategy passed to every selector.
        n_repeats:
            How many times to call ``select()`` per (query, selector) pair.
            Median latency is recorded.  Higher values reduce timer noise.

        Returns
        -------
        EvalRun
            Complete result with aggregate stats and overlap metrics.
        """
        eval_run = EvalRun.new(
            oracle_dir=oracle_dir,
            selector_names=list(self._selectors.keys()),
            reference_selector=self._reference,
            budget=budget,
            ordering=ordering,
            n_repeats=n_repeats,
        )

        # Normalize queries
        norm_queries = [
            {"id": q.get("id", f"q{i:03d}"), "query": q["query"]}
            for i, q in enumerate(queries)
        ]

        # Collect per-selector results
        for sel_name, selector in self._selectors.items():
            logger.info("Benchmarking selector: %s (%d queries)", sel_name, len(norm_queries))
            sel_result = self._run_selector(
                sel_name, selector, norm_queries, budget, ordering, n_repeats
            )
            eval_run.results[sel_name] = sel_result

        # Compute overlap vs reference for each non-reference selector
        ref_result = eval_run.results.get(self._reference)
        if ref_result:
            for sel_name, sel_result in eval_run.results.items():
                if sel_name == self._reference:
                    continue
                sel_result.overlap_vs_reference = compute_overlap(
                    sel_result.queries, ref_result.queries
                )

        return eval_run

    # ── internals ─────────────────────────────────────────────────────────────

    def _run_selector(
        self,
        name: str,
        selector: SelectorProtocol,
        queries: list[dict],
        budget: str | None,
        ordering: str,
        n_repeats: int,
    ) -> SelectorResult:
        active_path = getattr(selector, "active_path", name)
        query_results: list[QueryResult] = []

        for q in queries:
            qid = q["id"]
            text = q["query"]
            latencies: list[float] = []
            last_result: dict | None = None

            for _ in range(max(1, n_repeats)):
                t0 = time.perf_counter()
                try:
                    last_result = selector.select(text, budget=budget, ordering=ordering)
                except Exception:
                    logger.warning("Selector %s failed on query %r", name, text, exc_info=True)
                    last_result = None
                latencies.append((time.perf_counter() - t0) * 1000)

            median_ms = statistics.median(latencies)

            if last_result is None:
                logger.warning("Selector %s returned None for query %r", name, text)
                last_result = {"skills": [], "memory": [], "tools": [], "source": name}

            query_results.append(
                QueryResult(
                    id=qid,
                    query=text,
                    latency_ms=median_ms,
                    skills=last_result.get("skills", []),
                    memory=last_result.get("memory", []),
                    tools=last_result.get("tools", []),
                    source=last_result.get("source", active_path),
                )
            )

        agg = _aggregate(query_results)
        logger.info(
            "  %s: mean_latency=%.2fms p95=%.2fms mean_n=%.1f",
            name, agg.mean_latency_ms, agg.p95_latency_ms, agg.mean_n_total,
        )
        return SelectorResult(
            selector_name=name,
            active_path=active_path,
            queries=query_results,
            aggregate=agg,
        )


def _aggregate(results: list[QueryResult]) -> AggregateStats:
    if not results:
        return AggregateStats(
            n_queries=0,
            mean_latency_ms=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            mean_n_total=0.0,
        )
    latencies = sorted(r.latency_ms for r in results)
    n = len(latencies)
    p95_idx = max(0, int(n * 0.95) - 1)
    return AggregateStats(
        n_queries=n,
        mean_latency_ms=sum(latencies) / n,
        median_latency_ms=statistics.median(latencies),
        p95_latency_ms=latencies[p95_idx],
        mean_n_total=sum(r.n_total for r in results) / n,
    )
