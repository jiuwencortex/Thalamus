"""Component set overlap statistics between a baseline and a reference selector.

Overlap measures are computed per query and then averaged.  They answer:
  - Jaccard: how similar are the two selections overall?
  - Precision: what fraction of the baseline's components appear in the reference?
  - Recall: what fraction of the reference's components the baseline found?

These are *agreement metrics*, not quality metrics.  They measure how close a
baseline is to the Thalamus oracle selection, under the assumption that the oracle
selection is the "ground truth" for what should be selected.

High recall + low precision → baseline over-selects (returns too many components).
Low recall + high precision → baseline under-selects or selects different ones.
"""
from __future__ import annotations

from .result_schema import OverlapStats, QueryResult


def compute_overlap(
    baseline_queries: list[QueryResult],
    reference_queries: list[QueryResult],
) -> OverlapStats:
    """Compute averaged Jaccard / precision / recall over aligned query lists.

    Parameters
    ----------
    baseline_queries:
        Query results from the selector being evaluated.
    reference_queries:
        Query results from the reference selector (e.g. Thalamus).

    Returns
    -------
    OverlapStats
        Averaged over all queries where both selectors returned a non-empty set.

    Notes
    -----
    Queries are matched by index, not by ID, so both lists must be in the same
    order.  The :class:`BenchmarkRunner` ensures this.
    """
    jaccards: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    for base_q, ref_q in zip(baseline_queries, reference_queries):
        a = base_q.component_set
        b = ref_q.component_set
        if not a and not b:
            continue  # both empty — skip rather than dividing by zero

        intersection = len(a & b)
        union = len(a | b)

        jaccards.append(intersection / union if union > 0 else 1.0)
        precisions.append(intersection / len(a) if a else 0.0)
        recalls.append(intersection / len(b) if b else 0.0)

    n = len(jaccards)
    if n == 0:
        return OverlapStats(
            mean_jaccard=0.0, mean_precision=0.0, mean_recall=0.0, n_queries=0
        )

    return OverlapStats(
        mean_jaccard=sum(jaccards) / n,
        mean_precision=sum(precisions) / n,
        mean_recall=sum(recalls) / n,
        n_queries=n,
    )
