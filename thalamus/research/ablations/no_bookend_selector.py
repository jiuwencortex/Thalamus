"""Ablation R2-B: GA oracle without bookend ordering.

Tests Research Contribution C5: does bookend component ordering improve agent
quality on long-context tasks?

This selector wraps ClusterSelector and forces ``ordering="relevance"``
regardless of the ``ordering`` parameter passed by the caller.  It represents
"THALAMUS Path A without the bookend strategy".

Comparing this to ``thalamus-path-a-bookend`` (ordering=bookend) isolates the
contribution of the lost-in-the-middle mitigation strategy.  The expected
finding is that bookend ordering improves quality on tasks whose selected
context exceeds ~3k tokens.

See research-plan.md §Phase R2, ablation "No bookend (relevance order only)".
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class NoBookendSelector:
    """ClusterSelector wrapper that disables bookend ordering.

    Forces ``ordering="relevance"`` so the component lists are returned in
    descending relevance order (the default at build time) without the
    edge-placement rearrangement.

    Usage::

        selector = NoBookendSelector.load(oracle_dir)
        result = selector.select("Refactor auth module", budget="medium")
        # {"skills": [...most-relevant-first...], ..., "source": "no_bookend"}
    """

    def __init__(self, cluster_selector) -> None:
        self._cluster_selector = cluster_selector

    @classmethod
    def load(cls, oracle_dir: str | Path) -> "NoBookendSelector":
        """Load ClusterSelector from *oracle_dir*.

        Raises
        ------
        FileNotFoundError
            If ``context_configs.json`` is absent.
        """
        from thalamus.context_selectors.by_clusters.cluster_selector import ClusterSelector
        return cls(ClusterSelector.load(Path(oracle_dir)))

    # ── SelectorProtocol ──────────────────────────────────────────────────────

    def select(
        self,
        query: str,
        budget: str | None = None,
        ordering: str = "bookend",   # caller arg ignored — always relevance
    ) -> dict | None:
        """Return optimal GA config with relevance ordering (no bookend).

        The ``ordering`` parameter is accepted for interface compatibility but
        is always overridden to ``"relevance"``.
        """
        if not self.is_ready:
            return None

        effective_budget = budget or "medium"
        try:
            result = self._cluster_selector.select(
                query, budget=effective_budget, ordering="relevance"
            )
            result["source"] = "no_bookend"
            return result
        except Exception:
            logger.warning("NoBookendSelector.select failed", exc_info=True)
            return None

    @property
    def active_path(self) -> str:
        return "no_bookend"

    @property
    def is_ready(self) -> bool:
        return self._cluster_selector.is_ready()
