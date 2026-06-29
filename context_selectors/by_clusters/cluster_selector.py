# context_selector/cluster_selector.py
# Query-time context selection: embed query → find nearest cluster → return config.
from __future__ import annotations

import json
import logging
from pathlib import Path

from ...shared.query_clusterer import QueryClusterer
from ...shared.context_orderer import bookend_order

logger = logging.getLogger(__name__)

_BUDGET_NAMES = ("budget_small", "budget_medium", "budget_large")


class ClusterSelector:
    """Load pre-built context_configs.json and select the right config per query.

    Usage:
        selector = ClusterSelector.load(oracle_dir)
        config = selector.select(user_query, budget="medium")
        # config = {"skills": [...], "memory": [...], "tools": [...]}
    """

    def __init__(
        self,
        cluster_data: list[dict],
        clusterer_path: Path,
        budgets: dict[str, int],
    ):
        self._clusters = cluster_data
        self._clusterer_path = clusterer_path
        self._budgets = budgets
        self._clusterer = None  # lazy-loaded

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, oracle_dir: Path) -> "ClusterSelector":
        """Load from oracle_dir/context_configs.json and its companion .pkl file."""
        config_path = oracle_dir / "context_configs.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"context_configs.json not found in {oracle_dir}. "
                "Run: python -m jiuwenswarm.tools.oracle_builder build"
            )
        data = json.loads(config_path.read_text(encoding="utf-8"))
        model_path = config_path.with_suffix(".pkl")
        return cls(
            cluster_data=data["clusters"],
            clusterer_path=model_path,
            budgets=data.get("budgets", {"small": 2000, "medium": 4000, "large": 8000}),
        )

    # ── public API ────────────────────────────────────────────────────────────

    def select(
        self,
        query: str,
        budget: str = "medium",
        ordering: str = "relevance",
    ) -> dict:
        """Return the optimal config for a query at the given budget level.

        Parameters
        ----------
        budget:
            "small" | "medium" | "large"
        ordering:
            How to order the selected components in each list.
            - "relevance" (default): most-relevant first (order stored at build time)
            - "bookend": bookend pattern (most-relevant at edges, least in middle)
            - "none": preserve original insertion order

        Returns
        -------
        dict
            {"skills": [...], "memory": [...], "tools": [...],
             "fitness": float, "context_tokens": int}
        """
        clusterer = self._get_clusterer()
        cluster_id = clusterer.predict(query)
        config = self._lookup(cluster_id, budget)
        return self._apply_ordering(config, ordering)

    def select_all_budgets(self, query: str, ordering: str = "relevance") -> dict[str, dict]:
        """Return configs for all budget levels.

        Returns: {"small": {...}, "medium": {...}, "large": {...}}
        """
        clusterer = self._get_clusterer()
        cluster_id = clusterer.predict(query)
        return {
            budget_key.removeprefix("budget_"): self._apply_ordering(
                self._lookup(cluster_id, budget_key.removeprefix("budget_")),
                ordering,
            )
            for budget_key in _BUDGET_NAMES
        }

    def select_auto(
        self,
        query: str,
        ordering: str = "relevance",
    ) -> dict:
        """Select the optimal config, automatically choosing the budget level.

        Delegates budget estimation to BudgetEstimator which uses query-text
        heuristics (word count, multi-step markers) to pick small/medium/large.

        Parameters
        ----------
        query:
            Raw user query text.
        ordering:
            Same as select().  Default "relevance".

        Returns
        -------
        dict
            {"skills": [...], "memory": [...], "tools": [...],
             "budget": str, "fitness": float, "context_tokens": int}
        """
        from .budget_estimator import BudgetEstimator
        budget = BudgetEstimator().estimate(query)
        config = self.select(query, budget=budget, ordering=ordering)
        return {**config, "budget": budget}

    def is_ready(self) -> bool:
        """Return True if the model file exists and can be loaded."""
        return self._clusterer_path.exists()

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_ordering(config: dict, ordering: str) -> dict:
        """Apply component ordering to a config dict without mutating the original."""
        if ordering == "none":
            return config
        if ordering == "bookend":
            return {
                **config,
                "skills": bookend_order(config.get("skills", [])),
                "memory": bookend_order(config.get("memory", [])),
                "tools":  bookend_order(config.get("tools", [])),
            }
        # "relevance" — already sorted at build time; nothing to do
        return config

    def _lookup(self, cluster_id: int, budget: str) -> dict:
        budget_key = f"budget_{budget}"
        for cluster in self._clusters:
            if cluster["cluster_id"] == cluster_id:
                configs = cluster.get("optimal_configs", {})
                if budget_key in configs:
                    return configs[budget_key]
                # Fallback to any available budget
                for key in _BUDGET_NAMES:
                    if key in configs:
                        logger.warning(
                            "Budget '%s' not found for cluster %d; falling back to '%s'",
                            budget_key, cluster_id, key,
                        )
                        return configs[key]
        # No matching cluster (shouldn't happen)
        logger.warning("Cluster %d not found; returning empty config", cluster_id)
        return {"skills": [], "memory": [], "tools": [], "fitness": 0.0, "context_tokens": 0}

    def _get_clusterer(self):
        if self._clusterer is None:
            if not self._clusterer_path.exists():
                raise FileNotFoundError(
                    f"Clusterer model not found: {self._clusterer_path}. "
                    "Run: python -m jiuwenswarm.tools.oracle_builder build"
                )
            self._clusterer = QueryClusterer.load(self._clusterer_path)
        return self._clusterer
