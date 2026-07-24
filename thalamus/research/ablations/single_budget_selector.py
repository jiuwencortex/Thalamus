"""Ablation R2-C: GA oracle without budget adaptation (fixed budget).

Tests Research Contribution C6: does budget-adaptive selection outperform any
single fixed-budget policy on a mixed-complexity task suite?

This selector wraps ClusterSelector and ignores the caller-supplied ``budget``
parameter, always using a fixed tier.  Default is ``"medium"``.

Comparing this to ``ContextSelector`` (which uses ``BudgetEstimator`` to infer
the appropriate tier) isolates the contribution of budget adaptation.  The
expected finding: on a mixed task suite, auto-budget matches the best fixed
policy per task type while consuming fewer tokens on simple tasks.

Usage::

    # Fixed medium — the most common single-tier deployment choice
    selector = SingleBudgetSelector.load(oracle_dir, fixed_budget="medium")

    # Fixed large — use when unsure; quality upper bound for budget ablation
    selector = SingleBudgetSelector.load(oracle_dir, fixed_budget="large")

See research-plan.md §Phase R2, ablation "GA single-budget (no budget adaptation)".
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_BUDGETS = {"small", "medium", "large"}


class SingleBudgetSelector:
    """ClusterSelector wrapper with a fixed budget tier.

    The ``budget`` argument in :meth:`select` is ignored; the instance always
    uses ``self.fixed_budget``.

    Usage::

        selector = SingleBudgetSelector.load(oracle_dir, fixed_budget="medium")
        result = selector.select("Plan a DB migration")
        # Uses medium budget regardless of query complexity
    """

    def __init__(self, cluster_selector, fixed_budget: str = "medium") -> None:
        if fixed_budget not in _VALID_BUDGETS:
            raise ValueError(f"fixed_budget must be one of {_VALID_BUDGETS}")
        self._cluster_selector = cluster_selector
        self.fixed_budget = fixed_budget

    @classmethod
    def load(
        cls,
        oracle_dir: str | Path,
        fixed_budget: str = "medium",
    ) -> "SingleBudgetSelector":
        """Load ClusterSelector from *oracle_dir* with a fixed budget.

        Parameters
        ----------
        oracle_dir:
            Directory containing ``context_configs.json``.
        fixed_budget:
            ``"small"``, ``"medium"`` (default), or ``"large"``.
        """
        from thalamus.context_selectors.by_clusters.cluster_selector import ClusterSelector
        return cls(ClusterSelector.load(Path(oracle_dir)), fixed_budget=fixed_budget)

    # ── SelectorProtocol ──────────────────────────────────────────────────────

    def select(
        self,
        query: str,
        budget: str | None = None,   # ignored — uses self.fixed_budget
        ordering: str = "bookend",
    ) -> dict | None:
        """Return optimal GA config at the fixed budget tier.

        The ``budget`` parameter is accepted for interface compatibility but is
        always overridden to ``self.fixed_budget``.
        """
        if not self.is_ready:
            return None
        try:
            result = self._cluster_selector.select(
                query, budget=self.fixed_budget, ordering=ordering
            )
            result["source"] = f"single_budget_{self.fixed_budget}"
            return result
        except Exception:
            logger.warning("SingleBudgetSelector.select failed", exc_info=True)
            return None

    @property
    def active_path(self) -> str:
        return f"single_budget_{self.fixed_budget}"

    @property
    def is_ready(self) -> bool:
        return self._cluster_selector.is_ready()
