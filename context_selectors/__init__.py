"""Runtime context selection for agent queries.

Two independent backends:
  - cluster_selector.py     Phase 3 — cluster-based lookup (text in, instant, no model)
  - classifier_selector.py  Phase 4 — trained linear classifier (embedding in, per-component proba)

Utilities:
  - budget_estimator.py     Heuristic budget estimator (auto small/medium/large)
  - context_orderer.py      (in shared/) bookend ordering for lost-in-the-middle mitigation

Usage:
    # Phase 3 (cluster lookup — accepts query text):
    from context_selectors import ClusterSelector
    selector = ClusterSelector.load(oracle_dir)
    config = selector.select(user_query, budget="medium")             # explicit budget
    config = selector.select(user_query, budget="medium", ordering="bookend")  # bookend order
    config = selector.select_auto(user_query)                        # auto budget

    # Phase 4 (classifier — accepts pre-computed embedding):
    from context_selectors import ClassifierSelector
    selector = ClassifierSelector.load(oracle_dir)
    result = selector.select(query_embedding)  # query_embedding: np.ndarray
"""

from .by_clusters.cluster_selector import ClusterSelector
from .by_classifier.classifier_selector import ClassifierSelector
from .by_clusters.budget_estimator import BudgetEstimator

__all__ = ["ClusterSelector", "ClassifierSelector", "BudgetEstimator"]
