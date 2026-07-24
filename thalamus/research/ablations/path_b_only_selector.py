"""Ablation R2-D: Path B only (no Path A fallback or warm-start).

Tests Research Contribution C3: does the dual-path architecture outperform
either path in isolation across the full maturity curve?

This selector loads only the classifier (``classifier_current.pkl``) and
returns ``None`` if it is absent.  It has no fallback to Path A (cluster
lookup).  Running this against the full ``ContextSelector`` isolates the
contribution of Path A as warm-start during the first 100–500 turns while
Path B's training data is sparse.

The expected finding:
- Cold start (0 turns): ``path_b_only`` returns None for all queries (not
  ready).  ``ContextSelector`` uses Path A and produces valid selections.
- Early data (100 turns): ``path_b_only`` is active but produces lower
  quality than ``ContextSelector`` (Path A still better on many queries).
- Mature (500+ turns): both converge; ``ContextSelector`` benefit narrows.

See research-plan.md §Phase R2, ablation "Path B only from turn 1".
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PathBOnlySelector:
    """ClassifierSelector with no Path A fallback.

    Returns ``None`` when the classifier is not available.  No cluster lookup
    is attempted at any maturity stage.

    Usage::

        selector = PathBOnlySelector.load(oracle_dir)
        print(selector.is_ready)   # False at cold start (no classifier)
        result = selector.select("Deploy service to k8s")
        # None at cold start; dict once classifier.pkl is present
    """

    def __init__(self, classifier_selector, clusterer) -> None:
        """
        Parameters
        ----------
        classifier_selector:
            ``ClassifierSelector`` instance or ``None`` if not available.
        clusterer:
            ``QueryClusterer`` for embedding — loaded from the cluster pkl so
            we use the same embedding backend as Path B training.
        """
        self._classifier_selector = classifier_selector
        self._clusterer = clusterer

    @classmethod
    def load(cls, oracle_dir: str | Path) -> "PathBOnlySelector":
        """Attempt to load the classifier.  Never raises on missing files.

        Parameters
        ----------
        oracle_dir:
            Directory containing ``classifier_current.pkl`` and
            ``context_configs.pkl`` (for the embedding backend).
        """
        oracle_dir = Path(oracle_dir)

        classifier_selector = None
        clusterer = None

        try:
            from thalamus.context_selectors.by_classifier.classifier_selector import (
                ClassifierSelector,
            )
            classifier_selector = ClassifierSelector.load(oracle_dir)
        except FileNotFoundError:
            logger.debug("PathBOnlySelector: no classifier found in %s", oracle_dir)
        except Exception:
            logger.warning("PathBOnlySelector: classifier load failed", exc_info=True)

        try:
            from thalamus.shared.query_clusterer import QueryClusterer
            pkl_path = oracle_dir / "context_configs.pkl"
            if pkl_path.exists():
                clusterer = QueryClusterer.load(pkl_path)
        except Exception:
            logger.warning("PathBOnlySelector: clusterer load failed", exc_info=True)

        return cls(classifier_selector, clusterer)

    # ── SelectorProtocol ──────────────────────────────────────────────────────

    def select(
        self,
        query: str,
        budget: str | None = None,   # not used by classifier
        ordering: str = "bookend",   # not applied (classifier returns prob order)
    ) -> dict | None:
        """Return classifier prediction or None.

        No Path A fallback.  Returns ``None`` if classifier is unavailable.
        """
        if self._classifier_selector is None or self._clusterer is None:
            return None
        try:
            embedding = self._clusterer.transform(query)
            result = self._classifier_selector.select(embedding)
            result["source"] = "path_b_only"
            return result
        except Exception:
            logger.warning("PathBOnlySelector.select failed", exc_info=True)
            return None

    @property
    def active_path(self) -> str:
        return "path_b_only"

    @property
    def is_ready(self) -> bool:
        return self._classifier_selector is not None and self._clusterer is not None
