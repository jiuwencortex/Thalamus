"""Ablation R2-A: Top-k scoring selector (no GA).

Tests Research Contribution C1: does the genetic algorithm find better component
combinations than greedy individual-score ranking?

This selector replicates the *individual* relevance signal used inside the GA
fitness function but without any combinatorial search.  For each query it:

1. Vectorizes the query with the same TF-IDF model used to fit the baselines.
2. Computes ``rank_score_i = mean_score_i × cosine(query, component_texts_i)``
   for every component i.
3. Returns the top-k components by rank_score.

``mean_score_i`` is the blended quality score from the scoring matrices
(``real_data.updated_mean_score`` if present, else mean of baseline_cross_eval F1).
``cosine(...)`` is the TF-IDF cosine similarity of the query to the component's
example texts.

If the GA finds configurations that beat TopKSelector, it demonstrates that
component *interactions* (captured by population-based search) add value beyond
individual quality × relevance.

See research-plan.md §Phase R2, ablation "No GA (top-k scoring)".
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from thalamus.shared.context_orderer import bookend_order
from thalamus.research.baselines.component_catalog import ComponentCatalog, ComponentEntry

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FEATURES = 2000


class TopKSelector:
    """Greedy top-k selector using individual quality scores × query relevance.

    Ablation for Research Contribution C1: represents the "no combinatorial
    search" baseline.  If the GA beats this selector, evolutionary search adds
    measurable value beyond greedy individual ranking.

    Usage::

        selector = TopKSelector.load(oracle_dir)
        result = selector.select("Set up CI/CD pipeline", budget="medium")
        # {"skills": [...], "memory": [...], "tools": [...], "source": "topk"}
    """

    def __init__(
        self,
        catalog: ComponentCatalog,
        vectorizer: TfidfVectorizer,
        doc_matrix: np.ndarray,
        entries_order: list[ComponentEntry],
    ) -> None:
        self._catalog = catalog
        self._vectorizer = vectorizer
        self._doc_matrix = doc_matrix        # (n_components, n_features)
        self._entries_order = entries_order
        # Pre-compute mean_score vector for fast element-wise multiplication
        self._score_vec = np.array([e.mean_score for e in entries_order], dtype=float)

    @classmethod
    def load(
        cls,
        oracle_dir: str | Path,
        max_features: int = _DEFAULT_MAX_FEATURES,
    ) -> "TopKSelector":
        """Load catalog and fit TF-IDF vectorizer on component example texts.

        Parameters
        ----------
        oracle_dir:
            Directory containing scoring matrices and optionally
            ``context_configs.json``.
        max_features:
            TF-IDF vocabulary size.
        """
        oracle_dir = Path(oracle_dir)
        catalog = ComponentCatalog.load(oracle_dir)
        entries = catalog.entries()
        if not entries:
            raise FileNotFoundError(
                f"No scoring matrices found in {oracle_dir}. "
                "Run thalamus-score first."
            )
        docs = [" ".join(e.texts) if e.texts else e.name for e in entries]
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        doc_matrix = vectorizer.fit_transform(docs).toarray()
        logger.info(
            "TopKSelector: fit on %d components, vocab=%d",
            len(entries), len(vectorizer.vocabulary_),
        )
        return cls(catalog, vectorizer, doc_matrix, entries)

    # ── SelectorProtocol ──────────────────────────────────────────────────────

    def select(
        self,
        query: str,
        budget: str | None = None,
        ordering: str = "bookend",
    ) -> dict | None:
        """Return top-k components by ``mean_score × tfidf_cosine``.

        Parameters
        ----------
        query:
            Raw user query text.
        budget:
            Budget tier for k.  ``None`` defaults to ``"medium"``.
        ordering:
            ``"relevance"`` — most-relevant first.
            ``"bookend"``   — bookend reordering.
            ``"none"``      — corpus load order for top-k.
        """
        if not self.is_ready:
            return None

        k = min(self._catalog.count_for_budget(budget), len(self._entries_order))

        query_vec = self._vectorizer.transform([query]).toarray()   # (1, n_features)
        sims = cosine_similarity(query_vec, self._doc_matrix)[0]    # (n_components,)

        # Rank by quality-weighted relevance (mirrors GA fitness without combination search)
        rank_scores = self._score_vec * sims

        if ordering == "none":
            top_indices = set(np.argsort(rank_scores)[-k:])
            chosen = [e.name for i, e in enumerate(self._entries_order) if i in top_indices]
        else:
            top_indices = np.argsort(rank_scores)[::-1][:k]
            chosen = [self._entries_order[i].name for i in top_indices]
            if ordering == "bookend":
                chosen = bookend_order(chosen)

        result = self._catalog.as_result(chosen, source="topk")
        result["n_components"] = k
        return result

    @property
    def active_path(self) -> str:
        return "topk"

    @property
    def is_ready(self) -> bool:
        return len(self._entries_order) > 0
