# oracle_builder/evolutionary/config_builder_step03_fit_clusters.py
# Step 3: fit the query clusterer on all example texts.
# Supports TF-IDF (default) and sentence-transformer backends.
from __future__ import annotations

from ..._shared.query_clusterer import QueryClusterer


class ClusterFitter:
    """Fit a query clusterer on example query texts.

    Parameters
    ----------
    n_clusters:
        Number of K-means clusters.
    max_features:
        TF-IDF vocabulary size (ignored for sentence backend).
    embedder:
        "tfidf" (default) or "sentence".
    sentence_model:
        Sentence-transformer model name (only used when embedder="sentence").
    """

    def __init__(
        self,
        n_clusters: int,
        max_features: int,
        embedder: str = "tfidf",
        sentence_model: str = "all-MiniLM-L6-v2",
    ):
        self._n_clusters = n_clusters
        self._max_features = max_features
        self._embedder = embedder
        self._sentence_model = sentence_model

    def fit(self, all_texts: list[str]) -> QueryClusterer:
        """Fit and return the clusterer."""
        clusterer = QueryClusterer(
            n_clusters=self._n_clusters,
            max_features=self._max_features,
            embedder=self._embedder,
            sentence_model=self._sentence_model,
        )
        clusterer.fit(all_texts)
        return clusterer
