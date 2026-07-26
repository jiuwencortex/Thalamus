# oracle_builder/evolutionary/config_builder_step04_compute_cluster_centroids.py
# Step 4: set each component's query_centroid from its own example texts.
from __future__ import annotations

import numpy as np

from .component_info import ComponentInfo
from ..._shared.query_clusterer import QueryClusterer


class ClusterCentroidsComputer:
    """Compute each component's TF-IDF centroid from its own example texts."""

    def compute(self, components: list[ComponentInfo],
                example_texts_map: dict[str, list[str]],
                clusterer: QueryClusterer) -> None:
        """Mutate components in-place: set their query_centroid."""
        n_feat = clusterer.n_features
        for comp in components:
            texts = example_texts_map.get(comp.name, [])
            if texts:
                vecs = clusterer.transform_batch(texts)
                comp.query_centroid = vecs.mean(axis=0)
            else:
                comp.query_centroid = np.zeros(n_feat)
