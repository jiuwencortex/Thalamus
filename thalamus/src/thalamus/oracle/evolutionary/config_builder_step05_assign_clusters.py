# oracle_builder/evolutionary/config_builder_step05_assign_clusters.py
# Step 5: assign each example text to its cluster and build per-cluster text lists.
from __future__ import annotations

from .component_info import ComponentInfo
from ..._shared.query_clusterer import QueryClusterer


class ClusterAssigner:
    """Assign example texts to clusters and return per-cluster text lists."""

    def assign(self, components: list[ComponentInfo], example_texts_map: dict[str, list[str]],
               clusterer: QueryClusterer) -> dict[int, list[str]]:
        """Return {cluster_id: [texts_in_that_cluster]}."""
        all_texts: list[str] = []
        for comp in components:
            all_texts.extend(example_texts_map.get(comp.name, []))

        assignments = clusterer.predict_batch(all_texts)
        cluster_texts: dict[int, list[str]] = {c: [] for c in range(clusterer.n_clusters)}

        idx = 0
        for comp in components:
            for text in example_texts_map.get(comp.name, []):
                cluster_texts[assignments[idx]].append(text)
                idx += 1

        return cluster_texts
