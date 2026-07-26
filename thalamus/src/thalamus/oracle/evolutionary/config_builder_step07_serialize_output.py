# oracle_builder/evolutionary/config_builder_step07_serialize_output.py
# Step 7: serialize results to JSON and save the clusterer model.
from __future__ import annotations

import json
from pathlib import Path

from ..._shared.query_clusterer import QueryClusterer


class OutputSerializer:
    """Write context_configs.json and the clusterer .pkl model."""

    @staticmethod
    def serialize(result: dict, output_path: Path, clusterer: QueryClusterer) -> None:
        """Write JSON result and pickle the clusterer model alongside it."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                               encoding="utf-8")

        model_path = output_path.with_suffix(".pkl")
        clusterer.save(model_path)

        print(f"\nDone.")
        print(f"  Context configs → {output_path}")
        print(f"  Clusterer model → {model_path}")
