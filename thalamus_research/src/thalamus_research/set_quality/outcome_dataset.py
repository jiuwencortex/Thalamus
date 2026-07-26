"""Phase R4 — Set-level outcome dataset loader.

Loads logged agent turns and transforms them into (component_set, cluster_id,
outcome_quality) records suitable for training a set-level quality model.

**Turn log format** (``turns_YYYY-WNN.jsonl``):

    {
      "query_embedding": [0.12, -0.34, ...],
      "component_set": ["skill_a", "tool_b"],
      "outcome_quality": 0.84,
      "exploration": {"explored": false}
    }

For each turn, the cluster_id is recovered by applying the saved QueryClusterer
model to the stored query embedding.  This associates each (component_set, outcome)
pair with the query cluster it came from — which is the same information the GA
uses to optimise per-cluster configurations.

Usage::

    from thalamus_research.set_quality.outcome_dataset import OutcomeDataset

    ds = OutcomeDataset.load("/oracle", turn_log_dir="/oracle")
    print(f"{len(ds)} training records")

    for record in ds.records():
        print(record.component_set, record.cluster_id, record.outcome_quality)

    X, y = ds.to_arrays(catalog, extractor=None)  # numpy feature matrix + labels
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_TURN_LOG_GLOB = "turns_*.jsonl"
_MIN_OUTCOME_TURNS = 20   # warn below this threshold


@dataclass
class OutcomeRecord:
    """One (component_set, cluster_id, outcome) training example."""

    component_set: list[str]   # flat list of component names
    cluster_id: int
    outcome_quality: float     # ∈ [0, 1] from OutcomeScorer
    explored: bool             # True if from off-policy exploration


class OutcomeDataset:
    """Collection of set-level outcome records from turn logs.

    Parameters
    ----------
    records:
        List of :class:`OutcomeRecord` instances.
    """

    def __init__(self, records: list[OutcomeRecord]) -> None:
        self._records = records

    @classmethod
    def load(
        cls,
        oracle_dir: str | Path,
        turn_log_dir: str | Path | None = None,
        include_explored: bool = True,
    ) -> "OutcomeDataset":
        """Load turn logs and recover cluster IDs from the oracle clusterer.

        Parameters
        ----------
        oracle_dir:
            Directory containing ``context_configs.pkl`` (saved QueryClusterer).
        turn_log_dir:
            Directory containing ``turns_*.jsonl`` files.  Defaults to
            *oracle_dir* if not specified.
        include_explored:
            Whether to include turns from off-policy exploration.  Default
            True — explored turns are critical training signal for set-level
            quality; excluding them would re-introduce the selection bias the
            exploration mechanism was designed to correct.
        """
        oracle_dir = Path(oracle_dir)
        log_dir = Path(turn_log_dir) if turn_log_dir else oracle_dir

        clusterer = cls._load_clusterer(oracle_dir)
        raw_turns = cls._load_turn_logs(log_dir)

        records: list[OutcomeRecord] = []
        n_skipped = 0
        for turn in raw_turns:
            explored = turn.get("exploration", {}).get("explored", False)
            if not include_explored and explored:
                continue
            embedding = turn.get("query_embedding")
            component_set = turn.get("component_set", [])
            quality = turn.get("outcome_quality")
            if embedding is None or quality is None or not component_set:
                n_skipped += 1
                continue
            stored_cid = turn.get("cluster_id")
            if stored_cid is not None:
                cluster_id = int(stored_cid)
            else:
                cluster_id = cls._predict_cluster(clusterer, embedding)
            records.append(OutcomeRecord(
                component_set=list(component_set),
                cluster_id=int(cluster_id),
                outcome_quality=float(quality),
                explored=bool(explored),
            ))

        if n_skipped:
            logger.warning("Skipped %d turns with missing fields", n_skipped)
        if len(records) < _MIN_OUTCOME_TURNS:
            logger.warning(
                "Only %d outcome records loaded (recommend ≥ %d for reliable training)",
                len(records), _MIN_OUTCOME_TURNS,
            )
        logger.info("OutcomeDataset: %d records from %s", len(records), log_dir)
        return cls(records)

    # ── public API ────────────────────────────────────────────────────────────

    def records(self) -> list[OutcomeRecord]:
        """All outcome records in load order."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def to_arrays(
        self,
        catalog,                  # ComponentCatalog
        extractor=None,           # CoInclusionExtractor | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert records to (X, y) numpy arrays for model training.

        Parameters
        ----------
        catalog:
            :class:`~thalamus_research.baselines.component_catalog.ComponentCatalog`
            for score and type lookups.
        extractor:
            Optional :class:`~thalamus_research.cross_path.co_inclusion_extractor.CoInclusionExtractor`
            for pairwise co-inclusion features.  If None, co-inclusion features
            are set to 0.

        Returns
        -------
        X : np.ndarray of shape (n_records, n_features)
        y : np.ndarray of shape (n_records,)
        """
        from .interaction_features import compute_feature_vector

        rows = []
        ys = []
        for rec in self._records:
            fvec = compute_feature_vector(rec.component_set, rec.cluster_id, catalog, extractor)
            rows.append(fvec)
            ys.append(rec.outcome_quality)

        X = np.array(rows, dtype=float)
        y = np.array(ys, dtype=float)
        return X, y

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_clusterer(oracle_dir: Path):
        from thalamus._shared.query_clusterer import QueryClusterer
        pkl_path = oracle_dir / "context_configs.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"context_configs.pkl not found in {oracle_dir}. "
                "Run: thalamus-oracle evolve"
            )
        return QueryClusterer.load(pkl_path)

    @staticmethod
    def _load_turn_logs(log_dir: Path) -> list[dict]:
        records: list[dict] = []
        log_files = sorted(log_dir.glob(_TURN_LOG_GLOB))
        if not log_files:
            logger.warning("No turn log files in %s (pattern: %s)", log_dir, _TURN_LOG_GLOB)
        for path in log_files:
            try:
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", path.name, exc)
        return records

    @staticmethod
    def _predict_cluster(clusterer, embedding: list[float]) -> int:
        vec = np.array(embedding, dtype=float).reshape(1, -1)
        try:
            return int(clusterer._kmeans.predict(vec)[0])
        except ValueError:
            # Embedding dimension doesn't match clusterer feature space.
            # This happens when the turn was ingested with a different vectorizer
            # than the oracle's QueryClusterer. Fall back to cluster 0.
            logger.debug(
                "_predict_cluster: dimension mismatch (vec=%d, expected=%d), defaulting to 0",
                vec.shape[1],
                clusterer._kmeans.n_features_in_,
            )
            return 0
