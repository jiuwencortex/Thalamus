"""Phase R5 — Cross-deployment knowledge base.

A flat JSON store that maps component fingerprints to aggregate statistics
collected across multiple jiuwenswarm deployments.  When a new deployment is
initialised, the :class:`TransferInitializer` looks up matching fingerprints
and seeds the scoring matrix with prior statistics rather than starting from
scratch.

**KB entry schema**

    {
      "<sha256_fingerprint>": {
        "name_hint": "web_search",          # most recent name seen (informational)
        "n_deployments": 3,                 # deployments where this component appeared
        "n_turns": 412,                     # total turns in which it was included
        "mean_outcome_when_included": 0.73, # mean outcome quality when component in set
        "mean_outcome_when_excluded": 0.61, # mean outcome quality when excluded
        "mean_co_inclusion_score": 0.44,    # mean pairwise co-inclusion weight
        "updated_at": "2025-09-01T12:00:00Z"
      },
      ...
    }

Usage::

    from thalamus_research.meta_learning.knowledge_base import KnowledgeBase

    kb = KnowledgeBase("/shared/knowledge_base.json")

    # Register a deployment's outcome statistics
    kb.update_from_oracle("/oracle_deployment_1")
    kb.save()

    # Query
    entry = kb.get("abc123...")
    if entry:
        print(entry["mean_outcome_when_included"])
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_KB_FILE = "knowledge_base.json"


class KnowledgeBase:
    """Persistent cross-deployment knowledge store.

    Parameters
    ----------
    kb_path:
        Path to the JSON knowledge base file.  Created if absent.
    """

    def __init__(self, kb_path: str | Path) -> None:
        self._path = Path(kb_path)
        self._data: dict[str, dict] = {}
        if self._path.exists():
            self._load()

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def open(cls, kb_dir: str | Path, filename: str = _DEFAULT_KB_FILE) -> "KnowledgeBase":
        """Open (or create) a knowledge base in *kb_dir*."""
        path = Path(kb_dir) / filename
        return cls(path)

    # ── update ────────────────────────────────────────────────────────────────

    def update_from_oracle(self, oracle_dir: str | Path) -> int:
        """Ingest component statistics from a single oracle deployment.

        Reads ``context_configs.json`` for component names and
        ``turns_*.jsonl`` for outcome statistics.

        Parameters
        ----------
        oracle_dir:
            Path to a completed oracle directory.

        Returns
        -------
        int — number of fingerprint entries updated/created.
        """
        from .component_fingerprint import fingerprint_catalog
        from ..set_quality.outcome_dataset import OutcomeDataset

        oracle_dir = Path(oracle_dir)
        fps = fingerprint_catalog(oracle_dir)

        # Load outcome records (best-effort; skip if no turn logs)
        try:
            ds = OutcomeDataset.load(oracle_dir, include_explored=True)
            records = ds.records()
        except Exception:
            logger.debug("No turn logs in %s; skipping outcome stats", oracle_dir)
            records = []

        # Aggregate per-fingerprint statistics
        stats: dict[str, dict] = {fp: {"name_hint": name} for name, fp in fps.items()}
        included_outcomes: dict[str, list[float]] = {fp: [] for fp in fps.values()}
        excluded_outcomes: dict[str, list[float]] = {fp: [] for fp in fps.values()}

        for rec in records:
            rec_set = set(rec.component_set)
            for name, fp in fps.items():
                if fp not in included_outcomes:
                    continue
                if name in rec_set:
                    included_outcomes[fp].append(rec.outcome_quality)
                else:
                    excluded_outcomes[fp].append(rec.outcome_quality)

        updated = 0
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        for fp, meta in stats.items():
            inc = included_outcomes.get(fp, [])
            exc = excluded_outcomes.get(fp, [])

            existing = self._data.get(fp, {})
            prev_deployments = existing.get("n_deployments", 0)
            prev_turns = existing.get("n_turns", 0)

            self._data[fp] = {
                "name_hint": meta["name_hint"],
                "n_deployments": prev_deployments + 1,
                "n_turns": prev_turns + len(inc) + len(exc),
                "mean_outcome_when_included": _mean(inc) if inc else existing.get("mean_outcome_when_included"),
                "mean_outcome_when_excluded": _mean(exc) if exc else existing.get("mean_outcome_when_excluded"),
                "mean_co_inclusion_score": existing.get("mean_co_inclusion_score"),
                "updated_at": now_iso,
            }
            updated += 1

        logger.info(
            "KnowledgeBase.update_from_oracle: %d entries from %s",
            updated, oracle_dir,
        )
        return updated

    def upsert(self, fingerprint: str, entry: dict) -> None:
        """Insert or replace a KB entry."""
        self._data[fingerprint] = dict(entry)

    # ── query ─────────────────────────────────────────────────────────────────

    def get(self, fingerprint: str) -> dict | None:
        """Return the KB entry for *fingerprint*, or None if absent."""
        entry = self._data.get(fingerprint)
        return dict(entry) if entry else None

    def __len__(self) -> int:
        return len(self._data)

    def fingerprints(self) -> list[str]:
        return list(self._data.keys())

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write the knowledge base to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("KnowledgeBase saved: %d entries → %s", len(self._data), self._path)

    def _load(self) -> None:
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
            logger.info("KnowledgeBase loaded: %d entries from %s", len(self._data), self._path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load KnowledgeBase from %s: %s", self._path, exc)
            self._data = {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
