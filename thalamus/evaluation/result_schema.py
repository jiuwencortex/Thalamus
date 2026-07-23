"""Result schema for the evaluation harness.

All result types are plain dataclasses with ``to_dict()`` / ``from_dict()``
for JSON serialization.  No external serialization dependency required.

Schema overview::

    EvalRun
    ├── run_id, timestamp, oracle_dir, budget, ordering
    └── results: dict[selector_name, SelectorResult]
        ├── active_path, is_ready
        ├── overlap_vs_reference (Jaccard / precision / recall vs Thalamus)
        └── queries: list[QueryResult]
            ├── id, query, latency_ms, n_total, skills, memory, tools, source
            └── quality: float | None  ← filled in by jiuwenswarm later

Quality scores are ``null`` until a jiuwenswarm quality measurement pass fills
them in.  The harness produces the skeleton; quality measurement is a separate
step so it can run with the full agent stack without re-running selector latency.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class QueryResult:
    """Result for a single (query, selector) pair."""

    id: str                    # e.g. "q00", or a task ID from the input file
    query: str
    latency_ms: float          # median over n_repeats
    skills: list[str]
    memory: list[str]
    tools: list[str]
    source: str                # selector's active_path identifier
    n_total: int = 0           # total components selected (len skills+memory+tools)
    quality: float | None = None  # filled in by jiuwenswarm quality pass

    def __post_init__(self) -> None:
        if not self.n_total:
            self.n_total = len(self.skills) + len(self.memory) + len(self.tools)

    @property
    def component_set(self) -> frozenset[str]:
        return frozenset(self.skills + self.memory + self.tools)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "latency_ms": round(self.latency_ms, 3),
            "n_total": self.n_total,
            "skills": self.skills,
            "memory": self.memory,
            "tools": self.tools,
            "source": self.source,
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QueryResult":
        return cls(
            id=data["id"],
            query=data["query"],
            latency_ms=data["latency_ms"],
            skills=data.get("skills", []),
            memory=data.get("memory", []),
            tools=data.get("tools", []),
            source=data.get("source", ""),
            n_total=data.get("n_total", 0),
            quality=data.get("quality"),
        )


@dataclass
class OverlapStats:
    """Component set overlap between a baseline and a reference selector."""

    mean_jaccard: float      # |A ∩ B| / |A ∪ B| averaged over queries
    mean_precision: float    # |A ∩ B| / |A| — fraction of baseline in reference
    mean_recall: float       # |A ∩ B| / |B| — fraction of reference found by baseline
    n_queries: int

    def to_dict(self) -> dict:
        return {
            "mean_jaccard": round(self.mean_jaccard, 4),
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "n_queries": self.n_queries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OverlapStats":
        return cls(
            mean_jaccard=data["mean_jaccard"],
            mean_precision=data["mean_precision"],
            mean_recall=data["mean_recall"],
            n_queries=data["n_queries"],
        )


@dataclass
class AggregateStats:
    """Per-selector aggregate statistics over all queries."""

    n_queries: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    mean_n_total: float
    mean_quality: float | None = None  # None until quality pass completes

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "median_latency_ms": round(self.median_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "mean_n_total": round(self.mean_n_total, 2),
            "mean_quality": (
                round(self.mean_quality, 4) if self.mean_quality is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AggregateStats":
        return cls(
            n_queries=data["n_queries"],
            mean_latency_ms=data["mean_latency_ms"],
            median_latency_ms=data["median_latency_ms"],
            p95_latency_ms=data["p95_latency_ms"],
            mean_n_total=data["mean_n_total"],
            mean_quality=data.get("mean_quality"),
        )


@dataclass
class SelectorResult:
    """All results for one selector across the full query set."""

    selector_name: str
    active_path: str
    queries: list[QueryResult] = field(default_factory=list)
    aggregate: AggregateStats | None = None
    overlap_vs_reference: OverlapStats | None = None  # None for the reference selector

    def to_dict(self) -> dict:
        return {
            "selector_name": self.selector_name,
            "active_path": self.active_path,
            "aggregate": self.aggregate.to_dict() if self.aggregate else None,
            "overlap_vs_reference": (
                self.overlap_vs_reference.to_dict()
                if self.overlap_vs_reference else None
            ),
            "queries": [q.to_dict() for q in self.queries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SelectorResult":
        return cls(
            selector_name=data["selector_name"],
            active_path=data["active_path"],
            queries=[QueryResult.from_dict(q) for q in data.get("queries", [])],
            aggregate=(
                AggregateStats.from_dict(data["aggregate"])
                if data.get("aggregate") else None
            ),
            overlap_vs_reference=(
                OverlapStats.from_dict(data["overlap_vs_reference"])
                if data.get("overlap_vs_reference") else None
            ),
        )


@dataclass
class EvalRun:
    """Top-level evaluation run result."""

    run_id: str
    timestamp: str
    oracle_dir: str
    selector_names: list[str]
    reference_selector: str             # name of reference ("thalamus" or first selector)
    budget: str | None
    ordering: str
    n_repeats: int                      # latency measurement repetitions per query
    results: dict[str, SelectorResult] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        oracle_dir: str | Path,
        selector_names: list[str],
        reference_selector: str,
        budget: str | None,
        ordering: str,
        n_repeats: int,
    ) -> "EvalRun":
        return cls(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            oracle_dir=str(oracle_dir),
            selector_names=list(selector_names),
            reference_selector=reference_selector,
            budget=budget,
            ordering=ordering,
            n_repeats=n_repeats,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "oracle_dir": self.oracle_dir,
            "selector_names": self.selector_names,
            "reference_selector": self.reference_selector,
            "budget": self.budget,
            "ordering": self.ordering,
            "n_repeats": self.n_repeats,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalRun":
        obj = cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            oracle_dir=data["oracle_dir"],
            selector_names=data["selector_names"],
            reference_selector=data.get("reference_selector", data["selector_names"][0]),
            budget=data.get("budget"),
            ordering=data.get("ordering", "bookend"),
            n_repeats=data.get("n_repeats", 1),
        )
        obj.results = {k: SelectorResult.from_dict(v) for k, v in data.get("results", {}).items()}
        return obj
