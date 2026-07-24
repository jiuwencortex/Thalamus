"""Phase R3b — Path B policy convergence measurement.

Measures how similar Path B's selections are to Path A's selections as
logged turns accumulate.  This is the empirical counterpart to the
theoretical ε* derivation in :mod:`exploration_rate`.

**What we are measuring**

The core claim is:
- Without exploration (ε=0): Path B converges to Path A's policy.
- With exploration (ε = ε*): Path B learns a strictly *different* (and
  hopefully better) policy on queries where Path A's cluster boundaries
  are imprecise.

To measure this, we compare the component sets that Path A would choose
vs. the component sets that Path B actually chooses at each point in the
turn history.  The agreement metric is Jaccard similarity:

    agreement(t) = |S_A(q_t) ∩ S_B(q_t)| / |S_A(q_t) ∪ S_B(q_t)|

If Path B converges to Path A, agreement approaches 1.0 as t → ∞.
If Path B learns a different policy, agreement plateaus below 1.0.

**Input format**

Turn logs are ``turns_YYYY-WNN.jsonl`` files.  Each line is a JSON object:

    {
      "query_embedding": [0.12, -0.34, ...],   # dense vector
      "component_set": ["skill_a", "tool_b"],   # what was actually selected
      "outcome_quality": 0.84,
      "exploration": {"explored": true}         # or false
    }

We reconstruct Path A's selection by calling ``ClusterSelector.select()``
on the query embedding (via clusterer.predict()) and comparing with the
logged component set.

**Usage**::

    from thalamus.research.bandit.convergence import ConvergenceAnalyzer

    analyzer = ConvergenceAnalyzer.load("/oracle", turn_log_dir="/oracle")
    result = analyzer.analyze(window_size=50)
    result.print_report()
    data = result.to_dict()
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

_TURN_LOG_GLOB = "turns_*.jsonl"


@dataclass
class ConvergencePoint:
    """Agreement between Path B and Path A at a specific turn count."""

    turn_count: int
    mean_jaccard: float   # mean Jaccard similarity over the window
    std_jaccard: float    # std dev of Jaccard over the window
    n_explored: int       # number of off-policy (explored) turns in window

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConvergenceResult:
    """Full convergence curve for Path B vs Path A."""

    oracle_dir: str
    n_turns_total: int
    n_turns_explored: int
    exploration_fraction: float     # n_explored / n_total
    convergence_curve: list[ConvergencePoint]
    final_agreement: float          # mean Jaccard at last window
    converged_to_path_a: bool       # True if final_agreement ≥ 0.85

    def to_dict(self) -> dict:
        d = asdict(self)
        d["convergence_curve"] = [p.to_dict() for p in self.convergence_curve]
        return d

    def print_report(self) -> None:
        """Print an ASCII convergence curve to stdout."""
        print(f"\nPath B → Path A convergence analysis")
        print(f"Oracle: {self.oracle_dir}")
        print(f"Total turns: {self.n_turns_total}  |  "
              f"Explored: {self.n_turns_explored} "
              f"({self.exploration_fraction * 100:.1f}%)")
        print(f"Final agreement (Jaccard): {self.final_agreement:.3f}  |  "
              f"Converged to Path A: {self.converged_to_path_a}")
        print()
        print(f"{'Turns':>8}  {'Jaccard':>8}  {'±StdDev':>8}  {'Explored':>10}")
        print("-" * 42)
        for pt in self.convergence_curve:
            print(
                f"{pt.turn_count:>8}  "
                f"{pt.mean_jaccard:>8.3f}  "
                f"{pt.std_jaccard:>8.3f}  "
                f"{pt.n_explored:>10}"
            )
        if self.converged_to_path_a:
            print(
                "\nResult: Path B has converged to Path A's policy "
                f"(mean Jaccard ≥ 0.85).  "
                "Increase exploration rate ε or review off-policy sampling."
            )
        else:
            print(
                "\nResult: Path B has diverged from Path A "
                f"(mean Jaccard = {self.final_agreement:.3f} < 0.85).  "
                "Path B is learning a genuinely different policy."
            )


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


class ConvergenceAnalyzer:
    """Measure Path B policy convergence vs. Path A over logged turns.

    Parameters
    ----------
    cluster_selector:
        Loaded :class:`~thalamus.context_selectors.by_clusters.ClusterSelector`.
    clusterer:
        Loaded :class:`~thalamus.shared.query_clusterer.QueryClusterer`.
    turn_records:
        List of raw turn log dicts loaded from JSONL files.
    oracle_dir:
        String path to oracle dir (for reporting).
    """

    def __init__(
        self,
        cluster_selector,
        clusterer,
        turn_records: list[dict],
        oracle_dir: str,
    ) -> None:
        self._cluster_selector = cluster_selector
        self._clusterer = clusterer
        self._turns = turn_records
        self._oracle_dir = oracle_dir

    @classmethod
    def load(
        cls,
        oracle_dir: str | Path,
        turn_log_dir: str | Path | None = None,
    ) -> "ConvergenceAnalyzer":
        """Load ClusterSelector, QueryClusterer, and all turn logs.

        Parameters
        ----------
        oracle_dir:
            Directory with ``context_configs.json`` + ``context_configs.pkl``.
        turn_log_dir:
            Directory containing ``turns_*.jsonl`` files.
            Defaults to *oracle_dir* if not specified.

        Raises
        ------
        FileNotFoundError
            If ``context_configs.json`` is absent (oracle not built yet).
        """
        oracle_dir = Path(oracle_dir)
        log_dir = Path(turn_log_dir) if turn_log_dir else oracle_dir

        from thalamus.context_selectors.by_clusters.cluster_selector import ClusterSelector
        from thalamus.shared.query_clusterer import QueryClusterer

        cluster_selector = ClusterSelector.load(oracle_dir)
        pkl_path = oracle_dir / "context_configs.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"context_configs.pkl not found in {oracle_dir}")
        clusterer = QueryClusterer.load(pkl_path)

        turn_records = cls._load_turn_logs(log_dir)
        logger.info(
            "ConvergenceAnalyzer: loaded %d turn records from %s",
            len(turn_records), log_dir,
        )
        return cls(cluster_selector, clusterer, turn_records, str(oracle_dir))

    # ── public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        window_size: int = 50,
        budget: str = "medium",
        ordering: str = "none",
    ) -> ConvergenceResult:
        """Compute the Path B → Path A convergence curve.

        Parameters
        ----------
        window_size:
            Rolling window size for computing mean/std Jaccard.
        budget:
            Budget tier to use when calling ClusterSelector.  Default "medium".
        ordering:
            Ordering strategy for ClusterSelector.  Use "none" to compare raw
            component sets without ordering effects.

        Returns
        -------
        ConvergenceResult
        """
        if not self._turns:
            raise ValueError("No turn records loaded. Check turn_log_dir.")

        n_explored = sum(
            1 for t in self._turns
            if t.get("exploration", {}).get("explored", False)
        )

        curve: list[ConvergencePoint] = []

        for end in range(window_size, len(self._turns) + 1, max(1, window_size // 2)):
            window = self._turns[max(0, end - window_size):end]
            jaccards: list[float] = []
            n_exp_in_window = 0

            for record in window:
                embedding = record.get("query_embedding")
                logged_set = set(record.get("component_set", []))

                if not embedding:
                    continue

                if record.get("exploration", {}).get("explored", False):
                    n_exp_in_window += 1

                # Reconstruct what Path A would have chosen for this query
                try:
                    path_a_result = self._select_path_a_for_embedding(
                        embedding, budget, ordering
                    )
                    path_a_set = set(
                        path_a_result.get("skills", [])
                        + path_a_result.get("memory", [])
                        + path_a_result.get("tools", [])
                    )
                except Exception:
                    logger.debug("Path A lookup failed for a turn", exc_info=True)
                    continue

                jaccards.append(_jaccard(logged_set, path_a_set))

            if jaccards:
                mean_j = sum(jaccards) / len(jaccards)
                var_j = sum((x - mean_j) ** 2 for x in jaccards) / len(jaccards)
                std_j = var_j ** 0.5
                curve.append(ConvergencePoint(
                    turn_count=end,
                    mean_jaccard=round(mean_j, 4),
                    std_jaccard=round(std_j, 4),
                    n_explored=n_exp_in_window,
                ))

        final_agreement = curve[-1].mean_jaccard if curve else 0.0

        return ConvergenceResult(
            oracle_dir=self._oracle_dir,
            n_turns_total=len(self._turns),
            n_turns_explored=n_explored,
            exploration_fraction=round(n_explored / max(len(self._turns), 1), 4),
            convergence_curve=curve,
            final_agreement=final_agreement,
            converged_to_path_a=(final_agreement >= 0.85),
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _select_path_a_for_embedding(
        self,
        embedding: list[float],
        budget: str,
        ordering: str,
    ) -> dict:
        """Reconstruct Path A's selection for a stored embedding.

        The embedding vector is passed to QueryClusterer.predict_from_vector()
        to find the nearest cluster, then ClusterSelector.select() retrieves
        the precomputed config.
        """
        import numpy as np
        vec = np.array(embedding, dtype=float).reshape(1, -1)
        # QueryClusterer.predict() accepts text; we need predict from vector.
        # Use the internal kmeans model directly.
        cluster_id = int(self._clusterer._model.predict(vec)[0])
        return self._cluster_selector._lookup(cluster_id, budget)

    @staticmethod
    def _load_turn_logs(log_dir: Path) -> list[dict]:
        records: list[dict] = []
        log_files = sorted(log_dir.glob(_TURN_LOG_GLOB))
        if not log_files:
            logger.warning("No turn log files found in %s (pattern: %s)", log_dir, _TURN_LOG_GLOB)
            return records
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
