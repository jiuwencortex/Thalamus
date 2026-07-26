# enrichment/score_enricher.py
# Blend synthetic matrix scores with real-world evidence from logged turns.
# Updates scoring_matrix_*.json files in oracle_dir in-place.
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..._shared.outcome_scorer import compute_outcome_quality
from ..._shared.turn_logger import TurnLogger

logger = logging.getLogger(__name__)

# n_real examples per component at which the matrix is fully real-data-driven
_DEFAULT_N_NEEDED = 100


class ScoreEnricher:
    """Blend synthetic F1 scores with real outcome signals from logged turns.

    For each component C:
        real_scores[C] = [outcome_quality(T) for T where C was in context]

        synthetic_weight = max(0, 1 - n_real / n_needed)
        real_weight      = 1 - synthetic_weight

        updated_mean_score = (
            synthetic_weight × original_synthetic_mean
          + real_weight      × mean(real_scores[C])
        )

    The updated mean_score replaces ``mean_score`` in the JSON matrix file.
    A ``real_data`` block is appended to the JSON for traceability.

    This is a soft Bayesian-style replacement: synthetic scores are the prior;
    real data provides evidence that gradually shifts the estimate.
    """

    def __init__(
        self,
        oracle_dir: Path,
        log_dir: Path,
        n_needed: int = _DEFAULT_N_NEEDED,
        max_weeks: int = 8,
    ):
        self._oracle_dir = oracle_dir
        self._log_dir = log_dir
        self._n_needed = n_needed
        self._max_weeks = max_weeks

    def update(self) -> dict[str, int]:
        """Load turns, compute real scores, update all matrix files.

        Returns a summary dict: {component_name: n_real_samples}.
        """
        turn_logger = TurnLogger(self._log_dir)
        turns = turn_logger.load_turns(self._max_weeks)

        if not turns:
            logger.info("No turns logged yet; matrix files unchanged.")
            return {}

        # Collect real scores per component
        real_scores: dict[str, list[float]] = defaultdict(list)
        for turn in turns:
            quality = compute_outcome_quality(turn)
            config = turn.get("context_config", {})
            for cname in config.get("skills", []):
                real_scores[cname].append(quality)
            for cname in config.get("memory_sections", []):
                real_scores[cname].append(quality)
            for cname in config.get("tools", []):
                real_scores[cname].append(quality)

        # Update each matrix file
        patterns = [
            "scoring_matrix_skill_*.json",
            "scoring_matrix_mem_*.json",
            "scoring_matrix_tool_*.json",
        ]
        summary: dict[str, int] = {}

        for pat in patterns:
            for path in sorted(self._oracle_dir.glob(pat)):
                n_updated = self._update_file(path, real_scores)
                if n_updated > 0:
                    logger.info("Updated %s (%d component(s))", path.name, n_updated)

        for name, scores in real_scores.items():
            summary[name] = len(scores)

        logger.info(
            "Matrix update complete. %d component(s) had real data.",
            sum(1 for s in real_scores.values() if s),
        )
        return summary

    # ── private ───────────────────────────────────────────────────────────────

    def _update_file(self, path: Path, real_scores: dict[str, list[float]]) -> int:
        """Update one matrix JSON file. Returns number of components updated."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", path.name, e)
            return 0

        name = data.get("component_name") or data.get("skill_name") or path.stem
        scores = real_scores.get(name, [])
        n_real = len(scores)

        if n_real == 0:
            return 0

        # Compute original synthetic mean from baseline_cross_eval
        rows = data.get("baseline_cross_eval", [])
        f1_values = [r["scores"].get("f1", 0.0) for r in rows if "scores" in r]
        synthetic_mean = sum(f1_values) / len(f1_values) if f1_values else 0.0

        # Blending weights
        synthetic_weight = max(0.0, 1.0 - n_real / self._n_needed)
        real_weight = 1.0 - synthetic_weight
        real_mean = float(np.mean(scores))
        updated_mean = synthetic_weight * synthetic_mean + real_weight * real_mean

        data["real_data"] = {
            "n_real": n_real,
            "real_mean_score": round(real_mean, 4),
            "synthetic_weight": round(synthetic_weight, 4),
            "real_weight": round(real_weight, 4),
            "updated_mean_score": round(updated_mean, 4),
        }

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1
