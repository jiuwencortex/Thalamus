"""Phase R3a — Co-inclusion augmented fitness function for the GA.

The standard GA fitness function is:

    fitness(S, k, B) = mean_score(S, k) × relevance(S, k)

This is a *marginal* score: each component contributes independently.
It cannot model component interactions — two components that are jointly
necessary but individually mediocre both receive low fitness.

The co-inclusion augmented fitness adds a set-level interaction term:

    fitness_aug(S, k, B) = fitness(S, k, B) + λ × co_inclusion(S)

where ``co_inclusion(S)`` is the mean pairwise co-inclusion score derived
from the classifier's weight matrix (see :class:`CoInclusionExtractor`).

**Interpretation:**
- High ``co_inclusion(S)`` → the classifier has learned that members of S
  respond to the same queries → they are jointly useful → reward the set.
- Low / negative ``co_inclusion(S)`` → members are substitutes → penalize.

``λ`` (default 0.2) controls the strength of the interaction term.  At
``λ=0`` the augmented fitness reduces to the original fitness.

This module provides:

1. :class:`FitnessAugmentor` — wraps a ``CoInclusionExtractor`` and computes
   the augmentation term for any component set.

2. :func:`augment_fitness_config` — post-processes a ``context_configs.json``
   dict by re-scoring each cluster's optimal configs with the augmented
   fitness, potentially reordering the Pareto front.  This is the practical
   deployment artifact: the oracle re-runs with ``--use-classifier-prior``
   which calls this function.

Usage::

    from thalamus.research.cross_path.co_inclusion_extractor import CoInclusionExtractor
    from thalamus.research.cross_path.fitness_augmentor import FitnessAugmentor

    extractor = CoInclusionExtractor.load("/oracle")
    augmentor = FitnessAugmentor(extractor, lam=0.2)

    # Compute augmented fitness for a candidate set
    base_fitness = 0.73
    components = ["skill_ci", "skill_docker", "mem_project"]
    aug_fitness = augmentor.augment(base_fitness, components)

    # Re-score an entire context_configs.json
    import json
    config = json.loads(Path("/oracle/context_configs.json").read_text())
    updated = augmentor.rerank_configs(config)
    Path("/oracle/context_configs_augmented.json").write_text(
        json.dumps(updated, indent=2)
    )
"""
from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .co_inclusion_extractor import CoInclusionExtractor

logger = logging.getLogger(__name__)

_DEFAULT_LAMBDA = 0.2
_BUDGET_KEYS = ("budget_small", "budget_medium", "budget_large")


class FitnessAugmentor:
    """Augment GA fitness scores with classifier co-inclusion signal.

    Parameters
    ----------
    extractor:
        Loaded :class:`CoInclusionExtractor` providing pairwise co-inclusion
        scores from the classifier weight matrix.
    lam:
        Interaction weight λ.  The augmented fitness is::

            fitness_aug = base_fitness + lam * co_inclusion_score(component_set)

        Default 0.2.  At 0.0 this reduces to the original fitness.
    """

    def __init__(self, extractor: "CoInclusionExtractor", lam: float = _DEFAULT_LAMBDA) -> None:
        self._extractor = extractor
        self.lam = lam

    # ── public API ────────────────────────────────────────────────────────────

    def augment(self, base_fitness: float, component_names: list[str]) -> float:
        """Compute augmented fitness for a candidate component set.

        Parameters
        ----------
        base_fitness:
            The original GA fitness value for this set.
        component_names:
            Flat list of component names in the candidate set.

        Returns
        -------
        float
            ``base_fitness + lam × co_inclusion_score(component_names)``.
        """
        interaction = self._extractor.set_co_inclusion_score(component_names)
        return base_fitness + self.lam * interaction

    def rerank_configs(self, context_configs: dict) -> dict:
        """Re-score all cluster configs in a ``context_configs.json`` dict.

        For each cluster × budget tier, computes the augmented fitness and
        stores it in ``"fitness_augmented"``.  The config with the highest
        augmented fitness per budget tier is promoted to
        ``"best_augmented_config"``.

        The original ``"optimal_configs"`` structure is preserved; this
        function adds annotations rather than replacing the existing oracle.

        Parameters
        ----------
        context_configs:
            Parsed ``context_configs.json`` dict.

        Returns
        -------
        dict
            Deep copy of *context_configs* with augmented fitness annotations.
        """
        updated = copy.deepcopy(context_configs)
        n_augmented = 0

        for cluster in updated.get("clusters", []):
            configs = cluster.get("optimal_configs", {})
            for bkey in _BUDGET_KEYS:
                cfg = configs.get(bkey)
                if cfg is None:
                    continue
                all_components = (
                    cfg.get("skills", [])
                    + cfg.get("memory", [])
                    + cfg.get("tools", [])
                )
                base = cfg.get("fitness", 0.0)
                aug = self.augment(base, all_components)
                cfg["fitness_augmented"] = round(aug, 6)
                cfg["co_inclusion_score"] = round(
                    self._extractor.set_co_inclusion_score(all_components), 4
                )
                n_augmented += 1

        logger.info(
            "FitnessAugmentor: re-scored %d cluster×budget configs (λ=%.2f)",
            n_augmented, self.lam,
        )
        updated["_augmentation"] = {
            "lambda": self.lam,
            "n_classifier_components": self._extractor.n_components,
            "method": "co_inclusion_cosine",
        }
        return updated


def augment_fitness_config(
    oracle_dir: str,
    lam: float = _DEFAULT_LAMBDA,
    out_path: str | None = None,
) -> dict:
    """Convenience function: load oracle, augment, optionally write.

    Parameters
    ----------
    oracle_dir:
        Path to the oracle directory (must contain both
        ``context_configs.json`` and ``classifier_current.pkl``).
    lam:
        Interaction weight.
    out_path:
        If provided, write the augmented config to this path.

    Returns
    -------
    dict
        The augmented ``context_configs`` dict.
    """
    import json
    from pathlib import Path
    from .co_inclusion_extractor import CoInclusionExtractor

    oracle_dir_path = Path(oracle_dir)
    config_path = oracle_dir_path / "context_configs.json"
    if not config_path.exists():
        raise FileNotFoundError(f"context_configs.json not found in {oracle_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    extractor = CoInclusionExtractor.load(oracle_dir_path)
    augmentor = FitnessAugmentor(extractor, lam=lam)
    updated = augmentor.rerank_configs(config)

    if out_path is not None:
        Path(out_path).write_text(
            json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Augmented config written to %s", out_path)

    return updated
