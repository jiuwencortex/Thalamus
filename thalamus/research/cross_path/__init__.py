"""Phase R3a — Cross-path knowledge transfer.

Research goal: transfer the classifier's learned component co-inclusion patterns
(from Path B) back into the GA fitness function (Path A), so Path A improves
without requiring new real data.

**Research claim (C1 extension):** The logistic regression classifier learns
component co-inclusion patterns that are invisible to individual-component
scoring.  These patterns can transfer back to the GA fitness function, making
Path A better without requiring additional real data.

Implementation
--------------
- :class:`CoInclusionExtractor` — extract pairwise co-inclusion signal from
  the classifier weight matrix W (shape n_components × d_embed).
  Co-inclusion score(c_i, c_j) = cosine_similarity(W[i], W[j]).
- :class:`FitnessAugmentor` — augment GA fitness with the set-level
  co-inclusion score: ``fitness_aug = base + λ × mean_pairwise_cosine(S)``.
- :func:`augment_fitness_config` — convenience: load oracle → augment →
  write ``context_configs_augmented.json``.

CLI::

    thalamus-research cross-path --oracle-dir /oracle --top-pairs 20
    thalamus-research cross-path --oracle-dir /oracle --augment-configs --out augmented.json

Prerequisite: Phase R1 complete (baselines established, evaluation harness ready).
Path B classifier must have been trained (``thalamus-oracle train-classifier``).
"""
from .co_inclusion_extractor import CoInclusionExtractor, ComponentPair
from .fitness_augmentor import FitnessAugmentor, augment_fitness_config

__all__ = [
    "CoInclusionExtractor",
    "ComponentPair",
    "FitnessAugmentor",
    "augment_fitness_config",
]
