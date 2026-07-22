"""Enrichment: blend real interaction logs into component scoring matrix scores.

  - ScoreEnricher : read interaction_logs turns, update scoring_matrix_*.json

TurnLogger and OutcomeScorer now live in shared
(shared between this package and oracle_builder.policy).

Usage:
    from component_scoring.enrichment import ScoreEnricher
"""

from .score_enricher import ScoreEnricher

__all__ = [
    "ScoreEnricher",
]
