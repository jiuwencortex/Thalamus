from dataclasses import dataclass


@dataclass
class RebuildRecommendation:
    """Output of the retraining scheduler."""
    retrain_classifier: bool
    retrain_reasons: list[str]
    rebuild_oracle: bool
    rebuild_reasons: list[str]
