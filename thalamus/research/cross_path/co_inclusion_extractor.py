"""Phase R3a — Co-inclusion signal extraction from Path B classifier weights.

The logistic regression classifier (Path B) has a weight matrix W of shape
``(n_components, d_embed)``.  Row ``W[i]`` is the embedding-space direction that
predicts inclusion of component ``c_i``.

Two components c_i and c_j that are frequently included together (co-included)
will have weight vectors that point in similar directions in embedding space,
because they both respond strongly to the same query types.

**Co-inclusion score(c_i, c_j)** = cosine_similarity(W[i], W[j])

A high positive score indicates "both components are typically included for the
same queries" — i.e. they are jointly useful.  A high negative score indicates
"these components substitute for each other" — jointly including them is redundant.

This signal can augment the GA fitness function (Phase R3a): reward component
sets where members have high pairwise co-inclusion scores (joint utility) and
penalize sets where members have negative co-inclusion (redundancy).

Usage::

    from thalamus.research.cross_path.co_inclusion_extractor import CoInclusionExtractor

    extractor = CoInclusionExtractor.load("/oracle")
    matrix = extractor.co_inclusion_matrix()   # shape (n, n), symmetric
    top_pairs = extractor.top_pairs(k=20)      # [(score, name_i, name_j), ...]
    report = extractor.to_dict()               # serializable output
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ComponentPair:
    """A pair of components with their co-inclusion score."""

    name_a: str
    name_b: str
    co_inclusion_score: float   # cosine similarity of weight vectors
    interpretation: str         # "joint" | "redundant" | "neutral"

    def to_dict(self) -> dict:
        return asdict(self)


class CoInclusionExtractor:
    """Extract component co-inclusion signal from a trained classifier.

    Parameters
    ----------
    weight_matrix:
        Array of shape ``(n_components, d_embed)`` — the classifier's W matrix.
    component_names:
        List of component names corresponding to weight_matrix rows.
    """

    _JOINT_THRESHOLD = 0.4      # cosine ≥ this → jointly useful
    _REDUNDANT_THRESHOLD = -0.2  # cosine ≤ this → substitutes

    def __init__(
        self,
        weight_matrix: np.ndarray,
        component_names: list[str],
    ) -> None:
        if weight_matrix.shape[0] != len(component_names):
            raise ValueError(
                f"weight_matrix rows ({weight_matrix.shape[0]}) "
                f"!= len(component_names) ({len(component_names)})"
            )
        self._W = weight_matrix
        self._names = component_names
        self._n = len(component_names)

    @classmethod
    def load(cls, oracle_dir: str | Path) -> "CoInclusionExtractor":
        """Load the classifier weight matrix from *oracle_dir*/classifier_current.pkl.

        Raises
        ------
        FileNotFoundError
            If no classifier is found.  Path B must be trained first
            (``thalamus-oracle train-classifier``).
        """
        oracle_dir = Path(oracle_dir)
        import pickle

        pkl_path = oracle_dir / "classifier_current.pkl"
        if not pkl_path.exists():
            # Fallback to legacy name
            pkl_path = oracle_dir / "classifier.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"No classifier found in {oracle_dir}. "
                "Run: thalamus-oracle train-classifier"
            )
        with pkl_path.open("rb") as fh:
            model = pickle.load(fh)

        # Support both sklearn LogisticRegression and the THALAMUS
        # ComponentInclusionClassifier wrapper.
        if hasattr(model, "coef_"):
            # sklearn LogisticRegression: coef_ shape (n_components, d)
            W = np.array(model.coef_, dtype=float)
            names = list(getattr(model, "component_names_", [f"c{i}" for i in range(W.shape[0])]))
        elif hasattr(model, "_clf") and hasattr(model._clf, "coef_"):
            W = np.array(model._clf.coef_, dtype=float)
            names = list(getattr(model, "component_names", [f"c{i}" for i in range(W.shape[0])]))
        else:
            raise TypeError(
                f"Unrecognised classifier type: {type(model)}.  "
                "Expected sklearn LogisticRegression or ComponentInclusionClassifier."
            )

        logger.info(
            "CoInclusionExtractor: loaded W %s for %d components",
            W.shape, len(names),
        )
        return cls(W, names)

    # ── public API ────────────────────────────────────────────────────────────

    def co_inclusion_matrix(self) -> np.ndarray:
        """Pairwise cosine similarity of weight vectors.

        Returns
        -------
        np.ndarray
            Symmetric matrix of shape ``(n, n)``.  Diagonal is 1.0.
            Range: [-1, 1].  Values ≥ 0.4 suggest joint utility;
            values ≤ -0.2 suggest substitutability.
        """
        # L2-normalise rows
        norms = np.linalg.norm(self._W, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        W_norm = self._W / norms
        return W_norm @ W_norm.T

    def top_pairs(self, k: int = 20, mode: str = "joint") -> list[ComponentPair]:
        """Return the k component pairs with the highest (or lowest) co-inclusion.

        Parameters
        ----------
        k:
            Number of pairs to return.
        mode:
            ``"joint"`` — highest co-inclusion scores (frequently co-included).
            ``"redundant"`` — lowest scores (most substitutable).
        """
        matrix = self.co_inclusion_matrix()
        pairs: list[tuple[float, int, int]] = []
        for i in range(self._n):
            for j in range(i + 1, self._n):
                pairs.append((matrix[i, j], i, j))

        reverse = (mode == "joint")
        pairs.sort(key=lambda x: x[0], reverse=reverse)
        top = pairs[:k]

        return [
            ComponentPair(
                name_a=self._names[i],
                name_b=self._names[j],
                co_inclusion_score=round(float(score), 4),
                interpretation=self._interpret(score),
            )
            for score, i, j in top
        ]

    def set_co_inclusion_score(self, component_names: list[str]) -> float:
        """Mean pairwise co-inclusion score for a candidate component set.

        Used by the fitness augmentor to add a set-level reward term.
        Returns 0.0 for sets with fewer than 2 members.

        Parameters
        ----------
        component_names:
            Names of components in the candidate set.  Components not in the
            classifier are silently ignored.

        Returns
        -------
        float
            Mean cosine similarity of all pairwise weight vectors in the set.
            Range: [-1, 1].
        """
        idx = [self._names.index(n) for n in component_names if n in self._names]
        if len(idx) < 2:
            return 0.0
        matrix = self.co_inclusion_matrix()
        scores: list[float] = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                scores.append(matrix[idx[a], idx[b]])
        return float(sum(scores) / len(scores))

    def to_dict(self, top_k: int = 30) -> dict:
        """Serializable summary: top joint + top redundant pairs + metadata."""
        return {
            "n_components": self._n,
            "weight_matrix_shape": list(self._W.shape),
            "top_joint_pairs": [p.to_dict() for p in self.top_pairs(top_k, mode="joint")],
            "top_redundant_pairs": [p.to_dict() for p in self.top_pairs(top_k, mode="redundant")],
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _interpret(self, score: float) -> str:
        if score >= self._JOINT_THRESHOLD:
            return "joint"
        if score <= self._REDUNDANT_THRESHOLD:
            return "redundant"
        return "neutral"

    @property
    def component_names(self) -> list[str]:
        return list(self._names)

    @property
    def n_components(self) -> int:
        return self._n
