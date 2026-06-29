# context_selectors/budget_estimator.py
"""Heuristic budget estimator: map a query to small | medium | large.

Used by ClusterSelector.select_auto() when the caller does not want to
specify a budget level manually.

Heuristics (applied in order, first match wins):
  1. Very short query (< 8 words)           → "small"
  2. Multi-step / compound intent markers   → "large"
  3. Long query (> 35 words)               → "large"
  4. Default                               → "medium"
"""
from __future__ import annotations

import re

# Patterns that suggest a complex, multi-step or research-heavy request
_MULTI_STEP_RE = re.compile(
    r"\b("
    r"step[- ]by[- ]step"
    r"|first[\.,].*then"
    r"|and then\b"
    r"|after that\b"
    r"|set up.*and.*configure"
    r"|design.*and.*implement"
    r"|create.*pipeline"
    r"|end[- ]to[- ]end"
    r"|full\s+(workflow|solution|implementation)"
    r"|how do i .*(and|then|also)"
    r")\b",
    re.IGNORECASE,
)

_SHORT_QUERY_WORDS = 8
_LONG_QUERY_WORDS = 35


class BudgetEstimator:
    """Estimate the appropriate token budget for a user query.

    Usage::

        estimator = BudgetEstimator()
        budget = estimator.estimate("Fix the login bug")       # → "small"
        budget = estimator.estimate("Set up CI and deploy")   # → "large"
    """

    def estimate(self, query: str) -> str:
        """Return "small", "medium", or "large" based on query characteristics.

        Parameters
        ----------
        query:
            Raw user query text.

        Returns
        -------
        str
            One of "small", "medium", "large".
        """
        words = query.split()
        n_words = len(words)

        if n_words < _SHORT_QUERY_WORDS:
            return "small"

        if _MULTI_STEP_RE.search(query):
            return "large"

        if n_words > _LONG_QUERY_WORDS:
            return "large"

        return "medium"
