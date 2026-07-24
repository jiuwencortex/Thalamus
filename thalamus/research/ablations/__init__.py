"""Phase R2 — Ablation selectors for THALAMUS component isolation study.

Each selector isolates one design choice by removing or disabling it, allowing
the ``BenchmarkRunner`` to measure the quality degradation attributable to that
choice.

Ablation variants
-----------------
- :class:`TopKSelector`          — No GA: greedy quality×relevance ranking (C1)
- :class:`NoBookendSelector`     — GA without bookend ordering (C5)
- :class:`SingleBudgetSelector`  — GA with fixed budget, no adaptation (C6)
- :class:`PathBOnlySelector`     — Path B only, no Path A fallback (C3)

All implement :class:`~thalamus.research.baselines.protocol.SelectorProtocol`.

Not implemented as query-time selectors (require build-time changes):
- "No enrichment" ablation: re-run ``thalamus-score --type all`` (skip enrich)
  into a separate oracle directory, then compare using the standard
  ``ContextSelector.load(no_enrich_oracle_dir)``.
- "No exploration (ε=0)" ablation: filter turn logs to non-explored turns before
  training the classifier.  See :mod:`thalamus.research.bandit` for the
  exploration-rate analysis tooling.

Typical usage in ``BenchmarkRunner``::

    from thalamus.context_selectors import ContextSelector
    from thalamus.research.ablations import (
        TopKSelector, NoBookendSelector, SingleBudgetSelector, PathBOnlySelector
    )

    selectors = {
        "thalamus":          ContextSelector.load(oracle_dir),
        "topk":              TopKSelector.load(oracle_dir),
        "no_bookend":        NoBookendSelector.load(oracle_dir),
        "single_budget_med": SingleBudgetSelector.load(oracle_dir, "medium"),
        "path_b_only":       PathBOnlySelector.load(oracle_dir),
    }

CLI::

    thalamus-research ablation \\
        --oracle-dir /oracle \\
        --query-file tasks.jsonl \\
        --out results/ablation_run.json
"""
from .topk_selector import TopKSelector
from .no_bookend_selector import NoBookendSelector
from .single_budget_selector import SingleBudgetSelector
from .path_b_only_selector import PathBOnlySelector

__all__ = [
    "TopKSelector",
    "NoBookendSelector",
    "SingleBudgetSelector",
    "PathBOnlySelector",
]
