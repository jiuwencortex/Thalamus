"""THALAMUS research package.

Structure
---------
::

    research/
    ├── baselines/       R1 ✓ — AllSelector, RandomSelector, TFIDFSelector, BM25Selector, DenseSelector
    ├── evaluation/      R1 ✓ — BenchmarkRunner, EvalRun, OverlapStats, report
    ├── ablations/       R2 ✓ — TopKSelector, NoBookendSelector, SingleBudgetSelector, PathBOnlySelector
    ├── cross_path/      R3a ✓ — CoInclusionExtractor, FitnessAugmentor
    ├── bandit/          R3b ✓ — ExplorationRateEstimator, ConvergenceAnalyzer
    ├── set_quality/     R4 — set-level quality model (XGB / joint classifier as GA fitness)
    └── meta_learning/   R5 — cross-deployment warm-start from shared knowledge base

CLI
---
``thalamus-research`` — research commands only; ``thalamus-select`` — production only.

Subcommands
~~~~~~~~~~~
- ``baseline-lookup`` — run a query through retrieval baselines (R1)
- ``eval``            — benchmark all selectors on a query set (R1)
- ``ablation``        — run ablation study: TopK / NoBookend / SingleBudget / PathBOnly (R2)
- ``cross-path``      — co-inclusion analysis + augmented fitness (R3a)
- ``bandit``          — estimate ε* / measure Path B convergence (R3b)
"""
