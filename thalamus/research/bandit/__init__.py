"""Phase R3b — Contextual bandit formalization and exploration analysis.

Research goal: formalize context component selection as a multi-label
contextual bandit problem and derive the minimum off-policy exploration rate
required for Path B to converge to a policy *not dominated by Path A*.

**Formal model**

- State s: query embedding vector
- Action a: binary bitmask over n components (2^n actions)
- Reward r: outcome quality scalar ∈ [0, 1]
- Path A policy π_A(s): deterministic cluster lookup (pure exploitation)
- Path B policy π_B(s): logistic regression trained on logged (s, a, r)

**Why exploration is necessary (C4)**

Under ε=0 (pure exploitation), the turn log contains only the actions Path A
chose.  Components excluded by Path A receive no training signal.  Path B
converges to a policy that mirrors Path A rather than learning from outcomes.

**Minimum exploration rate (ε*)**

Derived from the requirement that every component receives at least n_min
training examples in each class (included / excluded) within T_target turns:

    ε* = max_i { max(0, (n_min/T_target - p_A(c_i)) / (0.5 - p_A(c_i))) }

where p_A(c_i) is the fraction of clusters in which Path A includes c_i.

**Modules**

- :class:`ExplorationRateEstimator` — derive ε* from oracle structure alone
  (no turn logs required).
- :class:`ConvergenceAnalyzer` — measure Path B → Path A selection agreement
  over logged turns (requires ``turns_*.jsonl`` files).

CLI::

    # Derive ε* from oracle structure
    thalamus-research bandit --oracle-dir /oracle --subcommand estimate-rate

    # Measure Path B convergence to Path A
    thalamus-research bandit --oracle-dir /oracle --subcommand convergence \\
        --turn-log-dir /logs --window-size 50

Prerequisite: Phase R1 complete (baselines + evaluation harness).
"""
from .exploration_rate import ExplorationRateEstimator, ExplorationRateResult, ComponentCoverage
from .convergence import ConvergenceAnalyzer, ConvergenceResult, ConvergencePoint

__all__ = [
    "ExplorationRateEstimator",
    "ExplorationRateResult",
    "ComponentCoverage",
    "ConvergenceAnalyzer",
    "ConvergenceResult",
    "ConvergencePoint",
]
