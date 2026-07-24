"""Phase R3b — Minimum exploration rate derivation.

Formalizes context component selection as a multi-label contextual bandit
problem and derives the minimum off-policy exploration rate ε* required for
the Path B classifier to converge to a policy *not dominated by Path A*.

**Problem formalization**

- **State** s: query embedding vector
- **Action** a: binary bitmask over the component library (2^n actions)
- **Reward** r: outcome quality scalar ∈ [0, 1]
- **Path A policy** π_A(s): deterministic — lookup(cluster(s), budget)
- **Path B policy** π_B(s): logistic regression on logged (s, a, r) tuples

**Why ε=0 fails**

Under pure exploitation (ε=0), the turn log only contains the actions π_A
chose.  For a component c_i that Path A includes in *p_A(c_i)* fraction of
clusters, the log contains:

    proportion of c_i=True turns = p_A(c_i)

Path B learns the conditional distribution P(c_i=1 | s) from these logs.
When p_A(c_i) is the same across all clusters (or Path A's action is near-
deterministic), Path B cannot learn anything about c_i's *marginal* quality
— it only learns that "Path A includes c_i in p_A(c_i) of cases", which
means it converges to Path A's policy rather than learning from outcomes.

**ε* derivation**

For each component c_i:

    p_A(c_i) = (number of clusters where c_i appears in the oracle config)
               / total_clusters

Under ε-greedy exploration, a fraction ε of turns use a random action
instead of Path A's choice.  Of these, each component is independently
included with probability 0.5 (uniform random over bitmask).

The effective proportion of c_i=True turns in the turn log is:

    p_total(c_i) = (1 - ε) × p_A(c_i) + ε × 0.5

For Path B to estimate P(c_i=1 | s) with sufficient evidence, we need at
least *n_min* turns in each class (c_i=True and c_i=False) within the
first *T_target* turns:

    min(p_total(c_i) × T_target, (1 - p_total(c_i)) × T_target) ≥ n_min

The binding constraint is on the *minority class*.  For components with
p_A(c_i) ≈ 0 (rarely chosen by Path A), the minority class is c_i=True:

    p_total(c_i) × T_target ≥ n_min
    ((1 - ε) × p_A(c_i) + ε × 0.5) × T_target ≥ n_min

Solving for ε:

    ε ≥ (n_min / T_target - p_A(c_i)) / (0.5 - p_A(c_i))

For p_A(c_i) < 0.5 (component under-represented in Path A), this gives the
minimum exploration rate needed for that component.

**ε*** is the maximum over all components:

    ε* = max_i { max(0, (n_min/T_target - p_A(c_i)) / (0.5 - p_A(c_i))) }

This is a *component-wise sufficient condition*, not a tight bound.
Phase R3b will validate it empirically on the jiuwenswarm task suite.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_N_MIN = 10          # minimum samples per class
_DEFAULT_T_TARGET = 500      # turns at "mature" maturity
_BUDGET_KEYS = ("budget_small", "budget_medium", "budget_large")


@dataclass
class ComponentCoverage:
    """Path A coverage analysis for a single component."""

    name: str
    p_path_a: float           # fraction of clusters where Path A includes this component
    p_total_at_epsilon: float  # effective inclusion rate under ε* exploration
    n_true_at_T: float         # expected c_i=True turns at T_target under ε*
    requires_exploration: bool  # True if ε=0 gives insufficient coverage

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExplorationRateResult:
    """Output of the ε* derivation."""

    epsilon_star: float               # derived minimum exploration rate
    n_min: int                        # minimum samples per class (input)
    T_target: int                     # target total turns (input)
    n_components: int                 # total components in oracle
    n_clusters: int                   # total clusters in oracle
    critical_component: str           # component that sets ε*
    critical_p_path_a: float          # Path A coverage of the critical component
    component_coverage: list[ComponentCoverage]  # per-component breakdown
    interpretation: str               # human-readable summary

    def to_dict(self) -> dict:
        d = asdict(self)
        d["component_coverage"] = [c.to_dict() for c in self.component_coverage]
        return d


class ExplorationRateEstimator:
    """Derive the minimum off-policy exploration rate from oracle structure.

    Parameters
    ----------
    oracle_dir:
        Directory containing ``context_configs.json``.  The Path A action
        distribution is derived from the per-cluster optimal configs.
    """

    def __init__(self, cluster_data: list[dict]) -> None:
        self._clusters = cluster_data

    @classmethod
    def load(cls, oracle_dir: str | Path) -> "ExplorationRateEstimator":
        """Load from *oracle_dir*/context_configs.json."""
        oracle_dir = Path(oracle_dir)
        config_path = oracle_dir / "context_configs.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"context_configs.json not found in {oracle_dir}. "
                "Run: thalamus-oracle evolve"
            )
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(data.get("clusters", []))

    # ── public API ────────────────────────────────────────────────────────────

    def estimate(
        self,
        n_min: int = _DEFAULT_N_MIN,
        T_target: int = _DEFAULT_T_TARGET,
    ) -> ExplorationRateResult:
        """Derive ε* for the current oracle configuration.

        Parameters
        ----------
        n_min:
            Minimum number of turns required in each class (c_i=True,
            c_i=False) for reliable classifier training.  Default: 10.
        T_target:
            Target total turns at which ε* should provide sufficient coverage.
            Corresponds to the "mature" maturity checkpoint.  Default: 500.

        Returns
        -------
        ExplorationRateResult
            Includes ε*, the critical component, and per-component breakdown.
        """
        # Build per-component Path A inclusion frequency
        p_path_a = self._compute_path_a_coverage()

        if not p_path_a:
            raise ValueError("No components found in oracle clusters.")

        n_clusters = len(self._clusters)
        coverage_list: list[ComponentCoverage] = []
        epsilon_star = 0.0
        critical_component = ""
        critical_p = 0.0

        for name, p_a in sorted(p_path_a.items()):
            # Required ε for this component (under-represented → p_a < 0.5)
            eps_i = _min_epsilon_for_component(p_a, n_min, T_target)

            p_total = (1 - eps_i) * p_a + eps_i * 0.5
            n_true_at_T = p_total * T_target

            coverage_list.append(ComponentCoverage(
                name=name,
                p_path_a=round(p_a, 4),
                p_total_at_epsilon=round(p_total, 4),
                n_true_at_T=round(n_true_at_T, 1),
                requires_exploration=(eps_i > 0.0),
            ))

            if eps_i > epsilon_star:
                epsilon_star = eps_i
                critical_component = name
                critical_p = p_a

        epsilon_star = round(epsilon_star, 4)
        coverage_list.sort(key=lambda c: c.p_path_a)

        interp = (
            f"ε* = {epsilon_star:.4f}: with this exploration rate, every component "
            f"will have at least {n_min} observed turns (in each class) within "
            f"{T_target} total turns.  "
            f"Critical component: '{critical_component}' appears in only "
            f"{critical_p * 100:.1f}% of clusters under Path A."
        )
        if epsilon_star == 0.0:
            interp = (
                f"ε* = 0: Path A provides sufficient coverage for all components "
                f"(min p_A = {min(p_path_a.values()):.2f}) at T={T_target} turns."
            )

        return ExplorationRateResult(
            epsilon_star=epsilon_star,
            n_min=n_min,
            T_target=T_target,
            n_components=len(p_path_a),
            n_clusters=n_clusters,
            critical_component=critical_component,
            critical_p_path_a=round(critical_p, 4),
            component_coverage=coverage_list,
            interpretation=interp,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _compute_path_a_coverage(self) -> dict[str, float]:
        """For each component, compute the fraction of clusters where Path A includes it."""
        n_clusters = len(self._clusters)
        if n_clusters == 0:
            return {}

        inclusion_counts: dict[str, int] = {}
        for cluster in self._clusters:
            seen_in_cluster: set[str] = set()
            for bkey in _BUDGET_KEYS:
                cfg = cluster.get("optimal_configs", {}).get(bkey, {})
                components = (
                    cfg.get("skills", [])
                    + cfg.get("memory", [])
                    + cfg.get("tools", [])
                )
                seen_in_cluster.update(components)
            for name in seen_in_cluster:
                inclusion_counts[name] = inclusion_counts.get(name, 0) + 1

        return {name: count / n_clusters for name, count in inclusion_counts.items()}


def _min_epsilon_for_component(
    p_a: float,
    n_min: int,
    T_target: int,
) -> float:
    """Minimum ε such that component with Path A rate p_a has ≥ n_min c=True turns.

    Derivation:
        p_total = (1 - ε) × p_a + ε × 0.5
        p_total × T_target ≥ n_min
        ε ≥ (n_min/T_target - p_a) / (0.5 - p_a)

    Returns 0.0 if p_a already gives sufficient coverage without exploration.
    """
    min_p_needed = n_min / T_target
    if p_a >= min_p_needed:
        return 0.0
    if math.isclose(p_a, 0.5, abs_tol=1e-9):
        # Degenerate: p_a = 0.5 but still insufficient (T_target too small)
        return 0.0
    eps = (min_p_needed - p_a) / (0.5 - p_a)
    return min(max(eps, 0.0), 1.0)
