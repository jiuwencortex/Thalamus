# oracle_builder/evolutionary/evolution/evolutionary_search_runner.py
# Evolutionary search over ContextGenome space — no LLM calls.
from __future__ import annotations

import numpy as np

from .context_genome import ContextGenome
from .fitness_computer import compute_fitness
from ..component_info import ComponentInfo


class EvolutionarySearchRunner:
    """Find Pareto-optimal context configurations using a genetic algorithm.

    No LLM calls during search — uses pre-computed matrix scores as proxy fitness.

    Key operators:
      Mutation   — flip one component bit (include ↔ exclude)
      Crossover  — mix bits from two parent genomes
      Selection  — tournament selection from the top-fitness pool
      Pareto     — return configurations where no other config is both
                   higher quality AND smaller in size
    """

    def __init__(self, components: list[ComponentInfo], population_size: int = 100, n_generations: int = 200,
                 mutation_rate: float = 0.05, lambda_: float = 0.1, max_tokens: int = 8000, seed: int = 42,
                 fitness_fn=None, cluster_id: int = 0):
        self._components = components
        self._n = len(components)
        self._pop_size = population_size
        self._n_gen = n_generations
        self._mut_rate = mutation_rate
        self._lambda = lambda_
        self._max_tokens = max_tokens
        self._rng = np.random.default_rng(seed)
        self._fitness_fn = fitness_fn       # callable(component_names, cluster_id) -> float | None
        self._cluster_id = cluster_id

    def run(self, query_embedding: np.ndarray) -> list[ContextGenome]:
        """Run the evolutionary loop; return the final Pareto-optimal configurations.

        query_embedding : TF-IDF centroid of the cluster (shape: n_features,)
        """
        if self._n == 0:
            return []

        population = self._init_population()
        self._evaluate(population, query_embedding)

        for _ in range(self._n_gen):
            offspring: list[ContextGenome] = []
            while len(offspring) < self._pop_size // 2:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                offspring.append(child)

            self._evaluate(offspring, query_embedding)
            # Combine and keep the best
            population = sorted(population + offspring, key=lambda g: g.fitness, reverse=True,)[: self._pop_size]

        return self._pareto_front(population)

    # ── private helpers ───────────────────────────────────────────────────────

    def _init_population(self) -> list[ContextGenome]:
        return [ContextGenome(bits=(self._rng.random(self._n) > 0.5))
                for _ in range(self._pop_size)]

    def _evaluate(self, population: list[ContextGenome], query_embedding: np.ndarray) -> None:
        for genome in population:
            if self._fitness_fn is not None:
                # R4: model-based set-level quality replaces marginal sum-of-scores
                total_tokens = sum(
                    c.body_tokens for c, b in zip(self._components, genome.bits) if b
                )
                names = genome.included_names(self._components)
                quality = self._fitness_fn(names, self._cluster_id)
                size_penalty = self._lambda * (total_tokens / max(self._max_tokens, 1))
                genome.fitness = quality - size_penalty
                genome.context_tokens = total_tokens
            else:
                genome.fitness, genome.context_tokens = compute_fitness(
                    genome, self._components, query_embedding, self._lambda, self._max_tokens
                )

    def _mutate(self, genome: ContextGenome) -> ContextGenome:
        bits = genome.bits.copy()
        flip_mask = self._rng.random(self._n) < self._mut_rate
        bits[flip_mask] = ~bits[flip_mask]
        return ContextGenome(bits=bits)

    def _crossover(self, p1: ContextGenome, p2: ContextGenome) -> ContextGenome:
        mask = self._rng.random(self._n) > 0.5
        bits = np.where(mask, p1.bits, p2.bits)
        return ContextGenome(bits=bits)

    def _tournament_select(self, population: list[ContextGenome], k: int = 3) -> ContextGenome:
        idx = self._rng.integers(0, len(population), size=min(k, len(population)))
        return max((population[i] for i in idx), key=lambda g: g.fitness)

    def _pareto_front(self, population: list[ContextGenome]) -> list[ContextGenome]:
        """Return the non-dominated set on the (fitness, context_tokens) objectives.

        A genome A dominates B if A has fitness ≥ B AND tokens ≤ B (at least one strict).
        Non-dominated = no other genome dominates it.
        Duplicates (identical bit patterns) are deduplicated — highest fitness is kept.
        """
        # Deduplicate by bit pattern first; keep highest fitness per unique config
        seen: dict[bytes, ContextGenome] = {}
        for genome in population:
            key = genome.bits.tobytes()
            if key not in seen or genome.fitness > seen[key].fitness:
                seen[key] = genome
        unique = list(seen.values())

        front: list[ContextGenome] = []
        for genome in unique:
            dominated = any(
                other is not genome
                and other.fitness >= genome.fitness
                and other.context_tokens <= genome.context_tokens
                and (other.fitness > genome.fitness or other.context_tokens < genome.context_tokens)
                for other in unique)
            if not dominated:
                front.append(genome)

        # Sort by token count ascending (smallest context first)
        return sorted(front, key=lambda g: g.context_tokens)
