# Phase 3 — Evolutionary Context Search

> **Goal:** Find the best combination of skills, memory sections, and tools for each
> query type — not just the best individual items, but the best set together.

---

## The limitation of per-component scoring

After Phases 1 and 2, we have relevance scores for every individual component:

```
For query: "Set up CI pipeline for the new microservice"

Skills:
  devops-toolkit        0.87
  code-reviewer         0.61

Memory sections:
  ## Deployment setup   0.91
  ## CI/CD pipeline     0.88

Tools:
  bash_exec             0.82
  web_search            0.55
```

A simple threshold (include everything above 0.5) gives a reasonable context. But
this misses something: **components interact with each other**.

Some combinations are better together than their individual scores suggest:
- `devops-toolkit` skill + `## Deployment setup` memory section together give the LLM
  complete context for infrastructure tasks. Either alone is less effective.

Some combinations are redundant:
- `web_search` and `web_fetch` score highly for the same queries. Including both is
  wasteful; one is usually enough.

Some combinations actively compete:
- Two skills that handle similar tasks may confuse the LLM about which one to use.
  One high-scoring skill is better than two overlapping ones.

Per-component scoring cannot capture these interaction effects.
We need to search over **combinations**, not individual items.

---

## The search problem

We want to find the **context configuration** — a specific set of components —
that produces the best outcome for a given query type.

A configuration is a list of which components to include:

```
Config example:
  Skills:  [devops-toolkit, code-reviewer]
  Memory:  [## Deployment setup, ## CI/CD pipeline]
  Tools:   [bash_exec, web_search]
```

With 50 skills, 10 memory sections, and 15 tools, there are 2^75 possible
configurations. We cannot try them all.

This is exactly the same problem as **Neural Architecture Search (NAS)** in AutoML,
where you want to find the best neural network architecture from an enormous space of
possibilities. NAS solved this with evolutionary algorithms and gradient-based search.
We apply the same ideas here.

---

## The key advantage: no LLM calls needed during search

In NAS, evaluating each architecture requires training a neural network — which is
slow and expensive. NAS uses **proxy metrics** (train for only a few steps, measure
a cheap estimate of quality) to make the search feasible.

We already have our proxy metrics: **the pre-computed matrices from Phases 1 and 2.**

For any configuration and any query, we can estimate quality using only matrix lookups
— no LLM calls during the search. This makes the search very fast.

The fitness of a configuration = weighted combination of the component scores for that
query type, minus a penalty for total context size.

---

## Evolutionary search algorithm

### Representation

A **context genome** is a bitmask over all available components:

```
[skill_devops_toolkit=1, skill_code_reviewer=1, skill_email_sender=0,
 mem_deployment_setup=1, mem_known_bugs=0, mem_team_contacts=0,
 tool_bash_exec=1, tool_web_search=1, tool_generate_image=0, ...]
```

Each bit is 1 (include) or 0 (exclude).

### Fitness function

For a given query type (represented by its embedding), the fitness of a genome is:

```
fitness = Σ (score_i × included_i × similarity_i) − λ × context_size

where:
  score_i       = pre-computed quality score for component i from its matrix
  included_i    = 1 if this component is in the genome, 0 otherwise
  similarity_i  = cosine similarity of the query to this component's example queries
  λ             = penalty weight for context size (controls quality vs cost tradeoff)
  context_size  = sum of token counts for all included components
```

This is computed entirely from pre-computed values. Fast, no LLM calls.

### Evolutionary operators

**Mutation:** Flip one random bit in the genome (include ↔ exclude one component).
Small, local change — explores nearby configurations.

**Crossover:** Take two parent genomes. For each component, randomly pick the value
from one parent or the other. Creates new combinations not seen in either parent.

**Selection:** Keep the genomes with the highest fitness scores. Remove the rest.
Run selection on a Pareto front (see below) rather than a single ranking.

### Pareto front: quality vs. cost

Instead of a single winner, we look for the **Pareto-optimal** configurations: the
ones where no other configuration is both higher quality AND smaller in size.

```
Quality
  ^
  |    *         ← max quality, large context
  |      *
  |         *
  |           *  ← min context, acceptable quality
  +------------> Context size
```

Each point on the curve is Pareto-optimal. The system exposes a `context_budget`
parameter: the operator sets how many tokens they want to spend on context, and the
system picks the Pareto-optimal configuration for that budget.

This gives a natural dial:
- `context_budget=small` → lean context, fast and cheap, may miss edge cases
- `context_budget=large` → comprehensive context, thorough, more expensive

### Algorithm outline

```
1. Start with population of N random genomes

2. For each generation:
   a. Compute fitness for each genome (matrix lookup, no LLM calls)
   b. Build Pareto front
   c. Select top genomes (tournament selection)
   d. Apply crossover to selected pairs
   e. Apply mutation to offspring
   f. Replace bottom of population with offspring

3. Run for K generations (typically 100–500)

4. Output: Pareto-optimal configurations for this query type
```

Typical parameters: N=100 genomes, K=200 generations.
No LLM calls. Runs in seconds on a laptop.

---

## Query clustering

We don't search for one config per individual user query — that would be a new search
for every message, which is unnecessary. Instead:

**Step 1:** Collect all `example_input` texts from across all matrices (skill matrix,
memory matrix, tool matrix). Embed them.

**Step 2:** Cluster the embeddings into K groups (e.g. K=20). Each cluster represents
a "query type":
- Cluster 1: code-editing tasks
- Cluster 2: information retrieval tasks
- Cluster 3: file management tasks
- Cluster 4: email and communication tasks
- etc.

**Step 3:** Run the evolutionary search once per cluster → optimal config per cluster.

**Step 4:** Store the results as `context_configs.json`:

```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "label": "code-editing",
      "centroid_embedding": [...],
      "optimal_configs": {
        "budget_small":  { "skills": [...], "memory": [...], "tools": [...] },
        "budget_medium": { "skills": [...], "memory": [...], "tools": [...] },
        "budget_large":  { "skills": [...], "memory": [...], "tools": [...] }
      }
    },
    ...
  ]
}
```

**At query time:** embed the user query → find nearest cluster centroid (one dot
product) → look up pre-computed config for the desired budget → use that config.

This makes query-time selection essentially instant.

---

## Integration with Phase 1 and 2

The evolutionary search reads from all three matrix directories:

```
~/.jiuwenswarm/agent/workspace/
  skill_matrix/        ← Phase 0 (existing)
  memory_matrix/       ← Phase 1
  tool_matrix/         ← Phase 2
  context_configs.json ← Phase 3 output
```

The search rebuilds `context_configs.json` whenever any matrix changes.

---

## What runs when

| Step | Trigger | Cost |
|---|---|---|
| Build skill matrix | Skills added/changed | ~800 LLM calls (40 skills × 20) |
| Build memory matrix | project.md or user.md changed | ~128 LLM calls |
| Build tool matrix | Tools added/changed | ~240 LLM calls |
| Run evolutionary search | Any matrix changed | 0 LLM calls, seconds of CPU |
| Query-time selection | Every user message | 1 embedding call + table lookup |

---

## DARTS-style gradient search (optional upgrade)

The evolutionary algorithm above is a good first implementation. An alternative is
to use **gradient-based search**, inspired by DARTS (one of the most widely used
NAS methods).

Instead of a bitmask (0/1), each component gets a **continuous weight** α between 0
and 1. The weights are passed through a softmax so they sum to 1.

The quality score becomes differentiable with respect to α:

```
quality(α) = Σ αᵢ × score_i × similarity_i − λ × Σ αᵢ × token_count_i
```

We can compute the gradient of quality with respect to each α and do gradient ascent.
After convergence, threshold the α values (e.g. include components where α > 0.3) to
get the final bitmask.

This finds the same kind of solution as the evolutionary algorithm but is faster to
converge for large component sets. It is more mathematically principled but also more
complex to implement.

**Recommendation:** Start with the evolutionary algorithm. It is easier to debug and
gives interpretable results. If the component count grows above ~200, switch to
DARTS-style gradient search.

---

## New code needed

### Files

```
jiuwenswarm/tools/context_optimizer/
  __init__.py
  __main__.py           ← CLI: python -m jiuwenswarm.tools.context_optimizer
  genome.py             ← ContextGenome dataclass + fitness function
  evolutionary.py       ← EvolutionarySearch class
  query_clusterer.py    ← Cluster all example queries into K groups
  config_builder.py     ← Run search per cluster, write context_configs.json
  cli.py                ← argparse entry point
```

No new LLM-calling code. All inputs are pre-computed matrices. The evolutionary
algorithm is pure Python (no ML framework needed for the basic version).

### Dependencies

- `numpy` — for embedding arithmetic (dot products, cosine similarity)
- `scikit-learn` — for K-means clustering (already likely available)
- No new LLM calls, no GPU required

---

## Implementation order

1. `genome.py` — represent a configuration as a bitmask, compute fitness from matrices
2. `evolutionary.py` — mutation, crossover, selection, Pareto front
3. Unit test: run search on synthetic scores, verify Pareto front makes sense
4. `query_clusterer.py` — embed example inputs, run K-means
5. `config_builder.py` — run search per cluster, serialize results
6. `cli.py` + `__main__.py`
7. Integration: extend context assembly to load `context_configs.json` at query time

**Total new code: ~400 lines.** No LLM calls, no GPU, no new external services.

---

## What Phase 3 unlocks

- Context configuration goes from "threshold on individual scores" to "searched optimal set"
- Pareto front gives operator control over the quality/cost tradeoff
- Pre-computed cluster configs make query-time selection instant
- The search naturally finds redundant components and removes them
- Foundation for Phase 4: real interaction data replaces synthetic matrix scores
