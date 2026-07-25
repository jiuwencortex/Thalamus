# THALAMUS: Combinatorial Context Selection for Production AI Agents

**[Author names redacted for review]**

*Preprint. Under review.*

---

## Abstract

Production AI agents suffer from a structural failure we term **Context Saturation**: as an agent's library of skills, memory sections, and tool definitions grows, loading all components unconditionally into every context window degrades answer quality, scales token cost with library size rather than query complexity, and distributes finite transformer attention across irrelevant content. The combinatorially correct solution — selecting, per query and per token budget, only the component subset that jointly maximizes outcome quality — requires knowing in advance which combinations of components produce the best results for each query type, without access to LLM calls during inference.

We present **THALAMUS**, a system that solves this problem by moving all expensive reasoning offline. A preparation pipeline uses LLM-driven evaluation to produce per-component relevance scores, blends those scores with real interaction evidence via Bayesian weighting, and then runs a genetic algorithm — entirely without LLM calls — over the exponential space of component bitmasks, finding Pareto-optimal configurations for each query cluster and token budget. At query time, context selection reduces to a nearest-cluster lookup completing in under ten milliseconds. A supervised logistic regression classifier trained on logged agent turns refines this assignment as operational data accumulates; an off-policy exploration mechanism grounded in contextual bandit theory prevents the classifier from collapsing to a biased policy.

Beyond this core design, THALAMUS incorporates four extensions that address specific structural limitations: (i) cross-path co-inclusion transfer, which extracts pairwise component interaction signal from the classifier's weight matrix and feeds it back into the genetic algorithm's fitness function; (ii) a minimum exploration rate derivation that analytically determines the off-policy fraction required to guarantee sufficient training coverage; (iii) a gradient-boosted regression model that replaces the hand-crafted linear fitness formula with a learned set-level quality predictor capturing non-linear interaction effects; and (iv) a cross-deployment meta-learning protocol that transfers component quality priors between deployment instances via SHA-256 content fingerprinting.

We define nine empirically testable research claims, a five-baseline comparison suite implementing a common protocol interface, a structured ablation study, and a 120-task evaluation harness. The full research package implementing these claims is described herein. Experimental results on the jiuwenswarm evaluation suite are in preparation.

---

## 1. Introduction

A production AI agent processes incoming user tasks through a context window assembled from components: skill instruction documents encoding domain knowledge, memory sections encoding project state and conventions, and tool definitions enabling external actions. The central design choice at the heart of agentic system deployment is the **context assembly policy**: given an incoming query, which components should be placed in the context window?

The answer uniformly adopted in practice is all of them. In small deployments this is harmless. A six-skill agent with two memory sections and four tools loads a modest context and operates adequately regardless of query type. The uniformity of the policy incurs no visible cost.

The cost appears at scale. When a production agent accumulates forty or more skill documents, multiple memory files encoding months of project history, and a rich tool library, the unconditional-inclusion strategy creates three compounding problems.

**Quality degradation.** Transformer attention is finite and distributed. A context window containing forty components, of which perhaps four are relevant to the incoming query, distributes the model's effective attention across thirty-six irrelevant documents. The model must identify which four components matter from a noisy context, a task it is not optimized for and that becomes progressively harder as library size grows.

**Cost scaling.** Token cost at the language model API level scales linearly with context length. Under the unconditional-inclusion policy, a simple query requiring two skill documents pays the same token cost as an architectural query requiring twenty. The policy structurally wastes tokens on every request, with waste growing proportionally to library size.

**Lost-in-the-middle attention decay.** Empirical work on long-context transformer models [Liu et al., 2024] establishes that models attend preferentially to material placed at the beginning and end of their context, with content placed in the interior receiving substantially weaker effective attention weight. A context window containing forty components places most relevant material in the middle, below the attention threshold at which it would meaningfully influence the model's generation.

These three failure modes compound: a larger library degrades quality, increases cost, and exacerbates positional attention decay simultaneously. We name this compound failure **Context Saturation** and present THALAMUS as a system engineered to close it.

The core insight is that the expensive question — "which components does this query type require?" — does not need to be answered at query time. If the optimal component configuration can be precomputed for each identifiable query type and each token budget, query-time selection reduces to a lookup. THALAMUS makes this precomputation tractable through three mechanisms: LLM-driven offline scoring of individual components; query space partitioning into manageable clusters; and a genetic algorithm that searches over the exponential space of component combinations without any LLM calls in its inner loop.

This architecture produces a system deployable from the first query, with no operational data requirement at cold start, that improves continuously as agent turn logs accumulate. It supports three component types (skills, memory sections, tools), three token budget tiers, two embedding backends (TF-IDF and sentence transformers), and a dual-path inference architecture with automatic maturity-based path selection.

**Contributions.** Beyond the core system, this paper presents:

- **C1 — Combinatorial selection outperforms independent retrieval:** The genetic algorithm finds component sets that outperform top-k retrieval by individual relevance score, particularly on tasks requiring multiple interacting components (§4.3, §7.1).
- **C2 — LLM-generated synthetic priors reduce cold-start sample complexity:** LLM-generated component scores provide a warm start that reduces the number of logged turns required for the classifier to exceed the genetic algorithm baseline (§4.2, §7.2).
- **C3 — Dual-path architecture dominates either path alone:** The unified selection system achieves higher quality than either the cluster-based or classifier-based path across the full maturity curve (§5, §7.3).
- **C4 — Off-policy exploration is necessary and sufficient for Path B convergence:** Without off-policy counterfactual turns, the classifier converges to an imitation of the cluster-based policy; with analytically derived minimum exploration rate, it learns a strictly superior policy (§4.4, §7.4).
- **C5 — Bookend context ordering improves quality on long-context tasks:** Placing the most-relevant components at the edges of the assembled context yields measurable quality improvement on tasks with assembled context exceeding three thousand tokens (§4.5, §7.5).
- **C6 — Budget-adaptive selection outperforms any fixed-budget policy:** Heuristic query-complexity estimation that selects among three budget tiers outperforms any fixed tier on a mixed-complexity evaluation suite (§4.6, §7.6).
- **C7 — Classifier co-inclusion signal improves genetic algorithm fitness:** Component interaction patterns extracted from the classifier's weight matrix and injected into the genetic algorithm's fitness function surface jointly-useful component pairs that the independent linear fitness formula misses (§6.2, §7.7).
- **C8 — Learned set-level fitness outperforms hand-crafted formula:** A gradient-boosted regressor trained on logged (component set, outcome quality) pairs captures non-linear pairwise interaction effects, yielding superior configurations compared to the independent sum-of-scores baseline fitness (§6.3, §7.8).
- **C9 — Cross-deployment fingerprint transfer reduces cold-start time:** Content-addressed component quality priors transferred from prior deployments reduce the number of turns required for a new deployment to reach steady-state configuration quality (§6.4, §7.9).

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

Retrieval-Augmented Generation [Lewis et al., 2020] retrieves documents from a corpus at inference time using dense or sparse similarity and prepends them to the model input. RAG has become the dominant approach for knowledge-grounded generation and provides the strongest baseline against which THALAMUS is measured. Its structural limitations in the agentic context are threefold: it treats every component independently, selecting by individual relevance rather than joint utility; it does not account for token budget constraints or inter-component combination effects; and it requires retrieval computation at query time. THALAMUS addresses all three by precomputing optimal combinations offline and performing selection via a sub-millisecond lookup.

### 2.2 LLM-Based Agent Frameworks

The agent framework literature [Park et al., 2023; Yao et al., 2023; Wang et al., 2024] has focused principally on multi-step reasoning, tool use protocols, and memory management. With few exceptions, these frameworks adopt fixed context assembly: all available tools, all memory sections, and all skill prompts are included in every context call. THALAMUS is orthogonal to the reasoning architecture and operates at the context assembly layer upstream of the agent's reasoning process. It is compatible with any downstream agent framework that accepts an externally assembled context.

### 2.3 Prompt Compression and Selection

Methods for prompt compression and selection [Li et al., 2023; Jiang et al., 2023] aim to reduce token costs by pruning or distilling individual documents. These methods treat each document independently — pruning its content or selecting among alternative phrasings — and do not address the combinatorial question of which documents to include. Optimization-based prompt selection [Khattab et al., 2023] finds effective prompting strategies for fixed task descriptions but optimizes over phrasings, not over inclusion sets from a dynamic component library.

### 2.4 Long-Context Attention and Positional Bias

Liu et al. [2024] provide the empirical foundation for the bookend ordering strategy described in §4.5. Their systematic evaluation of long-context transformers demonstrates that models attend preferentially to content at the beginning and end of the context window, with material in the interior receiving substantially weaker attention weight — a phenomenon they term "lost in the middle." THALAMUS's bookend ordering policy formalizes this finding into a deployable component ordering strategy.

### 2.5 Contextual Bandits and Off-Policy Learning

The off-policy exploration mechanism in the classifier training path (§4.4) is a practical instantiation of contextual bandit theory [Langford & Zhang, 2007; Li et al., 2010]. The component selection problem is naturally framed as a multi-label contextual bandit: state is the query embedding, action is the binary component inclusion bitmask, and reward is the outcome quality scalar. The minimum exploration rate derivation in §6.2 follows directly from the bandit coverage requirement: every (component, inclusion) pair must receive sufficient training examples for reliable logistic regression estimation.

### 2.6 Genetic Algorithms for Combinatorial Optimization

Genetic algorithms have been applied to combinatorial optimization in software product line configuration [Harman et al., 2012] and feature selection [Xue et al., 2016]. The agent context window optimization problem is structurally novel: the fitness function combines component relevance, query cluster coherence, and a hard token budget constraint, and the search space grows exponentially with library size. THALAMUS applies genetic search to this problem without any modification of the standard GA operators, relying on the fitness function to encode the domain-specific quality signal.

### 2.7 Gradient-Boosted Regression for Set Quality

The set-level quality model in §6.3 applies gradient-boosted regression [Friedman, 2001] to predict the outcome quality of a component set from a 14-dimensional feature vector derived from component scores, type composition, and pairwise co-inclusion statistics. This approach is motivated by the known limitations of additive models for combinatorial structures: linear combination of individual scores structurally cannot represent interaction effects, joint necessity, or mutual redundancy between components.

---

## 3. Problem Formulation

Let **C** = {c₁, c₂, …, cₙ} be the agent's **component library**: the union of its skill instruction documents, memory sections, and tool definitions. Each component cᵢ has a token cost τ(cᵢ) ∈ ℤ⁺.

Let **q** ∈ 𝒬 be an incoming user query drawn from the agent's operational query distribution. Let **B** ∈ ℝ⁺ be a token budget for the assembled component context.

**Definition (Context Selection Problem).** Find a subset **S** ⊆ **C** satisfying the budget constraint:

```
∑_{c ∈ S} τ(c) ≤ B
```

such that the expected task outcome quality E[Q(S, q)] is maximized, where Q: 2^C × 𝒬 → [0, 1] is a quality function mapping component sets and queries to scalar outcome quality.

**Why the problem is hard.** The candidate set space has cardinality 2ⁿ. Direct evaluation of any candidate set S requires a live agent execution and a ground-truth quality measurement, making exhaustive search infeasible for n > 20. Individual component quality scores are imperfect proxies for joint set utility: components may be jointly necessary (neither alone is useful, together they are), jointly redundant (either alone suffices, both together add noise), or mutually interfering (each produces good individual answers but generates contradictory instructions when both are present). Real outcome quality measurements arrive asynchronously, with delay, and sparsely in the early deployment period. No closed-form solution exists for arbitrary query distributions and arbitrary component libraries.

**THALAMUS's approximation.** THALAMUS constructs an approximation to the optimal selection function through five sequential steps:

1. **Offline scoring:** Estimate per-component relevance using LLM-driven evaluation (§4.1).
2. **Query clustering:** Partition the query space 𝒬 into K clusters {Ω₁, …, Ω_K} using K-means on component example embeddings. Precompute cluster centroids.
3. **Combinatorial search:** For each cluster Ωₖ and budget tier B, find the approximately optimal set S*(k, B) via genetic algorithm over the component bitmask space (§4.3).
4. **Configuration store:** Persist the resulting (cluster, budget) → S* mapping to a lookup table `context_configs.json`.
5. **Online refinement:** Train a per-component logistic regression classifier on logged (query, component set, quality) triples to refine the lookup as operational data accumulates (§4.4).

---

## 4. The THALAMUS Architecture

THALAMUS comprises four offline preparation phases and one online runtime phase. Phases 1–2 produce per-component quality scores. Phase 3 uses those scores to precompute optimal configurations. Phase 4 trains a classifier from operational data. The runtime (§5) performs inference.

### 4.1 Phase 1 — Component Scoring

**Goal.** Produce a scored evaluation matrix for every component in **C**, capturing its expected quality across a representative sample of query types.

For each component cᵢ, the scoring pipeline:

1. Generates M synthetic (query, expected\_answer) pairs covering the component's domain, using a language model prompted with the component's full text. Default M = 20.
2. Executes the agent with only cᵢ in context on each of the M synthetic queries and collects the agent's output.
3. Evaluates each output against the expected answer with four lexical metrics:
   - **F1 token overlap:** Unigram token-level F1 between output and reference.
   - **Bigram F1:** Phrase-level F1, capturing fluency in addition to coverage.
   - **Bag-of-words recall:** Recall of reference tokens in the output, without penalizing additional content.
   - **Length ratio:** Output-to-reference length ratio, penalizing both truncation and verbosity.
4. Writes the evaluation results to a structured JSON scoring matrix, one file per component.

The primary fitness signal used downstream is the mean F1 score across all M evaluation rows:

```
mean_score(cᵢ) = (1/M) × ∑_{j=1}^{M} F1(output_j, reference_j)
```

**Schema.** Each scoring matrix file follows the structure:
```json
{
  "component_name": "deploy_ci",
  "baseline_cross_eval": [
    {
      "example_input": "How do I set up CI for a Python project?",
      "candidate_output": "...",
      "scores": {
        "f1": 0.71, "bigram_f1": 0.64,
        "bow_recall": 0.78, "length_ratio": 0.92
      }
    }
  ]
}
```

**Component types.** Three component types are scored independently: skill instruction documents (discovered from SKILL.md files and Markdown sources), memory sections (structured documentation describing project state, conventions, and domain context), and tools (Python source files, discovered via AST parsing for classes whose name contains "Tool" and carries a docstring).

**Enrichment hook.** The schema includes a `real_data` field populated by Phase 2:
```json
"real_data": { "n_turns": 42, "updated_mean_score": 0.74 }
```
When this field is present, `updated_mean_score` supersedes the synthetic mean in all downstream computations.

### 4.2 Phase 2 — Score Enrichment

**Goal.** Blend LLM-generated synthetic scores with real interaction evidence to reduce dependency on the lexical proxy metrics as operational data accumulates.

The enrichment step processes weekly turn log files (`turns_YYYY-WNN.jsonl`), each containing one JSON line per agent interaction with fields `{query_embedding, component_set, outcome_quality, exploration}`. For each component cᵢ, it extracts all turns in which cᵢ was included and computes the empirical mean outcome quality. The enriched score blends this empirical mean with the synthetic prior via Bayesian weighting:

```
enriched_score(cᵢ) = (n × empirical_mean + α × synthetic_mean) / (n + α)
```

where n is the number of logged turns including cᵢ and α is a pseudo-count prior weight (default α = 5). At n = 0, this reduces to the synthetic score. As n grows, it converges to the empirical mean.

The enriched `updated_mean_score` is written to the `real_data` field of the scoring matrix and thereafter used in place of the synthetic score in the genetic algorithm fitness function.

**Research contribution C2.** This enrichment mechanism instantiates a warm-start hypothesis: LLM-generated scores provide a reasonably calibrated prior over component quality, such that the sample complexity required for the empirical mean to overtake the synthetic prior is substantially lower than the sample complexity required for cold-start learning from scratch. Section 7.2 quantifies this hypothesis experimentally.

### 4.3 Phase 3 — Evolutionary Oracle (Path A)

**Goal.** Precompute, for every (query cluster, token budget) pair, the approximately optimal component configuration using a genetic algorithm operating over the space of component bitmasks.

#### 4.3.1 Query Clustering

The query space is partitioned into K clusters (default K = 20, tunable via elbow/silhouette analysis with `--auto-k`) using K-means applied to TF-IDF vectors of component example inputs or to sentence-transformer embeddings of historical query texts. The cluster centroid model is serialized to `context_configs.pkl` for query-time assignment.

Automatic cluster count selection fits K-means for K ∈ {5, 7, 10, 15, 20, 25, 30} and selects the K minimizing a weighted combination of the within-cluster sum-of-squares elbow criterion and the silhouette coefficient.

#### 4.3.2 Marginal Fitness Function

For a candidate component set S, query cluster k with centroid vector **z**_k, and budget tier B with maximum token count B_max:

```
fitness(S, k, B) =
  [ ∑_{c ∈ S} mean_score(c) × max(cosine(q_c, z_k), 0) ]  −  λ × (token_cost(S) / B_max)

  where q_c is the component's example-query centroid vector
        λ is the size penalty weight (default 0.1)
        token_cost(S) = ∑_{c ∈ S} τ(c)
```

The cosine similarity term weights each component's mean score by the geometric alignment between its example queries and the cluster centroid, concentrating fitness credit on components whose domain is genuinely relevant to the query cluster. The penalty term λ × (token_cost / B_max) imposes a soft budget incentive on the GA, with the Pareto front over (fitness, token_cost) maintained to produce configurations spanning the full budget range.

This formulation is deliberately linear in individual component scores: it structurally cannot represent pairwise interaction effects. This limitation is addressed by the learned fitness extension in §6.3.

#### 4.3.3 Genetic Algorithm

The genetic algorithm operates over binary bitmasks of length n (n = number of components in C). Each generation:

- **Initialization:** Population of 100 individuals, each bit set to 1 with probability 0.5.
- **Evaluation:** Fitness computed by the formula above.
- **Selection:** Tournament selection, k = 3, from the top-fitness pool.
- **Crossover:** Uniform crossover — each bit inherited from parent 1 or parent 2 with equal probability.
- **Mutation:** Bit-flip mutation at rate 0.05 per bit per generation.
- **Elitism:** Combined parent and offspring pool sorted by fitness; top 100 individuals carried forward.
- **Termination:** 200 generations.

The algorithm runs once per (cluster, budget) pair. The final output is the Pareto front over (fitness, token\_cost): the set of configurations for which no other configuration achieves both higher fitness and lower token cost.

#### 4.3.4 Pareto Configuration Selection

From the Pareto front, the configuration selected for each budget tier is the highest-fitness configuration that satisfies the hard token budget constraint. Components in the selected configuration are stored in descending relevance order (cosine similarity to cluster centroid), supporting bookend ordering at query time without additional computation.

**Optional LLM Pareto validation.** The `--validate-pareto` flag adds an optional post-processing step that evaluates the top Pareto candidates using live LLM calls on a held-out query set, re-ranking them by observed outcome quality before selection. This corrects for combination-level synergies that the proxy fitness function cannot capture at the cost of one LLM call per candidate per validation query.

#### 4.3.5 Per-Cluster Hyperparameter Tuning

The size penalty weight λ is tuned independently per cluster from logged turn data: for each cluster, the system performs a grid search over λ ∈ {0.01, 0.05, 0.1, 0.2, 0.5} and selects the value maximizing mean outcome quality on turns assigned to that cluster. The per-cluster λ values are stored in `per_cluster_lambda.json` and applied in subsequent evolutionary oracle builds via `--use-cluster-lambda`.

**Output.** `context_configs.json` — a structured artifact mapping (cluster\_id, budget\_tier) → optimal component list, augmented with metadata (fitness score, token cost, example queries).

### 4.4 Phase 4 — Classifier Layer (Path B)

**Goal.** Train a per-component logistic regression classifier on logged agent turns, producing per-component inclusion probability scores from query embeddings, and deploy it as a query-time context selector.

#### 4.4.1 Model Architecture

For each component cᵢ, the classifier estimates:

```
P(include cᵢ | q) = σ(wᵢᵀ · φ(q) + bᵢ)
```

where φ(q) is the query embedding vector and σ is the logistic sigmoid. The N binary classifiers (N = |C|) are trained jointly in a single `LogisticRegression` call using one-versus-rest (OVR) decomposition. Prediction thresholds are tuned by a separate threshold optimizer from the same logged data.

**Regularization.** The regularization strength C (inverse L2 penalty) is tuned by grid search over C ∈ {0.01, 0.1, 1.0, 10.0} using leave-one-out cross-validation on logged turns, optimizing component inclusion F1.

#### 4.4.2 Training Data

Turn log files (`turns_YYYY-WNN.jsonl`) supply the training data. Each record provides:
- `query_embedding: list[float]` — the query vector (TF-IDF or sentence-transformer)
- `component_set: list[str]` — the components included in the agent's context
- `outcome_quality: float` — a scalar ∈ [0, 1] measuring task outcome quality
- `exploration: {"explored": bool}` — flag indicating off-policy counterfactual turns

The trainer reads the most recent `max_weeks` weeks of logs (default: 8) and requires at least `min_turns` turns (default: 10) to proceed.

#### 4.4.3 Off-Policy Exploration

The classifier's training data has a fundamental coverage problem: it observes only the components that were actually included on each turn. Without counterfactual data, the classifier cannot estimate the quality of component sets it was never asked to evaluate. In the limiting case, a classifier trained exclusively on Path A's selections converges to an imitation of Path A's policy — it reproduces the cluster-based lookup rather than learning from outcome feedback.

The off-policy exploration mechanism addresses this by directing the turn logger to override the selector's component choice on a random ε-fraction of production turns (default ε = 0.1), sampling an alternative component set drawn from a broader distribution. These turns are flagged `exploration.explored = true` in the log and are the primary source of the counterfactual coverage that enables Path B to learn beyond Path A's policy.

The minimum exploration rate ε* required for guaranteed training coverage is derived analytically in §6.2. The mechanism through which this exploration enables Path B to learn a strictly superior policy — rather than merely a different imitation — is quantified by the convergence analysis in §6.2 and tested empirically in §7.4.

**Output.** `classifier_current.pkl` — the weight matrix W ∈ ℝ^{n×d}, bias vector **b** ∈ ℝⁿ, component name list, and inclusion thresholds. Versioned classifier snapshots are maintained in `classifier_registry.json` with validation F1 scores.

### 4.5 Context-Aware Ordering

The list of selected components is ordered before assembly into the context window. THALAMUS provides three ordering strategies:

**`relevance`.** Components sorted by descending cosine similarity between the component's example-query centroid and the current query embedding. This is the natural output order of both Path A (stored in descending relevance order at build time) and Path B (sorted by `max(P(include_i|q), 1 - P(include_i|q))` as a proxy for confidence-weighted relevance).

**`bookend`.** The relevance-sorted list is rearranged to place the most-relevant components at the positional extremes of the assembled context. For a list [c₁, c₂, …, cₖ] sorted by descending relevance, the bookend permutation is:

```
[c₁, c₃, c₅, …, c₆, c₄, c₂]
```

Odd-indexed positions from the sorted list fill the beginning of the context; even-indexed positions (reversed) fill the end. The result is that the two most-relevant components occupy position 1 and position k, the next two most-relevant occupy position 2 and position k−1, and so on. This directly counteracts the lost-in-the-middle attention decay [Liu et al., 2024]: the components the agent most needs appear at positions receiving the strongest attentional weight.

**`none`.** Components returned in the order stored in `context_configs.json`, without rearrangement.

**Research contribution C5.** Section 7.5 measures the quality difference between `bookend` and `relevance` ordering as a function of assembled context size, across task categories.

### 4.6 Automatic Budget Estimation

When the calling system does not specify a token budget, `BudgetEstimator` infers query complexity from query characteristics and maps it to one of three token budget tiers (small, medium, large). The estimation heuristics examine:

- Query length (word count)
- Presence of architectural scope markers ("design", "migrate", "plan", "refactor across")
- Presence of multi-file markers ("all files", "entire codebase", "module A and module B")
- Task type signals ("explain", "write a test", "implement", "review")

The estimator maps these signals to budget tiers through a rule-based classifier. Budget-adaptive selection allows simple queries to use a small component set and costly queries to use the full budget without requiring the caller to specify this per request.

**Research contribution C6.** Section 7.6 compares auto-budget selection against fixed-budget policies (small-fixed, medium-fixed, large-fixed) on a mixed-complexity task suite, measuring quality-per-token and task success rate per category.

---

## 5. Runtime Selection: ContextSelector

At query time, `ContextSelector` is the unified entry point implementing a two-path, gracefully-degrading selection protocol. It selects among available inference paths based on which artifacts are present and meet minimum quality thresholds.

### 5.1 Path B — ClassifierSelector

When a trained classifier is available and the logged turn count exceeds `min_turns`, `ClassifierSelector` is the primary inference path:

1. Compute query embedding φ(q) using the saved embedding backend.
2. Apply the weight matrix: **p** = σ(W · φ(q) + **b**) ∈ [0, 1]ⁿ.
3. Apply per-component inclusion thresholds: Sᵦ = {cᵢ : pᵢ ≥ θᵢ}.
4. Compute confidence: `confidence = mean(max(p, 1-p))`.
5. If `confidence < min_confidence`, return `None` (fall through to Path A).
6. Apply ordering strategy to Sᵦ; return result.

### 5.2 Path A — ClusterSelector

When Path B is unavailable, below confidence threshold, or not yet trained, `ClusterSelector` provides the primary selection:

1. Vectorize the query using the saved TF-IDF or sentence-transformer backend.
2. Assign to nearest K-means cluster: `k = argmin_j ||φ(q) - μ_j||`.
3. Look up the precomputed configuration: `S_A = context_configs[k][budget_tier]`.
4. Apply ordering strategy; return result.

### 5.3 Fallback Protocol

The full selection protocol is:

```
1. If ClassifierSelector is ready AND turn_count >= min_turns:
     result = ClassifierSelector.select(q)
     if result is not None AND result.confidence >= min_confidence:
         return result  [Path B active]

2. If context_configs.json exists:
     return ClusterSelector.select(q, budget, ordering)  [Path A active]

3. return None  [both unavailable]
```

This guarantee — returning `None` when no artifact is available — means THALAMUS can be deployed into an existing agent system without changing its behavior in the cold-start period: the calling system receives `None` and falls through to its existing context assembly logic.

**`active_path` property.** Returns `"classifier"`, `"cluster"`, or `"none"` — the currently active inference path — enabling operational monitoring of system maturity transitions.

**Research contribution C3.** The dual-path architecture is hypothesized to outperform either path alone across the full maturity curve: Path A outperforms retrieval baselines at cold start; Path B outperforms Path A at the mature checkpoint; the unified ContextSelector achieves the higher of the two at every measured point. Section 7.3 tests this hypothesis.

---

## 6. Research Extensions

The core four-phase architecture described in §4–5 is accompanied by four research extensions that address specific limitations. These extensions are fully implemented within THALAMUS and constitute the primary research agenda beyond the baseline evaluation.

### 6.1 Ablation Study (R2)

To decompose the quality improvement of THALAMUS into attributable components, four ablation selectors are implemented, each removing exactly one architectural feature.

**TopKSelector (removes the genetic algorithm).** Selects the k highest-ranked components by the marginal score `rank_score(cᵢ) = mean_score(cᵢ) × cosine(φ(q), φ_texts(cᵢ))`, without any combinatorial search. This ablation isolates the contribution of genetic combinatorial optimization from the contribution of per-component scoring.

**NoBookendSelector (removes bookend ordering).** Wraps `ClusterSelector` and forces `ordering="relevance"` regardless of the caller's specification. This ablation isolates the contribution of bookend edge-placement on long-context task quality.

**SingleBudgetSelector (removes budget adaptation).** Ignores the caller-supplied `budget` parameter and applies a fixed budget tier across all queries. Run separately with each of {small, medium, large}. This ablation isolates the contribution of automatic query-complexity-based budget estimation.

**PathBOnlySelector (removes Path A fallback).** Returns `None` if the classifier is unavailable; performs no fallback to the cluster-based oracle. This ablation isolates the quality contribution of the dual-path architecture by measuring Path B alone across the maturity curve.

All four ablation selectors implement `SelectorProtocol` and are evaluated identically through the same benchmark harness as the baseline selectors and the full THALAMUS configurations.

### 6.2 Cross-Path Co-Inclusion Transfer (R3a)

**Motivation.** The genetic algorithm's fitness function (§4.3.2) is linear in individual component scores and structurally cannot represent interaction effects. However, the classifier trained in Phase 4 implicitly encodes joint component utility: if two components cᵢ and c_j are often selected together and the corresponding turns have high outcome quality, their weight vectors **w**ᵢ and **w**_j in the classifier will be geometrically aligned — they will respond similarly to the same queries.

**Co-inclusion extraction.** The co-inclusion signal is extracted from the classifier's weight matrix W ∈ ℝ^{n×d} by computing the pairwise cosine similarity matrix between L2-normalized weight vectors:

```
W_norm[i] = w_i / ||w_i||_2

co_inclusion(cᵢ, c_j) = W_norm[i] · W_norm[j]  ∈ [−1, 1]
```

A score near +1 indicates that cᵢ and c_j tend to be needed by the same query types — they are jointly useful. A score near −1 indicates that they respond to opposite query types and are typically substitutes.

For a component set S, the set-level co-inclusion score is the mean of all pairwise scores:

```
co_inclusion(S) = (2 / |S|(|S|−1)) × ∑_{i < j, cᵢ,c_j ∈ S} co_inclusion(cᵢ, c_j)
```

**Fitness augmentation.** The co-inclusion signal is injected into the genetic algorithm as an additive augmentation of the marginal fitness score:

```
fitness_aug(S, k, B) = fitness(S, k, B) + λ × co_inclusion(S)
```

where λ (default 0.2) controls the weight given to the classifier's implicit joint-utility signal relative to the individual scoring signal. At λ = 0, this reduces to the original fitness formula.

This cross-path transfer operationalizes a loop: Path B's classifier, trained on outcome data, provides interaction signal that improves Path A's genetic algorithm, which produces better configurations for the next round of Path B training.

**CLI integration.** `thalamus-oracle evolve --use-classifier-prior --prior-lambda 0.2`

**Research contribution C7.** Section 7.7 measures the configuration quality difference between GA runs with and without classifier co-inclusion augmentation, conditioned on classifier maturity.

### 6.3 Analytically Derived Minimum Exploration Rate (R3b)

**Problem.** The off-policy exploration rate ε must be chosen before deployment. Too low, and the classifier receives insufficient counterfactual coverage, converging toward Path A's policy. Too high, and the quality cost of forced suboptimal selections on ε-fraction of production turns is excessive.

**Formal setup.** Model component selection as a multi-label contextual bandit:
- **State** s ∈ 𝒮: query embedding
- **Action** a ∈ {0, 1}ⁿ: component inclusion bitmask
- **Reward** r ∈ [0, 1]: outcome quality scalar
- **Path A policy** πᴬ(s): deterministic, returns argmax of cluster lookup

Under Path A alone, component cᵢ is included with probability p_A(cᵢ) = fraction of query clusters in which cᵢ appears in the optimal configuration. The logistic regression classifier requires at least n_min labeled examples of both cᵢ = True and cᵢ = False to reliably estimate wᵢ. Under exploration rate ε, the effective inclusion probability for training purposes is:

```
p_total(cᵢ) = (1 − ε) × p_A(cᵢ) + ε × 0.5
```

(The exploration policy samples cᵢ with probability 0.5 uniformly.) The expected count of cᵢ = True turns in T total turns is p_total(cᵢ) × T. The constraint p_total(cᵢ) × T ≥ n_min must be satisfied for every component.

**ε* derivation.** Solving for the minimum ε satisfying the coverage constraint for the hardest-to-cover component:

```
ε*(n_min, T_target) = max_i { max( 0, (n_min/T_target − p_A(cᵢ)) / (0.5 − p_A(cᵢ)) ) }
```

This is the minimum exploration rate guaranteeing that every component receives at least n_min inclusion turns within T_target total turns, enabling reliable logistic regression estimation for all N classifiers simultaneously.

**Default parameters:** n_min = 10 (minimum samples per class), T_target = 500 (target mature turn count). The derived ε* is written to `exploration_rate.json` and can be read by the turn logger to configure exploration automatically.

**CLI integration.** `thalamus-oracle tune --auto-exploration --n-min 10 --T-target 500`

**Convergence measurement.** In addition to the analytical derivation, empirical Path B convergence is measured by `ConvergenceAnalyzer`, which tracks the Jaccard agreement between Path A's selections and Path B's selections over rolling windows of logged turns:

```
agreement(t) = |S_A(q_t) ∩ S_B(q_t)| / |S_A(q_t) ∪ S_B(q_t)|
```

Convergence to Path A is defined as `final_agreement ≥ 0.85` over the most recent window. A Path B classifier showing high agreement with Path A has failed to learn beyond Path A's policy and indicates insufficient counterfactual coverage — a diagnostic that the exploration rate should be increased.

**Research contribution C4.** Section 7.4 tests the hypothesis that exploration rate ε ≥ ε* is necessary and sufficient for Path B to learn a policy not dominated by Path A, and that ε < ε* leads to convergence (high Jaccard agreement) to Path A's policy.

### 6.4 Set-Level Quality Model (R4)

**Motivation.** The marginal fitness function (§4.3.2) scores component sets by summing individual component quality scores weighted by relevance. This formulation structurally cannot represent:

- **Joint necessity:** Two components cᵢ and c_j each have individually mediocre scores but together are required for a class of tasks.
- **Mutual redundancy:** Either cᵢ or c_j alone suffices, so including both wastes tokens without improving quality.
- **Mutual interference:** cᵢ and c_j together generate contradictory instructions that lower quality relative to either alone.

All three interaction types are empirically observable in operational data: pairs of components whose joint inclusion shows outcome quality significantly above or below the sum of their individual scores.

**Feature representation.** Each (component set, cluster) pair is represented by a 14-dimensional feature vector:

| Dimension | Feature | Description |
|-----------|---------|-------------|
| 0 | mean\_score | Mean catalog score across S |
| 1 | std\_score | Standard deviation of catalog scores |
| 2 | min\_score | Minimum score in S |
| 3 | max\_score | Maximum score in S |
| 4 | n\_components | \|S\| total |
| 5 | n\_skills | Count of "skill" type in S |
| 6 | n\_tools | Count of "tool" type in S |
| 7 | n\_memory | Count of "memory\_section" type in S |
| 8 | n\_other | Count of other types |
| 9 | mean\_co\_inclusion | Mean pairwise co-inclusion score over S (§6.2) |
| 10 | min\_co\_inclusion | Minimum pairwise co-inclusion over S |
| 11 | max\_co\_inclusion | Maximum pairwise co-inclusion over S |
| 12 | cluster\_id | Raw cluster integer |
| 13 | cluster\_id\_norm | cluster\_id / 100 |

Dimensions 9–11 are populated from the classifier co-inclusion matrix (§6.2) when a trained classifier is available; they collapse to 0 when absent.

**Model.** A `GradientBoostingRegressor` with parameters:

```python
n_estimators   = 200
max_depth      = 3
learning_rate  = 0.05
subsample      = 0.8
min_samples_leaf = 5
random_state   = 42
```

is trained on a dataset assembled from all logged turns: each turn provides one (feature vector, outcome\_quality) pair. Off-policy exploration turns (§4.4.3) are deliberately included in the training set, as they provide coverage of component sets that the primary policy would not have selected and are therefore the primary source of signal for learning interaction effects.

**Fallback.** If model inference fails (e.g., a component set outside the model's training distribution), `SetQualityFitness` falls back to the marginal mean-score heuristic, preserving the GA's functionality.

**GA integration.** When `--fitness-model xgb` is passed to the oracle build step, the learned fitness function replaces the marginal formula in the genetic algorithm's inner loop. The token budget penalty remains unchanged; only the quality term is replaced:

```
fitness(S, k, B) = SetQualityFitness(S, k) − λ × (token_cost(S) / B_max)
```

**CLI integration.** `thalamus-oracle evolve --fitness-model xgb --fitness-model-dir /oracle/set_quality_model`

**Research contribution C8.** Section 7.8 compares oracle quality between GA runs using the marginal fitness formula and the learned set-level fitness, measuring configuration quality on the 120-task evaluation suite at the mature data checkpoint.

### 6.5 Cross-Deployment Meta-Learning (R5)

**Motivation.** Every new deployment of THALAMUS begins with the cold-start problem: no logged turns, no trained classifier, and a genetic algorithm fitness function relying entirely on synthetic LLM-scored priors. At cold start, Path A uses proxy scores that may not reflect the specific component quality distribution of the new deployment's skill library. If the new deployment shares components — identical or structurally equivalent — with previously operated deployments, their accumulated interaction evidence is directly applicable.

**Content fingerprinting.** Each component is identified by a content-addressed SHA-256 fingerprint:

```
fingerprint(cᵢ) = SHA-256(name(cᵢ) ∥ "\x00" ∥ description(cᵢ) ∥ "\x00" ∥ body(cᵢ))
```

where ∥ denotes string concatenation. Components with identical fingerprints have identical content and are treated as the same component regardless of which deployment they appear in. This enables direct quality evidence transfer without requiring deployment-specific metadata.

**Knowledge base.** The cross-deployment knowledge base is a flat JSON dictionary mapping fingerprint → aggregated statistics:

```json
{
  "<sha256>": {
    "name_hint": "web_search",
    "n_deployments": 3,
    "n_turns": 412,
    "mean_outcome_when_included": 0.73,
    "mean_outcome_when_excluded": 0.61,
    "mean_co_inclusion_score": 0.44,
    "updated_at": "2025-09-01T12:00:00Z"
  }
}
```

The `mean_outcome_when_included` field provides the cross-deployment prior quality estimate for the component. `mean_outcome_when_excluded` captures the baseline quality of turns on which the component was not present, enabling estimation of the component's marginal contribution in future deployments.

**Transfer protocol.** When initializing a new deployment:

1. Compute fingerprints for all components in the new deployment's oracle directory.
2. For each component, look up its fingerprint in the shared knowledge base.
3. For matched components, extract `mean_outcome_when_included` as the prior quality score.
4. Write `{component_name: prior_score}` to `transfer_priors.json` in the oracle directory.

**GA integration.** The genetic algorithm automatically detects `transfer_priors.json` and blends the KB prior with the synthetic score at build time:

```
blended_score(cᵢ) = α × kb_prior(cᵢ) + (1 − α) × synthetic_score(cᵢ)
```

The blend weight α decays as real turns accumulate:

```
α = exp(−n_turns / τ),   τ = 200
```

At cold start (n_turns = 0), α = 1.0 and the KB prior fully replaces the synthetic score for matched components. As the deployment matures, α → 0 and the local evidence overtakes the cross-deployment prior. Unmatched components retain their synthetic scores throughout.

**CLI integration.** `thalamus-oracle meta-init --oracle-dir /new/oracle --kb-path /shared/knowledge_base.json`

**Research contribution C9.** Section 7.9 measures the cold-start quality difference between deployments with and without transfer priors, quantifying the turn count reduction required to reach steady-state configuration quality. This requires multi-deployment experimental data.

---

## 7. Experimental Evaluation

### 7.1 Evaluation Protocol

All selectors — baselines, ablations, and THALAMUS configurations — implement `SelectorProtocol`, a common runtime-checkable interface:

```python
class SelectorProtocol(Protocol):
    def select(self, query: str, budget: str | None, ordering: str) -> dict | None: ...
    @property
    def active_path(self) -> str: ...
    @property
    def is_ready(self) -> bool: ...
```

The output format is unified across all selectors: `{"skills": [...], "memory": [...], "tools": [...], "source": str}` or `None`. This enables evaluation runs that swap any selector into any position without modifying the evaluation harness.

`BenchmarkRunner` accepts a dictionary of `{selector_name: SelectorProtocol}` instances. For each query, it runs every selector `n_repeats` times (default: 3) and records median latency. It computes per-query component overlap statistics between each selector and a designated reference selector (Jaccard similarity, precision, recall). Results are serialized as `EvalRun` JSON with a unique run ID and ISO timestamp.

Quality measurement is performed in a separate pass that executes the selected component configurations against the real agent on the fixed task suite and records `outcome_quality` per (task, selector). This two-phase design separates the deterministic benchmarking (latency, overlap) from the expensive quality measurement.

### 7.2 Baseline Selectors

Five baseline selectors are implemented, all using `ComponentCatalog` to read the same component pool as the THALAMUS configurations.

**AllSelector.** Returns the complete component pool regardless of query or budget. Serves as a theoretical quality upper bound at maximum token cost. Any THALAMUS configuration achieving quality near AllSelector has effectively solved the selection problem.

**RandomSelector.** Samples k components uniformly at random (seeded for reproducibility), where k = `catalog.count_for_budget(budget)`. Serves as the chance baseline. The THALAMUS system must outperform random by a substantial margin for the selection problem to be non-trivial.

**TFIDFSelector.** Fits a `TfidfVectorizer(max_features=2000, ngram_range=(1,2), sublinear_tf=True)` on all component example texts, vectorizes each query, and returns the top-k components by cosine similarity. This represents standard lexical retrieval — the most common approach in production RAG systems.

**BM25Selector.** Implements Okapi BM25 (parameters k₁ = 1.5, b = 0.75) without external dependencies. BM25 is the industry-standard lexical retrieval model and the strongest lexical baseline. Score formula:

```
score(D, Q) = ∑_{t ∈ Q} IDF(t) × tf(t, D) × (k₁ + 1) / (tf(t, D) + k₁(1 − b + b|D|/avgdl))

IDF(t) = log((N − df(t) + 0.5) / (df(t) + 0.5) + 1)
```

where N is the corpus size, df(t) the document frequency of term t, and avgdl the average document length.

**DenseSelector.** Encodes all component texts and each query using a sentence transformer (`all-MiniLM-L6-v2`) and returns the top-k components by L2-normalized cosine similarity. DenseSelector represents current RAG practice and the strongest semantic retrieval baseline.

All five baselines return exactly k components per budget tier, where k is derived from the oracle's `context_configs.json` as the average component count across clusters for that tier. This ensures each selector returns the same number of components and comparisons are token-cost-neutral.

### 7.3 THALAMUS Configurations Under Evaluation

| Configuration | Description |
|---|---|
| thalamus-path-a | `ClusterSelector` with `ordering=relevance` |
| thalamus-path-a-bookend | `ClusterSelector` with `ordering=bookend` |
| thalamus-path-b | `ClassifierSelector` alone (no Path A fallback) |
| thalamus-full | `ContextSelector` (automatic path selection) |
| thalamus-path-a-prior | `ClusterSelector` with R3a co-inclusion augmentation |
| thalamus-path-a-xgb | `ClusterSelector` with R4 learned fitness |
| thalamus-path-a-transfer | `ClusterSelector` with R5 transfer priors |

### 7.4 Task Suite

The evaluation suite consists of 120 deterministic tasks across four categories:

| Category | Count | Characteristics |
|---|---|---|
| Simple single-skill | 30 | One primary skill document required; memory and tools optional |
| Multi-skill | 40 | Two or more skill documents required; interaction effects expected |
| Architecture | 30 | Broad domain coverage; large assembled context; long-context ordering effects expected |
| Memory-dependent | 20 | Requires project-specific context from memory sections; tests memory component scoring |

Tasks are pre-defined with fixed expected outputs and LLM judge prompts. The evaluation suite is not resampled between runs; all comparisons use the identical 120-task fixed set.

### 7.5 Evaluation Metrics

| Metric | Computation | Unit |
|---|---|---|
| Task success rate | Binary LLM judge pass/fail per task | % |
| LLM judge score | Continuous quality score (0–1) per task, mean over suite | scalar |
| Tokens consumed | Context tokens assembled, mean over suite | tokens/turn |
| Quality-per-token | judge\_score / tokens\_consumed | scalar |
| Selection latency | Wall-clock ms from query to returned component list | ms |
| Sample complexity | Turns for Path B to exceed Path A quality by >2% | turns |
| Jaccard overlap | \|S ∩ S_ref\| / \|S ∪ S_ref\| vs reference | [0, 1] |
| Precision / Recall | vs reference selector | [0, 1] |

Latency, Jaccard, precision, and recall are computed by `BenchmarkRunner` without LLM calls and are available immediately. Task success rate and LLM judge score require the quality measurement pass.

### 7.6 Maturity Checkpoints

Each configuration is evaluated at three operational maturity checkpoints:

- **Cold start (0 turns):** No logged turn data. Path B unavailable. Measures the quality of the offline-prepared oracle against retrieval baselines without any operational evidence.
- **Early data (100 turns):** Modest logged turn data available. Path B may be activated depending on the `min_turns` threshold. Measures learning speed and the transition between paths.
- **Mature (500+ turns):** Sufficient data for stable classifier training and empirical score enrichment. Measures asymptotic quality and the benefit of the full dual-path system.

The sample complexity metric (C2, C9) is measured as the turn count at which Path B quality first exceeds Path A quality by at least 2%, interpolated from the maturity curve.

### 7.7 Results — Baseline Comparison (C1, C2, C3)

*[Results pending completion of quality measurement runs.]*

**Expected structure:**

| Selector | Success Rate (Cold) | Success Rate (100T) | Success Rate (Mature) | Tokens/Turn |
|---|---|---|---|---|
| AllSelector | [X] | [X] | [X] | [X] |
| RandomSelector | [X] | [X] | [X] | [X] |
| TFIDFSelector | [X] | [X] | [X] | [X] |
| BM25Selector | [X] | [X] | [X] | [X] |
| DenseSelector | [X] | [X] | [X] | [X] |
| thalamus-path-a | [X] | [X] | [X] | [X] |
| thalamus-path-a-bookend | [X] | [X] | [X] | [X] |
| thalamus-path-b | — | [X] | [X] | [X] |
| thalamus-full | [X] | [X] | [X] | [X] |

*[Contribution C1: thalamus-path-a vs retrieval baselines at cold start.]*
*[Contribution C2: thalamus-full vs thalamus-path-b turn-count-to-threshold, synthetic prior vs no prior.]*
*[Contribution C3: thalamus-full vs thalamus-path-a vs thalamus-path-b across maturity curve.]*

### 7.8 Results — Ablation Study (C4, C5, C6)

*[Results pending completion of quality measurement runs.]*

**Expected structure:**

| Configuration | Multi-skill | Architecture | Memory-dep. | Quality/Token |
|---|---|---|---|---|
| thalamus-path-a (full) | [X] | [X] | [X] | [X] |
| TopKSelector (−GA) | [X] | [X] | [X] | [X] |
| NoBookendSelector (−bookend) | [X] | [X] | [X] | [X] |
| SingleBudgetSelector-small | [X] | [X] | [X] | [X] |
| SingleBudgetSelector-medium | [X] | [X] | [X] | [X] |
| SingleBudgetSelector-large | [X] | [X] | [X] | [X] |
| PathBOnlySelector | [X] | [X] | [X] | [X] |

*[Contribution C4: convergence curve (Jaccard agreement with Path A vs turn count), with and without ε ≥ ε*.]*
*[Contribution C5: bookend vs relevance ordering, conditioned on assembled context size.]*
*[Contribution C6: auto-budget vs fixed-budget policies, quality-per-token on mixed complexity suite.]*

### 7.9 Results — Research Extensions (C7, C8, C9)

*[Results pending completion of multi-maturity quality measurement runs and multi-deployment data collection for R5.]*

**R3a (C7) expected structure:**

| Condition | Multi-skill Success Rate | Architecture Success Rate |
|---|---|---|
| Path A (marginal fitness) | [X] | [X] |
| Path A + co-inclusion prior (λ=0.2) | [X] | [X] |
| Ablation: λ=0.0 | [X] | [X] |
| Ablation: λ=0.5 | [X] | [X] |

**R4 (C8) expected structure:**

| Fitness Function | Success Rate (Mature) | Quality/Token |
|---|---|---|
| Marginal (hand-crafted) | [X] | [X] |
| SetQualityModel (XGB) | [X] | [X] |
| In-sample R² | [X] | — |

**R5 (C9) expected structure:**

| Condition | Cold-start Quality | Turns to Steady State |
|---|---|---|
| No transfer (synthetic prior only) | [X] | [X] |
| With transfer\_priors.json | [X] | [X] |
| Match rate (% components fingerprint-matched) | [X] | — |

---

## 8. Discussion

*[Discussion section to be completed after experimental results are available.]*

Key questions to be addressed:

1. At what library size does THALAMUS's quality advantage over retrieval baselines become significant? Is there a threshold below which the system is not worth deploying?

2. How does the quality of the genetic algorithm's initial oracle (Path A) depend on the accuracy of the LLM-generated component scores? How much does the lexical proxy underestimate true component quality?

3. Is the dual-path transition smooth — does Path B quality increase monotonically with turn count, or are there oscillation periods? What is the sensitivity of the transition threshold to the classifier's regularization choice?

4. How much of the quality improvement from the full THALAMUS system is attributable to the GA vs the classifier? Does the relative contribution shift with library size?

5. How sensitive is the set-quality model's improvement (C8) to the size of the training dataset? What is the minimum turn count at which the learned fitness begins to outperform the marginal formula?

6. What is the empirical match rate of the knowledge base transfer (R5) across real-world deployments with partially overlapping component libraries?

---

## 9. Limitations

**Lexical scoring proxies.** The Phase 1 component scoring step measures token overlap between agent outputs and expected answers using F1, bigram F1, and bag-of-words recall. These metrics capture surface-level lexical agreement rather than semantic correctness. A component that produces semantically correct answers in different words receives an underestimated score. Phase 2 enrichment and Pareto validation partially correct this by introducing real outcome evidence, but the individual component scoring step retains the proxy until sufficient operational data accumulates. Semantic scoring with BERTScore or G-Eval would improve accuracy at higher per-component evaluation cost.

**Fixed cluster topology.** K-means with a fixed K imposes a static partition on the query space. New query subtypes emerging after the oracle is built may not align with existing cluster boundaries, receiving assignments to the closest existing cluster rather than a dedicated configuration. Adaptive clustering (online K-means update, cluster splitting heuristics) would address this at the cost of increased system complexity.

**Linear individual classifiers.** The N binary logistic regression classifiers (one per component) cannot represent joint necessity: if skill A and tool B are useful only together, neither individual classifier receives positive examples of this joint condition, and the system learns no signal about their interaction. A multi-label model with shared representation (multi-task logistic regression, multilayer perceptron) would capture these interactions at higher data and compute requirements.

**Set-quality model data requirement.** The gradient-boosted regressor in R4 requires sufficient logged turns with filled outcome quality labels to learn meaningful interaction effects. The minimum usable data size — below which the model reduces to random prediction over the interaction features — depends on library size and interaction density. Based on standard gradient boosting sample complexity, a minimum of 200–500 labeled turns is likely required before the learned fitness outperforms the marginal formula.

**Off-policy exploration cost.** The exploration mechanism (§4.4.3) degrades quality on ε-fraction of production turns by design. The ε* derivation (§6.2) provides the minimum coverage-satisfying rate, but does not optimize the exploration-exploitation tradeoff: a lower ε degrades fewer turns but takes longer to achieve coverage; a higher ε achieves coverage faster at higher quality cost. Optimal exploration scheduling (annealing ε as coverage accumulates) is not implemented.

**Bookend assumptions.** The bookend ordering strategy assumes that attention decay toward the middle of the context window is monotone and applies uniformly to all model families. The empirical evidence for this pattern [Liu et al., 2024] is established for specific architectures and may not generalize to all production model families or all context structure types.

**Transfer fingerprint precision.** SHA-256 content fingerprinting ensures exact content matching but misses semantically equivalent components that differ in wording. A component that is functionally identical but textually different from a KB entry receives no transfer benefit. Semantic fingerprinting (embedding-based similarity) would increase recall at the cost of potential false matches.

---

## 10. Future Work

**Semantic scoring.** Replacing the lexical Phase 1 metrics with a semantic judge (LLM-based G-Eval scoring, BERTScore) would improve the accuracy of the initial component scores and reduce the number of operational turns required for enrichment to correct significant proxy errors.

**Continuous cluster adaptation.** An online clustering mechanism that splits or merges clusters as the query distribution evolves would prevent oracle staleness in deployments where the query distribution shifts over time (e.g., due to expanding skill libraries or changing user behavior).

**Multi-label joint classifier.** Replacing the N independent binary classifiers with a single multi-label model sharing a query representation (or a graphical model capturing component dependencies) would enable Path B to represent joint necessity and mutual redundancy directly, without requiring the post-hoc co-inclusion extraction step.

**Optimal exploration scheduling.** Adaptive exploration rate control — beginning at ε* and decaying as component coverage accumulates — would reduce the total quality cost of the exploration phase while maintaining the coverage guarantee.

**Component lifecycle management.** When a component in the library is substantially modified (new content, revised instructions), its fingerprint changes and all accumulated interaction evidence becomes stale. A system for detecting and responding to component version changes — re-zeroing enrichment scores for modified components while preserving evidence for unchanged ones — would improve robustness to skill library evolution.

**Interaction graph regularization.** The co-inclusion matrix (§6.2) provides a pairwise similarity structure over components. This structure could be used as a regularizer for the logistic regression classifiers (graph-regularized multi-label classification), encouraging components with high co-inclusion to have similar weight vectors.

**Multi-deployment federated learning.** The current R5 architecture transfers static quality priors via a central knowledge base. A federated variant — in which multiple deployments share gradient updates while preserving per-deployment privacy — would enable more nuanced quality signal transfer than the aggregate statistics currently written to the KB.

---

## 11. Conclusion

*[Conclusion to be completed after experimental results are available.]*

THALAMUS addresses Context Saturation — the compound failure of quality degradation, cost scaling, and positional attention decay that emerges in large AI agent deployments using unconditional context assembly. Its design is grounded in the observation that the expensive context selection decision does not require query-time LLM reasoning: it can be precomputed offline and served as a sub-millisecond lookup.

The core architecture combines LLM-driven offline component scoring, query space clustering, genetic combinatorial search, Bayesian score enrichment, and a classifier trained from operational data with analytically grounded off-policy exploration. Four research extensions — cross-path co-inclusion transfer, analytically derived minimum exploration rate, learned set-level quality fitness, and cross-deployment fingerprint transfer — address specific structural limitations of the core system and constitute a distinct empirical research agenda.

Nine testable research claims are defined, a comprehensive evaluation framework is implemented, and experimental infrastructure is in place. The system is fully deployed and the evaluation pipeline is in preparation.

*[Final quantitative summary of findings to be inserted upon completion of experimental runs.]*

---

## References

[Friedman, 2001] Jerome H. Friedman. "Greedy Function Approximation: A Gradient Boosting Machine." *The Annals of Statistics*, 29(5), 2001.

[Harman et al., 2012] Mark Harman, S. Afshin Mansouri, Yuanyuan Zhang. "Search-Based Software Engineering: Trends, Techniques and Applications." *ACM Computing Surveys*, 45(1), 2012.

[Jiang et al., 2023] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu. "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models." *EMNLP 2023*.

[Khattab et al., 2023] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, et al. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." *arXiv:2310.03714*, 2023.

[Langford & Zhang, 2007] John Langford and Tong Zhang. "The Epoch-Greedy Algorithm for Multi-armed Bandits with Side Information." *NeurIPS 2007*.

[Lewis et al., 2020] Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.

[Li et al., 2010] Lihong Li, Wei Chu, John Langford, Robert E. Schapire. "A Contextual-Bandit Approach to Personalized News Article Recommendation." *WWW 2010*.

[Li et al., 2023] Minghao Li, Yingxiu Zhao, Bowen Yu, et al. "Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning." *arXiv:2402.10110*, 2024.

[Liu et al., 2024] Nelson F. Liu, Kevin Lin, John Hewitt, et al. "Lost in the Middle: How Language Models Use Long Contexts." *Transactions of the Association for Computational Linguistics*, 12, 2024.

[Park et al., 2023] Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, et al. "Generative Agents: Interactive Simulacra of Human Behavior." *UIST 2023*.

[Wang et al., 2024] Lei Wang, Chen Ma, Xueyang Feng, et al. "A Survey on Large Language Model based Autonomous Agents." *Frontiers of Computer Science*, 2024.

[Xue et al., 2016] Bing Xue, Mengjie Zhang, Will N. Browne, Xin Yao. "A Survey on Evolutionary Computation Approaches to Feature Selection." *IEEE Transactions on Evolutionary Computation*, 20(4), 2016.

[Yao et al., 2023] Shunyu Yao, Jeffrey Zhao, Dian Yu, et al. "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*.

---

## Appendix A — Package Structure

```
thalamus_research/
└── src/thalamus_research/
    ├── baselines/              R1: Baseline comparison selectors
    │   ├── protocol.py             SelectorProtocol (common interface)
    │   ├── component_catalog.py    ComponentCatalog (component pool loader)
    │   ├── all_selector.py         AllSelector (full-library upper bound)
    │   ├── random_selector.py      RandomSelector (chance baseline)
    │   ├── tfidf_selector.py       TFIDFSelector (lexical retrieval)
    │   ├── bm25_selector.py        BM25Selector (Okapi BM25 retrieval)
    │   └── dense_selector.py       DenseSelector (sentence-transformer retrieval)
    │
    ├── ablations/              R2: Ablation selectors (one architectural feature removed each)
    │   ├── topk_selector.py        TopKSelector (no GA, marginal ranking only)
    │   ├── no_bookend_selector.py  NoBookendSelector (relevance ordering only)
    │   ├── single_budget_selector.py  SingleBudgetSelector (fixed budget tier)
    │   └── path_b_only_selector.py PathBOnlySelector (no Path A fallback)
    │
    ├── cross_path/             R3a: Classifier → GA co-inclusion transfer
    │   ├── co_inclusion_extractor.py  CoInclusionExtractor (pairwise W·Wᵀ cosine)
    │   └── fitness_augmentor.py       FitnessAugmentor (additive λ × co_inclusion)
    │
    ├── bandit/                 R3b: Off-policy exploration rate formalization
    │   ├── exploration_rate.py     ExplorationRateEstimator (ε* derivation)
    │   └── convergence.py          ConvergenceAnalyzer (Jaccard agreement curve)
    │
    ├── set_quality/            R4: Learned set-level fitness model
    │   ├── outcome_dataset.py      OutcomeDataset (turn log → training records)
    │   ├── interaction_features.py compute_feature_vector (14-dim featurization)
    │   ├── set_quality_model.py    SetQualityModel (GBR training + inference)
    │   └── fitness_function.py     SetQualityFitness (GA-compatible callable)
    │
    ├── meta_learning/          R5: Cross-deployment fingerprint transfer
    │   ├── component_fingerprint.py  fingerprint_catalog (SHA-256 content addressing)
    │   ├── knowledge_base.py         KnowledgeBase (flat JSON cross-deployment KB)
    │   └── transfer_initializer.py   TransferInitializer (writes transfer_priors.json)
    │
    ├── cli.py                  Research CLI entry point (7 subcommands)
    └── cli_args_parser.py      Argument parser for all subcommands
```

---

## Appendix B — CLI Reference

```bash
# R1: Baseline evaluation
thalamus-research baseline-lookup \
    --oracle-dir /oracle \
    --query "Set up a CI pipeline" \
    --method tfidf bm25 dense thalamus \
    --budget medium

thalamus-research eval \
    --oracle-dir /oracle \
    --query-file eval/tasks.jsonl \
    --reference thalamus \
    --method all random tfidf bm25 dense \
    --n-repeats 3 \
    --out results/r1_baselines.json

# R2: Ablation study
thalamus-research ablation \
    --oracle-dir /oracle \
    --query-file eval/tasks.jsonl \
    --out results/r2_ablation.json

# R3a: Cross-path co-inclusion analysis
thalamus-research cross-path \
    --oracle-dir /oracle \
    --top-pairs 20

thalamus-research cross-path \
    --oracle-dir /oracle \
    --augment-configs \
    --lam 0.2 \
    --out context_configs_augmented.json

# R3b: Exploration rate and convergence
thalamus-research bandit \
    --oracle-dir /oracle \
    --subcommand estimate-rate \
    --n-min 10 --T-target 500 \
    --out results/r3b_epsilon.json

thalamus-research bandit \
    --oracle-dir /oracle \
    --turn-log-dir /oracle/online_logs \
    --subcommand convergence \
    --window-size 50 \
    --budget medium \
    --out results/r3b_convergence.json

# R4: Set-level quality model
thalamus-research set-quality \
    --oracle-dir /oracle \
    --turn-log-dir /oracle/online_logs \
    --model-dir /oracle/set_quality_model \
    --subcommand train \
    --out results/r4_train.json

thalamus-research set-quality \
    --oracle-dir /oracle \
    --turn-log-dir /oracle/online_logs \
    --model-dir /oracle/set_quality_model \
    --subcommand evaluate \
    --out results/r4_eval.json

# R5: Cross-deployment meta-learning
thalamus-research meta-learning \
    --oracle-dir /oracle/deployment_1 \
    --kb-path ~/.jiuwenswarm/knowledge_base.json \
    --subcommand extract \
    --out results/r5_extract.json

thalamus-research meta-learning \
    --oracle-dir /oracle/new_deployment \
    --kb-path ~/.jiuwenswarm/knowledge_base.json \
    --subcommand transfer \
    --out results/r5_transfer.json
```

---

## Appendix C — Turn Log Schema

```json
{
  "turn_id": "t-20250901-001",
  "timestamp": "2025-09-01T09:14:22Z",
  "query_embedding": [0.12, -0.07, ...],
  "component_set": ["skill_ci", "skill_docker", "mem_project"],
  "outcome_quality": 0.84,
  "exploration": {
    "explored": false,
    "exploration_rate": 0.1
  }
}
```

`outcome_quality` is a scalar ∈ [0, 1] computed by `OutcomeScorer` from task completion signals (task success, follow-up corrections, LLM judge score). `exploration.explored = true` marks turns where the turn logger overrode the selector's choice for counterfactual coverage.

---

## Appendix D — Scoring Matrix Schema

```json
{
  "component_name": "skill_ci",
  "real_data": {
    "n_turns": 42,
    "updated_mean_score": 0.74
  },
  "baseline_cross_eval": [
    {
      "example_input": "How do I set up CI for a Python project?",
      "candidate_output": "...",
      "scores": {
        "f1": 0.71,
        "bigram_f1": 0.64,
        "bow_recall": 0.78,
        "length_ratio": 0.92
      }
    }
  ]
}
```

`real_data.updated_mean_score` is written by Phase 2 score enrichment. When present, it supersedes the synthetic mean in all downstream computations. Its absence indicates the component has not yet accumulated sufficient turn data for enrichment.

---

## Appendix E — EvalRun Schema

```json
{
  "run_id": "3a8f-...",
  "timestamp": "2025-09-01T12:00:00Z",
  "oracle_dir": "/oracle",
  "reference_selector": "thalamus-path-a",
  "selector_results": {
    "bm25": {
      "queries": [
        {
          "query": "Set up CI pipeline",
          "budget": "medium",
          "selected_components": ["skill_ci", "skill_docker"],
          "latency_ms": 4.2,
          "quality": null,
          "overlap": {
            "jaccard": 0.50,
            "precision": 0.67,
            "recall": 0.40
          }
        }
      ],
      "aggregate": {
        "mean_latency_ms": 4.1,
        "p95_latency_ms": 6.8,
        "mean_n_components": 3.2,
        "mean_jaccard": 0.48,
        "mean_precision": 0.61,
        "mean_recall": 0.44
      }
    }
  }
}
```

`quality: null` is filled by the separate quality measurement pass. Latency and overlap statistics are available immediately after a benchmark run without quality measurement.
