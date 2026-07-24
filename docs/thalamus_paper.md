# THALAMUS: Combinatorial Context Selection for Production AI Agents

**[Author names redacted for review]**

*Preprint. Under review.*

---

## Abstract

Production AI agents suffer from a structural failure we term **Context Saturation**: as an
agent's library of skills, memory sections, and tools grows, loading all components into every
context window degrades answer quality, scales token cost with library size rather than query
complexity, and distributes the model's finite attention across irrelevant content. The
structurally correct response — selecting only the components relevant to each query — is easy
to state and hard to implement, because it requires knowing, in advance and without LLM calls,
which combinations of components produce the best outcomes for each query type.

We present **THALAMUS**, a four-phase system that solves this problem by moving all expensive
reasoning offline. Its preparation pipeline uses LLM-driven evaluation to score every component,
blends those scores with real interaction evidence via Bayesian weighting, then runs a genetic
algorithm — without any LLM calls — over the exponential space of component subsets, finding
Pareto-optimal configurations for each query cluster and token budget. At query time, selection
reduces to a nearest-cluster lookup completed in under ten milliseconds. A supervised classifier
trained on logged agent turns refines this assignment as operational data accumulates; an
off-policy exploration mechanism prevents the classifier from converging to a biased policy. A
unified `ContextSelector` facade provides automatic path selection and graceful fallback at every
maturity stage.

We describe six empirically testable research contributions distinguishing THALAMUS from
retrieval-based baselines: (C1) the genetic algorithm finds higher-quality component sets than
independent top-k retrieval; (C2) LLM-generated synthetic priors reduce cold-start sample
complexity; (C3) dual-path architecture outperforms either path alone across the maturity curve;
(C4) off-policy exploration prevents Path B from converging to the policy of Path A; (C5) bookend
ordering improves performance on long-context tasks; (C6) budget-adaptive selection outperforms
fixed-budget policies. We describe the baseline comparison suite, the evaluation protocol, and
the research package (`thalamus/research/`) that implements them. Experimental results on the
120-task jiuwenswarm evaluation suite are in progress.

---

## 1. Introduction

A production AI agent receives its task specification through a context window assembled from
components: skill instruction documents, memory sections encoding project state and conventions,
and tool definitions. The design question at the heart of agentic system deployment is: *which
components should be in the context window for each incoming query?*

The answer is almost universally the same: all of them. This works in small deployments. A
six-skill agent with two memory sections and four tools loads sixty kilobytes into every context
window, pays a small token premium, and produces adequate results. This decision has no visible
cost early.

The cost appears at scale. When a production agent accumulates forty-plus skill documents,
multiple memory files encoding months of project history, and a rich tool library, the
uniform-inclusion strategy creates three compounding problems. First, **quality degradation**:
transformer attention is finite, and a context window dominated by irrelevant instructions
distributes weight across noise. The instructions most relevant to the query must compete with
unrelated content for the model's effective attention. Second, **cost scaling**: token cost scales
with library size, not query complexity. A query that genuinely requires two skills pays the same
as one that needs twenty. Third, **the lost-in-the-middle effect**: empirical work on long-context
models [Liu et al., 2024] shows that models attend preferentially to the beginning and end of
their context, with material in the middle receiving substantially weaker effective weight.
Relevant components placed in the middle of a forty-document context suffer from this attention
decay regardless of their relevance.

We name this compound failure **Context Saturation** and present THALAMUS as a system
engineered to close it.

The core insight is simple: the expensive question — "which components does this query need?" —
does not need to be answered at query time. If we can precompute, for each identifiable query
type and each token budget, the optimal component configuration, then query-time selection
reduces to a fast lookup. THALAMUS makes this precomputation tractable by (a) using LLM-driven
evaluation to produce per-component relevance scores offline, (b) clustering the query space to
reduce the number of configurations to precompute, and (c) running a genetic algorithm to search
over the exponential space of component combinations without requiring LLM calls in the inner
loop.

This paper presents the complete THALAMUS design and the research methodology for evaluating it.
We make the following contributions:

- **C1 — Combinatorial oracle outperforms independent retrieval (§4.3):** We hypothesize and
  will empirically show that the GA-based oracle finds component sets that outperform the
  top-k ranked by individual relevance scores, particularly for tasks requiring multiple
  interacting components.
- **C2 — Synthetic priors reduce cold-start sample complexity (§4.2):** LLM-generated scores
  provide a warm start that reduces the number of logged turns required for the classifier
  (Path B) to exceed the GA baseline (Path A) quality.
- **C3 — Dual-path architecture dominates either path alone (§5):** The ContextSelector
  system, which deploys Path A immediately and transitions to Path B when sufficient data
  exists, achieves higher quality than either path alone across the full maturity curve.
- **C4 — Off-policy exploration is necessary for Path B convergence (§4.4):** Without
  off-policy exploration, Path B converges to Path A's policy; with exploration, it learns
  a strictly better policy on tasks where Path A's cluster boundaries are imprecise.
- **C5 — Bookend ordering improves long-context quality (§4.5):** Placing the most-relevant
  components at the edges of the context window yields measurable quality improvements on
  tasks that require processing more than 3k context tokens.
- **C6 — Budget-adaptive selection outperforms fixed-budget (§4.6):** Adapting the selected
  component count to inferred query complexity outperforms any single fixed-budget policy
  on a mixed-complexity task suite.

The paper is organized as follows. Section 2 covers related work. Section 3 formalizes the
problem. Section 4 presents the THALAMUS architecture. Section 5 describes the runtime and
the dual-path transition. Section 6 presents the evaluation framework, baselines, and research
agenda. Section 7 discusses limitations and future work. Section 8 concludes.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) [Lewis et al., 2020] retrieves documents from a corpus
at query time using dense or sparse similarity, prepending them to the model's input. RAG has
become the dominant approach for knowledge-grounded generation. Its limitations in the agentic
context are structural: (a) it treats each component independently, selecting by individual
relevance rather than joint utility; (b) it does not account for token budgets or combination
effects; (c) it requires retrieval calls at query time. THALAMUS addresses all three by
precomputing optimal combinations offline and completing selection via a lookup.

### 2.2 LLM-Based Agent Frameworks

The agent framework literature [Park et al., 2023; Yao et al., 2023; Wang et al., 2024] has
focused on multi-step reasoning, tool use, and memory. With rare exceptions, these frameworks
use fixed context assembly: all available tools, all memory sections, all skill prompts are
included in every call. THALAMUS is orthogonal to the reasoning architecture: it operates at
the context assembly layer and is compatible with any downstream agent framework.

### 2.3 Prompt Compression and Selection

Selective Reflection-Tuning [Li et al., 2023] and similar methods compress or distill prompt
content to reduce token costs. These methods operate on individual documents and do not address
the combinatorial selection problem. Prompt selection methods [Khattab et al., 2023; Dspy] use
optimization to find effective prompt configurations, but optimize over prompting strategies, not
component inclusion sets. THALAMUS optimizes the latter.

### 2.4 Long-Context Attention and Ordering

Liu et al. [2024] ("Lost in the Middle") is the empirical foundation for the bookend ordering
strategy in §4.5. Their finding that models attend preferentially to the beginning and end of
long contexts motivates placing the most-relevant components at context edges. THALAMUS's
bookend ordering formalizes this into a deployable selection policy.

### 2.5 Contextual Bandits and Off-Policy Learning

The off-policy exploration mechanism in Path B (§4.4) is a practical instantiation of
contextual bandit theory [Langford & Zhang, 2007; Li et al., 2010]. The classification of
component selection as a multi-label contextual bandit problem — where state is the query
embedding, action is the component bitmask, and reward is the outcome quality scalar — motivates
the minimum exploration rate derivation implemented in Phase R3b (§6.6).

### 2.6 Genetic Algorithms for Combinatorial Optimization

Genetic algorithms have been applied to combinatorial optimization in software configuration
[Harman et al., 2012] and feature selection [Xue et al., 2016]. THALAMUS applies them to
the novel problem of agent context window optimization, where the fitness function combines
component relevance, cluster coherence, and token budget constraint.

---

## 3. Problem Formulation

Let **C** = {c₁, c₂, …, cₙ} be the agent's component library — the union of its skill
instruction documents, memory sections, and tool definitions. Each component cᵢ has a token
cost τ(cᵢ).

Let **q** be an incoming user query (a natural language string). Let **B** ∈ ℝ⁺ be a token
budget for the combined component context.

**Definition (Context Selection Problem).** Find a subset **S** ⊆ **C** such that:

```
∑_{c ∈ S} τ(c) ≤ B
```

and the expected task outcome quality E[outcome(S, q)] is maximized, where outcome(·) is the
quality of the agent's response when its context is assembled from **S**.

**Why this is hard.** The search space has 2ⁿ candidate sets. Direct evaluation of each set
requires a live LLM call and a ground-truth reference. Individual component relevance scores
are imperfect proxies for joint set utility: components can be jointly necessary, redundant, or
mutually interfering. Real outcome labels arrive with delay and are sparse. No closed-form
solution exists for arbitrary query distributions and component libraries.

**THALAMUS's approach.** Approximate the optimal selection function by:
1. Partitioning the query space into K clusters via unsupervised embedding clustering.
2. For each cluster k and budget tier B, finding the set S*(k, B) that maximizes a proxy
   fitness function (§4.3) using a genetic algorithm.
3. Storing the resulting configurations in a lookup table: (cluster, budget) → S*.
4. At query time: assign the query to its nearest cluster and return the precomputed S*.
5. Refining the lookup with a supervised classifier (Path B) as operational data accumulates.

---

## 4. The THALAMUS Architecture

THALAMUS is organized in four sequential phases. Phases 1–3 constitute the offline preparation
pipeline. Phase 4 is an optional online learning layer.

### 4.1 Phase 1 — Component Scoring

**Goal.** Produce a scored matrix for every component in C, capturing its relevance signal
across a representative sample of query types.

For each component cᵢ, the `component_scoring` pipeline:

1. Generates M synthetic (query, expected_answer) pairs covering the component's domain,
   using a language model prompted with the component's text. Default M = 5.
2. Runs the agent with only cᵢ in context, on each of the M queries, and collects its output.
3. Evaluates each output against the expected answer with four lexical metrics: F1 token
   overlap, bigram F1 (captures phrase-level fluency), bag-of-words recall, and length ratio.
4. Writes the resulting `(metric → score)` vectors to a structured JSON scoring matrix.

**Output:** `scoring_matrix_skill_*.json`, `scoring_matrix_mem_*.json`,
`scoring_matrix_tool_*.json` — one file per component source file.

**Scoring matrix schema:**
```json
{
  "component_name": "deploy_ci",
  "baseline_cross_eval": [
    {
      "example_input": "How do I set up CI for a Python project?",
      "scores": { "f1": 0.71, "bigram_f1": 0.64, "bow_recall": 0.78, "length_ratio": 0.92 }
    }
  ]
}
```

**Mean score.** Each component's `mean_score` is the average of `scores.f1` across all
`baseline_cross_eval` rows. This scalar is used in the GA fitness function (§4.3).

**Limitations.** Lexical metrics measure token overlap, not semantic correctness. A component
that produces semantically correct answers in different words is underscored. This is a known
proxy — corrected progressively by Phase 2 enrichment and Phase 3 Pareto validation.

### 4.2 Phase 2 — Score Enrichment

**Goal.** Blend LLM-generated synthetic scores with real interaction evidence to reduce
dependency on the proxy metrics.

As the deployed agent accumulates operational turn logs, the `enrich` build stage:

1. Reads `turns_YYYY-WNN.jsonl` — one line per agent turn, with fields `{query_embedding,
   component_set, outcome_quality}`.
2. For each component cᵢ, extracts turns where cᵢ was included and groups by query cluster.
3. Computes per-cluster empirical mean outcome when cᵢ was included.
4. Blends the empirical mean with the synthetic score via Bayesian weighting:

```
enriched_score(cᵢ, k) = (n_turns × empirical_mean + α × synthetic_mean) / (n_turns + α)
```

where α is the pseudo-count prior weight (default: α = 5).

The enriched scores replace the raw synthetic scores in the scoring matrices, shifting the
fitness function toward real outcomes as data accumulates while preserving the synthetic prior
in the cold-start regime.

**Research question (C2).** Does the synthetic prior (α > 0) measurably reduce the number of
logged turns required for Path B to exceed Path A quality? We measure sample complexity as the
number of turns at which Path B's quality first exceeds Path A by a margin of 2%.

### 4.3 Phase 3 — Evolutionary Oracle (Path A)

**Goal.** Precompute optimal component configurations for every (query cluster, token budget)
pair, using a genetic algorithm that searches over component bitmasks.

**Query clustering.** The query space is partitioned into K clusters (default K = 20) using
K-means on either TF-IDF vectors or sentence-transformer embeddings of historical query texts
or component example inputs. The cluster centroid model is saved to `context_configs.pkl` for
query-time assignment.

**Fitness function.** For a candidate component set S, query cluster k, and budget B:

```
fitness(S, k, B) = mean_score(S, k) × relevance(S, k)   if token_cost(S) ≤ B
                   −∞                                      otherwise

mean_score(S, k) = (1/|S|) ∑_{c ∈ S} enriched_score(c, k)

relevance(S, k)  = (1/|S|) ∑_{c ∈ S} cosine(embed(c), centroid(k))
```

The fitness is the product of mean enriched score and mean query-cluster relevance. The
token budget constraint is enforced as a hard penalty. No interaction terms appear in this
formulation — this is the key limitation addressed by research Phase R4 (§6.4).

**Genetic algorithm.** Each individual is a binary bitmask of length n (n = number of
components). Fitness is evaluated by the formula above. The GA runs:
- Population: 50 individuals, tournament selection (k=3), uniform crossover, bit-flip mutation
- Termination: 200 generations or plateau detection
- Output: Pareto front over (fitness, token cost) — one configuration per budget tier

**Optional Pareto validation.** The `--validate-pareto` flag sends the top Pareto candidates
through real LLM evaluation on a held-out query set, correcting for combination-level synergies
that the proxy fitness function cannot capture. This is expensive (one LLM call per candidate
per query) and is optional.

**Output:** `context_configs.json` — mapping (cluster_id, budget_tier) → optimal component list,
stored in descending relevance order.

**Research question (C1).** Does the GA-optimized configuration outperform the top-k
independent component ranking (TF-IDF, BM25, dense)? We measure on the 120-task evaluation
suite across three budget tiers, comparing task success rate and quality-per-token.

### 4.4 Phase 4 — Classifier Layer (Path B)

**Goal.** Train a per-component logistic regression classifier on logged agent turns, producing
per-component inclusion probabilities from query embeddings.

**Model.** For each component cᵢ, a logistic regression model predicts:

```
P(include cᵢ | q) = σ(wᵢᵀ · embed(q) + bᵢ)
```

where `embed(q)` is the query embedding vector. Inclusion is decided by thresholding at 0.5.
Models are trained jointly in a single pass using `sklearn.linear_model.LogisticRegression`
with `multi_class='ovr'`.

**Training data.** Turn logs: `turns_YYYY-WNN.jsonl`. Each row: `{query_embedding: [...],
component_set: ["skill_a", ...], outcome_quality: 0.84, exploration: {explored: false}}`.
The trainer reads the most recent `max_weeks` weeks (default: 8). Training requires at least
`min_turns` turns (default: 10).

**Off-policy exploration (C4).** The classifier's training data is fundamentally biased:
it observes the components that were included, not the counterfactual alternatives. Without
intervention, the classifier converges to the selection policy of Path A — it learns to prefer
the components Path A chose, reinforcing an existing policy rather than learning a better one.

The exploration mechanism injects counterfactual inclusions at rate ε (default: 0.1). On
ε-fraction of turns, the turn logger overrides the selector's choice and samples a modified
component set from the full distribution. Turns flagged `exploration.explored = true` provide
coverage of component–query pairs that the selector would not have chosen organically.

**Research question (C4).** What is the minimum exploration rate ε* required for Path B to
converge to a policy not dominated by Path A? Phase R3b derives this formally from the
bandit theory (implemented in `research/bandit/exploration_rate.py`; empirical sweep pending).

**Output:** `classifier.pkl` — weight matrix W, bias vector b, component name list.

### 4.5 Context-Aware Ordering

Retrieved component lists are ordered before assembly. THALAMUS provides three ordering modes:

- **`relevance` (default):** Components sorted by their relevance score to the query cluster.
  Appropriate for agents that read the context sequentially.
- **`bookend`:** The relevance-sorted list is rearranged to place the most-relevant components
  at the edges. Formally, for a list [c₁, c₂, …, cₖ] sorted by descending relevance, the
  bookend order is [c₁, c₃, c₅, …, c₆, c₄, c₂]. This counteracts the lost-in-the-middle
  attention decay: the components the agent most needs appear at positions receiving the
  strongest attentional weight [Liu et al., 2024].
- **`none`:** Raw stored order, no rearrangement.

**Research question (C5).** On what proportion of task types does bookend ordering improve
quality? We hypothesize it benefits tasks with combined context above 3k tokens.

### 4.6 Automatic Budget Estimation

When the caller does not specify a token budget, `BudgetEstimator` infers it from query
characteristics: length, detected task type (architecture vs. single-file), and presence of
multi-file markers. The estimator maps inferred task complexity to one of three budget tiers
(compact, medium, full).

**Research question (C6).** Does budget-adaptive selection outperform any fixed-budget policy
on a mixed-complexity evaluation suite? We compare: auto-budget, compact-fixed, medium-fixed,
full-fixed. The auto-budget strategy is expected to match the best fixed policy on each task
type while consuming fewer tokens on simple tasks.

---

## 5. Runtime Selection: ContextSelector

At query time, `ContextSelector` is the unified entry point. It implements a two-path fallback
protocol:

```
if classifier.pkl exists AND turn_count >= min_turns:
    result = ClassifierSelector.select(query)
    if result.confidence >= threshold:
        return result
fall back to:
if context_configs.json exists:
    return ClusterSelector.select(query, budget, ordering)
return None
```

**Path B — ClassifierSelector.** Computes `embed(query)`, applies the classifier's weight
matrix to get per-component probabilities, thresholds at 0.5, assembles the component set,
and returns a confidence score (mean of max(p, 1−p) across components).

**Path A — ClusterSelector.** Vectorizes the query with the saved backend (TF-IDF or
sentence-transformer), assigns it to the nearest K-means cluster, and retrieves the
precomputed configuration from `context_configs.json` for the requested budget tier. Applies
ordering before returning.

**Fallback chain.** If neither artifact is available, `ContextSelector.select()` returns
`None`, allowing the caller to fall back to its own default. This guarantee means THALAMUS
can be deployed into any agent system without breaking existing behavior in the cold-start
regime.

**`active_path` property.** Returns `"classifier"`, `"cluster"`, or `"none"` — the currently
active inference path. Allows monitoring of system maturity transitions.

**Research question (C3).** The dual-path system is hypothesized to outperform either path
alone across the full maturity curve. Specifically: Path A outperforms retrieval baselines at
cold start (when the classifier has insufficient training data), while Path B outperforms Path A
as logged turns accumulate (because it learns from real outcomes rather than proxy scores). The
ContextSelector's automatic transition should therefore achieve higher quality than either path
alone at every measured maturity checkpoint.

---

## 6. Evaluation Framework

### 6.1 Baseline Selectors

We compare THALAMUS against five baselines, all implemented in `thalamus/research/baselines/`
and all implementing `SelectorProtocol` — the same interface as `ContextSelector`.

| Name | Implementation | Key properties |
|---|---|---|
| **AllSelector** | Returns all components | Quality upper bound; token cost = full library |
| **RandomSelector** | Seeded uniform random sample of k components | Chance baseline; reproducible |
| **TFIDFSelector** | `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` + cosine similarity | Standard lexical retrieval; no training |
| **BM25Selector** | Full Okapi BM25 (k₁=1.5, b=0.75); implemented without external dependencies | Strong lexical baseline; industry standard in search |
| **DenseSelector** | Sentence-transformer L2-normalized cosine similarity | Dense semantic retrieval; represents current RAG practice |

All baselines return k components, where k is derived from the oracle's `context_configs.json`
(average component count per budget tier) to ensure a fair comparison — each selector returns
the same number of components per budget tier.

The THALAMUS configurations under evaluation are:

| Name | Description |
|---|---|
| **thalamus-path-a** | `ClusterSelector` with `ordering=relevance` |
| **thalamus-path-a-bookend** | `ClusterSelector` with `ordering=bookend` |
| **thalamus-path-b** | `ClassifierSelector` (requires turn logs) |
| **thalamus-full** | `ContextSelector` (auto-selects best available path) |

### 6.2 Evaluation Harness

The `thalamus/research/evaluation/` package implements a structured benchmark harness.

**`BenchmarkRunner`** accepts a dictionary of `{selector_name: SelectorProtocol}` and a
designated reference selector. For each query in the task suite, it runs every selector
`n_repeats` times (default: 3) and records the median latency. It computes per-query component
overlap (Jaccard, precision, recall) between each baseline and the reference selector.
Results are serialized as an `EvalRun` JSON file with a unique `run_id` and ISO timestamp.

**`EvalRun` schema:**
```json
{
  "run_id": "3a8f…",
  "timestamp": "2025-01-15T12:00:00",
  "selector_results": {
    "tfidf": {
      "queries": [
        {
          "query": "Set up CI pipeline",
          "budget": "medium",
          "selected_components": ["skill_ci", "skill_docker"],
          "latency_ms": 4.2,
          "quality": null,
          "overlap": {"jaccard": 0.5, "precision": 0.67, "recall": 0.40}
        }
      ],
      "aggregate": {
        "mean_latency_ms": 4.1,
        "p95_latency_ms": 6.8,
        "mean_n_components": 3.2,
        "mean_jaccard": 0.48
      }
    }
  }
}
```

The `quality: null` field is a placeholder. Quality scores are filled by a subsequent
jiuwenswarm pass that executes each selected component configuration against the real agent
on the fixed task suite. This two-phase design separates the deterministic benchmarking
(latency, overlap) from the expensive quality measurement.

**CLI:**
```bash
# Quick baseline comparison on a single query
thalamus-research baseline-lookup --oracle-dir /oracle \
    --query "Set up a CI pipeline" \
    --method tfidf bm25 thalamus --budget medium

# Full benchmark run
thalamus-research eval --oracle-dir /oracle \
    --query-file eval/tasks.jsonl \
    --reference thalamus \
    --method all random tfidf bm25 \
    --n-repeats 3 \
    --out results/run_01.json
```

### 6.3 Task Suite

The evaluation suite consists of 120 deterministic tasks in four categories:

| Category | Count | Examples |
|---|---|---|
| Simple single-skill | 30 | "Write a unit test for function X", "Explain error Y" |
| Multi-skill | 40 | "Refactor module A then update its tests", "Add CI and update docs" |
| Architecture | 30 | "Plan a migration from monolith to services", "Design the auth layer" |
| Memory-dependent | 20 | "Continue the task from last session", "Apply the team convention for X" |

Tasks are pre-defined with expected outputs and LLM judge prompts. This fixed test set is
not sampled at evaluation time, ensuring reproducibility across evaluation runs.

### 6.4 Metrics

| Metric | Type | Description |
|---|---|---|
| Task success rate | Quality | Binary pass/fail per task, judged by LLM judge |
| LLM judge score | Quality | Continuous 0–1 quality score per task |
| Tokens consumed | Efficiency | Context tokens per turn, averaged over task suite |
| Quality-per-token | Joint | `judge_score / tokens_consumed` |
| Latency | Runtime | ms from query receipt to component list returned |
| Sample complexity | Learning speed | Turns for Path B to exceed Path A quality by >2% |
| Jaccard overlap | Component agreement | `|S ∩ S_ref| / |S ∪ S_ref|` vs reference selector |
| Precision / Recall | Component agreement | Precision and recall of S vs S_ref |

Latency and overlap are computed by `BenchmarkRunner` without LLM calls. Quality metrics
require the jiuwenswarm quality measurement pass.

### 6.5 Maturity Levels

Each configuration is evaluated at three checkpoints:

- **Cold start (0 turns):** Measures quality before any operational data. Path A and baselines
  only; Path B is unavailable.
- **Early data (100 turns):** Measures learning speed. Path B may or may not exceed Path A.
- **Mature (500+ turns):** Measures asymptotic quality. Path B is expected to dominate.

Sample complexity results compare configurations across this maturity curve.

### 6.6 Research Phase Schedule

| Phase | Research question | Implementation status |
|---|---|---|
| R1 | Does Path A beat retrieval baselines at cold start? | ✓ Baselines + harness implemented (`research/baselines/`, `research/evaluation/`); quality runs pending |
| R2 | Which THALAMUS components drive improvement? | ✓ All query-time ablation selectors implemented (`research/ablations/`: `TopKSelector`, `NoBookendSelector`, `SingleBudgetSelector`, `PathBOnlySelector`); quality runs pending |
| R3a | Does classifier-to-GA transfer improve Path A? | ✓ `CoInclusionExtractor` + `FitnessAugmentor` implemented (`research/cross_path/`); wiring into `oracle_builder evolve` pending |
| R3b | What is the minimum exploration rate ε\* for Path B convergence? | ✓ `ExplorationRateEstimator` + `ConvergenceAnalyzer` implemented (`research/bandit/`); empirical ε sweep pending |
| R4 | Does a learned set-level fitness outperform the hand-crafted formula? | Design complete; requires ~1000 turns of logged data |
| R5 | Can cross-deployment warm-start reduce cold-start time? | Design complete; requires multi-deployment data |

---

## 7. Limitations and Future Work

**Lexical scoring proxies.** Component scores derived from F1, bigram F1, and bag-of-words
recall measure token overlap, not semantic correctness. Phase 2 enrichment and Pareto validation
partially correct this, but the individual component scoring step remains a proxy. A semantic
judge (G-Eval, BERTScore) would improve accuracy at higher cost.

**Fixed cluster count.** K-means with fixed K = 20 cannot adapt to a shifting query
distribution. New query subtypes emerging post-deployment may fall into incorrect clusters.
Adaptive clustering or online cluster splitting would address this.

**Linear fitness function.** The GA fitness formula is a weighted sum of individual component
scores. It structurally cannot model interaction effects: two components that are jointly
necessary but individually mediocre both receive low fitness values and are likely excluded.
Phase R4 will replace this with a gradient-boosting set-level quality model.

**Independent classifiers.** The N separate binary classifiers cannot model joint necessity.
If skill A and tool B are only useful together, neither classifier learns this. A multi-label
classifier with a shared representation would capture these interactions but requires more data.

**Off-policy exploration cost.** The ε exploration mechanism improves classifier training at
the cost of degraded quality on ε-fraction of production turns. The optimal exploration rate
trades off exploration benefit against exploitation cost, and is not derived from data in the
current system. Phase R3b has derived the theoretically optimal rate analytically
(`ExplorationRateEstimator` in `research/bandit/`); empirical validation over a range of ε
values on the jiuwenswarm task suite remains pending.

**Bookend assumptions.** The bookend ordering strategy assumes monotone attention decay
toward the middle of the context. This is empirically supported for specific model families
[Liu et al., 2024] but may not generalize universally. Evaluation is recommended before
applying bookend ordering in deployments with unusual context structures.

**Future work.** Phase R3a has implemented co-inclusion extraction and fitness augmentation
tools (`research/cross_path/`); wiring into `oracle_builder evolve --use-classifier-prior`
remains pending. Phase R3b has implemented the ε* derivation and convergence measurement
tools (`research/bandit/`); empirical sweep over exploration rates remains pending.
Phase R4 will replace the linear fitness function with a gradient-boosting set-level quality
model. Phase R5 will extend the system to multi-deployment warm-start via a shared component
knowledge base.

---

## 8. Conclusion

We have presented THALAMUS, a system for adaptive context selection in production AI agents.
The core design principle is simple: move all expensive reasoning offline. The preparation
pipeline scores components, blends scores with real evidence, and runs a genetic algorithm to
find optimal component combinations for each query cluster and token budget. At query time,
selection reduces to a millisecond lookup. A classifier layer, trained on logged turns with
off-policy exploration, refines the lookup as operational data accumulates. A unified
ContextSelector facade provides automatic path selection and graceful fallback.

We have formalized six research contributions that distinguish THALAMUS from retrieval-based
baselines and defined an evaluation protocol and baseline comparison suite to test them. The
baseline selectors and evaluation harness are implemented in `thalamus/research/`. Experimental
results on the 120-task jiuwenswarm suite will be reported as the quality measurement pipeline
is completed.

The system is deployed in production in jiuwenswarm. Every research phase simultaneously
produces an artifact deployed in that system, ensuring that the research agenda and the
engineering agenda reinforce rather than compete with each other.

---

## References

[Lewis et al., 2020] Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. "Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.

[Liu et al., 2024] Nelson F. Liu, Kevin Lin, John Hewitt, et al. "Lost in the Middle: How
Language Models Use Long Contexts." *Transactions of the Association for Computational
Linguistics*, 12, 2024.

[Park et al., 2023] Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, et al. "Generative
Agents: Interactive Simulacra of Human Behavior." *UIST 2023*.

[Yao et al., 2023] Shunyu Yao, Jeffrey Zhao, Dian Yu, et al. "ReAct: Synergizing Reasoning
and Acting in Language Models." *ICLR 2023*.

[Wang et al., 2024] Lei Wang, Chen Ma, Xueyang Feng, et al. "A Survey on Large Language
Model based Autonomous Agents." *Frontiers of Computer Science*, 2024.

[Li et al., 2023] Minghao Li, Yingxiu Zhao, Bowen Yu, et al. "Selective Reflection-Tuning:
Student-Selected Data Recycling for LLM Instruction-Tuning." *arXiv:2402.10110*, 2024.

[Khattab et al., 2023] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, et al. "DSPy:
Compiling Declarative Language Model Calls into Self-Improving Pipelines."
*arXiv:2310.03714*, 2023.

[Langford & Zhang, 2007] John Langford and Tong Zhang. "The Epoch-Greedy Algorithm for
Multi-armed Bandits with Side Information." *NeurIPS 2007*.

[Li et al., 2010] Lihong Li, Wei Chu, John Langford, Robert E. Schapire. "A Contextual-Bandit
Approach to Personalized News Article Recommendation." *WWW 2010*.

[Harman et al., 2012] Mark Harman, S. Afshin Mansouri, Yuanyuan Zhang. "Search-Based Software
Engineering: Trends, Techniques and Applications." *ACM Computing Surveys*, 45(1), 2012.

[Xue et al., 2016] Bing Xue, Mengjie Zhang, Will N. Browne, Xin Yao. "A Survey on
Evolutionary Computation Approaches to Feature Selection." *IEEE Transactions on Evolutionary
Computation*, 20(4), 2016.

---

## Appendix A — Package Structure

```
thalamus/
├── component_scoring/      Phase 1 + 2 (offline, LLM evaluation)
├── oracle_builder/         Phase 3 + 4 (offline, GA + classifier training)
│   ├── evolutionary/       Seven-step oracle building pipeline
│   └── classifier/         Logistic regression trainer
├── context_selectors/      Runtime inference
│   ├── ContextSelector     Unified entry point (Path B → Path A → None)
│   ├── ClusterSelector     Phase 3 runtime lookup
│   ├── ClassifierSelector  Phase 4 runtime inference
│   └── BudgetEstimator     Heuristic budget inference
├── shared/                 Shared utilities
│   ├── TurnLogger          Turn logging + off-policy exploration
│   ├── OutcomeScorer       Quality scalar computation
│   ├── QueryClusterer      TF-IDF / sentence-transformer K-means
│   ├── ComponentInclusionClassifier   Logistic regression model
│   └── bookend_order       Context ordering strategy
└── research/               Research add-on (not loaded at query time)
    ├── baselines/          R1 ✓  AllSelector, RandomSelector, TFIDFSelector,
    │                             BM25Selector, DenseSelector
    ├── evaluation/         R1 ✓  BenchmarkRunner, EvalRun, OverlapStats, report
    ├── ablations/          R2 ✓  TopKSelector, NoBookendSelector,
    │                             SingleBudgetSelector, PathBOnlySelector
    ├── cross_path/         R3a ✓ CoInclusionExtractor, FitnessAugmentor
    ├── bandit/             R3b ✓ ExplorationRateEstimator, ConvergenceAnalyzer
    ├── set_quality/        R4    planned
    └── meta_learning/      R5    planned
```

## Appendix B — CLI Reference

```bash
# Production entry point (thalamus-select)
thalamus-select lookup --oracle-dir /oracle --query "..." [--budget auto] [--ordering bookend]
thalamus-select classify --oracle-dir /oracle --embedding-file emb.json

# Research entry point (thalamus-research) — 5 subcommands
thalamus-research baseline-lookup --oracle-dir /oracle --query "..." --method tfidf bm25

thalamus-research eval --oracle-dir /oracle --query-file tasks.jsonl \
    --reference thalamus --method all random tfidf bm25 --out results/r1.json

thalamus-research ablation --oracle-dir /oracle --query-file tasks.jsonl \
    --out results/ablation.json

thalamus-research cross-path --oracle-dir /oracle --top-pairs 20
thalamus-research cross-path --oracle-dir /oracle --augment-configs --lam 0.2 \
    --out context_configs_augmented.json

thalamus-research bandit --oracle-dir /oracle --subcommand estimate-rate \
    --n-min 10 --T-target 500
thalamus-research bandit --oracle-dir /oracle --subcommand convergence \
    --turn-log-dir /logs --window-size 50
```

## Appendix C — Scoring Matrix Schema

```json
{
  "component_name": "skill_name",
  "real_data": {
    "n_turns": 42,
    "updated_mean_score": 0.74
  },
  "baseline_cross_eval": [
    {
      "example_input": "Query text the component was evaluated on",
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

`real_data.updated_mean_score` is set by the Phase 2 enrichment step. If absent, the GA
uses the mean of `baseline_cross_eval[*].scores.f1`. The `BM25Selector` and `TFIDFSelector`
baselines use `baseline_cross_eval[*].example_input` as the corpus for retrieval indexing.
