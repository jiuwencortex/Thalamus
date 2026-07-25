# THALAMUS: A Self-Improving Context Selection System for Production AI Agents

**[Author names redacted for review]**

*Preprint. Under review.*

---

## Abstract

We present THALAMUS, a context selection system for production AI agents that resolves a fundamental tension: the components an agent needs most cannot be determined until the agent has accumulated operational experience, yet that experience is worthless for configuration unless the agent is already selecting components well.

Our central finding is that this tension is self-resolving. A classifier trained on (query, component set, outcome) triples implicitly encodes pairwise component interaction structure in its weight space: weight vectors of components jointly needed by the same query types become geometrically aligned. Extracting this structure — via cosine similarity between L2-normalized classifier weight vectors — and injecting it back into the genetic algorithm that generates the precomputed configurations creates a cross-path feedback loop in which Path B (the online classifier) continuously improves Path A (the offline combinatorial oracle). The result is a system whose configuration quality increases not merely because more data arrives, but because the two paths actively inform each other.

Beyond this feedback mechanism, we make three additional technical contributions: (i) an analytical derivation of the minimum off-policy exploration rate guaranteeing that every component receives sufficient counterfactual training coverage — turning exploration from a tuned hyperparameter into a principled bound; (ii) a gradient-boosted set-level fitness model that replaces the linear per-component scoring formula in the genetic algorithm's inner loop, capturing joint necessity and mutual redundancy that additive models structurally cannot represent; and (iii) a content-addressed cross-deployment transfer protocol that uses SHA-256 fingerprinting to transfer per-component quality priors between deployments sharing identical components.

These contributions rest on a system-level foundation: THALAMUS treats multi-component agentic context assembly as a combinatorial optimization problem distinct from document retrieval, precomputes optimal component configurations offline via genetic search over bitmasks of components, and serves selections in under ten milliseconds at query time via nearest-cluster lookup. We define nine testable claims, implement a five-baseline comparison suite, a four-condition ablation study, and a 120-task evaluation harness. Experimental results are in preparation.

---

## 1. Introduction

### 1.1 The Problem

A production AI agent maintains a library **C** of components: skill instruction documents, memory sections encoding project state, and tool definitions. At each turn, some subset of **C** is assembled into the context window and presented to the language model alongside the user query.

The universal practice is to include all components unconditionally. At small library sizes this is harmless. At scale, it produces three compounding failures.

**Quality degradation.** Transformer attention is finite. A context window containing forty components, of which four are relevant, distributes effective attention across thirty-six irrelevant documents. The model is not optimized to identify which four matter; performance degrades as library size grows.

**Cost scaling.** Context token cost scales linearly with library size under unconditional inclusion. A simple query paying for two components is charged for forty. Token spend grows with library size regardless of task complexity.

**Lost-in-the-middle decay.** Empirical work [Liu et al., 2024] establishes that language models attend preferentially to content at context edges, with material in the interior receiving substantially weaker effective attention weight. A forty-component context places most relevant material in the interior, below the attention threshold at which it meaningfully influences generation.

We call this compound failure **Context Saturation** and define it precisely in §3.

### 1.2 Why This Is Hard

The obvious solution — include only the components relevant to the current query — requires knowing which components are relevant before running the agent. The relevant components depend on query type. Query type is not available at system initialization. The interaction effects between components (pairs that are jointly necessary, mutually redundant, or contradictory) cannot be determined from individual component content alone; they require observing outcomes of actual agent turns.

This creates a temporal dependency problem: to select well, you need interaction evidence; to accumulate interaction evidence, you need to select well enough to observe meaningful outcomes. Neither can bootstrap the other from a cold start.

### 1.3 Our Approach and Key Finding

THALAMUS resolves the temporal dependency by splitting context selection into two interleaved mechanisms operating at different timescales.

**Path A (offline, precomputed):** Before deployment, a genetic algorithm searches over all 2ⁿ subsets of **C**, using LLM-generated per-component quality scores as a fitness signal. The search finds, for each cluster of similar query types and each token budget tier, the approximately optimal component configuration. These configurations are stored as a lookup table and serve selections at inference time via a nearest-cluster assignment completing in under ten milliseconds, with no LLM calls on the critical path.

**Path B (online, learned):** As the agent accumulates turn logs, a logistic regression classifier is trained on (query embedding, component set, outcome quality) triples, learning per-component inclusion probabilities from operational data. Once the classifier reaches a confidence threshold, it takes over as the primary selector.

This two-path structure is not itself novel. What is novel is the discovery that **the two paths actively improve each other through the geometry of the classifier's weight space.**

A logistic regression classifier trained to predict component inclusion learns a weight vector **w**ᵢ ∈ ℝᵈ for each component cᵢ. If two components cᵢ and cⱼ are often selected together on high-quality turns, they respond to the same query types, and their weight vectors become geometrically aligned: **w**ᵢ · **w**ⱼ / (‖**w**ᵢ‖ ‖**w**ⱼ‖) → 1. If they are substitutes, they diverge: the inner product → −1.

This pairwise alignment structure encodes exactly what the genetic algorithm's additive fitness function cannot represent: joint component utility. By extracting the classifier's weight-space co-inclusion matrix and injecting it as an additive term in the GA's fitness function, we create a directed information channel from the outcome-trained discriminative model back into the combinatorial optimizer. The GA's next oracle build benefits from all interaction evidence accumulated by Path B, without retraining the GA or requiring any architectural changes to either path.

The result is a self-improving loop:

> Path A selects → turns are logged → Path B trains → co-inclusion signal extracted → Path A fitness augmented → better configurations generated → Path B trains on better data → ...

Section 4 develops this feedback mechanism in detail.

### 1.4 Contributions

- **C1:** Genetic combinatorial search finds component configurations that outperform top-k retrieval by individual relevance score, particularly on tasks requiring multiple interacting components (§5.2, §8.1).
- **C2:** LLM-generated synthetic component scores provide a warm start that reduces the turn count required for Path B to exceed Path A quality (§5.1, §8.1).
- **C3:** The dual-path architecture dominates either path alone across the full maturity curve from cold start to the mature checkpoint (§5.4, §8.1).
- **C4:** Without off-policy exploration, the logistic regression classifier converges to an imitation of Path A's policy. An analytically derived minimum exploration rate ε* guarantees sufficient counterfactual coverage for Path B to learn a strictly superior policy (§6, §8.2).
- **C5:** Placing the most-relevant components at the edges of the assembled context (bookend ordering) yields measurable quality improvement on tasks with assembled context exceeding three thousand tokens (§5.3, §8.2).
- **C6:** Heuristic query-complexity budget estimation outperforms any fixed-budget policy on a mixed-complexity suite (§5.4, §8.2).
- **C7:** The cross-path co-inclusion feedback mechanism (§4) improves Path A configuration quality by surfacing jointly-useful component pairs that the additive linear fitness formula cannot represent (§8.3).
- **C8:** A gradient-boosted regressor trained on logged (component set, outcome) pairs replaces the additive fitness formula, capturing non-linear interaction effects; configurations found with the learned fitness outperform those found with the marginal formula (§7, §8.3).
- **C9:** Content-addressed component fingerprints enable quality prior transfer between deployments sharing identical components, reducing cold-start sample complexity (§7.3, §8.3).

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

RAG [Lewis et al., 2020] retrieves documents from a corpus at inference time using dense or sparse similarity and prepends them to the model input. It has three structural properties that distinguish it from the problem addressed here: it ranks components independently by individual relevance, ignoring joint utility; it does not enforce token budget constraints through combinatorial reasoning; and it performs retrieval at query time. THALAMUS addresses all three by precomputing optimal joint configurations offline and serving selections via a sub-millisecond lookup. The five baselines in §8 include TF-IDF, BM25, and dense retrieval selectors that implement the RAG paradigm directly, enabling direct comparison.

### 2.2 LLM Agent Frameworks

Agent frameworks [Park et al., 2023; Yao et al., 2023; Wang et al., 2024] have focused on reasoning architecture, tool use protocols, and memory management. With few exceptions, they adopt unconditional context assembly. THALAMUS operates at the context assembly layer upstream of any downstream reasoning architecture and is compatible with any agent framework that accepts an externally assembled context.

### 2.3 Prompt Compression and Selection

Prompt compression methods [Li et al., 2023; Jiang et al., 2023] and optimization-based selection [Khattab et al., 2023] address how to represent individual documents or optimize individual prompt phrasings. They do not address the combinatorial question of which documents to include from a dynamic library, nor do they model inter-component interaction effects. THALAMUS is orthogonal: it determines inclusion sets, not content representations.

### 2.4 Contextual Bandits and Off-Policy Learning

The exploration mechanism in Path B (§6) is a practical instantiation of contextual bandit theory [Langford & Zhang, 2007; Li et al., 2010]. The minimum exploration rate derivation is a direct analytical result from the coverage requirement for multi-label logistic regression. The contextual bandit literature provides the theoretical grounding but does not address the specific multi-label agentic context selection problem or the feedback mechanism between the bandit learner and the precomputed oracle.

### 2.5 Surrogate Model Optimization

The set-level fitness model in §7 replaces the hand-crafted fitness function in the GA's inner loop with a learned predictor — a form of surrogate model optimization [Jin, 2011]. Surrogate-assisted evolutionary algorithms are established in engineering design optimization. Their application to the combinatorial agentic context assembly problem, with a feature representation derived from both component catalog statistics and classifier weight-space co-inclusion signals, is novel.

### 2.6 Genetic Algorithms for Combinatorial Selection

Genetic algorithms have been applied to feature selection [Xue et al., 2016] and software product line configuration [Harman et al., 2012]. The agent context window optimization problem differs in that the fitness function encodes cluster-specific relevance and a hard token budget, and the search space is determined by a dynamic, user-defined component library. Standard GA operators apply without modification; the contribution is the fitness design and the cross-path feedback that improves it.

### 2.7 Long-Context Positional Bias

The bookend ordering strategy (§5.3) is grounded in empirical findings that language models attend preferentially to content at context edges [Liu et al., 2024]. THALAMUS formalizes this finding into a deployable component ordering policy and tests it as an isolated ablation condition.

---

## 3. Problem Formulation

Let **C** = {c₁, …, cₙ} be the agent's component library. Each component cᵢ has token cost τ(cᵢ) ∈ ℤ⁺. Let **q** ∈ 𝒬 be an incoming user query and **B** ∈ ℝ⁺ a token budget.

**Definition (Context Selection Problem).** Find S ⊆ **C** with ∑_{c ∈ S} τ(c) ≤ B maximizing expected outcome quality E[Q(S, q)], where Q: 2^C × 𝒬 → [0, 1].

**Why it is hard.** The candidate space has cardinality 2ⁿ. Direct evaluation of any S requires a live agent execution. Components may be jointly necessary (neither c alone is useful; together they are essential), mutually redundant (either suffices; both together add noise), or mutually interfering (each produces good answers individually; together they generate contradictory instructions). No closed-form solution exists for arbitrary Q and arbitrary **C**.

**Definition (Context Saturation).** An agent deployment exhibits Context Saturation when three conditions hold simultaneously: (a) outcome quality decreases as |**C**| grows under unconditional context assembly; (b) token cost scales with |**C**| rather than with task complexity; (c) relevant components receive sub-threshold attentional weight due to positional interior placement. All three conditions are functions of library size and are absent at small |**C**|.

THALAMUS constructs an approximation to the optimal selection policy through offline combinatorial precomputation over the component bitmask space (Path A), refined by an online discriminative classifier trained on accumulated outcome data (Path B), with a directed information channel from Path B to Path A through the classifier's weight-space geometry (§4).

---

## 4. The Cross-Path Feedback Mechanism

This section develops the central technical contribution: the directed information channel from the online classifier (Path B) to the offline combinatorial oracle (Path A).

### 4.1 What the Classifier Knows That the GA Does Not

The genetic algorithm used in Path A evaluates candidate component sets using a fitness function that sums individual component quality scores weighted by query-cluster relevance. This formulation is a sum over components: it structurally cannot represent interaction between them.

Consider a concrete failure mode. Components cᵢ ("database schema conventions") and cⱼ ("query optimization guidelines") each have mediocre individual quality scores on general queries, because neither is independently sufficient. On database performance queries, however, they are jointly necessary: an agent context missing either one consistently fails the task. The GA's additive fitness function assigns both low individual weights and will tend to exclude them from Pareto-optimal configurations.

Now consider the classifier trained on logged turns. On every database performance query where both cᵢ and cⱼ were included, the turn's high outcome quality generates strong positive gradient signal for both **w**ᵢ and **w**ⱼ toward the same query embedding region. Both weight vectors are pushed to respond strongly to the same query type subspace. Their cosine similarity in weight space increases as a direct consequence of their joint utility on a shared query distribution.

The classifier does not explicitly represent "cᵢ and cⱼ are jointly necessary." But it implicitly does, in a form that is directly accessible.

### 4.2 Co-Inclusion Signal Extraction

Given the classifier's weight matrix **W** ∈ ℝ^{n × d}, where row **w**ᵢ is the weight vector for component cᵢ:

**Co-inclusion matrix.** Compute the pairwise cosine similarity between L2-normalized weight vectors:

```
W_norm[i] = w_i / ‖w_i‖₂

co_inclusion(cᵢ, cⱼ) = W_norm[i] · W_norm[j]  ∈ [−1, 1]
```

A value near +1 indicates that cᵢ and cⱼ respond to the same query types — they are jointly useful. A value near −1 indicates they respond to opposite query types — they are substitutes and including both wastes budget. A value near 0 indicates orthogonal domains.

**Set-level co-inclusion score.** For a candidate component set S, the mean pairwise co-inclusion:

```
co_inclusion(S) = (2 / |S|(|S|−1)) × ∑_{i < j, cᵢ, cⱼ ∈ S} co_inclusion(cᵢ, cⱼ)
```

This scalar summarizes the degree to which the components in S tend to be needed together — a proxy for joint utility not captured by any additive function of individual scores.

### 4.3 Fitness Augmentation

The co-inclusion signal is injected into the GA as an additive augmentation of the marginal fitness formula:

```
fitness_aug(S, k, B) = fitness_marginal(S, k, B) + λ × co_inclusion(S)
```

where fitness_marginal is the standard per-component score weighted by query-cluster cosine similarity (§5.2) and λ ≥ 0 controls the weight of the interaction signal relative to individual scores. At λ = 0, this reduces to the original formula with no interaction information. At λ > 0, the GA is nudged toward component sets whose members tend to be jointly needed, according to the classifier's accumulated outcome evidence.

The geometry of this augmentation is important. It is not a supervision signal: the GA is not being told "include cᵢ and cⱼ together." It is a soft regularizer that biases the Pareto search toward configurations whose components operate in aligned query-type subspaces. Whether that alignment translates to improved task outcome quality is an empirical question tested by contribution C7 (§8.3).

### 4.4 The Feedback Loop

The resulting information flow is:

```
[Path A oracle] → configurations → [agent turns] → outcome data
       ↑                                              ↓
  fitness_aug ← co_inclusion ← W ← [Path B classifier trains]
```

Path A generates configurations. Those configurations determine which component sets are observed on production turns. Path B trains on those turns. Path B's weight matrix encodes the interaction structure latent in the outcome data. That structure is extracted and returned to Path A as a fitness augmentation on the next oracle rebuild.

This loop has two properties worth noting. First, it improves with data quality rather than merely data quantity: a classifier trained on low-quality turns with uninformative outcome scores produces weight vectors with weak alignment structure, and the co-inclusion signal is correspondingly weak. Second, it is asymmetric: Path B informs Path A at oracle rebuild time (typically weekly or monthly), while Path A informs Path B continuously by providing the configurations on which turns are logged.

**What breaks the loop.** If Path B's training data consists only of turns where Path A selected the components, the co-inclusion matrix reflects only configurations Path A already knows to select. It cannot surface interaction structure in configurations Path A did not explore. This is why the off-policy exploration mechanism (§6) is not merely useful but required for the feedback loop to escape Path A's existing policy.

---

## 5. System Architecture

This section describes the system that implements and enables the feedback mechanism. The architecture has four offline phases and one online runtime. Its primary purpose is to provide Path A and Path B as concrete, deployable instantiations of the abstract roles in §4.

### 5.1 Phase 1–2: Component Scoring

Each component cᵢ is scored offline by generating M = 20 synthetic (query, expected answer) pairs using a language model prompted with the component's full text, executing the agent with only cᵢ in context on each synthetic query, and evaluating each output against the expected answer with four lexical metrics: unigram F1, bigram F1, bag-of-words recall, and length ratio. The primary downstream signal is the mean F1 score:

```
mean_score(cᵢ) = (1/M) × ∑_{j=1}^{M} F1(outputⱼ, referenceⱼ)
```

These scores are proxy signals: they measure surface lexical agreement rather than semantic correctness, and they reflect individual component utility rather than joint utility. Both limitations are addressed downstream — by Phase 2 enrichment for the individual-utility problem, and by the feedback mechanism of §4 for the interaction problem.

**Phase 2 enrichment.** As turn logs accumulate, empirical mean outcome quality (computed over turns where cᵢ was included) is blended with the synthetic prior via Bayesian weighting:

```
enriched_score(cᵢ) = (n × empirical_mean + α × synthetic_mean) / (n + α)
```

where n is the logged turn count for cᵢ and α = 5 is the prior pseudo-count. At n = 0, this recovers the synthetic score. As n grows, it converges to the empirical mean.

### 5.2 Phase 3: Path A — Combinatorial Oracle

**Query space partitioning.** The query space 𝒬 is partitioned into K clusters (default K = 20) by K-means applied to TF-IDF vectors of component example texts or sentence-transformer embeddings of historical queries. The cluster centroid model is serialized for query-time assignment. Automatic cluster count selection (--auto-k) fits K-means for K ∈ {5, 7, 10, 15, 20, 25, 30} and selects K minimizing a weighted combination of the within-cluster sum-of-squares elbow and the silhouette coefficient.

**Marginal fitness function.** For candidate set S, cluster k with centroid **z**ₖ, and budget tier B with maximum token count B_max:

```
fitness(S, k, B) =
  [ ∑_{c ∈ S} mean_score(c) × max(cosine(q_c, z_k), 0) ]  −  λ × (token_cost(S) / B_max)

  where q_c is the component's example-query centroid vector
        λ is the size penalty weight (default 0.1)
```

The cosine term weights each component's mean score by the geometric alignment of its domain with the cluster centroid. The λ term imposes a soft budget incentive, with the Pareto front over (fitness, token_cost) maintained to produce configurations spanning the full budget range. This formulation is additive over components by construction; §4 and §7 address the resulting interaction blindness.

**Genetic algorithm.** Standard binary genetic algorithm over bitmasks of length n. Population size 100, 200 generations, uniform crossover, bit-flip mutation at rate 0.05, tournament selection (k=3), elitism (top 100 from combined parent-offspring pool). Runs once per (cluster, budget) pair. The Pareto front over (fitness, token_cost) is maintained; the selected configuration for each budget tier is the highest-fitness individual satisfying the hard token constraint.

**Per-cluster λ tuning.** For each cluster, λ is tuned independently from logged turn data via grid search over {0.01, 0.05, 0.1, 0.2, 0.5}, selecting the value maximizing mean outcome quality on turns assigned to that cluster. Per-cluster values are stored in `per_cluster_lambda.json` and applied on subsequent oracle builds.

**Optional LLM Pareto validation.** A post-processing step evaluates top Pareto candidates using live LLM calls on a held-out query set, re-ranking by observed outcome quality. This corrects for combination-level synergies that the proxy fitness cannot capture, at the cost of one LLM call per candidate per validation query.

### 5.3 Phase 4: Path B — Classifier Layer

**Architecture.** For each component cᵢ, a logistic regression classifier estimates:

```
P(include cᵢ | q) = σ(wᵢᵀ · φ(q) + bᵢ)
```

where φ(q) is the query embedding and σ is the logistic sigmoid. The N classifiers are trained jointly via one-versus-rest decomposition. Per-component inclusion thresholds are optimized separately by maximizing F1 on the training set. Regularization strength C ∈ {0.01, 0.1, 1.0, 10.0} is tuned by grid search with leave-one-out cross-validation.

**Training data.** Turn log files supply (query embedding, component set, outcome quality, exploration flag) records. The classifier reads the most recent 8 weeks of logs and requires at least 10 turns to proceed. Off-policy exploration turns (§6) are the primary source of counterfactual coverage.

**Off-policy exploration.** The turn logger overrides the selector's component choice on a random ε-fraction of production turns (default ε = 0.1), sampling an alternative component set. These turns are flagged in the log and provide the data needed for the feedback loop of §4 to escape Path A's existing policy. The minimum ε required for guaranteed coverage is derived analytically in §6.

### 5.4 Runtime: Dual-Path Selection

At query time, `ContextSelector` is the unified entry point implementing a two-path, gracefully-degrading selection protocol:

```
1. If Path B is ready AND turn_count ≥ min_turns:
     result = ClassifierSelector.select(q)
     if result.confidence ≥ min_confidence: return result  [Path B active]

2. If context_configs.json exists:
     return ClusterSelector.select(q, budget, ordering)  [Path A active]

3. return None  [cold start — calling system falls through to existing logic]
```

**Context ordering.** The selected component list is ordered before context assembly. Two strategies are provided. `relevance`: components sorted by descending cosine similarity between their example-query centroid and the current query embedding. `bookend`: the relevance-sorted list rearranged to place the most-relevant components at the positional extremes of the assembled context. For relevance-sorted list [c₁, c₂, …, cₖ], the bookend permutation interleaves:

```
[c₁, c₃, c₅, …, c₆, c₄, c₂]
```

placing the highest-relevance component at position 1 and the second-highest at position k, directly counteracting the lost-in-the-middle attention decay [Liu et al., 2024].

**Automatic budget estimation.** `BudgetEstimator` infers query complexity from word count, architectural scope markers ("design", "migrate", "refactor across"), multi-file markers, and task type signals ("explain", "implement", "review"), mapping to three token budget tiers. This enables budget-adaptive selection without requiring the caller to specify complexity per request.

**Maturity transition.** As logged turns accumulate, the active path transitions automatically from Path A (cold start) to Path B (mature), detectable via the `active_path` property. This transition is not abrupt: Path B is only activated when it reaches the confidence threshold, and the feedback mechanism of §4 ensures that Path A's configurations improve in parallel with Path B's training data quality.

---

## 6. Principled Exploration Rate Derivation

### 6.1 The Coverage Problem

The classifier training data has a fundamental gap: it observes only the components included on each logged turn. Without counterfactual data, the classifier cannot estimate the quality of component sets it was never asked to evaluate. In the limiting case (ε = 0), the classifier converges to an imitation of Path A's policy — it reproduces the cluster-based lookup rather than learning from outcome feedback. The co-inclusion matrix in this case reflects only interaction structure among configurations Path A already selects, and the feedback loop of §4 cannot escape Path A's existing policy.

This is not merely a practical concern. It is a structural impossibility: a classifier trained exclusively on Path A selections cannot represent any interaction structure outside Path A's support, regardless of how many turns are logged.

### 6.2 Formal Setup

Model component selection as a multi-label contextual bandit:
- **State** s ∈ 𝒮: query embedding
- **Action** a ∈ {0, 1}ⁿ: component inclusion bitmask
- **Reward** r ∈ [0, 1]: outcome quality scalar

Under Path A alone, component cᵢ is included with probability p_A(cᵢ) = fraction of query clusters in which cᵢ appears in the optimal configuration. The logistic regression classifier requires at least n_min labeled examples of both inclusion states (cᵢ = True, cᵢ = False) to reliably estimate **w**ᵢ. Under exploration rate ε, the exploration policy samples each component independently with probability 0.5 (uniformly random), so the effective inclusion probability for training is:

```
p_eff(cᵢ) = (1 − ε) × p_A(cᵢ) + ε × 0.5
```

The expected count of cᵢ = True turns in T total turns is p_eff(cᵢ) × T. The coverage constraint is:

```
p_eff(cᵢ) × T ≥ n_min  for all i
```

### 6.3 The Minimum Exploration Rate

Solving for the minimum ε satisfying the coverage constraint for the hardest-to-cover component (the one with p_A(cᵢ) furthest from 0.5):

```
ε*(n_min, T_target) = max_i { max( 0, (n_min/T_target − p_A(cᵢ)) / (0.5 − p_A(cᵢ)) ) }
```

This is the minimum exploration rate guaranteeing that every component receives at least n_min inclusion turns within T_target total turns, enabling reliable logistic regression estimation for all N classifiers simultaneously. With default parameters n_min = 10 and T_target = 500, ε* is computed directly from the oracle's per-cluster configurations (which determine p_A).

**Implications.** Components that Path A includes in almost every cluster (high p_A) are easy to cover: even small ε provides sufficient True examples. Components Path A rarely includes (low p_A) are harder to cover: a higher ε is required to drive sufficient inclusion examples into the training set. The max over i selects the binding constraint — the component most underserved by Path A's policy.

**Research contribution C4.** This derivation turns exploration rate from an empirically tuned hyperparameter into a computable bound. Contribution C4 tests the hypothesis that ε ≥ ε* is necessary and sufficient for Path B to learn a policy not dominated by Path A, and that ε < ε* leads to convergence (high Jaccard agreement between Path B and Path A selections) due to insufficient counterfactual coverage.

### 6.4 Convergence Measurement

Empirical Path B convergence is measured by tracking the Jaccard agreement between Path A's selections and Path B's selections over rolling windows of logged turns:

```
agreement(t) = |S_A(qₜ) ∩ S_B(qₜ)| / |S_A(qₜ) ∪ S_B(qₜ)|
```

A classifier showing `final_agreement ≥ 0.85` over the most recent window has converged to Path A's policy — a diagnostic that the exploration rate is insufficient for the feedback loop to escape Path A's support.

---

## 7. Extensions: Learned Fitness and Cross-Deployment Transfer

### 7.1 Set-Level Fitness Model

**Motivation.** The marginal fitness function of §5.2 cannot represent joint necessity, mutual redundancy, or mutual interference — three interaction types that are empirically observable in logged turn data and that the co-inclusion augmentation of §4 addresses only partially (co-inclusion captures alignment in weight space, not the magnitude of the interaction effect on outcome quality).

A direct approach: train a regression model on logged (component set, outcome quality) pairs to predict set-level outcome quality directly, and use this predictor as the fitness function in the GA's inner loop.

**Feature representation.** Each (component set S, cluster k) pair is represented by a 14-dimensional vector:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | mean\_score | Mean catalog score across S |
| 1 | std\_score | Standard deviation of catalog scores |
| 2 | min\_score | Minimum score in S |
| 3 | max\_score | Maximum score in S |
| 4 | n\_components | \|S\| total |
| 5 | n\_skills | Count of skill-type components |
| 6 | n\_tools | Count of tool-type components |
| 7 | n\_memory | Count of memory-section components |
| 8 | n\_other | Count of other types |
| 9 | mean\_co\_inclusion | Mean pairwise co-inclusion score over S (§4.2) |
| 10 | min\_co\_inclusion | Minimum pairwise co-inclusion over S |
| 11 | max\_co\_inclusion | Maximum pairwise co-inclusion over S |
| 12 | cluster\_id | Raw cluster integer |
| 13 | cluster\_id\_norm | cluster\_id / 100 |

Dimensions 9–11 incorporate the co-inclusion signal from §4 directly into the feature vector, creating a second path through which classifier interaction knowledge enters the GA's fitness evaluation. This is complementary to the additive augmentation of §4.3: the additive term shifts the fitness landscape; the feature encoding allows the GBR to learn non-linear functions of the interaction signal.

**Model.** A `GradientBoostingRegressor` (n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8) trained on all logged turns. Off-policy exploration turns are deliberately included: they provide coverage of component sets outside Path A's support and are the primary source of non-additive interaction signal.

**GA integration.** When the set-quality model is available, it replaces the quality term of the fitness function while the budget penalty remains:

```
fitness(S, k, B) = SetQualityModel(features(S, k)) − λ × (token_cost(S) / B_max)
```

Fallback to the marginal formula is maintained if model inference fails (e.g., component sets outside the training distribution).

**Research contribution C8.** Section 8.3 compares oracle quality between GA runs with the marginal formula and the learned set-level fitness at the mature data checkpoint, measuring the magnitude of the improvement attributable to learned interaction effects.

### 7.2 Minimum Data Requirement

The set-quality model requires sufficient logged turns with filled outcome quality labels to learn meaningful interaction effects. Below a threshold number of turns, the model reduces to near-random prediction over the interaction features. Based on gradient boosting sample complexity, a minimum of 200–500 labeled turns is likely required before the learned fitness outperforms the marginal formula — a threshold tested empirically in §8.3.

### 7.3 Cross-Deployment Transfer

**Motivation.** Every new deployment begins with no logged turns, no trained classifier, and a GA fitness function relying entirely on synthetic LLM-scored priors. If the new deployment shares components with previously operated deployments — identical content, different configuration — accumulated quality evidence from prior deployments is directly applicable to the warm-start problem.

**Content fingerprinting.** Each component is identified by a content-addressed SHA-256 fingerprint:

```
fingerprint(cᵢ) = SHA-256(name(cᵢ) ∥ "\x00" ∥ description(cᵢ) ∥ "\x00" ∥ body(cᵢ))
```

Components with identical fingerprints have identical content, regardless of deployment. This enables direct quality evidence transfer without deployment-specific metadata and without semantic matching (which would risk false positives from functionally similar but textually different components).

**Transfer protocol.** For a new deployment:

1. Compute fingerprints for all components in the new oracle directory.
2. Look up each fingerprint in a shared knowledge base recording `{fingerprint: mean_outcome_when_included}` aggregated across all prior deployments.
3. For matched components, extract `mean_outcome_when_included` as the prior quality estimate.
4. Write `{component_name: prior_score}` to `transfer_priors.json` in the oracle directory.

**GA integration.** At oracle build time, `transfer_priors.json` is detected automatically and blended with synthetic scores:

```
blended_score(cᵢ) = α × kb_prior(cᵢ) + (1 − α) × synthetic_score(cᵢ)
```

The blend weight α decays as local turns accumulate:

```
α = exp(−n_turns / τ),   τ = 200
```

At cold start (n_turns = 0), α = 1.0 and the KB prior fully replaces the synthetic score for matched components. As the deployment matures, α → 0 and local evidence dominates. Unmatched components retain synthetic scores throughout.

**Precision vs recall tradeoff.** SHA-256 fingerprinting provides perfect precision (no false matches) at the cost of recall: a semantically equivalent component that has been rephrased receives no transfer benefit. Semantic fingerprinting (embedding-based similarity) would increase recall at the risk of false matches; the tradeoff is left to future work (§10).

**Research contribution C9.** Section 8.3 measures the cold-start quality difference between deployments with and without transfer priors, quantifying the turn-count reduction required to reach steady-state configuration quality.

---

## 8. Experimental Evaluation

### 8.1 Evaluation Protocol

All selectors implement a common runtime-checkable `SelectorProtocol` interface, accepting a query string and budget specification and returning a structured component list or `None`. The unified interface enables evaluation runs that swap any selector into any harness position without modification.

`BenchmarkRunner` accepts a dictionary of selector instances. For each query, it runs every selector n_repeats=3 times and records median latency, per-query component overlap statistics against a designated reference selector (Jaccard similarity, precision, recall), and mean component counts per budget tier. Results are serialized with a unique run ID and ISO timestamp for reproducibility.

Quality measurement is performed in a separate pass that executes the selected configurations against the real agent on the fixed 120-task suite and records scalar `outcome_quality` per (task, selector). This two-phase design separates deterministic benchmarking (latency, overlap) from expensive quality measurement.

### 8.2 Baseline Selectors

Five baselines are implemented, all reading from the same component pool.

**AllSelector.** Returns the complete component pool at maximum token cost. Serves as the theoretical quality upper bound; THALAMUS configurations approaching AllSelector quality have solved the selection problem.

**RandomSelector.** Samples k components uniformly at random (seeded), where k equals the average component count returned by the THALAMUS oracle at that budget tier. Serves as the chance baseline.

**TFIDFSelector.** TF-IDF (max_features=2000, bigrams, sublinear TF) fitted on all component example texts; returns top-k by cosine similarity. Represents standard lexical retrieval.

**BM25Selector.** Okapi BM25 (k₁=1.5, b=0.75):

```
score(D, Q) = ∑_{t ∈ Q} IDF(t) × tf(t,D) × (k₁+1) / (tf(t,D) + k₁(1−b+b|D|/avgdl))
IDF(t) = log((N − df(t) + 0.5) / (df(t) + 0.5) + 1)
```

The strongest lexical retrieval baseline and standard in production RAG.

**DenseSelector.** Sentence transformer (all-MiniLM-L6-v2), top-k by L2-normalized cosine similarity. Represents current RAG practice with semantic embeddings.

All baselines return the same number of components k as the THALAMUS oracle at each budget tier, ensuring token-cost-neutral comparisons.

### 8.3 THALAMUS Configurations Under Evaluation

| Configuration | Description |
|---|---|
| thalamus-path-a | Path A only, relevance ordering |
| thalamus-path-a-bookend | Path A only, bookend ordering |
| thalamus-path-b | Path B only (no Path A fallback) |
| thalamus-full | Dual-path ContextSelector |
| thalamus-co-inclusion | Path A + cross-path co-inclusion augmentation (§4) |
| thalamus-xgb | Path A + learned set-level fitness (§7.1) |
| thalamus-transfer | Path A + cross-deployment transfer priors (§7.3) |

### 8.4 Task Suite

120 deterministic tasks across four categories:

| Category | Count | Characteristics |
|---|---|---|
| Simple single-skill | 30 | One primary component required; interaction effects absent |
| Multi-skill | 40 | Two or more components required; interaction effects expected |
| Architecture | 30 | Broad domain coverage; long assembled context; positional effects expected |
| Memory-dependent | 20 | Requires project-specific context; tests memory component scoring |

Tasks are pre-defined with fixed expected outputs and LLM judge prompts. The suite is not resampled between runs; all comparisons use the identical 120-task fixed set.

### 8.5 Evaluation Metrics

| Metric | Computation | Unit |
|---|---|---|
| Task success rate | Binary LLM judge pass/fail, mean over suite | % |
| LLM judge score | Continuous quality score (0–1), mean over suite | scalar |
| Tokens consumed | Context tokens assembled, mean over suite | tokens/turn |
| Quality-per-token | judge\_score / tokens\_consumed | scalar |
| Selection latency | Wall-clock ms from query to component list | ms |
| Sample complexity | Turns for Path B to exceed Path A quality by >2% | turns |
| Jaccard overlap | \|S ∩ S_ref\| / \|S ∪ S_ref\| vs reference | [0,1] |

Latency, Jaccard, precision, and recall are available immediately after benchmark runs without LLM calls. Task success rate and judge score require the quality measurement pass.

### 8.6 Maturity Checkpoints

Each configuration is evaluated at three points:

- **Cold start (0 turns):** No logged data. Path B unavailable. Tests the quality of the offline-prepared oracle against retrieval baselines with no operational evidence.
- **Early (100 turns):** Modest logged data. Path B may activate. Tests learning speed and path transition.
- **Mature (500+ turns):** Stable classifier and empirical enrichment. Tests asymptotic quality and the full feedback mechanism.

### 8.7 Results — Baseline and Dual-Path Comparison (C1, C2, C3)

*[Results pending completion of quality measurement runs.]*

| Selector | Success Rate (Cold) | Success Rate (100T) | Success Rate (Mature) | Tokens/Turn |
|---|---|---|---|---|
| AllSelector | [X] | [X] | [X] | [X] |
| RandomSelector | [X] | [X] | [X] | [X] |
| TFIDFSelector | [X] | [X] | [X] | [X] |
| BM25Selector | [X] | [X] | [X] | [X] |
| DenseSelector | [X] | [X] | [X] | [X] |
| thalamus-path-a | [X] | [X] | [X] | [X] |
| thalamus-path-b | — | [X] | [X] | [X] |
| thalamus-full | [X] | [X] | [X] | [X] |

*C1: thalamus-path-a vs retrieval baselines at cold start, conditioned on task category (multi-skill expected to show largest gap).*
*C2: thalamus-full vs thalamus-path-b — sample complexity (turns to threshold quality improvement) with vs without synthetic prior warm start.*
*C3: thalamus-full vs thalamus-path-a vs thalamus-path-b across the three maturity checkpoints — the dual-path curve.*

### 8.8 Results — Ablation Study (C4, C5, C6)

*[Results pending.]*

Four ablation selectors each remove exactly one architectural feature.

| Configuration | Multi-skill | Architecture | Memory-dep. | Quality/Token |
|---|---|---|---|---|
| thalamus-path-a (full) | [X] | [X] | [X] | [X] |
| TopKSelector (−GA combinatorial search) | [X] | [X] | [X] | [X] |
| NoBookendSelector (−bookend ordering) | [X] | [X] | [X] | [X] |
| SingleBudget-small (−auto budget) | [X] | [X] | [X] | [X] |
| SingleBudget-large (−auto budget) | [X] | [X] | [X] | [X] |
| PathBOnlySelector (−Path A fallback) | [X] | [X] | [X] | [X] |

*C4: Jaccard agreement curve (Path B vs Path A over turn count), with ε ≥ ε* vs ε < ε*. High agreement at ε < ε* is the diagnostic signature of Path B converging to Path A's policy.*
*C5: NoBookendSelector vs thalamus-path-a-bookend, conditioned on assembled context length. Effect expected to be absent below 3k tokens.*
*C6: Auto-budget vs each fixed-budget tier, quality-per-token on the mixed-complexity suite.*

### 8.9 Results — Feedback Mechanism and Extensions (C7, C8, C9)

*[Results pending, including multi-deployment data for C9.]*

**C7 — Co-inclusion feedback (§4):**

| Condition | Multi-skill | Architecture | Cold-start Disadvantage |
|---|---|---|---|
| Path A (marginal fitness) | [X] | [X] | — |
| Path A + co-inclusion (λ=0.2) | [X] | [X] | — |
| Path A + co-inclusion (λ=0.0) | [X] | [X] | — |
| Path A + co-inclusion (λ=0.5) | [X] | [X] | — |

*C7 is only meaningful at non-zero classifier maturity: at cold start, the co-inclusion matrix is uninitialized and λ has no effect.*

**C8 — Set-level fitness (§7.1):**

| Fitness Function | Success Rate (Mature) | Quality/Token | In-sample R² |
|---|---|---|---|
| Marginal (additive sum) | [X] | [X] | — |
| SetQualityModel (GBR) | [X] | [X] | [X] |

**C9 — Cross-deployment transfer (§7.3):**

| Condition | Cold-start Quality | Turns to Steady State | Fingerprint Match Rate |
|---|---|---|---|
| No transfer (synthetic prior) | [X] | [X] | 0% |
| With transfer priors | [X] | [X] | [X] |

---

## 9. Discussion

*[To be completed after experimental results are available.]*

Several questions will determine how broadly the results generalize:

**At what library size does the interaction signal become significant?** The co-inclusion feedback mechanism (§4) is vacuous at small library sizes where interactions are rare. A threshold component count — below which the additive fitness formula is adequate and above which the feedback mechanism provides a measurable improvement — would allow practitioners to decide whether to deploy the full system.

**Does the feedback loop converge or oscillate?** Each oracle rebuild replaces the previous configuration set. If Path B's training data is sensitive to which configuration was used to generate it, there is a risk of oscillation between oracle versions. Monitoring Jaccard stability between successive oracle builds would characterize this.

**How does the set-quality model behave near its minimum data threshold?** The GBR requires 200–500 labeled turns before it is expected to outperform the marginal formula. Between turn count 0 and this threshold, it may actively harm configuration quality by fitting noise in sparse interaction features. A switching criterion — use learned fitness only when R² on a validation split exceeds a minimum threshold — would prevent premature deployment of the learned fitness.

**What is the empirical fingerprint match rate?** The transfer protocol (§7.3) provides cold-start benefit only for fingerprint-matched components. The match rate depends on how frequently skill documents, memory sections, and tool definitions are shared verbatim across deployments. A low match rate (< 20%) would indicate that the transfer mechanism has limited practical impact, and semantic fingerprinting should be pursued.

---

## 10. Limitations

**Lexical scoring proxies.** The Phase 1 scoring step uses token overlap metrics (F1, bigram F1, bag-of-words recall) that capture surface agreement rather than semantic correctness. A component producing semantically correct but differently-worded outputs receives an underestimated score. Phase 2 enrichment partially corrects this over time, but the individual component scoring step retains the proxy limitation until sufficient operational turns accumulate.

**Additive fitness structural blindness.** The marginal fitness function cannot represent interaction effects regardless of how accurate the individual scores are. The co-inclusion augmentation (§4) and set-quality model (§7.1) address this, but both require a trained classifier — the feedback loop cannot provide interaction signal at cold start.

**Fixed cluster topology.** K-means with a fixed K imposes a static partition on the query space. Query subtypes emerging after oracle construction receive assignments to the nearest existing cluster. Adaptive clustering — online K-means update, cluster splitting — would address this at increased system complexity.

**N independent binary classifiers.** Logistic regression classifiers trained one-per-component cannot represent joint necessity: if cᵢ and cⱼ are useful only together, neither individual classifier receives a strong signal about this joint condition from any single turn. The co-inclusion matrix provides a post-hoc correction, but a multi-label model with shared representation (multi-task logistic regression, multilayer perceptron over the full inclusion vector) would capture joint effects directly.

**Exploration quality cost.** The exploration mechanism degrades outcome quality on ε-fraction of production turns by design. The ε* derivation provides the minimum coverage-satisfying rate but does not optimize the exploration-exploitation tradeoff over time. Annealing ε as coverage accumulates would reduce the total quality cost.

**Fingerprint precision vs recall.** SHA-256 fingerprinting provides exact match only. Rephrased but functionally identical components receive no transfer benefit. Semantic fingerprinting would increase recall at the risk of false positives.

**Bookend assumption generality.** The bookend ordering strategy assumes monotone attention decay toward the context interior. The empirical evidence [Liu et al., 2024] is established for specific architectures and may not generalize uniformly across all production model families.

---

## 11. Future Work

**Semantic component scoring.** Replacing lexical Phase 1 metrics with a semantic judge (BERTScore, LLM-based G-Eval) would improve the accuracy of initial component scores, particularly for components that produce correct outputs in varied phrasings. This would reduce the number of operational turns required for Phase 2 enrichment to correct significant proxy errors.

**Multi-label joint classifier.** Replacing N independent binary classifiers with a single multi-label model that shares a query representation across all N output dimensions would enable Path B to represent joint necessity directly. This would strengthen the co-inclusion signal extracted for the feedback mechanism.

**Online cluster adaptation.** A cluster-splitting heuristic triggered when a cluster's mean intra-cluster distance exceeds a threshold would allow the oracle topology to adapt as the query distribution evolves. This would prevent configuration staleness in long-running deployments with expanding skill libraries.

**Interaction graph regularization.** The co-inclusion matrix provides a pairwise similarity structure over components. This structure could be used as a regularizer on the logistic regression classifiers (graph-regularized multi-label classification), encouraging jointly-useful components to develop aligned weight vectors — a form of structure-inducing prior that might reduce the number of turns required for the co-inclusion signal to become informative.

**Federated cross-deployment transfer.** The current R5 protocol transfers static aggregate quality statistics via a central knowledge base. A federated variant — in which multiple deployments share gradient updates on a shared component quality model while preserving per-deployment data locality — would enable more nuanced transfer than scalar aggregate statistics while respecting privacy constraints.

**Optimal exploration scheduling.** Annealing the exploration rate from ε* toward zero as component coverage accumulates would reduce the total quality cost of the exploration phase while maintaining the coverage guarantee needed for the feedback loop.

---

## 12. Conclusion

THALAMUS addresses Context Saturation — the compound failure of quality degradation, cost scaling, and positional attention decay that emerges when AI agent deployments adopt unconditional context assembly at scale. Its design is grounded in a single observation: the online discriminative model (Path B) and the offline combinatorial optimizer (Path A) each possess information the other lacks, and there is a natural channel for the exchange — the geometry of the classifier's weight space.

The cross-path co-inclusion feedback mechanism makes the system self-improving in a technically precise sense: Path A generates configurations, those configurations produce logged turns, Path B trains on those turns, Path B's weight matrix encodes the interaction structure latent in the outcome data, that structure augments Path A's fitness function on the next oracle rebuild, and the loop repeats. The analytical minimum exploration rate derivation ensures the loop has sufficient counterfactual support to escape Path A's existing policy. The learned set-level fitness model provides a direct learned approximation to the set-quality function that the additive formula cannot represent.

Nine testable claims are defined, a comprehensive evaluation framework is implemented, and experimental results are in preparation.

*[Final quantitative summary to be inserted upon completion of experimental runs.]*

---

## References

[Friedman, 2001] Jerome H. Friedman. "Greedy Function Approximation: A Gradient Boosting Machine." *The Annals of Statistics*, 29(5), 2001.

[Harman et al., 2012] Mark Harman, S. Afshin Mansouri, Yuanyuan Zhang. "Search-Based Software Engineering: Trends, Techniques and Applications." *ACM Computing Surveys*, 45(1), 2012.

[Jiang et al., 2023] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu. "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models." *EMNLP 2023*.

[Jin, 2011] Yaochu Jin. "Surrogate-Assisted Evolutionary Computation: Recent Advances and Future Challenges." *Swarm and Evolutionary Computation*, 1(2), 2011.

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
    ├── baselines/              Baseline comparison selectors
    │   ├── protocol.py             SelectorProtocol (common interface)
    │   ├── component_catalog.py    ComponentCatalog (component pool loader)
    │   ├── all_selector.py         AllSelector
    │   ├── random_selector.py      RandomSelector
    │   ├── tfidf_selector.py       TFIDFSelector
    │   ├── bm25_selector.py        BM25Selector
    │   └── dense_selector.py       DenseSelector
    │
    ├── ablations/              Ablation selectors (one feature removed each)
    │   ├── topk_selector.py        TopKSelector (no GA)
    │   ├── no_bookend_selector.py  NoBookendSelector (relevance ordering only)
    │   ├── single_budget_selector.py  SingleBudgetSelector (fixed budget tier)
    │   └── path_b_only_selector.py PathBOnlySelector (no Path A fallback)
    │
    ├── cross_path/             §4: Classifier → GA co-inclusion transfer
    │   ├── co_inclusion_extractor.py  CoInclusionExtractor
    │   └── fitness_augmentor.py       FitnessAugmentor
    │
    ├── bandit/                 §6: Exploration rate formalization
    │   ├── exploration_rate.py     ExplorationRateEstimator (ε* derivation)
    │   └── convergence.py          ConvergenceAnalyzer (Jaccard agreement curve)
    │
    ├── set_quality/            §7.1: Learned set-level fitness model
    │   ├── outcome_dataset.py      OutcomeDataset (turn logs → training records)
    │   ├── interaction_features.py 14-dimensional feature vector computation
    │   ├── set_quality_model.py    GBR training and inference
    │   └── fitness_function.py     GA-compatible fitness callable
    │
    ├── meta_learning/          §7.3: Cross-deployment fingerprint transfer
    │   ├── component_fingerprint.py  SHA-256 content fingerprinting
    │   ├── knowledge_base.py         Cross-deployment knowledge base
    │   └── transfer_initializer.py   Writes transfer_priors.json
    │
    ├── cli.py                  Research CLI entry point
    └── cli_args_parser.py      Argument parser
```
