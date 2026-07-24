# THALAMUS — Research Plan

> **Mandate:** Evolve THALAMUS from a production engineering system into a research
> project with publishable contributions, while every research phase simultaneously
> produces artifacts deployed in jiuwenswarm. Research and integration are not
> sequential — each phase delivers both.

---

## 1. Research Position

### Problem

LLM agent systems load a fixed context window of skills, memory sections, and tools
for every query. As the component library grows, loading everything is infeasible:
token costs rise, LLM attention degrades for middle-positioned context, and irrelevant
components introduce noise. The naive alternative — TF-IDF retrieval of individual
components — treats each component independently and misses joint utility: two
components that are mediocre in isolation can be essential together.

**Formal problem statement:**

Given a component library `C = {c_1, ..., c_n}`, a token budget `B`, and a user query
`q`, find a subset `S ⊆ C` with `token_cost(S) ≤ B` that maximizes the expected task
quality `E[outcome(S, q)]`.

This is an NP-hard combinatorial optimization problem. The number of candidate sets is
`2^n`. Individual component scores are poor proxies for joint set utility. Real outcome
labels are sparse and arrive with delay.

### Research Claim

THALAMUS proposes that a two-stage offline+online approach — evolutionary combinatorial
search seeded by LLM-generated synthetic scores (Path A), followed by a supervised
linear classifier trained on real interaction outcomes (Path B) — achieves better
quality-per-token tradeoffs than retrieval-based alternatives, even under cold-start
conditions, because:

1. Synthetic prior data is a sufficient signal for the GA to find good component sets
   before any real data exists.
2. Cluster-based precomputation amortizes the combinatorial search across all future
   queries at negligible runtime cost.
3. A linear classifier trained on real outcomes outperforms GA-precomputed configs
   within ~200–500 turns, at <1 ms latency.
4. The Path A → Path B transition, managed by `ContextSelector`, is safe: Path A
   provides a stable floor that Path B only replaces when it demonstrably outperforms.

### Why This Is Research (Not Just Engineering)

The engineering artifact already exists. The research contribution is:

- A formal problem formulation that connects LLM agent context selection to
  combinatorial feature selection and contextual bandits literature.
- Empirical evidence that the above claims hold across tasks, component libraries,
  and maturity levels.
- Ablation results that isolate which system components drive improvement and
  which are incidental.
- A replicable evaluation protocol that can be applied to other agents and libraries.

---

## 2. Research Contributions

Each is a distinct claim testable with an ablation.

| # | Contribution | Claim | Ablation |
|---|---|---|---|
| C1 | GA combinatorial oracle | Joint set optimization outperforms top-k independent scoring | GA vs greedy vs random vs top-k by score |
| C2 | Synthetic-prior cold start | LLM-generated evaluation data is a useful Bayesian prior that reduces sample complexity for Path B | Blended vs synthetic-only vs real-only vs uniform prior |
| C3 | Dual-path maturity progression | Path A provides a stable floor; Path B replaces it only when sufficient data accumulates; transition is safe | Path B only (fails cold) vs Path A only (no improvement) vs dual-path |
| C4 | Off-policy exploration for multi-label context selection | Without exploration, Path B is biased toward configurations Path A selected; exploration reduces this bias | With vs without exploration; exploration rate sensitivity |
| C5 | Bookend ordering | Placing most-relevant components at context edges improves agent quality vs sorted or random ordering | Bookend vs relevance-sorted vs random vs none |
| C6 | Budget-aware selection | Adapting component set size to query complexity (small/medium/large) improves quality-per-token vs fixed budget | Adaptive budget vs fixed-small vs fixed-large vs fixed-medium |

---

## 3. Baselines

All baselines must be implemented and measured against on the same task suite.

| Baseline | Description | Represents |
|---|---|---|
| `all` | Load all components every turn | Upper bound on quality, lower bound on efficiency |
| `random-k` | Random sample of k components | Null hypothesis |
| `tfidf-top-k` | TF-IDF cosine similarity, independent per component | Current practice in agent-core (`SKILL_MODE_RECOMMENDATION`) |
| `bm25-top-k` | BM25 retrieval, independent per component | Stronger lexical retrieval baseline |
| `dense-rag` | Dense embedding retrieval (sentence-transformer), top-k | Best commonly-used retrieval baseline |
| `path-a-only` | THALAMUS Path A without enrichment or Path B | Ablation: does Path B actually help? |
| `path-b-only` | THALAMUS Path B from turn 1 (no warm-start) | Ablation: does cold-start Path A matter? |
| `thalamus-full` | THALAMUS dual-path with `ContextSelector` | Full system |

---

## 4. Evaluation Protocol

### 4.1 Task Suite (jiuwenswarm tasks)

Evaluation runs inside jiuwenswarm on a fixed set of coding agent tasks:

| Category | n tasks | Examples |
|---|---|---|
| Simple fix | 30 | Single-file bug, typo, obvious error |
| Multi-file feature | 30 | Add a new endpoint with tests |
| Architecture | 20 | Refactor a module, add dependency injection |
| Research + write | 20 | Explain a subsystem, write docs |
| Tool-heavy | 20 | CI pipeline, Docker setup, deploy scripts |

Tasks are pre-defined, deterministic inputs. This is a fixed test set — not sampled at
evaluation time.

### 4.2 Metrics

| Metric | Measures | How |
|---|---|---|
| Task success rate | Quality | LLM judge: binary pass/fail per task |
| LLM judge score | Quality | Continuous 0–1 quality score from judge model |
| Tokens consumed | Efficiency | Measured per turn, averaged over task suite |
| Quality-per-token | Joint | judge_score / tokens_consumed |
| Latency | Runtime cost | ms from query receipt to context config returned |
| Sample complexity | Path B learning speed | Turns needed for Path B to exceed Path A quality by >2% |

### 4.3 Maturity Levels

Each system is evaluated at three maturity checkpoints:

- **Cold start** (0 turns): measures cold-start quality
- **Early data** (100 turns): measures how fast the system improves
- **Mature** (500+ turns): measures asymptotic quality

### 4.4 Replication Package

All experiments produce a replication package:
- Task definitions, expected outputs, judge prompts
- Component scoring matrices (anonymized)
- Turn logs (query embeddings only, no raw text)
- Trained classifier weights + oracle configs
- Result tables + plotting code

---

## 5. Research Phases

Each phase has: a research goal, a method, a jiuwenswarm deployment artifact, and a
publication target.

---

### Phase R1 — Baseline Study and Formal Evaluation Framework

**Research goal:** Establish that the problem is real (independent top-k retrieval is
suboptimal) and build the evaluation infrastructure.

**Method:**
1. Implement all baselines (§3) as interchangeable `ContextSelector`-compatible adapters.
2. Run the task suite at cold start with all baselines + THALAMUS Path A.
3. Measure: task quality, tokens consumed, quality-per-token.
4. Report: does Path A beat retrieval baselines at cold start?

**Expected finding:** `thalamus-path-a` outperforms `tfidf-top-k` and `dense-rag` on
multi-file and architecture tasks (where component interactions matter). On simple
tasks, `tfidf-top-k` is competitive.

**Jiuwenswarm deployment output:**
- `BaselineSelector` adapter: makes it possible to A/B test any baseline vs Thalamus
  in production by swapping the `ContextSelector` with a baseline via config flag.
- `config.yaml` key `thalamus.selector: {thalamus|tfidf|bm25|dense|all|random}`
- Evaluation report confirms which selection strategy to use as jiuwenswarm default.

**Publication target:** System track or findings paper at EMNLP / ACL (LLM agents
efficiency). Minimal novelty required at this stage — the contribution is the problem
formulation + evaluation protocol.

**Implementation status: ✓ COMPLETE**

The following artifacts have been implemented under `thalamus/research/`:

| Artifact | Location | Notes |
|---|---|---|
| `SelectorProtocol` | `baselines/protocol.py` | `@runtime_checkable` Protocol; all baselines + `ContextSelector` satisfy it |
| `ComponentCatalog` | `baselines/component_catalog.py` | Loads all scoring matrices; derives per-budget `k` from `context_configs.json` |
| `AllSelector` | `baselines/all_selector.py` | Returns all components; quality upper bound |
| `RandomSelector` | `baselines/random_selector.py` | Seeded random sample of `k` components |
| `TFIDFSelector` | `baselines/tfidf_selector.py` | `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` + cosine similarity |
| `BM25Selector` | `baselines/bm25_selector.py` | Full Okapi BM25 from scratch (k1=1.5, b=0.75) — no external dependencies |
| `DenseSelector` | `baselines/dense_selector.py` | Sentence-transformer L2-normalized cosine; lazy import (`pip install thalamus[sentence]`) |
| `EvalRun` schema | `evaluation/result_schema.py` | `QueryResult`, `OverlapStats`, `AggregateStats`, `SelectorResult`, `EvalRun`; `quality=null` placeholder |
| `OverlapStats` | `evaluation/overlap_stats.py` | Jaccard, precision, recall per query; means over task suite |
| `BenchmarkRunner` | `evaluation/benchmark_runner.py` | Median latency over `n_repeats`; overlap vs reference selector; writes `EvalRun` JSON |
| ASCII report | `evaluation/report.py` | Comparison table: selector × {mean_ms, p95_ms, mean_n, jaccard, precision, recall, quality} |
| `thalamus-research` CLI | `research/cli.py`, `cli_args_parser.py` | `baseline-lookup` (single/multi selector comparison) + `eval` (full benchmark run) |

**Remaining before R1 results are reportable:**
1. Run `thalamus-research eval` on the 120-task jiuwenswarm suite to generate `EvalRun` JSON files.
2. Execute the jiuwenswarm quality measurement pass to fill in `quality` fields.
3. Produce the comparison table and write the R1 findings section.

---

### Phase R2 — Ablation Study: Isolating What Works

**Research goal:** Identify which components of THALAMUS drive improvement and
quantify each one's independent contribution.

**Method:**
Run the task suite at all three maturity levels with ablation variants:

| Ablation | System |
|---|---|
| No GA (top-k scoring) | Path A without evolutionary search |
| GA single-budget (no budget adaptation, fixed medium) | Path A without budget tiers |
| No enrichment (synthetic only, no Bayesian blending) | Path A Stage 1 only |
| No bookend (relevance order only) | Path A with `ordering=relevance` |
| No exploration (Path B, ε=0) | Path B trained only on exploitation turns |
| Path B only from turn 1 | No Path A warm-start |
| Full THALAMUS | All components |

Measure quality degradation from removing each component.

**Expected findings:**
- GA contributes most for architecture/multi-file tasks (interactions matter).
- Bookend contributes for long-context tasks (>3k tokens).
- Enrichment (Bayesian blending) reduces sample complexity for Path B by ~30%.
- Off-policy exploration prevents Path B from converging to a local optimum in the
  first 200 turns.

**Jiuwenswarm deployment output:**
- Tuned default hyperparameters for jiuwenswarm's Thalamus deployment
  (budget thresholds, GA population/generations, exploration rate, bookend vs relevance).
- These replace the current hardcoded defaults in `oracle_builder` and `context_selectors`.

**Publication target:** Extended version of R1 paper. Or standalone workshop paper
(NeurIPS/ICLR Efficient ML track, LLM Agents workshop).

**Implementation status: ✓ COMPLETE (query-time ablation selectors)**

| Ablation | Class | Location | Note |
|---|---|---|---|
| No GA (top-k scoring) | `TopKSelector` | `ablations/topk_selector.py` | Ranks by `mean_score × tfidf_cosine`; no combinatorial search |
| GA single-budget (fixed medium) | `SingleBudgetSelector` | `ablations/single_budget_selector.py` | Ignores budget arg; always uses `budget="medium"` |
| No bookend (relevance order) | `NoBookendSelector` | `ablations/no_bookend_selector.py` | Forces `ordering="relevance"` |
| Path B only from turn 1 | `PathBOnlySelector` | `ablations/path_b_only_selector.py` | No Path A fallback; returns None at cold start |

Build-time ablation variants (require a separate oracle directory):
- **No enrichment**: re-run `thalamus-score --type all` without the `enrich` step, store in separate oracle dir, evaluate with `ContextSelector.load(no_enrich_dir)`.
- **No exploration (ε=0)**: filter turn logs to `exploration.explored=false` before training classifier.  See `research/bandit/` for the exploration-rate analysis that quantifies what this loses.

CLI: `thalamus-research ablation --oracle-dir /oracle --query-file tasks.jsonl --out ablation.json`

**Remaining before R2 results are reportable:**
1. Run `thalamus-research ablation` on the 120-task suite to generate EvalRun JSONs.
2. Execute jiuwenswarm quality measurement pass to fill in `quality` fields.
3. Produce comparison table (ablation variants × task categories) and write R2 findings.

---

### Phase R3 — Novel Contributions: Cross-Path Learning and Bandit Formalization

**Research goal:** Propose genuinely new methods, not just novel combinations of
existing ones.

#### R3a — Cross-Path Knowledge Transfer

**Claim:** The logistic regression classifier (Path B) learns component co-inclusion
patterns that are invisible to individual-component scoring. These patterns can be
transferred back to improve the GA fitness function (Path A) — making Path A better
without requiring real data.

**Method:**
- After N turns of Path B training, extract the classifier weight matrix `W` (shape
  `n_components × d_embed`). High absolute weight covariance between components
  signals co-inclusion preference.
- Augment the GA fitness function: add a co-inclusion term that rewards component sets
  whose members have high co-occurrence probability under the classifier.
- Compare `path-a-original` vs `path-a-with-classifier-prior` at all maturity levels.

**Jiuwenswarm deployment output:**
- `oracle_builder evolve --use-classifier-prior` flag that reads `classifier_current.pkl`
  and augments the fitness function when it exists.
- Triggered automatically by `check-rebuild` when both classifier and updated component
  scores are available.

**Implementation status: ✓ COMPLETE (analysis tools)**

| Artifact | Class / Function | Location |
|---|---|---|
| Co-inclusion extraction | `CoInclusionExtractor` | `cross_path/co_inclusion_extractor.py` |
| Per-component pair analysis | `ComponentPair` | same |
| GA fitness augmentation | `FitnessAugmentor` | `cross_path/fitness_augmentor.py` |
| Convenience wrapper | `augment_fitness_config()` | same |
| CLI handler | `cmd_cross_path.run()` | `cross_path/cmd_cross_path.py` |

CLI:
```
# Inspect co-inclusion signal
thalamus-research cross-path --oracle-dir /oracle --top-pairs 20

# Produce augmented context_configs.json
thalamus-research cross-path --oracle-dir /oracle --augment-configs --lam 0.2 --out augmented.json
```

**Remaining:** Wire `--use-classifier-prior` flag into `oracle_builder evolve` to use
`FitnessAugmentor` during the GA build step (oracle_builder change, not research package).

#### R3b — Contextual Bandit Formalization

**Claim:** Context component selection is a multi-label contextual bandit problem.
Off-policy exploration under Path A's policy creates a biased dataset. We can quantify
this bias and bound the regret of the dual-path system under the exploration rate.

**Method:**
- Formalize: state = query embedding, action = component bitmask, reward = outcome.
- Path A policy: deterministic lookup (pure exploitation).
- Path B: supervised learning on logged (state, action, reward) tuples.
- Off-policy exploration: epsilon-greedy perturbation of Path A's action.
- Derive: minimum exploration rate `ε*` needed to ensure Path B converges to a policy
  not dominated by Path A.
- Empirically: measure Path B quality as a function of exploration rate on jiuwenswarm
  task suite.

**Jiuwenswarm deployment output:**
- `exploration_rate` auto-tuning: `thalamus-oracle tune --auto-exploration` that
  estimates the optimal ε from the divergence between Path A's action distribution and
  the full component space.

**Implementation status: ✓ COMPLETE (analysis tools)**

| Artifact | Class | Location |
|---|---|---|
| ε* derivation | `ExplorationRateEstimator` | `bandit/exploration_rate.py` |
| Per-component coverage | `ComponentCoverage`, `ExplorationRateResult` | same |
| Convergence measurement | `ConvergenceAnalyzer` | `bandit/convergence.py` |
| Convergence curve | `ConvergenceResult`, `ConvergencePoint` | same |
| CLI handler | `cmd_bandit.run()` | `bandit/cmd_bandit.py` |

**ε* formula** (derived in `exploration_rate.py`):

    ε* = max_i { max(0, (n_min/T_target - p_A(c_i)) / (0.5 - p_A(c_i))) }

where `p_A(c_i)` = fraction of clusters where Path A includes c_i.

CLI:
```
# Derive minimum exploration rate
thalamus-research bandit --oracle-dir /oracle --subcommand estimate-rate --n-min 10 --T-target 500

# Measure Path B convergence to Path A over turn history
thalamus-research bandit --oracle-dir /oracle --subcommand convergence --turn-log-dir /logs
```

**Remaining:** Wire estimated ε* into `TurnLogger`'s exploration rate parameter
(`oracle_builder tune --auto-exploration`).  Empirically validate the derivation
by sweeping ε ∈ {0, 0.05, 0.1, 0.2} on the jiuwenswarm task suite.

**Publication target:** R3 is the primary research contribution. Target: ICLR (main
track, LLM agents / RL for NLP), NeurIPS (Efficient Agents workshop), or ICML.

---

### Phase R4 — Set-Level Quality Modeling

**Research goal:** Replace the sum-of-independent-scores fitness function with a
model that predicts set-level quality directly.

This extends THALAMUS `future-roadmap.md` Gap A (interaction modeling) and puts it
on a research footing.

**Method:**
- Collect set-level outcome data: (component set, query embedding, outcome).
- Train a set-function approximator: start with a gradient-boosting model over pairwise
  component interaction features (Step 7 in IMPLEMENTATION_PLAN), then a joint
  classifier (Step 8), then a transformer over component embedding sets if data permits.
- Use the set-level model as the fitness function for the GA in Path A, replacing
  sum-of-individual-scores.
- Measure: does the GA with set-level fitness outperform the GA with marginal-score
  fitness on the jiuwenswarm task suite?

**Expected finding:** Set-level model outperforms independent scoring for libraries
with >50 components where component interactions are frequent (e.g., the same base
skill + multiple augmentation skills).

**Jiuwenswarm deployment output:**
- `oracle_builder evolve --fitness-model {marginal|xgb|joint}` flag.
- `xgb` mode available once sufficient logged data exists (threshold: ~1000 turns).
- Deployed as default for jiuwenswarm once R4 ablation validates improvement.

**Publication target:** Full-length paper at ACL / EMNLP (combining R1–R4).

---

### Phase R5 — Multi-Deployment Meta-Learning

**Research goal:** Reduce cold-start time for new jiuwenswarm deployments by
transferring knowledge from existing ones.

This extends `future-roadmap.md` Gap B (meta-learning across deployments) and
formalizes it as few-shot transfer learning.

**Method:**
- Define component identity across deployments: fingerprint = SHA-256(name + description
  + body_text). Matching fingerprints = same logical component.
- Maintain a cross-deployment knowledge base: for each fingerprint, store the
  mean_score, co-inclusion statistics, and classifier weight statistics aggregated
  across deployments.
- When a new deployment starts, warm-start its scoring matrices and classifier from
  the knowledge base entries that match its components.
- Measure: cold-start quality (turn 0) with vs without meta warm-start.

**Jiuwenswarm deployment output:**
- `thalamus-oracle meta-init --knowledge-base /shared/kb --oracle-dir /new-oracle`
  that populates a new deployment from a shared knowledge base.
- Knowledge base is a flat JSON store indexed by component fingerprint.

**Publication target:** Workshop paper at NeurIPS (Transfer Learning / Multi-task
Learning track). Prerequisite: at least 3 distinct jiuwenswarm deployments exist.

---

## 6. Dual-Mandate Summary

Every research phase delivers to both tracks:

| Phase | Research Deliverable | jiuwenswarm Deployment Deliverable |
|---|---|---|
| R1 | Baseline comparison + evaluation protocol | `BaselineSelector` adapter + selector config key |
| R2 | Ablation study + component attribution | Tuned default hyperparameters for all oracle + selector knobs |
| R3a | Cross-path knowledge transfer algorithm | `evolve --use-classifier-prior` flag |
| R3b | Bandit formalization + exploration rate bound | `--auto-exploration` tuning command |
| R4 | Set-level quality model + GA fitness improvement | `--fitness-model {marginal|xgb|joint}` flag |
| R5 | Meta-learning warm-start across deployments | `meta-init` command + shared knowledge base format |

---

## 7. What Changes in Thalamus to Support Research

Thalamus already has the full production system. The additions required for research:

1. **Baseline adapter interface** (R1): `BaselineSelector` protocol matching
   `ContextSelector.select(query) → dict | None`. Each baseline implements this
   interface. Swappable via config.

2. **Evaluation harness** (R1): `thalamus-eval` CLI command that runs the task suite,
   records quality + token + latency metrics, and writes a result JSON. Usable from
   jiuwenswarm CI.

3. **Fitness function extension** (R3a, R4): `FitnessFunction` protocol in
   `oracle_builder/evolutionary/`. Default implementation = current marginal sum.
   Alternate implementations = classifier-prior-augmented, XGB set-level model.

4. **Exploration rate analysis** (R3b): `tune --auto-exploration` command in
   `oracle_builder`.

5. **Knowledge base** (R5): `shared/knowledge_base.py` — flat fingerprint-indexed
   JSON with merge/init CLI operations.

None of these change existing behavior. All are additive extensions.

---

## 8. Publication Roadmap

| Phase complete | Target venue | Paper type |
|---|---|---|
| R1 | EMNLP 2026 System Findings | 4-page system description |
| R1 + R2 | ACL 2026 Workshop (LLM Agents) | 8-page paper |
| R1 + R2 + R3 | ICLR 2027 or NeurIPS 2026 | Full paper |
| R1–R4 | EMNLP 2027 | Journal-length full paper |

R3 (cross-path transfer + bandit formalization) is the primary novel contribution.
R1 and R2 are prerequisites that build the evaluation foundation.

---

## 9. Constraints and Non-Goals

**Constraints:**
- Every research phase must remain deployable in jiuwenswarm. No academic-only
  experiments that don't produce usable artifacts.
- All methods must preserve the <10ms Path A and <1ms Path B latency bounds.
- No changes to existing public APIs — all additions are new CLI flags or new adapters.

**Non-goals:**
- Fine-tuning the LLM itself (out of scope, not THALAMUS's problem).
- Learning component embeddings end-to-end (too data-hungry for the scale we operate at).
- Online learning at query time (the system is deliberately offline-first; online
  methods would break the latency guarantee).
