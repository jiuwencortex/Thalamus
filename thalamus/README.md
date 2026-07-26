# thalamus

Adaptive context selection for production AI agents.

THALAMUS solves **Context Saturation**: the failure mode where an agent's context window fills with irrelevant skill instructions, memory sections, and tool definitions because the system loads everything, every time. It precomputes the optimal component set for each query type and token budget, then retrieves it at runtime in under 10 ms with no LLM calls.

See [`docs/thalamus.md`](docs/thalamus.md) for the full system design and rationale.

---

## How It Works

Two paths, computed offline and looked up at runtime:

```
PATH A — EVOLUTIONARY (cold start, < ~500 turns)

  Offline:
    1. Component Scoring   — LLM generates (query, answer) pairs per component;
                             evaluates each component against those pairs.
                             Output: scoring_matrix_*.json per component.

    2. Score Enrichment    — Bayesian blend of synthetic scores with real turn
                             data as it accumulates (optional, improves over time).

    3. Query Clustering    — K-means over component example queries (TF-IDF or
                             sentence-transformer). Groups query space into types.

    4. Evolutionary Search — Genetic algorithm (no LLM calls) finds optimal
                             component bitmask per cluster × token budget.
                             Pareto front balances quality vs. token cost.

    5. Write Output        — context_configs.json  (lookup table)
                             context_configs.pkl   (fitted clusterer)

  Runtime:  vectorize query → predict cluster → JSON lookup  (<10 ms)


PATH B — CLASSIFIER (after sufficient turn logs accumulate)

  Offline:
    6. Turn Logging        — agent logs (query_embedding, context_config, outcome)
                             per turn to weekly JSONL files.

    7. Classifier Training — logistic regression per component trained on logged
                             turns; versioned, evaluated, promotion-gated.

  Runtime:  vectorize query → per-component logistic regression  (<1 ms)
```

Path A runs first (cold start). Path B takes over once enough operational data exists. The system degrades gracefully: if the oracle is missing, returns an empty selection; if the classifier is missing, falls back to Path A.

---

## Prerequisites

- Python 3.10+
- `openjiuwen` parent package — provides the LLM client (`openjiuwen.core.foundation.llm`)
- An LLM API key (OpenAI-compatible: OpenAI, Kimi, etc.)

---

## Setup

```bash
git clone <repo>
cd Thalamus
uv sync                    # installs both packages + registers CLI scripts
```

For editable installs (development):

```bash
uv pip install -e ./thalamus
```

---

## Running

```bash
# Full pipeline: score components → build oracle → validate
OPENAI_API_KEY=sk-...                                          \
SKILLS_DIR=/path/to/skills                                     \
PROJECT_DIR=/path/to/project                                   \
AGENT_CORE_DIR=/path/to/openjiuwen/agent-core                  \
JIUWENSWARM_DIR=/path/to/openjiuwen/jiuwenswarm                \
ORACLE_DIR=/path/to/oracle                                     \
bash thalamus/runners/sh/run_all.sh
```

**Tool discovery.** `run_01_score.sh` discovers tool directories automatically in this order:

1. `AGENT_CORE_DIR` → `<dir>/openjiuwen/harness/tools` (built-in tools: filesystem, shell, web, browser, multimodal, …)
2. `JIUWENSWARM_DIR` → `<dir>/jiuwenswarm/agents/harness/code/tools` (project-specific tools)
3. `TOOLS_DIR` → any extra custom directory
4. importlib probe — if openjiuwen is pip-installed, location is detected automatically
5. `pip show openjiuwen` fallback

Set `TOOLS_DIR=none` to skip tool scoring entirely. Without any of these, tools are skipped with an explanatory message.

| Script | What it runs |
|--------|--------------|
| `runners/sh/run_01_score.sh` | Phase 1–2: LLM scores all components |
| `runners/sh/run_02_oracle.sh` | Phase 3: genetic algorithm builds `context_configs.json` |
| `runners/sh/run_03_classifier.sh` | Phase 4: trains logistic regression from turn logs |
| `runners/sh/run_04_select.sh` | Runtime: resolves a single query to a context config |
| `runners/sh/run_05_r4_activate.sh` | R4: train XGB set-quality model + rebuild oracle (needs ~500 turns) |
| `runners/sh/run_06_r5_meta_init.sh` | R5: extract KB from mature oracle + warm-start a new deployment |

All scripts read from environment variables. See each script's header for the full list.

**Python runners (debug mode).** Each shell runner has a Python equivalent in `runners/py/` that imports and calls the CLI `main()` directly — no subprocess boundary, so breakpoints work anywhere in the call stack. Same env-var interface:

| Script | What it runs |
|--------|--------------|
| `runners/py/run_01_score.py` | Phase 1–2: component scoring |
| `runners/py/run_02_oracle.py` | Phase 3: evolutionary oracle |
| `runners/py/run_03_classifier.py` | Phase 4: classifier training |
| `runners/py/run_04_select.py` | Runtime: single query lookup |
| `runners/py/run_05_r4_activate.py` | R4: XGB quality model + oracle rebuild |
| `runners/py/run_06_r5_meta_init.py` | R5: KB extract + warm-start |
| `runners/py/run_all.py` | Full pipeline (calls sub-runners directly) |

```bash
ORACLE_DIR=/path/to/oracle python thalamus/runners/py/run_04_select.py
```

CLI equivalents:

```bash
# Core pipeline
thalamus-score  build --type all  --skills-dir ... --matrix-dir ... --model gpt-4o-mini --api-key ...
thalamus-oracle evolve            --oracle-dir ...
thalamus-select lookup            --oracle-dir ... --query "..." --budget auto

# Hyperparameter tuning (K, λ, classifier C/threshold)
thalamus-oracle tune --oracle-dir ...

# R3a — apply classifier co-inclusion prior after the GA (requires trained classifier)
thalamus-oracle evolve --oracle-dir ... --use-classifier-prior --prior-lambda 0.2

# R3b — derive minimum off-policy exploration rate ε* and write exploration_rate.json
thalamus-oracle tune   --oracle-dir ... --auto-exploration --n-min 10 --T-target 500

# R4 — rebuild oracle with XGB set-quality fitness (requires trained model in set_quality_model/)
thalamus-oracle evolve --oracle-dir ... --fitness-model xgb

# R5 — warm-start a new oracle from the cross-deployment knowledge base
thalamus-oracle meta-init --oracle-dir /new/oracle --kb-path ~/.jiuwenswarm/knowledge_base.json
```

---

## Python API

```python
# Select context for a query at runtime (Path A — cluster lookup)
from thalamus.selection import ClusterSelector

selector = ClusterSelector.load("/path/to/oracle")
config = selector.select("Set up a CI pipeline", budget="auto", ordering="bookend")
# config = {"skills": [...], "memory": [...], "tools": [...], "context_tokens": 2140, ...}

# Log an agent turn for classifier training (Path B)
from thalamus._shared import TurnLogger

logger = TurnLogger("/path/to/oracle/online_logs")
turn_id = logger.log_turn(
    query_embedding=embedding_vector,
    context_config=config,
    exploration_rate=0.05,
    all_component_names={"skills": [...], "memory": [...], "tools": [...]}
)
logger.update_outcome(turn_id, task_completed=True, follow_up_correction=False,
                      conversation_length=3, skills_used=["devops-toolkit"])
```

---

## Package Structure

```
thalamus/                 ← sub-project root
  pyproject.toml
  README.md
  docs/
    thalamus.md             # Full system design
    IMPLEMENTATION_PLAN.md  # Development status (Steps 1–8)
    future-roadmap.md       # Post-Step-8 directions
    SkillRouter.md          # Comparison with SkillRouter
    jiuwenswarm-integration-plan.md
  runners/
    sh/                     # Shell runners (production / CI)
      run_all.sh
      run_01_score.sh       # Phase 1–2: component scoring
      run_02_oracle.sh      # Phase 3: evolutionary oracle
      run_03_classifier.sh  # Phase 4: classifier training
      run_04_select.sh      # Runtime: single query lookup
      run_05_r4_activate.sh # R4: XGB quality model + oracle rebuild
      run_06_r5_meta_init.sh # R5: KB extract + warm-start
    py/                     # Python runners (debug mode — no subprocess boundary)
      run_all.py
      run_01_score.py
      run_02_oracle.py
      run_03_classifier.py
      run_04_select.py
      run_05_r4_activate.py
      run_06_r5_meta_init.py
  tests/
    test_skill_discovery.py
    run_tests.py
  src/
    thalamus/             ← importable Python package
      scoring/              # Phase 1–2: scan, evaluate, score components
        skills/             #   SKILL.md scanner + composer
        memory/             #   Markdown section scanner + composer
        tools/              #   Python AST tool scanner + composer
        enrichment/         #   Bayesian blend of real turn data into scores
        shared/             #   Generic pipeline: generator, evaluator, metrics
      oracle/               # Phase 3–4: build lookup table + train classifier
        evolutionary/       #   K-means clustering + genetic algorithm oracle
        classifier/         #   Logistic regression, versioning, tuning
        hyperparameters_tuner/ # Auto-tune K, λ, C, thresholds
        rebuild_recommender/   # Drift detection + staleness checker
      selection/            # Runtime: select context for a query
        by_clusters/        #   Path A: cluster lookup (ClusterSelector)
        by_classifier/      #   Path B: per-component probabilities
      _shared/              # Cross-package utilities
        turn_logger.py
        outcome_scorer.py
        query_clusterer.py
        context_orderer.py
```

---

## Tests

```bash
# From the workspace root:
python -m pytest thalamus/tests/ -v

# Or from inside thalamus/:
python -m pytest tests/ -v
```

---

## Oracle Directory Layout

After a full run:

```
<oracle-dir>/
  scoring_matrix_skill_<name>.json
  scoring_matrix_mem_<name>.json
  scoring_matrix_tool_<name>.json
  context_configs.json           # Path A lookup table
  context_configs.pkl            # fitted clusterer
  classifier_current.pkl         # Path B classifier (if trained)
  classifier_registry.json
  per_cluster_lambda.json        # tuned λ per cluster (written by: tune)
  exploration_rate.json          # derived ε* per component (written by: tune --auto-exploration)
  online_logs/
    turns_YYYY-WNN.jsonl
```
