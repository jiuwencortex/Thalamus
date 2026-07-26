# thalamus_research

Research tooling for the THALAMUS adaptive context selection system.

Provides retrieval baselines, an evaluation harness, and five experimental
phases (R1–R5) that study and improve THALAMUS design choices. Never imported
by production code — depends on `thalamus` but not vice versa.

See [`docs/research-plan.md`](docs/research-plan.md) for the full research programme.

---

## Research Phases

| Phase | Name | Status | What it does |
|-------|------|--------|--------------|
| R1 | Baselines + Evaluation | ✓ | TF-IDF, BM25, dense, random vs THALAMUS |
| R2 | Ablation Study | ✓ | Isolate each design choice (C1/C3/C5/C6) |
| R3a | Cross-Path Transfer | ✓ | Co-inclusion signal from classifier → GA fitness |
| R3b | Contextual Bandit | ✓ | Optimal exploration rate ε* derivation |
| R4 | Set-Level Quality Model | ✓ | GBR fitness replacing linear sum-of-scores |
| R5 | Meta-Learning | ✓ | Cross-deployment warm-start via fingerprint KB |

---

## Prerequisites

- `thalamus` installed (or on `PYTHONPATH`)
- A built oracle directory (run `thalamus-oracle evolve` first)

---

## Setup

```bash
cd Thalamus
uv sync                    # installs both packages + registers thalamus-research CLI
```

For editable installs (development):

```bash
uv pip install -e ./thalamus
uv pip install -e ./thalamus_research
```

---

## Running Experiments

```bash
# Set oracle directory (required for all runners)
export ORACLE_DIR=/path/to/oracle

# R1: baseline comparison
bash thalamus_research/runners/sh/run_01_baselines.sh

# R2: ablation study
bash thalamus_research/runners/sh/run_02_ablation.sh

# R3a: co-inclusion analysis (requires trained classifier)
bash thalamus_research/runners/sh/run_03_cross_path.sh

# R3b: exploration rate estimation
bash thalamus_research/runners/sh/run_04_bandit.sh

# R4: set-level quality model (requires turn logs with quality labels)
bash thalamus_research/runners/sh/run_05_set_quality.sh

# R5: cross-deployment meta-learning (requires multiple deployments)
KB_PATH=/shared/kb.json bash thalamus_research/runners/sh/run_06_meta_learning.sh

# All pre-data phases in sequence (R1 → R3b)
bash thalamus_research/runners/sh/run_all_experiments.sh
```

All runners read from `runners/sh/_config.sh` defaults and accept env var overrides:

```bash
ORACLE_DIR=/my/oracle QUERIES_FILE=/my/queries.json bash runners/sh/run_01_baselines.sh
```

**Python runners (debug mode).** Each shell runner has a Python equivalent in `runners/py/` that calls CLI `main()` directly — full call stack visible in the debugger. Shared config lives in `runners/py/_config.py` (same env-var names as `_config.sh`):

```bash
ORACLE_DIR=/path/to/oracle python thalamus_research/runners/py/run_01_baselines.py

# MODE flag works the same as in the shell runner:
MODE=convergence ORACLE_DIR=... python thalamus_research/runners/py/run_04_bandit.py
MODE=both        ORACLE_DIR=... python thalamus_research/runners/py/run_05_set_quality.py
```

---

## CLI Reference

```bash
thalamus-research <subcommand> [options]

Subcommands:
  baseline-lookup   Run a single query through retrieval baselines (R1)
  eval              Benchmark selectors on a query set, write JSON (R1)
  ablation          Ablation study: TopK / NoBookend / SingleBudget / PathBOnly (R2)
  cross-path        Co-inclusion analysis + GA fitness augmentation (R3a)
  bandit            Derive ε* and measure classifier convergence (R3b)
  set-quality       Train / evaluate set-level quality model (R4)
  meta-learning     Extract KB from oracle or warm-start new deployment (R5)
```

---

## Package Structure

```
thalamus_research/        ← sub-project root
  pyproject.toml
  README.md
  docs/
    research-plan.md      # Phase-by-phase research programme with status
    thalamus_paper.md     # arXiv-style paper draft
    thalamus_slides.md    # Slide deck
  runners/
    sh/                       # Shell runners (production / CI)
      _config.sh              #   Shared env-var defaults (sourced by all runners)
      run_all_experiments.sh  #   R1–R3b orchestrator
      run_01_baselines.sh     #   R1: baseline evaluation
      run_02_ablation.sh      #   R2: ablation study
      run_03_cross_path.sh    #   R3a: co-inclusion + fitness augmentation
      run_04_bandit.sh        #   R3b: ε* estimation + convergence
      run_05_set_quality.sh   #   R4: train + evaluate quality model
      run_06_meta_learning.sh #   R5: KB extract + transfer
    py/                       # Python runners (debug mode — no subprocess boundary)
      _config.py              #   Shared Config dataclass (same env-var names)
      run_all_experiments.py  #   R1–R3b orchestrator
      run_01_baselines.py
      run_02_ablation.py
      run_03_cross_path.py
      run_04_bandit.py        #   MODE=estimate|convergence|both
      run_05_set_quality.py   #   MODE=train|evaluate|both
      run_06_meta_learning.py #   MODE=extract|transfer|both
  tests/                  # Research-specific tests (placeholder)
  src/
    thalamus_research/    ← importable Python package
      baselines/          # R1: TFIDFSelector, BM25Selector, DenseSelector, etc.
      evaluation/         # R1: BenchmarkRunner, EvalRun, overlap statistics
      ablations/          # R2: TopKSelector, NoBookendSelector, ...
      cross_path/         # R3a: CoInclusionExtractor, FitnessAugmentor
      bandit/             # R3b: ExplorationRateEstimator, ConvergenceAnalyzer
      set_quality/        # R4: OutcomeDataset, SetQualityModel, SetQualityFitness
      meta_learning/      # R5: fingerprint_catalog, KnowledgeBase, TransferInitializer
      cli.py              # thalamus-research entry point
      cli_args_parser.py  # argument parser for all 7 subcommands
```
