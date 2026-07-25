from __future__ import annotations

import json
import math
import sys
import argparse

from .config_builder import ContextConfigBuilder
from .config_builder_step01_load_components import ComponentsLoader
from .config_builder_step02_collect_texts import TextsCollector


def cmd_build(args: argparse.Namespace) -> None:
    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    output = args.output or (args.oracle_dir / "context_configs.json")
    budgets = {"small":  args.budget_small,
               "medium": args.budget_medium,
               "large":  args.budget_large}

    validation_config = None
    if getattr(args, "validate_pareto", False):
        from .pareto_validator import ValidationConfig
        validation_config = ValidationConfig(
            model=getattr(args, "eval_model", "gpt-4o-mini"),
            api_key=getattr(args, "eval_api_key", None),
            api_base=getattr(args, "eval_api_base", "https://api.openai.com/v1"),
            queries_per_cluster=getattr(args, "eval_queries_per_cluster", 3),
        )
        print(
            f"Pareto validation enabled: model={validation_config.model}, "
            f"queries_per_cluster={validation_config.queries_per_cluster}"
        )

    # ── --auto-k: select optimal cluster count from data ─────────────────────
    n_clusters = args.n_clusters
    if getattr(args, "auto_k", False):
        print("Auto-K: loading example texts to select optimal cluster count...")
        loader = ComponentsLoader(args.oracle_dir)
        collector = TextsCollector()
        components, example_texts_map = loader.load()
        all_texts = collector.collect(components, example_texts_map)

        if all_texts:
            from ..hyperparameters_tuner.cluster_count_tuner import ClusterCountTuner
            tuner = ClusterCountTuner(max_features=args.max_features)
            result = tuner.tune(all_texts)
            n_clusters = result.best_k
            print(
                f"Auto-K selected K={n_clusters} "
                f"(method={result.method}) from {len(all_texts)} example texts."
            )
        else:
            print(
                "Auto-K: no example texts found; falling back to --n-clusters "
                f"({n_clusters}).",
                file=sys.stderr,
            )

    # ── --use-cluster-lambda: load per-cluster λ from tune results ────────────
    per_cluster_lambda = None
    if getattr(args, "use_cluster_lambda", False):
        from ..hyperparameters_tuner.clusters_lambda_tuner import load_per_cluster_lambda
        per_cluster_lambda = load_per_cluster_lambda(args.oracle_dir)
        if per_cluster_lambda:
            print(
                f"Per-cluster λ loaded from per_cluster_lambda.json "
                f"({len(per_cluster_lambda)} cluster(s) with tuned values)."
            )
        else:
            print(
                "WARNING: --use-cluster-lambda set but per_cluster_lambda.json not found. "
                "Run: oracle_builder tune --oracle-dir <dir>  first.",
                file=sys.stderr,
            )

    # ── R4: --fitness-model xgb: load set-level quality fitness function ───────
    fitness_fn = None
    if getattr(args, "fitness_model", "marginal") == "xgb":
        model_dir = getattr(args, "fitness_model_dir", None) or (args.oracle_dir / "set_quality_model")
        try:
            from thalamus_research.set_quality.fitness_function import SetQualityFitness
            from thalamus_research.baselines.component_catalog import ComponentCatalog
        except ImportError:
            print(
                "WARNING: --fitness-model xgb requires thalamus-research. "
                "Install with: uv pip install -e ./thalamus_research",
                file=sys.stderr,
            )
        else:
            try:
                catalog = ComponentCatalog.load(args.oracle_dir)
                fitness_fn = SetQualityFitness.load(model_dir=model_dir, catalog=catalog)
                print(f"R4: XGB set-quality fitness loaded from {model_dir}")
            except FileNotFoundError as exc:
                print(
                    f"WARNING: {exc}  "
                    "(run: bash thalamus_research/runners/run_05_set_quality.sh first)",
                    file=sys.stderr,
                )

    # ── R5: auto-load transfer_priors.json if present (written by meta-init) ──
    transfer_priors: dict[str, float] | None = None
    prior_alpha = 0.5
    priors_path = args.oracle_dir / "transfer_priors.json"
    if priors_path.exists():
        try:
            transfer_priors = json.loads(priors_path.read_text(encoding="utf-8"))
            # Alpha decays as real turns accumulate: α = exp(-n_turns / 200)
            log_dir = getattr(args, "log_dir", None) or (args.oracle_dir / "online_logs")
            n_turns = 0
            if log_dir and log_dir.exists():
                for f in log_dir.glob("turns_*.jsonl"):
                    with open(f) as fh:
                        n_turns += sum(1 for _ in fh)
            prior_alpha = math.exp(-n_turns / 200.0)
            print(
                f"R5: transfer_priors.json found — {len(transfer_priors)} priors, "
                f"α={prior_alpha:.3f} (n_turns={n_turns})"
            )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: could not read transfer_priors.json: {exc}", file=sys.stderr)

    builder = ContextConfigBuilder(
        oracle_dir=args.oracle_dir,
        n_clusters=n_clusters,
        max_features=args.max_features,
        population_size=args.population,
        n_generations=args.generations,
        mutation_rate=args.mutation_rate,
        lambda_=args.lambda_,
        budgets=budgets,
        embedder=getattr(args, "embedder", "tfidf"),
        sentence_model=getattr(args, "sentence_model", "all-MiniLM-L6-v2"),
        validation_config=validation_config,
        per_cluster_lambda=per_cluster_lambda,
        fitness_fn=fitness_fn,
        transfer_priors=transfer_priors,
        prior_alpha=prior_alpha,
    )
    builder.build(output)

    # ── R3a: --use-classifier-prior: re-rank configs with co-inclusion signal ─
    if getattr(args, "use_classifier_prior", False):
        try:
            from thalamus_research.cross_path.fitness_augmentor import augment_fitness_config
        except ImportError:
            print(
                "WARNING: --use-classifier-prior requires thalamus-research. "
                "Install with: uv pip install -e ./thalamus_research",
                file=sys.stderr,
            )
        else:
            lam = getattr(args, "prior_lambda", 0.2)
            print(f"Applying classifier co-inclusion prior (λ={lam}) to re-rank configs...")
            try:
                augment_fitness_config(
                    oracle_dir=str(args.oracle_dir),
                    lam=lam,
                    out_path=str(output),
                )
                print(f"Classifier prior applied → {output}")
            except FileNotFoundError as exc:
                print(
                    f"WARNING: {exc}  "
                    "(run: thalamus-oracle train-classifier first)",
                    file=sys.stderr,
                )
