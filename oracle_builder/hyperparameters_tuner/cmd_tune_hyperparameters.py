from __future__ import annotations

import sys


def _cmd_tune(args) -> None:
    """Run all hyperparameter tuners and save results to oracle_dir."""
    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    log_dir = getattr(args, "log_dir", None) or (args.oracle_dir / "online_logs")

    # ── Classifier C / threshold tuning ──────────────────────────────────────
    if not getattr(args, "skip_classifier_tune", False):
        print("=== Classifier hyperparameter search (C × threshold grid) ===")
        from .classifier_hyperparameter_search import HyperparameterSearch
        search = HyperparameterSearch(
            log_dir=log_dir,
            max_weeks=args.max_weeks,
            min_turns=args.min_turns,
        )
        train_turns, val_turns = search.split()
        total = len(train_turns) + len(val_turns)
        if total < args.min_turns:
            print(
                f"  Skipping: only {total} turns available (need {args.min_turns})."
            )
        else:
            print(f"  {total} turns: {len(train_turns)} train / {len(val_turns)} val")
            result = search.run(train_turns, val_turns)
            saved = search.save(result, args.oracle_dir)
            print(
                f"  Best C={result.best_C}  threshold={result.best_threshold}  "
                f"macro_F1={result.best_macro_f1:.4f}  → {saved.name}"
            )
    else:
        print("=== Classifier tuning skipped (--skip-classifier-tune) ===")

    # ── Cluster count K tuning ────────────────────────────────────────────────
    if not getattr(args, "skip_k_tune", False):
        print("=== Cluster count K tuning ===")
        from ..evolutionary.config_builder_step01_load_components import ComponentsLoader
        from ..evolutionary.config_builder_step02_collect_texts import TextsCollector
        from ..hyperparameters_tuner.cluster_count_tuner import ClusterCountTuner

        loader = ComponentsLoader(args.oracle_dir)
        collector = TextsCollector()
        components, example_texts_map = loader.load()
        all_texts = collector.collect(components, example_texts_map)

        if not all_texts:
            print("  Skipping: no example texts found in oracle_dir.")
        else:
            tuner = ClusterCountTuner()
            k_result = tuner.tune(all_texts)
            print(
                f"  Best K={k_result.best_k} (method={k_result.method}) "
                f"from {len(all_texts)} example texts."
            )
            if k_result.entries:
                print(f"  {'K':>4}  {'inertia':>12}  {'silhouette':>10}")
                for e in k_result.entries:
                    marker = "  <-- selected" if e.k == k_result.best_k else ""
                    print(f"  {e.k:>4}  {e.inertia:>12.1f}  {e.silhouette:>10.4f}{marker}")
            print(
                f"  Recommendation: use --n-clusters {k_result.best_k} on the next 'evolve' run."
            )
    else:
        print("=== Cluster count K tuning skipped (--skip-k-tune) ===")

    # ── Per-cluster λ tuning ──────────────────────────────────────────────────
    if not getattr(args, "skip_lambda_tune", False):
        print("=== Per-cluster λ tuning ===")
        from .clusters_lambda_tuner import LambdaTuner

        lambda_tuner = LambdaTuner(
            log_dir=log_dir,
            oracle_dir=args.oracle_dir,
            max_weeks=args.max_weeks,
        )
        lambda_result = lambda_tuner.tune(default_lambda=getattr(args, "default_lambda", 0.1))
        if not lambda_result.entries:
            print("  Skipping: context_configs.pkl not found or no logged turns.")
        else:
            saved = lambda_tuner.save(lambda_result, args.oracle_dir)
            n_tuned = sum(
                1 for e in lambda_result.entries
                if e.best_lambda != lambda_result.default_lambda
            )
            print(
                f"  {n_tuned}/{len(lambda_result.entries)} clusters have tuned λ values  → {saved.name}"
            )
            print(f"  {'cluster':>8}  {'n_turns':>8}  {'corr':>8}  {'λ':>6}")
            for e in lambda_result.entries:
                print(
                    f"  {e.cluster_id:>8}  {e.n_turns:>8}  "
                    f"{e.token_outcome_correlation:>8.3f}  {e.best_lambda:>6.3f}"
                )
    else:
        print("=== Per-cluster λ tuning skipped (--skip-lambda-tune) ===")
