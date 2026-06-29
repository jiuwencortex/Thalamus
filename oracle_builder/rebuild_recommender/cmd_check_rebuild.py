from __future__ import annotations


def _cmd_check_rebuild(args) -> None:
    """Check cached drift/staleness status and print rebuild recommendations."""
    import sys
    from .rebuild_recommendation_checker import RebuildRecommendationChecker

    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    log_dir = getattr(args, "log_dir", None) or (args.oracle_dir / "online_logs")

    scheduler = RebuildRecommendationChecker(
        oracle_dir=args.oracle_dir,
        log_dir=log_dir,
        new_turns_threshold=getattr(args, "new_turns_threshold", 200),
        drift_rebuild_min_turns=getattr(args, "drift_rebuild_min_turns", 50),
    )
    rec = scheduler.check()

    print("=== Rebuild Recommendation ===")

    if rec.retrain_classifier:
        print("\n  [ACTION NEEDED] Retrain the classifier:")
        for reason in rec.retrain_reasons:
            print(f"    • {reason}")
        print(
            "\n  Run: python -m jiuwenswarm.tools.oracle_builder train-classifier "
            f"--oracle-dir {args.oracle_dir}"
        )
    else:
        print("\n  Classifier is up-to-date — no retraining needed.")

    if rec.rebuild_oracle:
        print("\n  [ACTION NEEDED] Rebuild the oracle:")
        for reason in rec.rebuild_reasons:
            print(f"    • {reason}")
        print(
            "\n  Run: python -m jiuwenswarm.tools.oracle_builder evolve "
            f"--oracle-dir {args.oracle_dir}"
        )
    else:
        print("\n  Oracle is up-to-date — no rebuild needed.")

    # Exit code 2 signals that a rebuild is recommended (useful in CI/scripts)
    if rec.retrain_classifier or rec.rebuild_oracle:
        sys.exit(2)
