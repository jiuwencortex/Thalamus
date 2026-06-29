from __future__ import annotations


def _cmd_status(args) -> None:
    """Run fresh drift + staleness checks and print a health summary."""
    import sys
    from .staleness_checker import StalenessChecker
    from .classifier.model_registry import ModelRegistry
    from ..shared.distribution_monitor import DistributionMonitor

    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    log_dir = getattr(args, "log_dir", None) or (args.oracle_dir / "online_logs")

    # ── Drift check ───────────────────────────────────────────────────────────
    print("=== Distribution Drift ===")
    monitor = DistributionMonitor(
        log_dir=log_dir,
        oracle_dir=args.oracle_dir,
        recent_weeks=getattr(args, "recent_weeks", 1),
        baseline_weeks=getattr(args, "baseline_weeks", 4),
        js_threshold=getattr(args, "js_threshold", 0.15),
    )
    drift = monitor.check()
    monitor.save(drift, args.oracle_dir)
    drift_marker = "  [DRIFT DETECTED]" if drift.drift_detected else ""
    print(
        f"  JS divergence : {drift.js_divergence:.4f}  "
        f"(threshold={drift.threshold}){drift_marker}"
    )
    print(f"  Recent turns  : {drift.n_recent}  |  Baseline turns: {drift.n_baseline}")
    print(f"  {drift.message}")

    # ── Staleness check ───────────────────────────────────────────────────────
    print("\n=== Oracle Staleness ===")
    checker = StalenessChecker(args.oracle_dir)
    staleness = checker.check()
    checker.save(staleness, args.oracle_dir)
    stale_marker = "  [STALE]" if staleness.stale else ""
    print(f"  Oracle exists    : {staleness.oracle_exists}{stale_marker}")
    print(f"  Oracle mtime     : {staleness.oracle_mtime}")
    print(f"  Components in oracle   : {staleness.n_oracle_components}")
    print(f"  Scoring matrices found : {staleness.n_current_matrices}")
    if staleness.added_components:
        print(f"  Added (not in oracle)  : {', '.join(staleness.added_components)}")
    if staleness.removed_components:
        print(f"  Removed (in oracle)    : {', '.join(staleness.removed_components)}")
    if staleness.updated_components:
        print(f"  Updated (newer mtime)  : {', '.join(staleness.updated_components)}")
    print(f"  {staleness.message}")

    # ── Classifier registry ───────────────────────────────────────────────────
    print("\n=== Classifier Registry ===")
    registry = ModelRegistry(args.oracle_dir)
    versions = registry.list_versions()
    current = registry.get_current()
    if not versions:
        print("  No classifier versions registered yet.")
    else:
        print(f"  Registered versions: {len(versions)}")
        if current:
            print(
                f"  Current: {current.filename}  "
                f"F1={current.macro_f1:.4f}  AUC={current.macro_auc:.4f}  "
                f"trained={current.trained_at}"
            )
