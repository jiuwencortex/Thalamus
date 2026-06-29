from __future__ import annotations

import sys

from .classifier.model_registry import ModelRegistry


def _cmd_list_versions(args) -> None:
    """Print a table of all registered classifier versions."""
    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    registry = ModelRegistry(args.oracle_dir)
    versions = registry.list_versions()

    if not versions:
        print("No classifier versions registered yet.")
        print(f"Run: python -m thalamus.oracle_builder train-classifier --oracle-dir {args.oracle_dir}")
        return

    header = f"{'#':<4}  {'filename':<40}  {'trained_at':<22}  {'train':>6}  {'val':>5}  {'F1':>7}  {'AUC':>7}  active"
    print(header)
    print("-" * len(header))

    for idx, entry in enumerate(versions, start=1):
        active_marker = "  *" if entry.is_current else ""
        print(
            f"{idx:<4}  {entry.filename:<40}  {entry.trained_at:<22}  "
            f"{entry.n_train_turns:>6}  {entry.n_val_turns:>5}  "
            f"{entry.macro_f1:>7.4f}  {entry.macro_auc:>7.4f}{active_marker}"
        )
