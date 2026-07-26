"""CLI handler: thalamus-research set-quality — Phase R4 set-level quality model.

Two subcommands:

  train       Train a SetQualityModel from turn logs
  evaluate    Evaluate a saved model on held-out turn logs

Usage::

    # Train on all turn logs in /oracle
    thalamus-research set-quality --oracle-dir /oracle --subcommand train

    # Train with specific turn log dir and save to custom path
    thalamus-research set-quality --oracle-dir /oracle --subcommand train \\
        --turn-log-dir /logs --model-dir /oracle/set_quality_model

    # Evaluate a saved model (prints RMSE, R²)
    thalamus-research set-quality --oracle-dir /oracle --subcommand evaluate \\
        --model-dir /oracle/set_quality_model --turn-log-dir /logs/held_out
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run(args) -> None:  # noqa: ANN001
    """Entry point for ``thalamus-research set-quality``."""
    oracle_dir = Path(args.oracle_dir)
    if not oracle_dir.exists():
        logger.error("oracle-dir not found: %s", oracle_dir)
        sys.exit(1)

    subcommand: str = getattr(args, "subcommand", "train")
    model_dir: Path = Path(
        getattr(args, "model_dir", None) or oracle_dir / "set_quality_model"
    )
    turn_log_dir: Path | None = (
        Path(args.turn_log_dir) if getattr(args, "turn_log_dir", None) else None
    )
    out_path: Path | None = Path(args.out) if getattr(args, "out", None) else None
    include_explored: bool = not getattr(args, "exclude_explored", False)

    if subcommand == "train":
        _run_train(oracle_dir, model_dir, turn_log_dir, include_explored, out_path)
    elif subcommand == "evaluate":
        _run_evaluate(oracle_dir, model_dir, turn_log_dir, include_explored, out_path)
    else:
        logger.error("Unknown set-quality subcommand: %s", subcommand)
        sys.exit(1)


# ── subcommand handlers ───────────────────────────────────────────────────────


def _run_train(
    oracle_dir: Path,
    model_dir: Path,
    turn_log_dir: Path | None,
    include_explored: bool,
    out_path: Path | None,
) -> None:
    try:
        from ..baselines.component_catalog import ComponentCatalog
        from .outcome_dataset import OutcomeDataset
        from .set_quality_model import SetQualityModel
    except ImportError as exc:
        logger.error("Import error: %s", exc)
        sys.exit(1)

    # Load dataset
    try:
        ds = OutcomeDataset.load(
            oracle_dir,
            turn_log_dir=turn_log_dir,
            include_explored=include_explored,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if len(ds) == 0:
        logger.error("No outcome records found. Cannot train.")
        sys.exit(1)
    if len(ds) < 2:
        logger.error(
            "Only %d outcome record(s) found — need at least 2 to train "
            "(recommend ≥ 20). Collect more agent turns with non-empty "
            "component_set and re-run.", len(ds)
        )
        sys.exit(1)

    # Load catalog (needed for feature extraction)
    try:
        catalog = ComponentCatalog.load(oracle_dir)
    except Exception as exc:
        logger.error("Failed to load ComponentCatalog: %s", exc)
        sys.exit(1)

    # Optionally load co-inclusion extractor
    extractor = _try_load_extractor(oracle_dir)

    X, y = ds.to_arrays(catalog, extractor)
    print(f"\nPhase R4 — Set-Level Quality Model Training")
    print(f"Oracle: {oracle_dir}")
    print(f"Training records: {len(ds)}  (include_explored={include_explored})")
    print(f"Feature dim: {X.shape[1]}")
    print()

    model = SetQualityModel()
    model.fit(X, y)

    model.save(model_dir)
    print(f"Model saved to: {model_dir}")
    print(f"In-sample RMSE: {model.meta.get('in_sample_rmse', 'n/a')}")
    print(f"In-sample R²:   {model.meta.get('in_sample_r2', 'n/a')}")

    if out_path is not None:
        result = {"oracle_dir": str(oracle_dir), "model_dir": str(model_dir), **model.meta}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Training report written to: {out_path}", file=sys.stderr)


def _run_evaluate(
    oracle_dir: Path,
    model_dir: Path,
    turn_log_dir: Path | None,
    include_explored: bool,
    out_path: Path | None,
) -> None:
    import numpy as np

    try:
        from ..baselines.component_catalog import ComponentCatalog
        from .outcome_dataset import OutcomeDataset
        from .set_quality_model import SetQualityModel
    except ImportError as exc:
        logger.error("Import error: %s", exc)
        sys.exit(1)

    try:
        model = SetQualityModel.load(model_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    try:
        ds = OutcomeDataset.load(
            oracle_dir,
            turn_log_dir=turn_log_dir,
            include_explored=include_explored,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    try:
        catalog = ComponentCatalog.load(oracle_dir)
    except Exception as exc:
        logger.error("Failed to load ComponentCatalog: %s", exc)
        sys.exit(1)

    extractor = _try_load_extractor(oracle_dir)
    X, y = ds.to_arrays(catalog, extractor)

    if len(X) == 0:
        logger.error("No records to evaluate on.")
        sys.exit(1)

    y_pred = model.predict(X)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"\nPhase R4 — Set-Level Quality Model Evaluation")
    print(f"Oracle: {oracle_dir}")
    print(f"Model:  {model_dir}")
    print(f"Records: {len(y)}")
    print()
    print(f"RMSE:   {rmse:.4f}")
    print(f"R²:     {r2:.4f}")
    print()

    result = {
        "oracle_dir": str(oracle_dir),
        "model_dir": str(model_dir),
        "n_records": len(y),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Evaluation report written to: {out_path}", file=sys.stderr)


# ── helpers ───────────────────────────────────────────────────────────────────


def _try_load_extractor(oracle_dir: Path):
    """Attempt to load CoInclusionExtractor; return None on failure."""
    try:
        from ..cross_path.co_inclusion_extractor import CoInclusionExtractor
        return CoInclusionExtractor.load(oracle_dir)
    except Exception:
        logger.debug("CoInclusionExtractor not available; co-inclusion features set to 0")
        return None
