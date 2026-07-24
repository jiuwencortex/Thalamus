"""CLI handler: thalamus-research cross-path — Phase R3a cross-path analysis.

Extracts co-inclusion signal from the Path B classifier weight matrix and
optionally produces an augmented context_configs.json with fitness_augmented
annotations for comparison against the original GA fitness.

Usage::

    # Inspect co-inclusion signal (top 20 joint pairs + top 20 redundant pairs)
    thalamus-research cross-path \\
        --oracle-dir /oracle \\
        --top-pairs 20

    # Write co-inclusion analysis to JSON
    thalamus-research cross-path \\
        --oracle-dir /oracle \\
        --out co_inclusion.json

    # Produce augmented context_configs.json (GA fitness + classifier signal)
    thalamus-research cross-path \\
        --oracle-dir /oracle \\
        --augment-configs \\
        --lam 0.2 \\
        --out context_configs_augmented.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run(args) -> None:  # noqa: ANN001
    """Entry point for ``thalamus-research cross-path``."""
    oracle_dir = Path(args.oracle_dir)
    if not oracle_dir.exists():
        logger.error("oracle-dir not found: %s", oracle_dir)
        sys.exit(1)

    top_k: int = getattr(args, "top_pairs", 20)
    out_path: Path | None = Path(args.out) if getattr(args, "out", None) else None
    augment_configs: bool = getattr(args, "augment_configs", False)
    lam: float = float(getattr(args, "lam", 0.2))

    try:
        from thalamus.research.cross_path.co_inclusion_extractor import CoInclusionExtractor
        extractor = CoInclusionExtractor.load(oracle_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if augment_configs:
        # Produce augmented context_configs.json
        from thalamus.research.cross_path.fitness_augmentor import FitnessAugmentor
        import json as _json

        config_path = oracle_dir / "context_configs.json"
        if not config_path.exists():
            logger.error("context_configs.json not found in %s", oracle_dir)
            sys.exit(1)

        config = _json.loads(config_path.read_text(encoding="utf-8"))
        augmentor = FitnessAugmentor(extractor, lam=lam)
        updated = augmentor.rerank_configs(config)

        output = json.dumps(updated, indent=2, ensure_ascii=False)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output, encoding="utf-8")
            print(f"Augmented config written to: {out_path}", file=sys.stderr)
        else:
            print(output)
        return

    # Default: co-inclusion analysis report
    report = extractor.to_dict(top_k=top_k)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Co-inclusion analysis written to: {out_path}", file=sys.stderr)
    else:
        # Print human-readable summary
        print(f"\nCo-inclusion analysis: {report['n_components']} components")
        print(f"Weight matrix shape: {report['weight_matrix_shape']}")
        print(f"\nTop {top_k} JOINT pairs (cosine ≥ 0.4 → frequently co-included):")
        print(f"{'Score':>8}  {'Component A':<30} {'Component B':<30}")
        print("-" * 70)
        for pair in report["top_joint_pairs"]:
            print(
                f"{pair['co_inclusion_score']:>8.4f}  "
                f"{pair['name_a']:<30} {pair['name_b']:<30}"
            )
        print(f"\nTop {top_k} REDUNDANT pairs (cosine ≤ -0.2 → substitutable):")
        print(f"{'Score':>8}  {'Component A':<30} {'Component B':<30}")
        print("-" * 70)
        for pair in report["top_redundant_pairs"]:
            print(
                f"{pair['co_inclusion_score']:>8.4f}  "
                f"{pair['name_a']:<30} {pair['name_b']:<30}"
            )
        print(
            "\nNote: Use --augment-configs to produce a context_configs_augmented.json "
            "with fitness_augmented annotations for GA comparison."
        )
