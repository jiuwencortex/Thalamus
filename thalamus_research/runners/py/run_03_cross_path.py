#!/usr/bin/env python3
"""
R3a — Cross-path knowledge transfer (co-inclusion analysis).

Python equivalent of run_03_cross_path.sh — import-and-call, no subprocess boundary.

Extracts co-inclusion signal from the Path B classifier's weight matrix W:
  co_inclusion(c_i, c_j) = cosine(W[i], W[j])

High co-inclusion → the classifier learned these components are jointly useful.
This signal can augment the GA fitness function:
  fitness_aug = base_fitness + λ × mean_pairwise_co_inclusion(S)

Two modes (controlled by AUGMENT env var):
  0 (default)  Report only — print top joint and redundant component pairs.
  1            Report + write augmented context_configs.json.

Config via environment variables (see _config.py):
  ORACLE_DIR           Oracle directory (needs classifier_current.pkl)  (required)
  RESULTS_DIR          Output directory                                  (default: ORACLE_DIR/research_results)
  AUGMENT              0 | 1 — also produce augmented configs           (default: 0)
  LAM / CO_INCLUSION_LAM   λ weight for fitness augmentation            (default: 0.2)
  TOP_PAIRS            Number of pairs to print in report                (default: 20)

Output:
  $RESULTS_DIR/r3a_cross_path.json
  $RESULTS_DIR/context_configs_augmented.json   (if AUGMENT=1)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _config import cfg, section  # noqa: E402


def main() -> None:
    augment = os.environ.get("AUGMENT", "0") == "1"
    lam     = os.environ.get("LAM", cfg.co_inclusion_lam)

    section("R3a — Cross-Path Co-Inclusion Analysis")
    print(f"Oracle:     {cfg.oracle_dir}")
    print(f"λ (lambda): {lam}")
    print(f"Top pairs:  {cfg.top_pairs}")
    print(f"Augment:    {augment}")
    print()

    from thalamus_research.cli import main as research_main  # noqa: PLC0415

    # Always run the co-inclusion report
    sys.argv = [
        "thalamus-research", "cross-path",
        "--oracle-dir", cfg.oracle_dir,
        "--top-pairs",  cfg.top_pairs,
        "--lam",        lam,
        "--out",        os.path.join(cfg.results_dir, "r3a_cross_path.json"),
    ]
    research_main()

    print()
    print(f"Done. Report: {cfg.results_dir}/r3a_cross_path.json")

    # Optionally produce augmented context_configs.json
    if augment:
        aug_out = os.path.join(cfg.results_dir, "context_configs_augmented.json")
        print()
        print(f"── Augmenting context_configs.json (λ={lam}) ──")
        sys.argv = [
            "thalamus-research", "cross-path",
            "--oracle-dir",      cfg.oracle_dir,
            "--augment-configs",
            "--lam",             lam,
            "--out",             aug_out,
        ]
        research_main()
        print(f"Augmented configs: {aug_out}")
        print()
        print("To use augmented fitness in the GA:")
        print(f"  cp {aug_out} {cfg.oracle_dir}/context_configs.json")
        print("  (then re-run thalamus-oracle evolve)")


if __name__ == "__main__":
    main()
