from __future__ import annotations

import argparse
import sys

import numpy as np

from .classifier_selector import ClassifierSelector


def cmd_classify(args: argparse.Namespace) -> None:
    classifier_path = args.oracle_dir / "classifier.pkl"
    if not classifier_path.exists():
        print(f"ERROR: classifier.pkl not found in {args.oracle_dir}", file=sys.stderr)
        print("Run: python -m oracle_builder train-classifier --oracle-dir ...",
              file=sys.stderr)
        sys.exit(1)

    selector = ClassifierSelector.load(args.oracle_dir)

    if args.embedding:
        # Predict from a saved embedding file
        embedding = np.load(args.embedding)
        result = selector.select(embedding, threshold=args.threshold)

        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Skills:     {result['skills']}")
        print(f"Memory:     {result['memory']}")
        print(f"Tools:      {result['tools']}")
        if args.verbose:
            print("\nPer-component probabilities:")
            for name, prob in sorted(result["probabilities"].items(),
                                     key=lambda x: x[1], reverse=True):
                marker = "*" if prob >= args.threshold else " "
                print(f"  {marker} {prob:.4f}  {name}")
    else:
        # Show classifier model summary
        print(f"Components tracked: {selector.n_components}")
        print(f"Inclusion threshold: {args.threshold}")
        print()
        print("Component names:")
        for name in selector.component_names:
            print(f"  {name}")
