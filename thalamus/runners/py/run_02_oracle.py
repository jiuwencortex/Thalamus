#!/usr/bin/env python3
"""
Phase 3: Build the evolutionary oracle (context_configs.json).

Python equivalent of run_02_oracle.sh — import-and-call, no subprocess boundary.

Config via environment variables:
  ORACLE_DIR      Directory with scoring_matrix_*.json files  (required)
  EMBEDDER        tfidf | sentence                            (default: tfidf)
  N_CLUSTERS      K for K-means (overridden when AUTO_K=true) (default: 20)
  AUTO_K          true | false — elbow+silhouette auto-select (default: true)
  POPULATION      GA population size                          (default: 100)
  GENERATIONS     GA generation count                         (default: 200)
  MUTATION_RATE   Bit-flip probability                        (default: 0.05)
  LAMBDA          Token-penalty weight in fitness             (default: 0.1)
  BUDGET_SMALL    Token cap for small budget                  (default: 2000)
  BUDGET_MEDIUM   Token cap for medium budget                 (default: 4000)
  BUDGET_LARGE    Token cap for large budget                  (default: 8000)
"""
import os
import sys


def main() -> None:
    oracle_dir    = os.environ.get("ORACLE_DIR",     "~/.jiuwenswarm/agent/workspace/oracle")
    embedder      = os.environ.get("EMBEDDER",       "tfidf")
    n_clusters    = os.environ.get("N_CLUSTERS",     "20")
    auto_k        = os.environ.get("AUTO_K",         "true").lower() == "true"
    population    = os.environ.get("POPULATION",     "100")
    generations   = os.environ.get("GENERATIONS",    "200")
    mutation_rate = os.environ.get("MUTATION_RATE",  "0.05")
    lambda_       = os.environ.get("LAMBDA",         "0.1")
    budget_small  = os.environ.get("BUDGET_SMALL",   "2000")
    budget_medium = os.environ.get("BUDGET_MEDIUM",  "4000")
    budget_large  = os.environ.get("BUDGET_LARGE",   "8000")

    print("=== Phase 3: Oracle Building ===")
    print(f"  Oracle dir : {oracle_dir}")
    print(f"  Embedder   : {embedder}")
    print(f"  Auto-K     : {auto_k}")
    print(f"  Population : {population}  Generations: {generations}")
    print()

    from thalamus.oracle.cli import main as oracle_main  # noqa: PLC0415

    sys.argv = [
        "thalamus-oracle", "evolve",
        "--oracle-dir",    oracle_dir,
        "--embedder",      embedder,
        "--n-clusters",    n_clusters,
        "--population",    population,
        "--generations",   generations,
        "--mutation-rate", mutation_rate,
        "--lambda",        lambda_,
        "--budget-small",  budget_small,
        "--budget-medium", budget_medium,
        "--budget-large",  budget_large,
    ]
    if auto_k:
        sys.argv.append("--auto-k")

    oracle_main()

    print()
    print(f"Done. Oracle written to: {oracle_dir}/context_configs.json")


if __name__ == "__main__":
    main()
