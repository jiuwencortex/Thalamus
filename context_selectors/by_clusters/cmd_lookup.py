from __future__ import annotations

import argparse
import json

from .cluster_selector import ClusterSelector


def cmd_lookup(args: argparse.Namespace) -> None:
    selector = ClusterSelector.load(args.oracle_dir)
    ordering = getattr(args, "ordering", "relevance")

    if args.query:
        print(f"Query: {args.query!r}")
        budget = getattr(args, "budget", "medium")

        if budget == "auto":
            config = selector.select_auto(args.query, ordering=ordering)
            chosen_budget = config.pop("budget", "?")
            print(f"Auto-selected budget: {chosen_budget}")
            print(f"\n[budget_{chosen_budget}] tokens={config.get('context_tokens', '?')} fitness={config.get('fitness', '?')}")
            print(f"  skills: {config.get('skills', [])}")
            print(f"  memory: {config.get('memory', [])}")
            print(f"  tools:  {config.get('tools', [])}")
        else:
            configs = selector.select_all_budgets(args.query, ordering=ordering)
            for budget_name, config in configs.items():
                print(f"\n[budget_{budget_name}] tokens={config.get('context_tokens', '?')} fitness={config.get('fitness', '?')}")
                print(f"  skills: {config.get('skills', [])}")
                print(f"  memory: {config.get('memory', [])}")
                print(f"  tools:  {config.get('tools', [])}")
    else:
        # Show cluster summary
        config_path = args.oracle_dir / "context_configs.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))

        print(f"Clusters: {data['n_clusters']}  Components: {data['n_components']}")
        print(f"Vocabulary size: {data.get('vocabulary_size', '?')}")
        print(f"Built at: {data.get('built_at', '?')}")
        print()

        for cluster in data["clusters"]:
            cid = cluster["cluster_id"]
            n = cluster["n_queries"]
            examples = cluster.get("example_queries", [])[:2]
            ex_str = " | ".join(f'"{q[:40]}"' for q in examples)
            print(f"  Cluster {cid:>2}: {n:>4} queries  {ex_str}")
