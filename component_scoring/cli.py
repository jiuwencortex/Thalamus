# component_scoring/cli.py
# Unified entry point for building component scoring matrices.
# Supports: python -m component_scoring build --type skills|memory|tools
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig

from component_scoring.cli_args_parser import make_parser
from component_scoring.cli_builders import build_skills, build_memory, build_tools


def _make_model(args: argparse.Namespace) -> tuple[Model, str]:
    """Construct the LLM Model from CLI args."""
    client_cfg = ModelClientConfig(
        client_provider=args.provider,
        api_key=args.api_key,
        api_base=args.api_base,
        timeout=args.timeout,
        verify_ssl=False,
    )
    request_cfg = ModelRequestConfig(model=args.model)
    return Model(model_client_config=client_cfg, model_config=request_cfg), args.model


async def _main() -> None:
    args = make_parser()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    needs_llm = args.build_type != "enrich" and not args.dry_run

    if needs_llm and not args.api_key:
        print("ERROR: --api-key is required when not using --dry-run", file=sys.stderr)
        sys.exit(1)

    if needs_llm and not args.model:
        print("ERROR: --model is required unless --type enrich or --dry-run", file=sys.stderr)
        sys.exit(1)

    # For dry-run or enrich-only we never call the LLM — skip model construction
    model, model_name = (None, "") if not needs_llm else _make_model(args)

    if args.build_type in ("skills", "all"):
        if not args.dry_run:
            print("=== Building skills matrix ===")
        await build_skills(args, model, model_name)

    if args.build_type in ("memory", "all"):
        if not args.dry_run:
            print("=== Building memory matrix ===")
        await build_memory(args, model, model_name)

    if args.build_type in ("tools", "all"):
        if not args.dry_run:
            print("=== Building tools matrix ===")
        await build_tools(args, model, model_name)

    if args.build_type in ("enrich", "all"):
        print("=== Enriching matrices with interaction data ===")
        from .enrichment.score_enricher import ScoreEnricher
        log_dir = args.log_dir or (args.matrix_dir / "online_logs")
        enricher = ScoreEnricher(
            oracle_dir=args.matrix_dir,
            log_dir=log_dir,
            n_needed=args.n_needed,
            max_weeks=args.max_weeks,
        )
        summary = enricher.update()
        if not summary:
            print("No real interaction data found; matrix files unchanged.")
        else:
            print(f"Updated {len(summary)} component(s) with real data:")
            for name, n in sorted(summary.items()):
                print(f"  {name}: {n} real sample(s)")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
