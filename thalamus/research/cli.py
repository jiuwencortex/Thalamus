"""Entry point for the thalamus-research CLI.

Production commands (lookup, classify) stay in thalamus-select.
Research commands (baseline-lookup, eval, and future R2-R5 commands)
live here so the production CLI has no research dependencies.
"""
from __future__ import annotations

import logging
import sys

from .cli_args_parser import make_parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.command == "baseline-lookup":
        from .baselines.cmd_baseline_lookup import cmd_baseline_lookup
        cmd_baseline_lookup(args)
    elif args.command == "eval":
        from .evaluation.cmd_eval import cmd_eval
        cmd_eval(args)
    elif args.command == "ablation":
        from .ablations.cmd_ablation import run
        run(args)
    elif args.command == "cross-path":
        from .cross_path.cmd_cross_path import run
        run(args)
    elif args.command == "bandit":
        from .bandit.cmd_bandit import run
        run(args)


if __name__ == "__main__":
    main()
