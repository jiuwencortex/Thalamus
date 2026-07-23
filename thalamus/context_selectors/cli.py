# context_selectors/cli.py
from __future__ import annotations

import logging
import sys

from .cli_args_parser import make_parser
from .by_clusters.cmd_lookup import cmd_lookup
from .by_classifier.cmd_classify import cmd_classify


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

    if args.command == "lookup":
        cmd_lookup(args)
    elif args.command == "classify":
        cmd_classify(args)
    elif args.command == "baseline-lookup":
        from ..baselines.cmd_baseline_lookup import cmd_baseline_lookup
        cmd_baseline_lookup(args)
    elif args.command == "eval":
        from ..evaluation.cmd_eval import cmd_eval
        cmd_eval(args)


if __name__ == "__main__":
    main()
