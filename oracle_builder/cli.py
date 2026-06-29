# oracle_builder/cli.py
# Entry point: python -m jiuwenswarm.tools.oracle_builder build
from __future__ import annotations

import logging
import sys

from oracle_builder.cli_args_parser import make_parser
from oracle_builder.rebuild_recommender.cmd_check_rebuild import _cmd_check_rebuild
from oracle_builder.cmd_list_versions import _cmd_list_versions
from oracle_builder.cmd_status import _cmd_status
from oracle_builder.cmd_tune import _cmd_tune
from oracle_builder.evolutionary.cmd_build import cmd_build
from oracle_builder.classifier.cmd_train_classifier import cmd_train_classifier


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(levelname)s %(message)s", stream=sys.stderr)

    if args.command == "evolve":
        cmd_build(args)
    elif args.command == "train-classifier":
        cmd_train_classifier(args)
    elif args.command == "list-versions":
        _cmd_list_versions(args)
    elif args.command == "tune":
        _cmd_tune(args)
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "check-rebuild":
        _cmd_check_rebuild(args)


if __name__ == "__main__":
    main()
