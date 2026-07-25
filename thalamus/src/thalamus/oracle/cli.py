# oracle_builder/cli.py
# Entry point: python -m thalamus.oracle build
from __future__ import annotations

import logging
import sys

from .cli_args_parser import make_parser
from .rebuild_recommender.cmd_check_rebuild import _cmd_check_rebuild
from .cmd_list_versions import _cmd_list_versions
from .cmd_meta_init import cmd_meta_init
from .cmd_status import _cmd_status
from .cmd_tune import _cmd_tune
from .evolutionary.cmd_build import cmd_build
from .classifier.cmd_train_classifier import cmd_train_classifier


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
    elif args.command == "meta-init":
        cmd_meta_init(args)


if __name__ == "__main__":
    main()
