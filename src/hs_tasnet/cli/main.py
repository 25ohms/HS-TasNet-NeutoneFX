from __future__ import annotations

import argparse

from hs_tasnet.cli.eval_cli import run as eval_run
from hs_tasnet.cli.export_cli import run as export_run
from hs_tasnet.cli.infer_cli import run as infer_run
from hs_tasnet.cli.train_cli import run as train_run


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values, e.g. train.batch_size=8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="hs-tasnet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    _add_common_args(train_parser)
    train_parser.add_argument("--resume", default=None, help="Checkpoint path to resume")
    train_parser.set_defaults(func=train_run)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate")
    eval_parser.set_defaults(func=eval_run)

    infer_parser = subparsers.add_parser("infer", help="Run offline inference")
    _add_common_args(infer_parser)
    infer_parser.set_defaults(func=infer_run)

    export_parser = subparsers.add_parser("export", help="Export a Neutone model")
    _add_common_args(export_parser)
    export_parser.set_defaults(func=export_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
