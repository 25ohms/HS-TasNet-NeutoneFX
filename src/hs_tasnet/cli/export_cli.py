from __future__ import annotations

from argparse import Namespace

from hs_tasnet.export.export_hs_tasnet import export_from_cfg


def run(args: Namespace) -> None:
    export_from_cfg(args.cfg, overrides=args.override or [])
