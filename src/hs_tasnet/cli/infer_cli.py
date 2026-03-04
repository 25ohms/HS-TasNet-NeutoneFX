from __future__ import annotations

from argparse import Namespace

from hs_tasnet.infer.infer import infer
from hs_tasnet.utils.config import apply_overrides, load_config


def run(args: Namespace) -> None:
    cfg = load_config(args.cfg)
    cfg = apply_overrides(cfg, args.override or [])
    infer(cfg)
