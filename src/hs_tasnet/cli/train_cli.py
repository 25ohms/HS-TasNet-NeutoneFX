from __future__ import annotations

from argparse import Namespace

from hs_tasnet.train.train import train
from hs_tasnet.utils.config import apply_overrides, load_config


def run(args: Namespace) -> None:
    cfg = load_config(args.cfg)
    cfg = apply_overrides(cfg, args.override or [])
    train(cfg, resume=args.resume)
