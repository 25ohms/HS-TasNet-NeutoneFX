from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional

import torch

from hs_tasnet.utils.seed import get_rng_state, set_rng_state


def save_checkpoint(
    path: str | pathlib.Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    cfg: Dict[str, Any],
) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "rng": get_rng_state(),
        "cfg": cfg,
    }
    torch.save(payload, path)


def load_checkpoint_payload(
    path: str | pathlib.Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def prune_checkpoints(
    run_dir: str | pathlib.Path,
    keep_last: int | None,
) -> None:
    if keep_last is None:
        return
    keep_last = int(keep_last)
    if keep_last <= 0:
        raise ValueError("keep_last must be >= 1 when provided")

    run_dir = pathlib.Path(run_dir)
    checkpoints = sorted(run_dir.glob("checkpoint_epoch*.pt"))
    stale_checkpoints = checkpoints[:-keep_last]
    for checkpoint in stale_checkpoints:
        checkpoint.unlink()


def load_checkpoint(
    path: str | pathlib.Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> Dict[str, Any]:
    ckpt = load_checkpoint_payload(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and ckpt.get("rng") is not None:
        set_rng_state(ckpt["rng"])
    return ckpt
