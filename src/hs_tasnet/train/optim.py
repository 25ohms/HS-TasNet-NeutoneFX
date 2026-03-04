from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def build_optimizer(model: torch.nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=cfg.get("train", {}).get("lr", 1e-3))


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Dict
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_cfg = cfg.get("train", {}).get("scheduler")
    if not sched_cfg:
        return None
    step_size = sched_cfg.get("step_size", 1)
    gamma = sched_cfg.get("gamma", 0.98)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
