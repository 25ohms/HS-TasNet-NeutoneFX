from __future__ import annotations

from typing import Dict, Optional

import torch

VALID_TRAINING_LOSSES = {"l1", "mse", "si_snr"}


def get_optim_config(cfg: Dict) -> Dict:
    train_cfg = cfg.get("train", {})
    optim_cfg = dict(cfg.get("optim", {}))
    if "lr" not in optim_cfg and "lr" in train_cfg:
        optim_cfg["lr"] = train_cfg["lr"]
    if "clip_grad" not in optim_cfg and "grad_clip_norm" in train_cfg:
        optim_cfg["clip_grad"] = train_cfg["grad_clip_norm"]
    optim_cfg.setdefault("optim", "adam")
    optim_cfg.setdefault("lr", 1e-3)
    optim_cfg.setdefault("beta1", 0.9)
    optim_cfg.setdefault("beta2", 0.999)
    optim_cfg.setdefault("momentum", 0.9)
    optim_cfg.setdefault("weight_decay", 0.0)
    optim_cfg.setdefault("loss", "l1")
    optim_cfg.setdefault("clip_grad", 0.0)
    loss_name = str(optim_cfg["loss"]).lower()
    if loss_name not in VALID_TRAINING_LOSSES:
        raise ValueError(
            f"Unsupported optim.loss '{loss_name}'. Expected one of {sorted(VALID_TRAINING_LOSSES)}."
        )
    optim_cfg["loss"] = loss_name
    return optim_cfg


def build_optimizer(model: torch.nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    optim_cfg = get_optim_config(cfg)
    optim_name = str(optim_cfg.get("optim", "adam")).lower()
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    if optim_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(float(optim_cfg["beta1"]), float(optim_cfg["beta2"])),
            weight_decay=weight_decay,
        )
    if optim_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(float(optim_cfg["beta1"]), float(optim_cfg["beta2"])),
            weight_decay=weight_decay,
        )
    if optim_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(optim_cfg["momentum"]),
            weight_decay=weight_decay,
            nesterov=bool(optim_cfg.get("nesterov", False)),
        )
    raise ValueError(f"Unsupported optimizer '{optim_name}'. Expected adam, adamw, or sgd.")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Dict
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_cfg = cfg.get("train", {}).get("scheduler")
    if not sched_cfg:
        return None
    sched_name = str(sched_cfg.get("name", "step")).lower()
    if sched_name == "step":
        step_size = sched_cfg.get("step_size", 1)
        gamma = sched_cfg.get("gamma", 0.98)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if sched_name == "cosine":
        t_max = int(sched_cfg.get("t_max", cfg.get("train", {}).get("epochs", 1)))
        eta_min = float(sched_cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    raise ValueError(f"Unsupported scheduler '{sched_name}'. Expected step or cosine.")
