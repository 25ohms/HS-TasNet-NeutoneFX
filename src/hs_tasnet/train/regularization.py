from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def _reshape_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() < 2:
        raise ValueError("Singular-value regularization requires a tensor with at least 2 dims.")
    return weight.reshape(weight.shape[0], -1)


def compute_singular_value_penalty(model: nn.Module, cfg: Dict) -> torch.Tensor:
    device = next(model.parameters()).device
    reg_cfg = cfg.get("regularization", {}).get("singular_value", {})
    if not reg_cfg.get("enabled", False):
        return torch.zeros((), device=device)

    weight = float(reg_cfg.get("weight", 0.0))
    if weight <= 0:
        return torch.zeros((), device=device)

    name_filters = tuple(reg_cfg.get("target_patterns", []))
    max_entries = int(reg_cfg.get("max_entries", 0))
    q = max(1, int(reg_cfg.get("rank", 1)))
    niter = max(0, int(reg_cfg.get("niter", 1)))

    penalties = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.dim() < 2 or not name.endswith("weight"):
            continue
        if name_filters and not any(pattern in name for pattern in name_filters):
            continue
        matrix = _reshape_weight(parameter)
        effective_q = min(max(q, 1), min(matrix.shape))
        _, singular_values, _ = torch.svd_lowrank(matrix, q=effective_q, niter=niter)
        penalties.append(singular_values.max().pow(2))
        if max_entries > 0 and len(penalties) >= max_entries:
            break

    if not penalties:
        return torch.zeros((), device=device)
    return weight * torch.stack(penalties).mean()
