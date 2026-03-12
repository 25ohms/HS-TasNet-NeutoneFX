from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from hs_tasnet.losses.metrics import compute_waveform_metrics


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_cfg: Dict | None = None,
) -> Dict[str, float]:
    resolved_eval_cfg = eval_cfg or {}
    metric_names = resolved_eval_cfg.get("metrics", ["l1", "sdr"])
    target_channel_policy = resolved_eval_cfg.get("target_channel_policy", "strict")

    model.eval()
    totals: Dict[str, float] = {name: 0.0 for name in metric_names}
    count = 0
    with torch.no_grad():
        for mixture, stems in loader:
            mixture = mixture.to(device)
            stems = stems.to(device)
            pred = model(mixture)
            values = compute_waveform_metrics(
                pred,
                stems,
                metrics=metric_names,
                target_channel_policy=target_channel_policy,
            )
            for metric_name, value in values.items():
                totals[metric_name] += float(value.item())
            count += 1
    model.train()
    return {name: total / max(count, 1) for name, total in totals.items()}
