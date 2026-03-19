from __future__ import annotations

from typing import Dict, Iterable

import torch

from hs_tasnet.losses.waveform import l1_loss, mse_loss, si_snr, signal_distortion_ratio

VALID_METRICS = {"l1", "mse", "sdr", "si_snr"}
VALID_CHANNEL_POLICIES = {"strict", "mono_downmix", "first_channel"}


def align_waveform_target(
    pred: torch.Tensor, target: torch.Tensor, channel_policy: str = "strict"
) -> torch.Tensor:
    if channel_policy not in VALID_CHANNEL_POLICIES:
        raise ValueError(
            f"Unsupported target channel policy '{channel_policy}'. "
            f"Expected one of {sorted(VALID_CHANNEL_POLICIES)}."
        )

    if pred.dim() == 3 and target.dim() == 4:
        if channel_policy == "strict":
            if target.size(2) != 1:
                raise ValueError(
                    "Target has multiple channels while model output is mono per stem. "
                    "Set eval.target_channel_policy to 'mono_downmix' or 'first_channel'."
                )
            target = target.squeeze(2)
        elif channel_policy == "mono_downmix":
            target = target.mean(dim=2)
        else:
            target = target[:, :, 0, :]
    elif pred.dim() == 4 and target.dim() == 3:
        target = target.unsqueeze(2)

    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: pred={pred.shape}, target={target.shape}"
        )
    return target


def compute_waveform_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metrics: Iterable[str] = ("l1", "sdr"),
    target_channel_policy: str = "strict",
) -> Dict[str, torch.Tensor]:
    target = align_waveform_target(pred, target, channel_policy=target_channel_policy)
    values: Dict[str, torch.Tensor] = {}
    for metric in metrics:
        if metric not in VALID_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. Expected one of {sorted(VALID_METRICS)}."
            )
        if metric == "l1":
            values["l1"] = l1_loss(pred, target)
        elif metric == "mse":
            values["mse"] = mse_loss(pred, target)
        elif metric == "sdr":
            values["sdr"] = signal_distortion_ratio(pred, target)
        elif metric == "si_snr":
            values["si_snr"] = si_snr(pred, target)
    return values
