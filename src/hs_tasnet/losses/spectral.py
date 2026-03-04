import torch


def spectral_l1(pred_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred_spec - target_spec))
